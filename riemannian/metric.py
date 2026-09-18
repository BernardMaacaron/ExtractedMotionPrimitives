from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import rbdl

Metric = Callable[[np.ndarray], np.ndarray]


def symmetrize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def regularize_spd(A: np.ndarray, eig_floor: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(symmetrize(A))
    vals = np.maximum(vals, eig_floor)
    return symmetrize((vecs * vals) @ vecs.T)


def solve_metric_system(Gq: np.ndarray, b: np.ndarray, eig_floor: float = 1e-10) -> np.ndarray:
    return np.linalg.solve(regularize_spd(Gq, eig_floor=eig_floor), b)


def create_human_arm_model_9dof(height: float = 1.75, weight: float = 75.0) -> Any:
    model = rbdl.Model()
    model.gravity = np.array([0.0, 0.0, -9.81])

    m_upper_arm = 0.028 * weight
    m_forearm = 0.016 * weight
    m_hand = 0.006 * weight

    l_upper_arm = 0.186 * height
    l_forearm = 0.146 * height
    l_hand = 0.108 * height

    def add_spherical_joint(parent_id: int, name: str, mass: float, length: float,
                            com_pos: np.ndarray, inertia_diag: list[float]) -> int:
        body_x = rbdl.Body.fromMassComInertia(1e-6, np.zeros(3), np.eye(3) * 1e-6)
        joint_x = rbdl.Joint.fromJointType("JointTypeRevoluteX")
        id_x = model.AddBody(parent_id, rbdl.SpatialTransform(), joint_x, body_x, f"{name}_X")

        body_y = rbdl.Body.fromMassComInertia(1e-6, np.zeros(3), np.eye(3) * 1e-6)
        joint_y = rbdl.Joint.fromJointType("JointTypeRevoluteY")
        id_y = model.AddBody(id_x, rbdl.SpatialTransform(), joint_y, body_y, f"{name}_Y")

        body_z = rbdl.Body.fromMassComInertia(mass, com_pos, np.diag(inertia_diag))
        joint_z = rbdl.Joint.fromJointType("JointTypeRevoluteZ")
        trans = rbdl.SpatialTransform()
        trans.r = np.array([0.0, 0.0, length])
        return model.AddBody(id_y, trans, joint_z, body_z, f"{name}_Z")

    i_ua = (1.0 / 12.0) * m_upper_arm * (l_upper_arm**2)
    id_rshoulder = add_spherical_joint(
        0, "RShoulder", m_upper_arm, 0.0, np.array([0.0, 0.0, l_upper_arm / 2.0]), [i_ua, i_ua, 0.01])

    i_fa = (1.0 / 12.0) * m_forearm * (l_forearm**2)
    id_relbow = add_spherical_joint(
        id_rshoulder, "RElbow", m_forearm, l_upper_arm, np.array([0.0, 0.0, l_forearm / 2.0]), [i_fa, i_fa, 0.01])

    i_h = (1.0 / 12.0) * m_hand * (l_hand**2)
    add_spherical_joint(
        id_relbow, "RWrist", m_hand, l_forearm, np.array([0.0, 0.0, l_hand / 2.0]), [i_h, i_h, 0.01])

    return model


def mass_matrix(model: Any, q: np.ndarray) -> np.ndarray:
    q = np.ascontiguousarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q must be one-dimensional.")

    dim = int(model.dof_count)
    if len(q) != dim:
        raise ValueError(f"q has length {len(q)}, but the RBDL model has {dim} DoF.")

    M = np.zeros((dim, dim), order="C")
    rbdl.CompositeRigidBodyAlgorithm(model, q, M)
    return symmetrize(M)


def finite_difference_metric_derivative(metric: Metric, q: np.ndarray, step: float = 1e-5) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q must be one-dimensional.")
    if step <= 0.0:
        raise ValueError("step must be positive.")

    dim = len(q)
    dM = np.zeros((dim, dim, dim), dtype=float)
    for k in range(dim):
        dq = np.zeros(dim, dtype=float)
        dq[k] = step
        dM[k] = symmetrize((metric(q + dq) - metric(q - dq)) / (2.0 * step))
    return dM


@dataclass
class MassMetricModel:
    model: Any | None = None
    finite_difference_step: float = 1e-5
    eig_floor: float = 1e-8
    name: str = "mass_metric"

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = create_human_arm_model_9dof()
        self.dim = int(self.model.dof_count)
        self.is_constant = False

    def __call__(self, q: np.ndarray) -> np.ndarray:
        return regularize_spd(mass_matrix(self.model, q), eig_floor=self.eig_floor)

    def grad(self, q: np.ndarray) -> np.ndarray:
        return finite_difference_metric_derivative(self, q, self.finite_difference_step)


def inner(q: np.ndarray, u: np.ndarray, v: np.ndarray, G: Metric) -> float:
    return float(np.asarray(u, dtype=float).T @ G(q) @ np.asarray(v, dtype=float))


def norm(q: np.ndarray, u: np.ndarray, G: Metric) -> float:
    return float(np.sqrt(max(inner(q, u, u, G), 0.0)))


def angle(q: np.ndarray, u: np.ndarray, v: np.ndarray, G: Metric) -> float:
    denom = norm(q, u, G) * norm(q, v, G)
    if denom <= 0.0:
        raise ValueError("Cannot compute a Riemannian angle with a zero-norm vector.")
    cosine = np.clip(inner(q, u, v, G) / denom, -1.0, 1.0)
    return float(np.arccos(cosine))


def metric_gradient(G: Metric, q: np.ndarray) -> np.ndarray:
    if not hasattr(G, "grad"):
        raise TypeError("Metric object must provide grad(q).")
    return G.grad(q)


def christoffel_first_kind(G: Metric, q: np.ndarray) -> np.ndarray:
    dG = metric_gradient(G, q)
    return 0.5 * (dG.transpose(1, 2, 0) + dG.transpose(1, 0, 2) - dG)


def geodesic_acceleration(q: np.ndarray, v: np.ndarray, G: Metric) -> np.ndarray:
    Gamma = christoffel_first_kind(G, q)
    coriolis_like = np.einsum("ijk,j,k->i", Gamma, v, v)
    return -solve_metric_system(G(q), coriolis_like)


def path_length(q_path: np.ndarray, G: Metric) -> float:
    q_path = np.asarray(q_path, dtype=float)
    length = 0.0
    for i in range(len(q_path) - 1):
        dq = q_path[i + 1] - q_path[i]
        q_mid = 0.5 * (q_path[i] + q_path[i + 1])
        length += float(np.sqrt(max(dq.T @ G(q_mid) @ dq, 0.0)))
    return length
