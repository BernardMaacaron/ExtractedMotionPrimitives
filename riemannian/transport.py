from __future__ import annotations

import numpy as np

from riemannian.metric import Metric, christoffel_first_kind, norm, solve_metric_system


def parallel_transport_increment(q_mid: np.ndarray, dq: np.ndarray, v: np.ndarray, G: Metric) -> np.ndarray:
    Gamma = christoffel_first_kind(G, q_mid)
    connection_term = np.einsum("ijk,j,k->i", Gamma, dq, v)
    return -solve_metric_system(G(q_mid), connection_term)


def parallel_transport_along_polyline(v_start: np.ndarray, q_path: np.ndarray, G: Metric,
                                      preserve_norm: bool = True) -> np.ndarray:
    q_path = np.asarray(q_path, dtype=float)
    v = np.asarray(v_start, dtype=float).copy()
    if len(q_path) < 2:
        return v

    source_norm = norm(q_path[0], v, G)
    for i in range(len(q_path) - 1):
        q0 = q_path[i]
        q1 = q_path[i + 1]
        dq = q1 - q0
        if np.linalg.norm(dq) == 0.0:
            continue
        q_mid = 0.5 * (q0 + q1)
        v = v + parallel_transport_increment(q_mid, dq, v, G)
        if preserve_norm:
            current_norm = norm(q1, v, G)
            if current_norm == 0.0:
                raise ValueError("Parallel transport produced a zero-norm vector.")
            v *= source_norm / current_norm
    return v
