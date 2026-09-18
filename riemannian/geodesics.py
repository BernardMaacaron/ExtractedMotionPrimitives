from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from riemannian.metric import Metric, geodesic_acceleration


def straight_line_path(q0: np.ndarray, q1: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    s = np.linspace(0.0, 1.0, int(samples))[:, None]
    v = q1 - q0
    return q0[None, :] + s * v[None, :], np.repeat(v[None, :], len(s), axis=0)


def geodesic_rhs(_s: float, y: np.ndarray, G: Metric, dim: int) -> np.ndarray:
    q = y[:dim]
    v = y[dim:]
    return np.concatenate([v, geodesic_acceleration(q, v, G)])


def integrate_geodesic(q0: np.ndarray, v0: np.ndarray, G: Metric, samples: int = 80,
                       rtol: float = 1e-5, atol: float = 1e-7, max_step: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if q0.shape != v0.shape:
        raise ValueError("q0 and v0 must have the same shape.")

    dim = len(q0)
    s_eval = np.linspace(0.0, 1.0, int(samples))
    y0 = np.concatenate([q0, v0])
    sol = solve_ivp(lambda s, y: geodesic_rhs(s, y, G, dim), (0.0, 1.0), y0,
                    t_eval=s_eval, rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise RuntimeError(f"Geodesic integration failed: {sol.message}")
    y = sol.y.T
    if not np.all(np.isfinite(y)):
        raise RuntimeError("Geodesic integration produced non-finite values.")
    return y[:, :dim], y[:, dim:]


def log_map_shooting(q0: np.ndarray, q1: np.ndarray, G: Metric, samples: int = 80, max_nfev: int = 60,
                     endpoint_tol: float = 2e-3, max_step: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    initial = q1 - q0

    def endpoint_residual(v0: np.ndarray) -> np.ndarray:
        q_path, _v_path = integrate_geodesic(q0, v0, G, samples=2, max_step=max_step)
        return q_path[-1] - q1

    result = least_squares(endpoint_residual, initial, max_nfev=max_nfev, xtol=1e-5, ftol=1e-5, gtol=1e-5)
    v0 = result.x
    q_path, v_path = integrate_geodesic(q0, v0, G, samples=samples, max_step=max_step)
    endpoint_error = float(np.linalg.norm(q_path[-1] - q1))
    if endpoint_error > endpoint_tol:
        raise RuntimeError(f"Geodesic shooting endpoint error {endpoint_error:.6g} exceeds {endpoint_tol:.6g}.")
    info = {
        "method": "least_squares_shooting",
        "success": bool(result.success),
        "endpoint_error": endpoint_error,
        "nfev": int(result.nfev),
        "cost": float(result.cost),
    }
    return v0, q_path, v_path, info
