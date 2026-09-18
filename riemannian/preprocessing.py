from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def estimate_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float).reshape(-1)
    if len(t) < 2:
        raise ValueError("At least two time samples are required.")
    dt = np.diff(t)
    if np.any(dt <= 0.0):
        raise ValueError("Time samples must be strictly increasing.")
    return float(np.median(dt))


def smooth_and_differentiate(q: np.ndarray, t: np.ndarray, window_length: int = 31, polyorder: int = 2) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    if q.ndim != 2:
        raise ValueError("q must have shape (samples, dimensions).")

    n = len(q)
    window_length = int(window_length)
    polyorder = int(polyorder)
    if window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window_length must be odd.")
    if window_length <= polyorder:
        raise ValueError("Savitzky-Golay window_length must be greater than polyorder.")
    if window_length > n:
        raise ValueError("Savitzky-Golay window_length cannot exceed the number of samples.")

    dt = estimate_dt(t)
    q_smooth = savgol_filter(q, window_length, polyorder, axis=0)
    qdot = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt, axis=0)
    return q_smooth, qdot
