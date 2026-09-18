from __future__ import annotations

import numpy as np

from riemannian.metric import Metric, angle, norm
from riemannian.transport import parallel_transport_along_polyline


def metric_speeds(q: np.ndarray, qdot: np.ndarray, G: Metric) -> np.ndarray:
    return np.array([norm(q[i], qdot[i], G) for i in range(len(q))], dtype=float)


def first_meaningful_velocity_index(q: np.ndarray, qdot: np.ndarray, G: Metric, frac: float = 0.05) -> int:
    speeds = metric_speeds(q, qdot, G)
    max_speed = float(np.max(speeds)) if len(speeds) else 0.0
    if max_speed <= 0.0:
        raise ValueError("Trajectory has no nonzero Riemannian velocity.")
    hits = np.flatnonzero(speeds > frac * max_speed)
    if len(hits) == 0:
        raise ValueError("No meaningful velocity sample found.")
    return int(hits[0])


def segment_riemannian(q: np.ndarray, qdot: np.ndarray, G: Metric, delta_theta: float = np.pi / 3,
                       min_segment_samples: int = 10, speed_floor_fraction: float = 0.05,
                       angle_confirm_samples: int = 5) -> list[tuple[int, int]]:
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    if q.shape != qdot.shape:
        raise ValueError("q and qdot must have the same shape.")
    if len(q) < 2:
        raise ValueError("At least two samples are required for segmentation.")

    speeds = metric_speeds(q, qdot, G)
    speed_floor = float(speed_floor_fraction) * float(np.max(speeds))
    start = first_meaningful_velocity_index(q, qdot, G)
    segments: list[tuple[int, int]] = []
    v_ref = qdot[start].copy()
    q_prev = q[start].copy()
    exceed_count = 0

    for i in range(start + 1, len(q)):
        if speeds[i] < speed_floor:
            q_prev = q[i].copy()
            exceed_count = 0
            continue

        v_ref = parallel_transport_along_polyline(v_ref, np.vstack([q_prev, q[i]]), G)
        theta = angle(q[i], v_ref, qdot[i], G)
        exceed_count = exceed_count + 1 if theta > delta_theta else 0
        if exceed_count >= angle_confirm_samples and (i - start) >= min_segment_samples:
            segments.append((start, i - 1))
            start = i
            v_ref = qdot[start].copy()
            exceed_count = 0
        q_prev = q[i].copy()

    if start < len(q) - 1:
        segments.append((start, len(q) - 1))
    if not segments:
        raise ValueError("Riemannian segmentation produced no segments.")
    return segments
