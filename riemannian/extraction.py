from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from riemannian.geodesics import log_map_shooting
from riemannian.metric import Metric, norm, path_length
from riemannian.preprocessing import smooth_and_differentiate
from riemannian.segmentation import segment_riemannian

DEFAULT_RIEMPRIM_PARAMS = {
    "window_length": 31,
    "polyorder": 2,
    "delta_theta": float(np.pi / 3),
    "angle_confirm_samples": 5,
    "min_segment_samples": 10,
    "speed_floor_fraction": 0.05,
    "min_tiny_segment_duration": 0.15,

    "path_samples": 80,
    "log_map_max_nfev": 60,
    "endpoint_tol": 2e-3,
    "geodesic_max_step": 0.03,
}

def is_tiny_segment(segment: tuple[int, int], t: np.ndarray, min_duration: float) -> bool:
    start, end = segment
    return bool(float(t[end] - t[start]) < min_duration)


def merge_tiny_segments(segments: list[tuple[int, int]], t: np.ndarray,
                        min_duration: float) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str]]:
    merged = list(segments)
    status = {segment: "normal" for segment in merged}
    while True:
        tiny_index = next((i for i, segment in enumerate(merged)
                           if is_tiny_segment(segment, t, min_duration) and len(merged) > 1), None)
        if tiny_index is None:
            return merged, {segment: status.get(segment, "normal") for segment in merged}

        segment = merged[tiny_index]
        if tiny_index > 0:
            new_segment = (merged[tiny_index - 1][0], segment[1])
            del merged[tiny_index]
            merged[tiny_index - 1] = new_segment
        else:
            new_segment = (segment[0], merged[tiny_index + 1][1])
            del merged[tiny_index + 1]
            merged[tiny_index] = new_segment
        status[new_segment] = "merged_tiny"


def fit_arc_cubic(T: float, length: float, speed0: float, speedf: float) -> np.ndarray:
    if T <= 0.0 or length <= 1e-12:
        return np.zeros(4, dtype=float)
    A = np.array([[T**2, T**3], [2.0 * T, 3.0 * T**2]], dtype=float)
    b = np.array([length - speed0 * T, speedf - speed0], dtype=float)
    a2, a3 = np.linalg.solve(A, b)
    return np.array([0.0, speed0, a2, a3], dtype=float)


def smooth_boundary_speeds(primitives: list[dict]) -> list[dict]:
    for A, B in zip(primitives[:-1], primitives[1:]):
        speed = 0.5 * (A["speedf"] + B["speed0"])
        A["speedf"] = speed
        B["speed0"] = speed

    for P in primitives:
        P["arc_phase_coeffs"] = fit_arc_cubic(P["duration"], P["length"], P["speed0"], P["speedf"])
    return primitives


def extract_primitives(q: np.ndarray, t: np.ndarray, metric: Metric, *, window_length: int = 31, polyorder: int = 2,
                       delta_theta: float = np.pi / 3, angle_confirm_samples: int = 5,
                       min_segment_samples: int = 10, speed_floor_fraction: float = 0.05,
                       min_tiny_segment_duration: float = 0.15, path_samples: int = 80,
                       log_map_max_nfev: int = 60, endpoint_tol: float = 2e-3,
                       geodesic_max_step: float = 0.03, show_progress: bool = True) -> list[dict]:
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    if q.ndim != 2:
        raise ValueError("q must have shape (samples, dimensions).")
    if len(t) != len(q):
        raise ValueError("t and q must have the same number of samples.")

    q_smooth, qdot = smooth_and_differentiate(q, t, window_length=window_length, polyorder=polyorder)
    segments = segment_riemannian(
        q_smooth, qdot, metric, delta_theta=delta_theta,
        min_segment_samples=min_segment_samples, speed_floor_fraction=speed_floor_fraction,
        angle_confirm_samples=angle_confirm_samples)
    segments, segment_status = merge_tiny_segments(segments, t, min_tiny_segment_duration)

    primitives: list[dict] = []
    segment_bar = tqdm(segments, desc="Riemannian primitives", unit="primitive", leave=False,
                       disable=not show_progress)
    for primitive_index, (start, end) in enumerate(segment_bar):
        segment_bar.set_postfix(primitive=primitive_index, end=end, samples=end - start + 1)
        q0 = q_smooth[start]
        q1 = q_smooth[end]
        v0, q_path, v_path, info = log_map_shooting(
            q0, q1, metric, samples=path_samples, max_nfev=log_map_max_nfev,
            endpoint_tol=endpoint_tol, max_step=geodesic_max_step)
        primitives.append({
            "primitive_index": primitive_index,
            "start": int(start),
            "end": int(end),
            "t0": float(t[start]),
            "t1": float(t[end]),
            "duration": float(t[end] - t[start]),
            "q0": q0,
            "q1": q1,
            "v0": v0,
            "path": q_path,
            "velocity_path": v_path,
            "length": path_length(q_path, metric),
            "speed0": norm(q_smooth[start], qdot[start], metric),
            "speedf": norm(q_smooth[end], qdot[end], metric),
            "endpoint_error": float(info["endpoint_error"]),
            "tiny_segment_status": segment_status.get((start, end), "normal"),
            "shooting_info": info,
        })
    return smooth_boundary_speeds(primitives)
