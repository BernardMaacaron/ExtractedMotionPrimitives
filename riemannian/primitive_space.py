from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from riemannian.geodesics import integrate_geodesic, log_map_shooting
from riemannian.metric import Metric, norm
from riemannian.transport import parallel_transport_along_polyline


def load_extracted_primitives(root: str | Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    root = Path(root)
    metadata: list[dict[str, Any]] = []
    paths = []
    velocity_paths = []

    for summary_path in sorted(root.glob("sub-*/*/rep-*/primitives_summary.csv")):
        status = json.loads((summary_path.parent / "status.json").read_text())
        if status["status"] != "completed":
            raise RuntimeError(f"Primitive extraction is not complete: {summary_path.parent}")

        with np.load(summary_path.parent / "primitive_paths.npz") as arrays:
            with summary_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    primitive_index = int(row["primitive_index"])
                    key = f"primitive{primitive_index:03d}"
                    path = np.asarray(arrays[f"{key}_path"], dtype=float)
                    velocity_path = np.asarray(arrays[f"{key}_velocity_path"], dtype=float)
                    if path.shape != velocity_path.shape:
                        raise ValueError(f"Path/velocity shape mismatch for {summary_path.parent}/{key}")

                    metadata.append({
                        "subject": row["subject"],
                        "task": row["task"],
                        "repetition": int(row["repetition"]),
                        "primitive_index": primitive_index,
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                        "t0": float(row["t0"]),
                        "t1": float(row["t1"]),
                        "duration": float(row["duration"]),
                        "length": float(row["length"]),
                        "speed0": float(row["speed0"]),
                        "speedf": float(row["speedf"]),
                        "endpoint_error": float(row["endpoint_error"]),
                        "tiny_segment_status": row["tiny_segment_status"],
                    })
                    paths.append(path)
                    velocity_paths.append(velocity_path)

    if not paths:
        raise RuntimeError(f"No extracted primitives found under {root}")

    path_array = np.stack(paths)
    velocity_array = np.stack(velocity_paths)
    if path_array.ndim != 3:
        raise ValueError(f"Expected primitive paths with shape (primitives, phase, joints), got {path_array.shape}")
    return metadata, path_array, velocity_array


def riemannian_frechet_mean(
    points: np.ndarray, metric: Metric, *, max_iterations: int = 20, tolerance: float = 1e-6,
    log_map_max_nfev: int = 60, endpoint_tol: float = 2e-3, max_step: float = 0.03,
) -> tuple[np.ndarray, dict[str, float | int]]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"Expected points with shape (samples, dimensions), got {points.shape}")

    q_mean = points.mean(axis=0)
    for iteration in range(1, max_iterations + 1):
        log_vectors = np.empty_like(points)
        for index, point in enumerate(points):
            log_vectors[index], _, _, _ = log_map_shooting(
                q_mean, point, metric, samples=2, max_nfev=log_map_max_nfev,
                endpoint_tol=endpoint_tol, max_step=max_step)

        mean_tangent = log_vectors.mean(axis=0)
        update_norm = norm(q_mean, mean_tangent, metric)
        if update_norm < tolerance:
            return q_mean, {"iterations": iteration, "update_norm": update_norm}

        q_mean = integrate_geodesic(
            q_mean, mean_tangent, metric, samples=2, max_step=max_step)[0][-1]

    raise RuntimeError(
        f"Riemannian Fréchet mean did not converge in {max_iterations} iterations; "
        f"last update norm={update_norm:.6g}")


def transport_primitives_to_reference(
    paths: np.ndarray, velocity_paths: np.ndarray, q_ref: np.ndarray, metric: Metric, *,
    path_samples: int = 80, log_map_max_nfev: int = 60, endpoint_tol: float = 2e-3,
    max_step: float = 0.03, norm_tolerance: float = 1e-6, show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    paths = np.asarray(paths, dtype=float)
    velocity_paths = np.asarray(velocity_paths, dtype=float)
    q_ref = np.asarray(q_ref, dtype=float)
    if paths.shape != velocity_paths.shape:
        raise ValueError(f"Path/velocity shape mismatch: {paths.shape} != {velocity_paths.shape}")
    if paths.ndim != 3 or paths.shape[2] != len(q_ref):
        raise ValueError(f"Expected paths with shape (primitives, phase, {len(q_ref)}), got {paths.shape}")

    transported_paths = np.empty((len(paths), path_samples, paths.shape[2]), dtype=float)
    transported_velocity_paths = np.empty_like(transported_paths)
    transported_tangents = np.empty((len(paths), paths.shape[2]), dtype=float)
    source_norms = np.empty(len(paths), dtype=float)
    transported_norms = np.empty(len(paths), dtype=float)
    transport_endpoint_errors = np.empty(len(paths), dtype=float)

    indices = tqdm(
        range(len(paths)), desc="Transporting primitives", unit="primitive",
        disable=not show_progress)
    for index in indices:
        q0 = paths[index, 0]
        v0 = velocity_paths[index, 0]
        _, transport_path, _, info = log_map_shooting(
            q0, q_ref, metric, samples=path_samples, max_nfev=log_map_max_nfev,
            endpoint_tol=endpoint_tol, max_step=max_step)
        v_ref = parallel_transport_along_polyline(v0, transport_path, metric)
        q_path, v_path = integrate_geodesic(
            q_ref, v_ref, metric, samples=path_samples, max_step=max_step)

        source_norm = norm(q0, v0, metric)
        transported_norm = norm(q_ref, v_ref, metric)
        norm_error = abs(transported_norm - source_norm)
        if norm_error > norm_tolerance * max(1.0, source_norm):
            raise RuntimeError(
                f"Primitive {index} transport norm error {norm_error:.6g} exceeds "
                f"{norm_tolerance * max(1.0, source_norm):.6g}")
        if not np.all(np.isfinite(q_path)) or not np.all(np.isfinite(v_path)):
            raise RuntimeError(f"Primitive {index} regenerated geodesic contains non-finite values")
        if not np.allclose(q_path[0], q_ref, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Primitive {index} does not start at q_ref")

        transported_paths[index] = q_path
        transported_velocity_paths[index] = v_path
        transported_tangents[index] = v_ref
        source_norms[index] = source_norm
        transported_norms[index] = transported_norm
        transport_endpoint_errors[index] = float(info["endpoint_error"])

    diagnostics = {
        "source_tangent_norm": source_norms,
        "transported_tangent_norm": transported_norms,
        "transport_endpoint_error": transport_endpoint_errors,
    }
    return transported_paths, transported_velocity_paths, transported_tangents, diagnostics


def riemannian_path_distance(
    path_a: np.ndarray, path_b: np.ndarray, metric: Metric, *, log_map_max_nfev: int = 60,
    endpoint_tol: float = 2e-3, max_step: float = 0.03,
) -> float:
    path_a = np.asarray(path_a, dtype=float)
    path_b = np.asarray(path_b, dtype=float)
    if path_a.shape != path_b.shape or path_a.ndim != 2:
        raise ValueError(f"Expected matching rank-2 paths, got {path_a.shape} and {path_b.shape}")
    if len(path_a) < 2:
        raise ValueError("Path distance requires at least two phase samples")

    squared_distances = np.zeros(len(path_a), dtype=float)
    for phase_index, (q_a, q_b) in enumerate(zip(path_a, path_b)):
        if np.array_equal(q_a, q_b):
            continue
        log_vector, _, _, _ = log_map_shooting(
            q_a, q_b, metric, samples=2, max_nfev=log_map_max_nfev,
            endpoint_tol=endpoint_tol, max_step=max_step)
        squared_distances[phase_index] = norm(q_a, log_vector, metric) ** 2

    weights = np.ones(len(path_a), dtype=float)
    weights[[0, -1]] = 0.5
    return float(np.sqrt(np.dot(weights, squared_distances) / (len(path_a) - 1)))


def build_candidate_edges(
    tangents: np.ndarray, q_ref: np.ndarray, metric: Metric, *, neighbours: int = 4,
) -> np.ndarray:
    tangents = np.asarray(tangents, dtype=float)
    if tangents.ndim != 2 or len(tangents) <= neighbours + 1:
        raise ValueError(f"Need more than {neighbours + 1} tangent vectors, got {len(tangents)}")

    metric_factor = np.linalg.cholesky(metric(np.asarray(q_ref, dtype=float))).T
    coordinates = tangents @ metric_factor.T
    nearest = cKDTree(coordinates).query(coordinates, k=neighbours + 1)[1][:, 1:]

    edges: set[tuple[int, int]] = set()
    for source, targets in enumerate(nearest):
        edges.update((min(source, int(target)), max(source, int(target))) for target in targets)
    return np.asarray(sorted(edges), dtype=int)


def compute_local_edge_distances(
    paths: np.ndarray, path_metrics: np.ndarray, edges: np.ndarray,
) -> np.ndarray:
    paths = np.asarray(paths, dtype=float)
    path_metrics = np.asarray(path_metrics)
    edges = np.asarray(edges, dtype=int)
    if path_metrics.shape != (*paths.shape, paths.shape[-1]):
        raise ValueError(
            f"Path metric shape {path_metrics.shape} does not match paths {paths.shape}")

    distances = np.empty(len(edges), dtype=float)
    for index, (source, target) in enumerate(edges):
        delta = paths[target] - paths[source]
        symmetric_metric = 0.5 * (path_metrics[source] + path_metrics[target])
        squared_distances = np.einsum(
            "ti,tij,tj->t", delta, symmetric_metric, delta)
        weights = np.ones(len(delta), dtype=float)
        weights[[0, -1]] = 0.5
        distances[index] = np.sqrt(
            np.dot(weights, squared_distances) / (len(delta) - 1))
    return distances


def compute_exact_edge_distances(
    paths: np.ndarray, edges: np.ndarray, metric: Metric, *, log_map_max_nfev: int = 60,
    endpoint_tol: float = 2e-3, max_step: float = 0.03, show_progress: bool = True,
) -> np.ndarray:
    paths = np.asarray(paths, dtype=float)
    edges = np.asarray(edges, dtype=int)
    distances = np.empty(len(edges), dtype=float)
    iterator = tqdm(
        edges, desc="Computing exact path distances", unit="edge",
        disable=not show_progress)
    for index, (source, target) in enumerate(iterator):
        distances[index] = riemannian_path_distance(
            paths[source], paths[target], metric, log_map_max_nfev=log_map_max_nfev,
            endpoint_tol=endpoint_tol, max_step=max_step)
    return distances


def evaluate_path_metrics(
    paths: np.ndarray, metric: Metric, *, show_progress: bool = True
) -> np.ndarray:
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 3:
        raise ValueError(f"Expected paths with shape (primitives, phase, joints), got {paths.shape}")

    matrices = np.empty((*paths.shape, paths.shape[-1]), dtype=float)
    iterator = tqdm(
        range(len(paths)), desc="Evaluating path metrics", unit="primitive",
        disable=not show_progress)
    for primitive_index in iterator:
        for phase_index, q in enumerate(paths[primitive_index]):
            matrices[primitive_index, phase_index] = metric(q)
    return matrices


def save_common_base_library(
    output_dir: str | Path, *, source_root: str | Path, metadata: list[dict[str, Any]],
    paths: np.ndarray, velocity_paths: np.ndarray, tangents: np.ndarray, q_ref: np.ndarray,
    path_metrics: np.ndarray, diagnostics: dict[str, np.ndarray], edges: np.ndarray,
    local_distances: np.ndarray, train_mask: np.ndarray, exact_validation_mask: np.ndarray,
    exact_distances: np.ndarray, frechet_info: dict[str, float | int], joint_names: tuple[str, ...],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        output_dir / "common_base_primitives.npz", paths=paths, velocity_paths=velocity_paths,
        initial_tangents=tangents, q_ref=q_ref)
    np.savez_compressed(
        output_dir / "distance_edges.npz", source=edges[:, 0], target=edges[:, 1],
        local_distance=local_distances, train_mask=train_mask,
        exact_validation_mask=exact_validation_mask, exact_distance=exact_distances)
    np.save(output_dir / "path_metrics.npy", np.asarray(path_metrics))

    diagnostic_names = tuple(diagnostics)
    with (output_dir / "primitive_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(metadata[0]) + list(diagnostic_names)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(metadata):
            writer.writerow({
                **row,
                **{name: float(diagnostics[name][index]) for name in diagnostic_names},
            })

    manifest = {
        "id": f"RiemannianPrimitives/{output_dir.name}",
        "source_root": str(Path(source_root).resolve()),
        "primitive_count": len(metadata),
        "path_shape": list(paths.shape),
        "joint_names": list(joint_names),
        "units": "radians",
        "phase_samples": int(paths.shape[1]),
        "q_ref": np.asarray(q_ref, dtype=float).tolist(),
        "frechet_mean": frechet_info,
        "edge_count": int(len(edges)),
        "training_edge_count": int(np.sum(train_mask)),
        "heldout_edge_count": int(np.sum(~train_mask)),
        "exact_validation_edge_count": int(np.sum(exact_validation_mask)),
        "training_distance": "symmetric_local_riemannian_quadratic",
        "evaluation_distance": "shooting_based_l2_path_distance",
        "primitives_file": "common_base_primitives.npz",
        "metadata_file": "primitive_metadata.csv",
        "distance_edges_file": "distance_edges.npz",
        "path_metrics_file": "path_metrics.npy",
    }
    (output_dir / "common_base.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir
