from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from motion_primitives.paths import PROJECT_ROOT

from riemannian import MassMetricModel, extract_primitives
from motion_primitives.randg import RIGHT_ARM_9_JOINT_NAMES, RandG_dir, load_arm_joint_data


DEFAULT_SCORE_WEIGHTS = {
    "n_primitives": 1.0,
    "mean_endpoint_error": 1.0,
    "median_duration_penalty": 100.0,
    "failed_trials": 1000.0,
}


def stable_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:12]


def read_params(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def coerce_params(params: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(params)
    int_keys = ("window_length", "polyorder", "angle_confirm_samples", "min_segment_samples", "path_samples", "log_map_max_nfev")
    float_keys = ("delta_theta", "speed_floor_fraction", "min_tiny_segment_duration", "endpoint_tol", "geodesic_max_step")
    for key in int_keys:
        if key in coerced:
            coerced[key] = int(coerced[key])
    for key in float_keys:
        if key in coerced:
            coerced[key] = float(coerced[key])
    return coerced


def compute_score(diagnostics: dict[str, Any], score_weights: dict[str, float] | None = None) -> float:
    weights = DEFAULT_SCORE_WEIGHTS if score_weights is None else score_weights
    score = 0.0
    for key, weight in weights.items():
        score += float(weight) * float(diagnostics.get(key, 0.0))
    return float(score)


def evaluate_riemPrim_params(
    params: dict[str, Any],
    subjects: list[str],
    tasks: list[str],
    dataset_path: str | Path = RandG_dir,
    output_root: str | Path = PROJECT_ROOT / "results/ReachGrasp/riemannian/evaluate_riemPrim_params/local",
    max_repetitions: int | None = None,
    score_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, Any], Path]:
    defaults = dict(window_length=31, polyorder=2, delta_theta=float(np.pi / 3),
                    angle_confirm_samples=5, min_segment_samples=10, speed_floor_fraction=0.05,
                    min_tiny_segment_duration=0.15, path_samples=80, log_map_max_nfev=60,
                    endpoint_tol=2e-3, geodesic_max_step=0.03)
    unknown = params.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown extraction parameters: {sorted(unknown)}")
    params = coerce_params(defaults | params)
    weights = dict(DEFAULT_SCORE_WEIGHTS if score_weights is None else score_weights)
    run_payload = {
        "params": params,
        "score_weights": weights,
        "selection_policy": "minimize weighted scalar score; separate from viewer elbow selection",
        "subjects": list(subjects),
        "tasks": list(tasks),
        "dataset_path": str(dataset_path),
        "max_repetitions": max_repetitions,
        "angles_converted_to": "radians",
    }
    params_hash = stable_hash(run_payload)
    output_dir = Path(output_root) / params_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    metric = MassMetricModel(name="Full Riemannian M(q)")
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    path_arrays: dict[str, np.ndarray] = {}
    total_trials = 0
    start_time = time.perf_counter()

    for subject in subjects:
        for task in tasks:
            repetitions = load_arm_joint_data(
                subject, task, dataset_path=dataset_path,
                return_all_joints=True, split_repetitions=True,
                extract_movements=True, verbose=False)
            if not isinstance(repetitions, list):
                repetitions = [repetitions]
            if max_repetitions is not None:
                repetitions = repetitions[:max_repetitions]

            for trial_index, (t, joints, _info) in enumerate(repetitions):
                total_trials += 1
                trial_key = f"{subject}_{task}_trial{trial_index:03d}"
                try:
                    q = np.deg2rad(joints.loc[:, RIGHT_ARM_9_JOINT_NAMES].to_numpy(dtype=float))
                    primitives = extract_primitives(
                        q, t, metric,
                        **params,
                        show_progress=False)
                except Exception as exc:
                    failure_rows.append({
                        "subject": subject,
                        "task": task,
                        "trial_index": trial_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    continue

                for primitive in primitives:
                    primitive_key = f"{trial_key}_primitive{primitive['primitive_index']:03d}"
                    path_arrays[f"{primitive_key}_path"] = primitive["path"]
                    path_arrays[f"{primitive_key}_velocity_path"] = primitive["velocity_path"]
                    summary_rows.append({
                        "subject": subject,
                        "task": task,
                        "trial_index": trial_index,
                        "primitive_index": primitive["primitive_index"],
                        "start": primitive["start"],
                        "end": primitive["end"],
                        "t0": primitive["t0"],
                        "t1": primitive["t1"],
                        "duration": primitive["duration"],
                        "length": primitive["length"],
                        "endpoint_error": primitive["endpoint_error"],
                    })

    runtime_s = time.perf_counter() - start_time
    durations = np.array([row["duration"] for row in summary_rows], dtype=float)
    endpoint_errors = np.array([row["endpoint_error"] for row in summary_rows], dtype=float)
    total_duration = float(sum(row["duration"] for row in summary_rows))
    failed_trial_rate = float(len(failure_rows) / max(total_trials, 1))

    diagnostics = {
        "params_hash": params_hash,
        "n_subjects": len(subjects),
        "n_tasks": len(tasks),
        "total_trials": int(total_trials),
        "failed_trials": int(len(failure_rows)),
        "failed_trial_rate": failed_trial_rate,
        "n_primitives": int(len(summary_rows)),
        "mean_endpoint_error": float(np.mean(endpoint_errors)) if len(endpoint_errors) else 1e6,
        "max_endpoint_error": float(np.max(endpoint_errors)) if len(endpoint_errors) else 1e6,
        "median_duration": float(np.median(durations)) if len(durations) else 0.0,
        "min_duration": float(np.min(durations)) if len(durations) else 0.0,
        "segments_per_second": float(len(summary_rows) / max(total_duration, 1e-12)),
        "median_duration_penalty": float(max(0.0, 0.08 - np.median(durations))) if len(durations) else 1.0,
        "runtime_s": float(runtime_s),
    }
    score = compute_score(diagnostics, score_weights=weights)
    diagnostics["weighted_score_terms"] = {key: float(weight) * float(diagnostics[key])
                                            for key, weight in weights.items()}
    diagnostics["score"] = score

    if summary_rows:
        with open(output_dir / "primitives_summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if failure_rows:
        with open(output_dir / "failed_trials.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failure_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failure_rows)

    np.savez(output_dir / "primitive_paths.npz", **path_arrays)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_dir / "diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return score, diagnostics, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate riemPrim parameters locally or from NNI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Evaluation data and artifact scope.
    parser.add_argument("--subjects", nargs="+", default=["sub-01"],
                        help="ReachGrasp subjects included in the evaluation score.")
    parser.add_argument("--tasks", nargs="+", default=["FroRea"],
                        help="ReachGrasp tasks included in the evaluation score.")
    parser.add_argument("--dataset-path", type=str, default=RandG_dir, help="ReachGrasp dataset root.")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "results/ReachGrasp/riemannian/evaluate_riemPrim_params/local",
                        help="Root for hash-keyed evaluation artifacts and diagnostics.")
    parser.add_argument("--params-json", type=Path,
                        help="JSON parameter object; when supplied it replaces the extraction flags below.")
    parser.add_argument("--max-repetitions", type=int,
                        help="Maximum repetitions evaluated per subject/task; omit to use all repetitions.")

    # Primitive extraction parameters; meanings match extract_riemannian_primitives.py.
    parser.add_argument("--window-length", type=int, default=31,
                        help="Odd Savitzky-Golay window length in samples.")
    parser.add_argument("--polyorder", type=int, default=2, help="Savitzky-Golay polynomial order.")
    parser.add_argument("--delta-theta", type=float, default=float(np.pi / 3),
                        help="Radian velocity-direction change required for a segment boundary.")
    parser.add_argument("--angle-confirm-samples", type=int, default=5,
                        help="Consecutive samples required to confirm an angular boundary.")
    parser.add_argument("--speed-floor-fraction", type=float, default=0.05,
                        help="Fraction of peak speed below which direction estimates are ignored.")
    parser.add_argument("--min-segment-samples", type=int, default=10,
                        help="Minimum samples retained in a primitive segment.")
    parser.add_argument("--min-tiny-segment-duration", type=float, default=0.15,
                        help="Minimum duration in seconds before a short segment is merged.")
    parser.add_argument("--path-samples", type=int, default=80,
                        help="Phase samples stored along each reconstructed primitive path.")
    parser.add_argument("--log-map-max-nfev", type=int, default=60,
                        help="Maximum endpoint-shooting residual evaluations per primitive.")
    parser.add_argument("--endpoint-tol", type=float, default=2e-3,
                        help="Maximum accepted endpoint error in joint-space radians.")
    parser.add_argument("--geodesic-max-step", type=float, default=0.03,
                        help="Maximum integration step in normalized path phase.")
    return parser.parse_args()


def params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.params_json is not None:
        return read_params(args.params_json)
    return {
        "window_length": args.window_length,
        "polyorder": args.polyorder,
        "delta_theta": args.delta_theta,
        "angle_confirm_samples": args.angle_confirm_samples,
        "speed_floor_fraction": args.speed_floor_fraction,
        "min_segment_samples": args.min_segment_samples,
        "min_tiny_segment_duration": args.min_tiny_segment_duration,
        "path_samples": args.path_samples,
        "log_map_max_nfev": args.log_map_max_nfev,
        "endpoint_tol": args.endpoint_tol,
        "geodesic_max_step": args.geodesic_max_step,
    }


def main() -> None:
    args = parse_args()
    score, diagnostics, output_dir = evaluate_riemPrim_params(
        params_from_args(args), subjects=args.subjects, tasks=args.tasks,
        dataset_path=args.dataset_path, output_root=args.output_root,
        max_repetitions=args.max_repetitions)
    print(json.dumps({"default": score, "output_dir": str(output_dir), **diagnostics}, sort_keys=True))


if __name__ == "__main__":
    main()
