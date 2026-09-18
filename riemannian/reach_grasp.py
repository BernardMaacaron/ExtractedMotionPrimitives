from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from riemannian.extraction import extract_primitives
from riemannian.metric import MassMetricModel
from motion_primitives.randg import RIGHT_ARM_9_JOINT_NAMES, get_repetition_count, load_arm_joint_data


def extract_reach_grasp_primitives(
    subject: str, task: str, dataset_path: str | Path, output_dir: str | Path,
    params: dict[str, Any], overwrite: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    expected_repetitions = get_repetition_count(
        subject, task, dataset_path=dataset_path, extract_movements=True)
    repetitions = load_arm_joint_data(
        subject, task, dataset_path=dataset_path, return_all_joints=True,
        split_repetitions=True, extract_movements=True, verbose=False)
    if not isinstance(repetitions, list):
        raise TypeError(f"Expected split repetitions for {subject} {task}.")
    if len(repetitions) != expected_repetitions:
        raise ValueError(
            f"Expected {expected_repetitions} repetitions for {subject} {task}, "
            f"but loaded {len(repetitions)}.")

    root = Path(output_dir) / subject / task
    metric = MassMetricModel(name="Full Riemannian M(q)")
    completed = []
    resumed = []
    failures = []

    for repetition, (t, joints, _) in enumerate(repetitions):
        repetition_dir = root / f"rep-{repetition:02d}"
        status_path = repetition_dir / "status.json"
        if status_path.exists() and not overwrite:
            status = json.loads(status_path.read_text())
            if status["status"] == "completed":
                resumed.append(status)
                continue

        repetition_started = time.perf_counter()
        try:
            q = np.deg2rad(joints.loc[:, RIGHT_ARM_9_JOINT_NAMES].to_numpy(dtype=float))
            primitives = extract_primitives(q, t, metric, show_progress=False, **params)
            rows = []
            arrays = {}
            for primitive in primitives:
                key = f"primitive{primitive['primitive_index']:03d}"
                arrays[f"{key}_path"] = primitive["path"]
                arrays[f"{key}_velocity_path"] = primitive["velocity_path"]
                rows.append({
                    "subject": subject,
                    "task": task,
                    "repetition": repetition,
                    "primitive_index": primitive["primitive_index"],
                    "start": primitive["start"],
                    "end": primitive["end"],
                    "t0": primitive["t0"],
                    "t1": primitive["t1"],
                    "duration": primitive["duration"],
                    "length": primitive["length"],
                    "speed0": primitive["speed0"],
                    "speedf": primitive["speedf"],
                    "endpoint_error": primitive["endpoint_error"],
                    "tiny_segment_status": primitive["tiny_segment_status"],
                })

            repetition_dir.mkdir(parents=True, exist_ok=True)
            with (repetition_dir / "primitives_summary.csv").open(
                "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.savez_compressed(repetition_dir / "primitive_paths.npz", **arrays)

            status = {
                "subject": subject,
                "task": task,
                "repetition": repetition,
                "status": "completed",
                "n_primitives": len(primitives),
                "mean_endpoint_error": float(np.mean(
                    [primitive["endpoint_error"] for primitive in primitives])),
                "runtime_seconds": time.perf_counter() - repetition_started,
            }
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            completed.append(status)
        except Exception as exc:
            repetition_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "subject": subject,
                "task": task,
                "repetition": repetition,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            status_path.write_text(json.dumps(failure, indent=2) + "\n")
            failures.append(failure)

    statuses = completed + resumed
    primitive_count = sum(status["n_primitives"] for status in statuses)
    weighted_error = sum(
        status["mean_endpoint_error"] * status["n_primitives"] for status in statuses)
    return {
        "subject": subject,
        "task": task,
        "status": "failed" if failures else "completed",
        "expected_repetitions": expected_repetitions,
        "newly_completed_repetitions": len(completed),
        "resumed_repetitions": len(resumed),
        "failed_repetitions": failures,
        "n_primitives": primitive_count,
        "mean_endpoint_error": weighted_error / primitive_count if primitive_count else None,
        "runtime_seconds": time.perf_counter() - started,
    }
