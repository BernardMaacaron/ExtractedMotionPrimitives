"""Andy prescient-teleoperation dataset loading helpers used by SOC training code."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motion_primitives.paths import ANDY_DATA_ROOT


ANDY_SUBJECT = "iCub"
ANDY_DATA_DIR = os.fspath(ANDY_DATA_ROOT)
ANDY_SAMPLING_RATE = 250
ICUB_32_DOF_NAMES = tuple(f"iCub_DOF_{index:02d}" for index in range(32))
ICUB_POSITION_NAMES = (
    "waist_z", "right_hand_x", "right_hand_y", "right_hand_z",
    "left_hand_x", "left_hand_y", "left_hand_z")
ICUB_CENTER_OF_MASS_NAMES = ("center_of_mass_x", "center_of_mass_y", "center_of_mass_z")
_MODALITIES = {
    "joints": ("", ICUB_32_DOF_NAMES),
    "positions": ("p", ICUB_POSITION_NAMES),
    "center_of_mass": ("c", ICUB_CENTER_OF_MASS_NAMES),
}


def resolve_andy_data_root(dataset_path: str | os.PathLike[str] | None = None) -> Path:
    dataset_root = Path(dataset_path or ANDY_DATA_DIR).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Andy dataset directory does not exist: {dataset_root}")
    return dataset_root


def get_available_subjects(dataset_path: str | os.PathLike[str] | None = None) -> list[str]:
    resolve_andy_data_root(dataset_path)
    return [ANDY_SUBJECT]


def get_available_tasks(
    subject: str, dataset_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    if subject != ANDY_SUBJECT:
        raise KeyError(f"Unknown Andy dataset subject: {subject}")
    dataset_root = resolve_andy_data_root(dataset_path)
    multiple_tasks = dataset_root / "datasetMultipleTasks"
    return ["Goals", "Obstacles", *sorted(path.name for path in multiple_tasks.iterdir() if path.is_dir())]


def _task_directory(dataset_root: Path, task: str) -> Path:
    if task in ("Goals", "Obstacles"):
        task_dir = dataset_root / f"dataset{task}"
    else:
        task_dir = dataset_root / "datasetMultipleTasks" / task
    if not task_dir.is_dir():
        raise KeyError(f"Unknown Andy dataset task: {task}")
    return task_dir


def _joint_trial_paths(dataset_root: Path, task: str) -> list[Path]:
    task_dir = _task_directory(dataset_root, task)
    return sorted(
        (path for path in task_dir.glob("*.csv") if path.stem.isdigit()),
        key=lambda path: int(path.stem))


def get_repetition_count(
    subject: str, task: str, dataset_path: str | os.PathLike[str] | None = None,
) -> int:
    if subject != ANDY_SUBJECT:
        raise KeyError(f"Unknown Andy dataset subject: {subject}")
    return len(_joint_trial_paths(resolve_andy_data_root(dataset_path), task))


def load_trajectory_data(
    subject: str, task: str, modality: str = "joints",
    dataset_path: str | os.PathLike[str] | None = None, split_repetitions: bool = True,
    extract_movements: bool = True, verbose: bool = True,
) -> list[tuple[np.ndarray, pd.DataFrame, dict[str, Any]]]:
    """Load the dataset's already-segmented CSV movement trials."""
    if subject != ANDY_SUBJECT:
        raise KeyError(f"Unknown Andy dataset subject: {subject}")
    if not split_repetitions or not extract_movements:
        raise ValueError("Andy trajectories are stored as separate extracted movements.")
    prefix, column_names = _MODALITIES[modality]
    dataset_root = resolve_andy_data_root(dataset_path)
    joint_paths = _joint_trial_paths(dataset_root, task)
    repetitions = []
    for joint_path in joint_paths:
        path = joint_path.with_name(f"{prefix}{joint_path.name}")
        trajectory = pd.read_csv(path, header=None, names=column_names)
        time_values = np.arange(len(trajectory), dtype=float) / ANDY_SAMPLING_RATE
        repetitions.append((time_values, trajectory, {
            "sampling_rate": ANDY_SAMPLING_RATE,
            "n_joints": len(trajectory.columns),
            "duration": float(time_values[-1]),
            "subject": subject,
            "task": task,
            "dataset_path": os.fspath(dataset_root),
            "source_file": os.fspath(path),
            "repetition": int(joint_path.stem),
            "modality": modality,
            "units": "unspecified",
        }))
    if verbose:
        print(f"Loaded {len(repetitions)} Andy {modality} movements")
    return repetitions
