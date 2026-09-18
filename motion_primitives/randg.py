"""ReachGrasp dataset loading helpers used by SOC training code."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from reachgrasp.data_loading import (
    get_arm_joints,
    get_available_subjects as reachgrasp_get_available_subjects,
    get_available_tasks as reachgrasp_get_available_tasks,
    get_hand_joints,
    get_joint_groups,
    get_task_time_cuts,
    load_joint_angles_with_labels,
    load_time_cuts,
)
from motion_primitives.paths import REACHGRASP_ROOT


ARM_REGION_NAMES = ("Thorax_Left", "Thorax_Right", "Shoulder", "Elbow", "Wrist")
ARM_7_JOINT_NAMES = (
    "RShoulder_X",
    "RShoulder_Y",
    "RShoulder_Z",
    "RElbow_X",
    "RWrist_X",
    "RWrist_Y",
    "RWrist_Z",
)
RIGHT_ARM_9_JOINT_NAMES = (
    "RShoulder_X", "RShoulder_Y", "RShoulder_Z",
    "RElbow_X", "RElbow_Y", "RElbow_Z",
    "RWrist_X", "RWrist_Y", "RWrist_Z",
)
HAND_16_JOINT_NAMES = tuple(get_hand_joints())
ARM_HAND_23_JOINT_NAMES = ARM_7_JOINT_NAMES + HAND_16_JOINT_NAMES


def _default_randg_root() -> Path:
    environment_path = os.getenv("ARC_IIT_RANDG_DIR")
    if environment_path:
        return Path(environment_path).expanduser().resolve()
    return REACHGRASP_ROOT.resolve()


RandG_dir = os.fspath(_default_randg_root())


def resolve_randg_root(dataset_path: str | os.PathLike[str] | None = None) -> Path:
    dataset_root = Path(dataset_path or RandG_dir).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ReachGrasp dataset directory does not exist: {dataset_root}")
    return dataset_root


def get_available_subjects(dataset_path: str | os.PathLike[str] | None = None) -> list[str]:
    dataset_root = resolve_randg_root(dataset_path)
    return reachgrasp_get_available_subjects(base_path=os.fspath(dataset_root))


def get_available_tasks(
    subject: str, modality: str = "motion",
    dataset_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    dataset_root = resolve_randg_root(dataset_path)
    return reachgrasp_get_available_tasks(
        str(subject), modality=modality, base_path=os.fspath(dataset_root))


def get_repetition_count(
    subject: str, task: str, dataset_path: str | os.PathLike[str] | None = None,
    extract_movements: bool = True,
) -> int:
    dataset_root = resolve_randg_root(dataset_path)
    time_cuts = get_task_time_cuts(
        load_time_cuts(os.fspath(dataset_root)), str(subject), str(task))
    if not time_cuts:
        raise KeyError(f"No time cuts found for {subject} {task}.")
    if len(time_cuts) % 2:
        raise ValueError(f"Expected paired time cuts for {subject} {task}, got {len(time_cuts)} values.")
    task_periods = len(time_cuts) // 2
    return task_periods - 1 if extract_movements else task_periods


@lru_cache(maxsize=16)
def _load_arm_joint_data_cached(
    subject: str, task: str, dataset_root_str: str, return_all_joints: bool,
    split_repetitions: bool, extract_movements: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]] | list[tuple[np.ndarray, pd.DataFrame, dict[str, Any]]]:
    result = load_joint_angles_with_labels(
        subject, task, acquisition="vicon", base_path=dataset_root_str,
        split_repetitions=split_repetitions, extract_movements=extract_movements,
        verbose=False)
    joint_groups = get_joint_groups()

    if return_all_joints:
        selected_regions = list(joint_groups)
        selected_groups = joint_groups
        selected_joint_names = None
    else:
        selected_regions = list(ARM_REGION_NAMES)
        selected_groups = {region: joint_groups[region] for region in selected_regions}
        selected_joint_names = get_arm_joints()

    if isinstance(result, list):
        repetitions = []
        for time_values, joint_angles_full in result:
            joint_angles = joint_angles_full
            if selected_joint_names is not None:
                joint_angles = joint_angles_full[selected_joint_names]
            repetitions.append((time_values, joint_angles, {
                "groups": selected_groups,
                "regions": selected_regions,
                "sampling_rate": 100,
                "n_joints": len(joint_angles.columns),
                "duration": float(time_values[-1]) if len(time_values) > 0 else 0.0,
                "subject": subject,
                "task": task,
                "dataset_path": dataset_root_str,
            }))
        return repetitions

    time_values, joint_angles_full = result
    joint_angles = joint_angles_full
    if selected_joint_names is not None:
        joint_angles = joint_angles_full[selected_joint_names]
    return time_values, joint_angles, {
        "groups": selected_groups,
        "regions": selected_regions,
        "sampling_rate": 100,
        "n_joints": len(joint_angles.columns),
        "duration": float(time_values[-1]) if len(time_values) > 0 else 0.0,
        "subject": subject,
        "task": task,
        "dataset_path": dataset_root_str,
    }


def load_arm_joint_data(
    subject: str, task: str, dataset_path: str | os.PathLike[str] | None = None,
    return_all_joints: bool = False, split_repetitions: bool = True,
    extract_movements: bool = True, verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]] | list[tuple[np.ndarray, pd.DataFrame, dict[str, Any]]]:
    """Load Vicon arm trajectories and attach the metadata used by SOC."""
    dataset_root = resolve_randg_root(dataset_path)
    result = _load_arm_joint_data_cached(
        str(subject), str(task), os.fspath(dataset_root), bool(return_all_joints),
        bool(split_repetitions), bool(extract_movements))
    if split_repetitions and not isinstance(result, list):
        raise RuntimeError(f"Could not split repetitions for {subject} {task}.")
    if verbose and isinstance(result, list):
        period_type = "movements" if extract_movements else "task execution periods"
        print(f"Split data into {len(result)} {period_type}")
    return result


def interpolate_joints(
    time_values, joint_data, numOutput, simTime, dt, window_length=51,
    polyorder=3, noise_threshold=1e-4,
):
    joint_data_smooth = np.zeros_like(joint_data)
    for idx in range(numOutput):
        smoothed = savgol_filter(
            joint_data[:, idx], window_length=window_length, polyorder=polyorder)
        if np.max(smoothed) - np.min(smoothed) < noise_threshold:
            smoothed.fill(np.mean(smoothed))
        joint_data_smooth[:, idx] = smoothed

    sim_time = np.arange(0, simTime, dt)
    target_signal = np.zeros((len(sim_time), numOutput))
    for idx in range(numOutput):
        interp_func = interp1d(
            time_values, joint_data_smooth[:, idx], kind="cubic",
            fill_value="extrapolate")
        target_signal[:, idx] = interp_func(sim_time)

    if numOutput == 1:
        target_signal = target_signal.flatten()

    return target_signal, sim_time, joint_data_smooth
