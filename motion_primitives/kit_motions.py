"""KIT Motion-Language Dataset loading helpers used by SOC training code."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motion_primitives.paths import KIT_MOTIONS_ROOT


KIT_RIGHT_ARM_7_JOINT_NAMES = (
    "RSx_joint", "RSy_joint", "RSz_joint", "REx_joint", "REz_joint", "RWx_joint", "RWy_joint")
KIT_MOTIONS_DIR = os.fspath(KIT_MOTIONS_ROOT)
_TRIAL_NAME = re.compile(r"(?P<task>.+)_right_arm_(?P<repetition>\d+)$")


def resolve_kit_motions_root(dataset_path: str | os.PathLike[str] | None = None) -> Path:
    dataset_root = Path(dataset_path or KIT_MOTIONS_DIR).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"KIT Motions dataset directory does not exist: {dataset_root}")
    return dataset_root


def _trial_records(dataset_root: Path) -> list[tuple[str, str, int, Path]]:
    records = []
    for path in dataset_root.glob("files_motions_*/*.xml"):
        match = _TRIAL_NAME.fullmatch(path.stem)
        if match is None:
            continue
        motion = ET.parse(path).find("Motion")
        records.append((motion.attrib["name"], match["task"], int(match["repetition"]), path))
    return sorted(records, key=lambda record: (record[0], record[1], record[2]))


def get_available_subjects(dataset_path: str | os.PathLike[str] | None = None) -> list[str]:
    return sorted({record[0] for record in _trial_records(resolve_kit_motions_root(dataset_path))})


def get_available_tasks(subject: str, dataset_path: str | os.PathLike[str] | None = None) -> list[str]:
    records = _trial_records(resolve_kit_motions_root(dataset_path))
    return sorted({task for record_subject, task, _, _ in records if record_subject == subject})


def get_repetition_count(
    subject: str, task: str, dataset_path: str | os.PathLike[str] | None = None,
) -> int:
    records = _trial_records(resolve_kit_motions_root(dataset_path))
    return sum(record_subject == subject and record_task == task
               for record_subject, record_task, _, _ in records)


def _load_trial(path: Path, subject: str, task: str) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    root = ET.parse(path).getroot()
    motion = root.find("Motion")
    joint_names = [joint.attrib["name"] for joint in motion.findall("./JointOrder/Joint")]
    frames = motion.findall("./MotionFrames/MotionFrame")
    time_values = np.asarray([float(frame.findtext("Timestep")) for frame in frames])
    joint_angles = pd.DataFrame(
        [[float(value) for value in frame.findtext("JointPosition").split()] for frame in frames],
        columns=joint_names)
    return time_values, joint_angles, {
        "sampling_rate": 100,
        "n_joints": len(joint_names),
        "duration": float(time_values[-1]),
        "subject": subject,
        "task": task,
        "dataset_path": os.fspath(path.parents[1]),
        "source_file": os.fspath(path),
        "units": "radians",
    }


def load_arm_joint_data(
    subject: str, task: str, dataset_path: str | os.PathLike[str] | None = None,
    return_all_joints: bool = False, split_repetitions: bool = True,
    extract_movements: bool = True, verbose: bool = True,
) -> list[tuple[np.ndarray, pd.DataFrame, dict[str, Any]]]:
    """Load the dataset's already-segmented XML movement trials."""
    if not split_repetitions or not extract_movements:
        raise ValueError("KIT Motions trials are stored as separate extracted movements.")
    dataset_root = resolve_kit_motions_root(dataset_path)
    records = [record for record in _trial_records(dataset_root)
               if record[0] == subject and record[1] == task]
    if not records:
        raise KeyError(f"No KIT Motions trials found for {subject} {task}.")
    repetitions = []
    for _, _, repetition, path in records:
        time_values, joint_angles, info = _load_trial(path, subject, task)
        if not return_all_joints:
            joint_angles = joint_angles[list(KIT_RIGHT_ARM_7_JOINT_NAMES)]
            info["n_joints"] = len(joint_angles.columns)
        info["repetition"] = repetition
        repetitions.append((time_values, joint_angles, info))
    if verbose:
        print(f"Loaded {len(repetitions)} KIT Motions movements")
    return repetitions
