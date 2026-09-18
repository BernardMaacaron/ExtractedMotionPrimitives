import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motion_primitives import randg
from motion_primitives.randg import (
    ARM_7_JOINT_NAMES,
    get_available_tasks,
    get_repetition_count,
    interpolate_joints,
    load_arm_joint_data,
)


def make_dataset(root: Path) -> Path:
    subject = "sub-01"
    task = "Cyl"
    motion_dir = root / subject / "motion"
    emg_dir = root / subject / "emg"
    motion_dir.mkdir(parents=True)
    emg_dir.mkdir(parents=True)

    time = np.arange(100, dtype=float) / 100.0
    channels = np.column_stack([
        np.sin(2 * np.pi * (index + 1) * time / 10.0)
        for index in range(31)
    ])
    pd.DataFrame(np.column_stack([time, channels])).to_csv(
        motion_dir / f"{subject}_task-{task}_acq-vicon_motion.csv",
        header=False, index=False
    )
    pd.DataFrame(np.column_stack([time, np.ones((100, 1))])).to_csv(
        emg_dir / f"{subject}_task-Other_acq-cometa_emg.csv",
        header=False, index=False
    )

    cuts = {
        "Events_ReachGrasp": {
            "subjects": [{
                "subject_name": subject,
                "tasks": [{"task_name": task, "time2cut": [10, 20, 40, 50]}],
            }]
        }
    }
    (root / "timeCuts.json").write_text(json.dumps(cuts))
    return root


def test_task_discovery_defaults_to_motion(tmp_path):
    dataset = make_dataset(tmp_path)
    assert get_available_tasks("sub-01", dataset_path=dataset) == ["Cyl"]


def test_repetition_count_matches_extraction_mode(tmp_path):
    dataset = make_dataset(tmp_path)
    assert get_repetition_count("sub-01", "Cyl", dataset_path=dataset) == 1
    assert get_repetition_count(
        "sub-01", "Cyl", dataset_path=dataset, extract_movements=False
    ) == 2


def test_arm_loading_uses_process_local_cache(tmp_path):
    dataset = make_dataset(tmp_path)
    randg._load_arm_joint_data_cached.cache_clear()
    first = load_arm_joint_data(
        "sub-01", "Cyl", dataset_path=dataset,
        split_repetitions=True, verbose=False
    )
    second = load_arm_joint_data(
        "sub-01", "Cyl", dataset_path=dataset,
        split_repetitions=True, verbose=False
    )

    assert len(first) == len(second) == 1
    assert first[0][1].shape[1] == 15
    assert first is second
    assert randg._load_arm_joint_data_cached.cache_info().currsize == 1
    assert second[0][2]["sampling_rate"] == pytest.approx(100.0)
    assert second[0][2]["duration"] == pytest.approx(0.19)


def test_interpolation_returns_simulation_grid(tmp_path):
    dataset = make_dataset(tmp_path)
    repetitions = load_arm_joint_data(
        "sub-01", "Cyl", dataset_path=dataset,
        split_repetitions=True, verbose=False
    )
    time_values, joints, _ = repetitions[0]
    selected = joints.loc[:, ARM_7_JOINT_NAMES]

    target, simulation_time, smoothed = interpolate_joints(
        time_values, selected.values, len(ARM_7_JOINT_NAMES), 0.19, 0.01,
        window_length=11
    )
    assert target.shape == (19, len(ARM_7_JOINT_NAMES))
    assert simulation_time.shape == (19,)
    assert smoothed.shape == selected.shape
