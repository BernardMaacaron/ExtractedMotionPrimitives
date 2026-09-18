from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motion_primitives import randg


def test_arm_hand_joint_definition_uses_dataset_hand_joints() -> None:
    assert randg.HAND_16_JOINT_NAMES == tuple(randg.get_hand_joints())
    assert randg.ARM_HAND_23_JOINT_NAMES == randg.ARM_7_JOINT_NAMES + randg.HAND_16_JOINT_NAMES


def test_repetition_counts_use_time_cut_metadata(monkeypatch, tmp_path) -> None:
    time_cuts = list(range(22))
    monkeypatch.setattr(randg, "load_time_cuts", lambda base_path: {})
    monkeypatch.setattr(
        randg, "get_task_time_cuts",
        lambda data, subject, task: time_cuts)

    assert randg.get_repetition_count(
        "sub-01", "Pour", dataset_path=tmp_path, extract_movements=True) == 10
    assert randg.get_repetition_count(
        "sub-01", "Pour", dataset_path=tmp_path, extract_movements=False) == 11

    time_cuts.pop()
    with pytest.raises(ValueError, match="paired time cuts"):
        randg.get_repetition_count("sub-01", "Pour", dataset_path=tmp_path)


def test_loader_modes_are_cached_separately(monkeypatch, tmp_path) -> None:
    calls = []

    def load(subject, task, **kwargs):
        calls.append(kwargs["extract_movements"])
        count = 1 if kwargs["extract_movements"] else 2
        return [
            (np.array([0.0]), pd.DataFrame([[0.0]], columns=["joint"]))
            for _ in range(count)
        ]

    monkeypatch.setattr(randg, "load_joint_angles_with_labels", load)
    monkeypatch.setattr(randg, "get_joint_groups", lambda: {"all": ["joint"]})
    randg._load_arm_joint_data_cached.cache_clear()

    movements = randg.load_arm_joint_data(
        "sub-01", "FroRea", dataset_path=tmp_path, return_all_joints=True,
        split_repetitions=True, extract_movements=True, verbose=False)
    cached_movements = randg.load_arm_joint_data(
        "sub-01", "FroRea", dataset_path=tmp_path, return_all_joints=True,
        split_repetitions=True, extract_movements=True, verbose=False)
    task_periods = randg.load_arm_joint_data(
        "sub-01", "FroRea", dataset_path=tmp_path, return_all_joints=True,
        split_repetitions=True, extract_movements=False, verbose=False)

    assert cached_movements is movements
    assert len(task_periods) == 2
    assert calls == [True, False]
    assert randg._load_arm_joint_data_cached.cache_info().currsize == 2
