from __future__ import annotations

import json

import numpy as np
import pandas as pd

from riemannian import reach_grasp
from motion_primitives.randg import RIGHT_ARM_9_JOINT_NAMES


def primitive(index: int = 0) -> dict:
    return {
        "primitive_index": index,
        "start": 0,
        "end": 2,
        "t0": 0.0,
        "t1": 0.02,
        "duration": 0.02,
        "length": 1.0,
        "speed0": 0.0,
        "speedf": 0.0,
        "endpoint_error": 1e-6,
        "tiny_segment_status": "normal",
        "path": np.zeros((3, len(RIGHT_ARM_9_JOINT_NAMES))),
        "velocity_path": np.zeros((3, len(RIGHT_ARM_9_JOINT_NAMES))),
    }


def test_subject_action_loads_once_and_resumes(monkeypatch, tmp_path) -> None:
    calls = {"load": 0, "extract": 0}
    repetitions = [
        (np.array([0.0, 0.01, 0.02]), pd.DataFrame(
            np.zeros((3, len(RIGHT_ARM_9_JOINT_NAMES))), columns=RIGHT_ARM_9_JOINT_NAMES), {})
        for _ in range(2)
    ]

    def load(*args, **kwargs):
        calls["load"] += 1
        return repetitions

    def extract(*args, **kwargs):
        calls["extract"] += 1
        return [primitive()]

    monkeypatch.setattr(reach_grasp, "get_repetition_count", lambda *args, **kwargs: 2)
    monkeypatch.setattr(reach_grasp, "load_arm_joint_data", load)
    monkeypatch.setattr(reach_grasp, "MassMetricModel", lambda **kwargs: object())
    monkeypatch.setattr(reach_grasp, "extract_primitives", extract)

    first = reach_grasp.extract_reach_grasp_primitives(
        "sub-01", "FroRea", "dataset", tmp_path, {})
    assert first["newly_completed_repetitions"] == 2
    assert calls == {"load": 1, "extract": 2}

    second = reach_grasp.extract_reach_grasp_primitives(
        "sub-01", "FroRea", "dataset", tmp_path, {})
    assert second["resumed_repetitions"] == 2
    assert calls == {"load": 2, "extract": 2}

    reach_grasp.extract_reach_grasp_primitives(
        "sub-01", "FroRea", "dataset", tmp_path, {}, overwrite=True)
    assert calls == {"load": 3, "extract": 4}
    status = json.loads((tmp_path / "sub-01/FroRea/rep-00/status.json").read_text())
    assert status["status"] == "completed"


def test_repetition_failure_does_not_discard_success(monkeypatch, tmp_path) -> None:
    repetitions = [
        (np.array([0.0, 0.01, 0.02]), pd.DataFrame(
            np.zeros((3, len(RIGHT_ARM_9_JOINT_NAMES))), columns=RIGHT_ARM_9_JOINT_NAMES), {})
        for _ in range(2)
    ]
    calls = 0

    def extract(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("shooting failed")
        return [primitive()]

    monkeypatch.setattr(reach_grasp, "get_repetition_count", lambda *args, **kwargs: 2)
    monkeypatch.setattr(reach_grasp, "load_arm_joint_data", lambda *args, **kwargs: repetitions)
    monkeypatch.setattr(reach_grasp, "MassMetricModel", lambda **kwargs: object())
    monkeypatch.setattr(reach_grasp, "extract_primitives", extract)

    result = reach_grasp.extract_reach_grasp_primitives(
        "sub-01", "FroRea", "dataset", tmp_path, {})
    assert result["newly_completed_repetitions"] == 1
    assert result["failed_repetitions"][0]["repetition"] == 1
    assert json.loads((tmp_path / "sub-01/FroRea/rep-00/status.json").read_text())["status"] == "completed"
    assert json.loads((tmp_path / "sub-01/FroRea/rep-01/status.json").read_text())["status"] == "failed"
