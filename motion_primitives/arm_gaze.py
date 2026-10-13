"""3D-ARM-Gaze (DBAS22): target trials and the authors' seven-angle arm convention.

Raw positions stay in Unity world coordinates; joint extraction reflects the x axis
to match DBAS22_CodeOnline/code_python/tool_box/rot_quat_utils.py:quats2config.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from motion_primitives.paths import ARM_GAZE_ROOT

PHASES = ('initial_acquisition', 'test_RNP_after_pauses', 'test_RNP_after_targets_pairs')
ARM_7_JOINT_NAMES = ('shoulder_pitch', 'shoulder_roll', 'arm_yaw', 'elbow_pitch',
                     'forearm_yaw', 'wrist_pitch', 'wrist_roll')
EVENT_FIELDS = ('timestamp', 'tgtNumber', 'tgtType', 'tgtRed', 'objCatched', 'expeInPause',
                'returnInitNeutralPosture', 'nbTgtsValidated', 'nbTgtsSkipped')
GAZE_FIELDS = ('leftEyeOriginPosRefHead', 'leftEyeDirectionPosRefHead',
               'rightEyeOriginPosRefHead', 'rightEyeDirectionPosRefHead',
               'leftEyeValidity', 'rightEyeValidity', 'gazeFocusPos')
CONTEXT_FIELDS = ('tgtPos', 'tgtQuat', 'headSenPos', 'headSenQuat', 'armRootPos', 'armRootQuat',
                  'armRootInitPos', 'armRootInitQuat', 'endEffCustPos', 'endEffVirtPos', *GAZE_FIELDS)


def data_root(dataset_path=None):
    """Accept the release directory or its DBAS22_DataOnline child."""
    root = Path(ARM_GAZE_ROOT if dataset_path is None else dataset_path)
    return root / 'DBAS22_DataOnline' if (root / 'DBAS22_DataOnline').is_dir() else root


def get_available_subjects(dataset_path=None):
    return sorted((p.parent.name for p in data_root(dataset_path).glob('s*/subj_info.json')),
                  key=lambda name: int(name[1:]))


def get_available_tasks(subject, dataset_path=None):
    return [phase for phase in PHASES if (data_root(dataset_path) / subject / f'{phase}.json').is_file()]


def load_recording(subject, task, dataset_path=None, *, fields=None):
    """Read original samples, including the documented invalid final sample.

The release stores one sample per line. Stream that layout into arrays using the
published sample count, avoiding multi-GB nested Python objects. Explicit fields
allow access to additional sensors without importing the authors' GUI code.
"""
    if task not in PHASES:
        raise ValueError(f'Expected a reaching phase from {PHASES}, got {task!r}.')
    folder = data_root(dataset_path) / subject
    n = json.loads((folder / f'exclude_indexs_{task}.json').read_text())['number_of_original_data']
    if fields is None:
        fields = (*EVENT_FIELDS, *CONTEXT_FIELDS,
                  *(f'{joint}{arm}Quat' for arm in ('Cust', 'Virt') for joint in ('shou', 'elb', 'wri')))
    arrays = {}
    count = 0
    with (folder / f'{task}.json').open() as stream:
        if stream.readline().strip() != '{"samples":[':
            raise ValueError('Expected the DBAS22 one-sample-per-line JSON layout.')
        for line in stream:
            if line.strip() == ']}':
                break
            sample = json.loads(line.strip().removesuffix(','))
            for key in fields:
                value = sample[key]
                if isinstance(value, dict):
                    value = [value[axis] for axis in ('xyzw' if 'w' in value else 'xyz')]
                if count == 0:
                    a = np.asarray(value)
                    arrays[key] = np.empty((n, *a.shape), dtype=object if a.dtype.kind == 'U' else a.dtype)
                arrays[key][count] = value
            count += 1
    if count != n:
        raise ValueError(f'{subject}/{task}: expected {n} samples, read {count}.')
    return arrays


def joint_angles(data, arm='custom'):
    """Radians: relative shoulder XZY, elbow X, wrist YXZ with full forearm yaw.

The custom arm reproduces the authors' analysed teleoperation configuration.
The virtual option uses corrected shouVirtQuat, never shouVirtQuatRaw.
"""
    suffix = {'custom': 'Cust', 'virtual': 'Virt'}[arm]
    root, shoulder, elbow, wrist = [Rotation.from_quat(data[key] * [-1, 1, 1, -1])
        for key in ('armRootQuat', f'shou{suffix}Quat', f'elb{suffix}Quat', f'wri{suffix}Quat')]
    sh = (root.inv() * shoulder).as_euler('XZY')
    el = (shoulder.inv() * elbow).as_euler('XZY')[:, 0]
    forearm_pitch = shoulder * Rotation.from_euler('X', el[:, None])
    wr = (forearm_pitch.inv() * wrist).as_euler('YXZ')
    return np.column_stack((sh, el, wr))


def trial_table(data, subject, task, dataset_path=None):
    """One row per recorded target; retain original IDs and every rejection reason.

Use the published target exclusions (not the more aggressive sample exclusions).
Remove pauses plus two following samples, the first two samples, and neutral
posture intervals. Reject interrupted trials rather than joining across gaps.
Both repeated targets in the paired condition and unsuccessful trials are retained.
"""
    folder = data_root(dataset_path) / subject
    excluded = json.loads((folder / f'exclude_targets_{task}.json').read_text())['targets']
    targets = json.loads((folder / f'targets_{task}.json').read_text())['targets']
    numbers = data['tgtNumber']
    pause = data['expeInPause'].copy()
    pause[1:] |= data['expeInPause'][:-1]
    pause[2:] |= data['expeInPause'][:-2]
    pause[:2] = True
    active = ~(pause | data['returnInitNeutralPosture'])
    active[-1] = False
    bounds = np.r_[0, np.flatnonzero(np.diff(numbers)) + 1, len(numbers)]
    rows = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        target = int(numbers[start])
        indices = np.flatnonzero(active[start:stop]) + start
        reason = ''
        if target < 0 or target >= len(targets):
            reason = 'outside_target_set'
        elif target in excluded:
            reason = 'author_excluded_target'
        elif target == numbers[-1]:
            reason = 'last_target'  # Same conservative boundary as the authors' analysis.
        elif len(indices) < 2:
            reason = 'fewer_than_two_active_samples'
        elif np.any(np.diff(indices) != 1):
            reason = 'interrupted_by_pause_or_neutral_posture'
        a, b = (int(indices[0]), int(indices[-1]) + 1) if len(indices) else (int(start), int(start))
        validation_steps = np.flatnonzero(np.diff(data['nbTgtsValidated'][start:stop]) > 0) + start + 1
        validated = bool(len(validation_steps))
        rows.append(dict(subject=subject, action='reach', condition=task, target_number=target,
                         target_type=str(data['tgtType'][start]), success=validated, validated=validated,
                         validation_index=int(validation_steps[-1]) if validated else -1,
                         object_caught=bool(data['objCatched'][start]),
                         source_start_index=a, source_stop_index=b,
                         target_start_index=int(start), target_stop_index=int(stop),
                         retained=not reason, exclusion_reason=reason))
    table = pd.DataFrame(rows)
    if table.target_number.duplicated().any():
        raise ValueError(f'{subject}/{task}: non-contiguous repeated target IDs.')
    return table



def movement_bounds(data, row, arm='custom', onset_fraction=0.05, onset_samples=3):
    """Return stop-exclusive kinematic movement bounds inside one retained target trial.

    Onset is the first sustained hand-speed crossing of onset_fraction * peak speed.
    Successful trials end at the first sample of the final tgtRed run preceding the
    recorded validation event; failed trials retain the trial end/timeout.
    """
    a, b = int(row['source_start_index']), int(row['source_stop_index'])
    time = np.asarray(data['timestamp'][a:b], dtype=float)
    position_key = {'custom': 'endEffCustPos', 'virtual': 'endEffVirtPos'}[arm]
    position = np.asarray(data[position_key][a:b], dtype=float)
    dt = np.diff(time)
    speed = np.linalg.norm(np.diff(position, axis=0) / dt[:, None], axis=1)
    peak = float(speed.max(initial=0.0))
    threshold = onset_fraction * peak
    onset_local = 0
    if peak > 0 and len(speed) >= onset_samples:
        active = speed > threshold
        runs = np.convolve(active.astype(int), np.ones(onset_samples, dtype=int), mode='valid')
        crossings = np.flatnonzero(runs == onset_samples)
        if len(crossings):
            onset_local = int(crossings[0])

    stop = b
    arrival_index = -1
    if bool(row.get('success', row.get('validated', False))):
        validation_index = int(row.get('validation_index', -1))
        hi = min(b - 1, validation_index if validation_index >= 0 else b - 1)
        red = np.asarray(data['tgtRed'][a:hi + 1], dtype=bool)
        valid = np.flatnonzero(red)
        if len(valid):
            j = int(valid[-1])
            while j > 0 and red[j - 1]:
                j -= 1
            arrival_index = a + j
            stop = arrival_index + 1  # include the achieved target-entry sample

    start = a + onset_local
    onset_fallback = False
    if stop - start < 2:
        start = a
        onset_fallback = True
    return start, stop, dict(peak_hand_speed=peak, onset_speed_threshold=threshold,
                             arrival_index=arrival_index, onset_fallback=onset_fallback)

def get_repetition_count(subject, task, dataset_path=None):
    data = load_recording(subject, task, dataset_path, fields=EVENT_FIELDS)
    return int(trial_table(data, subject, task, dataset_path).retained.sum())


@lru_cache(maxsize=64)
def load_arm_joint_data(subject, task, dataset_path=None, return_all_joints=True,
                        split_repetitions=True, extract_movements=True, verbose=False, *, arm='custom',
                        trim_movement=True, onset_fraction=0.05, onset_samples=3):
    """Return (elapsed seconds, seven-angle DataFrame, metadata) for each retained target.

Repetition index is the retained trial's ordinal; target_number is the source ID.
All seven channels are arm joints, so return_all_joints does not change the output.
    Use load_recording for continuous raw data; gaps must not become one trajectory.
    Results are cached per recording for repeated SOC target loads; treat them as read-only.
"""
    if not split_repetitions or not extract_movements:
        raise ValueError('Use load_recording for continuous 3D-ARM-Gaze data.')
    suffix = {'custom': 'Cust', 'virtual': 'Virt'}[arm]
    data = load_recording(subject, task, dataset_path, fields=(*EVENT_FIELDS, 'armRootQuat',
                          *(f'{joint}{suffix}Quat' for joint in ('shou', 'elb', 'wri'))))
    table = trial_table(data, subject, task, dataset_path)
    repetitions = []
    for row in table[table.retained].to_dict('records'):
        if trim_movement:
            a, b, segmentation = movement_bounds(data, row, arm, onset_fraction, onset_samples)
        else:
            a, b = row['source_start_index'], row['source_stop_index']
            segmentation = {}
        part = {key: value[a:b] for key, value in data.items()}
        time = part['timestamp'] - part['timestamp'][0]
        q = np.unwrap(joint_angles(part, arm), axis=0)
        if not np.isfinite(q).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError(f'Invalid trajectory: {subject}/{task}/target-{row["target_number"]}.')
        info = dict(row, segmentation, repetition=len(repetitions), duration=float(time[-1]), units='radians',
                    n_joints=7, arm=arm, sampling_rate=float(1 / np.median(np.diff(time))),
                    dataset_path=str(data_root(dataset_path)),
                    source_file=str(data_root(dataset_path) / subject / f'{task}.json'))
        repetitions.append((time, pd.DataFrame(q, columns=ARM_7_JOINT_NAMES), info))
    if verbose:
        print(f'Loaded {len(repetitions)} 3D-ARM-Gaze target trials: {subject}/{task}')
    return repetitions
