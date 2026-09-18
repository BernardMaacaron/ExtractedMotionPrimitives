"""Known arm rotations, trial exclusions, gaze masks, and exported collection identity."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from motion_primitives import arm_gaze
from motion_primitives.preprocessing import load_collections


def test_arm_gaze_loading_and_preprocessing(tmp_path):
    folder = tmp_path / 'DBAS22_DataOnline' / 's1'
    folder.mkdir(parents=True)
    phase = arm_gaze.PHASES[0]
    n = 72
    q = np.linspace([-0.2, -0.1, 0.3, -0.8, 0.1, 0.2, -0.1],
                    [0.1, -0.2, 0.5, -0.3, 0.2, 0.1, 0.1], n)
    q[48:60, 2] = np.linspace(3., 3.3, 12)  # Cross the principal Euler branch within target 4.
    root = Rotation.from_euler('xyz', [0.1, -0.2, 0.3])
    shoulder = root * Rotation.from_euler('XZY', q[:, :3])
    elbow = shoulder * Rotation.from_euler('X', q[:, 3:4])
    wrist = elbow * Rotation.from_euler('YXZ', q[:, 4:])
    quats = dict(armRootQuat=np.tile(root.as_quat(), (n, 1)), shouCustQuat=shoulder.as_quat(),
                 elbCustQuat=elbow.as_quat(), wriCustQuat=wrist.as_quat())
    quats.update({key.replace('Cust', 'Virt'): value for key, value in list(quats.items()) if 'Cust' in key})
    samples = []
    for i in range(n):
        sample = dict(timestamp=10 + i / 90, tgtNumber=i // 12, tgtType='classic', tgtRed=i % 12 >= 8,
                      objCatched=bool((i // 12) % 2), expeInPause=i == 4,
                      returnInitNeutralPosture=36 <= i < 48,
                      nbTgtsValidated=int(i >= 23), nbTgtsSkipped=0)
        for key in arm_gaze.CONTEXT_FIELDS:
            if key in ('leftEyeValidity', 'rightEyeValidity'):
                sample[key] = 0 if key == 'leftEyeValidity' and i == 50 else 31
            else:
                sample[key] = dict(x=0., y=0., z=1.)
                if key.endswith('Quat'):
                    sample[key] = dict(x=0., y=0., z=0., w=1.)
        for key, values in quats.items():
            sample[key] = dict(zip('xyzw', values[i] * [-1, 1, 1, -1]))
        samples.append(sample)
    (folder / f'{phase}.json').write_text('{"samples":[\n' + ',\n'.join(map(json.dumps, samples)) + '\n]}\n')
    (folder / f'exclude_indexs_{phase}.json').write_text(json.dumps(
        dict(number_of_original_data=n, indexs=[50], timestamps=[samples[50]['timestamp']])))
    (folder / f'exclude_times_{phase}.json').write_text(json.dumps({'indices': [10 + 50 / 90, 10 + 51 / 90]}))
    (folder / f'exclude_targets_{phase}.json').write_text(json.dumps({'targets': [2]}))
    (folder / f'targets_{phase}.json').write_text(json.dumps({'targets': [{}] * 6}))
    (folder / 'subj_info.json').write_text('{"armSide":"R"}')
    (folder / 'range_of_motion.csv').write_text('joint,min,max\nsp,-2,1\n')

    data = arm_gaze.load_recording('s1', phase, tmp_path)
    expected = (q + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(arm_gaze.joint_angles(data), expected, atol=1e-14)
    np.testing.assert_allclose(arm_gaze.joint_angles(data, 'virtual'), expected, atol=1e-14)
    assert arm_gaze.get_available_subjects(tmp_path) == ['s1']
    assert arm_gaze.get_available_tasks('s1', tmp_path) == [phase]
    assert arm_gaze.get_repetition_count('s1', phase, tmp_path) == 2
    table = arm_gaze.trial_table(data, 's1', phase, tmp_path)
    assert table.loc[0, 'exclusion_reason'] == 'interrupted_by_pause_or_neutral_posture'
    assert table.loc[2, 'exclusion_reason'] == 'author_excluded_target'
    assert table.loc[3, 'exclusion_reason'] == 'fewer_than_two_active_samples'
    assert table.loc[5, 'exclusion_reason'] == 'last_target'
    repetitions = arm_gaze.load_arm_joint_data('s1', phase, tmp_path)
    assert [info['target_number'] for _, _, info in repetitions] == [1, 4]
    assert [info['validated'] for _, _, info in repetitions] == [True, False]
    np.testing.assert_allclose(repetitions[0][0], np.arange(12) / 90)
    np.testing.assert_allclose(repetitions[0][1], q[12:24], atol=1e-14)
    np.testing.assert_allclose(repetitions[1][1], q[48:60], atol=1e-14)
    with pytest.raises(ValueError, match='continuous'):
        arm_gaze.load_arm_joint_data('s1', phase, tmp_path, split_repetitions=False)

    script = Path(__file__).parents[1] / 'scripts/prepare_arm_gaze.py'
    spec = importlib.util.spec_from_file_location('prepare_arm_gaze', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.prepare(tmp_path, tmp_path / 'collection', samples=5)
    loaded = load_collections([output])
    assert loaded['q'].shape == (2, 5, 7)
    np.testing.assert_allclose(loaded['q'][:, [0, -1]], q[[[12, 23], [48, 59]]], atol=1e-14)
    assert loaded['metadata'].target_number.tolist() == [1, 4]
    assert loaded['metadata'].angle_wrap_events.tolist() == [0, 1]
    with np.load(output / 'native.npz') as native:
        np.testing.assert_array_equal(native['offsets'], [0, 12, 24])
    with np.load(output / f'context/s1/{phase}.npz') as context:
        assert np.isnan(context['leftEyeDirectionPosRefHead'][50]).all()
        assert np.isnan(context['gazeFocusPos'][50]).all()
        assert np.isfinite(context['rightEyeDirectionPosRefHead'][50]).all()
        assert context['author_flagged_sample'][50]
    assert pd.read_csv(output / 'trial_audit.csv').retained.sum() == 2
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, output)
