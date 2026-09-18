"""Export 3D-ARM-Gaze target trials, native gaze/context, and phase-normalized arm angles."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from motion_primitives import arm_gaze, preprocessing
from motion_primitives.paths import ARM_GAZE_ROOT, PROJECT_ROOT
from motion_primitives.preprocessing import endpoint_decomposition, phase_normalize


def prepare(dataset_path, output, samples=200, arm='custom', subjects=None):
    root, output = arm_gaze.data_root(dataset_path).resolve(), Path(output).resolve()
    if samples < 2:
        raise ValueError('At least two phase samples are required.')
    output.mkdir(parents=True, exist_ok=False)
    subjects = arm_gaze.get_available_subjects(root) if subjects is None else subjects
    pairs = [(subject, task) for subject in subjects for task in arm_gaze.get_available_tasks(subject, root)]
    rows, audits, recording_rows, native_q, native_time, phase_q, sources = [], [], [], [], [], [], []
    offsets = [0]
    for subject, task in tqdm(pairs, desc='3D-ARM-Gaze recordings'):
        data = arm_gaze.load_recording(subject, task, root)
        table = arm_gaze.trial_table(data, subject, task, root)
        n = len(data['timestamp'])
        index_info = json.loads((root / subject / f'exclude_indexs_{task}.json').read_text())
        sample_flag = np.zeros(n, dtype=bool)
        sample_flag[index_info['indexs']] = True
        np.testing.assert_allclose(data['timestamp'][sample_flag], index_info['timestamps'], rtol=0, atol=1e-6)
        context = {key: data[key] for key in (*arm_gaze.EVENT_FIELDS, *arm_gaze.CONTEXT_FIELDS)}
        context['tgtType'] = context['tgtType'].astype(str)
        context['author_flagged_sample'] = sample_flag
        for eye in ('left', 'right'):
            valid = data[f'{eye}EyeValidity'] == 31
            context[f'{eye}_gaze_valid'] = valid
            for quantity in ('Origin', 'Direction'):
                key = f'{eye}Eye{quantity}PosRefHead'
                context[key] = np.where(valid[:, None], data[key], np.nan)
        both = context['left_gaze_valid'] & context['right_gaze_valid']
        context['gazeFocusPos'] = np.where(both[:, None], data['gazeFocusPos'], np.nan)
        context_path = Path('context') / subject / f'{task}.npz'
        (output / context_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / context_path, **context)
        for repetition, row in enumerate(table[table.retained].to_dict('records')):
            a, b = row['source_start_index'], row['source_stop_index']
            part = {key: value[a:b] for key, value in data.items()}
            time = part['timestamp'] - part['timestamp'][0]
            raw_q = arm_gaze.joint_angles(part, arm)
            q = np.unwrap(raw_q, axis=0)
            phase_q.append(phase_normalize(time, q, samples))
            native_q.append(q)
            native_time.append(time)
            offsets.append(offsets[-1] + len(time))
            rows.append(dict(row, movement_id=f'3D-ARM-Gaze/{subject}/{task}/target-{row["target_number"]:03d}',
                             dataset='3D-ARM-Gaze', repetition=repetition, arm=arm,
                             duration_seconds=float(time[-1]), native_samples=len(time),
                             source_file=f'{subject}/{task}.json', context_file=str(context_path),
                             source_start_seconds=float(part['timestamp'][0]),
                             source_stop_seconds=float(part['timestamp'][-1]),
                             sampling_rate_hz=float(1 / np.median(np.diff(time))),
                             max_sample_gap_seconds=float(np.diff(time).max()),
                             angle_wrap_events=int(np.count_nonzero(np.abs(np.diff(raw_q, axis=0)) > np.pi)),
                             author_flagged_samples=int(sample_flag[a:b].sum()),
                             left_gaze_valid_fraction=float(context['left_gaze_valid'][a:b].mean()),
                             right_gaze_valid_fraction=float(context['right_gaze_valid'][a:b].mean())))
        audits.append(table)
        recording_rows.append(dict(subject=subject, action=task, samples=n,
                                   duration_seconds=float(data['timestamp'][-2] - data['timestamp'][0]),
                                   target_presentations=len(table), retained=int(table.retained.sum()),
                                   author_flagged_samples=int(sample_flag.sum()),
                                   pause_samples=int(data['expeInPause'].sum()),
                                   neutral_posture_samples=int(data['returnInitNeutralPosture'].sum())))
        for name in (f'{task}.json', f'targets_{task}.json', f'exclude_targets_{task}.json',
                     f'exclude_indexs_{task}.json', f'exclude_times_{task}.json'):
            path = root / subject / name
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            sources.append(dict(path=str(path.relative_to(root)), sha256=digest))
    for subject in subjects:
        for name in ('subj_info.json', 'range_of_motion.csv'):
            path = root / subject / name
            sources.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    metadata = pd.DataFrame(rows)
    q = np.stack(phase_q)
    h, f = endpoint_decomposition(q)
    ids, names = metadata.movement_id.to_numpy(dtype=str), np.array(arm_gaze.ARM_7_JOINT_NAMES)
    np.savez_compressed(output / 'native.npz', q=np.concatenate(native_q), time=np.concatenate(native_time),
                        offsets=np.array(offsets), movement_ids=ids, joint_names=names)
    np.savez_compressed(output / 'phase.npz', q=q, h=h, f=f, q_start=q[:, 0], q_goal=q[:, -1],
                        phase=np.linspace(0, 1, samples), movement_ids=ids, joint_names=names)
    metadata.to_csv(output / 'movements.csv', index=False)
    pd.concat(audits, ignore_index=True).to_csv(output / 'trial_audit.csv', index=False)
    pd.DataFrame(recording_rows).to_csv(output / 'recordings.csv', index=False)
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    datasets=[dict(name='3D-ARM-Gaze', root=str(root), acquisition='DBAS22')],
                    joint_names=names.tolist(), units='radians',
                    coordinates=f'DBAS22 {arm} arm; reflected Unity x; relative XZY/X/YXZ angles',
                    phase_samples=samples, movement_count=len(rows), subjects=subjects,
                    preprocessing=dict(segmentation='contiguous active target presentation, stop exclusive',
                        exclusions='author target list, first two samples, pauses plus two, RNP, last target, interrupted trials',
                        sample_exclusions='retained and flagged; target-level policy follows author analysis',
                        paired_targets='both retained with original target IDs', success_filter=None,
                        interpolation='linear in normalized recorded time, endpoints included',
                        smoothing=None, scaling=None, angle_unwrapping='numpy.unwrap per trial, period 2*pi',
                        endpoint_baseline='h(phi) = q_start + phi * (q_goal - q_start)'),
                    context=dict(files='context/<subject>/<phase>.npz', indexing='original source sample indices',
                        positions='metres, original left-handed Unity world frame; arbitrary across participants',
                        eye_origins='metres, original head frame', eye_directions='unit vectors, original head frame',
                        quaternions='xyzw, original Unity frame', gaze_mask='validity == 31; invalid values become NaN',
                        last_sample='retained in context for indexing, excluded from movements'),
                    arrays=dict(native='native.npz', phase='phase.npz'), metadata='movements.csv',
                    trial_audit='trial_audit.csv', recordings='recordings.csv', sources=sources,
                    code_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__), Path(arm_gaze.__file__), Path(preprocessing.__file__))})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(rows)} trials, seven {arm} arm angles, {samples} phase samples to {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path, default=ARM_GAZE_ROOT)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'collections/3D-ARM-Gaze/custom-phase200-v1')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--arm', choices=('custom', 'virtual'), default='custom')
    parser.add_argument('--subjects', nargs='+', help='Optional explicit subject subset, e.g. s1 s2.')
    args = parser.parse_args()
    prepare(args.dataset_path, args.output, args.samples, args.arm, args.subjects)
