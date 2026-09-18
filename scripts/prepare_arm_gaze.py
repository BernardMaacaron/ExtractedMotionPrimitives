"""Export schema-v2 3D-ARM-Gaze trials with kinematic movement bounds and named views."""
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


def prepare(dataset_path, output, samples=200, arm='custom', subjects=None,
            onset_fraction=0.05, onset_samples=3):
    root, output = arm_gaze.data_root(dataset_path).resolve(), Path(output).resolve()
    if samples < 2:
        raise ValueError('At least two phase samples are required.')
    output.mkdir(parents=True, exist_ok=False)
    subjects = arm_gaze.get_available_subjects(root) if subjects is None else subjects
    pairs = [(subject, condition) for subject in subjects
             for condition in arm_gaze.get_available_tasks(subject, root)]
    rows, audits, recording_rows, native_q, native_time, phase_q, sources = [], [], [], [], [], [], []
    offsets = [0]

    for subject, condition in tqdm(pairs, desc='3D-ARM-Gaze recordings'):
        data = arm_gaze.load_recording(subject, condition, root)
        table = arm_gaze.trial_table(data, subject, condition, root)
        n = len(data['timestamp'])
        index_info = json.loads((root / subject / f'exclude_indexs_{condition}.json').read_text())
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
        context_path = Path('context') / subject / f'{condition}.npz'
        (output / context_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / context_path, **context)

        for repetition, original in enumerate(table[table.retained].to_dict('records')):
            presentation_start = int(original['source_start_index'])
            presentation_stop = int(original['source_stop_index'])
            a, b, segmentation = arm_gaze.movement_bounds(
                data, original, arm=arm, onset_fraction=onset_fraction, onset_samples=onset_samples)
            part = {key: value[a:b] for key, value in data.items()}
            time = part['timestamp'] - part['timestamp'][0]
            raw_q = arm_gaze.joint_angles(part, arm)
            q = np.unwrap(raw_q, axis=0)
            phase_q.append(phase_normalize(time, q, samples))
            native_q.append(q)
            native_time.append(time)
            offsets.append(offsets[-1] + len(time))

            target_position = np.asarray(data['tgtPos'][a], dtype=float)
            target_orientation = np.asarray(data['tgtQuat'][a], dtype=float)
            trial_id = f'3D-ARM-Gaze/{subject}/{condition}/target-{original["target_number"]:03d}'
            row = dict(original)
            row.update(segmentation)
            row.update(
                trial_id=trial_id, movement_id=trial_id, dataset='3D-ARM-Gaze',
                action='reach', condition=condition, repetition=repetition, arm=arm,
                presentation_start_index=presentation_start,
                presentation_stop_index=presentation_stop,
                source_start_index=a, source_stop_index=b,
                duration_seconds=float(time[-1]), native_samples=len(time),
                source_file=f'{subject}/{condition}.json', context_file=str(context_path),
                source_start_seconds=float(part['timestamp'][0]),
                source_stop_seconds=float(part['timestamp'][-1]),
                sampling_rate_hz=float(1 / np.median(np.diff(time))),
                max_sample_gap_seconds=float(np.diff(time).max()),
                angle_wrap_events=int(np.count_nonzero(np.abs(np.diff(raw_q, axis=0)) > np.pi)),
                author_flagged_samples=int(sample_flag[a:b].sum()),
                left_gaze_valid_fraction=float(context['left_gaze_valid'][a:b].mean()),
                right_gaze_valid_fraction=float(context['right_gaze_valid'][a:b].mean()),
                target_position_x=target_position[0], target_position_y=target_position[1],
                target_position_z=target_position[2],
                target_orientation_x=target_orientation[0], target_orientation_y=target_orientation[1],
                target_orientation_z=target_orientation[2], target_orientation_w=target_orientation[3],
            )
            rows.append(row)

        audits.append(table)
        recording_rows.append(dict(
            subject=subject, condition=condition, samples=n,
            duration_seconds=float(data['timestamp'][-2] - data['timestamp'][0]),
            target_presentations=len(table), retained=int(table.retained.sum()),
            successful=int(table.loc[table.retained, 'success'].sum()),
            author_flagged_samples=int(sample_flag.sum()),
            pause_samples=int(data['expeInPause'].sum()),
            neutral_posture_samples=int(data['returnInitNeutralPosture'].sum())))

        for name in (f'{condition}.json', f'targets_{condition}.json', f'exclude_targets_{condition}.json',
                     f'exclude_indexs_{condition}.json', f'exclude_times_{condition}.json'):
            source = root / subject / name
            with source.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            sources.append(dict(path=str(source.relative_to(root)), sha256=digest))

    for subject in subjects:
        for name in ('subj_info.json', 'range_of_motion.csv'):
            source = root / subject / name
            sources.append(dict(path=str(source.relative_to(root)),
                                sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

    metadata = pd.DataFrame(rows)
    q = np.stack(phase_q)
    h, f = endpoint_decomposition(q)
    ids = metadata.trial_id.to_numpy(dtype=str)
    names = np.array(arm_gaze.ARM_7_JOINT_NAMES)
    successful = np.flatnonzero(metadata.success.to_numpy(dtype=bool))

    np.savez_compressed(output / 'native.npz', q=np.concatenate(native_q), time=np.concatenate(native_time),
                        offsets=np.array(offsets), trial_ids=ids, movement_ids=ids, joint_names=names)
    np.savez_compressed(output / 'phase.npz', q=q, h=h, f=f, q_start=q[:, 0], q_end=q[:, -1],
                        phase=np.linspace(0, 1, samples), trial_ids=ids, movement_ids=ids, joint_names=names)
    np.savez_compressed(output / 'views.npz', all=np.arange(len(metadata)), successful=successful)
    metadata.to_csv(output / 'movements.csv', index=False)
    pd.concat(audits, ignore_index=True).to_csv(output / 'trial_audit.csv', index=False)
    pd.DataFrame(recording_rows).to_csv(output / 'recordings.csv', index=False)

    manifest = dict(
        schema_version=2, created_utc=datetime.now(timezone.utc).isoformat(),
        datasets=[dict(name='3D-ARM-Gaze', root=str(root), acquisition='DBAS22')],
        joint_names=names.tolist(), units='radians',
        coordinates=f'DBAS22 {arm} arm; reflected Unity x; relative XZY/X/YXZ angles',
        phase_samples=samples, trial_count=len(rows), movement_count=len(rows), subjects=subjects,
        views=dict(all=len(rows), successful=int(len(successful))),
        preprocessing=dict(
            segmentation=('retained target trial; onset = first sustained hand-speed crossing; '
                          'successful stop = first sample of final tgtRed run before validation; '
                          'failed stop = retained presentation end; all bounds stop-exclusive'),
            onset_fraction=onset_fraction, onset_samples=onset_samples,
            exclusions='author target list, first two samples, pauses plus two, RNP, last target, interrupted trials',
            sample_exclusions='retained and flagged; target-level policy follows author analysis',
            paired_targets='both retained with original target IDs',
            interpolation='linear in normalized recorded time, endpoints included',
            smoothing=None, scaling=None, angle_unwrapping='numpy.unwrap per trial, period 2*pi',
            endpoint_baseline='h(phi) = q_start + phi * (q_end - q_start)'),
        context=dict(
            files='context/<subject>/<condition>.npz', indexing='original source sample indices',
            positions='metres, original left-handed Unity world frame; arbitrary across participants',
            eye_origins='metres, original head frame', eye_directions='unit vectors, original head frame',
            quaternions='xyzw, original Unity frame', gaze_mask='validity == 31; invalid values become NaN',
            last_sample='retained in context for indexing, excluded from trials'),
        arrays=dict(native='native.npz', phase='phase.npz', views='views.npz'),
        metadata='movements.csv', trial_audit='trial_audit.csv', recordings='recordings.csv',
        sources=sources,
        code_sha256={str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                     for source in (Path(__file__), Path(arm_gaze.__file__), Path(preprocessing.__file__))})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(rows)} trials ({len(successful)} successful), seven {arm} arm angles, '
          f'{samples} phase samples to {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path, default=ARM_GAZE_ROOT)
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'collections/3D-ARM-Gaze/custom-phase200-v2')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--arm', choices=('custom', 'virtual'), default='custom')
    parser.add_argument('--subjects', nargs='+', help='Optional explicit subject subset, e.g. s1 s2.')
    parser.add_argument('--onset-fraction', type=float, default=0.05)
    parser.add_argument('--onset-samples', type=int, default=3)
    args = parser.parse_args()
    prepare(args.dataset_path, args.output, args.samples, args.arm, args.subjects,
            args.onset_fraction, args.onset_samples)
