"""Export ReachGrasp native and phase trajectories without SOC artifacts or scaling."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from reachgrasp import data_loading
from motion_primitives.preprocessing import endpoint_decomposition, phase_normalize
from motion_primitives import preprocessing

ROOT = Path(__file__).resolve().parents[1]


def prepare(dataset_path, output, samples=200, exclude=()):
    dataset_path, output = Path(dataset_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pairs = [(subject, task) for subject in data_loading.get_available_subjects(str(dataset_path))
             for task in data_loading.get_available_tasks(subject, modality='motion', base_path=str(dataset_path))]
    rows, native_q, native_time, phase_q, offsets, sources = [], [], [], [], [0], []
    joint_names = None
    for subject, task in tqdm(pairs, desc='ReachGrasp recordings'):
        if f'{subject}/{task}' in exclude:
            continue
        stem = dataset_path / subject / 'motion' / f'{subject}_task-{task}_acq-vicon'
        channel_path = Path(f'{stem}_channels.tsv')
        channels = pd.read_csv(channel_path, sep='\t')
        if set(channels.units) != {'DEG'}:
            raise ValueError(f'Expected degree-valued joint angles: {channel_path}')
        movements = data_loading.load_joint_angles_with_labels(
            subject, task, base_path=str(dataset_path), split_repetitions=True,
            extract_movements=True, clamp_near_zero=False, verbose=False)
        cuts = data_loading.get_task_time_cuts(data_loading.load_time_cuts(str(dataset_path)), subject, task)
        for repetition, (time, joints) in enumerate(movements):
            if joint_names is None:
                joint_names = joints.columns.tolist()
                source_channel_names = channels.name.tolist()
            if joints.columns.tolist() != joint_names or channels.name.tolist() != source_channel_names:
                raise ValueError(f'Joint order mismatch: {subject}/{task}')
            q = joints.to_numpy(dtype=float)
            phase_q.append(phase_normalize(time, q, samples))
            native_q.append(q)
            native_time.append(time)
            offsets.append(offsets[-1] + len(time))
            rows.append(dict(movement_id=f'ReachGrasp/{subject}/{task}/rep-{repetition:02d}',
                             dataset='ReachGrasp', subject=subject, action=task, repetition=repetition,
                             duration_seconds=float(time[-1] - time[0]), native_samples=len(time),
                             source_file=str(Path(f'{stem}_motion.csv').relative_to(dataset_path)),
                             source_start_index=cuts[2 * repetition + 1],
                             source_stop_index=cuts[2 * repetition + 2]))
        for path in (Path(f'{stem}_motion.csv'), channel_path, Path(f'{stem}_motion.json')):
            sources.append(dict(path=str(path.relative_to(dataset_path)),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    metadata = pd.DataFrame(rows)
    q = np.stack(phase_q)
    h, f = endpoint_decomposition(q)
    ids = metadata.movement_id.to_numpy(dtype=str)
    np.savez_compressed(output / 'native.npz', q=np.concatenate(native_q), time=np.concatenate(native_time),
                        offsets=np.array(offsets), movement_ids=ids, joint_names=np.array(joint_names))
    np.savez_compressed(output / 'phase.npz', q=q, h=h, f=f, q_start=q[:, 0], q_goal=q[:, -1],
                        phase=np.linspace(0, 1, samples), movement_ids=ids, joint_names=np.array(joint_names))
    metadata.to_csv(output / 'movements.csv', index=False)
    cuts_path = dataset_path / 'timeCuts.json'
    sources.append(dict(path='timeCuts.json', sha256=hashlib.sha256(cuts_path.read_bytes()).hexdigest()))
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    datasets=[dict(name='ReachGrasp', root=str(dataset_path), acquisition='vicon')],
                    joint_names=joint_names, source_channel_names=source_channel_names,
                    units='degrees', coordinates='ReachGrasp Vicon joint angles',
                    phase_samples=samples, movement_count=len(rows), excluded_recordings=list(exclude),
                    preprocessing=dict(segmentation='end_i:start_(i+1), stop exclusive; time starts at zero',
                                       interpolation='linear in normalized recorded time, endpoints included',
                                       smoothing=None, scaling=None, near_zero_clamping=False,
                                       endpoint_baseline='h(phi) = q_start + phi * (q_goal - q_start)'),
                    arrays=dict(native='native.npz', phase='phase.npz'), metadata='movements.csv',
                    sources=sources, code_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__), Path(preprocessing.__file__), Path(data_loading.__file__))})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(rows)} movements, {len(joint_names)} joints, {samples} phase samples to {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path, default=ROOT.parent / 'ReachGrasp')
    parser.add_argument('--output', type=Path, default=ROOT / 'collections/ReachGrasp/vicon-phase200-v1')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--exclude', action='append', default=[], help='Explicit subject/task exclusion.')
    args = parser.parse_args()
    prepare(args.dataset_path, args.output, args.samples, args.exclude)
