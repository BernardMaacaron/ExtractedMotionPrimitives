"""Native trajectories, phase interpolation, endpoint decomposition, and collection loading."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def phase_normalize(time, q, samples=200):
    time, q = np.asarray(time, dtype=float), np.asarray(q, dtype=float)
    if time.ndim != 1 or q.ndim != 2 or len(time) != len(q) or len(time) < 2:
        raise ValueError('Expected time (T,) and q (T,D), with T >= 2.')
    if not np.isfinite(time).all() or not np.isfinite(q).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Trajectories must be finite and time strictly increasing.')
    if samples < 2:
        raise ValueError('At least two phase samples are required.')
    phase = (time - time[0]) / (time[-1] - time[0])
    grid = np.linspace(0, 1, samples)
    return np.column_stack([np.interp(grid, phase, joint) for joint in q.T])


def endpoint_decomposition(q):
    """Return h and f for q (..., time, joint) = h + f, with f zero at both endpoints."""
    phase = np.linspace(0, 1, q.shape[-2])[:, None]
    h = q[..., :1, :] + phase * (q[..., -1:, :] - q[..., :1, :])
    return h, q - h


def load_collections(paths, joint_names=None, view='all'):
    """Load and combine compatible prepared collections.

    view='successful' filters the success column if present (as in 3D-ARM-Gaze).
    Collections without success labels (e.g. ReachGrasp) retain all trials.
    No views.npz file is required.
    Canonical trial IDs are returned as trial_id while movement_id is retained
    as a compatibility alias.
    """
    if view not in (None, 'all', 'successful'):
        raise ValueError("view must be 'all' or 'successful'.")
    targets, rows, manifests = [], [], []

    for path in map(Path, paths):
        manifest = json.loads((path / 'manifest.json').read_text())
        if joint_names is None:
            joint_names = manifest['joint_names']
        indices = [manifest['joint_names'].index(name) for name in joint_names]

        if manifests:
            for key in ('units', 'coordinates', 'phase_samples', 'preprocessing'):
                if manifest[key] != manifests[0][key]:
                    raise ValueError(f'Incompatible {key}: {path}')

        metadata = pd.read_csv(path / 'movements.csv')
        if 'trial_id' not in metadata.columns:
            if 'movement_id' not in metadata.columns:
                raise ValueError(f'Metadata contains neither trial_id nor movement_id: {path}')
            metadata['trial_id'] = metadata['movement_id']
        if 'movement_id' not in metadata.columns:
            metadata['movement_id'] = metadata['trial_id']

        with np.load(path / 'phase.npz') as data:
            id_key = 'trial_ids' if 'trial_ids' in data.files else 'movement_ids'
            stored_ids = data[id_key].astype(str).tolist()
            if metadata.trial_id.astype(str).tolist() != stored_ids:
                raise ValueError(f'Trial order differs between metadata and arrays: {path}')
            q = data['q'][..., indices]

        if view == 'successful' and 'success' in metadata.columns:
            selected = metadata['success'].fillna(False).to_numpy(dtype=bool)
            metadata = metadata.loc[selected].reset_index(drop=True)
            q = q[selected]

        metadata['collection'] = str(path.resolve())
        rows.append(metadata)
        targets.append(q)
        manifests.append(manifest)

    metadata = pd.concat(rows, ignore_index=True)
    if metadata.trial_id.duplicated().any():
        raise ValueError('A trial occurs in more than one input collection.')

    q = np.concatenate(targets)
    h, f = endpoint_decomposition(q)
    return dict(
        q=q,
        h=h,
        f=f,
        q_start=q[:, 0],
        q_end=q[:, -1],
        metadata=metadata,
        joint_names=list(joint_names),
        manifests=manifests,
    )
