"""Native trajectories, linear phase interpolation, and endpoint decomposition."""
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
    return np.column_stack([np.interp(np.linspace(0, 1, samples), phase, joint) for joint in q.T])


def endpoint_decomposition(q):
    """Return h and f for q (..., time, joint) = h + f, with f zero at both endpoints."""
    phase = np.linspace(0, 1, q.shape[-2])[:, None]
    h = q[..., :1, :] + phase * (q[..., -1:, :] - q[..., :1, :])
    return h, q - h


def load_collections(paths, joint_names=None):
    """Combine explicitly compatible collections, retaining dataset-qualified movement identities."""
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
        with np.load(path / 'phase.npz') as data:
            if metadata.movement_id.tolist() != data['movement_ids'].tolist():
                raise ValueError(f'Movement order differs between metadata and arrays: {path}')
            targets.append(data['q'][..., indices])
        metadata['collection'] = str(path.resolve())
        rows.append(metadata)
        manifests.append(manifest)
    metadata = pd.concat(rows, ignore_index=True)
    if metadata.movement_id.duplicated().any():
        raise ValueError('A movement occurs in more than one input collection.')
    q = np.concatenate(targets)
    h, f = endpoint_decomposition(q)
    return dict(q=q, h=h, f=f, metadata=metadata, joint_names=list(joint_names), manifests=manifests)
