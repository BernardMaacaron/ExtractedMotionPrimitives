"""Check irregular native timing, exact endpoint separation, and collection identity."""
import json

import numpy as np
import pandas as pd
import pytest

from motion_primitives.preprocessing import endpoint_decomposition, load_collections, phase_normalize


def test_phase_and_endpoint_decomposition():
    time = np.array([5., 6., 9.])
    q = np.array([[0., 2.], [4., 3.], [0., 6.]])
    aligned = phase_normalize(time, q, 5)
    np.testing.assert_allclose(aligned[:, 0], [0, 4, 8 / 3, 4 / 3, 0])
    np.testing.assert_allclose(aligned[:, 1], [2, 3, 4, 5, 6])
    h, f = endpoint_decomposition(aligned[None])
    np.testing.assert_allclose(h + f, aligned[None])
    np.testing.assert_allclose(f[:, [0, -1]], 0)
    np.testing.assert_allclose(f[0, :, 1], 0)
    with pytest.raises(ValueError, match='strictly increasing'):
        phase_normalize([0, 1, 1], q)


def test_mix_compatible_collections(tmp_path):
    paths = [tmp_path / 'A', tmp_path / 'B']
    for index, path in enumerate(paths):
        path.mkdir()
        manifest = dict(joint_names=['x', 'y'], units='degrees', coordinates='same-model',
                        phase_samples=3, preprocessing={'interpolation': 'linear'})
        (path / 'manifest.json').write_text(json.dumps(manifest))
        pd.DataFrame([dict(movement_id=f'{path.name}/s/reach/rep-00', dataset=path.name)]).to_csv(
            path / 'movements.csv', index=False)
        np.savez(path / 'phase.npz', q=np.arange(6).reshape(1, 3, 2) + index,
                 movement_ids=[f'{path.name}/s/reach/rep-00'])
    mixed = load_collections(paths, joint_names=['y', 'x'])
    assert mixed['metadata'].dataset.tolist() == ['A', 'B']
    np.testing.assert_array_equal(mixed['q'][0, 0], [1, 0])
    np.testing.assert_allclose(mixed['q'], mixed['h'] + mixed['f'])
    with pytest.raises(ValueError, match='more than one'):
        load_collections([paths[0], paths[0]])
    manifest['units'] = 'radians'
    (paths[1] / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Incompatible units'):
        load_collections(paths)
