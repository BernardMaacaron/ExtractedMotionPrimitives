import pandas as pd

from motion_primitives.fada import split_repetitions


def test_dataset_identity_does_not_merge_repetitions():
    metadata = pd.DataFrame([dict(dataset=dataset, subject='s', action='reach', repetition=r)
                             for dataset, order in [('A', range(6)), ('B', reversed(range(6)))]
                             for r in order])
    splits = split_repetitions(metadata)
    assert splits.iloc[0] == splits.iloc[6] == 'discovery'
    assert splits.iloc[3] == splits.iloc[9] == 'validation'
    duplicated = pd.concat([metadata, metadata], ignore_index=True)
    shuffled = split_repetitions(duplicated, seed=42)
    assert shuffled[:12].tolist() == shuffled[12:].tolist()
