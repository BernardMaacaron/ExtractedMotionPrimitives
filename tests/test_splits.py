import pandas as pd

from motion_primitives.splits import train_validation_test_split


def test_trial_split_is_reproducible_and_complete():
    metadata = pd.DataFrame([
        dict(dataset='A', subject=f's{i % 4}', action='reach', condition='c1' if i % 2 else 'c2')
        for i in range(40)
    ])
    first = train_validation_test_split(metadata, seed=42)
    second = train_validation_test_split(metadata, seed=42)
    assert first.tolist() == second.tolist()
    assert set(first) == {'discovery', 'validation', 'test'}
    assert len(first) == len(metadata)


def test_trial_split_keeps_one_label_per_row():
    metadata = pd.DataFrame([
        dict(dataset='A', subject='s1', action='reach', condition='c1'),
        dict(dataset='A', subject='s1', action='reach', condition='c1'),
        dict(dataset='A', subject='s2', action='reach', condition='c2'),
        dict(dataset='A', subject='s2', action='reach', condition='c2'),
        dict(dataset='A', subject='s3', action='reach', condition='c1'),
        dict(dataset='A', subject='s3', action='reach', condition='c1'),
        dict(dataset='A', subject='s4', action='reach', condition='c2'),
        dict(dataset='A', subject='s4', action='reach', condition='c2'),
        dict(dataset='A', subject='s5', action='reach', condition='c1'),
        dict(dataset='A', subject='s5', action='reach', condition='c1'),
    ])
    split = train_validation_test_split(metadata, seed=7)
    assert len(split) == len(metadata)
