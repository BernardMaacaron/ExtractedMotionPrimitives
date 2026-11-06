"""Reproducible whole-trial dataset splitting."""

import numpy as np


def train_validation_test_split(metadata, *, seed=42, validation_size=0.2, test_size=0.2,
                                stratify_columns=None):
    """Return one discovery/validation/test label per trial using scikit-learn."""
    from sklearn.model_selection import train_test_split

    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError('validation_size and test_size must be > 0 and sum to < 1.')

    indices = np.arange(len(metadata))
    if stratify_columns is None:
        candidates = [c for c in ("dataset", "action", "condition") if c in metadata.columns]
        stratify_columns = [c for c in candidates if metadata[c].nunique(dropna=False) > 1]

    labels = None
    if stratify_columns:
        labels = metadata[list(stratify_columns)].fillna("<none>").astype(str).agg("|".join, axis=1)
        if labels.value_counts().min() < 2:
            labels = None

    discovery_validation, test = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True, stratify=labels)

    remaining_labels = labels.iloc[discovery_validation] if labels is not None else None
    validation_fraction = validation_size / (1.0 - test_size)
    discovery, validation = train_test_split(
        discovery_validation, test_size=validation_fraction, random_state=seed + 1,
        shuffle=True, stratify=remaining_labels)

    split = np.full(len(metadata), "discovery", dtype="<U10")
    split[validation] = "validation"
    split[test] = "test"
    return split
