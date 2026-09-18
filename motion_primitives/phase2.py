"""Minimal Phase-2 utilities: reproducible trial splits, FADA fits, and SCA fits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_split(metadata, *, seed=42, validation_size=0.2, test_size=0.2,
               stratify_columns=None):
    """Return one discovery/validation/test label per whole trial.

    The split is made once and can be reused for every representation and method.
    By default, stratify on dataset/action/condition columns that actually vary.
    """
    from sklearn.model_selection import train_test_split

    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be >0 and sum to <1.")

    n = len(metadata)
    indices = np.arange(n)
    if stratify_columns is None:
        candidates = [c for c in ("dataset", "action", "condition") if c in metadata]
        stratify_columns = [c for c in candidates if metadata[c].nunique(dropna=False) > 1]

    labels = None
    if stratify_columns:
        labels = metadata[list(stratify_columns)].fillna("<none>").astype(str).agg("|".join, axis=1)
        if labels.value_counts().min() < 2:
            labels = None

    discovery_validation, test = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True, stratify=labels)

    remaining_labels = labels.iloc[discovery_validation] if labels is not None else None
    relative_validation = validation_size / (1.0 - test_size)
    discovery, validation = train_test_split(
        discovery_validation, test_size=relative_validation, random_state=seed + 1,
        shuffle=True, stratify=remaining_labels)

    split = np.full(n, "discovery", dtype="<U10")
    split[validation] = "validation"
    split[test] = "test"
    return split


def nmse(actual, predicted):
    """Variance-normalized squared reconstruction error."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    denominator = np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2)
    return float(np.sum((actual - predicted) ** 2) / denominator)


def _device(backend, device):
    if backend != "torch":
        return "cpu"
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _spectra_to_time(spectrum, *, n_time, pad, samples, mean):
    reconstructed = np.fft.irfft(spectrum, n=n_time, axis=-1)
    reconstructed = reconstructed[..., pad:pad + samples].transpose(0, 2, 1)
    return reconstructed + mean[None, None, :]


def fit_fada(values, split, *, model="spatiotemporal", n_components=4,
             n_spatial=None, pad=None, energy=0.995, max_delay=20,
             backend="torch", device="auto", batch_size=512,
             iterations=20, delay_steps=40, restarts=2, seed=42):
    """Fit one FADA model on discovery trials and project all held-out trials.

    model:
      - "temporal": temporal primitives shared across trial/joint signals
      - "spatiotemporal": one multijoint time-course per primitive
      - "space_by_time": separable temporal and spatial primitive libraries
    """
    from .fada import FADA, prepare_spectra

    values = np.asarray(values, dtype=float)
    split = np.asarray(split)
    discovery = split == "discovery"
    if values.ndim != 3 or len(values) != len(split):
        raise ValueError("Expected values (trial, phase, joint) and one split label per trial.")
    if not discovery.any():
        raise ValueError("No discovery trials.")

    samples = values.shape[1]
    if pad is None:
        pad = samples // 2
    spectra, mean, cumulative_energy = prepare_spectra(values, discovery, pad, energy=energy)
    n_time = samples + 2 * pad
    k = spectra.shape[-1] - 1

    fitter = FADA(
        n_time=n_time, k=k, max_delay=max_delay, iterations=iterations,
        delay_steps=delay_steps, backend=backend, device=_device(backend, device),
        batch_size=batch_size)

    if model == "temporal":
        train = spectra[discovery].reshape(-1, 1, k + 1)
        result = fitter.fit(train, (n_components,), restarts=restarts, seed=seed)
        coefficients, delays, fitted = fitter.project(
            spectra.reshape(-1, 1, k + 1), result["library"])
        fitted = fitted.reshape(spectra.shape)
        coefficients = coefficients.reshape(len(values), values.shape[2], -1)
        delays = delays.reshape(len(values), values.shape[2], -1)
    elif model == "spatiotemporal":
        result = fitter.fit(spectra[discovery], (n_components,), restarts=restarts, seed=seed)
        coefficients, delays, fitted = fitter.project(spectra, result["library"])
    elif model == "space_by_time":
        n_spatial = values.shape[2] if n_spatial is None else n_spatial
        result = fitter.fit(spectra[discovery], (n_components, n_spatial),
                            restarts=restarts, seed=seed)
        coefficients, delays, fitted = fitter.project(spectra, result["library"])
    else:
        raise ValueError("model must be temporal, spatiotemporal, or space_by_time")

    reconstruction = _spectra_to_time(
        fitted, n_time=n_time, pad=pad, samples=samples, mean=mean)

    metrics = []
    for name in ("discovery", "validation", "test"):
        mask = split == name
        metrics.append(dict(
            split=name,
            trials=int(mask.sum()),
            nmse=nmse(values[mask], reconstruction[mask]),
            variance_explained=1.0 - nmse(values[mask], reconstruction[mask]),
        ))

    return dict(
        model=model,
        library=result["library"],
        coefficients=coefficients,
        delays=delays,
        reconstruction=reconstruction,
        metrics=pd.DataFrame(metrics),
        mean=mean,
        fourier_k=k,
        cumulative_energy=cumulative_energy,
        pad=pad,
        objective=result["objective"],
        history=result["history"],
        device=fitter.device,
    )


def fit_sca(values, split, *, n_components=4, lam_sparse=None,
            n_epochs=3000, seed=42, **kwargs):
    """Fit the official glaserlab SCA model on discovery samples only."""
    from sca.models import SCA

    values = np.asarray(values, dtype=float)
    split = np.asarray(split)
    discovery = split == "discovery"
    if values.ndim != 3 or len(values) != len(split):
        raise ValueError("Expected values (trial, phase, joint) and one split label per trial.")

    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    model = SCA(n_components=n_components, lam_sparse=lam_sparse,
                n_epochs=n_epochs, **kwargs)
    train = values[discovery].reshape(-1, values.shape[-1])
    model.fit(train)

    flat = values.reshape(-1, values.shape[-1])
    latent = model.transform(flat).reshape(len(values), values.shape[1], n_components)
    reconstruction = model.reconstruct(flat).reshape(values.shape)

    metrics = []
    for name in ("discovery", "validation", "test"):
        mask = split == name
        score = nmse(values[mask], reconstruction[mask])
        metrics.append(dict(split=name, trials=int(mask.sum()), nmse=score,
                            variance_explained=1.0 - score))

    return dict(model=model, latent=latent, reconstruction=reconstruction,
                metrics=pd.DataFrame(metrics))
