"""Fourier anechoic demixing: ST motifs or independently combined temporal/spatial factors.

Arrays use [trial, joint, frequency]; delays are in samples. SciPy uses variable
projection, Torch alternates batched coefficient solves and bounded delay updates.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
import torch


def variance_captured(actual, fitted):
    return float(1 - np.sum((actual - fitted) ** 2) /
                 np.sum((actual - actual.mean(axis=(0, 1), keepdims=True)) ** 2))


def split_repetitions(metadata, seed=None):
    """Split whole repetitions within actions; keep every row of a repetition together."""
    identity = ['dataset', 'subject', 'action', 'repetition']
    keys = metadata[identity].drop_duplicates().copy()
    if seed is None:
        position = keys.groupby(['dataset', 'action']).cumcount()
        short = keys.groupby(['dataset', 'action']).action.transform('size') < 5
        keys['split'] = np.select(
            ((short & position.eq(0)) | (~short & position.mod(5).lt(3)),
             (short & position.eq(1)) | (~short & position.mod(5).eq(3))),
            ('discovery', 'validation'), default='test')
        return metadata.merge(keys, on=identity, validate='many_to_one')['split']
    rng = np.random.default_rng(seed)
    keys['split'] = ''
    for _, group in keys.groupby(['dataset', 'action'], sort=True):
        indices = rng.permutation(group.index)
        n = len(indices)
        if n < 3:
            raise ValueError('Each action needs at least three repetitions.')
        n_validation = max(1, n // 5)
        keys.loc[indices[:n_validation], 'split'] = 'validation'
        keys.loc[indices[n_validation:2 * n_validation], 'split'] = 'test'
        keys.loc[indices[2 * n_validation:], 'split'] = 'discovery'
    return metadata.merge(keys, on=identity, validate='many_to_one')['split']


def prepare_spectra(target, discovery, pad, energy=0.995, min_k=4, max_k=80):
    mean = target[discovery].mean(axis=(0, 1))
    padded = np.pad((target - mean).transpose(0, 2, 1), ((0, 0), (0, 0), (pad, pad)))
    spectrum = np.fft.rfft(padded, axis=-1)
    weights = np.full(spectrum.shape[-1], 2.)
    weights[0] = 1
    if padded.shape[-1] % 2 == 0:
        weights[-1] = 1
    spectral_energy = (np.abs(spectrum[discovery]) ** 2).sum(axis=(0, 1)) * weights
    cumulative = np.cumsum(spectral_energy) / spectral_energy.sum()
    k = int(np.clip(np.searchsorted(cumulative, energy), min_k, min(max_k, (padded.shape[-1] - 1) // 2)))
    return spectrum[..., :k + 1], mean, cumulative


def pair_basis(library):
    if 'primitive_hat' in library:
        return library['primitive_hat']
    temporal, spatial = library['temporal_hat'], library['spatial']
    return (temporal[:, None, None, :] * spatial[None, :, :, None]).reshape(
        len(temporal) * len(spatial), spatial.shape[1], temporal.shape[1])


def reconstruct(library, coefficients, delays, omega):
    activation = coefficients[..., None] * np.exp(-1j * delays[..., None] * omega)
    return np.einsum('lpf,pmf->lmf', activation, pair_basis(library))


def real_lstsq(design, target):
    return np.linalg.lstsq(np.concatenate((design.real, design.imag), axis=-2),
                          np.concatenate((target.real, target.imag), axis=-2), rcond=None)[0]


@dataclass
class FADA:
    n_time: int
    k: int
    max_delay: float
    iterations: int = 20
    delay_steps: int = 40
    tolerance: float = 1e-6
    workers: int = 1
    backend: str = 'scipy'
    device: str = 'cpu'
    delay_lr: float = 0.5  # Samples per Adam step; independent of padded signal length.
    batch_size: int = 512

    def __post_init__(self):
        if not 0 <= self.k < (self.n_time + 1) // 2:
            raise ValueError('Retain frequencies below Nyquist for continuous real-valued shifts.')
        if self.backend not in ('scipy', 'torch'):
            raise ValueError(self.backend)
        self.omega = 2 * np.pi * np.arange(self.k + 1) / self.n_time
        self.weights = np.full(self.k + 1, np.sqrt(2.))
        self.weights[0] = 1

    def project_numpy(self, target, library, delays):
        basis = pair_basis(library)
        weights = np.tile(self.weights, target.shape[1])

        def fit_trial(arguments):
            x, delay = arguments
            y = x.reshape(-1, 1) * weights[:, None]

            def design_at(tau):
                shifted = basis * np.exp(-1j * tau[:, None, None] * self.omega)
                return shifted.reshape(len(basis), -1).T * weights[:, None]

            def residual(tau):
                design = design_at(tau)
                error = design @ real_lstsq(design, y) - y
                return np.concatenate((error.real.ravel(), error.imag.ravel()))

            if self.max_delay:
                delay = least_squares(residual, delay, bounds=(-self.max_delay, self.max_delay),
                                      max_nfev=self.delay_steps).x
            else:
                delay = np.zeros_like(delay)
            return real_lstsq(design_at(delay), y).ravel(), delay

        if self.workers == 1:
            results = list(map(fit_trial, zip(target, delays)))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(fit_trial, zip(target, delays)))
        coefficients, delays = map(np.stack, zip(*results))
        return coefficients, delays

    def update_numpy(self, target, library, coefficients, delays):
        activation = coefficients[..., None] * np.exp(-1j * delays[..., None] * self.omega)
        if 'primitive_hat' in library:
            # One solve per frequency, all joints as simultaneous right-hand sides.
            primitive = np.stack([np.linalg.lstsq(activation[..., f], target[..., f], rcond=None)[0]
                                  for f in range(self.k + 1)], axis=-1)
            primitive[..., 0] = primitive[..., 0].real
            return {'primitive_hat': primitive}
        nt, ns = len(library['temporal_hat']), len(library['spatial'])
        activation = activation.reshape(len(target), nt, ns, self.k + 1)
        design = np.einsum('lpqf,pf->lfq', activation, library['temporal_hat']) * self.weights[None, :, None]
        y = target.transpose(0, 2, 1) * self.weights[None, :, None]
        spatial = real_lstsq(design.reshape(-1, ns), y.reshape(-1, target.shape[1]))
        design = np.einsum('lpqf,qm->flmp', activation, spatial).reshape(self.k + 1, -1, nt)
        y = target.transpose(2, 0, 1).reshape(self.k + 1, -1, 1)
        temporal = np.stack([np.linalg.lstsq(a, b, rcond=None)[0][:, 0]
                             for a, b in zip(design, y)], axis=-1)
        temporal[:, 0] = temporal[:, 0].real
        return {'temporal_hat': temporal, 'spatial': spatial}

    def project_torch(self, target, library, delays):
        """Project trials in bounded batches; each trial keeps independent coefficients/delays."""
        omega = torch.as_tensor(self.omega, device=self.device, dtype=target.real.dtype)
        weights = torch.as_tensor(self.weights, device=self.device, dtype=target.real.dtype)
        basis = pair_basis(library)
        coefficients_out, delays_out = [], []

        for start in range(0, len(target), self.batch_size):
            stop = min(start + self.batch_size, len(target))
            batch = target[start:stop]
            tau = delays[start:stop].detach().clone().requires_grad_(bool(self.max_delay))
            y = (batch * weights).flatten(1).unsqueeze(-1)
            real_y = torch.cat((y.real, y.imag), dim=1)

            def design_at(current_tau):
                shifted = basis[None] * torch.exp(-1j * current_tau[:, :, None, None] * omega)
                design = (shifted * weights).flatten(2).transpose(1, 2)
                return torch.cat((design.real, design.imag), dim=1)

            if self.max_delay:
                optimizer = torch.optim.Adam([tau], lr=self.delay_lr)
                best_loss = torch.full((len(batch),), float('inf'), device=self.device,
                                       dtype=batch.real.dtype)
                best_tau = tau.detach().clone()
                for _ in range(self.delay_steps):
                    optimizer.zero_grad()
                    design = design_at(tau)
                    with torch.no_grad():
                        coeff = torch.linalg.pinv(design.detach()) @ real_y
                    loss = ((design @ coeff - real_y) ** 2).mean(dim=(1, 2))
                    improved = loss.detach() < best_loss
                    best_loss = torch.minimum(best_loss, loss.detach())
                    best_tau[improved] = tau.detach()[improved]
                    loss.sum().backward()
                    optimizer.step()
                    with torch.no_grad():
                        tau.clamp_(-self.max_delay, self.max_delay)
                with torch.no_grad():
                    design = design_at(tau)
                    coeff = torch.linalg.pinv(design) @ real_y
                    improved = ((design @ coeff - real_y) ** 2).mean(dim=(1, 2)) < best_loss
                    best_tau[improved] = tau.detach()[improved]
                tau = best_tau
            else:
                tau = torch.zeros_like(tau)

            with torch.no_grad():
                coeff = torch.linalg.pinv(design_at(tau)) @ real_y
            coefficients_out.append(coeff[..., 0])
            delays_out.append(tau.detach())

        return torch.cat(coefficients_out), torch.cat(delays_out)

    def update_torch(self, target, library, coefficients, delays):
        omega = torch.as_tensor(self.omega, device=self.device)
        weights = torch.as_tensor(self.weights, device=self.device)
        activation = coefficients[..., None] * torch.exp(-1j * delays[..., None] * omega)
        if 'primitive_hat' in library:
            primitive = (torch.linalg.pinv(activation.permute(2, 0, 1)) @
                         target.permute(2, 0, 1)).permute(1, 2, 0)
            primitive[..., 0] = primitive[..., 0].real.clone()
            return {'primitive_hat': primitive}
        nt, ns = len(library['temporal_hat']), len(library['spatial'])
        activation = activation.reshape(len(target), nt, ns, self.k + 1)
        design = torch.einsum('lpqf,pf->lfq', activation, library['temporal_hat'])
        design = (design * weights[None, :, None]).reshape(-1, ns)
        y = (target.permute(0, 2, 1) * weights[None, :, None]).reshape(-1, target.shape[1])
        spatial = torch.linalg.pinv(torch.cat((design.real, design.imag))) @ torch.cat((y.real, y.imag))
        design = torch.einsum('lpqf,qm->flmp', activation, spatial.to(target.dtype))
        y = target.permute(2, 0, 1).reshape(self.k + 1, -1, 1)
        temporal = (torch.linalg.pinv(design.reshape(self.k + 1, -1, nt)) @ y)[..., 0].T
        temporal[:, 0] = temporal[:, 0].real.clone()
        return {'temporal_hat': temporal, 'spatial': spatial}

    def initialize(self, target, order, rng, perturb=False, warm=None):
        if warm is not None:
            old_order = (len(warm['library']['primitive_hat']),) if 'primitive_hat' in warm['library'] else (
                len(warm['library']['temporal_hat']), len(warm['library']['spatial']))
            if tuple(order) == old_order and not perturb:
                return {key: value.copy() for key, value in warm['library'].items()}, warm['delays'].copy()
        residual = target if warm is None else target - reconstruct(
            warm['library'], warm['coefficients'], warm['delays'], self.omega)
        time = np.fft.irfft(residual, n=self.n_time, axis=-1)
        if len(order) == 1:
            _, _, vh = np.linalg.svd(time.reshape(len(time), -1), full_matrices=False)
            count = order[0] - (0 if warm is None else len(warm['library']['primitive_hat']))
            primitive = vh[:count].reshape(count, target.shape[1], self.n_time)
            new = np.fft.rfft(primitive, axis=-1)[..., :self.k + 1]
            library = {'primitive_hat': new if warm is None else np.concatenate(
                (warm['library']['primitive_hat'], new))}
        else:
            _, _, vh = np.linalg.svd(time.reshape(-1, self.n_time), full_matrices=False)
            u, _, _ = np.linalg.svd(time.transpose(1, 0, 2).reshape(target.shape[1], -1), full_matrices=False)
            previous_t = 0 if warm is None else len(warm['library']['temporal_hat'])
            previous_s = 0 if warm is None else len(warm['library']['spatial'])
            temporal = np.fft.rfft(vh[:order[0] - previous_t], axis=-1)[..., :self.k + 1]
            spatial = u[:, :order[1] - previous_s].T
            library = {'temporal_hat': temporal, 'spatial': spatial}
            if warm is not None:
                library = {key: np.concatenate((warm['library'][key], value))
                           for key, value in library.items()}
        shape = (len(target), *order)
        delays = np.zeros(shape)
        if warm is not None:
            old_shape = (len(target), len(warm['library']['primitive_hat'])) if len(order) == 1 else (
                len(target), len(warm['library']['temporal_hat']), len(warm['library']['spatial']))
            delays[tuple(slice(0, n) for n in old_shape)] = warm['delays'].reshape(old_shape)
        if perturb:
            library = {key: value + 0.01 * np.std(value) * rng.standard_normal(value.shape)
                       for key, value in library.items()}
            delays = rng.uniform(-0.1 * self.max_delay, 0.1 * self.max_delay, shape)
        return library, delays.reshape(len(target), -1)

    def project(self, target, library, delays=None):
        if delays is None:
            delays = np.zeros((len(target), len(pair_basis(library))))
        if self.backend == 'scipy':
            c, tau = self.project_numpy(target, library, delays)
        else:
            dtype = torch.complex64 if str(self.device).startswith('cuda') else None
            tensors = {key: torch.as_tensor(value, device=self.device, dtype=dtype) for key, value in library.items()}
            c, tau = self.project_torch(torch.as_tensor(target, device=self.device, dtype=dtype), tensors,
                                        torch.as_tensor(delays, device=self.device, dtype=torch.float32 if dtype else None))
            c, tau = c.cpu().numpy(), tau.cpu().numpy()
        return c, tau, reconstruct(library, c, tau, self.omega)

    def fit(self, target, order, restarts=2, seed=42, warm=None, initial=None):
        if len(order) == 1 and order[0] > min(len(target), target.shape[1] * self.n_time):
            raise ValueError('ST order exceeds discovery initialization rank.')
        if len(order) == 2 and (order[1] > target.shape[1] or order[0] > min(
                len(target) * target.shape[1], self.n_time)):
            raise ValueError('SBT order exceeds temporal or spatial initialization rank.')
        best = None
        use_torch = self.backend == 'torch'
        x = (torch.as_tensor(target, device=self.device, dtype=torch.complex64)
             if use_torch and str(self.device).startswith('cuda') else
             torch.as_tensor(target, device=self.device) if use_torch else target)
        if initial is None:
            initial = self.initialize(target, order, np.random.default_rng(seed), warm=warm)
        base_library, base_delays = initial
        for restart in range(restarts):
            rng = np.random.default_rng(seed + restart)
            library = {key: value.copy() for key, value in base_library.items()}
            delays = base_delays.copy()
            if restart:
                library = {key: value + 0.01 * np.std(value) * rng.standard_normal(value.shape)
                           for key, value in library.items()}
                delays = rng.uniform(-0.1 * self.max_delay, 0.1 * self.max_delay, delays.shape)
            if use_torch:
                library = {key: torch.as_tensor(value, device=self.device, dtype=x.dtype)
                           for key, value in library.items()}
                delays = torch.as_tensor(delays.copy(), device=self.device, dtype=x.real.dtype)
            history = []
            for _ in range(self.iterations):
                if use_torch:
                    c, delays = self.project_torch(x, library, delays)
                    library = self.update_torch(x, library, c, delays)
                    omega = torch.as_tensor(self.omega, device=self.device)
                    activation = c[..., None] * torch.exp(-1j * delays[..., None] * omega)
                    fitted = torch.einsum('lpf,pmf->lmf', activation, pair_basis(library))
                    weights = torch.as_tensor(self.weights, device=self.device)
                    objective = float(((x - fitted).abs() ** 2 * weights ** 2).mean())
                else:
                    c, delays = self.project_numpy(x, library, delays)
                    library = self.update_numpy(x, library, c, delays)
                    fitted = reconstruct(library, c, delays, self.omega)
                    objective = float(np.mean(np.abs((x - fitted) * self.weights) ** 2))
                history.append(objective)
                if len(history) > 1 and abs(history[-2] - objective) <= self.tolerance * history[-2]:
                    break
            if use_torch:
                library = {key: value.cpu().numpy() for key, value in library.items()}
                delays = delays.cpu().numpy()
            c, delays, fitted = self.project(target, library, delays)
            objective = float(np.mean(np.abs((target - fitted) * self.weights) ** 2))
            result = dict(library=library, coefficients=c, delays=delays, objective=objective,
                          history=np.array(history))
            if best is None or objective < best['objective']:
                best = result
        return best


def align_primitives(reference, candidate, max_shift):
    """Hungarian alignment by absolute Pearson correlation, permitting sample shifts.

    Inputs are [primitive, ..., time]. Spatial vectors use a singleton time axis
    and max_shift=0. Circular shifts follow the padded Fourier model.
    """
    a = reference.reshape(len(reference), -1)
    a = a - a.mean(axis=1, keepdims=True)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    scores = np.full((len(reference), len(candidate)), -np.inf)
    shifts = np.zeros(scores.shape, dtype=int)
    signs = np.ones(scores.shape)
    for shift in range(-max_shift, max_shift + 1):
        b = np.roll(candidate, shift, axis=-1).reshape(len(candidate), -1)
        b = b - b.mean(axis=1, keepdims=True)
        correlation = a @ (b / np.linalg.norm(b, axis=1, keepdims=True)).T
        improved = np.abs(correlation) > scores
        scores[improved] = np.abs(correlation[improved])
        shifts[improved] = shift
        signs[improved] = np.sign(correlation[improved])
    rows, columns = linear_sum_assignment(-scores)
    return rows, columns, scores[rows, columns], shifts[rows, columns], signs[rows, columns]


def fit_fada_collection(values, split, *, model='spatiotemporal', n_components=4,
                        n_spatial=None, pad=None, energy=0.995, max_delay=20,
                        backend='torch', device='auto', batch_size=512,
                        iterations=20, delay_steps=40, restarts=2, seed=42):
    """Fit FADA on discovery trials and project the fixed library onto every trial."""
    values = np.asarray(values, dtype=float)
    split = np.asarray(split)
    discovery = split == 'discovery'
    if values.ndim != 3 or len(values) != len(split):
        raise ValueError('Expected values (trial, phase, joint) and one split label per trial.')
    if not discovery.any():
        raise ValueError('No discovery trials.')

    samples = values.shape[1]
    if pad is None:
        pad = samples // 2
    spectra, mean, cumulative_energy = prepare_spectra(values, discovery, pad, energy=energy)
    n_time = samples + 2 * pad
    k = spectra.shape[-1] - 1

    if backend == 'torch' and device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif backend != 'torch':
        device = 'cpu'

    fitter = FADA(n_time=n_time, k=k, max_delay=max_delay, iterations=iterations,
                  delay_steps=delay_steps, backend=backend, device=device,
                  batch_size=batch_size)

    if model == 'temporal':
        train = spectra[discovery].reshape(-1, 1, k + 1)
        result = fitter.fit(train, (n_components,), restarts=restarts, seed=seed)
        coefficients, delays, fitted = fitter.project(
            spectra.reshape(-1, 1, k + 1), result['library'])
        fitted = fitted.reshape(spectra.shape)
        coefficients = coefficients.reshape(len(values), values.shape[2], -1)
        delays = delays.reshape(len(values), values.shape[2], -1)
    elif model == 'spatiotemporal':
        result = fitter.fit(spectra[discovery], (n_components,), restarts=restarts, seed=seed)
        coefficients, delays, fitted = fitter.project(spectra, result['library'])
    elif model == 'space_by_time':
        n_spatial = values.shape[2] if n_spatial is None else n_spatial
        result = fitter.fit(spectra[discovery], (n_components, n_spatial),
                            restarts=restarts, seed=seed)
        coefficients, delays, fitted = fitter.project(spectra, result['library'])
    else:
        raise ValueError('model must be temporal, spatiotemporal, or space_by_time')

    reconstructed = np.fft.irfft(fitted, n=n_time, axis=-1)
    reconstruction = reconstructed[..., pad:pad + samples].transpose(0, 2, 1)
    reconstruction = reconstruction + mean[None, None, :]

    return dict(model=model, library=result['library'], coefficients=coefficients,
                delays=delays, reconstruction=reconstruction, mean=mean, fourier_k=k,
                cumulative_energy=cumulative_energy, pad=pad, objective=result['objective'],
                history=result['history'], device=fitter.device)
