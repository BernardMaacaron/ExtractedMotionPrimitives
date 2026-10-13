"""Fourier-based Anechoic Demixing Algorithm (FADA), spatiotemporal model.

Paper: Chiovetto, Salatiello, d'Avella & Giese (2022), section 2.4,
'Toward a unifying framework for the modeling and identification of motor
primitives', Frontiers in Computational Neuroscience 16:926345.

This is an independent Python implementation of the *published steps*, not a
translation of the authors' MATLAB FADA-T toolbox. Specifically:
  (1) NMF of Fourier powers (paper Eq. 10; NMF explicitly permitted), with
      one nonnegative squared mixing weight per trial and vector primitive;
  (2) nonlinear least-squares fits of Fourier *phases* while holding the
      magnitudes from (1) fixed (paper's phase step);
  (3) nonlinear least-squares fits of weights and shifts holding sources fixed
      (paper's weights/delays step);
  (4) repeat (2)-(3) until convergence.

The paper doesn't prescribe numerical solver tolerances, padding, selected
Fourier cutoff, restarts, or exact implementation of constrained Eq. (10).
Those are explicitly documented implementation choices here. 'space_by_time'
requires a separately constrained version and is deliberately unsupported.

Input spectra: [trial, joint, frequency] from rfft of *uncentered* signals.
Delays in samples. Output keys remain compatible with original fada.py.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from sklearn.decomposition import NMF

ALGORITHM = 'chiovetto2022_power_nmf_phase_nlls_weight_delay_nlls_v1'


def variance_captured(actual, fitted):
    actual = np.asarray(actual)
    fitted = np.asarray(fitted)
    return float(1. - np.sum((actual - fitted) ** 2) /
                 np.sum((actual - actual.mean(axis=(0, 1), keepdims=True)) ** 2))


def prepare_spectra(target, discovery, pad, energy=0.995, min_k=4, max_k=80):
    """Zero-pad and truncate uncentered rFFT spectra; no hidden mean removal.

    Returns a zero mean for compatibility with callers expecting three outputs.
    The frequency cutoff by energy is an implementation choice, not a FADA step.
    """
    target = np.asarray(target)
    if target.ndim != 3:
        raise ValueError('target must be [trial, time, joint]')
    discovery = np.asarray(discovery, dtype=bool)
    if not discovery.any():
        raise ValueError('discovery is empty')
    pad = int(pad)
    padded = np.pad(target.transpose(0, 2, 1), ((0, 0), (0, 0), (pad, pad)))
    spectrum = np.fft.rfft(padded, axis=-1)
    power_weights = np.full(spectrum.shape[-1], 2.)
    power_weights[0] = 1.
    if padded.shape[-1] % 2 == 0:
        power_weights[-1] = 1.
    power = (np.abs(spectrum[discovery]) ** 2 * power_weights).sum(axis=(0, 1))
    if not np.any(power):
        raise ValueError('zero-energy discovery signals')
    cumulative = np.cumsum(power) / power.sum()
    k = int(np.clip(np.searchsorted(cumulative, energy), min_k,
                    min(max_k, (padded.shape[-1] - 1) // 2)))
    return spectrum[..., :k + 1], np.zeros(target.shape[-1]), cumulative


def pair_basis(library):
    if 'primitive_hat' in library:
        return np.asarray(library['primitive_hat'])
    raise NotImplementedError('space-by-time needs its own constrained FADA-T magnitude model')


def reconstruct(library, coefficients, delays, omega):
    primitive = pair_basis(library)
    coefficients = np.asarray(coefficients)
    delays = np.asarray(delays)
    phase = np.exp(-1j * delays[..., None] * np.asarray(omega))
    return np.einsum('lpf,pmf->lmf', coefficients[..., None] * phase, primitive)


def _stack_complex(z):
    return np.concatenate((z.real.ravel(), z.imag.ravel()))


def _rfft_weights(n_time, k):
    weights = np.full(k + 1, np.sqrt(2.))
    weights[0] = 1.
    if n_time % 2 == 0 and k == n_time // 2:
        weights[-1] = 1.
    return weights


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
    delay_lr: float = 0.5     # compatibility, unused: paper does not specify Adam
    batch_size: int = 512     # compatibility, unused
    power_steps: int = 500
    phase_steps: int = 40

    def __post_init__(self):
        if self.backend != 'scipy':
            raise ValueError('Paper-consistent nonlinear least squares currently uses backend="scipy" only')
        if not 0 <= self.k < (self.n_time + 1) // 2:
            raise ValueError('Retain frequencies strictly below Nyquist')
        if self.workers < 1:
            raise ValueError('workers must be >= 1')
        self.omega = 2 * np.pi * np.arange(self.k + 1) / self.n_time
        self.weights = _rfft_weights(self.n_time, self.k)

    def _validate(self, target):
        x = np.asarray(target, dtype=np.complex128)
        if x.ndim != 3 or x.shape[-1] != self.k + 1:
            raise ValueError(f'expected [trial,joint,{self.k + 1}] spectra')
        return x

    def initialize(self, target, order, rng, perturb=False, warm=None):
        """Paper Eq. 10: NMF of squared Fourier amplitudes.

        ST constraints: a^2[l,p] is shared across joints, with independent
        primitive Fourier magnitudes |nu[p,m,k]|. No mean subtraction.
        """
        x = self._validate(target)
        if len(order) != 1:
            raise NotImplementedError('Only spatiotemporal (and flattened temporal) FADA is implemented')
        p = int(order[0])
        if not 1 <= p <= min(len(x), x.shape[1] * x.shape[2]):
            raise ValueError('invalid primitive count')
        if warm is not None:
            raise NotImplementedError('Paper amplitude estimation is not compatible with SVD warm starts')

        # |C_lmk|^2 = sum_p a_lp^2 |nu_pmk|^2, fitted as a nonnegative matrix.
        powers = np.abs(x) ** 2
        nmf = NMF(n_components=p, init='random', random_state=int(rng.integers(0, 2**31 - 1)),
                  solver='mu', beta_loss='frobenius', max_iter=self.power_steps, tol=1e-5)
        squared_coeff = nmf.fit_transform(powers.reshape(len(x), -1))
        squared_magnitude = nmf.components_.reshape(p, x.shape[1], self.k + 1)
        coeff = np.sqrt(np.maximum(squared_coeff, 1e-15))
        magnitude = np.sqrt(np.maximum(squared_magnitude, 1e-15))
        # The real-signal zero-frequency coefficient must itself be real.
        angles = rng.uniform(-np.pi, np.pi, size=magnitude.shape)
        primitive = magnitude * np.exp(1j * angles)
        primitive[..., 0] = magnitude[..., 0] * rng.choice([-1., 1.],
                                                            size=magnitude[..., 0].shape)
        delays = rng.uniform(-self.max_delay, self.max_delay,
                             size=(len(x), p)) if self.max_delay else np.zeros((len(x), p))
        library = {'primitive_hat': primitive}
        return library, delays, coeff, magnitude

    def _update_phases(self, target, magnitude, coefficients, delays, primitive):
        """Paper phase step: nonlinear LS of phase angles, magnitudes fixed."""
        x = target
        new = np.array(primitive, copy=True)
        n_trials, n_joints, _ = x.shape
        # One independent nonlinear LS per (joint, positive frequency).
        tasks = [(m, f) for m in range(n_joints) for f in range(1, self.k + 1)]

        def solve(task):
            m, f = task
            mag = magnitude[:, m, f]
            design = coefficients * np.exp(-1j * delays * self.omega[f])
            y = x[:, m, f]

            def model(theta):
                return design @ (mag * np.exp(1j * theta))

            def fun(theta):
                return _stack_complex(model(theta) - y)

            def jac(theta):
                complex_jac = design * (1j * mag * np.exp(1j * theta))[None, :]
                return np.concatenate((complex_jac.real, complex_jac.imag), axis=0)

            phase0 = np.angle(primitive[:, m, f])
            opt = least_squares(fun, phase0, jac=jac, method='lm',
                                max_nfev=self.phase_steps)
            return m, f, mag * np.exp(1j * opt.x)

        if self.workers == 1:
            fitted = map(solve, tasks)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                fitted = list(pool.map(solve, tasks))
        for m, f, value in fitted:
            new[:, m, f] = value

        # DC bin: no arbitrary complex phase is allowed (real time signals).
        # Minimize its LS cost over allowable signs +/-1 by coordinate search.
        for m in range(n_joints):
            a = coefficients
            target_dc = x[:, m, 0].real
            source = new[:, m, 0].real
            pred = a @ source
            for _ in range(4):
                any_change = False
                for p in range(len(source)):
                    changed_pred = pred - 2 * a[:, p] * source[p]
                    if np.sum((changed_pred - target_dc) ** 2) + 1e-12 < np.sum((pred - target_dc) ** 2):
                        source[p] *= -1
                        pred = changed_pred
                        any_change = True
                if not any_change:
                    break
            new[:, m, 0] = source.astype(np.complex128)
        return new

    def _fit_trial(self, target, primitive, initial_coeff, initial_delays):
        """Paper step (b): fit joint trial weight/delay variables by nonlinear LS."""
        p = len(primitive)
        weighted_target = target * self.weights
        omega = self.omega[None, None, :]

        def design(tau):
            return primitive * np.exp(-1j * tau[:, None, None] * omega)

        def fun(v):
            a, tau = v[:p], v[p:]
            err = np.einsum('p,pmf->mf', a, design(tau)) - target
            return _stack_complex(err * self.weights)

        def jac(v):
            a, tau = v[:p], v[p:]
            shifted = design(tau)
            derivative_a = shifted
            derivative_tau = (-1j * self.omega[None, None, :]) * a[:, None, None] * shifted
            j = np.concatenate((derivative_a, derivative_tau), axis=0)
            j *= self.weights[None, None, :]
            return np.concatenate((j.real.reshape(2 * p, -1).T,
                                   j.imag.reshape(2 * p, -1).T), axis=0)

        if self.max_delay == 0:
            fixed = design(np.zeros(p)).reshape(p, -1).T * np.tile(self.weights, target.shape[0])[:, None]
            yy = weighted_target.ravel()
            mat = np.concatenate((fixed.real, fixed.imag), axis=0)
            vec = np.concatenate((yy.real, yy.imag))
            return np.linalg.lstsq(mat, vec, rcond=None)[0], np.zeros(p)

        low = np.r_[np.full(p, -np.inf), np.full(p, -self.max_delay)]
        high = np.r_[np.full(p, np.inf), np.full(p, self.max_delay)]
        x0 = np.r_[initial_coeff, np.clip(initial_delays, -self.max_delay, self.max_delay)]
        result = least_squares(fun, x0, jac=jac, bounds=(low, high),
                               method='trf', max_nfev=self.delay_steps)
        return result.x[:p], result.x[p:]

    def project(self, target, library, delays=None, coefficients=None):
        """Held-out trial projection, fixing all learned primitive Fourier values."""
        x = self._validate(target)
        primitive = pair_basis(library)
        p = len(primitive)
        if delays is None:
            delays = np.zeros((len(x), p))
        if coefficients is None:
            coefficients = np.zeros((len(x), p))
        tasks = [(x[l], primitive, coefficients[l], delays[l]) for l in range(len(x))]
        if self.workers == 1:
            fit = list(map(lambda v: self._fit_trial(*v), tasks))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                fit = list(pool.map(lambda v: self._fit_trial(*v), tasks))
        coef, tau = (np.stack(v) for v in zip(*fit))
        return coef, tau, reconstruct(library, coef, tau, self.omega)

    def fit(self, target, order, restarts=2, seed=42, warm=None, initial=None,
            progress=False):
        """Published magnitude -> phase -> weights/delays steps, with restarts."""
        x = self._validate(target)
        if len(order) != 1:
            raise NotImplementedError('Space-by-time FADA-T requires additional constrained updates')
        best = None
        outer = range(restarts)
        if progress:
            from tqdm.auto import tqdm
            outer = tqdm(outer, desc='FADA restarts')
        for restart in outer:
            rng = np.random.default_rng(seed + restart)
            if initial is not None:
                library, delays, coefficients, magnitude = initial
                library = {key: np.asarray(v).copy() for key, v in library.items()}
                delays = np.asarray(delays).copy()
                coefficients = np.asarray(coefficients).copy()
                magnitude = np.asarray(magnitude).copy()
            else:
                library, delays, coefficients, magnitude = self.initialize(x, order, rng, warm=warm)
            history = []
            steps = range(self.iterations)
            if progress:
                from tqdm.auto import tqdm
                steps = tqdm(steps, leave=False, desc=f'FADA phase/weight updates (restart {restart + 1})')
            for iteration in steps:
                new_primitive = self._update_phases(
                    x, magnitude, coefficients, delays, library['primitive_hat'])
                library = {'primitive_hat': new_primitive}
                coefficients, delays, fitted = self.project(x, library, delays, coefficients)
                objective = float(np.mean(np.abs((x - fitted) * self.weights) ** 2))
                history.append(objective)
                if len(history) > 1 and abs(history[-2] - objective) <= self.tolerance * max(history[-2], 1e-15):
                    break
            objective = history[-1]
            record = dict(library=library, coefficients=coefficients, delays=delays,
                          objective=objective, history=np.array(history), algorithm=ALGORITHM,
                          magnitude_power_error=float(np.mean((np.abs(x) ** 2 -
                                 np.einsum('lp,pmf->lmf', coefficients ** 2,
                                            magnitude ** 2)) ** 2)))
            if best is None or objective < best['objective']:
                best = record
        return best


def align_primitives(reference, candidate, max_shift):
    """Match primitives for stability analysis; same utility as original fada.py."""
    a = reference.reshape(len(reference), -1)
    a = a - a.mean(axis=1, keepdims=True)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-15)
    scores = np.full((len(reference), len(candidate)), -np.inf)
    shifts = np.zeros(scores.shape, dtype=int)
    signs = np.ones(scores.shape)
    for shift in range(-max_shift, max_shift + 1):
        b = np.roll(candidate, shift, axis=-1).reshape(len(candidate), -1)
        b = b - b.mean(axis=1, keepdims=True)
        correlation = a @ (b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-15)).T
        improved = np.abs(correlation) > scores
        scores[improved] = np.abs(correlation[improved])
        shifts[improved] = shift
        signs[improved] = np.sign(correlation[improved])
    rows, columns = linear_sum_assignment(-scores)
    return rows, columns, scores[rows, columns], shifts[rows, columns], signs[rows, columns]
