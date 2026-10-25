#!/usr/bin/env python3
"""Fixed old replay/DMD controls, with corrected train80 channel scales."""
import numpy as np


def dmd_periodic(observed, times, future, mask, rank, scales):
    scales = np.asarray(scales, dtype=np.float64)
    assert scales.shape == (3,) and np.all(scales > 0)
    fields = observed.astype(np.float64) * mask[None, None]
    Y = (fields / scales[None, :, None, None]).reshape(len(fields), -1).T
    U, s, Vh = np.linalg.svd(Y[:, :-1], full_matrices=False)
    assert len(s) and s[0] > 0
    r = min(rank, int(np.count_nonzero(s > s[0] * 1e-9)))
    U, s, V = U[:, :r], s[:r], Vh[:r].T
    AV = (Y[:, 1:] @ V) / s[None]
    eig, W = np.linalg.eig(U.T @ AV)
    Phi = AV @ W
    dt = float(np.mean(np.diff(times)))
    assert np.allclose(np.diff(times), dt, atol=1e-8) and dt > 0
    angular = np.angle(eig) / dt
    relative = np.asarray(times) - times[0]
    E = np.exp(1j * relative[:, None] * angular[None])
    gram = (Phi.conj().T @ Phi) * (E.conj().T @ E)
    rhs = np.sum(E.conj().T * (Phi.conj().T @ Y), axis=1)
    coefficients = np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]
    Ef = np.exp(1j * (np.asarray(future) - times[0])[:, None] * angular[None])
    pred = (Ef @ (Phi * coefficients[None]).T).real.reshape(len(future), 3, 64, 64)
    pred = (pred * scales[None, :, None, None]).astype(np.float32)
    assert np.isfinite(pred).all()
    return pred, {'rank': r, 'angular_frequencies': angular.tolist(),
                  'raw_eigen_moduli': abs(eig).tolist(),
                  'amplitudes': 'fixed; eigenvalues projected to unit circle',
                  'scales_source': 'corrected training80 only'}


def controls(support4, base, times, mask, configuration, frequency, scales):
    N = configuration['bootstrap_steps']
    assert N == 48 and frequency > 0
    observed = np.concatenate([support4, base[:N]]) * mask[None, None]
    observed_times = times[:4 + N]
    period = 1 / frequency
    assert period <= observed_times[-1] - observed_times[0], 'generated trajectory shorter than one replay period'
    phase_times = observed_times[-1] - np.mod(observed_times[-1] - times[4 + N:164], period)
    hi = np.searchsorted(observed_times, phase_times).clip(1, len(observed_times) - 1)
    lo = hi - 1
    alpha = (phase_times - observed_times[lo]) / (observed_times[hi] - observed_times[lo])
    replay = base.copy()
    replay[N:] = (1 - alpha[:, None, None, None]) * observed[lo] + alpha[:, None, None, None] * observed[hi]
    dmd, diagnostic = dmd_periodic(observed, observed_times, times[4:164], mask, 9, scales)
    dmd[:N] = base[:N]
    assert np.array_equal(replay[:N], base[:N]) and np.array_equal(dmd[:N], base[:N])
    return {'cycle_replay': (replay, {'frequency': frequency, 'cycle': 'last generated period; linear interpolation'}),
            'self_DMD_rank9': (dmd, diagnostic)}
