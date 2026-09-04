"""Thermodynamic estimators used by the JANUS alloy experiments."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

KB_EV = 8.617333262145e-5


def npt_log_density(energy, log_volume, beta, pressure=0.0, n_atoms=0):
    """Unnormalised log density in log-volume coordinates, including V**N Jacobian."""
    return -beta * (np.asarray(energy) + pressure * np.exp(log_volume)) + n_atoms * log_volume


def effective_sample_size(log_weights) -> float:
    weights = np.exp(np.asarray(log_weights) - logsumexp(log_weights))
    return float(1.0 / np.square(weights).sum())


def bar_delta(forward_work, reverse_work) -> float:
    """Solve the equal-sample Bennett acceptance-ratio equation."""
    wf, wr = np.asarray(forward_work), np.asarray(reverse_work)

    def equation(delta):
        return expit(delta - wf).mean() - expit(wr - delta).mean()

    scale = max(100.0, np.max(np.abs(np.r_[wf, wr])) + 10.0)
    return float(brentq(equation, -scale, scale))


def composition_free_energy(deltas, temperature: float) -> np.ndarray:
    """Accumulate dimensionless BAR increments into G(n)-G(0) in eV."""
    return KB_EV * temperature * np.r_[0.0, np.cumsum(deltas)]


def mixing_free_energy(g, n_atoms: int) -> np.ndarray:
    g = np.asarray(g)
    x = np.arange(g.size) / n_atoms
    return (g - (1 - x) * g[0] - x * g[-1]) / n_atoms


def mixing_free_energy_from_log_partitions(n_cr, log_z, temperature: float, n_atoms: int):
    """Fixed-composition path-weight estimate of ``G_mix/N`` in eV/atom."""
    n_cr, log_z = np.asarray(n_cr), np.asarray(log_z, dtype=float)
    if (
        n_cr.ndim != 1
        or n_cr.shape != log_z.shape
        or n_cr.size < 2
        or n_cr[0] != 0
        or n_cr[-1] != n_atoms
        or np.any(np.diff(n_cr) <= 0)
        or not np.all(np.isfinite(log_z))
        or temperature <= 0
    ):
        raise ValueError("require increasing fixed compositions from 0 through n_atoms")
    x = n_cr / n_atoms
    g = -KB_EV * temperature * log_z
    return x, (g - (1 - x) * g[0] - x * g[-1]) / n_atoms
