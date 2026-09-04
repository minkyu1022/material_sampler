"""Minimal log-space MBAR/WHAM for harmonic umbrella windows."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def umbrella_weights(
    coordinate,
    centers,
    bias_strength: float,
    counts,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> dict[str, np.ndarray | int | float]:
    """Return target-ensemble weights from samples pooled across umbrella windows."""
    coordinate = np.asarray(coordinate, dtype=float)
    centers = np.asarray(centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if coordinate.ndim != 1 or centers.ndim != 1 or counts.shape != centers.shape:
        raise ValueError("coordinate must be 1-D and centers/counts must agree")
    reduced_bias = 0.5 * bias_strength * (coordinate[None] - centers[:, None]) ** 2
    free_energy = np.zeros(len(centers))
    for iteration in range(1, max_iterations + 1):
        denominator = logsumexp(
            np.log(counts)[:, None] + free_energy[:, None] - reduced_bias, axis=0
        )
        updated = -logsumexp(-reduced_bias - denominator[None], axis=1)
        updated -= updated[0]
        if np.max(np.abs(updated - free_energy)) < tolerance:
            free_energy = updated
            break
        free_energy = updated
    else:
        raise RuntimeError("WHAM did not converge")
    log_weight = -logsumexp(
        np.log(counts)[:, None] + free_energy[:, None] - reduced_bias, axis=0
    )
    log_weight -= logsumexp(log_weight)
    weight = np.exp(log_weight)
    return {
        "weights": weight,
        "free_energies": free_energy,
        "iterations": iteration,
        "ess": 1 / np.square(weight).sum(),
    }
