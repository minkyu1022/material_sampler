import numpy as np
import pytest

from janus_reproduce.wham import umbrella_weights


def test_wham_recovers_uniform_target_from_harmonic_windows():
    rng = np.random.default_rng(4)
    grid = np.linspace(1.0, np.sqrt(2), 501)
    centers = np.linspace(grid[0], grid[-1], 7)
    strength = 500.0
    samples = []
    count = 4_000
    for center in centers:
        probability = np.exp(-0.5 * strength * (grid - center) ** 2)
        probability /= probability.sum()
        samples.append(rng.choice(grid, count, p=probability))
    coordinate = np.concatenate(samples)
    result = umbrella_weights(coordinate, centers, strength, np.full(len(centers), count))
    weighted_mean = np.sum(result["weights"] * coordinate)
    assert weighted_mean == pytest.approx((1 + np.sqrt(2)) / 2, abs=0.004)
    assert result["iterations"] < 10_000
    assert result["ess"] > 0.2 * len(coordinate)
