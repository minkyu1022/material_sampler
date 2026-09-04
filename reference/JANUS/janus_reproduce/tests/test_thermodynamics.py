import numpy as np
import pytest

from janus_reproduce.thermodynamics import (
    bar_delta,
    effective_sample_size,
    mixing_free_energy,
    mixing_free_energy_from_log_partitions,
    npt_log_density,
)


def test_npt_jacobian_and_ess():
    assert npt_log_density(2.0, 1.0, beta=3.0, n_atoms=4) == -2.0
    assert np.isclose(effective_sample_size(np.zeros(5)), 5.0)


def test_bar_and_linear_mixing_reference():
    assert abs(bar_delta(np.zeros(20), np.zeros(20))) < 1e-10
    assert np.allclose(mixing_free_energy([0, 1, 2], 2), 0)
    x, g_mix = mixing_free_energy_from_log_partitions(
        [0, 2, 4], [10, 14, 18], temperature=1000, n_atoms=4
    )
    assert np.allclose(x, [0, 0.5, 1])
    assert np.allclose(g_mix, 0)
    _, curved = mixing_free_energy_from_log_partitions(
        [0, 1, 2], [0, 2, 0], temperature=1000, n_atoms=2
    )
    assert np.isclose(curved[1], -8.617333262145e-2)
    with pytest.raises(ValueError, match="increasing fixed compositions"):
        mixing_free_energy_from_log_partitions([0, 2], [0, np.inf], 1000, 2)
