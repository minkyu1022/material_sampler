import numpy as np

from janus_reproduce.free_energy import (
    binodal,
    importance_weighted_bar,
    normalized_path_weights,
)
from janus_reproduce.observables import (
    atomic_volume,
    composition,
    mean_displacement,
    partial_rdf,
    warren_cowley_sro,
)


def test_basic_alloy_observables_and_periodicity():
    assert composition(["A", "A", "B"]) == {"A": 2 / 3, "B": 1 / 3}
    assert np.isclose(atomic_volume([2, 2, 2], 4), 2)
    assert np.isclose(mean_displacement([[1.9, 0, 0]], [[0.1, 0, 0]], [2, 2, 2]), 0.2)


def test_periodic_partial_rdf_and_warren_cowley_order():
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
    species = np.array([0, 1, 0, 1])
    radii, rdf = partial_rdf(positions, species, [4, 4, 4], 1.5, bins=3)
    assert radii.shape == (3,)
    assert rdf[(0, 1)][2] > 0 and rdf[(0, 0)][2] == 0
    elements, alpha = warren_cowley_sro(positions, species, [4, 4, 4], 1.1)
    assert np.array_equal(elements, [0, 1])
    assert np.allclose(np.diag(alpha), 1.0)
    assert np.allclose(alpha[[0, 1], [1, 0]], -1.0)


def test_weighted_bar_path_weights_and_binodal():
    weights, ess = normalized_path_weights(np.log([1, 1, 2]))
    assert np.allclose(weights, [0.25, 0.25, 0.5]) and np.isclose(ess, 8 / 3)
    assert np.isclose(importance_weighted_bar(np.ones(4) * 2, np.ones(3) * 2), 2)
    x = np.linspace(0, 1, 5)
    assert binodal(x, [0, 1, 2, 1, 0]) == (0.0, 1.0)
