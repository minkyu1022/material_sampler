import json
from types import SimpleNamespace

import numpy as np

from janus_reproduce.cuni import KB_EV_K
from janus_reproduce.cuni_reference_analysis import (
    integrated_autocorrelation_time,
    load_chain,
    mean_lattice_displacements,
    mixing_free_energy_from_semigrand,
    solve_semigrand_mbar,
    split_rhat,
)


def test_mean_displacement_is_invariant_to_periodic_wrapping():
    reference = np.array([[0.99, 0.0, 0.0], [0.49, 0.5, 0.5]])
    positions = np.array([[[0.01, 0.0, 0.0], [0.51, 0.5, 0.5]]])
    chain = SimpleNamespace(
        initial_fractional_positions=reference,
        fractional_positions=positions,
        volume=np.array([1000.0]),
        config_sweeps=np.array([0]),
    )
    assert np.isclose(mean_lattice_displacements(chain, np.array([0]))[0], 0.0)


def test_iat_and_split_rhat_detect_correlation_and_walker_offset():
    rng = np.random.default_rng(3)
    white = rng.normal(size=4096)
    correlated = np.convolve(white, np.ones(12) / 12, mode="same")
    assert integrated_autocorrelation_time(correlated) > integrated_autocorrelation_time(white)
    assert split_rhat(white, white.copy()) < 1.01
    assert split_rhat(white, white + 2) > 1.1


def test_loads_batched_trace_and_configuration_schema(tmp_path):
    path = tmp_path / "chain.npz"
    metadata = {
        "temperature_K": 800,
        "delta_mu_Cu_minus_Ni_eV": 0.84,
        "walker": 1,
        "n_atoms": 4,
        "production_start": 2,
    }
    np.savez(
        path,
        energy_eV=np.arange(6.0),
        n_cu=np.arange(6) % 5,
        log_volume=np.log(np.full(6, 40.0)),
        config_sweeps=np.array([0, 2, 4]),
        fractional_positions=np.zeros((3, 4, 3)),
        species=np.zeros((3, 4), np.uint8),
        metadata=np.asarray(json.dumps(metadata)),
    )
    chain = load_chain(path)
    assert chain.production_start == 2
    assert np.allclose(chain.volume, 40)
    assert len(chain.traces()["composition"]) == 4


def test_semigrand_mbar_and_legendre_fenchel_use_one_per_atom_division():
    # Identical histograms at identical states imply identical reduced free energies.
    hist = np.array([[5, 10, 5], [5, 10, 5]], float)
    f, diagnostics = solve_semigrand_mbar(hist, [0.0, 0.0], 600)
    assert np.allclose(f, 0)
    assert diagnostics["max_stationarity_error"] < 1e-11

    # Phi/N is kBT*f/N.  The LF result must therefore scale as 1/N, not 1/N^2.
    temperature, n_atoms = 600.0, 10
    mu = np.array([-1.0, 1.0])
    f = np.array([0.0, -n_atoms / (KB_EV_K * temperature)])
    x, g_mix = mixing_free_energy_from_semigrand(f, mu, temperature, n_atoms)
    assert np.array_equal(x, np.arange(11) / 10)
    assert np.isclose(g_mix[5], -0.5)
