import numpy as np
import pytest

torch = pytest.importorskip("torch")

from janus_reproduce.ising import (
    JANUSIsing,
    exact_ising_observables,
    ghost_wolff_samples,
    heat_bath_prob,
    ising_energy,
    masked_soft_ce,
    observables,
    sample_janus,
)


def test_periodic_energy_and_heat_bath_are_exact():
    spins = np.ones((4, 4), dtype=np.int8)
    assert ising_energy(spins) == -32
    assert heat_bath_prob(spins, 2.0, 0.0)[0, 0] == pytest.approx(1 / (1 + np.exp(-4)))


def test_ghost_wolff_tracks_field_direction():
    plus = ghost_wolff_samples(4, 2.5, 2.0, num_samples=300, burn_in=100, seed=3)
    minus = ghost_wolff_samples(4, 2.5, -2.0, num_samples=300, burn_in=100, seed=4)
    assert observables(plus)["magnetization"] > 0.25
    assert observables(minus)["magnetization"] < -0.25


def test_ghost_wolff_matches_exact_4x4_critical_observables():
    exact = exact_ising_observables(4, 2.2692)
    sampled = observables(
        ghost_wolff_samples(4, 2.2692, num_samples=4000, burn_in=600, chains=4, seed=5)
    )
    assert sampled["abs_magnetization"] == pytest.approx(exact["abs_magnetization"], abs=0.025)
    assert sampled["energy_per_site"] == pytest.approx(exact["energy_per_site"], abs=0.04)


def test_masked_model_loss_and_sampler():
    model = JANUSIsing(width=8, depth=2)
    terminals = torch.randint(0, 2, (4, 4, 4)).float().mul(2).sub(1)
    loss = masked_soft_ce(model, terminals, 2.0)
    loss.backward()
    assert torch.isfinite(loss)
    samples, log_prob = sample_janus(model, 3, 4, 2.0, return_log_prob=True)
    assert samples.shape == (3, 4, 4)
    assert set(samples.unique().tolist()) <= {-1.0, 1.0}
    assert log_prob.shape == (3,)
