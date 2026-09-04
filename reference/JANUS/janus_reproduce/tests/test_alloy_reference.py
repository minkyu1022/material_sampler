import numpy as np
import pytest
from ase import Atoms

from janus_reproduce.alloy_reference import (
    Replica,
    attempt_replica_exchange,
    canonical_npt_mc,
    fixed_composition_bar,
    fixed_composition_works,
    reduced_log_probability,
    semi_grand_npt_mc,
)


def toy_energy(atoms):
    numbers = atoms.get_atomic_numbers()
    return float(0.5 * np.square(atoms.positions).sum() + numbers.sum())


def test_semigrand_moves_and_log_volume_jacobian():
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.4, 0, 0]], cell=[2, 2, 2], pbc=True)
    result = semi_grand_npt_mc(
        atoms,
        toy_energy,
        beta=0.1,
        sweeps=20,
        burn_in=10,
        species=("H", "He"),
        chemical_potentials={"He": 20.0},
        adapt_interval=5,
        seed=4,
    )
    assert len(result.samples) == 20
    # Burn-in counters are reset when the fixed production kernel starts.
    assert result.stats["species"].attempts == 20
    assert any("He" in sample.get_chemical_symbols() for sample in result.samples)
    assert reduced_log_probability(atoms, 0.0, beta=0.0) == pytest.approx(
        len(atoms) * np.log(atoms.get_volume())
    )


def test_canonical_swaps_preserve_composition_and_adaptation_freezes():
    atoms = Atoms("HHe", positions=[[0, 0, 0], [0.2, 0, 0]], cell=[2, 2, 2], pbc=True)
    result = canonical_npt_mc(
        atoms,
        toy_energy,
        beta=0.2,
        sweeps=8,
        burn_in=4,
        species=("H", "He"),
        displacement_step=0,
        log_volume_step=0,
        adapt_interval=2,
        seed=3,
    )
    assert all(sorted(sample.get_chemical_symbols()) == ["H", "He"] for sample in result.samples)
    assert result.displacement_step == result.log_volume_step == 0


def test_replica_exchange_and_fixed_composition_bar_on_additive_potential():
    low = Atoms("HH", cell=[2, 2, 2], pbc=True)
    high = Atoms("HHe", cell=[2, 2, 2], pbc=True)
    left = Replica(low, toy_energy(low), beta=1.0, chemical_potentials={"He": 100.0})
    right = Replica(high, toy_energy(high), beta=1.0, chemical_potentials={"He": 0.0})
    assert attempt_replica_exchange(left, right, np.random.default_rng(0))
    assert left.atoms.get_chemical_symbols() == ["H", "He"]

    rung0 = [Atoms("HH", cell=[2, 2, 2], pbc=True) for _ in range(3)]
    rung1 = [Atoms("HHe", cell=[2, 2, 2], pbc=True) for _ in range(4)]
    forward, reverse = fixed_composition_works(rung0, rung1, toy_energy, 1.0, "H", "He")
    delta = fixed_composition_bar(forward, reverse)
    assert delta == pytest.approx(1.0 - np.log(2.0), abs=1e-10)
