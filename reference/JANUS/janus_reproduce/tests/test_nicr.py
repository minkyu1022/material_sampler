import numpy as np
import pytest
import torch

from janus_reproduce.alloy_model import AlloyPaiNN
from janus_reproduce.cuni_train import _euler_maruyama
from janus_reproduce.free_energy import canonical_ladder_bar
from janus_reproduce.nicr import (
    NICR_LATTICES,
    build_nicr,
    provisional_constrained_reveal,
    sample_fixed_conditions,
    substitution_energies,
)
from janus_reproduce.torch_eam import TorchEAM

NICR_POTENTIAL = "potentials/ni_co_cr/Ni-Co-Cr_v1.eam.fs"


@pytest.mark.parametrize("phase,n_atoms", [("fcc", 108), ("bcc", 128)])
def test_nicr_cells_have_paper_size_and_exact_composition(phase, n_atoms):
    atoms = build_nicr(phase, n_atoms // 3, lattice_constant=3.55 if phase == "fcc" else 2.88)
    assert len(atoms) == n_atoms
    assert atoms.get_chemical_symbols().count("Cr") == n_atoms // 3


@pytest.mark.parametrize("phase", ["fcc", "bcc"])
def test_provisional_conditions_cover_valid_fixed_composition_rungs(phase):
    generator = torch.Generator().manual_seed(7)
    temperature, n_cr = sample_fixed_conditions(2_000, phase, torch.device("cpu"), generator=generator)
    assert temperature.min() >= 600
    assert temperature.max() <= 1500
    assert n_cr.min() >= 0
    assert n_cr.max() <= NICR_LATTICES[phase].n_atoms


def test_canonical_bar_recovers_noninteracting_combinatorial_edge():
    n_atoms, n_cr = 12, 3
    result = canonical_ladder_bar(
        np.zeros((40, n_atoms - n_cr)),
        np.zeros((60, n_cr + 1)),
        beta=2.0,
        n_atoms=n_atoms,
        n_cr=n_cr,
    )
    expected = -np.log((n_atoms - n_cr) / (n_cr + 1))
    assert result["delta_beta_g"] == pytest.approx(expected)
    assert result["forward_one_sided"] == pytest.approx(expected)
    assert result["reverse_one_sided"] == pytest.approx(expected)


def test_canonical_bar_rejects_wrong_eligible_site_counts():
    with pytest.raises(ValueError, match="eligible-site"):
        canonical_ladder_bar(np.zeros((2, 3)), np.zeros((2, 4)), 1.0, 10, 3)


def test_fixed_composition_substitution_energies_feed_canonical_bar():
    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    samples = []
    for n_cr in (3, 4):
        atoms = build_nicr("fcc", n_cr, lattice_constant=3.55, seed=2)
        samples.append((
            torch.tensor(np.asarray(atoms.numbers == 24), dtype=torch.long)[None],
            torch.tensor(atoms.get_scaled_positions(), dtype=torch.float64)[None],
            torch.tensor([np.log(atoms.get_volume())], dtype=torch.float64),
        ))
    forward, _ = substitution_energies(oracle, *samples[0])
    _, reverse = substitution_energies(oracle, *samples[1])
    result = canonical_ladder_bar(forward, reverse, 1.0, 108, 3)
    assert np.isfinite(result["delta_beta_g"])


def test_provisional_reveal_terminates_at_exact_requested_composition():
    species = torch.full((4, 12), 2)
    logits = torch.randn(4, 12, 2, generator=torch.Generator().manual_seed(3))
    target = torch.tensor([0, 3, 7, 12])
    revealed, log_probability = provisional_constrained_reveal(
        species, logits, target, 1.0, generator=torch.Generator().manual_seed(4)
    )
    assert revealed.ne(2).all()
    torch.testing.assert_close(revealed.eq(1).sum(1), target)
    assert torch.isfinite(log_probability).all()
    assert log_probability[[0, 3]].eq(0).all()


def test_provisional_reveal_preserves_capacity_across_partial_steps():
    generator = torch.Generator().manual_seed(5)
    species = torch.full((2, 20), 2)
    logits = torch.zeros(2, 20, 2)
    target = torch.tensor([2, 18])
    for probability in (0.2, 0.3, 1.0):
        species, _ = provisional_constrained_reveal(
            species, logits, target, probability, generator=generator
        )
    torch.testing.assert_close(species.eq(1).sum(1), target)


@pytest.mark.parametrize("phase", ["fcc", "bcc"])
def test_shared_alloy_network_accepts_nicr_fixed_composition_condition(phase):
    spec = NICR_LATTICES[phase]
    model = AlloyPaiNN(
        features=8,
        layers=1,
        radial_basis=4,
        cutoff=spec.graph_cutoff,
        temperature_reference=1050,
        temperature_min=600,
        temperature_max=1500,
        condition_intercept=0,
        condition_slope=0,
        condition_scale=1,
    )
    reference = torch.tensor(
        build_nicr(phase, 0, lattice_constant=3.5 if phase == "fcc" else 2.8).get_scaled_positions(),
        dtype=torch.float32,
    )
    output = model(
        torch.full((1, spec.n_atoms), 2),
        torch.zeros(1, spec.n_atoms, 3),
        torch.tensor([np.log(11 * spec.n_atoms)]),
        reference,
        torch.tensor([0.5]),
        torch.tensor([1050.0]),
        torch.tensor([0.4]),
    )
    assert output.species_logits.shape == (1, spec.n_atoms, 2)


@pytest.mark.parametrize("phase,target", [("fcc", 37), ("bcc", 91)])
def test_shared_model_provisional_exact_n_rollout_reaches_finite_nicr_oracle(phase, target):
    torch.manual_seed(11)
    spec = NICR_LATTICES[phase]
    atoms = build_nicr(phase, 0, lattice_constant=3.5 if phase == "fcc" else 2.8)
    reference = torch.tensor(atoms.get_scaled_positions(), dtype=torch.float32)
    model = AlloyPaiNN(
        features=8, layers=1, radial_basis=4, cutoff=spec.graph_cutoff,
        temperature_reference=1050, temperature_min=600, temperature_max=1500,
        condition_intercept=0, condition_slope=0, condition_scale=1,
    )
    species = torch.full((1, spec.n_atoms), 2)
    displacement = 0.004 * torch.randn(1, spec.n_atoms, 3)
    displacement -= displacement.mean(1, keepdim=True)
    log_volume = torch.tensor([np.log(11.0 * spec.n_atoms)], dtype=torch.float32)
    temperature = torch.tensor([1050.0])
    composition = torch.tensor([target / spec.n_atoms])
    generator = torch.Generator().manual_seed(12)
    for step in range(2):
        output = model(
            species, displacement, log_volume, reference, step / 2,
            temperature, composition,
        )
        displacement = _euler_maruyama(
            displacement, output.b_u, output.s_u, torch.tensor([0.02]), 0.5,
            torch.randn(displacement.shape, generator=generator),
        )
        displacement -= displacement.mean(1, keepdim=True)
        log_volume = _euler_maruyama(
            log_volume, output.b_v, output.s_v, torch.tensor([0.02]), 0.5,
            torch.randn(log_volume.shape, generator=generator),
        )
        species, _ = provisional_constrained_reveal(
            species, output.species_logits, torch.tensor([target]),
            1.0 if step == 1 else 0.5, generator=generator,
        )
    assert species.eq(1).sum() == target
    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    labels = oracle.labels(species, reference[None].double() + displacement.double(), log_volume.double())
    assert torch.isfinite(labels.energy).all()
    assert torch.isfinite(labels.forces).all()
