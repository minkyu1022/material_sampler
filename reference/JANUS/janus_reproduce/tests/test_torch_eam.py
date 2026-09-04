from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.eam import EAM

from janus_reproduce.cuni import CuNiEAM, build_cuni_fcc
from janus_reproduce.nicr_unified_bct2d import cell_matrix, reference_sites
from janus_reproduce.torch_eam import TorchCuNiEAM, TorchEAM

POTENTIAL = Path(__file__).parents[1] / "potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy"
NICR_POTENTIAL = Path(__file__).parents[1] / "potentials/ni_co_cr/Ni-Co-Cr_v1.eam.fs"


def _inputs(atoms, device):
    species = torch.tensor(
        [0 if symbol == "Cu" else 1 for symbol in atoms.get_chemical_symbols()], device=device
    )
    fractional = torch.tensor(
        atoms.get_scaled_positions(wrap=False), dtype=torch.float64, device=device
    )
    log_volume = torch.tensor(np.log(atoms.get_volume()), dtype=torch.float64, device=device)
    return species, fractional, log_volume


def _ase_labels(atoms):
    oracle = CuNiEAM(POTENTIAL)
    state = oracle._attach(atoms.copy())
    energy = state.get_potential_energy()
    forces = state.get_forces()
    stress = state.get_stress(voigt=False)
    derivative = state.get_volume() * np.trace(stress) / 3
    return energy, forces, stress, derivative


def _batch_atoms():
    outputs = []
    for index, (cu_fraction, lattice_constant, displacement) in enumerate(
        ((0.23, 3.51, 0.015), (0.71, 3.67, 0.035))
    ):
        atoms = build_cuni_fcc(
            108,
            cu_fraction=cu_fraction,
            lattice_constant=lattice_constant,
            seed=20 + index,
        )
        atoms.positions += np.random.default_rng(30 + index).normal(
            scale=displacement, size=(108, 3)
        )
        outputs.append(atoms)
    return outputs


def _batch_inputs(atoms, device):
    inputs = [_inputs(state, device) for state in atoms]
    return tuple(torch.stack(values) for values in zip(*inputs, strict=True))


@pytest.mark.parametrize("random_displacement", [False, True], ids=["relaxed", "random"])
def test_cpu_energy_forces_and_stress_match_ase(random_displacement):
    atoms = build_cuni_fcc(
        108,
        cu_fraction=0.37 if random_displacement else 0.0,
        lattice_constant=3.55 if random_displacement else 3.52,
        seed=7,
    )
    if random_displacement:
        atoms.positions += np.random.default_rng(11).normal(scale=0.025, size=(108, 3))
    expected = _ase_labels(atoms)
    model = TorchCuNiEAM(POTENTIAL)
    labels = model.labels(*_inputs(atoms, "cpu"))

    # Float64 uses the same not-a-knot cubics as ASE; only reduction order differs.
    assert labels.energy.item() == pytest.approx(expected[0], abs=1e-10)
    assert labels.forces.detach().numpy() == pytest.approx(expected[1], abs=2e-10)
    assert labels.stress.detach().numpy() == pytest.approx(expected[2], abs=2e-12)
    assert labels.log_volume_derivative.item() == pytest.approx(expected[3], abs=2e-10)


def test_distinct_batched_labels_match_per_configuration_and_ase():
    atoms = _batch_atoms()
    model = TorchCuNiEAM(POTENTIAL)
    batched = model.labels(*_batch_inputs(atoms, "cpu"))

    assert batched.energy.shape == (2,)
    assert batched.forces.shape == (2, 108, 3)
    assert batched.stress.shape == (2, 3, 3)
    assert batched.log_volume_derivative.shape == (2,)
    for index, state in enumerate(atoms):
        individual = model.labels(*_inputs(state, "cpu"))
        expected = _ase_labels(state)
        assert batched.energy[index].item() == pytest.approx(individual.energy.item(), abs=1e-12)
        assert batched.forces[index].detach().numpy() == pytest.approx(
            individual.forces.detach().numpy(), abs=1e-12
        )
        assert batched.stress[index].detach().numpy() == pytest.approx(
            individual.stress.detach().numpy(), abs=1e-12
        )
        assert batched.log_volume_derivative[index].item() == pytest.approx(
            individual.log_volume_derivative.item(), abs=1e-12
        )
        assert batched.energy[index].item() == pytest.approx(expected[0], abs=1e-10)
        assert batched.forces[index].detach().numpy() == pytest.approx(expected[1], abs=2e-10)
        assert batched.stress[index].detach().numpy() == pytest.approx(expected[2], abs=2e-12)
        assert batched.log_volume_derivative[index].item() == pytest.approx(expected[3], abs=2e-10)


def test_distinct_batched_all_site_substitutions_match_per_configuration_and_ase():
    atoms = _batch_atoms()
    model = TorchCuNiEAM(POTENTIAL)
    site_energies = model.all_site_energies(*_batch_inputs(atoms, "cpu"))
    assert site_energies.shape == (2, 108, 2)

    oracle = CuNiEAM(POTENTIAL)
    for batch_index, state in enumerate(atoms):
        individual = model.all_site_energies(*_inputs(state, "cpu"))
        assert site_energies[batch_index].detach().numpy() == pytest.approx(
            individual.detach().numpy(), abs=1e-12
        )
        for site in (0, 53, 107):
            expected = []
            for symbol in ("Cu", "Ni"):
                trial = state.copy()
                trial.symbols[site] = symbol
                expected.append(oracle.energy(trial))
            assert site_energies[batch_index, site].detach().numpy() == pytest.approx(
                expected, abs=1e-10
            )


@pytest.mark.parametrize("phase,lattice", [("fcc", 3.5), ("bcc", 2.85)])
@pytest.mark.parametrize("element,index", [("Ni", 0), ("Cr", 2)])
def test_finnis_sinclair_native_energy_matches_ase(phase, lattice, element, index):
    atoms = bulk(element, phase, a=lattice, cubic=True)
    atoms.calc = EAM(potential=str(NICR_POTENTIAL))
    expected = atoms.get_potential_energy()
    species = torch.full((len(atoms),), index)
    fractional = torch.tensor(atoms.get_scaled_positions(), dtype=torch.float64)
    log_volume = torch.tensor(np.log(atoms.get_volume()), dtype=torch.float64)
    assert TorchEAM(NICR_POTENTIAL)(species, fractional, log_volume).item() == pytest.approx(
        expected, abs=1e-10
    )


def test_finnis_sinclair_subset_maps_binary_nicr_species_and_site_energies():
    atoms = bulk("Ni", "fcc", a=3.5, cubic=True)
    species = torch.tensor([0, 1, 0, 1])
    symbols = ["Ni", "Cr", "Ni", "Cr"]
    atoms.set_chemical_symbols(symbols)
    atoms.calc = EAM(potential=str(NICR_POTENTIAL))
    fractional = torch.tensor(atoms.get_scaled_positions(), dtype=torch.float64)
    log_volume = torch.tensor(np.log(atoms.get_volume()), dtype=torch.float64)
    model = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    assert model(species, fractional, log_volume).item() == pytest.approx(
        atoms.get_potential_energy(), abs=1e-10
    )
    assert model.all_site_energies(species, fractional, log_volume).shape == (4, 2)


def test_orthorhombic_bct_energy_forces_and_cell_derivative_match_ase():
    fractional = reference_sites()
    cell = cell_matrix(torch.tensor([2.68, 3.31], dtype=torch.float64))
    species = torch.arange(len(fractional)).remainder(2)
    symbols = ["Cr" if value else "Ni" for value in species.tolist()]
    atoms = Atoms(symbols, scaled_positions=fractional.numpy(), cell=cell.numpy(), pbc=True)
    atoms.positions += np.random.default_rng(91).normal(scale=0.005, size=(len(atoms), 3))
    fractional = torch.tensor(atoms.get_scaled_positions(wrap=False), dtype=torch.float64)
    atoms.calc = EAM(potential=str(NICR_POTENTIAL))
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()

    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    labels = oracle.labels_cell(species, fractional, cell)
    assert labels.energy.item() == pytest.approx(expected_energy, abs=2e-9)
    assert labels.forces.detach().numpy() == pytest.approx(expected_forces, abs=2e-9)

    epsilon = 1e-5
    for axis in (0, 2):
        energies = []
        for sign in (-1, 1):
            trial = atoms.copy()
            trial_cell = atoms.cell.array.copy()
            trial_cell[axis, axis] += sign * epsilon
            trial.set_cell(trial_cell, scale_atoms=True)
            trial.calc = EAM(potential=str(NICR_POTENTIAL))
            energies.append(trial.get_potential_energy())
        derivative = (energies[1] - energies[0]) / (2 * epsilon)
        assert labels.cell_derivative[axis, axis].item() == pytest.approx(derivative, abs=2e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_batched_labels_and_substitutions_match_cpu_and_ase():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    atoms = _batch_atoms()
    cpu_model = TorchCuNiEAM(POTENTIAL)
    cpu_labels = cpu_model.labels(*_batch_inputs(atoms, "cpu"))
    cpu_sites = cpu_model.all_site_energies(*_batch_inputs(atoms, "cpu"))
    cuda_model = TorchCuNiEAM(POTENTIAL).to(device)
    cuda_labels = cuda_model.labels(*_batch_inputs(atoms, device))
    cuda_sites = cuda_model.all_site_energies(*_batch_inputs(atoms, device))

    assert cuda_labels.energy.detach().cpu().numpy() == pytest.approx(
        cpu_labels.energy.detach().numpy(), abs=2e-9
    )
    assert cuda_labels.forces.detach().cpu().numpy() == pytest.approx(
        cpu_labels.forces.detach().numpy(), abs=2e-9
    )
    assert cuda_labels.stress.detach().cpu().numpy() == pytest.approx(
        cpu_labels.stress.detach().numpy(), abs=2e-11
    )
    assert cuda_labels.log_volume_derivative.detach().cpu().numpy() == pytest.approx(
        cpu_labels.log_volume_derivative.detach().numpy(), abs=2e-9
    )
    assert cuda_sites.detach().cpu().numpy() == pytest.approx(cpu_sites.detach().numpy(), abs=2e-9)
    for index, state in enumerate(atoms):
        expected = _ase_labels(state)
        assert cuda_labels.energy[index].item() == pytest.approx(expected[0], abs=2e-9)
        assert cuda_labels.forces[index].detach().cpu().numpy() == pytest.approx(
            expected[1], abs=2e-9
        )
        assert cuda_labels.stress[index].detach().cpu().numpy() == pytest.approx(
            expected[2], abs=2e-11
        )
        assert cuda_labels.log_volume_derivative[index].item() == pytest.approx(
            expected[3], abs=2e-9
        )
