"""Cu--Ni EAM oracle, priors, and published reference grids."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.eam import EAM
from scipy.special import expit

KB_EV_K = 8.617333262145e-5
T_REF = 900.0


def build_cuni_fcc(
    n_atoms: int = 108,
    *,
    cu_fraction: float = 0.0,
    lattice_constant: float = 3.55,
    seed: int = 0,
) -> Atoms:
    """Build the paper's 3^3 (108) or transfer 4^3 (256) fcc cell."""
    repeats = {108: 3, 256: 4}.get(n_atoms)
    if repeats is None:
        raise ValueError("Cu--Ni reference cells contain 108 or 256 atoms")
    if not 0.0 <= cu_fraction <= 1.0:
        raise ValueError("cu_fraction must be in [0, 1]")
    atoms = bulk("Ni", "fcc", a=lattice_constant, cubic=True).repeat((repeats,) * 3)
    count = round(cu_fraction * n_atoms)
    indices = np.random.default_rng(seed).permutation(n_atoms)[:count]
    symbols = np.full(n_atoms, "Ni", dtype=object)
    symbols[indices] = "Cu"
    atoms.set_chemical_symbols(symbols.tolist())
    return atoms


def temperatures_108() -> np.ndarray:
    """Published 15-temperature Cu--Ni reference grid (500--1200 K)."""
    return np.linspace(500.0, 1200.0, 15)


def delta_mu_108() -> np.ndarray:
    """Published 33-state grid: 25 meV wings and 10 meV crossover spacing."""
    wings = np.arange(0.600, 1.150 + 0.0125, 0.025)
    crossover = np.arange(0.780, 0.900 + 0.005, 0.010)
    return np.unique(np.round(np.r_[wings, crossover], 3))


def temperatures_256() -> np.ndarray:
    """Published seven-temperature size-transfer grid over 600--1200 K."""
    return np.linspace(600.0, 1200.0, 7)


def transition_delta_mu(temperature: float | np.ndarray) -> float | np.ndarray:
    """SI Table 1 linear Cu--Ni transition-line estimate in eV."""
    return 0.893 - 5.4e-5 * np.asarray(temperature)


def composition_prior(temperature: float, delta_mu: float) -> float:
    """Non-interacting composition estimate used to condition the prior."""
    return float(expit((delta_mu - transition_delta_mu(temperature)) / (KB_EV_K * temperature)))


@dataclass(frozen=True)
class VolumePrior:
    v_ni: float
    v_cu: float
    omega: float
    alpha: float
    sigma_log_volume: float
    temperature_ref: float = T_REF

    def atomic_volume(self, cu_fraction: float, temperature: float) -> float:
        base = (
            (1 - cu_fraction) * self.v_ni
            + cu_fraction * self.v_cu
            + self.omega * cu_fraction * (1 - cu_fraction)
        )
        return base * (1 + self.alpha * (temperature - self.temperature_ref))

    def mean_log_volume(self, n_atoms: int, cu_fraction: float, temperature: float) -> float:
        return math.log(n_atoms * self.atomic_volume(cu_fraction, temperature))


def fit_volume_prior(
    compositions: np.ndarray,
    temperatures: np.ndarray,
    atomic_volumes: np.ndarray,
    *,
    alpha_bounds: tuple[float, float] = (-1e-4, 1e-4),
    alpha_points: int = 2001,
    temperature_ref: float = T_REF,
) -> VolumePrior:
    """Fit SI Eq. S44 by its documented one-dimensional alpha scan."""
    c, t, y = map(np.asarray, (compositions, temperatures, atomic_volumes))
    if c.shape != t.shape or c.shape != y.shape or c.ndim != 1 or len(c) < 3:
        raise ValueError("calibration arrays must be equal-length 1-D arrays")
    design = np.column_stack((1 - c, c, c * (1 - c)))
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in np.linspace(*alpha_bounds, alpha_points):
        scale = 1 + alpha * (t - temperature_ref)
        if np.any(scale <= 0):
            continue
        coefficients, *_ = np.linalg.lstsq(design, y / scale, rcond=None)
        residual = float(np.square(scale * (design @ coefficients) - y).sum())
        if best is None or residual < best[0]:
            best = residual, float(alpha), coefficients
    if best is None:
        raise ValueError("alpha bounds leave no positive thermal scale")
    _, alpha, coefficients = best
    return VolumePrior(*map(float, coefficients), alpha, float("nan"), temperature_ref)


def fit_displacement_width(
    temperatures: np.ndarray,
    widths: np.ndarray,
    *,
    temperature_ref: float = T_REF,
) -> tuple[float, float]:
    """Fit log sigma = log sigma_ref + p log(T/T_ref) from SI."""
    temperatures, widths = np.asarray(temperatures), np.asarray(widths)
    if temperatures.shape != widths.shape or np.any(temperatures <= 0) or np.any(widths <= 0):
        raise ValueError("temperatures and widths must be positive arrays of equal shape")
    p, log_sigma_ref = np.polyfit(np.log(temperatures / temperature_ref), np.log(widths), 1)
    return float(np.exp(log_sigma_ref)), float(p)


def fractional_hessian(atoms: Atoms, energy: CuNiEAM, step: float = 1e-4) -> np.ndarray:
    """Central-difference the EAM forces to obtain d2U/du2 in fractional coordinates."""
    if step <= 0:
        raise ValueError("step must be positive")
    size = 3 * len(atoms)
    hessian = np.empty((size, size))
    scaled = atoms.get_scaled_positions(wrap=False)
    for column in range(size):
        gradients = []
        for direction in (-1, 1):
            trial = atoms.copy()
            displaced = scaled.copy()
            displaced.reshape(-1)[column] += direction * step
            trial.set_scaled_positions(displaced)
            trial.calc = energy.calculator
            gradients.append((-trial.get_forces() @ trial.cell.array.T).reshape(-1))
        hessian[:, column] = (gradients[1] - gradients[0]) / (2 * step)
    return (hessian + hessian.T) / 2


def quasiharmonic_width(hessian: np.ndarray, temperature: float) -> float:
    """SI isotropic fractional width, excluding translational/unstable modes."""
    hessian = np.asarray(hessian)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or hessian.shape[0] % 3:
        raise ValueError("hessian must have shape (3N, 3N)")
    eigenvalues = np.linalg.eigvalsh(hessian)
    positive = eigenvalues[eigenvalues > max(float(eigenvalues.max()), 1.0) * 1e-8]
    if temperature <= 0 or not len(positive):
        raise ValueError("temperature and stable Hessian modes are required")
    return float(np.sqrt(KB_EV_K * temperature * np.reciprocal(positive).sum() / len(eigenvalues)))


@dataclass(frozen=True)
class EAMLabels:
    energy: float
    forces: np.ndarray
    stress: np.ndarray
    log_volume_derivative: float
    substitution_energies: np.ndarray
    heat_bath: np.ndarray


class CuNiEAM:
    """Thin ASE adapter; the potential path is always explicit for provenance."""

    def __init__(self, potential: str | Path):
        self.path = Path(potential).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.calculator = EAM(potential=str(self.path))
        if set(self.calculator.elements) != {"Cu", "Ni"}:
            raise ValueError(f"expected a Cu--Ni EAM potential, found {self.calculator.elements}")

    def _attach(self, atoms: Atoms) -> Atoms:
        if not set(atoms.get_chemical_symbols()) <= {"Cu", "Ni"}:
            raise ValueError("CuNiEAM accepts only Cu and Ni atoms")
        atoms.calc = self.calculator
        return atoms

    def energy(self, atoms: Atoms) -> float:
        return float(self._attach(atoms).get_potential_energy())

    def labels(self, atoms: Atoms, temperature: float, delta_mu: float) -> EAMLabels:
        """Energy/force/stress and all-site [Ni, Cu] heat-bath labels."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        state = self._attach(atoms.copy())
        energy = float(state.get_potential_energy())
        forces = state.get_forces()
        stress = state.get_stress(voigt=False)
        site_energies = np.empty((len(state), 2))
        for index in range(len(state)):
            for species_index, species in enumerate(("Ni", "Cu")):
                trial = state.copy()
                trial.symbols[index] = species
                site_energies[index, species_index] = self.energy(trial)
        substitution = site_energies[:, 1] - site_energies[:, 0]
        probability_cu = expit((delta_mu - substitution) / (KB_EV_K * temperature))
        heat_bath = np.column_stack((1 - probability_cu, probability_cu))
        # ASE stress is dE/d(strain)/V; isotropic dE/dlog(V) uses strain=dlog(V)/3.
        derivative = float(state.get_volume() * np.trace(stress) / 3)
        return EAMLabels(energy, forces, stress, derivative, substitution, heat_bath)
