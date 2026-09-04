"""Small reference Monte Carlo utilities for binary and multicomponent alloys."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from scipy.optimize import brentq
from scipy.special import expit

Energy = Callable[[Atoms], float]


@dataclass
class MoveStats:
    attempts: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0


@dataclass
class MCResult:
    samples: list[Atoms]
    energies: np.ndarray
    stats: dict[str, MoveStats]
    displacement_step: float
    log_volume_step: float


def reduced_log_probability(
    atoms: Atoms,
    energy: float,
    beta: float,
    pressure: float = 0.0,
    chemical_potentials: Mapping[str, float] | None = None,
) -> float:
    """Unnormalised semi-grand NPT log density in log-volume coordinates."""
    chemical_potentials = chemical_potentials or {}
    chemical_term = sum(chemical_potentials.get(symbol, 0.0) for symbol in atoms.symbols)
    return float(
        -beta * (energy + pressure * atoms.get_volume() - chemical_term)
        + len(atoms) * math.log(atoms.get_volume())
    )


def _accept(log_ratio: float, rng: np.random.Generator) -> bool:
    return log_ratio >= 0.0 or math.log(rng.random()) < log_ratio


def _restore(target: Atoms, source: Atoms) -> None:
    target.set_cell(source.cell, scale_atoms=False)
    target.set_positions(source.positions)
    target.set_chemical_symbols(source.get_chemical_symbols())


def alloy_monte_carlo(
    atoms: Atoms,
    energy_fn: Energy,
    beta: float,
    sweeps: int,
    *,
    pressure: float = 0.0,
    species: Sequence[str] | None = None,
    chemical_potentials: Mapping[str, float] | None = None,
    canonical: bool = False,
    species_moves: int = 1,
    displacement_step: float = 0.05,
    log_volume_step: float = 0.01,
    burn_in: int = 0,
    thin: int = 1,
    adapt_interval: int = 50,
    target_acceptance: float = 0.3,
    seed: int | None = None,
) -> MCResult:
    """Run collective-displacement, log-volume, and species Metropolis sweeps.

    ``pressure`` and ``energy_fn`` must use compatible energy/volume units. Step-size
    adaptation is performed during burn-in and is frozen before samples are recorded.
    """
    if beta <= 0 or sweeps < 1 or burn_in < 0 or thin < 1 or species_moves < 0:
        raise ValueError("beta, sweeps, and thin must be positive; burn_in must be non-negative")
    if displacement_step < 0 or log_volume_step < 0 or adapt_interval < 1:
        raise ValueError("step sizes must be non-negative and adapt_interval positive")
    species = tuple(species or sorted(set(atoms.get_chemical_symbols())))
    if species_moves and len(species) < 2:
        raise ValueError("species moves require at least two species")

    rng = np.random.default_rng(seed)
    state = atoms.copy()
    energy = float(energy_fn(state))
    stats = {name: MoveStats() for name in ("displacement", "volume", "species")}
    windows = {name: MoveStats() for name in ("displacement", "volume")}
    samples: list[Atoms] = []
    energies: list[float] = []
    total_sweeps = burn_in + sweeps

    def record(name: str, accepted: bool) -> None:
        stats[name].attempts += 1
        stats[name].accepted += accepted
        if name in windows:
            windows[name].attempts += 1
            windows[name].accepted += accepted

    for sweep in range(total_sweeps):
        if sweep == burn_in:
            stats = {name: MoveStats() for name in stats}
        if displacement_step:
            old = state.copy()
            trial_energy = energy
            state.positions += rng.normal(scale=displacement_step, size=state.positions.shape)
            state.wrap()
            trial_energy = float(energy_fn(state))
            accepted = _accept(-beta * (trial_energy - energy), rng)
            if accepted:
                energy = trial_energy
            else:
                _restore(state, old)
            record("displacement", accepted)

        if log_volume_step:
            old = state.copy()
            old_volume = state.get_volume()
            dv = float(rng.normal(scale=log_volume_step))
            state.set_cell(state.cell * math.exp(dv / 3.0), scale_atoms=True)
            trial_energy = float(energy_fn(state))
            delta = trial_energy - energy + pressure * (state.get_volume() - old_volume)
            accepted = _accept(-beta * delta + len(state) * dv, rng)
            if accepted:
                energy = trial_energy
            else:
                _restore(state, old)
            record("volume", accepted)

        for _ in range(species_moves):
            old_symbols = state.get_chemical_symbols()
            if canonical:
                groups = [
                    [i for i, symbol in enumerate(old_symbols) if symbol == s] for s in species
                ]
                nonempty = [group for group in groups if group]
                if len(nonempty) < 2:
                    continue
                first, second = rng.choice(len(nonempty), size=2, replace=False)
                i, j = int(rng.choice(nonempty[first])), int(rng.choice(nonempty[second]))
                state.symbols[i], state.symbols[j] = state.symbols[j], state.symbols[i]
                chemical_delta = 0.0
            else:
                i = int(rng.integers(len(state)))
                old_symbol = old_symbols[i]
                choices = [candidate for candidate in species if candidate != old_symbol]
                new_symbol = str(rng.choice(choices))
                state.symbols[i] = new_symbol
                mu = chemical_potentials or {}
                chemical_delta = mu.get(new_symbol, 0.0) - mu.get(old_symbol, 0.0)
            trial_energy = float(energy_fn(state))
            accepted = _accept(-beta * (trial_energy - energy - chemical_delta), rng)
            if accepted:
                energy = trial_energy
            else:
                state.set_chemical_symbols(old_symbols)
            record("species", accepted)

        if sweep < burn_in and (sweep + 1) % adapt_interval == 0:
            displacement_step *= math.exp(
                windows["displacement"].acceptance_rate - target_acceptance
            )
            log_volume_step *= math.exp(windows["volume"].acceptance_rate - target_acceptance)
            windows = {name: MoveStats() for name in windows}

        if sweep >= burn_in and (sweep - burn_in) % thin == 0:
            samples.append(state.copy())
            energies.append(energy)

    return MCResult(samples, np.asarray(energies), stats, displacement_step, log_volume_step)


def semi_grand_npt_mc(*args, **kwargs) -> MCResult:
    """Run :func:`alloy_monte_carlo` with single-site species flips."""
    kwargs["canonical"] = False
    return alloy_monte_carlo(*args, **kwargs)


def canonical_npt_mc(*args, **kwargs) -> MCResult:
    """Run :func:`alloy_monte_carlo` with fixed-composition species swaps."""
    kwargs["canonical"] = True
    return alloy_monte_carlo(*args, **kwargs)


@dataclass
class Replica:
    atoms: Atoms
    energy: float
    beta: float
    pressure: float = 0.0
    chemical_potentials: Mapping[str, float] = field(default_factory=dict)


def attempt_replica_exchange(
    left: Replica, right: Replica, rng: np.random.Generator | None = None
) -> bool:
    """Attempt a full-configuration exchange between two thermodynamic states."""
    if len(left.atoms) != len(right.atoms):
        raise ValueError("replica exchange requires equal atom counts")
    rng = rng or np.random.default_rng()
    log_ratio = (
        reduced_log_probability(
            right.atoms, right.energy, left.beta, left.pressure, left.chemical_potentials
        )
        + reduced_log_probability(
            left.atoms, left.energy, right.beta, right.pressure, right.chemical_potentials
        )
        - reduced_log_probability(
            left.atoms, left.energy, left.beta, left.pressure, left.chemical_potentials
        )
        - reduced_log_probability(
            right.atoms, right.energy, right.beta, right.pressure, right.chemical_potentials
        )
    )
    if not _accept(log_ratio, rng):
        return False
    left.atoms, right.atoms = right.atoms, left.atoms
    left.energy, right.energy = right.energy, left.energy
    return True


def replica_exchange_sweep(
    replicas: Sequence[Replica], parity: int = 0, rng: np.random.Generator | None = None
) -> list[bool]:
    """Attempt disjoint nearest-neighbour exchanges using even/odd pairing."""
    rng = rng or np.random.default_rng()
    return [
        attempt_replica_exchange(replicas[i], replicas[i + 1], rng)
        for i in range(parity % 2, len(replicas) - 1, 2)
    ]


def substitution_energies(
    atoms: Atoms, energy_fn: Energy, old_species: str, new_species: str
) -> np.ndarray:
    """Return all eligible single-site substitution energies for one configuration."""
    base = float(energy_fn(atoms))
    work = []
    for index, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == old_species:
            trial = atoms.copy()
            trial.symbols[index] = new_species
            work.append(float(energy_fn(trial)) - base)
    return np.asarray(work)


def fixed_composition_works(
    samples_n: Sequence[Atoms],
    samples_n1: Sequence[Atoms],
    energy_fn: Energy,
    beta: float,
    species_a: str,
    species_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Eq. S16 forward/reverse works for the edge ``n -> n+1``."""
    if not samples_n or not samples_n1:
        raise ValueError("both neighboring rungs require samples")
    n_atoms = len(samples_n[0])
    n = samples_n[0].get_chemical_symbols().count(species_b)
    ratio = (n_atoms - n) / (n + 1)
    forward = np.stack(
        [beta * substitution_energies(x, energy_fn, species_a, species_b) for x in samples_n]
    ) - math.log(ratio)
    reverse = np.stack(
        [-beta * substitution_energies(x, energy_fn, species_b, species_a) for x in samples_n1]
    ) - math.log(ratio)
    return forward, reverse


def fixed_composition_bar(forward_work: np.ndarray, reverse_work: np.ndarray) -> float:
    """Return the dimensionless neighboring-rung free-energy difference by BAR."""
    forward, reverse = np.asarray(forward_work), np.asarray(reverse_work)
    if forward.ndim != 2 or reverse.ndim != 2 or not forward.size or not reverse.size:
        raise ValueError("works must be non-empty (configuration, substitution-site) arrays")
    nf, nr = forward.shape[0], reverse.shape[0]
    correction = math.log(nr / nf)

    def equation(delta: float) -> float:
        lhs = nf * expit(delta + correction - forward).mean()
        lhs += nr * expit(delta + correction - reverse).mean()
        return float(lhs - nr)

    scale = max(100.0, float(np.max(np.abs(np.r_[forward.ravel(), reverse.ravel()]))) + 20.0)
    return float(brentq(equation, -scale, scale))
