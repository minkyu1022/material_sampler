"""Published Ni--Cr lattice specifications and fixed-composition conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from torch import Tensor

from .torch_eam import TorchEAM


@dataclass(frozen=True)
class NiCrLatticeSpec:
    phase: str
    n_atoms: int
    repeats: int
    graph_cutoff: float
    transition_intercept: float
    transition_slope: float


NICR_LATTICES = {
    "fcc": NiCrLatticeSpec("fcc", 108, 3, 5.0, -0.909, -5.4e-5),
    "bcc": NiCrLatticeSpec("bcc", 128, 4, 5.3, -1.086, 0.9e-5),
}


def build_nicr(phase: str, n_cr: int, *, lattice_constant: float, seed: int = 0) -> Atoms:
    """Build a paper-sized FCC or BCC cell containing exactly ``n_cr`` Cr atoms."""
    spec = NICR_LATTICES[phase]
    if not 0 <= n_cr <= spec.n_atoms:
        raise ValueError(f"n_cr must be in [0, {spec.n_atoms}]")
    atoms = bulk("Ni", phase, a=lattice_constant, cubic=True).repeat((spec.repeats,) * 3)
    indices = np.random.default_rng(seed).permutation(spec.n_atoms)[:n_cr]
    symbols = np.full(spec.n_atoms, "Ni", dtype=object)
    symbols[indices] = "Cr"
    atoms.set_chemical_symbols(symbols.tolist())
    return atoms


def sample_fixed_conditions(
    count: int,
    phase: str,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Provisional smoke distribution: uniform inverse T and uniform composition rung."""
    spec = NICR_LATTICES[phase]
    inverse_temperature = torch.empty(count, device=device).uniform_(
        1 / 1500.0, 1 / 600.0, generator=generator
    )
    temperature = inverse_temperature.reciprocal()
    n_cr = torch.randint(
        spec.n_atoms + 1, (count,), device=device, generator=generator
    )
    return temperature, n_cr


def provisional_constrained_reveal(
    species: Tensor,
    logits: Tensor,
    target_cr: Tensor,
    reveal_probability: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Provisional sequential capacity rule that terminates at exactly ``target_cr``.

    This boundary is intentionally isolated for replacement by the author-confirmed rule.
    Species use ``0=Ni, 1=Cr, 2=mask`` and returned log probabilities include only
    non-forced categorical decisions.
    """
    if species.ndim != 2 or logits.shape != (*species.shape, 2):
        raise ValueError("require species [batch, sites] and binary logits [batch, sites, 2]")
    target_cr = torch.as_tensor(target_cr, device=species.device).long().reshape(-1)
    if len(target_cr) != len(species) or torch.any((target_cr < 0) | (target_cr > species.shape[1])):
        raise ValueError("target_cr must contain one valid rung per batch item")
    if not 0 <= reveal_probability <= 1:
        raise ValueError("reveal_probability must be in [0, 1]")
    output = species.clone()
    log_probability = torch.zeros(len(species), dtype=torch.float64, device=species.device)
    for batch in range(len(species)):
        masked = torch.where(output[batch].eq(2))[0]
        if not len(masked):
            if output[batch].eq(1).sum() != target_cr[batch]:
                raise ValueError("fully revealed input violates target composition")
            continue
        selected = masked[
            torch.rand(len(masked), device=species.device, generator=generator) < reveal_probability
        ]
        selected = selected[torch.randperm(len(selected), device=species.device, generator=generator)]
        for site in selected:
            cr_needed = target_cr[batch] - output[batch].eq(1).sum()
            sites_left = output[batch].eq(2).sum()
            if cr_needed == 0:
                choice = torch.tensor(0, device=species.device)
            elif cr_needed == sites_left:
                choice = torch.tensor(1, device=species.device)
            else:
                probability = logits[batch, site].softmax(-1)
                choice = torch.multinomial(probability, 1, generator=generator).squeeze(0)
                log_probability[batch] += probability[choice].double().log()
            output[batch, site] = choice
    return output, log_probability


def substitution_energies(
    oracle: TorchEAM,
    species: Tensor,
    fractional: Tensor,
    log_volume: Tensor,
) -> tuple[Tensor, Tensor]:
    """Eligible Ni->Cr and Cr->Ni total-energy changes for fixed-composition BAR."""
    if species.ndim != 2 or not torch.all(species.eq(1).sum(1) == species.eq(1).sum(1)[0]):
        raise ValueError("each batch item must have the same fixed composition")
    current = oracle(species, fractional, log_volume)
    alternatives = oracle.all_site_energies(species, fractional, log_volume)
    delta = alternatives - current[:, None, None]
    batch = len(species)
    n_cr = int(species[0].eq(1).sum())
    return (
        delta[..., 1][species.eq(0)].reshape(batch, species.shape[1] - n_cr),
        delta[..., 0][species.eq(1)].reshape(batch, n_cr),
    )
