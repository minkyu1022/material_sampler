"""Config-selectable discrete JANUS sampling kernels."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .fixed_cardinality import constrained_reveal_step


class BoundaryQuotaStep(NamedTuple):
    species: Tensor
    log_probability: Tensor
    forced: Tensor
    forced_to_cr: Tensor


def janus_tau_leap(
    species: Tensor,
    logits: Tensor,
    t0: float,
    t1: float,
    mask_token: int = 2,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """PAPER_CONFIRMED unconstrained absorbing-mask tau leap."""
    masked = species.eq(mask_token)
    probability = 1.0 if t1 == 1 else (t1 - t0) / (1 - t0)
    random = torch.rand(species.shape, device=species.device, generator=generator)
    revealed = masked & (random < probability)
    proposals = torch.multinomial(
        logits.softmax(-1).reshape(-1, logits.shape[-1]), 1, generator=generator
    ).reshape_as(species)
    output = torch.where(revealed, proposals, species)
    log_probability = torch.where(
        revealed,
        logits.log_softmax(-1).gather(-1, proposals[..., None]).squeeze(-1),
        0.0,
    ).double().sum(-1)
    return output, log_probability


def sequential_random_order(
    batch: int,
    sites: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """PUBLIC_CODE_REFERENCE-style independent random reveal permutations."""
    return torch.rand(batch, sites, device=device, generator=generator).argsort(-1)


def fixed_composition_boundary_quota(
    species: Tensor,
    logits: Tensor,
    target_cr: Tensor,
    site: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> BoundaryQuotaStep:
    """AUTHOR_CONFIRMED one-site reveal with boundary-only quota enforcement."""
    if species.ndim != 2 or logits.shape != (*species.shape, 2):
        raise ValueError("require species [batch,sites] and binary logits [batch,sites,2]")
    batch, sites = species.shape
    target_cr = torch.as_tensor(target_cr, device=species.device).long().reshape(-1)
    site = torch.as_tensor(site, device=species.device).long().reshape(-1)
    if len(target_cr) != batch or torch.any((target_cr < 0) | (target_cr > sites)):
        raise ValueError("target_cr must contain one valid composition per trajectory")
    if len(site) != batch or torch.any((site < 0) | (site >= sites)):
        raise ValueError("site must contain one valid reveal site per trajectory")
    row = torch.arange(batch, device=species.device)
    if not species[row, site].eq(2).all():
        raise ValueError("each reveal site must still be masked")

    revealed_cr = species.eq(1).sum(-1)
    masked = species.eq(2).sum(-1)
    remaining = target_cr - revealed_cr
    if torch.any((remaining < 0) | (remaining > masked)):
        raise ValueError("masked state has an invalid remaining Cr quota")

    force_ni = remaining.eq(0)
    force_cr = remaining.eq(masked)
    forced = force_ni | force_cr
    probability = logits[row, site].softmax(-1)
    sampled = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
    choice = torch.where(force_ni, 0, torch.where(force_cr, 1, sampled))
    output = species.clone()
    output[row, site] = choice
    log_probability = torch.where(
        forced,
        0.0,
        probability.gather(-1, choice[:, None]).squeeze(-1).double().log(),
    )
    return BoundaryQuotaStep(output, log_probability, forced, force_cr)


DISCRETE_SAMPLER_REGISTRY = {
    "janus_tau_leap": janus_tau_leap,
    "sequential_random_order": sequential_random_order,
    "fixed_composition_boundary_quota": fixed_composition_boundary_quota,
    "fixed_composition_dp": constrained_reveal_step,
}


def get_discrete_sampler(name: str):
    try:
        return DISCRETE_SAMPLER_REGISTRY[name]
    except KeyError as error:
        raise ValueError(
            f"unknown discrete sampler {name!r}; choose from {sorted(DISCRETE_SAMPLER_REGISTRY)}"
        ) from error
