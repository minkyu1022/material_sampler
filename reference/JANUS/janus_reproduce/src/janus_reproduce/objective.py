"""Hybrid continuous/discrete JANUS interpolation and objectives."""

from __future__ import annotations

import torch
from torch import Tensor

from .losses import sce, tsm
from .samplers import janus_tau_leap


def interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Linear stochastic interpolant x_t=(1-t)x_0+t x_1."""
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1 - t) * x0 + t * x1


def bounded_score_target(
    x0: Tensor,
    t: Tensor,
    target_score: Tensor,
    prior_mean: Tensor | float,
    prior_variance: Tensor | float,
    *,
    prior_score: Tensor | None = None,
) -> Tensor:
    """Bounded generalized score target from JANUS SI Eqs. S6--S7."""
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    c = t.square() / (t.square() + (1 - t).square())
    if prior_score is None:
        prior_score = -(x0 - torch.as_tensor(prior_mean, device=x0.device)) / torch.as_tensor(
            prior_variance, device=x0.device
        )
    return c / t.clamp_min(1e-6) * target_score + (1 - c) / (1 - t).clamp_min(1e-6) * prior_score


def mask_terminal(tokens: Tensor, t: Tensor, mask_token: int) -> tuple[Tensor, Tensor]:
    """Independently mask terminal tokens with probability 1-t."""
    while t.ndim < tokens.ndim:
        t = t.unsqueeze(-1)
    masked = torch.rand(tokens.shape, device=tokens.device) >= t
    return tokens.masked_fill(masked, mask_token), masked


def hybrid_loss(
    velocity: Tensor,
    velocity_target: Tensor,
    score: Tensor,
    score_target: Tensor,
    logits: Tensor,
    heat_bath: Tensor,
    masked: Tensor,
    discrete_weight: float = 2.0,
) -> Tensor:
    return tsm(velocity, velocity_target, score, score_target) + discrete_weight * sce(
        logits, heat_bath, masked
    )


def reveal_step(tokens: Tensor, logits: Tensor, t0: float, t1: float, mask_token: int) -> Tensor:
    """One simultaneous absorbing-mask tau-leap."""
    return janus_tau_leap(tokens, logits, t0, t1, mask_token)[0]
