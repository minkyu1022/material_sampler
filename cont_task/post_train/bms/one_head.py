"""One clean-endpoint head parameterization for a linear Gaussian bridge."""

from __future__ import annotations

import torch
from torch import Tensor


def _expand_batch(value: Tensor, target: Tensor) -> Tensor:
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def endpoint_to_velocity_score(
    x_t: Tensor,
    x1_hat: Tensor,
    t: Tensor,
    prior_variance: Tensor | float,
    prior_mean: Tensor | float = 0.0,
    *,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Derive velocity and marginal score from ``E[x1 | x_t]``.

    Assumes ``x_t = (1-t) x_0 + t x_1`` and
    ``x_0 ~ N(prior_mean, prior_variance I)``. The score is Tweedie's
    identity for the Gaussian corruption ``x_t | x_1``.
    """
    if x_t.shape != x1_hat.shape:
        raise ValueError("x_t and x1_hat must have identical shapes")
    t = _expand_batch(t.to(device=x_t.device, dtype=x_t.dtype), x_t)
    if torch.any((t < 0) | (t >= 1)):
        raise ValueError("t must satisfy 0 <= t < 1")
    one_minus_t = (1 - t).clamp_min(eps)
    variance = torch.as_tensor(prior_variance, device=x_t.device, dtype=x_t.dtype)
    mean = torch.as_tensor(prior_mean, device=x_t.device, dtype=x_t.dtype)
    if variance.ndim:
        variance = _expand_batch(variance, x_t)
    if mean.ndim:
        mean = _expand_batch(mean, x_t)
    if torch.any(variance <= 0):
        raise ValueError("prior_variance must be positive")
    velocity = (x1_hat - x_t) / one_minus_t
    score = ((1 - t) * mean + t * x1_hat - x_t) / (one_minus_t.square() * variance)
    return velocity, score


def project_masked_com(field: Tensor, pad_mask: Tensor) -> Tensor:
    """Remove per-structure translation while keeping padded sites zero."""
    real = (~pad_mask.bool()).unsqueeze(-1)
    count = real.sum(dim=-2, keepdim=True).clamp_min(1)
    centered = field - (field * real).sum(dim=-2, keepdim=True) / count
    return torch.where(real, centered, torch.zeros_like(centered))
