"""Analytic wrapped-normal primitives for fractional coordinates on a unit torus."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def torus_delta(target: Tensor, source: Tensor) -> Tensor:
    """Shortest signed displacement from ``source`` to ``target`` on [0, 1)."""
    return torch.remainder(target - source + 0.5, 1.0) - 0.5


def torus_interpolate(source: Tensor, target: Tensor, t: Tensor) -> Tensor:
    """Constant-speed shortest-geodesic interpolation on the unit torus."""
    while t.ndim < source.ndim:
        t = t.unsqueeze(-1)
    return torch.remainder(source + t * torus_delta(target, source), 1.0)


def _image_offsets(variance: Tensor, tail_tolerance: float) -> Tensor:
    if tail_tolerance <= 0 or tail_tolerance >= 1:
        raise ValueError("tail_tolerance must lie in (0, 1)")
    if torch.any(variance <= 0) or not torch.isfinite(variance).all():
        raise ValueError("variance must be finite and positive")
    sigma_max = float(variance.detach().max().sqrt().cpu())
    radius = math.ceil(0.5 + sigma_max * math.sqrt(2.0 * math.log(1.0 / tail_tolerance)))
    return torch.arange(-radius, radius + 1, device=variance.device, dtype=variance.dtype)


def _wrapped_terms(value: Tensor, mean: Tensor, variance: Tensor, tail_tolerance: float) -> tuple[Tensor, Tensor]:
    value, mean, variance = torch.broadcast_tensors(value, mean, variance)
    delta = torus_delta(value, mean)
    offsets = _image_offsets(variance, tail_tolerance)
    lifted = delta.unsqueeze(-1) + offsets
    log_terms = -0.5 * lifted.square() / variance.unsqueeze(-1)
    log_terms = log_terms - 0.5 * torch.log(2.0 * math.pi * variance).unsqueeze(-1)
    return lifted, log_terms


def wrapped_normal_log_prob(
    value: Tensor,
    mean: Tensor | float,
    variance: Tensor | float,
    *,
    tail_tolerance: float = 1e-12,
) -> Tensor:
    """Elementwise log density of ``Normal(mean, variance) mod 1``.

    The integer-image sum is truncated from a Gaussian tail bound determined by
    ``tail_tolerance``. Callers choose which coordinate dimensions to sum.
    """
    mean = torch.as_tensor(mean, device=value.device, dtype=value.dtype)
    variance = torch.as_tensor(variance, device=value.device, dtype=value.dtype)
    _, log_terms = _wrapped_terms(value, mean, variance, tail_tolerance)
    return torch.logsumexp(log_terms, dim=-1)


def wrapped_normal_score(
    value: Tensor,
    mean: Tensor | float,
    variance: Tensor | float,
    *,
    tail_tolerance: float = 1e-12,
) -> Tensor:
    """Elementwise score ``d/d value log q(value | mean, variance)``."""
    mean = torch.as_tensor(mean, device=value.device, dtype=value.dtype)
    variance = torch.as_tensor(variance, device=value.device, dtype=value.dtype)
    lifted, log_terms = _wrapped_terms(value, mean, variance, tail_tolerance)
    weights = torch.softmax(log_terms, dim=-1)
    variance = torch.broadcast_tensors(value, mean, variance)[2]
    return (weights * (-lifted / variance.unsqueeze(-1))).sum(dim=-1)
