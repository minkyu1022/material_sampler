"""Coordinate transformations shared by MP20 post-training objectives."""

from __future__ import annotations

import torch
from torch import Tensor


def cartesian_force_to_fractional_score(forces: Tensor, cell: Tensor, beta: Tensor | float) -> Tensor:
    """Convert Cartesian forces to ``grad_f log pi`` for ``r = f @ cell``.

    ``forces`` has shape ``(..., N, 3)`` and ``cell`` has shape ``(..., 3, 3)``.
    """
    if forces.shape[-1] != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError("forces and cell must end in (N, 3) and (3, 3)")
    score = torch.matmul(forces, cell.transpose(-1, -2))
    beta = torch.as_tensor(beta, device=score.device, dtype=score.dtype)
    while beta.ndim < score.ndim:
        beta = beta.unsqueeze(-1)
    return beta * score


def project_translation_zero_mode(field: Tensor, pad_mask: Tensor) -> Tensor:
    """Remove per-structure uniform translation and keep padded atoms at zero."""
    real = (~pad_mask.bool()).unsqueeze(-1)
    count = real.sum(dim=-2, keepdim=True).clamp_min(1)
    centered = field - (field * real).sum(dim=-2, keepdim=True) / count
    return torch.where(real, centered, torch.zeros_like(centered))
