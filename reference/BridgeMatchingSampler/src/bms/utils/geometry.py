# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Batched geometry helpers operating on position tensors of shape ``(B, N, 3)``."""

import torch


def get_mean(positions: torch.Tensor) -> torch.Tensor:
    """Compute the mean position per molecule in a batch.

    Args:
        positions: Tensor of shape ``(B, N, D)`` where ``B`` is the batch size,
            ``N`` is the number of atoms, and ``D`` is the spatial dimension.

    Returns:
        Tensor of shape ``(B, D)`` containing the mean positions.
    """
    return positions.mean(dim=1)


def subtract_mean(positions: torch.Tensor) -> torch.Tensor:
    """Subtract the per-molecule mean to center the coordinates.

    Args:
        positions: Tensor of shape ``(B, N, D)``.

    Returns:
        Tensor of shape ``(B, N, D)`` with the per-molecule mean removed.
    """
    return positions - positions.mean(dim=1, keepdim=True)


def is_mean_free(positions: torch.Tensor, atol: float = 1e-5) -> bool:
    """Check whether the positions are mean-free (centered at the origin)."""
    means = get_mean(positions)
    return torch.allclose(means, torch.zeros_like(means), atol=atol)
