# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Abstract potential interface and a helper to sum several potentials."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BasePotential(nn.Module, ABC):
    """Abstract base class for atomic potentials."""

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, pos: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Compute energy and forces for a batch of structures.

        Args:
            pos: Atomic coordinates of shape ``(B, N, 3)``.
            **kwargs: Additional arguments forwarded to the potential.

        Returns:
            Dict with at least ``"energy"`` ``(B,)`` and ``"forces"`` ``(B, N, 3)``.
        """


class SumPotential(BasePotential):
    """A potential that sums the contributions of multiple base potentials."""

    def __init__(self, potentials: list[BasePotential]):
        super().__init__()
        self.potentials = nn.ModuleList(potentials)

    def forward(self, pos: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        results = [potential(pos, **kwargs) for potential in self.potentials]
        keys = set.intersection(*(set(result.keys()) for result in results))
        return {key: sum(result[key] for result in results) for key in keys}
