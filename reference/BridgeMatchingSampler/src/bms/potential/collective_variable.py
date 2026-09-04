# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Collective variables used by the chirality restraints.

Only the dihedral (torsion) CV needed for the alanine-dipeptide chirality
restraints is included here.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AminoAcidChirality(nn.Module):
    """Sign of the chirality of an amino acid (+1 for D, -1 for L)."""

    def __init__(self, index_N: int, index_CA: int, index_CB: int, index_C: int):
        super().__init__()
        self.index_N = index_N
        self.index_CA = index_CA
        self.index_CB = index_CB
        self.index_C = index_C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_N = x[:, self.index_N]
        x_CA = x[:, self.index_CA]
        x_CB = x[:, self.index_CB]
        x_C = x[:, self.index_C]
        cross_product = torch.cross(x_C - x_CA, x_CB - x_CA, dim=-1)
        return torch.sign(torch.sum((x_N - x_CA) * cross_product, dim=-1))


class BaseCV(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Number of collective variables computed by this CV."""

    @property
    @abstractmethod
    def periodic(self) -> bool:
        """Whether the CV is periodic."""

    @torch.enable_grad()
    def vjp(self, x: torch.Tensor, grad_cv: torch.Tensor) -> torch.Tensor:
        """Vector-Jacobian product of the CV with ``grad_cv``."""

        def closure(x_):
            return self.forward(x_)

        _, pullback = torch.func.vjp(closure, x)
        return pullback(grad_cv)[0]


class TorsionCV(BaseCV):
    """Dihedral (torsion) angles defined by groups of four atom indices."""

    def __init__(
        self,
        indices: list[list[int]],
        chirality_indicator: AminoAcidChirality | None = None,
    ):
        super().__init__()
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.chirality_indicator = chirality_indicator

    @property
    def dim(self) -> int:
        return len(self.indices)

    @property
    def periodic(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0, p1, p2, p3 = x[:, self.indices].unbind(dim=2)
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1 = b1 / (b1.pow(2).sum(dim=-1, keepdim=True) + 1e-10).sqrt()
        v = b0 - b1 * (b0 * b1).sum(dim=-1, keepdim=True)
        w = b2 - b1 * (b2 * b1).sum(dim=-1, keepdim=True)
        torsions = torch.arctan2(
            (torch.cross(b1, v, dim=-1) * w).sum(dim=-1), (v * w).sum(dim=-1)
        )
        if self.chirality_indicator is not None:
            torsions = torsions * self.chirality_indicator(x)[:, None]
        return torsions
