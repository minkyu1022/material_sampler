# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Chirality restraint that penalizes flipped stereocenters.

Adding a flat-bottom harmonic penalty on the improper dihedral angles keeps the
sampled all-atom configurations in the correct (L) chirality, which the bare
classical force field does not enforce on its own.
"""

import torch

from bms.potential.base import BasePotential
from bms.potential.collective_variable import TorsionCV


class ChiralityRestraint(BasePotential):
    def __init__(
        self,
        indices: list[list[int]],
        location: float = -0.6154797086703873,  # -35 degrees
        force_constant: float = 25.0,  # eV / radian^2
        tolerance: float = 0.4363323129985824,  # 25 degrees
    ):
        """
        Args:
            indices: Atom indices defining the improper dihedral angles.
            location: Target dihedral angle in radians.
            force_constant: Force constant of the restraint in eV / rad^2.
            tolerance: Flat-bottom half-width in radians.
        """
        super().__init__()
        self.cv = TorsionCV(indices)
        self.location = location
        self.force_constant = force_constant
        self.tolerance = tolerance

    def forward(self, pos: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        cvs = self.cv(pos)  # [B, n_cv]
        disp = torch.atan2(
            torch.sin(cvs - self.location), torch.cos(cvs - self.location)
        )
        excess = torch.clamp(disp.abs() - self.tolerance, min=0.0)
        energy = torch.sum(0.5 * self.force_constant * excess.pow(2), dim=-1)
        forces = -self.cv.vjp(pos, self.force_constant * excess * disp.sign())
        return {"energy": energy, "forces": forces}
