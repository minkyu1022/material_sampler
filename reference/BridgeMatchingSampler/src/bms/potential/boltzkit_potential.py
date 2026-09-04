# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Bridge between BMS and a boltzkit molecular Boltzmann target.

The energy and forces are evaluated by boltzkit's ``MolecularBoltzmann`` system,
which wraps an OpenMM force field. The same ``system`` object is reused by the
training loop to load reference data and run the evaluation pipeline.
"""

from typing import Literal

import torch

from boltzkit.targets.boltzmann import MolecularBoltzmann

from bms.potential.base import BasePotential


class BoltzkitPotential(BasePotential):
    """Wraps a boltzkit ``MolecularBoltzmann`` target as a BMS potential."""

    def __init__(
        self, system: MolecularBoltzmann, device: Literal["cuda", "cpu"] = "cuda"
    ):
        super().__init__()
        self.system = system

    def forward(self, pos: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        if pos.isnan().any():
            raise ValueError("Input positions contain NaN values.")

        energy, forces = self.system.get_energy_and_forces(pos.cpu().numpy())

        return {
            "energy": torch.as_tensor(energy).to(pos),
            "forces": torch.as_tensor(
                forces.reshape(-1, self.system.n_atoms, 3)
            ).to(pos),
        }
