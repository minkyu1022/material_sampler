# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Optional temperature annealing schedule over training epochs."""

import numpy as np
import torch


class TemperatureAnnealer:
    """Tracks the current temperature as a function of the training epoch.

    The positions tensor is returned unmodified; only ``current_temperature`` is
    updated so that the training loop can read it.
    """

    def __init__(self, epochs: list[int], temperatures: list[float]):
        """
        Args:
            epochs: Epochs at which the temperature changes.
            temperatures: Target temperature for each epoch in ``epochs``.
        """
        super().__init__()
        if len(epochs) != len(temperatures):
            raise ValueError("epochs and temperatures must have the same length.")
        self.epochs = np.array(epochs, dtype=int)
        self.temperatures = np.array(temperatures, dtype=float)
        self.current_temperature = float(self.temperatures[0])

    def __call__(self, pos: torch.Tensor, epoch: int) -> torch.Tensor:
        self.current_temperature = float(
            np.interp(epoch, self.epochs, self.temperatures)
        )
        return pos
