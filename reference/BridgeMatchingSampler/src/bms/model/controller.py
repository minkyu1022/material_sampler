# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Thin wrapper around the backbone network that produces the SDE control."""

import torch
import torch.nn as nn

from bms.utils.geometry import subtract_mean


class ClippedModel(nn.Module):
    """Wraps a backbone, optionally clipping and centering its vector output."""

    def __init__(
        self,
        base_model: nn.Module,
        subtract_mean: bool = True,
        clip_value: float | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.clip_value = clip_value
        self.subtract_mean = subtract_mean

    def forward(
        self,
        time: torch.Tensor,
        data: torch.Tensor,  # [B, N, 3]
        return_potential: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if time.numel() == 1:
            time = time.expand(data.size(0))

        control = self.base_model(time, data, return_potential=return_potential)
        if return_potential:
            control, potential = control

        if self.clip_value is not None:
            control = control.clip(min=-self.clip_value, max=self.clip_value)

        if self.subtract_mean:
            control = subtract_mean(control)

        return (control, potential) if return_potential else control
