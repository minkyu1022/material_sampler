# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Training utilities: curriculum scheduling, EMA, and LR schedules."""

import copy
from collections import namedtuple
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import ListConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

CurrentTask = namedtuple("CurrentTask", ["name", "is_init_stage", "is_final_stage"])


class TrainingCurriculum:
    """Training curriculum defining which task to perform at each epoch."""

    def __init__(
        self,
        curriculum: ListConfig | Sequence[dict[str, int]],
        valid_tasks: Sequence[str] | None = None,
    ):
        """
        Args:
            curriculum: List of dicts where each dict maps a task name to the number
                of epochs to perform the task, e.g. ``[{"bridge_matching": 1000}]``.
            valid_tasks: List of valid tasks. If ``None``, every task in the
                curriculum is considered valid.
        """
        if valid_tasks:
            for task_dict in curriculum:
                task = list(task_dict.keys())[0]
                if task not in valid_tasks:
                    raise ValueError(
                        f"Task '{task}' is not in the list of valid tasks: "
                        f"{valid_tasks}"
                    )
        self.curriculum = curriculum
        self.cumulative_epochs = np.cumsum(
            [list(task_dict.values())[0] for task_dict in curriculum]
        )
        self.tasks = [list(task_dict.keys())[0] for task_dict in curriculum]

    def __len__(self) -> int:
        return self.cumulative_epochs[-1]

    def __getitem__(self, epoch_index: int) -> CurrentTask:
        if epoch_index < 0 or epoch_index >= len(self):
            raise IndexError("Epoch index out of bounds for the curriculum.")
        stage_index = np.searchsorted(self.cumulative_epochs, epoch_index, side="right")
        name = self.tasks[stage_index]
        is_init_stage = self.tasks.index(name) == stage_index
        is_final_stage = (
            list(reversed(self.tasks)).index(name) == len(self.tasks) - 1 - stage_index
        )
        return CurrentTask(
            name=name, is_init_stage=is_init_stage, is_final_stage=is_final_stage
        )

    def get_task_epochs(self, task_name: str) -> int:
        """Get the total number of epochs for a specific task."""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' is not in the curriculum.")
        return sum(
            task_dict[task_name]
            for task_dict in self.curriculum
            if task_name in task_dict
        )


class EMA(nn.Module):
    """Exponential Moving Average (EMA) of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999, start_step: int = 0):
        super().__init__()
        self.model = model
        self.decay = decay
        self.start_step = start_step
        self.shadow = copy.deepcopy(model)
        for param in self.shadow.parameters():
            param.detach_()

        self.parameter_names = {
            name
            for name, param in self.model.named_parameters()
            if param.is_floating_point() or param.is_complex()
        }
        self.buffer_names = {
            name
            for name, buffer in self.model.named_buffers()
            if buffer.is_floating_point() or buffer.is_complex()
        }

    @torch.no_grad()
    def update_ema(self, current_step: int | None = None):
        if not self.training:
            raise RuntimeError("EMA can only be updated during training.")

        if current_step is None:
            decay = self.decay
        elif current_step < self.start_step:
            decay = 0.0  # No decay before the start step
        else:
            current_step -= self.start_step
            decay = min(self.decay, (1 + current_step) / (10 + current_step))

        for name in self.parameter_names:
            shadow_param = self.shadow.get_parameter(name).data
            model_param = self.model.get_parameter(name).data
            shadow_param.lerp_(model_param, 1 - decay)
        for name in self.buffer_names:
            shadow_buffer = self.shadow.get_buffer(name).data
            model_buffer = self.model.get_buffer(name).data
            shadow_buffer.lerp_(model_buffer, 1 - decay)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.shadow(*args, **kwargs)


class CosineAnnealingWithWarmupScheduler(SequentialLR):
    """Cosine annealing scheduler with warmup for a specific task in the curriculum."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        curriculum: TrainingCurriculum,
        task_name: str,
        steps_per_epoch: int,
        warmup_epochs: int = 0,
        eta_min: float = 1e-6,
    ):
        if task_name not in curriculum.tasks:
            raise ValueError(f"Task '{task_name}' is not in the curriculum.")

        total_steps = curriculum.get_task_epochs(task_name) * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        anneal_steps = total_steps - warmup_steps

        if anneal_steps < 0:
            raise ValueError(
                f"Total steps for task '{task_name}' must be greater than "
                f"warmup steps ({total_steps} < {warmup_steps})."
            )

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=anneal_steps, eta_min=eta_min
        )
        super().__init__(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
