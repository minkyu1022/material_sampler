from __future__ import annotations

from typing import Callable

import torch

from src.components.sdes import BaseSDE


def clip(tensor: torch.Tensor, max_norm: float | None = None):
    if max_norm is not None:
        tensor = tensor.clip(min=-1.0 * max_norm, max=max_norm)
    return tensor


class ClippedCtrl(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        clip_norm: float | None = None,
        name: str = "ctrl",
        **kwargs,
    ):
        super().__init__()
        self.base_model = base_model
        self.clip_norm = clip_norm if clip_norm else None
        self.name = name

    def clipped_base_model(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return clip(self.base_model(t, x), max_norm=self.clip_norm)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.clipped_base_model(t, x)


class ScoreCtrl(ClippedCtrl):
    def __init__(
        self,
        *args,
        score: Callable,
        score_model: torch.nn.Module | None = None,
        detach_score: bool = True,
        scale_score: float = 1.0,
        clip_score: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.score_model = score_model
        self.score = score
        self.detach_score = detach_score
        self.scale_score = scale_score
        self.clip_score = clip_score

    def clipped_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        return clip(self.score(x), max_norm=self.clip_score)

    def clipped_score_model(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        output = clip(self.score_model(t, x), max_norm=self.clip_norm)
        assert output.shape in [(1, 1), (1, x.shape[-1]), x.shape, (x.shape[0], 1)]
        return output

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctrl = self.clipped_base_model(t, x)
        score = self.scale_score * self.clipped_score(t, x)
        if self.score_model is not None:
            score *= self.clipped_score_model(t, x)
        return ctrl + score


class LerpCtrl(ScoreCtrl):
    def __init__(
        self,
        *args,
        # sde: BaseSDE,
        prior_score: Callable,
        hard_constrain: bool = False,
        scale_lerp: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.sde = sde
        self.prior_score = prior_score
        self.hard_constrain = hard_constrain
        self.scale_lerp = scale_lerp

    def clipped_interpolated_score(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = self.score(x)
        try:
            output = torch.lerp(self.prior_score(x), output, t) ## Terminal T=1
        except:
            output = torch.lerp(self.prior_score(x), output, t[:,None]) ## Terminal T=1
        output = clip(output, max_norm=self.clip_score,)
        assert output.shape == x.shape
        return output

    def constrain(self, output, t):
        return 4 * output * (1 - t) * t ## Terminal T=1

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctrl = self.clipped_base_model(t, x)
        if self.hard_constrain:
            ctrl = self.constrain(ctrl, t)

        # Interpolated score
        score = self.scale_score * self.clipped_interpolated_score(t, x)
        if self.score_model is not None:
            score_model = self.clipped_score_model(t, x)
            if self.hard_constrain:
                score_model = self.constrain(score_model, t)
            score *= score_model

        return ctrl + score


class LerpPriorCtrl(LerpCtrl):
    def clipped_interpolated_score(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = (1.0 - t) * self.prior_score(x)
        output = clip(output, max_norm=self.clip_score)
        assert output.shape == x.shape
        return output

    def constrain(self, output, t):
        return 2 * output * t / self.terminal_t


class LerpTargetCtrl(LerpCtrl):
    def clipped_interpolated_score(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = t / self.score(x)
        output = clip(output, max_norm=self.clip_score)
        assert output.shape == x.shape
        return output

    def constrain(self, output, t):
        return 2 * output * (1.0 - t)

