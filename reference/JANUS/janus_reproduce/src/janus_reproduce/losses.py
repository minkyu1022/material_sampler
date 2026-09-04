"""Config-selectable JANUS loss components."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def tsm(
    velocity: Tensor,
    velocity_target: Tensor,
    score: Tensor,
    score_target: Tensor,
) -> Tensor:
    """JANUS velocity regression plus generalized target-score matching."""
    return F.mse_loss(velocity, velocity_target) + F.mse_loss(score, score_target)


def sce(logits: Tensor, heat_bath: Tensor, masked: Tensor) -> Tensor:
    """JANUS soft cross entropy against single-site heat-bath conditionals."""
    if not masked.any():
        return logits.sum() * 0
    return -(heat_bath[masked] * logits[masked].log_softmax(-1)).sum(-1).mean()


CONT_LOSS_REGISTRY = {"tsm": tsm}
DISC_LOSS_REGISTRY = {"sce": sce}


def get_loss(registry: dict[str, object], name: str):
    try:
        return registry[name]
    except KeyError as error:
        raise ValueError(f"unknown loss {name!r}; choose from {sorted(registry)}") from error
