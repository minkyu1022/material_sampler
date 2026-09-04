import pytest
import torch

from janus_reproduce.losses import CONT_LOSS_REGISTRY, DISC_LOSS_REGISTRY, get_loss, sce, tsm


def test_tsm_and_sce_registries_preserve_janus_objectives():
    velocity = torch.tensor([1.0, 2.0])
    score = torch.tensor([3.0, 4.0])
    assert CONT_LOSS_REGISTRY["tsm"](velocity, velocity - 1, score, score - 2) == 5
    logits = torch.tensor([[[0.0, 1.0], [2.0, -1.0]]])
    target = torch.tensor([[[0.25, 0.75], [0.8, 0.2]]])
    masked = torch.tensor([[True, False]])
    expected = -(target[masked] * logits[masked].log_softmax(-1)).sum(-1).mean()
    assert DISC_LOSS_REGISTRY["sce"](logits, target, masked) == expected
    assert tsm is CONT_LOSS_REGISTRY["tsm"] and sce is DISC_LOSS_REGISTRY["sce"]


def test_unknown_loss_fails_with_available_names():
    with pytest.raises(ValueError, match="tsm"):
        get_loss(CONT_LOSS_REGISTRY, "missing")
