import torch

from janus_reproduce.objective import interpolate, mask_terminal, reveal_step


def test_interpolant_endpoints():
    x0, x1 = torch.randn(4, 3), torch.randn(4, 3)
    assert torch.equal(interpolate(x0, x1, torch.tensor(0.0)), x0)
    assert torch.equal(interpolate(x0, x1, torch.tensor(1.0)), x1)


def test_mask_and_final_reveal():
    tokens = torch.tensor([[0, 1, 0]])
    masked, selected = mask_terminal(tokens, torch.tensor([0.0]), 2)
    assert selected.all() and masked.eq(2).all()
    logits = torch.tensor([[[100.0, 0.0], [100.0, 0.0], [100.0, 0.0]]])
    assert reveal_step(masked, logits, 0.9, 1.0, 2).eq(0).all()
