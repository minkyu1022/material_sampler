import pytest
import torch

from geometry import cartesian_force_to_fractional_score, project_translation_zero_mode


def test_force_conversion_matches_autograd_chain_rule():
    frac = torch.tensor([[[0.2, -0.3, 0.4], [0.1, 0.5, -0.2]]], dtype=torch.float64, requires_grad=True)
    cell = torch.tensor([[[2.0, 0.1, 0.0], [0.0, 3.0, 0.2], [0.0, 0.0, 4.0]]], dtype=torch.float64)
    beta = torch.tensor([1.7], dtype=torch.float64)
    cart = torch.matmul(frac, cell)
    energy = 0.5 * cart.square().sum()
    expected = -beta[:, None, None] * torch.autograd.grad(energy, frac)[0]
    actual = cartesian_force_to_fractional_score(-cart.detach(), cell, beta)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_translation_projection_respects_padding():
    field = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [9.0, 9.0, 9.0]]])
    mask = torch.tensor([[False, False, True]])
    projected = project_translation_zero_mode(field, mask)
    assert torch.allclose(projected[:, :2].sum(dim=1), torch.zeros(1, 3))
    assert torch.equal(projected[:, 2], torch.zeros(1, 3))


def test_force_conversion_rejects_wrong_shape():
    with pytest.raises(ValueError):
        cartesian_force_to_fractional_score(torch.zeros(2, 2), torch.eye(3), 1.0)
