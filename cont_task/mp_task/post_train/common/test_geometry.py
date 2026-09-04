import pytest
import torch

from geometry import (
    cartesian_force_to_fractional_score,
    cell_gradient_to_ltri_gradient,
    project_translation_zero_mode,
)


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


def test_ltri_cell_gradient_matches_autograd_and_finite_difference():
    params = torch.tensor(
        [[0.7, 0.23, 0.9, -0.17, 0.31, 1.1]], dtype=torch.float64, requires_grad=True
    )

    def decode(value):
        zero = torch.zeros_like(value[..., 0])
        return torch.stack(
            (
                value[..., 0].exp(), zero, zero,
                value[..., 1], value[..., 2].exp(), zero,
                value[..., 3], value[..., 4], value[..., 5].exp(),
            ), dim=-1,
        ).reshape(*value.shape[:-1], 3, 3)

    weights = torch.tensor(
        [[[0.4, -0.2, 0.1], [0.7, 0.3, -0.6], [-0.5, 0.8, 0.9]]], dtype=torch.float64
    )
    energy = (decode(params).square() * weights).sum()
    expected = torch.autograd.grad(energy, params)[0]
    cell = decode(params.detach()).requires_grad_(True)
    cell_gradient = torch.autograd.grad((cell.square() * weights).sum(), cell)[0]
    actual = cell_gradient_to_ltri_gradient(cell_gradient, params.detach())

    eps = 1e-6
    finite_difference = []
    for index in range(6):
        plus, minus = params.detach().clone(), params.detach().clone()
        plus[..., index] += eps
        minus[..., index] -= eps
        finite_difference.append(
            (((decode(plus).square() * weights).sum() - (decode(minus).square() * weights).sum()) / (2 * eps))
        )
    finite_difference = torch.stack(finite_difference).reshape_as(actual)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual, finite_difference, atol=1e-8, rtol=1e-8)
