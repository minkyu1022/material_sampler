import pytest
import torch

from torus import torus_delta, torus_interpolate, wrapped_normal_log_prob, wrapped_normal_score


def test_delta_and_interpolation_cross_boundary_by_shortest_path():
    source = torch.tensor([0.9], dtype=torch.float64)
    target = torch.tensor([0.1], dtype=torch.float64)
    assert torch.allclose(torus_delta(target, source), torch.tensor([0.2], dtype=torch.float64))
    assert torch.allclose(torus_interpolate(source, target, torch.tensor(0.5)), torch.tensor([0.0], dtype=torch.float64))


def test_wrapped_density_is_periodic_and_normalized():
    grid = (torch.arange(20000, dtype=torch.float64) + 0.5) / 20000
    logp = wrapped_normal_log_prob(grid, 0.91, 0.08**2)
    integral = torch.exp(logp).mean()
    assert integral.item() == pytest.approx(1.0, abs=2e-7)
    shifted = wrapped_normal_log_prob(grid + 3.0, -1.09, 0.08**2)
    assert torch.allclose(logp, shifted, atol=2e-12, rtol=0)


def test_analytic_score_matches_autograd_across_boundary():
    x = torch.tensor([0.01, 0.49, 0.99], dtype=torch.float64, requires_grad=True)
    logp = wrapped_normal_log_prob(x, 0.97, torch.tensor(0.12**2, dtype=torch.float64)).sum()
    expected = torch.autograd.grad(logp, x)[0]
    actual = wrapped_normal_score(x.detach(), 0.97, 0.12**2)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_small_variance_matches_euclidean_score_away_from_boundary():
    x = torch.tensor([0.42], dtype=torch.float64)
    mean, variance = 0.4, 0.03**2
    assert wrapped_normal_score(x, mean, variance).item() == pytest.approx(
        -(x.item() - mean) / variance, rel=1e-12
    )


def test_invalid_variance_is_rejected():
    with pytest.raises(ValueError):
        wrapped_normal_log_prob(torch.tensor([0.2]), 0.0, 0.0)
