import math

import pytest
import torch

from torus import (
    sample_wrapped_brownian_bridge,
    torus_delta,
    torus_interpolate,
    wrapped_normal_log_prob,
    wrapped_normal_score,
)


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


def test_bridge_has_exact_torus_endpoints():
    source = torch.tensor([0.9, 0.2], dtype=torch.float64)
    target = torch.tensor([0.1, 0.7], dtype=torch.float64)
    assert torch.allclose(sample_wrapped_brownian_bridge(source, target, 0.0, 0.3), source)
    assert torch.allclose(sample_wrapped_brownian_bridge(source, target, 0.3, 0.3), target)


def test_bridge_is_periodic_in_both_endpoints():
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    source = torch.tensor([0.93, 0.2], dtype=torch.float64)
    target = torch.tensor([0.07, 0.8], dtype=torch.float64)
    a = sample_wrapped_brownian_bridge(source, target, 0.08, 0.2, generator=generator_a)
    b = sample_wrapped_brownian_bridge(source + 2, target - 3, 0.08, 0.2, generator=generator_b)
    assert torch.allclose(a, b, atol=1e-12, rtol=0)


def test_bridge_samples_the_discrete_gaussian_winding_posterior():
    count = 50000
    source = torch.zeros(count, dtype=torch.float64)
    target = torch.full_like(source, 0.4)
    _, winding = sample_wrapped_brownian_bridge(
        source,
        target,
        0.2,
        0.5,
        generator=torch.Generator().manual_seed(19),
        return_winding=True,
    )
    offsets = torch.arange(-4, 5, dtype=torch.float64)
    expected = torch.softmax(-0.5 * (0.4 + offsets).square() / 0.5, dim=0)
    empirical = torch.stack([(winding == offset).double().mean() for offset in offsets])
    assert torch.allclose(empirical, expected, atol=0.006, rtol=0)


def test_endpoint_mean_is_not_a_sufficient_statistic_for_wrapped_score():
    x = torch.tensor(0.02, dtype=torch.float64)
    endpoints = torch.tensor([0.92, 0.25], dtype=torch.float64)
    weights = torch.tensor([0.3, 0.7], dtype=torch.float64)
    expected_score = (weights * wrapped_normal_score(x, endpoints, 0.12**2)).sum()
    circular_moment = (weights * torch.exp(2j * math.pi * endpoints)).sum()
    circular_mean = torch.remainder(torch.angle(circular_moment) / (2 * math.pi), 1.0)
    plug_in_score = wrapped_normal_score(x, circular_mean, 0.12**2)
    assert abs(expected_score - plug_in_score).item() > 1.0
