from pathlib import Path

import pytest
import torch

from janus_reproduce.alloy_model import AlloyPaiNN
from janus_reproduce.cuni import VolumePrior
from janus_reproduce.cuni_train import CuNiTrainConfig, _reference, _rollout
from janus_reproduce.free_energy import (
    cuni_prior_log_density,
    cuni_terminal_log_density,
    gaussian_path_log_ratio,
    include_zero_weight_attempts,
    path_weight_estimates,
    recenter_temperature_conditioning,
    revealed_token_log_probability,
)
from janus_reproduce.torch_eam import TorchCuNiEAM

POTENTIAL = Path(__file__).parents[1] / "potentials/cu_ni/Cu_Ni_Fischer_2018.eam.alloy"


def test_gaussian_path_ratio_matches_manual_scalar_normal_densities():
    x0, x1 = torch.tensor([[0.2]]), torch.tensor([[0.7]])
    velocity0, score0 = torch.tensor([[0.3]]), torch.tensor([[-0.4]])
    velocity1, score1 = torch.tensor([[-0.1]]), torch.tensor([[0.2]])
    g2, dt = torch.tensor([0.09]), 0.25
    actual = gaussian_path_log_ratio(x0, x1, velocity0, score0, velocity1, score1, g2, dt)
    variance = 2 * g2 * dt
    forward_mean = x0[:, 0] + (velocity0[:, 0] + g2 * score0[:, 0]) * dt
    backward_mean = x1[:, 0] - (velocity1[:, 0] - g2 * score1[:, 0]) * dt
    expected = -0.5 * (x0[:, 0] - backward_mean).square() / variance
    expected += 0.5 * (x1[:, 0] - forward_mean).square() / variance
    torch.testing.assert_close(actual, expected.double(), rtol=1e-6, atol=1e-7)


def test_gaussian_path_ratio_rejects_zero_diffusion():
    value = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="positive"):
        gaussian_path_log_ratio(value, value, value, value, value, value, torch.zeros(1), 0.1)


def test_gaussian_path_ratio_keeps_scalar_channel_batches_separate():
    x0 = torch.tensor([0.0, 1.0])
    x1 = torch.tensor([0.2, 1.4])
    zeros = torch.zeros(2)
    ratio = gaussian_path_log_ratio(
        x0, x1, zeros, zeros, torch.tensor([0.1, -0.2]), zeros, torch.ones(2), 0.1
    )
    assert ratio.shape == (2,)
    assert ratio[0] != ratio[1]


def test_exact_gaussian_bridge_path_estimator_converges():
    """Analytic 1-D Gaussian bridge validates Eq. 19--21 end to end."""
    torch.manual_seed(19)
    count = 20_000
    prior_mean, prior_sigma = 0.0, 1.3
    target_mean, target_sigma = 0.7, 0.55
    g2 = torch.full((count,), 0.04, dtype=torch.float64)

    def fields(x, t):
        mean = (1 - t) * prior_mean + t * target_mean
        variance = (1 - t) ** 2 * prior_sigma**2 + t**2 * target_sigma**2
        covariance = t * target_sigma**2 - (1 - t) * prior_sigma**2
        score = -(x - mean) / variance
        velocity = target_mean - prior_mean + covariance * (x - mean) / variance
        return velocity, score

    errors, effective_samples = [], []
    for steps in (10, 40, 160):
        x0 = prior_mean + prior_sigma * torch.randn(count, dtype=torch.float64)
        x = x0.clone()
        log_ratio = torch.zeros_like(x)
        dt = 1 / steps
        for step in range(steps):
            t0, t1 = step * dt, (step + 1) * dt
            b0, s0 = fields(x, t0)
            x1 = x + (b0 + g2 * s0) * dt + (2 * g2 * dt).sqrt() * torch.randn_like(x)
            b1, s1 = fields(x1, t1)
            log_ratio += gaussian_path_log_ratio(x, x1, b0, s0, b1, s1, g2, dt)
            x = x1
        log_prior = -0.5 * (((x0 - prior_mean) / prior_sigma).square() + torch.log(
            torch.tensor(2 * torch.pi * prior_sigma**2, dtype=torch.float64)
        ))
        log_target = -0.5 * (((x - target_mean) / target_sigma).square() + torch.log(
            torch.tensor(2 * torch.pi * target_sigma**2, dtype=torch.float64)
        ))
        log_weight = log_target - log_prior + log_ratio
        log_z, weights, ess = path_weight_estimates(log_weight)
        weighted_mean = torch.dot(weights, x)
        errors.append(abs(log_z.item()) + abs(weighted_mean.item() - target_mean))
        effective_samples.append(ess.item())
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.015
    assert effective_samples[-1] > 0.9 * count
    assert effective_samples == sorted(effective_samples)


def test_revealed_token_probability_sums_only_newly_revealed_sites():
    logits = torch.log(torch.tensor([[[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]]))
    tokens = torch.tensor([[0, 1, 1]])
    revealed = torch.tensor([[True, False, True]])
    actual = revealed_token_log_probability(logits, tokens, revealed)
    torch.testing.assert_close(actual, torch.log(torch.tensor([0.8 * 0.4])))


def test_temperature_recentering_preserves_all_model_outputs():
    torch.manual_seed(7)
    old = AlloyPaiNN(features=8, layers=1, radial_basis=4, temperature_reference=900.0)
    new = AlloyPaiNN(features=8, layers=1, radial_basis=4, temperature_reference=750.0)
    new.load_state_dict(recenter_temperature_conditioning(old.state_dict(), 900.0, 750.0))
    species = torch.randint(0, 3, (3, 4))
    displacement = 0.01 * torch.randn(3, 4, 3)
    reference = torch.rand(4, 3)
    log_volume = torch.full((3,), 3.8)
    time = torch.tensor([0.1, 0.5, 0.9])
    temperature = torch.tensor([600.0, 850.0, 1200.0])
    delta_mu = torch.tensor([0.7, 0.85, 1.0])
    old_output = old(species, displacement, log_volume, reference, time, temperature, delta_mu)
    new_output = new(species, displacement, log_volume, reference, time, temperature, delta_mu)
    for old_value, new_value in zip(old_output, new_output, strict=True):
        torch.testing.assert_close(old_value, new_value, rtol=2e-6, atol=2e-7)


def test_cuni_density_terms_and_path_estimates_include_volume_jacobian():
    displacement = torch.zeros(2, 4, 3)
    log_volume = torch.tensor([3.0, 4.0])
    sigma = torch.ones(2)
    prior = cuni_prior_log_density(displacement, log_volume, sigma, log_volume, 1.0)
    expected_prior = -0.5 * (3 * (4 - 1) + 1) * torch.log(torch.tensor(2 * torch.pi))
    torch.testing.assert_close(prior, expected_prior.expand(2))
    target = cuni_terminal_log_density(
        torch.zeros(2), torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]]), log_volume,
        torch.full((2,), 900.0), torch.zeros(2),
    )
    torch.testing.assert_close(target, 4 * log_volume)
    log_xi, weights, ess = path_weight_estimates(torch.log(torch.tensor([1.0, 3.0])))
    torch.testing.assert_close(log_xi, torch.log(torch.tensor(2.0, dtype=torch.float64)))
    torch.testing.assert_close(weights, torch.tensor([0.25, 0.75], dtype=torch.float64))
    torch.testing.assert_close(ess, torch.tensor(1.6, dtype=torch.float64))
    torch.testing.assert_close(
        include_zero_weight_attempts(log_xi, returned=2, attempted=4),
        torch.tensor(0.0, dtype=torch.float64),
    )


def test_real_cuni_100_step_rollout_accumulates_complete_hybrid_weight():
    torch.manual_seed(11)
    config = CuNiTrainConfig.smoke(
        POTENTIAL, Path("unused"), steps=100, rollout_batch=2,
        diffusion_u=0.001, diffusion_v=0.001,
    )
    model = AlloyPaiNN(features=8, layers=1, radial_basis=4)
    oracle = TorchCuNiEAM(POTENTIAL)
    prior = VolumePrior(11.0, 12.0, 0.0, 0.0, 0.02)
    result = _rollout(
        model, config, prior, _reference(4, torch.device("cpu")), 2,
        torch.device("cpu"), path_weights=True, oracle=oracle,
        path_weight_trace=True,
        temperature=torch.tensor(900.0), delta_mu=torch.tensor(0.84),
    )
    reconstructed = (
        result["log_target"] - result["log_prior"] - result["log_q_discrete"]
        + result["log_continuous_ratio"]
    )
    torch.testing.assert_close(result["log_weight"], reconstructed)
    torch.testing.assert_close(
        result["log_continuous_ratio"],
        result["log_continuous_u"] + result["log_continuous_v"],
    )
    torch.testing.assert_close(
        result["log_continuous_u"], result["log_continuous_u_steps"].sum(1)
    )
    assert result["log_continuous_u_steps"].shape == (2, 100)
    assert result["species"].lt(2).all()
    assert torch.isfinite(result["log_weight"]).all()
    assert result["normalized_weight"].sum().item() == pytest.approx(1.0)
    assert 1 <= result["ess"].item() <= 2
