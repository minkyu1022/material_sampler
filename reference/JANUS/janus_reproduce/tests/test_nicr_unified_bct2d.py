import math

import pytest
import torch

from janus_reproduce.alloy_model import minimum_image_displacements_cell
from janus_reproduce.free_energy import gaussian_path_log_ratio
from janus_reproduce.nicr_unified_bct2d import (
    N_ATOMS,
    BCTDomain,
    CellNormalization,
    JANUSUnifiedBCT2D,
    cell_matrix,
    log_coordinate_jacobian,
    reference_sites,
    transformed_target_log_density,
)
from janus_reproduce.nicr_unified_reference import ReferenceMCConfig, reference_mc
from janus_reproduce.nicr_unified_train import (
    UnifiedTrainConfig,
    _cell_prior_log_density,
    _cell_prior_score,
    _project_u_field,
    _sample_cell_prior,
    diffusion_scale,
    displacement_prior_log_density,
    displacement_prior_scale,
    flow_matching_loss,
    rollout,
    target_scores,
)
from janus_reproduce.objective import bounded_score_target
from janus_reproduce.torch_eam import TorchEAM

NICR_POTENTIAL = "potentials/ni_co_cr/Ni-Co-Cr_v1.eam.fs"


def test_common_reference_has_128_unique_sites_and_bain_endpoints():
    sites = reference_sites()
    assert sites.shape == (N_ATOMS, 3)
    assert torch.unique(sites, dim=0).shape[0] == N_ATOMS
    bcc = cell_matrix(torch.tensor([2.8, 2.8], dtype=torch.float64))
    fcc = cell_matrix(torch.tensor([2.8, 2.8 * math.sqrt(2)], dtype=torch.float64))
    assert bcc[2, 2] / bcc[0, 0] == pytest.approx(1.0)
    assert fcc[2, 2] / fcc[0, 0] == pytest.approx(math.sqrt(2))
    expected = sites * torch.tensor([4 * 2.8, 4 * 2.8, 4 * 2.8], dtype=torch.float64)
    torch.testing.assert_close(sites @ bcc, expected)
    bcc_distance = minimum_image_displacements_cell(sites[None], bcc[None]).norm(dim=-1)[0]
    fcc_distance = minimum_image_displacements_cell(sites[None], fcc[None]).norm(dim=-1)[0]
    bcc_nearest = bcc_distance[bcc_distance > 0].min()
    fcc_nearest = fcc_distance[fcc_distance > 0].min()
    assert bcc_distance.isclose(bcc_nearest).sum(1).unique().item() == 8
    assert fcc_distance.isclose(fcc_nearest).sum(1).unique().item() == 12


def test_cell_normalization_round_trip():
    normalization = CellNormalization()
    cell = torch.tensor(((2.8, 2.8), (2.60, 3.14), (2.7, 3.7)), dtype=torch.float64)
    torch.testing.assert_close(normalization.decode(normalization.encode(cell)), cell)
    endpoints = normalization.encode(
        torch.tensor(((2.765, 2.765), (2.462, 3.482)), dtype=torch.float64)
    )
    torch.testing.assert_close(endpoints[0], torch.tensor((1.0, -1.0), dtype=torch.float64))
    torch.testing.assert_close(endpoints[1], torch.tensor((-1.0, 1.0), dtype=torch.float64))


def test_log_coordinate_jacobian_matches_autograd():
    y = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
    cell = 2.765 * y.exp()
    jacobian = torch.autograd.functional.jacobian(lambda value: 2.765 * value.exp(), y)
    assert log_coordinate_jacobian(cell).item() == pytest.approx(torch.logdet(jacobian).item())


def test_transformed_target_includes_atomic_and_cell_jacobians_once():
    cell = torch.tensor([2.7, 3.1], dtype=torch.float64)
    energy = torch.tensor(-10.0, dtype=torch.float64)
    volume = torch.linalg.det(cell_matrix(cell))
    expected = -2 * energy + 3 * volume.log() + cell.log().sum()
    assert transformed_target_log_density(energy, cell, 2.0, n_atoms=4) == pytest.approx(expected)


def test_solid_bain_domain_rejects_nonlocal_or_unbounded_states():
    domain = BCTDomain()
    assert domain.contains(torch.tensor([2.765, 2.765]), torch.zeros(N_ATOMS, 3))
    assert domain.contains(torch.tensor([2.462, 3.482]), torch.zeros(N_ATOMS, 3))
    assert not domain.contains(torch.tensor([10.0, 10.0]), torch.zeros(N_ATOMS, 3))
    assert not domain.contains(torch.tensor([2.765, 2.765]), torch.full((N_ATOMS, 3), 0.1))
    localized = torch.zeros(N_ATOMS, 3)
    localized[0] = torch.tensor([0.13, 0.13, 0.13])
    assert localized.square().sum(-1).mean().sqrt() < domain.rms_u_max
    assert not domain.contains(torch.tensor([2.765, 2.765]), localized)
    value = transformed_target_log_density(
        torch.tensor(0.0), torch.tensor([10.0, 10.0]), 1.0
    )
    assert torch.isneginf(value)


def test_solid_domain_admits_mandated_high_temperature_bootstrap_diffusion():
    config = UnifiedTrainConfig()
    g2 = diffusion_scale(torch.tensor([1500.0]), config).square().item()
    sigma_u = displacement_prior_scale(torch.tensor([1500.0]), config).item()
    expected_endpoint_rms = math.sqrt(3 * (sigma_u**2 + 2 * g2))
    assert expected_endpoint_rms < BCTDomain().rms_u_max


def test_displacement_prior_width_uses_cuni_temperature_scaling():
    config = UnifiedTrainConfig()
    temperature = torch.tensor([750.0, 1500.0], dtype=torch.float64)
    actual = displacement_prior_scale(temperature, config)
    expected = torch.tensor(
        [config.sigma_u_ref, config.sigma_u_ref * math.sqrt(2)], dtype=torch.float64
    )
    torch.testing.assert_close(actual, expected)


def test_displacement_prior_log_density_uses_zero_com_rank_and_temperature():
    config = UnifiedTrainConfig()
    sigma = displacement_prior_scale(torch.tensor([750.0, 1500.0], dtype=torch.float64), config)
    standardized = torch.randn(
        1, N_ATOMS, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(4)
    )
    standardized -= standardized.mean(1, keepdim=True)
    value = sigma[:, None, None] * standardized
    actual = displacement_prior_log_density(value, sigma)
    expected_difference = -3 * (N_ATOMS - 1) * math.log(math.sqrt(2))
    assert (actual[1] - actual[0]).item() == pytest.approx(expected_difference, abs=1e-10)


def test_generalized_score_target_contains_terminal_and_prior_terms():
    u0 = torch.tensor([[[0.02, -0.01, 0.03]]], dtype=torch.float64)
    terminal_score = torch.tensor([[[2.0, -3.0, 4.0]]], dtype=torch.float64)
    t = torch.tensor([0.25], dtype=torch.float64)
    sigma2 = torch.tensor([[[0.01**2]]], dtype=torch.float64)
    c = t.square() / (t.square() + (1 - t).square())
    expected = c[:, None, None] / t[:, None, None] * terminal_score
    expected += (1 - c)[:, None, None] / (1 - t)[:, None, None] * (-u0 / sigma2)
    torch.testing.assert_close(
        bounded_score_target(u0, t, terminal_score, 0.0, sigma2), expected
    )


def test_symmetric_cell_mixture_prior_samples_both_endpoints_and_has_exact_density():
    config = UnifiedTrainConfig(cell_prior_scale=0.2, cell_prior_mixture=True)
    sample = _sample_cell_prior(
        2_000, torch.float64, torch.device("cpu"), config, torch.Generator().manual_seed(81)
    )
    assert (sample[:, 0] > 0).double().mean().item() == pytest.approx(0.5, abs=0.04)
    point = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    variance = config.cell_prior_scale**2
    components = torch.tensor(
        [
            -math.log(2 * math.pi * variance),
            -math.log(2 * math.pi * variance) - 4 / variance,
        ],
        dtype=torch.float64,
    )
    expected = torch.logsumexp(components, 0) - math.log(2)
    torch.testing.assert_close(_cell_prior_log_density(point, config)[0], expected)


def test_symmetric_cell_mixture_prior_score_matches_density_gradient():
    config = UnifiedTrainConfig(cell_prior_scale=0.35, cell_prior_mixture=True)
    point = torch.tensor(((0.2, -0.4), (-0.8, 0.9)), dtype=torch.float64, requires_grad=True)
    expected = torch.autograd.grad(_cell_prior_log_density(point, config).sum(), point)[0]
    torch.testing.assert_close(_cell_prior_score(point.detach(), config), expected)


def test_zero_com_gaussian_ratio_matches_projected_subspace_density():
    generator = torch.Generator().manual_seed(73)
    fields = [torch.randn(2, 5, 3, dtype=torch.float64, generator=generator) for _ in range(6)]
    x0, x1, b0, s0, b1, s1 = [value - value.mean(1, keepdim=True) for value in fields]
    g2, dt = torch.tensor([0.03, 0.05], dtype=torch.float64), 0.2
    actual = gaussian_path_log_ratio(x0, x1, b0, s0, b1, s1, g2, dt)
    variance = (2 * g2 * dt)[:, None, None]
    forward_residual = x1 - x0 - (b0 + g2[:, None, None] * s0) * dt
    backward_residual = x0 - x1 + (b1 - g2[:, None, None] * s1) * dt
    expected = 0.5 * (
        forward_residual.square().sum((1, 2)) - backward_residual.square().sum((1, 2))
    ) / variance[:, 0, 0]
    torch.testing.assert_close(actual, expected)


def test_unified_model_has_two_cell_channels_and_preserves_zero_com_outputs():
    model = JANUSUnifiedBCT2D(features=8, layers=1, radial_basis=4)
    species = torch.full((2, N_ATOMS), 2)
    disp_u = torch.zeros(2, N_ATOMS, 3)
    cell_z = torch.tensor(((0.0, -1.0), (0.0, 1.0)))
    output = model(
        species,
        disp_u,
        cell_z,
        reference_sites(dtype=torch.float32),
        torch.tensor([0.1, 0.9]),
        torch.tensor([600.0, 1500.0]),
        torch.tensor([0.25, 0.75]),
    )
    assert output.b_u.shape == output.s_u.shape == (2, N_ATOMS, 3)
    assert output.b_cell.shape == output.s_cell.shape == (2, 2)
    assert output.species_logits.shape == (2, N_ATOMS, 2)
    torch.testing.assert_close(output.b_u.mean(1), torch.zeros(2, 3))
    torch.testing.assert_close(output.s_u.mean(1), torch.zeros(2, 3))


def test_unified_vector_heads_convert_cartesian_velocity_and_score_to_fractional_units():
    model = JANUSUnifiedBCT2D(features=8, layers=1, radial_basis=4).double()
    model.b_u.anchor.data.fill_(1)
    model.s_u.anchor.data.fill_(1)
    disp_u = torch.linspace(-0.01, 0.01, N_ATOMS * 3, dtype=torch.float64).reshape(1, N_ATOMS, 3)
    disp_u -= disp_u.mean(1, keepdim=True)
    cell_z = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
    cell = cell_matrix(model.normalization.decode(cell_z))
    output = model(
        torch.full((1, N_ATOMS), 2), disp_u, cell_z, reference_sites()[None],
        0.5, 1000.0, 0.5,
    )
    torch.testing.assert_close(output.b_u, disp_u, atol=1e-12, rtol=1e-12)
    expected_score = torch.einsum(
        "bni,bij,bkj->bnk", disp_u, cell, cell
    )
    expected_score -= expected_score.mean(1, keepdim=True)
    torch.testing.assert_close(output.s_u, expected_score, atol=1e-10, rtol=1e-12)


def test_unified_target_scores_are_finite_and_zero_com():
    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    species = torch.arange(N_ATOMS).remainder(2)[None]
    disp_u = torch.zeros(1, N_ATOMS, 3, dtype=torch.float64)
    cell_z = torch.zeros(1, 2, dtype=torch.float64)
    log_density, score_u, score_cell = target_scores(
        oracle, species, disp_u, cell_z, torch.tensor([1050.0], dtype=torch.float64)
    )
    assert torch.isfinite(log_density).all()
    assert torch.isfinite(score_u).all()
    assert torch.isfinite(score_cell).all()
    torch.testing.assert_close(score_u.mean(1), torch.zeros(1, 3, dtype=torch.float64), atol=1e-12, rtol=0)


def test_fractional_u_and_cell_z_scores_match_finite_differences():
    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    species = torch.arange(N_ATOMS).remainder(2)[None]
    u = 0.001 * torch.randn(
        1, N_ATOMS, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(71)
    )
    u -= u.mean(1, keepdim=True)
    z = torch.zeros(1, 2, dtype=torch.float64)
    temperature = torch.tensor([1050.0], dtype=torch.float64)
    _, score_u, score_cell = target_scores(oracle, species, u, z, temperature)
    epsilon = 1e-5
    direction_u = torch.zeros_like(u)
    direction_u[0, 0, 0], direction_u[0, 1, 0] = 1, -1

    def density(test_u, test_z):
        return target_scores(
            oracle, species, test_u.clone(), test_z.clone(), temperature
        )[0].detach()

    finite_u = (density(u + epsilon * direction_u, z) - density(u - epsilon * direction_u, z)) / (2 * epsilon)
    assert finite_u.item() == pytest.approx((score_u * direction_u).sum().item(), rel=2e-5)
    direction_z = torch.tensor([[1.0, -0.4]], dtype=torch.float64)
    finite_z = (density(u, z + epsilon * direction_z) - density(u, z - epsilon * direction_z)) / (2 * epsilon)
    assert finite_z.item() == pytest.approx((score_cell * direction_z).sum().item(), rel=2e-5)


def test_exact_composition_rollout_uses_documented_diffusion_baseline():
    config = UnifiedTrainConfig(steps=3)
    assert diffusion_scale(torch.tensor([750.0]), config).square().item() == pytest.approx(0.02**2)
    model = JANUSUnifiedBCT2D(features=8, layers=1, radial_basis=4)
    result = rollout(
        model,
        torch.tensor([0, 37, 128]),
        torch.tensor([600.0, 1050.0, 1500.0]),
        config,
        generator=torch.Generator().manual_seed(19),
    )
    torch.testing.assert_close(result["species"].eq(1).sum(1), result["target_cr"])
    assert result["species"].ne(2).all()
    assert torch.isfinite(result["disp_u"]).all()
    assert torch.isfinite(result["cell_z"]).all()
    torch.testing.assert_close(result["disp_u"].mean(1), torch.zeros(3, 3), atol=1e-7, rtol=0)


def test_unified_path_weight_decomposition_is_finite():
    model = JANUSUnifiedBCT2D(features=8, layers=1, radial_basis=4).double()
    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    result = rollout(
        model,
        torch.tensor([64, 64]),
        torch.tensor([1050.0, 1050.0], dtype=torch.float64),
        UnifiedTrainConfig(steps=2, diffusion=0.001, cell_prior_scale=0.1),
        generator=torch.Generator().manual_seed(23),
        path_weights=True,
        oracle=oracle,
    )
    for key in (
        "log_target",
        "log_prior_u",
        "log_prior_cell",
        "log_q_species",
        "log_continuous_u",
        "log_continuous_cell",
        "log_weight",
        "normalized_weight",
        "ess",
    ):
        assert torch.isfinite(result[key]).all(), key
    assert result["normalized_weight"].sum() == pytest.approx(1.0)
    assert 1 <= result["ess"] <= 2


def test_path_weight_rollout_rejects_mixed_thermodynamic_states():
    model = JANUSUnifiedBCT2D(features=8, layers=1, radial_basis=4)
    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    with pytest.raises(ValueError, match="homogeneous"):
        rollout(
            model,
            torch.tensor([32, 64]),
            torch.tensor([900.0, 1200.0]),
            UnifiedTrainConfig(steps=2),
            path_weights=True,
            oracle=oracle,
        )


def test_clipped_u_fields_remain_on_zero_com_subspace():
    value = torch.randn(3, N_ATOMS, 3, generator=torch.Generator().manual_seed(81)) * 10
    projected = _project_u_field(value, 0.1)
    torch.testing.assert_close(projected.mean(1), torch.zeros(3, 3), atol=1e-8, rtol=0)
    assert projected.norm(dim=-1).max() <= 0.2


def test_tetragonal_reference_mc_preserves_composition_and_moves_both_cell_lengths():
    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    result = reference_mc(
        oracle,
        64,
        1050.0,
        ReferenceMCConfig(sweeps=3, burn_in=1, thin=1, species_moves=1),
        initial_cell_ac=torch.tensor([2.765, 2.765]),
        generator=torch.Generator().manual_seed(31),
    )
    assert result["species"].shape == (3, N_ATOMS)
    torch.testing.assert_close(result["species"].eq(1).sum(1), torch.full((3,), 64))
    assert result["cell_ac"].shape == (3, 2)
    assert torch.isfinite(result["cell_ac"]).all()
    assert result["cell_ac"][0].prod() > 0
    assert set(result["stats"]) == {"displacement", "cell", "bain", "species"}


def test_unified_flow_matching_update_is_finite():
    model = JANUSUnifiedBCT2D(features=8, layers=1, radial_basis=4).double()
    oracle = TorchEAM(NICR_POTENTIAL, species_indices=(0, 2))
    species = torch.stack((torch.arange(N_ATOMS).remainder(2), torch.arange(N_ATOMS).add(1).remainder(2)))
    terminal = {
        "species": species,
        "disp_u": torch.zeros(2, N_ATOMS, 3, dtype=torch.float64),
        "cell_z": torch.tensor(((0.0, -0.5), (0.0, 0.5)), dtype=torch.float64),
    }
    loss, components = flow_matching_loss(
        model,
        oracle,
        terminal,
        torch.tensor([900.0, 1200.0], dtype=torch.float64),
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
