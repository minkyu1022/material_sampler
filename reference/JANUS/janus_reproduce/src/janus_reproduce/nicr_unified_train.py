"""Training and rollout primitives for the unified Ni--Cr BCT-2D model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .cuni import KB_EV_K
from .cuni_train import _euler_maruyama
from .fixed_cardinality import constrained_nll, constrained_reveal_step
from .free_energy import gaussian_path_log_ratio, path_weight_estimates
from .nicr_unified_bct2d import (
    N_ATOMS,
    CellNormalization,
    JANUSUnifiedBCT2D,
    cell_matrix,
    reference_sites,
    transformed_target_log_density,
)
from .objective import bounded_score_target, interpolate, mask_terminal
from .torch_eam import TorchEAM


@dataclass(frozen=True)
class UnifiedTrainConfig:
    steps: int = 100
    diffusion: float = 0.02
    diffusion_temperature_ref: float = 750.0
    sigma_u_ref: float = 0.01004514620163701
    sigma_u_temperature_ref: float = 750.0
    sigma_u_exponent: float = 0.5
    cell_prior_scale: float = 1.0
    cell_prior_mixture: bool = False
    discrete_weight: float = 2.0
    target_score_u_clip: float = 100.0
    target_score_cell_clip: float = 1_000.0
    gradient_clip_norm: float = 1.0
    rollout_velocity_clip: float = 0.1
    rollout_velocity_cell_clip: float = 5.0
    rollout_score_clip: float = 1_000.0
    oracle_batch: int = 4


def diffusion_scale(temperature: Tensor, config: UnifiedTrainConfig) -> Tensor:
    """Author-confirmed Cu--Ni baseline reused for both normalized channels."""
    return config.diffusion * (temperature / config.diffusion_temperature_ref).sqrt()


def displacement_prior_scale(temperature: Tensor, config: UnifiedTrainConfig) -> Tensor:
    """Cu--Ni-calibrated fractional-displacement prior width at temperature ``T``."""
    return config.sigma_u_ref * (temperature / config.sigma_u_temperature_ref).pow(
        config.sigma_u_exponent
    )


def displacement_prior_log_density(value: Tensor, sigma_u: Tensor) -> Tensor:
    """Zero-COM Gaussian log density in its rank-``3*(N-1)`` subspace."""
    sigma_u = sigma_u.double()
    return -0.5 * (
        value.double().square().sum((-2, -1)) / sigma_u.square()
        + 3 * (N_ATOMS - 1) * torch.log(2 * torch.pi * sigma_u.square())
    )


def _sample_cell_prior(
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    config: UnifiedTrainConfig,
    generator: torch.Generator | None,
) -> Tensor:
    noise = config.cell_prior_scale * torch.randn(
        batch, 2, dtype=dtype, device=device, generator=generator
    )
    if not config.cell_prior_mixture:
        return noise
    component = torch.randint(2, (batch,), device=device, generator=generator)
    centers = noise.new_tensor(((1.0, -1.0), (-1.0, 1.0)))
    return noise + centers[component]


def _cell_prior_log_density(value: Tensor, config: UnifiedTrainConfig) -> Tensor:
    variance = value.new_tensor(config.cell_prior_scale**2)
    normalizer = 2 * torch.log(value.new_tensor(2 * torch.pi) * variance)
    centers = (
        value.new_tensor(((1.0, -1.0), (-1.0, 1.0)))
        if config.cell_prior_mixture
        else value.new_zeros((1, 2))
    )
    component = -0.5 * (
        (value[:, None] - centers[None]).square().sum(-1) / variance + normalizer
    )
    return torch.logsumexp(component, 1) - math.log(len(centers))


def _cell_prior_score(value: Tensor, config: UnifiedTrainConfig) -> Tensor:
    """Exact score of the zero-centered or symmetric two-component cell prior."""
    variance = value.new_tensor(config.cell_prior_scale**2)
    if not config.cell_prior_mixture:
        return -value / variance
    centers = value.new_tensor(((1.0, -1.0), (-1.0, 1.0)))
    logits = -0.5 * (value[:, None] - centers[None]).square().sum(-1) / variance
    posterior_center = torch.softmax(logits, 1) @ centers
    return (posterior_center - value) / variance


def target_scores(
    oracle: TorchEAM,
    species: Tensor,
    disp_u: Tensor,
    cell_z: Tensor,
    temperature: Tensor,
    *,
    normalization: CellNormalization | None = None,
    reference: Tensor | None = None,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Target log density and scores in fractional-u and normalized-cell space."""
    normalization = normalization or CellNormalization()
    disp_u = disp_u.requires_grad_(True)
    cell_z = cell_z.requires_grad_(True)
    reference = reference_sites(dtype=disp_u.dtype).to(disp_u.device) if reference is None else reference
    if reference.ndim == 2:
        reference = reference[None].expand(len(species), -1, -1)
    cell_ac = normalization.decode(cell_z)
    cell = cell_matrix(cell_ac)
    energy = oracle.forward_cell(species, reference + disp_u, cell)
    beta = 1 / (KB_EV_K * temperature)
    log_density = transformed_target_log_density(energy, cell_ac, beta, disp_u=disp_u)
    score_u, score_cell = torch.autograd.grad(
        log_density.sum(), (disp_u, cell_z), create_graph=create_graph
    )
    score_u = score_u - score_u.mean(1, keepdim=True)
    return log_density, score_u, score_cell


def _clip(value: Tensor, limit: float, vector: bool = False) -> Tensor:
    if not vector:
        return value.clamp(-limit, limit)
    norm = value.norm(dim=-1, keepdim=True)
    return value * (limit / norm.clamp_min(1e-12)).clamp_max(1)


def _project_u_field(value: Tensor, limit: float) -> Tensor:
    value = _clip(value, limit, vector=True)
    return value - value.mean(1, keepdim=True)


def flow_matching_loss(
    model: JANUSUnifiedBCT2D,
    oracle: TorchEAM,
    terminal: dict[str, Tensor],
    temperature: Tensor,
    config: UnifiedTrainConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One linear-interpolant hybrid FM/score-matching objective."""
    config = config or UnifiedTrainConfig()
    species_1 = terminal["species"]
    u_1 = terminal["disp_u"]
    z_1 = terminal["cell_z"]
    batch = len(species_1)
    sigma_u = displacement_prior_scale(temperature, config)
    u_0 = sigma_u[:, None, None] * torch.randn_like(u_1)
    u_0 -= u_0.mean(1, keepdim=True)
    z_0 = _sample_cell_prior(batch, z_1.dtype, z_1.device, config, None)
    t = torch.rand(batch, dtype=u_1.dtype, device=u_1.device)
    u_t, z_t = interpolate(u_0, u_1, t), interpolate(z_0, z_1, t)
    species_t, masked = mask_terminal(species_1, t, 2)
    if "score_u" in terminal and "score_cell" in terminal:
        terminal_score_u = terminal["score_u"]
        terminal_score_cell = terminal["score_cell"]
    else:
        _, terminal_score_u, terminal_score_cell = target_scores(
            oracle, species_1, u_1, z_1, temperature, normalization=model.normalization
        )
    clipped_terminal_score_u = _project_u_field(
        terminal_score_u.detach(), config.target_score_u_clip
    )
    target_score_u = bounded_score_target(
        u_0,
        t,
        clipped_terminal_score_u,
        0.0,
        sigma_u[:, None, None].square(),
    )
    target_score_cell = bounded_score_target(
        z_0,
        t,
        _clip(terminal_score_cell.detach(), config.target_score_cell_clip),
        0.0,
        config.cell_prior_scale**2,
        prior_score=_cell_prior_score(z_0, config),
    )
    output = model(
        species_t,
        u_t,
        z_t,
        reference_sites(dtype=u_t.dtype).to(u_t.device),
        t,
        temperature,
        species_1.eq(1).sum(1) / N_ATOMS,
    )
    discrete = constrained_nll(
        output.species_logits, species_1, masked, species_1.eq(1).sum(1)
    )
    components = {
        "velocity_u": F.mse_loss(output.b_u, u_1 - u_0),
        "score_u": F.mse_loss(output.s_u, target_score_u),
        "velocity_cell": F.mse_loss(output.b_cell, z_1 - z_0),
        "score_cell": F.mse_loss(output.s_cell, target_score_cell),
        "discrete": discrete,
    }
    loss = sum(value for key, value in components.items() if key != "discrete")
    loss = loss + config.discrete_weight * discrete
    return loss, components


@torch.no_grad()
def rollout(
    model: JANUSUnifiedBCT2D,
    target_cr: Tensor,
    temperature: Tensor,
    config: UnifiedTrainConfig | None = None,
    *,
    generator: torch.Generator | None = None,
    path_weights: bool = False,
    oracle: TorchEAM | None = None,
) -> dict[str, Tensor]:
    """Generate exact-composition unified BCT samples from the documented baseline prior."""
    config = config or UnifiedTrainConfig()
    if path_weights and oracle is None:
        raise ValueError("path_weights require an EAM oracle")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    target_cr = torch.as_tensor(target_cr, device=device).long().reshape(-1)
    temperature = torch.as_tensor(temperature, device=device, dtype=dtype).reshape(-1)
    if len(target_cr) != len(temperature) or torch.any((target_cr < 0) | (target_cr > N_ATOMS)):
        raise ValueError("target_cr and temperature must contain one valid value per sample")
    if path_weights and (
        torch.any(target_cr != target_cr[0]) or torch.any(temperature != temperature[0])
    ):
        raise ValueError("path-weight ESS requires one homogeneous (temperature,target_cr) state")
    batch = len(target_cr)
    species = torch.full((batch, N_ATOMS), 2, dtype=torch.long, device=device)
    sigma_u = displacement_prior_scale(temperature, config)
    disp_u = sigma_u[:, None, None] * torch.randn(
        batch, N_ATOMS, 3, device=device, dtype=dtype, generator=generator
    )
    disp_u -= disp_u.mean(1, keepdim=True)
    cell_z = _sample_cell_prior(batch, dtype, device, config, generator)
    reference = reference_sites(dtype=dtype).to(device)
    log_q_species = torch.zeros(batch, dtype=torch.float64, device=device)
    log_continuous_u = torch.zeros_like(log_q_species)
    log_continuous_cell = torch.zeros_like(log_q_species)
    initial_disp_u, initial_cell_z = disp_u.clone(), cell_z.clone()
    g = diffusion_scale(temperature, config)
    for step in range(config.steps):
        t0, t1 = step / config.steps, (step + 1) / config.steps
        output = model(
            species,
            disp_u,
            cell_z,
            reference,
            t0,
            temperature,
            target_cr / N_ATOMS,
        )
        if not all(torch.isfinite(value).all() for value in output):
            raise FloatingPointError(f"non-finite model output at rollout step {step}")
        b_u = _project_u_field(output.b_u, config.rollout_velocity_clip)
        s_u = _project_u_field(output.s_u, config.rollout_score_clip)
        b_cell = _clip(output.b_cell, config.rollout_velocity_cell_clip)
        s_cell = _clip(output.s_cell, config.rollout_score_clip)
        dt = t1 - t0
        previous_disp_u, previous_cell_z = disp_u, cell_z
        disp_u = _euler_maruyama(
            disp_u, b_u, s_u, g, dt,
            torch.randn(disp_u.shape, device=device, dtype=dtype, generator=generator),
        )
        disp_u -= disp_u.mean(1, keepdim=True)
        cell_z = _euler_maruyama(
            cell_z, b_cell, s_cell, g, dt,
            torch.randn(cell_z.shape, device=device, dtype=dtype, generator=generator),
        )
        probability = 1.0 if step + 1 == config.steps else dt / (1 - t0)
        species, delta_log_q = constrained_reveal_step(
            species,
            output.species_logits,
            target_cr,
            probability,
            generator=generator,
        )
        log_q_species += delta_log_q
        if path_weights:
            backward = model(
                species,
                disp_u,
                cell_z,
                reference,
                t1,
                temperature,
                target_cr / N_ATOMS,
            )
            if not all(torch.isfinite(value).all() for value in backward):
                raise FloatingPointError(f"non-finite backward model output at rollout step {step}")
            backward_b_u = _project_u_field(backward.b_u, config.rollout_velocity_clip)
            backward_s_u = _project_u_field(backward.s_u, config.rollout_score_clip)
            backward_b_cell = _clip(backward.b_cell, config.rollout_velocity_cell_clip)
            backward_s_cell = _clip(backward.s_cell, config.rollout_score_clip)
            log_continuous_u += gaussian_path_log_ratio(
                previous_disp_u,
                disp_u,
                b_u,
                s_u,
                backward_b_u,
                backward_s_u,
                g.square(),
                dt,
            )
            log_continuous_cell += gaussian_path_log_ratio(
                previous_cell_z,
                cell_z,
                b_cell,
                s_cell,
                backward_b_cell,
                backward_s_cell,
                g.square(),
                dt,
            )
    result = {
        "species": species,
        "disp_u": disp_u,
        "cell_z": cell_z,
        "cell_ac": model.normalization.decode(cell_z),
        "temperature": temperature,
        "target_cr": target_cr,
        "log_q_species": log_q_species,
    }
    if path_weights:
        assert oracle is not None
        cell_ac = model.normalization.decode(cell_z).double()
        energy = torch.cat(
            [
                oracle.forward_cell(
                    species[start : start + config.oracle_batch],
                    reference[None].double()
                    + disp_u[start : start + config.oracle_batch].double(),
                    cell_matrix(cell_ac[start : start + config.oracle_batch]),
                )
                for start in range(0, batch, config.oracle_batch)
            ]
        )
        beta = 1 / (KB_EV_K * temperature.double())
        log_target = transformed_target_log_density(energy, cell_ac, beta, disp_u=disp_u.double())
        log_prior_u = displacement_prior_log_density(initial_disp_u, sigma_u)
        log_prior_cell = _cell_prior_log_density(initial_cell_z.double(), config)
        log_weight = (
            log_target
            - log_prior_u
            - log_prior_cell
            - log_q_species
            + log_continuous_u
            + log_continuous_cell
        )
        if torch.isnan(log_weight).any() or torch.isposinf(log_weight).any():
            raise FloatingPointError("non-finite path weight")
        valid_domain = torch.isfinite(log_target)
        if valid_domain.any():
            log_xi, normalized_weight, ess = path_weight_estimates(log_weight)
        else:
            log_xi = log_weight.new_tensor(float("-inf"))
            normalized_weight = torch.zeros_like(log_weight)
            ess = log_weight.new_zeros(())
        result |= {
            "energy": energy,
            "log_target": log_target,
            "log_prior_u": log_prior_u,
            "log_prior_cell": log_prior_cell,
            "log_continuous_u": log_continuous_u,
            "log_continuous_cell": log_continuous_cell,
            "log_weight": log_weight,
            "log_xi": log_xi,
            "normalized_weight": normalized_weight,
            "ess": ess,
            "valid_domain": valid_domain,
        }
    return result
