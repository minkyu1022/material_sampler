from __future__ import annotations

import torch

from src.crystalite.edm_utils import karras_sigma_steps
from src.models.lattice_repr import ltri_params_to_lattice_matrix


def sample_packora_time(batch_size: int, device: torch.device | str) -> torch.Tensor:
    """Sample 0.98 Beta(1.9, 1) + 0.02 Uniform(0, 1)."""
    choose_uniform = torch.rand(batch_size, device=device) < 0.02
    beta = torch.rand(batch_size, device=device).pow(1.0 / 1.9)
    uniform = torch.rand(batch_size, device=device)
    return torch.where(choose_uniform, uniform, beta)


def torus_delta(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Minimum-image fractional displacement in [-0.5, 0.5)."""
    return torch.remainder(target - source + 0.5, 1.0) - 0.5


def linear_interpolant(
    prior_frac: torch.Tensor,
    clean_frac: torch.Tensor,
    prior_lattice: torch.Tensor,
    clean_lattice: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_atom = t[:, None, None]
    t_cell = t[:, None]
    frac_t = torch.remainder(prior_frac + t_atom * torus_delta(clean_frac, prior_frac), 1.0)
    lattice_t = (1.0 - t_cell) * prior_lattice + t_cell * clean_lattice
    return frac_t, lattice_t


def endpoint_velocity(
    endpoint_frac: torch.Tensor,
    current_frac: torch.Tensor,
    endpoint_lattice: torch.Tensor,
    current_lattice: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    remaining = (1.0 - t).clamp_min(eps)
    return (
        torus_delta(endpoint_frac, current_frac) / remaining[:, None, None],
        (endpoint_lattice - current_lattice) / remaining[:, None],
    )


def weighted_endpoint_l1(
    predicted_frac: torch.Tensor,
    clean_frac: torch.Tensor,
    predicted_lattice: torch.Tensor,
    clean_lattice: torch.Tensor,
    pad_mask: torch.Tensor,
    coord_weight: float = 10.0,
    lattice_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    real = (~pad_mask.bool()).unsqueeze(-1)
    coord_per_sample = (torus_delta(predicted_frac, clean_frac).abs() * real).sum((1, 2))
    coord_per_sample = coord_per_sample / (3.0 * real.sum((1, 2)).clamp_min(1))
    lattice_per_sample = (predicted_lattice - clean_lattice).abs().sum(1) / 6.0
    coord = coord_per_sample.mean()
    lattice = lattice_per_sample.mean()
    return {
        "loss_coord": coord,
        "loss_lattice": lattice,
        "loss_total": coord_weight * coord + lattice_weight * lattice,
    }


def masked_center(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    real = (~pad_mask.bool()).unsqueeze(-1)
    count = real.sum(1, keepdim=True).clamp_min(1)
    mean = (x * real).sum(1, keepdim=True) / count
    return torch.where(real, x - mean, torch.zeros_like(x))


@torch.no_grad()
def vfm_sampler(
    model,
    type_features: torch.Tensor,
    pad_mask: torch.Tensor,
    num_steps: int,
    generator: torch.Generator | None = None,
    autocast_dtype: torch.dtype | None = None,
    coordinate_repr: str = "fractional",
    coord_std: torch.Tensor | None = None,
    lattice_mean: torch.Tensor | None = None,
    lattice_std: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Euler rollout for the clean-endpoint VFM head used during training."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    device = pad_mask.device
    bsz, nmax = pad_mask.shape
    real = (~pad_mask.bool()).unsqueeze(-1)
    times = torch.arange(num_steps, device=device, dtype=torch.float32) / num_steps

    if coordinate_repr == "fractional":
        x = torch.remainder(
            torch.randn((bsz, nmax, 3), device=device, generator=generator), 1.0
        )
        lattice = torch.randn((bsz, 6), device=device, generator=generator)
        x = torch.where(real, x, torch.zeros_like(x))
        for step, t_scalar in enumerate(times):
            t = t_scalar.expand(bsz)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                endpoint = model(
                    type_features, x, lattice, pad_mask, t,
                    lattice_bias_feats=lattice, gem_sigma=1.0 - t,
                )
            dt = 1.0 / num_steps
            remaining = (1.0 - t_scalar).clamp_min(dt)
            velocity_x = torus_delta(endpoint["coord_vel"], x) / remaining
            velocity_lattice = (endpoint["lattice_vel"] - lattice) / remaining
            trial_x = torch.remainder(
                x + dt * velocity_x, 1.0
            )
            trial_lattice = lattice + dt * velocity_lattice
            if step < num_steps - 1:
                t_next = (t_scalar + dt).expand(bsz)
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    endpoint_next = model(
                        type_features, trial_x, trial_lattice, pad_mask, t_next,
                        lattice_bias_feats=trial_lattice, gem_sigma=1.0 - t_next,
                    )
                remaining_next = 1.0 - t_scalar - dt
                velocity_x_next = torus_delta(endpoint_next["coord_vel"], trial_x) / remaining_next
                velocity_lattice_next = (endpoint_next["lattice_vel"] - trial_lattice) / remaining_next
                x = torch.remainder(x + dt * (velocity_x + velocity_x_next) / 2, 1.0)
                lattice = lattice + dt * (velocity_lattice + velocity_lattice_next) / 2
            else:
                x, lattice = trial_x, trial_lattice
        return {
            "frac": torch.where(real, x, torch.zeros_like(x)),
            "lat": lattice,
        }

    if coordinate_repr != "cartesian":
        raise ValueError(f"unsupported coordinate_repr: {coordinate_repr}")
    if coord_std is None or lattice_mean is None or lattice_std is None:
        raise ValueError("Cartesian VFM requires normalization tensors")
    coord_std = coord_std.to(device=device).view(1, 1, 3)
    lattice_mean = lattice_mean.to(device=device).view(1, 6)
    lattice_std = lattice_std.to(device=device).view(1, 6)
    x = masked_center(
        torch.randn((bsz, nmax, 3), device=device, generator=generator), pad_mask
    )
    lattice = torch.randn((bsz, 6), device=device, generator=generator)
    for t_scalar in times:
        t = t_scalar.expand(bsz)
        physical_latent = lattice * lattice_std + lattice_mean
        cell = ltri_params_to_lattice_matrix(physical_latent)
        physical_coord = x * coord_std
        geometry_frac = torch.linalg.solve(
            cell.transpose(-1, -2), physical_coord.transpose(-1, -2)
        ).transpose(-1, -2)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            endpoint = model(
                type_features, x, lattice, pad_mask, t,
                lattice_bias_feats=physical_latent,
                gem_sigma=1.0 - t,
                geometry_frac_coords=geometry_frac,
            )
        dt = 1.0 / num_steps
        remaining = (1.0 - t_scalar).clamp_min(dt)
        target_x = masked_center(endpoint["coord_vel"].float(), pad_mask)
        x = masked_center(x + dt * (target_x - x) / remaining, pad_mask)
        lattice = lattice + dt * (endpoint["lattice_vel"].float() - lattice) / remaining
    physical_latent = lattice * lattice_std + lattice_mean
    cell = ltri_params_to_lattice_matrix(physical_latent)
    physical_coord = x * coord_std
    frac = torch.linalg.solve(
        cell.transpose(-1, -2), physical_coord.transpose(-1, -2)
    ).transpose(-1, -2)
    return {
        "frac": torch.where(real, torch.remainder(frac, 1.0), torch.zeros_like(frac)),
        "lat": physical_latent,
    }


@torch.no_grad()
def cartesian_vfm_edm_heun_sampler(
    model,
    type_features: torch.Tensor,
    pad_mask: torch.Tensor,
    num_steps: int,
    coord_std: torch.Tensor,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    s_churn: float = 60.0,
    s_min: float = 0.0,
    s_max: float = 999.0,
    s_noise: float = 1.003,
    generator: torch.Generator | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Packora endpoint model sampled with its selected EDM--Heun recipe."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    device = pad_mask.device
    bsz, nmax = pad_mask.shape
    real = (~pad_mask.bool()).unsqueeze(-1)
    coord_std = coord_std.to(device=device).view(1, 1, 3)
    lattice_mean = lattice_mean.to(device=device).view(1, 6)
    lattice_std = lattice_std.to(device=device).view(1, 6)
    sigmas = karras_sigma_steps(
        num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max,
        rho=rho, device=device,
    )
    coord = masked_center(
        torch.randn((bsz, nmax, 3), device=device, generator=generator), pad_mask
    ) * sigmas[0]
    lattice = torch.randn((bsz, 6), device=device, generator=generator) * sigmas[0]

    def endpoint(z_coord, z_lattice, sigma):
        # z = clean + sigma * noise and t = 1/(1+sigma), hence y_t = t*z.
        t_scalar = 1.0 / (1.0 + sigma)
        t = t_scalar.expand(bsz)
        y_coord = masked_center(t_scalar * z_coord, pad_mask)
        y_lattice = t_scalar * z_lattice
        physical_latent = y_lattice * lattice_std + lattice_mean
        cell = ltri_params_to_lattice_matrix(physical_latent)
        geometry_frac = torch.linalg.solve(
            cell.transpose(-1, -2), (y_coord * coord_std).transpose(-1, -2)
        ).transpose(-1, -2)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            pred = model(
                type_features, y_coord, y_lattice, pad_mask, t,
                lattice_bias_feats=physical_latent,
                gem_sigma=1.0 - t,
                geometry_frac_coords=geometry_frac,
            )
        return masked_center(pred["coord_vel"].float(), pad_mask), pred["lattice_vel"].float()

    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        gamma = min(s_churn / num_steps, 2.0**0.5 - 1.0) if s_min <= sigma_cur <= s_max else 0.0
        sigma_hat = sigma_cur * (1.0 + gamma)
        noise = (sigma_hat.square() - sigma_cur.square()).clamp_min(0).sqrt() * s_noise
        coord_hat = coord + noise * masked_center(
            torch.randn(coord.shape, device=device, generator=generator), pad_mask
        )
        lattice_hat = lattice + noise * torch.randn(
            lattice.shape, device=device, generator=generator
        )
        den_coord, den_lattice = endpoint(coord_hat, lattice_hat, sigma_hat)
        d_coord = (coord_hat - den_coord) / sigma_hat
        d_lattice = (lattice_hat - den_lattice) / sigma_hat
        coord = coord_hat + (sigma_next - sigma_hat) * d_coord
        lattice = lattice_hat + (sigma_next - sigma_hat) * d_lattice
        if i < num_steps - 1:
            den_coord_2, den_lattice_2 = endpoint(coord, lattice, sigma_next)
            d_coord_2 = (coord - den_coord_2) / sigma_next
            d_lattice_2 = (lattice - den_lattice_2) / sigma_next
            coord = coord_hat + (sigma_next - sigma_hat) * (d_coord + d_coord_2) / 2
            lattice = lattice_hat + (sigma_next - sigma_hat) * (d_lattice + d_lattice_2) / 2
        coord = masked_center(coord, pad_mask)

    physical_latent = lattice * lattice_std + lattice_mean
    cell = ltri_params_to_lattice_matrix(physical_latent)
    frac = torch.linalg.solve(
        cell.transpose(-1, -2), (coord * coord_std).transpose(-1, -2)
    ).transpose(-1, -2)
    return {
        "frac": torch.where(real, torch.remainder(frac, 1.0), torch.zeros_like(frac)),
        "lat": physical_latent,
    }


def compute_cartesian_vfm_loss(
    model,
    type_features: torch.Tensor,
    clean_frac: torch.Tensor,
    clean_lattice: torch.Tensor,
    pad_mask: torch.Tensor,
    t: torch.Tensor,
    coord_std: torch.Tensor,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    temperature_k: torch.Tensor | None = None,
    temperature_present: torch.Tensor | None = None,
    coord_weight: float = 10.0,
    lattice_weight: float = 1.0,
    augment_translation: bool = True,
) -> dict[str, torch.Tensor]:
    """Packora-style centered-Cartesian VFM with physical PBC geometry."""
    if clean_lattice.shape[-1] != 6:
        raise ValueError("Cartesian VFM requires six-dimensional ltri cell latents")
    coord_std = coord_std.to(clean_frac).view(1, 1, 3).clamp_min(1e-8)
    lattice_mean = lattice_mean.to(clean_lattice).view(1, 6)
    lattice_std = lattice_std.to(clean_lattice).view(1, 6).clamp_min(1e-8)
    frac = clean_frac
    if augment_translation:
        frac = torch.remainder(frac + torch.rand_like(frac[:, :1]), 1.0)
    clean_lattice_matrix = ltri_params_to_lattice_matrix(clean_lattice)
    clean_coord = masked_center(torch.einsum("bni,bij->bnj", frac, clean_lattice_matrix), pad_mask)
    clean_coord = clean_coord / coord_std
    clean_lattice_norm = (clean_lattice - lattice_mean) / lattice_std

    prior_coord = masked_center(torch.randn_like(clean_coord), pad_mask)
    prior_lattice = torch.randn_like(clean_lattice_norm)
    t_atom, t_cell = t[:, None, None], t[:, None]
    coord_t = (1.0 - t_atom) * prior_coord + t_atom * clean_coord
    lattice_t = (1.0 - t_cell) * prior_lattice + t_cell * clean_lattice_norm

    lattice_t_physical = lattice_t * lattice_std + lattice_mean
    lattice_t_matrix = ltri_params_to_lattice_matrix(lattice_t_physical)
    coord_t_physical = coord_t * coord_std
    geometry_frac = torch.linalg.solve(
        lattice_t_matrix.transpose(-1, -2), coord_t_physical.transpose(-1, -2)
    ).transpose(-1, -2)
    raw = model(
        type_features, coord_t, lattice_t, pad_mask, t,
        lattice_bias_feats=lattice_t_physical,
        temperature_k=temperature_k,
        temperature_present=temperature_present,
        gem_sigma=1.0 - t,
        geometry_frac_coords=geometry_frac,
    )
    predicted_coord = masked_center(raw["coord_vel"], pad_mask)
    real = (~pad_mask.bool()).unsqueeze(-1)
    coord = ((predicted_coord - clean_coord).abs() * real).sum((1, 2))
    coord = (coord / (3.0 * real.sum((1, 2)).clamp_min(1))).mean()
    lattice = (raw["lattice_vel"] - clean_lattice_norm).abs().mean()
    return {
        "loss_coord": coord,
        "loss_lattice": lattice,
        "loss_total": coord_weight * coord + lattice_weight * lattice,
        "predicted_coord": predicted_coord,
        "predicted_lattice": raw["lattice_vel"],
        "clean_coord": clean_coord,
        "clean_lattice": clean_lattice_norm,
        "coord_t": coord_t,
        "lattice_t": lattice_t,
        "t": t,
    }


def compute_vfm_loss(
    model,
    type_features: torch.Tensor,
    clean_frac: torch.Tensor,
    clean_lattice: torch.Tensor,
    pad_mask: torch.Tensor,
    t: torch.Tensor,
    temperature_k: torch.Tensor | None = None,
    temperature_present: torch.Tensor | None = None,
    coord_weight: float = 10.0,
    lattice_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Run one Packora-style VFM batch with wrapped-Gaussian fractional prior."""
    prior_frac = torch.remainder(torch.randn_like(clean_frac), 1.0)
    prior_lattice = torch.randn_like(clean_lattice)
    frac_t, lattice_t = linear_interpolant(
        prior_frac, clean_frac, prior_lattice, clean_lattice, t
    )
    frac_t = torch.where(pad_mask[..., None], torch.zeros_like(frac_t), frac_t)
    raw = model(
        type_features,
        frac_t,
        lattice_t,
        pad_mask,
        t,
        lattice_bias_feats=lattice_t,
        temperature_k=temperature_k,
        temperature_present=temperature_present,
        gem_sigma=1.0 - t,
    )
    losses = weighted_endpoint_l1(
        raw["coord_vel"],
        clean_frac,
        raw["lattice_vel"],
        clean_lattice,
        pad_mask,
        coord_weight,
        lattice_weight,
    )
    return {
        **losses,
        "predicted_frac": raw["coord_vel"],
        "predicted_lattice": raw["lattice_vel"],
        "frac_t": frac_t,
        "lattice_t": lattice_t,
        "t": t,
    }
