"""Fixed-composition 2D-tetragonal reference MC for unified Ni--Cr."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .cuni import KB_EV_K
from .nicr_unified_bct2d import (
    N_ATOMS,
    CellNormalization,
    cell_matrix,
    reference_sites,
    transformed_target_log_density,
)
from .torch_eam import TorchEAM


@dataclass(frozen=True)
class ReferenceMCConfig:
    sweeps: int = 100
    burn_in: int = 50
    thin: int = 5
    species_moves: int = 6
    displacement_step: float = 2e-4
    cell_z_step: float = 0.01
    bain_moves: int = 1
    bain_z_step: float = 0.05
    ratio_center: float | None = None
    ratio_bias: float = 0.0


def _accept(log_ratio: Tensor, generator: torch.Generator | None) -> bool:
    return bool(
        log_ratio >= 0
        or torch.rand((), device=log_ratio.device, generator=generator).log() < log_ratio
    )


@torch.no_grad()
def reference_mc(
    oracle: TorchEAM,
    target_cr: int,
    temperature: float,
    config: ReferenceMCConfig | None = None,
    *,
    normalization: CellNormalization | None = None,
    initial_cell_ac: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | dict[str, float]]:
    """Sample the transformed restricted-NPT target in ``(species,u,z_cell)``."""
    config = config or ReferenceMCConfig()
    normalization = normalization or CellNormalization()
    if not 0 <= target_cr <= N_ATOMS or temperature <= 0:
        raise ValueError("require a valid Cr count and positive temperature")
    device = oracle.coefficients.device
    dtype = oracle.coefficients.dtype
    species = torch.zeros(N_ATOMS, dtype=torch.long, device=device)
    species[torch.randperm(N_ATOMS, generator=generator, device=device)[:target_cr]] = 1
    disp_u = torch.zeros(N_ATOMS, 3, dtype=dtype, device=device)
    cell_z = (
        torch.zeros(2, dtype=dtype, device=device)
        if initial_cell_ac is None
        else normalization.encode(torch.as_tensor(initial_cell_ac, dtype=dtype, device=device))
    )
    reference = reference_sites(dtype=dtype).to(device)
    beta = torch.as_tensor(1 / (KB_EV_K * temperature), dtype=dtype, device=device)

    def target_density(current_species: Tensor, current_u: Tensor, current_z: Tensor) -> Tensor:
        cell_ac = normalization.decode(current_z)
        energy = oracle.forward_cell(current_species, reference + current_u, cell_matrix(cell_ac))
        return transformed_target_log_density(energy, cell_ac, beta, disp_u=current_u)

    def density(current_species: Tensor, current_u: Tensor, current_z: Tensor) -> Tensor:
        value = target_density(current_species, current_u, current_z)
        if config.ratio_center is not None:
            cell_ac = normalization.decode(current_z)
            ratio = cell_ac[1] / cell_ac[0]
            value -= 0.5 * config.ratio_bias * (ratio - config.ratio_center) ** 2
        return value

    log_density = density(species, disp_u, cell_z)
    accepted = {"displacement": 0, "cell": 0, "bain": 0, "species": 0}
    attempted = {"displacement": 0, "cell": 0, "bain": 0, "species": 0}
    samples = {
        "species": [], "disp_u": [], "cell_z": [],
        "log_density": [], "sampling_log_density": [],
    }
    for sweep in range(config.burn_in + config.sweeps):
        trial_u = disp_u + config.displacement_step * torch.randn(
            disp_u.shape, dtype=dtype, device=device, generator=generator
        )
        trial_u -= trial_u.mean(0, keepdim=True)
        trial_density = density(species, trial_u, cell_z)
        attempted["displacement"] += 1
        if _accept(trial_density - log_density, generator):
            disp_u, log_density = trial_u, trial_density
            accepted["displacement"] += 1

        trial_z = cell_z + config.cell_z_step * torch.randn(
            2, dtype=dtype, device=device, generator=generator
        )
        trial_density = density(species, disp_u, trial_z)
        attempted["cell"] += 1
        if _accept(trial_density - log_density, generator):
            cell_z, log_density = trial_z, trial_density
            accepted["cell"] += 1

        for _ in range(config.bain_moves):
            # q(z'|z)=N(z';-z,sigma^2 I) is exactly symmetric because z'+z=x'+x.
            trial_z = -cell_z + config.bain_z_step * torch.randn(
                2, dtype=dtype, device=device, generator=generator
            )
            trial_density = density(species, disp_u, trial_z)
            attempted["bain"] += 1
            if _accept(trial_density - log_density, generator):
                cell_z, log_density = trial_z, trial_density
                accepted["bain"] += 1

        for _ in range(config.species_moves):
            ni, cr = torch.where(species.eq(0))[0], torch.where(species.eq(1))[0]
            if not len(ni) or not len(cr):
                break
            i = ni[torch.randint(len(ni), (), generator=generator, device=device)]
            j = cr[torch.randint(len(cr), (), generator=generator, device=device)]
            trial_species = species.clone()
            trial_species[i], trial_species[j] = 1, 0
            trial_density = density(trial_species, disp_u, cell_z)
            attempted["species"] += 1
            if _accept(trial_density - log_density, generator):
                species, log_density = trial_species, trial_density
                accepted["species"] += 1

        if sweep >= config.burn_in and (sweep - config.burn_in) % config.thin == 0:
            samples["species"].append(species.clone())
            samples["disp_u"].append(disp_u.clone())
            samples["cell_z"].append(cell_z.clone())
            samples["log_density"].append(target_density(species, disp_u, cell_z))
            samples["sampling_log_density"].append(log_density.clone())
    result = {key: torch.stack(value) for key, value in samples.items()}
    result["cell_ac"] = normalization.decode(result["cell_z"])
    result["stats"] = {
        key: accepted[key] / attempted[key] if attempted[key] else math.nan for key in accepted
    }
    return result
