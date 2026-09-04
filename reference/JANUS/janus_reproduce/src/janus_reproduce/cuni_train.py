"""Distributed Cu--Ni JANUS rollout, EAM labeling, and replay training."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
from ase.build import bulk
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .alloy_model import AlloyPaiNN
from .cuni import KB_EV_K, VolumePrior
from .free_energy import (
    cuni_prior_log_density,
    gaussian_path_log_ratio,
    path_weight_estimates,
)
from .losses import CONT_LOSS_REGISTRY, DISC_LOSS_REGISTRY, get_loss
from .objective import bounded_score_target, interpolate, mask_terminal
from .samplers import get_discrete_sampler
from .torch_eam import TorchCuNiEAM


@dataclass(frozen=True)
class CuNiTrainConfig:
    potential: Path
    output: Path = Path("outputs/cuni_train")
    prior: Path | None = None
    n_atoms: int = 108
    steps: int = 100
    initial_buffer: int = 5_000
    rounds: int = 120
    fresh_per_round: int = 1_000
    replay_size: int = 5_000
    updates_per_round: int = 500
    global_batch: int = 96
    rollout_batch: int = 12
    temperature_min: float = 600.0
    temperature_max: float = 1_200.0
    delta_mu_wide: float = 0.30
    delta_mu_narrow: float = 0.06
    learning_rate: float = 3e-3
    continuous_loss: str = "tsm"
    discrete_loss: str = "sce"
    continuous_weight_u: float = 1.0
    continuous_weight_v: float = 1.0
    discrete_weight: float = 2.0
    discrete_sampler: str = "janus_tau_leap"
    features: int = 64
    layers: int = 4
    radial_basis: int = 16
    cutoff: float = 5.0
    sigma_u_ref: float = 0.004
    sigma_u_exponent: float = 0.5
    use_fitted_sigma_u_exponent: bool = False
    sigma_v_scale: float = 1.0
    diffusion_u: float = 0.0
    diffusion_v: float = 0.0
    diffusion_temperature_ref: float = 900.0
    target_score_u_clip: float | None = 100.0
    target_score_v_clip: float | None = 1_000.0
    rollout_velocity_clip: float | None = 0.1
    rollout_score_clip: float | None = 1_000.0
    gradient_clip_norm: float = 1.0
    optimizer: str = "adam"
    weight_decay: float = 0.0
    warmup_updates: int = 0
    minimum_learning_rate: float = 0.0
    seed: int = 2026
    bf16: bool = False
    resume: bool = True
    wandb_project: str | None = "janus-reproduce"
    checkpoint_version: int = 3

    @classmethod
    def smoke(cls, potential: Path, output: Path, **overrides) -> CuNiTrainConfig:
        values = {
            "potential": potential,
            "output": output,
            "n_atoms": 4,
            "steps": 2,
            "initial_buffer": 4,
            "rounds": 1,
            "fresh_per_round": 2,
            "replay_size": 4,
            "updates_per_round": 1,
            "global_batch": 2,
            "rollout_batch": 2,
            "features": 8,
            "layers": 1,
            "radial_basis": 4,
            "wandb_project": None,
        }
        values.update(overrides)
        return cls(**values)


class ReplayBuffer:
    fields = (
        "species",
        "displacement",
        "log_volume",
        "temperature",
        "delta_mu",
        "score_u",
        "score_v",
        "heat_bath",
        "energy",
        "score_u_clip_fraction",
    )

    def __init__(self, data: dict[str, Tensor] | None = None):
        self.data = {key: value.cpu() for key, value in (data or {}).items()}

    def __len__(self) -> int:
        return 0 if not self.data else len(self.data["species"])

    def add(self, values: dict[str, Tensor], limit: int) -> None:
        combined = {
            key: torch.cat((self.data[key], value.cpu()), 0) if self.data else value.cpu()
            for key, value in values.items()
        }
        if len(combined["species"]) > limit:
            keep = torch.randperm(len(combined["species"]))[:limit]
            combined = {key: value[keep] for key, value in combined.items()}
        self.data = combined

    def sample(self, count: int, device: torch.device) -> dict[str, Tensor]:
        index = torch.randint(len(self), (count,))
        return {key: value[index].to(device) for key, value in self.data.items()}


def _distributed() -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        if device.type == "cuda":
            dist.init_process_group("nccl", device_id=device)
        else:
            dist.init_process_group("gloo")
    return rank, world, device


def _split(total: int, rank: int, world: int) -> int:
    return total // world + (rank < total % world)


def _reference(n_atoms: int, device: torch.device) -> Tensor:
    repeats = round((n_atoms / 4) ** (1 / 3))
    if 4 * repeats**3 != n_atoms:
        raise ValueError("n_atoms must be a cubic repetition of the four-atom fcc cell")
    atoms = bulk("Ni", "fcc", a=3.55, cubic=True).repeat((repeats,) * 3)
    return torch.tensor(atoms.get_scaled_positions(wrap=False), dtype=torch.float32, device=device)


def _load_prior(config: CuNiTrainConfig) -> VolumePrior:
    if config.prior is None:
        return VolumePrior(3.52**3 / 4, 3.615**3 / 4, 0.0, 0.0, 0.02)
    import json

    values = json.loads(config.prior.read_text())
    return VolumePrior(*(values[key] for key in VolumePrior.__dataclass_fields__))


def _conditions(config: CuNiTrainConfig, count: int, device: torch.device) -> tuple[Tensor, Tensor]:
    inverse_temperature = torch.empty(count, device=device).uniform_(
        1 / config.temperature_max, 1 / config.temperature_min
    )
    temperature = inverse_temperature.reciprocal()
    center = 0.893 - 5.4e-5 * temperature
    width = torch.where(
        torch.rand(count, device=device) < 0.5, config.delta_mu_narrow, config.delta_mu_wide
    )
    return temperature, center + (2 * torch.rand(count, device=device) - 1) * width


def _prior_values(
    config: CuNiTrainConfig,
    prior: VolumePrior,
    temperature: Tensor,
    delta_mu: Tensor,
    atoms: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    p_cu = torch.sigmoid((delta_mu - (0.893 - 5.4e-5 * temperature)) / (KB_EV_K * temperature))
    sigma_u = config.sigma_u_ref * (temperature / prior.temperature_ref).pow(
        config.sigma_u_exponent
    )
    displacement = torch.randn(len(temperature), atoms, 3, device=temperature.device)
    displacement *= sigma_u[:, None, None]
    displacement -= displacement.mean(1, keepdim=True)
    atomic_volume = (
        (1 - p_cu) * prior.v_ni + p_cu * prior.v_cu + prior.omega * p_cu * (1 - p_cu)
    ) * (1 + prior.alpha * (temperature - prior.temperature_ref))
    mean_v = (atoms * atomic_volume).log()
    log_volume = mean_v + config.sigma_v_scale * prior.sigma_log_volume * torch.randn_like(mean_v)
    return displacement, log_volume, sigma_u, mean_v


def _autocast(config: CuNiTrainConfig, device: torch.device):
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda"
    )


def _euler_maruyama(
    state: Tensor, velocity: Tensor, score: Tensor, diffusion: Tensor, dt: float, noise: Tensor
) -> Tensor:
    while diffusion.ndim < state.ndim:
        diffusion = diffusion.unsqueeze(-1)
    return (
        state
        + (velocity + diffusion.square() * score) * dt
        + (2 * diffusion.square() * dt).sqrt() * noise
    )


def _clip_field(value: Tensor, limit: float | None, *, vector: bool) -> Tensor:
    if limit is None:
        return value
    if vector:
        norm = value.norm(dim=-1, keepdim=True)
        return value * (limit / norm.clamp_min(1e-12)).clamp_max(1)
    return value.clamp(-limit, limit)


@torch.no_grad()
def _rollout(
    model: nn.Module,
    config: CuNiTrainConfig,
    prior: VolumePrior,
    reference: Tensor,
    count: int,
    device: torch.device,
    *,
    path_weights: bool = False,
    oracle: TorchCuNiEAM | None = None,
    temperature: Tensor | None = None,
    delta_mu: Tensor | None = None,
    path_weight_trace: bool = False,
) -> dict[str, Tensor]:
    if path_weight_trace and not path_weights:
        raise ValueError("path_weight_trace requires path_weights")
    if path_weights and (oracle is None or config.diffusion_u <= 0 or config.diffusion_v <= 0):
        raise ValueError("path weights require an oracle and positive diffusion_u/diffusion_v")
    discrete_sampler = get_discrete_sampler(config.discrete_sampler)
    if config.discrete_sampler != "janus_tau_leap":
        raise ValueError("Cu-Ni requires the reproduced janus_tau_leap sampler")
    chunks: dict[str, list[Tensor]] = {
        key: [] for key in ("species", "displacement", "log_volume", "temperature", "delta_mu")
    }
    if path_weights:
        chunks |= {
            key: []
            for key in (
                "log_prior", "log_target", "log_q_discrete", "log_continuous_u",
                "log_continuous_v", "log_continuous_ratio", "log_prior_u", "log_prior_v",
                "log_target_energy", "log_target_chemical", "log_target_volume",
            )
        }
        if path_weight_trace:
            chunks["log_continuous_u_steps"] = []
    for start in range(0, count, config.rollout_batch):
        size = min(config.rollout_batch, count - start)
        if temperature is None or delta_mu is None:
            batch_temperature, batch_delta_mu = _conditions(config, size, device)
        else:
            batch_temperature = torch.as_tensor(
                temperature, device=device, dtype=reference.dtype
            ).reshape(-1)
            batch_delta_mu = torch.as_tensor(
                delta_mu, device=device, dtype=reference.dtype
            ).reshape(-1)
            batch_temperature = batch_temperature.expand(count)[start : start + size]
            batch_delta_mu = batch_delta_mu.expand(count)[start : start + size]
        displacement, log_volume, sigma_u, mean_v = _prior_values(
            config, prior, batch_temperature, batch_delta_mu, config.n_atoms
        )
        initial_displacement, initial_log_volume = displacement.clone(), log_volume.clone()
        species = torch.full((size, config.n_atoms), 2, dtype=torch.long, device=device)
        log_q_discrete = torch.zeros(size, device=device, dtype=torch.float64)
        log_continuous_u = torch.zeros(size, device=device, dtype=torch.float64)
        log_continuous_v = torch.zeros(size, device=device, dtype=torch.float64)
        log_continuous_u_steps = []
        for step in range(config.steps):
            t0, t1 = step / config.steps, (step + 1) / config.steps
            with _autocast(config, device):
                output = model(
                    species,
                    displacement,
                    log_volume,
                    reference,
                    t0,
                    batch_temperature,
                    batch_delta_mu,
                )
            if not all(torch.isfinite(value).all() for value in output):
                raise FloatingPointError(f"non-finite model output at rollout step {step}")
            b_u = _clip_field(output.b_u.float(), config.rollout_velocity_clip, vector=True)
            s_u = _clip_field(output.s_u.float(), config.rollout_score_clip, vector=True)
            b_v = _clip_field(output.b_v.float(), config.rollout_velocity_clip, vector=False)
            s_v = _clip_field(output.s_v.float(), config.rollout_score_clip, vector=False)
            dt = t1 - t0
            temperature_scale = (batch_temperature / config.diffusion_temperature_ref).sqrt()
            g_u = config.diffusion_u * temperature_scale
            g_v = config.diffusion_v * temperature_scale
            previous_displacement, previous_log_volume = displacement, log_volume
            displacement = _euler_maruyama(
                displacement,
                b_u,
                s_u,
                g_u,
                dt,
                torch.randn_like(displacement),
            )
            displacement -= displacement.mean(1, keepdim=True)
            log_volume = _euler_maruyama(
                log_volume,
                b_v,
                s_v,
                g_v,
                dt,
                torch.randn_like(log_volume),
            )
            if not (torch.isfinite(displacement).all() and torch.isfinite(log_volume).all()):
                raise FloatingPointError(f"non-finite continuous state at rollout step {step}")
            if path_weights:
                species, delta_log_q = discrete_sampler(
                    species, output.species_logits.float(), t0, t1, 2
                )
                log_q_discrete += delta_log_q
                with _autocast(config, device):
                    backward = model(
                        species,
                        displacement,
                        log_volume,
                        reference,
                        t1,
                        batch_temperature,
                        batch_delta_mu,
                    )
                backward_b_u = _clip_field(
                    backward.b_u.float(), config.rollout_velocity_clip, vector=True
                )
                backward_s_u = _clip_field(
                    backward.s_u.float(), config.rollout_score_clip, vector=True
                )
                backward_b_v = _clip_field(
                    backward.b_v.float(), config.rollout_velocity_clip, vector=False
                )
                backward_s_v = _clip_field(
                    backward.s_v.float(), config.rollout_score_clip, vector=False
                )
                delta_log_u = gaussian_path_log_ratio(
                    previous_displacement,
                    displacement,
                    b_u - b_u.mean(1, keepdim=True),
                    s_u - s_u.mean(1, keepdim=True),
                    backward_b_u - backward_b_u.mean(1, keepdim=True),
                    backward_s_u - backward_s_u.mean(1, keepdim=True),
                    g_u.square(),
                    dt,
                )
                log_continuous_u += delta_log_u
                if path_weight_trace:
                    log_continuous_u_steps.append(delta_log_u)
                log_continuous_v += gaussian_path_log_ratio(
                    previous_log_volume,
                    log_volume,
                    b_v,
                    s_v,
                    backward_b_v,
                    backward_s_v,
                    g_v.square(),
                    dt,
                )
            else:
                species = discrete_sampler(
                    species, output.species_logits.float(), t0, t1, 2
                )[0]
        values: dict[str, Tensor] = {
            "species": species,
            "displacement": displacement,
            "log_volume": log_volume,
            "temperature": batch_temperature,
            "delta_mu": batch_delta_mu,
        }
        if path_weights:
            assert oracle is not None
            energy = oracle(species, reference[None] + displacement, log_volume).double()
            log_prior = cuni_prior_log_density(
                initial_displacement,
                initial_log_volume,
                sigma_u,
                mean_v,
                config.sigma_v_scale * prior.sigma_log_volume,
            )
            u_dimensions = 3 * (config.n_atoms - 1)
            log_prior_u = -0.5 * (
                initial_displacement.square().sum((-2, -1)) / sigma_u.square()
                + u_dimensions * torch.log(2 * torch.pi * sigma_u.square())
            )
            beta = 1 / (KB_EV_K * batch_temperature)
            log_target_energy = -beta * energy
            log_target_chemical = beta * batch_delta_mu * species.eq(0).sum(-1)
            log_target_volume = config.n_atoms * log_volume
            values |= {
                "log_prior": log_prior,
                "log_prior_u": log_prior_u,
                "log_prior_v": log_prior - log_prior_u,
                "log_target": log_target_energy + log_target_chemical + log_target_volume,
                "log_target_energy": log_target_energy,
                "log_target_chemical": log_target_chemical,
                "log_target_volume": log_target_volume,
                "log_q_discrete": log_q_discrete,
                "log_continuous_u": log_continuous_u,
                "log_continuous_v": log_continuous_v,
                "log_continuous_ratio": log_continuous_u + log_continuous_v,
            }
            if path_weight_trace:
                values["log_continuous_u_steps"] = torch.stack(log_continuous_u_steps, 1)
        for key, value in values.items():
            chunks[key].append(value.detach())
    result = {key: torch.cat(value) for key, value in chunks.items()}
    if path_weights:
        result["log_weight"] = (
            result["log_target"]
            - result["log_prior"]
            - result["log_q_discrete"]
            + result["log_continuous_ratio"]
        )
        result["log_xi"], result["normalized_weight"], result["ess"] = path_weight_estimates(
            result["log_weight"]
        )
    return result


def _label(
    oracle: TorchCuNiEAM,
    states: dict[str, Tensor],
    config: CuNiTrainConfig,
    reference: Tensor,
) -> dict[str, Tensor]:
    output = {
        key: [] for key in ("score_u", "score_v", "heat_bath", "energy", "score_u_clip_fraction")
    }
    for index in range(len(states["species"])):
        species = states["species"][index : index + 1]
        displacement = states["displacement"][index : index + 1].double()
        log_volume = states["log_volume"][index : index + 1].double()
        fractional = displacement + reference.double()[None]
        labels = oracle.labels(species, fractional, log_volume)
        with torch.no_grad():
            site_energy = oracle.all_site_energies(species, fractional, log_volume)
            beta = 1 / (KB_EV_K * states["temperature"][index].double())
            # u is fractional: r = u L for the cubic cell, so ∇_u log p = β F_cart L.
            score_u = beta * labels.forces * log_volume.exp().pow(1 / 3)[:, None, None]
            score_norm = score_u.norm(dim=-1, keepdim=True)
            if config.target_score_u_clip is not None:
                score_u *= (
                    config.target_score_u_clip / score_norm.clamp_min(1e-12)
                ).clamp_max(1)
                clipped = score_norm.gt(config.target_score_u_clip).float().mean((-2, -1))
            else:
                clipped = torch.zeros(1, device=score_norm.device)
            score_u -= score_u.mean(1, keepdim=True)
            score_v = config.n_atoms - beta * labels.log_volume_derivative
            logits = torch.stack(
                (
                    -beta * (site_energy[..., 0] - states["delta_mu"][index]),
                    -beta * site_energy[..., 1],
                ),
                -1,
            )
            output["score_u"].append(score_u.float())
            output["score_v"].append(score_v.float())
            output["heat_bath"].append(logits.softmax(-1).float())
            output["energy"].append(labels.energy.detach().float())
            output["score_u_clip_fraction"].append(clipped.float())
    return states | {key: torch.cat(value) for key, value in output.items()}


def _train_updates(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    replay: ReplayBuffer,
    config: CuNiTrainConfig,
    prior: VolumePrior,
    reference: Tensor,
    local_batch: int,
    device: torch.device,
    completed_updates: int,
) -> float:
    model.train()
    continuous_loss = get_loss(CONT_LOSS_REGISTRY, config.continuous_loss)
    discrete_loss = get_loss(DISC_LOSS_REGISTRY, config.discrete_loss)
    total = 0.0
    total_updates = config.rounds * config.updates_per_round
    for update_index in range(config.updates_per_round):
        step = completed_updates + update_index
        if config.warmup_updates:
            if step < config.warmup_updates:
                learning_rate = config.learning_rate * (step + 1) / config.warmup_updates
            else:
                progress = (step - config.warmup_updates) / max(
                    total_updates - config.warmup_updates - 1, 1
                )
                learning_rate = config.minimum_learning_rate + 0.5 * (
                    config.learning_rate - config.minimum_learning_rate
                ) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        batch = replay.sample(local_batch, device)
        x0_u, x0_v, sigma_u, mean_v = _prior_values(
            config, prior, batch["temperature"], batch["delta_mu"], config.n_atoms
        )
        t = torch.rand(local_batch, device=device)
        u = interpolate(x0_u, batch["displacement"], t)
        v = interpolate(x0_v, batch["log_volume"], t)
        species, masked = mask_terminal(batch["species"], t, 2)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            output = model(
                species,
                u,
                v,
                reference,
                t,
                batch["temperature"],
                batch["delta_mu"],
            )
            target_s_u = bounded_score_target(
                x0_u,
                t,
                batch["score_u"],
                0.0,
                sigma_u[:, None, None].square(),
            )
            target_s_v = bounded_score_target(
                x0_v,
                t,
                _clip_field(batch["score_v"], config.target_score_v_clip, vector=False),
                mean_v,
                (config.sigma_v_scale * prior.sigma_log_volume) ** 2,
            )
            loss = (
                config.continuous_weight_u
                * continuous_loss(
                    output.b_u,
                    batch["displacement"] - x0_u,
                    output.s_u,
                    target_s_u,
                )
                + config.continuous_weight_v
                * continuous_loss(
                    output.b_v,
                    batch["log_volume"] - x0_v,
                    output.s_v,
                    target_s_v,
                )
                + config.discrete_weight
                * discrete_loss(
                    output.species_logits,
                    batch["heat_bath"],
                    masked,
                )
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
    return total / config.updates_per_round


def _checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    replay: ReplayBuffer,
    history: list[dict[str, float]],
    completed_round: int,
    config: CuNiTrainConfig,
    rank: int,
    world: int,
) -> None:
    local = {
        "buffer": replay.data,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    shards = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local, shards, dst=0)
    else:
        shards = [local]
    if rank:
        return
    module = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "shards": shards,
        "world_size": world,
        "round": completed_round,
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_cuni(config: CuNiTrainConfig) -> list[dict[str, float]]:
    if config.prior is not None:
        import json

        calibrated = json.loads(config.prior.read_text()).get("displacement_prior")
        if calibrated:
            config = replace(
                config,
                sigma_u_ref=calibrated["sigma_u_ref"],
                sigma_u_exponent=(
                    calibrated["temperature_exponent"]
                    if config.use_fitted_sigma_u_exponent
                    else 0.5
                ),
            )
    rank, world, device = _distributed()
    if config.global_batch % world:
        raise ValueError("global_batch must be divisible by the torchrun world size")
    if any(
        value % world
        for value in (config.initial_buffer, config.fresh_per_round, config.replay_size)
    ):
        raise ValueError(
            "initial_buffer, fresh_per_round, and replay_size must divide across ranks"
        )
    local_batch = config.global_batch // world
    torch.manual_seed(config.seed + rank)
    model: nn.Module = AlloyPaiNN(
        features=config.features,
        layers=config.layers,
        radial_basis=config.radial_basis,
        cutoff=config.cutoff,
        temperature_reference=config.diffusion_temperature_ref,
    ).to(device)
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(config.optimizer)
    if optimizer_type is None:
        raise ValueError("optimizer must be 'adam' or 'adamw'")
    optimizer = optimizer_type(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.bf16 and device.type == "cuda")
    oracle = TorchCuNiEAM(config.potential).to(device)
    prior = _load_prior(config)
    reference = _reference(config.n_atoms, device)
    replay = ReplayBuffer()
    history: list[dict[str, float]] = []
    checkpoint = config.output / "checkpoint.pt"
    start_round = 0
    if config.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["config"].get("checkpoint_version") != config.checkpoint_version:
            raise ValueError("checkpoint predates the audited sampler; start with --no-resume")
        if saved["world_size"] != world:
            raise ValueError("resume requires the same torchrun world size")
        module = model.module if isinstance(model, DistributedDataParallel) else model
        module.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        replay = ReplayBuffer(saved["shards"][rank]["buffer"])
        torch.set_rng_state(saved["shards"][rank]["rng"].cpu())
        if device.type == "cuda" and saved["shards"][rank]["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["shards"][rank]["cuda_rng"].cpu(), device)
        history, start_round = saved["history"], saved["round"]

    run = None
    if rank == 0 and config.wandb_project:
        import wandb

        run = wandb.init(
            project=config.wandb_project,
            name=config.output.name,
            id=hashlib.sha1(str(config.output.resolve()).encode()).hexdigest()[:12],
            config={
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            resume="allow",
        )

    if not len(replay):
        states = _rollout(
            model, config, prior, reference, _split(config.initial_buffer, rank, world), device
        )
        replay.add(
            _label(oracle, states, config, reference),
            _split(config.replay_size, rank, world),
        )
    for round_index in range(start_round, config.rounds):
        started = time.perf_counter()
        states = _rollout(
            model, config, prior, reference, _split(config.fresh_per_round, rank, world), device
        )
        rollout_seconds = time.perf_counter() - started
        started = time.perf_counter()
        labeled = _label(oracle, states, config, reference)
        label_seconds = time.perf_counter() - started
        replay.add(labeled, _split(config.replay_size, rank, world))
        started = time.perf_counter()
        loss = _train_updates(
            model,
            optimizer,
            scaler,
            replay,
            config,
            prior,
            reference,
            local_batch,
            device,
            round_index * config.updates_per_round,
        )
        train_seconds = time.perf_counter() - started
        metrics = {
            "round": float(round_index + 1),
            "loss": loss,
            "mean_energy": float(labeled["energy"].mean()),
            "mean_cu_fraction": float(labeled["species"].eq(0).float().mean()),
            "score_u_clip_fraction": float(labeled["score_u_clip_fraction"].mean()),
            "rollout_seconds": rollout_seconds,
            "label_seconds": label_seconds,
            "train_seconds": train_seconds,
        }
        history.append(metrics)
        started = time.perf_counter()
        _checkpoint(
            checkpoint,
            model,
            optimizer,
            scaler,
            replay,
            history,
            round_index + 1,
            config,
            rank,
            world,
        )
        metrics["checkpoint_seconds"] = time.perf_counter() - started
        if rank == 0:
            print(" ".join(f"{key}={value:.5g}" for key, value in metrics.items()), flush=True)
            if run:
                run.log(metrics, step=round_index + 1)
    if run:
        run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return history
