"""Paper-faithful separate FCC/BCC JANUS training for fixed-composition Ni--Cr."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from ase.build import bulk
from torch import Tensor

from .alloy_model import AlloyPaiNN
from .cuni import KB_EV_K, VolumePrior
from .cuni_train import ReplayBuffer, _clip_field, _euler_maruyama
from .free_energy import cuni_prior_log_density, gaussian_path_log_ratio, path_weight_estimates
from .losses import CONT_LOSS_REGISTRY, DISC_LOSS_REGISTRY, get_loss
from .nicr import NICR_LATTICES
from .objective import bounded_score_target, interpolate, mask_terminal
from .samplers import fixed_composition_boundary_quota, sequential_random_order
from .torch_eam import TorchEAM

PROVENANCE = {
    "fixed_composition_boundary_quota": "AUTHOR_CONFIRMED",
    "steps_equals_n": "AUTHOR_CONFIRMED",
    "training_masking": "PAPER_CONFIRMED",
    "architecture_optimizer_replay_gscale": "CUNI_REPRODUCED_BASELINE",
    "phase_prior": "PROVISIONAL_RECONSTRUCTION",
}

DIAGNOSTIC_ROUNDS = frozenset((1, 12, 24, 48, 72, 96, 120))


@dataclass(frozen=True)
class NiCrTrainConfig:
    phase: str
    potential: Path
    prior: Path
    output: Path
    target_cutoff: float
    cutoff_convention: str = "provisional_abrupt_header"
    parent_config: Path = Path("configs/cuni/reproduced_best.json")
    initial_buffer: int = 5_000
    rounds: int = 120
    fresh_per_round: int = 500
    replay_size: int = 5_000
    updates_per_round: int = 500
    batch_size: int = 96
    rollout_batch: int = 12
    temperature_min: float = 600.0
    temperature_max: float = 1_200.0
    temperature_values: tuple[float, ...] = ()
    composition_rungs: tuple[int, ...] = ()
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-5
    warmup_updates: int = 5_000
    optimizer: str = "adamw"
    weight_decay: float = 1e-3
    continuous_loss: str = "tsm"
    discrete_loss: str = "sce"
    continuous_weight_u: float = 1.0
    continuous_weight_v: float = 1.0
    discrete_weight: float = 2.0
    features: int = 64
    layers: int = 4
    radial_basis: int = 16
    sigma_u_ref: float = 0.01004514620163701
    sigma_u_exponent: float = 0.5
    sigma_v_scale: float = 1.0
    diffusion_u: float = 0.02
    diffusion_v: float = 0.02
    diffusion_temperature_ref: float = 750.0
    diffusion_scale: float = 1.0
    diffusion_scale_u: float | None = None
    diffusion_scale_v: float | None = None
    target_score_u_clip: float | None = 100.0
    target_score_v_clip: float | None = 1_000.0
    rollout_velocity_clip: float | None = 0.1
    rollout_score_clip: float | None = 1_000.0
    gradient_clip_norm: float = 100.0
    seed: int = 2026
    bf16: bool = False
    resume: bool = True
    wandb_project: str | None = "janus-reproduce"
    checkpoint_version: int = 1

    @property
    def spec(self):
        return NICR_LATTICES[self.phase]

    @property
    def steps(self) -> int:
        return self.spec.n_atoms

    @classmethod
    def from_json(cls, path: Path) -> NiCrTrainConfig:
        payload = json.loads(path.read_text())
        parent_path = Path(payload["parent"])
        parent = json.loads(parent_path.read_text())["config"]
        inherited = {
            "initial_buffer": parent["initial_buffer"],
            "rounds": parent["rounds"],
            "replay_size": parent["replay_size"],
            "updates_per_round": parent["updates_per_round"],
            "batch_size": parent["global_batch"],
            "rollout_batch": parent["rollout_batch"],
            "temperature_min": parent["temperature_min"],
            "temperature_max": parent["temperature_max"],
            "learning_rate": parent["learning_rate"],
            "minimum_learning_rate": parent["minimum_learning_rate"],
            "warmup_updates": parent["warmup_updates"],
            "optimizer": parent["optimizer"],
            "weight_decay": parent["weight_decay"],
            "discrete_weight": parent["discrete_weight"],
            "features": parent["features"],
            "layers": parent["layers"],
            "radial_basis": parent["radial_basis"],
            "sigma_u_ref": parent["sigma_u_ref"],
            "sigma_u_exponent": parent["sigma_u_exponent"],
            "sigma_v_scale": parent["sigma_v_scale"],
            "diffusion_u": parent["diffusion_u"],
            "diffusion_v": parent["diffusion_v"],
            "diffusion_temperature_ref": parent["diffusion_temperature_ref"],
            "target_score_u_clip": parent["target_score_u_clip"],
            "target_score_v_clip": parent["target_score_v_clip"],
            "rollout_velocity_clip": parent["rollout_velocity_clip"],
            "rollout_score_clip": parent["rollout_score_clip"],
            "gradient_clip_norm": parent["gradient_clip_norm"],
            "bf16": parent["bf16"],
        }
        inherited.update(payload["overrides"])
        for key in ("potential", "prior", "output", "parent_config"):
            if key in inherited:
                inherited[key] = Path(inherited[key])
        prior_values = json.loads(inherited["prior"].read_text())
        overrides = payload["overrides"]
        if "sigma_u_ref" not in overrides and "sigma_u_ref" in prior_values:
            inherited["sigma_u_ref"] = prior_values["sigma_u_ref"]
        if "sigma_u_exponent" not in overrides and "sigma_u_exponent" in prior_values:
            inherited["sigma_u_exponent"] = prior_values["sigma_u_exponent"]
        inherited["parent_config"] = parent_path
        for key in ("temperature_values", "composition_rungs"):
            if key in inherited:
                inherited[key] = tuple(inherited[key])
        return cls(**inherited)


def _reference(config: NiCrTrainConfig, device: torch.device) -> Tensor:
    lattice_constant = 3.5 if config.phase == "fcc" else 2.8
    atoms = bulk("Ni", config.phase, a=lattice_constant, cubic=True).repeat(
        (config.spec.repeats,) * 3
    )
    return torch.tensor(atoms.get_scaled_positions(wrap=False), dtype=torch.float32, device=device)


def _prior(config: NiCrTrainConfig) -> VolumePrior:
    values = json.loads(config.prior.read_text())
    return VolumePrior(*(values[key] for key in VolumePrior.__dataclass_fields__))


def _conditions(
    config: NiCrTrainConfig,
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if config.temperature_values:
        values = torch.tensor(config.temperature_values, device=device)
        temperature = values[
            torch.randint(len(values), (count,), device=device, generator=generator)
        ]
    else:
        inverse = torch.empty(count, device=device).uniform_(
            1 / config.temperature_max, 1 / config.temperature_min, generator=generator
        )
        temperature = inverse.reciprocal()
    if config.composition_rungs:
        values = torch.tensor(config.composition_rungs, device=device)
        target_cr = values[torch.randint(len(values), (count,), device=device, generator=generator)]
    else:
        target_cr = torch.randint(
            config.spec.n_atoms + 1, (count,), device=device, generator=generator
        )
    return temperature.float(), target_cr.long()


def _prior_values(
    config: NiCrTrainConfig,
    prior: VolumePrior,
    temperature: Tensor,
    target_cr: Tensor,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    fraction = target_cr / config.spec.n_atoms
    sigma_u = config.sigma_u_ref * (temperature / prior.temperature_ref).pow(
        config.sigma_u_exponent
    )
    displacement = sigma_u[:, None, None] * torch.randn(
        len(temperature),
        config.spec.n_atoms,
        3,
        device=temperature.device,
        generator=generator,
    )
    displacement -= displacement.mean(1, keepdim=True)
    atomic_volume = (
        (1 - fraction) * prior.v_ni
        + fraction * prior.v_cu
        + prior.omega * fraction * (1 - fraction)
    ) * (1 + prior.alpha * (temperature - prior.temperature_ref))
    mean_v = (config.spec.n_atoms * atomic_volume).log()
    log_volume = mean_v + config.sigma_v_scale * prior.sigma_log_volume * torch.randn(
        mean_v.shape, device=mean_v.device, generator=generator
    )
    return displacement, log_volume, sigma_u, mean_v


def _autocast(config: NiCrTrainConfig, device: torch.device):
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=config.bf16 and device.type == "cuda"
    )


@torch.no_grad()
def rollout_nicr(
    model: AlloyPaiNN,
    oracle: TorchEAM,
    config: NiCrTrainConfig,
    prior: VolumePrior,
    reference: Tensor,
    temperature: Tensor,
    target_cr: Tensor,
    *,
    generator: torch.Generator,
    path_weights: bool,
    path_diagnostics: bool = False,
) -> dict[str, Tensor]:
    device = reference.device
    batch = len(temperature)
    displacement, log_volume, sigma_u, mean_v = _prior_values(
        config, prior, temperature, target_cr, generator
    )
    initial_u, initial_v = displacement.clone(), log_volume.clone()
    species = torch.full((batch, config.spec.n_atoms), 2, dtype=torch.long, device=device)
    order = sequential_random_order(batch, config.spec.n_atoms, device, generator=generator)
    log_q = torch.zeros(batch, dtype=torch.float64, device=device)
    log_path_u = torch.zeros_like(log_q)
    log_path_v = torch.zeros_like(log_q)
    path_u_steps: list[Tensor] = []
    path_v_steps: list[Tensor] = []
    forced_count = torch.zeros(batch, dtype=torch.long, device=device)
    forced_to_cr_count = torch.zeros_like(forced_count)
    first_forced = torch.full_like(forced_count, config.spec.n_atoms)
    for step in range(config.steps):
        t0, t1 = step / config.steps, (step + 1) / config.steps
        with _autocast(config, device):
            output = model(
                species,
                displacement,
                log_volume,
                reference,
                t0,
                temperature,
                target_cr / config.spec.n_atoms,
            )
        if not all(torch.isfinite(value).all() for value in output):
            raise FloatingPointError(f"non-finite forward output at step {step}")
        b_u = _clip_field(output.b_u.float(), config.rollout_velocity_clip, vector=True)
        s_u = _clip_field(output.s_u.float(), config.rollout_score_clip, vector=True)
        b_v = _clip_field(output.b_v.float(), config.rollout_velocity_clip, vector=False)
        s_v = _clip_field(output.s_v.float(), config.rollout_score_clip, vector=False)
        dt = 1 / config.steps
        temperature_scale = (temperature / config.diffusion_temperature_ref).sqrt()
        scale_u = (
            config.diffusion_scale if config.diffusion_scale_u is None else config.diffusion_scale_u
        )
        scale_v = (
            config.diffusion_scale if config.diffusion_scale_v is None else config.diffusion_scale_v
        )
        g_u = config.diffusion_u * scale_u * temperature_scale
        g_v = config.diffusion_v * scale_v * temperature_scale
        previous_u, previous_v = displacement, log_volume
        displacement = _euler_maruyama(
            displacement,
            b_u,
            s_u,
            g_u,
            dt,
            torch.randn(displacement.shape, device=device, generator=generator),
        )
        displacement -= displacement.mean(1, keepdim=True)
        log_volume = _euler_maruyama(
            log_volume,
            b_v,
            s_v,
            g_v,
            dt,
            torch.randn(log_volume.shape, device=device, generator=generator),
        )
        reveal = fixed_composition_boundary_quota(
            species,
            output.species_logits.float(),
            target_cr,
            order[:, step],
            generator=generator,
        )
        species = reveal.species
        log_q += reveal.log_probability
        newly_forced = reveal.forced & first_forced.eq(config.spec.n_atoms)
        first_forced = torch.where(newly_forced, step, first_forced)
        forced_count += reveal.forced.long()
        forced_to_cr_count += reveal.forced_to_cr.long()
        if path_weights:
            with _autocast(config, device):
                backward = model(
                    species,
                    displacement,
                    log_volume,
                    reference,
                    t1,
                    temperature,
                    target_cr / config.spec.n_atoms,
                )
            if not all(torch.isfinite(value).all() for value in backward):
                raise FloatingPointError(f"non-finite backward output at step {step}")
            backward_b_u = _clip_field(
                backward.b_u.float(), config.rollout_velocity_clip, vector=True
            )
            backward_s_u = _clip_field(backward.s_u.float(), config.rollout_score_clip, vector=True)
            backward_b_v = _clip_field(
                backward.b_v.float(), config.rollout_velocity_clip, vector=False
            )
            backward_s_v = _clip_field(
                backward.s_v.float(), config.rollout_score_clip, vector=False
            )
            delta_path_u = gaussian_path_log_ratio(
                previous_u,
                displacement,
                b_u - b_u.mean(1, keepdim=True),
                s_u - s_u.mean(1, keepdim=True),
                backward_b_u - backward_b_u.mean(1, keepdim=True),
                backward_s_u - backward_s_u.mean(1, keepdim=True),
                g_u.square(),
                dt,
            )
            delta_path_v = gaussian_path_log_ratio(
                previous_v,
                log_volume,
                b_v,
                s_v,
                backward_b_v,
                backward_s_v,
                g_v.square(),
                dt,
            )
            log_path_u += delta_path_u
            log_path_v += delta_path_v
            if path_diagnostics:
                path_u_steps.append(delta_path_u)
                path_v_steps.append(delta_path_v)
    if not species.eq(1).sum(1).eq(target_cr).all():
        raise AssertionError("boundary quota rollout violated terminal composition")
    result = {
        "species": species,
        "displacement": displacement,
        "log_volume": log_volume,
        "temperature": temperature,
        "target_cr": target_cr,
        "log_q_discrete": log_q,
        "forced_count": forced_count,
        "forced_to_cr_count": forced_to_cr_count,
        "first_forced_step": first_forced,
    }
    if path_weights:
        energy = oracle(species, reference[None] + displacement, log_volume).double()
        log_prior = cuni_prior_log_density(
            initial_u,
            initial_v,
            sigma_u,
            mean_v,
            config.sigma_v_scale * prior.sigma_log_volume,
        )
        beta = 1 / (KB_EV_K * temperature.double())
        log_target = -beta * energy + config.spec.n_atoms * log_volume.double()
        log_weight = log_target - log_prior - log_q + log_path_u + log_path_v
        log_xi, normalized_weight, ess = path_weight_estimates(log_weight)
        result |= {
            "energy": energy,
            "log_prior": log_prior,
            "log_target": log_target,
            "log_continuous_u": log_path_u,
            "log_continuous_v": log_path_v,
            "log_weight": log_weight,
            "log_xi": log_xi,
            "normalized_weight": normalized_weight,
            "ess": ess,
        }
        if path_diagnostics:
            result["log_continuous_u_steps"] = torch.stack(path_u_steps, 1)
            result["log_continuous_v_steps"] = torch.stack(path_v_steps, 1)
    return result


def _label(
    oracle: TorchEAM,
    states: dict[str, Tensor],
    config: NiCrTrainConfig,
    reference: Tensor,
) -> dict[str, Tensor]:
    output = {key: [] for key in ("score_u", "score_v", "heat_bath", "energy")}
    for index in range(len(states["species"])):
        species = states["species"][index : index + 1]
        displacement = states["displacement"][index : index + 1].double()
        log_volume = states["log_volume"][index : index + 1].double()
        labels = oracle.labels(species, reference.double()[None] + displacement, log_volume)
        with torch.no_grad():
            site_energy = oracle.all_site_energies(
                species, reference.double()[None] + displacement, log_volume
            )
            beta = 1 / (KB_EV_K * states["temperature"][index].double())
            length = log_volume.exp().pow(1 / 3)
            score_u = beta * labels.forces * length[:, None, None]
            score_u -= score_u.mean(1, keepdim=True)
            output["score_u"].append(score_u.float())
            output["score_v"].append(
                (config.spec.n_atoms - beta * labels.log_volume_derivative).float()
            )
            output["heat_bath"].append((-beta * site_energy).softmax(-1).float())
            output["energy"].append(labels.energy.detach().float())
    return states | {key: torch.cat(value) for key, value in output.items()}


def _train_updates(
    model: AlloyPaiNN,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: NiCrTrainConfig,
    prior: VolumePrior,
    reference: Tensor,
    device: torch.device,
    generator: torch.Generator,
    completed_updates: int,
) -> tuple[float, dict[str, float], float]:
    continuous_loss = get_loss(CONT_LOSS_REGISTRY, config.continuous_loss)
    discrete_loss = get_loss(DISC_LOSS_REGISTRY, config.discrete_loss)
    sums = {key: 0.0 for key in ("u", "v", "discrete")}
    grad_sum = 0.0
    total_updates = config.rounds * config.updates_per_round
    for update in range(config.updates_per_round):
        step = completed_updates + update
        if step < config.warmup_updates:
            lr = config.learning_rate * (step + 1) / max(config.warmup_updates, 1)
        else:
            progress = (step - config.warmup_updates) / max(
                total_updates - config.warmup_updates - 1, 1
            )
            lr = config.minimum_learning_rate + 0.5 * (
                config.learning_rate - config.minimum_learning_rate
            ) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = replay.sample(config.batch_size, device)
        x0_u, x0_v, sigma_u, mean_v = _prior_values(
            config, prior, batch["temperature"], batch["target_cr"], generator
        )
        t = torch.rand(config.batch_size, device=device, generator=generator)
        u, v = (
            interpolate(x0_u, batch["displacement"], t),
            interpolate(x0_v, batch["log_volume"], t),
        )
        species, masked = mask_terminal(batch["species"], t, 2)
        target_s_u = bounded_score_target(
            x0_u,
            t,
            _clip_field(batch["score_u"], config.target_score_u_clip, vector=True),
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
        optimizer.zero_grad(set_to_none=True)
        output = model(
            species,
            u,
            v,
            reference,
            t,
            batch["temperature"],
            batch["target_cr"] / config.spec.n_atoms,
        )
        components = {
            "u": config.continuous_weight_u
            * continuous_loss(
                output.b_u,
                batch["displacement"] - x0_u,
                output.s_u,
                target_s_u,
            ),
            "v": config.continuous_weight_v
            * continuous_loss(
                output.b_v,
                batch["log_volume"] - x0_v,
                output.s_v,
                target_s_v,
            ),
            "discrete": config.discrete_weight
            * discrete_loss(output.species_logits, batch["heat_bath"], masked),
        }
        loss = sum(components.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Ni-Cr training loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        for key, value in components.items():
            sums[key] += float(value.detach())
        grad_sum += float(grad)
    return (
        sum(sums.values()) / config.updates_per_round,
        {key: value / config.updates_per_round for key, value in sums.items()},
        grad_sum / config.updates_per_round,
    )


def resolved_config(config: NiCrTrainConfig) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _run_provenance(config: NiCrTrainConfig) -> dict:
    potential_hash = hashlib.sha256(config.potential.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"), text=True, stderr=subprocess.DEVNULL
            )
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "UNAVAILABLE_NOT_A_GIT_REPOSITORY", None
    return {
        "resolved_config": resolved_config(config),
        "steps": config.steps,
        "n_atoms": config.spec.n_atoms,
        "graph_cutoff": config.spec.graph_cutoff,
        "target_cutoff": config.target_cutoff,
        "cutoff_convention": config.cutoff_convention,
        "potential_sha256": potential_hash,
        "git_commit": commit,
        "dirty_tree": dirty,
        "provenance_labels": PROVENANCE,
        "host": platform.node(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "pid": os.getpid(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def effective_hamiltonian_manifest(config: NiCrTrainConfig) -> dict:
    if config.target_cutoff != config.spec.graph_cutoff:
        raise ValueError("target and graph cutoffs must match for the candidate Hamiltonian")
    if config.cutoff_convention != "provisional_abrupt_header":
        raise ValueError("unsupported candidate cutoff convention")
    potential_hash = hashlib.sha256(config.potential.read_bytes()).hexdigest()
    expected = {
        "potential_sha256": potential_hash,
        "target_cutoff": config.target_cutoff,
        "cutoff_convention": config.cutoff_convention,
    }
    prior_values = json.loads(config.prior.read_text())
    if any(prior_values.get(key) != value for key, value in expected.items()):
        raise ValueError(f"prior Hamiltonian mismatch: expected {expected}")
    return {
        **expected,
        "graph_cutoff": config.spec.graph_cutoff,
        "volume_prior_hamiltonian": expected,
        "displacement_prior_hamiltonian": expected,
        "replay_hamiltonian": expected,
        "path_weight_hamiltonian": expected,
    }


def train_nicr(config: NiCrTrainConfig) -> list[dict]:
    if config.phase not in NICR_LATTICES:
        raise ValueError("phase must be fcc or bcc")
    effective_hamiltonian = effective_hamiltonian_manifest(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    config.output.mkdir(parents=True, exist_ok=True)
    provenance = _run_provenance(config)
    provenance["effective_hamiltonian"] = effective_hamiltonian
    (config.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    model = AlloyPaiNN(
        features=config.features,
        layers=config.layers,
        radial_basis=config.radial_basis,
        cutoff=config.spec.graph_cutoff,
        temperature_reference=config.diffusion_temperature_ref,
        temperature_min=config.temperature_min,
        temperature_max=config.temperature_max,
        condition_intercept=0.0,
        condition_slope=0.0,
        condition_scale=1.0,
    ).to(device)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(config.optimizer)
    if optimizer_type is None:
        raise ValueError("optimizer must be adam or adamw")
    optimizer = optimizer_type(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    oracle = TorchEAM(
        config.potential,
        species_indices=(0, 2),
        cutoff=config.target_cutoff,
    ).to(device)
    prior = _prior(config)
    reference = _reference(config, device)
    replay = ReplayBuffer()
    history = []
    checkpoint_path = config.output / "checkpoint.pt"
    start_round = 0
    if config.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["config"] != provenance["resolved_config"]:
            raise ValueError("resume config differs from saved resolved config")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        replay = ReplayBuffer(saved["replay"])
        generator.set_state(saved["generator_state"].cpu())
        history, start_round = saved["history"], saved["round"]

    def generate(count: int, *, weights: bool):
        chunks = []
        for start in range(0, count, config.rollout_batch):
            size = min(config.rollout_batch, count - start)
            temperature, target_cr = _conditions(config, size, device, generator)
            chunks.append(
                rollout_nicr(
                    model,
                    oracle,
                    config,
                    prior,
                    reference,
                    temperature,
                    target_cr,
                    generator=generator,
                    path_weights=weights,
                )
            )
        return {
            key: torch.cat(
                [chunk[key].reshape(1) if chunk[key].ndim == 0 else chunk[key] for chunk in chunks]
            )
            for key in chunks[0]
            if key not in {"log_xi", "normalized_weight", "ess"}
        }

    replay_fields = (
        "species",
        "displacement",
        "log_volume",
        "temperature",
        "target_cr",
        "score_u",
        "score_v",
        "heat_bath",
        "energy",
    )
    if not len(replay):
        replay.add(
            {
                key: value
                for key, value in _label(
                    oracle, generate(config.initial_buffer, weights=False), config, reference
                ).items()
                if key in replay_fields
            },
            config.replay_size,
        )
    for outer in range(start_round, config.rounds):
        started = time.perf_counter()
        generated = generate(config.fresh_per_round, weights=False)
        rollout_seconds = time.perf_counter() - started
        started = time.perf_counter()
        labeled = _label(oracle, generated, config, reference)
        label_seconds = time.perf_counter() - started
        replay.add({key: labeled[key] for key in replay_fields}, config.replay_size)
        loss, components, grad = _train_updates(
            model,
            optimizer,
            replay,
            config,
            prior,
            reference,
            device,
            generator,
            outer * config.updates_per_round,
        )
        # Importance weights normalize only within a fixed thermodynamic condition.
        # Mixing different (T, n_Cr) partition functions makes ESS meaningless.
        eval_count = min(64, config.rollout_batch * 2)
        eval_temperature = torch.full(
            (eval_count,),
            (config.temperature_min + config.temperature_max) / 2,
            device=device,
        )
        eval_target = torch.full(
            (eval_count,), config.spec.n_atoms // 2, dtype=torch.long, device=device
        )
        evaluation = rollout_nicr(
            model,
            oracle,
            config,
            prior,
            reference,
            eval_temperature,
            eval_target,
            generator=generator,
            path_weights=True,
        )
        first = evaluation["first_forced_step"]
        metrics = {
            "round": outer + 1,
            "loss": loss,
            "loss_components": components,
            "gradient_norm": grad,
            "replay_size": len(replay),
            "mean_energy_per_atom": float(labeled["energy"].mean() / config.spec.n_atoms),
            "mean_rms_u": float(labeled["displacement"].square().sum(-1).mean(-1).sqrt().mean()),
            "mean_volume_per_atom": float(labeled["log_volume"].exp().mean() / config.spec.n_atoms),
            "exact_composition": bool(
                generated["species"].eq(1).sum(1).eq(generated["target_cr"]).all()
            ),
            "forced_fraction": float(evaluation["forced_count"].float().mean() / config.steps),
            "forced_final_10pct_fraction": float(
                first.ge(math.floor(0.9 * config.steps)).float().mean()
            ),
            "first_forced_step_mean": float(first.float().mean()),
            "forced_to_cr_fraction": float(
                evaluation["forced_to_cr_count"].sum()
                / evaluation["forced_count"].sum().clamp_min(1)
            ),
            "ess": float(evaluation["ess"]),
            "std_log_weight": float(evaluation["log_weight"].std(unbiased=False)),
            "eval_temperature": float(eval_temperature[0]),
            "eval_target_cr": int(eval_target[0]),
            "finite_weights": bool(torch.isfinite(evaluation["log_weight"]).all()),
            "rollout_seconds": rollout_seconds,
            "label_seconds": label_seconds,
        }
        history.append(metrics)
        checkpoint_tmp = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "replay": replay.data,
                "generator_state": generator.get_state().cpu(),
                "history": history,
                "round": outer + 1,
                "config": provenance["resolved_config"],
                "provenance": provenance,
            },
            checkpoint_tmp,
        )
        checkpoint_tmp.replace(checkpoint_path)
        if outer + 1 in DIAGNOSTIC_ROUNDS:
            diagnostic_dir = config.output / "diagnostic_checkpoints"
            diagnostic_dir.mkdir(exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "round": outer + 1,
                    "config": provenance["resolved_config"],
                    "provenance": provenance,
                },
                diagnostic_dir / f"round_{outer + 1:03d}.pt",
            )
        with (config.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
    return history
