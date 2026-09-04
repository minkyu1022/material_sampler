"""End-to-end conditional Ising reproduction and artifact generation."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from torch import Tensor

from .ising import JANUSIsing, ghost_wolff_samples, heat_bath_prob, observables, train_step


@dataclass(frozen=True)
class IsingExperimentConfig:
    length: int = 16
    temperature_min: float = 1.5
    temperature_max: float = 3.2
    temperature_points: int = 11
    critical_temperature: float = 2.2692
    delta_mu_max: float = 0.4
    delta_mu_points: int = 11
    field_grid: tuple[float, ...] | None = None
    reveal_steps: int = 128
    rounds: int = 300
    batch_size: int = 512
    gradient_steps: int = 10
    learning_rate: float = 3e-3
    eval_samples: int = 512
    # SI: 3,000 total cluster steps with 600 discarded, hence 2,400 retained states.
    reference_samples: int = 2400
    reference_burn_in: int = 600
    reference_chains: int = 24
    reference_workers: int = 1
    width: int = 64
    depth: int = 4
    coexistence_fraction: float = 0.5
    narrow_condition_fraction: float = 0.0
    narrow_delta_mu_max: float = 0.04
    damping_eta: float = 0.0
    previous_model_interval: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.length,
            self.temperature_points,
            self.delta_mu_points,
            self.reveal_steps,
            self.rounds,
            self.batch_size,
            self.gradient_steps,
            self.eval_samples,
            self.reference_samples,
            self.reference_chains,
            self.reference_workers,
            self.width,
            self.depth,
            self.previous_model_interval,
        )
        if min(positive) < 1 or self.reference_burn_in < 0:
            raise ValueError("counts must be positive and burn-in non-negative")
        if not 0 < self.temperature_min <= self.temperature_max:
            raise ValueError("temperature range must be positive and ordered")
        if not self.temperature_min <= self.critical_temperature <= self.temperature_max:
            raise ValueError("critical_temperature must be inside the temperature range")
        if self.delta_mu_max < 0:
            raise ValueError("delta_mu_max must be non-negative")
        if not 0 <= self.coexistence_fraction <= 1:
            raise ValueError("coexistence_fraction must be between zero and one")
        if not 0 <= self.narrow_condition_fraction <= 1:
            raise ValueError("narrow_condition_fraction must be between zero and one")
        if not 0 < self.narrow_delta_mu_max <= self.delta_mu_max:
            raise ValueError("narrow_delta_mu_max must be inside the field window")
        if self.damping_eta < 0:
            raise ValueError("damping_eta must be non-negative")
        if self.reveal_steps > self.length * self.length:
            raise ValueError("reveal_steps cannot exceed the number of lattice sites")


@torch.no_grad()
def sample_in_blocks(
    model: JANUSIsing,
    batch_size: int,
    length: int,
    reveal_steps: int,
    temperature: float | Tensor,
    delta_mu: float | Tensor,
    *,
    device: torch.device,
    restore_zero_field_symmetry: bool = False,
) -> Tensor:
    """Reveal a random partition of all sites in the requested number of blocks."""
    sites = length * length
    state = torch.zeros((batch_size, length, length), device=device)
    reveal_at = torch.randint(reveal_steps, (batch_size, sites), device=device)
    flat = state.view(batch_size, sites)
    for step in range(reveal_steps):
        selected = reveal_at == step
        logits = model(state, step / reveal_steps, temperature, delta_mu).flatten(1)
        plus = torch.bernoulli(torch.sigmoid(logits))
        flat[selected] = plus[selected].mul(2).sub(1)

    if restore_zero_field_symmetry:
        # An acceptance-one global move restores the finite zero-field target's exact symmetry.
        field = torch.as_tensor(delta_mu, device=device)
        if field.ndim == 0:
            field = field.expand(batch_size)
        flip = field.eq(0) & torch.rand(batch_size, device=device).lt(0.5)
        state[flip] *= -1
    return state


def _condition_batch(
    config: IsingExperimentConfig, device: torch.device
) -> tuple[torch.Tensor, ...]:
    inverse_temperature = torch.empty(config.batch_size, device=device).uniform_(
        1.0 / config.temperature_max, 1.0 / config.temperature_min
    )
    temperature = inverse_temperature.reciprocal()
    delta_mu = torch.empty(config.batch_size, device=device).uniform_(
        -config.delta_mu_max, config.delta_mu_max
    )
    narrow = torch.rand(config.batch_size, device=device).lt(config.narrow_condition_fraction)
    delta_mu[narrow] = torch.empty(int(narrow.sum()), device=device).uniform_(
        -config.narrow_delta_mu_max, config.narrow_delta_mu_max
    )
    coexistence = torch.rand(config.batch_size, device=device).lt(config.coexistence_fraction)
    delta_mu[coexistence] = 0.0
    return temperature, delta_mu


def zero_field_errors(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Acceptance metrics for the zero-field phase-transition curve."""
    zero = [row for row in rows if row["delta_mu"] == 0]
    errors = [
        abs(row["janus"]["abs_magnetization"] - row["wolff"]["abs_magnetization"]) for row in zero
    ]
    critical = [
        error for row, error in zip(zero, errors, strict=True) if 2.0 <= row["temperature"] <= 2.6
    ]
    return {
        "zero_field_abs_magnetization_mae": float(np.mean(errors)),
        "critical_abs_magnetization_mae": float(np.mean(critical)) if critical else None,
        "zero_field_spin_symmetry_error": float(
            np.mean([abs(row["janus"]["up_fraction"] - 0.5) for row in zero])
        ),
    }


def _derived_seed(seed: int, temperature_index: int, field_index: int, stream: int) -> int:
    """Derive an order-independent uint32 seed for one grid point and sampler."""
    return int(
        np.random.SeedSequence([seed, temperature_index, field_index, stream]).generate_state(1)[0]
    )


def _wolff_reference(args: tuple[int, float, float, int, int, int, int]) -> np.ndarray:
    length, temperature, delta_mu, samples, burn_in, chains, seed = args
    return ghost_wolff_samples(
        length,
        temperature,
        delta_mu,
        num_samples=samples,
        burn_in=burn_in,
        chains=chains,
        seed=seed,
    )


def _chain_observables(reference: np.ndarray, delta_mu: float) -> tuple[list[dict[str, float]], dict[str, float | None]]:
    """Keep every chain mean and its across-chain standard error."""
    by_chain = reference if reference.ndim == 4 else reference[:, None]
    values = [observables(by_chain[:, chain], delta_mu=delta_mu) for chain in range(by_chain.shape[1])]
    standard_error = {
        key: (
            float(np.std([value[key] for value in values], ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else None
        )
        for key in values[0]
    }
    return values, standard_error


def evaluation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize grid agreement without discarding temperature or chain information."""
    observable_names = tuple(rows[0]["janus"])
    grid_errors = {
        name: {
            "mean_absolute_error": float(
                np.mean([abs(row["janus"][name] - row["wolff"][name]) for row in rows])
            ),
            "max_absolute_error": float(
                np.max([abs(row["janus"][name] - row["wolff"][name]) for row in rows])
            ),
        }
        for name in observable_names
    }
    zero_field_by_temperature = []
    for row in rows:
        if row["delta_mu"] != 0:
            continue
        zero_field_by_temperature.append(
            {
                "temperature": row["temperature"],
                "absolute_error": {
                    name: abs(row["janus"][name] - row["wolff"][name])
                    for name in observable_names
                },
                "wolff_chain_standard_error": row["wolff_chain_standard_error"],
            }
        )
    chain_uncertainty = {
        name: {
            "mean_standard_error": float(
                np.mean(
                    [
                        row["wolff_chain_standard_error"][name]
                        for row in rows
                        if row["wolff_chain_standard_error"][name] is not None
                    ]
                )
            ),
            "max_standard_error": float(
                np.max(
                    [
                        row["wolff_chain_standard_error"][name]
                        for row in rows
                        if row["wolff_chain_standard_error"][name] is not None
                    ]
                )
            ),
        }
        for name in observable_names
        if any(row["wolff_chain_standard_error"][name] is not None for row in rows)
    }
    return {
        "mean_up_error": grid_errors["up_fraction"]["mean_absolute_error"],
        "max_grid_up_fraction_error": grid_errors["up_fraction"]["max_absolute_error"],
        "grid_errors": grid_errors,
        "zero_field_by_temperature": zero_field_by_temperature,
        "wolff_chain_uncertainty": chain_uncertainty,
        **zero_field_errors(rows),
    }


def _temperature_grid(config: IsingExperimentConfig) -> np.ndarray:
    temperatures = np.linspace(
        config.temperature_min, config.temperature_max, config.temperature_points
    )
    temperatures[np.abs(temperatures - config.critical_temperature).argmin()] = (
        config.critical_temperature
    )
    return np.sort(temperatures)


def _field_grid(config: IsingExperimentConfig) -> np.ndarray:
    """Non-negative evaluation fields."""
    if config.field_grid is not None:
        fields = np.asarray(config.field_grid, dtype=float)
    elif config.delta_mu_points == 11 and config.delta_mu_max == 0.4:
        # Reconstructed from Fig. 2a's vector cell geometry; the SI omits the values.
        fields = np.asarray((0.0, 0.005, 0.01, 0.015, 0.02, 0.04, 0.06, 0.10, 0.15, 0.25, 0.40))
    else:
        fields = np.linspace(0.0, config.delta_mu_max, config.delta_mu_points)
    if len(fields) != config.delta_mu_points or fields[0] != 0 or fields[-1] != config.delta_mu_max:
        raise ValueError("field_grid must match delta_mu_points and span [0, delta_mu_max]")
    return fields


def _damped_train_step(
    model: JANUSIsing,
    previous_model: JANUSIsing,
    optimizer: torch.optim.Optimizer,
    terminals: Tensor,
    temperature: Tensor,
    delta_mu: Tensor,
    damping_eta: float,
) -> float:
    """Apply matching loss plus BMS-style fixed-point damping on the same masked sites."""
    optimizer.zero_grad(set_to_none=True)
    terminals = terminals.float()
    time = torch.rand(len(terminals), device=terminals.device)
    masked = torch.rand(terminals.shape, device=terminals.device) > time[:, None, None]
    state = terminals.masked_fill(masked, 0)
    logits = model(state, time, temperature, delta_mu)
    targets = heat_bath_prob(terminals, temperature, delta_mu)
    normalizer = masked.sum().clamp_min(1)
    matching = (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * masked
    ).sum() / normalizer
    with torch.no_grad():
        previous_logits = previous_model(state, time, temperature, delta_mu)
    damping = (F.mse_loss(logits, previous_logits, reduction="none") * masked).sum() / normalizer
    loss = matching + damping_eta * damping
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train_conditional(
    model: JANUSIsing,
    config: IsingExperimentConfig,
    output_dir: str | Path,
    *,
    resume: bool = False,
    run: Any = None,
) -> list[float]:
    """Train one amortized model over the full temperature/field rectangle."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    previous_model = deepcopy(model).eval()
    previous_model.requires_grad_(False)
    start_round, losses = 0, []
    if resume and checkpoint.exists():
        saved = torch.load(
            checkpoint, map_location=next(model.parameters()).device, weights_only=False
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        previous_model.load_state_dict(saved.get("previous_model", saved["model"]))
        start_round = saved["round"]
        losses = saved["losses"]

    device = next(model.parameters()).device
    for round_index in range(start_round, config.rounds):
        temperature, delta_mu = _condition_batch(config, device)
        terminals = sample_in_blocks(
            model,
            config.batch_size,
            config.length,
            config.reveal_steps,
            temperature,
            delta_mu,
            device=device,
        )
        for _ in range(config.gradient_steps):
            if config.damping_eta == 0:
                losses.append(train_step(model, optimizer, terminals, temperature, delta_mu))
            else:
                losses.append(
                    _damped_train_step(
                        model,
                        previous_model,
                        optimizer,
                        terminals,
                        temperature,
                        delta_mu,
                        config.damping_eta,
                    )
                )
        if (round_index + 1) % config.previous_model_interval == 0:
            previous_model.load_state_dict(model.state_dict())
        torch.save(
            {
                "model": model.state_dict(),
                "previous_model": previous_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "round": round_index + 1,
                "losses": losses,
                "config": asdict(config),
            },
            checkpoint,
        )
        if run is not None:
            run.log({"train/loss": losses[-1], "train/round": round_index + 1})
    return losses


def _mirror(row: dict[str, Any]) -> dict[str, Any]:
    mirrored = json.loads(json.dumps(row))
    mirrored["delta_mu"] = -row["delta_mu"]
    for method in ("janus", "wolff"):
        mirrored[method]["up_fraction"] = 1.0 - row[method]["up_fraction"]
        mirrored[method]["magnetization"] = -row[method]["magnetization"]
    for source, target in zip(
        row["wolff_chain_observables"], mirrored["wolff_chain_observables"], strict=True
    ):
        target["up_fraction"] = 1.0 - source["up_fraction"]
        target["magnetization"] = -source["magnetization"]
    return mirrored


@torch.no_grad()
def evaluate_grid(
    model: JANUSIsing, config: IsingExperimentConfig, *, run: Any = None
) -> list[dict[str, Any]]:
    """Evaluate the published 11x11 non-negative-field grid and exact spin symmetry."""
    temperatures = _temperature_grid(config)
    fields = _field_grid(config)
    rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    cuda_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    conditions = [
        (temperature_index, field_index, float(temperature), float(delta_mu))
        for temperature_index, temperature in enumerate(temperatures)
        for field_index, delta_mu in enumerate(fields)
    ]
    executor = (
        ProcessPoolExecutor(
            max_workers=config.reference_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        if config.reference_workers > 1
        else None
    )
    references = {}
    if executor is not None:
        references = {
            (temperature_index, field_index): executor.submit(
                _wolff_reference,
                (
                    config.length,
                    temperature,
                    delta_mu,
                    config.reference_samples,
                    config.reference_burn_in,
                    config.reference_chains,
                    _derived_seed(config.seed, temperature_index, field_index, 1),
                ),
            )
            for temperature_index, field_index, temperature, delta_mu in conditions
        }
    try:
        for temperature_index, field_index, temperature, delta_mu in conditions:
            janus_seed = _derived_seed(config.seed, temperature_index, field_index, 0)
            wolff_seed = _derived_seed(config.seed, temperature_index, field_index, 1)
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(janus_seed)
                generated = sample_in_blocks(
                    model,
                    config.eval_samples,
                    config.length,
                    config.reveal_steps,
                    float(temperature),
                    float(delta_mu),
                    device=device,
                    restore_zero_field_symmetry=delta_mu == 0,
                )
            reference = (
                references[(temperature_index, field_index)].result()
                if executor is not None
                else _wolff_reference(
                    (
                        config.length,
                        temperature,
                        delta_mu,
                        config.reference_samples,
                        config.reference_burn_in,
                        config.reference_chains,
                        wolff_seed,
                    )
                )
            )
            chain_values, chain_standard_error = _chain_observables(
                reference, float(delta_mu)
            )
            row = {
                "temperature": float(temperature),
                "delta_mu": float(delta_mu),
                "janus": observables(generated, delta_mu=float(delta_mu)),
                "wolff": observables(reference, delta_mu=float(delta_mu)),
                "wolff_chain_observables": chain_values,
                "wolff_chain_standard_error": chain_standard_error,
                "seeds": {"janus": janus_seed, "wolff": wolff_seed},
            }
            rows.append(row)
            if delta_mu > 0:
                rows.append(_mirror(row))
            if run is not None:
                run.log(
                    {
                        "eval/temperature": temperature,
                        "eval/delta_mu": delta_mu,
                        "eval/up_error": abs(
                            row["janus"]["up_fraction"] - row["wolff"]["up_fraction"]
                        ),
                    }
                )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    return rows


def save_plots(
    rows: list[dict[str, Any]], examples: dict[str, tuple[float, torch.Tensor]], output_dir: str | Path
) -> list[Path]:
    output = Path(output_dir)
    temperatures = sorted({row["temperature"] for row in rows})
    fields = sorted({row["delta_mu"] for row in rows})
    paths: list[Path] = []
    population_cmap = LinearSegmentedColormap.from_list(
        "janus_population", ((78 / 255, 133 / 255, 189 / 255), (1, 1, 1), (194 / 255, 71 / 255, 86 / 255))
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, method in zip(axes, ("janus", "wolff"), strict=True):
        values = np.array(
            [
                [
                    next(
                        row[method]["up_fraction"]
                        for row in rows
                        if row["temperature"] == temperature and row["delta_mu"] == field
                    )
                    for field in fields
                ]
                for temperature in temperatures
            ]
        )
        image = axis.pcolormesh(
            fields,
            temperatures,
            values,
            shading="nearest",
            cmap=population_cmap,
            vmin=0,
            vmax=1,
        )
        axis.set_xscale("symlog", linthresh=0.02, linscale=1.5)
        axis.set_xlim(fields[0], fields[-1])
        axis.set_ylim(temperatures[0], temperatures[-1])
        tick_fields = (fields[0], -0.02, 0.0, 0.02, fields[-1])
        axis.set_xticks(tick_fields, [f"{value:g}" for value in tick_fields])
        axis.set_yticks(np.linspace(temperatures[0], temperatures[-1], 4))
        axis.set(title=method.upper(), xlabel="Δμ", ylabel="T")
    fig.colorbar(image, ax=axes, label="spin-up fraction")
    paths.append(output / "population_map.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    zero = [row for row in rows if row["delta_mu"] == 0]
    for method in ("janus", "wolff"):
        axis.plot(
            [row["temperature"] for row in zero],
            [row[method]["abs_magnetization"] for row in zero],
            "o-",
            label=method.upper(),
        )
    axis.set(xlabel="T", ylabel="|m|", ylim=(0, 1))
    axis.legend()
    paths.append(output / "abs_magnetization.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    count = min(8, min(len(states) for _, states in examples.values()))
    fig, axes = plt.subplots(3, count, figsize=(2 * count, 6), squeeze=False, constrained_layout=True)
    for row_index, (label, (temperature, states)) in enumerate(examples.items()):
        for column, state in enumerate(states[:count].cpu()):
            axes[row_index, column].imshow(state, cmap="coolwarm", vmin=-1, vmax=1)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_title(f"{label}: T={temperature:g}", loc="left")
    paths.append(output / "config_examples.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)
    return paths


def _checkpoint_provenance(checkpoint: Path, config: IsingExperimentConfig) -> dict[str, Any]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_round": saved.get("round"),
        "training_config": saved.get("config"),
        "evaluation_config": asdict(config),
        "reference_protocol": {
            "sampler": "independent ghost-spin Wolff chains",
            "chains": config.reference_chains,
            "cluster_steps_per_chain_total": (
                config.reference_burn_in + config.reference_samples
            ),
            "cluster_steps_per_chain_burn_in": config.reference_burn_in,
            "retained_states_per_chain": config.reference_samples,
        },
        "seed_derivation": "numpy.random.SeedSequence([base_seed, temperature_index, field_index, stream])",
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
    }


@torch.no_grad()
def _configuration_examples(
    model: JANUSIsing, config: IsingExperimentConfig, device: torch.device
) -> dict[str, tuple[float, Tensor]]:
    conditions = {
        "below Tc": config.temperature_min,
        "near Tc": config.critical_temperature,
        "above Tc": config.temperature_max,
    }
    examples = {}
    cuda_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    for index, (label, temperature) in enumerate(conditions.items()):
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(_derived_seed(config.seed, index, 0, 2))
            examples[label] = (
                temperature,
                sample_in_blocks(
                    model,
                    8,
                    config.length,
                    config.reveal_steps,
                    temperature,
                    0.0,
                    device=device,
                    restore_zero_field_symmetry=True,
                ),
            )
    return examples


def _evaluate_and_save(
    model: JANUSIsing,
    config: IsingExperimentConfig,
    output: Path,
    checkpoint: Path,
    *,
    run: Any = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = evaluate_grid(model, config, run=run)
    plots = save_plots(rows, _configuration_examples(model, config, next(model.parameters()).device), output)
    result = {
        **(extra_metrics or {}),
        **evaluation_metrics(rows),
        "provenance": _checkpoint_provenance(checkpoint, config),
        "grid": rows,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if run is not None:
        run.log(
            {
                f"eval/{key}": value
                for key, value in result.items()
                if isinstance(value, (int, float))
            }
        )
        for path in plots:
            run.log({path.stem: __import__("wandb").Image(str(path))})
    return result


def evaluate_checkpoint(
    checkpoint: str | Path,
    config: IsingExperimentConfig,
    output_dir: str | Path,
    *,
    run: Any = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate an existing checkpoint without resuming or mutating training."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(checkpoint)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_eval_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = JANUSIsing(config.width, config.depth).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return _evaluate_and_save(model, config, output, checkpoint, run=run)


def run_experiment(
    config: IsingExperimentConfig,
    output_dir: str | Path,
    *,
    resume: bool = False,
    run: Any = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and save every reproducibility artifact."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    model = JANUSIsing(config.width, config.depth).to(device)
    losses = train_conditional(model, config, output, resume=resume, run=run)
    model.eval()
    return _evaluate_and_save(
        model,
        config,
        output,
        output / "checkpoint.pt",
        run=run,
        extra_metrics={"loss": losses[-1]},
    )
