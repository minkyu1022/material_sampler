"""Batched, resumable Cu--Ni semi-grand NPT reference Monte Carlo."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .cuni import KB_EV_K, build_cuni_fcc, delta_mu_108, temperatures_108


@dataclass(frozen=True)
class ReferenceState:
    index: int
    temperature: float
    delta_mu: float
    walker: int  # 0: all Ni; 1: all Cu


@dataclass(frozen=True)
class ReferenceConfig:
    total_sweeps: int = 40_000
    burn_in: int = 2_000
    config_thin: int = 100
    species_moves: int = 6
    displacement_step: float = 0.01 / math.sqrt(108)
    log_volume_step: float = 0.03 / math.sqrt(108)
    adapt_interval: int = 50
    target_acceptance: float = 0.3
    pressure: float = 0.0  # eV / Angstrom^3; the published run uses P=0.
    checkpoint_interval: int = 500
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.total_sweeps < 1 or not 0 <= self.burn_in < self.total_sweeps:
            raise ValueError("require 0 <= burn_in < total_sweeps")
        if self.config_thin < 1 or self.species_moves < 0 or self.adapt_interval < 1:
            raise ValueError("thin/adapt_interval must be positive and species_moves non-negative")
        if self.displacement_step < 0 or self.log_volume_step < 0:
            raise ValueError("proposal widths must be non-negative")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")


def reference_states_108() -> list[ReferenceState]:
    """Return the exact published 15 x 33 x 2 N=108 state/walker grid."""
    return [
        ReferenceState(index, float(temperature), float(delta_mu), walker)
        for index, (temperature, delta_mu, walker) in enumerate(
            (t, mu, w) for t in temperatures_108() for mu in delta_mu_108() for w in range(2)
        )
    ]


def output_path(output: Path, state: ReferenceState) -> Path:
    return output / (
        f"N108_T{state.temperature:07.2f}_mu{state.delta_mu:07.4f}_walker{state.walker}.npz"
    )


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _accept(log_ratio: Tensor, generator: torch.Generator) -> Tensor:
    uniform = torch.rand(log_ratio.shape, device=log_ratio.device, generator=generator)
    return (log_ratio >= 0) | (uniform.log() < log_ratio)


def volume_log_acceptance(
    beta: Tensor,
    delta_energy: Tensor,
    delta_volume: Tensor,
    delta_log_volume: Tensor,
    *,
    pressure: float,
    n_atoms: int,
) -> Tensor:
    """Semi-grand NPT log-volume Metropolis ratio, including the V**N Jacobian."""
    return -beta * (delta_energy + pressure * delta_volume) + n_atoms * delta_log_volume


def species_log_acceptance(
    beta: Tensor, delta_energy: Tensor, delta_mu: Tensor, delta_n_cu: Tensor
) -> Tensor:
    """Semi-grand single-site flip Metropolis ratio for delta_mu=mu_Cu-mu_Ni."""
    return beta * (delta_mu * delta_n_cu - delta_energy)


def _batch_signature(states: list[ReferenceState], config: ReferenceConfig) -> str:
    source = json.dumps(
        {"states": [asdict(state) for state in states], "config": asdict(config)}, sort_keys=True
    )
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _initial_state(
    states: list[ReferenceState], device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    atoms = build_cuni_fcc(108)
    fractional = (
        torch.as_tensor(atoms.get_scaled_positions(wrap=False), dtype=dtype, device=device)
        .expand(len(states), -1, -1)
        .clone()
    )
    log_volume = torch.full(
        (len(states),), math.log(atoms.get_volume()), dtype=dtype, device=device
    )
    species = torch.stack(
        [torch.full((108,), 1 - state.walker, dtype=torch.long, device=device) for state in states]
    )
    return species, fractional, log_volume


def run_reference_batch(
    oracle: nn.Module,
    states: list[ReferenceState],
    output: str | Path,
    *,
    config: ReferenceConfig | None = None,
    checkpoint: str | Path | None = None,
) -> list[Path]:
    """Run independent chains together and write one analysis-ready NPZ per chain.

    Scalar traces contain every sweep, including burn-in. ``production_start`` in
    metadata marks the raw 38,000-sweep production slice used for IAT/thinning.
    """
    if not states:
        return []
    config = config or ReferenceConfig()
    output = Path(output)
    parameter = next(oracle.parameters(), None)
    buffer = next(oracle.buffers(), None)
    anchor = parameter if parameter is not None else buffer
    if anchor is None:
        raise ValueError("oracle must have a parameter or buffer defining device/dtype")
    device, dtype = anchor.device, anchor.dtype
    if not dtype.is_floating_point:
        raise ValueError("oracle must use a floating-point dtype")

    signature = _batch_signature(states, config)
    checkpoint = Path(checkpoint) if checkpoint else output / f"checkpoint_{signature}.pt"
    temperatures = torch.tensor([state.temperature for state in states], dtype=dtype, device=device)
    delta_mu = torch.tensor([state.delta_mu for state in states], dtype=dtype, device=device)
    beta = 1 / (KB_EV_K * temperatures)
    temperature_scale = (temperatures / temperatures_108().max()).sqrt()
    generator = torch.Generator(device=device)

    batch, atoms = len(states), 108
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError(f"checkpoint does not match requested batch: {checkpoint}")
        sweep_start = int(saved["sweep"])
        species = saved["species"].to(device)
        fractional = saved["fractional"].to(device=device, dtype=dtype)
        log_volume = saved["log_volume"].to(device=device, dtype=dtype)
        energy = saved["energy"].to(device=device, dtype=dtype)
        displacement_step = saved["displacement_step"].to(device=device, dtype=dtype)
        volume_step = saved["volume_step"].to(device=device, dtype=dtype)
        traces = saved["traces"]
        accepted = saved["accepted"].to(device)
        attempted = saved["attempted"].to(device)
        window_accepted = saved["window_accepted"].to(device)
        window_attempted = saved["window_attempted"].to(device)
        config_sweeps = list(saved["config_sweeps"])
        config_fractional = list(saved["config_fractional"])
        config_species = list(saved["config_species"])
        generator.set_state(saved["rng_state"])
    else:
        sweep_start = 0
        species, fractional, log_volume = _initial_state(states, device, dtype)
        with torch.inference_mode():
            energy = oracle(species, fractional, log_volume)
        displacement_step = config.displacement_step * temperature_scale
        volume_step = config.log_volume_step * temperature_scale
        traces = {
            "energy_eV": torch.empty((config.total_sweeps, batch), dtype=torch.float64),
            "n_cu": torch.empty((config.total_sweeps, batch), dtype=torch.int16),
            "log_volume": torch.empty((config.total_sweeps, batch), dtype=torch.float64),
        }
        accepted = torch.zeros((3, batch), dtype=torch.long, device=device)
        attempted = torch.zeros_like(accepted)
        window_accepted = torch.zeros((2, batch), dtype=torch.long, device=device)
        window_attempted = torch.zeros_like(window_accepted)
        config_sweeps: list[int] = []
        config_fractional: list[Tensor] = []
        config_species: list[Tensor] = []
        generator.manual_seed(config.seed + states[0].index)

    def record(move: int, accept: Tensor, production: bool) -> None:
        if production:
            accepted[move] += accept
            attempted[move] += 1
        elif move < 2:
            window_accepted[move] += accept
            window_attempted[move] += 1

    def save_checkpoint(next_sweep: int) -> None:
        _atomic_torch_save(
            {
                "signature": signature,
                "sweep": next_sweep,
                "species": species.cpu(),
                "fractional": fractional.cpu(),
                "log_volume": log_volume.cpu(),
                "energy": energy.cpu(),
                "displacement_step": displacement_step.cpu(),
                "volume_step": volume_step.cpu(),
                "traces": traces,
                "accepted": accepted.cpu(),
                "attempted": attempted.cpu(),
                "window_accepted": window_accepted.cpu(),
                "window_attempted": window_attempted.cpu(),
                "config_sweeps": config_sweeps,
                "config_fractional": config_fractional,
                "config_species": config_species,
                "rng_state": generator.get_state(),
            },
            checkpoint,
        )

    with torch.inference_mode():
        for sweep in range(sweep_start, config.total_sweeps):
            production = sweep >= config.burn_in

            trial_fractional = (
                fractional
                + torch.randn(fractional.shape, dtype=dtype, device=device, generator=generator)
                * displacement_step[:, None, None]
            ).remainder(1)
            trial_energy = oracle(species, trial_fractional, log_volume)
            move_accept = _accept(-beta * (trial_energy - energy), generator)
            fractional = torch.where(move_accept[:, None, None], trial_fractional, fractional)
            energy = torch.where(move_accept, trial_energy, energy)
            record(0, move_accept, production)

            delta_log_volume = (
                torch.randn((batch,), dtype=dtype, device=device, generator=generator) * volume_step
            )
            trial_log_volume = log_volume + delta_log_volume
            trial_energy = oracle(species, fractional, trial_log_volume)
            delta_volume = trial_log_volume.exp() - log_volume.exp()
            log_ratio = volume_log_acceptance(
                beta,
                trial_energy - energy,
                delta_volume,
                delta_log_volume,
                pressure=config.pressure,
                n_atoms=atoms,
            )
            move_accept = _accept(log_ratio, generator)
            log_volume = torch.where(move_accept, trial_log_volume, log_volume)
            energy = torch.where(move_accept, trial_energy, energy)
            record(1, move_accept, production)

            for _ in range(config.species_moves):
                site = torch.randint(atoms, (batch,), device=device, generator=generator)
                row = torch.arange(batch, device=device)
                trial_species = species.clone()
                old_species = species[row, site]
                trial_species[row, site] = 1 - old_species
                trial_energy = oracle(trial_species, fractional, log_volume)
                delta_n_cu = 2 * old_species - 1  # Ni->Cu: +1; Cu->Ni: -1.
                log_ratio = species_log_acceptance(
                    beta, trial_energy - energy, delta_mu, delta_n_cu
                )
                move_accept = _accept(log_ratio, generator)
                species = torch.where(move_accept[:, None], trial_species, species)
                energy = torch.where(move_accept, trial_energy, energy)
                record(2, move_accept, production)

            traces["energy_eV"][sweep] = energy.double().cpu()
            traces["n_cu"][sweep] = species.eq(0).sum(1).short().cpu()
            traces["log_volume"][sweep] = log_volume.double().cpu()

            if not production and (sweep + 1) % config.adapt_interval == 0:
                rates = window_accepted / window_attempted.clamp_min(1)
                displacement_step *= torch.exp(rates[0] - config.target_acceptance)
                volume_step *= torch.exp(rates[1] - config.target_acceptance)
                window_accepted.zero_()
                window_attempted.zero_()

            if production and (sweep - config.burn_in) % config.config_thin == 0:
                config_sweeps.append(sweep)
                config_fractional.append(fractional.float().cpu())
                config_species.append(species.byte().cpu())

            next_sweep = sweep + 1
            if next_sweep % config.checkpoint_interval == 0:
                save_checkpoint(next_sweep)

    saved_fractional = torch.stack(config_fractional).numpy()
    saved_species_cu = 1 - torch.stack(config_species).numpy()
    initial_fractional = build_cuni_fcc(108).get_scaled_positions(wrap=False)
    paths = []
    move_names = ("displacement", "volume", "species")
    for chain, state in enumerate(states):
        path = output_path(output, state)
        metadata = {
            "schema": "janus-cuni-reference-v1",
            "n_atoms": atoms,
            "temperature_K": state.temperature,
            "delta_mu_Cu_minus_Ni_eV": state.delta_mu,
            "walker": state.walker,
            "initial_phase": "all-Ni" if state.walker == 0 else "all-Cu",
            "total_sweeps": config.total_sweeps,
            "burn_in": config.burn_in,
            "production_start": config.burn_in,
            "config_stride": config.config_thin,
            "config": asdict(config),
            "proposal_temperature_scaling": "sqrt(T/Tmax), Tmax=1200 K",
            "final_displacement_step": float(displacement_step[chain]),
            "final_log_volume_step": float(volume_step[chain]),
            "production_acceptance": {
                name: float(accepted[index, chain] / attempted[index, chain].clamp_min(1))
                for index, name in enumerate(move_names)
            },
            "production_accepted": {
                name: int(accepted[index, chain]) for index, name in enumerate(move_names)
            },
            "production_attempts": {
                name: int(attempted[index, chain]) for index, name in enumerate(move_names)
            },
            "potential": str(getattr(oracle, "path", "unknown")),
            "dtype": str(dtype),
        }
        _atomic_npz(
            path,
            energy_eV=traces["energy_eV"][:, chain].numpy(),
            n_cu=traces["n_cu"][:, chain].numpy(),
            log_volume=traces["log_volume"][:, chain].numpy(),
            config_sweeps=np.asarray(config_sweeps, dtype=np.int32),
            fractional_positions=saved_fractional[:, chain],
            initial_fractional_positions=initial_fractional,
            species=saved_species_cu[:, chain],  # canonical output encoding: 1=Cu, 0=Ni
            metadata=np.asarray(json.dumps(metadata)),
        )
        paths.append(path)
    checkpoint.unlink(missing_ok=True)
    return paths
