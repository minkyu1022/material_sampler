#!/usr/bin/env python3
"""Evaluate a trained Cu--Ni JANUS checkpoint with hybrid path weights."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from janus_reproduce.alloy_model import AlloyPaiNN
from janus_reproduce.cuni import delta_mu_108, temperatures_108
from janus_reproduce.cuni_train import CuNiTrainConfig, _load_prior, _reference, _rollout
from janus_reproduce.torch_eam import TorchCuNiEAM


def load_checkpoint(path: Path, device: torch.device):
    saved = torch.load(path, map_location=device, weights_only=False)
    allowed = {field.name for field in fields(CuNiTrainConfig)}
    values = {key: value for key, value in saved["config"].items() if key in allowed}
    for key in ("potential", "output", "prior"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    config = CuNiTrainConfig(**values)
    model = AlloyPaiNN(
        features=config.features,
        layers=config.layers,
        radial_basis=config.radial_basis,
        cutoff=config.cutoff,
        temperature_reference=config.diffusion_temperature_ref,
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--condition", action="append", metavar="T,MU")
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error("require 0 <= --shard < --num-shards")

    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    if args.steps is not None:
        config = replace(config, steps=args.steps)
    torch.manual_seed(config.seed + args.shard)
    prior = _load_prior(config)
    oracle = TorchCuNiEAM(config.potential).to(device)
    reference = _reference(config.n_atoms, device)
    states = (
        [tuple(map(float, item.split(","))) for item in args.condition]
        if args.condition
        else [(t, mu) for t in temperatures_108() for mu in delta_mu_108()]
    )
    if args.limit_states is not None:
        states = states[: args.limit_states]
    states = states[args.shard :: args.num_shards]

    rows, raw = [], []
    for index, (temperature, delta_mu) in enumerate(states, 1):
        result = _rollout(
            model,
            config,
            prior,
            reference,
            args.samples,
            device,
            path_weights=True,
            oracle=oracle,
            temperature=torch.tensor(temperature, device=device),
            delta_mu=torch.tensor(delta_mu, device=device),
        )
        weights = result["normalized_weight"]
        cu_fraction = result["species"].eq(0).float().mean(-1).double()
        cell_length = (result["log_volume"] / 3).exp()
        displacement = (
            result["displacement"].norm(dim=-1).mean(-1) * cell_length
        ).double()
        atomic_volume = result["log_volume"].double().exp() / config.n_atoms
        rows.append(
            [
                temperature,
                delta_mu,
                result["log_xi"].item(),
                result["ess"].item(),
                torch.dot(weights, cu_fraction).item(),
                torch.dot(weights, displacement).item(),
                torch.dot(weights, atomic_volume).item(),
                cu_fraction.mean().item(),
                displacement.mean().item(),
                atomic_volume.mean().item(),
                result["log_weight"].double().std().item(),
                result["log_continuous_u"].double().std().item(),
                result["log_continuous_v"].double().std().item(),
                result["log_q_discrete"].double().std().item(),
            ]
        )
        if args.save_raw:
            raw.append({key: value.cpu() for key, value in result.items()})
        print(f"state={index}/{len(states)} T={temperature:g} mu={delta_mu:g} ESS={rows[-1][3]:.3g}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        columns=np.array([
            "temperature", "delta_mu", "log_xi", "ess", "cu_fraction",
            "mean_displacement", "atomic_volume", "raw_cu_fraction",
            "raw_mean_displacement", "raw_atomic_volume", "log_weight_std",
            "log_continuous_u_std", "log_continuous_v_std", "log_q_discrete_std",
        ]),
        values=np.asarray(rows),
        checkpoint=str(args.checkpoint),
        samples=args.samples,
    )
    if args.save_raw:
        torch.save(raw, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
