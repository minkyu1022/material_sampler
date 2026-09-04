#!/usr/bin/env python3
"""Evaluate candidate Ni--Cr JANUS rungs and retain inputs for weighted BAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from janus_reproduce.alloy_model import AlloyPaiNN
from janus_reproduce.free_energy import path_weight_estimates
from janus_reproduce.nicr import substitution_energies
from janus_reproduce.nicr_train import (
    NiCrTrainConfig,
    _prior,
    _reference,
    effective_hamiltonian_manifest,
    resolved_config,
    rollout_nicr,
)
from janus_reproduce.torch_eam import TorchEAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--rung-start", type=int, default=0)
    parser.add_argument("--rung-stop", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = NiCrTrainConfig.from_json(args.config)
    hamiltonian = effective_hamiltonian_manifest(config)
    stop = config.spec.n_atoms + 1 if args.rung_stop is None else args.rung_stop
    if not 0 <= args.rung_start < stop <= config.spec.n_atoms + 1:
        raise ValueError("invalid rung interval")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.output / "checkpoint.pt", map_location=device, weights_only=False)
    if checkpoint.get("config") != resolved_config(config):
        raise ValueError("checkpoint config does not match requested candidate config")
    if checkpoint.get("provenance", {}).get("effective_hamiltonian") != hamiltonian:
        raise ValueError("checkpoint Hamiltonian provenance mismatch")

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
    model.load_state_dict(checkpoint["model"])
    model.eval()
    oracle = TorchEAM(
        config.potential, species_indices=(0, 2), cutoff=config.target_cutoff
    ).to(device)
    prior, reference = _prior(config), _reference(config, device)
    args.output.mkdir(parents=True, exist_ok=True)

    for n_cr in range(args.rung_start, stop):
        if n_cr % args.shard_count != args.shard_index:
            continue
        generator = torch.Generator(device=device).manual_seed(args.seed + 10_000 * int(args.temperature) + n_cr)
        chunks = []
        for start in range(0, args.samples, args.batch_size):
            count = min(args.batch_size, args.samples - start)
            chunks.append(
                rollout_nicr(
                    model,
                    oracle,
                    config,
                    prior,
                    reference,
                    torch.full((count,), args.temperature, device=device),
                    torch.full((count,), n_cr, dtype=torch.long, device=device),
                    generator=generator,
                    path_weights=True,
                )
            )
        result = {
            key: torch.cat([part[key] for part in chunks])
            for key in ("species", "displacement", "log_volume", "energy", "log_weight")
        }
        log_xi, _, ess = path_weight_estimates(result["log_weight"])
        forward, reverse = [], []
        for index in range(args.samples):
            f, r = substitution_energies(
                oracle,
                result["species"][index : index + 1],
                reference[None] + result["displacement"][index : index + 1],
                result["log_volume"][index : index + 1],
            )
            forward.append(f.cpu())
            reverse.append(r.cpu())
        payload = {
            "phase": config.phase,
            "temperature": args.temperature,
            "target_cr": n_cr,
            "samples": args.samples,
            "checkpoint_round": checkpoint["round"],
            "ess": float(ess),
            "log_xi": float(log_xi),
            "std_log_weight": float(result["log_weight"].std(unbiased=False)),
            "mean_energy_per_atom": float(result["energy"].mean() / config.spec.n_atoms),
            "mean_rms_u": float(result["displacement"].square().sum(-1).mean(-1).sqrt().mean()),
            "mean_volume_per_atom": float(result["log_volume"].exp().mean() / config.spec.n_atoms),
            "finite": bool(torch.isfinite(result["log_weight"]).all()),
            "effective_hamiltonian": hamiltonian,
        }
        stem = args.output / f"T{args.temperature:g}_n{n_cr}"
        stem.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
        torch.save(
            {
                "log_weight": result["log_weight"].cpu(),
                "forward_delta_u": torch.cat(forward) if forward else torch.empty(args.samples, 0),
                "reverse_delta_u": torch.cat(reverse) if reverse else torch.empty(args.samples, 0),
            },
            stem.with_suffix(".pt"),
        )
        print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
