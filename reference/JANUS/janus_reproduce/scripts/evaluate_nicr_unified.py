#!/usr/bin/env python3
"""Evaluate a unified Ni--Cr checkpoint without modifying training state."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import torch
from self_bootstrap_nicr_unified import _metrics

from janus_reproduce.free_energy import path_weight_estimates
from janus_reproduce.nicr_unified_bct2d import JANUSUnifiedBCT2D
from janus_reproduce.nicr_unified_train import UnifiedTrainConfig, rollout
from janus_reproduce.thermodynamics import mixing_free_energy_from_log_partitions
from janus_reproduce.torch_eam import TorchEAM


def _batched_rollout(model, oracle, config, temperature, target_cr, samples, batch_size, generator):
    chunks = []
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        chunks.append(
            rollout(
                model,
                torch.full((count,), target_cr, device=next(model.parameters()).device),
                torch.full((count,), temperature, device=next(model.parameters()).device),
                config,
                generator=generator,
                path_weights=True,
                oracle=oracle,
            )
        )
        print(
            json.dumps(
                {
                    "phase": "rollout",
                    "temperature": temperature,
                    "target_cr": target_cr,
                    "completed": start + count,
                    "samples": samples,
                }
            ),
            flush=True,
        )
    result = {
        key: torch.cat([chunk[key] for chunk in chunks])
        for key in chunks[0]
        if key not in {"log_xi", "normalized_weight", "ess"}
    }
    if result["valid_domain"].any():
        result["log_xi"], result["normalized_weight"], result["ess"] = path_weight_estimates(
            result["log_weight"]
        )
    else:
        result["log_xi"] = result["log_weight"].new_tensor(float("-inf"))
        result["normalized_weight"] = torch.zeros_like(result["log_weight"])
        result["ess"] = result["log_weight"].new_zeros(())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperatures", type=float, nargs="+", default=(600, 1050, 1200, 1500))
    parser.add_argument(
        "--target-cr-values", type=int, nargs="+", default=(0, 32, 64, 96, 112, 128)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--rollout-velocity-cell-clip", type=float)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint["args"]
    config = UnifiedTrainConfig(**checkpoint["config"])
    if args.rollout_velocity_cell_clip is not None:
        config = replace(config, rollout_velocity_cell_clip=args.rollout_velocity_cell_clip)
    model = JANUSUnifiedBCT2D(
        features=saved_args["features"], layers=saved_args["layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    oracle = TorchEAM(args.potential, species_indices=(0, 2)).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    metrics, samples = [], {}
    for temperature, target_cr in product(args.temperatures, args.target_cr_values):
        started = time.perf_counter()
        result = _batched_rollout(
            model, oracle, config, temperature, target_cr, args.samples, args.batch_size, generator
        )
        attempted = args.samples
        key = f"T{temperature:g}_Cr{target_cr}"
        metric = _metrics(result, temperature, target_cr, attempted, attempted)
        metric["wall_seconds"] = time.perf_counter() - started
        metric["attempted_samples_per_second"] = attempted / metric["wall_seconds"]
        metrics.append(metric)
        if args.save_samples:
            samples[key] = {name: value.detach().cpu() for name, value in result.items()}
        print(json.dumps(metrics[-1]), flush=True)

    mixing_free_energy, mixing_free_energy_notes = {}, {}
    for temperature in args.temperatures:
        rows = sorted(
            (row for row in metrics if row["temperature"] == temperature),
            key=lambda row: row["target_cr"],
        )
        key = f"T{temperature:g}"
        if rows[0]["target_cr"] != 0 or rows[-1]["target_cr"] != 128:
            mixing_free_energy_notes[key] = "unavailable: pure Ni/Cr endpoints were not evaluated"
        elif not all(math.isfinite(row["log_xi"]) for row in rows):
            mixing_free_energy_notes[key] = "unavailable: at least one log-partition estimate is non-finite"
        else:
            x, g_mix = mixing_free_energy_from_log_partitions(
                [row["target_cr"] for row in rows],
                [row["log_xi"] for row in rows],
                temperature,
                128,
            )
            mixing_free_energy[key] = {
                "x_cr": x.tolist(),
                "g_mix_eV_per_atom": g_mix.tolist(),
                "estimator": "direct fixed-composition path-weight log-partition estimate",
            }
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_outer_round": checkpoint["outer_round"],
        "samples_per_condition": args.samples,
        "seed": args.seed,
        "config": asdict(config),
        "conditions": metrics,
        "mixing_free_energy": mixing_free_energy,
        "mixing_free_energy_notes": mixing_free_energy_notes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.save_samples:
        torch.save(samples, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
