#!/usr/bin/env python3
"""Generate a sharded BCC/FCC-seeded unified Ni--Cr reference buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from janus_reproduce.nicr_unified_reference import ReferenceMCConfig, reference_mc
from janus_reproduce.torch_eam import TorchEAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--sweeps", type=int, default=50)
    parser.add_argument("--burn-in", type=int, default=25)
    args = parser.parse_args()
    device = torch.device(args.device)
    oracle = TorchEAM(args.potential, species_indices=(0, 2)).to(device)
    conditions = [
        (temperature, n_cr, phase, cell)
        for temperature in (600.0, 1050.0, 1500.0)
        for n_cr in (16, 32, 64, 96, 112)
        for phase, cell in (("bcc", (2.765, 2.765)), ("fcc", (2.462, 3.482)))
    ]
    rows, stats = [], []
    for index, (temperature, n_cr, phase, cell) in enumerate(conditions):
        if index % args.shards != args.shard:
            continue
        result = reference_mc(
            oracle,
            n_cr,
            temperature,
            ReferenceMCConfig(
                sweeps=args.sweeps,
                burn_in=args.burn_in,
                thin=5,
                species_moves=6,
            ),
            initial_cell_ac=torch.tensor(cell, device=device),
            generator=torch.Generator(device=device).manual_seed(10_000 + index),
        )
        count = len(result["species"])
        rows.append(
            {
                key: result[key].cpu()
                for key in ("species", "disp_u", "cell_z", "cell_ac", "log_density")
            }
            | {
                "temperature": torch.full((count,), temperature),
                "target_cr": torch.full((count,), n_cr),
                "seed_phase": torch.full((count,), 0 if phase == "bcc" else 1),
                "chain_id": torch.full((count,), index),
            }
        )
        stats.append(
            {
                "temperature": temperature,
                "target_cr": n_cr,
                "seed_phase": phase,
                "acceptance": result["stats"],
            }
        )
    data = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "target_measure": "zero-COM fractional u: V^(N-1) da dc",
                "potential": str(args.potential),
                "stats": stats,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"shard": args.shard, "samples": len(data["species"]), "stats": stats}))


if __name__ == "__main__":
    main()
