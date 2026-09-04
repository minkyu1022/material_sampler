#!/usr/bin/env python3
"""Generate harmonic c/a umbrella windows for one unified Ni--Cr state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from janus_reproduce.nicr_unified_reference import ReferenceMCConfig, reference_mc
from janus_reproduce.torch_eam import TorchEAM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--temperature", type=float, default=1050.0)
    parser.add_argument("--target-cr", type=int, default=64)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--sweeps", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--bias", type=float, default=1_000.0)
    parser.add_argument("--windows", type=int, default=9)
    args = parser.parse_args()
    device = torch.device(args.device)
    oracle = TorchEAM(args.potential, species_indices=(0, 2)).to(device)
    centers = torch.linspace(1, math.sqrt(2), args.windows)
    rows, stats = [], []
    primitive_volume = 2.765**3
    for index, center in enumerate(centers):
        if index % args.shards != args.shard:
            continue
        ratio = float(center)
        a = (primitive_volume / ratio) ** (1 / 3)
        result = reference_mc(
            oracle,
            args.target_cr,
            args.temperature,
            ReferenceMCConfig(
                sweeps=args.sweeps,
                burn_in=args.burn_in,
                thin=5,
                species_moves=6,
                bain_moves=0,
                ratio_center=ratio,
                ratio_bias=args.bias,
            ),
            initial_cell_ac=torch.tensor((a, a * ratio), device=device),
            generator=torch.Generator(device=device).manual_seed(30_000 + index),
        )
        count = len(result["species"])
        rows.append(
            {key: result[key].cpu() for key in result if key != "stats"}
            | {
                "window": torch.full((count,), index),
                "center": torch.full((count,), ratio),
                "temperature": torch.full((count,), args.temperature),
                "target_cr": torch.full((count,), args.target_cr),
            }
        )
        stats.append({"window": index, "center": ratio, "acceptance": result["stats"]})
    data = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
