#!/usr/bin/env python3
"""Opposing-seed convergence diagnostics for unified BCT reference chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def integrated_time(values: np.ndarray) -> float:
    centered = values - values.mean()
    variance = np.dot(centered, centered) / len(values)
    if variance == 0:
        return float("inf")
    total = 1.0
    for lag in range(1, len(values)):
        correlation = np.dot(centered[:-lag], centered[lag:]) / ((len(values) - lag) * variance)
        if correlation <= 0:
            break
        total += 2 * correlation
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shards = [torch.load(path, weights_only=True) for path in sorted(args.reference.glob("shard_*.pt"))]
    data = {key: torch.cat([shard[key] for shard in shards]) for key in shards[0]}
    ratio = (data["cell_ac"][:, 1] / data["cell_ac"][:, 0]).numpy()
    rows = []
    for temperature in sorted(data["temperature"].unique().tolist()):
        for target_cr in sorted(data["target_cr"].unique().tolist()):
            condition = data["temperature"].eq(temperature) & data["target_cr"].eq(target_cr)
            chains = [ratio[condition.numpy() & data["seed_phase"].eq(seed).numpy()] for seed in (0, 1)]
            if not all(len(chain) for chain in chains):
                continue
            means = [float(chain.mean()) for chain in chains]
            within = sum(np.var(chain, ddof=1) for chain in chains) / 2
            between = len(chains[0]) * np.var(means, ddof=1)
            variance = (len(chains[0]) - 1) / len(chains[0]) * within + between / len(chains[0])
            rhat = float(np.sqrt(variance / within)) if within > 0 else float("inf")
            tau = [integrated_time(chain) for chain in chains]
            rows.append(
                {
                    "temperature": temperature,
                    "target_cr": target_cr,
                    "mean_c_over_a_bcc_seed": means[0],
                    "mean_c_over_a_fcc_seed": means[1],
                    "rhat": rhat,
                    "ess_bcc_seed": len(chains[0]) / tau[0],
                    "ess_fcc_seed": len(chains[1]) / tau[1],
                    "bcc_to_fcc_fraction": float(np.mean(chains[0] > (1 + np.sqrt(2)) / 2)),
                    "fcc_to_bcc_fraction": float(np.mean(chains[1] < (1 + np.sqrt(2)) / 2)),
                }
            )
    payload = {
        "status": "reference convergence diagnostic",
        "samples": len(ratio),
        "all_rhat_below_1.1": all(row["rhat"] < 1.1 for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
