#!/usr/bin/env python3
"""WHAM analysis for unified BCT c/a umbrella windows."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from janus_reproduce.wham import umbrella_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path)
    parser.add_argument("--bias", type=float, default=1_000.0)
    args = parser.parse_args()
    shards = [torch.load(path, weights_only=True) for path in sorted(args.input.glob("shard_*.pt"))]
    data = {key: torch.cat([shard[key] for shard in shards]) for key in shards[0]}
    ratio = (data["cell_ac"][:, 1] / data["cell_ac"][:, 0]).numpy()
    centers = np.sort(data["center"].unique().numpy())
    counts = np.asarray([data["center"].eq(center).sum() for center in centers])
    result = umbrella_weights(ratio, centers, args.bias, counts)
    weight = result["weights"]
    midpoint = (1 + np.sqrt(2)) / 2
    bins = np.linspace(0.95, 1.48, 55)
    probability, edges = np.histogram(ratio, bins=bins, weights=weight)
    nonzero = probability > 0
    free_energy = np.full_like(probability, np.nan)
    free_energy[nonzero] = -np.log(probability[nonzero])
    free_energy[nonzero] -= np.nanmin(free_energy)
    overlap_bins = np.linspace(0.95, 1.48, 161)
    window_ratio = [ratio[data["center"].numpy() == center] for center in centers]
    adjacent_overlap = []
    for left, right in pairwise(window_ratio):
        left_hist = np.histogram(left, overlap_bins, density=True)[0]
        right_hist = np.histogram(right, overlap_bins, density=True)[0]
        adjacent_overlap.append(float(np.minimum(left_hist, right_hist).sum() * np.diff(overlap_bins)[0]))
    payload = {
        "status": "representative umbrella/WHAM validation",
        "samples": len(ratio),
        "wham_iterations": int(result["iterations"]),
        "wham_ess": float(result["ess"]),
        "bcc_side_probability": float(weight[ratio < midpoint].sum()),
        "fcc_side_probability": float(weight[ratio >= midpoint].sum()),
        "ratio_mean": float(np.sum(weight * ratio)),
        "adjacent_histogram_overlap": adjacent_overlap,
        "minimum_adjacent_overlap": min(adjacent_overlap),
        "overlap_gate_passed": min(adjacent_overlap) >= 0.05,
        "bin_centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "free_energy_kT": free_energy.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    if args.weights_output:
        args.weights_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(weight), args.weights_output)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
