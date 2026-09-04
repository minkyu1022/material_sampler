#!/usr/bin/env python3
"""Fit Cartesian/cell normalization used by the Cu-Ni Packora-style branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.lattice_repr import ltri_params_to_lattice_matrix, y1_to_ltri_params

torch.set_num_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = torch.load(args.input, weights_only=False)
    coord_sum = torch.zeros(3, dtype=torch.float64)
    coord_sq = torch.zeros(3, dtype=torch.float64)
    lattice = []
    atoms = 0
    for item in items:
        y1 = item["Y1"].to(torch.float64)
        latent = y1_to_ltri_params(y1)
        matrix = ltri_params_to_lattice_matrix(latent)
        cart = item["F1"].to(torch.float64) @ matrix
        cart = cart - cart.mean(0, keepdim=True)
        coord_sum += cart.sum(0)
        coord_sq += cart.square().sum(0)
        atoms += len(cart)
        lattice.append(latent)
    lattice_tensor = torch.stack(lattice)
    coord_mean = coord_sum / atoms
    coord_std = (coord_sq / atoms - coord_mean.square()).clamp_min(1e-12).sqrt()
    report = {
        "schema": "crystalite-cartesian-ltri-normalization-v1",
        "structures": len(items),
        "atoms": atoms,
        "coordinate_gauge": "canonical-cell Cartesian coordinates with arithmetic COM removed",
        "coord_mean": coord_mean.tolist(),
        "coord_std": coord_std.tolist(),
        "lattice_mean": lattice_tensor.mean(0).tolist(),
        "lattice_std": lattice_tensor.std(0, correction=0).clamp_min(1e-8).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
