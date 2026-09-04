#!/usr/bin/env python3
"""Document unified-model graph connectivity across the Bain path."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from janus_reproduce.alloy_model import minimum_image_displacements_cell
from janus_reproduce.nicr_unified_bct2d import cell_matrix, reference_sites


def connected(adjacency: torch.Tensor) -> bool:
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbour in torch.where(adjacency[node])[0].tolist():
            if neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
    return len(seen) == len(adjacency)


def main() -> None:
    output = Path("outputs/nicr_unified_bct2d_N128/graph_cutoff_validation.json")
    sites = reference_sites()[None]
    primitive_volume = 2.765**3
    rows = []
    for ratio in torch.linspace(1, math.sqrt(2), 9):
        a = (primitive_volume / float(ratio)) ** (1 / 3)
        cell = cell_matrix(torch.tensor([[a, a * float(ratio)]], dtype=torch.float64))
        distance = minimum_image_displacements_cell(sites, cell).norm(dim=-1)[0]
        for cutoff in (5.0, 5.3, 6.0):
            adjacency = distance.lt(cutoff) & distance.gt(1e-10)
            degree = adjacency.sum(1)
            rows.append(
                {
                    "c_over_a": float(ratio),
                    "cutoff_A": cutoff,
                    "degree_min": int(degree.min()),
                    "degree_mean": float(degree.double().mean()),
                    "degree_max": int(degree.max()),
                    "connected": connected(adjacency),
                }
            )
    payload = {
        "status": "new-model graph hyperparameter validation; not paper-faithful",
        "selected_cutoff_A": 5.3,
        "criterion": "connected at every sampled Bain ratio with no zero-degree sites",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
