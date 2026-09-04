#!/usr/bin/env python3
"""Aggregate adjacent candidate Ni--Cr rungs with importance-weighted BAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from janus_reproduce.free_energy import canonical_ladder_bar
from janus_reproduce.thermodynamics import KB_EV, mixing_free_energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--n-atoms", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows, tensors = [], []
    for n_cr in range(args.n_atoms + 1):
        stem = args.input / f"T{args.temperature:g}_n{n_cr}"
        rows.append(json.loads(stem.with_suffix(".json").read_text()))
        tensors.append(torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=False))
    manifest = rows[0]["effective_hamiltonian"]
    if any(row["effective_hamiltonian"] != manifest for row in rows):
        raise ValueError("mixed Hamiltonians in rung inputs")
    if any(not row["finite"] for row in rows):
        raise ValueError("non-finite rung weights")

    beta = 1 / (KB_EV * args.temperature)
    rng = np.random.default_rng(args.seed)
    edges = []
    for n_cr in range(args.n_atoms):
        left, right = tensors[n_cr], tensors[n_cr + 1]
        edge = canonical_ladder_bar(
            left["forward_delta_u"].numpy(),
            right["reverse_delta_u"].numpy(),
            beta,
            args.n_atoms,
            n_cr,
            left["log_weight"].numpy(),
            right["log_weight"].numpy(),
        )
        bootstrap = []
        for _ in range(args.bootstrap):
            li = rng.integers(len(left["log_weight"]), size=len(left["log_weight"]))
            ri = rng.integers(len(right["log_weight"]), size=len(right["log_weight"]))
            try:
                bootstrap.append(
                    canonical_ladder_bar(
                        left["forward_delta_u"].numpy()[li],
                        right["reverse_delta_u"].numpy()[ri],
                        beta,
                        args.n_atoms,
                        n_cr,
                        left["log_weight"].numpy()[li],
                        right["log_weight"].numpy()[ri],
                    )["delta_beta_g"]
                )
            except ValueError:
                pass
        edge |= {
            "n_cr": n_cr,
            "bootstrap_success": len(bootstrap),
            "bootstrap_std_delta_beta_g": float(np.std(bootstrap, ddof=1)) if len(bootstrap) > 1 else None,
            "reliable": min(edge["forward_ess"], edge["reverse_ess"]) >= 2 and len(bootstrap) >= 0.8 * args.bootstrap,
        }
        edges.append(edge)
    beta_g = np.r_[0.0, np.cumsum([edge["delta_beta_g"] for edge in edges])]
    g_eV = beta_g / beta
    absolute_g_eV = g_eV - rows[0]["log_xi"] / beta
    payload = {
        "phase": rows[0]["phase"],
        "temperature": args.temperature,
        "n_atoms": args.n_atoms,
        "effective_hamiltonian": manifest,
        "estimator": "importance-weighted adjacent-rung BAR",
        "free_energy_reference": "G(n_Cr=0)=0 within this lattice",
        "n_cr": list(range(args.n_atoms + 1)),
        "x_cr": (np.arange(args.n_atoms + 1) / args.n_atoms).tolist(),
        "relative_free_energy_eV": g_eV.tolist(),
        "absolute_free_energy_eV": absolute_g_eV.tolist(),
        "mixing_free_energy_eV_per_atom": mixing_free_energy(g_eV, args.n_atoms).tolist(),
        "rung_ess": [row["ess"] for row in rows],
        "rung_std_log_weight": [row["std_log_weight"] for row in rows],
        "edges": edges,
        "reliability_note": "Edges with ESS < 2 or <80% successful bootstrap solves are retained but flagged; no edge is silently dropped.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"phase": payload["phase"], "temperature": args.temperature, "edges": len(edges), "flagged_edges": sum(not edge["reliable"] for edge in edges)}))


if __name__ == "__main__":
    main()
