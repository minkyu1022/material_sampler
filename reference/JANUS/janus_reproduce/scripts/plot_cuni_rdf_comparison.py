#!/usr/bin/env python3
"""Compare path-weighted JANUS and reference-MC partial RDFs at 800 K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from janus_reproduce.cuni_reference_analysis import averaged_partial_rdf, load_chain, state_thinning
from janus_reproduce.cuni_train import _reference


def weighted_rdf(result: dict[str, torch.Tensor], reference: np.ndarray, *, raw: bool = False):
    edges = np.linspace(0, 5.3, 266)
    shell = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
    totals = {name: np.zeros(265) for name in ("Cu-Cu", "Cu-Ni", "Ni-Ni")}
    species = result["species"].numpy()
    fractional = reference[None] + result["displacement"].numpy()
    volumes = result["log_volume"].double().exp().numpy()
    weights = (
        np.full(len(species), 1 / len(species))
        if raw else result["normalized_weight"].numpy()
    )
    upper = np.triu_indices(species.shape[1], 1)
    for symbols, positions, volume, weight in zip(species, fractional, volumes, weights, strict=True):
        delta = positions[:, None] - positions[None, :]
        delta -= np.round(delta)
        distance = np.linalg.norm(delta * np.cbrt(volume), axis=-1)[upper]
        cu = symbols == 0
        a, b = cu[upper[0]], cu[upper[1]]
        masks = {"Cu-Cu": a & b, "Cu-Ni": a ^ b, "Ni-Ni": ~a & ~b}
        n_cu, atoms = int(cu.sum()), len(cu)
        possible = {
            "Cu-Cu": n_cu * (n_cu - 1) / 2,
            "Cu-Ni": n_cu * (atoms - n_cu),
            "Ni-Ni": (atoms - n_cu) * (atoms - n_cu - 1) / 2,
        }
        for name, mask in masks.items():
            if possible[name]:
                totals[name] += weight * np.histogram(distance[mask], bins=edges)[0] * volume / (
                    possible[name] * shell
                )
    return (edges[:-1] + edges[1:]) / 2, totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--reference-chains", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unweighted", action="store_true")
    args = parser.parse_args()
    results = torch.load(args.raw, map_location="cpu", weights_only=False)
    selections = json.loads(args.reference_manifest.read_text())["rdf_selection"]
    reference = _reference(108, torch.device("cpu")).numpy()
    colors = ("#CC8B51", "#B3531E", "#71320F")
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.4), sharex=True)
    for result, selection, color in zip(results, selections, colors, strict=True):
        radius, janus = weighted_rdf(result, reference, raw=args.unweighted)
        mu = selection["state"]["delta_mu"]
        paths = sorted(args.reference_chains.glob(f"N108_T0800.00_mu0{mu:.4f}_walker*.npz"))
        chains = [load_chain(path) for path in paths]
        ref_radius, ref = averaged_partial_rdf(chains, state_thinning(chains), r_max=5.3, dr=0.02)
        composition = result["species"].eq(0).float().mean(-1).double()
        x = (
            composition.mean()
            if args.unweighted else torch.dot(result["normalized_weight"], composition)
        )
        for axis, pair in zip(axes, ("Cu-Cu", "Ni-Ni"), strict=True):
            axis.plot(radius, janus[pair], color=color, linewidth=2, label=rf"$\langle x_{{Cu}}\rangle={x:.2f}$")
            axis.plot(ref_radius, ref[pair], "--", color="#333333", linewidth=1.3)
    axes[0].set(ylabel=r"$g_{Cu-Cu}$", xlim=(2.0, 5.25), ylim=(0, 6.2), yticks=(0, 2.5, 5.0))
    axes[1].set(xlabel="r (Å)", ylabel=r"$g_{Ni-Ni}$", xlim=(2.0, 5.25), ylim=(0, 6.2),
                    yticks=(0, 2.5, 5.0), xticks=(2, 2.5, 3, 3.5, 4, 4.5, 5))
    axes[0].plot([], [], "--", color="#333333", label="Reference")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
