#!/usr/bin/env python3
"""Combine sharded JANUS evaluation and compare it with N=108 reference MC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from janus_reproduce.cuni_reference_analysis import mixing_free_energy_from_semigrand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shards = sorted(args.input.glob("shard_*.npz"))
    if not shards:
        raise SystemExit("no completed shards")
    values = np.concatenate([np.load(path)["values"] for path in shards])
    values = values[np.lexsort((values[:, 1], values[:, 0]))]
    temperatures, mus = np.unique(values[:, 0]), np.unique(values[:, 1])
    if values.shape[0] != temperatures.size * mus.size:
        raise SystemExit(f"incomplete grid: {values.shape[0]}/{temperatures.size * mus.size}")
    grid = values.reshape(temperatures.size, mus.size, -1)
    reference = np.load(args.reference)
    if not (np.array_equal(temperatures, reference["temperatures_K"]) and np.array_equal(mus, reference["delta_mu_eV"])):
        raise SystemExit("JANUS and reference grids differ")
    args.output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), sharex=True, sharey=True)
    panels = (reference["raw_mean_x_cu"], grid[:, :, 7], grid[:, :, 4], grid[:, :, 3])
    titles = ("Reference MC", "JANUS raw", "JANUS path-weighted", "JANUS ESS (64 paths)")
    for axis, panel, title in zip(axes, panels, titles, strict=True):
        image = axis.pcolormesh(mus, temperatures, panel, shading="nearest", cmap="viridis")
        axis.set(title=title, xlabel=r"$\Delta\mu_{Cu-Ni}$ (eV)")
        fig.colorbar(image, ax=axis)
    axes[0].set_ylabel("Temperature (K)")
    fig.tight_layout()
    fig.savefig(args.output / "composition_and_ess.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for target in (600.0, 900.0, 1200.0):
        row = np.argmin(abs(temperatures - target))
        order = np.argsort(grid[row, :, 4])
        x = grid[row, order, 4]
        axes[0].plot(x, grid[row, order, 5], ".-", label=f"{temperatures[row]:g} K")
        axes[1].plot(x, grid[row, order, 6], ".-")
        reference_order = np.argsort(reference["raw_mean_x_cu"][row])
        reference_x = reference["raw_mean_x_cu"][row, reference_order]
        axes[0].plot(
            reference_x, reference["mean_displacement_A"][row, reference_order], "--",
            color=axes[0].lines[-1].get_color(),
        )
        axes[1].plot(
            reference_x, reference["atomic_volume_A3"][row, reference_order], "--",
            color=axes[1].lines[-1].get_color(),
        )
    axes[0].set(xlabel=r"$\langle x_{Cu}\rangle$", ylabel=r"mean displacement ($\AA$)")
    axes[1].set(xlabel=r"$\langle x_{Cu}\rangle$", ylabel=r"$V/N$ ($\AA^3$/atom)")
    axes[0].legend()
    fig.suptitle("solid: JANUS path-weighted; dashed: reference MC")
    fig.tight_layout()
    fig.savefig(args.output / "weighted_displacement_and_volume.png", dpi=200)
    plt.close(fig)

    diagnostics = {"states": int(values.shape[0]), "samples_per_state": 64, "ess": {}}
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for target in (600.0, 900.0, 1200.0):
        row = np.argmin(abs(temperatures - target))
        x, g_mix = mixing_free_energy_from_semigrand(-grid[row, :, 2], mus, temperatures[row], 108)
        ax.plot(x, 1000 * g_mix, label=f"JANUS {temperatures[row]:g} K")
    reference_manifest = json.loads((args.reference.parent / "diagnostics_and_conventions.json").read_text())
    for key, curve in reference_manifest["free_energy_curves"].items():
        ax.plot(curve["x"], 1000 * np.asarray(curve["Gmix_eV_atom"]), "--", label=f"MC {key[2:]} K")
    ax.set(xlabel=r"$x_{Cu}$", ylabel=r"$G_{mix}/N$ (meV/atom)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "mixing_free_energy_comparison.png", dpi=200)
    plt.close(fig)

    ess = grid[:, :, 3]
    diagnostics["ess"] = {
        "min": float(ess.min()), "median": float(np.median(ess)), "max": float(ess.max()),
        "states_ge_10": int(np.sum(ess >= 10)), "states_ge_32": int(np.sum(ess >= 32)),
    }
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    np.savez_compressed(args.output / "combined.npz", values=values)
    print(json.dumps(diagnostics))


if __name__ == "__main__":
    main()
