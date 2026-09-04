#!/usr/bin/env python3
"""Render Cu--Ni N=108 panels with the JANUS paper's axes and palette."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from janus_reproduce.cuni_reference_analysis import mixing_free_energy_from_semigrand

COLORS = ("#0072B2", "#009E73", "#D55E00")
COMPOSITION_CMAP = LinearSegmentedColormap.from_list(
    "janus_composition", ("#8294A0", "#F3F1ED", "#CB7035")
)


def crossover(mus: np.ndarray, composition: np.ndarray) -> float:
    order = np.argsort(composition)
    return float(np.interp(0.5, composition[order], mus[order]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = np.concatenate([np.load(path)["values"] for path in sorted(args.input.glob("shard_*.npz"))])
    values = values[np.lexsort((values[:, 1], values[:, 0]))]
    temperatures, mus = np.unique(values[:, 0]), np.unique(values[:, 1])
    grid = values.reshape(len(temperatures), len(mus), -1)
    reference = np.load(args.reference)
    keep = temperatures >= 600
    args.output.mkdir(parents=True, exist_ok=True)

    for name, column in (("weighted", 4), ("raw", 7)):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True)
        panels = (reference["raw_mean_x_cu"][keep], grid[keep, :, column])
        for axis, panel, title in zip(axes, panels, ("Reference", "JANUS"), strict=True):
            image = axis.pcolormesh(
                mus, temperatures[keep], panel, shading="nearest", cmap=COMPOSITION_CMAP,
                vmin=0, vmax=1,
            )
            line = [crossover(mus, row) for row in panel]
            axis.plot(line, temperatures[keep], "k--", linewidth=1.4)
            axis.set(title=title, xlim=(0.59, 1.16), ylim=(575, 1250), xticks=(0.6, 0.8, 1.0),
                     yticks=(600, 800, 1000, 1200), xlabel=r"$\Delta\mu$ (eV)")
        axes[0].set_ylabel("T (K)")
        colorbar = fig.colorbar(image, ax=axes, ticks=(0, 0.5, 1), fraction=0.045, pad=0.04)
        colorbar.set_label(r"$\langle x_{Cu}\rangle$")
        fig.subplots_adjust(left=0.1, right=0.88, bottom=0.16, top=0.86, wspace=0.08)
        fig.savefig(args.output / f"figure2b_{name}.png", dpi=240)
        plt.close(fig)

    for name, displacement_column, volume_column in (
        ("weighted", 5, 6), ("raw", 8, 9)
    ):
        fig, axes = plt.subplots(2, 1, figsize=(5.2, 6.0), sharex=True)
        for target, color in zip((600.0, 900.0, 1200.0), COLORS, strict=True):
            row = int(np.argmin(abs(temperatures - target)))
            axes[0].plot(mus, grid[row, :, displacement_column], color=color, linewidth=2,
                         label=f"{target:g} K")
            axes[1].plot(mus, grid[row, :, volume_column], color=color, linewidth=2)
            axes[0].plot(mus, reference["mean_displacement_A"][row], "--", color="#333333",
                         linewidth=1.4)
            axes[1].plot(mus, reference["atomic_volume_A3"][row], "--", color="#333333",
                         linewidth=1.4)
        axes[0].set(ylabel=r"$\langle|u|\rangle$ (Å)", ylim=(0.13, 0.29),
                    yticks=(0.15, 0.20, 0.25))
        axes[1].set(xlabel=r"$\Delta\mu$ (eV)", ylabel=r"$\langle V\rangle$ ($\AA^3$/atom)",
                    xlim=(0.6, 1.15), xticks=np.arange(0.6, 1.11, 0.1), ylim=(10.95, 12.7),
                    yticks=(11.0, 11.5, 12.0, 12.5))
        axes[0].plot([], [], "--", color="#333333", label="Reference")
        axes[0].legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(args.output / f"figure2c_{name}.png", dpi=240)
        plt.close(fig)

    manifest = json.loads((args.reference.parent / "diagnostics_and_conventions.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for target, color in zip((600.0, 900.0, 1200.0), COLORS, strict=True):
        row = int(np.argmin(abs(temperatures - target)))
        x, janus = mixing_free_energy_from_semigrand(-grid[row, :, 2], mus, temperatures[row], 108)
        reference_curve = manifest["free_energy_curves"][f"T={target:g}"]
        ax.plot(x, 1000 * janus, color=color, linewidth=2, label=f"{target:g} K")
        ax.plot(reference_curve["x"], 1000 * np.asarray(reference_curve["Gmix_eV_atom"]),
                "--", color="#333333", linewidth=1.4)
    ax.axhline(0, color="#999999", linestyle=":", linewidth=1)
    ax.set(xlabel=r"$\langle x_{Cu}\rangle$", ylabel=r"$G_{mix}/N$ (meV)", xlim=(0, 1),
           xticks=np.arange(0, 1.01, 0.2), ylim=(-50, 5), yticks=(-40, -20, 0))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "figure2e_N108.png", dpi=240)


if __name__ == "__main__":
    main()
