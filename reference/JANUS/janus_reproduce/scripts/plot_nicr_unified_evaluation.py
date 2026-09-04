#!/usr/bin/env python3
"""Plot unified Ni--Cr free energy and structural-mode diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from janus_reproduce.free_energy import common_tangent, lower_convex_hull


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    median_ess = statistics.median(row["ess"] for row in data["conditions"])
    quality_warning = "LOW ESS — diagnostic only" if median_ess < 10 else None

    curves = data["mixing_free_energy"]
    if curves:
        fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
        for key, curve in curves.items():
            x = np.asarray(curve["x_cr"])
            g = 1_000 * np.asarray(curve["g_mix_eV_per_atom"])
            ax.plot(x, g, "o-", label=key.removeprefix("T") + " K")
            if key == "T1200":
                hull = lower_convex_hull(x, g)
                ax.plot(x[hull], g[hull], "k--", lw=1, label="1200 K lower hull")
                tangent = common_tangent(x, g)
                if tangent:
                    left, right, _ = tangent
                    i, j = np.searchsorted(x, (left, right))
                    ax.plot(x[[i, j]], g[[i, j]], "k-", lw=2, label="common tangent")
                    ax.set_title(f"1200 K coexistence candidate: {left:.3f}–{right:.3f}")
        ax.axhline(0, color="0.7", lw=0.8)
        if quality_warning:
            ax.text(
                0.5,
                0.98,
                quality_warning,
                color="crimson",
                fontweight="bold",
                ha="center",
                va="top",
                transform=ax.transAxes,
            )
        ax.set(xlabel=r"$x_{Cr}$", ylabel=r"$G_{mix}/N$ (meV/atom)")
        ax.legend(fontsize=8)
        fig.savefig(args.output / "unified_mixing_free_energy_diagnostic.png", dpi=200)
        plt.close(fig)

    rows = data["conditions"]
    temperatures = sorted({row["temperature"] for row in rows})
    compositions = sorted({row["target_cr"] for row in rows})
    lookup = {(row["temperature"], row["target_cr"]): row for row in rows}
    fields = (
        ("ess", "ESS"),
        ("std_log_weight", "std(log W)"),
        ("valid_fraction", "valid fraction"),
        ("fcc_fraction", "raw FCC fraction"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    if quality_warning:
        fig.suptitle(quality_warning, color="crimson", fontweight="bold")
    for ax, (field, title) in zip(axes.flat, fields, strict=True):
        values = np.array(
            [[lookup[(temperature, composition)][field] for composition in compositions]
             for temperature in temperatures]
        )
        image = ax.imshow(values, aspect="auto", origin="lower")
        ax.set(
            title=title,
            xlabel=r"$x_{Cr}$",
            ylabel="T (K)",
            xticks=range(len(compositions)),
            xticklabels=[f"{n / 128:g}" for n in compositions],
            yticks=range(len(temperatures)),
            yticklabels=[f"{t:g}" for t in temperatures],
        )
        fig.colorbar(image, ax=ax)
    fig.savefig(args.output / "sampler_quality_heatmaps.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
