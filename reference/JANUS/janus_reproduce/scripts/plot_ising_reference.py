#!/usr/bin/env python3
"""Plot the two Ising Fig. 2a reference observables from saved Wolff metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("wolff_fig2a_reference.png"))
    args = parser.parse_args()
    rows = json.loads(args.metrics.read_text())["grid"]
    temperatures = sorted({row["temperature"] for row in rows})
    fields = sorted({row["delta_mu"] for row in rows})
    spin_up = np.array(
        [
            [
                next(
                    row["wolff"]["up_fraction"]
                    for row in rows
                    if row["temperature"] == temperature and row["delta_mu"] == field
                )
                for field in fields
            ]
            for temperature in temperatures
        ]
    )
    zero = sorted((row for row in rows if row["delta_mu"] == 0), key=lambda row: row["temperature"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    image = axes[0].imshow(
        spin_up,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="coolwarm",
    )
    tick_fields = (fields[0], -0.02, 0.0, 0.02, fields[-1])
    axes[0].set_xticks(
        [
            min(range(len(fields)), key=lambda index: abs(fields[index] - value))
            for value in tick_fields
        ],
        [f"{value:g}" for value in tick_fields],
    )
    axes[0].set_yticks(
        np.linspace(0, len(temperatures) - 1, 4),
        [f"{value:.1f}" for value in np.linspace(temperatures[0], temperatures[-1], 4)],
    )
    axes[0].set(xlabel=r"$\Delta\mu/J$", ylabel=r"$k_B T/J$", title="Ghost-Wolff spin-up fraction")
    fig.colorbar(image, ax=axes[0], label=r"$\langle x_\uparrow\rangle$")

    axes[1].plot(
        [row["temperature"] for row in zero],
        [row["wolff"]["abs_magnetization"] for row in zero],
        "o-",
        label="Ghost-Wolff",
    )
    axes[1].axvline(2.2692, color="black", linestyle="--", linewidth=1, label=r"$T_c$")
    axes[1].set(
        xlabel=r"$k_B T/J$",
        ylabel=r"$\langle |m|\rangle$",
        ylim=(0, 1.03),
        title=r"Zero field ($\Delta\mu=0$)",
    )
    axes[1].legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
