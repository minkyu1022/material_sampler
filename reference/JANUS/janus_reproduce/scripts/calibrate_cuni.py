#!/usr/bin/env python3
"""Fit the SI Cu--Ni volume prior from short energy-only NPT runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from janus_reproduce.alloy_reference import canonical_npt_mc
from janus_reproduce.cuni import (
    KB_EV_K,
    CuNiEAM,
    build_cuni_fcc,
    fit_displacement_width,
    fit_volume_prior,
    fractional_hessian,
    quasiharmonic_width,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/cuni_prior.json"))
    parser.add_argument("--sweeps", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hessian", action="store_true")
    args = parser.parse_args()
    oracle = CuNiEAM(args.potential)
    compositions = np.linspace(0, 1, 5)
    temperatures = np.array((600.0, 900.0, 1200.0))
    rows: list[tuple[float, float, float, float]] = []
    for composition_index, composition in enumerate(compositions):
        atoms = build_cuni_fcc(
            108, cu_fraction=float(composition), seed=args.seed + composition_index
        )
        atoms.calc = oracle.calculator
        FIRE(FrechetCellFilter(atoms, hydrostatic_strain=True), logfile=None).run(
            fmax=0.02, steps=300
        )
        for temperature_index, temperature in enumerate(temperatures):
            result = canonical_npt_mc(
                atoms,
                oracle.energy,
                beta=1 / (KB_EV_K * temperature),
                sweeps=args.sweeps,
                burn_in=args.burn_in,
                thin=5,
                species=("Ni", "Cu"),
                species_moves=0,
                displacement_step=0.01 / np.sqrt(len(atoms)),
                log_volume_step=0.03 / np.sqrt(len(atoms)),
                seed=args.seed + 10 * composition_index + temperature_index,
            )
            log_volumes = np.log([sample.get_volume() for sample in result.samples])
            rows.append(
                (
                    float(composition),
                    temperature,
                    float(np.exp(log_volumes).mean() / len(atoms)),
                    float(log_volumes.std(ddof=1)),
                )
            )
    data = np.asarray(rows)
    prior = fit_volume_prior(data[:, 0], data[:, 1], data[:, 2])
    prior = replace(prior, sigma_log_volume=float(np.sqrt(np.mean(np.square(data[:, 3])))))
    displacement_fit = None
    if args.hessian:
        widths = []
        width_temperatures = []
        for composition_index, composition in enumerate(compositions):
            for temperature in temperatures:
                atoms = build_cuni_fcc(
                    108, cu_fraction=float(composition), seed=args.seed + composition_index
                )
                target_volume = 108 * prior.atomic_volume(float(composition), float(temperature))
                atoms.set_cell(
                    atoms.cell * (target_volume / atoms.get_volume()) ** (1 / 3), scale_atoms=True
                )
                atoms.calc = oracle.calculator
                FIRE(atoms, logfile=None).run(fmax=0.02, steps=300)
                widths.append(quasiharmonic_width(fractional_hessian(atoms, oracle), temperature))
                width_temperatures.append(temperature)
        displacement_fit = dict(
            zip(
                ("sigma_u_ref", "temperature_exponent"),
                fit_displacement_width(np.asarray(width_temperatures), np.asarray(widths)),
                strict=True,
            )
        )
    payload = {
        **prior.__dict__,
        "potential": str(oracle.path),
        "observations": [
            {"cu_fraction": c, "temperature_K": t, "volume_A3_per_atom": v, "sigma_log_volume": s}
            for c, t, v, s in rows
        ],
        "displacement_prior": displacement_fit,
        "sigma_v_reconstruction": {
            "recipe": "equal-state RMS of within-state log(V) standard deviations",
            "state_grid": "five compositions x three temperatures",
            "status": "reconstruction choice; exact SI recipe and value are unpublished",
        },
        "note": None if args.hessian else "rerun with --hessian for the SI displacement-prior fit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
