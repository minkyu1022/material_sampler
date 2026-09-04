#!/usr/bin/env python3
"""Calibrate provisional Ni--Cr continuous priors from the fixed potential."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from ase.calculators.eam import EAM
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from janus_reproduce.alloy_reference import canonical_npt_mc
from janus_reproduce.cuni import KB_EV_K, fit_volume_prior
from janus_reproduce.nicr import NICR_LATTICES, build_nicr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=NICR_LATTICES)
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=250)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    spec = NICR_LATTICES[args.phase]
    compositions = np.linspace(0, 1, 3 if args.smoke else 5)
    temperatures = np.array((600.0, 1050.0, 1500.0))
    calculator = EAM(potential=str(args.potential))

    def energy(atoms):
        atoms.calc = calculator
        return float(atoms.get_potential_energy())

    rows = []
    for composition_index, composition in enumerate(compositions):
        atoms = build_nicr(
            args.phase,
            round(composition * spec.n_atoms),
            lattice_constant=3.5 if args.phase == "fcc" else 2.8,
            seed=args.seed + composition_index,
        )
        atoms.calc = calculator
        FIRE(FrechetCellFilter(atoms, hydrostatic_strain=True), logfile=None).run(
            fmax=0.03 if args.smoke else 0.02, steps=100 if args.smoke else 300
        )
        for temperature_index, temperature in enumerate(temperatures):
            result = canonical_npt_mc(
                atoms,
                energy,
                beta=1 / (KB_EV_K * temperature),
                sweeps=args.sweeps,
                burn_in=args.burn_in,
                thin=max(1, (args.sweeps - args.burn_in) // 20),
                species=("Ni", "Cr"),
                species_moves=6,
                displacement_step=0.01 / np.sqrt(spec.n_atoms),
                log_volume_step=0.03 / np.sqrt(spec.n_atoms),
                seed=args.seed + 10 * composition_index + temperature_index,
            )
            log_volumes = np.log([sample.get_volume() for sample in result.samples])
            rows.append((composition, temperature, np.exp(log_volumes).mean() / spec.n_atoms,
                         log_volumes.std(ddof=1)))
    data = np.asarray(rows)
    prior = fit_volume_prior(data[:, 0], data[:, 1], data[:, 2], temperature_ref=1050.0)
    prior = replace(prior, sigma_log_volume=float(np.sqrt(np.mean(np.square(data[:, 3])))))
    payload = {
        **prior.__dict__,
        "phase": args.phase,
        "second_component": "Cr",
        "potential": str(args.potential),
        "status": "provisional smoke calibration" if args.smoke else "full calibration",
        "observations": [
            {"cr_fraction": c, "temperature_K": t, "volume_A3_atom": v, "sigma_log_volume": s}
            for c, t, v, s in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
