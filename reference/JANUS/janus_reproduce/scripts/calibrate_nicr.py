#!/usr/bin/env python3
"""Calibrate provisional Ni--Cr continuous priors from the fixed potential."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--cutoff-convention", default="provisional_abrupt_header")
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
    if not 0 < args.cutoff <= calculator.cutoff:
        raise ValueError(f"cutoff must be in (0, {calculator.cutoff}]")
    calculator.cutoff = args.cutoff

    def energy(atoms):
        atoms.calc = calculator
        return float(atoms.get_potential_energy())

    rows, displacement_rows = [], []
    for composition_index, composition in enumerate(compositions):
        atoms = build_nicr(
            args.phase,
            round(composition * spec.n_atoms),
            lattice_constant=3.5 if args.phase == "fcc" else 2.8,
            seed=args.seed + composition_index,
        )
        ideal_fractional = atoms.get_scaled_positions(wrap=True).copy()
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
            sample_sigmas = []
            for sample in result.samples:
                delta = (sample.get_scaled_positions(wrap=True) - ideal_fractional + 0.5) % 1.0 - 0.5
                delta -= delta.mean(axis=0, keepdims=True)
                sample_sigmas.append(float(np.sqrt(np.mean(delta**2))))
            rows.append((composition, temperature, np.exp(log_volumes).mean() / spec.n_atoms,
                         log_volumes.std(ddof=1)))
            displacement_rows.append((composition, temperature, float(np.mean(sample_sigmas))))
    data = np.asarray(rows)
    prior = fit_volume_prior(data[:, 0], data[:, 1], data[:, 2], temperature_ref=1050.0)
    prior = replace(prior, sigma_log_volume=float(np.sqrt(np.mean(np.square(data[:, 3])))))
    sigma_u_ref = float(np.mean([sigma for _, temperature, sigma in displacement_rows if temperature == 1050.0]))
    payload = {
        **prior.__dict__,
        "phase": args.phase,
        "second_component": "Cr",
        "potential": str(args.potential),
        "potential_sha256": hashlib.sha256(args.potential.read_bytes()).hexdigest(),
        "target_cutoff": args.cutoff,
        "cutoff_convention": args.cutoff_convention,
        "sigma_u_ref": sigma_u_ref,
        "sigma_u_exponent": 0.5,
        "status": "provisional smoke calibration" if args.smoke else "full calibration",
        "observations": [
            {"cr_fraction": c, "temperature_K": t, "volume_A3_atom": v, "sigma_log_volume": s}
            for c, t, v, s in rows
        ],
        "displacement_observations": [
            {"cr_fraction": c, "temperature_K": t, "sigma_u_fractional": s}
            for c, t, s in displacement_rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
