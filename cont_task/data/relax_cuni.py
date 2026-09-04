#!/usr/bin/env python3
"""Relax Cu-Ni reference-MC snapshots with the reference EAM."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import BFGS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--maxstep", type=float, default=0.1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as data:
        frame = args.frame % len(data["fractional_positions"])
        fractional = data["fractional_positions"][frame].astype(float)
        species_index = data["species"][frame]
        volume = float(np.exp(data["log_volume"][data["config_sweeps"][frame]]))
        source_metadata = json.loads(str(data["metadata"]))

    # Reference-MC storage uses 1=Cu, 0=Ni (not the torch-EAM channel order).
    symbols = np.where(species_index == 1, "Cu", "Ni").tolist()
    cell_length = volume ** (1.0 / 3.0)
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=fractional,
        cell=np.eye(3) * cell_length,
        pbc=True,
    )
    initial_numbers = atoms.numbers.copy()
    write(args.output / "raw.extxyz", atoms)
    write(args.output / "raw.cif", atoms)

    atoms.calc = EAM(potential=str(args.potential.resolve()))
    initial_energy = float(atoms.get_potential_energy())
    optimizer = BFGS(
        FrechetCellFilter(atoms, scalar_pressure=0.0),
        trajectory=str(args.output / "relax.traj"),
        logfile=str(args.output / "relax.log"),
        maxstep=args.maxstep,
    )
    converged = bool(optimizer.run(fmax=args.fmax, steps=args.max_steps))
    final_energy = float(atoms.get_potential_energy())
    max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    max_stress = float(np.abs(atoms.get_stress(voigt=True)).max())
    order_preserved = bool(np.array_equal(initial_numbers, atoms.numbers))

    write(args.output / "relaxed.extxyz", atoms)
    write(args.output / "relaxed.cif", atoms)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "potential": str(args.potential.resolve()),
        "potential_sha256": sha256(args.potential),
        "source_metadata": source_metadata,
        "frame": frame,
        "n_atoms": len(atoms),
        "n_cu": symbols.count("Cu"),
        "optimizer": "ase.optimize.BFGS",
        "cell_filter": "ase.filters.FrechetCellFilter",
        "full_cell_dof": True,
        "scalar_pressure_eV_A3": 0.0,
        "fmax_eV_A": args.fmax,
        "max_steps": args.max_steps,
        "maxstep": args.maxstep,
        "optimizer_steps": optimizer.nsteps,
        "converged": converged,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "final_max_force_eV_A": max_force,
        "final_max_abs_stress_eV_A3": max_stress,
        "initial_volume_A3": volume,
        "final_volume_A3": float(atoms.get_volume()),
        "species_coordinate_order_preserved": order_preserved,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not order_preserved:
        raise RuntimeError("species order changed during relaxation")


if __name__ == "__main__":
    main()
