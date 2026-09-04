#!/usr/bin/env python3
"""Compare Ni--Cr pure-phase anchors under candidate oracle cutoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from scipy.optimize import minimize_scalar

from janus_reproduce.torch_eam import TorchEAM

PUBLISHED = {"fcc": -0.909, "bcc": -1.086}


def relaxed_energy(model: TorchEAM, element: str, phase: str) -> tuple[float, float]:
    species_index = model.elements.index(element)

    def energy(lattice_constant: float) -> float:
        atoms = bulk(element, phase, a=lattice_constant, cubic=True)
        species = torch.full((len(atoms),), species_index)
        fractional = torch.as_tensor(atoms.get_scaled_positions(), dtype=torch.float64)
        log_volume = torch.tensor(np.log(atoms.get_volume()), dtype=torch.float64)
        return model(species, fractional, log_volume).item() / len(atoms)

    bounds = (3.0, 4.1) if phase == "fcc" else (2.4, 3.4)
    result = minimize_scalar(energy, bounds=bounds, method="bounded", options={"xatol": 1e-9})
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x), float(result.fun)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("potential", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"potential": str(args.potential), "published_anchors_eV": PUBLISHED, "cutoffs": {}}
    for cutoff in (6.0, 5.0, 5.3):
        model = TorchEAM(args.potential, cutoff=cutoff)
        phases = {}
        for phase in ("fcc", "bcc"):
            ni_a, ni_energy = relaxed_energy(model, "Ni", phase)
            cr_a, cr_energy = relaxed_energy(model, "Cr", phase)
            difference = cr_energy - ni_energy
            phases[phase] = {
                "ni_lattice_A": ni_a,
                "cr_lattice_A": cr_a,
                "ni_energy_eV_atom": ni_energy,
                "cr_energy_eV_atom": cr_energy,
                "cr_minus_ni_eV_atom": difference,
                "anchor_abs_error_eV_atom": abs(difference - PUBLISHED[phase]),
            }
        report["cutoffs"][str(cutoff)] = phases
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
