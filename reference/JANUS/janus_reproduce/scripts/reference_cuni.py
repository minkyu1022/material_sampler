#!/usr/bin/env python3
"""Run resumable shards of the published Cu--Ni semi-grand NPT reference."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from janus_reproduce.alloy_reference import semi_grand_npt_mc
from janus_reproduce.cuni import (
    KB_EV_K,
    CuNiEAM,
    build_cuni_fcc,
    delta_mu_108,
    temperatures_108,
    temperatures_256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/cuni_reference"))
    parser.add_argument("--n-atoms", type=int, choices=(108, 256), default=108)
    parser.add_argument("--delta-mu", type=float, nargs="+")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sweeps", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def save_result(path: Path, result, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = result.samples
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            positions=np.stack([atoms.positions for atoms in samples]),
            cells=np.stack([atoms.cell.array for atoms in samples]),
            species=np.stack(
                [[symbol == "Cu" for symbol in atoms.get_chemical_symbols()] for atoms in samples]
            ).astype(np.uint8),
            energies=result.energies,
            metadata=np.asarray(json.dumps(metadata)),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be smaller than num-shards")
    oracle = CuNiEAM(args.potential)
    temperatures = temperatures_108() if args.n_atoms == 108 else temperatures_256()
    if args.n_atoms == 256 and args.delta_mu is None:
        raise SystemExit(
            "the paper omits the 11 N=256 chemical potentials; pass --delta-mu explicitly"
        )
    delta_mu = np.asarray(args.delta_mu) if args.delta_mu else delta_mu_108()
    total_sweeps = args.sweeps or (40_000 if args.n_atoms == 108 else 24_000)
    burn_in = (
        args.burn_in if args.burn_in is not None else (2_000 if args.n_atoms == 108 else 12_000)
    )
    if burn_in >= total_sweeps:
        raise SystemExit("burn-in must be smaller than total sweeps")
    production_sweeps = total_sweeps - burn_in
    states = [
        (temperature, mu, walker)
        for temperature in temperatures
        for mu in delta_mu
        for walker in range(2)
    ]
    for task_index, (temperature, mu, walker) in enumerate(states):
        if task_index % args.num_shards != args.shard_index:
            continue
        path = args.output / (
            f"N{args.n_atoms}_T{temperature:07.2f}_mu{mu:07.4f}_walker{walker}.npz"
        )
        if path.exists():
            continue
        # Opposing pure phases are the SI convergence certificate.
        atoms = build_cuni_fcc(args.n_atoms, cu_fraction=float(walker), seed=args.seed + task_index)
        result = semi_grand_npt_mc(
            atoms,
            oracle.energy,
            beta=1 / (KB_EV_K * temperature),
            sweeps=production_sweeps,
            burn_in=burn_in,
            thin=args.thin,
            species=("Ni", "Cu"),
            chemical_potentials={"Cu": float(mu)},
            species_moves=6,
            displacement_step=0.01 / np.sqrt(args.n_atoms) * np.sqrt(temperature / 1_200),
            log_volume_step=0.03 / np.sqrt(args.n_atoms) * np.sqrt(temperature / 1_200),
            seed=args.seed + task_index,
        )
        save_result(
            path,
            result,
            {
                "temperature_K": float(temperature),
                "delta_mu_Cu_minus_Ni_eV": float(mu),
                "walker": walker,
                "n_atoms": args.n_atoms,
                "total_sweeps": total_sweeps,
                "production_sweeps": production_sweeps,
                "burn_in": burn_in,
                "thin": args.thin,
                "final_displacement_step": result.displacement_step,
                "final_log_volume_step": result.log_volume_step,
                "potential": str(oracle.path),
                "acceptance": {key: value.acceptance_rate for key, value in result.stats.items()},
            },
        )


if __name__ == "__main__":
    main()
