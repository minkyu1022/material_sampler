#!/usr/bin/env python3
"""Resumable parallel relaxation of all Cu-Ni reference-MC frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import BFGS

_CALCULATOR = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_worker(potential: str) -> None:
    global _CALCULATOR
    _CALCULATOR = EAM(potential=potential)


def completed_frames(manifest: Path) -> set[int]:
    if not manifest.exists():
        return set()
    complete = set()
    lines = manifest.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                valid_text = "\n".join(lines[:index])
                manifest.write_text(valid_text + ("\n" if valid_text else ""))
                continue
            raise
        frame = int(record["frame"])
        required = (
            manifest.parent / "raw" / f"frame_{frame:04d}.extxyz",
            manifest.parent / "relaxed" / f"frame_{frame:04d}.extxyz",
            manifest.parent / "cif" / f"frame_{frame:04d}.cif",
        )
        if record.get("status") == "converged" and all(
            path.is_file() and path.stat().st_size > 0 for path in required
        ):
            complete.add(frame)
    return complete


def species_to_symbols(species: np.ndarray) -> list[str]:
    """Decode the reference producer's canonical 1=Cu, 0=Ni encoding."""
    if not np.isin(species, (0, 1)).all():
        raise ValueError("species must use the canonical binary 1=Cu, 0=Ni encoding")
    return np.where(species == 1, "Cu", "Ni").tolist()


def write_or_validate_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = {key: value for key, value in manifest.items() if key != "created_utc"}
        previous = {key: value for key, value in existing.items() if key != "created_utc"}
        if previous != comparable:
            raise RuntimeError(f"refusing incompatible resume in {path.parent}")
        return
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def relax_chain(task: tuple[str, str, int, float, int, float]) -> dict:
    input_name, output_name, frame_limit, fmax, max_steps, maxstep = task
    input_path, output = Path(input_name), Path(output_name)
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    (output / "relaxed").mkdir(exist_ok=True)
    (output / "cif").mkdir(exist_ok=True)
    manifest_path = output / "frames.jsonl"
    done = completed_frames(manifest_path)

    with np.load(input_path) as data:
        fractional_all = data["fractional_positions"]
        species_all = data["species"]
        sweeps = data["config_sweeps"]
        log_volume = data["log_volume"]
        source_metadata = json.loads(str(data["metadata"]))
        count = len(fractional_all) if frame_limit <= 0 else min(frame_limit, len(fractional_all))

        completed = failed = skipped = 0
        started = time.monotonic()
        with manifest_path.open("a", buffering=1) as manifest:
            for frame in range(count):
                if frame in done:
                    skipped += 1
                    continue
                frame_started = time.monotonic()
                record = {
                    "frame": frame,
                    "sweep": int(sweeps[frame]),
                    "source": str(input_path.resolve()),
                }
                try:
                    species_index = species_all[frame]
                    symbols = species_to_symbols(species_index)
                    volume = float(np.exp(log_volume[sweeps[frame]]))
                    atoms = Atoms(
                        symbols=symbols,
                        scaled_positions=fractional_all[frame].astype(float),
                        cell=np.eye(3) * volume ** (1.0 / 3.0),
                        pbc=True,
                    )
                    initial_numbers = atoms.numbers.copy()
                    raw_path = output / "raw" / f"frame_{frame:04d}.extxyz"
                    relaxed_path = output / "relaxed" / f"frame_{frame:04d}.extxyz"
                    cif_path = output / "cif" / f"frame_{frame:04d}.cif"
                    write(raw_path, atoms)

                    atoms.calc = _CALCULATOR
                    initial_energy = float(atoms.get_potential_energy())
                    optimizer = BFGS(
                        FrechetCellFilter(atoms, scalar_pressure=0.0),
                        logfile=None,
                        maxstep=maxstep,
                    )
                    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
                    final_energy = float(atoms.get_potential_energy())
                    order_preserved = bool(np.array_equal(initial_numbers, atoms.numbers))
                    if not order_preserved:
                        raise RuntimeError("species order changed during relaxation")
                    write(relaxed_path, atoms)
                    write(cif_path, atoms)
                    record.update(
                        status="converged" if converged else "not_converged",
                        n_atoms=len(atoms),
                        n_cu=symbols.count("Cu"),
                        initial_energy_eV=initial_energy,
                        final_energy_eV=final_energy,
                        initial_volume_A3=volume,
                        final_volume_A3=float(atoms.get_volume()),
                        final_max_force_eV_A=float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                        final_max_abs_stress_eV_A3=float(np.abs(atoms.get_stress(voigt=True)).max()),
                        optimizer_steps=optimizer.nsteps,
                        species_coordinate_order_preserved=True,
                    )
                    completed += int(converged)
                    failed += int(not converged)
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    failed += 1
                record["elapsed_seconds"] = time.monotonic() - frame_started
                manifest.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "chain": input_path.name,
        "source_metadata": source_metadata,
        "requested_frames": count,
        "converged": completed,
        "failed_or_not_converged": failed,
        "skipped_existing": skipped,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--potential", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--chain-limit", type=int, default=0)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--maxstep", type=float, default=0.1)
    args = parser.parse_args()

    inputs = sorted(args.input_dir.glob("*.npz"))
    if args.chain_limit > 0:
        inputs = inputs[: args.chain_limit]
    if not inputs:
        raise FileNotFoundError(f"no .npz files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "potential": str(args.potential.resolve()),
        "potential_sha256": file_sha256(args.potential),
        "optimizer": "ase.optimize.BFGS",
        "cell_filter": "ase.filters.FrechetCellFilter",
        "full_cell_dof": True,
        "scalar_pressure_eV_A3": 0.0,
        "workers": args.workers,
        "chains": len(inputs),
        "frame_limit": args.frame_limit,
        "fmax_eV_A": args.fmax,
        "max_steps": args.max_steps,
        "maxstep": args.maxstep,
    }
    write_or_validate_manifest(args.output_dir / "run_manifest.json", run_manifest)
    (args.output_dir / "runtime_pid").write_text(f"{os.getpid()}\n")

    tasks = [
        (
            str(path),
            str(args.output_dir / path.stem),
            args.frame_limit,
            args.fmax,
            args.max_steps,
            args.maxstep,
        )
        for path in inputs
    ]
    progress = args.output_dir / "progress.jsonl"
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(str(args.potential.resolve()),),
    ) as pool, progress.open("a", buffering=1) as log:
        futures = [pool.submit(relax_chain, task) for task in tasks]
        for future in as_completed(futures):
            summary = future.result()
            summary["reported_utc"] = datetime.now(timezone.utc).isoformat()
            log.write(json.dumps(summary, sort_keys=True) + "\n")
            print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
