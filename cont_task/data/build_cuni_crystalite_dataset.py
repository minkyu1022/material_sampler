#!/usr/bin/env python3
"""Convert converged Cu-Ni relaxations to Crystalite-compatible token data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor


def lattice_to_y(structure) -> torch.Tensor:
    lengths = np.asarray(structure.lattice.abc)
    angles = np.asarray(structure.lattice.angles)
    return torch.tensor(
        np.r_[np.log(lengths), np.cos(np.deg2rad(angles))], dtype=torch.float32
    )


def latest_records(root: Path):
    for manifest in sorted(root.glob("*/frames.jsonl")):
        records = {}
        for line in manifest.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[int(record["frame"])] = record
        for frame, record in sorted(records.items()):
            yield manifest.parent, frame, record


def structure_fingerprint(atom_types: torch.Tensor, frac: np.ndarray, lattice: np.ndarray) -> str:
    """Permutation/translation/rotation-invariant approximate duplicate key."""
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.round(delta)
    distances = np.linalg.norm(delta @ lattice, axis=-1)
    upper = np.triu_indices(len(frac), 1)
    z1, z2 = atom_types.numpy()[upper[0]], atom_types.numpy()[upper[1]]
    pair_kind = np.minimum(z1, z2) * 100 + np.maximum(z1, z2)
    parts = [np.bincount(atom_types.numpy(), minlength=30).astype(np.int16).tobytes()]
    for kind in np.unique(pair_kind):
        parts.append(np.sort(np.round(distances[upper][pair_kind == kind], 4)).astype(np.float32).tobytes())
    payload = b"".join(
        parts
    )
    return hashlib.sha256(payload).hexdigest()


def write_crystalite_splits(items: list[dict], data_root: Path) -> dict:
    groups = {}
    for item in items:
        groups.setdefault(item["n_cu"], []).append(item)
    train, val = [], []
    for group in groups.values():
        group.sort(key=lambda item: item["mp_id"])
        if len(group) == 1:
            train.extend(group)
            continue
        for index, item in enumerate(group):
            (val if index % 20 == 0 else train).append(item)
    processed = data_root / "processed"
    raw = data_root / "raw"
    processed.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    torch.save(train, processed / "mp20_tokens_train_nmax108.pt")
    torch.save(val, processed / "mp20_tokens_val_nmax108.pt")
    (raw / "train.csv").write_text("material_id,cif\n")
    (raw / "val.csv").write_text("material_id,cif\n")
    return {"train": len(train), "val": len(val), "root": str(data_root.resolve())}


def convert(root: Path, output: Path, limit: int = 0, crystalite_root: Path | None = None) -> dict:
    items, failures, diversity = [], [], {}
    source_cache = source_fractional = source_species_all = source_sweeps = source_log_volume = None
    source_name = None
    for chain, frame, record in latest_records(root):
        if limit and len(items) >= limit:
            break
        if record.get("status") != "converged":
            failures.append(
                {
                    "chain": chain.name,
                    "frame": frame,
                    "stage": "relaxation",
                    "status": record.get("status"),
                    "error": record.get("error"),
                }
            )
            continue
        path = chain / "relaxed" / f"frame_{frame:04d}.extxyz"
        try:
            atoms = read(path)
            raw_atoms = read(chain / "raw" / f"frame_{frame:04d}.extxyz")
            if record["source"] != source_name:
                if source_cache is not None:
                    source_cache.close()
                source_cache = np.load(record["source"])
                source_fractional = source_cache["fractional_positions"]
                source_species_all = source_cache["species"]
                source_sweeps = source_cache["config_sweeps"]
                source_log_volume = source_cache["log_volume"]
                source_name = record["source"]
            source_species = source_species_all[frame]
            expected_numbers = np.where(source_species == 1, 29, 28)
            if not np.array_equal(raw_atoms.numbers, expected_numbers):
                raise RuntimeError("source-to-raw species-coordinate order mismatch")
            if not np.array_equal(atoms.numbers, expected_numbers):
                raise RuntimeError("raw-to-relaxed species-coordinate order mismatch")
            expected_frac = source_fractional[frame] % 1.0
            raw_frac = raw_atoms.get_scaled_positions(wrap=True) % 1.0
            delta = (raw_frac - expected_frac + 0.5) % 1.0 - 0.5
            if not np.allclose(delta, 0.0, atol=1e-6):
                raise RuntimeError("source-to-raw fractional coordinates mismatch")
            expected_volume = float(np.exp(source_log_volume[source_sweeps[frame]]))
            if not np.isclose(raw_atoms.get_volume(), expected_volume, rtol=1e-6):
                raise RuntimeError("source-to-raw cell volume mismatch")
            expected_cell = np.eye(3) * expected_volume ** (1.0 / 3.0)
            if not np.allclose(raw_atoms.cell.array, expected_cell, atol=1e-6, rtol=1e-6):
                raise RuntimeError("source-to-raw cubic cell matrix mismatch")
            before = atoms.get_chemical_symbols()
            structure = AseAtomsAdaptor.get_structure(atoms).get_reduced_structure()
            after = [str(site.specie) for site in structure]
            if before != after:
                raise RuntimeError("Niggli reduction changed species-coordinate order")
            atom_types = torch.tensor(
                [site.specie.Z for site in structure], dtype=torch.long
            )
            frac = np.asarray(structure.frac_coords) % 1.0
            n_cu = int((atom_types == 29).sum())
            fingerprint = structure_fingerprint(atom_types, frac, structure.lattice.matrix)
            items.append(
                {
                    "mp_id": f"{chain.name}_frame_{frame:04d}",
                    "A0": atom_types,
                    "F1": torch.tensor(
                        frac, dtype=torch.float32
                    ),
                    "Y1": lattice_to_y(structure),
                    "pad_mask": torch.zeros(len(structure), dtype=torch.bool),
                    "num_atoms": len(structure),
                    "n_cu": n_cu,
                    "initial_energy_eV": record["initial_energy_eV"],
                    "final_energy_eV": record["final_energy_eV"],
                    "initial_volume_A3": record["initial_volume_A3"],
                    "final_volume_A3": record["final_volume_A3"],
                    "structure_fingerprint": fingerprint,
                    "source": str(path.resolve()),
                }
            )
            group = diversity.setdefault(n_cu, {"fingerprints": set(), "energies": [], "volumes": []})
            group["fingerprints"].add(fingerprint)
            group["energies"].append(float(record["final_energy_eV"]))
            group["volumes"].append(float(record["final_volume_A3"]))
            if len(items) % 1_000 == 0:
                output.with_suffix(".progress.json").write_text(
                    json.dumps({"processed": len(items), "failures": len(failures)}) + "\n"
                )
        except Exception as exc:
            failures.append(
                {"chain": chain.name, "frame": frame, "stage": "preprocessing", "error": f"{type(exc).__name__}: {exc}"}
            )

    if source_cache is not None:
        source_cache.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, output)
    counts = Counter(item["n_cu"] for item in items)
    diversity_stats = {}
    for n_cu, values in sorted(diversity.items()):
        energies, volumes = np.asarray(values["energies"]), np.asarray(values["volumes"])
        diversity_stats[str(n_cu)] = {
            "count": len(energies),
            "unique_fingerprints": len(values["fingerprints"]),
            "approx_duplicate_count": len(energies) - len(values["fingerprints"]),
            "final_energy_eV_mean": float(energies.mean()),
            "final_energy_eV_std": float(energies.std()),
            "final_volume_A3_mean": float(volumes.mean()),
            "final_volume_A3_std": float(volumes.std()),
        }
    split = write_crystalite_splits(items, crystalite_root) if crystalite_root else None
    report = {
        "schema": "crystalite-mp20-token-compatible-v1",
        "items": len(items),
        "failures": len(failures),
        "species_sorted": False,
        "species_coordinate_order_preserved": not failures,
        "niggli_reduced": True,
        "primitive_reduced": False,
        "composition_counts": {str(key): value for key, value in sorted(counts.items())},
        "diversity_by_n_cu": diversity_stats,
        "duplicate_definition": "SHA256(composition and species-pair minimum-image distance spectra rounded 1e-4 A)",
        "crystalite_split": split,
        "output": str(output.resolve()),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crystalite-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output, args.limit, args.crystalite_root), indent=2))


if __name__ == "__main__":
    main()
