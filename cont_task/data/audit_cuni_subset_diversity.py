#!/usr/bin/env python3
"""Audit whether a paused Cu--Ni relaxation subset is diverse enough for training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase.io import read


CHAIN = re.compile(r"N108_T(?P<t>[0-9.]+)_mu(?P<mu>[0-9.]+)_walker(?P<w>[01])")


def latest_records(root: Path) -> list[tuple[Path, dict]]:
    output = []
    for manifest in sorted(root.glob("*/frames.jsonl")):
        records = {}
        for line in manifest.read_text().splitlines():
            try:
                record = json.loads(line)
                records[int(record["frame"])] = record
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        output.extend((manifest.parent, record) for record in records.values() if record.get("status") == "converged")
    return output


def fingerprint(task: tuple[str, int]) -> dict:
    chain_text, frame = task
    chain = Path(chain_text)
    raw = read(chain / "raw" / f"frame_{frame:04d}.extxyz")
    relaxed = read(chain / "relaxed" / f"frame_{frame:04d}.extxyz")
    frac = relaxed.get_scaled_positions(wrap=True)
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.round(delta)
    distance = np.linalg.norm(delta @ relaxed.cell.array, axis=-1)
    upper = np.triu_indices(len(frac), 1)
    z1, z2 = relaxed.numbers[upper[0]], relaxed.numbers[upper[1]]
    kinds = np.minimum(z1, z2) * 100 + np.maximum(z1, z2)
    parts = []
    for kind in np.unique(kinds):
        parts.append(np.sort(np.round(distance[upper][kinds == kind], 3)).astype(np.float32).tobytes())
    raw_frac = raw.get_scaled_positions(wrap=True)
    displacement = (frac - raw_frac + 0.5) % 1.0 - 0.5
    displacement -= displacement.mean(0, keepdims=True)
    return {
        "fingerprint": hashlib.sha256(b"".join(parts)).hexdigest(),
        "rms_relaxation_A": float(np.sqrt(np.mean(np.square(displacement @ relaxed.cell.array)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-composition", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    records = latest_records(args.input)
    source_cache, composition_records = {}, defaultdict(list)
    temperatures, chemical_potentials, walkers, chains = set(), set(), set(), set()
    for chain, record in records:
        source = record["source"]
        if source not in source_cache:
            with np.load(source) as data:
                source_cache[source] = data["species"].sum(1).astype(int)
        n_cu = int(source_cache[source][int(record["frame"])])
        composition_records[n_cu].append((chain, record))
        match = CHAIN.fullmatch(chain.name)
        if match:
            temperatures.add(float(match["t"]))
            chemical_potentials.add(float(match["mu"]))
            walkers.add(int(match["w"]))
            chains.add(chain.name)

    tasks, selected_compositions = [], []
    for n_cu, rows in sorted(composition_records.items()):
        indices = np.linspace(0, len(rows) - 1, min(len(rows), args.samples_per_composition), dtype=int)
        for index in indices:
            chain, record = rows[index]
            tasks.append((str(chain), int(record["frame"])))
            selected_compositions.append(n_cu)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fingerprints = list(pool.map(fingerprint, tasks, chunksize=8))

    sampled = defaultdict(list)
    for n_cu, row in zip(selected_compositions, fingerprints):
        sampled[n_cu].append(row)
    counts = Counter({n_cu: len(rows) for n_cu, rows in composition_records.items()})
    per_composition = {}
    for n_cu, rows in sorted(composition_records.items()):
        energies = np.asarray([record["final_energy_eV"] for _, record in rows]) / 108
        volumes = np.asarray([record["final_volume_A3"] for _, record in rows]) / 108
        sample = sampled[n_cu]
        unique = len({row["fingerprint"] for row in sample})
        rms = np.asarray([row["rms_relaxation_A"] for row in sample])
        per_composition[str(n_cu)] = {
            "count": len(rows),
            "sampled_for_fingerprint": len(sample),
            "unique_sample_fingerprints": unique,
            "sample_duplicate_fraction": 1 - unique / len(sample),
            "energy_eV_atom_std": float(energies.std()),
            "volume_A3_atom_std": float(volumes.std()),
            "rms_relaxation_A_mean": float(rms.mean()),
            "rms_relaxation_A_std": float(rms.std()),
        }
    count_values = np.asarray(list(counts.values()))
    duplicate_fractions = np.asarray([row["sample_duplicate_fraction"] for row in per_composition.values()])
    payload = {
        "records": len(records),
        "failed_or_nonconverged": 0,
        "composition_groups": len(counts),
        "missing_compositions": sorted(set(range(109)) - set(counts)),
        "composition_count_min": int(count_values.min()),
        "composition_count_median": float(np.median(count_values)),
        "composition_count_max": int(count_values.max()),
        "temperatures": sorted(temperatures),
        "chemical_potentials": sorted(chemical_potentials),
        "walkers": sorted(walkers),
        "chains_represented": len(chains),
        "fingerprint_definition": "species-pair minimum-image distance spectra rounded to 1e-3 A",
        "sample_duplicate_fraction_median": float(np.median(duplicate_fractions)),
        "sample_duplicate_fraction_max": float(duplicate_fractions.max()),
        "per_composition": per_composition,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "per_composition"}, indent=2))


if __name__ == "__main__":
    main()
