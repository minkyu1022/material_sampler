#!/usr/bin/env python3
"""Create a deterministic composition-stratified Cu-Ni overfit subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.size <= 0 or not 0.0 < args.val_fraction < 1.0:
        parser.error("--size must be positive and --val-fraction must be in (0,1)")

    items = torch.load(args.input, weights_only=False)
    groups: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        groups[int(item["n_cu"])].append(item)
    rng = random.Random(args.seed)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    while len(selected) < min(args.size, len(items)):
        added = False
        for n_cu in sorted(groups):
            if groups[n_cu]:
                selected.append(groups[n_cu].pop())
                added = True
                if len(selected) == min(args.size, len(items)):
                    break
        if not added:
            break

    by_composition: dict[int, list[dict]] = defaultdict(list)
    for item in selected:
        by_composition[int(item["n_cu"])].append(item)
    train, val = [], []
    for n_cu in sorted(by_composition):
        group = by_composition[n_cu]
        rng.shuffle(group)
        n_val = min(len(group) - 1, max(1, round(len(group) * args.val_fraction))) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)

    processed = args.output_root / "processed"
    raw = args.output_root / "raw"
    processed.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    train_path = processed / "mp20_tokens_train_nmax108.pt"
    val_path = processed / "mp20_tokens_val_nmax108.pt"
    torch.save(train, train_path)
    torch.save(val, val_path)
    (raw / "train.csv").write_text("material_id,cif\n")
    (raw / "val.csv").write_text("material_id,cif\n")

    selected_counts = Counter(int(item["n_cu"]) for item in selected)
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "seed": args.seed,
        "selection": "balanced round-robin over available N_Cu groups",
        "requested_size": args.size,
        "selected": len(selected),
        "train": len(train),
        "val": len(val),
        "available_compositions": sorted(int(key) for key in groups),
        "composition_counts": {str(key): value for key, value in sorted(selected_counts.items())},
        "train_sha256": sha256(train_path),
        "val_sha256": sha256(val_path),
    }
    (args.output_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
