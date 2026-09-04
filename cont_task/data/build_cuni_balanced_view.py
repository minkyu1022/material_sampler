#!/usr/bin/env python3
"""Create a deterministic composition-capped training view without altering the master data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def stable_key(item: dict) -> bytes:
    return hashlib.sha256(item["mp_id"].encode()).digest()


def build(master: Path, output: Path, cap: int) -> dict:
    items = torch.load(master, map_location="cpu", weights_only=False)
    groups: dict[int, list[dict]] = defaultdict(list)
    for item in items:
        groups[int(item["n_cu"])].append(item)

    selected = []
    for n_cu in sorted(groups):
        # Stable hash sampling avoids input-order bias and spreads chain/T/mu/walker IDs.
        selected.extend(sorted(groups[n_cu], key=stable_key)[:cap])
    selected.sort(key=lambda item: item["mp_id"])

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected, output)
    source_hash = hashlib.sha256(master.read_bytes()).hexdigest()
    report = {
        "schema": "cuni-composition-capped-view-v1",
        "source": str(master.resolve()),
        "source_sha256": source_hash,
        "output": str(output.resolve()),
        "cap_per_n_cu": cap,
        "selection": "lowest SHA256(mp_id) per n_cu; deterministic and input-order independent",
        "master_items": len(items),
        "selected_items": len(selected),
        "master_counts": dict(sorted(Counter(int(x["n_cu"]) for x in items).items())),
        "selected_counts": dict(sorted(Counter(int(x["n_cu"]) for x in selected).items())),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(build(args.master, args.output, args.cap), indent=2))


if __name__ == "__main__":
    main()
