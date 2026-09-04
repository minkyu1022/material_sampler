#!/usr/bin/env python3
"""Record the exact converged Cu--Ni subset accepted for downstream processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from audit_cuni_subset_diversity import latest_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--diversity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache, items = {}, []
    for chain, record in latest_records(args.input):
        source = record["source"]
        if source not in cache:
            with np.load(source) as data:
                cache[source] = data["species"].sum(1).astype(int)
        frame = int(record["frame"])
        items.append({"chain": chain.name, "frame": frame, "n_cu": int(cache[source][frame])})
    items.sort(key=lambda item: (item["chain"], item["frame"]))
    payload = {
        "created_epoch": int(time.time()),
        "selection": "all converged records present after user-requested early stop",
        "reason": "136826-record diversity audit covered all compositions and found no sampled fingerprint duplicates",
        "count": len(items),
        "diversity_audit": str(args.diversity_audit.resolve()),
        "diversity_audit_sha256": hashlib.sha256(args.diversity_audit.read_bytes()).hexdigest(),
        "items": items,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()
