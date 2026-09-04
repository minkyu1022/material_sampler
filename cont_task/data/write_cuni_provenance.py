#!/usr/bin/env python3
"""Write reproducible input/code/environment provenance for Cu-Ni Phase 1."""
from __future__ import annotations
import argparse, hashlib, json, os, platform
from datetime import datetime, timezone
from pathlib import Path
import ase, numpy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.source.glob("*.npz"))
    code = [
        "cont_task/data/relax_cuni.py",
        "cont_task/data/relax_cuni_dataset.py",
        "cont_task/data/build_cuni_crystalite_dataset.py",
        "cont_task/data/validate_cuni_phase1.py",
        "cont_task/data/cuni_progress.py",
        "cont_task/data/monitor_cuni_relax.sh",
        "cont_task/data/write_cuni_provenance.py",
        "cont_task/data/PHASE1_RUNBOOK.md",
    ]
    locks = ["reference/JANUS/janus_reproduce/uv.lock", "reference/crystalite/uv.lock"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"directory": str(args.source.resolve()), "files": len(files),
                   "bytes": sum(p.stat().st_size for p in files),
                   "sha256_by_file": {p.name: sha256(p) for p in files}},
        "relaxation": json.loads((args.output / "run_manifest.json").read_text()),
        "command": (args.output / "command.txt").read_text().strip(),
        "code_sha256": {name: sha256(args.root / name) for name in code},
        "locks": {name: sha256(args.root / name) for name in locks},
        "environment": {"python": platform.python_version(), "ase": ase.__version__,
                        "numpy": numpy.__version__, "platform": platform.platform(),
                        "cpu_count": os.cpu_count()},
        "forbidden_activation_checkpointing_used": False,
        "eight_ddp_main_training_started": False,
        "invalid_preproduction_artifacts": [
            "cont_task/data/pilot/N108_T1200_mu089_w0_f0 (old inverted species mapping)",
            "cont_task/data/benchmark_8x1 (old inverted species mapping)",
            "cont_task/data/benchmark_8x1_singlethread_v2 (old inverted species mapping)",
        ],
    }
    (args.output / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
