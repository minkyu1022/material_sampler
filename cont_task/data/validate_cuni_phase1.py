#!/usr/bin/env python3
"""Final completeness gate for the Cu-Ni relaxation dataset."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from build_cuni_crystalite_dataset import lattice_to_y


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(item: dict) -> str:
    digest = hashlib.sha256()
    for key in ("A0", "F1", "Y1", "pad_mask"):
        digest.update(item[key].contiguous().numpy().tobytes())
    digest.update(int(item["num_atoms"]).to_bytes(4, "little"))
    return digest.hexdigest()


def validate_item_source(task) -> str | None:
    relaxed, item = task
    try:
        chain, frame_text = item["mp_id"].rsplit("_frame_", 1)
        expected_source = (Path(relaxed) / chain / "relaxed" / f"frame_{int(frame_text):04d}.extxyz").resolve()
        if Path(item["source"]).resolve() != expected_source:
            raise AssertionError("source path does not match mp_id")
        structure = AseAtomsAdaptor.get_structure(read(expected_source)).get_reduced_structure()
        expected_a0 = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
        expected_f1 = torch.tensor(np.asarray(structure.frac_coords) % 1.0, dtype=torch.float32)
        expected_y1 = lattice_to_y(structure)
        if not torch.equal(item["A0"], expected_a0):
            raise AssertionError("A0 differs from relaxed structure")
        if not torch.allclose(item["F1"], expected_f1, atol=1e-6, rtol=0):
            raise AssertionError("F1 differs from relaxed structure")
        if not torch.allclose(item["Y1"], expected_y1, atol=1e-6, rtol=0):
            raise AssertionError("Y1 differs from relaxed structure")
        return None
    except Exception as exc:
        return f"{item.get('mp_id')}: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--relaxed", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    expected_keys = set()
    expected_compositions = Counter()
    for path in sorted(args.source.glob("*.npz")):
        with np.load(path) as data:
            expected_keys.update((path.stem, frame) for frame in range(len(data["fractional_positions"])))
            expected_compositions.update(data["species"].sum(axis=1).astype(int).tolist())

    records = {}
    malformed, duplicate_records = [], []
    for path in args.relaxed.glob("*/frames.jsonl"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(f"{path}:{number}")
                continue
            key = (path.parent.name, int(record["frame"]))
            if key in records:
                duplicate_records.append(f"{path}:{number}:{key}")
            records[key] = record
    converged = {key: value for key, value in records.items() if value.get("status") == "converged"}
    bad = {key: value for key, value in records.items() if value.get("status") != "converged"}

    artifacts, missing_or_empty = {"raw": [], "relaxed": [], "cif": []}, []
    for chain, frame in converged:
        root = args.relaxed / chain
        paths = {
            "raw": root / "raw" / f"frame_{frame:04d}.extxyz",
            "relaxed": root / "relaxed" / f"frame_{frame:04d}.extxyz",
            "cif": root / "cif" / f"frame_{frame:04d}.cif",
        }
        for kind, path in paths.items():
            artifacts[kind].append(path)
            if not path.is_file() or path.stat().st_size == 0:
                missing_or_empty.append(str(path))

    items = torch.load(args.processed, weights_only=False)
    item_ids, schema_errors = set(), []
    for index, item in enumerate(items):
        try:
            item_ids.add(item["mp_id"])
            assert item["A0"].shape == (108,) and item["A0"].dtype == torch.long
            assert item["F1"].shape == (108, 3) and torch.isfinite(item["F1"]).all()
            assert item["Y1"].shape == (6,) and torch.isfinite(item["Y1"]).all()
            assert item["pad_mask"].shape == (108,) and not item["pad_mask"].any()
            assert item["num_atoms"] == 108
            assert set(item["A0"].tolist()) <= {28, 29}
            assert item["n_cu"] == int((item["A0"] == 29).sum())
        except Exception as exc:
            schema_errors.append(f"item {index}: {type(exc).__name__}: {exc}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        source_errors = [
            error for error in pool.map(
                validate_item_source,
                ((str(args.relaxed), item) for item in items),
                chunksize=32,
            ) if error is not None
        ]
    expected_item_ids = {f"{chain}_frame_{frame:04d}" for chain, frame in converged}

    processed_manifest_path = args.processed.with_suffix(".manifest.json")
    failures_path = args.processed.with_suffix(".failures.json")
    processed = json.loads(processed_manifest_path.read_text())
    failures = json.loads(failures_path.read_text())
    split = processed.get("crystalite_split") or {}
    split_root = Path(split.get("root", "/missing"))
    train_path = split_root / "processed/mp20_tokens_train_nmax108.pt"
    val_path = split_root / "processed/mp20_tokens_val_nmax108.pt"
    train = torch.load(train_path, weights_only=False) if train_path.exists() else []
    val = torch.load(val_path, weights_only=False) if val_path.exists() else []
    split_ids = {item["mp_id"] for item in train + val}
    train_compositions = {item["n_cu"] for item in train}
    all_compositions = {item["n_cu"] for item in items}
    main_digests = {item["mp_id"]: token_digest(item) for item in items}
    split_payload_matches = all(main_digests.get(item["mp_id"]) == token_digest(item) for item in train + val)

    sample_cifs = artifacts["cif"][:: max(1, len(artifacts["cif"]) // 100)] if artifacts["cif"] else []
    unreadable_cifs = []
    for path in sample_cifs[:100]:
        try:
            cif_atoms = read(path)
            relaxed_atoms = read(path.parent.parent / "relaxed" / f"{path.stem}.extxyz")
            if not np.array_equal(cif_atoms.numbers, relaxed_atoms.numbers):
                raise AssertionError("CIF species order differs from relaxed extxyz")
            if not np.allclose(cif_atoms.cell.cellpar(), relaxed_atoms.cell.cellpar(), atol=1e-6, rtol=1e-7):
                raise AssertionError("CIF lattice parameters differ from relaxed extxyz")
            delta = (cif_atoms.get_scaled_positions(wrap=True) - relaxed_atoms.get_scaled_positions(wrap=True) + 0.5) % 1.0 - 0.5
            if np.max(np.abs(delta)) >= 1e-6:
                raise AssertionError("CIF fractional-coordinate round-trip error exceeds 1e-6")
        except Exception as exc:
            unreadable_cifs.append(f"{path}: {exc}")

    run = json.loads((args.relaxed / "run_manifest.json").read_text())
    provenance_path = args.relaxed / "provenance_manifest.json"
    provenance = json.loads(provenance_path.read_text())
    repo_root = Path(__file__).resolve().parents[2]
    provenance_hashes_match = all(
        (repo_root / name).is_file() and sha256(repo_root / name) == digest
        for section in ("code_sha256", "locks")
        for name, digest in provenance.get(section, {}).items()
    )
    source_hashes = provenance.get("source", {}).get("sha256_by_file", {})
    source_hashes_match = len(source_hashes) == len(list(args.source.glob("*.npz"))) and all(
        (args.source / name).is_file() and sha256(args.source / name) == digest
        for name, digest in source_hashes.items()
    )
    required_record_fields = {
        "initial_energy_eV", "final_energy_eV", "initial_volume_A3", "final_volume_A3",
        "final_max_force_eV_A", "final_max_abs_stress_eV_A3", "optimizer_steps",
        "species_coordinate_order_preserved",
    }
    record_values_valid = all(
        required_record_fields <= set(record)
        and all(np.isfinite(record[key]) for key in required_record_fields - {"species_coordinate_order_preserved"})
        and record["optimizer_steps"] <= run["max_steps"]
        and record["final_energy_eV"] <= record["initial_energy_eV"] + 1e-8
        and record["final_max_force_eV_A"] <= run["fmax_eV_A"] * 1.05
        and record["species_coordinate_order_preserved"] is True
        for record in converged.values()
    )
    checks = {
        "exact_source_frame_identity": set(records) == expected_keys,
        "no_malformed_records": not malformed,
        "no_duplicate_records": not duplicate_records,
        "all_relaxations_converged": set(converged) == expected_keys and not bad,
        "relaxation_record_values_valid": record_values_valid,
        "all_structure_artifacts_nonempty": not missing_or_empty,
        "sampled_cifs_readable": not unreadable_cifs,
        "processed_ids_exact": item_ids == expected_item_ids and len(item_ids) == len(items),
        "processed_schema_valid": not schema_errors,
        "processed_payloads_match_relaxed": not source_errors,
        "processed_count_matches_converged": processed["items"] == len(converged) == len(items),
        "failure_list_exists_and_empty": failures_path.exists() and not failures,
        "statistics_complete": sum(processed["composition_counts"].values()) == len(items)
        and sum(v["count"] for v in processed["diversity_by_n_cu"].values()) == len(items),
        "composition_counts_match_source": processed["composition_counts"]
        == {str(key): value for key, value in sorted(expected_compositions.items())},
        "crystalite_train_val_discoverable": train_path.exists() and val_path.exists()
        and (split_root / "raw/train.csv").exists() and (split_root / "raw/val.csv").exists(),
        "crystalite_split_exact": split_ids == item_ids and len(train) + len(val) == len(items),
        "crystalite_split_payloads_exact": split_payload_matches,
        "every_composition_present_in_train": train_compositions == all_compositions,
        "species_coordinate_order_preserved": processed["species_coordinate_order_preserved"],
        "potential_hash_matches": run["potential_sha256"] == sha256(Path(run["potential"])),
        "optimizer_is_bfgs_frechet": run["optimizer"] == "ase.optimize.BFGS"
        and run["cell_filter"] == "ase.filters.FrechetCellFilter" and run["full_cell_dof"] is True,
        "provenance_present": provenance_path.exists() and bool(provenance.get("locks")) and bool(provenance.get("source")),
        "provenance_hashes_match": provenance_hashes_match,
        "source_hashes_match": source_hashes_match,
        "recorded_command_matches": provenance.get("command") == (args.relaxed / "command.txt").read_text().strip(),
        "provenance_relaxation_matches_run": provenance.get("relaxation") == run,
        "no_activation_checkpointing_or_early_8ddp_main_training": provenance.get("forbidden_activation_checkpointing_used") is False
        and provenance.get("eight_ddp_main_training_started") is False,
    }
    report = {
        "expected_structures": len(expected_keys), "recorded": len(records),
        "converged": len(converged), "failed_or_not_converged": len(bad),
        "malformed_records": malformed, "duplicate_records": duplicate_records,
        "missing_or_empty_artifacts": missing_or_empty,
        "unreadable_sampled_cifs": unreadable_cifs, "processed_schema_errors": schema_errors,
        "processed_source_errors": source_errors,
        "checks": checks, "passed": all(checks.values()),
    }
    destination = args.relaxed / "validation_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    markdown = [
        "# Cu–Ni Phase 1 validation", "",
        f"**Result:** {'PASS' if report['passed'] else 'FAIL'}", "",
        f"- Expected / recorded / converged: {len(expected_keys)} / {len(records)} / {len(converged)}",
        f"- Failed or not converged: {len(bad)}", f"- Processed items: {len(items)}", "",
        "## Completion checks", "", "| Check | Result |", "|---|---|",
    ]
    markdown.extend(f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    markdown.extend(["", "## Composition and structural-diversity statistics", "",
                     "| N(Cu) | Samples | Unique approximate fingerprints | Approx. duplicates | Energy std (eV) | Volume std (Å³) |",
                     "|---:|---:|---:|---:|---:|---:|"])
    for n_cu, values in processed["diversity_by_n_cu"].items():
        markdown.append(
            f"| {n_cu} | {values['count']} | {values['unique_fingerprints']} | "
            f"{values['approx_duplicate_count']} | {values['final_energy_eV_std']:.6g} | "
            f"{values['final_volume_A3_std']:.6g} |"
        )
    markdown.extend(["", f"Duplicate metric: {processed['duplicate_definition']}", "",
                     f"Failure list: `{failures_path}`", f"Provenance: `{provenance_path}`", ""])
    markdown_path = args.relaxed / "PHASE1_VALIDATION.md"
    markdown_path.write_text("\n".join(markdown))
    provenance["final_artifacts"] = {
        kind: {"count": len(paths), "bytes": sum(path.stat().st_size for path in paths if path.exists())}
        for kind, paths in artifacts.items()
    }
    provenance["final_artifacts"].update({
        "processed_pt": {"path": str(args.processed.resolve()), "bytes": args.processed.stat().st_size, "sha256": sha256(args.processed)},
        "processed_manifest_sha256": sha256(processed_manifest_path),
        "failure_list_sha256": sha256(failures_path),
        "crystalite_train_pt": {"path": str(train_path), "bytes": train_path.stat().st_size, "sha256": sha256(train_path)},
        "crystalite_val_pt": {"path": str(val_path), "bytes": val_path.stat().st_size, "sha256": sha256(val_path)},
        "validation_report_sha256": sha256(destination),
        "validation_markdown_sha256": sha256(markdown_path),
    })
    provenance["validation_passed"] = report["passed"]
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
