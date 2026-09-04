import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).with_name("relax_cuni_dataset.py")
SPEC = importlib.util.spec_from_file_location("relax_cuni_dataset", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_completed_frames(tmp_path):
    manifest = tmp_path / "frames.jsonl"
    for folder in ("raw", "relaxed", "cif"):
        path = tmp_path / folder
        path.mkdir()
        artifact = path / "frame_0001.extxyz" if folder != "cif" else path / "frame_0001.cif"
        artifact.write_text("complete")
    manifest.write_text(
        '{"frame": 1, "status": "converged"}\n'
        '{"frame": 2, "status": "failed"}\n'
        '{"frame":'
    )
    assert MOD.completed_frames(manifest) == {1}
    assert manifest.read_text() == (
        '{"frame": 1, "status": "converged"}\n'
        '{"frame": 2, "status": "failed"}\n'
    )


def test_completed_frames_rejects_empty_artifact(tmp_path):
    manifest = tmp_path / "frames.jsonl"
    for folder in ("raw", "relaxed", "cif"):
        path = tmp_path / folder
        path.mkdir()
        (path / "frame_0001.extxyz" if folder != "cif" else path / "frame_0001.cif").touch()
    manifest.write_text('{"frame": 1, "status": "converged"}\n')
    assert MOD.completed_frames(manifest) == set()


def test_reference_species_encoding_is_one_for_cu():
    species = np.array([0, 1, 1, 0], dtype=np.uint8)
    assert MOD.species_to_symbols(species) == ["Ni", "Cu", "Cu", "Ni"]


def test_manifest_refuses_incompatible_resume(tmp_path):
    path = tmp_path / "run_manifest.json"
    MOD.write_or_validate_manifest(path, {"created_utc": "first", "fmax": 0.01})
    MOD.write_or_validate_manifest(path, {"created_utc": "second", "fmax": 0.01})
    try:
        MOD.write_or_validate_manifest(path, {"created_utc": "third", "fmax": 0.02})
    except RuntimeError:
        pass
    else:
        raise AssertionError("incompatible manifest was accepted")
