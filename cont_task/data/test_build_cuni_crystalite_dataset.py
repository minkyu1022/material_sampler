import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE = Path(__file__).with_name("build_cuni_crystalite_dataset.py")
SPEC = importlib.util.spec_from_file_location("build_cuni_crystalite_dataset", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class Lattice:
    abc = (2.0, 3.0, 4.0)
    angles = (90.0, 90.0, 120.0)


class Structure:
    lattice = Lattice()


def test_lattice_encoding_matches_crystalite():
    got = MOD.lattice_to_y(Structure()).numpy()
    expected = np.r_[np.log([2.0, 3.0, 4.0]), np.cos(np.deg2rad([90, 90, 120]))]
    np.testing.assert_allclose(got, expected, atol=1e-7)


def test_fingerprint_is_translation_invariant():
    types = torch.tensor([28, 29])
    frac = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
    cell = np.eye(3) * 3.5
    shifted = (frac + np.array([0.31, 0.17, 0.43])) % 1.0
    assert MOD.structure_fingerprint(types, frac, cell) == MOD.structure_fingerprint(types, shifted, cell)
    assert MOD.structure_fingerprint(types, frac, cell) == MOD.structure_fingerprint(types.flip(0), frac[::-1].copy(), cell)


def test_singleton_composition_stays_in_train(tmp_path):
    def item(name, n_cu):
        return {"mp_id": name, "n_cu": n_cu}

    split = MOD.write_crystalite_splits(
        [item("only", 1), item("a", 2), item("b", 2)], tmp_path
    )
    train = torch.load(tmp_path / "processed/mp20_tokens_train_nmax108.pt", weights_only=False)
    val = torch.load(tmp_path / "processed/mp20_tokens_val_nmax108.pt", weights_only=False)
    assert {entry["mp_id"] for entry in train} == {"only", "b"}
    assert {entry["mp_id"] for entry in val} == {"a"}
    assert split["train"] == 2 and split["val"] == 1
