import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).with_name("build_cuni_balanced_view.py")
SPEC = importlib.util.spec_from_file_location("build_cuni_balanced_view", MODULE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_balanced_view_caps_only_large_compositions_and_is_deterministic(tmp_path):
    items = ([{"mp_id": f"a{i}", "n_cu": 0} for i in range(5)]
             + [{"mp_id": f"b{i}", "n_cu": 1} for i in range(2)])
    master = tmp_path / "master.pt"
    torch.save(items, master)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    MOD.build(master, first, 3)
    MOD.build(master, second, 3)
    one = torch.load(first, weights_only=False)
    two = torch.load(second, weights_only=False)
    assert [x["mp_id"] for x in one] == [x["mp_id"] for x in two]
    assert sum(x["n_cu"] == 0 for x in one) == 3
    assert sum(x["n_cu"] == 1 for x in one) == 2
