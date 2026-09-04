from pathlib import Path

from janus_reproduce.config import load_config


def test_all_published_configs_validate():
    root = Path(__file__).parents[1] / "configs"
    for path in root.glob("*.toml"):
        assert load_config(path)["system"]
