"""TOML configuration loading with the few validation rules that protect correctness."""

import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    if config.get("system") == "ising":
        if config["reveal_steps"] < 1 or config["lattice_size"] < 2:
            raise ValueError("Ising lattice and reveal step counts must be positive")
    elif config.get("system"):
        if config["n_atoms"] < 1 or config["steps"] < 1:
            raise ValueError("Alloy atom and sampler step counts must be positive")
    else:
        raise ValueError("missing system")
    return config
