#!/usr/bin/env python3
"""Train one paper-faithful fixed-composition Ni--Cr JANUS lattice model."""

from __future__ import annotations

import argparse
from pathlib import Path

from janus_reproduce.nicr_train import NiCrTrainConfig, train_nicr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train_nicr(NiCrTrainConfig.from_json(args.config))


if __name__ == "__main__":
    main()
