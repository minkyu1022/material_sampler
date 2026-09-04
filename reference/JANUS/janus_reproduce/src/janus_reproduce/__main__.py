from __future__ import annotations

import argparse
import json

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="janus_reproduce")
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(load_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
