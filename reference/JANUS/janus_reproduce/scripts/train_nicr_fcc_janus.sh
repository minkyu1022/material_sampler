#!/usr/bin/env bash
set -euo pipefail
config="${JANUS_CONFIG:-configs/nicr_fcc/janus_fixed_composition.json}"
exec .venv/bin/python scripts/train_nicr_janus.py --config "$config" "$@"
