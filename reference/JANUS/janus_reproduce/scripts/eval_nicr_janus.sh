#!/usr/bin/env bash
set -euo pipefail
exec .venv/bin/python scripts/evaluate_nicr_janus.py "$@"
