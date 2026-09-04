#!/usr/bin/env bash
set -euo pipefail
exec .venv/bin/python scripts/evaluate_cuni.py "$@"
