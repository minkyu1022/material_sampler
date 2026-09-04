#!/usr/bin/env bash
set -euo pipefail
exec .venv/bin/python scripts/train_cuni.py "$@"
