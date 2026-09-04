#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CRYSTALITE="$ROOT/reference/crystalite"
TASK="$ROOT/cont_task/mp_task"

cd "$CRYSTALITE"
uv run python src/data/download_datasets.py --datasets mp20 --out "$TASK/data"
uv run python - "$TASK/pre_train/checkpoints" <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import sys

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
print(hf_hub_download(
    repo_id="joshrosie/crystalite-datasets",
    repo_type="dataset",
    filename="csp_mp20_best.pt",
    local_dir=str(destination),
))
PY
