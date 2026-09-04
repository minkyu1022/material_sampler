#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/spml_minkyu_kim/joint_sampler
REPORT="$ROOT/cont_task/data/cuni_reference_relaxed_full_v2/validation_report.json"
DATA="$ROOT/cont_task/data/crystalite_cuni"
OUT="${1:-$ROOT/cont_task/pre_train/runs/cuni_csp_152m_b64_$(date +%Y%m%d_%H%M%S)}"

python3 - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"Phase-1 gate failed: missing {p}")
r = json.loads(p.read_text())
if not r.get("passed") or r.get("converged") != 376200:
    raise SystemExit(f"Phase-1 gate failed: passed={r.get('passed')} converged={r.get('converged')}")
print("Phase-1 gate: PASS")
PY

mkdir -p "$OUT"
printf '%q ' "$0" "$@" > "$OUT/launcher_command.txt"
printf '\n' >> "$OUT/launcher_command.txt"
cp "$ROOT/reference/crystalite/uv.lock" "$OUT/uv.lock"

cd "$ROOT/reference/crystalite"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec uv run python src/train_crystalite.py \
  --devices 8 \
  --csp \
  --data_root "$DATA" \
  --dataset_name custom \
  --output_dir "$OUT" \
  --nmax 108 \
  --bf16 \
  --max_steps 1400000 \
  --type_encoding atomic_number \
  --batch_size 64 \
  --d_model 768 \
  --n_heads 12 \
  --n_layers 14 \
  --use_distance_bias \
  --use_edge_bias \
  --edge_bias_n_freqs 12 \
  --edge_bias_hidden_dim 256 \
  --edge_bias_n_rbf 32 \
  --gem_per_layer \
  --lattice_embed_mode mlp \
  --lattice_repr ltri \
  --coord_loss_mode frac_mse \
  --loss_weights 10 20 10 \
  --sigma_data_coord 0.3 \
  --sigma_data_lattice 0.3 \
  --ema_decay 0.99999 \
  --sample_frequency 1000 \
  --sample_count 256 \
  --sample_chunk_size 32 \
  --best_ckpt
