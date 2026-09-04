#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/cuni_overfit_common.sh"
GPUS=${GPUS:-2,3}
NAME=${NAME:-b_packora_fractional_vfm}
mkdir -p "$OUTPUT_ROOT/$NAME"
cd "$CRYSTALITE"
exec env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. uv run python src/train_crystalite.py \
  "${COMMON_ARGS[@]}" --devices 2 --training_objective vfm_l1 --coordinate_repr fractional \
  --vfm_coord_weight 10 --vfm_lattice_weight 1 --temperature_conditioning \
  --temperature_dropout 0.1 --wandb_project cuni-overfit --wandb_name "$NAME" \
  --output_dir "$OUTPUT_ROOT/$NAME"

