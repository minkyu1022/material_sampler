#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/cuni_overfit_common.sh"
GPUS=${GPUS:-0,1}
NAME=${NAME:-a_crystalite_fractional_edm}
mkdir -p "$OUTPUT_ROOT/$NAME"
cd "$CRYSTALITE"
exec env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. uv run python src/train_crystalite.py \
  "${COMMON_ARGS[@]}" --devices 2 --training_objective edm --coordinate_repr fractional \
  --coord_loss_mode frac_mse --loss_weights 10 20 10 \
  --sigma_data_coord 0.3 --sigma_data_lattice 0.3 \
  --wandb_project cuni-overfit --wandb_name "$NAME" --output_dir "$OUTPUT_ROOT/$NAME"

