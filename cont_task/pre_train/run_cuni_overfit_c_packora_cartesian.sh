#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/cuni_overfit_common.sh"
GPUS=${GPUS:-4,5}
NAME=${NAME:-c_packora_cartesian_normalized_vfm}
PROFILE=${PROFILE:-$DATA_ROOT/normalization.json}
AUGMENT_TRANSLATION=${AUGMENT_TRANSLATION:-0}
[[ -f $PROFILE ]] || { echo "missing $PROFILE" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/$NAME"
EXTRA_ARGS=(--best_ckpt)
if (( AUGMENT_TRANSLATION == 0 )); then
  EXTRA_ARGS+=(--no-augment_translation)
else
  EXTRA_ARGS+=(--augment_translation)
fi
cd "$CRYSTALITE"
exec env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. uv run python src/train_crystalite.py \
  "${COMMON_ARGS[@]}" --devices 2 --training_objective vfm_l1 --coordinate_repr cartesian \
  --normalization_profile "$PROFILE" "${EXTRA_ARGS[@]}" \
  --vfm_coord_weight 10 --vfm_lattice_weight 1 --temperature_conditioning \
  --temperature_dropout 0.1 --wandb_project cuni-overfit --wandb_name "$NAME" \
  --output_dir "$OUTPUT_ROOT/$NAME"
