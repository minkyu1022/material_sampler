#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/spml_minkyu_kim/joint_sampler}
CRYSTALITE=${CRYSTALITE:-$ROOT/reference/crystalite}
DATA_ROOT=${DATA_ROOT:-$ROOT/cont_task/data/cuni_overfit_10k}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/cont_task/pre_train/outputs/cuni_overfit}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
DMODEL=${DMODEL:-768}
NHEADS=${NHEADS:-12}
NLAYERS=${NLAYERS:-14}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
EMA_DECAY=${EMA_DECAY:-0.9999}
SEED=${SEED:-20260904}
PAIR=${PAIR:-0}

MANIFEST=$DATA_ROOT/subset_manifest.json
[[ -f $MANIFEST ]] || { echo "missing $MANIFEST" >&2; exit 1; }
TRAIN_SIZE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["train"])' "$MANIFEST")
GLOBAL_BATCH=$((BATCH_SIZE * 2))
STEPS_PER_EPOCH=$(((TRAIN_SIZE + GLOBAL_BATCH - 1) / GLOBAL_BATCH))
MAX_STEPS=${MAX_STEPS_OVERRIDE:-$((EPOCHS * STEPS_PER_EPOCH))}
PAIR_ARGS=()
(( PAIR == 0 )) || PAIR_ARGS+=(--pair)

COMMON_ARGS=(
  --csp --dataset_name custom --data_root "$DATA_ROOT" --nmax 108
  --type_encoding atomic_number --batch_size "$BATCH_SIZE" --num_workers 4
  --d_model "$DMODEL" --n_heads "$NHEADS" --n_layers "$NLAYERS"
  --use_distance_bias --use_edge_bias --edge_bias_n_freqs 12
  --edge_bias_hidden_dim 256 --edge_bias_n_rbf 32 --gem_per_layer
  --coord_head_mode direct --lattice_embed_mode mlp --lattice_repr ltri
  --lr "$LR" --weight_decay "$WEIGHT_DECAY" --lr_warmup_steps "$WARMUP_STEPS"
  --max_steps "$MAX_STEPS" --log_every 25 --val_every "$STEPS_PER_EPOCH"
  --val_batches 16 --ema_decay "$EMA_DECAY" --ckpt_every "$STEPS_PER_EPOCH"
  --ckpt_latest_only --sample_frequency 0 --bf16 --seed "$SEED"
  "${PAIR_ARGS[@]}"
)
