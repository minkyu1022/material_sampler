#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${NICR_EVAL_OUT:-outputs/nicr_candidate_full_rung_eval}
SAMPLES=${NICR_EVAL_SAMPLES:-256}
FCC_CONFIG=${NICR_FCC_CONFIG:-configs/nicr_fcc/janus_fixed_composition.json}
BCC_CONFIG=${NICR_BCC_CONFIG:-configs/nicr_bcc/janus_fixed_composition.json}
TEMPERATURES=(${NICR_EVAL_TEMPERATURES:-600 750 900 1050 1200 1350 1500})
cd "$ROOT"
mkdir -p "$OUT"/{fcc,bcc} logs/nicr_candidate_full_rung_eval

run_shard() {
  local phase=$1 config=$2 gpu=$3 shard=$4
  for temperature in "${TEMPERATURES[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/evaluate_nicr_janus.py \
      --config "$config" --temperature "$temperature" --samples "$SAMPLES" \
      --shard-index "$shard" --shard-count 2 --output "$OUT/$phase"
  done
}

run_shard fcc "$FCC_CONFIG" 0 0 > logs/nicr_candidate_full_rung_eval/fcc_0.log 2>&1 & p0=$!
run_shard fcc "$FCC_CONFIG" 1 1 > logs/nicr_candidate_full_rung_eval/fcc_1.log 2>&1 & p1=$!
run_shard bcc "$BCC_CONFIG" 2 0 > logs/nicr_candidate_full_rung_eval/bcc_0.log 2>&1 & p2=$!
run_shard bcc "$BCC_CONFIG" 3 1 > logs/nicr_candidate_full_rung_eval/bcc_1.log 2>&1 & p3=$!
printf '%s\n' "$p0 $p1 $p2 $p3" > logs/nicr_candidate_full_rung_eval/pids
wait "$p0" "$p1" "$p2" "$p3"

for temperature in "${TEMPERATURES[@]}"; do
  .venv/bin/python scripts/aggregate_nicr_bar.py --input "$OUT/fcc" --n-atoms 108 \
    --temperature "$temperature" --output "$OUT/fcc/T${temperature}_bar.json"
  .venv/bin/python scripts/aggregate_nicr_bar.py --input "$OUT/bcc" --n-atoms 128 \
    --temperature "$temperature" --output "$OUT/bcc/T${temperature}_bar.json"
done
.venv/bin/python scripts/plot_nicr_fig3b_from_rungs.py --root "$OUT" --output "$OUT/fig3b"
