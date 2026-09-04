#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/spml_minkyu_kim/joint_sampler
OUT=$ROOT/cont_task/data/cuni_reference_relaxed_full_v2
PROCESSED=$ROOT/cont_task/data/processed/cuni_tokens.pt
BALANCED=$ROOT/cont_task/data/processed/cuni_tokens_balanced_cap1000.pt
CRYSTALITE_DATA=$ROOT/cont_task/data/crystalite_cuni
SEND=$ROOT/.agent-tools/discord/discord-control.sh
PID=$(cat "$OUT/postprocess.pid")
LAST=$(date +%s)
report() { printf '%s\n' "$1" | "$SEND" send --stdin >> "$OUT/postprocess_monitor.log" 2>&1; }

while kill -0 "$PID" 2>/dev/null; do
  sleep 30
  now=$(date +%s)
  if (( now - LAST >= 1740 )); then
    progress=$(cat "${PROCESSED%.pt}.progress.json" 2>/dev/null || echo '{"processed":0,"failures":0}')
    report "Cu–Ni frozen-subset preprocessing checkpoint: PID $PID alive, progress $progress, target 136826."
    LAST=$now
  fi
done
test -s "$PROCESSED" || { report "Cu–Ni subset preprocessing failed. Log: $OUT/preprocess_subset.log"; exit 1; }
cd "$ROOT/reference/crystalite"
uv run python "$ROOT/cont_task/data/build_cuni_balanced_view.py" \
  --master "$PROCESSED" --output "$BALANCED" --cap 1000 \
  > "$OUT/balanced_view.log" 2>&1 || {
    report "Cu–Ni balanced training-view build failed. Log: $OUT/balanced_view.log"
    exit 1
  }
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run python \
  "$ROOT/cont_task/data/validate_cuni_phase1.py" \
  --source "$ROOT/reference/JANUS/janus_reproduce/outputs/cuni_reference_n108_full" \
  --relaxed "$OUT" --processed "$PROCESSED" --balanced "$BALANCED" --workers 32 \
  --subset-manifest "$OUT/frozen_subset_manifest.json" > "$OUT/validation_subset.log" 2>&1 || {
    report "Cu–Ni frozen-subset validation failed. Log: $OUT/validation_subset.log"
    exit 1
  }
report "Cu–Ni frozen 136826-subset preprocessing and full validation PASS. Master: $PROCESSED; balanced cap-1000 training view: $BALANCED; report: $OUT/validation_report.json"
