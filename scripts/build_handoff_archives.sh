#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$ROOT/handoff_archives}
mkdir -p "$DEST"
cd "$ROOT"

archive() {
  local name=$1; shift
  for path in "$@"; do [[ -e "$path" ]] || { echo "missing required path: $path" >&2; return 1; }; done
  tar -cf "$DEST/$name.tar" "$@"
  sha256sum "$DEST/$name.tar" > "$DEST/$name.tar.sha256"
}

archive cuni_reference_mc \
  reference/JANUS/janus_reproduce/outputs/cuni_reference_n108_full

if pgrep -f '[r]elax_cuni_dataset.py.*cuni_reference_relaxed_full_v2' >/dev/null; then
  echo 'Phase-1 relaxation is still live; refusing an inconsistent snapshot.' >&2
  echo 'Run this script again after validation completes.' >&2
  exit 2
fi

archive cuni_phase1_relaxed \
  cont_task/data/cuni_reference_relaxed_full_v2 \
  cont_task/data/processed \
  cont_task/data/crystalite_cuni

archive cuni_pretraining \
  cont_task/data/cuni_overfit_10k \
  cont_task/pre_train/outputs/cuni_overfit/a_100e_20260904_053741/checkpoints/final.pt \
  cont_task/pre_train/outputs/cuni_overfit/b_100e_20260904_053741/checkpoints/final.pt \
  cont_task/pre_train/outputs/cuni_overfit/c_cartesian_fixed_direct_20260904_084715/checkpoints/final.pt \
  cont_task/pre_train/outputs/cuni_overfit/b_nfe_diagnostic \
  cont_task/pre_train/outputs/cuni_overfit/match_full_400nfe_regular

ls -lh "$DEST"/*.tar "$DEST"/*.sha256
