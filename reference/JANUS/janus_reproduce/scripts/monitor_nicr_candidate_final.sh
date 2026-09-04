#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/spml_minkyu_kim/joint_sampler/reference/JANUS/janus_reproduce
LOG="$ROOT/logs/nicr_candidate_final"
SEND=/home/spml_minkyu_kim/joint_sampler/.agent-tools/discord/discord-control.sh
FCC_CAL=$(cat "$LOG/calibrate_fcc.pid")
BCC_CAL=$(cat "$LOG/calibrate_bcc.pid")
LAST=$(date +%s)
echo $$ > "$LOG/monitor.pid"

report() {
  printf '%s\n' "$1" | "$SEND" send --stdin >> "$LOG/monitor.log" 2>&1
}
trap 'report "Ni–Cr candidate pipeline failed at monitor line $LINENO. Inspect $LOG."' ERR

while kill -0 "$FCC_CAL" 2>/dev/null || kill -0 "$BCC_CAL" 2>/dev/null; do
  sleep 30
  now=$(date +%s)
  if (( now - LAST >= 1740 )); then
    report "Ni–Cr candidate prior calibration checkpoint: FCC PID $FCC_CAL $(kill -0 "$FCC_CAL" 2>/dev/null && echo alive || echo ended), BCC PID $BCC_CAL $(kill -0 "$BCC_CAL" 2>/dev/null && echo alive || echo ended)."
    LAST=$now
  fi
done

cd "$ROOT"
for phase in fcc bcc; do
  test -s "outputs/nicr_calibration_candidate/${phase}_prior.json" || {
    report "Ni–Cr candidate ${phase^^} prior calibration failed; output is missing. Log: $LOG/calibrate_${phase}.log"
    exit 1
  }
done

.venv/bin/python - <<'PY' > "$LOG/hamiltonian_gate.log"
import json
import math
from pathlib import Path
from janus_reproduce.nicr_train import NiCrTrainConfig, effective_hamiltonian_manifest
for phase in ('fcc', 'bcc'):
    prior = json.loads(Path(f'outputs/nicr_calibration_candidate/{phase}_prior.json').read_text())
    if prior.get('status') != 'full calibration':
        raise ValueError(f'{phase} prior is not a full calibration')
    if len(prior.get('observations', ())) != 15 or len(prior.get('displacement_observations', ())) != 15:
        raise ValueError(f'{phase} prior calibration grid is incomplete')
    if not all(math.isfinite(value) for row in prior['observations'] for value in row.values() if isinstance(value, (int, float))):
        raise ValueError(f'{phase} prior contains non-finite observations')
    config = NiCrTrainConfig.from_json(Path(f'configs/nicr_{phase}/janus_fixed_composition.json'))
    print(json.dumps({phase: effective_hamiltonian_manifest(config)}, sort_keys=True))
PY

for phase in fcc bcc; do
  test ! -e "outputs/nicr_janus_${phase}_candidate_diagnostic/checkpoint.pt" || {
    report "Ni–Cr candidate ${phase^^} diagnostic fresh-run gate failed: checkpoint already exists."
    exit 1
  }
done

for phase in fcc bcc; do
  gpu=0; test "$phase" = bcc && gpu=1
  config="configs/nicr_${phase}/candidate_diagnostic.json"
  command="cd '$ROOT' && echo \$\$ > '$LOG/train_${phase}.pid' && exec env CUDA_VISIBLE_DEVICES=$gpu JANUS_CONFIG=$config ./scripts/train_nicr_${phase}_janus.sh > '$LOG/train_${phase}.log' 2>&1"
  printf '%s\n' "$command" > "$LOG/train_${phase}.command"
  tmux new-session -d -s "nicr-${phase}-candidate-final" "$command"
done
sleep 3
report "Ni–Cr candidate Hamiltonian gate PASS. Required 24-round pre-production diagnostic started: FCC GPU0 PID $(cat "$LOG/train_fcc.pid"), BCC GPU1 PID $(cat "$LOG/train_bcc.pid"). Logs: $LOG/train_{fcc,bcc}.log"

FCC=$(cat "$LOG/train_fcc.pid")
BCC=$(cat "$LOG/train_bcc.pid")
while kill -0 "$FCC" 2>/dev/null || kill -0 "$BCC" 2>/dev/null; do
  sleep 30
  now=$(date +%s)
  if (( now - LAST >= 1740 )); then
    fcc_round=$(wc -l < outputs/nicr_janus_fcc_candidate_diagnostic/metrics.jsonl 2>/dev/null || echo 0)
    bcc_round=$(wc -l < outputs/nicr_janus_bcc_candidate_diagnostic/metrics.jsonl 2>/dev/null || echo 0)
    report "Ni–Cr candidate diagnostic checkpoint: FCC GPU0 PID $FCC round $fcc_round/24; BCC GPU1 PID $BCC round $bcc_round/24."
    LAST=$now
  fi
done

for phase in fcc bcc; do
  test -s "outputs/nicr_janus_${phase}_candidate_diagnostic/checkpoint.pt" || {
    report "Ni–Cr candidate ${phase^^} diagnostic failed; checkpoint missing. Log: $LOG/train_${phase}.log"
    exit 1
  }
done

.venv/bin/python - <<'PY' > "$LOG/diagnostic_checkpoint_gate.log"
from pathlib import Path
import torch
from janus_reproduce.nicr_train import NiCrTrainConfig, effective_hamiltonian_manifest
for phase in ('fcc', 'bcc'):
    config = NiCrTrainConfig.from_json(Path(f'configs/nicr_{phase}/candidate_diagnostic.json'))
    expected = effective_hamiltonian_manifest(config)
    checkpoint = torch.load(config.output / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if checkpoint.get('round') != 24:
        raise ValueError(f'{phase} diagnostic stopped at round {checkpoint.get("round")}')
    if checkpoint.get('provenance', {}).get('effective_hamiltonian') != expected:
        raise ValueError(f'{phase} checkpoint Hamiltonian provenance mismatch')
    print(phase, 'round=24', 'hamiltonian=PASS')
PY

EVAL_OUT=outputs/nicr_candidate_diagnostic_eval
mkdir -p "$EVAL_OUT/fcc" "$EVAL_OUT/bcc"
launch_eval() {
  local gpu=$1 phase=$2; shift 2
  local command="cd '$ROOT' && echo \$\$ > '$LOG/eval_${phase}.pid' && { true"
  local n
  for n in "$@"; do
    command+=" && env CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/audit_nicr_path_weights.py --config configs/nicr_${phase}/candidate_diagnostic.json --temperature 1200 --target-cr $n --samples 256 --batch-size 24 --seed $((2026 + n)) --output '$EVAL_OUT/$phase/T1200_n${n}.json'"
  done
  command+="; } > '$LOG/eval_${phase}.log' 2>&1"
  tmux new-session -d -s "nicr-${phase}-candidate-eval" "$command"
}
launch_eval 0 fcc 27 54 81
launch_eval 1 bcc 32 64 96
sleep 3
report "Ni–Cr 24-round training completed. Matched 1200 K candidate diagnostics started: FCC GPU0 PID $(cat "$LOG/eval_fcc.pid"), BCC GPU1 PID $(cat "$LOG/eval_bcc.pid"); 256 samples at x_Cr=0.25/0.50/0.75."
for phase in fcc bcc; do
  pid=$(cat "$LOG/eval_${phase}.pid")
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
for item in fcc:27 fcc:54 fcc:81 bcc:32 bcc:64 bcc:96; do
  phase=${item%%:*}; n=${item##*:}
  test -s "$EVAL_OUT/$phase/T1200_n${n}.json" || {
    report "Ni–Cr candidate diagnostic evaluation failed: missing $phase n=$n. Logs: $LOG/eval_{fcc,bcc}.log"
    exit 1
  }
done
report "Ni–Cr corrected-Hamiltonian 24-round diagnostics and matched 1200 K evaluations completed. Results are ready for g_u/g_v and production-training gate review."
