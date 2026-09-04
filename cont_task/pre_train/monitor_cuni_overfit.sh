#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/home/spml_minkyu_kim/joint_sampler}
MANIFEST=${1:?run manifest required}
INTERVAL=${INTERVAL:-1800}
DISCORD="$ROOT/.agent-tools/discord/discord-control.sh"
declare -A reported

report() {
  local lines=() alive=0 key sess pane gpus log state progress
  while IFS='|' read -r key sess pane gpus log; do
    if tmux has-session -t "$sess" 2>/dev/null; then
      state="running PID $pane"
      alive=1
    else
      state="exited"
      if [[ -f "$log.exit" ]]; then state+=" code $(cat "$log.exit")"; fi
    fi
    progress=$(tr '\r' '\n' < "$log" | grep -E 'train:.*[0-9]+/7100' | tail -1 | sed -E 's/.*train: *//')
    lines+=("$key GPUs $gpus — $state — ${progress:-initializing}")
  done < "$MANIFEST"
  local gpu relax msg
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n '1,6p' | paste -sd ';' -)
  relax=$(python3 "$ROOT/cont_task/data/cuni_progress.py" "$ROOT/cont_task/data/cuni_reference_relaxed_full_v2")
  msg=$(printf 'Cu–Ni checkpoint (%s Montreal)\n%s\nRelaxation: %s\nGPU: %s' \
    "$(TZ=America/Toronto date '+%m-%d %H:%M %Z')" \
    "$(printf '%s\n' "${lines[@]}")" "$relax" "$gpu")
  printf '%s\n' "$msg" | "$DISCORD" send --stdin >> "$ROOT/cont_task/pre_train/logs/overfit_100e/monitor.log" 2>&1
  return "$alive"
}

while :; do
  report && exit 0
  sleep "$INTERVAL"
done
