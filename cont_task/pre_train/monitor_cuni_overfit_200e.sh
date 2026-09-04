#!/usr/bin/env bash
set -u
ROOT=${ROOT:-/home/spml_minkyu_kim/joint_sampler}
STAMP=${1:?stamp required}
INTERVAL=${INTERVAL:-1800}
LOGROOT="$ROOT/cont_task/pre_train/logs/overfit_200e_$STAMP"
SEND="$ROOT/.agent-tools/discord/discord-control.sh"
progress() {
  local log=$1
  [[ -f $log ]] || { echo queued; return; }
  tr '\r' '\n' < "$log" | grep -Eo '[0-9]+/14200' | tail -1 || echo initializing
}
while :; do
  a=$(progress "$LOGROOT/A200.log")
  b=$(progress "$LOGROOT/B200.log")
  c=$(progress "$LOGROOT/C200.log")
  sa=exited; sb=exited; sc=queued
  tmux has-session -t cuni-a-200e 2>/dev/null && sa=running
  tmux has-session -t cuni-b-200e 2>/dev/null && sb=running
  tmux has-session -t cuni-c-200e 2>/dev/null && sc=running
  tmux has-session -t cuni-c-200e-queue 2>/dev/null || [[ $sc == running ]] || sc=exited
  relax=$(python3 "$ROOT/cont_task/data/cuni_progress.py" "$ROOT/cont_task/data/cuni_reference_relaxed_full_v2")
  printf 'Cu–Ni 200-epoch checkpoint (%s Montreal)\nA: %s, %s\nB: %s, %s\nC: %s, %s\nPhase1: %s\n' \
    "$(TZ=America/Toronto date '+%m-%d %H:%M %Z')" "$sa" "$a" "$sb" "$b" "$sc" "$c" "$relax" | "$SEND" send --stdin >> "$LOGROOT/monitor.log" 2>&1
  [[ $sa == exited && $sb == exited && $sc == exited ]] && exit 0
  sleep "$INTERVAL"
done
