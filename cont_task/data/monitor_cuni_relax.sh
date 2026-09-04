#!/usr/bin/env bash
set -u
ROOT=/home/spml_minkyu_kim/joint_sampler
OUT="$ROOT/cont_task/data/cuni_reference_relaxed_full_v2"
PROCESSED="$ROOT/cont_task/data/processed/cuni_tokens.pt"
CRYSTALITE_DATA="$ROOT/cont_task/data/crystalite_cuni"
SEND="$ROOT/.agent-tools/discord/discord-control.sh"
TOTAL=376200
START=$(date -d "$(cat "$OUT/started_at.txt")" +%s)
LAST_REPORT=$(date +%s)
RESTARTS=0
DISK_WARNED=0

snapshot() { python3 "$ROOT/cont_task/data/cuni_progress.py" "$OUT"; }
field() { python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"; }
current_pid() { cat "$OUT/runtime_pid" 2>/dev/null || cat "$OUT/pid"; }
restart_relaxation() {
  RESTARTS=$((RESTARTS+1))
  (( RESTARTS <= 3 )) || return 1
  : > "$OUT/runtime_pid"
  tmux kill-session -t cuni-relax-full-v2 2>/dev/null || true
  tmux new-session -d -s cuni-relax-full-v2 "cd '$ROOT' && exec bash '$OUT/command.txt' >> '$OUT/stdout.log' 2>&1"
  for _ in $(seq 1 30); do
    [[ -s "$OUT/runtime_pid" ]] && break
    sleep 1
  done
  [[ -s "$OUT/runtime_pid" ]]
}

while true; do
  sleep 30
  SNAP=$(snapshot)
  DONE=$(printf '%s' "$SNAP" | field converged)
  BAD=$(printf '%s' "$SNAP" | field bad)
  RECORDED=$(printf '%s' "$SNAP" | field recorded)
  PID=$(current_pid)
  NOW=$(date +%s)
  AVAILABLE=$(df --output=avail -B1 "$OUT" | tail -1 | tr -d ' ')
  ELAPSED=$((NOW-START))
  if (( DONE > 0 )); then
    REMAIN=$(((TOTAL-DONE)*ELAPSED/DONE))
    ETA_EPOCH=$((NOW+REMAIN))
  else
    ETA_EPOCH=0
  fi
  printf '{"updated_epoch":%s,"pid":%s,"recorded":%s,"converged":%s,"bad":%s,"total":%s,"elapsed_seconds":%s,"eta_epoch":%s,"available_bytes":%s}\n' \
    "$NOW" "$PID" "$RECORDED" "$DONE" "$BAD" "$TOTAL" "$ELAPSED" "$ETA_EPOCH" "$AVAILABLE" > "$OUT/status.json"
  if (( AVAILABLE < 107374182400 && DISK_WARNED == 0 )); then
    printf 'Cu–Ni relaxation 디스크 경고: 남은 용량 %s GiB. PID %s, 완료 %s/%s.\n' "$((AVAILABLE/1073741824))" "$PID" "$DONE" "$TOTAL" | "$SEND" send --stdin
    DISK_WARNED=1
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    if (( RECORDED < TOTAL || BAD > 0 )); then
      printf 'Cu–Ni relaxation 비정상/미수렴 종료 감지: PID %s, 기록 %s/%s, 실패/미수렴 %s. resumable job 자동 재시작을 시도합니다.\n' "$PID" "$RECORDED" "$TOTAL" "$BAD" | "$SEND" send --stdin
      for _ in $(seq 1 12); do
        pgrep -f '[r]elax_cuni_dataset.py.*cuni_reference_relaxed_full_v2' >/dev/null || break
        sleep 5
      done
      if pgrep -f '[r]elax_cuni_dataset.py.*cuni_reference_relaxed_full_v2' >/dev/null; then
        printf 'Cu–Ni orphan worker가 60초 후에도 남아 있어 종료 후 복구합니다.\n' | "$SEND" send --stdin
        pkill -TERM -f '[r]elax_cuni_dataset.py.*cuni_reference_relaxed_full_v2' || true
        sleep 5
        if pgrep -f '[r]elax_cuni_dataset.py.*cuni_reference_relaxed_full_v2' >/dev/null; then
          pkill -KILL -f '[r]elax_cuni_dataset.py.*cuni_reference_relaxed_full_v2' || true
          sleep 2
        fi
        if pgrep -f '[r]elax_cuni_dataset.py.*cuni_reference_relaxed_full_v2' >/dev/null; then
          printf 'Cu–Ni orphan worker 종료 실패. 동시 writer 위험 때문에 자동 재시작을 중단합니다.\n' | "$SEND" send --stdin
          exit 1
        fi
      fi
      if restart_relaxation; then
        printf 'Cu–Ni relaxation 자동 복구 완료: 새 PID %s, 기존 converged frame은 건너뜁니다.\n' "$(current_pid)" | "$SEND" send --stdin
        continue
      fi
      printf 'Cu–Ni relaxation 자동 복구 3회 실패. 로그: %s/stdout.log\n' "$OUT" | "$SEND" send --stdin
      exit 1
    fi
    break
  fi
  if (( NOW - LAST_REPORT >= 1740 )); then
    if (( DONE > 0 )); then
      ETA=$(TZ=America/Toronto date -d "@$ETA_EPOCH" '+%Y-%m-%d %H:%M %Z')
    else
      ETA='산정 중'
    fi
    printf 'Cu–Ni relaxation 30분 checkpoint: PID %s alive, 완료 %s/%s, 실패/미수렴 %s, Montreal ETA %s.\n' "$PID" "$DONE" "$TOTAL" "$BAD" "$ETA" | "$SEND" send --stdin
    LAST_REPORT=$NOW
  fi
done

SNAP=$(snapshot); DONE=$(printf '%s' "$SNAP" | field converged); BAD=$(printf '%s' "$SNAP" | field bad)
printf 'Cu–Ni relaxation process ended: 완료 %s/%s, 실패/미수렴 %s. 후처리와 전체 validation을 시작합니다.\n' "$DONE" "$TOTAL" "$BAD" | "$SEND" send --stdin
mkdir -p "$(dirname "$PROCESSED")"
cd "$ROOT/reference/crystalite" || exit 1
for ATTEMPT in 1 2 3; do
  if OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run python "$ROOT/cont_task/data/build_cuni_crystalite_dataset.py" --input "$OUT" --output "$PROCESSED" --crystalite-root "$CRYSTALITE_DATA" > "$OUT/preprocess.log" 2>&1 && \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run python "$ROOT/cont_task/data/validate_cuni_phase1.py" --source "$ROOT/reference/JANUS/janus_reproduce/outputs/cuni_reference_n108_full" --relaxed "$OUT" --processed "$PROCESSED" --workers 32 > "$OUT/validation.log" 2>&1; then
    printf 'Cu–Ni Phase 1 자동 후처리 및 validation 통과. processed dataset: %s, report: %s/validation_report.json\n' "$PROCESSED" "$OUT" | "$SEND" send --stdin
    exit 0
  fi
  printf 'Cu–Ni Phase 1 후처리/validation 실패 (시도 %s/3). 로그를 보존하고 60초 후 재시도합니다: %s/preprocess.log, %s/validation.log\n' "$ATTEMPT" "$OUT" "$OUT" | "$SEND" send --stdin
  sleep 60
done
exit 1
