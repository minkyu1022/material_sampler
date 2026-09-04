#!/usr/bin/env bash
set -Eeuo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "$TOOLS_DIR/profile.env"
export PATH="$HOME/.local/bin:$PATH"

pane_id() {
  tmux list-panes -t "$TMUX_SESSION" -F '#{pane_id}' 2>/dev/null | head -n 1
}

launch_agent() {
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    return
  fi
  local launch_command
  printf -v launch_command \
    'export OMX_NOTIFY_PROFILE=%q OMX_LAUNCH_POLICY=direct; exec omx resume %q --dangerously-bypass-approvals-and-sandbox' \
    "$PROFILE_NAME" "$CODEX_RESUME_SESSION"
  tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT" "$launch_command"
  sleep 2
}

restart_listener() {
  local pane
  pane="$(pane_id)"
  [[ "$pane" =~ ^%[0-9]+$ ]] || {
    echo "No live pane for tmux session $TMUX_SESSION" >&2
    return 1
  }
  OMX_REPLY_PROFILE_PANES="$PROFILE_NAME:$pane" \
    node "$TOOLS_DIR/multi_channel_reply_listener.mjs" restart
}

send_message() {
  local pane
  pane="$(pane_id)"
  [[ "$pane" =~ ^%[0-9]+$ ]] || {
    echo "No live pane for tmux session $TMUX_SESSION" >&2
    return 1
  }
  OMX_RESULT_PROJECT="$PROJECT_ROOT" \
  OMX_RESULT_TMUX_SESSION="$TMUX_SESSION" \
  OMX_RESULT_TMUX_PANE="$pane" \
    node "$TOOLS_DIR/send_discord_status.mjs" \
      --profile "$PROFILE_NAME" "$@"
}

action="${1:-status}"
if (($#)); then shift; fi
case "$action" in
  start)
    launch_agent
    restart_listener
    send_message "Discord remote control ready. Reply to this exact message with: 확인"
    ;;
  launch)
    launch_agent
    ;;
  restart-listener)
    restart_listener
    ;;
  stop-listener)
    node "$TOOLS_DIR/multi_channel_reply_listener.mjs" stop
    ;;
  status)
    tmux list-panes -t "$TMUX_SESSION" \
      -F '#{session_name} #{pane_id} #{pane_current_command}' 2>/dev/null || true
    node "$TOOLS_DIR/multi_channel_reply_listener.mjs" status
    ;;
  verify)
    OMX_NOTIFY_PROFILE="$PROFILE_NAME" \
      node "$TOOLS_DIR/configure_omx_discord.mjs" verify
    ;;
  send)
    send_message "$@"
    ;;
  capture)
    tmux capture-pane -pt "$TMUX_SESSION" -S -100
    ;;
  *)
    echo "Usage: discord-control.sh start|launch|restart-listener|stop-listener|status|verify|send|capture" >&2
    exit 2
    ;;
esac
