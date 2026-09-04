#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
TESTED_CODEX_VERSION="${TESTED_CODEX_VERSION:-0.144.4}"
TESTED_OMX_VERSION="${TESTED_OMX_VERSION:-0.20.3}"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PROFILE_NAME="${OMX_NOTIFY_PROFILE:-}"
CHANNEL_ID="${DISCORD_CHANNEL_ID:-}"
AUTHORIZED_USER_ID="${DISCORD_USER_ID:-}"
TMUX_SESSION="${TMUX_SESSION:-}"
INSTALL_MISSING=1
LAUNCH_AGENT=1
PATCH_AGENTS=1
REUSE_SESSION=0
DRY_RUN=0
BOT_TOKEN="${DISCORD_BOT_TOKEN:-${OMX_DISCORD_NOTIFIER_BOT_TOKEN:-}}"
unset DISCORD_BOT_TOKEN OMX_DISCORD_NOTIFIER_BOT_TOKEN
trap 'unset BOT_TOKEN DISCORD_BOT_TOKEN OMX_DISCORD_NOTIFIER_BOT_TOKEN' EXIT

usage() {
  cat <<'EOF'
Usage:
  bash setup_discord_remote_agent.sh \
    --project /absolute/project/path \
    --profile short-profile-name \
    --channel-id 123456789012345678 \
    --user-id 123456789012345678 \
    [--tmux-session codex-profile]

Options:
  --no-install       Do not install missing Codex/OMX CLIs.
  --no-launch        Configure files/profile only; do not start tmux/listener.
  --no-patch-agents  Do not add the Discord contract to project AGENTS.md.
  --reuse-session    Reuse an existing named tmux session after inspection.
  --dry-run          Validate arguments and print the plan without changes.
  -h, --help         Show this help.

Bot tokens are never accepted as command-line arguments. Set DISCORD_BOT_TOKEN
in the environment or enter it at the hidden terminal prompt.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --project)
      [[ $# -ge 2 ]] || die "--project requires a value"
      PROJECT_DIR="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      PROFILE_NAME="$2"
      shift 2
      ;;
    --channel-id)
      [[ $# -ge 2 ]] || die "--channel-id requires a value"
      CHANNEL_ID="$2"
      shift 2
      ;;
    --user-id)
      [[ $# -ge 2 ]] || die "--user-id requires a value"
      AUTHORIZED_USER_ID="$2"
      shift 2
      ;;
    --tmux-session)
      [[ $# -ge 2 ]] || die "--tmux-session requires a value"
      TMUX_SESSION="$2"
      shift 2
      ;;
    --no-install)
      INSTALL_MISSING=0
      shift
      ;;
    --no-launch)
      LAUNCH_AGENT=0
      shift
      ;;
    --no-patch-agents)
      PATCH_AGENTS=0
      shift
      ;;
    --reuse-session)
      REUSE_SESSION=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -d "$PROJECT_DIR" ]] || die "project directory does not exist: $PROJECT_DIR"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd -P)"

if [[ -z "$PROFILE_NAME" ]]; then
  PROFILE_NAME="$(basename "$PROJECT_DIR" | tr -c 'A-Za-z0-9._-' '-')"
  PROFILE_NAME="${PROFILE_NAME#-}"
  PROFILE_NAME="${PROFILE_NAME%-}"
fi
[[ "$PROFILE_NAME" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "profile may contain only letters, digits, dot, underscore, and hyphen"
[[ "$CHANNEL_ID" =~ ^[0-9]{17,20}$ ]] ||
  die "--channel-id must be a 17-20 digit Discord channel ID"
[[ "$AUTHORIZED_USER_ID" =~ ^[0-9]{17,20}$ ]] ||
  die "--user-id must be a 17-20 digit Discord user ID"

if [[ -z "$TMUX_SESSION" ]]; then
  TMUX_SESSION="codex-${PROFILE_NAME}"
fi
[[ "$TMUX_SESSION" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "tmux session may contain only letters, digits, dot, underscore, and hyphen"

cat <<EOF
Discord remote-agent installation plan
  project:      $PROJECT_DIR
  profile:      $PROFILE_NAME
  channel ID:   $CHANNEL_ID
  user ID:      $AUTHORIZED_USER_ID
  tmux session: $TMUX_SESSION
  install CLI:  $INSTALL_MISSING
  launch agent: $LAUNCH_AGENT
  patch AGENTS: $PATCH_AGENTS
EOF

if ((DRY_RUN)); then
  echo "Dry run complete; no files or external services were changed."
  exit 0
fi

for command_name in node npm tmux; do
  command -v "$command_name" >/dev/null 2>&1 ||
    die "$command_name is required; install Node.js 20+ and tmux first"
done

if ((LAUNCH_AGENT)) && ((REUSE_SESSION == 0)) &&
  tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  die "tmux session '$TMUX_SESSION' already exists; inspect it and rerun with --reuse-session or another name"
fi

NODE_MAJOR="$(node -p 'Number(process.versions.node.split(".")[0])')"
((NODE_MAJOR >= 20)) || die "Node.js 20 or newer is required"

export PATH="$HOME/.local/bin:$PATH"
install_cli_if_missing() {
  local command_name="$1"
  local package_spec="$2"
  if command -v "$command_name" >/dev/null 2>&1; then
    return
  fi
  ((INSTALL_MISSING)) || die "$command_name is missing; rerun without --no-install"
  echo "Installing $package_spec under $HOME/.local ..."
  npm install --global --prefix "$HOME/.local" "$package_spec"
  command -v "$command_name" >/dev/null 2>&1 ||
    die "$command_name was not found after installing $package_spec"
}

install_cli_if_missing codex "@openai/codex@$TESTED_CODEX_VERSION"
install_cli_if_missing omx "oh-my-codex@$TESTED_OMX_VERSION"

if ! codex login status >/dev/null 2>&1; then
  die "Codex is not authenticated. Run 'codex login', then rerun this script."
fi

echo "Preparing OMX user configuration ..."
omx setup --scope user --merge-agents
omx doctor

TOOLS_DIR="$PROJECT_DIR/.agent-tools/discord"
install -d -m 700 "$TOOLS_DIR"
for helper in \
  configure_omx_discord.mjs \
  multi_channel_reply_listener.mjs \
  send_discord_status.mjs; do
  [[ -f "$SCRIPT_DIR/$helper" ]] || die "bundle is missing $helper"
  install -m 700 "$SCRIPT_DIR/$helper" "$TOOLS_DIR/$helper"
  node --check "$TOOLS_DIR/$helper"
done

RUNTIME_ENV="$TOOLS_DIR/profile.env"
umask 077
{
  printf 'PROFILE_NAME=%q\n' "$PROFILE_NAME"
  printf 'PROJECT_ROOT=%q\n' "$PROJECT_DIR"
  printf 'TMUX_SESSION=%q\n' "$TMUX_SESSION"
} >"$RUNTIME_ENV"
chmod 600 "$RUNTIME_ENV"

CONTROL_SCRIPT="$TOOLS_DIR/discord-control.sh"
[[ -f "$SCRIPT_DIR/discord-control.sh" ]] || die "bundle is missing discord-control.sh"
install -m 700 "$SCRIPT_DIR/discord-control.sh" "$CONTROL_SCRIPT"
bash -n "$CONTROL_SCRIPT"

CONFIG_PATH="$HOME/.codex/.omx-config.json"
has_stored_token=0
if [[ -f "$CONFIG_PATH" ]] && node -e '
  const fs = require("fs");
  const p = process.argv[1];
  const root = JSON.parse(fs.readFileSync(p, "utf8"));
  const n = root.notifications || {};
  const found = Boolean(n["discord-bot"]?.botToken) ||
    Object.values(n.profiles || {}).some((x) => x?.["discord-bot"]?.botToken);
  process.exit(found ? 0 : 1);
' "$CONFIG_PATH"; then
  has_stored_token=1
fi

if [[ -z "$BOT_TOKEN" && $has_stored_token -eq 0 ]]; then
  [[ -t 0 ]] || die "set DISCORD_BOT_TOKEN through a secret environment before non-interactive setup"
  read -rsp 'Discord bot token: ' BOT_TOKEN
  printf '\n'
fi

export OMX_NOTIFY_PROFILE="$PROFILE_NAME"
export OMX_NOTIFY_DEFAULT_PROFILE="$PROFILE_NAME"
export DISCORD_CHANNEL_ID="$CHANNEL_ID"
export DISCORD_USER_ID="$AUTHORIZED_USER_ID"
export OMX_DISCORD_LIFECYCLE_EVENTS=0
if [[ -n "$BOT_TOKEN" ]]; then
  export DISCORD_BOT_TOKEN="$BOT_TOKEN"
fi
node "$TOOLS_DIR/configure_omx_discord.mjs" configure-profile
unset BOT_TOKEN DISCORD_BOT_TOKEN OMX_DISCORD_NOTIFIER_BOT_TOKEN
unset OMX_DISCORD_LIFECYCLE_EVENTS
chmod 600 "$CONFIG_PATH"

OMX_NOTIFY_PROFILE="$PROFILE_NAME" \
  node "$TOOLS_DIR/configure_omx_discord.mjs" verify

if ((PATCH_AGENTS)); then
  AGENTS_FILE="$PROJECT_DIR/AGENTS.md"
  START_MARKER='<!-- DISCORD_REMOTE_CONTROL:START -->'
  if [[ ! -f "$AGENTS_FILE" ]] || ! grep -Fq "$START_MARKER" "$AGENTS_FILE"; then
    cat >>"$AGENTS_FILE" <<EOF

<!-- DISCORD_REMOTE_CONTROL:START -->
## Discord Remote Control

- This project's Discord profile is \`$PROFILE_NAME\`.
- Treat input prefixed with \`[reply:discord]\` as a user request from Discord.
- Complete and verify the work, then mirror the concise final answer to Discord.
- Send text with real newlines through standard input:
  \`printf '%s\\n' 'RESPONSE' | "$CONTROL_SCRIPT" send --stdin\`
- Add one artifact path after \`--stdin\` when sending a plot, table, report, or archive.
- Use the helper for every outbound message so Discord replies route to this tmux pane.
- Send only the polished answer, never command logs, hidden reasoning, credentials, or tokens.
- Report long-job starts, meaningful checkpoints, failures, completion, and ETA changes over 15%.
<!-- DISCORD_REMOTE_CONTROL:END -->
EOF
  fi
fi

if ((LAUNCH_AGENT)); then
  "$CONTROL_SCRIPT" start
else
  echo "Configuration complete without launch. Start later with:"
  echo "  $CONTROL_SCRIPT start"
fi

cat <<EOF

Installation complete.
Control command:
  $CONTROL_SCRIPT status
  $CONTROL_SCRIPT verify
  $CONTROL_SCRIPT restart-listener

In Discord, reply to the exact bot smoke-test message with: 확인
General channel messages are intentionally ignored.
EOF
