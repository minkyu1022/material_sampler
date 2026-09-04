#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

bash -n setup_discord_remote_agent.sh
bash -n discord-control.sh
for file in \
  configure_omx_discord.mjs \
  multi_channel_reply_listener.mjs \
  send_discord_status.mjs; do
  node --check "$file"
done

tmp_project="$(mktemp -d)"
trap 'rm -rf "$tmp_project"' EXIT
bash setup_discord_remote_agent.sh \
  --project "$tmp_project" \
  --profile dry-run-profile \
  --channel-id 123456789012345678 \
  --user-id 223456789012345678 \
  --tmux-session dry-run-session \
  --dry-run >/dev/null

mock_home="$tmp_project/home"
mock_bin="$mock_home/.local/bin"
mock_tools="$tmp_project/project/.agent-tools/discord"
mkdir -p "$mock_bin" "$mock_tools"
install -m 700 discord-control.sh "$mock_tools/discord-control.sh"
cat >"$mock_tools/profile.env" <<EOF
PROFILE_NAME=dry-run-profile
PROJECT_ROOT=$tmp_project/project
TMUX_SESSION=dry-run-session
EOF
cat >"$mock_bin/tmux" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  list-panes)
    printf '%s\n' '%42'
    ;;
  has-session)
    exit 0
    ;;
  capture-pane)
    printf '%s\n' 'mock capture'
    ;;
esac
EOF
cat >"$mock_bin/node" <<'EOF'
#!/usr/bin/env bash
printf 'node'
printf ' %q' "$@"
printf '\n'
EOF
chmod 700 "$mock_bin/tmux" "$mock_bin/node"

control_output="$(
  HOME="$mock_home" PATH="$mock_bin:$PATH" \
    "$mock_tools/discord-control.sh" send 'routing test'
)"
grep -Fq -- '--profile dry-run-profile routing\ test' <<<"$control_output"

if [[ -f SHA256SUMS ]]; then
  sha256sum -c SHA256SUMS >/dev/null
fi

grep -Fq '(process.env.OMX_REPLY_PROFILE_PANES || "")' \
  multi_channel_reply_listener.mjs

if grep -RInE \
  --exclude='SHA256SUMS' \
  --exclude='verify_bundle.sh' \
  'DISCORD_BOT_TOKEN=[A-Za-z0-9_-]{16}' .; then
  echo "Bundle contains a forbidden server-specific fallback or token assignment." >&2
  exit 1
fi

echo "Portable Discord bootstrap checks passed."
