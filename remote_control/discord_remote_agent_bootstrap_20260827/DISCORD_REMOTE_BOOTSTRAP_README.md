# Portable Discord Remote Codex Bootstrap

This directory is task-agnostic. It connects one project-local Codex/OMX tmux
session to one private Discord channel. A different project can use the same
files with a different profile, channel ID, and tmux session.

## 1. Discord preparation

In the [Discord Developer Portal](https://discord.com/developers/applications):

1. Create an application and bot.
2. Enable **Message Content Intent** under **Bot > Privileged Gateway Intents**.
3. Invite the bot to a private channel with only these permissions:
   **View Channels**, **Send Messages**, **Read Message History**,
   **Add Reactions**, and **Attach Files**. Do not grant Administrator.
4. Enable Discord Developer Mode, then copy the private channel ID and the
   authorized user's numeric user ID.
5. Keep the bot token private. Do not put it in Discord, source files, command
   arguments, Git, issue trackers, or this archive.

## 2. One-command server setup

Unpack the archive on the target server and run:

```bash
bash setup_discord_remote_agent.sh \
  --project /absolute/path/to/project \
  --profile project-short-name \
  --channel-id YOUR_DISCORD_CHANNEL_ID \
  --user-id YOUR_DISCORD_USER_ID \
  --tmux-session codex-project-short-name
```

The script asks for the bot token through hidden terminal input. For an
automated setup, place the token in the process environment as
`DISCORD_BOT_TOKEN`; never pass it as a command-line argument.

The bootstrap performs all of the following:

- checks Node.js 20+ and tmux;
- installs tested Codex CLI `0.144.4` and OMX `0.20.3` under `~/.local` only
  when they are missing;
- checks Codex login and runs `omx setup` plus `omx doctor`;
- installs the project-local helpers under `.agent-tools/discord`;
- creates a quiet Discord profile so terminal transcript and `Session Idle`
  noise are not posted;
- restricts inbound messages to the supplied numeric user ID;
- patches the project `AGENTS.md` with the response-mirroring contract;
- starts the agent in tmux, starts the reply listener, and posts a routed smoke
  message.

The final end-to-end step is manual: reply to that exact bot message with
`확인`. A normal channel message is intentionally ignored.

## 3. Daily operation

The installed project gets this control command:

```bash
CONTROL=/absolute/path/to/project/.agent-tools/discord/discord-control.sh

$CONTROL status
$CONTROL verify
$CONTROL restart-listener
$CONTROL capture
printf '%s\n' 'manual status message' | $CONTROL send --stdin
```

After a server reboot, run `$CONTROL start`. Do not start a second listener for
the same Discord channel.

## 4. What keeps working after disconnect

SSH, VS Code, and a client-side VPN may be closed when all of these remain true:

- the Codex process is alive in server-side tmux;
- the detached listener is alive;
- the server itself has outbound HTTPS access to Discord;
- the server is not suspended or rebooted.

If Discord access depends on an SSH/VPN tunnel running on the client, closing
that tunnel also closes remote control.

## 5. Security boundary

Discord replies become input to a live coding agent. Use a private channel,
one authorized numeric user ID, and a dedicated channel per project/server.
The connection does not disable Codex sandbox or approval policies. Rotate the
bot token immediately if exposed.

The token is stored by OMX in `$HOME/.codex/.omx-config.json`, which the setup
forces to mode `0600`. Project-local `profile.env` contains routing identifiers
but no token.

## 6. Handoff to another agent

Give the agent this archive and the following instruction:

```text
Read DISCORD_REMOTE_BOOTSTRAP_README.md. Run verify_bundle.sh first, then run
setup_discord_remote_agent.sh for the target project. Do not request or print
the bot token in chat; have the user enter it in the hidden terminal prompt.
Finish the exact-message Discord Reply smoke test before claiming completion.
```

See `GENERIC_SETUP_GUIDE.md` for bot creation, multi-channel operation,
troubleshooting, token rotation, and the full security checklist.
