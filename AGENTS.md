
<!-- DISCORD_REMOTE_CONTROL:START -->
## Discord Remote Control

- This project's Discord profile is `joint-sampler`.
- Treat input prefixed with `[reply:discord]` as a user request from Discord.
- Complete and verify the work, then mirror the concise final answer to Discord.
- Send text with real newlines through standard input:
  `printf '%s\n' 'RESPONSE' | "/home/spml_minkyu_kim/joint_sampler/.agent-tools/discord/discord-control.sh" send --stdin`
- Add one artifact path after `--stdin` when sending a plot, table, report, or archive.
- Use the helper for every outbound message so Discord replies route to this tmux pane.
- Send only the polished answer, never command logs, hidden reasoning, credentials, or tokens.
- Report long-job starts, meaningful checkpoints, failures, completion, and ETA changes over 15%.
<!-- DISCORD_REMOTE_CONTROL:END -->
