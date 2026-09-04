# Discord Remote Codex Agent

가장 빠른 설치 경로는
[DISCORD_REMOTE_BOOTSTRAP_README.md](DISCORD_REMOTE_BOOTSTRAP_README.md)와
`setup_discord_remote_agent.sh`다. Bot 생성, multi-channel, 보안 및 장애 대응까지
포함한 전체 문서는 [GENERIC_SETUP_GUIDE.md](GENERIC_SETUP_GUIDE.md)다.

공유할 파일:

```text
GENERIC_SETUP_GUIDE.md
DISCORD_REMOTE_BOOTSTRAP_README.md
setup_discord_remote_agent.sh
discord-control.sh
verify_bundle.sh
configure_omx_discord.mjs
multi_channel_reply_listener.mjs
send_discord_status.mjs
```

먼저 `bash verify_bundle.sh`를 실행한 다음 bootstrap에 대상 환경의 profile,
Discord channel/user ID, tmux session, project directory를 넘긴다. Bot token은
문서나 채팅으로 전달하지 않고 대상 server terminal의 secret input으로 입력한다.
