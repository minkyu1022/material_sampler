# Agent Runbook

자동 설치의 canonical entrypoint는 `setup_discord_remote_agent.sh`이고, 상세 설명은
[DISCORD_REMOTE_BOOTSTRAP_README.md](DISCORD_REMOTE_BOOTSTRAP_README.md)다. 수동 설치와
장애 대응의 canonical runbook은 [GENERIC_SETUP_GUIDE.md](GENERIC_SETUP_GUIDE.md)다.

Agent는 먼저 `bash verify_bundle.sh`를 통과시킨 뒤 다음 순서로 실행한다.

1. 가이드 2절의 placeholder 값을 사용자 환경에서 수집한다.
2. Bot token은 Discord나 채팅으로 요청하지 않고 terminal secret input을 사용한다.
3. 세 helper script를 설치하고 `node --check`로 검증한다.
4. Quiet OMX profile을 구성한다.
5. Profile을 상속한 Codex/OMX tmux session을 시작한다.
6. Multi-channel listener를 시작한다.
7. Project `AGENTS.md`에 Discord response-mirroring contract를 추가한다.
8. Routed Reply smoke test와 attachment test를 완료한다.
9. Token, 개인 ID, 절대 project 경로가 문서나 Git에 남지 않았는지 검사한다.

설정값이 명확하면 재확인을 요구하지 말고 진행한다. Bot 생성, token 발급, Discord
server authorize처럼 사용자 계정 권한이 필요한 단계만 사용자에게 수행 방법을
안내한다.
