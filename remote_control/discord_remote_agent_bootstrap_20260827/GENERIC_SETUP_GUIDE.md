# Discord Remote Codex Agent: Generic Setup Guide

이 문서는 특정 사용자, 서버, 채널, tmux 이름, project 경로를 가정하지 않는
독립형 설치 가이드다. Agent가 이 파일과 같은 디렉터리의 세 helper script를 받은
뒤, 한 개 이상의 Codex/OMX tmux 세션을 Discord 채널과 양방향으로 연결하는 것을
목표로 한다.

검증 기준 버전:

- Linux server
- Node.js 20 이상
- Codex CLI 0.144.x
- oh-my-codex (OMX) 0.20.3
- Discord API v10

OMX 내부 notification API를 사용하므로 OMX를 업그레이드한 뒤에는 반드시 이
문서의 전체 smoke test를 다시 수행한다.

## 1. 완성되는 연결

```text
Codex/OMX tmux pane
  -> send_discord_status.mjs
  -> Discord bot message
  -> 사용자가 그 bot message에 Reply
  -> multi_channel_reply_listener.mjs
  -> 원래 tmux pane에 [reply:discord] 입력
  -> agent가 작업 후 정리된 답변과 artifact를 Discord로 전송
```

핵심 제약:

- 일반 채널 메시지는 명령으로 주입하지 않는다.
- helper로 보낸 bot message에 대한 Discord **Reply**만 처리한다.
- 허용된 Discord user ID의 Reply만 처리한다.
- Discord에는 tmux transcript가 아니라 agent가 작성한 정리된 답변만 보낸다.
- Webhook은 양방향 Reply routing을 제공하지 않으므로 사용하지 않는다.

## 2. 필요한 값과 파일

설정 전에 다음 값을 준비한다. 아래 문자열은 모두 placeholder이며 실제 값으로
교체해야 한다.

| 값 | 의미 | 예시 형식 |
|---|---|---|
| `<PROFILE_NAME>` | project/channel을 구분하는 짧은 이름 | `project-a` |
| `<CHANNEL_ID>` | 연결할 Discord text channel ID | 17-20자리 숫자 |
| `<AUTHORIZED_USER_ID>` | 원격 명령을 보낼 Discord 사용자 ID | 17-20자리 숫자 |
| `<TMUX_SESSION>` | Codex를 실행할 tmux session 이름 | `codex-project-a` |
| `<PROJECT_DIRECTORY>` | 대상 project 디렉터리 | agent가 실행될 repository |

필수 helper 파일:

```text
configure_omx_discord.mjs
multi_channel_reply_listener.mjs
send_discord_status.mjs
```

Bot token은 표에 적거나 문서에 저장하지 않는다. 설치 중 사용자 terminal에서
secret input으로 한 번만 입력한다.

## 3. Discord bot 생성

### 3.1 Application과 bot user 생성

1. [Discord Developer Portal](https://discord.com/developers/applications)에
   로그인한다.
2. **New Application**을 누르고 용도를 식별할 수 있는 이름을 입력한다.
3. 생성된 application의 **Bot** 메뉴로 이동한다.
4. Bot user가 아직 없으면 **Add Bot**을 누른다. 최근 UI에서 자동 생성되어 있으면
   이 단계는 생략한다.
5. **Reset Token** 또는 **View/Copy Token**을 사용해 bot token을 발급한다.
6. Token은 password와 동일하게 취급한다. Discord 메시지, issue, Markdown, Git,
   shell command argument에 붙여 넣지 않는다.

Token을 잃어버렸다면 기존 token을 추측하거나 복구하려 하지 말고 Developer
Portal에서 reset한다. Reset 즉시 이전 token은 무효가 된다.

### 3.2 Message Content Intent 활성화

**Bot > Privileged Gateway Intents**에서 **Message Content Intent**를 활성화한다.
Listener가 Reply 본문을 읽으려면 필요하다. Discord 문서상 이 intent는 Gateway뿐
아니라 message content를 반환하는 HTTP API에도 영향을 준다.

Guild Members Intent와 Presence Intent는 이 구성에 필요하지 않다.

### 3.3 Bot을 Discord server에 초대

Developer Portal의 **OAuth2 > URL Generator** 또는 **Installation** 화면에서:

1. Scope로 `bot`을 선택한다.
2. 다음 bot permission만 선택한다.
   - View Channels
   - Send Messages
   - Read Message History
   - Add Reactions
   - Attach Files
3. `Administrator`는 선택하지 않는다.
4. 생성된 URL을 열고 대상 Discord server를 선택해 authorize한다.

`application.commands` scope는 이 Reply 기반 구성에는 필요하지 않다.

### 3.4 Private channel과 권한 설정

Project마다 전용 private text channel을 권장한다.

1. Channel permission에서 `@everyone`의 View Channel을 거부한다.
2. 허용 사용자와 bot role에만 View Channel을 허용한다.
3. Bot에 위 5개 permission이 실제 channel override에서도 허용되는지 확인한다.

Discord channel은 사실상 원격 shell 입력 경계이므로 일반 대화 채널과 공유하지
않는다.

### 3.5 Channel ID와 User ID 확인

Discord desktop에서 **User Settings > Advanced > Developer Mode**를 켠다.

- Channel 우클릭 > **Copy Channel ID**
- 허용할 사용자 우클릭 > **Copy User ID**

Username이나 display name은 바뀔 수 있으므로 allowlist에는 숫자 User ID를 쓴다.

## 4. Server 준비

### 4.1 필수 command 확인

```bash
command -v node npm jq tmux codex omx
node --version
codex --version
omx --version
```

Node.js는 20 이상이어야 한다. `jq`는 상태 확인에만 쓰지만 설치를 권장한다.

### 4.2 Codex와 OMX 설치

이미 설치되어 있으면 버전 확인만 한다. 없다면:

```bash
npm install -g @openai/codex
npm install -g oh-my-codex@0.20.3
codex login
omx setup --scope user --merge-agents
omx doctor
```

Global npm directory에 쓰기 권한이 없다면 Node version manager를 사용하거나 npm
user prefix를 설정한다. `sudo npm install`로 root-owned config를 만들지 않는 편이
안전하다.

`codex login status`와 `omx doctor`가 성공해야 다음 단계로 간다. Doctor warning은
내용을 검토하되 failed check가 있으면 먼저 해결한다.

## 5. Helper 설치

Project root에서 실행한다.

```bash
cd <PROJECT_DIRECTORY>
export PROJECT_ROOT="$(pwd -P)"
export TOOLS_DIR="$PROJECT_ROOT/.agent-tools/discord"
export SOURCE_DIR="<DIRECTORY_CONTAINING_THE_THREE_MJS_FILES>"

install -d -m 700 "$TOOLS_DIR"
install -m 700 \
  "$SOURCE_DIR/configure_omx_discord.mjs" \
  "$SOURCE_DIR/multi_channel_reply_listener.mjs" \
  "$SOURCE_DIR/send_discord_status.mjs" \
  "$TOOLS_DIR/"

node --check "$TOOLS_DIR/configure_omx_discord.mjs"
node --check "$TOOLS_DIR/multi_channel_reply_listener.mjs"
node --check "$TOOLS_DIR/send_discord_status.mjs"
```

세 script를 임의로 일부만 복사하지 않는다. `send_discord_status.mjs`가 outbound
message와 tmux pane mapping을 등록하고, listener가 그 mapping을 사용한다.

## 6. OMX notification profile 설정

### 6.1 Shell 변수 준비

Placeholder를 실제 값으로 바꾼다.

```bash
export PROFILE_NAME="<PROFILE_NAME>"
export CHANNEL_ID="<CHANNEL_ID>"
export AUTHORIZED_USER_ID="<AUTHORIZED_USER_ID>"
export TMUX_SESSION="<TMUX_SESSION>"

export OMX_NOTIFY_PROFILE="$PROFILE_NAME"
export OMX_NOTIFY_DEFAULT_PROFILE="$PROFILE_NAME"
export DISCORD_CHANNEL_ID="$CHANNEL_ID"
export DISCORD_USER_ID="$AUTHORIZED_USER_ID"
```

Profile 이름은 영문자, 숫자, `.`, `_`, `-`만 사용한다.

### 6.2 Bot token을 안전하게 입력

Token을 command argument로 넘기지 않는다. 다음 입력은 shell history에 token을
남기지 않는다.

```bash
umask 077
read -rsp 'Discord bot token: ' DISCORD_BOT_TOKEN
printf '\n'
export DISCORD_BOT_TOKEN
```

같은 Unix account의 다른 process가 환경변수를 읽을 수 있는 위협 모델이라면,
설정 중 다른 untrusted process를 실행하지 말고 설정 직후 즉시 unset한다.

### 6.3 정리된 agent 답변만 보내는 quiet profile 생성

Lifecycle notification을 끄면 `Session Idle`, 최근 tmux output 같은 자동 메시지가
Discord에 나타나지 않는다. Agent가 helper로 명시적으로 보낸 답변만 전송된다.

```bash
export OMX_DISCORD_LIFECYCLE_EVENTS=0
node "$TOOLS_DIR/configure_omx_discord.mjs" configure-profile

unset DISCORD_BOT_TOKEN OMX_DISCORD_NOTIFIER_BOT_TOKEN
unset OMX_DISCORD_LIFECYCLE_EVENTS
chmod 600 "$HOME/.codex/.omx-config.json"
```

Session start/idle/stop 알림까지 원하는 경우에만
`OMX_DISCORD_LIFECYCLE_EVENTS=1`로 profile을 다시 설정한다.

설정 script는 기존 config를 timestamp가 붙은 backup으로 보존한다. Token은
`$HOME/.codex/.omx-config.json`에 저장되므로 이 파일은 반드시 `0600`이어야 한다.

### 6.4 Profile 검증

```bash
OMX_NOTIFY_PROFILE="$PROFILE_NAME" \
  node "$TOOLS_DIR/configure_omx_discord.mjs" verify

node "$TOOLS_DIR/configure_omx_discord.mjs" status | jq
stat -c '%a %n' "$HOME/.codex/.omx-config.json"
```

성공 조건:

- `verified: true`
- 올바른 profile과 channel ID
- `hasBotToken: true`
- config permission `600`
- 출력 어디에도 token 값이 없음

## 7. Codex agent를 tmux에서 시작

Profile은 Codex/OMX process가 시작되기 **전에** environment로 들어가야 한다.

```bash
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists; inspect it before reusing it"
else
  tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT" \
    "export OMX_NOTIFY_PROFILE='$PROFILE_NAME'; exec omx --direct"
fi

tmux list-panes -t "$TMUX_SESSION" \
  -F '#{session_name} #{pane_id} #{pane_current_command}'
```

기존 session에 무조건 `send-keys`로 launch command를 넣지 않는다. 이미 Codex가 실행
중일 수 있으며, 이 경우 command가 사용자 prompt로 잘못 주입될 수 있다.

외부에서 이미 tmux를 관리하므로 `omx --direct`를 사용해 OMX가 별도의 detached tmux를
다시 만드는 것을 막는다.

기존 Codex process가 이미 profile 없이 실행 중이면 실행 중인 process의 environment를
나중에 바꾸지 못한다. Session ID를 기록하고 정상 종료한 뒤 다음처럼 resume한다.

```bash
export OMX_NOTIFY_PROFILE="$PROFILE_NAME"
codex resume <CODEX_SESSION_ID>
```

Safety/approval 설정은 기존 Codex 정책을 유지한다. Discord 연결을 이유로 sandbox나
approval을 자동 해제하지 않는다.

## 8. Multi-channel Reply listener 시작

한 Unix account/server에서는 listener 하나만 실행한다. Profile이 하나여도 동일한
multi-channel listener를 사용하면 나중에 profile을 추가하기 쉽다.

```bash
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" restart
sleep 5
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" status | jq
```

성공 조건:

- `running: true`
- profile의 `channelId`가 올바름
- `lastPollAt`이 약 3초마다 갱신
- `errors`가 지속적으로 증가하지 않음

10초 간격으로 두 번 확인해 timestamp가 실제로 움직이는지 확인한다.

```bash
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" status | jq
sleep 10
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" status | jq
```

## 9. Project AGENTS.md contract

Reply가 pane에 들어오는 것과 agent 답변이 Discord로 돌아가는 것은 별개다. Project의
`AGENTS.md`에 아래 contract를 추가하고 placeholder를 실제 profile로 바꾼다.

```md
## Discord Remote Control

- This project's Discord notification profile is `<PROFILE_NAME>`.
- Treat input prefixed with `[reply:discord]` as a user request received from Discord.
- Complete and verify the requested work, then mirror the concise user-facing answer to Discord.
- Send text through standard input so real newlines are preserved:
  `printf '%s' "<RESPONSE>" | node "$PWD/.agent-tools/discord/send_discord_status.mjs" --profile <PROFILE_NAME> --stdin`
- Add one artifact path as the second positional argument when sending a plot, table, report, or archive.
- Always use the helper so each outbound bot message is registered to the current tmux pane.
- Send only the polished answer, not command logs, tmux transcript, hidden reasoning, or secrets.
- For long jobs, report start, material checkpoints, completion, failures, and ETA changes over 15%.
- Never print, request in Discord, or transmit the Discord bot token.
```

응답이 2,000자를 넘으면 Discord message를 억지로 잘라 보내지 말고 짧은 요약과 `.md`
attachment를 보낸다.

## 10. End-to-end routed smoke test

### 10.1 Bot message를 올바른 pane에 등록

현재 tmux pane 안에서 helper를 실행하면 `TMUX_PANE`을 자동 사용한다. 외부 shell에서
실행할 때는 pane을 명시한다.

```bash
PANE_ID="$(tmux list-panes -t "$TMUX_SESSION" -F '#{pane_id}' | head -n 1)"

OMX_RESULT_PROJECT="$PROJECT_ROOT" \
OMX_RESULT_TMUX_SESSION="$TMUX_SESSION" \
OMX_RESULT_TMUX_PANE="$PANE_ID" \
node "$TOOLS_DIR/send_discord_status.mjs" --profile "$PROFILE_NAME" \
  'Routing smoke test. Reply to this exact bot message with: 확인'
```

`success: true`와 Discord message ID가 출력되어야 한다.

### 10.2 Discord에서 Reply

Discord에서 방금 bot이 보낸 메시지의 **Reply** 기능을 사용해 `확인`을 보낸다.
채널에 새 일반 메시지를 작성하면 주입되지 않는 것이 정상이다.

### 10.3 Server에서 주입 확인

```bash
tmux capture-pane -pt "$TMUX_SESSION" -S -100
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" status | jq
tail -n 20 "$HOME/.omx/state/reply-session-registry.jsonl" | jq -c .
tail -n 100 "$HOME/.omx/state/multi-channel-reply-listener.log"
```

성공 조건:

1. 올바른 pane에 `[reply:discord] 확인`이 나타남
2. 해당 profile의 `messagesInjected`가 1 증가함
3. 다른 pane에는 입력이 나타나지 않음
4. Agent가 helper를 호출해 정리된 답변을 같은 Discord channel에 보냄

## 11. 일상적인 사용

### 11.1 Text 전송

```bash
node "$TOOLS_DIR/send_discord_status.mjs" --profile "$PROFILE_NAME" \
  '현재 단계 3/5, ETA 40분'
```

여러 줄 메시지는 `--stdin`으로 실제 줄바꿈을 전달한다. JSON-escaped 문자열을
positional argument로 넘기지 않는다. Helper도 literal backslash-n을 거부한다.

```bash
printf '%s' $'현재 상태\n\n- 단계 3/5\n- ETA 40분' | \
  node "$TOOLS_DIR/send_discord_status.mjs" --profile "$PROFILE_NAME" --stdin
```

### 11.2 Artifact 첨부

두 번째 positional argument로 파일 하나를 첨부한다.

```bash
printf '%s' '중간 결과 plot' | \
  node "$TOOLS_DIR/send_discord_status.mjs" --profile "$PROFILE_NAME" \
    --stdin results/curve.png

printf '%s' '최종 metric table' | \
  node "$TOOLS_DIR/send_discord_status.mjs" --profile "$PROFILE_NAME" \
    --stdin results/metrics.csv

printf '%s' '상세 보고서' | \
  node "$TOOLS_DIR/send_discord_status.mjs" --profile "$PROFILE_NAME" \
    --stdin reports/final_report.md
```

Helper는 PNG, JPG, CSV, JSON, Markdown, PDF, TXT를 적절한 content type으로 보낸다.
Discord upload size 제한보다 큰 파일은 압축하거나 외부 artifact storage를 사용한다.

### 11.3 Listener 운영

```bash
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" status | jq
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" restart
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" stop
```

## 12. 여러 project/channel 연결

한 bot token을 여러 private channel에 재사용할 수 있다. 각 project마다:

1. 고유 Discord channel을 만든다.
2. 고유 profile 이름을 정한다.
3. `configure-profile`을 새 profile/channel ID로 반복한다.
4. 각 Codex process를 해당 `OMX_NOTIFY_PROFILE`로 시작한다.
5. Listener 하나가 모든 profile을 poll하게 한다.

두 번째 profile은 기존 config의 token을 재사용할 수 있어 token을 다시 export할 필요가
없다.

```bash
export OMX_NOTIFY_PROFILE="<SECOND_PROFILE>"
export OMX_NOTIFY_DEFAULT_PROFILE="<DEFAULT_PROFILE>"
export DISCORD_CHANNEL_ID="<SECOND_CHANNEL_ID>"
export DISCORD_USER_ID="<AUTHORIZED_USER_ID>"
export OMX_DISCORD_LIFECYCLE_EVENTS=0

node "$TOOLS_DIR/configure_omx_discord.mjs" configure-profile
unset OMX_DISCORD_LIFECYCLE_EVENTS
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" restart
```

중요:

- Profile마다 서로 다른 channel ID를 사용한다.
- 같은 channel을 여러 server/listener가 동시에 poll하지 않는다.
- Outbound helper 호출에는 항상 정확한 `--profile`을 넣는다.
- 다른 pane 대신 message를 보낼 때는 `OMX_RESULT_PROJECT`,
  `OMX_RESULT_TMUX_SESSION`, `OMX_RESULT_TMUX_PANE`을 모두 지정한다.

## 13. SSH, VS Code, VPN 종료 후 동작

다음 조건이면 client의 SSH, VS Code, VPN을 끊어도 Discord 원격 대화가 계속된다.

- Codex가 server의 tmux 안에서 실행 중임
- Detached listener process가 server에서 실행 중임
- Server가 Discord API로 outbound HTTPS 연결 가능
- Server 자체가 절전/종료되지 않음

Client VPN이 단지 SSH 접속 경로라면 끊어도 된다. 반대로 server의 Discord 인터넷
연결이 client의 VPN/SSH tunnel에 의존한다면 끊으면 동작하지 않는다.

Server reboot 후에는 tmux agent와 listener를 다시 시작해야 한다. 장기 운영에서는
listener를 user-level systemd service로 관리하고, Codex tmux session의 재개 정책은
site 운영 규칙에 맞게 별도로 설정한다.

## 14. Token rotate와 연결 해제

### Token rotate

1. Discord Developer Portal > Bot > Reset Token
2. 새 token을 terminal의 `read -rsp`로 입력
3. 모든 profile에 `configure-profile` 재실행
4. Token unset
5. Listener restart와 smoke test 수행

### 연결 해제

```bash
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" stop
```

그 다음 Discord server에서 bot을 제거하거나 channel permission을 회수한다. Config에
남은 token까지 폐기하려면 Developer Portal에서 token을 reset한 뒤 사용하지 않는다.

## 15. Troubleshooting

### Bot message가 전송되지 않음

```bash
OMX_NOTIFY_PROFILE="$PROFILE_NAME" \
  node "$TOOLS_DIR/configure_omx_discord.mjs" verify
```

| 상태 | 일반적인 원인 |
|---|---|
| HTTP 401 | token 오류 또는 reset된 이전 token |
| HTTP 403 | bot의 server/channel permission 부족 |
| HTTP 404 | channel ID 오류 또는 bot이 channel을 볼 수 없음 |
| HTTP 429 | Discord rate limit; 전송 빈도를 낮추고 재시도 |

### Reply가 pane에 주입되지 않음

```bash
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" status | jq
tail -n 120 "$HOME/.omx/state/multi-channel-reply-listener.log"
tmux list-panes -a -F '#{session_name} #{pane_id} #{pane_current_command}'
tail -n 20 "$HOME/.omx/state/reply-session-registry.jsonl" | jq -c .
```

다음 순서로 확인한다.

1. 일반 메시지가 아니라 helper가 보낸 bot message에 Reply했는가?
2. Reply 대상 message ID가 registry에 있는가?
3. Reply 작성자의 숫자 User ID가 allowlist와 일치하는가?
4. Profile의 `lastPollAt`이 움직이는가?
5. Registry의 pane ID가 현재 살아 있고 Codex process를 실행하는가?
6. Message Content Intent가 활성화되어 있는가?
7. Bot에 View Channel과 Read Message History가 있는가?

### Reply는 들어오지만 Discord 답변이 없음

- `AGENTS.md`의 Discord contract가 실제 project에 적용됐는지 확인한다.
- Agent가 마지막에 `send_discord_status.mjs`를 호출했는지 확인한다.
- Listener는 입력 전달만 담당한다. Agent 답변 mirror는 helper가 담당한다.

### Discord에 Session Idle 또는 긴 terminal output이 표시됨

Quiet profile로 다시 설정한다.

```bash
export OMX_NOTIFY_PROFILE="$PROFILE_NAME"
export OMX_DISCORD_LIFECYCLE_EVENTS=0
node "$TOOLS_DIR/configure_omx_discord.mjs" configure-profile
unset OMX_DISCORD_LIFECYCLE_EVENTS
```

Agent contract에서도 command output이나 tmux transcript를 보내지 않도록 명시한다.

### 서로 다른 project의 메시지가 섞임

- 각 Codex process가 올바른 `OMX_NOTIFY_PROFILE`을 상속했는지 확인한다.
- 각 helper 호출의 `--profile`을 확인한다.
- Profile마다 고유 channel ID인지 확인한다.
- Registry의 message ID와 pane ID mapping을 확인한다.

### 중복 주입

같은 channel을 poll하는 listener가 둘 이상일 가능성이 높다. Unix account당
multi-channel listener 하나만 남기고, 다른 server에는 별도 Discord channel을 쓴다.

### Listener PID는 있으나 동작하지 않음

PID만 보지 말고 `lastPollAt`을 10초 간격으로 두 번 확인한다. 움직이지 않으면:

```bash
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" restart
sleep 5
node "$TOOLS_DIR/multi_channel_reply_listener.mjs" status | jq
```

## 16. Security checklist

- [ ] Private Discord channel을 사용한다.
- [ ] Bot에 Administrator permission을 주지 않았다.
- [ ] `authorizedDiscordUserIds`에는 필요한 사용자만 있다.
- [ ] Bot token을 Discord/chat/Git/log/process argument에 노출하지 않았다.
- [ ] `$HOME/.codex/.omx-config.json` permission이 `0600`이다.
- [ ] `$HOME/.omx/state`가 다른 사용자에게 쓰기 가능하지 않다.
- [ ] 일반 channel message는 명령으로 처리되지 않는다.
- [ ] Discord 연결이 Codex sandbox/approval 정책을 우회하지 않는다.
- [ ] Token 노출 시 즉시 reset하는 절차를 알고 있다.
- [ ] Server와 channel마다 listener ownership이 하나뿐이다.

## 17. Completion checklist

- [ ] Discord application과 bot user가 생성됐다.
- [ ] Message Content Intent가 활성화됐다.
- [ ] Bot이 최소 권한으로 private channel에 들어왔다.
- [ ] Channel ID와 authorized User ID를 확인했다.
- [ ] Codex login과 `omx doctor`가 성공했다.
- [ ] 세 helper script의 `node --check`가 성공했다.
- [ ] Quiet profile이 올바른 channel로 검증됐다.
- [ ] Codex가 profile을 상속한 tmux pane에서 실행된다.
- [ ] Listener의 `lastPollAt`이 계속 갱신된다.
- [ ] Routed smoke message의 Reply가 정확한 pane에 들어온다.
- [ ] Agent의 정리된 답변이 Discord로 돌아온다.
- [ ] Plot/table attachment 전송이 성공한다.
- [ ] SSH/VPN disconnect 후에도 server-side 연결이 유지된다.

## 18. 공식 참고 문서

- [Discord Bots overview](https://docs.discord.com/developers/bots/overview)
- [Discord OAuth2 and permissions](https://docs.discord.com/developers/platform/oauth2-and-permissions)
- [Discord Gateway intents and Message Content Intent](https://docs.discord.com/developers/events/gateway#message-content-intent)
- [Discord Message resource](https://docs.discord.com/developers/resources/message)
- [Discord Developer Mode and ID copying](https://support.discord.com/hc/en-us/articles/206346498-Where-can-I-find-my-User-Server-Message-ID)
- [OpenAI Codex repository](https://github.com/openai/codex)
