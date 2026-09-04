#!/usr/bin/env node

import { execFileSync, spawn } from "node:child_process";
import {
  appendFile,
  chmod,
  mkdir,
  readFile,
  rename,
  unlink,
  writeFile,
} from "node:fs/promises";
import { realpathSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const action = process.argv[2] || "status";
const scriptPath = fileURLToPath(import.meta.url);
const stateDir = join(homedir(), ".omx", "state");
const pidPath = join(stateDir, "multi-channel-reply-listener.pid");
const statePath = join(
  stateDir,
  "multi-channel-reply-listener-state.json",
);
const logPath = join(stateDir, "multi-channel-reply-listener.log");
const noisyAcknowledgementPrefix = "Injected into Codex CLI session.";
const profilePanes = new Map(
  (process.env.OMX_REPLY_PROFILE_PANES || "")
    .split(",")
    .map((entry) => entry.split(":"))
    .filter(([profile, pane]) => profile && /^%\d+$/.test(pane)),
);

async function discordFetchWithoutTranscriptAcknowledgement(input, init = {}) {
  if (init.method === "POST" && typeof init.body === "string") {
    try {
      const payload = JSON.parse(init.body);
      if (payload?.content?.startsWith(noisyAcknowledgementPrefix)) {
        return new Response(null, { status: 204 });
      }
    } catch {
      // Non-JSON Discord requests are forwarded unchanged.
    }
  }
  return fetch(input, init);
}

function notificationModuleUrl(filename) {
  let moduleRoot = process.env.OMX_NOTIFICATIONS_MODULE_DIR || "";
  if (!moduleRoot) {
    try {
      const omxExecutable = realpathSync(
        process.env.OMX_EXECUTABLE ||
          execFileSync("which", ["omx"], { encoding: "utf8" }).trim(),
      );
      moduleRoot = resolve(dirname(omxExecutable), "..", "notifications");
    } catch {
      moduleRoot = join(
        homedir(),
        ".local",
        "lib",
        "node_modules",
        "oh-my-codex",
        "dist",
        "notifications",
      );
    }
  }
  return pathToFileURL(join(moduleRoot, filename)).href;
}

async function readJson(path, fallback = null) {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return fallback;
    throw error;
  }
}

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  const temporaryPath = `${path}.tmp.${process.pid}`;
  await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, {
    mode: 0o600,
  });
  await rename(temporaryPath, path);
  await chmod(path, 0o600);
}

async function appendLog(message) {
  const timestamp = new Date().toISOString();
  await mkdir(dirname(logPath), { recursive: true, mode: 0o700 });
  await appendFile(logPath, `[${timestamp}] ${message}\n`, { mode: 0o600 });
}

function pidAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function currentPid() {
  const raw = await readFile(pidPath, "utf8").catch(() => "");
  const pid = Number.parseInt(raw.trim(), 10);
  return pidAlive(pid) ? pid : null;
}

async function loadTargets() {
  const configModule = await import(notificationModuleUrl("config.js"));
  const listenerModule = await import(
    notificationModuleUrl("reply-listener.js")
  );
  const tmuxModule = await import(notificationModuleUrl("tmux-detector.js"));
  const reply = configModule.getReplyConfig();
  if (!reply?.enabled) {
    throw new Error("notifications.reply.enabled is not true");
  }
  const requestedProfiles = (process.env.OMX_REPLY_PROFILES || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  const profileNames = requestedProfiles.length
    ? requestedProfiles
    : configModule.listProfiles();
  if (!profileNames.length) {
    throw new Error("no notification profiles configured");
  }

  const targets = profileNames.map((profileName) => {
    const profile = configModule.getNotificationConfig(profileName);
    const discord = profile?.["discord-bot"];
    if (!profile?.enabled || !discord?.enabled) {
      throw new Error(`profile ${profileName} has no enabled Discord bot`);
    }
    if (!discord.botToken || !discord.channelId) {
      throw new Error(`profile ${profileName} is missing token or channel ID`);
    }
    return {
      profileName,
      channelId: discord.channelId,
      config: listenerModule.normalizeReplyListenerConfig({
        ...reply,
        discordEnabled: true,
        discordBotToken: discord.botToken,
        discordChannelId: discord.channelId,
        discordMention: discord.mention,
      }),
    };
  });
  if (new Set(targets.map((target) => target.channelId)).size !== targets.length) {
    throw new Error("notification profiles must use distinct Discord channels");
  }
  return { listenerModule, tmuxModule, reply, targets };
}

function paneRunsCodex(paneId) {
  if (!/^%\d+$/.test(paneId)) return false;
  try {
    const panePid = Number.parseInt(
      execFileSync(
        "tmux",
        ["display-message", "-p", "-t", paneId, "#{pane_pid}"],
        { encoding: "utf8" },
      ).trim(),
      10,
    );
    const processes = execFileSync("ps", ["-eo", "pid=,ppid=,args="], {
      encoding: "utf8",
    })
      .split("\n")
      .map((line) => line.match(/^\s*(\d+)\s+(\d+)\s+(.*)$/))
      .filter(Boolean)
      .map((match) => ({
        pid: Number.parseInt(match[1], 10),
        ppid: Number.parseInt(match[2], 10),
        args: match[3],
      }));
    const descendants = [panePid];
    for (let index = 0; index < descendants.length; index += 1) {
      for (const process of processes) {
        if (process.ppid === descendants[index]) descendants.push(process.pid);
      }
    }
    return processes.some(
      (process) =>
        descendants.includes(process.pid) &&
        /(?:^|[\/\s])codex(?:\s|$)/i.test(process.args),
    );
  } catch {
    return false;
  }
}

async function latestMessageId(target) {
  const response = await fetch(
    `https://discord.com/api/v10/channels/${target.channelId}/messages?limit=1`,
    {
      headers: { Authorization: `Bot ${target.config.discordBotToken}` },
      signal: AbortSignal.timeout(10000),
    },
  );
  if (!response.ok) {
    throw new Error(
      `profile ${target.profileName} Discord HTTP ${response.status}`,
    );
  }
  const messages = await response.json();
  return Array.isArray(messages) && messages.length ? messages[0].id : null;
}

async function run() {
  const existingPid = await currentPid();
  if (existingPid && existingPid !== process.pid) {
    throw new Error(`listener already running as PID ${existingPid}`);
  }
  await writeFile(pidPath, `${process.pid}\n`, { mode: 0o600 });

  const { listenerModule, tmuxModule, reply, targets } = await loadTargets();
  const previous = (await readJson(statePath, {})) || {};
  const state = {
    isRunning: true,
    pid: process.pid,
    startedAt: new Date().toISOString(),
    lastPollAt: null,
    profiles: {},
  };
  const limiters = new Map();

  for (const target of targets) {
    const old = previous.profiles?.[target.profileName];
    const sameChannel = old?.channelId === target.channelId;
    state.profiles[target.profileName] = {
      isRunning: true,
      pid: process.pid,
      startedAt: state.startedAt,
      lastPollAt: null,
      telegramLastUpdateId: null,
      discordLastMessageId: sameChannel
        ? old.discordLastMessageId
        : await latestMessageId(target),
      messagesInjected: sameChannel ? old.messagesInjected || 0 : 0,
      errors: sameChannel ? old.errors || 0 : 0,
      channelId: target.channelId,
    };
    limiters.set(
      target.profileName,
      new listenerModule.RateLimiter(reply.rateLimitPerMinute),
    );
  }

  let persistenceQueue = Promise.resolve();
  const persist = () => {
    persistenceQueue = persistenceQueue
      .then(() => writeJson(statePath, state))
      .catch((error) =>
        appendLog(
          `state write error: ${error instanceof Error ? error.message : String(error)}`,
        ),
      );
    return persistenceQueue;
  };
  const shutdown = async (signal) => {
    state.isRunning = false;
    state.stoppedAt = new Date().toISOString();
    state.stopSignal = signal;
    for (const channelState of Object.values(state.profiles)) {
      channelState.isRunning = false;
    }
    await persist().catch(() => {});
    await unlink(pidPath).catch(() => {});
    process.exit(0);
  };
  process.on("SIGTERM", () => void shutdown("SIGTERM"));
  process.on("SIGINT", () => void shutdown("SIGINT"));

  await appendLog(
    `started profiles=${targets.map((target) => target.profileName).join(",")}`,
  );
  await persist();

  while (state.isRunning) {
    state.lastPollAt = new Date().toISOString();
    for (const target of targets) {
      const channelState = state.profiles[target.profileName];
      channelState.lastPollAt = state.lastPollAt;
      try {
        await listenerModule.pollDiscordOnce(
          target.config,
          channelState,
          limiters.get(target.profileName),
          {
            fetchImpl: discordFetchWithoutTranscriptAcknowledgement,
            injectReplyImpl: (paneId, text, platform, config) => {
              let destinationPane = paneId;
              if (!paneRunsCodex(destinationPane)) {
                const fallbackPane = profilePanes.get(target.profileName);
                if (fallbackPane && paneRunsCodex(fallbackPane)) {
                  void appendLog(
                    `[${target.profileName}] pane ${paneId} has no Codex process; rerouted to ${fallbackPane}`,
                  );
                  destinationPane = fallbackPane;
                } else {
                  void appendLog(
                    `[${target.profileName}] pane ${paneId} has no Codex process; reply skipped`,
                  );
                  return false;
                }
              }
              if (!paneRunsCodex(destinationPane)) {
                void appendLog(
                  `[${target.profileName}] pane ${destinationPane} lost its Codex process; reply skipped`,
                );
                return false;
              }
              const prefix = config.includePrefix ? `[reply:${platform}] ` : "";
              const deliveryReminder =
                `\n[discord-delivery] Mirror the final answer with profile ${target.profileName} using the project helper.`;
              const sanitized = listenerModule.sanitizeReplyInput(
                prefix + text + deliveryReminder,
              );
              return tmuxModule.sendToPane(
                destinationPane,
                sanitized.slice(0, config.maxMessageLength),
                true,
              );
            },
            writeDaemonStateImpl: () => void persist(),
            logImpl: (message) =>
              void appendLog(`[${target.profileName}] ${message}`),
          },
        );
      } catch (error) {
        channelState.errors += 1;
        channelState.lastError =
          error instanceof Error ? error.message : String(error);
        await appendLog(
          `[${target.profileName}] poll error: ${channelState.lastError}`,
        );
      }
    }
    await persist();
    await new Promise((resolvePromise) =>
      setTimeout(resolvePromise, reply.pollIntervalMs),
    );
  }
}

async function start() {
  const existingPid = await currentPid();
  if (existingPid) {
    return { success: true, message: "already running", pid: existingPid };
  }
  const nativeListener = await import(
    notificationModuleUrl("reply-listener.js")
  );
  if (nativeListener.isDaemonRunning()) {
    nativeListener.stopReplyListener();
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 1000));
  }
  const child = spawn(process.execPath, [scriptPath, "run"], {
    detached: true,
    stdio: "ignore",
    env: {
      HOME: process.env.HOME,
      PATH: process.env.PATH,
      OMX_NOTIFICATIONS_MODULE_DIR:
        process.env.OMX_NOTIFICATIONS_MODULE_DIR || "",
      OMX_REPLY_PROFILES: process.env.OMX_REPLY_PROFILES || "",
      OMX_REPLY_PROFILE_PANES:
        process.env.OMX_REPLY_PROFILE_PANES || "",
    },
  });
  child.unref();
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 1500));
  const pid = await currentPid();
  if (!pid) throw new Error("multi-channel listener failed to start");
  return { success: true, message: "started", pid };
}

async function stop() {
  const pid = await currentPid();
  if (!pid) return { success: true, message: "not running", pid: null };
  process.kill(pid, "SIGTERM");
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
    if (!pidAlive(pid)) break;
  }
  return { success: true, message: "stopped", pid };
}

async function status() {
  const state = await readJson(statePath, null);
  const pid = await currentPid();
  return {
    success: true,
    running: Boolean(pid),
    pid,
    state,
  };
}

async function main() {
  let result;
  if (action === "run") {
    await run();
    return;
  }
  if (action === "start") result = await start();
  else if (action === "stop") result = await stop();
  else if (action === "restart") {
    await stop();
    result = await start();
  } else if (action === "status") result = await status();
  else {
    throw new Error(
      "usage: multi_channel_reply_listener.mjs start|stop|restart|status",
    );
  }
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

try {
  await main();
} catch (error) {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  await appendLog(`fatal: ${message}`).catch(() => {});
  if (action === "run") await unlink(pidPath).catch(() => {});
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
}
