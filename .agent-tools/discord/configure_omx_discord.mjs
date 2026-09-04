#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import {
  chmod,
  copyFile,
  mkdir,
  readFile,
  rename,
  writeFile,
} from "node:fs/promises";
import { realpathSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const action = process.argv[2] || "status";
const configPath = join(homedir(), ".codex", ".omx-config.json");

function requireProfileName(value) {
  if (!/^[A-Za-z0-9._-]+$/.test(value || "")) {
    throw new Error(
      "OMX_NOTIFY_PROFILE must contain only letters, digits, dot, underscore, or hyphen",
    );
  }
  return value;
}

function requireSnowflake(name, value) {
  if (!/^\d{17,20}$/.test(value || "")) {
    throw new Error(`${name} must be a 17-20 digit Discord ID`);
  }
  return value;
}

function notificationModuleUrl(filename) {
  let moduleRoot = process.env.OMX_NOTIFICATIONS_MODULE_DIR || "";
  if (!moduleRoot) {
    try {
      const omxExecutable = execFileSync("which", ["omx"], {
        encoding: "utf8",
      }).trim();
      const omxEntrypoint = realpathSync(omxExecutable);
      moduleRoot = resolve(dirname(omxEntrypoint), "..", "notifications");
    } catch {
      const npmRoot = execFileSync("npm", ["root", "-g"], {
        encoding: "utf8",
      }).trim();
      moduleRoot = join(
        npmRoot,
        "oh-my-codex",
        "dist",
        "notifications",
      );
    }
  }
  return pathToFileURL(
    join(moduleRoot, filename),
  ).href;
}

async function readJson(path, fallback = {}) {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return fallback;
    throw error;
  }
}

function envFlag(name, fallback) {
  const value = process.env[name];
  if (value === undefined || value === "") return fallback;
  if (["1", "true", "yes", "on"].includes(value.toLowerCase())) return true;
  if (["0", "false", "no", "off"].includes(value.toLowerCase())) return false;
  throw new Error(`${name} must be true/false, yes/no, on/off, or 1/0`);
}

function discordEvents(enabled = true) {
  return Object.fromEntries(
    [
      "session-start",
      "session-idle",
      "ask-user-question",
      "session-stop",
      "session-end",
    ].map((event) => [event, { enabled }]),
  );
}

function findStoredBotToken(root, profileName) {
  return (
    root.notifications?.profiles?.[profileName]?.["discord-bot"]?.botToken ||
    root.notifications?.["discord-bot"]?.botToken ||
    Object.values(root.notifications?.profiles || {}).find(
      (profile) => profile?.["discord-bot"]?.botToken,
    )?.["discord-bot"]?.botToken ||
    ""
  );
}

async function writeConfig(root) {
  await mkdir(dirname(configPath), { recursive: true, mode: 0o700 });
  try {
    const stamp = new Date().toISOString().replaceAll(":", "-");
    await copyFile(configPath, `${configPath}.bak.${stamp}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const temporaryPath = `${configPath}.tmp.${process.pid}`;
  await writeFile(temporaryPath, `${JSON.stringify(root, null, 2)}\n`, {
    mode: 0o600,
  });
  await rename(temporaryPath, configPath);
  await chmod(configPath, 0o600);
}

async function configure() {
  const botToken =
    process.env.DISCORD_BOT_TOKEN ||
    process.env.OMX_DISCORD_NOTIFIER_BOT_TOKEN;
  if (!botToken) {
    throw new Error(
      "set DISCORD_BOT_TOKEN or OMX_DISCORD_NOTIFIER_BOT_TOKEN",
    );
  }
  const channelId = requireSnowflake(
    "DISCORD_CHANNEL_ID",
    process.env.DISCORD_CHANNEL_ID ||
      process.env.OMX_DISCORD_NOTIFIER_CHANNEL,
  );
  const userId = requireSnowflake(
    "DISCORD_USER_ID",
    process.env.DISCORD_USER_ID,
  );
  const idleCooldownSeconds = Number.parseInt(
    process.env.OMX_DISCORD_IDLE_COOLDOWN_SECONDS || "60",
    10,
  );
  const lifecycleEventsEnabled = envFlag(
    "OMX_DISCORD_LIFECYCLE_EVENTS",
    true,
  );

  const root = await readJson(configPath);
  root.notifications ||= {};
  root.notifications.enabled = true;
  root.notifications.verbosity = "session";
  root.notifications.idleCooldownSeconds = Number.isFinite(idleCooldownSeconds)
    ? Math.max(0, idleCooldownSeconds)
    : 60;
  root.notifications["discord-bot"] = {
    enabled: true,
    botToken,
    channelId,
    mention: `<@${userId}>`,
  };
  root.notifications.reply = {
    enabled: true,
    pollIntervalMs: 3000,
    maxMessageLength: 2000,
    rateLimitPerMinute: 10,
    includePrefix: true,
    authorizedDiscordUserIds: [userId],
    acknowledgeReplies: false,
  };
  root.notifications.events ||= {};
  for (const [event, setting] of Object.entries(
    discordEvents(lifecycleEventsEnabled),
  )) {
    root.notifications.events[event] = {
      ...(root.notifications.events[event] || {}),
      ...setting,
    };
  }

  await writeConfig(root);

  process.stdout.write(
    `${JSON.stringify({
      configured: true,
      configPath,
      channelId,
      authorizedDiscordUserIds: [userId],
      botTokenStored: true,
      lifecycleEventsEnabled,
    })}\n`,
  );
}

async function configureProfile() {
  const profileName = requireProfileName(process.env.OMX_NOTIFY_PROFILE);
  const channelId = requireSnowflake(
    "DISCORD_CHANNEL_ID",
    process.env.DISCORD_CHANNEL_ID ||
      process.env.OMX_DISCORD_NOTIFIER_CHANNEL,
  );
  const userId = requireSnowflake(
    "DISCORD_USER_ID",
    process.env.DISCORD_USER_ID,
  );
  const root = await readJson(configPath);
  const botToken =
    process.env.DISCORD_BOT_TOKEN ||
    process.env.OMX_DISCORD_NOTIFIER_BOT_TOKEN ||
    findStoredBotToken(root, profileName);
  if (!botToken) {
    throw new Error(
      "no stored bot token found; set DISCORD_BOT_TOKEN through secret input",
    );
  }
  const requestedIdleCooldown = Number.parseInt(
    process.env.OMX_DISCORD_IDLE_COOLDOWN_SECONDS || "0",
    10,
  );
  const lifecycleEventsEnabled = envFlag(
    "OMX_DISCORD_LIFECYCLE_EVENTS",
    true,
  );

  root.notifications ||= {};
  root.notifications.enabled = true;
  root.notifications.defaultProfile =
    process.env.OMX_NOTIFY_DEFAULT_PROFILE ||
    root.notifications.defaultProfile ||
    profileName;
  root.notifications.profiles ||= {};
  root.notifications.profiles[profileName] = {
    enabled: true,
    verbosity: "session",
    idleCooldownSeconds: Number.isFinite(requestedIdleCooldown)
      ? Math.max(0, requestedIdleCooldown)
      : 0,
    "discord-bot": {
      enabled: true,
      botToken,
      channelId,
      mention: `<@${userId}>`,
    },
    events: discordEvents(lifecycleEventsEnabled),
  };
  root.notifications.reply = {
    enabled: true,
    pollIntervalMs: 3000,
    maxMessageLength: 2000,
    rateLimitPerMinute: 10,
    includePrefix: true,
    authorizedDiscordUserIds: [userId],
    acknowledgeReplies: false,
  };

  // Once profiles exist, retaining a flat transport makes accidental fallback
  // silently target the old channel. The default profile is the safe fallback.
  delete root.notifications["discord-bot"];
  await writeConfig(root);

  process.stdout.write(
    `${JSON.stringify({
      configured: true,
      configPath,
      profileName,
      defaultProfile: root.notifications.defaultProfile,
      channelId,
      authorizedDiscordUserIds: [userId],
      botTokenStored: true,
      lifecycleEventsEnabled,
    })}\n`,
  );
}

async function listenerModules() {
  const config = await import(notificationModuleUrl("config.js"));
  const listener = await import(notificationModuleUrl("reply-listener.js"));
  return { config, listener };
}

async function start() {
  const { config, listener } = await listenerModules();
  const fullConfig = config.getNotificationConfig();
  const replyConfig = config.getReplyConfig();
  const platformConfig = config.getReplyListenerPlatformConfig(fullConfig);
  if (!replyConfig?.enabled) {
    throw new Error("notifications.reply.enabled is not true");
  }
  const current = listener.getReplyListenerStatus();
  if (current.state?.isRunning) {
    listener.stopReplyListener();
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  const result = listener.startReplyListener({
    ...replyConfig,
    ...platformConfig,
  });
  process.stdout.write(`${JSON.stringify(result)}\n`);
  if (!result.success) process.exitCode = 1;
}

async function stop() {
  const { listener } = await listenerModules();
  process.stdout.write(`${JSON.stringify(listener.stopReplyListener())}\n`);
}

async function status() {
  const { config, listener } = await listenerModules();
  const activeProfile = process.env.OMX_NOTIFY_PROFILE || null;
  const fullConfig = config.getNotificationConfig(activeProfile || undefined);
  const discord = fullConfig?.["discord-bot"];
  const reply = config.getReplyConfig();
  const listenerStatus = listener.getReplyListenerStatus();
  process.stdout.write(
    `${JSON.stringify({
      configPath,
      activeProfile,
      defaultProfile: (await readJson(configPath)).notifications
        ?.defaultProfile || null,
      profiles: config.listProfiles().map((profileName) => {
        const profile = config.getNotificationConfig(profileName);
        return {
          profileName,
          enabled: profile?.enabled === true,
          channelId: profile?.["discord-bot"]?.channelId || null,
          hasBotToken: Boolean(profile?.["discord-bot"]?.botToken),
        };
      }),
      notificationsEnabled: fullConfig?.enabled === true,
      discordBot: {
        enabled: discord?.enabled === true,
        channelId: discord?.channelId || null,
        hasBotToken: Boolean(discord?.botToken),
      },
      reply: reply
        ? {
            enabled: reply.enabled,
            pollIntervalMs: reply.pollIntervalMs,
            authorizedDiscordUserIds: reply.authorizedDiscordUserIds,
          }
        : null,
      listener: listenerStatus,
    })}\n`,
  );
}

async function verifyDiscord() {
  const { config } = await listenerModules();
  const profileName = process.env.OMX_NOTIFY_PROFILE || undefined;
  const discord = config.getNotificationConfig(profileName)?.["discord-bot"];
  if (!discord?.enabled || !discord.botToken || !discord.channelId) {
    throw new Error("Discord bot configuration is incomplete");
  }
  const headers = { Authorization: `Bot ${discord.botToken}` };
  const [botResponse, channelResponse] = await Promise.all([
    fetch("https://discord.com/api/v10/users/@me", { headers }),
    fetch(`https://discord.com/api/v10/channels/${discord.channelId}`, {
      headers,
    }),
  ]);
  if (!botResponse.ok || !channelResponse.ok) {
    throw new Error(
      `Discord verification failed: bot=${botResponse.status}, channel=${channelResponse.status}`,
    );
  }
  const bot = await botResponse.json();
  const channel = await channelResponse.json();
  process.stdout.write(
    `${JSON.stringify({
      verified: true,
      profileName: profileName || null,
      bot: `${bot.username}#${bot.discriminator}`,
      channelId: channel.id,
      channelName: channel.name || null,
    })}\n`,
  );
}

const actions = {
  configure,
  "configure-profile": configureProfile,
  start,
  restart: start,
  stop,
  status,
  verify: verifyDiscord,
};

if (!actions[action]) {
  throw new Error(
    "usage: configure_omx_discord.mjs configure|configure-profile|start|restart|stop|status|verify",
  );
}

await actions[action]();
