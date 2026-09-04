#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { realpathSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, dirname, extname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const readMessageFromStdin = args.includes("--stdin");
const profileIndex = args.indexOf("--profile");
const profileName =
  profileIndex >= 0
    ? args[profileIndex + 1]
    : process.env.OMX_NOTIFY_PROFILE || "";
if (profileIndex >= 0 && !profileName) {
  throw new Error("--profile requires a profile name");
}
const profileValueIndex = profileIndex >= 0 ? profileIndex + 1 : -1;
const values = args.filter(
  (_value, index) =>
    index !== profileIndex &&
    index !== profileValueIndex &&
    args[index] !== "--dry-run" &&
    args[index] !== "--stdin",
);

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8");
}

const message = readMessageFromStdin
  ? (await readStdin()).replace(/\n$/, "")
  : values[0] || "";
const attachment = readMessageFromStdin ? values[0] || "" : values[1] || "";
const projectPath = process.env.OMX_RESULT_PROJECT || process.cwd();
let tmuxPaneId = process.env.OMX_RESULT_TMUX_PANE || process.env.TMUX_PANE || "";
let tmuxSession = process.env.OMX_RESULT_TMUX_SESSION || "";

if (!message) {
  throw new Error(
    "usage: send_discord_status.mjs [--profile NAME] [--stdin] MESSAGE [ATTACHMENT] [--dry-run]",
  );
}
if (message.includes("\\n")) {
  throw new Error(
    "message contains a literal backslash-n sequence; use --stdin with real newlines",
  );
}

if (!tmuxPaneId && tmuxSession) {
  tmuxPaneId = execFileSync(
    "tmux",
    ["list-panes", "-t", tmuxSession, "-F", "#{pane_id}"],
    { encoding: "utf8" },
  )
    .trim()
    .split("\n")[0];
}
if (!tmuxPaneId) {
  throw new Error(
    "set OMX_RESULT_TMUX_PANE or OMX_RESULT_TMUX_SESSION so replies route to the correct agent",
  );
}
if (!tmuxSession) {
  tmuxSession = execFileSync(
    "tmux",
    ["display-message", "-p", "-t", tmuxPaneId, "#{session_name}"],
    { encoding: "utf8" },
  ).trim();
}

let sessionId = tmuxSession;
try {
  const session = JSON.parse(
    await readFile(join(projectPath, ".omx", "state", "session.json"), "utf8"),
  );
  sessionId = session.session_id || sessionId;
} catch {
  // The tmux pane remains a valid routing target without OMX session state.
}

if (dryRun) {
  process.stdout.write(
    `${JSON.stringify({ message, attachment, profileName: profileName || null, projectPath, tmuxPaneId, tmuxSession })}\n`,
  );
  process.exit(0);
}

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
const { getNotificationConfig } = await import(
  pathToFileURL(join(moduleRoot, "config.js")).href
);
const { registerMessage } = await import(
  pathToFileURL(join(moduleRoot, "session-registry.js")).href
);

const discord = getNotificationConfig(profileName || undefined)?.["discord-bot"];
if (!discord?.enabled || !discord.botToken || !discord.channelId) {
  throw new Error("Discord bot notifications are not configured");
}

const mention = discord.mention || "";
const mentionedUsers = [...mention.matchAll(/<@(\d+)>/g)].map(
  (match) => match[1],
);
const payload = {
  content: `${mention ? `${mention}\n` : ""}${message}`.slice(0, 2000),
  allowed_mentions: { parse: [], users: mentionedUsers, roles: [] },
};
const request = {
  method: "POST",
  headers: { Authorization: `Bot ${discord.botToken}` },
};

if (attachment) {
  const contentTypes = {
    ".csv": "text/csv",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".txt": "text/plain",
  };
  const data = await readFile(attachment);
  const form = new FormData();
  form.append("payload_json", JSON.stringify(payload));
  form.append(
    "files[0]",
    new Blob([data], {
      type: contentTypes[extname(attachment).toLowerCase()] ||
        "application/octet-stream",
    }),
    basename(attachment),
  );
  request.body = form;
} else {
  request.headers["Content-Type"] = "application/json";
  request.body = JSON.stringify(payload);
}

const response = await fetch(
  `https://discord.com/api/v10/channels/${discord.channelId}/messages`,
  request,
);
if (!response.ok) {
  throw new Error(`Discord HTTP ${response.status}: ${await response.text()}`);
}
const sent = await response.json();
const registered = registerMessage({
  platform: "discord-bot",
  messageId: sent.id,
  sessionId,
  tmuxPaneId,
  tmuxSessionName: tmuxSession,
  event: "session-idle",
  createdAt: new Date().toISOString(),
  projectPath,
});
if (!registered) {
  throw new Error("message sent, but reply routing registration failed");
}
process.stdout.write(
  `${JSON.stringify({ success: true, messageId: sent.id, profileName: profileName || null, tmuxPaneId })}\n`,
);
