#!/usr/bin/env node

import { execFile } from "node:child_process";
import {
  access,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { localAppServerBinaryPath } from "../lib/electron-dev-sidecar.mjs";
import { writeTerminalExternalBackend } from "./terminal-gate-fixture.mjs";

const execFileAsync = promisify(execFile);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "../..");
const cliBinaryName = process.platform === "win32" ? "lime.exe" : "lime";
const defaultCliBinaryPath = path.join(
  rootDir,
  "lime-rs",
  "target",
  "debug",
  cliBinaryName,
);
const prompt = "tui gate b prompt";
const queuePrompt = "queued follow-up for editing";
const completedText = "TUI_GATE_B_COMPLETED";
const scenarios = (
  process.env.LIME_TUI_GATE_B_SCENARIOS ||
  "complete,approval,user-input,interrupt,failure,queue-edit"
)
  .split(",")
  .map((scenario) => scenario.trim())
  .filter(Boolean);

async function main() {
  const cliBinaryPath = path.resolve(
    process.env.LIME_CLI_BIN || defaultCliBinaryPath,
  );
  const appServerBinaryPath = path.resolve(
    process.env.APP_SERVER_BIN ||
      localAppServerBinaryPath({ repoRoot: rootDir }),
  );
  await Promise.all([
    assertBinaryExists(cliBinaryPath, "lime"),
    assertBinaryExists(appServerBinaryPath, "app-server"),
  ]);

  const tempDir = await mkdtemp(path.join(tmpdir(), "tui-gate-b-"));
  try {
    const backendPath = path.join(tempDir, "tui-backend.mjs");
    const ledgerPath = path.join(tempDir, "tui-backend.jsonl");
    const permissionConfigPath = path.join(tempDir, "permission-profile.yaml");
    await writeFile(
      permissionConfigPath,
      [
        "default_permissions: named-fixture",
        "permissions:",
        "  named-fixture:",
        "    extends: ':workspace'",
        "    description: TUI Gate B named permission profile",
        "",
      ].join("\n"),
    );
    for (const scenario of scenarios) {
      await writeTerminalExternalBackend(backendPath, {
        completedText,
        command: "printf tui-gate-b",
        scenario,
      });

      await execFileAsync(
        process.env.CARGO || "cargo",
        [
          "test",
          "--manifest-path",
          path.join(rootDir, "lime-rs", "Cargo.toml"),
          "-p",
          "tui",
          "runtime::pty_tests::real_pty_restores_terminal_after_visible_turn_completion",
          "--",
          "--exact",
          "--nocapture",
        ],
        {
          cwd: rootDir,
          encoding: "utf8",
          env: {
            ...process.env,
            LIME_TEST_TUI_GATE_B: "1",
            LIME_TEST_TERMINAL_SCENARIO: scenario,
            LIME_TEST_CLI_BIN: cliBinaryPath,
            LIME_TEST_APP_SERVER_BIN: appServerBinaryPath,
            LIME_TEST_TERMINAL_BACKEND: backendPath,
            LIME_TEST_TERMINAL_LEDGER: ledgerPath,
            LIME_TEST_TERMINAL_CWD: tempDir,
            LIME_TEST_NODE_BIN: process.execPath,
            LIME_TEST_TERMINAL_PROMPT: prompt,
            LIME_TEST_TERMINAL_QUEUE_PROMPT: queuePrompt,
            LIME_TEST_TERMINAL_COMPLETED_TEXT: completedText,
            LIME_TEST_PERMISSION_CONFIG: permissionConfigPath,
            LIME_TEST_PERMISSION_PROFILE: "named-fixture",
          },
          maxBuffer: 2 * 1024 * 1024,
          timeout: 60_000,
          windowsHide: true,
        },
      );
    }

    const ledger = await readJsonLines(ledgerPath);
    const turnStarts = scenarios.map((scenario) => {
      const entry = ledger.find(
        (candidate) =>
          candidate?.kind === "turnStart" && candidate.scenario === scenario,
      );
      if (!entry)
        throw new Error(
          `external backend did not record TUI turnStart for ${scenario}`,
        );
      assertEqual(entry.inputText, prompt, `${scenario} backend input`);
      assertNonEmptyString(entry.threadId, `${scenario} canonical thread id`);
      assertNonEmptyString(entry.turnId, `${scenario} canonical turn id`);
      return entry;
    });
    const expectedSequences = {
      complete: "message.delta,item.started,item.completed,turn.completed",
      approval: "item.started,action.required",
      "user-input": "item.started,action.required",
      interrupt: "message.delta",
      failure: "runtime.error,turn.failed",
      "queue-edit": "message.delta",
    };
    for (const [index, scenario] of scenarios.entries()) {
      assertEqual(
        turnStarts[index].eventTypes.join(","),
        expectedSequences[scenario],
        `${scenario} runtime event sequence`,
      );
    }
    if (scenarios.includes("approval")) {
      const approvalResponse = ledger.find(
        (entry) =>
          entry?.kind === "actionRespond" && entry.scenario === "approval",
      );
      if (!approvalResponse)
        throw new Error("approval response did not reach App Server backend");
      assertEqual(
        approvalResponse.decision,
        "allow_once",
        "command approval decision",
      );
    }
    if (scenarios.includes("user-input")) {
      const userInputResponse = ledger.find(
        (entry) =>
          entry?.kind === "actionRespond" && entry.scenario === "user-input",
      );
      if (!userInputResponse)
        throw new Error(
          "request_user_input response did not reach App Server backend",
        );
    }
    if (scenarios.includes("interrupt")) {
      const interruptResponse = ledger.find(
        (entry) =>
          entry?.kind === "turnCancel" && entry.scenario === "interrupt",
      );
      if (!interruptResponse)
        throw new Error("interrupt did not reach App Server backend");
    }
    if (scenarios.includes("queue-edit")) {
      const queueEditTurnStart = turnStarts.find(
        (entry) => entry.scenario === "queue-edit",
      );
      const queueEditCancel = ledger.find(
        (entry) =>
          entry?.kind === "turnCancel" && entry.scenario === "queue-edit",
      );
      if (!queueEditCancel) {
        throw new Error(
          "queue-edit cleanup interrupt did not reach App Server backend",
        );
      }
      const runtimeEvents = await readRuntimeEvents(tempDir);
      const queueAdded = runtimeEvents.find(
        (event) =>
          event?.type === "queue.added" &&
          event.payload?.source === "thread/queue/add" &&
          event.payload?.content?.text === queuePrompt,
      );
      if (!queueAdded) {
        throw new Error(
          "thread/queue/add was not recorded in the canonical event log",
        );
      }
      const queuedSubmissionId = queueAdded.payload?.queuedSubmissionId;
      assertNonEmptyString(queuedSubmissionId, "queued submission id");
      assertEqual(
        queueAdded.threadId,
        queueEditTurnStart.threadId,
        "queue edit canonical thread identity",
      );
      const queueRemoved = runtimeEvents.find(
        (event) =>
          event?.type === "queue.removed" &&
          event.payload?.source === "thread/queue/delete" &&
          event.payload?.queuedSubmissionId === queuedSubmissionId,
      );
      if (!queueRemoved) {
        throw new Error(
          "thread/queue/delete was not recorded in the canonical event log",
        );
      }
      assertEqual(
        queueRemoved.threadId,
        queueAdded.threadId,
        "queue edit thread identity",
      );
      if (!(queueRemoved.sequence > queueAdded.sequence)) {
        throw new Error(
          `queue removal must follow queue addition: add=${queueAdded.sequence}, remove=${queueRemoved.sequence}`,
        );
      }
    }
    const turnStart = turnStarts[0];

    console.log(
      [
        "[smoke:tui-gate-b] ok",
        `cli=${cliBinaryPath}`,
        `appServer=${appServerBinaryPath}`,
        `thread=${turnStart.threadId}`,
        `turn=${turnStart.turnId}`,
        `events=${turnStart.eventTypes.join(",")}`,
        scenarios.includes("queue-edit") ? "queue-edit=ok" : null,
        "terminal=restored",
      ]
        .filter(Boolean)
        .join(" "),
    );
  } finally {
    if (process.env.LIME_KEEP_TUI_GATE_B_TMP !== "1") {
      await rm(tempDir, { recursive: true, force: true });
    } else {
      console.error(`[smoke:tui-gate-b] kept temp dir ${tempDir}`);
    }
  }
}

async function assertBinaryExists(targetPath, label) {
  try {
    await access(targetPath);
  } catch {
    throw new Error(
      `${label} binary not found: ${targetPath}\n` +
        '先构建：cargo build --manifest-path "lime-rs/Cargo.toml" -p cli -p app-server',
    );
  }
}

async function readJsonLines(filePath) {
  const content = await readFile(filePath, "utf8");
  return content
    .split(/\r?\n/u)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

async function readRuntimeEvents(tempDir) {
  const sessionsDir = path.join(
    tempDir,
    "data",
    "runtime",
    "events",
    "sessions",
  );
  const entries = await readdir(sessionsDir, { withFileTypes: true }).catch(
    () => [],
  );
  const eventFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => path.join(sessionsDir, entry.name));
  const eventGroups = await Promise.all(eventFiles.map(readJsonLines));
  return eventGroups.flat();
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(
      `unexpected ${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
    );
  }
}

function assertNonEmptyString(value, label) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`missing ${label}: ${JSON.stringify(value)}`);
  }
}

main().catch((error) => {
  console.error(
    `[smoke:tui-gate-b] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
