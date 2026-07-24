#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import electronPath from "electron";
import { _electron as electron } from "playwright";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { withElectronFixtureSystemPath } from "../lib/electron-fixture-runtime-env.mjs";
import {
  FILE_CHANGE_BATCH_PROMPT_MARKER,
  FILE_CHANGE_BATCH_SCENARIO,
  clickFileChangeApprovalDecision,
  renderFileChangeGateBBackendScript,
  waitForFileChangeApprovalPending,
  waitForFileChangeTerminalGui,
  waitForFileChangeTerminalReadModel,
} from "./lib/code-artifact-file-change-gate-b.mjs";

const DEFAULTS = {
  appUrl: "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "code-artifact-workbench-electron-fixture",
  ),
  prefix: "code-artifact-workbench-electron-fixture",
  timeoutMs: 180_000,
  intervalMs: 500,
  keepTemp: false,
  scenario: "direct-session",
};

const LOG_PREFIX = "[smoke:code-artifact-workbench-electron-fixture]";
const TEMP_CLEANUP_RETRY_COUNT = 8;
const TEMP_CLEANUP_RETRY_DELAY_MS = 250;
const FINAL_PAGE_OPERATION_TIMEOUT_MS = 15_000;
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_METHOD_THREAD_START = "thread/start";
const APP_SERVER_METHOD_TURN_START = "turn/start";
const APP_SERVER_METHOD_THREAD_READ = "thread/read";
const APP_SERVER_METHOD_THREAD_LIST = "thread/list";
const APP_SERVER_METHOD_ARTIFACT_WRITE = "artifact/write";
const APP_SERVER_METHOD_WORKSPACE_DEFAULT_ENSURE = "workspace/default/ensure";
let SESSION_ID = `pending-code-artifact-workbench-${Date.now()}-${process.pid}`;
let THREAD_ID = "pending-code-artifact-workbench-thread";
let TURN_ID = "pending-code-artifact-workbench-turn";
const SESSION_TITLE = "代码产物工作台 Electron fixture";
const USER_PROMPT = "生成一个 TypeScript greeting 代码产物，并打开工作台验证。";
const GUI_CODING_PROMPT =
  "@代码 修复 coding-target.test.ts 中 codingWorkbenchSmoke 失败的问题，并补一个回归测试";
const FILE_CHANGE_BATCH_DECLINE_PROMPT = `${FILE_CHANGE_BATCH_PROMPT_MARKER} Decline.`;
const FILE_CHANGE_BATCH_CANCEL_PROMPT = `${FILE_CHANGE_BATCH_PROMPT_MARKER} Cancel.`;
const CODING_RECOVERY_PROMPT_INTRO = "请继续修复本轮编程任务中的失败输出。";
const ASSISTANT_ARTIFACT_TEXT = "已生成代码产物，可在工作台查看。";
const FINAL_DONE_TEXT = "CODE_ARTIFACT_WORKBENCH_DONE";
const ARTIFACT_ID = "code-artifact-workbench-electron:greeting";
const ARTIFACT_PATH =
  ".lime/qc/code-artifact-workbench-electron-fixture/src/greeting.ts";
const TOOL_CALL_ID = "code-artifact-workbench-electron:tool:webfetch";
const TOOL_NAME = "WebFetch";
const TOOL_OUTPUT_PREVIEW =
  "已获取 fixture 工具事实: https://example.com/lime-workbench-tool";
const CODING_FILE_PATH =
  ".lime/qc/code-artifact-workbench-electron-fixture/src/coding-target.ts";
const CODING_FILE_DISPLAY_PATH = "src/coding-target.ts";
const CODING_ADDED_FILE_PATH =
  ".lime/qc/code-artifact-workbench-electron-fixture/src/added.ts";
const CODING_DELETED_FILE_PATH =
  ".lime/qc/code-artifact-workbench-electron-fixture/src/deleted.ts";
const CODING_MOVE_SOURCE_PATH =
  ".lime/qc/code-artifact-workbench-electron-fixture/src/source.ts";
const CODING_MOVE_DESTINATION_PATH =
  ".lime/qc/code-artifact-workbench-electron-fixture/src/destination.ts";
const CODING_ARTIFACT_ID = "code-artifact-workbench-electron:coding-target";
const CODING_COMMAND_ID = "code-artifact-workbench-electron:command:test";
const CODING_COMMAND_TEXT = "npm test -- coding-target";
const CODING_COMMAND_FAILURE_PREVIEW =
  "FAIL coding-target.test.ts: expected codingWorkbenchSmoke to be true";
const CODING_COMMAND_SUCCESS_PREVIEW =
  "PASS coding-target.test.ts: codingWorkbenchSmoke is true";
const CODING_TEST_SUITE = "coding-target";
const CODING_FILE_PREVIEW = "export const codingWorkbenchSmoke = true;";
const ARTIFACT_PREVIEW_TEXT =
  "export function greeting() { return 'Hello Lime Workbench'; }";
const ARTIFACT_CONTENT = [
  "export function greeting() {",
  "  return 'Hello Lime Workbench';",
  "}",
  "",
  "export const workbenchSmoke = true;",
  "",
].join("\n");

function expectedUserPrompt(options) {
  if (options.scenario === FILE_CHANGE_BATCH_SCENARIO) {
    return FILE_CHANGE_BATCH_DECLINE_PROMPT;
  }
  return options.scenario === "gui-coding-input"
    ? GUI_CODING_PROMPT
    : USER_PROMPT;
}

function printHelp() {
  console.log(`
Code Artifact Workbench Electron Fixture Smoke

用途:
  启动真实 Electron Desktop Host，验证代码产物 / Coding Workbench current 主链。
  默认通过 App Server JSON-RPC current 主链创建一个带代码产物 artifact.snapshot 的会话，
  再在 GUI 里从侧栏历史打开该会话并点击工作台。gui-coding-input 场景会先通过真实
  GUI 输入框发送一条 coding 请求，再验证 Workbench 面板可用。

边界:
  本脚本使用一次性本地 external backend fixture，不调用正式模型后端，不使用
  APP_SERVER_BACKEND_MODE=mock，不走 Tauri / legacy command / renderer mock fallback
  作为成功证据。

用法:
  node scripts/electron/code-artifact-workbench-fixture-smoke.mjs

选项:
  --app-url <url>        可选 renderer dev server，例如 http://127.0.0.1:1420/
  --evidence-dir <path>  证据目录
  --prefix <name>        证据文件前缀
  --scenario <name>      direct-session | gui-coding-input | file-change-batch，默认 direct-session
  --timeout-ms <ms>      总超时，默认 180000
  --interval-ms <ms>     轮询间隔，默认 500
  --keep-temp            保留临时目录便于调试
  -h, --help             显示帮助
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--app-url" && next) {
      options.appUrl = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--evidence-dir" && next) {
      options.evidenceDir = path.resolve(next.trim());
      index += 1;
      continue;
    }
    if (arg === "--prefix" && next) {
      options.prefix = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--scenario" && next) {
      options.scenario = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && next) {
      options.timeoutMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && next) {
      options.intervalMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--keep-temp") {
      options.keepTemp = true;
      continue;
    }
    throw new Error(`未知参数: ${arg}`);
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!options.evidenceDir || !options.prefix) {
    throw new Error("--evidence-dir / --prefix 均不能为空");
  }
  if (
    ![
      "direct-session",
      "gui-coding-input",
      FILE_CHANGE_BATCH_SCENARIO,
    ].includes(options.scenario)
  ) {
    throw new Error(
      "--scenario 只能是 direct-session、gui-coding-input 或 file-change-batch",
    );
  }
  return options;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withTimeout(promise, timeoutMs, label) {
  let timeout = null;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(() => {
          reject(new Error(`${label} 超时 ${timeoutMs}ms`));
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timeout) {
      clearTimeout(timeout);
    }
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertNoRendererErrors(consoleErrors, pageErrors) {
  assert(
    consoleErrors.length === 0,
    `Electron renderer console error: ${JSON.stringify(consoleErrors)}`,
  );
  assert(
    pageErrors.length === 0,
    `Electron renderer page error: ${JSON.stringify(pageErrors)}`,
  );
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

async function waitForAppUrlReady(options) {
  if (!options.appUrl) {
    return null;
  }

  const startedAt = Date.now();
  let lastError = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const response = await fetch(options.appUrl, { method: "GET" });
      if (response.ok) {
        return {
          url: options.appUrl,
          status: response.status,
          waitedMs: Date.now() - startedAt,
        };
      }
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `renderer dev server 未就绪: ${options.appUrl}; lastError=${lastError}`,
  );
}

function sanitizeText(value) {
  const sanitized = String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|session|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]")
    .replace(/sk-[A-Za-z0-9._-]+/g, "sk-[redacted]");
  return sanitized.length > 2_000
    ? `${sanitized.slice(0, 2_000)}... [truncated ${sanitized.length - 2_000} chars]`
    : sanitized;
}

function sanitizeJson(value, depth = 0) {
  if (depth > 8) {
    return "[truncated-depth]";
  }
  if (typeof value === "string") {
    return sanitizeText(value);
  }
  if (
    value === null ||
    value === undefined ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 100).map((item) => sanitizeJson(item, depth + 1));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 180)
        .map(([key, item]) => [key, sanitizeJson(item, depth + 1)]),
    );
  }
  return sanitizeText(String(value));
}

function writeJsonFile(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function cleanupTempRoot(tempRoot) {
  try {
    fs.rmSync(tempRoot, {
      recursive: true,
      force: true,
      maxRetries: TEMP_CLEANUP_RETRY_COUNT,
      retryDelay: TEMP_CLEANUP_RETRY_DELAY_MS,
    });
  } catch (error) {
    console.warn(
      `${LOG_PREFIX} temp cleanup skipped path=${tempRoot} error=${sanitizeText(error)}`,
    );
  }
}

function createTempRuntimeEnv() {
  const tempRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "code-artifact-workbench-electron-"),
  );
  const home = path.join(tempRoot, "home");
  const xdgDataHome = path.join(tempRoot, "xdg-data");
  const localAppData = path.join(tempRoot, "local-app-data");
  const roamingAppData = path.join(tempRoot, "roaming-app-data");
  const electronUserDataDir = path.join(tempRoot, "electron-user-data");
  const agentRoot = path.join(tempRoot, "agent");
  const backendPath = path.join(tempRoot, "code-artifact-backend.mjs");
  const backendLedgerPath = path.join(tempRoot, "code-artifact-backend.jsonl");

  for (const dir of [
    home,
    xdgDataHome,
    localAppData,
    roamingAppData,
    electronUserDataDir,
    agentRoot,
  ]) {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(backendLedgerPath, "");
  writeFixtureBackend(backendPath);

  return {
    tempRoot,
    electronUserDataDir,
    backendPath,
    backendLedgerPath,
    env: withElectronFixtureSystemPath({
      ...process.env,
      HOME: home,
      XDG_DATA_HOME: xdgDataHome,
      APPDATA: roamingAppData,
      LOCALAPPDATA: localAppData,
      LIME_AGENT_RUNTIME_ROOT: agentRoot,
    }),
  };
}

function writeFixtureBackend(backendPath) {
  fs.writeFileSync(
    backendPath,
    `#!/usr/bin/env node
import { appendFileSync, readFileSync } from "node:fs";

const ledgerPath = process.argv[2];
const input = JSON.parse(readFileSync(0, "utf8"));
const request = input.request || {};
const session = request.session || {};
const turn = request.turn || {};
const sessionId = String(session.sessionId || "");
const threadId = String(session.threadId || "");
const turnId = String(turn.turnId || "");
const inputText = readRuntimeInputText(request);
const providerPreference = request.providerPreference;
const modelPreference = request.modelPreference;
const requestMetadata = readApplicationMetadata(request.metadata);
const includeBackendArtifact =
  process.env.CODE_ARTIFACT_WORKBENCH_FIXTURE_SCENARIO !== "direct-session";

function readRuntimeInputText(request) {
  const parts = Array.isArray(request?.input?.parts) ? request.input.parts : [];
  return parts
    .map((part) => (typeof part?.Text?.text === "string" ? part.Text.text : ""))
    .join("");
}

function readApplicationMetadata(metadata) {
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
    return {};
  }
  const entry = metadata.additionalContext?.metadata;
  if (
    !entry ||
    entry.kind !== "application" ||
    typeof entry.value !== "string"
  ) {
    return {};
  }
  try {
    const parsed = JSON.parse(entry.value);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

if (ledgerPath) {
  appendFileSync(ledgerPath, JSON.stringify({
    kind: input.kind,
    sessionId,
    threadId,
    turnId,
    inputText,
    providerPreference,
    modelPreference,
    requestMetadata,
    recordedAt: new Date().toISOString()
  }) + "\\n");
}

${renderFileChangeGateBBackendScript()}

if (input.kind === "turnStart") {
  const isRecoveryTurn =
    inputText.includes(${JSON.stringify(CODING_RECOVERY_PROMPT_INTRO)}) &&
    inputText.includes("${CODING_COMMAND_TEXT}") &&
    inputText.includes("${CODING_COMMAND_FAILURE_PREVIEW}");
  const isCodingPrompt = inputText.includes("coding-target.test.ts") || isRecoveryTurn;
  const commandPreview = isCodingPrompt && !isRecoveryTurn
    ? "${CODING_COMMAND_FAILURE_PREVIEW}"
    : "${CODING_COMMAND_SUCCESS_PREVIEW}";
  const commandExitCode = isCodingPrompt && !isRecoveryTurn ? 1 : 0;
  const assistantText = isRecoveryTurn
    ? "已继续修复 coding-target，并通过 npm test -- coding-target。"
    : "${ASSISTANT_ARTIFACT_TEXT}";
  const turnScopedExecutionId = (baseId) =>
    isRecoveryTurn ? baseId + ":" + turnId : baseId;
  const assistantItemId = turnScopedExecutionId("code-artifact-workbench-electron:assistant");
  const toolCallId = turnScopedExecutionId("${TOOL_CALL_ID}");
  const fileChangeItemId = turnScopedExecutionId("${CODING_ARTIFACT_ID}");
  const commandId = turnScopedExecutionId("${CODING_COMMAND_ID}");
  const toolStartedAtMs = Date.now();
  const canonicalToolItem = (status, sequence, output = null) => {
    const updatedAtMs = Date.now();
    return {
      sessionId,
      threadId,
      turnId,
      itemId: toolCallId,
      sequence,
      ordinal: 4,
      createdAtMs: toolStartedAtMs,
      updatedAtMs,
      completedAtMs: output ? updatedAtMs : undefined,
      kind: "tool",
      status,
      payload: {
        type: "tool",
        call_id: toolCallId,
        name: "${TOOL_NAME}",
        arguments: [
          { name: "url", value: "https://example.com/lime-workbench-tool" },
          { name: "purpose", value: "code-artifact-workbench-electron-fixture" }
        ],
        output
      },
      metadata: {
        source: "code-artifact-workbench-electron-fixture"
      }
    };
  };
  const canonicalCommandItem = (status, sequence) => {
    const updatedAtMs = Date.now();
    return {
      sessionId,
      threadId,
      turnId,
      itemId: commandId,
      sequence,
      ordinal: sequence,
      createdAtMs: toolStartedAtMs,
      updatedAtMs,
      completedAtMs: status === "completed" || status === "failed" ? updatedAtMs : undefined,
      kind: "command",
      status,
      payload: {
        type: "command",
        command: "${CODING_COMMAND_TEXT}",
        cwd: ".",
        output: commandPreview,
        exit_code: commandExitCode
      },
      metadata: {
        source: "code-artifact-workbench-electron-fixture"
      }
    };
  };
  const canonicalFileItem = (status, sequence) => {
    const updatedAtMs = Date.now();
    return {
      sessionId,
      threadId,
      turnId,
      itemId: fileChangeItemId,
      sequence,
      ordinal: sequence,
      createdAtMs: toolStartedAtMs,
      updatedAtMs,
      completedAtMs: status === "completed" || status === "failed" ? updatedAtMs : undefined,
      kind: "file",
      status,
      payload: {
        type: "file",
        changes: [
          {
            path: "${CODING_ADDED_FILE_PATH}",
            kind: { type: "add" },
            diff: "+export const added = true;"
          },
          {
            path: "${CODING_DELETED_FILE_PATH}",
            kind: { type: "delete" },
            diff: "-export const deleted = true;"
          },
          {
            path: "${CODING_FILE_PATH}",
            kind: { type: "update" },
            diff: "-export const codingWorkbenchSmoke = false;\\n+export const codingWorkbenchSmoke = true;"
          },
          {
            path: "${CODING_MOVE_SOURCE_PATH}",
            kind: {
              type: "update",
              move_path: "${CODING_MOVE_DESTINATION_PATH}"
            },
            diff: "-export const source = true;\\n+export const destination = true;"
          }
        ],
        status: status === "completed" ? "applied" : "proposed"
      },
      metadata: {
        source: "code-artifact-workbench-electron-fixture",
        artifactId: "${CODING_ARTIFACT_ID}"
      }
    };
  };
  const events = [
      {
        type: "message.delta",
        payload: {
          itemId: assistantItemId,
          role: "assistant",
          text: assistantText,
          phase: "final_answer"
        }
      },
      {
        type: "item.started",
        payload: {
          item: canonicalToolItem("inProgress", 2)
        }
      },
      {
        type: "item.completed",
        payload: {
          item: canonicalToolItem("completed", 3, {
            text: "${TOOL_OUTPUT_PREVIEW}",
            structuredContent: { source: "fixture" },
            durationMs: Math.max(0, Date.now() - toolStartedAtMs),
            truncated: false
          })
        }
      },
      {
        type: "item.started",
        payload: {
          item: canonicalFileItem("inProgress", 4)
        }
      },
      {
        type: "item.completed",
        payload: {
          item: canonicalFileItem("completed", 5)
        }
      },
      {
        type: "item.started",
        payload: {
          item: canonicalCommandItem("inProgress", 6)
        }
      },
      {
        type: "item.completed",
        payload: {
          item: canonicalCommandItem(commandExitCode === 0 ? "completed" : "failed", 7)
        }
      },
      ...(includeBackendArtifact
        ? [
            {
              type: "artifact.snapshot",
              payload: {
                artifact: {
                  artifactId: "${ARTIFACT_ID}",
                  filePath: "${ARTIFACT_PATH}",
                  content: ${JSON.stringify(ARTIFACT_CONTENT)},
                  metadata: {
                    complete: true,
                    artifactKind: "code_file",
                    artifactTitle: "Greeting TypeScript fixture",
                    artifactStatus: "ready",
                    language: "typescript",
                    previewText: "${ARTIFACT_PREVIEW_TEXT}",
                    source: "code-artifact-workbench-electron-fixture"
                  }
                }
              }
            }
          ]
        : []),
      {
        type: "message.completed",
        payload: {
          itemId: assistantItemId,
          role: "assistant",
          text: assistantText,
          phase: "final_answer",
          status: "completed"
        }
      },
      {
        type: "turn.completed",
        payload: {
          status: "completed",
          text: "${FINAL_DONE_TEXT}"
        }
      }
    ];
  if (ledgerPath) {
    appendFileSync(ledgerPath, JSON.stringify({
      kind: "backendEvents",
      sessionId,
      threadId,
      turnId,
      executionIds: {
        assistantItemId,
        toolCallId,
        fileChangeItemId,
        commandId
      },
      eventTypes: events.map((event) => event.type),
      recordedAt: new Date().toISOString()
    }) + "\\n");
  }
  console.log(JSON.stringify({ events }));
  process.exit(0);
}

console.log(JSON.stringify({ events: [] }));
`,
    { mode: 0o755 },
  );
}

function readJsonl(filePath) {
  if (!fs.existsSync(filePath)) {
    return [];
  }
  return fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function persistBackendLedgerEvidence(sourcePath, evidencePath) {
  let backendLedger = [];
  let readError = null;
  try {
    backendLedger = readJsonl(sourcePath);
  } catch (error) {
    readError = sanitizeText(
      error instanceof Error ? error.stack || error.message : String(error),
    );
  }
  writeJsonFile(evidencePath, backendLedger.map(sanitizeJson));
  return { backendLedger, readError };
}

function parseJsonRpcLine(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function decodeJsonRpcLines(lines) {
  return Array.isArray(lines)
    ? lines.map(parseJsonRpcLine).filter(Boolean)
    : [];
}

function readTraceMessages(raw) {
  try {
    const parsed = JSON.parse(raw || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function collectTraceRequestMethods(traceMessages) {
  return traceMessages
    .filter((entry) => entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND)
    .flatMap((entry) =>
      decodeJsonRpcLines(entry?.args_preview?.request?.lines).map(
        (message) => message.method,
      ),
    )
    .filter(Boolean);
}

function collectTraceJsonRpcMessages(traceMessages) {
  return traceMessages
    .filter((entry) => entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND)
    .flatMap((entry) =>
      decodeJsonRpcLines(entry?.args_preview?.request?.lines),
    );
}

function findArtifactWriteElectronIpcTrace(traceMessages) {
  for (const entry of traceMessages) {
    if (
      entry?.command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND ||
      entry?.transport !== "electron-ipc" ||
      entry?.status !== "success"
    ) {
      continue;
    }
    for (const message of decodeJsonRpcLines(
      entry?.args_preview?.request?.lines,
    )) {
      if (message?.method !== APP_SERVER_METHOD_ARTIFACT_WRITE) {
        continue;
      }
      return {
        command: entry.command,
        transport: entry.transport,
        status: entry.status,
        method: message.method,
        requestId: message.id ?? null,
        threadId: message.params?.threadId ?? null,
        turnId: message.params?.turnId ?? null,
        artifactRef: message.params?.artifact?.artifactRef ?? null,
      };
    }
  }
  return null;
}

function readApplicationContextValue(additionalContext, key) {
  const entry = additionalContext?.[key];
  if (
    !entry ||
    entry.kind !== "application" ||
    typeof entry.value !== "string"
  ) {
    return null;
  }
  try {
    const parsed = JSON.parse(entry.value);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function readTurnStartApplicationMetadata(message) {
  return (
    readApplicationContextValue(
      message?.params?.additionalContext,
      "metadata",
    ) || {}
  );
}

function readTurnStartInputText(message) {
  const input = message?.params?.input;
  if (!Array.isArray(input)) {
    return "";
  }
  return input
    .filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("");
}

function isCodingRecoveryPromptText(value) {
  const text = String(value || "");
  return (
    text.includes(CODING_RECOVERY_PROMPT_INTRO) &&
    text.includes(CODING_COMMAND_TEXT) &&
    text.includes(CODING_COMMAND_FAILURE_PREVIEW)
  );
}

function findCodingRecoveryTurnStart(messages) {
  return messages
    .filter((message) => message?.method === APP_SERVER_METHOD_TURN_START)
    .find((message) => {
      const metadata = readTurnStartApplicationMetadata(message);
      const harness = metadata.harness || {};
      return (
        isCodingRecoveryPromptText(readTurnStartInputText(message)) &&
        harness.coding_workbench_recovery?.schemaVersion ===
          "coding-workbench-recovery/v1"
      );
    });
}

function collectArtifactSummaries(readResult) {
  const detail = readResult?.detail;
  const candidates = [
    ...(Array.isArray(detail?.artifacts) ? detail.artifacts : []),
    ...(Array.isArray(detail?.thread_read?.artifacts)
      ? detail.thread_read.artifacts
      : []),
  ];
  for (const item of readThreadItems(readResult)) {
    if (item?.type !== "fileChange") continue;
    for (const change of Array.isArray(item.changes) ? item.changes : []) {
      candidates.push({ id: item.id, path: change?.path });
    }
  }
  return candidates.filter(
    (artifact) => artifact && typeof artifact === "object",
  );
}

function collectToolCalls(readResult) {
  const detail = readResult?.detail;
  const candidates = [
    ...(Array.isArray(detail?.tool_calls) ? detail.tool_calls : []),
    ...(Array.isArray(detail?.toolCalls) ? detail.toolCalls : []),
    ...(Array.isArray(detail?.thread_read?.tool_calls)
      ? detail.thread_read.tool_calls
      : []),
    ...(Array.isArray(detail?.thread_read?.toolCalls)
      ? detail.thread_read.toolCalls
      : []),
    ...readThreadItems(readResult)
      .filter((item) => item?.type === "dynamicToolCall")
      .map((item) => ({
        ...item,
        name: item.tool,
        output: JSON.stringify(item.contentItems || []),
      })),
  ];
  return candidates.filter((toolCall) => {
    return toolCall && typeof toolCall === "object";
  });
}

function readThreadItems(readResult) {
  return (
    Array.isArray(readResult?.thread?.turns) ? readResult.thread.turns : []
  ).flatMap((turn) => (Array.isArray(turn?.items) ? turn.items : []));
}

function summarizeFileChangeBatches(readResult) {
  return readThreadItems(readResult)
    .filter((item) => item?.type === "fileChange")
    .map((item) => ({
      id: item.id,
      status: item.status,
      changes: Array.isArray(item.changes) ? item.changes : [],
    }));
}

function hasCodexFileChangeBatch(readResult) {
  return summarizeFileChangeBatches(readResult).some((item) => {
    const changes = item.changes;
    return (
      item.status === "completed" &&
      changes.length === 4 &&
      changes[0]?.path === CODING_ADDED_FILE_PATH &&
      changes[0]?.kind?.type === "add" &&
      changes[1]?.path === CODING_DELETED_FILE_PATH &&
      changes[1]?.kind?.type === "delete" &&
      changes[2]?.path === CODING_FILE_PATH &&
      changes[2]?.kind?.type === "update" &&
      changes[3]?.path === CODING_MOVE_SOURCE_PATH &&
      changes[3]?.kind?.type === "update" &&
      changes[3]?.kind?.move_path === CODING_MOVE_DESTINATION_PATH &&
      changes.every((change) => typeof change?.diff === "string")
    );
  });
}

function readLatestThreadTurnId(readResult) {
  const turn = readLatestThreadTurn(readResult);
  const turnId = turn?.id;
  return typeof turnId === "string" && turnId.trim() ? turnId.trim() : null;
}

function readLatestThreadTurn(readResult) {
  const turns = Array.isArray(readResult?.thread?.turns)
    ? readResult.thread.turns
    : [];
  return turns.at(-1) || null;
}

function findFixtureToolCall(readResult) {
  return collectToolCalls(readResult).find((toolCall) => {
    const toolCallId = String(
      toolCall.id ||
        toolCall.tool_call_id ||
        toolCall.toolCallId ||
        toolCall.toolId ||
        "",
    );
    const toolName = String(
      toolCall.tool_name || toolCall.toolName || toolCall.name || "",
    );
    return toolCallId === TOOL_CALL_ID && toolName === TOOL_NAME;
  });
}

function hasToolTimelineProjection(readResult) {
  const toolCall = findFixtureToolCall(readResult);
  if (!toolCall) {
    return false;
  }
  const status = String(toolCall.status || "").toLowerCase();
  const output = String(
    toolCall.output_preview ||
      toolCall.outputPreview ||
      toolCall.output ||
      JSON.stringify(toolCall.contentItems || ""),
  );
  return status === "completed" && output.includes(TOOL_OUTPUT_PREVIEW);
}

function hasHistoricalOperationalDetailsHidden(timelineProcessEvidence) {
  return (
    timelineProcessEvidence?.historicalTimelinePreviewCount > 0 &&
    timelineProcessEvidence?.toolCallRowCount === 0 &&
    timelineProcessEvidence?.operationalTimelineDetailsCount === 0
  );
}

function hasSessionConversationShell(snapshot) {
  return (
    snapshot?.hasMessageList === true || snapshot?.hasConversationShell === true
  );
}

function hasSessionWorkbenchAnchor(snapshot) {
  return (
    snapshot?.hasWorkbenchToggle === true ||
    snapshot?.hasTaskCenterWorkbenchTab === true ||
    snapshot?.hasWorkbenchEntry === true ||
    snapshot?.hasTaskCenterShell === true ||
    snapshot?.hasWorkbenchSidebar === true ||
    snapshot?.hasArtifactWorkbenchShell === true ||
    snapshot?.hasCanvasWorkbenchShell === true
  );
}

function hasSessionArtifactAnchor(snapshot) {
  return (
    snapshot?.hasArtifactPath === true ||
    snapshot?.hasCodeText === true ||
    hasSessionWorkbenchAnchor(snapshot)
  );
}

function hasHydratedSessionSnapshot(snapshot) {
  if (!snapshot || snapshot.isRestoringSession === true) {
    return false;
  }

  const hasAssistantCompletionCopy =
    snapshot.hasDoneText === true || snapshot.hasGeneratedText === true;
  const hasDirectArtifactHydration =
    snapshot.hasUserPrompt === true &&
    hasAssistantCompletionCopy &&
    hasSessionArtifactAnchor(snapshot);
  const hasWorkbenchShellHydration =
    snapshot.hasUserPrompt === true &&
    (hasAssistantCompletionCopy ||
      snapshot.hasArtifactPath === true ||
      snapshot.hasCodeText === true) &&
    (snapshot.hasWorkbenchToggle === true ||
      snapshot.hasTaskCenterWorkbenchTab === true) &&
    hasSessionConversationShell(snapshot);

  return hasDirectArtifactHydration || hasWorkbenchShellHydration;
}

async function inspectHistoricalTimelineSummary(page, options) {
  const startedAt = Date.now();
  const timeoutMs = Math.min(options.timeoutMs, 15_000);
  let lastSnapshot = null;

  while (Date.now() - startedAt < timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => {
      const isVisible = (element) => {
        if (!(element instanceof HTMLElement)) {
          return false;
        }
        const style = window.getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          rect.width > 0 &&
          rect.height > 0
        );
      };
      const historicalTimelinePreviews = Array.from(
        document.querySelectorAll(
          '[data-testid^="message-list-historical-timeline-preview:"]',
        ),
      ).filter(isVisible);
      const toolCallRows = Array.from(
        document.querySelectorAll('[data-testid="tool-call-row"]'),
      ).filter(isVisible);
      const operationalTimelineDetails = Array.from(
        document.querySelectorAll(
          [
            'details[data-testid^="agent-thread-block:"][data-testid$=":process"]',
            'details[data-testid^="agent-thread-block:"][data-testid$=":approval"]',
          ].join(","),
        ),
      ).filter(isVisible);
      const text = document.body?.innerText || "";
      return {
        historicalTimelinePreviewCount: historicalTimelinePreviews.length,
        toolCallRowCount: toolCallRows.length,
        operationalTimelineDetailsCount: operationalTimelineDetails.length,
        bodyText: text,
      };
    });
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.historicalTimelinePreviewCount > 0 &&
      snapshot.toolCallRowCount === 0 &&
      snapshot.operationalTimelineDetailsCount === 0
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }

  return lastSnapshot;
}

function hasCodeArtifactProjection(readResult) {
  return collectArtifactSummaries(readResult).some((artifact) => {
    const artifactId = String(
      artifact.artifactId || artifact.artifactRef || artifact.id || "",
    );
    const artifactPath = String(
      artifact.path || artifact.filePath || artifact.file_path || "",
    );
    return (
      (artifactId === ARTIFACT_ID && artifactPath === ARTIFACT_PATH) ||
      artifactPath === ARTIFACT_PATH
    );
  });
}

function hasCodingProjection(readResult) {
  const items = readThreadItems(readResult);
  return (
    items.some(
      (item) =>
        item?.type === "fileChange" &&
        item.status === "completed" &&
        item.changes?.some((change) => change?.path === CODING_FILE_PATH),
    ) &&
    items.some(
      (item) =>
        item?.type === "commandExecution" &&
        (item.status === "completed" || item.status === "failed") &&
        String(item.command || "").includes(CODING_COMMAND_TEXT) &&
        String(item.aggregatedOutput || "").includes("codingWorkbenchSmoke"),
    )
  );
}

function hasCodingSuccessProjection(readResult) {
  const turn = readLatestThreadTurn(readResult);
  const items = Array.isArray(turn?.items) ? turn.items : [];
  return (
    turn?.status === "completed" &&
    items.some(
      (item) =>
        item?.type === "fileChange" &&
        item.status === "completed" &&
        item.changes?.some((change) => change?.path === CODING_FILE_PATH),
    ) &&
    items.some(
      (item) =>
        item?.type === "commandExecution" &&
        item.status === "completed" &&
        item.exitCode === 0 &&
        String(item.command || "").includes(CODING_COMMAND_TEXT) &&
        String(item.aggregatedOutput || "").includes(
          CODING_COMMAND_SUCCESS_PREVIEW,
        ),
    )
  );
}

async function ensureDefaultWorkspace(page) {
  const ensured = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_WORKSPACE_DEFAULT_ENSURE,
    {},
  );
  const workspace = ensured.result?.workspace;
  const workspaceId = String(workspace?.id || "").trim();
  assert(workspaceId, "workspace/default/ensure 未返回可用 workspace.id");
  return {
    workspaceId,
    rootPath: workspace?.rootPath || workspace?.root_path || null,
    workspace,
  };
}

async function bindGuiWorkspaceToFixture(page, workspaceId) {
  return await page.evaluate(
    ({ workspaceId, sessionId }) => {
      const lastProjectKey = "agent_last_project_id";
      const sessionWorkspaceKey = `agent_session_workspace_${sessionId}`;
      window.localStorage.setItem("lime:agent-debug", "1");
      window.localStorage.setItem(lastProjectKey, JSON.stringify(workspaceId));
      window.localStorage.setItem(
        sessionWorkspaceKey,
        JSON.stringify(workspaceId),
      );
      window.dispatchEvent(
        new CustomEvent("agent-persisted-project-id-changed", {
          detail: {
            key: lastProjectKey,
            projectId: workspaceId,
          },
        }),
      );
      window.dispatchEvent(new Event("focus"));
      return {
        lastProject: window.localStorage.getItem(lastProjectKey),
        sessionWorkspace: window.localStorage.getItem(sessionWorkspaceKey),
      };
    },
    { workspaceId, sessionId: SESSION_ID },
  );
}

async function navigateGuiToWorkspaceScopedAgent(page, options, workspaceId) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  let clickedNewConversation = false;

  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ workspaceId }) => {
        const text = document.body?.innerText || "";
        const recentShelf = document.querySelector(
          '[data-testid="app-sidebar-recent-conversations"]',
        );
        const recentShelfText = recentShelf?.textContent || "";
        const hasConversationList = Boolean(recentShelf);
        const buttons = Array.from(document.querySelectorAll("button")).map(
          (button) => ({
            title: button.getAttribute("title") || "",
            text: button.textContent || "",
            aria: button.getAttribute("aria-label") || "",
            testId: button.getAttribute("data-testid") || "",
          }),
        );
        return {
          url: window.location.href,
          localStorageWorkspace: window.localStorage.getItem(
            "agent_last_project_id",
          ),
          localStorageMatchesWorkspace:
            window.localStorage.getItem("agent_last_project_id") ===
            JSON.stringify(workspaceId),
          hasConversationList,
          recentShelfText,
          hasNewConversationButton: buttons.some((button) =>
            [button.title, button.text, button.aria].some((label) =>
              label.includes("新建对话"),
            ),
          ),
          hasWorkspaceShell: Boolean(
            document.querySelector('[data-testid="agent-chat-workspace"]') ||
            document.querySelector('[data-testid="chat-workspace"]') ||
            document.querySelector(
              '[data-testid="theme-workbench-harness-toggle"]',
            ) ||
            document.querySelector('[data-testid="toggle-harness"]'),
          ),
          bodyText: text,
        };
      },
      { workspaceId },
    );

    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;

    if (
      clickedNewConversation &&
      snapshot.hasConversationList &&
      snapshot.localStorageMatchesWorkspace
    ) {
      return snapshot;
    }

    const clicked = await page.evaluate(() => {
      const buttons = Array.from(document.querySelectorAll("button"));
      const newConversationButton = buttons.find((button) => {
        const label = [
          button.getAttribute("title") || "",
          button.getAttribute("aria-label") || "",
          button.textContent || "",
        ].join("\n");
        return label.includes("新建对话");
      });
      if (newConversationButton instanceof HTMLElement) {
        newConversationButton.click();
        return true;
      }
      window.dispatchEvent(new Event("focus"));
      return false;
    });
    clickedNewConversation = clickedNewConversation || clicked;

    await sleep(options.intervalMs);
  }

  throw new Error(
    `GUI 未进入 workspace-scoped Agent 状态: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForInputReady(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ sessionId }) => {
        const textarea = document.querySelector(
          'textarea[name="agent-chat-message"]',
        );
        const rect = textarea?.getBoundingClientRect();
        const style = textarea ? window.getComputedStyle(textarea) : null;
        const bodyText = document.body?.innerText || "";
        const textareaSessionId =
          textarea instanceof HTMLTextAreaElement
            ? textarea.getAttribute("data-session-id") || null
            : null;
        const activeConversation = document.querySelector(
          '[data-testid="app-sidebar-conversation-open"][aria-current="page"]',
        );
        const activeConversationMatches =
          (activeConversation?.getAttribute("title") || "").includes(
            "代码产物工作台 Electron fixture",
          ) ||
          (activeConversation?.textContent || "").includes(
            "代码产物工作台 Electron fixture",
          );
        const restoring =
          bodyText.includes("正在恢复生成会话") ||
          bodyText.includes("正在同步最近一次生成会话") ||
          bodyText.includes("Restoring generation session") ||
          bodyText.includes("Syncing the latest generation session");
        const visible = Boolean(
          textarea &&
          rect &&
          rect.width > 16 &&
          rect.height > 16 &&
          style?.visibility !== "hidden" &&
          style?.display !== "none",
        );
        return {
          url: window.location.href,
          sessionId: textareaSessionId,
          sessionMatches: textareaSessionId === sessionId,
          activeConversationMatches,
          restoring,
          hasTextarea: Boolean(textarea),
          textareaVisible: visible,
          textareaDisabled:
            textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
          textareaValue:
            textarea instanceof HTMLTextAreaElement ? textarea.value : null,
          hasInputbarCore: Boolean(
            document.querySelector('[data-testid="inputbar-core-container"]'),
          ),
          bodyText,
        };
      },
      { sessionId: SESSION_ID },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    const hasWritableInput =
      !snapshot.restoring &&
      snapshot.hasTextarea &&
      snapshot.textareaVisible &&
      snapshot.textareaDisabled === false;
    if (
      hasWritableInput &&
      (snapshot.sessionMatches ||
        snapshot.activeConversationMatches ||
        snapshot.hasInputbarCore)
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Coding 输入框未就绪: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

async function sendPromptFromGui(page, options, prompt) {
  const before = await waitForInputReady(page, options);
  const textarea = page.locator('textarea[name="agent-chat-message"]').first();
  await textarea.fill(prompt);
  const afterFill = await page.evaluate((prompt) => {
    const input = document.querySelector('textarea[name="agent-chat-message"]');
    return {
      value: input instanceof HTMLTextAreaElement ? input.value : null,
      promptVisibleInTextarea:
        input instanceof HTMLTextAreaElement ? input.value === prompt : false,
    };
  }, prompt);
  assert(
    afterFill.promptVisibleInTextarea,
    `输入框未保留 coding 请求: ${JSON.stringify(sanitizeJson(afterFill))}`,
  );

  const clicked = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll("button"));
    const sendButton = buttons.find((button) => {
      const label = [
        button.getAttribute("aria-label") || "",
        button.getAttribute("title") || "",
        button.textContent || "",
      ].join("\n");
      return (
        (label.includes("发送") || /\bSend\b/i.test(label)) && !button.disabled
      );
    });
    if (sendButton instanceof HTMLElement) {
      sendButton.click();
      return {
        clicked: true,
        label:
          sendButton.getAttribute("aria-label") ||
          sendButton.getAttribute("title") ||
          sendButton.textContent ||
          "send",
      };
    }
    return {
      clicked: false,
      labels: buttons.map((button) =>
        [
          button.getAttribute("aria-label") || "",
          button.getAttribute("title") || "",
          button.textContent || "",
        ].join(" / "),
      ),
    };
  });
  assert(
    clicked?.clicked,
    `未找到可点击发送按钮: ${JSON.stringify(sanitizeJson(clicked))}`,
  );
  return {
    before,
    afterFill,
    clicked,
  };
}

async function startFileChangeBatchTurnFromGui(
  page,
  options,
  prompt,
  decision,
) {
  const beforeLedgerCount = readJsonl(options.backendLedgerPath || "").length;
  const guiInput = await sendPromptFromGui(page, options, prompt);
  const startedAt = Date.now();
  let pendingEntry = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const ledger = readJsonl(options.backendLedgerPath || "");
    pendingEntry = ledger
      .slice(beforeLedgerCount)
      .find(
        (entry) =>
          entry?.kind === "fileChangeGateBEvents" && entry?.phase === "pending",
      );
    if (
      pendingEntry?.turnId &&
      pendingEntry?.itemId &&
      pendingEntry?.requestId
    ) {
      break;
    }
    await sleep(options.intervalMs);
  }
  assert(
    pendingEntry?.turnId && pendingEntry?.itemId && pendingEntry?.requestId,
    `FileChange ${decision} 未到达 backend pending`,
  );
  const identity = {
    threadId: pendingEntry.threadId,
    turnId: pendingEntry.turnId,
    itemId: pendingEntry.itemId,
    requestId: pendingEntry.requestId,
    prompt: "Apply the exact Add/Delete/Update/Move file batch?",
  };
  const pendingGui = await waitForFileChangeApprovalPending(
    page,
    options,
    identity,
  );
  const click = await clickFileChangeApprovalDecision(
    page,
    options,
    decision,
    identity,
  );
  const terminal = await waitForFileChangeTerminalReadModel(page, options, {
    ...identity,
    decision,
    invokeThreadRead: (targetPage, params) =>
      invokeAppServerFromPage(
        targetPage,
        APP_SERVER_METHOD_THREAD_READ,
        params,
      ),
  });
  const terminalGui = await waitForFileChangeTerminalGui(page, options, {
    status: decision === "cancel" ? "inProgress" : "declined",
  });
  return sanitizeJson({
    prompt,
    decision,
    guiInput,
    identity,
    pendingGui,
    click,
    terminal,
    terminalGui,
  });
}

function summarizeListVisibility(listResult) {
  const threads = Array.isArray(listResult?.result?.data)
    ? listResult.result.data
    : [];
  const matchingThread = threads.find(
    (thread) => thread?.id === THREAD_ID && thread?.sessionId === SESSION_ID,
  );
  return {
    count: threads.length,
    containsFixtureSession: Boolean(matchingThread),
    fixtureSession: matchingThread ?? null,
  };
}

function isTransientPageEvaluationError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return (
    message.includes("Execution context was destroyed") ||
    message.includes("most likely because of a navigation") ||
    message.includes("Cannot find context with specified id")
  );
}

async function evaluatePageSnapshot(page, pageFunction, arg) {
  try {
    return await page.evaluate(pageFunction, arg);
  } catch (error) {
    if (isTransientPageEvaluationError(error)) {
      return null;
    }
    throw error;
  }
}

async function waitForRendererReady(page, options, onSnapshot) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => ({
      url: window.location.href,
      title: document.title || "",
      electron: window.__LIME_ELECTRON__ === true,
      hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
      supportsAppServer:
        typeof window.electronAPI?.supportsCommand === "function" &&
        window.electronAPI.supportsCommand("app_server_handle_json_lines"),
      startupVisible: Boolean(
        document.querySelector("[data-lime-startup-shell]"),
      ),
      appSidebarVisible: Boolean(
        document.querySelector('[data-testid="app-sidebar"]'),
      ),
      bodyText: document.body?.innerText || "",
    }));
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    onSnapshot?.(snapshot);
    if (
      snapshot.electron &&
      snapshot.hasInvokeBridge &&
      snapshot.supportsAppServer &&
      !snapshot.startupVisible &&
      snapshot.appSidebarVisible
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error("Electron renderer / App Server bridge 未就绪");
}

async function clearInvokeBuffers(page) {
  return await page.evaluate(() => {
    window.localStorage.removeItem("lime_invoke_error_buffer_v1");
    window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
    return {
      cleared: true,
      clearedAt: new Date().toISOString(),
    };
  });
}

async function invokeAppServerFromPage(page, method, params = {}) {
  return await page.evaluate(
    async ({ command, method, params }) => {
      const invoke = window.electronAPI?.invoke;
      if (typeof invoke !== "function") {
        throw new Error("Electron preload invoke bridge is unavailable");
      }
      const id = `code-artifact-workbench-${Date.now()}-${Math.random()}`;
      const response = await invoke(command, {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id,
              method,
              params,
            }),
          ],
        },
      });
      const messages = Array.isArray(response?.lines)
        ? response.lines
            .map((line) => {
              try {
                return JSON.parse(line);
              } catch {
                return null;
              }
            })
            .filter(Boolean)
        : [];
      const error = messages.find(
        (message) => message?.id === id && message.error,
      );
      if (error) {
        throw new Error(`${method} failed: ${JSON.stringify(error.error)}`);
      }
      const result = messages.find(
        (message) =>
          message?.id === id &&
          Object.prototype.hasOwnProperty.call(message, "result"),
      );
      if (!result) {
        throw new Error(`${method} did not return a JSON-RPC result`);
      }
      return {
        result: result.result,
        messages,
      };
    },
    {
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      method,
      params,
    },
  );
}

async function writeArtifactFromCurrentRenderer(page, options, params) {
  assert(
    options.appUrl,
    "direct-session typed artifact/write Gate B requires --app-url so the current Renderer client is exercised",
  );
  const moduleUrl = new URL("/src/lib/api/appServer.ts", options.appUrl).href;
  return await page.evaluate(
    async ({ moduleUrl, params }) => {
      const module = await import(moduleUrl);
      if (typeof module.createAppServerClient !== "function") {
        throw new Error("Renderer current AppServerClient is unavailable");
      }
      const response = await module
        .createAppServerClient()
        .writeArtifact(params);
      return response.result;
    },
    { moduleUrl, params },
  );
}

async function waitForArtifactWriteElectronIpcTrace(page, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 15_000)) {
    const traceRaw = await page.evaluate(() =>
      window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    );
    const evidence = findArtifactWriteElectronIpcTrace(
      readTraceMessages(traceRaw),
    );
    if (evidence) {
      return evidence;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    "未在 Renderer current safeInvoke trace 中观察到 artifact/write Electron IPC",
  );
}

async function initializeAppServer(page) {
  const initialize = await invokeAppServerFromPage(page, "initialize", {
    clientInfo: {
      name: "code-artifact-workbench-electron-fixture",
      version: "1.0.0",
    },
    capabilities: { eventMethods: ["agentSession/event"] },
  });
  await page.evaluate(async (command) => {
    await window.electronAPI.invoke(command, {
      request: {
        lines: [JSON.stringify({ jsonrpc: "2.0", method: "initialized" })],
      },
    });
  }, APP_SERVER_HANDLE_JSON_LINES_COMMAND);
  return initialize.result;
}

function summarizeCodeArtifactRead(readResult, requests = []) {
  return {
    requestMethods: Array.from(
      new Set(requests.map((request) => request.method).filter(Boolean)),
    ),
    detailItemCount: Array.isArray(readResult?.detail?.items)
      ? readResult.detail.items.length
      : null,
    detailArtifactCount: Array.isArray(readResult?.detail?.artifacts)
      ? readResult.detail.artifacts.length
      : null,
    threadReadArtifactCount: Array.isArray(
      readResult?.detail?.thread_read?.artifacts,
    )
      ? readResult.detail.thread_read.artifacts.length
      : null,
    threadReadToolCallCount: Array.isArray(
      readResult?.detail?.thread_read?.tool_calls,
    )
      ? readResult.detail.thread_read.tool_calls.length
      : null,
    codingProjectionPersisted: hasCodingProjection(readResult),
    codexFileChangeBatchPersisted: hasCodexFileChangeBatch(readResult),
    fileChangeBatches: summarizeFileChangeBatches(readResult),
    toolTimelineProjectionPersisted: hasToolTimelineProjection(readResult),
    fixtureToolCall: findFixtureToolCall(readResult) ?? null,
    latestTurnStatus: readLatestThreadTurn(readResult)?.status ?? null,
    artifactProjectionPersisted: hasCodeArtifactProjection(readResult),
    detailTextIncludesArtifact: JSON.stringify(readResult || {}).includes(
      ARTIFACT_ID,
    ),
  };
}

async function waitForCodeArtifactReadModel(
  page,
  options,
  { requests = [], timeoutMs = 60_000, requireCodingSuccess = false } = {},
) {
  const startedAt = Date.now();
  let lastRead = null;
  while (Date.now() - startedAt < timeoutMs) {
    let read = null;
    try {
      const params = {
        threadId: THREAD_ID,
        includeTurns: true,
      };
      requests.push({ method: APP_SERVER_METHOD_THREAD_READ, params });
      read = await invokeAppServerFromPage(
        page,
        APP_SERVER_METHOD_THREAD_READ,
        params,
      );
    } catch (error) {
      if (!isTransientPageEvaluationError(error)) {
        throw error;
      }
      await waitForRendererReady(page, {
        ...options,
        timeoutMs: Math.min(15_000, options.timeoutMs),
      });
      await sleep(500);
      continue;
    }
    lastRead = read.result;
    const text = JSON.stringify(read.result || {});
    if (
      hasCodeArtifactProjection(read.result) &&
      hasToolTimelineProjection(read.result) &&
      hasCodingProjection(read.result) &&
      (!requireCodingSuccess || hasCodingSuccessProjection(read.result))
    ) {
      return read.result;
    }
    await sleep(500);
  }

  throw new Error(
    `代码产物会话未完成，或未持久化 artifact.snapshot / tool_calls / coding facts: ${JSON.stringify(
      summarizeCodeArtifactRead(lastRead, requests),
    )}`,
  );
}

async function waitForCodeArtifactTurnTerminal(
  page,
  options,
  { requests = [], timeoutMs = 60_000 } = {},
) {
  const startedAt = Date.now();
  let lastRead = null;
  while (Date.now() - startedAt < timeoutMs) {
    const params = {
      threadId: THREAD_ID,
      includeTurns: true,
    };
    requests.push({ method: APP_SERVER_METHOD_THREAD_READ, params });
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_THREAD_READ,
      params,
    );
    lastRead = read.result;
    if (
      readLatestThreadTurn(read.result)?.status === "completed" &&
      hasToolTimelineProjection(read.result) &&
      hasCodingProjection(read.result)
    ) {
      return read.result;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `代码产物会话未在 typed artifact/write 前到达 canonical terminal: ${JSON.stringify(
      summarizeCodeArtifactRead(lastRead, requests),
    )}`,
  );
}

async function startCodeArtifactSession(page, workspace, requests) {
  async function call(method, params = {}) {
    requests?.push({ method, params });
    return await invokeAppServerFromPage(page, method, params);
  }

  const session = await call(APP_SERVER_METHOD_THREAD_START, {
    model: "fixture-model",
    modelProvider: "fixture-provider",
    serviceName: SESSION_TITLE,
    threadSource: "appServer",
    historyMode: "paginated",
    cwd: workspace.rootPath || undefined,
    runtimeWorkspaceRoots: workspace.rootPath
      ? [workspace.rootPath]
      : undefined,
  });
  const thread = session.result?.thread;
  const sessionId = String(thread?.sessionId || "").trim();
  const threadId = String(thread?.id || "").trim();
  assert(sessionId, "thread/start 未返回 canonical sessionId");
  assert(threadId, "thread/start 未返回 canonical thread.id");
  SESSION_ID = sessionId;
  THREAD_ID = threadId;
  TURN_ID = `pending-turn-${threadId}`;

  await notifyAgentRuntimeSessionChanged(page, workspace.workspaceId);

  return {
    session: session.result,
    sessionId: SESSION_ID,
    threadId: THREAD_ID,
  };
}

async function notifyAgentRuntimeSessionChanged(page, workspaceId) {
  await page.evaluate(
    ({ sessionId, workspaceId }) => {
      window.dispatchEvent(
        new CustomEvent("lime:agent-runtime-sessions-changed", {
          detail: {
            reason: "external",
            sessionId,
            workspaceId,
          },
        }),
      );
      window.dispatchEvent(new Event("focus"));
    },
    { sessionId: SESSION_ID, workspaceId },
  );
}

async function createCodeArtifactSession(page, options, workspace) {
  const requests = [];

  const session = await startCodeArtifactSession(page, workspace, requests);

  async function call(method, params = {}) {
    requests.push({ method, params });
    return await invokeAppServerFromPage(page, method, params);
  }

  const turn = await call(APP_SERVER_METHOD_TURN_START, {
    threadId: THREAD_ID,
    clientUserMessageId: `code-artifact-workbench-${Date.now()}`,
    input: [{ type: "text", text: USER_PROMPT }],
    model: "fixture-model",
    approvalPolicy: "never",
    sandboxPolicy: "workspace-write",
    responsesapiClientMetadata: {
      source: "code-artifact-workbench-electron-fixture",
    },
  });
  const turnId = String(turn.result?.turn?.id || "").trim();
  assert(turnId, "turn/start 未返回 canonical turn.id");
  TURN_ID = turnId;

  await waitForCodeArtifactTurnTerminal(page, options, { requests });
  const artifactWriteParams = {
    threadId: THREAD_ID,
    turnId: TURN_ID,
    artifact: {
      artifactRef: ARTIFACT_ID,
      path: ARTIFACT_PATH,
      title: "Greeting TypeScript fixture",
      kind: "code_file",
      status: "ready",
      content: ARTIFACT_CONTENT,
      metadata: {
        complete: true,
        artifactKind: "code_file",
        artifactTitle: "Greeting TypeScript fixture",
        artifactStatus: "ready",
        language: "typescript",
        previewText: ARTIFACT_PREVIEW_TEXT,
        source: "code-artifact-workbench-electron-fixture",
      },
    },
  };
  requests.push({
    method: APP_SERVER_METHOD_ARTIFACT_WRITE,
    params: artifactWriteParams,
  });
  const artifactWrite = await writeArtifactFromCurrentRenderer(
    page,
    options,
    artifactWriteParams,
  );
  assert(
    artifactWrite?.threadId === THREAD_ID &&
      artifactWrite?.turnId === TURN_ID &&
      artifactWrite?.artifactRef === ARTIFACT_ID &&
      typeof artifactWrite?.eventId === "string" &&
      artifactWrite.eventId.length > 0 &&
      artifactWrite?.sidecar?.contentStatus === "available",
    `artifact/write 未返回完整 typed 保存证据: ${JSON.stringify(
      sanitizeJson(artifactWrite),
    )}`,
  );
  const artifactWriteIpcTrace = await waitForArtifactWriteElectronIpcTrace(
    page,
    options,
  );
  assert(
    artifactWriteIpcTrace.threadId === THREAD_ID &&
      artifactWriteIpcTrace.turnId === TURN_ID &&
      artifactWriteIpcTrace.artifactRef === ARTIFACT_ID,
    `artifact/write Electron IPC identity 不一致: ${JSON.stringify(
      sanitizeJson(artifactWriteIpcTrace),
    )}`,
  );

  const read = await waitForCodeArtifactReadModel(page, options, { requests });

  return {
    session: session.session,
    turn: turn.result,
    sessionId: SESSION_ID,
    threadId: THREAD_ID,
    turnId: TURN_ID,
    artifactWrite,
    artifactWriteIpcTrace,
    read,
    requests,
  };
}

async function waitForGuiSessionVisible(page, options) {
  const startedAt = Date.now();
  let lastRefreshAt = 0;
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ title, sessionId }) => {
        const text = document.body?.innerText || "";
        const recentShelf = document.querySelector(
          '[data-testid="app-sidebar-recent-conversations"]',
        );
        const buttons = Array.from(document.querySelectorAll("button")).map(
          (button) => ({
            title: button.getAttribute("title") || "",
            text: button.textContent || "",
            aria: button.getAttribute("aria-label") || "",
            testId: button.getAttribute("data-testid") || "",
          }),
        );
        const traceRaw = window.localStorage.getItem(
          "lime_invoke_trace_buffer_v1",
        );
        const errorRaw = window.localStorage.getItem(
          "lime_invoke_error_buffer_v1",
        );
        return {
          url: window.location.href,
          hasSessionTitle: text.includes(title),
          hasRecentShelf: Boolean(recentShelf),
          recentShelfText: recentShelf?.textContent || "",
          hasWorkbenchToggle: Boolean(
            document.querySelector(
              '[data-testid="theme-workbench-harness-toggle"]',
            ) || document.querySelector('[data-testid="toggle-harness"]'),
          ),
          hasWorkbenchSidebar: Boolean(
            document.querySelector('[data-testid="general-workbench-sidebar"]'),
          ),
          activeConversation: Boolean(
            document.querySelector(
              `[data-testid="app-sidebar-recent-conversations"] button[aria-current="page"]`,
            ),
          ),
          matchingButtonCount: buttons.filter(
            (button) =>
              button.title.includes(title) ||
              button.text.includes(title) ||
              button.aria.includes(title) ||
              button.title.includes(sessionId) ||
              button.text.includes(sessionId),
          ).length,
          appServerListCallCount: (() => {
            try {
              const entries = JSON.parse(traceRaw || "[]");
              return Array.isArray(entries)
                ? entries.filter((entry) =>
                    JSON.stringify(entry).includes("thread/list"),
                  ).length
                : 0;
            } catch {
              return 0;
            }
          })(),
          invokeErrorRaw: errorRaw,
          bodyText: text,
        };
      },
      { title: SESSION_TITLE, sessionId: SESSION_ID },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.hasSessionTitle ||
      snapshot.matchingButtonCount > 0 ||
      snapshot.hasWorkbenchToggle
    ) {
      return snapshot;
    }
    if (Date.now() - lastRefreshAt > 2_000) {
      lastRefreshAt = Date.now();
      await page.evaluate(() => {
        window.dispatchEvent(new Event("focus"));
      });
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `GUI 未显示代码产物 fixture 会话: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function openFixtureSessionFromSidebar(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ title }) => {
        const conversationButtons = Array.from(
          document.querySelectorAll(
            '[data-testid="app-sidebar-conversation-open"]',
          ),
        );
        const button = conversationButtons.find(
          (candidate) =>
            (candidate.getAttribute("title") || "").includes(title) ||
            (candidate.textContent || "").includes(title),
        );
        if (!button) {
          const candidates = Array.from(document.querySelectorAll("button"));
          const moreButton = candidates.find((candidate) =>
            (candidate.textContent || "").includes("查看更多对话"),
          );
          moreButton?.click();
          return {
            clicked: false,
            reason: "not-found",
            conversationButtonCount: conversationButtons.length,
            bodyText: document.body?.innerText || "",
          };
        }
        button.click();
        return {
          clicked: true,
          reason: "clicked",
          title: button.getAttribute("title") || "",
          text: button.textContent || "",
          ariaCurrent: button.getAttribute("aria-current") || "",
          conversationButtonCount: conversationButtons.length,
        };
      },
      { title: SESSION_TITLE },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (snapshot.clicked) {
      return;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `侧栏未找到 fixture 会话: ${SESSION_TITLE}: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForFixtureSessionOpenedFromSidebar(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ title, sessionId }) => {
        const text = document.body?.innerText || "";
        const mainText = document.querySelector("main")?.textContent || "";
        const activeConversation = document.querySelector(
          '[data-testid="app-sidebar-conversation-open"][aria-current="page"]',
        );
        const textarea = document.querySelector(
          'textarea[name="agent-chat-message"]',
        );
        const textareaSessionId =
          textarea instanceof HTMLTextAreaElement
            ? textarea.getAttribute("data-session-id") || null
            : null;
        const restoring =
          text.includes("正在恢复生成会话") ||
          text.includes("正在同步最近一次生成会话") ||
          text.includes("Restoring generation session") ||
          text.includes("Syncing the latest generation session");
        const agentPerfEntries =
          typeof window.__LIME_AGENTUI_PERF__?.entries === "function"
            ? window.__LIME_AGENTUI_PERF__.entries().slice(-20)
            : [];
        const traceRaw = window.localStorage.getItem(
          "lime_invoke_trace_buffer_v1",
        );
        const frontendDebugEntries = (() => {
          try {
            const entries = JSON.parse(traceRaw || "[]");
            return Array.isArray(entries)
              ? entries
                  .filter((entry) =>
                    JSON.stringify(entry).includes("report_frontend_debug_log"),
                  )
                  .slice(-20)
              : [];
          } catch {
            return [];
          }
        })();
        return {
          hasSessionTitle: text.includes(title),
          hasSessionTitleInMain: mainText.includes(title),
          activeConversationTitle:
            activeConversation?.getAttribute("title") || null,
          activeConversationText: activeConversation?.textContent || null,
          activeConversationMatches:
            (activeConversation?.getAttribute("title") || "").includes(title) ||
            (activeConversation?.textContent || "").includes(title),
          textareaSessionId,
          sessionMatches: textareaSessionId === sessionId,
          restoring,
          hasEmptyState: Boolean(
            document.querySelector('[data-testid="empty-state"]'),
          ),
          hasConversationShell: Boolean(
            document.querySelector('[data-testid="agent-chat-workspace"]') ||
            document.querySelector('[data-testid="chat-workspace"]') ||
            document.querySelector('[data-testid="message-list"]') ||
            document.querySelector('[data-testid="message-list-frame"]'),
          ),
          hasTaskCenterShell: Boolean(
            document.querySelector(
              '[data-testid="task-center-chrome-shell"]',
            ) ||
            document.querySelector('[data-testid="task-center-tab-strip"]'),
          ),
          hasWorkbenchTab: Boolean(
            document.querySelector('[data-testid="task-center-tab-workbench"]'),
          ),
          hasCanvasWorkbenchShell: Boolean(
            document.querySelector('[data-testid="canvas-workbench-shell"]') ||
            document.querySelector('[data-testid="canvas-workbench-layout"]'),
          ),
          agentPerfEntries,
          frontendDebugEntries,
          bodyText: text,
        };
      },
      { title: SESSION_TITLE, sessionId: SESSION_ID },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    const hasOpenedShell =
      snapshot.hasConversationShell ||
      snapshot.hasTaskCenterShell ||
      snapshot.hasWorkbenchTab ||
      snapshot.hasCanvasWorkbenchShell;
    const hasTargetSession =
      snapshot.sessionMatches ||
      snapshot.activeConversationMatches ||
      snapshot.hasSessionTitleInMain;
    if (!snapshot.restoring && hasTargetSession && hasOpenedShell) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `侧栏点击后 fixture 会话未进入 current 工作区: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForSessionHydrated(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ artifactPath, doneText, userPrompt, assistantArtifactText }) => {
        const text = document.body?.innerText || "";
        const normalizeText = (value) =>
          String(value || "")
            .replace(/\s+/g, " ")
            .trim();
        const normalizedText = normalizeText(text);
        const normalizedUserPrompt = normalizeText(userPrompt);
        return {
          url: window.location.href,
          isRestoringSession:
            text.includes("正在恢复生成会话") ||
            text.includes("正在同步最近一次生成会话"),
          hasUserPrompt:
            text.includes(userPrompt) ||
            (normalizedUserPrompt.length > 0 &&
              normalizedText.includes(normalizedUserPrompt)),
          hasDoneText: text.includes(doneText),
          hasGeneratedText: text.includes(assistantArtifactText),
          hasArtifactPath: text.includes(artifactPath),
          hasCodeText: text.includes("Hello Lime Workbench"),
          hasMessageList: Boolean(
            document.querySelector('[data-testid="message-list"]') ||
            document.querySelector('[data-testid="message-list-frame"]'),
          ),
          hasWorkbenchToggle: Boolean(
            document.querySelector(
              '[data-testid="theme-workbench-harness-toggle"]',
            ) || document.querySelector('[data-testid="toggle-harness"]'),
          ),
          hasTaskCenterShell: Boolean(
            document.querySelector(
              '[data-testid="task-center-chrome-shell"]',
            ) ||
            document.querySelector('[data-testid="task-center-tab-strip"]'),
          ),
          hasTaskCenterWorkbenchTab: Boolean(
            document.querySelector('[data-testid="task-center-tab-workbench"]'),
          ),
          hasWorkbenchEntry: Array.from(
            document.querySelectorAll("button"),
          ).some((button) => {
            const label = [
              button.getAttribute("title") || "",
              button.getAttribute("aria-label") || "",
              button.textContent || "",
            ].join("\n");
            return label.includes("工作台") || label.includes("Workbench");
          }),
          hasWorkbenchSidebar: Boolean(
            document.querySelector('[data-testid="general-workbench-sidebar"]'),
          ),
          hasArtifactWorkbenchShell: Boolean(
            document.querySelector('[data-testid="artifact-workbench-shell"]'),
          ),
          hasCanvasWorkbenchShell: Boolean(
            document.querySelector('[data-testid="canvas-workbench-shell"]') ||
            document.querySelector('[data-testid="canvas-workbench-layout"]'),
          ),
          hasHarnessPanel: Boolean(
            document.querySelector('[data-testid="harness-status-panel"]'),
          ),
          hasConversationShell: Boolean(
            document.querySelector('[data-testid="agent-chat-workspace"]') ||
            document.querySelector('[data-testid="chat-workspace"]') ||
            document.querySelector('[data-testid="message-list"]') ||
            document.querySelector('[data-testid="message-list-frame"]'),
          ),
          bodyText: text,
        };
      },
      {
        artifactPath: ARTIFACT_PATH,
        doneText: FINAL_DONE_TEXT,
        userPrompt: expectedUserPrompt(options),
        assistantArtifactText: ASSISTANT_ARTIFACT_TEXT,
      },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (hasHydratedSessionSnapshot(snapshot)) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `代码产物会话未在 GUI 中完成 hydrate: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function openWorkbench(page, options) {
  await waitForSessionHydrated(page, options);
  const existing = await evaluatePageSnapshot(page, () => {
    const text = document.body?.innerText || "";
    const isVisible = (element) => {
      if (!(element instanceof HTMLElement)) {
        return false;
      }
      const style = window.getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        rect.width > 0 &&
        rect.height > 0
      );
    };
    const canvasWorkbenchShell =
      document.querySelector('[data-testid="canvas-workbench-shell"]') ||
      document.querySelector('[data-testid="canvas-workbench-layout"]');
    const canvasWorkbenchPanel = document.querySelector(
      '[data-testid^="canvas-workbench-panel-"]',
    );
    const changesTab = document.querySelector(
      '[data-canvas-tab-key="changes"]',
    );
    return {
      hasCanvasWorkbenchShell: isVisible(canvasWorkbenchShell),
      hasCanvasWorkbenchPanel: isVisible(canvasWorkbenchPanel),
      hasChangesTab: isVisible(changesTab),
      bodyText: text,
    };
  });
  if (
    existing?.hasCanvasWorkbenchShell ||
    existing?.hasCanvasWorkbenchPanel ||
    existing?.hasChangesTab
  ) {
    return {
      clicked: { clicked: false, selector: "existing-canvas-workbench" },
      snapshot: {
        hasWorkbenchSidebar: false,
        hasHarnessPanel: false,
        hasArtifactWorkbenchShell: false,
        hasCanvasWorkbenchShell: existing.hasCanvasWorkbenchShell,
        hasCanvasWorkbenchPanel: existing.hasCanvasWorkbenchPanel,
        hasCurrentProgressTab: false,
        hasArtifactSummary: false,
        bodyText: existing.bodyText,
      },
    };
  }

  const clicked = await evaluatePageSnapshot(page, () => {
    const isVisible = (element) => {
      if (!(element instanceof HTMLElement)) {
        return false;
      }
      const style = window.getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        rect.width > 0 &&
        rect.height > 0
      );
    };
    const direct =
      [
        document.querySelector('[data-testid="task-center-workbench-toggle"]'),
        document.querySelector('[data-testid="task-center-tab-workbench"]'),
        document.querySelector(
          '[data-testid="theme-workbench-harness-toggle"]',
        ),
        document.querySelector('[data-testid="toggle-harness"]'),
      ].find(isVisible) || null;
    if (direct instanceof HTMLElement) {
      direct.click();
      return {
        clicked: true,
        selector:
          direct.getAttribute("data-testid") ||
          direct.getAttribute("aria-label") ||
          direct.textContent ||
          "workbench-toggle",
      };
    }
    const button = Array.from(document.querySelectorAll("button")).find(
      (candidate) => {
        if (!isVisible(candidate)) {
          return false;
        }
        const label = [
          candidate.getAttribute("title") || "",
          candidate.getAttribute("aria-label") || "",
          candidate.textContent || "",
        ].join("\n");
        return label.includes("工作台") || label.includes("Workbench");
      },
    );
    if (button) {
      button.click();
      return {
        clicked: true,
        selector:
          button.getAttribute("data-testid") ||
          button.getAttribute("aria-label") ||
          button.textContent ||
          "workbench-button",
      };
    }
    return { clicked: false, selector: null };
  });
  assert(clicked?.clicked, "未找到可点击的工作台入口");

  const startedAt = Date.now();
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => {
      const text = document.body?.innerText || "";
      const isVisible = (element) => {
        if (!(element instanceof HTMLElement)) {
          return false;
        }
        const style = window.getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          rect.width > 0 &&
          rect.height > 0
        );
      };
      return {
        hasWorkbenchSidebar: isVisible(
          document.querySelector('[data-testid="general-workbench-sidebar"]'),
        ),
        hasHarnessPanel: isVisible(
          document.querySelector('[data-testid="harness-status-panel"]'),
        ),
        hasArtifactWorkbenchShell: isVisible(
          document.querySelector('[data-testid="artifact-workbench-shell"]'),
        ),
        hasCanvasWorkbenchShell: isVisible(
          document.querySelector('[data-testid="canvas-workbench-shell"]') ||
            document.querySelector('[data-testid="canvas-workbench-layout"]'),
        ),
        hasCanvasWorkbenchPanel: isVisible(
          document.querySelector('[data-testid^="canvas-workbench-panel-"]'),
        ),
        hasCurrentProgressTab: text.includes("当前进展"),
        hasArtifactSummary: false,
        bodyText: text,
      };
    });
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    if (
      snapshot.hasWorkbenchSidebar ||
      snapshot.hasHarnessPanel ||
      snapshot.hasArtifactWorkbenchShell ||
      snapshot.hasCanvasWorkbenchShell ||
      snapshot.hasCanvasWorkbenchPanel ||
      snapshot.hasCurrentProgressTab
    ) {
      return {
        clicked,
        snapshot,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error("点击工作台后未出现工作台内容");
}

async function collectCodingWorkbenchGuiEvidence(
  page,
  options,
  { outputPreview = CODING_COMMAND_SUCCESS_PREVIEW } = {},
) {
  const tabEvidenceTimeoutMs = Math.min(
    options.timeoutMs,
    FINAL_PAGE_OPERATION_TIMEOUT_MS,
  );
  const tabs = [
    {
      key: "changes",
      panelTestId: "canvas-workbench-panel-changes",
      expectedTexts: [
        CODING_FILE_DISPLAY_PATH,
        "src/added.ts",
        "src/deleted.ts",
        "src/source.ts -> src/destination.ts",
      ],
    },
    {
      key: "outputs",
      panelTestId: "canvas-workbench-panel-outputs",
      expectedTexts: [CODING_COMMAND_TEXT, outputPreview, CODING_TEST_SUITE],
    },
    {
      key: "logs",
      panelTestId: "canvas-workbench-panel-logs",
      expectedTexts: [
        CODING_FILE_DISPLAY_PATH,
        CODING_COMMAND_TEXT,
        CODING_TEST_SUITE,
      ],
    },
  ];
  const evidence = {};

  for (const tab of tabs) {
    const clicked = await evaluatePageSnapshot(
      page,
      ({ key }) => {
        const isVisible = (element) => {
          if (!(element instanceof HTMLElement)) {
            return false;
          }
          const style = window.getComputedStyle(element);
          const rect = element.getBoundingClientRect();
          return (
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            rect.width > 0 &&
            rect.height > 0
          );
        };
        const visibleWorkbenchRoot = () => {
          const visibleShell = Array.from(
            document.querySelectorAll('[data-testid="canvas-workbench-shell"]'),
          ).find(isVisible);
          if (visibleShell instanceof HTMLElement) {
            return visibleShell;
          }
          const visibleLayout = Array.from(
            document.querySelectorAll(
              '[data-testid="canvas-workbench-layout"]',
            ),
          ).find(isVisible);
          if (!(visibleLayout instanceof HTMLElement)) {
            return null;
          }
          const shell = visibleLayout.closest(
            '[data-testid="canvas-workbench-shell"]',
          );
          return shell instanceof HTMLElement ? shell : visibleLayout;
        };
        const root = visibleWorkbenchRoot();
        if (!root) {
          return false;
        }
        const tabButton = Array.from(
          root.querySelectorAll(`[data-canvas-tab-key="${key}"]`),
        ).find(isVisible);
        if (tabButton instanceof HTMLElement) {
          tabButton.click();
          if (key === "changes") {
            const expandButton = Array.from(
              document.querySelectorAll(
                '[data-testid="file-changes-summary-toggle"]',
              ),
            ).find(
              (candidate) =>
                isVisible(candidate) &&
                candidate.getAttribute("aria-expanded") !== "true",
            );
            if (expandButton instanceof HTMLElement) {
              expandButton.click();
            }
          }
          return true;
        }
        return false;
      },
      { key: tab.key },
    );
    if (!clicked) {
      evidence[tab.key] = {
        clicked: false,
        panelVisible: false,
        expectedTextsPresent: false,
        bodyText: "",
      };
      continue;
    }

    const startedAt = Date.now();
    let lastSnapshot = null;
    while (Date.now() - startedAt < tabEvidenceTimeoutMs) {
      const snapshot = await evaluatePageSnapshot(
        page,
        ({ panelTestId, expectedTexts }) => {
          const isVisible = (element) => {
            if (!(element instanceof HTMLElement)) {
              return false;
            }
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return (
              style.display !== "none" &&
              style.visibility !== "hidden" &&
              rect.width > 0 &&
              rect.height > 0
            );
          };
          const visibleWorkbenchRoot = () => {
            const visibleShell = Array.from(
              document.querySelectorAll(
                '[data-testid="canvas-workbench-shell"]',
              ),
            ).find(isVisible);
            if (visibleShell instanceof HTMLElement) {
              return visibleShell;
            }
            const visibleLayout = Array.from(
              document.querySelectorAll(
                '[data-testid="canvas-workbench-layout"]',
              ),
            ).find(isVisible);
            if (!(visibleLayout instanceof HTMLElement)) {
              return null;
            }
            const shell = visibleLayout.closest(
              '[data-testid="canvas-workbench-shell"]',
            );
            return shell instanceof HTMLElement ? shell : visibleLayout;
          };
          const root = visibleWorkbenchRoot();
          if (!root) {
            return {
              clicked: true,
              panelVisible: false,
              expectedTexts: expectedTexts.map((text) => ({
                text,
                present: false,
              })),
              expectedTextsPresent: false,
              bodyText: document.body?.innerText || "",
            };
          }
          const panel = Array.from(
            root.querySelectorAll(`[data-testid="${panelTestId}"]`),
          ).find(isVisible);
          const panelText = panel?.textContent || "";
          const bodyText = document.body?.innerText || "";
          const evidenceText =
            panelTestId === "canvas-workbench-panel-changes"
              ? bodyText
              : panelText;
          return {
            clicked: true,
            panelVisible: Boolean(panel),
            expectedTexts: expectedTexts.map((text) => ({
              text,
              present: evidenceText.includes(text),
            })),
            expectedTextsPresent: expectedTexts.every((text) =>
              evidenceText.includes(text),
            ),
            bodyText,
          };
        },
        {
          panelTestId: tab.panelTestId,
          expectedTexts: tab.expectedTexts,
        },
      );
      if (!snapshot) {
        await sleep(options.intervalMs);
        continue;
      }
      lastSnapshot = snapshot;
      if (snapshot.panelVisible && snapshot.expectedTextsPresent) {
        break;
      }
      await sleep(options.intervalMs);
    }
    evidence[tab.key] = sanitizeJson(lastSnapshot);
  }

  return evidence;
}

async function waitForCodingRecoveryGuiTerminal(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(
      page,
      ({ outputPreview }) => {
        const bodyText = document.body?.innerText || "";
        const outputPanel = document.querySelector(
          '[data-testid="coding-workbench-output-projection"]',
        );
        const outputText = outputPanel?.textContent || "";
        const generating =
          bodyText.includes("正在生成回复") ||
          bodyText.includes("Generating response");
        return {
          generating,
          hasSuccessPreview:
            outputText.includes(outputPreview) ||
            bodyText.includes(outputPreview),
          hasRecoveryAssistantText: bodyText.includes(
            "已继续修复 coding-target，并通过 npm test -- coding-target。",
          ),
          outputPanelVisible: Boolean(outputPanel),
          bodyText,
        };
      },
      { outputPreview: CODING_COMMAND_SUCCESS_PREVIEW },
    );
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      !snapshot.generating &&
      snapshot.hasSuccessPreview &&
      snapshot.hasRecoveryAssistantText
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `恢复回合已在 App Server 完成，但 GUI 未收口到成功终态: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function clickCodingWorkbenchRecovery(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => {
      const isVisible = (element) => {
        if (!(element instanceof HTMLElement)) {
          return false;
        }
        const style = window.getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          rect.width > 0 &&
          rect.height > 0
        );
      };
      const visibleWorkbenchRoot = () => {
        const visibleShell = Array.from(
          document.querySelectorAll('[data-testid="canvas-workbench-shell"]'),
        ).find(isVisible);
        if (visibleShell instanceof HTMLElement) {
          return visibleShell;
        }
        const visibleLayout = Array.from(
          document.querySelectorAll('[data-testid="canvas-workbench-layout"]'),
        ).find(isVisible);
        if (!(visibleLayout instanceof HTMLElement)) {
          return null;
        }
        const shell = visibleLayout.closest(
          '[data-testid="canvas-workbench-shell"]',
        );
        return shell instanceof HTMLElement ? shell : visibleLayout;
      };
      const root = visibleWorkbenchRoot();
      if (!root) {
        return {
          outputTabClicked: false,
          outputPanelVisible: false,
          recoveryPanelVisible: false,
          buttonVisible: false,
          hasCommandText: false,
          hasFailurePreview: false,
          hasSuccessPreview: false,
          panelText: "",
        };
      }
      const outputTab = Array.from(
        root.querySelectorAll('[data-canvas-tab-key="outputs"]'),
      ).find(isVisible);
      if (outputTab instanceof HTMLElement) {
        outputTab.click();
      }
      const panel = Array.from(
        root.querySelectorAll('[data-testid="canvas-workbench-panel-outputs"]'),
      ).find(isVisible);
      const recoveryPanel = panel?.querySelector(
        '[data-testid="coding-workbench-recovery"]',
      );
      const button =
        panel?.querySelector(
          '[data-testid="coding-workbench-recovery-submit"]',
        ) ||
        Array.from(panel?.querySelectorAll("button") || []).find((candidate) =>
          (candidate.textContent || "").includes("继续修复"),
        );
      const buttonVisible = isVisible(button);
      const panelText = panel?.textContent || "";
      return {
        outputTabClicked: outputTab instanceof HTMLElement,
        outputPanelVisible: Boolean(panel),
        recoveryPanelVisible: isVisible(recoveryPanel),
        buttonVisible,
        hasCommandText: panelText.includes("npm test -- coding-target"),
        hasFailurePreview: panelText.includes("FAIL coding-target.test.ts"),
        hasSuccessPreview: panelText.includes("PASS coding-target.test.ts"),
        panelText,
      };
    });
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (
      snapshot.outputPanelVisible &&
      snapshot.recoveryPanelVisible &&
      snapshot.buttonVisible
    ) {
      break;
    }
    await sleep(options.intervalMs);
  }

  assert(
    lastSnapshot?.outputPanelVisible &&
      lastSnapshot?.recoveryPanelVisible &&
      lastSnapshot?.buttonVisible,
    `未找到可点击的继续修复入口: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );

  const clicked = await evaluatePageSnapshot(page, () => {
    const isVisible = (element) => {
      if (!(element instanceof HTMLElement)) {
        return false;
      }
      const style = window.getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        rect.width > 0 &&
        rect.height > 0
      );
    };
    const visibleWorkbenchRoot = () => {
      const visibleShell = Array.from(
        document.querySelectorAll('[data-testid="canvas-workbench-shell"]'),
      ).find(isVisible);
      if (visibleShell instanceof HTMLElement) {
        return visibleShell;
      }
      const visibleLayout = Array.from(
        document.querySelectorAll('[data-testid="canvas-workbench-layout"]'),
      ).find(isVisible);
      if (!(visibleLayout instanceof HTMLElement)) {
        return null;
      }
      const shell = visibleLayout.closest(
        '[data-testid="canvas-workbench-shell"]',
      );
      return shell instanceof HTMLElement ? shell : visibleLayout;
    };
    const root = visibleWorkbenchRoot();
    if (!root) {
      return false;
    }
    const panel = Array.from(
      root.querySelectorAll('[data-testid="canvas-workbench-panel-outputs"]'),
    ).find(isVisible);
    if (!panel) {
      return false;
    }
    const button =
      panel.querySelector('[data-testid="coding-workbench-recovery-submit"]') ||
      Array.from(panel.querySelectorAll("button")).find((candidate) =>
        (candidate.textContent || "").includes("继续修复"),
      );
    if (button instanceof HTMLElement) {
      button.click();
      return true;
    }
    return false;
  });
  assert(clicked, "继续修复按钮点击失败");

  const traceStartedAt = Date.now();
  let lastTrace = null;
  while (Date.now() - traceStartedAt < 30_000) {
    const traceRaw = await page.evaluate(() =>
      window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    );
    const traceMessages = readTraceMessages(traceRaw);
    const turnStartMessages = collectTraceJsonRpcMessages(traceMessages).filter(
      (message) => message.method === APP_SERVER_METHOD_TURN_START,
    );
    const recoveryTurnStart = findCodingRecoveryTurnStart(turnStartMessages);
    lastTrace = {
      traceCount: traceMessages.length,
      turnStartCount: turnStartMessages.length,
      recoveryTurnStart: recoveryTurnStart || null,
    };
    if (recoveryTurnStart) {
      const metadata = readTurnStartApplicationMetadata(recoveryTurnStart);
      const recovery = metadata.harness?.coding_workbench_recovery || null;
      return sanitizeJson({
        clicked: true,
        panel: lastSnapshot,
        inputText: readTurnStartInputText(recoveryTurnStart) || null,
        recovery,
      });
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `继续修复点击后未在 turn/start trace 中找到结构化 recovery metadata: ${JSON.stringify(
      sanitizeJson(lastTrace),
    )}`,
  );
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const backendLedgerEvidencePath = path.join(
    options.evidenceDir,
    `${options.prefix}-backend-ledger.json`,
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-workbench.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );

  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: {
      ...runtimeEnv.env,
      APP_SERVER_BIN: appServerBinary,
    },
  });
  const summary = {
    ok: false,
    scenarioId: "code-artifact-workbench-electron-fixture",
    scenario: options.scenario,
    prompt:
      options.scenario === "gui-coding-input" ? GUI_CODING_PROMPT : USER_PROMPT,
    sessionId: SESSION_ID,
    threadId: THREAD_ID,
    turnId: TURN_ID,
    workspaceId: null,
    workspace: null,
    artifactId: ARTIFACT_ID,
    artifactPath: ARTIFACT_PATH,
    appUrl: options.appUrl || null,
    checkedAt: new Date().toISOString(),
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    electronUserDataDir: options.keepTemp
      ? runtimeEnv.electronUserDataDir
      : null,
    backendPath: options.keepTemp ? runtimeEnv.backendPath : null,
    backendLedgerPath: options.keepTemp ? runtimeEnv.backendLedgerPath : null,
    backendLedger: backendLedgerEvidencePath,
    screenshot: null,
    consoleErrors: [],
    pageErrors: [],
    rendererSnapshot: null,
    initialize: null,
    guiWorkspaceBinding: null,
    guiWorkspaceNavigation: null,
    guiCodingInput: null,
    sessionCreation: null,
    sessionListVisibility: null,
    guiSessionVisible: null,
    guiSessionDirectOpen: null,
    guiSessionOpenAfterInput: null,
    sessionHydrated: null,
    workbench: null,
    codingWorkbenchGuiEvidence: null,
    codingRecoveryEvidence: null,
    assertions: {},
    summary: summaryPath,
  };

  let app = null;
  let page = null;
  const consoleErrors = [];
  const pageErrors = [];

  try {
    if (options.appUrl) {
      logStage("wait-app-url");
      summary.rendererDevServer = sanitizeJson(
        await waitForAppUrlReady(options),
      );
    }

    logStage("launch-electron");
    app = await electron.launch({
      executablePath: electronPath,
      args: ["--use-mock-keychain", "."],
      cwd: process.cwd(),
      env: {
        ...runtimeEnv.env,
        ...appServerEnv,
        APP_SERVER_BACKEND_MODE: "external",
        APP_SERVER_BACKEND_COMMAND: process.execPath,
        APP_SERVER_BACKEND_ARGS: JSON.stringify([
          runtimeEnv.backendPath,
          runtimeEnv.backendLedgerPath,
        ]),
        APP_SERVER_BACKEND_TIMEOUT_MS: "10000",
        CODE_ARTIFACT_WORKBENCH_FIXTURE_SCENARIO: options.scenario,
        ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
        LIME_ELECTRON_E2E: "1",
        LIME_ELECTRON_BRAND_DEV_APP: "0",
        LIME_ELECTRON_CLEAR_RENDERER_CACHE: "0",
        LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
        ...(options.appUrl ? { VITE_DEV_SERVER_URL: options.appUrl } : {}),
      },
      timeout: options.timeoutMs,
    });

    app.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(sanitizeText(message.text()));
      }
    });

    page = await app.firstWindow({ timeout: options.timeoutMs });
    page.on("pageerror", (error) => {
      pageErrors.push(sanitizeText(error?.message || String(error)));
    });
    page.setDefaultTimeout(options.timeoutMs);
    await page.setViewportSize({ width: 1440, height: 1000 });

    logStage("wait-renderer");
    const rendererSnapshot = await waitForRendererReady(
      page,
      options,
      (snapshot) => {
        summary.rendererSnapshot = sanitizeJson(snapshot);
      },
    );
    summary.rendererSnapshot = sanitizeJson(rendererSnapshot);
    await clearInvokeBuffers(page);

    logStage("initialize-app-server");
    summary.initialize = sanitizeJson(await initializeAppServer(page));

    logStage("ensure-default-workspace");
    const workspace = await ensureDefaultWorkspace(page);
    summary.workspaceId = workspace.workspaceId;
    summary.workspace = sanitizeJson(workspace);

    let sessionCreation = null;

    if (
      options.scenario === "gui-coding-input" ||
      options.scenario === FILE_CHANGE_BATCH_SCENARIO
    ) {
      logStage("start-empty-code-artifact-session");
      const requests = [];
      const startedSession = await startCodeArtifactSession(
        page,
        workspace,
        requests,
      );
      sessionCreation = {
        session: startedSession.session,
        turn: null,
        sessionId: startedSession.sessionId,
        threadId: startedSession.threadId,
        turnId: null,
        read: null,
        requests,
      };
    } else {
      logStage("create-code-artifact-session");
      sessionCreation = await createCodeArtifactSession(
        page,
        options,
        workspace,
      );
    }
    summary.sessionId = SESSION_ID;
    summary.threadId = THREAD_ID;
    summary.turnId = TURN_ID;

    logStage("bind-gui-workspace");
    summary.guiWorkspaceBinding = sanitizeJson(
      await bindGuiWorkspaceToFixture(page, workspace.workspaceId),
    );

    logStage("verify-session-list");
    const unscopedList = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_THREAD_LIST,
      {
        archived: false,
        limit: 20,
      },
    );
    const workspaceList = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_THREAD_LIST,
      {
        archived: false,
        cwd: workspace.rootPath,
        limit: 20,
      },
    );
    summary.sessionListVisibility = sanitizeJson({
      unscoped: summarizeListVisibility(unscopedList),
      workspace: summarizeListVisibility(workspaceList),
    });

    logStage("navigate-gui-workspace");
    summary.guiWorkspaceNavigation = sanitizeJson(
      await navigateGuiToWorkspaceScopedAgent(
        page,
        options,
        workspace.workspaceId,
      ),
    );

    logStage("wait-session-visible-in-sidebar");
    summary.guiSessionVisible = sanitizeJson(
      await waitForGuiSessionVisible(page, options),
    );

    logStage("open-session-from-sidebar");
    summary.guiSessionDirectOpenClick = sanitizeJson(
      await openFixtureSessionFromSidebar(page, options),
    );

    logStage("wait-session-opened-from-sidebar");
    summary.guiSessionDirectOpen = sanitizeJson(
      await waitForFixtureSessionOpenedFromSidebar(page, options),
    );

    summary.invokeBuffersClearedBeforeScenario = sanitizeJson(
      await clearInvokeBuffers(page),
    );

    if (options.scenario === FILE_CHANGE_BATCH_SCENARIO) {
      summary.fileChangeBatchLifecycleTraceEnabled = sanitizeJson(
        await page.evaluate(() => {
          window.localStorage.setItem("lime:debug:claw-trace-enabled:v1", "on");
          window.localStorage.removeItem(
            "lime:debug:app-server-server-request-lifecycle:v1",
          );
          return {
            enabled:
              window.localStorage.getItem(
                "lime:debug:claw-trace-enabled:v1",
              ) === "on",
            lifecycleCleared: true,
          };
        }),
      );
      const fileChangeOptions = {
        ...options,
        backendLedgerPath: runtimeEnv.backendLedgerPath,
      };
      logStage("file-change-batch-decline");
      summary.fileChangeBatchDecline = await startFileChangeBatchTurnFromGui(
        page,
        fileChangeOptions,
        FILE_CHANGE_BATCH_DECLINE_PROMPT,
        "decline",
      );
      logStage("file-change-batch-cancel");
      summary.fileChangeBatchCancel = await startFileChangeBatchTurnFromGui(
        page,
        fileChangeOptions,
        FILE_CHANGE_BATCH_CANCEL_PROMPT,
        "cancel",
      );
      const finalRead = await invokeAppServerFromPage(
        page,
        APP_SERVER_METHOD_THREAD_READ,
        { threadId: THREAD_ID, includeTurns: true },
      );
      sessionCreation.read = finalRead.result;
      sessionCreation.turnId = summary.fileChangeBatchDecline.identity.turnId;
      summary.fileChangeBatchFinalRead = sanitizeJson(finalRead.result);
    } else if (options.scenario === "gui-coding-input") {
      logStage("send-coding-prompt-from-gui");
      summary.guiCodingInput = sanitizeJson(
        await sendPromptFromGui(page, options, GUI_CODING_PROMPT),
      );
      logStage("wait-gui-coding-read-model");
      sessionCreation.read = await waitForCodeArtifactReadModel(page, options, {
        requests: sessionCreation.requests,
        timeoutMs: options.timeoutMs,
      });
      logStage("open-session-after-gui-coding-input");
      summary.guiSessionOpenAfterInputClick = sanitizeJson(
        await openFixtureSessionFromSidebar(page, options),
      );
      summary.guiSessionOpenAfterInput = sanitizeJson(
        await waitForFixtureSessionOpenedFromSidebar(page, options),
      );
    }

    const canonicalTurnId =
      sessionCreation.turnId || readLatestThreadTurnId(sessionCreation.read);
    if (canonicalTurnId) {
      TURN_ID = canonicalTurnId;
    }
    summary.sessionCreation = sanitizeJson({
      ...summarizeCodeArtifactRead(
        sessionCreation.read,
        sessionCreation.requests,
      ),
      sessionId: sessionCreation.sessionId,
      threadId: sessionCreation.threadId,
      turnId: canonicalTurnId,
      artifactWrite: sessionCreation.artifactWrite ?? null,
      artifactWriteIpcTrace: sessionCreation.artifactWriteIpcTrace ?? null,
      guiPromptSubmitted:
        options.scenario === "gui-coding-input"
          ? summary.guiCodingInput?.clicked?.clicked === true &&
            summary.guiCodingInput?.afterFill?.promptVisibleInTextarea === true
          : null,
    });

    if (options.scenario === FILE_CHANGE_BATCH_SCENARIO) {
      const declineValidation =
        summary.fileChangeBatchDecline?.terminal?.validation;
      const cancelValidation =
        summary.fileChangeBatchCancel?.terminal?.validation;
      const lifecycleEntries = await page.evaluate(() => {
        try {
          const parsed = JSON.parse(
            window.localStorage.getItem(
              "lime:debug:app-server-server-request-lifecycle:v1",
            ) || "[]",
          );
          return Array.isArray(parsed) ? parsed : [];
        } catch {
          return [];
        }
      });
      const traceRaw = await page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      );
      const errorRaw = await page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      );
      const traceMessages = readTraceMessages(traceRaw);
      const fileChangeLifecycle = [
        summary.fileChangeBatchDecline,
        summary.fileChangeBatchCancel,
      ].map((entry) => {
        const request = lifecycleEntries.find(
          (candidate) =>
            candidate?.kind === "request" &&
            candidate?.method === "item/fileChange/requestApproval" &&
            candidate?.threadId === entry?.identity?.threadId &&
            candidate?.turnId === entry?.identity?.turnId &&
            candidate?.itemId === entry?.identity?.itemId,
        );
        const response = lifecycleEntries.find(
          (candidate) =>
            candidate?.kind === "response" && candidate?.id === request?.id,
        );
        const resolved = lifecycleEntries.find(
          (candidate) =>
            candidate?.kind === "resolved" && candidate?.id === request?.id,
        );
        return { request, response, resolved };
      });
      const backendLedger = readJsonl(runtimeEnv.backendLedgerPath);
      const backendDecisions = backendLedger
        .filter(
          (entry) =>
            entry?.kind === "fileChangeGateBEvents" &&
            ["decline", "cancel"].includes(entry?.decision),
        )
        .map((entry) => entry.decision);
      const cancelBackendEntry = backendLedger.find(
        (entry) =>
          entry?.kind === "fileChangeGateBEvents" &&
          entry?.decision === "cancel",
      );
      const electronIpcTrace = traceMessages.filter(
        (entry) =>
          entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
          entry?.transport === "electron-ipc",
      );
      summary.fileChangeBatchServerRequestLifecycle =
        sanitizeJson(fileChangeLifecycle);
      summary.fileChangeBatchBackendDecisions = backendDecisions;
      summary.fileChangeBatchTraceMethods =
        collectTraceRequestMethods(traceMessages);
      summary.assertions = {
        electronPreloadBridge: rendererSnapshot.electron === true,
        electronIpcAppServerBridge: electronIpcTrace.length > 0,
        typedFileChangeServerRequests: fileChangeLifecycle.every(
          ({ request, response, resolved }, index) =>
            request?.method === "item/fileChange/requestApproval" &&
            response?.decision === (index === 0 ? "decline" : "cancel") &&
            resolved?.id === request.id,
        ),
        backendDecisionsReceived:
          backendDecisions.includes("decline") &&
          backendDecisions.includes("cancel"),
        declineExactBatch: declineValidation?.valid === true,
        declineTurnCompleted:
          summary.fileChangeBatchDecline?.terminal?.turnStatus === "completed",
        declineFileStatus:
          summary.fileChangeBatchDecline?.terminal?.expectedStatus ===
          "declined",
        cancelExactBatch: cancelValidation?.valid === true,
        cancelTurnInterrupted:
          summary.fileChangeBatchCancel?.terminal?.turnStatus === "interrupted",
        cancelFileStatus:
          summary.fileChangeBatchCancel?.terminal?.expectedStatus ===
          "inProgress",
        cancelDoesNotSynthesizeFileTerminal:
          !cancelBackendEntry?.eventTypes?.some((eventType) =>
            ["patch.applied", "patch.declined", "patch.failed"].includes(
              eventType,
            ),
          ),
        terminalGuiExactBatch:
          summary.fileChangeBatchDecline?.terminalGui?.rowCount === 4 &&
          summary.fileChangeBatchDecline?.terminalGui?.status === "declined" &&
          summary.fileChangeBatchCancel?.terminalGui?.rowCount === 4 &&
          summary.fileChangeBatchCancel?.terminalGui?.status === "inProgress",
        pendingCleared:
          summary.fileChangeBatchDecline?.terminal?.unexpectedPendingItemIds
            ?.length === 0 &&
          summary.fileChangeBatchCancel?.terminal?.unexpectedPendingItemIds
            ?.length === 0,
        noInvokeErrors: !errorRaw,
      };
      for (const [key, passed] of Object.entries(summary.assertions)) {
        assert(passed, `断言失败: ${key}`);
      }
      summary.consoleErrors = consoleErrors;
      summary.pageErrors = pageErrors;
      assertNoRendererErrors(consoleErrors, pageErrors);
      summary.ok = true;
      summary.completedAt = new Date().toISOString();
      await withTimeout(
        page.screenshot({ path: screenshotPath, fullPage: true }),
        FINAL_PAGE_OPERATION_TIMEOUT_MS,
        "保存 FileChange Gate B 截图",
      );
      summary.screenshot = screenshotPath;
      writeJsonFile(summaryPath, summary);
      console.log(`${LOG_PREFIX} summary=${summaryPath}`);
      console.log(`${LOG_PREFIX} pass session=${SESSION_ID}`);
      return;
    }

    logStage("wait-session-hydrated");
    const sessionHydrated = await waitForSessionHydrated(page, options);
    summary.sessionHydrated = sanitizeJson(sessionHydrated);

    logStage("inspect-historical-timeline-summary");
    summary.timelineProcessEvidence = sanitizeJson(
      await inspectHistoricalTimelineSummary(page, options),
    );

    logStage("open-workbench");
    const workbench = await openWorkbench(page, options);
    summary.workbench = sanitizeJson(workbench);

    logStage("collect-coding-workbench-gui-evidence");
    summary.codingWorkbenchGuiEvidence = sanitizeJson(
      await collectCodingWorkbenchGuiEvidence(page, options, {
        outputPreview:
          options.scenario === "gui-coding-input"
            ? CODING_COMMAND_FAILURE_PREVIEW
            : CODING_COMMAND_SUCCESS_PREVIEW,
      }),
    );

    if (options.scenario === "gui-coding-input") {
      logStage("click-coding-workbench-recovery");
      summary.codingRecoveryEvidence = sanitizeJson(
        await clickCodingWorkbenchRecovery(page, options),
      );
      logStage("wait-recovery-read-model");
      summary.recoveryRead = sanitizeJson(
        await waitForCodeArtifactReadModel(page, options, {
          requests: sessionCreation.requests,
          timeoutMs: options.timeoutMs,
          requireCodingSuccess: true,
        }),
      );
      logStage("wait-gui-recovery-terminal");
      summary.guiRecoveryTerminal = sanitizeJson(
        await waitForCodingRecoveryGuiTerminal(page, options),
      );
      logStage("verify-session-after-recovery");
      summary.guiSessionOpenAfterRecovery = sanitizeJson(
        await waitForFixtureSessionOpenedFromSidebar(page, options),
      );
      logStage("open-workbench-after-recovery");
      summary.workbenchAfterRecovery = sanitizeJson(
        await openWorkbench(page, options),
      );
      logStage("collect-coding-workbench-gui-evidence-after-recovery");
      summary.codingWorkbenchGuiEvidenceAfterRecovery = sanitizeJson(
        await collectCodingWorkbenchGuiEvidence(page, options, {
          outputPreview: CODING_COMMAND_SUCCESS_PREVIEW,
        }),
      );
    }

    const backendLedgerEvidence = persistBackendLedgerEvidence(
      runtimeEnv.backendLedgerPath,
      backendLedgerEvidencePath,
    );
    if (backendLedgerEvidence.readError) {
      throw new Error(
        `读取 backend ledger 失败: ${backendLedgerEvidence.readError}`,
      );
    }
    const { backendLedger } = backendLedgerEvidence;
    const pageText = await withTimeout(
      page.evaluate(() => document.body?.innerText || ""),
      FINAL_PAGE_OPERATION_TIMEOUT_MS,
      "读取页面正文",
    );
    const traceRaw = await withTimeout(
      page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      ),
      FINAL_PAGE_OPERATION_TIMEOUT_MS,
      "读取 invoke trace",
    );
    const errorRaw = await withTimeout(
      page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      ),
      FINAL_PAGE_OPERATION_TIMEOUT_MS,
      "读取 invoke error",
    );
    const traceMessages = readTraceMessages(traceRaw);
    const appServerRequestMethods = Array.from(
      new Set(
        [
          ...(summary.sessionCreation?.requestMethods ?? []),
          ...collectTraceRequestMethods(traceMessages),
        ].filter(Boolean),
      ),
    );
    const backendRecoveryTurnStart = backendLedger.find(
      (entry) =>
        entry.kind === "turnStart" &&
        isCodingRecoveryPromptText(entry.inputText),
    );
    const backendRecoveryEvents = backendLedger.find(
      (entry) =>
        entry.kind === "backendEvents" &&
        entry.turnId === backendRecoveryTurnStart?.turnId,
    );
    const recoveryExecutionIds = Object.values(
      backendRecoveryEvents?.executionIds || {},
    ).filter((value) => typeof value === "string" && value.trim());
    const traceRecoveryTurnStart = findCodingRecoveryTurnStart(
      collectTraceJsonRpcMessages(traceMessages),
    );
    const traceRecoveryMetadata = readTurnStartApplicationMetadata(
      traceRecoveryTurnStart,
    );
    const traceRecoveryContext =
      traceRecoveryMetadata.harness?.coding_workbench_recovery || null;
    const guiRecoveryContext = summary.codingRecoveryEvidence?.recovery || null;
    const guiRecoveryCommandId = String(
      guiRecoveryContext?.sourceIds?.commandId || "",
    ).trim();
    const guiRecoveryTestRunId = String(
      guiRecoveryContext?.sourceIds?.testRunId || "",
    ).trim();
    const traceRecoveryThreadId = String(
      traceRecoveryTurnStart?.params?.threadId || "",
    ).trim();
    const backendTurnStartObserved = backendLedger.some(
      (entry) => entry.kind === "turnStart",
    );
    const backendEmittedEventTypes = Array.from(
      new Set(
        backendLedger
          .filter((entry) => entry.kind === "backendEvents")
          .flatMap((entry) =>
            Array.isArray(entry.eventTypes) ? entry.eventTypes : [],
          )
          .filter((type) => typeof type === "string" && type.trim()),
      ),
    );
    const appServerJsonRpcObserved =
      appServerRequestMethods.includes(APP_SERVER_METHOD_TURN_START) ||
      backendTurnStartObserved;
    const historicalOperationalDetailsHidden =
      hasHistoricalOperationalDetailsHidden(summary.timelineProcessEvidence);
    const codingOutputsSnapshot =
      options.scenario === "gui-coding-input"
        ? summary.codingWorkbenchGuiEvidenceAfterRecovery?.outputs
        : summary.codingWorkbenchGuiEvidence?.outputs;
    const codingOutputsBodyText = codingOutputsSnapshot?.bodyText || "";
    const codingOutputsVisibleEvidence =
      codingOutputsSnapshot?.panelVisible === true &&
      codingOutputsSnapshot?.expectedTextsPresent === true;
    const codingOutputsRecoveredBodyEvidence =
      options.scenario === "gui-coding-input" &&
      codingOutputsBodyText.includes(CODING_FILE_PATH) &&
      codingOutputsBodyText.includes(CODING_FILE_PREVIEW);
    const recoveryTurns = Array.isArray(summary.recoveryRead?.thread?.turns)
      ? summary.recoveryRead.thread.turns
      : [];
    const canonicalItemLifecycleClean =
      recoveryTurns.length >= 2 &&
      recoveryTurns.every((turn) => {
        const items = Array.isArray(turn?.items) ? turn.items : [];
        const itemIds = items
          .map((item) => String(item?.id || "").trim())
          .filter(Boolean);
        return (
          turn?.status === "completed" &&
          items.filter((item) => item?.type === "agentMessage").length === 1 &&
          new Set(itemIds).size === itemIds.length &&
          items.every((item) => item?.status !== "inProgress")
        );
      });
    const assertions = {
      electronPreloadBridge: rendererSnapshot.electron === true,
      appServerJsonRpcUsed: appServerJsonRpcObserved,
      externalFixtureBackendUsed: backendTurnStartObserved,
      canonicalSessionIdentity:
        summary.sessionCreation?.sessionId === SESSION_ID &&
        summary.sessionCreation?.threadId === THREAD_ID &&
        typeof summary.sessionCreation?.turnId === "string" &&
        summary.sessionCreation.turnId.length > 0,
      liveProviderNotUsed: backendLedger.every(
        (entry) =>
          entry.kind !== "turnStart" ||
          entry.requestMetadata?.harness?.coding_workbench_recovery
            ?.schemaVersion === "coding-workbench-recovery/v1" ||
          ((!entry.providerPreference ||
            entry.providerPreference === "fixture-provider") &&
            (!entry.modelPreference ||
              entry.modelPreference.trim().length > 0)),
      ),
      artifactPersisted:
        summary.sessionCreation?.artifactProjectionPersisted === true,
      typedArtifactWriteUsed:
        options.scenario !== "direct-session" ||
        (summary.sessionCreation?.artifactWrite?.threadId === THREAD_ID &&
          summary.sessionCreation?.artifactWrite?.turnId === TURN_ID &&
          summary.sessionCreation?.artifactWrite?.artifactRef === ARTIFACT_ID &&
          typeof summary.sessionCreation?.artifactWrite?.eventId === "string" &&
          summary.sessionCreation.artifactWrite.eventId.length > 0 &&
          summary.sessionCreation?.artifactWrite?.sidecar?.contentStatus ===
            "available"),
      typedArtifactWriteElectronIpcTrace:
        options.scenario !== "direct-session" ||
        (summary.sessionCreation?.artifactWriteIpcTrace?.command ===
          APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
          summary.sessionCreation?.artifactWriteIpcTrace?.transport ===
            "electron-ipc" &&
          summary.sessionCreation?.artifactWriteIpcTrace?.status ===
            "success" &&
          summary.sessionCreation?.artifactWriteIpcTrace?.method ===
            APP_SERVER_METHOD_ARTIFACT_WRITE &&
          summary.sessionCreation?.artifactWriteIpcTrace?.threadId ===
            THREAD_ID &&
          summary.sessionCreation?.artifactWriteIpcTrace?.turnId === TURN_ID &&
          summary.sessionCreation?.artifactWriteIpcTrace?.artifactRef ===
            ARTIFACT_ID),
      directSessionBackendDidNotInjectArtifact:
        options.scenario !== "direct-session" ||
        !backendEmittedEventTypes.includes("artifact.snapshot"),
      toolTimelinePersisted:
        summary.sessionCreation?.toolTimelineProjectionPersisted === true,
      codingProjectionPersisted:
        summary.sessionCreation?.codingProjectionPersisted === true,
      codexFileChangeBatchPersisted:
        summary.sessionCreation?.codexFileChangeBatchPersisted === true,
      guiHydratedSession: hasHydratedSessionSnapshot(summary.sessionHydrated),
      workbenchOpened:
        summary.workbench?.snapshot?.hasWorkbenchSidebar === true ||
        summary.workbench?.snapshot?.hasHarnessPanel === true ||
        summary.workbench?.snapshot?.hasArtifactWorkbenchShell === true ||
        summary.workbench?.snapshot?.hasCanvasWorkbenchShell === true ||
        summary.workbench?.snapshot?.hasCanvasWorkbenchPanel === true ||
        summary.workbench?.snapshot?.hasCurrentProgressTab === true ||
        summary.workbench?.snapshot?.hasArtifactSummary === true,
      pageMentionsArtifactOrCode:
        pageText.includes(ARTIFACT_PATH) ||
        pageText.includes("Hello Lime Workbench") ||
        pageText.includes("产物"),
      historicalOperationalDetailsHidden,
      codingChangesEvidencePresent:
        summary.codingWorkbenchGuiEvidence?.changes?.panelVisible === true &&
        summary.codingWorkbenchGuiEvidence?.changes?.expectedTextsPresent ===
          true,
      codingOutputsEvidencePresent:
        options.scenario === "direct-session"
          ? summary.sessionCreation?.codingProjectionPersisted === true &&
            historicalOperationalDetailsHidden
          : codingOutputsVisibleEvidence || codingOutputsRecoveredBodyEvidence,
      codingLogsEvidencePresent:
        options.scenario === "direct-session"
          ? historicalOperationalDetailsHidden
          : summary.codingWorkbenchGuiEvidence?.logs?.panelVisible === true &&
            summary.codingWorkbenchGuiEvidence?.logs?.expectedTextsPresent ===
              true,
      codingRecoveryGuiSubmitted:
        options.scenario !== "gui-coding-input" ||
        (summary.codingRecoveryEvidence?.clicked === true &&
          guiRecoveryContext?.schemaVersion ===
            "coding-workbench-recovery/v1" &&
          guiRecoveryCommandId.length > 0 &&
          guiRecoveryTestRunId.length > 0),
      codingRecoveryReachedBackend:
        options.scenario !== "gui-coding-input" ||
        (isCodingRecoveryPromptText(backendRecoveryTurnStart?.inputText) &&
          backendRecoveryTurnStart?.requestMetadata?.harness
            ?.coding_workbench_recovery?.schemaVersion ===
            "coding-workbench-recovery/v1" &&
          backendRecoveryTurnStart?.requestMetadata?.harness
            ?.coding_workbench_recovery?.sourceIds?.commandId ===
            guiRecoveryCommandId &&
          backendRecoveryTurnStart?.requestMetadata?.harness
            ?.coding_workbench_recovery?.sourceIds?.testRunId ===
            guiRecoveryTestRunId),
      codingRecoveryTraceWire:
        options.scenario !== "gui-coding-input" ||
        (isCodingRecoveryPromptText(
          readTurnStartInputText(traceRecoveryTurnStart),
        ) &&
          traceRecoveryContext?.schemaVersion ===
            "coding-workbench-recovery/v1" &&
          traceRecoveryContext?.sourceIds?.commandId === guiRecoveryCommandId &&
          traceRecoveryContext?.sourceIds?.testRunId === guiRecoveryTestRunId),
      codingRecoveryReadCompleted:
        options.scenario !== "gui-coding-input" ||
        (hasCodingSuccessProjection(summary.recoveryRead) &&
          readLatestThreadTurnId(summary.recoveryRead) ===
            backendRecoveryTurnStart?.turnId &&
          readLatestThreadTurnId(summary.recoveryRead) !==
            summary.sessionCreation?.turnId),
      codingRecoveryCanonicalIdentity:
        options.scenario !== "gui-coding-input" ||
        (backendRecoveryTurnStart?.sessionId === SESSION_ID &&
          backendRecoveryTurnStart?.threadId === THREAD_ID &&
          typeof backendRecoveryTurnStart?.turnId === "string" &&
          backendRecoveryTurnStart.turnId.length > 0 &&
          backendRecoveryEvents?.sessionId === SESSION_ID &&
          backendRecoveryEvents?.threadId === THREAD_ID &&
          backendRecoveryEvents?.turnId === backendRecoveryTurnStart.turnId),
      codingRecoveryTraceThreadIdentity:
        options.scenario !== "gui-coding-input" ||
        traceRecoveryThreadId === THREAD_ID,
      canonicalItemLifecycleClean:
        options.scenario !== "gui-coding-input" || canonicalItemLifecycleClean,
      recoveryExecutionIdsTurnScoped:
        options.scenario !== "gui-coding-input" ||
        (typeof backendRecoveryTurnStart?.turnId === "string" &&
          backendRecoveryTurnStart.turnId.length > 0 &&
          recoveryExecutionIds.length === 4 &&
          recoveryExecutionIds.every((executionId) =>
            executionId.endsWith(`:${backendRecoveryTurnStart.turnId}`),
          )),
      guiPromptSubmitted:
        options.scenario !== "gui-coding-input" ||
        summary.sessionCreation?.guiPromptSubmitted === true,
      backendEmittedCurrentTerminal:
        backendEmittedEventTypes.includes("turn.completed"),
      backendDidNotEmitLegacyTerminal: !backendEmittedEventTypes.some((type) =>
        [
          "done",
          "final_done",
          "cancelled",
          "turn.done",
          "turn.final_done",
          "turn.cancelled",
        ].includes(type),
      ),
      noInvokeErrors: !errorRaw,
    };

    assertNoRendererErrors(consoleErrors, pageErrors);

    for (const [key, passed] of Object.entries(assertions)) {
      assert(passed, `断言失败: ${key}`);
    }

    await withTimeout(
      page.screenshot({ path: screenshotPath, fullPage: true }),
      FINAL_PAGE_OPERATION_TIMEOUT_MS,
      "保存成功截图",
    );
    summary.screenshot = screenshotPath;
    summary.consoleErrors = consoleErrors;
    summary.pageErrors = pageErrors;
    summary.appServerRequestMethods = appServerRequestMethods;
    summary.backendKinds = backendLedger.map((entry) => entry.kind);
    summary.backendEmittedEventTypes = backendEmittedEventTypes;
    summary.traceRecoveryTurnStart = sanitizeJson(traceRecoveryTurnStart);
    summary.historicalOperationalDetailsHidden =
      historicalOperationalDetailsHidden;
    summary.assertions = assertions;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
    console.log(`${LOG_PREFIX} pass session=${SESSION_ID}`);
  } catch (error) {
    try {
      const backendLedgerEvidence = persistBackendLedgerEvidence(
        runtimeEnv.backendLedgerPath,
        backendLedgerEvidencePath,
      );
      summary.backendKinds = backendLedgerEvidence.backendLedger.map(
        (entry) => entry.kind,
      );
      summary.backendEmittedEventTypes = Array.from(
        new Set(
          backendLedgerEvidence.backendLedger
            .filter((entry) => entry.kind === "backendEvents")
            .flatMap((entry) =>
              Array.isArray(entry.eventTypes) ? entry.eventTypes : [],
            ),
        ),
      );
      if (backendLedgerEvidence.readError) {
        summary.backendLedgerReadError = backendLedgerEvidence.readError;
      }
    } catch (backendLedgerError) {
      summary.backendLedgerEvidenceError = sanitizeText(backendLedgerError);
    }
    try {
      if (page) {
        const traceRaw = await withTimeout(
          page.evaluate(() =>
            window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
          ),
          FINAL_PAGE_OPERATION_TIMEOUT_MS,
          "失败后读取 invoke trace",
        );
        const errorRaw = await withTimeout(
          page.evaluate(() =>
            window.localStorage.getItem("lime_invoke_error_buffer_v1"),
          ),
          FINAL_PAGE_OPERATION_TIMEOUT_MS,
          "失败后读取 invoke error",
        );
        summary.invokeTrace = sanitizeJson(
          (() => {
            try {
              return JSON.parse(traceRaw || "[]");
            } catch {
              return traceRaw;
            }
          })(),
        );
        summary.invokeErrors = sanitizeJson(
          (() => {
            try {
              return JSON.parse(errorRaw || "[]");
            } catch {
              return errorRaw;
            }
          })(),
        );
      }
    } catch (traceError) {
      summary.invokeTraceError = sanitizeText(traceError);
    }
    summary.error = sanitizeText(
      error instanceof Error ? error.stack || error.message : String(error),
    );
    summary.consoleErrors = consoleErrors;
    summary.pageErrors = pageErrors;
    try {
      if (page) {
        await withTimeout(
          page.screenshot({ path: failureScreenshotPath, fullPage: true }),
          FINAL_PAGE_OPERATION_TIMEOUT_MS,
          "保存失败截图",
        );
        summary.screenshot = failureScreenshotPath;
      }
    } catch (screenshotError) {
      summary.screenshotError = sanitizeText(screenshotError);
    }
    writeJsonFile(summaryPath, summary);
    console.error(summary.error);
    console.error(`${LOG_PREFIX} failureSummary=${summaryPath}`);
    process.exitCode = 1;
  } finally {
    if (app) {
      try {
        await withTimeout(
          app.close(),
          FINAL_PAGE_OPERATION_TIMEOUT_MS,
          "关闭 Electron",
        );
      } catch (closeError) {
        console.warn(
          `${LOG_PREFIX} electron close skipped error=${sanitizeText(closeError)}`,
        );
        try {
          const childProcess =
            typeof app.process === "function" ? app.process() : null;
          if (childProcess && !childProcess.killed) {
            childProcess.kill("SIGTERM");
          }
        } catch (killError) {
          console.warn(
            `${LOG_PREFIX} electron kill skipped error=${sanitizeText(killError)}`,
          );
        }
      }
    }
    if (!options.keepTemp) {
      cleanupTempRoot(runtimeEnv.tempRoot);
    }
  }
}

await run();
