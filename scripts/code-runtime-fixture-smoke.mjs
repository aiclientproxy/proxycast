#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  assertSmoke,
  invokeDevBridge,
  sleep,
  summarizeEvidencePack,
  summarizeThreadRead,
  threadSettled,
  waitForHealth,
} from "./lib/agent-runtime-smoke-core.mjs";
import {
  fixtureChatRequestCount,
  workspaceIdFromDefaultProject,
  workspaceRootFromDefaultProject,
} from "./lib/agent-runtime-smoke-support.mjs";
import {
  assertLiveProviderSmokeAllowed,
  liveProviderSmokeAllowed,
} from "./lib/live-provider-smoke-gate.mjs";
import { startOpenAiCompatibleFixtureServer } from "./lib/openai-compatible-fixture-server.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const DEFAULT_OUTPUT = path.join(
  rootDir,
  ".lime/qc/code-runtime-fixture-smoke.json",
);
const DEFAULT_HEALTH_URL = "http://127.0.0.1:3030/health";
const DEFAULT_INVOKE_URL = "http://127.0.0.1:3030/invoke";
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_INTERVAL_MS = 1_000;
const LOG_PREFIX = "[smoke:code-runtime-fixture]";
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_METHOD_THREAD_START = "thread/start";
const APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE = "thread/settings/update";
const APP_SERVER_METHOD_TURN_START = "turn/start";
const APP_SERVER_METHOD_THREAD_READ = "thread/read";
const APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST =
  "agentSession/fileCheckpoint/list";
const APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF =
  "agentSession/fileCheckpoint/diff";
const APP_SERVER_METHOD_EVIDENCE_EXPORT = "evidence/export";
const FIXTURE_RELATIVE_PATH = ".lime/qc/code-runtime-fixture/src/greeting.ts";
const INITIAL_CONTENT = [
  "export function greeting() {",
  "  return 'Hello from initial fixture';",
  "}",
  "",
].join("\n");
const UPDATED_CONTENT = [
  "export function greeting() {",
  "  return 'Hello Lime Runtime';",
  "}",
  "",
  "export const runtimeVerified = true;",
  "",
].join("\n");

function printHelp() {
  console.log(`
Lime Code Runtime Fixture Smoke

用途:
  通过 localhost fixture 验证自然语言工具请求默认进入 current Agent Runtime，
  并完成 Read/Write/Bash、文件落盘、checkpoint/diff/evidence pack 的离线闭环。

用法:
  npm run smoke:code-runtime-fixture

选项:
  --output <path>       evidence JSON 输出路径，默认 .lime/qc/code-runtime-fixture-smoke.json
  --health-url <url>    DevBridge health 地址，默认 ${DEFAULT_HEALTH_URL}
  --invoke-url <url>    DevBridge invoke 地址，默认 ${DEFAULT_INVOKE_URL}
  --timeout-ms <ms>     总等待超时，默认 ${DEFAULT_TIMEOUT_MS}
  --interval-ms <ms>    轮询间隔，默认 ${DEFAULT_INTERVAL_MS}
  --allow-live-provider 保留统一 live gate 语义；本 smoke 默认且推荐使用 localhost fixture
  --no-write            只运行校验并打印摘要，不写 evidence JSON
  -h, --help            显示帮助
`);
}

function parseArgs(argv) {
  const options = {
    output: DEFAULT_OUTPUT,
    healthUrl: DEFAULT_HEALTH_URL,
    invokeUrl: DEFAULT_INVOKE_URL,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    intervalMs: DEFAULT_INTERVAL_MS,
    allowLiveProvider: liveProviderSmokeAllowed(),
    write: true,
    logPrefix: LOG_PREFIX,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--output" && argv[index + 1]) {
      options.output = path.resolve(rootDir, String(argv[index + 1]));
      index += 1;
      continue;
    }
    if (arg === "--health-url" && argv[index + 1]) {
      options.healthUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--invoke-url" && argv[index + 1]) {
      options.invokeUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && argv[index + 1]) {
      options.timeoutMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && argv[index + 1]) {
      options.intervalMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--no-write") {
      options.write = false;
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
  if (!options.healthUrl || !options.invokeUrl) {
    throw new Error("--health-url / --invoke-url 不能为空");
  }

  return options;
}

function writeEvidence(outputPath, evidence) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(evidence, null, 2)}\n`);
  return outputPath;
}

function writeEvidenceWithFallback(outputPath, evidence) {
  try {
    return writeEvidence(outputPath, evidence);
  } catch (error) {
    if (outputPath !== DEFAULT_OUTPUT) {
      throw error;
    }
    const fallbackPath = path.join(
      os.tmpdir(),
      "code-runtime-fixture-smoke.json",
    );
    const writtenPath = writeEvidence(fallbackPath, evidence);
    console.warn(
      `${LOG_PREFIX} default evidence write failed, fallback=${writtenPath}: ${error.message}`,
    );
    return writtenPath;
  }
}

let appServerRequestId = 1;

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

async function invokeAppServer(
  options,
  method,
  params = {},
  timeoutMs = options.timeoutMs,
) {
  const id = `code-runtime-fixture-${appServerRequestId++}`;
  const response = await invokeDevBridge(
    options,
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    {
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
    },
    timeoutMs,
  );
  const messages = decodeJsonRpcLines(response?.lines);
  const error = messages.find((message) => message?.id === id && message.error);
  if (error) {
    throw new Error(`${method} failed: ${JSON.stringify(error.error)}`);
  }
  const result = messages.find(
    (message) => message?.id === id && "result" in message,
  );
  if (!result) {
    throw new Error(`${method} did not return a JSON-RPC response`);
  }
  return {
    result: result.result,
    response: result,
    notifications: messages.filter(
      (message) => message?.method && !("id" in message),
    ),
    messages,
  };
}

function resolveWorkspaceRelativePath(relativePath) {
  return relativePath.split("/").join(path.sep);
}

function prepareFixtureFile(workspaceRoot) {
  const targetPath = path.join(
    workspaceRoot,
    resolveWorkspaceRelativePath(FIXTURE_RELATIVE_PATH),
  );
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.writeFileSync(targetPath, INITIAL_CONTENT, "utf8");
  return targetPath;
}

function buildCodeRuntimeMetadata() {
  return {
    artifactKind: "code_file",
    artifactTitle: "Code Runtime Fixture Greeting",
    artifactStatus: "ready",
    artifactVersionNo: 2,
    artifactVersionId: "code-runtime-fixture:greeting:v2",
    artifactVersions: [
      {
        id: "code-runtime-fixture:greeting:v1",
        versionNo: 1,
      },
      {
        id: "code-runtime-fixture:greeting:v2",
        versionNo: 2,
      },
    ],
    artifactVersionDiff: {
      summary: "Update greeting() to return Hello Lime Runtime.",
      changedFiles: [FIXTURE_RELATIVE_PATH],
      hunks: [
        {
          path: FIXTURE_RELATIVE_PATH,
          before: "Hello from initial fixture",
          after: "Hello Lime Runtime",
        },
      ],
    },
    previewText: "export function greeting() returns Hello Lime Runtime.",
  };
}

function buildFixtureResponses() {
  const nodeCommand = [
    "const fs = require('node:fs');",
    `const text = fs.readFileSync('${FIXTURE_RELATIVE_PATH}', 'utf8');`,
    "if (!text.includes('Hello Lime Runtime')) {",
    "  throw new Error('updated greeting was not written');",
    "}",
    "console.log('CODE_RUNTIME_TEST_OK');",
  ].join(" ");

  return [
    {
      type: "tool_call",
      id: "call-code-runtime-read",
      name: "Read",
      arguments: {
        path: FIXTURE_RELATIVE_PATH,
      },
    },
    {
      type: "tool_call",
      id: "call-code-runtime-write",
      name: "Write",
      arguments: {
        path: FIXTURE_RELATIVE_PATH,
        content: UPDATED_CONTENT,
        metadata: buildCodeRuntimeMetadata(),
      },
    },
    {
      type: "tool_call",
      id: "call-code-runtime-bash",
      name: "Bash",
      arguments: {
        command: `node -e "${nodeCommand}"`,
        timeout: 30,
      },
    },
    {
      type: "text",
      content: "CODE_RUNTIME_DONE",
    },
  ];
}

async function resolveWorkspaceRoot(options, workspace, workspaceId) {
  const directRoot = workspaceRootFromDefaultProject(workspace);
  if (directRoot) {
    return directRoot;
  }

  const ensured = await invokeDevBridge(options, "workspace_ensure_ready", {
    id: workspaceId,
  });
  const ensuredRoot = workspaceRootFromDefaultProject(ensured);
  assertSmoke(ensuredRoot, "默认 workspace 缺少 rootPath");
  return ensuredRoot;
}

function requestMessagesText(body) {
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  return messages
    .map((message) => {
      if (typeof message?.content === "string") {
        return message.content;
      }
      return JSON.stringify(message?.content || "");
    })
    .join("\n");
}

function requestBodyText(fixtureRequests) {
  return fixtureRequests
    .map((request) => JSON.stringify(request?.body || {}))
    .join("\n");
}

function valueContains(value, needle) {
  return JSON.stringify(value || {}).includes(needle);
}

function checkpointIdFromList(checkpointList) {
  const checkpoints = Array.isArray(checkpointList?.checkpoints)
    ? checkpointList.checkpoints
    : [];
  const match =
    checkpoints.find(
      (checkpoint) =>
        checkpoint?.path === FIXTURE_RELATIVE_PATH &&
        (checkpoint?.version_id || checkpoint?.versionId || checkpoint?.kind),
    ) ||
    checkpoints.find(
      (checkpoint) => checkpoint?.path === FIXTURE_RELATIVE_PATH,
    ) ||
    checkpoints[0];
  return String(match?.checkpoint_id || match?.checkpointId || "").trim();
}

function checkpointDiffContainsExpectedChange(checkpointDiff) {
  const diff =
    checkpointDiff?.diff ||
    checkpointDiff?.artifactVersionDiff ||
    checkpointDiff?.checkpoint?.metadata?.artifactVersionDiff;
  const text = JSON.stringify(diff || {});
  return (
    text.includes(FIXTURE_RELATIVE_PATH) &&
    text.includes("Hello from initial fixture") &&
    text.includes("Hello Lime Runtime")
  );
}

function summarizeCheckpointList(checkpointList) {
  const checkpoints = Array.isArray(checkpointList?.checkpoints)
    ? checkpointList.checkpoints
    : [];
  return {
    checkpointCount:
      checkpointList?.checkpoint_count ?? checkpointList?.checkpointCount ?? 0,
    paths: checkpoints.map((checkpoint) => checkpoint?.path).filter(Boolean),
  };
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function normalizeAppServerTurn(turn) {
  if (!turn || typeof turn !== "object") {
    return null;
  }
  return {
    ...turn,
    id: turn.id || turn.turnId || turn.turn_id || "",
    status: turn.status || "",
  };
}

function appServerSessionDetailFromRead(readResult) {
  const detail = asRecord(readResult?.detail) || {};
  const session = asRecord(readResult?.session) || {};
  const businessObjectRef = asRecord(session.businessObjectRef) || {};
  const metadata = asRecord(businessObjectRef.metadata) || {};
  const turns = Array.isArray(detail.turns)
    ? detail.turns.map(normalizeAppServerTurn).filter(Boolean)
    : Array.isArray(readResult?.turns)
      ? readResult.turns.map(normalizeAppServerTurn).filter(Boolean)
      : [];
  return {
    ...detail,
    id: detail.id || detail.session_id || detail.sessionId || session.sessionId,
    session_id:
      detail.session_id ||
      detail.sessionId ||
      session.sessionId ||
      session.session_id ||
      "",
    thread_id: detail.thread_id || detail.threadId || session.threadId || "",
    workspace_id:
      detail.workspace_id || detail.workspaceId || session.workspaceId || "",
    execution_strategy:
      detail.execution_strategy ||
      detail.executionStrategy ||
      metadata.executionStrategy ||
      null,
    turns,
  };
}

function appServerThreadReadFromSessionRead(readResult) {
  const detail = appServerSessionDetailFromRead(readResult);
  const detailThreadRead =
    asRecord(detail.thread_read) || asRecord(detail.threadRead);
  if (detailThreadRead) {
    return detailThreadRead;
  }

  const turns = Array.isArray(detail.turns) ? detail.turns : [];
  const activeTurn =
    turns.find((turn) =>
      [
        "accepted",
        "queued",
        "running",
        "waitingAction",
        "waiting_action",
      ].includes(String(turn?.status || "")),
    ) || null;
  const latestTurn = turns.at(-1) || null;
  const latestTurnStatus = latestTurn?.status || null;
  return {
    source: APP_SERVER_METHOD_THREAD_READ,
    session_id: detail.session_id || "",
    active_turn_id: activeTurn?.id || activeTurn?.turnId || null,
    status: activeTurn ? "running" : "idle",
    turns,
    queued_turns: Array.isArray(detail.queued_turns)
      ? detail.queued_turns
      : Array.isArray(detail.queuedTurns)
        ? detail.queuedTurns
        : [],
    diagnostics: {
      latest_turn_status: latestTurnStatus,
    },
    runtime_summary: {
      latestTurnStatus,
    },
    model_routing: detail.execution_runtime?.routing_decision || null,
  };
}

function appServerEvidenceSummary(evidenceExport) {
  if (!evidenceExport) {
    return null;
  }
  const events = Array.isArray(evidenceExport.events)
    ? evidenceExport.events
    : [];
  const artifacts = Array.isArray(evidenceExport.artifacts)
    ? evidenceExport.artifacts
    : [];
  const turns = Array.isArray(evidenceExport.turns) ? evidenceExport.turns : [];
  return {
    sessionId: evidenceExport.session?.sessionId || null,
    exportedAt: evidenceExport.exportedAt || null,
    turnCount: turns.length,
    eventCount: events.length,
    artifactCount: artifacts.length,
    evidencePack: summarizeEvidencePack(evidenceExport.evidencePack),
  };
}

function buildRuntimeRequest({ fixture, turnId, workspaceId }) {
  return {
    providerConfig: fixture.provider.providerConfig,
    providerPreference: fixture.provider.providerPreference,
    modelPreference: fixture.provider.modelPreference,
    approvalPolicy: "never",
    sandboxPolicy: "danger-full-access",
    workspaceId,
    executionStrategy: "react",
    metadata: {
      harness: {
        access_mode: "full-access",
        skip_mcp_prewarm: true,
        code_runtime_fixture: {
          scenario_id: "natural-language-code-runtime-fixture",
          source: "smoke:code-runtime-fixture",
        },
      },
    },
  };
}

async function waitForRuntimeCompletion(
  options,
  sessionId,
  fixture,
  targetPath,
) {
  const startedAt = Date.now();
  let lastSnapshot = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    const sessionReadResponse = await invokeAppServer(
      options,
      APP_SERVER_METHOD_THREAD_READ,
      {
        sessionId,
        historyLimit: 50,
      },
      30_000,
    );
    const sessionRead = sessionReadResponse.result;
    const threadRead = appServerThreadReadFromSessionRead(sessionRead);
    const sessionDetail = appServerSessionDetailFromRead(sessionRead);
    const fileContent = fs.existsSync(targetPath)
      ? fs.readFileSync(targetPath, "utf8")
      : "";
    lastSnapshot = {
      threadRead: summarizeThreadRead(threadRead),
      session: {
        id: sessionDetail?.id || null,
        executionStrategy:
          sessionDetail?.execution_strategy ||
          sessionDetail?.executionStrategy ||
          null,
        itemCount: Array.isArray(sessionDetail?.items)
          ? sessionDetail.items.length
          : 0,
        turnCount: Array.isArray(sessionDetail?.turns)
          ? sessionDetail.turns.length
          : 0,
      },
      fixtureChatRequestCount: fixtureChatRequestCount(fixture.requests),
      fileUpdated: fileContent.includes("Hello Lime Runtime"),
    };

    if (
      lastSnapshot.fixtureChatRequestCount >= 4 &&
      lastSnapshot.fileUpdated &&
      threadSettled(threadRead)
    ) {
      return {
        threadRead,
        sessionDetail,
        sessionRead,
        snapshot: lastSnapshot,
      };
    }

    await sleep(options.intervalMs);
  }

  throw new Error(
    `${LOG_PREFIX} wait runtime completion timeout; last=${JSON.stringify(lastSnapshot)}`,
  );
}

async function runSmoke(options) {
  console.log(`${LOG_PREFIX} stage=health`);
  const health = await waitForHealth(options);

  console.log(`${LOG_PREFIX} stage=fixture-provider`);
  if (options.allowLiveProvider) {
    assertLiveProviderSmokeAllowed({
      allowed: options.allowLiveProvider,
      scriptName: "smoke:code-runtime-fixture",
    });
  }
  const fixture = await startOpenAiCompatibleFixtureServer({
    scriptedResponses: buildFixtureResponses(),
  });
  console.log(
    `${LOG_PREFIX} provider=localhost-fixture baseUrl=${fixture.baseUrl}`,
  );

  try {
    console.log(`${LOG_PREFIX} stage=workspace`);
    const workspace = await invokeDevBridge(
      options,
      "get_or_create_default_project",
      {},
      30_000,
    );
    const workspaceId = workspaceIdFromDefaultProject(workspace);
    assertSmoke(workspaceId, "默认 workspace 缺少 id");
    const workspaceRoot = await resolveWorkspaceRoot(
      options,
      workspace,
      workspaceId,
    );
    const targetPath = prepareFixtureFile(workspaceRoot);

    console.log(`${LOG_PREFIX} stage=session`);
    const sessionName = `Code runtime fixture ${new Date().toISOString()}`;
    const requestedSessionId = `code-runtime-fixture-${Date.now()}-${process.pid}`;
    const sessionResult = await invokeAppServer(
      options,
      APP_SERVER_METHOD_THREAD_START,
      {
        sessionId: requestedSessionId,
        appId: "desktop",
        workspaceId,
        businessObjectRef: {
          kind: "agent.session",
          id: `agent-session:${workspaceId}:${requestedSessionId}`,
          title: sessionName,
          metadata: {
            title: sessionName,
            executionStrategy: "react",
            runStartHooks: false,
            harness: {
              hiddenFromUserRecents: true,
              source: "smoke:code-runtime-fixture",
            },
          },
        },
      },
      30_000,
    );
    const sessionId = sessionResult.result?.session?.sessionId;
    assertSmoke(sessionId, "thread/start 未返回 sessionId");
    const threadId =
      sessionResult.result?.session?.threadId ??
      sessionResult.result?.thread?.id ??
      sessionId;

    await invokeAppServer(
      options,
      APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE,
      {
        threadId,
        modelProvider: fixture.provider.providerPreference,
        model: fixture.provider.modelPreference,
      },
      30_000,
    );

    console.log(`${LOG_PREFIX} stage=submit-turn session=${sessionId}`);
    const turnId = `code-runtime-fixture-${Date.now()}-${process.pid}`;
    const turnMessage =
      "请修复这个 TypeScript fixture，让 greeting() 返回 Hello Lime Runtime，然后运行一个最小校验命令。";
    const runtimeRequest = buildRuntimeRequest({
      fixture,
      turnId,
      workspaceId,
    });
    const turnResult = await invokeAppServer(
      options,
      APP_SERVER_METHOD_TURN_START,
      {
        sessionId,
        turnId,
        input: {
          text: turnMessage,
        },
        runtimeOptions: {
          stream: true,
          eventName: `code_runtime_fixture_${turnId}`,
          runtimeRequest,
        },
        queueIfBusy: false,
        skipPreSubmitResume: true,
      },
    );
    assertSmoke(
      turnResult.result?.turn?.turnId === turnId,
      "turn/start 未返回同一 turnId",
    );

    console.log(`${LOG_PREFIX} stage=wait-runtime`);
    const finalState = await waitForRuntimeCompletion(
      options,
      sessionId,
      fixture,
      targetPath,
    );

    console.log(`${LOG_PREFIX} stage=file-checkpoints`);
    const checkpointListResult = await invokeAppServer(
      options,
      APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST,
      {
        sessionId,
      },
    );
    const checkpointList = checkpointListResult.result;
    const checkpointId = checkpointIdFromList(checkpointList);
    assertSmoke(checkpointId, "未找到 runtime file checkpoint");
    const checkpointDiffResult = await invokeAppServer(
      options,
      APP_SERVER_METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF,
      {
        sessionId,
        checkpointId,
      },
    );
    const checkpointDiff = checkpointDiffResult.result;

    console.log(`${LOG_PREFIX} stage=export-evidence`);
    const evidenceExportResult = await invokeAppServer(
      options,
      APP_SERVER_METHOD_EVIDENCE_EXPORT,
      {
        sessionId,
        turnId,
        includeEvents: true,
        includeArtifacts: true,
        includeEvidencePack: true,
      },
      30_000,
    );
    const evidenceExport = evidenceExportResult.result;

    const finalContent = fs.readFileSync(targetPath, "utf8");
    const fixtureBodyText = requestBodyText(fixture.requests);
    const firstRequestText = requestMessagesText(fixture.requests[0]?.body);
    const detailText = JSON.stringify(finalState.sessionDetail || {});
    const threadReadText = JSON.stringify(finalState.threadRead || {});
    const sessionReadText = JSON.stringify(finalState.sessionRead || {});
    const evidenceExportText = JSON.stringify(evidenceExport || {});
    const checkpointDiffText = JSON.stringify(checkpointDiff || {});
    const assertions = {
      appServerJsonRpcSubmitTurn:
        turnResult.result?.turn?.sessionId === sessionId &&
        turnResult.result?.turn?.turnId === turnId,
      appServerSessionReadObserved:
        finalState.sessionRead?.session?.sessionId === sessionId,
      appServerEvidenceExportObserved:
        evidenceExport?.session?.sessionId === sessionId,
      fixtureProviderUsed: fixtureChatRequestCount(fixture.requests) >= 4,
      naturalLanguageWithoutAtCode:
        firstRequestText.includes("Hello Lime Runtime") &&
        !firstRequestText.includes("@代码") &&
        !firstRequestText.includes("@code"),
      noHarnessCodeCommand: !fixtureBodyText.includes("code_command"),
      sessionDefaultedToReact:
        finalState.sessionDetail?.execution_strategy === "react" ||
        finalState.sessionDetail?.executionStrategy === "react",
      appServerRuntimeObserved:
        detailText.includes('"execution_strategy":"react"') ||
        detailText.includes('"executionStrategy":"react"') ||
        threadReadText.includes('"execution_strategy":"react"') ||
        threadReadText.includes('"executionStrategy":"react"'),
      readToolObserved:
        valueContains(finalState.sessionDetail, "Read") ||
        sessionReadText.includes("Read") ||
        evidenceExportText.includes("Read"),
      writeToolObserved:
        valueContains(finalState.sessionDetail, "Write") ||
        sessionReadText.includes("Write") ||
        evidenceExportText.includes("Write"),
      bashToolObserved:
        valueContains(finalState.sessionDetail, "Bash") ||
        sessionReadText.includes("Bash") ||
        evidenceExportText.includes("Bash"),
      workspaceFileUpdated: finalContent.includes("Hello Lime Runtime"),
      checkpointCreated:
        Number(
          checkpointList?.checkpoint_count ??
            checkpointList?.checkpointCount ??
            0,
        ) > 0,
      checkpointDiffAvailable:
        checkpointDiffContainsExpectedChange(checkpointDiff),
      evidenceExported: Boolean(evidenceExport),
      evidenceMentionsCodeFile: evidenceExportText.includes(
        FIXTURE_RELATIVE_PATH,
      ),
    };

    for (const [key, passed] of Object.entries(assertions)) {
      assertSmoke(passed, `断言失败: ${key}`);
    }

    const evidence = {
      schemaVersion: "v1",
      scenarioId: "natural-language-code-runtime-fixture",
      status: "pass",
      generatedAt: new Date().toISOString(),
      command: "smoke:code-runtime-fixture",
      coverage: {
        usesAppServerJsonRpcSubmitTurn: true,
        usesAppServerSessionRead: true,
        usesAppServerEvidenceExport: true,
        usesAppServerFileCheckpointCurrent: true,
        usesDefaultReactExecutionStrategy: true,
        usesLocalhostFixtureProvider: true,
        avoidsAtCodeCommandRoute: true,
        verifiesReadWriteBashTools: true,
        verifiesWorkspaceFileMutation: true,
        verifiesCheckpointDiff: true,
        verifiesEvidenceExport: true,
      },
      devBridge: {
        healthStatus: health?.status || null,
      },
      workspace: {
        id: workspaceId,
        root: workspaceRoot,
        fixturePath: FIXTURE_RELATIVE_PATH,
      },
      provider: {
        providerPreference: fixture.provider.providerPreference,
        providerName: fixture.provider.providerName,
        modelPreference: fixture.provider.modelPreference,
        source: fixture.provider.source,
      },
      runtime: {
        sessionId,
        turnId,
        finalSnapshot: finalState.snapshot,
        checkpointList: summarizeCheckpointList(checkpointList),
        checkpointId,
        checkpointDiff,
      },
      evidenceExport: appServerEvidenceSummary(evidenceExport),
      assertions,
    };

    if (options.write) {
      const writtenPath = writeEvidenceWithFallback(options.output, evidence);
      console.log(`${LOG_PREFIX} evidence=${writtenPath}`);
    }

    console.log(`${LOG_PREFIX} pass session=${sessionId}`);
    return evidence;
  } finally {
    await fixture.close();
  }
}

runSmoke(parseArgs(process.argv.slice(2))).catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
