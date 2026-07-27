#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";
import {
  REQUIRED_LIVE_WEB_TOOL_NAMES,
  liveWebToolEvidenceFromSession,
  liveWebToolStreamEvidenceFromEvents,
} from "./lib/claw-chat-live-web-tool-evidence.mjs";

const DEFAULTS = {
  appUrl: "http://127.0.0.1:1420/",
  healthUrl: "http://127.0.0.1:3030/health",
  invokeUrl: "http://127.0.0.1:3030/invoke",
  timeoutMs: 180_000,
  intervalMs: 1_000,
  providerPreference:
    process.env.LIME_AGENT_QC_PROVIDER ||
    process.env.LIME_E2E_PROVIDER ||
    process.env.LIME_DEFAULT_PROVIDER ||
    "",
  modelPreference:
    process.env.LIME_AGENT_QC_MODEL ||
    process.env.LIME_E2E_MODEL ||
    process.env.LIME_DEFAULT_MODEL ||
    "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "claw-chat-ready-streaming",
  ),
  prefix: "claw-chat-ready-streaming",
};

const POST_HEALTH_SETTLE_MS = 1_500;
const ONBOARDING_VERSION = "1.1.0";
const LONG_TURN_LINE_COUNT = 80;
const LONG_PROMPT = [
  `E2E 中断测试：请输出 ${LONG_TURN_LINE_COUNT} 行。`,
  "每一行都必须使用格式：中断测试第 N 行。",
  "从 1 开始递增，不要合并行，不要提前总结。",
  "如果收到停止请求应立即停止，不要补完剩余行。",
].join("\n");
const RECOVERY_EXPECTED_TEXT = "复原完成";
const RECOVERY_PROMPT = `停止后恢复测试：这是一个新的独立回合，请忽略上一条输出 ${LONG_TURN_LINE_COUNT} 行的要求。只输出“复原完成”这四个字，不要输出行号、解释或其他内容。`;
const LIVE_WEB_TOOL_PROMPT = [
  "@搜索 关键词:联网工具验证 今天 AI 行业公开新闻 站点:全网 时间:今天 深度:深度 重点:WebSearch 后选择一个公开来源 URL 并用 WebFetch 打开 输出:两句话回答来源标题、URL 和从页面确认到的一点事实。",
  "如果 WebSearch 或 WebFetch 不可用，请明确说明不可用；不要用训练记忆替代工具调用。",
].join("\n");
const MODEL_AVAILABILITY_PROMPT = "请只回复 QC_OK。";
const MAX_MODEL_AVAILABILITY_CANDIDATES = 12;
const FAST_RESPONSE_MODE_STORAGE_KEY = "lime:agent-fast-response-mode";
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
const APP_SERVER_METHOD_THREAD_READ = "thread/read";
const APP_SERVER_METHOD_TURN_START = "turn/start";
const APP_SERVER_METHOD_TURN_INTERRUPT = "turn/interrupt";
const APP_SERVER_METHOD_AGENT_SESSION_EVENT = "agentSession/event";
const APP_SERVER_METHOD_WORKSPACE_DEFAULT_ENSURE = "workspace/default/ensure";
const APP_SERVER_METHOD_MODEL_PROVIDER_LIST = "modelProvider/list";
const APP_SERVER_METHOD_MODEL_PROVIDER_READ = "modelProvider/read";
const APP_SERVER_METHOD_MODEL_PROVIDER_TEST_CHAT = "modelProvider/testChat";
const APP_SERVER_METHOD_MODEL_PROVIDER_UI_STATE_READ =
  "modelProviderUiState/read";
const APP_SERVER_METHOD_SKILL_MANAGEMENT_LIST = "skillManagement/list";

let appServerSmokeRequestId = 1;

function printHelp() {
  console.log(`
Lime Claw Chat Ready Streaming Smoke

用途:
  通过真实 Claw 聊天页面验证 workspace ready、流式增量、中断与恢复下一轮，
  并输出 GUI / runtime / console / network 四类证据。

用法:
  npm run smoke:claw-chat-ready-streaming
  npm run smoke:claw-chat-ready-streaming -- --provider-preference deepseek --model-preference deepseek-v4-flash

选项:
  --app-url <url>               前端地址，默认 http://127.0.0.1:1420/
  --health-url <url>            DevBridge 健康检查地址，默认 http://127.0.0.1:3030/health
  --invoke-url <url>            DevBridge invoke 地址，默认 http://127.0.0.1:3030/invoke
  --timeout-ms <ms>             总超时，默认 180000
  --interval-ms <ms>            轮询间隔，默认 1000
  --provider-preference <id>    可选，显式指定 provider
  --model-preference <model>    可选，显式指定 model
  --evidence-dir <path>         证据目录，默认 .lime/qc/gui-evidence/claw-chat-ready-streaming
  --prefix <name>               证据文件前缀，默认 claw-chat-ready-streaming
  -h, --help                    显示帮助
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--app-url" && argv[index + 1]) {
      options.appUrl = String(argv[index + 1]).trim();
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
    if (arg === "--provider-preference" && argv[index + 1]) {
      options.providerPreference = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--model-preference" && argv[index + 1]) {
      options.modelPreference = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--evidence-dir" && argv[index + 1]) {
      options.evidenceDir = path.resolve(String(argv[index + 1]).trim());
      index += 1;
      continue;
    }
    if (arg === "--prefix" && argv[index + 1]) {
      options.prefix = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!options.appUrl) {
    throw new Error("--app-url 不能为空");
  }
  if (!options.evidenceDir) {
    throw new Error("--evidence-dir 不能为空");
  }
  if (!options.prefix) {
    throw new Error("--prefix 不能为空");
  }

  return options;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function logStage(label) {
  console.log(`[smoke:claw-chat-ready-streaming] stage=${label}`);
}

function readConsoleText(consoleMessages) {
  return consoleMessages.map((item) => String(item?.text || "")).join("\n");
}

async function waitForHealth(options) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const response = await fetch(options.healthUrl, { method: "GET" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      if (payload?.transport !== "electron-host") {
        throw new Error(
          `DevBridge transport must be electron-host, got ${String(
            payload?.transport || "unknown",
          )}`,
        );
      }
      console.log(
        `[smoke:claw-chat-ready-streaming] DevBridge 已就绪 (${Date.now() - startedAt}ms)${
          payload?.status ? ` status=${payload.status}` : ""
        }`,
      );
      return payload;
    } catch (error) {
      lastError = error;
      await sleep(options.intervalMs);
    }
  }

  const detail =
    lastError instanceof Error
      ? lastError.message
      : String(lastError || "unknown error");
  throw new Error(
    `[smoke:claw-chat-ready-streaming] DevBridge 未就绪，请先启动 npm run electron:dev。最后错误: ${detail}`,
  );
}

async function invoke(options, cmd, args, timeoutMs = options.timeoutMs) {
  const response = await fetch(options.invokeUrl, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({ cmd, args }),
    signal: AbortSignal.timeout(timeoutMs),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}; ${text}`);
  }

  const payload = text ? JSON.parse(text) : null;
  if (payload?.error) {
    throw new Error(String(payload.error));
  }

  return payload?.result;
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

function attachAppServerRequestMessages(entry) {
  if (entry?.cmd !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
    return entry;
  }
  entry.appServer = {
    ...(entry.appServer || {}),
    requestMessages: decodeJsonRpcLines(entry.args?.request?.lines),
  };
  return entry;
}

function attachAppServerResponsePayload(entry, payload) {
  if (
    entry?.cmd !== APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
    entry?.cmd !== APP_SERVER_DRAIN_EVENTS_COMMAND
  ) {
    return entry;
  }
  if (entry.cmd === APP_SERVER_DRAIN_EVENTS_COMMAND) {
    entry.appServer = {
      ...(entry.appServer || {}),
      drainMessages: decodeJsonRpcLines(payload?.result?.lines),
    };
    return entry;
  }
  entry.appServer = {
    ...(entry.appServer || {}),
    responseMessages: decodeJsonRpcLines(payload?.result?.lines),
  };
  return entry;
}

function appServerMethodRecords(invokes, method, options = {}) {
  const direction = options.direction || "any";
  const records = [];
  for (const [invokeIndex, entry] of invokes.entries()) {
    if (entry?.cmd !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
      continue;
    }
    const groups = [
      ["request", entry.appServer?.requestMessages || []],
      ["response", entry.appServer?.responseMessages || []],
    ];
    for (const [messageDirection, messages] of groups) {
      if (direction !== "any" && direction !== messageDirection) {
        continue;
      }
      for (const [messageIndex, message] of messages.entries()) {
        if (message?.method !== method) {
          continue;
        }
        records.push({
          entry,
          invokeIndex,
          messageIndex,
          direction: messageDirection,
          message,
          params: message.params || {},
        });
      }
    }
  }
  return records;
}

function findAppServerMethodRecord(invokes, method, predicate, options = {}) {
  return (
    appServerMethodRecords(invokes, method, options).find((record) =>
      predicate ? predicate(record) : true,
    ) || null
  );
}

function appServerParamSessionId(params) {
  return String(params?.sessionId || params?.session_id || "");
}

function appServerParamTurnId(params) {
  return String(params?.turnId || params?.turn_id || "");
}

function appServerTurnInputText(params) {
  return String(params?.input?.text || params?.message || "");
}

function appServerRuntimeOptions(params) {
  return params?.runtimeOptions || {};
}

function runtimeRequestFromAppServerParams(params) {
  return appServerRuntimeOptions(params).runtimeRequest || {};
}

function runtimeRequestProviderModel(runtimeRequest) {
  const request = runtimeRequest || {};
  const modelPolicy = request?.metadata?.harness?.model_request_policy || {};
  return {
    provider:
      request.providerPreference ||
      modelPolicy.provider_id ||
      modelPolicy.providerId ||
      null,
    model:
      request.modelPreference ||
      modelPolicy.model_id ||
      modelPolicy.modelId ||
      null,
    source:
      request.providerPreference || request.modelPreference
        ? "runtime-request"
        : modelPolicy.provider_id || modelPolicy.model_id
          ? "model-request-policy"
          : null,
  };
}

function appServerTurnEvidenceFromRecord(record) {
  const params = record?.params || {};
  return {
    method: record?.message?.method || "",
    sessionId: appServerParamSessionId(params),
    turnId: appServerParamTurnId(params),
    eventName: appServerRuntimeOptions(params).eventName || null,
    providerPreference:
      runtimeRequestFromAppServerParams(params).providerPreference || null,
    modelPreference:
      runtimeRequestFromAppServerParams(params).modelPreference || null,
  };
}

function appServerMethodSeen(invokes, method, options = {}) {
  return appServerMethodRecords(invokes, method, options).length > 0;
}

function appServerDrainEventRecords(invokes) {
  const records = [];
  for (const [invokeIndex, entry] of invokes.entries()) {
    if (entry?.cmd !== APP_SERVER_DRAIN_EVENTS_COMMAND) {
      continue;
    }
    for (const [messageIndex, message] of (
      entry.appServer?.drainMessages || []
    ).entries()) {
      if (message?.method !== APP_SERVER_METHOD_AGENT_SESSION_EVENT) {
        continue;
      }
      records.push({
        entry,
        invokeIndex,
        messageIndex,
        direction: "drain",
        message,
        params: message.params || {},
      });
    }
  }
  return records;
}

function appServerEventRecords(invokes) {
  return [
    ...appServerMethodRecords(invokes, APP_SERVER_METHOD_AGENT_SESSION_EVENT, {
      direction: "response",
    }),
    ...appServerDrainEventRecords(invokes),
  ];
}

function eventRecordMatchesTurn(record, { sessionId, turnId }) {
  const params = record?.params || {};
  const eventSessionId = appServerEventSessionId(params);
  const eventTurnId = appServerEventTurnId(params);
  return (
    (!sessionId || !eventSessionId || eventSessionId === sessionId) &&
    (!turnId || !eventTurnId || eventTurnId === turnId)
  );
}

function eventRecordStrictlyMatchesTurn(record, { sessionId, turnId }) {
  const params = record?.params || {};
  const eventSessionId = appServerEventSessionId(params);
  const eventTurnId = appServerEventTurnId(params);
  return (
    (!sessionId || eventSessionId === sessionId) &&
    (!turnId || eventTurnId === turnId)
  );
}

function consoleStreamEventEvidence(consoleMessages, { sessionId, turnId }) {
  const markers = [
    "AgentStream.firstEvent",
    "AgentStream.firstRuntimeStatus",
    "AgentStream.runtimeKeepalive",
    "AgentStream.firstTextDelta",
  ];
  const lineIndex = consoleMessages.findIndex((item) => {
    const text = String(item?.text || "");
    return (
      markers.some((marker) => text.includes(marker)) &&
      (!sessionId || text.includes(sessionId)) &&
      (!turnId || text.includes(turnId) || !text.includes("turnId"))
    );
  });
  if (lineIndex < 0) {
    return null;
  }
  const line = consoleMessages[lineIndex];
  return {
    source: "renderer-agent-stream-console",
    lineIndex,
    text: String(line?.text || ""),
  };
}

function appServerCurrentEventEvidence(invokes, consoleMessages, turnRef) {
  const networkRecord =
    appServerEventRecords(invokes).find((record) =>
      eventRecordMatchesTurn(record, turnRef),
    ) || null;
  if (networkRecord) {
    return {
      source:
        networkRecord.direction === "drain"
          ? "app-server-drain-events"
          : "app-server-json-lines-response",
      sessionId: turnRef.sessionId,
      turnId: turnRef.turnId,
      invokeIndex: networkRecord.invokeIndex,
      messageIndex: networkRecord.messageIndex,
      eventType:
        networkRecord.params?.event?.type ||
        networkRecord.params?.payload?.type ||
        null,
    };
  }
  return consoleStreamEventEvidence(consoleMessages, turnRef);
}

function isSmokeDiagnosticJsonRpcId(id) {
  return String(id || "").startsWith("smoke-");
}

function appServerResponseForRequestRecord(record) {
  const requestId = record?.message?.id;
  if (requestId === undefined || requestId === null) {
    return null;
  }

  return (
    (record.entry?.appServer?.responseMessages || []).find(
      (message) => message?.id === requestId && "result" in message,
    ) || null
  );
}

function appServerMethodSucceeded(invokes, method) {
  return appServerMethodRecords(invokes, method, {
    direction: "request",
  }).some((record) => Boolean(appServerResponseForRequestRecord(record)));
}

function appServerEventSessionId(params) {
  const event = params?.event || params?.payload || {};
  return String(
    event?.sessionId ||
      event?.session_id ||
      params?.sessionId ||
      params?.session_id ||
      "",
  );
}

function appServerEventTurnId(params) {
  const event = params?.event || params?.payload || {};
  return String(
    event?.turnId || event?.turn_id || params?.turnId || params?.turn_id || "",
  );
}

function appServerReadResponseHasSessionTurn(response, sessionId, turnId) {
  if (!response?.result) {
    return false;
  }

  const detail = appServerSessionDetailFromRead(response.result);
  if (sessionId && detail.session_id && detail.session_id !== sessionId) {
    return false;
  }
  if (!turnId) {
    return true;
  }

  return (detail.turns || []).some((turn) => turn?.id === turnId);
}

function appServerSessionReadAfterEventEvidence(
  invokes,
  { sessionId, turnId },
  options = {},
) {
  const eventMatcher = options.strictEventScope
    ? eventRecordStrictlyMatchesTurn
    : eventRecordMatchesTurn;
  const eventRecords = appServerEventRecords(invokes).filter((record) =>
    eventMatcher(record, { sessionId, turnId }),
  );
  const readRecords = appServerMethodRecords(
    invokes,
    APP_SERVER_METHOD_THREAD_READ,
    { direction: "request" },
  ).filter((record) => {
    const params = record.params || {};
    return (
      !isSmokeDiagnosticJsonRpcId(record.message?.id) &&
      (!sessionId || appServerParamSessionId(params) === sessionId)
    );
  });

  for (const eventRecord of eventRecords) {
    for (const readRecord of readRecords) {
      if (readRecord.invokeIndex <= eventRecord.invokeIndex) {
        continue;
      }
      const response = appServerResponseForRequestRecord(readRecord);
      if (!appServerReadResponseHasSessionTurn(response, sessionId, turnId)) {
        continue;
      }
      return {
        sessionId,
        turnId,
        eventInvokeIndex: eventRecord.invokeIndex,
        eventMessageIndex: eventRecord.messageIndex,
        readInvokeIndex: readRecord.invokeIndex,
        readMessageIndex: readRecord.messageIndex,
        readRequestId: String(readRecord.message?.id || ""),
        source: "gui-network-after-event",
      };
    }
  }

  return null;
}

function appServerSessionReadTurnEvidence(invokes, { sessionId, turnId }) {
  const readRecords = appServerMethodRecords(
    invokes,
    APP_SERVER_METHOD_THREAD_READ,
    { direction: "request" },
  ).filter((record) => {
    const params = record.params || {};
    return (
      !isSmokeDiagnosticJsonRpcId(record.message?.id) &&
      (!sessionId || appServerParamSessionId(params) === sessionId)
    );
  });

  for (const readRecord of readRecords) {
    const response = appServerResponseForRequestRecord(readRecord);
    if (!appServerReadResponseHasSessionTurn(response, sessionId, turnId)) {
      continue;
    }
    return {
      sessionId,
      turnId,
      readInvokeIndex: readRecord.invokeIndex,
      readMessageIndex: readRecord.messageIndex,
      readRequestId: String(readRecord.message?.id || ""),
      source: "gui-network-session-read-turn",
    };
  }

  return null;
}

async function appServerRpc(
  options,
  method,
  params,
  timeoutMs = options.timeoutMs,
) {
  const id = `smoke-${appServerSmokeRequestId++}`;
  const request =
    params === undefined ? { id, method } : { id, method, params };
  const result = await invoke(
    options,
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    {
      request: {
        lines: [`${JSON.stringify(request)}\n`],
      },
    },
    timeoutMs,
  );
  const messages = decodeJsonRpcLines(result?.lines);
  const error = messages.find((message) => message?.id === id && message.error);
  if (error) {
    throw new Error(String(error.error?.message || "App Server RPC error"));
  }
  const response = messages.find(
    (message) => message?.id === id && "result" in message,
  );
  if (!response) {
    throw new Error(`App Server RPC ${method} 缺少响应`);
  }
  return {
    result: response.result,
    response,
    notifications: messages.filter(
      (message) => message?.method && !("id" in message),
    ),
    messages,
  };
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
  const detail =
    readResult?.detail && typeof readResult.detail === "object"
      ? readResult.detail
      : {};
  const turns = Array.isArray(detail.turns)
    ? detail.turns.map(normalizeAppServerTurn).filter(Boolean)
    : Array.isArray(readResult?.turns)
      ? readResult.turns.map(normalizeAppServerTurn).filter(Boolean)
      : [];
  return {
    ...detail,
    session_id:
      detail.session_id ||
      detail.sessionId ||
      readResult?.session?.sessionId ||
      readResult?.session?.session_id ||
      "",
    turns,
  };
}

function appServerThreadReadFromSessionRead(readResult) {
  const detail = appServerSessionDetailFromRead(readResult);
  const currentThreadRead =
    detail.thread_read && typeof detail.thread_read === "object"
      ? detail.thread_read
      : detail.threadRead && typeof detail.threadRead === "object"
        ? detail.threadRead
        : {};
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
    ...currentThreadRead,
    source: APP_SERVER_METHOD_THREAD_READ,
    session_id: detail.session_id || "",
    active_turn_id: activeTurn?.id || null,
    status: currentThreadRead.status || (activeTurn ? "running" : "idle"),
    queued_turns: Array.isArray(detail.queued_turns)
      ? detail.queued_turns
      : Array.isArray(detail.queuedTurns)
        ? detail.queuedTurns
        : [],
    diagnostics: {
      ...(currentThreadRead.diagnostics || {}),
      latest_turn_status: latestTurnStatus,
    },
    runtime_summary: {
      ...(currentThreadRead.runtime_summary ||
        currentThreadRead.runtimeSummary ||
        {}),
      latestTurnStatus,
    },
    model_routing:
      currentThreadRead.model_routing ||
      currentThreadRead.modelRouting ||
      detail.execution_runtime?.routing_decision ||
      null,
  };
}

async function readAppServerSession(options, sessionId, timeoutMs = 20_000) {
  const response = await appServerRpc(
    options,
    APP_SERVER_METHOD_THREAD_READ,
    { sessionId },
    timeoutMs,
  );
  return appServerSessionDetailFromRead(response.result);
}

async function readAppServerThreadRead(options, sessionId, timeoutMs = 20_000) {
  const response = await appServerRpc(
    options,
    APP_SERVER_METHOD_THREAD_READ,
    { sessionId },
    timeoutMs,
  );
  return appServerThreadReadFromSessionRead(response.result);
}

async function cancelAppServerTurn(
  options,
  sessionId,
  turnId,
  timeoutMs = 20_000,
) {
  if (!sessionId || !turnId) {
    return false;
  }
  await appServerRpc(
    options,
    APP_SERVER_METHOD_TURN_INTERRUPT,
    { sessionId, turnId },
    timeoutMs,
  );
  return true;
}

async function waitForCondition(label, predicate, timeoutMs, intervalMs) {
  const startedAt = Date.now();
  let lastValue = null;

  while (Date.now() - startedAt < timeoutMs) {
    lastValue = await predicate();
    if (lastValue) {
      return lastValue;
    }
    await sleep(intervalMs);
  }

  throw new Error(
    `[smoke:claw-chat-ready-streaming] ${label} 超时，最后结果: ${JSON.stringify(lastValue)}`,
  );
}

function normalizeProviderId(provider) {
  return String(
    provider?.id || provider?.provider_id || provider?.providerId || "",
  ).trim();
}

function providerMatchesId(provider, providerId) {
  const normalizedProviderId = normalizeProviderId(provider);
  return (
    normalizedProviderId === providerId ||
    provider?.provider_name === providerId ||
    provider?.providerName === providerId ||
    provider?.name === providerId
  );
}

function providerEnabled(provider) {
  return provider?.enabled !== false;
}

function providerHasCredential(provider) {
  const count = Number(provider?.api_key_count ?? provider?.apiKeyCount);
  if (Number.isFinite(count)) {
    return count > 0;
  }
  if (typeof provider?.has_api_key === "boolean") {
    return provider.has_api_key;
  }
  if (typeof provider?.hasApiKey === "boolean") {
    return provider.hasApiKey;
  }

  return true;
}

function providerReadyForLiveRuntime(provider) {
  return providerEnabled(provider) && providerHasCredential(provider);
}

function modelLooksChatCapable(modelName) {
  const normalized = String(modelName || "")
    .trim()
    .toLowerCase();
  if (!normalized) {
    return false;
  }

  return !/(image|images|banana|embed|embedding|tts|transcrib|audio|video|bfl|flux|stable|kling|wan)/i.test(
    normalized,
  );
}

function modelLooksLightweight(modelName) {
  return /flash|mini|lite/i.test(String(modelName || ""));
}

function modelLooksExpensive(modelName) {
  return /sonnet|opus|pro|thinking|reasoning/i.test(String(modelName || ""));
}

function modelLooksToolReliable(modelName) {
  return (
    modelLooksChatCapable(modelName) &&
    !modelLooksLightweight(modelName) &&
    !modelLooksExpensive(modelName)
  );
}

function pickModelPreference(provider) {
  const candidates = [
    provider?.default_model,
    provider?.defaultModel,
    ...(Array.isArray(provider?.models) ? provider.models : []),
  ]
    .map((value) =>
      typeof value === "string"
        ? value
        : String(value?.name || value?.id || value?.model || "").trim(),
    )
    .filter(Boolean);

  return (
    candidates.find((value) => modelLooksToolReliable(value)) ||
    candidates.find(
      (value) => /deepseek/i.test(value) && modelLooksChatCapable(value),
    ) ||
    candidates.find((value) => modelLooksChatCapable(value)) ||
    ""
  );
}

function findProviderById(providers, providerId) {
  const normalizedProviderId = String(providerId || "").trim();
  if (!normalizedProviderId) {
    return null;
  }

  return (
    providers.find((provider) =>
      providerMatchesId(provider, normalizedProviderId),
    ) || null
  );
}

function pickProvider(providers, preferredProviderId) {
  const ready = providers.filter((provider) =>
    providerReadyForLiveRuntime(provider),
  );
  const enabled = providers.filter((provider) => providerEnabled(provider));
  if (preferredProviderId) {
    return (
      ready.find((provider) =>
        providerMatchesId(provider, preferredProviderId),
      ) ||
      enabled.find((provider) =>
        providerMatchesId(provider, preferredProviderId),
      ) ||
      providers.find((provider) =>
        providerMatchesId(provider, preferredProviderId),
      ) ||
      null
    );
  }

  for (const providerId of ["deepseek", "doubao", "lime-hub"]) {
    const match = ready.find(
      (provider) => normalizeProviderId(provider) === providerId,
    );
    if (match) {
      return match;
    }
  }

  return ready[0] || enabled[0] || null;
}

function candidateRank(candidate) {
  const providerId = candidate.providerPreference.toLowerCase();
  const model = candidate.modelPreference.toLowerCase();
  const toolReliable = modelLooksToolReliable(model);
  const deepseekModel = /deepseek/.test(model);
  const lightweight = modelLooksLightweight(model);
  const expensive = modelLooksExpensive(model);

  const sourcePenalty =
    candidate.source === "agent-status" && toolReliable
      ? 0
      : candidate.source === "agent-status"
        ? 8
        : 2;
  const providerPenalty = ["deepseek", "siliconflow-cn", "lime-hub"].includes(
    providerId,
  )
    ? 0
    : providerId === "doubao"
      ? 1
      : 3;
  const modelPenalty = toolReliable
    ? 0
    : deepseekModel
      ? 4
      : expensive
        ? 18
        : lightweight
          ? 12
          : 6;

  return sourcePenalty + providerPenalty + modelPenalty;
}

function uniqueProviderCandidates(candidates) {
  const seen = new Set();
  return candidates.filter((candidate) => {
    const key = `${candidate.providerPreference}::${candidate.modelPreference}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

async function buildProviderCandidate(
  options,
  provider,
  modelPreference,
  source,
) {
  const providerId =
    normalizeProviderId(provider) ||
    String(provider?.provider_name || "").trim();
  if (!providerId) {
    return null;
  }

  let providerDetail = provider;
  try {
    const providerRead = await appServerRpc(
      options,
      APP_SERVER_METHOD_MODEL_PROVIDER_READ,
      { providerId },
      30_000,
    );
    providerDetail = providerRead.result?.provider || provider;
  } catch (error) {
    console.warn(
      `[smoke:claw-chat-ready-streaming] 读取 provider 详情失败，使用列表摘要继续: ${error.message}`,
    );
  }

  const model = modelPreference || pickModelPreference(providerDetail);
  if (!model || !modelLooksChatCapable(model)) {
    return null;
  }

  return {
    providerPreference: providerId,
    modelPreference: model,
    source,
  };
}

async function verifyProviderCandidate(options, candidate) {
  const response = await appServerRpc(
    options,
    APP_SERVER_METHOD_MODEL_PROVIDER_TEST_CHAT,
    {
      providerId: candidate.providerPreference,
      modelName: candidate.modelPreference,
      prompt: MODEL_AVAILABILITY_PROMPT,
    },
    60_000,
  );
  return Boolean(response.result?.success);
}

function sanitizeProbeError(error) {
  const message =
    error instanceof Error ? error.message : String(error || "unknown");
  return message.replace(/\s+/g, " ").slice(0, 160);
}

async function resolveProviderPreference(options, agentStatus, providers) {
  const explicitProvider = String(options.providerPreference || "").trim();
  const explicitModel = String(options.modelPreference || "").trim();
  if (explicitProvider && explicitModel) {
    return {
      providerPreference: explicitProvider,
      modelPreference: explicitModel,
      source: "explicit",
    };
  }

  const statusProvider = String(
    agentStatus?.provider_selector || agentStatus?.provider_name || "",
  ).trim();
  const statusModel = String(agentStatus?.model_name || "").trim();
  const providerListAvailable =
    Array.isArray(providers) && providers.length > 0;
  const statusProviderRecord = providerListAvailable
    ? findProviderById(providers, statusProvider)
    : null;
  const statusProviderReady =
    statusProvider &&
    statusModel &&
    (!providerListAvailable ||
      (statusProviderRecord &&
        providerReadyForLiveRuntime(statusProviderRecord)));
  const providerList = Array.isArray(providers) ? providers : [];
  const selected = pickProvider(providerList, explicitProvider || "");

  if (explicitProvider || explicitModel) {
    if (!selected) {
      throw new Error(
        "[smoke:claw-chat-ready-streaming] 未找到可用 provider；请传 --provider-preference / --model-preference 或先配置本地 provider",
      );
    }
    const selectedMatchesStatusProvider =
      statusProvider && providerMatchesId(selected, statusProvider);
    const candidate = await buildProviderCandidate(
      options,
      selected,
      explicitModel || (selectedMatchesStatusProvider ? statusModel : ""),
      explicitProvider || explicitModel
        ? "partial-explicit"
        : "auto-enabled-provider",
    );
    if (!candidate) {
      const providerId =
        normalizeProviderId(selected) ||
        String(selected?.provider_name || "").trim();
      throw new Error(
        `[smoke:claw-chat-ready-streaming] provider ${providerId} 缺少可用聊天模型；请传 --model-preference`,
      );
    }
    return candidate;
  }

  const autoCandidates = [];
  if (!explicitProvider && !explicitModel && statusProviderReady) {
    const statusCandidate = await buildProviderCandidate(
      options,
      statusProviderRecord || {
        id: statusProvider,
        models: [{ id: statusModel }],
      },
      statusModel,
      "agent-status",
    );
    if (statusCandidate) {
      autoCandidates.push(statusCandidate);
    }
  }
  for (const provider of providerList.filter((item) =>
    providerReadyForLiveRuntime(item),
  )) {
    const candidate = await buildProviderCandidate(
      options,
      provider,
      "",
      "auto-enabled-provider",
    );
    if (candidate) {
      autoCandidates.push(candidate);
    }
  }

  const candidates = uniqueProviderCandidates(autoCandidates)
    .sort((left, right) => candidateRank(left) - candidateRank(right))
    .slice(0, MAX_MODEL_AVAILABILITY_CANDIDATES);
  if (candidates.length === 0) {
    const providerId =
      normalizeProviderId(selected) ||
      String(selected?.provider_name || statusProvider || "").trim();
    throw new Error(
      `[smoke:claw-chat-ready-streaming] provider ${providerId} 缺少可用聊天模型；请传 --model-preference`,
    );
  }

  const probeFailures = [];
  for (const candidate of candidates) {
    try {
      if (await verifyProviderCandidate(options, candidate)) {
        return {
          ...candidate,
          source: "auto-probed-provider",
        };
      }
      probeFailures.push(
        `${candidate.providerPreference}/${candidate.modelPreference}: failed`,
      );
    } catch (error) {
      probeFailures.push(
        `${candidate.providerPreference}/${candidate.modelPreference}: ${sanitizeProbeError(
          error,
        )}`,
      );
    }
  }

  throw new Error(
    `[smoke:claw-chat-ready-streaming] 未找到可通过短聊天探测的 provider/model；候选失败: ${probeFailures.join(
      " | ",
    )}`,
  );
}

function textOfMessage(message) {
  if (!message || !Array.isArray(message.content)) {
    return "";
  }
  return message.content
    .map((part) => {
      if (part && typeof part.text === "string") {
        return part.text;
      }
      if (part && typeof part.output === "string") {
        return part.output;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

function allAssistantText(session) {
  return (session?.messages || [])
    .filter((message) => message?.role === "assistant")
    .map(textOfMessage)
    .join("\n");
}

function allAgentItemText(session) {
  return (session?.items || [])
    .filter((item) => item?.type === "agent_message")
    .map((item) => {
      if (typeof item?.text === "string") {
        return item.text;
      }
      if (typeof item?.content === "string") {
        return item.content;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

function findTurn(session, turnId) {
  return (session?.turns || []).find((turn) => turn?.id === turnId) || null;
}

function queuedTurnCount(session) {
  return Array.isArray(session?.queued_turns) ? session.queued_turns.length : 0;
}

function activeTurnIdFromThreadRead(threadRead) {
  return threadRead?.active_turn_id || threadRead?.activeTurnId || null;
}

function latestTurnStatusFromThreadRead(threadRead) {
  return (
    threadRead?.diagnostics?.latest_turn_status ||
    threadRead?.runtime_summary?.latestTurnStatus ||
    threadRead?.status ||
    null
  );
}

function isTerminalTurnStatus(status) {
  return ["aborted", "canceled", "completed", "failed"].includes(
    String(status || ""),
  );
}

function isInterruptedTurnStatus(status) {
  return ["aborted", "canceled"].includes(String(status || ""));
}

function syntheticInterruptedTurn(turnId, threadRead) {
  return {
    id: turnId,
    status: "canceled",
    synthetic: true,
    source: "thread-read-idle-after-interrupt",
    threadStatus: latestTurnStatusFromThreadRead(threadRead),
  };
}

function interruptDrainedWithoutRecordedTurn({
  session,
  threadRead,
  snapshot,
  interruptScoped,
}) {
  const threadStatus = String(
    latestTurnStatusFromThreadRead(threadRead) || "",
  ).toLowerCase();
  return (
    interruptScoped &&
    queuedTurnCount(session) === 0 &&
    queuedTurnCount(threadRead) === 0 &&
    !activeTurnIdFromThreadRead(threadRead) &&
    (threadStatus === "idle" ||
      threadStatus === "aborted" ||
      threadStatus === "canceled") &&
    snapshot &&
    !snapshot.stopVisible &&
    !snapshot.finalLineSeen
  );
}

function normalizeConsoleLine(item) {
  const location =
    item.location && item.location.url
      ? ` @ ${item.location.url}:${item.location.lineNumber ?? ""}`
      : "";
  return `[${item.type}] ${item.text}${location}`;
}

function isBenignConsoleError(item) {
  void item;
  return false;
}

function composerDomHelperScript() {
  return `
    function isElementVisible(element) {
      if (!element || !(element instanceof HTMLElement)) {
        return false;
      }
      const style = window.getComputedStyle(element);
      if (
        style.visibility === "hidden" ||
        style.display === "none" ||
        Number(style.opacity || "1") === 0
      ) {
        return false;
      }
      const rect = element.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    }

    function isComposerWritable(element) {
      if (!element || !(element instanceof HTMLElement)) {
        return false;
      }
      if (element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement) {
        return !element.disabled && !element.readOnly;
      }
      return element.isContentEditable;
    }

    function readComposerValue(element) {
      if (!element) {
        return "";
      }
      if (element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement) {
        return element.value || "";
      }
      return element.textContent || "";
    }

    function scoreComposerCandidate(element) {
      let score = 0;
      if (element.matches('textarea[name="agent-chat-message"]')) score += 100;
      if (element.closest('[data-testid="inputbar-core-container"]')) score += 80;
      if (element.closest('[data-testid="inputbar-connected-composer"]')) score += 60;
      if (element.matches("textarea")) score += 40;
      if (element.matches("input")) score += 20;
      if (element.matches('[contenteditable="true"], [contenteditable="plaintext-only"], [role="textbox"]')) score += 10;
      if (isElementVisible(element)) score += 5;
      if (isComposerWritable(element)) score += 5;
      const rect = element.getBoundingClientRect();
      score += Math.max(0, Math.min(1000, rect.bottom)) / 10000;
      return score;
    }

    function findComposerCandidates() {
      const selectors = [
        '[data-testid="inputbar-core-container"] textarea',
        '[data-testid="inputbar-connected-composer"] textarea',
        'textarea[name="agent-chat-message"]',
        '[data-testid="inputbar-core-container"] [contenteditable="true"]',
        '[data-testid="inputbar-connected-composer"] [contenteditable="true"]',
        '[data-testid="inputbar-core-container"] [contenteditable="plaintext-only"]',
        '[data-testid="inputbar-connected-composer"] [contenteditable="plaintext-only"]',
        '[data-testid="inputbar-core-container"] [role="textbox"]',
        '[data-testid="inputbar-connected-composer"] [role="textbox"]',
        'textarea',
        'input[type="text"]',
        '[contenteditable="true"]',
        '[contenteditable="plaintext-only"]',
        '[role="textbox"]',
      ];
      return Array.from(document.querySelectorAll(selectors.join(",")))
        .filter((element) => element instanceof HTMLElement)
        .map((element) => ({
          element,
          visible: isElementVisible(element),
          writable: isComposerWritable(element),
          score: scoreComposerCandidate(element),
        }))
        .sort((left, right) => left.score - right.score);
    }

    function findComposerElement() {
      const candidates = findComposerCandidates().filter(
        (candidate) => candidate.visible && candidate.writable,
      );
      return candidates.at(-1)?.element || null;
    }

    function findComposerRoot(element) {
      if (!element || !(element instanceof HTMLElement)) {
        return null;
      }
      return (
        element.closest('[data-testid="inputbar-core-container"]') ||
        element.closest('[data-testid="inputbar-connected-composer"]') ||
        element.closest("form") ||
        element.parentElement
      );
    }

    function isSendButtonForComposer(button, composer) {
      if (!button || !(button instanceof HTMLButtonElement)) {
        return false;
      }
      if (button.disabled || !isElementVisible(button)) {
        return false;
      }
      const label = (
        button.getAttribute("aria-label") ||
        button.getAttribute("title") ||
        button.textContent ||
        ""
      ).trim();
      if (label !== "发送") {
        return false;
      }
      const root = findComposerRoot(composer);
      return Boolean(root && root.contains(button));
    }

    function findComposerSendButton() {
      const composer = findComposerElement();
      if (!composer) {
        return null;
      }
      return (
        Array.from(document.querySelectorAll("button")).find((button) =>
          isSendButtonForComposer(button, composer),
        ) || null
      );
    }
  `;
}

function buildPageSnapshotScript(recoveryExpectedText, longTurnLineCount) {
  return `(() => {
    ${composerDomHelperScript()}
    const composerCandidates = findComposerCandidates();
    const composer = findComposerElement();
    const stopButton =
      Array.from(document.querySelectorAll("button")).find(
        (button) => button.getAttribute("aria-label") === "停止",
      ) ?? null;
    const sendButton =
      Array.from(document.querySelectorAll("button")).find(
        (button) => button.getAttribute("title") === "发送",
      ) ?? null;
    const bodyText = document.body?.innerText || "";
    const streamLinePattern = /中断测试第\\s*\\d+\\s*行/;
    const finalLinePattern = new RegExp("中断测试第\\\\s*" + ${JSON.stringify(
      String(longTurnLineCount),
    )} + "\\\\s*行");
    const recoveryExpected = ${JSON.stringify(recoveryExpectedText)};
    const recoveryTextCount = recoveryExpected
      ? bodyText.split(recoveryExpected).length - 1
      : 0;
    const streamText = bodyText
      .split("\\n")
      .filter((line) => streamLinePattern.test(line))
      .join("\\n");
    return {
      href: window.location.href,
      ready: Boolean(composer) && isComposerWritable(composer),
      hasTextarea: Boolean(composer && composer.matches("textarea")),
      textareaValue: composer ? readComposerValue(composer) : "",
      composerCandidateCount: composerCandidates.length,
      composerDebug: composerCandidates.slice(-5).map((candidate) => {
        const rect = candidate.element.getBoundingClientRect();
        return {
          tag: candidate.element.tagName.toLowerCase(),
          name: candidate.element.getAttribute("name"),
          testId: candidate.element.getAttribute("data-testid"),
          placeholder: candidate.element.getAttribute("placeholder"),
          visible: candidate.visible,
          writable: candidate.writable,
          score: candidate.score,
          rect: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          },
        };
      }),
      sendDisabled: Boolean(sendButton?.disabled),
      stopVisible: Boolean(stopButton),
      longPromptVisible: bodyText.includes("E2E 中断测试"),
      stoppedMarker: bodyText.includes("用户已停止当前执行"),
      recoveryVisible: bodyText.includes(recoveryExpected),
      recoveryTextCount,
      streamLineCount: streamText ? streamText.split("\\n").filter(Boolean).length : 0,
      streamTextLength: streamText.length,
      finalLineSeen: finalLinePattern.test(streamText),
      buttons: Array.from(document.querySelectorAll("button")).map((button) => ({
        text: (button.textContent || "").trim(),
        aria: button.getAttribute("aria-label"),
        title: button.getAttribute("title"),
        disabled: Boolean(button.disabled),
      })),
    };
  })()`;
}

async function readPageSnapshot(page) {
  return page
    .evaluate(
      buildPageSnapshotScript(RECOVERY_EXPECTED_TEXT, LONG_TURN_LINE_COUNT),
    )
    .catch(() => null);
}

async function waitForComposerReady(page, timeoutMs = 60_000) {
  await page.waitForFunction(
    `(() => {
      ${composerDomHelperScript()}
      const composer = findComposerElement();
      return Boolean(composer && isComposerWritable(composer));
    })()`,
    null,
    { timeout: timeoutMs },
  );
}

async function fillComposer(page, value) {
  const result = await page.evaluate(
    new Function(
      "value",
      `
        ${composerDomHelperScript()}
        const composer = findComposerElement();
        if (!composer) {
          return { ok: false, reason: "composer-not-found" };
        }
        if (!isComposerWritable(composer)) {
          return { ok: false, reason: "composer-not-writable" };
        }
        composer.focus();
        if (composer instanceof HTMLTextAreaElement || composer instanceof HTMLInputElement) {
          const prototype = composer instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype
            : HTMLInputElement.prototype;
          const valueSetter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
          if (valueSetter) {
            valueSetter.call(composer, value);
          } else {
            composer.value = value;
          }
          composer.dispatchEvent(new InputEvent("input", {
            bubbles: true,
            inputType: "insertText",
            data: value,
          }));
          composer.dispatchEvent(new Event("change", { bubbles: true }));
          composer.setSelectionRange(value.length, value.length);
        } else {
          composer.textContent = value;
          composer.dispatchEvent(new InputEvent("input", {
            bubbles: true,
            inputType: "insertText",
            data: value,
          }));
          composer.dispatchEvent(new Event("change", { bubbles: true }));
        }
        return {
          ok: true,
          tag: composer.tagName.toLowerCase(),
          value: readComposerValue(composer),
        };
      `,
    ),
    value,
  );
  assert(result?.ok, `输入框填充失败: ${result?.reason || "unknown"}`);
  assert(
    String(result.value || "").includes(String(value).slice(0, 12)),
    "输入框填充后未读回目标内容",
  );
  return result;
}

async function clickComposerSendButton(page, timeoutMs = 30_000) {
  await page.waitForFunction(
    `(() => {
      ${composerDomHelperScript()}
      return Boolean(findComposerSendButton());
    })()`,
    null,
    { timeout: timeoutMs },
  );
  const result = await page.evaluate(`(() => {
    ${composerDomHelperScript()}
    const button = findComposerSendButton();
    if (!button) {
      return { ok: false, reason: "send-button-not-found" };
    }
    button.click();
    return { ok: true };
  })()`);
  assert(result?.ok, `点击输入框发送按钮失败: ${result?.reason || "unknown"}`);
}

async function submitComposer(
  page,
  value,
  observeSubmitted,
  timeoutMs = 30_000,
  submitLabel = "等待 composer button 触发 submit",
) {
  await waitForComposerReady(page, timeoutMs);
  await fillComposer(page, value);
  await page.keyboard.press("Enter");
  const submittedByEnter = await waitForCondition(
    "等待 Enter 触发 composer submit",
    observeSubmitted,
    1_500,
    100,
  ).catch(() => null);
  if (submittedByEnter) {
    return submittedByEnter;
  }

  const snapshotAfterEnter = await readPageSnapshot(page);
  if (
    snapshotAfterEnter?.textareaValue &&
    snapshotAfterEnter.textareaValue.includes(String(value).slice(0, 12))
  ) {
    await clickComposerSendButton(page, timeoutMs);
  } else {
    await fillComposer(page, value);
    await clickComposerSendButton(page, timeoutMs);
  }

  return await waitForCondition(submitLabel, observeSubmitted, timeoutMs, 250);
}

async function clickLastEnabledButton(page, label, timeoutMs = 30_000) {
  await page.waitForFunction(
    (buttonLabel) => {
      const buttons = Array.from(document.querySelectorAll("button"));
      const button = buttons
        .filter((item) => {
          const text = (item.textContent || "").trim();
          return (
            item.getAttribute("aria-label") === buttonLabel ||
            item.getAttribute("title") === buttonLabel ||
            text === buttonLabel
          );
        })
        .at(-1);
      return Boolean(button && !button.disabled);
    },
    label,
    { timeout: timeoutMs },
  );
  await page.getByRole("button", { name: label }).last().click({
    force: true,
    timeout: timeoutMs,
  });
}

function isLikelyDetachedBlankTaskSnapshot(snapshot) {
  return (
    Boolean(snapshot?.ready) &&
    !snapshot?.stopVisible &&
    !snapshot?.stoppedMarker &&
    !snapshot?.recoveryVisible &&
    Number(snapshot?.streamLineCount || 0) === 0 &&
    Number(snapshot?.streamTextLength || 0) === 0
  );
}

function recoveryResultVisible(snapshot, persisted = false) {
  return Boolean(
    snapshot?.recoveryVisible &&
    (Number(snapshot?.recoveryTextCount || 0) >= 1 || persisted),
  );
}

async function openSessionFromSidebar(page, sessionId) {
  return page
    .evaluate(
      ({ sessionId: targetSessionId }) => {
        const normalizedSessionId = String(targetSessionId || "").trim();
        if (!normalizedSessionId) {
          return { opened: false, reason: "missing-session-id" };
        }

        const buttons = Array.from(
          document.querySelectorAll(
            '[data-testid="app-sidebar-conversation-open"], button',
          ),
        );
        const target = buttons.find((button) => {
          const label = [
            button.getAttribute("data-session-id") || "",
            button.getAttribute("title") || "",
            button.getAttribute("aria-label") || "",
            button.textContent || "",
          ].join("\n");
          return label.includes(normalizedSessionId);
        });
        if (!(target instanceof HTMLElement)) {
          return { opened: false, reason: "session-entry-missing" };
        }
        target.click();
        return { opened: true, reason: "clicked-sidebar-session-id" };
      },
      { sessionId },
    )
    .catch((error) => ({
      opened: false,
      reason: error instanceof Error ? error.message : String(error),
    }));
}

async function waitForSessionFromSidebar(page, sessionId, timeoutMs = 20_000) {
  return waitForCondition(
    "等待目标会话出现在 sidebar",
    async () => {
      const result = await openSessionFromSidebar(page, sessionId);
      return result.opened ? result : null;
    },
    timeoutMs,
    250,
  ).catch((error) => ({
    opened: false,
    reason: error instanceof Error ? error.message : String(error),
  }));
}

function writeJsonFile(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function consoleNetworkSummary(consoleMessages, failedRequests) {
  const consoleErrors = consoleMessages.filter(
    (item) => item.type === "error" || item.type === "pageerror",
  );
  const benignConsoleErrors = consoleErrors.filter(isBenignConsoleError);
  const blockingConsoleErrors = consoleErrors.filter(
    (item) => !isBenignConsoleError(item),
  );
  const consoleWarnings = consoleMessages.filter(
    (item) => item.type === "warning",
  );
  const mockFallbackLines = consoleMessages
    .filter((item) => item.text.includes("[Mock]"))
    .map(normalizeConsoleLine);

  return {
    consoleErrors,
    benignConsoleErrors,
    blockingConsoleErrors,
    consoleWarnings,
    mockFallbackLines,
    networkErrorTop: Object.entries(
      failedRequests.reduce((acc, item) => {
        const key = `${item.failure} ${item.url}`;
        acc[key] = (acc[key] || 0) + 1;
        return acc;
      }, {}),
    )
      .sort((left, right) => right[1] - left[1])
      .slice(0, 12),
  };
}

function writeConsoleNetworkEvidence(
  evidenceDir,
  prefix,
  consoleMessages,
  invokes,
  failedRequests,
) {
  const summary = consoleNetworkSummary(consoleMessages, failedRequests);
  writeJsonFile(path.join(evidenceDir, `${prefix}-network-invoke.json`), {
    invokes,
    failedRequests,
  });
  fs.writeFileSync(
    path.join(evidenceDir, `${prefix}-console.txt`),
    [
      `Errors: ${summary.consoleErrors.length}`,
      `BlockingErrors: ${summary.blockingConsoleErrors.length}`,
      `BenignErrors: ${summary.benignConsoleErrors.length}`,
      `Warnings: ${summary.consoleWarnings.length}`,
      `MockLines: ${summary.mockFallbackLines.length}`,
      "",
      ...consoleMessages.map(normalizeConsoleLine),
      "",
    ].join("\n"),
    "utf8",
  );
  return summary;
}

function buildStorageBootstrapScript(storageMarker, storageOverrides) {
  return `(() => {
    const marker = ${JSON.stringify(storageMarker)};
    const overrides = ${JSON.stringify(storageOverrides)};
    const keys = Object.keys(overrides);
    if (!window.sessionStorage.getItem(marker)) {
      const original = {};
      for (const key of keys) {
        original[key] = window.localStorage.getItem(key);
      }
      window.sessionStorage.setItem(marker, JSON.stringify(original));
    }
    for (const [key, value] of Object.entries(overrides)) {
      window.localStorage.setItem(key, String(value));
    }
    return true;
  })()`;
}

function buildRestoreLocalStorageScript(storageMarker) {
  return `((marker) => {
    const raw = window.sessionStorage.getItem(marker);
    if (!raw) return false;
    const original = JSON.parse(raw);
    for (const [key, value] of Object.entries(original)) {
      if (value === null) {
        window.localStorage.removeItem(key);
      } else {
        window.localStorage.setItem(key, String(value));
      }
    }
    window.sessionStorage.removeItem(marker);
    return true;
  })(${JSON.stringify(storageMarker)})`;
}

async function launchPlaywrightContext() {
  const userDataDir = fs.mkdtempSync(
    path.join(os.tmpdir(), `lime-claw-chat-ready-streaming-${process.pid}-`),
  );
  const launchOptions = {
    headless: true,
    viewport: { width: 1440, height: 960 },
  };

  try {
    const context = await chromium.launchPersistentContext(userDataDir, {
      ...launchOptions,
      channel: "chrome",
    });
    return { context, userDataDir };
  } catch (chromeError) {
    console.warn(
      `[smoke:claw-chat-ready-streaming] Chrome channel 启动失败，尝试 Playwright 自带 Chromium: ${
        chromeError instanceof Error ? chromeError.message : String(chromeError)
      }`,
    );
    const context = await chromium.launchPersistentContext(
      userDataDir,
      launchOptions,
    );
    return { context, userDataDir };
  }
}

async function main() {
  if (typeof fetch !== "function") {
    throw new Error("当前 Node 运行时不支持 fetch，请使用 Node 18+");
  }

  const options = parseArgs(process.argv.slice(2));
  const prefix = options.prefix;
  const evidenceDir = options.evidenceDir;
  fs.mkdirSync(evidenceDir, { recursive: true });
  const hasExplicitProviderPreference = Boolean(
    options.providerPreference && options.modelPreference,
  );

  logStage("wait-health");
  const health = await waitForHealth(options);
  await sleep(POST_HEALTH_SETTLE_MS);

  logStage("prepare-runtime");
  const defaultProject =
    (await invoke(
      options,
      "get_or_create_default_project",
      undefined,
      30_000,
    ).catch(() => null)) || null;
  const workspaceId = defaultProject?.id || "default";
  const providerListResponse = await appServerRpc(
    options,
    APP_SERVER_METHOD_MODEL_PROVIDER_LIST,
    {},
    30_000,
  ).catch(() => ({ result: { providers: [] } }));
  const providers = Array.isArray(providerListResponse.result?.providers)
    ? providerListResponse.result.providers
    : [];
  const providerUiStateResponse = await appServerRpc(
    options,
    APP_SERVER_METHOD_MODEL_PROVIDER_UI_STATE_READ,
    { key: "agent-runtime-provider-ui-state" },
    30_000,
  ).catch(() => ({ result: { value: null } }));
  const providerUiStateValue = providerUiStateResponse.result?.value;
  let providerUiState = null;
  if (typeof providerUiStateValue === "string") {
    try {
      providerUiState = JSON.parse(providerUiStateValue);
    } catch {
      providerUiState = null;
    }
  }

  const enabledProviders = Array.isArray(providers)
    ? providers.filter((provider) => providerEnabled(provider))
    : [];
  const agentStatus = {
    initialized: true,
    provider_configured:
      hasExplicitProviderPreference || enabledProviders.length > 0,
    provider_name: options.providerPreference || null,
    provider_selector: options.providerPreference || null,
    model_name: options.modelPreference || null,
    source: "modelProvider/list",
  };
  assert(
    Boolean(agentStatus?.provider_configured) || enabledProviders.length > 0,
    `当前 Agent provider 未配置，无法执行 Claw 流式 smoke: ${JSON.stringify(agentStatus)}`,
  );
  const providerResolution = await resolveProviderPreference(
    options,
    agentStatus,
    enabledProviders,
  );
  const preferredProvider = providerResolution.providerPreference;
  const preferredModel = providerResolution.modelPreference;

  console.log(
    `[smoke:claw-chat-ready-streaming] live runtime provider: provider=${preferredProvider} model=${preferredModel} source=${providerResolution.source}`,
  );

  logStage("launch-browser");
  const { context, userDataDir } = await launchPlaywrightContext();
  let page = null;
  const consoleMessages = [];
  const invokes = [];
  const failedRequests = [];
  const storageMarker = `${prefix}:original-local-storage`;

  const summary = {
    scenarioId: "claw-chat-ready-streaming",
    prefix,
    verdict: "fail",
    appUrl: options.appUrl,
    bridge: health,
    e2eBoundary: {
      runner: "playwright-chromium",
      browserAutomation: "chromium.launchPersistentContext",
      currentBridge: "DevBridge transport=electron-host",
      provenPath:
        "Chromium GUI -> DevBridge -> Electron Desktop Host IPC -> App Server JSON-RPC -> RuntimeCore/backend",
      electronHostTransportRequired: true,
      electronRendererIpcE2e: false,
      electronLaunch: false,
      note: "该 smoke 证明浏览器 GUI 经 DevBridge(electron-host transport) 进入 Electron Desktop Host 和 App Server current 主链；它不是 Playwright _electron.launch() renderer IPC E2E。",
    },
    workspaceId,
    providerPreference: preferredProvider,
    modelPreference: preferredModel,
    providerSelectionBoundary: {
      providerReadMethods: [
        APP_SERVER_METHOD_MODEL_PROVIDER_LIST,
        APP_SERVER_METHOD_MODEL_PROVIDER_READ,
        APP_SERVER_METHOD_MODEL_PROVIDER_TEST_CHAT,
        APP_SERVER_METHOD_MODEL_PROVIDER_UI_STATE_READ,
      ],
      runtimeSelection: "runtimeOptions.runtimeRequest",
      backendProviderMutation: false,
      legacyConfigureProviderCommand: "retired-not-invoked",
    },
    agentStatusBefore: {
      initialized: Boolean(agentStatus?.initialized),
      provider_configured: Boolean(agentStatus?.provider_configured),
      provider_name: agentStatus?.provider_name || null,
      provider_selector: agentStatus?.provider_selector || null,
      model_name: agentStatus?.model_name || null,
      credential_uuid: providerUiState?.credential_uuid ? "[redacted]" : null,
    },
    steps: [],
  };
  let submittedSessionId = "";
  let submittedLongTurnId = "";
  let submittedFollowSessionId = "";
  let submittedFollowTurnId = "";
  let submittedLiveWebSessionId = "";
  let submittedLiveWebTurnId = "";
  const invokeEntryByRequest = new WeakMap();

  try {
    const scopedProviderKey = `agent_pref_provider_${workspaceId}`;
    const scopedModelKey = `agent_pref_model_${workspaceId}`;
    const scopedMigratedKey = `agent_pref_migrated_${workspaceId}`;
    const storageOverrides = {
      lime_onboarding_complete: "true",
      lime_onboarding_version: ONBOARDING_VERSION,
      lime_user_profile: "developer",
      [FAST_RESPONSE_MODE_STORAGE_KEY]: "off",
      agent_pref_provider: JSON.stringify(preferredProvider),
      agent_pref_model: JSON.stringify(preferredModel),
      agent_pref_provider_global: JSON.stringify(preferredProvider),
      agent_pref_model_global: JSON.stringify(preferredModel),
      [scopedProviderKey]: JSON.stringify(preferredProvider),
      [scopedModelKey]: JSON.stringify(preferredModel),
      [scopedMigratedKey]: JSON.stringify(true),
    };

    await context.addInitScript(
      buildStorageBootstrapScript(storageMarker, storageOverrides),
    );
    page = context.pages()[0] ?? (await context.newPage());

    page.on("console", (message) => {
      consoleMessages.push({
        type: message.type(),
        text: message.text(),
        location: message.location(),
      });
    });
    page.on("pageerror", (error) => {
      consoleMessages.push({
        type: "pageerror",
        text: error.message,
        location: {},
      });
    });
    page.on("request", (request) => {
      const requestUrl = request.url();
      if (
        request.method() !== "POST" ||
        (requestUrl !== options.invokeUrl && !requestUrl.endsWith("/invoke"))
      ) {
        return;
      }
      const postData = request.postData();
      if (!postData) {
        return;
      }
      try {
        const parsed = JSON.parse(postData);
        const entry = attachAppServerRequestMessages({
          at: new Date().toISOString(),
          cmd: parsed.cmd,
          args: parsed.args,
        });
        invokes.push(entry);
        invokeEntryByRequest.set(request, entry);
      } catch {
        const entry = {
          at: new Date().toISOString(),
          cmd: "<parse-error>",
          raw: postData,
        };
        invokes.push(entry);
        invokeEntryByRequest.set(request, entry);
      }
    });
    page.on("response", async (response) => {
      const request = response.request();
      const requestUrl = request.url();
      if (
        request.method() !== "POST" ||
        (requestUrl !== options.invokeUrl && !requestUrl.endsWith("/invoke"))
      ) {
        return;
      }
      const entry = invokeEntryByRequest.get(request);
      if (!entry) {
        return;
      }
      try {
        const text = await response.text();
        const payload = text ? JSON.parse(text) : null;
        entry.response = {
          at: new Date().toISOString(),
          status: response.status(),
          ok: response.ok(),
        };
        attachAppServerResponsePayload(entry, payload);
      } catch (error) {
        entry.response = {
          at: new Date().toISOString(),
          status: response.status(),
          ok: response.ok(),
          error: error instanceof Error ? error.message : String(error),
        };
      }
    });
    page.on("requestfailed", (request) => {
      failedRequests.push({
        url: request.url(),
        method: request.method(),
        failure: request.failure()?.errorText || "unknown",
      });
    });

    logStage("open-app");
    await page.goto(options.appUrl, { waitUntil: "commit", timeout: 60_000 });
    await page
      .waitForLoadState("networkidle", { timeout: 30_000 })
      .catch(() => undefined);
    await page
      .getByRole("button", { name: "新建任务" })
      .click({ timeout: 20_000 })
      .catch(() => undefined);

    logStage("wait-composer-ready");
    await waitForComposerReady(page, 60_000);
    const readySnapshot = await readPageSnapshot(page);
    assert(readySnapshot, "读取 ready 页面快照失败");
    summary.readySnapshot = readySnapshot;
    summary.steps.push("ready");
    const workspaceReadySignal = await waitForCondition(
      "等待默认工作区完成加载",
      async () => {
        const consoleText = readConsoleText(consoleMessages);
        if (
          consoleText.includes("AgentChatPage.loadData.projectOnlyComplete")
        ) {
          return "AgentChatPage.loadData.projectOnlyComplete";
        }
        if (consoleText.includes("AgentChatPage.loadData.projectLoaded")) {
          return "AgentChatPage.loadData.projectLoaded";
        }
        if (consoleText.includes("AgentChatPage.workspaceCheck.success")) {
          return "AgentChatPage.workspaceCheck.success";
        }
        if (consoleText.includes("AgentChatPage.loadData.noProject")) {
          const snapshot = await readPageSnapshot(page);
          const workspaceReady = appServerMethodSucceeded(
            invokes,
            APP_SERVER_METHOD_WORKSPACE_DEFAULT_ENSURE,
          );
          const skillsReady = appServerMethodSucceeded(
            invokes,
            APP_SERVER_METHOD_SKILL_MANAGEMENT_LIST,
          );
          if (snapshot?.ready && workspaceReady) {
            return skillsReady
              ? "AgentChatPage.loadData.noProject+skillManagement/list"
              : "AgentChatPage.loadData.noProject+workspace/default/ensure";
          }
        }
        return null;
      },
      90_000,
      250,
    );
    summary.workspaceReadySignal = workspaceReadySignal;
    summary.steps.push("workspace-ready");
    await page.screenshot({
      path: path.join(evidenceDir, `${prefix}-01-ready.png`),
      fullPage: true,
    });

    logStage("submit-long-turn");
    const longSubmitStart = invokes.length;
    const longSubmit = await submitComposer(
      page,
      LONG_PROMPT,
      () =>
        findAppServerMethodRecord(
          invokes.slice(longSubmitStart),
          APP_SERVER_METHOD_TURN_START,
          (record) =>
            appServerTurnInputText(record.params).includes("E2E 中断测试"),
          { direction: "request" },
        ),
      30_000,
      "等待长 turn App Server submit",
    );
    const longParams = longSubmit.params || {};
    const longRequest = {
      sessionId: appServerParamSessionId(longParams),
      turnId: appServerParamTurnId(longParams),
      runtimeRequest: runtimeRequestFromAppServerParams(longParams),
    };
    const sessionId = String(longRequest.sessionId || "");
    const longTurnId = String(longRequest.turnId || "");
    assert(sessionId, "长 turn 缺少 sessionId");
    assert(longTurnId, "长 turn 缺少 turnId");
    submittedSessionId = sessionId;
    submittedLongTurnId = longTurnId;
    summary.sessionId = sessionId;
    summary.longTurnId = longTurnId;
    summary.longSubmitAppServer = appServerTurnEvidenceFromRecord(longSubmit);
    summary.longSubmitRuntimeRequest = longRequest.runtimeRequest || null;
    summary.longTurnOpenAttempts = [
      await openSessionFromSidebar(page, sessionId),
    ];
    summary.longTurnVisibleSnapshot = await waitForCondition(
      "等待长 turn 会话挂载",
      async () => {
        const snapshot = await readPageSnapshot(page);
        return snapshot?.stopVisible || snapshot?.streamTextLength > 0
          ? snapshot
          : null;
      },
      10_000,
      250,
    ).catch((error) => {
      summary.longTurnVisibleWait = {
        passed: false,
        error: error instanceof Error ? error.message : String(error),
      };
      return null;
    });

    let lastOpenAttemptAt = Date.now();
    let longTurnFastCompletedBeforeInterrupt = false;
    const firstDelta = await waitForCondition(
      "等待首个流式增量与停止按钮",
      async () => {
        const snapshot = await readPageSnapshot(page);
        if (!snapshot) {
          return null;
        }
        if (
          isLikelyDetachedBlankTaskSnapshot(snapshot) &&
          Date.now() - lastOpenAttemptAt >= 2_000
        ) {
          lastOpenAttemptAt = Date.now();
          summary.longTurnOpenAttempts.push(
            await openSessionFromSidebar(page, sessionId),
          );
        }
        const session = await readAppServerSession(
          options,
          sessionId,
          20_000,
        ).catch(() => null);
        const turn = findTurn(session, longTurnId);
        if (turn && ["failed", "aborted"].includes(turn.status)) {
          summary.preStreamTurn = turn;
          const errorMessage = String(
            turn.error_message || turn.errorMessage || "",
          ).trim();
          throw new Error(
            `[smoke:claw-chat-ready-streaming] 长 turn 在首个流式增量前结束: status=${turn.status}${
              errorMessage ? ` error=${errorMessage}` : ""
            }`,
          );
        }
        if (turn?.status === "completed") {
          summary.preStreamTurn = turn;
          longTurnFastCompletedBeforeInterrupt = true;
          return {
            ...snapshot,
            runtimeStreamEventSeen: true,
            runtimeTurn: turn,
            runtimeTurnStatus: turn.status,
            fastCompletedBeforeInterrupt: true,
          };
        }
        const runtimeStreamEventSeen = consoleMessages.some(
          (item) =>
            String(item?.text || "").includes("AgentStream.firstEvent") ||
            String(item?.text || "").includes(
              "AgentStream.firstRuntimeStatus",
            ) ||
            String(item?.text || "").includes("AgentStream.runtimeKeepalive"),
        );
        return snapshot?.stopVisible &&
          (snapshot.streamTextLength > 0 || runtimeStreamEventSeen)
          ? {
              ...snapshot,
              runtimeStreamEventSeen,
              runtimeTurn: turn || null,
              runtimeTurnStatus: turn?.status || null,
            }
          : null;
      },
      90_000,
      250,
    );
    summary.firstDelta = firstDelta;
    summary.steps.push("running-stop-visible");
    await page.screenshot({
      path: path.join(evidenceDir, `${prefix}-02-running-stop-visible.png`),
      fullPage: true,
    });

    summary.runningTurnObserved = firstDelta.runtimeTurn || null;
    summary.longTurnFastCompletedBeforeInterrupt =
      longTurnFastCompletedBeforeInterrupt ||
      firstDelta.fastCompletedBeforeInterrupt === true;

    logStage("interrupt-long-turn");
    let latestSession = null;
    if (summary.longTurnFastCompletedBeforeInterrupt) {
      const [session, threadRead, snapshot] = await Promise.all([
        readAppServerSession(options, sessionId, 20_000).catch(() => null),
        readAppServerThreadRead(options, sessionId, 20_000).catch(() => null),
        readPageSnapshot(page),
      ]);
      const turn = findTurn(session, longTurnId) || firstDelta.runtimeTurn;
      latestSession = session;
      summary.interruptSkipped = {
        reason: "long-turn-fast-completed-before-stop",
        status: turn?.status || null,
        note: "Provider 在停止按钮可验证前完成了长 turn；继续验证恢复回合与 live WebSearch/WebFetch，不把此分支误判为产品卡死。",
      };
      summary.interruptedTurn = turn || null;
      summary.interruptedTurnStatus = turn?.status || null;
      summary.interruptedThreadRead = threadRead || null;
      summary.interruptedSnapshot = snapshot || null;
      summary.interruptHasTurnScope = false;
      summary.interruptDrainedInferred = false;
      await page.screenshot({
        path: path.join(evidenceDir, `${prefix}-03-after-stop.png`),
        fullPage: true,
      });
      summary.steps.push("fast-completed-before-stop");
    } else {
      const interruptStart = invokes.length;
      await clickLastEnabledButton(page, "停止");
      const interruptInvoke = await waitForCondition(
        "等待 App Server turn cancel invoke",
        () =>
          findAppServerMethodRecord(
            invokes.slice(interruptStart),
            APP_SERVER_METHOD_TURN_INTERRUPT,
            (record) => appServerParamSessionId(record.params) === sessionId,
            { direction: "request" },
          ),
        30_000,
        250,
      );
      const interruptParams = interruptInvoke.params || {};
      const interruptRequest = {
        session_id: appServerParamSessionId(interruptParams),
        turn_id: appServerParamTurnId(interruptParams),
      };
      summary.interruptRequest = interruptRequest;
      summary.interruptAppServer = {
        method: interruptInvoke.message?.method || "",
        sessionId: interruptRequest.session_id,
        turnId: interruptRequest.turn_id,
      };
      summary.interruptHasTurnScope =
        interruptRequest.session_id === sessionId &&
        interruptRequest.turn_id === longTurnId;
      await page.screenshot({
        path: path.join(evidenceDir, `${prefix}-03-after-stop.png`),
        fullPage: true,
      });
      summary.steps.push("after-stop");

      const interrupted = await waitForCondition(
        "等待 turn 中断完成",
        async () => {
          const [session, threadRead, snapshot] = await Promise.all([
            readAppServerSession(options, sessionId, 20_000).catch(() => null),
            readAppServerThreadRead(options, sessionId, 20_000).catch(
              () => null,
            ),
            readPageSnapshot(page),
          ]);
          const turn = findTurn(session, longTurnId);
          if (turn && isTerminalTurnStatus(turn.status)) {
            return { turn, session, threadRead, snapshot };
          }
          if (
            interruptDrainedWithoutRecordedTurn({
              session,
              threadRead,
              snapshot,
              interruptScoped: summary.interruptHasTurnScope,
            })
          ) {
            return {
              turn: syntheticInterruptedTurn(longTurnId, threadRead),
              session,
              threadRead,
              snapshot,
              inferredInterruptDrained: true,
            };
          }
          return null;
        },
        120_000,
        1_000,
      );
      latestSession = interrupted.session;
      summary.interruptedTurn = interrupted.turn;
      summary.interruptedTurnStatus = interrupted.turn?.status || null;
      summary.interruptedThreadRead = interrupted.threadRead || null;
      summary.interruptedSnapshot = interrupted.snapshot || null;
      summary.interruptDrainedInferred = Boolean(
        interrupted.inferredInterruptDrained,
      );
    }
    summary.queueCountAfterInterrupt = queuedTurnCount(latestSession);

    await waitForComposerReady(page, 60_000).catch(() => undefined);

    logStage("restore-interrupted-session-before-recovery");
    summary.beforeRecoveryOpenAttempt = await waitForSessionFromSidebar(
      page,
      sessionId,
      20_000,
    );
    summary.beforeRecoverySnapshot = await waitForCondition(
      "等待中断会话重新挂载",
      async () => {
        const snapshot = await readPageSnapshot(page);
        return snapshot &&
          !isLikelyDetachedBlankTaskSnapshot(snapshot) &&
          (snapshot.longPromptVisible ||
            snapshot.stoppedMarker ||
            snapshot.streamTextLength > 0)
          ? snapshot
          : null;
      },
      15_000,
      250,
    ).catch((error) => {
      summary.beforeRecoverySessionRestore = {
        passed: false,
        error: error instanceof Error ? error.message : String(error),
      };
      return null;
    });

    logStage("submit-recovery-turn");
    const followSubmitStart = invokes.length;
    const followSubmit = await submitComposer(
      page,
      RECOVERY_PROMPT,
      () =>
        findAppServerMethodRecord(
          invokes.slice(followSubmitStart),
          APP_SERVER_METHOD_TURN_START,
          (record) =>
            appServerTurnInputText(record.params).includes("停止后恢复测试"),
          { direction: "request" },
        ),
      30_000,
      "等待恢复 turn App Server submit",
    );
    const followParams = followSubmit.params || {};
    const followRequest = {
      sessionId: appServerParamSessionId(followParams),
      turnId: appServerParamTurnId(followParams),
      runtimeRequest: runtimeRequestFromAppServerParams(followParams),
    };
    const followSessionId = String(followRequest.sessionId || sessionId);
    const followTurnId = String(followRequest.turnId || "");
    assert(followSessionId, "恢复 turn 缺少 sessionId");
    assert(followTurnId, "恢复 turn 缺少 turnId");
    submittedFollowSessionId = followSessionId;
    submittedFollowTurnId = followTurnId;
    summary.followSessionId = followSessionId;
    summary.followTurnId = followTurnId;
    summary.followUsesOriginalSession = followSessionId === sessionId;
    summary.followSubmitAppServer =
      appServerTurnEvidenceFromRecord(followSubmit);
    summary.followSubmitRuntimeRequest = followRequest.runtimeRequest || null;

    const followCompleted = await waitForCondition(
      "等待恢复 turn 完成",
      async () => {
        const session = await readAppServerSession(
          options,
          followSessionId,
          20_000,
        ).catch(() => null);
        const turn = findTurn(session, followTurnId);
        return turn && isTerminalTurnStatus(turn.status)
          ? { turn, session }
          : null;
      },
      120_000,
      1_000,
    );
    latestSession = followCompleted.session || latestSession;
    summary.followTurn = followCompleted.turn;
    summary.followTurnStatus = followCompleted.turn?.status || null;
    const followAssistantText = [
      allAssistantText(latestSession),
      allAgentItemText(latestSession),
    ]
      .filter(Boolean)
      .join("\n");
    summary.followPersistedRecoveryBeforeGuiWait = followAssistantText.includes(
      RECOVERY_EXPECTED_TEXT,
    );
    let recoverySnapshot = null;
    try {
      recoverySnapshot = await waitForCondition(
        "等待 GUI 出现恢复结果",
        async () => {
          const snapshot = await readPageSnapshot(page);
          return recoveryResultVisible(
            snapshot,
            summary.followPersistedRecoveryBeforeGuiWait,
          )
            ? snapshot
            : null;
        },
        60_000,
        500,
      );
      summary.recoveryVisibleSource = "live-stream";
    } catch (recoveryWaitError) {
      const recoveryWaitMessage =
        recoveryWaitError instanceof Error
          ? recoveryWaitError.message
          : String(recoveryWaitError);
      summary.recoveryVisibleInitialWait = {
        passed: false,
        error: recoveryWaitMessage,
      };
      latestSession =
        (await readAppServerSession(options, followSessionId, 20_000).catch(
          () => null,
        )) || latestSession;
      const persistedAssistantText = [
        allAssistantText(latestSession),
        allAgentItemText(latestSession),
      ]
        .filter(Boolean)
        .join("\n");
      summary.recoveryPersistedBeforeRefresh = persistedAssistantText.includes(
        RECOVERY_EXPECTED_TEXT,
      );
      if (!summary.recoveryPersistedBeforeRefresh) {
        throw recoveryWaitError;
      }

      const detachedSnapshot = await readPageSnapshot(page);
      summary.recoveryDetachedSnapshot = detachedSnapshot;
      if (isLikelyDetachedBlankTaskSnapshot(detachedSnapshot)) {
        logStage("restore-session-after-recovery-persisted");
        summary.recoveryVisibleRestoreAttempt = await waitForSessionFromSidebar(
          page,
          followSessionId,
          20_000,
        );
        recoverySnapshot = await waitForCondition(
          "等待重新打开目标会话后 GUI 出现恢复结果",
          async () => {
            const snapshot = await readPageSnapshot(page);
            return recoveryResultVisible(
              snapshot,
              summary.recoveryPersistedBeforeRefresh,
            )
              ? snapshot
              : null;
          },
          45_000,
          500,
        );
        summary.recoveryVisibleSource =
          "post-session-restore-runtime-persistence";
      } else {
        logStage("refresh-after-recovery-persisted");
        await page.reload({ waitUntil: "commit", timeout: 60_000 });
        await page
          .waitForLoadState("networkidle", { timeout: 30_000 })
          .catch(() => undefined);
        recoverySnapshot = await waitForCondition(
          "等待刷新后 GUI 出现恢复结果",
          async () => {
            const snapshot = await readPageSnapshot(page);
            return snapshot?.recoveryVisible ? snapshot : null;
          },
          60_000,
          500,
        );
        summary.recoveryVisibleSource = "post-refresh-runtime-persistence";
      }
    }
    summary.recoverySnapshot = recoverySnapshot;
    summary.steps.push("recovery-final");
    await page.screenshot({
      path: path.join(evidenceDir, `${prefix}-04-recovery-final.png`),
      fullPage: true,
    });

    logStage("submit-live-web-tools-turn");
    const liveWebSubmitStart = invokes.length;
    const liveWebSubmittedAt = Date.now();
    const liveWebSubmit = await submitComposer(
      page,
      LIVE_WEB_TOOL_PROMPT,
      () =>
        findAppServerMethodRecord(
          invokes.slice(liveWebSubmitStart),
          APP_SERVER_METHOD_TURN_START,
          (record) =>
            appServerTurnInputText(record.params).includes("@搜索") &&
            appServerTurnInputText(record.params).includes("联网工具验证"),
          { direction: "request" },
        ),
      30_000,
    );
    const liveWebParams = liveWebSubmit.params || {};
    const liveWebRequest = {
      sessionId: appServerParamSessionId(liveWebParams) || followSessionId,
      turnId: appServerParamTurnId(liveWebParams),
      runtimeRequest: runtimeRequestFromAppServerParams(liveWebParams),
    };
    const liveWebSessionId = String(liveWebRequest.sessionId || "");
    const liveWebTurnId = String(liveWebRequest.turnId || "");
    assert(liveWebSessionId, "live WebSearch/WebFetch turn 缺少 sessionId");
    assert(liveWebTurnId, "live WebSearch/WebFetch turn 缺少 turnId");
    submittedLiveWebSessionId = liveWebSessionId;
    submittedLiveWebTurnId = liveWebTurnId;
    summary.liveWebToolPrompt = LIVE_WEB_TOOL_PROMPT;
    summary.liveWebSessionId = liveWebSessionId;
    summary.liveWebTurnId = liveWebTurnId;
    summary.liveWebSubmitAppServer =
      appServerTurnEvidenceFromRecord(liveWebSubmit);
    summary.liveWebSubmitRuntimeRequest = liveWebRequest.runtimeRequest || null;
    summary.liveWebSearchMode =
      liveWebRequest.runtimeRequest?.searchMode || null;
    assert(
      liveWebRequest.runtimeRequest?.webSearch === true,
      "live WebSearch/WebFetch turn 必须显式提交 webSearch=true",
    );
    assert(
      liveWebRequest.runtimeRequest?.searchMode === "required",
      'live WebSearch/WebFetch turn 必须显式提交 searchMode="required"',
    );

    const liveWebCompleted = await waitForCondition(
      "等待 live WebSearch/WebFetch turn 完成",
      async () => {
        const session = await readAppServerSession(
          options,
          liveWebSessionId,
          20_000,
        ).catch(() => null);
        const turn = findTurn(session, liveWebTurnId);
        if (!turn || !isTerminalTurnStatus(turn.status)) {
          return null;
        }
        return {
          turn,
          session,
          toolEvidence: liveWebToolEvidenceFromSession(session, {
            turnId: liveWebTurnId,
          }),
        };
      },
      180_000,
      1_000,
    );
    latestSession = liveWebCompleted.session || latestSession;
    summary.liveWebCompletedElapsedMs = Date.now() - liveWebSubmittedAt;
    summary.liveWebTurn = liveWebCompleted.turn;
    summary.liveWebTurnStatus = liveWebCompleted.turn?.status || null;
    summary.liveWebToolEvidence = liveWebCompleted.toolEvidence;
    if (!summary.liveWebToolEvidence?.allRequiredOutputPresentForTurn) {
      const settledToolEvidence = await waitForCondition(
        "等待 read model 出现 live WebSearch/WebFetch 工具输出事实",
        async () => {
          const session = await readAppServerSession(
            options,
            liveWebSessionId,
            20_000,
          ).catch(() => null);
          const toolEvidence = liveWebToolEvidenceFromSession(session, {
            turnId: liveWebTurnId,
          });
          summary.liveWebToolEvidenceWait = {
            passed: false,
            latest: toolEvidence,
          };
          return toolEvidence.allRequiredOutputPresentForTurn
            ? { session, toolEvidence }
            : null;
        },
        30_000,
        1_000,
      ).catch((toolEvidenceWaitError) => {
        summary.liveWebToolEvidenceWait = {
          passed: false,
          error:
            toolEvidenceWaitError instanceof Error
              ? toolEvidenceWaitError.message
              : String(toolEvidenceWaitError),
          latest: summary.liveWebToolEvidenceWait?.latest || null,
        };
        return null;
      });
      if (settledToolEvidence) {
        latestSession = settledToolEvidence.session || latestSession;
        summary.liveWebToolEvidence = settledToolEvidence.toolEvidence;
        summary.liveWebToolEvidenceWait = {
          passed: true,
          latest: settledToolEvidence.toolEvidence,
        };
      }
    }
    summary.steps.push("live-web-tools-final");
    await page.screenshot({
      path: path.join(evidenceDir, `${prefix}-05-live-web-tools-final.png`),
      fullPage: true,
    });

    logStage("collect-runtime-evidence");
    const runtimeReadSessionId = summary.liveWebSessionId || followSessionId;
    const threadRead = await readAppServerThreadRead(
      options,
      runtimeReadSessionId,
      20_000,
    ).catch((error) => ({
      __error: error instanceof Error ? error.message : String(error),
    }));
    const assistantText = [
      allAssistantText(latestSession),
      allAgentItemText(latestSession),
    ]
      .filter(Boolean)
      .join("\n");
    const {
      consoleErrors,
      benignConsoleErrors,
      blockingConsoleErrors,
      consoleWarnings,
      mockFallbackLines,
      networkErrorTop,
    } = consoleNetworkSummary(consoleMessages, failedRequests);
    const activeTurnId = activeTurnIdFromThreadRead(threadRead);
    const runtimeMockLines = mockFallbackLines.filter((line) =>
      /agent_runtime_(submit_turn|interrupt_turn|get_session|get_thread_read)/.test(
        line,
      ),
    );
    const peripheralMockLines = mockFallbackLines.filter(
      (line) => !runtimeMockLines.includes(line),
    );

    summary.threadRead = threadRead?.__error
      ? { error: threadRead.__error }
      : threadRead;
    summary.threadReadStatus = latestTurnStatusFromThreadRead(
      summary.threadRead,
    );
    const latestRuntimeRouting =
      latestSession?.execution_runtime?.routing_decision ||
      threadRead?.model_routing ||
      {};
    const longProviderModel = runtimeRequestProviderModel(
      longRequest.runtimeRequest,
    );
    summary.longRoutingEvidence = {
      provider: longProviderModel.provider,
      model: longProviderModel.model,
      source: longProviderModel.source,
      note:
        longProviderModel.source === "model-request-policy"
          ? "首次空任务 turn 使用 harness model_request_policy 锁定 provider/model，未重复写入顶层 runtimeRequest。"
          : "长 turn 显式提交 runtimeRequest provider/model。",
    };
    const followProviderPreferenceHonored =
      followRequest.runtimeRequest?.providerPreference === preferredProvider ||
      latestRuntimeRouting?.selectedProvider === preferredProvider ||
      latestRuntimeRouting?.requestedProvider === preferredProvider;
    const followModelPreferenceHonored =
      followRequest.runtimeRequest?.modelPreference === preferredModel ||
      latestRuntimeRouting?.selectedModel === preferredModel ||
      latestRuntimeRouting?.requestedModel === preferredModel;
    const liveWebProviderPreferenceHonored =
      liveWebRequest.runtimeRequest?.providerPreference === preferredProvider ||
      latestRuntimeRouting?.selectedProvider === preferredProvider ||
      latestRuntimeRouting?.requestedProvider === preferredProvider;
    const liveWebModelPreferenceHonored =
      liveWebRequest.runtimeRequest?.modelPreference === preferredModel ||
      latestRuntimeRouting?.selectedModel === preferredModel ||
      latestRuntimeRouting?.requestedModel === preferredModel;
    summary.followRoutingEvidence = {
      selectedProvider: latestRuntimeRouting?.selectedProvider || null,
      selectedModel: latestRuntimeRouting?.selectedModel || null,
      requestedProvider: latestRuntimeRouting?.requestedProvider || null,
      requestedModel: latestRuntimeRouting?.requestedModel || null,
      decisionSource: latestRuntimeRouting?.decisionSource || null,
      note:
        followRequest.runtimeRequest?.providerPreference ||
        followRequest.runtimeRequest?.modelPreference
          ? "恢复 turn 显式提交 provider/model。"
          : "恢复 turn 复用 session_default；以 runtime routing_decision 校验 selected/requested provider/model 未漂移。",
    };
    summary.queueCountFinal = queuedTurnCount(latestSession);
    summary.activeTurnIdFinal = activeTurnId;
    summary.assistantContainsRecovery = assistantText.includes(
      RECOVERY_EXPECTED_TEXT,
    );
    summary.runtimeStreamLineCount = (
      assistantText.match(/中断测试第\s*\d+\s*行/g) || []
    ).length;
    summary.interruptedAssistantContainsFinalLine = assistantText.includes(
      `中断测试第 ${LONG_TURN_LINE_COUNT} 行`,
    );
    summary.consoleErrorCount = consoleErrors.length;
    summary.blockingConsoleErrorCount = blockingConsoleErrors.length;
    summary.benignConsoleErrorCount = benignConsoleErrors.length;
    summary.consoleWarningCount = consoleWarnings.length;
    summary.networkErrorCount = failedRequests.length;
    summary.networkErrorTop = networkErrorTop;
    summary.devBridgeCommands = [...new Set(invokes.map((item) => item.cmd))];
    const liveWebReadAfterEvent = appServerSessionReadAfterEventEvidence(
      invokes,
      {
        sessionId: submittedLiveWebSessionId,
        turnId: submittedLiveWebTurnId,
      },
      { strictEventScope: true },
    );
    const liveWebSessionReadTurn = appServerSessionReadTurnEvidence(invokes, {
      sessionId: submittedLiveWebSessionId,
      turnId: submittedLiveWebTurnId,
    });
    const liveWebCurrentStreamingEvent = appServerCurrentEventEvidence(
      invokes,
      consoleMessages,
      {
        sessionId: submittedLiveWebSessionId,
        turnId: submittedLiveWebTurnId,
      },
    );
    const liveWebToolStreamEvidence = liveWebToolStreamEvidenceFromEvents(
      appServerEventRecords(invokes),
      {
        sessionId: submittedLiveWebSessionId,
        turnId: submittedLiveWebTurnId,
      },
    );
    const liveWebSearchReadModelCompleted = Boolean(
      summary.liveWebToolEvidence?.requiredForTurn?.find(
        (item) => item.name === "WebSearch",
      )?.completed,
    );
    const liveWebFetchReadModelCompleted = Boolean(
      summary.liveWebToolEvidence?.requiredForTurn?.find(
        (item) => item.name === "WebFetch",
      )?.completed,
    );
    const liveWebRequiredReadModelToolsCompleted = Boolean(
      summary.liveWebToolEvidence?.allRequiredCompletedForTurn,
    );
    const liveWebRequiredReadModelToolOutputsPresent = Boolean(
      summary.liveWebToolEvidence?.allRequiredOutputPresentForTurn,
    );
    const liveWebAppServerEvent =
      appServerEventRecords(invokes).find((record) =>
        eventRecordStrictlyMatchesTurn(record, {
          sessionId: submittedLiveWebSessionId,
          turnId: submittedLiveWebTurnId,
        }),
      ) || null;
    const appServerReadAfterEvent =
      liveWebReadAfterEvent ||
      appServerSessionReadAfterEventEvidence(invokes, {
        sessionId: followSessionId,
        turnId: followTurnId,
      }) ||
      appServerSessionReadAfterEventEvidence(invokes, {
        sessionId,
        turnId: longTurnId,
      });
    const appServerSessionReadTurn =
      liveWebSessionReadTurn ||
      appServerSessionReadTurnEvidence(invokes, {
        sessionId: followSessionId,
        turnId: followTurnId,
      }) ||
      appServerSessionReadTurnEvidence(invokes, {
        sessionId,
        turnId: longTurnId,
      });
    const currentStreamingEvent =
      liveWebCurrentStreamingEvent ||
      appServerCurrentEventEvidence(invokes, consoleMessages, {
        sessionId: followSessionId,
        turnId: followTurnId,
      }) ||
      appServerCurrentEventEvidence(invokes, consoleMessages, {
        sessionId,
        turnId: longTurnId,
      });
    const runtimeRecoveryEvidence = {
      source: summary.assistantContainsRecovery
        ? "app-server-session-read-assistant-text"
        : summary.followTurnStatus === "completed"
          ? "app-server-session-read-turn-completed-gui-stream-text"
          : "missing",
      assistantTextPersistedInRuntimeRead: summary.assistantContainsRecovery,
      followTurnCompletedInRuntimeRead:
        summary.followTurnStatus === "completed",
      recoveryVisibleInGui: Boolean(recoverySnapshot?.recoveryVisible),
      note: summary.assistantContainsRecovery
        ? "App Server read model 已包含恢复正文。"
        : "当前 App Server read model 只返回 turns；恢复正文由 streaming event 渲染到 GUI，因此以 turn completed + GUI 可见证明恢复闭环。",
    };
    summary.appServerMethods = [
      ...new Set(
        invokes.flatMap((item) =>
          [
            ...(item.appServer?.requestMessages || []),
            ...(item.appServer?.responseMessages || []),
            ...(item.appServer?.drainMessages || []),
          ]
            .map((message) => message?.method)
            .filter(Boolean),
        ),
      ),
    ];
    summary.appServerEvidence = {
      turnStartCount: appServerMethodRecords(
        invokes,
        APP_SERVER_METHOD_TURN_START,
        { direction: "request" },
      ).length,
      turnCancelCount: appServerMethodRecords(
        invokes,
        APP_SERVER_METHOD_TURN_INTERRUPT,
        { direction: "request" },
      ).length,
      sessionReadCount: appServerMethodRecords(
        invokes,
        APP_SERVER_METHOD_THREAD_READ,
        { direction: "request" },
      ).length,
      eventNotificationCount: appServerMethodRecords(
        invokes,
        APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        { direction: "response" },
      ).length,
      drainEventNotificationCount: appServerDrainEventRecords(invokes).length,
      liveWebReadAfterEvent,
      liveWebSessionReadTurn,
      liveWebAppServerEvent,
      liveWebCurrentStreamingEvent,
      liveWebToolStreamEvidence,
      readAfterEvent: appServerReadAfterEvent,
      sessionReadTurn: appServerSessionReadTurn,
      currentStreamingEvent,
      runtimeRecovery: runtimeRecoveryEvidence,
    };
    summary.mockFallbackLines = mockFallbackLines;
    const longTurnFastCompleteAccepted =
      summary.longTurnFastCompletedBeforeInterrupt === true &&
      summary.interruptedTurnStatus === "completed";
    const longTurnCanBeInterrupted =
      Boolean(firstDelta?.stopVisible) &&
      appServerMethodSeen(
        invokes,
        APP_SERVER_METHOD_TURN_INTERRUPT,
        {
          direction: "request",
        },
      ) &&
      Boolean(summary.interruptHasTurnScope) &&
      isInterruptedTurnStatus(summary.interruptedTurnStatus);
    summary.assertions = {
      workspaceReady: Boolean(readySnapshot?.ready),
      devBridgeHealthy: health?.status === "ok" || Boolean(health),
      electronHostBridge: health?.transport === "electron-host",
      streamFirstDeltaSeen: Boolean(
        firstDelta?.streamTextLength > 0 || firstDelta?.runtimeStreamEventSeen,
      ),
      streamGrowthObserved:
        Number(recoverySnapshot?.streamTextLength || 0) >=
          Number(firstDelta?.streamTextLength || 0) ||
        summary.runtimeStreamLineCount >=
          Number(firstDelta?.streamLineCount || 1),
      stopButtonVisible: Boolean(firstDelta?.stopVisible),
      longTurnFastCompleteAccepted,
      longTurnCanBeInterrupted,
      appServerTurnStartSeen: appServerMethodSeen(
        invokes,
        APP_SERVER_METHOD_TURN_START,
        { direction: "request" },
      ),
      appServerTurnCancelSeen: longTurnFastCompleteAccepted
        ? false
        : appServerMethodSeen(
            invokes,
            APP_SERVER_METHOD_TURN_INTERRUPT,
            { direction: "request" },
          ),
      appServerSessionReadSeen: appServerMethodSeen(
        invokes,
        APP_SERVER_METHOD_THREAD_READ,
        { direction: "request" },
      ),
      appServerEventSeen:
        appServerMethodSeen(invokes, APP_SERVER_METHOD_AGENT_SESSION_EVENT, {
          direction: "response",
        }) || Boolean(currentStreamingEvent),
      appServerSessionReadAfterEventSeen: Boolean(
        appServerReadAfterEvent || appServerSessionReadTurn,
      ),
      liveWebAppServerEventSeen: Boolean(liveWebAppServerEvent),
      liveWebSessionReadAfterEventSeen: Boolean(
        liveWebReadAfterEvent || liveWebSessionReadTurn,
      ),
      liveWebSearchToolEventsSeen: Boolean(
        (liveWebToolStreamEvidence.required.find(
          (item) => item.name === "WebSearch",
        )?.started &&
          liveWebToolStreamEvidence.required.find(
            (item) => item.name === "WebSearch",
          )?.result) ||
        liveWebSearchReadModelCompleted,
      ),
      liveWebFetchToolEventsSeen: Boolean(
        (liveWebToolStreamEvidence.required.find(
          (item) => item.name === "WebFetch",
        )?.started &&
          liveWebToolStreamEvidence.required.find(
            (item) => item.name === "WebFetch",
          )?.result) ||
        liveWebFetchReadModelCompleted,
      ),
      liveWebRequiredToolEventsSeen: Boolean(
        liveWebToolStreamEvidence.allRequiredToolEventsForTurn ||
        liveWebRequiredReadModelToolsCompleted,
      ),
      liveWebRequiredToolEventOutputsPresent: Boolean(
        liveWebToolStreamEvidence.allRequiredOutputPresentForTurn ||
        liveWebRequiredReadModelToolOutputsPresent,
      ),
      liveWebRequiredToolEventOrderValid: Boolean(
        liveWebToolStreamEvidence.allRequiredResultAfterStartForTurn ||
        liveWebRequiredReadModelToolsCompleted,
      ),
      liveWebTurnCompletedEventSeen: Boolean(
        liveWebToolStreamEvidence.terminalEventSeen ||
        summary.liveWebTurnStatus === "completed",
      ),
      interruptScopedToLongTurn: longTurnFastCompleteAccepted
        ? false
        : Boolean(summary.interruptHasTurnScope),
      interruptedTurnCanceled: longTurnFastCompleteAccepted
        ? false
        : isInterruptedTurnStatus(summary.interruptedTurnStatus),
      recoveryTurnCompleted: summary.followTurnStatus === "completed",
      recoveryPersistedInRuntime:
        summary.assistantContainsRecovery ||
        (summary.followTurnStatus === "completed" &&
          Boolean(recoverySnapshot?.recoveryVisible)),
      recoveryVisibleInGui: Boolean(recoverySnapshot?.recoveryVisible),
      longProviderPreferenceHonored:
        longProviderModel.provider === preferredProvider,
      longModelPreferenceHonored: longProviderModel.model === preferredModel,
      followProviderPreferenceHonored,
      followModelPreferenceHonored,
      liveWebProviderPreferenceHonored,
      liveWebModelPreferenceHonored,
      liveWebTurnCompleted: summary.liveWebTurnStatus === "completed",
      liveWebSearchCompleted: liveWebSearchReadModelCompleted,
      liveWebFetchCompleted: liveWebFetchReadModelCompleted,
      liveWebRequiredToolsCompleted: liveWebRequiredReadModelToolsCompleted,
      liveWebRequiredToolOutputsPresent:
        liveWebRequiredReadModelToolOutputsPresent,
      liveWebExplicitSearchRequired:
        liveWebRequest.runtimeRequest?.webSearch === true &&
        liveWebRequest.runtimeRequest?.searchMode === "required",
      noRuntimeMockFallbackSeen: runtimeMockLines.length === 0,
      noBlockingConsoleErrors: blockingConsoleErrors.length === 0,
    };
    summary.mockFallbackClassification = {
      runtimeMockLines,
      peripheralMockLines,
      note:
        runtimeMockLines.length === 0
          ? "未观察到聊天 runtime submit/interrupt/get_session/get_thread_read 的 mock fallback；如有 [Mock] 日志，仅属周边 web-mode 能力。"
          : "观察到聊天 runtime 关键命令 mock fallback，需视为真实路径失败。",
    };
    summary.consoleNetwork = {
      consoleErrorCount: consoleErrors.length,
      blockingConsoleErrorCount: blockingConsoleErrors.length,
      benignConsoleErrorCount: benignConsoleErrors.length,
      consoleWarningCount: consoleWarnings.length,
      networkErrorCount: failedRequests.length,
      networkErrorTop: summary.networkErrorTop,
      note:
        failedRequests.length === 0
          ? "未观察到 requestfailed。"
          : "requestfailed 需结合 URL 判断是否为页面 reload/close 导致的 abort，不能直接等同产品失败。",
    };
    summary.evidenceFiles = {
      screenshots: [
        `${prefix}-01-ready.png`,
        `${prefix}-02-running-stop-visible.png`,
        `${prefix}-03-after-stop.png`,
        `${prefix}-04-recovery-final.png`,
        `${prefix}-05-live-web-tools-final.png`,
      ],
      console: `${prefix}-console.txt`,
      network: `${prefix}-network-invoke.json`,
      runtimeSession: `${prefix}-runtime-session.json`,
      threadRead: `${prefix}-thread-read.json`,
      summary: `${prefix}-summary.json`,
    };

    summary.verdict =
      summary.assertions.workspaceReady &&
      summary.assertions.devBridgeHealthy &&
      summary.assertions.electronHostBridge &&
      summary.assertions.streamFirstDeltaSeen &&
      summary.assertions.streamGrowthObserved &&
      summary.assertions.appServerTurnStartSeen &&
      summary.assertions.appServerSessionReadSeen &&
      summary.assertions.appServerEventSeen &&
      summary.assertions.appServerSessionReadAfterEventSeen &&
      (summary.assertions.longTurnCanBeInterrupted ||
        summary.assertions.longTurnFastCompleteAccepted) &&
      summary.assertions.recoveryTurnCompleted &&
      summary.assertions.recoveryPersistedInRuntime &&
      summary.assertions.recoveryVisibleInGui &&
      summary.assertions.longProviderPreferenceHonored &&
      summary.assertions.longModelPreferenceHonored &&
      summary.assertions.followProviderPreferenceHonored &&
      summary.assertions.followModelPreferenceHonored &&
      summary.assertions.liveWebProviderPreferenceHonored &&
      summary.assertions.liveWebModelPreferenceHonored &&
      summary.assertions.liveWebTurnCompleted &&
      summary.assertions.liveWebAppServerEventSeen &&
      summary.assertions.liveWebSessionReadAfterEventSeen &&
      summary.assertions.liveWebSearchToolEventsSeen &&
      summary.assertions.liveWebFetchToolEventsSeen &&
      summary.assertions.liveWebRequiredToolEventsSeen &&
      summary.assertions.liveWebRequiredToolEventOutputsPresent &&
      summary.assertions.liveWebRequiredToolEventOrderValid &&
      summary.assertions.liveWebTurnCompletedEventSeen &&
      summary.assertions.liveWebSearchCompleted &&
      summary.assertions.liveWebFetchCompleted &&
      summary.assertions.liveWebRequiredToolsCompleted &&
      summary.assertions.liveWebRequiredToolOutputsPresent &&
      summary.assertions.liveWebExplicitSearchRequired &&
      summary.assertions.noRuntimeMockFallbackSeen &&
      summary.assertions.noBlockingConsoleErrors &&
      summary.queueCountFinal === 0 &&
      summary.activeTurnIdFinal === null &&
      summary.blockingConsoleErrorCount === 0
        ? "pass"
        : "fail";

    writeJsonFile(
      path.join(evidenceDir, `${prefix}-runtime-session.json`),
      latestSession,
    );
    writeJsonFile(
      path.join(evidenceDir, `${prefix}-thread-read.json`),
      threadRead,
    );
    writeConsoleNetworkEvidence(
      evidenceDir,
      prefix,
      consoleMessages,
      invokes,
      failedRequests,
    );
    writeJsonFile(path.join(evidenceDir, `${prefix}-summary.json`), summary);
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    if (page) {
      summary.failureSnapshot = await readPageSnapshot(page);
      await page
        .screenshot({
          path: path.join(evidenceDir, `${prefix}-99-failure.png`),
          fullPage: true,
        })
        .catch(() => undefined);
    }
    const failureEvidence = {
      screenshots: [`${prefix}-99-failure.png`],
      console: `${prefix}-console.txt`,
      network: `${prefix}-network-invoke.json`,
      runtimeSession: `${prefix}-runtime-session.json`,
      threadRead: `${prefix}-thread-read.json`,
      summary: `${prefix}-summary.json`,
    };
    summary.evidenceFiles = summary.evidenceFiles || failureEvidence;
    const failureConsoleNetwork = writeConsoleNetworkEvidence(
      evidenceDir,
      prefix,
      consoleMessages,
      invokes,
      failedRequests,
    );
    summary.consoleErrorCount = failureConsoleNetwork.consoleErrors.length;
    summary.blockingConsoleErrorCount =
      failureConsoleNetwork.blockingConsoleErrors.length;
    summary.benignConsoleErrorCount =
      failureConsoleNetwork.benignConsoleErrors.length;
    summary.consoleWarningCount = failureConsoleNetwork.consoleWarnings.length;
    summary.networkErrorCount = failedRequests.length;
    summary.networkErrorTop = failureConsoleNetwork.networkErrorTop;
    summary.devBridgeCommands = [...new Set(invokes.map((item) => item.cmd))];
    summary.appServerMethods = [
      ...new Set(
        invokes.flatMap((item) =>
          [
            ...(item.appServer?.requestMessages || []),
            ...(item.appServer?.responseMessages || []),
          ]
            .map((message) => message?.method)
            .filter(Boolean),
        ),
      ),
    ];

    if (submittedSessionId) {
      summary.failureCleanupInterrupt = await cancelAppServerTurn(
        options,
        submittedSessionId,
        submittedLongTurnId,
        20_000,
      )
        .then(() => ({ attempted: true, status: "sent" }))
        .catch((interruptError) => ({
          attempted: true,
          status: "failed",
          error:
            interruptError instanceof Error
              ? interruptError.message
              : String(interruptError),
        }));
      if (
        submittedFollowSessionId &&
        (submittedFollowSessionId !== submittedSessionId ||
          submittedFollowTurnId !== submittedLongTurnId)
      ) {
        summary.failureCleanupFollowInterrupt = await cancelAppServerTurn(
          options,
          submittedFollowSessionId,
          submittedFollowTurnId,
          20_000,
        )
          .then(() => ({ attempted: true, status: "sent" }))
          .catch((interruptError) => ({
            attempted: true,
            status: "failed",
            error:
              interruptError instanceof Error
                ? interruptError.message
                : String(interruptError),
          }));
      }
      if (
        submittedLiveWebSessionId &&
        (submittedLiveWebSessionId !== submittedFollowSessionId ||
          submittedLiveWebTurnId !== submittedFollowTurnId)
      ) {
        summary.failureCleanupLiveWebInterrupt = await cancelAppServerTurn(
          options,
          submittedLiveWebSessionId,
          submittedLiveWebTurnId,
          20_000,
        )
          .then(() => ({ attempted: true, status: "sent" }))
          .catch((interruptError) => ({
            attempted: true,
            status: "failed",
            error:
              interruptError instanceof Error
                ? interruptError.message
                : String(interruptError),
          }));
      }
      const failureSessionId =
        submittedLiveWebSessionId ||
        submittedFollowSessionId ||
        submittedSessionId;
      const failureSession = await readAppServerSession(
        options,
        failureSessionId,
        20_000,
      ).catch((sessionError) => ({
        __error:
          sessionError instanceof Error
            ? sessionError.message
            : String(sessionError),
      }));
      const failureThreadRead = await readAppServerThreadRead(
        options,
        failureSessionId,
        20_000,
      ).catch((threadError) => ({
        __error:
          threadError instanceof Error
            ? threadError.message
            : String(threadError),
      }));
      writeJsonFile(
        path.join(evidenceDir, `${prefix}-runtime-session.json`),
        failureSession,
      );
      writeJsonFile(
        path.join(evidenceDir, `${prefix}-thread-read.json`),
        failureThreadRead,
      );
      summary.failureRuntime = {
        sessionId: failureSessionId,
        sessionError: failureSession?.__error || null,
        threadReadError: failureThreadRead?.__error || null,
        turns: Array.isArray(failureSession?.turns)
          ? failureSession.turns.map((turn) => ({
              id: turn?.id || null,
              status: turn?.status || null,
              error: turn?.error || turn?.error_message || null,
            }))
          : [],
        latestTurnStatus:
          failureThreadRead?.diagnostics?.latest_turn_status ||
          failureThreadRead?.runtime_summary?.latestTurnStatus ||
          failureThreadRead?.status ||
          null,
      };
    }
    writeJsonFile(path.join(evidenceDir, `${prefix}-summary.json`), summary);
    throw error;
  } finally {
    try {
      if (page) {
        await page
          .evaluate(buildRestoreLocalStorageScript(storageMarker))
          .catch(() => undefined);
      }
    } finally {
      await context.close().catch(() => undefined);
      fs.rmSync(userDataDir, { recursive: true, force: true });
    }
  }

  console.log(JSON.stringify(summary, null, 2));
  if (summary.verdict !== "pass") {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  const detail = error instanceof Error ? error.message : String(error);
  console.error(`[smoke:claw-chat-ready-streaming] ${detail}`);
  process.exit(1);
});
