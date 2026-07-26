#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import electronPath from "electron";
import { _electron as electron, chromium } from "playwright";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { ensureElectronFixtureBuild } from "../lib/electron-fixture-build.mjs";
import { createTempRuntimeEnv } from "./claw-chat-current-fixture-backend-file.mjs";
import {
  waitForBackendLedgerEntry,
  waitForBackendLedgerTurnStart,
  sanitizeBackendLedgerForEvidence,
  summarizeBackendLedger,
} from "./claw-chat-current-fixture-backend-ledger.mjs";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  APP_SERVER_METHOD_SESSION_LIST,
  APP_SERVER_METHOD_SESSION_READ,
  APP_SERVER_METHOD_SESSION_START,
  APP_SERVER_METHOD_SESSION_THREAD_RESUME,
  APP_SERVER_METHOD_SESSION_TURN_CANCEL,
  APP_SERVER_METHOD_SESSION_TURN_START,
  APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE,
  FIXTURE_MODEL,
  FIXTURE_PROVIDER,
  NEWS_PROMPT,
  SESSION_ID,
  SESSION_TITLE,
  THREAD_ID,
} from "./claw-chat-current-fixture-constants.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import {
  waitForGuiChatCanceled,
  waitForStopButtonVisibleAndClick,
} from "./claw-chat-current-fixture-gui-completion-waits.mjs";
import {
  collectReadModelTurns,
  readModelLatestTurnStatus,
  readModelTurnId,
  readModelTurnStatus,
  summarizeReadModelQueueState,
} from "./claw-chat-current-fixture-read-model-core.mjs";
import { waitForSessionReadCanceled } from "./claw-chat-current-fixture-read-model-waits.mjs";
import {
  bindGuiWorkspaceAndModelPreferences,
  clearInvokeBuffers,
  collectRuntimeEvents,
  drainAppServerEventsFromPage,
  ensureDefaultWorkspace,
  initializeAppServer,
  invokeAppServerFromPage,
  readTraceMessages,
  reloadRendererDocument,
  summarizeRuntimeEvents,
  waitForAppUrlReady,
  waitForRendererReady,
} from "./claw-chat-current-fixture-rpc.mjs";
import {
  createFixtureSession,
  navigateGuiToWorkspaceScopedAgent,
  openFixtureSessionFromSidebar,
  waitForGuiSessionVisible,
} from "./claw-chat-current-fixture-session.mjs";
import {
  assert,
  cleanupTempRoot,
  readJsonl,
  sanitizeJson,
  sanitizeText,
  sleep,
  writeJsonFile,
} from "./claw-chat-current-fixture-utils.mjs";

const LOG_PREFIX = "[cdp:reopen-running-turn]";
const DEFAULTS = {
  appUrl: "",
  cdpPort: 9239,
  cdpUrl: "",
  evidenceDir: path.join(process.cwd(), ".lime", "cdp-evidence"),
  prefix: "reopen-running-turn-cdp-gate",
  presentationMode: "foreground",
  reopenMode: "reload",
  timeoutMs: 180_000,
  intervalMs: 250,
  keepTemp: false,
  multiRunningSessions: false,
};
const REOPEN_MODES = new Set(["reload", "restart"]);
const PRESENTATION_MODES = new Set(["foreground", "background"]);
const MULTI_RUNNING_SECONDARY_SESSION_ID = "sess_cdp_multi_running_secondary";
const MULTI_RUNNING_SECONDARY_THREAD_ID = "thread_cdp_multi_running_secondary";
const MULTI_RUNNING_SECONDARY_TITLE = "CDP 多运行会话后台任务";
const MULTI_RUNNING_SECONDARY_PROMPT = "整理今天的国际新闻（后台第二会话）";

function printHelp() {
  console.log(`
Reopen Running Turn CDP Gate

用途:
  启动真实 Electron Desktop Host，通过 chromium.connectOverCDP attach
  真实 renderer，使用 controlled external fixture 制造 running turn，
  然后 reload renderer 或重启 Electron，验证同一 sessionId/turnId 的
  主区、侧栏、输入框运行态保持一致。reload 模式还要求产品恢复逻辑
  自动调用 thread/resume；restart 模式只声明 cold-start
  后 read model / GUI 恢复，不声明 external backend 子进程跨重启存活。

边界:
  这是 Gate B controlled fixture：覆盖 Electron/preload/IPC、
  app_server_handle_json_lines、App Server JSON-RPC、read model、GUI 可见状态。
  不调用正式模型后端，不证明 live Provider。

用法:
  node scripts/agent-runtime/reopen-running-turn-cdp-gate.mjs

选项:
  --app-url <url>        可选 renderer dev server，例如 http://127.0.0.1:1420/
  --cdp-port <port>      Electron remote debugging port，默认 ${DEFAULTS.cdpPort}
  --cdp-url <url>        显式 CDP endpoint；默认由 cdp-port 生成
  --evidence-dir <path>  证据目录，默认 .lime/cdp-evidence
  --prefix <name>        证据文件前缀
  --presentation-mode <mode>
                         foreground 或 background，默认 ${DEFAULTS.presentationMode}
  --reopen-mode <mode>   reload 或 restart，默认 ${DEFAULTS.reopenMode}
  --timeout-ms <ms>      总超时，默认 ${DEFAULTS.timeoutMs}
  --interval-ms <ms>     轮询间隔，默认 ${DEFAULTS.intervalMs}
  --multi-running-sessions
                         额外创建一个后台 running session，验证多未完成会话侧栏状态隔离
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
    if (arg === "--cdp-port" && next) {
      options.cdpPort = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--cdp-url" && next) {
      options.cdpUrl = next.trim();
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
    if (arg === "--presentation-mode" && next) {
      options.presentationMode = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--reopen-mode" && next) {
      options.reopenMode = next.trim();
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
    if (arg === "--multi-running-sessions") {
      options.multiRunningSessions = true;
      continue;
    }
    throw new Error(`未知参数: ${arg}`);
  }
  if (!Number.isFinite(options.cdpPort) || options.cdpPort < 1) {
    throw new Error("--cdp-port 必须是有效端口");
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!options.cdpUrl) {
    options.cdpUrl = `http://127.0.0.1:${options.cdpPort}`;
  }
  if (!options.prefix) {
    throw new Error("--prefix 不能为空");
  }
  if (!PRESENTATION_MODES.has(options.presentationMode)) {
    throw new Error("--presentation-mode 必须是 foreground 或 background");
  }
  if (!REOPEN_MODES.has(options.reopenMode)) {
    throw new Error("--reopen-mode 必须是 reload 或 restart");
  }
  return options;
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

async function fetchJson(url) {
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}`);
  }
  return await response.json();
}

function summarizeCdpVersion(version) {
  return sanitizeJson({
    browser: version?.Browser ?? version?.browser ?? null,
    protocolVersion: version?.["Protocol-Version"] ?? null,
    userAgent: version?.["User-Agent"] ?? null,
    webSocketDebuggerUrl: version?.webSocketDebuggerUrl ? "present" : null,
  });
}

function summarizeCdpTargets(targets) {
  return (Array.isArray(targets) ? targets : []).map((target) =>
    sanitizeJson({
      id: target?.id ?? null,
      type: target?.type ?? null,
      title: target?.title ?? null,
      url: target?.url ?? null,
    }),
  );
}

async function waitForCdpEndpoint(options) {
  const startedAt = Date.now();
  let lastError = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const [version, targets] = await Promise.all([
        fetchJson(`${options.cdpUrl}/json/version`),
        fetchJson(`${options.cdpUrl}/json/list`),
      ]);
      return {
        url: options.cdpUrl,
        waitedMs: Date.now() - startedAt,
        version: summarizeCdpVersion(version),
        targets: summarizeCdpTargets(targets),
      };
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      await sleep(options.intervalMs);
    }
  }
  throw new Error(`Electron CDP endpoint 未就绪: ${lastError}`);
}

async function findElectronCdpPage(browser, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const pages = browser.contexts().flatMap((context) => context.pages());
    for (const page of pages) {
      const snapshot = await page
        .evaluate(() => ({
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
        }))
        .catch(() => null);
      lastSnapshot = snapshot;
      if (
        snapshot?.electron &&
        snapshot.hasInvokeBridge &&
        snapshot.supportsAppServer
      ) {
        return page;
      }
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `未找到真实 Electron renderer CDP 页签: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

function buildElectronLaunchEnv(options, runtimeEnv, appServerEnv) {
  return {
    ...runtimeEnv.env,
    ...appServerEnv,
    APP_SERVER_BACKEND_MODE: "external",
    APP_SERVER_BACKEND_COMMAND: process.execPath,
    APP_SERVER_BACKEND_ARGS: JSON.stringify([
      runtimeEnv.backendPath,
      runtimeEnv.backendLedgerPath,
      runtimeEnv.cancelSignalPath,
    ]),
    APP_SERVER_BACKEND_TIMEOUT_MS: String(options.timeoutMs),
    CLAW_CHAT_FIXTURE_SCENARIO: "cancel",
    ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
    LIME_ALLOW_LIVE_PROVIDER_SMOKE: "0",
    LIME_REAL_API_TEST: "0",
    LIME_ELECTRON_E2E: "1",
    LIME_ELECTRON_BRAND_DEV_APP: "0",
    LIME_ELECTRON_CLEAR_RENDERER_CACHE: "0",
    LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
    LIME_ELECTRON_REMOTE_DEBUGGING_PORT: String(options.cdpPort),
    LIME_TRACE_EXPORT_OUTPUT_DIR: path.join(
      runtimeEnv.tempRoot,
      "trace-exports",
    ),
    LIME_SUPPORT_BUNDLE_OUTPUT_DIR: path.join(
      runtimeEnv.tempRoot,
      "support-bundles",
    ),
    ...(options.appUrl ? { VITE_DEV_SERVER_URL: options.appUrl } : {}),
  };
}

function bindElectronConsoleErrors(target, consoleErrors) {
  target.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(sanitizeText(message.text()));
    }
  });
}

async function launchElectronCdpGate(
  options,
  runtimeEnv,
  appServerEnv,
  consoleErrors,
  stage,
) {
  logStage(stage);
  const launchedApp = await electron.launch({
    executablePath: electronPath,
    args: [
      `--remote-debugging-port=${options.cdpPort}`,
      "--use-mock-keychain",
      ".",
    ],
    cwd: process.cwd(),
    env: buildElectronLaunchEnv(options, runtimeEnv, appServerEnv),
    timeout: options.timeoutMs,
  });
  bindElectronConsoleErrors(launchedApp, consoleErrors);

  const firstWindow = await launchedApp.firstWindow({
    timeout: options.timeoutMs,
  });
  bindElectronConsoleErrors(firstWindow, consoleErrors);
  firstWindow.setDefaultTimeout(options.timeoutMs);
  await firstWindow.setViewportSize({ width: 1440, height: 1000 });

  logStage(`${stage}:wait-cdp-endpoint`);
  const cdpEndpoint = await waitForCdpEndpoint(options);

  logStage(`${stage}:connect-over-cdp`);
  const connectedBrowser = await chromium.connectOverCDP(options.cdpUrl);
  const page = await findElectronCdpPage(connectedBrowser, options);
  page.setDefaultTimeout(options.timeoutMs);
  await page.setViewportSize({ width: 1440, height: 1000 });

  return {
    app: launchedApp,
    browser: connectedBrowser,
    page,
    cdpEndpoint,
    cdpPage: sanitizeJson({
      url: page.url(),
      title: await page.title().catch(() => ""),
    }),
  };
}

async function closeElectronCdpGate(app, browser) {
  const startedAt = Date.now();
  const result = {
    closed: false,
    durationMs: 0,
    browserCloseError: null,
    appCloseError: null,
  };
  try {
    await browser?.close();
  } catch (error) {
    result.browserCloseError = sanitizeText(error);
  }
  try {
    await app?.close();
  } catch (error) {
    result.appCloseError = sanitizeText(error);
  }
  result.durationMs = Date.now() - startedAt;
  result.closed = !result.browserCloseError && !result.appCloseError;
  return result;
}

async function readRendererTrace(page) {
  const raw = await page.evaluate(() =>
    window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
  );
  return readTraceMessages(raw);
}

function isClosedPageError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return (
    message.includes("Target page, context or browser has been closed") ||
    message.includes("Target closed") ||
    message.includes("Browser has been closed")
  );
}

function isTransientGuiSamplingError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return (
    isClosedPageError(error) ||
    message.includes("Execution context was destroyed") ||
    message.includes("Most likely because of a navigation") ||
    message.includes("Cannot find context with specified id")
  );
}

async function clearInvokeBuffersWithRetry(page, options) {
  const startedAt = Date.now();
  let lastError = null;
  const timeoutMs = Math.min(options.timeoutMs, 15_000);
  while (Date.now() - startedAt < timeoutMs) {
    try {
      await clearInvokeBuffers(page);
      return;
    } catch (error) {
      if (!isTransientGuiSamplingError(error)) {
        throw error;
      }
      lastError = error;
      await sleep(options.intervalMs);
    }
  }
  throw new Error(
    `清理 renderer trace buffer 超时: ${sanitizeText(lastError)}`,
  );
}

async function readRendererTraceFromPageRef(pageRef, browser, options) {
  if (pageRef.current?.isClosed?.()) {
    pageRef.current = await findElectronCdpPage(browser, options);
  }
  try {
    return await readRendererTrace(pageRef.current);
  } catch (error) {
    if (!browser || !isClosedPageError(error)) {
      throw error;
    }
    pageRef.current = await findElectronCdpPage(browser, options);
    return await readRendererTrace(pageRef.current);
  }
}

function decodeJsonRpcLines(lines) {
  if (!Array.isArray(lines)) {
    return [];
  }
  return lines
    .map((line) => {
      try {
        return JSON.parse(String(line));
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function summarizeTraceMessages(traceMessages) {
  const appServerEntries = traceMessages.filter(
    (entry) => entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  );
  const methodEntries = appServerEntries.flatMap((entry, index) =>
    decodeJsonRpcLines(entry?.args_preview?.request?.lines).map((message) => ({
      index,
      command: entry.command ?? null,
      transport: entry.transport ?? null,
      status: entry.status ?? null,
      method: message.method ?? null,
      id: message.id ?? null,
      sessionId:
        message.params?.sessionId ?? message.params?.session_id ?? null,
      threadId: message.params?.threadId ?? message.params?.thread_id ?? null,
      turnId: message.params?.turnId ?? message.params?.turn_id ?? null,
      eventName:
        message.params?.runtimeOptions?.eventName ??
        message.params?.runtime_options?.event_name ??
        null,
      promptLength:
        typeof message.params?.input?.text === "string"
          ? message.params.input.text.length
          : null,
    })),
  );
  return sanitizeJson({
    entryCount: traceMessages.length,
    appServerEntryCount: appServerEntries.length,
    methods: Array.from(
      new Set(methodEntries.map((entry) => entry.method).filter(Boolean)),
    ),
    electronIpcSuccessCount: methodEntries.filter(
      (entry) =>
        entry.transport === "electron-ipc" && entry.status === "success",
    ).length,
    methodEntries,
  });
}

function traceEntrySignature(entry) {
  return JSON.stringify({
    timestamp: entry?.timestamp ?? null,
    command: entry?.command ?? null,
    transport: entry?.transport ?? null,
    status: entry?.status ?? null,
    args_preview: entry?.args_preview ?? null,
  });
}

function createTraceCursor(traceMessages) {
  const lastEntry = traceMessages[traceMessages.length - 1] ?? null;
  return {
    length: traceMessages.length,
    lastTimestamp: lastEntry?.timestamp ?? null,
    lastSignature: lastEntry ? traceEntrySignature(lastEntry) : null,
  };
}

function traceMessagesAfterCursor(traceMessages, cursor) {
  if (typeof cursor === "number") {
    return traceMessages.slice(cursor);
  }

  const lastSignature =
    typeof cursor?.lastSignature === "string" ? cursor.lastSignature : null;
  if (lastSignature) {
    for (let index = traceMessages.length - 1; index >= 0; index -= 1) {
      if (traceEntrySignature(traceMessages[index]) === lastSignature) {
        return traceMessages.slice(index + 1);
      }
    }
  }

  const cursorTimestampMs = Date.parse(String(cursor?.lastTimestamp ?? ""));
  if (Number.isFinite(cursorTimestampMs)) {
    return traceMessages.filter((entry) => {
      const timestampMs = Date.parse(String(entry?.timestamp ?? ""));
      return Number.isFinite(timestampMs) && timestampMs > cursorTimestampMs;
    });
  }

  return traceMessages.slice(Number(cursor?.length ?? 0));
}

function traceMethodEntriesAfter(traceMessages, cursor, method) {
  const entries = traceMessagesAfterCursor(traceMessages, cursor).filter(
    (entry) => entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  );
  return entries.flatMap((entry) =>
    decodeJsonRpcLines(entry?.args_preview?.request?.lines)
      .filter((message) => message?.method === method)
      .map((message) => ({
        transport: entry.transport ?? null,
        status: entry.status ?? null,
        method: message.method,
        sessionId:
          message.params?.sessionId ?? message.params?.session_id ?? null,
        threadId: message.params?.threadId ?? message.params?.thread_id ?? null,
        turnId: message.params?.turnId ?? message.params?.turn_id ?? null,
        params: sanitizeJson(message.params ?? {}),
      })),
  );
}

function normalizeText(value) {
  return String(value ?? "");
}

function turnStartTraceTurnId(entry) {
  return entry?.turnId ?? entry?.turn_id ?? null;
}

function turnStartTraceText(entry) {
  return entry?.text ?? entry?.inputText ?? null;
}

function collectTurnStartTraceEvidence(inputSend) {
  const afterClick = inputSend?.afterClick ?? {};
  const candidates = [];
  if (afterClick.matchingTurnStartTrace) {
    candidates.push(afterClick.matchingTurnStartTrace);
  }
  if (Array.isArray(afterClick.appServerTurnStartTrace)) {
    candidates.push(...afterClick.appServerTurnStartTrace);
  }
  return candidates;
}

function summarizeTurnStartEvidence(summary, { sessionId, turnId, prompt }) {
  const guiTraceMatched =
    collectTurnStartTraceEvidence(summary.inputSend).find((entry) => {
      const entrySessionId = entry?.sessionId ?? entry?.session_id ?? null;
      const entryTurnId = turnStartTraceTurnId(entry);
      const entryText = turnStartTraceText(entry);
      const status = normalizeText(entry?.status).toLowerCase();
      const transport = normalizeText(entry?.transport).toLowerCase();
      return (
        entrySessionId === sessionId &&
        (!entryTurnId || entryTurnId === turnId) &&
        normalizeText(entryText).includes(prompt) &&
        status === "success" &&
        transport === "electron-ipc"
      );
    }) ?? null;

  const backendTurnStart = summary.backendTurnStart ?? {};
  const backendLedgerMatched =
    backendTurnStart.kind === "turnStart" &&
    backendTurnStart.sessionId === sessionId &&
    backendTurnStart.turnId === turnId &&
    normalizeText(backendTurnStart.inputText).includes(prompt);

  return sanitizeJson({
    guiTraceMatched: guiTraceMatched
      ? {
          sessionId: guiTraceMatched.sessionId ?? null,
          turnId: turnStartTraceTurnId(guiTraceMatched),
          status: guiTraceMatched.status ?? null,
          transport: guiTraceMatched.transport ?? null,
          timestamp: guiTraceMatched.timestamp ?? null,
        }
      : null,
    backendLedgerMatched,
    backendTurnStart: backendLedgerMatched
      ? {
          sessionId: backendTurnStart.sessionId ?? null,
          turnId: backendTurnStart.turnId ?? null,
          kind: backendTurnStart.kind ?? null,
          recordedAt: backendTurnStart.recordedAt ?? null,
        }
      : null,
    matched: Boolean(guiTraceMatched) && backendLedgerMatched,
  });
}

function hasSuccessfulTraceEvidence(evidence, method, { sessionId, turnId }) {
  const matched = evidence?.matched ?? {};
  return (
    matched.method === method &&
    matched.sessionId === sessionId &&
    (!turnId || !matched.turnId || matched.turnId === turnId) &&
    matched.transport === "electron-ipc" &&
    matched.status === "success"
  );
}

async function waitForTraceMethodAfter(
  pageRef,
  options,
  cursor,
  method,
  { sessionId = SESSION_ID, turnId = null, browser = null } = {},
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const traceMessages = await readRendererTraceFromPageRef(
      pageRef,
      browser,
      options,
    );
    const entries = traceMethodEntriesAfter(traceMessages, cursor, method);
    const matched = entries.find((entry) => {
      if (sessionId && entry.sessionId && entry.sessionId !== sessionId) {
        return false;
      }
      if (turnId && entry.turnId && entry.turnId !== turnId) {
        return false;
      }
      return entry.transport === "electron-ipc" && entry.status === "success";
    });
    lastSummary = sanitizeJson({
      cursor,
      method,
      traceCount: traceMessages.length,
      entries,
    });
    if (matched) {
      return {
        matched: sanitizeJson(matched),
        summary: lastSummary,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `未观察到 reload 后产品 trace method=${method}: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}

function summarizeReadModelRunningState(readModel, turnId, prompt) {
  const serialized = JSON.stringify(readModel || {});
  const queueState = summarizeReadModelQueueState(readModel);
  const turns = collectReadModelTurns(readModel).map((turn) => ({
    turnId: readModelTurnId(turn),
    status: readModelTurnStatus(turn),
  }));
  const matchedTurn = turns.find((turn) => turn.turnId === turnId) ?? null;
  const statusValues = [
    queueState.detailStatus,
    queueState.threadStatus,
    queueState.latestTurnStatus,
    readModelLatestTurnStatus(readModel),
    matchedTurn?.status,
  ].map((status) => String(status ?? "").toLowerCase());
  const hasRunningStatus = statusValues.some((status) =>
    ["running", "active", "streaming", "in_progress"].includes(status),
  );
  const hasTerminalStatus = statusValues.some((status) =>
    ["completed", "failed", "canceled", "cancelled"].includes(status),
  );
  return sanitizeJson({
    ...queueState,
    latestTurnStatus: readModelLatestTurnStatus(readModel),
    matchedTurn,
    turns,
    includesPrompt: serialized.includes(prompt),
    includesInitialOutput: serialized.includes("以下是今日国际新闻简要整理"),
    sameActiveTurn: queueState.activeTurnId === turnId,
    hasRunningStatus,
    hasTerminalStatus,
    running:
      (queueState.activeTurnId === turnId || hasRunningStatus) &&
      !hasTerminalStatus,
  });
}

async function waitForReadModelRunning(
  page,
  options,
  requestLog,
  turnId,
  { requireContent = true, sessionId = SESSION_ID, prompt = NEWS_PROMPT } = {},
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        sessionId,
        historyLimit: 100,
      },
      requestLog,
    );
    lastSummary = summarizeReadModelRunningState(read.result, turnId, prompt);
    if (
      lastSummary.running &&
      (!requireContent ||
        (lastSummary.includesPrompt && lastSummary.includesInitialOutput))
    ) {
      return {
        readModel: read.result,
        summary: lastSummary,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `read model 未保持同一 running turn session=${sessionId}: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}

function summarizeSessionList(result, turnId, sessionId = SESSION_ID) {
  const sessions = Array.isArray(result?.sessions)
    ? result.sessions
    : Array.isArray(result?.items)
      ? result.items
      : [];
  const matched = sessions.find(
    (session) =>
      session?.sessionId === sessionId ||
      session?.session_id === sessionId ||
      session?.id === sessionId,
  );
  const serialized = JSON.stringify(matched || {});
  const statusValues = [
    matched?.runtimeStatus,
    matched?.runtime_status,
    matched?.status,
    matched?.state,
    matched?.latestTurnStatus,
    matched?.latest_turn_status,
    matched?.runtimeSummary?.latestTurnStatus,
    matched?.runtime_summary?.latestTurnStatus,
  ].map((status) => String(status ?? "").toLowerCase());
  const running = statusValues.some((status) =>
    ["running", "active", "streaming", "in_progress"].includes(status),
  );
  const terminal = statusValues.some((status) =>
    ["completed", "failed", "canceled", "cancelled"].includes(status),
  );
  return sanitizeJson({
    count: sessions.length,
    matched: matched
      ? {
          sessionId:
            matched.sessionId ?? matched.session_id ?? matched.id ?? null,
          title: matched.title ?? null,
          runtimeStatus:
            matched.runtimeStatus ?? matched.runtime_status ?? null,
          status: matched.status ?? matched.state ?? null,
          latestTurnStatus:
            matched.latestTurnStatus ?? matched.latest_turn_status ?? null,
          activeTurnId:
            matched.activeTurnId ??
            matched.active_turn_id ??
            matched.threadRead?.activeTurnId ??
            matched.thread_read?.active_turn_id ??
            null,
          includesTurnId: turnId ? serialized.includes(turnId) : null,
        }
      : null,
    running:
      Boolean(matched) && (running || serialized.includes(turnId)) && !terminal,
    terminal,
  });
}

async function waitForSessionListRunning(
  page,
  options,
  requestLog,
  turnId,
  sessionId = SESSION_ID,
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const list = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_LIST,
      { includeArchived: true, limit: 20 },
      requestLog,
    );
    lastSummary = summarizeSessionList(list.result, turnId, sessionId);
    if (lastSummary.running) {
      return lastSummary;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `session list 未显示 running session=${sessionId}: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}

async function waitForSessionListNotRunning(
  page,
  options,
  requestLog,
  turnId,
  sessionId = SESSION_ID,
) {
  const startedAt = Date.now();
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const list = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_LIST,
      { includeArchived: true, limit: 20 },
      requestLog,
    );
    lastSummary = summarizeSessionList(list.result, turnId, sessionId);
    if (lastSummary.matched && !lastSummary.running) {
      return lastSummary;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `session list 取消后仍显示 running session=${sessionId}: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}

function buildRunningSessionSpec(overrides = {}) {
  return {
    sessionId: overrides.sessionId ?? SESSION_ID,
    threadId: overrides.threadId ?? THREAD_ID,
    title: overrides.title ?? SESSION_TITLE,
    prompt: overrides.prompt ?? NEWS_PROMPT,
  };
}

function buildMultiRunningSessionSpecs(primaryTurnId, secondaryTurnId) {
  return [
    {
      ...buildRunningSessionSpec(),
      turnId: primaryTurnId,
    },
    {
      ...buildRunningSessionSpec({
        sessionId: MULTI_RUNNING_SECONDARY_SESSION_ID,
        threadId: MULTI_RUNNING_SECONDARY_THREAD_ID,
        title: MULTI_RUNNING_SECONDARY_TITLE,
        prompt: MULTI_RUNNING_SECONDARY_PROMPT,
      }),
      turnId: secondaryTurnId,
    },
  ];
}

async function createFixtureSessionForSpec(page, workspace, requestLog, spec) {
  const { workspaceId, rootPath } = workspace;
  assert(rootPath, "workspace/default/ensure 未返回可用 rootPath");
  const session = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_SESSION_START,
    {
      sessionId: spec.sessionId,
      threadId: spec.threadId,
      appId: "desktop",
      workspaceId,
      workingDir: rootPath,
      businessObjectRef: {
        kind: "agent.session",
        id: `agent-session:${workspaceId}:${spec.sessionId}`,
        title: spec.title,
        metadata: {
          title: spec.title,
          workingDir: rootPath,
          working_dir: rootPath,
          executionStrategy: "react",
          runStartHooks: false,
          harness: {
            hiddenFromUserRecents: false,
            source: "smoke:reopen-running-turn-cdp-gate:multi-running",
          },
        },
      },
    },
    requestLog,
  );
  const update = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_THREAD_SETTINGS_UPDATE,
    {
      threadId: spec.threadId,
      modelProvider: FIXTURE_PROVIDER,
      model: FIXTURE_MODEL,
      approvalPolicy: "never",
      sandboxPolicy: "danger-full-access",
      toolPreferences: {
        searchMode: "auto",
      },
    },
    requestLog,
  );
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
    },
    { sessionId: spec.sessionId, workspaceId },
  );
  return {
    session: session.result,
    update: update.result,
  };
}

async function startRunningTurnForSpec(page, requestLog, spec) {
  const turnId = `turn_${spec.sessionId}_${Date.now()}`;
  const eventName = `agentSession/event/${spec.sessionId}`;
  const turnStart = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_SESSION_TURN_START,
    {
      sessionId: spec.sessionId,
      turnId,
      input: {
        text: spec.prompt,
      },
      runtimeOptions: {
        stream: true,
        eventName,
        runtimeRequest: {
          providerPreference: FIXTURE_PROVIDER,
          modelPreference: FIXTURE_MODEL,
          metadata: {
            harness: {
              source: "smoke:reopen-running-turn-cdp-gate:multi-running",
            },
          },
        },
      },
      queueIfBusy: false,
      skipPreSubmitResume: true,
    },
    requestLog,
  );
  return {
    turnId,
    turnStart: turnStart.result,
  };
}

async function createSecondaryRunningSession(
  page,
  options,
  workspace,
  requestLog,
  runtimeEnv,
) {
  const spec = buildRunningSessionSpec({
    sessionId: MULTI_RUNNING_SECONDARY_SESSION_ID,
    threadId: MULTI_RUNNING_SECONDARY_THREAD_ID,
    title: MULTI_RUNNING_SECONDARY_TITLE,
    prompt: MULTI_RUNNING_SECONDARY_PROMPT,
  });
  const session = await createFixtureSessionForSpec(
    page,
    workspace,
    requestLog,
    spec,
  );
  const started = await startRunningTurnForSpec(page, requestLog, spec);
  const backendTurn = await waitForBackendLedgerTurnStart(
    runtimeEnv.backendLedgerPath,
    spec.prompt,
    options,
  );
  const turnId = String(backendTurn.entry?.turnId || started.turnId).trim();
  assert(turnId, "secondary external backend ledger 未记录 turnId");
  return sanitizeJson({
    spec,
    session,
    turnStart: started.turnStart,
    backendTurnStart: backendTurn.entry,
    turnId,
  });
}

async function sampleGuiSidebarRunningSessions(page, specs) {
  return await page.evaluate((sessionSpecs) => {
    const rows = Array.from(
      document.querySelectorAll(
        '[data-testid="app-sidebar-conversation-open"]',
      ),
    );
    return sessionSpecs.map((spec) => {
      const matchingRows = rows.filter((row) => {
        const label = [
          row.getAttribute("title") || "",
          row.getAttribute("aria-label") || "",
          row.textContent || "",
        ].join("\n");
        return label.includes(spec.title);
      });
      const row =
        matchingRows.find((candidate) =>
          Boolean(
            candidate
              .querySelector(
                '[data-testid="app-sidebar-conversation-runtime-status"]',
              )
              ?.getAttribute("data-status"),
          ),
        ) ??
        matchingRows[0] ??
        null;
      const statusNode = row?.querySelector(
        '[data-testid="app-sidebar-conversation-runtime-status"]',
      );
      return {
        sessionId: spec.sessionId,
        title: spec.title,
        titleFound: Boolean(row),
        status: statusNode?.getAttribute("data-status") || "",
        statusText: statusNode?.textContent || "",
        rowText: row?.textContent || "",
      };
    });
  }, specs);
}

async function waitForGuiSidebarSessionsRunning(page, options, specs, label) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      lastSnapshot = sanitizeJson(
        await sampleGuiSidebarRunningSessions(page, specs),
      );
    } catch (error) {
      if (!isTransientGuiSamplingError(error)) {
        throw error;
      }
      lastSnapshot = {
        transientError: sanitizeText(error),
        url: typeof page.url === "function" ? page.url() : "",
      };
      await sleep(options.intervalMs);
      continue;
    }
    if (
      Array.isArray(lastSnapshot) &&
      lastSnapshot.length === specs.length &&
      lastSnapshot.every(
        (entry) => entry.titleFound && entry.status === "running",
      )
    ) {
      return {
        allRunning: true,
        sessions: lastSnapshot,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `GUI 侧栏多 running 状态不一致 (${label}): ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function cancelSecondaryRunningSession(
  page,
  options,
  requestLog,
  runtimeEnv,
  secondary,
) {
  if (!secondary?.turnId) {
    return sanitizeJson({
      skipped: true,
      reason: "secondary running session was not created",
    });
  }
  const turnCancel = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_SESSION_TURN_CANCEL,
    {
      sessionId: MULTI_RUNNING_SECONDARY_SESSION_ID,
      turnId: secondary.turnId,
    },
    requestLog,
  );
  const backendTurnCancel = await waitForBackendLedgerEntry(
    runtimeEnv.backendLedgerPath,
    (entry) =>
      entry.kind === "turnCancel" &&
      entry.sessionId === MULTI_RUNNING_SECONDARY_SESSION_ID &&
      entry.turnId === secondary.turnId,
    options,
  );
  const readModelCanceled = await waitForSessionReadCanceled(
    page,
    options,
    requestLog,
    {
      sessionId: MULTI_RUNNING_SECONDARY_SESSION_ID,
      prompt: MULTI_RUNNING_SECONDARY_PROMPT,
      partialText: "以下是今日国际新闻简要整理",
    },
  );
  const sessionListNotRunning = await waitForSessionListNotRunning(
    page,
    options,
    requestLog,
    secondary.turnId,
    MULTI_RUNNING_SECONDARY_SESSION_ID,
  );
  return sanitizeJson({
    turnCancel: turnCancel.result,
    backendTurnCancel: backendTurnCancel.entry,
    readModelCanceled: summarizeReadModelRunningState(
      readModelCanceled,
      secondary.turnId,
      MULTI_RUNNING_SECONDARY_PROMPT,
    ),
    sessionListNotRunning,
  });
}

async function sampleGuiHomeBackgroundRecoveryState(page, turnId) {
  return await page.evaluate(
    ({ prompt, title, sessionId, turnId }) => {
      const isVisible = (node) => {
        if (!(node instanceof HTMLElement)) {
          return false;
        }
        const rect = node.getBoundingClientRect();
        const style = window.getComputedStyle(node);
        return (
          rect.width > 8 &&
          rect.height > 8 &&
          style.visibility !== "hidden" &&
          style.display !== "none"
        );
      };
      const main = document.querySelector("main") ?? document.body;
      const mainText = main?.innerText || "";
      const homeStartSurface = document.querySelector(
        '[data-testid="home-start-surface"]',
      );
      const homeRecoveryCard = document.querySelector(
        '[data-testid="home-unfinished-session-card"]',
      );
      const messageLists = Array.from(
        document.querySelectorAll(
          '[data-testid="message-list"], [data-testid="message-list-frame"]',
        ),
      );
      const messageTurnGroups = Array.from(
        document.querySelectorAll('[data-testid="message-turn-group"]'),
      );
      const textareas = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).filter((node) => node instanceof HTMLTextAreaElement);
      const textareaSessionIds = textareas.map(
        (textarea) => textarea.dataset.sessionId || null,
      );
      const sidebarRows = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      );
      const matchingSidebarRows = sidebarRows.filter((row) => {
        const label = [
          row.getAttribute("title") || "",
          row.getAttribute("aria-label") || "",
          row.textContent || "",
        ].join("\n");
        return label.includes(title);
      });
      const sidebarRow =
        matchingSidebarRows.find((row) =>
          Boolean(
            row
              .querySelector(
                '[data-testid="app-sidebar-conversation-runtime-status"]',
              )
              ?.getAttribute("data-status"),
          ),
        ) ??
        matchingSidebarRows.find((row) => {
          const label = [
            row.getAttribute("title") || "",
            row.getAttribute("aria-label") || "",
            row.textContent || "",
          ].join("\n");
          return label.includes(title);
        }) ??
        null;
      const sidebarStatusNode = sidebarRow?.querySelector(
        '[data-testid="app-sidebar-conversation-runtime-status"]',
      );
      const sidebarStatus =
        sidebarStatusNode?.getAttribute("data-status") || "";
      const mainPromptVisible = mainText.includes(prompt);
      const mainAssistantOutputVisible =
        mainText.includes("以下是今日国际新闻简要整理");
      const activeSessionTextareaVisible = textareas.some(
        (textarea) =>
          textarea.dataset.sessionId === sessionId && isVisible(textarea),
      );
      const activeSessionMessageVisible =
        messageLists.some((list) => isVisible(list)) &&
        (mainPromptVisible ||
          mainAssistantOutputVisible ||
          messageTurnGroups.some((group) =>
            (group.textContent || "").includes(turnId),
          ));
      return {
        url: window.location.href,
        homeStartVisible: isVisible(homeStartSurface),
        hasHomeStartSurface: Boolean(homeStartSurface),
        homeRecoveryCardVisible: isVisible(homeRecoveryCard),
        homeRecoveryCardStatus:
          homeRecoveryCard?.getAttribute("data-status") || "",
        homeRecoveryCardText: homeRecoveryCard?.textContent || "",
        homeRecoveryCardTitle: homeRecoveryCard?.getAttribute("title") || "",
        homeRecoveryCardTitleFound: Boolean(
          homeRecoveryCard &&
          (homeRecoveryCard.textContent || "").includes(title),
        ),
        mainPromptVisible,
        mainAssistantOutputVisible,
        messageListCount: messageLists.length,
        messageTurnGroupCount: messageTurnGroups.length,
        textareaCount: textareas.length,
        textareaSessionIds,
        activeSessionTextareaVisible,
        activeSessionMessageVisible,
        activeDetailBoundToSession:
          activeSessionTextareaVisible || activeSessionMessageVisible,
        sidebarTitleFound: Boolean(sidebarRow),
        sidebarStatus,
        sidebarStatusText: sidebarStatusNode?.textContent || "",
        sidebarRowText: sidebarRow?.textContent || "",
      };
    },
    {
      prompt: NEWS_PROMPT,
      title: SESSION_TITLE,
      sessionId: SESSION_ID,
      turnId,
    },
  );
}

async function waitForGuiHomeBackgroundRecovery(page, options, turnId, label) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      lastSnapshot = sanitizeJson(
        await sampleGuiHomeBackgroundRecoveryState(page, turnId),
      );
    } catch (error) {
      if (!isTransientGuiSamplingError(error)) {
        throw error;
      }
      lastSnapshot = {
        transientError: sanitizeText(error),
        url: typeof page.url === "function" ? page.url() : "",
      };
      await sleep(options.intervalMs);
      continue;
    }
    if (
      lastSnapshot.homeStartVisible &&
      lastSnapshot.homeRecoveryCardVisible &&
      lastSnapshot.homeRecoveryCardStatus === "running" &&
      lastSnapshot.homeRecoveryCardTitleFound &&
      !lastSnapshot.activeDetailBoundToSession &&
      lastSnapshot.sidebarTitleFound &&
      lastSnapshot.sidebarStatus === "running"
    ) {
      return lastSnapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `GUI 首页后台恢复状态不一致 (${label}): ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function openFixtureSessionFromHomeRecoveryCard(page, options, turnId) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const click = await page
      .evaluate(() => {
        const card = document.querySelector(
          '[data-testid="home-unfinished-session-card"]',
        );
        if (!(card instanceof HTMLElement)) {
          return { clicked: false, reason: "card-not-found" };
        }
        if (card.getAttribute("data-status") !== "running") {
          return {
            clicked: false,
            reason: "card-not-running",
            status: card.getAttribute("data-status") || "",
            text: card.textContent || "",
          };
        }
        card.click();
        return {
          clicked: true,
          status: card.getAttribute("data-status") || "",
          title: card.getAttribute("title") || "",
          text: card.textContent || "",
        };
      })
      .catch((error) => ({
        clicked: false,
        reason: error instanceof Error ? error.message : String(error),
      }));
    lastSnapshot = sanitizeJson({ click });
    if (click.clicked === true) {
      const running = await waitForGuiRunningConsistency(
        page,
        options,
        turnId,
        "opened-from-home-recovery-card",
      );
      return sanitizeJson({ click, running });
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `首页恢复卡未能打开会话详情: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

async function navigateGuiToNewTaskHome(page, options, turnId) {
  const startedAt = Date.now();
  let clicked = false;
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    if (!clicked) {
      const click = await page
        .evaluate(() => {
          const button = document.querySelector(
            [
              '[data-testid="app-sidebar-home-button"]',
              '[aria-label="返回 Lime 首页"]',
              '[aria-label="Back to Lime home"]',
            ].join(","),
          );
          if (!(button instanceof HTMLElement)) {
            return { clicked: false, reason: "button-not-found" };
          }
          button.click();
          return {
            clicked: true,
            title: button.getAttribute("title") || "",
            aria: button.getAttribute("aria-label") || "",
            text: button.textContent || "",
          };
        })
        .catch((error) => ({
          clicked: false,
          reason: error instanceof Error ? error.message : String(error),
        }));
      clicked = click.clicked === true;
      lastSnapshot = sanitizeJson({ click });
    }

    const homeSnapshot = sanitizeJson(
      await sampleGuiHomeBackgroundRecoveryState(page, turnId),
    );
    lastSnapshot = {
      ...(lastSnapshot && typeof lastSnapshot === "object" ? lastSnapshot : {}),
      home: homeSnapshot,
    };
    if (
      homeSnapshot.homeStartVisible &&
      !homeSnapshot.activeDetailBoundToSession
    ) {
      return lastSnapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `GUI 未回到新任务首页: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

async function sampleGuiRunningState(page, turnId) {
  return await page.evaluate(
    ({ prompt, title, turnId }) => {
      const bodyText = document.body?.innerText || "";
      const main = document.querySelector("main") ?? document.body;
      const mainText = main?.innerText || bodyText;
      const turnGroups = Array.from(
        document.querySelectorAll('[data-testid="message-turn-group"]'),
      );
      const promptTurnGroup =
        [...turnGroups]
          .reverse()
          .find((group) => (group.innerText || "").includes(prompt)) ?? null;
      const scope = promptTurnGroup ?? main ?? document;
      const scopedText = promptTurnGroup?.innerText || mainText;
      const statusNodes = Array.from(
        scope.querySelectorAll(
          [
            '[data-testid="assistant-first-token-runtime-status"]',
            '[data-testid="message-runtime-status-pill"]',
            '[data-testid="inputbar-runtime-status-line"]',
          ].join(","),
        ),
      ).map((node) => ({
        testId: node.getAttribute("data-testid") || "",
        status: node.getAttribute("data-status") || "",
        text: node.textContent || "",
      }));
      const inputbar = document.querySelector(
        '[data-testid="inputbar-core-container"]',
      );
      const inputbarText = inputbar?.textContent || "";
      const inputbarButtons = Array.from(
        inputbar?.querySelectorAll("button") || [],
      ).map((button) => ({
        title: button.getAttribute("title") || "",
        aria: button.getAttribute("aria-label") || "",
        text: button.textContent || "",
        disabled: button.disabled,
      }));
      const inputbarButtonLabels = inputbarButtons.map((button) =>
        [button.title, button.aria, button.text].join("\n"),
      );
      const inputbarHasStopButton = inputbarButtons.some((button) => {
        const label = [button.title, button.aria, button.text].join("\n");
        return (
          !button.disabled &&
          (label.includes("停止") ||
            label.includes("终止") ||
            /\bStop\b/i.test(label))
        );
      });
      const inputbarHasRunningText =
        inputbarText.includes("正在输出") ||
        inputbarText.includes("正在生成") ||
        inputbarButtonLabels.some(
          (label) => label.includes("正在输出") || label.includes("正在生成"),
        );
      const sidebarRows = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      );
      const matchingSidebarRows = sidebarRows.filter((row) => {
        const label = [
          row.getAttribute("title") || "",
          row.getAttribute("aria-label") || "",
          row.textContent || "",
        ].join("\n");
        return label.includes(title);
      });
      const sidebarRow =
        matchingSidebarRows.find((row) =>
          Boolean(
            row
              .querySelector(
                '[data-testid="app-sidebar-conversation-runtime-status"]',
              )
              ?.getAttribute("data-status"),
          ),
        ) ??
        matchingSidebarRows.find((row) => {
          const label = [
            row.getAttribute("title") || "",
            row.getAttribute("aria-label") || "",
            row.textContent || "",
          ].join("\n");
          return label.includes(title);
        }) ??
        null;
      const sidebarStatusNode = sidebarRow?.querySelector(
        '[data-testid="app-sidebar-conversation-runtime-status"]',
      );
      const sidebarStatus =
        sidebarStatusNode?.getAttribute("data-status") || "";
      const sidebarStatusText = sidebarStatusNode?.textContent || "";
      const globalStopButtonVisible = Array.from(
        document.querySelectorAll("button"),
      ).some((button) => {
        const label = [
          button.getAttribute("title") || "",
          button.getAttribute("aria-label") || "",
          button.textContent || "",
        ].join("\n");
        return (
          !button.disabled &&
          (label.includes("停止") ||
            label.includes("终止") ||
            /\bStop\b/i.test(label))
        );
      });
      const textarea = document.querySelector(
        'textarea[name="agent-chat-message"]',
      );
      return {
        url: window.location.href,
        hasPrompt: scopedText.includes(prompt),
        hasAssistantOutput: scopedText.includes("以下是今日国际新闻简要整理"),
        hasTurnIdInDom: turnId ? scopedText.includes(turnId) : null,
        statusNodes,
        mainRunningStatus:
          scopedText.includes("正在输出") ||
          scopedText.includes("正在生成") ||
          statusNodes.some(
            (entry) =>
              entry.status === "running" ||
              entry.text.includes("正在输出") ||
              entry.text.includes("正在生成"),
          ),
        inputbarHasStopButton,
        inputbarHasRunningText,
        inputbarButtonLabels,
        globalStopButtonVisible,
        sidebarTitleFound: Boolean(sidebarRow),
        sidebarStatus,
        sidebarStatusText,
        textareaSessionId:
          textarea instanceof HTMLTextAreaElement
            ? textarea.dataset.sessionId || null
            : null,
        textareaDisabled:
          textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
        scopedText,
      };
    },
    { prompt: NEWS_PROMPT, title: SESSION_TITLE, turnId },
  );
}

async function waitForGuiRunningConsistency(page, options, turnId, label) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      lastSnapshot = sanitizeJson(await sampleGuiRunningState(page, turnId));
    } catch (error) {
      if (!isTransientGuiSamplingError(error)) {
        throw error;
      }
      lastSnapshot = {
        transientError: sanitizeText(error),
        url: typeof page.url === "function" ? page.url() : "",
      };
      await sleep(options.intervalMs);
      continue;
    }
    if (
      lastSnapshot.hasPrompt &&
      lastSnapshot.hasAssistantOutput &&
      (lastSnapshot.mainRunningStatus ||
        lastSnapshot.globalStopButtonVisible) &&
      lastSnapshot.inputbarHasStopButton &&
      lastSnapshot.sidebarTitleFound &&
      lastSnapshot.sidebarStatus === "running"
    ) {
      return lastSnapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `GUI running 状态不一致 (${label}): ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForGuiIdleConsistency(
  page,
  options,
  turnId,
  { requirePrompt = true } = {},
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      lastSnapshot = sanitizeJson(await sampleGuiRunningState(page, turnId));
    } catch (error) {
      if (!isTransientGuiSamplingError(error)) {
        throw error;
      }
      lastSnapshot = {
        transientError: sanitizeText(error),
        url: typeof page.url === "function" ? page.url() : "",
      };
      await sleep(options.intervalMs);
      continue;
    }
    if (
      (!requirePrompt || lastSnapshot.hasPrompt) &&
      !lastSnapshot.inputbarHasStopButton &&
      !lastSnapshot.globalStopButtonVisible &&
      lastSnapshot.sidebarTitleFound &&
      lastSnapshot.sidebarStatus !== "running"
    ) {
      return lastSnapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `GUI 取消后仍存在 running 状态: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForCanceledEventAfterReload(page, options, turnId) {
  const startedAt = Date.now();
  let events = [];
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const drained = await drainAppServerEventsFromPage(page, 100);
    events = [...events, ...collectRuntimeEvents(drained.messages)];
    lastSummary = summarizeRuntimeEvents(events, turnId);
    if (lastSummary.terminalTypes?.includes("turn.canceled")) {
      return {
        events: sanitizeJson(events.filter((event) => event.turnId === turnId)),
        summary: lastSummary,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `reload 后未通过 agentSession/event 观察到同一 turn 取消终态: ${JSON.stringify(
      sanitizeJson(lastSummary),
    )}`,
  );
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  ensureElectronFixtureBuild({
    appUrl: options.appUrl,
    logPrefix: LOG_PREFIX,
    rootDir: process.cwd(),
  });
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const screenshotBeforeReloadPath = path.join(
    options.evidenceDir,
    `${options.prefix}-before-${options.reopenMode}.png`,
  );
  const screenshotAfterReloadPath = path.join(
    options.evidenceDir,
    `${options.prefix}-after-${options.reopenMode}.png`,
  );
  const screenshotHomeBeforeReopenPath = path.join(
    options.evidenceDir,
    `${options.prefix}-home-before-${options.reopenMode}.png`,
  );
  const screenshotHomeAfterReopenPath = path.join(
    options.evidenceDir,
    `${options.prefix}-home-after-${options.reopenMode}.png`,
  );
  const screenshotAfterCancelPath = path.join(
    options.evidenceDir,
    `${options.prefix}-after-cancel.png`,
  );
  const backendLedgerEvidencePath = path.join(
    options.evidenceDir,
    `${options.prefix}-backend-ledger.json`,
  );
  const tracePath = path.join(
    options.evidenceDir,
    `${options.prefix}-trace-summary.json`,
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
  const requestLog = [];
  const consoleErrors = [];
  const summary = {
    ok: false,
    schemaVersion: "reopen-running-turn-cdp-gate.v1",
    scenarioId: "reopen-running-turn-cdp-gate",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      options.reopenMode === "restart"
        ? `真实 Electron CDP + preload IPC + app_server_handle_json_lines + App Server JSON-RPC + external controlled fixture；证明 Electron/App Server 重启后同一 running turnId 的 read model 与 GUI 状态可恢复一致；presentation=${options.presentationMode}；不要求 thread/resume，不证明后台 backend 子进程跨重启继续存活或 live Provider。`
        : `真实 Electron CDP + preload IPC + app_server_handle_json_lines + App Server JSON-RPC + external controlled fixture；证明 reload 后同一 running turnId 由产品恢复逻辑续接；presentation=${options.presentationMode}；不证明 live Provider。`,
    completedGateB: false,
    backendMode: "external",
    presentationMode: options.presentationMode,
    reopenMode: options.reopenMode,
    multiRunningSessions: options.multiRunningSessions,
    appUrl: options.appUrl || null,
    cdpUrl: options.cdpUrl,
    cdpPort: options.cdpPort,
    sessionId: SESSION_ID,
    threadId: THREAD_ID,
    prompt: NEWS_PROMPT,
    checkedAt: new Date().toISOString(),
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    electronUserDataDir: options.keepTemp
      ? runtimeEnv.electronUserDataDir
      : null,
    backendPath: options.keepTemp ? runtimeEnv.backendPath : null,
    backendLedgerPath: options.keepTemp ? runtimeEnv.backendLedgerPath : null,
    backendLedger: backendLedgerEvidencePath,
    traceSummaryPath: tracePath,
    consoleErrors,
    forbiddenFallbacks: [
      "APP_SERVER_BACKEND_MODE=mock",
      "mockPriorityCommands",
      "defaultMocks",
      "invokeMockOnly",
      "legacy agent_runtime_* production truth",
    ],
    assertions: {},
  };

  let app = null;
  let browser = null;
  try {
    if (options.appUrl) {
      logStage("wait-app-url");
      summary.rendererDevServer = sanitizeJson(
        await waitForAppUrlReady(options),
      );
    }

    const initialLaunch = await launchElectronCdpGate(
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      "launch-electron",
    );
    app = initialLaunch.app;
    browser = initialLaunch.browser;
    let page = initialLaunch.page;
    const pageRef = { current: page };
    summary.cdpEndpoint = initialLaunch.cdpEndpoint;
    summary.cdpPage = initialLaunch.cdpPage;

    logStage("wait-renderer");
    summary.rendererSnapshot = sanitizeJson(
      await waitForRendererReady(page, options),
    );
    await clearInvokeBuffersWithRetry(page, options);

    logStage("initialize-app-server");
    summary.initialize = sanitizeJson(
      await initializeAppServer(page, requestLog),
    );

    logStage("ensure-default-workspace");
    const workspace = await ensureDefaultWorkspace(page, requestLog);
    summary.workspace = sanitizeJson(workspace);

    logStage("bind-gui-workspace-model");
    summary.guiWorkspaceBinding = sanitizeJson(
      await bindGuiWorkspaceAndModelPreferences(page, workspace.workspaceId, {
        provider: FIXTURE_PROVIDER,
        model: FIXTURE_MODEL,
      }),
    );

    logStage("create-fixture-session");
    summary.sessionCreation = sanitizeJson(
      await createFixtureSession(page, workspace, requestLog, {
        provider: FIXTURE_PROVIDER,
        model: FIXTURE_MODEL,
      }),
    );
    if (options.multiRunningSessions) {
      logStage("create-secondary-running-session");
      summary.multiRunningSecondary = sanitizeJson(
        await createSecondaryRunningSession(
          page,
          options,
          workspace,
          requestLog,
          runtimeEnv,
        ),
      );
      summary.multiRunningSecondaryReadModelBeforePrimary = sanitizeJson(
        await waitForReadModelRunning(
          page,
          options,
          requestLog,
          summary.multiRunningSecondary.turnId,
          {
            sessionId: MULTI_RUNNING_SECONDARY_SESSION_ID,
            prompt: MULTI_RUNNING_SECONDARY_PROMPT,
          },
        ),
      );
      summary.multiRunningSecondarySessionListBeforePrimary = sanitizeJson(
        await waitForSessionListRunning(
          page,
          options,
          requestLog,
          summary.multiRunningSecondary.turnId,
          MULTI_RUNNING_SECONDARY_SESSION_ID,
        ),
      );
    } else {
      summary.multiRunningSecondary = sanitizeJson({
        skipped: true,
        reason: "--multi-running-sessions not enabled",
      });
    }

    logStage("navigate-gui-workspace");
    summary.guiWorkspaceNavigation = sanitizeJson(
      await navigateGuiToWorkspaceScopedAgent(
        page,
        options,
        workspace.workspaceId,
      ),
    );

    logStage("open-fixture-session");
    summary.guiSessionVisible = sanitizeJson(
      await waitForGuiSessionVisible(page, options),
    );
    summary.guiSessionOpened = sanitizeJson(
      await openFixtureSessionFromSidebar(page, options, requestLog),
    );

    logStage("send-running-turn");
    summary.inputSend = sanitizeJson(
      await sendPromptFromGui(page, options, NEWS_PROMPT, {
        expectedSessionId: SESSION_ID,
        requireTurnStart: true,
      }),
    );
    const backendTurn = await waitForBackendLedgerTurnStart(
      runtimeEnv.backendLedgerPath,
      NEWS_PROMPT,
      options,
    );
    const turnId = String(backendTurn.entry?.turnId || "").trim();
    assert(turnId, "external backend ledger 未记录 turnId");
    summary.turnId = turnId;
    summary.backendTurnStart = sanitizeJson(backendTurn.entry);
    const multiRunningSpecs = options.multiRunningSessions
      ? buildMultiRunningSessionSpecs(
          turnId,
          summary.multiRunningSecondary?.turnId,
        )
      : [];

    logStage("assert-running-before-reload");
    summary.readModelRunningBeforeReload = sanitizeJson(
      await waitForReadModelRunning(page, options, requestLog, turnId),
    );
    summary.sessionListRunningBeforeReload = sanitizeJson(
      await waitForSessionListRunning(page, options, requestLog, turnId),
    );
    summary.guiRunningBeforeReload = sanitizeJson(
      await waitForGuiRunningConsistency(
        page,
        options,
        turnId,
        "before-reload",
      ),
    );
    if (options.multiRunningSessions) {
      summary.multiRunningSidebarBeforeReopen = sanitizeJson(
        await waitForGuiSidebarSessionsRunning(
          page,
          options,
          multiRunningSpecs,
          "before-reopen",
        ),
      );
      summary.multiRunningSecondaryReadModelBeforeReopen = sanitizeJson(
        await waitForReadModelRunning(
          page,
          options,
          requestLog,
          summary.multiRunningSecondary.turnId,
          {
            sessionId: MULTI_RUNNING_SECONDARY_SESSION_ID,
            prompt: MULTI_RUNNING_SECONDARY_PROMPT,
          },
        ),
      );
      summary.multiRunningSecondarySessionListBeforeReopen = sanitizeJson(
        await waitForSessionListRunning(
          page,
          options,
          requestLog,
          summary.multiRunningSecondary.turnId,
          MULTI_RUNNING_SECONDARY_SESSION_ID,
        ),
      );
    } else {
      summary.multiRunningSidebarBeforeReopen = sanitizeJson({
        skipped: true,
        reason: "--multi-running-sessions not enabled",
      });
    }
    await page.screenshot({
      path: screenshotBeforeReloadPath,
      fullPage: true,
      timeout: 15_000,
    });
    summary.screenshotBeforeReload = screenshotBeforeReloadPath;

    if (options.presentationMode === "background") {
      logStage("navigate-home-before-reopen");
      summary.homeBeforeReopenNavigation = sanitizeJson(
        await navigateGuiToNewTaskHome(page, options, turnId),
      );
      summary.homeBackgroundBeforeReopen = sanitizeJson(
        await waitForGuiHomeBackgroundRecovery(
          page,
          options,
          turnId,
          "before-reopen",
        ),
      );
      if (options.multiRunningSessions) {
        summary.multiRunningHomeSidebarBeforeReopen = sanitizeJson(
          await waitForGuiSidebarSessionsRunning(
            page,
            options,
            multiRunningSpecs,
            "home-before-reopen",
          ),
        );
      }
      await page.screenshot({
        path: screenshotHomeBeforeReopenPath,
        fullPage: true,
        timeout: 15_000,
      });
      summary.screenshotHomeBeforeReopen = screenshotHomeBeforeReopenPath;
    } else {
      summary.homeBeforeReopenNavigation = sanitizeJson({
        skipped: true,
        reason: "foreground presentation keeps the session detail open",
      });
      summary.homeBackgroundBeforeReopen = sanitizeJson({
        skipped: true,
        reason: "foreground presentation keeps the session detail open",
      });
    }

    const traceCursorBeforeReload = createTraceCursor(
      await readRendererTraceFromPageRef(pageRef, browser, options),
    );
    page = pageRef.current;
    summary.traceCursorBeforeReload = traceCursorBeforeReload;

    if (options.reopenMode === "restart") {
      logStage("restart-electron:close");
      summary.restartClose = sanitizeJson(
        await closeElectronCdpGate(app, browser),
      );
      app = null;
      browser = null;
      pageRef.current = null;
      await sleep(500);

      const restartedLaunch = await launchElectronCdpGate(
        options,
        runtimeEnv,
        appServerEnv,
        consoleErrors,
        "restart-electron:launch",
      );
      app = restartedLaunch.app;
      browser = restartedLaunch.browser;
      page = restartedLaunch.page;
      pageRef.current = page;
      summary.cdpEndpointAfterRestart = restartedLaunch.cdpEndpoint;
      summary.cdpPageAfterRestart = restartedLaunch.cdpPage;
      summary.restart = sanitizeJson({
        relaunched: true,
        close: summary.restartClose,
        cdpPage: restartedLaunch.cdpPage,
      });
    } else {
      logStage("reload-renderer");
      summary.reload = sanitizeJson(
        await reloadRendererDocument(page, options),
      );
    }

    summary.rendererSnapshotAfterReopen = sanitizeJson(
      await waitForRendererReady(page, options),
    );
    summary.rendererSnapshotAfterReload = summary.rendererSnapshotAfterReopen;
    summary.guiSessionVisibleAfterReopen = sanitizeJson(
      await waitForGuiSessionVisible(page, options),
    );
    summary.guiSessionVisibleAfterReload = summary.guiSessionVisibleAfterReopen;
    if (options.presentationMode === "background") {
      logStage(`assert-home-background-after-${options.reopenMode}`);
      summary.homeBackgroundAfterReopen = sanitizeJson(
        await waitForGuiHomeBackgroundRecovery(
          page,
          options,
          turnId,
          `after-${options.reopenMode}`,
        ),
      );
      if (options.multiRunningSessions) {
        summary.multiRunningHomeSidebarAfterReopen = sanitizeJson(
          await waitForGuiSidebarSessionsRunning(
            page,
            options,
            multiRunningSpecs,
            `home-after-${options.reopenMode}`,
          ),
        );
      }
      summary.homeBackgroundAfterReload = summary.homeBackgroundAfterReopen;
      await page.screenshot({
        path: screenshotHomeAfterReopenPath,
        fullPage: true,
        timeout: 15_000,
      });
      summary.screenshotHomeAfterReopen = screenshotHomeAfterReopenPath;
      summary.screenshotHomeAfterReload = screenshotHomeAfterReopenPath;
    } else {
      summary.homeBackgroundAfterReopen = sanitizeJson({
        skipped: true,
        reason: "foreground presentation opens the session detail after reopen",
      });
      summary.homeBackgroundAfterReload = summary.homeBackgroundAfterReopen;
    }
    summary.guiSessionOpenedAfterReopen = sanitizeJson(
      options.presentationMode === "background"
        ? await openFixtureSessionFromHomeRecoveryCard(page, options, turnId)
        : await openFixtureSessionFromSidebar(page, options, requestLog),
    );
    summary.guiSessionOpenedAfterReload = summary.guiSessionOpenedAfterReopen;

    logStage(`assert-product-resume-after-${options.reopenMode}`);
    if (options.reopenMode === "restart") {
      summary.threadResumeTraceAfterReopen = sanitizeJson({
        skipped: true,
        reason:
          "restart 模式只声明 cold-start 后 read model / GUI running 状态恢复；external backend 子进程不作为跨重启存活声明，thread/resume 不是 restart Gate B 必需断言。",
      });
    } else {
      summary.threadResumeTraceAfterReopen = sanitizeJson(
        await waitForTraceMethodAfter(
          pageRef,
          options,
          traceCursorBeforeReload,
          APP_SERVER_METHOD_SESSION_THREAD_RESUME,
          { sessionId: SESSION_ID, turnId, browser },
        ),
      );
    }
    summary.threadResumeTraceAfterReload = summary.threadResumeTraceAfterReopen;
    page = pageRef.current;
    summary.readModelRunningAfterReopen = sanitizeJson(
      await waitForReadModelRunning(page, options, requestLog, turnId, {
        requireContent: options.reopenMode !== "restart",
      }),
    );
    summary.readModelRunningAfterReload = summary.readModelRunningAfterReopen;
    summary.sessionListRunningAfterReopen = sanitizeJson(
      await waitForSessionListRunning(page, options, requestLog, turnId),
    );
    summary.sessionListRunningAfterReload =
      summary.sessionListRunningAfterReopen;
    summary.guiRunningAfterReopen = sanitizeJson(
      await waitForGuiRunningConsistency(
        page,
        options,
        turnId,
        `after-${options.reopenMode}`,
      ),
    );
    summary.guiRunningAfterReload = summary.guiRunningAfterReopen;
    if (options.multiRunningSessions) {
      summary.multiRunningSidebarAfterReopen = sanitizeJson(
        await waitForGuiSidebarSessionsRunning(
          page,
          options,
          multiRunningSpecs,
          `after-${options.reopenMode}`,
        ),
      );
      summary.multiRunningSecondaryReadModelAfterReopen = sanitizeJson(
        await waitForReadModelRunning(
          page,
          options,
          requestLog,
          summary.multiRunningSecondary.turnId,
          {
            sessionId: MULTI_RUNNING_SECONDARY_SESSION_ID,
            prompt: MULTI_RUNNING_SECONDARY_PROMPT,
            requireContent: options.reopenMode !== "restart",
          },
        ),
      );
      summary.multiRunningSecondarySessionListAfterReopen = sanitizeJson(
        await waitForSessionListRunning(
          page,
          options,
          requestLog,
          summary.multiRunningSecondary.turnId,
          MULTI_RUNNING_SECONDARY_SESSION_ID,
        ),
      );
    }
    await page.screenshot({
      path: screenshotAfterReloadPath,
      fullPage: true,
      timeout: 15_000,
    });
    summary.screenshotAfterReopen = screenshotAfterReloadPath;
    summary.screenshotAfterReload = screenshotAfterReloadPath;

    const traceCursorBeforeCancel = createTraceCursor(
      await readRendererTraceFromPageRef(pageRef, browser, options),
    );
    page = pageRef.current;
    summary.traceCursorBeforeCancel = traceCursorBeforeCancel;

    logStage("stop-after-reload");
    summary.stopClick = sanitizeJson(
      await waitForStopButtonVisibleAndClick(page, options, {
        prompt: NEWS_PROMPT,
        visibleOutputText: "以下是今日国际新闻简要整理",
        requireVisibleOutput: true,
      }),
    );
    summary.turnCancelTraceAfterReload = sanitizeJson(
      await waitForTraceMethodAfter(
        pageRef,
        options,
        traceCursorBeforeCancel,
        APP_SERVER_METHOD_SESSION_TURN_CANCEL,
        { sessionId: SESSION_ID, turnId, browser },
      ),
    );
    summary.turnCancelTraceAfterReopen = summary.turnCancelTraceAfterReload;
    page = pageRef.current;
    if (options.reopenMode === "restart") {
      summary.backendTurnCancel = sanitizeJson({
        skipped: true,
        reason:
          "restart 模式只声明产品恢复绑定与 GUI/read model 收口；不声明 external backend 子进程跨 Electron/App Server 重启存活。",
      });
    } else {
      summary.backendTurnCancel = sanitizeJson(
        await waitForBackendLedgerEntry(
          runtimeEnv.backendLedgerPath,
          (entry) =>
            entry.kind === "turnCancel" &&
            entry.sessionId === SESSION_ID &&
            entry.turnId === turnId,
          options,
        ),
      );
    }
    summary.eventCanceledAfterReload = sanitizeJson(
      await waitForCanceledEventAfterReload(page, options, turnId),
    );
    summary.eventCanceledAfterReopen = summary.eventCanceledAfterReload;
    summary.readModelCanceled = sanitizeJson(
      await waitForSessionReadCanceled(page, options, requestLog, {
        sessionId: SESSION_ID,
        prompt: NEWS_PROMPT,
        partialText: "以下是今日国际新闻简要整理",
        requireContent: options.reopenMode !== "restart",
      }),
    );
    summary.guiCanceled = sanitizeJson(
      await waitForGuiChatCanceled(page, options, {
        prompt: NEWS_PROMPT,
        partialText: "以下是今日国际新闻简要整理",
      }),
    );
    summary.sessionListAfterCancel = sanitizeJson(
      await waitForSessionListNotRunning(page, options, requestLog, turnId),
    );
    summary.guiIdleAfterCancel = sanitizeJson(
      await waitForGuiIdleConsistency(page, options, turnId, {
        requirePrompt: options.reopenMode !== "restart",
      }),
    );
    if (options.multiRunningSessions) {
      summary.multiRunningSecondaryStillRunningAfterPrimaryCancel =
        sanitizeJson(
          await waitForSessionListRunning(
            page,
            options,
            requestLog,
            summary.multiRunningSecondary.turnId,
            MULTI_RUNNING_SECONDARY_SESSION_ID,
          ),
        );
      summary.multiRunningSecondarySidebarAfterPrimaryCancel = sanitizeJson(
        await waitForGuiSidebarSessionsRunning(
          page,
          options,
          [
            {
              ...buildRunningSessionSpec({
                sessionId: MULTI_RUNNING_SECONDARY_SESSION_ID,
                threadId: MULTI_RUNNING_SECONDARY_THREAD_ID,
                title: MULTI_RUNNING_SECONDARY_TITLE,
                prompt: MULTI_RUNNING_SECONDARY_PROMPT,
              }),
              turnId: summary.multiRunningSecondary.turnId,
            },
          ],
          "secondary-after-primary-cancel",
        ),
      );
      summary.multiRunningSecondaryCleanup = sanitizeJson(
        await cancelSecondaryRunningSession(
          page,
          options,
          requestLog,
          runtimeEnv,
          summary.multiRunningSecondary,
        ),
      );
    } else {
      summary.multiRunningSecondaryCleanup = sanitizeJson({
        skipped: true,
        reason: "--multi-running-sessions not enabled",
      });
    }
    await page.screenshot({
      path: screenshotAfterCancelPath,
      fullPage: true,
      timeout: 15_000,
    });
    summary.screenshotAfterCancel = screenshotAfterCancelPath;

    const traceMessages = await readRendererTraceFromPageRef(
      pageRef,
      browser,
      options,
    );
    page = pageRef.current;
    summary.traceSummary = summarizeTraceMessages(traceMessages);
    writeJsonFile(tracePath, summary.traceSummary);

    const backendLedger = readJsonl(runtimeEnv.backendLedgerPath);
    writeJsonFile(
      backendLedgerEvidencePath,
      sanitizeBackendLedgerForEvidence(backendLedger),
    );
    summary.backend = sanitizeJson(summarizeBackendLedger(backendLedger));
    summary.appServerRequestMethods = requestLog
      .map((entry) => entry.method)
      .filter(Boolean);
    summary.turnStartEvidence = summarizeTurnStartEvidence(summary, {
      sessionId: SESSION_ID,
      turnId,
      prompt: NEWS_PROMPT,
    });

    const methods = new Set(summary.traceSummary.methods ?? []);
    const requestMethods = new Set(summary.appServerRequestMethods ?? []);
    const threadResumeSeenAfterReopen = hasSuccessfulTraceEvidence(
      summary.threadResumeTraceAfterReopen,
      APP_SERVER_METHOD_SESSION_THREAD_RESUME,
      { sessionId: SESSION_ID, turnId },
    );
    const threadResumeAssertion =
      summary.reopenMode === "restart"
        ? {
            threadResumeNotRequiredForRestart:
              summary.threadResumeTraceAfterReopen?.skipped === true,
          }
        : {
            threadResumeSeen: threadResumeSeenAfterReopen,
          };
    const homeBackgroundAssertions =
      summary.presentationMode === "background"
        ? {
            homeBackgroundBeforeReopen:
              summary.homeBackgroundBeforeReopen?.homeStartVisible === true &&
              summary.homeBackgroundBeforeReopen?.homeRecoveryCardVisible ===
                true &&
              summary.homeBackgroundBeforeReopen?.homeRecoveryCardStatus ===
                "running" &&
              summary.homeBackgroundBeforeReopen?.homeRecoveryCardTitleFound ===
                true &&
              summary.homeBackgroundBeforeReopen?.activeDetailBoundToSession ===
                false &&
              summary.homeBackgroundBeforeReopen?.sidebarStatus === "running",
            homeBackgroundAfterReopen:
              summary.homeBackgroundAfterReopen?.homeStartVisible === true &&
              summary.homeBackgroundAfterReopen?.homeRecoveryCardVisible ===
                true &&
              summary.homeBackgroundAfterReopen?.homeRecoveryCardStatus ===
                "running" &&
              summary.homeBackgroundAfterReopen?.homeRecoveryCardTitleFound ===
                true &&
              summary.homeBackgroundAfterReopen?.activeDetailBoundToSession ===
                false &&
              summary.homeBackgroundAfterReopen?.sidebarStatus === "running",
            homeRecoveryCardOpenedAfterReopen:
              summary.guiSessionOpenedAfterReopen?.click?.clicked === true &&
              summary.guiSessionOpenedAfterReopen?.running?.sidebarStatus ===
                "running" &&
              summary.guiSessionOpenedAfterReopen?.running
                ?.inputbarHasStopButton === true,
          }
        : {
            homeBackgroundSkippedInForeground:
              summary.homeBackgroundAfterReopen?.skipped === true,
          };
    const multiRunningAssertions = options.multiRunningSessions
      ? {
          multiRunningSecondaryStarted:
            summary.multiRunningSecondary?.session?.session != null &&
            typeof summary.multiRunningSecondary?.turnId === "string" &&
            summary.multiRunningSecondary.turnId.length > 0,
          multiRunningPrimaryAndSecondarySidebarBeforeReopen:
            summary.multiRunningSidebarBeforeReopen?.allRunning === true,
          multiRunningPrimaryAndSecondarySidebarAfterReopen:
            summary.multiRunningSidebarAfterReopen?.allRunning === true,
          multiRunningHomeKeepsPrimaryRecoveryCard:
            summary.presentationMode !== "background" ||
            (summary.homeBackgroundAfterReopen?.homeRecoveryCardTitleFound ===
              true &&
              !String(
                summary.homeBackgroundAfterReopen?.homeRecoveryCardText || "",
              ).includes(MULTI_RUNNING_SECONDARY_TITLE)),
          multiRunningSecondaryStillRunningAfterPrimaryCancel:
            summary.multiRunningSecondaryStillRunningAfterPrimaryCancel
              ?.running === true,
          multiRunningSecondaryCleanupCanceled:
            summary.multiRunningSecondaryCleanup?.sessionListNotRunning
              ?.running === false,
        }
      : {
          multiRunningSessionsSkipped: true,
        };
    summary.assertions = {
      electronPreloadBridge: summary.rendererSnapshot?.electron === true,
      hasInvokeBridge: summary.rendererSnapshot?.hasInvokeBridge === true,
      supportsAppServer: summary.rendererSnapshot?.supportsAppServer === true,
      cdpConnected:
        typeof summary.cdpEndpoint?.version?.browser === "string" ||
        typeof summary.cdpEndpoint?.version?.userAgent === "string",
      turnStartSeen: summary.turnStartEvidence?.matched === true,
      ...threadResumeAssertion,
      turnCancelSeen: hasSuccessfulTraceEvidence(
        summary.turnCancelTraceAfterReload,
        APP_SERVER_METHOD_SESSION_TURN_CANCEL,
        { sessionId: SESSION_ID, turnId },
      ),
      sessionReadSeen:
        requestMethods.has(APP_SERVER_METHOD_SESSION_READ) ||
        methods.has(APP_SERVER_METHOD_SESSION_READ),
      sessionListSeen:
        requestMethods.has(APP_SERVER_METHOD_SESSION_LIST) ||
        methods.has(APP_SERVER_METHOD_SESSION_LIST),
      reopenModeKnown: REOPEN_MODES.has(summary.reopenMode),
      presentationModeKnown: PRESENTATION_MODES.has(summary.presentationMode),
      ...homeBackgroundAssertions,
      sameTurnBeforeReload:
        summary.readModelRunningBeforeReload?.summary?.sameActiveTurn ===
          true ||
        summary.readModelRunningBeforeReload?.summary?.matchedTurn?.turnId ===
          turnId,
      sameTurnAfterReopen:
        summary.readModelRunningAfterReopen?.summary?.sameActiveTurn === true ||
        summary.readModelRunningAfterReopen?.summary?.matchedTurn?.turnId ===
          turnId,
      sameTurnAfterReload:
        summary.readModelRunningAfterReopen?.summary?.sameActiveTurn === true ||
        summary.readModelRunningAfterReopen?.summary?.matchedTurn?.turnId ===
          turnId,
      guiRunningBeforeReload:
        summary.guiRunningBeforeReload?.sidebarStatus === "running" &&
        summary.guiRunningBeforeReload?.inputbarHasStopButton === true,
      guiRunningAfterReopen:
        summary.guiRunningAfterReopen?.sidebarStatus === "running" &&
        summary.guiRunningAfterReopen?.inputbarHasStopButton === true,
      guiRunningAfterReload:
        summary.guiRunningAfterReopen?.sidebarStatus === "running" &&
        summary.guiRunningAfterReopen?.inputbarHasStopButton === true,
      canceledEventAfterReopen:
        summary.eventCanceledAfterReopen?.summary?.terminalTypes?.includes(
          "turn.canceled",
        ) === true,
      canceledEventAfterReload:
        summary.eventCanceledAfterReopen?.summary?.terminalTypes?.includes(
          "turn.canceled",
        ) === true,
      guiIdleAfterCancel:
        summary.guiIdleAfterCancel?.sidebarStatus !== "running" &&
        summary.guiIdleAfterCancel?.inputbarHasStopButton === false,
      noMockBackend: summary.backendMode === "external",
      ...multiRunningAssertions,
    };
    for (const [name, passed] of Object.entries(summary.assertions)) {
      assert(passed, `reopen running turn CDP assertion failed: ${name}`);
    }

    summary.completedGateB = true;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
    console.log(`${LOG_PREFIX} trace=${tracePath}`);
  } catch (error) {
    summary.error = sanitizeText(
      error instanceof Error ? error.stack || error.message : String(error),
    );
    try {
      const backendLedger = readJsonl(runtimeEnv.backendLedgerPath);
      writeJsonFile(
        backendLedgerEvidencePath,
        sanitizeBackendLedgerForEvidence(backendLedger),
      );
      summary.backend = sanitizeJson(summarizeBackendLedger(backendLedger));
    } catch (ledgerError) {
      summary.backendLedgerError = sanitizeText(ledgerError);
    }
    writeJsonFile(summaryPath, summary);
    console.error(`${LOG_PREFIX} failureSummary=${summaryPath}`);
    throw error;
  } finally {
    await browser?.close().catch(() => undefined);
    await app?.close().catch(() => undefined);
    if (!options.keepTemp) {
      cleanupTempRoot(runtimeEnv.tempRoot);
    }
  }
}

await run().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : error);
  process.exit(1);
});
