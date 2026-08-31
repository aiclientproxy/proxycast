#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { ensureElectronFixtureBuild } from "../lib/electron-fixture-build.mjs";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
} from "./lib/mcp-config-fixture-evidence.mjs";
import {
  appServerCallFromPage,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  sanitizeText,
  waitForPageCondition,
} from "./mcp-config-fixture-smoke.mjs";

export const THREAD_FORK_TITLE = "Thread Fork Gate B canonical thread";
export const THREAD_FORK_REQUIRED_METHODS = [
  "thread/start",
  "thread/fork",
  "thread/list",
  "thread/read",
  "thread/resume",
];
export const THREAD_FORK_FORBIDDEN_METHODS = [
  "turn/start",
  "agentSession/create",
  "agentSession/fork",
];

const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "thread-fork-electron-gate-b",
  ),
  prefix: "thread-fork-electron-gate-b",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LOG_PREFIX = "[smoke:thread-fork-gate-b]";
const FORK_ACTION_LABEL = /分叉对话|分叉對話|Fork|会話を分岐|대화 분기/u;

function printHelp() {
  console.log(`
Thread Fork Electron Gate B

用途:
  从真实 Electron Thread header 菜单执行 Fork，验证 parent/child lineage、
  侧栏激活项、成功反馈和 canonical read/resume identity。

边界:
  使用 unavailable backend；不启动 Turn、不调用模型，不使用 mock backend、
  renderer fallback 或 legacy session fork。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function parseThreadForkGateArgs(argv, defaults = DEFAULTS) {
  const options = { ...defaults, help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
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
  if (options.help) return options;
  if (!PREFIX_PATTERN.test(String(options.prefix ?? ""))) {
    throw new Error("invalid evidence prefix");
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  return options;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function requestSummary(request) {
  return {
    method: request.method,
    transport: request.transport,
    status: request.status,
  };
}

function sanitizeDiagnosticText(value) {
  return sanitizeText(value)
    .replace(/\/(?:Users|home)\/[^\s"'`]+/g, "[local-path]")
    .replace(/[A-Za-z]:\\[^\s"'`]+/g, "[local-path]");
}

export function summarizeThreadForkEvidence({
  traceRaw,
  errorRaw,
  parentThreadId,
  parentSessionId,
  childRead,
  listedThreads = [],
  notifications = [],
  beforeDom,
  menuDom,
  afterDom,
  setupRequests = [],
  observedActionRequests = [],
}) {
  const traceEntries = parseInvokeTraceRaw(traceRaw);
  const actionRequests = mergeObservedRequests(
    parseJsonRpcRequestsFromInvokeTrace(traceRaw),
    observedActionRequests,
  );
  const requests = [...setupRequests, ...actionRequests];
  const electronRequests = requests.filter(
    (request) =>
      request.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
      request.transport === "electron-ipc" &&
      request.status === "success",
  );
  const methods = Array.from(
    new Set(electronRequests.map((request) => request.method)),
  );
  const childThread = childRead?.thread ?? null;
  const childThreadId = String(childThread?.id || "").trim();
  const childSessionId = String(childThread?.sessionId || "").trim();
  const forkRequests = actionRequests.filter(
    (request) => request.method === "thread/fork",
  );
  const matchingForkRequests = forkRequests.filter(
    (request) =>
      request.transport === "electron-ipc" &&
      request.status === "success" &&
      request.params?.threadId === parentThreadId,
  );
  const childOpenRequests = actionRequests.filter(
    (request) =>
      ["thread/read", "thread/resume"].includes(request.method) &&
      request.transport === "electron-ipc" &&
      request.status === "success" &&
      request.params?.threadId === childThreadId,
  );
  const listedIds = new Set(
    listedThreads
      .map((thread) => String(thread?.id || "").trim())
      .filter(Boolean),
  );
  const childStartedNotifications = notifications.filter(
    (message) =>
      message?.method === "thread/started" &&
      message?.params?.thread?.id === childThreadId &&
      message?.params?.thread?.forkedFromId === parentThreadId,
  );
  const relevantTrace = traceEntries.filter((entry) =>
    [
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      APP_SERVER_DRAIN_EVENTS_COMMAND,
    ].includes(entry?.command),
  );

  return {
    identity: {
      parentChildDistinct:
        Boolean(parentThreadId) &&
        Boolean(childThreadId) &&
        parentThreadId !== childThreadId,
      forkRequestMatchesParent:
        forkRequests.length === 1 && matchingForkRequests.length === 1,
      childReadMatchesActiveSession:
        childThreadId === afterDom?.activeThreadId &&
        childSessionId === afterDom?.activeSessionId,
      childReadPreservesForkLineage:
        childThread?.forkedFromId === parentThreadId,
      parentAndChildListed:
        listedIds.has(parentThreadId) && listedIds.has(childThreadId),
      childStartedNotificationMatches: childStartedNotifications.length > 0,
      parentSessionWasActive: beforeDom?.activeSessionId === parentSessionId,
      childOpenMethods: Array.from(
        new Set(childOpenRequests.map((request) => request.method)),
      ),
    },
    bridge: {
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      methods,
      missingMethods: THREAD_FORK_REQUIRED_METHODS.filter(
        (method) => !methods.includes(method),
      ),
      forbiddenMethods: THREAD_FORK_FORBIDDEN_METHODS.filter((method) =>
        methods.includes(method),
      ),
      appServerHandleJsonLinesSeen: electronRequests.length > 0,
      appServerDrainEventsSeen: relevantTrace.some(
        (entry) =>
          entry.command === APP_SERVER_DRAIN_EVENTS_COMMAND &&
          entry.transport === "electron-ipc" &&
          entry.status === "success",
      ),
      mockFallbackHitCount: relevantTrace.filter(
        (entry) => entry.transport !== "electron-ipc",
      ).length,
      failedInvokeCount: relevantTrace.filter(
        (entry) => entry.status !== "success",
      ).length,
    },
    gui: {
      headerTitlePreserved:
        beforeDom?.headerTitle === THREAD_FORK_TITLE &&
        afterDom?.headerTitle === THREAD_FORK_TITLE,
      actionMenuVisible: menuDom?.actionMenuVisible === true,
      forkActionVisible: menuDom?.forkActionVisible === true,
      successToastVisible: afterDom?.successToastVisible === true,
      parentActiveBefore: beforeDom?.parentActive === true,
      childActiveAfter: afterDom?.childActive === true,
      matchingConversationCount: Number(
        afterDom?.matchingConversationCount ?? 0,
      ),
    },
    errors: {
      invokeErrorCount: parseInvokeTraceRaw(errorRaw).length,
    },
    requests: electronRequests
      .filter((request) =>
        THREAD_FORK_REQUIRED_METHODS.includes(request.method),
      )
      .map(requestSummary),
  };
}

function mergeObservedRequests(traceRequests, observedRequests) {
  const merged = [];
  const seen = new Set();
  for (const request of [...observedRequests, ...traceRequests]) {
    const key = JSON.stringify([
      request?.method ?? null,
      request?.params?.threadId ?? null,
      request?.transport ?? null,
      request?.status ?? null,
    ]);
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(request);
  }
  return merged;
}

export function summarizeThreadForkFailure({
  traceRaw,
  errorRaw,
  toasts = [],
  dom = {},
  observer = {},
  consoleErrors = [],
  pageErrors = [],
}) {
  const requests = parseJsonRpcRequestsFromInvokeTrace(traceRaw);
  const invokeErrors = parseInvokeTraceRaw(errorRaw);
  const summarizeEntry = (entry) => ({
    command: String(entry?.command || "") || null,
    transport: String(entry?.transport || "") || null,
    status: String(entry?.status || "") || null,
    ...(entry?.error
      ? { error: sanitizeDiagnosticText(String(entry.error)).slice(0, 500) }
      : {}),
  });
  return {
    methods: Array.from(new Set(requests.map((request) => request.method))),
    forkTrace: requests
      .filter((request) => request.method === "thread/fork")
      .slice(-3)
      .map(summarizeEntry),
    invokeErrors: invokeErrors.slice(-5).map(summarizeEntry),
    toasts: toasts.slice(-5).map((toast) => ({
      type: String(toast?.type || "") || null,
      text: sanitizeDiagnosticText(String(toast?.text || "")).slice(0, 300),
    })),
    dom: {
      headerTitle: sanitizeDiagnosticText(String(dom?.headerTitle || "")).slice(
        0,
        200,
      ),
      activeConversationPresent: dom?.activeConversationPresent === true,
      matchingConversationCount: Number(dom?.matchingConversationCount ?? 0),
    },
    observer: {
      methods: Array.from(
        new Set(
          (observer?.requests ?? [])
            .map((request) => String(request?.method || ""))
            .filter(Boolean),
        ),
      ),
      forkSeen: observer?.forkSeen === true,
      childReadSeen: observer?.childReadSeen === true,
      resumeSeen: observer?.resumeSeen === true,
      successToastSeen: observer?.successToastSeen === true,
      errorToastSeen: observer?.errorToastSeen === true,
      activeSessionChanged: observer?.activeSessionChanged === true,
      headerSessionChanged: observer?.headerSessionChanged === true,
      composerSessionChanged: observer?.composerSessionChanged === true,
    },
    consoleErrors: consoleErrors
      .slice(-5)
      .map((error) => sanitizeDiagnosticText(String(error)).slice(0, 500)),
    pageErrors: pageErrors
      .slice(-5)
      .map((error) => sanitizeDiagnosticText(String(error)).slice(0, 500)),
  };
}

export function assertThreadForkEvidence(evidence) {
  assert(
    evidence.bridge.appServerHandleJsonLinesSeen,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    evidence.bridge.appServerDrainEventsSeen,
    "Fork 后未观察到 app_server_drain_events",
  );
  assert(
    evidence.bridge.missingMethods.length === 0,
    `缺少 Fork current method: ${evidence.bridge.missingMethods.join(", ")}`,
  );
  assert(
    evidence.bridge.forbiddenMethods.length === 0,
    `Fork 命中退役 method: ${evidence.bridge.forbiddenMethods.join(", ")}`,
  );
  assert(
    evidence.bridge.mockFallbackHitCount === 0,
    "Fork 命中了非 electron-ipc transport",
  );
  assert(evidence.bridge.failedInvokeCount === 0, "Fork current invoke 失败");
  assert(
    evidence.identity.parentChildDistinct,
    "Fork 未创建新的 Thread identity",
  );
  assert(
    evidence.identity.forkRequestMatchesParent,
    "thread/fork 未精确命中 parent Thread",
  );
  assert(
    evidence.identity.childReadMatchesActiveSession,
    "Fork 后侧栏 active session 与 canonical child Thread 不一致",
  );
  assert(
    evidence.identity.childReadPreservesForkLineage,
    "child thread/read 未保留 forkedFromId",
  );
  assert(
    evidence.identity.parentAndChildListed,
    "thread/list 未同时保留 parent 和 child",
  );
  assert(
    evidence.identity.childStartedNotificationMatches,
    "thread/started notification 未携带同一 parent/child identity",
  );
  assert(
    evidence.identity.parentSessionWasActive,
    "Fork 前 parent Thread 未处于 active 状态",
  );
  assert(
    evidence.identity.childOpenMethods.includes("thread/read") &&
      evidence.identity.childOpenMethods.includes("thread/resume"),
    "Fork 后未通过 thread/read + thread/resume 打开 child Thread",
  );
  assert(evidence.gui.headerTitlePreserved, "Fork 后 Thread 标题未继承");
  assert(evidence.gui.actionMenuVisible, "Thread action menu 不可见");
  assert(evidence.gui.forkActionVisible, "Thread Fork action 不可见");
  assert(evidence.gui.successToastVisible, "Fork 成功反馈不可见");
  assert(evidence.gui.parentActiveBefore, "Fork 前 parent 侧栏项未激活");
  assert(evidence.gui.childActiveAfter, "Fork 后 child 侧栏项未激活");
  assert(
    evidence.gui.matchingConversationCount === 2,
    `Fork 后应保留两个同名 Thread，实际: ${evidence.gui.matchingConversationCount}`,
  );
  assert(
    evidence.errors.invokeErrorCount === 0,
    `观察到 invoke error: ${evidence.errors.invokeErrorCount}`,
  );
}

async function waitForConversation(page, options) {
  await waitForPageCondition(
    page,
    options,
    (title) =>
      Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      ).some(
        (button) =>
          button.getAttribute("title") === title ||
          button.textContent?.includes(title),
      ),
    "Thread Fork parent 侧栏会话未出现",
    THREAD_FORK_TITLE,
  );
}

async function clickConversation(page) {
  const clicked = await page.evaluate((title) => {
    const button = Array.from(
      document.querySelectorAll(
        '[data-testid="app-sidebar-conversation-open"]',
      ),
    ).find(
      (candidate) =>
        candidate.getAttribute("title") === title ||
        candidate.textContent?.includes(title),
    );
    if (!(button instanceof HTMLButtonElement)) return false;
    button.click();
    return true;
  }, THREAD_FORK_TITLE);
  assert(clicked, "无法点击 Thread Fork parent 侧栏会话");
}

async function readThreadDom(page, parentSessionId = "") {
  return await page.evaluate(
    ({ title, parentSessionId }) => {
      const rows = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      );
      const matchingRows = rows.filter(
        (row) =>
          row.getAttribute("title") === title ||
          row.textContent?.includes(title),
      );
      const activeRow = rows.find(
        (row) => row.getAttribute("aria-current") === "page",
      );
      const header = document.querySelector(
        '[data-testid="thread-workspace-header-title"]',
      );
      return {
        headerTitle: header?.textContent?.trim() || "",
        activeSessionId: activeRow?.getAttribute("data-session-id") || "",
        activeThreadId: "",
        parentActive:
          activeRow?.getAttribute("data-session-id") === parentSessionId,
        childActive:
          Boolean(activeRow?.getAttribute("data-session-id")) &&
          activeRow?.getAttribute("data-session-id") !== parentSessionId,
        matchingConversationCount: matchingRows.length,
        successToastVisible: false,
      };
    },
    { title: THREAD_FORK_TITLE, parentSessionId },
  );
}

async function readRecentNotifications(page) {
  return await page.evaluate(async (command) => {
    const invoke = window.electronAPI?.invoke;
    if (typeof invoke !== "function") {
      throw new Error("Electron preload invoke bridge is unavailable");
    }
    const response = await invoke(command, {
      request: { includeRecent: true, limit: 100 },
    });
    return Array.isArray(response?.lines)
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
  }, APP_SERVER_DRAIN_EVENTS_COMMAND);
}

async function waitForForkOutcome(
  page,
  options,
  parentThreadId,
  parentSessionId,
) {
  return await waitForPageCondition(
    page,
    options,
    ({ parentThreadId, parentSessionId, requiredMethods }) => {
      let trace = [];
      try {
        trace = JSON.parse(
          window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
        );
      } catch {
        trace = [];
      }
      const requests = trace.flatMap((entry) => {
        if (entry?.command !== "app_server_handle_json_lines") return [];
        const lines = entry?.args_preview?.request?.lines;
        if (!Array.isArray(lines)) return [];
        return lines.flatMap((line) => {
          try {
            return [
              {
                ...JSON.parse(String(line)),
                command: entry.command,
                transport: entry.transport,
                status: entry.status,
              },
            ];
          } catch {
            return [];
          }
        });
      });
      const state = window.__LIME_THREAD_FORK_GATE_REQUEST_STATE__ || {
        requests: [],
      };
      const requestKeys = new Set(
        state.requests.map((request) =>
          JSON.stringify([
            request?.method ?? null,
            request?.params?.threadId ?? null,
            request?.transport ?? null,
            request?.status ?? null,
          ]),
        ),
      );
      for (const request of requests) {
        if (!requiredMethods.includes(request?.method)) continue;
        const key = JSON.stringify([
          request?.method ?? null,
          request?.params?.threadId ?? null,
          request?.transport ?? null,
          request?.status ?? null,
        ]);
        if (requestKeys.has(key)) continue;
        requestKeys.add(key);
        state.requests.push(request);
      }
      window.__LIME_THREAD_FORK_GATE_REQUEST_STATE__ = state;
      const forkSeen = state.requests.some(
        (request) =>
          request?.method === "thread/fork" &&
          request?.params?.threadId === parentThreadId,
      );
      const childRead = state.requests.find(
        (request) =>
          request?.method === "thread/read" &&
          typeof request?.params?.threadId === "string" &&
          request.params.threadId !== parentThreadId,
      );
      const childThreadId = childRead?.params?.threadId || "";
      const resumeSeen = state.requests.some(
        (request) =>
          request?.method === "thread/resume" &&
          request?.params?.threadId === childThreadId,
      );
      const activeRow = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      ).find((row) => row.getAttribute("aria-current") === "page");
      const activeSessionId = activeRow?.getAttribute("data-session-id") || "";
      const headerSessionId =
        document
          .querySelector('[data-testid="thread-workspace-header"]')
          ?.getAttribute("data-session-id") || "";
      const composerSessionId =
        document
          .querySelector('textarea[name="agent-chat-message"]')
          ?.getAttribute("data-session-id") || "";
      const toastState = Array.from(
        document.querySelectorAll("[data-sonner-toast]"),
      ).reduce(
        (state, toast) => {
          const type = toast.getAttribute("data-type") || "";
          if (type === "success") state.successSeen = true;
          if (type === "error") {
            state.errorSeen = true;
            state.errorText = toast.textContent?.trim() || "";
          }
          return state;
        },
        window.__LIME_THREAD_FORK_GATE_TOAST_STATE__ || {
          successSeen: false,
          errorSeen: false,
          errorText: "",
        },
      );
      window.__LIME_THREAD_FORK_GATE_TOAST_STATE__ = toastState;
      let invokeErrors = [];
      try {
        invokeErrors = JSON.parse(
          window.localStorage.getItem("lime_invoke_error_buffer_v1") || "[]",
        );
      } catch {
        invokeErrors = [];
      }
      const failedForkTrace = trace.some((entry) => {
        if (entry?.status === "success") return false;
        const lines = entry?.args_preview?.request?.lines;
        return Array.isArray(lines)
          ? lines.some((line) => {
              try {
                return JSON.parse(String(line))?.method === "thread/fork";
              } catch {
                return false;
              }
            })
          : false;
      });
      state.forkSeen = forkSeen;
      state.childReadSeen = Boolean(childThreadId);
      state.resumeSeen = resumeSeen;
      state.successToastSeen = toastState.successSeen;
      state.errorToastSeen = toastState.errorSeen;
      state.activeSessionChanged =
        Boolean(activeSessionId) && activeSessionId !== parentSessionId;
      state.headerSessionChanged =
        Boolean(headerSessionId) && headerSessionId !== parentSessionId;
      state.composerSessionChanged =
        Boolean(composerSessionId) && composerSessionId !== parentSessionId;
      if (toastState.errorSeen || invokeErrors.length > 0 || failedForkTrace) {
        return {
          status: "error",
          errorText: toastState.errorText,
          invokeErrorCount: invokeErrors.length,
          failedForkTrace,
        };
      }
      if (
        !forkSeen ||
        !childThreadId ||
        !resumeSeen ||
        !activeSessionId ||
        activeSessionId === parentSessionId ||
        !toastState.successSeen
      ) {
        return null;
      }
      return {
        status: "success",
        childThreadId,
        activeSessionId,
        observedRequests: state.requests,
        successToastSeen: toastState.successSeen,
      };
    },
    "Thread Fork GUI 未进入明确成功或失败终态",
    {
      parentThreadId,
      parentSessionId,
      requiredMethods: THREAD_FORK_REQUIRED_METHODS,
    },
  );
}

async function run() {
  const options = parseThreadForkGateArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );
  const runtimeEnv = createTempRuntimeEnv();
  const workspaceRoot = path.join(runtimeEnv.tempRoot, "workspace");
  fs.mkdirSync(workspaceRoot, { recursive: true });
  let handle = null;
  const consoleErrors = [];
  const pageErrors = [];
  const summary = {
    schemaVersion: 1,
    scenarioId: "CODEX-ALIGN-thread-fork",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron GUI thread/fork from one paginated canonical source Thread to one canonical forked Thread, including sidebar activation, forkedFromId lineage, read/resume and notification identity. It does not start a Turn or call a model.",
    backendMode: "unavailable",
    ok: false,
    checkedAt: new Date().toISOString(),
    identity: null,
    bridge: null,
    gui: null,
    errors: null,
    requests: [],
    screenshot: null,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
  };

  try {
    ensureElectronFixtureBuild({
      logPrefix: LOG_PREFIX,
      rootDir: process.cwd(),
    });
    const appServerBinary = resolveDevAppServerBinary({
      env: runtimeEnv.env,
      repoRoot: process.cwd(),
      forceBuild: false,
    });
    const appServerEnv = resolveElectronAppServerRuntimeEnv({
      env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
    });

    console.log(`${LOG_PREFIX} stage=launch-electron`);
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    await handle.page.evaluate(() => {
      window.localStorage.setItem("lime.app-sidebar.collapsed", "false");
    });

    console.log(`${LOG_PREFIX} stage=create-parent-thread`);
    const started = await appServerCallFromPage(handle.page, "thread/start", {
      model: "fixture-model",
      modelProvider: "fixture-provider",
      cwd: workspaceRoot,
      serviceName: THREAD_FORK_TITLE,
      historyMode: "paginated",
      threadSource: "fixture",
    });
    const parentThreadId = String(started.result?.thread?.id || "").trim();
    const parentSessionId = String(
      started.result?.thread?.sessionId || "",
    ).trim();
    assert(parentThreadId, "thread/start 未返回 parent Thread ID");
    assert(parentSessionId, "thread/start 未返回 parent Session ID");
    const setupRequests = [
      {
        command: started.appServerCommand,
        method: started.method,
        transport: "electron-ipc",
        status: "success",
        params: {},
      },
    ];

    console.log(`${LOG_PREFIX} stage=open-parent-from-sidebar`);
    await handle.page.reload({ waitUntil: "domcontentloaded" });
    await waitForPageCondition(
      handle.page,
      options,
      () =>
        window.__LIME_ELECTRON__ === true &&
        typeof window.electronAPI?.invoke === "function" &&
        Boolean(document.querySelector('[data-testid="app-sidebar"]')),
      "Electron renderer reload 未就绪",
    );
    await waitForConversation(handle.page, options);
    await clickConversation(handle.page);
    await handle.page
      .locator('[data-testid="thread-workspace-header-action-menu"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    const beforeDom = await readThreadDom(handle.page, parentSessionId);
    await handle.page.evaluate(() => {
      window.localStorage.removeItem("lime_invoke_error_buffer_v1");
      window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
      window.__LIME_THREAD_FORK_GATE_TOAST_STATE__ = {
        successSeen: false,
        errorSeen: false,
        errorText: "",
      };
      window.__LIME_THREAD_FORK_GATE_REQUEST_STATE__ = { requests: [] };
    });

    console.log(`${LOG_PREFIX} stage=open-fork-menu`);
    await handle.page
      .locator('[data-testid="thread-workspace-header-action-menu"]')
      .click();
    const forkAction = handle.page
      .getByRole("menuitem")
      .filter({ hasText: FORK_ACTION_LABEL });
    await forkAction.waitFor({ state: "visible", timeout: options.timeoutMs });
    const menuDom = {
      actionMenuVisible: true,
      forkActionVisible: await forkAction.isVisible(),
    };

    console.log(`${LOG_PREFIX} stage=fork-from-gui`);
    await forkAction.click();
    const forked = await waitForForkOutcome(
      handle.page,
      options,
      parentThreadId,
      parentSessionId,
    );
    if (forked.status !== "success") {
      throw new Error(
        `thread/fork GUI failed: ${forked.errorText || `invokeErrors=${forked.invokeErrorCount}, failedTrace=${forked.failedForkTrace}`}`,
      );
    }

    console.log(`${LOG_PREFIX} stage=read-child-lineage`);
    const childReadResponse = await appServerCallFromPage(
      handle.page,
      "thread/read",
      { threadId: forked.childThreadId, includeTurns: false },
    );
    const listResponse = await appServerCallFromPage(
      handle.page,
      "thread/list",
      { archived: false, limit: 100 },
    );
    const notifications = await readRecentNotifications(handle.page);
    const afterDom = await readThreadDom(handle.page, parentSessionId);
    afterDom.activeThreadId = forked.childThreadId;
    afterDom.successToastVisible = forked.successToastSeen === true;
    const observed = await handle.page.evaluate(() => ({
      traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    }));
    const evidence = summarizeThreadForkEvidence({
      ...observed,
      parentThreadId,
      parentSessionId,
      childRead: childReadResponse.result,
      listedThreads: listResponse.result?.data ?? [],
      notifications,
      beforeDom,
      menuDom,
      afterDom,
      setupRequests,
      observedActionRequests: forked.observedRequests,
    });
    assertThreadForkEvidence(evidence);
    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );
    assert(
      pageErrors.length === 0,
      `观察到 page error: ${pageErrors.join(" | ")}`,
    );

    await handle.page.screenshot({ path: screenshotPath, fullPage: true });
    summary.ok = true;
    summary.identity = evidence.identity;
    summary.bridge = evidence.bridge;
    summary.gui = evidence.gui;
    summary.errors = {
      ...evidence.errors,
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      mockFallbackHitCount: evidence.bridge.mockFallbackHitCount,
    };
    summary.requests = evidence.requests;
    summary.screenshot = `${options.prefix}.png`;
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
  } catch (error) {
    summary.failure = sanitizeDiagnosticText(
      error instanceof Error ? error.message : String(error),
    );
    summary.errors = {
      ...(summary.errors ?? {}),
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    if (handle?.page) {
      const failureState = await handle.page
        .evaluate(() => {
          const readRaw = (key) => window.localStorage.getItem(key);
          const rows = Array.from(
            document.querySelectorAll(
              '[data-testid="app-sidebar-conversation-open"]',
            ),
          );
          return {
            traceRaw: readRaw("lime_invoke_trace_buffer_v1"),
            errorRaw: readRaw("lime_invoke_error_buffer_v1"),
            toasts: Array.from(
              document.querySelectorAll("[data-sonner-toast]"),
            ).map((toast) => ({
              type: toast.getAttribute("data-type") || "",
              text: toast.textContent?.trim() || "",
            })),
            dom: {
              headerTitle:
                document
                  .querySelector(
                    '[data-testid="thread-workspace-header-title"]',
                  )
                  ?.textContent?.trim() || "",
              activeConversationPresent: rows.some(
                (row) => row.getAttribute("aria-current") === "page",
              ),
              matchingConversationCount: rows.filter(
                (row) =>
                  row.getAttribute("title") ===
                    "Thread Fork Gate B canonical thread" ||
                  row.textContent?.includes(
                    "Thread Fork Gate B canonical thread",
                  ),
              ).length,
            },
            observer: window.__LIME_THREAD_FORK_GATE_REQUEST_STATE__ || {},
          };
        })
        .catch(() => null);
      summary.failureDiagnostics = summarizeThreadForkFailure({
        traceRaw: failureState?.traceRaw,
        errorRaw: failureState?.errorRaw,
        toasts: failureState?.toasts,
        dom: failureState?.dom,
        observer: failureState?.observer,
        consoleErrors,
        pageErrors,
      });
      await handle.page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
    }
    writeJsonFile(summaryPath, summary);
    throw error;
  } finally {
    await closeElectronFixture(handle);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  run().catch((error) => {
    console.error(
      `${LOG_PREFIX} failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    process.exitCode = 1;
  });
}
