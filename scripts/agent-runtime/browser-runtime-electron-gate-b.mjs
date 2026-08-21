#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import electronPath from "electron";
import { _electron as electron } from "playwright";
import { startOpenAiCompatibleFixtureServer } from "../lib/openai-compatible-fixture-server.mjs";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { ensureElectronFixtureBuild } from "../lib/electron-fixture-build.mjs";
import { APP_SERVER_METHOD_SESSION_READ } from "./claw-chat-current-fixture-constants.mjs";
import { runBrowserCancelScenario } from "./browser-runtime-electron-gate-b-cancel.mjs";
import {
  installBrowserApprovalPage,
  runBrowserApprovalScenario,
} from "./browser-runtime-electron-gate-b-approval.mjs";
import { runBrowserUserControlScenario } from "./browser-runtime-electron-gate-b-user-control.mjs";
import { runBrowserDisconnectScenario } from "./browser-runtime-electron-gate-b-disconnect.mjs";
import { runBrowserDownloadScenario } from "./browser-runtime-electron-gate-b-download.mjs";
import { runBrowserPermissionScenario } from "./browser-runtime-electron-gate-b-permission.mjs";
import { runBrowserWindowCloseScenario } from "./browser-runtime-electron-gate-b-window-close.mjs";
import { runBrowserProjectionGateA } from "./browser-runtime-gate-a.mjs";
import {
  bindGuiWorkspaceAndModelPreferences,
  ensureDefaultWorkspace,
  initializeAppServer,
  invokeAppServerFromPage,
  waitForRendererReady,
} from "./claw-chat-current-fixture-rpc.mjs";
import {
  createFixtureSession,
  navigateGuiToWorkspaceScopedAgent,
  openFixtureSessionFromSidebar,
  waitForGuiSessionVisible,
} from "./claw-chat-current-fixture-session.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import { createTempRuntimeEnv } from "./claw-chat-current-fixture-backend-file.mjs";
import {
  mergeInvokeTraceEvidence,
  startInvokeTraceEvidenceCollector,
} from "./claw-chat-current-fixture-invoke-trace.mjs";
import {
  assert,
  cleanupTempRoot,
  sanitizeJson,
  sanitizeText,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

const LOG_PREFIX = "[smoke:browser-runtime-electron-gate-b]";
const MODEL_NAME = "fixture-browser-runtime";
const PROVIDER_API_KEY = "fixture-browser-runtime-key";
const BROWSER_URL = "https://example.com/browser-runtime-gate-b";
const USER_NAVIGATION_URL = "https://example.com/browser-runtime-gate-b-user";
const FINAL_MARKER = "BROWSER_RUNTIME_GATE_B_DONE";
const DEFAULT_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-summary.json",
);
const DEFAULT_CANCEL_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-cancel-summary.json",
);
const DEFAULT_WINDOW_CLOSE_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-window-close-summary.json",
);
const DEFAULT_DISCONNECT_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-disconnect-summary.json",
);
const DEFAULT_PERMISSION_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-permission-summary.json",
);
const DEFAULT_DOWNLOAD_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-download-summary.json",
);
const DEFAULT_APPROVAL_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-approval-summary.json",
);
const DEFAULT_USER_CONTROL_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-electron-gate-b/browser-runtime-electron-gate-b-user-control-summary.json",
);
const DEFAULT_PROJECTION_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/browser-runtime-gate-a/browser-runtime-gate-a-summary.json",
);
const SCENARIOS = new Set([
  "projection",
  "lifecycle",
  "approval",
  "user-control",
  "cancel",
  "window-close",
  "disconnect",
  "permission",
  "download",
]);
const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "canceled",
  "interrupted",
]);

export function parseArgs(argv) {
  const options = {
    output: DEFAULT_OUTPUT,
    timeoutMs: 180_000,
    intervalMs: 500,
    appUrl: process.env.VITE_DEV_SERVER_URL?.trim() || "",
    keepTemp: false,
    scenario: "lifecycle",
  };
  let outputExplicit = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      return options;
    }
    if (arg === "--output" && next) {
      options.output = path.resolve(next);
      outputExplicit = true;
      index += 1;
      continue;
    }
    if (arg === "--scenario" && next) {
      options.scenario = String(next).trim();
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
    if (arg === "--app-url" && next) {
      options.appUrl = String(next).trim();
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
    throw new Error("--timeout-ms 必须 >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须 >= 100");
  }
  if (!SCENARIOS.has(options.scenario)) {
    throw new Error(
      `--scenario 必须是 projection、lifecycle、approval、user-control、cancel、window-close、disconnect、permission 或 download，收到: ${options.scenario || "<empty>"}`,
    );
  }
  if (!outputExplicit && options.scenario === "cancel") {
    options.output = DEFAULT_CANCEL_OUTPUT;
  }
  if (!outputExplicit && options.scenario === "window-close") {
    options.output = DEFAULT_WINDOW_CLOSE_OUTPUT;
  }
  if (!outputExplicit && options.scenario === "disconnect") {
    options.output = DEFAULT_DISCONNECT_OUTPUT;
  }
  if (!outputExplicit && options.scenario === "permission") {
    options.output = DEFAULT_PERMISSION_OUTPUT;
  }
  if (!outputExplicit && options.scenario === "download") {
    options.output = DEFAULT_DOWNLOAD_OUTPUT;
  }
  if (!outputExplicit && options.scenario === "approval") {
    options.output = DEFAULT_APPROVAL_OUTPUT;
  }
  if (!outputExplicit && options.scenario === "user-control") {
    options.output = DEFAULT_USER_CONTROL_OUTPUT;
  }
  if (!outputExplicit && options.scenario === "projection") {
    options.output = DEFAULT_PROJECTION_OUTPUT;
  }
  return options;
}

export function extractBrowserState(value) {
  const seen = new Set();
  const visit = (candidate) => {
    if (typeof candidate === "string") {
      try {
        return visit(JSON.parse(candidate));
      } catch {
        return null;
      }
    }
    if (!candidate || typeof candidate !== "object" || seen.has(candidate)) {
      return null;
    }
    seen.add(candidate);
    if (
      typeof candidate.tabId === "string" &&
      typeof candidate.browserSessionId === "string" &&
      typeof candidate.url === "string"
    ) {
      return candidate;
    }
    if (Array.isArray(candidate)) {
      for (const item of candidate) {
        const found = visit(item);
        if (found) return found;
      }
      return null;
    }
    for (const item of Object.values(candidate)) {
      const found = visit(item);
      if (found) return found;
    }
    return null;
  };
  return visit(value);
}

export function extractBrowserObservation(value) {
  const seen = new Set();
  const visit = (candidate) => {
    if (typeof candidate === "string") {
      try {
        return visit(JSON.parse(candidate));
      } catch {
        return null;
      }
    }
    if (!candidate || typeof candidate !== "object" || seen.has(candidate)) {
      return null;
    }
    seen.add(candidate);
    if (
      typeof candidate.snapshotId === "string" &&
      Number.isInteger(candidate.pageRevision)
    ) {
      return candidate;
    }
    for (const item of Array.isArray(candidate)
      ? candidate
      : Object.values(candidate)) {
      const found = visit(item);
      if (found) return found;
    }
    return null;
  };
  return visit(value);
}

function toolResultFromRequest(body) {
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  const toolMessage = [...messages]
    .reverse()
    .find((message) => message?.role === "tool");
  if (!toolMessage) return null;
  const content = toolMessage.content;
  if (typeof content !== "string") return content;
  try {
    return JSON.parse(content);
  } catch {
    return content;
  }
}

async function createBrowserProviderFixture({
  approvalScenario = false,
  connectionDiagnostics = false,
  userControlScenario = false,
} = {}) {
  let releaseUserNavigation;
  let releaseFinalResponse;
  const userNavigationCompleted = new Promise((resolve) => {
    releaseUserNavigation = resolve;
  });
  const finalResponseAllowed = new Promise((resolve) => {
    releaseFinalResponse = resolve;
  });
  const scenario = {
    initial: null,
    approvedMutation: null,
    declinedMutationFailure: null,
    secondObservation: null,
    latestAfterUserNavigation: null,
    recovered: null,
    staleMutationFailure: null,
    userControlFailure: null,
    releaseUserNavigation: () => releaseUserNavigation?.(),
    releaseFinalResponse: () => releaseFinalResponse?.(),
    releaseAll: () => {
      releaseUserNavigation?.();
      releaseFinalResponse?.();
    },
  };
  const scriptedResponses = approvalScenario
    ? createBrowserApprovalResponses(scenario, {
        userControl: userControlScenario,
      })
    : createBrowserLifecycleResponses(
        scenario,
        userNavigationCompleted,
        finalResponseAllowed,
      );
  const server = await startOpenAiCompatibleFixtureServer({
    model: MODEL_NAME,
    apiKey: PROVIDER_API_KEY,
    modelRuntimeFeatures: ["streaming", "tool_calling", "custom_tools"],
    connectionDiagnostics,
    scriptedResponses,
  });
  return Object.assign(server, { scenario });
}

function createBrowserLifecycleResponses(
  scenario,
  userNavigationCompleted,
  finalResponseAllowed,
) {
  return [
    {
      type: "tool_call",
      id: "call-browser-open-tabs",
      name: "browser__openTabs",
      arguments: {},
    },
    ({ body }) => {
      const state = extractBrowserState(toolResultFromRequest(body));
      if (!state) {
        throw new Error("openTabs 未返回可用于 claimTab 的 Browser state");
      }
      return {
        type: "tool_call",
        id: "call-browser-claim-tab",
        name: "browser__claimTab",
        arguments: {
          tabId: state.tabId,
          title: state.title,
          url: state.url,
          pageRevision: state.pageRevision,
        },
      };
    },
    ({ body }) => {
      const state = extractBrowserState(toolResultFromRequest(body));
      if (!state) {
        throw new Error("claimTab 未返回可用于 observe 的 Browser state");
      }
      return {
        type: "tool_call",
        id: "call-browser-observe",
        name: "browser__observe",
        arguments: { tabId: state.tabId },
      };
    },
    ({ body }) => {
      const result = toolResultFromRequest(body);
      const observed = extractBrowserState(result);
      const observation = extractBrowserObservation(result);
      if (!observed || typeof observed.webContentsId !== "number") {
        throw new Error("observe 未返回带 webContentsId 的同页 Browser state");
      }
      if (!observation?.snapshotId) {
        throw new Error("observe 未返回可用于 stale-control 的 snapshotId");
      }
      scenario.initial = { observation, state: observed };
      return userNavigationCompleted.then(() => ({
        type: "tool_call",
        id: "call-browser-open-tabs-after-user-navigation",
        name: "browser__openTabs",
        arguments: {},
      }));
    },
    ({ body }) => {
      const state = extractBrowserState(toolResultFromRequest(body));
      if (!state) {
        throw new Error("用户导航后的 openTabs 未返回 Browser state");
      }
      scenario.latestAfterUserNavigation = state;
      return {
        type: "tool_call",
        id: "call-browser-reclaim-tab",
        name: "browser__claimTab",
        arguments: {
          tabId: state.tabId,
          title: state.title,
          url: state.url,
          pageRevision: state.pageRevision,
        },
      };
    },
    ({ body }) => {
      const state = extractBrowserState(toolResultFromRequest(body));
      if (!state || !scenario.initial?.observation?.snapshotId) {
        throw new Error("重新 claim 后缺少 stale snapshot mutation 输入");
      }
      return {
        type: "tool_call",
        id: "call-browser-stale-press",
        name: "browser__press",
        arguments: {
          tabId: state.tabId,
          snapshotId: scenario.initial.observation.snapshotId,
          key: "ArrowDown",
        },
      };
    },
    ({ body }) => {
      const failure = String(toolResultFromRequest(body) || "");
      if (!failure.includes("Browser page snapshot is stale")) {
        throw new Error(`旧 snapshot mutation 未按预期拒绝: ${failure}`);
      }
      scenario.staleMutationFailure = "stale_snapshot_rejected";
      return {
        type: "tool_call",
        id: "call-browser-observe-after-user-navigation",
        name: "browser__observe",
        arguments: {
          tabId: scenario.latestAfterUserNavigation?.tabId,
        },
      };
    },
    ({ body }) => {
      const result = toolResultFromRequest(body);
      const observed = extractBrowserState(result);
      const observation = extractBrowserObservation(result);
      if (!observed || !observation?.snapshotId) {
        throw new Error("重新 observe 未返回 Browser state 与 snapshotId");
      }
      scenario.recovered = { observation, state: observed };
      return finalResponseAllowed.then(() => ({
        type: "text",
        content: `${FINAL_MARKER}:${JSON.stringify({
          activeTurnId: observed.activeTurnId,
          browserSessionId: observed.browserSessionId,
          initialPageRevision: scenario.initial?.observation?.pageRevision,
          ownerWebContentsId: observed.ownerWebContentsId,
          pageRevision: observed.pageRevision,
          recoveredPageRevision: observation.pageRevision,
          staleMutationRejected:
            scenario.staleMutationFailure === "stale_snapshot_rejected",
          tabId: observed.tabId,
          threadId: observed.threadId,
          title: observed.title,
          url: observed.url,
          viewId: observed.viewId,
          webContentsId: observed.webContentsId,
          windowId: observed.windowId,
        })}`,
      }));
    },
  ];
}

function createBrowserApprovalResponses(
  scenario,
  { userControl = false } = {},
) {
  const dangerousNode = (observation) =>
    observation?.nodes?.find(
      (node) =>
        Number.isInteger(node?.backendNodeId) &&
        String(node?.name || "").includes("Delete account"),
    );
  return [
    {
      type: "tool_call",
      id: "call-browser-open-tabs",
      name: "browser__openTabs",
      arguments: {},
    },
    ({ body }) => {
      const state = extractBrowserState(toolResultFromRequest(body));
      if (!state) {
        throw new Error("openTabs 未返回可用于 claimTab 的 Browser state");
      }
      return {
        type: "tool_call",
        id: "call-browser-claim-tab",
        name: "browser__claimTab",
        arguments: {
          tabId: state.tabId,
          title: state.title,
          url: state.url,
          pageRevision: state.pageRevision,
        },
      };
    },
    ({ body }) => {
      const state = extractBrowserState(toolResultFromRequest(body));
      if (!state) {
        throw new Error("claimTab 未返回可用于 observe 的 Browser state");
      }
      return {
        type: "tool_call",
        id: "call-browser-observe",
        name: "browser__observe",
        arguments: { tabId: state.tabId },
      };
    },
    ({ body }) => {
      const result = toolResultFromRequest(body);
      const observed = extractBrowserState(result);
      const observation = extractBrowserObservation(result);
      if (!observed || typeof observed.webContentsId !== "number") {
        throw new Error("observe 未返回带 webContentsId 的同页 Browser state");
      }
      if (!observation?.snapshotId) {
        throw new Error("observe 未返回可用于 stale-control 的 snapshotId");
      }
      const target = dangerousNode(observation);
      if (!target) {
        throw new Error("observe 未返回 Delete account 危险按钮 node identity");
      }
      scenario.initial = { observation, state: observed, target };
      return {
        type: "tool_call",
        id: "call-browser-sensitive-click-approved",
        name: "browser__click",
        arguments: {
          backendNodeId: target.backendNodeId,
          snapshotId: observation.snapshotId,
          tabId: observed.tabId,
        },
      };
    },
    ({ body }) => {
      const result = toolResultFromRequest(body);
      if (userControl) {
        const failure = JSON.stringify(result);
        if (!/stale|invalid|approval token/i.test(failure)) {
          throw new Error(`Browser 用户接管后旧审批未 fail closed: ${failure}`);
        }
        scenario.userControlFailure = failure;
        return {
          type: "text",
          content: `${FINAL_MARKER}:${JSON.stringify({
            staleApprovalRejected: true,
            tabId: scenario.initial?.state?.tabId,
            threadId: scenario.initial?.state?.threadId,
            webContentsId: scenario.initial?.state?.webContentsId,
          })}`,
        };
      }
      const state = extractBrowserState(result);
      if (!state) {
        throw new Error("批准后的 Browser click 未返回 completed state");
      }
      scenario.approvedMutation = { result, state };
      return {
        type: "tool_call",
        id: "call-browser-observe-before-decline",
        name: "browser__observe",
        arguments: { tabId: state.tabId },
      };
    },
    ({ body }) => {
      const result = toolResultFromRequest(body);
      const observed = extractBrowserState(result);
      const observation = extractBrowserObservation(result);
      const target = dangerousNode(observation);
      if (!observed || !observation?.snapshotId || !target) {
        throw new Error("批准后重新 observe 未返回危险按钮 fresh snapshot");
      }
      scenario.secondObservation = { observation, state: observed, target };
      return {
        type: "tool_call",
        id: "call-browser-sensitive-click-declined",
        name: "browser__click",
        arguments: {
          backendNodeId: target.backendNodeId,
          snapshotId: observation.snapshotId,
          tabId: observed.tabId,
        },
      };
    },
    ({ body }) => {
      const failure = JSON.stringify(toolResultFromRequest(body));
      if (!/拒绝|declin|approval/i.test(failure)) {
        throw new Error(
          `Browser decline 未返回 canonical approval failure: ${failure}`,
        );
      }
      scenario.declinedMutationFailure = failure;
      return {
        type: "text",
        content: `${FINAL_MARKER}:${JSON.stringify({
          approvalResumed: true,
          declineTerminal: true,
          tabId: scenario.secondObservation?.state?.tabId,
          threadId: scenario.secondObservation?.state?.threadId,
          webContentsId: scenario.secondObservation?.state?.webContentsId,
        })}`,
      };
    },
  ];
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

async function closeFixtureElectron(app) {
  if (!app) {
    return;
  }
  const pid = app.process()?.pid ?? null;
  await Promise.race([app.close().catch(() => undefined), sleep(5_000)]);
  if (!pid) {
    return;
  }
  try {
    process.kill(pid, 0);
    process.kill(pid, "SIGTERM");
    await sleep(500);
    try {
      process.kill(pid, 0);
      process.kill(pid, "SIGKILL");
    } catch {
      // The fixture exited after SIGTERM.
    }
  } catch {
    // The fixture exited during app.close().
  }
}

async function waitForBrowserSurface(page, options) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    last = await page.evaluate(() => {
      const panel = document.querySelector(
        '[data-testid="right-surface-browser-panel"]',
      );
      const workspace = document.querySelector(
        '[data-testid="browser-workspace"]',
      );
      const rect = workspace?.getBoundingClientRect();
      return {
        panelVisible: Boolean(
          panel && rect && rect.width > 200 && rect.height > 200,
        ),
        sessionId: workspace?.getAttribute("data-browser-session-id") || null,
        tabId: workspace?.getAttribute("data-browser-tab-id") || null,
        threadId: workspace?.getAttribute("data-browser-thread-id") || null,
        webContentsId:
          Number(workspace?.getAttribute("data-browser-web-contents-id")) ||
          null,
        activeSurface:
          document
            .querySelector('[data-testid="workspace-right-surface-host"]')
            ?.getAttribute("data-surface") || null,
      };
    });
    if (
      last?.panelVisible &&
      last.sessionId &&
      last.tabId &&
      last.threadId &&
      last.webContentsId
    ) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Browser workspace 未完成挂载: ${JSON.stringify(sanitizeJson(last))}`,
  );
}

async function readBrowserWorkspaceState(page) {
  return await page.evaluate(() => {
    const workspace = document.querySelector(
      '[data-testid="browser-workspace"]',
    );
    const address = workspace?.querySelector("form input");
    return {
      activeTurnId:
        workspace?.getAttribute("data-browser-active-turn-id") || null,
      address:
        address instanceof HTMLInputElement ? address.value || null : null,
      controlOwner:
        workspace?.getAttribute("data-browser-control-owner") || null,
      pageRevision:
        Number(workspace?.getAttribute("data-browser-page-revision")) || 0,
      sessionId: workspace?.getAttribute("data-browser-session-id") || null,
      tabId: workspace?.getAttribute("data-browser-tab-id") || null,
      threadId: workspace?.getAttribute("data-browser-thread-id") || null,
      viewId: workspace?.getAttribute("data-browser-view-id") || null,
      webContentsId:
        Number(workspace?.getAttribute("data-browser-web-contents-id")) || null,
    };
  });
}

async function waitForBrowserWorkspaceState(
  page,
  options,
  predicate,
  description,
) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    last = await readBrowserWorkspaceState(page);
    if (predicate(last)) return last;
    await sleep(options.intervalMs);
  }
  throw new Error(`${description}: ${JSON.stringify(sanitizeJson(last))}`);
}

async function waitForScenarioValue(options, read, description) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < options.timeoutMs) {
    const value = read();
    if (value) return value;
    await sleep(options.intervalMs);
  }
  throw new Error(`${description} 超时`);
}

async function readBrowserDebuggerState(app, webContentsId) {
  return await app.evaluate(({ webContents }, targetId) => {
    const target = webContents.fromId(targetId);
    return {
      attached: Boolean(
        target && !target.isDestroyed() && target.debugger.isAttached(),
      ),
      exists: Boolean(target && !target.isDestroyed()),
      webContentsId: target?.id ?? null,
    };
  }, webContentsId);
}

async function destroyBrowserWebContents(app, webContentsId) {
  return await app.evaluate(({ webContents }, targetId) => {
    const target = webContents.fromId(targetId);
    if (!target) {
      return { exists: false };
    }
    if (!target.isDestroyed()) {
      // 延迟到 evaluate 返回后终止 native renderer，避免破坏 Playwright 的主进程响应。
      setImmediate(() => {
        if (!target.isDestroyed()) {
          target.forcefullyCrashRenderer();
        }
      });
    }
    return { exists: true, trigger: "render-process-gone" };
  }, webContentsId);
}

async function waitForTerminalThread(page, options, threadId, requestLog) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      { threadId, includeTurns: true },
      requestLog,
    );
    const turns = Array.isArray(read.result?.thread?.turns)
      ? read.result.thread.turns
      : Array.isArray(read.result?.turns)
        ? read.result.turns
        : [];
    last = turns.at(-1) || null;
    const status = String(last?.status || last?.state || "").toLowerCase();
    if (TERMINAL_STATUSES.has(status)) return { turn: last, read: read.result };
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Browser Gate B 回合未结束: ${JSON.stringify(sanitizeJson(last))}`,
  );
}

async function navigateVisibleBrowserTab(page, options, tabId) {
  const address = page
    .locator('[data-testid="browser-workspace"] form input')
    .first();
  assert((await address.count()) === 1, "Browser workspace 地址栏不可用");
  await address.fill(USER_NAVIGATION_URL);
  await address.press("Enter");

  const startedAt = Date.now();
  let lastTrace = [];
  while (Date.now() - startedAt < options.timeoutMs) {
    lastTrace = await page.evaluate(() => {
      try {
        return JSON.parse(
          window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
        );
      } catch {
        return [];
      }
    });
    const navigation = summarizeInvokeTrace(lastTrace).find(
      (entry) =>
        entry.command === "browser_tab_navigate" &&
        entry.status === "success" &&
        entry.args?.tabId === tabId &&
        entry.args?.url === USER_NAVIGATION_URL,
    );
    if (navigation) return { navigation, trace: lastTrace };
    await sleep(options.intervalMs);
  }
  throw new Error(
    `用户导航未命中同一 Browser tab: ${JSON.stringify(
      sanitizeJson(summarizeInvokeTrace(lastTrace)),
    )}`,
  );
}

function summarizeInvokeTrace(trace) {
  return (Array.isArray(trace) ? trace : [])
    .filter((entry) => entry?.transport === "electron-ipc")
    .filter((entry) =>
      ["browser_tab_mount", "browser_tab_navigate"].includes(entry?.command),
    )
    .map((entry) => ({
      command: entry.command,
      status: entry.status,
      args: entry.args_preview || null,
    }));
}

export function extractFinalBrowserState(value) {
  const text = String(value || "");
  const marker = `${FINAL_MARKER}:`;
  const markerIndex = text.lastIndexOf(marker);
  if (markerIndex < 0) return null;
  const payload = text
    .slice(markerIndex + marker.length)
    .split("\n", 1)[0]
    ?.trim();
  if (!payload) return null;
  try {
    const parsed = JSON.parse(payload);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

export function isReclaimedBrowserWorkspaceState({
  activeTurnId,
  initialWebContentsId,
  recovered,
  state,
}) {
  return (
    state?.controlOwner === "agent" &&
    state?.activeTurnId === activeTurnId &&
    state?.webContentsId === initialWebContentsId &&
    state?.pageRevision === recovered?.observation?.pageRevision
  );
}

export function buildAssertions({
  debuggerAfterTerminal,
  debuggerBeforeTerminal,
  destroyed,
  gui,
  initial,
  latestAfterUserNavigation,
  recovered,
  released,
  staleMutationFailure,
  trace,
  providerRequests,
  identity,
  turnId,
  finalText,
}) {
  const browserTrace = summarizeInvokeTrace(trace);
  const mount = browserTrace.find(
    (entry) =>
      entry.command === "browser_tab_mount" && entry.status === "success",
  );
  const navigate = browserTrace.find(
    (entry) =>
      entry.command === "browser_tab_navigate" && entry.status === "success",
  );
  const providerBodies = (
    Array.isArray(providerRequests) ? providerRequests : []
  )
    .filter((request) => request?.path === "/v1/chat/completions")
    .map((request) => request.body);
  const toolNames = providerBodies.flatMap((body) =>
    Array.isArray(body?.tools)
      ? body.tools.map((tool) => tool?.function?.name || tool?.name || null)
      : [],
  );
  const dynamicToolCallCount = providerBodies.filter(
    (body) =>
      Array.isArray(body?.messages) &&
      body.messages.some((message) => message?.role === "tool"),
  ).length;
  const observed = extractFinalBrowserState(finalText);
  return {
    electronBrowserMounted: Boolean(mount?.status === "success"),
    browserSurfaceVisible:
      gui?.panelVisible === true && gui?.activeSurface === "browser",
    canonicalThreadBound: gui?.threadId === identity?.threadId,
    sameTabAcrossMountAndNavigate:
      Boolean(mount?.args?.tabId) &&
      Boolean(navigate?.args?.tabId) &&
      mount.args.tabId === navigate.args.tabId,
    webContentsBound:
      Number.isInteger(gui?.webContentsId) && gui.webContentsId > 0,
    sameBrowserSessionAsAgent:
      Boolean(gui?.sessionId) && gui.sessionId === observed?.browserSessionId,
    sameTabAsAgent: Boolean(gui?.tabId) && gui.tabId === observed?.tabId,
    sameThreadAsAgent:
      Boolean(identity?.threadId) &&
      identity.threadId === observed?.threadId &&
      gui?.threadId === observed?.threadId,
    sameTurnAsAgent: Boolean(turnId) && turnId === observed?.activeTurnId,
    sameWebContentsAsAgent:
      Number.isInteger(gui?.webContentsId) &&
      gui.webContentsId === observed?.webContentsId,
    nativeAgentRouteBound:
      typeof observed?.viewId === "string" &&
      observed.viewId.length > 0 &&
      Number.isInteger(observed?.windowId) &&
      observed.windowId > 0 &&
      Number.isInteger(observed?.ownerWebContentsId) &&
      observed.ownerWebContentsId > 0,
    userNavigationRelinquishedAgentControl:
      latestAfterUserNavigation?.controlOwner === "user" &&
      latestAfterUserNavigation?.activeTurnId === null,
    userNavigationInvalidatedSnapshot:
      Number.isInteger(initial?.observation?.pageRevision) &&
      Number.isInteger(latestAfterUserNavigation?.pageRevision) &&
      latestAfterUserNavigation.pageRevision > initial.observation.pageRevision,
    staleSnapshotMutationRejected:
      staleMutationFailure === "stale_snapshot_rejected" &&
      observed?.staleMutationRejected === true,
    reclaimedSameTabAfterUserNavigation:
      recovered?.state?.browserSessionId === gui?.sessionId &&
      recovered?.state?.tabId === gui?.tabId &&
      recovered?.state?.threadId === identity?.threadId &&
      recovered?.state?.webContentsId === gui?.webContentsId &&
      recovered?.state?.activeTurnId === turnId &&
      recovered?.state?.controlOwner === "agent",
    reobservedFreshSnapshot:
      Boolean(initial?.observation?.snapshotId) &&
      Boolean(recovered?.observation?.snapshotId) &&
      initial.observation.snapshotId !== recovered.observation.snapshotId &&
      recovered.observation.pageRevision === recovered?.state?.pageRevision &&
      recovered.observation.pageRevision >
        latestAfterUserNavigation?.pageRevision,
    debuggerAttachedDuringRecoveredObserve:
      debuggerBeforeTerminal?.exists === true &&
      debuggerBeforeTerminal?.attached === true &&
      debuggerBeforeTerminal?.webContentsId === gui?.webContentsId,
    terminalReleasedUserTab:
      released?.controlOwner === "released" &&
      released?.activeTurnId === null &&
      released?.sessionId === gui?.sessionId &&
      released?.tabId === gui?.tabId &&
      released?.webContentsId === gui?.webContentsId,
    terminalDetachedDebugger:
      debuggerAfterTerminal?.exists === true &&
      debuggerAfterTerminal?.attached === false &&
      debuggerAfterTerminal?.webContentsId === gui?.webContentsId,
    destroyedWebContentsClosesRoute:
      destroyed?.tabId === null && destroyed?.webContentsId === null,
    dynamicBrowserToolsAdvertised: toolNames.includes("browser__openTabs"),
    dynamicBrowserToolReturned: dynamicToolCallCount >= 1,
    dynamicBrowserRoundTripVisible: providerBodies.length >= 8,
    finalAssistantVisible: String(finalText || "").includes(FINAL_MARKER),
    hostResolvedNotificationHidden: !String(finalText || "").includes(
      "serverRequest/resolved",
    ),
  };
}

function printHelp() {
  console.log(`
Browser Electron Gate B

用法:
  node scripts/agent-runtime/browser-runtime-electron-gate-b.mjs [options]

选项:
  --output <path>       证据 JSON 路径
  --scenario <name>     projection（Gate A）、lifecycle（默认）、approval、user-control、cancel、window-close、disconnect、permission 或 download
  --app-url <url>       可选 renderer dev server
  --timeout-ms <ms>     总超时，默认 180000
  --interval-ms <ms>    轮询间隔，默认 500
  --keep-temp           保留临时 fixture 目录
`);
}

export async function run(options) {
  ensureElectronFixtureBuild({
    appUrl: options.appUrl,
    logPrefix: LOG_PREFIX,
    rootDir: process.cwd(),
  });
  fs.mkdirSync(path.dirname(options.output), { recursive: true });
  const runtimeEnv = createTempRuntimeEnv();
  const providerFixture = await createBrowserProviderFixture({
    approvalScenario: ["approval", "user-control"].includes(options.scenario),
    userControlScenario: options.scenario === "user-control",
    connectionDiagnostics: [
      "cancel",
      "disconnect",
      "permission",
      "download",
    ].includes(options.scenario),
  });
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const requestLog = [];
  const consoleErrors = [];
  const pageErrors = [];
  let disconnectMainDiagnosticCount = 0;
  let app = null;
  let invokeTraceCollector = null;
  try {
    logStage("launch-electron");
    app = await electron.launch({
      executablePath: electronPath,
      args: ["--use-mock-keychain", "."],
      cwd: process.cwd(),
      env: {
        ...runtimeEnv.env,
        ...appServerEnv,
        APP_SERVER_BACKEND_MODE: "runtime",
        ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
        LIME_ELECTRON_E2E: "1",
        LIME_ELECTRON_BRAND_DEV_APP: "0",
        LIME_ELECTRON_CLEAR_RENDERER_CACHE: "1",
        LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
        ...(options.appUrl ? { VITE_DEV_SERVER_URL: options.appUrl } : {}),
      },
      timeout: options.timeoutMs,
    });
    app.on("console", (message) => {
      const messageText = sanitizeText(message.text());
      if (message.type() === "error") consoleErrors.push(messageText);
      if (
        options.scenario === "disconnect" &&
        messageText.includes("app-server") &&
        disconnectMainDiagnosticCount < 40
      ) {
        disconnectMainDiagnosticCount += 1;
        console.log(
          `${LOG_PREFIX} main-console type=${message.type()} text=${messageText}`,
        );
      }
    });
    const page = await app.firstWindow({ timeout: options.timeoutMs });
    page.on("pageerror", (error) =>
      pageErrors.push(sanitizeText(error.message)),
    );
    page.setDefaultTimeout(options.timeoutMs);
    await page.setViewportSize({ width: 1440, height: 1000 });
    await waitForRendererReady(page, options, () => undefined);

    logStage("initialize-app-server");
    await initializeAppServer(page, requestLog);
    const workspace = await ensureDefaultWorkspace(page, requestLog);
    const provider = await invokeAppServerFromPage(
      page,
      "modelProvider/create",
      {
        name: `Browser Gate B ${process.pid}`,
        providerType: "openai",
        apiHost: `${providerFixture.baseUrl}/v1`,
      },
      requestLog,
    );
    const providerId = String(provider.result?.provider?.id || "").trim();
    assert(providerId, "Browser fixture provider 创建失败");
    await invokeAppServerFromPage(
      page,
      "modelProvider/update",
      {
        providerId,
        enabled: true,
        sortOrder: 0,
        models: [],
      },
      requestLog,
    );
    await invokeAppServerFromPage(
      page,
      "modelProviderKey/create",
      {
        providerId,
        apiKey: PROVIDER_API_KEY,
        alias: "browser-runtime-gate-b",
        replaceExisting: true,
      },
      requestLog,
    );
    const fetched = await invokeAppServerFromPage(
      page,
      "modelProvider/fetchModels",
      { providerId },
      requestLog,
    );
    assert(
      Array.isArray(fetched.result?.models) &&
        fetched.result.models.some((model) => model?.id === MODEL_NAME),
      "Browser fixture provider 未返回可用模型",
    );
    await bindGuiWorkspaceAndModelPreferences(page, workspace.workspaceId, {
      provider: providerId,
      model: MODEL_NAME,
    });

    logStage("create-and-open-thread");
    const created = await createFixtureSession(page, workspace, requestLog, {
      provider: providerId,
      model: MODEL_NAME,
    });
    const identity = created.identity;
    const sessionOptions = {
      ...options,
      sessionId: identity.sessionId,
      threadId: identity.threadId,
    };
    await page.evaluate(
      ({ sessionId, workspaceId }) => {
        window.dispatchEvent(
          new CustomEvent("lime:agent-runtime-sessions-changed", {
            detail: { reason: "external", sessionId, workspaceId },
          }),
        );
      },
      { sessionId: identity.sessionId, workspaceId: workspace.workspaceId },
    );
    await navigateGuiToWorkspaceScopedAgent(
      page,
      sessionOptions,
      workspace.workspaceId,
    );
    await waitForGuiSessionVisible(page, sessionOptions);
    await openFixtureSessionFromSidebar(page, sessionOptions, requestLog);

    logStage("request-and-mount-browser");
    invokeTraceCollector = startInvokeTraceEvidenceCollector(page, {
      intervalMs: options.intervalMs,
    });
    await invokeAppServerFromPage(
      page,
      "workspaceRightSurface/request",
      {
        workspaceId: workspace.workspaceId,
        workspaceRoot: workspace.rootPath,
        sessionId: identity.sessionId,
        surfaceKind: "browser",
        origin: "runtime",
        priority: "foreground",
        reason: "browser_runtime_gate_b",
        candidateId: "browser-runtime-gate-b",
        ttlMs: 120_000,
        metadata: {
          browser: {
            launchUrl: BROWSER_URL,
            title: "Browser Runtime Gate B",
          },
        },
      },
      requestLog,
    );
    const guiBeforeTurn = await waitForBrowserSurface(page, options);
    if (options.scenario === "projection") {
      logStage("run-browser-projection-gate-a");
      const gateAEvidence = await runBrowserProjectionGateA({
        guiBeforeTurn,
        identity,
        options,
        page,
        readBrowserWorkspaceState,
      });
      gateAEvidence.diagnostics = {
        consoleErrors,
        pageErrors,
      };
      fs.writeFileSync(
        options.output,
        `${JSON.stringify(gateAEvidence, null, 2)}\n`,
      );
      if (
        gateAEvidence.failedAssertions.length > 0 ||
        consoleErrors.length > 0 ||
        pageErrors.length > 0
      ) {
        throw new Error(
          `Browser Gate A 断言失败: ${[
            ...gateAEvidence.failedAssertions,
            ...(consoleErrors.length > 0 ? ["consoleErrors"] : []),
            ...(pageErrors.length > 0 ? ["pageErrors"] : []),
          ].join(", ")}`,
        );
      }
      return gateAEvidence;
    }
    if (["approval", "user-control"].includes(options.scenario)) {
      await installBrowserApprovalPage(app, guiBeforeTurn.webContentsId);
    }

    logStage("run-browser-dynamic-tools");
    await sendPromptFromGui(
      page,
      sessionOptions,
      "请使用当前 Browser 打开标签并观察页面，然后返回结果。",
      { expectedSessionId: identity.sessionId },
    );
    logStage("wait-initial-agent-observation");
    const initial = await waitForScenarioValue(
      options,
      () => providerFixture.scenario.initial,
      "Agent 初次 observe",
    );
    const activeTurnId = initial.state?.activeTurnId || null;
    assert(activeTurnId, "初次 observe 未绑定 active turn");
    const agentControlled = await waitForBrowserWorkspaceState(
      page,
      options,
      (state) =>
        state.controlOwner === "agent" &&
        state.activeTurnId === activeTurnId &&
        state.webContentsId === initial.state.webContentsId,
      "Browser Workspace 未投影 Agent 控制态",
    );

    if (options.scenario === "approval") {
      const approvalEvidence = await runBrowserApprovalScenario({
        activeTurnId,
        agentControlled,
        app,
        consoleErrors,
        finalMarker: FINAL_MARKER,
        guiBeforeTurn,
        identity,
        initial,
        logStage,
        options,
        page,
        pageErrors,
        providerFixture,
        readBrowserDebuggerState,
        readBrowserWorkspaceState,
        requestLog,
        waitForBrowserWorkspaceState,
        waitForScenarioValue,
        waitForTerminalThread,
      });
      fs.writeFileSync(
        options.output,
        `${JSON.stringify(approvalEvidence, null, 2)}\n`,
      );
      if (approvalEvidence.failedAssertions.length > 0) {
        throw new Error(
          `Browser approval Gate B 断言失败: ${approvalEvidence.failedAssertions.join(", ")}`,
        );
      }
      return approvalEvidence;
    }

    if (options.scenario === "user-control") {
      const userControlEvidence = await runBrowserUserControlScenario({
        activeTurnId,
        agentControlled,
        app,
        consoleErrors,
        finalMarker: FINAL_MARKER,
        guiBeforeTurn,
        identity,
        initial,
        logStage,
        options,
        page,
        pageErrors,
        providerFixture,
        readBrowserDebuggerState,
        readBrowserWorkspaceState,
        requestLog,
        waitForBrowserWorkspaceState,
        waitForScenarioValue,
        waitForTerminalThread,
      });
      fs.writeFileSync(
        options.output,
        `${JSON.stringify(userControlEvidence, null, 2)}\n`,
      );
      if (userControlEvidence.failedAssertions.length > 0) {
        throw new Error(
          `Browser user-control Gate B 断言失败: ${userControlEvidence.failedAssertions.join(", ")}`,
        );
      }
      return userControlEvidence;
    }

    if (options.scenario === "cancel") {
      const cancelEvidence = await runBrowserCancelScenario({
        activeTurnId,
        agentControlled,
        app,
        consoleErrors,
        finalMarker: FINAL_MARKER,
        guiBeforeTurn,
        identity,
        initial,
        logStage,
        options,
        page,
        pageErrors,
        providerFixture,
        readBrowserDebuggerState,
        readBrowserWorkspaceState,
        requestLog,
        waitForBrowserWorkspaceState,
        waitForTerminalThread,
      });
      fs.writeFileSync(
        options.output,
        `${JSON.stringify(cancelEvidence, null, 2)}\n`,
      );
      if (cancelEvidence.failedAssertions.length > 0) {
        throw new Error(
          `Browser cancel Gate B 断言失败: ${cancelEvidence.failedAssertions.join(", ")}`,
        );
      }
      return cancelEvidence;
    }

    if (options.scenario === "window-close") {
      const windowCloseEvidence = await runBrowserWindowCloseScenario({
        agentControlled,
        app,
        consoleErrors,
        guiBeforeTurn,
        identity,
        initial,
        options,
        page,
        pageErrors,
        providerFixture,
        readBrowserDebuggerState,
      });
      fs.writeFileSync(
        options.output,
        `${JSON.stringify(windowCloseEvidence, null, 2)}\n`,
      );
      if (windowCloseEvidence.failedAssertions.length > 0) {
        throw new Error(
          `Browser window-close Gate B 断言失败: ${windowCloseEvidence.failedAssertions.join(", ")}`,
        );
      }
      return windowCloseEvidence;
    }

    if (options.scenario === "disconnect") {
      const disconnectEvidence = await runBrowserDisconnectScenario({
        activeTurnId,
        agentControlled,
        app,
        consoleErrors,
        finalMarker: FINAL_MARKER,
        guiBeforeTurn,
        identity,
        initial,
        logStage,
        options,
        page,
        pageErrors,
        providerFixture,
        readBrowserDebuggerState,
        readBrowserWorkspaceState,
        waitForBrowserWorkspaceState,
      });
      fs.writeFileSync(
        options.output,
        `${JSON.stringify(disconnectEvidence, null, 2)}\n`,
      );
      if (disconnectEvidence.failedAssertions.length > 0) {
        throw new Error(
          `Browser disconnect Gate B 断言失败: ${disconnectEvidence.failedAssertions.join(", ")}`,
        );
      }
      return disconnectEvidence;
    }

    if (options.scenario === "permission") {
      const permissionEvidence = await runBrowserPermissionScenario({
        activeTurnId,
        agentControlled,
        app,
        consoleErrors,
        finalMarker: FINAL_MARKER,
        guiBeforeTurn,
        identity,
        initial,
        logStage,
        options,
        page,
        pageErrors,
        providerFixture,
        readBrowserDebuggerState,
        readBrowserWorkspaceState,
        requestLog,
        waitForBrowserWorkspaceState,
        waitForTerminalThread,
      });
      fs.writeFileSync(
        options.output,
        `${JSON.stringify(permissionEvidence, null, 2)}\n`,
      );
      if (permissionEvidence.failedAssertions.length > 0) {
        throw new Error(
          `Browser permission Gate B 断言失败: ${permissionEvidence.failedAssertions.join(", ")}`,
        );
      }
      return permissionEvidence;
    }

    if (options.scenario === "download") {
      const downloadEvidence = await runBrowserDownloadScenario({
        activeTurnId,
        agentControlled,
        app,
        consoleErrors,
        finalMarker: FINAL_MARKER,
        guiBeforeTurn,
        identity,
        initial,
        logStage,
        options,
        page,
        pageErrors,
        providerFixture,
        readBrowserDebuggerState,
        readBrowserWorkspaceState,
        requestLog,
        waitForBrowserWorkspaceState,
        waitForTerminalThread,
      });
      fs.writeFileSync(
        options.output,
        `${JSON.stringify(downloadEvidence, null, 2)}\n`,
      );
      if (downloadEvidence.failedAssertions.length > 0) {
        throw new Error(
          `Browser download Gate B 断言失败: ${downloadEvidence.failedAssertions.join(", ")}`,
        );
      }
      return downloadEvidence;
    }

    logStage("user-navigation-during-active-turn");
    const userNavigation = await navigateVisibleBrowserTab(
      page,
      options,
      agentControlled.tabId,
    );
    const latestAfterUserNavigation = await waitForBrowserWorkspaceState(
      page,
      options,
      (state) =>
        state.controlOwner === "user" &&
        state.activeTurnId === null &&
        state.webContentsId === initial.state.webContentsId &&
        state.pageRevision > initial.observation.pageRevision &&
        state.address === USER_NAVIGATION_URL,
      "用户导航后 Browser Workspace 未释放 Agent 控制",
    );
    providerFixture.scenario.releaseUserNavigation();

    logStage("reclaim-reject-stale-and-reobserve");
    const recovered = await waitForScenarioValue(
      options,
      () => providerFixture.scenario.recovered,
      "Agent 重新 claim/observe",
    );
    let reclaimed;
    try {
      reclaimed = await waitForBrowserWorkspaceState(
        page,
        options,
        (state) =>
          isReclaimedBrowserWorkspaceState({
            activeTurnId,
            initialWebContentsId: initial.state.webContentsId,
            recovered,
            state,
          }),
        "Browser Workspace 未投影重新 claim 状态",
      );
    } catch (error) {
      throw new Error(
        `${error instanceof Error ? error.message : String(error)}; ` +
          `scenario=${JSON.stringify(
            sanitizeJson({
              initial: providerFixture.scenario.initial,
              latestAfterUserNavigation:
                providerFixture.scenario.latestAfterUserNavigation,
              recovered: providerFixture.scenario.recovered,
              staleMutationFailure:
                providerFixture.scenario.staleMutationFailure,
            }),
          )}; ` +
          `providerRequests=${JSON.stringify(
            providerFixture.requests.map((request) => ({
              path: request.path,
              responseError: request.responseError || null,
              lastMessage: request.body?.messages?.at(-1) || null,
            })),
          )}`,
      );
    }
    const debuggerBeforeTerminal = await readBrowserDebuggerState(
      app,
      reclaimed.webContentsId,
    );
    providerFixture.scenario.releaseFinalResponse();

    let terminal;
    try {
      terminal = await waitForTerminalThread(
        page,
        options,
        identity.threadId,
        requestLog,
      );
    } catch (error) {
      const lastRead = requestLog
        .filter((entry) => entry?.method === APP_SERVER_METHOD_SESSION_READ)
        .at(-1);
      throw new Error(
        `${error instanceof Error ? error.message : String(error)}; ` +
          `providerRequests=${JSON.stringify(
            providerFixture.requests.map((request) => ({
              path: request.path,
              responseError: request.responseError || null,
              lastMessage: request.body?.messages?.at(-1) || null,
            })),
          )}; ` +
          `lastRead=${JSON.stringify(sanitizeJson(lastRead?.response || null))}`,
      );
    }
    const terminalTurnId = terminal.turn?.id || terminal.turn?.turnId || null;
    assert(
      terminalTurnId === activeTurnId,
      "Browser Gate B terminal turn 与 active turn 不一致",
    );
    const released = await waitForBrowserWorkspaceState(
      page,
      options,
      (state) =>
        state.controlOwner === "released" &&
        state.activeTurnId === null &&
        state.webContentsId === initial.state.webContentsId,
      "turn terminal 后 Browser 用户 tab 未 release",
    );
    const debuggerAfterTerminal = await readBrowserDebuggerState(
      app,
      released.webContentsId,
    );
    const guiShellAfterTurn = await page.evaluate(() => ({
      bodyText: document.body?.innerText || "",
      panelVisible: Boolean(
        document.querySelector('[data-testid="right-surface-browser-panel"]'),
      ),
      activeSurface:
        document
          .querySelector('[data-testid="workspace-right-surface-host"]')
          ?.getAttribute("data-surface") || null,
    }));
    const guiAfterTurn = {
      ...(await readBrowserWorkspaceState(page)),
      ...guiShellAfterTurn,
    };
    const finalText = guiAfterTurn.bodyText;
    const { bodyText: _bodyText, ...guiAfterTurnEvidence } = guiAfterTurn;
    const collectedInvokeTrace = await invokeTraceCollector.stop();
    invokeTraceCollector = null;
    const traceBeforeDestroyRaw = await page.evaluate(() => {
      try {
        return JSON.parse(
          window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
        );
      } catch {
        return [];
      }
    });
    const traceBeforeDestroy = mergeInvokeTraceEvidence(
      collectedInvokeTrace,
      traceBeforeDestroyRaw,
    );

    logStage("destroy-native-browser-webcontents");
    const destruction = await destroyBrowserWebContents(
      app,
      released.webContentsId,
    );
    const guiAfterDestroyed = await waitForBrowserWorkspaceState(
      page,
      options,
      (state) => state.tabId === null && state.webContentsId === null,
      "native WebContents destroyed 后 Browser route 未关闭",
    );

    const traceAfterTurn = traceBeforeDestroy;
    const assertions = buildAssertions({
      debuggerAfterTerminal,
      debuggerBeforeTerminal,
      destroyed: guiAfterDestroyed,
      gui: guiAfterTurn,
      initial,
      latestAfterUserNavigation,
      recovered,
      released,
      staleMutationFailure: providerFixture.scenario.staleMutationFailure,
      trace: traceAfterTurn,
      providerRequests: providerFixture.requests,
      identity,
      turnId: terminalTurnId,
      finalText,
    });
    const failedAssertions = Object.entries(assertions)
      .filter(([, passed]) => !passed)
      .map(([name]) => name);
    const evidence = sanitizeJson({
      schemaVersion: "lime.browser_runtime_electron_gate_b.v1",
      status: failedAssertions.length === 0 ? "pass" : "fail",
      generatedAt: new Date().toISOString(),
      proofLevel: "Gate B",
      claimBoundary:
        "真实 Electron WebContentsView、preload IPC、App Server JSON-RPC、dynamic browser tool 和 current read model 的本地闭环；不代表 live provider 或跨平台打包验证。",
      identity,
      gui: {
        beforeTurn: guiBeforeTurn,
        afterTurn: {
          ...guiAfterTurnEvidence,
          finalAssistantVisible: finalText.includes(FINAL_MARKER),
        },
        afterDestroyed: guiAfterDestroyed,
      },
      provider: {
        model: MODEL_NAME,
        requestCount: providerFixture.requests.length,
        toolCalls: providerFixture.requests
          .filter((request) => request?.path === "/v1/chat/completions")
          .map(
            (request) => request?.body?.messages?.at(-1)?.tool_calls || null,
          ),
      },
      terminal: {
        status: terminal.turn?.status || terminal.turn?.state || null,
        turnId: terminalTurnId,
      },
      observedBrowserState: extractFinalBrowserState(finalText),
      lifecycle: {
        initial: {
          activeTurnId: initial.state.activeTurnId,
          pageRevision: initial.observation.pageRevision,
          tabId: initial.state.tabId,
          webContentsId: initial.state.webContentsId,
        },
        userNavigation: {
          ...userNavigation.navigation,
          activeTurnId: latestAfterUserNavigation.activeTurnId,
          controlOwner: latestAfterUserNavigation.controlOwner,
          pageRevision: latestAfterUserNavigation.pageRevision,
          tabId: latestAfterUserNavigation.tabId,
          webContentsId: latestAfterUserNavigation.webContentsId,
        },
        staleMutation: providerFixture.scenario.staleMutationFailure,
        recovered: {
          activeTurnId: recovered.state.activeTurnId,
          controlOwner: recovered.state.controlOwner,
          pageRevision: recovered.observation.pageRevision,
          tabId: recovered.state.tabId,
          webContentsId: recovered.state.webContentsId,
        },
        terminal: {
          controlOwner: released.controlOwner,
          debuggerAttachedBefore: debuggerBeforeTerminal.attached,
          debuggerAttachedAfter: debuggerAfterTerminal.attached,
          tabId: released.tabId,
          webContentsId: released.webContentsId,
        },
        destroyed: {
          browserTabId: guiAfterDestroyed.tabId,
          webContentsId: guiAfterDestroyed.webContentsId,
          panelVisible: guiAfterDestroyed.panelVisible,
          trigger: destruction.trigger || null,
        },
      },
      invokeTrace: summarizeInvokeTrace(traceAfterTurn),
      assertions,
      failedAssertions,
      diagnostics: { consoleErrors, pageErrors },
    });
    fs.writeFileSync(options.output, `${JSON.stringify(evidence, null, 2)}\n`);
    if (failedAssertions.length > 0) {
      throw new Error(
        `Browser Gate B 断言失败: ${failedAssertions.join(", ")}`,
      );
    }
    return evidence;
  } finally {
    if (invokeTraceCollector) {
      await invokeTraceCollector.stop().catch(() => undefined);
    }
    providerFixture.scenario.releaseAll();
    await closeFixtureElectron(app);
    await providerFixture.close().catch(() => undefined);
    if (!options.keepTemp) cleanupTempRoot(runtimeEnv.tempRoot);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    process.exit(0);
  }
  run(options)
    .then(() => {
      console.log(`${LOG_PREFIX} result=pass output=${options.output}`);
    })
    .catch((error) => {
      console.error(`${LOG_PREFIX} result=fail error=${sanitizeText(error)}`);
      process.exitCode = 1;
    });
}
