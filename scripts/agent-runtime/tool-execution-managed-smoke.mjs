#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import electronPath from "electron";
import { _electron as electron } from "playwright";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import {
  AGENT_CONTROL_CAPACITY_FINAL_TEXT,
  AGENT_CONTROL_CAPACITY_GATE_B_BATCH_ID,
  AGENT_CONTROL_FINAL_TEXT,
  AGENT_CONTROL_RESIDENCY_FINAL_TEXT,
  AGENT_CONTROL_RESIDENCY_GATE_B_BATCH_ID,
  PARENT_OWNED_PLACEHOLDERS,
  AGENT_CONTROL_VISIBLE_DOM_GATE_B_BATCH_ID,
  buildAgentControlVisibleDomAssertions,
  buildAgentControlCapacityVisibleDomAssertions,
  buildAgentControlResidencyVisibleDomAssertions,
} from "./agent-control-visible-dom-gate-b.mjs";
import {
  ROLLOUT_BUDGET_FINAL_TEXT,
  ROLLOUT_BUDGET_GATE_B_BATCH_ID,
  buildRolloutBudgetVisibleDomAssertions,
} from "./rollout-budget-visible-dom-gate-b.mjs";
import {
  buildDeferredMcpVisibleDomAssertions,
  DEFERRED_MCP_TOOL_SEARCH_FINAL_TEXT,
  DEFERRED_MCP_TOOL_SEARCH_GATE_B_BATCH_ID,
} from "./deferred-mcp-tool-search-gate-b.mjs";
import {
  TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_FINAL_TEXT,
  TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_GATE_B_BATCH_ID,
  buildToolOrchestratorManagedNetworkRetryVisibleDomAssertions,
  TOOL_ORCHESTRATOR_SANDBOX_RETRY_FINAL_TEXT,
  TOOL_ORCHESTRATOR_SANDBOX_RETRY_GATE_B_BATCH_ID,
  buildToolOrchestratorSandboxRetryVisibleDomAssertions,
} from "./tool-orchestrator-visible-dom-gate-b.mjs";
import {
  buildSoakSummary,
  childArgsForRound,
  collectProcessTreeSnapshot,
  collectRestoredSoakRounds,
  collectSoakRoundObservation,
  resolveSoakConfig,
  roundEvidencePath,
  waitForProcessIdsExit,
} from "./tool-execution-soak-evidence.mjs";
import { runManagedColdRestarts } from "./tool-execution-managed-restart.mjs";
import {
  buildToolExecutionTurnStartParams,
  normalizeToolExecutionThreadReadResponse,
} from "./tool-execution-current-contract.mjs";
import {
  cleanupToolExecutionTempRoot,
  createToolExecutionTempRuntimeEnv,
} from "./tool-execution-managed-runtime-env.mjs";
import {
  readToolExecutionEvidence,
  resolveToolExecutionEvidencePath,
  screenshotPathForEvidence,
  writeToolExecutionEvidence,
} from "./tool-execution-managed-evidence.mjs";

const LOG_PREFIX = "[smoke:agent-runtime-tool-execution:managed]";
const DEFAULT_TIMEOUT_MS = 300_000;
const INTERVAL_MS = 500;
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const DEFAULT_EVIDENCE_OUTPUT = path.resolve(
  ".lime/qc/agent-runtime-tool-execution-smoke.json",
);
const NAVIGATION_RESTORE_STORAGE_KEY = "lime.appNavigation.restore.v1";
const INVOKE_TRACE_STORAGE_KEY = "lime_invoke_trace_buffer_v1";
const INVOKE_ERROR_STORAGE_KEY = "lime_invoke_error_buffer_v1";

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sanitizeText(value) {
  return String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|session|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]")
    .replace(/sk-[A-Za-z0-9._-]+/g, "sk-[redacted]");
}

function timeoutFromArgs(args) {
  const index = args.indexOf("--timeout-ms");
  if (index >= 0 && args[index + 1]) {
    const value = Number(args[index + 1]);
    if (Number.isFinite(value) && value >= 30_000) {
      return value;
    }
  }
  return DEFAULT_TIMEOUT_MS;
}

function valueFromArgs(args, name) {
  const index = args.indexOf(name);
  return index >= 0 && args[index + 1] ? String(args[index + 1]) : null;
}

function visibleDomGateBKindFromArgs(args) {
  const batchId = valueFromArgs(args, "--batch");
  if (batchId === AGENT_CONTROL_VISIBLE_DOM_GATE_B_BATCH_ID) {
    return "agent-control";
  }
  if (batchId === AGENT_CONTROL_CAPACITY_GATE_B_BATCH_ID) {
    return "agent-capacity";
  }
  if (batchId === AGENT_CONTROL_RESIDENCY_GATE_B_BATCH_ID) {
    return "agent-residency";
  }
  if (batchId === ROLLOUT_BUDGET_GATE_B_BATCH_ID) {
    return "rollout-budget";
  }
  if (batchId === DEFERRED_MCP_TOOL_SEARCH_GATE_B_BATCH_ID) {
    return "deferred-mcp";
  }
  if (batchId === TOOL_ORCHESTRATOR_SANDBOX_RETRY_GATE_B_BATCH_ID) {
    return "sandbox-retry";
  }
  if (batchId === TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_GATE_B_BATCH_ID) {
    return "managed-network-retry";
  }
  return null;
}

async function waitForRendererReady(page, timeoutMs) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const snapshot = await page.evaluate(
        (command) => ({
          url: window.location.href,
          title: document.title || "",
          electron: window.__LIME_ELECTRON__ === true,
          hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
          supportsAppServer:
            typeof window.electronAPI?.supportsCommand === "function" &&
            window.electronAPI.supportsCommand(command),
          startupVisible: Boolean(
            document.querySelector("[data-lime-startup-shell]"),
          ),
          appSidebarVisible: Boolean(
            document.querySelector('[data-testid="app-sidebar"]'),
          ),
          bodyText: document.body?.innerText || "",
        }),
        APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      );
      lastSnapshot = snapshot;
      if (
        snapshot.electron &&
        snapshot.hasInvokeBridge &&
        snapshot.supportsAppServer &&
        !snapshot.startupVisible &&
        snapshot.appSidebarVisible
      ) {
        return snapshot;
      }
    } catch (error) {
      lastSnapshot = { error: sanitizeText(error) };
    }
    await sleep(INTERVAL_MS);
  }
  throw new Error(
    `Electron renderer / App Server bridge 未就绪: ${JSON.stringify(lastSnapshot)}`,
  );
}

async function launchManagedElectron({
  appServerEnv,
  consoleErrors,
  runtimeEnv,
  timeoutMs,
}) {
  const app = await electron.launch({
    executablePath: electronPath,
    args: ["--use-mock-keychain", "."],
    cwd: process.cwd(),
    env: {
      ...runtimeEnv.env,
      ...appServerEnv,
      ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
      LIME_ELECTRON_E2E: "1",
      LIME_ELECTRON_BRAND_DEV_APP: "0",
      LIME_ELECTRON_CLEAR_RENDERER_CACHE: "0",
      LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
    },
    timeout: timeoutMs,
  });
  const page = await app.firstWindow({ timeout: timeoutMs });
  page.setDefaultTimeout(timeoutMs);
  await page.setViewportSize({ width: 1440, height: 1000 });
  page.on("console", (message) => {
    if (
      message.text().includes("WorkspaceSubagentNavigation") ||
      message.text().includes("useAgentSession.switchTopic")
    ) {
      console.log(`${LOG_PREFIX} renderer-debug=${message.text()}`);
    }
    if (message.type() === "error") {
      const sourceUrl = String(message.location()?.url || "").trim();
      consoleErrors.push(
        sanitizeText(
          sourceUrl ? `${message.text()} source=${sourceUrl}` : message.text(),
        ).slice(0, 700),
      );
    }
  });
  const rendererSnapshot = await waitForRendererReady(page, timeoutMs);
  return { app, page, rendererSnapshot };
}

function readJsonBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    request.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf8").trim();
        resolve(raw ? JSON.parse(raw) : {});
      } catch (error) {
        reject(error);
      }
    });
    request.on("error", reject);
  });
}

function writeJson(response, status, payload) {
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "access-control-allow-origin": "*",
    "access-control-allow-headers": "content-type",
    "access-control-allow-methods": "GET,POST,OPTIONS",
  });
  response.end(`${JSON.stringify(payload)}\n`);
}

async function readInvokeDiagnostics(page) {
  return await page.evaluate(
    ({ errorKey, traceKey }) => {
      const readArray = (key) => {
        try {
          const parsed = JSON.parse(localStorage.getItem(key) || "[]");
          return Array.isArray(parsed) ? parsed : [];
        } catch {
          return [];
        }
      };
      const calls = [];
      for (const entry of readArray(traceKey)) {
        if (entry?.command !== "app_server_handle_json_lines") continue;
        const lines = Array.isArray(entry?.args_preview?.request?.lines)
          ? entry.args_preview.request.lines
          : [];
        for (const line of lines) {
          try {
            const message = typeof line === "string" ? JSON.parse(line) : line;
            if (typeof message?.method !== "string") continue;
            calls.push({
              method: message.method,
              threadId: String(
                message?.params?.threadId ?? message?.params?.thread_id ?? "",
              ),
              transport: String(entry?.transport || ""),
              status: String(entry?.status || ""),
            });
          } catch {
            // Ignore malformed diagnostic previews; the product request already failed elsewhere.
          }
        }
      }
      return {
        appServerCalls: calls,
        invokeErrorCount: readArray(errorKey).length,
      };
    },
    {
      errorKey: INVOKE_ERROR_STORAGE_KEY,
      traceKey: INVOKE_TRACE_STORAGE_KEY,
    },
  );
}

async function restoreAgentSessionRoute(page, sessionId, timeoutMs) {
  await page.evaluate(
    ({ errorKey, navigationKey, sessionId, traceKey }) => {
      localStorage.removeItem(errorKey);
      localStorage.removeItem(traceKey);
      localStorage.setItem("lime:agent-debug", "1");
      sessionStorage.setItem(
        navigationKey,
        JSON.stringify({
          page: "agent",
          params: { initialSessionId: sessionId },
        }),
      );
    },
    {
      errorKey: INVOKE_ERROR_STORAGE_KEY,
      navigationKey: NAVIGATION_RESTORE_STORAGE_KEY,
      sessionId,
      traceKey: INVOKE_TRACE_STORAGE_KEY,
    },
  );
  await page.reload({ waitUntil: "domcontentloaded", timeout: timeoutMs });
  return await waitForRendererReady(page, timeoutMs);
}

async function waitForDomCountToDrop(page, selector, previousCount, timeoutMs) {
  await page.waitForFunction(
    ({ previousCount, selector }) =>
      document.querySelectorAll(selector).length < previousCount,
    { previousCount, selector },
    { timeout: Math.min(timeoutMs, 30_000) },
  );
}

async function materializeHistoricalTimelines(page, timeoutMs) {
  const historicalPreviewSelector =
    '[data-testid^="message-list-historical-timeline-preview:"]';
  const materializedTimelineSelector = '[data-testid="agent-thread-flow"]';
  const historicalPreviews = page.locator(historicalPreviewSelector);
  const materializedTimelines = page.locator(materializedTimelineSelector);
  for (let attempt = 0; attempt < 10; attempt += 1) {
    const previousCount = await historicalPreviews.count();
    if (previousCount === 0) break;
    const previousTimelineCount = await materializedTimelines.count();
    await historicalPreviews.first().click();
    await page.waitForFunction(
      ({
        historicalPreviewSelector,
        materializedTimelineSelector,
        previousCount,
        previousTimelineCount,
      }) =>
        document.querySelectorAll(historicalPreviewSelector).length <
          previousCount ||
        document.querySelectorAll(materializedTimelineSelector).length >
          previousTimelineCount,
      {
        historicalPreviewSelector,
        materializedTimelineSelector,
        previousCount,
        previousTimelineCount,
      },
      { timeout: Math.min(timeoutMs, 30_000) },
    );
  }
}

async function expandHistoricalToolRows(page, timeoutMs) {
  await materializeHistoricalTimelines(page, timeoutMs);

  const closedProcessSelector =
    'details[data-testid*="agent-thread-block:"][data-testid$=":process"]:not([open])';
  await page.waitForFunction(
    ({ closedProcessSelector }) =>
      document.querySelector('[data-testid="tool-call-row"]') !== null ||
      document.querySelector(closedProcessSelector) !== null,
    { closedProcessSelector },
    { timeout: Math.min(timeoutMs, 30_000) },
  );

  const closedProcessBlocks = page.locator(closedProcessSelector);
  for (let attempt = 0; attempt < 10; attempt += 1) {
    const previousCount = await closedProcessBlocks.count();
    if (previousCount === 0) break;
    await closedProcessBlocks.first().locator("summary").click();
    await waitForDomCountToDrop(
      page,
      closedProcessSelector,
      previousCount,
      timeoutMs,
    );
  }

  const closedSubagentSelector =
    'details[data-testid*="agent-thread-block:"][data-testid$=":subagent"]:not([open])';
  const closedSubagentBlocks = page.locator(closedSubagentSelector);
  for (let attempt = 0; attempt < 10; attempt += 1) {
    const previousCount = await closedSubagentBlocks.count();
    if (previousCount === 0) break;
    await closedSubagentBlocks.first().locator("summary").click();
    await waitForDomCountToDrop(
      page,
      closedSubagentSelector,
      previousCount,
      timeoutMs,
    );
  }
}

async function findTypedToolRow(page, toolName, timeoutMs) {
  const handle = await page.waitForFunction(
    (expectedToolName) =>
      Array.from(
        document.querySelectorAll('[data-testid="tool-call-row"]'),
      ).find(
        (node) => node.getAttribute("data-tool-name") === expectedToolName,
      ) || null,
    toolName,
    { timeout: Math.min(timeoutMs, 30_000) },
  );
  const row = handle.asElement();
  if (!row) {
    await handle.dispose();
    throw new Error(`目标会话缺少 typed Tool row: ${toolName}`);
  }
  await row.waitForElementState("visible", { timeout: timeoutMs });
  return row;
}

async function listTypedToolRows(page) {
  return await page
    .locator('[data-testid="tool-call-row"]')
    .evaluateAll((nodes) =>
      nodes.map((node) => ({
        id: node.getAttribute("data-tool-call-id"),
        name: node.getAttribute("data-tool-name"),
        status: node.getAttribute("data-tool-status"),
        text: (node.textContent || "").trim().slice(0, 500),
        visible:
          window.getComputedStyle(node).display !== "none" &&
          window.getComputedStyle(node).visibility !== "hidden" &&
          node.getBoundingClientRect().height > 0,
      })),
    );
}

async function listFileChangeGroups(page) {
  return await page
    .locator('[data-testid="timeline-file-artifact-group"]')
    .evaluateAll((nodes) =>
      nodes.map((node) => {
        const card = node.querySelector(
          '[data-testid="file-changes-summary-card"]',
        );
        return {
          status: card?.getAttribute("data-file-status") || null,
          fileRowCount: node.querySelectorAll(
            '[data-testid="file-changes-summary-file-row"]',
          ).length,
          visible:
            window.getComputedStyle(node).display !== "none" &&
            window.getComputedStyle(node).visibility !== "hidden" &&
            node.getBoundingClientRect().height > 0,
          text: (node.textContent || "").trim().slice(0, 500),
        };
      }),
    );
}

async function listSubagentActivityRows(page) {
  return await page
    .locator('[data-testid="subagent-activity-row"]')
    .evaluateAll((nodes) =>
      nodes.map((node) => ({
        itemId: node.getAttribute("data-subagent-activity-item-id"),
        activityKind: node.getAttribute("data-subagent-activity-kind"),
        threadId: node.getAttribute("data-subagent-thread-id"),
        visible:
          window.getComputedStyle(node).display !== "none" &&
          window.getComputedStyle(node).visibility !== "hidden" &&
          node.getBoundingClientRect().height > 0,
        text: (node.textContent || "").trim().slice(0, 240),
      })),
    );
}

async function snapshotToolRow(row) {
  return await row.evaluate((node) => {
    const style = window.getComputedStyle(node);
    const toolName = node.getAttribute("data-tool-name");
    const toolStatus = node.getAttribute("data-tool-status");
    return {
      visible:
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        node.getBoundingClientRect().height > 0,
      completed: toolStatus === "completed",
      toolName,
      toolStatus,
      text: (node.textContent || "").trim().slice(0, 240),
    };
  });
}

async function readAgentControlDomState({
  expectedFinalText = AGENT_CONTROL_FINAL_TEXT,
  page,
  sessionId,
  timeoutMs,
}) {
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: timeoutMs });
  const finalText = page.getByText(expectedFinalText, { exact: false });
  await finalText.first().waitFor({ state: "visible", timeout: timeoutMs });

  await expandHistoricalToolRows(page, timeoutMs);
  await page.waitForFunction(
    () =>
      document.querySelector('[data-testid="subagent-activity-row"]') !== null,
    undefined,
    { timeout: Math.min(timeoutMs, 30_000) },
  );
  return {
    activeSessionId: await input.getAttribute("data-session-id"),
    typedToolRows: await listTypedToolRows(page),
    subagentActivityRows: await listSubagentActivityRows(page),
    finalAssistantTextVisible: await finalText.first().isVisible(),
  };
}

async function readRolloutBudgetDomState({ page, sessionId, timeoutMs }) {
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: timeoutMs });
  const reply = await invokeAppServerJsonRpcRaw(page, "thread/read", {
    threadId: sessionId,
    includeTurns: true,
  });
  const thread = reply?.result?.thread ?? null;
  if (!thread) {
    throw new Error(
      `预算 Gate B thread/read 未返回 canonical thread: ${JSON.stringify(reply?.error ?? null)}`,
    );
  }
  const latestTurn = Array.isArray(thread.turns) ? thread.turns.at(-1) : null;
  const bodyText = await page.evaluate(() =>
    (document.body.textContent || "").trim().slice(-2_000),
  );
  const diagnostics = await readInvokeDiagnostics(page);
  return {
    activeSessionId: await input.getAttribute("data-session-id"),
    typedToolRows: await listTypedToolRows(page),
    subagentActivityRows: await listSubagentActivityRows(page),
    finalAssistantTextVisible: bodyText.includes(ROLLOUT_BUDGET_FINAL_TEXT),
    failureVisible:
      bodyText.includes("rollout budget") ||
      bodyText.includes("usage_limit_exceeded") ||
      bodyText.includes("回合失败") ||
      bodyText.includes("Turn failed"),
    thread,
    latestTurnError: latestTurn?.error ?? null,
    appServerCalls: diagnostics.appServerCalls,
    invokeErrorCount: diagnostics.invokeErrorCount,
  };
}

function turnStartCountForThread(diagnostics, threadId) {
  return (diagnostics?.appServerCalls ?? []).filter(
    (call) => call?.method === "turn/start" && call?.threadId === threadId,
  ).length;
}

async function openSubagentActivityThread(page, childThreadId, timeoutMs) {
  const rows = page.locator('[data-testid="subagent-activity-row"]');
  for (let index = 0; index < (await rows.count()); index += 1) {
    const row = rows.nth(index);
    if ((await row.getAttribute("data-subagent-thread-id")) !== childThreadId) {
      continue;
    }
    const button = row.locator("button").last();
    if ((await button.count()) === 0) {
      continue;
    }
    console.log(
      `${LOG_PREFIX} open-subagent-button=${JSON.stringify({
        text: (await button.textContent())?.trim() || "",
        threadId: childThreadId,
      })}`,
    );
    await button.click({ timeout: timeoutMs });
    return;
  }
  throw new Error(`未找到可打开的 canonical child Thread: ${childThreadId}`);
}

async function collectParentOwnedChildGateB({
  outputPath,
  page,
  parentSessionId,
  subagentActivityRows,
  timeoutMs,
}) {
  const childThreadIds = [
    ...new Set(
      subagentActivityRows
        .map((row) => String(row?.threadId || "").trim())
        .filter(Boolean),
    ),
  ];
  if (childThreadIds.length !== 1) {
    throw new Error(
      `AgentControl Gate B 需要唯一 child Thread，实际为 ${JSON.stringify(childThreadIds)}`,
    );
  }
  const childThreadId = childThreadIds[0];
  const readReply = await invokeAppServerJsonRpcRaw(page, "thread/read", {
    threadId: childThreadId,
    includeTurns: true,
  });
  if (readReply?.error || !readReply?.result?.thread) {
    throw new Error(
      `读取 canonical child Thread 失败: ${JSON.stringify(readReply?.error ?? null)}`,
    );
  }
  const canonicalThread = readReply.result.thread;
  await openSubagentActivityThread(page, childThreadId, timeoutMs);
  try {
    await page.waitForFunction(
      ({ childSessionId, parentOwnedPlaceholders }) => {
        const textarea = Array.from(
          document.querySelectorAll('textarea[name="agent-chat-message"]'),
        ).find((node) => {
          const rect = node.getBoundingClientRect();
          return (
            rect.width > 0 &&
            rect.height > 0 &&
            node instanceof HTMLTextAreaElement &&
            node.dataset.sessionId === childSessionId
          );
        });
        return (
          textarea instanceof HTMLTextAreaElement &&
          textarea.disabled &&
          parentOwnedPlaceholders.includes(
            textarea.getAttribute("placeholder") || "",
          )
        );
      },
      {
        childSessionId: canonicalThread.sessionId,
        parentOwnedPlaceholders: PARENT_OWNED_PLACEHOLDERS,
      },
      { timeout: Math.min(timeoutMs, 30_000) },
    );
  } catch (error) {
    const diagnostics = await readInvokeDiagnostics(page);
    const navigationState = await page.evaluate(() => ({
      bodyText: (document.body.textContent || "").trim().slice(-1_000),
      textareas: Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).map((node) => ({
        disabled:
          node instanceof HTMLTextAreaElement ? node.disabled : undefined,
        sessionId:
          node instanceof HTMLTextAreaElement
            ? node.dataset.sessionId
            : undefined,
        visible: node.getBoundingClientRect().height > 0,
      })),
      toasts: Array.from(document.querySelectorAll("[data-sonner-toast]")).map(
        (node) => (node.textContent || "").trim().slice(0, 300),
      ),
    }));
    throw new Error(
      `parent-owned child GUI navigation failed: canonical=${JSON.stringify({
        canAcceptDirectInput: canonicalThread.canAcceptDirectInput ?? null,
        parentThreadId: canonicalThread.parentThreadId ?? null,
        sessionId: canonicalThread.sessionId ?? null,
        threadId: canonicalThread.id ?? null,
      })} dom=${JSON.stringify(navigationState)} calls=${JSON.stringify(
        diagnostics.appServerCalls.slice(-12),
      )}; ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  const dom = await page.evaluate(
    ({ childSessionId, childThreadId }) => {
      const visible = (node) => {
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      };
      const textarea = Array.from(
        document.querySelectorAll('textarea[name="agent-chat-message"]'),
      ).find(
        (node) =>
          node instanceof HTMLTextAreaElement &&
          node.dataset.sessionId === childSessionId &&
          visible(node),
      );
      if (!(textarea instanceof HTMLTextAreaElement)) {
        throw new Error("parent-owned child textarea missing");
      }
      const core = textarea.closest('[data-testid="inputbar-core-container"]');
      const send = core?.querySelector('[data-testid="send-btn"]');
      const accessMode = Array.from(
        document.querySelectorAll(
          '[data-testid="inputbar-access-mode-select"]',
        ),
      ).find(visible);
      const modelSelectors = Array.from(
        document.querySelectorAll('[data-testid="model-selector"]'),
      ).filter(visible);
      const taskModes = Array.from(
        document.querySelectorAll('[data-testid="inputbar-task-mode-status"]'),
      ).filter(visible);
      return {
        activeSessionId: textarea.dataset.sessionId || null,
        childThreadId,
        textareaVisible: textarea.getBoundingClientRect().height > 0,
        textareaDisabled: textarea.disabled,
        placeholder: textarea.getAttribute("placeholder") || "",
        controls: {
          sendButtonPresent: send instanceof HTMLButtonElement,
          sendDisabled:
            send instanceof HTMLButtonElement ? send.disabled : null,
          sendUnavailable:
            !(send instanceof HTMLButtonElement) || send.disabled,
          accessModeDisabled:
            accessMode instanceof HTMLSelectElement && accessMode.disabled,
          modelSelectorCount: modelSelectors.length,
          modelSelectorsDisabled:
            modelSelectors.length > 0 &&
            modelSelectors.every(
              (node) => node instanceof HTMLButtonElement && node.disabled,
            ),
          taskModeDisabled:
            taskModes.length === 0 ||
            taskModes.every(
              (node) => node instanceof HTMLButtonElement && node.disabled,
            ),
        },
      };
    },
    { childSessionId: canonicalThread.sessionId, childThreadId },
  );

  const diagnosticsBeforeAttempt = await readInvokeDiagnostics(page);
  const turnStartCountBefore = turnStartCountForThread(
    diagnosticsBeforeAttempt,
    childThreadId,
  );
  const uiAttempt = await page.evaluate(() => {
    const textarea = Array.from(
      document.querySelectorAll('textarea[name="agent-chat-message"]'),
    ).find((node) => {
      const rect = node.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    });
    const core = textarea?.closest('[data-testid="inputbar-core-container"]');
    const send = core?.querySelector('[data-testid="send-btn"]');
    const dispatchedEnter = textarea instanceof HTMLTextAreaElement;
    if (textarea instanceof HTMLTextAreaElement) {
      textarea.dispatchEvent(
        new KeyboardEvent("keydown", {
          bubbles: true,
          cancelable: true,
          key: "Enter",
        }),
      );
    }
    if (send instanceof HTMLButtonElement) {
      send.click();
    }
    return {
      dispatchedEnter,
      sendButtonPresent: send instanceof HTMLButtonElement,
      clickedDisabledSend: send instanceof HTMLButtonElement && send.disabled,
      sendUnavailable: !(send instanceof HTMLButtonElement) || send.disabled,
    };
  });
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
  const diagnosticsAfterAttempt = await readInvokeDiagnostics(page);
  const turnStartCountAfter = turnStartCountForThread(
    diagnosticsAfterAttempt,
    childThreadId,
  );

  const rejectedReply = await invokeAppServerJsonRpcRaw(page, "turn/start", {
    threadId: childThreadId,
    input: [{ type: "text", text: "parent-owned direct input must fail" }],
  });
  const screenshotPath = screenshotPathForEvidence(
    outputPath,
    "parent-owned-child",
  );
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });

  return {
    parentSessionId,
    childThreadId,
    canonicalThread: {
      id: canonicalThread.id,
      sessionId: canonicalThread.sessionId,
      parentThreadId: canonicalThread.parentThreadId ?? null,
      canAcceptDirectInput: canonicalThread.canAcceptDirectInput ?? null,
    },
    dom,
    uiAttempt: {
      ...uiAttempt,
      turnStartCountBefore,
      turnStartCountAfter,
    },
    serverRejection: {
      code: rejectedReply?.error?.code ?? null,
      message: rejectedReply?.error?.message ?? null,
      hasResult: Object.hasOwn(rejectedReply ?? {}, "result"),
    },
    screenshotPath: path.relative(process.cwd(), screenshotPath),
  };
}

async function collectDeferredMcpVisibleDomGateB({
  consoleErrors,
  evidence,
  outputPath,
  page,
  rendererSnapshot,
  timeoutMs,
}) {
  const sessionId = String(evidence?.runtime?.sessionId || "").trim();
  const deferredToolName = String(
    evidence?.scenarioRuntimeContext?.deferredToolName || "",
  ).trim();
  if (!sessionId || !deferredToolName) {
    throw new Error("deferred MCP evidence 缺少 sessionId 或 deferredToolName");
  }

  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: timeoutMs });
  const finalText = page.getByText(DEFERRED_MCP_TOOL_SEARCH_FINAL_TEXT, {
    exact: false,
  });
  await finalText.first().waitFor({ state: "visible", timeout: timeoutMs });

  await expandHistoricalToolRows(page, timeoutMs);
  const typedToolRows = await listTypedToolRows(page);
  console.log(`${LOG_PREFIX} typed-tool-rows=${JSON.stringify(typedToolRows)}`);
  const deferredRow = await findTypedToolRow(page, deferredToolName, timeoutMs);

  const diagnostics = await readInvokeDiagnostics(page);
  const screenshotPath = screenshotPathForEvidence(outputPath);
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const snapshot = {
    proofLevel: "Gate B",
    claimBoundary:
      "real Electron host/preload/App Server/runtime/read-model to visible DOM; localhost provider fixture, not live-provider proof",
    url: page.url(),
    electron: rendererSnapshot.electron === true,
    hasInvokeBridge: rendererSnapshot.hasInvokeBridge === true,
    supportsAppServer: rendererSnapshot.supportsAppServer === true,
    sessionId,
    activeSessionId: await input.getAttribute("data-session-id"),
    typedToolRows,
    deferredToolRow: await snapshotToolRow(deferredRow),
    finalAssistantTextVisible: await finalText.first().isVisible(),
    appServerCalls: diagnostics.appServerCalls,
    invokeErrorCount: diagnostics.invokeErrorCount,
    consoleErrorCount: consoleErrors.length,
    consoleErrors: consoleErrors.slice(0, 10),
    screenshotPath: path.relative(process.cwd(), screenshotPath),
  };
  const assertions = buildDeferredMcpVisibleDomAssertions({
    deferredToolName,
    evidence,
    snapshot,
  });
  return { assertions, snapshot };
}

async function collectToolOrchestratorSandboxRetryVisibleDomGateB({
  consoleErrors,
  evidence,
  outputPath,
  page,
  rendererSnapshot,
  timeoutMs,
}) {
  const sessionId = String(evidence?.runtime?.sessionId || "").trim();
  if (!sessionId) {
    throw new Error("sandbox retry evidence 缺少 sessionId");
  }
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: timeoutMs });
  const finalText = page.getByText(
    TOOL_ORCHESTRATOR_SANDBOX_RETRY_FINAL_TEXT,
    { exact: false },
  );
  await finalText.first().waitFor({ state: "visible", timeout: timeoutMs });
  await materializeHistoricalTimelines(page, timeoutMs);

  const typedToolRows = await listTypedToolRows(page);
  const fileChangeGroups = await listFileChangeGroups(page);
  const diagnostics = await readInvokeDiagnostics(page);
  const screenshotPath = screenshotPathForEvidence(
    outputPath,
    "sandbox-retry-visible-dom",
  );
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const snapshot = {
    proofLevel: "Gate B",
    claimBoundary:
      "real Electron host/preload/App Server/runtime/read-model to one visible file-change group after typed apply_patch sandbox approval retry; localhost provider fixture, not live-provider proof",
    url: page.url(),
    electron: rendererSnapshot.electron === true,
    hasInvokeBridge: rendererSnapshot.hasInvokeBridge === true,
    supportsAppServer: rendererSnapshot.supportsAppServer === true,
    sessionId,
    activeSessionId: await input.getAttribute("data-session-id"),
    typedToolRows,
    fileChangeGroups,
    finalAssistantTextVisible: await finalText.first().isVisible(),
    appServerCalls: diagnostics.appServerCalls,
    invokeErrorCount: diagnostics.invokeErrorCount,
    consoleErrorCount: consoleErrors.length,
    consoleErrors: consoleErrors.slice(0, 10),
    screenshotPath: path.relative(process.cwd(), screenshotPath),
  };
  const assertions =
    buildToolOrchestratorSandboxRetryVisibleDomAssertions({
      evidence,
      snapshot,
    });
  return { assertions, snapshot };
}

async function collectToolOrchestratorManagedNetworkRetryVisibleDomGateB({
  consoleErrors,
  evidence,
  outputPath,
  page,
  rendererSnapshot,
  timeoutMs,
}) {
  const sessionId = String(evidence?.runtime?.sessionId || "").trim();
  if (!sessionId) {
    throw new Error("managed-network retry evidence 缺少 sessionId");
  }
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: timeoutMs });
  const finalText = page.getByText(
    TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_FINAL_TEXT,
    { exact: false },
  );
  await finalText.first().waitFor({ state: "visible", timeout: timeoutMs });
  await expandHistoricalToolRows(page, timeoutMs);
  const endpointProof = page.getByText(
    "TOOL_ORCHESTRATOR_MANAGED_NETWORK_ENDPOINT_OK",
    { exact: false },
  );
  await endpointProof.first().waitFor({ state: "visible", timeout: timeoutMs });

  const typedToolRows = await listTypedToolRows(page);
  const diagnostics = await readInvokeDiagnostics(page);
  const screenshotPath = screenshotPathForEvidence(
    outputPath,
    "managed-network-retry-visible-dom",
  );
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const snapshot = {
    proofLevel: "Gate B",
    claimBoundary:
      "real Electron host/preload/App Server/runtime/read-model to one canonical exec_command identity after managed-network denial and typed network approval retry; localhost endpoint only, not live-provider proof",
    url: page.url(),
    electron: rendererSnapshot.electron === true,
    hasInvokeBridge: rendererSnapshot.hasInvokeBridge === true,
    supportsAppServer: rendererSnapshot.supportsAppServer === true,
    sessionId,
    activeSessionId: await input.getAttribute("data-session-id"),
    typedToolRows,
    endpointProofVisible: await endpointProof.first().isVisible(),
    finalAssistantTextVisible: await finalText.first().isVisible(),
    appServerCalls: diagnostics.appServerCalls,
    invokeErrorCount: diagnostics.invokeErrorCount,
    consoleErrorCount: consoleErrors.length,
    consoleErrors: consoleErrors.slice(0, 10),
    screenshotPath: path.relative(process.cwd(), screenshotPath),
  };
  const assertions =
    buildToolOrchestratorManagedNetworkRetryVisibleDomAssertions({
      evidence,
      snapshot,
    });
  return { assertions, snapshot };
}

async function collectAgentControlCapacityVisibleDomGateB({
  consoleErrors,
  evidence,
  outputPath,
  page,
  rendererSnapshot,
  timeoutMs,
}) {
  const sessionId = String(evidence?.runtime?.sessionId || "").trim();
  if (!sessionId) {
    throw new Error("AgentControl capacity evidence 缺少 sessionId");
  }
  const domState = await readAgentControlDomState({
    expectedFinalText: AGENT_CONTROL_CAPACITY_FINAL_TEXT,
    page,
    sessionId,
    timeoutMs,
  });
  const diagnostics = await readInvokeDiagnostics(page);
  const screenshotPath = screenshotPathForEvidence(
    outputPath,
    "capacity-visible-dom",
  );
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const snapshot = {
    proofLevel: "Gate B",
    claimBoundary:
      "real Electron host/preload/App Server/runtime/read-model to parallel AgentControl capacity rows and rejection; localhost provider fixture, not live-provider proof",
    url: page.url(),
    electron: rendererSnapshot.electron === true,
    hasInvokeBridge: rendererSnapshot.hasInvokeBridge === true,
    supportsAppServer: rendererSnapshot.supportsAppServer === true,
    sessionId,
    activeSessionId: domState.activeSessionId,
    typedToolRows: domState.typedToolRows,
    subagentActivityRows: domState.subagentActivityRows,
    finalAssistantTextVisible: domState.finalAssistantTextVisible,
    appServerCalls: diagnostics.appServerCalls,
    invokeErrorCount: diagnostics.invokeErrorCount,
    consoleErrorCount: consoleErrors.length,
    consoleErrors: consoleErrors.slice(0, 10),
    screenshotPath: path.relative(process.cwd(), screenshotPath),
  };
  const assertions = buildAgentControlCapacityVisibleDomAssertions({ snapshot });
  return { assertions, snapshot };
}

async function collectAgentControlResidencyVisibleDomGateB({
  coldRestart,
  consoleErrors,
  evidence,
  outputPath,
  page,
  rendererSnapshot,
  timeoutMs,
}) {
  const sessionId = String(evidence?.runtime?.sessionId || "").trim();
  if (!sessionId) {
    throw new Error("AgentControl residency evidence 缺少 sessionId");
  }
  const domState = await readAgentControlDomState({
    expectedFinalText: AGENT_CONTROL_RESIDENCY_FINAL_TEXT,
    page,
    sessionId,
    timeoutMs,
  });
  const diagnostics = await readInvokeDiagnostics(page);
  const screenshotPath = screenshotPathForEvidence(
    outputPath,
    "residency-visible-dom",
  );
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const snapshot = {
    proofLevel: "Gate B",
    claimBoundary:
      "real Electron host/preload/App Server/runtime/read-model to terminal-slot reuse and resident LRU cold reload; localhost provider fixture, not live-provider proof",
    url: page.url(),
    electron: rendererSnapshot.electron === true,
    hasInvokeBridge: rendererSnapshot.hasInvokeBridge === true,
    supportsAppServer: rendererSnapshot.supportsAppServer === true,
    sessionId,
    activeSessionId: domState.activeSessionId,
    typedToolRows: domState.typedToolRows,
    subagentActivityRows: domState.subagentActivityRows,
    finalAssistantTextVisible: domState.finalAssistantTextVisible,
    residency: {
      terminalSlotReused:
        evidence?.assertions?.rootCreatedFourDistinctChildren === true,
      lruColdReload:
        evidence?.assertions?.followupTaskReloadedFirstChild === true,
    },
    coldRestart,
    appServerCalls: diagnostics.appServerCalls,
    invokeErrorCount: diagnostics.invokeErrorCount,
    consoleErrorCount: consoleErrors.length,
    consoleErrors: consoleErrors.slice(0, 10),
    screenshotPath: path.relative(process.cwd(), screenshotPath),
  };
  const assertions = buildAgentControlResidencyVisibleDomAssertions({ snapshot });
  return { assertions, snapshot };
}

async function collectRolloutBudgetVisibleDomGateB({
  coldRestart,
  consoleErrors,
  evidence,
  outputPath,
  page,
  rendererSnapshot,
  timeoutMs,
}) {
  const sessionId = String(evidence?.runtime?.sessionId || "").trim();
  if (!sessionId) {
    throw new Error("rollout budget evidence 缺少 sessionId");
  }
  const domState = await readRolloutBudgetDomState({
    page,
    sessionId,
    timeoutMs,
  });
  const screenshotPath = screenshotPathForEvidence(
    outputPath,
    "rollout-budget-visible-dom",
  );
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const snapshot = {
    proofLevel: "Gate B",
    claimBoundary:
      "real Electron host/preload/App Server/runtime/read-model to rollout-budget exhaustion and restart admission rejection; localhost provider fixture, not live-provider proof",
    url: page.url(),
    electron: rendererSnapshot.electron === true,
    hasInvokeBridge: rendererSnapshot.hasInvokeBridge === true,
    supportsAppServer: rendererSnapshot.supportsAppServer === true,
    sessionId,
    activeSessionId: domState.activeSessionId,
    typedToolRows: domState.typedToolRows,
    subagentActivityRows: domState.subagentActivityRows,
    finalAssistantTextVisible: domState.finalAssistantTextVisible,
    failureVisible: domState.failureVisible,
    thread: domState.thread,
    latestTurnError: domState.latestTurnError,
    coldRestart,
    restartRejection: coldRestart?.restartRejection ?? null,
    appServerCalls: domState.appServerCalls,
    invokeErrorCount: domState.invokeErrorCount,
    consoleErrorCount: consoleErrors.length,
    consoleErrors: consoleErrors.slice(0, 10),
    screenshotPath: path.relative(process.cwd(), screenshotPath),
  };
  const assertions = buildRolloutBudgetVisibleDomAssertions({ snapshot });
  return { assertions, snapshot };
}

async function collectAgentControlVisibleDomGateB({
  coldRestart,
  consoleErrors,
  evidence,
  outputPath,
  page,
  preRestart,
  rendererSnapshot,
  timeoutMs,
}) {
  const sessionId = String(evidence?.runtime?.sessionId || "").trim();
  if (!sessionId) {
    throw new Error("AgentControl evidence 缺少 sessionId");
  }

  const domState = await readAgentControlDomState({
    page,
    sessionId,
    timeoutMs,
  });
  const { subagentActivityRows, typedToolRows } = domState;
  const threadId = String(evidence?.runtime?.threadId || "").trim();
  if (!threadId) {
    throw new Error("AgentControl evidence 缺少 threadId");
  }
  const parentReadReply = await invokeAppServerJsonRpcRaw(page, "thread/read", {
    threadId,
    includeTurns: true,
  });
  if (parentReadReply?.error || !parentReadReply?.result?.thread) {
    throw new Error(
      `冷重启后读取 parent canonical Thread 失败: ${JSON.stringify(parentReadReply?.error ?? null)}`,
    );
  }
  const restoredParent = normalizeToolExecutionThreadReadResponse(
    parentReadReply.result,
  );
  const waitAgentStates =
    restoredParent.thread_items.find((item) => item?.tool_name === "wait_agent")
      ?.agent_states ?? [];
  console.log(`${LOG_PREFIX} typed-tool-rows=${JSON.stringify(typedToolRows)}`);
  console.log(
    `${LOG_PREFIX} subagent-activity-rows=${JSON.stringify(subagentActivityRows)}`,
  );

  const screenshotPath = screenshotPathForEvidence(
    outputPath,
    "cold-restart-visible-dom",
  );
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  const parentOwnedChild = await collectParentOwnedChildGateB({
    outputPath,
    page,
    parentSessionId: sessionId,
    subagentActivityRows,
    timeoutMs,
  });
  const diagnostics = await readInvokeDiagnostics(page);
  const snapshot = {
    proofLevel: "Gate B",
    claimBoundary:
      "real Electron host/preload/App Server/runtime/read-model to six AgentControl Tool rows, canonical wait_agent states, and SubAgent activity DOM; localhost provider fixture, not live-provider proof",
    url: page.url(),
    electron: rendererSnapshot.electron === true,
    hasInvokeBridge: rendererSnapshot.hasInvokeBridge === true,
    supportsAppServer: rendererSnapshot.supportsAppServer === true,
    coldRestart,
    preRestart,
    sessionId,
    activeSessionId: domState.activeSessionId,
    typedToolRows,
    subagentActivityRows,
    waitAgentStates,
    parentOwnedChild,
    finalAssistantTextVisible: domState.finalAssistantTextVisible,
    appServerCalls: diagnostics.appServerCalls,
    invokeErrorCount: diagnostics.invokeErrorCount,
    consoleErrorCount: consoleErrors.length,
    consoleErrors: consoleErrors.slice(0, 10),
    screenshotPath: path.relative(process.cwd(), screenshotPath),
  };
  const assertions = buildAgentControlVisibleDomAssertions({
    evidence,
    snapshot,
  });
  return { assertions, snapshot };
}

async function invokeElectron(page, command, args) {
  return await page.evaluate(
    async ({ command, args }) => {
      const invoke = window.electronAPI?.invoke;
      if (typeof invoke !== "function") {
        throw new Error("Electron preload invoke bridge is unavailable");
      }
      return await invoke(command, args);
    },
    { command, args },
  );
}

async function invokeAppServerJsonRpcRaw(page, method, params) {
  return await page.evaluate(
    async ({ command, method, params }) => {
      const invoke = window.electronAPI?.invoke;
      if (typeof invoke !== "function") {
        throw new Error("Electron preload invoke bridge is unavailable");
      }
      const id = `agent-control-parent-owned-${Date.now()}-${Math.random()}`;
      const response = await invoke(command, {
        request: {
          lines: [JSON.stringify({ jsonrpc: "2.0", id, method, params })],
        },
      });
      const lines = response?.result?.lines ?? response?.lines;
      const messages = (Array.isArray(lines) ? lines : [])
        .map((line) => {
          try {
            return JSON.parse(String(line));
          } catch {
            return null;
          }
        })
        .filter(Boolean);
      return messages.find((message) => message?.id === id) ?? null;
    },
    { command: APP_SERVER_HANDLE_JSON_LINES_COMMAND, method, params },
  );
}

async function startBridgeProxy(page) {
  const server = createServer((request, response) => {
    void (async () => {
      if (request.method === "OPTIONS") {
        writeJson(response, 204, {});
        return;
      }
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      if (request.method === "GET" && url.pathname === "/health") {
        writeJson(response, 200, {
          status: "ok",
          transport: "managed-electron-host",
        });
        return;
      }
      if (request.method === "POST" && url.pathname === "/invoke") {
        const body = await readJsonBody(request);
        const command = typeof body.cmd === "string" ? body.cmd.trim() : "";
        if (!command) {
          writeJson(response, 400, { error: "cmd is required" });
          return;
        }
        try {
          const result = await invokeElectron(page, command, body.args ?? {});
          writeJson(response, 200, { result });
        } catch (error) {
          writeJson(response, 200, { error: sanitizeText(error) });
        }
        return;
      }
      writeJson(response, 404, { error: "not found" });
    })().catch((error) => {
      writeJson(response, 200, { error: sanitizeText(error) });
    });
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  if (!port) {
    throw new Error("managed DevBridge proxy 未获得监听端口");
  }
  return {
    server,
    baseUrl: `http://127.0.0.1:${port}`,
  };
}

function runChild(args, bridgeBaseUrl) {
  const childArgs = [
    "scripts/agent-runtime/tool-execution-smoke.mjs",
    ...args,
    "--health-url",
    `${bridgeBaseUrl}/health`,
    "--invoke-url",
    `${bridgeBaseUrl}/invoke`,
  ];
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, childArgs, {
      cwd: process.cwd(),
      env: process.env,
      stdio: "inherit",
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      resolve({
        code: typeof code === "number" ? code : signal ? 1 : 0,
        signal: signal || "",
      });
    });
  });
}

function writeRolloutBudgetGateConfig(runtimeEnv) {
  const configPath = path.join(runtimeEnv.electronUserDataDir, "config.yaml");
  fs.writeFileSync(
    configPath,
    [
      "agent:",
      "  rollout_budget:",
      "    limit_tokens: 1",
      "    reminder_at_remaining_tokens: []",
      "    sampling_token_weight: 1.0",
      "    prefill_token_weight: 1.0",
      "",
    ].join("\n"),
    "utf8",
  );
  runtimeEnv.env.LIME_CONFIG_PATH = configPath;
  return configPath;
}

async function closeServer(server) {
  if (!server) {
    return;
  }
  await new Promise((resolve) => server.close(resolve));
}

async function closeElectronApp(app) {
  if (!app) {
    return;
  }
  try {
    await app.close();
  } catch (error) {
    console.warn(
      `${LOG_PREFIX} electron close skipped: ${sanitizeText(error)}`,
    );
    try {
      const childProcess =
        typeof app.process === "function" ? app.process() : null;
      if (childProcess && !childProcess.killed) {
        childProcess.kill("SIGTERM");
      }
    } catch {
      // best effort cleanup
    }
  }
}

async function main() {
  const childArgs = process.argv.slice(2);
  const timeoutMs = timeoutFromArgs(childArgs);
  const visibleDomGateBKind = visibleDomGateBKindFromArgs(childArgs);
  const coldRestartRequested = childArgs.includes("--cold-restart");
  const soakConfig = resolveSoakConfig(childArgs);
  if (
    coldRestartRequested &&
    !["agent-control", "agent-residency", "rollout-budget"].includes(
      visibleDomGateBKind,
    )
  ) {
    throw new Error(
      "--cold-restart 只允许用于 AgentControl 或 rollout-budget batch",
    );
  }
  if (
    ["agent-control", "agent-residency"].includes(visibleDomGateBKind) &&
    !coldRestartRequested
  ) {
    throw new Error(
      "agent-control-tools visible DOM Gate B 必须显式启用 --cold-restart",
    );
  }
  if (visibleDomGateBKind && childArgs.includes("--no-write")) {
    throw new Error("visible-DOM Gate B 需要写入结构化 evidence");
  }
  if (soakConfig.enabled && visibleDomGateBKind !== "agent-control") {
    throw new Error("SOAK 多轮模式当前只允许用于 agent-control-tools batch");
  }
  const outputPath = resolveToolExecutionEvidencePath(
    childArgs,
    DEFAULT_EVIDENCE_OUTPUT,
  );
  const runtimeEnv = createToolExecutionTempRuntimeEnv();
  const rolloutBudgetConfigPath =
    visibleDomGateBKind === "rollout-budget"
      ? writeRolloutBudgetGateConfig(runtimeEnv)
      : null;
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
  let app = null;
  let page = null;
  let bridge = null;
  const consoleErrors = [];
  const processSnapshots = [];
  const restartRecords = [];
  const soakRounds = [];
  const soakRoundEvidencePaths = [];
  try {
    console.log(`${LOG_PREFIX} stage=launch-electron`);
    const launched = await launchManagedElectron({
      appServerEnv,
      consoleErrors,
      runtimeEnv,
      timeoutMs,
    });
    app = launched.app;
    page = launched.page;
    const initialElectronPid = app.process().pid;

    console.log(
      `${LOG_PREFIX} renderer ready url=${launched.rendererSnapshot.url} title=${launched.rendererSnapshot.title}`,
    );

    console.log(`${LOG_PREFIX} stage=start-bridge-proxy`);
    bridge = await startBridgeProxy(page);
    console.log(`${LOG_PREFIX} bridge=${bridge.baseUrl}`);

    for (let roundIndex = 0; roundIndex < soakConfig.rounds; roundIndex += 1) {
      const roundStartedAt = Date.now();
      const roundOutputPath = roundEvidencePath(
        outputPath,
        roundIndex,
        soakConfig.rounds,
      );
      console.log(
        `${LOG_PREFIX} stage=runtime-round round=${roundIndex + 1}/${soakConfig.rounds}`,
      );
      const childStartedAt = Date.now();
      const result = await runChild(
        childArgsForRound(childArgs, roundOutputPath),
        bridge.baseUrl,
      );
      const childDurationMs = Date.now() - childStartedAt;
      if (result.code !== 0) {
        process.exitCode = result.code;
        return;
      }
      if (soakConfig.enabled) {
        soakRoundEvidencePaths.push(roundOutputPath);
        const evidenceReadStartedAt = Date.now();
        const evidence = readToolExecutionEvidence(roundOutputPath);
        const evidenceReadDurationMs = Date.now() - evidenceReadStartedAt;
        const processSnapshotStartedAt = Date.now();
        const processSnapshot = collectProcessTreeSnapshot(
          app.process().pid,
          `round-${roundIndex + 1}`,
        );
        const processSnapshotDurationMs = Date.now() - processSnapshotStartedAt;
        processSnapshots.push(processSnapshot);
        const observationStartedAt = Date.now();
        const observation = await collectSoakRoundObservation({
          evidence,
          outputPath: roundOutputPath,
          page,
          processSnapshot,
          roundIndex,
        });
        const observationDurationMs = Date.now() - observationStartedAt;
        observation.phaseTimings = {
          childDurationMs,
          evidenceReadDurationMs,
          processSnapshotDurationMs,
          observationDurationMs,
        };
        observation.durationMs = Date.now() - roundStartedAt;
        soakRounds.push(observation);
      }
    }
    if (visibleDomGateBKind) {
      console.log(`${LOG_PREFIX} stage=visible-dom-gate-b`);
      const evidence = readToolExecutionEvidence(outputPath);
      const sessionId = String(evidence?.runtime?.sessionId || "").trim();
      consoleErrors.length = 0;
      let coldRestart = null;
      let preRestart = null;
      let rendererSnapshot = null;
      if (["agent-control", "agent-residency"].includes(visibleDomGateBKind)) {
        const expectedFinalText =
          visibleDomGateBKind === "agent-residency"
            ? AGENT_CONTROL_RESIDENCY_FINAL_TEXT
            : AGENT_CONTROL_FINAL_TEXT;
        rendererSnapshot = await restoreAgentSessionRoute(
          page,
          sessionId,
          timeoutMs,
        );
        const preRestartDomState = await readAgentControlDomState({
          expectedFinalText,
          page,
          sessionId,
          timeoutMs,
        });
        const preRestartScreenshotPath = screenshotPathForEvidence(
          outputPath,
          "pre-restart-visible-dom",
        );
        fs.mkdirSync(path.dirname(preRestartScreenshotPath), {
          recursive: true,
        });
        await page.screenshot({
          path: preRestartScreenshotPath,
          fullPage: true,
        });
        preRestart = {
          ...preRestartDomState,
          screenshotPath: path.relative(
            process.cwd(),
            preRestartScreenshotPath,
          ),
        };
        consoleErrors.length = 0;
        const restartResult = await runManagedColdRestarts({
          app,
          appServerEnv,
          bridge,
          closeElectronApp,
          closeServer,
          consoleErrors,
          count: soakConfig.coldRestarts,
          initialElectronPid,
          launchManagedElectron,
          logPrefix: LOG_PREFIX,
          readAgentControlDomState,
          readRestoredDomState: (args) =>
            readAgentControlDomState({ ...args, expectedFinalText }),
          restoreAgentSessionRoute,
          runtimeEnv,
          sessionId,
          timeoutMs,
        });
        app = restartResult.app;
        bridge = restartResult.bridge;
        page = restartResult.page;
        rendererSnapshot = restartResult.rendererSnapshot;
        processSnapshots.push(...restartResult.processSnapshots);
        restartRecords.push(...restartResult.restartRecords);
        coldRestart = {
          initialElectronPid,
          restartedElectronPid: app.process().pid,
          restartCount: restartRecords.length,
          restarts: restartRecords,
          electronProcessReplaced: restartRecords.every(
            (restart) => restart.electronProcessReplaced === true,
          ),
        };
      } else if (visibleDomGateBKind === "rollout-budget") {
        rendererSnapshot = await restoreAgentSessionRoute(
          page,
          sessionId,
          timeoutMs,
        );
        const restartResult = await runManagedColdRestarts({
          app,
          appServerEnv,
          bridge,
          closeElectronApp,
          closeServer,
          consoleErrors,
          count: 1,
          initialElectronPid,
          launchManagedElectron,
          logPrefix: LOG_PREFIX,
          readAgentControlDomState,
          readRestoredDomState: readRolloutBudgetDomState,
          restoreAgentSessionRoute,
          runtimeEnv,
          sessionId,
          timeoutMs,
        });
        app = restartResult.app;
        bridge = restartResult.bridge;
        page = restartResult.page;
        rendererSnapshot = restartResult.rendererSnapshot;
        processSnapshots.push(...restartResult.processSnapshots);
        restartRecords.push(...restartResult.restartRecords);
        const restartParams = buildToolExecutionTurnStartParams({
          clientUserMessageId: `rollout-budget-restart-${Date.now()}`,
          message: "retry after rollout budget exhaustion",
          model: String(evidence?.provider?.modelPreference || "lime-fixture-chat"),
          threadId: sessionId,
          workspaceRoot: evidence?.workspace?.root,
        });
        const configRead = await invokeAppServerJsonRpcRaw(page, "config/read", {
          includeLayers: true,
        });
        const configuredRolloutBudget =
          configRead?.result?.config?.agent?.rollout_budget ??
          configRead?.result?.config?.agent?.rolloutBudget ??
          null;
        const runtimeConfig = {
          configReadSucceeded: Boolean(configRead?.result?.config),
          rolloutBudgetEnabled: Boolean(configuredRolloutBudget),
          rolloutBudgetLimitTokens:
            typeof configuredRolloutBudget?.limit_tokens === "number"
              ? configuredRolloutBudget.limit_tokens
              : typeof configuredRolloutBudget?.limitTokens === "number"
                ? configuredRolloutBudget.limitTokens
                : null,
        };
        const resumedThread = await invokeAppServerJsonRpcRaw(
          page,
          "thread/resume",
          {
            threadId: sessionId,
            excludeTurns: true,
          },
        );
        const restartRejection = await invokeAppServerJsonRpcRaw(
          page,
          "turn/start",
          restartParams,
        );
        coldRestart = {
          initialElectronPid,
          restartedElectronPid: app.process().pid,
          restartCount: restartRecords.length,
          restarts: restartRecords,
          electronProcessReplaced: restartRecords.every(
            (restart) => restart.electronProcessReplaced === true,
          ),
          runtimeConfig,
          resumedThread,
          restartRejection,
        };
      } else {
        rendererSnapshot = await restoreAgentSessionRoute(
          page,
          sessionId,
          timeoutMs,
        );
      }
      const visibleDomGateB =
        visibleDomGateBKind === "agent-control"
          ? await collectAgentControlVisibleDomGateB({
              coldRestart,
              consoleErrors,
              evidence,
              outputPath,
              page,
              preRestart,
              rendererSnapshot,
              timeoutMs,
            })
          : visibleDomGateBKind === "agent-residency"
            ? await collectAgentControlResidencyVisibleDomGateB({
                coldRestart,
                consoleErrors,
                evidence,
                outputPath,
                page,
                rendererSnapshot,
                timeoutMs,
              })
          : visibleDomGateBKind === "agent-capacity"
            ? await collectAgentControlCapacityVisibleDomGateB({
                consoleErrors,
                evidence,
                outputPath,
                page,
                rendererSnapshot,
                timeoutMs,
              })
          : visibleDomGateBKind === "sandbox-retry"
              ? await collectToolOrchestratorSandboxRetryVisibleDomGateB({
                  consoleErrors,
                  evidence,
                  outputPath,
                  page,
                  rendererSnapshot,
                  timeoutMs,
                })
              : visibleDomGateBKind === "managed-network-retry"
                ? await collectToolOrchestratorManagedNetworkRetryVisibleDomGateB({
                    consoleErrors,
                    evidence,
                    outputPath,
                    page,
                    rendererSnapshot,
                    timeoutMs,
                  })
              : visibleDomGateBKind === "rollout-budget"
                ? await collectRolloutBudgetVisibleDomGateB({
                    coldRestart,
                    consoleErrors,
                    evidence,
                    outputPath,
                    page,
                    rendererSnapshot,
                    timeoutMs,
                  })
                : await collectDeferredMcpVisibleDomGateB({
                  consoleErrors,
                  evidence,
                  outputPath,
                  page,
                  rendererSnapshot,
                  timeoutMs,
                });
      const failedAssertions = Object.entries(visibleDomGateB.assertions)
        .filter(([, passed]) => passed !== true)
        .map(([name]) => name);
      evidence.gui = {
        ...(evidence.gui && typeof evidence.gui === "object"
          ? evidence.gui
          : {}),
        [visibleDomGateBKind === "agent-control"
          ? "agentControlVisibleDomGateB"
          : visibleDomGateBKind === "agent-residency"
            ? "agentControlResidencyVisibleDomGateB"
            : visibleDomGateBKind === "agent-capacity"
            ? "agentControlCapacityVisibleDomGateB"
            : visibleDomGateBKind === "sandbox-retry"
              ? "toolOrchestratorSandboxRetryVisibleDomGateB"
              : visibleDomGateBKind === "managed-network-retry"
                ? "toolOrchestratorManagedNetworkRetryVisibleDomGateB"
                : visibleDomGateBKind === "rollout-budget"
                  ? "rolloutBudgetVisibleDomGateB"
                  : "visibleDomGateB"]: visibleDomGateB.snapshot,
      };
      evidence.assertions = {
        ...(evidence.assertions && typeof evidence.assertions === "object"
          ? evidence.assertions
          : {}),
        ...visibleDomGateB.assertions,
      };
      evidence.failedAssertions = [
        ...(Array.isArray(evidence.failedAssertions)
          ? evidence.failedAssertions
          : []),
        ...failedAssertions,
      ];
      evidence.status = evidence.failedAssertions.length > 0 ? "fail" : "pass";
      writeToolExecutionEvidence(outputPath, evidence);
      if (failedAssertions.length > 0) {
        throw new Error(
          `${visibleDomGateBKind} visible-DOM Gate B 失败: ${failedAssertions.join(", ")}`,
        );
      }
      console.log(
        `${LOG_PREFIX} visible-dom-gate-b=pass evidence=${outputPath} screenshot=${visibleDomGateB.snapshot.screenshotPath}`,
      );
    }
    if (soakConfig.enabled) {
      const finalProcessSnapshot = collectProcessTreeSnapshot(
        app.process().pid,
        "pre-final-shutdown",
      );
      const restoredRounds = await collectRestoredSoakRounds({
        evidencePaths: soakRoundEvidencePaths,
        page,
        processSnapshot: finalProcessSnapshot,
        readEvidence: readToolExecutionEvidence,
      });
      await closeElectronApp(app);
      app = null;
      const finalShutdown = await waitForProcessIdsExit(
        finalProcessSnapshot.processes.map((entry) => entry.pid),
      );
      const evidence = readToolExecutionEvidence(outputPath);
      const soak = buildSoakSummary({
        finalShutdown,
        processSnapshots,
        restoredRounds,
        restarts: restartRecords,
        rounds: soakRounds,
      });
      const failedSoakAssertions = Object.entries(soak.assertions)
        .filter(([, passed]) => passed !== true)
        .map(([name]) => name);
      evidence.soak = soak;
      evidence.assertions = {
        ...(evidence.assertions && typeof evidence.assertions === "object"
          ? evidence.assertions
          : {}),
        ...soak.assertions,
      };
      evidence.failedAssertions = [
        ...new Set([
          ...(Array.isArray(evidence.failedAssertions)
            ? evidence.failedAssertions
            : []),
          ...failedSoakAssertions,
        ]),
      ];
      evidence.status = evidence.failedAssertions.length > 0 ? "fail" : "pass";
      writeToolExecutionEvidence(outputPath, evidence);
      if (failedSoakAssertions.length > 0) {
        throw new Error(`SOAK-01 失败: ${failedSoakAssertions.join(", ")}`);
      }
      console.log(
        `${LOG_PREFIX} soak=pass rounds=${soak.roundCount} restarts=${soak.restartCount} evidence=${outputPath}`,
      );
    }
    process.exitCode = 0;
  } finally {
    await closeServer(bridge?.server);
    await closeElectronApp(app);
    cleanupToolExecutionTempRoot(runtimeEnv.tempRoot, {
      logPrefix: LOG_PREFIX,
      sanitizeText,
    });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : error);
  process.exitCode = 1;
});
