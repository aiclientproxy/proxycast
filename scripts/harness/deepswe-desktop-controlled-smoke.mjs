#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import {
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  sanitizeText,
  sleep,
} from "../electron/mcp-config-fixture-smoke.mjs";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { startOpenAiCompatibleFixtureServer } from "../lib/openai-compatible-fixture-server.mjs";
import {
  bindGuiWorkspaceAndModelPreferences,
  clearInvokeBuffers,
  ensureDefaultWorkspace,
  initializeAppServer,
  invokeAppServerFromPage,
  waitForRendererReady,
} from "../agent-runtime/claw-chat-current-fixture-rpc.mjs";
import { setInputbarAccessMode } from "../agent-runtime/claw-chat-current-fixture-gui-actions.mjs";
import {
  createToolExecutionThreadCurrent,
  normalizeToolExecutionThreadReadResponse,
  provisionToolExecutionFixtureProvider,
} from "../agent-runtime/tool-execution-current-contract.mjs";
import {
  DEEPSWE_DESKTOP_TRIAL_SCHEMA,
  evaluateDesktopSuite,
  evaluateDesktopTrial,
  loadDesktopManifest,
  preflightDesktopManifest,
  sha256,
} from "./deepswe-desktop-contract.mjs";
import {
  controlledFixtureForTask,
  controlledFixtureResponses,
  controlledFixtureTaskIds,
} from "./deepswe-desktop-controlled-fixtures.mjs";

const LOG_PREFIX = "[harness:deepswe:desktop:controlled]";
const DEFAULT_TIMEOUT_MS = 240_000;
const DEFAULT_INTERVAL_MS = 250;
const DEFAULT_OUTPUT_DIR = ".lime/benchmark/v2/desktop/controlled";
const NAVIGATION_RESTORE_STORAGE_KEY = "lime.appNavigation.restore.v1";
const INVOKE_TRACE_STORAGE_KEY = "lime_invoke_trace_buffer_v1";
const INVOKE_ERROR_STORAGE_KEY = "lime_invoke_error_buffer_v1";
const FAILURE_PROBE_TIMEOUT_MS = 2_000;
const APP_SERVER_EXIT_CONSOLE_MARKER = "[electron-host] app-server exited";
const APP_SERVER_RESTART_FAILED_CONSOLE_MARKER =
  "[electron-host] app-server restart failed";
const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "interrupted",
  "cancelled",
  "canceled",
  "aborted",
]);

function usage() {
  return `
DeepSWE Desktop Controlled Product Smoke

Usage:
  node scripts/harness/deepswe-desktop-controlled-smoke.mjs [options]

Options:
  --task <id|all>            Run one Desktop Smoke 5 task or all, default all
  --output-dir <path>        Evidence directory, default ${DEFAULT_OUTPUT_DIR}
  --timeout-ms <ms>          Per-task timeout, default ${DEFAULT_TIMEOUT_MS}
  --interval-ms <ms>         Poll interval, default ${DEFAULT_INTERVAL_MS}
  --electron-executable <p>  Optional packaged Electron executable
  -h, --help                 Show this help

This runner uses a controlled localhost provider and synthetic language fixtures.
It proves the desktop product path, not live model quality or Pier correctness.
`;
}

export function parseArgs(argv) {
  const options = {
    task: "all",
    outputDir: path.resolve(DEFAULT_OUTPUT_DIR),
    timeoutMs: DEFAULT_TIMEOUT_MS,
    intervalMs: DEFAULT_INTERVAL_MS,
    electronExecutable: null,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    if (arg === "--task" && argv[index + 1]) {
      options.task = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--output-dir" && argv[index + 1]) {
      options.outputDir = path.resolve(String(argv[index + 1]));
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
    if (arg === "--electron-executable" && argv[index + 1]) {
      options.electronExecutable = path.resolve(String(argv[index + 1]));
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 60_000) {
    throw new Error("--timeout-ms must be >= 60000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be >= 100");
  }
  const taskIds = controlledFixtureTaskIds();
  if (options.task !== "all" && !taskIds.includes(options.task)) {
    throw new Error(`Desktop Smoke 5 task not found: ${options.task}`);
  }
  return options;
}

function timestampId(date = new Date()) {
  return date
    .toISOString()
    .replace(/[-:]/gu, "")
    .replace(/\.\d{3}Z$/u, "Z");
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

export function classifyElectronHostLifecycleConsole(text) {
  const normalized = String(text || "");
  if (normalized.includes(APP_SERVER_EXIT_CONSOLE_MARKER)) {
    return {
      event: "app-server-exited",
      source: "electron-main-console",
    };
  }
  if (normalized.includes(APP_SERVER_RESTART_FAILED_CONSOLE_MARKER)) {
    return {
      event: "app-server-restart-failed",
      source: "electron-main-console",
    };
  }
  return null;
}

function isProcessAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function electronProcessPid(app) {
  if (typeof app?.process !== "function") return null;
  try {
    return app.process()?.pid ?? null;
  } catch {
    return null;
  }
}

function observeWithTimeout(operation, timeoutMs) {
  let timer = null;
  const operationPromise = Promise.resolve()
    .then(operation)
    .then((value) => ({ status: "ok", value }))
    .catch((error) => ({
      status: "error",
      error: sanitizeText(error),
    }));
  const timeoutPromise = new Promise((resolve) => {
    timer = setTimeout(
      () => resolve({ status: "timeout", timeoutMs }),
      timeoutMs,
    );
  });
  return Promise.race([operationPromise, timeoutPromise]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

export function attachElectronHostLifecycleDiagnostics(app, eventSink) {
  if (!app || typeof app.on !== "function" || !Array.isArray(eventSink)) {
    return;
  }
  app.on("console", (message) => {
    const event = classifyElectronHostLifecycleConsole(
      typeof message?.text === "function" ? message.text() : "",
    );
    if (!event) return;
    eventSink.push({ ...event, observedAt: new Date().toISOString() });
  });
}

function summarizeFailureBridgeProbe(result, threadId) {
  if (result.status !== "ok") {
    return {
      method: "thread/read",
      status: result.status,
      ...(result.error ? { error: result.error } : {}),
      ...(result.timeoutMs ? { timeoutMs: result.timeoutMs } : {}),
    };
  }
  const thread = result.value?.result?.thread;
  const turns = Array.isArray(thread?.turns) ? thread.turns : [];
  const turn = turns.at(-1) || null;
  return {
    method: "thread/read",
    status: "ok",
    threadFound: Boolean(thread),
    threadIdMatches: String(thread?.id || "") === String(threadId || ""),
    latestTurn: turn
      ? {
          idPresent: Boolean(turn.id),
          status: turnStatus(turn),
        }
      : null,
  };
}

export async function collectControlledFailureDiagnostics({
  electronHandle,
  identity,
  probeTimeoutMs = FAILURE_PROBE_TIMEOUT_MS,
}) {
  const app = electronHandle?.app;
  const page = electronHandle?.page;
  const electronPid = electronProcessPid(app);
  const diagnostics = {
    schemaVersion: "deepswe-desktop-controlled-runtime-diagnostics-v1",
    probeTimeoutMs,
    electron: {
      pid: electronPid,
      processAlive: electronPid === null ? null : isProcessAlive(electronPid),
    },
    renderer: {
      status: "skipped",
    },
    bridge: {
      status: "skipped",
      method: "thread/read",
    },
  };

  if (!page || typeof page.evaluate !== "function") {
    return diagnostics;
  }

  const rendererProbe = await observeWithTimeout(
    () =>
      page.evaluate(() => ({
        url: window.location.href,
        electron: window.__LIME_ELECTRON__ === true,
        hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
        supportsAppServer:
          typeof window.electronAPI?.supportsCommand === "function" &&
          window.electronAPI.supportsCommand("app_server_handle_json_lines"),
      })),
    probeTimeoutMs,
  );
  if (rendererProbe.status === "ok") {
    diagnostics.renderer = {
      status: "ok",
      ...rendererProbe.value,
    };
  } else {
    diagnostics.renderer = {
      status: rendererProbe.status,
      ...(rendererProbe.error ? { error: rendererProbe.error } : {}),
      ...(rendererProbe.timeoutMs
        ? { timeoutMs: rendererProbe.timeoutMs }
        : {}),
    };
  }

  const threadId = String(identity?.threadId || "").trim();
  if (!threadId) {
    return diagnostics;
  }
  const bridgeProbe = await observeWithTimeout(
    () =>
      invokeAppServerFromPage(page, "thread/read", {
        threadId,
        includeTurns: true,
      }),
    probeTimeoutMs,
  );
  diagnostics.bridge = summarizeFailureBridgeProbe(bridgeProbe, threadId);
  return diagnostics;
}

export function buildControlledFailureEvidence({
  error,
  fixtureServer,
  runId,
  stage,
  taskId,
  runtimeDiagnostics = null,
  sidecarLifecycleEvents = [],
}) {
  const providerRequests = (fixtureServer?.requests || [])
    .filter((request) => request.path === "/v1/chat/completions")
    .map((request, index) => ({
      index: index + 1,
      path: request.path,
      responseKind: request.responseKind || null,
      responseToolName: request.responseToolName || null,
      stream: request.body?.stream === true,
      messageCount: Array.isArray(request.body?.messages)
        ? request.body.messages.length
        : 0,
      toolCount: Array.isArray(request.body?.tools)
        ? request.body.tools.length
        : 0,
    }));
  return {
    schemaVersion: "deepswe-desktop-controlled-failure-v1",
    generatedAt: new Date().toISOString(),
    runId,
    taskId,
    stage,
    error: error instanceof Error ? error.message : String(error),
    providerRequestCount: providerRequests.length,
    providerRequests,
    connectionDiagnostics: fixtureServer?.connectionDiagnostics || [],
    runtimeDiagnostics,
    sidecarLifecycleEvents,
  };
}

function writeFixtureFiles(workspaceRoot, fixture) {
  for (const [relativePath, content] of Object.entries(fixture.files)) {
    const absolutePath = path.join(workspaceRoot, relativePath);
    fs.mkdirSync(path.dirname(absolutePath), { recursive: true });
    fs.writeFileSync(absolutePath, content);
  }
}

function changedFixtureFiles(workspaceRoot, fixture) {
  const finalFiles = { ...fixture.files, ...fixture.finalFiles };
  return Object.entries(finalFiles)
    .filter(([relativePath, content]) => {
      const absolutePath = path.join(workspaceRoot, relativePath);
      return (
        !Object.hasOwn(fixture.files, relativePath) ||
        !fs.existsSync(absolutePath) ||
        fs.readFileSync(absolutePath, "utf8") !== fixture.files[relativePath] ||
        content !== fixture.files[relativePath]
      );
    })
    .map(([relativePath]) => relativePath)
    .filter((relativePath) => {
      const expected = finalFiles[relativePath];
      const absolutePath = path.join(workspaceRoot, relativePath);
      return (
        fs.existsSync(absolutePath) &&
        fs.readFileSync(absolutePath, "utf8") === expected
      );
    })
    .sort();
}

function copyComparisonFiles(targetRoot, files) {
  for (const [relativePath, content] of Object.entries(files)) {
    const absolutePath = path.join(targetRoot, relativePath);
    fs.mkdirSync(path.dirname(absolutePath), { recursive: true });
    fs.writeFileSync(absolutePath, content);
  }
}

export function captureControlledPatch(workspaceRoot, fixture) {
  const compareRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "deepswe-desktop-patch-"),
  );
  const beforeRoot = path.join(compareRoot, "before");
  const afterRoot = path.join(compareRoot, "after");
  fs.mkdirSync(beforeRoot);
  fs.mkdirSync(afterRoot);
  copyComparisonFiles(beforeRoot, fixture.files);
  const candidateFiles = { ...fixture.files, ...fixture.finalFiles };
  for (const relativePath of Object.keys(candidateFiles)) {
    const absolutePath = path.join(workspaceRoot, relativePath);
    if (!fs.existsSync(absolutePath)) continue;
    const destination = path.join(afterRoot, relativePath);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(absolutePath, destination);
  }
  const result = spawnSync(
    "git",
    ["diff", "--no-index", "--binary", "--no-ext-diff", "before", "after"],
    {
      cwd: compareRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  if (![0, 1].includes(result.status) || result.error) {
    throw new Error(
      `controlled patch capture failed: ${result.error?.message || result.stderr}`,
    );
  }
  const patch = String(result.stdout || "")
    .replaceAll("a/before/", "a/")
    .replaceAll("b/after/", "b/")
    .replaceAll("--- before/", "--- a/")
    .replaceAll("+++ after/", "+++ b/");
  return {
    patch,
    patchSha256: sha256(patch),
    patchBytes: Buffer.byteLength(patch),
    changedFiles: changedFixtureFiles(workspaceRoot, fixture),
    comparisonRoot: compareRoot,
  };
}

async function invokeResult(page, requestLog, options, method, params) {
  void options;
  const response = await invokeAppServerFromPage(
    page,
    method,
    params,
    requestLog,
  );
  return response.result;
}

async function bindActualSessionPreferences(
  page,
  { model, providerId, sessionId, workspaceId },
) {
  await bindGuiWorkspaceAndModelPreferences(page, workspaceId, {
    model,
    provider: providerId,
    sessionId,
  });
  await page.evaluate(
    ({ model, providerId, sessionId, workspaceId }) => {
      const set = (key, value) =>
        window.localStorage.setItem(key, JSON.stringify(value));
      set(`agent_pref_provider_${workspaceId}`, providerId);
      set(`agent_pref_model_${workspaceId}`, model);
      set(`agent_topic_model_pref_${workspaceId}_${sessionId}`, {
        providerType: providerId,
        model,
      });
      set(`agent_topic_model_pref_global_${sessionId}`, {
        providerType: providerId,
        model,
      });
      set(`agent_session_workspace_${sessionId}`, workspaceId);
      window.sessionStorage.setItem(
        "lime.appNavigation.restore.v1",
        JSON.stringify({
          page: "agent",
          params: { initialSessionId: sessionId },
        }),
      );
    },
    { model, providerId, sessionId, workspaceId },
  );
}

async function restoreThreadInGui(page, options, preferences) {
  await bindActualSessionPreferences(page, preferences);
  await page.reload({
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  await waitForRendererReady(page, options);
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${preferences.sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: options.timeoutMs });
  await page.waitForFunction(
    (sessionId) => {
      const textarea = document.querySelector(
        `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
      );
      return textarea instanceof HTMLTextAreaElement && !textarea.disabled;
    },
    preferences.sessionId,
    { timeout: options.timeoutMs },
  );
  return input;
}

async function submitInstruction(page, input, instruction, options) {
  await clearInvokeBuffers(page);
  await input.fill(instruction);
  const send = input
    .locator('xpath=ancestor::*[@data-testid="inputbar-core-container"]')
    .locator('[data-testid="send-btn"]');
  await send.waitFor({ state: "visible", timeout: options.timeoutMs });
  await send.click({ timeout: options.timeoutMs });
}

async function waitForApprovalPrompt(page, options) {
  const prompt = page.locator('[data-testid="inputbar-approval-prompt"]');
  await prompt.waitFor({ state: "visible", timeout: options.timeoutMs });
  const approvalButton = prompt.locator(
    'button[data-decision="allow_once"], button[data-decision="allow_for_session"]',
  );
  await approvalButton.first().waitFor({
    state: "visible",
    timeout: options.timeoutMs,
  });
  return prompt.evaluate((element) => ({
    requestId:
      element.closest("[data-request-id]")?.getAttribute("data-request-id") ??
      null,
    availableDecisions: Array.from(
      element.querySelectorAll("button[data-decision]"),
    ).map((button) => button.getAttribute("data-decision")),
    text: (element.textContent || "").trim().slice(0, 1_000),
  }));
}

async function clickApprovalDecision(page, options, decision) {
  const button = page.locator(
    `[data-testid="inputbar-approval-prompt"] button[data-decision="${decision}"]`,
  );
  await button.waitFor({ state: "visible", timeout: options.timeoutMs });
  await button.click({ timeout: options.timeoutMs });
}

function latestTurn(thread) {
  const turns = Array.isArray(thread?.turns) ? thread.turns : [];
  return turns.at(-1) ?? null;
}

function turnStatus(turn) {
  return String(turn?.status || "")
    .trim()
    .toLowerCase();
}

function providerRequestCount(fixtureServer) {
  return fixtureServer.requests.filter(
    (request) => request.path === "/v1/chat/completions",
  ).length;
}

async function waitForTurnWithProviderRequest(
  page,
  options,
  threadId,
  fixtureServer,
  requestLog,
  minimumProviderRequests,
) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const response = await invokeAppServerFromPage(
      page,
      "thread/read",
      { threadId, includeTurns: true },
      requestLog,
    );
    const thread = response.result?.thread;
    const turn = latestTurn(thread);
    last = {
      raw: response.result,
      thread,
      turn,
      providerRequestCount: providerRequestCount(fixtureServer),
    };
    if (
      turn &&
      turnStatus(turn) === "inprogress" &&
      last.providerRequestCount >= minimumProviderRequests
    ) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `desktop recovery turn did not become active: ${sanitizeText(
      JSON.stringify({
        providerRequestCount: last?.providerRequestCount ?? 0,
        status: turnStatus(last?.turn),
      }),
    )}`,
  );
}

async function waitForSpecificTerminalThread(
  page,
  options,
  threadId,
  expectedTurnId,
  fixtureServer,
  requestLog,
  acceptableStatuses = TERMINAL_STATUSES,
) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const response = await invokeAppServerFromPage(
      page,
      "thread/read",
      { threadId, includeTurns: true },
      requestLog,
    );
    const thread = response.result?.thread;
    const turn = latestTurn(thread);
    last = {
      raw: response.result,
      thread,
      turn,
      providerRequestCount: providerRequestCount(fixtureServer),
    };
    if (
      String(turn?.id || "") === expectedTurnId &&
      acceptableStatuses.has(turnStatus(turn))
    ) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `desktop recovery turn did not reach terminal: ${sanitizeText(
      JSON.stringify({
        expectedTurnId,
        providerRequestCount: last?.providerRequestCount ?? 0,
        latestTurnId: last?.turn?.id ?? null,
        status: turnStatus(last?.turn),
        acceptableStatuses: [...acceptableStatuses],
      }),
    )}`,
  );
}

async function waitForTerminalThread(
  page,
  options,
  threadId,
  fixtureServer,
  requestLog,
) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const response = await invokeAppServerFromPage(
      page,
      "thread/read",
      { threadId, includeTurns: true },
      requestLog,
    );
    const thread = response.result?.thread;
    const turn = latestTurn(thread);
    last = {
      raw: response.result,
      thread,
      turn,
      providerRequestCount: providerRequestCount(fixtureServer),
    };
    if (
      turn &&
      TERMINAL_STATUSES.has(turnStatus(turn)) &&
      last.providerRequestCount >= 6
    ) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `desktop task did not reach terminal: ${sanitizeText(
      JSON.stringify({
        providerRequestCount: last?.providerRequestCount ?? 0,
        status: turnStatus(last?.turn),
      }),
    )}`,
  );
}

async function expandHistoricalRows(page) {
  const previews = page.locator(
    '[data-testid^="message-list-historical-timeline-preview:"]',
  );
  for (let index = 0; index < 10 && (await previews.count()) > 0; index += 1) {
    await previews.first().click();
  }
}

async function readVisibleState(page, options, sessionId, fixture) {
  const finalText = page
    .getByText(fixture.finalMarker, { exact: false })
    .first();
  await finalText.waitFor({ state: "visible", timeout: options.timeoutMs });
  await expandHistoricalRows(page);
  const toolRows = await page
    .locator('[data-testid="tool-call-row"]')
    .evaluateAll((nodes) =>
      nodes.map((node) => ({
        name: node.getAttribute("data-tool-name"),
        status: node.getAttribute("data-tool-status"),
        visible:
          window.getComputedStyle(node).display !== "none" &&
          window.getComputedStyle(node).visibility !== "hidden" &&
          node.getBoundingClientRect().height > 0,
        text: (node.textContent || "").trim().slice(0, 700),
      })),
    );
  const diffGroups = await page
    .locator('[data-testid="timeline-file-artifact-group"]')
    .evaluateAll((nodes) =>
      nodes.map((node) => ({
        visible:
          window.getComputedStyle(node).display !== "none" &&
          window.getComputedStyle(node).visibility !== "hidden" &&
          node.getBoundingClientRect().height > 0,
        fileCount: node.querySelectorAll(
          '[data-testid="file-changes-summary-file-row"]',
        ).length,
        text: (node.textContent || "").trim().slice(0, 700),
      })),
    );
  return {
    activeSessionId: await page
      .locator(
        `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
      )
      .getAttribute("data-session-id"),
    terminalVisible: await finalText.isVisible(),
    testOutputVisible: toolRows.some((row) =>
      row.text.includes(fixture.testMarker),
    ),
    toolRows,
    diffGroups,
  };
}

function primaryArtifactPreviewMarker(fixture) {
  const initialLines = new Set(
    String(fixture.files[fixture.primaryPath] || "")
      .split(/\r?\n/u)
      .map((line) => line.trim()),
  );
  const marker = String(fixture.finalFiles[fixture.primaryPath] || "")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .find((line) => line && !initialLines.has(line));
  assert(
    marker,
    `controlled fixture has no preview marker: ${fixture.primaryPath}`,
  );
  return marker;
}

async function openPrimaryArtifactPreview(page, options, fixture) {
  const marker = primaryArtifactPreviewMarker(fixture);
  const row = page
    .locator('[data-testid="message-artifact-card"]')
    .filter({ hasText: fixture.primaryPath })
    .first();
  await row.waitFor({ state: "visible", timeout: options.timeoutMs });
  await row.click();

  const renderer = page
    .locator('[data-testid="canvas-workbench-code-preview"]')
    .last();
  await renderer.waitFor({ state: "visible", timeout: options.timeoutMs });
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const contentVisible = await renderer
      .evaluate(
        (node, expected) =>
          String(node.textContent || "").includes(String(expected)),
        marker,
      )
      .catch(() => false);
    const unavailableErrorVisible = await page
      .getByText("App Server artifact 内容不可用", { exact: false })
      .isVisible()
      .catch(() => false);
    last = { contentVisible, unavailableErrorVisible };
    if (contentVisible && !unavailableErrorVisible) {
      const diagnostics = await readInvokeDiagnostics(page);
      return {
        status: "pass",
        path: fixture.primaryPath,
        contentVisible: true,
        unavailableErrorVisible: false,
        artifactReadSeen: diagnostics.calls.some(
          (call) => call.method === "artifact/read",
        ),
        markerSha256: sha256(marker),
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `artifact preview did not expose workspace content: ${sanitizeText(
      JSON.stringify({ path: fixture.primaryPath, ...last }),
    )}`,
  );
}

async function readInvokeDiagnostics(page) {
  return await page.evaluate(
    ({ errorKey, traceKey }) => {
      const readArray = (key) => {
        try {
          const value = JSON.parse(window.localStorage.getItem(key) || "[]");
          return Array.isArray(value) ? value : [];
        } catch {
          return [];
        }
      };
      const traces = readArray(traceKey);
      const calls = traces.flatMap((entry) => {
        if (entry?.command !== "app_server_handle_json_lines") return [];
        const lines = Array.isArray(entry?.args_preview?.request?.lines)
          ? entry.args_preview.request.lines
          : [];
        return lines.flatMap((line) => {
          try {
            const message = typeof line === "string" ? JSON.parse(line) : line;
            return typeof message?.method === "string"
              ? [{ method: message.method, status: entry?.status || null }]
              : [];
          } catch {
            return [];
          }
        });
      });
      return {
        calls,
        invokeErrorCount: readArray(errorKey).length,
        mockFallbackHitCount: traces.filter((entry) => {
          if (entry?.mock === true || entry?.mockFallback === true) return true;
          return [entry?.transport, entry?.source, entry?.fallback].some(
            (value) =>
              typeof value === "string" && value.toLowerCase().includes("mock"),
          );
        }).length,
      };
    },
    { errorKey: INVOKE_ERROR_STORAGE_KEY, traceKey: INVOKE_TRACE_STORAGE_KEY },
  );
}

function providerToolNames(request) {
  const tools = Array.isArray(request?.body?.tools) ? request.body.tools : [];
  return tools
    .map((tool) => String(tool?.function?.name || tool?.name || "").trim())
    .filter(Boolean);
}

async function closeControlledElectronFixture(handle) {
  const app = handle?.app;
  if (!app) return;
  const pid = typeof app.process === "function" ? app.process()?.pid : null;
  await Promise.race([closeElectronFixture(handle), sleep(5_000)]);
  if (!pid) return;
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
    // The fixture exited during closeElectronFixture().
  }
}

function summarizeProvider(fixtureServer) {
  const requests = fixtureServer.requests.filter(
    (request) => request.path === "/v1/chat/completions",
  );
  const requiredTools = ["Read", "Glob", "Grep", "apply_patch", "exec_command"];
  return {
    source: "localhost-controlled-fixture",
    requestCount: requests.length,
    requestErrors: requests
      .map((request) => request.responseError)
      .filter(Boolean),
    requiredToolsAdvertised: Object.fromEntries(
      requiredTools.map((tool) => [
        tool,
        requests.some((request) => providerToolNames(request).includes(tool)),
      ]),
    ),
  };
}

function normalizedProjectedLifecycle(threadRead) {
  return threadRead.thread_items.map((item) => ({
    id: item.call_id,
    name: item.tool_name,
    status: item.status,
    success: item.success,
    output: String(item.output || "").slice(0, 2_000),
  }));
}

function providerToolOutput(request, callId) {
  const messages = Array.isArray(request?.body?.messages)
    ? request.body.messages
    : [];
  const message = messages.find(
    (candidate) =>
      String(candidate?.tool_call_id || candidate?.toolCallId || "") === callId,
  );
  if (message) return String(message.content || "");
  const serialized = JSON.stringify(request?.body || {});
  return serialized.includes(callId) ? "[structured tool output observed]" : "";
}

function controlledToolLifecycle(taskId, fixtureServer, projectedLifecycle) {
  const expectedCalls = controlledFixtureResponses(taskId).filter(
    (response) => response.type === "tool_call" && !response.recovery,
  );
  const requests = fixtureServer.requests.filter(
    (request) => request.path === "/v1/chat/completions",
  );
  return expectedCalls.map((call) => {
    const output = requests
      .map((request) => providerToolOutput(request, call.id))
      .find(Boolean);
    const projected = projectedLifecycle.find(
      (item) => item.id === call.id || item.id === `item_${call.id}`,
    );
    return {
      id: call.id,
      name: call.name,
      status: output ? "completed" : projected?.status || "missing",
      success: output ? true : (projected?.success ?? null),
      output: String(output || projected?.output || "").slice(0, 2_000),
      evidenceSource: output ? "provider_tool_output" : "canonical_read_model",
      projectedAs: projected?.name || null,
      projectedStatus: projected?.status || null,
    };
  });
}

async function runRecoveryScenarios({
  fixture,
  fixtureServer,
  identity,
  options,
  page,
  preferences,
  requestLog,
  taskId,
  workspaceRoot,
}) {
  const approvalInput = await restoreThreadInGui(page, options, preferences);
  await setInputbarAccessMode(page, options, "current", {
    expectedSessionId: preferences.sessionId,
  });
  await submitInstruction(
    page,
    approvalInput,
    `Desktop controlled approval resume ${identity.threadId}`,
    options,
  );
  const approvalPending = await waitForApprovalPrompt(page, options);
  const approvalActive = await waitForTurnWithProviderRequest(
    page,
    options,
    identity.threadId,
    fixtureServer,
    requestLog,
    7,
  );
  const approvalTurnId = String(approvalActive.turn?.id || "").trim();
  assert(approvalTurnId, "approval recovery turn did not return turn identity");
  await clickApprovalDecision(page, options, "allow_once");
  const approvalTerminal = await waitForSpecificTerminalThread(
    page,
    options,
    identity.threadId,
    approvalTurnId,
    fixtureServer,
    requestLog,
    new Set(["completed"]),
  );
  const approvalNormalized = normalizeToolExecutionThreadReadResponse(
    approvalTerminal.raw,
  );
  const approvalExecTools = approvalNormalized.thread_items.filter(
    (item) => item.tool_name === "exec_command",
  );
  const approvalTool =
    approvalExecTools.find(
      (item) =>
        item.id === `desktop-${taskId}-approval-resume` ||
        JSON.stringify(item).includes(fixture.recovery.approvalResumeFile),
    ) || approvalExecTools.at(-1);
  const approvalMarkerPath = path.join(
    workspaceRoot,
    fixture.recovery.approvalResumeFile,
  );
  const approvalDoneVisible = await page
    .getByText(fixture.recovery.approvalResumeDoneText, { exact: false })
    .first()
    .isVisible()
    .catch(() => false);
  const approvalDoneInReadModel = JSON.stringify(approvalTerminal.raw).includes(
    fixture.recovery.approvalResumeDoneText,
  );

  const cancelInput = await restoreThreadInGui(page, options, preferences);
  await setInputbarAccessMode(page, options, "current", {
    expectedSessionId: preferences.sessionId,
  });
  await submitInstruction(
    page,
    cancelInput,
    `Desktop controlled approval cancel ${identity.threadId}`,
    options,
  );
  const cancelPending = await waitForApprovalPrompt(page, options);
  const cancelActive = await waitForTurnWithProviderRequest(
    page,
    options,
    identity.threadId,
    fixtureServer,
    requestLog,
    9,
  );
  const cancelTurnId = String(cancelActive.turn?.id || "").trim();
  assert(cancelTurnId, "cancel recovery turn did not return turn identity");
  await clickApprovalDecision(page, options, "cancel");
  const cancelTerminal = await waitForSpecificTerminalThread(
    page,
    options,
    identity.threadId,
    cancelTurnId,
    fixtureServer,
    requestLog,
  );
  const cancelNormalized = normalizeToolExecutionThreadReadResponse(
    cancelTerminal.raw,
  );
  const cancelExecTools = cancelNormalized.thread_items.filter(
    (item) => item.tool_name === "exec_command",
  );
  const cancelTool =
    cancelExecTools.find(
      (item) => item.id === `desktop-${taskId}-cancel-no-ghost-write`,
    ) || cancelExecTools.at(-1);
  const cancelMarkerPath = path.join(
    workspaceRoot,
    fixture.recovery.cancelNoGhostWriteFile,
  );
  const cancelTerminalStatus = turnStatus(cancelTerminal.turn);
  const cancelProviderRequestCount = providerRequestCount(fixtureServer);
  const approvalPassed =
    approvalPending.requestId &&
    approvalTerminal.turn?.id === approvalTurnId &&
    turnStatus(approvalTerminal.turn) === "completed" &&
    fs.existsSync(approvalMarkerPath) &&
    approvalTool?.status === "completed" &&
    (approvalDoneVisible || approvalDoneInReadModel);
  const cancelPassed =
    cancelPending.requestId &&
    cancelTerminal.turn?.id === cancelTurnId &&
    ["cancelled", "canceled", "interrupted", "aborted"].includes(
      cancelTerminalStatus,
    ) &&
    !fs.existsSync(cancelMarkerPath) &&
    cancelTool?.status !== "completed" &&
    cancelProviderRequestCount === 9;
  return {
    cancelNoGhostWrite: {
      status: cancelPassed ? "pass" : "fail",
      turnId: cancelTurnId,
      requestId: cancelPending.requestId,
      terminalStatus: cancelTerminalStatus,
      toolStatus: cancelTool?.status || null,
      toolId: cancelTool?.id || null,
      markerPath: path.relative(workspaceRoot, cancelMarkerPath),
      markerExists: fs.existsSync(cancelMarkerPath),
      providerRequestCount: cancelProviderRequestCount,
    },
    approvalResume: {
      status: approvalPassed ? "pass" : "fail",
      turnId: approvalTurnId,
      requestId: approvalPending.requestId,
      terminalStatus: turnStatus(approvalTerminal.turn),
      toolStatus: approvalTool?.status || null,
      toolId: approvalTool?.id || null,
      markerPath: path.relative(workspaceRoot, approvalMarkerPath),
      markerExists: fs.existsSync(approvalMarkerPath),
      doneVisible: approvalDoneVisible,
      doneInReadModel: approvalDoneInReadModel,
    },
  };
}

function taskInstruction(repoRoot, task) {
  return fs.readFileSync(path.resolve(repoRoot, task.instructionPath));
}

export async function runControlledTask({
  manifest,
  options,
  repoRoot,
  runId,
  task,
}) {
  const runtimeEnv = createTempRuntimeEnv();
  const rustupHome =
    process.env.RUSTUP_HOME || path.join(os.homedir(), ".rustup");
  const cargoHome = process.env.CARGO_HOME || path.join(os.homedir(), ".cargo");
  if (fs.existsSync(rustupHome)) runtimeEnv.env.RUSTUP_HOME = rustupHome;
  if (fs.existsSync(cargoHome)) runtimeEnv.env.CARGO_HOME = cargoHome;
  const goEnv = spawnSync("go", ["env", "GOCACHE", "GOMODCACHE", "GOPATH"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "ignore"],
  });
  if (goEnv.status === 0) {
    const [goCache, goModuleCache, goPath] = goEnv.stdout
      .trim()
      .split(/\r?\n/u)
      .map((value) => value.trim());
    if (goCache && fs.existsSync(goCache)) runtimeEnv.env.GOCACHE = goCache;
    if (goModuleCache && fs.existsSync(goModuleCache)) {
      runtimeEnv.env.GOMODCACHE = goModuleCache;
    }
    if (goPath && fs.existsSync(goPath)) runtimeEnv.env.GOPATH = goPath;
  }
  runtimeEnv.env.GOTOOLCHAIN = "local";
  const requestLog = [];
  const consoleErrors = [];
  const pageErrors = [];
  const sidecarLifecycleEvents = [];
  const fixtureDefinition = controlledFixtureForTask(task.id);
  const instruction = taskInstruction(repoRoot, task);
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  let electronHandle = null;
  let fixtureServer = null;
  let identity = null;
  let stage = "fixture";
  try {
    console.log(`${LOG_PREFIX} task=${task.id} stage=fixture`);
    fixtureServer = await startOpenAiCompatibleFixtureServer({
      captureConnectionDiagnostics: true,
      deferScriptedToolCallsUntilAvailable: true,
      scriptedResponses: controlledFixtureResponses(task.id),
    });
    stage = "electron";
    console.log(`${LOG_PREFIX} task=${task.id} stage=electron`);
    electronHandle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "runtime",
    });
    attachElectronHostLifecycleDiagnostics(
      electronHandle.app,
      sidecarLifecycleEvents,
    );
    const { page, rendererSnapshot } = electronHandle;
    await initializeAppServer(page, requestLog);
    const workspace = await ensureDefaultWorkspace(page, requestLog);
    const workspaceRoot = path.resolve(String(workspace.rootPath || ""));
    assert(workspaceRoot, "workspace/default/ensure did not return rootPath");
    writeFixtureFiles(workspaceRoot, fixtureDefinition);

    const invoke = (ignored, method, params) =>
      invokeResult(page, requestLog, ignored, method, params);
    const provider = await provisionToolExecutionFixtureProvider({
      fixture: fixtureServer,
      invoke,
      options,
    });
    identity = await createToolExecutionThreadCurrent({
      invoke,
      options,
      provider,
      title: `DeepSWE Desktop controlled ${task.id}`,
      workspaceRoot,
    });
    const preferences = {
      model: provider.modelPreference,
      providerId: provider.providerPreference,
      sessionId: identity.sessionId,
      workspaceId: workspace.workspaceId,
    };

    stage = "gui-submit";
    console.log(`${LOG_PREFIX} task=${task.id} stage=gui-submit`);
    const input = await restoreThreadInGui(page, options, preferences);
    await submitInstruction(page, input, instruction.toString("utf8"), options);
    stage = "terminal-wait";
    const terminal = await waitForTerminalThread(
      page,
      options,
      identity.threadId,
      fixtureServer,
      requestLog,
    );
    const normalized = normalizeToolExecutionThreadReadResponse(terminal.raw);
    const projectedToolLifecycle = normalizedProjectedLifecycle(normalized);
    const toolLifecycle = controlledToolLifecycle(
      task.id,
      fixtureServer,
      projectedToolLifecycle,
    );
    const turnId = String(terminal.turn?.id || "").trim();
    assert(turnId, "terminal thread did not return turn identity");
    const firstVisible = await readVisibleState(
      page,
      options,
      identity.sessionId,
      fixtureDefinition,
    );
    stage = "reopen";
    console.log(`${LOG_PREFIX} task=${task.id} stage=reopen`);
    await restoreThreadInGui(page, options, preferences);
    const reopenedVisible = await readVisibleState(
      page,
      options,
      identity.sessionId,
      fixtureDefinition,
    );
    stage = "artifact-preview";
    console.log(`${LOG_PREFIX} task=${task.id} stage=artifact-preview`);
    const artifactPreview = await openPrimaryArtifactPreview(
      page,
      options,
      fixtureDefinition,
    );
    const artifactPreviewScreenshotPath = path.join(
      options.outputDir,
      runId,
      `${task.id}.artifact-preview.png`,
    );
    fs.mkdirSync(path.dirname(artifactPreviewScreenshotPath), {
      recursive: true,
    });
    await page.screenshot({
      path: artifactPreviewScreenshotPath,
      fullPage: true,
    });
    const patch = captureControlledPatch(workspaceRoot, fixtureDefinition);
    assert(patch.patchBytes > 0, "controlled task produced an empty patch");
    stage = "recovery-approval";
    console.log(`${LOG_PREFIX} task=${task.id} stage=recovery-approval`);
    const recovery = await runRecoveryScenarios({
      fixture: fixtureDefinition,
      fixtureServer,
      identity,
      options,
      page,
      preferences,
      requestLog,
      taskId: task.id,
      workspaceRoot,
    });
    const diagnostics = await readInvokeDiagnostics(page);
    const execCall = toolLifecycle.find((item) => item.name === "exec_command");
    const testOutputVisible =
      firstVisible.testOutputVisible ||
      reopenedVisible.testOutputVisible ||
      String(execCall?.output || "").includes(fixtureDefinition.testMarker);
    const expectedVisibleTools = ["Glob", "Grep", "exec_command"];
    const visibleToolNames = new Set(
      reopenedVisible.toolRows
        .filter((row) => row.visible && row.status === "completed")
        .map((row) => row.name),
    );
    const providerEvidence = summarizeProvider(fixtureServer);
    const screenshotPath = path.join(
      options.outputDir,
      runId,
      `${task.id}.png`,
    );
    fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
    await page.screenshot({ path: screenshotPath, fullPage: true });

    let evidence = {
      schemaVersion: DEEPSWE_DESKTOP_TRIAL_SCHEMA,
      trialId: `${runId}-${task.id}`,
      trialKind: "controlled_product_smoke",
      generatedAt: new Date().toISOString(),
      claimBoundary:
        "real Electron/preload/IPC/App Server/runtime/read model/GUI and native language commands with a controlled localhost provider and synthetic repository fixture; not live model or Pier evidence",
      taskId: task.id,
      language: task.language,
      repository: task.repository,
      sourceCommit: task.baseCommit,
      instructionSha256: sha256(instruction),
      workspace: workspaceRoot,
      workspaceRetention: {
        retained: true,
        reason:
          "preserved for evidence inspection; no automatic destructive cleanup",
        runtimeTempRoot: runtimeEnv.tempRoot,
      },
      identity: {
        sessionId: identity.sessionId,
        threadId: identity.threadId,
        turnId,
      },
      patchSha256: patch.patchSha256,
      patchBytes: patch.patchBytes,
      changedFiles: patch.changedFiles,
      toolLifecycle,
      projectedToolLifecycle,
      testResult: {
        command: fixtureDefinition.testCommand,
        status:
          execCall?.status === "completed" &&
          execCall?.success !== false &&
          testOutputVisible
            ? "pass"
            : "fail",
        exitCode:
          execCall?.status === "completed" && execCall?.success !== false
            ? 0
            : 1,
        outputVisible: testOutputVisible,
        marker: fixtureDefinition.testMarker,
      },
      provider: providerEvidence,
      gui: {
        electron: rendererSnapshot.electron === true,
        preloadInvokeBridge: rendererSnapshot.hasInvokeBridge === true,
        identity: {
          sessionId: reopenedVisible.activeSessionId,
          threadId: identity.threadId,
          turnId,
        },
        terminalVisible:
          firstVisible.terminalVisible && reopenedVisible.terminalVisible,
        toolLifecycleVisible:
          expectedVisibleTools.every((tool) => visibleToolNames.has(tool)) &&
          reopenedVisible.diffGroups.some(
            (group) => group.visible && group.fileCount > 0,
          ),
        diffArtifactVisible: reopenedVisible.diffGroups.some(
          (group) => group.visible && group.fileCount > 0,
        ),
        artifactPreview: {
          ...artifactPreview,
          screenshotPath: path.relative(
            repoRoot,
            artifactPreviewScreenshotPath,
          ),
        },
        toolRows: reopenedVisible.toolRows,
        diffGroups: reopenedVisible.diffGroups,
        consoleErrorCount: consoleErrors.length,
        pageErrorCount: pageErrors.length,
        consoleErrors: consoleErrors.slice(0, 10),
        pageErrors: pageErrors.slice(0, 10),
        screenshotPath: path.relative(repoRoot, screenshotPath),
      },
      readModel: {
        sessionId: normalized.session_id,
        threadId: normalized.thread_id,
        turnId,
        terminalStatus: turnStatus(terminal.turn),
        toolCount: toolLifecycle.length,
      },
      bridge: {
        appServerHandleJsonLinesSeen: diagnostics.calls.some(
          (call) =>
            call.method === "turn/start" || call.method === "thread/read",
        ),
        requestMethods: requestLog.map((entry) => entry.method),
        calls: diagnostics.calls,
        mockFallbackHitCount: diagnostics.mockFallbackHitCount,
        invokeErrorCount: diagnostics.invokeErrorCount,
      },
      recovery: {
        sessionReopen: {
          status:
            reopenedVisible.activeSessionId === identity.sessionId &&
            reopenedVisible.terminalVisible
              ? "pass"
              : "fail",
          sessionId: reopenedVisible.activeSessionId,
        },
        cancelNoGhostWrite: recovery.cancelNoGhostWrite,
        approvalResume: recovery.approvalResume,
      },
      verifier: {
        status: "not_run",
        reason:
          "controlled synthetic fixture is not a DeepSWE candidate; Pier must only receive the live task patch",
        patchSha256: null,
        artifacts: [],
      },
      gateB: { pass: false },
      desktopCodingPass: false,
    };
    let verdict = evaluateDesktopTrial({ evidence, manifest, repoRoot });
    evidence = {
      ...evidence,
      gateB: {
        pass: verdict.gateBPass,
        assertions: verdict.assertions,
        failedAssertions: verdict.failedGateBAssertions,
      },
      desktopCodingPass: false,
    };
    verdict = evaluateDesktopTrial({ evidence, manifest, repoRoot });
    evidence.verdict = verdict;
    const evidencePath = path.join(
      options.outputDir,
      runId,
      `${task.id}.trial.json`,
    );
    const patchPath = path.join(
      options.outputDir,
      runId,
      `${task.id}.patch.diff`,
    );
    fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
    fs.writeFileSync(patchPath, patch.patch);
    evidence.patchPath = path.relative(repoRoot, patchPath);
    writeJson(evidencePath, evidence);
    console.log(
      `${LOG_PREFIX} task=${task.id} gateB=${verdict.gateBPass ? "pass" : "fail"} evidence=${evidencePath}`,
    );
    return evidence;
  } catch (error) {
    const runtimeDiagnostics = await collectControlledFailureDiagnostics({
      electronHandle,
      identity,
    });
    const failurePath = path.join(
      options.outputDir,
      runId,
      `${task.id}.failure.json`,
    );
    writeJson(
      failurePath,
      buildControlledFailureEvidence({
        error,
        fixtureServer,
        runId,
        runtimeDiagnostics,
        sidecarLifecycleEvents,
        stage,
        taskId: task.id,
      }),
    );
    console.error(
      `${LOG_PREFIX} task=${task.id} stage=${stage} failure=${failurePath}`,
    );
    throw error;
  } finally {
    await closeControlledElectronFixture(electronHandle);
    await fixtureServer?.close().catch(() => undefined);
  }
}

export async function runControlledSuite(options, repoRoot = process.cwd()) {
  const { manifest } = loadDesktopManifest(repoRoot);
  const preflight = preflightDesktopManifest({ repoRoot, manifest });
  if (preflight.status !== "pass") {
    throw new Error(
      `Desktop Smoke 5 preflight failed: ${JSON.stringify(preflight.checks.filter((check) => !check.passed))}`,
    );
  }
  const selectedTasks =
    options.task === "all"
      ? manifest.tasks
      : manifest.tasks.filter((task) => task.id === options.task);
  const runId = timestampId();
  const evidenceList = [];
  for (const task of selectedTasks) {
    evidenceList.push(
      await runControlledTask({ manifest, options, repoRoot, runId, task }),
    );
  }
  const suite = {
    ...evaluateDesktopSuite({ evidenceList, manifest, repoRoot }),
    generatedAt: new Date().toISOString(),
    runId,
    mode: "controlled_product_smoke",
    preflight,
    claimBoundary:
      "controlled provider desktop product-path evidence only; DesktopCodingPass remains false until live DeepSWE and Pier use the same patch",
  };
  const summaryPath = path.join(options.outputDir, runId, "summary.json");
  writeJson(summaryPath, suite);
  const allSelectedPassed =
    suite.trials.every((trial) => trial.gateBPass && trial.claimConsistent) &&
    suite.assertions.recoveryCoverageComplete;
  if (!allSelectedPassed) process.exitCode = 1;
  console.log(
    `${LOG_PREFIX} status=${suite.status} gateB=${allSelectedPassed ? "pass" : "fail"} desktopCodingPass=false summary=${summaryPath}`,
  );
  return suite;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  await runControlledSuite(options);
}

if (
  process.argv[1] &&
  pathToFileURL(process.argv[1]).href === import.meta.url
) {
  main().catch((error) => {
    console.error(
      error instanceof Error ? error.stack || error.message : error,
    );
    process.exitCode = 1;
  });
}
