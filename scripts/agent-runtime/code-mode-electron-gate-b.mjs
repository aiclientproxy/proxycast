#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
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
} from "./claw-chat-current-fixture-rpc.mjs";

const LOG_PREFIX = "[smoke:code-mode-electron-gate-b]";
const DEFAULT_OUTPUT = path.resolve(
  ".lime/qc/gui-evidence/code-mode-electron-gate-b/code-mode-electron-gate-b-summary.json",
);
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_INTERVAL_MS = 250;
const OFFICIAL_OPENAI_BASE_URL = "http://api.openai.com/v1";
const MODEL_NAME = "fixture-code-mode";
const PROVIDER_API_KEY = "fixture-code-mode-key";
const PROVIDER_TYPE = "gateway";
const PROMPT = "运行 CodeMode 专项桌面闭环，并返回最终可见结果。";
const EXEC_CALL_ID = "call-code-mode-gate-b";
const EXEC_OUTPUT_MARKER = "CODE_MODE_GATE_B_OK";
const FINAL_TEXT = "CODE_MODE_GATE_B_VISIBLE";
const NAVIGATION_RESTORE_STORAGE_KEY = "lime.appNavigation.restore.v1";
const INVOKE_TRACE_STORAGE_KEY = "lime_invoke_trace_buffer_v1";
const INVOKE_ERROR_STORAGE_KEY = "lime_invoke_error_buffer_v1";
const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
  "canceled",
  "interrupted",
]);

function usage() {
  return `
CodeMode Electron Gate B

Usage:
  node scripts/agent-runtime/code-mode-electron-gate-b.mjs [options]

Options:
  --output <path>       Evidence JSON path
  --timeout-ms <ms>     Overall timeout, default ${DEFAULT_TIMEOUT_MS}
  --interval-ms <ms>    Predicate polling interval, default ${DEFAULT_INTERVAL_MS}
  -h, --help            Show this help
`;
}

export function parseArgs(argv) {
  const options = {
    output: DEFAULT_OUTPUT,
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
    if (arg === "--output" && argv[index + 1]) {
      options.output = path.resolve(String(argv[index + 1]));
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
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be >= 100");
  }
  return options;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

function modelConfig() {
  return {
    id: MODEL_NAME,
    capability: {
      taskFamilies: ["chat"],
      inputModalities: ["text"],
      outputModalities: ["text"],
      runtimeFeatures: [
        "streaming",
        "tool_calling",
        "custom_tools",
        "responses_api",
      ],
      capabilities: {
        vision: false,
        tools: true,
        streaming: true,
        jsonMode: false,
        functionCalling: true,
        reasoning: false,
      },
    },
  };
}

function customToolOutput(body) {
  const input = Array.isArray(body?.input) ? body.input : [];
  return input.find(
    (item) =>
      item?.type === "custom_tool_call_output" &&
      item?.call_id === EXEC_CALL_ID,
  );
}

async function startCodeModeFixture() {
  return await startOpenAiCompatibleFixtureServer({
    apiKey: PROVIDER_API_KEY,
    model: MODEL_NAME,
    modelRuntimeFeatures: modelConfig().capability.runtimeFeatures,
    modelToolMode: "code_mode",
    scriptedResponses: [
      {
        type: "custom_tool_call",
        id: EXEC_CALL_ID,
        name: "exec",
        input: `text("${EXEC_OUTPUT_MARKER}");`,
      },
      ({ body }) => {
        const output = customToolOutput(body);
        if (!String(output?.output || "").includes(EXEC_OUTPUT_MARKER)) {
          throw new Error(
            "second Responses request is missing the CodeMode custom_tool_call_output marker",
          );
        }
        return { type: "text", content: FINAL_TEXT };
      },
    ],
  });
}

function installProviderProxy(runtimeEnv, proxyUrl) {
  Object.assign(runtimeEnv.env, {
    HTTP_PROXY: proxyUrl,
    http_proxy: proxyUrl,
    NO_PROXY: "127.0.0.1,localhost,::1",
    no_proxy: "127.0.0.1,localhost,::1",
  });
}

async function provisionProvider(page, requestLog) {
  const created = await invokeAppServerFromPage(
    page,
    "modelProvider/create",
    {
      name: `CodeMode Gate B ${process.pid}`,
      providerType: PROVIDER_TYPE,
      apiHost: OFFICIAL_OPENAI_BASE_URL,
    },
    requestLog,
  );
  const providerId = String(created.result?.provider?.id || "").trim();
  assert(providerId, "modelProvider/create did not return provider.id");

  await invokeAppServerFromPage(
    page,
    "modelProvider/update",
    {
      providerId,
      enabled: true,
      sortOrder: 0,
      models: [modelConfig()],
    },
    requestLog,
  );
  await invokeAppServerFromPage(
    page,
    "modelProviderKey/create",
    {
      providerId,
      apiKey: PROVIDER_API_KEY,
      alias: "code-mode-gate-b",
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
  const models = Array.isArray(fetched.result?.models)
    ? fetched.result.models
    : [];
  const model = models.find((candidate) => candidate?.id === MODEL_NAME);
  assert(model, "modelProvider/fetchModels did not return the CodeMode model");
  return {
    providerId,
    model,
    source: fetched.result?.source || null,
  };
}

async function createThread(page, requestLog, providerId, workspaceRoot) {
  const response = await invokeAppServerFromPage(
    page,
    "thread/start",
    {
      cwd: workspaceRoot,
      historyMode: "paginated",
      model: MODEL_NAME,
      modelProvider: providerId,
      runtimeWorkspaceRoots: [workspaceRoot],
      serviceName: `CodeMode Gate B ${new Date().toISOString()}`,
      threadSource: "appServer",
    },
    requestLog,
  );
  const threadId = String(response.result?.thread?.id || "").trim();
  const sessionId = String(response.result?.thread?.sessionId || "").trim();
  assert(
    threadId && sessionId,
    "thread/start did not return canonical identity",
  );
  return { threadId, sessionId };
}

async function restoreThreadInGui(
  page,
  options,
  { providerId, sessionId, workspaceId },
) {
  await bindGuiWorkspaceAndModelPreferences(page, workspaceId, {
    sessionId,
    provider: providerId,
    model: MODEL_NAME,
  });
  await page.evaluate(
    ({ navigationKey, sessionId }) => {
      sessionStorage.setItem(
        navigationKey,
        JSON.stringify({
          page: "agent",
          params: { initialSessionId: sessionId },
        }),
      );
    },
    { navigationKey: NAVIGATION_RESTORE_STORAGE_KEY, sessionId },
  );
  await page.reload({
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  await waitForRendererReady(page, options);
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: options.timeoutMs });
  await page.waitForFunction(
    (activeSessionId) => {
      const textarea = document.querySelector(
        `textarea[name="agent-chat-message"][data-session-id="${activeSessionId}"]`,
      );
      return textarea instanceof HTMLTextAreaElement && !textarea.disabled;
    },
    sessionId,
    { timeout: options.timeoutMs },
  );
  await clearInvokeBuffers(page);
  return input;
}

async function submitTurnFromGui(page, input, options) {
  await input.fill(PROMPT);
  await page.waitForFunction(
    ({ prompt, sessionId }) => {
      const textarea = document.querySelector(
        `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
      );
      const core = textarea?.closest('[data-testid="inputbar-core-container"]');
      const send = core?.querySelector('[data-testid="send-btn"]');
      return (
        textarea instanceof HTMLTextAreaElement &&
        textarea.value === prompt &&
        send instanceof HTMLButtonElement &&
        !send.disabled
      );
    },
    {
      prompt: PROMPT,
      sessionId: await input.getAttribute("data-session-id"),
    },
    { timeout: options.timeoutMs },
  );
  const send = input
    .locator('xpath=ancestor::*[@data-testid="inputbar-core-container"]')
    .locator('[data-testid="send-btn"]');
  await send.click({ timeout: options.timeoutMs });
}

async function readThread(page, threadId, requestLog) {
  const response = await invokeAppServerFromPage(
    page,
    "thread/read",
    { threadId, includeTurns: true },
    requestLog,
  );
  return response.result?.thread || null;
}

function turnStatus(turn) {
  return String(turn?.status || "")
    .trim()
    .toLowerCase();
}

function findPromptTurn(thread) {
  const turns = Array.isArray(thread?.turns) ? thread.turns : [];
  return [...turns]
    .reverse()
    .find((turn) => JSON.stringify(turn).includes(PROMPT));
}

async function waitForTerminalThread(
  page,
  options,
  threadId,
  fixture,
  requestLog,
) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const thread = await readThread(page, threadId, requestLog);
    const turn = findPromptTurn(thread);
    last = {
      thread,
      turn,
      responseRequestCount: fixture.requests.filter(
        (request) => request.path === "/v1/responses",
      ).length,
    };
    if (
      turn &&
      TERMINAL_STATUSES.has(turnStatus(turn)) &&
      last.responseRequestCount >= 2
    ) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `CodeMode turn did not reach terminal state: ${sanitizeText(
      JSON.stringify({
        status: turnStatus(last?.turn),
        responseRequestCount: last?.responseRequestCount || 0,
      }),
    )}`,
  );
}

function itemToolName(item) {
  return String(item?.name || item?.tool || item?.toolName || "").trim();
}

function itemStatus(item) {
  return String(item?.status || "")
    .trim()
    .toLowerCase();
}

function readOuterExecItem(turn) {
  const items = Array.isArray(turn?.items) ? turn.items : [];
  return items.find((item) => itemToolName(item) === "exec") || null;
}

async function expandHistoricalToolRows(page) {
  const previews = page.locator(
    '[data-testid^="message-list-historical-timeline-preview:"]',
  );
  for (let index = 0; index < 10 && (await previews.count()) > 0; index += 1) {
    await previews.first().click();
  }
}

async function readVisibleState(page, options, sessionId) {
  const finalText = page.getByText(FINAL_TEXT, { exact: false }).first();
  await finalText.waitFor({ state: "visible", timeout: options.timeoutMs });
  await expandHistoricalToolRows(page);
  const rows = await page
    .locator('[data-testid="tool-call-row"]')
    .evaluateAll((nodes) =>
      nodes.map((node) => ({
        name: node.getAttribute("data-tool-name"),
        status: node.getAttribute("data-tool-status"),
        visible:
          window.getComputedStyle(node).display !== "none" &&
          window.getComputedStyle(node).visibility !== "hidden" &&
          node.getBoundingClientRect().height > 0,
        text: (node.textContent || "").trim().slice(0, 300),
      })),
    );
  return {
    activeSessionId: await page
      .locator(
        `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
      )
      .getAttribute("data-session-id"),
    finalAssistantTextVisible: await finalText.isVisible(),
    toolRows: rows,
  };
}

async function readInvokeDiagnostics(page) {
  return await page.evaluate(
    ({ errorKey, traceKey }) => {
      const readArray = (key) => {
        try {
          const value = JSON.parse(localStorage.getItem(key) || "[]");
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
            if (typeof message?.method !== "string") return [];
            return [
              {
                method: message.method,
                threadId: String(message?.params?.threadId || ""),
                transport: String(entry?.transport || ""),
                status: String(entry?.status || ""),
              },
            ];
          } catch {
            return [];
          }
        });
      });
      const mockFallbackHitCount = traces.filter((entry) => {
        if (entry?.mock === true || entry?.mockFallback === true) return true;
        return [entry?.transport, entry?.source, entry?.fallback].some(
          (value) =>
            typeof value === "string" && value.toLowerCase().includes("mock"),
        );
      }).length;
      return {
        calls,
        invokeErrorCount: readArray(errorKey).length,
        mockFallbackHitCount,
      };
    },
    { errorKey: INVOKE_ERROR_STORAGE_KEY, traceKey: INVOKE_TRACE_STORAGE_KEY },
  );
}

function requestTool(request, name) {
  const tools = Array.isArray(request?.body?.tools) ? request.body.tools : [];
  return tools.find((tool) => String(tool?.name || "").trim() === name) || null;
}

export function summarizeProviderEvidence(fixture) {
  const responses = fixture.requests.filter(
    (request) => request.path === "/v1/responses",
  );
  const first = responses[0] || null;
  const second = responses[1] || null;
  const execTool = requestTool(first, "exec");
  const output = customToolOutput(second?.body);
  return {
    discoveryRequestCount: fixture.modelRequests.length,
    discoveryUsedOfficialHost: fixture.modelRequests.some(
      (request) => request.host === "api.openai.com",
    ),
    responsesRequestCount: responses.length,
    responsesUsedOfficialHost: responses.every(
      (request) => request.host === "api.openai.com",
    ),
    firstRequestAdvertisedExec: Boolean(execTool),
    execToolType: execTool?.type || null,
    execFormatType: execTool?.format?.type || null,
    execFormatSyntax: execTool?.format?.syntax || null,
    secondRequestHasCustomToolOutput: Boolean(output),
    customToolOutputContainsMarker: String(output?.output || "").includes(
      EXEC_OUTPUT_MARKER,
    ),
    requestErrors: responses
      .map((request) => request.responseError)
      .filter(Boolean),
  };
}

export function buildAssertions({
  diagnostics,
  errors,
  model,
  outerExec,
  processEvidence,
  provider,
  rendererSnapshot,
  thread,
  turn,
  visible,
}) {
  const runtimeFeatures =
    model?.runtimeFeatures || model?.runtime_features || [];
  const toolMode = model?.toolMode || model?.tool_mode || null;
  const turnItems = Array.isArray(turn?.items) ? turn.items : [];
  const execRow = visible.toolRows.find((row) => row.name === "exec");
  const turnStartCall = diagnostics.calls.find(
    (call) =>
      call.method === "turn/start" &&
      call.transport === "electron-ipc" &&
      call.status === "success",
  );
  return {
    realElectronHost:
      rendererSnapshot?.electron === true &&
      rendererSnapshot?.hasInvokeBridge === true &&
      rendererSnapshot?.supportsAppServer === true,
    standaloneCodeModeHost:
      Number.isInteger(processEvidence?.electronPid) &&
      Number.isInteger(processEvidence?.appServerPid) &&
      Number.isInteger(processEvidence?.codeModeHostPid) &&
      new Set([
        processEvidence.electronPid,
        processEvidence.appServerPid,
        processEvidence.codeModeHostPid,
      ]).size === 3,
    codeModeHostOwnedByAppServer:
      processEvidence?.codeModeHostParentPid === processEvidence?.appServerPid,
    explicitCodeModeModel:
      toolMode === "code_mode" && runtimeFeatures.includes("custom_tools"),
    officialHostCapabilityPath:
      provider.discoveryUsedOfficialHost && provider.responsesUsedOfficialHost,
    providerAdvertisedCustomExec:
      provider.firstRequestAdvertisedExec &&
      provider.execToolType === "custom" &&
      provider.execFormatType === "grammar" &&
      provider.execFormatSyntax === "lark",
    providerReceivedCustomToolOutput:
      provider.secondRequestHasCustomToolOutput &&
      provider.customToolOutputContainsMarker,
    canonicalOuterExecCompleted:
      Boolean(outerExec) && itemStatus(outerExec) === "completed",
    publicCodeCellAbsent: !turnItems.some((item) =>
      String(item?.type || "")
        .toLowerCase()
        .includes("codecell"),
    ),
    turnCompleted: turnStatus(turn) === "completed",
    finalAssistantProjected:
      JSON.stringify(thread).includes(FINAL_TEXT) &&
      visible.finalAssistantTextVisible,
    execToolVisibleAndCompleted:
      Boolean(execRow?.visible) && execRow?.status === "completed",
    guiTurnUsedCurrentElectronBridge: Boolean(turnStartCall),
    productionMockFallbackZero: diagnostics.mockFallbackHitCount === 0,
    invokeErrorsZero: diagnostics.invokeErrorCount === 0,
    consoleErrorsZero: errors.console.length === 0 && errors.page.length === 0,
    providerRequestErrorsZero: provider.requestErrors.length === 0,
  };
}

export function readCodeModeProcessEvidence({
  electronPid,
  appServerBinary,
  platform = process.platform,
  runner = execFileSync,
} = {}) {
  const rows = readProcessTable({ platform, runner });
  const descendants = descendantProcessIds(rows, electronPid);
  const normalizedAppServer = normalizeProcessPath(appServerBinary, platform);
  const appServer = rows.find(
    (row) =>
      descendants.has(row.pid) &&
      normalizeProcessPath(row.command, platform).includes(normalizedAppServer),
  );
  const codeModeHostName =
    platform === "win32" ? "code-mode-host.exe" : "code-mode-host";
  const codeModeHost = rows.find(
    (row) =>
      descendants.has(row.pid) &&
      row.ppid === appServer?.pid &&
      normalizeProcessPath(row.command, platform).includes(
        normalizeProcessPath(codeModeHostName, platform),
      ),
  );
  return {
    electronPid: Number(electronPid) || null,
    appServerPid: appServer?.pid ?? null,
    appServerParentPid: appServer?.ppid ?? null,
    codeModeHostPid: codeModeHost?.pid ?? null,
    codeModeHostParentPid: codeModeHost?.ppid ?? null,
    appServerCommand: appServer?.command ?? null,
    codeModeHostCommand: codeModeHost?.command ?? null,
  };
}

function readProcessTable({ platform, runner }) {
  if (platform === "win32") {
    const output = String(
      runner(
        "powershell.exe",
        [
          "-NoProfile",
          "-Command",
          "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress",
        ],
        { encoding: "utf8" },
      ),
    ).trim();
    const parsed = output ? JSON.parse(output) : [];
    return (Array.isArray(parsed) ? parsed : [parsed]).map((row) => ({
      pid: Number(row.ProcessId),
      ppid: Number(row.ParentProcessId),
      command: String(row.CommandLine || ""),
    }));
  }
  const output = String(
    runner("ps", ["-axo", "pid=,ppid=,command="], { encoding: "utf8" }),
  );
  return output
    .split(/\r?\n/u)
    .map((line) => line.match(/^\s*(\d+)\s+(\d+)\s+(.*)$/u))
    .filter(Boolean)
    .map((match) => ({
      pid: Number(match[1]),
      ppid: Number(match[2]),
      command: match[3],
    }));
}

function descendantProcessIds(rows, rootPid) {
  const descendants = new Set([Number(rootPid)]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const row of rows) {
      if (!descendants.has(row.pid) && descendants.has(row.ppid)) {
        descendants.add(row.pid);
        changed = true;
      }
    }
  }
  return descendants;
}

function normalizeProcessPath(value, platform) {
  const normalized = String(value || "").replaceAll("\\", "/");
  return platform === "win32" ? normalized.toLowerCase() : normalized;
}

function writeEvidence(outputPath, evidence) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(evidence, null, 2)}\n`);
}

export async function runGateB(options) {
  const runtimeEnv = createTempRuntimeEnv();
  const requestLog = [];
  const errors = { console: [], page: [] };
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  let fixture = null;
  let electronHandle = null;
  try {
    logStage("fixture");
    fixture = await startCodeModeFixture();
    installProviderProxy(runtimeEnv, fixture.baseUrl);

    logStage("electron");
    electronHandle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors: errors.console,
      pageErrors: errors.page,
      backendMode: "runtime",
    });
    const { page, rendererSnapshot } = electronHandle;
    await initializeAppServer(page, requestLog);
    const workspace = await ensureDefaultWorkspace(page, requestLog);
    const workspaceRoot = String(workspace.rootPath || "").trim();
    assert(workspaceRoot, "workspace/default/ensure did not return rootPath");

    logStage("provider");
    const provisioned = await provisionProvider(page, requestLog);
    const identity = await createThread(
      page,
      requestLog,
      provisioned.providerId,
      workspaceRoot,
    );

    logStage("gui-turn");
    const input = await restoreThreadInGui(page, options, {
      providerId: provisioned.providerId,
      sessionId: identity.sessionId,
      workspaceId: workspace.workspaceId,
    });
    await submitTurnFromGui(page, input, options);
    const terminal = await waitForTerminalThread(
      page,
      options,
      identity.threadId,
      fixture,
      requestLog,
    );
    const visible = await readVisibleState(page, options, identity.sessionId);
    const diagnostics = await readInvokeDiagnostics(page);
    const provider = summarizeProviderEvidence(fixture);
    const outerExec = readOuterExecItem(terminal.turn);
    const processEvidence = readCodeModeProcessEvidence({
      electronPid: electronHandle.app.process().pid,
      appServerBinary,
    });

    const screenshotPath = path.join(
      path.dirname(options.output),
      "code-mode-electron-gate-b.png",
    );
    fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
    await page.screenshot({ path: screenshotPath, fullPage: true });

    const assertions = buildAssertions({
      diagnostics,
      errors,
      model: provisioned.model,
      outerExec,
      processEvidence,
      provider,
      rendererSnapshot,
      thread: terminal.thread,
      turn: terminal.turn,
      visible,
    });
    const failedAssertions = Object.entries(assertions)
      .filter(([, passed]) => !passed)
      .map(([name]) => name);
    const evidence = {
      schemaVersion: "v1",
      scenarioId: "code-mode-electron-gate-b",
      status: failedAssertions.length === 0 ? "pass" : "fail",
      generatedAt: new Date().toISOString(),
      proofLevel: "Gate B",
      claimBoundary:
        "real Electron host/preload/IPC/App Server runtime/standalone code-mode-host process/read model/visible DOM with a controlled Responses fixture; not a live-provider or cross-platform packaged parity claim",
      url: page.url(),
      identity: {
        sessionId: identity.sessionId,
        threadId: identity.threadId,
        turnId: terminal.turn?.id || null,
      },
      model: {
        id: provisioned.model?.id || null,
        source: provisioned.source,
        toolMode:
          provisioned.model?.toolMode || provisioned.model?.tool_mode || null,
        runtimeFeatures:
          provisioned.model?.runtimeFeatures ||
          provisioned.model?.runtime_features ||
          [],
      },
      provider,
      processes: processEvidence,
      canonical: {
        turnStatus: turnStatus(terminal.turn),
        outerExec: outerExec
          ? {
              id: outerExec.id || null,
              type: outerExec.type || null,
              name: itemToolName(outerExec),
              status: itemStatus(outerExec),
            }
          : null,
        publicItemTypes: (terminal.turn?.items || []).map(
          (item) => item?.type || null,
        ),
      },
      gui: {
        electron: rendererSnapshot.electron === true,
        activeSessionId: visible.activeSessionId,
        finalAssistantTextVisible: visible.finalAssistantTextVisible,
        toolRows: visible.toolRows,
        consoleErrors: errors.console.slice(0, 10),
        pageErrors: errors.page.slice(0, 10),
        screenshotPath: path.relative(process.cwd(), screenshotPath),
      },
      bridge: diagnostics,
      requestMethods: requestLog.map((entry) => entry.method),
      assertions,
      failedAssertions,
    };
    writeEvidence(options.output, evidence);
    console.log(`${LOG_PREFIX} evidence=${options.output}`);
    if (failedAssertions.length > 0) {
      throw new Error(
        `CodeMode Electron Gate B failed: ${failedAssertions.join(", ")}`,
      );
    }
    console.log(`${LOG_PREFIX} pass thread=${identity.threadId}`);
    return evidence;
  } finally {
    await closeElectronFixture(electronHandle);
    await fixture?.close().catch(() => undefined);
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  await runGateB(options);
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
