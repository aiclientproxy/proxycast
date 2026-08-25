#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { ensureElectronFixtureBuild } from "../lib/electron-fixture-build.mjs";
import { startOpenAiCompatibleFixtureServer } from "../lib/openai-compatible-fixture-server.mjs";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  LEGACY_MCP_COMMANDS,
  sanitizeJson,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  appServerCallFromPage,
  assert,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
  sanitizeText,
  sleep,
} from "./mcp-config-fixture-smoke.mjs";
import {
  cleanupServer,
  createAndStartMcpServer,
  createRepositoryProvider,
  createRuntimeThread,
  ensureWorkspace,
  openRuntimeThreadInGui,
  readElectronRuntime,
  startRuntimeTurn,
  waitForMcpTools,
  waitForTurnCompletion,
} from "./orchestrator-skills-gate-b.mjs";
import {
  APPS_CONNECTOR_ID,
  APPS_LINK_ID,
  APPS_RESOURCE_MARKER,
  APPS_RESOURCE_URI,
  APPS_SERVER_NAME,
  APPS_TOOL_NAME,
  parseOrchestratorGateArgs,
  readJsonLines,
  writeOrchestratorMcpFixture,
} from "./lib/orchestrator-skills-gate-b-core.mjs";

const LOG_PREFIX = "[smoke:mcp-resource-origin-gate-b]";
const FINAL_TEXT = "MCP_RESOURCE_ORIGIN_TURN_DONE";
const TOOL_CALL_ID = "call-mcp-resource-origin";
const FORGED_CONNECTOR_ID = "forged-renderer-connector";
const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "mcp-resource-origin-electron-gate-b",
  ),
  prefix: "mcp-resource-origin-electron-gate-b",
  timeoutMs: 240_000,
  intervalMs: 250,
  keepTemp: false,
};
const REQUIRED_FIRST_RUN_METHODS = [
  "workspace/default/ensure",
  "modelProvider/create",
  "modelProvider/update",
  "modelProviderKey/create",
  "model/list",
  "mcpServer/create",
  "mcpServer/start",
  "mcpTool/list",
  "thread/start",
  "thread/settings/update",
  "turn/start",
  "thread/read",
  "mcpServer/resource/read",
];
const REQUIRED_COLD_RUN_METHODS = [
  "mcpServer/start",
  "mcpTool/list",
  "thread/read",
  "mcpServer/resource/read",
];

function printHelp() {
  console.log(`
MCP Resource Origin Electron Gate B

用途:
  通过真实 Electron/preload/App Server/RuntimeCore/provider/MCP/read model 验证
  codex_apps canonical appContext、origin-scoped resource read 与冷恢复 GUI resource。

边界:
  使用 localhost OpenAI-compatible provider 与临时 stdio MCP fixture；不调用
  正式模型，不使用 App Server mock backend、renderer fallback 或 legacy MCP facade。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

export function findMcpResourceOriginItems(value, items = []) {
  if (Array.isArray(value)) {
    value.forEach((entry) => findMcpResourceOriginItems(entry, items));
    return items;
  }
  if (!value || typeof value !== "object") return items;
  if (
    value.type === "mcpToolCall" &&
    value.server === APPS_SERVER_NAME &&
    value.mcpAppResourceUri === APPS_RESOURCE_URI
  ) {
    items.push(value);
  }
  Object.values(value).forEach((entry) =>
    findMcpResourceOriginItems(entry, items),
  );
  return items;
}

function assertCanonicalOriginItem(item, runtime) {
  assert(item, "canonical read model 缺少 codex_apps resource Item");
  assert(item.id === `item_${TOOL_CALL_ID}`, "canonical Item identity 漂移");
  assert(item.status === "completed", "MCP origin Item 未完成");
  assert(item.tool === "apps_ping", "MCP origin tool identity 漂移");
  assert(item.appContext?.connectorId === APPS_CONNECTOR_ID, "connectorId 未进入 appContext");
  assert(item.appContext?.linkId === APPS_LINK_ID, "linkId 未进入 appContext");
  assert(item.appContext?.resourceUri === APPS_RESOURCE_URI, "resourceUri 未进入 appContext");
  assert(
    JSON.stringify(item.arguments || {}).includes(APPS_LINK_ID),
    "canonical arguments 缺少 link authority",
  );
  assert(
    JSON.stringify(runtime).includes(runtime.threadId) &&
      JSON.stringify(runtime).includes(runtime.turnId),
    "Thread/Turn identity 缺失",
  );
  return {
    appContextStable: true,
    itemId: item.id,
    status: item.status,
    tool: item.tool,
  };
}

function summarizeTrace(
  traceRaw,
  requiredMethods,
  runtime,
  observedMethods = new Set(),
) {
  const trace = parseInvokeTraceRaw(traceRaw);
  const requests = parseJsonRpcRequestsFromInvokeTrace(traceRaw);
  const commands = Array.from(
    new Set(trace.map((entry) => entry?.command).filter(Boolean)),
  );
  const electronRequests = requests.filter(
    (request) => request.transport === "electron-ipc",
  );
  const methods = Array.from(
    new Set([
      ...observedMethods,
      ...electronRequests.map((request) => request.method),
    ]),
  );
  const resourceReads = electronRequests.filter(
    (request) =>
      request.method === "mcpServer/resource/read" &&
      request.params?.threadId === runtime.threadId &&
      request.params?.originCallId === `item_${TOOL_CALL_ID}` &&
      request.params?.server === APPS_SERVER_NAME &&
      request.params?.uri === APPS_RESOURCE_URI,
  );
  const htmlLoads = trace.filter(
    (entry) =>
      entry?.command === "embedded_browser_view_load_html" &&
      entry?.transport === "electron-ipc" &&
      entry?.status === "success" &&
      entry?.args_preview?.source === "mcpApp" &&
      entry?.args_preview?.sourceUri === APPS_RESOURCE_URI,
  );
  return {
    appServerHandleJsonLinesSeen: commands.includes(
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    ),
    electronIpcSeen: electronRequests.length > 0,
    htmlLoadCount: htmlLoads.length,
    legacyCommands: LEGACY_MCP_COMMANDS.filter((command) =>
      commands.includes(command),
    ),
    methods,
    missingMethods: requiredMethods.filter((method) => !methods.includes(method)),
    mockFallbackHitCount: trace.filter(
      (entry) =>
        entry?.mock === true ||
        entry?.mockFallback === true ||
        (entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
          entry?.transport !== "electron-ipc"),
    ).length,
    resourceReadCount: resourceReads.length,
    resourceReadConnectorIds: resourceReads.map(
      (request) => request.params?.connectorId ?? null,
    ),
  };
}

async function readMcpAppWebContents(app) {
  return await app.evaluate(
    async ({ BrowserWindow, webContents }, marker) => {
      const windowContents = new Set(
        BrowserWindow.getAllWindows().map((window) => window.webContents.id),
      );
      return await Promise.all(
        webContents
          .getAllWebContents()
          .filter(
            (entry) =>
              !entry.isDestroyed() && !windowContents.has(entry.id),
          )
          .map(async (entry) => {
            let visibleMarker = null;
            try {
              visibleMarker = await entry.executeJavaScript(
                `document.body?.innerText?.includes(${JSON.stringify(marker)}) === true`,
              );
            } catch {
              visibleMarker = null;
            }
            return {
              id: entry.id,
              markerVisible: visibleMarker === true,
              title: entry.getTitle(),
              url: entry.getURL(),
            };
          }),
      );
    },
    APPS_RESOURCE_MARKER,
  );
}

async function waitForResourceSurface({ app, page, options, runtime }) {
  const tab = page.locator(
    '[data-testid="workspace-right-surface-tab-appSurface"]',
  );
  await tab.waitFor({ state: "visible", timeout: options.timeoutMs });
  await tab.click();

  const expectedViewId = `plugin-surface-mcp-app-item_${TOOL_CALL_ID}`;
  const frame = page.locator(
    `[data-testid="workspace-plugin-surface-frame"][data-view-id="${expectedViewId}"]`,
  );
  await frame.waitFor({ state: "visible", timeout: options.timeoutMs });

  const startedAt = Date.now();
  let latestTrace = null;
  let latestWebContents = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 60_000)) {
    const traceRaw =
      (await page.evaluate(() =>
        localStorage.getItem("lime_invoke_trace_buffer_v1"),
      )) || "";
    latestTrace = summarizeTrace(traceRaw, [], runtime);
    latestWebContents = await readMcpAppWebContents(app);
    if (
      latestTrace.resourceReadCount >= 1 &&
      latestTrace.htmlLoadCount >= 1 &&
      latestWebContents.some((entry) => entry.markerVisible)
    ) {
      return {
        frameMounted: (await frame.getAttribute("data-mounted")) === "true",
        trace: latestTrace,
        viewId: expectedViewId,
        webContentsMarkerVisible: true,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `MCP resource surface 未完成: ${JSON.stringify({ latestTrace, latestWebContents })}`,
  );
}

async function clearInvokeEvidence(page) {
  await page.evaluate(() => {
    localStorage.removeItem("lime_invoke_trace_buffer_v1");
    localStorage.removeItem("lime_invoke_error_buffer_v1");
  });
}

async function collectRunEvidence(
  page,
  requiredMethods,
  runtime,
  observedMethods = new Set(),
) {
  const traceRaw =
    (await page.evaluate(() =>
      localStorage.getItem("lime_invoke_trace_buffer_v1"),
    )) || "";
  const errorRaw =
    (await page.evaluate(() =>
      localStorage.getItem("lime_invoke_error_buffer_v1"),
    )) || "";
  const evidence = summarizeTrace(
    traceRaw,
    requiredMethods,
    runtime,
    observedMethods,
  );
  evidence.invokeErrorCount = parseInvokeTraceRaw(errorRaw).length;
  assert(evidence.appServerHandleJsonLinesSeen, "缺少 app_server_handle_json_lines");
  assert(evidence.electronIpcSeen, "缺少 electron-ipc transport");
  assert(
    evidence.missingMethods.length === 0,
    `缺少 current methods: ${evidence.missingMethods.join(", ")}`,
  );
  assert(evidence.legacyCommands.length === 0, "观察到 legacy MCP facade");
  assert(evidence.mockFallbackHitCount === 0, "观察到 production mock fallback");
  assert(evidence.invokeErrorCount === 0, "观察到 renderer invoke error");
  return evidence;
}

function summarizeResourceLedger(ledgerPath, runtime) {
  const ledger = readJsonLines(ledgerPath);
  const appReads = ledger.filter(
    (entry) =>
      entry?.type === "resource_read" && entry?.uri === APPS_RESOURCE_URI,
  );
  return {
    appReadCount: appReads.length,
    canonicalAuthorityOnEveryRead: appReads.every(
      (entry) =>
        entry.threadId === runtime.threadId &&
        entry.linkId === APPS_LINK_ID &&
        Array.isArray(entry.selectedConnectorIds) &&
        entry.selectedConnectorIds.length === 1 &&
        entry.selectedConnectorIds[0] === APPS_CONNECTOR_ID,
    ),
    distinctResourceProcessCount: new Set(
      appReads.map((entry) => entry.pid).filter(Number.isInteger),
    ).size,
    toolCallCount: ledger.filter(
      (entry) =>
        entry?.type === "tool_call" &&
        entry?.role === "apps" &&
        entry?.name === "apps_ping",
    ).length,
  };
}

export async function run() {
  const options = parseOrchestratorGateArgs(process.argv.slice(2), DEFAULTS);
  if (options.help) {
    printHelp();
    return;
  }

  ensureElectronFixtureBuild({ logPrefix: LOG_PREFIX, rootDir: process.cwd() });
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const rawPath = path.join(options.evidenceDir, `${options.prefix}-raw.json`);
  const firstScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-first.png`,
  );
  const coldScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-cold.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );
  const runtimeEnv = createTempRuntimeEnv();
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: {
      ...runtimeEnv.env,
      APP_SERVER_BIN: resolveDevAppServerBinary({
        env: runtimeEnv.env,
        repoRoot: process.cwd(),
        forceBuild: false,
      }),
    },
  });
  const summary = {
    schemaVersion: 1,
    scenarioId: "CODEX-ALIGN-mcp-resource-origin",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron/preload/IPC/App Server RuntimeCore provider turn produces a completed codex_apps canonical Item, Renderer opens its origin-scoped MCP App resource, and a second Electron/App Server process restores the same authority without rerunning the provider or tool. Local fixtures do not claim live provider, remote connector, compaction, Windows, or packaged behavior.",
    startedAt: new Date().toISOString(),
    completedAt: null,
    result: "fail",
    backendMode: "runtime",
    electronLaunchCount: 0,
    electronMainProcessChanged: false,
    canonical: null,
    firstRun: null,
    coldRestore: null,
    authority: null,
    errors: null,
    artifacts: {
      coldScreenshot: coldScreenshotPath,
      firstScreenshot: firstScreenshotPath,
      raw: rawPath,
      summary: summaryPath,
    },
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
  };
  const raw = {};
  const consoleErrors = [];
  const pageErrors = [];
  let provider = null;
  let handle = null;
  let page = null;
  let appsServer = null;

  try {
    logStage("start-local-runtime-fixtures");
    provider = await startOpenAiCompatibleFixtureServer({
      scriptedResponses: [
        {
          type: "tool_call",
          id: TOOL_CALL_ID,
          name: APPS_TOOL_NAME,
          arguments: {
            link_id: APPS_LINK_ID,
            message: "resource-origin",
          },
        },
        { type: "text", content: FINAL_TEXT },
      ],
    });
    const mcpFixture = writeOrchestratorMcpFixture(runtimeEnv.tempRoot);

    logStage("launch-first-real-electron-runtime");
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "runtime",
    });
    page = handle.page;
    const firstElectron = await readElectronRuntime(handle.app);
    summary.electronLaunchCount = 1;
    assert(handle.rendererSnapshot.electron, "真实 Electron renderer 未就绪");
    assert(handle.rendererSnapshot.hasInvokeBridge, "Electron preload invoke 未就绪");
    await clearInvokeEvidence(page);

    logStage("create-codex-apps-runtime-and-turn");
    const observedMethods = new Set();
    appsServer = await createAndStartMcpServer(
      page,
      mcpFixture,
      { name: APPS_SERVER_NAME, role: "apps" },
      observedMethods,
    );
    await waitForMcpTools(page, [APPS_TOOL_NAME], options, observedMethods);
    const workspace = await ensureWorkspace(page, observedMethods);
    const route = await createRepositoryProvider(
      page,
      provider,
      "MCP Resource Origin Gate B",
      observedMethods,
    );
    let runtime = await createRuntimeThread(
      page,
      workspace,
      route,
      "MCP resource origin Gate B",
      observedMethods,
    );
    await openRuntimeThreadInGui(page, runtime, options);
    runtime = await startRuntimeTurn(
      page,
      runtime,
      workspace,
      "Call the calendar app resource origin probe and confirm completion.",
      [APPS_TOOL_NAME],
      observedMethods,
    );
    const completed = await waitForTurnCompletion(
      page,
      runtime,
      FINAL_TEXT,
      provider,
      options,
      observedMethods,
    );
    const firstItems = findMcpResourceOriginItems(completed.result);
    assert(firstItems.length === 1, "首轮 canonical MCP origin Item 数量异常");
    const canonical = assertCanonicalOriginItem(firstItems[0], runtime);

    logStage("open-gui-resource-and-prove-renderer-connector-is-not-authority");
    const firstSurface = await waitForResourceSurface({
      app: handle.app,
      page,
      options,
      runtime,
    });
    const spoofedRead = await appServerCallFromPage(
      page,
      "mcpServer/resource/read",
      {
        threadId: runtime.threadId,
        originCallId: canonical.itemId,
        server: APPS_SERVER_NAME,
        uri: APPS_RESOURCE_URI,
        connectorId: FORGED_CONNECTOR_ID,
      },
    );
    observedMethods.add(spoofedRead.method);
    assert(
      spoofedRead.result?.originCallId === canonical.itemId &&
        JSON.stringify(spoofedRead.result).includes(APPS_RESOURCE_MARKER),
      "origin read 未返回 canonical resource",
    );
    let mismatchedUriError = null;
    try {
      await appServerCallFromPage(page, "mcpServer/resource/read", {
        threadId: runtime.threadId,
        originCallId: canonical.itemId,
        server: APPS_SERVER_NAME,
        uri: "ui://calendar/wrong.html",
        connectorId: APPS_CONNECTOR_ID,
      });
    } catch (error) {
      mismatchedUriError = String(error);
    }
    assert(
      mismatchedUriError?.includes("does not match the requested resource"),
      "mismatched origin URI 未 fail closed",
    );
    await page.screenshot({ path: firstScreenshotPath, fullPage: true });
    const firstBridge = await collectRunEvidence(
      page,
      REQUIRED_FIRST_RUN_METHODS,
      runtime,
      observedMethods,
    );
    const providerRequestCountBeforeRestart = provider.requests.length;
    const ledgerBeforeRestart = summarizeResourceLedger(
      mcpFixture.ledgerPath,
      runtime,
    );
    assert(
      ledgerBeforeRestart.appReadCount >= 2 &&
        ledgerBeforeRestart.canonicalAuthorityOnEveryRead,
      "首轮 MCP wire 未使用 canonical connector/link authority",
    );

    logStage("restart-electron-and-app-server");
    await closeElectronFixture(handle);
    handle = null;
    page = null;
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "runtime",
    });
    page = handle.page;
    const coldElectron = await readElectronRuntime(handle.app);
    summary.electronLaunchCount = 2;
    summary.electronMainProcessChanged = firstElectron.pid !== coldElectron.pid;
    assert(summary.electronMainProcessChanged, "cold restore 未启动新的 Electron 主进程");
    await clearInvokeEvidence(page);
    const coldObservedMethods = new Set();
    const coldStart = await appServerCallFromPage(page, "mcpServer/start", {
      name: APPS_SERVER_NAME,
    });
    coldObservedMethods.add(coldStart.method);
    await waitForMcpTools(
      page,
      [APPS_TOOL_NAME],
      options,
      coldObservedMethods,
    );

    logStage("restore-canonical-item-and-visible-resource");
    const coldRead = await appServerCallFromPage(page, "thread/read", {
      threadId: runtime.threadId,
      includeTurns: true,
    });
    coldObservedMethods.add(coldRead.method);
    const coldItems = findMcpResourceOriginItems(coldRead.result);
    assert(coldItems.length === 1, "cold restore canonical MCP Item 数量异常");
    const coldCanonical = assertCanonicalOriginItem(coldItems[0], runtime);
    assert(coldCanonical.itemId === canonical.itemId, "cold restore Item identity 漂移");
    await openRuntimeThreadInGui(page, runtime, options);
    const coldSurface = await waitForResourceSurface({
      app: handle.app,
      page,
      options,
      runtime,
    });
    await page.screenshot({ path: coldScreenshotPath, fullPage: true });
    const coldBridge = await collectRunEvidence(
      page,
      REQUIRED_COLD_RUN_METHODS,
      runtime,
      coldObservedMethods,
    );
    const ledger = summarizeResourceLedger(mcpFixture.ledgerPath, runtime);
    assert(
      ledger.appReadCount >= 3 &&
        ledger.canonicalAuthorityOnEveryRead &&
        ledger.distinctResourceProcessCount >= 2,
      `cold resource authority 不完整: ${JSON.stringify(ledger)}`,
    );
    assert(ledger.toolCallCount === 1, "cold restore 偷偷重跑了 MCP tool");
    assert(
      provider.requests.length === providerRequestCountBeforeRestart,
      "cold restore 偷偷重跑了 provider turn",
    );
    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );
    assert(
      pageErrors.length === 0,
      `观察到 page error: ${pageErrors.join(" | ")}`,
    );

    summary.canonical = canonical;
    summary.firstRun = {
      bridge: firstBridge,
      surface: firstSurface,
    };
    summary.coldRestore = {
      bridge: coldBridge,
      canonicalIdentityStable: coldCanonical.itemId === canonical.itemId,
      providerNotReexecuted: true,
      surface: coldSurface,
      toolNotReexecuted: ledger.toolCallCount === 1,
    };
    summary.authority = {
      ...ledger,
      forgedRendererConnectorIgnored: true,
      mismatchedUriRejected: true,
    };
    summary.errors = {
      consoleErrorCount: consoleErrors.length,
      invokeErrorCount:
        firstBridge.invokeErrorCount + coldBridge.invokeErrorCount,
      pageErrorCount: pageErrors.length,
    };
    raw.bridge = sanitizeJson({ first: firstBridge, cold: coldBridge });
    raw.ledger = sanitizeJson(ledger);
    raw.surface = sanitizeJson({ first: firstSurface, cold: coldSurface });
    summary.result = "pass";
  } catch (error) {
    summary.error = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    if (page) {
      await page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
      summary.artifacts.failureScreenshot = failureScreenshotPath;
    }
    throw error;
  } finally {
    summary.completedAt = new Date().toISOString();
    summary.errors ??= {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    writeJsonFile(rawPath, raw);
    writeJsonFile(summaryPath, summary);
    await cleanupServer(page, appsServer);
    await closeElectronFixture(handle);
    await provider?.close().catch(() => undefined);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }

  console.log(`${LOG_PREFIX} pass summary=${summaryPath}`);
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  run().catch((error) => {
    console.error(
      `${LOG_PREFIX} failed: ${error instanceof Error ? error.message : String(error)}`,
    );
    process.exitCode = 1;
  });
}
