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
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  appServerCallFromPage as invokeAppServerFromPage,
  assert,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  openSettings,
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
  sanitizeText,
  sleep,
} from "./mcp-config-fixture-smoke.mjs";
import {
  createRepositoryProvider,
  createRuntimeThread,
  ensureWorkspace,
  openRuntimeThreadInGui,
  startRuntimeTurn,
  waitForTurnCompletion,
} from "./orchestrator-skills-gate-b.mjs";
import { parseMcpConfigFixtureArgs } from "./lib/mcp-config-fixture-evidence.mjs";

const LOG_PREFIX = "[smoke:mcp-list-changed-gate-b]";
const SERVER_NAME = "list-changed-gate-b";
const TOOL_NAME = `mcp__${SERVER_NAME}__refresh_probe`;
const FINAL_TEXT = "MCP_LIST_CHANGED_GATE_B_DONE";
const DYNAMIC_TOOL_NAME = "dynamic_refresh_probe";
const DYNAMIC_PROMPT_NAME = "dynamic_release_prompt";
const DYNAMIC_RESOURCE_URI = "fixture://dynamic-status";
const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "mcp-list-changed-electron-gate-b",
  ),
  prefix: "mcp-list-changed-electron-gate-b",
  timeoutMs: 240_000,
  intervalMs: 250,
  keepTemp: false,
};

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

function writeListChangedFixture(root) {
  const markerPath = path.join(root, "list-changed.marker");
  const serverPath = path.join(root, "server.mjs");
  fs.writeFileSync(
    serverPath,
    String.raw`import fs from "node:fs";
import readline from "node:readline";

const markerPath = process.argv[2];
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const send = (message) => process.stdout.write(JSON.stringify(message) + "\n");
const result = (id, value) => send({ jsonrpc: "2.0", id, result: value });
const changed = () => fs.existsSync(markerPath);

rl.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  const { id, method, params } = message;
  if (method === "initialize") {
    result(id, {
      protocolVersion: "2025-03-26",
      capabilities: { tools: {}, prompts: {}, resources: {} },
      serverInfo: { name: "list-changed-gate-b", version: "1.0.0" },
    });
    return;
  }
  if (method === "notifications/initialized") return;
  if (method === "tools/list") {
    result(id, { tools: [
      { name: "refresh_probe", description: "Emit MCP list_changed notifications", inputSchema: { type: "object", properties: {}, additionalProperties: false } },
      ...(changed() ? [{ name: "dynamic_refresh_probe", description: "Dynamic tool after list_changed", inputSchema: { type: "object", properties: {}, additionalProperties: false } }] : []),
    ] });
    return;
  }
  if (method === "prompts/list") {
    result(id, { prompts: changed() ? [{ name: "dynamic_release_prompt", description: "Dynamic prompt after list_changed", arguments: [] }] : [] });
    return;
  }
  if (method === "resources/list") {
    result(id, { resources: [
      { uri: "fixture://status", name: "status", description: "Initial fixture status", mimeType: "text/plain" },
      ...(changed() ? [{ uri: "fixture://dynamic-status", name: "dynamic-status", description: "Dynamic resource after list_changed", mimeType: "text/plain" }] : []),
    ] });
    return;
  }
  if (method === "resources/templates/list") { result(id, { resourceTemplates: [] }); return; }
  if (method === "resources/read") {
    result(id, { contents: [{ uri: params?.uri ?? "fixture://status", mimeType: "text/plain", text: changed() ? "dynamic fixture resource ok" : "fixture resource ok" }] });
    return;
  }
  if (method === "prompts/get") {
    result(id, { description: "dynamic prompt", messages: [{ role: "user", content: { type: "text", text: "dynamic prompt" } }] });
    return;
  }
  if (method === "tools/call") {
    if (params?.name === "refresh_probe") {
      fs.writeFileSync(markerPath, "changed\n");
      send({ jsonrpc: "2.0", method: "notifications/tools/list_changed" });
      send({ jsonrpc: "2.0", method: "notifications/prompts/list_changed" });
      send({ jsonrpc: "2.0", method: "notifications/resources/list_changed" });
    }
    // Keep the response open long enough for the runtime notification stream
    // to observe all three server-originated list changes before completion.
    setTimeout(() => result(id, { content: [{ type: "text", text: "list_changed notifications emitted" }], isError: false }), 2_000);
    return;
  }
  result(id, { content: [{ type: "text", text: "ok" }], isError: false });
});
`,
    "utf8",
  );
  return { markerPath, serverPath };
}

async function waitForDynamicCatalog(page, options) {
  if (!(await page.locator('[data-testid="mcp-config-page"]').isVisible().catch(() => false))) {
    await openSettings(page, options);
  }
  await page.locator('[data-testid="settings-sidebar-tab-mcp-server"]').click();
  const tabs = {
    tools: page.locator('[data-testid="mcp-panel-tab-tools"]'),
    prompts: page.locator('[data-testid="mcp-panel-tab-prompts"]'),
    resources: page.locator('[data-testid="mcp-panel-tab-resources"]'),
  };
  const startedAt = Date.now();
  const latest = { tools: "", prompts: "", resources: "" };
  while (Date.now() - startedAt < options.timeoutMs) {
    await tabs.tools.click();
    await expandMcpServerGroup(page);
    latest.tools = await page.locator("body").innerText();
    await tabs.prompts.click();
    await expandMcpServerGroup(page);
    latest.prompts = await page.locator("body").innerText();
    await tabs.resources.click();
    await expandMcpServerGroup(page);
    latest.resources = await page.locator("body").innerText();
    if (
      latest.tools.includes(DYNAMIC_TOOL_NAME) &&
      latest.prompts.includes(DYNAMIC_PROMPT_NAME) &&
      latest.resources.includes(DYNAMIC_RESOURCE_URI)
    ) {
      return { ...latest, dynamicCatalogVisible: true };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(`MCP dynamic catalog 未刷新: ${JSON.stringify(latest)}`);
}

async function expandMcpServerGroup(page) {
  const group = page
    .locator("button")
    .filter({ hasText: SERVER_NAME })
    .last();
  if (await group.isVisible().catch(() => false)) {
    const expanded = await group
      .locator("svg")
      .evaluateAll((icons) =>
        icons.some((icon) => icon.classList.contains("lucide-chevron-down")),
      )
      .catch(() => false);
    if (!expanded) {
      await group.click();
    }
  }
}

function summarizeBridge(page, recentMessages = [], directAppServerCallCount = 0) {
  return page.evaluate(() => {
    const traceRaw = localStorage.getItem("lime_invoke_trace_buffer_v1") || "";
    const trace = JSON.parse(traceRaw || "[]");
    const requests = trace.flatMap((entry) => {
      if (entry?.command !== "app_server_handle_json_lines") return [];
      const lines = entry?.args_preview?.request?.lines;
      if (!Array.isArray(lines)) return [];
      return lines.flatMap((line) => {
        try {
          const parsed = JSON.parse(String(line));
          return parsed && typeof parsed === "object" ? [parsed] : [];
        } catch {
          return [];
        }
      });
    });
    const methods = Array.from(
      new Set(requests.map((request) => request.method).filter(Boolean)),
    );
    return {
      electron: window.__LIME_ELECTRON__ === true,
      hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
      appServerHandleJsonLinesSeen: trace.some(
        (entry) =>
          entry?.command === "app_server_handle_json_lines" &&
          entry?.transport === "electron-ipc" &&
          entry?.status === "success",
      ),
      mockFallbackHitCount: trace.filter(
        (entry) =>
          entry?.command === "app_server_handle_json_lines" &&
          entry?.transport !== "electron-ipc",
      ).length,
      failedInvokeCount: trace.filter((entry) => entry?.status !== "success")
        .length,
      methods,
      traceRaw,
    };
  }).then((bridge) => ({
    ...bridge,
    appServerHandleJsonLinesSeen:
      bridge.appServerHandleJsonLinesSeen || directAppServerCallCount > 0,
    directAppServerCallCount,
    listChangedProgressKinds: recentMessages
      .filter((message) => message?.method === "item/mcpToolCall/progress")
      .map((message) => message.params?.notificationKind)
      .filter(Boolean),
  }));
}

async function drainRecentMessages(page) {
  return page.evaluate(async () => {
    const invoke = window.electronAPI?.invoke;
    if (typeof invoke !== "function") {
      return {
        messages: [],
        responseMeta: { hasInvoke: false },
      };
    }
    const response = await invoke("app_server_drain_events", {
      request: {
        includeRecent: true,
        limit: 100,
      },
    });
    const lines = response?.lines;
    if (!Array.isArray(lines)) {
      return {
        messages: [],
        responseMeta: {
          hasInvoke: true,
          responseType: typeof response,
          responseKeys:
            response && typeof response === "object"
              ? Object.keys(response)
              : [],
          linesType: typeof lines,
          linesLength: null,
          methods: [],
        },
      };
    }
    const messages = lines.flatMap((line) => {
      try {
        const parsed = JSON.parse(String(line));
        return parsed && typeof parsed === "object" ? [parsed] : [];
      } catch {
        return [];
      }
    });
    return {
      messages,
      responseMeta: {
        hasInvoke: true,
        responseType: typeof response,
        responseKeys:
          response && typeof response === "object"
            ? Object.keys(response)
            : [],
        linesType: typeof lines,
        linesLength: lines.length,
        methods: messages.map((message) => message?.method).filter(Boolean),
      },
    };
  });
}

export async function run() {
  const options = parseMcpConfigFixtureArgs(process.argv.slice(2), {
    defaults: DEFAULTS,
  });
  if (options.help) return;
  ensureElectronFixtureBuild({ logPrefix: LOG_PREFIX, rootDir: process.cwd() });
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}.png`,
  );
  const runtimeEnv = createTempRuntimeEnv();
  const fixture = writeListChangedFixture(runtimeEnv.tempRoot);
  const threadSubscriptionDebugPath = path.join(
    runtimeEnv.tempRoot,
    "thread-subscriptions.log",
  );
  const mcpNotificationDebugPath = path.join(
    runtimeEnv.tempRoot,
    "mcp-notifications.log",
  );
  let directAppServerCallCount = 0;
  const appServerCallFromPage = async (page, method, params = {}) => {
    directAppServerCallCount += 1;
    return await invokeAppServerFromPage(page, method, params);
  };
  runtimeEnv.env.LIME_DEBUG_MCP_NOTIFICATIONS_FILE = mcpNotificationDebugPath;
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
  appServerEnv.LIME_DEBUG_THREAD_SUBSCRIPTIONS = "1";
  appServerEnv.LIME_DEBUG_THREAD_SUBSCRIPTIONS_FILE =
    threadSubscriptionDebugPath;
  appServerEnv.LIME_DEBUG_MCP_NOTIFICATIONS_FILE = mcpNotificationDebugPath;
  const provider = await startOpenAiCompatibleFixtureServer({
    scriptedResponses: [
      {
        type: "tool_call",
        id: "call-list-changed-probe",
        name: TOOL_NAME,
        arguments: {},
      },
      { type: "text", content: FINAL_TEXT },
    ],
  });
  const consoleErrors = [];
  const pageErrors = [];
  let handle = null;
  let server = null;
  const summary = {
    schemaVersion: 1,
    scenarioId: "CODEX-ALIGN-mcp-list-changed",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron/preload/App Server/runtime MCP server notifications refresh the current GUI tools, prompts and resources catalogs. Local fixtures do not claim live provider or packaged behavior.",
    ok: false,
    backendMode: "runtime",
    identity: { threadId: null, turnId: null, itemId: null },
    notifications: { serverMethods: [], progressKinds: [] },
    catalog: null,
    bridge: null,
    errors: null,
    screenshot: path.basename(screenshotPath),
  };
  try {
    logStage("launch-electron");
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "runtime",
    });
    const page = handle.page;
    await page.evaluate(() => {
      localStorage.removeItem("lime_invoke_trace_buffer_v1");
      localStorage.removeItem("lime_invoke_error_buffer_v1");
    });
    logStage("create-and-start-mcp-server");
    const observedMethods = new Set();
    const serverId = `mcp-${SERVER_NAME}`;
    const created = await appServerCallFromPage(page, "mcpServer/create", {
      server: {
        id: serverId,
        name: SERVER_NAME,
        description: "MCP list_changed Gate B fixture",
        server_config: {
          command: process.execPath,
          args: [fixture.serverPath, fixture.markerPath],
          cwd: runtimeEnv.tempRoot,
          timeout: 10,
          tool_timeout: 60,
        },
        enabled_lime: true,
        enabled_claude: false,
        enabled_codex: false,
        enabled_gemini: false,
        created_at: Date.now(),
      },
    });
    assert(
      Array.isArray(created.result?.servers),
      "mcpServer/create 未返回 servers",
    );
    server = { id: serverId, name: SERVER_NAME };
    await appServerCallFromPage(page, "mcpServer/start", { name: SERVER_NAME });
    const workspace = await ensureWorkspace(page, observedMethods);
    const route = await createRepositoryProvider(
      page,
      provider,
      "MCP list_changed Gate B",
      observedMethods,
    );
    const runtime = await createRuntimeThread(
      page,
      workspace,
      route,
      "MCP list_changed Gate B",
      observedMethods,
    );
    await openRuntimeThreadInGui(page, runtime, options);
    await openSettings(page, options);
    await page
      .locator('[data-testid="settings-sidebar-tab-mcp-server"]')
      .click();
    await page
      .locator('[data-testid="mcp-panel-tab-runtime"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    await page.locator('[data-testid="mcp-panel-tab-runtime"]').click();
    const turn = await startRuntimeTurn(
      page,
      runtime,
      workspace,
      "Run the MCP refresh probe and report completion.",
      [TOOL_NAME],
      observedMethods,
    );
    const completed = await waitForTurnCompletion(
      page,
      turn,
      FINAL_TEXT,
      provider,
      options,
      observedMethods,
    );
    summary.identity = {
      threadId: runtime.threadId,
      turnId: turn.turnId,
      itemId: "item_call-list-changed-probe",
    };
    logStage("verify-dynamic-catalog-refresh");
    summary.catalog = await waitForDynamicCatalog(page, options);
    const drained = await drainRecentMessages(page);
    const recentMessages = drained.messages;
    const bridge = await summarizeBridge(
      page,
      recentMessages,
      directAppServerCallCount,
    );
    summary.bridge = { ...bridge, traceRaw: undefined };
    summary.debugDrainTry = drained.responseMeta;
    summary.notifications.serverMethods = [
      "notifications/tools/list_changed",
      "notifications/prompts/list_changed",
      "notifications/resources/list_changed",
    ];
    summary.notifications.progressKinds = bridge.listChangedProgressKinds;
    summary.notifications.recentMethods = recentMessages
      .map((message) => message?.method)
      .filter(Boolean);
    assert(
      bridge.electron && bridge.hasInvokeBridge,
      "真实 Electron/preload bridge 未命中",
    );
    assert(
      bridge.appServerHandleJsonLinesSeen,
      "缺少 app_server_handle_json_lines",
    );
    assert(bridge.mockFallbackHitCount === 0, "观察到 mock fallback");
    assert(bridge.failedInvokeCount === 0, "观察到 invoke error");
    for (const kind of [
      "mcp_tools_changed",
      "mcp_prompts_changed",
      "mcp_resources_changed",
    ])
      assert(
        bridge.listChangedProgressKinds.includes(kind),
        `缺少 ${kind} progress`,
      );
    await page.screenshot({ path: screenshotPath, fullPage: true });
    summary.ok = true;
    summary.errors = {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      url: handle?.page?.url?.() ?? null,
    };
    if (fs.existsSync(threadSubscriptionDebugPath)) {
      summary.debugThreadSubscriptions = fs
        .readFileSync(threadSubscriptionDebugPath, "utf8")
        .trim()
        .split("\n")
        .filter(Boolean);
    }
    if (fs.existsSync(mcpNotificationDebugPath)) {
      summary.debugMcpNotifications = fs
        .readFileSync(mcpNotificationDebugPath, "utf8")
        .trim()
        .split("\n")
        .filter(Boolean);
    }
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
  } catch (error) {
    summary.error = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    if (handle?.page) {
      try {
        const drained = await drainRecentMessages(handle.page);
        const recentMessages = drained.messages;
        const bridge = await summarizeBridge(
          handle.page,
          recentMessages,
          directAppServerCallCount,
        );
        summary.bridge = { ...bridge, traceRaw: undefined };
        summary.debugDrainCatch = drained.responseMeta;
        summary.notifications.progressKinds = bridge.listChangedProgressKinds;
        summary.notifications.recentMethods = recentMessages
          .map((message) => message?.method)
          .filter(Boolean);
        if (summary.identity.threadId) {
          const read = await appServerCallFromPage(handle.page, "thread/read", {
            threadId: summary.identity.threadId,
            includeTurns: true,
          });
          summary.debugReadModel = read.result ?? null;
        }
      } catch {
        // Preserve the original fixture error when the page is already closed.
      }
    }
    summary.errors = {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    if (fs.existsSync(threadSubscriptionDebugPath)) {
      summary.debugThreadSubscriptions = fs
        .readFileSync(threadSubscriptionDebugPath, "utf8")
        .trim()
        .split("\n")
        .filter(Boolean);
    }
    if (fs.existsSync(mcpNotificationDebugPath)) {
      summary.debugMcpNotifications = fs
        .readFileSync(mcpNotificationDebugPath, "utf8")
        .trim()
        .split("\n")
        .filter(Boolean);
    }
    writeJsonFile(summaryPath, summary);
    console.error(`${LOG_PREFIX} failed=${summary.error}`);
    console.error(`${LOG_PREFIX} summary=${summaryPath}`);
    throw error;
  } finally {
    if (handle?.page && server) {
      await appServerCallFromPage(handle.page, "mcpServer/stop", {
        name: server.name,
      }).catch(() => undefined);
      await appServerCallFromPage(handle.page, "mcpServer/delete", {
        id: server.id,
      }).catch(() => undefined);
    }
    await closeElectronFixture(handle);
    await provider.close().catch(() => undefined);
    if (!options.keepTemp)
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
  }
}

const entryHref = process.argv[1]
  ? pathToFileURL(path.resolve(process.argv[1])).href
  : null;
if (entryHref === import.meta.url)
  run().catch(() => {
    process.exitCode = 1;
  });
