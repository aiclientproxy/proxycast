#!/usr/bin/env node

import { once } from "node:events";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { WebSocketServer } from "ws";

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

export const ENVIRONMENT_LIFECYCLE_ID = "remote-lifecycle-gate-b";
export const ENVIRONMENT_LIFECYCLE_TITLE =
  "Environment lifecycle Gate B canonical thread";
export const ENVIRONMENT_LIFECYCLE_REQUIRED_METHODS = [
  "environment/add",
  "environment/status",
  "thread/start",
];
export const ENVIRONMENT_LIFECYCLE_OPEN_METHODS = [
  "thread/read",
  "thread/resume",
];
const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
const LOG_PREFIX = "[smoke:environment-lifecycle-gate-b]";
const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "environment-lifecycle-electron-gate-b",
  ),
  prefix: "environment-lifecycle-electron-gate-b",
  timeoutMs: 120_000,
  intervalMs: 200,
  keepTemp: false,
};
const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function printHelp() {
  console.log(`
Environment lifecycle Electron Gate B

用途:
  通过真实 Electron、App Server 和本地 exec-server fixture 创建远端 Environment，
  验证 GUI 可见的 connected -> disconnected -> reconnected 生命周期。

边界:
  使用 unavailable backend，不启动 Turn、不调用模型；fixture 只承接 Codex exec-server
  initialize/environment info/status，不使用 mock backend 或 renderer fallback。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function parseEnvironmentLifecycleGateArgs(argv, defaults = DEFAULTS) {
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

async function startRemoteEnvironmentFixture() {
  const requests = [];
  const sockets = [];
  const pendingReconnectResponses = [];
  let reconnectReleased = false;
  const server = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await once(server, "listening");
  server.on("connection", (socket) => {
    const connection = sockets.length + 1;
    sockets.push(socket);
    socket.on("message", (data) => {
      let request;
      try {
        request = JSON.parse(data.toString("utf8"));
      } catch {
        return;
      }
      const method = typeof request?.method === "string" ? request.method : "";
      requests.push({ connection, method });
      if (!Object.prototype.hasOwnProperty.call(request ?? {}, "id")) {
        return;
      }
      const respond = () => {
        if (socket.readyState !== 1) return;
        let result;
        switch (method) {
          case "initialize":
            result = { sessionId: `environment-gate-b-${connection}` };
            break;
          case "environment/info":
            result = {
              shell: { name: "fixture-sh", path: "/bin/fixture-sh" },
              cwd: "file:///remote/workspace",
            };
            break;
          case "environment/status":
            result = { status: "ready" };
            break;
          default:
            socket.send(
              JSON.stringify({
                jsonrpc: "2.0",
                id: request.id,
                error: { code: -32601, message: `unsupported method: ${method}` },
              }),
            );
            return;
        }
        socket.send(
          JSON.stringify({ jsonrpc: "2.0", id: request.id, result }),
        );
      };
      if (connection > 1 && !reconnectReleased) {
        pendingReconnectResponses.push(respond);
        return;
      }
      respond();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("remote Environment fixture did not bind a TCP port");
  }
  return {
    url: `ws://127.0.0.1:${address.port}`,
    requests,
    connectionCount: () => sockets.length,
    disconnectFirst: () => sockets[0]?.terminate(),
    releaseReconnect: () => {
      reconnectReleased = true;
      for (const respond of pendingReconnectResponses.splice(0)) respond();
    },
    close: async () => {
      for (const socket of sockets) socket.terminate();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

function requestSummary(request) {
  return {
    method: request.method,
    status: request.status,
    transport: request.transport,
  };
}

export function summarizeEnvironmentLifecycleEvidence({
  traceRaw,
  errorRaw,
  stages,
  threadId,
  remoteRequests = [],
  setupRequests = [],
}) {
  const trace = parseInvokeTraceRaw(traceRaw);
  const requests = [
    ...setupRequests,
    ...parseJsonRpcRequestsFromInvokeTrace(traceRaw),
  ];
  const electronRequests = requests.filter(
    (request) =>
      request.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
      request.transport === "electron-ipc" &&
      request.status === "success",
  );
  const methods = Array.from(
    new Set(electronRequests.map((request) => request.method)),
  );
  const openRequests = electronRequests.filter(
    (request) =>
      ENVIRONMENT_LIFECYCLE_OPEN_METHODS.includes(request.method) &&
      request.params?.threadId === threadId,
  );
  const environmentRequests = requests.filter((request) =>
    ["environment/add", "environment/status"].includes(request.method),
  );
  const matchingEnvironmentRequests = environmentRequests.filter(
    (request) =>
      request.transport === "electron-ipc" &&
      request.status === "success" &&
      request.params?.environmentId === ENVIRONMENT_LIFECYCLE_ID,
  );
  const relevantTrace = trace.filter((entry) =>
    [
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      APP_SERVER_DRAIN_EVENTS_COMMAND,
    ].includes(entry?.command),
  );
  const remoteConnections = Array.from(
    new Set(remoteRequests.map((request) => request.connection)),
  );
  return {
    identity: {
      threadId,
      canonicalThreadOpenHitCount: openRequests.length,
      environmentRequestsMatchIdentity:
        environmentRequests.length > 0 &&
        environmentRequests.length === matchingEnvironmentRequests.length,
    },
    bridge: {
      methods,
      missingMethods: ENVIRONMENT_LIFECYCLE_REQUIRED_METHODS.filter(
        (method) => !methods.includes(method),
      ),
      drainEventsSeen: relevantTrace.some(
        (entry) => entry.command === APP_SERVER_DRAIN_EVENTS_COMMAND,
      ),
      electronIpcHitCount: relevantTrace.filter(
        (entry) => entry.transport === "electron-ipc",
      ).length,
      mockFallbackHitCount: relevantTrace.filter(
        (entry) => entry.transport !== "electron-ipc",
      ).length,
      failedInvokeCount: relevantTrace.filter(
        (entry) => entry.status !== "success",
      ).length,
    },
    gui: stages,
    remote: {
      connectionCount: remoteConnections.length,
      methodsByConnection: remoteConnections.map((connection) => ({
        connection,
        methods: Array.from(
          new Set(
            remoteRequests
              .filter((request) => request.connection === connection)
              .map((request) => request.method),
          ),
        ),
      })),
    },
    errors: {
      invokeErrorCount: parseInvokeTraceRaw(errorRaw).length,
    },
    requests: electronRequests
      .filter((request) =>
        [
          ...ENVIRONMENT_LIFECYCLE_REQUIRED_METHODS,
          ...ENVIRONMENT_LIFECYCLE_OPEN_METHODS,
        ].includes(request.method),
      )
      .map(requestSummary),
  };
}

function stageMatches(stage, status, method) {
  return (
    stage?.visible === true &&
    stage?.environmentId === ENVIRONMENT_LIFECYCLE_ID &&
    stage?.status === status &&
    stage?.protocolMethod === method &&
    stage?.triggerLifecycle === status
  );
}

export function assertEnvironmentLifecycleEvidence(evidence) {
  assert(
    evidence.bridge.missingMethods.length === 0,
    `缺少 Environment current method: ${evidence.bridge.missingMethods.join(", ")}`,
  );
  assert(evidence.bridge.drainEventsSeen, "未观察到 app_server_drain_events");
  assert(
    evidence.bridge.mockFallbackHitCount === 0,
    "Environment lifecycle 命中了非 electron-ipc transport",
  );
  assert(evidence.bridge.failedInvokeCount === 0, "current bridge invoke 失败");
  assert(
    evidence.identity.environmentRequestsMatchIdentity,
    "Environment request identity 不一致",
  );
  assert(
    evidence.identity.canonicalThreadOpenHitCount > 0,
    "GUI 未通过 current Thread read/resume 打开 canonical Thread",
  );
  assert(
    stageMatches(
      evidence.gui.connected,
      "connected",
      "thread/environment/connected",
    ),
    "GUI 未显示初始 connected 状态",
  );
  assert(
    stageMatches(
      evidence.gui.disconnected,
      "disconnected",
      "thread/environment/disconnected",
    ),
    "GUI 未显示 disconnected 状态",
  );
  assert(
    stageMatches(
      evidence.gui.reconnected,
      "connected",
      "thread/environment/connected",
    ),
    "GUI 未显示 reconnect 后的 connected 状态",
  );
  assert(
    evidence.remote.connectionCount >= 2,
    "exec-server fixture 未观察到重连",
  );
  for (const connection of evidence.remote.methodsByConnection.slice(0, 2)) {
    for (const method of [
      "initialize",
      "environment/info",
      "environment/status",
    ]) {
      assert(
        connection.methods.includes(method),
        `exec-server connection ${connection.connection} 缺少 ${method}`,
      );
    }
  }
  assert(
    evidence.errors.invokeErrorCount === 0,
    `观察到 invoke error: ${evidence.errors.invokeErrorCount}`,
  );
}

async function waitForConversation(page, options) {
  return await waitForPageCondition(
    page,
    options,
    (title) => {
      const button = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      ).find(
        (candidate) =>
          candidate.getAttribute("title") === title ||
          candidate.textContent?.includes(title),
      );
      return button ? true : null;
    },
    "Environment Gate B 侧栏会话未出现",
    ENVIRONMENT_LIFECYCLE_TITLE,
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
  }, ENVIRONMENT_LIFECYCLE_TITLE);
  assert(clicked, "无法点击 Environment Gate B 侧栏会话");
}

async function waitForEnvironmentStage(page, options, status) {
  return await waitForPageCondition(
    page,
    options,
    ({ environmentId, expectedStatus }) => {
      const row = document.querySelector(
        `[data-testid="task-center-environment-runtime"][data-environment-id="${environmentId}"]`,
      );
      const trigger = document.querySelector(
        '[data-testid="task-center-environment-trigger"]',
      );
      if (
        !(row instanceof HTMLElement) ||
        row.dataset.environmentStatus !== expectedStatus ||
        trigger?.getAttribute("data-environment-lifecycle") !== expectedStatus
      ) {
        return null;
      }
      const bounds = row.getBoundingClientRect();
      const style = window.getComputedStyle(row);
      return {
        visible:
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          bounds.width > 0 &&
          bounds.height > 0,
        environmentId: row.dataset.environmentId,
        status: row.dataset.environmentStatus,
        protocolMethod: row.dataset.protocolMethod,
        triggerLifecycle: trigger?.getAttribute("data-environment-lifecycle"),
        text: row.textContent?.trim() || "",
      };
    },
    `Environment GUI 未进入 ${status} 状态`,
    { environmentId: ENVIRONMENT_LIFECYCLE_ID, expectedStatus: status },
  );
}

async function waitForRemoteConnection(fixture, count, timeoutMs) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(timeoutMs, 45_000)) {
    if (fixture.connectionCount() >= count) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`exec-server fixture 未达到 ${count} 次连接`);
}

async function run() {
  const options = parseEnvironmentLifecycleGateArgs(process.argv.slice(2));
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
  const disconnectedScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-disconnected.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );
  const runtimeEnv = createTempRuntimeEnv();
  const workspaceRoot = path.join(runtimeEnv.tempRoot, "workspace");
  fs.mkdirSync(workspaceRoot, { recursive: true });
  const remoteFixture = await startRemoteEnvironmentFixture();
  const consoleErrors = [];
  const pageErrors = [];
  let handle = null;
  const summary = {
    schemaVersion: 1,
    scenarioId: "CODEX-ALIGN-environment-lifecycle",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron Environment status and lifecycle projection for one canonical Thread. It does not start a Turn or call a model.",
    backendMode: "unavailable",
    ok: false,
    checkedAt: new Date().toISOString(),
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
      backendMode: "unavailable",
    });
    await handle.page.evaluate(() => {
      window.localStorage.removeItem("lime_invoke_error_buffer_v1");
      window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
      window.localStorage.setItem("lime.app-sidebar.collapsed", "false");
    });

    console.log(`${LOG_PREFIX} stage=register-environment`);
    const added = await appServerCallFromPage(handle.page, "environment/add", {
      environmentId: ENVIRONMENT_LIFECYCLE_ID,
      execServerUrl: remoteFixture.url,
      connectTimeoutMs: 10_000,
    });
    await waitForRemoteConnection(remoteFixture, 1, options.timeoutMs);
    let statusResult = null;
    for (let attempt = 0; attempt < 80; attempt += 1) {
      statusResult = await appServerCallFromPage(
        handle.page,
        "environment/status",
        { environmentId: ENVIRONMENT_LIFECYCLE_ID },
      );
      if (statusResult.result?.status === "ready") break;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    assert(statusResult?.result?.status === "ready", "远端 Environment 未就绪");

    console.log(`${LOG_PREFIX} stage=create-canonical-thread`);
    const started = await appServerCallFromPage(handle.page, "thread/start", {
      model: "fixture-model",
      modelProvider: "fixture-provider",
      cwd: workspaceRoot,
      serviceName: ENVIRONMENT_LIFECYCLE_TITLE,
      historyMode: "legacy",
      threadSource: "fixture",
      environments: [
        {
          environmentId: ENVIRONMENT_LIFECYCLE_ID,
          cwd: "/remote/workspace",
          runtimeWorkspaceRoots: ["/remote/workspace"],
        },
      ],
    });
    const threadId = started.result?.thread?.id;
    assert(
      typeof threadId === "string" && threadId,
      "thread/start 未返回 Thread ID",
    );
    const setupRequests = [
      {
        command: added.appServerCommand,
        method: added.method,
        transport: "electron-ipc",
        status: "success",
        params: { environmentId: ENVIRONMENT_LIFECYCLE_ID },
      },
      {
        command: statusResult.appServerCommand,
        method: statusResult.method,
        transport: "electron-ipc",
        status: "success",
        params: { environmentId: ENVIRONMENT_LIFECYCLE_ID },
      },
      {
        command: started.appServerCommand,
        method: started.method,
        transport: "electron-ipc",
        status: "success",
        params: { threadId, environments: [ENVIRONMENT_LIFECYCLE_ID] },
      },
    ];

    console.log(`${LOG_PREFIX} stage=open-thread`);
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
      .locator('[data-testid="task-center-environment-trigger"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    await handle.page
      .locator('[data-testid="task-center-environment-trigger"]')
      .click();
    const connected = await waitForEnvironmentStage(
      handle.page,
      options,
      "connected",
    );

    console.log(`${LOG_PREFIX} stage=disconnect`);
    remoteFixture.disconnectFirst();
    const disconnected = await waitForEnvironmentStage(
      handle.page,
      options,
      "disconnected",
    );
    await handle.page.screenshot({
      path: disconnectedScreenshotPath,
      fullPage: true,
    });

    console.log(`${LOG_PREFIX} stage=reconnect`);
    remoteFixture.releaseReconnect();
    await waitForRemoteConnection(remoteFixture, 2, options.timeoutMs);
    const reconnected = await waitForEnvironmentStage(
      handle.page,
      options,
      "connected",
    );
    const observed = await handle.page.evaluate(() => ({
      traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    }));
    const evidence = summarizeEnvironmentLifecycleEvidence({
      ...observed,
      stages: { connected, disconnected, reconnected },
      threadId,
      remoteRequests: remoteFixture.requests,
      setupRequests,
    });
    assertEnvironmentLifecycleEvidence(evidence);
    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );
    assert(
      pageErrors.length === 0,
      `观察到 page error: ${pageErrors.join(" | ")}`,
    );

    await handle.page.screenshot({ path: screenshotPath, fullPage: true });
    Object.assign(summary, {
      ok: true,
      identity: evidence.identity,
      bridge: evidence.bridge,
      gui: evidence.gui,
      remote: evidence.remote,
      errors: {
        ...evidence.errors,
        consoleErrorCount: consoleErrors.length,
        pageErrorCount: pageErrors.length,
      },
      requests: evidence.requests,
      screenshots: [
        `${options.prefix}-disconnected.png`,
        `${options.prefix}.png`,
      ],
      tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    });
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
  } catch (error) {
    Object.assign(summary, {
      failure: sanitizeText(
        error instanceof Error ? error.message : String(error),
      ),
      errors: {
        consoleErrorCount: consoleErrors.length,
        pageErrorCount: pageErrors.length,
      },
    });
    if (handle?.page) {
      await handle.page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
    }
    writeJsonFile(summaryPath, summary);
    throw error;
  } finally {
    await closeElectronFixture(handle);
    await remoteFixture.close();
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
