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

export const THREAD_QUEUE_REQUIRED_METHODS = [
  "thread/start",
  "thread/queue/add",
  "thread/queue/list",
];
export const THREAD_QUEUE_OPEN_METHODS = ["thread/read", "thread/resume"];
export const THREAD_QUEUE_MARKER = "Queue Gate B pending submission";
export const THREAD_QUEUE_TITLE = "Queue Gate B canonical thread";

const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "thread-queue-electron-gate-b",
  ),
  prefix: "thread-queue-electron-gate-b",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LOG_PREFIX = "[smoke:thread-queue-gate-b]";

function printHelp() {
  console.log(`
Thread Queue Electron Gate B

用途:
  通过真实 Electron 创建 canonical Thread 和 durable Queue submission，
  再从 GUI 侧栏打开同一 Thread，验证队列 projection 用户可见。

边界:
  使用 unavailable backend；不启动 Turn、不调用模型，不使用 mock backend、
  renderer fallback 或旧 queued snapshot。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function parseThreadQueueGateArgs(argv, defaults = DEFAULTS) {
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
    threadId:
      typeof request.params?.threadId === "string"
        ? request.params.threadId
        : null,
  };
}

export function summarizeThreadQueueEvidence({
  traceRaw,
  errorRaw,
  dom,
  threadId,
  queuedSubmissionId,
  setupRequests = [],
}) {
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
  const queueRequests = requests.filter((request) =>
    request.method.startsWith("thread/queue/"),
  );
  const matchingQueueRequests = queueRequests.filter(
    (request) =>
      request.transport === "electron-ipc" &&
      request.status === "success" &&
      request.params?.threadId === threadId,
  );
  const openRequests = electronRequests.filter(
    (request) =>
      THREAD_QUEUE_OPEN_METHODS.includes(request.method) &&
      request.params?.threadId === threadId,
  );
  const commands = Array.from(
    new Set(
      [
        ...setupRequests.map((request) => request.command),
        ...parseInvokeTraceRaw(traceRaw).map((entry) => entry?.command),
      ].filter(Boolean),
    ),
  );
  return {
    identity: {
      threadId,
      queuedSubmissionId,
      queueRequestsMatchThread:
        queueRequests.length > 0 &&
        queueRequests.length === matchingQueueRequests.length,
      canonicalThreadOpenHitCount: openRequests.length,
    },
    bridge: {
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      appServerHandleJsonLinesSeen: commands.includes(
        APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      ),
      methods,
      missingMethods: THREAD_QUEUE_REQUIRED_METHODS.filter(
        (method) => !methods.includes(method),
      ),
      electronIpcHitCount: electronRequests.length,
      mockFallbackHitCount: requests.filter(
        (request) =>
          (THREAD_QUEUE_REQUIRED_METHODS.includes(request.method) ||
            request.method.startsWith("thread/queue/")) &&
          request.transport !== "electron-ipc",
      ).length,
    },
    gui: {
      conversationButtonVisible: dom?.conversationButtonVisible === true,
      headerTitle: typeof dom?.headerTitle === "string" ? dom.headerTitle : "",
      queueStatusVisible: dom?.queueStatusVisible === true,
      queueStatusText:
        typeof dom?.queueStatusText === "string" ? dom.queueStatusText : "",
      queueItemCount: Number(dom?.queueItemCount ?? 0),
      markerVisible: dom?.markerVisible === true,
    },
    errors: {
      invokeErrorCount: parseInvokeTraceRaw(errorRaw).length,
    },
    requests: electronRequests
      .filter(
        (request) =>
          THREAD_QUEUE_REQUIRED_METHODS.includes(request.method) ||
          THREAD_QUEUE_OPEN_METHODS.includes(request.method),
      )
      .map(requestSummary),
  };
}

export function assertThreadQueueEvidence(evidence) {
  assert(
    evidence.bridge.appServerHandleJsonLinesSeen,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    evidence.bridge.missingMethods.length === 0,
    `缺少 Queue current method: ${evidence.bridge.missingMethods.join(", ")}`,
  );
  assert(
    evidence.bridge.mockFallbackHitCount === 0,
    "Queue 命中了非 electron-ipc transport",
  );
  assert(
    evidence.identity.queueRequestsMatchThread,
    "Queue request 与 canonical Thread identity 不一致",
  );
  assert(
    evidence.identity.canonicalThreadOpenHitCount > 0,
    "GUI 未通过 thread/read 或 thread/resume 打开 canonical Thread",
  );
  assert(evidence.gui.conversationButtonVisible, "侧栏会话入口不可见");
  assert(
    evidence.gui.headerTitle.includes(THREAD_QUEUE_TITLE),
    `Thread header 标题不正确: ${evidence.gui.headerTitle}`,
  );
  assert(evidence.gui.queueStatusVisible, "thread-queue-status 不可见");
  assert(evidence.gui.markerVisible, "Queue marker 未在 GUI 中可见");
  assert(
    evidence.gui.queueItemCount === 1,
    `Queue GUI 条目数量不正确: ${evidence.gui.queueItemCount}`,
  );
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
      const buttons = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      );
      const button = buttons.find(
        (candidate) =>
          candidate.getAttribute("title") === title ||
          candidate.textContent?.includes(title),
      );
      return button
        ? {
            title: button.getAttribute("title") || "",
            text: button.textContent || "",
          }
        : null;
    },
    "Queue Gate B 侧栏会话未出现",
    THREAD_QUEUE_TITLE,
  );
}

async function clickConversation(page) {
  const clicked = await page.evaluate((title) => {
    const buttons = Array.from(
      document.querySelectorAll(
        '[data-testid="app-sidebar-conversation-open"]',
      ),
    );
    const button = buttons.find(
      (candidate) =>
        candidate.getAttribute("title") === title ||
        candidate.textContent?.includes(title),
    );
    if (!(button instanceof HTMLButtonElement)) return false;
    button.click();
    return true;
  }, THREAD_QUEUE_TITLE);
  assert(clicked, "无法点击 Queue Gate B 侧栏会话");
}

async function readGuiAndTrace(page) {
  return await page.evaluate(
    ({ marker, title }) => {
      const conversationButton = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      ).find(
        (candidate) =>
          candidate.getAttribute("title") === title ||
          candidate.textContent?.includes(title),
      );
      const header = document.querySelector(
        '[data-testid="thread-workspace-header-title"]',
      );
      const status = document.querySelector(
        '[data-testid="thread-queue-status"]',
      );
      const items = Array.from(
        status?.querySelectorAll('[data-testid="thread-queue-items"] > li') ??
          [],
      );
      const isVisible = (element) => {
        if (!(element instanceof HTMLElement)) return false;
        const style = window.getComputedStyle(element);
        const bounds = element.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          bounds.width > 0 &&
          bounds.height > 0
        );
      };
      return {
        dom: {
          conversationButtonVisible: isVisible(conversationButton),
          headerTitle: header?.textContent?.trim() || "",
          queueStatusVisible: isVisible(status),
          queueStatusText: status?.textContent?.trim() || "",
          queueItemCount: items.length,
          markerVisible: Boolean(
            items.some((item) => item.textContent?.includes(marker)),
          ),
        },
        traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    { marker: THREAD_QUEUE_MARKER, title: THREAD_QUEUE_TITLE },
  );
}

async function run() {
  const options = parseThreadQueueGateArgs(process.argv.slice(2));
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
    scenarioId: "CODEX-ALIGN-thread-queue",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron durable pending Thread Queue projection for one canonical Thread. It does not start a Turn or call a model.",
    backendMode: "unavailable",
    ok: false,
    checkedAt: new Date().toISOString(),
    identity: null,
    bridge: null,
    gui: null,
    errors: null,
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
      window.localStorage.removeItem("lime_invoke_error_buffer_v1");
      window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
      window.localStorage.setItem("lime.app-sidebar.collapsed", "false");
    });

    console.log(`${LOG_PREFIX} stage=create-canonical-thread`);
    const started = await appServerCallFromPage(handle.page, "thread/start", {
      model: "fixture-model",
      modelProvider: "fixture-provider",
      cwd: workspaceRoot,
      serviceName: THREAD_QUEUE_TITLE,
      historyMode: "legacy",
      threadSource: "fixture",
    });
    const threadId = started.result?.thread?.id;
    assert(
      typeof threadId === "string" && threadId,
      "thread/start 未返回 Thread ID",
    );

    console.log(`${LOG_PREFIX} stage=add-queue-submission`);
    const added = await appServerCallFromPage(handle.page, "thread/queue/add", {
      threadId,
      input: [{ type: "text", text: THREAD_QUEUE_MARKER }],
      clientUserMessageId: "queue-gate-b-client-message",
    });
    const queuedSubmission = added.result?.queuedSubmission;
    assert(
      typeof queuedSubmission?.id === "string" && queuedSubmission.id,
      "thread/queue/add 未返回 QueuedSubmission ID",
    );
    assert(
      queuedSubmission?.input?.[0]?.text === THREAD_QUEUE_MARKER,
      "thread/queue/add 返回的 marker 不正确",
    );
    const setupRequests = [
      {
        command: started.appServerCommand,
        method: started.method,
        transport: "electron-ipc",
        status: "success",
        params: {},
      },
      {
        command: added.appServerCommand,
        method: added.method,
        transport: "electron-ipc",
        status: "success",
        params: { threadId },
      },
    ];

    console.log(`${LOG_PREFIX} stage=open-thread-from-sidebar`);
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
      .locator('[data-testid="thread-queue-status"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    await handle.page
      .locator('[data-testid="thread-queue-items"]')
      .getByText(THREAD_QUEUE_MARKER, { exact: true })
      .waitFor({ state: "visible", timeout: options.timeoutMs });

    const observed = await readGuiAndTrace(handle.page);
    const evidence = summarizeThreadQueueEvidence({
      ...observed,
      threadId,
      queuedSubmissionId: queuedSubmission.id,
      setupRequests,
    });
    assertThreadQueueEvidence(evidence);
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
    summary.identity = {
      queueRequestsMatchThread: evidence.identity.queueRequestsMatchThread,
      canonicalThreadOpenHitCount:
        evidence.identity.canonicalThreadOpenHitCount,
      queuedSubmissionVisible: evidence.gui.markerVisible,
    };
    summary.bridge = evidence.bridge;
    summary.gui = evidence.gui;
    summary.errors = {
      ...evidence.errors,
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      mockFallbackHitCount: evidence.bridge.mockFallbackHitCount,
    };
    summary.requests = evidence.requests.map(
      ({ method, transport, status }) => ({
        method,
        transport,
        status,
      }),
    );
    summary.screenshot = `${options.prefix}.png`;
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
  } catch (error) {
    summary.failure = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    summary.errors = {
      ...(summary.errors ?? {}),
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    if (handle?.page) {
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
