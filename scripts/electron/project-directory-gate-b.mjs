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

export const PROJECT_DIRECTORY_REQUIRED_METHODS = [
  "project/create",
  "thread/start",
  "project/list",
  "thread/read",
  "thread/metadata/update",
];
export const PROJECT_DIRECTORY_INITIAL_NAME = "Project Gate B Alpha";
export const PROJECT_DIRECTORY_SELECTED_NAME = "Project Gate B Beta";
export const PROJECT_DIRECTORY_THREAD_TITLE = "Project Gate B canonical thread";

const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "project-directory-electron-gate-b",
  ),
  prefix: "project-directory-electron-gate-b",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LOG_PREFIX = "[smoke:project-directory-gate-b]";

function printHelp() {
  console.log(`
Project Directory Electron Gate B

用途:
  通过真实 Electron 创建两个 Project 和一个 canonical Thread，
  再从 GUI Project 目录切换当前 Thread 归属并验证 cold readback。

边界:
  使用 unavailable backend；不启动 Turn、不调用模型，不使用 workspace API、
  renderer fallback 或 raw IPC 作为 GUI 完成证据。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function parseProjectDirectoryGateArgs(argv, defaults = DEFAULTS) {
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

export function summarizeProjectDirectoryEvidence({
  traceRaw,
  errorRaw,
  dom,
  threadId,
  selectedProjectId,
  backendProjectId,
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
  const assignmentRequests = requests.filter(
    (request) => request.method === "thread/metadata/update",
  );
  const matchingAssignments = assignmentRequests.filter(
    (request) =>
      request.transport === "electron-ipc" &&
      request.status === "success" &&
      request.params?.threadId === threadId &&
      request.params?.projectId === selectedProjectId,
  );
  const threadReads = electronRequests.filter(
    (request) =>
      request.method === "thread/read" && request.params?.threadId === threadId,
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
      backendProjectMatchesSelection:
        Boolean(selectedProjectId) && backendProjectId === selectedProjectId,
      metadataUpdateMatchesThreadAndProject:
        assignmentRequests.length > 0 &&
        matchingAssignments.length === assignmentRequests.length,
      canonicalThreadReadHitCount: threadReads.length,
    },
    bridge: {
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      appServerHandleJsonLinesSeen: commands.includes(
        APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      ),
      methods,
      missingMethods: PROJECT_DIRECTORY_REQUIRED_METHODS.filter(
        (method) => !methods.includes(method),
      ),
      electronIpcHitCount: electronRequests.length,
      mockFallbackHitCount: requests.filter(
        (request) =>
          PROJECT_DIRECTORY_REQUIRED_METHODS.includes(request.method) &&
          request.transport !== "electron-ipc",
      ).length,
    },
    gui: {
      conversationButtonVisible: dom?.conversationButtonVisible === true,
      selectorVisible: dom?.selectorVisible === true,
      directoryVisible: dom?.directoryVisible === true,
      initialProjectVisible: dom?.initialProjectVisible === true,
      selectedProjectVisible: dom?.selectedProjectVisible === true,
      projectOptionCount: Number(dom?.projectOptionCount ?? 0),
    },
    errors: {
      invokeErrorCount: parseInvokeTraceRaw(errorRaw).length,
    },
    requests: electronRequests
      .filter((request) =>
        PROJECT_DIRECTORY_REQUIRED_METHODS.includes(request.method),
      )
      .map((request) => ({
        method: request.method,
        status: request.status,
        transport: request.transport,
      })),
  };
}

export function assertProjectDirectoryEvidence(evidence) {
  assert(
    evidence.bridge.appServerHandleJsonLinesSeen,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    evidence.bridge.missingMethods.length === 0,
    `缺少 Project current method: ${evidence.bridge.missingMethods.join(", ")}`,
  );
  assert(
    evidence.bridge.mockFallbackHitCount === 0,
    "Project 目录命中了非 electron-ipc transport",
  );
  assert(
    evidence.identity.metadataUpdateMatchesThreadAndProject,
    "GUI Project assignment 与 canonical Thread/Project identity 不一致",
  );
  assert(
    evidence.identity.canonicalThreadReadHitCount > 0,
    "GUI 未通过 thread/read 打开 canonical Thread",
  );
  assert(
    evidence.identity.backendProjectMatchesSelection,
    "thread/read 未恢复 GUI 选择的 Project",
  );
  assert(evidence.gui.conversationButtonVisible, "侧栏会话入口不可见");
  assert(evidence.gui.selectorVisible, "thread-project-selector 不可见");
  assert(evidence.gui.directoryVisible, "thread-project-directory 不可见");
  assert(
    evidence.gui.initialProjectVisible,
    "初始 Thread Project 未在 GUI 恢复",
  );
  assert(
    evidence.gui.selectedProjectVisible,
    "切换后的 Thread Project 未在 GUI 可见",
  );
  assert(
    evidence.gui.projectOptionCount >= 2,
    `Project 目录条目不足: ${evidence.gui.projectOptionCount}`,
  );
  assert(
    evidence.errors.invokeErrorCount === 0,
    `观察到 invoke error: ${evidence.errors.invokeErrorCount}`,
  );
}

export function sanitizeProjectDirectoryFailure(error, sensitivePaths = []) {
  let failure = sanitizeText(
    error instanceof Error ? error.message : String(error),
  );
  for (const sensitivePath of sensitivePaths) {
    const normalizedPath = String(sensitivePath ?? "").trim();
    if (normalizedPath) {
      failure = failure.replaceAll(normalizedPath, "[local-path]");
    }
  }
  return failure;
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
      return button
        ? {
            text: button.textContent || "",
            title: button.getAttribute("title") || "",
          }
        : null;
    },
    "Project Gate B 侧栏会话未出现",
    PROJECT_DIRECTORY_THREAD_TITLE,
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
  }, PROJECT_DIRECTORY_THREAD_TITLE);
  assert(clicked, "无法点击 Project Gate B 侧栏会话");
}

async function readVisibleState(page, selectedProjectId) {
  return await page.evaluate(
    ({ initialName, selectedName, selectedProjectId, title }) => {
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
      const conversationButton = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      ).find(
        (candidate) =>
          candidate.getAttribute("title") === title ||
          candidate.textContent?.includes(title),
      );
      const selector = document.querySelector(
        '[data-testid="thread-project-selector"]',
      );
      const directory = document.querySelector(
        '[data-testid="thread-project-directory"]',
      );
      return {
        dom: {
          conversationButtonVisible: isVisible(conversationButton),
          selectorVisible: isVisible(selector),
          directoryVisible: isVisible(directory),
          initialProjectVisible: Boolean(
            directory?.textContent?.includes(initialName),
          ),
          selectedProjectVisible:
            selector?.getAttribute("data-thread-project-id") ===
              selectedProjectId &&
            Boolean(selector?.textContent?.includes(selectedName)),
          projectOptionCount:
            directory?.querySelectorAll(
              'button[data-project-id]:not([data-project-id=""])',
            ).length ?? 0,
        },
        traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    {
      initialName: PROJECT_DIRECTORY_INITIAL_NAME,
      selectedName: PROJECT_DIRECTORY_SELECTED_NAME,
      selectedProjectId,
      title: PROJECT_DIRECTORY_THREAD_TITLE,
    },
  );
}

async function run() {
  const options = parseProjectDirectoryGateArgs(process.argv.slice(2));
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
  const firstRoot = path.join(workspaceRoot, "alpha");
  const secondRoot = path.join(workspaceRoot, "beta");
  fs.mkdirSync(firstRoot, { recursive: true });
  fs.mkdirSync(secondRoot, { recursive: true });
  let handle = null;
  const consoleErrors = [];
  const pageErrors = [];
  const summary = {
    schemaVersion: 1,
    scenarioId: "CODEX-ALIGN-project-directory",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron Project directory and canonical Thread assignment. It does not start a Turn or call a model.",
    backendMode: "unavailable",
    ok: false,
    checkedAt: new Date().toISOString(),
    identity: null,
    bridge: null,
    gui: null,
    errors: null,
    screenshot: null,
    tempKept: options.keepTemp,
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

    console.log(`${LOG_PREFIX} stage=create-projects-and-thread`);
    const firstProjectCall = await appServerCallFromPage(
      handle.page,
      "project/create",
      {
        idempotencyKey: "project-directory-gate-b-alpha",
        name: PROJECT_DIRECTORY_INITIAL_NAME,
        roots: [{ path: firstRoot }],
      },
    );
    const secondProjectCall = await appServerCallFromPage(
      handle.page,
      "project/create",
      {
        idempotencyKey: "project-directory-gate-b-beta",
        name: PROJECT_DIRECTORY_SELECTED_NAME,
        roots: [{ path: secondRoot }],
      },
    );
    const initialProjectId = firstProjectCall.result?.project?.id;
    const selectedProjectId = secondProjectCall.result?.project?.id;
    assert(
      typeof initialProjectId === "string" && initialProjectId,
      "首个 Project 创建失败",
    );
    assert(
      typeof selectedProjectId === "string" && selectedProjectId,
      "第二个 Project 创建失败",
    );
    const started = await appServerCallFromPage(handle.page, "thread/start", {
      model: "fixture-model",
      modelProvider: "fixture-provider",
      cwd: workspaceRoot,
      serviceName: PROJECT_DIRECTORY_THREAD_TITLE,
      historyMode: "legacy",
      threadSource: "fixture",
      projectId: initialProjectId,
    });
    const threadId = started.result?.thread?.id;
    assert(
      typeof threadId === "string" && threadId,
      "thread/start 未返回 Thread ID",
    );
    const setupRequests = [firstProjectCall, secondProjectCall, started].map(
      (call) => ({
        command: call.appServerCommand,
        method: call.method,
        params: {},
        status: "success",
        transport: "electron-ipc",
      }),
    );

    console.log(`${LOG_PREFIX} stage=open-thread-and-switch-project`);
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
    const selector = handle.page.locator(
      '[data-testid="thread-project-selector"]',
    );
    await selector.waitFor({ state: "visible", timeout: options.timeoutMs });
    await waitForPageCondition(
      handle.page,
      options,
      ({ initialProjectId, initialName }) => {
        const element = document.querySelector(
          '[data-testid="thread-project-selector"]',
        );
        return element?.getAttribute("data-thread-project-id") ===
          initialProjectId && element?.textContent?.includes(initialName)
          ? true
          : null;
      },
      "初始 Thread Project 未从 thread/read 恢复",
      { initialProjectId, initialName: PROJECT_DIRECTORY_INITIAL_NAME },
    );
    await selector.click();
    const directory = handle.page.locator(
      '[data-testid="thread-project-directory"]',
    );
    await directory.waitFor({ state: "visible", timeout: options.timeoutMs });
    await directory
      .locator('button[data-project-id]:not([data-project-id=""])')
      .filter({ hasText: PROJECT_DIRECTORY_SELECTED_NAME })
      .click();
    await waitForPageCondition(
      handle.page,
      options,
      ({ selectedProjectId, selectedName }) => {
        const element = document.querySelector(
          '[data-testid="thread-project-selector"]',
        );
        return element?.getAttribute("data-thread-project-id") ===
          selectedProjectId && element?.textContent?.includes(selectedName)
          ? true
          : null;
      },
      "切换后的 Thread Project 未在 GUI 生效",
      { selectedProjectId, selectedName: PROJECT_DIRECTORY_SELECTED_NAME },
    );

    console.log(`${LOG_PREFIX} stage=verify-canonical-readback`);
    await selector.click();
    await directory.waitFor({ state: "visible", timeout: options.timeoutMs });
    const readback = await appServerCallFromPage(handle.page, "thread/read", {
      threadId,
      includeTurns: false,
    });
    const backendProjectId = readback.result?.thread?.projectId;
    const observed = await readVisibleState(handle.page, selectedProjectId);
    const evidence = summarizeProjectDirectoryEvidence({
      ...observed,
      threadId,
      selectedProjectId,
      backendProjectId,
      setupRequests,
    });
    assertProjectDirectoryEvidence(evidence);
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
    summary.failure = sanitizeProjectDirectoryFailure(error, [
      runtimeEnv.tempRoot,
      process.cwd(),
    ]);
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
