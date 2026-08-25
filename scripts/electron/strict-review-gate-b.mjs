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
  assert,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  sanitizeText,
  waitForPageCondition,
} from "./mcp-config-fixture-smoke.mjs";

export const STRICT_REVIEW_METHOD = "autoApprovalReview/strictReviewRequired";
export const STRICT_REVIEW_STARTED_AT_MS = 1_783_814_400_100;
export const STRICT_REVIEW_TITLE = "Strict Review Gate B canonical thread";
export const STRICT_REVIEW_PROMPT =
  "Check a sensitive command before continuing.";
export const STRICT_REVIEW_REQUIRED_METHODS = [
  "thread/start",
  "thread/resume",
  "turn/start",
];
const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
const STRICT_REVIEW_RUNTIME_EVENT = "guardian.review.started";
const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "strict-review-electron-gate-b",
  ),
  prefix: "strict-review-electron-gate-b",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LOG_PREFIX = "[smoke:strict-review-gate-b]";

function printHelp() {
  console.log(`
Strict Review Electron Gate B

用途:
  通过真实 Electron 和 external fixture backend 发出 Guardian review started，
  验证 App Server exact strict-review notification 与 Composer 可见状态。

边界:
  fixture 不调用正式模型、不使用 mock backend 或 renderer fallback；
  只证明 strict-review notification、同一 Thread/Turn identity 与 GUI 投影。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function parseStrictReviewGateArgs(argv, defaults = DEFAULTS) {
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

function writeFixtureBackend(backendPath) {
  fs.writeFileSync(
    backendPath,
    `#!/usr/bin/env node
import { appendFileSync, readFileSync } from "node:fs";
import { setTimeout as delay } from "node:timers/promises";

const ledgerPath = process.argv[2];
const input = JSON.parse(readFileSync(0, "utf8"));
const request = input.request || {};
const session = request.session || {};
const turn = request.turn || {};
const sessionId = String(session.sessionId || "");
const threadId = String(session.threadId || "");
const turnId = String(turn.turnId || "");
const events = input.kind === "turnStart" ? [{
  type: ${JSON.stringify(STRICT_REVIEW_RUNTIME_EVENT)},
  payload: {
    reviewId: "strict-review-gate-b",
    startedAtMs: ${STRICT_REVIEW_STARTED_AT_MS},
    targetItemId: "strict-review-command",
    action: {
      type: "command",
      source: "shell",
      command: "git status --short",
      cwd: "/workspace"
    }
  }
}] : [];

if (ledgerPath) {
  appendFileSync(ledgerPath, JSON.stringify({
    kind: input.kind,
    sessionId,
    threadId,
    turnId,
    eventTypes: events.map((event) => event.type)
  }) + "\\n");
}
console.log(JSON.stringify({ events }));
if (input.kind === "turnStart") {
  await delay(5_000);
}
`,
    "utf8",
  );
}

function readBackendLedger(ledgerPath) {
  if (!fs.existsSync(ledgerPath)) return [];
  return fs
    .readFileSync(ledgerPath, "utf8")
    .split(/\r?\n/u)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function relevantRequestSummary(request) {
  return {
    method: request.method,
    status: request.status,
    transport: request.transport,
  };
}

export function summarizeStrictReviewEvidence({
  traceRaw,
  errorRaw,
  dom,
  threadId,
  turnId,
  backendLedger = [],
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
  const commands = Array.from(
    new Set(trace.map((entry) => entry?.command).filter(Boolean)),
  );
  const backendTurn = backendLedger.find(
    (entry) =>
      entry.kind === "turnStart" &&
      entry.eventTypes?.includes(STRICT_REVIEW_RUNTIME_EVENT),
  );
  const relevantTrace = trace.filter((entry) =>
    [
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      APP_SERVER_DRAIN_EVENTS_COMMAND,
    ].includes(entry?.command),
  );
  return {
    identity: {
      backendMatchesCanonicalTurn:
        backendTurn?.threadId === threadId && backendTurn?.turnId === turnId,
      domMatchesCanonicalTurn:
        dom?.threadId === threadId && dom?.turnId === turnId,
      exactStartedAt: Number(dom?.startedAtMs) === STRICT_REVIEW_STARTED_AT_MS,
    },
    bridge: {
      methods,
      missingMethods: STRICT_REVIEW_REQUIRED_METHODS.filter(
        (method) => !methods.includes(method),
      ),
      appServerHandleJsonLinesSeen: methods.length > 0,
      appServerDrainEventsSeen: commands.includes(
        APP_SERVER_DRAIN_EVENTS_COMMAND,
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
    gui: {
      statusVisible: dom?.statusVisible === true,
      inputbarVisible: dom?.inputbarVisible === true,
      exactProtocolMethod: dom?.protocolMethod === STRICT_REVIEW_METHOD,
      titleVisible: dom?.titleVisible === true,
      descriptionVisible: dom?.descriptionVisible === true,
      nextStepVisible: dom?.nextStepVisible === true,
    },
    backend: {
      eventTypes: Array.isArray(backendTurn?.eventTypes)
        ? backendTurn.eventTypes
        : [],
    },
    errors: {
      invokeErrorCount: parseInvokeTraceRaw(errorRaw).length,
    },
    requests: electronRequests
      .filter((request) =>
        STRICT_REVIEW_REQUIRED_METHODS.includes(request.method),
      )
      .map(relevantRequestSummary),
  };
}

export function assertStrictReviewEvidence(evidence) {
  assert(
    evidence.bridge.appServerHandleJsonLinesSeen,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    evidence.bridge.appServerDrainEventsSeen,
    "未观察到 app_server_drain_events",
  );
  assert(
    evidence.bridge.missingMethods.length === 0,
    `缺少 Strict Review current method: ${evidence.bridge.missingMethods.join(", ")}`,
  );
  assert(
    evidence.bridge.mockFallbackHitCount === 0,
    "Strict Review 命中了非 electron-ipc transport",
  );
  assert(evidence.bridge.failedInvokeCount === 0, "current bridge invoke 失败");
  assert(
    evidence.identity.backendMatchesCanonicalTurn,
    "backend Guardian event 与 canonical Thread/Turn identity 不一致",
  );
  assert(
    evidence.identity.domMatchesCanonicalTurn,
    "GUI strict-review 状态与 canonical Thread/Turn identity 不一致",
  );
  assert(evidence.identity.exactStartedAt, "GUI strict-review 开始时间不一致");
  assert(evidence.gui.statusVisible, "strict-review 状态不可见");
  assert(evidence.gui.inputbarVisible, "Strict Review 状态替换了 Inputbar");
  assert(
    evidence.gui.exactProtocolMethod,
    "GUI 未绑定 exact Strict Review method",
  );
  assert(evidence.gui.titleVisible, "Strict Review 标题不可见");
  assert(evidence.gui.descriptionVisible, "Strict Review 状态说明不可见");
  assert(evidence.gui.nextStepVisible, "Strict Review 下一步不可见");
  assert(
    evidence.errors.invokeErrorCount === 0,
    `观察到 invoke error: ${evidence.errors.invokeErrorCount}`,
  );
}

async function waitForConversation(page, options) {
  await waitForPageCondition(
    page,
    options,
    (title) => {
      const buttons = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      );
      return buttons.some(
        (button) =>
          button.getAttribute("title") === title ||
          button.textContent?.includes(title),
      );
    },
    "Strict Review Gate B 侧栏会话未出现",
    STRICT_REVIEW_TITLE,
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
  }, STRICT_REVIEW_TITLE);
  assert(clicked, "无法点击 Strict Review Gate B 侧栏会话");
}

async function readGuiAndTrace(page) {
  return await page.evaluate(() => {
    const status = document.querySelector(
      '[data-testid="strict-review-status"]',
    );
    const inputbar = document.querySelector(
      'textarea[name="agent-chat-message"]',
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
    const text = status?.textContent || "";
    return {
      dom: {
        statusVisible: isVisible(status),
        inputbarVisible: isVisible(inputbar),
        protocolMethod: status?.getAttribute("data-protocol-method") || "",
        threadId: status?.getAttribute("data-thread-id") || "",
        turnId: status?.getAttribute("data-turn-id") || "",
        startedAtMs: status?.getAttribute("data-started-at-ms") || "",
        titleVisible: /严格|嚴格|Strict|厳格|엄격/u.test(text),
        descriptionVisible: text.trim().length > 20,
        nextStepVisible: /保持|開啟|Keep|開いたまま|열어 두/u.test(text),
      },
      traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    };
  });
}

async function run() {
  const options = parseStrictReviewGateArgs(process.argv.slice(2));
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
  const backendPath = path.join(
    runtimeEnv.tempRoot,
    "strict-review-backend.mjs",
  );
  const ledgerPath = path.join(
    runtimeEnv.tempRoot,
    "strict-review-ledger.jsonl",
  );
  fs.mkdirSync(workspaceRoot, { recursive: true });
  writeFixtureBackend(backendPath);
  runtimeEnv.env.APP_SERVER_BACKEND_COMMAND = process.execPath;
  runtimeEnv.env.APP_SERVER_BACKEND_ARGS = JSON.stringify([
    backendPath,
    ledgerPath,
  ]);
  runtimeEnv.env.APP_SERVER_BACKEND_TIMEOUT_MS = "10000";
  let handle = null;
  let canonicalThreadId = null;
  let canonicalTurnId = null;
  let setupRequests = [];
  const consoleErrors = [];
  const pageErrors = [];
  const summary = {
    schemaVersion: 1,
    scenarioId: "CODEX-ALIGN-strict-review",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron exact strict-review notification and visible Composer state for one canonical Thread/Turn. It does not prove a live provider or a real dangerous command.",
    backendMode: "external",
    ok: false,
    checkedAt: new Date().toISOString(),
    identity: null,
    bridge: null,
    gui: null,
    backend: null,
    errors: null,
    screenshot: null,
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
      backendMode: "external",
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
      serviceName: STRICT_REVIEW_TITLE,
      historyMode: "legacy",
      threadSource: "fixture",
    });
    const threadId = String(started.result?.thread?.id || "").trim();
    assert(threadId, "thread/start 未返回 Thread ID");
    canonicalThreadId = threadId;
    setupRequests = [
      {
        command: started.appServerCommand,
        method: started.method,
        transport: "electron-ipc",
        status: "success",
        params: {},
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
      .locator('textarea[name="agent-chat-message"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });

    const resumed = await appServerCallFromPage(handle.page, "thread/resume", {
      threadId,
      excludeTurns: true,
    });
    setupRequests.push({
      command: resumed.appServerCommand,
      method: resumed.method,
      transport: "electron-ipc",
      status: "success",
      params: {},
    });
    assert(
      resumed.result?.thread?.id === threadId,
      "thread/resume 未恢复同一 canonical Thread",
    );

    console.log(`${LOG_PREFIX} stage=start-strict-review-turn`);
    const turn = await appServerCallFromPage(handle.page, "turn/start", {
      threadId,
      clientUserMessageId: `strict-review-${Date.now()}-${process.pid}`,
      input: [{ type: "text", text: STRICT_REVIEW_PROMPT }],
      cwd: workspaceRoot,
      runtimeWorkspaceRoots: [workspaceRoot],
      model: "fixture-model",
      approvalPolicy: "never",
      sandboxPolicy: "danger-full-access",
    });
    const turnId = String(turn.result?.turn?.id || "").trim();
    assert(turnId, "turn/start 未返回 Turn ID");
    canonicalTurnId = turnId;
    setupRequests.push({
      command: turn.appServerCommand,
      method: turn.method,
      transport: "electron-ipc",
      status: "success",
      params: {},
    });
    await handle.page
      .locator('[data-testid="strict-review-status"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });

    const observed = await readGuiAndTrace(handle.page);
    const evidence = summarizeStrictReviewEvidence({
      ...observed,
      threadId,
      turnId,
      backendLedger: readBackendLedger(ledgerPath),
      setupRequests,
    });
    assertStrictReviewEvidence(evidence);
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
    summary.backend = evidence.backend;
    summary.errors = {
      ...evidence.errors,
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    summary.screenshot = path.basename(screenshotPath);
  } catch (error) {
    summary.error = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    if (handle?.page && canonicalThreadId && canonicalTurnId) {
      const observed = await readGuiAndTrace(handle.page).catch(() => null);
      if (observed) {
        const partialEvidence = summarizeStrictReviewEvidence({
          ...observed,
          threadId: canonicalThreadId,
          turnId: canonicalTurnId,
          backendLedger: readBackendLedger(ledgerPath),
          setupRequests,
        });
        summary.partialEvidence = partialEvidence;
      }
    }
    summary.errors = {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    if (handle?.page && !handle.page.isClosed()) {
      await handle.page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
      summary.screenshot = path.basename(failureScreenshotPath);
    }
    throw error;
  } finally {
    writeJsonFile(summaryPath, summary);
    await closeElectronFixture(handle);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { force: true, recursive: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  run().catch((error) => {
    console.error(
      `${LOG_PREFIX} ${sanitizeText(error?.message || String(error))}`,
    );
    process.exitCode = 1;
  });
}
