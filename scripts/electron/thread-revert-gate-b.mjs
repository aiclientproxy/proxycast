#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { setTimeout as delay } from "node:timers/promises";
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

export const THREAD_REVERT_FIRST_USER_MARKER =
  "Thread Revert Gate B first user marker";
export const THREAD_REVERT_FIRST_ASSISTANT_MARKER =
  "Thread Revert Gate B first assistant marker";
export const THREAD_REVERT_SECOND_USER_MARKER =
  "Thread Revert Gate B second user marker";
export const THREAD_REVERT_SECOND_ASSISTANT_MARKER =
  "Thread Revert Gate B second assistant marker";
export const THREAD_REVERT_TITLE = "Thread Revert Gate B canonical thread";
export const THREAD_REVERT_WORKSPACE_CONTENT =
  "workspace-content-must-survive-thread-revert\n";
export const THREAD_REVERT_REQUIRED_METHODS = [
  "thread/start",
  "turn/start",
  "thread/revert",
  "thread/read",
];

const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "thread-revert-electron-gate-b",
  ),
  prefix: "thread-revert-electron-gate-b",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LOG_PREFIX = "[smoke:thread-revert-gate-b]";

function printHelp() {
  console.log(`
Thread Revert Electron Gate B

用途:
  通过真实 Electron 和 external fixture backend 创建两个 canonical Turn，
  再从第二轮用户消息执行“恢复到此消息前”，验证历史替换与工作区文件不变。

边界:
  fixture 不调用正式模型，不使用 mock backend、renderer fallback 或旧历史入口；
  只证明 current thread/revert GUI 闭环与 read model refresh。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function parseThreadRevertGateArgs(argv, defaults = DEFAULTS) {
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

const ledgerPath = process.argv[2];
const input = JSON.parse(readFileSync(0, "utf8"));
const request = input.request || {};
const session = request.session || {};
const turn = request.turn || {};
const sessionId = String(session.sessionId || "");
const threadId = String(session.threadId || "");
const turnId = String(turn.turnId || "");
const parts = Array.isArray(request?.input?.parts) ? request.input.parts : [];
const inputText = parts
  .map((part) => typeof part?.Text?.text === "string" ? part.Text.text : "")
  .join("");
const assistantText = inputText.includes(${JSON.stringify(THREAD_REVERT_FIRST_USER_MARKER)})
  ? ${JSON.stringify(THREAD_REVERT_FIRST_ASSISTANT_MARKER)}
  : ${JSON.stringify(THREAD_REVERT_SECOND_ASSISTANT_MARKER)};
const assistantItemId = "thread-revert-gate-b:assistant:" + turnId;
const events = input.kind === "turnStart" ? [
  {
    type: "message.delta",
    payload: {
      itemId: assistantItemId,
      role: "assistant",
      text: assistantText,
      phase: "final_answer"
    }
  },
  {
    type: "message.completed",
    payload: {
      itemId: assistantItemId,
      role: "assistant",
      text: assistantText,
      phase: "final_answer",
      status: "completed"
    }
  },
  {
    type: "turn.completed",
    payload: {
      status: "completed",
      text: assistantText
    }
  }
] : [];

if (ledgerPath) {
  appendFileSync(ledgerPath, JSON.stringify({
    kind: input.kind,
    sessionId,
    threadId,
    turnId,
    inputText,
    assistantText,
    eventTypes: events.map((event) => event.type)
  }) + "\\n");
}
console.log(JSON.stringify({ events }));
`,
    { mode: 0o755 },
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

function requestSummary(request) {
  return {
    method: request.method,
    transport: request.transport,
    status: request.status,
  };
}

export function summarizeThreadRevertEvidence({
  traceRaw,
  errorRaw,
  beforeDom,
  dialogDom,
  afterDom,
  threadId,
  firstTurnId,
  secondTurnId,
  setupRequests = [],
  backendLedger = [],
  workspaceContentBefore,
  workspaceContentAfter,
}) {
  const actionTrace = parseInvokeTraceRaw(traceRaw);
  const actionRequests = parseJsonRpcRequestsFromInvokeTrace(traceRaw);
  const requests = [...setupRequests, ...actionRequests];
  const electronRequests = requests.filter(
    (request) =>
      request.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
      request.transport === "electron-ipc" &&
      request.status === "success",
  );
  const methods = Array.from(
    new Set(electronRequests.map((request) => request.method)),
  );
  const revertRequests = actionRequests.filter(
    (request) => request.method === "thread/revert",
  );
  const matchingRevertRequests = revertRequests.filter(
    (request) =>
      request.transport === "electron-ipc" &&
      request.status === "success" &&
      request.params?.threadId === threadId &&
      request.params?.beforeTurnId === secondTurnId,
  );
  const refreshRequests = actionRequests.filter(
    (request) =>
      request.method === "thread/read" &&
      request.transport === "electron-ipc" &&
      request.status === "success" &&
      request.params?.threadId === threadId,
  );
  const backendTurns = backendLedger.filter(
    (entry) => entry.kind === "turnStart" && entry.threadId === threadId,
  );
  const relevantTrace = actionTrace.filter((entry) =>
    [
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      APP_SERVER_DRAIN_EVENTS_COMMAND,
    ].includes(entry?.command),
  );
  const actionMethods = Array.from(
    new Set(actionRequests.map((request) => request.method)),
  );

  return {
    identity: {
      setupTurnsMatchThread:
        backendTurns.some((entry) => entry.turnId === firstTurnId) &&
        backendTurns.some((entry) => entry.turnId === secondTurnId),
      targetMatchesSecondTurn:
        beforeDom?.targetThreadId === threadId &&
        beforeDom?.targetBeforeTurnId === secondTurnId,
      revertRequestMatchesSecondTurn:
        revertRequests.length === 1 && matchingRevertRequests.length === 1,
      threadHeaderPreserved:
        beforeDom?.headerTitle === THREAD_REVERT_TITLE &&
        afterDom?.headerTitle === THREAD_REVERT_TITLE,
    },
    bridge: {
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      methods,
      actionMethods,
      missingMethods: THREAD_REVERT_REQUIRED_METHODS.filter(
        (method) => !methods.includes(method),
      ),
      appServerHandleJsonLinesSeen: electronRequests.length > 0,
      appServerDrainEventsSeen: relevantTrace.some(
        (entry) =>
          entry.command === APP_SERVER_DRAIN_EVENTS_COMMAND &&
          entry.transport === "electron-ipc" &&
          entry.status === "success",
      ),
      refreshAfterRevertHitCount: refreshRequests.length,
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
      firstTurnVisibleBefore:
        beforeDom?.firstUserVisible === true &&
        beforeDom?.firstAssistantVisible === true,
      secondTurnVisibleBefore:
        beforeDom?.secondUserVisible === true &&
        beforeDom?.secondAssistantVisible === true,
      targetTriggerVisible: beforeDom?.targetTriggerVisible === true,
      dialogVisible: dialogDom?.dialogVisible === true,
      dialogExplainsHistoryRemoval: dialogDom?.explainsHistoryRemoval === true,
      dialogExplainsThreadPreserved:
        dialogDom?.explainsThreadPreserved === true,
      dialogExplainsFilesPreserved: dialogDom?.explainsFilesPreserved === true,
      confirmVisible: dialogDom?.confirmVisible === true,
      successStatus: afterDom?.statusState === "success",
      firstTurnVisibleAfter:
        afterDom?.firstUserVisible === true &&
        afterDom?.firstAssistantVisible === true,
      secondTurnRemovedAfter:
        afterDom?.secondUserVisible === false &&
        afterDom?.secondAssistantVisible === false,
    },
    backend: {
      turnStartCount: backendTurns.length,
      terminalEventCount: backendTurns.filter((entry) =>
        entry.eventTypes?.includes("turn.completed"),
      ).length,
    },
    workspace: {
      preserved:
        workspaceContentBefore === THREAD_REVERT_WORKSPACE_CONTENT &&
        workspaceContentAfter === THREAD_REVERT_WORKSPACE_CONTENT,
    },
    errors: {
      invokeErrorCount: parseInvokeTraceRaw(errorRaw).length,
    },
    requests: electronRequests
      .filter((request) =>
        THREAD_REVERT_REQUIRED_METHODS.includes(request.method),
      )
      .map(requestSummary),
  };
}

export function assertThreadRevertEvidence(evidence) {
  assert(
    evidence.bridge.appServerHandleJsonLinesSeen,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    evidence.bridge.appServerDrainEventsSeen,
    "GUI action 后未观察到 app_server_drain_events",
  );
  assert(
    evidence.bridge.missingMethods.length === 0,
    `缺少 Thread Revert current method: ${evidence.bridge.missingMethods.join(", ")}`,
  );
  assert(
    evidence.bridge.mockFallbackHitCount === 0,
    "Thread Revert 命中了非 electron-ipc transport",
  );
  assert(evidence.bridge.failedInvokeCount === 0, "current bridge invoke 失败");
  assert(
    evidence.bridge.refreshAfterRevertHitCount > 0,
    "thread/revert 后未通过 thread/read 刷新 canonical read model",
  );
  assert(
    evidence.identity.setupTurnsMatchThread,
    "external backend Turn 与 canonical Thread identity 不一致",
  );
  assert(
    evidence.identity.targetMatchesSecondTurn,
    "GUI Revert target 与第二个 canonical Turn identity 不一致",
  );
  assert(
    evidence.identity.revertRequestMatchesSecondTurn,
    "thread/revert 未精确命中第二个 canonical Turn",
  );
  assert(
    evidence.identity.threadHeaderPreserved,
    "thread/revert 后 Thread header identity 未保留",
  );
  assert(evidence.gui.firstTurnVisibleBefore, "恢复前第一轮历史不可见");
  assert(evidence.gui.secondTurnVisibleBefore, "恢复前第二轮历史不可见");
  assert(evidence.gui.targetTriggerVisible, "第二轮恢复入口不可见");
  assert(evidence.gui.dialogVisible, "Thread Revert 确认弹窗不可见");
  assert(
    evidence.gui.dialogExplainsHistoryRemoval,
    "确认弹窗未说明历史移除范围",
  );
  assert(
    evidence.gui.dialogExplainsThreadPreserved,
    "确认弹窗未说明 Thread identity 保留",
  );
  assert(
    evidence.gui.dialogExplainsFilesPreserved,
    "确认弹窗未说明本地文件不回滚",
  );
  assert(evidence.gui.confirmVisible, "Thread Revert 确认按钮不可见");
  assert(evidence.gui.successStatus, "Thread Revert GUI 未进入 success 状态");
  assert(evidence.gui.firstTurnVisibleAfter, "Thread Revert 错误移除了第一轮");
  assert(evidence.gui.secondTurnRemovedAfter, "Thread Revert 未移除第二轮");
  assert(
    evidence.backend.turnStartCount === 2 &&
      evidence.backend.terminalEventCount === 2,
    "external backend 未完成两个 canonical Turn",
  );
  assert(evidence.workspace.preserved, "Thread Revert 修改了工作区文件");
  assert(
    evidence.errors.invokeErrorCount === 0,
    `观察到 invoke error: ${evidence.errors.invokeErrorCount}`,
  );
}

async function waitForConversation(page, options) {
  await waitForPageCondition(
    page,
    options,
    (title) =>
      Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      ).some(
        (button) =>
          button.getAttribute("title") === title ||
          button.textContent?.includes(title),
      ),
    "Thread Revert Gate B 侧栏会话未出现",
    THREAD_REVERT_TITLE,
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
  }, THREAD_REVERT_TITLE);
  assert(clicked, "无法点击 Thread Revert Gate B 侧栏会话");
}

async function readHistoryDom(page, { threadId, secondTurnId }) {
  return await page.evaluate(
    ({ markers, threadId, secondTurnId }) => {
      const text = document.body.textContent || "";
      const target = Array.from(
        document.querySelectorAll('[data-testid="thread-revert-trigger"]'),
      ).find(
        (candidate) =>
          candidate.getAttribute("data-thread-id") === threadId &&
          candidate.getAttribute("data-before-turn-id") === secondTurnId,
      );
      const header = document.querySelector(
        '[data-testid="thread-workspace-header-title"]',
      );
      const status = document.querySelector(
        '[data-testid="thread-revert-status"]',
      );
      const isVisible = (element) => {
        if (!(element instanceof HTMLElement)) return false;
        const style = window.getComputedStyle(element);
        const bounds = element.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          style.opacity !== "0" &&
          style.pointerEvents !== "none" &&
          bounds.width > 0 &&
          bounds.height > 0
        );
      };
      return {
        headerTitle: header?.textContent?.trim() || "",
        firstUserVisible: text.includes(markers.firstUser),
        firstAssistantVisible: text.includes(markers.firstAssistant),
        secondUserVisible: text.includes(markers.secondUser),
        secondAssistantVisible: text.includes(markers.secondAssistant),
        targetTriggerVisible: isVisible(target),
        targetThreadId: target?.getAttribute("data-thread-id") || "",
        targetBeforeTurnId: target?.getAttribute("data-before-turn-id") || "",
        statusState: status?.getAttribute("data-state") || "",
      };
    },
    {
      markers: {
        firstUser: THREAD_REVERT_FIRST_USER_MARKER,
        firstAssistant: THREAD_REVERT_FIRST_ASSISTANT_MARKER,
        secondUser: THREAD_REVERT_SECOND_USER_MARKER,
        secondAssistant: THREAD_REVERT_SECOND_ASSISTANT_MARKER,
      },
      threadId,
      secondTurnId,
    },
  );
}

async function readDialogDom(page) {
  return await page.evaluate(() => {
    const dialog = document.querySelector('[role="dialog"]');
    const confirm = document.querySelector(
      '[data-testid="thread-revert-confirm"]',
    );
    const text = dialog?.textContent || "";
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
      dialogVisible: isVisible(dialog),
      confirmVisible: isVisible(confirm),
      explainsHistoryRemoval:
        /移除|removed|削除|제거/u.test(text) &&
        /之后|之後|later|以降|이후/u.test(text),
      explainsThreadPreserved: /保留|保持|stays|保持され|유지/u.test(text),
      explainsFilesPreserved:
        /不会回滚|不會回復|not rolled back|元に戻りません|되돌리지 않습니다/u.test(
          text,
        ),
    };
  });
}

async function startFixtureTurn(page, { threadId, marker, workspaceRoot }) {
  return await appServerCallFromPage(page, "turn/start", {
    threadId,
    clientUserMessageId: `thread-revert-${Date.now()}-${Math.random()
      .toString(16)
      .slice(2)}`,
    input: [{ type: "text", text: marker }],
    cwd: workspaceRoot,
    runtimeWorkspaceRoots: [workspaceRoot],
    model: "fixture-model",
    approvalPolicy: "never",
    sandboxPolicy: "danger-full-access",
  });
}

async function waitForTurnCompletion(
  page,
  options,
  { threadId, turnId, userMarker, assistantMarker },
) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latest = await appServerCallFromPage(page, "thread/read", {
      threadId,
      includeTurns: true,
    });
    const serialized = JSON.stringify(latest.result || {});
    if (
      serialized.includes(turnId) &&
      serialized.includes(userMarker) &&
      serialized.includes(assistantMarker)
    ) {
      return latest;
    }
    await delay(options.intervalMs);
  }
  throw new Error(`canonical read model 未完成 Turn: ${turnId || "<missing>"}`);
}

async function run() {
  const options = parseThreadRevertGateArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const confirmScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-confirm.png`,
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
  const workspaceFile = path.join(workspaceRoot, "revert-invariant.txt");
  const backendPath = path.join(
    runtimeEnv.tempRoot,
    "thread-revert-backend.mjs",
  );
  const ledgerPath = path.join(
    runtimeEnv.tempRoot,
    "thread-revert-ledger.jsonl",
  );
  fs.mkdirSync(workspaceRoot, { recursive: true });
  fs.writeFileSync(workspaceFile, THREAD_REVERT_WORKSPACE_CONTENT, "utf8");
  writeFixtureBackend(backendPath);
  runtimeEnv.env.APP_SERVER_BACKEND_COMMAND = process.execPath;
  runtimeEnv.env.APP_SERVER_BACKEND_ARGS = JSON.stringify([
    backendPath,
    ledgerPath,
  ]);
  runtimeEnv.env.APP_SERVER_BACKEND_TIMEOUT_MS = "10000";

  let handle = null;
  let observed = null;
  let setupRequests = [];
  let threadId = null;
  let firstTurnId = null;
  let secondTurnId = null;
  const consoleErrors = [];
  const pageErrors = [];
  const summary = {
    schemaVersion: 1,
    scenarioId: "CODEX-ALIGN-thread-revert",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron GUI thread/revert for two completed canonical Turns, canonical read-model refresh, preserved Thread identity and unchanged workspace file. It does not prove a live provider, packaged app or Windows.",
    backendMode: "external",
    ok: false,
    checkedAt: new Date().toISOString(),
    identity: null,
    bridge: null,
    gui: null,
    backend: null,
    workspace: null,
    errors: null,
    screenshots: [],
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
      backendMode: "external",
    });
    await handle.page.evaluate(() => {
      window.localStorage.removeItem("lime_invoke_error_buffer_v1");
      window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
      window.localStorage.setItem("lime.app-sidebar.collapsed", "false");
    });

    console.log(`${LOG_PREFIX} stage=create-paginated-thread`);
    const started = await appServerCallFromPage(handle.page, "thread/start", {
      model: "fixture-model",
      modelProvider: "fixture-provider",
      cwd: workspaceRoot,
      serviceName: THREAD_REVERT_TITLE,
      historyMode: "paginated",
      threadSource: "fixture",
    });
    threadId = String(started.result?.thread?.id || "").trim();
    assert(threadId, "thread/start 未返回 Thread ID");
    setupRequests.push({
      command: started.appServerCommand,
      method: started.method,
      transport: "electron-ipc",
      status: "success",
      params: {},
    });

    console.log(`${LOG_PREFIX} stage=complete-first-turn`);
    const firstTurn = await startFixtureTurn(handle.page, {
      threadId,
      marker: THREAD_REVERT_FIRST_USER_MARKER,
      workspaceRoot,
    });
    firstTurnId = String(firstTurn.result?.turn?.id || "").trim();
    assert(firstTurnId, "第一轮 turn/start 未返回 Turn ID");
    setupRequests.push({
      command: firstTurn.appServerCommand,
      method: firstTurn.method,
      transport: "electron-ipc",
      status: "success",
      params: { threadId },
    });
    const firstRead = await waitForTurnCompletion(handle.page, options, {
      threadId,
      turnId: firstTurnId,
      userMarker: THREAD_REVERT_FIRST_USER_MARKER,
      assistantMarker: THREAD_REVERT_FIRST_ASSISTANT_MARKER,
    });
    setupRequests.push({
      command: firstRead.appServerCommand,
      method: firstRead.method,
      transport: "electron-ipc",
      status: "success",
      params: { threadId },
    });

    console.log(`${LOG_PREFIX} stage=complete-second-turn`);
    const secondTurn = await startFixtureTurn(handle.page, {
      threadId,
      marker: THREAD_REVERT_SECOND_USER_MARKER,
      workspaceRoot,
    });
    secondTurnId = String(secondTurn.result?.turn?.id || "").trim();
    assert(secondTurnId, "第二轮 turn/start 未返回 Turn ID");
    setupRequests.push({
      command: secondTurn.appServerCommand,
      method: secondTurn.method,
      transport: "electron-ipc",
      status: "success",
      params: { threadId },
    });
    const secondRead = await waitForTurnCompletion(handle.page, options, {
      threadId,
      turnId: secondTurnId,
      userMarker: THREAD_REVERT_SECOND_USER_MARKER,
      assistantMarker: THREAD_REVERT_SECOND_ASSISTANT_MARKER,
    });
    setupRequests.push({
      command: secondRead.appServerCommand,
      method: secondRead.method,
      transport: "electron-ipc",
      status: "success",
      params: { threadId },
    });

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
    await handle.page
      .getByText(THREAD_REVERT_SECOND_ASSISTANT_MARKER, { exact: true })
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    await handle.page
      .locator(
        `[data-testid="thread-revert-trigger"][data-before-turn-id="${secondTurnId}"]`,
      )
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    await handle.page
      .locator(
        `[data-message-role="user"][data-runtime-turn-id="${secondTurnId}"]`,
      )
      .hover();
    await waitForPageCondition(
      handle.page,
      options,
      ({ threadId, secondTurnId }) => {
        const target = Array.from(
          document.querySelectorAll('[data-testid="thread-revert-trigger"]'),
        ).find(
          (candidate) =>
            candidate.getAttribute("data-thread-id") === threadId &&
            candidate.getAttribute("data-before-turn-id") === secondTurnId,
        );
        if (!(target instanceof HTMLElement)) return false;
        const style = window.getComputedStyle(target);
        return style.opacity !== "0" && style.pointerEvents !== "none";
      },
      "第二轮 Thread Revert hover footer 未进入可交互状态",
      { threadId, secondTurnId },
    );

    const beforeDom = await readHistoryDom(handle.page, {
      threadId,
      secondTurnId,
    });
    const workspaceContentBefore = fs.readFileSync(workspaceFile, "utf8");

    console.log(`${LOG_PREFIX} stage=open-revert-dialog`);
    await handle.page
      .locator(
        `[data-testid="thread-revert-trigger"][data-before-turn-id="${secondTurnId}"]`,
      )
      .click();
    await handle.page
      .locator('[data-testid="thread-revert-confirm"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    const dialogDom = await readDialogDom(handle.page);
    await handle.page.screenshot({
      path: confirmScreenshotPath,
      fullPage: true,
    });

    await handle.page.evaluate(() => {
      window.localStorage.removeItem("lime_invoke_error_buffer_v1");
      window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
    });

    console.log(`${LOG_PREFIX} stage=confirm-revert`);
    await handle.page.locator('[data-testid="thread-revert-confirm"]').click();
    await waitForPageCondition(
      handle.page,
      options,
      ({ firstMarker, secondMarker }) => {
        const status = document.querySelector(
          '[data-testid="thread-revert-status"]',
        );
        const text = document.body.textContent || "";
        const trace = JSON.parse(
          window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
        );
        const hasDrain = trace.some(
          (entry) =>
            entry?.command === "app_server_drain_events" &&
            entry?.transport === "electron-ipc" &&
            entry?.status === "success",
        );
        return (
          status?.getAttribute("data-state") === "success" &&
          text.includes(firstMarker) &&
          !text.includes(secondMarker) &&
          hasDrain
        );
      },
      "Thread Revert GUI 未完成历史替换与 canonical refresh",
      {
        firstMarker: THREAD_REVERT_FIRST_ASSISTANT_MARKER,
        secondMarker: THREAD_REVERT_SECOND_USER_MARKER,
      },
    );

    const afterDom = await readHistoryDom(handle.page, {
      threadId,
      secondTurnId,
    });
    const actionTrace = await handle.page.evaluate(() => ({
      traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    }));
    const workspaceContentAfter = fs.readFileSync(workspaceFile, "utf8");
    observed = {
      ...actionTrace,
      beforeDom,
      dialogDom,
      afterDom,
      threadId,
      firstTurnId,
      secondTurnId,
      setupRequests,
      backendLedger: readBackendLedger(ledgerPath),
      workspaceContentBefore,
      workspaceContentAfter,
    };
    const evidence = summarizeThreadRevertEvidence(observed);
    assertThreadRevertEvidence(evidence);
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
    summary.workspace = evidence.workspace;
    summary.errors = {
      ...evidence.errors,
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      mockFallbackHitCount: evidence.bridge.mockFallbackHitCount,
    };
    summary.requests = evidence.requests;
    summary.screenshots = [
      path.basename(confirmScreenshotPath),
      path.basename(screenshotPath),
    ];
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
  } catch (error) {
    summary.failure = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    if (observed) {
      summary.partialEvidence = summarizeThreadRevertEvidence(observed);
    }
    summary.errors = {
      ...(summary.errors ?? {}),
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    if (handle?.page && !handle.page.isClosed()) {
      await handle.page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
      summary.screenshots = [path.basename(failureScreenshotPath)];
    }
    throw error;
  } finally {
    writeJsonFile(summaryPath, summary);
    await closeElectronFixture(handle);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  run().catch((error) => {
    console.error(
      `${LOG_PREFIX} failed: ${sanitizeText(
        error instanceof Error ? error.message : String(error),
      )}`,
    );
    process.exitCode = 1;
  });
}
