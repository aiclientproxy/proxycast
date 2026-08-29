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
  sanitizeJson,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  appServerCallFromPage,
  assert,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  sanitizeText,
  sleep,
} from "./mcp-config-fixture-smoke.mjs";

const LOG_PREFIX = "[smoke:scheduled-tasks-electron-fixture]";
const FINAL_TEXT = "SCHEDULED_TASK_GATE_B_DONE";
const FIXTURE_MODEL = "scheduled-task-fixture-model";
const TASK_TITLE = "Scheduled Task Gate B";
const TASK_PROMPT = `Reply exactly ${FINAL_TEXT}.`;
const TERMINAL_STATUSES = new Set(["completed", "success"]);

export const REQUIRED_SCHEDULED_TASK_METHODS = [
  "scheduledTask/list",
  "scheduledTask/create",
  "scheduledTask/read",
  "scheduledTask/run/list",
  "scheduledTask/run/start",
  "thread/read",
];

const LEGACY_METHOD_PREFIXES = [
  "automationJob/",
  "automationSchedule/",
  "automationScheduler/",
];
const LEGACY_COMMANDS = [
  "create_automation_job",
  "get_automation_jobs",
  "get_automation_run_history",
  "run_automation_job_now",
];
const CREATE_LABELS = [
  "创建任务",
  "建立任務",
  "Create task",
  "タスクを作成",
  "작업 만들기",
];
const MANUAL_LABELS = [
  "手动设置",
  "手動設定",
  "Set up manually",
  "手動で設定",
  "직접 설정",
];
const MORE_LABELS = ["更多操作", "More actions", "その他の操作", "추가 작업"];
const RUN_LABELS = [
  "立即运行",
  "立即執行",
  "Run now",
  "今すぐ実行",
  "지금 실행",
];
const OPEN_RUN_LABELS = [
  "打开运行对话",
  "開啟執行對話",
  "Open run conversation",
  "実行の会話を開く",
  "실행 대화 열기",
];
const TITLE_LABELS = [
  "任务名称",
  "任務名稱",
  "Task name",
  "タスク名",
  "작업 이름",
];
const PROMPT_LABELS = [
  "任务说明",
  "任務說明",
  "Task instructions",
  "タスクの指示",
  "작업 지침",
];
const MODEL_LABELS = ["模型", "Model", "モデル", "모델"];

const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "scheduled-tasks-electron-fixture",
  ),
  prefix: "scheduled-tasks-electron-fixture",
  timeoutMs: 180_000,
  intervalMs: 250,
  keepTemp: false,
};

function printHelp() {
  console.log(`
Scheduled Tasks Electron Gate B Fixture

用途:
  在真实 Electron 输入框选择 fixture Provider/模型，从一级导航创建已安排任务并立即运行，
  验证 Runtime provider、canonical Thread/Turn/read model、运行历史与对话恢复。

边界:
  使用 localhost OpenAI-compatible provider 与隔离用户数据；不调用正式模型，
  不使用 App Server mock backend、renderer mock fallback 或旧 Automation facade。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function parseScheduledTasksFixtureArgs(argv, defaults = DEFAULTS) {
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
    throw new Error(`unknown argument: ${arg}`);
  }
  if (options.help) return options;
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be >= 100");
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(options.prefix)) {
    throw new Error("invalid evidence prefix");
  }
  return options;
}

function parseTrace(raw) {
  try {
    const parsed = JSON.parse(raw || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function methodsFromEntries(entries) {
  return Array.from(
    new Set(
      entries.flatMap((entry) =>
        (entry?.args_preview?.request?.lines ?? []).flatMap((line) => {
          try {
            const request = JSON.parse(String(line));
            return typeof request?.method === "string" ? [request.method] : [];
          } catch {
            return [];
          }
        }),
      ),
    ),
  );
}

export function summarizeScheduledTasksTrace(traceRaw, errorRaw = null) {
  const entries = parseTrace(traceRaw);
  const appServerEntries = entries.filter(
    (entry) => entry?.command === "app_server_handle_json_lines",
  );
  const electronEntries = appServerEntries.filter(
    (entry) => entry?.transport === "electron-ipc",
  );
  const methods = methodsFromEntries(electronEntries);
  const commands = new Set(entries.map((entry) => entry?.command));
  const legacyMethods = methods.filter((method) =>
    LEGACY_METHOD_PREFIXES.some((prefix) => method.startsWith(prefix)),
  );
  const legacyCommands = LEGACY_COMMANDS.filter((command) =>
    commands.has(command),
  );
  return {
    appServerIpcHitCount: electronEntries.length,
    methods,
    missingMethods: REQUIRED_SCHEDULED_TASK_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    legacyMethods,
    legacyCommands,
    mockFallbackHitCount: appServerEntries.length - electronEntries.length,
    invokeErrorCount: parseTrace(errorRaw).length,
  };
}

async function readElectronRuntime(app) {
  return await app.evaluate(async ({ app: electronApp }) => ({
    appVersion: electronApp.getVersion(),
    electronVersion: process.versions.electron,
    platform: process.platform,
    arch: process.arch,
  }));
}

async function roleLocator(page, role, labels) {
  for (const label of labels) {
    const locator = page.getByRole(role, { name: label, exact: true });
    if ((await locator.count()) > 0) return locator.first();
  }
  throw new Error(`${role} not found: ${labels.join(" / ")}`);
}

async function labelLocator(page, labels) {
  for (const label of labels) {
    const locator = page.getByLabel(label, { exact: true });
    if ((await locator.count()) > 0) return locator.first();
  }
  throw new Error(`field not found: ${labels.join(" / ")}`);
}

async function createFixtureProvider(page, fixture) {
  const created = await appServerCallFromPage(page, "modelProvider/create", {
    name: `Scheduled Tasks Gate B ${Date.now()}`,
    providerType: fixture.provider.providerName,
    apiHost: fixture.provider.providerConfig.baseUrl,
  });
  const providerId = String(created.result?.provider?.id || "").trim();
  assert(providerId, "modelProvider/create did not return provider.id");
  await appServerCallFromPage(page, "modelProvider/update", {
    providerId,
    enabled: true,
    models: [
      {
        id: fixture.provider.modelPreference,
        capability: fixture.provider.providerConfig.modelCapabilities,
      },
    ],
  });
  const key = await appServerCallFromPage(page, "modelProviderKey/create", {
    providerId,
    apiKey: fixture.provider.providerConfig.apiKey,
    alias: "scheduled-tasks-gate-b",
    replaceExisting: true,
  });
  assert(key.result?.key?.id, "modelProviderKey/create did not return key.id");
  const catalog = await appServerCallFromPage(page, "model/list", {
    includeHidden: true,
    limit: 500,
  });
  const selected = (catalog.result?.data ?? []).find(
    (candidate) =>
      candidate?.providerId === providerId &&
      candidate?.model === fixture.provider.modelPreference,
  );
  assert(selected, "fixture provider route is absent from model/list");
  return {
    providerId,
    model: fixture.provider.modelPreference,
  };
}

function encodeModelRouteSelector(providerId, model) {
  return `route:${Buffer.from(providerId, "utf8").toString("base64url")}.${Buffer.from(model, "utf8").toString("base64url")}`;
}

async function selectFixtureRouteForWorkspace(page, route) {
  const ensured = await appServerCallFromPage(
    page,
    "workspace/default/ensure",
    {},
  );
  const workspaceId = String(ensured.result?.workspace?.id || "").trim();
  assert(workspaceId, "workspace/default/ensure did not return workspace.id");
  await page.evaluate(
    ({ model, providerId, workspaceId: selectedWorkspaceId }) => {
      window.localStorage.setItem(
        "agent_last_project_id",
        JSON.stringify(selectedWorkspaceId),
      );
      window.localStorage.setItem(
        `agent_pref_provider_${selectedWorkspaceId}`,
        JSON.stringify(providerId),
      );
      window.localStorage.setItem(
        `agent_pref_model_${selectedWorkspaceId}`,
        JSON.stringify(model),
      );
    },
    { model: route.model, providerId: route.providerId, workspaceId },
  );
  return workspaceId;
}

async function clearInvokeBuffers(page) {
  await page.evaluate(() => {
    window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
    window.localStorage.removeItem("lime_invoke_error_buffer_v1");
  });
}

async function createTaskFromGui(page, route) {
  await page
    .locator('[data-testid="app-sidebar-nav-scheduled-tasks"]')
    .click();
  await page
    .getByText(TASK_TITLE, { exact: true })
    .waitFor({
      state: "detached",
      timeout: 5_000,
    })
    .catch(() => undefined);

  const createButton = await roleLocator(page, "button", CREATE_LABELS);
  await createButton.click();
  const titleField = await labelLocator(page, TITLE_LABELS).catch(() => null);
  if (!titleField) {
    const manual = await roleLocator(page, "menuitem", MANUAL_LABELS);
    await manual.click();
  }
  await (await labelLocator(page, TITLE_LABELS)).fill(TASK_TITLE);
  await (await labelLocator(page, PROMPT_LABELS)).fill(TASK_PROMPT);
  const modelField = await labelLocator(page, MODEL_LABELS);
  assert(
    (await modelField.inputValue()) === `${route.providerId} / ${route.model}`,
    "Scheduled Task editor did not inherit the Composer provider/model selection",
  );
  assert(
    (await modelField.getAttribute("readonly")) !== null,
    "Scheduled Task editor model selection is not read-only",
  );

  const createActions = await Promise.all(
    CREATE_LABELS.map(async (label) => {
      const locator = page.getByRole("button", { name: label, exact: true });
      return { locator, count: await locator.count() };
    }),
  );
  const action = createActions.find((candidate) => candidate.count > 0);
  assert(action, "Scheduled Task create action is unavailable");
  await action.locator.last().click();
  await page.getByText(TASK_TITLE, { exact: true }).last().waitFor({
    state: "visible",
  });

  const listed = await appServerCallFromPage(page, "scheduledTask/list", {
    limit: 200,
  });
  const task = (listed.result?.items ?? []).find(
    (candidate) => candidate?.title === TASK_TITLE,
  );
  assert(task?.id, "created Scheduled Task is absent from scheduledTask/list");
  const read = await appServerCallFromPage(page, "scheduledTask/read", {
    id: task.id,
  });
  return read.result?.task ?? read.result;
}

async function runTaskFromGui(page) {
  await (await roleLocator(page, "button", MORE_LABELS)).click();
  await (await roleLocator(page, "menuitem", RUN_LABELS)).click();
}

async function waitForCompletedRun(page, options) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const listed = await appServerCallFromPage(page, "scheduledTask/list", {
      limit: 200,
    });
    const task = (listed.result?.items ?? []).find(
      (candidate) => candidate?.title === TASK_TITLE,
    );
    if (task?.id) {
      const history = await appServerCallFromPage(
        page,
        "scheduledTask/run/list",
        { taskId: task.id, limit: 20 },
      );
      const run = history.result?.runs?.[0] ?? null;
      latest = { run, task };
      if (run?.status === "error" || run?.status === "failed") {
        throw new Error(
          `Scheduled Task run failed: ${run.error || run.status}`,
        );
      }
      if (
        TERMINAL_STATUSES.has(run?.status) &&
        run?.sessionId &&
        run?.threadId &&
        run?.turnId
      ) {
        const read = await appServerCallFromPage(page, "thread/read", {
          threadId: run.threadId,
          includeTurns: true,
        });
        const serialized = JSON.stringify(read.result || {});
        if (serialized.includes(FINAL_TEXT)) {
          return { read, run, task };
        }
      }
    }
    await sleep(options.intervalMs);
  }
  throw new Error(`Scheduled Task did not complete: ${JSON.stringify(latest)}`);
}

async function openRunFromGui(page, run) {
  const openRun = await roleLocator(page, "button", OPEN_RUN_LABELS);
  await openRun.click();
  await page
    .locator(
      `textarea[name="agent-chat-message"][data-session-id="${run.sessionId}"]`,
    )
    .waitFor({ state: "visible" });
  await page.getByText(FINAL_TEXT, { exact: false }).last().waitFor({
    state: "visible",
  });
}

async function collectTrace(page) {
  return await page.evaluate(() => ({
    traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
  }));
}

export async function run() {
  const options = parseScheduledTasksFixtureArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  ensureElectronFixtureBuild({ logPrefix: LOG_PREFIX, rootDir: process.cwd() });
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const file = (suffix) =>
    path.join(options.evidenceDir, `${options.prefix}${suffix}`);
  const summaryPath = file("-summary.json");
  const rawPath = file("-raw.json");
  const screenshotPath = file(".png");
  const failureScreenshotPath = file("-failure.png");
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
    scenarioId: "SCHEDULED-TASKS-01-run-and-open-canonical-thread",
    proofLevel: "Gate B",
    claimBoundary:
      "Real Electron/preload/IPC/App Server RuntimeCore provider turn proves the top-level Scheduled Tasks GUI can create a task with the Composer-selected provider/model route, run it immediately, project canonical Agent Run and Thread/Turn identities, and open the completed conversation. Localhost provider and isolated user data do not claim live-provider or Windows behavior.",
    testOnly: true,
    startedAt: new Date().toISOString(),
    completedAt: null,
    result: "fail",
    backendMode: "runtime",
    electron: null,
    route: null,
    task: null,
    gui: null,
    bridge: null,
    errors: null,
    artifacts: { screenshot: path.basename(screenshotPath) },
  };
  const raw = {};
  const consoleErrors = [];
  const pageErrors = [];
  let handle = null;
  let page = null;
  let providerFixture = null;

  try {
    console.log(`${LOG_PREFIX} stage=start-provider-fixture`);
    providerFixture = await startOpenAiCompatibleFixtureServer({
      content: FINAL_TEXT,
      model: FIXTURE_MODEL,
    });
    console.log(`${LOG_PREFIX} stage=launch-electron`);
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "runtime",
    });
    page = handle.page;
    summary.electron = await readElectronRuntime(handle.app);
    assert(handle.rendererSnapshot.electron, "real Electron renderer missing");
    assert(
      handle.rendererSnapshot.hasInvokeBridge,
      "Electron preload invoke bridge missing",
    );

    console.log(`${LOG_PREFIX} stage=configure-provider`);
    const route = await createFixtureProvider(page, providerFixture);
    const workspaceId = await selectFixtureRouteForWorkspace(page, route);
    const expectedModelRoute = encodeModelRouteSelector(
      route.providerId,
      route.model,
    );
    summary.route = {
      providerId: route.providerId,
      model: route.model,
      workspaceId,
      providerConfigured: true,
      providerRequestCount: providerFixture.requests.length,
      providerAuthorizationMatched: null,
      selectedInComposer: true,
      persistedTaskRouteMatched: null,
    };
    await clearInvokeBuffers(page);

    console.log(`${LOG_PREFIX} stage=create-task-through-gui`);
    const createdTask = await createTaskFromGui(page, route);
    assert(
      createdTask?.execution?.modelId === expectedModelRoute,
      "Scheduled Task did not persist the selected opaque provider/model route",
    );
    console.log(`${LOG_PREFIX} stage=run-task-through-gui`);
    await runTaskFromGui(page);
    const completed = await waitForCompletedRun(page, options);
    const serializedRead = JSON.stringify(completed.read.result || {});
    assert(
      serializedRead.includes(route.providerId) &&
        serializedRead.includes(route.model),
      "canonical Thread does not preserve the selected provider/model route",
    );
    assert(providerFixture.requests.length > 0, "provider was not invoked");
    assert(
      providerFixture.requests.every(
        (request) =>
          request.authorization ===
          `Bearer ${providerFixture.provider.providerConfig.apiKey}`,
      ),
      "provider authorization mismatch",
    );

    console.log(`${LOG_PREFIX} stage=open-run-conversation`);
    await openRunFromGui(page, completed.run);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    const traceRaw = await collectTrace(page);
    const bridge = summarizeScheduledTasksTrace(
      traceRaw.traceRaw,
      traceRaw.errorRaw,
    );
    assert(
      bridge.missingMethods.length === 0,
      `missing current methods: ${bridge.missingMethods.join(", ")}`,
    );
    assert(bridge.legacyMethods.length === 0, "legacy Automation method seen");
    assert(
      bridge.legacyCommands.length === 0,
      "legacy Automation command seen",
    );
    assert(bridge.mockFallbackHitCount === 0, "mock fallback seen");
    assert(bridge.invokeErrorCount === 0, "renderer invoke error seen");
    assert(
      consoleErrors.length === 0,
      `console error: ${consoleErrors.join(" | ")}`,
    );
    assert(pageErrors.length === 0, `page error: ${pageErrors.join(" | ")}`);

    summary.route = {
      providerId: route.providerId,
      model: route.model,
      workspaceId,
      providerConfigured: true,
      providerRequestCount: providerFixture.requests.length,
      providerAuthorizationMatched: true,
      selectedInComposer: true,
      persistedTaskRouteMatched: true,
    };
    summary.task = {
      taskId: completed.task.id,
      runId: completed.run.id,
      sessionId: completed.run.sessionId,
      threadId: completed.run.threadId,
      turnId: completed.run.turnId,
      status: completed.run.status,
      composerSelectedProviderModel: true,
    };
    summary.gui = {
      topLevelNavigationUsed: true,
      taskCreated: true,
      runStartedFromMoreMenu: true,
      runHistoryVisible: true,
      canonicalConversationOpened: true,
      finalTextVisible: true,
    };
    summary.bridge = bridge;
    summary.errors = {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      invokeErrorCount: bridge.invokeErrorCount,
      mockFallbackHitCount: bridge.mockFallbackHitCount,
      legacyHitCount:
        bridge.legacyMethods.length + bridge.legacyCommands.length,
    };
    raw.bridge = sanitizeJson(bridge);
    raw.canonicalReadModel = {
      threadIdentityMatched: serializedRead.includes(completed.run.threadId),
      turnIdentityMatched: serializedRead.includes(completed.run.turnId),
      providerRouteMatched:
        serializedRead.includes(route.providerId) &&
        serializedRead.includes(route.model),
      finalTextMatched: serializedRead.includes(FINAL_TEXT),
    };
    summary.result = "pass";
  } catch (error) {
    summary.error = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    if (page) {
      await page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
      summary.artifacts.failureScreenshot = path.basename(
        failureScreenshotPath,
      );
      const traceRaw = await collectTrace(page).catch(() => null);
      if (traceRaw) {
        raw.failureBridge = sanitizeJson(
          summarizeScheduledTasksTrace(traceRaw.traceRaw, traceRaw.errorRaw),
        );
      }
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
    await closeElectronFixture(handle);
    await providerFixture?.close().catch(() => undefined);
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
