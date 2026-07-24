#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import electronPath from "electron";
import { _electron as electron } from "playwright";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";

const DEFAULTS = {
  appUrl: "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "plugin-runtime-electron-fixture",
  ),
  prefix: "plugin-runtime-electron-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

const APP_ID = "content-factory-app";
const ENTRY_KEY = "writer";
const WORKSPACE_ID = "plugin-runtime-electron-workspace";
const SESSION_ID = "plugin-runtime-electron-session";
const TASK_ID = "plugin-runtime-electron-task";
const TURN_ID = "plugin-runtime-electron-turn";
const REQUEST_ID = "plugin-runtime-electron-action";
const EVENT_NAME = `plugin_runtime:${APP_ID}:${TASK_ID}`;
const COMMANDS = {
  startTask: "plugin_runtime_start_task",
  getTask: "plugin_runtime_get_task",
  cancelTask: "plugin_runtime_cancel_task",
  submitHostResponse: "plugin_runtime_submit_host_response",
};
const METHOD_PROOF = [
  "thread/start",
  "turn/start",
  "thread/read",
  "item/tool/requestUserInput",
  "serverRequest/resolved",
  "turn/interrupt",
];

function printHelp() {
  console.log(`
Plugin Runtime Electron Fixture Smoke

Purpose:
  Launch a real Electron Desktop Host with an explicit App Server external
  backend fixture, then invoke the Plugin task facade commands through the
  Electron preload bridge. The fixture backend records the App Server runtime
  requests it receives so this smoke can prove the facade preserved the typed
  RuntimeOptions.runtimeRequest.

Target path:
  Frontend -> Electron Desktop Host IPC -> App Server JSON-RPC
    -> RuntimeCore / external backend fixture

This is a smoke fixture, not a product fallback. The external backend is passed
only through this process environment. The production default remains
fail-closed when no real backend is configured.

Usage:
  node scripts/plugin/runtime-electron-fixture-smoke.mjs

Options:
  --app-url <url>        Optional renderer dev server, e.g. http://127.0.0.1:1420/
  --evidence-dir <path>  Evidence directory
  --prefix <name>        Evidence file prefix
  --timeout-ms <ms>      Total timeout, default 120000
  --interval-ms <ms>     Poll interval, default 250
  --keep-temp            Keep temporary dirs for debugging
  -h, --help             Show help
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--app-url" && next) {
      options.appUrl = next.trim();
      index += 1;
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
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be a number >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be a number >= 100");
  }
  if (!options.evidenceDir || !options.prefix) {
    throw new Error("--evidence-dir and --prefix are required");
  }
  return options;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function logStage(stage) {
  console.log(`[smoke:plugin-runtime-electron-fixture] stage=${stage}`);
}

function sanitizeText(value) {
  const sanitized = String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|session|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]");
  return sanitized.length > 2_000
    ? `${sanitized.slice(0, 2_000)}... [truncated ${sanitized.length - 2_000} chars]`
    : sanitized;
}

function sanitizeJson(value, depth = 0) {
  if (depth > 8) {
    return "[truncated-depth]";
  }
  if (typeof value === "string") {
    return sanitizeText(value);
  }
  if (
    value === null ||
    value === undefined ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 100).map((item) => sanitizeJson(item, depth + 1));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 140)
        .map(([key, item]) => [key, sanitizeJson(item, depth + 1)]),
    );
  }
  return sanitizeText(String(value));
}

function createTempRuntimeEnv() {
  const tempRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "plugin-runtime-electron-"),
  );
  const home = path.join(tempRoot, "home");
  const xdgDataHome = path.join(tempRoot, "xdg-data");
  const localAppData = path.join(tempRoot, "local-app-data");
  const roamingAppData = path.join(tempRoot, "roaming-app-data");
  const electronUserDataDir = path.join(tempRoot, "electron-user-data");
  const backendPath = path.join(tempRoot, "plugin-runtime-backend.mjs");
  const ledgerPath = path.join(
    tempRoot,
    "plugin-runtime-backend-ledger.jsonl",
  );

  for (const dir of [
    home,
    xdgDataHome,
    localAppData,
    roamingAppData,
    electronUserDataDir,
  ]) {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(ledgerPath, "");
  writeFixtureBackend(backendPath);

  return {
    tempRoot,
    backendPath,
    ledgerPath,
    electronUserDataDir,
    env: {
      ...process.env,
      HOME: home,
      XDG_DATA_HOME: xdgDataHome,
      APPDATA: roamingAppData,
      LOCALAPPDATA: localAppData,
    },
  };
}

function writeFixtureBackend(backendPath) {
  fs.writeFileSync(
    backendPath,
    `#!/usr/bin/env node
import { appendFileSync, readFileSync } from "node:fs";

const ledgerPath = process.argv[2];
const input = JSON.parse(readFileSync(0, "utf8"));

if (ledgerPath) {
  appendFileSync(ledgerPath, JSON.stringify({
    kind: input.kind,
    request: input.request,
    recordedAt: new Date().toISOString()
  }) + "\\n");
}

if (input.kind === "turnStart") {
  console.log(JSON.stringify({
    events: [
      {
        type: "message.delta",
        payload: {
          backend: "plugin-runtime-electron-fixture",
          text: "fixture task accepted"
        }
      },
      {
        type: "action.required",
        payload: {
          requestId: "${REQUEST_ID}",
          actionType: "ask_user",
          message: "fixture awaits host response"
        }
      }
    ]
  }));
  process.exit(0);
}

if (input.kind === "actionRespond") {
  console.log(JSON.stringify({
    events: [
      {
        type: "action.resolved",
        payload: {
          requestId: input.request.requestId,
          confirmed: input.request.confirmed
        }
      }
    ]
  }));
  process.exit(0);
}

if (input.kind === "turnCancel") {
  console.log(JSON.stringify({
    events: [
      {
        type: "turn.canceled",
        payload: {
          turnId: input.request.turn?.turnId
        }
      }
    ]
  }));
  process.exit(0);
}

console.log(JSON.stringify({ events: [] }));
`,
    { mode: 0o755 },
  );
}

function readBackendLedger(ledgerPath) {
  if (!fs.existsSync(ledgerPath)) {
    return [];
  }
  return fs
    .readFileSync(ledgerPath, "utf8")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function writeJsonFile(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function isTransientPageEvaluationError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return (
    message.includes("Execution context was destroyed") ||
    message.includes("most likely because of a navigation") ||
    message.includes("Cannot find context with specified id")
  );
}

async function evaluatePageSnapshot(page, pageFunction, arg) {
  try {
    return await page.evaluate(pageFunction, arg);
  } catch (error) {
    if (isTransientPageEvaluationError(error)) {
      return null;
    }
    throw error;
  }
}

async function waitForRendererReady(page, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => ({
      url: window.location.href,
      electron: window.__LIME_ELECTRON__ === true,
      hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
      supportsStart:
        typeof window.electronAPI?.supportsCommand === "function" &&
        window.electronAPI.supportsCommand("plugin_runtime_start_task"),
      startupVisible: Boolean(
        document.querySelector("[data-lime-startup-shell]"),
      ),
      appSidebarVisible: Boolean(
        document.querySelector('[data-testid="app-sidebar"]'),
      ),
      bodyText: document.body?.innerText || "",
    }));
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    if (
      snapshot.electron &&
      snapshot.hasInvokeBridge &&
      snapshot.supportsStart &&
      !snapshot.startupVisible &&
      snapshot.appSidebarVisible
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    "Electron renderer / Plugin runtime facade bridge not ready",
  );
}

async function clearInvokeBuffers(page) {
  await page.evaluate(() => {
    window.localStorage.removeItem("lime_invoke_error_buffer_v1");
    window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
  });
}

async function invokePluginRuntimeLifecycle(page) {
  return await page.evaluate(
    async ({
      appId,
      entryKey,
      workspaceId,
      sessionId,
      taskId,
      turnId,
      requestId,
      eventName,
      commands,
    }) => {
      const invoke = window.electronAPI?.invoke;
      if (typeof invoke !== "function") {
        throw new Error("Electron preload invoke bridge is unavailable");
      }

      const start = await invoke(commands.startTask, {
        request: {
          appId,
          entryKey,
          workspaceId,
          sessionId,
          taskId,
          turnId,
          taskKind: "content_factory.write",
          title: "Electron fixture task",
          prompt: "Generate a short fixture draft",
          input: { topic: "Plugin runtime Electron fixture" },
          expectedOutput: { markdown: true },
          eventName,
          queueIfBusy: true,
          skipPreSubmitResume: false,
          metadata: {
            source: "plugin-runtime-electron-fixture",
          },
          runtimeRequest: {
            providerConfig: {
              providerName: "fixture-provider",
              modelName: "fixture-model",
            },
            providerPreference: "fixture-provider",
            modelPreference: "fixture-model",
            systemPrompt: "You are running inside an Electron fixture.",
            reasoningEffort: "low",
            approvalPolicy: "never",
            sandboxPolicy: "workspace-write",
            webSearch: false,
            executionStrategy: "fixture",
            metadata: {
              turn_source: "plugin",
              fixture: true,
            },
          },
        },
      });

      const firstRead = await invoke(commands.getTask, {
        request: {
          appId,
          taskId,
          sessionId,
        },
      });

      const hostResponse = await invoke(commands.submitHostResponse, {
        request: {
          appId,
          taskId,
          runtimeRequest: {
            session_id: sessionId,
            request_id: requestId,
            action_type: "ask_user",
            confirmed: true,
            response: "continue from Electron fixture",
            user_data: { source: "electron-fixture" },
            metadata: { source: "plugin-runtime-electron-fixture" },
            event_name: `${eventName}:host_response`,
            action_scope: {
              session_id: sessionId,
              turn_id: turnId,
            },
          },
        },
      });

      const cancel = await invoke(commands.cancelTask, {
        request: {
          appId,
          taskId,
          sessionId,
        },
      });

      const finalRead = await invoke(commands.getTask, {
        request: {
          appId,
          taskId,
          sessionId,
        },
      });

      return {
        start,
        firstRead,
        hostResponse,
        cancel,
        finalRead,
      };
    },
    {
      appId: APP_ID,
      entryKey: ENTRY_KEY,
      workspaceId: WORKSPACE_ID,
      sessionId: SESSION_ID,
      taskId: TASK_ID,
      turnId: TURN_ID,
      requestId: REQUEST_ID,
      eventName: EVENT_NAME,
      commands: COMMANDS,
    },
  );
}

function assertLifecycleResult(result) {
  assert(
    result?.start?.status === "accepted",
    "start_task did not accept task",
  );
  assert(result.start.sessionId === SESSION_ID, "start_task session mismatch");
  assert(result.start.turnId === TURN_ID, "start_task turn mismatch");
  assert(
    result.firstRead?.taskStatus === "blocked",
    `get_task before response should be blocked, got ${result.firstRead?.taskStatus}`,
  );
  assert(
    result.hostResponse?.status === "submitted",
    "submit_host_response did not submit",
  );
  assert(result.cancel?.cancelled === true, "cancel_task did not cancel");
  assert(
    result.cancel?.status === "cancelled",
    `cancel_task status mismatch: ${result.cancel?.status}`,
  );
  assert(
    result.finalRead?.taskStatus === "cancelled",
    `final get_task should be cancelled, got ${result.finalRead?.taskStatus}`,
  );
}

function assertBackendLedger(ledgerEntries) {
  const kinds = ledgerEntries.map((entry) => entry.kind);
  for (const kind of ["turnStart", "actionRespond", "turnCancel"]) {
    assert(kinds.includes(kind), `backend ledger missing ${kind}`);
  }

  const turnStart = ledgerEntries.find((entry) => entry.kind === "turnStart");
  const actionRespond = ledgerEntries.find(
    (entry) => entry.kind === "actionRespond",
  );
  const turnCancel = ledgerEntries.find((entry) => entry.kind === "turnCancel");
  const runtimeRequest = turnStart?.request?.runtimeOptions?.runtimeRequest;

  assert(
    turnStart?.request?.session?.sessionId === SESSION_ID,
    "turnStart session mismatch",
  );
  assert(
    turnStart?.request?.turn?.turnId === TURN_ID,
    "turnStart turn mismatch",
  );
  assert(
    turnStart?.request?.runtimeOptions?.eventName === EVENT_NAME,
    "turnStart eventName mismatch",
  );
  assert(runtimeRequest, "turnStart missing runtimeRequest");
  assert(
    runtimeRequest.providerConfig?.providerName === "fixture-provider",
    "runtimeRequest.providerConfig was not preserved",
  );
  assert(
    runtimeRequest.systemPrompt === "You are running inside an Electron fixture.",
    "runtimeRequest.systemPrompt was not preserved",
  );
  assert(
    runtimeRequest.providerPreference === "fixture-provider" &&
      runtimeRequest.modelPreference === "fixture-model",
    "provider/model preference was not preserved",
  );
  assert(
    actionRespond?.request?.requestId === REQUEST_ID,
    "actionRespond requestId mismatch",
  );
  assert(
    actionRespond?.request?.confirmed === true,
    "actionRespond confirmed mismatch",
  );
  assert(
    actionRespond?.request?.actionScope?.turnId === TURN_ID,
    "actionRespond actionScope turn mismatch",
  );
  assert(
    turnCancel?.request?.turn?.turnId === TURN_ID,
    "turnCancel turn mismatch",
  );
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const backendLedgerEvidencePath = path.join(
    options.evidenceDir,
    `${options.prefix}-backend-ledger.json`,
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
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: runtimeEnv.env,
  });
  const summary = {
    ok: false,
    appId: APP_ID,
    sessionId: SESSION_ID,
    taskId: TASK_ID,
    turnId: TURN_ID,
    checkedAt: new Date().toISOString(),
    appUrl: options.appUrl || null,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    electronUserDataDir: options.keepTemp
      ? runtimeEnv.electronUserDataDir
      : null,
    backendPath: options.keepTemp ? runtimeEnv.backendPath : null,
    backendLedgerPath: options.keepTemp ? runtimeEnv.ledgerPath : null,
    backendLedgerEvidencePath,
    methodProof: METHOD_PROOF,
    electronPreloadBridge: false,
    commandsInvoked: Object.values(COMMANDS),
    lifecycle: null,
    backendKinds: [],
    screenshot: null,
    summary: summaryPath,
  };

  let app = null;
  let page = null;
  const consoleErrors = [];

  try {
    logStage("launch-electron");
    app = await electron.launch({
      executablePath: electronPath,
      args: ["--use-mock-keychain", "."],
      cwd: process.cwd(),
      env: {
        ...runtimeEnv.env,
        ...appServerEnv,
        APP_SERVER_BACKEND_MODE: "external",
        APP_SERVER_BACKEND_COMMAND: process.execPath,
        APP_SERVER_BACKEND_ARGS: JSON.stringify([
          runtimeEnv.backendPath,
          runtimeEnv.ledgerPath,
        ]),
        APP_SERVER_BACKEND_TIMEOUT_MS: "5000",
        ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
        LIME_ELECTRON_E2E: "1",
        LIME_ELECTRON_BRAND_DEV_APP: "0",
        LIME_ELECTRON_CLEAR_RENDERER_CACHE: "0",
        LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
        ...(options.appUrl ? { VITE_DEV_SERVER_URL: options.appUrl } : {}),
      },
      timeout: options.timeoutMs,
    });

    app.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(sanitizeText(message.text()));
      }
    });

    page = await app.firstWindow({ timeout: options.timeoutMs });
    page.setDefaultTimeout(options.timeoutMs);
    await page.setViewportSize({ width: 1440, height: 1000 });

    logStage("wait-renderer");
    const rendererSnapshot = await waitForRendererReady(page, options);
    summary.electronPreloadBridge =
      rendererSnapshot.electron && rendererSnapshot.hasInvokeBridge;
    await clearInvokeBuffers(page);

    logStage("invoke-plugin-runtime-facade");
    const lifecycle = await invokePluginRuntimeLifecycle(page);
    assertLifecycleResult(lifecycle);
    summary.lifecycle = sanitizeJson(lifecycle);

    logStage("verify-backend-ledger");
    const backendLedger = readBackendLedger(runtimeEnv.ledgerPath);
    assertBackendLedger(backendLedger);
    summary.backendKinds = backendLedger.map((entry) => entry.kind);
    writeJsonFile(backendLedgerEvidencePath, backendLedger.map(sanitizeJson));

    await page.screenshot({ path: screenshotPath, fullPage: true });
    summary.screenshot = screenshotPath;
    summary.consoleErrors = consoleErrors;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(summaryPath, summary);
    console.log(
      `[smoke:plugin-runtime-electron-fixture] summary=${summaryPath}`,
    );
    console.log(
      `[smoke:plugin-runtime-electron-fixture] backendKinds=${summary.backendKinds.join(",")}`,
    );
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    summary.consoleErrors = consoleErrors;
    writeJsonFile(summaryPath, summary);
    if (page) {
      try {
        await page.screenshot({
          path: failureScreenshotPath,
          fullPage: true,
        });
        summary.failureScreenshot = failureScreenshotPath;
        writeJsonFile(summaryPath, summary);
      } catch {
        // ignore screenshot failure
      }
    }
    throw error;
  } finally {
    if (app) {
      await app.close().catch(() => undefined);
    }
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

run().catch((error) => {
  console.error(
    `[smoke:plugin-runtime-electron-fixture] failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
