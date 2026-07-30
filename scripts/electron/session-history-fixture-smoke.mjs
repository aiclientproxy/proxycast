#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { execFileSync } from "node:child_process";
import electronPath from "electron";
import { _electron as electron } from "playwright";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { withElectronFixtureSystemPath } from "../lib/electron-fixture-runtime-env.mjs";
import {
  THREAD_READ_PAGE_ISOMORPHIC,
  seedThreadReadPageIsomorphicCanonicalThread,
} from "./lib/session-history-thread-read-isomorphic-fixture.mjs";
import {
  THREAD_READ_LONG_LIST,
  seedThreadReadLongListCanonicalThread,
} from "./lib/session-history-long-list-fixture.mjs";
import {
  assertThreadReadLongListDomOracle,
  runThreadReadLongListDomOracle,
} from "./lib/session-history-long-list-oracle.mjs";
import {
  assertThreadReadPageIsomorphicDomOracle,
  assertThreadReadPageIsomorphicReadModel,
  runThreadReadPageIsomorphicDomOracle,
  runThreadReadPageIsomorphicReadPhase,
  ThreadReadPageIsomorphicDomError,
} from "./lib/session-history-thread-read-isomorphic-oracle.mjs";
import {
  assertThreadArchivePhase,
  assertThreadUnarchivePhase,
  runThreadArchivePhase,
  runThreadUnarchivePhase,
} from "./lib/session-history-thread-archive-oracle.mjs";

const DEFAULTS = {
  appUrl: "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "agent-session-history-electron-fixture",
  ),
  prefix: "agent-session-history-electron-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const CURRENT_METHODS = [
  "initialize",
  "thread/start",
  "thread/archive",
  "thread/unarchive",
  "thread/read",
  "thread/list",
  "thread/turns/list",
  "thread/items/list",
  "thread/resume",
];
const FORBIDDEN_METHODS = [
  "turn/start",
  "agentSession/update",
  "agentSession/archiveMany",
];
const LAST_PROJECT_ID_KEY = "agent_last_project_id";
const OPENED_PROJECT_IDS_KEY = "agent_opened_project_ids";
const APP_SIDEBAR_COLLAPSED_STORAGE_KEY = "lime.app-sidebar.collapsed";
const SQLITE3_BINARY = process.env.SQLITE3_BIN?.trim() || "sqlite3";

function printHelp() {
  console.log(`
Agent Session History Electron Fixture Smoke

用途:
  启动真实 Electron Desktop Host，通过 app_server_handle_json_lines 验证
  canonical archive/unarchive 文件移动与 thread/read/list/turns/resume Gate B。

边界:
  fixture 创建一个 canonical Thread 验证归档生命周期，再直接 seed canonical
  ThreadStore 验证历史读取；不启动 Turn，不使用 legacy command 或 mock fallback。

用法:
  node scripts/electron/session-history-fixture-smoke.mjs

选项:
  --app-url <url>        可选 renderer dev server，例如 http://127.0.0.1:1420/
  --evidence-dir <path>  证据目录
  --prefix <name>        证据文件前缀
  --timeout-ms <ms>      总超时，默认 120000
  --interval-ms <ms>     轮询间隔，默认 250
  --keep-temp            保留临时目录便于调试
  -h, --help             显示帮助
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
    } else if (arg === "--evidence-dir" && next) {
      options.evidenceDir = path.resolve(next.trim());
      index += 1;
    } else if (arg === "--prefix" && next) {
      options.prefix = next.trim();
      index += 1;
    } else if (arg === "--timeout-ms" && next) {
      options.timeoutMs = Number(next);
      index += 1;
    } else if (arg === "--interval-ms" && next) {
      options.intervalMs = Number(next);
      index += 1;
    } else if (arg === "--keep-temp") {
      options.keepTemp = true;
    }
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!options.evidenceDir || !options.prefix) {
    throw new Error("--evidence-dir / --prefix 均不能为空");
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
  console.log(`[smoke:agent-session-history-electron-fixture] stage=${stage}`);
}

function createTempRuntimeEnv() {
  const tempRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "agent-session-history-electron-"),
  );
  const home = path.join(tempRoot, "home");
  const xdgDataHome = path.join(tempRoot, "xdg-data");
  const localAppData = path.join(tempRoot, "local-app-data");
  const roamingAppData = path.join(tempRoot, "roaming-app-data");
  const electronUserDataDir = path.join(tempRoot, "electron-user-data");
  const persistedWorkspaceRoot = path.join(tempRoot, "persisted-workspace");
  for (const dir of [
    home,
    xdgDataHome,
    localAppData,
    roamingAppData,
    electronUserDataDir,
    persistedWorkspaceRoot,
  ]) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return {
    tempRoot,
    electronUserDataDir,
    persistedWorkspaceRoot,
    env: withElectronFixtureSystemPath({
      ...process.env,
      HOME: home,
      XDG_DATA_HOME: xdgDataHome,
      APPDATA: roamingAppData,
      LOCALAPPDATA: localAppData,
    }),
  };
}

function sanitizeText(value) {
  const sanitized = String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]");
  return sanitized.length > 2_000
    ? `${sanitized.slice(0, 2_000)}... [truncated ${sanitized.length - 2_000} chars]`
    : sanitized;
}

function sanitizeJson(value, depth = 0) {
  if (depth > 8) return "[truncated-depth]";
  if (typeof value === "string") return sanitizeText(value);
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
        .slice(0, 160)
        .map(([key, item]) => [key, sanitizeJson(item, depth + 1)]),
    );
  }
  return sanitizeText(String(value));
}

function writeJsonFile(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function sqlLiteral(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function runSqlite(dbPath, sql) {
  if (!fs.existsSync(dbPath)) {
    throw new Error(`fixture 数据库尚未创建: ${dbPath}`);
  }
  try {
    execFileSync(SQLITE3_BINARY, [dbPath], {
      input: sql,
      encoding: "utf8",
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (error) {
    throw new Error(
      `${SQLITE3_BINARY} fixture seed 失败: ${error?.message || error}; stdout=${sanitizeText(error?.stdout)}; stderr=${sanitizeText(error?.stderr)}`,
    );
  }
}

function findRolloutPaths(runtimeEnv, threadId) {
  const agentRoot = path.join(runtimeEnv.electronUserDataDir, "app-server");
  const collect = (relativeRoot) => {
    const absoluteRoot = path.join(agentRoot, relativeRoot);
    if (!fs.existsSync(absoluteRoot)) return [];
    const found = [];
    const visit = (directory) => {
      for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
        const absolutePath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
          visit(absolutePath);
        } else if (
          entry.isFile() &&
          entry.name.endsWith(".jsonl") &&
          entry.name.includes(threadId)
        ) {
          found.push(path.relative(agentRoot, absolutePath));
        }
      }
    };
    visit(absoluteRoot);
    return found.sort();
  };
  return {
    active: collect("sessions"),
    archived: collect("archived_sessions"),
  };
}

async function waitForRendererReady(page, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const snapshot = await page.evaluate(() => ({
        electron: window.__LIME_ELECTRON__ === true,
        hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
        supportsAppServer:
          typeof window.electronAPI?.supportsCommand === "function" &&
          window.electronAPI.supportsCommand("app_server_handle_json_lines"),
        startupVisible: Boolean(
          document.querySelector("[data-lime-startup-shell]"),
        ),
        appSidebarVisible: Boolean(
          document.querySelector('[data-testid="app-sidebar"]'),
        ),
      }));
      if (
        snapshot.electron &&
        snapshot.hasInvokeBridge &&
        snapshot.supportsAppServer &&
        !snapshot.startupVisible &&
        snapshot.appSidebarVisible
      ) {
        return snapshot;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (
        !message.includes("Execution context was destroyed") &&
        !message.includes("Cannot find context with specified id")
      ) {
        throw error;
      }
    }
    await sleep(options.intervalMs);
  }
  throw new Error("Electron renderer / App Server bridge 未就绪");
}

async function clearInvokeBuffers(page) {
  await page.evaluate(() => {
    window.localStorage.removeItem("lime_invoke_error_buffer_v1");
    window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
  });
}

async function setSidebarWorkspace(page, workspaceId) {
  await page.evaluate(
    ({ workspaceId, lastProjectIdKey, openedProjectIdsKey, collapsedKey }) => {
      if (workspaceId) {
        window.localStorage.setItem(
          lastProjectIdKey,
          JSON.stringify(workspaceId),
        );
        window.localStorage.setItem(
          openedProjectIdsKey,
          JSON.stringify([workspaceId]),
        );
      } else {
        window.localStorage.removeItem(lastProjectIdKey);
        window.localStorage.setItem(openedProjectIdsKey, JSON.stringify([]));
      }
      window.localStorage.setItem(collapsedKey, "false");
      window.dispatchEvent(
        new CustomEvent("agent-persisted-project-id-changed", {
          detail: { key: lastProjectIdKey, projectId: workspaceId || null },
        }),
      );
      window.dispatchEvent(
        new CustomEvent("agent-opened-project-ids-changed", {
          detail: { projectIds: workspaceId ? [workspaceId] : [] },
        }),
      );
      window.dispatchEvent(
        new CustomEvent("lime:app-sidebar-collapse", {
          detail: { collapsed: false },
        }),
      );
      window.dispatchEvent(new Event("focus"));
    },
    {
      workspaceId,
      lastProjectIdKey: LAST_PROJECT_ID_KEY,
      openedProjectIdsKey: OPENED_PROJECT_IDS_KEY,
      collapsedKey: APP_SIDEBAR_COLLAPSED_STORAGE_KEY,
    },
  );
}

async function launchElectronFixture({
  options,
  runtimeEnv,
  appServerEnv,
  consoleErrors,
  pageErrors,
}) {
  const app = await electron.launch({
    executablePath: electronPath,
    args: ["--use-mock-keychain", "."],
    cwd: process.cwd(),
    env: {
      ...runtimeEnv.env,
      ...appServerEnv,
      APP_SERVER_BACKEND_MODE: "unavailable",
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
  const page = await app.firstWindow({ timeout: options.timeoutMs });
  page.on("pageerror", (error) => pageErrors.push(sanitizeText(error.message)));
  page.setDefaultTimeout(options.timeoutMs);
  await page.setViewportSize({ width: 1440, height: 1000 });
  const rendererSnapshot = await waitForRendererReady(page, options);
  await clearInvokeBuffers(page);
  return { app, page, rendererSnapshot };
}

async function closeElectronFixture(handle) {
  if (handle?.app) {
    await handle.app.close().catch(() => undefined);
    await sleep(500);
  }
}

async function startCanonicalHistoryThread(page, cwd, options = {}) {
  return await page.evaluate(
    async ({ command, cwd, requestId, title }) => {
      const invoke = window.electronAPI?.invoke;
      if (typeof invoke !== "function") {
        throw new Error("Electron preload invoke bridge is unavailable");
      }
      const id = requestId;
      const params = {
        model: "fixture-model",
        modelProvider: "fixture-provider",
        cwd,
        serviceName: title,
        historyMode: "legacy",
        threadSource: "fixture",
      };
      const response = await invoke(command, {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id,
              method: "thread/start",
              params,
            }),
          ],
        },
      });
      const messages = (response?.lines ?? [])
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch {
            return null;
          }
        })
        .filter(Boolean);
      const error = messages.find(
        (message) => message?.id === id && message.error,
      );
      if (error) {
        throw new Error(`thread/start failed: ${JSON.stringify(error.error)}`);
      }
      const result = messages.find(
        (message) =>
          message?.id === id &&
          Object.prototype.hasOwnProperty.call(message, "result"),
      );
      if (!result?.result?.thread?.id) {
        throw new Error("thread/start did not return a canonical thread");
      }
      return { result: result.result, params, messages };
    },
    {
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      cwd,
      requestId: options.requestId || "thread-read-isomorphic-setup",
      title: options.title || THREAD_READ_PAGE_ISOMORPHIC.title,
    },
  );
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const rawEvidencePath = path.join(
    options.evidenceDir,
    `${options.prefix}-raw.json`,
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );
  const longListScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-long-list.png`,
  );
  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    appUrl: options.appUrl || null,
    backendMode: "unavailable",
    proofLevel: "Gate B controlled fixture",
    currentMethods: CURRENT_METHODS,
    forbiddenMethods: FORBIDDEN_METHODS,
    electronPreloadBridge: false,
    databaseBootstrapRestart: false,
    threadArchiveSummary: null,
    threadUnarchiveSummary: null,
    archivedRolloutPaths: null,
    restoredRolloutPaths: null,
    threadReadPageIsomorphicSeed: null,
    threadReadPageIsomorphicSummary: null,
    threadReadLongListSeed: null,
    threadReadLongListSummary: null,
    consoleErrors: [],
    pageErrors: [],
    screenshot: null,
    longListScreenshot: null,
    rawEvidence: rawEvidencePath,
    summary: summaryPath,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
  };
  const rawEvidence = {};
  const consoleErrors = [];
  const pageErrors = [];
  let handle = null;

  try {
    logStage("bootstrap-current-databases");
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    summary.electronPreloadBridge =
      handle.rendererSnapshot.electron &&
      handle.rendererSnapshot.hasInvokeBridge;

    logStage("thread-archive-current");
    const archiveResult = await runThreadArchivePhase(
      handle.page,
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      runtimeEnv.persistedWorkspaceRoot,
    );
    rawEvidence.threadArchive = sanitizeJson(archiveResult);
    summary.threadArchiveSummary = sanitizeJson(
      assertThreadArchivePhase(archiveResult),
    );
    await closeElectronFixture(handle);
    handle = null;
    summary.databaseBootstrapRestart = true;

    const archivedRolloutPaths = findRolloutPaths(
      runtimeEnv,
      summary.threadArchiveSummary.threadId,
    );
    assert(
      archivedRolloutPaths.active.length === 0 &&
        archivedRolloutPaths.archived.length === 1,
      `thread/archive 未形成唯一 archived rollout: ${JSON.stringify(archivedRolloutPaths)}`,
    );
    summary.archivedRolloutPaths = archivedRolloutPaths;

    logStage("thread-unarchive-after-restart");
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    const unarchiveResult = await runThreadUnarchivePhase(
      handle.page,
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      summary.threadArchiveSummary.threadId,
    );
    rawEvidence.threadUnarchive = sanitizeJson(unarchiveResult);
    summary.threadUnarchiveSummary = sanitizeJson(
      assertThreadUnarchivePhase(
        unarchiveResult,
        summary.threadArchiveSummary.threadId,
      ),
    );

    const restoredRolloutPaths = findRolloutPaths(
      runtimeEnv,
      summary.threadArchiveSummary.threadId,
    );
    assert(
      restoredRolloutPaths.active.length === 1 &&
        restoredRolloutPaths.archived.length === 0,
      `thread/unarchive 未恢复唯一 dated rollout: ${JSON.stringify(restoredRolloutPaths)}`,
    );
    summary.restoredRolloutPaths = restoredRolloutPaths;

    logStage("start-thread-read-page-isomorphic");
    const historySetup = await startCanonicalHistoryThread(
      handle.page,
      runtimeEnv.persistedWorkspaceRoot,
    );
    rawEvidence.threadReadPageIsomorphicSetup = sanitizeJson(historySetup);

    logStage("seed-thread-read-page-isomorphic");
    const threadSeed = seedThreadReadPageIsomorphicCanonicalThread({
      runtimeEnv,
      runSqlite,
      sqlLiteral,
      thread: historySetup?.result?.thread,
    });
    summary.threadReadPageIsomorphicSeed = sanitizeJson(threadSeed);

    logStage("run-electron-thread-read-page-isomorphic");
    await clearInvokeBuffers(handle.page);
    const threadReadResult = await runThreadReadPageIsomorphicReadPhase(
      handle.page,
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    );
    rawEvidence.threadReadPageIsomorphicRead = sanitizeJson(threadReadResult);
    const threadReadSummary =
      assertThreadReadPageIsomorphicReadModel(threadReadResult);
    await clearInvokeBuffers(handle.page);
    await setSidebarWorkspace(
      handle.page,
      THREAD_READ_PAGE_ISOMORPHIC.workspaceId,
    );
    let threadDomResult;
    try {
      threadDomResult = await runThreadReadPageIsomorphicDomOracle(
        handle.page,
        options,
      );
    } catch (error) {
      if (error instanceof ThreadReadPageIsomorphicDomError) {
        rawEvidence.threadReadPageIsomorphicDomFailure = sanitizeJson(
          error.evidence,
        );
      }
      throw error;
    }
    rawEvidence.threadReadPageIsomorphicDom = sanitizeJson(threadDomResult);
    summary.threadReadPageIsomorphicSummary = sanitizeJson({
      read: threadReadSummary,
      dom: assertThreadReadPageIsomorphicDomOracle(threadDomResult),
    });
    await handle.page.screenshot({ path: screenshotPath, fullPage: true });

    logStage("start-thread-read-long-list");
    const longListSetup = await startCanonicalHistoryThread(
      handle.page,
      runtimeEnv.persistedWorkspaceRoot,
      {
        requestId: "thread-read-long-list-setup",
        title: THREAD_READ_LONG_LIST.title,
      },
    );
    rawEvidence.threadReadLongListSetup = sanitizeJson(longListSetup);

    logStage("seed-thread-read-long-list");
    const longListSeed = seedThreadReadLongListCanonicalThread({
      runtimeEnv,
      runSqlite,
      sqlLiteral,
      thread: longListSetup?.result?.thread,
    });
    summary.threadReadLongListSeed = sanitizeJson({
      sessionId: longListSeed.sessionId,
      threadId: longListSeed.threadId,
      turnCount: longListSeed.turnCount,
      itemCount: longListSeed.itemCount,
    });

    logStage("run-electron-thread-read-long-list");
    await handle.page.reload({ waitUntil: "domcontentloaded" });
    await waitForRendererReady(handle.page, options);
    await clearInvokeBuffers(handle.page);
    await setSidebarWorkspace(handle.page, THREAD_READ_LONG_LIST.workspaceId);
    await handle.page.evaluate(() => window.__LIME_AGENTUI_PERF__?.clear?.());
    const longListDomResult = await runThreadReadLongListDomOracle(
      handle.page,
      options,
    );
    rawEvidence.threadReadLongListDom = sanitizeJson(longListDomResult);
    summary.threadReadLongListSummary = sanitizeJson(
      assertThreadReadLongListDomOracle(longListDomResult),
    );
    await handle.page.screenshot({
      path: longListScreenshotPath,
      fullPage: false,
    });
    await closeElectronFixture(handle);
    handle = null;

    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );
    assert(
      pageErrors.length === 0,
      `观察到 page error: ${pageErrors.join(" | ")}`,
    );
    summary.consoleErrors = consoleErrors;
    summary.pageErrors = pageErrors;
    summary.screenshot = screenshotPath;
    summary.longListScreenshot = longListScreenshotPath;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(rawEvidencePath, rawEvidence);
    writeJsonFile(summaryPath, summary);
    console.log(
      `[smoke:agent-session-history-electron-fixture] summary=${summaryPath}`,
    );
    console.log(
      `[smoke:agent-session-history-electron-fixture] methods=${summary.threadReadPageIsomorphicSummary.read.requestMethods.join(",")}`,
    );
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    summary.consoleErrors = consoleErrors;
    summary.pageErrors = pageErrors;
    if (Object.keys(rawEvidence).length > 0) {
      writeJsonFile(rawEvidencePath, rawEvidence);
    }
    if (handle?.page) {
      try {
        await handle.page.screenshot({
          path: failureScreenshotPath,
          fullPage: true,
        });
        summary.failureScreenshot = failureScreenshotPath;
      } catch {
        // 截图失败不覆盖原始错误。
      }
    }
    writeJsonFile(summaryPath, summary);
    throw error;
  } finally {
    if (handle) await closeElectronFixture(handle);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

run().catch((error) => {
  console.error(
    `[smoke:agent-session-history-electron-fixture] failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
