#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { ensureElectronFixtureBuild } from "../lib/electron-fixture-build.mjs";
import { writeJsonFile } from "../mcp/lib/current-smoke-transport.mjs";
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

const DEFAULTS = {
  evidenceDir: null,
  prefix: "apps-catalog-gate-b",
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const LOG_PREFIX = "[smoke:apps-catalog-gate-b]";
const PLUGIN_ID = "apps-catalog-gate-b-plugin";
const PLUGIN_NAME = "Apps Catalog Gate B";
const APP_ID = "apps-catalog-gate-b-app";
const REQUIRED_METHODS = [
  "plugin/list",
  "plugin/install",
  "plugin/read",
  "plugin/enabled/set",
  "app/list",
  "app/read",
  "app/installed",
];
const LEGACY_COMMANDS = [
  "plugin_runtime_start_task",
  "plugin_runtime_get_task",
  "plugin_runtime_submit_host_response",
  "plugin_runtime_cancel_task",
];

function printHelp() {
  console.log(`
Apps Catalog Electron Gate B

用途:
  验证真实 Electron App Center 消费 App Server app/list、app/read、app/installed，
  并在 GUI 停用 Plugin 后由 typed app/list/updated 触发 Apps readiness fresh read。

边界:
  本地 Plugin 没有 hosted connector tool snapshot，必须保持 callable=false。
  本场景不调用 live provider，不使用 App Server mock backend 或 renderer fallback。

选项:
  --run-id <id> --evidence-dir <path> --prefix <name>
  --timeout-ms <ms> --interval-ms <ms> --keep-temp -h|--help
`);
}

function standaloneRunId() {
  const timestamp = new Date().toISOString().replace(/[-:.]/g, "");
  const suffix = Math.floor(Math.random() * 1_000_000)
    .toString()
    .padStart(6, "0");
  return `standalone-apps-catalog-${timestamp}-${suffix}`;
}

export function parseArgs(argv, defaults = DEFAULTS) {
  const options = { ...defaults, help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    if (arg === "--run-id" && next) {
      options.runId = next.trim();
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
    throw new Error(`未知参数: ${arg}`);
  }
  if (options.help) return options;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(options.runId || "")) {
    options.runId = standaloneRunId();
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(options.prefix || "")) {
    throw new Error("--prefix 只能包含字母、数字、点、下划线和连字符");
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  options.evidenceDir ||= path.join(
    process.cwd(),
    ".lime",
    "qc",
    "project-gates",
    options.runId,
    "apps-catalog-gate-b",
  );
  return options;
}

function preparePluginPackage(runtimeEnv) {
  const root = path.join(runtimeEnv.tempRoot, "apps-catalog-plugin");
  const manifestDir = path.join(root, ".codex-plugin");
  fs.mkdirSync(manifestDir, { recursive: true });
  fs.writeFileSync(
    path.join(root, "plugin.json"),
    `${JSON.stringify(
      {
        $schema: "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        name: PLUGIN_ID,
        version: "1.0.0",
        description: "Apps catalog Gate B fixture",
      },
      null,
      2,
    )}\n`,
  );
  fs.writeFileSync(
    path.join(manifestDir, "plugin.json"),
    `${JSON.stringify(
      {
        interface: {
          displayName: PLUGIN_NAME,
          shortDescription: "Apps catalog readiness fixture",
        },
        apps: "./apps.json",
      },
      null,
      2,
    )}\n`,
  );
  fs.writeFileSync(
    path.join(root, "apps.json"),
    `${JSON.stringify(
      {
        apps: {
          [PLUGIN_NAME]: { id: APP_ID, category: "productivity" },
        },
      },
      null,
      2,
    )}\n`,
  );
  return { root };
}

async function drainPendingNotifications(page) {
  return await page.evaluate(async () => {
    const response = await window.electronAPI.invoke(
      "app_server_drain_events",
      { request: { limit: 50 } },
    );
    const lines = Array.isArray(response?.lines) ? response.lines : [];
    return lines.flatMap((line) => {
      try {
        const message = JSON.parse(String(line));
        return typeof message?.method === "string" ? [message.method] : [];
      } catch {
        return [];
      }
    });
  });
}

async function readTraceSnapshot(page) {
  const buffers = await page.evaluate(() => ({
    errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
  }));
  const trace = parseInvokeTraceRaw(buffers.traceRaw);
  const requests = parseJsonRpcRequestsFromInvokeTrace(buffers.traceRaw);
  const methodCounts = {};
  for (const request of requests) {
    methodCounts[request.method] = (methodCounts[request.method] ?? 0) + 1;
  }
  const commands = Array.from(
    new Set(trace.map((entry) => entry?.command).filter(Boolean)),
  );
  return {
    commands,
    errorCount: parseInvokeTraceRaw(buffers.errorRaw).length,
    legacyCommands: LEGACY_COMMANDS.filter((command) =>
      commands.includes(command),
    ),
    methodCounts,
    methods: Object.keys(methodCounts),
    mockFallbackHitCount: trace.filter(
      (entry) =>
        entry?.mock === true ||
        entry?.mockFallback === true ||
        (entry?.command === "app_server_handle_json_lines" &&
          entry?.transport !== "electron-ipc"),
    ).length,
    traceErrorCount: trace.filter((entry) => entry?.status === "error").length,
    traceRaw: buffers.traceRaw,
  };
}

async function waitForReadinessRow(page, options, expected) {
  const locator = page.locator(
    `[data-testid="plugin-catalog-app-readiness-${APP_ID}"]`,
  );
  const deadline = Date.now() + options.timeoutMs;
  let latest = null;
  while (Date.now() < deadline) {
    if (await locator.isVisible().catch(() => false)) {
      latest = {
        callable: await locator.getAttribute("data-callable"),
        enabled: await locator.getAttribute("data-enabled"),
        status: await locator.getAttribute("data-status"),
        text: (await locator.textContent())?.trim() ?? "",
      };
      if (
        latest.callable === String(expected.callable) &&
        latest.enabled === String(expected.enabled) &&
        latest.status === expected.status
      ) {
        return latest;
      }
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Apps readiness 行未收敛: expected=${JSON.stringify(expected)} actual=${JSON.stringify(latest)}`,
  );
}

async function waitForNotificationRefresh(page, options, baselineCounts) {
  const deadline = Date.now() + options.timeoutMs;
  let latestTrace = null;
  let latestRow = null;
  while (Date.now() < deadline) {
    latestTrace = await readTraceSnapshot(page);
    latestRow = await waitForReadinessRow(
      page,
      { ...options, timeoutMs: Math.min(1_000, options.timeoutMs) },
      { callable: false, enabled: false, status: "disabled" },
    ).catch(() => null);
    const listCount = latestTrace.methodCounts["app/list"] ?? 0;
    const installedCount = latestTrace.methodCounts["app/installed"] ?? 0;
    if (
      latestRow &&
      listCount > (baselineCounts["app/list"] ?? 0) &&
      installedCount > (baselineCounts["app/installed"] ?? 0)
    ) {
      return { row: latestRow, trace: latestTrace };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `app/list/updated 后未观察到 fresh read: baseline=${JSON.stringify(baselineCounts)} latest=${JSON.stringify(latestTrace?.methodCounts)} row=${JSON.stringify(latestRow)}`,
  );
}

function appById(result) {
  return result?.data?.find((app) => app?.id === APP_ID) ?? null;
}

function installedAppById(result) {
  return result?.apps?.find((app) => app?.id === APP_ID) ?? null;
}

export async function run(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  if (options.help) {
    printHelp();
    return;
  }
  ensureElectronFixtureBuild({
    logPrefix: LOG_PREFIX,
    rootDir: process.cwd(),
  });
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const rawPath = path.join(options.evidenceDir, `${options.prefix}-raw.json`);
  const initialScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-pending.png`,
  );
  const disabledScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-disabled.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );
  const runtimeEnv = createTempRuntimeEnv();
  const plugin = preparePluginPackage(runtimeEnv);
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const consoleErrors = [];
  const pageErrors = [];
  const observedMethods = new Set();
  let handle = null;
  let page = null;
  const summary = {
    schemaVersion: 1,
    scenarioId: "APPS-01-catalog-readiness-notification",
    proofLevel: "Gate B",
    claimBoundary:
      "Real Electron App Center Apps catalog/read/installed projection and typed app/list/updated refresh for a local Plugin. Local callable remains false; this does not claim hosted connector tool readiness or live provider behavior.",
    runId: options.runId,
    startedAt: new Date().toISOString(),
    completedAt: null,
    result: "fail",
    backendMode: "unavailable",
    pluginId: PLUGIN_ID,
    appId: APP_ID,
    initialReadiness: null,
    disabledReadiness: null,
    installNotificationObserved: false,
    notificationFreshReadObserved: false,
    requiredMethods: REQUIRED_METHODS,
    missingRequiredMethods: [...REQUIRED_METHODS],
    bridge: {
      electron: false,
      preloadInvoke: false,
      command: "app_server_handle_json_lines",
      transport: null,
    },
    errors: {
      consoleErrorCount: 0,
      pageErrorCount: 0,
      invokeErrorCount: 0,
      traceErrorCount: 0,
      mockFallbackHitCount: 0,
      legacyCommandHitCount: 0,
    },
    artifacts: {
      initialScreenshot: initialScreenshotPath,
      disabledScreenshot: disabledScreenshotPath,
      raw: rawPath,
      summary: summaryPath,
    },
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
  };
  const raw = {};

  try {
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "unavailable",
    });
    page = handle.page;
    summary.bridge.electron = handle.rendererSnapshot.electron;
    summary.bridge.preloadInvoke = handle.rendererSnapshot.hasInvokeBridge;

    const catalog = await appServerCallFromPage(page, "plugin/list", {
      marketplacePaths: [plugin.root],
    });
    observedMethods.add(catalog.method);
    const candidate = catalog.result?.plugins?.find(
      (entry) => entry?.id === PLUGIN_ID,
    );
    assert(candidate?.appsCount === 1, "plugin/list 未返回 Apps fixture");
    const installed = await appServerCallFromPage(page, "plugin/install", {
      sourcePath: plugin.root,
      marketplaceId: "apps-catalog-gate-b",
      source: "repo",
      expectedDigest: candidate.contentDigest,
    });
    observedMethods.add(installed.method);
    assert(installed.result?.plugin?.enabled === true, "Plugin 安装后未启用");
    const installNotifications = await drainPendingNotifications(page);
    summary.installNotificationObserved =
      installNotifications.includes("app/list/updated");
    assert(
      summary.installNotificationObserved,
      "plugin/install 未发布 typed app/list/updated",
    );

    await page.locator('[data-testid="app-sidebar-nav-plugins"]').click();
    await page.locator('[data-testid="plugin-catalog-loading"]').waitFor({
      state: "hidden",
      timeout: options.timeoutMs,
    });
    await page
      .locator(`[data-testid="plugin-catalog-details-${PLUGIN_ID}"]`)
      .click();
    summary.initialReadiness = await waitForReadinessRow(page, options, {
      callable: false,
      enabled: true,
      status: "pending",
    });
    await page.screenshot({ path: initialScreenshotPath, fullPage: true });

    const appsList = await appServerCallFromPage(page, "app/list", {});
    observedMethods.add(appsList.method);
    const appsRead = await appServerCallFromPage(page, "app/read", {
      appIds: [APP_ID],
      includeTools: true,
    });
    observedMethods.add(appsRead.method);
    const appsInstalled = await appServerCallFromPage(
      page,
      "app/installed",
      {},
    );
    observedMethods.add(appsInstalled.method);
    const appInfo = appById(appsList.result);
    const appRuntime = installedAppById(appsInstalled.result);
    assert(
      appInfo?.isEnabled === true && appInfo?.isAccessible === true,
      "app/list 初始 enabled/accessibility 不正确",
    );
    assert(
      appsRead.result?.apps?.[0]?.id === APP_ID &&
        appsRead.result?.apps?.[0]?.toolSummaries?.length === 0,
      "app/read 未返回 Apps metadata/tool summary",
    );
    assert(
      appRuntime?.enabled === true && appRuntime?.callable === false,
      "app/installed 必须保持 local callable=false",
    );
    const baselineTrace = await readTraceSnapshot(page);

    await page
      .locator(`[data-testid="plugin-catalog-actions-${PLUGIN_ID}"]`)
      .click();
    await page
      .locator(`[data-testid="plugin-catalog-toggle-${PLUGIN_ID}"]`)
      .click();
    const refreshed = await waitForNotificationRefresh(
      page,
      options,
      baselineTrace.methodCounts,
    );
    summary.disabledReadiness = refreshed.row;
    summary.notificationFreshReadObserved = true;
    await page.screenshot({ path: disabledScreenshotPath, fullPage: true });

    const disabledList = await appServerCallFromPage(page, "app/list", {});
    observedMethods.add(disabledList.method);
    const disabledInstalled = await appServerCallFromPage(
      page,
      "app/installed",
      {},
    );
    observedMethods.add(disabledInstalled.method);
    assert(
      appById(disabledList.result)?.isEnabled === false,
      "app/list 未投影 disabled 状态",
    );
    assert(
      installedAppById(disabledInstalled.result)?.enabled === false,
      "app/installed 未投影 disabled 状态",
    );

    const finalTrace = await readTraceSnapshot(page);
    const finalMethods = Array.from(
      new Set([...observedMethods, ...finalTrace.methods]),
    );
    summary.missingRequiredMethods = REQUIRED_METHODS.filter(
      (method) => !finalMethods.includes(method),
    );
    summary.bridge.transport = finalTrace.commands.includes(
      "app_server_handle_json_lines",
    )
      ? "electron-ipc"
      : null;
    summary.errors = {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      invokeErrorCount: finalTrace.errorCount,
      traceErrorCount: finalTrace.traceErrorCount,
      mockFallbackHitCount: finalTrace.mockFallbackHitCount,
      legacyCommandHitCount: finalTrace.legacyCommands.length,
    };
    assert(summary.bridge.electron, "未运行在真实 Electron renderer");
    assert(summary.bridge.preloadInvoke, "Electron preload invoke 不可用");
    assert(
      summary.bridge.transport === "electron-ipc",
      "未观察到 app_server_handle_json_lines electron-ipc",
    );
    assert(
      summary.missingRequiredMethods.length === 0,
      `缺少 current method: ${summary.missingRequiredMethods.join(", ")}`,
    );
    assert(consoleErrors.length === 0, "Renderer console error 不为零");
    assert(pageErrors.length === 0, "Renderer page error 不为零");
    assert(finalTrace.errorCount === 0, "invoke error buffer 不为零");
    assert(finalTrace.traceErrorCount === 0, "invoke trace error 不为零");
    assert(
      finalTrace.mockFallbackHitCount === 0,
      "观察到 production mock fallback",
    );
    assert(
      finalTrace.legacyCommands.length === 0,
      "观察到 legacy Plugin command",
    );

    raw.initial = {
      catalog: catalog.result,
      install: installed.result,
      installNotifications,
      appsList: appsList.result,
      appsRead: appsRead.result,
      appsInstalled: appsInstalled.result,
      traceMethodCounts: baselineTrace.methodCounts,
    };
    raw.disabled = {
      appsList: disabledList.result,
      appsInstalled: disabledInstalled.result,
      traceMethodCounts: finalTrace.methodCounts,
      observedMethods: finalMethods,
      commands: finalTrace.commands,
    };
    summary.result = "pass";
    summary.completedAt = new Date().toISOString();
    writeJsonFile(rawPath, raw);
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
  } catch (error) {
    summary.error = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    summary.errors.consoleErrorCount = consoleErrors.length;
    summary.errors.pageErrorCount = pageErrors.length;
    writeJsonFile(summaryPath, summary);
    if (page) {
      await page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
    }
    throw error;
  } finally {
    if (handle) await closeElectronFixture(handle);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  run().catch((error) => {
    console.error(
      `${LOG_PREFIX} ${error instanceof Error ? error.message : String(error)}`,
    );
    process.exitCode = 1;
  });
}
