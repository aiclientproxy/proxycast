#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { writeJsonFile } from "../mcp/lib/current-smoke-transport.mjs";
import {
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  openSettings,
  parseInvokeTraceRaw,
  sanitizeText,
  waitForPageCondition,
} from "./mcp-config-fixture-smoke.mjs";
import {
  applyFailedSettingsWebSearchEvidence,
  applyPassingSettingsWebSearchEvidence,
  createSettingsWebSearchEvidence,
  parseSettingsWebSearchFixtureArgs,
  summarizeSettingsWebSearchTrace,
} from "./lib/settings-web-search-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "settings-web-search-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

function printHelp() {
  console.log(`
Settings Web Search Electron Fixture

Usage:
  node scripts/electron/settings-web-search-fixture-smoke.mjs --run-id <id>

Options:
  --run-id <id> --evidence-dir <path> --prefix <name>
  --timeout-ms <ms> --interval-ms <ms> --keep-temp -h|--help
`);
}

async function openWebSearchSettings(page, options) {
  await openSettings(page, options);
  await page.evaluate(() => {
    window.localStorage.setItem(
      "lime.app-config.changed-at",
      `settings-web-search-fixture-${Date.now()}`,
    );
  });
  await page.locator('[data-testid="settings-sidebar-tab-web-search"]').click();
}

async function readWebSearchState(
  page,
  options,
  { expectedEngine = null, requireSave = false } = {},
) {
  return await waitForPageCondition(
    page,
    options,
    ({ expectedEngine, requireSave }) => {
      const bodyText = document.body?.innerText ?? "";
      const active =
        document
          .querySelector('[data-testid="settings-sidebar-tab-web-search"]')
          ?.getAttribute("data-active") === "true";
      const engineSelect = document.querySelector("#web-search-engine");
      const engine =
        engineSelect instanceof HTMLSelectElement ? engineSelect.value : null;
      const loadingVisible =
        document.querySelectorAll(".animate-pulse").length > 0;
      const errorVisible =
        bodyText.includes("加载配置失败:") || bodyText.includes("保存失败:");
      const routeControlReady =
        engineSelect instanceof HTMLSelectElement && !engineSelect.disabled;
      const allSavedVisible = bodyText.includes("所有更改已保存");
      const saveConfirmed = bodyText.includes("网络搜索设置已保存");
      const traceRaw = window.localStorage.getItem(
        "lime_invoke_trace_buffer_v1",
      );
      let commands = [];
      let appServerMethods = [];
      try {
        const entries = JSON.parse(traceRaw || "[]");
        const safeEntries = Array.isArray(entries) ? entries : [];
        commands = safeEntries
          .filter((entry) => entry?.transport === "electron-ipc")
          .map((entry) => entry?.command);
        appServerMethods = safeEntries.flatMap((entry) => {
          if (
            entry?.command !== "app_server_handle_json_lines" ||
            entry?.transport !== "electron-ipc"
          ) {
            return [];
          }
          const lines = entry?.args_preview?.request?.lines;
          if (!Array.isArray(lines)) return [];
          return lines.flatMap((line) => {
            try {
              const request = JSON.parse(String(line));
              return typeof request?.method === "string"
                ? [request.method]
                : [];
            } catch {
              return [];
            }
          });
        });
      } catch {
        commands = [];
        appServerMethods = [];
      }
      if (
        !active ||
        !bodyText.includes("网络搜索") ||
        !routeControlReady ||
        loadingVisible ||
        errorVisible ||
        !appServerMethods.includes("config/read") ||
        (expectedEngine && engine !== expectedEngine) ||
        (requireSave &&
          (!appServerMethods.includes("config/batchWrite") ||
            !saveConfirmed ||
            !allSavedVisible))
      ) {
        return null;
      }
      return {
        webSearchTabActive: true,
        routeControlReady,
        engine,
        loadingVisible,
        errorVisible,
        saveConfirmed,
        allSavedVisible,
        traceRaw,
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    "Web Search Settings did not reach the required current route state",
    { expectedEngine, requireSave },
  );
}

async function run() {
  const options = parseSettingsWebSearchFixtureArgs(process.argv.slice(2), {
    defaults: DEFAULTS,
  });
  if (options.help) {
    printHelp();
    return;
  }
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const file = (suffix) =>
    path.join(options.evidenceDir, `${options.prefix}${suffix}`);
  const summaryPath = file("-summary.json");
  const rawEvidencePath = file("-raw.json");
  const savedScreenshotPath = file("-saved.png");
  const restartScreenshotPath = file("-restart.png");
  const restoredScreenshotPath = file("-restored.png");
  const failureScreenshotPath = file("-failure.png");
  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const summary = createSettingsWebSearchEvidence({
    candidateRunId: options.runId,
    startedAt: new Date().toISOString(),
    prefix: options.prefix,
  });
  let handle = null;
  let page = null;
  const consoleErrors = [];
  const pageErrors = [];
  const traceRaws = [];
  const errorRaws = [];
  const rendererSnapshots = [];
  try {
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    rendererSnapshots.push(handle.rendererSnapshot);
    page = handle.page;
    await openWebSearchSettings(page, options);
    const initialState = await readWebSearchState(page, options);
    const originalEngine = initialState.engine;
    const changedEngine =
      originalEngine === "google" ? "xiaohongshu" : "google";
    await page.locator("#web-search-engine").selectOption(changedEngine);
    await page.getByRole("button", { name: "保存", exact: true }).click();
    const savedState = await readWebSearchState(page, options, {
      expectedEngine: changedEngine,
      requireSave: true,
    });
    traceRaws.push(savedState.traceRaw);
    errorRaws.push(savedState.errorRaw);
    await page.screenshot({ path: savedScreenshotPath, fullPage: true });
    await closeElectronFixture(handle);
    handle = null;
    page = null;

    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    rendererSnapshots.push(handle.rendererSnapshot);
    page = handle.page;
    await openWebSearchSettings(page, options);
    const restartState = await readWebSearchState(page, options, {
      expectedEngine: changedEngine,
    });
    await page.screenshot({ path: restartScreenshotPath, fullPage: true });
    await page.locator("#web-search-engine").selectOption(originalEngine);
    await page.getByRole("button", { name: "保存", exact: true }).click();
    const restoredSaveState = await readWebSearchState(page, options, {
      expectedEngine: originalEngine,
      requireSave: true,
    });
    traceRaws.push(restartState.traceRaw, restoredSaveState.traceRaw);
    errorRaws.push(restartState.errorRaw, restoredSaveState.errorRaw);
    await closeElectronFixture(handle);
    handle = null;
    page = null;

    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    rendererSnapshots.push(handle.rendererSnapshot);
    page = handle.page;
    await openWebSearchSettings(page, options);
    const finalState = await readWebSearchState(page, options, {
      expectedEngine: originalEngine,
    });
    traceRaws.push(finalState.traceRaw);
    errorRaws.push(finalState.errorRaw);
    await page.screenshot({ path: restoredScreenshotPath, fullPage: true });
    const trace = summarizeSettingsWebSearchTrace(traceRaws);
    applyPassingSettingsWebSearchEvidence(summary, {
      completedAt: new Date().toISOString(),
      electronLaunchCount: rendererSnapshots.filter(
        (snapshot) => snapshot.electron,
      ).length,
      preloadLaunchCount: rendererSnapshots.filter(
        (snapshot) => snapshot.hasInvokeBridge,
      ).length,
      isolatedUserData: runtimeEnv.electronUserDataDir.startsWith(
        runtimeEnv.tempRoot,
      ),
      webSearchTabActive: finalState.webSearchTabActive,
      routeControlReady: finalState.routeControlReady,
      routeChanged: changedEngine !== originalEngine,
      saveConfirmed: savedState.saveConfirmed && savedState.allSavedVisible,
      restartReadback: restartState.engine === changedEngine,
      restorationSaveConfirmed:
        restoredSaveState.saveConfirmed && restoredSaveState.allSavedVisible,
      restorationReadback: finalState.engine === originalEngine,
      loadingVisible: finalState.loadingVisible,
      errorVisible: finalState.errorVisible,
      trace,
      consoleErrors,
      pageErrors,
      invokeErrorCount: errorRaws.reduce(
        (count, raw) => count + parseInvokeTraceRaw(raw).length,
        0,
      ),
      savedScreenshotWritten: fs.existsSync(savedScreenshotPath),
      restartScreenshotWritten: fs.existsSync(restartScreenshotPath),
      restoredScreenshotWritten: fs.existsSync(restoredScreenshotPath),
    });
    writeJsonFile(rawEvidencePath, {
      lifecycle: {
        isolatedUserData: true,
        routeChanged: true,
        saveConfirmed: true,
        restartReadback: true,
        restorationSaveConfirmed: true,
        restorationReadback: true,
      },
      appServerMethods: trace.methods,
      hostCommands: trace.hostCommands,
    });
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:settings-web-search-fixture] summary=${summaryPath}`);
  } catch (error) {
    applyFailedSettingsWebSearchEvidence(summary, error);
    summary.consoleErrors = consoleErrors.map(sanitizeText);
    summary.pageErrors = pageErrors.map(sanitizeText);
    summary.failureTrace = summarizeSettingsWebSearchTrace(traceRaws);
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

run().catch((error) => {
  console.error(
    `[smoke:settings-web-search-fixture] ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
