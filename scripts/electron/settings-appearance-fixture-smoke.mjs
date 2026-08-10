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
  applyFailedSettingsAppearanceEvidence,
  applyPassingSettingsAppearanceEvidence,
  createSettingsAppearanceEvidence,
  parseSettingsAppearanceFixtureArgs,
  summarizeSettingsAppearanceTrace,
} from "./lib/settings-appearance-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "settings-appearance-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const THEME_LABELS = {
  light: "浅色",
  dark: "深色",
  system: "跟随系统",
};
const BEHAVIOR_ARIA = "切换推荐自动附带选中内容";

function printHelp() {
  console.log(`
Settings Appearance Electron Fixture

Usage:
  node scripts/electron/settings-appearance-fixture-smoke.mjs --run-id <id>

Options:
  --run-id <id> --evidence-dir <path> --prefix <name>
  --timeout-ms <ms> --interval-ms <ms> --keep-temp -h|--help
`);
}

async function openAppearanceSettings(page, options) {
  await openSettings(page, options);
  await page.evaluate(() => {
    window.localStorage.setItem(
      "lime.app-config.changed-at",
      `settings-appearance-fixture-${Date.now()}`,
    );
  });
  await page.locator('[data-testid="settings-sidebar-tab-appearance"]').click();
}

async function readAppearanceState(
  page,
  options,
  { expectedTheme = null, expectedBehavior = null, requireSave = false } = {},
) {
  return await waitForPageCondition(
    page,
    options,
    ({ expectedTheme, expectedBehavior, requireSave, behaviorAria }) => {
      const bodyText = document.body?.innerText ?? "";
      const active =
        document
          .querySelector('[data-testid="settings-sidebar-tab-appearance"]')
          ?.getAttribute("data-active") === "true";
      const behaviorSwitch = document.querySelector(
        `[aria-label="${behaviorAria}"]`,
      );
      const behavior = behaviorSwitch?.getAttribute("aria-checked") === "true";
      const theme =
        window.localStorage.getItem("theme") ||
        document.documentElement.dataset.limeTheme ||
        "system";
      const loadingVisible =
        document.querySelectorAll(".animate-pulse").length > 0;
      const errorVisible = bodyText.includes("外观设置失败");
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
          return Array.isArray(entry?.args_preview?.request?.lines)
            ? entry.args_preview.request.lines
            : [];
        });
      } catch {
        commands = [];
        appServerMethods = [];
      }
      if (
        !active ||
        !bodyText.includes("外观") ||
        !(behaviorSwitch instanceof HTMLButtonElement) ||
        behaviorSwitch.hasAttribute("disabled") ||
        loadingVisible ||
        errorVisible ||
        !appServerMethods.includes("config/read") ||
        (expectedTheme && theme !== expectedTheme) ||
        (typeof expectedBehavior === "boolean" &&
          behavior !== expectedBehavior) ||
        (requireSave && !appServerMethods.includes("config/batchWrite"))
      ) {
        return null;
      }
      return {
        appearanceTabActive: true,
        themeControlReady: true,
        behaviorControlReady: true,
        theme,
        behavior,
        loadingVisible,
        errorVisible,
        traceRaw,
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    "Appearance Settings did not reach the required dual-owner state",
    {
      expectedTheme,
      expectedBehavior,
      requireSave,
      behaviorAria: BEHAVIOR_ARIA,
    },
  );
}

async function selectTheme(page, theme) {
  const label = THEME_LABELS[theme];
  await page.getByRole("button").filter({ hasText: label }).first().click();
}

async function toggleBehavior(page) {
  await page.getByRole("switch", { name: BEHAVIOR_ARIA, exact: true }).click();
}

async function run() {
  const options = parseSettingsAppearanceFixtureArgs(process.argv.slice(2), {
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
  const behaviorScreenshotPath = file("-behavior.png");
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
  const summary = createSettingsAppearanceEvidence({
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
    await openAppearanceSettings(page, options);
    const initialState = await readAppearanceState(page, options);
    const originalTheme = initialState.theme;
    const originalBehavior = initialState.behavior;
    const changedTheme = originalTheme === "light" ? "dark" : "light";
    const changedBehavior = !originalBehavior;
    await selectTheme(page, changedTheme);
    await toggleBehavior(page);
    const savedState = await readAppearanceState(page, options, {
      expectedTheme: changedTheme,
      expectedBehavior: changedBehavior,
      requireSave: true,
    });
    traceRaws.push(savedState.traceRaw);
    errorRaws.push(savedState.errorRaw);
    await page.screenshot({ path: behaviorScreenshotPath, fullPage: true });
    await page
      .getByRole("heading", { name: "主题模式", exact: true })
      .scrollIntoViewIfNeeded();
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
    await openAppearanceSettings(page, options);
    const restartState = await readAppearanceState(page, options, {
      expectedTheme: changedTheme,
      expectedBehavior: changedBehavior,
    });
    await page.screenshot({ path: restartScreenshotPath, fullPage: true });
    await selectTheme(page, originalTheme);
    await toggleBehavior(page);
    const restoredSaveState = await readAppearanceState(page, options, {
      expectedTheme: originalTheme,
      expectedBehavior: originalBehavior,
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
    await openAppearanceSettings(page, options);
    const finalState = await readAppearanceState(page, options, {
      expectedTheme: originalTheme,
      expectedBehavior: originalBehavior,
    });
    traceRaws.push(finalState.traceRaw);
    errorRaws.push(finalState.errorRaw);
    await page.screenshot({ path: restoredScreenshotPath, fullPage: true });
    const trace = summarizeSettingsAppearanceTrace(traceRaws);
    applyPassingSettingsAppearanceEvidence(summary, {
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
      appearanceTabActive: finalState.appearanceTabActive,
      themeControlReady: finalState.themeControlReady,
      behaviorControlReady: finalState.behaviorControlReady,
      themeChanged: changedTheme !== originalTheme,
      behaviorChanged: changedBehavior !== originalBehavior,
      restartThemeReadback: restartState.theme === changedTheme,
      restartBehaviorReadback: restartState.behavior === changedBehavior,
      restorationSaved:
        restoredSaveState.theme === originalTheme &&
        restoredSaveState.behavior === originalBehavior,
      restorationThemeReadback: finalState.theme === originalTheme,
      restorationBehaviorReadback: finalState.behavior === originalBehavior,
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
      behaviorScreenshotWritten: fs.existsSync(behaviorScreenshotPath),
      restartScreenshotWritten: fs.existsSync(restartScreenshotPath),
      restoredScreenshotWritten: fs.existsSync(restoredScreenshotPath),
    });
    writeJsonFile(rawEvidencePath, {
      lifecycle: {
        isolatedUserData: true,
        themeChanged: true,
        behaviorChanged: true,
        restartThemeReadback: true,
        restartBehaviorReadback: true,
        restorationSaved: true,
        restorationThemeReadback: true,
        restorationBehaviorReadback: true,
      },
      appServerMethods: trace.methods,
      hostCommands: trace.hostCommands,
    });
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:settings-appearance-fixture] summary=${summaryPath}`);
  } catch (error) {
    applyFailedSettingsAppearanceEvidence(summary, error);
    summary.consoleErrors = consoleErrors.map(sanitizeText);
    summary.pageErrors = pageErrors.map(sanitizeText);
    summary.failureTrace = summarizeSettingsAppearanceTrace(traceRaws);
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
    `[smoke:settings-appearance-fixture] ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
