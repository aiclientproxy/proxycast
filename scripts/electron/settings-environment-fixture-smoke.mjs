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
  readConfigFromPage,
  parseInvokeTraceRaw,
  sanitizeText,
  waitForPageCondition,
} from "./mcp-config-fixture-smoke.mjs";
import {
  ENVIRONMENT_REQUIRED_HOST_COMMANDS,
  applyFailedSettingsEnvironmentEvidence,
  applyPassingSettingsEnvironmentEvidence,
  createSettingsEnvironmentEvidence,
  parseSettingsEnvironmentFixtureArgs,
  summarizeSettingsEnvironmentTrace,
} from "./lib/settings-environment-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "settings-environment-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

async function readEnvironmentState(page, options) {
  return await waitForPageCondition(
    page,
    options,
    ({ requiredHostCommands }) => {
      const bodyText = document.body?.innerText ?? "";
      const active =
        document
          .querySelector('[data-testid="settings-sidebar-tab-environment"]')
          ?.getAttribute("data-active") === "true";
      const loadingVisible =
        document.querySelectorAll(".animate-pulse").length > 0;
      const errorVisible =
        bodyText.includes("加载环境变量配置失败") ||
        bodyText.includes("刷新环境预览失败");
      const traceRaw = window.localStorage.getItem(
        "lime_invoke_trace_buffer_v1",
      );
      let commands = [];
      try {
        const entries = JSON.parse(traceRaw || "[]");
        commands = (Array.isArray(entries) ? entries : [])
          .filter((entry) => entry?.transport === "electron-ipc")
          .map((entry) => entry?.command);
      } catch {
        commands = [];
      }
      if (
        !active ||
        !bodyText.includes("环境变量") ||
        loadingVisible ||
        errorVisible ||
        !requiredHostCommands.every((command) => commands.includes(command))
      ) {
        return null;
      }
      return {
        url: window.location.href,
        environmentActive: true,
        loadingVisible,
        errorVisible,
        traceRaw,
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    "Environment Settings did not reach a terminal current read state",
    { requiredHostCommands: ENVIRONMENT_REQUIRED_HOST_COMMANDS },
  );
}

async function probeCurrentConfig(page) {
  const { config } = await readConfigFromPage(page);
  return Boolean(
    typeof config.default_provider === "string" &&
      config.default_provider.trim(),
  );
}

async function run() {
  const options = parseSettingsEnvironmentFixtureArgs(process.argv.slice(2), {
    defaults: DEFAULTS,
  });
  if (options.help) return;
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const file = (suffix) =>
    path.join(options.evidenceDir, `${options.prefix}${suffix}`);
  const summaryPath = file("-summary.json");
  const rawEvidencePath = file("-raw.json");
  const screenshotPath = file(".png");
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
  const summary = createSettingsEnvironmentEvidence({
    candidateRunId: options.runId,
    startedAt: new Date().toISOString(),
    prefix: options.prefix,
  });
  let handle = null;
  let page = null;
  const consoleErrors = [];
  const pageErrors = [];
  try {
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    page = handle.page;
    await openSettings(page, options);
    await page.evaluate(() => {
      window.localStorage.setItem(
        "lime.app-config.changed-at",
        `settings-environment-fixture-${Date.now()}`,
      );
    });
    await page
      .locator('[data-testid="settings-sidebar-tab-environment"]')
      .click();
    const configShapeValid = await probeCurrentConfig(page);
    const state = await readEnvironmentState(page, options);
    const trace = summarizeSettingsEnvironmentTrace(state.traceRaw);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    applyPassingSettingsEnvironmentEvidence(summary, {
      completedAt: new Date().toISOString(),
      electronRenderer: handle.rendererSnapshot.electron,
      preloadInvoke: handle.rendererSnapshot.hasInvokeBridge,
      configShapeValid,
      environmentActive: state.environmentActive,
      loadingVisible: state.loadingVisible,
      errorVisible: state.errorVisible,
      trace,
      consoleErrors,
      pageErrors,
      invokeErrorCount: parseInvokeTraceRaw(state.errorRaw).length,
      screenshotWritten: fs.existsSync(screenshotPath),
    });
    writeJsonFile(rawEvidencePath, {
      url: state.url,
      appServerMethods: trace.appServerMethods,
      hostCommands: trace.hostCommands,
    });
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:settings-environment-fixture] summary=${summaryPath}`);
  } catch (error) {
    applyFailedSettingsEnvironmentEvidence(summary, error);
    summary.consoleErrors = consoleErrors.map(sanitizeText);
    summary.pageErrors = pageErrors.map(sanitizeText);
    if (page) {
      const traceRaw = await page
        .evaluate(() =>
          window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        )
        .catch(() => null);
      summary.failureTrace = summarizeSettingsEnvironmentTrace(traceRaw);
    }
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
    `[smoke:settings-environment-fixture] ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
