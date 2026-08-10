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
  MEDIA_SERVICES_REQUIRED_HOST_COMMANDS,
  MEDIA_SERVICES_REQUIRED_CONFIG_METHODS,
  MEDIA_SERVICES_REQUIRED_METHODS,
  applyFailedSettingsMediaServicesEvidence,
  applyPassingSettingsMediaServicesEvidence,
  createSettingsMediaServicesEvidence,
  parseSettingsMediaServicesFixtureArgs,
  summarizeSettingsMediaServicesTrace,
} from "./lib/settings-media-services-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "settings-media-services-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

function printHelp() {
  console.log(`
Settings Media Services Electron Fixture

Usage:
  node scripts/electron/settings-media-services-fixture-smoke.mjs --run-id <id>

Options:
  --run-id <id> --evidence-dir <path> --prefix <name>
  --timeout-ms <ms> --interval-ms <ms> --keep-temp -h|--help
`);
}

async function readMediaServicesState(page, options) {
  return await waitForPageCondition(
    page,
    options,
    ({ requiredMethods, requiredConfigMethods, requiredHostCommands }) => {
      const bodyText = document.body?.innerText ?? "";
      const active =
        document
          .querySelector('[data-testid="settings-sidebar-tab-media-services"]')
          ?.getAttribute("data-active") === "true";
      const loadingVisible =
        document.querySelectorAll(".animate-pulse").length > 0;
      const errorVisible =
        bodyText.includes("加载语音设置失败") ||
        bodyText.includes("保存服务模型配置失败");
      const serviceModelsVisible = bodyText.includes("服务模型");
      const imageServiceVisible = bodyText.includes("图片服务模型");
      const videoServiceVisible = bodyText.includes("视频服务模型");
      const voiceServiceVisible = bodyText.includes("语音服务模型");
      const configControlsReady = Array.from(
        document.querySelectorAll(
          'input[type="number"], button[role="switch"]',
        ),
      ).some((element) => !element.hasAttribute("disabled"));
      const traceRaw = window.localStorage.getItem(
        "lime_invoke_trace_buffer_v1",
      );
      let methods = [];
      let hostCommands = [];
      try {
        const entries = JSON.parse(traceRaw || "[]");
        const safeEntries = Array.isArray(entries) ? entries : [];
        hostCommands = safeEntries
          .filter((entry) => entry?.transport === "electron-ipc")
          .map((entry) => entry?.command);
        methods = safeEntries.flatMap((entry) => {
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
        methods = [];
        hostCommands = [];
      }
      if (
        !active ||
        loadingVisible ||
        errorVisible ||
        !serviceModelsVisible ||
        !imageServiceVisible ||
        !videoServiceVisible ||
        !voiceServiceVisible ||
        !configControlsReady ||
        !requiredMethods.every((method) => methods.includes(method)) ||
        !requiredConfigMethods.every((method) => methods.includes(method)) ||
        !requiredHostCommands.every((command) => hostCommands.includes(command))
      ) {
        return null;
      }
      return {
        url: window.location.href,
        mediaServicesActive: true,
        serviceModelsVisible,
        imageServiceVisible,
        videoServiceVisible,
        voiceServiceVisible,
        configControlsReady,
        loadingVisible,
        errorVisible,
        traceRaw,
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    "Media Services Settings did not reach terminal current readiness",
    {
      requiredMethods: MEDIA_SERVICES_REQUIRED_METHODS,
      requiredConfigMethods: MEDIA_SERVICES_REQUIRED_CONFIG_METHODS,
      requiredHostCommands: MEDIA_SERVICES_REQUIRED_HOST_COMMANDS,
    },
  );
}

async function run() {
  const options = parseSettingsMediaServicesFixtureArgs(process.argv.slice(2), {
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
  const screenshotPath = file(".png");
  const imageScreenshotPath = file("-image.png");
  const videoScreenshotPath = file("-video.png");
  const readinessScreenshotPath = file("-readiness.png");
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
  const summary = createSettingsMediaServicesEvidence({
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
      window.localStorage.removeItem("lime_invoke_error_buffer_v1");
      window.localStorage.setItem(
        "lime.app-config.changed-at",
        `settings-media-services-fixture-${Date.now()}`,
      );
    });
    await page
      .locator('[data-testid="settings-sidebar-tab-media-services"]')
      .click();
    const state = await readMediaServicesState(page, options);
    const trace = summarizeSettingsMediaServicesTrace(state.traceRaw);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    await page
      .getByRole("heading", { name: "图片服务模型", exact: true })
      .scrollIntoViewIfNeeded();
    await page.screenshot({ path: imageScreenshotPath, fullPage: true });
    await page
      .getByRole("heading", { name: "视频服务模型", exact: true })
      .scrollIntoViewIfNeeded();
    await page.screenshot({ path: videoScreenshotPath, fullPage: true });
    await page
      .getByRole("heading", { name: "语音服务模型", exact: true })
      .scrollIntoViewIfNeeded();
    await page.screenshot({ path: readinessScreenshotPath, fullPage: true });
    applyPassingSettingsMediaServicesEvidence(summary, {
      completedAt: new Date().toISOString(),
      electronRenderer: handle.rendererSnapshot.electron,
      preloadInvoke: handle.rendererSnapshot.hasInvokeBridge,
      mediaServicesActive: state.mediaServicesActive,
      serviceModelsVisible: state.serviceModelsVisible,
      imageServiceVisible: state.imageServiceVisible,
      videoServiceVisible: state.videoServiceVisible,
      voiceServiceVisible: state.voiceServiceVisible,
      configControlsReady: state.configControlsReady,
      loadingVisible: state.loadingVisible,
      errorVisible: state.errorVisible,
      trace,
      consoleErrors,
      pageErrors,
      invokeErrorCount: parseInvokeTraceRaw(state.errorRaw).length,
      screenshotWritten: fs.existsSync(screenshotPath),
      imageScreenshotWritten: fs.existsSync(imageScreenshotPath),
      videoScreenshotWritten: fs.existsSync(videoScreenshotPath),
      readinessScreenshotWritten: fs.existsSync(readinessScreenshotPath),
    });
    const runtimeUrl = new URL(state.url);
    writeJsonFile(rawEvidencePath, {
      runtime: {
        protocol: runtimeUrl.protocol,
        nativeStartup: runtimeUrl.searchParams.get("nativeStartup") === "1",
      },
      appServerMethods: trace.methods,
      hostCommands: trace.hostCommands,
      readiness: {
        serviceModelsVisible: state.serviceModelsVisible,
        imageServiceVisible: state.imageServiceVisible,
        videoServiceVisible: state.videoServiceVisible,
        voiceServiceVisible: state.voiceServiceVisible,
        configControlsReady: state.configControlsReady,
      },
    });
    writeJsonFile(summaryPath, summary);
    console.log(
      `[smoke:settings-media-services-fixture] summary=${summaryPath}`,
    );
  } catch (error) {
    applyFailedSettingsMediaServicesEvidence(summary, error);
    summary.consoleErrors = consoleErrors.map(sanitizeText);
    summary.pageErrors = pageErrors.map(sanitizeText);
    if (page) {
      const traceRaw = await page
        .evaluate(() =>
          window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        )
        .catch(() => null);
      summary.failureTrace = summarizeSettingsMediaServicesTrace(traceRaw);
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
    `[smoke:settings-media-services-fixture] ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
