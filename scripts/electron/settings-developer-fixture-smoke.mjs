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
  applyFailedSettingsDeveloperEvidence,
  applyPassingSettingsDeveloperEvidence,
  createSettingsDeveloperEvidence,
  parseSettingsDeveloperFixtureArgs,
  summarizeSettingsDeveloperTrace,
} from "./lib/settings-developer-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "settings-developer-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

function printHelp() {
  console.log(`
Settings Developer Electron Fixture

Usage:
  node scripts/electron/settings-developer-fixture-smoke.mjs --run-id <id>

Options:
  --run-id <id> --evidence-dir <path> --prefix <name>
  --timeout-ms <ms> --interval-ms <ms> --keep-temp -h|--help
`);
}

async function installClipboardSink(page) {
  return await page.evaluate(() => {
    const sink = {
      writeCount: 0,
      textLength: 0,
      jsonObject: false,
      payloadShape: {},
    };
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text) => {
          const value = String(text ?? "");
          let payload = null;
          try {
            payload = JSON.parse(value);
          } catch {
            payload = null;
          }
          const runtimeSnapshot =
            payload?.runtime_snapshot &&
            typeof payload.runtime_snapshot === "object"
              ? payload.runtime_snapshot
              : null;
          const currentLogPath = String(
            payload?.log_storage_diagnostics?.current_log_path ?? "",
          )
            .replace(/\\/g, "/")
            .replace(/\/+/g, "/");
          sink.writeCount += 1;
          sink.textLength = value.length;
          sink.jsonObject = Boolean(
            payload && typeof payload === "object" && !Array.isArray(payload),
          );
          sink.payloadShape = {
            generatedAt: typeof payload?.generated_at === "string",
            desktopRuntime: payload?.runtime === "desktop-host",
            persistedLogTail: Array.isArray(payload?.persisted_log_tail),
            serverDiagnostics: Boolean(
              payload?.server_diagnostics &&
              typeof payload.server_diagnostics === "object",
            ),
            logStorageDiagnostics: Boolean(
              payload?.log_storage_diagnostics &&
              typeof payload.log_storage_diagnostics === "object",
            ),
            currentLogPathAtAgentRoot: currentLogPath.endsWith(
              "/app-server/observability/log/lime.log",
            ),
            windowsStartupDiagnostics: Boolean(
              payload?.windows_startup_diagnostics &&
              typeof payload.windows_startup_diagnostics === "object",
            ),
            runtimeSnapshot: Boolean(runtimeSnapshot),
            configSummary: Boolean(
              runtimeSnapshot?.config_summary &&
              typeof runtimeSnapshot.config_summary === "object",
            ),
            providerSummary: Boolean(
              runtimeSnapshot?.api_key_provider_summary &&
              typeof runtimeSnapshot.api_key_provider_summary === "object",
            ),
            mcpSummary: Boolean(
              runtimeSnapshot?.mcp_summary &&
              typeof runtimeSnapshot.mcp_summary === "object",
            ),
          };
        },
      },
    });
    window.__LIME_SETTINGS_DEVELOPER_CLIPBOARD_SINK__ = sink;
    return window.navigator.clipboard?.writeText instanceof Function;
  });
}

async function openDeveloperSettings(page, options) {
  await openSettings(page, options);
  await page.evaluate(() => {
    window.localStorage.setItem(
      "lime.app-config.changed-at",
      `settings-developer-fixture-${Date.now()}`,
    );
  });
  await page.locator('[data-testid="settings-sidebar-tab-developer"]').click();
  const developerTab = page.locator(
    '[data-testid="developer-lab-tab-developer"]',
  );
  await developerTab.waitFor({
    state: "visible",
    timeout: Math.min(45_000, options.timeoutMs),
  });
  if ((await developerTab.getAttribute("data-state")) !== "active") {
    await developerTab.click();
  }
}

function copyJsonButton(page) {
  return page
    .getByRole("button", {
      name: /复制纯 JSON|複製純 JSON|Copy raw JSON|JSON.*コピー|JSON 복사/i,
    })
    .first();
}

async function readDeveloperReadyState(page, options) {
  return await waitForPageCondition(
    page,
    options,
    () => {
      const bodyText = document.body?.innerText ?? "";
      const developerTabActive =
        document
          .querySelector('[data-testid="settings-sidebar-tab-developer"]')
          ?.getAttribute("data-active") === "true";
      const developerLabActive =
        document
          .querySelector('[data-testid="developer-lab-tab-developer"]')
          ?.getAttribute("data-state") === "active";
      const copyButton = Array.from(document.querySelectorAll("button")).find(
        (button) =>
          /复制纯 JSON|複製純 JSON|Copy raw JSON|JSON.*コピー|JSON 복사/i.test(
            button.textContent ?? "",
          ),
      );
      if (
        !developerTabActive ||
        !developerLabActive ||
        !copyButton ||
        copyButton.disabled ||
        (!bodyText.includes("诊断日志") &&
          !bodyText.includes("診斷日誌") &&
          !bodyText.includes("Diagnostic Logs"))
      ) {
        return null;
      }
      return {
        developerTabActive,
        developerLabActive,
        copyJsonActionReady: true,
      };
    },
    "Developer Settings did not reach a ready diagnostic state",
  );
}

async function readDeveloperTerminalState(page, options) {
  return await waitForPageCondition(
    page,
    options,
    () => {
      const bodyText = document.body?.innerText ?? "";
      const sink = window.__LIME_SETTINGS_DEVELOPER_CLIPBOARD_SINK__;
      const diagnosticSuccess =
        bodyText.includes("纯 JSON 诊断信息已复制") ||
        bodyText.includes("純 JSON 診斷資訊已複製") ||
        bodyText.includes("Raw JSON diagnostics copied");
      const loadingVisible = bodyText.includes("正在收集诊断信息");
      const errorVisible =
        bodyText.includes("复制纯 JSON 失败") ||
        bodyText.includes("複製純 JSON 失敗") ||
        bodyText.includes("Failed to copy raw JSON");
      if (!diagnosticSuccess || !sink || sink.writeCount !== 1) {
        return null;
      }
      return {
        diagnosticSuccess,
        loadingVisible,
        errorVisible,
        clipboard: {
          writeCount: sink.writeCount,
          textLength: sink.textLength,
          jsonObject: sink.jsonObject,
          payloadShape: sink.payloadShape,
        },
        traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    "Developer diagnostic collection did not reach copied terminal state",
  );
}

async function run() {
  const options = parseSettingsDeveloperFixtureArgs(process.argv.slice(2), {
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
  const summary = createSettingsDeveloperEvidence({
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
    const clipboardSinkInstalled = await installClipboardSink(page);
    await openDeveloperSettings(page, options);
    const readyState = await readDeveloperReadyState(page, options);
    const button = copyJsonButton(page);
    await button.scrollIntoViewIfNeeded();
    await button.click();
    const state = await readDeveloperTerminalState(page, options);
    const trace = summarizeSettingsDeveloperTrace(state.traceRaw);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    applyPassingSettingsDeveloperEvidence(summary, {
      completedAt: new Date().toISOString(),
      electronRenderer: handle.rendererSnapshot.electron,
      preloadInvoke: handle.rendererSnapshot.hasInvokeBridge,
      isolatedUserData: runtimeEnv.electronUserDataDir.startsWith(
        runtimeEnv.tempRoot,
      ),
      ...readyState,
      diagnosticSuccess: state.diagnosticSuccess,
      clipboardSinkInstalled,
      clipboard: state.clipboard,
      loadingVisible: state.loadingVisible,
      errorVisible: state.errorVisible,
      trace,
      consoleErrors,
      pageErrors,
      invokeErrorCount: parseInvokeTraceRaw(state.errorRaw).length,
      screenshotWritten: fs.existsSync(screenshotPath),
    });
    writeJsonFile(rawEvidencePath, {
      collection: {
        isolatedUserData: true,
        diagnosticSuccess: true,
        clipboardSinkTestOnly: true,
      },
      clipboard: state.clipboard,
      appServerMethods: trace.methods,
      configMethods: trace.methods,
    });
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:settings-developer-fixture] summary=${summaryPath}`);
  } catch (error) {
    applyFailedSettingsDeveloperEvidence(summary, error);
    summary.consoleErrors = consoleErrors.map(sanitizeText);
    summary.pageErrors = pageErrors.map(sanitizeText);
    if (page) {
      const traceRaw = await page
        .evaluate(() =>
          window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        )
        .catch(() => null);
      summary.failureTrace = summarizeSettingsDeveloperTrace(traceRaw);
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
    `[smoke:settings-developer-fixture] ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
