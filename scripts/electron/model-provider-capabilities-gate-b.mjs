#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
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
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  sanitizeText,
} from "./mcp-config-fixture-smoke.mjs";

export const MODEL_PROVIDER_CAPABILITIES_METHOD =
  "modelProvider/capabilities/read";
export const EXPECTED_PROVIDER_CAPABILITIES = [false, true, true];

const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "model-provider-capabilities-electron-gate-b",
  ),
  prefix: "model-provider-capabilities-electron-gate-b",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};
const PREFIX_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LOG_PREFIX = "[smoke:model-provider-capabilities-gate-b]";

function printHelp() {
  console.log(`
Model Provider Capabilities Electron Gate B

用途:
  从真实 Electron 首页打开 ModelSelector，验证当前运行 provider capability
  经 app_server_handle_json_lines -> modelProvider/capabilities/read 投影到 GUI。

边界:
  使用隔离 OpenAI Responses 官方路由配置和 unavailable backend；不调用模型、
  不访问网络、不读取真实凭证，不使用 mock backend 或 renderer fallback。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function parseModelProviderCapabilitiesArgs(argv, defaults = DEFAULTS) {
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

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function seedOfficialResponsesConfig(runtimeEnv) {
  const configPath = path.join(runtimeEnv.electronUserDataDir, "config.yaml");
  fs.writeFileSync(
    configPath,
    [
      'default_provider: "openai-response"',
      "providers:",
      "  openai:",
      "    enabled: true",
      '    base_url: "https://api.openai.com/v1"',
      "",
    ].join("\n"),
    "utf8",
  );
  runtimeEnv.env.LIME_CONFIG_PATH = configPath;
}

export function summarizeModelProviderCapabilitiesEvidence({
  traceRaw,
  errorRaw,
  dom,
}) {
  const requests = parseJsonRpcRequestsFromInvokeTrace(traceRaw);
  const targetRequests = requests.filter(
    (request) => request.method === MODEL_PROVIDER_CAPABILITIES_METHOD,
  );
  const electronTargetRequests = targetRequests.filter(
    (request) =>
      request.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
      request.transport === "electron-ipc" &&
      request.status === "success",
  );
  const commands = Array.from(
    new Set(
      parseInvokeTraceRaw(traceRaw)
        .map((entry) => entry?.command)
        .filter(Boolean),
    ),
  );
  return {
    bridge: {
      command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      appServerHandleJsonLinesSeen: commands.includes(
        APP_SERVER_HANDLE_JSON_LINES_COMMAND,
      ),
      method: MODEL_PROVIDER_CAPABILITIES_METHOD,
      electronIpcHitCount: electronTargetRequests.length,
      mockFallbackHitCount:
        targetRequests.length - electronTargetRequests.length,
    },
    gui: {
      selectorVisible: dom?.selectorVisible === true,
      panelVisible: dom?.panelVisible === true,
      badgeLabels: Array.isArray(dom?.badgeLabels) ? dom.badgeLabels : [],
      activeStates: Array.isArray(dom?.activeStates) ? dom.activeStates : [],
      loadingVisible: dom?.loadingVisible === true,
    },
    errors: {
      invokeErrorCount: parseInvokeTraceRaw(errorRaw).length,
    },
    requests: electronTargetRequests.map((request) => ({
      method: request.method,
      transport: request.transport,
      status: request.status,
    })),
  };
}

export function assertModelProviderCapabilitiesEvidence(evidence) {
  assert(
    evidence.bridge.appServerHandleJsonLinesSeen,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    evidence.bridge.electronIpcHitCount > 0,
    `未观察到 ${MODEL_PROVIDER_CAPABILITIES_METHOD} electron-ipc success`,
  );
  assert(
    evidence.bridge.mockFallbackHitCount === 0,
    "provider capability 命中了非 electron-ipc transport",
  );
  assert(evidence.gui.selectorVisible, "ModelSelector 不可见");
  assert(evidence.gui.panelVisible, "provider capability panel 不可见");
  assert(!evidence.gui.loadingVisible, "provider capability 仍处于 loading");
  assert(
    JSON.stringify(evidence.gui.activeStates) ===
      JSON.stringify(EXPECTED_PROVIDER_CAPABILITIES),
    `provider capability GUI 状态不正确: ${JSON.stringify(
      evidence.gui.activeStates,
    )}`,
  );
  assert(
    evidence.gui.badgeLabels.length === EXPECTED_PROVIDER_CAPABILITIES.length,
    `provider capability badge 数量不正确: ${evidence.gui.badgeLabels.length}`,
  );
  assert(
    evidence.errors.invokeErrorCount === 0,
    `观察到 invoke error: ${evidence.errors.invokeErrorCount}`,
  );
}

async function readGuiAndTrace(page) {
  return await page.evaluate(() => {
    const selector = document.querySelector('[data-testid="model-selector"]');
    const panel = document.querySelector(
      '[data-testid="model-selector-provider-capability-panel"]',
    );
    const badges = Array.from(
      panel?.querySelectorAll(
        '[data-testid="model-provider-capabilities"] > span',
      ) ?? [],
    );
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
      dom: {
        selectorVisible: isVisible(selector),
        panelVisible: isVisible(panel),
        badgeLabels: badges.map((badge) => badge.textContent?.trim() || ""),
        activeStates: badges.map((badge) =>
          badge.classList.contains("text-emerald-700"),
        ),
        loadingVisible: Boolean(panel?.textContent?.includes("正在读取能力")),
      },
      traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    };
  });
}

async function run() {
  const options = parseModelProviderCapabilitiesArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
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
  let handle = null;
  const consoleErrors = [];
  const pageErrors = [];
  const summary = {
    schemaVersion: 1,
    scenarioId: "CODEX-ALIGN-model-provider-capabilities",
    proofLevel: "Gate B controlled fixture",
    claimBoundary:
      "Real Electron ModelSelector projection for the configured current provider. It does not call a model or prove live provider readiness.",
    backendMode: "unavailable",
    providerRoute: "official-openai-responses-fixture",
    ok: false,
    checkedAt: new Date().toISOString(),
    bridge: null,
    gui: null,
    errors: null,
    screenshot: null,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
  };

  try {
    ensureElectronFixtureBuild({
      logPrefix: LOG_PREFIX,
      rootDir: process.cwd(),
    });
    seedOfficialResponsesConfig(runtimeEnv);
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
    });
    await handle.page.evaluate(() => {
      window.localStorage.removeItem("lime_invoke_error_buffer_v1");
      window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
    });

    console.log(`${LOG_PREFIX} stage=open-model-selector`);
    const selector = handle.page
      .locator('[data-testid="model-selector"]')
      .first();
    await selector.waitFor({ state: "visible", timeout: options.timeoutMs });
    await selector.click();
    await handle.page
      .locator('[data-testid="model-selector-provider-capability-panel"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    await handle.page
      .locator('[data-testid="model-provider-capabilities"]')
      .waitFor({ state: "visible", timeout: options.timeoutMs });

    const observed = await readGuiAndTrace(handle.page);
    const evidence = summarizeModelProviderCapabilitiesEvidence(observed);
    assertModelProviderCapabilitiesEvidence(evidence);
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
    summary.bridge = evidence.bridge;
    summary.gui = evidence.gui;
    summary.errors = {
      ...evidence.errors,
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      mockFallbackHitCount: evidence.bridge.mockFallbackHitCount,
    };
    summary.requests = evidence.requests;
    summary.screenshot = `${options.prefix}.png`;
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
  } catch (error) {
    summary.failure = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    summary.errors = {
      ...(summary.errors ?? {}),
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    if (handle?.page) {
      await handle.page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
    }
    writeJsonFile(summaryPath, summary);
    throw error;
  } finally {
    await closeElectronFixture(handle);
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  run().catch((error) => {
    console.error(
      `${LOG_PREFIX} failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    process.exitCode = 1;
  });
}
