#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import electronPath from "electron";
import { _electron as electron } from "playwright";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  sanitizeJson,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  CONTEXT7_CONFIG_URL,
  CONTEXT7_ENV_VAR_NAME,
  CONTEXT7_HEADER_NAME,
  CONTEXT7_PRESET_NAME,
  MCP_CREATE_LIST_REQUIRED_METHODS,
  applyFailedMcpSettingsScenarioEvidence,
  applyPassingMcpSettingsScenarioEvidence,
  assertContext7Server,
  assertMcpElectronEvidence,
  createMcpSettingsScenarioEvidence,
  parseMcpConfigFixtureArgs,
  parseInvokeTraceRaw,
  summarizeContext7Server,
  summarizeMcpElectronEvidence,
} from "./lib/mcp-config-fixture-evidence.mjs";

export {
  assertContext7Server,
  getServerConfig,
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
  summarizeContext7Server,
} from "./lib/mcp-config-fixture-evidence.mjs";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: null,
  prefix: "mcp-config-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

const LOG_PREFIX = "[smoke:mcp-config-fixture]";

function printHelp() {
  console.log(`
MCP Config Electron Fixture Smoke

用途:
  启动真实 Electron Desktop Host，在设置页 MCP 配置管理中创建 Context7 配置，
  并通过 preload app_server_handle_json_lines -> mcpServer/list 验证落库。

边界:
  APP_SERVER_BACKEND_MODE=unavailable；不启动 Context7、不调用真实 provider、
  不读取或写入真实 key，不使用 mock backend / renderer fallback / 旧 MCP facade。

用法:
  node scripts/electron/mcp-config-fixture-smoke.mjs --run-id <project-gate-run-id>

选项:
  --run-id <id> --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

export function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

export function sanitizeText(value) {
  const sanitized = String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|session|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]")
    .replace(/sk-[A-Za-z0-9._-]+/g, "sk-[redacted]");
  return sanitized.length > 2_000
    ? `${sanitized.slice(0, 2_000)}... [truncated ${
        sanitized.length - 2_000
      } chars]`
    : sanitized;
}

export function createTempRuntimeEnv() {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "mcp-config-"));
  const home = path.join(tempRoot, "home");
  const xdgDataHome = path.join(tempRoot, "xdg-data");
  const localAppData = path.join(tempRoot, "local-app-data");
  const roamingAppData = path.join(tempRoot, "roaming-app-data");
  const electronUserDataDir = path.join(tempRoot, "electron-user-data");
  const appServerDataDir = path.join(electronUserDataDir, "app-server");

  for (const dir of [
    home,
    xdgDataHome,
    localAppData,
    roamingAppData,
    electronUserDataDir,
    appServerDataDir,
  ]) {
    fs.mkdirSync(dir, { recursive: true });
  }

  return {
    tempRoot,
    home,
    electronUserDataDir,
    appServerDataDir,
    env: {
      ...process.env,
      HOME: home,
      XDG_DATA_HOME: xdgDataHome,
      APPDATA: roamingAppData,
      LOCALAPPDATA: localAppData,
    },
  };
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
      supportsAppServer:
        typeof window.electronAPI?.supportsCommand === "function" &&
        window.electronAPI.supportsCommand("app_server_handle_json_lines"),
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
      snapshot.supportsAppServer &&
      !snapshot.startupVisible &&
      snapshot.appSidebarVisible
    ) {
      return snapshot;
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

export async function launchElectronFixture({
  options,
  runtimeEnv,
  appServerEnv,
  consoleErrors,
  pageErrors = [],
  backendMode = "unavailable",
}) {
  const packagedExecutable = options.electronExecutable || null;
  const app = await electron.launch({
    executablePath: packagedExecutable || electronPath,
    args: packagedExecutable
      ? ["--use-mock-keychain"]
      : ["--use-mock-keychain", "."],
    cwd: process.cwd(),
    env: {
      ...runtimeEnv.env,
      ...appServerEnv,
      APP_SERVER_BACKEND_MODE: backendMode,
      ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
      LIME_ELECTRON_E2E: "1",
      LIME_ELECTRON_BRAND_DEV_APP: "0",
      LIME_ELECTRON_CLEAR_RENDERER_CACHE: "0",
      LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
    },
    timeout: options.timeoutMs,
  });

  try {
    app.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(sanitizeText(message.text()));
      }
    });

    const page = await app.firstWindow({ timeout: options.timeoutMs });
    page.on("pageerror", (error) => {
      pageErrors.push(sanitizeText(error.message));
    });
    page.setDefaultTimeout(options.timeoutMs);
    await page.setViewportSize({ width: 1440, height: 1000 });
    const rendererSnapshot = await waitForRendererReady(page, options);
    await clearInvokeBuffers(page);

    return { app, page, rendererSnapshot };
  } catch (error) {
    await app.close().catch(() => undefined);
    throw error;
  }
}

export async function closeElectronFixture(handle) {
  if (handle?.app) {
    await handle.app.close().catch(() => undefined);
    await sleep(500);
  }
}

export async function waitForPageCondition(
  page,
  options,
  predicate,
  message,
  arg = {},
) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(45_000, options.timeoutMs)) {
    const result = await evaluatePageSnapshot(page, predicate, arg);
    if (result) {
      return result;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(message);
}

export async function openMcpConfigSettings(page, options) {
  await openSettings(page, options);
  await page.locator('[data-testid="settings-sidebar-tab-mcp-server"]').click();
  await page.locator('[data-testid="mcp-panel-tab-config"]').waitFor({
    state: "visible",
    timeout: Math.min(45_000, options.timeoutMs),
  });
  await page.locator('[data-testid="mcp-panel-tab-config"]').click();
  await page.locator('[data-testid="mcp-config-page"]').waitFor({
    state: "visible",
    timeout: Math.min(45_000, options.timeoutMs),
  });
}

export async function openSettings(page, options) {
  await page.locator('[data-testid="app-sidebar-account-button"]').click();
  await page.locator('[data-testid="app-sidebar-account-menu"]').waitFor({
    state: "visible",
    timeout: Math.min(30_000, options.timeoutMs),
  });

  const clicked = await page.evaluate(() => {
    const menu = document.querySelector(
      '[data-testid="app-sidebar-account-menu"]',
    );
    const buttons = Array.from(menu?.querySelectorAll("button") ?? []);
    const target = buttons.find((button) => {
      const text = button.textContent || "";
      const aria = button.getAttribute("aria-label") || "";
      return /模型设置|AI 服务商|AI Providers|Model Settings/.test(
        `${text}\n${aria}`,
      );
    });
    if (!(target instanceof HTMLButtonElement)) {
      return false;
    }
    target.click();
    return true;
  });
  assert(clicked, "未找到账号菜单里的模型设置入口");

  await page.locator('[data-testid="settings-top-header"]').waitFor({
    state: "visible",
    timeout: Math.min(45_000, options.timeoutMs),
  });
}

export async function createContext7ConfigFromGui(
  page,
  { configUrl = CONTEXT7_CONFIG_URL, envVarName = CONTEXT7_ENV_VAR_NAME } = {},
) {
  await page.locator('[data-testid="mcp-config-create-server"]').click();
  await page.locator('[data-testid="mcp-config-preset-context7"]').click();
  await page
    .locator('[data-testid="mcp-config-connection-url"]')
    .fill(configUrl);
  await page
    .locator('[data-testid="mcp-config-env-header-env-var"]')
    .first()
    .fill(envVarName);

  const formSnapshot = await page.evaluate(() => ({
    name: document.querySelector('[data-testid="mcp-config-name"]')?.value,
    url: document.querySelector('[data-testid="mcp-config-connection-url"]')
      ?.value,
    envVars: Array.from(
      document.querySelectorAll(
        '[data-testid="mcp-config-env-header-env-var"]',
      ),
    ).map((input) => input.value),
    json: document.querySelector('[data-testid="mcp-config-json"]')?.value,
    text: document.querySelector('[data-testid="mcp-config-page"]')
      ?.textContent,
  }));

  assert(
    formSnapshot.name === CONTEXT7_PRESET_NAME,
    `Context7 preset 名称不正确: ${formSnapshot.name}`,
  );
  assert(
    formSnapshot.url === configUrl,
    `Context7 URL 未写入结构化表单: ${formSnapshot.url}`,
  );
  assert(
    formSnapshot.envVars.includes(envVarName),
    "Context7 env_http_headers 环境变量名未写入结构化表单",
  );
  assert(
    String(formSnapshot.json || "").includes(configUrl) &&
      String(formSnapshot.json || "").includes(envVarName),
    "Context7 JSON 配置未随结构化表单同步",
  );

  await page.locator('[data-testid="mcp-config-save"]').click();
  return formSnapshot;
}

export async function appServerCallFromPage(page, method, params = {}) {
  return await page.evaluate(
    async ({ command, method, params }) => {
      const invoke = window.electronAPI?.invoke;
      if (typeof invoke !== "function") {
        throw new Error("Electron preload invoke bridge is unavailable");
      }
      const id = `mcp-config-fixture-${Date.now()}-${Math.random()
        .toString(16)
        .slice(2)}`;
      const response = await invoke(command, {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id,
              method,
              params,
            }),
          ],
        },
      });
      const decoded = Array.isArray(response?.lines)
        ? response.lines
            .map((line) => {
              try {
                return JSON.parse(line);
              } catch {
                return null;
              }
            })
            .filter(Boolean)
        : [];
      const error = decoded.find(
        (message) => message?.id === id && message.error,
      );
      if (error) {
        throw new Error(`${method} failed: ${JSON.stringify(error.error)}`);
      }
      const result = decoded.find(
        (message) =>
          message?.id === id &&
          Object.prototype.hasOwnProperty.call(message, "result"),
      );
      if (!result) {
        throw new Error(`${method} did not return a JSON-RPC result`);
      }
      return {
        appServerCommand: command,
        method,
        result: result.result,
        decoded,
        traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    },
    { command: APP_SERVER_HANDLE_JSON_LINES_COMMAND, method, params },
  );
}

function valuesEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

export async function readConfigFromPage(page) {
  const response = await appServerCallFromPage(page, "config/read", {
    includeLayers: true,
  });
  const config = response.result?.config;
  const layers = response.result?.layers;
  const layer = Array.isArray(layers) && layers.length === 1 ? layers[0] : null;
  const version = layer?.version;
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new Error("config/read 未返回有效配置");
  }
  if (typeof version !== "string" || !version) {
    throw new Error("config/read 未返回有效配置版本");
  }
  return {
    config,
    version,
    filePath:
      layer?.name?.type === "user" && typeof layer.name.file === "string"
        ? layer.name.file
        : null,
    response,
  };
}

export async function writeConfigEditsFromPage(
  page,
  { edits, expectedVersion },
) {
  if (!Array.isArray(edits) || edits.length === 0) return null;
  const response = await appServerCallFromPage(page, "config/batchWrite", {
    edits,
    expectedVersion,
    reloadUserConfig: true,
  });
  if (response.result?.status !== "ok") {
    throw new Error("config/batchWrite 未返回有效写入结果");
  }
  return response;
}

export async function updateConfigFromPage(page, updater) {
  const snapshot = await readConfigFromPage(page);
  const nextConfig = await updater(snapshot.config);
  const edits = Object.entries(nextConfig).flatMap(([keyPath, value]) =>
    valuesEqual(snapshot.config[keyPath], value)
      ? []
      : [{ keyPath, value, mergeStrategy: "replace" }],
  );
  const write = await writeConfigEditsFromPage(page, {
    edits,
    expectedVersion: snapshot.version,
  });
  return { ...snapshot, config: nextConfig, write };
}

export async function waitForContext7Server(page, options) {
  const startedAt = Date.now();
  let lastResult = null;
  while (Date.now() - startedAt < Math.min(45_000, options.timeoutMs)) {
    lastResult = await appServerCallFromPage(page, "mcpServer/list", {});
    const server = (lastResult.result?.servers ?? []).find(
      (item) => item?.name === CONTEXT7_PRESET_NAME,
    );
    if (server) {
      return { listResult: lastResult, server };
    }
    await sleep(options.intervalMs);
  }
  throw new Error("mcpServer/list 未读回 GUI 保存的 Context7 配置");
}

export async function run() {
  const options = parseMcpConfigFixtureArgs(process.argv.slice(2), {
    defaults: DEFAULTS,
  });
  if (options.help) {
    printHelp();
    return;
  }
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

  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: {
      ...runtimeEnv.env,
      APP_SERVER_BIN: appServerBinary,
    },
  });

  const summary = {
    ...createMcpSettingsScenarioEvidence({
      candidateRunId: options.runId,
      startedAt: new Date().toISOString(),
      prefix: options.prefix,
    }),
    ok: false,
    checkedAt: new Date().toISOString(),
    backendMode: "unavailable",
    electronPreloadBridge: false,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    electronUserDataDir: options.keepTemp
      ? runtimeEnv.electronUserDataDir
      : null,
    appServerDataDir: options.keepTemp
      ? runtimeEnv.appServerDataDir
      : path.join("electron-user-data", "app-server"),
    appServerBinary: options.keepTemp
      ? appServerBinary
      : path.basename(appServerBinary),
    context7Preset: {
      serverName: CONTEXT7_PRESET_NAME,
      urlHost: new URL(CONTEXT7_CONFIG_URL).host,
      envHttpHeaderNames: [CONTEXT7_HEADER_NAME],
      envHttpHeaderEnvVars: [CONTEXT7_ENV_VAR_NAME],
    },
    guiCreatedContext7: false,
    context7Server: null,
    appServerHandleJsonLinesSeen: false,
    electronRequestMethods: [],
    missingRequiredMethods: [...MCP_CREATE_LIST_REQUIRED_METHODS],
    legacyMcpCommandsSeen: [],
    consoleErrors: [],
    screenshot: null,
    rawEvidence: `${options.prefix}-raw.json`,
    summary: `${options.prefix}-summary.json`,
  };

  let app = null;
  let page = null;
  const consoleErrors = [];
  const pageErrors = [];
  const rawEvidence = {};

  try {
    logStage("launch-electron");
    const handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
    });
    app = handle.app;
    page = handle.page;
    summary.electronPreloadBridge =
      handle.rendererSnapshot.electron &&
      handle.rendererSnapshot.hasInvokeBridge;

    logStage("open-mcp-config-settings");
    await openMcpConfigSettings(page, options);

    logStage("create-context7-config-from-gui");
    rawEvidence.formSnapshot = sanitizeJson(
      await createContext7ConfigFromGui(page),
    );
    const { listResult, server } = await waitForContext7Server(page, options);
    assertContext7Server(server);
    summary.guiCreatedContext7 = true;
    summary.context7Server = summarizeContext7Server(server);
    rawEvidence.mcpServerList = sanitizeJson(listResult);

    const evidence = summarizeMcpElectronEvidence({
      listResult,
      traceRaw: listResult.traceRaw,
    });
    assertMcpElectronEvidence(evidence);
    summary.appServerHandleJsonLinesSeen =
      evidence.appServerHandleJsonLinesSeen;
    summary.electronIpcSeen = evidence.electronIpcSeen;
    summary.electronIpcHitCount = evidence.electronIpcHitCount;
    summary.electronRequestMethods = evidence.requestMethods;
    summary.electronIpcRequestMethods = evidence.electronIpcRequestMethods;
    summary.missingRequiredMethods = evidence.missingRequiredMethods;
    summary.legacyMcpCommandsSeen = evidence.legacyMcpCommandsSeen;
    summary.mockFallbackHitCount = evidence.mockFallbackHitCount;
    rawEvidence.electronRequests = sanitizeJson(evidence.requests);

    await page.screenshot({ path: screenshotPath, fullPage: true });
    await closeElectronFixture(handle);
    app = null;
    page = null;

    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );

    summary.consoleErrors = consoleErrors;
    summary.pageErrors = pageErrors;
    summary.screenshot = `${options.prefix}.png`;
    applyPassingMcpSettingsScenarioEvidence(summary, {
      completedAt: new Date().toISOString(),
      electronRenderer: handle.rendererSnapshot.electron,
      preloadInvoke: summary.electronPreloadBridge,
      electronEvidence: evidence,
      guiCreatedContext7: summary.guiCreatedContext7,
      context7Server: summary.context7Server,
      consoleErrors,
      pageErrors,
      invokeErrorCount: parseInvokeTraceRaw(listResult.errorRaw).length,
      screenshotWritten: fs.existsSync(screenshotPath),
    });
    summary.ok = true;
    writeJsonFile(rawEvidencePath, rawEvidence);
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
    console.log(`${LOG_PREFIX} context7=${summary.context7Server?.id ?? ""}`);
  } catch (error) {
    applyFailedMcpSettingsScenarioEvidence(summary, error);
    summary.consoleErrors = consoleErrors;
    summary.pageErrors = pageErrors;
    if (Object.keys(rawEvidence).length > 0) {
      writeJsonFile(rawEvidencePath, rawEvidence);
    }
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
        // 截图失败不覆盖原始错误。
      }
    }
    throw error;
  } finally {
    if (app) {
      await closeElectronFixture({ app });
    }
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
