#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { chromium } from "playwright";

const DEFAULTS = {
  appUrl: "http://127.0.0.1:1420/",
  healthUrl: "http://127.0.0.1:3030/health",
  invokeUrl: "http://127.0.0.1:3030/invoke",
  timeoutMs: 120_000,
  intervalMs: 1_000,
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "model-registry-current",
  ),
  prefix: "model-registry-current",
  headless: true,
};

const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const REQUIRED_APP_SERVER_METHODS = [
  "model/list",
  "modelPreferences/list",
  "modelSyncState/read",
];
const LEGACY_MODEL_COMMANDS = [
  "get_model_registry",
  "get_model_preferences",
  "get_model_sync_state",
  "get_models_for_provider",
  "get_models_by_tier",
  "get_provider_alias_config",
  "get_all_alias_configs",
];
const FORBIDDEN_MODEL_SIDE_EFFECT_COMMANDS = [
  "refresh_model_registry",
  "search_models",
  "toggle_model_favorite",
  "hide_model",
  "record_model_usage",
  "fetch_provider_models_auto",
];
const FORBIDDEN_APP_SERVER_METHODS = ["turn/start"];

function printHelp() {
  console.log(`
Model Registry Current Smoke

用途:
  通过真实设置页“添加模型”面板和 DevBridge /invoke 网络请求，验证模型
  registry / preference / sync 只读链走 App Server JSON-RPC current 主链，
  而不是 legacy get_model_* 命令或 mock。脚本不刷新 registry、不收藏、
  不隐藏模型、不 live fetch Provider 模型，也不发送真实 Agent turn。

用法:
  node scripts/model-registry-current-smoke.mjs

选项:
  --app-url <url>        前端地址，默认 http://127.0.0.1:1420/
  --health-url <url>     DevBridge 健康检查地址，默认 http://127.0.0.1:3030/health
  --invoke-url <url>     DevBridge invoke 地址，默认 http://127.0.0.1:3030/invoke
  --timeout-ms <ms>      总超时，默认 120000
  --interval-ms <ms>     健康检查轮询间隔，默认 1000
  --evidence-dir <path>  证据目录，默认 .lime/qc/gui-evidence/model-registry-current
  --prefix <name>        证据文件前缀，默认 model-registry-current
  --headed               使用有界面 Chrome
  -h, --help             显示帮助
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--app-url" && argv[index + 1]) {
      options.appUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--health-url" && argv[index + 1]) {
      options.healthUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--invoke-url" && argv[index + 1]) {
      options.invokeUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && argv[index + 1]) {
      options.timeoutMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && argv[index + 1]) {
      options.intervalMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--evidence-dir" && argv[index + 1]) {
      options.evidenceDir = path.resolve(String(argv[index + 1]).trim());
      index += 1;
      continue;
    }
    if (arg === "--prefix" && argv[index + 1]) {
      options.prefix = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--headed") {
      options.headless = false;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!options.appUrl) {
    throw new Error("--app-url 不能为空");
  }
  if (!options.healthUrl) {
    throw new Error("--health-url 不能为空");
  }
  if (!options.invokeUrl) {
    throw new Error("--invoke-url 不能为空");
  }
  if (!options.evidenceDir) {
    throw new Error("--evidence-dir 不能为空");
  }
  if (!options.prefix) {
    throw new Error("--prefix 不能为空");
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
  console.log(`[smoke:model-registry-current] stage=${stage}`);
}

async function waitForHealth(options) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const response = await fetch(options.healthUrl, {
        signal: AbortSignal.timeout(Math.min(5_000, options.timeoutMs)),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      console.log(
        `[smoke:model-registry-current] DevBridge 已就绪 (${Date.now() - startedAt}ms) status=${payload?.status ?? response.status}`,
      );
      return payload;
    } catch (error) {
      lastError = error;
      await sleep(options.intervalMs);
    }
  }

  const detail =
    lastError instanceof Error
      ? lastError.message
      : String(lastError || "unknown error");
  throw new Error(
    `[smoke:model-registry-current] DevBridge 未就绪，请先启动 npm run electron:dev。最后错误: ${detail}`,
  );
}

function normalizeUrl(value) {
  try {
    const parsed = new URL(value);
    parsed.hash = "";
    return parsed.toString();
  } catch {
    return value;
  }
}

function requestMatchesInvoke(url, invokeUrl) {
  const requestUrl = normalizeUrl(url);
  const expectedUrl = normalizeUrl(invokeUrl);
  return requestUrl === expectedUrl || requestUrl.endsWith("/invoke");
}

function parseJson(value) {
  if (!value) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function parseJsonRpcLine(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function decodeJsonRpcLines(lines) {
  return Array.isArray(lines)
    ? lines.map(parseJsonRpcLine).filter(Boolean)
    : [];
}

function sanitizeText(value) {
  const sanitized = String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|session|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]");
  return sanitized.length > 1_200
    ? `${sanitized.slice(0, 1_200)}... [truncated ${sanitized.length - 1_200} chars]`
    : sanitized;
}

function sanitizeJson(value, depth = 0) {
  if (depth > 5) {
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
    return value.slice(0, 40).map((item) => sanitizeJson(item, depth + 1));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 80)
        .map(([key, item]) => [key, sanitizeJson(item, depth + 1)]),
    );
  }
  return sanitizeText(String(value));
}

async function openModelAddPanel(page, options) {
  await page.goto(options.appUrl, {
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  await page.waitForSelector('[data-testid="app-sidebar-account-button"]', {
    timeout: options.timeoutMs,
  });

  await page.getByTestId("app-sidebar-account-button").click();
  await page.waitForSelector('[data-testid="app-sidebar-account-menu"]', {
    timeout: options.timeoutMs,
  });

  const clicked = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll("button"));
    const target = buttons.find((button) => {
      const text = button.textContent || "";
      const aria = button.getAttribute("aria-label") || "";
      return /模型设置|AI 服务商|AI Providers|Model Settings/.test(
        `${text}\n${aria}`,
      );
    });
    if (!target) {
      return false;
    }
    target.click();
    return true;
  });

  assert(clicked, "未找到账号菜单里的模型设置入口");
  await page.waitForSelector('[data-testid="settings-top-header"]', {
    timeout: options.timeoutMs,
  });
  await page.waitForSelector('[data-testid="api-key-provider-section"]', {
    timeout: options.timeoutMs,
  });
  await page.waitForSelector('[data-testid="api-key-provider-detail"]', {
    timeout: options.timeoutMs,
  });

  const addModelClicked = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll("button"));
    const target = buttons.find((button) => {
      const text = button.textContent || "";
      const aria = button.getAttribute("aria-label") || "";
      return /添加模型|Add model/i.test(`${text}\n${aria}`);
    });
    if (!target) {
      return false;
    }
    target.click();
    return true;
  });

  assert(addModelClicked, "未找到添加模型入口");
  await page.waitForSelector('[data-testid="model-add-catalog"]', {
    timeout: options.timeoutMs,
  });
}

function collectInvokeEntry(requestPayload, responsePayload, requestUrl) {
  const requestMessages = decodeJsonRpcLines(
    requestPayload?.args?.request?.lines,
  );
  const responseMessages = decodeJsonRpcLines(responsePayload?.result?.lines);
  return {
    url: requestUrl,
    cmd: requestPayload?.cmd ?? null,
    appServerRequests: requestMessages
      .filter((message) => typeof message?.method === "string")
      .map((message) => ({
        id: message.id ?? null,
        method: message.method,
        params: sanitizeJson(message.params ?? {}),
      })),
    responseMessageCount: responseMessages.length,
    responseMessages: responseMessages.map(sanitizeJson),
  };
}

function findResponseForRequest(entry, request) {
  return entry.responseMessages.find(
    (message) => request.id !== null && message?.id === request.id,
  );
}

function summarizeInvokeEntries(entries) {
  const appServerRequests = entries.flatMap((entry) =>
    entry.appServerRequests.map((request) => ({ entry, request })),
  );
  const appServerMethodsSeen = Array.from(
    new Set(appServerRequests.map(({ request }) => request.method)),
  ).sort();
  const modelListRequests = appServerRequests.filter(
    ({ request }) => request.method === "model/list",
  );
  const modelPreferencesListRequests = appServerRequests.filter(
    ({ request }) => request.method === "modelPreferences/list",
  );
  const modelSyncStateReadRequests = appServerRequests.filter(
    ({ request }) => request.method === "modelSyncState/read",
  );
  const modelListResponses = modelListRequests.map(({ entry, request }) =>
    findResponseForRequest(entry, request),
  );
  const modelPreferencesListResponses = modelPreferencesListRequests.map(
    ({ entry, request }) => findResponseForRequest(entry, request),
  );
  const modelSyncStateReadResponses = modelSyncStateReadRequests.map(
    ({ entry, request }) => findResponseForRequest(entry, request),
  );
  const legacyModelCommandsSeen = Array.from(
    new Set(
      entries
        .map((entry) =>
          typeof entry.cmd === "string" &&
          LEGACY_MODEL_COMMANDS.includes(entry.cmd)
            ? entry.cmd
            : null,
        )
        .filter(Boolean),
    ),
  ).sort();
  const forbiddenModelSideEffectCommandsSeen = Array.from(
    new Set(
      entries
        .map((entry) =>
          typeof entry.cmd === "string" &&
          FORBIDDEN_MODEL_SIDE_EFFECT_COMMANDS.includes(entry.cmd)
            ? entry.cmd
            : null,
        )
        .filter(Boolean),
    ),
  ).sort();
  const forbiddenAppServerMethodsSeen = appServerMethodsSeen.filter((method) =>
    FORBIDDEN_APP_SERVER_METHODS.includes(method),
  );
  const appServerHandleJsonLinesSeen = entries.some(
    (entry) => entry.cmd === APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  );

  return {
    appServerHandleJsonLinesSeen,
    appServerMethodsSeen,
    legacyModelCommandsSeen,
    forbiddenModelSideEffectCommandsSeen,
    forbiddenAppServerMethodsSeen,
    modelListRequestCount: modelListRequests.length,
    modelPreferencesListRequestCount: modelPreferencesListRequests.length,
    modelSyncStateReadRequestCount: modelSyncStateReadRequests.length,
    modelListResponsesValid:
      modelListResponses.length > 0 &&
      modelListResponses.every(
        (message) =>
          Array.isArray(message?.result?.data) &&
          (message.result.nextCursor === null ||
            typeof message.result.nextCursor === "string"),
      ),
    modelPreferencesListResponsesValid:
      modelPreferencesListResponses.length > 0 &&
      modelPreferencesListResponses.every((message) =>
        Array.isArray(message?.result?.preferences),
      ),
    modelSyncStateReadResponsesValid:
      modelSyncStateReadResponses.length > 0 &&
      modelSyncStateReadResponses.every(
        (message) =>
          message?.result?.syncState &&
          typeof message.result.syncState === "object",
      ),
    modelListRequests: modelListRequests.map(({ request }) => request),
    missingRequiredAppServerMethods: REQUIRED_APP_SERVER_METHODS.filter(
      (method) => !appServerMethodsSeen.includes(method),
    ),
  };
}

async function waitForInvokeEvidence(entries, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(30_000, options.timeoutMs)) {
    const observed = summarizeInvokeEntries(entries);
    if (
      observed.appServerHandleJsonLinesSeen &&
      observed.missingRequiredAppServerMethods.length === 0 &&
      observed.modelListResponsesValid &&
      observed.modelPreferencesListResponsesValid &&
      observed.modelSyncStateReadResponsesValid &&
      observed.legacyModelCommandsSeen.length === 0 &&
      observed.forbiddenModelSideEffectCommandsSeen.length === 0 &&
      observed.forbiddenAppServerMethodsSeen.length === 0
    ) {
      return observed;
    }
    await sleep(250);
  }
  return summarizeInvokeEntries(entries);
}

function resolveLocalChromeExecutablePath() {
  const candidates = [
    process.env.PLAYWRIGHT_CHROME_EXECUTABLE,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  ].filter(Boolean);

  return candidates.find((candidate) => fs.existsSync(candidate)) ?? undefined;
}

function writeJsonFile(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const networkPath = path.join(
    options.evidenceDir,
    `${options.prefix}-network-invoke.json`,
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );

  let browser = null;
  let tmpProfileDir = null;
  const consoleErrors = [];
  const failedRequests = [];
  const invokeEntries = [];
  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    appUrl: options.appUrl,
    healthUrl: options.healthUrl,
    invokeUrl: options.invokeUrl,
    health: null,
    providerPageMounted: false,
    apiKeyProviderDetailMounted: false,
    modelAddCatalogMounted: false,
    appServerHandleJsonLinesSeen: false,
    appServerMethodsSeen: [],
    legacyModelCommandsSeen: [],
    forbiddenModelSideEffectCommandsSeen: [],
    forbiddenAppServerMethodsSeen: [],
    modelListRequestCount: 0,
    modelPreferencesListRequestCount: 0,
    modelSyncStateReadRequestCount: 0,
    modelListResponsesValid: false,
    modelPreferencesListResponsesValid: false,
    modelSyncStateReadResponsesValid: false,
    missingRequiredAppServerMethods: [...REQUIRED_APP_SERVER_METHODS],
    consoleErrors: [],
    failedRequests: [],
    screenshot: null,
    network: networkPath,
    summary: summaryPath,
  };

  try {
    logStage("wait-health");
    summary.health = await waitForHealth(options);
    assert(
      summary.health?.transport === "electron-host",
      `DevBridge transport 不是 electron-host: ${summary.health?.transport ?? "unknown"}`,
    );

    logStage("launch-browser");
    tmpProfileDir = fs.mkdtempSync(
      path.join(os.tmpdir(), "model-registry-current-"),
    );
    browser = await chromium.launchPersistentContext(tmpProfileDir, {
      headless: options.headless,
      viewport: { width: 1440, height: 1000 },
      executablePath: resolveLocalChromeExecutablePath(),
    });
    const page = await browser.newPage();

    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(sanitizeText(message.text()));
      }
    });
    page.on("requestfailed", (request) => {
      failedRequests.push({
        url: sanitizeText(request.url()),
        method: request.method(),
        failure: sanitizeText(request.failure()?.errorText || "unknown"),
      });
    });
    page.on("response", async (response) => {
      try {
        const request = response.request();
        if (
          request.method() !== "POST" ||
          !requestMatchesInvoke(request.url(), options.invokeUrl)
        ) {
          return;
        }
        const requestPayload = parseJson(request.postData() || "");
        const responsePayload = parseJson(await response.text());
        invokeEntries.push(
          collectInvokeEntry(requestPayload, responsePayload, request.url()),
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (message.includes("No resource with given identifier found")) {
          return;
        }
        console.warn(
          "[smoke:model-registry-current] 解析 invoke 网络证据失败:",
          message,
        );
      }
    });

    logStage("open-model-add-panel");
    await openModelAddPanel(page, options);
    summary.providerPageMounted =
      (await page.locator('[data-testid="api-key-provider-section"]').count()) >
      0;
    summary.apiKeyProviderDetailMounted =
      (await page.locator('[data-testid="api-key-provider-detail"]').count()) >
      0;
    summary.modelAddCatalogMounted =
      (await page.locator('[data-testid="model-add-catalog"]').count()) > 0;

    const observed = await waitForInvokeEvidence(invokeEntries, options);
    Object.assign(summary, observed);
    summary.consoleErrors = consoleErrors;
    summary.failedRequests = failedRequests.slice(0, 20);
    summary.screenshot = screenshotPath;

    await page.screenshot({ path: screenshotPath, fullPage: true });
    writeJsonFile(networkPath, {
      entries: invokeEntries,
      summary: observed,
    });

    assert(summary.providerPageMounted, "AI 服务商设置页未挂载");
    assert(summary.apiKeyProviderDetailMounted, "Provider 详情区未挂载");
    assert(summary.modelAddCatalogMounted, "添加模型 catalog 面板未挂载");
    assert(
      summary.appServerHandleJsonLinesSeen,
      "未观察到 app_server_handle_json_lines",
    );
    assert(
      summary.missingRequiredAppServerMethods.length === 0,
      `缺少 App Server JSON-RPC 方法: ${summary.missingRequiredAppServerMethods.join(", ")}`,
    );
    assert(
      summary.modelListResponsesValid,
      "model/list response 不符合 { data, nextCursor }",
    );
    assert(
      summary.modelPreferencesListResponsesValid,
      "modelPreferences/list response 缺少 preferences 数组",
    );
    assert(
      summary.modelSyncStateReadResponsesValid,
      "modelSyncState/read response 缺少 syncState 对象",
    );
    assert(
      summary.legacyModelCommandsSeen.length === 0,
      `观察到 legacy model 命令: ${summary.legacyModelCommandsSeen.join(", ")}`,
    );
    assert(
      summary.forbiddenModelSideEffectCommandsSeen.length === 0,
      `观察到禁止的 model side-effect 命令: ${summary.forbiddenModelSideEffectCommandsSeen.join(", ")}`,
    );
    assert(
      summary.forbiddenAppServerMethodsSeen.length === 0,
      `观察到禁止的 App Server method: ${summary.forbiddenAppServerMethodsSeen.join(", ")}`,
    );

    summary.ok = true;
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:model-registry-current] summary=${summaryPath}`);
    console.log(`[smoke:model-registry-current] screenshot=${screenshotPath}`);
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    summary.consoleErrors = consoleErrors;
    summary.failedRequests = failedRequests.slice(0, 20);
    const observed = summarizeInvokeEntries(invokeEntries);
    Object.assign(summary, observed);
    writeJsonFile(networkPath, {
      entries: invokeEntries,
      summary: observed,
    });
    writeJsonFile(summaryPath, summary);

    try {
      const pages = browser?.pages?.() ?? [];
      const page = pages.at(-1);
      if (page) {
        await page.screenshot({
          path: failureScreenshotPath,
          fullPage: true,
        });
        summary.screenshot = failureScreenshotPath;
        writeJsonFile(summaryPath, summary);
      }
    } catch {
      // 失败截图只是诊断证据，不能阻断错误上抛。
    }

    console.error(`[smoke:model-registry-current] summary=${summaryPath}`);
    throw error;
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
    if (tmpProfileDir) {
      fs.rmSync(tmpProfileDir, { recursive: true, force: true });
    }
  }
}

await run();
