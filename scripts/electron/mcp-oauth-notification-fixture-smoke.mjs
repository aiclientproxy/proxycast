#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  LEGACY_MCP_COMMANDS,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import { startMcpOAuthFixtureProvider } from "../mcp/oauth-fixture-smoke.mjs";
import {
  appServerCallFromPage,
  assert,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  openSettings,
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
  sanitizeText,
  sleep,
} from "./mcp-config-fixture-smoke.mjs";
import { parseMcpConfigFixtureArgs } from "./lib/mcp-config-fixture-evidence.mjs";

const APP_SERVER_DRAIN_EVENTS_COMMAND = "app_server_drain_events";
const OPEN_EXTERNAL_URL_COMMAND = "open_external_url";
const OAUTH_COMPLETED_METHOD = "mcpServer/oauthLogin/completed";
const REQUIRED_POST_LOGIN_METHODS = [
  "mcpServer/oauth/login",
  "mcpServerStatus/list",
  "mcpTool/list",
];
const OAUTH_REQUIRED_COPY =
  /需要授权|需要授權|Authorization required|認可が必要|인증 필요/i;
const OAUTH_AUTHORIZED_COPY = /已授权|已授權|Authorized|認可済み|인증됨/i;
const OAUTH_COMPLETED_COPY =
  /授权已完成|授權已完成|authorization completed|認可が完了|인증이 완료/i;
const EXTERNAL_URL_CAPTURE_KEY = "__limeMcpOauthExternalUrls";

const DEFAULTS = {
  runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "mcp-oauth-notification",
  ),
  prefix: "mcp-oauth-notification-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

const LOG_PREFIX = "[smoke:mcp-oauth-notification-fixture]";

function printHelp() {
  console.log(`
MCP OAuth Notification Electron Fixture Smoke

用途:
  从真实 Electron 设置页点击 MCP OAuth 登录，使用本地 OAuth provider 完成回调，
  并证明 App Server typed notification 自动刷新 GUI 授权状态和完成提示。

边界:
  系统浏览器打开动作在隔离 Electron fixture 中被捕获，再由测试进程跟随同一授权 URL；
  OAuth runtime、HTTP provider、callback、App Server event drain 与 Renderer 均走 current 主链。

用法:
  npm run smoke:mcp-oauth-notification-electron-fixture

选项:
  --run-id <id> --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

async function installExternalUrlCapture(app) {
  await app.evaluate(({ shell }, captureKey) => {
    globalThis[captureKey] = [];
    shell.openExternal = async (url) => {
      globalThis[captureKey].push(String(url));
    };
  }, EXTERNAL_URL_CAPTURE_KEY);
}

async function waitForCapturedAuthorizationUrl(app, provider, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(45_000, options.timeoutMs)) {
    const urls = await app.evaluate(
      (_electron, captureKey) => globalThis[captureKey] ?? [],
      EXTERNAL_URL_CAPTURE_KEY,
    );
    const authorizationUrl = urls.find((url) =>
      String(url).startsWith(`${provider.baseUrl}/authorize`),
    );
    if (authorizationUrl) {
      return authorizationUrl;
    }
    await sleep(options.intervalMs);
  }
  throw new Error("GUI 登录未通过 Electron open_external_url 打开授权地址");
}

async function openMcpRuntimeSettings(page, options) {
  await openSettings(page, options);
  await page.locator('[data-testid="settings-sidebar-tab-mcp-server"]').click();
  const runtimeTab = page.locator('[data-testid="mcp-panel-tab-runtime"]');
  await runtimeTab.waitFor({
    state: "visible",
    timeout: Math.min(45_000, options.timeoutMs),
  });
  await runtimeTab.click();
}

function serverCard(page, serverName) {
  return page
    .getByText(serverName, { exact: true })
    .first()
    .locator(
      'xpath=ancestor::div[contains(concat(" ", normalize-space(@class), " "), " border ")][1]',
    );
}

async function waitForServerState(page, serverName, copy, options) {
  const card = serverCard(page, serverName);
  await card.waitFor({
    state: "visible",
    timeout: Math.min(45_000, options.timeoutMs),
  });
  await card.getByText(copy).waitFor({
    state: "visible",
    timeout: Math.min(45_000, options.timeoutMs),
  });
  return card;
}

async function clickOauthLogin(page, serverName, options) {
  const card = await waitForServerState(
    page,
    serverName,
    OAUTH_REQUIRED_COPY,
    options,
  );
  const loginButton = card
    .getByRole("button")
    .filter({ hasText: /登录|登入|Log in|ログイン|로그인/i });
  await loginButton.waitFor({
    state: "visible",
    timeout: Math.min(45_000, options.timeoutMs),
  });
  await loginButton.click();
}

async function clearInvokeEvidence(page) {
  await page.evaluate(() => {
    window.localStorage.removeItem("lime_invoke_error_buffer_v1");
    window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
  });
}

async function readInvokeEvidence(page) {
  return await page.evaluate(() => ({
    traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    url: window.location.href,
    documentLocale: document.documentElement.lang,
    electron: window.__LIME_ELECTRON__ === true,
    hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
  }));
}

function summarizeTrace(traceRaw) {
  const trace = parseInvokeTraceRaw(traceRaw);
  const requests = parseJsonRpcRequestsFromInvokeTrace(traceRaw);
  const requestMethods = [
    ...new Set(requests.map((request) => request.method)),
  ];
  const commands = [
    ...new Set(trace.map((entry) => entry?.command).filter(Boolean)),
  ];
  const relevantCommands = new Set([
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    APP_SERVER_DRAIN_EVENTS_COMMAND,
    OPEN_EXTERNAL_URL_COMMAND,
  ]);
  const relevantEntries = trace.filter((entry) =>
    relevantCommands.has(entry?.command),
  );
  return {
    commands,
    requestMethods,
    appServerHandleJsonLinesHitCount: trace.filter(
      (entry) => entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    ).length,
    appServerDrainEventsHitCount: trace.filter(
      (entry) => entry?.command === APP_SERVER_DRAIN_EVENTS_COMMAND,
    ).length,
    openExternalUrlHitCount: trace.filter(
      (entry) => entry?.command === OPEN_EXTERNAL_URL_COMMAND,
    ).length,
    electronIpcHitCount: relevantEntries.filter(
      (entry) => entry?.transport === "electron-ipc",
    ).length,
    mockFallbackHitCount: relevantEntries.filter(
      (entry) => entry?.transport !== "electron-ipc",
    ).length,
    failedInvokeCount: relevantEntries.filter(
      (entry) => entry?.status !== "success",
    ).length,
    missingPostLoginMethods: REQUIRED_POST_LOGIN_METHODS.filter(
      (method) => !requestMethods.includes(method),
    ),
    legacyMcpCommandsSeen: LEGACY_MCP_COMMANDS.filter((command) =>
      commands.includes(command),
    ),
  };
}

function assertTraceEvidence(evidence) {
  assert(
    evidence.appServerHandleJsonLinesHitCount > 0,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    evidence.appServerDrainEventsHitCount > 0,
    "未观察到 app_server_drain_events，无法证明 notification drain",
  );
  assert(
    evidence.openExternalUrlHitCount > 0,
    "未观察到 Electron open_external_url",
  );
  assert(
    evidence.missingPostLoginMethods.length === 0,
    `OAuth 完成后缺少自动刷新 method: ${evidence.missingPostLoginMethods.join(", ")}`,
  );
  assert(evidence.mockFallbackHitCount === 0, "观察到非 Electron IPC fallback");
  assert(evidence.failedInvokeCount === 0, "观察到 current bridge invoke 失败");
  assert(
    evidence.legacyMcpCommandsSeen.length === 0,
    `观察到 legacy MCP 命令: ${evidence.legacyMcpCommandsSeen.join(", ")}`,
  );
}

function assertProviderEvidence(provider) {
  const authorizeQuery = provider.state.authorizeQueries.at(-1) ?? {};
  assert(
    provider.state.authorizeQueries.length > 0,
    "本地 OAuth provider 未收到 authorize 请求",
  );
  assert(
    authorizeQuery.scope === "fixture.read",
    `OAuth scope 漂移: ${authorizeQuery.scope || "<none>"}`,
  );
  assert(
    provider.state.tokenRequests.some(
      (request) =>
        request.grant_type === "authorization_code" &&
        request.code === "fixture-auth-code",
    ),
    "本地 OAuth provider 未收到 authorization_code token 请求",
  );
}

async function deleteFixtureServer(page, serverId) {
  if (!page || page.isClosed()) {
    return;
  }
  await appServerCallFromPage(page, "mcpServer/delete", { id: serverId }).catch(
    (error) => {
      console.warn(
        `${LOG_PREFIX} fixture cleanup failed: ${sanitizeText(
          error instanceof Error ? error.message : String(error),
        )}`,
      );
    },
  );
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
    forceBuild: true,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: {
      ...runtimeEnv.env,
      APP_SERVER_BIN: appServerBinary,
    },
  });
  const provider = await startMcpOAuthFixtureProvider();
  const serverId = `mcp-oauth-notification-${Date.now()}`;
  const serverName = serverId.replace(/[^a-zA-Z0-9_-]/g, "-");
  const consoleErrors = [];
  const pageErrors = [];
  const summary = {
    schemaVersion: 1,
    scenarioId: "MCP-OAUTH-01-typed-notification",
    proofLevel: "Gate B",
    claimBoundary:
      "Real Electron/preload/IPC, App Server OAuth login and event drain, local OAuth provider callback, and automatic Renderer status/toast projection. System browser navigation is intercepted test-only and followed by the fixture process.",
    testOnly: true,
    ok: false,
    startedAt: new Date().toISOString(),
    completedAt: null,
    pageUrl: null,
    documentLocale: null,
    serverName,
    notificationMethod: OAUTH_COMPLETED_METHOD,
    electronPreloadBridge: false,
    oauthRequiredVisible: false,
    oauthAuthorizedVisible: false,
    completionToastVisible: false,
    provider: {
      host: new URL(provider.baseUrl).host,
      authorizeRequestCount: 0,
      tokenRequestCount: 0,
      scope: null,
    },
    trace: null,
    consoleErrors,
    pageErrors,
    invokeErrorCount: 0,
    screenshot: null,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    failure: null,
  };

  let handle = null;
  let page = null;
  try {
    logStage("launch-electron");
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "unavailable",
    });
    page = handle.page;
    summary.electronPreloadBridge =
      handle.rendererSnapshot.electron &&
      handle.rendererSnapshot.hasInvokeBridge;

    logStage("capture-external-browser-url");
    await installExternalUrlCapture(handle.app);

    logStage("create-oauth-server");
    const createResult = await appServerCallFromPage(page, "mcpServer/create", {
      server: {
        id: serverId,
        name: serverName,
        description: "MCP OAuth typed notification fixture",
        server_config: {
          transport: "streamable_http",
          url: provider.mcpUrl,
          timeout: 3,
          scopes: ["fixture.read"],
        },
        enabled_lime: true,
        enabled_claude: false,
        enabled_codex: false,
        enabled_gemini: false,
        created_at: Date.now(),
      },
    });
    assert(
      Array.isArray(createResult.result?.servers),
      "mcpServer/create 未返回 servers",
    );

    logStage("open-mcp-runtime-settings");
    await openMcpRuntimeSettings(page, options);
    await waitForServerState(page, serverName, OAUTH_REQUIRED_COPY, options);
    summary.oauthRequiredVisible = true;

    await clearInvokeEvidence(page);

    logStage("click-oauth-login");
    await clickOauthLogin(page, serverName, options);
    const authorizationUrl = await waitForCapturedAuthorizationUrl(
      handle.app,
      provider,
      options,
    );

    const authorizedState = waitForServerState(
      page,
      serverName,
      OAUTH_AUTHORIZED_COPY,
      options,
    );
    const completionToast = page
      .locator("[data-sonner-toast]")
      .filter({ hasText: serverName })
      .filter({ hasText: OAUTH_COMPLETED_COPY })
      .waitFor({
        state: "visible",
        timeout: Math.min(45_000, options.timeoutMs),
      });

    logStage("complete-provider-callback");
    const authResponse = await fetch(authorizationUrl, {
      redirect: "follow",
      signal: AbortSignal.timeout(Math.min(30_000, options.timeoutMs)),
    });
    assert(
      authResponse.ok,
      `OAuth callback 跳转失败: HTTP ${authResponse.status}`,
    );
    await Promise.all([authorizedState, completionToast]);
    summary.oauthAuthorizedVisible = true;
    summary.completionToastVisible = true;

    assertProviderEvidence(provider);
    summary.provider.authorizeRequestCount =
      provider.state.authorizeQueries.length;
    summary.provider.tokenRequestCount = provider.state.tokenRequests.length;
    summary.provider.scope =
      provider.state.authorizeQueries.at(-1)?.scope ?? null;

    const invokeEvidence = await readInvokeEvidence(page);
    const traceEvidence = summarizeTrace(invokeEvidence.traceRaw);
    assert(invokeEvidence.electron, "页面不是 Electron renderer");
    assert(invokeEvidence.hasInvokeBridge, "preload invoke bridge 不可用");
    assertTraceEvidence(traceEvidence);
    const invokeErrors = parseInvokeTraceRaw(invokeEvidence.errorRaw);
    assert(
      invokeErrors.length === 0,
      `观察到 invoke error: ${invokeErrors
        .map((entry) => entry?.error)
        .filter(Boolean)
        .join(" | ")}`,
    );
    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );
    assert(
      pageErrors.length === 0,
      `观察到 page error: ${pageErrors.join(" | ")}`,
    );

    await page.screenshot({ path: screenshotPath, fullPage: true });
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    summary.pageUrl = invokeEvidence.url;
    summary.documentLocale = invokeEvidence.documentLocale;
    summary.trace = traceEvidence;
    summary.invokeErrorCount = invokeErrors.length;
    summary.screenshot = path.basename(screenshotPath);
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
    console.log(`${LOG_PREFIX} notification=${OAUTH_COMPLETED_METHOD}`);
  } catch (error) {
    summary.completedAt = new Date().toISOString();
    summary.failure = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    summary.provider.authorizeRequestCount =
      provider.state.authorizeQueries.length;
    summary.provider.tokenRequestCount = provider.state.tokenRequests.length;
    summary.provider.scope =
      provider.state.authorizeQueries.at(-1)?.scope ?? null;
    if (page && !page.isClosed()) {
      try {
        const invokeEvidence = await readInvokeEvidence(page);
        summary.pageUrl = invokeEvidence.url;
        summary.documentLocale = invokeEvidence.documentLocale;
        summary.trace = summarizeTrace(invokeEvidence.traceRaw);
        summary.invokeErrorCount = parseInvokeTraceRaw(
          invokeEvidence.errorRaw,
        ).length;
        await page.screenshot({ path: failureScreenshotPath, fullPage: true });
        summary.failureScreenshot = path.basename(failureScreenshotPath);
      } catch {
        // 截图失败不覆盖原始错误。
      }
    }
    writeJsonFile(summaryPath, summary);
    throw error;
  } finally {
    await deleteFixtureServer(page, serverId);
    await provider.close().catch(() => undefined);
    if (handle) {
      await closeElectronFixture(handle);
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
