#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import electronPath from "electron";
import { _electron as electron } from "playwright";
import { startOpenAiCompatibleFixtureServer } from "../lib/openai-compatible-fixture-server.mjs";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import {
  NORMAL_FINAL_TEXT,
  REQUIRED_METHODS,
  SOURCE_THREAD_ID,
  WORKSPACE_ID,
  assert,
  buildProviderScriptedResponses,
  clearInvokeBuffers,
  createPageAppServerClient,
  createRepositoryProvider,
  createTempRuntimeEnv,
  initializeAndCommitImport,
  providerRequestSummaries,
  runImportedAndNormalTurns,
  sanitizeJson,
  sanitizeText,
  summarizeAndAssertBridge,
  summarizeAndAssertFixture,
  TERMINAL_INTERACTION_CHARS,
  TERMINAL_INTERACTION_SUMMARY,
  waitForRendererReady,
  writeJsonFile,
} from "./lib/codex-import-continuation-fixture.mjs";

const DEFAULTS = {
  appUrl: "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "codex-import-continuation-fixture",
  ),
  prefix: "codex-import-continuation-fixture",
  timeoutMs: 180_000,
  intervalMs: 250,
  keepTemp: false,
};

const LOG_PREFIX = "[smoke:codex-import-continuation-fixture]";
const NAVIGATION_RESTORE_STORAGE_KEY = "lime.appNavigation.restore.v1";

function printHelp() {
  console.log(`
Codex Import Unified Exec Electron Fixture Smoke

用途:
  启动真实 Electron Desktop Host 与 runtime backend，导入 Codex rollout 后
  使用本地 OpenAI-compatible provider 触发 exec_command，再在普通新会话中
  重复同一命令。验证导入零重放、unified exec 工具面、canonical Command
  Item 与普通/导入会话同构。

边界:
  只调用 localhost provider fixture，不调用正式模型；不使用 external/mock
  backend、renderer mock fallback、legacy Bash/PowerShell 工具或旧 runtime command。

用法:
  node scripts/electron/codex-import-continuation-fixture-smoke.mjs

选项:
  --app-url <url>        可选 renderer dev server，例如 http://127.0.0.1:1420/
  --evidence-dir <path>  证据目录
  --prefix <name>        证据文件前缀
  --timeout-ms <ms>      总超时，默认 180000
  --interval-ms <ms>     轮询间隔，默认 250
  --keep-temp            保留临时目录便于调试
  -h, --help             显示帮助
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--app-url" && next) {
      options.appUrl = next.trim();
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
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!options.evidenceDir || !options.prefix) {
    throw new Error("--evidence-dir / --prefix 均不能为空");
  }
  return options;
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

function evidencePaths(options) {
  return {
    summary: path.join(options.evidenceDir, `${options.prefix}-summary.json`),
    raw: path.join(options.evidenceDir, `${options.prefix}-raw.json`),
    provider: path.join(
      options.evidenceDir,
      `${options.prefix}-provider-requests.json`,
    ),
    screenshot: path.join(options.evidenceDir, `${options.prefix}.png`),
    failureScreenshot: path.join(
      options.evidenceDir,
      `${options.prefix}-failure.png`,
    ),
  };
}

async function openSessionInRenderer(page, options, sessionId) {
  await page.evaluate(
    ({ navigationKey, activeSessionId }) => {
      window.sessionStorage.setItem(
        navigationKey,
        JSON.stringify({
          page: "agent",
          params: { initialSessionId: activeSessionId },
        }),
      );
    },
    {
      navigationKey: NAVIGATION_RESTORE_STORAGE_KEY,
      activeSessionId: sessionId,
    },
  );
  await page.reload({
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  await waitForRendererReady(page, options);
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: options.timeoutMs });
  assert(
    (await input.getAttribute("data-session-id")) === sessionId,
    "Renderer 未恢复 unified exec canonical session",
  );
}

async function waitForGuiTerminalInteraction(
  page,
  options,
  { sessionId, threadId, finalText, phase },
) {
  const startedAt = Date.now();
  const timeoutMs = Math.min(options.timeoutMs, 30_000);
  let lastSnapshot = null;
  while (Date.now() - startedAt < timeoutMs) {
    const historicalPreview = page
      .locator('[data-testid^="message-list-historical-timeline-preview:"]')
      .first();
    if (await historicalPreview.isVisible().catch(() => false)) {
      await historicalPreview.click();
    }
    // Completed process groups are collapsed by the production timeline. Expand
    // the command group before inspecting its real tool row.
    await page.evaluate(() => {
      for (const group of document.querySelectorAll(
        '[data-testid="streaming-process-group"]',
      )) {
        const button = group.querySelector("button[aria-expanded]");
        if (button?.getAttribute("aria-expanded") !== "true") {
          button?.click();
        }
      }
    });
    lastSnapshot = await page.evaluate(
      ({ finalText, rawStdinText, threadId }) => {
      const bodyText = document.body?.innerText || "";
      const rows = Array.from(
        document.querySelectorAll('[data-testid="tool-call-row"]'),
      ).map((row) => ({
        text: row.textContent || "",
        toolName: row.getAttribute("data-tool-name") || null,
        status: row.getAttribute("data-tool-status") || null,
      }));
      const messageListFrame = document.querySelector(
        '[data-testid="message-list-frame"]',
      );
      const turnStartTrace = (() => {
        try {
          const entries = JSON.parse(
            window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
          );
          if (!Array.isArray(entries)) {
            return null;
          }
          for (const entry of entries) {
            if (entry?.command !== "app_server_handle_json_lines") {
              continue;
            }
            const lines = entry?.args_preview?.request?.lines;
            if (!Array.isArray(lines)) {
              continue;
            }
            for (const line of lines) {
              try {
                const message = JSON.parse(String(line));
                if (
                  message?.method === "turn/start" &&
                  message?.params?.threadId === threadId
                ) {
                  return {
                    method: message.method,
                    threadId: message.params.threadId,
                    transport: entry.transport || null,
                    status: entry.status || null,
                  };
                }
              } catch {
                // Ignore unrelated malformed diagnostic lines.
              }
            }
          }
        } catch {
          return null;
        }
        return null;
      })();
      return {
        url: window.location.href,
        messageListSessionId:
          messageListFrame?.getAttribute("data-session-id") || null,
        commandRows: rows.filter((row) => row.toolName === "exec_command"),
        hasSummary: bodyText.includes("sent 9 chars"),
        hasRawStdin: bodyText.includes(rawStdinText),
        hasFinalText: bodyText.includes(finalText),
        turnStartTrace,
      };
      },
      {
        finalText,
        rawStdinText: TERMINAL_INTERACTION_CHARS.trim(),
        threadId,
      },
    );
    if (
      lastSnapshot?.messageListSessionId &&
      lastSnapshot.messageListSessionId === sessionId &&
      lastSnapshot.hasSummary &&
      lastSnapshot.hasFinalText &&
      !lastSnapshot.hasRawStdin &&
      lastSnapshot.commandRows.some((row) => row.status === "completed")
    ) {
      return {
        phase,
        sessionId,
        threadId,
        url: lastSnapshot.url,
        messageListSessionId: lastSnapshot.messageListSessionId,
        commandRowCount: lastSnapshot.commandRows.length,
        terminalInteractionSummary: TERMINAL_INTERACTION_SUMMARY,
        turnStartTrace: lastSnapshot.turnStartTrace,
        rawStdinProjected: false,
        finalTextVisible: true,
      };
    }
    await new Promise((resolve) => setTimeout(resolve, options.intervalMs));
  }
  throw new Error(
    `${phase} GUI terminal interaction 摘要未出现: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function selectRuntimeRouteInRenderer(
  page,
  options,
  { sessionId, route },
) {
  assert(
    route?.providerName && route?.model,
    "Renderer 模型选择缺少 provider/model route",
  );
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  const modelSelector = input.locator(
    'xpath=ancestor::*[@data-testid="inputbar-core-container"][1]//*[@data-testid="model-selector"]',
  );
  await modelSelector.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 10_000),
  });
  await modelSelector.click();

  const popover = page.locator('[data-model-selector-popover="true"]');
  await popover.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 10_000),
  });
  const providerButton = popover
    .locator("button")
    .filter({ hasText: route.providerName })
    .first();
  await providerButton.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 10_000),
  });
  await providerButton.click();

  const modelButton = popover
    .locator("button")
    .filter({ hasText: route.model })
    .first();
  await modelButton.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 10_000),
  });
  await modelButton.click();
  await popover.waitFor({
    state: "hidden",
    timeout: Math.min(options.timeoutMs, 10_000),
  });

  const selectionHandle = await page.waitForFunction(
    ({ expectedModel, expectedProvider, activeSessionId }) => {
      const textarea = document.querySelector(
        `textarea[name="agent-chat-message"][data-session-id="${activeSessionId}"]`,
      );
      const container = textarea?.closest(
        '[data-testid="inputbar-core-container"]',
      );
      const trigger = container?.querySelector('[data-testid="model-selector"]');
      const text = trigger?.textContent?.trim() || "";
      const title = trigger?.getAttribute("title") || "";
      return text.includes(expectedModel) && title.includes(expectedProvider)
        ? { text, title }
        : null;
    },
    {
      expectedModel: route.model,
      expectedProvider: route.providerName,
      activeSessionId: sessionId,
    },
    { timeout: Math.min(options.timeoutMs, 10_000) },
  );
  const selection = await selectionHandle.jsonValue();
  return {
    sessionId,
    providerId: route.providerId,
    providerName: route.providerName,
    model: route.model,
    triggerText: selection.text,
    triggerTitle: selection.title,
  };
}

async function runNormalTurnInRenderer(
  page,
  options,
  { normalSessionId, normalThreadId, route, text },
) {
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${normalSessionId}"]`,
  );
  await input.fill(text);
  const sendButton = input.locator(
    'xpath=ancestor::*[@data-testid="inputbar-core-container"][1]//*[@data-testid="send-btn"]',
  );
  await page.waitForFunction(
    ({ activeSessionId, expectedModel, expectedText }) => {
      const textarea = document.querySelector(
        `textarea[name="agent-chat-message"][data-session-id="${activeSessionId}"]`,
      );
      const container = textarea?.closest(
        '[data-testid="inputbar-core-container"]',
      );
      const trigger = container?.querySelector('[data-testid="model-selector"]');
      const button = container?.querySelector('[data-testid="send-btn"]');
      return Boolean(
        textarea instanceof HTMLTextAreaElement &&
          textarea.value === expectedText &&
          trigger?.textContent?.includes(expectedModel) &&
          button instanceof HTMLButtonElement &&
          !button.disabled,
      );
    },
    {
      activeSessionId: normalSessionId,
      expectedModel: route.model,
      expectedText: text,
    },
    { timeout: Math.min(options.timeoutMs, 10_000) },
  );
  await sendButton.click({ timeout: Math.min(options.timeoutMs, 10_000) });
  return await waitForGuiTerminalInteraction(page, options, {
    sessionId: normalSessionId,
    threadId: normalThreadId,
    finalText: NORMAL_FINAL_TEXT,
    phase: "live",
  });
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const paths = evidencePaths(options);
  const runtimeEnv = createTempRuntimeEnv();
  const providerScript = buildProviderScriptedResponses(runtimeEnv);
  let providerFixture = null;
  let app = null;
  let page = null;
  let client = null;
  const consoleErrors = [];
  const guiTerminalInteraction = { live: null, reloaded: null };
  let guiModelSelection = null;

  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    appUrl: options.appUrl || null,
    sourceThreadId: SOURCE_THREAD_ID,
    workspaceId: WORKSPACE_ID,
    backendMode: "runtime",
    requiredMethods: REQUIRED_METHODS,
    appServerBinary,
    electronPreloadBridge: false,
    providerBaseUrl: null,
    gateBBridge: null,
    fixtureSummary: null,
    importedIdentity: null,
    guiTerminalInteraction,
    guiModelSelection,
    consoleErrors,
    screenshot: null,
    rawEvidence: paths.raw,
    providerEvidence: paths.provider,
    summary: paths.summary,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
  };

  try {
    logStage("start-local-provider");
    providerFixture = await startOpenAiCompatibleFixtureServer({
      scriptedResponses: providerScript.responses,
    });
    summary.providerBaseUrl = providerFixture.baseUrl;

    logStage("launch-electron-runtime");
    app = await electron.launch({
      executablePath: electronPath,
      args: ["--use-mock-keychain", "."],
      cwd: process.cwd(),
      env: {
        ...runtimeEnv.env,
        ...appServerEnv,
        APP_SERVER_BACKEND_MODE: "runtime",
        ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
        LIME_ELECTRON_E2E: "1",
        LIME_ELECTRON_BRAND_DEV_APP: "0",
        LIME_ELECTRON_CLEAR_RENDERER_CACHE: "0",
        LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
        ...(options.appUrl ? { VITE_DEV_SERVER_URL: options.appUrl } : {}),
      },
      timeout: options.timeoutMs,
    });
    app.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(sanitizeText(message.text()));
      }
    });

    page = await app.firstWindow({ timeout: options.timeoutMs });
    page.setDefaultTimeout(options.timeoutMs);
    await page.setViewportSize({ width: 1440, height: 1000 });

    logStage("wait-renderer");
    const renderer = await waitForRendererReady(page, options);
    summary.electronPreloadBridge =
      renderer.electron && renderer.hasInvokeBridge;
    await clearInvokeBuffers(page);
    client = createPageAppServerClient(page);

    logStage("commit-import-zero-replay");
    const initial = await initializeAndCommitImport(
      client,
      runtimeEnv,
      {
        ...options,
        beforeCommit: () =>
          createRepositoryProvider(client, providerFixture.provider),
      },
    );
    const route = initial.beforeCommitResult;
    summary.importedIdentity = sanitizeJson({
      jobSessionId: initial.sessionId,
      jobThreadId: initial.threadId,
      readSessionId: initial.importedRead?.thread?.sessionId ?? null,
      readThreadId: initial.importedRead?.thread?.id ?? null,
    });
    const providerRequestsAfterCommit = providerFixture.requests.length;
    assert(
      providerRequestsAfterCommit === 0,
      `导入 commit 触发了 ${providerRequestsAfterCommit} 次 provider 请求`,
    );

    logStage("run-imported-and-normal-unified-exec");
    const turns = await runImportedAndNormalTurns(client, {
      importedSessionId: initial.sessionId,
      importedThreadId: initial.threadId,
      route,
      runtimeEnv,
      command: providerScript.command,
      options,
      onNormalThreadReady: async ({
        normalSessionId,
        normalThreadId,
        route: normalRoute,
      }) => {
        logStage("open-normal-thread-before-live-turn");
        summary.normalIdentity = {
          sessionId: normalSessionId,
          threadId: normalThreadId,
        };
        await openSessionInRenderer(page, options, normalSessionId);
        assert(normalThreadId, "normal thread identity 不能为空");
        logStage("select-normal-thread-fixture-model");
        guiModelSelection = await selectRuntimeRouteInRenderer(page, options, {
          sessionId: normalSessionId,
          route: normalRoute,
        });
        summary.guiModelSelection = sanitizeJson(guiModelSelection);
      },
      runNormalTurnInRenderer: async (context) => {
        logStage("send-normal-turn-through-renderer");
        guiTerminalInteraction.live = await runNormalTurnInRenderer(
          page,
          options,
          context,
        );
        return guiTerminalInteraction.live;
      },
      onNormalTurnCompleted: async ({
        normalSessionId,
        normalThreadId,
      }) => {
        logStage("reload-and-assert-terminal-interaction-recovery");
        await openSessionInRenderer(page, options, normalSessionId);
        guiTerminalInteraction.reloaded =
          await waitForGuiTerminalInteraction(page, options, {
            sessionId: normalSessionId,
            threadId: normalThreadId,
            finalText: NORMAL_FINAL_TEXT,
            phase: "reloaded",
          });
      },
    });
    writeJsonFile(
      paths.raw,
      sanitizeJson({ initial, turns, requests: client.requests }),
    );
    writeJsonFile(
      paths.provider,
      sanitizeJson(providerRequestSummaries(providerFixture.requests)),
    );
    summary.gateBBridge = sanitizeJson({
      ...summarizeAndAssertBridge(client),
      rendererTurnStart: guiTerminalInteraction.live?.turnStartTrace ?? null,
    });
    const fixtureSummary = summarizeAndAssertFixture({
      client,
      initial,
      turns,
      providerRequestsAfterCommit,
      providerRequests: providerFixture.requests,
      command: providerScript.command,
      runtimeEnv,
    });
    summary.fixtureSummary = sanitizeJson(fixtureSummary);

    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );

    await page.screenshot({ path: paths.screenshot, fullPage: true });
    summary.screenshot = paths.screenshot;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(paths.summary, summary);
    console.log(`${LOG_PREFIX} summary=${paths.summary}`);
    console.log(
      `${LOG_PREFIX} importedSession=${fixtureSummary.sessionId} normalSession=${fixtureSummary.normalSessionId} providerRequests=${fixtureSummary.providerRequests.length}`,
    );
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    if (page) {
      try {
        summary.failureDiagnostics = await page.evaluate(() => {
          const readJson = (key) => {
            try {
              const parsed = JSON.parse(localStorage.getItem(key) || "[]");
              return Array.isArray(parsed) ? parsed : [];
            } catch {
              return [];
            }
          };
          const trace = readJson("lime_invoke_trace_buffer_v1");
          return {
            bodyText: document.body?.innerText || "",
            processGroups: Array.from(
              document.querySelectorAll('[data-testid="streaming-process-group"]'),
            ).map((group) => ({
              text: group.textContent || "",
              expanded:
                group.querySelector("button")?.getAttribute("aria-expanded") ||
                null,
            })),
            toolRows: Array.from(
              document.querySelectorAll('[data-testid="tool-call-row"]'),
            ).map((row) => ({
              text: row.textContent || "",
              toolName: row.getAttribute("data-tool-name") || null,
              status: row.getAttribute("data-tool-status") || null,
            })),
            trace: trace.slice(-40),
            invokeErrors: readJson("lime_invoke_error_buffer_v1").slice(-20),
          };
        });
      } catch {
        summary.failureDiagnostics = { unavailable: true };
      }
    }
    summary.failureClient = sanitizeJson({
      requests: client?.requests ?? [],
      messages: client?.messages ?? [],
    });
    if (client && summary.normalIdentity?.threadId) {
      try {
        summary.failureThreadRead = sanitizeJson(
          await client.call("thread/read", {
            threadId: summary.normalIdentity.threadId,
            includeTurns: true,
          }),
        );
      } catch {
        summary.failureThreadRead = null;
      }
    }
    if (providerFixture) {
      summary.failureProviderRequests = sanitizeJson(
        providerRequestSummaries(providerFixture.requests),
      );
    }
    if (page) {
      try {
        await page.screenshot({
          path: paths.failureScreenshot,
          fullPage: true,
        });
        summary.failureScreenshot = paths.failureScreenshot;
      } catch {
        // The summary still records the original product failure.
      }
    }
    writeJsonFile(paths.summary, summary);
    throw error;
  } finally {
    if (app) {
      await app.close().catch(() => undefined);
    }
    if (providerFixture) {
      await providerFixture.close().catch(() => undefined);
    }
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

run().catch((error) => {
  console.error(
    `${LOG_PREFIX} failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
