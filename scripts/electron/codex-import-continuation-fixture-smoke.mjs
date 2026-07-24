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
  REQUIRED_METHODS,
  SOURCE_THREAD_ID,
  WORKSPACE_ID,
  assert,
  buildProviderScriptedResponses,
  clearInvokeBuffers,
  createPageAppServerClient,
  createTempRuntimeEnv,
  initializeAndCommitImport,
  providerRequestSummaries,
  runImportedAndNormalTurns,
  sanitizeJson,
  sanitizeText,
  summarizeAndAssertBridge,
  summarizeAndAssertFixture,
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

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const paths = evidencePaths(options);
  const runtimeEnv = createTempRuntimeEnv();
  const providerScript = buildProviderScriptedResponses(runtimeEnv);
  let providerFixture = null;
  let app = null;
  let page = null;
  const consoleErrors = [];

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
    const client = createPageAppServerClient(page);

    logStage("commit-import-zero-replay");
    const initial = await initializeAndCommitImport(
      client,
      runtimeEnv,
      options,
    );
    const providerRequestsAfterCommit = providerFixture.requests.length;
    assert(
      providerRequestsAfterCommit === 0,
      `导入 commit 触发了 ${providerRequestsAfterCommit} 次 provider 请求`,
    );

    logStage("run-imported-and-normal-unified-exec");
    const turns = await runImportedAndNormalTurns(client, {
      importedSessionId: initial.sessionId,
      importedThreadId: initial.threadId,
      provider: providerFixture.provider,
      runtimeEnv,
      command: providerScript.command,
      options,
    });
    writeJsonFile(
      paths.raw,
      sanitizeJson({ initial, turns, requests: client.requests }),
    );
    writeJsonFile(
      paths.provider,
      sanitizeJson(providerRequestSummaries(providerFixture.requests)),
    );
    summary.gateBBridge = sanitizeJson(summarizeAndAssertBridge(client));
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
