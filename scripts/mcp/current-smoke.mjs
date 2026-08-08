#!/usr/bin/env node

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import {
  FIXTURE_METHODS,
  OAUTH_FIXTURE_METHODS,
  REQUIRED_READ_METHODS,
  assert,
  invokeAppServerMethod,
  invokeBridgeCommand,
  runFixtureChecks,
  runReadChecks,
  sanitizeJson,
  summarizeInvokeEntries,
  waitForHealth,
  writeJsonFile,
} from "./lib/current-smoke-core.mjs";
import { writeMcpFixture } from "./lib/current-smoke-fixture.mjs";
import {
  LIVE_PROVIDER_METHODS,
  describeMcpLiveProviderEnv,
  runMcpLiveProviderSmoke,
} from "./live-provider-smoke.mjs";
import { runMcpOAuthFixtureSmoke } from "./oauth-fixture-smoke.mjs";

const DEFAULTS = {
  healthUrl: "http://127.0.0.1:3030/health",
  invokeUrl: "http://127.0.0.1:3030/invoke",
  timeoutMs: 120_000,
  intervalMs: 1_000,
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "mcp-current",
  ),
  prefix: "mcp-current",
  allowWriteFixture: false,
  allowOAuthFixture: false,
  allowLiveProvider: false,
  cleanupFixture: true,
};

function printHelp() {
  console.log(`
MCP Current Smoke

用途:
  通过 DevBridge /invoke 调用 app_server_handle_json_lines，验证 MCP 获取与使用
  走 App Server JSON-RPC current 主链，而不是旧 Tauri MCP facade。

用法:
  npm run smoke:mcp-current
  npm run smoke:mcp-current -- --allow-write-fixture
  npm run smoke:mcp-current -- --allow-oauth-fixture
  npm run smoke:mcp-current -- --allow-live-provider

选项:
  --health-url <url>       DevBridge 健康检查地址，默认 http://127.0.0.1:3030/health
  --invoke-url <url>       DevBridge invoke 地址，默认 http://127.0.0.1:3030/invoke
  --timeout-ms <ms>        总超时，默认 120000
  --interval-ms <ms>       健康检查轮询间隔，默认 1000
  --evidence-dir <path>    证据目录，默认 .lime/qc/gui-evidence/mcp-current
  --prefix <name>          证据文件前缀，默认 mcp-current
  --allow-write-fixture    创建临时 stdio MCP server，覆盖 start / tool call / resource read
  --allow-oauth-fixture    创建本地 OAuth provider，覆盖 mcpServer/oauth/login 与系统浏览器网关
  --allow-live-provider    使用环境变量指定的真实 streamable HTTP MCP provider 做 live-gated E2E
  --keep-fixture           保留本脚本创建的临时 fixture 目录
  -h, --help               显示帮助

Live provider 环境变量:
  ${describeMcpLiveProviderEnv().join("\n  ")}

Live provider 安全约束:
  LIME_MCP_LIVE_SERVER_URL 只接受 http/https，且不得包含 username、password、query 或 hash
  LIME_MCP_LIVE_BEARER_TOKEN_ENV_VAR 必须是环境变量名，不是 token 字面量
  LIME_MCP_LIVE_ENV_HTTP_HEADERS_JSON 的 value 必须是环境变量名，例如 {"X-Api-Key":"MCP_PROVIDER_API_KEY"}
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--health-url" && argv[index + 1]) {
      options.healthUrl = String(argv[++index]).trim();
      continue;
    }
    if (arg === "--invoke-url" && argv[index + 1]) {
      options.invokeUrl = String(argv[++index]).trim();
      continue;
    }
    if (arg === "--timeout-ms" && argv[index + 1]) {
      options.timeoutMs = Number(argv[++index]);
      continue;
    }
    if (arg === "--interval-ms" && argv[index + 1]) {
      options.intervalMs = Number(argv[++index]);
      continue;
    }
    if (arg === "--evidence-dir" && argv[index + 1]) {
      options.evidenceDir = path.resolve(String(argv[++index]).trim());
      continue;
    }
    if (arg === "--prefix" && argv[index + 1]) {
      options.prefix = String(argv[++index]).trim();
      continue;
    }
    if (arg === "--allow-write-fixture") {
      options.allowWriteFixture = true;
      continue;
    }
    if (arg === "--allow-oauth-fixture") {
      options.allowOAuthFixture = true;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--keep-fixture") {
      options.cleanupFixture = false;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
    throw new Error(`未知参数: ${arg}`);
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
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

async function run() {
  if (typeof fetch !== "function") {
    throw new Error("当前 Node 运行时不支持 fetch，请使用 Node 18+");
  }

  const options = parseArgs(process.argv.slice(2));
  if (
    options.allowLiveProvider &&
    !String(process.env.LIME_MCP_LIVE_SERVER_URL || "").trim()
  ) {
    throw new Error(
      "--allow-live-provider requires LIME_MCP_LIVE_SERVER_URL to point at a real streamable HTTP MCP server",
    );
  }
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const networkPath = path.join(
    options.evidenceDir,
    `${options.prefix}-network-invoke.json`,
  );
  const invokeEntries = [];
  let fixture = null;

  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    healthUrl: options.healthUrl,
    invokeUrl: options.invokeUrl,
    smokeMode: [
      "direct-devbridge-app-server-json-rpc",
      options.allowWriteFixture ? "stdio-fixture" : null,
      options.allowOAuthFixture ? "oauth-fixture" : null,
      options.allowLiveProvider ? "live-provider" : null,
    ]
      .filter(Boolean)
      .join("-with-"),
    classification:
      "MCP current path must use app_server_handle_json_lines -> App Server JSON-RPC; legacy mcp_* Tauri facade is guard-only.",
    allowWriteFixture: options.allowWriteFixture,
    allowOAuthFixture: options.allowOAuthFixture,
    allowLiveProvider: options.allowLiveProvider,
    cleanupFixture: options.cleanupFixture,
    health: null,
    fixture: null,
    oauthFixture: null,
    liveProvider: null,
    appServerHandleJsonLinesSeen: false,
    openExternalUrlSeen: false,
    appServerMethodsSeen: [],
    legacyMcpCommandsSeen: [],
    missingReadMethods: [...REQUIRED_READ_METHODS],
    missingFixtureMethods: options.allowWriteFixture
      ? [...FIXTURE_METHODS]
      : [],
    missingOAuthFixtureMethods: options.allowOAuthFixture
      ? [...OAUTH_FIXTURE_METHODS]
      : [],
    missingLiveProviderMethods: options.allowLiveProvider
      ? [...LIVE_PROVIDER_METHODS]
      : [],
    mcpCounts: {
      servers: null,
      statusServers: null,
      tools: null,
      prompts: null,
      resources: null,
      resourceTemplates: null,
    },
    network: networkPath,
    summary: summaryPath,
  };

  try {
    console.log(
      "[smoke:mcp-current] live_provider_submission=status:not_submitted reason:本 smoke 只验证 MCP current JSON-RPC，不提交 Agent turn。",
    );
    summary.health = await waitForHealth(options);

    await runReadChecks(options, invokeEntries);

    if (options.allowWriteFixture) {
      fixture = await writeMcpFixture();
      summary.fixture = sanitizeJson({
        root: fixture.root,
        serverPath: fixture.serverPath,
      });
      Object.assign(
        summary.fixture,
        await runFixtureChecks(options, invokeEntries, fixture),
      );
    }

    if (options.allowOAuthFixture) {
      summary.oauthFixture = await runMcpOAuthFixtureSmoke({
        options,
        entries: invokeEntries,
        invokeAppServerMethod,
        invokeBridgeCommand,
      });
    }

    if (options.allowLiveProvider) {
      summary.liveProvider = await runMcpLiveProviderSmoke({
        options,
        entries: invokeEntries,
        invokeAppServerMethod,
      });
    }

    const observed = summarizeInvokeEntries(invokeEntries);
    Object.assign(summary, observed);
    summary.missingLiveProviderMethods = options.allowLiveProvider
      ? LIVE_PROVIDER_METHODS.filter(
          (method) => !summary.appServerMethodsSeen.includes(method),
        )
      : [];
    writeJsonFile(networkPath, {
      entries: invokeEntries,
      summary: observed,
    });

    assert(
      summary.health?.transport === "electron-host",
      `DevBridge transport 应为 electron-host，实际 ${summary.health?.transport ?? "unknown"}`,
    );
    assert(
      summary.appServerHandleJsonLinesSeen,
      "未观察到 app_server_handle_json_lines",
    );
    assert(
      summary.missingReadMethods.length === 0,
      `缺少 MCP read current methods: ${summary.missingReadMethods.join(", ")}`,
    );
    if (options.allowWriteFixture) {
      assert(
        summary.missingFixtureMethods.length === 0,
        `缺少 MCP fixture current methods: ${summary.missingFixtureMethods.join(", ")}`,
      );
      assert(summary.fixture?.fixtureToolName, "未记录 fixture MCP tool name");
      assert(
        summary.fixture?.outputSchemaStructuredContentSeen === true,
        "未记录 fixture MCP tool output_schema structuredContent",
      );
      assert(
        summary.fixture?.structuredContentEcho?.echoedMessage ===
          "hello current MCP",
        "未记录 fixture MCP tool structuredContent",
      );
      assert(
        summary.fixture?.resourceTemplateUriTemplate === "fixture://item/{id}",
        "未记录 fixture MCP resource template",
      );
      assert(
        summary.fixture?.failureIsolation?.failedStartObserved === true &&
          summary.fixture?.failureIsolation?.failedServerReportedStopped ===
            true &&
          summary.fixture?.failureIsolation?.healthyServerStillRunning ===
            true &&
          summary.fixture?.failureIsolation?.healthyToolStillListed === true &&
          summary.fixture?.failureIsolation?.healthyResourceReadAfterFailure ===
            true,
        "MCP 单 server 故障隔离证据不完整",
      );
    }
    if (options.allowOAuthFixture) {
      assert(
        summary.missingOAuthFixtureMethods.length === 0,
        `缺少 MCP OAuth fixture current methods: ${summary.missingOAuthFixtureMethods.join(", ")}`,
      );
      assert(
        summary.openExternalUrlSeen,
        "未观察到 open_external_url current 网关",
      );
      assert(
        summary.oauthFixture?.authStatus?.mode === "oauth" &&
          summary.oauthFixture?.authStatus?.available === true,
        "MCP OAuth fixture 未记录已授权状态",
      );
    }
    if (options.allowLiveProvider) {
      assert(
        summary.missingLiveProviderMethods.length === 0,
        `缺少 MCP live provider current methods: ${summary.missingLiveProviderMethods.join(", ")}`,
      );
      assert(
        summary.liveProvider?.serverName,
        "MCP live provider 未记录 serverName",
      );
      if (summary.liveProvider?.provider?.toolName) {
        assert(
          summary.liveProvider?.calledTool?.toolName,
          "MCP live provider 未记录指定工具调用结果",
        );
      }
      if (summary.liveProvider?.provider?.resourceUriProvided) {
        assert(
          summary.liveProvider?.readResource?.uriMatchesExpected === true,
          "MCP live provider 未记录指定资源读取结果",
        );
      }
    }
    assert(
      summary.legacyMcpCommandsSeen.length === 0,
      `观察到 legacy MCP 命令: ${summary.legacyMcpCommandsSeen.join(", ")}`,
    );

    summary.ok = true;
    writeJsonFile(summaryPath, summary);
    console.log(`[smoke:mcp-current] summary=${summaryPath}`);
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    const observed = summarizeInvokeEntries(invokeEntries);
    Object.assign(summary, observed);
    summary.missingLiveProviderMethods = options.allowLiveProvider
      ? LIVE_PROVIDER_METHODS.filter(
          (method) => !summary.appServerMethodsSeen.includes(method),
        )
      : [];
    writeJsonFile(networkPath, {
      entries: invokeEntries,
      summary: observed,
    });
    writeJsonFile(summaryPath, summary);

    console.error(`[smoke:mcp-current] summary=${summaryPath}`);
    throw error;
  } finally {
    if (fixture && options.cleanupFixture) {
      await fsp.rm(fixture.root, { recursive: true, force: true });
    }
  }
}

run().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
