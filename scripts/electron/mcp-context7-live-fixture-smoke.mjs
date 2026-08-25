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
  sanitizeJson,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  appServerCallFromPage,
  assert,
  assertContext7Server,
  closeElectronFixture,
  createContext7ConfigFromGui,
  createTempRuntimeEnv,
  getServerConfig,
  launchElectronFixture,
  openMcpConfigSettings,
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
  sanitizeText,
  sleep,
  summarizeContext7Server,
  waitForContext7Server,
} from "./mcp-config-fixture-smoke.mjs";

const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "mcp-context7-live-fixture",
  ),
  prefix: "mcp-context7-live-fixture",
  timeoutMs: 120_000,
  intervalMs: 500,
  keepTemp: false,
};

const LOG_PREFIX = "[smoke:mcp-context7-live-fixture]";
const CONTEXT7_SERVER_NAME = "Context7";
const CONTEXT7_LIVE_URL = "https://mcp.context7.com/mcp";
const CONTEXT7_HEADER_NAME = "CONTEXT7_API_KEY";
const CONTEXT7_ENV_VAR_NAME = "CONTEXT7_API_KEY";
const CONTEXT7_LIBRARY_ID = "/openai/openai-agents-python";
const REQUIRED_METHODS = [
  "mcpServer/create",
  "mcpServer/list",
  "mcpServer/start",
  "mcpServerStatus/list",
  "mcpTool/list",
  "mcpTool/listForContext",
  "mcpTool/search",
  "thread/start",
  "mcpServer/tool/call",
];

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

function printHelp() {
  console.log(`
MCP Context7 Live Electron Fixture Smoke

用途:
  启动真实 Electron Desktop Host，经设置页 GUI 创建 Context7 配置，然后通过
  app_server_handle_json_lines 启动 Context7、搜索 MCP 工具并调用 query-docs。

边界:
  这是 live-gated 远程 MCP provider 验证；只记录 host、工具名、env var 名和
  调用摘要，不写入 key、header value、完整工具正文或旧 MCP facade 证据。

用法:
  node scripts/electron/mcp-context7-live-fixture-smoke.mjs

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
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

function contentSummary(result) {
  const content = Array.isArray(result?.content) ? result.content : [];
  const structuredContent =
    result?.structuredContent ?? result?.structured_content;
  return {
    contentCount: content.length,
    contentTypes: Array.from(
      new Set(content.map((item) => item?.type).filter(Boolean)),
    ).sort(),
    structuredContentKeys:
      structuredContent && typeof structuredContent === "object"
        ? Object.keys(structuredContent).sort()
        : [],
    isError: result?.is_error ?? result?.isError ?? null,
  };
}

function isContext7Tool(tool, suffix) {
  const name = String(tool?.name || "");
  return /^mcp__.*context7.*__/i.test(name) && name.endsWith(`__${suffix}`);
}

function assertToolResult(method, result) {
  const summary = contentSummary(result);
  assert(summary.isError !== true, `${method} returned isError=true`);
  assert(summary.contentCount > 0, `${method} returned empty content`);
  return summary;
}

async function cleanupContext7Server(call, server) {
  if (!server?.id) {
    return;
  }
  await call("mcpServer/stop", {
    name: server.name ?? CONTEXT7_SERVER_NAME,
  }).catch((error) => {
    console.warn(
      `${LOG_PREFIX} context7 stop failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  });
  await call("mcpServer/delete", {
    id: server.id,
  }).catch((error) => {
    console.warn(
      `${LOG_PREFIX} context7 delete failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  });
}

async function waitForRunningContext7(call, options) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < Math.min(60_000, options.timeoutMs)) {
    latest = await call("mcpServerStatus/list", {});
    const server = (latest.result?.data ?? []).find(
      (item) => item?.name === CONTEXT7_SERVER_NAME,
    );
    if (server?.runtimeStatus === "connected") {
      return { statusResult: latest, runtimeServer: server };
    }
    await sleep(options.intervalMs);
  }
  throw new Error("Context7 未进入 running 状态");
}

async function waitForContext7Tools(call, options) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < Math.min(60_000, options.timeoutMs)) {
    const toolsResult = await call("mcpTool/list", {});
    const toolsForContextResult = await call("mcpTool/listForContext", {
      caller: "assistant",
      includeDeferred: true,
    });
    const searchResult = await call("mcpTool/search", {
      query: "query-docs",
      caller: "tool_search",
      limit: 20,
    });
    const tools = Array.isArray(toolsResult.result?.tools)
      ? toolsResult.result.tools
      : [];
    const toolsForContext = Array.isArray(toolsForContextResult.result?.tools)
      ? toolsForContextResult.result.tools
      : [];
    const searchedTools = Array.isArray(searchResult.result?.tools)
      ? searchResult.result.tools
      : [];
    const resolveTool =
      tools.find((tool) => isContext7Tool(tool, "resolve-library-id")) ??
      searchedTools.find((tool) => isContext7Tool(tool, "resolve-library-id"));
    const queryDocsTool =
      tools.find((tool) => isContext7Tool(tool, "query-docs")) ??
      searchedTools.find((tool) => isContext7Tool(tool, "query-docs"));
    latest = {
      toolsResult,
      toolsForContextResult,
      searchResult,
      tools,
      toolsForContext,
      searchedTools,
      resolveTool,
      queryDocsTool,
    };
    if (
      resolveTool &&
      queryDocsTool &&
      toolsForContext.some((tool) => tool?.name === queryDocsTool.name) &&
      searchedTools.some((tool) => tool?.name === queryDocsTool.name)
    ) {
      return latest;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Context7 工具未出现在 list/listForContext/search: ${JSON.stringify(
      sanitizeJson({
        toolCount: latest?.tools?.length ?? 0,
        toolsForContextCount: latest?.toolsForContext?.length ?? 0,
        searchedToolCount: latest?.searchedTools?.length ?? 0,
      }),
    )}`,
  );
}

function summarizeTrace(traceRaw, observedMethods) {
  const requests = parseJsonRpcRequestsFromInvokeTrace(traceRaw);
  const commands = Array.from(
    new Set(
      parseInvokeTraceRaw(traceRaw)
        .map((entry) => entry?.command)
        .filter(Boolean),
    ),
  );
  const methods = Array.from(
    new Set([
      ...Array.from(observedMethods),
      ...requests.map((request) => request.method).filter(Boolean),
    ]),
  );
  return {
    appServerHandleJsonLinesSeen:
      observedMethods.size > 0 ||
      commands.includes(APP_SERVER_HANDLE_JSON_LINES_COMMAND),
    requestMethods: methods,
    missingRequiredMethods: REQUIRED_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    legacyMcpCommandsSeen: LEGACY_MCP_COMMANDS.filter((command) =>
      commands.includes(command),
    ),
    requests,
  };
}

function assertTraceSummary(summary) {
  assert(
    summary.appServerHandleJsonLinesSeen,
    "未观察到 app_server_handle_json_lines",
  );
  assert(
    summary.missingRequiredMethods.length === 0,
    `缺少 App Server current method: ${summary.missingRequiredMethods.join(", ")}`,
  );
  assert(
    summary.legacyMcpCommandsSeen.length === 0,
    `观察到 legacy MCP 命令: ${summary.legacyMcpCommandsSeen.join(", ")}`,
  );
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
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
    ok: false,
    checkedAt: new Date().toISOString(),
    backendMode: "unavailable",
    context7ApiKeyEnvPresent: Boolean(process.env.CONTEXT7_API_KEY),
    context7Preset: {
      serverName: CONTEXT7_SERVER_NAME,
      urlHost: new URL(CONTEXT7_LIVE_URL).host,
      envHttpHeaderNames: [CONTEXT7_HEADER_NAME],
      envHttpHeaderEnvVars: [CONTEXT7_ENV_VAR_NAME],
    },
    electronPreloadBridge: false,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    electronUserDataDir: options.keepTemp
      ? runtimeEnv.electronUserDataDir
      : null,
    appServerBinary: options.keepTemp
      ? appServerBinary
      : path.basename(appServerBinary),
    context7Server: null,
    runtimeServer: null,
    queryDocsToolName: null,
    resolveLibraryToolName: null,
    resolveLibraryCall: null,
    queryDocsCall: null,
    appServerHandleJsonLinesSeen: false,
    electronRequestMethods: [],
    missingRequiredMethods: [...REQUIRED_METHODS],
    legacyMcpCommandsSeen: [],
    consoleErrors: [],
    screenshot: null,
    rawEvidence: rawEvidencePath,
    summary: summaryPath,
  };

  let app = null;
  let page = null;
  let createdServer = null;
  const consoleErrors = [];
  const rawEvidence = {};
  const observedMethods = new Set();
  const call = async (method, params = {}) => {
    const result = await appServerCallFromPage(page, method, params);
    observedMethods.add(method);
    return result;
  };

  try {
    logStage("launch-electron");
    const handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
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
      await createContext7ConfigFromGui(page, {
        configUrl: CONTEXT7_LIVE_URL,
        envVarName: CONTEXT7_ENV_VAR_NAME,
      }),
    );
    const { listResult, server } = await waitForContext7Server(page, options);
    observedMethods.add("mcpServer/create");
    observedMethods.add("mcpServer/list");
    assertContext7Server(server, {
      configUrl: CONTEXT7_LIVE_URL,
      envVarName: CONTEXT7_ENV_VAR_NAME,
    });
    createdServer = server;
    summary.context7Server = summarizeContext7Server(server);
    rawEvidence.mcpServerList = sanitizeJson(listResult);
    assert(
      getServerConfig(server)?.env_http_headers?.[CONTEXT7_HEADER_NAME] ===
        CONTEXT7_ENV_VAR_NAME,
      "Context7 live fixture 未通过 env var 名引用 API key header",
    );

    logStage("start-context7");
    await call("mcpServer/start", {
      name: CONTEXT7_SERVER_NAME,
    });
    const { statusResult, runtimeServer } = await waitForRunningContext7(
      call,
      options,
    );
    summary.runtimeServer = sanitizeJson({
      name: runtimeServer?.name ?? null,
      runtimeStatus: runtimeServer?.runtimeStatus ?? null,
      authStatus: runtimeServer?.authStatus ?? null,
    });
    rawEvidence.status = sanitizeJson(statusResult);

    logStage("discover-context7-tools");
    const toolEvidence = await waitForContext7Tools(call, options);
    summary.resolveLibraryToolName = toolEvidence.resolveTool.name;
    summary.queryDocsToolName = toolEvidence.queryDocsTool.name;
    rawEvidence.toolDiscovery = sanitizeJson({
      toolNames: toolEvidence.tools.map((tool) => tool?.name).filter(Boolean),
      toolsForContextNames: toolEvidence.toolsForContext
        .map((tool) => tool?.name)
        .filter(Boolean),
      searchedToolNames: toolEvidence.searchedTools
        .map((tool) => tool?.name)
        .filter(Boolean),
    });

    logStage("call-context7-resolve-library");
    const threadResult = await call("thread/start", {
      model: "gpt-5.4",
      modelProvider: "openai",
    });
    const threadId = threadResult.result?.thread?.id;
    if (typeof threadId !== "string" || threadId.length === 0) {
      throw new Error("thread/start did not return a thread id");
    }
    const resolveCall = await call("mcpServer/tool/call", {
      threadId,
      server: CONTEXT7_SERVER_NAME,
      tool: toolEvidence.resolveTool.name.replace(
        `mcp__${CONTEXT7_SERVER_NAME}__`,
        "",
      ),
      arguments: {
        libraryName: "openai agents python",
        query: "openai agents python",
      },
    });
    summary.resolveLibraryCall = {
      toolName: toolEvidence.resolveTool.name,
      ...assertToolResult(
        "mcpServer/tool/call resolve-library-id",
        resolveCall.result,
      ),
    };
    rawEvidence.resolveLibraryCall = sanitizeJson(resolveCall);

    logStage("call-context7-query-docs");
    const queryDocsCall = await call("mcpServer/tool/call", {
      threadId,
      server: CONTEXT7_SERVER_NAME,
      tool: toolEvidence.queryDocsTool.name.replace(
        `mcp__${CONTEXT7_SERVER_NAME}__`,
        "",
      ),
      arguments: {
        libraryId: CONTEXT7_LIBRARY_ID,
        query: "AI Agent 是什么",
      },
    });
    summary.queryDocsCall = {
      toolName: toolEvidence.queryDocsTool.name,
      libraryId: CONTEXT7_LIBRARY_ID,
      ...assertToolResult(
        "mcpServer/tool/call query-docs",
        queryDocsCall.result,
      ),
    };
    rawEvidence.queryDocsCall = sanitizeJson(queryDocsCall);

    const trace = summarizeTrace(queryDocsCall.traceRaw, observedMethods);
    assertTraceSummary(trace);
    summary.appServerHandleJsonLinesSeen = trace.appServerHandleJsonLinesSeen;
    summary.electronRequestMethods = trace.requestMethods;
    summary.missingRequiredMethods = trace.missingRequiredMethods;
    summary.legacyMcpCommandsSeen = trace.legacyMcpCommandsSeen;
    rawEvidence.electronRequests = sanitizeJson(trace.requests);

    await cleanupContext7Server(call, createdServer);
    createdServer = null;

    await page.screenshot({ path: screenshotPath, fullPage: true });
    await closeElectronFixture(handle);
    app = null;
    page = null;

    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );

    summary.consoleErrors = consoleErrors;
    summary.screenshot = screenshotPath;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(rawEvidencePath, rawEvidence);
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
    console.log(`${LOG_PREFIX} queryDocs=${summary.queryDocsToolName ?? ""}`);
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    summary.consoleErrors = consoleErrors;
    if (Object.keys(rawEvidence).length > 0) {
      writeJsonFile(rawEvidencePath, rawEvidence);
    }
    writeJsonFile(summaryPath, summary);
    if (page) {
      try {
        if (createdServer) {
          await cleanupContext7Server(call, createdServer);
        }
        await page.screenshot({
          path: failureScreenshotPath,
          fullPage: true,
        });
        summary.failureScreenshot = failureScreenshotPath;
        writeJsonFile(summaryPath, summary);
      } catch {
        // 截图或清理失败不覆盖原始错误。
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
