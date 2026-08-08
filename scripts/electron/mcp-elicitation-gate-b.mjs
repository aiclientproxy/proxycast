#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { ensureElectronFixtureBuild } from "../lib/electron-fixture-build.mjs";
import { startOpenAiCompatibleFixtureServer } from "../lib/openai-compatible-fixture-server.mjs";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  LEGACY_MCP_COMMANDS,
  sanitizeJson,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  appServerCallFromPage,
  assert,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
  sanitizeText,
  sleep,
} from "./mcp-config-fixture-smoke.mjs";
import {
  findPluginPackageMcpAppItems,
  installPluginPackageEmbeddedBrowserLifecycleCapture,
  PLUGIN_PACKAGE_MCP_APP_MARKER,
  PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI,
  pluginPackageMcpAppHtml,
  waitForPluginPackageMcpAppHistoryUnavailable,
  waitForPluginPackageMcpAppSurface,
} from "./plugin-mcp-app-gate-b.mjs";

const DEFAULTS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "mcp-elicitation-gate-b",
  ),
  prefix: "mcp-elicitation-gate-b",
  timeoutMs: 240_000,
  intervalMs: 250,
  keepTemp: false,
};

const LOG_PREFIX = "[smoke:mcp-elicitation-gate-b]";
const FINAL_TEXT = "MCP_ELICITATION_GATE_B_DONE";
const DISABLED_BOUNDARY_FINAL_TEXT = "PLUGIN_DISABLED_BOUNDARY_DONE";
const TOOL_SUFFIX = "release_check";
const MCP_TOOL_CALL_ID = "call-mcp-elicitation-release-check";
const PLUGIN_PACKAGE_ID = "mcp-elicitation-plugin";
const PLUGIN_PACKAGE_CONFIG_NAME = `${PLUGIN_PACKAGE_ID}@local`;
const PLUGIN_PACKAGE_SERVER_ID = "demo";
const PLUGIN_PACKAGE_SKILL_ID = "release-check";
const PLUGIN_PACKAGE_SKILL_BODY_MARKER =
  "Use the package MCP release check tool when the user asks for a release check.";
const PLUGIN_PACKAGE_DISPLAY_NAME = PLUGIN_PACKAGE_ID;
const PLUGIN_PACKAGE_BUNDLED_ID = "browser";
const PLUGIN_PACKAGE_BUNDLED_MARKETPLACE_ID = "openai-bundled";
const PLUGIN_PACKAGE_BUNDLED_VERSION = "1.0.0";
const DYNAMIC_TOOL_NAME = "desktop__appInfo";
const DYNAMIC_TOOL_CALL_ID = "call-desktop-app-info";
const NAVIGATION_RESTORE_STORAGE_KEY = "lime.appNavigation.restore.v1";
const REQUIRED_METHODS = [
  "workspace/default/ensure",
  "modelProvider/create",
  "modelProvider/update",
  "modelProviderKey/create",
  "model/list",
  "mcpServer/create",
  "mcpServer/start",
  "mcpTool/list",
  "thread/start",
  "thread/settings/update",
  "turn/start",
  "thread/read",
];
const PLUGIN_PACKAGE_REQUIRED_METHODS = [
  "workspace/default/ensure",
  "modelProvider/create",
  "modelProvider/update",
  "modelProviderKey/create",
  "model/list",
  "plugin/list",
  "plugin/install",
  "plugin/read",
  "plugin/installed",
  "plugin/enabled/set",
  "plugin/uninstall",
  "mcpServer/resource/read",
  "thread/start",
  "thread/settings/update",
  "turn/start",
  "thread/read",
];

function printHelp() {
  console.log(`
MCP Elicitation Gate B

用途:
  启动真实 Electron Desktop Host、localhost OpenAI-compatible provider fixture 和
  临时 stdio MCP server，验证 Agent turn -> scoped MCP tool -> elicitation/create
  -> App Server reverse request -> Renderer 表单 -> MCP tool result -> provider final text。

边界:
  使用 APP_SERVER_BACKEND_MODE=runtime 与真实 Electron preload/JSONL bridge。
  不使用显式管理面工具调用证明、通用 action 回答、mock backend、renderer mock
  或 legacy MCP facade 作为成功路径。Gate B 同时校验 runtime MCP client 在 initialize
  请求中广告 form elicitation capability。

用法:
  npm run smoke:mcp-elicitation-gate-b

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

function parseArgs(argv, defaults = DEFAULTS) {
  const options = { ...defaults };
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
  return options;
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

function makeServerName() {
  return `ElicitationGate${Date.now().toString(36)}${process.pid.toString(36)}`;
}

function toolName(serverName) {
  return `mcp__${serverName}__${TOOL_SUFFIX}`;
}

function readRepositoryCommit(rootDir) {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: rootDir,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

async function readElectronRuntime(app) {
  return app.evaluate(({ app: electronApp }) => ({
    appVersion: electronApp.getVersion(),
    arch: process.arch,
    electronVersion: process.versions.electron ?? null,
    pid: process.pid,
    platform: process.platform,
  }));
}

function pluginPackageServerName() {
  return `plugin__${PLUGIN_PACKAGE_ID}__${PLUGIN_PACKAGE_SERVER_ID}`;
}

function readJsonLines(filePath) {
  if (!fs.existsSync(filePath)) return [];
  return fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [JSON.parse(line)];
      } catch {
        return [];
      }
    });
}

function writeElicitationFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lime-mcp-elicitation-"));
  const serverPath = path.join(root, "elicitation-fixture.mjs");
  const ledgerPath = path.join(root, "elicitation-ledger.jsonl");
  fs.writeFileSync(
    serverPath,
    String.raw`import fs from "node:fs";
import readline from "node:readline";

const ledgerPath = process.argv[2];
const mcpAppResourceUri = ${JSON.stringify(PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI)};
const mcpAppHtml = ${JSON.stringify(pluginPackageMcpAppHtml())};
const pending = new Map();
let nextElicitationId = 1;
let initializedProtocolVersion = null;
let initializedCapabilities = null;
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

function send(message) {
  process.stdout.write(JSON.stringify(message) + "\n");
}

function result(id, value) {
  send({ jsonrpc: "2.0", id, result: value });
}

function record(value) {
  fs.appendFileSync(ledgerPath, JSON.stringify(value) + "\n");
}

function isExactEmptyObject(value) {
  return value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length === 0;
}

function supportsFormElicitation() {
  return initializedProtocolVersion === "2025-06-18" &&
    initializedCapabilities !== null &&
    typeof initializedCapabilities === "object" &&
    !Array.isArray(initializedCapabilities) &&
    Object.keys(initializedCapabilities).length === 1 &&
    isExactEmptyObject(initializedCapabilities.elicitation);
}

rl.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  const { id, method, params } = message;

  if (method === "initialize") {
    initializedProtocolVersion = params?.protocolVersion ?? null;
    initializedCapabilities = params?.capabilities ?? null;
    record({
      type: "initialize",
      pid: process.pid,
      protocolVersion: initializedProtocolVersion,
      clientCapabilities: initializedCapabilities,
    });
    result(id, {
      protocolVersion: initializedProtocolVersion ?? "2025-03-26",
      capabilities: { resources: {}, tools: {} },
      serverInfo: { name: "elicitation-gate-b-fixture", version: "1.0.0" },
    });
    return;
  }
  if (method === "notifications/initialized") return;
  if (method === "tools/list") {
    result(id, {
      tools: [{
        name: "release_check",
        description: "Request a release confirmation through MCP elicitation",
        inputSchema: {
          type: "object",
          "x-lime": {
            deferred_loading: false,
            always_visible: true,
            allowed_callers: ["assistant"],
          },
          properties: { release: { type: "string" } },
          required: ["release"],
          additionalProperties: false,
        },
        _meta: { ui: { resourceUri: mcpAppResourceUri } },
      }],
    });
    return;
  }
  if (method === "tools/call") {
    record({ type: "tool_call", pid: process.pid, name: params?.name ?? null });
    if (!supportsFormElicitation()) {
      record({
        type: "capability_missing",
        pid: process.pid,
        protocolVersion: initializedProtocolVersion,
        clientCapabilities: initializedCapabilities,
      });
      result(id, {
        content: [{ type: "text", text: "runtime client did not advertise form elicitation" }],
        isError: true,
      });
      return;
    }
    const elicitationId = "elicitation-" + nextElicitationId;
    nextElicitationId += 1;
    pending.set(elicitationId, { toolCallId: id, release: params?.arguments?.release ?? null });
    send({
      jsonrpc: "2.0",
      id: elicitationId,
      method: "elicitation/create",
      params: {
        message: "Confirm the release check",
        requestedSchema: {
          type: "object",
          properties: { confirmed: { type: "boolean", title: "confirmed" } },
          required: ["confirmed"],
          additionalProperties: false,
        },
      },
    });
    return;
  }
  if (method === "resources/read") {
    record({ type: "resource_read", pid: process.pid, uri: params?.uri ?? null });
    result(id, {
      contents: [{
        uri: params?.uri,
        mimeType: "text/html;profile=mcp-app",
        text: mcpAppHtml,
        _meta: {
          ui: {
            csp: {
              baseUriDomains: [],
              connectDomains: [],
              frameDomains: [],
              resourceDomains: [],
            },
          },
        },
      }],
    });
    return;
  }
  if (pending.has(String(id))) {
    const request = pending.get(String(id));
    pending.delete(String(id));
    const action = message?.result?.action ?? "missing";
    const content = message?.result?.content ?? null;
    record({ type: "elicitation_result", pid: process.pid, action, content, release: request.release });
    result(request.toolCallId, {
      content: [{ type: "text", text: "release_check:" + action }],
      structuredContent: { action, confirmed: content?.confirmed ?? null },
      isError: action !== "accept" || content?.confirmed !== true,
    });
    return;
  }
  result(id, { content: [], isError: true });
});
`,
    "utf8",
  );
  return { ledgerPath, root, serverPath };
}

function preparePluginPackage(fixture) {
  const skillDir = path.join(
    fixture.root,
    "skills",
    PLUGIN_PACKAGE_SKILL_ID,
  );
  fs.mkdirSync(skillDir, { recursive: true });
  fs.writeFileSync(
    path.join(skillDir, "SKILL.md"),
    `---\nname: ${PLUGIN_PACKAGE_SKILL_ID}\ndescription: Use the package release check workflow.\n---\n\n# Release check\n\n${PLUGIN_PACKAGE_SKILL_BODY_MARKER}\n`,
    "utf8",
  );
  fs.writeFileSync(
    path.join(fixture.root, "plugin.json"),
    `${JSON.stringify(
      {
        $schema:
          "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
        name: PLUGIN_PACKAGE_ID,
        version: "1.0.0",
        description: "Agent Plugins standard package Gate B fixture",
      },
      null,
      2,
    )}\n`,
  );
  fs.writeFileSync(
    path.join(fixture.root, "mcp.json"),
    `${JSON.stringify(
      {
        $schema:
          "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
        mcpServers: {
          [PLUGIN_PACKAGE_SERVER_ID]: {
            type: "stdio",
            command: "node",
            args: [path.basename(fixture.serverPath), fixture.ledgerPath],
            cwd: "${PLUGIN_ROOT}",
          },
        },
      },
      null,
      2,
    )}\n`,
  );
  return {
    id: PLUGIN_PACKAGE_ID,
    root: fixture.root,
    runtimeServerName: pluginPackageServerName(),
    serverId: PLUGIN_PACKAGE_SERVER_ID,
    skillId: PLUGIN_PACKAGE_SKILL_ID,
    version: "1.0.0",
  };
}

async function cleanupMcpServer(page, server) {
  if (!page || !server?.id) return;
  await appServerCallFromPage(page, "mcpServer/stop", {
    name: server.name,
  }).catch(() => undefined);
  await appServerCallFromPage(page, "mcpServer/delete", {
    id: server.id,
  }).catch(() => undefined);
}

async function cleanupPluginPackage(page, plugin) {
  if (!page || !plugin?.id) return;
  await appServerCallFromPage(page, "plugin/uninstall", {
    pluginId: plugin.id,
  }).catch(() => undefined);
}

async function observeAppServerMethodFromTrace(
  page,
  method,
  options,
  observedMethods,
) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 45_000)) {
    const traceRaw =
      (await page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      )) || "";
    if (
      parseJsonRpcRequestsFromInvokeTrace(traceRaw).some(
        (request) => request?.method === method,
      )
    ) {
      observedMethods.add(method);
      return traceRaw;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(`Renderer 未通过 App Server bridge 调用 ${method}`);
}

async function armControlledPluginSourceDialog(app, sourcePath) {
  await app.evaluate(({ dialog }, controlledSourcePath) => {
    globalThis.__limePluginPackageDirectoryDialogCalls = [];
    const originalShowOpenDialog = dialog.showOpenDialog.bind(dialog);
    dialog.showOpenDialog = async (...args) => {
      const dialogOptions = args.at(-1) ?? null;
      globalThis.__limePluginPackageDirectoryDialogCalls.push(dialogOptions);
      dialog.showOpenDialog = originalShowOpenDialog;
      return {
        canceled: false,
        filePaths: [controlledSourcePath],
      };
    };
  }, sourcePath);
}

async function readControlledPluginSourceDialogCalls(app) {
  return app.evaluate(() =>
    Array.isArray(globalThis.__limePluginPackageDirectoryDialogCalls)
      ? globalThis.__limePluginPackageDirectoryDialogCalls
      : [],
  );
}

async function waitForPluginInstalledState(
  page,
  pluginId,
  expectedEnabled,
  options,
  observedMethods,
) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 45_000)) {
    latest = await appServerCallFromPage(page, "plugin/installed", {});
    observedMethods.add(latest.method);
    const plugin = latest.result?.plugins?.find(
      (candidate) => candidate?.id === pluginId,
    );
    if (plugin && plugin.enabled === expectedEnabled) {
      return { plugin, response: latest };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `plugin/installed 状态未收敛: plugin=${pluginId} enabled=${expectedEnabled} latest=${JSON.stringify(latest?.result)}`,
  );
}

async function openPluginPackageAppCenter(
  page,
  options,
  observedMethods,
  screenshotPath,
) {
  await page.locator('[data-testid="app-sidebar-nav-plugins"]').click();
  await page.locator('[data-testid="plugin-catalog-install-local"]').waitFor({
    state: "visible",
    timeout: options.timeoutMs,
  });
  await page.locator('[data-testid="plugin-catalog-loading"]').waitFor({
    state: "hidden",
    timeout: options.timeoutMs,
  });

  const catalog = await appServerCallFromPage(page, "plugin/list", {});
  observedMethods.add(catalog.method);
  const bundled = catalog.result?.plugins?.find(
    (candidate) => candidate?.id === PLUGIN_PACKAGE_BUNDLED_ID,
  );
  assert(bundled, "plugin/list 未返回 bundled Browser Plugin");
  assert(
    bundled.source === "bundled" &&
      bundled.marketplaceId === PLUGIN_PACKAGE_BUNDLED_MARKETPLACE_ID &&
      bundled.version === PLUGIN_PACKAGE_BUNDLED_VERSION,
    `bundled Plugin identity/version 不一致: ${JSON.stringify(bundled)}`,
  );
  const bundledCard = page.locator(
    `[data-testid="plugin-catalog-card-${PLUGIN_PACKAGE_BUNDLED_ID}"]`,
  );
  await bundledCard.waitFor({ state: "visible", timeout: options.timeoutMs });
  assert(
    (await bundledCard.textContent())?.includes(PLUGIN_PACKAGE_BUNDLED_VERSION),
    "App Center bundled 卡片未显示 manifest version",
  );
  await page.screenshot({ path: screenshotPath, fullPage: true });
  return { bundled, catalog };
}

async function installPluginPackageFromAppCenter({
  app,
  page,
  plugin,
  options,
  observedMethods,
  screenshotPath,
}) {
  await armControlledPluginSourceDialog(app, plugin.root);
  await page.locator('[data-testid="plugin-catalog-install-local"]').click();
  const review = page.locator('[data-testid="plugin-catalog-install-review"]');
  await review.waitFor({ state: "visible", timeout: options.timeoutMs });
  assert(
    (await review.textContent())?.includes(plugin.id),
    "App Center 安装 review 未显示 Plugin identity",
  );
  const directoryDialogCalls = await readControlledPluginSourceDialogCalls(app);
  assert(
    directoryDialogCalls.length === 1 &&
      directoryDialogCalls[0]?.properties?.includes("openDirectory"),
    `未观察到 Electron 原生目录选择: ${JSON.stringify(directoryDialogCalls)}`,
  );
  await page.screenshot({ path: screenshotPath, fullPage: true });

  await page.locator('[data-testid="plugin-catalog-confirm-install"]').click();
  await review.waitFor({ state: "hidden", timeout: options.timeoutMs });
  await observeAppServerMethodFromTrace(
    page,
    "plugin/install",
    options,
    observedMethods,
  );
  await page
    .locator(`[data-testid="plugin-catalog-actions-${plugin.id}"]`)
    .waitFor({ state: "visible", timeout: options.timeoutMs });

  const installed = await waitForPluginInstalledState(
    page,
    plugin.id,
    true,
    options,
    observedMethods,
  );
  const read = await appServerCallFromPage(page, "plugin/read", {
    pluginId: plugin.id,
  });
  observedMethods.add(read.method);
  const summary = read.result?.plugin?.summary;
  assert(summary?.id === plugin.id, "plugin/read identity 不一致");
  assert(
    String(summary?.contentDigest || "").startsWith("sha256:"),
    "plugin/read 未返回 content digest",
  );
  assert(
    read.result?.plugin?.mcpServers?.some(
      (server) => server?.id === plugin.serverId,
    ),
    "plugin/read 未投影标准 mcp.json server",
  );
  assert(
    summary?.mcpServersCount === 1,
    `plugin/read 标准 mcp.json capability 数量异常: ${summary?.mcpServersCount}`,
  );
  assert(
    read.result?.plugin?.skills?.some(
      (skill) => skill?.id === plugin.skillId,
    ),
    "plugin/read 未投影标准 skills/<name>/SKILL.md",
  );
  assert(
    summary?.skillsCount === 1,
    `plugin/read 标准 Skill capability 数量异常: ${summary?.skillsCount}`,
  );
  return {
    directoryDialogCalls,
    installed,
    pluginSkillProjected: true,
    read,
    standardManifestSeen:
      summary?.id === plugin.id && summary?.version === plugin.version,
    standardMcpConfigSeen: true,
  };
}

async function setPluginPackageEnabledFromAppCenter(
  page,
  pluginId,
  enabled,
  options,
  observedMethods,
) {
  await page.locator('[data-testid="app-sidebar-nav-plugins"]').click();
  await page.locator('[data-testid="plugin-catalog-install-local"]').waitFor({
    state: "visible",
    timeout: options.timeoutMs,
  });
  await page.locator('[data-testid="plugin-catalog-loading"]').waitFor({
    state: "hidden",
    timeout: options.timeoutMs,
  });
  await page
    .locator(`[data-testid="plugin-catalog-actions-${pluginId}"]`)
    .waitFor({ state: "visible", timeout: options.timeoutMs });
  await page.locator(`[data-testid="plugin-catalog-actions-${pluginId}"]`).click();
  await page.locator(`[data-testid="plugin-catalog-toggle-${pluginId}"]`).click();
  await observeAppServerMethodFromTrace(
    page,
    "plugin/enabled/set",
    options,
    observedMethods,
  );
  return waitForPluginInstalledState(
    page,
    pluginId,
    enabled,
    options,
    observedMethods,
  );
}

async function verifyPluginPackageDisabledNewThreadBoundary({
  page,
  fixture,
  plugin,
  options,
  observedMethods,
}) {
  const workspace = await ensureWorkspace(page, observedMethods);
  const route = await createRepositoryProvider(page, fixture, observedMethods);
  const start = await appServerCallFromPage(page, "thread/start", {
    cwd: workspace.rootPath,
    historyMode: "paginated",
    model: route.model,
    modelProvider: route.providerId,
    runtimeWorkspaceRoots: [workspace.rootPath],
    serviceName: "Plugin disabled boundary",
    threadSource: "appServer",
  });
  observedMethods.add(start.method);
  const sessionId = String(start.result?.thread?.sessionId || "").trim();
  const threadId = String(start.result?.thread?.id || "").trim();
  assert(sessionId && threadId, "禁用边界 thread/start 未返回 identity");
  const update = await appServerCallFromPage(page, "thread/settings/update", {
    threadId,
    modelProvider: route.providerId,
    model: route.model,
  });
  observedMethods.add(update.method);

  await openRuntimeThreadInGui(page, sessionId, options);
  const guiModelSelection = await selectRuntimeRouteInRenderer(page, options, {
    sessionId,
    route,
  });
  await page.locator('[data-testid="inputbar-plus-trigger"]').click();
  await page.locator('[data-testid="inputbar-plus-menu"]').waitFor({
    state: "visible",
    timeout: options.timeoutMs,
  });
  await page.locator('[data-testid="inputbar-plus-plugins"]').click();
  const pluginPanel = page.locator(
    '[data-testid="inputbar-plus-panel-plugins"]',
  );
  await pluginPanel.waitFor({ state: "visible", timeout: options.timeoutMs });
  const disabledOption = pluginPanel
    .locator('[data-testid="inputbar-plugin-option"]')
    .filter({ hasText: PLUGIN_PACKAGE_DISPLAY_NAME })
    .first();
  await disabledOption.waitFor({
    state: "visible",
    timeout: options.timeoutMs,
  });
  const disabledOptionBlocked = await disabledOption.isDisabled();
  assert(
    disabledOptionBlocked,
    "禁用 Plugin 在新 Thread 的 Claw picker 中仍可调用",
  );
  await page.keyboard.press("Escape");
  await page.keyboard.press("Escape");

  const turn = await appServerCallFromPage(page, "turn/start", {
    threadId,
    clientUserMessageId: `plugin-disabled-${Date.now()}-${process.pid}`,
    input: [
      {
        type: "text",
        text: "Confirm the disabled plugin boundary without using plugin tools.",
      },
    ],
    cwd: workspace.rootPath,
    runtimeWorkspaceRoots: [workspace.rootPath],
    model: route.model,
    approvalPolicy: "never",
    sandboxPolicy: "danger-full-access",
    additionalContext: {
      metadata: {
        kind: "application",
        value: JSON.stringify({
          harness: { source: "smoke:plugin-catalog-disabled-boundary" },
          tool_scope: {
            allowed_tools: [
              DYNAMIC_TOOL_NAME,
              toolName(plugin.runtimeServerName),
            ],
          },
        }),
      },
    },
  });
  observedMethods.add(turn.method);
  const turnId = String(turn.result?.turn?.id || "").trim();
  assert(turnId, "禁用边界 turn/start 未返回 turn.id");

  const startedAt = Date.now();
  let latestRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latestRead = await appServerCallFromPage(page, "thread/read", {
      threadId,
      includeTurns: true,
    });
    observedMethods.add(latestRead.method);
    if (
      fixture.requests.length >= 1 &&
      JSON.stringify(latestRead.result || {}).includes(
        DISABLED_BOUNDARY_FINAL_TEXT,
      )
    ) {
      break;
    }
    await sleep(options.intervalMs);
  }
  assert(
    fixture.requests.length >= 1 &&
      JSON.stringify(latestRead?.result || {}).includes(
        DISABLED_BOUNDARY_FINAL_TEXT,
      ),
    "禁用边界 turn 未进入 terminal read model",
  );
  const requests = providerRequestSummary(fixture.requests);
  const expectedToolName = toolName(plugin.runtimeServerName);
  assert(
    requests.every((request) => !request.toolNames.includes(expectedToolName)),
    "禁用 Plugin 的 MCP tool 泄露到新 Thread provider request",
  );
  assert(
    findPluginPackageMcpAppItems(latestRead.result).length === 0,
    "禁用 Plugin 在新 Thread 生成了 MCP tool item",
  );

  return {
    pluginPickerBlocked: disabledOptionBlocked,
    guiModelSelection,
    providerRequests: requests,
    providerToolAbsent: true,
    sessionId,
    start,
    threadId,
    threadPluginItemsAbsent: true,
    turn,
    turnId,
    update,
  };
}

async function waitForPluginUninstalled(
  page,
  pluginId,
  options,
  observedMethods,
) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 45_000)) {
    latest = await appServerCallFromPage(page, "plugin/installed", {});
    observedMethods.add(latest.method);
    if (
      !latest.result?.plugins?.some((candidate) => candidate?.id === pluginId)
    ) {
      return latest;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `plugin/installed 卸载状态未收敛: plugin=${pluginId} latest=${JSON.stringify(latest?.result)}`,
  );
}

async function uninstallPluginPackageFromAppCenter({
  page,
  pluginId,
  options,
  observedMethods,
  screenshotPath,
}) {
  await page.locator('[data-testid="app-sidebar-nav-plugins"]').click();
  await page.locator('[data-testid="plugin-catalog-install-local"]').waitFor({
    state: "visible",
    timeout: options.timeoutMs,
  });
  await page.locator('[data-testid="plugin-catalog-loading"]').waitFor({
    state: "hidden",
    timeout: options.timeoutMs,
  });
  const actions = page.locator(`[data-testid="plugin-catalog-actions-${pluginId}"]`);
  await actions.waitFor({ state: "visible", timeout: options.timeoutMs });
  await actions.click();
  await page.locator(`[data-testid="plugin-catalog-uninstall-${pluginId}"]`).click();
  const confirmation = page.locator(
    '[data-testid="plugin-catalog-uninstall-confirm"]',
  );
  await confirmation.waitFor({ state: "visible", timeout: options.timeoutMs });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  await confirmation.locator("button").last().click();
  await confirmation.waitFor({ state: "hidden", timeout: options.timeoutMs });
  await observeAppServerMethodFromTrace(
    page,
    "plugin/uninstall",
    options,
    observedMethods,
  );
  await waitForPluginUninstalled(page, pluginId, options, observedMethods);
}

async function waitForTool(page, expectedToolName, options, observedMethods) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latest = await appServerCallFromPage(page, "mcpTool/list", {});
    observedMethods.add(latest.method);
    const tools = Array.isArray(latest.result?.tools)
      ? latest.result.tools
      : [];
    if (tools.some((tool) => tool?.name === expectedToolName)) return latest;
    await sleep(options.intervalMs);
  }
  throw new Error(
    `MCP tool 未就绪: ${expectedToolName}; latest=${JSON.stringify(latest?.result)}`,
  );
}

async function ensureWorkspace(page, observedMethods) {
  const response = await appServerCallFromPage(
    page,
    "workspace/default/ensure",
    {},
  );
  observedMethods.add(response.method);
  const workspaceId = String(response.result?.workspace?.id || "").trim();
  const rootPath = String(
    response.result?.workspace?.root_path ||
      response.result?.workspace?.rootPath ||
      "",
  ).trim();
  assert(
    workspaceId && rootPath,
    "workspace/default/ensure 未返回 workspace identity",
  );
  return { response, rootPath, workspaceId };
}

async function createRepositoryProvider(page, fixture, observedMethods) {
  const providerName = `MCP elicitation Gate B ${Date.now()}`;
  const created = await appServerCallFromPage(page, "modelProvider/create", {
    name: providerName,
    providerType: fixture.provider.providerName,
    apiHost: fixture.provider.providerConfig.baseUrl,
  });
  observedMethods.add(created.method);
  const providerId = String(created.result?.provider?.id || "").trim();
  assert(providerId, "modelProvider/create 未返回 provider.id");

  const updated = await appServerCallFromPage(page, "modelProvider/update", {
    providerId,
    enabled: true,
    sortOrder: 1,
    models: [
      {
        id: fixture.provider.modelPreference,
        capability: fixture.provider.providerConfig.modelCapabilities,
      },
    ],
  });
  observedMethods.add(updated.method);

  const key = await appServerCallFromPage(page, "modelProviderKey/create", {
    providerId,
    apiKey: fixture.provider.providerConfig.apiKey,
    alias: "mcp-elicitation-gate-b",
    replaceExisting: true,
  });
  observedMethods.add(key.method);
  assert(key.result?.key?.id, "modelProviderKey/create 未返回 key.id");

  const catalog = await appServerCallFromPage(page, "model/list", {
    includeHidden: true,
    limit: 500,
  });
  observedMethods.add(catalog.method);
  const model = Array.isArray(catalog.result?.data)
    ? catalog.result.data.find(
        (candidate) =>
          candidate?.providerId === providerId &&
          candidate?.model === fixture.provider.modelPreference,
      )
    : null;
  assert(model, "model/list 未返回可执行 fixture route");
  const capability = model.capabilitySnapshot;
  assert(
    capability?.taskFamilies?.includes("chat") &&
      capability?.outputModalities?.includes("text") &&
      capability?.runtimeFeatures?.includes("tool_calling"),
    "fixture route 缺少 chat/text/tool_calling capability",
  );

  return {
    catalogModel: model,
    model: fixture.provider.modelPreference,
    providerId,
    providerName,
  };
}

async function openRuntimeThreadInGui(page, sessionId, options) {
  await page.evaluate(
    ({ navigationKey, sessionId: activeSessionId }) => {
      sessionStorage.setItem(
        navigationKey,
        JSON.stringify({
          page: "agent",
          params: { initialSessionId: activeSessionId },
        }),
      );
    },
    { navigationKey: NAVIGATION_RESTORE_STORAGE_KEY, sessionId },
  );
  await page.reload({
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: options.timeoutMs });
  assert(
    (await input.getAttribute("data-session-id")) === sessionId,
    "Renderer 未恢复 MCP Gate B canonical session",
  );
  return { activeSessionId: sessionId };
}

async function readRendererHistoryRequests(page, method, sessionId) {
  const traceRaw =
    (await page.evaluate(() =>
      window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    )) || "";
  return parseJsonRpcRequestsFromInvokeTrace(traceRaw).filter(
    (request) =>
      request?.method === method && request?.params?.threadId === sessionId,
  );
}

async function waitForRendererHistoryHydrationAfterClick(
  page,
  sessionId,
  previousCounts,
  options,
) {
  const methods = ["thread/read", "thread/items/list", "thread/turns/list"];
  const startedAt = Date.now();
  let requestsByMethod = {};
  while (Date.now() - startedAt < options.timeoutMs) {
    requestsByMethod = Object.fromEntries(
      await Promise.all(
        methods.map(async (method) => [
          method,
          await readRendererHistoryRequests(page, method, sessionId),
        ]),
      ),
    );
    if (
      methods.every(
        (method) => requestsByMethod[method].length > previousCounts[method],
      )
    ) {
      return {
        requests: Object.fromEntries(
          methods.map((method) => [
            method,
            {
              count: requestsByMethod[method].length,
              latest: requestsByMethod[method].at(-1),
              previousCount: previousCounts[method],
            },
          ]),
        ),
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `侧栏点击后 Renderer 未完成 canonical history hydration: session=${sessionId} previous=${JSON.stringify(previousCounts)} actual=${JSON.stringify(Object.fromEntries(methods.map((method) => [method, requestsByMethod[method]?.length ?? 0])))}`,
  );
}

async function readRendererHistoryRequestCounts(page, sessionId) {
  const methods = ["thread/read", "thread/items/list", "thread/turns/list"];
  return Object.fromEntries(
    await Promise.all(
      methods.map(async (method) => [
        method,
        (await readRendererHistoryRequests(page, method, sessionId)).length,
      ]),
    ),
  );
}

async function readRendererSessionSwitchSuccessEntries(page, sessionId) {
  return page.evaluate((activeSessionId) => {
    const entries = window.__LIME_AGENTUI_PERF__?.entries?.() ?? [];
    return entries.filter(
      (entry) =>
        entry?.phase === "session.switch.success" &&
        entry?.sessionId === activeSessionId,
    );
  }, sessionId);
}

async function readRendererSessionSwitchSuccessCount(page, sessionId) {
  return (await readRendererSessionSwitchSuccessEntries(page, sessionId))
    .length;
}

async function waitForRendererSessionSwitchSuccessAfterClick(
  page,
  sessionId,
  previousCount,
  options,
) {
  const startedAt = Date.now();
  let count = 0;
  while (Date.now() - startedAt < options.timeoutMs) {
    const entries = await readRendererSessionSwitchSuccessEntries(
      page,
      sessionId,
    );
    count = entries.length;
    if (count > previousCount) {
      return { count, latest: entries.at(-1) ?? null, previousCount };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `侧栏点击后 Renderer session switch 未完成: session=${sessionId} previous=${previousCount} actual=${count}`,
  );
}

async function switchRuntimeThreadInGuiViaSidebar(
  page,
  { sourceSessionId, targetSessionId },
  options,
) {
  assert(
    sourceSessionId && targetSessionId && sourceSessionId !== targetSessionId,
    "侧栏冷恢复需要两个不同的 canonical session",
  );
  const sourceButton = page.locator(
    `[data-testid="app-sidebar-conversation-open"][data-session-id="${sourceSessionId}"]`,
  );
  const targetButton = page.locator(
    `[data-testid="app-sidebar-conversation-open"][data-session-id="${targetSessionId}"]`,
  );
  await sourceButton.waitFor({ state: "visible", timeout: options.timeoutMs });
  await targetButton.waitFor({ state: "visible", timeout: options.timeoutMs });
  const sourceReadCountsBefore = await readRendererHistoryRequestCounts(
    page,
    sourceSessionId,
  );
  const sourceSwitchSuccessCountBefore =
    await readRendererSessionSwitchSuccessCount(page, sourceSessionId);
  await sourceButton.click();
  const sourceInput = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${sourceSessionId}"]`,
  );
  await sourceInput.waitFor({ state: "visible", timeout: options.timeoutMs });
  const sourceHydration = await waitForRendererHistoryHydrationAfterClick(
    page,
    sourceSessionId,
    sourceReadCountsBefore,
    options,
  );
  const sourceSwitch = await waitForRendererSessionSwitchSuccessAfterClick(
    page,
    sourceSessionId,
    sourceSwitchSuccessCountBefore,
    options,
  );

  const targetReadCountsBefore = await readRendererHistoryRequestCounts(
    page,
    targetSessionId,
  );
  const targetSwitchSuccessCountBefore =
    await readRendererSessionSwitchSuccessCount(page, targetSessionId);
  await targetButton.click();
  const targetInput = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${targetSessionId}"]`,
  );
  await targetInput.waitFor({ state: "visible", timeout: options.timeoutMs });
  await page.waitForFunction(
    ({ sessionId }) =>
      document
        .querySelector(
          `[data-testid="app-sidebar-conversation-open"][data-session-id="${sessionId}"]`,
        )
        ?.getAttribute("aria-current") === "page",
    { sessionId: targetSessionId },
    { timeout: options.timeoutMs },
  );
  const targetHydration = await waitForRendererHistoryHydrationAfterClick(
    page,
    targetSessionId,
    targetReadCountsBefore,
    options,
  );
  const targetSwitch = await waitForRendererSessionSwitchSuccessAfterClick(
    page,
    targetSessionId,
    targetSwitchSuccessCountBefore,
    options,
  );

  return {
    sourceHydration,
    sourceSessionId,
    sourceSwitch,
    targetHydration,
    targetSessionId,
    targetSwitch,
  };
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
      const trigger = container?.querySelector(
        '[data-testid="model-selector"]',
      );
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

function findPluginPackageMentionIdentity(result) {
  const thread = result?.thread;
  for (const turn of thread?.turns ?? []) {
    for (const item of turn?.items ?? []) {
      const mention = item?.content?.find(
        (part) =>
          part?.type === "mention" &&
          part?.path === `plugin://${PLUGIN_PACKAGE_CONFIG_NAME}`,
      );
      if (mention) {
        return {
          mention,
          threadId: String(thread?.id || "").trim(),
          turnId: String(turn?.id || "").trim(),
          userItemId: String(item?.id || "").trim(),
        };
      }
    }
  }
  return null;
}

async function waitForPluginPackageGuiTurn(
  page,
  threadId,
  options,
  observedMethods,
) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latest = await appServerCallFromPage(page, "thread/read", {
      threadId,
      includeTurns: true,
    });
    observedMethods.add(latest.method);
    const identity = findPluginPackageMentionIdentity(latest.result);
    if (
      identity?.threadId === threadId &&
      identity.turnId &&
      identity.userItemId
    ) {
      return { identity, read: latest };
    }
    await sleep(10);
  }
  throw new Error(
    `Claw turn 未投影 plugin://${PLUGIN_PACKAGE_CONFIG_NAME} mention: ${JSON.stringify(latest?.result)}`,
  );
}

async function startPluginPackageTurnFromGui({
  page,
  runtime,
  options,
  observedMethods,
  screenshotPath,
}) {
  const guiModelSelection = await selectRuntimeRouteInRenderer(page, options, {
    sessionId: runtime.sessionId,
    route: runtime.route,
  });
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${runtime.sessionId}"]`,
  );
  await input.fill(
    `Use $${PLUGIN_PACKAGE_SKILL_ID} to run the release check through the available MCP tool.`,
  );
  await page.locator('[data-testid="inputbar-plus-trigger"]').click();
  await page.locator('[data-testid="inputbar-plus-menu"]').waitFor({
    state: "visible",
    timeout: options.timeoutMs,
  });
  await page.locator('[data-testid="inputbar-plus-plugins"]').click();
  const pluginPanel = page.locator(
    '[data-testid="inputbar-plus-panel-plugins"]',
  );
  await pluginPanel.waitFor({ state: "visible", timeout: options.timeoutMs });
  const pluginOption = pluginPanel
    .locator('[data-testid="inputbar-plugin-option"]')
    .filter({ hasText: PLUGIN_PACKAGE_DISPLAY_NAME })
    .first();
  await pluginOption.waitFor({ state: "visible", timeout: options.timeoutMs });
  await pluginOption.click();
  const badge = page.locator('[data-testid="inputbar-plugin-badge"]');
  await badge.waitFor({ state: "visible", timeout: options.timeoutMs });
  assert(
    (await badge.textContent())?.includes(PLUGIN_PACKAGE_DISPLAY_NAME),
    "Claw Plugin badge identity 不一致",
  );
  assert(
    (await input.inputValue()).startsWith(`@${PLUGIN_PACKAGE_DISPLAY_NAME}`),
    "Claw composer 未写入 Plugin 显式触发前缀",
  );
  await page.screenshot({ path: screenshotPath, fullPage: true });
  await page.keyboard.press("Escape");
  await page.keyboard.press("Escape");

  await page.locator('[data-testid="send-btn"]').last().click();
  await observeAppServerMethodFromTrace(
    page,
    "turn/start",
    options,
    observedMethods,
  );
  const guiTurn = await waitForPluginPackageGuiTurn(
    page,
    runtime.threadId,
    options,
    observedMethods,
  );
  return {
    dynamicLifecycle: { statuses: [], snapshots: [] },
    guiModelSelection,
    guiTurn,
    turn: {
      method: "turn/start",
      result: { turn: { id: guiTurn.identity.turnId } },
      source: "renderer",
    },
    turnId: guiTurn.identity.turnId,
    userItemId: guiTurn.identity.userItemId,
  };
}

async function startRuntimeTurn(
  page,
  {
    fixture,
    observedMethods,
    options,
    pluginPackage = false,
    pluginMentionScreenshotPath,
    serverName,
    workspaceRoot,
  },
) {
  const expectedToolName = toolName(serverName);
  const route = await createRepositoryProvider(page, fixture, observedMethods);
  const start = await appServerCallFromPage(page, "thread/start", {
    cwd: workspaceRoot,
    historyMode: "paginated",
    model: route.model,
    modelProvider: route.providerId,
    runtimeWorkspaceRoots: [workspaceRoot],
    serviceName: "MCP elicitation Gate B",
    threadSource: "appServer",
  });
  observedMethods.add(start.method);
  const sessionId = String(start.result?.thread?.sessionId || "").trim();
  const threadId = String(start.result?.thread?.id || "").trim();
  assert(sessionId && threadId, "thread/start 未返回 canonical identity");

  const update = await appServerCallFromPage(page, "thread/settings/update", {
    threadId,
    modelProvider: route.providerId,
    model: route.model,
  });
  observedMethods.add(update.method);
  const gui = await openRuntimeThreadInGui(page, sessionId, options);
  if (pluginPackage) {
    const guiRuntime = await startPluginPackageTurnFromGui({
      page,
      runtime: { route, sessionId, threadId },
      options,
      observedMethods,
      screenshotPath: pluginMentionScreenshotPath,
    });
    return {
      ...guiRuntime,
      expectedToolName,
      gui,
      route,
      sessionId,
      start,
      threadId,
      update,
    };
  }
  const dynamicLifecyclePromise = probeDynamicToolLifecycle(
    page,
    threadId,
    options,
    observedMethods,
  );
  const turn = await appServerCallFromPage(page, "turn/start", {
    threadId,
    clientUserMessageId: `mcp-elicitation-${Date.now()}-${process.pid}`,
    input: [
      {
        type: "text",
        text: "Read the desktop app information, then run the release check through the available MCP tool.",
      },
    ],
    cwd: workspaceRoot,
    runtimeWorkspaceRoots: [workspaceRoot],
    model: route.model,
    approvalPolicy: "never",
    sandboxPolicy: "danger-full-access",
    additionalContext: {
      metadata: {
        kind: "application",
        value: JSON.stringify({
          harness: {
            source: "smoke:mcp-elicitation-gate-b",
            skip_mcp_prewarm: false,
          },
          tool_scope: {
            allowed_tools: [DYNAMIC_TOOL_NAME, expectedToolName],
          },
        }),
      },
    },
  });
  observedMethods.add(turn.method);
  const turnId = String(turn.result?.turn?.id || "").trim();
  assert(turnId, "turn/start 未返回 canonical turn.id");
  const dynamicLifecycle = await dynamicLifecyclePromise;
  return {
    dynamicLifecycle,
    expectedToolName,
    gui,
    route,
    sessionId,
    start,
    threadId,
    turn,
    turnId,
    update,
  };
}

async function waitForElicitationForm(page, options) {
  const layer = page.locator(
    '[data-testid="pending-interaction-layer"][data-interaction-kind="mcp_elicitation"]',
  );
  await layer.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 90_000),
  });
  const form = layer.locator('[data-testid="mcp-server-elicitation-form"]');
  await form.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 30_000),
  });
  await form.locator('input[type="checkbox"]').check();
  const checked = await form.locator('input[type="checkbox"]').isChecked();
  assert(checked, "Renderer MCP 表单未提交 confirmed=true");
  const rootDialogAbsent = (await page.getByRole("dialog").count()) === 0;
  assert(rootDialogAbsent, "MCP elicitation 不得恢复根部 Dialog");
  return { form, rootDialogAbsent };
}

async function submitElicitation(page, form, options) {
  const submit = form.getByRole("button", { name: /提交|Submit/ });
  await submit.click({ timeout: Math.min(options.timeoutMs, 30_000) });
  await form.waitFor({
    state: "hidden",
    timeout: Math.min(options.timeoutMs, 60_000),
  });
  return (
    (await page
      .locator('[data-testid="mcp-server-elicitation-form"]')
      .count()) === 0
  );
}

function providerRequestSummary(requests) {
  return requests.map((request, index) => {
    const serializedPrompt = JSON.stringify(request.body ?? {});
    return {
      index,
      path: request.path,
      pluginSkillContextSeen:
        serializedPrompt.includes(
          `<name>${PLUGIN_PACKAGE_SKILL_ID}</name>`,
        ) && serializedPrompt.includes(PLUGIN_PACKAGE_SKILL_BODY_MARKER),
      stream: request.body?.stream === true,
      toolNames: (request.body?.tools ?? [])
        .map((tool) => String(tool?.function?.name || tool?.name || "").trim())
        .filter(Boolean),
    };
  });
}

function dynamicToolItems(value, items = []) {
  if (Array.isArray(value)) {
    value.forEach((entry) => dynamicToolItems(entry, items));
    return items;
  }
  if (!value || typeof value !== "object") return items;
  if (
    value.type === "dynamicToolCall" &&
    value.namespace === "desktop" &&
    value.tool === "appInfo"
  ) {
    items.push(value);
  }
  Object.values(value).forEach((entry) => dynamicToolItems(entry, items));
  return items;
}

async function probeDynamicToolLifecycle(
  page,
  threadId,
  options,
  observedMethods,
) {
  const startedAt = Date.now();
  const statuses = new Set();
  const snapshots = [];
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 15_000)) {
    const read = await appServerCallFromPage(page, "thread/read", {
      threadId,
      includeTurns: true,
    });
    observedMethods.add(read.method);
    const items = dynamicToolItems(read.result);
    for (const item of items) {
      if (typeof item?.status === "string") statuses.add(item.status);
    }
    if (items.length > 0) {
      snapshots.push(sanitizeJson(items));
    }
    if (statuses.has("inProgress") && statuses.has("completed")) {
      return { statuses: Array.from(statuses), snapshots };
    }
    await sleep(10);
  }
  return { statuses: Array.from(statuses), snapshots };
}

function isExactEmptyObject(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length === 0
  );
}

function mcpInitializeCapabilityEvidence(
  ledger,
  { managementInitializeRequired = true } = {},
) {
  const accepted = ledger.find(
    (entry) =>
      entry?.type === "elicitation_result" &&
      entry?.action === "accept" &&
      entry?.content?.confirmed === true,
  );
  const runtimeInitialize = ledger.find(
    (entry) => entry?.type === "initialize" && entry?.pid === accepted?.pid,
  );
  const runtimeCapabilities = runtimeInitialize?.clientCapabilities;
  const runtimeCapabilityExact =
    runtimeCapabilities !== null &&
    typeof runtimeCapabilities === "object" &&
    !Array.isArray(runtimeCapabilities) &&
    Object.keys(runtimeCapabilities).length === 1 &&
    isExactEmptyObject(runtimeCapabilities.elicitation);
  const managementInitialize = ledger.find(
    (entry) =>
      entry?.type === "initialize" &&
      entry?.pid !== accepted?.pid &&
      entry?.clientCapabilities !== null &&
      typeof entry.clientCapabilities === "object" &&
      !Array.isArray(entry.clientCapabilities) &&
      !Object.prototype.hasOwnProperty.call(
        entry.clientCapabilities,
        "elicitation",
      ),
  );
  const capabilityMissingCount = ledger.filter(
    (entry) => entry?.type === "capability_missing",
  ).length;
  const managementConnectionAbsent = !ledger.some(
    (entry) => entry?.type === "initialize" && entry?.pid !== accepted?.pid,
  );
  return {
    acceptedPid: accepted?.pid ?? null,
    capabilityMissingCount,
    managementInitialize: managementInitialize ?? null,
    managementConnectionAbsent,
    managementElicitationCapabilityAbsent: managementInitializeRequired
      ? Boolean(managementInitialize)
      : managementConnectionAbsent,
    runtimeCapabilityExact,
    runtimeInitialize: runtimeInitialize ?? null,
    runtimeProtocolCurrent: runtimeInitialize?.protocolVersion === "2025-06-18",
  };
}

async function waitForCompletion(
  page,
  runtime,
  fixture,
  ledgerPath,
  options,
  observedMethods,
  { managementInitializeRequired = true } = {},
) {
  const startedAt = Date.now();
  let latestRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latestRead = await appServerCallFromPage(page, "thread/read", {
      threadId: runtime.threadId,
      includeTurns: true,
    });
    observedMethods.add(latestRead.method);
    const ledger = readJsonLines(ledgerPath);
    const capabilityEvidence = mcpInitializeCapabilityEvidence(ledger, {
      managementInitializeRequired,
    });
    const serialized = JSON.stringify(latestRead.result || {});
    if (
      fixture.requests.length >= 2 &&
      capabilityEvidence.runtimeProtocolCurrent &&
      capabilityEvidence.runtimeCapabilityExact &&
      capabilityEvidence.managementElicitationCapabilityAbsent &&
      capabilityEvidence.capabilityMissingCount === 0 &&
      serialized.includes(FINAL_TEXT)
    ) {
      return { capabilityEvidence, ledger, read: latestRead };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `MCP elicitation Gate B 未完成: provider=${fixture.requests.length} ledger=${JSON.stringify(readJsonLines(ledgerPath))} read=${JSON.stringify(latestRead?.result)}`,
  );
}

function electronEvidence(traceRaw, observedMethods, requiredMethods) {
  const trace = parseInvokeTraceRaw(traceRaw);
  const methods = Array.from(
    new Set([
      ...observedMethods,
      ...parseJsonRpcRequestsFromInvokeTrace(traceRaw).map(
        (item) => item.method,
      ),
    ]),
  );
  const commands = Array.from(
    new Set(trace.map((item) => item?.command).filter(Boolean)),
  );
  return {
    appServerHandleJsonLinesSeen: commands.includes(
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    ),
    legacyMcpCommandsSeen: LEGACY_MCP_COMMANDS.filter((command) =>
      commands.includes(command),
    ),
    mockFallbackHitCount: trace.filter(
      (entry) =>
        entry?.mock === true ||
        entry?.mockFallback === true ||
        (entry?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
          entry?.transport !== "electron-ipc"),
    ).length,
    missingRequiredMethods: requiredMethods.filter(
      (method) => !methods.includes(method),
    ),
    requestMethods: methods,
  };
}

export async function run({ pluginPackage = false } = {}) {
  const defaults = pluginPackage
    ? {
        ...DEFAULTS,
        evidenceDir: path.join(
          process.cwd(),
          ".lime",
          "qc",
          "gui-evidence",
          "plugin-package-electron-gate-b",
        ),
        prefix: "plugin-package-electron-gate-b",
      }
    : DEFAULTS;
  const options = parseArgs(process.argv.slice(2), defaults);
  const requiredMethods = pluginPackage
    ? PLUGIN_PACKAGE_REQUIRED_METHODS
    : REQUIRED_METHODS;
  ensureElectronFixtureBuild({
    logPrefix: LOG_PREFIX,
    rootDir: process.cwd(),
  });
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const rawPath = path.join(options.evidenceDir, `${options.prefix}-raw.json`);
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}.png`,
  );
  const appCenterScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-app-center-bundled.png`,
  );
  const installReviewScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-install-review.png`,
  );
  const pluginMentionScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-claw-plugin-mention.png`,
  );
  const uninstallScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-uninstall.png`,
  );
  const postUninstallHistoryScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-post-uninstall-history.png`,
  );
  const coldRestoreScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-cold-restore.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );
  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: pluginPackage,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const startedAt = new Date().toISOString();
  const summary = {
    ok: false,
    checkedAt: startedAt,
    startedAt,
    completedAt: null,
    appVersion: null,
    platform: null,
    arch: null,
    electronVersion: null,
    electronLaunchCount: 0,
    electronMainProcessPids: [],
    repositoryCommit: readRepositoryCommit(process.cwd()),
    backendMode: "runtime",
    proofLevel: "Gate B",
    scenario: pluginPackage ? "plugin-package-mcp-runtime" : "mcp-elicitation",
    pluginId: pluginPackage ? PLUGIN_PACKAGE_ID : null,
    pluginMarketplaceId: null,
    pluginSource: null,
    pluginVersion: null,
    pluginRuntimeServerName: pluginPackage ? pluginPackageServerName() : null,
    pluginContentDigest: null,
    standardManifestSeen: false,
    standardMcpConfigSeen: false,
    pluginSkillProjected: false,
    pluginSkillContextSeen: false,
    sessionId: null,
    threadId: null,
    turnId: null,
    userItemId: null,
    toolItemId: null,
    toolCallId: pluginPackage ? MCP_TOOL_CALL_ID : null,
    surfaceId: null,
    resourceUri: pluginPackage ? PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI : null,
    appCenterBundledListVisible: false,
    appCenterBundledIdentityStable: false,
    appCenterInstallReviewVisible: false,
    appCenterInstallCompleted: false,
    appCenterEnableToggleCompleted: false,
    disabledBoundaryCompleted: false,
    disabledNewSessionId: null,
    disabledNewThreadId: null,
    disabledNewTurnId: null,
    disabledPluginPickerBlocked: false,
    disabledProviderToolAbsent: false,
    disabledThreadPluginItemsAbsent: false,
    enabledNewBoundaryCapabilityRestored: false,
    nativeDirectoryDialogObserved: false,
    clawPluginPickerVisible: false,
    clawPluginBadgeVisible: false,
    clawFixtureModelSelected: false,
    clawStructuredMentionObserved: false,
    clawThreadTurnItemIdentityStable: false,
    pluginMentionPath: pluginPackage
      ? `plugin://${PLUGIN_PACKAGE_CONFIG_NAME}`
      : null,
    pluginMentionUserItemId: null,
    uninstallViaAppCenterCompleted: false,
    installedProjectionClearedAfterUninstall: false,
    historyReadableAfterUninstall: false,
    historyMcpRuntimeNotRestartedAfterUninstall: false,
    historyProviderNotReexecutedAfterUninstall: false,
    coldRestoreCompleted: false,
    coldRestoreInstalledProjectionRecovered: false,
    coldRestoreIdentityStable: false,
    coldRestoreExplicitSurfaceOpen: false,
    coldRestoreCanonicalHydrationViaSidebar: false,
    coldRestoreMcpProcessRestarted: false,
    coldRestoreProviderNotReexecuted: false,
    coldRestoreToolNotReexecuted: false,
    coldRestoreScreenshot: pluginPackage ? coldRestoreScreenshotPath : null,
    mcpAppHistoryUnavailableAfterUninstall: false,
    historyToolNotReexecutedAfterUninstall: false,
    productionMockFallbackHitCount: null,
    capabilityAdvertisementRequired: true,
    dynamicToolRequired: !pluginPackage,
    capabilityMissingCount: null,
    managementElicitationCapabilityAbsent: false,
    runtimeClientCapabilities: null,
    runtimeInitializeProtocolVersion: null,
    electronPreloadBridge: false,
    appServerHandleJsonLinesSeen: false,
    rendererFormVisible: false,
    rendererConfirmedSubmitted: false,
    formClosedAfterResolved: false,
    rootDialogAbsent: false,
    mcpLedgerAccepted: false,
    providerFinalTextObserved: false,
    mcpAppCanonicalIdentityStable: false,
    mcpAppHtmlLoadCount: 0,
    mcpAppResourceReadCount: 0,
    mcpAppResourceUri: pluginPackage ? PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI : null,
    mcpAppRestoredAfterReload: false,
    mcpAppRightSurfaceVisible: false,
    mcpAppToolCallCount: 0,
    mcpAppWebContentsMarker: null,
    dynamicToolProviderResultObserved: false,
    dynamicToolCanonicalCompleted: false,
    dynamicToolStartedObserved: false,
    dynamicToolRequestHiddenFromRenderer: false,
    providerRequestCount: 0,
    consoleErrors: [],
    missingRequiredMethods: [...requiredMethods],
    legacyMcpCommandsSeen: [],
    screenshot: null,
    appCenterScreenshot: pluginPackage ? appCenterScreenshotPath : null,
    installReviewScreenshot: pluginPackage ? installReviewScreenshotPath : null,
    pluginMentionScreenshot: pluginPackage ? pluginMentionScreenshotPath : null,
    uninstallScreenshot: pluginPackage ? uninstallScreenshotPath : null,
    postUninstallHistoryScreenshot: pluginPackage
      ? postUninstallHistoryScreenshotPath
      : null,
    summary: summaryPath,
    rawEvidence: rawPath,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
  };
  const raw = {};
  const consoleErrors = [];
  const observedMethods = new Set();
  let app = null;
  let page = null;
  let fixture = null;
  let disabledBoundaryFixture = null;
  let mcpFixture = null;
  let server = null;
  let plugin = null;
  let dynamicToolLifecycleTraceRaw = "";

  try {
    logStage("start-provider-fixture");
    const serverName = pluginPackage ? pluginPackageServerName() : makeServerName();
    const expectedToolName = toolName(serverName);
    fixture = await startOpenAiCompatibleFixtureServer({
      scriptedResponses: [
        ...(!pluginPackage
          ? [
              {
                type: "tool_call",
                id: DYNAMIC_TOOL_CALL_ID,
                name: DYNAMIC_TOOL_NAME,
                arguments: {},
              },
            ]
          : []),
        {
          type: "tool_call",
          id: MCP_TOOL_CALL_ID,
          name: expectedToolName,
          arguments: { release: "gate-b" },
        },
        { type: "text", content: FINAL_TEXT },
      ],
    });
    mcpFixture = writeElicitationFixture();
    if (pluginPackage) {
      plugin = preparePluginPackage(mcpFixture);
    }

    logStage("launch-electron");
    const handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      backendMode: "runtime",
    });
    app = handle.app;
    page = handle.page;
    const electronRuntime = await readElectronRuntime(app);
    summary.appVersion = electronRuntime.appVersion;
    summary.arch = electronRuntime.arch;
    summary.electronVersion = electronRuntime.electronVersion;
    summary.platform = electronRuntime.platform;
    summary.electronLaunchCount = 1;
    summary.electronMainProcessPids = [electronRuntime.pid];
    raw.electronRuntime = sanitizeJson(electronRuntime);
    summary.electronPreloadBridge =
      handle.rendererSnapshot.electron &&
      handle.rendererSnapshot.hasInvokeBridge;

    if (pluginPackage) {
      await installPluginPackageEmbeddedBrowserLifecycleCapture(page);
      logStage("open-plugin-catalog-app-center");
      const appCenterCatalog = await openPluginPackageAppCenter(
        page,
        options,
        observedMethods,
        appCenterScreenshotPath,
      );
      summary.appCenterBundledListVisible = true;
      summary.appCenterBundledIdentityStable = true;
      raw.appCenterCatalog = sanitizeJson(appCenterCatalog);

      logStage("install-plugin-catalog-from-app-center");
      const pluginLifecycle = await installPluginPackageFromAppCenter({
        app,
        page,
        plugin,
        options,
        observedMethods,
        screenshotPath: installReviewScreenshotPath,
      });
      summary.appCenterInstallReviewVisible = true;
      summary.nativeDirectoryDialogObserved = true;
      summary.appCenterInstallCompleted = true;
      const installedSummary = pluginLifecycle.read.result?.plugin?.summary;
      summary.pluginMarketplaceId = installedSummary?.marketplaceId ?? null;
      summary.pluginSource = installedSummary?.source ?? null;
      summary.pluginVersion = installedSummary?.version ?? null;
      summary.pluginContentDigest = installedSummary?.contentDigest ?? null;
      summary.standardManifestSeen = pluginLifecycle.standardManifestSeen;
      summary.standardMcpConfigSeen = pluginLifecycle.standardMcpConfigSeen;
      summary.pluginSkillProjected = pluginLifecycle.pluginSkillProjected;
      raw.pluginLifecycle = sanitizeJson(pluginLifecycle);

      logStage("toggle-plugin-catalog-from-app-center");
      const disabled = await setPluginPackageEnabledFromAppCenter(
        page,
        plugin.id,
        false,
        options,
        observedMethods,
      );
      logStage("verify-plugin-catalog-disabled-new-thread-boundary");
      disabledBoundaryFixture = await startOpenAiCompatibleFixtureServer({
        scriptedResponses: [
          { type: "text", content: DISABLED_BOUNDARY_FINAL_TEXT },
        ],
      });
      const disabledBoundary = await verifyPluginPackageDisabledNewThreadBoundary({
        page,
        fixture: disabledBoundaryFixture,
        plugin,
        options,
        observedMethods,
      });
      summary.disabledBoundaryCompleted = true;
      summary.disabledNewSessionId = disabledBoundary.sessionId;
      summary.disabledNewThreadId = disabledBoundary.threadId;
      summary.disabledNewTurnId = disabledBoundary.turnId;
      summary.disabledPluginPickerBlocked =
        disabledBoundary.pluginPickerBlocked;
      summary.disabledProviderToolAbsent = disabledBoundary.providerToolAbsent;
      summary.disabledThreadPluginItemsAbsent =
        disabledBoundary.threadPluginItemsAbsent;
      raw.pluginDisabledBoundary = sanitizeJson(disabledBoundary);

      const enabled = await setPluginPackageEnabledFromAppCenter(
        page,
        plugin.id,
        true,
        options,
        observedMethods,
      );
      summary.appCenterEnableToggleCompleted = true;
      raw.pluginEnableToggle = sanitizeJson({ disabled, enabled });
    } else {
      logStage("create-and-start-mcp-server");
      const serverId = `mcp-elicitation-${Date.now()}-${process.pid}`;
      const created = await appServerCallFromPage(page, "mcpServer/create", {
        server: {
          id: serverId,
          name: serverName,
          description: "MCP elicitation Gate B fixture",
          server_config: {
            command: process.execPath,
            args: [mcpFixture.serverPath, mcpFixture.ledgerPath],
            cwd: mcpFixture.root,
            timeout: 10,
            tool_timeout: 60,
          },
          enabled_lime: true,
          enabled_claude: false,
          enabled_codex: false,
          enabled_gemini: false,
          created_at: Date.now(),
        },
      });
      observedMethods.add(created.method);
      server = { id: serverId, name: serverName };
      raw.mcpServerCreate = sanitizeJson(created);
      const startedServer = await appServerCallFromPage(
        page,
        "mcpServer/start",
        { name: serverName },
      );
      observedMethods.add(startedServer.method);
      raw.mcpServerStart = sanitizeJson(startedServer);
      raw.mcpToolList = sanitizeJson(
        await waitForTool(page, expectedToolName, options, observedMethods),
      );
    }

    logStage("start-agent-turn");
    const workspace = await ensureWorkspace(page, observedMethods);
    raw.workspace = sanitizeJson(workspace.response);
    const runtime = await startRuntimeTurn(page, {
      fixture,
      observedMethods,
      options,
      pluginPackage,
      pluginMentionScreenshotPath,
      serverName,
      workspaceRoot: workspace.rootPath,
    });
    raw.runtime = sanitizeJson(runtime);
    summary.sessionId = runtime.sessionId ?? null;
    summary.threadId = runtime.threadId ?? null;
    summary.turnId = runtime.turnId ?? null;
    summary.userItemId = runtime.userItemId ?? null;
    raw.dynamicToolLifecycle = sanitizeJson(runtime.dynamicLifecycle);
    if (pluginPackage) {
      summary.clawPluginPickerVisible = true;
      summary.clawPluginBadgeVisible = true;
      summary.clawFixtureModelSelected =
        runtime.guiModelSelection?.providerId === runtime.route.providerId &&
        runtime.guiModelSelection?.model === runtime.route.model;
      summary.clawStructuredMentionObserved = true;
      summary.clawThreadTurnItemIdentityStable =
        runtime.guiTurn?.identity?.threadId === runtime.threadId &&
        runtime.guiTurn?.identity?.turnId === runtime.turnId &&
        Boolean(runtime.userItemId);
      summary.pluginMentionUserItemId = runtime.userItemId ?? null;
    }
    raw.providerRequestsBeforeElicitation = sanitizeJson(
      providerRequestSummary(fixture.requests),
    );
    raw.mcpLedgerBeforeElicitation = sanitizeJson(
      readJsonLines(mcpFixture.ledgerPath),
    );

    logStage("submit-renderer-form");
    const { form, rootDialogAbsent } = await waitForElicitationForm(
      page,
      options,
    );
    summary.rendererFormVisible = true;
    summary.rootDialogAbsent = rootDialogAbsent;
    dynamicToolLifecycleTraceRaw =
      (await page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      )) || "";
    raw.dynamicToolLifecycleTrace = sanitizeJson(
      parseInvokeTraceRaw(dynamicToolLifecycleTraceRaw),
    );
    await page.screenshot({ path: screenshotPath, fullPage: true });
    summary.screenshot = screenshotPath;
    summary.formClosedAfterResolved = await submitElicitation(
      page,
      form,
      options,
    );
    summary.rendererConfirmedSubmitted = true;

    logStage("wait-provider-final");
    const completion = await waitForCompletion(
      page,
      runtime,
      fixture,
      mcpFixture.ledgerPath,
      options,
      observedMethods,
      { managementInitializeRequired: !pluginPackage },
    );
    raw.completion = sanitizeJson(completion.read);
    raw.mcpLedger = sanitizeJson(completion.ledger);
    raw.mcpInitializeCapabilityEvidence = sanitizeJson(
      completion.capabilityEvidence,
    );
    summary.capabilityMissingCount =
      completion.capabilityEvidence.capabilityMissingCount;
    summary.managementElicitationCapabilityAbsent =
      completion.capabilityEvidence.managementElicitationCapabilityAbsent;
    summary.runtimeClientCapabilities =
      completion.capabilityEvidence.runtimeInitialize?.clientCapabilities ??
      null;
    summary.runtimeInitializeProtocolVersion =
      completion.capabilityEvidence.runtimeInitialize?.protocolVersion ?? null;
    const providerRequests = providerRequestSummary(fixture.requests);
    raw.providerRequests = sanitizeJson(providerRequests);
    if (pluginPackage) {
      summary.enabledNewBoundaryCapabilityRestored =
        providerRequests[0]?.toolNames?.includes(expectedToolName) === true;
      summary.pluginSkillContextSeen = providerRequests.some(
        (request) => request.pluginSkillContextSeen,
      );
    }
    summary.providerRequestCount = providerRequests.length;
    summary.dynamicToolProviderResultObserved = JSON.stringify(
      fixture.requests.slice(1),
    ).includes(DYNAMIC_TOOL_CALL_ID);
    summary.mcpLedgerAccepted = completion.ledger.some(
      (entry) =>
        entry?.action === "accept" && entry?.content?.confirmed === true,
    );
    summary.providerFinalTextObserved = JSON.stringify(
      completion.read.result,
    ).includes(FINAL_TEXT);
    const traceRaw = completion.read.traceRaw;
    const dynamicItems = dynamicToolItems(completion.read.result);
    summary.dynamicToolCanonicalCompleted =
      dynamicItems.length === 1 &&
      dynamicItems[0]?.status === "completed" &&
      dynamicItems[0]?.success === true;
    const combinedTraceRaw = `${dynamicToolLifecycleTraceRaw}\n${traceRaw}`;
    summary.dynamicToolStartedObserved =
      runtime.dynamicLifecycle.statuses.includes("inProgress") &&
      runtime.dynamicLifecycle.statuses.includes("completed");
    summary.dynamicToolRequestHiddenFromRenderer = !combinedTraceRaw.includes(
      '"method":"item/tool/call"',
    );
    raw.dynamicToolItems = sanitizeJson(dynamicItems);
    let evidence = electronEvidence(traceRaw, observedMethods, requiredMethods);
    summary.appServerHandleJsonLinesSeen =
      evidence.appServerHandleJsonLinesSeen;
    summary.missingRequiredMethods = evidence.missingRequiredMethods;
    summary.legacyMcpCommandsSeen = evidence.legacyMcpCommandsSeen;
    summary.productionMockFallbackHitCount = evidence.mockFallbackHitCount;
    raw.electronEvidence = sanitizeJson(evidence);
    summary.consoleErrors = [...consoleErrors];

    if (pluginPackage) {
      logStage("verify-plugin-catalog-mcp-app-surface");
      const completionMention = findPluginPackageMentionIdentity(
        completion.read.result,
      );
      assert(
        completionMention?.threadId === runtime.threadId &&
          completionMention?.turnId === runtime.turnId &&
          completionMention?.userItemId === runtime.userItemId,
        "Claw structured Plugin mention 未保持 Thread/Turn/Item identity",
      );
      const mcpAppItems = findPluginPackageMcpAppItems(completion.read.result);
      assert(
        mcpAppItems.length === 1,
        `thread/read 未观察到唯一 MCP App item: ${JSON.stringify(mcpAppItems)}`,
      );
      runtime.mcpAppItemId = String(mcpAppItems[0]?.id || "").trim();
      assert(runtime.mcpAppItemId, "MCP App item 缺少 canonical id");
      summary.toolItemId = runtime.mcpAppItemId;
      await openRuntimeThreadInGui(page, runtime.sessionId, options);
      const firstSurface = await waitForPluginPackageMcpAppSurface({
        app,
        page,
        options,
        runtime,
      });
      assert(
        firstSurface.traceEvidence.resourceReadCount === 2 &&
          firstSurface.traceEvidence.htmlLoadCount === 2,
        "首次显式恢复前后的 MCP App resource/HTML load 计数异常",
      );
      summary.mcpAppRightSurfaceVisible = true;
      summary.mcpAppWebContentsMarker = PLUGIN_PACKAGE_MCP_APP_MARKER;
      summary.surfaceId = firstSurface.viewId;
      raw.mcpAppFirstSurface = sanitizeJson(firstSurface);

      logStage("reload-plugin-catalog-mcp-app-surface");
      await openRuntimeThreadInGui(page, runtime.sessionId, options);
      const restoredRead = await appServerCallFromPage(page, "thread/read", {
        threadId: runtime.threadId,
        includeTurns: true,
      });
      observedMethods.add(restoredRead.method);
      const restoredItems = findPluginPackageMcpAppItems(restoredRead.result);
      assert(restoredItems.length === 1, "reload 后 MCP App item 数量不稳定");
      assert(
        restoredItems[0]?.id === runtime.mcpAppItemId &&
          restoredItems[0]?.mcpAppResourceUri ===
            PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI &&
          restoredItems[0]?.pluginId === PLUGIN_PACKAGE_ID,
        "reload 后 Plugin/item/resource identity 漂移",
      );
      const restoredSurface = await waitForPluginPackageMcpAppSurface({
        app,
        page,
        options,
        runtime,
      });
      assert(
        restoredSurface.traceEvidence.resourceReadCount === 3 &&
          restoredSurface.traceEvidence.htmlLoadCount === 3,
        "Renderer reload 后 MCP App resource/HTML load 计数异常",
      );
      summary.mcpAppRestoredAfterReload = true;
      summary.mcpAppCanonicalIdentityStable =
        restoredSurface.containerId === firstSurface.containerId &&
        restoredSurface.viewId === firstSurface.viewId;
      raw.mcpAppRestoredRead = sanitizeJson(restoredRead);
      raw.mcpAppRestoredSurface = sanitizeJson(restoredSurface);
      await page.screenshot({ path: screenshotPath, fullPage: true });

      const preColdRestoreLedger = readJsonLines(mcpFixture.ledgerPath);
      const preColdRestoreResourceReadCount = preColdRestoreLedger.filter(
        (entry) =>
          entry?.type === "resource_read" &&
          entry?.uri === PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI,
      ).length;
      const preColdRestoreToolCallCount = preColdRestoreLedger.filter(
        (entry) => entry?.type === "tool_call" && entry?.name === TOOL_SUFFIX,
      ).length;
      const preColdRestoreProviderRequestCount = fixture.requests.length;

      logStage("cold-restart-plugin-package-history");
      await closeElectronFixture({ app });
      app = null;
      page = null;
      const coldHandle = await launchElectronFixture({
        options,
        runtimeEnv,
        appServerEnv,
        consoleErrors,
        backendMode: "runtime",
      });
      app = coldHandle.app;
      page = coldHandle.page;
      await installPluginPackageEmbeddedBrowserLifecycleCapture(page);
      const coldElectronRuntime = await readElectronRuntime(app);
      summary.electronLaunchCount = 2;
      summary.electronMainProcessPids.push(coldElectronRuntime.pid);
      assert(
        new Set(summary.electronMainProcessPids).size === 2,
        "cold restore 未启动新的 Electron 主进程",
      );
      const coldInstalled = await waitForPluginInstalledState(
        page,
        plugin.id,
        true,
        options,
        observedMethods,
      );
      summary.coldRestoreInstalledProjectionRecovered = true;
      const coldNavigation = await switchRuntimeThreadInGuiViaSidebar(
        page,
        {
          sourceSessionId: summary.disabledNewSessionId,
          targetSessionId: runtime.sessionId,
        },
        options,
      );
      summary.coldRestoreCanonicalHydrationViaSidebar = true;
      raw.coldRestoreNavigation = sanitizeJson(coldNavigation);
      const coldRead = await appServerCallFromPage(page, "thread/read", {
        threadId: runtime.threadId,
        includeTurns: true,
      });
      observedMethods.add(coldRead.method);
      const coldMention = findPluginPackageMentionIdentity(coldRead.result);
      const coldItems = findPluginPackageMcpAppItems(coldRead.result);
      assert(
        coldMention?.threadId === runtime.threadId &&
          coldMention?.turnId === runtime.turnId &&
          coldMention?.userItemId === runtime.userItemId &&
          coldItems.length === 1 &&
          coldItems[0]?.id === runtime.mcpAppItemId &&
          coldItems[0]?.pluginId === PLUGIN_PACKAGE_ID,
        "cold restore 后 Plugin Thread/Turn/Item identity 漂移",
      );
      const coldSurfaceTab = page.locator(
        '[data-testid="workspace-right-surface-tab-appSurface"]',
      );
      await coldSurfaceTab.waitFor({
        state: "visible",
        timeout: options.timeoutMs,
      });
      await coldSurfaceTab.click();
      summary.coldRestoreExplicitSurfaceOpen = true;
      const coldSurface = await waitForPluginPackageMcpAppSurface({
        app,
        page,
        options,
        runtime,
      });
      assert(
        coldSurface.containerId === firstSurface.containerId &&
          coldSurface.viewId === firstSurface.viewId,
        "cold restore 后 Plugin surface identity 漂移",
      );
      const coldLedger = readJsonLines(mcpFixture.ledgerPath);
      const coldResourceReadCount = coldLedger.filter(
        (entry) =>
          entry?.type === "resource_read" &&
          entry?.uri === PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI,
      ).length;
      const coldToolCallCount = coldLedger.filter(
        (entry) => entry?.type === "tool_call" && entry?.name === TOOL_SUFFIX,
      ).length;
      const runtimeMcpProcessIds = Array.from(
        new Set(
          coldLedger
            .filter((entry) => entry?.type === "initialize")
            .map((entry) => entry?.pid)
            .filter((pid) => Number.isInteger(pid)),
        ),
      );
      assert(
        coldResourceReadCount - preColdRestoreResourceReadCount ===
          coldSurface.traceEvidence.resourceReadCount &&
          coldSurface.traceEvidence.resourceReadCount ===
            coldSurface.traceEvidence.htmlLoadCount,
        "cold restore 后 MCP resource/HTML load 未一一对应",
      );
      assert(
        runtimeMcpProcessIds.length === 2,
        `cold restore 未重启 Plugin MCP runtime process: ${JSON.stringify(runtimeMcpProcessIds)}`,
      );
      assert(
        fixture.requests.length === preColdRestoreProviderRequestCount &&
          coldToolCallCount === preColdRestoreToolCallCount &&
          coldToolCallCount === 1,
        "cold restore 偷偷重跑了 provider turn 或 MCP tool",
      );
      summary.coldRestoreCompleted = true;
      summary.coldRestoreIdentityStable = true;
      summary.coldRestoreMcpProcessRestarted = true;
      summary.coldRestoreProviderNotReexecuted = true;
      summary.coldRestoreToolNotReexecuted = true;
      summary.mcpAppCanonicalIdentityStable =
        summary.mcpAppCanonicalIdentityStable &&
        coldSurface.containerId === firstSurface.containerId &&
        coldSurface.viewId === firstSurface.viewId;
      raw.coldElectronRuntime = sanitizeJson(coldElectronRuntime);
      raw.coldRestoreInstalled = sanitizeJson(coldInstalled);
      raw.mcpAppColdRestoreRead = sanitizeJson(coldRead);
      raw.mcpAppColdRestoreSurface = sanitizeJson(coldSurface);
      raw.mcpAppColdRestoreLedger = sanitizeJson(coldLedger);
      await page.screenshot({
        path: coldRestoreScreenshotPath,
        fullPage: true,
      });

      logStage("uninstall-plugin-catalog-from-app-center");
      await uninstallPluginPackageFromAppCenter({
        page,
        pluginId: plugin.id,
        options,
        observedMethods,
        screenshotPath: uninstallScreenshotPath,
      });
      summary.uninstallViaAppCenterCompleted = true;
      summary.installedProjectionClearedAfterUninstall = true;

      logStage("restore-plugin-catalog-history-after-uninstall");
      await openRuntimeThreadInGui(page, runtime.sessionId, options);
      const postUninstallRead = await appServerCallFromPage(
        page,
        "thread/read",
        {
          threadId: runtime.threadId,
          includeTurns: true,
        },
      );
      observedMethods.add(postUninstallRead.method);
      const postUninstallMention = findPluginPackageMentionIdentity(
        postUninstallRead.result,
      );
      const postUninstallItems = findPluginPackageMcpAppItems(
        postUninstallRead.result,
      );
      assert(
        postUninstallMention?.threadId === runtime.threadId &&
          postUninstallMention?.turnId === runtime.turnId &&
          postUninstallMention?.userItemId === runtime.userItemId,
        "卸载后 Plugin mention 的 Thread/Turn/Item identity 漂移",
      );
      assert(
        postUninstallItems.length === 1 &&
          postUninstallItems[0]?.id === runtime.mcpAppItemId &&
          postUninstallItems[0]?.pluginId === PLUGIN_PACKAGE_ID,
        "卸载后 Plugin MCP 历史 item 不可读或 identity 漂移",
      );
      summary.historyReadableAfterUninstall = true;
      const postUninstallSurface =
        await waitForPluginPackageMcpAppHistoryUnavailable({
          app,
          page,
          options,
          runtime,
        });
      assert(
        postUninstallSurface.traceEvidence.resourceReadCount ===
          coldSurface.traceEvidence.resourceReadCount &&
          postUninstallSurface.traceEvidence.htmlLoadCount ===
            coldSurface.traceEvidence.htmlLoadCount,
        "卸载后历史 surface 不得重新读取 MCP resource 或加载 HTML",
      );
      summary.mcpAppHistoryUnavailableAfterUninstall = true;
      summary.mcpAppCanonicalIdentityStable =
        summary.mcpAppCanonicalIdentityStable &&
        postUninstallSurface.containerId === firstSurface.containerId &&
        postUninstallSurface.viewId === firstSurface.viewId;
      summary.mcpAppHtmlLoadCount =
        restoredSurface.traceEvidence.htmlLoadCount +
        coldSurface.traceEvidence.htmlLoadCount;
      const finalLedger = readJsonLines(mcpFixture.ledgerPath);
      summary.mcpAppResourceReadCount = finalLedger.filter(
        (entry) =>
          entry?.type === "resource_read" &&
          entry?.uri === PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI,
      ).length;
      summary.mcpAppToolCallCount = finalLedger.filter(
        (entry) => entry?.type === "tool_call" && entry?.name === TOOL_SUFFIX,
      ).length;
      summary.historyToolNotReexecutedAfterUninstall =
        summary.mcpAppToolCallCount === 1;
      const finalRuntimeMcpProcessIds = Array.from(
        new Set(
          finalLedger
            .filter((entry) => entry?.type === "initialize")
            .map((entry) => entry?.pid)
            .filter((pid) => Number.isInteger(pid)),
        ),
      );
      summary.historyMcpRuntimeNotRestartedAfterUninstall =
        finalRuntimeMcpProcessIds.length === runtimeMcpProcessIds.length;
      summary.historyProviderNotReexecutedAfterUninstall =
        fixture.requests.length === preColdRestoreProviderRequestCount;
      assert(
        summary.mcpAppResourceReadCount === summary.mcpAppHtmlLoadCount &&
          summary.mcpAppResourceReadCount === coldResourceReadCount,
        "跨进程与卸载后恢复的 MCP resource/HTML 累计计数异常",
      );
      assert(
        summary.historyMcpRuntimeNotRestartedAfterUninstall &&
          summary.historyProviderNotReexecutedAfterUninstall &&
          summary.historyToolNotReexecutedAfterUninstall,
        "卸载后查看历史不得重启 MCP runtime、provider 或 tool",
      );
      raw.mcpAppPostUninstallRead = sanitizeJson(postUninstallRead);
      raw.mcpAppPostUninstallSurface = sanitizeJson(postUninstallSurface);
      raw.mcpAppFinalLedger = sanitizeJson(finalLedger);
      await page.screenshot({
        path: postUninstallHistoryScreenshotPath,
        fullPage: true,
      });

      const finalTraceRaw =
        (await page.evaluate(() =>
          window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        )) || "";
      evidence = electronEvidence(
        finalTraceRaw,
        observedMethods,
        requiredMethods,
      );
      summary.appServerHandleJsonLinesSeen =
        evidence.appServerHandleJsonLinesSeen;
      summary.missingRequiredMethods = evidence.missingRequiredMethods;
      summary.legacyMcpCommandsSeen = evidence.legacyMcpCommandsSeen;
      summary.productionMockFallbackHitCount = evidence.mockFallbackHitCount;
      raw.electronEvidence = sanitizeJson(evidence);
    }

    assert(summary.electronPreloadBridge, "Electron preload bridge 未就绪");
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
      `观察到 legacy MCP facade: ${summary.legacyMcpCommandsSeen.join(", ")}`,
    );
    if (pluginPackage) {
      assert(
        summary.appVersion &&
          summary.platform &&
          summary.arch &&
          summary.electronVersion &&
          summary.repositoryCommit,
        "Gate B evidence 缺少应用版本、平台、Electron 或 commit",
      );
      assert(
        summary.electronLaunchCount === 2 &&
          summary.electronMainProcessPids.length === 2 &&
          summary.coldRestoreCompleted &&
          summary.coldRestoreInstalledProjectionRecovered &&
          summary.coldRestoreIdentityStable &&
          summary.coldRestoreExplicitSurfaceOpen &&
          summary.coldRestoreCanonicalHydrationViaSidebar &&
          summary.coldRestoreMcpProcessRestarted &&
          summary.coldRestoreProviderNotReexecuted &&
          summary.coldRestoreToolNotReexecuted,
        "Plugin cold restore Gate B 未通过",
      );
      assert(
        summary.pluginId === PLUGIN_PACKAGE_ID &&
          summary.pluginMarketplaceId === "local" &&
          summary.pluginSource === "local" &&
          summary.pluginVersion === plugin.version &&
          String(summary.pluginContentDigest || "").startsWith("sha256:"),
        "Gate B evidence 缺少稳定 Plugin source/version/digest identity",
      );
      assert(
        summary.standardManifestSeen &&
          summary.standardMcpConfigSeen &&
          summary.pluginSkillProjected &&
          summary.pluginSkillContextSeen,
        "Gate B 未证明标准 plugin.json、mcp.json 与 Skill provider context",
      );
      assert(
        summary.sessionId === runtime.sessionId &&
          summary.threadId === runtime.threadId &&
          summary.turnId === runtime.turnId &&
          summary.userItemId === runtime.userItemId &&
          summary.toolItemId === runtime.mcpAppItemId &&
          summary.toolCallId === MCP_TOOL_CALL_ID &&
          summary.surfaceId &&
          summary.resourceUri === PLUGIN_PACKAGE_MCP_APP_RESOURCE_URI,
        "Gate B evidence 缺少 Thread/Turn/Item/tool/surface identity",
      );
      assert(
        summary.appCenterBundledListVisible &&
          summary.appCenterBundledIdentityStable,
        "App Center bundled Plugin 列表或 identity 未通过",
      );
      assert(
        summary.nativeDirectoryDialogObserved &&
          summary.appCenterInstallReviewVisible &&
          summary.appCenterInstallCompleted &&
          summary.appCenterEnableToggleCompleted,
        "App Center 目录安装/启停用户流程未通过",
      );
      assert(
        summary.disabledBoundaryCompleted &&
          summary.disabledNewSessionId &&
          summary.disabledNewThreadId &&
          summary.disabledNewTurnId &&
          summary.disabledPluginPickerBlocked &&
          summary.disabledProviderToolAbsent &&
          summary.disabledThreadPluginItemsAbsent &&
          summary.enabledNewBoundaryCapabilityRestored,
        "Plugin 禁用新 Thread 隔离或重新启用边界未通过",
      );
      assert(
        summary.clawPluginPickerVisible &&
          summary.clawPluginBadgeVisible &&
          summary.clawFixtureModelSelected &&
          summary.clawStructuredMentionObserved &&
          summary.clawThreadTurnItemIdentityStable,
        "Claw Plugin picker/mention identity 未通过",
      );
      assert(
        summary.uninstallViaAppCenterCompleted &&
          summary.installedProjectionClearedAfterUninstall &&
          summary.historyReadableAfterUninstall &&
          summary.mcpAppHistoryUnavailableAfterUninstall &&
          summary.historyMcpRuntimeNotRestartedAfterUninstall &&
          summary.historyProviderNotReexecutedAfterUninstall &&
          summary.historyToolNotReexecutedAfterUninstall,
        "App Center 卸载或卸载后历史恢复未通过",
      );
      assert(
        !evidence.requestMethods.includes("mcpServer/create") &&
          !evidence.requestMethods.includes("mcpServer/start"),
        "Plugin package MCP 不得通过管理面 mcpServer/create/start 注入",
      );
      assert(
        completion.capabilityEvidence.managementConnectionAbsent,
        "Plugin package MCP Gate B 不得额外启动 management MCP connection",
      );
      assert(
        summary.mcpAppRightSurfaceVisible,
        "Plugin MCP App 右侧面板不可见",
      );
      assert(
        summary.mcpAppRestoredAfterReload,
        "Plugin MCP App reload 后未恢复",
      );
      assert(
        summary.mcpAppCanonicalIdentityStable,
        "Plugin MCP App canonical identity 不稳定",
      );
      assert(
        summary.mcpAppResourceReadCount >= 4,
        `Plugin MCP App resource read 次数异常: ${summary.mcpAppResourceReadCount}`,
      );
      assert(
        summary.mcpAppHtmlLoadCount === summary.mcpAppResourceReadCount,
        `Plugin MCP App HTML load 次数异常: ${summary.mcpAppHtmlLoadCount}`,
      );
      assert(
        summary.mcpAppToolCallCount === 1,
        `reload 不得重复执行旧 MCP tool action: ${summary.mcpAppToolCallCount}`,
      );
    }
    assert(
      providerRequests[0]?.toolNames?.includes(expectedToolName),
      `首个 provider request 未携带 scoped MCP tool: ${expectedToolName}`,
    );
    assert(
      providerRequests.length >= (pluginPackage ? 2 : 3),
      pluginPackage
        ? "provider 未完成 Plugin MCP tool result 后的后续请求"
        : "provider 未完成 dynamic tool 与 MCP tool result 后的后续请求",
    );
    if (!pluginPackage) {
      assert(
        providerRequests[0]?.toolNames?.includes(DYNAMIC_TOOL_NAME),
        `首个 provider request 未携带 Electron dynamic tool: ${DYNAMIC_TOOL_NAME}`,
      );
      assert(
        summary.dynamicToolProviderResultObserved,
        "dynamic tool response 未进入 provider 后续请求",
      );
      assert(
        summary.dynamicToolCanonicalCompleted,
        "thread/read 未观察到唯一 completed desktop/appInfo DynamicToolCall",
      );
      assert(
        summary.dynamicToolStartedObserved,
        "未观察到 desktop/appInfo DynamicToolCall inProgress 投影",
      );
      assert(
        summary.dynamicToolRequestHiddenFromRenderer,
        "item/tool/call reverse request 泄露到 Renderer",
      );
    }
    assert(
      summary.rendererConfirmedSubmitted,
      "Renderer 未提交 confirmed=true",
    );
    assert(
      summary.mcpLedgerAccepted,
      "MCP fixture 未收到 accept confirmed=true",
    );
    assert(
      summary.providerFinalTextObserved,
      "provider final text 未进入 current read model",
    );
    assert(
      summary.formClosedAfterResolved,
      "serverRequest/resolved 后 Composer 表单未关闭",
    );
    assert(summary.rootDialogAbsent, "MCP elicitation 出现重复根部 Dialog");
    assert(
      completion.capabilityEvidence.runtimeProtocolCurrent &&
        completion.capabilityEvidence.runtimeCapabilityExact,
      `runtime MCP initialize capability 非 current shape: ${JSON.stringify(summary.runtimeClientCapabilities)}`,
    );
    assert(
      summary.managementElicitationCapabilityAbsent,
      "management MCP initialize 不得广告 elicitation capability",
    );
    assert(
      summary.capabilityMissingCount === 0,
      `存在未广告 capability 的 runtime tool call: ${summary.capabilityMissingCount}`,
    );
    assert(
      consoleErrors.length === 0,
      `Renderer console error: ${JSON.stringify(consoleErrors)}`,
    );
    assert(
      summary.productionMockFallbackHitCount === 0,
      "观察到 production mock fallback 命中",
    );
    summary.ok = true;
  } catch (error) {
    summary.error = sanitizeText(
      error instanceof Error ? error.stack || error.message : String(error),
    );
    summary.consoleErrors = [...consoleErrors];
    if (page) {
      await page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
      summary.failureScreenshot = failureScreenshotPath;
    }
    throw error;
  } finally {
    summary.completedAt = new Date().toISOString();
    writeJsonFile(rawPath, raw);
    writeJsonFile(summaryPath, summary);
    if (pluginPackage) {
      await cleanupPluginPackage(page, plugin);
    } else {
      await cleanupMcpServer(page, server);
    }
    await closeElectronFixture({ app });
    if (fixture) await fixture.close().catch(() => undefined);
    if (disabledBoundaryFixture) {
      await disabledBoundaryFixture.close().catch(() => undefined);
    }
    if (mcpFixture && !options.keepTemp) {
      fs.rmSync(mcpFixture.root, { recursive: true, force: true });
    }
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
  console.log(`${LOG_PREFIX} pass summary=${summaryPath}`);
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  run().catch((error) => {
    console.error(
      `${LOG_PREFIX} failed: ${error instanceof Error ? error.message : String(error)}`,
    );
    process.exitCode = 1;
  });
}
