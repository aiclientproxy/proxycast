#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { ensureElectronFixtureBuild } from "../lib/electron-fixture-build.mjs";
import { startOpenAiCompatibleFixtureServer } from "../lib/openai-compatible-fixture-server.mjs";
import {
  sanitizeJson,
  writeJsonFile,
} from "../mcp/lib/current-smoke-transport.mjs";
import {
  appServerCallFromPage,
  assert,
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
  readConfigFromPage,
  sanitizeText,
  sleep,
  updateConfigFromPage,
} from "./mcp-config-fixture-smoke.mjs";
import {
  APPS_SERVER_NAME,
  APPS_TOOL_NAME,
  collectElectronEvidence,
  ORDINARY_SERVER_NAME,
  ORDINARY_TOOL_NAME,
  parseOrchestratorGateArgs,
  SKILL_READ_TOOL_NAME,
  SKILL_RESOURCE_URI,
  SKILL_SEARCH_TOOL_NAME,
  summarizeMcpLedger,
  summarizeProviderRequests,
  summarizeReadModel,
  writeOrchestratorMcpFixture,
} from "./lib/orchestrator-skills-gate-b-core.mjs";

const LOG_PREFIX = "[smoke:orchestrator-skills-gate-b]";
const ENABLED_FINAL_TEXT = "ORCHESTRATOR_SKILL_GATE_B_DONE";
const DISABLED_FINAL_TEXT = "ORCHESTRATOR_MCP_DISABLED_BOUNDARY_DONE";
const SEARCH_CALL_ID = "call-orchestrator-skill-search";
const READ_CALL_ID = "call-orchestrator-skill-read";
const ORDINARY_CALL_ID = "call-ordinary-mcp-boundary";
const NAVIGATION_RESTORE_STORAGE_KEY = "lime.appNavigation.restore.v1";

function printHelp() {
  console.log(`
Orchestrator Skills/MCP Electron Gate B

用途:
  通过真实 Electron/preload/App Server/runtime/provider/read model 验证远端 Skill
  discovery、skill_search、read_mcp_resource 与 GUI 最终文本；随后关闭
  orchestrator.mcp，证明 codex_apps 工具隐藏而普通 MCP 工具仍可执行。

边界:
  使用 localhost OpenAI-compatible provider 与临时 stdio MCP fixtures；不使用
  App Server mock backend、renderer mock fallback、legacy MCP facade 或正式模型。

选项:
  --evidence-dir <path> --prefix <name> --timeout-ms <ms>
  --interval-ms <ms> --keep-temp -h|--help
`);
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

async function readElectronRuntime(app) {
  return await app.evaluate(async ({ app: electronApp }) => ({
    appVersion: electronApp.getVersion(),
    electronVersion: process.versions.electron,
    platform: process.platform,
    arch: process.arch,
    pid: process.pid,
  }));
}

async function createAndStartMcpServer(
  page,
  fixture,
  definition,
  observedMethods,
) {
  const id = `orchestrator-gate-b-${definition.role}-${Date.now()}-${process.pid}`;
  const created = await appServerCallFromPage(page, "mcpServer/create", {
    server: {
      id,
      name: definition.name,
      description: `Orchestrator Gate B ${definition.role} fixture`,
      server_config: {
        command: process.execPath,
        args: [fixture.serverPath, fixture.ledgerPath, definition.role],
        cwd: fixture.root,
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
  const started = await appServerCallFromPage(page, "mcpServer/start", {
    name: definition.name,
  });
  observedMethods.add(started.method);
  return { id, name: definition.name };
}

async function waitForMcpTools(page, expectedNames, options, observedMethods) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latest = await appServerCallFromPage(page, "mcpTool/list", {});
    observedMethods.add(latest.method);
    const names = (latest.result?.tools ?? []).map((tool) => tool?.name);
    if (expectedNames.every((name) => names.includes(name))) return names;
    await sleep(options.intervalMs);
  }
  throw new Error(
    `MCP tools 未就绪: ${expectedNames.join(", ")}; latest=${JSON.stringify(latest?.result)}`,
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
  assert(workspaceId && rootPath, "workspace/default/ensure 未返回 identity");
  return { rootPath, workspaceId };
}

async function createRepositoryProvider(page, fixture, label, observedMethods) {
  const providerName = `${label} ${Date.now()}`;
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
    alias: "orchestrator-skills-gate-b",
    replaceExisting: true,
  });
  observedMethods.add(key.method);
  assert(key.result?.key?.id, "modelProviderKey/create 未返回 key.id");

  const catalog = await appServerCallFromPage(page, "model/list", {
    includeHidden: true,
    limit: 500,
  });
  observedMethods.add(catalog.method);
  const model = (catalog.result?.data ?? []).find(
    (candidate) =>
      candidate?.providerId === providerId &&
      candidate?.model === fixture.provider.modelPreference,
  );
  assert(
    model?.capabilitySnapshot?.runtimeFeatures?.includes("tool_calling"),
    "fixture route 缺少 tool_calling capability",
  );
  return { model: fixture.provider.modelPreference, providerId, providerName };
}

async function createRuntimeThread(
  page,
  workspace,
  route,
  label,
  observedMethods,
) {
  const start = await appServerCallFromPage(page, "thread/start", {
    cwd: workspace.rootPath,
    historyMode: "paginated",
    model: route.model,
    modelProvider: route.providerId,
    runtimeWorkspaceRoots: [workspace.rootPath],
    serviceName: label,
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
  return { route, sessionId, threadId };
}

async function openRuntimeThreadInGui(page, runtime, options) {
  await page.evaluate(
    ({ navigationKey, sessionId }) => {
      sessionStorage.setItem(
        navigationKey,
        JSON.stringify({
          page: "agent",
          params: { initialSessionId: sessionId },
        }),
      );
    },
    {
      navigationKey: NAVIGATION_RESTORE_STORAGE_KEY,
      sessionId: runtime.sessionId,
    },
  );
  await page.reload({
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${runtime.sessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: options.timeoutMs });
  assert(
    (await input.getAttribute("data-session-id")) === runtime.sessionId,
    "GUI session identity 漂移",
  );
}

async function startRuntimeTurn(
  page,
  runtime,
  workspace,
  prompt,
  allowedTools,
  observedMethods,
) {
  const turn = await appServerCallFromPage(page, "turn/start", {
    threadId: runtime.threadId,
    clientUserMessageId: `orchestrator-gate-b-${Date.now()}-${process.pid}`,
    input: [{ type: "text", text: prompt }],
    cwd: workspace.rootPath,
    runtimeWorkspaceRoots: [workspace.rootPath],
    model: runtime.route.model,
    approvalPolicy: "never",
    sandboxPolicy: "danger-full-access",
    additionalContext: {
      metadata: {
        kind: "application",
        value: JSON.stringify({
          harness: { source: "smoke:orchestrator-skills-gate-b" },
          tool_scope: { allowed_tools: allowedTools },
        }),
      },
    },
  });
  observedMethods.add(turn.method);
  const turnId = String(turn.result?.turn?.id || "").trim();
  assert(turnId, "turn/start 未返回 canonical turn.id");
  return { ...runtime, prompt, turnId };
}

async function waitForTurnCompletion(
  page,
  runtime,
  finalText,
  fixture,
  options,
  observedMethods,
) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latest = await appServerCallFromPage(page, "thread/read", {
      threadId: runtime.threadId,
      includeTurns: true,
    });
    observedMethods.add(latest.method);
    const serialized = JSON.stringify(latest.result || {});
    if (fixture.requests.length > 0 && serialized.includes(finalText)) break;
    await sleep(options.intervalMs);
  }
  const serialized = JSON.stringify(latest?.result || {});
  assert(
    serialized.includes(finalText),
    `${finalText} 未进入 canonical read model`,
  );
  assert(
    serialized.includes(runtime.turnId),
    "canonical read model 缺少 turn identity",
  );
  assert(
    serialized.includes(runtime.prompt),
    "canonical read model 缺少 user item",
  );
  await openRuntimeThreadInGui(page, runtime, options);
  await page.getByText(finalText, { exact: false }).last().waitFor({
    state: "visible",
    timeout: options.timeoutMs,
  });
  return { ...latest, guiFinalVisible: true };
}

async function cleanupServer(page, server) {
  if (!page || !server) return;
  await appServerCallFromPage(page, "mcpServer/stop", {
    name: server.name,
  }).catch(() => undefined);
  await appServerCallFromPage(page, "mcpServer/delete", {
    id: server.id,
  }).catch(() => undefined);
}

export async function run() {
  const options = parseOrchestratorGateArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  ensureElectronFixtureBuild({ logPrefix: LOG_PREFIX, rootDir: process.cwd() });
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const rawPath = path.join(options.evidenceDir, `${options.prefix}-raw.json`);
  const enabledScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-enabled.png`,
  );
  const disabledScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-disabled.png`,
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );
  const runtimeEnv = createTempRuntimeEnv();
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: {
      ...runtimeEnv.env,
      APP_SERVER_BIN: resolveDevAppServerBinary({
        env: runtimeEnv.env,
        repoRoot: process.cwd(),
        forceBuild: false,
      }),
    },
  });
  const summary = {
    schemaVersion: 1,
    scenarioId: "ORCHESTRATOR-01-remote-skill-and-mcp-boundary",
    proofLevel: "Gate B",
    claimBoundary:
      "Real Electron/preload/IPC/App Server RuntimeCore provider turn proves session-owned codex_apps Skill discovery and bounded read, then config/batchWrite disables only the codex_apps MCP tool catalog while an ordinary MCP tool remains executable. Local deterministic fixtures do not claim live provider or remote network behavior.",
    startedAt: new Date().toISOString(),
    completedAt: null,
    result: "fail",
    backendMode: "runtime",
    electron: null,
    config: null,
    enabled: null,
    disabledBoundary: null,
    mcp: null,
    bridge: null,
    errors: null,
    artifacts: {
      enabledScreenshot: enabledScreenshotPath,
      disabledScreenshot: disabledScreenshotPath,
      raw: rawPath,
      summary: summaryPath,
    },
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
  };
  const raw = {};
  const observedMethods = new Set();
  const consoleErrors = [];
  const pageErrors = [];
  let handle = null;
  let page = null;
  let enabledProvider = null;
  let disabledProvider = null;
  let appsServer = null;
  let ordinaryServer = null;

  try {
    logStage("start-local-fixtures");
    enabledProvider = await startOpenAiCompatibleFixtureServer({
      scriptedResponses: [
        {
          type: "tool_call",
          id: SEARCH_CALL_ID,
          name: SKILL_SEARCH_TOOL_NAME,
          arguments: { query: "release notes", limit: 3 },
        },
        {
          type: "tool_call",
          id: READ_CALL_ID,
          name: SKILL_READ_TOOL_NAME,
          arguments: { server: APPS_SERVER_NAME, uri: SKILL_RESOURCE_URI },
        },
        { type: "text", content: ENABLED_FINAL_TEXT },
      ],
    });
    disabledProvider = await startOpenAiCompatibleFixtureServer({
      scriptedResponses: [
        {
          type: "tool_call",
          id: ORDINARY_CALL_ID,
          name: ORDINARY_TOOL_NAME,
          arguments: { message: "disabled-boundary" },
        },
        { type: "text", content: DISABLED_FINAL_TEXT },
      ],
    });
    const mcpFixture = writeOrchestratorMcpFixture(runtimeEnv.tempRoot);

    logStage("launch-real-electron-runtime");
    handle = await launchElectronFixture({
      options,
      runtimeEnv,
      appServerEnv,
      consoleErrors,
      pageErrors,
      backendMode: "runtime",
    });
    page = handle.page;
    summary.electron = await readElectronRuntime(handle.app);
    assert(handle.rendererSnapshot.electron, "真实 Electron renderer 未就绪");
    assert(
      handle.rendererSnapshot.hasInvokeBridge,
      "Electron preload invoke 未就绪",
    );

    logStage("verify-default-config-and-start-mcp");
    const initialConfig = await readConfigFromPage(page);
    observedMethods.add(initialConfig.response.method);
    const defaultSkillsEnabled =
      initialConfig.config.orchestrator?.skills?.enabled ?? true;
    const defaultMcpEnabled =
      initialConfig.config.orchestrator?.mcp?.enabled ?? true;
    assert(
      defaultSkillsEnabled && defaultMcpEnabled,
      "Orchestrator 默认开关未启用",
    );
    appsServer = await createAndStartMcpServer(
      page,
      mcpFixture,
      { name: APPS_SERVER_NAME, role: "apps" },
      observedMethods,
    );
    ordinaryServer = await createAndStartMcpServer(
      page,
      mcpFixture,
      { name: ORDINARY_SERVER_NAME, role: "ordinary" },
      observedMethods,
    );
    await waitForMcpTools(
      page,
      [APPS_TOOL_NAME, ORDINARY_TOOL_NAME],
      options,
      observedMethods,
    );

    const workspace = await ensureWorkspace(page, observedMethods);
    const enabledRoute = await createRepositoryProvider(
      page,
      enabledProvider,
      "Orchestrator Skills Gate B",
      observedMethods,
    );
    let enabledRuntime = await createRuntimeThread(
      page,
      workspace,
      enabledRoute,
      "Orchestrator remote Skill Gate B",
      observedMethods,
    );
    await openRuntimeThreadInGui(page, enabledRuntime, options);

    logStage("run-remote-skill-turn");
    enabledRuntime = await startRuntimeTurn(
      page,
      enabledRuntime,
      workspace,
      "Use the delivery release notes Skill. Search its metadata, read its SKILL.md resource, then confirm completion.",
      [
        SKILL_SEARCH_TOOL_NAME,
        SKILL_READ_TOOL_NAME,
        APPS_TOOL_NAME,
        ORDINARY_TOOL_NAME,
      ],
      observedMethods,
    );
    const enabledRead = await waitForTurnCompletion(
      page,
      enabledRuntime,
      ENABLED_FINAL_TEXT,
      enabledProvider,
      options,
      observedMethods,
    );
    await page.screenshot({ path: enabledScreenshotPath, fullPage: true });
    const enabledRequests = summarizeProviderRequests(enabledProvider.requests);
    assert(enabledRequests.length >= 3, "远端 Skill provider sequence 未完成");
    assert(
      enabledRequests[0].toolNames.includes(SKILL_SEARCH_TOOL_NAME),
      "首步缺少 skill_search",
    );
    assert(
      enabledRequests[0].toolNames.includes(APPS_TOOL_NAME),
      "启用时 codex_apps tool 未暴露",
    );
    assert(
      enabledRequests[0].toolNames.includes(ORDINARY_TOOL_NAME),
      "启用时普通 MCP tool 未暴露",
    );
    assert(
      enabledRequests[0].hasSkillPackage,
      "Orchestrator Skill metadata 未进入 turn snapshot",
    );
    assert(
      !enabledRequests[0].hasSkillBody,
      "首次 provider 请求提前读取了 SKILL.md body",
    );
    assert(
      enabledRequests.at(-1).hasSkillBody,
      "read_mcp_resource 结果未进入 provider history",
    );
    const enabledReadModel = summarizeReadModel(
      enabledRead,
      enabledRuntime,
      [SEARCH_CALL_ID, READ_CALL_ID],
      ENABLED_FINAL_TEXT,
    );
    assert(
      Object.values(enabledReadModel).every(Boolean),
      `远端 Skill read model 不完整: ${JSON.stringify(enabledReadModel)}`,
    );

    logStage("disable-apps-mcp-through-current-config-control-plane");
    const configWrite = await updateConfigFromPage(page, () => ({
      orchestrator: { skills: { enabled: true }, mcp: { enabled: false } },
    }));
    observedMethods.add(configWrite.write?.method);
    const disabledConfig = await readConfigFromPage(page);
    observedMethods.add(disabledConfig.response.method);
    assert(
      disabledConfig.config.orchestrator?.mcp?.enabled === false &&
        disabledConfig.config.orchestrator?.skills?.enabled === true,
      "config/read 未读回 Orchestrator disabled boundary",
    );

    const disabledRoute = await createRepositoryProvider(
      page,
      disabledProvider,
      "Orchestrator MCP Disabled Boundary",
      observedMethods,
    );
    let disabledRuntime = await createRuntimeThread(
      page,
      workspace,
      disabledRoute,
      "Orchestrator MCP disabled boundary",
      observedMethods,
    );
    await openRuntimeThreadInGui(page, disabledRuntime, options);

    logStage("prove-ordinary-mcp-survives-disabled-boundary");
    disabledRuntime = await startRuntimeTurn(
      page,
      disabledRuntime,
      workspace,
      "Call the ordinary MCP boundary probe and confirm completion.",
      [
        SKILL_SEARCH_TOOL_NAME,
        SKILL_READ_TOOL_NAME,
        APPS_TOOL_NAME,
        ORDINARY_TOOL_NAME,
      ],
      observedMethods,
    );
    const disabledRead = await waitForTurnCompletion(
      page,
      disabledRuntime,
      DISABLED_FINAL_TEXT,
      disabledProvider,
      options,
      observedMethods,
    );
    await page.screenshot({ path: disabledScreenshotPath, fullPage: true });
    const disabledRequests = summarizeProviderRequests(
      disabledProvider.requests,
    );
    assert(
      disabledRequests.length >= 2,
      "disabled boundary provider sequence 未完成",
    );
    assert(
      disabledRequests[0].toolNames.includes(ORDINARY_TOOL_NAME),
      "orchestrator.mcp=false 错误隐藏了普通 MCP tool",
    );
    assert(
      !disabledRequests[0].toolNames.includes(APPS_TOOL_NAME),
      "orchestrator.mcp=false 未隐藏 codex_apps tool",
    );
    const disabledReadModel = summarizeReadModel(
      disabledRead,
      disabledRuntime,
      [ORDINARY_CALL_ID],
      DISABLED_FINAL_TEXT,
    );
    assert(
      Object.values(disabledReadModel).every(Boolean),
      `disabled boundary read model 不完整: ${JSON.stringify(disabledReadModel)}`,
    );

    const ledgerSummary = summarizeMcpLedger(mcpFixture.ledgerPath);
    assert(
      ledgerSummary.runtimePidObserved,
      "未观察到 session-owned resource read process",
    );
    assert(
      ledgerSummary.frozenTurnResourceListCount === 1,
      `同一 Turn 重复发现 Orchestrator Skill: ${ledgerSummary.frozenTurnResourceListCount}`,
    );
    assert(
      ledgerSummary.exactResourceReadCount === 1,
      "SKILL.md 未被精确读取一次",
    );
    assert(
      ledgerSummary.appsToolCallCount === 0,
      "Gate B 不应调用 codex_apps probe tool",
    );
    assert(
      ledgerSummary.ordinaryToolCallCount === 1,
      "普通 MCP tool 未执行一次",
    );

    const bridge = await collectElectronEvidence(page, observedMethods);
    assert(
      bridge.appServerHandleJsonLinesSeen && bridge.electronIpcSeen,
      "Electron current bridge 证据缺失",
    );
    assert(
      bridge.missingRequiredMethods.length === 0,
      `缺少 current methods: ${bridge.missingRequiredMethods.join(", ")}`,
    );
    assert(
      bridge.legacyMcpCommandsSeen.length === 0,
      "观察到 legacy MCP facade",
    );
    assert(
      bridge.mockFallbackHitCount === 0,
      "观察到 production mock fallback",
    );
    assert(bridge.invokeErrorCount === 0, "观察到 renderer invoke error");
    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );
    assert(
      pageErrors.length === 0,
      `观察到 page error: ${pageErrors.join(" | ")}`,
    );

    summary.config = {
      defaultSkillsEnabled,
      defaultMcpEnabled,
      disabledSkillsEnabled: disabledConfig.config.orchestrator.skills.enabled,
      disabledMcpEnabled: disabledConfig.config.orchestrator.mcp.enabled,
      currentControlPlaneWrite: configWrite.write?.result?.status === "ok",
    };
    summary.enabled = {
      providerRequests: enabledRequests,
      readModel: enabledReadModel,
    };
    summary.disabledBoundary = {
      providerRequests: disabledRequests,
      readModel: disabledReadModel,
      appsToolHidden: !disabledRequests[0].toolNames.includes(APPS_TOOL_NAME),
      ordinaryToolAvailable:
        disabledRequests[0].toolNames.includes(ORDINARY_TOOL_NAME),
    };
    summary.mcp = ledgerSummary;
    summary.bridge = bridge;
    summary.errors = {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
      invokeErrorCount: bridge.invokeErrorCount,
      mockFallbackHitCount: bridge.mockFallbackHitCount,
      legacyCommandHitCount: bridge.legacyMcpCommandsSeen.length,
    };
    raw.provider = sanitizeJson({
      enabled: enabledRequests,
      disabled: disabledRequests,
    });
    raw.readModel = sanitizeJson({
      enabled: enabledReadModel,
      disabled: disabledReadModel,
    });
    raw.mcp = sanitizeJson(ledgerSummary);
    raw.bridge = sanitizeJson(bridge);
    summary.result = "pass";
  } catch (error) {
    summary.error = sanitizeText(
      error instanceof Error ? error.message : String(error),
    );
    if (page) {
      await page
        .screenshot({ path: failureScreenshotPath, fullPage: true })
        .catch(() => undefined);
      summary.artifacts.failureScreenshot = failureScreenshotPath;
    }
    throw error;
  } finally {
    summary.completedAt = new Date().toISOString();
    summary.errors ??= {
      consoleErrorCount: consoleErrors.length,
      pageErrorCount: pageErrors.length,
    };
    writeJsonFile(rawPath, raw);
    writeJsonFile(summaryPath, summary);
    await cleanupServer(page, ordinaryServer);
    await cleanupServer(page, appsServer);
    await closeElectronFixture(handle);
    await disabledProvider?.close().catch(() => undefined);
    await enabledProvider?.close().catch(() => undefined);
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
