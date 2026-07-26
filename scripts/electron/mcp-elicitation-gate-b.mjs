#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
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
const TOOL_SUFFIX = "release_check";
const REQUIRED_METHODS = [
  "workspace/default/ensure",
  "mcpServer/create",
  "mcpServer/start",
  "mcpTool/list",
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
      capabilities: { tools: {} },
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
      }],
    });
    return;
  }
  if (method === "tools/call") {
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

async function cleanupMcpServer(page, server) {
  if (!page || !server?.id) return;
  await appServerCallFromPage(page, "mcpServer/stop", {
    name: server.name,
  }).catch(() => undefined);
  await appServerCallFromPage(page, "mcpServer/delete", {
    id: server.id,
  }).catch(() => undefined);
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
  assert(workspaceId, "workspace/default/ensure 未返回 workspace.id");
  return { response, workspaceId };
}

async function startRuntimeTurn(
  page,
  { fixture, observedMethods, serverName, workspaceId },
) {
  const sessionId = `mcp-elicitation-${Date.now()}-${process.pid}`;
  const threadId = `${sessionId}-thread`;
  const turnId = `${sessionId}-turn`;
  const expectedToolName = toolName(serverName);
  const start = await appServerCallFromPage(page, "thread/start", {
    sessionId,
    threadId,
    appId: "desktop",
    workspaceId,
    businessObjectRef: {
      kind: "agent.session",
      id: `agent-session:${workspaceId}:${sessionId}`,
      title: "MCP elicitation Gate B",
      metadata: {
        title: "MCP elicitation Gate B",
        executionStrategy: "react",
        runStartHooks: false,
        harness: { source: "smoke:mcp-elicitation-gate-b" },
      },
    },
  });
  observedMethods.add(start.method);
  const update = await appServerCallFromPage(page, "thread/settings/update", {
    threadId,
    modelProvider: fixture.provider.providerPreference,
    model: fixture.provider.modelPreference,
  });
  observedMethods.add(update.method);
  const turn = await appServerCallFromPage(page, "turn/start", {
    sessionId,
    turnId,
    input: { text: "Run the release check through the available MCP tool." },
    runtimeOptions: {
      stream: true,
      eventName: `mcp_elicitation_${turnId}`,
      runtimeRequest: {
        providerPreference: fixture.provider.providerPreference,
        modelPreference: fixture.provider.modelPreference,
        providerConfig: fixture.provider.providerConfig,
        approvalPolicy: "never",
        sandboxPolicy: "danger-full-access",
        executionStrategy: "react",
        metadata: {
          harness: {
            source: "smoke:mcp-elicitation-gate-b",
            skip_mcp_prewarm: false,
          },
          tool_scope: { allowed_tools: [expectedToolName] },
        },
      },
    },
    queueIfBusy: false,
    skipPreSubmitResume: true,
  });
  observedMethods.add(turn.method);
  return { expectedToolName, sessionId, start, threadId, turn, turnId, update };
}

async function waitForElicitationDialog(page, options) {
  const dialog = page.getByRole("dialog");
  await dialog.waitFor({
    state: "visible",
    timeout: Math.min(options.timeoutMs, 90_000),
  });
  await dialog.locator('input[type="checkbox"]').check();
  const checked = await dialog.locator('input[type="checkbox"]').isChecked();
  assert(checked, "Renderer MCP 表单未提交 confirmed=true");
  return dialog;
}

async function submitElicitation(page, dialog, options) {
  const submit = dialog.getByRole("button", { name: /提交|Submit/ });
  await submit.click({ timeout: Math.min(options.timeoutMs, 30_000) });
  await dialog.waitFor({
    state: "hidden",
    timeout: Math.min(options.timeoutMs, 60_000),
  });
  return (await page.getByRole("dialog").count()) === 0;
}

function providerRequestSummary(requests) {
  return requests.map((request, index) => ({
    index,
    path: request.path,
    stream: request.body?.stream === true,
    toolNames: (request.body?.tools ?? [])
      .map((tool) => String(tool?.function?.name || tool?.name || "").trim())
      .filter(Boolean),
  }));
}

function isExactEmptyObject(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).length === 0
  );
}

function mcpInitializeCapabilityEvidence(ledger) {
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
  return {
    acceptedPid: accepted?.pid ?? null,
    capabilityMissingCount,
    managementInitialize: managementInitialize ?? null,
    managementElicitationCapabilityAbsent: Boolean(managementInitialize),
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
) {
  const startedAt = Date.now();
  let latestRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latestRead = await appServerCallFromPage(page, "thread/read", {
      sessionId: runtime.sessionId,
      historyLimit: 80,
    });
    observedMethods.add(latestRead.method);
    const ledger = readJsonLines(ledgerPath);
    const capabilityEvidence = mcpInitializeCapabilityEvidence(ledger);
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

function electronEvidence(traceRaw, observedMethods) {
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
    missingRequiredMethods: REQUIRED_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    requestMethods: methods,
  };
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
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
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    backendMode: "runtime",
    proofLevel: "Gate B",
    capabilityAdvertisementRequired: true,
    capabilityMissingCount: null,
    managementElicitationCapabilityAbsent: false,
    runtimeClientCapabilities: null,
    runtimeInitializeProtocolVersion: null,
    electronPreloadBridge: false,
    appServerHandleJsonLinesSeen: false,
    rendererFormVisible: false,
    rendererConfirmedSubmitted: false,
    dialogClosedAfterResolved: false,
    mcpLedgerAccepted: false,
    providerFinalTextObserved: false,
    providerRequestCount: 0,
    consoleErrors: [],
    missingRequiredMethods: [...REQUIRED_METHODS],
    legacyMcpCommandsSeen: [],
    screenshot: null,
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
  let mcpFixture = null;
  let server = null;

  try {
    logStage("start-provider-fixture");
    const serverName = makeServerName();
    const expectedToolName = toolName(serverName);
    fixture = await startOpenAiCompatibleFixtureServer({
      scriptedResponses: [
        {
          type: "tool_call",
          id: "call-mcp-elicitation-release-check",
          name: expectedToolName,
          arguments: { release: "gate-b" },
        },
        { type: "text", content: FINAL_TEXT },
      ],
    });
    mcpFixture = writeElicitationFixture();

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
    summary.electronPreloadBridge =
      handle.rendererSnapshot.electron &&
      handle.rendererSnapshot.hasInvokeBridge;

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
    const startedServer = await appServerCallFromPage(page, "mcpServer/start", {
      name: serverName,
    });
    observedMethods.add(startedServer.method);
    raw.mcpServerStart = sanitizeJson(startedServer);
    raw.mcpToolList = sanitizeJson(
      await waitForTool(page, expectedToolName, options, observedMethods),
    );

    logStage("start-agent-turn");
    const workspace = await ensureWorkspace(page, observedMethods);
    raw.workspace = sanitizeJson(workspace.response);
    const runtime = await startRuntimeTurn(page, {
      fixture,
      observedMethods,
      serverName,
      workspaceId: workspace.workspaceId,
    });
    raw.runtime = sanitizeJson(runtime);

    logStage("submit-renderer-form");
    const dialog = await waitForElicitationDialog(page, options);
    summary.rendererFormVisible = true;
    await page.screenshot({ path: screenshotPath, fullPage: true });
    summary.screenshot = screenshotPath;
    summary.dialogClosedAfterResolved = await submitElicitation(
      page,
      dialog,
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
    summary.providerRequestCount = providerRequests.length;
    summary.mcpLedgerAccepted = completion.ledger.some(
      (entry) =>
        entry?.action === "accept" && entry?.content?.confirmed === true,
    );
    summary.providerFinalTextObserved = JSON.stringify(
      completion.read.result,
    ).includes(FINAL_TEXT);
    const traceRaw = completion.read.traceRaw;
    const evidence = electronEvidence(traceRaw, observedMethods);
    summary.appServerHandleJsonLinesSeen =
      evidence.appServerHandleJsonLinesSeen;
    summary.missingRequiredMethods = evidence.missingRequiredMethods;
    summary.legacyMcpCommandsSeen = evidence.legacyMcpCommandsSeen;
    raw.electronEvidence = sanitizeJson(evidence);
    summary.consoleErrors = [...consoleErrors];

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
    assert(
      providerRequests[0]?.toolNames?.includes(expectedToolName),
      `首个 provider request 未携带 scoped MCP tool: ${expectedToolName}`,
    );
    assert(
      providerRequests.length >= 2,
      "provider 未完成 tool result 后的第二次请求",
    );
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
      summary.dialogClosedAfterResolved,
      "serverRequest/resolved 后表单未关闭",
    );
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
    writeJsonFile(rawPath, raw);
    writeJsonFile(summaryPath, summary);
    await cleanupMcpServer(page, server);
    await closeElectronFixture({ app });
    if (fixture) await fixture.close().catch(() => undefined);
    if (mcpFixture && !options.keepTemp) {
      fs.rmSync(mcpFixture.root, { recursive: true, force: true });
    }
    if (!options.keepTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
  console.log(`${LOG_PREFIX} pass summary=${summaryPath}`);
}

run().catch((error) => {
  console.error(
    `${LOG_PREFIX} failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
