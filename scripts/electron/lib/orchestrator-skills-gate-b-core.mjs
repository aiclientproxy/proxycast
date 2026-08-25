import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  LEGACY_MCP_COMMANDS,
} from "../../mcp/lib/current-smoke-transport.mjs";
import {
  parseInvokeTraceRaw,
  parseJsonRpcRequestsFromInvokeTrace,
} from "../mcp-config-fixture-smoke.mjs";

export const APPS_SERVER_NAME = "codex_apps";
export const ORDINARY_SERVER_NAME = "ordinary_fixture";
export const APPS_TOOL_NAME = "mcp__codex_apps__apps_ping";
export const ORDINARY_TOOL_NAME = "mcp__ordinary_fixture__ordinary_ping";
export const APPS_CONNECTOR_ID = "calendar";
export const APPS_LINK_ID = "link-calendar";
export const APPS_RESOURCE_URI = "ui://calendar/event.html";
export const APPS_RESOURCE_MARKER = "MCP_RESOURCE_ORIGIN_GATE_B_READY";
export const SKILL_SEARCH_TOOL_NAME = "skill_search";
export const SKILL_READ_TOOL_NAME = "read_mcp_resource";
export const SKILL_PACKAGE_URI = "skill://delivery/release-notes";
export const SKILL_RESOURCE_URI = `${SKILL_PACKAGE_URI}/SKILL.md`;
export const SKILL_BODY_MARKER = "ORCHESTRATOR_RELEASE_NOTES_BODY_READY";
const DEFAULT_GATE_OPTIONS = {
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "orchestrator-skills-gate-b",
  ),
  prefix: "orchestrator-skills-gate-b",
  timeoutMs: 240_000,
  intervalMs: 250,
  keepTemp: false,
};
export const REQUIRED_METHODS = [
  "config/read",
  "config/batchWrite",
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

export function parseOrchestratorGateArgs(
  argv,
  defaults = DEFAULT_GATE_OPTIONS,
) {
  const options = { ...defaults, help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
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
  if (
    !options.help &&
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(options.prefix)
  ) {
    throw new Error("--prefix 只能包含字母、数字、点、下划线和连字符");
  }
  if (
    !options.help &&
    (!Number.isFinite(options.timeoutMs) || options.timeoutMs <= 0)
  ) {
    throw new Error("--timeout-ms 必须为正数");
  }
  if (
    !options.help &&
    (!Number.isFinite(options.intervalMs) || options.intervalMs <= 0)
  ) {
    throw new Error("--interval-ms 必须为正数");
  }
  return options;
}

export function readJsonLines(filePath) {
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

export function writeOrchestratorMcpFixture(tempRoot) {
  const root = path.join(tempRoot, "orchestrator-skills-mcp");
  const serverPath = path.join(root, "server.mjs");
  const ledgerPath = path.join(root, "ledger.jsonl");
  fs.mkdirSync(root, { recursive: true });
  fs.writeFileSync(
    serverPath,
    String.raw`import fs from "node:fs";
import readline from "node:readline";

const ledgerPath = process.argv[2];
const role = process.argv[3];
const packageUri = ${JSON.stringify(SKILL_PACKAGE_URI)};
const resourceUri = ${JSON.stringify(SKILL_RESOURCE_URI)};
const appResourceUri = ${JSON.stringify(APPS_RESOURCE_URI)};
const appConnectorId = ${JSON.stringify(APPS_CONNECTOR_ID)};
const appLinkId = ${JSON.stringify(APPS_LINK_ID)};
const appResourceMarker = ${JSON.stringify(APPS_RESOURCE_MARKER)};
const skillBody = ${JSON.stringify(`---\nname: release-notes\ndescription: Prepare deterministic release notes.\n---\n\n# Release notes\n\n${SKILL_BODY_MARKER}\n`)};
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

function send(message) {
  process.stdout.write(JSON.stringify(message) + "\n");
}

function result(id, value) {
  send({ jsonrpc: "2.0", id, result: value });
}

function record(value) {
  fs.appendFileSync(ledgerPath, JSON.stringify({ ...value, pid: process.pid, role }) + "\n");
}

rl.on("line", (line) => {
  if (!line.trim()) return;
  const message = JSON.parse(line);
  const { id, method, params } = message;
  if (method === "initialize") {
    record({ type: "initialize", protocolVersion: params?.protocolVersion ?? null });
    result(id, {
      protocolVersion: params?.protocolVersion ?? "2025-03-26",
      capabilities: { resources: {}, tools: {} },
      serverInfo: { name: role + "-orchestrator-gate-b", version: "1.0.0" },
    });
    return;
  }
  if (method === "notifications/initialized") return;
  if (method === "tools/list") {
    record({ type: "tools_list" });
    const name = role === "apps" ? "apps_ping" : "ordinary_ping";
    result(id, {
      tools: [{
        name,
        description: "Deterministic " + role + " boundary probe",
        ...(role === "apps" ? {
          _meta: {
            connector_id: appConnectorId,
            connector_name: "Calendar",
            link_id: appLinkId,
            ui: { resourceUri: appResourceUri },
            _codex_apps: {
              resource_uri: "/calendar/" + appLinkId + "/apps_ping",
              requires_explicit_link_id: true,
            },
          },
        } : {}),
        inputSchema: {
          type: "object",
          "x-lime": {
            deferred_loading: false,
            always_visible: true,
            allowed_callers: ["assistant"],
          },
          properties: {
            message: { type: "string" },
            ...(role === "apps" ? { link_id: { type: "string" } } : {}),
          },
          required: role === "apps" ? ["message", "link_id"] : ["message"],
          additionalProperties: false,
        },
      }],
    });
    return;
  }
  if (method === "tools/call") {
    record({ type: "tool_call", name: params?.name ?? null });
    result(id, {
      content: [{ type: "text", text: role + "_pong" }],
      structuredContent: { role, ok: true },
      isError: false,
    });
    return;
  }
  if (method === "resources/list") {
    record({ type: "resources_list", cursor: params?.cursor ?? null });
    result(id, {
      resources: role === "apps" ? [{
        uri: packageUri,
        name: "release-notes",
        description: "Prepare deterministic release notes from delivery facts.",
        mimeType: "mcp/skill",
        _meta: {
          skill_name: "release-notes",
          plugin_name: "delivery",
          source: "plugin",
          allow_implicit_invocation: true,
        },
      }] : [],
    });
    return;
  }
  if (method === "resources/templates/list") {
    result(id, { resourceTemplates: [] });
    return;
  }
  if (method === "resources/read") {
    record({
      type: "resource_read",
      uri: params?.uri ?? null,
      threadId: params?._meta?.threadId ?? null,
      selectedConnectorIds:
        params?._meta?.["x-codex-turn-metadata"]?.mcp_request_meta
          ?.selected_connector_ids ?? null,
      linkId:
        params?._meta?.["x-codex-turn-metadata"]?.mcp_request_meta?.link_id ??
        null,
    });
    if (role === "apps" && params?.uri === resourceUri) {
      result(id, {
        contents: [{ uri: resourceUri, mimeType: "text/markdown", text: skillBody }],
      });
      return;
    }
    if (role === "apps" && params?.uri === appResourceUri) {
      result(id, {
        contents: [{
          uri: appResourceUri,
          mimeType: "text/html;profile=mcp-app",
          text: "<!doctype html><main data-resource-origin=\"ready\">" +
            appResourceMarker + "</main>",
        }],
      });
      return;
    }
    {
      send({ jsonrpc: "2.0", id, error: { code: -32602, message: "unknown resource" } });
      return;
    }
  }
  send({ jsonrpc: "2.0", id, error: { code: -32601, message: "unsupported method" } });
});
`,
    "utf8",
  );
  return { ledgerPath, root, serverPath };
}

function requestToolNames(request) {
  return (request?.body?.tools ?? [])
    .map((tool) => String(tool?.function?.name || tool?.name || "").trim())
    .filter(Boolean);
}

export function summarizeProviderRequests(requests) {
  return requests.map((request, index) => {
    const serialized = JSON.stringify(request.body ?? {});
    return {
      index,
      path: request.path,
      stream: request.body?.stream === true,
      responseKind: request.responseKind ?? null,
      responseToolName: request.responseToolName ?? null,
      toolNames: requestToolNames(request),
      hasSkillPackage: serialized.includes(SKILL_PACKAGE_URI),
      hasSkillBody: serialized.includes(SKILL_BODY_MARKER),
    };
  });
}

export function summarizeReadModel(read, runtime, expectedCallIds, finalText) {
  const serialized = JSON.stringify(read.result || {});
  return {
    threadIdentityStable:
      String(read.result?.thread?.id || "") === runtime.threadId ||
      serialized.includes(runtime.threadId),
    turnIdentityStable: serialized.includes(runtime.turnId),
    userItemVisible: serialized.includes(runtime.prompt),
    finalTextVisible: serialized.includes(finalText),
    guiFinalVisible: read.guiFinalVisible === true,
    expectedToolCallsVisible: expectedCallIds.every((callId) =>
      serialized.includes(callId),
    ),
  };
}

export function summarizeMcpLedger(ledgerPath) {
  const ledger = readJsonLines(ledgerPath);
  const resourceRead = ledger.find(
    (entry) =>
      entry?.type === "resource_read" && entry?.uri === SKILL_RESOURCE_URI,
  );
  const runtimePid = resourceRead?.pid ?? null;
  return {
    initializeCount: ledger.filter((entry) => entry?.type === "initialize")
      .length,
    runtimePidObserved: Number.isInteger(runtimePid),
    frozenTurnResourceListCount: ledger.filter(
      (entry) => entry?.type === "resources_list" && entry?.pid === runtimePid,
    ).length,
    exactResourceReadCount: ledger.filter(
      (entry) =>
        entry?.type === "resource_read" && entry?.uri === SKILL_RESOURCE_URI,
    ).length,
    appsToolCallCount: ledger.filter(
      (entry) => entry?.type === "tool_call" && entry?.role === "apps",
    ).length,
    ordinaryToolCallCount: ledger.filter(
      (entry) =>
        entry?.type === "tool_call" &&
        entry?.role === "ordinary" &&
        entry?.name === "ordinary_ping",
    ).length,
  };
}

export async function collectElectronEvidence(page, observedMethods) {
  const traceRaw =
    (await page.evaluate(() =>
      localStorage.getItem("lime_invoke_trace_buffer_v1"),
    )) || "";
  const errorRaw =
    (await page.evaluate(() =>
      localStorage.getItem("lime_invoke_error_buffer_v1"),
    )) || "";
  const trace = parseInvokeTraceRaw(traceRaw);
  const requests = parseJsonRpcRequestsFromInvokeTrace(traceRaw);
  const requestMethods = Array.from(
    new Set([...observedMethods, ...requests.map((request) => request.method)]),
  );
  const commands = Array.from(
    new Set(trace.map((entry) => entry?.command).filter(Boolean)),
  );
  return {
    appServerHandleJsonLinesSeen: commands.includes(
      APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    ),
    electronIpcSeen: requests.some(
      (request) => request.transport === "electron-ipc",
    ),
    requestMethods,
    missingRequiredMethods: REQUIRED_METHODS.filter(
      (method) => !requestMethods.includes(method),
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
    invokeErrorCount: parseInvokeTraceRaw(errorRaw).length,
  };
}
