#!/usr/bin/env node

import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import process from "node:process";
import {
  assertLiveProviderSmokeAllowed,
  liveProviderSmokeAllowed,
} from "../lib/live-provider-smoke-gate.mjs";
import {
  createAgentSessionCurrent,
  readAgentRuntimeThreadCurrent,
  startAgentSessionTurnCurrent,
} from "../lib/agent-runtime-smoke-core.mjs";

const DEFAULTS = {
  mode: "replay",
  healthUrl: "http://127.0.0.1:3030/health",
  invokeUrl: "http://127.0.0.1:3030/invoke",
  timeoutMs: 300_000,
  intervalMs: 1_000,
  appId: "content-factory-app",
  sessionId: "",
  taskId: "",
  expectedRef: "",
  output: "",
  workspaceId: "",
  providerPreference: "",
  modelPreference: "",
  connectorId: "notion",
  action: "createPage",
  title: "P18.7-E runtime outbox smoke",
  expectAdapterReadiness: "",
  expectExternalStatus: "",
  expectNextRequired: "",
  expectSecretDeliveryStatus: "",
  expectSecretDeliverySource: "",
  expectSecretDeliveryTarget: "",
  expectSecretDeliveryLeaseObserved: "",
  expectSecretDeliveryLeaseRefExposed: "",
  expectSecretDeliveryLeaseHandleStatus: "",
  expectSecretDeliveryCredentialMaterialExposed: "",
  expectSecretDeliveryTokenExposed: "",
  expectDeliveryStatus: "",
  expectDeliveryExternalPlatformDelivered: "",
  expectProductionPlatformDelivered: "",
  externalDeliveryWebhookUrl: "",
  externalDeliveryWebhookUrlEnv: "",
  externalDeliveryWebhookUrlFile: "",
  externalDeliveryWebhookSource: "",
  externalDeliveryWebhookLabel: "connector-outbox-smoke-webhook",
  externalDeliveryLocalWebhook: false,
  allowLiveProvider: liveProviderSmokeAllowed(),
};

const TOOL_METADATA_BEGIN = "[Lime \u5de5\u5177\u5143\u6570\u636e\u5f00\u59cb]";
const TOOL_METADATA_END = "[Lime \u5de5\u5177\u5143\u6570\u636e\u7ed3\u675f]";
const PROVIDER_PICK_ORDER = [
  "deepseek",
  "openai",
  "anthropic",
  "gemini",
  "azure-openai",
];
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_METHOD_MODEL_PROVIDER_LIST = "modelProvider/list";
const APP_SERVER_METHOD_MODEL_PROVIDER_READ = "modelProvider/read";

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--help") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--external-delivery-local-webhook") {
      options.externalDeliveryLocalWebhook = true;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg.startsWith("--") && next !== undefined && !next.startsWith("--")) {
      const key = arg
        .slice(2)
        .replace(/-([a-z])/g, (_, value) => value.toUpperCase());
      if (key in options) {
        options[key] = next;
        index += 1;
      }
    }
  }

  options.timeoutMs = Number(options.timeoutMs);
  options.intervalMs = Number(options.intervalMs);
  if (!["replay", "live"].includes(options.mode)) {
    throw new Error("--mode must be replay or live");
  }
  if (
    options.mode !== "live" &&
    (options.externalDeliveryLocalWebhook ||
      options.externalDeliveryWebhookUrl ||
      options.externalDeliveryWebhookUrlEnv ||
      options.externalDeliveryWebhookUrlFile)
  ) {
    throw new Error("external delivery webhook options require --mode live");
  }
  resolveExternalDeliveryWebhookUrl(options);
  if (
    options.externalDeliveryLocalWebhook &&
    (options.externalDeliveryWebhookUrl ||
      options.externalDeliveryWebhookUrlEnv ||
      options.externalDeliveryWebhookUrlFile)
  ) {
    throw new Error(
      "--external-delivery-local-webhook cannot be combined with remote webhook URL options",
    );
  }
  if (
    options.externalDeliveryWebhookUrl &&
    !isSupportedExternalDeliveryWebhookUrl(options.externalDeliveryWebhookUrl)
  ) {
    throw new Error(
      "external delivery webhook URL must be https://, http://127.0.0.1:<port>, or http://localhost:<port>",
    );
  }
  if (
    options.externalDeliveryWebhookSource === "cli" &&
    !isLocalExternalDeliveryWebhookUrl(options.externalDeliveryWebhookUrl)
  ) {
    throw new Error(
      "remote external delivery webhook URL must be provided through --external-delivery-webhook-url-env or --external-delivery-webhook-url-file",
    );
  }
  if (options.mode === "live") {
    const expectsExternalDelivery =
      options.externalDeliveryLocalWebhook ||
      options.externalDeliveryWebhookUrl;
    options.expectAdapterReadiness ||=
      "host_managed_secret_delivery_adapter_ready";
    options.expectExternalStatus ||= expectsExternalDelivery
      ? "delivered"
      : "not_delivered";
    options.expectNextRequired ||= expectsExternalDelivery
      ? "external_platform_delivery_complete"
      : "external_platform_delivery";
    options.expectSecretDeliveryStatus ||= "ready";
    options.expectSecretDeliverySource ||= "host_managed_secret_delivery_fact";
    options.expectSecretDeliveryTarget ||= "cloud_overlay_worker";
    options.expectSecretDeliveryLeaseObserved ||= "true";
    options.expectSecretDeliveryLeaseRefExposed ||= "false";
    options.expectSecretDeliveryLeaseHandleStatus ||= "host_managed";
    options.expectSecretDeliveryCredentialMaterialExposed ||= "false";
    options.expectSecretDeliveryTokenExposed ||= "false";
    options.expectDeliveryStatus ||= expectsExternalDelivery
      ? "delivered_to_external_platform"
      : "accepted_by_local_cloud_overlay_worker";
    options.expectDeliveryExternalPlatformDelivered ||= expectsExternalDelivery
      ? "true"
      : "false";
    options.expectProductionPlatformDelivered ||= "false";
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs <= 0) {
    throw new Error("--timeout-ms must be a positive number");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs <= 0) {
    throw new Error("--interval-ms must be a positive number");
  }
  if (options.mode === "replay" && (!options.sessionId || !options.taskId)) {
    throw new Error("--mode replay requires --session-id and --task-id");
  }
  if (!options.output) {
    const stamp = new Date()
      .toISOString()
      .replace(/[-:]/g, "")
      .replace(/\..+/, "");
    options.output = path.join(
      process.cwd(),
      ".lime",
      "qc",
      "gui-evidence",
      "plugins",
      `p18-7-e-connector-outbox-runtime-smoke-${stamp}.json`,
    );
  }
  return options;
}

function printHelp() {
  console.log(`Usage:
  node scripts/plugin/connector-outbox-smoke.mjs --mode replay --session-id <id> --task-id <id>
  node scripts/plugin/connector-outbox-smoke.mjs --mode live [--provider-preference <provider>] [--model-preference <model>]

Options:
  --mode <replay|live>             Default replay. Replay reads an existing runtime session only.
  --session-id <id>                Existing AgentRuntime session id for replay.
  --task-id <id>                   Plugin task id for replay or live task projection.
  --expected-ref <ref>             Expected outbox evidence ref. If omitted, any outbox:// ref passes.
  --provider-preference <id>       Optional for live mode; auto-picks an enabled provider when omitted.
  --model-preference <model>       Optional for live mode; auto-picks a fast configured model when omitted.
  --allow-live-provider            Required for live mode unless LIME_ALLOW_LIVE_PROVIDER_SMOKE=1 / LIME_REAL_API_TEST=1 is set.
  --connector-id <id>              Default notion.
  --action <action>                Default createPage.
  --title <title>                  Live connector input title.
  --expect-adapter-readiness <v>   Optional replay assertion; live defaults to host_managed_secret_delivery_adapter_ready.
  --expect-external-status <v>     Optional replay assertion; live defaults to not_delivered, or delivered with external webhook.
  --expect-next-required <v>       Optional replay assertion; live defaults to external_platform_delivery, or external_platform_delivery_complete with external webhook.
  --expect-secret-delivery-status <v>
                                   Optional replay assertion; live defaults to ready.
  --expect-secret-delivery-source <v>
                                   Optional replay assertion; live defaults to host_managed_secret_delivery_fact.
  --expect-secret-delivery-target <v>
                                   Optional replay assertion; live defaults to cloud_overlay_worker.
  --expect-secret-delivery-lease-observed <true|false>
                                   Optional replay assertion; live defaults to true.
  --expect-secret-delivery-lease-ref-exposed <true|false>
                                   Optional replay assertion; live defaults to false.
  --expect-secret-delivery-lease-handle-status <v>
                                   Optional replay assertion; live defaults to host_managed.
  --expect-secret-delivery-credential-material-exposed <true|false>
                                   Optional replay assertion; live defaults to false.
  --expect-secret-delivery-token-exposed <true|false>
                                   Optional replay assertion; live defaults to false.
  --expect-delivery-status <v>     Optional replay assertion; live defaults to accepted_by_local_cloud_overlay_worker, or delivered_to_external_platform with external webhook.
  --expect-delivery-external-platform-delivered <true|false>
                                   Optional replay assertion; live defaults to false, or true with external webhook.
  --expect-production-platform-delivered <true|false>
                                   Optional replay assertion; live defaults to false because webhook receipt is not production platform delivery.
  --external-delivery-local-webhook
                                   Live mode only. Starts a local webhook, injects it through internalRequest, and expects delivered_to_external_platform.
  --external-delivery-webhook-url <url>
                                   Live mode only. Local/non-sensitive target only; remote secret URLs must use env/file.
  --external-delivery-webhook-url-env <env>
                                   Live mode only. Reads the Host-managed webhook URL from an environment variable; avoids leaking the URL through process args.
  --external-delivery-webhook-url-file <path>
                                   Live mode only. Reads the Host-managed webhook URL from a local secret file; the summary records only the source kind.
  --external-delivery-webhook-label <label>
                                   Optional App-safe label for the Host-managed webhook target.
  --output <path>                  Evidence JSON path.
`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function resolveExternalDeliveryWebhookUrl(options) {
  if (options.externalDeliveryWebhookUrl) {
    options.externalDeliveryWebhookUrl =
      options.externalDeliveryWebhookUrl.trim();
    options.externalDeliveryWebhookSource = "cli";
    return;
  }
  if (options.externalDeliveryWebhookUrlEnv) {
    const envName = options.externalDeliveryWebhookUrlEnv.trim();
    const value = process.env[envName]?.trim() || "";
    if (!value) {
      throw new Error(
        `environment variable ${envName} is required for external delivery webhook`,
      );
    }
    options.externalDeliveryWebhookUrl = value;
    options.externalDeliveryWebhookSource = "env";
    return;
  }
  if (options.externalDeliveryWebhookUrlFile) {
    const filePath = path.resolve(options.externalDeliveryWebhookUrlFile);
    const value = fs.readFileSync(filePath, "utf8").trim();
    if (!value) {
      throw new Error("external delivery webhook URL file is empty");
    }
    options.externalDeliveryWebhookUrl = value;
    options.externalDeliveryWebhookSource = "file";
  }
}

function isSupportedExternalDeliveryWebhookUrl(value) {
  const url = value.trim();
  if (url.startsWith("https://")) {
    return true;
  }
  return isLocalExternalDeliveryWebhookUrl(url);
}

function isLocalExternalDeliveryWebhookUrl(value) {
  const url = value.trim();
  return (
    url.startsWith("http://127.0.0.1:") || url.startsWith("http://localhost:")
  );
}

async function readJson(url, init, timeoutMs) {
  const response = await fetch(url, {
    ...init,
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  return { ok: response.ok, status: response.status, body, text };
}

function startLocalWebhookServer() {
  const requests = [];
  const server = http.createServer((request, response) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      requests.push({
        method: request.method,
        url: request.url,
        headers: request.headers,
        body: Buffer.concat(chunks).toString("utf8"),
      });
      response.writeHead(200, { "content-type": "text/plain" });
      response.end("OK");
    });
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("local webhook server did not expose a TCP address"));
        return;
      }
      resolve({
        url: `http://127.0.0.1:${address.port}`,
        requests,
        close: () =>
          new Promise((closeResolve) => {
            server.close(() => closeResolve());
          }),
      });
    });
  });
}

async function waitForHealth(options) {
  const startedAt = Date.now();
  let lastError = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const response = await readJson(options.healthUrl, {}, 5_000);
      if (response.ok) {
        return response.body;
      }
      lastError = new Error(`health HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `DevBridge health unavailable: ${lastError?.message || "timeout"}`,
  );
}

async function invoke(options, cmd, args, timeoutMs = 30_000) {
  const response = await readJson(
    options.invokeUrl,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ cmd, args }),
    },
    timeoutMs,
  );
  if (!response.ok) {
    throw new Error(`${cmd} HTTP ${response.status}: ${response.text}`);
  }
  if (response.body?.error) {
    throw new Error(`${cmd} error: ${response.body.error}`);
  }
  return response.body?.result;
}

let appServerRequestId = 1;

async function invokeAppServerMethod(
  options,
  method,
  params,
  timeoutMs = 30_000,
) {
  const id = `connector-outbox-${appServerRequestId++}`;
  const request =
    params === undefined ? { id, method } : { id, method, params };
  const result = await invoke(
    options,
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    { request: { lines: [`${JSON.stringify(request)}\n`] } },
    timeoutMs,
  );
  const responseLines = result?.result?.lines ?? result?.lines;
  const messages = (Array.isArray(responseLines) ? responseLines : [])
    .map((line) => {
      try {
        return JSON.parse(String(line));
      } catch {
        return null;
      }
    })
    .filter(Boolean);
  const error = messages.find((message) => message.id === id && message.error);
  if (error) {
    throw new Error(
      `${method} error: ${error.error?.message || "App Server JSON-RPC error"}`,
    );
  }
  const response = messages.find(
    (message) => message.id === id && Object.hasOwn(message, "result"),
  );
  if (!response) {
    throw new Error(`${method} missing App Server response`);
  }
  return response.result;
}

function uniqueStrings(values) {
  return [
    ...new Set(values.filter((value) => typeof value === "string" && value)),
  ];
}

function getEvidenceSummaryRefs(threadRead) {
  const summary =
    threadRead?.evidence_summary || threadRead?.evidenceSummary || {};
  return summary.evidence_refs || summary.evidenceRefs || [];
}

function getToolCalls(threadRead) {
  return threadRead?.tool_calls || threadRead?.toolCalls || [];
}

function getTaskEvents(task) {
  return task?.taskEvents || task?.events || task?.runtimeProcess?.events || [];
}

function eventType(event) {
  return String(event?.eventType || event?.type || "");
}

function collectTaskEvidenceRefs(events) {
  return uniqueStrings(
    events.map((event) => event?.evidenceRef || event?.payload?.evidenceRef),
  );
}

function collectOutboxRefs(...groups) {
  return uniqueStrings(
    groups.flat().filter((value) => value.startsWith("outbox://")),
  );
}

function collectDeliveryRefs(...groups) {
  return uniqueStrings(
    groups.flat().filter((value) => value.startsWith("delivery://")),
  );
}

function getToolEvidenceRefs(toolCall) {
  return toolCall?.evidence_refs || toolCall?.evidenceRefs || [];
}

function toolOutput(toolCall) {
  return String(
    toolCall?.output ||
      toolCall?.output_preview ||
      toolCall?.outputPreview ||
      "",
  );
}

function parseJsonObject(text) {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value)
      ? value
      : null;
  } catch {
    return null;
  }
}

function parseToolOutputPayload(output) {
  const markerIndex = output.indexOf(TOOL_METADATA_BEGIN);
  const prefix = (
    markerIndex >= 0 ? output.slice(0, markerIndex) : output
  ).trim();
  const prefixPayload = prefix ? parseJsonObject(prefix) : null;
  if (prefixPayload) {
    return prefixPayload;
  }

  if (markerIndex < 0) {
    return null;
  }
  const afterBegin = output.slice(markerIndex + TOOL_METADATA_BEGIN.length);
  const endIndex = afterBegin.indexOf(TOOL_METADATA_END);
  if (endIndex < 0) {
    return null;
  }
  const metadata = parseJsonObject(afterBegin.slice(0, endIndex).trim());
  return metadata?.result && typeof metadata.result === "object"
    ? metadata.result
    : metadata;
}

function isTrustedConnectorOutputPayload(payload) {
  const source = String(
    payload?.source || payload?.result?.source || "",
  ).trim();
  return [
    "plugin_connector_cloud_overlay_outbox_adapter",
    "plugin_connector_fixture_adapter",
  ].includes(source);
}

function toolNameFor(connectorId, action) {
  return `connector__${connectorId}__${action}`;
}

function buildExpectedRef(options, refs) {
  if (options.expectedRef) {
    return options.expectedRef;
  }
  return refs.find((value) => value.startsWith("outbox://")) || "";
}

function normalizeExpectation(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

function addExpectedAssertion(assertions, expectations, key, actual, expected) {
  const normalizedExpected = normalizeExpectation(expected);
  if (!normalizedExpected) {
    return;
  }
  expectations[key] = expected;
  assertions[key] = normalizeExpectation(actual) === normalizedExpected;
}

function addExpectedPrefixAssertion(
  assertions,
  expectations,
  key,
  actual,
  expectedPrefix,
) {
  const normalizedPrefix = String(expectedPrefix ?? "").trim();
  if (!normalizedPrefix) {
    return;
  }
  expectations[key] = expectedPrefix;
  assertions[key] = String(actual ?? "")
    .trim()
    .startsWith(normalizedPrefix);
}

async function readProjection(options, sessionId, taskId) {
  const threadRead = await readAgentRuntimeThreadCurrent(options, sessionId);
  const task = await invoke(options, "plugin_runtime_get_task", {
    request: {
      appId: options.appId,
      taskId,
      sessionId,
    },
  });
  return { threadRead, task };
}

async function waitForProjection(options, sessionId, taskId, expectedToolName) {
  const startedAt = Date.now();
  let latest = null;
  let latestWithEvidence = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latest = await readProjection(options, sessionId, taskId);
    const status = latest.threadRead?.status;
    const toolCalls = getToolCalls(latest.threadRead);
    const matchingTool = toolCalls.find(
      (toolCall) =>
        (toolCall.tool_name || toolCall.toolName) === expectedToolName &&
        toolCall.status === "completed",
    );
    const refs = collectOutboxRefs(
      matchingTool?.evidence_refs || matchingTool?.evidenceRefs || [],
      getEvidenceSummaryRefs(latest.threadRead),
      collectTaskEvidenceRefs(getTaskEvents(latest.task)),
    );
    if (matchingTool && refs.length > 0) {
      latestWithEvidence = latest;
      if (status === "completed") {
        return latest;
      }
    }
    if (["failed", "cancelled", "canceled"].includes(status)) {
      return latest;
    }
    await sleep(options.intervalMs);
  }
  return latestWithEvidence || latest;
}

async function getWorkspaceId(options) {
  if (options.workspaceId) {
    return options.workspaceId;
  }
  const workspace = await invoke(
    options,
    "get_or_create_default_project",
    undefined,
  );
  assert(workspace?.id, "get_or_create_default_project returned no id");
  return workspace.id;
}

function normalizeProviderId(provider) {
  return String(
    provider?.id || provider?.provider_id || provider?.providerId || "",
  ).trim();
}

function providerEnabled(provider) {
  return provider?.enabled !== false;
}

function pickModelPreference(provider) {
  const candidates = [
    ...(Array.isArray(provider?.models) ? provider.models : []),
  ]
    .map((value) =>
      typeof value === "string"
        ? value
        : String(value?.name || value?.id || value?.model || "").trim(),
    )
    .filter(Boolean);

  return (
    candidates.find((value) => /flash|mini|lite/i.test(value)) ||
    candidates[0] ||
    ""
  );
}

function pickProvider(providers, preferredProviderId) {
  const enabled = providers.filter((provider) => providerEnabled(provider));
  if (preferredProviderId) {
    return (
      enabled.find(
        (provider) => normalizeProviderId(provider) === preferredProviderId,
      ) ||
      providers.find(
        (provider) => normalizeProviderId(provider) === preferredProviderId,
      ) ||
      null
    );
  }

  for (const providerId of PROVIDER_PICK_ORDER) {
    const match = enabled.find(
      (provider) => normalizeProviderId(provider) === providerId,
    );
    if (match) {
      return match;
    }
  }

  return enabled[0] || null;
}

async function resolveProviderPreference(options) {
  const explicitProvider = String(options.providerPreference || "").trim();
  const explicitModel = String(options.modelPreference || "").trim();
  if (explicitProvider && explicitModel) {
    return {
      providerPreference: explicitProvider,
      modelPreference: explicitModel,
      source: "explicit",
    };
  }

  const providerList = await invokeAppServerMethod(
    options,
    APP_SERVER_METHOD_MODEL_PROVIDER_LIST,
    {},
    30_000,
  );
  const providers = providerList?.providers;
  const selected = pickProvider(
    Array.isArray(providers) ? providers : [],
    explicitProvider,
  );
  const providerId = normalizeProviderId(selected);
  assert(
    providerId,
    "No enabled provider found; pass --provider-preference and --model-preference or configure a local provider",
  );

  let providerDetail = selected;
  try {
    providerDetail =
      (
        await invokeAppServerMethod(
          options,
          APP_SERVER_METHOD_MODEL_PROVIDER_READ,
          { providerId },
          30_000,
        )
      )?.provider || selected;
  } catch (error) {
    console.warn(
      `[plugin-connector-outbox-smoke] provider detail unavailable, using list summary: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }

  const modelPreference = explicitModel || pickModelPreference(providerDetail);
  assert(
    modelPreference,
    `Provider ${providerId} has no configured model; pass --model-preference`,
  );

  return {
    providerPreference: providerId,
    modelPreference,
    source:
      explicitProvider || explicitModel
        ? "partial-explicit"
        : "auto-enabled-provider",
  };
}

function buildLiveMetadata(options, taskId, idempotencyKey) {
  const toolName = toolNameFor(options.connectorId, options.action);
  const leaseRef = `secret-lease://connector/${options.connectorId}/${options.action}/${idempotencyKey}`;
  const leaseExpiresAt = new Date(Date.now() + 15 * 60_000).toISOString();
  const publicSecretDelivery = {
    status: "ready",
    binding: "host_managed",
    source: "host_managed_secret_delivery_fact",
    target: "cloud_overlay_worker",
    leaseObserved: true,
    leaseRefExposed: false,
    leaseHandleStatus: "host_managed",
    credentialMaterialExposed: false,
    tokenExposed: false,
  };
  const connectorRuntimeFacts = {
    connectorId: options.connectorId,
    status: "authorized",
    authorizationStatus: "authorized",
    source: "plugin_connector_authorization_task",
    secretBinding: "host_managed",
    tokenExposed: false,
    secretDelivery: publicSecretDelivery,
  };
  const publicRequest = {
    capability: "lime.connectors",
    method: "invoke",
    appId: options.appId,
    entryKey: "connector-outbox-smoke",
    taskId,
    toolName,
    action: options.action,
    input: {
      connectorId: options.connectorId,
      action: options.action,
      idempotencyKey,
      input: {
        title: options.title,
      },
      connectorRuntimeFacts,
    },
    policy: {
      owner: "lime_agent_runtime",
      scope: "plugin_session",
      approvalRequired: true,
      sandboxRequired: false,
      mutationExposed: false,
      tokenExposed: false,
      secretBinding: "host_managed",
    },
    idempotencyKey,
  };
  const externalDelivery = options.externalDeliveryWebhookUrl
    ? {
        status: "ready",
        binding: "host_managed",
        channel: "webhook",
        target: options.externalDeliveryWebhookUrl,
        targetLabel: options.externalDeliveryWebhookLabel,
        targetExposed: false,
        credentialMaterialExposed: false,
        tokenExposed: false,
      }
    : null;

  return {
    harness: {
      task_mode_enabled: true,
      plugin_tool_execution: {
        source: "plugin_runtime",
        request: publicRequest,
        internalRequest: {
          ...publicRequest,
          input: {
            ...publicRequest.input,
            connectorRuntimeFacts: {
              ...connectorRuntimeFacts,
              secretDelivery: {
                ...publicSecretDelivery,
                leaseRef,
                expiresAt: leaseExpiresAt,
                ...(externalDelivery ? { externalDelivery } : {}),
              },
            },
          },
        },
      },
    },
    lime_runtime: {
      surface: "plugin",
      task_id: taskId,
      tool_surface: "plugin_tool_execution",
    },
    tool_scope: {
      source: "plugin_runtime",
      reason: "plugin_tool_execution_request",
      mode: "tool_runtime_owner_binding",
    },
  };
}

async function runLive(options) {
  const workspaceId = await getWorkspaceId(options);
  const providerPreference = await resolveProviderPreference(options);
  const stamp = `${Date.now()}-${process.pid}`;
  const sessionId = await createAgentSessionCurrent(options, {
    workspaceId,
    title: `P18.7-E connector outbox smoke ${stamp}`,
    metadata: {
      harness: {
        hiddenFromUserRecents: true,
        source: "smoke:plugin-connector-outbox",
        connectorId: options.connectorId,
        action: options.action,
      },
    },
  });
  const taskId = options.taskId || `plugin-connector-outbox-runtime-${stamp}`;
  const turnId = `connector-outbox-${stamp}`;
  const eventName = `agent_stream_${sessionId}_${turnId}`;
  const idempotencyKey = `${options.connectorId}-${options.action}-${stamp}`;
  const expectedToolName = toolNameFor(options.connectorId, options.action);

  await startAgentSessionTurnCurrent(options, {
    sessionId,
    workspaceId,
    message: [
      `Call ${expectedToolName} exactly once.`,
      `Use exactly this tool argument JSON: ${JSON.stringify({
        connectorId: options.connectorId,
        action: options.action,
        input: { title: options.title },
        reason: `Plugin connector outbox smoke ${idempotencyKey}`,
      })}.`,
      `Do not use connectorId values other than "${options.connectorId}".`,
      `Do not use action values other than "${options.action}".`,
      "Do not call the external provider API directly.",
      "Do not request or output tokens or secrets.",
      "After the tool call, summarize the queued outbox result in one sentence.",
    ].join(" "),
    eventName,
    turnId,
    runtimeRequest: {
      providerPreference: providerPreference.providerPreference,
      modelPreference: providerPreference.modelPreference,
      metadata: buildLiveMetadata(options, taskId, idempotencyKey),
    },
    skipPreSubmitResume: true,
  });

  const projection = await waitForProjection(
    options,
    sessionId,
    taskId,
    expectedToolName,
  );
  return {
    mode: "live",
    workspaceId,
    sessionId,
    taskId,
    turnId,
    expectedToolName,
    providerPreference,
    projection,
  };
}

async function runReplay(options) {
  const expectedToolName = toolNameFor(options.connectorId, options.action);
  const projection = await readProjection(
    options,
    options.sessionId,
    options.taskId,
  );
  return {
    mode: "replay",
    sessionId: options.sessionId,
    taskId: options.taskId,
    expectedToolName,
    projection,
  };
}

function buildSummary(options, health, result, localWebhook = null) {
  const { threadRead, task } = result.projection;
  const toolCalls = getToolCalls(threadRead);
  const events = getTaskEvents(task);
  const toolEvents = events.filter((event) =>
    eventType(event).includes("toolCall"),
  );
  const evidenceEvents = events.filter((event) =>
    eventType(event).includes("evidence"),
  );
  const namedToolCalls = toolCalls.filter(
    (toolCall) =>
      (toolCall.tool_name || toolCall.toolName) === result.expectedToolName,
  );
  const expectedRefHint =
    options.expectedRef ||
    collectOutboxRefs(
      ...namedToolCalls.map((toolCall) => getToolEvidenceRefs(toolCall)),
      getEvidenceSummaryRefs(threadRead),
      collectTaskEvidenceRefs(events),
    )[0] ||
    "";
  const matchingToolCall =
    namedToolCalls.find((toolCall) =>
      expectedRefHint
        ? getToolEvidenceRefs(toolCall).includes(expectedRefHint)
        : false,
    ) ||
    namedToolCalls.find(
      (toolCall) =>
        toolCall.status === "completed" && toolCall.success !== false,
    ) ||
    namedToolCalls[0] ||
    toolCalls[0] ||
    null;
  const toolEvidenceRefs = getToolEvidenceRefs(matchingToolCall);
  const threadEvidenceRefs = getEvidenceSummaryRefs(threadRead);
  const taskEvidenceRefs = collectTaskEvidenceRefs(events);
  const outboxRefs = collectOutboxRefs(
    toolEvidenceRefs,
    threadEvidenceRefs,
    taskEvidenceRefs,
  );
  const expectedRef = expectedRefHint || buildExpectedRef(options, outboxRefs);
  const output = toolOutput(matchingToolCall);
  const outputPayload = parseToolOutputPayload(output);
  const secretDelivery = outputPayload?.secretDelivery || {};
  const delivery = outputPayload?.delivery || {};
  const productionDelivery = outputPayload?.productionDelivery || {};
  const externalDelivery = delivery?.externalDelivery || {};
  const deliveryReceiptRef =
    typeof delivery?.receiptRef === "string" ? delivery.receiptRef : "";
  const outputContainsBoundedMetadata =
    output.includes(TOOL_METADATA_BEGIN) && output.includes(TOOL_METADATA_END);
  const outputContainsTrustedConnectorJson =
    isTrustedConnectorOutputPayload(outputPayload);
  const outputPayloadText = JSON.stringify(outputPayload ?? {});
  const publicProjectionText = JSON.stringify({
    threadRead,
    task,
  });
  const assertions = {
    devBridgeHealthy: health?.status === "ok" || Boolean(health),
    threadReadCompleted: threadRead?.status === "completed",
    connectorToolCallProjected:
      (matchingToolCall?.tool_name || matchingToolCall?.toolName) ===
      result.expectedToolName,
    outputEvidenceSourceObserved:
      outputContainsBoundedMetadata || outputContainsTrustedConnectorJson,
    outboxEvidenceRefObserved: Boolean(expectedRef),
    toolCallEvidenceProjected: expectedRef
      ? toolEvidenceRefs.includes(expectedRef)
      : false,
    threadEvidenceProjected: expectedRef
      ? threadEvidenceRefs.includes(expectedRef)
      : false,
    taskEventEvidenceProjected: expectedRef
      ? taskEvidenceRefs.includes(expectedRef)
      : false,
    secretDeliveryConcreteLeaseRefNotExposed:
      !output.includes("secret-lease://connector/") &&
      !outputPayloadText.includes("secret-lease://connector/") &&
      !publicProjectionText.includes("secret-lease://connector/"),
  };
  const expectations = {};
  addExpectedAssertion(
    assertions,
    expectations,
    "adapterReadinessExpected",
    outputPayload?.adapterReadiness,
    options.expectAdapterReadiness,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "externalStatusExpected",
    outputPayload?.externalStatus,
    options.expectExternalStatus,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "nextRequiredExpected",
    outputPayload?.next?.required,
    options.expectNextRequired,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "secretDeliveryStatusExpected",
    secretDelivery?.status,
    options.expectSecretDeliveryStatus,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "secretDeliverySourceExpected",
    secretDelivery?.source,
    options.expectSecretDeliverySource,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "secretDeliveryTargetExpected",
    secretDelivery?.target,
    options.expectSecretDeliveryTarget,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "secretDeliveryLeaseObservedExpected",
    secretDelivery?.leaseObserved,
    options.expectSecretDeliveryLeaseObserved,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "secretDeliveryLeaseRefExposedExpected",
    secretDelivery?.leaseRefExposed,
    options.expectSecretDeliveryLeaseRefExposed,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "secretDeliveryLeaseHandleStatusExpected",
    secretDelivery?.leaseHandleStatus,
    options.expectSecretDeliveryLeaseHandleStatus,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "secretDeliveryCredentialMaterialExposureExpected",
    secretDelivery?.credentialMaterialExposed,
    options.expectSecretDeliveryCredentialMaterialExposed,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "secretDeliveryTokenExposureExpected",
    secretDelivery?.tokenExposed,
    options.expectSecretDeliveryTokenExposed,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "deliveryStatusExpected",
    delivery?.status,
    options.expectDeliveryStatus,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "deliveryExternalPlatformDeliveredExpected",
    delivery?.externalPlatformDelivered,
    options.expectDeliveryExternalPlatformDelivered,
  );
  addExpectedAssertion(
    assertions,
    expectations,
    "productionPlatformDeliveredExpected",
    productionDelivery?.productionPlatformDelivered,
    options.expectProductionPlatformDelivered,
  );
  if (
    options.expectDeliveryStatus ||
    options.expectDeliveryExternalPlatformDelivered
  ) {
    assertions.deliveryReceiptRefObserved =
      deliveryReceiptRef.startsWith("delivery://");
    assertions.deliveryToolCallEvidenceProjected = deliveryReceiptRef
      ? toolEvidenceRefs.includes(deliveryReceiptRef)
      : false;
    assertions.deliveryThreadEvidenceProjected = deliveryReceiptRef
      ? threadEvidenceRefs.includes(deliveryReceiptRef)
      : false;
    assertions.deliveryTaskEventEvidenceProjected = deliveryReceiptRef
      ? taskEvidenceRefs.includes(deliveryReceiptRef)
      : false;
  }
  if (options.externalDeliveryWebhookUrl) {
    assertions.externalDeliveryTargetNotExposed =
      !output.includes(options.externalDeliveryWebhookUrl) &&
      !outputPayloadText.includes(options.externalDeliveryWebhookUrl) &&
      !publicProjectionText.includes(options.externalDeliveryWebhookUrl);
  }
  if (options.externalDeliveryLocalWebhook) {
    assertions.externalDeliveryLocalWebhookReceived =
      (localWebhook?.requests?.length || 0) > 0;
  }
  const deliveryEvidenceRefs = collectDeliveryRefs(
    toolEvidenceRefs,
    threadEvidenceRefs,
    taskEvidenceRefs,
  );

  return {
    kind: "p18-7-e-connector-outbox-runtime-smoke",
    generatedAt: new Date().toISOString(),
    mode: result.mode,
    health,
    appId: options.appId,
    sessionId: result.sessionId,
    taskId: result.taskId,
    expectedToolName: result.expectedToolName,
    expectedRef,
    expectations,
    providerPreference: result.providerPreference || null,
    threadRead: {
      status: threadRead?.status || null,
      profileStatus:
        threadRead?.profile_status || threadRead?.profileStatus || null,
      toolCallCount: toolCalls.length,
      firstToolName:
        matchingToolCall?.tool_name || matchingToolCall?.toolName || null,
      toolEvidenceRefs,
      evidenceSummaryRefs: threadEvidenceRefs,
      outputContainsBoundedMetadata,
      outputContainsTrustedConnectorJson,
      adapterReadiness: outputPayload?.adapterReadiness || null,
      externalStatus: outputPayload?.externalStatus || null,
      nextRequired: outputPayload?.next?.required || null,
      secretDeliveryStatus: secretDelivery?.status || null,
      secretDeliverySource: secretDelivery?.source || null,
      secretDeliveryTarget: secretDelivery?.target || null,
      secretDeliveryLeaseObserved: secretDelivery?.leaseObserved ?? null,
      secretDeliveryLeaseRefExposed: secretDelivery?.leaseRefExposed ?? null,
      secretDeliveryLeaseHandleStatus:
        secretDelivery?.leaseHandleStatus || null,
      secretDeliveryConcreteLeaseRefObserved:
        typeof secretDelivery?.leaseRef === "string",
      secretDeliveryCredentialMaterialExposed:
        secretDelivery?.credentialMaterialExposed ?? null,
      secretDeliveryTokenExposed: secretDelivery?.tokenExposed ?? null,
      deliveryStatus: delivery?.status || null,
      deliveryReceiptRef: deliveryReceiptRef || null,
      deliveryEvidenceRefs,
      deliveryExternalPlatformDelivered:
        delivery?.externalPlatformDelivered ?? null,
      productionDeliveryStatus: productionDelivery?.status || null,
      productionDeliveryProofLevel: productionDelivery?.proofLevel || null,
      productionDeliveryNextRequired: productionDelivery?.nextRequired || null,
      productionPlatformDelivered:
        productionDelivery?.productionPlatformDelivered ?? null,
      externalDeliveryChannel: externalDelivery?.channel || null,
      externalDeliveryTargetHash: externalDelivery?.targetHash || null,
      externalDeliveryTargetLabel: externalDelivery?.targetLabel || null,
      externalDeliveryTargetExposed: externalDelivery?.targetExposed ?? null,
      externalDeliveryProofLevel: externalDelivery?.proofLevel || null,
      externalDeliveryProductionPlatformDelivered:
        externalDelivery?.productionPlatformDelivered ?? null,
      externalDeliveryHttpStatus: externalDelivery?.httpStatus ?? null,
    },
    externalDeliveryWebhook: options.externalDeliveryWebhookUrl
      ? {
          configured: true,
          targetSource: options.externalDeliveryWebhookSource || null,
          localServer: Boolean(options.externalDeliveryLocalWebhook),
          receivedRequestCount: localWebhook?.requests?.length || 0,
        }
      : null,
    pluginTask: {
      status: task?.status || null,
      profileStatus: task?.profileStatus || task?.profile_status || null,
      eventCount: events.length,
      toolEventCount: toolEvents.length,
      evidenceEventCount: evidenceEvents.length,
      evidenceRefs: taskEvidenceRefs,
    },
    assertions,
    caveats: [
      result.mode === "replay"
        ? "Replay mode reads an existing runtime session and does not call the model provider."
        : "Live mode submits a new AgentRuntime turn and may call the configured model provider.",
      options.externalDeliveryWebhookUrl
        ? "This proves runtime outbox/evidence projection, Host-managed secret-delivery facts, and Host-managed HTTP webhook delivery; it does not prove external OAuth handshake or raw secret material exposure to the App/model."
        : "This proves runtime outbox/evidence projection, Host-managed secret-delivery facts, and local cloud-overlay worker intake receipts; it does not prove external OAuth handshake, raw secret material exposure to the App/model, or external platform delivery.",
      options.externalDeliveryWebhookUrl
        ? "When a webhook is configured, this proves only Host-managed HTTP webhook delivery; it is not proof of Notion/Slack production delivery."
        : null,
    ].filter(Boolean),
  };
}

function writeSummary(output, summary) {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(summary, null, 2)}\n`);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.mode === "live") {
    assertLiveProviderSmokeAllowed({
      allowed: options.allowLiveProvider,
      scriptName: "scripts/plugin/connector-outbox-smoke.mjs --mode live",
    });
  }
  const localWebhook =
    options.mode === "live" && options.externalDeliveryLocalWebhook
      ? await startLocalWebhookServer()
      : null;
  if (localWebhook) {
    options.externalDeliveryWebhookUrl = localWebhook.url;
    options.externalDeliveryWebhookSource = "local";
    options.externalDeliveryWebhookLabel ||=
      "local-connector-outbox-smoke-webhook";
  }
  let summary;
  try {
    const health = await waitForHealth(options);
    const result =
      options.mode === "live"
        ? await runLive(options)
        : await runReplay(options);
    summary = buildSummary(options, health, result, localWebhook);
    writeSummary(options.output, summary);
  } finally {
    await localWebhook?.close();
  }

  console.log(
    JSON.stringify(
      {
        output: options.output,
        mode: summary.mode,
        assertions: summary.assertions,
      },
      null,
      2,
    ),
  );

  const failed = Object.entries(summary.assertions).filter(
    ([, value]) => !value,
  );
  if (failed.length > 0) {
    throw new Error(
      `connector outbox smoke assertions failed: ${failed
        .map(([key]) => key)
        .join(", ")}`,
    );
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
