#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { startOpenAiCompatibleFixtureServer } from "../lib/openai-compatible-fixture-server.mjs";
import {
  closeElectronFixture,
  createTempRuntimeEnv,
  launchElectronFixture,
} from "../electron/mcp-config-fixture-smoke.mjs";
import {
  ensureDefaultWorkspace,
  initializeAppServer,
  invokeAppServerFromPage,
} from "./claw-chat-current-fixture-rpc.mjs";

const LOG_PREFIX = "[smoke:provider-generation-pending-route-gate-b]";
const DEFAULT_OUTPUT = path.resolve(
  ".lime/qc/provider-generation-pending-route-gate-b.json",
);
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_INTERVAL_MS = 250;
const PROVIDER_NAME = "Pending Route Gate B";
const MODEL_NAME = "pending-route-fixture-model";
const PARENT_MARKER = "P0_05_PROVIDER_GENERATION_PARENT";
const CHILD_MARKER = "P0_05_PROVIDER_GENERATION_CHILD";
const PARENT_DONE = "P0_05_PROVIDER_GENERATION_PARENT_DONE";
const CHILD_DONE = "P0_05_PROVIDER_GENERATION_CHILD_DONE";
const SPAWN_CALL_ID = "call-p0-05-provider-generation-spawn";
const SPAWN_TASK_NAME = "pending_route_child";
const ROUTE_GENERATION_SQL =
  "SELECT value FROM settings WHERE key = 'model_route_generation';";
const NAVIGATION_RESTORE_STORAGE_KEY = "lime.appNavigation.restore.v1";
const INVOKE_TRACE_STORAGE_KEY = "lime_invoke_trace_buffer_v1";
const INVOKE_ERROR_STORAGE_KEY = "lime_invoke_error_buffer_v1";
const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "canceled",
  "cancelled",
  "aborted",
]);

function usage() {
  return `
Provider-generation PendingRoute cold-restart Gate B

Usage:
  node scripts/agent-runtime/provider-generation-pending-route-gate-b.mjs [options]

Options:
  --output <path>       Evidence JSON path
  --timeout-ms <ms>     Overall wait timeout, default ${DEFAULT_TIMEOUT_MS}
  --interval-ms <ms>    Poll interval, default ${DEFAULT_INTERVAL_MS}
  --cleanup-temp        Delete the isolated runtime directory after the run
  -h, --help            Show this help
`;
}

export function parseArgs(argv) {
  const options = {
    output: DEFAULT_OUTPUT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    intervalMs: DEFAULT_INTERVAL_MS,
    cleanupTemp: false,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    if (arg === "--output" && argv[index + 1]) {
      options.output = path.resolve(String(argv[index + 1]));
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && argv[index + 1]) {
      options.timeoutMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && argv[index + 1]) {
      options.intervalMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--cleanup-temp") {
      options.cleanupTemp = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be >= 100");
  }
  return options;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function sha256(value) {
  return createHash("sha256").update(String(value)).digest("hex");
}

function stableAgentDigest(parts) {
  return sha256(parts.join("\u001f"));
}

export function deriveDurableIdentity({ parentThreadId, parentTurnId }) {
  const childSessionId = `agent-${stableAgentDigest([
    parentThreadId,
    parentThreadId,
    parentTurnId,
    SPAWN_CALL_ID,
    "session",
  ])}`;
  const childThreadId = `thread-${stableAgentDigest([
    parentThreadId,
    parentThreadId,
    parentTurnId,
    SPAWN_CALL_ID,
    "thread",
  ])}`;
  const messageId = `agent-control-message-${stableAgentDigest([
    parentThreadId,
    parentThreadId,
    parentTurnId,
    SPAWN_CALL_ID,
    "spawn_agent",
    childThreadId,
  ])}`;
  const mailboxDigest = sha256(messageId);
  return {
    childSessionId,
    childThreadId,
    messageId,
    mailboxTurnId: `mailbox-turn-${mailboxDigest}`,
    mailboxItemId: `item_mailbox-item-${mailboxDigest}`,
  };
}

export function electronCallsFromRequestLog(requestLog) {
  return (Array.isArray(requestLog) ? requestLog : []).flatMap((entry) => {
    if (typeof entry?.method !== "string") return [];
    const completed = Object.prototype.hasOwnProperty.call(entry, "response");
    const failed = Object.prototype.hasOwnProperty.call(entry, "error");
    if (!completed && !failed) return [];
    return [
      {
        method: entry.method,
        transport: "electron-ipc",
        status: completed ? "success" : "error",
      },
    ];
  });
}

function messageText(content) {
  if (typeof content === "string") return content;
  return JSON.stringify(content ?? "");
}

function requestUserText(body) {
  return (Array.isArray(body?.messages) ? body.messages : [])
    .filter((message) => message?.role === "user")
    .map((message) => messageText(message?.content))
    .join("\n");
}

function requestToolCallNames(body) {
  return (Array.isArray(body?.messages) ? body.messages : []).flatMap(
    (message) =>
      (Array.isArray(message?.tool_calls) ? message.tool_calls : [])
        .map((call) => String(call?.function?.name || "").trim())
        .filter(Boolean),
  );
}

export function buildProviderScriptedResponse(context) {
  const body = context?.body;
  const userText = requestUserText(body);
  if (userText.includes(CHILD_MARKER)) {
    return { type: "text", content: CHILD_DONE };
  }
  if (!requestToolCallNames(body).includes("spawn_agent")) {
    return {
      type: "tool_call",
      id: SPAWN_CALL_ID,
      name: "spawn_agent",
      arguments: {
        task_name: SPAWN_TASK_NAME,
        message: `${CHILD_MARKER}: complete the durable child turn without tools.`,
        fork_turns: "none",
      },
    };
  }
  return { type: "text", content: PARENT_DONE };
}

export function createProviderResponseController() {
  let parentRequestBlocked = false;
  let parentRequestReleased = false;
  let parentPauseCount = 0;
  let releaseParentRequest;
  const parentRelease = new Promise((resolve) => {
    releaseParentRequest = resolve;
  });

  return {
    async respond(context) {
      const response = buildProviderScriptedResponse(context);
      if (
        response.type === "tool_call" &&
        response.name === "spawn_agent" &&
        !parentRequestBlocked
      ) {
        parentRequestBlocked = true;
        parentPauseCount += 1;
        await parentRelease;
      }
      return response;
    },
    releaseParentRequest() {
      if (parentRequestReleased) return;
      parentRequestReleased = true;
      releaseParentRequest();
    },
    snapshot() {
      return {
        parentRequestBlocked,
        parentRequestReleased,
        parentPauseCount,
      };
    },
  };
}

function readRouteGeneration(databasePath) {
  const sqliteBinary = process.env.SQLITE3_BIN?.trim() || "sqlite3";
  const raw = execFileSync(sqliteBinary, [databasePath, ROUTE_GENERATION_SQL], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  }).trim();
  const generation = Number(raw || "0");
  assert(
    Number.isSafeInteger(generation) && generation >= 0,
    `Invalid model route generation: ${raw}`,
  );
  return generation;
}

function collectObjects(value, target = []) {
  if (!value || typeof value !== "object") return target;
  target.push(value);
  for (const child of Array.isArray(value) ? value : Object.values(value)) {
    collectObjects(child, target);
  }
  return target;
}

function stringField(record, keys) {
  for (const key of keys) {
    const value = record?.[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

export function canonicalRecordsWithId(payload, expectedId) {
  return collectObjects(payload).filter(
    (record) => stringField(record, ["id"]) === expectedId,
  );
}

function terminalCanonicalRecordCount(payload, expectedId) {
  return canonicalRecordsWithId(payload, expectedId).filter((record) =>
    TERMINAL_STATUSES.has(stringField(record, ["status"]).toLowerCase()),
  ).length;
}

export function hasExactlyOneCanonicalRecord(payload, expectedId) {
  return canonicalRecordsWithId(payload, expectedId).length === 1;
}

export function hasExactlyOneTerminalCanonicalRecord(payload, expectedId) {
  const records = canonicalRecordsWithId(payload, expectedId);
  return (
    records.length === 1 &&
    TERMINAL_STATUSES.has(stringField(records[0], ["status"]).toLowerCase())
  );
}

function providerRequestCount(requests, marker) {
  return requests.filter((request) =>
    requestUserText(request?.body).includes(marker),
  ).length;
}

async function waitFor(label, options, probe) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      last = await probe();
      if (last?.ready) return last;
    } catch (error) {
      last = { error: error instanceof Error ? error.message : String(error) };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(`${label} timed out: ${JSON.stringify(last)}`);
}

export function buildV2ThreadReadParams(threadId) {
  return { threadId, includeTurns: true };
}

export function buildV2TurnStartParams({
  clientUserMessageId,
  threadId,
  workspaceRoot,
}) {
  return {
    threadId,
    clientUserMessageId,
    input: [
      {
        type: "text",
        text: `${PARENT_MARKER}: spawn one durable child and then finish.`,
      },
    ],
    cwd: workspaceRoot,
    runtimeWorkspaceRoots: [workspaceRoot],
    model: MODEL_NAME,
    approvalPolicy: "never",
    sandboxPolicy: "danger-full-access",
    responsesapiClientMetadata: {
      source: "provider-generation-pending-route-gate-b",
    },
  };
}

async function readThread(page, threadId, requestLog) {
  return (
    await invokeAppServerFromPage(
      page,
      "thread/read",
      buildV2ThreadReadParams(threadId),
      requestLog,
    )
  ).result;
}

async function createRepositoryProviderWithKey(page, fixture, requestLog) {
  const created = await invokeAppServerFromPage(
    page,
    "modelProvider/create",
    {
      name: PROVIDER_NAME,
      providerType: "openai",
      apiHost: fixture.baseUrl,
    },
    requestLog,
  );
  const providerId = String(created.result?.provider?.id || "").trim();
  assert(providerId, "modelProvider/create did not return provider.id");
  await updateRepositoryProvider(page, providerId, requestLog);
  const key = await invokeAppServerFromPage(
    page,
    "modelProviderKey/create",
    {
      providerId,
      apiKey: fixture.provider.providerConfig.apiKey,
      alias: "pending-route-gate-b-initial",
      replaceExisting: true,
    },
    requestLog,
  );
  const keyId = String(key.result?.key?.id || "").trim();
  assert(keyId, "modelProviderKey/create did not return key.id");
  return { keyId, providerId };
}

async function updateRepositoryProvider(page, providerId, requestLog) {
  return await invokeAppServerFromPage(
    page,
    "modelProvider/update",
    {
      providerId,
      enabled: true,
      sortOrder: 1,
      models: [{ id: MODEL_NAME }],
    },
    requestLog,
  );
}

async function createParentSession(page, workspace, providerId, requestLog) {
  const started = await invokeAppServerFromPage(
    page,
    "thread/start",
    {
      cwd: workspace.rootPath,
      historyMode: "paginated",
      model: MODEL_NAME,
      modelProvider: providerId,
      runtimeWorkspaceRoots: [workspace.rootPath],
      serviceName: "Pending route Gate B",
      threadSource: "appServer",
    },
    requestLog,
  );
  const sessionId = String(started.result?.thread?.sessionId || "").trim();
  const threadId = String(started.result?.thread?.id || "").trim();
  assert(
    sessionId && threadId,
    "thread/start did not return canonical identity",
  );
  return { sessionId, threadId };
}

async function startParentTurn({ identity, page, requestLog, workspaceRoot }) {
  const response = await invokeAppServerFromPage(
    page,
    "turn/start",
    buildV2TurnStartParams({
      clientUserMessageId: `p0-05-provider-generation-${Date.now()}-${process.pid}`,
      threadId: identity.threadId,
      workspaceRoot,
    }),
    requestLog,
  );
  const turnId = String(response.result?.turn?.id || "").trim();
  assert(turnId, "turn/start did not return canonical turn.id");
  return turnId;
}

async function launch(options, runtimeEnv, appServerEnv, errors) {
  return await launchElectronFixture({
    options,
    runtimeEnv,
    appServerEnv,
    consoleErrors: errors.console,
    pageErrors: errors.page,
    backendMode: "runtime",
  });
}

async function initialize(page, requestLog) {
  await initializeAppServer(page, requestLog);
  return await ensureDefaultWorkspace(page, requestLog);
}

async function restoreChildInGui(page, childSessionId, options) {
  await page.evaluate(
    ({ errorKey, navigationKey, sessionId, traceKey }) => {
      localStorage.removeItem(errorKey);
      localStorage.removeItem(traceKey);
      sessionStorage.setItem(
        navigationKey,
        JSON.stringify({
          page: "agent",
          params: { initialSessionId: sessionId },
        }),
      );
    },
    {
      errorKey: INVOKE_ERROR_STORAGE_KEY,
      navigationKey: NAVIGATION_RESTORE_STORAGE_KEY,
      sessionId: childSessionId,
      traceKey: INVOKE_TRACE_STORAGE_KEY,
    },
  );
  await page.reload({
    waitUntil: "domcontentloaded",
    timeout: options.timeoutMs,
  });
  const input = page.locator(
    `textarea[name="agent-chat-message"][data-session-id="${childSessionId}"]`,
  );
  await input.waitFor({ state: "visible", timeout: options.timeoutMs });
  const finalText = page.getByText(CHILD_DONE, { exact: false }).first();
  await finalText.waitFor({ state: "visible", timeout: options.timeoutMs });
  return {
    electron: await page.evaluate(() => window.__LIME_ELECTRON__ === true),
    activeChildSession:
      (await input.getAttribute("data-session-id")) === childSessionId,
    childTerminalTextVisible: await finalText.isVisible(),
  };
}

async function readInvokeDiagnostics(page) {
  return await page.evaluate(
    ({ errorKey, traceKey }) => {
      const readArray = (key) => {
        try {
          const parsed = JSON.parse(localStorage.getItem(key) || "[]");
          return Array.isArray(parsed) ? parsed : [];
        } catch {
          return [];
        }
      };
      const traces = readArray(traceKey);
      const calls = [];
      for (const entry of traces) {
        if (entry?.command !== "app_server_handle_json_lines") continue;
        const lines = Array.isArray(entry?.args_preview?.request?.lines)
          ? entry.args_preview.request.lines
          : [];
        for (const line of lines) {
          try {
            const message = typeof line === "string" ? JSON.parse(line) : line;
            if (typeof message?.method !== "string") continue;
            calls.push({
              method: message.method,
              transport: String(entry?.transport || ""),
              status: String(entry?.status || ""),
            });
          } catch {
            // A malformed diagnostic preview is not product evidence.
          }
        }
      }
      return {
        calls,
        invokeErrorCount: readArray(errorKey).length,
        mockFallbackHitCount: traces.filter((entry) => {
          if (entry?.mock === true || entry?.mockFallback === true) return true;
          return [
            entry?.transport,
            entry?.source,
            entry?.fallback,
            entry?.fallbackMode,
          ].some(
            (value) =>
              typeof value === "string" && value.toLowerCase().includes("mock"),
          );
        }).length,
      };
    },
    { errorKey: INVOKE_ERROR_STORAGE_KEY, traceKey: INVOKE_TRACE_STORAGE_KEY },
  );
}

function methodObserved(calls, method) {
  return calls.some(
    (call) =>
      call.method === method &&
      call.transport === "electron-ipc" &&
      call.status === "success",
  );
}

function writeEvidence(outputPath, evidence) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(evidence, null, 2)}\n`);
}

export async function runGateB(options) {
  const runtimeEnv = createTempRuntimeEnv();
  const databasePath = path.join(runtimeEnv.appServerDataDir, "lime.db");
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: { ...runtimeEnv.env, APP_SERVER_BIN: appServerBinary },
  });
  const providerResponses = createProviderResponseController();
  const fixture = await startOpenAiCompatibleFixtureServer({
    model: MODEL_NAME,
    scriptedResponses: [(context) => providerResponses.respond(context)],
  });
  const requestLog = [];
  const errors = { console: [], page: [] };
  let handle = null;

  try {
    console.log(`${LOG_PREFIX} stage=launch-electron`);
    handle = await launch(options, runtimeEnv, appServerEnv, errors);
    let page = handle.page;
    let workspace = await initialize(page, requestLog);

    console.log(`${LOG_PREFIX} stage=create-provider-with-first-key`);
    const { keyId: initialKeyId, providerId } =
      await createRepositoryProviderWithKey(page, fixture, requestLog);
    const generationWithFirstKey = readRouteGeneration(databasePath);
    const parent = await createParentSession(
      page,
      workspace,
      providerId,
      requestLog,
    );
    const parentTurnId = await startParentTurn({
      identity: parent,
      page,
      requestLog,
      workspaceRoot: workspace.rootPath,
    });
    const durable = deriveDurableIdentity({
      parentThreadId: parent.threadId,
      parentTurnId,
    });

    console.log(`${LOG_PREFIX} stage=wait-parent-provider-pause`);
    const pausedParent = await waitFor(
      "parent provider request pause",
      options,
      async () => {
        const pause = providerResponses.snapshot();
        return {
          ready:
            pause.parentRequestBlocked &&
            providerRequestCount(fixture.requests, PARENT_MARKER) === 1,
          ...pause,
          childRequestCount: providerRequestCount(
            fixture.requests,
            CHILD_MARKER,
          ),
        };
      },
    );
    const pausedParentRequest = fixture.requests.find((request) =>
      requestUserText(request?.body).includes(PARENT_MARKER),
    );
    const parentRequestAuthorizedBeforeDelete =
      pausedParentRequest?.authorization ===
      `Bearer ${fixture.provider.providerConfig.apiKey}`;

    console.log(`${LOG_PREFIX} stage=delete-first-key-over-electron-ipc`);
    await invokeAppServerFromPage(
      page,
      "modelProviderKey/delete",
      { keyId: initialKeyId },
      requestLog,
    );
    const generationAfterDelete = readRouteGeneration(databasePath);
    const providerWithoutKey = await invokeAppServerFromPage(
      page,
      "modelProvider/read",
      { providerId },
      requestLog,
    );
    const apiKeyCountAfterDelete = Number(
      providerWithoutKey.result?.provider?.apiKeyCount ?? -1,
    );
    assert(apiKeyCountAfterDelete === 0, "provider key delete did not commit");
    providerResponses.releaseParentRequest();

    const beforeRestart = await waitFor(
      "parent completion and pending child",
      options,
      async () => {
        const [parentRead, childRead, childList] = await Promise.all([
          readThread(page, parent.threadId, requestLog),
          readThread(page, durable.childThreadId, requestLog),
          invokeAppServerFromPage(
            page,
            "thread/list",
            { parentThreadId: parent.threadId, limit: 20 },
            requestLog,
          ).then((response) => response.result),
        ]);
        const parentTerminal = JSON.stringify(parentRead).includes(PARENT_DONE);
        const childListed = JSON.stringify(childList).includes(
          durable.childThreadId,
        );
        const childRequestCount = providerRequestCount(
          fixture.requests,
          CHILD_MARKER,
        );
        return {
          ready: parentTerminal && childListed && childRequestCount === 0,
          parentTerminal,
          childListed,
          childRequestCount,
          pendingChildTurnAbsent:
            canonicalRecordsWithId(childRead, durable.mailboxTurnId).length ===
            0,
        };
      },
    );
    const preRestartDiagnostics = await readInvokeDiagnostics(page);

    console.log(`${LOG_PREFIX} stage=cold-restart-electron`);
    const firstElectronPid = handle.app.process().pid;
    await closeElectronFixture(handle);
    handle = await launch(options, runtimeEnv, appServerEnv, errors);
    page = handle.page;
    workspace = await initialize(page, requestLog);
    const restartedElectronPid = handle.app.process().pid;
    const generationAfterRestart = readRouteGeneration(databasePath);
    const childAfterRestart = await readThread(
      page,
      durable.childThreadId,
      requestLog,
    );

    console.log(`${LOG_PREFIX} stage=idempotent-provider-update`);
    await updateRepositoryProvider(page, providerId, requestLog);
    const generationAfterIdempotentUpdate = readRouteGeneration(databasePath);
    const childBeforeCredential = await readThread(
      page,
      durable.childThreadId,
      requestLog,
    );

    console.log(`${LOG_PREFIX} stage=recreate-key`);
    const recreatedKey = await invokeAppServerFromPage(
      page,
      "modelProviderKey/create",
      {
        providerId,
        apiKey: fixture.provider.providerConfig.apiKey,
        alias: "pending-route-gate-b",
        replaceExisting: true,
      },
      requestLog,
    );
    const recreatedKeyId = String(recreatedKey.result?.key?.id || "").trim();
    assert(recreatedKeyId, "recreated provider key is missing key.id");
    const committedGeneration = readRouteGeneration(databasePath);
    const afterCredential = await waitFor(
      "child recovery after credential commit",
      options,
      async () => {
        const childRead = await readThread(
          page,
          durable.childThreadId,
          requestLog,
        );
        const childRequestCount = providerRequestCount(
          fixture.requests,
          CHILD_MARKER,
        );
        const mailboxTurnTerminalCount = terminalCanonicalRecordCount(
          childRead,
          durable.mailboxTurnId,
        );
        return {
          ready:
            childRequestCount === 1 &&
            mailboxTurnTerminalCount >= 1 &&
            JSON.stringify(childRead).includes(CHILD_DONE),
          childRead,
          childRequestCount,
          mailboxTurnTerminalCount,
        };
      },
    );

    await updateRepositoryProvider(page, providerId, requestLog);
    const finalGeneration = readRouteGeneration(databasePath);
    const finalChildRead = await readThread(
      page,
      durable.childThreadId,
      requestLog,
    );
    const finalChildRequestCount = providerRequestCount(
      fixture.requests,
      CHILD_MARKER,
    );
    const mailboxTurnRecords = canonicalRecordsWithId(
      finalChildRead,
      durable.mailboxTurnId,
    );
    const mailboxItemRecords = canonicalRecordsWithId(
      finalChildRead,
      durable.mailboxItemId,
    );

    const runtimeDiagnostics = await readInvokeDiagnostics(page);
    console.log(`${LOG_PREFIX} stage=visible-child-terminal`);
    const gui = await restoreChildInGui(page, durable.childSessionId, options);
    const guiDiagnostics = await readInvokeDiagnostics(page);
    const diagnostics = {
      calls: [
        ...electronCallsFromRequestLog(requestLog),
        ...preRestartDiagnostics.calls,
        ...runtimeDiagnostics.calls,
        ...guiDiagnostics.calls,
      ],
      invokeErrorCount:
        preRestartDiagnostics.invokeErrorCount +
        runtimeDiagnostics.invokeErrorCount +
        guiDiagnostics.invokeErrorCount,
      mockFallbackHitCount:
        preRestartDiagnostics.mockFallbackHitCount +
        runtimeDiagnostics.mockFallbackHitCount +
        guiDiagnostics.mockFallbackHitCount,
    };

    const assertions = {
      realElectronHost: gui.electron === true,
      electronProcessReplaced: firstElectronPid !== restartedElectronPid,
      parentProviderRequestPausedBeforeKeyDelete:
        pausedParent.parentRequestBlocked &&
        pausedParent.parentPauseCount === 1 &&
        pausedParent.childRequestCount === 0,
      parentRequestUsedFirstCredential: parentRequestAuthorizedBeforeDelete,
      firstCredentialDeletedBeforeSpawnRelease:
        apiKeyCountAfterDelete === 0 &&
        generationAfterDelete === generationWithFirstKey + 1,
      providerGenerationSurvivedRestart:
        generationAfterRestart === generationAfterDelete,
      idempotentUpdateDidNotAdvanceGeneration:
        generationAfterIdempotentUpdate === generationAfterRestart,
      credentialCommitAdvancedOneGeneration:
        committedGeneration === generationAfterIdempotentUpdate + 1,
      recreatedCredentialUsesNewIdentity: recreatedKeyId !== initialKeyId,
      laterIdempotentUpdateDidNotAdvanceGeneration:
        finalGeneration === committedGeneration,
      childIdentitySurvivedRestart: JSON.stringify(childAfterRestart).includes(
        durable.childThreadId,
      ),
      triggerMailStayedUnmaterializedBeforeCredential:
        canonicalRecordsWithId(childBeforeCredential, durable.mailboxTurnId)
          .length === 0,
      noChildProviderRequestBeforeCredential:
        beforeRestart.childRequestCount === 0,
      childProviderRequestExactlyOnce: finalChildRequestCount === 1,
      mailboxTurnTerminalExactlyOnce: hasExactlyOneTerminalCanonicalRecord(
        finalChildRead,
        durable.mailboxTurnId,
      ),
      mailboxItemMaterializedExactlyOnce: hasExactlyOneCanonicalRecord(
        finalChildRead,
        durable.mailboxItemId,
      ),
      childTerminalVisibleInGui:
        gui.activeChildSession && gui.childTerminalTextVisible,
      threadReadObservedThroughElectronIpc: methodObserved(
        diagnostics.calls,
        "thread/read",
      ),
      providerUpdateObservedThroughElectronIpc: methodObserved(
        diagnostics.calls,
        "modelProvider/update",
      ),
      credentialCommitObservedThroughElectronIpc: methodObserved(
        diagnostics.calls,
        "modelProviderKey/create",
      ),
      credentialDeleteObservedThroughElectronIpc: methodObserved(
        diagnostics.calls,
        "modelProviderKey/delete",
      ),
      mockFallbackClear: diagnostics.mockFallbackHitCount === 0,
      invokeErrorsClear: diagnostics.invokeErrorCount === 0,
      consoleErrorsClear: errors.console.length === 0,
      pageErrorsClear: errors.page.length === 0,
    };
    const failedAssertions = Object.entries(assertions)
      .filter(([, passed]) => passed !== true)
      .map(([name]) => name);
    const evidence = {
      schemaVersion: "lime.provider_generation_pending_route_gate_b.v1",
      status: failedAssertions.length === 0 ? "pass" : "fail",
      proofLevel: "Gate B",
      generatedAt: new Date().toISOString(),
      transport: {
        calls: diagnostics.calls,
        invokeErrorCount: diagnostics.invokeErrorCount,
        mockFallbackHitCount: diagnostics.mockFallbackHitCount,
      },
      generation: {
        withFirstCredential: generationWithFirstKey,
        afterCredentialDelete: generationAfterDelete,
        afterRestart: generationAfterRestart,
        afterIdempotentUpdate: generationAfterIdempotentUpdate,
        afterCredentialCommit: committedGeneration,
        final: finalGeneration,
      },
      identityHashes: {
        parentSession: sha256(parent.sessionId),
        parentThread: sha256(parent.threadId),
        parentTurn: sha256(parentTurnId),
        initialCredential: sha256(initialKeyId),
        recreatedCredential: sha256(recreatedKeyId),
        childSession: sha256(durable.childSessionId),
        childThread: sha256(durable.childThreadId),
        mailboxMessage: sha256(durable.messageId),
        mailboxTurn: sha256(durable.mailboxTurnId),
        mailboxItem: sha256(durable.mailboxItemId),
      },
      counts: {
        providerRequests: fixture.requests.length,
        parentProviderRequests: providerRequestCount(
          fixture.requests,
          PARENT_MARKER,
        ),
        childProviderRequests: finalChildRequestCount,
        mailboxTurnRecords: mailboxTurnRecords.length,
        mailboxItemRecords: mailboxItemRecords.length,
        consoleErrors: errors.console.length,
        pageErrors: errors.page.length,
      },
      lifecycle: {
        parentProviderRequestPausedBeforeDelete:
          pausedParent.parentRequestBlocked,
        parentProviderRequestAuthorizedBeforeDelete:
          parentRequestAuthorizedBeforeDelete,
        parentProviderRequestReleasedAfterDelete:
          providerResponses.snapshot().parentRequestReleased,
        providerCredentialAbsentBeforeSpawn: apiKeyCountAfterDelete === 0,
        pendingChildObservedBeforeRestart: beforeRestart.ready,
        pendingChildTurnAbsentBeforeRestart:
          beforeRestart.pendingChildTurnAbsent,
        pendingChildTurnAbsentAfterRestart:
          canonicalRecordsWithId(childAfterRestart, durable.mailboxTurnId)
            .length === 0,
        pendingChildTurnAbsentBeforeCredential:
          canonicalRecordsWithId(childBeforeCredential, durable.mailboxTurnId)
            .length === 0,
        recoveredChildObserved: afterCredential.ready,
        guiChildTerminalObserved:
          gui.activeChildSession && gui.childTerminalTextVisible,
      },
      assertions,
      failedAssertions,
    };
    writeEvidence(options.output, evidence);
    assert(
      failedAssertions.length === 0,
      `Gate B assertions failed: ${failedAssertions.join(", ")}`,
    );
    console.log(`${LOG_PREFIX} pass evidence=${options.output}`);
    return evidence;
  } finally {
    providerResponses.releaseParentRequest();
    await closeElectronFixture(handle);
    await fixture.close();
    if (options.cleanupTemp) {
      fs.rmSync(runtimeEnv.tempRoot, { recursive: true, force: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
  } else {
    runGateB(options).catch((error) => {
      console.error(
        `${LOG_PREFIX} failed: ${
          error instanceof Error ? error.stack || error.message : String(error)
        }`,
      );
      process.exitCode = 1;
    });
  }
}
