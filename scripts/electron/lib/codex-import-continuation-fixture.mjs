import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";

export const APP_SERVER_HANDLE_JSON_LINES_COMMAND =
  "app_server_handle_json_lines";
export const WORKSPACE_ID = "codex-import-continuation-workspace";
export const SOURCE_THREAD_ID = "codex-import-continuation-thread";
export const IMPORTED_TURN_ID = "codex-import-live-exec-turn";
export const NORMAL_TURN_ID = "codex-normal-live-exec-turn";
export const IMPORTED_USER_TEXT = "请运行测试并修复失败";
export const IMPORTED_REASONING_TEXT =
  "I need to inspect the test failure first.";
export const IMPORTED_ASSISTANT_TEXT = "已完成修复。";
export const IMPORTED_CONTINUE_TEXT = "在导入会话中执行统一命令并汇报结果";
export const NORMAL_USER_TEXT = "在普通会话中执行统一命令并汇报结果";
export const IMPORTED_FINAL_TEXT = "CODEX_IMPORTED_UNIFIED_EXEC_DONE";
export const NORMAL_FINAL_TEXT = "CODEX_NORMAL_UNIFIED_EXEC_DONE";
export const COMMAND_OUTPUT_MARKER = "CODEX_UNIFIED_EXEC_OK";
export const REQUIRED_METHODS = [
  "initialize",
  "conversationImport/thread/commit",
  "modelProvider/create",
  "modelProvider/update",
  "modelProviderKey/create",
  "thread/read",
  "thread/start",
  "agentSession/update",
  "turn/start",
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

export function sanitizeText(value) {
  const sanitized = String(value ?? "")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|session|token)[^=\s]*=)(["']?)[^\s"']+/gi,
      "$1$2[redacted]",
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]+/gi, "$1[redacted]")
    .replace(/sk-[A-Za-z0-9._-]+/g, "sk-[redacted]");
  return sanitized.length > 2_000
    ? `${sanitized.slice(0, 2_000)}... [truncated ${
        sanitized.length - 2_000
      } chars]`
    : sanitized;
}

export function sanitizeJson(value, depth = 0) {
  if (depth > 8) {
    return "[truncated-depth]";
  }
  if (typeof value === "string") {
    return sanitizeText(value);
  }
  if (
    value === null ||
    value === undefined ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 120).map((item) => sanitizeJson(item, depth + 1));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 180)
        .map(([key, item]) => [key, sanitizeJson(item, depth + 1)]),
    );
  }
  return sanitizeText(String(value));
}

export function writeJsonFile(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

export function createTempRuntimeEnv() {
  const tempRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "codex-import-continuation-"),
  );
  const home = path.join(tempRoot, "home");
  const xdgDataHome = path.join(tempRoot, "xdg-data");
  const localAppData = path.join(tempRoot, "local-app-data");
  const roamingAppData = path.join(tempRoot, "roaming-app-data");
  const electronUserDataDir = path.join(tempRoot, "electron-user-data");
  const sourceRoot = path.join(tempRoot, "codex-home");
  const workspaceRoot = path.join(tempRoot, "workspace");
  const rolloutPath = path.join(sourceRoot, "rollout-continuation.jsonl");

  for (const dir of [
    home,
    xdgDataHome,
    localAppData,
    roamingAppData,
    electronUserDataDir,
    sourceRoot,
    workspaceRoot,
  ]) {
    fs.mkdirSync(dir, { recursive: true });
  }
  writeCodexRolloutFixture(rolloutPath, workspaceRoot);

  return {
    tempRoot,
    electronUserDataDir,
    sourceRoot,
    rolloutPath,
    workspaceRoot,
    env: {
      ...process.env,
      HOME: home,
      XDG_DATA_HOME: xdgDataHome,
      APPDATA: roamingAppData,
      LOCALAPPDATA: localAppData,
      CODEX_HOME: sourceRoot,
    },
  };
}

export function buildProviderScriptedResponses(runtimeEnv) {
  const escapedNodePath = process.execPath.replaceAll('"', '\\"');
  const command = `"${escapedNodePath}" -e "process.stdout.write('${COMMAND_OUTPUT_MARKER}')"`;
  const toolArguments = {
    cmd: command,
    workdir: runtimeEnv.workspaceRoot,
    yield_time_ms: 10_000,
    max_output_tokens: 2_000,
  };
  return {
    command,
    responses: [
      {
        type: "tool_call",
        id: "call-imported-live-exec",
        name: "exec_command",
        arguments: toolArguments,
      },
      { type: "text", content: IMPORTED_FINAL_TEXT },
      {
        type: "tool_call",
        id: "call-normal-live-exec",
        name: "exec_command",
        arguments: toolArguments,
      },
      { type: "text", content: NORMAL_FINAL_TEXT },
    ],
  };
}

function writeCodexRolloutFixture(rolloutPath, workspaceRoot) {
  const lines = [
    {
      timestamp: "2026-06-16T00:00:00.000Z",
      type: "session_meta",
      payload: {
        id: SOURCE_THREAD_ID,
        timestamp: "2026-06-16T00:00:00.000Z",
        cwd: workspaceRoot,
        source: "cli",
        model_provider: "openai",
      },
    },
    {
      timestamp: "2026-06-16T00:00:01.000Z",
      type: "event_msg",
      payload: {
        type: "user_message",
        message: `## My request for Codex: ${IMPORTED_USER_TEXT}`,
      },
    },
    {
      timestamp: "2026-06-16T00:00:02.000Z",
      type: "response_item",
      payload: {
        type: "reasoning",
        content: [{ type: "reasoning_text", text: IMPORTED_REASONING_TEXT }],
      },
    },
    {
      timestamp: "2026-06-16T00:00:03.000Z",
      type: "response_item",
      payload: {
        type: "function_call",
        id: "call_exec",
        call_id: "call_exec",
        name: "exec_command",
        arguments: JSON.stringify({ cmd: "npm test", workdir: workspaceRoot }),
      },
    },
    {
      timestamp: "2026-06-16T00:00:04.000Z",
      type: "event_msg",
      payload: {
        type: "exec_approval_request",
        call_id: "call_exec",
        command: ["npm", "test"],
      },
    },
    {
      timestamp: "2026-06-16T00:00:05.000Z",
      type: "response_item",
      payload: {
        type: "function_call_output",
        call_id: "call_exec",
        output: "Exit code: 0\nWall time: 0 seconds\nOutput:\nok",
      },
    },
    {
      timestamp: "2026-06-16T00:00:06.000Z",
      type: "response_item",
      payload: {
        type: "web_search_call",
        id: "call_search",
        call_id: "call_search",
        action: "search_query",
        query: "Lime history import",
      },
    },
    {
      timestamp: "2026-06-16T00:00:07.000Z",
      type: "event_msg",
      payload: {
        type: "web_search_end",
        call_id: "call_search",
        action: "search_query",
        query: "Lime history import",
      },
    },
    {
      timestamp: "2026-06-16T00:00:08.000Z",
      type: "event_msg",
      payload: {
        type: "patch_apply_end",
        call_id: "call_patch",
        success: true,
        changes: {
          [path.join(workspaceRoot, "src/lib.rs")]: { type: "modify" },
        },
      },
    },
    {
      timestamp: "2026-06-16T00:00:09.000Z",
      type: "event_msg",
      payload: { type: "agent_message", message: IMPORTED_ASSISTANT_TEXT },
    },
  ];
  fs.writeFileSync(
    rolloutPath,
    `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`,
  );
}

function isTransientPageEvaluationError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return (
    message.includes("Execution context was destroyed") ||
    message.includes("most likely because of a navigation") ||
    message.includes("Cannot find context with specified id")
  );
}

async function evaluatePageSnapshot(page, pageFunction, arg) {
  try {
    return await page.evaluate(pageFunction, arg);
  } catch (error) {
    if (isTransientPageEvaluationError(error)) {
      return null;
    }
    throw error;
  }
}

export async function waitForRendererReady(page, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => ({
      url: window.location.href,
      electron: window.__LIME_ELECTRON__ === true,
      hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
      supportsAppServer:
        typeof window.electronAPI?.supportsCommand === "function" &&
        window.electronAPI.supportsCommand("app_server_handle_json_lines"),
      startupVisible: Boolean(
        document.querySelector("[data-lime-startup-shell]"),
      ),
      appSidebarVisible: Boolean(
        document.querySelector('[data-testid="app-sidebar"]'),
      ),
    }));
    if (
      snapshot?.electron &&
      snapshot.hasInvokeBridge &&
      snapshot.supportsAppServer &&
      !snapshot.startupVisible &&
      snapshot.appSidebarVisible
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error("Electron renderer / App Server bridge 未就绪");
}

export async function clearInvokeBuffers(page) {
  await page.evaluate(() => {
    window.localStorage.removeItem("lime_invoke_error_buffer_v1");
    window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
  });
}

export function createPageAppServerClient(page) {
  let requestIndex = 0;
  const requests = [];
  const messages = [];
  const bridgeFacts = [];

  async function exchange(payload) {
    const envelope = await page.evaluate(
      async ({ command, payload: requestPayload }) => {
        const invoke = window.electronAPI?.invoke;
        if (typeof invoke !== "function") {
          throw new Error("Electron preload invoke bridge is unavailable");
        }
        return {
          bridge: {
            electron: window.__LIME_ELECTRON__ === true,
            hasInvoke: true,
            supportsCommand:
              typeof window.electronAPI?.supportsCommand === "function" &&
              window.electronAPI.supportsCommand(command),
          },
          response: await invoke(command, {
            request: { lines: [JSON.stringify(requestPayload)] },
          }),
        };
      },
      { command: APP_SERVER_HANDLE_JSON_LINES_COMMAND, payload },
    );
    bridgeFacts.push(envelope?.bridge ?? null);
    const response = envelope?.response;
    const decoded = Array.isArray(response?.lines)
      ? response.lines
          .map((line) => {
            try {
              return JSON.parse(line);
            } catch {
              return null;
            }
          })
          .filter(Boolean)
      : [];
    messages.push(...decoded);
    return decoded;
  }

  return {
    requests,
    messages,
    bridgeFacts,
    async call(method, params = {}) {
      const id = `codex-import-unified-exec-${++requestIndex}`;
      requests.push({ id, method });
      const decoded = await exchange({ jsonrpc: "2.0", id, method, params });
      const error = decoded.find(
        (message) => message?.id === id && message.error,
      );
      if (error) {
        throw new Error(`${method} failed: ${JSON.stringify(error.error)}`);
      }
      const response = decoded.find(
        (message) =>
          message?.id === id &&
          Object.prototype.hasOwnProperty.call(message, "result"),
      );
      if (!response) {
        throw new Error(`${method} did not return a JSON-RPC result`);
      }
      return response.result;
    },
    async notify(method, params = undefined) {
      requests.push({ id: null, method });
      await exchange({ jsonrpc: "2.0", method, ...(params ? { params } : {}) });
    },
  };
}

export async function waitForConversationImportJob(
  client,
  initialJob,
  { timeoutMs = 120_000, intervalMs = 100 } = {},
) {
  assert(
    initialJob?.jobId,
    "conversationImport/thread/commit did not return jobId",
  );
  const startedAt = Date.now();
  let job = initialJob;
  while (Date.now() - startedAt < timeoutMs) {
    if (job.status === "completed") {
      assert(
        job.result?.session?.sessionId && job.result?.session?.threadId,
        "conversation import job completed without canonical session identity",
      );
      return job;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "conversation import job failed");
    }
    await sleep(intervalMs);
    const read = await client.call("conversationImport/job/read", {
      jobId: initialJob.jobId,
    });
    job = read?.job;
    assert(
      job?.jobId === initialJob.jobId,
      "conversationImport/job/read returned a different job",
    );
  }
  throw new Error(`conversation import job timed out: ${initialJob.jobId}`);
}

export async function initializeAndCommitImport(
  client,
  runtimeEnv,
  options = {},
) {
  const initialize = await client.call("initialize", {
    clientInfo: { name: "codex-import-unified-exec-fixture", version: "1.0.0" },
    capabilities: { eventMethods: ["agentSession/event"] },
  });
  await client.notify("initialized");
  const commit = await client.call("conversationImport/thread/commit", {
    sourceClient: "codex",
    sourceRoot: runtimeEnv.sourceRoot,
    sourcePath: runtimeEnv.rolloutPath,
    appId: "desktop",
    workspaceId: WORKSPACE_ID,
    confirmed: true,
  });
  const job = await waitForConversationImportJob(client, commit?.job, options);
  const sessionId = job.result?.session?.sessionId;
  const threadId = job.result?.session?.threadId;
  assert(sessionId, "conversation import job result did not return sessionId");
  assert(threadId, "conversation import job result did not return threadId");
  const importedRead = await client.call("thread/read", {
    threadId,
    includeTurns: true,
  });
  return { initialize, commit, job, importedRead, sessionId, threadId };
}

function canonicalTurns(read) {
  return Array.isArray(read?.thread?.turns) ? read.thread.turns : [];
}

function canonicalItems(read) {
  return canonicalTurns(read).flatMap((turn) =>
    (Array.isArray(turn?.items) ? turn.items : []).map((item) => ({
      item,
      turnId: turn?.id ?? null,
    })),
  );
}

function contentTextFromUserItem(item) {
  return (Array.isArray(item?.content) ? item.content : [])
    .map((part) => (typeof part?.text === "string" ? part.text : ""))
    .join("")
    .trim();
}

function findCompletedCommand(read, turnId, expectedCommand) {
  return canonicalItems(read).find(
    ({ item, turnId: itemTurnId }) =>
      item?.type === "commandExecution" &&
      itemTurnId === turnId &&
      item?.status === "completed" &&
      item?.command === expectedCommand &&
      item?.exitCode === 0 &&
      String(item?.aggregatedOutput || "").includes(COMMAND_OUTPUT_MARKER),
  )?.item;
}

async function waitForTurnCompletion(
  client,
  { threadId, turnId, expectedCommand, finalText, timeoutMs, intervalMs },
) {
  const startedAt = Date.now();
  let latestRead = null;
  while (Date.now() - startedAt < timeoutMs) {
    latestRead = await client.call("thread/read", {
      threadId,
      includeTurns: true,
    });
    const items = canonicalItems(latestRead);
    const turnCompleted = canonicalTurns(latestRead).some(
      (turn) => turn?.id === turnId && turn?.status === "completed",
    );
    if (
      turnCompleted &&
      findCompletedCommand(latestRead, turnId, expectedCommand) &&
      items.some(
        ({ item, turnId: itemTurnId }) =>
          itemTurnId === turnId &&
          item?.type === "agentMessage" &&
          String(item?.text || "").includes(finalText),
      )
    ) {
      return latestRead;
    }
    await sleep(intervalMs);
  }
  throw new Error(
    `等待 unified exec turn 完成超时: thread=${threadId} turn=${turnId}`,
  );
}

async function createRepositoryProvider(client, provider) {
  const created = await client.call("modelProvider/create", {
    name: `Codex import continuation fixture ${Date.now()}`,
    providerType: provider.providerName,
    apiHost: provider.providerConfig.baseUrl,
  });
  const providerId = String(created?.provider?.id || "").trim();
  assert(providerId, "modelProvider/create did not return provider.id");
  await client.call("modelProvider/update", {
    providerId,
    enabled: true,
    sortOrder: 1,
    customModels: [provider.modelPreference],
  });
  const key = await client.call("modelProviderKey/create", {
    providerId,
    apiKey: provider.providerConfig.apiKey,
    alias: "codex-import-continuation-fixture",
    replaceExisting: true,
  });
  assert(key?.key?.id, "modelProviderKey/create did not return key.id");
  return { providerId, model: provider.modelPreference };
}

async function updateSessionProvider(client, sessionId, route) {
  return await client.call("agentSession/update", {
    sessionId,
    providerSelector: route.providerId,
    providerName: route.providerId,
    modelName: route.model,
    executionStrategy: "react",
  });
}

async function startUnifiedExecTurn(
  client,
  { threadId, clientUserMessageId, text, route, runtimeEnv },
) {
  return await client.call("turn/start", {
    threadId,
    clientUserMessageId,
    input: [{ type: "text", text }],
    cwd: runtimeEnv.workspaceRoot,
    runtimeWorkspaceRoots: [runtimeEnv.workspaceRoot],
    model: route.model,
    approvalPolicy: "never",
    sandboxPolicy: "danger-full-access",
    additionalContext: {
      metadata: {
        kind: "application",
        value: JSON.stringify({
          harness: {
            source: "smoke:codex-import-continuation-fixture",
            access_mode: "full-access",
            skip_mcp_prewarm: true,
          },
        }),
      },
    },
  });
}

export async function runImportedAndNormalTurns(
  client,
  { importedSessionId, importedThreadId, provider, runtimeEnv, command, options },
) {
  const route = await createRepositoryProvider(client, provider);
  await updateSessionProvider(client, importedSessionId, route);
  const importedTurn = await startUnifiedExecTurn(client, {
    threadId: importedThreadId,
    clientUserMessageId: IMPORTED_TURN_ID,
    text: IMPORTED_CONTINUE_TEXT,
    route,
    runtimeEnv,
  });
  const importedTurnId = String(importedTurn?.turn?.id || "").trim();
  assert(importedTurnId, "imported turn/start did not return canonical turn.id");
  const importedRead = await waitForTurnCompletion(client, {
    threadId: importedThreadId,
    turnId: importedTurnId,
    expectedCommand: command,
    finalText: IMPORTED_FINAL_TEXT,
    timeoutMs: options.timeoutMs,
    intervalMs: options.intervalMs,
  });

  const normalStart = await client.call("thread/start", {
    model: route.model,
    modelProvider: route.providerId,
    cwd: runtimeEnv.workspaceRoot,
    runtimeWorkspaceRoots: [runtimeEnv.workspaceRoot],
    historyMode: "paginated",
    serviceName: "Codex unified exec normal fixture",
    threadSource: "appServer",
  });
  const normalSessionId = String(normalStart?.thread?.sessionId || "").trim();
  const normalThreadId = String(normalStart?.thread?.id || "").trim();
  assert(
    normalSessionId && normalThreadId,
    "thread/start did not return canonical session/thread identity",
  );
  const normalTurn = await startUnifiedExecTurn(client, {
    threadId: normalThreadId,
    clientUserMessageId: NORMAL_TURN_ID,
    text: NORMAL_USER_TEXT,
    route,
    runtimeEnv,
  });
  const normalTurnId = String(normalTurn?.turn?.id || "").trim();
  assert(normalTurnId, "normal turn/start did not return canonical turn.id");
  const normalRead = await waitForTurnCompletion(client, {
    threadId: normalThreadId,
    turnId: normalTurnId,
    expectedCommand: command,
    finalText: NORMAL_FINAL_TEXT,
    timeoutMs: options.timeoutMs,
    intervalMs: options.intervalMs,
  });
  return {
    importedTurn,
    importedTurnId,
    importedRead,
    normalSessionId,
    normalThreadId,
    normalStart,
    normalTurn,
    normalTurnId,
    normalRead,
    route,
  };
}

function requestToolNames(body) {
  if (!Array.isArray(body?.tools)) {
    return [];
  }
  return body.tools
    .map((tool) =>
      String(
        tool?.function?.name || tool?.name || tool?.tool?.function?.name || "",
      ).trim(),
    )
    .filter(Boolean);
}

export function providerRequestSummaries(requests) {
  return requests.map((request, index) => ({
    index,
    path: request?.path || null,
    model: request?.body?.model || null,
    stream: request?.body?.stream === true,
    toolNames: requestToolNames(request?.body),
    responseKind: request?.responseKind || null,
    responseToolName: request?.responseToolName || null,
    responseError: request?.responseError || null,
  }));
}

function commandShape(item) {
  return {
    type: item?.type ?? null,
    status: item?.status ?? null,
    command: item?.command ?? null,
    cwd: item?.cwd ?? null,
    aggregatedOutput: item?.aggregatedOutput ?? null,
    exitCode: item?.exitCode ?? null,
  };
}

function historicalImportFacts(importedRead, importJob, runtimeEnv) {
  const turns = canonicalTurns(importedRead);
  const items = canonicalItems(importedRead).map(({ item }) => item);
  return {
    turnsLength: turns.length,
    itemsLength: items.length,
    hasUserMessage: items.some(
      (item) =>
        item?.type === "userMessage" &&
        contentTextFromUserItem(item) === IMPORTED_USER_TEXT,
    ),
    hasAssistantMessage: items.some(
      (item) =>
        item?.type === "agentMessage" &&
        String(item?.text || "").includes(IMPORTED_ASSISTANT_TEXT),
    ),
    hasReasoningItem: items.some(
      (item) =>
        item?.type === "reasoning" &&
        [...(item?.summary || []), ...(item?.content || [])].includes(
          IMPORTED_REASONING_TEXT,
        ),
    ),
    hasCommandItem: items.some(
      (item) =>
        item?.type === "commandExecution" &&
        String(item?.id || "").endsWith("call_exec") &&
        item?.command === "npm test",
    ),
    hasPatchItem: items.some(
      (item) =>
        item?.type === "fileChange" &&
        Array.isArray(item?.changes) &&
        item.changes.some(
          (change) =>
            change?.path === path.join(runtimeEnv.workspaceRoot, "src/lib.rs"),
        ),
    ),
    hasWebSearchItem: items.some(
      (item) =>
        (item?.type === "webSearch" ||
          (item?.type === "dynamicToolCall" && item?.tool === "web_search")) &&
        String(item?.id || "").endsWith("call_search"),
    ),
    hasApprovalFidelity: importJob?.result?.summary?.fidelity?.approvals === 1,
  };
}

export function summarizeAndAssertFixture({
  client,
  initial,
  turns,
  providerRequestsAfterCommit,
  providerRequests,
  command,
  runtimeEnv,
}) {
  const requestMethods = Array.from(
    new Set(client.requests.map((request) => request.method)),
  );
  const providerSummaries = providerRequestSummaries(providerRequests);
  const importedCommand = findCompletedCommand(
    turns.importedRead,
    turns.importedTurnId,
    command,
  );
  const normalCommand = findCompletedCommand(
    turns.normalRead,
    turns.normalTurnId,
    command,
  );
  const importedShape = commandShape(importedCommand);
  const normalShape = commandShape(normalCommand);
  const historical = historicalImportFacts(
    initial.importedRead,
    initial.job,
    runtimeEnv,
  );
  const retiredTools = ["Bash", "PowerShell", "BashTool", "PowerShellTool"];

  assert(providerRequestsAfterCommit === 0, "导入 commit 触发了 provider 请求");
  assert(
    REQUIRED_METHODS.every((method) => requestMethods.includes(method)),
    `缺少 App Server current method: ${REQUIRED_METHODS.filter(
      (method) => !requestMethods.includes(method),
    ).join(", ")}`,
  );
  assert(
    initial.job?.result?.canContinue === true,
    "导入会话未标记 canContinue",
  );
  assert(historical.hasUserMessage, "导入用户消息未进入 canonical Thread");
  assert(historical.hasAssistantMessage, "导入助手消息未进入 canonical Thread");
  assert(historical.hasReasoningItem, "导入 reasoning 未进入 canonical items");
  assert(historical.hasCommandItem, "导入 command 未进入 canonical items");
  assert(historical.hasPatchItem, "导入 patch 未进入 canonical items");
  assert(historical.hasWebSearchItem, "导入 web_search 未进入 canonical items");
  assert(historical.hasApprovalFidelity, "导入 approval fidelity 计数丢失");
  assert(importedCommand, "导入续聊未产生 completed Command Item");
  assert(normalCommand, "普通会话未产生 completed Command Item");
  assert(
    JSON.stringify(importedShape) === JSON.stringify(normalShape),
    `导入续聊与普通会话 Command Item 不同构: ${JSON.stringify({ importedShape, normalShape })}`,
  );
  assert(
    providerSummaries.length === 4,
    "两个 unified exec turn 应产生 4 次 provider 请求",
  );
  for (const request of providerSummaries) {
    assert(
      request.toolNames.includes("exec_command") &&
        request.toolNames.includes("write_stdin"),
      `provider request 未暴露 unified exec pair: ${request.toolNames.join(", ")}`,
    );
    assert(
      retiredTools.every((toolName) => !request.toolNames.includes(toolName)),
      `provider request 仍暴露 retired shell tool: ${request.toolNames.join(", ")}`,
    );
  }

  return {
    requestMethods,
    sessionId: initial.sessionId,
    threadId: initial.threadId,
    normalSessionId: turns.normalSessionId,
    normalThreadId: turns.normalThreadId,
    providerRequestsAfterCommit,
    providerRequests: providerSummaries,
    historical,
    importedCommandShape: importedShape,
    normalCommandShape: normalShape,
    commandShapesIsomorphic:
      JSON.stringify(importedShape) === JSON.stringify(normalShape),
  };
}

export function summarizeAndAssertBridge(client) {
  const turnStartCount = client.requests.filter(
    (request) => request.method === "turn/start",
  ).length;
  assert(client.bridgeFacts.length > 0, "未记录 Electron preload bridge facts");
  assert(
    client.bridgeFacts.every(
      (fact) => fact?.electron && fact?.hasInvoke && fact?.supportsCommand,
    ),
    "App Server 调用未全部经过真实 Electron preload bridge",
  );
  assert(
    turnStartCount === 2,
    "未通过 current method 发起导入/普通两次 turn start",
  );
  return {
    electron: true,
    preloadInvoke: true,
    command: APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    callCount: client.bridgeFacts.length,
    turnStartCount,
  };
}
