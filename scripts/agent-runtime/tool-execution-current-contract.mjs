const PROVIDER_NAME = "Lime Tool Execution Fixture";

function compactRecord(record) {
  return Object.fromEntries(
    Object.entries(record).filter(([, value]) => value !== undefined),
  );
}

function normalizedText(value) {
  return String(value ?? "").trim();
}

function normalizeTurnStatus(value) {
  const status = normalizedText(value);
  if (status === "inProgress") {
    return "in_progress";
  }
  return status.toLowerCase();
}

function threadStatusName(status) {
  if (typeof status === "string") {
    return status;
  }
  return normalizedText(status?.type);
}

function canonicalToolName(item) {
  switch (item?.type) {
    case "dynamicToolCall":
      return normalizedText(item.tool);
    case "mcpToolCall": {
      const tool = normalizedText(item.tool);
      if (tool.startsWith("mcp__")) {
        return tool;
      }
      const server = normalizedText(item.server);
      return server && tool ? `mcp__${server}__${tool}` : tool;
    }
    case "collabAgentToolCall":
      return {
        closeAgent: "interrupt_agent",
        resumeAgent: "followup_task",
        sendInput: "send_message",
        spawnAgent: "spawn_agent",
        wait: "wait_agent",
      }[item.tool];
    case "commandExecution":
      return "exec_command";
    case "fileChange":
      return "apply_patch";
    case "imageView":
      return "view_image";
    case "webSearch":
      return "WebSearch";
    default:
      return "";
  }
}

function canonicalToolOutput(item) {
  const contentItems = item?.contentItems;
  if (Array.isArray(contentItems)) {
    const text = contentItems
      .map((content) =>
        content?.type === "inputText" && typeof content?.text === "string"
          ? content.text
          : "",
      )
      .filter(Boolean)
      .join("\n");
    if (text) {
      return text;
    }
    return JSON.stringify(contentItems);
  }
  const output = item?.result ?? item?.aggregatedOutput ?? item?.error ?? null;
  if (typeof output === "string") {
    return output;
  }
  return output == null ? "" : JSON.stringify(output);
}

function normalizeCanonicalToolCall(item, turnId) {
  const toolName = canonicalToolName(item);
  const callId = normalizedText(item?.id);
  if (!toolName || !callId) {
    return null;
  }
  const status = normalizeTurnStatus(item?.status);
  const agentStates =
    item?.type === "collabAgentToolCall"
      ? Object.entries(item?.agentsStates ?? {}).map(([threadId, state]) => ({
          thread_id: threadId,
          status: normalizedText(state?.status),
          message: state?.message ?? null,
        }))
      : undefined;
  return compactRecord({
    type: "tool_call",
    call_id: callId,
    tool_name: toolName,
    status,
    success:
      item?.success ??
      (status === "completed" ? true : status === "failed" ? false : null),
    output: canonicalToolOutput(item),
    arguments: item?.arguments ?? null,
    agent_states: agentStates,
    turn_id: turnId,
  });
}

export function buildToolExecutionProviderUpdateParams(providerId, provider) {
  return {
    providerId,
    enabled: true,
    sortOrder: 0,
    models: [
      {
        id: provider.modelPreference,
        capability: provider.providerConfig.modelCapabilities,
      },
    ],
  };
}

export function buildToolExecutionThreadStartParams({
  provider,
  title,
  workspaceRoot,
}) {
  return {
    cwd: workspaceRoot,
    historyMode: "paginated",
    model: provider.modelPreference,
    modelProvider: provider.providerPreference,
    runtimeWorkspaceRoots: [workspaceRoot],
    serviceName: title,
    threadSource: "appServer",
  };
}

export function buildToolExecutionTurnStartParams({
  approvalPolicy = "never",
  clientUserMessageId,
  message,
  metadata,
  model,
  sandboxPolicy = "danger-full-access",
  threadId,
  workspaceRoot,
}) {
  return {
    threadId,
    clientUserMessageId,
    input: [{ type: "text", text: message }],
    cwd: workspaceRoot,
    runtimeWorkspaceRoots: [workspaceRoot],
    approvalPolicy,
    sandboxPolicy,
    model,
    responsesapiClientMetadata: {
      source: "smoke:agent-runtime-tool-execution",
    },
    additionalContext: {
      metadata: {
        kind: "application",
        value: JSON.stringify(metadata ?? {}),
      },
    },
  };
}

export function normalizeToolExecutionThreadReadResponse(response) {
  const thread = response?.thread;
  if (!thread || typeof thread !== "object" || Array.isArray(thread)) {
    throw new Error("thread/read 未返回 canonical thread");
  }
  const threadId = normalizedText(thread.id);
  const sessionId = normalizedText(thread.sessionId);
  if (!threadId || !sessionId) {
    throw new Error("thread/read 未返回 canonical thread/session identity");
  }
  const turns = Array.isArray(thread.turns) ? thread.turns : [];
  const canonicalItems = turns.flatMap((turn) =>
    (Array.isArray(turn?.items) ? turn.items : []).map((item) => ({
      item,
      turnId: normalizedText(turn?.id),
    })),
  );
  const toolCalls = canonicalItems
    .map(({ item, turnId }) => normalizeCanonicalToolCall(item, turnId))
    .filter(Boolean);
  const latestTurn = turns.at(-1) ?? null;
  const latestTurnStatus = normalizeTurnStatus(latestTurn?.status);
  const activeTurn = [...turns]
    .reverse()
    .find((turn) => normalizeTurnStatus(turn?.status) === "in_progress");
  const statusType = threadStatusName(thread.status);
  const status =
    statusType === "active"
      ? "running"
      : statusType === "systemError"
        ? "failed"
        : statusType || latestTurnStatus || "idle";
  const pendingRequests = toolCalls
    .filter(
      (call) =>
        call.status === "in_progress" &&
        call.tool_name === "request_user_input",
    )
    .map((call) => ({
      id: call.call_id,
      payload: call.arguments,
      status: "pending",
      threadId,
      turnId: call.turn_id,
    }));

  return {
    thread_id: threadId,
    session_id: sessionId,
    status,
    active_turn_id: normalizedText(activeTurn?.id) || null,
    turns,
    thread_items: toolCalls,
    pending_requests: pendingRequests,
    diagnostics: {
      latestTurnStatus: latestTurnStatus || null,
      latestTurnError: latestTurn?.error ?? null,
    },
    session_detail: {
      id: sessionId,
      thread_id: threadId,
      turns,
      items: canonicalItems.map(({ item }) => item),
      canonicalThread: thread,
    },
  };
}

export async function provisionToolExecutionFixtureProvider({
  fixture,
  invoke,
  options,
}) {
  const created = await invoke(options, "modelProvider/create", {
    name: `${PROVIDER_NAME} ${process.pid}`,
    providerType: "openai",
    apiHost: fixture.baseUrl,
  });
  const providerId = normalizedText(created?.provider?.id);
  if (!providerId) {
    throw new Error("modelProvider/create 未返回 provider.id");
  }
  await invoke(
    options,
    "modelProvider/update",
    buildToolExecutionProviderUpdateParams(providerId, fixture.provider),
  );
  const key = await invoke(options, "modelProviderKey/create", {
    providerId,
    apiKey: fixture.provider.providerConfig.apiKey,
    alias: "tool-execution-fixture",
    replaceExisting: true,
  });
  if (!normalizedText(key?.key?.id)) {
    throw new Error("modelProviderKey/create 未返回 key.id");
  }
  return {
    ...fixture.provider,
    providerPreference: providerId,
  };
}

export async function createToolExecutionThreadCurrent({
  invoke,
  options,
  provider,
  title,
  workspaceRoot,
}) {
  const response = await invoke(
    options,
    "thread/start",
    buildToolExecutionThreadStartParams({ provider, title, workspaceRoot }),
  );
  const threadId = normalizedText(response?.thread?.id);
  const sessionId = normalizedText(response?.thread?.sessionId);
  if (!threadId || !sessionId) {
    throw new Error("thread/start 未返回 canonical thread/session identity");
  }
  return { sessionId, threadId };
}

export async function startToolExecutionTurnCurrent(
  options,
  { sessionId, workspaceRoot, message, turnId, runtimeRequest = {} },
  invoke,
) {
  const response = await invoke(
    options,
    "turn/start",
    buildToolExecutionTurnStartParams({
      clientUserMessageId: turnId,
      approvalPolicy: runtimeRequest.approvalPolicy,
      message,
      metadata: runtimeRequest.metadata,
      model: runtimeRequest.modelPreference,
      sandboxPolicy: runtimeRequest.sandboxPolicy,
      threadId: sessionId,
      workspaceRoot,
    }),
  );
  const canonicalTurnId = normalizedText(response?.turn?.id);
  if (!canonicalTurnId) {
    throw new Error("turn/start 未返回 canonical turn.id");
  }
  return { ...response, canonicalTurnId };
}

export async function readToolExecutionThreadCurrent(
  options,
  threadId,
  _readOptions,
  invoke,
) {
  const response = await invoke(options, "thread/read", {
    threadId,
    includeTurns: true,
  });
  return normalizeToolExecutionThreadReadResponse(response);
}
