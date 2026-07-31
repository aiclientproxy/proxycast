import {
  APP_SERVER_METHOD_EVIDENCE_EXPORT,
  APP_SERVER_METHOD_SESSION_LIST,
  APP_SERVER_METHOD_SESSION_READ,
  APP_SERVER_METHOD_SESSION_TURN_START,
  ASSISTANT_DONE_TEXT,
  EVENT_READ_PROBE_DONE_TEXT,
  EVENT_READ_PROBE_PROMPT,
  EVENT_READ_PROBE_READ_TEXT,
  EVENT_READ_PROBE_TOOL_CALL_ID,
  EVENT_READ_PROBE_TOOL_NAME,
  EVENT_READ_PROBE_TOOL_OUTPUT,
  FIXTURE_MODEL,
  FIXTURE_PROVIDER,
  MCP_STRUCTURED_CONTENT_ANSWER,
  MCP_STRUCTURED_CONTENT_DONE_TEXT,
  MCP_STRUCTURED_CONTENT_PROMPT,
  MCP_STRUCTURED_CONTENT_REFERENCE_ID,
  MCP_STRUCTURED_CONTENT_TOOL_CALL_ID,
  MCP_STRUCTURED_CONTENT_TOOL_NAME,
  NEWS_PROMPT,
  PLAN_DONE_TEXT,
  PLAN_PROMPT,
  PLAN_STEPS,
  SKILLS_RUNTIME_SCENARIO,
  summarizeSkillsRuntimeEvidenceExport,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  collectRuntimeEvents,
  drainAppServerEventsFromPage,
  invokeAppServerFromPage,
  mergeRuntimeEvents,
  summarizeRuntimeEvents,
} from "./claw-chat-current-fixture-rpc.mjs";
import {
  collectReadModelToolCalls,
  findReadModelToolCall,
  readModelLatestTurnStatus,
  summarizeSkillsRuntimeReadModel,
} from "./claw-chat-current-fixture-read-model-core.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

function requireThreadId(options, explicitThreadId) {
  const threadId = explicitThreadId?.trim() || options.threadId?.trim();
  if (!threadId) {
    throw new Error("Claw fixture 缺少 canonical threadId");
  }
  return threadId;
}

export async function waitForCanonicalThreadIdBySessionId(
  page,
  options,
  requestLog,
  sessionId,
) {
  const normalizedSessionId = sessionId?.trim();
  if (!normalizedSessionId) {
    throw new Error("Claw fixture 缺少 canonical sessionId");
  }

  const startedAt = Date.now();
  let lastList = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const list = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_LIST,
      { archived: false, limit: 100 },
      requestLog,
    );
    lastList = list.result;
    const threads = Array.isArray(list.result?.data) ? list.result.data : [];
    const root = threads.find(
      (thread) =>
        thread?.sessionId === normalizedSessionId && !thread?.parentThreadId,
    );
    const match =
      root ??
      threads.find((thread) => thread?.sessionId === normalizedSessionId);
    const threadId = typeof match?.id === "string" ? match.id.trim() : "";
    if (threadId) {
      return threadId;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `thread/list 未解析 sessionId 对应的 canonical threadId: ${JSON.stringify(
      sanitizeJson(lastList),
    )}`,
  );
}

export async function waitForSessionReadCompleted(
  page,
  options,
  requestLog,
  {
    threadId,
    prompt = NEWS_PROMPT,
    doneText = ASSISTANT_DONE_TEXT,
    summaryText = "今日国际新闻简要整理",
  } = {},
) {
  const canonicalThreadId = requireThreadId(options, threadId);
  const startedAt = Date.now();
  let lastRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: canonicalThreadId,
        includeTurns: true,
      },
      requestLog,
    );
    lastRead = read.result;
    const serialized = JSON.stringify(read.result || {});
    if (
      readModelLatestTurnStatus(read.result) === "completed" &&
      serialized.includes(prompt) &&
      (serialized.includes(doneText) || serialized.includes(summaryText))
    ) {
      return read.result;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server read model 未完成输入闭环: ${JSON.stringify(
      sanitizeJson(lastRead),
    )}`,
  );
}

export async function waitForSessionReadFailedAfterAnswer(
  page,
  options,
  requestLog,
  { threadId, prompt, partialText, failureText } = {},
) {
  const canonicalThreadId = requireThreadId(options, threadId);
  const startedAt = Date.now();
  let lastRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: canonicalThreadId,
        includeTurns: true,
      },
      requestLog,
    );
    lastRead = read.result;
    const serialized = JSON.stringify(read.result || {});
    if (
      readModelLatestTurnStatus(read.result) === "failed" &&
      serialized.includes(prompt) &&
      serialized.includes(partialText) &&
      serialized.includes(failureText)
    ) {
      return read.result;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server read model 未完成 failed-after-answer 闭环: ${JSON.stringify(
      sanitizeJson(lastRead),
    )}`,
  );
}

export async function waitForSessionReadSkillsRuntimeCompleted(
  page,
  options,
  requestLog,
  scenario = SKILLS_RUNTIME_SCENARIO,
  threadId = options.threadId,
) {
  const canonicalThreadId = requireThreadId(options, threadId);
  const startedAt = Date.now();
  let lastRead = null;
  let lastSummary = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: canonicalThreadId,
        includeTurns: true,
      },
      requestLog,
    );
    lastRead = read.result;
    lastSummary = summarizeSkillsRuntimeReadModel(lastRead, scenario);
    if (
      lastSummary.includesPrompt === true &&
      (lastSummary.includesAssistantDone === true ||
        lastSummary.includesAssistantSummary === true) &&
      lastSummary.includesSkillSearchTool === true &&
      lastSummary.includesSkillTool === true &&
      lastSummary.readModelTurnTerminal === true
    ) {
      return {
        readModel: lastRead,
        summary: lastSummary,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server skills runtime read model 未完成 terminal turn 闭环: ${JSON.stringify(
      sanitizeJson({
        summary: lastSummary,
        readModel: lastRead,
      }),
    )}`,
  );
}

export async function waitForSessionReadMcpStructuredContentCompleted(
  page,
  options,
  requestLog,
) {
  const readModel = await waitForSessionReadCompleted(
    page,
    options,
    requestLog,
    {
      threadId: options.threadId,
      prompt: MCP_STRUCTURED_CONTENT_PROMPT,
      doneText: MCP_STRUCTURED_CONTENT_DONE_TEXT,
      summaryText: "MCP structuredContent 展示验证完成",
    },
  );
  const serialized = JSON.stringify(readModel || {});
  const toolCall = findReadModelToolCallWithStructuredContent(
    readModel,
    MCP_STRUCTURED_CONTENT_TOOL_CALL_ID,
    MCP_STRUCTURED_CONTENT_TOOL_NAME,
  );
  const structuredContent = readToolCallStructuredContent(toolCall);
  const structuredSerialized = JSON.stringify(structuredContent || {});
  const outputText = String(
    toolCall?.output_preview ??
      toolCall?.outputPreview ??
      toolCall?.output ??
      "",
  );

  return {
    readModel,
    summary: sanitizeJson({
      detailItemCount: Array.isArray(readModel?.detail?.items)
        ? readModel.detail.items.length
        : null,
      toolCallCount: collectReadModelToolCalls(readModel).length,
      latestTurnStatus: readModelLatestTurnStatus(readModel),
      includesPrompt: serialized.includes(MCP_STRUCTURED_CONTENT_PROMPT),
      includesAssistantDone: serialized.includes(
        MCP_STRUCTURED_CONTENT_DONE_TEXT,
      ),
      includesAssistantSummary: serialized.includes(
        "MCP structuredContent 展示验证完成",
      ),
      includesMcpTool: Boolean(toolCall),
      toolName: toolCall?.tool_name ?? toolCall?.toolName ?? null,
      toolStatus: toolCall?.status ?? null,
      includesStructuredContent: Boolean(structuredContent),
      structuredContentAnswerVisible: structuredSerialized.includes(
        MCP_STRUCTURED_CONTENT_ANSWER,
      ),
      structuredContentReferenceVisible: structuredSerialized.includes(
        MCP_STRUCTURED_CONTENT_REFERENCE_ID,
      ),
      outputContainsEnvelope:
        outputText.includes("request_metadata") &&
        outputText.includes("mcp_tool_result_projection") &&
        outputText.includes("diagnostics"),
    }),
  };
}

export function findReadModelToolCallWithStructuredContent(
  readModel,
  toolCallId,
  toolName,
) {
  const matches = collectReadModelToolCalls(readModel).filter((toolCall) => {
    const id = String(
      toolCall.id ??
        toolCall.tool_call_id ??
        toolCall.toolCallId ??
        toolCall.toolId ??
        "",
    );
    const name = String(
      toolCall.tool_name ??
        toolCall.toolName ??
        toolCall.tool ??
        toolCall.name ??
        "",
    );
    return id === toolCallId && name === toolName;
  });
  return (
    matches.find((toolCall) => Boolean(readToolCallStructuredContent(toolCall))) ??
    matches[0] ??
    null
  );
}

export function readToolCallStructuredContent(toolCall) {
  const direct =
    toolCall?.structured_content ??
    toolCall?.structuredContent ??
    toolCall?.result?.structuredContent ??
    toolCall?.result?.structured_content ??
    null;
  if (direct) {
    return direct;
  }

  const contentItems =
    toolCall?.contentItems ??
    toolCall?.content_items ??
    (Array.isArray(toolCall?.output) ? toolCall.output : []);
  for (const item of Array.isArray(contentItems) ? contentItems : []) {
    const text =
      typeof item === "string"
        ? item
        : typeof item?.text === "string"
          ? item.text
          : typeof item?.inputText?.text === "string"
            ? item.inputText.text
            : null;
    if (!text) {
      continue;
    }
    try {
      const parsed = JSON.parse(text);
      const nested = parsed?.structuredContent ?? parsed?.structured_content;
      if (nested) {
        return nested;
      }
      if (
        parsed &&
        typeof parsed === "object" &&
        !Array.isArray(parsed) &&
        !Object.hasOwn(parsed, "request_metadata") &&
        !Object.hasOwn(parsed, "diagnostics")
      ) {
        // v2 DynamicToolCall stores the structured block itself as an inputText item.
        return parsed;
      }
    } catch {
      // Non-JSON content items are ordinary tool output.
    }
  }
  return null;
}

export async function waitForSessionReadPlanCompleted(
  page,
  options,
  requestLog,
) {
  const threadId = requireThreadId(options);
  const startedAt = Date.now();
  let lastRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId,
        includeTurns: true,
      },
      requestLog,
    );
    lastRead = read.result;
    const serialized = JSON.stringify(read.result || {});
    if (
      serialized.includes(PLAN_PROMPT) &&
      serialized.includes("<proposed_plan>") &&
      serialized.includes("</proposed_plan>") &&
      PLAN_STEPS.every((step) => serialized.includes(step.step))
    ) {
      return read.result;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server read model 未读回 proposed_plan 计划块: ${JSON.stringify(
      sanitizeJson(lastRead),
    )}`,
  );
}

export async function waitForSessionReadContainsTurn(
  page,
  options,
  requestLog,
  turnId,
  expectedText,
) {
  const threadId = requireThreadId(options);
  const startedAt = Date.now();
  let lastRead = null;
  const timeoutMs = Math.min(options.timeoutMs, 30_000);
  while (Date.now() - startedAt < timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId,
        includeTurns: true,
      },
      requestLog,
    );
    lastRead = read.result;
    const serialized = JSON.stringify(read.result || {});
    if (serialized.includes(turnId) && serialized.includes(expectedText)) {
      return read.result;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server read model 未读回 event/read probe turn: ${JSON.stringify(
      sanitizeJson(lastRead),
    )}`,
  );
}

export async function exportSkillsRuntimeEvidencePack(
  page,
  requestLog,
  scenario = SKILLS_RUNTIME_SCENARIO,
  sessionId,
) {
  const canonicalSessionId = sessionId?.trim();
  if (!canonicalSessionId) {
    throw new Error("Claw fixture 缺少 canonical sessionId");
  }
  const exportResult = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_EVIDENCE_EXPORT,
    {
      sessionId: canonicalSessionId,
      includeEvents: true,
      includeArtifacts: true,
      includeEvidencePack: true,
    },
    requestLog,
  );
  return {
    result: exportResult.result,
    summary: sanitizeJson(
      summarizeSkillsRuntimeEvidenceExport(exportResult.result, scenario),
    ),
  };
}

export async function waitForRuntimeEventsForTurn(
  page,
  options,
  turnId,
  initialMessages,
) {
  const startedAt = Date.now();
  const timeoutMs = Math.min(options.timeoutMs, 30_000);
  let events = collectRuntimeEvents(initialMessages);
  let drainAttempts = 0;

  while (Date.now() - startedAt < timeoutMs) {
    const summary = summarizeRuntimeEvents(events, turnId);
    if (
      summary.scopedEventCount > 0 &&
      summary.hasTextDelta &&
      summary.hasToolStarted &&
      summary.hasToolResult &&
      summary.hasTerminal
    ) {
      return {
        events,
        summary: {
          ...summary,
          drainAttempts,
        },
      };
    }

    const drained = await drainAppServerEventsFromPage(page, 50);
    drainAttempts += 1;
    events = mergeRuntimeEvents(events, collectRuntimeEvents(drained.messages));
    await sleep(options.intervalMs);
  }

  throw new Error(
    `未观察到 direct v2 notification 同 turn 终态: ${JSON.stringify(
      summarizeRuntimeEvents(events, turnId),
    )}`,
  );
}

export async function runEventReadProbe(page, options, requestLog) {
  const threadId = requireThreadId(options);
  const clientUserMessageId = `event-read-probe-${Date.now()}`;
  const turnStart = await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_SESSION_TURN_START,
    {
      threadId,
      clientUserMessageId,
      input: [{ type: "text", text: EVENT_READ_PROBE_PROMPT }],
      model: FIXTURE_MODEL,
      responsesapiClientMetadata: {
        source: "smoke:claw-chat-current-fixture:event-read-probe",
        provider: FIXTURE_PROVIDER,
      },
    },
    requestLog,
  );
  const turnId = turnStart.result?.turn?.id;
  if (typeof turnId !== "string" || !turnId.trim()) {
    throw new Error(
      `turn/start 未返回 canonical turnId: ${JSON.stringify(
        sanitizeJson(turnStart.result),
      )}`,
    );
  }

  const eventObservation = await waitForRuntimeEventsForTurn(
    page,
    options,
    turnId,
    turnStart.messages,
  );
  const readModel = await waitForSessionReadContainsTurn(
    page,
    options,
    requestLog,
    turnId,
    EVENT_READ_PROBE_READ_TEXT,
  );
  const toolCall = findReadModelToolCall(
    readModel,
    EVENT_READ_PROBE_TOOL_CALL_ID,
    EVENT_READ_PROBE_TOOL_NAME,
  );
  const toolOutput = String(
    toolCall?.output_preview ??
      toolCall?.outputPreview ??
      toolCall?.output ??
      "",
  );

  return sanitizeJson({
    turnId,
    clientUserMessageId,
    turnStartResult: {
      turnId: turnStart.result?.turn?.id ?? null,
      status: turnStart.result?.turn?.status ?? null,
      messageCount: turnStart.messages.length,
      notificationCount: collectRuntimeEvents(turnStart.messages).length,
    },
    events: eventObservation.summary,
    readModel: {
      containsTurnId: JSON.stringify(readModel || {}).includes(turnId),
      containsDoneText: JSON.stringify(readModel || {}).includes(
        EVENT_READ_PROBE_DONE_TEXT,
      ),
      containsReadText: JSON.stringify(readModel || {}).includes(
        EVENT_READ_PROBE_READ_TEXT,
      ),
      toolCallCount: collectReadModelToolCalls(readModel).length,
      containsToolCall: Boolean(toolCall),
      toolName: toolCall?.tool_name ?? toolCall?.toolName ?? null,
      toolStatus: toolCall?.status ?? null,
      containsToolOutput: toolOutput.includes(EVENT_READ_PROBE_TOOL_OUTPUT),
      toolTurnId: toolCall?.turn_id ?? toolCall?.turnId ?? null,
      latestTurnStatus:
        readModel?.detail?.thread_read?.runtime_summary?.latestTurnStatus ??
        readModel?.detail?.thread_read?.status ??
        readModel?.detail?.status ??
        null,
    },
  });
}

export async function waitForSessionReadCanceled(
  page,
  options,
  requestLog,
  {
    threadId,
    prompt = NEWS_PROMPT,
    partialText = "",
    requireContent = true,
  } = {},
) {
  const canonicalThreadId = requireThreadId(options, threadId);
  const startedAt = Date.now();
  let lastRead = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const read = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_READ,
      {
        threadId: canonicalThreadId,
        includeTurns: true,
      },
      requestLog,
    );
    lastRead = read.result;
    const latestTurnStatus = readModelLatestTurnStatus(read.result);
    const serialized = JSON.stringify(read.result || {});
    if (
      latestTurnStatus === "interrupted" &&
      (!requireContent ||
        (serialized.includes(prompt) &&
          (!partialText || serialized.includes(partialText))))
    ) {
      return read.result;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `App Server read model 未完成取消闭环: ${JSON.stringify(
      sanitizeJson(lastRead),
    )}`,
  );
}
