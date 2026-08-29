import {
  isRecord,
  nonNegativeInteger,
  normalizeString,
  positiveInteger,
} from "./deepswe-value-utils.mjs";

const TOOL_ITEM_TYPES = new Set([
  "command",
  "command_execution",
  "file_artifact",
  "mcpToolCall",
  "mcp_tool_call",
  "tool",
  "toolCall",
  "tool_call",
]);

function normalizedStringArray(value) {
  return Array.isArray(value)
    ? [...new Set(value.map(normalizeString).filter(Boolean))].sort()
    : [];
}

function providerStepUsage(payload) {
  const runtimeEvent = isRecord(payload?.runtimeEvent)
    ? payload.runtimeEvent
    : payload;
  const raw = isRecord(runtimeEvent?.usage)
    ? runtimeEvent.usage
    : isRecord(payload?.usage)
      ? payload.usage
      : null;
  if (!raw) {
    return null;
  }
  const inputTokens = nonNegativeInteger(
    raw.input_tokens ??
      raw.inputTokens ??
      raw.prompt_tokens ??
      raw.promptTokens,
  );
  const outputTokens = nonNegativeInteger(
    raw.output_tokens ??
      raw.outputTokens ??
      raw.completion_tokens ??
      raw.completionTokens,
  );
  if (inputTokens == null || outputTokens == null) {
    return null;
  }
  const cachedInputTokens =
    nonNegativeInteger(
      raw.cached_input_tokens ??
        raw.cachedInputTokens ??
        raw.cache_read_input_tokens ??
        raw.cacheReadInputTokens,
    ) ?? 0;
  const cacheCreationInputTokens =
    nonNegativeInteger(
      raw.cache_creation_input_tokens ??
        raw.cacheCreationInputTokens ??
        raw.cache_write_input_tokens ??
        raw.cacheWriteInputTokens,
    ) ?? 0;
  return {
    inputTokens,
    outputTokens,
    cachedInputTokens,
    cacheCreationInputTokens,
    budgetTokens: Math.max(0, inputTokens - cachedInputTokens) + outputTokens,
  };
}

export function currentTurns(currentFacts) {
  const candidates = [
    currentFacts?.threadRead?.turns,
    currentFacts?.threadRead?.thread?.turns,
    currentFacts?.sessionRead?.detail?.thread_read?.turns,
    currentFacts?.sessionRead?.detail?.threadRead?.turns,
    currentFacts?.sessionRead?.detail?.turns,
    currentFacts?.sessionRead?.turns,
  ];
  return candidates.find(Array.isArray) || [];
}

export function currentItems(currentFacts) {
  const turns = currentTurns(currentFacts);
  const turnItems = turns.flatMap((turn) =>
    Array.isArray(turn?.items) ? turn.items : [],
  );
  const candidates = [
    ...turnItems,
    ...(Array.isArray(currentFacts?.threadRead?.thread_items)
      ? currentFacts.threadRead.thread_items
      : []),
    ...(Array.isArray(currentFacts?.threadRead?.threadItems)
      ? currentFacts.threadRead.threadItems
      : []),
    ...(Array.isArray(
      currentFacts?.sessionRead?.detail?.thread_read?.thread_items,
    )
      ? currentFacts.sessionRead.detail.thread_read.thread_items
      : []),
    ...(Array.isArray(currentFacts?.sessionRead?.detail?.items)
      ? currentFacts.sessionRead.detail.items
      : []),
    ...(Array.isArray(currentFacts?.sessionRead?.items)
      ? currentFacts.sessionRead.items
      : []),
  ];
  const seen = new Set();
  return candidates.filter((item) => {
    const id = normalizeString(item?.id || item?.item_id || item?.itemId);
    if (!id) return true;
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

function currentProviderStepRecords(currentFacts) {
  const runtimeSteps = (currentFacts?.runtimeEvents || [])
    .filter(
      (event) => event?.type === "provider.step" && isRecord(event.payload),
    )
    .map((event) => ({
      ...event.payload,
      sequence: event.sequence,
      timestamp: event.timestamp,
    }));
  if (runtimeSteps.length > 0) {
    return runtimeSteps;
  }
  const records = [];
  const add = (value) => {
    if (Array.isArray(value)) records.push(...value.filter(isRecord));
    else if (isRecord(value)) records.push(value);
  };
  const threadRead = currentFacts?.threadRead;
  const detail = currentFacts?.sessionRead?.detail;
  const readModels = [
    threadRead,
    threadRead?.diagnostics,
    threadRead?.runtime_summary,
    threadRead?.runtimeSummary,
    detail?.thread_read,
    detail?.threadRead,
    detail?.diagnostics,
    detail?.runtime_summary,
    detail?.runtimeSummary,
  ];
  for (const record of readModels) {
    add(record?.provider_steps ?? record?.providerSteps);
  }
  for (const turn of currentTurns(currentFacts)) {
    add(turn?.provider_steps ?? turn?.providerSteps);
    add(turn?.metadata?.provider_steps ?? turn?.metadata?.providerSteps);
  }
  for (const item of currentItems(currentFacts)) {
    const metadata = isRecord(item?.metadata) ? item.metadata : {};
    add(metadata.provider_steps ?? metadata.providerSteps);
    add(metadata.provider_step ?? metadata.providerStep);
  }
  return records;
}

function providerRequestToolsByAttempt(currentFacts) {
  return new Map(
    (currentFacts?.runtimeEvents || [])
      .filter(
        (event) =>
          event?.type === "provider.request.started" && isRecord(event.payload),
      )
      .map((event) => [
        positiveInteger(event.payload.attempt),
        normalizedStringArray(
          event.payload.tool_names ?? event.payload.toolNames,
        ),
      ])
      .filter(([attempt]) => attempt != null),
  );
}

export function currentUsage(currentFacts) {
  const detail = currentFacts?.sessionRead?.detail;
  const threadRead = currentFacts?.threadRead;
  const turn = currentTurns(currentFacts).find(
    (candidate) =>
      normalizeString(
        candidate?.id || candidate?.turn_id || candidate?.turnId,
      ) === normalizeString(currentFacts?.turnId),
  );
  return (
    turn?.usage ||
    threadRead?.diagnostics?.latest_turn_usage ||
    threadRead?.diagnostics?.latestTurnUsage ||
    threadRead?.runtime_summary?.latest_turn_usage ||
    threadRead?.runtimeSummary?.latestTurnUsage ||
    detail?.thread_read?.diagnostics?.latest_turn_usage ||
    detail?.thread_read?.diagnostics?.latestTurnUsage ||
    detail?.runtime_summary?.latest_turn_usage ||
    detail?.runtime_summary?.latestTurnUsage ||
    null
  );
}

function providerStepRecord(record, index, toolNames) {
  const runtimeEvent = isRecord(record?.runtimeEvent)
    ? record.runtimeEvent
    : record;
  const usage = providerStepUsage(runtimeEvent);
  return {
    sequence: nonNegativeInteger(
      runtimeEvent?.sequence ?? runtimeEvent?.ordinal,
    ),
    timestamp:
      normalizeString(
        runtimeEvent?.timestamp ??
          runtimeEvent?.updatedAt ??
          runtimeEvent?.updated_at,
      ) || null,
    attempt: positiveInteger(runtimeEvent?.attempt) ?? index + 1,
    completed:
      runtimeEvent?.completed === true ||
      /^(completed|succeeded|success|stop)$/i.test(
        normalizeString(runtimeEvent?.status ?? runtimeEvent?.finishReason),
      ),
    finishReason:
      normalizeString(
        runtimeEvent?.finish_reason ?? runtimeEvent?.finishReason,
      ) || null,
    output: {
      textChars:
        nonNegativeInteger(
          runtimeEvent?.text_output_chars ?? runtimeEvent?.textOutputChars,
        ) ?? 0,
      reasoningChars:
        nonNegativeInteger(
          runtimeEvent?.reasoning_output_chars ??
            runtimeEvent?.reasoningOutputChars,
        ) ?? 0,
      toolCalls:
        nonNegativeInteger(
          runtimeEvent?.tool_call_count ?? runtimeEvent?.toolCallCount,
        ) ?? 0,
    },
    toolNames: normalizedStringArray(
      runtimeEvent?.tool_names ??
        runtimeEvent?.toolNames ??
        runtimeEvent?.tools ??
        toolNames,
    ),
    usage,
  };
}

export function providerStepsFromCurrentFacts(
  currentFacts,
  { maxProviderSteps = null, tokenBudget = null } = {},
) {
  const stepLimit = positiveInteger(maxProviderSteps);
  const tokenLimit = positiveInteger(tokenBudget);
  const itemToolNames = currentItems(currentFacts)
    .filter((item) => item?.kind === "tool" || TOOL_ITEM_TYPES.has(item?.type))
    .map((item) => item?.name || item?.tool_name || item?.payload?.name)
    .filter(Boolean);
  const requestTools = providerRequestToolsByAttempt(currentFacts);
  const steps = currentProviderStepRecords(currentFacts).map(
    (record, index) => {
      const attempt = positiveInteger(record?.attempt) ?? index + 1;
      return providerStepRecord(
        record,
        index,
        requestTools.get(attempt) ?? itemToolNames,
      );
    },
  );
  const usage = steps.reduce(
    (total, step) => {
      if (!step.usage) {
        return total;
      }
      total.stepsWithUsage += 1;
      total.inputTokens += step.usage.inputTokens;
      total.outputTokens += step.usage.outputTokens;
      total.cachedInputTokens += step.usage.cachedInputTokens;
      total.cacheCreationInputTokens += step.usage.cacheCreationInputTokens;
      total.budgetTokens += step.usage.budgetTokens;
      return total;
    },
    {
      stepsWithUsage: 0,
      inputTokens: 0,
      outputTokens: 0,
      cachedInputTokens: 0,
      cacheCreationInputTokens: 0,
      budgetTokens: 0,
    },
  );
  const fallbackUsage = providerStepUsage({
    usage: currentUsage(currentFacts),
  });
  if (steps.length === 0 && fallbackUsage) {
    usage.stepsWithUsage = 0;
    usage.inputTokens = fallbackUsage.inputTokens;
    usage.outputTokens = fallbackUsage.outputTokens;
    usage.cachedInputTokens = fallbackUsage.cachedInputTokens;
    usage.cacheCreationInputTokens = fallbackUsage.cacheCreationInputTokens;
    usage.budgetTokens = fallbackUsage.budgetTokens;
  }
  const reasons = [];
  if (stepLimit != null && steps.length >= stepLimit) {
    reasons.push("provider_steps");
  }
  if (tokenLimit != null && usage.budgetTokens >= tokenLimit) {
    reasons.push("token_budget");
  }
  const toolSnapshots = steps.map((step) => ({
    sequence: step.sequence,
    timestamp: step.timestamp,
    attempt: step.attempt,
    toolNames: step.toolNames,
  }));
  const snapshotsWithTools = toolSnapshots.filter(
    (snapshot) => snapshot.toolNames.length > 0,
  );
  const uniqueToolNames = [
    ...new Set(snapshotsWithTools.flatMap((snapshot) => snapshot.toolNames)),
  ].sort();
  return {
    schemaVersion: "deepswe-provider-steps-v1",
    generatedAt: new Date().toISOString(),
    budgets: {
      maxProviderSteps: stepLimit,
      tokenBudget: tokenLimit,
      exhausted: reasons.length > 0,
      reasons,
      remainingProviderSteps:
        stepLimit == null ? null : Math.max(0, stepLimit - steps.length),
      remainingTokens:
        tokenLimit == null
          ? null
          : Math.max(0, tokenLimit - usage.budgetTokens),
    },
    stepCount: steps.length,
    usageStatus:
      steps.length === 0
        ? "missing"
        : usage.stepsWithUsage === steps.length
          ? "complete"
          : "partial",
    usage,
    toolCatalog: {
      status:
        toolSnapshots.length === 0
          ? "missing"
          : snapshotsWithTools.length === toolSnapshots.length
            ? "complete"
            : "partial",
      requestCount: toolSnapshots.length,
      requestsWithTools: snapshotsWithTools.length,
      uniqueToolNames,
      applyPatchAvailableOnEveryRequest:
        snapshotsWithTools.length === 0
          ? null
          : snapshotsWithTools.length === toolSnapshots.length &&
            snapshotsWithTools.every((snapshot) =>
              snapshot.toolNames.includes("apply_patch"),
            ),
      requests: toolSnapshots,
    },
    steps,
  };
}

export function toolLifecycleFromCurrentFacts(currentFacts) {
  const toolItems = currentItems(currentFacts).filter((item) => {
    const itemType = normalizeString(item?.type);
    const payloadType = normalizeString(item?.payload?.type);
    return (
      item?.kind === "tool" ||
      TOOL_ITEM_TYPES.has(itemType) ||
      TOOL_ITEM_TYPES.has(payloadType)
    );
  });
  return {
    schemaVersion: "deepswe-tool-lifecycle-v1",
    itemCount: toolItems.length,
    items: toolItems,
  };
}

export function terminalMessageFromCurrentFacts(currentFacts, turnId) {
  const turn = currentTurns(currentFacts).find(
    (candidate) =>
      normalizeString(
        candidate?.id || candidate?.turn_id || candidate?.turnId,
      ) === normalizeString(turnId),
  );
  const diagnostics =
    currentFacts?.threadRead?.diagnostics ||
    currentFacts?.sessionRead?.detail?.thread_read?.diagnostics ||
    currentFacts?.sessionRead?.detail?.diagnostics ||
    {};
  return normalizeString(
    turn?.error?.message ||
      turn?.error ||
      turn?.failure?.message ||
      turn?.failure ||
      diagnostics.latest_turn_error_message ||
      diagnostics.latestTurnErrorMessage,
  );
}

export function providerStepExhaustion(providerSteps) {
  const steps = Array.isArray(providerSteps?.steps) ? providerSteps.steps : [];
  const lastStep = steps.at(-1);
  if (
    !providerSteps?.budgets?.reasons?.includes("provider_steps") ||
    normalizeString(lastStep?.finishReason).toLowerCase() !== "tool_call"
  ) {
    return null;
  }
  return {
    reasons: [...providerSteps.budgets.reasons],
    stepCount: providerSteps.stepCount,
    usage: providerSteps.usage,
  };
}
