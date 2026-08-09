import { normalizeLegacyThreadItem } from "./agentTextNormalization";
import type { AgentEvent } from "./agentProtocolEventTypes";
import type {
  AgentThreadItem,
  AgentThreadTurn,
  AgentTokenUsage,
} from "./agentProtocolCoreTypes";
import {
  normalizeProviderTraceEvent,
  normalizeRecord,
  pickStringField,
} from "./agentProtocolParserUtils";

export function parseAgentContentEvent(
  type: string,
  event: Record<string, unknown>,
): AgentEvent | null {
  switch (type) {
    case "thread_started":
      return {
        type: "thread_started",
        thread_id: (event.thread_id as string) || "",
      };
    case "turn_started":
      return {
        type: "turn_started",
        turn: event.turn as AgentThreadTurn,
      };
    case "item_started":
      return {
        type: "item_started",
        item: normalizeLegacyThreadItem(event.item as AgentThreadItem),
      };
    case "item_updated":
      return {
        type: "item_updated",
        item: normalizeLegacyThreadItem(event.item as AgentThreadItem),
      };
    case "item_completed":
      return {
        type: "item_completed",
        item: normalizeLegacyThreadItem(event.item as AgentThreadItem),
      };
    case "turn_completed":
    case "turn.completed":
      return {
        type: "turn_completed",
        turn: event.turn as AgentThreadTurn,
        text: pickStringField(event, "text", "delta", "message", "content"),
        usage: event.usage as AgentTokenUsage | undefined,
      };
    case "turn_failed":
    case "turn.failed":
      return {
        type: "turn_failed",
        turn: event.turn as AgentThreadTurn,
      };
    case "turn_canceled":
    case "turn.canceled":
      return {
        type: "turn_canceled",
        turn: event.turn as AgentThreadTurn,
      };
    case "text_delta":
    case "message.delta": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      return {
        type: "text_delta",
        text:
          pickStringField(source, "text", "delta", "message", "content") || "",
        itemId: pickStringField(
          source,
          "itemId",
          "item_id",
          "id",
          "messageId",
          "message_id",
        ),
        phase: pickStringField(
          source,
          "phase",
          "messagePhase",
          "message_phase",
        ),
      };
    }
    case "text_delta_batch":
    case "message.delta_batch":
    case "message.batch": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      const text =
        pickStringField(source, "text", "delta", "message", "content") || "";
      const chunks = Array.isArray(event.chunks)
        ? event.chunks.filter(
            (chunk): chunk is string => typeof chunk === "string",
          )
        : payload && Array.isArray(payload.chunks)
          ? payload.chunks.filter(
              (chunk): chunk is string => typeof chunk === "string",
            )
          : text
            ? [text]
            : [];
      return {
        type: "text_delta_batch",
        text,
        chunks,
        itemId: pickStringField(
          source,
          "itemId",
          "item_id",
          "id",
          "messageId",
          "message_id",
        ),
        phase: pickStringField(
          source,
          "phase",
          "messagePhase",
          "message_phase",
        ),
        boundary:
          typeof event.boundary === "string"
            ? event.boundary
            : payload && typeof payload.boundary === "string"
              ? payload.boundary
              : "provider",
      };
    }
    case "thinking_delta":
      return {
        type: "thinking_delta",
        text: (event.text as string) || "",
      };
    case "provider_trace":
    case "provider.request.started":
    case "provider.first_event.received":
    case "provider.first_text_delta.received":
    case "provider.failed":
    case "provider.canceled":
      return normalizeProviderTraceEvent(type, event);
    case "reasoning_started":
    case "reasoning.started":
    case "reasoning.start": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      return {
        type: "reasoning_started",
        reasoningId: pickStringField(
          source,
          "reasoningId",
          "reasoning_id",
          "id",
        ),
        model: source.model,
        providerMetadata:
          normalizeRecord(source.providerMetadata) ??
          normalizeRecord(source.provider_metadata) ??
          normalizeRecord(source.metadata),
      };
    }
    case "reasoning_delta":
    case "reasoning.delta": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      return {
        type: "reasoning_delta",
        reasoningId: pickStringField(
          source,
          "reasoningId",
          "reasoning_id",
          "id",
        ),
        text:
          pickStringField(source, "text", "delta", "message", "content") || "",
        delta: pickStringField(source, "delta", "text", "message", "content"),
        model: source.model,
        providerMetadata:
          normalizeRecord(source.providerMetadata) ??
          normalizeRecord(source.provider_metadata) ??
          normalizeRecord(source.metadata),
      };
    }
    case "reasoning_summary_delta": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      const identity = readReasoningItemIdentity(source);
      const summaryIndex = readReasoningPartIndex(
        source,
        "summaryIndex",
        "summary_index",
      );
      const delta = readReasoningDelta(source);
      if (!identity || summaryIndex === null || delta === null) {
        return null;
      }
      return {
        type: "reasoning_summary_delta",
        ...identity,
        summaryIndex,
        text: delta,
        delta,
      };
    }
    case "reasoning_summary_part_added": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      const identity = readReasoningItemIdentity(source);
      const summaryIndex = readReasoningPartIndex(
        source,
        "summaryIndex",
        "summary_index",
      );
      if (!identity || summaryIndex === null) {
        return null;
      }
      return {
        type: "reasoning_summary_part_added",
        ...identity,
        summaryIndex,
      };
    }
    case "reasoning_content_delta": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      const identity = readReasoningItemIdentity(source);
      const contentIndex = readReasoningPartIndex(
        source,
        "contentIndex",
        "content_index",
      );
      const delta = readReasoningDelta(source);
      if (!identity || contentIndex === null || delta === null) {
        return null;
      }
      return {
        type: "reasoning_content_delta",
        ...identity,
        contentIndex,
        text: delta,
        delta,
      };
    }
    case "reasoning_final":
    case "reasoning.final": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      return {
        type: "reasoning_final",
        reasoningId: pickStringField(
          source,
          "reasoningId",
          "reasoning_id",
          "id",
        ),
        text:
          pickStringField(source, "text", "delta", "message", "content") || "",
        model: source.model,
        providerMetadata:
          normalizeRecord(source.providerMetadata) ??
          normalizeRecord(source.provider_metadata) ??
          normalizeRecord(source.metadata),
      };
    }
    case "reasoning_ended":
    case "reasoning.ended":
    case "reasoning.end": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      return {
        type: "reasoning_ended",
        reasoningId: pickStringField(
          source,
          "reasoningId",
          "reasoning_id",
          "id",
        ),
        status: pickStringField(source, "status"),
        model: source.model,
        providerMetadata:
          normalizeRecord(source.providerMetadata) ??
          normalizeRecord(source.provider_metadata) ??
          normalizeRecord(source.metadata),
      };
    }
    case "plan_delta":
    case "plan.delta":
    case "plan_final":
    case "plan.final": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      const isFinal = type === "plan_final" || type === "plan.final";
      const planEvent = {
        text:
          pickStringField(source, "text", "delta", "message", "content") || "",
        delta: pickStringField(source, "delta", "text", "message", "content"),
        plan: source.plan,
        explanation: pickStringField(source, "explanation"),
        sourceItemId: pickStringField(source, "sourceItemId", "source_item_id"),
        toolCallId: pickStringField(source, "toolCallId", "tool_call_id"),
        revisionId: pickStringField(source, "revisionId", "revision_id"),
        source: pickStringField(source, "source"),
      };
      return isFinal
        ? {
            type: "plan_final",
            ...planEvent,
          }
        : {
            type: "plan_delta",
            ...planEvent,
          };
    }
    case "turn_plan_updated": {
      const plan = readTurnPlanUpdate(event.plan);
      if (
        !plan ||
        (event.explanation !== undefined &&
          typeof event.explanation !== "string")
      ) {
        return null;
      }
      return {
        type: "turn_plan_updated",
        plan,
        ...(typeof event.explanation === "string"
          ? { explanation: event.explanation }
          : {}),
      };
    }
    case "turn_diff_updated":
      return typeof event.unified_diff === "string"
        ? {
            type: "turn_diff_updated",
            unified_diff: event.unified_diff,
          }
        : null;
    default:
      return null;
  }
}

function readTurnPlanUpdate(value: unknown): Array<{
  step: string;
  status: "pending" | "in_progress" | "completed";
}> | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const plan = [] as Array<{
    step: string;
    status: "pending" | "in_progress" | "completed";
  }>;
  for (const candidate of value) {
    const item = normalizeRecord(candidate);
    if (
      !item ||
      Object.keys(item).some((key) => key !== "step" && key !== "status") ||
      typeof item.step !== "string" ||
      (item.status !== "pending" &&
        item.status !== "in_progress" &&
        item.status !== "completed")
    ) {
      return null;
    }
    plan.push({ step: item.step, status: item.status });
  }
  return plan;
}

function readReasoningItemIdentity(
  source: Record<string, unknown>,
): { itemId: string; reasoningId: string } | null {
  const itemId = readConsistentReasoningIdentityAlias(
    source,
    "itemId",
    "item_id",
  );
  const reasoningId = readConsistentReasoningIdentityAlias(
    source,
    "reasoningId",
    "reasoning_id",
  );
  if (
    !itemId ||
    reasoningId === null ||
    (reasoningId && reasoningId !== itemId)
  ) {
    return null;
  }
  return { itemId, reasoningId: itemId };
}

function readConsistentReasoningIdentityAlias(
  source: Record<string, unknown>,
  camelKey: string,
  snakeKey: string,
): string | null | undefined {
  const camelValue = source[camelKey];
  const snakeValue = source[snakeKey];
  if (
    (camelValue !== undefined && typeof camelValue !== "string") ||
    (snakeValue !== undefined && typeof snakeValue !== "string")
  ) {
    return null;
  }
  const camelText = typeof camelValue === "string" ? camelValue.trim() : "";
  const snakeText = typeof snakeValue === "string" ? snakeValue.trim() : "";
  if (camelText && snakeText && camelText !== snakeText) {
    return null;
  }
  return camelText || snakeText || undefined;
}

function readReasoningPartIndex(
  source: Record<string, unknown>,
  camelKey: string,
  snakeKey: string,
): number | null {
  const camelValue = source[camelKey];
  const snakeValue = source[snakeKey];
  if (
    (camelValue !== undefined && !isReasoningPartIndex(camelValue)) ||
    (snakeValue !== undefined && !isReasoningPartIndex(snakeValue)) ||
    (camelValue !== undefined &&
      snakeValue !== undefined &&
      camelValue !== snakeValue)
  ) {
    return null;
  }
  const value = camelValue ?? snakeValue;
  return isReasoningPartIndex(value) ? value : null;
}

function isReasoningPartIndex(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    Number.isInteger(value) &&
    value >= 0
  );
}

function readReasoningDelta(source: Record<string, unknown>): string | null {
  return typeof source.delta === "string" ? source.delta : null;
}
