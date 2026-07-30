import type { AgentSessionExecutionRuntimeRoutingDecision } from "./agentExecutionRuntime";
import type {
  AgentActionRequiredScope,
  AgentToolExecutionResult,
  AgentToolResultImage,
} from "./agentProtocolCoreTypes";
import type {
  AgentEvent,
  AgentEventProviderTrace,
  AgentProviderTraceStage,
} from "./agentProtocolEventTypes";

export function routingDecisionFromEvent(
  event: Record<string, unknown>,
): AgentSessionExecutionRuntimeRoutingDecision {
  const routingDecision =
    (event.routing_decision as Record<string, unknown> | undefined) ||
    (event.routingDecision as Record<string, unknown> | undefined) ||
    {};
  const merged: Record<string, unknown> = { ...routingDecision };
  for (const [sourceKey, targetKey] of [
    ["fallbackApplied", "fallbackApplied"],
    ["fallback_applied", "fallbackApplied"],
    ["requestedSelection", "requestedSelection"],
    ["requested_selection", "requestedSelection"],
    ["routingAttempts", "routingAttempts"],
    ["routing_attempts", "routingAttempts"],
  ] as const) {
    if (event[sourceKey] !== undefined && merged[targetKey] === undefined) {
      merged[targetKey] = event[sourceKey];
    }
  }
  return merged as unknown as AgentSessionExecutionRuntimeRoutingDecision;
}

export function normalizeActionRequiredScope(
  value: unknown,
): AgentActionRequiredScope | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }

  const record = value as Record<string, unknown>;
  const scope = {
    session_id:
      typeof record.session_id === "string"
        ? record.session_id
        : typeof record.sessionId === "string"
          ? record.sessionId
          : undefined,
    thread_id:
      typeof record.thread_id === "string"
        ? record.thread_id
        : typeof record.threadId === "string"
          ? record.threadId
          : undefined,
    turn_id:
      typeof record.turn_id === "string"
        ? record.turn_id
        : typeof record.turnId === "string"
          ? record.turnId
          : undefined,
  };

  return scope.session_id || scope.thread_id || scope.turn_id
    ? scope
    : undefined;
}

export function normalizeRecord(
  value: unknown,
): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

export function normalizeOptionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

export function normalizeOptionalBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

export function pickStringField(
  record: Record<string, unknown>,
  ...keys: string[]
): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string") {
      return value;
    }
  }
  return undefined;
}

export function pickStringArrayField(
  record: Record<string, unknown>,
  ...keys: string[]
): string[] | undefined {
  for (const key of keys) {
    const value = record[key];
    if (Array.isArray(value)) {
      const values = value.filter(
        (item): item is string => typeof item === "string",
      );
      return values.length ? values : undefined;
    }
  }
  return undefined;
}

export function normalizeToolArguments(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function normalizeToolResultOutput(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value === undefined || value === null) {
    return "";
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function normalizeToolExecutionResult(
  event: Record<string, unknown>,
): AgentToolExecutionResult {
  const rawResult = normalizeRecord(event.result);
  const source = rawResult || event;
  const error = typeof source.error === "string" ? source.error : undefined;
  const status =
    typeof source.status === "string"
      ? source.status
      : typeof event.status === "string"
        ? event.status
        : undefined;
  const rawType = typeof event.type === "string" ? event.type : undefined;
  const success =
    typeof source.success === "boolean"
      ? source.success
      : rawType === "tool.failed" ||
          rawType === "tool_failed" ||
          status === "failed"
        ? false
        : !error;

  return {
    success,
    output: normalizeToolResultOutput(
      source.output ?? source.text ?? source.content,
    ),
    error,
    images: Array.isArray(source.images)
      ? (source.images as AgentToolResultImage[])
      : undefined,
    metadata: normalizeRecord(source.metadata),
    structuredContent:
      source.structuredContent ??
      source.structured_content ??
      event.structuredContent ??
      event.structured_content,
    structured_content:
      source.structured_content ??
      source.structuredContent ??
      event.structured_content ??
      event.structuredContent,
  };
}

export function withAgentEventEnvelope<TEvent extends AgentEvent>(
  source: Record<string, unknown>,
  event: TEvent,
): TEvent {
  const preserveActionRequestId =
    (event.type === "action_required" || event.type === "action_resolved") &&
    typeof event.request_id === "string" &&
    event.request_id.length > 0;
  return {
    ...event,
    event_id:
      typeof source.event_id === "string"
        ? source.event_id
        : typeof source.eventId === "string"
          ? source.eventId
          : event.event_id,
    protocol_method:
      typeof source.protocol_method === "string"
        ? source.protocol_method
        : typeof source.protocolMethod === "string"
          ? source.protocolMethod
          : event.protocol_method,
    protocol_revision:
      typeof source.protocol_revision === "string"
        ? source.protocol_revision
        : typeof source.protocolRevision === "string"
          ? source.protocolRevision
          : event.protocol_revision,
    renderer_event_received_at:
      typeof source.renderer_event_received_at === "number" &&
      Number.isFinite(source.renderer_event_received_at)
        ? source.renderer_event_received_at
        : typeof source.rendererEventReceivedAt === "number" &&
            Number.isFinite(source.rendererEventReceivedAt)
          ? source.rendererEventReceivedAt
          : event.renderer_event_received_at,
    request_id: preserveActionRequestId
      ? event.request_id
      : typeof source.request_id === "string"
        ? source.request_id
        : typeof source.requestId === "string"
          ? source.requestId
          : event.request_id,
    run_id:
      typeof source.run_id === "string"
        ? source.run_id
        : typeof source.runId === "string"
          ? source.runId
          : event.run_id,
    sequence:
      typeof source.sequence === "number" && Number.isFinite(source.sequence)
        ? source.sequence
        : event.sequence,
    session_id:
      typeof source.session_id === "string"
        ? source.session_id
        : typeof source.sessionId === "string"
          ? source.sessionId
          : event.session_id,
    server_event_emitted_at:
      typeof source.server_event_emitted_at === "number" &&
      Number.isFinite(source.server_event_emitted_at)
        ? source.server_event_emitted_at
        : typeof source.serverEventEmittedAt === "number" &&
            Number.isFinite(source.serverEventEmittedAt)
          ? source.serverEventEmittedAt
          : event.server_event_emitted_at,
    thread_id:
      typeof source.thread_id === "string"
        ? source.thread_id
        : typeof source.threadId === "string"
          ? source.threadId
          : event.thread_id,
    trace_id:
      typeof source.trace_id === "string"
        ? source.trace_id
        : typeof source.traceId === "string"
          ? source.traceId
          : event.trace_id,
    turn_id:
      typeof source.turn_id === "string"
        ? source.turn_id
        : typeof source.turnId === "string"
          ? source.turnId
          : event.turn_id,
    timestamp:
      typeof source.timestamp === "string" ? source.timestamp : event.timestamp,
  };
}

export function providerTraceStageFromEventType(
  type: string,
): AgentProviderTraceStage | undefined {
  switch (type) {
    case "provider.request.started":
      return "request_started";
    case "provider.first_event.received":
      return "first_event_received";
    case "provider.first_text_delta.received":
      return "first_text_delta_received";
    case "provider.failed":
      return "failed";
    case "provider.canceled":
      return "canceled";
    default:
      return undefined;
  }
}

export function normalizeProviderTraceEvent(
  type: string,
  event: Record<string, unknown>,
): AgentEventProviderTrace | null {
  const payload = normalizeRecord(event.payload);
  const source = payload ?? event;
  const stage =
    pickStringField(source, "stage") ?? providerTraceStageFromEventType(type);
  if (!stage) {
    return null;
  }

  return {
    type: "provider_trace",
    stage,
    provider: pickStringField(source, "provider", "providerId", "provider_id"),
    model: pickStringField(source, "model", "modelName", "model_name"),
    attempt: normalizeOptionalNumber(source.attempt),
    elapsed_ms: normalizeOptionalNumber(source.elapsed_ms ?? source.elapsedMs),
    text_chars: normalizeOptionalNumber(source.text_chars ?? source.textChars),
    status: pickStringField(source, "status"),
    failure_category: pickStringField(
      source,
      "failure_category",
      "failureCategory",
    ),
    retryable: normalizeOptionalBoolean(source.retryable),
    non_retryable_provider_rejection: normalizeOptionalBoolean(
      source.non_retryable_provider_rejection ??
        source.nonRetryableProviderRejection,
    ),
    cancel_reason: pickStringField(source, "cancel_reason", "cancelReason"),
    provider_request_id: pickStringField(
      source,
      "provider_request_id",
      "providerRequestId",
    ),
    provider_request_id_header: pickStringField(
      source,
      "provider_request_id_header",
      "providerRequestIdHeader",
    ),
    runtime_provider_backend: pickStringField(
      source,
      "runtime_provider_backend",
      "runtimeProviderBackend",
    ),
    runtime_provider_selector: pickStringField(
      source,
      "runtime_provider_selector",
      "runtimeProviderSelector",
    ),
    runtime_provider_protocol: pickStringField(
      source,
      "runtime_provider_protocol",
      "runtimeProviderProtocol",
    ),
    runtime_provider_active_model: pickStringField(
      source,
      "runtime_provider_active_model",
      "runtimeProviderActiveModel",
    ),
    runtime_event_type: pickStringField(source, "runtime_event_type") ?? type,
  };
}
