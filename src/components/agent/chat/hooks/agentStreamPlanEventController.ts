import type { AgentEvent, AgentThreadItem } from "@/lib/api/agentProtocol";

function readStructuredPlanStepText(value: unknown): string {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return "";
  }
  const record = value as Record<string, unknown>;
  for (const key of ["step", "text", "content", "title"]) {
    const field = record[key];
    if (typeof field === "string" && field.trim()) {
      return field.trim();
    }
  }
  return "";
}

function planTextFromStructuredPlan(value: unknown): string {
  if (!Array.isArray(value)) {
    return "";
  }
  return value
    .map(readStructuredPlanStepText)
    .filter((step) => step.length > 0)
    .map((step) => `- ${step}`)
    .join("\n");
}

export function buildAgentStreamPlanThreadItem(params: {
  activeSessionId: string;
  event: Extract<AgentEvent, { type: "plan_delta" | "plan_final" }>;
  fallbackTurnId?: string | null;
  now?: string;
  pendingItemKey?: string;
  previousItem?: AgentThreadItem;
  sequence?: number | null;
}): AgentThreadItem | null {
  const now = params.now ?? new Date().toISOString();
  const turnId =
    params.event.turn_id ||
    params.fallbackTurnId?.trim() ||
    params.activeSessionId;
  const canonicalItemId =
    params.event.source === "app_server_v2"
      ? params.event.sourceItemId
      : undefined;
  const revisionId =
    params.event.revisionId ||
    canonicalItemId ||
    `${params.event.type}:${turnId}:${params.event.sequence ?? "live"}`;
  const itemId = canonicalItemId || `plan:${revisionId}`;
  const text = planEventText(params.event, params.previousItem);
  if (!text.trim()) {
    return null;
  }
  return {
    id: itemId,
    thread_id: params.event.thread_id || params.activeSessionId,
    turn_id: turnId,
    sequence:
      params.sequence ??
      params.event.sequence ??
      params.previousItem?.sequence ??
      0,
    status: params.event.type === "plan_final" ? "completed" : "in_progress",
    started_at:
      params.previousItem?.started_at || params.event.timestamp || now,
    completed_at: params.event.type === "plan_final" ? now : undefined,
    updated_at: now,
    type: "plan",
    text,
    metadata: {
      revisionId,
      source:
        params.event.source ||
        (params.event.toolCallId ? "update_plan" : "live_event"),
      ...(params.event.plan !== undefined ? { plan: params.event.plan } : {}),
      ...(params.event.explanation
        ? { explanation: params.event.explanation }
        : {}),
      ...(params.event.sourceItemId
        ? { source_item_id: params.event.sourceItemId }
        : {}),
      ...(params.event.toolCallId
        ? { tool_call_id: params.event.toolCallId }
        : {}),
      ...(params.pendingItemKey
        ? { pending_item_key: params.pendingItemKey }
        : {}),
    },
  };
}

function planEventText(
  event: Extract<AgentEvent, { type: "plan_delta" | "plan_final" }>,
  previousItem: AgentThreadItem | undefined,
): string {
  const eventText = event.text || "";
  const delta = event.delta || "";
  if (
    event.type === "plan_delta" &&
    previousItem?.type === "plan" &&
    delta &&
    eventText === delta
  ) {
    return appendPlanDelta(previousItem.text, delta);
  }
  return eventText || delta || planTextFromStructuredPlan(event.plan);
}

function appendPlanDelta(previousText: string, delta: string): string {
  if (!previousText) {
    return delta;
  }
  if (previousText.endsWith(delta)) {
    return previousText;
  }
  if (delta.startsWith(previousText)) {
    return delta;
  }
  return `${previousText}${delta}`;
}
