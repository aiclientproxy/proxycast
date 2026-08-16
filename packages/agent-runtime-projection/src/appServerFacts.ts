import type {
  AgentRuntimeExecutionEvent,
  AgentRuntimeExecutionEventKind,
  AgentRuntimeExecutionEventStatus,
  AgentUiProjectionState,
} from "@limecloud/agent-ui-contracts";
import {
  type AgentUiPhase,
  normalizeRuntimeStatusValue,
  isLegacyRuntimeTurnTerminalEventClass,
  normalizeRuntimeTurnTerminalEventClass,
  runtimeStatusForTerminalEventClass,
  runtimeTurnTerminalProjectionFromStatus,
} from "@limecloud/agent-ui-contracts";

import {
  compactProjectionFields,
  definedString,
  readNumberField,
  readRecord,
  readStringArrayField,
  readStringField,
} from "./normalization.js";
import { buildAgentUiCollaborationPayloadMetadata } from "./collaborationFacts.js";
import { extractAgentUiToolLifecyclePayloadMetadata } from "./toolLifecycleMetadata.js";
import { projectAgentUiState } from "./uiState.js";

export interface AppServerAgentSessionFact {
  sessionId: string;
  threadId: string;
  appId?: string;
  workspaceId?: string;
  status?: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface AppServerAgentTurnFact {
  turnId: string;
  sessionId?: string;
  threadId?: string;
  status?: string;
  startedAt?: string;
  completedAt?: string;
}

export interface AppServerAgentEventFact {
  eventId: string;
  sequence?: number;
  sessionId?: string;
  threadId?: string;
  turnId?: string;
  type: string;
  timestamp?: string;
  payload?: unknown;
}

export interface AppServerSessionReadFacts {
  session: AppServerAgentSessionFact;
  turns?: AppServerAgentTurnFact[];
  detail?: unknown;
}

export interface AppServerFactsReplayInput {
  readModel?: AppServerSessionReadFacts;
  events?: AppServerAgentEventFact[];
  sourceCount?: number;
}

export interface AppServerFactsProjectionResult {
  events: AgentRuntimeExecutionEvent[];
  state: AgentUiProjectionState<AgentRuntimeExecutionEvent>;
  diagnostics: string[];
}

export function projectAppServerEventsToExecutionEvents(
  events: readonly AppServerAgentEventFact[] | undefined,
): AgentRuntimeExecutionEvent[] {
  return dedupeExecutionEvents(
    (events ?? [])
      .filter(
        (event) =>
          event.type !== "subagent.status_changed" &&
          event.type !== "subagent_status_changed",
      )
      .map(projectAppServerEventToExecutionEvent),
  );
}

export function projectAppServerSessionReadToExecutionEvents(
  readModel: AppServerSessionReadFacts,
): AgentRuntimeExecutionEvent[] {
  const session = readModel.session;
  const turns = readModel.turns ?? [];
  const events: AgentRuntimeExecutionEvent[] = [];

  events.push(
    compactProjectionFields({
      id: `appserver:session:${session.sessionId}:snapshot`,
      schemaVersion: "lime-runtime-event/v0.1",
      kind: "state",
      status: statusFromSessionStatus(session.status),
      eventClass: "snapshot.updated",
      runtimeId: "app-server",
      threadId: session.threadId,
      title: "Session snapshot updated",
      detail: session.status,
      payload: compactProjectionFields({
        sessionId: session.sessionId,
        appId: session.appId,
        workspaceId: session.workspaceId,
        status: session.status,
        turnCount: turns.length,
      }),
      createdAt:
        definedString(session.updatedAt) ??
        definedString(session.createdAt) ??
        defaultTimestamp(),
    } satisfies AgentRuntimeExecutionEvent),
  );

  turns.forEach((turn, index) => {
    events.push(projectAppServerTurnToExecutionEvent(turn, session, index));
  });

  return dedupeExecutionEvents(events);
}

export function replayAppServerFacts(
  input: AppServerFactsReplayInput,
): AppServerFactsProjectionResult {
  const diagnostics: string[] = [];
  const events = [
    ...(input.readModel
      ? projectAppServerSessionReadToExecutionEvents(input.readModel)
      : []),
    ...projectAppServerEventsToExecutionEvents(input.events),
  ];
  const dedupedEvents = dedupeExecutionEvents(events).sort(compareEvents);

  if (dedupedEvents.length === 0) {
    diagnostics.push("app_server_facts_empty");
  }

  return {
    events: dedupedEvents,
    state: projectAgentUiState({
      executionEvents: dedupedEvents,
      sourceCount: input.sourceCount,
    }),
    diagnostics,
  };
}

function projectAppServerEventToExecutionEvent(
  event: AppServerAgentEventFact,
): AgentRuntimeExecutionEvent {
  const payload = readRecord(event.payload) ?? {};
  const payloadStatus = readStringField(payload, [
    "status",
    "runtime_status",
    "state",
  ]);
  const eventClass = normalizeEventClass(event.type);
  const text = readText(payload);
  const actionId = readStringField(payload, [
    "request_id",
    "requestId",
    "action_id",
    "actionId",
    "id",
  ]);
  const toolCallId = readStringField(payload, [
    "toolCallId",
    "tool_call_id",
    "toolId",
    "tool_id",
    "id",
  ]);
  const artifactRefs = normalizeRefs([
    ...readStringArrayField(payload, ["artifactRefs", "artifact_refs"]),
    readStringField(payload, ["artifactRef", "artifact_ref"]),
    readStringField(payload, ["artifactId", "artifact_id"]),
  ]);
  const evidenceRefs = normalizeRefs([
    ...readStringArrayField(payload, ["evidenceRefs", "evidence_refs"]),
    readStringField(payload, ["evidenceRef", "evidence_ref"]),
    readStringField(payload, ["evidenceId", "evidence_id"]),
  ]);

  const projectedStatus =
    (isLegacyRuntimeTurnTerminalEventClass(event.type)
      ? undefined
      : (runtimeTurnTerminalProjectionFromStatus(payloadStatus)?.status ??
        statusFromDomainPayloadStatus(payloadStatus))) ??
    statusForEventClass(eventClass);
  const taskId = readStringField(payload, ["taskId", "task_id"]);
  const subagentId = readStringField(payload, ["subagentId", "subagent_id"]);
  const runId = readStringField(payload, ["runId", "run_id"]);
  const stepId = readStringField(payload, ["stepId", "step_id"]);
  const collaborationMetadata = isCollaborationEvent(eventClass)
    ? buildAgentUiCollaborationPayloadMetadata({
        payload,
        sourceType: event.type,
        collaborationKind: collaborationKindForEventClass(eventClass),
        surface: collaborationSurfaceForEventClass(eventClass),
        phase: collaborationPhaseForEvent(eventClass, payloadStatus),
        status: payloadStatus ?? projectedStatus,
        runtimeEntity: runtimeEntityForCollaborationEvent(eventClass),
        runtimeStatus: projectedStatus,
        taskId,
        agentId: subagentId,
        parentThreadId: readStringField(payload, [
          "parentThreadId",
          "parent_thread_id",
        ]),
        transcriptRef: readStringField(payload, [
          "transcriptRef",
          "transcript_ref",
        ]),
        handoffId: readStringField(payload, ["handoffId", "handoff_id"]),
      })
    : {};

  return compactProjectionFields({
    id: `appserver:${event.eventId}`,
    schemaVersion: "lime-runtime-event/v0.1",
    kind: kindForEventClass(eventClass),
    status: projectedStatus,
    eventClass,
    runtimeId: "app-server",
    threadId: event.threadId ?? event.sessionId,
    turnId: event.turnId,
    taskId,
    subagentId,
    runId,
    stepId,
    toolCallId: isToolEvent(eventClass) ? toolCallId : undefined,
    actionId: isActionEvent(eventClass) ? actionId : undefined,
    artifactId:
      readStringField(payload, ["artifactId", "artifact_id"]) ??
      artifactRefs[0],
    evidenceId:
      readStringField(payload, ["evidenceId", "evidence_id"]) ??
      evidenceRefs[0],
    sequence: event.sequence,
    phase: phaseForEventClass(eventClass),
    owner: ownerForEventClass(eventClass),
    title: titleForEvent(eventClass, payload),
    detail: detailForEvent(eventClass, payload, text),
    refIds: normalizeRefs(readStringArrayField(payload, ["refIds", "ref_ids"])),
    artifactRefs,
    evidenceRefs,
    payload: compactProjectionFields({
      ...payload,
      sourceType: event.type,
      sessionId: event.sessionId,
      text,
      messageId: readMessageId(payload),
      actionKind: readStringField(payload, ["action_type", "actionType"]),
      controls: controlsForAction(payload),
      ...extractAgentUiToolLifecyclePayloadMetadata(payload),
      ...collaborationMetadata,
    }),
    createdAt: event.timestamp ?? defaultTimestamp(),
    completedAt: isCompletedEvent(eventClass) ? event.timestamp : undefined,
  } satisfies AgentRuntimeExecutionEvent);
}

function isCollaborationEvent(eventClass: string): boolean {
  return (
    eventClass.startsWith("handoff.") ||
    eventClass.startsWith("review.") ||
    eventClass.startsWith("task.")
  );
}

function collaborationKindForEventClass(eventClass: string): string {
  if (eventClass.startsWith("handoff.")) return "specialist_handoff";
  if (eventClass.startsWith("review.")) return "collaboration_review";
  return "collaboration_task";
}

function collaborationSurfaceForEventClass(eventClass: string): string {
  if (eventClass.startsWith("handoff.")) return "handoff_lane";
  if (eventClass.startsWith("review.")) return "review_lane";
  return "task_capsule";
}

function runtimeEntityForCollaborationEvent(eventClass: string): string {
  return eventClass.startsWith("task.") ? "work_item" : "subagent_turn";
}

function collaborationPhaseForEvent(
  eventClass: string,
  status: string | undefined,
): AgentUiPhase {
  const normalized = normalizeStatus(status);
  switch (normalized) {
    case "completed":
    case "done":
    case "success":
      return "completed";
    case "failed":
    case "error":
    case "not_found":
      return "failed";
    case "aborted":
    case "cancelled":
    case "canceled":
    case "closed":
      return "cancelled";
    case "queued":
    case "pending":
    case "waiting":
    case "blocked":
      return "waiting";
    case "running":
    case "accepted":
      return "acting";
    default:
      return phaseForEventClass(eventClass) as AgentUiPhase;
  }
}

function projectAppServerTurnToExecutionEvent(
  turn: AppServerAgentTurnFact,
  session: AppServerAgentSessionFact,
  index: number,
): AgentRuntimeExecutionEvent {
  const eventClass = eventClassForTurnSnapshotStatus(turn.status);
  const terminalProjection = runtimeTurnTerminalProjectionFromStatus(
    turn.status,
  );
  return compactProjectionFields({
    id: `appserver:turn:${turn.turnId}:status:${turn.status ?? "unknown"}`,
    schemaVersion: "lime-runtime-event/v0.1",
    kind: "state",
    status:
      terminalProjection?.status ?? statusFromTurnSnapshotStatus(turn.status),
    eventClass,
    runtimeId: "app-server",
    threadId: turn.threadId ?? session.threadId,
    turnId: turn.turnId,
    sequence: index + 1,
    phase: terminalProjection?.phase ?? phaseForEventClass(eventClass),
    title: titleForTurnStatus(turn.status),
    detail: turn.status,
    payload: compactProjectionFields({
      sessionId: turn.sessionId ?? session.sessionId,
      nativeStatus: turn.status,
    }),
    createdAt:
      definedString(turn.completedAt) ??
      definedString(turn.startedAt) ??
      definedString(session.updatedAt) ??
      defaultTimestamp(),
    completedAt: turn.completedAt,
  } satisfies AgentRuntimeExecutionEvent);
}

function normalizeEventClass(type: string): string {
  const terminalEventClass = normalizeRuntimeTurnTerminalEventClass(type);
  if (terminalEventClass) return terminalEventClass;
  if (type === "message.delta_batch" || type === "message.batch") {
    return "model.delta";
  }
  if (type === "message.delta") return "model.delta";
  if (
    type === "message" ||
    type === "message.completed" ||
    type === "item.completed"
  ) {
    return "model.completed";
  }
  if (type === "thinking.delta") return "reasoning.delta";
  if (type === "artifact.snapshot") return "artifact.changed";
  if (type === "runtime.status") return "run.status";
  return type;
}

function normalizeStatus(value: string | undefined): string | undefined {
  return normalizeRuntimeStatusValue(value);
}

function statusFromDomainPayloadStatus(
  status: string | undefined,
): AgentRuntimeExecutionEventStatus | undefined {
  const normalized = normalizeStatus(status);
  switch (normalized) {
    case "failed":
    case "error":
    case "expired":
    case "closed":
    case "aborted":
    case "cancelled":
    case "not_found":
      return "failed";
    case "canceled":
      return "canceled";
    case "completed":
    case "done":
    case "success":
      return "completed";
    case "blocked":
    case "waiting":
    case "needs_input":
      return "blocked";
    case "running":
    case "queued":
    case "pending":
      return "running";
    default:
      return undefined;
  }
}

function kindForEventClass(eventClass: string): AgentRuntimeExecutionEventKind {
  if (eventClass.startsWith("model.")) return "model";
  if (eventClass.startsWith("reasoning.")) return "note";
  if (eventClass.startsWith("tool.")) return "tool";
  if (eventClass.startsWith("action.")) return "action";
  if (eventClass.startsWith("artifact.")) return "draft";
  if (eventClass.startsWith("evidence.")) return "evidence";
  if (
    eventClass.startsWith("subagent.") ||
    eventClass.startsWith("handoff.") ||
    eventClass.startsWith("review.") ||
    eventClass.startsWith("task.")
  ) {
    return "handoff";
  }
  return "state";
}

function statusForEventClass(
  eventClass: string,
): AgentRuntimeExecutionEventStatus {
  const terminalStatus = runtimeStatusForTerminalEventClass(eventClass);
  if (terminalStatus) return terminalStatus;
  if (eventClass.endsWith(".failed") || eventClass === "runtime.error") {
    return "failed";
  }
  if (eventClass === "action.required" || eventClass === "handoff.requested") {
    return "pending";
  }
  if (
    eventClass.endsWith(".completed") ||
    eventClass.endsWith(".resolved") ||
    eventClass.endsWith(".result") ||
    eventClass.endsWith(".changed") ||
    eventClass.endsWith(".verdict")
  ) {
    return "completed";
  }
  return "running";
}

function statusFromSessionStatus(
  status: string | undefined,
): AgentRuntimeExecutionEventStatus {
  const normalized = normalizeStatus(status);
  const terminalStatus =
    runtimeTurnTerminalProjectionFromStatus(normalized)?.status;
  if (terminalStatus) return terminalStatus;
  if (normalized === "completed" || normalized === "idle") return "completed";
  if (status === "waitingAction") return "blocked";
  if (normalized === "running") return "running";
  return "pending";
}

function statusFromTurnSnapshotStatus(
  status: string | undefined,
): AgentRuntimeExecutionEventStatus {
  const normalized = normalizeStatus(status);
  const terminalStatus =
    runtimeTurnTerminalProjectionFromStatus(normalized)?.status;
  if (terminalStatus) return terminalStatus;
  if (status === "waitingAction") return "pending";
  if (normalized === "running" || normalized === "accepted") return "running";
  return "pending";
}

function eventClassForTurnSnapshotStatus(status: string | undefined): string {
  const terminalEventClass =
    runtimeTurnTerminalProjectionFromStatus(status)?.eventClass;
  if (terminalEventClass) return terminalEventClass;
  const normalized = normalizeStatus(status);
  if (status === "waitingAction") return "action.required";
  if (normalized === "running" || normalized === "accepted") {
    return "turn.started";
  }
  return "turn.submitted";
}

function phaseForEventClass(eventClass: string): string {
  const terminalPhase = runtimeTurnTerminalProjectionFromStatus(
    runtimeStatusForTerminalEventClass(eventClass),
  )?.phase;
  if (terminalPhase) return terminalPhase;
  if (eventClass === "action.required") return "action_required";
  if (eventClass.endsWith(".failed") || eventClass === "runtime.error") {
    return "failed";
  }
  if (
    eventClass.endsWith(".completed") ||
    eventClass.endsWith(".resolved") ||
    eventClass.endsWith(".result") ||
    eventClass.endsWith(".changed")
  ) {
    return "completed";
  }
  if (eventClass.startsWith("tool.")) return "tool_running";
  if (eventClass.startsWith("model.")) return "streaming";
  return "preparing";
}

function ownerForEventClass(eventClass: string): string {
  if (eventClass.startsWith("tool.")) return "runtime";
  if (eventClass.startsWith("artifact.")) return "artifact";
  if (eventClass.startsWith("evidence.")) return "evidence";
  return "runtime";
}

function titleForEvent(
  eventClass: string,
  payload: Record<string, unknown>,
): string {
  return (
    readStringField(payload, ["title", "name", "toolName", "tool_name"]) ??
    eventClass
  );
}

function titleForTurnStatus(status: string | undefined): string {
  if (status === "completed") return "Turn completed";
  if (status === "failed") return "Turn failed";
  if (status === "canceled") return "Turn canceled";
  if (status === "waitingAction") return "Action required";
  if (status === "queued") return "Turn queued";
  return "Turn started";
}

function detailForEvent(
  eventClass: string,
  payload: Record<string, unknown>,
  text: string | undefined,
): string | undefined {
  if (eventClass.startsWith("model.") || eventClass.startsWith("reasoning.")) {
    return text;
  }
  return readStringField(payload, ["detail", "message", "error", "reason"]);
}

function readText(payload: Record<string, unknown>): string | undefined {
  const direct = readStringField(payload, ["text", "delta", "message"]);
  if (direct) return direct;

  const message = readRecord(payload.message);
  const messageText = readStringField(message, ["text", "delta", "message"]);
  if (messageText) return messageText;

  const content = Array.isArray(message?.content)
    ? message.content
    : Array.isArray(payload.content)
      ? payload.content
      : [];
  return content
    .map((part) => readStringField(readRecord(part), ["text", "content"]))
    .filter((value): value is string => Boolean(value))
    .join("");
}

function readMessageId(payload: Record<string, unknown>): string | undefined {
  const message = readRecord(payload.message);
  return (
    readStringField(payload, ["messageId", "message_id"]) ??
    readStringField(message, ["id", "messageId", "message_id"])
  );
}

function controlsForAction(payload: Record<string, unknown>): string[] {
  const controls = readStringArrayField(payload, ["controls"]);
  if (controls.length > 0) return controls;

  const actionType = readStringField(payload, ["action_type", "actionType"]);
  if (actionType === "ask_user" || actionType === "elicitation") {
    return ["answer"];
  }
  if (actionType === "tool_confirmation") {
    return ["approve", "reject"];
  }
  return [];
}

function isToolEvent(eventClass: string): boolean {
  return eventClass.startsWith("tool.");
}

function isActionEvent(eventClass: string): boolean {
  return eventClass.startsWith("action.");
}

function isCompletedEvent(eventClass: string): boolean {
  const status = statusForEventClass(eventClass);
  return status === "completed" || status === "failed" || status === "canceled";
}

function normalizeRefs(values: Array<string | undefined>): string[] {
  return Array.from(
    new Set(
      values
        .map((value) => definedString(value))
        .filter((value): value is string => Boolean(value)),
    ),
  );
}

function dedupeExecutionEvents(
  events: readonly AgentRuntimeExecutionEvent[],
): AgentRuntimeExecutionEvent[] {
  const byId = new Map<string, AgentRuntimeExecutionEvent>();
  events.forEach((event) => {
    byId.set(event.id, event);
  });
  return Array.from(byId.values());
}

function compareEvents(
  left: AgentRuntimeExecutionEvent,
  right: AgentRuntimeExecutionEvent,
): number {
  const leftSequence = readNumberField({ sequence: left.sequence }, [
    "sequence",
  ]);
  const rightSequence = readNumberField({ sequence: right.sequence }, [
    "sequence",
  ]);
  if (leftSequence !== undefined && rightSequence !== undefined) {
    return leftSequence - rightSequence;
  }
  return left.createdAt.localeCompare(right.createdAt);
}

function defaultTimestamp(): string {
  return "1970-01-01T00:00:00.000Z";
}
