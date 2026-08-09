import type { AgentRuntimeLifecycleNotification } from "@limecloud/app-server-client";
import {
  createRuntimeSequenceVerifier,
  type AgentRuntimeExecutionEvent,
  type RuntimeSequenceViolation,
} from "@limecloud/agent-ui-contracts";

export type AgentRuntimeSequenceVerifierMode =
  | "fail-closed"
  | "collect-diagnostics"
  | "off";

export interface AgentRuntimeSequenceVerifierLike {
  push(event: AgentRuntimeExecutionEvent): RuntimeSequenceViolation[];
  getViolations(): RuntimeSequenceViolation[];
  finalize?(): RuntimeSequenceViolation[];
}

export interface AgentRuntimeSequenceVerifierOptions {
  verifier?: AgentRuntimeSequenceVerifierLike;
  mode?: AgentRuntimeSequenceVerifierMode;
}

export interface AgentRuntimeSequenceVerificationResult {
  accepted: boolean;
  violations: RuntimeSequenceViolation[];
}

type CanonicalThread = Extract<
  AgentRuntimeLifecycleNotification,
  { method: "thread/started" }
>["params"]["thread"];
type CanonicalTurn = Extract<
  AgentRuntimeLifecycleNotification,
  { method: "turn/started" | "turn/completed" }
>["params"]["turn"];
type CanonicalThreadItem = Extract<
  AgentRuntimeLifecycleNotification,
  { method: "item/started" | "item/completed" }
>["params"]["item"];
type EntityLifecycleNotification = Extract<
  AgentRuntimeLifecycleNotification,
  {
    method:
      | "thread/started"
      | "turn/started"
      | "turn/completed"
      | "item/started"
      | "item/completed";
  }
>;
type GuardianReviewNotification = Extract<
  AgentRuntimeLifecycleNotification,
  {
    method:
      | "item/autoApprovalReview/started"
      | "item/autoApprovalReview/completed";
  }
>;
type ItemStreamingNotification = Extract<
  AgentRuntimeLifecycleNotification,
  {
    method:
      | "item/agentMessage/delta"
      | "item/commandExecution/outputDelta"
      | "item/commandExecution/terminalInteraction"
      | "item/fileChange/patchUpdated"
      | "item/mcpToolCall/progress"
      | "item/plan/delta"
      | "item/reasoning/summaryTextDelta"
      | "item/reasoning/summaryPartAdded"
      | "item/reasoning/textDelta";
  }
>;
const CANONICAL_ITEM_TYPES = [
  "userMessage",
  "hookPrompt",
  "agentMessage",
  "plan",
  "reasoning",
  "commandExecution",
  "fileChange",
  "mcpToolCall",
  "dynamicToolCall",
  "collabAgentToolCall",
  "subAgentActivity",
  "webSearch",
  "imageView",
  "sleep",
  "imageGeneration",
  "enteredReviewMode",
  "exitedReviewMode",
  "contextCompaction",
  "unknownItem",
] as const;
type CanonicalItemType = (typeof CANONICAL_ITEM_TYPES)[number];
const CANONICAL_ITEM_TYPE_SET = new Set<string>(CANONICAL_ITEM_TYPES);

const STATUSFUL_ITEM_TYPES = new Set<CanonicalItemType>([
  "commandExecution",
  "fileChange",
  "mcpToolCall",
  "dynamicToolCall",
  "collabAgentToolCall",
  "imageGeneration",
]);

export class AgentRuntimeSequenceViolationError extends Error {
  readonly violations: RuntimeSequenceViolation[];

  constructor(violations: RuntimeSequenceViolation[]) {
    super(
      violations.length > 0
        ? `Agent runtime event sequence violation: ${violations.map((violation) => violation.code).join(", ")}`
        : "Agent runtime event sequence violation.",
    );
    this.name = "AgentRuntimeSequenceViolationError";
    this.violations = violations;
  }
}

export class AgentRuntimeEventSequenceGate {
  readonly #verifier: AgentRuntimeSequenceVerifierLike | undefined;
  readonly #mode: AgentRuntimeSequenceVerifierMode;

  constructor(options: AgentRuntimeSequenceVerifierOptions = {}) {
    this.#mode = options.mode ?? "fail-closed";
    this.#verifier =
      this.#mode === "off"
        ? undefined
        : (options.verifier ?? createRuntimeSequenceVerifier());
  }

  verify(
    notification: AgentRuntimeLifecycleNotification,
  ): AgentRuntimeSequenceVerificationResult {
    if (
      !this.#verifier ||
      isItemStreamingNotification(notification) ||
      isGuardianReviewNotification(notification) ||
      notification.method === "thread/settings/updated"
    ) {
      return { accepted: true, violations: [] };
    }
    const event = runtimeExecutionEventFromLifecycleNotification(notification);
    const violations = this.#verifier.push(event);
    return {
      accepted: this.#mode !== "fail-closed" || violations.length === 0,
      violations,
    };
  }

  getViolations(): RuntimeSequenceViolation[] {
    return this.#verifier?.getViolations() ?? [];
  }

  sequenceViolationError(): AgentRuntimeSequenceViolationError {
    return new AgentRuntimeSequenceViolationError(this.getViolations());
  }
}

function isItemStreamingNotification(
  notification: AgentRuntimeLifecycleNotification,
): notification is ItemStreamingNotification {
  const method = notification.method;
  return (
    method === "item/agentMessage/delta" ||
    method === "item/commandExecution/outputDelta" ||
    method === "item/commandExecution/terminalInteraction" ||
    method === "item/fileChange/patchUpdated" ||
    method === "item/mcpToolCall/progress" ||
    method === "item/plan/delta" ||
    method === "item/reasoning/summaryTextDelta" ||
    method === "item/reasoning/summaryPartAdded" ||
    method === "item/reasoning/textDelta"
  );
}

function isGuardianReviewNotification(
  notification: AgentRuntimeLifecycleNotification,
): notification is GuardianReviewNotification {
  return (
    notification.method === "item/autoApprovalReview/started" ||
    notification.method === "item/autoApprovalReview/completed"
  );
}

/**
 * Convert a canonical Thread/Turn/Item entity into the package's verifier
 * input. Lifecycle state is read only from the canonical v2 entity.
 */
export function runtimeExecutionEventFromLifecycleNotification(
  event: EntityLifecycleNotification,
): AgentRuntimeExecutionEvent {
  if (event.method === "thread/started") {
    const thread = requireEntity(event.params.thread, "Thread");
    const threadId = requireEntityId(thread, "Thread");
    const observedAtMs = unixSecondsToMillis(
      thread.updatedAt,
      "Thread.updatedAt",
    );
    return {
      id: `thread:${threadId}:started:${observedAtMs}`,
      runtimeId: "app-server",
      threadId,
      sequence: observedAtMs,
      kind: "state",
      status: threadStatus(thread.status?.type),
      eventClass: "run.status",
      title: thread.name ?? (thread.preview || "Thread"),
      payload: { lifecycle: event, canonical: thread },
      createdAt: dateFromUnixSeconds(thread.createdAt, "Thread.createdAt"),
    };
  }

  if (event.method === "turn/started" || event.method === "turn/completed") {
    const turn = requireEntity(event.params.turn, "Turn");
    const turnId = requireEntityId(turn, "Turn");
    const observedAt = turnTimestamp(turn, event.method);
    const observedAtMs = unixSecondsToMillis(observedAt, "Turn timestamp");
    return {
      id: `turn:${turnId}:${turn.status}:${observedAtMs}`,
      runtimeId: "app-server",
      threadId: event.params.threadId,
      turnId,
      sequence: observedAtMs,
      kind: "state",
      status: turnStatus(turn.status),
      eventClass: turnEventClass(turn.status),
      title: "Turn",
      payload: { lifecycle: event, canonical: turn },
      createdAt: dateFromUnixSeconds(
        turn.startedAt ?? observedAt,
        "Turn.startedAt",
      ),
      completedAt:
        turn.completedAt == null
          ? undefined
          : dateFromUnixSeconds(turn.completedAt, "Turn.completedAt"),
    };
  }

  const item = requireEntity(event.params.item, "ThreadItem");
  const itemId = requireStringField(item, "id");
  const type = canonicalItemType(item);
  const status = itemExecutionStatus(item, type, event.method);
  const observedAtMs =
    event.method === "item/started"
      ? event.params.startedAtMs
      : event.params.completedAtMs;
  requireUnixMilliseconds(observedAtMs, `${event.method} timestamp`);
  return {
    id: `item:${itemId}:${type}:${status}:${observedAtMs}`,
    runtimeId: "app-server",
    threadId: event.params.threadId,
    turnId: event.params.turnId,
    toolCallId: itemToolCallId(type, itemId),
    sequence: observedAtMs,
    kind: itemRuntimeKind(type),
    status,
    eventClass: itemEventClass(type, status),
    title: itemTitle(item, type),
    payload: { lifecycle: event, canonical: item },
    createdAt: dateFromUnixMilliseconds(observedAtMs),
    completedAt:
      event.method === "item/completed"
        ? dateFromUnixMilliseconds(observedAtMs)
        : undefined,
  };
}

function dateFromUnixSeconds(value: number, field: string): string {
  return new Date(requireUnixSeconds(value, field) * 1_000).toISOString();
}

function requireUnixSeconds(value: number, field: string): number {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${field} must be a finite Unix timestamp.`);
  }
  return value;
}

function unixSecondsToMillis(value: number, field: string): number {
  return requireUnixSeconds(value, field) * 1_000;
}

function requireUnixMilliseconds(value: number, field: string): number {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${field} must be a finite Unix timestamp.`);
  }
  return value;
}

function dateFromUnixMilliseconds(value: number): string {
  return new Date(value).toISOString();
}

function threadStatus(
  status: NonNullable<CanonicalThread["status"]>["type"] | undefined,
): AgentRuntimeExecutionEvent["status"] {
  switch (status) {
    case "systemError":
      return "failed";
    case "active":
      return "running";
    case "notLoaded":
    case "idle":
    case undefined:
      return "pending";
  }
}

function turnTimestamp(
  turn: CanonicalTurn,
  method: "turn/started" | "turn/completed",
): number {
  const value =
    method === "turn/completed"
      ? (turn.completedAt ?? turn.startedAt)
      : turn.startedAt;
  if (value == null || !Number.isFinite(value)) {
    throw new TypeError(`${method} requires a finite Turn timestamp.`);
  }
  return value;
}

function turnStatus(
  status: CanonicalTurn["status"],
): AgentRuntimeExecutionEvent["status"] {
  switch (status) {
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "interrupted":
      return "canceled";
    case "inProgress":
      return "running";
    default:
      throw new TypeError(`Unsupported canonical Turn status '${status}'.`);
  }
}

function turnEventClass(status: CanonicalTurn["status"]): string {
  switch (status) {
    case "completed":
      return "turn.completed";
    case "failed":
      return "turn.failed";
    case "interrupted":
      return "turn.canceled";
    case "inProgress":
      return "turn.started";
    default:
      throw new TypeError(`Unsupported canonical Turn status '${status}'.`);
  }
}

function itemExecutionStatus(
  item: CanonicalThreadItem,
  itemType: CanonicalItemType,
  method: "item/started" | "item/completed",
): AgentRuntimeExecutionEvent["status"] {
  if (!STATUSFUL_ITEM_TYPES.has(itemType)) {
    return method === "item/started" ? "running" : "completed";
  }
  const status = requireStringField(item, "status");
  switch (status) {
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "declined":
      return "canceled";
    case "inProgress":
      return "running";
    default:
      throw new TypeError(
        `Unsupported ${itemType} status '${status}' in canonical ThreadItem.`,
      );
  }
}

function itemEventClass(
  type: CanonicalItemType,
  status: AgentRuntimeExecutionEvent["status"],
): string {
  const family = itemEventFamily(type);
  if (family === "tool") {
    if (status === "running") return "tool.started";
    if (status === "completed") return "tool.result";
    return "tool.failed";
  }
  if (family === "model") {
    if (status === "running") return "model.delta";
    if (status === "completed") return "model.completed";
    return "model.failed";
  }
  if (family === "command") {
    if (status === "running") return "command.started";
    return "command.exited";
  }
  if (family === "patch") {
    if (status === "running") return "patch.started";
    if (status === "completed") return "patch.completed";
    return "patch.failed";
  }
  if (status === "failed") return `${family}.failed`;
  if (status === "canceled") return `${family}.canceled`;
  if (status === "completed") return `${family}.completed`;
  return `${family}.started`;
}

function itemEventFamily(type: CanonicalItemType): string {
  switch (type) {
    case "agentMessage":
    case "userMessage":
    case "hookPrompt":
    case "plan":
    case "reasoning":
      return "model";
    case "contextCompaction":
      return "context";
    case "fileChange":
      return "patch";
    case "imageView":
      return "artifact";
    case "subAgentActivity":
    case "collabAgentToolCall":
      return "handoff";
    case "enteredReviewMode":
    case "exitedReviewMode":
      return "runtime";
    case "unknownItem":
      return "runtime";
    case "commandExecution":
      return "command";
    case "mcpToolCall":
    case "dynamicToolCall":
    case "webSearch":
    case "sleep":
    case "imageGeneration":
      return "tool";
  }
}

function itemRuntimeKind(
  type: CanonicalItemType,
): AgentRuntimeExecutionEvent["kind"] {
  switch (itemEventFamily(type)) {
    case "model":
      return "model";
    case "context":
      return "context";
    case "artifact":
    case "patch":
      return "draft";
    case "handoff":
      return "handoff";
    case "runtime":
      return "note";
    default:
      return "tool";
  }
}

function itemTitle(item: CanonicalThreadItem, type: CanonicalItemType): string {
  switch (type) {
    case "agentMessage":
    case "plan":
      return optionalStringField(item, "text") ?? type;
    case "userMessage":
      return userMessageTitle(item);
    case "mcpToolCall":
    case "dynamicToolCall":
    case "collabAgentToolCall":
      return optionalStringField(item, "tool") ?? type;
    case "commandExecution":
      return optionalStringField(item, "command") ?? type;
    case "imageView":
      return optionalStringField(item, "path") ?? type;
    case "subAgentActivity":
      return optionalStringField(item, "agentPath") ?? type;
    case "webSearch":
      return optionalStringField(item, "query") ?? type;
    case "enteredReviewMode":
    case "exitedReviewMode":
      return optionalStringField(item, "review") ?? type;
    default:
      return type;
  }
}

function canonicalItemType(item: CanonicalThreadItem): CanonicalItemType {
  const type = optionalStringField(item, "type");
  if (!type || !CANONICAL_ITEM_TYPE_SET.has(type)) {
    throw new TypeError(
      "Canonical ThreadItem.type must be a supported v2 item tag.",
    );
  }
  return type as CanonicalItemType;
}

function itemToolCallId(
  type: CanonicalItemType,
  itemId: string,
): string | undefined {
  switch (type) {
    case "commandExecution":
    case "fileChange":
    case "mcpToolCall":
    case "dynamicToolCall":
    case "collabAgentToolCall":
    case "webSearch":
    case "sleep":
    case "imageGeneration":
      return itemId;
    default:
      return undefined;
  }
}

function userMessageTitle(item: CanonicalThreadItem): string {
  const content = recordField(item, "content");
  if (!Array.isArray(content)) {
    return "userMessage";
  }
  for (const part of content) {
    if (
      isRecord(part) &&
      part.type === "text" &&
      typeof part.text === "string"
    ) {
      return part.text;
    }
  }
  return "userMessage";
}

function requireStringField(value: object, field: string): string {
  const result = optionalStringField(value, field);
  if (!result) {
    throw new TypeError(`Canonical ThreadItem.${field} must be a string.`);
  }
  return result;
}

function requireEntity<Value extends object>(
  value: Value,
  entity: string,
): Value {
  if (!isRecord(value)) {
    throw new TypeError(`Canonical ${entity} must be an object.`);
  }
  return value;
}

function requireEntityId(value: object, entity: string): string {
  const id = optionalStringField(value, "id");
  if (!id) {
    throw new TypeError(`Canonical ${entity}.id must be a string.`);
  }
  return id;
}

function optionalStringField(value: object, field: string): string | undefined {
  const result = recordField(value, field);
  return typeof result === "string" ? result : undefined;
}

function recordField(value: object, field: string): unknown {
  return (value as Record<string, unknown>)[field];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
