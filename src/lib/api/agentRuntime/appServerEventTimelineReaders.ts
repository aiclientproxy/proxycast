import type { AppServerAgentEvent } from "@/lib/api/appServer";
import {
  normalizeRecord,
  readBoolean,
  readFiniteNumber,
  readString,
  readStringArray,
  readToolCallId,
  readToolName,
} from "./appServerEventPayloadUtils";
import { readCanonicalToolThreadItem } from "./appServerCanonicalItemReader";

export function readArtifactSnapshotSignalFromPayload(
  payload: Record<string, unknown>,
  event: AppServerAgentEvent,
): Record<string, unknown> {
  const artifact = normalizeRecord(payload.artifact) ?? payload;
  const artifactRef = readString(
    artifact,
    "artifactRef",
    "artifact_ref",
    "artifactId",
    "artifact_id",
    "id",
  );
  const artifactId =
    readString(artifact, "artifactId", "artifact_id", "id", "artifactRef") ??
    event.eventId;
  const filePath = readString(
    artifact,
    "filePath",
    "file_path",
    "path",
    "artifactPath",
    "artifact_path",
  );
  const sidecarRef =
    normalizeRecord(artifact.sidecarRef) ?? normalizeRecord(payload.sidecarRef);
  const metadata = {
    ...(normalizeRecord(artifact.metadata) ??
      normalizeRecord(payload.metadata) ??
      {}),
    sessionId: event.sessionId,
    ...(event.threadId ? { threadId: event.threadId } : {}),
    ...(event.turnId ? { turnId: event.turnId } : {}),
    artifactId,
    ...(artifactRef ? { artifactRef, appServerArtifactRef: artifactRef } : {}),
    ...(filePath ? { filePath } : {}),
    ...(sidecarRef ? { sidecarRef } : {}),
    ...copyDefinedFields(artifact, [
      "contentStatus",
      "contentBytes",
      "contentSha256",
    ]),
    ...copyDefinedFields(payload, [
      "contentStatus",
      "contentBytes",
      "contentSha256",
    ]),
  };
  return {
    ...artifact,
    artifactId,
    artifact_id: readString(artifact, "artifact_id") ?? artifactId,
    ...(artifactRef ? { artifactRef, artifact_ref: artifactRef } : {}),
    ...(filePath ? { filePath, file_path: filePath } : {}),
    ...(typeof artifact.content === "string"
      ? { content: artifact.content }
      : typeof payload.content === "string"
        ? { content: payload.content }
        : {}),
    metadata,
  };
}

function copyDefinedFields(
  record: Record<string, unknown>,
  keys: string[],
): Record<string, unknown> {
  return Object.fromEntries(
    keys
      .map((key) => [key, record[key]] as const)
      .filter(([, value]) => typeof value !== "undefined"),
  );
}

function normalizeHookTimelineStatus(
  status: string | undefined,
): "in_progress" | "completed" | "failed" {
  switch (status) {
    case "running":
    case "in_progress":
      return "in_progress";
    case "failed":
    case "blocked":
    case "stopped":
      return "failed";
    default:
      return "completed";
  }
}

function readHookRunPayload(payload: Record<string, unknown>) {
  return normalizeRecord(payload.run) ?? payload;
}

function readHookOutputEntries(value: unknown):
  | Array<{
      kind: string;
      text: string;
    }>
  | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const entries = value
    .map((entry) => {
      const record = normalizeRecord(entry);
      if (!record) {
        return null;
      }
      const kind = readString(record, "kind", "type");
      const text = readString(record, "text", "message", "content");
      return kind && text ? { kind, text } : null;
    })
    .filter((entry): entry is { kind: string; text: string } => Boolean(entry));
  return entries.length > 0 ? entries : undefined;
}

export function readHookItemFromPayload(
  payload: Record<string, unknown>,
  event: AppServerAgentEvent,
): Record<string, unknown> {
  const run = readHookRunPayload(payload);
  const rawStatus =
    readString(run, "status", "hookStatus", "hook_status") ??
    (event.type.endsWith(".started") || event.type.endsWith("/started")
      ? "running"
      : "completed");
  const status = normalizeHookTimelineStatus(rawStatus);
  const runId =
    readString(run, "id", "runId", "run_id", "hookRunId", "hook_run_id") ??
    event.eventId;
  const entries = readHookOutputEntries(run.entries ?? payload.entries);
  const output =
    readString(run, "output", "text", "message", "statusMessage") ??
    entries?.map((entry) => `${entry.kind}: ${entry.text}`).join("\n");
  const metadata =
    normalizeRecord(run.metadata) ?? normalizeRecord(payload.metadata);

  return {
    ...readAgentThreadItemBase(run, event, status),
    id: `item_${runId}`,
    turn_id: event.turnId,
    type: "hook",
    status,
    completed_at:
      status === "in_progress"
        ? undefined
        : (readString(run, "completedAt", "completed_at") ?? event.timestamp),
    run_id: runId,
    event_name: readString(run, "eventName", "event_name", "hookEvent"),
    handler_type: readString(run, "handlerType", "handler_type"),
    execution_mode: readString(run, "executionMode", "execution_mode"),
    scope: readString(run, "scope", "hookScope"),
    source_path: readString(run, "sourcePath", "source_path"),
    source: readString(run, "source"),
    display_order: readFiniteNumber(run, "displayOrder", "display_order"),
    status_message: readString(
      run,
      "statusMessage",
      "status_message",
      "message",
    ),
    duration_ms: readFiniteNumber(run, "durationMs", "duration_ms"),
    entries,
    output,
    target_item_id: readString(
      run,
      "targetItemId",
      "target_item_id",
      "toolCallId",
      "tool_call_id",
    ),
    hook_status: rawStatus,
    metadata: {
      ...(metadata ?? {}),
      eventClass: event.type,
      raw: run,
    },
  };
}

function readCommandString(payload: Record<string, unknown>): string {
  const argv = Array.isArray(payload.commandArgv)
    ? payload.commandArgv.filter(
        (part): part is string => typeof part === "string",
      )
    : [];
  return (
    readString(
      payload,
      "canonicalCommand",
      "canonical_command",
      "command",
      "commandSummary",
      "command_summary",
    ) ??
    argv.join(" ") ??
    ""
  );
}

function readCommandOutput(payload: Record<string, unknown>): string {
  return (
    readString(
      payload,
      "aggregated_output",
      "aggregatedOutput",
      "output",
      "preview",
      "delta",
      "text",
    ) ?? ""
  );
}

export function readCommandExecutionItemFromPayload(
  payload: Record<string, unknown>,
  event: AppServerAgentEvent,
  status: "in_progress" | "completed" | "failed",
): Record<string, unknown> {
  const commandId = readToolCallId(payload) ?? event.eventId;
  const startedAt =
    readString(payload, "startedAt", "started_at", "timestamp") ??
    event.timestamp;
  const exitCode = readFiniteNumber(payload, "exitCode", "exit_code");
  const metadata = normalizeRecord(payload.metadata);
  return {
    id: commandId,
    thread_id: event.threadId ?? event.sessionId,
    turn_id: event.turnId ?? "",
    sequence: event.sequence,
    status,
    started_at: startedAt,
    completed_at: status === "in_progress" ? undefined : event.timestamp,
    updated_at: event.timestamp,
    type: "command_execution",
    command: readCommandString(payload),
    cwd: readString(payload, "cwd", "workingDirectory", "working_dir") ?? "",
    aggregated_output: readCommandOutput(payload),
    exit_code: exitCode,
    error: readString(payload, "error", "message"),
    metadata: {
      ...(metadata ?? {}),
      eventClass: event.type,
      outputRef: readString(payload, "outputRef", "output_ref"),
      contentRef: readString(payload, "contentRef", "content_ref"),
      refIds: readStringArray(payload, "refIds", "ref_ids"),
    },
  };
}

export function readPatchItemFromPayload(
  payload: Record<string, unknown>,
  event: AppServerAgentEvent,
  status: "in_progress" | "completed" | "failed",
): Record<string, unknown> {
  const patchId =
    readString(
      payload,
      "patchId",
      "patch_id",
      "toolCallId",
      "tool_call_id",
      "id",
    ) ?? event.eventId;
  const changes = readPatchChanges(payload);
  const paths = readPatchPaths(payload, changes);
  const stdout = readString(payload, "stdout", "output", "summary");
  const stderr = readString(payload, "stderr", "error", "message", "reason");
  const metadata = normalizeRecord(payload.metadata);
  return {
    id: patchId,
    thread_id: event.threadId ?? event.sessionId,
    turn_id: event.turnId ?? "",
    sequence: event.sequence,
    status,
    started_at:
      readString(payload, "startedAt", "started_at", "timestamp") ??
      event.timestamp,
    completed_at: status === "in_progress" ? undefined : event.timestamp,
    updated_at: event.timestamp,
    type: "patch",
    changes,
    file_status:
      readString(payload, "status", "state") ??
      (status === "in_progress"
        ? "inProgress"
        : status === "completed"
          ? "completed"
          : "failed"),
    text:
      readString(payload, "text", "patch", "message") ??
      (paths.length > 0
        ? `Patch changed ${paths.join(", ")}`
        : "Patch applied"),
    summary: paths.length > 0 ? paths : undefined,
    paths: paths.length > 0 ? paths : undefined,
    success:
      readBoolean(payload, "success") ??
      (status === "failed" ? false : status === "completed" ? true : undefined),
    stdout,
    stderr,
    metadata: {
      ...(metadata ?? {}),
      eventClass: event.type,
      autoApproved: readBoolean(payload, "autoApproved", "auto_approved"),
      status: readString(payload, "status"),
    },
  };
}

function readPatchChanges(payload: Record<string, unknown>) {
  if (!Array.isArray(payload.changes)) {
    return [];
  }
  return payload.changes.flatMap((value) => {
    const change = normalizeRecord(value);
    const path = readString(change ?? {}, "path", "filePath", "file_path");
    const kindRecord = normalizeRecord(change?.kind);
    const kindType =
      readString(kindRecord ?? {}, "type") ??
      readString(change ?? {}, "kind", "changeKind", "change_kind");
    if (
      !path ||
      (kindType !== "add" && kindType !== "delete" && kindType !== "update")
    ) {
      return [];
    }
    const movePath =
      readString(kindRecord ?? {}, "move_path") ??
      readString(change ?? {}, "movePath", "move_path");
    return [
      {
        path,
        kind:
          kindType === "update" && movePath
            ? { type: kindType, move_path: movePath }
            : { type: kindType },
        diff: readString(change ?? {}, "diff", "patch", "content") ?? "",
      },
    ];
  });
}

function readPatchPaths(
  payload: Record<string, unknown>,
  changes: Array<{ path: string }>,
): string[] {
  const directPaths =
    readStringArray(payload, "paths", "changedFiles", "changed_files") ?? [];
  if (directPaths.length > 0) {
    return directPaths;
  }
  return changes.map((change) => change.path);
}

export function readFileReadItemFromPayload(
  payload: Record<string, unknown>,
  event: AppServerAgentEvent,
): Record<string, unknown> {
  const path = readString(payload, "path", "filePath", "file_path") ?? "";
  const contentRef = readString(payload, "contentRef", "content_ref");
  const outputRef = readString(payload, "outputRef", "output_ref");
  const metadata = normalizeRecord(payload.metadata);
  return {
    id: readToolCallId(payload) ?? event.eventId,
    thread_id: event.threadId ?? event.sessionId,
    turn_id: event.turnId ?? "",
    sequence: event.sequence,
    status: "completed",
    started_at: event.timestamp,
    completed_at: event.timestamp,
    updated_at: event.timestamp,
    type: "file_artifact",
    path,
    source: "file_read",
    metadata: {
      ...(metadata ?? {}),
      eventClass: event.type,
      toolCallId: readToolCallId(payload),
      toolName: readToolName(payload),
      outputRef,
      contentRef,
      refIds: readStringArray(payload, "refIds", "ref_ids"),
      startLine: readFiniteNumber(payload, "startLine", "start_line"),
      endLine: readFiniteNumber(payload, "endLine", "end_line"),
      fileType: readString(payload, "fileType", "file_type"),
    },
  };
}

export function readActionScope(
  payload: Record<string, unknown>,
  event: AppServerAgentEvent,
): Record<string, unknown> | undefined {
  const scope = normalizeRecord(payload.scope);
  const sessionId =
    readString(scope ?? {}, "session_id", "sessionId") ?? event.sessionId;
  const threadId =
    readString(scope ?? {}, "thread_id", "threadId") ??
    event.threadId ??
    event.sessionId;
  const turnId = readString(scope ?? {}, "turn_id", "turnId") ?? event.turnId;
  if (!sessionId && !threadId && !turnId) {
    return undefined;
  }
  return {
    ...(sessionId ? { session_id: sessionId } : {}),
    ...(threadId ? { thread_id: threadId } : {}),
    ...(turnId ? { turn_id: turnId } : {}),
  };
}

export function readActionQuestions(
  payload: Record<string, unknown>,
): unknown[] | undefined {
  if (Array.isArray(payload.questions)) {
    return payload.questions;
  }
  const data = normalizeRecord(payload.data);
  if (Array.isArray(data?.questions)) {
    return data.questions;
  }
  return undefined;
}

export function readActionResolvedData(
  payload: Record<string, unknown>,
): Record<string, unknown> {
  const data = normalizeRecord(payload.data);
  return {
    ...(data ?? {}),
    ...(readBoolean(payload, "approved", "confirmed", "approve") !== undefined
      ? {
          approved: readBoolean(payload, "approved", "confirmed", "approve"),
        }
      : {}),
    ...(readString(payload, "feedback", "message", "reason")
      ? { feedback: readString(payload, "feedback", "message", "reason") }
      : {}),
    ...(readString(payload, "decision", "status")
      ? { decision: readString(payload, "decision", "status") }
      : {}),
    ...(readString(payload, "permission_mode", "permissionMode")
      ? {
          permission_mode: readString(
            payload,
            "permission_mode",
            "permissionMode",
          ),
        }
      : {}),
  };
}

export function projectTextDeltaBatchPayload(
  basePayload: Record<string, unknown>,
  payload: Record<string, unknown>,
): Record<string, unknown> {
  const chunks = readStringArray(payload, "chunks", "deltas") ?? [];
  return {
    ...basePayload,
    type: "text_delta_batch",
    text: readAgentMessageDeltaText(payload) ?? chunks.join(""),
    chunks,
    itemId: readAgentMessageItemId(payload),
    item_id: readAgentMessageItemId(payload),
    phase: readAgentMessagePhase(payload),
    boundary:
      readString(payload, "boundary", "streamBoundary", "stream_kind") ??
      "provider",
  };
}

export function readAgentMessageItemId(
  payload: Record<string, unknown>,
): string | undefined {
  return readString(
    payload,
    "itemId",
    "item_id",
    "id",
    "messageId",
    "message_id",
  );
}

export function readAgentMessageDeltaText(
  payload: Record<string, unknown>,
): string | undefined {
  return (
    readString(payload, "text", "delta", "message") ??
    readString(
      normalizeRecord(payload.content) ?? {},
      "text",
      "delta",
      "message",
    )
  );
}

export function readAgentMessagePhase(
  payload: Record<string, unknown>,
): string | undefined {
  return readString(
    payload,
    "phase",
    "messagePhase",
    "message_phase",
    "streamPhase",
    "stream_phase",
  );
}

export function readAgentMessageFromPayload(
  payload: Record<string, unknown>,
  timestamp: string,
) {
  const message = normalizeRecord(payload.message);
  const content = Array.isArray(message?.content)
    ? message.content
    : Array.isArray(payload.content)
      ? payload.content
      : readString(payload, "text", "delta", "message")
        ? [
            {
              type: "text",
              text: readString(payload, "text", "delta", "message") ?? "",
            },
          ]
        : [];

  return {
    id: readString(message ?? payload, "id", "messageId"),
    role: readString(message ?? payload, "role") ?? "assistant",
    content,
    timestamp: readTimestampMs(message?.timestamp, timestamp),
  };
}

export function readCanonicalAgentThreadTurn(
  turn: Record<string, unknown>,
  event: AppServerAgentEvent,
  fallbackStatus: string,
): Record<string, unknown> {
  const timestamp = event.timestamp;
  const error = normalizeRecord(turn.error);
  const createdAt = readTurnTimestamp(
    turn,
    timestamp,
    ["created_at", "createdAt"],
    ["createdAtMs"],
  );
  return {
    id: readString(turn, "id", "turnId", "turn_id") ?? event.turnId ?? "",
    thread_id:
      readString(turn, "thread_id", "threadId") ??
      event.threadId ??
      event.sessionId,
    prompt_text: readString(turn, "prompt_text", "promptText", "prompt") ?? "",
    status: normalizeAgentThreadTurnStatus(
      readString(turn, "status"),
      fallbackStatus,
    ),
    started_at: readTurnTimestamp(
      turn,
      createdAt,
      ["started_at", "startedAt"],
      ["startedAtMs"],
    ),
    completed_at: readOptionalTurnTimestamp(
      turn,
      ["completed_at", "completedAt"],
      ["completedAtMs"],
    ),
    error_message:
      readString(turn, "error_message", "errorMessage", "message") ??
      readString(error ?? {}, "message") ??
      (fallbackStatus === "failed" ? "App Server turn failed" : undefined),
    unified_diff: readString(turn, "unified_diff", "unifiedDiff"),
    moderation_metadata: Object.prototype.hasOwnProperty.call(
      turn,
      "moderation_metadata",
    )
      ? turn.moderation_metadata
      : turn.moderationMetadata,
    created_at: createdAt,
    updated_at: readTurnTimestamp(
      turn,
      timestamp,
      ["updated_at", "updatedAt"],
      ["updatedAtMs"],
    ),
  };
}

function normalizeAgentThreadTurnStatus(
  status: string | undefined,
  fallbackStatus: string,
): string {
  switch (status) {
    case "notStarted":
    case "inProgress":
      return "running";
    case "interrupted":
      return "interrupted";
    default:
      return status ?? fallbackStatus;
  }
}

function readTurnTimestamp(
  turn: Record<string, unknown>,
  fallback: string,
  stringKeys: string[],
  millisKeys: string[],
): string {
  return (
    readString(turn, ...stringKeys) ??
    timestampFromMillis(readFiniteNumber(turn, ...millisKeys)) ??
    fallback
  );
}

function readOptionalTurnTimestamp(
  turn: Record<string, unknown>,
  stringKeys: string[],
  millisKeys: string[],
): string | undefined {
  return (
    readString(turn, ...stringKeys) ??
    timestampFromMillis(readFiniteNumber(turn, ...millisKeys))
  );
}

function timestampFromMillis(value: number | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? undefined
    : timestamp.toISOString();
}

export function readAgentThreadItemFromPayload(
  payload: Record<string, unknown>,
  event: AppServerAgentEvent,
  fallbackStatus: "in_progress" | "completed" | "failed",
): Record<string, unknown> {
  const item = normalizeRecord(payload.item) ?? payload;
  const canonicalToolItem = readCanonicalToolThreadItem(item, event);
  if (canonicalToolItem) {
    return canonicalToolItem;
  }
  const itemType = readString(item, "type") ?? "agent_message";
  const baseItem = readAgentThreadItemBase(item, event, fallbackStatus);

  if (itemType === "agent_message") {
    return {
      ...item,
      ...baseItem,
      type: "agent_message",
      text: readString(item, "text", "content", "message") ?? "",
      phase: readString(item, "phase"),
    };
  }

  return {
    ...item,
    ...baseItem,
    type: itemType,
  };
}

export function readUserMessageItemFromPayload(
  payload: Record<string, unknown>,
  event: AppServerAgentEvent,
): Record<string, unknown> {
  const input = normalizeRecord(payload.input);
  const contentRecord = normalizeRecord(payload.content);
  const content =
    readString(payload, "text", "message", "content") ??
    readString(input ?? {}, "text", "message", "content") ??
    readString(contentRecord ?? {}, "text", "message", "content") ??
    "";
  return {
    ...readAgentThreadItemBase(payload, event, "completed"),
    type: "user_message",
    content,
  };
}

function readAgentThreadItemBase(
  item: Record<string, unknown>,
  event: AppServerAgentEvent,
  fallbackStatus: "in_progress" | "completed" | "failed",
): Record<string, unknown> {
  return {
    id:
      readString(item, "id", "itemId", "item_id", "messageId", "message_id") ??
      event.eventId,
    thread_id:
      readString(item, "thread_id", "threadId") ??
      event.threadId ??
      event.sessionId,
    turn_id:
      readString(item, "turn_id", "turnId") ?? event.turnId ?? event.sessionId,
    sequence:
      typeof item.sequence === "number" ? item.sequence : event.sequence,
    status:
      readString(item, "status") ??
      (readString(item, "completed_at", "completedAt")
        ? "completed"
        : fallbackStatus),
    started_at: readString(item, "started_at", "startedAt") ?? event.timestamp,
    completed_at: readString(item, "completed_at", "completedAt"),
    updated_at: readString(item, "updated_at", "updatedAt") ?? event.timestamp,
  };
}

function readTimestampMs(value: unknown, fallback: string): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  const parsedFallback = Date.parse(fallback);
  return Number.isFinite(parsedFallback) ? parsedFallback : Date.now();
}
