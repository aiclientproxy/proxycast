import type {
  AgentUiProjectionContext,
  AgentUiProjectionEvent,
  AgentUiRuntimeStatus,
} from "@limecloud/agent-ui-contracts";

import {
  compactProjectionFields,
  definedString,
  readBooleanField,
  readRecord,
  readStringArray,
  readStringField,
} from "./normalization.js";

export type AgentUiThreadForkLineageIssueCode =
  | "missing_source_thread_id"
  | "missing_fork_thread_id"
  | "fork_reused_source_thread_id"
  | "missing_forked_from_id"
  | "forked_from_mismatch"
  | "read_model_lost_forked_from_id"
  | "copied_turn_suffix_leaked"
  | "parent_item_ids_missing"
  | "sidebar_lineage_missing"
  | "history_lineage_missing"
  | "source_rollout_mutated"
  | "fork_rendered_as_plain_thread"
  | "parent_thread_id_confused_with_forked_from_id";

export interface AgentUiThreadForkLineageIssue {
  code: AgentUiThreadForkLineageIssueCode;
  path: string;
  message: string;
}

export interface AgentUiThreadForkLineageProjectionInput {
  sourceThreadId?: string | null;
  forkThreadId?: string | null;
  forkRequest?: unknown;
  forkResponse?: unknown;
  readResponse?: unknown;
  sourceTurnOrder?: unknown;
  parentItemIds?: unknown;
  sidebarEntries?: unknown;
  historyEvents?: unknown;
  sourceRolloutUnchanged?: boolean | null;
  renderedPlainThread?: boolean | null;
  timestamp?: string | null;
}

export interface AgentUiForkedTurnSnapshot {
  id: string;
  itemIds: string[];
}

export interface AgentUiLineageSurfaceSnapshot {
  present: boolean;
  forkThreadId?: string;
  forkedFromId?: string;
  forkedAtTurnId?: string;
  parentItemIds: string[];
}

export interface AgentUiThreadForkLineageSnapshot {
  sourceThreadId?: string;
  forkThreadId?: string;
  sessionId?: string;
  forkedFromId?: string;
  readForkedFromId?: string;
  parentThreadId?: string;
  forkedAtTurnId?: string;
  copiedTurnIds: string[];
  parentItemIds: string[];
  sourceTurnOrder: string[];
  sidebarLineage: AgentUiLineageSurfaceSnapshot;
  historyLineage: AgentUiLineageSurfaceSnapshot;
  sourceRolloutUnchanged?: boolean;
  renderedPlainThread: boolean;
  lineageComplete: boolean;
  validationIssues: AgentUiThreadForkLineageIssue[];
}

function issue(
  code: AgentUiThreadForkLineageIssueCode,
  path: string,
  message: string,
): AgentUiThreadForkLineageIssue {
  return { code, path, message };
}

function recordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => readRecord(item))
    .filter((item): item is Record<string, unknown> => Boolean(item));
}

function idList(value: unknown): string[] {
  const strings = readStringArray(value);
  if (strings.length > 0) return strings;
  return recordArray(value)
    .map((record) => readStringField(record, ["id", "turnId", "turn_id", "itemId", "item_id"]))
    .filter((item): item is string => Boolean(item));
}

function threadRecord(value: unknown): Record<string, unknown> | undefined {
  const record = readRecord(value);
  return readRecord(record?.thread) ?? record;
}

function threadId(record: Record<string, unknown> | undefined): string | undefined {
  return readStringField(record, ["id", "threadId", "thread_id"]);
}

function forkedFromId(record: Record<string, unknown> | undefined): string | undefined {
  return readStringField(record, ["forkedFromId", "forked_from_id"]);
}

function parentThreadId(record: Record<string, unknown> | undefined): string | undefined {
  return readStringField(record, ["parentThreadId", "parent_thread_id"]);
}

function turnSnapshots(thread: Record<string, unknown> | undefined): AgentUiForkedTurnSnapshot[] {
  return recordArray(thread?.turns).flatMap((turn) => {
    const id = readStringField(turn, ["id", "turnId", "turn_id"]);
    if (!id) return [];
    const itemIds = recordArray(turn.items)
      .map((item) => readStringField(item, ["id", "itemId", "item_id"]))
      .filter((item): item is string => Boolean(item));
    return [{ id, itemIds }];
  });
}

function firstSurfaceLineage(
  records: readonly Record<string, unknown>[],
  forkThreadId: string | undefined,
): AgentUiLineageSurfaceSnapshot {
  const record =
    records.find((item) => {
      const id = readStringField(item, ["threadId", "thread_id", "id"]);
      return forkThreadId ? id === forkThreadId : Boolean(id);
    }) ?? records[0];
  if (!record) {
    return {
      present: false,
      parentItemIds: [],
    };
  }
  return compactProjectionFields({
    present: true,
    forkThreadId: readStringField(record, ["threadId", "thread_id", "id"]),
    forkedFromId: forkedFromId(record),
    forkedAtTurnId: readStringField(record, [
      "forkedAtTurnId",
      "forked_at_turn_id",
      "lastTurnId",
      "last_turn_id",
    ]),
    parentItemIds: idList(record.parentItemIds ?? record.parent_item_ids),
  } satisfies AgentUiLineageSurfaceSnapshot);
}

function sourceRolloutUnchanged(
  input: AgentUiThreadForkLineageProjectionInput,
): boolean | undefined {
  if (typeof input.sourceRolloutUnchanged === "boolean") {
    return input.sourceRolloutUnchanged;
  }
  const response = readRecord(input.forkResponse);
  return readBooleanField(response, [
    "sourceRolloutUnchanged",
    "source_rollout_unchanged",
  ]);
}

function requestedLastTurnId(input: AgentUiThreadForkLineageProjectionInput): string | undefined {
  const request = readRecord(input.forkRequest);
  return readStringField(request, ["lastTurnId", "last_turn_id"]);
}

function validateSnapshot(
  snapshot: Omit<AgentUiThreadForkLineageSnapshot, "validationIssues">,
): AgentUiThreadForkLineageIssue[] {
  const issues: AgentUiThreadForkLineageIssue[] = [];

  if (!snapshot.sourceThreadId) {
    issues.push(
      issue(
        "missing_source_thread_id",
        "$.forkRequest.threadId",
        "thread/fork must record the source thread id.",
      ),
    );
  }
  if (!snapshot.forkThreadId) {
    issues.push(
      issue(
        "missing_fork_thread_id",
        "$.forkResponse.thread.id",
        "thread/fork must return the new fork thread id.",
      ),
    );
  }
  if (
    snapshot.sourceThreadId &&
    snapshot.forkThreadId &&
    snapshot.sourceThreadId === snapshot.forkThreadId
  ) {
    issues.push(
      issue(
        "fork_reused_source_thread_id",
        "$.forkResponse.thread.id",
        "Forked threads must get a new id instead of reusing the source thread id.",
      ),
    );
  }
  if (!snapshot.forkedFromId) {
    issues.push(
      issue(
        "missing_forked_from_id",
        "$.forkResponse.thread.forkedFromId",
        "Forked threads must expose forkedFromId.",
      ),
    );
  } else if (
    snapshot.sourceThreadId &&
    snapshot.forkedFromId !== snapshot.sourceThreadId
  ) {
    issues.push(
      issue(
        "forked_from_mismatch",
        "$.forkResponse.thread.forkedFromId",
        "forkedFromId must point at the source thread.",
      ),
    );
  }
  if (
    snapshot.readForkedFromId &&
    snapshot.forkedFromId &&
    snapshot.readForkedFromId !== snapshot.forkedFromId
  ) {
    issues.push(
      issue(
        "read_model_lost_forked_from_id",
        "$.readResponse.thread.forkedFromId",
        "thread/read must return the same forkedFromId as thread/fork.",
      ),
    );
  }
  if (!snapshot.readForkedFromId && snapshot.forkedFromId) {
    issues.push(
      issue(
        "read_model_lost_forked_from_id",
        "$.readResponse.thread.forkedFromId",
        "thread/read must retain forkedFromId for forked threads.",
      ),
    );
  }
  if (snapshot.parentThreadId && !snapshot.forkedFromId) {
    issues.push(
      issue(
        "parent_thread_id_confused_with_forked_from_id",
        "$.forkResponse.thread.parentThreadId",
        "parentThreadId is for subagents and must not replace forkedFromId.",
      ),
    );
  }
  if (snapshot.forkedAtTurnId && snapshot.sourceTurnOrder.length > 0) {
    const forkIndex = snapshot.sourceTurnOrder.indexOf(snapshot.forkedAtTurnId);
    const copied = new Set(snapshot.copiedTurnIds);
    const leaked = snapshot.sourceTurnOrder
      .slice(forkIndex + 1)
      .some((turnId) => copied.has(turnId));
    if (forkIndex >= 0 && leaked) {
      issues.push(
        issue(
          "copied_turn_suffix_leaked",
          "$.forkResponse.thread.turns",
          "Forking at lastTurnId must drop later source turns.",
        ),
      );
    }
  }
  if (snapshot.parentItemIds.length === 0) {
    issues.push(
      issue(
        "parent_item_ids_missing",
        "$.forkResponse.thread.turns[].items",
        "Fork lineage must retain parent item ids for history replay.",
      ),
    );
  }
  for (const [surface, lineage] of [
    ["sidebar", snapshot.sidebarLineage],
    ["history", snapshot.historyLineage],
  ] as const) {
    if (!lineage.present || lineage.forkedFromId !== snapshot.sourceThreadId) {
      issues.push(
        issue(
          surface === "sidebar"
            ? "sidebar_lineage_missing"
            : "history_lineage_missing",
          `$.${surface}`,
          `${surface} surface must expose fork lineage for the forked thread.`,
        ),
      );
    }
  }
  if (snapshot.sourceRolloutUnchanged === false) {
    issues.push(
      issue(
        "source_rollout_mutated",
        "$.sourceRolloutUnchanged",
        "Forking must not mutate the source rollout.",
      ),
    );
  }
  if (snapshot.renderedPlainThread) {
    issues.push(
      issue(
        "fork_rendered_as_plain_thread",
        "$.renderedPlainThread",
        "Forked threads must not be rendered as plain threads without lineage.",
      ),
    );
  }
  return issues;
}

function runtimeStatus(
  issues: readonly AgentUiThreadForkLineageIssue[],
): AgentUiRuntimeStatus {
  return issues.length > 0 ? "failed" : "completed";
}

export function extractCodexThreadForkLineageSnapshot(
  input: AgentUiThreadForkLineageProjectionInput,
): AgentUiThreadForkLineageSnapshot {
  const request = readRecord(input.forkRequest);
  const forkThread = threadRecord(input.forkResponse);
  const readThread = threadRecord(input.readResponse);
  const sourceThreadId =
    definedString(input.sourceThreadId ?? undefined) ??
    readStringField(request, ["threadId", "thread_id"]);
  const forkThreadId =
    definedString(input.forkThreadId ?? undefined) ?? threadId(forkThread);
  const turns = turnSnapshots(forkThread);
  const parentItemIds = idList(input.parentItemIds);
  const copiedParentItemIds =
    parentItemIds.length > 0
      ? parentItemIds
      : turns.flatMap((turn) => turn.itemIds);
  const forkedAtTurnId =
    requestedLastTurnId(input) ?? turns[turns.length - 1]?.id;
  const base = {
    sourceThreadId,
    forkThreadId,
    sessionId: readStringField(forkThread, ["sessionId", "session_id"]),
    forkedFromId: forkedFromId(forkThread),
    readForkedFromId: forkedFromId(readThread),
    parentThreadId: parentThreadId(forkThread),
    forkedAtTurnId,
    copiedTurnIds: turns.map((turn) => turn.id),
    parentItemIds: copiedParentItemIds,
    sourceTurnOrder: idList(input.sourceTurnOrder),
    sidebarLineage: firstSurfaceLineage(recordArray(input.sidebarEntries), forkThreadId),
    historyLineage: firstSurfaceLineage(recordArray(input.historyEvents), forkThreadId),
    sourceRolloutUnchanged: sourceRolloutUnchanged(input),
    renderedPlainThread: Boolean(input.renderedPlainThread),
    lineageComplete: false,
  };
  const validationIssues = validateSnapshot(base);
  return {
    ...base,
    lineageComplete: validationIssues.length === 0,
    validationIssues,
  };
}

export function buildCodexThreadForkLineageProjectionEvent(
  input: AgentUiThreadForkLineageProjectionInput,
  context: AgentUiProjectionContext = {},
): AgentUiProjectionEvent {
  const snapshot = extractCodexThreadForkLineageSnapshot(input);
  const status = runtimeStatus(snapshot.validationIssues);
  return compactProjectionFields({
    type: "context.changed",
    sourceType: "thread_fork_lineage_projection",
    sequence: context.sequence,
    timestamp: definedString(input.timestamp ?? undefined) ?? context.timestamp,
    sessionId: snapshot.sessionId ?? definedString(context.sessionId ?? undefined),
    threadId: snapshot.forkThreadId ?? definedString(context.threadId ?? undefined),
    runId: definedString(context.runId ?? undefined),
    turnId: snapshot.forkedAtTurnId ?? definedString(context.turnId ?? undefined),
    owner: "context",
    scope: "thread",
    phase: status === "failed" ? "failed" : "completed",
    surface: "timeline_evidence",
    persistence: "snapshot",
    control: "open_detail",
    runtimeEntity: "agent_turn",
    runtimeStatus: status,
    latestTurnStatus: status,
    payload: {
      threadForkLineageEvent: "thread_fork_lineage_snapshot",
      sourceThreadId: snapshot.sourceThreadId,
      forkThreadId: snapshot.forkThreadId,
      forkedFromId: snapshot.forkedFromId,
      forkedAtTurnId: snapshot.forkedAtTurnId,
      parentItemIds: snapshot.parentItemIds,
      lineageComplete: snapshot.lineageComplete,
      threadForkLineage: snapshot,
      validationIssues: snapshot.validationIssues,
    },
  } satisfies AgentUiProjectionEvent);
}
