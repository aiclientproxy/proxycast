import type { AppServerAgentSessionOverview } from "./appServerSessionClient";
import type { AppServerAgentEvent } from "../appServer";
import type {
  AgentThreadItem,
  AgentThreadTurn,
  AgentThreadTurnStatus,
} from "../agentProtocol";
import type {
  AgentRuntimeProfileStatus,
  AgentSessionDetail,
} from "./sessionTypes";
import { readCanonicalThreadItem } from "./appServerCanonicalItemReader";
import {
  createConversationProjectionReducer,
  RENDER_PROJECTION_REFERENCE_REVISION,
} from "./conversationProjection";

export interface CanonicalThreadListProjectionOptions {
  archived?: boolean;
}

export function readCanonicalThreadListResponse(
  value: unknown,
  options: CanonicalThreadListProjectionOptions = {},
): AppServerAgentSessionOverview[] | null {
  if (!isRecord(value) || !Array.isArray(value.data)) {
    return null;
  }

  const sessions: AppServerAgentSessionOverview[] = [];
  for (const thread of value.data) {
    const normalized = readCanonicalThreadOverview(thread, options);
    if (!normalized) {
      return null;
    }
    sessions.push(normalized);
  }
  return sessions;
}

function readCanonicalThreadOverview(
  value: unknown,
  options: CanonicalThreadListProjectionOptions,
): AppServerAgentSessionOverview | null {
  const thread = readCanonicalThreadRecord(value);
  if (!thread) {
    return null;
  }

  const metadata = canonicalThreadMetadata(thread);
  const turns = Array.isArray(thread.turns) ? thread.turns : [];
  const createdAtMs = canonicalThreadTime(thread, "createdAt");
  const updatedAtMs = canonicalThreadTime(thread, "updatedAt");
  const threadId = canonicalThreadId(thread);
  const sessionId = readStringField(thread, "sessionId");
  if (
    !threadId ||
    !sessionId ||
    createdAtMs === undefined ||
    updatedAtMs === undefined
  ) {
    return null;
  }

  const archived = thread.archived === true || options.archived === true;
  const threadStatus = canonicalThreadStatus(thread.status);
  if (!threadStatus) {
    return null;
  }
  const model =
    readOptionalStringField(thread, "modelProvider") ||
    metadataString(metadata, "model", "modelName", "model_name") ||
    "";
  const title =
    readOptionalStringField(thread, "name") ||
    readOptionalStringField(thread, "preview") ||
    undefined;
  const section = readSection(thread);
  const sectionEnteredAt = readOptionalNumberField(
    thread,
    "sectionEnteredAt",
    "section_entered_at",
  );
  const latestTurn = turns.filter((turn) => !isQueuedTurn(turn)).at(-1);
  const activeTurn = turns.find((turn) => {
    if (!isRecord(turn)) {
      return false;
    }
    const queue = isRecord(turn.queue)
      ? readStringField(turn.queue, "state")
      : "";
    return (
      (readStringField(turn, "status") === "inProgress" &&
        queue !== "queued") ||
      queue === "running"
    );
  });
  const queuedTurnCount = turns.filter(isQueuedTurn).length;

  return omitUndefined({
    sessionId,
    threadId,
    parentThreadId: readOptionalStringField(thread, "parentThreadId"),
    canAcceptDirectInput: readOptionalBooleanField(
      thread,
      "canAcceptDirectInput",
      "can_accept_direct_input",
    ),
    title,
    businessObjectRefMetadata: metadata,
    model,
    createdAt: canonicalTimestamp(createdAtMs),
    updatedAt: canonicalTimestamp(updatedAtMs),
    archivedAt: archived ? canonicalTimestamp(updatedAtMs) : null,
    workspaceId: metadataString(metadata, "workspaceId", "workspace_id"),
    workingDir:
      metadataString(metadata, "workingDir", "working_dir") ??
      readOptionalStringField(thread, "cwd"),
    executionStrategy: metadataString(
      metadata,
      "executionStrategy",
      "execution_strategy",
    ),
    messagesCount: canonicalThreadMessageCount(turns, metadata),
    threadStatus,
    latestTurnStatus: canonicalTurnStatus(
      isRecord(latestTurn) ? latestTurn.status : undefined,
    ),
    activeTurnId: isRecord(activeTurn)
      ? canonicalTurnId(activeTurn)
      : undefined,
    queuedTurnCount,
    section,
    sectionEnteredAt:
      sectionEnteredAt === undefined ? undefined : sectionEnteredAt * 1_000,
  }) as AppServerAgentSessionOverview;
}

function readCanonicalThreadRecord(
  value: unknown,
): Record<string, unknown> | null {
  if (!isRecord(value)) {
    return null;
  }
  const sessionId = readStringField(value, "sessionId");
  const threadId = canonicalThreadId(value);
  const createdAtMs = canonicalThreadTime(value, "createdAt");
  const updatedAtMs = canonicalThreadTime(value, "updatedAt");
  if (
    !sessionId ||
    !threadId ||
    createdAtMs === undefined ||
    updatedAtMs === undefined
  ) {
    return null;
  }
  return value;
}

export function readCanonicalThreadDetail(
  value: unknown,
): AgentSessionDetail | null {
  if (!isRecord(value) || !isRecord(value.thread)) {
    return null;
  }
  const thread = readCanonicalThreadRecord(value.thread);
  if (!thread) {
    return null;
  }

  const sessionId = readStringField(thread, "sessionId");
  const threadId = canonicalThreadId(thread);
  const createdAtMs = canonicalThreadTime(thread, "createdAt");
  const updatedAtMs = canonicalThreadTime(thread, "updatedAt");
  if (
    !sessionId ||
    !threadId ||
    createdAtMs === undefined ||
    updatedAtMs === undefined
  ) {
    return null;
  }

  const metadata = canonicalThreadMetadata(thread);
  const section = readSection(thread);
  const sectionEnteredAt = readOptionalNumberField(
    thread,
    "sectionEnteredAt",
    "section_entered_at",
  );
  const rawTurns = Array.isArray(thread.turns) ? thread.turns : [];
  const turns: AgentThreadTurn[] = [];
  for (const turn of rawTurns) {
    if (isQueuedTurn(turn)) {
      continue;
    }
    const projected = canonicalTurnToRuntimeTurn(turn, threadId, updatedAtMs);
    if (!projected) {
      return null;
    }
    turns.push(projected);
  }
  const items = canonicalThreadItems(thread, rawTurns);
  if (!items) {
    return null;
  }
  const threadStatus = canonicalThreadStatus(thread.status);
  if (!threadStatus) {
    return null;
  }
  const profileStatus = canonicalProfileStatus(threadStatus);
  const articleWorkspace = metadataRecord(
    metadata,
    "articleWorkspace",
    "article_workspace",
  );
  const artifacts = metadataRecordArray(metadata, "artifacts");
  const todoItems = canonicalTurnPlanItems(rawTurns);

  return {
    id: sessionId,
    thread_id: threadId,
    name:
      readOptionalStringField(thread, "name") ||
      readOptionalStringField(thread, "preview") ||
      sessionId,
    created_at: createdAtMs,
    updated_at: updatedAtMs,
    model:
      readOptionalStringField(thread, "modelProvider") ||
      metadataString(metadata, "model", "modelName", "model_name"),
    workspace_id: metadataString(metadata, "workspaceId", "workspace_id"),
    working_dir:
      metadataString(metadata, "workingDir", "working_dir") ??
      readOptionalStringField(thread, "cwd"),
    execution_strategy: executionStrategyFromProtocol(
      metadataString(metadata, "executionStrategy", "execution_strategy"),
    ),
    messages_count: canonicalThreadMessageCount(rawTurns, metadata),
    messages: [],
    turns,
    items,
    thread_read: {
      thread_id: threadId,
      can_accept_direct_input: readOptionalBooleanField(
        thread,
        "canAcceptDirectInput",
        "can_accept_direct_input",
      ),
      status: threadStatus,
      profile_status: profileStatus,
      active_turn_id: turns.find((turn) => turn.status === "running")?.id,
      turns: turns.map((turn) => ({
        turn_id: turn.id,
        status: canonicalProfileStatus(turn.status),
        native_status: turn.status,
      })),
      session_business_object_ref_metadata: metadata,
      article_workspace: articleWorkspace,
      articleWorkspace,
      artifacts,
    },
    todo_items: todoItems,
    section,
    section_entered_at:
      sectionEnteredAt === undefined ? undefined : sectionEnteredAt * 1_000,
  };
}

function canonicalTurnPlanItems(turns: unknown[]): Array<{
  content: string;
  status: "pending" | "in_progress" | "completed";
}> {
  let latest: Array<{
    content: string;
    status: "pending" | "in_progress" | "completed";
  }> | null = null;
  for (const turn of turns) {
    if (!isRecord(turn) || !Array.isArray(turn.items)) {
      continue;
    }
    for (const item of turn.items) {
      const snapshot = canonicalTurnPlanSnapshot(item);
      if (snapshot !== null) {
        latest = snapshot;
      }
    }
  }
  return latest ?? [];
}

function canonicalTurnPlanSnapshot(value: unknown): Array<{
  content: string;
  status: "pending" | "in_progress" | "completed";
}> | null {
  if (
    !isRecord(value) ||
    readStringField(value, "type") !== "dynamicToolCall"
  ) {
    return null;
  }
  const tool = readStringField(value, "tool")
    .replace(/[\s_-]+/gu, "")
    .toLowerCase();
  if (
    (tool !== "updateplan" && tool !== "updateplantool") ||
    readStringField(value, "status") !== "completed" ||
    value.success === false
  ) {
    return null;
  }
  const plan = canonicalUpdatePlanArgument(readField(value, "arguments"));
  if (!plan) {
    return null;
  }
  const snapshot: Array<{
    content: string;
    status: "pending" | "in_progress" | "completed";
  }> = [];
  for (const candidate of plan) {
    if (!isRecord(candidate)) {
      return null;
    }
    const content = readStringField(candidate, "step").trim();
    const status = readStringField(candidate, "status");
    if (
      !content ||
      (status !== "pending" &&
        status !== "in_progress" &&
        status !== "completed")
    ) {
      return null;
    }
    snapshot.push({ content, status });
  }
  return snapshot;
}

function canonicalUpdatePlanArgument(value: unknown): unknown[] | null {
  if (isRecord(value)) {
    return Array.isArray(value.plan) ? value.plan : null;
  }
  if (!Array.isArray(value)) {
    return null;
  }

  const planArguments = value.filter(
    (candidate) =>
      isRecord(candidate) && readStringField(candidate, "name") === "plan",
  );
  if (planArguments.length !== 1) {
    return null;
  }
  const planValue = (planArguments[0] as Record<string, unknown>).value;
  if (Array.isArray(planValue)) {
    return planValue;
  }
  if (typeof planValue !== "string") {
    return null;
  }
  try {
    const parsed = JSON.parse(planValue) as unknown;
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function canonicalThreadItems(
  thread: Record<string, unknown>,
  turns: unknown[],
): AgentThreadItem[] | null {
  const sessionId = readStringField(thread, "sessionId");
  const threadId = canonicalThreadId(thread);
  if (!sessionId || !threadId) {
    return null;
  }
  const fallbackTimestampMs =
    canonicalThreadTime(thread, "updatedAt") ??
    canonicalThreadTime(thread, "createdAt") ??
    Date.now();
  const reducer = createConversationProjectionReducer({ threadId });
  for (const turn of turns) {
    if (isQueuedTurn(turn)) {
      continue;
    }
    if (!isRecord(turn) || !Array.isArray(turn.items)) {
      return null;
    }
    const turnId = canonicalTurnId(turn);
    if (!turnId) {
      return null;
    }
    const turnProjection = canonicalTurnToRuntimeTurn(
      turn,
      threadId,
      fallbackTimestampMs,
    );
    if (!turnProjection) {
      return null;
    }
    reducer.dispatch({
      type:
        turnProjection.status === "running" ? "turn_started" : "turn_completed",
      source: "read",
      event_id: `thread-read:turn:${turnId}`,
      protocol_method: "thread/read",
      protocol_revision: RENDER_PROJECTION_REFERENCE_REVISION,
      turn: turnProjection,
    });
    for (const [itemIndex, item] of turn.items.entries()) {
      if (!isRecord(item)) {
        return null;
      }
      const createdAtMs =
        canonicalThreadTime(item, "createdAt") ??
        canonicalThreadTime(turn, "startedAt") ??
        fallbackTimestampMs;
      const event: AppServerAgentEvent = {
        eventId: `thread-read:${canonicalItemId(item)}`,
        sequence: itemIndex,
        sessionId,
        threadId,
        turnId,
        type: canonicalReadItemEventType(readField(turn, "status")),
        timestamp: canonicalTimestamp(createdAtMs),
        payload: { item },
      };
      const projected = readCanonicalThreadItem(item, event);
      if (!projected) {
        return null;
      }
      const typedItem = projected as unknown as AgentThreadItem;
      reducer.dispatch({
        type:
          typedItem.status === "in_progress"
            ? "item_started"
            : "item_completed",
        source: "read",
        event_id: event.eventId,
        protocol_method: "thread/read",
        protocol_revision: RENDER_PROJECTION_REFERENCE_REVISION,
        item: typedItem,
      });
    }
  }
  return [...reducer.getProjection().items];
}

function isQueuedTurn(value: unknown): boolean {
  return (
    isRecord(value) &&
    isRecord(value.queue) &&
    readStringField(value.queue, "state") === "queued"
  );
}

function userInputText(value: unknown): string {
  if (!Array.isArray(value)) {
    return typeof value === "string" ? value : "";
  }
  return value
    .map((part) => {
      if (!isRecord(part)) {
        return "";
      }
      return readOptionalStringField(part, "text") ?? "";
    })
    .join("");
}

function canonicalTurnPromptText(value: Record<string, unknown>): string {
  if (Array.isArray(value.items)) {
    const userItem = value.items.find(
      (item) =>
        isRecord(item) && readStringField(item, "type") === "userMessage",
    );
    if (isRecord(userItem)) {
      const prompt = userInputText(userItem.content);
      if (prompt) {
        return prompt;
      }
    }
  }
  return (
    readOptionalStringField(value, "prompt") ||
    readOptionalStringField(value, "message") ||
    ""
  );
}

function canonicalTurnToRuntimeTurn(
  value: unknown,
  threadId: string,
  fallbackTimestampMs: number,
): AgentThreadTurn | null {
  if (!isRecord(value)) {
    return null;
  }
  const turnId = canonicalTurnId(value);
  const status = canonicalTurnStatus(readField(value, "status"));
  if (!turnId || !status) {
    return null;
  }
  const createdAtMs =
    canonicalThreadTime(value, "startedAt") ??
    canonicalThreadTime(value, "createdAt") ??
    fallbackTimestampMs;
  const completedAtMs = canonicalThreadTime(value, "completedAt");
  const startedAt = canonicalTimestamp(createdAtMs);
  const errorMessage = isRecord(value.error)
    ? readOptionalStringField(value.error, "message")
    : undefined;
  return {
    id: turnId,
    thread_id: readOptionalStringField(value, "threadId") ?? threadId,
    prompt_text: canonicalTurnPromptText(value),
    status,
    started_at: startedAt,
    completed_at:
      completedAtMs === undefined
        ? undefined
        : canonicalTimestamp(completedAtMs),
    error_message: errorMessage,
    created_at: startedAt,
    updated_at: canonicalTimestamp(
      canonicalThreadTime(value, "updatedAt") ?? completedAtMs ?? createdAtMs,
    ),
  };
}

function canonicalThreadMessageCount(
  turns: unknown[],
  metadata: Record<string, unknown> | undefined,
): number {
  const metadataCount =
    readNumberField(metadata ?? {}, "messagesCount", "messages_count") ??
    readNumberField(metadata ?? {}, "messageCount", "message_count");
  if (metadataCount !== undefined) {
    return metadataCount;
  }
  return turns.reduce<number>((count, turn) => {
    if (!isRecord(turn) || !Array.isArray(turn.items)) {
      return count;
    }
    return (
      count +
      turn.items.filter(
        (item) =>
          (isRecord(item) &&
            isRecord(item.payload) &&
            ["userMessage", "agentMessage"].includes(
              readStringField(item.payload, "type"),
            )) ||
          ["userMessage", "agentMessage"].includes(
            readStringField(item, "type"),
          ),
      ).length
    );
  }, 0);
}

function canonicalThreadStatus(value: unknown): string | undefined {
  const type = isRecord(value) ? readStringField(value, "type") : value;
  switch (type) {
    case "active": {
      const flags =
        isRecord(value) && Array.isArray(value.activeFlags)
          ? value.activeFlags
          : [];
      return flags.some(
        (flag) => flag === "waitingOnApproval" || flag === "waitingOnUserInput",
      )
        ? "waitingAction"
        : "running";
    }
    case "systemError":
      return "failed";
    case "idle":
      return "idle";
    case "notLoaded":
      return "unknown";
    default:
      return undefined;
  }
}

function canonicalTurnStatus(
  value: unknown,
): AgentThreadTurnStatus | undefined {
  switch (value) {
    case "inProgress":
      return "running";
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "interrupted":
      return "interrupted";
    default:
      return undefined;
  }
}

function canonicalReadItemEventType(turnStatus: unknown): string {
  return turnStatus === "inProgress" ? "item.started" : "item.completed";
}

function canonicalProfileStatus(value: string): AgentRuntimeProfileStatus {
  switch (value) {
    case "waitingAction":
      return "blocked";
    case "interrupted":
      return "cancelled";
    case "idle":
    case "queued":
    case "running":
    case "blocked":
    case "completed":
    case "failed":
    case "cancelled":
    case "stale":
    case "unknown":
      return value;
    default:
      return "unknown";
  }
}

function canonicalThreadId(value: Record<string, unknown>): string {
  return readStringField(value, "id");
}

function canonicalTurnId(value: Record<string, unknown>): string {
  return readStringField(value, "id");
}

function canonicalItemId(value: Record<string, unknown>): string {
  return readStringField(value, "id");
}

function canonicalThreadTime(
  value: Record<string, unknown>,
  field: string,
): number | undefined {
  const seconds = readNumberField(value, field);
  return seconds === undefined ? undefined : seconds * 1_000;
}

function canonicalThreadMetadata(
  thread: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const metadata = readOptionalObjectField(thread, "metadata");
  if (metadata) {
    return metadata;
  }
  const extra = readOptionalObjectField(thread, "extra");
  return extra
    ? (readOptionalObjectField(extra, "metadata") ?? extra)
    : undefined;
}

/**
 * Imported threads are restored from another client and need the complete
 * canonical history before the renderer builds its timeline.
 */
export function isImportedCanonicalThread(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }

  const metadata = canonicalThreadMetadata(value);
  return hasImportedMetadata(metadata) || hasImportedMetadata(value);
}

function hasImportedMetadata(
  value: Record<string, unknown> | undefined,
): boolean {
  if (!value) {
    return false;
  }

  for (const key of [
    "imported",
    "isImported",
    "importedContinuation",
    "imported_continuation",
    "importedThreadSettings",
    "imported_thread_settings",
    "sourceClient",
    "source_client",
    "sourceThreadId",
    "source_thread_id",
    "sourceProvenance",
    "source_provenance",
    "codexImportFidelity",
    "codex_import_fidelity",
    "conversationImport",
    "conversation_import",
  ]) {
    const marker = value[key];
    if (marker === true) {
      return true;
    }
    if (typeof marker === "string" && marker.trim()) {
      return true;
    }
    if (isRecord(marker) && hasImportedMetadata(marker)) {
      return true;
    }
  }

  return false;
}

function metadataString(
  metadata: Record<string, unknown> | undefined,
  ...keys: string[]
): string | undefined {
  if (!metadata) {
    return undefined;
  }
  for (const key of keys) {
    const value = metadata[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
}

function metadataRecord(
  metadata: Record<string, unknown> | undefined,
  ...keys: string[]
): Record<string, unknown> | undefined {
  if (!metadata) {
    return undefined;
  }
  for (const key of keys) {
    if (isRecord(metadata[key])) {
      return metadata[key];
    }
  }
  return undefined;
}

function metadataRecordArray(
  metadata: Record<string, unknown> | undefined,
  key: string,
): Record<string, unknown>[] | undefined {
  const value = metadata?.[key];
  if (!Array.isArray(value) || !value.every(isRecord)) {
    return undefined;
  }
  return value;
}

function canonicalTimestamp(value: number): string {
  const timestamp = new Date(value);
  return Number.isNaN(timestamp.getTime())
    ? new Date(0).toISOString()
    : timestamp.toISOString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function readField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): unknown {
  if (Object.prototype.hasOwnProperty.call(record, camelKey)) {
    return record[camelKey];
  }
  return snakeKey ? record[snakeKey] : undefined;
}

function readStringField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): string {
  const value = readField(record, camelKey, snakeKey);
  return typeof value === "string" ? value : "";
}

function readOptionalStringField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): string | undefined {
  const value = readField(record, camelKey, snakeKey);
  return typeof value === "string" ? value : undefined;
}

function readOptionalNumberField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): number | undefined {
  const value = readField(record, camelKey, snakeKey);
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function readSection(
  thread: Record<string, unknown>,
): { id: string; name: string } | undefined {
  const value = readField(thread, "section");
  if (!isRecord(value)) {
    return undefined;
  }
  const id = readStringField(value, "id").trim();
  const name = readStringField(value, "name").trim();
  return id && name ? { id, name } : undefined;
}

function readNumberField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): number | undefined {
  const value = readField(record, camelKey, snakeKey);
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function readOptionalBooleanField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): boolean | undefined {
  const value = readField(record, camelKey, snakeKey);
  return typeof value === "boolean" ? value : undefined;
}

function readOptionalObjectField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): Record<string, unknown> | undefined {
  const value = readField(record, camelKey, snakeKey);
  return isRecord(value) ? value : undefined;
}

function omitUndefined<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  ) as T;
}

function executionStrategyFromProtocol(value: unknown): "react" | undefined {
  return value === "react" ? "react" : undefined;
}
