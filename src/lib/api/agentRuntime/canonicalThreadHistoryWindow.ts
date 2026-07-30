import { AppServerClient } from "@/lib/api/appServer";
import {
  METHOD_THREAD_ITEMS_LIST,
  METHOD_THREAD_TURNS_LIST,
  type ThreadItemsListResponse,
  type ThreadTurnsListResponse,
} from "../../../../packages/app-server-client/src/protocol";
import {
  AGENT_RUNTIME_DEFAULT_HISTORY_LIMIT,
  type AgentRuntimeGetSessionOptions,
} from "./requestTypes";

const HISTORY_PAGE_LIMIT = 100;

type CanonicalThreadHistoryClient = Pick<AppServerClient, "request">;

type CanonicalItemEntry = {
  item: Record<string, unknown>;
  turnId: string;
};

type CanonicalItemPage = {
  entriesDescending: CanonicalItemEntry[];
  nextCursor: string | null;
};

type CanonicalTurnPage = {
  nextCursor: string | null;
  turnsDescending: Record<string, unknown>[];
};

export interface CanonicalThreadHistoryWindow {
  historyCursor: {
    item_cursor: string | null;
    turn_cursor: string | null;
    loaded_entry_count: number;
    loaded_turn_count: number;
    loaded_item_count: number;
    has_more: boolean;
  };
  historyLimit: number;
  historyTruncated: boolean;
  thread: Record<string, unknown>;
}

export async function readCanonicalThreadHistoryWindow(
  client: CanonicalThreadHistoryClient,
  thread: Record<string, unknown>,
  options?: AgentRuntimeGetSessionOptions,
): Promise<CanonicalThreadHistoryWindow> {
  const threadId = readStringField(thread, "id");
  if (!threadId) {
    throw new Error("thread/read returned an empty canonical thread id");
  }

  const historyLimit = normalizeHistoryLimit(options?.historyLimit);
  const embeddedTurns = readEmbeddedTurns(thread);
  if (embeddedTurns) {
    const loadedItemCount = embeddedTurns.reduce(
      (count, turn) => count + (Array.isArray(turn.items) ? turn.items.length : 0),
      0,
    );
    return buildHistoryWindow({
      historyLimit,
      itemCursor: null,
      thread,
      turnCursor: null,
      turns: embeddedTurns,
      loadedItemCount,
    });
  }

  const shouldReadItems = shouldReadOwnerPage(options, "historyItemCursor");
  const shouldReadTurns = shouldReadOwnerPage(options, "historyTurnCursor");
  const itemPage = shouldReadItems
    ? await readItemPage(
        client,
        threadId,
        normalizeOpaqueCursor(options?.historyItemCursor),
        historyLimit,
      )
    : { entriesDescending: [], nextCursor: null };
  const turnPage = shouldReadTurns
    ? await readTurnPage(
        client,
        threadId,
        normalizeOpaqueCursor(options?.historyTurnCursor),
        historyLimit,
      )
    : { turnsDescending: [], nextCursor: null };
  const turns = attachSelectedItems(
    turnPage.turnsDescending,
    itemPage.entriesDescending,
  );

  return buildHistoryWindow({
    historyLimit,
    itemCursor: itemPage.nextCursor,
    thread,
    turnCursor: turnPage.nextCursor,
    turns,
    loadedItemCount: itemPage.entriesDescending.length,
  });
}

function buildHistoryWindow(params: {
  historyLimit: number;
  itemCursor: string | null;
  loadedItemCount: number;
  thread: Record<string, unknown>;
  turnCursor: string | null;
  turns: Record<string, unknown>[];
}): CanonicalThreadHistoryWindow {
  const loadedTurnCount = params.turns.length;
  const hasMore = params.itemCursor !== null || params.turnCursor !== null;
  return {
    historyCursor: {
      item_cursor: params.itemCursor,
      turn_cursor: params.turnCursor,
      loaded_entry_count: loadedTurnCount + params.loadedItemCount,
      loaded_turn_count: loadedTurnCount,
      loaded_item_count: params.loadedItemCount,
      has_more: hasMore,
    },
    historyLimit: params.historyLimit,
    historyTruncated: hasMore,
    thread: {
      ...params.thread,
      turns: params.turns,
    },
  };
}

function normalizeHistoryLimit(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return AGENT_RUNTIME_DEFAULT_HISTORY_LIMIT;
  }
  return Math.trunc(value);
}

function readEmbeddedTurns(
  thread: Record<string, unknown>,
): Record<string, unknown>[] | null {
  if (readStringField(thread, "historyMode") === "paginated") {
    return null;
  }
  if (!Array.isArray(thread.turns) || thread.turns.length === 0) {
    return null;
  }
  return thread.turns.every(
    (turn) =>
      isRecord(turn) && (Array.isArray(turn.items) || isActiveTurn(turn)),
  )
    ? (thread.turns as Record<string, unknown>[])
    : null;
}

function shouldReadOwnerPage(
  options: AgentRuntimeGetSessionOptions | undefined,
  key: "historyItemCursor" | "historyTurnCursor",
): boolean {
  return !options || !(key in options) || normalizeOpaqueCursor(options[key]) !== null;
}

async function readItemPage(
  client: CanonicalThreadHistoryClient,
  threadId: string,
  initialCursor: string | null,
  historyLimit: number,
): Promise<CanonicalItemPage> {
  const entriesDescending: CanonicalItemEntry[] = [];
  const seenCursors = new Set<string>();
  let cursor = initialCursor;

  do {
    const response = await client.request<ThreadItemsListResponse>(
      METHOD_THREAD_ITEMS_LIST,
      omitUndefined({
        threadId,
        cursor: cursor ?? undefined,
        limit: historyLimit === 0 ? HISTORY_PAGE_LIMIT : historyLimit,
        sortDirection: "desc",
      }),
    );
    if (!isRecord(response.result) || !Array.isArray(response.result.data)) {
      throw new Error("thread/items/list did not return item page");
    }
    for (const value of response.result.data) {
      if (!isRecord(value) || !isRecord(value.item)) {
        throw new Error("thread/items/list returned an invalid item entry");
      }
      const turnId = readStringField(value, "turnId");
      if (!turnId) {
        throw new Error("thread/items/list returned an item without turnId");
      }
      entriesDescending.push({ item: value.item, turnId });
    }

    const nextCursor = readNextCursor(response.result);
    assertCursorAdvanced(cursor, nextCursor, "thread/items/list");
    if (historyLimit > 0 || nextCursor === null) {
      return { entriesDescending, nextCursor };
    }
    if (seenCursors.has(nextCursor)) {
      throw new Error("thread/items/list returned a repeated cursor");
    }
    seenCursors.add(nextCursor);
    cursor = nextCursor;
  } while (cursor !== null);

  return { entriesDescending, nextCursor: null };
}

async function readTurnPage(
  client: CanonicalThreadHistoryClient,
  threadId: string,
  initialCursor: string | null,
  historyLimit: number,
): Promise<CanonicalTurnPage> {
  const turnsDescending: Record<string, unknown>[] = [];
  const seenCursors = new Set<string>();
  let cursor = initialCursor;

  do {
    const response = await client.request<ThreadTurnsListResponse>(
      METHOD_THREAD_TURNS_LIST,
      omitUndefined({
        threadId,
        cursor: cursor ?? undefined,
        limit: historyLimit === 0 ? HISTORY_PAGE_LIMIT : historyLimit,
        sortDirection: "desc",
        itemsView: "summary",
      }),
    );
    if (!isRecord(response.result) || !Array.isArray(response.result.data)) {
      throw new Error("thread/turns/list did not return turn page");
    }
    for (const value of response.result.data) {
      if (!isRecord(value)) {
        throw new Error("thread/turns/list returned an invalid turn");
      }
      const turnId = readStringField(value, "id");
      if (!turnId) {
        throw new Error("thread/turns/list returned a turn without id");
      }
      turnsDescending.push(value);
    }

    const nextCursor = readNextCursor(response.result);
    assertCursorAdvanced(cursor, nextCursor, "thread/turns/list");
    const shouldContinue = nextCursor !== null && historyLimit === 0;
    if (!shouldContinue) {
      return { turnsDescending, nextCursor };
    }
    if (seenCursors.has(nextCursor)) {
      throw new Error("thread/turns/list returned a repeated cursor");
    }
    seenCursors.add(nextCursor);
    cursor = nextCursor;
  } while (cursor !== null);

  return { turnsDescending, nextCursor: null };
}

function attachSelectedItems(
  turnsDescending: Record<string, unknown>[],
  entriesDescending: CanonicalItemEntry[],
): Record<string, unknown>[] {
  const itemsByTurnId = groupItemsByTurnId(entriesDescending);
  const attachedTurnIds = new Set<string>();
  const selectedTurnsDescending: Record<string, unknown>[] =
    turnsDescending.flatMap((turn) => {
      const turnId = readStringField(turn, "id");
      const selectedItems = itemsByTurnId.get(turnId);
      if (!selectedItems && !shouldRetainTurnWithoutItems(turn)) {
        return [];
      }
      attachedTurnIds.add(turnId);
      return [{ ...turn, items: selectedItems ?? [] }];
    });

  for (const entry of entriesDescending) {
    if (attachedTurnIds.has(entry.turnId)) {
      continue;
    }
    attachedTurnIds.add(entry.turnId);
    selectedTurnsDescending.push({
      id: entry.turnId,
      status: "completed",
      items: itemsByTurnId.get(entry.turnId) ?? [],
    });
  }

  return selectedTurnsDescending.reverse();
}

function groupItemsByTurnId(
  entriesDescending: CanonicalItemEntry[],
): Map<string, Record<string, unknown>[]> {
  const itemsByTurnId = new Map<string, Record<string, unknown>[]>();
  for (const entry of entriesDescending) {
    const items = itemsByTurnId.get(entry.turnId) ?? [];
    items.unshift(entry.item);
    itemsByTurnId.set(entry.turnId, items);
  }
  return itemsByTurnId;
}

function isActiveTurn(turn: Record<string, unknown>): boolean {
  const queueState = isRecord(turn.queue)
    ? readStringField(turn.queue, "state")
    : "";
  return (
    readStringField(turn, "status") === "inProgress" ||
    queueState === "queued" ||
    queueState === "running"
  );
}

function shouldRetainTurnWithoutItems(turn: Record<string, unknown>): boolean {
  if (isActiveTurn(turn)) {
    return true;
  }
  const status = readStringField(turn, "status");
  return Boolean(status) && status !== "completed";
}

function normalizeOpaqueCursor(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function readNextCursor(value: Record<string, unknown>): string | null {
  return normalizeOpaqueCursor(value.nextCursor);
}

function assertCursorAdvanced(
  currentCursor: string | null,
  nextCursor: string | null,
  method: string,
): void {
  if (currentCursor !== null && nextCursor === currentCursor) {
    throw new Error(`${method} did not advance its cursor`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function readStringField(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return typeof value === "string" ? value : "";
}

function omitUndefined<T extends Record<string, unknown>>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  ) as T;
}
