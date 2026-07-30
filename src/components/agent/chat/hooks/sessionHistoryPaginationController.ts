export interface SessionHistoryWindowState {
  loadedEntries: number;
  loadedTurns: number;
  loadedItems: number;
  hasMore: boolean;
  itemCursor: string | null;
  turnCursor: string | null;
  isLoadingFull: boolean;
  error: string | null;
}

export interface SessionHistoryDetailLike {
  history_cursor?: {
    item_cursor?: string | null;
    turn_cursor?: string | null;
    loaded_entry_count?: number | null;
    loaded_turn_count?: number | null;
    loaded_item_count?: number | null;
    has_more?: boolean | null;
  } | null;
  items?: readonly unknown[];
  turns?: readonly unknown[];
}

export interface SessionHistoryPageRequestPlan {
  itemCursor: string | null;
  turnCursor: string | null;
  loadingWindow: SessionHistoryWindowState;
  nextHistoryLimit: number;
  requestOptions: {
    historyItemCursor: string | null;
    historyTurnCursor: string | null;
    historyLimit: number;
  };
}

export interface SessionHistoryPageResultPlan {
  detailLoadedEntries: number;
  detailLoadedTurns: number;
  detailLoadedItems: number;
  nextHistoryWindow: SessionHistoryWindowState | null;
  nextLoadedEntries: number;
  nextLoadedTurns: number;
  nextLoadedItems: number;
}

export function normalizeOpaqueCursor(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

export function normalizeNonNegativeInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.trunc(value)
    : null;
}

function resolveDetailLoadedCounts(detail: SessionHistoryDetailLike): {
  entries: number;
  turns: number;
  items: number;
} {
  const turns =
    normalizeNonNegativeInteger(detail.history_cursor?.loaded_turn_count) ??
    detail.turns?.length ??
    0;
  const items =
    normalizeNonNegativeInteger(detail.history_cursor?.loaded_item_count) ??
    detail.items?.length ??
    0;
  const entries =
    normalizeNonNegativeInteger(detail.history_cursor?.loaded_entry_count) ??
    turns + items;
  return { entries, turns, items };
}

export function resolveSessionHistoryWindowFromDetail(
  detail: SessionHistoryDetailLike,
): SessionHistoryWindowState | null {
  const itemCursor = normalizeOpaqueCursor(detail.history_cursor?.item_cursor);
  const turnCursor = normalizeOpaqueCursor(detail.history_cursor?.turn_cursor);
  const hasMore = itemCursor !== null || turnCursor !== null;
  if (!hasMore) {
    return null;
  }

  const loaded = resolveDetailLoadedCounts(detail);
  return {
    loadedEntries: loaded.entries,
    loadedTurns: loaded.turns,
    loadedItems: loaded.items,
    hasMore: true,
    itemCursor,
    turnCursor,
    isLoadingFull: false,
    error: null,
  };
}

export function buildSessionHistoryPageRequestPlan(params: {
  currentHistoryWindow?: SessionHistoryWindowState | null;
  pageSize: number;
}): SessionHistoryPageRequestPlan | null {
  const current = params.currentHistoryWindow;
  if (!current || current.isLoadingFull || !current.hasMore) {
    return null;
  }

  const itemCursor = normalizeOpaqueCursor(current.itemCursor);
  const turnCursor = normalizeOpaqueCursor(current.turnCursor);
  if (itemCursor === null && turnCursor === null) {
    return null;
  }
  const nextHistoryLimit = normalizeNonNegativeInteger(params.pageSize) ?? 0;
  if (nextHistoryLimit <= 0) {
    return null;
  }

  return {
    itemCursor,
    turnCursor,
    loadingWindow: { ...current, isLoadingFull: true, error: null },
    nextHistoryLimit,
    requestOptions: {
      historyItemCursor: itemCursor,
      historyTurnCursor: turnCursor,
      historyLimit: nextHistoryLimit,
    },
  };
}

export function buildSessionHistoryPageResultPlan(params: {
  currentHistoryWindow: SessionHistoryWindowState;
  detail: SessionHistoryDetailLike;
}): SessionHistoryPageResultPlan {
  const detailLoaded = resolveDetailLoadedCounts(params.detail);
  const itemCursor = normalizeOpaqueCursor(
    params.detail.history_cursor?.item_cursor,
  );
  const turnCursor = normalizeOpaqueCursor(
    params.detail.history_cursor?.turn_cursor,
  );
  const hasMore = itemCursor !== null || turnCursor !== null;
  const nextLoadedEntries =
    params.currentHistoryWindow.loadedEntries + detailLoaded.entries;
  const nextLoadedTurns =
    params.currentHistoryWindow.loadedTurns + detailLoaded.turns;
  const nextLoadedItems =
    params.currentHistoryWindow.loadedItems + detailLoaded.items;

  return {
    detailLoadedEntries: detailLoaded.entries,
    detailLoadedTurns: detailLoaded.turns,
    detailLoadedItems: detailLoaded.items,
    nextLoadedEntries,
    nextLoadedTurns,
    nextLoadedItems,
    nextHistoryWindow: hasMore
      ? {
          loadedEntries: nextLoadedEntries,
          loadedTurns: nextLoadedTurns,
          loadedItems: nextLoadedItems,
          hasMore: true,
          itemCursor,
          turnCursor,
          isLoadingFull: false,
          error: null,
        }
      : null,
  };
}
