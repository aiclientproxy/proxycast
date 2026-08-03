import { describe, expect, it } from "vitest";
import {
  buildSessionHistoryPageRequestPlan,
  buildSessionHistoryPageResultPlan,
  normalizeNonNegativeInteger,
  normalizeOpaqueCursor,
  resolveSessionHistoryWindowFromDetail,
} from "./sessionHistoryPaginationController";

describe("sessionHistoryPaginationController", () => {
  it("应保持 opaque cursor 原值并归一化计数边界", () => {
    expect(normalizeOpaqueCursor(" cursor with spaces ")).toBe(
      " cursor with spaces ",
    );
    expect(normalizeOpaqueCursor("")).toBeNull();
    expect(normalizeNonNegativeInteger(1.8)).toBe(1);
    expect(normalizeNonNegativeInteger(-1)).toBeNull();
  });

  it("应只根据 Item/Turn cursor 构造 history window", () => {
    expect(
      resolveSessionHistoryWindowFromDetail({
        items: [{ id: "item-1" }, { id: "item-2" }],
        turns: [{ id: "turn-1" }],
        history_cursor: {
          item_cursor: "opaque-item-page-2",
          turn_cursor: null,
          loaded_entry_count: 3,
          loaded_turn_count: 1,
          loaded_item_count: 2,
          has_more: true,
        },
      }),
    ).toEqual({
      loadedEntries: 3,
      loadedTurns: 1,
      loadedItems: 2,
      hasMore: true,
      itemCursor: "opaque-item-page-2",
      turnCursor: null,
      isLoadingFull: false,
      error: null,
    });
  });

  it("两个 owner 都到 EOF 时不应保留 history window", () => {
    expect(
      resolveSessionHistoryWindowFromDetail({
        items: [{ id: "item-1" }],
        turns: [{ id: "turn-1" }],
        history_cursor: {
          item_cursor: null,
          turn_cursor: null,
          loaded_entry_count: 2,
          loaded_turn_count: 1,
          loaded_item_count: 1,
          has_more: true,
        },
      }),
    ).toBeNull();
  });

  it("historyLimit 为 0 时应标记为全量历史，即使没有后续 cursor", () => {
    expect(
      resolveSessionHistoryWindowFromDetail({
        history_limit: 0,
        items: [{ id: "item-1" }],
        turns: [{ id: "turn-1" }],
        history_cursor: {
          item_cursor: null,
          turn_cursor: null,
          loaded_entry_count: 2,
          loaded_turn_count: 1,
          loaded_item_count: 1,
          has_more: false,
        },
      }),
    ).toEqual({
      loadedEntries: 2,
      loadedTurns: 1,
      loadedItems: 1,
      hasMore: false,
      itemCursor: null,
      turnCursor: null,
      isLoadingFull: false,
      error: null,
      isFullyLoaded: true,
    });
  });

  it("应使用两个 owner cursor 构造下一页请求", () => {
    const currentHistoryWindow = {
      loadedEntries: 60,
      loadedTurns: 20,
      loadedItems: 40,
      hasMore: true,
      itemCursor: "opaque-item-page-2",
      turnCursor: null,
      isLoadingFull: false,
      error: "old",
    };

    expect(
      buildSessionHistoryPageRequestPlan({
        currentHistoryWindow,
        pageSize: 50,
      }),
    ).toEqual({
      itemCursor: "opaque-item-page-2",
      turnCursor: null,
      loadingWindow: {
        ...currentHistoryWindow,
        isLoadingFull: true,
        error: null,
      },
      nextHistoryLimit: 50,
      requestOptions: {
        historyItemCursor: "opaque-item-page-2",
        historyTurnCursor: null,
        historyLimit: 50,
      },
    });
  });

  it("加载中或没有可继续的 owner cursor 时不应重复请求", () => {
    expect(
      buildSessionHistoryPageRequestPlan({
        currentHistoryWindow: {
          loadedEntries: 60,
          loadedTurns: 20,
          loadedItems: 40,
          hasMore: true,
          itemCursor: "opaque-item-page-2",
          turnCursor: null,
          isLoadingFull: true,
          error: null,
        },
        pageSize: 50,
      }),
    ).toBeNull();
    expect(
      buildSessionHistoryPageRequestPlan({
        currentHistoryWindow: {
          loadedEntries: 60,
          loadedTurns: 20,
          loadedItems: 40,
          hasMore: true,
          itemCursor: null,
          turnCursor: null,
          isLoadingFull: false,
          error: null,
        },
        pageSize: 50,
      }),
    ).toBeNull();
  });

  it("应累计 Turn/Item entry count 并仅按下一 cursor 决定 hasMore", () => {
    const currentHistoryWindow = {
      loadedEntries: 60,
      loadedTurns: 20,
      loadedItems: 40,
      hasMore: true,
      itemCursor: "opaque-item-page-2",
      turnCursor: "opaque-turn-page-2",
      isLoadingFull: true,
      error: null,
    };

    expect(
      buildSessionHistoryPageResultPlan({
        currentHistoryWindow,
        detail: {
          history_cursor: {
            item_cursor: "opaque-item-page-3",
            turn_cursor: null,
            loaded_entry_count: 75,
            loaded_turn_count: 25,
            loaded_item_count: 50,
            has_more: true,
          },
        },
      }),
    ).toEqual({
      detailLoadedEntries: 75,
      detailLoadedTurns: 25,
      detailLoadedItems: 50,
      nextLoadedEntries: 135,
      nextLoadedTurns: 45,
      nextLoadedItems: 90,
      nextHistoryWindow: {
        loadedEntries: 135,
        loadedTurns: 45,
        loadedItems: 90,
        hasMore: true,
        itemCursor: "opaque-item-page-3",
        turnCursor: null,
        isLoadingFull: false,
        error: null,
      },
    });

    expect(
      buildSessionHistoryPageResultPlan({
        currentHistoryWindow,
        detail: {
          items: [{ id: "item-3" }],
          turns: [{ id: "turn-3" }],
          history_cursor: {
            item_cursor: null,
            turn_cursor: null,
            has_more: false,
          },
        },
      }).nextHistoryWindow,
    ).toBeNull();
  });
});
