import { describe, expect, it, vi } from "vitest";
import {
  revertThreadHistory,
  type ThreadRevertAppServerClient,
} from "./threadRevert";

describe("revertThreadHistory", () => {
  it("应通过 typed thread/revert 返回同一 Thread 的 metadata-only 结果", async () => {
    const revertThread = vi.fn().mockResolvedValue({
      result: {
        thread: { id: "thread-1", turns: [] },
        turnsBackwardsCursor: "turn-cursor",
        itemsBackwardsCursor: "item-cursor",
      },
    });

    await expect(
      revertThreadHistory(
        { threadId: " thread-1 ", beforeTurnId: " turn-2 " },
        { revertThread } as ThreadRevertAppServerClient,
      ),
    ).resolves.toEqual({
      threadId: "thread-1",
      turnsBackwardsCursor: "turn-cursor",
      itemsBackwardsCursor: "item-cursor",
    });
    expect(revertThread).toHaveBeenCalledWith({
      threadId: "thread-1",
      beforeTurnId: "turn-2",
    });
  });

  it("应拒绝空 identity、Thread 漂移和非 metadata-only 响应", async () => {
    const validClient = {
      revertThread: vi.fn(),
    } as ThreadRevertAppServerClient;
    await expect(
      revertThreadHistory(
        { threadId: " ", beforeTurnId: "turn-1" },
        validClient,
      ),
    ).rejects.toThrow("threadId is required");

    const mismatchedClient = {
      revertThread: vi.fn().mockResolvedValue({
        result: { thread: { id: "thread-other", turns: [] } },
      }),
    } as ThreadRevertAppServerClient;
    await expect(
      revertThreadHistory(
        { threadId: "thread-1", beforeTurnId: "turn-1" },
        mismatchedClient,
      ),
    ).rejects.toThrow("invalid thread");

    const hydratedClient = {
      revertThread: vi.fn().mockResolvedValue({
        result: { thread: { id: "thread-1", turns: [{ id: "turn-1" }] } },
      }),
    } as ThreadRevertAppServerClient;
    await expect(
      revertThreadHistory(
        { threadId: "thread-1", beforeTurnId: "turn-1" },
        hydratedClient,
      ),
    ).rejects.toThrow("metadata-only history");
  });
});
