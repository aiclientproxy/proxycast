import { describe, expect, it, vi } from "vitest";
import { readPromptHistory, type PromptHistoryClient } from "./promptHistory";

describe("promptHistory", () => {
  it("reads bounded newest-first pages with a stable log identity", async () => {
    const readPromptHistoryMock = vi
      .fn()
      .mockResolvedValueOnce({
        result: {
          logId: "inode-1",
          entryCount: 3,
          data: [
            { offset: 2, sessionId: "thread-2", ts: 2, text: "newest" },
            { offset: 1, sessionId: "thread-1", ts: 1, text: "middle" },
          ],
          nextCursor: "1",
        },
      })
      .mockResolvedValueOnce({
        result: {
          logId: "inode-1",
          entryCount: 3,
          data: [
            { offset: 0, sessionId: "thread-0", ts: 0, text: "oldest" },
          ],
          nextCursor: null,
        },
      });
    const client = {
      readPromptHistory: readPromptHistoryMock,
      appendPromptHistory: vi.fn(),
    } as unknown as PromptHistoryClient;

    await expect(readPromptHistory(client, 3)).resolves.toMatchObject([
      { text: "newest" },
      { text: "middle" },
      { text: "oldest" },
    ]);
    expect(readPromptHistoryMock).toHaveBeenNthCalledWith(2, {
      cursor: "1",
      limit: 3,
      logId: "inode-1",
    });
  });
});
