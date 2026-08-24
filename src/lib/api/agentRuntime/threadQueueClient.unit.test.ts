import { describe, expect, it, vi } from "vitest";
import type { QueuedSubmission } from "@limecloud/app-server-client";
import {
  createThreadQueueClient,
  parseQueuedSubmission,
  type ThreadQueueAppServerClient,
} from "./threadQueueClient";

const first: QueuedSubmission = {
  clientUserMessageId: "client-1",
  id: "queued-1",
  input: [{ type: "text", text: "first" }],
};
const second: QueuedSubmission = {
  clientUserMessageId: "client-2",
  id: "queued-2",
  input: [{ type: "image", url: "sidecar://media/input-2", detail: "high" }],
};

describe("threadQueueClient", () => {
  it("应通过 typed App Server client 读取全部分页 Queue", async () => {
    const appServerClient = {
      listThreadQueue: vi
        .fn()
        .mockResolvedValueOnce({
          result: { data: [first], nextCursor: "1" },
        })
        .mockResolvedValueOnce({
          result: { data: [second], nextCursor: null },
        }),
    } as ThreadQueueAppServerClient;
    const client = createThreadQueueClient({ appServerClient });

    await expect(client.listThreadQueue(" thread-1 ")).resolves.toEqual([
      first,
      second,
    ]);
    expect(appServerClient.listThreadQueue).toHaveBeenNthCalledWith(1, {
      threadId: "thread-1",
      limit: 100,
    });
    expect(appServerClient.listThreadQueue).toHaveBeenNthCalledWith(2, {
      threadId: "thread-1",
      limit: 100,
      cursor: "1",
    });
  });

  it("应拒绝空 thread、损坏 submission 和循环 cursor", async () => {
    const invalidClient = createThreadQueueClient({
      appServerClient: {
        listThreadQueue: vi.fn().mockResolvedValue({
          result: {
            data: [{ ...first, input: [] }],
            nextCursor: null,
          },
        }),
      } as ThreadQueueAppServerClient,
    });
    await expect(invalidClient.listThreadQueue(" ")).rejects.toThrow(
      "threadId is required",
    );
    await expect(invalidClient.listThreadQueue("thread-1")).rejects.toThrow(
      "invalid queued submission",
    );

    const repeatedCursor = createThreadQueueClient({
      appServerClient: {
        listThreadQueue: vi.fn().mockResolvedValue({
          result: { data: [first], nextCursor: "same" },
        }),
      } as ThreadQueueAppServerClient,
    });
    await expect(repeatedCursor.listThreadQueue("thread-1")).rejects.toThrow(
      "repeated nextCursor",
    );
  });

  it("只接受 generated Queue input 的 canonical shape", () => {
    expect(parseQueuedSubmission(first)).toEqual(first);
    expect(
      parseQueuedSubmission({
        ...first,
        input: [{ type: "remote-image", url: "https://example.test" }],
      }),
    ).toBeNull();
    expect(
      parseQueuedSubmission({ ...first, clientUserMessageId: "" }),
    ).toBeNull();
  });
});
