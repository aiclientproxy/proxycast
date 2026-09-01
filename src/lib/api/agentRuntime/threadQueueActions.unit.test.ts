import { describe, expect, it, vi } from "vitest";
import {
  createThreadQueueActions,
  type ThreadQueueActionAppServerClient,
} from "./threadQueueActions";

function createClient(): ThreadQueueActionAppServerClient {
  return {
    addThreadQueue: vi.fn().mockResolvedValue({
      result: {
        queuedSubmission: {
          id: "queued-1",
          clientUserMessageId: "client-1",
          input: [{ type: "text", text: "hello" }],
        },
      },
    }),
    updateThreadQueue: vi.fn().mockResolvedValue({
      result: {
        queuedSubmission: {
          id: "queued-1",
          clientUserMessageId: "client-1",
          input: [{ type: "text", text: "updated" }],
        },
      },
    }),
    deleteThreadQueue: vi.fn().mockResolvedValue({ result: { deleted: true } }),
    reorderThreadQueue: vi.fn().mockResolvedValue({ result: {} }),
    startThreadQueue: vi.fn().mockResolvedValue({
      result: { turn: { id: "turn-1" } },
    }),
  } as ThreadQueueActionAppServerClient;
}

describe("threadQueueActions", () => {
  it("所有写操作都通过 typed App Server 方法并复制输入数组", async () => {
    const client = createClient();
    const actions = createThreadQueueActions({ appServerClient: client });
    const input = [{ type: "text" as const, text: "hello" }];

    await expect(
      actions.addThreadQueue({
        threadId: " thread-1 ",
        clientUserMessageId: " client-1 ",
        input,
      }),
    ).resolves.toMatchObject({ id: "queued-1" });
    await actions.updateThreadQueue({
      threadId: "thread-1",
      queuedSubmissionId: "queued-1",
      input,
    });
    await actions.deleteThreadQueue({
      threadId: "thread-1",
      queuedSubmissionId: "queued-1",
    });
    await actions.reorderThreadQueue({
      threadId: "thread-1",
      queuedSubmissionIds: ["queued-1"],
    });
    await actions.startThreadQueue({
      threadId: "thread-1",
      queuedSubmissionId: "queued-1",
    });

    expect(client.addThreadQueue).toHaveBeenCalledWith({
      threadId: "thread-1",
      clientUserMessageId: "client-1",
      input,
    });
    expect(client.updateThreadQueue).toHaveBeenCalledWith({
      threadId: "thread-1",
      queuedSubmissionId: "queued-1",
      input,
    });
    expect(client.deleteThreadQueue).toHaveBeenCalledWith({
      threadId: "thread-1",
      queuedSubmissionId: "queued-1",
    });
    expect(client.reorderThreadQueue).toHaveBeenCalledWith({
      threadId: "thread-1",
      queuedSubmissionIds: ["queued-1"],
    });
    expect(client.startThreadQueue).toHaveBeenCalledWith({
      threadId: "thread-1",
      queuedSubmissionId: "queued-1",
    });
  });

  it("拒绝空 ID 和空输入，避免把非法请求送进协议层", async () => {
    const actions = createThreadQueueActions({
      appServerClient: createClient(),
    });
    await expect(
      actions.addThreadQueue({
        threadId: "",
        clientUserMessageId: "client-1",
        input: [{ type: "text", text: "x" }],
      }),
    ).rejects.toThrow("threadId is required");
    await expect(
      actions.addThreadQueue({
        threadId: "thread-1",
        clientUserMessageId: "client-1",
        input: [],
      }),
    ).rejects.toThrow("input is required");
  });
});
