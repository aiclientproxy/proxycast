import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppServerReviewStartResponse } from "./appServer";
import { startReview } from "./review";

function createResponse(): { result: AppServerReviewStartResponse } {
  return {
    result: {
      turn: {
        id: "turn-review",
        items: [],
        itemsView: "notLoaded",
        status: "inProgress",
      },
      reviewThreadId: "thread-1",
    },
  };
}

function createClient() {
  return {
    startReview: vi.fn().mockResolvedValue(createResponse()),
  };
}

describe("review gateway", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("通过 typed App Server client 启动 inline review", async () => {
    const client = createClient();

    await expect(
      startReview(
        {
          threadId: " thread-1 ",
          delivery: "inline",
          target: {
            type: "commit",
            sha: " abc123 ",
            title: " Tidy colors ",
          },
        },
        {},
        { client },
      ),
    ).resolves.toEqual(
      expect.objectContaining({ reviewThreadId: "thread-1" }),
    );
    expect(client.startReview).toHaveBeenCalledWith({
      threadId: "thread-1",
      delivery: "inline",
      target: {
        type: "commit",
        sha: "abc123",
        title: "Tidy colors",
      },
    });
  });

  it.each([
    ["threadId", { threadId: " ", target: { type: "uncommittedChanges" } }],
    [
      "branch",
      {
        threadId: "thread-1",
        target: { type: "baseBranch", branch: " " },
      },
    ],
    [
      "sha",
      { threadId: "thread-1", target: { type: "commit", sha: " " } },
    ],
    [
      "instructions",
      {
        threadId: "thread-1",
        target: { type: "custom", instructions: " " },
      },
    ],
  ] as const)("拒绝空白 %s", async (field, params) => {
    await expect(
      startReview(params, {}, { client: createClient() }),
    ).rejects.toThrow(field);
  });

  it("将空白 commit title 规范为 null", async () => {
    const client = createClient();

    await startReview(
      {
        threadId: "thread-1",
        target: { type: "commit", sha: "abc123", title: " " },
      },
      {},
      { client },
    );

    expect(client.startReview).toHaveBeenCalledWith(
      expect.objectContaining({
        target: { type: "commit", sha: "abc123", title: null },
      }),
    );
  });

  it("应先监听同一 thread，再发起 review/start，并在匹配终态后自动解除监听", async () => {
    const order: string[] = [];
    const unlisten = vi.fn();
    let eventHandler: ((event: { payload: unknown }) => void) | null = null;
    const listenRuntimeEvent = vi.fn(async (_eventName, handler) => {
      order.push("listen");
      eventHandler = handler;
      return unlisten;
    });
    const client = createClient();
    client.startReview.mockImplementationOnce(async () => {
      order.push("request");
      return createResponse();
    });
    const onTerminal = vi.fn();

    await startReview(
      {
        threadId: "thread-1",
        target: { type: "uncommittedChanges" },
      },
      { onTerminal },
      { client, listenRuntimeEvent },
    );

    expect(order).toEqual(["listen", "request"]);
    expect(listenRuntimeEvent).toHaveBeenCalledWith(
      "agentSession/event/thread-1",
      expect.any(Function),
    );
    eventHandler?.({
      payload: {
        type: "turn.completed",
        turn: { id: "turn-review", status: "completed", items: [] },
      },
    });
    await Promise.resolve();
    await Promise.resolve();

    expect(onTerminal).toHaveBeenCalledWith(
      expect.objectContaining({ type: "turn_completed" }),
    );
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("应捕获 admission 响应前到达的快速终态", async () => {
    let eventHandler: ((event: { payload: unknown }) => void) | null = null;
    let resolveRequest: ((value: ReturnType<typeof createResponse>) => void) | null =
      null;
    let markListenerReady: (() => void) | null = null;
    let markRequestReady: (() => void) | null = null;
    const listenerReady = new Promise<void>((resolve) => {
      markListenerReady = resolve;
    });
    const requestReady = new Promise<void>((resolve) => {
      markRequestReady = resolve;
    });
    const unlisten = vi.fn();
    const listenRuntimeEvent = vi.fn(async (_eventName, handler) => {
      eventHandler = handler;
      markListenerReady?.();
      return unlisten;
    });
    const client = {
      startReview: vi.fn(
        () =>
          new Promise<ReturnType<typeof createResponse>>((resolve) => {
            resolveRequest = resolve;
            markRequestReady?.();
          }),
      ),
    };
    const onTerminal = vi.fn(() => {
      throw new Error("refresh failed");
    });
    const request = startReview(
      {
        threadId: "thread-1",
        target: { type: "uncommittedChanges" },
      },
      { onTerminal },
      { client, listenRuntimeEvent },
    );
    await Promise.all([listenerReady, requestReady]);

    eventHandler?.({
      payload: {
        type: "turn_completed",
        turn: { id: "turn-review", status: "completed", items: [] },
      },
    });
    resolveRequest?.(createResponse());

    await expect(request).resolves.toEqual(
      expect.objectContaining({ reviewThreadId: "thread-1" }),
    );
    await Promise.resolve();
    await Promise.resolve();
    expect(onTerminal).toHaveBeenCalledTimes(1);
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("review/start 失败或终态等待超时都应解除监听", async () => {
    vi.useFakeTimers();
    const requestFailureUnlisten = vi.fn();
    const timeoutUnlisten = vi.fn();
    const failingClient = {
      startReview: vi.fn().mockRejectedValue(new Error("request failed")),
    };

    await expect(
      startReview(
        {
          threadId: "thread-1",
          target: { type: "uncommittedChanges" },
        },
        { onTerminal: vi.fn() },
        {
          client: failingClient,
          listenRuntimeEvent: vi
            .fn()
            .mockResolvedValue(requestFailureUnlisten),
        },
      ),
    ).rejects.toThrow("request failed");
    expect(requestFailureUnlisten).toHaveBeenCalledTimes(1);

    await startReview(
      {
        threadId: "thread-1",
        target: { type: "uncommittedChanges" },
      },
      { onTerminal: vi.fn() },
      {
        client: createClient(),
        listenRuntimeEvent: vi.fn().mockResolvedValue(timeoutUnlisten),
        terminalTimeoutMs: 50,
      },
    );
    await vi.advanceTimersByTimeAsync(50);
    expect(timeoutUnlisten).toHaveBeenCalledTimes(1);
  });
});
