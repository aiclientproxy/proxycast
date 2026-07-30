import { describe, expect, it, vi } from "vitest";
import {
  METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
  METHOD_MCP_SERVER_ELICITATION_REQUEST,
  METHOD_SERVER_REQUEST_RESOLVED,
  type JsonRpcRequest,
} from "../../../packages/app-server-client/src/protocol";
import {
  APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
  AppServerServerRequestDispatcher,
} from "./appServerServerRequest";
import { AppServerEventBus } from "./appServerEventBus";
import { CLAW_TRACE_DEBUG_OVERRIDE_KEY } from "../developerFeatures";

function createHarness() {
  let subscription:
    | {
        onNotifications?: (
          notifications: Array<{ method: string; params?: unknown }>,
        ) => void;
        onServerRequests?: (requests: JsonRpcRequest[]) => void;
      }
    | undefined;
  const responder = {
    respondServerRequest: vi.fn(async () => undefined),
    rejectServerRequest: vi.fn(async () => undefined),
  };
  const eventBus = {
    subscribe: vi.fn((nextSubscription) => {
      subscription = nextSubscription;
      return vi.fn();
    }),
  };
  const dispatcher = new AppServerServerRequestDispatcher(responder, eventBus);
  return {
    dispatcher,
    eventBus,
    responder,
    publish: (request: JsonRpcRequest) =>
      subscription?.onServerRequests?.([request]),
    resolve: (requestId: JsonRpcRequest["id"], threadId = "thread-1") =>
      subscription?.onNotifications?.([
        {
          method: METHOD_SERVER_REQUEST_RESOLVED,
          params: { requestId, threadId },
        },
      ]),
    notify: (notification: { method: string; params?: unknown }) =>
      subscription?.onNotifications?.([notification]),
  };
}

function elicitationRequest(id = "app-server-request:1"): JsonRpcRequest {
  return {
    id,
    method: METHOD_MCP_SERVER_ELICITATION_REQUEST,
    params: {
      threadId: "thread-1",
      turnId: "turn-1",
      serverName: "form-server",
      mode: "form",
      _meta: null,
      message: "Choose a value",
      requestedSchema: {
        type: "object",
        properties: {},
      },
    },
  };
}

describe("AppServerServerRequestDispatcher", () => {
  it("共享 event bus 同时路由 notification 与 server request", async () => {
    const request = elicitationRequest("app-server-request:event-bus");
    const notification = {
      method: "agentSession/event",
      params: { event: { eventId: "event-1" } },
    };
    const drainEvents = vi
      .fn()
      .mockResolvedValueOnce([notification, request])
      .mockResolvedValue([]);
    const eventBus = new AppServerEventBus({ drainEvents });
    const onNotifications = vi.fn();
    const onServerRequests = vi.fn();
    const unsubscribe = eventBus.subscribe({
      getDrainOptions: () => ({ intervalMs: 1, limit: 2 }),
      onNotifications,
      onServerRequests,
    });

    await vi.waitFor(() => {
      expect(onNotifications).toHaveBeenCalledWith([notification]);
      expect(onServerRequests).toHaveBeenCalledWith([request]);
    });
    unsubscribe();
    eventBus.reset();
  });

  it("按 method 注册 typed handler 并以原 outer id 回包", async () => {
    const harness = createHarness();
    const handler = vi.fn(async () => ({
      action: "accept",
      content: { confirmed: true },
    }));
    harness.dispatcher.register(METHOD_MCP_SERVER_ELICITATION_REQUEST, handler);

    const request = elicitationRequest();
    await expect(harness.dispatcher.dispatch(request)).resolves.toBe(true);

    expect(handler).toHaveBeenCalledWith(
      request.params,
      request,
      expect.any(AbortSignal),
    );
    expect(harness.responder.respondServerRequest).toHaveBeenCalledWith(
      request.id,
      {
        action: "accept",
        content: { confirmed: true },
      },
    );
  });

  it("接受 Codex command approval current server request", async () => {
    const harness = createHarness();
    const handler = vi.fn(async () => ({ decision: "Accept" }));
    harness.dispatcher.register(
      METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
      handler,
    );
    const request = {
      id: "app-server-request:approval",
      method: METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
      params: {
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: "item-command-1",
        startedAtMs: 1,
      },
    };

    await expect(harness.dispatcher.dispatch(request)).resolves.toBe(true);
    expect(harness.responder.respondServerRequest).toHaveBeenCalledWith(
      request.id,
      { decision: "Accept" },
    );
  });

  it("Claw debug 模式只记录脱敏 server-request 生命周期", async () => {
    window.localStorage.setItem(CLAW_TRACE_DEBUG_OVERRIDE_KEY, "on");
    window.localStorage.removeItem(
      APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
    );
    try {
      const harness = createHarness();
      harness.dispatcher.register(
        METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
        async () => ({ decision: "acceptForSession" }),
      );
      const request = {
        id: "app-server-request:approval-trace",
        method: METHOD_ITEM_COMMAND_EXECUTION_REQUEST_APPROVAL,
        params: {
          threadId: "thread-1",
          turnId: "turn-1",
          itemId: "item-command-1",
          approvalId: "approval-1",
          command: "must-not-be-recorded",
          reason: "must-not-be-recorded",
        },
      };

      await expect(harness.dispatcher.dispatch(request)).resolves.toBe(true);
      harness.resolve(request.id);
      harness.notify({
        method: "item/completed",
        params: { threadId: "thread-1", turnId: "turn-1" },
      });

      const trace = JSON.parse(
        window.localStorage.getItem(
          APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
        ) || "[]",
      );
      expect(trace).toEqual([
        {
          sequence: 1,
          kind: "request",
          id: request.id,
          method: request.method,
          threadId: "thread-1",
          turnId: "turn-1",
          itemId: "item-command-1",
          approvalId: "approval-1",
        },
        {
          sequence: 2,
          kind: "response",
          id: request.id,
          method: request.method,
          threadId: "thread-1",
          turnId: "turn-1",
          itemId: "item-command-1",
          approvalId: "approval-1",
          decision: "acceptForSession",
        },
        {
          sequence: 3,
          kind: "resolved",
          method: METHOD_SERVER_REQUEST_RESOLVED,
          id: request.id,
          threadId: "thread-1",
        },
        {
          sequence: 4,
          kind: "terminal",
          method: "item/completed",
          threadId: "thread-1",
          turnId: "turn-1",
        },
      ]);
      expect(JSON.stringify(trace)).not.toContain("must-not-be-recorded");
    } finally {
      window.localStorage.removeItem(CLAW_TRACE_DEBUG_OVERRIDE_KEY);
      window.localStorage.removeItem(
        APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
      );
    }
  });

  it("未知 method 与 handler 错误都 fail closed 为 JSON-RPC error", async () => {
    const harness = createHarness();
    await expect(
      harness.dispatcher.dispatch({
        id: "app-server-request:unknown",
        method: "unknown/serverRequest",
        params: {},
      }),
    ).resolves.toBe(false);
    expect(harness.responder.rejectServerRequest).toHaveBeenLastCalledWith(
      "app-server-request:unknown",
      expect.objectContaining({ code: -32601 }),
    );

    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      async () => {
        throw new Error("elicitation handler failed");
      },
    );
    await expect(
      harness.dispatcher.dispatch(elicitationRequest("app-server-request:2")),
    ).resolves.toBe(false);
    expect(harness.responder.rejectServerRequest).toHaveBeenLastCalledWith(
      "app-server-request:2",
      {
        code: -32000,
        message: "elicitation handler failed",
      },
    );
  });

  it("同 outer id 并发重复请求只执行一次", async () => {
    const harness = createHarness();
    let release: (() => void) | undefined;
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      () =>
        new Promise((resolve) => {
          release = () => resolve({ action: "decline" });
        }),
    );
    const request = elicitationRequest("app-server-request:duplicate");
    const first = harness.dispatcher.dispatch(request);
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    await expect(harness.dispatcher.dispatch(request)).resolves.toBe(false);
    release?.();
    await expect(first).resolves.toBe(true);
    expect(harness.responder.respondServerRequest).toHaveBeenCalledTimes(1);
  });

  it("同 outer id 完成后重放不再次执行 handler", async () => {
    const harness = createHarness();
    const handler = vi.fn(async () => ({ action: "decline" }));
    harness.dispatcher.register(METHOD_MCP_SERVER_ELICITATION_REQUEST, handler);
    const request = elicitationRequest("app-server-request:settled");

    await expect(harness.dispatcher.dispatch(request)).resolves.toBe(true);
    await expect(harness.dispatcher.dispatch(request)).resolves.toBe(false);

    expect(handler).toHaveBeenCalledTimes(1);
    expect(harness.responder.respondServerRequest).toHaveBeenCalledTimes(1);
  });

  it("reset 后允许新连接复用旧 outer id", async () => {
    const harness = createHarness();
    const firstHandler = vi.fn(async () => ({ action: "decline" }));
    const request = elicitationRequest("app-server-request:reset");
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      firstHandler,
    );
    await expect(harness.dispatcher.dispatch(request)).resolves.toBe(true);

    harness.dispatcher.reset();
    const secondHandler = vi.fn(async () => ({ action: "cancel" }));
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      secondHandler,
    );
    await expect(harness.dispatcher.dispatch(request)).resolves.toBe(true);

    expect(firstHandler).toHaveBeenCalledTimes(1);
    expect(secondHandler).toHaveBeenCalledTimes(1);
  });

  it("resolved 早于 request 时保留 tombstone 且不打开 handler", async () => {
    const harness = createHarness();
    const handler = vi.fn(async () => ({ action: "decline" }));
    harness.dispatcher.register(METHOD_MCP_SERVER_ELICITATION_REQUEST, handler);
    const request = elicitationRequest("app-server-request:resolved-first");

    harness.resolve(request.id);

    await expect(harness.dispatcher.dispatch(request)).resolves.toBe(false);
    expect(handler).not.toHaveBeenCalled();
    expect(harness.responder.respondServerRequest).not.toHaveBeenCalled();
    expect(harness.responder.rejectServerRequest).not.toHaveBeenCalled();
  });

  it("重复 resolved 对同一 request key 保持幂等", () => {
    window.localStorage.setItem(CLAW_TRACE_DEBUG_OVERRIDE_KEY, "on");
    window.localStorage.removeItem(
      APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
    );
    try {
      const harness = createHarness();
      harness.dispatcher.register(
        METHOD_MCP_SERVER_ELICITATION_REQUEST,
        async () => ({ action: "decline" }),
      );

      harness.resolve("app-server-request:resolved-duplicate");
      harness.resolve("app-server-request:resolved-duplicate");

      const trace = JSON.parse(
        window.localStorage.getItem(
          APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
        ) || "[]",
      );
      expect(
        trace.filter((entry: { kind?: unknown }) => entry.kind === "resolved"),
      ).toHaveLength(1);
    } finally {
      window.localStorage.removeItem(CLAW_TRACE_DEBUG_OVERRIDE_KEY);
      window.localStorage.removeItem(
        APP_SERVER_SERVER_REQUEST_LIFECYCLE_TRACE_KEY,
      );
    }
  });

  it("resolved 中止正在等待的 handler 并禁止迟到回包", async () => {
    const harness = createHarness();
    let handlerSignal: AbortSignal | undefined;
    let release: (() => void) | undefined;
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      (_params, _request, signal) =>
        new Promise((resolve) => {
          handlerSignal = signal;
          release = () => resolve({ action: "accept", content: {} });
        }),
    );
    const request = elicitationRequest("app-server-request:remote-cancel");
    const dispatched = harness.dispatcher.dispatch(request);
    await vi.waitFor(() => expect(handlerSignal).toBeDefined());

    harness.resolve(request.id);
    expect(handlerSignal?.aborted).toBe(true);
    release?.();

    await expect(dispatched).resolves.toBe(false);
    expect(harness.responder.respondServerRequest).not.toHaveBeenCalled();
    expect(harness.responder.rejectServerRequest).not.toHaveBeenCalled();
  });

  it("不同 thread 的 resolved 不得中止 handler", async () => {
    const harness = createHarness();
    let handlerSignal: AbortSignal | undefined;
    let release: (() => void) | undefined;
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      (_params, _request, signal) =>
        new Promise((resolve) => {
          handlerSignal = signal;
          release = () => resolve({ action: "decline" });
        }),
    );
    const request = elicitationRequest("app-server-request:wrong-thread");
    const dispatched = harness.dispatcher.dispatch(request);
    await vi.waitFor(() => expect(handlerSignal).toBeDefined());

    harness.resolve(request.id, "thread-other");
    expect(handlerSignal?.aborted).toBe(false);
    release?.();

    await expect(dispatched).resolves.toBe(true);
    expect(harness.responder.respondServerRequest).toHaveBeenCalledWith(
      request.id,
      { action: "decline" },
    );
  });

  it("turn/completed 只中止同 Thread/Turn 的 pending handler", async () => {
    const harness = createHarness();
    const signals = new Map<string, AbortSignal>();
    const releases = new Map<string, () => void>();
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      (params, _request, signal) =>
        new Promise((resolve) => {
          const turnId = (params as { turnId: string }).turnId;
          signals.set(turnId, signal);
          releases.set(turnId, () => resolve({ action: "decline" }));
        }),
    );
    const first = elicitationRequest("app-server-request:terminal-first");
    const second = {
      ...elicitationRequest("app-server-request:terminal-second"),
      params: {
        ...elicitationRequest().params,
        turnId: "turn-2",
      },
    };
    const firstDispatch = harness.dispatcher.dispatch(first);
    const secondDispatch = harness.dispatcher.dispatch(second);
    await vi.waitFor(() => expect(signals.size).toBe(2));

    harness.notify({
      method: "turn/completed",
      params: { threadId: "thread-1", turnId: "turn-1" },
    });

    expect(signals.get("turn-1")?.aborted).toBe(true);
    expect(signals.get("turn-2")?.aborted).toBe(false);
    releases.get("turn-1")?.();
    releases.get("turn-2")?.();
    await expect(firstDispatch).resolves.toBe(false);
    await expect(secondDispatch).resolves.toBe(true);
  });

  it("thread/closed 中止该 Thread 的全部 pending handler", async () => {
    const harness = createHarness();
    let handlerSignal: AbortSignal | undefined;
    let release: (() => void) | undefined;
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      (_params, _request, signal) =>
        new Promise((resolve) => {
          handlerSignal = signal;
          release = () => resolve({ action: "decline" });
        }),
    );
    const dispatched = harness.dispatcher.dispatch(
      elicitationRequest("app-server-request:thread-closed"),
    );
    await vi.waitFor(() => expect(handlerSignal).toBeDefined());

    harness.notify({
      method: "thread/closed",
      params: { threadId: "thread-1" },
    });

    expect(handlerSignal?.aborted).toBe(true);
    release?.();
    await expect(dispatched).resolves.toBe(false);
    expect(harness.responder.respondServerRequest).not.toHaveBeenCalled();
  });

  it("缺少 threadId 的 resolved fail closed 且不中止 handler", async () => {
    const harness = createHarness();
    let handlerSignal: AbortSignal | undefined;
    let release: (() => void) | undefined;
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      (_params, _request, signal) =>
        new Promise((resolve) => {
          handlerSignal = signal;
          release = () => resolve({ action: "decline" });
        }),
    );
    const request = elicitationRequest("app-server-request:missing-thread");
    const dispatched = harness.dispatcher.dispatch(request);
    await vi.waitFor(() => expect(handlerSignal).toBeDefined());

    harness.notify({
      method: METHOD_SERVER_REQUEST_RESOLVED,
      params: { requestId: request.id },
    });
    expect(handlerSignal?.aborted).toBe(false);
    release?.();

    await expect(dispatched).resolves.toBe(true);
    expect(harness.responder.respondServerRequest).toHaveBeenCalledWith(
      request.id,
      { action: "decline" },
    );
  });

  it("reset 中止全部 handler 且旧 handler 完成后不回包", async () => {
    const harness = createHarness();
    let handlerSignal: AbortSignal | undefined;
    let release: (() => void) | undefined;
    harness.dispatcher.register(
      METHOD_MCP_SERVER_ELICITATION_REQUEST,
      (_params, _request, signal) =>
        new Promise((resolve) => {
          handlerSignal = signal;
          release = () => resolve({ action: "decline" });
        }),
    );
    const dispatched = harness.dispatcher.dispatch(
      elicitationRequest("app-server-request:reset-in-flight"),
    );
    await vi.waitFor(() => expect(handlerSignal).toBeDefined());

    harness.dispatcher.reset();
    expect(handlerSignal?.aborted).toBe(true);
    release?.();

    await expect(dispatched).resolves.toBe(false);
    expect(harness.responder.respondServerRequest).not.toHaveBeenCalled();
    expect(harness.responder.rejectServerRequest).not.toHaveBeenCalled();
  });
});
