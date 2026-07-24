import { afterEach, describe, expect, it, vi } from "vitest";
import {
  METHOD_SERVER_REQUEST_RESOLVED,
  type JsonRpcRequest,
} from "../../../packages/app-server-client/src/protocol";
import {
  AppServerEventBus,
  getDefaultAppServerEventBus,
  resetDefaultAppServerEventBusForTests,
} from "./appServerEventBus";

describe("AppServerEventBus", () => {
  afterEach(() => {
    resetDefaultAppServerEventBusForTests();
    vi.useRealTimers();
  });

  it("重置默认 event bus 后旧 drain loop 应退出", async () => {
    vi.useFakeTimers();
    const oldDrainEvents = vi.fn().mockResolvedValue([]);
    const oldEventBus = getDefaultAppServerEventBus({
      drainEvents: oldDrainEvents,
    });
    oldEventBus.subscribe({ onNotifications: vi.fn() });
    await Promise.resolve();

    expect(oldDrainEvents).toHaveBeenCalledTimes(1);

    resetDefaultAppServerEventBusForTests();
    const nextDrainEvents = vi.fn().mockResolvedValue([]);
    const nextEventBus = getDefaultAppServerEventBus({
      drainEvents: nextDrainEvents,
    });
    const unsubscribe = nextEventBus.subscribe({ onNotifications: vi.fn() });
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(500);

    expect(oldDrainEvents).toHaveBeenCalledTimes(1);
    expect(nextDrainEvents.mock.calls.length).toBeGreaterThan(1);

    unsubscribe();
    await vi.runOnlyPendingTimersAsync();
  });

  it("includeRecent subscription 应以对象参数 drain 最近镜像事件", async () => {
    vi.useFakeTimers();
    const drainEvents = vi.fn().mockResolvedValue([]);
    const eventBus = new AppServerEventBus({ drainEvents });

    const unsubscribe = eventBus.subscribe({
      getDrainOptions: () => ({
        includeRecent: true,
        intervalMs: 1_000,
        limit: 7,
      }),
      onNotifications: vi.fn(),
    });

    await Promise.resolve();

    expect(drainEvents).toHaveBeenCalledWith({
      includeRecent: true,
      limit: 7,
    });

    unsubscribe();
    await vi.runOnlyPendingTimersAsync();
  });

  it("includeRecent subscription 不应被 fast-first limit 压成 1", async () => {
    vi.useFakeTimers();
    const drainEvents = vi.fn().mockResolvedValue([]);
    const eventBus = new AppServerEventBus({ drainEvents });

    const unsubscribeRecent = eventBus.subscribe({
      getDrainOptions: () => ({
        includeRecent: true,
        intervalMs: 1_000,
        limit: 7,
      }),
      onNotifications: vi.fn(),
    });
    const unsubscribeFastFirst = eventBus.subscribe({
      getDrainOptions: () => ({
        intervalMs: 10,
        limit: 1,
      }),
      onNotifications: vi.fn(),
    });

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);

    expect(drainEvents).toHaveBeenLastCalledWith({
      includeRecent: true,
      limit: 7,
    });

    unsubscribeFastFirst();
    unsubscribeRecent();
    await vi.runOnlyPendingTimersAsync();
  });

  it("handler 尚未注册时保留 server request 并只投递一次", async () => {
    vi.useFakeTimers();
    const request: JsonRpcRequest = {
      id: "app-server-request:pending",
      method: "mcpServer/elicitation/request",
      params: {},
    };
    const drainEvents = vi
      .fn()
      .mockResolvedValueOnce([request])
      .mockResolvedValueOnce([request])
      .mockResolvedValue([]);
    const eventBus = new AppServerEventBus({ drainEvents });
    const unsubscribeNotifications = eventBus.subscribe({
      getDrainOptions: () => ({ intervalMs: 1, limit: 1 }),
      onNotifications: vi.fn(),
    });

    await vi.advanceTimersByTimeAsync(1);
    expect(drainEvents).toHaveBeenCalledTimes(2);
    const onServerRequests = vi.fn();
    const unsubscribeRequests = eventBus.subscribe({ onServerRequests });

    expect(onServerRequests).toHaveBeenCalledTimes(1);
    expect(onServerRequests).toHaveBeenCalledWith([request]);

    unsubscribeRequests();
    unsubscribeNotifications();
    eventBus.reset();
    await vi.runOnlyPendingTimersAsync();
  });

  it("只允许一个 server request handler", () => {
    const eventBus = new AppServerEventBus({ drainEvents: vi.fn() });
    const unsubscribe = eventBus.subscribe({ onServerRequests: vi.fn() });

    expect(() => eventBus.subscribe({ onServerRequests: vi.fn() })).toThrow(
      "already has a server request handler",
    );

    unsubscribe();
    eventBus.reset();
  });

  it.each([
    ["request 后 resolved", false],
    ["resolved 后 request", true],
  ])(
    "%s 且 handler 晚注册时不投递已撤销请求",
    async (_label, resolvedFirst) => {
      vi.useFakeTimers();
      const request: JsonRpcRequest = {
        id: "app-server-request:resolved-before-handler",
        method: "mcpServer/elicitation/request",
        params: { threadId: "thread-1" },
      };
      const resolved = {
        method: METHOD_SERVER_REQUEST_RESOLVED,
        params: { requestId: request.id, threadId: "thread-1" },
      };
      const drainEvents = vi
        .fn()
        .mockResolvedValueOnce(resolvedFirst ? [resolved, request] : [request])
        .mockResolvedValueOnce(resolvedFirst ? [] : [resolved])
        .mockResolvedValue([]);
      const eventBus = new AppServerEventBus({ drainEvents });
      const unsubscribeNotifications = eventBus.subscribe({
        getDrainOptions: () => ({ intervalMs: 1, limit: 2 }),
        onNotifications: vi.fn(),
      });

      await vi.advanceTimersByTimeAsync(resolvedFirst ? 1 : 2);
      const onServerRequests = vi.fn();
      const unsubscribeRequests = eventBus.subscribe({ onServerRequests });

      expect(onServerRequests).not.toHaveBeenCalled();

      unsubscribeRequests();
      unsubscribeNotifications();
      eventBus.reset();
      await vi.runOnlyPendingTimersAsync();
    },
  );

  it("reset 后应清除旧 generation 的 resolved tombstone", async () => {
    vi.useFakeTimers();
    const request: JsonRpcRequest = {
      id: "app-server-request:reset-tombstone",
      method: "mcpServer/elicitation/request",
      params: { threadId: "thread-1" },
    };
    const resolved = {
      method: METHOD_SERVER_REQUEST_RESOLVED,
      params: { requestId: request.id, threadId: "thread-1" },
    };
    const drainEvents = vi
      .fn()
      .mockResolvedValueOnce([resolved])
      .mockResolvedValue([]);
    const eventBus = new AppServerEventBus({ drainEvents });
    eventBus.subscribe({
      getDrainOptions: () => ({ intervalMs: 1, limit: 1 }),
      onNotifications: vi.fn(),
    });
    await Promise.resolve();

    eventBus.reset();
    drainEvents
      .mockReset()
      .mockResolvedValueOnce([request])
      .mockResolvedValue([]);
    const onServerRequests = vi.fn();
    const unsubscribeRequests = eventBus.subscribe({
      getDrainOptions: () => ({ intervalMs: 1, limit: 1 }),
      onServerRequests,
    });
    await vi.advanceTimersByTimeAsync(1);

    expect(onServerRequests).toHaveBeenCalledWith([request]);

    unsubscribeRequests();
    eventBus.reset();
    await vi.runOnlyPendingTimersAsync();
  });

  it("reset 后应忽略旧 generation 尚未完成的 drain", async () => {
    vi.useFakeTimers();
    const request: JsonRpcRequest = {
      id: "app-server-request:old-drain",
      method: "mcpServer/elicitation/request",
      params: { threadId: "thread-1" },
    };
    const resolved = {
      method: METHOD_SERVER_REQUEST_RESOLVED,
      params: { requestId: request.id, threadId: "thread-1" },
    };
    let resolveOldDrain: ((messages: unknown[]) => void) | undefined;
    const drainEvents = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<unknown[]>((resolve) => {
            resolveOldDrain = resolve;
          }),
      )
      .mockResolvedValueOnce([request])
      .mockResolvedValue([]);
    const eventBus = new AppServerEventBus({ drainEvents });
    eventBus.subscribe({ onNotifications: vi.fn() });
    await Promise.resolve();

    eventBus.reset();
    const onServerRequests = vi.fn();
    const unsubscribeRequests = eventBus.subscribe({ onServerRequests });
    resolveOldDrain?.([resolved, request]);
    await vi.advanceTimersByTimeAsync(0);

    expect(onServerRequests).toHaveBeenCalledTimes(1);
    expect(onServerRequests).toHaveBeenCalledWith([request]);

    unsubscribeRequests();
    eventBus.reset();
    await vi.runOnlyPendingTimersAsync();
  });

  it("缺少 threadId 的 resolved 不得撤销后续 server request", async () => {
    vi.useFakeTimers();
    const request: JsonRpcRequest = {
      id: "app-server-request:unscoped-resolved",
      method: "mcpServer/elicitation/request",
      params: {},
    };
    const drainEvents = vi
      .fn()
      .mockResolvedValueOnce([
        {
          method: METHOD_SERVER_REQUEST_RESOLVED,
          params: { requestId: request.id },
        },
        request,
      ])
      .mockResolvedValue([]);
    const eventBus = new AppServerEventBus({ drainEvents });
    const onServerRequests = vi.fn();
    const unsubscribe = eventBus.subscribe({
      getDrainOptions: () => ({ intervalMs: 1, limit: 2 }),
      onServerRequests,
    });

    await vi.advanceTimersByTimeAsync(1);

    expect(onServerRequests).toHaveBeenCalledWith([request]);

    unsubscribe();
    eventBus.reset();
    await vi.runOnlyPendingTimersAsync();
  });

  it("不同 thread 的 resolved 不得吞掉同 outer id 请求", async () => {
    vi.useFakeTimers();
    const request: JsonRpcRequest = {
      id: "app-server-request:thread-scope",
      method: "mcpServer/elicitation/request",
      params: { threadId: "thread-2" },
    };
    const drainEvents = vi
      .fn()
      .mockResolvedValueOnce([
        {
          method: METHOD_SERVER_REQUEST_RESOLVED,
          params: { requestId: request.id, threadId: "thread-1" },
        },
        request,
      ])
      .mockResolvedValue([]);
    const eventBus = new AppServerEventBus({ drainEvents });
    const onServerRequests = vi.fn();
    const unsubscribe = eventBus.subscribe({
      getDrainOptions: () => ({ intervalMs: 1, limit: 2 }),
      onServerRequests,
    });

    await vi.advanceTimersByTimeAsync(1);

    expect(onServerRequests).toHaveBeenCalledWith([request]);

    unsubscribe();
    eventBus.reset();
    await vi.runOnlyPendingTimersAsync();
  });
});
