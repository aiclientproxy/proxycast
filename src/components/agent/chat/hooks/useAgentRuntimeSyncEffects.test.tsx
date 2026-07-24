import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanupRuntimeSyncEffectsTestHarness,
  createAppServerThreadClientMock,
  createThreadTurn,
  flushCoalescedRefresh,
  mockHasDesktopHostEventListenerCapability,
  mockHasDevBridgeEventListenerCapability,
  mockIsAppServerBridgeAvailable,
  mountHook,
  runtimeSyncRefreshRequest,
  setupRuntimeSyncEffectsTestHarness,
  terminalRefreshRequest,
} from "./useAgentRuntimeSyncEffects.testHarness";
import { listenAgentRuntimeEvent } from "@/lib/api/agentRuntimeEvents";
import { createThreadClient } from "@/lib/api/agentRuntime/threadClient";
import { createApplicationAdditionalContext } from "@/lib/api/agentProtocolOps";

describe("useAgentRuntimeSyncEffects", () => {
  beforeEach(() => {
    setupRuntimeSyncEffectsTestHarness();
  });

  afterEach(() => {
    cleanupRuntimeSyncEffectsTestHarness();
  });

  it("发送结束后应刷新当前会话详情", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      isSending: true,
      refreshSessionDetail,
    });

    try {
      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({ isSending: false });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).toHaveBeenCalledTimes(1);
      expect(refreshSessionDetail).toHaveBeenCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.sendSettled"),
      );
    } finally {
      harness.unmount();
    }
  });

  it("恢复队列工作时应轮询刷新当前会话详情", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      queuedTurnCount: 1,
      refreshSessionDetail,
    });

    try {
      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await act(async () => {
        vi.advanceTimersByTime(1500);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(1);
      expect(refreshSessionDetail).toHaveBeenCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.recoveredPoll"),
      );

      await harness.render({
        queuedTurnCount: 0,
        threadTurns: [createThreadTurn({ status: "running" })],
      });
      await act(async () => {
        vi.advanceTimersByTime(1500);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(2);
    } finally {
      harness.unmount();
    }
  });

  it("仅 thread_read 标记为 running 时也应继续轮询刷新", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      threadReadStatus: "running",
      refreshSessionDetail,
    });

    try {
      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await act(async () => {
        vi.advanceTimersByTime(1500);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(1);
      expect(refreshSessionDetail).toHaveBeenCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.recoveredPoll"),
      );
    } finally {
      harness.unmount();
    }
  });

  it("thread_read 只有 running status 和 active_turn_id 时也应继续轮询刷新", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      threadReadStatus: "running",
      threadRead: {
        status: "running",
        active_turn_id: "turn-running",
        turns: [],
      },
      refreshSessionDetail,
    });

    try {
      await act(async () => {
        vi.advanceTimersByTime(1500);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(1);
      expect(refreshSessionDetail).toHaveBeenCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.recoveredPoll"),
      );
    } finally {
      harness.unmount();
    }
  });

  it("current thread_read 已 idle 时本地 running turn 不应继续 recovered poll", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      threadReadStatus: "idle",
      threadRead: {
        status: "idle",
        turns: [],
      },
      threadTurns: [
        createThreadTurn({
          status: "running",
          updated_at: "2026-03-29T00:05:00.000Z",
        }),
      ],
      refreshSessionDetail,
    });

    try {
      await act(async () => {
        vi.advanceTimersByTime(3000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("failed read model 下残留 running turn 不应继续 recovered poll", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      threadReadStatus: "failed",
      threadTurns: [createThreadTurn({ status: "running" })],
      refreshSessionDetail,
    });

    try {
      await act(async () => {
        vi.advanceTimersByTime(3000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("发送中收到 failed read model 时即使本地 turn 残留 running 也应收起 active stream", async () => {
    const settleActiveRuntimeStream = vi.fn();
    const harness = await mountHook({
      isSending: true,
      currentStreamTurnId: "turn-current",
      threadReadStatus: "failed",
      threadRead: {
        thread_id: "thread-1",
        status: "failed",
        active_turn_id: "turn-current",
        turns: [],
      },
      threadTurns: [
        createThreadTurn({ id: "turn-current", status: "running" }),
      ],
      settleActiveRuntimeStream,
    });

    try {
      expect(settleActiveRuntimeStream).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("发送中收到 canceled read model 且本地 turn 残留 running 时也应收起 active stream", async () => {
    const settleActiveRuntimeStream = vi.fn();
    const harness = await mountHook({
      isSending: true,
      currentStreamTurnId: "turn-current",
      threadReadStatus: "canceled",
      threadRead: {
        thread_id: "thread-1",
        status: "canceled",
        active_turn_id: "turn-current",
        turns: [],
      },
      threadTurns: [
        createThreadTurn({ id: "turn-current", status: "running" }),
      ],
      settleActiveRuntimeStream,
    });

    try {
      expect(settleActiveRuntimeStream).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("发送中收到 cancelled read model 投影且本地 turn 残留 running 时也应收起 active stream", async () => {
    const settleActiveRuntimeStream = vi.fn();
    const harness = await mountHook({
      isSending: true,
      currentStreamTurnId: "turn-current",
      threadReadStatus: "cancelled",
      threadRead: {
        thread_id: "thread-1",
        status: "cancelled",
        active_turn_id: "turn-current",
        turns: [],
      },
      threadTurns: [
        createThreadTurn({ id: "turn-current", status: "running" }),
      ],
      settleActiveRuntimeStream,
    });

    try {
      expect(settleActiveRuntimeStream).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("恢复出的陈旧 running turn 不应持续轮询完整详情", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      threadReadStatus: "running",
      threadTurns: [
        createThreadTurn({
          status: "running",
          started_at: "2026-03-28T22:00:00.000Z",
          created_at: "2026-03-28T22:00:00.000Z",
          updated_at: "2026-03-28T22:05:00.000Z",
        }),
      ],
      refreshSessionDetail,
    });

    try {
      await act(async () => {
        vi.advanceTimersByTime(3000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("收到当前 turn 的 App Server runtime event 后应刷新 read model", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const refreshSessionReadModel = vi.fn(async () => true);
    const listeners = new Map<string, (event: { payload: unknown }) => void>();
    const runtime = {
      listenToTurnEvents: vi.fn(async (eventName, handler) => {
        listeners.set(
          eventName,
          handler as (event: { payload: unknown }) => void,
        );
        return () => {
          listeners.delete(eventName);
        };
      }),
    };
    const harness = await mountHook({
      runtime,
      currentTurnEventName: "agent_stream_assistant-1",
      isSending: true,
      refreshSessionDetail,
      refreshSessionReadModel,
    });

    try {
      expect(runtime.listenToTurnEvents).toHaveBeenCalledWith(
        "agent_stream_assistant-1",
        expect.any(Function),
      );

      await act(async () => {
        listeners.get("agent_stream_assistant-1")?.({
          payload: {
            type: "runtime_status",
            status: {
              phase: "running",
              title: "处理中",
            },
          },
        });
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      expect(refreshSessionReadModel).not.toHaveBeenCalled();
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await act(async () => {
        listeners.get("agent_stream_assistant-1")?.({
          payload: {
            type: "turn.completed",
            turn: {
              id: "turn-1",
              thread_id: "thread-1",
              prompt_text: "继续",
              status: "completed",
              started_at: "2026-03-29T00:05:00.000Z",
              completed_at: "2026-03-29T00:05:01.000Z",
              created_at: "2026-03-29T00:05:00.000Z",
              updated_at: "2026-03-29T00:05:01.000Z",
            },
          },
        });
        await Promise.resolve();
      });

      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await act(async () => {
        listeners.get("agent_stream_assistant-1")?.({
          payload: {
            type: "turn.failed",
            turn: {
              id: "turn-2",
              thread_id: "thread-1",
              prompt_text: "失败",
              status: "failed",
              started_at: "2026-03-29T00:06:00.000Z",
              completed_at: "2026-03-29T00:06:01.000Z",
              created_at: "2026-03-29T00:06:00.000Z",
              updated_at: "2026-03-29T00:06:01.000Z",
              error_message: "失败",
            },
          },
        });
        await Promise.resolve();
      });

      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await act(async () => {
        listeners.get("agent_stream_assistant-1")?.({
          payload: {
            type: "turn.canceled",
            turn: {
              id: "turn-3",
              thread_id: "thread-1",
              prompt_text: "停止",
              status: "canceled",
              started_at: "2026-03-29T00:05:02.000Z",
              completed_at: "2026-03-29T00:05:03.000Z",
              created_at: "2026-03-29T00:05:02.000Z",
              updated_at: "2026-03-29T00:05:03.000Z",
              error_message: "已停止",
            },
          },
        });
        await Promise.resolve();
      });

      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({ isSending: false });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({
        isSending: false,
        currentTurnEventName: null,
      });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      expect(refreshSessionReadModel).toHaveBeenCalledTimes(1);
      expect(refreshSessionReadModel).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("发送态 read model repair 到终态时应收起当前 runtime stream", async () => {
    const settleActiveRuntimeStream = vi.fn();
    const harness = await mountHook({
      isSending: true,
      threadReadStatus: "running",
      threadTurns: [
        createThreadTurn({
          status: "running",
          started_at: "2026-03-29T00:04:55.000Z",
          updated_at: "2026-03-29T00:04:58.000Z",
        }),
      ],
      settleActiveRuntimeStream,
    });

    try {
      expect(settleActiveRuntimeStream).not.toHaveBeenCalled();

      await harness.render({
        isSending: true,
        threadReadStatus: "completed",
        threadTurns: [
          createThreadTurn({
            status: "completed",
            started_at: "2026-03-29T00:04:55.000Z",
            completed_at: "2026-03-29T00:05:01.000Z",
            updated_at: "2026-03-29T00:05:01.000Z",
          }),
        ],
      });

      expect(settleActiveRuntimeStream).toHaveBeenCalledTimes(1);
      expect(settleActiveRuntimeStream).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("未观察到活跃 read model 前不应因 idle 快照误停新请求", async () => {
    const settleActiveRuntimeStream = vi.fn();
    const harness = await mountHook({
      isSending: true,
      threadReadStatus: "idle",
      threadTurns: [],
      settleActiveRuntimeStream,
    });

    try {
      await act(async () => {
        await Promise.resolve();
      });

      expect(settleActiveRuntimeStream).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("未观察到 running 快照时也应在 read model 已完成后收起发送态", async () => {
    const settleActiveRuntimeStream = vi.fn();
    const harness = await mountHook({
      isSending: true,
      threadReadStatus: "completed",
      threadTurns: [
        createThreadTurn({
          status: "completed",
          started_at: "2026-03-29T00:04:55.000Z",
          completed_at: "2026-03-29T00:05:01.000Z",
          updated_at: "2026-03-29T00:05:01.000Z",
        }),
      ],
      settleActiveRuntimeStream,
    });

    try {
      await act(async () => {
        await Promise.resolve();
      });

      expect(settleActiveRuntimeStream).toHaveBeenCalledTimes(1);
      expect(settleActiveRuntimeStream).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("stale queued read model 但 turn 已 terminal 时应收起发送态", async () => {
    const settleActiveRuntimeStream = vi.fn();
    const harness = await mountHook({
      isSending: true,
      threadReadStatus: "queued",
      queuedTurnCount: 0,
      threadTurns: [
        createThreadTurn({
          status: "completed",
          started_at: "2026-03-29T00:04:55.000Z",
          completed_at: "2026-03-29T00:05:01.000Z",
          updated_at: "2026-03-29T00:05:01.000Z",
        }),
      ],
      settleActiveRuntimeStream,
    });

    try {
      await act(async () => {
        await Promise.resolve();
      });

      expect(settleActiveRuntimeStream).toHaveBeenCalledTimes(1);
      expect(settleActiveRuntimeStream).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("当前 turn 的 text_delta 不应触发完整 read model 刷新", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const refreshSessionReadModel = vi.fn(async () => true);
    const listeners = new Map<string, (event: { payload: unknown }) => void>();
    const runtime = {
      listenToTurnEvents: vi.fn(async (eventName, handler) => {
        listeners.set(
          eventName,
          handler as (event: { payload: unknown }) => void,
        );
        return () => {
          listeners.delete(eventName);
        };
      }),
    };
    const harness = await mountHook({
      runtime,
      currentTurnEventName: "agent_stream_assistant-2",
      isSending: true,
      refreshSessionDetail,
      refreshSessionReadModel,
    });

    try {
      await act(async () => {
        listeners.get("agent_stream_assistant-2")?.({
          payload: {
            type: "text_delta",
            text: "增量内容",
          },
        });
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      expect(refreshSessionReadModel).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("当前 turn 的连续状态事件应合并为一次 read model 刷新", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const refreshSessionReadModel = vi.fn(async () => true);
    const listeners = new Map<string, (event: { payload: unknown }) => void>();
    const runtime = {
      listenToTurnEvents: vi.fn(async (eventName, handler) => {
        listeners.set(
          eventName,
          handler as (event: { payload: unknown }) => void,
        );
        return () => {
          listeners.delete(eventName);
        };
      }),
    };
    const harness = await mountHook({
      runtime,
      currentTurnEventName: "agent_stream_assistant-coalesced",
      isSending: true,
      refreshSessionDetail,
      refreshSessionReadModel,
    });

    try {
      await act(async () => {
        const listener = listeners.get("agent_stream_assistant-coalesced");
        listener?.({
          payload: {
            type: "runtime_status",
            status: { phase: "running" },
          },
        });
        listener?.({
          payload: {
            type: "turn.completed",
            turn: {
              id: "turn-completed",
              thread_id: "thread-1",
              prompt_text: "完成",
              status: "completed",
              started_at: "2026-03-29T00:05:00.000Z",
              completed_at: "2026-03-29T00:05:01.000Z",
              created_at: "2026-03-29T00:05:00.000Z",
              updated_at: "2026-03-29T00:05:01.000Z",
            },
          },
        });
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({ isSending: false });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({
        isSending: false,
        currentTurnEventName: null,
      });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      expect(terminalRefreshRequest("runtimeSync.event")).toEqual({
        source: "runtimeSync.event",
        detailMergeMode: "terminal_reconcile",
      });
      expect(refreshSessionReadModel).toHaveBeenCalledTimes(1);
      expect(refreshSessionReadModel).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("App Server turn notification 应通过当前 stream event 触发 read model 刷新", async () => {
    const eventName = "agent_stream_app-server-p3-126";
    const refreshSessionDetail = vi.fn(async () => true);
    const refreshSessionReadModel = vi.fn(async () => true);
    const appServerClient = createAppServerThreadClientMock();
    vi.mocked(appServerClient.startTurn).mockResolvedValueOnce({
      id: 1,
      result: {
        turn: {
          turnId: "turn-1",
          sessionId: "session-1",
          threadId: "thread-1",
          status: "accepted",
        },
      },
      response: {
        id: 1,
        result: {
          turn: {
            turnId: "turn-1",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "accepted",
          },
        },
      },
      messages: [],
      notifications: [
        {
          method: "turn/completed",
          params: {
            threadId: "thread-1",
            turn: {
              id: "turn-1",
              status: "completed",
              items: [],
              itemsView: "full",
              startedAt: Date.parse("2026-06-06T00:00:00.000Z") / 1_000,
              completedAt: Date.parse("2026-06-06T00:00:01.000Z") / 1_000,
            },
          },
        },
      ],
    });
    const threadClient = createThreadClient({
      appServerClient,
      invokeCommand: vi.fn(),
      isAppServerTurnLifecycleAvailable: () => true,
    });
    const runtime = {
      listenToTurnEvents: vi.fn((name, handler) =>
        listenAgentRuntimeEvent(name, handler),
      ),
    };
    const harness = await mountHook({
      runtime,
      currentTurnEventName: eventName,
      isSending: true,
      refreshSessionDetail,
      refreshSessionReadModel,
    });

    try {
      await vi.waitFor(() => {
        expect(runtime.listenToTurnEvents).toHaveBeenCalledWith(
          eventName,
          expect.any(Function),
        );
      });

      await act(async () => {
        await threadClient.submitAgentRuntimeTurn({
          threadId: "thread-1",
          input: [{ type: "text", text: "继续" }],
          additionalContext: createApplicationAdditionalContext({
            rendererEventName: eventName,
          }),
        });
        await Promise.resolve();
      });

      expect(appServerClient.startTurn).toHaveBeenCalledWith(
        expect.objectContaining({
          threadId: "thread-1",
          input: [{ type: "text", text: "继续" }],
        }),
      );
      expect(refreshSessionDetail).not.toHaveBeenCalled();
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({ isSending: false });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({
        isSending: false,
        currentTurnEventName: null,
      });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      expect(refreshSessionReadModel).toHaveBeenCalledTimes(1);
      expect(refreshSessionReadModel).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("收到当前 turn 的取消终态后应刷新 read model", async () => {
    const refreshSessionDetail = vi.fn(async () => true);
    const refreshSessionReadModel = vi.fn(async () => true);
    const listeners = new Map<string, (event: { payload: unknown }) => void>();
    const runtime = {
      listenToTurnEvents: vi.fn(async (eventName, handler) => {
        listeners.set(
          eventName,
          handler as (event: { payload: unknown }) => void,
        );
        return () => {
          listeners.delete(eventName);
        };
      }),
    };
    const harness = await mountHook({
      runtime,
      currentTurnEventName: "agent_stream_assistant-cancel",
      isSending: true,
      refreshSessionDetail,
      refreshSessionReadModel,
    });

    try {
      await act(async () => {
        listeners.get("agent_stream_assistant-cancel")?.({
          payload: {
            type: "turn.canceled",
            turn: {
              id: "turn-canceled",
              thread_id: "thread-1",
              prompt_text: "停止",
              status: "canceled",
              started_at: "2026-03-29T00:05:00.000Z",
              completed_at: "2026-03-29T00:05:01.000Z",
              created_at: "2026-03-29T00:05:00.000Z",
              updated_at: "2026-03-29T00:05:01.000Z",
            },
          },
        });
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({ isSending: false });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({
        isSending: false,
        currentTurnEventName: null,
      });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();
      expect(refreshSessionReadModel).toHaveBeenCalledTimes(1);
      expect(refreshSessionReadModel).toHaveBeenCalledWith("session-1");
    } finally {
      harness.unmount();
    }
  });

  it("浏览器 DevBridge 发送中但无原生事件能力时，应轮询刷新当前会话详情", async () => {
    mockIsAppServerBridgeAvailable.mockReturnValue(true);
    mockHasDesktopHostEventListenerCapability.mockReturnValue(false);
    mockHasDevBridgeEventListenerCapability.mockReturnValue(false);

    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      isSending: true,
      refreshSessionDetail,
    });

    try {
      expect(refreshSessionDetail).toHaveBeenCalledTimes(1);
      expect(refreshSessionDetail).toHaveBeenLastCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.poll"),
      );

      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(2);

      await harness.render({ isSending: false });
      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(3);
    } finally {
      harness.unmount();
    }
  });

  it("fallback poll 启动后拿到当前 turn event 时应停止轮询刷新", async () => {
    mockIsAppServerBridgeAvailable.mockReturnValue(true);
    mockHasDesktopHostEventListenerCapability.mockReturnValue(false);
    mockHasDevBridgeEventListenerCapability.mockReturnValue(false);

    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      isSending: true,
      refreshSessionDetail,
    });

    try {
      expect(refreshSessionDetail).toHaveBeenCalledTimes(1);
      expect(refreshSessionDetail).toHaveBeenLastCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.poll"),
      );

      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(2);

      await harness.render({
        currentTurnEventName: "agent_stream_late-bound",
        isSending: true,
      });

      await act(async () => {
        vi.advanceTimersByTime(3000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(2);
    } finally {
      harness.unmount();
    }
  });

  it("事件桥已绑定当前 turn event 时，不应再走轮询刷新", async () => {
    mockIsAppServerBridgeAvailable.mockReturnValue(true);
    mockHasDesktopHostEventListenerCapability.mockReturnValue(false);
    mockHasDevBridgeEventListenerCapability.mockReturnValue(true);

    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      currentTurnEventName: "agent_stream_event-bound",
      isSending: true,
      refreshSessionDetail,
    });

    try {
      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await act(async () => {
        vi.advanceTimersByTime(2000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({ isSending: false });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await harness.render({
        isSending: false,
        currentTurnEventName: null,
      });
      await flushCoalescedRefresh();

      expect(refreshSessionDetail).toHaveBeenCalledTimes(1);
      expect(refreshSessionDetail).toHaveBeenCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.sendSettled"),
      );
    } finally {
      harness.unmount();
    }
  });

  it("事件桥可用但当前 turn event 未恢复时，应在绑定宽限期后轮询刷新", async () => {
    mockIsAppServerBridgeAvailable.mockReturnValue(true);
    mockHasDesktopHostEventListenerCapability.mockReturnValue(false);
    mockHasDevBridgeEventListenerCapability.mockReturnValue(true);

    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      isSending: true,
      refreshSessionDetail,
    });

    try {
      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await act(async () => {
        vi.advanceTimersByTime(249);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();

      await act(async () => {
        vi.advanceTimersByTime(1);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(1);
      expect(refreshSessionDetail).toHaveBeenLastCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.poll"),
      );

      await act(async () => {
        vi.advanceTimersByTime(1000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).toHaveBeenCalledTimes(2);

      await harness.render({ isSending: false });

      await flushCoalescedRefresh();

      expect(refreshSessionDetail).toHaveBeenCalledTimes(3);
      expect(refreshSessionDetail).toHaveBeenLastCalledWith(
        "session-1",
        runtimeSyncRefreshRequest("runtimeSync.sendSettled"),
      );
    } finally {
      harness.unmount();
    }
  });

  it("事件桥在宽限期内绑定当前 turn event 时，不应启动 fallback poll", async () => {
    mockIsAppServerBridgeAvailable.mockReturnValue(true);
    mockHasDesktopHostEventListenerCapability.mockReturnValue(false);
    mockHasDevBridgeEventListenerCapability.mockReturnValue(true);

    const refreshSessionDetail = vi.fn(async () => true);
    const harness = await mountHook({
      isSending: true,
      refreshSessionDetail,
    });

    try {
      await act(async () => {
        vi.advanceTimersByTime(200);
        await Promise.resolve();
      });

      await harness.render({
        currentTurnEventName: "agent_stream_bound-in-grace",
        isSending: true,
      });

      await act(async () => {
        vi.advanceTimersByTime(3000);
        await Promise.resolve();
      });

      expect(refreshSessionDetail).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });
});
