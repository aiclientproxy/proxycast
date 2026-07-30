import { describe, expect, it, vi } from "vitest";
import type { Dispatch, SetStateAction } from "react";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type { AgentThreadItem, AgentThreadTurn } from "@/lib/api/agentProtocol";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import { createConversationProjectionReducer } from "@/lib/api/agentRuntime/conversationProjection";
import type { ActionRequired, Message } from "../types";
import type { ActiveStreamState } from "./agentStreamSubmissionLifecycle";
import {
  bindRecoveredAgentStreamThread,
  rememberLocallyInterruptedAgentStreamBinding,
  rememberLocallyStartedAgentStreamBinding,
  resolveAgentStreamResumeBindingTarget,
} from "./agentStreamResumeBinding";

function createStateSetter<T>(state: { current: T }) {
  return ((next: T | ((prev: T) => T)) => {
    state.current =
      typeof next === "function"
        ? (next as (prev: T) => T)(state.current)
        : next;
  }) as Dispatch<SetStateAction<T>>;
}

describe("agentStreamResumeBinding", () => {
  it("running read model 应解析成固定 session event 绑定目标", () => {
    expect(
      resolveAgentStreamResumeBindingTarget({
        sessionId: "session-1",
        threadBusy: true,
        currentTurnId: null,
        threadRead: {
          thread_id: "thread-1",
          status: "running",
          active_turn_id: "turn-1",
          turns: [
            {
              turn_id: "turn-1",
              status: "running",
              started_at: new Date().toISOString(),
            },
          ],
        },
        threadTurns: [],
      }),
    ).toEqual({
      eventName: "agentSession/event/session-1",
      sessionId: "session-1",
      threadId: "thread-1",
      turnId: "turn-1",
      startedAt: null,
    });
  });

  it("缺少 active_turn_id 时不应从兼容 running turn 猜恢复目标", () => {
    expect(
      resolveAgentStreamResumeBindingTarget({
        sessionId: "session-1",
        threadBusy: true,
        currentTurnId: null,
        threadRead: {
          thread_id: "thread-1",
          status: "running",
          active_turn_id: null,
          turns: [
            {
              turnId: "turn-current-1",
              status: "running",
              startedAt: new Date().toISOString(),
            },
          ],
        } as AgentRuntimeThreadReadModel,
        threadTurns: [],
      }),
    ).toBeNull();
  });

  it("只有 thread 级 running 或孤立 active_turn_id 时不恢复 active stream", () => {
    expect(
      resolveAgentStreamResumeBindingTarget({
        sessionId: "session-1",
        threadBusy: true,
        currentTurnId: "turn-stale",
        threadRead: {
          thread_id: "thread-1",
          status: "running",
          active_turn_id: "turn-stale",
          turns: [],
        },
        threadTurns: [],
      }),
    ).toBeNull();
  });

  it("只有 queued turn 时不绑定 active stream", () => {
    expect(
      resolveAgentStreamResumeBindingTarget({
        sessionId: "session-1",
        threadBusy: true,
        currentTurnId: null,
        threadRead: {
          thread_id: "thread-1",
          status: "queued",
        },
        threadTurns: [],
      }),
    ).toBeNull();
  });

  it("只有本地 stale running turn 时不绑定 active stream", () => {
    expect(
      resolveAgentStreamResumeBindingTarget({
        sessionId: "session-1",
        threadBusy: true,
        currentTurnId: "turn-stale",
        threadRead: null,
        threadTurns: [
          {
            id: "turn-stale",
            thread_id: "thread-1",
            prompt_text: "继续",
            status: "running",
            started_at: "2026-03-29T00:00:00.000Z",
            created_at: "2026-03-29T00:00:00.000Z",
            updated_at: "2026-03-29T00:00:01.000Z",
          },
        ],
      }),
    ).toBeNull();
  });

  it("running turn 存在时应恢复 active stream", () => {
    expect(
      resolveAgentStreamResumeBindingTarget({
        sessionId: "session-1",
        threadBusy: true,
        currentTurnId: null,
        threadRead: {
          thread_id: "thread-1",
          status: "running",
          active_turn_id: "turn-running-1",
          turns: [
            {
              turn_id: "turn-running-1",
              status: "running",
            },
          ],
        },
        threadTurns: [],
      }),
    ).toEqual({
      eventName: "agentSession/event/session-1",
      sessionId: "session-1",
      threadId: "thread-1",
      turnId: "turn-running-1",
      startedAt: null,
    });
  });

  it("failed read model 残留 active turn 时不应恢复 active stream", () => {
    expect(
      resolveAgentStreamResumeBindingTarget({
        sessionId: "session-1",
        threadBusy: true,
        currentTurnId: "turn-stale",
        threadRead: {
          thread_id: "thread-1",
          status: "failed",
          profile_status: "failed",
          active_turn_id: "turn-stale",
          turns: [
            {
              turn_id: "turn-stale",
              status: "running",
            },
          ],
        },
        threadTurns: [
          {
            id: "turn-stale",
            thread_id: "thread-1",
            prompt_text: "继续",
            status: "running",
            started_at: "2026-03-29T00:00:00.000Z",
            created_at: "2026-03-29T00:00:00.000Z",
            updated_at: "2026-03-29T00:00:01.000Z",
          },
        ],
      }),
    ).toBeNull();
  });

  it("已有 live stream listener 时不应抢占并恢复 session event", async () => {
    const liveUnlisten = vi.fn();
    const runtime = {
      listenToTurnEvents: vi.fn(async () => vi.fn()),
      resumeThread: vi.fn(async () => true),
    };
    const activeStreamState: { current: ActiveStreamState | null } = {
      current: null,
    };
    const listenerMapRef = {
      current: new Map<string, () => void>([
        ["agent_stream_assistant-1", liveUnlisten],
      ]),
    };
    const messages = { current: [] as Message[] };
    const threadTurns = { current: [] as AgentThreadTurn[] };
    const threadItems = { current: [] as AgentThreadItem[] };
    const pendingActions = { current: [] as ActionRequired[] };
    const executionRuntime = {
      current: null as AgentSessionExecutionRuntime | null,
    };
    const currentTurnId = { current: null as string | null };
    let isSending = true;
    const setIsSending = (next: boolean | ((previous: boolean) => boolean)) => {
      isSending =
        typeof next === "function"
          ? (next as (previous: boolean) => boolean)(isSending)
          : next;
    };

    const cleanup = await bindRecoveredAgentStreamThread({
      activeStreamRef: activeStreamState,
      appendThinkingToParts: (parts, textDelta) => [
        ...parts,
        { type: "thinking", text: textDelta },
      ],
      clearActiveStreamIfMatch: vi.fn(() => false),
      executionStrategy: "react",
      getMessages: () => messages.current,
      getThreadItems: () => threadItems.current,
      listenerMapRef,
      refreshSessionReadModel: vi.fn(async () => true),
      runtime,
      setActiveStream: (nextActive) => {
        activeStreamState.current = nextActive;
      },
      setCurrentTurnId: createStateSetter(currentTurnId),
      setExecutionRuntime: createStateSetter(executionRuntime),
      setIsSending,
      setMessages: createStateSetter(messages),
      setPendingActions: createStateSetter(pendingActions),
      setThreadItems: createStateSetter(threadItems),
      setThreadTurns: createStateSetter(threadTurns),
      target: {
        eventName: "agentSession/event/session-1",
        sessionId: "session-1",
        threadId: "thread-1",
        turnId: "turn-1",
        startedAt: "2026-07-06T00:00:00.000Z",
      },
      warnedKeysRef: { current: new Set<string>() },
    });

    expect(cleanup).toBeNull();
    expect(runtime.listenToTurnEvents).not.toHaveBeenCalled();
    expect(runtime.resumeThread).not.toHaveBeenCalled();
    expect(listenerMapRef.current.get("agent_stream_assistant-1")).toBe(
      liveUnlisten,
    );
    expect(activeStreamState.current).toBeNull();
    expect(isSending).toBe(true);
  });

  it("同标签本地刚提交的 running session 不应提前恢复 session event", async () => {
    const runtime = {
      listenToTurnEvents: vi.fn(async () => vi.fn()),
      resumeThread: vi.fn(async () => true),
    };
    const activeStreamState: { current: ActiveStreamState | null } = {
      current: null,
    };
    const listenerMapRef = { current: new Map<string, () => void>() };
    const messages = { current: [] as Message[] };
    const threadTurns = { current: [] as AgentThreadTurn[] };
    const threadItems = { current: [] as AgentThreadItem[] };
    const pendingActions = { current: [] as ActionRequired[] };
    const executionRuntime = {
      current: null as AgentSessionExecutionRuntime | null,
    };
    const currentTurnId = { current: null as string | null };
    let isSending = false;
    const setIsSending = (next: boolean | ((previous: boolean) => boolean)) => {
      isSending =
        typeof next === "function"
          ? (next as (previous: boolean) => boolean)(isSending)
          : next;
    };

    rememberLocallyStartedAgentStreamBinding({
      assistantMsgId: "assistant-local-1",
      eventName: "agent_stream_assistant-local-1",
      sessionId: "local-session-1",
      turnId: "local-turn-1",
    });

    const cleanup = await bindRecoveredAgentStreamThread({
      activeStreamRef: activeStreamState,
      appendThinkingToParts: (parts, textDelta) => [
        ...parts,
        { type: "thinking", text: textDelta },
      ],
      clearActiveStreamIfMatch: vi.fn(() => false),
      executionStrategy: "react",
      getMessages: () => messages.current,
      getThreadItems: () => threadItems.current,
      listenerMapRef,
      refreshSessionReadModel: vi.fn(async () => true),
      runtime,
      setActiveStream: (nextActive) => {
        activeStreamState.current = nextActive;
      },
      setCurrentTurnId: createStateSetter(currentTurnId),
      setExecutionRuntime: createStateSetter(executionRuntime),
      setIsSending,
      setMessages: createStateSetter(messages),
      setPendingActions: createStateSetter(pendingActions),
      setThreadItems: createStateSetter(threadItems),
      setThreadTurns: createStateSetter(threadTurns),
      target: {
        eventName: "agentSession/event/local-session-1",
        sessionId: "local-session-1",
        threadId: "local-thread-1",
        turnId: "local-turn-1",
        startedAt: "2026-07-06T00:00:00.000Z",
      },
      warnedKeysRef: { current: new Set<string>() },
    });

    expect(cleanup).toBeNull();
    expect(runtime.listenToTurnEvents).not.toHaveBeenCalled();
    expect(runtime.resumeThread).not.toHaveBeenCalled();
    expect(activeStreamState.current).toBeNull();
    expect(isSending).toBe(false);
  });

  it("同标签本地刚停止的 running session 不应被 stale read model 重新恢复", async () => {
    const runtime = {
      listenToTurnEvents: vi.fn(async () => vi.fn()),
      resumeThread: vi.fn(async () => true),
    };
    const activeStreamState: { current: ActiveStreamState | null } = {
      current: null,
    };
    const listenerMapRef = { current: new Map<string, () => void>() };
    const messages = { current: [] as Message[] };
    const threadTurns = { current: [] as AgentThreadTurn[] };
    const threadItems = { current: [] as AgentThreadItem[] };
    const pendingActions = { current: [] as ActionRequired[] };
    const executionRuntime = {
      current: null as AgentSessionExecutionRuntime | null,
    };
    const currentTurnId = { current: null as string | null };
    let isSending = false;
    const setIsSending = (next: boolean | ((previous: boolean) => boolean)) => {
      isSending =
        typeof next === "function"
          ? (next as (previous: boolean) => boolean)(isSending)
          : next;
    };

    rememberLocallyInterruptedAgentStreamBinding({
      assistantMsgId: "assistant-interrupted-1",
      eventName: "agent_stream_interrupted-1",
      sessionId: "interrupted-session-1",
      turnId: "interrupted-turn-1",
    });

    const cleanup = await bindRecoveredAgentStreamThread({
      activeStreamRef: activeStreamState,
      appendThinkingToParts: (parts, textDelta) => [
        ...parts,
        { type: "thinking", text: textDelta },
      ],
      clearActiveStreamIfMatch: vi.fn(() => false),
      executionStrategy: "react",
      getMessages: () => messages.current,
      getThreadItems: () => threadItems.current,
      listenerMapRef,
      refreshSessionReadModel: vi.fn(async () => true),
      runtime,
      setActiveStream: (nextActive) => {
        activeStreamState.current = nextActive;
      },
      setCurrentTurnId: createStateSetter(currentTurnId),
      setExecutionRuntime: createStateSetter(executionRuntime),
      setIsSending,
      setMessages: createStateSetter(messages),
      setPendingActions: createStateSetter(pendingActions),
      setThreadItems: createStateSetter(threadItems),
      setThreadTurns: createStateSetter(threadTurns),
      target: {
        eventName: "agentSession/event/interrupted-session-1",
        sessionId: "interrupted-session-1",
        threadId: "interrupted-thread-1",
        turnId: "interrupted-turn-1",
        startedAt: "2026-07-06T00:00:00.000Z",
      },
      warnedKeysRef: { current: new Set<string>() },
    });

    expect(cleanup).toBeNull();
    expect(runtime.listenToTurnEvents).not.toHaveBeenCalled();
    expect(runtime.resumeThread).not.toHaveBeenCalled();
    expect(activeStreamState.current).toBeNull();
    expect(isSending).toBe(false);
  });

  it("恢复绑定默认只显示 reasoning summary，并在终态事件后清理 active stream", async () => {
    const unlisten = vi.fn();
    let eventHandler: ((event: { payload: unknown }) => void) | null = null;
    const replayReducer = createConversationProjectionReducer({
      threadId: "thread-1",
    });
    replayReducer.dispatch({
      type: "turn_started",
      source: "replay",
      event_id: "thread-resume:turn:thread-1:turn-1",
      protocol_method: "thread/resume",
      turn: {
        id: "turn-1",
        thread_id: "thread-1",
        prompt_text: "继续",
        status: "running",
        started_at: "2026-07-06T00:00:00.000Z",
        created_at: "2026-07-06T00:00:00.000Z",
        updated_at: "2026-07-06T00:00:00.000Z",
      },
    });
    replayReducer.dispatch({
      type: "item_started",
      source: "replay",
      event_id: "thread-resume:item:thread-1:turn-1:message-1",
      protocol_method: "thread/resume",
      item: {
        id: "message-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        sequence: 2,
        status: "in_progress",
        started_at: "2026-07-06T00:00:00.000Z",
        updated_at: "2026-07-06T00:00:00.000Z",
        type: "agent_message",
        text: "恢复中的输出",
      },
    });
    const runtime = {
      listenToTurnEvents: vi.fn(async (_eventName, handler) => {
        eventHandler = handler;
        return unlisten;
      }),
      resumeThread: vi.fn(async (_threadId, consumeReplay) => {
        consumeReplay?.(replayReducer);
        return true;
      }),
    };
    const activeStreamState: { current: ActiveStreamState | null } = {
      current: null,
    };
    const listenerMapRef = { current: new Map<string, () => void>() };
    const messages = { current: [] as Message[] };
    const threadTurns = { current: [] as AgentThreadTurn[] };
    const threadItems = { current: [] as AgentThreadItem[] };
    const pendingActions = { current: [] as ActionRequired[] };
    const executionRuntime = {
      current: null as AgentSessionExecutionRuntime | null,
    };
    const currentTurnId = { current: null as string | null };
    let isSending = false;
    const setActiveStream = (nextActive: ActiveStreamState | null) => {
      activeStreamState.current = nextActive;
      isSending = Boolean(nextActive);
    };
    const setIsSending = (next: boolean | ((previous: boolean) => boolean)) => {
      isSending =
        typeof next === "function"
          ? (next as (previous: boolean) => boolean)(isSending)
          : next;
    };
    const clearActiveStreamIfMatch = (eventName: string) => {
      if (activeStreamState.current?.eventName !== eventName) {
        return false;
      }
      setActiveStream(null);
      return true;
    };

    await bindRecoveredAgentStreamThread({
      activeStreamRef: activeStreamState,
      appendThinkingToParts: (parts, textDelta) => [
        ...parts,
        { type: "thinking", text: textDelta },
      ],
      clearActiveStreamIfMatch,
      executionStrategy: "react",
      getMessages: () => messages.current,
      getThreadItems: () => threadItems.current,
      listenerMapRef,
      refreshSessionReadModel: vi.fn(async () => true),
      runtime,
      setActiveStream,
      setCurrentTurnId: createStateSetter(currentTurnId),
      setExecutionRuntime: createStateSetter(executionRuntime),
      setIsSending,
      setMessages: createStateSetter(messages),
      setPendingActions: createStateSetter(pendingActions),
      setThreadItems: createStateSetter(threadItems),
      setThreadTurns: createStateSetter(threadTurns),
      target: {
        eventName: "agentSession/event/session-1",
        sessionId: "session-1",
        threadId: "thread-1",
        turnId: "turn-1",
        startedAt: "2026-07-06T00:00:00.000Z",
      },
      warnedKeysRef: { current: new Set<string>() },
    });

    expect(activeStreamState.current).toMatchObject({
      eventName: "agentSession/event/session-1",
      sessionId: "session-1",
      turnId: "turn-1",
    });
    expect(isSending).toBe(true);
    expect(runtime.listenToTurnEvents).toHaveBeenCalledWith(
      "agentSession/event/session-1",
      expect.any(Function),
    );
    expect(runtime.resumeThread).toHaveBeenCalledWith(
      "thread-1",
      expect.any(Function),
    );
    expect(threadItems.current).toEqual([
      expect.objectContaining({
        id: "message-1",
        status: "in_progress",
        text: "恢复中的输出",
      }),
    ]);

    eventHandler?.({
      payload: {
        type: "thinking_delta",
        text: "不应恢复显示的 raw reasoning",
        session_id: "session-1",
        turn_id: "turn-1",
      },
    });
    eventHandler?.({
      payload: {
        type: "reasoning_summary_delta",
        text: "正在整理回答。",
        delta: "正在整理回答。",
        item_id: "reasoning-1",
        summary_index: 0,
        sequence: 1,
        session_id: "session-1",
        turn_id: "turn-1",
      },
    });
    eventHandler?.({
      payload: {
        type: "text_delta",
        text: "继续输出",
        delta: "继续输出",
        item_id: "message-1",
        phase: "final_answer",
        sequence: 2,
        session_id: "session-1",
        turn_id: "turn-1",
      },
    });
    eventHandler?.({
      payload: {
        type: "item_completed",
        protocol_method: "item/completed",
        item: {
          id: "message-1",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 99,
          status: "completed",
          started_at: "2026-07-06T00:00:09.000Z",
          completed_at: "2026-07-06T00:00:01.000Z",
          updated_at: "2026-07-06T00:00:01.000Z",
          type: "agent_message",
          text: "live 权威快照",
        },
      },
    });
    eventHandler?.({
      payload: {
        type: "turn_completed",
        protocol_method: "turn/completed",
        session_id: "session-1",
        turn_id: "turn-1",
        turn: {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "",
          status: "completed",
          started_at: "2026-07-06T00:00:00.000Z",
          completed_at: "2026-07-06T00:00:01.000Z",
          created_at: "2026-07-06T00:00:00.000Z",
          updated_at: "2026-07-06T00:00:01.000Z",
        },
      },
    });

    expect(messages.current[0]).toMatchObject({
      role: "assistant",
      content: "继续输出",
      isThinking: false,
      runtimeTurnId: "turn-1",
    });
    expect(messages.current[0]?.thinkingContent).toBe("正在整理回答。");
    expect(messages.current[0]?.contentParts).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "thinking",
          text: "不应恢复显示的 raw reasoning",
        }),
      ]),
    );
    expect(
      threadItems.current.find((item) => item.id === "message-1"),
    ).toMatchObject({
      id: "message-1",
      sequence: 2,
      started_at: "2026-07-06T00:00:00.000Z",
      status: "completed",
      text: "live 权威快照",
    });
    expect(activeStreamState.current).toBeNull();
    expect(isSending).toBe(false);
    expect(unlisten).toHaveBeenCalledTimes(1);
  });
});
