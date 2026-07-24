import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import {
  clearAgentUiPerformanceMetrics,
  getAgentUiPerformanceMetrics,
} from "@/lib/agentUiPerformanceMetrics";
import type { Message } from "../types";
import { handleAgentStreamThreadItemLifecycleEvent } from "./agentStreamRuntimeLifecycleEvents";
import { handleTurnStreamEvent } from "./agentStreamRuntimeHandler";

describe("agentStreamRuntimeLifecycleEvents", () => {
  afterEach(() => {
    clearAgentUiPerformanceMetrics();
    vi.restoreAllMocks();
  });

  it("运行中的 canonical AgentMessage snapshot 应立即同步到同一 commentary part", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-18T12:15:54.122Z"),
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const item: AgentThreadItem = {
      id: "agent-message-commentary",
      thread_id: "thread-1",
      turn_id: "turn-1",
      type: "agent_message",
      status: "in_progress",
      ordinal: 103,
      sequence: 120,
      text: "先核对仓库状态。",
      phase: "commentary",
      started_at: "2026-07-18T12:15:54.122Z",
      updated_at: "2026-07-18T12:15:56.000Z",
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn(
      (
        value:
          | AgentThreadItem[]
          | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
      ) => {
        threadItems = typeof value === "function" ? value(threadItems) : value;
      },
    );

    handleAgentStreamThreadItemLifecycleEvent({
      assistantMsgId: "assistant-1",
      event: { type: "item_updated", item },
      pendingItemKey: "pending-item",
      requestState: {
        accumulatedContent: "",
        requestLogId: null,
        requestStartedAt: 0,
        requestFinished: false,
      },
      setters: {
        getThreadItems: () => threadItems,
        setCurrentTurnId: vi.fn(),
        setMessages: setMessages as never,
        setPendingActions: vi.fn() as never,
        setThreadItems: setThreadItems as never,
        setThreadTurns: vi.fn() as never,
      },
    });

    expect(threadItems).toEqual([expect.objectContaining(item)]);
    expect(messages[0]).toMatchObject({
      runtimeTurnId: "turn-1",
      contentParts: [
        {
          type: "text",
          text: "先核对仓库状态。",
          metadata: expect.objectContaining({
            itemId: "agent-message-commentary",
            phase: "commentary",
            sequence: 103,
          }),
        },
      ],
    });
  });

  it("final_answer 前缀 snapshot 应替换累计正文并记录显式最终段", () => {
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      hasFinalAnswerRequiredProcessBoundary: true,
      hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary: false,
      maxFinalAnswerRequiredProcessEventSequence: 40,
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setters = {
      getThreadItems: () => threadItems,
      setCurrentTurnId: vi.fn(),
      setMessages: vi.fn((value: unknown) => value) as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn(
        (
          value:
            | AgentThreadItem[]
            | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
        ) => {
          threadItems =
            typeof value === "function" ? value(threadItems) : value;
        },
      ) as never,
      setThreadTurns: vi.fn() as never,
    };
    const snapshot = (text: string, sequence: number): AgentThreadItem => ({
      id: "agent-message-final",
      thread_id: "thread-1",
      turn_id: "turn-1",
      type: "agent_message",
      status: "in_progress",
      ordinal: 50,
      sequence,
      text,
      phase: "final_answer",
      started_at: "2026-07-18T12:15:54.122Z",
      updated_at: "2026-07-18T12:15:56.000Z",
    });

    handleAgentStreamThreadItemLifecycleEvent({
      assistantMsgId: "assistant-1",
      event: { type: "item_updated", item: snapshot("最终", 50) },
      pendingItemKey: "pending-item",
      requestState,
      setters,
    });
    handleAgentStreamThreadItemLifecycleEvent({
      assistantMsgId: "assistant-1",
      event: { type: "item_updated", item: snapshot("最终答复", 51) },
      pendingItemKey: "pending-item",
      requestState,
      setters,
    });

    expect(requestState).toMatchObject({
      accumulatedContent: "最终答复",
      renderedContent: "最终答复",
      activeTextSegmentItemId: "agent-message-final",
      activeTextSegmentPhase: "final_answer",
      activeTextSegmentSequence: 51,
      activeTextSegmentTurnId: "turn-1",
      activeTextSegmentStartOffset: 0,
      activeTextSegmentFinalEligibility: "explicit_final",
      latestAssistantTextEventSequence: 51,
      hasAssistantTextAfterLatestFinalAnswerRequiredProcessBoundary: true,
    });
  });

  it("commentary snapshot 不应污染最终累计正文", () => {
    const requestState = {
      accumulatedContent: "已有最终正文",
      renderedContent: "已有最终正文",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };

    handleAgentStreamThreadItemLifecycleEvent({
      assistantMsgId: "assistant-1",
      event: {
        type: "item_updated",
        item: {
          id: "agent-message-commentary",
          thread_id: "thread-1",
          turn_id: "turn-1",
          type: "agent_message",
          status: "in_progress",
          ordinal: 60,
          sequence: 60,
          text: "继续检查工具输出。",
          phase: "commentary",
          started_at: "2026-07-18T12:15:54.122Z",
          updated_at: "2026-07-18T12:15:56.000Z",
        },
      },
      pendingItemKey: "pending-item",
      requestState,
      setters: {
        getThreadItems: () => [],
        setCurrentTurnId: vi.fn(),
        setMessages: vi.fn((value: unknown) => value) as never,
        setPendingActions: vi.fn() as never,
        setThreadItems: vi.fn((value: unknown) => value) as never,
        setThreadTurns: vi.fn() as never,
      },
    });

    expect(requestState.accumulatedContent).toBe("已有最终正文");
    expect(requestState.renderedContent).toBe("已有最终正文");
  });

  it("canonical AgentMessage snapshot 应记录首个可见文本与首帧指标", () => {
    const requestAnimationFrame = vi
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((callback) => {
        callback(performance.now());
        return 1;
      });
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-18T12:15:54.122Z"),
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const item: AgentThreadItem = {
      id: "agent-message-commentary",
      thread_id: "thread-1",
      turn_id: "turn-1",
      type: "agent_message",
      status: "in_progress",
      ordinal: 103,
      sequence: 120,
      text: "先核对仓库状态。",
      phase: "commentary",
      started_at: "2026-07-18T12:15:54.122Z",
      updated_at: "2026-07-18T12:15:56.000Z",
    };
    const setMessages = (
      value: Message[] | ((prev: Message[]) => Message[]),
    ) => {
      messages = typeof value === "function" ? value(messages) : value;
    };
    const setThreadItems = (
      value:
        | AgentThreadItem[]
        | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
    ) => {
      threadItems = typeof value === "function" ? value(threadItems) : value;
    };

    handleTurnStreamEvent({
      data: {
        type: "item_updated",
        item,
        renderer_event_received_at: 190,
        server_event_emitted_at: 180,
      },
      requestState: {
        accumulatedContent: "",
        firstEventReceivedAt: 170,
        performanceTrace: {
          rendererEventReceivedAt: 190,
          serverEventEmittedAt: 180,
          sessionId: "session-1",
        },
        requestLogId: null,
        requestStartedAt: 100,
        requestFinished: false,
      },
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: vi.fn(),
        clearOptimisticTurn: vi.fn(),
        disposeListener: vi.fn(),
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-stream-snapshot-metrics",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "核对仓库",
      runtime: {} as never,
      warnedKeysRef: { current: new Set() },
      actionLoggedKeys: new Set(),
      toolLogIdByToolId: new Map(),
      toolStartedAtByToolId: new Map(),
      toolNameByToolId: new Map(),
      getThreadItems: () => threadItems,
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    const entries = getAgentUiPerformanceMetrics();
    expect(entries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          phase: "agentStream.firstTextDelta",
          metrics: expect.objectContaining({
            source: "agent_message_snapshot",
          }),
        }),
        expect.objectContaining({ phase: "agentStream.firstTextPaint" }),
      ]),
    );
    expect(requestAnimationFrame).toHaveBeenCalledTimes(2);
  });

  it("首字绘制指标不应被后续事件的 trace 覆盖", () => {
    const pendingFrames: Array<(timestamp: number) => void> = [];
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      pendingFrames.push(callback);
      return pendingFrames.length;
    });
    const requestState = {
      accumulatedContent: "",
      firstEventReceivedAt: 170,
      performanceTrace: {
        rendererEventReceivedAt: 200,
        serverEventEmittedAt: 180,
        sessionId: "session-1",
      },
      requestLogId: null,
      requestStartedAt: 100,
      requestFinished: false,
    };

    handleTurnStreamEvent({
      data: {
        type: "item_updated",
        item: {
          id: "agent-message-final",
          thread_id: "thread-1",
          turn_id: "turn-1",
          type: "agent_message",
          status: "in_progress",
          ordinal: 1,
          sequence: 1,
          text: "你好。",
          phase: "final_answer",
          started_at: "2026-07-23T00:00:00.000Z",
          updated_at: "2026-07-23T00:00:00.100Z",
        },
      },
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: vi.fn(),
        clearOptimisticTurn: vi.fn(),
        disposeListener: vi.fn(),
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "agent-stream-first-text-trace",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "你好",
      runtime: {} as never,
      warnedKeysRef: { current: new Set() },
      actionLoggedKeys: new Set(),
      toolLogIdByToolId: new Map(),
      toolStartedAtByToolId: new Map(),
      toolNameByToolId: new Map(),
      getThreadItems: () => [],
      setMessages: vi.fn() as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    requestState.performanceTrace = {
      rendererEventReceivedAt: 900,
      serverEventEmittedAt: 900,
      sessionId: "session-1",
    };
    pendingFrames.shift()?.(performance.now());
    pendingFrames.shift()?.(performance.now());

    expect(
      getAgentUiPerformanceMetrics().find(
        (entry) => entry.phase === "agentStream.firstTextPaint",
      )?.metrics,
    ).toMatchObject({
      serverEventEmittedAt: 180,
      rendererEventReceivedAt: 200,
      serverToRendererDeltaMs: 20,
    });
  });

  it("晚到的较短 final snapshot 不应截断已累计的连续 delta", () => {
    const firstText = "以下是今日国际新闻简要整理：";
    const secondText =
      "全球市场继续关注能源与供应链变化，国际组织呼吁加强协调。CLAW_NEWS_FIXTURE_DONE";
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: firstText,
        timestamp: new Date("2026-07-19T20:57:30.149Z"),
        contentParts: [
          {
            type: "text",
            text: firstText,
            metadata: {
              itemId: "agent-message-final-turn-1",
              phase: "final_answer",
            },
          },
        ],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = (
      value: Message[] | ((prev: Message[]) => Message[]),
    ) => {
      messages = typeof value === "function" ? value(messages) : value;
    };
    const setThreadItems = (
      value:
        | AgentThreadItem[]
        | ((prev: AgentThreadItem[]) => AgentThreadItem[]),
    ) => {
      threadItems = typeof value === "function" ? value(threadItems) : value;
    };
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: vi.fn(),
        clearOptimisticTurn: vi.fn(),
        disposeListener: vi.fn(),
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts: NonNullable<Message["contentParts"]>) =>
          parts,
      },
      eventName: "agent-stream-final-snapshot-prefix",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      content: "整理今天的国际新闻",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      getThreadItems: () => threadItems,
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        itemId: "agent-message-final-turn-1",
        turn_id: "turn-1",
        text: firstText,
      },
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        itemId: "agent-message-final-turn-1",
        turn_id: "turn-1",
        text: secondText,
      },
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_completed",
        item: {
          id: "agent-message-final-turn-1",
          thread_id: "thread-1",
          turn_id: "turn-1",
          type: "agent_message",
          status: "completed",
          ordinal: 1,
          sequence: 3,
          text: firstText,
          phase: "final_answer",
          started_at: "2026-07-19T20:57:30.149Z",
          completed_at: "2026-07-19T20:57:30.425Z",
          updated_at: "2026-07-19T20:57:30.425Z",
        },
      },
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "整理今天的国际新闻",
          status: "completed",
          started_at: "2026-07-19T20:57:30.000Z",
          completed_at: "2026-07-19T20:57:30.425Z",
          created_at: "2026-07-19T20:57:30.000Z",
          updated_at: "2026-07-19T20:57:30.425Z",
        },
      },
    });

    expect(messages[0]?.content).toBe(`${firstText}${secondText}`);
    expect(
      messages[0]?.contentParts
        ?.filter((part) => part.type === "text")
        .map((part) => part.text)
        .join(""),
    ).toBe(`${firstText}${secondText}`);
  });
});
