import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentEvent } from "@/lib/api/agentProtocol";
import type { AgentThreadItem, Message } from "../types";
import { clearAgentUiProjectionEvents } from "../projection/conversationProjectionStore";
import { handleTurnStreamEvent } from "./agentStreamRuntimeHandler";
import { clearAllAgentStreamTextOverlays } from "./agentStreamTextOverlayStore";
import { loadAgentSessionCachedSnapshot } from "./agentSessionScopedStorage";

const { mockToast } = vi.hoisted(() => ({
  mockToast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}));

vi.mock("sonner", () => ({
  toast: mockToast,
}));

describe("agentStreamRuntimeHandler storage", () => {
  afterEach(() => {
    vi.useRealTimers();
    mockToast.success.mockReset();
    mockToast.error.mockReset();
    mockToast.info.mockReset();
    mockToast.warning.mockReset();
    clearAgentUiProjectionEvents();
    clearAllAgentStreamTextOverlays();
    window.localStorage.clear();
    window.sessionStorage.clear();
  });

  it("Skill launch 即使未开启全局 thinking，也应保留流式思考证据", () => {
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-30T10:00:00.000Z"),
        isThinking: true,
        inlineProcessRetention: "skill",
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-skill-thinking-retain-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "/brand-product-knowledge-builder 生成资料",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "thinking_delta",
        text: "先读取 Skill 约束。",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "最终回复",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-1",
          thread_id: "session-1",
          prompt_text: "",
          status: "completed",
          started_at: "2026-06-12T00:00:00.000Z",
          completed_at: "2026-06-12T00:00:01.000Z",
          created_at: "2026-06-12T00:00:00.000Z",
          updated_at: "2026-06-12T00:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.thinkingContent).toBe("先读取 Skill 约束。");
    expect(messages[0]?.content).toBe("最终回复");
    expect(messages[0]?.isThinking).toBe(false);
    expect(messages[0]?.contentParts).toEqual([
      { type: "thinking", text: "先读取 Skill 约束。" },
      { type: "text", text: "最终回复" },
    ]);
    expect(
      loadAgentSessionCachedSnapshot("workspace-1", "session-1")?.messages[0],
    ).toMatchObject({
      content: "最终回复",
      thinkingContent: "先读取 Skill 约束。",
      inlineProcessRetention: "skill",
      contentParts: [
        { type: "thinking", text: "先读取 Skill 约束。" },
        { type: "text", text: "最终回复" },
      ],
    });
  });

  it("Codex reasoning v2 应分段显示 summary 并由 completed snapshot 单次接管", () => {
    let messages: Message[] = [
      {
        id: "assistant-reasoning-v2",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-23T02:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      currentTurnId: "turn-reasoning-v2",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
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
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-reasoning-v2-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-reasoning-v2",
      activeSessionId: "session-reasoning-v2",
      resolvedWorkspaceId: "workspace-reasoning-v2",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "验证 reasoning v2",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_summary_delta",
        itemId: "reasoning-v2",
        reasoningId: "reasoning-v2",
        summaryIndex: 0,
        text: "ha",
        delta: "ha",
        sequence: 2,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_summary_delta",
        itemId: "reasoning-v2",
        reasoningId: "reasoning-v2",
        summaryIndex: 0,
        text: "ha",
        delta: "ha",
        sequence: 3,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_summary_part_added",
        itemId: "reasoning-v2",
        reasoningId: "reasoning-v2",
        summaryIndex: 1,
        sequence: 4,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_summary_delta",
        itemId: "reasoning-v2",
        reasoningId: "reasoning-v2",
        summaryIndex: 1,
        text: "第二段",
        delta: "第二段",
        sequence: 5,
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_content_delta",
        itemId: "reasoning-v2",
        reasoningId: "reasoning-v2",
        contentIndex: 0,
        text: "raw chain of thought",
        delta: "raw chain of thought",
        sequence: 6,
      } as AgentEvent,
    });

    expect(messages[0]?.thinkingContent).toBe("haha\n\n第二段");
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "thinking",
        text: "haha",
        metadata: expect.objectContaining({
          source: "streamed_reasoning_summary",
          threadItemId: "reasoning-v2",
          summaryIndex: 0,
        }),
      },
      {
        type: "thinking",
        text: "第二段",
        metadata: expect.objectContaining({
          source: "streamed_reasoning_summary",
          threadItemId: "reasoning-v2",
          summaryIndex: 1,
        }),
      },
    ]);
    expect(threadItems).toEqual([
      expect.objectContaining({
        id: "streamed-reasoning:turn-reasoning-v2:reasoning-v2:summary:0",
        status: "completed",
        text: "haha",
      }),
      expect.objectContaining({
        id: "streamed-reasoning:turn-reasoning-v2:reasoning-v2:summary:1",
        status: "in_progress",
        text: "第二段",
      }),
    ]);

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_updated",
        item: {
          id: "reasoning-v2",
          thread_id: "session-reasoning-v2",
          turn_id: "turn-reasoning-v2",
          sequence: 2,
          status: "in_progress",
          started_at: "2026-07-23T02:00:00.000Z",
          updated_at: "2026-07-23T02:00:00.500Z",
          type: "reasoning",
          text: "haha\n\n第二段",
          summary: ["haha", "第二段"],
          content: [],
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "reasoning_summary_delta",
        itemId: "reasoning-v2",
        reasoningId: "reasoning-v2",
        summaryIndex: 1,
        text: "，继续",
        delta: "，继续",
        sequence: 7,
      } as AgentEvent,
    });

    expect(
      messages[0]?.contentParts?.filter((part) => part.type === "thinking"),
    ).toEqual([
      expect.objectContaining({
        type: "thinking",
        text: "haha\n\n第二段，继续",
        metadata: expect.objectContaining({
          threadItemId: "reasoning-v2",
          summaryIndex: 1,
        }),
      }),
    ]);

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_completed",
        item: {
          id: "reasoning-v2",
          thread_id: "session-reasoning-v2",
          turn_id: "turn-reasoning-v2",
          sequence: 2,
          status: "completed",
          started_at: "2026-07-23T02:00:00.000Z",
          completed_at: "2026-07-23T02:00:01.000Z",
          updated_at: "2026-07-23T02:00:01.000Z",
          type: "reasoning",
          text: "haha\n\n第二段，继续",
          summary: ["haha", "第二段，继续"],
          content: ["raw chain of thought"],
        },
      } as AgentEvent,
    });

    expect(messages[0]?.contentParts).toEqual([
      expect.objectContaining({
        type: "thinking",
        text: "haha\n\n第二段，继续",
        metadata: expect.objectContaining({
          source: "thread_item_reasoning",
          threadItemId: "reasoning-v2",
        }),
      }),
    ]);
    expect(messages[0]?.thinkingContent).not.toContain("raw chain of thought");
    expect(threadItems).toEqual([
      expect.objectContaining({
        id: "reasoning-v2",
        status: "completed",
        text: "haha\n\n第二段，继续",
      }),
    ]);
  });

  it("普通会话 item reasoning 应在 text_delta 和 turn_completed 后继续保留到消息过程", () => {
    let messages: Message[] = [
      {
        id: "assistant-web-tools",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-20T10:00:00.000Z"),
        isThinking: true,
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-web-search",
              name: "WebSearch",
              arguments: JSON.stringify({ query: "Lime WebSearch rendering" }),
              status: "completed",
              result: { success: true, output: "" },
              startTime: new Date("2026-06-20T10:00:00.100Z"),
              endTime: new Date("2026-06-20T10:00:00.200Z"),
            },
          },
        ],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-web-tools-reasoning-retain-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-web-tools",
      activeSessionId: "session-web-tools",
      resolvedWorkspaceId: "workspace-web-tools",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "验证网页搜索渲染",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: vi.fn() as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_updated",
        item: {
          id: "reasoning-web-tools",
          thread_id: "session-web-tools",
          turn_id: "turn-web-tools",
          sequence: 2,
          status: "in_progress",
          started_at: "2026-06-20T10:00:00.300Z",
          updated_at: "2026-06-20T10:00:00.300Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        event_id: "evt-final-web-tools",
        sequence: 3,
        session_id: "session-web-tools",
        thread_id: "session-web-tools",
        turn_id: "turn-web-tools",
        itemId: "final-web-tools",
        phase: "final_answer",
        text: "网页搜索渲染结论：最终正文继续输出。",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-web-tools",
          thread_id: "session-web-tools",
          prompt_text: "验证网页搜索渲染",
          status: "completed",
          started_at: "2026-06-20T10:00:00.000Z",
          completed_at: "2026-06-20T10:00:01.000Z",
          created_at: "2026-06-20T10:00:00.000Z",
          updated_at: "2026-06-20T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.content).toBe("网页搜索渲染结论：最终正文继续输出。");
    expect(messages[0]?.isThinking).toBe(false);
    expect(messages[0]?.thinkingContent).toBeUndefined();
    expect(messages[0]?.contentParts).toEqual([
      expect.objectContaining({
        type: "tool_use",
        toolCall: expect.objectContaining({
          id: "tool-web-search",
          status: "completed",
        }),
      }),
      expect.objectContaining({
        type: "thinking",
        text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        metadata: expect.objectContaining({
          source: "thread_item_reasoning",
          threadItemId: "reasoning-web-tools",
          sequence: 2,
          turnId: "turn-web-tools",
        }),
      }),
      expect.objectContaining({
        type: "text",
        text: "网页搜索渲染结论：最终正文继续输出。",
        metadata: expect.objectContaining({
          itemId: "final-web-tools",
          phase: "final_answer",
          sequence: 3,
          turnId: "turn-web-tools",
        }),
      }),
    ]);
  });

  it("WebSearch 与 WebFetch 之间的 reasoning 在 canonical Tool Item 完成后应保留顺序", () => {
    let messages: Message[] = [
      {
        id: "assistant-web-tools-full-chain",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-20T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn((value: unknown) => {
      threadItems =
        typeof value === "function"
          ? (value as (prev: AgentThreadItem[]) => AgentThreadItem[])(
              threadItems,
            )
          : (value as AgentThreadItem[]);
    });
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-web-tools-full-chain-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-web-tools-full-chain",
      activeSessionId: "session-web-tools-full-chain",
      resolvedWorkspaceId: "workspace-web-tools-full-chain",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "验证网页搜索渲染",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_started",
        turn: {
          id: "turn-web-tools-full-chain",
          thread_id: "session-web-tools-full-chain",
          prompt_text: "验证网页搜索渲染",
          status: "running",
          started_at: "2026-06-20T10:00:00.000Z",
          created_at: "2026-06-20T10:00:00.000Z",
          updated_at: "2026-06-20T10:00:00.000Z",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_completed",
        item: {
          id: "tool-web-search-full-chain",
          thread_id: "session-web-tools-full-chain",
          turn_id: "turn-web-tools-full-chain",
          sequence: 2,
          status: "completed",
          started_at: "2026-06-20T10:00:00.100Z",
          completed_at: "2026-06-20T10:00:00.200Z",
          updated_at: "2026-06-20T10:00:00.200Z",
          type: "tool_call",
          tool_name: "WebSearch",
          arguments: { query: "Lime WebSearch rendering" },
          output: JSON.stringify({
            results: [
              {
                title: "Lime WebSearch Rendering Source",
                url: "https://example.com/lime-websearch-rendering",
                snippet: "Search source used to verify inline rendering",
              },
            ],
          }),
          success: true,
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_updated",
        item: {
          id: "reasoning-web-tools-full-chain",
          thread_id: "session-web-tools-full-chain",
          turn_id: "turn-web-tools-full-chain",
          sequence: 3,
          status: "in_progress",
          started_at: "2026-06-20T10:00:00.300Z",
          updated_at: "2026-06-20T10:00:00.300Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_completed",
        item: {
          id: "tool-web-fetch-full-chain",
          thread_id: "session-web-tools-full-chain",
          turn_id: "turn-web-tools-full-chain",
          sequence: 4,
          status: "completed",
          started_at: "2026-06-20T10:00:00.500Z",
          completed_at: "2026-06-20T10:00:00.700Z",
          updated_at: "2026-06-20T10:00:00.700Z",
          type: "tool_call",
          tool_name: "WebFetch",
          arguments: {
            url: "https://example.com/lime-websearch-rendering",
          },
          output: JSON.stringify({
            bytes: 2048,
            code: 200,
            codeText: "OK",
            result: "WebFetch 正文摘要。",
          }),
          success: true,
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_completed",
        item: {
          id: "reasoning-web-tools-full-chain",
          thread_id: "session-web-tools-full-chain",
          turn_id: "turn-web-tools-full-chain",
          sequence: 3,
          status: "completed",
          started_at: "2026-06-20T10:00:00.300Z",
          completed_at: "2026-06-20T10:00:00.400Z",
          updated_at: "2026-06-20T10:00:00.400Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        text: "网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        turn: {
          id: "turn-web-tools-full-chain",
          thread_id: "session-web-tools-full-chain",
          prompt_text: "验证网页搜索渲染",
          status: "completed",
          started_at: "2026-06-20T10:00:00.000Z",
          completed_at: "2026-06-20T10:00:01.000Z",
          created_at: "2026-06-20T10:00:00.000Z",
          updated_at: "2026-06-20T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.isThinking).toBe(false);
    expect(messages[0]?.runtimeTurnId).toBe("turn-web-tools-full-chain");
    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "tool_use",
      "thinking",
      "tool_use",
      "text",
    ]);
    expect(messages[0]?.contentParts?.[1]).toMatchObject({
      type: "thinking",
      text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
      metadata: {
        source: "thread_item_reasoning",
        threadItemId: "reasoning-web-tools-full-chain",
        turnId: "turn-web-tools-full-chain",
      },
    });
  });

  it("工具完成事件不应覆盖开始序号，避免 WebSearch 中间 reasoning 被挤出过程组", () => {
    vi.useFakeTimers();

    let messages: Message[] = [
      {
        id: "assistant-web-tools-stable-sequence",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-20T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn((value: unknown) => {
      threadItems =
        typeof value === "function"
          ? (value as (prev: AgentThreadItem[]) => AgentThreadItem[])(
              threadItems,
            )
          : (value as AgentThreadItem[]);
    });
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-web-tools-stable-sequence-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-web-tools-stable-sequence",
      activeSessionId: "session-web-tools-stable-sequence",
      resolvedWorkspaceId: "workspace-web-tools-stable-sequence",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "验证网页搜索渲染",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
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
        event_id: "evt-intro",
        sequence: 1,
        session_id: "session-web-tools-stable-sequence",
        thread_id: "session-web-tools-stable-sequence",
        turn_id: "turn-web-tools-stable-sequence",
        text: "我先联网核实目标页面来源。\n",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        event_id: "evt-search-start",
        sequence: 2,
        session_id: "session-web-tools-stable-sequence",
        thread_id: "session-web-tools-stable-sequence",
        turn_id: "turn-web-tools-stable-sequence",
        tool_id: "tool-web-search-stable-sequence",
        tool_name: "WebSearch",
        arguments: JSON.stringify({ query: "Lime WebSearch rendering" }),
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        event_id: "evt-search-end",
        sequence: 3,
        session_id: "session-web-tools-stable-sequence",
        thread_id: "session-web-tools-stable-sequence",
        turn_id: "turn-web-tools-stable-sequence",
        tool_id: "tool-web-search-stable-sequence",
        result: {
          success: true,
          output: JSON.stringify({
            results: [
              {
                title: "Lime WebSearch Rendering Source",
                url: "https://example.com/lime-websearch-rendering",
                snippet: "Search source used to verify inline rendering",
              },
            ],
          }),
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_updated",
        event_id: "evt-reasoning-update",
        sequence: 4,
        session_id: "session-web-tools-stable-sequence",
        thread_id: "session-web-tools-stable-sequence",
        turn_id: "turn-web-tools-stable-sequence",
        item: {
          id: "reasoning-web-tools-stable-sequence",
          thread_id: "session-web-tools-stable-sequence",
          turn_id: "turn-web-tools-stable-sequence",
          sequence: 3,
          status: "in_progress",
          started_at: "2026-06-20T10:00:00.300Z",
          updated_at: "2026-06-20T10:00:00.300Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        event_id: "evt-fetch-start",
        sequence: 5,
        session_id: "session-web-tools-stable-sequence",
        thread_id: "session-web-tools-stable-sequence",
        turn_id: "turn-web-tools-stable-sequence",
        tool_id: "tool-web-fetch-stable-sequence",
        tool_name: "WebFetch",
        arguments: JSON.stringify({
          url: "https://example.com/lime-websearch-rendering",
        }),
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        event_id: "evt-fetch-end",
        sequence: 6,
        session_id: "session-web-tools-stable-sequence",
        thread_id: "session-web-tools-stable-sequence",
        turn_id: "turn-web-tools-stable-sequence",
        tool_id: "tool-web-fetch-stable-sequence",
        result: {
          success: true,
          output: JSON.stringify({
            bytes: 2048,
            code: 200,
            codeText: "OK",
            result: "WebFetch 正文摘要。",
          }),
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        event_id: "evt-final-delta",
        sequence: 7,
        session_id: "session-web-tools-stable-sequence",
        thread_id: "session-web-tools-stable-sequence",
        turn_id: "turn-web-tools-stable-sequence",
        itemId: "final-web-tools-stable-sequence",
        phase: "final_answer",
        text: "网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。",
      } as AgentEvent,
    });
    vi.runOnlyPendingTimers();
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        event_id: "evt-turn-completed",
        sequence: 8,
        session_id: "session-web-tools-stable-sequence",
        thread_id: "session-web-tools-stable-sequence",
        turn_id: "turn-web-tools-stable-sequence",
        turn: {
          id: "turn-web-tools-stable-sequence",
          thread_id: "session-web-tools-stable-sequence",
          prompt_text: "验证网页搜索渲染",
          status: "completed",
          started_at: "2026-06-20T10:00:00.000Z",
          completed_at: "2026-06-20T10:00:01.000Z",
          created_at: "2026-06-20T10:00:00.000Z",
          updated_at: "2026-06-20T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "text",
      "tool_use",
      "thinking",
      "tool_use",
      "text",
    ]);
    expect(messages[0]?.contentParts?.[1]).toMatchObject({
      type: "tool_use",
      metadata: { sequence: 2 },
    });
    expect(messages[0]?.contentParts?.[2]).toMatchObject({
      type: "thinking",
      text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
    });
    expect(messages[0]?.contentParts?.[3]).toMatchObject({
      type: "tool_use",
      metadata: { sequence: 5 },
    });
  });

  it("缺少 turn_started 时仍应按 item reasoning 的 turnId 保留 WebSearch/WebFetch 中间思考", () => {
    let messages: Message[] = [
      {
        id: "assistant-web-tools-no-turn-start",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-20T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn((value: unknown) => {
      threadItems =
        typeof value === "function"
          ? (value as (prev: AgentThreadItem[]) => AgentThreadItem[])(
              threadItems,
            )
          : (value as AgentThreadItem[]);
    });
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-web-tools-no-turn-start-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-web-tools-no-turn-start",
      activeSessionId: "session-web-tools-no-turn-start",
      resolvedWorkspaceId: "workspace-web-tools-no-turn-start",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "验证网页搜索渲染",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        event_id: "evt-search-start",
        session_id: "session-web-tools-no-turn-start",
        thread_id: "session-web-tools-no-turn-start",
        turn_id: "turn-web-tools-no-turn-start",
        tool_id: "tool-web-search-no-turn-start",
        tool_name: "WebSearch",
        arguments: JSON.stringify({ query: "Lime WebSearch rendering" }),
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        event_id: "evt-search-end",
        session_id: "session-web-tools-no-turn-start",
        thread_id: "session-web-tools-no-turn-start",
        turn_id: "turn-web-tools-no-turn-start",
        tool_id: "tool-web-search-no-turn-start",
        result: {
          success: true,
          output: JSON.stringify({
            results: [
              {
                title: "Lime WebSearch Rendering Source",
                url: "https://example.com/lime-websearch-rendering",
                snippet: "Search source used to verify inline rendering",
              },
            ],
          }),
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_updated",
        event_id: "evt-reasoning-update",
        session_id: "session-web-tools-no-turn-start",
        thread_id: "session-web-tools-no-turn-start",
        turn_id: "turn-web-tools-no-turn-start",
        item: {
          id: "reasoning-web-tools-no-turn-start",
          thread_id: "session-web-tools-no-turn-start",
          turn_id: "turn-web-tools-no-turn-start",
          sequence: 3,
          status: "in_progress",
          started_at: "2026-06-20T10:00:00.300Z",
          updated_at: "2026-06-20T10:00:00.300Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        event_id: "evt-fetch-start",
        session_id: "session-web-tools-no-turn-start",
        thread_id: "session-web-tools-no-turn-start",
        turn_id: "turn-web-tools-no-turn-start",
        tool_id: "tool-web-fetch-no-turn-start",
        tool_name: "WebFetch",
        arguments: JSON.stringify({
          url: "https://example.com/lime-websearch-rendering",
        }),
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        event_id: "evt-fetch-end",
        session_id: "session-web-tools-no-turn-start",
        thread_id: "session-web-tools-no-turn-start",
        turn_id: "turn-web-tools-no-turn-start",
        tool_id: "tool-web-fetch-no-turn-start",
        result: {
          success: true,
          output: JSON.stringify({
            bytes: 2048,
            code: 200,
            codeText: "OK",
            result: "WebFetch 正文摘要。",
          }),
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        event_id: "evt-final-delta",
        session_id: "session-web-tools-no-turn-start",
        thread_id: "session-web-tools-no-turn-start",
        turn_id: "turn-web-tools-no-turn-start",
        text: "网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。",
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        event_id: "evt-turn-completed",
        session_id: "session-web-tools-no-turn-start",
        thread_id: "session-web-tools-no-turn-start",
        turn_id: "turn-web-tools-no-turn-start",
        turn: {
          id: "turn-web-tools-no-turn-start",
          thread_id: "session-web-tools-no-turn-start",
          prompt_text: "验证网页搜索渲染",
          status: "completed",
          started_at: "2026-06-20T10:00:00.000Z",
          completed_at: "2026-06-20T10:00:01.000Z",
          created_at: "2026-06-20T10:00:00.000Z",
          updated_at: "2026-06-20T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.runtimeTurnId).toBe("turn-web-tools-no-turn-start");
    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "tool_use",
      "thinking",
      "tool_use",
      "text",
    ]);
  });

  it("工具事件缺少 sequence 时应使用 current thread item sequence 重排中间 reasoning", () => {
    vi.useFakeTimers();

    let messages: Message[] = [
      {
        id: "assistant-web-tools-item-sequence",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-20T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    let threadItems: AgentThreadItem[] = [];
    const setMessages = vi.fn(
      (value: Message[] | ((prev: Message[]) => Message[])) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );
    const setThreadItems = vi.fn((value: unknown) => {
      threadItems =
        typeof value === "function"
          ? (value as (prev: AgentThreadItem[]) => AgentThreadItem[])(
              threadItems,
            )
          : (value as AgentThreadItem[]);
    });
    const baseOptions = {
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      eventName: "agent-runtime-web-tools-item-sequence-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-web-tools-item-sequence",
      activeSessionId: "session-web-tools-item-sequence",
      resolvedWorkspaceId: "workspace-web-tools-item-sequence",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "验证网页搜索渲染",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: setMessages as never,
      setPendingActions: vi.fn() as never,
      getThreadItems: () => threadItems,
      setThreadItems: setThreadItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    };

    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        tool_id: "tool-web-search-item-sequence",
        tool_name: "WebSearch",
        arguments: JSON.stringify({ query: "Lime WebSearch rendering" }),
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        tool_id: "tool-web-search-item-sequence",
        result: {
          success: true,
          output: "search result",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_updated",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        item: {
          id: "reasoning-web-tools-item-sequence",
          thread_id: "session-web-tools-item-sequence",
          turn_id: "turn-web-tools-item-sequence",
          sequence: 3,
          status: "in_progress",
          started_at: "2026-06-20T10:00:00.300Z",
          updated_at: "2026-06-20T10:00:00.300Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_start",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        tool_id: "tool-web-fetch-item-sequence",
        tool_name: "WebFetch",
        arguments: JSON.stringify({
          url: "https://example.com/lime-websearch-rendering",
        }),
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "tool_end",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        tool_id: "tool-web-fetch-item-sequence",
        result: {
          success: true,
          output: "fetched page",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_completed",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        item: {
          id: "tool-web-search-item-sequence",
          thread_id: "session-web-tools-item-sequence",
          turn_id: "turn-web-tools-item-sequence",
          sequence: 2,
          status: "completed",
          started_at: "2026-06-20T10:00:00.100Z",
          completed_at: "2026-06-20T10:00:00.200Z",
          updated_at: "2026-06-20T10:00:00.200Z",
          type: "tool_call",
          tool_name: "WebSearch",
          arguments: { query: "Lime WebSearch rendering" },
          output: "search result",
          success: true,
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_completed",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        item: {
          id: "tool-web-fetch-item-sequence",
          thread_id: "session-web-tools-item-sequence",
          turn_id: "turn-web-tools-item-sequence",
          sequence: 4,
          status: "completed",
          started_at: "2026-06-20T10:00:00.500Z",
          completed_at: "2026-06-20T10:00:00.700Z",
          updated_at: "2026-06-20T10:00:00.700Z",
          type: "tool_call",
          tool_name: "WebFetch",
          arguments: { url: "https://example.com/lime-websearch-rendering" },
          output: "fetched page",
          success: true,
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "item_completed",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        item: {
          id: "reasoning-web-tools-item-sequence",
          thread_id: "session-web-tools-item-sequence",
          turn_id: "turn-web-tools-item-sequence",
          sequence: 3,
          status: "completed",
          started_at: "2026-06-20T10:00:00.300Z",
          completed_at: "2026-06-20T10:00:00.400Z",
          updated_at: "2026-06-20T10:00:00.400Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        },
      } as AgentEvent,
    });
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "text_delta",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        text: "网页搜索渲染结论：搜索来源已展开，读取页面已归入同一过程，最终正文继续输出。",
      } as AgentEvent,
    });
    vi.runOnlyPendingTimers();
    handleTurnStreamEvent({
      ...baseOptions,
      data: {
        type: "turn_completed",
        session_id: "session-web-tools-item-sequence",
        thread_id: "session-web-tools-item-sequence",
        turn_id: "turn-web-tools-item-sequence",
        turn: {
          id: "turn-web-tools-item-sequence",
          thread_id: "session-web-tools-item-sequence",
          prompt_text: "验证网页搜索渲染",
          status: "completed",
          started_at: "2026-06-20T10:00:00.000Z",
          completed_at: "2026-06-20T10:00:01.000Z",
          created_at: "2026-06-20T10:00:00.000Z",
          updated_at: "2026-06-20T10:00:01.000Z",
        },
      } as AgentEvent,
    });

    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "tool_use",
      "thinking",
      "tool_use",
      "text",
    ]);
    expect(messages[0]?.contentParts?.[0]).toMatchObject({
      type: "tool_use",
      metadata: { sequence: 2 },
    });
    expect(messages[0]?.contentParts?.[1]).toMatchObject({
      type: "thinking",
      metadata: {
        sequence: 3,
        threadItemId: "reasoning-web-tools-item-sequence",
      },
    });
    expect(messages[0]?.contentParts?.[2]).toMatchObject({
      type: "tool_use",
      metadata: { sequence: 4 },
    });
  });

  it("item_completed 的 canonical imageView media item 应同步到当前 assistant 消息", () => {
    let messages: Message[] = [
      {
        id: "assistant-media-reference",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T10:00:00.000Z"),
        isThinking: true,
        contentParts: [],
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
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

    handleTurnStreamEvent({
      requestState,
      callbacks: {
        activateStream: vi.fn(),
        isStreamActivated: () => true,
        clearOptimisticItem: () => {},
        clearOptimisticTurn: () => {},
        disposeListener: () => {},
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (
          parts: NonNullable<Message["contentParts"]>,
          textDelta: string,
        ) => [...parts, { type: "thinking" as const, text: textDelta }],
      },
      data: {
        type: "item_completed",
        session_id: "session-media-reference",
        thread_id: "session-media-reference",
        turn_id: "turn-media-reference",
        item: {
          id: "agent-media-reference-1",
          thread_id: "session-media-reference",
          turn_id: "turn-media-reference",
          sequence: 5,
          status: "completed",
          started_at: "2026-07-07T10:00:00.100Z",
          completed_at: "2026-07-07T10:00:00.200Z",
          updated_at: "2026-07-07T10:00:00.200Z",
          type: "media",
          uri: "/tmp/lime-media/fixture-image-1.png",
          mime_type: "image/png",
        },
      } as AgentEvent,
      eventName: "agent-runtime-media-reference-content-parts-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-media-reference",
      activeSessionId: "session-media-reference",
      resolvedWorkspaceId: "workspace-media-reference",
      effectiveExecutionStrategy: "react" as const,
      surfaceThinkingDeltas: false,
      content: "验证媒体引用展示",
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
    });

    expect(messages[0]?.runtimeTurnId).toBe("turn-media-reference");
    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "media_reference",
    ]);
    expect(messages[0]?.contentParts?.[0]).toMatchObject({
      type: "media_reference",
      reference: {
        uri: "/tmp/lime-media/fixture-image-1.png",
        mimeType: "image/png",
        title: "fixture-image-1.png",
        sourcePath: "/tmp/lime-media/fixture-image-1.png",
      },
      metadata: {
        source: "agent_media_reference",
        threadItemId: "agent-media-reference-1",
        turnId: "turn-media-reference",
        sequence: 5,
        sourcePath: "/tmp/lime-media/fixture-image-1.png",
      },
    });
  });
});
