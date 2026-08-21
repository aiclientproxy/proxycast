import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentThreadItem, AgentThreadTurn } from "@/lib/api/agentProtocol";
import type { Message } from "../types";
import type { ActiveStreamState } from "./agentStreamSubmissionLifecycle";
import {
  buildInterruptedMessageContentPatch,
  settleInterruptedMessageProcess,
  stopActiveAgentStream,
} from "./agentStreamFlowControl";
import {
  clearAllAgentStreamTextOverlays,
  getAgentStreamTextOverlay,
  upsertAgentStreamTextOverlay,
} from "./agentStreamTextOverlayStore";

function createStateSetter<T>(getValue: () => T, setValue: (value: T) => void) {
  return (next: T | ((prev: T) => T)) => {
    setValue(
      typeof next === "function" ? (next as (prev: T) => T)(getValue()) : next,
    );
  };
}

async function flushMicrotasks(times = 3) {
  for (let index = 0; index < times; index += 1) {
    await Promise.resolve();
  }
}

describe("agentStreamFlowControl", () => {
  afterEach(() => {
    clearAllAgentStreamTextOverlays();
  });

  it("stopActiveAgentStream 应清理 optimistic 状态并刷新 read model", async () => {
    let threadItems: AgentThreadItem[] = [
      {
        id: "pending-item:1",
        thread_id: "session-1",
        turn_id: "pending-turn:1",
        sequence: 0,
        status: "in_progress",
        started_at: "2026-03-29T00:00:00.000Z",
        updated_at: "2026-03-29T00:00:00.000Z",
        type: "turn_summary",
        text: "running",
      },
    ];
    let threadTurns: AgentThreadTurn[] = [
      {
        id: "pending-turn:1",
        thread_id: "session-1",
        prompt_text: "继续执行",
        status: "running",
        started_at: "2026-03-29T00:00:00.000Z",
        created_at: "2026-03-29T00:00:00.000Z",
        updated_at: "2026-03-29T00:00:00.000Z",
      },
    ];
    let currentTurnId: string | null = "pending-turn:1";
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-03-29T00:00:00.000Z"),
        isThinking: true,
        toolCalls: [
          {
            id: "tool-running-1",
            name: "Bash",
            arguments: JSON.stringify({ command: "python3 news.py" }),
            status: "running",
            startTime: new Date("2026-03-29T00:00:01.000Z"),
          },
        ],
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-running-1",
              name: "Bash",
              arguments: JSON.stringify({ command: "python3 news.py" }),
              status: "running",
              startTime: new Date("2026-03-29T00:00:01.000Z"),
            },
          },
        ],
      },
    ];
    let activeStream = {
      assistantMsgId: "assistant-1",
      eventName: "stream-1",
      sessionId: "session-1",
      turnId: "turn-runtime-1",
      pendingTurnKey: "pending-turn:1",
      pendingItemKey: "pending-item:1",
    };
    const removeStreamListener = vi.fn();
    const interruptTurn = vi.fn(async () => true);
    const refreshSessionReadModel = vi.fn(async () => true);
    const notify = {
      info: vi.fn(),
      error: vi.fn(),
    };

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-1" },
      threadId: "thread-1",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn,
      } as never,
      removeStreamListener,
      refreshSessionReadModel,
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => threadTurns,
        (value) => {
          threadTurns = value;
        },
      ),
      setCurrentTurnId: createStateSetter(
        () => currentTurnId,
        (value) => {
          currentTurnId = value;
        },
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setActiveStream: (next) => {
        activeStream = next as never;
      },
      notify,
    });

    expect(removeStreamListener).toHaveBeenCalledWith("stream-1");
    expect(interruptTurn).toHaveBeenCalledWith(
      "thread-1",
      "turn-runtime-1",
      "stream-1",
    );
    await Promise.resolve();
    await Promise.resolve();
    expect(refreshSessionReadModel).toHaveBeenCalledWith("session-1");
    expect(threadItems).toEqual([]);
    expect(threadTurns).toEqual([]);
    expect(currentTurnId).toBeNull();
    expect(messages[0]?.content).toBe("(已停止)");
    expect(messages[0]?.isThinking).toBe(false);
    expect(messages[0]?.toolCalls?.[0]).toMatchObject({
      status: "failed",
      result: {
        success: false,
        output: "",
        error: "本轮已中止",
      },
    });
    const interruptedToolPart = messages[0]?.contentParts?.[0];
    expect(interruptedToolPart?.type).toBe("tool_use");
    if (interruptedToolPart?.type === "tool_use") {
      expect(interruptedToolPart.toolCall.status).toBe("failed");
      expect(interruptedToolPart.toolCall.result?.error).toBe("本轮已中止");
    }
    expect(activeStream).toBeNull();
    expect(notify.info).toHaveBeenCalledWith("已停止生成");
  });

  it("stopActiveAgentStream 应以 current turn 兜底发送 interrupt", async () => {
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "assistant-current-turn",
      eventName: "stream-current-turn",
      sessionId: "session-current-turn",
      pendingItemKey: "pending-item:current-turn",
      pendingTurnKey: "pending-turn:current-turn",
    };
    const interruptTurn = vi.fn(async () => true);

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-current-turn" },
      threadId: "thread-current-turn",
      currentTurnId: "turn-current",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn,
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => [] as AgentThreadItem[],
        () => undefined,
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => "turn-current" as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => [] as Message[],
        () => undefined,
      ),
      getMessages: () => [],
      getThreadItems: () => [],
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    await flushMicrotasks();
    expect(interruptTurn).toHaveBeenCalledWith(
      "thread-current-turn",
      "turn-current",
      "stream-current-turn",
    );
  });

  it("stopActiveAgentStream 应在 reload binding 已释放时按 current turn 补回正文", async () => {
    let messages: Message[] = [];
    let threadItems: AgentThreadItem[] = [];
    const canonicalThreadItems: AgentThreadItem[] = [
      {
        id: "agent-message-final-turn-reload",
        type: "agent_message",
        thread_id: "thread-reload",
        turn_id: "turn-reload",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-07-10T00:00:01.000Z",
        updated_at: "2026-07-10T00:00:01.000Z",
        text: "以下是今日国际新闻简要整理：",
        phase: "final_answer",
      },
    ];

    await stopActiveAgentStream({
      activeStream: null,
      sessionIdRef: { current: "session-reload" },
      threadId: "thread-reload",
      currentTurnId: "turn-reload",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({
          thread_items: canonicalThreadItems,
        })),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => "turn-reload" as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => threadItems,
      setActiveStream: vi.fn(),
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    await flushMicrotasks();
    expect(messages[0]).toMatchObject({
      role: "assistant",
      runtimeTurnId: "turn-reload",
      content: "以下是今日国际新闻简要整理：\n\n(已停止)",
    });
  });

  it("stopActiveAgentStream 应从恢复消息的 text parts 保留 partial 正文", async () => {
    let messages: Message[] = [
      {
        id: "recovered-assistant:turn-parts",
        role: "assistant",
        content: "(已停止)",
        contentParts: [
          { type: "text", text: "以下是今日国际新闻简要整理：" },
          { type: "text", text: "(已停止)" },
        ],
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
        runtimeTurnId: "turn-parts",
      },
    ];
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "recovered-assistant:turn-parts",
      eventName: "agentSession/event/session-parts",
      sessionId: "session-parts",
      turnId: "turn-parts",
    };

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-parts" },
      threadId: "thread-parts",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: vi.fn(),
      setThreadTurns: vi.fn(),
      setCurrentTurnId: vi.fn(),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => [],
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(messages[0]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
    expect(activeStream).toBeNull();
  });

  it("stopActiveAgentStream 不应等待 cancel 后端返回才解除 UI 停止态", async () => {
    let resolveInterrupt!: (value: boolean) => void;
    const interruptPromise = new Promise<boolean>((resolve) => {
      resolveInterrupt = resolve;
    });
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "assistant-1",
      eventName: "stream-slow-cancel",
      sessionId: "session-1",
      turnId: "turn-runtime-1",
      pendingTurnKey: "pending-turn:1",
      pendingItemKey: "pending-item:1",
    };
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-03-29T00:00:00.000Z"),
        isThinking: true,
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    let threadTurns: AgentThreadTurn[] = [];
    let currentTurnId: string | null = null;
    const interruptTurn = vi.fn(() => interruptPromise);
    const refreshSessionReadModel = vi.fn(async () => true);
    const notify = {
      info: vi.fn(),
      error: vi.fn(),
    };

    const stopPromise = stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-1" },
      threadId: "thread-1",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn,
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel,
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => threadTurns,
        (value) => {
          threadTurns = value;
        },
      ),
      setCurrentTurnId: createStateSetter(
        () => currentTurnId,
        (value) => {
          currentTurnId = value;
        },
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify,
    });

    await stopPromise;

    expect(interruptTurn).toHaveBeenCalledWith(
      "thread-1",
      "turn-runtime-1",
      "stream-slow-cancel",
    );
    expect(activeStream).toBeNull();
    expect(messages[0]?.content).toBe("(已停止)");
    expect(messages[0]?.isThinking).toBe(false);
    expect(refreshSessionReadModel).not.toHaveBeenCalled();
    expect(notify.info).toHaveBeenCalledWith("已停止生成");

    resolveInterrupt(true);
    await Promise.resolve();
    await Promise.resolve();

    expect(refreshSessionReadModel).toHaveBeenCalledWith("session-1");
  });

  it("stopActiveAgentStream 对已有部分输出也应追加已停止终态", async () => {
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "assistant-partial",
      eventName: "stream-partial-cancel",
      sessionId: "session-1",
      turnId: "turn-runtime-1",
      pendingTurnKey: "pending-turn:1",
      pendingItemKey: "pending-item:1",
    };
    let messages: Message[] = [
      {
        id: "assistant-partial",
        role: "assistant",
        content: "以下是今日国际新闻简要整理：",
        contentParts: [
          {
            type: "text",
            text: "以下是今日国际新闻简要整理：",
          },
        ],
        timestamp: new Date("2026-03-29T00:00:00.000Z"),
        isThinking: true,
      },
    ];
    let threadItems: AgentThreadItem[] = [];

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-1" },
      threadId: "thread-1",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => null as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => threadItems,
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(messages[0]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "text",
        text: "以下是今日国际新闻简要整理：",
      },
      {
        type: "text",
        text: "(已停止)",
      },
    ]);
    expect(messages[0]?.isThinking).toBe(false);
    expect(activeStream).toBeNull();
  });

  it("stopActiveAgentStream 应提交并清理同一 assistant 的流式正文 overlay", async () => {
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "assistant-overlay-partial",
      eventName: "stream-overlay-cancel",
      sessionId: "session-1",
      turnId: "turn-runtime-1",
      pendingTurnKey: "pending-turn:1",
      pendingItemKey: "pending-item:1",
    };
    let messages: Message[] = [
      {
        id: "assistant-overlay-partial",
        role: "assistant",
        content: "",
        contentParts: [],
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
        isThinking: true,
      },
    ];
    let threadItems: AgentThreadItem[] = [];

    upsertAgentStreamTextOverlay({
      messageId: "assistant-overlay-partial",
      eventName: "stream-overlay-cancel",
      content: "以下是今日国际新闻简要整理：",
      phase: "final_answer",
    });

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-1" },
      threadId: "thread-1",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => null as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => threadItems,
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(messages[0]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "text",
        text: "以下是今日国际新闻简要整理：",
      },
      {
        type: "text",
        text: "(已停止)",
      },
    ]);
    expect(threadItems[0]).toMatchObject({
      type: "agent_message",
      turn_id: "turn-runtime-1",
      text: "以下是今日国际新闻简要整理：\n\n(已停止)",
    });
    expect(getAgentStreamTextOverlay("assistant-overlay-partial")).toBeNull();
  });

  it("stopActiveAgentStream 应从同一 runtime turn 的 canonical item 保留正文", async () => {
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "recovered-assistant:turn-runtime-1",
      eventName: "agentSession/event/session-1",
      sessionId: "session-1",
      turnId: "turn-runtime-1",
      pendingTurnKey: "turn-runtime-1",
      pendingItemKey: "recovered-turn-summary:turn-runtime-1",
    };
    let messages: Message[] = [
      {
        id: "recovered-assistant:turn-runtime-1",
        role: "assistant",
        content: "",
        contentParts: [],
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
        isThinking: true,
        runtimeTurnId: "turn-runtime-1",
      },
    ];
    const threadItems: AgentThreadItem[] = [
      {
        id: "agent-message-final-turn-runtime-1",
        type: "agent_message",
        thread_id: "thread-1",
        turn_id: "turn-runtime-1",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-07-10T00:00:01.000Z",
        updated_at: "2026-07-10T00:00:01.000Z",
        text: "以下是今日国际新闻简要整理：",
        phase: "final_answer",
      },
    ];

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-1" },
      threadId: "thread-1",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems.splice(0, threadItems.length, ...value);
        },
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => "turn-runtime-1" as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => threadItems,
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(messages[0]).toMatchObject({
      id: "recovered-assistant:turn-runtime-1",
      content: "以下是今日国际新闻简要整理：\n\n(已停止)",
      contentParts: [
        { type: "text", text: "以下是今日国际新闻简要整理：" },
        { type: "text", text: "(已停止)" },
      ],
      isThinking: false,
      runtimeTurnId: "turn-runtime-1",
    });
  });

  it("stopActiveAgentStream 应保留缺少 thread_id 的同 turn 恢复正文", async () => {
    let messages: Message[] = [
      {
        id: "recovered-assistant:turn-runtime-missing-thread",
        role: "assistant",
        content: "",
        contentParts: [],
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
        isThinking: true,
        runtimeTurnId: "turn-runtime-missing-thread",
      },
    ];
    const threadItems: AgentThreadItem[] = [
      {
        id: "agent-message-final-turn-runtime-missing-thread",
        type: "agent_message",
        thread_id: "",
        turn_id: "turn-runtime-missing-thread",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-07-10T00:00:01.000Z",
        updated_at: "2026-07-10T00:00:01.000Z",
        text: "以下是今日国际新闻简要整理：",
        phase: "final_answer",
      },
    ];

    await stopActiveAgentStream({
      activeStream: {
        assistantMsgId: "recovered-assistant:turn-runtime-missing-thread",
        eventName: "agentSession/event/session-missing-thread",
        sessionId: "session-missing-thread",
        turnId: "turn-runtime-missing-thread",
      },
      sessionIdRef: { current: "session-missing-thread" },
      threadId: "thread-missing-thread",
      currentTurnId: "turn-runtime-missing-thread",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems.splice(0, threadItems.length, ...value);
        },
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => "turn-runtime-missing-thread" as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => threadItems,
      setActiveStream: vi.fn(),
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(messages[0]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
  });

  it("stopActiveAgentStream 应在 read model 刷新后补回 canonical partial 正文", async () => {
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "recovered-assistant:turn-runtime-refresh",
      eventName: "agentSession/event/session-refresh",
      sessionId: "session-refresh",
      turnId: "turn-runtime-refresh",
    };
    let messages: Message[] = [
      {
        id: "recovered-assistant:turn-runtime-refresh",
        role: "assistant",
        content: "(已停止)",
        contentParts: [{ type: "text", text: "(已停止)" }],
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
        runtimeTurnId: "turn-runtime-refresh",
      },
    ];
    let threadItems: AgentThreadItem[] = [];
    const canonicalThreadItems: AgentThreadItem[] = [
      {
        id: "agent-message-final-turn-runtime-refresh",
        type: "agent_message",
        thread_id: "thread-refresh",
        turn_id: "turn-runtime-refresh",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-07-10T00:00:01.000Z",
        updated_at: "2026-07-10T00:00:01.000Z",
        text: "以下是今日国际新闻简要整理：",
        phase: "final_answer",
      },
    ];
    const refreshSessionReadModel = vi.fn(async () => true);
    const getSessionReadModel = vi.fn(async () => ({
      thread_items: canonicalThreadItems,
    }));

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-refresh" },
      threadId: "thread-refresh",
      runtime: {
        getSessionReadModel,
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel,
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => null as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => threadItems,
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    await flushMicrotasks();
    expect(refreshSessionReadModel).toHaveBeenCalledWith("session-refresh");
    expect(getSessionReadModel).toHaveBeenCalledWith("session-refresh");
    expect(messages[0]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
    expect(threadItems[0]).toMatchObject({
      type: "agent_message",
      turn_id: "turn-runtime-refresh",
      text: "以下是今日国际新闻简要整理：\n\n(已停止)",
    });
  });

  it("stopActiveAgentStream 应在恢复消息壳被替换后按 turnId 补回正文", async () => {
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "recovered-assistant:turn-runtime-replaced",
      eventName: "agentSession/event/session-replaced",
      sessionId: "session-replaced",
      turnId: "turn-runtime-replaced",
    };
    let messages: Message[] = [];
    let threadItems: AgentThreadItem[] = [];
    const canonicalThreadItems: AgentThreadItem[] = [
      {
        id: "agent-message-final-turn-runtime-replaced",
        type: "agent_message",
        thread_id: "thread-replaced",
        turn_id: "turn-runtime-replaced",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-07-10T00:00:01.000Z",
        updated_at: "2026-07-10T00:00:01.000Z",
        text: "以下是今日国际新闻简要整理：",
        phase: "final_answer",
      },
    ];

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-replaced" },
      threadId: "thread-replaced",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({
          thread_items: canonicalThreadItems,
        })),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => null as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => threadItems,
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    await flushMicrotasks();
    expect(messages[0]).toMatchObject({
      role: "assistant",
      runtimeTurnId: "turn-runtime-replaced",
      content: "以下是今日国际新闻简要整理：\n\n(已停止)",
    });
  });

  it("停止时应优先更新同 turn 的可见 canonical message，而不是恢复壳", async () => {
    const runtimeTurnId = "turn-visible-canonical";
    let messages: Message[] = [
      {
        id: "recovered-assistant:visible-canonical",
        role: "assistant",
        content: "(已停止)",
        contentParts: [{ type: "text", text: "(已停止)" }],
        timestamp: new Date("2026-07-10T00:00:00.000Z"),
        runtimeTurnId,
      },
      {
        id: "agent-message-final-visible-canonical",
        role: "assistant",
        content: "以下是今日国际新闻简要整理：",
        contentParts: [
          { type: "text", text: "以下是今日国际新闻简要整理：" },
        ],
        timestamp: new Date("2026-07-10T00:00:01.000Z"),
        runtimeTurnId,
      },
    ];

    await stopActiveAgentStream({
      activeStream: {
        assistantMsgId: "recovered-assistant:visible-canonical",
        eventName: "agentSession/event/visible-canonical",
        sessionId: "session-visible-canonical",
        turnId: runtimeTurnId,
      },
      sessionIdRef: { current: "session-visible-canonical" },
      threadId: "thread-visible-canonical",
      currentTurnId: runtimeTurnId,
      runtime: {
        getSessionReadModel: vi.fn(async () => ({ thread_items: [] })),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: vi.fn(),
      setThreadTurns: vi.fn(),
      setCurrentTurnId: vi.fn(),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => [],
      setActiveStream: vi.fn(),
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(messages[0]?.content).toBe("(已停止)");
    expect(messages[1]?.content).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
  });

  it("stopActiveAgentStream 应把 pending 本地停止标记绑定到真实 runtime turn", async () => {
    let threadItems: AgentThreadItem[] = [
      {
        id: "pending-item:cancel",
        thread_id: "session-1",
        turn_id: "pending-turn:cancel",
        sequence: 0,
        status: "in_progress",
        started_at: "2026-07-10T00:00:00.000Z",
        updated_at: "2026-07-10T00:00:00.000Z",
        type: "turn_summary",
        text: "正在生成回复",
      },
      {
        id: "item-real-cancel",
        thread_id: "session-1",
        turn_id: "turn-real-cancel",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-07-10T00:00:01.000Z",
        updated_at: "2026-07-10T00:00:01.000Z",
        type: "agent_message",
        text: "以下是今日国际新闻简要整理：",
      },
    ];
    let threadTurns: AgentThreadTurn[] = [
      {
        id: "pending-turn:cancel",
        thread_id: "session-1",
        prompt_text: "整理今天的国际新闻",
        status: "running",
        started_at: "2026-07-10T00:00:00.000Z",
        created_at: "2026-07-10T00:00:00.000Z",
        updated_at: "2026-07-10T00:00:00.000Z",
      },
    ];
    let currentTurnId: string | null = "pending-turn:cancel";
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "assistant-cancel",
      eventName: "stream-real-turn-cancel",
      sessionId: "session-1",
      pendingTurnKey: "pending-turn:cancel",
      pendingItemKey: "pending-item:cancel",
    };
    let messages: Message[] = [
      {
        id: "assistant-cancel",
        role: "assistant",
        content: "以下是今日国际新闻简要整理：",
        contentParts: [
          {
            type: "text",
            text: "以下是今日国际新闻简要整理：",
          },
        ],
        timestamp: new Date("2026-07-10T00:00:01.000Z"),
        isThinking: true,
        runtimeTurnId: "pending-turn:cancel",
      },
    ];
    const interruptTurn = vi.fn(async () => true);
    upsertAgentStreamTextOverlay({
      messageId: "assistant-cancel",
      eventName: "stream-real-turn-cancel",
      content: "以下是今日国际新闻简要整理：",
      turnId: "turn-real-cancel",
    });

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-1" },
      threadId: "thread-1",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn,
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => threadTurns,
        (value) => {
          threadTurns = value;
        },
      ),
      setCurrentTurnId: createStateSetter(
        () => currentTurnId,
        (value) => {
          currentTurnId = value;
        },
      ),
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      getMessages: () => messages,
      getThreadItems: () => threadItems,
      setActiveStream: (next) => {
        activeStream = next;
      },
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(getAgentStreamTextOverlay("assistant-cancel")).toBeNull();
    expect(interruptTurn).toHaveBeenCalledWith(
      "thread-1",
      "turn-real-cancel",
      "stream-real-turn-cancel",
    );
    expect(threadItems.map((item) => item.id)).toEqual(["item-real-cancel"]);
    expect(threadItems[0]).toMatchObject({
      type: "agent_message",
      text: "以下是今日国际新闻简要整理：\n\n(已停止)",
      contentParts: [
        {
          type: "text",
          text: "以下是今日国际新闻简要整理：",
        },
        {
          type: "text",
          text: "(已停止)",
        },
      ],
    });
    expect(threadTurns).toEqual([]);
    expect(currentTurnId).toBeNull();
    expect(messages[0]).toMatchObject({
      content: "以下是今日国际新闻简要整理：\n\n(已停止)",
      isThinking: false,
      runtimeTurnId: "turn-real-cancel",
      runtimeStatus: undefined,
    });
    expect(
      messages[0]?.contentParts?.some(
        (part) => part.type === "text" && part.text === "(已停止)",
      ),
    ).toBe(true);
    expect(activeStream).toBeNull();
  });

  it("buildInterruptedMessageContentPatch 已有停止标记时应保持幂等", () => {
    const patch = buildInterruptedMessageContentPatch({
      id: "assistant-already-stopped",
      role: "assistant",
      content: "已输出\n\n(已停止)",
      contentParts: [
        {
          type: "text",
          text: "已输出",
        },
        {
          type: "text",
          text: "(已停止)",
        },
      ],
      timestamp: new Date("2026-03-29T00:00:00.000Z"),
    });

    expect(patch.content).toBe("已输出\n\n(已停止)");
    expect(patch.contentParts).toEqual([
      {
        type: "text",
        text: "已输出",
      },
      {
        type: "text",
        text: "(已停止)",
      },
    ]);
  });

  it("stopActiveAgentStream 应把 output-free 中断输入恢复为富输入请求", async () => {
    const image = {
      data: "image-data",
      mediaType: "image/png",
    };
    const pathReference = {
      id: "file:/tmp/report.md",
      path: "/tmp/report.md",
      name: "report.md",
      isDir: false,
      source: "file_manager" as const,
    };
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "assistant-thinking",
      eventName: "stream-restore",
      sessionId: "session-1",
      turnId: "turn-runtime-1",
      submittedDraft: {
        text: "继续生成提纲",
        images: [image],
        pathReferences: [pathReference],
        inputCapabilityRoute: {
          kind: "installed_skill",
          skillKey: "draft",
          skillName: "起草",
        },
      },
    };
    const onRestoreInterruptedInput = vi.fn();
    const restoreOrder: string[] = [];
    let resolveRefresh!: (value: boolean) => void;
    const refreshDone = new Promise<boolean>((resolve) => {
      resolveRefresh = resolve;
    });
    const refreshSessionReadModel = vi.fn(() => {
      restoreOrder.push("refresh");
      return refreshDone;
    });
    onRestoreInterruptedInput.mockImplementation(() => {
      restoreOrder.push("restore");
    });

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-1" },
      threadId: "thread-1",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel,
      setThreadItems: createStateSetter(
        () => [] as AgentThreadItem[],
        () => undefined,
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => null as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => [] as Message[],
        () => undefined,
      ),
      getMessages: () => [
        {
          id: "assistant-thinking",
          role: "assistant",
          content: "",
          timestamp: new Date("2026-03-29T00:00:00.000Z"),
          contentParts: [{ type: "thinking", text: "正在思考" }],
        },
      ],
      setActiveStream: (next) => {
        activeStream = next;
      },
      onRestoreInterruptedInput,
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(activeStream).toBeNull();
    expect(onRestoreInterruptedInput).toHaveBeenCalledWith({
      requestId: expect.any(String),
      reason: "thinking_only_cancelled_turn",
      draft: {
        text: "继续生成提纲",
        images: [image],
        pathReferences: [pathReference],
        textElements: [],
        inputCapabilityRoute: {
          kind: "installed_skill",
          skillKey: "draft",
          skillName: "起草",
        },
      },
    });
    expect(restoreOrder[0]).toBe("restore");

    await flushMicrotasks();
    expect(refreshSessionReadModel).toHaveBeenCalledWith("session-1");
    expect(restoreOrder).toEqual(["restore", "refresh"]);

    resolveRefresh(true);
    await flushMicrotasks();

    expect(onRestoreInterruptedInput).toHaveBeenCalledTimes(1);
  });

  it("stopActiveAgentStream 在 active stream 尚无草稿时应使用提交兜底草稿", async () => {
    const image = {
      data: "image-data",
      mediaType: "image/png",
    };
    const pathReference = {
      id: "file:/tmp/report.md",
      path: "/tmp/report.md",
      name: "report.md",
      isDir: false,
      source: "file_manager" as const,
    };
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "assistant-thinking",
      eventName: "stream-restore-before-active-draft",
      sessionId: "session-1",
      turnId: "turn-runtime-1",
    };
    const onRestoreInterruptedInput = vi.fn();

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-1" },
      threadId: "thread-1",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => [] as AgentThreadItem[],
        () => undefined,
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => null as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => [] as Message[],
        () => undefined,
      ),
      getMessages: () => [
        {
          id: "assistant-thinking",
          role: "assistant",
          content: "",
          timestamp: new Date("2026-03-29T00:00:00.000Z"),
          isThinking: true,
        },
      ],
      setActiveStream: (next) => {
        activeStream = next;
      },
      submittedDraftFallback: {
        text: "继续生成提纲",
        images: [image],
        pathReferences: [pathReference],
        inputCapabilityRoute: {
          kind: "installed_skill",
          skillKey: "draft",
          skillName: "起草",
        },
      },
      onRestoreInterruptedInput,
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(activeStream).toBeNull();
    expect(onRestoreInterruptedInput).toHaveBeenCalledWith({
      requestId: expect.any(String),
      reason: "thinking_only_cancelled_turn",
      draft: {
        text: "继续生成提纲",
        images: [image],
        pathReferences: [pathReference],
        textElements: [],
        inputCapabilityRoute: {
          kind: "installed_skill",
          skillKey: "draft",
          skillName: "起草",
        },
      },
    });
  });

  it("stopActiveAgentStream 只有 provider/runtime 过程态时仍应恢复富输入", async () => {
    const image = {
      data: "image-data",
      mediaType: "image/png",
      sourcePath: "/tmp/rich-restore-fixture.png",
    };
    const pathReference = {
      id: "file:/tmp/clawstream-rich-restore-fixture.md",
      path: "/tmp/clawstream-rich-restore-fixture.md",
      name: "clawstream-rich-restore-fixture.md",
      isDir: false,
      source: "file_manager" as const,
    };
    let activeStream: ActiveStreamState | null = {
      assistantMsgId: "assistant-runtime-status",
      eventName: "stream-provider-trace-only",
      sessionId: "session-rich-restore",
      turnId: "turn-provider-trace-only",
      pendingTurnKey: "pending-turn:provider-trace-only",
      pendingItemKey: "pending-item:provider-trace-only",
      submittedDraft: {
        text: "请结合这个截图、文件和 Capability Report 技能，先不要输出正文。",
        images: [image],
        pathReferences: [pathReference],
        inputCapabilityRoute: {
          kind: "installed_skill",
          skillKey: "capability-report",
          skillName: "Capability Report",
        },
      },
    };
    const onRestoreInterruptedInput = vi.fn();

    await stopActiveAgentStream({
      activeStream,
      sessionIdRef: { current: "session-rich-restore" },
      threadId: "thread-rich-restore",
      runtime: {
        getSessionReadModel: vi.fn(async () => ({})),
        interruptTurn: vi.fn(async () => true),
      } as never,
      removeStreamListener: vi.fn(),
      refreshSessionReadModel: vi.fn(async () => true),
      setThreadItems: createStateSetter(
        () => [] as AgentThreadItem[],
        () => undefined,
      ),
      setThreadTurns: createStateSetter(
        () => [] as AgentThreadTurn[],
        () => undefined,
      ),
      setCurrentTurnId: createStateSetter(
        () => "pending-turn:provider-trace-only" as string | null,
        () => undefined,
      ),
      setMessages: createStateSetter(
        () => [] as Message[],
        () => undefined,
      ),
      getMessages: () => [
        {
          id: "assistant-runtime-status",
          role: "assistant",
          content: "正在生成回复",
          timestamp: new Date("2026-03-29T00:00:00.000Z"),
          isThinking: true,
          contentParts: [],
          runtimeStatus: {
            phase: "preparing",
            title: "正在生成回复",
            detail: "正在输出",
            checkpoints: ["provider.request.started"],
          },
        },
      ],
      setActiveStream: (next) => {
        activeStream = next;
      },
      onRestoreInterruptedInput,
      notify: {
        info: vi.fn(),
        error: vi.fn(),
      },
    });

    expect(activeStream).toBeNull();
    expect(onRestoreInterruptedInput).toHaveBeenCalledWith({
      requestId: expect.any(String),
      reason: "thinking_only_cancelled_turn",
      draft: {
        text: "请结合这个截图、文件和 Capability Report 技能，先不要输出正文。",
        images: [image],
        pathReferences: [pathReference],
        textElements: [],
        inputCapabilityRoute: {
          kind: "installed_skill",
          skillKey: "capability-report",
          skillName: "Capability Report",
        },
      },
    });
  });

  it("settleInterruptedMessageProcess 应把运行中工具标记为本轮已中止", () => {
    const message = settleInterruptedMessageProcess({
      id: "assistant-with-running-tool",
      role: "assistant",
      content: "已完成部分整理",
      timestamp: new Date("2026-03-29T00:00:00.000Z"),
      toolCalls: [
        {
          id: "tool-running-1",
          name: "Bash",
          arguments: JSON.stringify({ command: "python3 news.py" }),
          status: "running",
          startTime: new Date("2026-03-29T00:00:01.000Z"),
        },
        {
          id: "tool-completed-1",
          name: "WebSearch",
          arguments: JSON.stringify({ query: "world news" }),
          status: "completed",
          result: {
            success: true,
            output: "已找到 3 条来源",
          },
          startTime: new Date("2026-03-29T00:00:02.000Z"),
          endTime: new Date("2026-03-29T00:00:03.000Z"),
        },
      ],
      contentParts: [
        {
          type: "tool_use",
          toolCall: {
            id: "tool-running-1",
            name: "Bash",
            arguments: JSON.stringify({ command: "python3 news.py" }),
            status: "running",
            startTime: new Date("2026-03-29T00:00:01.000Z"),
          },
        },
      ],
    });

    expect(message.toolCalls?.[0]).toMatchObject({
      id: "tool-running-1",
      status: "failed",
      result: {
        success: false,
        output: "",
        error: "本轮已中止",
      },
    });
    expect(message.toolCalls?.[0]?.endTime).toBeInstanceOf(Date);
    expect(message.toolCalls?.[1]).toMatchObject({
      id: "tool-completed-1",
      status: "completed",
      result: {
        success: true,
        output: "已找到 3 条来源",
      },
    });
    const toolPart = message.contentParts?.[0];
    expect(toolPart?.type).toBe("tool_use");
    if (toolPart?.type === "tool_use") {
      expect(toolPart.toolCall).toMatchObject({
        id: "tool-running-1",
        status: "failed",
        result: {
          error: "本轮已中止",
        },
      });
    }
  });
});
