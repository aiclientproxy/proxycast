import { describe, expect, it } from "vitest";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentThreadItem, Message } from "../types";
import { buildHydratedAgentSessionSnapshot } from "./agentSessionState";
import {
  hasLocallyInterruptedAgentStreamBinding,
  rememberLocallyInterruptedAgentStreamBinding,
} from "./agentStreamResumeBinding";

function createMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: overrides.id ?? "message-1",
    role: overrides.role ?? "assistant",
    content: overrides.content ?? "默认内容",
    timestamp: overrides.timestamp ?? new Date("2026-06-30T00:00:00.000Z"),
    ...overrides,
  };
}

function createAgentMessageItem(
  overrides: Partial<AgentThreadItem> = {},
): AgentThreadItem {
  return {
    id: overrides.id ?? "item-1",
    thread_id: overrides.thread_id ?? "topic-runtime-sync",
    turn_id: overrides.turn_id ?? "turn-runtime-sync",
    sequence: overrides.sequence ?? 1,
    type: "agent_message",
    text:
      "text" in overrides && typeof overrides.text === "string"
        ? overrides.text
        : "默认 thread item",
    status: overrides.status ?? "completed",
    started_at: overrides.started_at ?? "2026-06-30T00:00:00.000Z",
    completed_at: overrides.completed_at ?? "2026-06-30T00:00:02.000Z",
    updated_at: overrides.updated_at ?? "2026-06-30T00:00:02.000Z",
    ...overrides,
  } as AgentThreadItem;
}

describe("agentSessionState runtimeSync detail refresh", () => {
  it("App Server detail 协议形状应能水合出用户与助手消息", () => {
    const detail = {
      id: "topic-app-server-detail",
      thread_id: "thread-app-server-detail",
      created_at: 1783480361000,
      updated_at: 1783480369000,
      messages_count: 2,
      messages: [
        {
          id: "turn-app-server:user",
          role: "user",
          content: [
            {
              type: "text",
              text: "请只回复：收到-CDP正常对话",
            },
          ],
        },
        {
          id: "turn-app-server:assistant",
          role: "assistant",
          content: [
            {
              type: "text",
              text: "收到-CDP正常对话",
            },
          ],
        },
      ],
      items: [
        {
          id: "assistant:chatcmpl-app-server",
          type: "agent_message",
          status: "completed",
          turn_id: "turn-app-server",
          text: "收到-CDP正常对话",
        },
      ],
      turns: [
        {
          completedAt: "2026-07-08T03:12:49.191Z",
          sessionId: "topic-app-server-detail",
          startedAt: "2026-07-08T03:12:41.154Z",
          status: "completed",
          threadId: "thread-app-server-detail",
          turnId: "turn-app-server",
        },
      ],
      thread_read: {
        thread_id: "thread-app-server-detail",
        status: "completed",
        active_turn_id: null,
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-app-server-detail",
      detail,
      currentSessionId: "topic-app-server-detail",
      currentMessages: [],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [
        {
          id: "topic-app-server-detail",
          title: "请只回复：收到-CDP正常对话",
          createdAt: new Date("2026-07-08T03:12:41.154Z"),
          updatedAt: new Date("2026-07-08T03:12:49.191Z"),
          workspaceId: null,
          messagesCount: 2,
          executionStrategy: "react",
          status: "done",
          lastPreview: "收到-CDP正常对话",
          isPinned: false,
          hasUnread: false,
          tag: null,
          sourceSessionId: "topic-app-server-detail",
        },
      ],
    });

    expect(result.snapshot.messages.map((message) => message.content)).toEqual([
      "请只回复：收到-CDP正常对话",
      "收到-CDP正常对话",
    ]);
  });

  it("runtimeSync detail rebase 不应让不兼容历史消息覆盖当前时间线", () => {
    const liveText =
      "这是当前 turn 已经流出来的较长回答，session detail refresh 只能合入元数据，不能把它替换成另一条历史回答。";
    const currentMessages = [
      createMessage({
        id: "local-user-current",
        role: "user",
        content: "当前正在流式输出的问题",
        timestamp: new Date("2026-06-30T10:00:00.000Z"),
      }),
      createMessage({
        id: "local-assistant-current",
        role: "assistant",
        content: liveText,
        timestamp: new Date("2026-06-30T10:00:01.000Z"),
      }),
    ];
    const currentThreadItems = [
      createAgentMessageItem({
        id: "item-live",
        text: liveText,
        status: "in_progress",
      }),
    ];
    const detail = {
      id: "topic-runtime-sync",
      created_at: 1782800000,
      updated_at: 1782800001,
      messages: [
        {
          role: "user",
          timestamp: 1782799000,
          content: [{ type: "text", text: "另一条不兼容的历史问题" }],
        },
        {
          role: "assistant",
          timestamp: 1782799001,
          content: [{ type: "text", text: "另一条历史回答" }],
        },
      ],
      items: [
        createAgentMessageItem({
          id: "item-live",
          text: "这是当前 turn 已经流出来的较长回答",
          status: "completed",
          updated_at: "2026-06-30T10:00:03.000Z",
        }),
      ],
      thread_read: {
        thread_id: "topic-runtime-sync",
        status: "waiting_request",
        active_turn_id: "turn-runtime-sync",
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-runtime-sync",
      detail,
      currentSessionId: "topic-runtime-sync",
      currentMessages,
      currentThreadTurns: [],
      currentThreadItems,
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "runtime_sync",
    });

    expect(result.snapshot.messages.map((message) => message.content)).toEqual(
      currentMessages.map((message) => message.content),
    );
    expect(result.snapshot.threadItems).toHaveLength(1);
    expect(result.snapshot.threadItems[0]).toMatchObject({
      id: "item-live",
      status: "completed",
      text: liveText,
    });
    expect(result.snapshot.threadRead?.status).toBe("waiting_request");
    expect(result.snapshot).not.toHaveProperty("queuedTurns");
  });

  it("runtimeSync terminal detail 应允许最终 transcript 接管 pending 壳", () => {
    const detail = {
      id: "topic-runtime-sync-terminal",
      created_at: 1782800000,
      updated_at: 1782800001,
      messages: [
        {
          role: "user",
          timestamp: 1782800000,
          content: [{ type: "text", text: "最终用户问题" }],
        },
        {
          role: "assistant",
          timestamp: 1782800001,
          content: [{ type: "text", text: "最终 assistant 总结" }],
        },
      ],
      turns: [
        {
          id: "turn-runtime-sync-terminal",
          thread_id: "topic-runtime-sync-terminal-thread",
          prompt_text: "最终用户问题",
          status: "completed",
          started_at: "2026-06-30T10:00:00.000Z",
          completed_at: "2026-06-30T10:00:02.000Z",
          created_at: "2026-06-30T10:00:00.000Z",
          updated_at: "2026-06-30T10:00:02.000Z",
        },
      ],
      thread_read: {
        thread_id: "topic-runtime-sync-terminal-thread",
        status: "completed",
        active_turn_id: "turn-runtime-sync-terminal",
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-runtime-sync-terminal",
      detail,
      currentSessionId: "topic-runtime-sync-terminal",
      currentMessages: [
        createMessage({
          id: "local-pending-user",
          role: "user",
          content: "pending 壳里的临时问题",
        }),
        createMessage({
          id: "local-pending-assistant",
          role: "assistant",
          content: "正在生成回复",
        }),
      ],
      currentThreadTurns: [
        {
          id: "pending-turn:stale-renderer-shell",
          thread_id: "topic-runtime-sync-terminal-thread",
          prompt_text: "pending 壳里的临时问题",
          status: "running",
          started_at: "2026-06-30T10:00:00.000Z",
          created_at: "2026-06-30T10:00:00.000Z",
          updated_at: "2026-06-30T10:00:01.000Z",
        },
      ],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "runtime_sync",
    });

    expect(result.snapshot.messages.map((message) => message.content)).toEqual([
      "最终用户问题",
      "最终 assistant 总结",
    ]);
    expect(result.snapshot.currentTurnId).toBe("turn-runtime-sync-terminal");
    expect(result.snapshot.threadRead?.status).toBe("completed");
    expect(result.snapshot.threadTurns.map((turn) => turn.id)).toEqual([
      "turn-runtime-sync-terminal",
    ]);
  });

  it("runtimeSync terminal detail 接管 pending 壳时应迁移本地已停止标记", () => {
    const turnId = "turn-runtime-sync-terminal-interrupted";
    const promptText = "整理今天的国际新闻";
    const finalText = "以下是今日国际新闻简要整理：";
    const detail = {
      id: "topic-runtime-sync-terminal-interrupted",
      created_at: 1782800000,
      updated_at: 1782800001,
      messages: [
        {
          role: "user",
          timestamp: 1782800000,
          content: [{ type: "text", text: promptText }],
        },
        {
          role: "assistant",
          timestamp: 1782800001,
          content: [{ type: "text", text: finalText }],
        },
      ],
      turns: [
        {
          id: turnId,
          thread_id: "topic-runtime-sync-terminal-interrupted-thread",
          prompt_text: promptText,
          status: "completed",
          started_at: "2026-07-10T00:00:00.000Z",
          completed_at: "2026-07-10T00:00:02.000Z",
          created_at: "2026-07-10T00:00:00.000Z",
          updated_at: "2026-07-10T00:00:02.000Z",
        },
      ],
      items: [
        createAgentMessageItem({
          id: "item-runtime-sync-terminal-interrupted-final",
          thread_id: "topic-runtime-sync-terminal-interrupted-thread",
          turn_id: turnId,
          text: finalText,
          status: "completed",
          sequence: 1,
          started_at: "2026-07-10T00:00:01.000Z",
          completed_at: "2026-07-10T00:00:02.000Z",
          updated_at: "2026-07-10T00:00:02.000Z",
        }),
      ],
      thread_read: {
        thread_id: "topic-runtime-sync-terminal-interrupted-thread",
        status: "completed",
        active_turn_id: turnId,
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-runtime-sync-terminal-interrupted",
      detail,
      currentSessionId: "topic-runtime-sync-terminal-interrupted",
      currentMessages: [
        createMessage({
          id: "local-pending-user",
          role: "user",
          content: promptText,
          runtimeTurnId: "pending-turn:interrupted",
        }),
        createMessage({
          id: "local-pending-assistant",
          role: "assistant",
          content: "(已停止)",
          contentParts: [{ type: "text", text: "(已停止)" }],
          isThinking: false,
          runtimeTurnId: "pending-turn:interrupted",
        }),
      ],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "runtime_sync",
    });

    expect(result.snapshot.messages.map((message) => message.content)).toEqual([
      promptText,
      `${finalText}\n\n(已停止)`,
    ]);
    expect(result.snapshot.messages[1]).toMatchObject({
      role: "assistant",
      runtimeTurnId: turnId,
      isThinking: false,
    });
    expect(result.snapshot.threadItems).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "agent_message",
          turn_id: turnId,
          text: `${finalText}\n\n(已停止)`,
        }),
      ]),
    );
  });

  it("runtimeSync terminal detail 只有 thread items 时也应替换本地 thinking pending 壳", () => {
    const turnId = "turn-runtime-sync-items-terminal";
    const promptText = "CDP 修复复测 marker";
    const finalText = "收到-CDP 修复复测 marker";
    const detail = {
      id: "topic-runtime-sync-items-terminal",
      created_at: 1782800000,
      updated_at: 1782800001,
      messages: [],
      turns: [
        {
          id: turnId,
          thread_id: "topic-runtime-sync-items-terminal-thread",
          prompt_text: promptText,
          status: "completed",
          started_at: "2026-06-30T10:00:00.000Z",
          completed_at: "2026-06-30T10:00:02.000Z",
          created_at: "2026-06-30T10:00:00.000Z",
          updated_at: "2026-06-30T10:00:02.000Z",
        },
      ],
      items: [
        createAgentMessageItem({
          id: "item-runtime-sync-items-terminal-final",
          thread_id: "topic-runtime-sync-items-terminal-thread",
          turn_id: turnId,
          text: finalText,
          status: "completed",
          sequence: 1,
          started_at: "2026-06-30T10:00:01.000Z",
          completed_at: "2026-06-30T10:00:02.000Z",
          updated_at: "2026-06-30T10:00:02.000Z",
        }),
      ],
      thread_read: {
        thread_id: "topic-runtime-sync-items-terminal-thread",
        status: "completed",
        active_turn_id: turnId,
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-runtime-sync-items-terminal",
      detail,
      currentSessionId: "topic-runtime-sync-items-terminal",
      currentMessages: [
        createMessage({
          id: "local-pending-user",
          role: "user",
          content: promptText,
          runtimeTurnId: turnId,
        }),
        createMessage({
          id: "local-pending-assistant",
          role: "assistant",
          content: "正在生成回复",
          isThinking: true,
          runtimeTurnId: turnId,
          runtimeStatus: {
            phase: "generating",
            title: "正在输出",
            detail: "正在等待模型继续输出。",
          },
        }),
      ],
      currentThreadTurns: [
        {
          id: turnId,
          thread_id: "topic-runtime-sync-items-terminal-thread",
          prompt_text: promptText,
          status: "running",
          started_at: "2026-06-30T10:00:00.000Z",
          created_at: "2026-06-30T10:00:00.000Z",
          updated_at: "2026-06-30T10:00:01.000Z",
        },
      ],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "runtime_sync",
    });

    expect(result.snapshot.messages.map((message) => message.content)).toEqual([
      promptText,
      finalText,
    ]);
    expect(result.snapshot.messages[1]).toMatchObject({
      role: "assistant",
      isThinking: false,
      runtimeTurnId: turnId,
    });
    expect(result.snapshot.messages[1]?.runtimeStatus).toBeUndefined();
    expect(result.snapshot.currentTurnId).toBe(turnId);
    expect(result.snapshot.threadRead?.status).toBe("completed");
  });

  it("历史回合的已停止标记不能污染后续已完成回合", () => {
    const canceledTurnId = "turn-runtime-sync-canceled-history";
    const completedTurnId = "turn-runtime-sync-continued-completed";
    const continuedText = "继续输出已恢复：正文已经正常收口";
    const detail = {
      id: "topic-runtime-sync-cancel-then-continue",
      created_at: 1782800000,
      updated_at: 1782800004,
      messages: [
        {
          role: "user",
          timestamp: 1782800000,
          content: [{ type: "text", text: "整理今天的国际新闻" }],
        },
        {
          role: "assistant",
          timestamp: 1782800001,
          content: [{ type: "text", text: "(已停止)" }],
        },
        {
          role: "user",
          timestamp: 1782800002,
          content: [{ type: "text", text: "继续输出" }],
        },
        {
          role: "assistant",
          timestamp: 1782800003,
          content: [{ type: "text", text: continuedText }],
        },
      ],
      turns: [
        {
          id: canceledTurnId,
          thread_id: "topic-runtime-sync-cancel-then-continue-thread",
          prompt_text: "整理今天的国际新闻",
          status: "canceled",
          started_at: "2026-07-10T00:00:00.000Z",
          completed_at: "2026-07-10T00:00:01.000Z",
          created_at: "2026-07-10T00:00:00.000Z",
          updated_at: "2026-07-10T00:00:01.000Z",
        },
        {
          id: completedTurnId,
          thread_id: "topic-runtime-sync-cancel-then-continue-thread",
          prompt_text: "继续输出",
          status: "completed",
          started_at: "2026-07-10T00:00:02.000Z",
          completed_at: "2026-07-10T00:00:04.000Z",
          created_at: "2026-07-10T00:00:02.000Z",
          updated_at: "2026-07-10T00:00:04.000Z",
        },
      ],
      items: [
        createAgentMessageItem({
          id: "item-runtime-sync-canceled-history",
          thread_id: "topic-runtime-sync-cancel-then-continue-thread",
          turn_id: canceledTurnId,
          text: "(已停止)",
          sequence: 1,
        }),
        createAgentMessageItem({
          id: "item-runtime-sync-continued-completed",
          thread_id: "topic-runtime-sync-cancel-then-continue-thread",
          turn_id: completedTurnId,
          text: continuedText,
          sequence: 2,
        }),
      ],
      thread_read: {
        thread_id: "topic-runtime-sync-cancel-then-continue-thread",
        status: "completed",
        active_turn_id: completedTurnId,
        turns: [
          { turn_id: canceledTurnId, status: "canceled" },
          { turn_id: completedTurnId, status: "completed" },
        ],
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-runtime-sync-cancel-then-continue",
      detail,
      currentSessionId: "topic-runtime-sync-cancel-then-continue",
      currentMessages: [
        createMessage({
          id: "local-canceled-user",
          role: "user",
          content: "整理今天的国际新闻",
          runtimeTurnId: canceledTurnId,
        }),
        createMessage({
          id: "local-canceled-assistant",
          role: "assistant",
          content: "(已停止)",
          contentParts: [{ type: "text", text: "(已停止)" }],
          isThinking: false,
          runtimeTurnId: canceledTurnId,
        }),
        createMessage({
          id: "local-continued-user",
          role: "user",
          content: "继续输出",
          runtimeTurnId: completedTurnId,
        }),
        createMessage({
          id: "local-continued-assistant",
          role: "assistant",
          content: continuedText,
          isThinking: false,
          runtimeTurnId: completedTurnId,
        }),
      ],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "terminal_reconcile",
    });

    const continuedAssistant = result.snapshot.messages.find(
      (message) => message.runtimeTurnId === completedTurnId,
    );
    expect(continuedAssistant).toMatchObject({
      role: "assistant",
      content: continuedText,
      isThinking: false,
    });
    expect(continuedAssistant?.content).not.toContain("(已停止)");
  });

  it("terminalReconcile detached canceled detail 应迁移 pending-turn 已停止终态", () => {
    const realTurnId = "turn-runtime-sync-detached-canceled-real";
    const promptText = "整理今天的国际新闻";
    const finalText = "以下是今日国际新闻简要整理：";
    const detail = {
      id: "topic-runtime-sync-detached-canceled",
      created_at: 1782800000,
      updated_at: 1782800001,
      messages: [],
      turns: [
        {
          id: realTurnId,
          thread_id: "topic-runtime-sync-detached-canceled-thread",
          prompt_text: promptText,
          status: "canceled",
          started_at: "2026-07-10T00:00:00.000Z",
          completed_at: "2026-07-10T00:00:02.000Z",
          created_at: "2026-07-10T00:00:00.000Z",
          updated_at: "2026-07-10T00:00:02.000Z",
        },
      ],
      items: [
        createAgentMessageItem({
          id: "item-runtime-sync-detached-canceled-final",
          thread_id: "topic-runtime-sync-detached-canceled-thread",
          turn_id: realTurnId,
          text: finalText,
          status: "completed",
          sequence: 1,
          started_at: "2026-07-10T00:00:01.000Z",
          completed_at: "2026-07-10T00:00:02.000Z",
          updated_at: "2026-07-10T00:00:02.000Z",
        }),
      ],
      thread_read: {
        thread_id: "topic-runtime-sync-detached-canceled-thread",
        status: "canceled",
        active_turn_id: realTurnId,
        turns: [
          {
            turn_id: realTurnId,
            status: "cancelled",
            native_status: "canceled",
          },
        ],
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-runtime-sync-detached-canceled",
      detail,
      currentSessionId: null,
      currentMessages: [
        createMessage({
          id: "local-pending-user",
          role: "user",
          content: promptText,
          runtimeTurnId: "pending-turn:detached-canceled",
        }),
        createMessage({
          id: "local-pending-assistant",
          role: "assistant",
          content: "(已停止)",
          contentParts: [{ type: "text", text: "(已停止)" }],
          isThinking: false,
          runtimeStatus: undefined,
          runtimeTurnId: "pending-turn:detached-canceled",
        }),
      ],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "terminal_reconcile",
    });

    expect(result.snapshot.messages).toHaveLength(2);
    expect(result.snapshot.messages[0]).toMatchObject({
      id: "local-pending-user",
      role: "user",
      content: promptText,
    });
    expect(result.snapshot.messages[1]).toMatchObject({
      id: "local-pending-assistant",
      role: "assistant",
      content: `${finalText}\n\n(已停止)`,
      runtimeTurnId: realTurnId,
      isThinking: false,
    });
    expect(
      result.snapshot.messages[1]?.contentParts?.some(
        (part) => part.type === "text" && part.text === "(已停止)",
      ),
    ).toBe(true);
    expect(result.snapshot.currentTurnId).toBe(realTurnId);
    expect(result.snapshot.threadRead?.status).toBe("canceled");
  });

  it("runtimeSync canceled detail 只有 thread items 时应保留已停止终态标记", () => {
    const turnId = "turn-runtime-sync-items-canceled";
    const promptText = "整理今天的国际新闻";
    const finalText = "以下是今日国际新闻简要整理：";
    const detail = {
      id: "topic-runtime-sync-items-canceled",
      created_at: 1782800000,
      updated_at: 1782800001,
      messages: [],
      turns: [],
      items: [
        createAgentMessageItem({
          id: "item-runtime-sync-items-canceled-final",
          thread_id: "topic-runtime-sync-items-canceled-thread",
          turn_id: turnId,
          text: finalText,
          status: "completed",
          sequence: 1,
          started_at: "2026-07-10T00:00:01.000Z",
          completed_at: "2026-07-10T00:00:02.000Z",
          updated_at: "2026-07-10T00:00:02.000Z",
        }),
      ],
      thread_read: {
        thread_id: "topic-runtime-sync-items-canceled-thread",
        status: "canceled",
        active_turn_id: undefined,
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-runtime-sync-items-canceled",
      detail,
      currentSessionId: "topic-runtime-sync-items-canceled",
      currentMessages: [
        createMessage({
          id: "local-pending-user",
          role: "user",
          content: promptText,
          runtimeTurnId: turnId,
        }),
        createMessage({
          id: "local-pending-assistant",
          role: "assistant",
          content: "",
          isThinking: true,
          runtimeTurnId: turnId,
        }),
      ],
      currentThreadTurns: [
        {
          id: turnId,
          thread_id: "topic-runtime-sync-items-canceled-thread",
          prompt_text: promptText,
          status: "running",
          started_at: "2026-07-10T00:00:00.000Z",
          created_at: "2026-07-10T00:00:00.000Z",
          updated_at: "2026-07-10T00:00:01.000Z",
        },
      ],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "runtime_sync",
    });

    expect(result.snapshot.messages.map((message) => message.content)).toEqual([
      promptText,
      `${finalText}\n\n(已停止)`,
    ]);
    expect(
      result.snapshot.messages[1]?.contentParts?.some(
        (part) => part.type === "text" && part.text === "(已停止)",
      ),
    ).toBe(true);
    expect(result.snapshot.messages[1]?.isThinking).toBe(false);
    expect(result.snapshot.currentTurnId).toBe(turnId);
    expect(result.snapshot.threadRead?.status).toBe("canceled");
  });

  it("runtimeSync canceled 只出现在 thread_read.turns 时也应保留已停止终态标记", () => {
    const turnId = "turn-runtime-sync-thread-read-canceled";
    const promptText = "整理今天的国际新闻";
    const finalText = "以下是今日国际新闻简要整理：";
    const detail = {
      id: "topic-runtime-sync-thread-read-canceled",
      created_at: 1782800000,
      updated_at: 1782800001,
      messages: [],
      turns: [
        {
          id: turnId,
          thread_id: "topic-runtime-sync-thread-read-canceled-thread",
          prompt_text: promptText,
          status: "running",
          started_at: "2026-07-10T00:00:00.000Z",
          created_at: "2026-07-10T00:00:00.000Z",
          updated_at: "2026-07-10T00:00:01.000Z",
        },
      ],
      items: [
        createAgentMessageItem({
          id: "item-runtime-sync-thread-read-canceled-final",
          thread_id: "topic-runtime-sync-thread-read-canceled-thread",
          turn_id: turnId,
          text: finalText,
          status: "completed",
          sequence: 1,
          started_at: "2026-07-10T00:00:01.000Z",
          completed_at: "2026-07-10T00:00:02.000Z",
          updated_at: "2026-07-10T00:00:02.000Z",
        }),
      ],
      thread_read: {
        thread_id: "topic-runtime-sync-thread-read-canceled-thread",
        status: "canceled",
        active_turn_id: turnId,
        turns: [
          {
            turn_id: turnId,
            status: "cancelled",
            native_status: "canceled",
          },
        ],
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-runtime-sync-thread-read-canceled",
      detail,
      currentSessionId: "topic-runtime-sync-thread-read-canceled",
      currentMessages: [
        createMessage({
          id: "local-pending-user",
          role: "user",
          content: promptText,
          runtimeTurnId: turnId,
        }),
        createMessage({
          id: "local-pending-assistant",
          role: "assistant",
          content: "",
          isThinking: true,
          runtimeTurnId: turnId,
        }),
      ],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "runtime_sync",
    });

    expect(result.snapshot.messages.map((message) => message.content)).toEqual([
      promptText,
      `${finalText}\n\n(已停止)`,
    ]);
    expect(
      result.snapshot.messages[1]?.contentParts?.some(
        (part) => part.type === "text" && part.text === "(已停止)",
      ),
    ).toBe(true);
    expect(result.snapshot.messages[1]?.isThinking).toBe(false);
    expect(result.snapshot.currentTurnId).toBe(turnId);
  });

  it("terminal detail 只有 completed partial 时应消费本地停止绑定并保留已停止终态", () => {
    const topicId = "topic-runtime-sync-local-interrupted-binding";
    const turnId = "turn-runtime-sync-local-interrupted-binding";
    const promptText = "整理今天的国际新闻";
    const finalText = "以下是今日国际新闻简要整理：";

    rememberLocallyInterruptedAgentStreamBinding({
      assistantMsgId: "local-interrupted-assistant",
      eventName: "agent_stream_local-interrupted-binding",
      sessionId: topicId,
    });

    const detail = {
      id: topicId,
      created_at: 1782800000,
      updated_at: 1782800001,
      messages: [
        {
          id: `${turnId}:user`,
          role: "user",
          content: [{ type: "text", text: promptText }],
        },
        {
          id: `${turnId}:assistant`,
          role: "assistant",
          content: [{ type: "text", text: finalText }],
        },
      ],
      turns: [
        {
          id: turnId,
          thread_id: `${topicId}-thread`,
          prompt_text: promptText,
          status: "running",
          started_at: "2026-07-10T00:00:00.000Z",
          created_at: "2026-07-10T00:00:00.000Z",
          updated_at: "2026-07-10T00:00:02.000Z",
        },
      ],
      items: [
        createAgentMessageItem({
          id: "item-runtime-sync-local-interrupted-binding-final",
          thread_id: `${topicId}-thread`,
          turn_id: turnId,
          text: finalText,
          status: "completed",
          sequence: 1,
          started_at: "2026-07-10T00:00:01.000Z",
          completed_at: "2026-07-10T00:00:02.000Z",
          updated_at: "2026-07-10T00:00:02.000Z",
        }),
      ],
      thread_read: {
        thread_id: `${topicId}-thread`,
        status: "running",
        active_turn_id: turnId,
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId,
      detail,
      currentSessionId: topicId,
      currentMessages: [],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      detailMergeMode: "terminal_reconcile",
    });

    expect(result.snapshot.messages.map((message) => message.content)).toEqual([
      promptText,
      `${finalText}\n\n(已停止)`,
    ]);
    expect(result.snapshot.messages[1]).toMatchObject({
      role: "assistant",
      runtimeTurnId: turnId,
      isThinking: false,
    });
    expect(
      result.snapshot.messages[1]?.contentParts?.some(
        (part) => part.type === "text" && part.text === "(已停止)",
      ),
    ).toBe(true);
    expect(result.snapshot.threadItems).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "agent_message",
          turn_id: turnId,
          text: `${finalText}\n\n(已停止)`,
        }),
      ]),
    );
    expect(
      hasLocallyInterruptedAgentStreamBinding({
        eventName: `agentSession/event/${topicId}`,
        sessionId: topicId,
        threadId: `${topicId}-thread`,
        turnId,
      }),
    ).toBe(false);
  });
});
