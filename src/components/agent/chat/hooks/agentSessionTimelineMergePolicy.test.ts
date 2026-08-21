import { describe, expect, it } from "vitest";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import {
  hasAssistantActivitySnapshot,
  mergeRuntimeSyncThreadItems,
  resolveAgentSessionTimelineMergeDecision,
  shouldSkipStaleEmptyMessagesRefSync,
} from "./agentSessionTimelineMergePolicy";

function createMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: overrides.id ?? "message-1",
    role: overrides.role ?? "assistant",
    content: overrides.content ?? "默认内容",
    timestamp: overrides.timestamp ?? new Date("2026-07-01T00:00:00.000Z"),
    ...overrides,
  };
}

function createTurn(overrides: Partial<AgentThreadTurn> = {}): AgentThreadTurn {
  return {
    id: overrides.id ?? "turn-1",
    thread_id: overrides.thread_id ?? "thread-1",
    prompt_text: overrides.prompt_text ?? "继续",
    status: overrides.status ?? "running",
    started_at: overrides.started_at ?? "2026-07-01T00:00:00.000Z",
    created_at: overrides.created_at ?? "2026-07-01T00:00:00.000Z",
    updated_at: overrides.updated_at ?? "2026-07-01T00:00:01.000Z",
    ...overrides,
  };
}

function createAgentMessageItem(
  overrides: Partial<AgentThreadItem> = {},
): AgentThreadItem {
  return {
    id: overrides.id ?? "item-1",
    thread_id: overrides.thread_id ?? "thread-1",
    turn_id: overrides.turn_id ?? "turn-1",
    sequence: overrides.sequence ?? 1,
    type: "agent_message",
    text:
      "text" in overrides && typeof overrides.text === "string"
        ? overrides.text
        : "默认 thread item",
    status: overrides.status ?? "in_progress",
    started_at: overrides.started_at ?? "2026-07-01T00:00:00.000Z",
    completed_at: overrides.completed_at,
    updated_at: overrides.updated_at ?? "2026-07-01T00:00:01.000Z",
    ...overrides,
  } as AgentThreadItem;
}

describe("agentSessionTimelineMergePolicy", () => {
  it("应把首轮 preparing assistant 识别为本地活动快照", () => {
    expect(
      hasAssistantActivitySnapshot([
        createMessage({
          id: "user-1",
          role: "user",
          content: "你好",
        }),
        createMessage({
          id: "assistant-preparing",
          role: "assistant",
          content: "",
          isThinking: true,
          runtimeStatus: {
            phase: "preparing",
            title: "正在准备回复",
            detail: "正在建立运行时上下文",
          },
        }),
      ]),
    ).toBe(true);
  });

  it("纯文本历史 assistant 不应被识别为本地活动快照", () => {
    expect(
      hasAssistantActivitySnapshot([
        createMessage({
          id: "assistant-history",
          role: "assistant",
          content: "历史回复",
        }),
      ]),
    ).toBe(false);
  });

  it("已停止终态 marker 应被识别为可保留的本地快照", () => {
    expect(
      hasAssistantActivitySnapshot([
        createMessage({
          id: "assistant-canceled",
          role: "assistant",
          content: "(已停止)",
          contentParts: [{ type: "text", text: "(已停止)" }],
          isThinking: false,
          runtimeStatus: undefined,
          runtimeTurnId: "pending-turn:cancelled",
        }),
      ]),
    ).toBe(true);
  });

  it("旧空 state 不应覆盖已经写入 ref 的活跃本地发送预览", () => {
    const activePreview = [
      createMessage({
        id: "user-1",
        role: "user",
        content: "你好",
      }),
      createMessage({
        id: "assistant-preparing",
        role: "assistant",
        content: "",
        runtimeStatus: {
          phase: "preparing",
          title: "正在准备回复",
          detail: "正在建立运行时上下文",
        },
      }),
    ];

    expect(
      shouldSkipStaleEmptyMessagesRefSync({
        nextMessages: [],
        currentRefMessages: activePreview,
      }),
    ).toBe(true);
  });

  it("没有本地活动 assistant 时，空 state 仍可同步到 ref", () => {
    expect(
      shouldSkipStaleEmptyMessagesRefSync({
        nextMessages: [],
        currentRefMessages: [
          createMessage({
            id: "assistant-history",
            role: "assistant",
            content: "历史回复",
          }),
        ],
      }),
    ).toBe(false);
  });

  it("history_hydrate 兼容同一会话时应允许保留本地 timeline 状态", () => {
    const decision = resolveAgentSessionTimelineMergeDecision({
      mode: "history_hydrate",
      mayPreserveExistingTimelineBySession: true,
      localMessages: [
        createMessage({
          role: "user",
          content: "帮我整理今天的国际新闻",
        }),
      ],
      hydratedMessagesForCompatibility: [
        createMessage({
          role: "user",
          content: "整理今天的国际新闻",
        }),
      ],
      threadRead: {
        thread_id: "thread-1",
        status: "idle",
      },
      incomingTurns: [],
    });

    expect(decision).toMatchObject({
      mode: "history_hydrate",
      isLocalTimelineCompatible: true,
      shouldPreserveByRuntimeSync: false,
      shouldPreserveBySession: true,
      shouldIgnoreIncompatibleHydratedMessages: false,
    });
  });

  it("history_hydrate 不兼容历史时不应保留本地 timeline", () => {
    const decision = resolveAgentSessionTimelineMergeDecision({
      mode: "history_hydrate",
      mayPreserveExistingTimelineBySession: true,
      localMessages: [
        createMessage({
          role: "user",
          content: "当前会话问题",
        }),
      ],
      hydratedMessagesForCompatibility: [
        createMessage({
          role: "user",
          content: "另一个历史会话问题",
        }),
      ],
      threadRead: {
        thread_id: "thread-1",
        status: "idle",
      },
      incomingTurns: [],
    });

    expect(decision).toMatchObject({
      mode: "history_hydrate",
      isLocalTimelineCompatible: false,
      shouldPreserveByRuntimeSync: false,
      shouldPreserveBySession: false,
      shouldIgnoreIncompatibleHydratedMessages: false,
    });
  });

  it("runtime_sync 非终态且历史不兼容时应保留本地 timeline", () => {
    const decision = resolveAgentSessionTimelineMergeDecision({
      mode: "runtime_sync",
      mayPreserveExistingTimelineBySession: true,
      localMessages: [
        createMessage({
          role: "user",
          content: "当前正在运行的问题",
        }),
      ],
      hydratedMessagesForCompatibility: [
        createMessage({
          role: "user",
          content: "另一条历史问题",
        }),
      ],
      threadRead: {
        thread_id: "thread-1",
        status: "running",
      },
      incomingTurns: [createTurn({ status: "running" })],
    });

    expect(decision).toMatchObject({
      mode: "runtime_sync",
      hasIncomingTerminalTimeline: false,
      isLocalTimelineCompatible: false,
      shouldPreserveByRuntimeSync: true,
      shouldPreserveBySession: true,
      shouldIgnoreIncompatibleHydratedMessages: true,
    });
  });

  it("runtime_sync 终态 detail 应允许服务端 transcript 接管 pending 壳", () => {
    const decision = resolveAgentSessionTimelineMergeDecision({
      mode: "runtime_sync",
      mayPreserveExistingTimelineBySession: true,
      localMessages: [
        createMessage({
          role: "user",
          content: "pending 壳临时问题",
        }),
      ],
      hydratedMessagesForCompatibility: [
        createMessage({
          role: "user",
          content: "最终用户问题",
        }),
      ],
      threadRead: {
        thread_id: "thread-1",
        status: "completed",
      },
      incomingTurns: [createTurn({ status: "completed" })],
    });

    expect(decision).toMatchObject({
      mode: "runtime_sync",
      hasIncomingTerminalTimeline: true,
      isLocalTimelineCompatible: false,
      shouldPreserveByRuntimeSync: false,
      shouldPreserveBySession: false,
      shouldIgnoreIncompatibleHydratedMessages: false,
    });
  });

  it("runtime_sync 应把 thread_read.turns 里的终态视为 terminal timeline", () => {
    const decision = resolveAgentSessionTimelineMergeDecision({
      mode: "runtime_sync",
      mayPreserveExistingTimelineBySession: true,
      localMessages: [
        createMessage({
          role: "user",
          content: "pending 壳临时问题",
        }),
      ],
      hydratedMessagesForCompatibility: [
        createMessage({
          role: "user",
          content: "最终用户问题",
        }),
      ],
      threadRead: {
        thread_id: "thread-1",
        status: "idle",
        turns: [
          {
            turn_id: "turn-1",
            status: "completed",
          },
        ],
      },
      incomingTurns: [],
    });

    expect(decision).toMatchObject({
      mode: "runtime_sync",
      hasIncomingTerminalTimeline: true,
      shouldPreserveByRuntimeSync: false,
      shouldIgnoreIncompatibleHydratedMessages: false,
    });
  });

  it("runtime_sync 不应仅凭 diagnostics 最新 turn 终态接管 pending 壳", () => {
    const decision = resolveAgentSessionTimelineMergeDecision({
      mode: "runtime_sync",
      mayPreserveExistingTimelineBySession: true,
      localMessages: [
        createMessage({
          role: "user",
          content: "pending 壳临时问题",
        }),
      ],
      hydratedMessagesForCompatibility: [
        createMessage({
          role: "user",
          content: "最终用户问题",
        }),
      ],
      threadRead: {
        thread_id: "thread-1",
        status: "running",
        diagnostics: {
          latest_turn_status: "completed",
          warning_count: 0,
          context_compaction_count: 0,
          failed_tool_call_count: 0,
          failed_command_count: 0,
          pending_request_count: 0,
        },
      },
      incomingTurns: [],
    });

    expect(decision).toMatchObject({
      mode: "runtime_sync",
      hasIncomingTerminalTimeline: false,
      shouldPreserveByRuntimeSync: true,
      shouldIgnoreIncompatibleHydratedMessages: true,
    });
  });

  it("runtime_sync item 合并不应把本地较长 agent_message 截成旧前缀", () => {
    const result = mergeRuntimeSyncThreadItems(
      [
        createAgentMessageItem({
          text: "已经显示的完整流式回答，包含更多后续内容。",
          status: "in_progress",
        }),
      ],
      [
        createAgentMessageItem({
          text: "已经显示的完整流式回答",
          status: "completed",
          updated_at: "2026-07-01T00:00:05.000Z",
        }),
      ],
    );

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({
      id: "item-1",
      status: "completed",
      text: "已经显示的完整流式回答，包含更多后续内容。",
      updated_at: "2026-07-01T00:00:05.000Z",
    });
  });

  it("read model 的 partial 应覆盖终态事件留下的停止标记正文", () => {
    const result = mergeRuntimeSyncThreadItems(
      [
        createAgentMessageItem({
          text: "(已停止)",
          status: "completed",
        }),
      ],
      [
        createAgentMessageItem({
          text: "以下是今日国际新闻简要整理：",
          status: "completed",
        }),
      ],
    );

    expect(result[0]).toMatchObject({
      text: "以下是今日国际新闻简要整理：",
      status: "completed",
    });
  });

  it("read model 的 partial 与本地停止标记合并时应保留更长正文", () => {
    const result = mergeRuntimeSyncThreadItems(
      [
        createAgentMessageItem({
          text: "以下是今日国际新闻简要整理：\n\n(已停止)",
          status: "completed",
        }),
      ],
      [
        createAgentMessageItem({
          text: "(已停止)",
          status: "completed",
        }),
      ],
    );

    expect(result[0]?.text).toBe(
      "以下是今日国际新闻简要整理：\n\n(已停止)",
    );
  });
});
