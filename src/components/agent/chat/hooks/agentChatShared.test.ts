import { describe, expect, it } from "vitest";

import type { AgentSessionInfo } from "@/lib/api/agentRuntime/sessionTypes";
import type { Message } from "../types";
import {
  buildLiveTaskSnapshot,
  deriveTaskLiveState,
  deriveTaskStatusFromLiveState,
  extractTaskPreviewFromMessages,
  mapSessionToTopic,
  resolveRecentTopicActionLabel,
  resolveRecentTopicCandidate,
} from "./agentChatShared";

function createPendingActionMessages(
  prompt = "请先确认发布标题后继续执行。",
): Message[] {
  const startedAt = new Date("2026-03-15T09:45:00.000Z");

  return [
    {
      id: "msg-user",
      role: "user",
      content: "帮我写一篇公众号发布文案",
      timestamp: startedAt,
    },
    {
      id: "msg-pending-action",
      role: "assistant",
      content: "",
      timestamp: new Date(startedAt.getTime() + 1000),
      actionRequests: [
        {
          requestId: "req-user-action-1",
          actionType: "ask_user",
          prompt,
        },
      ],
      contentParts: [
        {
          type: "action_required",
          actionRequired: {
            requestId: "req-user-action-1",
            actionType: "ask_user",
            prompt,
          },
        },
      ],
    },
  ];
}

describe("agentChatShared", () => {
  it("App Server 毫秒时间戳不应在任务标题里显示 Invalid Date", () => {
    const topic = mapSessionToTopic({
      id: "session-ms",
      created_at: 1780847017766,
      updated_at: 1780847020000,
      messages_count: 0,
    });

    expect(topic.title).toBe("任务 2026/6/7");
    expect(topic.title).not.toContain("Invalid Date");
    expect(topic.createdAt.toISOString()).toBe("2026-06-07T15:43:37.766Z");
    expect(topic.updatedAt.toISOString()).toBe("2026-06-07T15:43:40.000Z");
  });

  it("旧秒级时间戳仍应映射为有效任务日期", () => {
    const topic = mapSessionToTopic({
      id: "session-seconds",
      created_at: 1780847017,
      updated_at: 1780847020,
      messages_count: 0,
    });

    expect(topic.title).toBe("任务 2026/6/7");
    expect(topic.createdAt.toISOString()).toBe("2026-06-07T15:43:37.000Z");
    expect(topic.updatedAt.toISOString()).toBe("2026-06-07T15:43:40.000Z");
  });

  it("缺失时间戳时不应把 epoch 日期暴露成任务标题", () => {
    const malformedSession = {
      id: "session-missing-time",
      messages_count: 0,
    } as Partial<AgentSessionInfo> as AgentSessionInfo;
    const topic = mapSessionToTopic(malformedSession);

    expect(topic.title).toBe("新任务");
    expect(topic.title).not.toContain("1970");
  });

  it("App Server list overview 的未完成状态应投影到最近会话 topic", () => {
    const freshUpdatedAt = Date.now();

    expect(
      mapSessionToTopic({
        id: "session-running",
        name: "运行中会话",
        created_at: 1780847017766,
        updated_at: freshUpdatedAt,
        messages_count: 2,
        thread_status: "running",
        latest_turn_status: "accepted",
        active_turn_id: "turn-running",
      }),
    ).toEqual(
      expect.objectContaining({
        status: "running",
        statusReason: "default",
        lastPreview: "正在继续输出。",
      }),
    );

    expect(
      mapSessionToTopic({
        id: "session-waiting",
        name: "待确认会话",
        created_at: 1780847017766,
        updated_at: 1780847020000,
        messages_count: 2,
        thread_status: "waitingAction",
        latest_turn_status: "waitingAction",
        active_turn_id: "turn-waiting",
      }),
    ).toEqual(
      expect.objectContaining({
        status: "waiting",
        statusReason: "user_action",
        lastPreview: "等待你确认后继续。",
      }),
    );

    expect(
      mapSessionToTopic({
        id: "session-queued",
        name: "排队会话",
        created_at: 1780847017766,
        updated_at: 1780847020000,
        messages_count: 2,
        thread_status: "running",
        latest_turn_status: "queued",
        active_turn_id: "turn-running",
        queued_turn_count: 1,
      }),
    ).toEqual(
      expect.objectContaining({
        status: "queued",
        statusReason: "default",
        lastPreview: "仍有排队中的请求。",
      }),
    );
  });

  it("终态 overview 即使带 stale active turn 也不应投影成运行中", () => {
    expect(
      mapSessionToTopic({
        id: "session-completed-stale",
        name: "已完成会话",
        created_at: 1780847017766,
        updated_at: 1780847020000,
        messages_count: 2,
        thread_status: "completed",
        latest_turn_status: "running",
        active_turn_id: "turn-stale",
      }),
    ).toEqual(
      expect.objectContaining({
        status: "done",
        statusReason: "default",
      }),
    );
  });

  it("待处理 action request 未提交时应优先判定为待处理", () => {
    const messages = createPendingActionMessages();

    expect(
      deriveTaskStatusFromLiveState({
        messages,
        isSending: false,
        pendingActionCount: 0,
        workspaceError: false,
      }),
    ).toBe("waiting");

    expect(
      buildLiveTaskSnapshot({
        messages,
        isSending: false,
        pendingActionCount: 0,
        workspaceError: false,
      }),
    ).toEqual(
      expect.objectContaining({
        status: "waiting",
        statusReason: "user_action",
      }),
    );

    expect(
      deriveTaskLiveState({
        messages,
        isSending: false,
        pendingActionCount: 0,
        workspaceError: false,
      }),
    ).toEqual({
      status: "waiting",
      statusReason: "user_action",
    });
  });

  it("应优先展示待处理请求的摘要", () => {
    const messages =
      createPendingActionMessages("请先确认发布标题后继续执行。");

    expect(extractTaskPreviewFromMessages(messages)).toBe(
      "请先确认发布标题后继续执行。",
    );
  });

  it("图片占位符不应直接出现在任务摘要里", () => {
    const now = new Date("2026-03-19T00:00:00.000Z");
    const messages: Message[] = [
      {
        id: "msg-user-image",
        role: "user",
        content: "[Image #1]",
        images: [
          {
            mediaType: "image/png",
            data: "aGVsbG8=",
          },
        ],
        timestamp: now,
      },
    ];

    expect(extractTaskPreviewFromMessages(messages)).toBe("已附加图片");
  });

  it("助手内部图片标签应转换为自然语言摘要", () => {
    const now = new Date("2026-03-19T00:00:00.000Z");
    const messages: Message[] = [
      {
        id: "msg-assistant-image",
        role: "assistant",
        content: "[Image #1]",
        timestamp: now,
      },
    ];

    expect(extractTaskPreviewFromMessages(messages)).toBe("图片处理中");
  });

  it("最新工具调用失败时应映射为工具失败", () => {
    expect(
      deriveTaskLiveState({
        messages: [
          {
            id: "msg-tool-failed",
            role: "assistant",
            content: "",
            timestamp: new Date("2026-03-15T09:45:01.000Z"),
            toolCalls: [
              {
                id: "tool-1",
                name: "write_file",
                arguments: "{}",
                status: "failed",
                startTime: new Date("2026-03-15T09:45:01.000Z"),
                endTime: new Date("2026-03-15T09:45:02.000Z"),
              },
            ],
          },
        ],
        isSending: false,
        pendingActionCount: 0,
        workspaceError: false,
      }),
    ).toEqual({
      status: "failed",
      statusReason: "tool_failure",
    });
  });

  it("同一回合已有最终助手正文时局部工具失败不应覆盖完成态", () => {
    const finalText = "## 今日国际新闻简报\n\n- 要闻摘要，附来源。";

    expect(
      deriveTaskLiveState({
        messages: [
          {
            id: "msg-news-with-partial-tool-failure",
            role: "assistant",
            content: finalText,
            timestamp: new Date("2026-06-02T10:00:20.000Z"),
            contentParts: [
              {
                type: "tool_use",
                toolCall: {
                  id: "tool-search-ok",
                  name: "WebSearch",
                  arguments: '{"query":"international news"}',
                  status: "completed",
                  startTime: new Date("2026-06-02T10:00:01.000Z"),
                  endTime: new Date("2026-06-02T10:00:03.000Z"),
                },
              },
              {
                type: "tool_use",
                toolCall: {
                  id: "tool-fetch-failed",
                  name: "WebFetch",
                  arguments: '{"url":"https://example.invalid/news"}',
                  status: "failed",
                  startTime: new Date("2026-06-02T10:00:04.000Z"),
                  endTime: new Date("2026-06-02T10:00:05.000Z"),
                },
              },
              {
                type: "text",
                text: finalText,
              },
            ],
            toolCalls: [
              {
                id: "tool-search-ok",
                name: "WebSearch",
                arguments: '{"query":"international news"}',
                status: "completed",
                startTime: new Date("2026-06-02T10:00:01.000Z"),
                endTime: new Date("2026-06-02T10:00:03.000Z"),
              },
              {
                id: "tool-fetch-failed",
                name: "WebFetch",
                arguments: '{"url":"https://example.invalid/news"}',
                status: "failed",
                startTime: new Date("2026-06-02T10:00:04.000Z"),
                endTime: new Date("2026-06-02T10:00:05.000Z"),
              },
            ],
          },
        ],
        isSending: false,
        pendingActionCount: 0,
        workspaceError: false,
      }),
    ).toEqual({
      status: "done",
      statusReason: "default",
    });
  });

  it("当前线程仍在运行时，不应把最新 assistant 消息误判为已完成", () => {
    const messages: Message[] = [
      {
        id: "msg-tool-request",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-22T10:59:16.000Z"),
        toolCalls: [
          {
            id: "tool-1",
            name: "WebSearch",
            arguments: '{"query":"AI agent trends"}',
            status: "running",
            startTime: new Date("2026-04-22T10:59:16.000Z"),
          },
        ],
      },
    ];

    expect(
      deriveTaskLiveState({
        messages,
        isSending: false,
        pendingActionCount: 0,
        queuedTurnCount: 0,
        threadStatus: "running",
        workspaceError: false,
      }),
    ).toEqual({
      status: "running",
      statusReason: "default",
    });
  });

  it("当前线程只有排队请求时应投影为排队中", () => {
    expect(
      deriveTaskLiveState({
        messages: [],
        isSending: false,
        pendingActionCount: 0,
        queuedTurnCount: 1,
        threadStatus: "queued",
        workspaceError: false,
      }),
    ).toEqual({
      status: "queued",
      statusReason: "default",
    });
  });

  it("首字前等待中的 assistant 草稿不应因 isSending 清空而误判为已完成", () => {
    const messages: Message[] = [
      {
        id: "msg-user-first-token-waiting",
        role: "user",
        content: "帮我解释启动状态",
        timestamp: new Date("2026-06-07T10:00:00.000Z"),
      },
      {
        id: "msg-assistant-first-token-waiting",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-07T10:00:01.000Z"),
        isThinking: true,
        runtimeStatus: {
          phase: "routing",
          title: "正在生成回复",
          detail: "运行时已开始处理，等待首个输出。",
        },
      },
    ];

    expect(
      deriveTaskLiveState({
        messages,
        isSending: false,
        pendingActionCount: 0,
        queuedTurnCount: 0,
        threadStatus: "completed",
        workspaceError: false,
      }),
    ).toEqual({
      status: "running",
      statusReason: "default",
    });

    expect(
      buildLiveTaskSnapshot({
        messages,
        isSending: false,
        pendingActionCount: 0,
        queuedTurnCount: 0,
        threadStatus: "completed",
        workspaceError: false,
      }),
    ).toEqual(
      expect.objectContaining({
        status: "running",
        lastPreview: "帮我解释启动状态",
      }),
    );
  });

  it("应优先返回最近可继续的会话候选", () => {
    const topics = [
      {
        id: "topic-done",
        title: "最近结果",
        createdAt: new Date("2026-03-15T09:40:00.000Z"),
        updatedAt: new Date("2026-03-15T09:45:00.000Z"),
        workspaceId: "workspace-1",
        messagesCount: 4,
        executionStrategy: "react" as const,
        status: "done" as const,
        statusReason: "default" as const,
        lastPreview: "结果已产出。",
        hasUnread: false,
        tag: null,
        sourceSessionId: "topic-done",
      },
      {
        id: "topic-running",
        title: "运行中任务",
        createdAt: new Date("2026-03-15T09:48:00.000Z"),
        updatedAt: new Date("2026-03-15T09:49:00.000Z"),
        workspaceId: "workspace-1",
        messagesCount: 3,
        executionStrategy: "react" as const,
        status: "running" as const,
        statusReason: "default" as const,
        lastPreview: "正在继续输出。",
        hasUnread: false,
        tag: null,
        sourceSessionId: "topic-running",
      },
      {
        id: "topic-waiting",
        title: "待继续任务",
        createdAt: new Date("2026-03-15T09:46:00.000Z"),
        updatedAt: new Date("2026-03-15T09:50:00.000Z"),
        workspaceId: "workspace-1",
        messagesCount: 2,
        executionStrategy: "react" as const,
        status: "waiting" as const,
        statusReason: "user_action" as const,
        lastPreview: "等待补充标题。",
        hasUnread: false,
        tag: null,
        sourceSessionId: "topic-waiting",
      },
      {
        id: "topic-current",
        title: "当前任务",
        createdAt: new Date("2026-03-15T09:52:00.000Z"),
        updatedAt: new Date("2026-03-15T09:55:00.000Z"),
        workspaceId: "workspace-1",
        messagesCount: 1,
        executionStrategy: "react" as const,
        status: "draft" as const,
        statusReason: "default" as const,
        lastPreview: "当前草稿。",
        hasUnread: false,
        tag: null,
        sourceSessionId: "topic-current",
      },
    ];

    expect(resolveRecentTopicCandidate(topics, "topic-current")?.id).toBe(
      "topic-waiting",
    );
  });

  it("最近会话动作文案应随会话状态变化", () => {
    expect(
      resolveRecentTopicActionLabel({
        status: "waiting",
        statusReason: "user_action",
        messagesCount: 2,
      }),
    ).toBe("继续最近会话");

    expect(
      resolveRecentTopicActionLabel({
        status: "done",
        statusReason: "default",
        messagesCount: 4,
      }),
    ).toBe("回看最近结果");

    expect(
      resolveRecentTopicActionLabel({
        status: "draft",
        statusReason: "default",
        messagesCount: 0,
      }),
    ).toBe("打开最近会话");
  });
});
