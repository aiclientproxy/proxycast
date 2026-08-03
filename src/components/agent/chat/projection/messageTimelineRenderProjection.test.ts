import { describe, expect, it } from "vitest";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import {
  buildCurrentTurnTimelineProjection,
  buildMessageGroupsProjection,
  buildMessageRenderGroupsProjection,
  buildTimelineByMessageIdProjection,
  type CurrentTurnTimelineProjection,
  resolveLastAssistantMessage,
} from "./messageTimelineRenderProjection";

const TOOL_START_TIME = new Date("2026-06-20T12:00:00.000Z");

function message(id: string, role: "user" | "assistant"): Message {
  return {
    id,
    role,
    content: id,
    timestamp: new Date(`2026-05-05T00:00:0${id.slice(-1)}.000Z`),
  } as Message;
}

function messageAt(
  id: string,
  role: "user" | "assistant",
  timestamp: string,
  runtimeTurnId?: string,
): Message {
  return {
    id,
    role,
    content: id,
    timestamp: new Date(timestamp),
    runtimeTurnId,
  } as Message;
}

function turn(id: string): AgentThreadTurn {
  return {
    id,
    status: "completed",
    started_at: "2026-05-05T00:00:00.000Z",
  } as AgentThreadTurn;
}

function item(id: string, turnId: string): AgentThreadItem {
  return {
    id,
    turn_id: turnId,
    type: "tool_call",
    sequence: 1,
    started_at: "2026-05-05T00:00:00.000Z",
  } as AgentThreadItem;
}

function internalSummaryItem(id: string, turnId: string): AgentThreadItem {
  return {
    id,
    thread_id: "thread-1",
    turn_id: turnId,
    type: "turn_summary",
    sequence: 1,
    status: "completed",
    started_at: "2026-05-05T00:00:00.000Z",
    updated_at: "2026-05-05T00:00:00.000Z",
    text: "runtime status should stay outside the conversation timeline",
    metadata: {
      sourceType: "runtime_status",
      surface: "runtime_status",
      visibility: "diagnostics",
    },
  } as AgentThreadItem;
}

describe("messageTimelineRenderProjection", () => {
  it("不允许构建历史 timeline 时应返回空映射", () => {
    expect(
      buildTimelineByMessageIdProjection({
        canBuildHistoricalTimeline: false,
        renderedMessages: [message("message-1", "assistant")],
        renderedTurns: [turn("turn-1")],
        renderedThreadItems: [item("item-1", "turn-1")],
      }).size,
    ).toBe(0);
  });

  it("应解析最后一条 assistant 消息", () => {
    expect(
      resolveLastAssistantMessage([
        message("message-1", "assistant"),
        message("message-2", "user"),
        message("message-3", "assistant"),
      ])?.id,
    ).toBe("message-3");
  });

  it("当前 turn 未映射到消息时应挂到最后一条 assistant 消息", () => {
    const projection = buildCurrentTurnTimelineProjection({
      activeCurrentTurnId: "turn-current",
      activeCurrentTurn: turn("turn-current"),
      lastAssistantMessageId: "message-tail",
      timelineByMessageId: new Map(),
      renderedThreadItems: [
        item("item-1", "turn-current"),
        item("item-2", "turn-other"),
      ],
    });

    expect(projection).toMatchObject({
      messageId: "message-tail",
      turn: { id: "turn-current" },
      items: [{ id: "item-1" }],
    });
  });

  it("当前 turn timeline 不应把内部路由摘要暴露给默认聊天流", () => {
    const projection = buildCurrentTurnTimelineProjection({
      activeCurrentTurnId: "turn-current",
      activeCurrentTurn: turn("turn-current"),
      lastAssistantMessageId: "message-tail",
      timelineByMessageId: new Map(),
      renderedThreadItems: [
        internalSummaryItem("summary-internal", "turn-current"),
        item("tool-visible", "turn-current"),
      ],
    });

    expect(projection?.items).toEqual([
      expect.objectContaining({ id: "tool-visible" }),
    ]);
  });

  it("当前 turn 已显式绑定其它消息时不应回退到最后一条 assistant", () => {
    const projection = buildCurrentTurnTimelineProjection({
      activeCurrentTurnId: "turn-current",
      activeCurrentTurn: turn("turn-current"),
      lastAssistantMessageId: "assistant-old",
      timelineByMessageId: new Map(),
      renderedMessages: [
        messageAt("user-old", "user", "2026-05-05T00:00:00.000Z"),
        messageAt(
          "assistant-old",
          "assistant",
          "2026-05-05T00:00:01.000Z",
          "turn-old",
        ),
        messageAt("user-current", "user", "2026-05-05T00:00:02.000Z"),
      ],
      renderedThreadItems: [item("item-1", "turn-current")],
    });

    expect(projection).toBeNull();
  });

  it("当前 turn 应优先使用 assistant message 的显式 runtimeTurnId 绑定", () => {
    const projection = buildCurrentTurnTimelineProjection({
      activeCurrentTurnId: "turn-current",
      activeCurrentTurn: turn("turn-current"),
      lastAssistantMessageId: "assistant-old",
      timelineByMessageId: new Map(),
      renderedMessages: [
        messageAt("user-old", "user", "2026-05-05T00:00:00.000Z"),
        messageAt("assistant-old", "assistant", "2026-05-05T00:00:01.000Z"),
        messageAt("user-current", "user", "2026-05-05T00:00:02.000Z"),
        messageAt(
          "assistant-current",
          "assistant",
          "2026-05-05T00:00:03.000Z",
          "turn-current",
        ),
      ],
      renderedThreadItems: [
        item("item-1", "turn-current"),
        item("item-2", "turn-old"),
      ],
    });

    expect(projection).toMatchObject({
      messageId: "assistant-current",
      turn: { id: "turn-current" },
      items: [{ id: "item-1" }],
    });
  });

  it("turn 记录暂缺时应按 assistant runtimeTurnId 保留过程 timeline", () => {
    const projection = buildTimelineByMessageIdProjection({
      canBuildHistoricalTimeline: true,
      renderedMessages: [
        messageAt("user-1", "user", "2026-05-05T00:00:00.000Z"),
        messageAt(
          "assistant-1",
          "assistant",
          "2026-05-05T00:00:02.000Z",
          "turn-runtime-1",
        ),
      ],
      renderedTurns: [],
      renderedThreadItems: [
        {
          id: "reasoning-runtime-1",
          thread_id: "thread-1",
          turn_id: "turn-runtime-1",
          sequence: 1,
          status: "completed",
          started_at: "2026-05-05T00:00:01.000Z",
          completed_at: "2026-05-05T00:00:02.000Z",
          updated_at: "2026-05-05T00:00:02.000Z",
          type: "reasoning",
          text: "先分析。",
        },
      ],
    });

    expect(projection.get("assistant-1")).toMatchObject({
      messageId: "assistant-1",
      turn: {
        id: "turn-runtime-1",
        thread_id: "thread-1",
        status: "completed",
      },
      items: [{ id: "reasoning-runtime-1", type: "reasoning" }],
    });
  });

  it("assistant 已有实时工具过程但 runtimeTurnId 未对齐时应按工具 ID 绑定持久化 timeline", () => {
    const projection = buildTimelineByMessageIdProjection({
      canBuildHistoricalTimeline: true,
      renderedMessages: [
        messageAt("user-web-tools", "user", "2026-06-20T12:00:00.000Z"),
        {
          ...messageAt(
            "assistant-web-tools",
            "assistant",
            "2026-06-20T12:00:04.000Z",
            "pending-turn-web-tools",
          ),
          content: "网页搜索渲染结论。",
          contentParts: [
            {
              type: "tool_use",
              toolCall: {
                id: "tool-search-web-tools",
                name: "WebSearch",
                arguments: JSON.stringify({ query: "Lime WebSearch" }),
                status: "completed",
                startTime: TOOL_START_TIME,
              },
            },
            {
              type: "tool_use",
              toolCall: {
                id: "tool-fetch-web-tools",
                name: "WebFetch",
                arguments: JSON.stringify({
                  url: "https://example.com/lime-websearch-rendering",
                }),
                status: "completed",
                startTime: TOOL_START_TIME,
              },
            },
            { type: "text", text: "网页搜索渲染结论。" },
          ],
        } satisfies Message,
      ],
      renderedTurns: [
        {
          ...turn("turn-web-tools"),
          started_at: "2026-06-20T12:00:00.000Z",
          completed_at: "2026-06-20T12:00:04.000Z",
        },
      ],
      renderedThreadItems: [
        {
          id: "tool-search-web-tools",
          thread_id: "thread-web-tools",
          turn_id: "turn-web-tools",
          sequence: 1,
          status: "completed",
          started_at: "2026-06-20T12:00:00.200Z",
          completed_at: "2026-06-20T12:00:00.500Z",
          updated_at: "2026-06-20T12:00:00.500Z",
          type: "tool_call",
          tool_name: "WebSearch",
        },
        {
          id: "reasoning-web-tools",
          thread_id: "thread-web-tools",
          turn_id: "turn-web-tools",
          sequence: 2,
          status: "completed",
          started_at: "2026-06-20T12:00:00.600Z",
          completed_at: "2026-06-20T12:00:00.900Z",
          updated_at: "2026-06-20T12:00:00.900Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        },
        {
          id: "tool-fetch-web-tools",
          thread_id: "thread-web-tools",
          turn_id: "turn-web-tools",
          sequence: 3,
          status: "completed",
          started_at: "2026-06-20T12:00:01.000Z",
          completed_at: "2026-06-20T12:00:01.400Z",
          updated_at: "2026-06-20T12:00:01.400Z",
          type: "tool_call",
          tool_name: "WebFetch",
        },
      ],
    });

    expect(projection.get("assistant-web-tools")).toMatchObject({
      messageId: "assistant-web-tools",
      turn: { id: "turn-web-tools" },
      items: [
        { id: "tool-search-web-tools" },
        { id: "reasoning-web-tools", type: "reasoning" },
        { id: "tool-fetch-web-tools" },
      ],
    });
  });

  it("应为消息组补齐 timeline 与 active 标记", () => {
    const messages = [
      message("message-user", "user"),
      message("message-assistant", "assistant"),
    ];
    const groups = buildMessageGroupsProjection(messages);
    const renderGroups = buildMessageRenderGroupsProjection({
      messageGroups: groups,
      timelineByMessageId: new Map([
        [
          "message-assistant",
          {
            messageId: "message-assistant",
            turn: turn("turn-1"),
            items: [item("item-1", "turn-1")],
          },
        ],
      ]),
      currentTurnTimeline: null,
      lastAssistantMessageId: "message-assistant",
    });

    expect(renderGroups).toHaveLength(1);
    expect(renderGroups[0]).toMatchObject({
      lastAssistantId: "message-assistant",
      timelineMessageId: "message-assistant",
      isActiveGroup: true,
      timeline: {
        messageId: "message-assistant",
        turn: { id: "turn-1" },
      },
    });
  });

  it("当前 turn 已显式映射到较早 assistant 时应挂回该消息所在组", () => {
    const messages = [
      messageAt("user-earlier", "user", "2026-05-05T00:00:00.000Z"),
      messageAt(
        "assistant-earlier",
        "assistant",
        "2026-05-05T00:00:01.000Z",
        "turn-current",
      ),
      messageAt("assistant-latest", "assistant", "2026-05-05T00:00:02.000Z"),
    ];
    const groups = buildMessageGroupsProjection(messages);
    const currentTurnTimeline = buildCurrentTurnTimelineProjection({
      activeCurrentTurnId: "turn-current",
      activeCurrentTurn: { ...turn("turn-current"), status: "running" },
      lastAssistantMessageId: "assistant-latest",
      timelineByMessageId: new Map(),
      renderedMessages: messages,
      renderedThreadItems: [item("item-current", "turn-current")],
    });

    const renderGroups = buildMessageRenderGroupsProjection({
      messageGroups: groups,
      timelineByMessageId: new Map(),
      currentTurnTimeline,
      lastAssistantMessageId: "assistant-latest",
    });

    expect(currentTurnTimeline).toMatchObject({
      messageId: "assistant-earlier",
      items: [{ id: "item-current" }],
    });
    expect(renderGroups).toHaveLength(1);
    expect(renderGroups[0]).toMatchObject({
      lastAssistantId: "assistant-latest",
      timelineMessageId: "assistant-earlier",
      timeline: {
        messageId: "assistant-earlier",
        turn: { id: "turn-current" },
      },
      isActiveGroup: true,
    });
  });

  it("同一 turn 映射到多个消息组时只保留一个 timeline owner", () => {
    const messages = [
      messageAt("user-first", "user", "2026-05-05T00:00:00.000Z"),
      messageAt("assistant-first", "assistant", "2026-05-05T00:00:01.000Z"),
      messageAt("user-second", "user", "2026-05-05T00:00:02.000Z"),
      messageAt("assistant-second", "assistant", "2026-05-05T00:00:03.000Z"),
    ];
    const groups = buildMessageGroupsProjection(messages);
    const sharedTimeline = {
      turn: turn("turn-shared"),
      items: [item("item-shared", "turn-shared")],
    };

    const renderGroups = buildMessageRenderGroupsProjection({
      messageGroups: groups,
      timelineByMessageId: new Map([
        [
          "assistant-first",
          { ...sharedTimeline, messageId: "assistant-first" },
        ],
        [
          "assistant-second",
          { ...sharedTimeline, messageId: "assistant-second" },
        ],
      ]),
      currentTurnTimeline: null,
      lastAssistantMessageId: "assistant-second",
    });

    expect(
      renderGroups.map((group) => group.timeline?.turn.id ?? null),
    ).toEqual(["turn-shared", null]);
    expect(renderGroups[1]?.timelineMessageId).toBeNull();
  });

  it("当前 turn 与历史映射指向同一消息时应优先保留完整持久化 timeline", () => {
    const messages = [
      messageAt("user-web-tools", "user", "2026-06-20T12:00:00.000Z"),
      messageAt(
        "assistant-web-tools",
        "assistant",
        "2026-06-20T12:00:04.000Z",
        "pending-turn-web-tools",
      ),
    ];
    const groups = buildMessageGroupsProjection(messages);
    const persistedTimeline = {
      messageId: "assistant-web-tools",
      turn: {
        ...turn("turn-web-tools"),
        status: "completed",
      } satisfies AgentThreadTurn,
      items: [
        item("tool-search-web-tools", "turn-web-tools"),
        {
          id: "reasoning-web-tools",
          thread_id: "thread-web-tools",
          turn_id: "turn-web-tools",
          sequence: 2,
          status: "completed",
          started_at: "2026-06-20T12:00:00.600Z",
          completed_at: "2026-06-20T12:00:00.900Z",
          updated_at: "2026-06-20T12:00:00.900Z",
          type: "reasoning",
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
        } as AgentThreadItem,
        item("tool-fetch-web-tools", "turn-web-tools"),
      ],
    };
    const currentTurnTimeline: CurrentTurnTimelineProjection = {
      messageId: "assistant-web-tools",
      turn: { ...turn("turn-web-tools"), status: "running" },
      items: [
        item("tool-search-web-tools", "turn-web-tools"),
        item("tool-fetch-web-tools", "turn-web-tools"),
      ],
    };

    const renderGroups = buildMessageRenderGroupsProjection({
      messageGroups: groups,
      timelineByMessageId: new Map([
        ["assistant-web-tools", persistedTimeline],
      ]),
      currentTurnTimeline,
      lastAssistantMessageId: "assistant-web-tools",
    });

    expect(renderGroups[0]?.timeline).toBe(persistedTimeline);
    expect(renderGroups[0]?.timeline?.items.map((entry) => entry.id)).toEqual([
      "tool-search-web-tools",
      "reasoning-web-tools",
      "tool-fetch-web-tools",
    ]);
  });
});
