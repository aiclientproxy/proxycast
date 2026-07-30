import { describe, expect, it } from "vitest";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import {
  buildMessageGroupsProjection,
  buildMessageRenderGroupsProjection,
  buildTimelineByMessageIdProjection,
} from "../projection/messageTimelineRenderProjection";
import { resolveMessageListItemProjection } from "../components/messageListItemProjection";
import { buildHydratedAgentSessionSnapshot } from "./agentSessionState";

function createMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: overrides.id ?? "message-1",
    role: overrides.role ?? "assistant",
    content: overrides.content ?? "默认内容",
    timestamp: overrides.timestamp ?? new Date("2026-03-29T00:00:00.000Z"),
    ...overrides,
  };
}

function createTurn(overrides: Partial<AgentThreadTurn> = {}): AgentThreadTurn {
  return {
    id: overrides.id ?? "turn-1",
    thread_id: overrides.thread_id ?? "thread-1",
    status: overrides.status ?? "completed",
    prompt_text: overrides.prompt_text ?? "默认 turn",
    started_at: overrides.started_at ?? "2026-03-29T00:00:00.000Z",
    completed_at: overrides.completed_at ?? "2026-03-29T00:00:02.000Z",
    created_at: overrides.created_at ?? "2026-03-29T00:00:00.000Z",
    updated_at: overrides.updated_at ?? "2026-03-29T00:00:02.000Z",
    ...overrides,
  };
}

function createItem(overrides: Partial<AgentThreadItem> = {}): AgentThreadItem {
  return {
    id: overrides.id ?? "item-1",
    thread_id: overrides.thread_id ?? "thread-1",
    turn_id: overrides.turn_id ?? "turn-1",
    sequence: overrides.sequence ?? 1,
    type: overrides.type ?? "agent_message",
    text:
      "text" in overrides && typeof overrides.text === "string"
        ? overrides.text
        : "默认 item",
    status: overrides.status ?? "completed",
    started_at: overrides.started_at ?? "2026-03-29T00:00:00.000Z",
    completed_at: overrides.completed_at ?? "2026-03-29T00:00:02.000Z",
    updated_at: overrides.updated_at ?? "2026-03-29T00:00:02.000Z",
    ...overrides,
  } as AgentThreadItem;
}

describe("agentSessionState WebTools hydrate", () => {
  it("hydrate App Server WebSearch/WebFetch 时由 threadItems 保留中间 reasoning 顺序", () => {
    const turnId = "turn-web-tools-hydrate";
    const detail = {
      id: "topic-web-tools-hydrate",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [],
      turns: [
        createTurn({
          id: turnId,
          thread_id: "topic-web-tools-hydrate",
          prompt_text: "验证网页搜索渲染",
          started_at: "2026-06-20T10:00:00.000Z",
          completed_at: "2026-06-20T10:00:04.000Z",
          updated_at: "2026-06-20T10:00:04.000Z",
        }),
      ],
      items: [
        createItem({
          id: "user-web-tools-hydrate",
          type: "user_message",
          turn_id: turnId,
          sequence: 1,
          content: "验证网页搜索渲染",
          started_at: "2026-06-20T10:00:00.000Z",
          completed_at: "2026-06-20T10:00:00.000Z",
          updated_at: "2026-06-20T10:00:00.000Z",
        }),
        createItem({
          id: "tool-search-hydrate",
          type: "tool_call",
          turn_id: turnId,
          sequence: 2,
          tool_name: "WebSearch",
          arguments: { query: "Lime WebSearch rendering" },
          output: "Lime WebSearch Rendering Source",
          success: true,
          metadata: { sequence: 2 },
          status: "completed",
          started_at: "2026-06-20T10:00:01.000Z",
          completed_at: "2026-06-20T10:00:01.200Z",
          updated_at: "2026-06-20T10:00:01.200Z",
        } as Partial<AgentThreadItem>),
        createItem({
          id: "reasoning-web-tools-hydrate",
          type: "reasoning",
          turn_id: turnId,
          sequence: 3,
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
          metadata: { sequence: 3 },
          status: "completed",
          started_at: "2026-06-20T10:00:02.000Z",
          completed_at: "2026-06-20T10:00:02.100Z",
          updated_at: "2026-06-20T10:00:02.100Z",
        }),
        createItem({
          id: "tool-fetch-hydrate",
          type: "tool_call",
          turn_id: turnId,
          sequence: 4,
          tool_name: "WebFetch",
          arguments: {
            url: "https://example.com/lime-websearch-rendering",
          },
          output: "# 五年级选购指南",
          success: true,
          metadata: { sequence: 4 },
          status: "completed",
          started_at: "2026-06-20T10:00:03.000Z",
          completed_at: "2026-06-20T10:00:03.200Z",
          updated_at: "2026-06-20T10:00:03.200Z",
        } as Partial<AgentThreadItem>),
        createItem({
          id: "assistant-web-tools-final",
          type: "agent_message",
          turn_id: turnId,
          sequence: 5,
          phase: "final_answer",
          text: "网页搜索渲染结论。",
          status: "completed",
          started_at: "2026-06-20T10:00:04.000Z",
          completed_at: "2026-06-20T10:00:04.000Z",
          updated_at: "2026-06-20T10:00:04.000Z",
        } as Partial<AgentThreadItem>),
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-web-tools-hydrate",
      detail,
      currentSessionId: null,
      currentMessages: [],
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
      syncSessionId: true,
    });

    expect(result.snapshot.messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(
      result.snapshot.messages[1]?.contentParts?.map((part) => part.type),
    ).toEqual(["text"]);
    expect(result.snapshot.threadItems.map((item) => item.id)).toEqual([
      "user-web-tools-hydrate",
      "tool-search-hydrate",
      "reasoning-web-tools-hydrate",
      "tool-fetch-hydrate",
      "assistant-web-tools-final",
    ]);
  });

  it("同会话 hydrate 应保留 canonical WebTools 顺序，但历史 GUI 只投影 final", () => {
    const turnId = "turn-web-tools-current";
    const topicId = "topic-web-tools-current";
    const currentMessages: Message[] = [
      createMessage({
        id: "local-user-web-tools-current",
        role: "user",
        content: "验证网页搜索渲染",
        timestamp: new Date("2026-06-20T13:00:00.000Z"),
      }),
      createMessage({
        id: "local-assistant-web-tools-current",
        role: "assistant",
        content: "我先联网核实目标页面来源。\n\n网页搜索渲染结论。",
        runtimeTurnId: turnId,
        timestamp: new Date("2026-06-20T13:00:04.000Z"),
        contentParts: [
          {
            type: "text",
            text: "我先联网核实目标页面来源。",
            metadata: {
              source: "agent_text_delta",
              itemId: "commentary-web-tools-current",
              phase: "commentary",
              sequence: 1,
              turnId,
            },
          },
          {
            type: "tool_use",
            toolCall: {
              id: "tool-search-current",
              name: "WebSearch",
              arguments: JSON.stringify({ query: "Lime WebSearch rendering" }),
              status: "completed",
              startTime: new Date("2026-06-20T13:00:01.000Z"),
              endTime: new Date("2026-06-20T13:00:01.200Z"),
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
            },
          },
          {
            type: "tool_use",
            toolCall: {
              id: "tool-fetch-current",
              name: "WebFetch",
              arguments: JSON.stringify({
                url: "https://example.com/lime-websearch-rendering",
              }),
              status: "completed",
              startTime: new Date("2026-06-20T13:00:03.000Z"),
              endTime: new Date("2026-06-20T13:00:03.200Z"),
              result: {
                success: true,
                output: "# 五年级选购指南",
              },
            },
          },
          {
            type: "text",
            text: "网页搜索渲染结论。",
          },
        ],
      }),
    ];
    const detail = {
      id: topicId,
      thread_id: topicId,
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1781982000,
          content: [{ type: "text", text: "验证网页搜索渲染" }],
        },
        {
          role: "assistant",
          timestamp: 1781982004,
          content: [{ type: "text", text: "网页搜索渲染结论。" }],
        },
      ],
      turns: [
        createTurn({
          id: turnId,
          thread_id: topicId,
          prompt_text: "验证网页搜索渲染",
          status: "completed",
          started_at: "2026-06-20T13:00:00.000Z",
          completed_at: "2026-06-20T13:00:04.000Z",
          updated_at: "2026-06-20T13:00:04.000Z",
        }),
      ],
      items: [
        createItem({
          id: "commentary-web-tools-current",
          type: "agent_message",
          turn_id: turnId,
          sequence: 1,
          text: "我先联网核实目标页面来源。",
          phase: "commentary",
          status: "completed",
          started_at: "2026-06-20T13:00:00.500Z",
          completed_at: "2026-06-20T13:00:00.800Z",
          updated_at: "2026-06-20T13:00:00.800Z",
        } as Partial<AgentThreadItem>),
        createItem({
          id: "tool-search-current",
          type: "tool_call",
          turn_id: turnId,
          sequence: 2,
          tool_name: "WebSearch",
          arguments: { query: "Lime WebSearch rendering" },
          output: "Lime WebSearch Rendering Source",
          success: true,
          metadata: { sequence: 2 },
          status: "completed",
          started_at: "2026-06-20T13:00:01.000Z",
          completed_at: "2026-06-20T13:00:01.200Z",
          updated_at: "2026-06-20T13:00:01.200Z",
        } as Partial<AgentThreadItem>),
        createItem({
          id: "reasoning-web-tools-current",
          type: "reasoning",
          turn_id: turnId,
          sequence: 3,
          text: "搜索结果还需要继续筛掉广告软文，我先读取有效来源。",
          metadata: { sequence: 3 },
          status: "completed",
          started_at: "2026-06-20T13:00:02.000Z",
          completed_at: "2026-06-20T13:00:02.100Z",
          updated_at: "2026-06-20T13:00:02.100Z",
        }),
        createItem({
          id: "tool-fetch-current",
          type: "tool_call",
          turn_id: turnId,
          sequence: 4,
          tool_name: "WebFetch",
          arguments: {
            url: "https://example.com/lime-websearch-rendering",
          },
          output: "# 五年级选购指南",
          success: true,
          metadata: { sequence: 4 },
          status: "completed",
          started_at: "2026-06-20T13:00:03.000Z",
          completed_at: "2026-06-20T13:00:03.200Z",
          updated_at: "2026-06-20T13:00:03.200Z",
        } as Partial<AgentThreadItem>),
        createItem({
          id: "summary-web-tools-current",
          type: "turn_summary",
          turn_id: turnId,
          sequence: 5,
          text: "已搜索网页 1 次，读取网页 1 次",
          status: "completed",
          started_at: "2026-06-20T13:00:00.000Z",
          completed_at: "2026-06-20T13:00:04.000Z",
          updated_at: "2026-06-20T13:00:04.000Z",
        } as Partial<AgentThreadItem>),
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId,
      detail,
      currentSessionId: topicId,
      currentMessages,
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
    });
    expect(result.snapshot.messages[1]?.content).toContain(
      "我先联网核实目标页面来源。",
    );
    expect(result.snapshot.messages[1]?.content).toContain("网页搜索渲染结论");
    expect(
      result.snapshot.messages[1]?.contentParts?.map((part) => part.type),
    ).toEqual(["text", "tool_use", "tool_use", "text"]);
    const messageGroups = buildMessageGroupsProjection(
      result.snapshot.messages,
    );
    const timelineByMessageId = buildTimelineByMessageIdProjection({
      canBuildHistoricalTimeline: true,
      renderedMessages: result.snapshot.messages,
      renderedTurns: result.snapshot.threadTurns,
      renderedThreadItems: result.snapshot.threadItems,
    });
    const renderGroups = buildMessageRenderGroupsProjection({
      messageGroups,
      timelineByMessageId,
      currentTurnTimeline: null,
      lastAssistantMessageId: "local-assistant-web-tools-current",
    });
    const renderGroup = renderGroups[0];

    expect(renderGroup?.timelineMessageId).toBe(
      "local-assistant-web-tools-current",
    );
    expect(renderGroup?.timeline?.items.map((item) => item.id)).toEqual([
      "commentary-web-tools-current",
      "tool-search-current",
      "reasoning-web-tools-current",
      "tool-fetch-current",
      "summary-web-tools-current",
    ]);

    const projection = resolveMessageListItemProjection({
      activeCurrentTurnId: null,
      activePendingA2UISource: null,
      canOpenSavedSiteContent: false,
      expandedHistoricalAssistantMessageIds: new Set(),
      expandedLongHistoricalMessageIds: new Set(),
      group: renderGroup as never,
      hasActiveInteractiveRuntime: false,
      isRestoredHistoryWindow: false,
      isSending: false,
      lastAssistantMessageId: "local-assistant-web-tools-current",
      message: result.snapshot.messages[1] as Message,
      shouldDeferHistoricalAssistantMessageDetails: () => false,
      shouldDeferThreadItemsScan: false,
      streamingTextOverlay: null,
    });

    expect(projection.rendererContentParts?.map((part) => part.type)).toEqual([
      "text",
    ]);
    expect(projection.actionContent).toBe("网页搜索渲染结论。");
    expect(projection.shouldRenderCompactPrimaryTimeline).toBe(true);
    expect(projection.primaryTimeline?.items.map((item) => item.id)).toEqual([
      "tool-search-current",
      "reasoning-web-tools-current",
      "tool-fetch-current",
    ]);
  });
});
