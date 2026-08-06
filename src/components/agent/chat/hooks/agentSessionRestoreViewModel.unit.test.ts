import { describe, expect, it } from "vitest";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import type { AgentSessionCachedSnapshot } from "./agentSessionScopedStorage";
import {
  buildAgentSessionRestoreViewModel,
  buildCachedTopicSnapshotViewModel,
} from "./agentSessionRestoreViewModel";
import type { Topic } from "./agentChatShared";

function createMessage(
  index: number,
  overrides: Partial<Message> = {},
): Message {
  return {
    id: overrides.id ?? `message-${index}`,
    role: overrides.role ?? (index % 2 === 0 ? "assistant" : "user"),
    content: overrides.content ?? `message ${index}`,
    timestamp:
      overrides.timestamp ?? new Date(`2026-05-31T00:00:0${index}.000Z`),
    ...overrides,
  };
}

function createTurn(
  index: number,
  overrides: Partial<AgentThreadTurn> = {},
): AgentThreadTurn {
  return {
    id: overrides.id ?? `turn-${index}`,
    thread_id: overrides.thread_id ?? "thread-1",
    prompt_text: overrides.prompt_text ?? `prompt ${index}`,
    status: overrides.status ?? "completed",
    started_at: overrides.started_at ?? "2026-05-31T00:00:00.000Z",
    completed_at: overrides.completed_at ?? "2026-05-31T00:00:01.000Z",
    created_at: overrides.created_at ?? "2026-05-31T00:00:00.000Z",
    updated_at: overrides.updated_at ?? "2026-05-31T00:00:01.000Z",
    ...overrides,
  };
}

function createItem(
  index: number,
  overrides: Partial<AgentThreadItem> = {},
): AgentThreadItem {
  return {
    id: overrides.id ?? `item-${index}`,
    thread_id: overrides.thread_id ?? "thread-1",
    turn_id: overrides.turn_id ?? `turn-${index}`,
    sequence: overrides.sequence ?? index,
    type: overrides.type ?? "turn_summary",
    text:
      "text" in overrides && typeof overrides.text === "string"
        ? overrides.text
        : `item ${index}`,
    status: overrides.status ?? "completed",
    started_at: overrides.started_at ?? "2026-05-31T00:00:00.000Z",
    completed_at: overrides.completed_at ?? "2026-05-31T00:00:01.000Z",
    updated_at: overrides.updated_at ?? "2026-05-31T00:00:01.000Z",
    ...overrides,
  } as AgentThreadItem;
}

function createCachedSnapshot(
  overrides: Partial<AgentSessionCachedSnapshot> = {},
): AgentSessionCachedSnapshot {
  return {
    messages: [createMessage(1, { id: "cached-message", role: "user" })],
    threadTurns: [createTurn(1, { id: "cached-turn" })],
    threadItems: [createItem(1, { id: "cached-item", turn_id: "cached-turn" })],
    currentTurnId: "cached-turn",
    ...overrides,
  };
}

function createTopic(overrides: Partial<Topic> = {}): Topic {
  return {
    id: overrides.id ?? "topic-1",
    title: overrides.title ?? "Topic 1",
    createdAt: overrides.createdAt ?? new Date("2026-05-30T00:00:00.000Z"),
    updatedAt: overrides.updatedAt ?? new Date("2026-05-31T00:00:00.000Z"),
    workspaceId: overrides.workspaceId ?? "workspace-1",
    messagesCount: overrides.messagesCount ?? 1,
    executionStrategy: overrides.executionStrategy ?? "react",
    status: overrides.status ?? "done",
    lastPreview: overrides.lastPreview ?? "preview",
    hasUnread: overrides.hasUnread ?? false,
    tag: overrides.tag ?? null,
    sourceSessionId: overrides.sourceSessionId ?? "topic-1",
  };
}

describe("agentSessionRestoreViewModel", () => {
  it("无 transient messages 时应使用 cached snapshot 但不伪造 history cursor", () => {
    const viewModel = buildAgentSessionRestoreViewModel({
      cachedSnapshot: createCachedSnapshot({
        cacheMetadata: {
          storageKind: "transient",
          freshness: "fresh",
          updatedAt: 1,
          lastAccessedAt: 1,
          expiresAt: 2,
          staleUntil: 3,
          sessionUpdatedAt: 4,
          messagesCount: 10,
          historyTruncated: true,
        },
      }),
      scopedCurrentTurnId: null,
      scopedItems: [],
      scopedMessages: [],
      scopedSessionCandidate: "session-1",
      scopedTurns: [],
    });

    expect(viewModel).toMatchObject({
      sessionId: "session-1",
      currentTurnId: "cached-turn",
      historyWindow: null,
    });
    expect(viewModel.messages.map((message) => message.id)).toEqual([
      "cached-message",
    ]);
    expect(viewModel.threadTurns.map((turn) => turn.id)).toEqual([
      "cached-turn",
    ]);
    expect(viewModel.threadItems.map((item) => item.id)).toEqual([
      "cached-item",
    ]);
  });

  it("缓存仅有 Message count 时不应恢复 canonical history window", () => {
    const viewModel = buildAgentSessionRestoreViewModel({
      cachedSnapshot: createCachedSnapshot({
        cacheMetadata: {
          storageKind: "transient",
          freshness: "fresh",
          updatedAt: 1,
          lastAccessedAt: 1,
          expiresAt: 2,
          staleUntil: 3,
          sessionUpdatedAt: 4,
          messagesCount: null,
          historyTruncated: true,
        },
      }),
      scopedCurrentTurnId: null,
      scopedItems: [],
      scopedMessages: [],
      scopedSessionCandidate: "session-unknown-total",
      scopedTurns: [],
    });

    expect(viewModel.historyWindow).toBeNull();
  });

  it("有 transient messages 时应合并 cached messages 并优先保留 transient turns/items/currentTurn", () => {
    const viewModel = buildAgentSessionRestoreViewModel({
      cachedSnapshot: createCachedSnapshot({
        messages: [
          createMessage(1, {
            id: "cached-user",
            role: "user",
            content: "同一个问题",
          }),
          createMessage(2, {
            id: "cached-assistant",
            role: "assistant",
            content: "缓存回答",
          }),
        ],
        threadTurns: [createTurn(1, { id: "cached-turn" })],
        threadItems: [
          createItem(1, { id: "cached-item", turn_id: "cached-turn" }),
        ],
        currentTurnId: "cached-turn",
      }),
      scopedCurrentTurnId: "transient-turn",
      scopedItems: [
        createItem(2, { id: "transient-item", turn_id: "transient-turn" }),
      ],
      scopedMessages: [
        createMessage(1, {
          id: "transient-user",
          role: "user",
          content: "同一个问题",
        }),
      ],
      scopedSessionCandidate: "session-1",
      scopedTurns: [createTurn(2, { id: "transient-turn" })],
    });

    expect(viewModel.messages.map((message) => message.id)).toEqual([
      "cached-user",
      "cached-assistant",
    ]);
    expect(viewModel.threadTurns.map((turn) => turn.id)).toEqual([
      "transient-turn",
    ]);
    expect(viewModel.threadItems.map((item) => item.id)).toEqual([
      "transient-item",
    ]);
    expect(viewModel.currentTurnId).toBe("transient-turn");
    expect(viewModel.historyWindow).toBeNull();
  });

  it("没有 transient turns/items 时应回退 cached timeline，但 currentTurn 优先取 transient", () => {
    const viewModel = buildAgentSessionRestoreViewModel({
      cachedSnapshot: createCachedSnapshot(),
      scopedCurrentTurnId: "local-current",
      scopedItems: [],
      scopedMessages: [createMessage(1, { id: "local-message" })],
      scopedSessionCandidate: "session-1",
      scopedTurns: [],
    });

    expect(viewModel.threadTurns.map((turn) => turn.id)).toEqual([
      "cached-turn",
    ]);
    expect(viewModel.threadItems.map((item) => item.id)).toEqual([
      "cached-item",
    ]);
    expect(viewModel.currentTurnId).toBe("local-current");
  });

  it("无 cached snapshot 时应只使用 scoped snapshot 并规范化 legacy items", () => {
    const viewModel = buildAgentSessionRestoreViewModel({
      cachedSnapshot: null,
      scopedCurrentTurnId: null,
      scopedItems: [
        createItem(1, { id: "visible-item" }),
        createItem(2, {
          id: "empty-output",
          type: "agent_message",
          text: "",
        }),
      ],
      scopedMessages: [createMessage(1, { id: "local-message" })],
      scopedSessionCandidate: "session-1",
      scopedTurns: [createTurn(1, { id: "local-turn" })],
    });

    expect(viewModel.messages.map((message) => message.id)).toEqual([
      "local-message",
    ]);
    expect(viewModel.threadTurns.map((turn) => turn.id)).toEqual([
      "local-turn",
    ]);
    expect(viewModel.threadItems.map((item) => item.id)).toEqual([
      "visible-item",
      "empty-output",
    ]);
    expect(viewModel.historyWindow).toBeNull();
  });

  it("应从 cached topic snapshot 构造可直接应用的会话快照和打点上下文", () => {
    const viewModel = buildCachedTopicSnapshotViewModel({
      cachedSnapshot: createCachedSnapshot({
        cacheMetadata: {
          storageKind: "persisted",
          freshness: "stale",
          updatedAt: 1,
          lastAccessedAt: 1,
          expiresAt: 2,
          staleUntil: 3,
          sessionUpdatedAt: 4,
          messagesCount: 5,
          historyTruncated: true,
        },
      }),
      selectedTopic: createTopic({ id: "topic-1", messagesCount: 3 }),
      topicId: "topic-1",
    });

    expect(viewModel).toMatchObject({
      sessionId: "topic-1",
      currentTurnId: "cached-turn",
      historyWindow: null,
      metricContext: {
        cacheFreshness: "stale",
        cacheStorageKind: "persisted",
        cachedMessagesCount: 1,
        cachedThreadItemsCount: 1,
        cachedTurnsCount: 1,
        topicId: "topic-1",
      },
    });
  });

  it("cached topic snapshot 未截断且总数不超过缓存消息数时不应创建历史窗口", () => {
    const viewModel = buildCachedTopicSnapshotViewModel({
      cachedSnapshot: createCachedSnapshot({
        cacheMetadata: {
          storageKind: "transient",
          freshness: "fresh",
          updatedAt: 1,
          lastAccessedAt: 1,
          expiresAt: 2,
          staleUntil: 3,
          sessionUpdatedAt: 4,
          messagesCount: 1,
          historyTruncated: false,
        },
      }),
      selectedTopic: createTopic({ id: "topic-1", messagesCount: 10 }),
      topicId: "topic-1",
    });

    expect(viewModel.historyWindow).toBeNull();
  });
});
