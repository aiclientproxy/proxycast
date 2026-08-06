import { describe, expect, it } from "vitest";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";

import {
  ACTIVE_SESSION_TRANSIENT_ITEMS_LIMIT,
  ACTIVE_SESSION_TRANSIENT_MESSAGES_LIMIT,
  ACTIVE_SESSION_TRANSIENT_TURNS_LIMIT,
  applyTopicExecutionStrategyToTopics,
  applyTopicSnapshotToTopics,
  mapSessionDetailToTopic,
  prependVerifiedSessionTopicFromDetail,
  resolveRestoreCandidateSanitizationPlan,
  resolveRuntimeThreadStatusFromSessionDetail,
  selectActiveSessionTransientItems,
  selectActiveSessionTransientMessages,
  selectActiveSessionTransientTurns,
  upsertFreshSessionDraftTopic,
  upsertTopicFromSessionDetail,
} from "./agentSessionTopicViewModel";
import type { Topic } from "./agentChatShared";

function createDetail(
  overrides: Partial<AgentSessionDetail> = {},
): AgentSessionDetail {
  return {
    id: "session-1",
    name: "会话 1",
    created_at: 1,
    updated_at: 2,
    messages_count: 0,
    messages: [],
    ...overrides,
  };
}

function createTopic(overrides: Partial<Topic> = {}): Topic {
  return {
    id: "topic-1",
    title: "Topic 1",
    createdAt: new Date(1_000),
    updatedAt: new Date(2_000),
    workspaceId: "workspace-1",
    messagesCount: 1,
    executionStrategy: "react",
    status: "done",
    statusReason: "default",
    lastPreview: "preview",
    hasUnread: false,
    tag: null,
    sourceSessionId: "topic-1",
    ...overrides,
  };
}

function createMessage(index: number): Message {
  return {
    id: `message-${index}`,
    role: index % 2 === 0 ? "assistant" : "user",
    content: `message ${index}`,
    timestamp: new Date(index),
  };
}

function createTurn(index: number): AgentThreadTurn {
  return {
    id: `turn-${index}`,
    thread_id: "thread-1",
    prompt_text: `prompt ${index}`,
    status: "completed",
    started_at: new Date(index).toISOString(),
    completed_at: new Date(index + 1).toISOString(),
    created_at: new Date(index).toISOString(),
    updated_at: new Date(index + 1).toISOString(),
  };
}

function createItem(index: number, turnId?: string): AgentThreadItem {
  return {
    id: `item-${index}`,
    thread_id: "thread-1",
    turn_id: turnId ?? "",
    sequence: index,
    type: "turn_summary",
    text: `item ${index}`,
    status: "completed",
    started_at: new Date(index).toISOString(),
    completed_at: new Date(index + 1).toISOString(),
    updated_at: new Date(index + 1).toISOString(),
  };
}

describe("agentSessionTopicViewModel", () => {
  it("应清洗 restore candidate 并拒绝跨工作区或辅助会话", () => {
    expect(
      resolveRestoreCandidateSanitizationPlan({
        candidateSessionId: "  ",
        mappedWorkspaceId: null,
        workspaceId: "workspace-1",
      }),
    ).toEqual({ kind: "empty" });

    expect(
      resolveRestoreCandidateSanitizationPlan({
        candidateSessionId: "title-gen-child-1",
        mappedWorkspaceId: "workspace-1",
        workspaceId: "workspace-1",
      }),
    ).toEqual({
      kind: "skip_auxiliary",
      candidateSessionId: "title-gen-child-1",
      workspaceId: "workspace-1",
    });

    expect(
      resolveRestoreCandidateSanitizationPlan({
        candidateSessionId: "session-1",
        mappedWorkspaceId: "workspace-default",
        workspaceId: "workspace-1",
      }),
    ).toEqual({
      kind: "reject_workspace",
      candidateSessionId: "session-1",
      mappedWorkspaceId: "workspace-default",
      workspaceId: "workspace-1",
    });

    expect(
      resolveRestoreCandidateSanitizationPlan({
        candidateSessionId: "session-1",
        mappedWorkspaceId: "workspace-2",
        workspaceId: "workspace-1",
      }),
    ).toEqual({
      kind: "reject_workspace",
      candidateSessionId: "session-1",
      mappedWorkspaceId: "workspace-2",
      workspaceId: "workspace-1",
    });
  });

  it("restore candidate 无映射或同工作区映射时应保留", () => {
    expect(
      resolveRestoreCandidateSanitizationPlan({
        candidateSessionId: " session-1 ",
        mappedWorkspaceId: null,
        workspaceId: "workspace-1",
      }),
    ).toEqual({
      kind: "accept",
      sessionId: "session-1",
    });

    expect(
      resolveRestoreCandidateSanitizationPlan({
        candidateSessionId: "session-1",
        mappedWorkspaceId: "workspace-1",
        workspaceId: "workspace-1",
      }),
    ).toEqual({
      kind: "accept",
      sessionId: "session-1",
    });

    expect(
      resolveRestoreCandidateSanitizationPlan({
        candidateSessionId: "session-1",
        mappedWorkspaceId: "__invalid__",
        workspaceId: "workspace-1",
      }),
    ).toEqual({
      kind: "accept",
      sessionId: "session-1",
    });
  });

  it("应从 session detail 的 canonical 状态推导 topic runtime 状态", () => {
    expect(
      resolveRuntimeThreadStatusFromSessionDetail(
        createDetail({
          thread_read: {
            thread_id: "thread-1",
            status: "waiting_request",
            pending_requests: [],
            incidents: [],
          },
        }),
      ),
    ).toBe("waiting");

    expect(
      resolveRuntimeThreadStatusFromSessionDetail(
        createDetail({
          thread_read: {
            thread_id: "thread-1",
            status: "queued",
          },
        }),
      ),
    ).toBe("queued");
  });

  it("应把 runtime detail 映射为 topic 并覆盖运行态展示", () => {
    const topic = mapSessionDetailToTopic(
      "session-running",
      createDetail({
        name: "运行中任务",
        messages_count: 2,
        workspace_id: "workspace-detail",
        working_dir: "/repo/detail",
        thread_read: {
          thread_id: "thread-1",
          status: "running",
          pending_requests: [],
          incidents: [],
        },
      }),
      "workspace-fallback",
    );

    expect(topic).toMatchObject({
      id: "session-running",
      title: "运行中任务",
      workspaceId: "workspace-detail",
      workingDir: "/repo/detail",
      messagesCount: 2,
      status: "running",
      statusReason: "default",
    });
  });

  it("应把 runtime detail 中的 queued 状态映射为排队 topic", () => {
    const topic = mapSessionDetailToTopic(
      "session-queued",
      createDetail({
        name: "排队任务",
        messages_count: 2,
        thread_read: {
          thread_id: "thread-1",
          status: "queued",
          pending_requests: [],
          incidents: [],
        },
      }),
      null,
    );

    expect(topic).toMatchObject({
      id: "session-queued",
      title: "排队任务",
      messagesCount: 2,
      status: "queued",
      statusReason: "default",
    });
  });

  it("upsert topic 时应保留本地 unread/tag 和服务端列表位置", () => {
    const existing = createTopic({
      id: "topic-1",
      updatedAt: new Date(2_000),
      hasUnread: true,
      tag: "重点",
    });
    const newer = createTopic({
      id: "topic-2",
      updatedAt: new Date(4_000),
    });
    const detailTopic = createTopic({
      id: "topic-1",
      title: "远端新标题",
      updatedAt: new Date(5_000),
      workingDir: "/repo/runtime",
      hasUnread: false,
      tag: null,
    });

    expect(
      upsertTopicFromSessionDetail([existing, newer], detailTopic),
    ).toEqual([
      {
        ...detailTopic,
        hasUnread: true,
        tag: "重点",
        workingDir: "/repo/runtime",
      },
      newer,
    ]);
  });

  it("详情刷新只有通用 fallback 标题时不应覆盖已有真实标题", () => {
    const existing = createTopic({
      id: "topic-1",
      title: "Claw 新闻输入 Electron fixture",
      updatedAt: new Date(2_000),
    });
    const detailTopic = createTopic({
      id: "topic-1",
      title: "任务 1970/1/1",
      updatedAt: new Date(3_000),
    });

    expect(
      upsertTopicFromSessionDetail([existing], detailTopic)[0],
    ).toMatchObject({
      id: "topic-1",
      title: "Claw 新闻输入 Electron fixture",
      updatedAt: new Date(3_000),
    });
  });

  it("详情刷新不应改写 App Server 返回的话题顺序", () => {
    const legacy = createTopic({
      id: "topic-legacy",
      createdAt: new Date(1_000),
      updatedAt: new Date(2_000),
      workspaceId: null,
    });
    const current = createTopic({
      id: "topic-current",
      createdAt: new Date(1_000),
      updatedAt: new Date(2_000),
      workspaceId: "workspace-1",
    });

    expect(
      upsertTopicFromSessionDetail(
        [legacy, current],
        {
          ...legacy,
          title: "历史详情已刷新",
        },
      ).map((topic) => topic.id),
    ).toEqual(["topic-legacy", "topic-current"]);
  });

  it("session detail 缺少 workspace 时不应抹掉列表已确认的 workspace 归属", () => {
    const existing = createTopic({
      id: "topic-current",
      workspaceId: "workspace-1",
    });
    const detailTopic = createTopic({
      id: "topic-current",
      title: "详情标题",
      workspaceId: null,
    });

    expect(
      upsertTopicFromSessionDetail([existing], detailTopic)[0],
    ).toMatchObject({
      id: "topic-current",
      title: "详情标题",
      workspaceId: "workspace-1",
    });
  });

  it("补入不在当前页的已验证会话时应放在列表首位", () => {
    const current = createTopic({
      id: "topic-current",
      createdAt: new Date(1_000),
      updatedAt: new Date(2_000),
      workspaceId: "workspace-1",
    });

    expect(
      prependVerifiedSessionTopicFromDetail(
        [current],
        "topic-legacy",
        createDetail({
          id: "topic-legacy",
          created_at: 1,
          updated_at: 2,
          messages_count: 1,
          messages: [],
        }),
      ).map((topic) => topic.id),
    ).toEqual(["topic-legacy", "topic-current"]);
  });

  it("应把新建 session 草稿插入到 topic 顶部并去重", () => {
    const existing = createTopic({
      id: "session-new",
      title: "旧草稿",
      updatedAt: new Date(1_000),
    });
    const other = createTopic({ id: "topic-2", title: "保留任务" });
    const now = new Date(8_000);

    expect(
      upsertFreshSessionDraftTopic([existing, other], {
        createdAt: now,
        executionStrategy: "react",
        sessionId: "session-new",
        sessionName: "  新任务标题  ",
        workspaceId: "workspace-2",
      }),
    ).toEqual([
      {
        id: "session-new",
        title: "新任务标题",
        createdAt: now,
        updatedAt: now,
        workspaceId: "workspace-2",
        workingDir: null,
        messagesCount: 0,
        executionStrategy: "react",
        status: "draft",
        lastPreview: "等待你补充任务需求后开始执行。",
        hasUnread: false,
        tag: null,
        sourceSessionId: "session-new",
      },
      other,
    ]);
  });

  it("远端校验通过后应只在 topic 缺失时补入 session", () => {
    const detail = createDetail({
      name: "远端会话",
      created_at: 5,
      updated_at: 6,
      messages_count: 20,
      messages: [{ id: "message-1" } as never, { id: "message-2" } as never],
      execution_strategy: "code_orchestrated" as never,
      workspace_id: "workspace-remote",
      working_dir: "/tmp/workspace",
    });
    const existing = createTopic({ id: "topic-1" });
    const nextTopics = prependVerifiedSessionTopicFromDetail(
      [existing],
      "session-remote",
      detail,
    );

    expect(nextTopics[0]).toMatchObject({
      id: "session-remote",
      title: "远端会话",
      messagesCount: 2,
      executionStrategy: "react",
      workspaceId: "workspace-remote",
    });
    expect(nextTopics[1]).toBe(existing);
    expect(
      prependVerifiedSessionTopicFromDetail(
        nextTopics,
        "session-remote",
        detail,
      ),
    ).toBe(nextTopics);
  });

  it("应只更新目标 topic 的 execution strategy", () => {
    const target = createTopic({
      id: "topic-1",
      executionStrategy: "react",
    });
    const other = createTopic({
      id: "topic-2",
      executionStrategy: "react",
    });

    expect(
      applyTopicExecutionStrategyToTopics([target, other], "topic-1", "react"),
    ).toEqual([
      {
        ...target,
        executionStrategy: "react",
      },
      other,
    ]);
  });

  it("应只把 live snapshot 写回目标 topic", () => {
    const original = createTopic({
      id: "topic-1",
      messagesCount: 1,
      updatedAt: new Date(1_000),
    });
    const other = createTopic({
      id: "topic-2",
      messagesCount: 2,
      updatedAt: new Date(2_000),
    });
    const updatedAt = new Date(6_000);

    expect(
      applyTopicSnapshotToTopics([original, other], "topic-1", {
        updatedAt,
        messagesCount: 3,
        status: "running",
        statusReason: "default",
        lastPreview: "正在执行",
        hasUnread: true,
      }),
    ).toEqual([
      {
        ...original,
        updatedAt,
        messagesCount: 3,
        status: "running",
        statusReason: "default",
        lastPreview: "正在执行",
        hasUnread: true,
      },
      other,
    ]);
  });

  it("live snapshot 更新最近活动时间后不应改写服务端列表顺序", () => {
    const stale = createTopic({
      id: "topic-stale",
      updatedAt: new Date(1_000),
    });
    const recent = createTopic({
      id: "topic-recent",
      updatedAt: new Date(4_000),
    });

    expect(
      applyTopicSnapshotToTopics([recent, stale], "topic-stale", {
        updatedAt: new Date(8_000),
      }).map((topic) => topic.id),
    ).toEqual(["topic-recent", "topic-stale"]);
  });

  it("live snapshot 未改变或目标不存在时应复用原 topic 数组", () => {
    const original = createTopic({ id: "topic-1", messagesCount: 1 });
    const topics = [original];

    expect(
      applyTopicSnapshotToTopics(topics, "topic-1", {
        updatedAt: original.updatedAt,
        messagesCount: original.messagesCount,
        status: original.status,
        statusReason: original.statusReason,
        lastPreview: original.lastPreview,
        hasUnread: original.hasUnread,
      }),
    ).toBe(topics);

    expect(
      applyTopicSnapshotToTopics(topics, "missing-topic", {
        messagesCount: 2,
      }),
    ).toBe(topics);
  });

  it("应裁剪 transient messages 和 turns 到尾部窗口", () => {
    const messages = Array.from(
      { length: ACTIVE_SESSION_TRANSIENT_MESSAGES_LIMIT + 2 },
      (_, index) => createMessage(index),
    );
    const turns = Array.from(
      { length: ACTIVE_SESSION_TRANSIENT_TURNS_LIMIT + 2 },
      (_, index) => createTurn(index),
    );

    expect(selectActiveSessionTransientMessages(messages)[0]?.id).toBe(
      "message-2",
    );
    expect(selectActiveSessionTransientTurns(turns)[0]?.id).toBe("turn-2");
  });

  it("应按保留 turn 裁剪 transient items，并保留无 turn_id 的邻近项", () => {
    const turns = Array.from(
      { length: ACTIVE_SESSION_TRANSIENT_TURNS_LIMIT + 1 },
      (_, index) => createTurn(index),
    );
    const items = [
      createItem(1, "turn-0"),
      createItem(2, "turn-1"),
      createItem(3, "turn-2"),
      createItem(4),
    ];

    expect(selectActiveSessionTransientItems(items, turns)).toEqual([
      createItem(2, "turn-1"),
      createItem(3, "turn-2"),
      createItem(4),
    ]);
  });

  it("没有 retained turn 时应只按 item 尾部窗口裁剪", () => {
    const items = Array.from(
      { length: ACTIVE_SESSION_TRANSIENT_ITEMS_LIMIT + 2 },
      (_, index) => createItem(index),
    );

    expect(selectActiveSessionTransientItems(items, [])[0]?.id).toBe("item-2");
  });
});
