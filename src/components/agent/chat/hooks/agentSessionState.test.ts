import { describe, expect, it } from "vitest";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import {
  buildHydratedAgentSessionSnapshot,
  createEmptyAgentSessionSnapshot,
  hasActiveRuntimeTurn,
  hasSessionHydrationActivity,
  resolveMissingSessionFromTopicsAction,
  resolveRestorableTopicSessionId,
  shouldDeferSessionDetailHydration,
  shouldPreserveActiveLocalSessionDuringBackgroundRestoreInitialization,
  shouldSkipAlreadyHydratedSession,
} from "./agentSessionState";

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

describe("agentSessionState", () => {
  it("应创建可复用的空会话快照", () => {
    const runtime = {
      session_id: "session-1",
      provider_name: "openai",
      model_name: "gpt-5.4-mini",
      source: "session",
    } satisfies AgentSessionExecutionRuntime;

    const snapshot = createEmptyAgentSessionSnapshot({
      executionRuntime: runtime,
    });

    expect(snapshot.sessionId).toBeNull();
    expect(snapshot.messages).toEqual([]);
    expect(snapshot.threadTurns).toEqual([]);
    expect(snapshot.executionRuntime).toBe(runtime);
  });

  it("后台恢复初始化不应覆盖首发后正在运行的本地会话", () => {
    expect(
      shouldPreserveActiveLocalSessionDuringBackgroundRestoreInitialization({
        activeStreamingTimeline: false,
        messagesCount: 2,
        sessionId: "session-active-local",
        shouldRestoreSessionInForeground: false,
        threadItemsCount: 0,
        threadTurnsCount: 1,
      }),
    ).toBe(true);
    expect(
      shouldPreserveActiveLocalSessionDuringBackgroundRestoreInitialization({
        activeStreamingTimeline: true,
        messagesCount: 0,
        sessionId: "session-active-stream",
        shouldRestoreSessionInForeground: false,
        threadItemsCount: 0,
        threadTurnsCount: 0,
      }),
    ).toBe(true);
  });

  it("前台恢复或没有本地 session 时不启用后台首发保护", () => {
    expect(
      shouldPreserveActiveLocalSessionDuringBackgroundRestoreInitialization({
        activeStreamingTimeline: true,
        messagesCount: 2,
        sessionId: "session-foreground",
        shouldRestoreSessionInForeground: true,
        threadItemsCount: 1,
        threadTurnsCount: 1,
      }),
    ).toBe(false);
    expect(
      shouldPreserveActiveLocalSessionDuringBackgroundRestoreInitialization({
        activeStreamingTimeline: true,
        messagesCount: 2,
        sessionId: null,
        shouldRestoreSessionInForeground: false,
        threadItemsCount: 1,
        threadTurnsCount: 1,
      }),
    ).toBe(false);
  });

  it("应在 restore 目标失效时回退到最新有效话题", () => {
    const resolved = resolveRestorableTopicSessionId("session-stale", [
      {
        id: "session-active",
        title: "活跃会话",
        createdAt: new Date("2026-03-28T00:00:00.000Z"),
        updatedAt: new Date("2026-03-29T00:00:00.000Z"),
        workspaceId: "ws-1",
        messagesCount: 1,
        executionStrategy: "react",
        status: "done",
        lastPreview: "已完成",
        isPinned: false,
        hasUnread: false,
        tag: null,
        sourceSessionId: "session-active",
      },
    ]);

    expect(resolved).toBe("session-active");
  });

  it("限量话题列表未命中候选时，应保留候选会话用于直接恢复", () => {
    const resolved = resolveRestorableTopicSessionId(
      "session-detached",
      [
        {
          id: "session-active",
          title: "活跃会话",
          createdAt: new Date("2026-03-28T00:00:00.000Z"),
          updatedAt: new Date("2026-03-29T00:00:00.000Z"),
          workspaceId: "ws-1",
          messagesCount: 1,
          executionStrategy: "react",
          status: "done",
          lastPreview: "已完成",
          isPinned: false,
          hasUnread: false,
          tag: null,
          sourceSessionId: "session-active",
        },
      ],
      { allowDetachedCandidate: true },
    );

    expect(resolved).toBe("session-detached");
  });

  it("切到其他会话且命中有效本地快照时应允许延后 detail hydration", () => {
    expect(
      shouldDeferSessionDetailHydration({
        currentSessionId: "topic-current",
        topicId: "topic-target",
        cachedSnapshot: {
          messages: [
            createMessage({
              id: "cached-message",
              role: "assistant",
              content: "本地快照里的最近消息",
            }),
          ],
          threadTurns: [],
          threadItems: [],
          currentTurnId: null,
        },
      }),
    ).toBe(true);
  });

  it("从空态打开命中缓存的话题时也应允许先回放快照", () => {
    expect(
      shouldDeferSessionDetailHydration({
        currentSessionId: null,
        topicId: "topic-target",
        cachedSnapshot: {
          messages: [createMessage()],
          threadTurns: [],
          threadItems: [],
          currentTurnId: null,
        },
      }),
    ).toBe(true);
  });

  it("当前会话或 resume hook 场景不应延后 detail hydration", () => {
    expect(
      shouldDeferSessionDetailHydration({
        currentSessionId: "topic-target",
        topicId: "topic-target",
        cachedSnapshot: {
          messages: [createMessage()],
          threadTurns: [],
          threadItems: [],
          currentTurnId: null,
        },
      }),
    ).toBe(false);
    expect(
      shouldDeferSessionDetailHydration({
        currentSessionId: "topic-current",
        topicId: "topic-target",
        resumeSessionStartHooks: true,
        cachedSnapshot: {
          messages: [createMessage()],
          threadTurns: [],
          threadItems: [],
          currentTurnId: null,
        },
      }),
    ).toBe(false);
  });

  it("pending shell 空壳命中历史 topic 时不应跳过 detail hydration", () => {
    expect(
      shouldSkipAlreadyHydratedSession({
        currentTurnId: null,
        hydratedSessionId: "topic-history",
        messagesCount: 0,
        selectedTopic: {
          messagesCount: 2,
          status: "done",
        },
        sessionId: "topic-history",
        threadItemsCount: 0,
        threadTurnsCount: 0,
      }),
    ).toBe(false);
  });

  it("已 hydrate 且本地已有内容时应跳过重复 detail hydration", () => {
    expect(
      shouldSkipAlreadyHydratedSession({
        currentTurnId: null,
        hydratedSessionId: "topic-history",
        messagesCount: 1,
        selectedTopic: {
          messagesCount: 2,
          status: "done",
        },
        sessionId: "topic-history",
        threadItemsCount: 0,
        threadTurnsCount: 0,
      }),
    ).toBe(true);
  });

  it("已 hydrate 且 read model 仍在运行时应跳过重复 detail hydration", () => {
    expect(
      shouldSkipAlreadyHydratedSession({
        currentTurnId: null,
        hydratedSessionId: "topic-running",
        messagesCount: 0,
        selectedTopic: {
          messagesCount: 0,
          status: "running",
        },
        sessionId: "topic-running",
        threadReadStatus: "running",
        threadItemsCount: 0,
        threadTurnsCount: 0,
      }),
    ).toBe(true);
  });

  it("已 hydrate 的空草稿 topic 应跳过 detail hydration", () => {
    expect(
      shouldSkipAlreadyHydratedSession({
        currentTurnId: null,
        hydratedSessionId: "topic-draft",
        messagesCount: 0,
        selectedTopic: {
          messagesCount: 0,
          status: "draft",
        },
        sessionId: "topic-draft",
        threadItemsCount: 0,
        threadTurnsCount: 0,
      }),
    ).toBe(true);
  });

  it("会话未出现在 topics 时应按状态决定清空、跳过或远程校验", () => {
    const base = {
      currentTurnId: null,
      detachedSessionId: null,
      sessionId: "session-1",
      threadItemsCount: 0,
      threadTurnsCount: 0,
      topicsCount: 1,
      topicsReady: true,
      topicExists: false,
    };

    expect(resolveMissingSessionFromTopicsAction(base)).toEqual({
      kind: "clear_inactive",
    });
    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        remoteConfirmed: true,
      }),
    ).toEqual({ kind: "verify_remote" });
    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        restoreCandidateMayLagTopics: true,
      }),
    ).toEqual({ kind: "verify_remote" });
    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        currentTurnId: "turn-1",
      }),
    ).toEqual({ kind: "verify_remote" });
    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        detachedSessionId: "session-1",
      }),
    ).toEqual({ kind: "skip_detached" });
    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        sessionId: "title-gen-session-1",
      }),
    ).toEqual({ kind: "clear_auxiliary" });
  });

  it("canonical queued status 应阻塞下一轮用户提交", () => {
    expect(
      hasActiveRuntimeTurn({
        threadReadStatus: "queued",
        turns: [createTurn({ status: "completed" })],
      }),
    ).toBe(true);

    expect(
      hasActiveRuntimeTurn({
        threadRead: {
          thread_id: "thread-queued",
          status: "queued",
          pending_requests: [],
          incidents: [],
        },
        threadReadStatus: "queued",
        turns: [createTurn({ status: "completed" })],
      }),
    ).toBe(true);

    expect(
      hasActiveRuntimeTurn({
        threadReadStatus: "queued",
        turns: [
          createTurn({
            status: "running",
            started_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          }),
        ],
      }),
    ).toBe(true);
  });

  it("无权威 read model 时陈旧本地 running turn 不应永久阻塞输入框", () => {
    expect(
      hasActiveRuntimeTurn({
        threadReadStatus: null,
        turns: [
          createTurn({
            status: "running",
            started_at: "2026-03-29T00:00:00.000Z",
            updated_at: "2026-03-29T00:00:01.000Z",
          }),
        ],
      }),
    ).toBe(false);
  });

  it("无权威 read model 时近期本地 running turn 可短暂阻塞重复提交", () => {
    expect(
      hasActiveRuntimeTurn({
        threadReadStatus: null,
        turns: [
          createTurn({
            status: "running",
            started_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          }),
        ],
      }),
    ).toBe(true);
  });

  it("running read model 只有 active_turn_id 时也应阻塞下一轮用户提交", () => {
    expect(
      hasActiveRuntimeTurn({
        threadReadStatus: "running",
        threadRead: {
          thread_id: "thread-running",
          status: "running",
          active_turn_id: "turn-running",
          turns: [],
        },
        turns: [],
      }),
    ).toBe(true);
  });

  it("read model 已明确 idle 时不应被残留 running turn 判定为 active runtime", () => {
    expect(
      hasActiveRuntimeTurn({
        threadReadStatus: "idle",
        threadRead: {
          thread_id: "thread-1",
          status: "idle",
          turns: [
            {
              turn_id: "turn-running-1",
              status: "running",
            },
          ],
        },
        turns: [],
      }),
    ).toBe(false);

    expect(
      hasActiveRuntimeTurn({
        threadReadStatus: "idle",
        threadRead: {
          thread_id: "thread-1",
          status: "idle",
          profile_status: "running",
          turns: [],
        },
        turns: [],
      }),
    ).toBe(true);
  });

  it("显式终态 read model 不应被残留 running turn 判定为 active runtime", () => {
    expect(
      hasActiveRuntimeTurn({
        threadReadStatus: "failed",
        threadRead: {
          thread_id: "thread-1",
          status: "failed",
          active_turn_id: "turn-stale",
          turns: [
            {
              turn_id: "turn-stale",
              status: "running",
            },
          ],
        },
        turns: [createTurn({ status: "running" })],
      }),
    ).toBe(false);
  });

  it("topics 未就绪、无 session 或 topic 已存在时不应处理缺失会话", () => {
    const base = {
      currentTurnId: "turn-1",
      detachedSessionId: null,
      sessionId: "session-1",
      threadItemsCount: 1,
      threadTurnsCount: 1,
      topicsCount: 1,
      topicsReady: true,
      topicExists: false,
    };

    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        topicsReady: false,
      }),
    ).toEqual({ kind: "none" });
    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        sessionId: null,
      }),
    ).toEqual({ kind: "none" });
    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        topicsCount: 0,
      }),
    ).toEqual({ kind: "none" });
    expect(
      resolveMissingSessionFromTopicsAction({
        ...base,
        topicExists: true,
      }),
    ).toEqual({ kind: "none" });
  });

  it("同会话 hydrate 且后端缺失 execution_runtime 时应保留本地运行态", () => {
    const currentMessages = [
      createMessage({
        id: "local-user",
        role: "user",
        content: "继续保持这条本地消息",
      }),
    ];
    const currentTurns = [createTurn({ id: "turn-local" })];
    const currentItems = [
      createItem({
        id: "item-local",
        type: "tool_call",
        tool_name: "exec_command",
        arguments: { cmd: "pwd" },
      } as Partial<AgentThreadItem>),
    ];
    const currentExecutionRuntime = {
      session_id: "topic-1",
      provider_name: "openai",
      model_name: "gpt-5.4-mini",
      source: "session",
    } satisfies AgentSessionExecutionRuntime;
    const detail = {
      id: "topic-1",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [],
      turns: [createTurn({ id: "turn-remote" })],
      items: [
        createItem({
          id: "item-remote",
          type: "tool_call",
          tool_name: "exec_command",
          arguments: { cmd: "pwd" },
        } as Partial<AgentThreadItem>),
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-1",
      detail,
      currentSessionId: "topic-1",
      currentMessages,
      currentThreadTurns: currentTurns,
      currentThreadItems: currentItems,
      currentExecutionRuntime,
      currentExecutionStrategy: "react",
      topics: [],
    });

    expect(result.snapshot.sessionId).toBe("topic-1");
    expect(result.snapshot.workingDir).toBeNull();
    expect(result.snapshot.messages).toEqual(currentMessages);
    expect(result.snapshot.threadTurns.map((turn) => turn.id)).toEqual([
      "turn-local",
      "turn-remote",
    ]);
    expect(result.snapshot.threadItems.map((item) => item.id)).toEqual([
      "item-local",
      "item-remote",
    ]);
    expect(result.snapshot.currentTurnId).toBe("turn-remote");
    expect(result.snapshot.executionRuntime).toEqual(currentExecutionRuntime);
    expect(result.snapshot).not.toHaveProperty("queuedTurns");
  });

  it("hydrate 会把 App Server session working_dir 作为会话工作目录事实源", () => {
    const detail = {
      id: "topic-with-working-dir",
      created_at: 1700000000,
      updated_at: 1700000001,
      working_dir: " /workspace/runtime-cwd ",
      messages: [],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-with-working-dir",
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

    expect(result.snapshot.sessionId).toBe("topic-with-working-dir");
    expect(result.snapshot.workingDir).toBe("/workspace/runtime-cwd");
  });

  it("同一 canonical SubAgent item 经 live 后 cold hydrate 仍只保留一个 durable identity", () => {
    const liveItem = createItem({
      id: "item_subagent-call-1",
      type: "subagent_activity",
      status: "completed",
      status_label: "started",
      session_id: "thread-child",
    } as Partial<AgentThreadItem>);
    const coldItem = createItem({
      id: "item_subagent-call-1",
      type: "subagent_activity",
      status: "completed",
      status_label: "interacted",
      session_id: "thread-child",
    } as Partial<AgentThreadItem>);
    const detail = {
      id: "topic-subagent",
      thread_id: "thread-1",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [],
      turns: [createTurn()],
      items: [coldItem],
      thread_read: {
        thread_id: "thread-1",
        thread_items: [coldItem],
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-subagent",
      detail,
      currentSessionId: "topic-subagent",
      currentMessages: [],
      currentThreadTurns: [createTurn()],
      currentThreadItems: [liveItem],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
    });

    expect(
      result.snapshot.threadItems.filter(
        (item) => item.id === "item_subagent-call-1",
      ),
    ).toEqual([expect.objectContaining({ status_label: "interacted" })]);
  });

  it("同会话 hydrate 时远端纯正文不应刷新掉本地 assistant 执行过程", () => {
    const now = new Date("2026-04-08T10:00:00.000Z");
    const currentMessages = [
      createMessage({
        id: "local-user",
        role: "user",
        content: "继续保存文章",
        timestamp: new Date("2026-04-08T09:59:59.000Z"),
      }),
      createMessage({
        id: "local-assistant",
        role: "assistant",
        content: "内容已保存到项目目录。",
        timestamp: now,
        thinkingContent: "先抓正文，再下载图片。",
        contentParts: [
          {
            type: "thinking",
            text: "先抓正文，再下载图片。",
          },
          {
            type: "tool_use",
            toolCall: {
              id: "tool-site-1",
              name: "site_run_adapter",
              arguments: '{"url":"https://x.com/example/article/1"}',
              status: "completed",
              startTime: now,
              endTime: now,
              result: {
                success: true,
                output: "saved: articles/google-cloud-tech.md",
              },
            },
          },
          {
            type: "text",
            text: "内容已保存到项目目录。",
          },
        ],
        toolCalls: [
          {
            id: "tool-site-1",
            name: "site_run_adapter",
            arguments: '{"url":"https://x.com/example/article/1"}',
            status: "completed",
            startTime: now,
            endTime: now,
            result: {
              success: true,
              output: "saved: articles/google-cloud-tech.md",
            },
          },
        ],
      }),
    ];
    const detail = {
      id: "topic-1",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1710000000,
          content: [{ type: "text", text: "继续保存文章" }],
        },
        {
          role: "assistant",
          timestamp: 1710000001,
          content: [{ type: "text", text: "内容已保存到项目目录。" }],
        },
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-1",
      detail,
      currentSessionId: "topic-1",
      currentMessages,
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
    });

    expect(result.snapshot.messages[1]?.thinkingContent).toBe(
      "先抓正文，再下载图片。",
    );
    expect(result.snapshot.messages[1]?.contentParts).toEqual([
      {
        type: "thinking",
        text: "先抓正文，再下载图片。",
      },
      {
        type: "tool_use",
        toolCall: {
          id: "tool-site-1",
          name: "site_run_adapter",
          arguments: '{"url":"https://x.com/example/article/1"}',
          status: "completed",
          startTime: now,
          endTime: now,
          result: {
            success: true,
            output: "saved: articles/google-cloud-tech.md",
          },
        },
      },
      {
        type: "text",
        text: "内容已保存到项目目录。",
      },
    ]);
    expect(result.snapshot.messages[1]?.toolCalls?.[0]).toMatchObject({
      id: "tool-site-1",
      status: "completed",
    });
  });

  it("同会话 hydrate 时远端短正文片段不应截断本地已完成 assistant 输出", () => {
    const currentMessages = [
      createMessage({
        id: "local-user-search",
        role: "user",
        content: "帮我分析一下学习机怎么选",
        timestamp: new Date("2026-06-18T08:30:00.000Z"),
      }),
      createMessage({
        id: "local-assistant-search",
        role: "assistant",
        content:
          "根据我的搜索结果，建议优先比较权威评测、教材覆盖、护眼能力和售后渠道，再决定是否购买科大讯飞。",
        timestamp: new Date("2026-06-18T08:30:20.000Z"),
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-websearch-learning-device",
              name: "WebSearch",
              arguments: '{"query":"学习机 权威评测对比"}',
              status: "completed",
              startTime: new Date("2026-06-18T08:30:01.000Z"),
              endTime: new Date("2026-06-18T08:30:05.000Z"),
              result: {
                success: true,
                output: "https://example.com/review",
              },
            },
          },
          {
            type: "text",
            text: "根据我的搜索结果，建议优先比较权威评测、教材覆盖、护眼能力和售后渠道，再决定是否购买科大讯飞。",
          },
        ],
      }),
      createMessage({
        id: "local-user-follow",
        role: "user",
        content: "T30 Pro 和 T90 有什么区别呢",
        timestamp: new Date("2026-06-18T08:31:00.000Z"),
      }),
    ];
    const detail = {
      id: "topic-search",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1781767800,
          content: [{ type: "text", text: "帮我分析一下学习机怎么选" }],
        },
        {
          role: "assistant",
          timestamp: 1781767820,
          content: [{ type: "text", text: "根据我" }],
        },
        {
          role: "user",
          timestamp: 1781767860,
          content: [{ type: "text", text: "T30 Pro 和 T90 有什么区别呢" }],
        },
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-search",
      detail,
      currentSessionId: "topic-search",
      currentMessages,
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
    });

    expect(result.snapshot.messages[1]?.content).toContain("建议优先比较");
    expect(result.snapshot.messages[1]?.content).not.toBe("根据我");
    expect(result.snapshot.messages[1]?.contentParts?.at(-1)).toMatchObject({
      type: "text",
      text: expect.stringContaining("售后渠道"),
    });
    expect(result.snapshot.messages[2]?.content).toBe(
      "T30 Pro 和 T90 有什么区别呢",
    );
  });

  it("同会话 hydrate 保护本地 timeline 时仍应合入 App Server thread_read.tool_calls", () => {
    const currentMessages = [
      createMessage({
        id: "local-user-tool-read",
        role: "user",
        content: "生成 TypeScript greeting 代码产物",
        timestamp: new Date("2026-06-07T10:41:40.000Z"),
      }),
      createMessage({
        id: "local-assistant-tool-read",
        role: "assistant",
        content: "已生成代码产物，可在工作台查看。",
        timestamp: new Date("2026-06-07T10:41:42.000Z"),
        contentParts: [
          {
            type: "text",
            text: "已生成代码产物，可在工作台查看。",
          },
        ],
      }),
    ];
    const detail = {
      id: "topic-tool-read",
      thread_id: "thread-tool-read",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1780704100,
          content: [
            { type: "text", text: "生成 TypeScript greeting 代码产物" },
          ],
        },
        {
          role: "assistant",
          timestamp: 1780704102,
          content: [{ type: "text", text: "已生成代码产物，可在工作台查看。" }],
        },
      ],
      turns: [
        createTurn({
          id: "turn-tool-read",
          thread_id: "thread-tool-read",
          prompt_text: "生成 TypeScript greeting 代码产物",
          status: "completed",
          started_at: "2026-06-07T10:41:40.000Z",
          completed_at: "2026-06-07T10:41:42.000Z",
          updated_at: "2026-06-07T10:41:42.000Z",
        }),
      ],
      thread_read: {
        thread_id: "thread-tool-read",
        status: "completed",
        profile_status: "completed",
        turns: [
          {
            turn_id: "turn-tool-read",
            status: "completed",
            native_status: "completed",
          },
        ],
        pending_requests: [],
        incidents: [],
        tool_calls: [
          {
            tool_call_id: "tool-webfetch-read",
            turn_id: "turn-tool-read",
            tool_name: "WebFetch",
            status: "completed",
            started_at: "2026-06-07T10:41:41.000Z",
            finished_at: "2026-06-07T10:41:42.000Z",
            arguments: {
              url: "https://example.com/lime-workbench-tool",
            },
            output_preview:
              "已获取 fixture 工具事实: https://example.com/lime-workbench-tool",
            output:
              "已获取 fixture 工具事实: https://example.com/lime-workbench-tool",
            success: true,
          },
        ],
      },
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-tool-read",
      detail,
      currentSessionId: "topic-tool-read",
      currentMessages,
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
    });

    expect(result.snapshot.messages).toHaveLength(2);
    expect(result.snapshot.messages[1]).toMatchObject({
      id: "local-assistant-tool-read",
      role: "assistant",
      content: "已生成代码产物，可在工作台查看。",
      runtimeTurnId: "turn-tool-read",
      toolCalls: [
        {
          id: "tool-webfetch-read",
          name: "WebFetch",
          status: "completed",
          result: {
            success: true,
            output:
              "已获取 fixture 工具事实: https://example.com/lime-workbench-tool",
          },
        },
      ],
    });
    expect(
      result.snapshot.messages[1]?.contentParts?.map((part) => part.type),
    ).toEqual(["tool_use", "text"]);
  });

  it("同会话 hydrate 时远端暂未返回最新 assistant 消息也应保留本地尾部消息", () => {
    const now = new Date("2026-04-08T10:00:02.000Z");
    const currentMessages = [
      createMessage({
        id: "local-user",
        role: "user",
        content: "继续保存文章",
        timestamp: new Date("2026-04-08T10:00:00.000Z"),
      }),
      createMessage({
        id: "local-assistant",
        role: "assistant",
        content: "内容已保存到项目目录。",
        timestamp: now,
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-site-2",
              name: "site_run_adapter",
              arguments: '{"url":"https://x.com/example/article/2"}',
              status: "completed",
              startTime: new Date("2026-04-08T10:00:01.000Z"),
              endTime: now,
              result: {
                success: true,
                output: "saved: articles/google-cloud-tech-2.md",
              },
            },
          },
          {
            type: "text",
            text: "内容已保存到项目目录。",
          },
        ],
        toolCalls: [
          {
            id: "tool-site-2",
            name: "site_run_adapter",
            arguments: '{"url":"https://x.com/example/article/2"}',
            status: "completed",
            startTime: new Date("2026-04-08T10:00:01.000Z"),
            endTime: now,
            result: {
              success: true,
              output: "saved: articles/google-cloud-tech-2.md",
            },
          },
        ],
      }),
    ];
    const detail = {
      id: "topic-1",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1710000001,
          content: [{ type: "text", text: "继续保存文章" }],
        },
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-1",
      detail,
      currentSessionId: "topic-1",
      currentMessages,
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
    });

    expect(result.snapshot.messages).toHaveLength(2);
    expect(result.snapshot.messages[1]?.role).toBe("assistant");
    expect(
      result.snapshot.messages[1]?.contentParts?.some(
        (part) =>
          part.type === "tool_use" && part.toolCall.id === "tool-site-2",
      ),
    ).toBe(true);
  });

  it("同会话 hydrate 时远端最后停在 user 且时间更晚，也应保留本地 assistant 尾部", () => {
    const currentMessages = [
      createMessage({
        id: "local-user-earlier",
        role: "user",
        content: "导出这篇文章",
        timestamp: new Date("2026-04-08T10:00:00.000Z"),
      }),
      createMessage({
        id: "local-assistant-earlier",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-04-08T10:00:00.500Z"),
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-site-earlier",
              name: "site_run_adapter",
              arguments: '{"url":"https://x.com/example/article/earlier"}',
              status: "completed",
              startTime: new Date("2026-04-08T10:00:00.100Z"),
              endTime: new Date("2026-04-08T10:00:00.500Z"),
              result: {
                success: true,
                output: "saved: articles/example-earlier.md",
              },
            },
          },
        ],
        toolCalls: [
          {
            id: "tool-site-earlier",
            name: "site_run_adapter",
            arguments: '{"url":"https://x.com/example/article/earlier"}',
            status: "completed",
            startTime: new Date("2026-04-08T10:00:00.100Z"),
            endTime: new Date("2026-04-08T10:00:00.500Z"),
            result: {
              success: true,
              output: "saved: articles/example-earlier.md",
            },
          },
        ],
      }),
    ];
    const detail = {
      id: "topic-earlier-tail",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [
        {
          role: "user",
          timestamp: 1712570401,
          content: [{ type: "text", text: "导出这篇文章" }],
        },
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-earlier-tail",
      detail,
      currentSessionId: "topic-earlier-tail",
      currentMessages,
      currentThreadTurns: [],
      currentThreadItems: [],
      currentExecutionRuntime: null,
      currentExecutionStrategy: "react",
      topics: [],
    });

    expect(result.snapshot.messages).toHaveLength(2);
    expect(result.snapshot.messages[1]?.role).toBe("assistant");
    expect(result.snapshot.messages[1]?.toolCalls?.[0]?.id).toBe(
      "tool-site-earlier",
    );
  });

  it("应从历史恢复快照中过滤辅助标题 turn，保留真实用户 turn 作为当前回合", () => {
    const detail = {
      id: "topic-with-auxiliary-turn",
      created_at: 1700000000,
      updated_at: 1700000001,
      messages: [],
      turns: [
        createTurn({
          id: "turn-user-search",
          thread_id: "topic-with-auxiliary-turn",
          prompt_text: "@搜索 OpenAI 最新模型公告",
          status: "failed",
          error_message:
            "运行时权限声明需要真实确认，当前 turn 已在模型执行前等待用户确认：confirmationStatus=not_requested，askProfileKeys=web_search。已创建真实权限确认请求；请确认后重试或恢复本轮执行。",
          started_at: "2026-05-06T19:29:06.522Z",
          completed_at: "2026-05-06T19:29:06.862Z",
          created_at: "2026-05-06T19:29:06.522Z",
          updated_at: "2026-05-06T19:29:06.862Z",
        }),
        createTurn({
          id: "auxiliary-runtime-projection-title",
          thread_id: "topic-with-auxiliary-turn",
          prompt_text: "辅助标题生成 · 我来帮你搜索 OpenAI 最新模型...",
          status: "completed",
          started_at: "2026-05-06T19:29:55.849Z",
          completed_at: "2026-05-06T19:29:55.896Z",
          created_at: "2026-05-06T19:29:55.849Z",
          updated_at: "2026-05-06T19:29:55.896Z",
        }),
      ],
      items: [
        createItem({
          id: "auxiliary-runtime-artifact",
          thread_id: "topic-with-auxiliary-turn",
          turn_id: "auxiliary-runtime-projection-title",
          type: "file_artifact",
          path: ".lime/harness/sessions/topic-with-auxiliary-turn/auxiliary-runtime/title.json",
          source: "auxiliary_runtime_projection",
          metadata: {
            artifactType: "auxiliary_runtime_projection",
          },
        } as Partial<AgentThreadItem>),
      ],
    } satisfies AgentSessionDetail;

    const result = buildHydratedAgentSessionSnapshot({
      topicId: "topic-with-auxiliary-turn",
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

    expect(result.snapshot.messages).toHaveLength(1);
    expect(result.snapshot.messages[0]?.content).toBe(
      "@搜索 OpenAI 最新模型公告",
    );
    expect(result.snapshot.threadTurns).toHaveLength(1);
    expect(result.snapshot.currentTurnId).toBe("turn-user-search");
    expect(result.snapshot.threadItems).toEqual([]);
  });

  it("应按本地时间线活动判断是否需要校验丢失会话", () => {
    expect(
      hasSessionHydrationActivity({
        currentTurnId: null,
        threadTurnsCount: 0,
        threadItemsCount: 0,
      }),
    ).toBe(false);
    expect(
      hasSessionHydrationActivity({
        currentTurnId: "turn-1",
        threadTurnsCount: 0,
        threadItemsCount: 0,
      }),
    ).toBe(true);
  });
});
