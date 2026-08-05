import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  createDeferred,
  flushEffects,
  mockGetAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockScheduleMinimumDelayIdleTask,
  mountHook,
  seedSession,
  seedSessionSnapshots,
} from "../useAgentChat.testUtils";

describe("useAgentChat 偏好持久化 - snapshot hydration", () => {
  it("冷恢复已有消息缓存时仍应合并新增 canonical MCP item", async () => {
    const workspaceId = "ws-cold-restore-mcp-item";
    const sessionId = "session-cold-restore-mcp-item";
    const threadId = "thread-cold-restore-mcp-item";
    const turnId = "turn-cold-restore-mcp-item";
    const topicsDeferred = createDeferred<never>();

    seedSession(workspaceId, sessionId);
    mockListAgentRuntimeSessions.mockReturnValue(topicsDeferred.promise);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      thread_id: threadId,
      workspace_id: workspaceId,
      messages: [],
      turns: [
        {
          id: turnId,
          thread_id: threadId,
          status: "completed",
          prompt_text: "运行插件检查",
          started_at: "2026-08-04T00:00:00.000Z",
          completed_at: "2026-08-04T00:00:02.000Z",
          created_at: "2026-08-04T00:00:00.000Z",
          updated_at: "2026-08-04T00:00:02.000Z",
        },
      ],
      items: [
        {
          id: "item-mcp-app-cold-restore",
          thread_id: threadId,
          turn_id: turnId,
          sequence: 1,
          type: "tool_call",
          tool_name: "mcp__plugin__demo__release_check",
          status: "completed",
          success: true,
          started_at: "2026-08-04T00:00:01.000Z",
          completed_at: "2026-08-04T00:00:02.000Z",
          updated_at: "2026-08-04T00:00:02.000Z",
          metadata: {
            canonical_type: "mcpToolCall",
            plugin_id: "demo-plugin",
            server: "mcp__plugin__demo",
            mcp_app_resource_uri: "ui://demo/release-check.html",
          },
        },
      ],
      thread_read: {
        thread_id: threadId,
        status: "idle",
      },
      execution_strategy: "react",
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      expect(harness.getValue().sessionId).toBe(sessionId);
      expect(harness.getValue().messages).toHaveLength(1);

      await act(async () => {
        await harness.getValue().switchTopic(sessionId);
      });
      await flushEffects();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(harness.getValue().threadItems).toContainEqual(
        expect.objectContaining({
          id: "item-mcp-app-cold-restore",
          metadata: expect.objectContaining({
            canonical_type: "mcpToolCall",
            mcp_app_resource_uri: "ui://demo/release-check.html",
          }),
        }),
      );

      mockGetAgentRuntimeSession.mockClear();
      await act(async () => {
        await harness.getValue().switchTopic(sessionId);
      });
      expect(mockGetAgentRuntimeSession).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("直接会话水合失败时应向导航层返回错误结果", async () => {
    const workspaceId = "ws-topic-hydration-error";
    const hydrationError = new Error(
      "thread/read did not return canonical session detail",
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "topic-invalid-import",
        name: "损坏的导入会话",
        created_at: Math.floor(Date.now() / 1000),
        messages_count: 1,
        workspace_id: workspaceId,
      },
    ]);
    mockGetAgentRuntimeSession.mockRejectedValue(hydrationError);
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      let result: unknown;
      await act(async () => {
        result = await harness.getValue().switchTopic("topic-invalid-import", {
          forceRefresh: true,
        });
      });
      expect(result).toBe("error");
    } finally {
      harness.unmount();
    }
  });

  it("切换到命中过往快照的话题时应先立即回放本地 tail，并立即后台拉取远端详情", async () => {
    const workspaceId = "ws-topic-history-warm-restore";
    const createdAt = Math.floor(Date.now() / 1000);
    const deferredTopicDetail = createDeferred<{
      id: string;
      workspace_id?: string;
      messages: Array<{
        role: "assistant" | "user";
        timestamp: number;
        content: Array<{ type: "text"; text: string }>;
      }>;
      turns: [];
      items: [];
      execution_strategy: "react";
    }>();

    seedSessionSnapshots(workspaceId, {
      "topic-a": {
        messages: [
          {
            id: "topic-a-cached-user",
            role: "user",
            content: "继续完善上一个方案",
            timestamp: "2026-04-24T10:00:00.000Z",
          },
          {
            id: "topic-a-cached-assistant",
            role: "assistant",
            content: "这是本地快照里的最近结果。",
            timestamp: "2026-04-24T10:00:02.000Z",
          },
        ],
        threadTurns: [],
        threadItems: [],
        currentTurnId: null,
        updatedAt: Date.now(),
      },
    });
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "topic-a",
        name: "历史任务 A",
        created_at: createdAt,
        messages_count: 2,
        workspace_id: workspaceId,
      },
    ]);
    mockGetAgentRuntimeSession.mockImplementation(async (topicId: string) => {
      if (topicId === "topic-a") {
        return deferredTopicDetail.promise;
      }

      return {
        id: topicId,
        workspace_id: workspaceId,
        messages: [],
        turns: [],
        items: [],
        execution_strategy: "react" as const,
      };
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      mockScheduleMinimumDelayIdleTask.mockClear();

      let switchPromise: Promise<unknown> | null = null;
      await act(async () => {
        switchPromise = harness.getValue().switchTopic("topic-a");
        await Promise.resolve();
      });

      expect(harness.getValue().sessionId).toBe("topic-a");
      expect(harness.getValue().messages).toHaveLength(2);
      expect(harness.getValue().messages[1]?.content).toBe(
        "这是本地快照里的最近结果。",
      );
      await expect(switchPromise).resolves.toBeUndefined();

      expect(mockScheduleMinimumDelayIdleTask).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeSession).toHaveBeenCalledTimes(1);

      await act(async () => {
        deferredTopicDetail.resolve({
          id: "topic-a",
          messages: [
            {
              role: "user",
              timestamp: Math.floor(
                new Date("2026-04-24T10:00:01.000Z").getTime() / 1000,
              ),
              content: [{ type: "text", text: "继续完善上一个方案" }],
            },
            {
              role: "assistant",
              timestamp: Math.floor(
                new Date("2026-04-24T10:00:05.000Z").getTime() / 1000,
              ),
              content: [{ type: "text", text: "这是远端补全后的最终结果。" }],
            },
          ],
          turns: [],
          items: [],
          execution_strategy: "react",
        });
      });
      await flushEffects();

      expect(harness.getValue().messages.at(-1)?.content).toBe(
        "这是远端补全后的最终结果。",
      );
    } finally {
      harness.unmount();
    }
  });

  it("切换到 stale 快照话题时应先回放缓存，不伪造 history cursor，并立即后台刷新", async () => {
    const workspaceId = "ws-topic-history-stale-refresh";
    const nowMs = Date.now();
    const deferredTopicDetail = createDeferred<{
      id: string;
      workspace_id?: string;
      messages: Array<{
        role: "assistant" | "user";
        timestamp: number;
        content: Array<{ type: "text"; text: string }>;
      }>;
      turns: [];
      items: [];
      execution_strategy: "react";
    }>();

    seedSession(workspaceId, "topic-current");
    seedSessionSnapshots(workspaceId, {
      "topic-stale": {
        messages: [
          {
            id: "topic-stale-cached-assistant",
            role: "assistant",
            content: "这是 stale 快照里的最近结果。",
            timestamp: "2026-04-24T10:00:02.000Z",
          },
        ],
        threadTurns: [],
        threadItems: [],
        currentTurnId: null,
        updatedAt: nowMs - 10_000,
        lastAccessedAt: nowMs - 10_000,
        expiresAt: nowMs - 1_000,
        staleUntil: nowMs + 60_000,
        sessionUpdatedAt: nowMs - 10_000,
        messagesCount: 2,
        historyTruncated: true,
      },
    });
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "topic-current",
        name: "当前任务",
        created_at: Math.floor(nowMs / 1000),
        messages_count: 1,
        workspace_id: workspaceId,
      },
      {
        id: "topic-stale",
        name: "历史任务 stale",
        created_at: Math.floor(nowMs / 1000),
        updated_at: Math.floor((nowMs + 1_000) / 1000),
        messages_count: 2,
        workspace_id: workspaceId,
      },
    ]);
    mockGetAgentRuntimeSession.mockImplementation(async (topicId: string) => {
      if (topicId === "topic-stale") {
        return deferredTopicDetail.promise;
      }

      return {
        id: topicId,
        workspace_id: workspaceId,
        messages: [],
        turns: [],
        items: [],
        execution_strategy: "react" as const,
      };
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      mockGetAgentRuntimeSession.mockClear();
      mockScheduleMinimumDelayIdleTask.mockClear();

      await act(async () => {
        void harness.getValue().switchTopic("topic-stale");
        await Promise.resolve();
      });

      expect(harness.getValue().sessionId).toBe("topic-stale");
      expect(harness.getValue().messages[0]?.content).toBe(
        "这是 stale 快照里的最近结果。",
      );
      expect(harness.getValue().sessionHistoryWindow).toBeNull();
      expect(mockScheduleMinimumDelayIdleTask).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        "topic-stale",
        expect.objectContaining({ historyLimit: 40 }),
      );

      await act(async () => {
        deferredTopicDetail.resolve({
          id: "topic-stale",
          messages: [
            {
              role: "assistant",
              timestamp: Math.floor(nowMs / 1000),
              content: [{ type: "text", text: "这是远端刷新后的结果。" }],
            },
          ],
          turns: [],
          items: [],
          execution_strategy: "react",
        });
      });
      await flushEffects();

      expect(harness.getValue().messages[0]?.content).toBe(
        "这是远端刷新后的结果。",
      );
    } finally {
      harness.unmount();
    }
  });

  it("切换到无本地快照的话题时应先进入目标会话加载态", async () => {
    const workspaceId = "ws-topic-history-cold-shell";
    const currentTopicId = "topic-current-shell";
    const topicId = "topic-cold-shell";
    const now = Math.floor(Date.now() / 1000);
    const deferredTopicDetail = createDeferred<{
      id: string;
      workspace_id?: string;
      messages: Array<{
        role: "assistant" | "user";
        timestamp: number;
        content: Array<{ type: "text"; text: string }>;
      }>;
      turns: [];
      items: [];
      execution_strategy: "react";
    }>();
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: currentTopicId,
        name: "当前任务",
        created_at: now - 10,
        messages_count: 1,
        workspace_id: workspaceId,
      },
      {
        id: topicId,
        name: "冷启动历史任务",
        created_at: now,
        messages_count: 12,
        workspace_id: workspaceId,
      },
    ]);
    seedSession(workspaceId, currentTopicId);
    mockGetAgentRuntimeSession.mockImplementation(async (sessionId: string) => {
      if (sessionId === topicId) {
        return deferredTopicDetail.promise;
      }

      return {
        id: sessionId,
        workspace_id: workspaceId,
        messages: [],
        turns: [],
        items: [],
        execution_strategy: "react" as const,
      };
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      mockGetAgentRuntimeSession.mockClear();

      await act(async () => {
        void harness.getValue().switchTopic(topicId);
        await Promise.resolve();
      });

      expect(harness.getValue().sessionId).toBe(topicId);
      expect(harness.getValue().messages).toEqual([]);
      expect(harness.getValue().isSessionHydrating).toBe(true);
      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        topicId,
        expect.objectContaining({ historyLimit: 40 }),
      );

      await act(async () => {
        deferredTopicDetail.resolve({
          id: topicId,
          messages: [
            {
              role: "assistant",
              timestamp: now,
              content: [{ type: "text", text: "冷启动远端结果已加载。" }],
            },
          ],
          turns: [],
          items: [],
          execution_strategy: "react",
        });
      });
      await flushEffects();

      expect(harness.getValue().isSessionHydrating).toBe(false);
      expect(harness.getValue().messages[0]?.content).toBe(
        "冷启动远端结果已加载。",
      );
    } finally {
      harness.unmount();
    }
  });
});
