import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  flushEffects,
  mockGetAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockResumeAgentRuntimeThread,
  mockSafeListen,
  mountHook,
  seedSession,
  seedSessionSnapshots,
} from "../useAgentChat.testUtils";

describe("useAgentChat 偏好持久化 - session restore", () => {
  it("恢复候选会话时应先由 runtime 确认工作区归属", async () => {
    const consoleWarnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => {});
    const workspaceId = "ws-restore-runtime-guard";
    const sessionId = "session-restore-runtime-guard";
    seedSession(workspaceId, sessionId);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      workspace_id: "ws-other-runtime",
      messages: [],
      turns: [],
      items: [],
      queued_turns: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(harness.getValue().sessionId).toBeNull();
      expect(
        sessionStorage.getItem(`agent_curr_sessionId_${workspaceId}`),
      ).toBe("null");
      expect(localStorage.getItem(`agent_last_sessionId_${workspaceId}`)).toBe(
        "null",
      );
    } finally {
      consoleWarnSpy.mockRestore();
      harness.unmount();
    }
  });

  it("恢复失效会话时不应请求不存在的会话详情", async () => {
    const workspaceId = "ws-stale-session";
    const staleSessionId = "session-stale";
    const activeSessionId = "session-active";
    const now = Math.floor(Date.now() / 1000);

    seedSession(workspaceId, staleSessionId);
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: activeSessionId,
        name: "可用会话",
        created_at: now - 10,
        updated_at: now,
        messages_count: 1,
        workspace_id: workspaceId,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: activeSessionId,
      created_at: now - 10,
      updated_at: now,
      workspace_id: workspaceId,
      messages: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      await flushEffects();

      expect(
        mockGetAgentRuntimeSession.mock.calls.some(
          ([sessionId]) => sessionId === staleSessionId,
        ),
      ).toBe(false);
      expect(harness.getValue().sessionId).toBe(activeSessionId);
    } finally {
      harness.unmount();
    }
  });

  it("仅 transient 记录的运行中恢复候选未出现在话题列表时，应先读取详情而不是清空", async () => {
    const workspaceId = "ws-transient-running-restore";
    const sessionId = "session-transient-running";
    const turnId = "turn-transient-running";
    const now = Math.floor(Date.now() / 1000);

    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-other",
        name: "其他会话",
        created_at: now - 10,
        updated_at: now,
        messages_count: 1,
        workspace_id: workspaceId,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      created_at: now - 20,
      updated_at: now,
      workspace_id: workspaceId,
      messages: [],
      turns: [
        {
          id: turnId,
          thread_id: "thread-transient-running",
          status: "running",
          created_at: now,
          updated_at: now,
        },
      ],
      items: [],
      queued_turns: [],
      thread_read: {
        thread_id: "thread-transient-running",
        status: "running",
        active_turn_id: turnId,
        turns: [
          {
            turn_id: turnId,
            status: "running",
          },
        ],
      },
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      await flushEffects();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(harness.getValue().sessionId).toBe(sessionId);
      expect(harness.getValue().threadRead?.status).toBe("running");
    } finally {
      harness.unmount();
    }
  });

  it("项目页刷新时应使用 global running 候选恢复 detached 会话并接回输出", async () => {
    const workspaceId = "ws-global-running-restore";
    const sessionId = "session-global-running-in-project";
    const turnId = "turn-global-running-in-project";
    const threadId = "thread-global-running-in-project";
    const now = Math.floor(Date.now() / 1000);

    sessionStorage.setItem(
      "agent_curr_sessionId_global",
      JSON.stringify(sessionId),
    );
    localStorage.setItem(
      "agent_last_sessionId_global",
      JSON.stringify(sessionId),
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-other-project",
        name: "项目里的其他会话",
        created_at: now - 10,
        updated_at: now - 10,
        messages_count: 1,
        workspace_id: workspaceId,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      created_at: now - 20,
      updated_at: now,
      messages: [
        {
          role: "user",
          timestamp: now - 2,
          content: [{ type: "text", text: "刷新后继续输出" }],
        },
      ],
      turns: [
        {
          id: turnId,
          thread_id: threadId,
          status: "running",
          created_at: now - 1,
          updated_at: now,
        },
      ],
      items: [],
      queued_turns: [],
      thread_read: {
        thread_id: threadId,
        status: "running",
        profile_status: "running",
        active_turn_id: turnId,
        turns: [
          {
            turn_id: turnId,
            status: "running",
          },
        ],
      },
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      await flushEffects();
      await flushEffects();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(harness.getValue().sessionId).toBe(sessionId);
      expect(harness.getValue().currentTurnId).toBe(turnId);
      expect(harness.getValue().threadRead).toMatchObject({
        thread_id: threadId,
        status: "running",
        active_turn_id: turnId,
      });
      expect(mockSafeListen).toHaveBeenCalledWith(
        `agentSession/event/${sessionId}`,
        expect.any(Function),
      );
      expect(mockResumeAgentRuntimeThread).toHaveBeenCalledWith({
        threadId,
      });
    } finally {
      harness.unmount();
    }
  });

  it("话题列表应按工作区映射过滤，排除其他项目会话", async () => {
    const workspaceId = "ws-filter-current";
    const createdAt = Math.floor(Date.now() / 1000);

    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "topic-current",
        name: "当前项目话题",
        created_at: createdAt,
        messages_count: 2,
        workspace_id: workspaceId,
      },
      {
        id: "topic-other",
        name: "其他项目话题",
        created_at: createdAt,
        messages_count: 3,
        workspace_id: "ws-filter-other",
      },
      {
        id: "topic-legacy",
        name: "历史未映射话题",
        created_at: createdAt,
        messages_count: 1,
      },
    ]);

    localStorage.setItem(
      "agent_session_workspace_topic-current",
      JSON.stringify("ws-stale-current"),
    );
    localStorage.setItem(
      "agent_session_workspace_topic-other",
      JSON.stringify(workspaceId),
    );

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      expect(harness.getValue().topics.map((topic) => topic.id)).toEqual([
        "topic-current",
      ]);
      expect(
        JSON.parse(
          localStorage.getItem("agent_session_workspace_topic-current") ||
            "null",
        ),
      ).toBe(workspaceId);
      expect(
        JSON.parse(
          localStorage.getItem("agent_session_workspace_topic-other") || "null",
        ),
      ).toBe("ws-filter-other");
    } finally {
      harness.unmount();
    }
  });

  it("切换话题后应恢复各自模型选择", async () => {
    const workspaceId = "ws-topic-memory";
    const createdAt = Math.floor(Date.now() / 1000);

    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "topic-a",
        name: "话题 A",
        created_at: createdAt,
        messages_count: 0,
      },
      {
        id: "topic-b",
        name: "话题 B",
        created_at: createdAt,
        messages_count: 0,
      },
    ]);
    mockGetAgentRuntimeSession.mockImplementation(async (topicId: string) => ({
      id: topicId,
      messages: [],
      execution_strategy: "react",
    }));

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic("topic-a");
      });
      act(() => {
        harness.getValue().setProviderType("gemini");
        harness.getValue().setModel("gemini-2.5-pro");
      });
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic("topic-b");
      });
      act(() => {
        harness.getValue().setProviderType("deepseek");
        harness.getValue().setModel("deepseek-chat");
      });
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic("topic-a");
      });
      await act(async () => {
        await new Promise((resolve) => {
          window.setTimeout(resolve, 0);
        });
      });
      await flushEffects();

      const value = harness.getValue();
      expect(value.providerType).toBe("gemini");
      expect(value.model).toBe("gemini-2.5-pro");
      expect(
        JSON.parse(
          localStorage.getItem(
            `agent_topic_model_pref_${workspaceId}_topic-a`,
          ) || "null",
        ),
      ).toEqual({
        providerType: "gemini",
        model: "gemini-2.5-pro",
      });
      expect(
        JSON.parse(
          localStorage.getItem(
            `agent_topic_model_pref_${workspaceId}_topic-b`,
          ) || "null",
        ),
      ).toEqual({
        providerType: "deepseek",
        model: "deepseek-chat",
      });
    } finally {
      harness.unmount();
    }
  });

  it("选择模型后立即切换话题也应保存当前话题选择", async () => {
    const workspaceId = "ws-topic-memory-immediate";
    const createdAt = Math.floor(Date.now() / 1000);

    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "topic-a",
        name: "话题 A",
        created_at: createdAt,
        messages_count: 0,
      },
      {
        id: "topic-b",
        name: "话题 B",
        created_at: createdAt,
        messages_count: 0,
      },
    ]);
    mockGetAgentRuntimeSession.mockImplementation(async (topicId: string) => ({
      id: topicId,
      messages: [],
      execution_strategy: "react",
    }));

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic("topic-a");
      });

      await act(async () => {
        harness.getValue().setProviderType("zhipu");
        harness.getValue().setModel("glm-4.7");
        await harness.getValue().switchTopic("topic-b");
      });

      await act(async () => {
        harness.getValue().setProviderType("gemini");
        harness.getValue().setModel("gemini-3-pro-preview");
      });
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic("topic-a");
      });
      await flushEffects();

      const value = harness.getValue();
      expect(value.providerType).toBe("zhipu");
      expect(value.model).toBe("glm-4.7");
      expect(
        JSON.parse(
          localStorage.getItem(
            `agent_topic_model_pref_${workspaceId}_topic-a`,
          ) || "null",
        ),
      ).toEqual({
        providerType: "zhipu",
        model: "glm-4.7",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切到其他话题后再恢复历史话题时应优先还原该话题自己的本地快照", async () => {
    const workspaceId = "ws-topic-history-session-snapshot";
    const createdAt = Math.floor(Date.now() / 1000);
    seedSessionSnapshots(workspaceId, {
      "topic-a": {
        messages: [
          {
            id: "topic-a-user-local",
            role: "user",
            content: "把文章保存成 markdown 并下载图片",
            timestamp: "2026-04-09T09:00:00.000Z",
          },
          {
            id: "topic-a-assistant-local",
            role: "assistant",
            content: "内容已保存到项目目录。",
            timestamp: "2026-04-09T09:00:02.000Z",
            thinkingContent: "先恢复 topic-a 自己的本地执行轨迹。",
            contentParts: [
              {
                type: "thinking",
                text: "先恢复 topic-a 自己的本地执行轨迹。",
              },
              {
                type: "tool_use",
                toolCall: {
                  id: "tool-topic-a-1",
                  name: "site_run_adapter",
                  arguments: '{"url":"https://x.com/example/article/a"}',
                  status: "completed",
                  startTime: "2026-04-09T09:00:01.000Z",
                  endTime: "2026-04-09T09:00:02.000Z",
                  result: {
                    success: true,
                    output: "saved: articles/topic-a.md",
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
                id: "tool-topic-a-1",
                name: "site_run_adapter",
                arguments: '{"url":"https://x.com/example/article/a"}',
                status: "completed",
                startTime: "2026-04-09T09:00:01.000Z",
                endTime: "2026-04-09T09:00:02.000Z",
                result: {
                  success: true,
                  output: "saved: articles/topic-a.md",
                },
              },
            ],
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
        messages_count: 1,
        workspace_id: workspaceId,
      },
      {
        id: "topic-b",
        name: "当前任务 B",
        created_at: createdAt,
        messages_count: 1,
        workspace_id: workspaceId,
      },
    ]);
    mockGetAgentRuntimeSession.mockImplementation(async (topicId: string) => {
      if (topicId === "topic-a") {
        return {
          id: "topic-a",
          workspace_id: workspaceId,
          messages: [
            {
              role: "user",
              timestamp: 1712653200,
              content: [
                {
                  type: "text",
                  text: "把文章保存成 markdown 并下载图片",
                },
              ],
            },
          ],
          turns: [],
          items: [],
          queued_turns: [],
          execution_strategy: "react",
        };
      }

      return {
        id: "topic-b",
        workspace_id: workspaceId,
        messages: [
          {
            role: "assistant",
            timestamp: 1712653201,
            content: [{ type: "text", text: "这是另一个话题" }],
          },
        ],
        turns: [],
        items: [],
        queued_turns: [],
        execution_strategy: "react",
      };
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic("topic-b");
      });
      await flushEffects();

      expect(harness.getValue().messages).toHaveLength(1);
      expect(harness.getValue().messages[0]?.content).toBe("这是另一个话题");

      await act(async () => {
        await harness.getValue().switchTopic("topic-a");
      });
      for (
        let attempt = 0;
        attempt < 3 &&
        harness.getValue().messages[1]?.thinkingContent !== undefined;
        attempt += 1
      ) {
        await act(async () => {
          await new Promise((resolve) => {
            window.setTimeout(resolve, 0);
          });
        });
        await flushEffects();
      }

      const value = harness.getValue();
      expect(value.messages).toHaveLength(2);
      expect(value.messages[1]?.content).toBe("内容已保存到项目目录。");
      expect(value.messages[1]?.thinkingContent).toBeUndefined();
      expect(value.messages[1]?.toolCalls).toBeUndefined();
      expect(value.messages[1]?.contentParts).toEqual([
        {
          type: "text",
          text: "内容已保存到项目目录。",
        },
      ]);
      expect(
        value.messages.some((message) => message.content === "这是另一个话题"),
      ).toBe(false);
    } finally {
      harness.unmount();
    }
  });
});
