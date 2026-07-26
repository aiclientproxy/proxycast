import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  captureTurnStream,
  createDeferred,
  flushEffects,
  flushRuntimeDetailRefresh,
  mockCreateAgentRuntimeSession,
  mockGetAgentRuntimeSession,
  mockGetAgentRuntimeThreadRead,
  mockInterruptAgentRuntimeTurn,
  mockListAgentRuntimeSessions,
  mockResumeAgentRuntimeThread,
  mockSubmitAgentRuntimeTurn,
  mountHook,
  seedSession,
  seedSessionSnapshots,
} from "../useAgentChat.testUtils";

describe("useAgentChat 任务快照", () => {
  it("停止后刷新会话详情暂未返回历史时，应保留右侧本地对话内容", async () => {
    const workspaceId = "ws-task-stop-refresh-empty-history";
    const sessionId = "session-task-stop-refresh-empty-history";
    captureTurnStream();
    mockCreateAgentRuntimeSession.mockResolvedValue(sessionId);
    mockListAgentRuntimeSessions.mockResolvedValue([]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      name: "当前任务",
      created_at: 1700000300,
      updated_at: 1700000301,
      messages: [],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "帮我继续整理这份任务",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      await flushEffects();

      await act(async () => {
        await harness.getValue().stopSending();
      });

      await flushEffects();
      await flushRuntimeDetailRefresh();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(harness.getValue().messages).toHaveLength(2);
      expect(harness.getValue().messages[0]?.content).toContain(
        "帮我继续整理这份任务",
      );
      expect(harness.getValue().messages[1]?.content).toBe("(已停止)");
    } finally {
      harness.unmount();
    }
  });

  it("恢复态会话执行 stopSending 时也应刷新 thread_read", async () => {
    const workspaceId = "ws-stop-refresh";
    const sessionId = "session-stop-refresh";
    seedSession(workspaceId, sessionId);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [],
      turns: [],
      items: [],
    });
    mockGetAgentRuntimeThreadRead.mockResolvedValue({
      thread_id: "thread-stop-refresh",
      status: "interrupted",
      pending_requests: [],
      incidents: [],
      interrupt_state: "interrupted",
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      await act(async () => {
        await harness.getValue().stopSending();
      });

      await flushRuntimeDetailRefresh();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(mockInterruptAgentRuntimeTurn).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeThreadRead).toHaveBeenCalledWith(sessionId);
      expect(harness.getValue().threadRead).toMatchObject({
        thread_id: "thread-stop-refresh",
        status: "interrupted",
        interrupt_state: "interrupted",
      });
    } finally {
      harness.unmount();
    }
  });

  it("空会话快照稳定后不应继续自发重渲染", async () => {
    const workspaceId = "ws-task-stable";
    const sessionId = "session-task-stable";
    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    mockListAgentRuntimeSessions.mockImplementation(async () => [
      {
        id: sessionId,
        name: "任务稳定性",
        created_at: 1700000100,
        updated_at: 1700000101,
        messages_count: 0,
      },
    ]);
    mockGetAgentRuntimeSession.mockImplementation(async () => ({
      id: sessionId,
      created_at: 1700000100,
      updated_at: 1700000101,
      messages: [],
      turns: [],
      items: [],
    }));

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      await flushEffects();

      let topic = harness
        .getValue()
        .topics.find((item) => item.id === sessionId);
      for (let attempt = 0; !topic && attempt < 3; attempt += 1) {
        await flushEffects();
        topic = harness.getValue().topics.find((item) => item.id === sessionId);
      }
      expect(topic).toBeTruthy();
      expect(topic?.updatedAt.getTime()).toBe(1700000101 * 1000);

      const settledRenderCount = harness.getRenderCount();

      await flushEffects();
      await flushEffects();

      expect(harness.getRenderCount()).toBe(settledRenderCount);
    } finally {
      harness.unmount();
    }
  });

  it("恢复带本地时间线缓存的会话时仍应向后端刷新详情", async () => {
    const workspaceId = "ws-hydrate-timeline-cache";
    const sessionId = "session-hydrate-timeline-cache";
    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    sessionStorage.setItem(
      `agent_messages_${workspaceId}`,
      JSON.stringify([
        {
          id: "msg-local-cache",
          role: "assistant",
          content: "本地缓存里的旧草稿",
          timestamp: new Date().toISOString(),
        },
      ]),
    );
    sessionStorage.setItem(
      `agent_thread_turns_${workspaceId}`,
      JSON.stringify([
        {
          id: "turn-local-cache",
          thread_id: "thread-local-cache",
          prompt_text: "旧的本地缓存 turn",
          status: "completed",
          started_at: "2026-03-26T00:00:00.000Z",
          completed_at: "2026-03-26T00:00:05.000Z",
          created_at: "2026-03-26T00:00:00.000Z",
          updated_at: "2026-03-26T00:00:05.000Z",
        },
      ]),
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "缓存恢复会话",
        created_at: 1700000100,
        updated_at: 1700000101,
        messages_count: 1,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [],
      turns: [],
      items: [],
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
      expect(harness.getValue()).not.toHaveProperty("queuedTurns");
    } finally {
      harness.unmount();
    }
  });

  it("页面刷新恢复到排队会话时只水合运行态，不自动调用 legacy resume", async () => {
    const workspaceId = "ws-auto-resume-after-reload";
    const sessionId = "session-auto-resume-after-reload";
    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "刷新后继续执行",
        created_at: 1700000100,
        updated_at: 1700000101,
        messages_count: 1,
        thread_status: "queued",
        queued_turn_count: 1,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      name: "刷新后继续执行",
      created_at: 1700000100,
      updated_at: 1700000101,
      messages: [],
      turns: [],
      items: [],
      thread_read: {
        thread_id: "thread-after-reload",
        status: "queued",
        pending_requests: [],
        incidents: [],
      },
    });
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      await flushEffects();

      expect(mockResumeAgentRuntimeThread).not.toHaveBeenCalled();
      expect(mockSubmitAgentRuntimeTurn).not.toHaveBeenCalled();
      expect(mockGetAgentRuntimeThreadRead).not.toHaveBeenCalled();
      expect(harness.getValue().threadRead).toMatchObject({
        thread_id: "thread-after-reload",
        status: "queued",
      });
      expect(harness.getValue().queuedTurnCount).toBe(1);
      expect(
        harness.getValue().topics.find((topic) => topic.id === sessionId),
      ).toMatchObject({
        status: "queued",
      });
    } finally {
      harness.unmount();
    }
  });

  it("自动恢复当前会话时应优先回放 session 快照里的用户输入与 Skill 思考", async () => {
    const workspaceId = "ws-auto-restore-skill-snapshot";
    const sessionId = "session-auto-restore-skill-snapshot";
    const now = Date.now();
    const deferredDetail = createDeferred<{
      id: string;
      created_at: number;
      updated_at: number;
      messages_count: number;
      messages: Array<{
        role: "assistant" | "user";
        timestamp: number;
        content: Array<{ type: "text"; text: string }>;
      }>;
      turns: [];
      items: [];
    }>();

    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    seedSessionSnapshots(workspaceId, {
      [sessionId]: {
        messages: [
          {
            id: "cached-analysis-user",
            role: "user",
            content: "@analysis 帮我分析一下今天的国际形势",
            timestamp: "2026-05-13T17:51:40.000Z",
          },
          {
            id: "cached-analysis-assistant",
            role: "assistant",
            content: "# 分析结果\n\n## 结论\n国际形势分析结果。",
            timestamp: "2026-05-13T17:51:42.000Z",
            runtimeTurnId: "skill-exec-cached-analysis-assistant",
            inlineProcessRetention: "skill",
            thinkingContent: "先识别 analysis Skill，再组织结论。",
            contentParts: [
              {
                type: "thinking",
                text: "先识别 analysis Skill，再组织结论。",
              },
              {
                type: "text",
                text: "# 分析结果\n\n## 结论\n国际形势分析结果。",
              },
            ],
          },
        ],
        threadTurns: [],
        threadItems: [],
        currentTurnId: null,
        updatedAt: now,
        lastAccessedAt: now,
        expiresAt: now + 60_000,
        staleUntil: now + 120_000,
        sessionUpdatedAt: now,
        messagesCount: 2,
        historyTruncated: false,
      },
    });
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "Skill 快照恢复",
        created_at: 1700000100,
        updated_at: Math.floor(now / 1000),
        messages_count: 2,
      },
    ]);
    mockGetAgentRuntimeSession.mockReturnValue(deferredDetail.promise);

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      expect(harness.getValue().sessionId).toBe(sessionId);
      expect(harness.getValue().messages[0]).toMatchObject({
        id: "cached-analysis-user",
        role: "user",
        content: "@analysis 帮我分析一下今天的国际形势",
      });
      expect(harness.getValue().messages[1]).toMatchObject({
        id: "cached-analysis-assistant",
        role: "assistant",
        thinkingContent: "先识别 analysis Skill，再组织结论。",
      });

      await act(async () => {
        deferredDetail.resolve({
          id: sessionId,
          created_at: 1700000100,
          updated_at: Math.floor(now / 1000),
          messages_count: 1,
          messages: [
            {
              role: "assistant",
              timestamp: Math.floor(
                new Date("2026-05-13T17:51:45.000Z").getTime() / 1000,
              ),
              content: [
                {
                  type: "text",
                  text: "# 分析结果\n\n## 结论\n国际形势分析结果。",
                },
              ],
            },
          ],
          turns: [],
          items: [],
        });
        await deferredDetail.promise;
      });
      await flushEffects();

      expect(harness.getValue().messages).toHaveLength(2);
      expect(harness.getValue().messages[0]).toMatchObject({
        id: "cached-analysis-user",
        role: "user",
        content: "@analysis 帮我分析一下今天的国际形势",
      });
      expect(harness.getValue().messages[1]).toMatchObject({
        id: "cached-analysis-assistant",
        role: "assistant",
        runtimeTurnId: "skill-exec-cached-analysis-assistant",
        inlineProcessRetention: "skill",
        thinkingContent: "先识别 analysis Skill，再组织结论。",
      });
      expect(harness.getValue().messages[1]?.contentParts?.[0]).toMatchObject({
        type: "thinking",
        text: "先识别 analysis Skill，再组织结论。",
      });
    } finally {
      harness.unmount();
    }
  });

  it("自动恢复当前会话时 scoped 消息缺用户输入，应从 session 快照合并 Skill 输入与思考", async () => {
    const workspaceId = "ws-skill-scoped-cache-merge";
    const sessionId = "session-skill-scoped-cache-merge";
    const now = Date.now();
    const deferredDetail = createDeferred<{
      id: string;
      created_at: number;
      updated_at: number;
      messages_count: number;
      messages: Array<{
        role: "assistant" | "user";
        timestamp: number;
        content: Array<{ type: "text"; text: string }>;
      }>;
      turns: [];
      items: [];
    }>();

    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    sessionStorage.setItem(
      `agent_messages_${workspaceId}`,
      JSON.stringify([
        {
          id: "scoped-analysis-assistant",
          role: "assistant",
          content: "# 分析结果\n\n## 结论\n国际形势分析结果。",
          timestamp: "2026-05-13T17:51:45.000Z",
          contentParts: [
            {
              type: "text",
              text: "# 分析结果\n\n## 结论\n国际形势分析结果。",
            },
          ],
        },
      ]),
    );
    seedSessionSnapshots(workspaceId, {
      [sessionId]: {
        messages: [
          {
            id: "cached-analysis-user-for-scoped",
            role: "user",
            content: "@analysis 帮我分析一下今天的国际形势",
            timestamp: "2026-05-13T17:51:40.000Z",
          },
          {
            id: "cached-analysis-assistant-for-scoped",
            role: "assistant",
            content: "# 分析结果\n\n## 结论\n国际形势分析结果。",
            timestamp: "2026-05-13T17:51:42.000Z",
            runtimeTurnId: "skill-exec-cached-analysis-assistant-for-scoped",
            inlineProcessRetention: "skill",
            thinkingContent: "先识别 analysis Skill，再组织结论。",
            contentParts: [
              {
                type: "thinking",
                text: "先识别 analysis Skill，再组织结论。",
              },
              {
                type: "text",
                text: "# 分析结果\n\n## 结论\n国际形势分析结果。",
              },
            ],
          },
        ],
        threadTurns: [],
        threadItems: [],
        currentTurnId: null,
        updatedAt: now,
        lastAccessedAt: now,
        expiresAt: now + 60_000,
        staleUntil: now + 120_000,
        sessionUpdatedAt: now,
        messagesCount: 2,
        historyTruncated: false,
      },
    });
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "Skill scoped 快照恢复",
        created_at: 1700000100,
        updated_at: Math.floor(now / 1000),
        messages_count: 2,
      },
    ]);
    mockGetAgentRuntimeSession.mockReturnValue(deferredDetail.promise);

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      expect(harness.getValue().messages).toHaveLength(2);
      expect(harness.getValue().messages[0]).toMatchObject({
        id: "cached-analysis-user-for-scoped",
        role: "user",
        content: "@analysis 帮我分析一下今天的国际形势",
      });
      expect(harness.getValue().messages[1]).toMatchObject({
        id: "cached-analysis-assistant-for-scoped",
        role: "assistant",
        runtimeTurnId: "skill-exec-cached-analysis-assistant-for-scoped",
        inlineProcessRetention: "skill",
        thinkingContent: "先识别 analysis Skill，再组织结论。",
      });
    } finally {
      harness.unmount();
    }
  });

  it("恢复会话时远端暂未返回最新 assistant 消息也应保留本地执行过程", async () => {
    const workspaceId = "ws-hydrate-missing-assistant-tail";
    const sessionId = "session-hydrate-missing-assistant-tail";
    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    sessionStorage.setItem(
      `agent_messages_${workspaceId}`,
      JSON.stringify([
        {
          id: "msg-local-user",
          role: "user",
          content: "把文章保存到项目里",
          timestamp: "2026-04-08T10:00:00.000Z",
        },
        {
          id: "msg-local-assistant",
          role: "assistant",
          content: "内容已保存到项目目录。",
          timestamp: "2026-04-08T10:00:02.000Z",
          contentParts: [
            {
              type: "tool_use",
              toolCall: {
                id: "tool-site-tail-1",
                name: "site_run_adapter",
                arguments: '{"url":"https://x.com/example/article/2"}',
                status: "completed",
                startTime: "2026-04-08T10:00:01.000Z",
                endTime: "2026-04-08T10:00:02.000Z",
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
              id: "tool-site-tail-1",
              name: "site_run_adapter",
              arguments: '{"url":"https://x.com/example/article/2"}',
              status: "completed",
              startTime: "2026-04-08T10:00:01.000Z",
              endTime: "2026-04-08T10:00:02.000Z",
              result: {
                success: true,
                output: "saved: articles/google-cloud-tech-2.md",
              },
            },
          ],
        },
      ]),
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "会话恢复缺 assistant",
        created_at: 1700000100,
        updated_at: 1700000101,
        messages_count: 1,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [
        {
          role: "user",
          timestamp: 1712570401,
          content: [{ type: "text", text: "把文章保存到项目里" }],
        },
      ],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      await flushEffects();

      const value = harness.getValue();
      expect(value.messages).toHaveLength(2);
      expect(value.messages[1]?.role).toBe("assistant");
      expect(
        value.messages[1]?.contentParts?.some(
          (part) =>
            part.type === "tool_use" && part.toolCall.id === "tool-site-tail-1",
        ),
      ).toBe(true);
    } finally {
      harness.unmount();
    }
  });

  it("应将当前任务的真实摘要与状态回写到任务列表", async () => {
    const workspaceId = "ws-task-snapshot";
    const sessionId = "session-task-snapshot";
    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    sessionStorage.setItem(
      `agent_messages_${workspaceId}`,
      JSON.stringify([
        {
          id: "msg-task-1",
          role: "assistant",
          content: "请先整理需求清单，再拆出里程碑。",
          timestamp: new Date().toISOString(),
        },
      ]),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [
        {
          role: "assistant",
          timestamp: 1700000001,
          content: [
            {
              type: "output_text",
              text: "请先整理需求清单，再拆出里程碑。",
            },
          ],
        },
      ],
      turns: [],
      items: [],
    });
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "任务 A",
        created_at: 1700000000,
        updated_at: 1700000001,
        messages_count: 1,
      },
    ]);

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      let topic = harness
        .getValue()
        .topics.find((item) => item.id === sessionId);
      for (
        let attempt = 0;
        (!topic || topic.status !== "done") && attempt < 5;
        attempt += 1
      ) {
        await flushEffects();
        topic = harness.getValue().topics.find((item) => item.id === sessionId);
      }
      expect(topic).toBeTruthy();
      expect(topic?.status).toBe("done");
      expect(topic?.messagesCount).toBe(1);
      expect(topic?.lastPreview).toContain("请先整理需求清单");
    } finally {
      harness.unmount();
    }
  });

  it("发送中应将当前任务标记为进行中并同步最新摘要", async () => {
    const workspaceId = "ws-task-running";
    const sessionId = "session-task-running";
    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify(sessionId),
    );
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "任务 B",
        created_at: 1700000010,
        updated_at: 1700000011,
        messages_count: 0,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "帮我输出一版任务拆解",
            [],
            false,
            false,
            false,
            "react",
          );
      });
      await flushEffects();

      const topic = harness
        .getValue()
        .topics.find((item) => item.id === sessionId);
      expect(topic).toBeTruthy();
      expect(topic?.status).toBe("running");
      expect(topic?.messagesCount).toBeGreaterThanOrEqual(1);
      expect(topic?.lastPreview).toContain("帮我输出一版任务拆解");
    } finally {
      harness.unmount();
    }
  });
});
