import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  captureTurnStream,
  completedTurn,
  createDeferred,
  flushEffects,
  mockCreateAgentRuntimeSession,
  mockDeleteAgentRuntimeSession,
  mockGenerateAgentRuntimeSessionTitle,
  mockGetAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockScheduleMinimumDelayIdleTask,
  mockSetAgentRuntimeThreadName,
  mockSubmitAgentRuntimeTurn,
  mountHook,
} from "../useAgentChat.testUtils";

describe("useAgentChat 兼容接口 - topics", () => {
  it("已有 session execution_strategy 时不应重复随 turn 提交 execution_strategy", async () => {
    const workspaceId = "ws-runtime-strategy-reuse";
    const topicId = "topic-runtime-strategy-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      messages: [],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续沿用当前执行策略",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]?.runtimeOptions
          ?.runtimeRequest?.executionStrategy,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("切换 legacy executionStrategy 且 session 同步未完成时也不随 turn 提交 execution_strategy", async () => {
    const workspaceId = "ws-runtime-strategy-pending-sync";
    const topicId = "topic-runtime-strategy-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      messages: [],
      turns: [],
      items: [],
    });
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });

      act(() => {
        harness.getValue().setExecutionStrategy("react");
      });
      await flushEffects();
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("切换执行策略后立即发送", [], false, false, false);
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]?.runtimeOptions
          ?.runtimeRequest?.executionStrategy,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("已有 recent_access_mode 时发送消息应沿用恢复后的正式权限策略", async () => {
    const workspaceId = "ws-runtime-access-restore";
    const topicId = "topic-runtime-access-restore";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_access_mode: "read-only",
        source: "session",
      },
      messages: [],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      expect(harness.getValue().accessMode).toBe("read-only");
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "沿用只读权限继续分析",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]?.approvalPolicy,
      ).toBe("on-request");
      expect(mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]?.permissions).toBe(
        ":read-only",
      );
    } finally {
      harness.unmount();
    }
  });

  it("流式 turn_context / model_change 应更新 executionRuntime，并在结束后仅保留 last runtime", async () => {
    const stream = captureTurnStream();
    const harness = mountHook("ws-execution-runtime");

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().triggerAIGuide("请输出结构化结果");
      });

      await act(async () => {
        stream.emit({
          type: "turn_context",
          session_id: "created-session",
          thread_id: "created-session",
          turn_id: "turn-runtime-1",
          output_schema_runtime: {
            source: "turn",
            strategy: "native",
            providerName: "openai",
            modelName: "gpt-5.4",
          },
        });
      });

      expect(harness.getValue().executionRuntime).toMatchObject({
        session_id: "created-session",
        source: "turn_context",
        provider_name: "openai",
        model_name: "gpt-5.4",
      });
      expect(harness.getValue().activeExecutionRuntime).toMatchObject({
        model_name: "gpt-5.4",
      });

      await act(async () => {
        stream.emit({
          type: "model_change",
          model: "gpt-5.4-mini",
          mode: "responses",
        });
      });

      expect(harness.getValue().executionRuntime).toMatchObject({
        source: "model_change",
        model_name: "gpt-5.4-mini",
        mode: "responses",
      });
      expect(harness.getValue().activeExecutionRuntime).toMatchObject({
        model_name: "gpt-5.4-mini",
      });

      await act(async () => {
        stream.emit({
          type: "turn_completed",
          turn: completedTurn(),
        });
      });

      expect(harness.getValue().executionRuntime).toMatchObject({
        model_name: "gpt-5.4-mini",
      });
      expect(harness.getValue().activeExecutionRuntime).toBeNull();
    } finally {
      harness.unmount();
    }
  });

  it("renameTopic 应调用后端并刷新话题标题", async () => {
    const createdAt = Math.floor(Date.now() / 1000);
    mockListAgentRuntimeSessions
      .mockResolvedValue([
        {
          id: "topic-1",
          name: "新标题",
          created_at: createdAt,
          messages_count: 2,
          workspace_id: "ws-rename",
        },
      ])
      .mockResolvedValueOnce([
        {
          id: "topic-1",
          name: "旧标题",
          created_at: createdAt,
          messages_count: 2,
          workspace_id: "ws-rename",
        },
      ]);

    const harness = mountHook("ws-rename");

    try {
      await flushEffects();
      await flushEffects();

      await act(async () => {
        await harness.getValue().renameTopic("topic-1", "新标题");
      });

      expect(mockSetAgentRuntimeThreadName).toHaveBeenCalledWith({
        threadId: "topic-1",
        name: "新标题",
      });

      const renamedTopic = harness
        .getValue()
        .topics.find((topic) => topic.id === "topic-1");
      expect(renamedTopic?.title).toBe("新标题");
    } finally {
      harness.unmount();
    }
  });

  it("自动标题生成进行中时，话题状态刷新不应取消导航标题回写", async () => {
    const workspaceId = "ws-auto-title";
    const sessionId = "session-auto-title";
    const generatedTitle = "支付页错误定位";
    const deferredTitle = createDeferred<string>();
    mockCreateAgentRuntimeSession.mockResolvedValue(sessionId);
    mockGenerateAgentRuntimeSessionTitle.mockReturnValueOnce(
      deferredTitle.promise,
    );

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().createFreshSession();
      });
      await flushEffects();

      await act(async () => {
        harness.getValue().setMessages([
          {
            id: "msg-user-title",
            role: "user",
            content: "帮我定位支付页提交时报 500 的问题",
            timestamp: new Date(),
          },
          {
            id: "msg-assistant-title",
            role: "assistant",
            content: "我会先检查支付页请求链路。",
            timestamp: new Date(),
          },
        ]);
      });
      await flushEffects();

      expect(mockGenerateAgentRuntimeSessionTitle).toHaveBeenCalledWith(
        sessionId,
        expect.stringContaining("帮我定位支付页提交时报 500 的问题"),
      );

      await act(async () => {
        harness.getValue().updateTopicSnapshot(sessionId, {
          lastPreview: "正在检查支付页请求链路。",
          messagesCount: 2,
          status: "running",
        });
      });

      await act(async () => {
        deferredTitle.resolve(generatedTitle);
        await Promise.resolve();
      });
      await flushEffects();

      expect(mockSetAgentRuntimeThreadName).toHaveBeenCalledWith({
        threadId: sessionId,
        name: generatedTitle,
      });
      expect(
        harness.getValue().topics.find((topic) => topic.id === sessionId)
          ?.title,
      ).toBe(generatedTitle);
    } finally {
      harness.unmount();
    }
  });

  it("自动标题应覆盖新对话占位标题", async () => {
    const workspaceId = "ws-auto-title-new-dialogue";
    const sessionId = "session-new-dialogue-title";
    const generatedTitle = "国际新闻摘要";
    mockCreateAgentRuntimeSession.mockResolvedValue(sessionId);
    mockGenerateAgentRuntimeSessionTitle.mockResolvedValue(generatedTitle);

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().createFreshSession("新对话");
      });
      await flushEffects();

      await act(async () => {
        harness.getValue().setMessages([
          {
            id: "msg-user-new-dialogue-title",
            role: "user",
            content: "请用 WebSearch 查询今天国际新闻，简短列出 3 条。",
            timestamp: new Date(),
          },
        ]);
      });
      await flushEffects();

      expect(mockGenerateAgentRuntimeSessionTitle).toHaveBeenCalledWith(
        sessionId,
        expect.stringContaining("今天国际新闻"),
      );
      expect(mockScheduleMinimumDelayIdleTask).toHaveBeenCalledWith(
        expect.any(Function),
        expect.objectContaining({
          idleTimeoutMs: 2000,
          minimumDelayMs: 400,
        }),
      );
      expect(mockSetAgentRuntimeThreadName).toHaveBeenCalledWith({
        threadId: sessionId,
        name: generatedTitle,
      });
      expect(
        harness.getValue().topics.find((topic) => topic.id === sessionId)
          ?.title,
      ).toBe(generatedTitle);
    } finally {
      harness.unmount();
    }
  });

  it("deleteTopic 应调用后端并刷新话题列表", async () => {
    const createdAt = Math.floor(Date.now() / 1000);
    let currentSessions = [
      {
        id: "topic-1",
        name: "旧标题",
        created_at: createdAt,
        messages_count: 2,
      },
    ];

    mockListAgentRuntimeSessions.mockImplementation(
      async () => currentSessions,
    );
    mockDeleteAgentRuntimeSession.mockImplementation(async () => {
      currentSessions = [];
    });

    const harness = mountHook("ws-delete");

    try {
      await flushEffects();
      await flushEffects();

      await act(async () => {
        await harness.getValue().deleteTopic("topic-1");
      });

      expect(mockDeleteAgentRuntimeSession).toHaveBeenCalledTimes(1);
      expect(mockDeleteAgentRuntimeSession).toHaveBeenCalledWith("topic-1");

      const deletedTopic = harness
        .getValue()
        .topics.find((topic) => topic.id === "topic-1");
      expect(deletedTopic).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });
});
