import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  captureContextCompactionStream,
  captureTurnStream,
  completedTurn,
  createDeferred,
  flushEffects,
  mockCompactAgentRuntimeSession,
  mockCreateAgentRuntimeSession,
  mockGenerateAgentRuntimeSessionTitle,
  mockGetAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockScheduleMinimumDelayIdleTask,
  mockSubmitAgentRuntimeTurn,
  mockToast,
  mockUpdateAgentRuntimeSession,
  mountHook,
  seedSession,
} from "../useAgentChat.testUtils";

describe("useAgentChat slash skill 执行链路", () => {
  it("普通 slash skill 应回到 Agent Runtime turn 主链而不是预执行 execute_skill", async () => {
    const workspaceId = "ws-slash-skill";
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "/content_post_with_cover 写一篇春季新品文案",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]).toEqual(
        expect.objectContaining({
          input: [
            {
              type: "text",
              text: "/content_post_with_cover 写一篇春季新品文案",
            },
          ],
        }),
      );
    } finally {
      harness.unmount();
    }
  });

  it("普通 slash skill 不再尝试旧 preflight，直接进入 Agent Runtime turn", async () => {
    const workspaceId = "ws-slash-fallback";
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "/content_post_with_cover 写一篇春季新品文案",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("无项目普通对话应创建 detached 会话并发送到 Agent Runtime turn", async () => {
    mockCreateAgentRuntimeSession.mockResolvedValue("session-detached-chat");
    const harness = mountHook("");

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("你好", [], false, false, false, "react");
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledWith(
        undefined,
        undefined,
        "react",
        expect.objectContaining({
          runStartHooks: true,
          workingDir: null,
        }),
      );
      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [{ type: "text", text: "你好" }],
          threadId: "session-detached-chat",
        }),
      );
      expect(mockToast.error).not.toHaveBeenCalledWith(
        expect.stringContaining("项目"),
      );
    } finally {
      harness.unmount();
    }
  });

  it("命中 /compact 时应走本地压缩分支而非 chat_stream", async () => {
    const workspaceId = "ws-slash-compact";
    seedSession(workspaceId, "session-slash-compact");
    const harness = mountHook(workspaceId);
    const stream = captureContextCompactionStream();

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .sendMessage("/compact", [], false, false, false, "react");
      });

      expect(mockCompactAgentRuntimeSession).toHaveBeenCalledWith({
        session_id: "session-slash-compact",
        event_name: stream.getEventName(),
      });
      expect(mockSubmitAgentRuntimeTurn).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("命中 /clear 时应清空当前任务且不发送 chat_stream", async () => {
    const workspaceId = "ws-slash-clear";
    seedSession(workspaceId, "session-slash-clear");
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "session-slash-clear",
      messages: [
        {
          role: "assistant",
          timestamp: 1700000001,
          content: [
            {
              type: "output_text",
              text: "hello",
            },
          ],
        },
      ],
      turns: [],
      items: [],
      queued_turns: [],
    });
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      let messages = harness.getValue().messages;
      for (
        let attempt = 0;
        messages.length !== 1 && attempt < 5;
        attempt += 1
      ) {
        await flushEffects();
        messages = harness.getValue().messages;
      }
      expect(messages).toHaveLength(1);

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("/clear", [], false, false, false, "react");
      });

      expect(mockSubmitAgentRuntimeTurn).not.toHaveBeenCalled();
      expect(harness.getValue().messages).toEqual([]);
      expect(harness.getValue().sessionId).toBeNull();
      expect(mockToast.success).toHaveBeenCalledWith("已清空当前任务");
    } finally {
      harness.unmount();
    }
  });

  it("命中 /new 标题 时应创建新任务且不发送 chat_stream", async () => {
    const workspaceId = "ws-slash-new";
    const harness = mountHook(workspaceId);

    mockCreateAgentRuntimeSession.mockResolvedValue("session-slash-new");

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .sendMessage("/new 重构输入命令", [], false, false, false, "react");
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledWith(
        workspaceId,
        "重构输入命令",
        "react",
        {
          runStartHooks: true,
          workingDir: null,
          metadata: expect.objectContaining({
            modelName: expect.any(String),
            providerSelector: expect.any(String),
          }),
        },
      );
      expect(mockSubmitAgentRuntimeTurn).not.toHaveBeenCalled();
      expect(harness.getValue().sessionId).toBe("session-slash-new");
      expect(mockToast.success).not.toHaveBeenCalledWith(
        "已创建新任务：重构输入命令",
      );
    } finally {
      harness.unmount();
    }
  });

  it("并发 ensureSession 应复用同一个新会话创建请求", async () => {
    const deferredSession = createDeferred<string>();
    const harness = mountHook();

    mockCreateAgentRuntimeSession.mockImplementationOnce(
      async () => deferredSession.promise,
    );

    try {
      let firstEnsurePromise: Promise<string | null> | null = null;
      let secondEnsurePromise: Promise<string | null> | null = null;

      await act(async () => {
        firstEnsurePromise = harness.getValue().ensureSession();
        secondEnsurePromise = harness.getValue().ensureSession();
        await Promise.resolve();
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledTimes(1);

      await act(async () => {
        deferredSession.resolve("session-concurrent");
        await Promise.all([firstEnsurePromise, secondEnsurePromise]);
      });

      expect(await firstEnsurePromise).toBe("session-concurrent");
      expect(await secondEnsurePromise).toBe("session-concurrent");

      await act(async () => {
        expect(await harness.getValue().ensureSession()).toBe(
          "session-concurrent",
        );
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("首条发送创建新会话时不应额外回写 provider/model 或 accessMode", async () => {
    const workspaceId = "ws-first-send-no-eager-sync";
    const selectedProvider = "openai";
    const selectedModel = "gpt-5.5";
    localStorage.setItem(
      `agent_pref_provider_${workspaceId}`,
      JSON.stringify(selectedProvider),
    );
    localStorage.setItem(
      `agent_pref_model_${workspaceId}`,
      JSON.stringify(selectedModel),
    );
    const harness = mountHook(workspaceId);

    mockCreateAgentRuntimeSession.mockResolvedValue("session-first-send");

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "请先开始处理这个任务",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledWith(
        workspaceId,
        undefined,
        "react",
        {
          runStartHooks: true,
          workingDir: null,
          metadata: {
            providerSelector: selectedProvider,
            modelName: selectedModel,
            executionRuntime: {
              providerSelector: selectedProvider,
              modelName: selectedModel,
            },
            extensionData: {
              "lime_provider_routing.v0": {
                providerSelector: selectedProvider,
              },
            },
          },
        },
      );
      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]?.runtimeOptions
          ?.runtimeRequest?.providerPreference,
      ).toBeUndefined();
      expect(
        mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]?.runtimeOptions
          ?.runtimeRequest?.modelPreference,
      ).toBeUndefined();
      expect(mockUpdateAgentRuntimeSession).not.toHaveBeenCalledWith({
        session_id: "session-first-send",
        provider_name: harness.getValue().providerType,
        model_name: harness.getValue().model,
      });
      expect(mockUpdateAgentRuntimeSession).not.toHaveBeenCalledWith({
        session_id: "session-first-send",
        recent_access_mode: harness.getValue().accessMode,
      });
      expect(mockListAgentRuntimeSessions).toHaveBeenCalledTimes(1);
    } finally {
      harness.unmount();
    }
  });

  it("首轮发送尚未完成建会话时，后续发送应转入排队并复用同一个会话", async () => {
    const workspaceId = "ws-first-send-gated-queue";
    const deferredSession = createDeferred<string>();
    const harness = mountHook(workspaceId);

    mockCreateAgentRuntimeSession.mockImplementationOnce(
      async () => deferredSession.promise,
    );

    try {
      await flushEffects();

      let firstSendPromise: Promise<void> | null = null;
      await act(async () => {
        firstSendPromise = harness
          .getValue()
          .sendMessage("先处理第一条任务", [], false, false, false, "react");
        await Promise.resolve();
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledTimes(1);
      expect(harness.getValue().messages).toHaveLength(0);

      let secondSendPromise: Promise<void> | null = null;
      await act(async () => {
        secondSendPromise = harness
          .getValue()
          .sendMessage("再补充第二条任务", [], false, false, false, "react");
        await Promise.resolve();
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledTimes(1);
      expect(harness.getValue().messages).toHaveLength(0);
      expect(mockSubmitAgentRuntimeTurn).not.toHaveBeenCalled();

      await act(async () => {
        deferredSession.resolve("session-first-send-gated");
        await Promise.all([firstSendPromise, secondSendPromise]);
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledTimes(1);
      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(2);
      expect(harness.getValue().sessionId).toBe("session-first-send-gated");
    } finally {
      harness.unmount();
    }
  });

  it("新建会话后不应被过期的话题恢复结果抢回旧会话", async () => {
    const workspaceId = "ws-first-send-stale-restore-guard";
    const staleTopicId = "topic-stale-restore";
    const createdSessionId = "session-fresh-after-race";
    const createdAt = Math.floor(Date.now() / 1000);
    const staleTopicDetail = createDeferred<{
      id: string;
      messages: [];
      execution_strategy: "react";
    }>();
    const harness = mountHook(workspaceId);
    let currentSessions = [
      {
        id: staleTopicId,
        name: "旧会话",
        created_at: createdAt,
        messages_count: 0,
        workspace_id: workspaceId,
      },
    ];

    mockListAgentRuntimeSessions.mockImplementation(
      async () => currentSessions,
    );
    mockCreateAgentRuntimeSession.mockImplementation(async () => {
      currentSessions = [
        {
          id: createdSessionId,
          name: "新任务",
          created_at: createdAt + 1,
          messages_count: 0,
          workspace_id: workspaceId,
        },
        ...currentSessions,
      ];
      return createdSessionId;
    });
    mockGetAgentRuntimeSession.mockImplementation((topicId: string) => {
      if (topicId === staleTopicId) {
        return staleTopicDetail.promise;
      }
      return Promise.resolve({
        id: topicId,
        messages: [],
        execution_strategy: "react" as const,
      });
    });

    try {
      await flushEffects();
      await flushEffects();

      let staleSwitchPromise: Promise<void> | null = null;
      await act(async () => {
        staleSwitchPromise = harness.getValue().switchTopic(staleTopicId);
        await Promise.resolve();
      });

      await act(async () => {
        const resolvedSessionId = await harness.getValue().createFreshSession();
        expect(resolvedSessionId).toBe(createdSessionId);
      });
      await flushEffects();

      expect(harness.getValue().sessionId).toBe(createdSessionId);

      await act(async () => {
        staleTopicDetail.resolve({
          id: staleTopicId,
          messages: [],
          execution_strategy: "react",
        });
        await staleSwitchPromise;
      });
      await flushEffects();

      expect(harness.getValue().sessionId).toBe(createdSessionId);

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续发送，必须留在新会话",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [{ type: "text", text: "继续发送，必须留在新会话" }],
          threadId: createdSessionId,
        }),
      );
    } finally {
      harness.unmount();
    }
  });

  it("从旧会话新建任务时应立即清空当前消息，避免新标签继续显示旧对话", async () => {
    const workspaceId = "ws-create-fresh-clears-stale-messages";
    const createdSessionId = "session-fresh-empty-tab";
    mockCreateAgentRuntimeSession.mockResolvedValue(createdSessionId);

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      act(() => {
        harness.getValue().setMessages([
          {
            id: "msg-stale-user",
            role: "user",
            content: "这是旧会话内容",
            timestamp: new Date("2026-04-25T10:00:00.000Z"),
          },
          {
            id: "msg-stale-assistant",
            role: "assistant",
            content: "这是旧会话回复",
            timestamp: new Date("2026-04-25T10:00:01.000Z"),
          },
        ]);
      });
      expect(harness.getValue().messages).toHaveLength(2);

      await act(async () => {
        const newSessionId = await harness.getValue().createFreshSession();
        expect(newSessionId).toBe(createdSessionId);
      });

      expect(harness.getValue().sessionId).toBe(createdSessionId);
      expect(harness.getValue().messages).toEqual([]);
      expect(harness.getValue().turns).toEqual([]);
      expect(harness.getValue().threadItems).toEqual([]);
      expect(mockGenerateAgentRuntimeSessionTitle).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("新建会话后旧会话的流事件不应继续写入当前消息列表", async () => {
    const workspaceId = "ws-create-fresh-detach-old-stream";
    const previousSessionId = "session-old-stream";
    const createdSessionId = "session-fresh-after-stream";
    seedSession(workspaceId, previousSessionId);
    mockCreateAgentRuntimeSession.mockResolvedValue(createdSessionId);
    const stream = captureTurnStream();
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("请继续生成配图", [], false, false, false, "react");
      });

      expect(stream.getEventName()).toMatch(/^agent_stream_/);
      expect(harness.getValue().messages.length).toBeGreaterThan(0);

      await act(async () => {
        const newSessionId = await harness.getValue().createFreshSession();
        expect(newSessionId).toBe(createdSessionId);
      });

      expect(harness.getValue().sessionId).toBe(createdSessionId);
      expect(harness.getValue().messages).toEqual([]);
      expect(harness.getValue().turns).toEqual([]);
      expect(harness.getValue().threadItems).toEqual([]);
      expect(harness.getValue().isSending).toBe(false);

      await act(async () => {
        stream.emit({
          type: "turn_started",
          turn: {
            id: "turn-old-stream-1",
            thread_id: previousSessionId,
            prompt_text: "请继续生成配图",
            status: "running",
            started_at: "2026-05-13T10:00:00.000Z",
            created_at: "2026-05-13T10:00:00.000Z",
            updated_at: "2026-05-13T10:00:00.000Z",
          },
        });
        stream.emit({
          type: "text_delta",
          text: "这段旧流不应该出现在新会话里",
        });
        stream.emit({
          type: "turn_completed",
          turn: completedTurn(),
        });
      });

      expect(harness.getValue().sessionId).toBe(createdSessionId);
      expect(harness.getValue().messages).toEqual([]);
      expect(harness.getValue().turns).toEqual([]);
      expect(harness.getValue().threadItems).toEqual([]);
    } finally {
      harness.unmount();
    }
  });

  it("新建任务返回前不应等待本地持久化回填", async () => {
    const workspaceId = "ws-create-fresh-fast-return";
    const createdSessionId = "session-fresh-fast-return";
    const scheduledTasks: Array<() => void> = [];
    mockCreateAgentRuntimeSession.mockResolvedValue(createdSessionId);
    mockScheduleMinimumDelayIdleTask.mockImplementation((task: () => void) => {
      scheduledTasks.push(task);
      return () => undefined;
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        const newSessionId = await harness
          .getValue()
          .createFreshSession("新对话");
        expect(newSessionId).toBe(createdSessionId);
      });

      expect(harness.getValue().sessionId).toBe(createdSessionId);
      expect(
        sessionStorage.getItem(`agent_curr_sessionId_${workspaceId}`),
      ).toBe(JSON.stringify(createdSessionId));
      expect(localStorage.getItem(`agent_last_sessionId_${workspaceId}`)).toBe(
        JSON.stringify(createdSessionId),
      );
      expect(sessionStorage.getItem(`agent_messages_${workspaceId}`)).toBe(
        null,
      );
      expect(
        localStorage.getItem(
          `agent_topic_model_pref_${workspaceId}_${createdSessionId}`,
        ),
      ).not.toBe(null);
      expect(scheduledTasks).toHaveLength(1);

      act(() => {
        scheduledTasks[0]?.();
      });

      expect(sessionStorage.getItem(`agent_messages_${workspaceId}`)).toBe(
        "[]",
      );
      expect(
        localStorage.getItem(
          `agent_topic_model_pref_${workspaceId}_${createdSessionId}`,
        ),
      ).not.toBe(null);
    } finally {
      harness.unmount();
    }
  });

  it("新建任务后 ensureSession(targetSessionId) 应复用已确认会话", async () => {
    const workspaceId = "ws-created-session-target-fast-path";
    const createdSessionId = "session-created-target-fast-path";
    mockCreateAgentRuntimeSession.mockResolvedValue(createdSessionId);

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        const newSessionId = await harness
          .getValue()
          .createFreshSession("新对话");
        expect(newSessionId).toBe(createdSessionId);
      });

      mockGetAgentRuntimeSession.mockClear();

      await act(async () => {
        const ensuredSessionId = await harness.getValue().ensureSession({
          targetSessionId: createdSessionId,
          skipSessionRestore: true,
          skipSessionStartHooks: true,
        });
        expect(ensuredSessionId).toBe(createdSessionId);
      });

      expect(mockGetAgentRuntimeSession).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });

  it("新建任务失败后应释放创建锁，允许恢复桥接后再次新建", async () => {
    const workspaceId = "ws-create-fresh-retry-after-bridge-error";
    const recoveredSessionId = "session-after-create-retry";
    mockCreateAgentRuntimeSession
      .mockRejectedValueOnce(new Error("bridge health check failed"))
      .mockResolvedValueOnce(recoveredSessionId);

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await expect(
          harness.getValue().createFreshSession("新对话"),
        ).rejects.toThrow("bridge health check failed");
      });

      await act(async () => {
        const recoveredSession = await harness
          .getValue()
          .createFreshSession("新对话");
        expect(recoveredSession).toBe(recoveredSessionId);
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledTimes(2);
      expect(harness.getValue().sessionId).toBe(recoveredSessionId);
    } finally {
      harness.unmount();
    }
  });

  it("手动切换话题时不应继续停留在自动恢复占位态", async () => {
    const workspaceId = "ws-manual-topic-switch-hide-auto-restore";
    const listSessionsDeferred = createDeferred<
      Array<{
        id: string;
        name: string;
        created_at: number;
        messages_count: number;
        workspace_id: string;
      }>
    >();
    const topicDetailDeferred = createDeferred<{
      id: string;
      messages: [];
      execution_strategy: "react";
    }>();

    sessionStorage.setItem(
      `agent_curr_sessionId_${workspaceId}`,
      JSON.stringify("topic-auto-restore"),
    );
    mockListAgentRuntimeSessions.mockImplementation(
      async () => listSessionsDeferred.promise,
    );
    mockGetAgentRuntimeSession.mockImplementation(async (topicId: string) => {
      if (topicId === "topic-manual") {
        return topicDetailDeferred.promise;
      }

      return {
        id: topicId,
        messages: [],
        execution_strategy: "react" as const,
      };
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      expect(harness.getValue().isAutoRestoringSession).toBe(true);

      let switchPromise: Promise<unknown> | null = null;
      await act(async () => {
        switchPromise = harness.getValue().switchTopic("topic-manual");
        await Promise.resolve();
      });

      expect(harness.getValue().isAutoRestoringSession).toBe(false);

      await act(async () => {
        listSessionsDeferred.resolve([]);
        topicDetailDeferred.resolve({
          id: "topic-manual",
          messages: [],
          execution_strategy: "react",
        });
        await switchPromise;
      });
      await flushEffects();

      expect(harness.getValue().sessionId).toBe("topic-manual");
      expect(harness.getValue().isAutoRestoringSession).toBe(false);
    } finally {
      harness.unmount();
    }
  });

  it("命中 /review 时应转换为预置 prompt 后走 chat_stream", async () => {
    const workspaceId = "ws-slash-review";
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .sendMessage("/review lime-rs", [], false, false, false, "react");
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]).toEqual(
        expect.objectContaining({
          input: [
            expect.objectContaining({
              type: "text",
              text: expect.stringContaining("请对以下对象进行代码审查"),
            }),
          ],
        }),
      );
      expect(mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]).toEqual(
        expect.objectContaining({
          input: [
            expect.objectContaining({
              type: "text",
              text: expect.stringContaining("lime-rs"),
            }),
          ],
        }),
      );
    } finally {
      harness.unmount();
    }
  });

  it("命中 /status 时应追加本地 assistant 状态消息", async () => {
    const workspaceId = "ws-slash-status";
    seedSession(workspaceId, "session-slash-status");
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .sendMessage("/status", [], false, false, false, "react");
      });

      const latestMessage =
        harness.getValue().messages[harness.getValue().messages.length - 1];
      expect(latestMessage).toEqual(
        expect.objectContaining({
          role: "assistant",
          content: expect.stringContaining("当前会话状态："),
        }),
      );
      expect(latestMessage?.content).toContain("session-slash-status");
      expect(mockSubmitAgentRuntimeTurn).not.toHaveBeenCalled();
    } finally {
      harness.unmount();
    }
  });
});
