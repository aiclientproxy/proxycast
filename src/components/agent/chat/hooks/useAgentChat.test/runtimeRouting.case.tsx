import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  captureTurnStream,
  completedTurn,
  createDeferred,
  flushEffects,
  flushRuntimeDetailRefresh,
  getAgentStreamTextOverlay,
  mockGetAgentRuntimeSession,
  mockGetAgentRuntimeThreadRead,
  mockReadAgentRuntimeThread,
  mockSubmitAgentRuntimeTurn,
  mockToast,
  mountHook,
  seedSession,
} from "../useAgentChat.testUtils";

describe("useAgentChat runtime routing", () => {
  it("旧搜索开关应由发送边界裁掉且不提交 search_mode", async () => {
    const workspaceId = "ws-search-mode-allowed";
    seedSession(workspaceId, "session-search-mode-allowed");
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "帮我看看今天的黄金价格",
            [],
            true,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [
            {
              type: "text",
              text: "帮我看看今天的黄金价格",
            },
          ],
        }),
      );
      expect(mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]).not.toHaveProperty(
        "searchMode",
      );
    } finally {
      harness.unmount();
    }
  });

  it("runtime_status 与 reasoning_summary_delta 应在 turn_completed 前持续保留", async () => {
    const workspaceId = "ws-runtime-status-stream";
    seedSession(workspaceId, "session-runtime-status-stream");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "请先分析，再决定要不要搜索",
            [],
            true,
            true,
            false,
            "react",
          );
      });

      act(() => {
        stream.emit({
          type: "runtime_status",
          status: {
            phase: "routing",
            title: "已决定：先深度思考",
            detail: "先做更充分的意图理解，再决定是否调用搜索。",
            checkpoints: ["thinking 已开启", "搜索与工具保持候选状态"],
          },
        });
      });

      expect(
        harness
          .getValue()
          .threadItems.some(
            (item) =>
              item.type === "turn_summary" &&
              typeof item.text === "string" &&
              item.text.includes("先深度思考"),
          ),
      ).toBe(true);

      act(() => {
        stream.emit({
          type: "reasoning_summary_delta",
          text: "先判断任务是直接回答还是需要联网。",
          delta: "先判断任务是直接回答还是需要联网。",
          item_id: "reasoning-runtime-status-stream",
          summary_index: 0,
          sequence: 1,
        });
        stream.emit({
          type: "text_delta",
          itemId: "agent-message-final-runtime-status-stream",
          phase: "final_answer",
          text: "我会先分析你的诉求。",
        });
      });

      let assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.runtimeStatus).toMatchObject({
        phase: "routing",
        title: "先深度思考",
      });
      expect(
        assistantMessage?.contentParts?.some(
          (part) =>
            part.type === "thinking" &&
            part.text.includes("先判断任务是直接回答还是需要联网"),
        ),
      ).toBe(true);
      expect(
        `${assistantMessage?.content ?? ""}${
          getAgentStreamTextOverlay(assistantMessage?.id)?.content ?? ""
        }`,
      ).toContain("我会先分析你的诉求。");

      act(() => {
        stream.emit({
          type: "turn_completed",
          turn: completedTurn("turn-real-1"),
        });
      });

      assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.runtimeStatus).toBeUndefined();
      expect(assistantMessage?.isThinking).toBe(false);
      expect(assistantMessage?.content).toContain("我会先分析你的诉求。");
      expect(assistantMessage?.thinkingContent).toContain(
        "先判断任务是直接回答还是需要联网",
      );
      expect(
        assistantMessage?.contentParts?.some(
          (part) =>
            part.type === "thinking" &&
            part.text.includes("先判断任务是直接回答还是需要联网"),
        ),
      ).toBe(true);
      expect(getAgentStreamTextOverlay(assistantMessage?.id)).toBeNull();
    } finally {
      harness.unmount();
    }
  });

  it("并发详情请求晚返回时不应把已完成的连续 delta 截回首段", async () => {
    const workspaceId = "ws-runtime-detail-stale-prefix";
    const sessionId = "session-runtime-detail-stale-prefix";
    const turnId = "turn-runtime-detail-stale-prefix";
    const firstText = "以下是今日国际新闻简要整理：";
    const secondText =
      "全球市场继续关注能源与供应链变化，国际组织呼吁加强协调。CLAW_NEWS_FIXTURE_DONE";
    seedSession(workspaceId, sessionId);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      thread_id: "thread-runtime-detail-stale-prefix",
      created_at: 1,
      updated_at: 1,
      messages: [],
    });
    mockGetAgentRuntimeThreadRead.mockResolvedValue({
      thread_id: "thread-runtime-detail-stale-prefix",
      status: "idle",
      pending_requests: [],
      incidents: [],
      queued_turns: [],
    });
    mockReadAgentRuntimeThread.mockResolvedValue({
      thread: {
        cliVersion: "test",
        createdAt: 0.001,
        cwd: "/tmp/runtime-detail-stale-prefix",
        ephemeral: false,
        id: "thread-runtime-detail-stale-prefix",
        modelProvider: "fixture-provider",
        preview: "",
        sessionId,
        status: { type: "idle" },
        source: "appServer",
        turns: [],
        updatedAt: 0.001,
      },
    });
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();
    const deferredDetail =
      createDeferred<Awaited<ReturnType<typeof mockGetAgentRuntimeSession>>>();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("整理今天的国际新闻", [], false, false, false, "react");
      });

      mockGetAgentRuntimeSession.mockImplementationOnce(
        () => deferredDetail.promise,
      );
      act(() => {
        stream.emit({
          type: "warning",
          message: "触发运行态详情同步",
        });
      });
      await flushRuntimeDetailRefresh();

      act(() => {
        stream.emit({
          type: "text_delta",
          itemId: `agent-message-final-${turnId}`,
          text: firstText,
        });
        stream.emit({
          type: "text_delta",
          itemId: `agent-message-final-${turnId}`,
          text: secondText,
        });
        stream.emit({
          type: "turn_completed",
          turn: completedTurn(turnId),
        });
      });

      expect(
        [...harness.getValue().messages]
          .reverse()
          .find((message) => message.role === "assistant")?.content,
      ).toBe(`${firstText}${secondText}`);

      deferredDetail.resolve({
        id: sessionId,
        thread_id: "thread-runtime-detail-stale-prefix",
        created_at: 1,
        updated_at: 2,
        messages: [
          {
            id: `${turnId}:user`,
            role: "user",
            timestamp: 1,
            content: [{ type: "text", text: "整理今天的国际新闻" }],
          },
          {
            id: `${turnId}:assistant`,
            role: "assistant",
            timestamp: 2,
            content: [{ type: "text", text: firstText }],
          },
        ],
        turns: [
          {
            ...completedTurn(turnId),
            thread_id: "thread-runtime-detail-stale-prefix",
          },
        ],
        items: [
          {
            id: `agent-message-final-${turnId}`,
            thread_id: "thread-runtime-detail-stale-prefix",
            turn_id: turnId,
            sequence: 1,
            type: "agent_message",
            text: firstText,
            phase: "final_answer",
            status: "completed",
            started_at: "2026-07-19T20:57:30.149Z",
            completed_at: "2026-07-19T20:57:30.425Z",
            updated_at: "2026-07-19T20:57:30.425Z",
          },
        ],
        thread_read: {
          thread_id: "thread-runtime-detail-stale-prefix",
          status: "completed",
          active_turn_id: turnId,
        },
      });
      await flushEffects();

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((message) => message.role === "assistant");
      expect(assistantMessage?.content).toBe(`${firstText}${secondText}`);
      expect(
        assistantMessage?.contentParts
          ?.filter((part) => part.type === "text")
          .map((part) => part.text)
          .join(""),
      ).toBe(`${firstText}${secondText}`);
    } finally {
      harness.unmount();
    }
  });

  it("turn_completed 立即刷新会话详情时不应丢掉本地已到达的 reasoning 过程", async () => {
    const workspaceId = "ws-final-done-retain-local-reasoning";
    const sessionId = "session-final-done-retain-local-reasoning";
    seedSession(workspaceId, sessionId);
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [
        {
          role: "user",
          timestamp: 1710000000,
          content: [{ type: "text", text: "请先分析，再回答" }],
        },
        {
          role: "assistant",
          timestamp: 1710000005,
          content: [{ type: "output_text", text: "最终回答。" }],
        },
      ],
      turns: [],
      items: [],
    });

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("请先分析，再回答", [], false, true, false, "react");
      });

      act(() => {
        stream.emit({
          type: "turn_started",
          turn: {
            id: "turn-local-reasoning",
            thread_id: sessionId,
            prompt_text: "请先分析，再回答",
            status: "running",
            started_at: "2026-05-13T04:20:00.000Z",
            created_at: "2026-05-13T04:20:00.000Z",
            updated_at: "2026-05-13T04:20:00.000Z",
          },
        });
        stream.emit({
          type: "item_completed",
          item: {
            id: "reasoning-local-retained",
            thread_id: sessionId,
            turn_id: "turn-local-reasoning",
            sequence: 1,
            status: "completed",
            started_at: "2026-05-13T04:20:00.100Z",
            completed_at: "2026-05-13T04:20:00.900Z",
            updated_at: "2026-05-13T04:20:00.900Z",
            type: "reasoning",
            text: "先分析本轮产品资料整理边界。",
          },
        });
        stream.emit({
          type: "text_delta",
          text: "最终回答。",
        });
        stream.emit({
          type: "turn_completed",
          turn: completedTurn("turn-real-1"),
        });
      });

      await flushEffects();
      await flushRuntimeDetailRefresh();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(harness.getValue().threadItems).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "reasoning-local-retained",
            type: "reasoning",
            status: "completed",
          }),
        ]),
      );
      expect(harness.getValue().turns).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "turn-local-reasoning",
            status: "running",
          }),
        ]),
      );
    } finally {
      harness.unmount();
    }
  });

  it("turn_completed 后应主动刷新会话详情以恢复持久化执行轨迹", async () => {
    const workspaceId = "ws-final-done-refresh";
    const sessionId = "session-final-done-refresh";
    seedSession(workspaceId, sessionId);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [
        {
          role: "user",
          timestamp: 1710000000,
          content: [{ type: "text", text: "请先分析，再回答" }],
        },
        {
          role: "assistant",
          timestamp: 1710000005,
          content: [
            { type: "thinking", thinking: "先分析意图。" },
            { type: "output_text", text: "分析完成，下面是回答。" },
          ],
        },
      ],
      turns: [
        {
          id: "turn-real-1",
          thread_id: sessionId,
          prompt_text: "请先分析，再回答",
          status: "completed",
          started_at: "2026-03-18T09:45:22.762244Z",
          completed_at: "2026-03-18T09:45:54.994500Z",
          created_at: "2026-03-18T09:45:22.762244Z",
          updated_at: "2026-03-18T09:45:54.994500Z",
        },
      ],
      items: [
        {
          id: "turn-summary-real-1",
          thread_id: sessionId,
          turn_id: "turn-real-1",
          sequence: 1,
          status: "completed",
          started_at: "2026-03-18T09:45:22.900000Z",
          completed_at: "2026-03-18T09:45:23.100000Z",
          updated_at: "2026-03-18T09:45:23.100000Z",
          type: "turn_summary",
          text: "已决定：直接回答优先",
        },
        {
          id: "reasoning-real-1",
          thread_id: sessionId,
          turn_id: "turn-real-1",
          sequence: 2,
          status: "completed",
          started_at: "2026-03-18T09:45:23.200000Z",
          completed_at: "2026-03-18T09:45:24.100000Z",
          updated_at: "2026-03-18T09:45:24.100000Z",
          type: "reasoning",
          text: "先分析意图。",
        },
      ],
    });
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("请先分析，再回答", [], false, true, false, "react");
      });

      act(() => {
        stream.emit({
          type: "turn_completed",
          turn: completedTurn("turn-real-1"),
        });
      });

      await flushEffects();
      await flushRuntimeDetailRefresh();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(harness.getValue().currentTurnId).toBe("turn-real-1");
      expect(harness.getValue().threadItems).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "turn-summary-real-1",
            type: "turn_summary",
            status: "completed",
            text: "直接回答优先",
          }),
          expect.objectContaining({
            id: "reasoning-real-1",
            type: "reasoning",
            status: "completed",
          }),
        ]),
      );
    } finally {
      harness.unmount();
    }
  });

  it("turn_completed 前未收到正文时应给出明确失败提示，而不是静默无响应", async () => {
    const workspaceId = "ws-empty-final-response";
    seedSession(workspaceId, "session-empty-final-response");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "帮我汇总今天的国际新闻",
            [],
            true,
            false,
            false,
            "react",
          );
      });

      act(() => {
        stream.emit({
          type: "tool_start",
          tool_name: "WebSearch",
          tool_id: "tool-search-1",
          arguments: JSON.stringify({ query: "今天的国际新闻" }),
        });
        stream.emit({
          type: "tool_end",
          tool_id: "tool-search-1",
          result: {
            success: true,
            output: "https://example.com/world-news",
          },
        });
        stream.emit({
          type: "turn_completed",
          turn: completedTurn(),
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.content).toContain(
        "执行失败：模型未输出最终答复，请重试",
      );
      expect(assistantMessage?.runtimeStatus).toMatchObject({
        phase: "failed",
      });
      expect(mockToast.error).toHaveBeenCalledWith(
        "模型未输出最终答复，请重试",
      );
    } finally {
      harness.unmount();
    }
  });

  it("turn_completed 无正文但已保存站点导出结果时不应误报缺少最终答复", async () => {
    const workspaceId = "ws-empty-final-site-export";
    seedSession(workspaceId, "session-empty-final-site-export");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "请导出这篇文章并保存到项目",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      act(() => {
        stream.emit({
          type: "tool_start",
          tool_name: "site_run_adapter",
          tool_id: "tool-site-export-1",
          arguments: JSON.stringify({
            adapter_name: "x/article-export",
          }),
        });
        stream.emit({
          type: "tool_end",
          tool_id: "tool-site-export-1",
          result: {
            success: true,
            output: "exports/x-article-export/article/index.md",
            metadata: {
              tool_family: "site",
              saved_content: {
                content_id: "content-site-export-1",
                project_id: "project-site-export-1",
                markdown_relative_path:
                  "exports/x-article-export/article/index.md",
                images_relative_dir: "exports/x-article-export/article/images",
              },
            },
          },
        });
        stream.emit({
          type: "turn_completed",
          turn: completedTurn(),
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.content).toContain(
        "本轮执行已完成，详细过程与产物已保留在当前对话中。",
      );
      expect(mockToast.error).not.toHaveBeenCalledWith(
        "模型未输出最终答复，请重试",
      );
    } finally {
      harness.unmount();
    }
  });

  it("turn_completed 无正文且没有真实产物信号时，也应保留本地 assistant 过程并标记失败", async () => {
    const workspaceId = "ws-empty-final-refresh-user-only";
    const sessionId = "session-empty-final-refresh-user-only";
    seedSession(workspaceId, sessionId);
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();
    mockGetAgentRuntimeSession.mockResolvedValueOnce({
      id: sessionId,
      messages: [
        {
          role: "user",
          timestamp: 1710000000,
          content: [{ type: "text", text: "导出并保存这篇文章" }],
        },
      ],
      turns: [],
      items: [],
    });

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("导出并保存这篇文章", [], false, false, false, "react");
      });

      act(() => {
        stream.emit({
          type: "tool_start",
          tool_name: "site_run_adapter",
          tool_id: "tool-site-refresh-1",
          arguments: JSON.stringify({ url: "https://x.com/example/article/1" }),
        });
        stream.emit({
          type: "tool_end",
          tool_id: "tool-site-refresh-1",
          result: {
            success: true,
            output: "saved: articles/example.md",
          },
        });
        stream.emit({
          type: "turn_completed",
          turn: completedTurn(),
        });
      });

      await flushEffects();
      await flushRuntimeDetailRefresh();

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");
      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(assistantMessage).toBeTruthy();
      expect(assistantMessage?.content).toContain(
        "执行失败：模型未输出最终答复，请重试",
      );
      expect(assistantMessage?.runtimeStatus).toMatchObject({
        phase: "failed",
      });
      expect(
        assistantMessage?.contentParts?.some(
          (part) =>
            part.type === "tool_use" &&
            part.toolCall.id === "tool-site-refresh-1",
        ),
      ).toBe(true);
      expect(mockToast.error).toHaveBeenCalledWith(
        "模型未输出最终答复，请重试",
      );
    } finally {
      harness.unmount();
    }
  });

  it("stream error 命中空最终答复且没有真实产物信号时应落成失败态", async () => {
    const workspaceId = "ws-thread-empty-final-reply-soft-complete";
    seedSession(workspaceId, "session-thread-empty-final-reply-soft-complete");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "请继续导出并保存内容",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      act(() => {
        stream.emit({
          type: "turn_started",
          turn: {
            id: "turn-empty-final-reply-soft-complete-1",
            thread_id: "session-thread-empty-final-reply-soft-complete",
            prompt_text: "请继续导出并保存内容",
            status: "running",
            started_at: "2026-03-20T10:00:00.000Z",
            created_at: "2026-03-20T10:00:00.000Z",
            updated_at: "2026-03-20T10:00:00.000Z",
          },
        });
        stream.emit({
          type: "tool_start",
          tool_name: "site_run_adapter",
          tool_id: "tool-site-run-1",
          arguments: JSON.stringify({ adapter_name: "x/article-export" }),
        });
        stream.emit({
          type: "tool_end",
          tool_id: "tool-site-run-1",
          result: {
            success: true,
            output: "saved/x-article-export/index.md",
          },
        });
        stream.emit({
          type: "turn_completed",
          turn: {
            id: "turn-empty-final-reply-soft-complete-1",
            thread_id: "session-thread-empty-final-reply-soft-complete",
            prompt_text: "请继续导出并保存内容",
            status: "completed",
            started_at: "2026-03-20T10:00:00.000Z",
            completed_at: "2026-03-20T10:00:05.000Z",
            created_at: "2026-03-20T10:00:00.000Z",
            updated_at: "2026-03-20T10:00:05.000Z",
          },
        });
        stream.emit({
          type: "error",
          message:
            "已完成当前回合的工具执行，但模型未输出最终答复。\n尝试记录: 已执行非联网工具（tool_start=1, tool_end=1）",
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.content).toContain(
        "执行失败：模型未输出最终答复，请重试",
      );
      expect(assistantMessage?.runtimeStatus).toMatchObject({
        phase: "failed",
      });
      expect(harness.getValue().turns).toEqual([
        expect.objectContaining({
          id: "turn-empty-final-reply-soft-complete-1",
          status: "completed",
        }),
      ]);
      expect(mockToast.error).toHaveBeenCalledWith(
        "模型未输出最终答复，请重试",
      );
    } finally {
      harness.unmount();
    }
  });

  it("收到通用任务 tool_end 后应把任务预览挂到当前助手消息", async () => {
    const workspaceId = "ws-generic-task-preview";
    seedSession(workspaceId, "session-generic-task-preview");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "帮我找一组咖啡馆木桌背景素材",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      act(() => {
        stream.emit({
          type: "tool_start",
          tool_name: "Bash",
          tool_id: "tool-resource-task-1",
          arguments: JSON.stringify({
            command:
              "lime task create resource-search --json --resource-type image --query 'cozy coffee table'",
          }),
        });
        stream.emit({
          type: "tool_end",
          tool_id: "tool-resource-task-1",
          result: {
            success: true,
            output: "任务已提交",
            metadata: {
              task_id: "task-resource-1",
              task_type: "modal_resource_search",
              status: "pending_submit",
              artifact_path:
                ".lime/tasks/modal_resource_search/task-resource-1.json",
              project_id: "project-resource-1",
              content_id: "content-resource-1",
            },
          },
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.taskPreview).toMatchObject({
        kind: "modal_resource_search",
        taskId: "task-resource-1",
        taskType: "modal_resource_search",
        status: "running",
        artifactPath: ".lime/tasks/modal_resource_search/task-resource-1.json",
        projectId: "project-resource-1",
        contentId: "content-resource-1",
      });
    } finally {
      harness.unmount();
    }
  });

  it("收到联网搜图 tool_end 后应挂载素材预览并生成可打开的 artifact", async () => {
    const workspaceId = "ws-web-image-search-preview";
    seedSession(workspaceId, "session-web-image-search-preview");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "帮我找一组咖啡馆木桌背景图",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      act(() => {
        stream.emit({
          type: "tool_start",
          tool_name: "lime_search_web_images",
          tool_id: "tool-web-search-1",
          arguments: JSON.stringify({
            query: "cozy coffee table",
            count: 3,
            aspect: "landscape",
          }),
        });
        stream.emit({
          type: "tool_end",
          tool_id: "tool-web-search-1",
          result: {
            success: true,
            output: "已返回 3 张候选图片",
            metadata: {
              provider: "pexels",
              result: {
                provider: "pexels",
                query: "cozy coffee table",
                returnedCount: 3,
                aspect: "landscape",
                hits: [
                  {
                    id: "hit-1",
                    thumbnail_url: "https://pexels.example/1-thumb.jpg",
                    content_url: "https://pexels.example/1.jpg",
                    width: 1600,
                    height: 900,
                    name: "cozy coffee table 1",
                    host_page_url: "https://www.pexels.com/photo/1",
                  },
                  {
                    id: "hit-2",
                    thumbnail_url: "https://pexels.example/2-thumb.jpg",
                    content_url: "https://pexels.example/2.jpg",
                    width: 1600,
                    height: 900,
                    name: "cozy coffee table 2",
                    host_page_url: "https://www.pexels.com/photo/2",
                  },
                  {
                    id: "hit-3",
                    thumbnail_url: "https://pexels.example/3-thumb.jpg",
                    content_url: "https://pexels.example/3.jpg",
                    width: 1600,
                    height: 900,
                    name: "cozy coffee table 3",
                    host_page_url: "https://www.pexels.com/photo/3",
                  },
                ],
              },
            },
          },
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.taskPreview).toMatchObject({
        kind: "modal_resource_search",
        taskId: "resource-search:tool-web-search-1",
        taskType: "modal_resource_search",
        status: "complete",
        artifactPath: ".lime/runtime/resource-search/tool-web-search-1.md",
        metaItems: expect.arrayContaining(["Pexels", "3 个候选"]),
      });
      expect(
        assistantMessage?.taskPreview &&
          "imageCandidates" in assistantMessage.taskPreview
          ? assistantMessage.taskPreview.imageCandidates
          : undefined,
      ).toHaveLength(3);
      expect(assistantMessage?.artifacts?.[0]?.meta).toMatchObject({
        artifactKind: "brief",
        provider: "pexels",
        query: "cozy coffee table",
        returnedCount: 3,
      });
    } finally {
      harness.unmount();
    }
  });
});
