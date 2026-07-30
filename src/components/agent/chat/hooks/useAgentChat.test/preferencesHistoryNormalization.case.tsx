import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  flushEffects,
  mockGetAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockUpdateAgentRuntimeThreadSettings,
  mountHook,
} from "../useAgentChat.testUtils";

describe("useAgentChat 偏好持久化 - history normalization", () => {
  it("切换话题时多个 fallback 元数据应合并成一次 session update", async () => {
    const workspaceId = "ws-topic-metadata-batch-fallback";
    const topicId = "topic-metadata-batch-fallback";
    localStorage.setItem(
      `agent_topic_model_pref_${workspaceId}_${topicId}`,
      JSON.stringify({
        providerType: "gemini",
        model: "gemini-2.5-pro",
      }),
    );
    localStorage.setItem(
      `agent_session_access_mode_${workspaceId}_${topicId}`,
      JSON.stringify("read-only"),
    );
    localStorage.setItem(
      `agent_execution_strategy_${workspaceId}`,
      JSON.stringify("auto"),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      messages: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      await flushEffects();

      expect(mockUpdateAgentRuntimeThreadSettings).toHaveBeenCalledTimes(1);
      expect(mockUpdateAgentRuntimeThreadSettings).toHaveBeenCalledWith({
        threadId: topicId,
        approvalPolicy: "on-request",
        sandboxPolicy: "read-only",
        modelProvider: "gemini",
        model: "gemini-2.5-pro",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应保留工具调用历史并恢复 elicitation 回答文本", async () => {
    const workspaceId = "ws-history-hydrate";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-history",
      execution_strategy: "react",
      messages: [
        {
          role: "assistant",
          timestamp: now,
          content: [
            {
              type: "tool_request",
              id: "tool-1",
              tool_name: "Ask",
              arguments: { question: "请选择" },
            },
          ],
        },
        {
          role: "user",
          timestamp: now + 1,
          content: [
            {
              type: "action_required",
              action_type: "elicitation_response",
              data: { user_data: { answer: "自动执行（Auto）" } },
            },
          ],
        },
        {
          role: "assistant",
          timestamp: now + 2,
          content: [{ type: "text", text: "已收到你的选择，继续执行。" }],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-history");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(3);
      expect(value.messages[0]).toMatchObject({
        role: "assistant",
      });
      expect(
        value.messages[0]?.contentParts?.some(
          (part) => part.type === "tool_use" && part.toolCall.id === "tool-1",
        ),
      ).toBe(true);
      expect(value.messages[1]).toMatchObject({
        role: "user",
        content: "自动执行（Auto）",
      });
      expect(value.messages[2]).toMatchObject({
        role: "assistant",
        content: "已收到你的选择，继续执行。",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应从 thread_read.thread_items 恢复 revisioned proposed_plan", async () => {
    const workspaceId = "ws-history-thread-read-plan";
    const sessionId = "topic-thread-read-plan";
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "Claw 新闻输入 Electron fixture",
        created_at: 1782259735,
        updated_at: 1782259738,
        messages_count: 2,
        workspace_id: workspaceId,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      thread_id: "thread-read-plan-items",
      created_at: 1782259735,
      updated_at: 1782259738,
      execution_strategy: "react",
      messages_count: 2,
      messages: [],
      items: [],
      turns: [
        {
          id: "turn-plan-1",
          thread_id: "thread-read-plan-items",
          prompt_text: "先给我一个修复计划，不要直接改代码",
          status: "completed",
          started_at: "2026-06-23T10:00:00.000Z",
          completed_at: "2026-06-23T10:00:03.000Z",
          created_at: "2026-06-23T10:00:00.000Z",
          updated_at: "2026-06-23T10:00:03.000Z",
        },
      ],
      thread_read: {
        thread_id: "thread-read-plan-items",
        status: "completed",
        thread_items: [
          {
            id: "item-user-plan-1",
            type: "user_message",
            thread_id: "thread-read-plan-items",
            turn_id: "turn-plan-1",
            sequence: 1,
            status: "completed",
            started_at: "2026-06-23T10:00:00.000Z",
            updated_at: "2026-06-23T10:00:00.000Z",
            completed_at: "2026-06-23T10:00:00.000Z",
            content: "先给我一个修复计划，不要直接改代码",
          },
          {
            id: "item-plan-1",
            type: "plan",
            thread_id: "thread-read-plan-items",
            turn_id: "turn-plan-1",
            sequence: 2,
            status: "completed",
            started_at: "2026-06-23T10:00:01.000Z",
            updated_at: "2026-06-23T10:00:03.000Z",
            completed_at: "2026-06-23T10:00:03.000Z",
            text: [
              "- 确认计划模式请求进入 App Server",
              "- 输出 proposed_plan",
              "- 验证右侧计划轨显示",
            ].join("\n"),
            metadata: {
              source: "proposed_plan",
              revisionId: "proposed_plan:fixture-1",
            },
          },
        ],
      },
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(sessionId);
      });
      await flushEffects();

      const value = harness.getValue();
      expect(value.sessionId).toBe(sessionId);
      expect(value.messages).toHaveLength(1);
      expect(value.messages[0]).toMatchObject({
        role: "user",
        content: "先给我一个修复计划，不要直接改代码",
      });
      expect(value.threadItems.map((item) => item.id)).toEqual([
        "item-user-plan-1",
        "item-plan-1",
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("加载更早历史时应按窗口递增请求，避免一次拉全量", async () => {
    const workspaceId = "ws-full-history";
    const topicId = "topic-full-history";
    const now = Math.floor(Date.now() / 1000);
    const recentMessages = Array.from({ length: 40 }, (_, index) => ({
      id: `recent-${index}`,
      role: (index % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
      timestamp: now - (40 - index),
      content: [
        {
          type: "text",
          text: index === 39 ? "最近一条回复" : `最近历史消息 ${index}`,
        },
      ],
    }));
    const olderPageMessages = Array.from({ length: 50 }, (_, index) => ({
      id: `older-${index}`,
      role: (index % 2 === 0 ? "user" : "assistant") as "user" | "assistant",
      timestamp: now - (90 - index),
      content: [
        {
          type: "text",
          text: index === 0 ? "更早的问题" : `更早历史消息 ${index}`,
        },
      ],
    }));
    mockGetAgentRuntimeSession
      .mockResolvedValueOnce({
        id: topicId,
        created_at: now,
        updated_at: now,
        messages_count: 1,
        history_limit: 40,
        history_cursor: {
          item_cursor: "opaque-item-page-2",
          turn_cursor: null,
          loaded_entry_count: 40,
          loaded_turn_count: 0,
          loaded_item_count: 40,
          has_more: true,
        },
        history_truncated: true,
        execution_strategy: "react",
        messages: recentMessages,
        turns: [],
        items: [],
      })
      .mockResolvedValueOnce({
        id: topicId,
        created_at: now,
        updated_at: now,
        messages_count: 1,
        history_limit: 50,
        history_cursor: {
          item_cursor: "opaque-item-page-3",
          turn_cursor: null,
          loaded_entry_count: 50,
          loaded_turn_count: 0,
          loaded_item_count: 50,
          has_more: true,
        },
        history_truncated: true,
        execution_strategy: "react",
        messages: olderPageMessages,
        turns: [],
        items: [],
      });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      await flushEffects();

      expect(harness.getValue().sessionHistoryWindow).toMatchObject({
        loadedEntries: 40,
        loadedItems: 40,
        loadedTurns: 0,
        itemCursor: "opaque-item-page-2",
        turnCursor: null,
        isLoadingFull: false,
      });

      await act(async () => {
        await harness.getValue().loadFullSessionHistory();
      });
      await flushEffects();

      expect(mockGetAgentRuntimeSession).toHaveBeenLastCalledWith(
        topicId,
        expect.objectContaining({
          historyLimit: 50,
          historyItemCursor: "opaque-item-page-2",
          historyTurnCursor: null,
        }),
      );
      expect(harness.getValue().messages).toHaveLength(90);
      expect(harness.getValue().messages[0]?.content).toBe("更早的问题");
      expect(harness.getValue().messages.at(-1)?.content).toBe("最近一条回复");
      expect(harness.getValue().sessionHistoryWindow).toMatchObject({
        loadedEntries: 90,
        loadedItems: 90,
        loadedTurns: 0,
        itemCursor: "opaque-item-page-3",
        turnCursor: null,
        isLoadingFull: false,
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应恢复 input_image 历史消息", async () => {
    const workspaceId = "ws-history-image";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-image",
      execution_strategy: "react",
      messages: [
        {
          role: "user",
          timestamp: now,
          content: [
            {
              type: "input_text",
              text: "请参考这张图",
            },
            {
              type: "input_image",
              image_url: "data:image/png;base64,aGVsbG8=",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: now + 1,
          content: [{ type: "output_text", text: "已收到图片" }],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-image");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(2);
      expect(value.messages[0]).toMatchObject({
        role: "user",
        content: "请参考这张图",
      });
      expect(value.messages[0]?.images).toHaveLength(1);
      expect(value.messages[0]?.images?.[0]).toMatchObject({
        mediaType: "image/png",
        data: "aGVsbG8=",
      });
      expect(value.messages[1]).toMatchObject({
        role: "assistant",
        content: "已收到图片",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应将仅含 tool_response 协议的空白 user 消息归一为 assistant 轨迹", async () => {
    const workspaceId = "ws-history-empty-user-tool-response";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-empty-user",
      execution_strategy: "react",
      messages: [
        {
          role: "user",
          timestamp: now,
          content: [
            { type: "text", text: "/canvas-design 帮我设计一张科技感的海报" },
          ],
        },
        {
          role: "assistant",
          timestamp: now + 1,
          content: [{ type: "text", text: "我来帮你设计一张科技感的海报！" }],
        },
        {
          role: "user",
          timestamp: now + 2,
          content: [
            {
              type: "tool_response",
              id: "call_xxx",
              success: true,
              output: "",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: now + 3,
          content: [{ type: "text", text: "好的！让我为你创建一张科技海报。" }],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-empty-user");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(2);
      expect(value.messages.map((msg) => msg.role)).toEqual([
        "user",
        "assistant",
      ]);
      expect(value.messages[1]?.content).toContain(
        "我来帮你设计一张科技感的海报！",
      );
      expect(value.messages[1]?.content).toContain(
        "好的！让我为你创建一张科技海报。",
      );
      expect(
        value.messages.some((msg) => msg.content.trim().length === 0),
      ).toBe(false);
      expect(
        value.messages[1]?.contentParts?.some(
          (part) =>
            part.type === "tool_use" &&
            part.toolCall.id === "call_xxx" &&
            part.toolCall.status === "completed",
        ),
      ).toBe(true);
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应从 tool_response 输出中提取图片并写入工具结果", async () => {
    const workspaceId = "ws-history-tool-image";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-tool-image",
      execution_strategy: "react",
      messages: [
        {
          role: "assistant",
          timestamp: now,
          content: [{ type: "text", text: "正在处理海报" }],
        },
        {
          role: "tool",
          timestamp: now + 1,
          content: [
            {
              type: "tool_response",
              id: "tool-image-1",
              success: true,
              output:
                "图片生成完成\ndata:image/png;base64,aGVsbG8=\n你可以继续编辑",
            },
          ],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-tool-image");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(1);
      const toolPart = value.messages[0]?.contentParts?.find(
        (part) =>
          part.type === "tool_use" && part.toolCall.id === "tool-image-1",
      );
      expect(toolPart?.type).toBe("tool_use");
      if (toolPart?.type === "tool_use") {
        expect(toolPart.toolCall.result?.images?.[0]?.src).toBe(
          "data:image/png;base64,aGVsbG8=",
        );
      }
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应从历史 tool_response 元数据恢复图片任务预览", async () => {
    const workspaceId = "ws-history-tool-image-task-preview";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-tool-image-task-preview",
      execution_strategy: "react",
      messages: [
        {
          role: "assistant",
          timestamp: now,
          content: [
            { type: "text", text: "正在生成珠江夜景封面" },
            {
              type: "tool_request",
              id: "tool-image-task-preview-1",
              tool_name: "bash",
              arguments: {
                command:
                  'lime media image generate --prompt "珠江夜景封面" --size 1024x1024 --count 1',
              },
            },
          ],
        },
        {
          role: "tool",
          timestamp: now + 1,
          content: [
            {
              type: "tool_response",
              id: "tool-image-task-preview-1",
              success: true,
              output: "图片任务已提交",
              metadata: {
                task_id: "task-image-history-preview-1",
                task_type: "image_generate",
                status: "succeeded",
                project_id: "project-history-preview-1",
                content_id: "content-history-preview-1",
                requested_count: 1,
                received_count: 1,
              },
            },
          ],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-tool-image-task-preview");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(1);
      expect(value.messages[0]).toMatchObject({
        role: "assistant",
        imageWorkbenchPreview: {
          taskId: "task-image-history-preview-1",
          prompt: "珠江夜景封面",
          status: "complete",
          size: "1024x1024",
          imageCount: 1,
          projectId: "project-history-preview-1",
          contentId: "content-history-preview-1",
        },
      });
      expect(
        value.messages[0]?.contentParts?.some(
          (part) =>
            part.type === "tool_use" &&
            part.toolCall.id === "tool-image-task-preview-1" &&
            part.toolCall.status === "completed",
        ),
      ).toBe(true);
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应清洗 tool_response error 中的 Lime 元数据块", async () => {
    const workspaceId = "ws-history-tool-error-metadata";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-tool-error-metadata",
      execution_strategy: "react",
      messages: [
        {
          role: "assistant",
          timestamp: now,
          content: [{ type: "text", text: "正在连接浏览器" }],
        },
        {
          role: "tool",
          timestamp: now + 1,
          content: [
            {
              type: "tool_response",
              id: "tool-error-1",
              success: true,
              error: [
                "CDP 连接失败，请检查目标页面",
                "",
                "[Lime 工具元数据开始]",
                JSON.stringify({
                  reported_success: false,
                  exit_code: 1,
                  sandboxed: true,
                }),
                "[Lime 工具元数据结束]",
              ].join("\n"),
            },
          ],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-tool-error-metadata");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(1);

      const toolCall = value.messages[0]?.toolCalls?.find(
        (item) => item.id === "tool-error-1",
      );
      expect(toolCall?.status).toBe("failed");
      expect(toolCall?.result?.error).toBe("CDP 连接失败，请检查目标页面");
      expect(toolCall?.result?.error).not.toContain("Lime 工具元数据");
      expect(toolCall?.result?.metadata).toMatchObject({
        reported_success: false,
        exit_code: 1,
        sandboxed: true,
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应合并同一工具调用的 running/completed 轨迹为一条", async () => {
    const workspaceId = "ws-history-tool-dedupe";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-tool-dedupe",
      execution_strategy: "react",
      messages: [
        {
          role: "assistant",
          timestamp: now,
          content: [
            {
              type: "tool_request",
              id: "call_dup_1",
              tool_name: "bash",
              arguments: { command: "echo hi", background: true },
            },
          ],
        },
        {
          role: "user",
          timestamp: now + 1,
          content: [
            {
              type: "tool_response",
              id: "call_dup_1",
              success: true,
              output: "done",
            },
          ],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-tool-dedupe");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(1);

      const toolParts = (value.messages[0]?.contentParts || []).filter(
        (part) => part.type === "tool_use" && part.toolCall.id === "call_dup_1",
      );
      expect(toolParts).toHaveLength(1);
      if (toolParts[0]?.type === "tool_use") {
        expect(toolParts[0].toolCall.status).toBe("completed");
      }
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应合并连续 assistant 历史片段", async () => {
    const workspaceId = "ws-history-merge";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-merge",
      execution_strategy: "react",
      messages: [
        {
          role: "assistant",
          timestamp: now,
          content: [{ type: "text", text: "先执行工具" }],
        },
        {
          role: "tool",
          timestamp: now + 1,
          content: [
            {
              type: "tool_response",
              id: "tool-merge-1",
              success: true,
              output: "ok",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: now + 2,
          content: [{ type: "text", text: "工具执行完成" }],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-merge");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(1);
      expect(value.messages[0]).toMatchObject({
        role: "assistant",
        content: "先执行工具\n\n工具执行完成",
      });
      expect(
        value.messages[0]?.contentParts?.some(
          (part) =>
            part.type === "tool_use" &&
            part.toolCall.id === "tool-merge-1" &&
            part.toolCall.status === "completed",
        ),
      ).toBe(true);
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时应去重相邻重复历史消息", async () => {
    const workspaceId = "ws-history-adjacent-dedupe";
    const now = Math.floor(Date.now() / 1000);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-adjacent-dedupe",
      execution_strategy: "react",
      messages: [
        {
          role: "user",
          timestamp: now,
          content: [{ type: "text", text: "你好" }],
        },
        {
          role: "user",
          timestamp: now + 1,
          content: [{ type: "text", text: "你好" }],
        },
        {
          role: "assistant",
          timestamp: now + 2,
          content: [{ type: "text", text: "你好，我在。" }],
        },
        {
          role: "assistant",
          timestamp: now + 3,
          content: [{ type: "text", text: "你好，我在。" }],
        },
      ],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-adjacent-dedupe");
      });

      const value = harness.getValue();
      expect(value.messages).toHaveLength(2);
      expect(value.messages[0]).toMatchObject({
        role: "user",
        content: "你好",
      });
      expect(value.messages[1]).toMatchObject({
        role: "assistant",
        content: "你好，我在。",
      });
    } finally {
      harness.unmount();
    }
  });
});
