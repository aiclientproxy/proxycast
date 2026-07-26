import { describe, expect, it } from "vitest";

import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";

import {
  appendTextToParts,
  hydrateSessionDetailMessages,
  mergeHydratedMessagesWithLocalState,
  normalizeHistoryMessages,
} from "./agentChatHistory";

describe("agentChatHistory core hydrate", () => {
  it("追加累计 text_delta 时不应重复吐字", () => {
    expect(
      appendTextToParts([{ type: "text", text: "测试" }], "测试回复已完成"),
    ).toEqual([{ type: "text", text: "测试回复已完成" }]);
  });

  it("历史 output_text 以累计快照存储时应恢复为单份正文", () => {
    const detail: AgentSessionDetail = {
      id: "session-cumulative-text",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000201,
          content: [
            { type: "output_text", text: "测试回复" } as never,
            { type: "output_text", text: "测试回复已完成。" } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-cumulative-text",
    );

    expect(messages[0]?.content).toBe("测试回复已完成。");
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "text",
        text: "测试回复已完成。",
      },
    ]);
  });

  it("hydrate 合并短正文片段时不应把 renderer 最终 text part 截断", () => {
    const fullAnswer =
      "根据我的搜索结果，T30 Pro 更偏向高阶学习规划，T90 更偏向基础学科覆盖和价格平衡。";
    const localMessages = [
      {
        id: "local-user-learning-device",
        role: "user" as const,
        content: "科大讯飞学习机怎么选",
        timestamp: new Date("2026-06-18T08:30:00.000Z"),
      },
      {
        id: "local-assistant-learning-device",
        role: "assistant" as const,
        content: fullAnswer,
        timestamp: new Date("2026-06-18T08:30:20.000Z"),
        contentParts: [
          {
            type: "tool_use" as const,
            toolCall: {
              id: "tool-websearch-learning-device",
              name: "WebSearch",
              arguments: '{"query":"科大讯飞 T30 Pro T90 区别"}',
              status: "completed" as const,
              startTime: new Date("2026-06-18T08:30:02.000Z"),
              endTime: new Date("2026-06-18T08:30:06.000Z"),
              result: {
                success: true,
                output: "搜索完成",
              },
            },
          },
          {
            type: "text" as const,
            text: fullAnswer,
          },
        ],
      },
    ];
    const hydratedMessages = [
      {
        id: "session-learning-device-0",
        role: "user" as const,
        content: "科大讯飞学习机怎么选",
        timestamp: new Date("2026-06-18T08:30:00.000Z"),
      },
      {
        id: "session-learning-device-1",
        role: "assistant" as const,
        content: "根据我",
        timestamp: new Date("2026-06-18T08:30:20.000Z"),
        contentParts: [
          {
            type: "text" as const,
            text: "根据我",
          },
        ],
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );
    const assistantTextParts = mergedMessages[1]?.contentParts?.filter(
      (part) => part.type === "text",
    );

    expect(mergedMessages[1]?.content).toBe(fullAnswer);
    expect(assistantTextParts?.at(-1)).toMatchObject({ text: fullAnswer });
    expect(assistantTextParts?.map((part) => part.text)).not.toContain(
      "根据我",
    );
  });

  it("已完成 assistant 历史消息不应保留 running 工具状态", () => {
    const timestamp = new Date("2026-06-07T10:34:45.000Z");
    const messages = normalizeHistoryMessages([
      {
        id: "history-assistant-news-complete",
        role: "assistant",
        content: "根据多源检索结果，以下是 2026年6月7日 的主要国际新闻整理。",
        timestamp,
        isThinking: false,
        contentParts: [
          {
            type: "text",
            text: "我来搜索今天（2026年6月7日）的国际新闻。",
          },
          {
            type: "tool_use",
            toolCall: {
              id: "tool-web-search-stale-running",
              name: "WebSearch",
              arguments: '{"query":"2026年6月7日 国际新闻"}',
              status: "running",
              startTime: timestamp,
            },
          },
          {
            type: "text",
            text: "根据多源检索结果，以下是 2026年6月7日 的主要国际新闻整理。",
          },
        ],
        toolCalls: [
          {
            id: "tool-web-search-stale-running",
            name: "WebSearch",
            arguments: '{"query":"2026年6月7日 国际新闻"}',
            status: "running",
            startTime: timestamp,
          },
        ],
      },
    ]);

    expect(messages).toHaveLength(1);
    expect(messages[0]?.toolCalls?.[0]).toMatchObject({
      id: "tool-web-search-stale-running",
      status: "completed",
      result: {
        success: true,
        output: "",
      },
    });
    const toolPart = messages[0]?.contentParts?.find(
      (part) => part.type === "tool_use",
    );
    expect(toolPart?.type).toBe("tool_use");
    if (toolPart?.type === "tool_use") {
      expect(toolPart.toolCall.status).toBe("completed");
    }
  });

  it("App Server read detail.messages 当前形状应直接恢复用户与助手消息", () => {
    const detail: AgentSessionDetail = {
      id: "session-app-server-messages",
      created_at: 1,
      updated_at: 2,
      messages_count: 2,
      history_limit: 2,
      history_offset: 0,
      history_cursor: {
        oldest_message_id: null,
        start_index: 0,
        loaded_count: 2,
      },
      history_truncated: false,
      messages: [
        {
          role: "user",
          timestamp: 1780704000,
          runtimeTurnId: "turn-app-server-message-user",
          content: [
            {
              type: "text",
              text: "请整理 App Server 对话历史",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1780704002,
          runtime_turn_id: "turn-app-server-message-assistant",
          content: [
            {
              type: "text",
              text: "已从 App Server detail.messages 读取。",
            },
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-app-server-messages",
    );

    expect(messages).toHaveLength(2);
    expect(messages.map((message) => message.id)).toEqual([
      "session-app-server-messages-0",
      "session-app-server-messages-1",
    ]);
    expect(messages[0]).toMatchObject({
      role: "user",
      content: "请整理 App Server 对话历史",
      contentParts: [
        {
          type: "text",
          text: "请整理 App Server 对话历史",
        },
      ],
      runtimeTurnId: "turn-app-server-message-user",
    });
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content: "已从 App Server detail.messages 读取。",
      contentParts: [
        {
          type: "text",
          text: "已从 App Server detail.messages 读取。",
        },
      ],
      runtimeTurnId: "turn-app-server-message-assistant",
    });
  });

  it("App Server read detail.messages 错序时应按 runtime turn 恢复 user -> assistant", () => {
    const detail: AgentSessionDetail = {
      id: "session-app-server-messages-out-of-order",
      created_at: 1,
      updated_at: 2,
      messages_count: 4,
      history_limit: 4,
      history_offset: 0,
      history_cursor: {
        oldest_message_id: null,
        start_index: 0,
        loaded_count: 4,
      },
      history_truncated: false,
      messages: [
        {
          role: "user",
          timestamp: 1780704000,
          runtimeTurnId: "turn-native-1",
          content: [{ type: "text", text: "第一轮问题" }],
        },
        {
          role: "user",
          timestamp: 1780704002,
          runtimeTurnId: "turn-native-2",
          content: [{ type: "text", text: "第二轮问题" }],
        },
        {
          role: "assistant",
          timestamp: 1780704001,
          runtime_turn_id: "turn-native-1",
          usage: {
            input_tokens: 8,
            output_tokens: 13,
          },
          content: [
            {
              type: "thinking",
              text: "先分析第一轮。",
            },
            {
              type: "text",
              text: "第一轮回答",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1780704003,
          runtime_turn_id: "turn-native-2",
          content: [
            {
              type: "tool_request",
              id: "call-native-2",
              tool_name: "read_file",
              arguments: { path: "/tmp/native-2.txt" },
            } as never,
            {
              type: "tool_response",
              id: "call-native-2",
              success: true,
              output: "第二轮工具输出",
            } as never,
            {
              type: "text",
              text: "第二轮回答",
            },
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-app-server-messages-out-of-order",
    );

    expect(messages.map((message) => message.content)).toEqual([
      "第一轮问题",
      "第一轮回答",
      "第二轮问题",
      "第二轮回答",
    ]);
    expect(messages[1]).toMatchObject({
      role: "assistant",
      runtimeTurnId: "turn-native-1",
      usage: {
        input_tokens: 8,
        output_tokens: 13,
      },
      thinkingContent: "先分析第一轮。",
    });
    expect(messages[3]).toMatchObject({
      role: "assistant",
      runtimeTurnId: "turn-native-2",
    });
    expect(
      messages[3]?.toolCalls?.some(
        (toolCall) =>
          toolCall.id === "call-native-2" &&
          toolCall.name === "read_file" &&
          toolCall.status === "completed",
      ),
    ).toBe(true);
  });

  it("历史 tool_response 应继承同一工具请求参数以恢复文件预览入口", () => {
    const detail: AgentSessionDetail = {
      id: "session-history-tool-response-arguments",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1780704050,
          content: [
            {
              type: "tool_request",
              id: "call-read-imported-preview",
              tool_name: "read_file",
              arguments: {
                path: "/workspace/imported-local-history/docs/imported-preview.md",
              },
            } as never,
            {
              type: "tool_response",
              id: "call-read-imported-preview",
              success: true,
              output: "导入会话 Markdown 预览内容",
            } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-history-tool-response-arguments",
    );

    const completedToolPart = messages[0]?.contentParts?.find(
      (part) =>
        part.type === "tool_use" &&
        part.toolCall.id === "call-read-imported-preview" &&
        part.toolCall.status === "completed",
    );
    expect(completedToolPart?.type).toBe("tool_use");
    if (completedToolPart?.type === "tool_use") {
      expect(completedToolPart.toolCall.arguments).toBe(
        '{"path":"/workspace/imported-local-history/docs/imported-preview.md"}',
      );
    }
  });

  it("App Server read detail.thread_read.thread_items 应恢复 revisioned proposed_plan 历史", () => {
    const detail: AgentSessionDetail = {
      id: "session-thread-read-plan-items",
      thread_id: "thread-read-plan-items",
      created_at: 1,
      updated_at: 2,
      messages: [],
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
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-thread-read-plan-items",
    );

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(messages[0]?.content).toBe("先给我一个修复计划，不要直接改代码");
    const planTextPart = messages[1]?.contentParts?.find(
      (part) => part.type === "text" && part.text.includes("<proposed_plan>"),
    );
    expect(planTextPart).toMatchObject({
      type: "text",
      text: expect.stringContaining("输出 proposed_plan"),
    });
  });

  it("App Server thread_read.tool_calls 应合入已恢复助手消息", () => {
    const detail: AgentSessionDetail = {
      id: "session-app-server-tool-calls",
      thread_id: "thread-app-server-tool-calls",
      created_at: 1,
      updated_at: 2,
      messages_count: 2,
      history_limit: 2,
      history_offset: 0,
      history_cursor: {
        oldest_message_id: null,
        start_index: 0,
        loaded_count: 2,
      },
      history_truncated: false,
      messages: [
        {
          role: "user",
          timestamp: 1780704100,
          content: [
            {
              type: "text",
              text: "生成 TypeScript greeting 代码产物",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1780704102,
          content: [
            {
              type: "text",
              text: "已生成代码产物，可在工作台查看。",
            },
          ],
        },
      ],
      turns: [
        {
          id: "turn-tool-read",
          thread_id: "thread-app-server-tool-calls",
          prompt_text: "生成 TypeScript greeting 代码产物",
          status: "completed",
          started_at: "2026-06-07T10:41:40.000Z",
          completed_at: "2026-06-07T10:41:42.000Z",
          created_at: "2026-06-07T10:41:40.000Z",
          updated_at: "2026-06-07T10:41:42.000Z",
        },
      ],
      thread_read: {
        thread_id: "thread-app-server-tool-calls",
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
            structured_content: {
              summary: "结构化工具事实已保留",
              ids: ["doc-1"],
            },
            success: true,
          },
        ],
      },
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-app-server-tool-calls",
    );

    expect(messages).toHaveLength(2);
    expect(messages[1]).toMatchObject({
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
            structuredContent: {
              summary: "结构化工具事实已保留",
              ids: ["doc-1"],
            },
          },
        },
      ],
    });
    expect(messages[1]?.contentParts?.map((part) => part.type)).toEqual([
      "tool_use",
      "text",
    ]);

    const messagesWithoutTimelineFallback = hydrateSessionDetailMessages(
      detail,
      "session-app-server-tool-calls",
      { includeTimelineFallback: false },
    );

    expect(messagesWithoutTimelineFallback).toHaveLength(2);
    expect(messagesWithoutTimelineFallback[1]).toMatchObject({
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
      messagesWithoutTimelineFallback[1]?.contentParts?.map(
        (part) => part.type,
      ),
    ).toEqual(["tool_use", "text"]);

    const compactMessagesWithoutTimelineFallback = hydrateSessionDetailMessages(
      detail,
      "session-app-server-tool-calls",
      {
        compactCompletedHistory: true,
        includeTimelineFallback: false,
      },
    );

    expect(compactMessagesWithoutTimelineFallback).toHaveLength(2);
    expect(compactMessagesWithoutTimelineFallback[1]).toMatchObject({
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
      compactMessagesWithoutTimelineFallback[1]?.contentParts?.map(
        (part) => part.type,
      ),
    ).toEqual(["tool_use", "text"]);
  });

  it("App Server thread_read 图片工具应合入同一 turn 助手寒暄消息并恢复轻卡", () => {
    const fullOutput = JSON.stringify({
      success: true,
      task_id: "task-thread-read-image-1",
      task_type: "image_generate",
      task_family: "image_generation",
      status: "pending_submit",
      normalized_status: "pending",
      path: ".lime/tasks/image_generate/task-thread-read-image-1.json",
      artifact_path: ".lime/tasks/image_generate/task-thread-read-image-1.json",
      record: {
        payload: {
          prompt: "从花城汇看广州塔的春天照片",
          model: "fal-ai/nano-banana-pro",
          presentation: {
            assistant_intro: "我先按花城汇视角构图，保留春花、广场和广州塔。",
            result_captions: {
              complete: "完成了，花城汇望向广州塔的春日画面已经生成。",
            },
          },
        },
      },
    });
    const detail: AgentSessionDetail = {
      id: "session-thread-read-image-tool-calls",
      thread_id: "thread-thread-read-image-tool-calls",
      created_at: 1,
      updated_at: 2,
      messages_count: 2,
      history_limit: 2,
      history_offset: 0,
      history_cursor: {
        oldest_message_id: null,
        start_index: 0,
        loaded_count: 2,
      },
      history_truncated: false,
      messages: [
        {
          role: "user",
          timestamp: 1780704200,
          runtimeTurnId: "turn-thread-read-image",
          content: [
            {
              type: "text",
              text: "@Nanobanana Pro 生成一张广州塔，从花城汇看过去的春天的照片",
            },
          ],
        },
        {
          role: "assistant",
          timestamp: 1780704202,
          runtime_turn_id: "turn-thread-read-image",
          content: [
            {
              type: "thinking",
              text: "先判断花城汇视角、春花前景和广州塔主体。",
            },
            {
              type: "text",
              text: "我先按花城汇视角构图，保留春花、广场和广州塔。",
            },
          ],
        } as never,
      ],
      turns: [
        {
          id: "turn-thread-read-image",
          thread_id: "thread-thread-read-image-tool-calls",
          prompt_text:
            "@Nanobanana Pro 生成一张广州塔，从花城汇看过去的春天的照片",
          status: "completed",
          started_at: "2026-07-03T13:17:36.000Z",
          completed_at: "2026-07-03T13:17:42.000Z",
          created_at: "2026-07-03T13:17:36.000Z",
          updated_at: "2026-07-03T13:17:42.000Z",
        },
      ],
      thread_read: {
        thread_id: "thread-thread-read-image-tool-calls",
        status: "completed",
        profile_status: "completed",
        turns: [
          {
            turn_id: "turn-thread-read-image",
            status: "completed",
            native_status: "completed",
          },
        ],
        pending_requests: [],
        incidents: [],
        tool_calls: [
          {
            tool_call_id: "tool-thread-read-image",
            turn_id: "turn-thread-read-image",
            tool_name: "lime_create_image_generation_task",
            status: "completed",
            started_at: "2026-07-03T13:17:38.000Z",
            finished_at: "2026-07-03T13:17:39.000Z",
            arguments: {
              prompt: "从花城汇看广州塔的春天照片",
              model: "fal-ai/nano-banana-pro",
            },
            output: fullOutput,
            output_preview: fullOutput.slice(0, 160),
            success: true,
          },
        ],
      },
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-thread-read-image-tool-calls",
    );

    expect(messages).toHaveLength(2);
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content: "我先按花城汇视角构图，保留春花、广场和广州塔。",
      runtimeTurnId: "turn-thread-read-image",
      thinkingContent: "先判断花城汇视角、春花前景和广州塔主体。",
      imageWorkbenchPreview: {
        taskId: "task-thread-read-image-1",
        prompt: "从花城汇看广州塔的春天照片",
        status: "running",
        modelName: "fal-ai/nano-banana-pro",
        caption: "完成了，花城汇望向广州塔的春日画面已经生成。",
      },
    });
    expect(messages[1]?.contentParts?.map((part) => part.type)).toEqual([
      "thinking",
      "tool_use",
      "text",
    ]);
    expect(messages[1]?.toolCalls?.[0]).toMatchObject({
      id: "tool-thread-read-image",
      name: "lime_create_image_generation_task",
      status: "completed",
    });
    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
  });

  it("App Server 图片工具历史应优先用完整 output 恢复图片轻卡", () => {
    const fullOutput = JSON.stringify({
      success: true,
      task_id: "task-history-image-1",
      task_type: "image_generate",
      task_family: "image",
      status: "pending_submit",
      normalized_status: "pending",
      path: ".lime/tasks/image_generate/task-history-image-1.json",
      artifact_path: ".lime/tasks/image_generate/task-history-image-1.json",
      record: {
        payload: {
          prompt: "深圳夏天午后的城市照片",
          count: 1,
        },
      },
    });
    const detail: AgentSessionDetail = {
      id: "session-history-image-tool",
      thread_id: "thread-history-image-tool",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          id: "turn-history-image:user",
          role: "user",
          timestamp: 1_783_057_990,
          content: [
            {
              type: "text",
              text: "@配图 深圳夏天午后的城市照片",
            },
          ] as never,
        },
      ],
      items: [
        {
          id: "tool-history-image",
          type: "tool_call",
          thread_id: "thread-history-image-tool",
          turn_id: "turn-history-image",
          sequence: 4,
          tool_name: "lime_create_image_generation_task",
          status: "completed",
          started_at: "2026-07-03T05:53:18.515Z",
          completed_at: "2026-07-03T05:53:18.572Z",
          updated_at: "2026-07-03T05:53:18.572Z",
          arguments: {
            prompt: "深圳夏天午后的城市照片",
            projectRootPath:
              "/Users/coso/Library/Application Support/lime/projects/demo",
          },
          output: fullOutput,
          output_preview: `${fullOutput.slice(0, 160)}, "record":...[1201]`,
          output_truncated: true,
          success: true,
        } as never,
      ],
      turns: [
        {
          id: "turn-history-image",
          thread_id: "thread-history-image-tool",
          prompt_text: "@配图 深圳夏天午后的城市照片",
          status: "completed",
          started_at: "2026-07-03T05:53:10.393Z",
          completed_at: "2026-07-03T05:53:18.586Z",
          created_at: "2026-07-03T05:53:10.393Z",
          updated_at: "2026-07-03T05:53:18.586Z",
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-history-image-tool",
    );

    expect(messages[1]).toMatchObject({
      role: "assistant",
      imageWorkbenchPreview: {
        taskId: "task-history-image-1",
        prompt: "深圳夏天午后的城市照片",
        status: "running",
        taskFilePath:
          "/Users/coso/Library/Application Support/lime/projects/demo/.lime/tasks/image_generate/task-history-image-1.json",
        artifactPath: ".lime/tasks/image_generate/task-history-image-1.json",
      },
    });
  });

  it("本地历史导入的 detail.items 应按 turn 合入已恢复助手消息", () => {
    const detail: AgentSessionDetail = {
      id: "session-codex-import-timeline",
      thread_id: "thread-codex-import-timeline",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          id: "turn-codex:user",
          role: "user",
          timestamp: 1_781_000_001,
          content: [{ type: "text", text: "run it" }] as never,
        },
        {
          id: "turn-codex:assistant",
          role: "assistant",
          timestamp: 1_781_000_004,
          runtimeTurnId: "turn-codex",
          content: [{ type: "text", text: "done" }] as never,
        } as never,
      ],
      turns: [
        {
          id: "turn-codex",
          thread_id: "thread-codex-import-timeline",
          prompt_text: "run it",
          status: "completed",
          started_at: "2026-06-16T00:00:01.000Z",
          completed_at: "2026-06-16T00:00:04.000Z",
          created_at: "2026-06-16T00:00:01.000Z",
          updated_at: "2026-06-16T00:00:04.000Z",
        },
      ],
      items: [
        {
          id: "reasoning-codex",
          thread_id: "thread-codex-import-timeline",
          turn_id: "turn-codex",
          sequence: 1,
          type: "reasoning",
          status: "completed",
          text: "I need to run the tests first.",
          started_at: "2026-06-16T00:00:01.150Z",
          completed_at: "2026-06-16T00:00:01.150Z",
          updated_at: "2026-06-16T00:00:01.150Z",
        } as never,
        {
          id: "command-codex",
          thread_id: "thread-codex-import-timeline",
          turn_id: "turn-codex",
          sequence: 2,
          type: "command_execution",
          status: "completed",
          command: "npm test",
          cwd: "/workspace/app",
          aggregated_output: "Exit code: 0\nOutput:\nok",
          exit_code: 0,
          metadata: {
            imported: true,
            source_client: "codex",
          },
          started_at: "2026-06-16T00:00:02.000Z",
          completed_at: "2026-06-16T00:00:03.000Z",
          updated_at: "2026-06-16T00:00:03.000Z",
        } as never,
        {
          id: "assistant-codex",
          thread_id: "thread-codex-import-timeline",
          turn_id: "turn-codex",
          sequence: 6,
          type: "agent_message",
          status: "completed",
          text: "done",
          started_at: "2026-06-16T00:00:04.000Z",
          completed_at: "2026-06-16T00:00:04.000Z",
          updated_at: "2026-06-16T00:00:04.000Z",
        } as never,
        {
          id: "patch-codex",
          thread_id: "thread-codex-import-timeline",
          turn_id: "turn-codex",
          sequence: 3,
          type: "patch",
          status: "completed",
          text: "Patch changed /workspace/app/src/lib.rs",
          paths: ["/workspace/app/src/lib.rs"],
          summary: ["/workspace/app/src/lib.rs"],
          success: true,
          stdout: "Success. Updated files",
          started_at: "2026-06-16T00:00:03.100Z",
          completed_at: "2026-06-16T00:00:03.100Z",
          updated_at: "2026-06-16T00:00:03.100Z",
        } as never,
        {
          id: "search-codex",
          thread_id: "thread-codex-import-timeline",
          turn_id: "turn-codex",
          sequence: 4,
          type: "web_search",
          status: "completed",
          query: "npm test failure",
          action: "search_query",
          output: "search result summary",
          started_at: "2026-06-16T00:00:03.200Z",
          completed_at: "2026-06-16T00:00:03.300Z",
          updated_at: "2026-06-16T00:00:03.300Z",
        } as never,
        {
          id: "approval-codex",
          thread_id: "thread-codex-import-timeline",
          turn_id: "turn-codex",
          sequence: 5,
          type: "approval_request",
          status: "completed",
          request_id: "approval-codex",
          action_type: "tool_confirmation",
          tool_name: "exec_command",
          prompt: "Approve imported command: npm test",
          response: { decision: "imported_read_only" },
          started_at: "2026-06-16T00:00:03.400Z",
          completed_at: "2026-06-16T00:00:03.500Z",
          updated_at: "2026-06-16T00:00:03.500Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-codex-import-timeline",
    );

    expect(messages).toHaveLength(2);
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content: "done",
      runtimeTurnId: "turn-codex",
      toolCalls: [
        {
          id: "command-codex",
          name: "exec_command",
          status: "completed",
          result: {
            success: true,
            output: "ok",
            metadata: {
              imported: true,
              source_client: "codex",
              exit_code: 0,
              cwd: "/workspace/app",
            },
          },
        },
        {
          id: "patch-codex",
          name: "apply_patch",
          status: "completed",
        },
        {
          id: "search-codex",
          name: "web_search",
          status: "completed",
        },
      ],
      actionRequests: [
        {
          requestId: "approval-codex",
          status: "submitted",
        },
      ],
    });
    expect(messages[1]?.contentParts?.map((part) => part.type)).toEqual([
      "thinking",
      "tool_use",
      "tool_use",
      "tool_use",
      "action_required",
      "text",
    ]);
    expect(messages[1]?.contentParts?.[1]).toMatchObject({
      type: "tool_use",
      toolCall: {
        id: "command-codex",
        result: {
          metadata: {
            imported: true,
            source_client: "codex",
            exit_code: 0,
            cwd: "/workspace/app",
          },
        },
      },
    });
    expect(messages[1]?.toolCalls?.[0]?.result?.output).not.toContain(
      "Exit code:",
    );
    expect(messages[1]?.toolCalls?.[0]?.result?.output).not.toContain(
      "Output:",
    );
  });
});
