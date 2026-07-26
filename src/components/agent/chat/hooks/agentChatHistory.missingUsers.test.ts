import { beforeEach, describe, expect, it } from "vitest";

import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import { changeLimeLocale } from "@/i18n/createI18n";

import {
  hydrateSessionDetailMessages,
  mergeHydratedMessagesWithLocalState,
  shouldCompactCompletedSessionHistory,
} from "./agentChatHistory";
import { mergeMissingUserMessagesFromTimeline } from "./agentChatHistoryTimelineMerge";

describe("agentChatHistory missing user recovery", () => {
  const internalRuntimeErrorMessage =
    "运行时返回内部错误，已保留详情用于排查。请稍后重试，或检查服务商与工具连接状态。";

  beforeEach(async () => {
    await changeLimeLocale("zh-CN");
  });

  it("带附件的 canonical 用户消息不应再从同一 runtime turn 补一份无附件副本", () => {
    const messages = [
      {
        id: "loaded-user-1",
        role: "user" as const,
        content: "第一轮",
        images: [
          {
            data: "",
            mediaType: "image/png",
            sourcePath: "/tmp/first.png",
          } as never,
        ],
        timestamp: new Date("2026-07-16T08:00:00.000Z"),
        runtimeTurnId: "turn-1",
      },
      {
        id: "loaded-assistant-1",
        role: "assistant" as const,
        content: "",
        contentParts: [
          {
            type: "tool_use" as const,
            toolCall: {
              id: "tool-1",
              name: "exec_command",
              status: "completed" as const,
            },
          },
        ],
        timestamp: new Date("2026-07-16T08:00:01.000Z"),
        runtimeTurnId: "turn-1",
      },
      {
        id: "loaded-user-2",
        role: "user" as const,
        content: "第二轮",
        images: [
          {
            data: "",
            mediaType: "image/png",
            sourcePath: "/tmp/second.png",
          } as never,
        ],
        timestamp: new Date("2026-07-16T08:00:02.000Z"),
        runtimeTurnId: "turn-2",
      },
      {
        id: "loaded-assistant-2",
        role: "assistant" as const,
        content: "",
        contentParts: [
          {
            type: "tool_use" as const,
            toolCall: {
              id: "tool-2",
              name: "exec_command",
              status: "completed" as const,
            },
          },
        ],
        timestamp: new Date("2026-07-16T08:00:03.000Z"),
        runtimeTurnId: "turn-2",
      },
    ];
    const detail = {
      id: "session-attachment-users",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "第一轮",
          status: "completed",
          started_at: "2026-07-16T08:00:00.000Z",
          created_at: "2026-07-16T08:00:00.000Z",
          updated_at: "2026-07-16T08:00:01.000Z",
        },
        {
          id: "turn-2",
          thread_id: "thread-1",
          prompt_text: "第二轮",
          status: "completed",
          started_at: "2026-07-16T08:00:02.000Z",
          created_at: "2026-07-16T08:00:02.000Z",
          updated_at: "2026-07-16T08:00:03.000Z",
        },
      ],
      items: [],
    } as unknown as AgentSessionDetail;

    const merged = mergeMissingUserMessagesFromTimeline(
      messages,
      detail,
      detail.id,
    );

    expect(merged.filter((message) => message.role === "user")).toHaveLength(2);
    expect(merged.map((message) => message.id)).toEqual([
      "loaded-user-1",
      "loaded-assistant-1",
      "loaded-user-2",
      "loaded-assistant-2",
    ]);
  });

  it("App Server failed read model 应恢复用户请求并追加失败助手消息", () => {
    const detail: AgentSessionDetail = {
      id: "session-app-server-failed-read",
      thread_id: "thread-app-server-failed-read",
      created_at: 1,
      updated_at: 2,
      messages_count: 1,
      history_limit: 80,
      history_offset: 0,
      history_cursor: {
        oldest_message_id: null,
        start_index: 0,
        loaded_count: 1,
      },
      history_truncated: false,
      messages: [
        {
          role: "user",
          timestamp: 1780834200,
          content: [
            {
              type: "text",
              text: "整理今天的国际新闻",
            },
          ],
        },
      ],
      turns: [
        {
          id: "turn-news-failed",
          thread_id: "thread-app-server-failed-read",
          prompt_text: "整理今天的国际新闻",
          status: "failed",
          error_message:
            "Request failed: failed to connect to token-plan-cn.xiaomimimo.com",
          started_at: "2026-06-07T09:30:00.000Z",
          completed_at: "2026-06-07T09:30:12.000Z",
          created_at: "2026-06-07T09:30:00.000Z",
          updated_at: "2026-06-07T09:30:12.000Z",
        },
      ],
      items: [
        {
          id: "item-news-error",
          thread_id: "thread-app-server-failed-read",
          turn_id: "turn-news-failed",
          sequence: 3,
          type: "error",
          status: "failed",
          message:
            "Request failed: failed to connect to token-plan-cn.xiaomimimo.com",
          started_at: "2026-06-07T09:30:12.000Z",
          completed_at: "2026-06-07T09:30:12.000Z",
          updated_at: "2026-06-07T09:30:12.000Z",
        } as never,
      ],
      thread_read: {
        thread_id: "thread-app-server-failed-read",
        status: "failed",
        profile_status: "failed",
        active_turn_id: undefined,
        turns: [
          {
            turn_id: "turn-news-failed",
            status: "failed",
            native_status: "failed",
          },
        ],
        pending_requests: [],
        incidents: [],
        diagnostics: {
          latest_turn_status: "failed",
          latest_turn_started_at: "2026-06-07T09:30:00.000Z",
          latest_turn_completed_at: "2026-06-07T09:30:12.000Z",
          latest_turn_updated_at: "2026-06-07T09:30:12.000Z",
          latest_turn_error_message:
            "Request failed: failed to connect to token-plan-cn.xiaomimimo.com",
          warning_count: 0,
          context_compaction_count: 0,
          failed_tool_call_count: 0,
          failed_command_count: 0,
          pending_request_count: 0,
        },
      },
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-app-server-failed-read",
    );

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      role: "user",
      content: "整理今天的国际新闻",
    });
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content: expect.stringContaining("执行失败："),
      isThinking: false,
      runtimeTurnId: "turn-news-failed",
      runtimeStatus: {
        phase: "failed",
        title: "当前处理失败",
        detail: internalRuntimeErrorMessage,
      },
    });
    expect(messages[1]?.content).toBe(
      `执行失败：${internalRuntimeErrorMessage}`,
    );
    expect(messages[1]?.content).not.toContain("token-plan-cn.xiaomimimo.com");
    expect(messages[1]?.contentParts).toEqual([
      {
        type: "text",
        text: messages[1]?.content,
      },
    ]);
  });

  it("后端 messages 只有助手图片轨迹时应从真实 turn 补回用户指令", () => {
    const detail: AgentSessionDetail = {
      id: "session-image-history-user-fallback",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000005,
          content: [
            {
              type: "output_text",
              text: "好啊，生成：画一张三国主要人物的群像海报",
            } as never,
            {
              type: "tool_request",
              id: "tool-image-history-user-fallback",
              tool_name: "lime_create_image_generation_task",
              arguments: {
                prompt: "画一张三国主要人物的群像海报，电影感，国风，高清",
                size: "1024x1024",
              },
            } as never,
            {
              type: "tool_response",
              id: "tool-image-history-user-fallback",
              success: true,
              output: "图片任务已提交",
              metadata: {
                task_id: "task-image-history-user-fallback",
                task_type: "image_generate",
                status: "succeeded",
                requested_count: 1,
                received_count: 1,
              },
            } as never,
          ],
        },
      ],
      turns: [
        {
          id: "turn-image-history-user-fallback",
          thread_id: "session-image-history-user-fallback",
          prompt_text: "@配图 画一张三国主要人物的群像海报，电影感，国风，高清",
          status: "completed",
          started_at: "2026-05-06T10:00:00.000Z",
          completed_at: "2026-05-06T10:00:05.000Z",
          created_at: "2026-05-06T10:00:00.000Z",
          updated_at: "2026-05-06T10:00:05.000Z",
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-image-history-user-fallback",
    );

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(messages[0]).toMatchObject({
      role: "user",
      content: "@配图 画一张三国主要人物的群像海报，电影感，国风，高清",
    });
    expect(messages[1]).toMatchObject({
      role: "assistant",
      imageWorkbenchPreview: {
        taskId: "task-image-history-user-fallback",
        prompt: "画一张三国主要人物的群像海报，电影感，国风，高清",
        status: "complete",
      },
    });
    expect(messages[1]?.content).toContain("三国主要人物的群像海报");
  });

  it("current read model 图片工具历史恢复时应补回前置寒暄", () => {
    const detail: AgentSessionDetail = {
      id: "session-image-thread-item-intro",
      thread_id: "thread-image-thread-item-intro",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-image-thread-item-intro",
          thread_id: "thread-image-thread-item-intro",
          prompt_text: "画一张广州夏天的图",
          status: "completed",
          started_at: "2026-05-06T10:00:00.000Z",
          completed_at: "2026-05-06T10:00:05.000Z",
          created_at: "2026-05-06T10:00:00.000Z",
          updated_at: "2026-05-06T10:00:05.000Z",
        },
      ],
      items: [
        {
          id: "item-image-thread-item-user",
          thread_id: "thread-image-thread-item-intro",
          turn_id: "turn-image-thread-item-intro",
          sequence: 1,
          type: "user_message",
          status: "completed",
          text: "画一张广州夏天的图",
          started_at: "2026-05-06T10:00:00.000Z",
          completed_at: "2026-05-06T10:00:00.000Z",
          updated_at: "2026-05-06T10:00:00.000Z",
        } as never,
        {
          id: "tool-image-thread-item-intro",
          thread_id: "thread-image-thread-item-intro",
          turn_id: "turn-image-thread-item-intro",
          sequence: 2,
          type: "tool_call",
          status: "completed",
          tool_name: "lime_create_image_generation_task",
          arguments: {
            prompt: "画一张广州夏天的图",
            model: "gpt-image-1",
          },
          output: "图片任务已提交",
          success: true,
          metadata: {
            task_id: "task-image-thread-item-intro",
            task_type: "image_generate",
            status: "succeeded",
            received_count: 1,
            model: "gpt-image-1",
          },
          started_at: "2026-05-06T10:00:01.000Z",
          completed_at: "2026-05-06T10:00:05.000Z",
          updated_at: "2026-05-06T10:00:05.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-image-thread-item-intro",
    );

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(messages[1]).toMatchObject({
      role: "assistant",
      imageWorkbenchPreview: {
        taskId: "task-image-thread-item-intro",
        prompt: "画一张广州夏天的图",
        status: "complete",
      },
    });
    expect(messages[1]?.content).toContain("广州夏天");
  });

  it("后端连续两轮只有助手图片轨迹时应按 turn 顺序补回各自用户指令", () => {
    const detail: AgentSessionDetail = {
      id: "session-image-history-two-turns",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000005,
          content: [
            {
              type: "output_text",
              text: "好啊，生成：广州塔春天照片",
            } as never,
            {
              type: "tool_response",
              id: "tool-image-history-two-turns-1",
              success: true,
              output: "图片任务已提交",
              metadata: {
                task_id: "task-image-history-two-turns-1",
                task_type: "image_generate",
                status: "succeeded",
                received_count: 1,
              },
            } as never,
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000015,
          content: [
            {
              type: "output_text",
              text: "好啊，生成：青柠极简插画",
            } as never,
            {
              type: "tool_response",
              id: "tool-image-history-two-turns-2",
              success: true,
              output: "图片任务已提交",
              metadata: {
                task_id: "task-image-history-two-turns-2",
                task_type: "image_generate",
                status: "succeeded",
                received_count: 1,
              },
            } as never,
          ],
        },
      ],
      turns: [
        {
          id: "turn-image-history-two-turns-1",
          thread_id: "session-image-history-two-turns",
          prompt_text: "@配图 生成一张广州塔春天照片",
          status: "completed",
          started_at: "2026-05-06T10:00:00.000Z",
          completed_at: "2026-05-06T10:00:05.000Z",
          created_at: "2026-05-06T10:00:00.000Z",
          updated_at: "2026-05-06T10:00:05.000Z",
        },
        {
          id: "turn-image-history-two-turns-2",
          thread_id: "session-image-history-two-turns",
          prompt_text: "@配图 再生成一张青柠极简插画",
          status: "completed",
          started_at: "2026-05-06T10:00:10.000Z",
          completed_at: "2026-05-06T10:00:15.000Z",
          created_at: "2026-05-06T10:00:10.000Z",
          updated_at: "2026-05-06T10:00:15.000Z",
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-image-history-two-turns",
    );

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(messages[0]?.content).toBe("@配图 生成一张广州塔春天照片");
    expect(messages[2]?.content).toBe("@配图 再生成一张青柠极简插画");
    expect(messages[1]?.imageWorkbenchPreview?.taskId).toBe(
      "task-image-history-two-turns-1",
    );
    expect(messages[3]?.imageWorkbenchPreview?.taskId).toBe(
      "task-image-history-two-turns-2",
    );
  });

  it("后端只缺部分用户图片指令时也应按时间补回缺失轮次", () => {
    const detail: AgentSessionDetail = {
      id: "session-image-history-partial-user-gap",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "user",
          timestamp: 1710000000,
          content: [
            {
              type: "input_text",
              text: "@配图 生成一张广州塔春天照片",
            } as never,
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000005,
          content: [
            {
              type: "output_text",
              text: "好啊，生成：广州塔春天照片",
            } as never,
            {
              type: "tool_response",
              id: "tool-image-history-partial-user-gap-1",
              success: true,
              output: "图片任务已提交",
              metadata: {
                task_id: "task-image-history-partial-user-gap-1",
                task_type: "image_generate",
                status: "succeeded",
                received_count: 1,
              },
            } as never,
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000015,
          content: [
            {
              type: "output_text",
              text: "好啊，生成：青柠极简插画",
            } as never,
            {
              type: "tool_response",
              id: "tool-image-history-partial-user-gap-2",
              success: true,
              output: "图片任务已提交",
              metadata: {
                task_id: "task-image-history-partial-user-gap-2",
                task_type: "image_generate",
                status: "succeeded",
                received_count: 1,
              },
            } as never,
          ],
        },
      ],
      turns: [
        {
          id: "turn-image-history-partial-user-gap-1",
          thread_id: "session-image-history-partial-user-gap",
          prompt_text: "@配图 生成一张广州塔春天照片",
          status: "completed",
          started_at: "2024-03-09T16:00:00.000Z",
          completed_at: "2024-03-09T16:00:05.000Z",
          created_at: "2024-03-09T16:00:00.000Z",
          updated_at: "2024-03-09T16:00:05.000Z",
        },
        {
          id: "turn-image-history-partial-user-gap-2",
          thread_id: "session-image-history-partial-user-gap",
          prompt_text: "@配图 再生成一张青柠极简插画",
          status: "completed",
          started_at: "2024-03-09T16:00:10.000Z",
          completed_at: "2024-03-09T16:00:15.000Z",
          created_at: "2024-03-09T16:00:10.000Z",
          updated_at: "2024-03-09T16:00:15.000Z",
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-image-history-partial-user-gap",
    );

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(messages[0]?.content).toBe("@配图 生成一张广州塔春天照片");
    expect(messages[2]?.content).toBe("@配图 再生成一张青柠极简插画");
    expect(messages[1]?.imageWorkbenchPreview?.taskId).toBe(
      "task-image-history-partial-user-gap-1",
    );
    expect(messages[3]?.imageWorkbenchPreview?.taskId).toBe(
      "task-image-history-partial-user-gap-2",
    );
  });

  it("已完成旧会话压缩水合时应保留工具过程并让最终正文接在工具之后", () => {
    const detail: AgentSessionDetail = {
      id: "session-compact-history",
      created_at: 1,
      updated_at: 2,
      history_limit: 40,
      turns: [
        {
          id: "turn-compact-history",
          thread_id: "session-compact-history",
          prompt_text: "恢复旧会话",
          status: "completed",
          started_at: "2026-04-30T10:00:00.000Z",
          completed_at: "2026-04-30T10:00:05.000Z",
          created_at: "2026-04-30T10:00:00.000Z",
          updated_at: "2026-04-30T10:00:05.000Z",
        },
      ],
      items: [
        {
          id: "thinking-compact-history",
          thread_id: "session-compact-history",
          turn_id: "turn-compact-history",
          sequence: 1,
          type: "reasoning",
          text: "大量思考过程",
          status: "completed",
          started_at: "2026-04-30T10:00:00.500Z",
          completed_at: "2026-04-30T10:00:01.000Z",
          updated_at: "2026-04-30T10:00:01.000Z",
        } as never,
        {
          id: "item-compact-history",
          thread_id: "session-compact-history",
          turn_id: "turn-compact-history",
          sequence: 2,
          type: "tool_call",
          tool_name: "Bash",
          arguments: { command: "printf slow" },
          output: "x".repeat(12_000),
          status: "completed",
          started_at: "2026-04-30T10:00:01.000Z",
          completed_at: "2026-04-30T10:00:02.000Z",
          updated_at: "2026-04-30T10:00:02.000Z",
        } as never,
      ],
      messages: [
        {
          role: "assistant",
          timestamp: 1710000005,
          content: [
            { type: "thinking", thinking: "大量思考过程" } as never,
            {
              type: "tool_request",
              id: "call-heavy",
              tool_name: "Bash",
              arguments: { command: "printf slow" },
            } as never,
            {
              type: "tool_response",
              id: "call-heavy",
              output: "x".repeat(12_000),
              success: true,
            } as never,
            { type: "output_text", text: "最终回复正文" } as never,
          ],
        },
      ],
    };

    expect(shouldCompactCompletedSessionHistory(detail)).toBe(true);

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-compact-history",
      { compactCompletedHistory: true },
    );

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      role: "user",
      content: "恢复旧会话",
    });
    expect(messages[1]).toMatchObject({
      role: "assistant",
      content: "最终回复正文",
      thinkingContent: "大量思考过程",
    });
    expect(messages[1]?.toolCalls?.[0]).toMatchObject({
      id: "item-compact-history",
      name: "Bash",
      status: "completed",
    });
    expect(messages[1]?.contentParts?.map((part) => part.type)).toEqual([
      "thinking",
      "tool_use",
      "text",
    ]);
    expect(messages[1]?.contentParts?.at(-1)).toEqual({
      type: "text",
      text: "最终回复正文",
    });
  });

  it("仍在运行的会话即使请求压缩水合，也应保留工具过程", () => {
    const detail: AgentSessionDetail = {
      id: "session-running-history",
      created_at: 1,
      updated_at: 2,
      history_limit: 40,
      turns: [
        {
          id: "turn-running-history",
          thread_id: "session-running-history",
          prompt_text: "继续执行",
          status: "running",
          started_at: "2026-04-30T10:00:00.000Z",
          created_at: "2026-04-30T10:00:00.000Z",
          updated_at: "2026-04-30T10:00:01.000Z",
        },
      ],
      messages: [
        {
          role: "assistant",
          timestamp: 1710000005,
          content: [
            {
              type: "tool_request",
              id: "call-running",
              tool_name: "Bash",
              arguments: { command: "sleep 1" },
            } as never,
          ],
        },
      ],
    };

    expect(shouldCompactCompletedSessionHistory(detail)).toBe(false);

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-running-history",
      { compactCompletedHistory: true },
    );

    const assistantMessage = messages.find(
      (message) => message.role === "assistant",
    );
    expect(assistantMessage?.toolCalls?.[0]).toMatchObject({
      id: "call-running",
      status: "running",
    });
    expect(
      assistantMessage?.contentParts?.some((part) => part.type === "tool_use"),
    ).toBe(true);
  });

  it("水合历史时应在保留本地可见文本的同时收敛同一工具调用的终态", () => {
    const localMessages = [
      {
        id: "local-user-tool-state",
        role: "user" as const,
        content: "请继续整理结果",
        timestamp: new Date("2026-05-12T10:00:00.000Z"),
      },
      {
        id: "local-assistant-tool-state",
        role: "assistant" as const,
        content: "本地仍在整理结果",
        contentParts: [
          {
            type: "text" as const,
            text: "本地仍在整理结果",
          },
          {
            type: "tool_use" as const,
            toolCall: {
              id: "tool-state-1",
              name: "WebFetch",
              arguments: JSON.stringify({
                url: "https://example.com/article",
              }),
              status: "running" as const,
              startTime: new Date("2026-05-12T10:00:01.000Z"),
            },
          },
        ],
        toolCalls: [
          {
            id: "tool-state-1",
            name: "WebFetch",
            arguments: JSON.stringify({
              url: "https://example.com/article",
            }),
            status: "running" as const,
            startTime: new Date("2026-05-12T10:00:01.000Z"),
          },
        ],
        timestamp: new Date("2026-05-12T10:00:02.000Z"),
      },
    ];
    const hydratedMessages = [
      {
        id: "history-user-tool-state",
        role: "user" as const,
        content: "请继续整理结果",
        timestamp: new Date("2026-05-12T10:00:01.000Z"),
      },
      {
        id: "local-assistant-tool-state",
        role: "assistant" as const,
        content:
          "历史消息内容过大，首屏已省略完整内容；需要时可加载完整历史查看。",
        contentParts: [
          {
            type: "tool_use" as const,
            toolCall: {
              id: "tool-state-1",
              name: "WebFetch",
              arguments: JSON.stringify({
                url: "https://example.com/article",
              }),
              status: "completed" as const,
              startTime: new Date("2026-05-12T10:00:01.000Z"),
              endTime: new Date("2026-05-12T10:00:03.000Z"),
              result: {
                success: true,
                output: "已整理完毕",
              },
            },
          },
        ],
        toolCalls: [
          {
            id: "tool-state-1",
            name: "WebFetch",
            arguments: JSON.stringify({
              url: "https://example.com/article",
            }),
            status: "completed" as const,
            startTime: new Date("2026-05-12T10:00:01.000Z"),
            endTime: new Date("2026-05-12T10:00:03.000Z"),
            result: {
              success: true,
              output: "已整理完毕",
            },
          },
        ],
        timestamp: new Date("2026-05-12T10:00:03.000Z"),
      },
    ];

    const mergedMessages = mergeHydratedMessagesWithLocalState(
      localMessages,
      hydratedMessages,
    );

    expect(mergedMessages).toHaveLength(2);
    expect(mergedMessages[1]?.content).toBe("本地仍在整理结果");
    expect(mergedMessages[1]?.toolCalls?.[0]).toMatchObject({
      id: "tool-state-1",
      status: "completed",
    });
    expect(mergedMessages[1]?.contentParts?.[0]).toEqual({
      type: "text",
      text: "本地仍在整理结果",
    });
    expect(
      mergedMessages[1]?.contentParts?.find(
        (part) =>
          part.type === "tool_use" && part.toolCall.id === "tool-state-1",
      ),
    ).toMatchObject({
      type: "tool_use",
      toolCall: {
        id: "tool-state-1",
        status: "completed",
      },
    });
  });
});
