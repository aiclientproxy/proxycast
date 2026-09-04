import { describe, expect, it } from "vitest";

import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";

import { hydrateSessionDetailMessages } from "./agentChatHistory";

describe("agentChatHistory compaction and previews", () => {
  it("imported Message 应保留来源 ID，不依赖 canonical history count", () => {
    const detail: AgentSessionDetail = {
      id: "session-page",
      created_at: 1,
      updated_at: 2,
      messages_count: 100,
      history_limit: 2,
      history_truncated: true,
      messages: [
        {
          id: "imported-user-1",
          role: "user",
          timestamp: 1710000000,
          content: [{ type: "text", text: "更早问题" }],
        },
        {
          id: "imported-assistant-1",
          role: "assistant",
          timestamp: 1710000005,
          content: [{ type: "text", text: "更早回答" }],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(detail, "session-page");

    expect(messages.map((message) => message.id)).toEqual([
      "imported-user-1",
      "imported-assistant-1",
    ]);
  });

  it("canonical cursor 不应参与 imported Message ID 合成", () => {
    const detail: AgentSessionDetail = {
      id: "session-cursor-page",
      created_at: 1,
      updated_at: 2,
      messages_count: 100,
      history_limit: 2,
      history_cursor: {
        item_cursor: "opaque-item-page-2",
        turn_cursor: null,
        loaded_entry_count: 2,
        loaded_turn_count: 0,
        loaded_item_count: 2,
        has_more: true,
      },
      history_truncated: true,
      messages: [
        {
          id: "imported-cursor-user",
          role: "user",
          timestamp: 1710000000,
          content: [{ type: "text", text: "Cursor 更早问题" }],
        },
        {
          id: "imported-cursor-assistant",
          role: "assistant",
          timestamp: 1710000005,
          content: [{ type: "text", text: "Cursor 更早回答" }],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-cursor-page",
    );

    expect(messages.map((message) => message.id)).toEqual([
      "imported-cursor-user",
      "imported-cursor-assistant",
    ]);
  });

  it("应兼容 reasoning 字段的历史恢复格式", () => {
    const detail: AgentSessionDetail = {
      id: "session-2",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000100,
          content: [
            { type: "reasoning", reasoning: "先列提纲" } as never,
            { type: "reasoning", reasoning: "，再展开正文" } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(detail, "session-2");

    expect(messages[0]?.contentParts).toEqual([
      {
        type: "thinking",
        text: "先列提纲，再展开正文",
      },
    ]);
    expect(messages[0]?.thinkingContent).toBe("先列提纲，再展开正文");
  });

  it("历史恢复时应去重 reasoning delta 与 final 的同文 thinking", () => {
    const thinkingText = "采用暖色调，聚焦阳光穿透树叶的光斑与远处高楼轮廓。";
    const detail: AgentSessionDetail = {
      id: "session-reasoning-duplicate",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000100,
          content: [
            { type: "reasoning", text: thinkingText } as never,
            { type: "reasoning", text: thinkingText } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-reasoning-duplicate",
    );

    expect(messages[0]?.contentParts).toEqual([
      {
        type: "thinking",
        text: thinkingText,
      },
    ]);
    expect(messages[0]?.thinkingContent).toBe(thinkingText);
  });

  it("应在历史恢复时清理 assistant 正文中的工具协议残留", () => {
    const detail: AgentSessionDetail = {
      id: "session-protocol-cleanup",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000150,
          content: [
            {
              type: "output_text",
              text: '<tool_result>{"output":"saved"}</tool_result>\n\n文章已保存为 Markdown。',
            } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-protocol-cleanup",
    );

    expect(messages[0]?.content).toBe("文章已保存为 Markdown。");
    expect(messages[0]?.contentParts).toEqual([
      {
        type: "text",
        text: "文章已保存为 Markdown。",
      },
    ]);
  });

  it("压缩水合时应从历史纯 JSON 图片工具输出恢复轻卡并清理提交摘要", () => {
    const imageTaskOutput = JSON.stringify({
      success: true,
      task_id: "task-history-json-image-1",
      task_type: "image_generate",
      task_family: "image",
      status: "pending_submit",
      normalized_status: "pending",
      path: ".lime/tasks/image_generate/task-history-json-image-1.json",
      absolute_path:
        "/workspace/.lime/tasks/image_generate/task-history-json-image-1.json",
      artifact_path:
        ".lime/tasks/image_generate/task-history-json-image-1.json",
      progress: {
        phase: "pending_submit",
        message: "任务已创建，等待进入队列",
      },
      record: {
        payload: {
          prompt: "青柠插画",
          count: 1,
          size: "1024x1024",
          session_id: "session-history-json-image",
        },
      },
    });
    const detail: AgentSessionDetail = {
      id: "session-history-json-image",
      created_at: 1,
      updated_at: 2,
      history_limit: 80,
      messages: [
        {
          role: "user",
          timestamp: 1710000300,
          content: [{ type: "text", text: "@配图 青柠插画" } as never],
        },
        {
          role: "assistant",
          timestamp: 1710000301,
          content: [
            {
              type: "tool_request",
              id: "tool-image-json-1",
              tool_name: "lime_create_image_generation_task",
              arguments: {
                prompt: "青柠插画",
                count: 1,
                size: "1024x1024",
              },
            } as never,
          ],
        },
        {
          role: "user",
          timestamp: 1710000302,
          content: [
            {
              type: "tool_response",
              id: "tool-image-json-1",
              output: imageTaskOutput,
              success: true,
            } as never,
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000303,
          content: [
            {
              type: "output_text",
              text: [
                "青柠插画配图任务已提交！",
                "任务类型：image_generate",
                "任务 ID：task-history-json-image-1",
                "任务文件：.lime/tasks/image_generate/task-history-json-image-1.json",
                "状态：pending_submit",
              ].join("\n"),
            } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-history-json-image",
      { compactCompletedHistory: true },
    );
    const assistant = messages.find((message) => message.role === "assistant");

    expect(messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(messages[0]).toMatchObject({
      role: "user",
      content: "@配图 青柠插画",
    });
    expect(assistant?.content).not.toContain("任务 ID");
    expect(assistant?.content).not.toContain("pending_submit");
    expect(assistant?.imageWorkbenchPreview).toMatchObject({
      taskId: "task-history-json-image-1",
      prompt: "青柠插画",
      status: "running",
      taskFilePath:
        "/workspace/.lime/tasks/image_generate/task-history-json-image-1.json",
      artifactPath: ".lime/tasks/image_generate/task-history-json-image-1.json",
    });
  });

  it("应从历史 assistant 消息恢复 token usage", () => {
    const detail: AgentSessionDetail = {
      id: "session-usage",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000300,
          content: [{ type: "output_text", text: "图片已经生成完成" } as never],
          usage: {
            input_tokens: 12000,
            output_tokens: 19000,
            cached_input_tokens: 4000,
            cache_creation_input_tokens: 1200,
          },
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(detail, "session-usage");

    expect(messages[0]?.usage).toEqual({
      input_tokens: 12000,
      output_tokens: 19000,
      cached_input_tokens: 4000,
      cache_creation_input_tokens: 1200,
    });
  });

  it("assistant 消息自身没有 usage 时应从同 turn read model 恢复", () => {
    const detail: AgentSessionDetail = {
      id: "session-turn-usage",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "user",
          timestamp: 1710000299,
          runtime_turn_id: "turn-image-usage",
          content: [{ type: "text", text: "@配图 生成深圳夏天照片" } as never],
        },
        {
          role: "assistant",
          timestamp: 1710000300,
          runtime_turn_id: "turn-image-usage",
          content: [
            {
              type: "output_text",
              text: "好的，这就为您生成深圳夏日午后的街头摄影画面。",
            } as never,
          ],
        },
      ],
      thread_read: {
        turns: [
          {
            turn_id: "turn-image-usage",
            status: "completed",
            usage: {
              input_tokens: 1175,
              output_tokens: 112,
            },
          } as never,
        ],
      } as never,
    };

    const messages = hydrateSessionDetailMessages(detail, "session-turn-usage");

    expect(messages[1]?.usage).toEqual({
      input_tokens: 1175,
      output_tokens: 112,
      cached_input_tokens: undefined,
      cache_creation_input_tokens: undefined,
    });
  });

  it("process-only thread item 不应为 token usage 合成 Message", () => {
    const detail: AgentSessionDetail = {
      id: "session-thread-item-image-usage",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-image-thread-item",
          status: "completed",
          usage: {
            input_tokens: 31000,
            output_tokens: 119,
          },
        } as never,
      ],
      items: [
        {
          id: "reasoning-1",
          type: "reasoning",
          turn_id: "turn-image-thread-item",
          thread_id: "thread-image",
          sequence: 1,
          status: "completed",
          text: "先确认深圳夏天午后的光线。",
          started_at: "2026-07-03T10:00:00.000Z",
          updated_at: "2026-07-03T10:00:00.000Z",
        } as never,
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-thread-item-image-usage",
    );

    expect(messages).toEqual([]);
  });

  it("合并相邻 assistant 历史消息时也应保留最后一条 usage", () => {
    const detail: AgentSessionDetail = {
      id: "session-adjacent-usage",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "user",
          timestamp: 1710000300,
          content: [{ type: "text", text: "帮我分析这个仓库" } as never],
        },
        {
          role: "assistant",
          timestamp: 1710000301,
          content: [
            { type: "output_text", text: "我先做一次轻量侦查。" } as never,
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000302,
          content: [
            {
              type: "output_text",
              text: "## 阶段结论\n\n已经找到关键线索。",
            } as never,
          ],
          usage: {
            input_tokens: 38483,
            output_tokens: 2406,
            cached_input_tokens: 36976,
            cache_creation_input_tokens: 0,
          },
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-adjacent-usage",
    );

    expect(messages).toHaveLength(2);
    expect(messages[1]?.role).toBe("assistant");
    expect(messages[1]?.content).toContain("我先做一次轻量侦查。");
    expect(messages[1]?.content).toContain("已经找到关键线索。");
    expect(messages[1]?.content).not.toContain("阶段结论");
    expect(messages[1]?.usage).toEqual({
      input_tokens: 38483,
      output_tokens: 2406,
      cached_input_tokens: 36976,
      cache_creation_input_tokens: 0,
    });
  });

  it("相邻 assistant 都带 thinking 时不应盲合并，避免跨轮思考串味", () => {
    const detail: AgentSessionDetail = {
      id: "session-adjacent-thinking",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "user",
          timestamp: 1710000300,
          content: [{ type: "text", text: "整理会议纪要" } as never],
        },
        {
          role: "assistant",
          timestamp: 1710000301,
          content: [
            { type: "thinking", text: "先整理会议纪要。" } as never,
            { type: "output_text", text: "会议纪要已整理。" } as never,
          ],
        },
        {
          role: "assistant",
          timestamp: 1710000302,
          content: [
            { type: "thinking", text: "只计算 2+2。" } as never,
            { type: "output_text", text: "2+2 等于 4。" } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-adjacent-thinking",
    );

    expect(messages).toHaveLength(3);
    expect(messages[1]?.thinkingContent).toBe("先整理会议纪要。");
    expect(messages[1]?.content).toBe("会议纪要已整理。");
    expect(messages[2]?.thinkingContent).toBe("只计算 2+2。");
    expect(messages[2]?.content).toBe("2+2 等于 4。");
  });

  it("应从历史 tool_response 恢复图片任务预览，并保留同一任务的连续 assistant 轨迹", () => {
    const detail: AgentSessionDetail = {
      id: "session-history-image-task-preview",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000400,
          content: [
            { type: "text", text: "正在生成广州塔夜景海报" } as never,
            {
              type: "tool_request",
              id: "tool-image-history-1",
              tool_name: "lime_create_image_generation_task",
              arguments: {
                prompt: "广州塔夜景海报",
                size: "1536x1024",
                count: 1,
              },
            } as never,
          ],
        },
        {
          role: "tool",
          timestamp: 1710000401,
          content: [
            {
              type: "tool_response",
              id: "tool-image-history-1",
              success: true,
              output: "图片任务已提交",
              metadata: {
                task_id: "task-image-history-1",
                task_type: "image_generate",
                status: "succeeded",
                project_id: "project-history-1",
                content_id: "content-history-1",
                requested_count: 1,
                received_count: 1,
              },
            } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-history-image-task-preview",
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({
      role: "assistant",
      imageWorkbenchPreview: {
        taskId: "task-image-history-1",
        prompt: "广州塔夜景海报",
        status: "complete",
        size: "1536x1024",
        imageCount: 1,
        projectId: "project-history-1",
        contentId: "content-history-1",
      },
    });
    expect(messages[0]?.contentParts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "text" }),
        expect.objectContaining({
          type: "tool_use",
          toolCall: expect.objectContaining({
            id: "tool-image-history-1",
            status: "completed",
          }),
        }),
      ]),
    );
  });

  it("应从历史 tool_response 恢复视频任务预览，并保留视频结果地址与时长信息", () => {
    const detail: AgentSessionDetail = {
      id: "session-history-video-task-preview",
      created_at: 1,
      updated_at: 2,
      messages: [
        {
          role: "assistant",
          timestamp: 1710000500,
          content: [
            { type: "text", text: "正在生成广州塔城市短片" } as never,
            {
              type: "tool_request",
              id: "tool-video-history-1",
              tool_name: "video_generate",
              arguments: {
                prompt: "广州塔城市短片",
                duration: 15,
                aspect_ratio: "16:9",
                resolution: "720p",
              },
            } as never,
          ],
        },
        {
          role: "tool",
          timestamp: 1710000501,
          content: [
            {
              type: "tool_response",
              id: "tool-video-history-1",
              success: true,
              output: "视频任务已完成",
              metadata: {
                task_id: "task-video-history-1",
                task_type: "video_generate",
                status: "succeeded",
                project_id: "project-video-history-1",
                content_id: "content-video-history-1",
                result: {
                  videos: [
                    {
                      url: "https://example.com/history-video.mp4",
                      duration_ms: 15000,
                    },
                  ],
                },
              },
            } as never,
          ],
        },
      ],
    };

    const messages = hydrateSessionDetailMessages(
      detail,
      "session-history-video-task-preview",
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({
      role: "assistant",
      taskPreview: {
        kind: "video_generate",
        taskId: "task-video-history-1",
        status: "complete",
        prompt: "广州塔城市短片",
        durationSeconds: 15,
        aspectRatio: "16:9",
        resolution: "720p",
        projectId: "project-video-history-1",
        contentId: "content-video-history-1",
        videoUrl: "https://example.com/history-video.mp4",
      },
    });
  });
});
