import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  findStreamingRendererCallByContent,
  mockStreamingRenderer,
  mockAgentThreadTimeline,
  render,
} from "./MessageList.testHarness";
import type { Message } from "./MessageList.testHarness";

describe("MessageList artifact filtering", () => {
  it("文件变更汇总不应吞掉 canonical file artifact 卡片", () => {
    const now = new Date();
    const turnId = "turn-file-change-dedup";
    const messages: Message[] = [
      {
        id: "msg-assistant-file-change-dedup",
        role: "assistant",
        content: "CODE_RUNTIME_DONE",
        timestamp: now,
        contentParts: [
          { type: "text", text: "CODE_RUNTIME_DONE" },
          {
            type: "file_changes_batch",
            aggregate: {
              files: [
                {
                  path: "src/greeting.ts",
                  kind: "update",
                  linesAdded: 1,
                  linesRemoved: 1,
                  diff: [],
                  truncated: false,
                  source: "backend",
                  status: "completed",
                },
              ],
              totalAdded: 1,
              totalRemoved: 1,
              fileCount: 1,
            },
          },
        ],
        artifacts: [
          {
            id: "artifact-greeting",
            type: "code",
            title: "greeting.ts",
            content:
              "export function greeting() { return 'Hello Lime Runtime'; }",
            status: "complete",
            meta: {
              filePath:
                "/Users/coso/Library/Application Support/lime/projects/demo/src/greeting.ts",
              filename: "greeting.ts",
            },
            position: { start: 0, end: 64 },
            createdAt: now.getTime(),
            updatedAt: now.getTime(),
          },
        ],
      },
    ];

    const container = render(messages, {
      currentTurnId: turnId,
      turns: [
        {
          id: turnId,
          thread_id: "thread-1",
          prompt_text: "修复 greeting.ts",
          status: "completed",
          started_at: "2026-06-02T10:01:00.000Z",
          completed_at: "2026-06-02T10:01:05.000Z",
          created_at: "2026-06-02T10:01:00.000Z",
          updated_at: "2026-06-02T10:01:05.000Z",
        },
      ],
      threadItems: [
        {
          id: "artifact-file-change-document",
          thread_id: "thread-1",
          turn_id: turnId,
          sequence: 3,
          type: "file_artifact",
          path: ".lime/qc/code-runtime-fixture/src/greeting.ts",
          source: "artifact_snapshot",
          content:
            "export function greeting() { return 'Hello Lime Runtime'; }",
          status: "completed",
          started_at: "2026-06-02T10:01:01.000Z",
          completed_at: "2026-06-02T10:01:02.000Z",
          updated_at: "2026-06-02T10:01:02.000Z",
        },
        {
          id: "artifact-file-change-absolute",
          thread_id: "thread-1",
          turn_id: turnId,
          sequence: 4,
          type: "file_artifact",
          path: "/Users/coso/Library/Application Support/lime/projects/code-runtime-fixture/src/greeting.ts",
          source: "tool_result",
          content: "点击在画布中打开完整内容。",
          status: "completed",
          started_at: "2026-06-02T10:01:03.000Z",
          completed_at: "2026-06-02T10:01:04.000Z",
          updated_at: "2026-06-02T10:01:04.000Z",
        },
      ],
    });

    expect(
      container.querySelector('[data-testid="message-artifact-card"]'),
    ).toBeNull();
    expect(
      container.querySelectorAll(
        '[data-testid="timeline-file-artifact-card"]',
      ),
    ).toHaveLength(0);
    expect(
      container.querySelector(
        '[data-testid="message-list-historical-timeline-preview:trailing"]',
      ),
    ).not.toBeNull();
    act(() =>
      container
        .querySelector<HTMLButtonElement>(
          '[data-testid="message-list-historical-timeline-preview:trailing"]',
        )
        ?.click(),
    );
    expect(
      container.querySelectorAll(
        '[data-testid="timeline-file-artifact-card"]',
      ),
    ).toHaveLength(2);
    expect(
      container
        .querySelector('[data-testid="streaming-renderer"]')
        ?.getAttribute("data-content-parts"),
    ).toBe("2");
  });

  it("内容发布主链产物卡片应优先显示预览/上传/发布语义标题", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-content-post-artifact",
        role: "assistant",
        content: "已整理渠道预览稿",
        timestamp: now,
        artifacts: [
          {
            id: "artifact-content-post-preview",
            type: "document",
            title: "20260408-preview.md",
            content: "# 春日咖啡活动",
            status: "complete",
            meta: {
              filePath: "content-posts/20260408-preview.md",
              filename: "20260408-preview.md",
              contentPostIntent: "preview",
              contentPostLabel: "渠道预览稿",
              contentPostPlatformLabel: "小红书",
            },
            position: { start: 0, end: 0 },
            createdAt: now.getTime(),
            updatedAt: now.getTime(),
          },
        ],
      },
    ];

    const container = render(messages);

    expect(container.textContent).toContain("渠道预览稿");
    expect(container.textContent).toContain(
      "content-posts/20260408-preview.md",
    );
    const titleNode = Array.from(container.querySelectorAll("div")).find(
      (node) => node.textContent === "渠道预览稿",
    );
    expect(titleNode?.textContent).toBe("渠道预览稿");
  });

  it("不应把 .lime/tasks 下的内部任务快照 JSON 渲染成用户可见产物卡片", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-hidden-task-json",
        role: "assistant",
        content: "图片任务进行中",
        timestamp: now,
        artifacts: [
          {
            id: "artifact-hidden-task-json",
            type: "code",
            title: "task-image-1.json",
            content: '{"status":"running"}',
            status: "complete",
            meta: {
              filePath: ".lime/tasks/image_generate/task-image-1.json",
              filename: "task-image-1.json",
            },
            position: { start: 0, end: 0 },
            createdAt: now.getTime(),
            updatedAt: now.getTime(),
          },
        ],
      },
    ];

    const container = render(messages);
    expect(container.textContent).toContain("图片任务进行中");
    expect(container.textContent).not.toContain("task-image-1.json");
  });

  it("不应把 .lime/artifacts 下的内部 artifact 文稿 JSON 渲染成用户可见产物卡片", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-hidden-conversation-artifact-json",
        role: "assistant",
        content: "内部文稿已同步。",
        timestamp: now,
        artifacts: [
          {
            id: "artifact-hidden-conversation-artifact-json",
            type: "document",
            title: "report.artifact.json",
            content: '{"schemaVersion":"artifact_document.v1"}',
            status: "complete",
            meta: {
              filePath: ".lime/artifacts/thread-1/report.artifact.json",
              filename: "report.artifact.json",
            },
            position: { start: 0, end: 0 },
            createdAt: now.getTime(),
            updatedAt: now.getTime(),
          },
        ],
      },
    ];

    const container = render(messages);
    expect(container.textContent).toContain("内部文稿已同步。");
    expect(container.textContent).not.toContain("report.artifact.json");
  });

  it("完成态应先渲染 compact 过程摘要，再渲染正文和产物", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-order",
        role: "assistant",
        content: "已生成发布文案",
        timestamp: now,
        artifacts: [
          {
            id: "artifact-order",
            type: "document",
            title: "publish.md",
            content: "# Publish",
            status: "complete",
            meta: {
              filePath: "articles/publish.md",
              filename: "publish.md",
            },
            position: { start: 0, end: 0 },
            createdAt: now.getTime(),
            updatedAt: now.getTime(),
          },
        ],
      },
    ];

    const container = render(messages, {
      turns: [
        {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "发布文章",
          status: "completed",
          started_at: "2026-03-15T09:00:00Z",
          completed_at: "2026-03-15T09:00:05Z",
          created_at: "2026-03-15T09:00:00Z",
          updated_at: "2026-03-15T09:00:05Z",
        },
      ],
      threadItems: [
        {
          id: "item-1",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 1,
          status: "completed",
          started_at: "2026-03-15T09:00:01Z",
          completed_at: "2026-03-15T09:00:02Z",
          updated_at: "2026-03-15T09:00:02Z",
          type: "plan",
          text: "1. 打开页面\n2. 发布文章",
        },
      ],
    });

    const streaming = container.querySelector(
      '[data-testid="streaming-renderer"]',
    );
    const processSummary = container.querySelector(
      '[data-testid="message-list-historical-timeline-preview:leading"]',
    );
    const artifactButton = Array.from(
      container.querySelectorAll("button"),
    ).find((node) => node.textContent?.includes("publish.md"));

    expect(processSummary).not.toBeNull();
    expect(streaming).not.toBeNull();
    expect(artifactButton).toBeDefined();
    const processSummaryNode = processSummary as Node;
    const streamingNode = streaming as Node;
    const artifactButtonNode = artifactButton as Node;
    expect(
      findStreamingRendererCallByContent("已生成发布文案")?.contentParts,
    ).toEqual([{ type: "text", text: "已生成发布文案" }]);
    expect(
      processSummaryNode.compareDocumentPosition(streamingNode) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      streamingNode.compareDocumentPosition(artifactButtonNode) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("助手消息上的 actionRequests 应继续留在正文链路，不再重复透传给 timeline", () => {
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-action",
        role: "assistant",
        content: "请先确认文章标题。",
        timestamp: now,
        actionRequests: [
          {
            requestId: "req-ask-title",
            actionType: "ask_user",
            prompt: "请先确认文章标题",
            questions: [{ question: "这篇文章的最终标题是什么？" }],
          },
        ],
      },
    ];

    render(messages, {
      turns: [
        {
          id: "turn-action",
          thread_id: "thread-1",
          prompt_text: "确认文章标题",
          status: "aborted",
          started_at: "2026-03-15T09:00:00Z",
          completed_at: "2026-03-15T09:00:05Z",
          created_at: "2026-03-15T09:00:00Z",
          updated_at: "2026-03-15T09:00:05Z",
        },
      ],
      threadItems: [
        {
          id: "item-action-1",
          thread_id: "thread-1",
          turn_id: "turn-action",
          sequence: 1,
          status: "completed",
          started_at: "2026-03-15T09:00:01Z",
          completed_at: "2026-03-15T09:00:02Z",
          updated_at: "2026-03-15T09:00:02Z",
          type: "tool_call",
          tool_name: "browser_navigate",
          arguments: { url: "https://mp.weixin.qq.com" },
        },
      ],
    });

    const timelineProps = mockAgentThreadTimeline.mock.calls.map(
      ([props]) =>
        props as {
          actionRequests?: Array<Record<string, unknown>>;
          placement?: string;
        },
    );

    expect(
      timelineProps.every((props) => props.actionRequests === undefined),
    ).toBe(true);
    expect(mockStreamingRenderer).toHaveBeenCalledWith(
      expect.objectContaining({
        renderProposedPlanBlocks: true,
      }),
    );
  });

  it("应向执行轨迹透传已保存站点内容打开回调", () => {
    const onOpenSavedSiteContent = vi.fn();
    const now = new Date();
    const messages: Message[] = [
      {
        id: "msg-assistant-site-timeline",
        role: "assistant",
        content: "站点结果已沉淀。",
        timestamp: now,
      },
    ];

    render(messages, {
      onOpenSavedSiteContent,
      turns: [
        {
          id: "turn-site-open",
          thread_id: "thread-1",
          prompt_text: "采集站点内容",
          status: "completed",
          started_at: "2026-03-25T09:00:00Z",
          completed_at: "2026-03-25T09:00:05Z",
          created_at: "2026-03-25T09:00:00Z",
          updated_at: "2026-03-25T09:00:05Z",
        },
      ],
      threadItems: [
        {
          id: "item-site-open-1",
          thread_id: "thread-1",
          turn_id: "turn-site-open",
          sequence: 1,
          status: "completed",
          started_at: "2026-03-25T09:00:01Z",
          completed_at: "2026-03-25T09:00:02Z",
          updated_at: "2026-03-25T09:00:02Z",
          type: "tool_call",
          tool_name: "lime_site_run",
          arguments: { adapter_name: "github/search" },
        },
      ],
    });

    expect(
      findStreamingRendererCallByContent("站点结果已沉淀。"),
    ).toMatchObject({
      onOpenSavedSiteContent,
    });
  });

  it("当前 turn 已映射到较早助手消息时，不应被最新助手消息抢占", () => {
    const messages: Message[] = [
      {
        id: "msg-user-earlier",
        role: "user",
        content: "先做第一轮分析",
        timestamp: new Date("2026-03-15T09:00:00Z"),
      },
      {
        id: "msg-assistant-earlier",
        role: "assistant",
        content: "先给出一段中间反馈。",
        timestamp: new Date("2026-03-15T09:00:05Z"),
        runtimeTurnId: "turn-latest",
      },
      {
        id: "msg-user-latest",
        role: "user",
        content: "继续下一轮",
        timestamp: new Date("2026-03-15T09:00:10Z"),
      },
      {
        id: "msg-assistant-latest",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-03-15T09:00:20Z"),
        runtimeStatus: {
          phase: "preparing",
          title: "排队中",
          detail: "等待上一轮完成后继续。",
          checkpoints: [],
        },
      },
    ];

    const container = render(messages, {
      currentTurnId: "turn-latest",
      turns: [
        {
          id: "turn-latest",
          thread_id: "thread-1",
          prompt_text: "继续执行",
          status: "running",
          started_at: "2026-03-15T09:00:00Z",
          completed_at: "2026-03-15T09:00:06Z",
          created_at: "2026-03-15T09:00:00Z",
          updated_at: "2026-03-15T09:00:06Z",
        },
      ],
      threadItems: [
        {
          id: "item-latest",
          thread_id: "thread-1",
          turn_id: "turn-latest",
          sequence: 1,
          status: "completed",
          started_at: "2026-03-15T09:00:01Z",
          completed_at: "2026-03-15T09:00:02Z",
          updated_at: "2026-03-15T09:00:02Z",
          type: "plan",
          text: "继续执行当前任务",
        },
      ],
    });

    const streamingNodes = Array.from(
      container.querySelectorAll('[data-testid="streaming-renderer"]'),
    );

    expect(streamingNodes).toHaveLength(1);
    expect(
      findStreamingRendererCallByContent("先给出一段中间反馈。")?.contentParts,
    ).toEqual([
      expect.objectContaining({
        type: "text",
        text: expect.stringContaining("<proposed_plan>"),
      }),
    ]);
    expect(
      container.querySelector('[data-testid="assistant-message-meta-footer"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="agent-thread-reliability-panel"]'),
    ).toBeNull();
  });

  it("应不再在消息区渲染 reliability panel，避免占用对话列表空间", () => {
    const messages: Message[] = [
      {
        id: "msg-assistant-earlier",
        role: "assistant",
        content: "较早的中间反馈。",
        timestamp: new Date("2026-03-15T09:00:05Z"),
      },
      {
        id: "msg-assistant-latest",
        role: "assistant",
        content: "最新回合的输出。",
        timestamp: new Date("2026-03-15T09:00:20Z"),
      },
    ];

    const container = render(messages, {
      currentTurnId: "turn-latest",
      turns: [
        {
          id: "turn-latest",
          thread_id: "thread-1",
          prompt_text: "继续执行发布",
          status: "running",
          started_at: "2026-03-15T09:00:00Z",
          created_at: "2026-03-15T09:00:00Z",
          updated_at: "2026-03-15T09:00:06Z",
        },
      ],
      threadItems: [
        {
          id: "item-latest",
          thread_id: "thread-1",
          turn_id: "turn-latest",
          sequence: 1,
          status: "completed",
          started_at: "2026-03-15T09:00:01Z",
          completed_at: "2026-03-15T09:00:02Z",
          updated_at: "2026-03-15T09:00:02Z",
          type: "plan",
          text: "继续执行当前任务",
        },
      ],
      pendingActions: [
        {
          requestId: "req-1",
          actionType: "ask_user",
          prompt: "请确认是否继续发布",
          status: "pending",
        },
      ],
    });

    expect(
      container.querySelector('[data-testid="agent-thread-reliability-panel"]'),
    ).toBeNull();
    expect(
      findStreamingRendererCallByContent("较早的中间反馈。")?.contentParts,
    ).toEqual([
      expect.objectContaining({
        type: "text",
        text: expect.stringContaining("<proposed_plan>"),
      }),
    ]);
  });

  it("继续回合的 compact 过程摘要应挂在第二次对话组", () => {
    const messages: Message[] = [
      {
        id: "msg-user-first",
        role: "user",
        content: "帮我做一份 PPT 大纲",
        timestamp: new Date("2026-05-11T00:20:46Z"),
      },
      {
        id: "msg-assistant-first",
        role: "assistant",
        content: "执行失败：402 Payment Required",
        timestamp: new Date("2026-05-11T00:20:47Z"),
        runtimeTurnId: "turn-first",
      },
      {
        id: "msg-user-continue",
        role: "user",
        content: "继续",
        timestamp: new Date("2026-05-11T00:26:18Z"),
      },
      {
        id: "msg-assistant-continue",
        role: "assistant",
        content: "好的",
        timestamp: new Date("2026-05-11T00:26:19Z"),
        runtimeTurnId: "turn-continue",
      },
    ];

    const container = render(messages, {
      currentTurnId: "turn-continue",
      turns: [
        {
          id: "turn-first",
          thread_id: "thread-1",
          prompt_text: "帮我做一份 PPT 大纲",
          status: "failed",
          started_at: "2026-05-11T00:20:46Z",
          completed_at: "2026-05-11T00:20:47Z",
          created_at: "2026-05-11T00:20:46Z",
          updated_at: "2026-05-11T00:20:47Z",
        },
        {
          id: "turn-continue",
          thread_id: "thread-1",
          prompt_text: "继续",
          status: "completed",
          started_at: "2026-05-11T00:26:18Z",
          completed_at: "2026-05-11T00:26:24Z",
          created_at: "2026-05-11T00:26:18Z",
          updated_at: "2026-05-11T00:26:24Z",
        },
      ],
      threadItems: [
        {
          id: "error-first",
          thread_id: "thread-1",
          turn_id: "turn-first",
          sequence: 1,
          status: "failed",
          started_at: "2026-05-11T00:20:47Z",
          updated_at: "2026-05-11T00:20:47Z",
          type: "error",
          message: "Agent provider execution failed: 402 Payment Required",
        },
        {
          id: "process-continue",
          thread_id: "thread-1",
          turn_id: "turn-continue",
          sequence: 1,
          status: "completed",
          started_at: "2026-05-11T00:26:19Z",
          completed_at: "2026-05-11T00:26:24Z",
          updated_at: "2026-05-11T00:26:24Z",
          type: "plan",
          text: "等待用户补充 PPT 信息",
        },
      ],
    });

    const firstAssistant = Array.from(
      container.querySelectorAll('[data-testid="streaming-renderer"]'),
    ).find((node) => node.textContent?.includes("402 Payment Required"));
    const continueAssistant = Array.from(
      container.querySelectorAll('[data-testid="streaming-renderer"]'),
    ).find((node) => node.textContent?.includes("好的"));

    expect(firstAssistant).toBeTruthy();
    expect(continueAssistant).toBeTruthy();
    expect(findStreamingRendererCallByContent("好的")?.contentParts).toEqual([
      { type: "text", text: "好的" },
    ]);
    expect(
      container.querySelector(
        '[data-runtime-turn-id="turn-continue"] [data-testid="message-list-historical-timeline-preview:leading"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-runtime-turn-id="turn-continue"] [data-timeline-items="plan:process-continue"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-runtime-turn-id="turn-first"] [data-timeline-items="plan:process-continue"]',
      ),
    ).toBeNull();
    expect(
      findStreamingRendererCallByContent("执行失败：402 Payment Required")
        ?.contentParts,
    ).toEqual([{ type: "text", text: "执行失败：402 Payment Required" }]);
  });
});
