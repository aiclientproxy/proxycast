import { act } from "react";
import { describe, expect, it, vi } from "vitest";

import { mockMarkdownRenderer } from "./StreamingRenderer.testMocks";
import {
  installStreamingRendererTestHarness,
  renderStreamingRendererHarness as renderHarness,
} from "./StreamingRenderer.testHarness";

installStreamingRendererTestHarness();

describe("StreamingRenderer thinking and status", () => {
  it("抑制过程流时，非交错模式不应重复渲染思考、工具和确认卡", () => {
    const { container } = renderHarness({
      content: "最终回答内容",
      thinkingContent: "这段思考应由 timeline 承载",
      toolCalls: [
        {
          id: "tool-suppressed-fallback",
          name: "functions.exec_command",
          arguments: JSON.stringify({ cmd: "rg -n duplicate src" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-28T12:10:00.000Z"),
          endTime: new Date("2026-03-28T12:10:01.000Z"),
        },
      ],
      actionRequests: [
        {
          requestId: "req-suppressed-fallback",
          actionType: "tool_confirmation",
          status: "pending",
          prompt: "请确认是否继续",
        },
      ],
      onPermissionResponse: vi.fn(),
      suppressProcessFlow: true,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inline-tool-process-step"]'),
    ).toBeNull();
    expect(container.querySelector("details")).toBeNull();
    expect(
      container.querySelector('[data-testid="decision-panel"]'),
    ).toBeNull();
    expect(container.textContent).toContain("最终回答内容");
  });

  it("抑制过程流时，交错模式只保留正文片段", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "这段思考应由 timeline 渲染",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-suppressed-interleaved",
            name: "functions.exec_command",
            arguments: JSON.stringify({ cmd: "sed -n '1,80p' src/app.tsx" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-03-28T12:12:00.000Z"),
            endTime: new Date("2026-03-28T12:12:01.000Z"),
          },
        },
        {
          type: "action_required",
          actionRequired: {
            requestId: "req-suppressed-interleaved",
            actionType: "tool_confirmation",
            status: "pending",
            prompt: "请确认是否继续",
          },
        },
        {
          type: "text",
          text: "这里只保留最终正文。",
        },
      ],
      onPermissionResponse: vi.fn(),
      suppressProcessFlow: true,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inline-tool-process-step"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="decision-panel"]'),
    ).toBeNull();
    expect(container.textContent).toContain("这里只保留最终正文。");
  });

  it("等待首个事件时不应再把 agent 运行状态插入正文顶部", () => {
    const { container } = renderHarness({
      content: "",
      isStreaming: true,
      runtimeStatus: {
        phase: "preparing",
        title: "正在准备处理",
        detail: "正在理解请求并准备当前阶段。",
        checkpoints: ["对话优先执行", "等待首个事件"],
      },
      showRuntimeStatusInline: true,
    });

    expect(
      container.querySelector('[data-testid="agent-runtime-status"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("正在准备处理");
  });

  it("高风险服务进入稳妥顺序处理时，正文顶部不应再出现运行态卡", () => {
    const { container } = renderHarness({
      content: "",
      isStreaming: true,
      runtimeStatus: {
        phase: "routing",
        title: "当前服务较忙，稍后开始处理",
        detail:
          "当前服务在同时处理过多请求时容易直接失败，系统已切换为更稳妥的顺序处理。",
        checkpoints: ["当前服务仅同时处理 1 条此类请求"],
        metadata: {
          concurrency_scope: "provider_global",
          concurrency_phase: "queued",
          retryable_overload: true,
        },
      },
      showRuntimeStatusInline: true,
    });

    expect(
      container.querySelector('[data-testid="agent-runtime-status"]'),
    ).toBeNull();
  });

  it("正文已经开始输出后，不应再在正文区域重复渲染运行态", () => {
    const { container } = renderHarness({
      content: "我来帮你先打开 GitHub 搜索页。",
      isStreaming: true,
      runtimeStatus: {
        phase: "routing",
        title: "正在搜索 GitHub",
        detail: "已经打开搜索页，准备补充筛选条件。",
        checkpoints: ["浏览器已就绪", "准备应用最近更新时间筛选"],
      },
      showRuntimeStatusInline: true,
    });

    expect(
      container.querySelector('[data-testid="agent-runtime-status"]'),
    ).toBeNull();
  });

  it("交错内容模式下也不应在正文区域渲染运行态卡", () => {
    const { container } = renderHarness({
      content: "",
      isStreaming: true,
      runtimeStatus: {
        phase: "context",
        title: "正在整理搜索结果",
        detail: "已拿到页面内容，正在提取最近一个月更新的仓库。",
        checkpoints: ["页面内容已获取"],
      },
      showRuntimeStatusInline: true,
      contentParts: [
        {
          type: "text",
          text: "我已经打开 GitHub 搜索页，接下来开始筛选结果。",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-runtime-status-inline",
            name: "browser_snapshot",
            arguments: JSON.stringify({ page: "github-search" }),
            status: "running",
            result: undefined,
            startTime: new Date("2026-03-30T12:00:00.000Z"),
          },
        },
      ],
    });

    expect(
      container.querySelector('[data-testid="agent-runtime-status"]'),
    ).toBeNull();
  });

  it("已取消的运行状态也不应在正文顶部额外渲染状态卡", () => {
    const { container } = renderHarness({
      content: "",
      isStreaming: true,
      runtimeStatus: {
        phase: "cancelled",
        title: "图片任务已取消",
        detail: "任务已停止，不会继续生成新的图片结果。",
        checkpoints: ["已保留当前任务记录", "可在图片画布重新生成"],
      },
      showRuntimeStatusInline: true,
    });

    expect(
      container.querySelector('[data-testid="agent-runtime-status"]'),
    ).toBeNull();
  });

  it("思考内容进入流式阶段后应展开，完成后没有后续内容时保持展开", () => {
    const { container, rerender } = renderHarness({
      content: "",
      thinkingContent: "第一步：分析问题",
      isStreaming: false,
    });

    const initialDetails = container.querySelector("details");
    expect(initialDetails).toBeTruthy();
    expect((initialDetails as HTMLDetailsElement).open).toBe(false);

    rerender({
      content: "",
      thinkingContent: "第一步：分析问题\n第二步：调用工具",
      isStreaming: true,
    });

    const streamingDetails = container.querySelector("details");
    expect(streamingDetails).toBeTruthy();
    expect((streamingDetails as HTMLDetailsElement).open).toBe(true);
    expect(container.textContent).toContain("思考中");
    expect(container.textContent).toContain("第二步：调用工具");

    rerender({
      content: "",
      thinkingContent: "第一步：分析问题\n第二步：调用工具",
      isStreaming: false,
    });

    const completedDetails = container.querySelector("details");
    expect(completedDetails).toBeTruthy();
    expect((completedDetails as HTMLDetailsElement).open).toBe(true);
    expect(container.textContent).toContain("已完成思考");
    expect(container.textContent).toContain("第二步：调用工具");
  });

  it("思考后出现正文且未超过屏占比阈值时应继续展开", async () => {
    const rectSpy = vi
      .spyOn(HTMLElement.prototype, "getBoundingClientRect")
      .mockImplementation(function getBoundingClientRectMock(
        this: HTMLElement,
      ) {
        if (
          this.querySelector?.(
            '[data-testid="markdown-renderer"], [data-testid="streaming-markdown-pending-tail"]',
          )
        ) {
          return {
            bottom: 120,
            height: 120,
            left: 0,
            right: 0,
            top: 0,
            width: 0,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          };
        }
        return {
          bottom: 0,
          height: 0,
          left: 0,
          right: 0,
          top: 0,
          width: 0,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        };
      });

    try {
      const { container } = renderHarness({
        content: "下面开始输出正文。",
        thinkingContent: "第一步：分析问题\n第二步：确认边界",
        isStreaming: true,
      });

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      const details = container.querySelector("details");
      expect(details).toBeTruthy();
      expect((details as HTMLDetailsElement).open).toBe(true);
      expect(container.textContent).toContain("第二步：确认边界");
    } finally {
      rectSpy.mockRestore();
    }
  });

  it("思考后出现正文且超过屏占比阈值时应自动折叠", async () => {
    const rectSpy = vi
      .spyOn(HTMLElement.prototype, "getBoundingClientRect")
      .mockImplementation(function getBoundingClientRectMock(
        this: HTMLElement,
      ) {
        if (
          this.querySelector?.(
            '[data-testid="markdown-renderer"], [data-testid="streaming-markdown-pending-tail"]',
          )
        ) {
          return {
            bottom: 420,
            height: 420,
            left: 0,
            right: 0,
            top: 0,
            width: 0,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          };
        }
        return {
          bottom: 0,
          height: 0,
          left: 0,
          right: 0,
          top: 0,
          width: 0,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        };
      });

    const previousInnerHeight = window.innerHeight;
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      value: 800,
    });

    try {
      const { container } = renderHarness({
        content: "下面开始输出正文。",
        thinkingContent: "第一步：分析问题\n第二步：确认边界",
        isStreaming: true,
      });

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      const details = container.querySelector("details");
      expect(details).toBeTruthy();
      expect((details as HTMLDetailsElement).open).toBe(false);
      expect(container.textContent).toContain("思考中");
      expect(container.textContent).not.toContain("第二步：确认边界");
    } finally {
      Object.defineProperty(window, "innerHeight", {
        configurable: true,
        value: previousInnerHeight,
      });
      rectSpy.mockRestore();
    }
  });

  it("思考块应使用统一状态标签，并在完成态保留首行摘要", () => {
    const { container, rerender } = renderHarness({
      content: "",
      thinkingContent: "先生成一版草稿\n- 再根据反馈快速迭代",
      isStreaming: false,
    });

    expect(container.textContent).toContain("已完成思考");
    expect(container.textContent).toContain("先生成一版草稿");
    expect(container.textContent).not.toContain("思考中");

    rerender({
      content: "",
      thinkingContent: "先生成一版草稿\n- 再根据反馈快速迭代",
      isStreaming: true,
    });

    expect(container.textContent).toContain("思考中");
    expect(container.textContent).toContain("先生成一版草稿");
  });

  it("包含工具的运行中过程应先流式展开思考，完成后逐条保留工具记录", () => {
    const { container, rerender } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "先确认过程组行高\n再和工具行对齐",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-process-group-running",
            name: "functions.exec_command",
            arguments: JSON.stringify({ cmd: "rg -n thinking src" }),
            status: "running",
            startTime: new Date("2026-03-29T08:40:00.000Z"),
          },
        },
      ],
      isStreaming: true,
    });

    const runningThinking = container.querySelector<HTMLDetailsElement>(
      '[data-testid="thinking-block"] details',
    );
    expect(runningThinking).not.toBeNull();
    expect(runningThinking?.open).toBe(true);
    expect(container.textContent).toContain("思考中");
    expect(container.textContent).not.toContain("已完成思考");
    expect(container.textContent).toContain("先确认过程组行高");

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="inline-tool-process-step"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("rg -n thinking src");

    rerender({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "先确认过程组行高\n再和工具行对齐",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-process-group-running",
            name: "functions.exec_command",
            arguments: JSON.stringify({ cmd: "rg -n thinking src" }),
            status: "completed",
            startTime: new Date("2026-03-29T08:40:00.000Z"),
            result: { success: true, output: "ok" },
            endTime: new Date("2026-03-29T08:40:01.000Z"),
          },
        },
      ],
      isStreaming: false,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(container.textContent).toContain("已完成思考");
    expect(container.textContent).toContain("rg -n thinking src");
    expect(
      container.querySelector('[data-testid="inline-tool-process-step"]'),
    ).not.toBeNull();
  });

  it("内容工作台工具记录应保持正文前后顺序并隐藏协议细节", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "text",
          text: "先把内容工作台任务放在正确位置。",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-content-video-1",
            name: "video_generate",
            arguments: JSON.stringify({ prompt: "产品演示短片" }),
            status: "completed",
            result: {
              success: true,
              output: JSON.stringify({
                artifact_path: ".lime/tasks/video_generate/demo.json",
              }),
            },
            startTime: new Date("2026-06-03T08:00:00.000Z"),
            endTime: new Date("2026-06-03T08:00:01.000Z"),
          },
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-content-audio-1",
            name: "lime_create_audio_generation_task",
            arguments: JSON.stringify({ prompt: "播客旁白" }),
            status: "completed",
            result: {
              success: true,
              output: JSON.stringify({
                artifact_path: ".lime/tasks/audio_generate/demo.json",
              }),
            },
            startTime: new Date("2026-06-03T08:00:02.000Z"),
            endTime: new Date("2026-06-03T08:00:03.000Z"),
          },
        },
        {
          type: "text",
          text: "内容任务已发起，继续整理最终说明。",
        },
      ],
    });

    const markdownNodes = Array.from(
      container.querySelectorAll<HTMLElement>(
        '[data-testid="markdown-renderer"]',
      ),
    );
    const introNode = markdownNodes.find((node) =>
      node.textContent?.includes("先把内容工作台任务放在正确位置。"),
    );
    const finalNode = markdownNodes.find((node) =>
      node.textContent?.includes("内容任务已发起，继续整理最终说明。"),
    );
    const toolSteps = container.querySelectorAll<HTMLElement>(
      '[data-testid="inline-tool-process-step"]',
    );
    const firstToolStep = toolSteps[0] ?? null;

    expect(introNode).not.toBeNull();
    expect(firstToolStep).not.toBeNull();
    expect(finalNode).not.toBeNull();
    expect(
      introNode!.compareDocumentPosition(firstToolStep!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      firstToolStep!.compareDocumentPosition(finalNode!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    expect(toolSteps).toHaveLength(2);

    const text = container.textContent || "";
    expect(text).not.toContain(".lime/tasks");
    expect(text).not.toContain("artifact_path");
    expect(text).not.toContain("video_generate");
  });

  it("流式思考块不应把被切碎的过程文本交给 Markdown 解析", () => {
    mockMarkdownRenderer.mockClear();

    const { container } = renderHarness({
      content: "",
      thinkingContent: [
        "目录",
        "",
        "也",
        "",
        "不存在。",
        "",
        "可能",
        "",
        "整个",
        "",
        ".lime",
        "",
        "目录",
        "",
        "都不",
        "",
        "存在。",
      ].join("\n"),
      isStreaming: true,
    });

    const details = container.querySelector("details");
    act(() => {
      if (details) {
        (details as HTMLDetailsElement).open = true;
      }
      details?.dispatchEvent(new Event("toggle", { bubbles: true }));
    });

    expect(mockMarkdownRenderer).not.toHaveBeenCalled();
    const pendingTail = container.querySelector(
      '[data-testid="streaming-markdown-pending-tail"]',
    );
    expect(pendingTail?.textContent).toContain("目录");
    expect(pendingTail?.textContent).toContain("不存在。");
  });

  it("思考与普通工具应按时序独立渲染，不再压缩为批次摘要", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "先确认过程组行高\n再和工具行对齐",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-thinking-inline-style",
            name: "functions.exec_command",
            arguments: JSON.stringify({ cmd: "rg -n thinking src" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-03-29T08:40:00.000Z"),
            endTime: new Date("2026-03-29T08:40:01.000Z"),
          },
        },
      ],
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(container.textContent).toContain("已完成思考");
    expect(container.textContent).toContain("rg -n thinking src");

    const thinkingBlock = container.querySelector(
      '[data-testid="thinking-block"]',
    );
    const toolStep = container.querySelector(
      '[data-testid="inline-tool-process-step"]',
    );

    expect(thinkingBlock).not.toBeNull();
    expect(toolStep).not.toBeNull();
    expect(
      thinkingBlock!.compareDocumentPosition(toolStep!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(thinkingBlock?.getAttribute("data-visual-style")).not.toBe(
      "grouped-inline",
    );
  });

  it("仅思考内容应保留为独立思考块，不再切到第二套过程组", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "**Inspecting folder for details**",
        },
      ],
      isStreaming: false,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    const thinkingBlock = container.querySelector(
      '[data-testid="thinking-block"]',
    );
    expect(thinkingBlock).not.toBeNull();
    expect(thinkingBlock?.getAttribute("data-visual-style")).not.toBe(
      "grouped-inline",
    );
    expect(container.textContent).toContain("已完成思考");
    expect(container.textContent).toContain(
      "**Inspecting folder for details**",
    );
  });
});
