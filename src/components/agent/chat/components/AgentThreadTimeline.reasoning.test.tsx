import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import type { AgentThreadItem } from "../types";
import {
  at,
  createBaseItem,
  createStructuredA2UIParseResult,
  parseAIResponseMock,
  renderTimeline,
} from "./AgentThreadTimeline.testFixtures";

describe("AgentThreadTimeline", () => {
  it("思考摘要中的 A2UI 代码块应切换为结构化预览", () => {
    parseAIResponseMock.mockReturnValue(createStructuredA2UIParseResult());

    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("summary-1", 1),
        status: "in_progress",
        completed_at: undefined,
        updated_at: at(1),
        type: "turn_summary",
        text: "```a2ui\n{}\n```",
      },
    ];

    const container = renderTimeline(items, {
      isCurrentTurn: true,
      turn: {
        status: "running",
      },
    });

    expect(
      container.querySelector('[data-testid="timeline-a2ui-card"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("处理中");
    expect(container.textContent).toContain("请先确认以下选项：");
    expect(container.textContent).not.toContain("```a2ui");
  });
  it("runtime status turn_summary 完成态应降级为中性进展提示", () => {
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("summary-1", 1),
        type: "turn_summary",
        text: "runtime status should not be shown as conversation prose",
        metadata: {
          sourceType: "runtime_status",
          surface: "runtime_status",
          visibility: "diagnostics",
        },
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    expect(container.textContent).toContain("当前进展");
    expect(container.textContent).not.toContain("runtime status should not");
    expect(container.textContent).not.toContain("已完成思考");
  });
  it("reasoning 中的 A2UI 代码块不应被跳过", () => {
    parseAIResponseMock.mockReturnValue(createStructuredA2UIParseResult());

    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-1", 1),
        status: "in_progress",
        completed_at: undefined,
        updated_at: at(1),
        type: "reasoning",
        text: "```a2ui\n{}\n```",
      },
    ];

    const container = renderTimeline(items, {
      isCurrentTurn: true,
      turn: {
        status: "running",
      },
    });

    expect(
      container.querySelector('[data-testid="timeline-a2ui-card"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("请先确认以下选项：");
    expect(container.textContent).not.toContain("```a2ui");
  });
  it("已完成 reasoning 中的 A2UI 代码块应直接显示结构化预览", () => {
    parseAIResponseMock.mockReturnValue(createStructuredA2UIParseResult());

    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-1", 1),
        type: "reasoning",
        text: "```a2ui\n{}\n```",
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    expect(
      container.querySelector('[data-testid="timeline-a2ui-card"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("请先确认以下选项：");
    expect(container.textContent).not.toContain("```a2ui");
  });
  it("纯 reasoning 阶段仅在时间线中出现一次", () => {
    const reasoningText = "先核对执行链路，再立即恢复当前运行。";
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-1", 1),
        type: "reasoning",
        text: reasoningText,
      },
    ];

    const container = renderTimeline(items, {
      isCurrentTurn: true,
      turn: {
        status: "running",
      },
    });

    expect(
      container.querySelector(
        '[data-testid="agent-thread-block:1:process:details"]',
      ),
    ).toBeNull();
    expect(container.textContent).not.toContain("思考摘要");
    expect((container.textContent?.split(reasoningText).length ?? 1) - 1).toBe(
      1,
    );
  });
  it("运行中的单条 reasoning 应直接显示为一个思考块，避免摘要壳嵌套详情壳", () => {
    const reasoningText = "正在梳理 PPT 大纲所需的关键输入。";
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-active-1", 1),
        type: "reasoning",
        status: "in_progress",
        completed_at: undefined,
        text: reasoningText,
      },
    ];

    const container = renderTimeline(items, {
      isCurrentTurn: true,
      turn: {
        status: "running",
      },
    });

    expect(
      container.querySelector(
        '[data-testid="agent-thread-block:1:process:details"]',
      ),
    ).toBeNull();
    expect((container.textContent?.split("思考中").length ?? 1) - 1).toBe(1);
    expect(container.textContent).toContain(reasoningText);
  });
  it("已完成的单条思考应默认只保留摘要，展开后再显示完整正文", () => {
    const reasoningText =
      "先核对执行链路，再立即恢复当前运行。\n随后补齐自动续提。";
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-1", 1),
        type: "reasoning",
        text: reasoningText,
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    const block = container.querySelector<HTMLDetailsElement>(
      '[data-testid="agent-thread-block:1:process"]',
    );
    const summary = block?.querySelector("summary");
    expect(block?.open).toBe(false);
    expect(container.textContent).toContain(
      "先核对执行链路，再立即恢复当前运行。",
    );
    expect(container.textContent).not.toContain("随后补齐自动续提。");

    act(() => {
      summary?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(
      container.querySelector(
        '[data-testid="agent-thread-block:1:process:details"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("随后补齐自动续提。");
  });
  it("reasoning 展开后应保留被切碎的来源行，不再压平成 prose", () => {
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-1", 1),
        type: "reasoning",
        text: [
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
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    const summary = container.querySelector("summary");
    act(() => {
      summary?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    const markdownBlocks = container.querySelectorAll(
      '[data-testid="markdown-renderer"]',
    );
    expect(markdownBlocks[0]?.textContent).toContain("目录\n\n也\n\n不存在。");
    expect(markdownBlocks[0]?.textContent).toContain(
      "可能\n\n整个\n\n.lime\n\n目录",
    );
  });
  it("reasoning 缺少正文时应回退显示 summary", () => {
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-1", 1),
        type: "reasoning",
        text: "",
        summary: ["先判断任务类型", "再决定是否联网"],
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    expect(container.textContent).toContain("先判断任务类型");
    expect(container.textContent).toContain("再决定是否联网");
  });
  it("reasoning 同时存在 summary 与 raw text 时只应显示 summary", () => {
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-1", 1),
        type: "reasoning",
        text: "这里是更完整的正文。",
        summary: ["先判断任务类型", "再决定是否联网"],
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    const summary = container.querySelector("summary");
    expect(container.textContent).toContain("先判断任务类型");
    expect(container.textContent).toContain("再决定是否联网");
    expect(container.textContent).not.toContain("这里是更完整的正文。");

    act(() => {
      summary?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(container.textContent).not.toContain("这里是更完整的正文。");
  });
  it("clawstream reasoning-first-visible hydrate 展开后也不应显示 raw reasoning", () => {
    const summaryText = "摘要：先确认用户只需要一个标记。";
    const rawReasoningText =
      "完整推理：用户只要求输出一个标记，因此不需要启动额外工具，也不应把 raw reasoning 拼进最终正文。";
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-clawstream-hydrate", 1),
        type: "reasoning",
        text: summaryText,
        summary: [summaryText],
        content: [rawReasoningText],
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    const block = container.querySelector<HTMLDetailsElement>(
      '[data-testid="agent-thread-block:1:process"]',
    );
    const summary = block?.querySelector("summary");

    expect(block?.open).toBe(false);
    expect(container.textContent).toContain(summaryText);
    expect(container.textContent).not.toContain(rawReasoningText);
    expect((container.textContent?.split(summaryText).length ?? 1) - 1).toBe(1);

    act(() => {
      summary?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(block?.open).toBe(true);
    expect(container.textContent).not.toContain(rawReasoningText);
    expect((container.textContent?.split(summaryText).length ?? 1) - 1).toBe(1);
    expect(
      (container.textContent?.split(rawReasoningText).length ?? 1) - 1,
    ).toBe(0);
  });
  it("canonical reasoning 只有 raw content 时展开后也不应泄漏", () => {
    const rawReasoningText = "完整思考：先理解问候，再直接给出简短回应。";
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-content-only", 1),
        type: "reasoning",
        text: rawReasoningText,
        summary: [],
        content: [rawReasoningText],
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    const block = container.querySelector<HTMLDetailsElement>(
      '[data-testid="agent-thread-block:1:process"]',
    );
    const summary = block?.querySelector("summary");
    expect(block?.open).toBe(false);
    expect(container.textContent).not.toContain(rawReasoningText);

    act(() => {
      summary?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(block?.open).toBe(true);
    expect(container.textContent).not.toContain(rawReasoningText);
  });
  it("reasoning 的 summary 与正文相同时不应重复渲染", () => {
    const repeatedText = "先判断任务类型\n\n再决定是否联网";
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("reasoning-1", 1),
        type: "reasoning",
        text: repeatedText,
        summary: ["先判断任务类型", "再决定是否联网"],
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    const summary = container.querySelector("summary");
    act(() => {
      summary?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(
      (container.textContent?.split("先判断任务类型").length ?? 1) - 1,
    ).toBe(1);
    expect(
      (container.textContent?.split("再决定是否联网").length ?? 1) - 1,
    ).toBe(1);
  });
  it("已完成的 request_user_input 应以只读 A2UI 卡片回显", () => {
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("input-1", 1),
        type: "request_user_input",
        request_id: "req-ask-1",
        action_type: "ask_user",
        prompt: "请选择执行模式",
        questions: [
          {
            question: "请选择执行模式",
            options: [{ label: "自动执行" }, { label: "确认后执行" }],
          },
        ],
        response: { answer: "自动执行" },
      },
    ];

    const container = renderTimeline(items, {
      turn: {
        status: "completed",
      },
    });

    expect(
      container.querySelector('[data-testid="timeline-a2ui-card"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="decision-panel"]'),
    ).toBeNull();
  });
  it("真实子任务 item 应支持查看子任务详情", () => {
    const onOpenSubagentSession = vi.fn();
    const items: AgentThreadItem[] = [
      {
        ...createBaseItem("subagent-1", 1),
        type: "subagent_activity",
        status: "completed",
        status_label: "interacted",
        title: "Image #1",
        summary: "封面图已生成",
        role: "image_editor",
        model: "gpt-image-1",
        session_id: "thread-child",
      },
    ];

    const container = renderTimeline(items, {
      onOpenSubagentSession,
    });

    expect(container.textContent).toContain("图片任务 1");
    expect(container.textContent).not.toContain("Image #1");
    expect(container.textContent).toContain("子任务：图片任务 1");

    const activityRow = container.querySelector(
      '[data-testid="subagent-activity-row"]',
    );
    expect(activityRow?.getAttribute("data-subagent-activity-item-id")).toBe(
      "subagent-1",
    );
    expect(activityRow?.getAttribute("data-subagent-activity-kind")).toBe(
      "interacted",
    );
    expect(activityRow?.getAttribute("data-subagent-thread-id")).toBe(
      "thread-child",
    );

    const button = Array.from(
      container.querySelectorAll<HTMLButtonElement>("button"),
    ).find((element) => element.textContent?.includes("查看子任务详情"));

    expect(button).toBeTruthy();

    act(() => {
      button?.click();
    });

    expect(onOpenSubagentSession).toHaveBeenCalledWith("thread-child");
  });
});
