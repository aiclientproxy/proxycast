import { act } from "react";
import { describe, expect, it } from "vitest";

import {
  installStreamingRendererTestHarness,
  renderStreamingRendererHarness as renderHarness,
} from "./StreamingRenderer.testHarness";

installStreamingRendererTestHarness();

describe("StreamingRenderer process groups", () => {
  it("流式纯思考 fallback 应直接展开输出", () => {
    const { container } = renderHarness({
      content: "",
      thinkingContent: "先理解用户意图，再同步计划状态。",
      isStreaming: true,
    });

    const thinkingBlock = container.querySelector<HTMLElement>(
      '[data-testid="thinking-block"]',
    );
    const details = thinkingBlock?.querySelector<HTMLDetailsElement>("details");

    expect(thinkingBlock).not.toBeNull();
    expect(details?.open).toBe(true);
    expect(container.textContent).toContain("思考中");
    expect(container.textContent).toContain("先理解用户意图");
  });

  it("流式交错纯思考应直接展开输出", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "先确认最新计划，再准备实施。",
        },
      ],
      isStreaming: true,
    });

    const thinkingBlock = container.querySelector<HTMLElement>(
      '[data-testid="thinking-block"]',
    );
    const details = thinkingBlock?.querySelector<HTMLDetailsElement>("details");

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(thinkingBlock).not.toBeNull();
    expect(details?.open).toBe(true);
    expect(container.textContent).toContain("思考中");
    expect(container.textContent).toContain("先确认最新计划");
  });

  it("非交错模式应逐条保留工具过程记录", () => {
    const { container } = renderHarness({
      content: "最终结论",
      thinkingContent: "先检查滚动触发逻辑\n再确认输出展开时机",
      toolCalls: [
        {
          id: "tool-process-group-fallback",
          name: "functions.exec_command",
          arguments: JSON.stringify({ cmd: "rg -n scrollKey src" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-25T10:02:00.000Z"),
          endTime: new Date("2026-03-25T10:02:01.000Z"),
        },
      ],
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(container.textContent).toContain("已完成思考");
    expect(container.textContent).toContain("先检查滚动触发逻辑");
    expect(
      container
        .querySelector('[data-testid="inline-tool-process-step"]')
        ?.getAttribute("data-grouped"),
    ).toBe("no");
    expect(
      container.querySelectorAll('[data-testid="inline-tool-process-step"]')
        .length,
    ).toBe(1);
    expect(container.textContent).toContain("最终结论");
  });

  it("命令过程组应显示已发送输入的脱敏摘要", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "tool_use",
          toolCall: {
            id: "command-terminal-interactions",
            name: "exec_command",
            arguments: JSON.stringify({ command: "read input" }),
            status: "running",
            metadata: {
              terminal_interactions: [
                {
                  process_id: "unified-exec-1000",
                  stdin: "sent 9 chars",
                },
                {
                  process_id: "unified-exec-1000",
                  stdin: "raw-input-must-not-render",
                },
              ],
            },
            startTime: new Date("2026-03-25T10:02:00.000Z"),
          },
        },
      ],
      isStreaming: true,
    });

    expect(container.textContent).toContain("已发送输入：sent 9 chars");
    expect(container.textContent).not.toContain("raw-input-must-not-render");
  });

  it("交错内容中的思考与工具应按连续执行流逐条渲染", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "先检查 auto-scroll 触发条件\n确认是否只跟踪最后一项",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-process-group-interleaved",
            name: "functions.exec_command",
            arguments: JSON.stringify({
              cmd: "sed -n '1,120p' src/messages.tsx",
            }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-03-25T10:03:00.000Z"),
            endTime: new Date("2026-03-25T10:03:01.000Z"),
          },
        },
        {
          type: "thinking",
          text: "根因已经定位\n准备收口实现",
        },
        {
          type: "text",
          text: "已经定位到滚动没有跟随增量输出。",
        },
      ],
      isStreaming: false,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(container.textContent).toContain("先检查 auto-scroll 触发条件");
    expect(
      container
        .querySelector('[data-testid="inline-tool-process-step"]')
        ?.getAttribute("data-grouped"),
    ).toBe("no");
    expect(container.textContent).toContain("已经定位到滚动没有跟随增量输出。");
  });

  it("交错工具应保留 thread item DOM owner 与稳定工具行标识", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "tool_use",
          metadata: {
            source: "agent_thread_item",
            threadItemId: "history-replay-visual-mcp-read-file",
            turnId: "turn-history-replay",
            sequence: 4,
          },
          toolCall: {
            id: "history-replay-visual-mcp-read-file",
            name: "mcp__filesystem__read_file",
            arguments: JSON.stringify({ path: "README.md" }),
            status: "running",
            metadata: {
              source: "agent_thread_item",
              threadItemId: "history-replay-visual-mcp-read-file",
              turnId: "turn-history-replay",
              sequence: 4,
            },
            startTime: new Date("2026-07-09T10:00:04.000Z"),
          },
        },
      ],
      isStreaming: false,
    });

    const owner = container.querySelector(
      '[data-thread-item-id="history-replay-visual-mcp-read-file"]',
    );

    expect(owner).not.toBeNull();
    expect(
      owner?.querySelector('[data-testid="inline-tool-process-step"]'),
    ).not.toBeNull();
    expect(
      owner?.querySelector('[data-testid="tool-call-row"]'),
    ).not.toBeNull();
  });

  it("交错 agent message 应保留 canonical identity 与 phase DOM owner", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "text",
          text: "我会先检查项目。",
          metadata: {
            source: "agent_thread_item",
            threadItemId: "agent-message-commentary-1",
            turnId: "turn-agent-message-owner",
            sequence: 1,
            phase: "commentary",
          },
        },
        {
          type: "text",
          text: "检查完成。",
          metadata: {
            source: "agent_thread_item",
            threadItemId: "agent-message-final-1",
            turnId: "turn-agent-message-owner",
            sequence: 2,
            phase: "final_answer",
          },
        },
      ],
      isStreaming: false,
    });

    const parts = container.querySelectorAll(
      '[data-testid="agent-message-text-part"]',
    );
    expect(parts).toHaveLength(2);
    expect(parts[0]?.getAttribute("data-thread-item-id")).toBe(
      "agent-message-commentary-1",
    );
    expect(parts[0]?.getAttribute("data-message-phase")).toBe("commentary");
    expect(parts[1]?.getAttribute("data-thread-item-id")).toBe(
      "agent-message-final-1",
    );
    expect(parts[1]?.getAttribute("data-message-phase")).toBe("final_answer");
  });

  it("交错内容里相邻多次命令应逐条保留过程记录", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "先盘点目录",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-interleaved-1",
            name: "Bash",
            arguments: JSON.stringify({ command: "ls /tmp" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-05-29T10:00:00.000Z"),
            endTime: new Date("2026-05-29T10:00:01.000Z"),
          },
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-interleaved-2",
            name: "Bash",
            arguments: JSON.stringify({ command: "stat /tmp" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-05-29T10:00:02.000Z"),
            endTime: new Date("2026-05-29T10:00:03.000Z"),
          },
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-interleaved-3",
            name: "Bash",
            arguments: JSON.stringify({ command: "du -sh /tmp" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-05-29T10:00:04.000Z"),
            endTime: new Date("2026-05-29T10:00:05.000Z"),
          },
        },
        {
          type: "text",
          text: "目录已盘点。",
        },
      ],
      isStreaming: false,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(container.textContent).toContain("先盘点目录");

    const toolSteps = container.querySelectorAll(
      '[data-testid="inline-tool-process-step"]',
    );
    expect(toolSteps.length).toBe(3);
    expect(toolSteps[0]?.getAttribute("data-grouped")).toBe("no");
    expect(container.textContent).toContain("ls /tmp");
    expect(container.textContent).toContain("stat /tmp");
    expect(container.textContent).toContain("du -sh /tmp");
    expect(container.textContent).toContain("目录已盘点。");
  });

  it("追加后续工具时不应替换已显示的工具过程记录", () => {
    const baseProps = {
      content: "",
      isStreaming: true,
      contentParts: [
        {
          type: "tool_use" as const,
          toolCall: {
            id: "tool-preserve-read",
            name: "Read",
            arguments: JSON.stringify({ file_path: "ThinkingBlock.tsx" }),
            status: "completed" as const,
            result: { success: true, output: "ok" },
            startTime: new Date("2026-06-25T09:00:00.000Z"),
            endTime: new Date("2026-06-25T09:00:01.000Z"),
          },
        },
      ],
    };
    const { container, rerender } = renderHarness(baseProps);

    expect(container.textContent).toContain("ThinkingBlock.tsx");
    expect(
      container.querySelectorAll('[data-testid="inline-tool-process-step"]')
        .length,
    ).toBe(1);

    rerender({
      ...baseProps,
      contentParts: [
        ...baseProps.contentParts,
        {
          type: "tool_use",
          toolCall: {
            id: "tool-preserve-rg",
            name: "Bash",
            arguments: JSON.stringify({ command: "rg -n groupMarker src" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-06-25T09:00:02.000Z"),
            endTime: new Date("2026-06-25T09:00:03.000Z"),
          },
        },
      ],
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(
      container.querySelectorAll('[data-testid="inline-tool-process-step"]')
        .length,
    ).toBe(2);
    expect(container.textContent).toContain("ThinkingBlock.tsx");
    expect(container.textContent).toContain("rg -n groupMarker src");
  });

  it("消息仍在输出时，已失败的工具批次也应默认折叠，避免工具输出切开正文", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "text",
          text: "## 调研简报\n\n摘要：已整理出当前可用的主要来源。",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-failed-after-answer-1",
            name: "WebSearch",
            arguments: JSON.stringify({ query: "current sources" }),
            status: "failed",
            result: {
              success: false,
              output: "Execution failed: HTTP 401 Unknown Error",
            },
            startTime: new Date("2026-06-02T09:00:00.000Z"),
            endTime: new Date("2026-06-02T09:00:01.000Z"),
          },
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-failed-after-answer-2",
            name: "WebFetch",
            arguments: JSON.stringify({ url: "https://example.com/source" }),
            status: "failed",
            result: {
              success: false,
              output: "Fetching data issues",
            },
            startTime: new Date("2026-06-02T09:00:02.000Z"),
            endTime: new Date("2026-06-02T09:00:03.000Z"),
          },
        },
      ],
      isStreaming: true,
    });

    const processGroup = container.querySelector<HTMLButtonElement>(
      '[data-testid="streaming-process-group"] button',
    );

    expect(processGroup?.getAttribute("aria-expanded")).toBe("false");
    expect(container.textContent).toContain("调研简报");
    expect(container.textContent).toContain("已搜索网页 1 次，读取网页 1 次");
    expect(container.textContent).not.toContain("current sources");
    expect(container.textContent).not.toContain("https://example.com/source");
    expect(container.textContent).not.toContain("Execution failed");
    expect(container.textContent).not.toContain("Fetching data issues");

    act(() => {
      processGroup?.click();
    });

    expect(container.textContent).toContain("current sources");
    expect(container.textContent).toContain("example.com/source");
    expect(container.textContent).not.toContain("https://example.com/source");
    expect(container.textContent).not.toContain("Execution failed");
    expect(container.textContent).not.toContain("Fetching data issues");
  });

  it("消息仍在输出时，普通运行工具逐条显示但隐藏实时原始输出", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "text",
          text: "我先核实今天的国际新闻，再整理成简报。",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-running-command-1",
            name: "Bash",
            arguments: JSON.stringify({ command: "python scrape.py" }),
            status: "running",
            result: {
              success: true,
              output: "raw html payload should stay hidden while grouped",
            },
            startTime: new Date("2026-06-02T09:00:00.000Z"),
          },
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-running-command-2",
            name: "Bash",
            arguments: JSON.stringify({ command: "python normalize.py" }),
            status: "completed",
            result: {
              success: true,
              output: "another raw payload should stay hidden while grouped",
            },
            startTime: new Date("2026-06-02T09:00:02.000Z"),
            endTime: new Date("2026-06-02T09:00:03.000Z"),
          },
        },
        {
          type: "text",
          text: "## 国际新闻简报\n\n- 正在整理已确认来源。",
        },
      ],
      isStreaming: true,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(container.textContent).toContain("我先核实今天的国际新闻");
    expect(
      container.querySelectorAll('[data-testid="inline-tool-process-step"]')
        .length,
    ).toBe(2);
    expect(container.textContent).toContain("python scrape.py");
    expect(container.textContent).toContain("python normalize.py");
    expect(container.textContent).toContain("国际新闻简报");
    expect(container.textContent).not.toContain("raw html payload");
    expect(container.textContent).not.toContain("another raw payload");
  });

  it("尾部 Skill 仍在运行时应显示工具过程且隐藏协议 payload", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "thinking",
          text: "先读取技能说明，再执行 Skill。",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-running-skill-tail",
            name: "skill",
            arguments: JSON.stringify({
              skill: "brand-product-knowledge-builder",
              input: "整理产品知识库",
            }),
            status: "running",
            progress: { message: "正在运行 Skill" },
            result: {
              success: true,
              output: "partial skill protocol payload should stay hidden",
            },
            metadata: {
              tool_family: "skill",
              skill_name: "brand-product-knowledge-builder",
            },
            startTime: new Date("2026-06-02T09:00:00.000Z"),
          },
        },
      ],
      isStreaming: true,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(container.textContent).toContain("先读取技能说明");
    expect(container.textContent).toContain("brand-product-knowledge-builder");
    expect(container.textContent).toContain("正在运行 Skill");
    expect(container.textContent).not.toContain("partial skill protocol");
  });

  it("消息仍在输出时，completed Skill 过程应显示工具行但隐藏协议 payload", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "tool_use",
          toolCall: {
            id: "tool-completed-skill-before-final",
            name: "skill",
            arguments: JSON.stringify({
              skill: "capability-report",
              input: "生成能力报告",
            }),
            status: "completed",
            result: {
              success: true,
              output: JSON.stringify({
                runtime_enable_source: "legacy_tool_event",
                internal_payload: "skill protocol payload should stay hidden",
              }),
            },
            metadata: {
              tool_family: "skill",
              skill_name: "capability-report",
            },
            startTime: new Date("2026-06-02T09:00:00.000Z"),
            endTime: new Date("2026-06-02T09:00:01.000Z"),
          },
        },
        {
          type: "text",
          text: "我正在整理最终能力报告。",
        },
      ],
      isStreaming: true,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(container.textContent).toContain("capability-report");
    expect(container.textContent).toContain("我正在整理最终能力报告");
    expect(container.textContent).not.toContain("legacy_tool_event");
    expect(container.textContent).not.toContain("internal_payload");
    expect(container.textContent).not.toContain("skill protocol payload");
  });

  it("交错工具之间的过程状态自述不应作为正文块显示", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "tool_use",
          toolCall: {
            id: "tool-status-narration-before",
            name: "WebSearch",
            arguments: JSON.stringify({ query: "current sources" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-06-02T09:00:00.000Z"),
            endTime: new Date("2026-06-02T09:00:01.000Z"),
          },
        },
        {
          type: "text",
          text: "第一轮搜索结果质量不高，我继续从更可靠的页面聚合要点，避免把无关结果混进去。",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-status-narration-after",
            name: "WebFetch",
            arguments: JSON.stringify({ url: "https://example.com/source" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-06-02T09:00:02.000Z"),
            endTime: new Date("2026-06-02T09:00:03.000Z"),
          },
        },
        {
          type: "text",
          text: "## 调研简报\n\n- 已确认主要来源。",
        },
      ],
      isStreaming: false,
    });

    expect(container.textContent).not.toContain("第一轮搜索结果质量不高");
    expect(container.textContent).toContain("调研简报");
    expect(container.textContent).toContain("已确认主要来源");
  });

  it("交错内容里的工具记录应保持原始时序且不跨正文合并", () => {
    const { container } = renderHarness({
      content: "",
      contentParts: [
        {
          type: "text",
          text: "先说明检查目标。",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-before-text-1",
            name: "Bash",
            arguments: JSON.stringify({ command: "ls /tmp" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-05-29T10:00:00.000Z"),
            endTime: new Date("2026-05-29T10:00:01.000Z"),
          },
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-before-text-2",
            name: "Bash",
            arguments: JSON.stringify({ command: "pwd" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-05-29T10:00:02.000Z"),
            endTime: new Date("2026-05-29T10:00:03.000Z"),
          },
        },
        {
          type: "text",
          text: "中间结论已经确认。",
        },
        {
          type: "tool_use",
          toolCall: {
            id: "tool-after-text-1",
            name: "Bash",
            arguments: JSON.stringify({ command: "git status --short" }),
            status: "completed",
            result: { success: true, output: "ok" },
            startTime: new Date("2026-05-29T10:00:04.000Z"),
            endTime: new Date("2026-05-29T10:00:05.000Z"),
          },
        },
        {
          type: "text",
          text: "最终结论。",
        },
      ],
      isStreaming: false,
    });

    expect(
      container.querySelector('[data-testid="streaming-process-group"]'),
    ).toBeNull();
    expect(
      container.querySelectorAll('[data-testid="inline-tool-process-step"]')
        .length,
    ).toBe(3);

    const renderedText = container.textContent || "";
    const firstTextIndex = renderedText.indexOf("先说明检查目标。");
    const firstToolIndex = renderedText.indexOf("ls /tmp");
    const secondToolIndex = renderedText.indexOf("pwd");
    const middleTextIndex = renderedText.indexOf("中间结论已经确认。");
    const thirdToolIndex = renderedText.indexOf("git status --short");
    const finalTextIndex = renderedText.indexOf("最终结论。");

    expect(firstTextIndex).toBeGreaterThanOrEqual(0);
    expect(firstToolIndex).toBeGreaterThan(firstTextIndex);
    expect(secondToolIndex).toBeGreaterThan(firstToolIndex);
    expect(middleTextIndex).toBeGreaterThan(secondToolIndex);
    expect(thirdToolIndex).toBeGreaterThan(middleTextIndex);
    expect(finalTextIndex).toBeGreaterThan(thirdToolIndex);
  });
});
