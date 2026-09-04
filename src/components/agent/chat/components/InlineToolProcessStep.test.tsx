import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import type { AgentToolCallState as ToolCallState } from "@/lib/api/agentProtocol";
import { changeLimeLocale } from "@/i18n/createI18n";
import { renderTool } from "./InlineToolProcessStep.testHarness";

describe("InlineToolProcessStep", () => {
  it("工具行应暴露 canonical typed Tool identity", () => {
    const { container } = renderTool({
      id: "tool-row-identity-1",
      name: "tool_search",
      status: "completed",
      startTime: new Date("2026-07-15T01:30:00.000Z"),
      endTime: new Date("2026-07-15T01:30:01.000Z"),
    });

    const row = container.querySelector('[data-testid="tool-call-row"]');
    expect(row?.getAttribute("data-tool-name")).toBe("tool_search");
    expect(row?.getAttribute("data-tool-status")).toBe("completed");
  });

  it("命令工具行只显示已发送输入的脱敏摘要", () => {
    const { container } = renderTool({
      id: "command-terminal-interaction",
      name: "exec_command",
      arguments: JSON.stringify({ command: "read input" }),
      status: "running",
      metadata: {
        terminal_interactions: [
          { process_id: "unified-exec-1000", stdin: "sent 9 chars" },
          {
            process_id: "unified-exec-1000",
            stdin: "raw-input-must-not-render",
          },
        ],
      },
      startTime: new Date("2026-07-15T01:30:00.000Z"),
    });

    expect(container.textContent).toContain("已发送输入：sent 9 chars");
    expect(container.textContent).not.toContain("raw-input-must-not-render");
  });

  it("内联工具过程节点应消费 Soul lifecycle descriptor metadata", () => {
    const { container } = renderTool({
      id: "tool-soul-lifecycle-inline-1",
      name: "WebSearch",
      arguments: JSON.stringify({ query: "Lime Soul lifecycle" }),
      status: "running",
      metadata: {
        soul_lifecycle: {
          surface: "tool_lifecycle",
          phase: "before_tool",
          status: "running",
          styleLevel: "L1",
          riskLevel: "normal",
          toneVariant: "cheeky_sassy",
          profileId: "cheeky_sassy_executor",
          packId: "com.lime.soul.cheeky-sassy-executor",
        },
        soul_surface: "tool_lifecycle",
        soul_phase: "before_tool",
        style_level: "L1",
        risk_level: "normal",
        tone_variant: "cheeky_sassy",
        profile_id: "cheeky_sassy_executor",
        pack_id: "com.lime.soul.cheeky-sassy-executor",
      },
      startTime: new Date("2026-07-06T12:00:00.000Z"),
    });

    const step = container.querySelector(
      '[data-testid="inline-tool-process-step"]',
    );
    expect(step?.getAttribute("data-soul-lifecycle")).toBe("yes");
    expect(step?.getAttribute("data-soul-surface")).toBe("tool_lifecycle");
    expect(step?.getAttribute("data-soul-phase")).toBe("before_tool");
    expect(step?.getAttribute("data-soul-style-level")).toBe("L1");
    expect(step?.getAttribute("data-soul-risk-level")).toBe("normal");
    expect(step?.getAttribute("data-soul-tone-variant")).toBe("cheeky_sassy");
    expect(step?.getAttribute("data-soul-profile-id")).toBe(
      "cheeky_sassy_executor",
    );
    expect(step?.getAttribute("data-soul-pack-id")).toBe(
      "com.lime.soul.cheeky-sassy-executor",
    );
  });

  it("运行中的读取工具应展示前置意图摘要", () => {
    const { container } = renderTool({
      id: "tool-read-running-1",
      name: "Read",
      arguments: JSON.stringify({ file_path: "src/app.tsx" }),
      status: "running",
      startTime: new Date("2026-04-13T09:58:00.000Z"),
    });

    expect(container.textContent).toContain("先查看 app.tsx");
    expect(container.textContent).not.toContain("执行完成");
  });

  it("运行中的工具收到输出增量时应优先展示实时输出摘要", () => {
    const { container } = renderTool({
      id: "tool-streaming-output-1",
      name: "mcp__runner__execute",
      arguments: JSON.stringify({ command: "npm test" }),
      status: "running",
      result: {
        success: true,
        output: "正在运行 12 个测试用例",
        metadata: {
          streaming: true,
        },
      },
      progress: {
        message: "正在处理第 2 项",
        progress: 2,
        total: 4,
      },
      startTime: new Date("2026-05-09T10:00:00.000Z"),
    });

    expect(container.textContent).toContain("实时输出：正在运行 12 个测试用例");
    expect(container.textContent).not.toContain("进度：正在处理第 2 项");
  });

  it("工具过程摘要应随当前语言切换，不硬编码中文", async () => {
    await changeLimeLocale("en-US");

    const { container } = renderTool({
      id: "tool-streaming-output-i18n-1",
      name: "mcp__runner__execute",
      arguments: JSON.stringify({ command: "npm test" }),
      status: "running",
      result: {
        success: true,
        output: "running 12 tests",
      },
      progress: {
        message: "processing item 2",
        progress: 2,
        total: 4,
      },
      metadata: {
        execution_origin: "preload",
        skill_title: "analysis",
      },
      startTime: new Date("2026-05-09T10:00:00.000Z"),
    });

    expect(container.textContent).toContain("Live output: running 12 tests");
    expect(container.textContent).toContain("System pre-run");
    expect(container.textContent).toContain("Skill: analysis");
    expect(container.textContent).not.toContain("实时输出");
    expect(container.textContent).not.toContain("系统预执行");
  });

  it("高频工具过程摘要应随当前语言切换", async () => {
    await changeLimeLocale("en-US");

    const toolSearch = renderTool({
      id: "tool-summary-i18n-search-tool",
      name: "ToolSearch",
      arguments: JSON.stringify({ query: "select:Read,Write" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          query: "select:Read,Write",
          count: 2,
          notes: [],
          tools: [{ name: "Read" }, { name: "Write" }],
        }),
      },
      startTime: new Date("2026-05-09T10:01:00.000Z"),
      endTime: new Date("2026-05-09T10:01:01.000Z"),
    });
    expect(toolSearch.container.textContent).toContain(
      "2 available tools confirmed",
    );
    expect(toolSearch.container.textContent).not.toContain("已确认可用工具");

    const webFetch = renderTool({
      id: "tool-summary-i18n-fetch-failed",
      name: "WebFetch",
      arguments: JSON.stringify({ url: "https://example.com/unavailable" }),
      status: "failed",
      result: {
        success: false,
        error: "404 Not Found",
        output: "",
      },
      startTime: new Date("2026-05-09T10:02:00.000Z"),
      endTime: new Date("2026-05-09T10:02:01.000Z"),
    });
    expect(webFetch.container.textContent).toContain(
      "Source temporarily unavailable",
    );
    expect(webFetch.container.textContent).not.toContain("来源暂时无法读取");

    const command = renderTool({
      id: "tool-summary-i18n-command",
      name: "Bash",
      arguments: JSON.stringify({ command: "git status --short" }),
      status: "running",
      startTime: new Date("2026-05-09T10:03:00.000Z"),
    });
    expect(command.container.textContent).toContain(
      "Checking workspace state first",
    );
    expect(command.container.textContent).not.toContain("先确认工作区状态");

    const browser = renderTool({
      id: "tool-summary-i18n-browser",
      name: "mcp__playwright__browser_navigate",
      arguments: JSON.stringify({ url: "https://example.com/page" }),
      status: "running",
      startTime: new Date("2026-05-09T10:04:00.000Z"),
    });
    expect(browser.container.textContent).toContain("Opening example.com");
    expect(browser.container.textContent).not.toContain("先打开");
  });

  it("任务、技能、MCP 与计划过程摘要应随当前语言切换", async () => {
    await changeLimeLocale("en-US");

    const skill = renderTool({
      id: "tool-summary-i18n-skill",
      name: "Skill",
      arguments: JSON.stringify({ name: "analysis" }),
      status: "running",
      startTime: new Date("2026-05-09T10:05:00.000Z"),
    });
    expect(skill.container.textContent).toContain(
      "Executing Skill analysis first",
    );
    expect(skill.container.textContent).not.toContain("先执行技能");

    const taskCreate = renderTool({
      id: "tool-summary-i18n-task-create",
      name: "TaskCreateTool",
      arguments: JSON.stringify({ title: "Daily trends" }),
      status: "completed",
      startTime: new Date("2026-05-09T10:06:00.000Z"),
      endTime: new Date("2026-05-09T10:06:01.000Z"),
    });
    expect(taskCreate.container.textContent).toContain("Started Daily trends");
    expect(taskCreate.container.textContent).not.toContain("已开始");

    const mcpResources = renderTool({
      id: "tool-summary-i18n-mcp-resources",
      name: "ListMcpResourcesTool",
      arguments: JSON.stringify({ server: "docs" }),
      status: "completed",
      startTime: new Date("2026-05-09T10:07:00.000Z"),
      endTime: new Date("2026-05-09T10:07:01.000Z"),
    });
    expect(mcpResources.container.textContent).toContain("Reviewed docs");
    expect(mcpResources.container.textContent).not.toContain("已查看");

    const mcpAuth = renderTool({
      id: "tool-summary-i18n-mcp-auth",
      name: "McpAuthTool",
      status: "completed",
      startTime: new Date("2026-05-09T10:08:00.000Z"),
      endTime: new Date("2026-05-09T10:08:01.000Z"),
    });
    expect(mcpAuth.container.textContent).toContain(
      "MCP authorization completed",
    );
    expect(mcpAuth.container.textContent).not.toContain("已完成 MCP 授权");

    const updatePlan = renderTool({
      id: "tool-summary-i18n-update-plan",
      name: "update_plan",
      status: "running",
      startTime: new Date("2026-05-09T10:09:00.000Z"),
    });
    expect(updatePlan.container.textContent).toContain("Updating plan first");
    expect(updatePlan.container.textContent).not.toContain("先更新计划");
  });

  it("ToolSearch 在流式阶段应保持结构化预览，不自动展开原始 JSON", () => {
    const { container } = renderTool(
      {
        id: "tool-search-streaming-1",
        name: "ToolSearch",
        arguments: JSON.stringify({ query: "select:Read,Write" }),
        status: "completed",
        result: {
          success: true,
          output: JSON.stringify({
            query: "select:Read,Write",
            count: 2,
            notes: [],
            tools: [{ name: "Read" }, { name: "Write" }],
          }),
        },
        startTime: new Date("2026-04-13T10:00:00.000Z"),
        endTime: new Date("2026-04-13T10:00:01.000Z"),
      },
      { isMessageStreaming: true },
    );

    expect(container.textContent).toContain("已确认可用工具 2 个");
    expect(container.textContent).toContain("查看文件");
    expect(container.textContent).toContain("保存文件");
    expect(container.textContent).not.toContain("查询：");
    expect(container.textContent).not.toContain("select:Read,Write");
    expect(
      container.querySelector(
        '[data-testid="inline-tool-process-tool-search-result"]',
      ),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-renderer"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain('"tools"');
  });

  it("MCP 工具过程应从 structuredContent 展示正文而不是协议包络", () => {
    const { container } = renderTool({
      id: "tool-mcp-structured-process-1",
      name: "mcp__docs__search_docs",
      arguments: JSON.stringify({ query: "structured content" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          request_metadata: {
            projection: "mcp_tool_result_projection",
            trace_id: "trace-structured-content",
          },
          diagnostics: {
            elapsed_ms: 12,
            raw_transport_payload: "doc-hidden-envelope",
          },
          content: [
            {
              type: "text",
              text: "control-plane envelope only; user answer is stored in structuredContent",
            },
          ],
        }),
        structuredContent: {
          answer: "MCP 结构化答案已进入 Agent Chat GUI",
          ids: ["doc-structured-1"],
        },
      },
      startTime: new Date("2026-06-21T13:10:00.000Z"),
      endTime: new Date("2026-06-21T13:10:01.000Z"),
    });

    expect(container.textContent).toContain(
      "MCP 结构化答案已进入 Agent Chat GUI",
    );
    expect(container.textContent).toContain("doc-structured-1");
    expect(container.textContent).not.toContain("control-plane envelope only");
    expect(container.textContent).not.toContain("request_metadata");
    expect(container.textContent).not.toContain("doc-hidden-envelope");
    expect(container.textContent).not.toContain("mcp_tool_result_projection");
  });

  it("MCP structuredContent 应在折叠摘要中优先于通用完成文案", () => {
    const { container } = renderTool({
      id: "tool-mcp-structured-summary-1",
      name: "mcp__docs__diagnostic_probe",
      arguments: JSON.stringify({ query: "structured content" }),
      status: "completed",
      result: {
        success: true,
        output: "已完成这一步",
        structuredContent: {
          answer: "MCP 结构化答案已进入 Agent Chat GUI",
          ids: ["doc-structured-1"],
        },
      },
      startTime: new Date("2026-06-21T13:10:00.000Z"),
      endTime: new Date("2026-06-21T13:10:01.000Z"),
    });

    expect(container.textContent).toContain(
      "MCP 结构化答案已进入 Agent Chat GUI",
    );
    expect(container.textContent).toContain("doc-structured-1");
  });

  it("超长工具结果在流式阶段应默认收起原始详情", () => {
    const { container } = renderTool(
      {
        id: "tool-read-large-streaming-1",
        name: "Read",
        arguments: JSON.stringify({ file_path: "src/main.ts" }),
        status: "completed",
        result: {
          success: true,
          output: "A".repeat(1600),
        },
        startTime: new Date("2026-04-13T10:05:00.000Z"),
        endTime: new Date("2026-04-13T10:05:01.000Z"),
      },
      { isMessageStreaming: true },
    );

    expect(container.textContent).toContain("已查看 main.ts");
    expect(
      container.querySelector('[data-testid="markdown-renderer"]'),
    ).toBeNull();
  });

  it("工具详情遇到伪标签输出时应先转义再渲染", () => {
    const { container } = renderTool({
      id: "tool-bash-ink-tags-1",
      name: "Bash",
      arguments: JSON.stringify({ command: "echo demo" }),
      status: "completed",
      result: {
        success: true,
        output: "<text>正在整理</text>\n<spinner />",
      },
      startTime: new Date("2026-04-13T10:07:00.000Z"),
      endTime: new Date("2026-04-13T10:07:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="展开过程详情"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain(
      "&lt;text&gt;正在整理&lt;/text&gt;",
    );
    expect(container.textContent).toContain("&lt;spinner /&gt;");
  });

  it("文件工具应暴露稳定打开入口并携带原始文件路径", () => {
    const onFileClick = vi.fn();
    const { container } = renderTool(
      {
        id: "tool-read-file-open-1",
        name: "read_file",
        arguments: JSON.stringify({
          path: "/tmp/imported-local-history/docs/imported-preview.docx",
        }),
        status: "completed",
        result: {
          success: true,
          output: "导入会话 DOCX 预览内容",
        },
        startTime: new Date("2026-06-17T10:08:00.000Z"),
        endTime: new Date("2026-06-17T10:08:01.000Z"),
      },
      { onFileClick },
    );

    const button = container.querySelector(
      '[data-testid="inline-tool-open-file"]',
    ) as HTMLButtonElement | null;

    expect(button).not.toBeNull();
    expect(button?.getAttribute("data-file-path")).toBe(
      "/tmp/imported-local-history/docs/imported-preview.docx",
    );

    act(() => {
      button?.click();
    });

    expect(onFileClick).toHaveBeenCalledWith(
      "/tmp/imported-local-history/docs/imported-preview.docx",
      "导入会话 DOCX 预览内容",
    );
  });

  it("导入文件工具参数为对象时也应暴露稳定打开入口", () => {
    const onFileClick = vi.fn();
    const filePath = "/tmp/imported-local-history/docs/imported-preview.md";
    const { container } = renderTool(
      {
        id: "tool-read-file-open-object-1",
        name: "read_file",
        arguments: JSON.stringify({ path: filePath }),
        status: "completed",
        result: {
          success: true,
          output: "导入会话 Markdown 预览内容",
        },
        startTime: new Date("2026-06-17T10:08:00.000Z"),
        endTime: new Date("2026-06-17T10:08:01.000Z"),
      } as ToolCallState,
      { onFileClick },
    );

    const button = container.querySelector(
      '[data-testid="inline-tool-open-file"]',
    ) as HTMLButtonElement | null;

    expect(button).not.toBeNull();
    expect(button?.getAttribute("data-file-path")).toBe(filePath);

    act(() => {
      button?.click();
    });

    expect(onFileClick).toHaveBeenCalledWith(
      filePath,
      "导入会话 Markdown 预览内容",
    );
  });

  it("canonical named argument 数组也应暴露文件打开入口", () => {
    const onFileClick = vi.fn();
    const filePath = "/tmp/imported-local-history/docs/imported-preview.md";
    const { container } = renderTool(
      {
        id: "tool-read-file-open-named-array-1",
        name: "read_file",
        arguments: JSON.stringify([{ name: "path", value: filePath }]),
        status: "completed",
        result: {
          success: true,
          output: "导入会话 Markdown 预览内容",
        },
        startTime: new Date("2026-06-17T10:08:00.000Z"),
        endTime: new Date("2026-06-17T10:08:01.000Z"),
      } as ToolCallState,
      { onFileClick },
    );

    const button = container.querySelector(
      '[data-testid="inline-tool-open-file"]',
    ) as HTMLButtonElement | null;
    expect(button?.getAttribute("data-file-path")).toBe(filePath);

    act(() => {
      button?.click();
    });

    expect(onFileClick).toHaveBeenCalledWith(
      filePath,
      "导入会话 Markdown 预览内容",
    );
  });

  it("工具详情遇到 TypeScript 尖括号语法时也应转义再渲染", () => {
    const { container } = renderTool({
      id: "tool-read-typescript-tags-1",
      name: "Read",
      arguments: JSON.stringify({ file_path: "src/schema.ts" }),
      status: "completed",
      result: {
        success: true,
        output:
          "type OutputSchema<T> = keyof T\ncontentBlockParam: ContentBlockParam<typeof schema>",
      },
      startTime: new Date("2026-04-13T10:08:00.000Z"),
      endTime: new Date("2026-04-13T10:08:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="展开过程详情"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("&lt;T&gt;");
    expect(container.textContent).toContain("&lt;typeof schema&gt;");
  });

  it("记忆工具应展示本轮使用的路径、引用和分页证据", () => {
    const { container } = renderTool({
      id: "tool-memory-read-1",
      name: "memory_read",
      arguments: JSON.stringify({ path: "MEMORY.md" }),
      status: "completed",
      result: {
        success: true,
        output: "Remember the launch tone.",
        metadata: {
          operation: "read",
          path: "MEMORY.md",
          rootScope: "workspace",
          citation: {
            path: "MEMORY.md",
            startLine: 3,
            endLine: 6,
          },
        },
      },
      startTime: new Date("2026-06-19T10:10:00.000Z"),
      endTime: new Date("2026-06-19T10:10:01.000Z"),
    });

    expect(container.textContent).toContain("已读取记忆 MEMORY.md");
    expect(container.textContent).toContain("记忆使用证据");
    expect(container.textContent).toContain("路径：MEMORY.md");
    expect(container.textContent).toContain("引用：MEMORY.md:3-6");
    expect(
      container.querySelector('[data-testid="inline-tool-memory-evidence"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="markdown-renderer"]'),
    ).toBeNull();
  });

  it("记忆搜索工具应在英文环境展示命中数和截断状态", async () => {
    await changeLimeLocale("en-US");

    const { container } = renderTool({
      id: "tool-memory-search-1",
      name: "memory_search",
      arguments: JSON.stringify({ queries: ["launch tone"] }),
      status: "completed",
      result: {
        success: true,
        output: "Found 2 memory hits. More hits are available via nextCursor.",
        metadata: {
          operation: "search",
          hits: [
            { path: "MEMORY.md", matchLineNumber: 4 },
            { path: "rollout_summaries/thread.md", matchLineNumber: 8 },
          ],
          truncated: true,
          nextCursor: "cursor-2",
        },
      },
      startTime: new Date("2026-06-19T10:11:00.000Z"),
      endTime: new Date("2026-06-19T10:11:01.000Z"),
    });

    expect(container.textContent).toContain("Memory searched launch tone");
    expect(container.textContent).toContain("Searched memory with 2 hit(s)");
    expect(container.textContent).toContain("2 hit(s)");
    expect(container.textContent).toContain("More results are available");
    expect(container.textContent).not.toContain("已搜索记忆");
  });

  it("Skill 过程步骤应能展开查看本次执行读取的 SKILL.md", () => {
    const { container } = renderTool({
      id: "skill:analysis-run-1",
      name: "Skill",
      arguments: JSON.stringify({
        skill: "analysis",
        display_name: "analysis",
        source: "SKILL.md",
      }),
      status: "completed",
      result: {
        success: true,
        output: "已从 SKILL.md 读取并执行 Skill：analysis",
        metadata: {
          tool_family: "skill",
          skill_name: "analysis",
          skill_display_name: "analysis",
          skill_source: "SKILL.md",
          agent_skills_standard: true,
          markdown_content_bytes: 86,
          skill_markdown_content:
            "---\nname: analysis\ndescription: 分析任务\n---\n\n# Analysis Skill\n\n必须先确认可见上下文。",
        },
      },
      startTime: new Date("2026-05-14T04:30:00.000Z"),
      endTime: new Date("2026-05-14T04:30:02.000Z"),
    });

    expect(container.textContent).toContain("已执行技能 analysis");
    expect(container.textContent).toContain("SKILL.md");
    expect(container.textContent).not.toContain("skill_markdown_content");

    act(() => {
      const skillButton = container.querySelector(
        'button[title="查看 SKILL.md"]',
      ) as HTMLButtonElement | null;
      skillButton?.click();
    });

    expect(
      container.querySelector(
        '[data-testid="inline-tool-skill-content-panel"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("执行时读取的 SKILL.md");
    expect(container.textContent).toContain("随本次执行记录保存");
    expect(container.textContent).toContain("Agent Skills 标准");
    expect(container.textContent).toContain("展开 SKILL.md 内容");
    expect(container.textContent).not.toContain("Analysis Skill");

    act(() => {
      const expandBodyButton = Array.from(
        container.querySelectorAll("button"),
      ).find((button) => button.textContent?.includes("展开 SKILL.md 内容")) as
        | HTMLButtonElement
        | undefined;
      expandBodyButton?.click();
    });

    expect(
      container.querySelector('[data-testid="inline-tool-skill-content-body"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("收起 SKILL.md 内容");
    expect(container.textContent).toContain("Analysis Skill");
    expect(container.textContent).toContain("必须先确认可见上下文。");
    expect(container.textContent).not.toContain("tool_family");
    expect(container.textContent).not.toContain("skill_markdown_content");
  });

  it("服务技能过程步骤不应暴露兼容文案或原始 JSON", () => {
    const { container } = renderTool({
      id: "tool-service-skill-1",
      name: "lime_run_service_skill",
      arguments: JSON.stringify({
        skill_title: "渠道预览",
        service_skill_id: "channel-preview",
        slot_values: {
          platform: "小红书",
        },
      }),
      status: "running",
      result: {
        success: true,
        output: JSON.stringify({
          service_skill_id: "channel-preview",
          slot_values: {
            platform: "小红书",
          },
          status: "running",
        }),
      },
      metadata: {
        skill_title: "渠道预览",
      },
      startTime: new Date("2026-06-21T10:10:00.000Z"),
    });

    expect(container.textContent).toContain("执行服务技能中 渠道预览");
    expect(container.textContent).toContain("先执行服务技能 渠道预览");
    expect(container.textContent).not.toContain("实时输出");
    expect(container.textContent).not.toContain("兼容");
    expect(container.textContent).not.toContain("service_skill_id");
    expect(container.textContent).not.toContain("slot_values");
    expect(container.textContent).not.toContain("channel-preview");
    expect(
      container.querySelector(
        '[data-testid="inline-tool-skill-content-panel"]',
      ),
    ).toBeNull();
    expect(container.querySelector('button[title="查看 SKILL.md"]')).toBeNull();
  });

  it("服务技能完成态嵌套结果也不应暴露结构化 JSON 包络", () => {
    const { container } = renderTool({
      id: "tool-service-skill-completed-1",
      name: "lime_run_service_skill",
      arguments: JSON.stringify({
        skill_title: "渠道预览",
        service_skill_id: "channel-preview",
      }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          result: {
            output: {
              data: {
                serviceSkillId: "channel-preview",
                slotValues: {
                  platform: "小红书",
                },
                status: "completed",
              },
            },
          },
        }),
      },
      metadata: {
        skill_title: "渠道预览",
      },
      startTime: new Date("2026-06-21T10:10:00.000Z"),
      endTime: new Date("2026-06-21T10:10:05.000Z"),
    });

    expect(container.textContent).toContain("已执行服务技能 渠道预览");
    expect(container.textContent).toContain("已完成服务技能执行 渠道预览");
    expect(container.textContent).not.toContain("serviceSkillId");
    expect(container.textContent).not.toContain("slotValues");
    expect(container.textContent).not.toContain("channel-preview");
    expect(container.textContent).not.toContain("实时输出");
    expect(container.textContent).not.toContain("兼容");
    expect(
      container.querySelector('[data-testid="markdown-renderer"]'),
    ).toBeNull();
  });

  it("普通 SkillTool gate proof 不应渲染为 JSON 明细", () => {
    const { container } = renderTool({
      id: "tool-skill-gate-proof-1",
      name: "SkillTool",
      arguments: JSON.stringify({ skill: "capability-report" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          allow: {
            phase: "skill_tool_gate_allow",
            hasRequest: true,
            hasDecision: true,
            hasResult: true,
            hasSourceMetadata: true,
            request: {
              toolName: "SkillTool",
              sessionId: "skill-source-session",
              skill: "capability-report",
              authorizationScope: "session",
            },
            decision: {
              action: "allow",
              gate: "session_allowlist",
              enabled: true,
              allowlisted: true,
              reason: "workspace_skill_runtime_enable_allowlist_matched",
            },
            result: {
              status: "passed",
              permissionBehavior: "Allow",
              sourceMetadataAttached: true,
              workspaceSkillRuntimeEnableAttached: true,
            },
          },
          sourceMetadata: {
            sourceDraftId: "capdraft-1",
            sourceVerificationReportId: "capver-1",
          },
          workspace_skill_runtime_enable: {
            source: "manual_session_enable",
            approval: "manual",
            bindings: [{ skill: "project:capability-report" }],
          },
          summary:
            "SkillTool allow/deny events both contain request, decision and result.",
        }),
      },
      startTime: new Date("2026-06-21T10:20:00.000Z"),
      endTime: new Date("2026-06-21T10:20:03.000Z"),
    });

    expect(container.textContent).toContain("已执行技能 capability-report");
    expect(container.textContent).toContain(
      "运行启用 · 手动会话 · 人工确认 · 1 个绑定",
    );
    expect(container.textContent).not.toContain("permissionBehavior");
    expect(container.textContent).not.toContain(
      "workspaceSkillRuntimeEnableAttached",
    );
    expect(container.textContent).not.toContain(
      "workspace_skill_runtime_enable",
    );
    expect(container.textContent).not.toContain("sourceMetadata");
    expect(container.textContent).not.toContain("skill-source-session");
    expect(container.textContent).not.toContain("SkillTool allow/deny");
    expect(
      container.querySelector('[data-testid="markdown-renderer"]'),
    ).toBeNull();
  });

  it("普通 SkillTool 正常输出不应被 gate 包络过滤误吞", () => {
    const { container } = renderTool({
      id: "tool-skill-output-1",
      name: "SkillTool",
      arguments: JSON.stringify({ skill: "capability-report" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          output: "已完成能力分析。",
          sourceMetadata: {
            sourceDraftId: "capdraft-1",
          },
          workspaceSkillRuntimeEnable: {
            enabledSkillNames: ["capability-report"],
          },
        }),
      },
      startTime: new Date("2026-06-21T10:21:00.000Z"),
      endTime: new Date("2026-06-21T10:21:03.000Z"),
    });

    expect(container.textContent).toContain("已完成能力分析。");
    expect(container.textContent).not.toContain("运行启用");
    expect(container.textContent).not.toContain("sourceDraftId");
    expect(container.textContent).not.toContain("workspaceSkillRuntimeEnable");
  });

  it("非命令工具的协议诊断包络不应渲染为 JSON 明细", () => {
    const { container } = renderTool({
      id: "tool-runtime-diagnostic-envelope-1",
      name: "mcp__runtime__diagnostic_probe",
      arguments: JSON.stringify({ probe: "tool-result-projection" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          request_metadata: {
            turnId: "turn-1",
            route: "turn/start",
          },
          diagnostics: {
            source: "runtime",
            code: "tool_result_projection",
          },
          metadata: {
            durationMs: 12,
          },
        }),
      },
      startTime: new Date("2026-06-21T10:22:00.000Z"),
      endTime: new Date("2026-06-21T10:22:03.000Z"),
    });

    expect(container.textContent).toContain("已完成");
    expect(container.textContent).not.toContain("request_metadata");
    expect(container.textContent).not.toContain("turn/start");
    expect(container.textContent).not.toContain("tool_result_projection");
    expect(
      container.querySelector('[data-testid="markdown-renderer"]'),
    ).toBeNull();
  });

  it("命令工具的 JSON stdout 不应被协议包络过滤误吞", () => {
    const { container } = renderTool({
      id: "tool-command-json-stdout-1",
      name: "Bash",
      arguments: JSON.stringify({ command: "node inspect.js" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          metadata: {
            durationMs: 12,
          },
          result: {
            ok: true,
          },
        }),
      },
      startTime: new Date("2026-06-21T10:23:00.000Z"),
      endTime: new Date("2026-06-21T10:23:03.000Z"),
    });

    expect(container.textContent).toContain('"durationMs"');
    expect(container.textContent).toContain('"ok"');
  });

  it("ToolSearch 展开后应展示结构化工具摘要，而不是原始 JSON", () => {
    const { container } = renderTool({
      id: "tool-search-1",
      name: "ToolSearch",
      arguments: JSON.stringify({ query: "select:Read,Write" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          query: "select:Read,Write",
          count: 2,
          notes: [],
          tools: [
            {
              name: "Read",
              source: "native_registry",
              description: "Read a file from disk",
              always_visible: true,
            },
            {
              name: "Write",
              source: "native_registry",
              description: "Write content to a file",
              always_visible: true,
            },
          ],
        }),
      },
      startTime: new Date("2026-04-13T10:10:00.000Z"),
      endTime: new Date("2026-04-13T10:10:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="展开过程详情"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(
      container.querySelector(
        '[data-testid="inline-tool-process-tool-search-result"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("已确认可用工具 2 个");
    expect(container.textContent).toContain("找到工具：2 个");
    expect(container.textContent).toContain("查看文件");
    expect(container.textContent).toContain("保存文件");
    expect(
      container.querySelector('[data-testid="markdown-renderer"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain('"always_visible":true');
    expect(container.textContent).not.toContain("Read a file from disk");
    expect(container.textContent).not.toContain("查询：select:Read,Write");
    expect(container.textContent).not.toContain("原生工具");
    expect(container.textContent).not.toContain("默认可见");
  });

  it("Bash 协议错误折叠态应展示底层原因而不是内部错误码", () => {
    const { container } = renderTool({
      id: "tool-bash-failed-1",
      name: "Bash",
      arguments: JSON.stringify({
        command: "set -e\np='/Users/coso/.yansu-agent'\nls \"$p\"",
      }),
      status: "failed",
      result: {
        success: false,
        error: "-32603: -32002: sandbox 执行失败: Operation not permitted",
        output: "",
      },
      startTime: new Date("2026-04-13T10:22:00.000Z"),
      endTime: new Date("2026-04-13T10:22:01.000Z"),
    });

    expect(container.textContent).toContain(
      "执行失败：sandbox 执行失败: Operation not permitted",
    );
    expect(container.textContent).not.toContain("-32603");
    expect(container.textContent).not.toContain("-32002");

    act(() => {
      const toggle = container.querySelector(
        'button[title="展开过程详情"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain(
      "原始错误：-32603: -32002: sandbox 执行失败: Operation not permitted",
    );
  });

  it("图片生成任务失败时不应展示内部错误码、工具名或长提示词", () => {
    const { container } = renderTool({
      id: "tool-image-generate-failed-1",
      name: "lime_create_image_generation_task",
      arguments: JSON.stringify({
        prompt: "A comic book style illustration of a formal statue",
      }),
      status: "failed",
      result: {
        success: false,
        error: "-32603: -32002: lime_create_image_generation_task",
        output: "",
      },
      startTime: new Date("2026-05-14T10:22:00.000Z"),
      endTime: new Date("2026-05-14T10:22:01.000Z"),
    });

    expect(container.textContent).toContain("生成失败");
    expect(container.textContent).not.toContain("开始失败");
    expect(container.textContent).not.toContain("-32603");
    expect(container.textContent).not.toContain("-32002");
    expect(container.textContent).not.toContain(
      "lime_create_image_generation_task",
    );
    expect(container.textContent).not.toContain(
      "A comic book style illustration",
    );

    act(() => {
      const toggle = container.querySelector(
        'button[title="展开过程详情"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("生成失败");
    expect(container.textContent).not.toContain("-32603");
    expect(container.textContent).not.toContain(
      "lime_create_image_generation_task",
    );
  });

  it("内容工作台任务失败时不应展示内部错误码或工具名", () => {
    const { container } = renderTool({
      id: "tool-video-generate-failed-1",
      name: "video_generate",
      arguments: JSON.stringify({
        prompt: "生成一个产品演示视频",
      }),
      status: "failed",
      result: {
        success: false,
        error: "-32603: -32002: video_generate",
        output: "",
      },
      startTime: new Date("2026-05-14T10:22:00.000Z"),
      endTime: new Date("2026-05-14T10:22:01.000Z"),
    });

    expect(container.textContent).toContain("视频生成失败");
    expect(container.textContent).not.toContain("-32603");
    expect(container.textContent).not.toContain("-32002");
    expect(container.textContent).not.toContain("video_generate");

    act(() => {
      const toggle = container.querySelector(
        'button[title="展开过程详情"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("视频生成失败");
    expect(container.textContent).not.toContain("-32603");
    expect(container.textContent).not.toContain("video_generate");
  });

  it("完成态过程卡不应重复展示执行完成与原始工具名", () => {
    const { container } = renderTool({
      id: "tool-inline-ask-user-1",
      name: "request_user_input",
      arguments: JSON.stringify({ question: "需要继续吗？" }),
      status: "completed",
      result: {
        success: true,
        output: "用户已确认继续。",
      },
      startTime: new Date("2026-04-13T10:30:00.000Z"),
      endTime: new Date("2026-04-13T10:30:01.000Z"),
    });

    expect(container.textContent).toContain("已收集 需要继续吗？");
    expect(container.textContent).not.toContain("执行完成");
    expect(container.textContent).not.toContain("Ask User Question");
  });
});
