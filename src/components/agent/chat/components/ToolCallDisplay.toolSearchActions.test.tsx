import { act } from "react";
import { describe, expect, it } from "vitest";
import { renderTool, renderToolList } from "./ToolCallDisplay.testFixtures";

describe("ToolCallDisplay tool search and actions", () => {
  it("ToolSearch 展开后应展示结构化工具摘要，而不是原始 JSON", () => {
    const { container } = renderTool({
      id: "tool-search-bridge-1",
      name: "ToolSearch",
      arguments: JSON.stringify({ query: "select:Read,Write" }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          query: "select:Read,Write",
          caller: "assistant",
          count: 2,
          notes: [],
          tools: [
            {
              name: "Read",
              source: "native_registry",
              description: "Read a file from disk",
              callable: true,
              call_name: "Read",
              activation: null,
              always_visible: true,
            },
            {
              name: "Write",
              source: "native_registry",
              description: "Write content to a file",
              callable: true,
              call_name: "Write",
              activation: null,
              always_visible: true,
            },
          ],
        }),
      },
      startTime: new Date("2026-04-10T04:00:00.000Z"),
      endTime: new Date("2026-04-10T04:00:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(
      container.querySelector('[data-testid="tool-call-tool-search-result"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("找到工具：2 个");
    expect(container.textContent).toContain("查看文件");
    expect(container.textContent).toContain("保存文件");
    expect(container.textContent).not.toContain('"caller"');
    expect(container.textContent).not.toContain('"tools"');
    expect(container.textContent).not.toContain('"call_name"');
    expect(container.textContent).not.toContain("Read a file from disk");
    expect(container.textContent).not.toContain('"always_visible":true');
    expect(container.textContent).not.toContain("查询：select:Read,Write");
    expect(container.textContent).not.toContain("原生工具");
    expect(container.textContent).not.toContain("默认可见");
    expect(
      container.querySelector('[data-testid="tool-call-rendered-result"]'),
    ).toBeNull();
  });

  it("ToolSearch 在流式阶段不应自动展开内部结果", () => {
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
        startTime: new Date("2026-04-10T04:05:00.000Z"),
        endTime: new Date("2026-04-10T04:05:01.000Z"),
      },
      { isMessageStreaming: true },
    );

    expect(
      container.querySelector('[data-testid="tool-call-tool-search-result"]'),
    ).toBeNull();
    expect(container.textContent).not.toContain("找到工具：2 个");
    expect(container.textContent).not.toContain("Read a file from disk");
  });

  it("ToolSearch 分组子行应使用过程摘要，不应展示已搜索可用工具", () => {
    const { container } = renderTool(
      {
        id: "tool-search-grouped-1",
        name: "ToolSearch",
        arguments: JSON.stringify({ query: "select:WebSearch" }),
        status: "completed",
        result: {
          success: true,
          output: JSON.stringify({
            query: "select:WebSearch",
            caller: "assistant",
            count: 1,
            notes: [],
            tools: [
              {
                name: "WebSearch",
                source: "native_registry",
                callable: true,
                call_name: "WebSearch",
                activation: null,
              },
            ],
          }),
        },
        startTime: new Date("2026-04-10T04:06:00.000Z"),
        endTime: new Date("2026-04-10T04:06:01.000Z"),
      },
      { grouped: true },
    );

    expect(container.textContent).toContain("已确认可用工具 1 个 · 搜索网页");
    expect(container.textContent).not.toContain("已搜索 可用工具");
    expect(container.textContent).not.toContain("确认 工具入口");
    expect(container.textContent).not.toContain("WebSearch");
  });

  it("应为浏览器、委派、任务输出与交互类工具生成具体动作句", () => {
    const { container } = renderToolList({
      toolCalls: [
        {
          id: "tool-browser-1",
          name: "mcp__site__navigate",
          arguments: JSON.stringify({ url: "https://example.com/docs" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-20T12:20:00.000Z"),
          endTime: new Date("2026-03-20T12:20:01.000Z"),
        },
        {
          id: "tool-subagent-1",
          name: "Agent",
          arguments: JSON.stringify({ description: "修复登录页" }),
          status: "running",
          startTime: new Date("2026-03-20T12:20:02.000Z"),
        },
        {
          id: "tool-output-1",
          name: "TaskOutput",
          arguments: JSON.stringify({ task_id: "video-task-1" }),
          status: "completed",
          result: { success: true, output: "done" },
          startTime: new Date("2026-03-20T12:20:03.000Z"),
          endTime: new Date("2026-03-20T12:20:04.000Z"),
        },
        {
          id: "tool-skill-1",
          name: "load_skill",
          arguments: JSON.stringify({ name: "lime-governance" }),
          status: "completed",
          result: { success: true, output: "loaded" },
          startTime: new Date("2026-03-20T12:20:05.000Z"),
          endTime: new Date("2026-03-20T12:20:06.000Z"),
        },
        {
          id: "tool-glob-1",
          name: "glob",
          arguments: JSON.stringify({ pattern: "src/**/*.tsx" }),
          status: "completed",
          result: { success: true, output: "matched" },
          startTime: new Date("2026-03-20T12:20:07.000Z"),
          endTime: new Date("2026-03-20T12:20:08.000Z"),
        },
        {
          id: "tool-input-1",
          name: "request_user_input",
          arguments: JSON.stringify({ question: "需要继续吗？" }),
          status: "running",
          startTime: new Date("2026-03-20T12:20:09.000Z"),
        },
        {
          id: "tool-list-peers-1",
          name: "ListPeers",
          arguments: JSON.stringify({}),
          status: "completed",
          result: { success: true, output: "[]" },
          startTime: new Date("2026-03-20T12:20:10.000Z"),
          endTime: new Date("2026-03-20T12:20:11.000Z"),
        },
        {
          id: "tool-team-create-1",
          name: "TeamCreate",
          arguments: JSON.stringify({ team_name: "当前子代理组" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-03-20T12:20:11.000Z"),
          endTime: new Date("2026-03-20T12:20:12.000Z"),
        },
        {
          id: "tool-team-delete-1",
          name: "TeamDelete",
          arguments: JSON.stringify({ team_name: "当前子代理组" }),
          status: "completed",
          result: { success: true, output: "{}" },
          startTime: new Date("2026-03-20T12:20:12.000Z"),
          endTime: new Date("2026-03-20T12:20:13.000Z"),
        },
        {
          id: "tool-remote-trigger-1",
          name: "RemoteTrigger",
          arguments: JSON.stringify({
            action: "run",
            trigger_id: "remote-1",
          }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-20T12:20:14.000Z"),
          endTime: new Date("2026-03-20T12:20:15.000Z"),
        },
        {
          id: "tool-cron-delete-1",
          name: "CronDelete",
          arguments: JSON.stringify({ id: "cron-job-1" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-20T12:20:16.000Z"),
          endTime: new Date("2026-03-20T12:20:17.000Z"),
        },
      ],
    });

    expect(container.textContent).toContain("已打开 https://example.com/docs");
    expect(container.textContent).toContain("子任务处理中 修复登录页");
    expect(container.textContent).toContain("已查看结果 video-task-1");
    expect(container.textContent).toContain("已加载技能 lime-governance");
    expect(container.textContent).toContain("已找到 src/**/*.tsx");
    expect(container.textContent).toContain("等待输入 需要继续吗？");
    expect(container.textContent).toContain("已查看 当前子代理组");
    expect(container.textContent).toContain("已创建 当前子代理组");
    expect(container.textContent).toContain("已删除 当前子代理组");
    expect(container.textContent).toContain("RemoteTrigger");
    expect(container.textContent).toContain("CronDelete");
  });

  it("历史 gated runtime 工具即使当前未注册也应展示可读过程", () => {
    const { container } = renderToolList({
      toolCalls: [
        {
          id: "tool-config-history-1",
          name: "ConfigTool",
          arguments: JSON.stringify({ key: "model" }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-20T12:30:00.000Z"),
          endTime: new Date("2026-03-20T12:30:01.000Z"),
        },
        {
          id: "tool-cron-create-history-1",
          name: "CronCreateTool",
          arguments: JSON.stringify({
            id: "daily-summary",
            prompt: "整理今天的国际新闻",
          }),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-20T12:30:02.000Z"),
          endTime: new Date("2026-03-20T12:30:03.000Z"),
        },
        {
          id: "tool-remote-trigger-history-1",
          name: "RemoteTriggerTool",
          arguments: JSON.stringify({ trigger_id: "remote-news" }),
          status: "failed",
          result: {
            success: false,
            error: "-32603: -32002: remote trigger runtime is not configured",
            output: "",
          },
          startTime: new Date("2026-03-20T12:30:04.000Z"),
          endTime: new Date("2026-03-20T12:30:05.000Z"),
        },
        {
          id: "tool-worktree-history-1",
          name: "EnterWorktreeTool",
          arguments: JSON.stringify({}),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-20T12:30:08.000Z"),
          endTime: new Date("2026-03-20T12:30:09.000Z"),
        },
        {
          id: "tool-worktree-history-2",
          name: "ExitWorktreeTool",
          arguments: JSON.stringify({}),
          status: "completed",
          result: { success: true, output: "ok" },
          startTime: new Date("2026-03-20T12:30:10.000Z"),
          endTime: new Date("2026-03-20T12:30:11.000Z"),
        },
      ],
    });

    expect(container.textContent).toContain("ConfigTool");
    expect(container.textContent).toContain("CronCreateTool");
    expect(container.textContent).toContain("Remote Trigger Tool");
    expect(container.textContent).toContain("已完成 2 个步骤");
    expect(container.textContent).toContain("EnterWorktreeTool");
    expect(container.textContent).toContain("ExitWorktreeTool");
    expect(container.textContent).not.toContain("-32603");
    expect(container.textContent).not.toContain('"trigger_id"');
    expect(container.textContent).not.toContain('"operation"');
  });

  it("写文件工具应通过 artifact protocol 解析嵌套产物路径", () => {
    const { container } = renderTool({
      id: "tool-write-nested-1",
      name: "write_file",
      arguments: JSON.stringify({
        payload: {
          artifact_paths: ["content-posts\\final.md"],
        },
      }),
      status: "completed",
      result: {
        success: true,
        output: "# 最终稿",
      },
      startTime: new Date("2026-03-25T09:00:00.000Z"),
      endTime: new Date("2026-03-25T09:00:01.000Z"),
    });

    expect(container.textContent).toContain("已保存 final.md");
  });
});
