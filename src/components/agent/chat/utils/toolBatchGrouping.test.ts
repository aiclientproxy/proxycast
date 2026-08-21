import { describe, expect, it } from "vitest";

import { limeI18nResources } from "@/i18n/createI18n";
import { SUPPORTED_LOCALES } from "@/i18n/locales";
import type { AgentToolCallState } from "@/lib/api/agentProtocol";

import { summarizeStreamingToolBatch } from "./toolBatchGrouping";

const TOOL_BATCH_COPY_KEYS = [
  "agentChat.toolBatch.separator.clause",
  "agentChat.toolBatch.webSearch.fallback.searchAndFetch",
  "agentChat.toolBatch.webSearch.fallback.searchOnly",
  "agentChat.toolBatch.webSearch.latestHint",
  "agentChat.toolBatch.webSearch.title.running.singleWithHint",
  "agentChat.toolBatch.webSearch.title.completed.singleWithHint",
  "agentChat.toolBatch.webSearch.title.running.searchAndFetch",
  "agentChat.toolBatch.webSearch.title.completed.searchAndFetch",
  "agentChat.toolBatch.webSearch.title.running.searchOnly",
  "agentChat.toolBatch.webSearch.title.completed.searchOnly",
  "agentChat.toolBatch.webSearch.count.searchAndFetch",
  "agentChat.toolBatch.webSearch.count.searchOnly",
  "agentChat.toolBatch.webSearch.rawDetail.running.searchAndFetch",
  "agentChat.toolBatch.webSearch.rawDetail.running.searchOnly",
  "agentChat.toolBatch.webSearch.rawDetail.completed.searchAndFetch",
  "agentChat.toolBatch.webSearch.rawDetail.completed.searchOnly",
  "agentChat.toolBatch.exploration.title.mixed",
  "agentChat.toolBatch.exploration.title.read",
  "agentChat.toolBatch.exploration.title.search",
  "agentChat.toolBatch.exploration.title.list",
  "agentChat.toolBatch.exploration.detail.read",
  "agentChat.toolBatch.exploration.detail.search",
  "agentChat.toolBatch.exploration.detail.list",
  "agentChat.toolBatch.exploration.latestHint",
  "agentChat.toolBatch.exploration.count.read",
  "agentChat.toolBatch.exploration.count.search",
  "agentChat.toolBatch.exploration.count.list",
  "agentChat.toolBatch.exploration.count.steps",
  "agentChat.toolBatch.exploration.rawDetail",
  "agentChat.toolBatch.browser.title",
  "agentChat.toolBatch.browser.fallbackLine",
  "agentChat.toolBatch.browser.latestHint",
  "agentChat.toolBatch.browser.count",
  "agentChat.toolBatch.browser.rawDetail",
] as const;

function createToolCall(
  name: string,
  argumentsValue?: Record<string, unknown>,
): AgentToolCallState {
  return {
    id: `${name}-1`,
    name,
    status: "completed",
    arguments: argumentsValue ? JSON.stringify(argumentsValue) : undefined,
    startTime: new Date("2026-04-14T00:00:00.000Z"),
    endTime: new Date("2026-04-14T00:00:01.000Z"),
    result: {
      success: true,
      output: "ok",
    },
  };
}

describe("toolBatchGrouping", () => {
  it("工具批次 copy key 应覆盖所有 current locale", () => {
    for (const locale of SUPPORTED_LOCALES) {
      for (const key of TOOL_BATCH_COPY_KEYS) {
        expect(limeI18nResources[locale].agent).toHaveProperty(key);
      }
    }
  });

  it("应把 MCP 搜索与读取归入探索批次", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("mcp__github__search_code", {
        query: "repo:lime tool runtime",
      }),
      createToolCall("mcp__github__get_file_contents", {
        path: "docs/guide.md",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "exploration",
        title: "已探索项目",
        countLabel: "读 1 / 搜 1",
        rawDetailLabel: "展开查看探索明细",
      }),
    );
    expect(summary?.supportingLines).toContain("查看了 1 个文件，搜索 1 次");
    expect(summary?.supportingLines).toContain("最新线索：guide.md");
  });

  it("未知动态 MCP list/read/search 工具应保持探索折叠而不是退回散列工具", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("mcp__docs__list_pages", {
        path: "docs",
      }),
      createToolCall("mcp__docs__read_page", {
        path: "docs/runtime.md",
      }),
      createToolCall("mcp__linear__query_issues", {
        query: "runtime empty final",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "exploration",
        title: "已探索项目",
        countLabel: "读 1 / 搜 1 / 列 1",
        rawDetailLabel: "展开查看探索明细",
      }),
    );
    expect(summary?.supportingLines).toContain(
      "查看了 1 个文件，搜索 1 次，列了 1 个目录",
    );
    expect(summary?.supportingLines).toContain("最新线索：runtime empty final");
  });

  it("应让 REPL 调用被吸收到探索批次而不打断摘要", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("Read", {
        file_path: "src/main.ts",
      }),
      createToolCall("REPLTool", {
        code: 'rg "tool inventory" src',
      }),
      createToolCall("Grep", {
        pattern: "tool inventory",
        path: "src",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "exploration",
        countLabel: "读 1 / 搜 1",
      }),
    );
  });

  it("技能辅助 alias 应被吸收到探索批次而不打断摘要", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("Read", {
        file_path: "src/main.ts",
      }),
      createToolCall("ListSkills"),
      createToolCall("LoadSkill", {
        name: "browser",
      }),
      createToolCall("Grep", {
        pattern: "tool inventory",
        path: "src",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "exploration",
        title: "已探索项目",
        countLabel: "读 1 / 搜 1",
      }),
    );
  });

  it("子任务控制 alias 不应被错误折叠成探索或网页搜索批次", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("WaitAgent", { id: "agent-1" }),
      createToolCall("ResumeAgent", { id: "agent-1" }),
      createToolCall("CloseAgent", { id: "agent-1" }),
    ]);

    expect(summary).toBeNull();
  });

  it("应把 WebSearch 工具批次展示为网页搜索轨迹", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("web_search", {
        query: "today world news Reuters",
      }),
      createToolCall("WebSearchTool", {
        query: "AP world news June 2026",
      }),
      createToolCall("mcp__system__web_search", {
        url: "https://apnews.com/hub/world-news",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "已搜索网页 3 次",
        countLabel: "3 次",
        rawDetailLabel: "展开查看搜索来源",
      }),
    );
    expect(summary?.supportingLines).toContain("today world news Reuters");
    expect(summary?.supportingLines).toContain("AP world news June 2026");
    expect(summary?.supportingLines).toContain(
      "https://apnews.com/hub/world-news",
    );
  });

  it("应优先使用 tool_process_facts.operationKind 聚合未知工具批次", () => {
    const summary = summarizeStreamingToolBatch([
      {
        ...createToolCall("RuntimeProvidedTool"),
        metadata: {
          tool_process_facts: {
            source: "runtime_facts",
            toolName: "RuntimeProvidedTool",
            operationKind: "web_search",
            subject: "Soul output surface",
          },
        },
      },
      {
        ...createToolCall("RuntimeFetchTool"),
        metadata: {
          tool_process_facts: {
            source: "runtime_facts",
            toolName: "RuntimeFetchTool",
            operation_kind: "web_fetch",
            subject: "https://example.com/soul",
          },
        },
      },
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "已搜索网页 1 次，读取网页 1 次",
        countLabel: "搜 1 / 读 1",
      }),
    );
    expect(summary?.supportingLines).toEqual(
      expect.arrayContaining(["Soul output surface", "example.com/soul"]),
    );
  });

  it("应把工具生命周期 Soul metadata 透传到批次 descriptor", () => {
    const summary = summarizeStreamingToolBatch([
      {
        ...createToolCall("web_search", {
          query: "Lime Soul tool lifecycle",
        }),
        metadata: {
          soul_lifecycle: {
            surface: "tool_lifecycle",
            phase: "after_tool_success",
            status: "completed",
            styleLevel: "L2",
            riskLevel: "normal",
            toneVariant: "cheeky_sassy",
            profileId: "cheeky_sassy_executor",
            packId: "com.lime.soul.cheeky-sassy-executor",
          },
          soul_surface: "tool_lifecycle",
          soul_phase: "after_tool_success",
          style_level: "L2",
          risk_level: "normal",
          tone_variant: "cheeky_sassy",
          profile_id: "cheeky_sassy_executor",
          pack_id: "com.lime.soul.cheeky-sassy-executor",
        },
      },
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        soulLifecycle: expect.objectContaining({
          phase: "after_tool_success",
          surface: "tool_lifecycle",
        }),
        soulSurface: "tool_lifecycle",
        soulPhase: "after_tool_success",
        styleLevel: "L2",
        riskLevel: "normal",
        toneVariant: "cheeky_sassy",
        profileId: "cheeky_sassy_executor",
        packId: "com.lime.soul.cheeky-sassy-executor",
      }),
    );
  });

  it("单条 WebSearch 也应生成网页搜索摘要", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("web_search", {
        query: "Lime history import",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "已搜索网页：Lime history import",
        countLabel: "1 次",
        rawDetailLabel: "展开查看搜索来源",
      }),
    );
    expect(summary?.supportingLines).toContain("Lime history import");
  });

  it("Codex web_search action object 应使用 query 作为搜索标题而不是渲染 JSON", () => {
    const summary = summarizeStreamingToolBatch([
      {
        ...createToolCall("web_search", {
          action: {
            type: "search_query",
            query: "codex desktop search rendering",
          },
        }),
        status: "running",
      },
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "正在搜索网页 codex desktop search rendering",
      }),
    );
    expect(summary?.supportingLines.join("\n")).toContain(
      "codex desktop search rendering",
    );
    expect(summary?.supportingLines.join("\n")).not.toContain("{");
    expect(summary?.supportingLines.join("\n")).not.toContain("search_query");
  });

  it("全部 WebSearch 失败时仍应聚合为轻量网页搜索轨迹", () => {
    const diagnosticOutput = JSON.stringify({
      metadata: {
        web_search: {
          attempts: [
            {
              error: "缺少环境变量 TAVILY_API_KEY",
            },
          ],
        },
      },
    });
    const summary = summarizeStreamingToolBatch([
      {
        ...createToolCall("web_search", {
          query: "today world news Reuters",
        }),
        status: "failed",
        result: {
          success: false,
          output: diagnosticOutput,
          error: diagnosticOutput,
        },
      },
      {
        ...createToolCall("mcp__system__web_search", {
          query: "AP world news June 2026",
        }),
        status: "failed",
        result: {
          success: false,
          output: diagnosticOutput,
          error: diagnosticOutput,
        },
      },
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "已搜索网页 2 次",
        countLabel: "2 次",
        rawDetailLabel: "展开查看搜索来源",
      }),
    );
    expect(summary?.supportingLines).toEqual(
      expect.arrayContaining([
        "today world news Reuters",
        "AP world news June 2026",
      ]),
    );
    expect(summary?.supportingLines.join("\n")).not.toContain("TAVILY_API_KEY");
    expect(summary?.supportingLines.join("\n")).not.toContain("metadata");
  });

  it("运行中的 WebSearch / WebFetch 应展示搜索进行态而不是完成态", () => {
    const summary = summarizeStreamingToolBatch([
      {
        ...createToolCall("web_search", {
          query: "today world news Reuters",
        }),
        status: "running",
      },
      {
        ...createToolCall("WebFetch", {
          url: "https://www.reuters.com/world/",
        }),
        status: "running",
      },
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "正在搜索网页 1 次，读取网页 1 次",
        countLabel: "搜 1 / 读 1",
        rawDetailLabel: "展开查看搜索与读取进度",
      }),
    );
    expect(summary?.supportingLines).toEqual(
      expect.arrayContaining(["today world news Reuters", "reuters.com/world"]),
    );
  });

  it("应从 WebSearch 结果中提取来源行而不暴露原始输出", () => {
    const summary = summarizeStreamingToolBatch([
      {
        ...createToolCall("WebSearch", {
          query: "today world news",
        }),
        result: {
          success: true,
          output: JSON.stringify({
            results: [
              {
                title: "了解必应 获取新版必应壁纸应用 增值电信业务经营许可证",
                url: "https://www.bing.com/search?q=today+world+news",
              },
              {
                title: "Reuters World News",
                url: "https://www.reuters.com/world/",
              },
            ],
          }),
        },
      },
      {
        ...createToolCall("mcp__news__web_search", {
          query: "global headlines",
        }),
        result: {
          success: true,
          output: "[AP World News](https://apnews.com/hub/world-news)",
        },
      },
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "已搜索网页 2 次",
      }),
    );
    expect(summary?.supportingLines).toEqual(
      expect.arrayContaining([
        "today world news",
        "Reuters World News",
        "global headlines",
        "AP World News",
      ]),
    );
    expect(summary?.supportingLines.join("\n")).not.toContain("必应");
    expect(summary?.supportingLines.join("\n")).not.toContain(
      "增值电信业务经营许可证",
    );
  });

  it("应把 provider SearchQuery 别名展示为网页搜索轨迹", () => {
    const summary = summarizeStreamingToolBatch([
      {
        ...createToolCall("SearchQuery", {
          q: "2026-06-03 international news",
        }),
        result: {
          success: true,
          output: JSON.stringify({
            results: [
              {
                title: "Reuters World News",
                url: "https://www.reuters.com/world/",
              },
            ],
          }),
        },
      },
      createToolCall("search_query", {
        q: "UN global headlines June 2026",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "已搜索网页 2 次",
        countLabel: "2 次",
        rawDetailLabel: "展开查看搜索来源",
      }),
    );
    expect(summary?.supportingLines).toEqual(
      expect.arrayContaining([
        "2026-06-03 international news",
        "Reuters World News",
        "UN global headlines June 2026",
      ]),
    );
  });

  it("Context7 文档工具应归入探索摘要而不是网页搜索来源", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("ResolveLibraryId", {
        libraryName: "Next.js",
      }),
      createToolCall("QueryDocs", {
        query: "React useEffect cleanup",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "exploration",
        title: "已探索项目",
        countLabel: "读 1 / 搜 1",
        rawDetailLabel: "展开查看探索明细",
      }),
    );
    expect(summary?.supportingLines.join("\n")).not.toContain("搜索来源");
  });

  it("结构化数据工具不应被折叠成网页搜索批次", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("finance", {
        ticker: "AAPL",
      }),
      createToolCall("weather", {
        location: "Tokyo",
      }),
      createToolCall("time", {
        utc_offset: "+09:00",
      }),
    ]);

    expect(summary).toBeNull();
  });

  it("WebSearch 后续 WebFetch 成功或失败都应吸收到网页搜索批次", () => {
    const summary = summarizeStreamingToolBatch([
      {
        ...createToolCall("web_search", {
          query: "June 2 2026 world news",
        }),
        result: {
          success: true,
          output: JSON.stringify({
            results: [
              {
                title: "Reuters World News",
                url: "https://www.reuters.com/world/",
              },
            ],
          }),
        },
      },
      {
        ...createToolCall("WebFetch", {
          url: "https://www.reuters.com/world/",
        }),
        status: "failed",
        result: {
          success: false,
          output: "503 Service Unavailable",
          error: "503 Service Unavailable",
        },
      },
      {
        ...createToolCall("WebFetch", {
          url: "https://news.un.org/en/",
        }),
        result: {
          success: true,
          output: JSON.stringify({
            code: 200,
            result: "large raw page payload should not become the batch title",
          }),
        },
      },
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "web_search",
        title: "已搜索网页 1 次，读取网页 2 次",
        countLabel: "搜 1 / 读 2",
        rawDetailLabel: "展开查看搜索与读取来源",
      }),
    );
    expect(summary?.supportingLines).toEqual(
      expect.arrayContaining([
        "June 2 2026 world news",
        "Reuters World News",
        "reuters.com/world",
        "news.un.org/en",
      ]),
    );
    expect(summary?.supportingSections).toEqual([
      {
        kind: "web_search_sources",
        lines: ["June 2 2026 world news", "Reuters World News"],
      },
      {
        kind: "web_fetch_pages",
        lines: ["reuters.com/world", "news.un.org/en"],
      },
    ]);
  });

  it("普通搜索工具仍应按项目线索展示，避免混成 WebSearch", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("mcp__github__search_code", {
        query: "runtime policy",
      }),
      createToolCall("Grep", {
        pattern: "runtime policy",
        path: "src",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "exploration",
        title: "已搜索关键线索",
        countLabel: "搜 2",
        rawDetailLabel: "展开查看探索明细",
      }),
    );
  });

  it("动态 MCP mutation 不应被探索批次吸收", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("mcp__github__search_code", {
        query: "runtime policy",
      }),
      createToolCall("mcp__github__create_issue", {
        title: "修复工具渲染",
      }),
    ]);

    expect(summary).toBeNull();
  });

  it("MCP resource 列表与读取应保留探索批次摘要", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("ListMcpResourcesTool", {
        server: "docs",
      }),
      createToolCall("ReadMcpResourceTool", {
        uri: "file:///guide.md",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "exploration",
        title: "已查看关键文件",
        countLabel: "读 1 / 列 1",
        rawDetailLabel: "展开查看探索明细",
      }),
    );
  });

  it("应继续把浏览器 MCP 步骤聚合为页面检查摘要", () => {
    const summary = summarizeStreamingToolBatch([
      createToolCall("mcp__site__navigate", {
        url: "https://example.com",
      }),
      createToolCall("mcp__site__click", {
        selector: "#cta",
      }),
    ]);

    expect(summary).toEqual(
      expect.objectContaining({
        kind: "browser",
        title: "已检查页面",
        countLabel: "2 步",
      }),
    );
    expect(summary?.supportingLines).toContain("https://example.com");
    expect(summary?.supportingLines).toContain("#cta");
  });
});
