import { describe, expect, it } from "vitest";

import {
  classifyMcpToolOperationKind,
  isBrowserToolName,
  isUpdatePlanToolName,
  isUnifiedWebFetchToolName,
  isUnifiedWebSearchToolName,
  parseMcpToolName,
} from "./toolNameFamily";

describe("toolNameFamily", () => {
  it("应按 MCP 命名解析 server 与 inner tool", () => {
    expect(parseMcpToolName("mcp__news__web_search")).toEqual({
      serverName: "news",
      innerToolName: "web_search",
      normalizedInnerToolName: "web_search",
    });
    expect(parseMcpToolName("mcp__site__browserSnapshot")).toEqual({
      serverName: "site",
      innerToolName: "browserSnapshot",
      normalizedInnerToolName: "browser_snapshot",
    });
  });

  it("应识别 WebSearch alias 与动态 MCP web_search", () => {
    expect(isUnifiedWebSearchToolName("WebSearch")).toBe(true);
    expect(isUnifiedWebSearchToolName("WebSearchTool")).toBe(true);
    expect(isUnifiedWebSearchToolName("web_search")).toBe(true);
    expect(isUnifiedWebSearchToolName("mcp__system__web_search")).toBe(true);
    expect(isUnifiedWebSearchToolName("mcp__news__web_search")).toBe(true);
  });

  it("不应把普通 MCP 搜索误判为 WebSearch", () => {
    expect(isUnifiedWebSearchToolName("mcp__github__search_code")).toBe(false);
    expect(isUnifiedWebSearchToolName("mcp__linear__query_issues")).toBe(false);
  });

  it("应识别 WebFetch alias 与动态 MCP web_fetch", () => {
    expect(isUnifiedWebFetchToolName("WebFetch")).toBe(true);
    expect(isUnifiedWebFetchToolName("WebFetchTool")).toBe(true);
    expect(isUnifiedWebFetchToolName("web_fetch")).toBe(true);
    expect(isUnifiedWebFetchToolName("mcp__system__web_fetch")).toBe(true);
    expect(isUnifiedWebFetchToolName("mcp__news__web_fetch")).toBe(true);
  });

  it("应保留 MCP 普通操作族分类", () => {
    expect(classifyMcpToolOperationKind("mcp__github__search_code")).toBe(
      "search",
    );
    expect(classifyMcpToolOperationKind("mcp__github__get_file_contents")).toBe(
      "read",
    );
    expect(classifyMcpToolOperationKind("mcp__docs__read_page")).toBe("read");
    expect(classifyMcpToolOperationKind("mcp__docs__list_pages")).toBe("list");
    expect(classifyMcpToolOperationKind("mcp__browser__click")).toBe("browser");
    expect(classifyMcpToolOperationKind("mcp__github__create_issue")).toBe(
      "mutation",
    );
    expect(classifyMcpToolOperationKind("mcp__linear__update_task")).toBe(
      "mutation",
    );
    expect(classifyMcpToolOperationKind("mcp__runner__execute_job")).toBe(
      "mutation",
    );
  });

  it("应识别浏览器工具族", () => {
    expect(isBrowserToolName("browser_navigate")).toBe(true);
    expect(isBrowserToolName("mcp__playwright__browser_click")).toBe(true);
    expect(isBrowserToolName("mcp__github__search_code")).toBe(false);
  });

  it("应识别 update_plan 工具别名", () => {
    expect(isUpdatePlanToolName("update_plan")).toBe(true);
    expect(isUpdatePlanToolName("UpdatePlanTool")).toBe(true);
    expect(isUpdatePlanToolName("update-plan")).toBe(true);
    expect(isUpdatePlanToolName("task_update")).toBe(false);
  });
});
