import { describe, expect, it } from "vitest";
import {
  collectArtifactDocumentSourcesFromToolCalls,
  mergeArtifactDocuments,
} from "./artifactToolSources";

describe("artifactToolSources", () => {
  it("应从搜索工具结果提取 web 来源", () => {
    const sources = collectArtifactDocumentSourcesFromToolCalls([
      {
        id: "tool-search-1",
        name: "WebSearch",
        arguments: JSON.stringify({ query: "Lime artifact workbench" }),
        result: {
          success: true,
          output: JSON.stringify({
            results: [
              {
                title: "Artifact Workbench",
                url: "https://example.com/workbench",
                snippet: "统一交付层与版本来源面板。",
              },
            ],
          }),
        },
      },
    ]);

    expect(sources).toEqual([
      expect.objectContaining({
        id: "web:https://example.com/workbench",
        type: "search_result",
        label: "Artifact Workbench",
        locator: expect.objectContaining({
          url: "https://example.com/workbench",
        }),
        snippet: "统一交付层与版本来源面板。",
      }),
    ]);
  });

  it("应从 current 浏览器工具结果提取 browser 来源", () => {
    const sources = collectArtifactDocumentSourcesFromToolCalls([
      {
        id: "tool-browser-1",
        name: "browser__navigate",
        arguments: JSON.stringify({ url: "https://example.com/publish" }),
        result: {
          success: true,
          output: "已打开页面",
          metadata: {
            tool_family: "browser",
            browser_session: {
              session_id: "session-1",
              profile_key: "social-publish",
            },
            page_info: {
              title: "发布页",
              url: "https://example.com/publish",
            },
          },
        },
      },
    ]);

    expect(sources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "web",
          label: "发布页",
          locator: expect.objectContaining({
            url: "https://example.com/publish",
          }),
        }),
      ]),
    );
  });

  it("应从文件工具参数提取 file 来源", () => {
    const sources = collectArtifactDocumentSourcesFromToolCalls([
      {
        id: "tool-file-1",
        name: "read_file",
        arguments: JSON.stringify({
          path: "research/market-notes.md",
        }),
        result: {
          success: true,
          output: "# 市场笔记",
        },
      },
    ]);

    expect(sources).toEqual([
      expect.objectContaining({
        id: "file:research/market-notes.md",
        type: "file",
        label: "market-notes.md",
        locator: expect.objectContaining({
          path: "research/market-notes.md",
        }),
      }),
    ]);
  });

  it("应在合并文档时保留来源与 sourceLinks", () => {
    const merged = mergeArtifactDocuments(
      {
        schemaVersion: "artifact_document.v1",
        artifactId: "artifact-document:demo",
        kind: "analysis",
        title: "季度复盘",
        status: "ready",
        language: "zh-CN",
        summary: "主摘要",
        blocks: [
          {
            id: "hero-1",
            type: "hero_summary",
            summary: "主摘要",
            sourceIds: ["source-1"],
          },
        ],
        sources: [
          {
            id: "source-1",
            type: "web",
            label: "OpenAI Blog",
            locator: {
              url: "https://openai.com/blog",
            },
          },
        ],
        metadata: {
          sourceLinks: [
            {
              artifactId: "artifact-document:demo",
              blockId: "hero-1",
              sourceId: "source-1",
              sourceType: "web",
              sourceRef: "https://openai.com/blog",
            },
          ],
        },
      },
      {
        schemaVersion: "artifact_document.v1",
        artifactId: "artifact-document:demo",
        kind: "analysis",
        title: "季度复盘",
        status: "ready",
        language: "zh-CN",
        summary: undefined,
        blocks: [
          {
            id: "hero-1",
            type: "hero_summary",
            summary: "主摘要",
          },
        ],
        sources: [
          {
            id: "source-2",
            type: "web",
            label: "产品文档",
            locator: {
              url: "https://example.com/docs",
            },
          },
        ],
        metadata: {},
      },
    );

    expect(merged?.summary).toBe("主摘要");
    expect(merged?.sources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "source-1" }),
        expect.objectContaining({ id: "source-2" }),
      ]),
    );
    expect(merged?.metadata.sourceLinks).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          blockId: "hero-1",
          sourceId: "source-1",
        }),
      ]),
    );
  });
});
