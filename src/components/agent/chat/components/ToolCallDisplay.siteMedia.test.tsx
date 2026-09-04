import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import { renderTool } from "./ToolCallDisplay.testFixtures";

describe("ToolCallDisplay site and media results", () => {
  it("Skill 工具调用应能查看本次执行读取的 SKILL.md 内容", () => {
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
      const skillContentButton = Array.from(
        container.querySelectorAll("button"),
      ).find((button) => button.textContent?.includes("SKILL.md"));
      skillContentButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(
      container.querySelector('[data-testid="tool-call-skill-content-panel"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("执行时读取的 SKILL.md");
    expect(container.textContent).toContain("随本次执行记录保存");
    expect(container.textContent).toContain("Agent Skills 标准");
    expect(container.textContent).toContain("展开 SKILL.md 内容");
    expect(container.textContent).not.toContain("Analysis Skill");

    act(() => {
      const expandBodyButton = Array.from(
        container.querySelectorAll("button"),
      ).find((button) => button.textContent?.includes("展开 SKILL.md 内容"));
      expandBodyButton?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(
      container.querySelector('[data-testid="tool-call-skill-content-body"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("收起 SKILL.md 内容");
    expect(container.textContent).toContain("Analysis Skill");
    expect(container.textContent).toContain("必须先确认可见上下文。");
    expect(container.textContent).not.toContain("tool_family");
    expect(container.textContent).not.toContain("skill_markdown_content");
  });

  it("完整 SkillTool 结果卡应隐藏 gate proof 协议包络", () => {
    const { container } = renderTool({
      id: "skill:gate-proof-1",
      name: "Skill",
      arguments: JSON.stringify({
        skill: "analysis",
        input: "分析当前问题",
      }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          phase: "skill_tool_gate.completed",
          request: {
            toolName: "SkillTool",
            skillName: "analysis",
          },
          decision: {
            action: "allow",
            gate: "workspaceSkillRuntimeEnableAttached",
          },
          result: {
            status: "allowed",
          },
          sourceMetadata: {
            source: "skill-source-session",
            permissionBehavior: "workspaceSkillRuntimeEnableAttached",
          },
        }),
      },
      startTime: new Date("2026-06-21T12:05:00.000Z"),
      endTime: new Date("2026-06-21T12:05:01.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(
      container.querySelector('[data-testid="tool-call-rendered-result"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("已执行技能 analysis");
    expect(container.textContent).not.toContain("permissionBehavior");
    expect(container.textContent).not.toContain(
      "workspaceSkillRuntimeEnableAttached",
    );
    expect(container.textContent).not.toContain("sourceMetadata");
    expect(container.textContent).not.toContain("skill-source-session");
  });

  it("站点能力工具结果应展示自动保存结果与脚本来源", () => {
    const { container } = renderTool({
      id: "tool-site-run-1",
      name: "lime_site_run",
      arguments: JSON.stringify({
        adapter_name: "github/search",
        args: { query: "mcp" },
      }),
      status: "completed",
      result: {
        success: true,
        output: JSON.stringify({
          ok: true,
          adapter: "github/search",
          data: { items: [{ title: "modelcontextprotocol/servers" }] },
        }),
        metadata: {
          tool_family: "site",
          adapter_name: "github/search",
          saved_content: {
            content_id: "content-1",
            project_id: "project-1",
            title: "GitHub MCP 搜索结果",
            project_root_path: "/Users/coso/.proxycast/projects/project-1",
            markdown_relative_path:
              "exports/x-article-export/github-mcp/index.md",
            images_relative_dir: "exports/x-article-export/github-mcp/images",
            image_count: 7,
          },
          saved_project_id: "project-1",
          saved_by: "context_project",
          adapter_source_kind: "server_synced",
          adapter_source_version: "2026-03-25",
        },
      },
      startTime: new Date("2026-03-25T12:10:00.000Z"),
      endTime: new Date("2026-03-25T12:10:01.000Z"),
    });

    expect(container.textContent).toContain("已执行 github/search");

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain(
      "结果已自动保存到当前项目：GitHub MCP 搜索结果",
    );
    expect(container.textContent).toContain("已导出 Markdown 文稿");
    expect(container.textContent).toContain("附带图片 7 张");
    expect(container.textContent).not.toContain(
      "exports/x-article-export/github-mcp/index.md",
    );
    expect(container.textContent).not.toContain(
      "exports/x-article-export/github-mcp/images",
    );
    expect(container.textContent).not.toContain("项目目录：");
    expect(container.textContent).not.toContain("脚本来源：");
  });

  it("站点能力工具结果应支持直接打开已保存内容", () => {
    const onOpenSavedSiteContent = vi.fn();
    const { container } = renderTool(
      {
        id: "tool-site-run-open-1",
        name: "lime_site_run",
        arguments: JSON.stringify({
          adapter_name: "github/search",
          args: { query: "lime" },
        }),
        status: "completed",
        result: {
          success: true,
          output: "ok",
          metadata: {
            tool_family: "site",
            saved_content: {
              content_id: "content-open-1",
              project_id: "project-open-1",
              title: "Lime 搜索结果",
            },
            saved_by: "context_project",
          },
        },
        startTime: new Date("2026-03-25T12:20:00.000Z"),
        endTime: new Date("2026-03-25T12:20:01.000Z"),
      },
      { onOpenSavedSiteContent },
    );

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    act(() => {
      const openButton = Array.from(container.querySelectorAll("button")).find(
        (button) => button.textContent?.includes("打开已保存内容"),
      ) as HTMLButtonElement | undefined;
      openButton?.click();
    });

    expect(onOpenSavedSiteContent).toHaveBeenCalledWith({
      projectId: "project-open-1",
      contentId: "content-open-1",
      title: "Lime 搜索结果",
    });
  });

  it("站点能力工具存在导出 Markdown 时应优先打开项目文件目标", () => {
    const onOpenSavedSiteContent = vi.fn();
    const { container } = renderTool(
      {
        id: "tool-site-run-open-markdown-1",
        name: "lime_site_run",
        arguments: JSON.stringify({
          adapter_name: "x/article",
          args: { url: "https://x.com/google/article/1" },
        }),
        status: "completed",
        result: {
          success: true,
          output: "ok",
          metadata: {
            tool_family: "site",
            saved_content: {
              content_id: "content-open-markdown-1",
              project_id: "project-open-markdown-1",
              title: "Google Cloud 周报",
              markdown_relative_path:
                "exports/social-article/google-cloud/index.md",
            },
            saved_by: "context_project",
          },
        },
        startTime: new Date("2026-03-25T12:22:00.000Z"),
        endTime: new Date("2026-03-25T12:22:01.000Z"),
      },
      { onOpenSavedSiteContent },
    );

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    act(() => {
      const openButton = Array.from(container.querySelectorAll("button")).find(
        (button) => button.textContent?.includes("在下方预览导出 Markdown"),
      ) as HTMLButtonElement | undefined;
      openButton?.click();
    });

    expect(onOpenSavedSiteContent).toHaveBeenCalledWith({
      projectId: "project-open-markdown-1",
      contentId: "content-open-markdown-1",
      title: "Google Cloud 周报",
      preferredTarget: "project_file",
      projectFile: {
        relativePath: "exports/social-article/google-cloud/index.md",
      },
    });
  });

  it("工具结果图片预览浮层应使用浅色主题遮罩", () => {
    const { container } = renderTool({
      id: "tool-image-preview-1",
      name: "lime_image_search",
      arguments: JSON.stringify({ query: "青绿色海报" }),
      status: "completed",
      result: {
        success: true,
        output: "已返回 1 张图片",
        images: [
          {
            src: "https://example.com/poster.png",
            mimeType: "image/png",
            origin: "tool_payload",
          },
        ],
      },
      startTime: new Date("2026-04-11T03:00:00.000Z"),
      endTime: new Date("2026-04-11T03:00:01.000Z"),
    });

    const previewButton = container.querySelector(
      'button[title="点击查看大图"]',
    ) as HTMLButtonElement | null;
    expect(previewButton).not.toBeNull();

    act(() => {
      previewButton?.click();
    });

    const overlayButton = Array.from(
      document.body.querySelectorAll("button.fixed"),
    ).find((button) => button.querySelector('img[alt="工具结果图片"]')) as
      | HTMLButtonElement
      | undefined;

    expect(overlayButton).toBeTruthy();
    expect(overlayButton?.className).toContain("backdrop-blur-[2px]");
    expect(overlayButton?.className).not.toContain("bg-black/70");
    expect(
      document.body
        .querySelector('img[alt="工具结果图片"]')
        ?.getAttribute("src"),
    ).toBe("https://example.com/poster.png");
  });

  it("图片生成任务失败结果面板不应展示内部协议错误", () => {
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
    expect(container.textContent).not.toContain(
      "A comic book style illustration",
    );

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("生成失败");
    expect(container.textContent).not.toContain("-32603");
    expect(container.textContent).not.toContain("-32002");
    expect(container.textContent).not.toContain(
      "lime_create_image_generation_task",
    );
  });

  it("内容工作台任务失败结果面板不应展示内部协议错误", () => {
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
    expect(container.textContent).not.toContain("video_generate");

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("视频生成失败");
    expect(container.textContent).not.toContain("-32002");
    expect(container.textContent).not.toContain("video_generate");
  });

  it("站点能力工具失败时应展示未保存原因", () => {
    const { container } = renderTool({
      id: "tool-site-run-2",
      name: "lime_site_run",
      arguments: JSON.stringify({
        adapter_name: "zhihu/search",
        args: { query: "lime" },
      }),
      status: "failed",
      result: {
        success: false,
        error: "执行失败",
        output: "",
        metadata: {
          tool_family: "site",
          adapter_name: "zhihu/search",
          save_skipped_project_id: "project-2",
          save_skipped_by: "context_project",
          save_error_message: "数据库写入失败",
        },
      },
      startTime: new Date("2026-03-25T12:12:00.000Z"),
      endTime: new Date("2026-03-25T12:12:03.000Z"),
    });

    act(() => {
      const toggle = container.querySelector(
        'button[title="查看结果"]',
      ) as HTMLButtonElement | null;
      toggle?.click();
    });

    expect(container.textContent).toContain("执行失败，未保存到当前项目");
    expect(container.textContent).toContain("自动保存失败：数据库写入失败");
  });
});
