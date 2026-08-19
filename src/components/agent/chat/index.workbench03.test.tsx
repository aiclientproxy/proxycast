import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  createMockAgentChatUnifiedState,
  flushEffects,
  getIndexTestMocks,
  mountPage,
  observedWorkspaceIds,
  renderPage,
} from "./index.testFixtures";
import * as fileBrowserModule from "@/lib/api/fileBrowser";

const {
  mockCanvasWorkbenchLayout,
  mockCanvasWorkbenchLayoutState,
  mockMessageList,
  mockUseAgentChatUnified,
} = getIndexTestMocks();

describe("AgentChatPage 通用工作台", { timeout: 20_000 }, () => {
  it("历史旧导出结果与同会话新到的 saved_content 都不应自动打开画布，应等待用户手动点开", async () => {
    mockCanvasWorkbenchLayoutState.renderPreviewProbe = true;
    vi.spyOn(fileBrowserModule, "readFilePreview").mockResolvedValue({
      path: "/tmp/project-site-export/exports/x-article-export/latest/index.md",
      content: "# 最新导出\n\n![封面](images/cover.png)",
      isBinary: false,
      size: 42,
      error: null,
    });

    let messages = [
      {
        id: "msg-site-user-1",
        role: "user" as const,
        content: "帮我导出这篇 X 长文",
        timestamp: new Date("2026-04-08T10:00:00.000Z"),
      },
      {
        id: "msg-site-assistant-old",
        role: "assistant" as const,
        content: "历史导出已完成",
        timestamp: new Date("2026-04-08T10:00:01.000Z"),
        toolCalls: [
          {
            id: "tool-site-old",
            name: "site_run_adapter",
            status: "completed" as const,
            startTime: new Date("2026-04-08T10:00:01.100Z"),
            endTime: new Date("2026-04-08T10:00:02.000Z"),
            result: {
              success: true,
              output: "ok",
              metadata: {
                tool_family: "site",
                saved_content: {
                  content_id: "content-site-export",
                  project_id: "project-site-export",
                  markdown_relative_path:
                    "exports/x-article-export/history/index.md",
                },
              },
            },
          },
        ],
      },
    ];

    mockUseAgentChatUnified.mockImplementation(
      ({ workspaceId }: { workspaceId: string }) => {
        observedWorkspaceIds.push(workspaceId);
        return createMockAgentChatUnifiedState({
          messages,
          sessionId: "session-site-export",
        });
      },
    );

    const mounted = mountPage({
      projectId: "project-site-export",
      contentId: "content-site-export",
      theme: "general",
      lockTheme: true,
    });
    await flushEffects(10);

    expect(fileBrowserModule.readFilePreview).not.toHaveBeenCalled();
    expect(
      mounted.container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat");

    messages = [
      ...messages,
      {
        id: "msg-site-assistant-new",
        role: "assistant" as const,
        content: "最新导出已完成",
        timestamp: new Date("2099-04-09T12:00:01.000Z"),
        toolCalls: [
          {
            id: "tool-site-new",
            name: "site_run_adapter",
            status: "completed" as const,
            startTime: new Date("2099-04-09T12:00:01.100Z"),
            endTime: new Date("2099-04-09T12:00:02.000Z"),
            result: {
              success: true,
              output: "ok",
              metadata: {
                tool_family: "site",
                saved_content: {
                  content_id: "content-site-export",
                  project_id: "project-site-export",
                  markdown_relative_path:
                    "exports/x-article-export/latest/index.md",
                },
              },
            },
          },
        ],
      },
    ];

    mounted.rerender({});
    await flushEffects(12);

    expect(fileBrowserModule.readFilePreview).not.toHaveBeenCalled();
    expect(
      mounted.container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat");
  });

  it("历史任务携带 initialProjectFileOpenTarget 时应直接恢复真实 Markdown 文件预览", async () => {
    mockCanvasWorkbenchLayoutState.renderPreviewProbe = true;
    vi.spyOn(fileBrowserModule, "readFilePreview").mockResolvedValue({
      path: "/tmp/project-history-export/exports/x-article-export/history/index.md",
      content: "# 历史导出\n\n![插图](images/history-cover.png)",
      isBinary: false,
      size: 52,
      error: null,
    });

    const container = renderPage({
      projectId: "project-history-export",
      contentId: "content-history-export",
      theme: "general",
      lockTheme: true,
      initialProjectFileOpenTarget: {
        relativePath: "exports/x-article-export/history/index.md",
        requestKey: 20260409,
      },
    });
    await flushEffects(12);

    expect(fileBrowserModule.readFilePreview).toHaveBeenCalledTimes(1);
    expect(fileBrowserModule.readFilePreview).toHaveBeenCalledWith(
      "/tmp/project-history-export/exports/x-article-export/history/index.md",
      64 * 1024,
    );
    expect(
      container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat-canvas");

    const generalCanvas = container.querySelector(
      '[data-testid="general-canvas"]',
    ) as HTMLDivElement | null;
    expect(generalCanvas).not.toBeNull();
    expect(generalCanvas?.dataset.filename).toBe(
      "exports/x-article-export/history/index.md",
    );
    expect(generalCanvas?.dataset.baseFilePath).toBe(
      "/tmp/project-history-export/exports/x-article-export/history/index.md",
    );
    expect(generalCanvas?.dataset.contentType).toBe("markdown");
    expect(generalCanvas?.dataset.content || "").toContain(
      "![插图](images/history-cover.png)",
    );
    expect(
      container.querySelector('[data-testid="artifact-renderer"]'),
    ).toBeNull();
  });

  it("同项目内打开 saved site content 时应直接恢复真实 Markdown 文件预览", async () => {
    mockCanvasWorkbenchLayoutState.renderPreviewProbe = true;
    vi.spyOn(fileBrowserModule, "readFilePreview").mockResolvedValue({
      path: "/tmp/project-inline-export/exports/x-article-export/latest/index.md",
      content: "# 当前导出\n\n![封面](images/cover.png)",
      isBinary: false,
      size: 43,
      error: null,
    });

    const onNavigate = vi.fn();
    const container = renderPage({
      projectId: "project-inline-export",
      theme: "general",
      lockTheme: true,
      onNavigate,
    });
    await flushEffects(10);

    const latestMessageListProps = mockMessageList.mock.calls.at(-1)?.[0] as
      | {
          onOpenSavedSiteContent?: (target: {
            projectId: string;
            contentId: string;
            preferredTarget: "project_file" | "content";
            projectFile?: {
              relativePath?: string | null;
            } | null;
          }) => void | Promise<void>;
        }
      | undefined;

    await act(async () => {
      await latestMessageListProps?.onOpenSavedSiteContent?.({
        projectId: "project-inline-export",
        contentId: "content-inline-export",
        preferredTarget: "project_file",
        projectFile: {
          relativePath: "exports/x-article-export/latest/index.md",
        },
      });
    });
    await flushEffects(12);

    expect(onNavigate).not.toHaveBeenCalled();
    expect(fileBrowserModule.readFilePreview).toHaveBeenCalledWith(
      "/tmp/project-inline-export/exports/x-article-export/latest/index.md",
      64 * 1024,
    );
    expect(
      container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat-canvas");

    const workbench = container.querySelector(
      '[data-testid="canvas-workbench-layout-mock"]',
    ) as HTMLDivElement | null;
    expect(workbench).not.toBeNull();
    expect(workbench?.dataset.defaultPreviewFilePath).toBe(
      "exports/x-article-export/latest/index.md",
    );
    expect(workbench?.dataset.defaultPreviewAbsolutePath).toBe(
      "/tmp/project-inline-export/exports/x-article-export/latest/index.md",
    );
    expect(workbench?.dataset.defaultPreviewContentType).toBe("markdown");
    expect(workbench?.dataset.defaultPreviewContent || "").toContain(
      "![封面](images/cover.png)",
    );
  });

  it("工具结果只提供文件路径时应读取真实文件预览再打开通用画布", async () => {
    mockCanvasWorkbenchLayoutState.renderPreviewProbe = true;
    vi.spyOn(fileBrowserModule, "readFilePreview").mockResolvedValue({
      path: "/tmp/project-tool-file/src/components/App.tsx",
      content: "export function App() {\n  return <main>Lime</main>;\n}\n",
      isBinary: false,
      size: 56,
      error: null,
    });

    const container = renderPage({
      projectId: "project-tool-file",
      theme: "general",
      lockTheme: true,
    });
    await flushEffects(10);

    const latestMessageListProps = mockMessageList.mock.calls.at(-1)?.[0] as
      | {
          onFileClick?: (fileName: string, content: string) => void;
        }
      | undefined;

    act(() => {
      latestMessageListProps?.onFileClick?.("src/components/App.tsx", "");
    });
    await flushEffects(12);

    expect(fileBrowserModule.readFilePreview).toHaveBeenCalledWith(
      "/tmp/project-tool-file/src/components/App.tsx",
      64 * 1024,
    );
    expect(
      container
        .querySelector('[data-testid="layout-transition"]')
        ?.getAttribute("data-mode"),
    ).toBe("chat-canvas");

    const workbench = container.querySelector(
      '[data-testid="canvas-workbench-layout-mock"]',
    ) as HTMLDivElement | null;
    expect(workbench).not.toBeNull();
    expect(workbench?.dataset.defaultPreviewFilePath).toBe(
      "src/components/App.tsx",
    );
    expect(workbench?.dataset.defaultPreviewAbsolutePath).toBe(
      "/tmp/project-tool-file/src/components/App.tsx",
    );
    expect(workbench?.dataset.defaultPreviewContentType).toBe("code");
    expect(workbench?.dataset.defaultPreviewContent || "").toContain(
      "return <main>Lime</main>;",
    );
  });

  it("时间线文件预览不应以无 selection 的旧请求覆盖目标 artifact", async () => {
    vi.spyOn(fileBrowserModule, "readFilePreview").mockResolvedValue({
      path: "/tmp/project-imported-file/docs/imported-preview.html",
      content:
        "<!doctype html><html><body>导入会话 HTML 预览内容</body></html>",
      isBinary: false,
      size: 72,
      error: null,
    });

    renderPage({
      projectId: "project-imported-file",
      theme: "general",
      lockTheme: true,
    });
    await flushEffects(10);

    const latestMessageListProps = mockMessageList.mock.calls.at(-1)?.[0] as
      | {
          onOpenArtifactFromTimeline?: (target: {
            filePath: string;
            content: string;
            timelineItemId: string;
            openMode: "file_preview";
          }) => void;
        }
      | undefined;

    act(() => {
      latestMessageListProps?.onOpenArtifactFromTimeline?.({
        filePath: "docs/imported-preview.html",
        content: "",
        timelineItemId: "tool-read-imported-preview-html",
        openMode: "file_preview",
      });
    });
    await flushEffects(12);

    expect(fileBrowserModule.readFilePreview).toHaveBeenCalledWith(
      "/tmp/project-imported-file/docs/imported-preview.html",
      64 * 1024,
    );
    const workbenchProps = mockCanvasWorkbenchLayout.mock.calls.at(-1)?.[0] as
      | {
          previewOpenRequest?: {
            filePath?: string | null;
            selectionKey?: string | null;
          } | null;
        }
      | undefined;
    expect(workbenchProps?.previewOpenRequest).toMatchObject({
      filePath: "docs/imported-preview.html",
      selectionKey: expect.stringMatching(/^artifact:preview-file-/),
    });
  });

});
