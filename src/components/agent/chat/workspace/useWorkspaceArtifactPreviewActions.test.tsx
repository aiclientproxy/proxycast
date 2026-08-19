import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Artifact } from "@/lib/artifact/types";
import * as fileBrowserModule from "@/lib/api/fileBrowser";
import { useWorkspaceArtifactPreviewActions } from "./useWorkspaceArtifactPreviewActions";

const { mockHasArtifactPreviewScope, mockReadArtifactPreviewContent } =
  vi.hoisted(() => ({
    mockHasArtifactPreviewScope: vi.fn(),
    mockReadArtifactPreviewContent: vi.fn(),
  }));

vi.mock("@/lib/api/agentRuntime/appServerArtifactClient", () => ({
  hasAgentRuntimeArtifactPreviewScope: mockHasArtifactPreviewScope,
  readAgentRuntimeArtifactPreviewContent: mockReadArtifactPreviewContent,
}));

vi.mock("../hooks/useArtifactAutoPreviewSync", () => ({
  useArtifactAutoPreviewSync: vi.fn(),
}));

vi.mock("sonner", () => ({
  toast: {
    info: vi.fn(),
    error: vi.fn(),
  },
}));

type HookProps = Parameters<typeof useWorkspaceArtifactPreviewActions>[0];

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

function createArtifact(overrides: Partial<Artifact> = {}): Artifact {
  const content = overrides.content ?? "# 研究简报";

  return {
    id: overrides.id ?? "artifact-doc-1",
    type: overrides.type ?? "document",
    title: overrides.title ?? "report.md",
    content,
    status: overrides.status ?? "complete",
    meta: {
      filePath: overrides.meta?.filePath ?? "report.md",
      filename: overrides.meta?.filename ?? "report.md",
      ...overrides.meta,
    },
    position: overrides.position ?? { start: 0, end: content.length },
    createdAt: overrides.createdAt ?? 1,
    updatedAt: overrides.updatedAt ?? 1,
    error: overrides.error,
  };
}

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<
    typeof useWorkspaceArtifactPreviewActions
  > | null = null;

  const defaultProps: HookProps = {
    activeTheme: "general",
    mappedTheme: "general",
    layoutMode: "chat-canvas",
    isThemeWorkbench: false,
    isGeneralCanvasOpen: true,
    artifacts: [],
    currentCanvasArtifact: null,
    taskFiles: [],
    sessionFiles: [],
    readSessionFile: vi.fn(async () => null),
    upsertGeneralArtifact: vi.fn(),
    setSelectedArtifactId: vi.fn(),
    setArtifactViewMode: vi.fn(),
    setLayoutMode: vi.fn(),
    setTaskFiles: vi.fn(),
    setSelectedFileId: vi.fn(),
    setGeneralCanvasState: vi.fn(),
    setCanvasState: vi.fn(),
  };

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceArtifactPreviewActions(currentProps);
    return null;
  }

  const render = async (nextProps?: Partial<HookProps>) => {
    await act(async () => {
      root.render(<Probe {...defaultProps} {...props} {...nextProps} />);
      await Promise.resolve();
    });
  };

  mountedRoots.push({ container, root });

  return {
    render,
    getValue: () => {
      if (!latestValue) {
        throw new Error("hook 尚未初始化");
      }
      return latestValue;
    },
    defaultProps: { ...defaultProps, ...props },
  };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  mockHasArtifactPreviewScope.mockReset();
  mockHasArtifactPreviewScope.mockReturnValue(false);
  mockReadArtifactPreviewContent.mockReset();
});

async function flushAsyncWork(times = 4) {
  for (let index = 0; index < times; index += 1) {
    await Promise.resolve();
  }
}

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  vi.restoreAllMocks();
});

describe("useWorkspaceArtifactPreviewActions", () => {
  it("打开普通 artifact 时应进入工作台预览", async () => {
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const setGeneralCanvasState = vi.fn();
    const artifact = createArtifact();
    const { render, getValue } = renderHook({
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
      setGeneralCanvasState,
    });

    await render();

    act(() => {
      getValue().handleArtifactClick(artifact);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(setSelectedArtifactId).toHaveBeenCalledWith("artifact-doc-1");
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: "artifact-doc-1",
    });
    expect(setGeneralCanvasState).toHaveBeenCalledTimes(1);
  });

  it("文章编辑器 详情页里的打开预览按钮仍应进入 workbench", async () => {
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const artifact = createArtifact({
      id: "preview-article-workspace-2",
      title: "文章编辑器",
      content: "# 文章编辑器",
      meta: {
        openedFrom: "general-workbench-file",
      },
    });
    const { render, getValue } = renderHook({
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
    });

    await render();

    await act(async () => {
      await getValue().openArtifactInWorkbench(artifact);
    });

    expect(setSelectedArtifactId).toHaveBeenCalledWith("preview-article-workspace-2");
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: "preview-article-workspace-2",
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
  });

  it("通用模式打开文件预览时应投影为 source-backed preview artifact", async () => {
    const upsertGeneralArtifact = vi.fn();
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const { render, getValue } = renderHook({
      upsertGeneralArtifact,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
    });

    await render();

    act(() => {
      getValue().handleFileClick(
        ".lime/artifacts/thread-1/report.md",
        "# 研究简报\n\n这里是预览内容。",
      );
    });

    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact).toEqual(
      expect.objectContaining({
        id: expect.stringMatching(/^preview-file-/),
        type: "document",
        title: "report.md",
        content: "# 研究简报\n\n这里是预览内容。",
        meta: expect.objectContaining({
          previewArtifact: true,
          isSourceBacked: true,
          source: "file",
          sourceRef: ".lime/artifacts/thread-1/report.md",
          sourcePath: ".lime/artifacts/thread-1/report.md",
          filePath: ".lime/artifacts/thread-1/report.md",
          filename: "report.md",
          contentKind: "markdown",
          renderMode: "canvas",
          lifecycle: "transient",
        }),
      }),
    );
    expect(setSelectedArtifactId).toHaveBeenCalledWith(artifact.id);
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath: ".lime/artifacts/thread-1/report.md",
      selectionKey: `artifact:${artifact.id}`,
    });
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: artifact.id,
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
    expect(setGeneralCanvasState).toHaveBeenCalledWith(expect.any(Function));
  });

  it("非通用模式打开带文本的 PDF 文件时应作为文档切到工作台预览", async () => {
    const setTaskFiles = vi.fn();
    const setSelectedFileId = vi.fn();
    const setCanvasState = vi.fn();
    const setLayoutMode = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const { render, getValue } = renderHook({
      activeTheme: "coding",
      isThemeWorkbench: false,
      setTaskFiles,
      setSelectedFileId,
      setCanvasState,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
    });
    const filePath =
      "/tmp/imported-local-history/docs/imported-preview.pdf";
    const content =
      "导入会话 PDF 预览内容：可解析文本流应进入文档预览。";

    await render();

    act(() => {
      getValue().handleFileClick(filePath, content);
    });

    expect(setSelectedFileId).toHaveBeenCalledTimes(1);
    const selectedFileId = setSelectedFileId.mock.calls[0]?.[0] as string;
    expect(selectedFileId).toEqual(expect.any(String));
    expect(setTaskFiles).toHaveBeenCalledTimes(1);
    const updater = setTaskFiles.mock.calls[0]?.[0] as (
      files: Array<{
        id: string;
        name: string;
        type: string;
        content?: string;
      }>,
    ) => unknown;
    expect(updater([])).toEqual([
      expect.objectContaining({
        id: selectedFileId,
        name: filePath,
        type: "document",
        content,
      }),
    ]);
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath,
      selectionKey: `task:${selectedFileId}`,
    });
    expect(setCanvasState).toHaveBeenCalledWith(expect.any(Function));
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
  });

  it("通用模式打开媒体 preview artifact 时不应再按空文件懒加载", async () => {
    const readSessionFile = vi.fn(async () => "");
    const upsertGeneralArtifact = vi.fn();
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const artifact = createArtifact({
      id: "preview-session_file-message-attachment",
      title: "attachment-1",
      content: "data:image/png;base64,aGVsbG8=",
      meta: {
        previewArtifact: true,
        isSourceBacked: true,
        source: "session_file",
        sourceRef: "message-1:attachment:0",
        sourcePath: "message-1:attachment:0",
        filePath: "message-1:attachment:0",
        filename: "attachment-1",
        contentKind: "image",
        renderMode: "media",
        previewUrl: "data:image/png;base64,aGVsbG8=",
      },
    });
    const { render, getValue } = renderHook({
      readSessionFile,
      upsertGeneralArtifact,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
      isGeneralCanvasOpen: true,
    });

    await render();

    act(() => {
      getValue().handleArtifactClick(artifact);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(readSessionFile).not.toHaveBeenCalled();
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    expect(upsertGeneralArtifact).toHaveBeenCalledWith(
      expect.objectContaining({
        id: artifact.id,
        meta: expect.objectContaining({
          previewArtifact: true,
          contentKind: "image",
          renderMode: "media",
          previewUrl: "data:image/png;base64,aGVsbG8=",
        }),
      }),
    );
    expect(setGeneralCanvasState).toHaveBeenCalledWith(expect.any(Function));
    const resetGeneralCanvas = setGeneralCanvasState.mock
      .calls[0]?.[0] as (state: {
      isOpen: boolean;
      contentType: string;
      content: string;
      filename?: string;
      sourcePath?: string;
      isEditing: boolean;
    }) => unknown;
    expect(
      resetGeneralCanvas({
        isOpen: true,
        contentType: "markdown",
        content: "# 上轮对话",
        filename: "上轮对话.md",
        sourcePath: "/tmp/last-turn.md",
        isEditing: true,
      }),
    ).toEqual({
      isOpen: false,
      contentType: "empty",
      content: "",
      filename: undefined,
      sourcePath: undefined,
      isEditing: false,
    });
    expect(setSelectedArtifactId).toHaveBeenCalledWith(artifact.id);
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: artifact.id,
    });
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath: null,
      selectionKey: `artifact:${artifact.id}`,
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
  });

  it("通用模式打开来源摘要 preview artifact 时不应把 URL 当文件懒加载", async () => {
    const readFilePreviewSpy = vi.spyOn(fileBrowserModule, "readFilePreview");
    const upsertGeneralArtifact = vi.fn();
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const artifact = createArtifact({
      id: "preview-url-example",
      title: "在线报告",
      content: "",
      meta: {
        previewArtifact: true,
        isSourceBacked: true,
        source: "url",
        sourceRef: "https://example.com/report",
        sourcePath: "https://example.com/report",
        filePath: "https://example.com/report",
        filename: "report",
        contentKind: "markdown",
        renderMode: "inline",
      },
    });
    const { render, getValue } = renderHook({
      upsertGeneralArtifact,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
      isGeneralCanvasOpen: true,
    });

    await render();

    act(() => {
      getValue().handleArtifactClick(artifact);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(readFilePreviewSpy).not.toHaveBeenCalled();
    expect(upsertGeneralArtifact).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "preview-url-example",
        meta: expect.objectContaining({
          source: "url",
          renderMode: "inline",
        }),
      }),
    );
    expect(setSelectedArtifactId).toHaveBeenCalledWith("preview-url-example");
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath: null,
      selectionKey: "artifact:preview-url-example",
    });
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: "preview-url-example",
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
  });

  it("通用模式打开 LayeredDesignDocument 工程文件应进入 canvas:design 主链", async () => {
    const upsertGeneralArtifact = vi.fn();
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const designJson = JSON.stringify({
      id: "main-app-layered-design",
      title: "主应用图层设计",
      canvas: { width: 1080, height: 1440 },
      layers: [
        {
          id: "headline",
          name: "标题层",
          type: "text",
          text: "真实主应用入口",
          x: 96,
          y: 120,
          width: 888,
          height: 120,
          zIndex: 10,
        },
      ],
      assets: [],
      editHistory: [],
      createdAt: "2026-05-08T00:00:00.000Z",
      updatedAt: "2026-05-08T00:00:00.000Z",
    });
    const { render, getValue } = renderHook({
      upsertGeneralArtifact,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
    });

    await render();

    act(() => {
      getValue().handleFileClick(
        ".lime/layered-designs/main-app.layered-design/design.json",
        designJson,
      );
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact).toEqual(
      expect.objectContaining({
        type: "canvas:design",
        title: "主应用图层设计",
        content: expect.stringContaining('"id": "main-app-layered-design"'),
        meta: expect.objectContaining({
          filePath: ".lime/layered-designs/main-app.layered-design/design.json",
          filename: "design.json",
          platform: "layered-design",
          designId: "main-app-layered-design",
          openedFrom: "general-workbench-file",
        }),
      }),
    );
    expect(setSelectedArtifactId).toHaveBeenCalledWith(artifact.id);
    expect(setSelectedArtifactId).not.toHaveBeenCalledWith(null);
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: artifact.id,
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
    expect(setGeneralCanvasState).toHaveBeenCalledTimes(1);
  });

  it("非通用模式点击文件时应显式请求工作台切到新任务文件预览", async () => {
    const randomUUIDSpy = vi
      .spyOn(crypto, "randomUUID")
      .mockReturnValue("11111111-1111-4111-8111-111111111111");
    const setTaskFiles = vi.fn();
    const setSelectedFileId = vi.fn();
    const setCanvasState = vi.fn();
    const setLayoutMode = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const { render, getValue } = renderHook({
      activeTheme: "article",
      mappedTheme: "general",
      isThemeWorkbench: true,
      setTaskFiles,
      setSelectedFileId,
      setCanvasState,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
    });

    await render();

    act(() => {
      getValue().handleFileClick(
        "imported-preview.md",
        "# 导入会话 Markdown 预览内容",
      );
    });

    expect(randomUUIDSpy).toHaveBeenCalledTimes(1);
    expect(setSelectedFileId).toHaveBeenCalledWith(
      "11111111-1111-4111-8111-111111111111",
    );
    expect(setTaskFiles).toHaveBeenCalledTimes(1);
    const updateTaskFiles = setTaskFiles.mock.calls[0]?.[0];
    expect(typeof updateTaskFiles).toBe("function");
    expect(updateTaskFiles([])).toEqual([
      expect.objectContaining({
        id: "11111111-1111-4111-8111-111111111111",
        name: "imported-preview.md",
        type: "document",
        content: "# 导入会话 Markdown 预览内容",
      }),
    ]);
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath: "imported-preview.md",
      selectionKey: "task:11111111-1111-4111-8111-111111111111",
    });
    expect(setCanvasState).toHaveBeenCalledWith(expect.any(Function));
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
  });

  it("读取带 App Server scope 的 artifact 时应走 artifact/read current 主链", async () => {
    const readFilePreviewSpy = vi.spyOn(fileBrowserModule, "readFilePreview");
    mockHasArtifactPreviewScope.mockReturnValue(true);
    mockReadArtifactPreviewContent.mockResolvedValueOnce({
      artifactRef: "artifact-report",
      artifactId: "artifact-report",
      filePath: ".app-server/artifacts/report.md",
      content: "# App Server 正文",
    });
    const artifact = createArtifact({
      id: "artifact-report",
      content: "",
      meta: {
        filePath: ".app-server/artifacts/report.md",
        filename: "report.md",
        sessionId: "session-1",
        turnId: "turn-1",
        artifactRef: "artifact-report",
      },
    });
    const { render, getValue } = renderHook();

    await render();

    const preview = await getValue().handleHarnessLoadFilePreview(
      ".app-server/artifacts/report.md",
      artifact,
    );

    expect(mockHasArtifactPreviewScope).toHaveBeenCalledWith(
      artifact,
      ".app-server/artifacts/report.md",
    );
    expect(mockReadArtifactPreviewContent).toHaveBeenCalledWith(
      artifact,
      ".app-server/artifacts/report.md",
    );
    expect(readFilePreviewSpy).not.toHaveBeenCalled();
    expect(preview).toEqual({
      artifactId: "artifact-report",
      artifactRef: "artifact-report",
      path: ".app-server/artifacts/report.md",
      content: "# App Server 正文",
      isBinary: false,
      size: "# App Server 正文".length,
      error: null,
    });
  });

  it("带 App Server scope 的 artifact 内容不可用时不应回退旧文件预览", async () => {
    const readFilePreviewSpy = vi.spyOn(fileBrowserModule, "readFilePreview");
    mockHasArtifactPreviewScope.mockReturnValue(true);
    mockReadArtifactPreviewContent.mockResolvedValueOnce(null);
    const artifact = createArtifact({
      id: "artifact-report",
      content: "",
      meta: {
        filePath: "report.md",
        filename: "report.md",
        sessionId: "session-1",
        artifactRef: "artifact-report",
      },
    });
    const { render, getValue } = renderHook({
      taskFiles: [
        {
          id: "task-report",
          name: "report.md",
          type: "document",
          content: "# 旧任务文件",
          version: 1,
          createdAt: 1,
          updatedAt: 1,
        },
      ],
    });

    await render();

    const preview = await getValue().handleHarnessLoadFilePreview(
      "report.md",
      artifact,
    );

    expect(mockHasArtifactPreviewScope).toHaveBeenCalledWith(
      artifact,
      "report.md",
    );
    expect(mockReadArtifactPreviewContent).toHaveBeenCalledWith(
      artifact,
      "report.md",
    );
    expect(readFilePreviewSpy).not.toHaveBeenCalled();
    expect(preview).toEqual({
      path: "report.md",
      content: null,
      isBinary: false,
      size: 0,
      error: "App Server artifact 内容不可用",
    });
  });

  it("读取带目录的真实结果路径时不应回退到同名裸任务文件", async () => {
    const readFilePreviewSpy = vi
      .spyOn(fileBrowserModule, "readFilePreview")
      .mockResolvedValue({
        path: "/tmp/project/exports/x-article-export/latest/index.md",
        content: "# 真实导出",
        isBinary: false,
        size: 12,
        error: null,
      });

    const { render, getValue } = renderHook({
      taskFiles: [
        {
          id: "task-index",
          name: "index.md",
          type: "document",
          content: "# 过程文件",
          version: 1,
          createdAt: 1,
          updatedAt: 1,
        },
      ],
    });

    await render();

    const preview = await getValue().handleHarnessLoadFilePreview(
      "/tmp/project/exports/x-article-export/latest/index.md",
    );

    expect(readFilePreviewSpy).toHaveBeenCalledWith(
      "/tmp/project/exports/x-article-export/latest/index.md",
      64 * 1024,
    );
    expect(preview).toEqual({
      path: "/tmp/project/exports/x-article-export/latest/index.md",
      content: "# 真实导出",
      isBinary: false,
      size: 12,
      error: null,
    });
  });

  it("读取裸文件名时仍可回退到同名任务文件", async () => {
    const readFilePreviewSpy = vi.spyOn(fileBrowserModule, "readFilePreview");
    const { render, getValue } = renderHook({
      taskFiles: [
        {
          id: "task-index",
          name: "index.md",
          type: "document",
          content: "# 过程文件",
          version: 1,
          createdAt: 1,
          updatedAt: 1,
        },
      ],
    });

    await render();

    const preview = await getValue().handleHarnessLoadFilePreview("index.md");

    expect(readFilePreviewSpy).not.toHaveBeenCalled();
    expect(preview).toEqual({
      path: "index.md",
      content: "# 过程文件",
      isBinary: false,
      size: "# 过程文件".length,
      error: null,
    });
  });

  it("读取裸文件名时可回退到带目录的同名任务文件", async () => {
    const readFilePreviewSpy = vi.spyOn(fileBrowserModule, "readFilePreview");
    const { render, getValue } = renderHook({
      taskFiles: [
        {
          id: "task-export-index",
          name: "exports/x-article-export/latest/index.md",
          type: "document",
          content: "# 正式结果",
          version: 1,
          createdAt: 1,
          updatedAt: 1,
        },
      ],
    });

    await render();

    const preview = await getValue().handleHarnessLoadFilePreview("index.md");

    expect(readFilePreviewSpy).not.toHaveBeenCalled();
    expect(preview).toEqual({
      path: "exports/x-article-export/latest/index.md",
      content: "# 正式结果",
      isBinary: false,
      size: "# 正式结果".length,
      error: null,
    });
  });

  it("点击占位任务文件时应按需读取会话内容并打开 preview artifact", async () => {
    const readSessionFile = vi.fn(async () => "# 会话主稿\n\n按需恢复");
    const setTaskFiles = vi.fn();
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const upsertGeneralArtifact = vi.fn();
    const placeholderFile = {
      id: "session-file:content-posts/draft.md",
      name: "content-posts/draft.md",
      type: "document" as const,
      version: 1,
      createdAt: 100,
      updatedAt: 100,
    };
    const { render, getValue } = renderHook({
      taskFiles: [placeholderFile],
      sessionFiles: [
        {
          name: "content-posts/draft.md",
          fileType: "document",
          metadata: {
            contentPostIntent: "draft",
          },
          size: 20,
          createdAt: 100,
          updatedAt: 200,
        },
      ],
      readSessionFile,
      upsertGeneralArtifact,
      setTaskFiles,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
    });

    await render();

    await act(async () => {
      getValue().handleTaskFileClick(placeholderFile);
      await flushAsyncWork();
    });

    expect(readSessionFile).toHaveBeenCalledWith("content-posts/draft.md");
    expect(setTaskFiles).toHaveBeenCalledTimes(1);

    const taskFilesUpdater = setTaskFiles.mock.calls[0]?.[0];
    expect(typeof taskFilesUpdater).toBe("function");
    expect(taskFilesUpdater([placeholderFile])).toEqual([
      expect.objectContaining({
        id: "session-file:content-posts/draft.md",
        name: "content-posts/draft.md",
        type: "document",
        content: "# 会话主稿\n\n按需恢复",
        metadata: expect.objectContaining({
          contentPostIntent: "draft",
        }),
        createdAt: 100,
        updatedAt: 200,
      }),
    ]);
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const openedArtifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(openedArtifact).toEqual(
      expect.objectContaining({
        id: expect.stringMatching(/^preview-file-/),
        title: "draft.md",
        content: "# 会话主稿\n\n按需恢复",
        meta: expect.objectContaining({
          previewArtifact: true,
          sourceRef: "content-posts/draft.md",
          contentKind: "markdown",
        }),
      }),
    );
    expect(setSelectedArtifactId).toHaveBeenCalledWith(openedArtifact?.id);
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: openedArtifact?.id,
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
    expect(setGeneralCanvasState).toHaveBeenCalledWith(expect.any(Function));
  });

  it("通用模式打开真实 HTML 路径时应读取文件并投影为支持 Desktop Host 预览的 artifact", async () => {
    const readFilePreviewSpy = vi
      .spyOn(fileBrowserModule, "readFilePreview")
      .mockResolvedValue({
        path: "/tmp/project/prototype.html",
        content: "<!doctype html><html><body>Lime</body></html>",
        isBinary: false,
        size: 44,
        error: null,
      });
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const upsertGeneralArtifact = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const { render, getValue } = renderHook({
      upsertGeneralArtifact,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
    });

    await render();

    await act(async () => {
      getValue().handleFileClick("/tmp/project/prototype.html", "");
      await flushAsyncWork();
    });

    expect(readFilePreviewSpy).toHaveBeenCalledWith(
      "/tmp/project/prototype.html",
      64 * 1024,
    );
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact).toEqual(
      expect.objectContaining({
        type: "html",
        title: "prototype.html",
        content: "<!doctype html><html><body>Lime</body></html>",
        meta: expect.objectContaining({
          previewArtifact: true,
          sourcePath: "/tmp/project/prototype.html",
          filePath: "/tmp/project/prototype.html",
          contentKind: "html",
          renderMode: "external_window",
          capabilities: expect.objectContaining({
            externalWindow: true,
          }),
        }),
      }),
    );
    expect(setSelectedArtifactId).toHaveBeenCalledWith(artifact.id);
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath: "/tmp/project/prototype.html",
      selectionKey: `artifact:${artifact.id}`,
    });
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: artifact.id,
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
    expect(setGeneralCanvasState).toHaveBeenCalledWith(expect.any(Function));
  });

  it("通用模式打开真实 DOCX 路径时应读取抽取文本并投影为 document_text artifact", async () => {
    const docxPath =
      "/Users/coso/Documents/other/谢晶_个人IP知识库v1.0_深澜智能.docx";
    const readFilePreviewSpy = vi
      .spyOn(fileBrowserModule, "readFilePreview")
      .mockResolvedValue({
        path: docxPath,
        content: "个人 IP 知识库\n\n深澜智能",
        isBinary: false,
        size: 55007,
        error: null,
      });
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const upsertGeneralArtifact = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const { render, getValue } = renderHook({
      upsertGeneralArtifact,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
    });

    await render();

    await act(async () => {
      getValue().handleFileClick(docxPath, "");
      await flushAsyncWork();
    });

    expect(readFilePreviewSpy).toHaveBeenCalledWith(docxPath, 64 * 1024);
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact).toEqual(
      expect.objectContaining({
        type: "document",
        title: "谢晶_个人IP知识库v1.0_深澜智能.docx",
        content: "个人 IP 知识库\n\n深澜智能",
        meta: expect.objectContaining({
          previewArtifact: true,
          isSourceBacked: true,
          source: "file",
          sourceRef: docxPath,
          sourcePath: docxPath,
          filePath: docxPath,
          filename: "谢晶_个人IP知识库v1.0_深澜智能.docx",
          fileKind: "docx",
          contentKind: "document",
          renderMode: "document_text",
          lifecycle: "transient",
        }),
      }),
    );
    expect(setSelectedArtifactId).toHaveBeenCalledWith(artifact.id);
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath: docxPath,
      selectionKey: `artifact:${artifact.id}`,
    });
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: artifact.id,
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
    expect(setGeneralCanvasState).toHaveBeenCalledWith(expect.any(Function));
  });

  it("通用模式打开可抽取文本的 PDF 路径时应投影为 document_text artifact", async () => {
    const pdfPath = "/Users/coso/Documents/other/汇报材料.pdf";
    const readFilePreviewSpy = vi.spyOn(fileBrowserModule, "readFilePreview").mockResolvedValue({
      path: pdfPath,
      content: "PDF 正文预览\n\n关键结论",
      isBinary: false,
      size: 55007,
      error: null,
    });
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const upsertGeneralArtifact = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const { render, getValue } = renderHook({
      upsertGeneralArtifact,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
    });

    await render();

    await act(async () => {
      getValue().handleFileClick(pdfPath, "");
      await flushAsyncWork();
    });

    expect(readFilePreviewSpy).toHaveBeenCalledWith(pdfPath, 64 * 1024);
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact).toEqual(
      expect.objectContaining({
        type: "document",
        title: "汇报材料.pdf",
        content: "PDF 正文预览\n\n关键结论",
        meta: expect.objectContaining({
          previewArtifact: true,
          isSourceBacked: true,
          source: "file",
          sourceRef: pdfPath,
          sourcePath: pdfPath,
          filePath: pdfPath,
          filename: "汇报材料.pdf",
          fileKind: "pdf",
          contentKind: "document",
          renderMode: "document_text",
          lifecycle: "transient",
          capabilities: expect.objectContaining({
            preview: true,
            systemOpen: true,
          }),
        }),
      }),
    );
    expect(setSelectedArtifactId).toHaveBeenCalledWith(artifact.id);
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath: pdfPath,
      selectionKey: `artifact:${artifact.id}`,
    });
    expect(setArtifactViewMode).toHaveBeenCalledWith("preview", {
      artifactId: artifact.id,
    });
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
    expect(setGeneralCanvasState).toHaveBeenCalledWith(expect.any(Function));
  });

  it("通用模式打开可抽取文本的 Excel 路径时应复用 document_text artifact", async () => {
    const xlsxPath = "/Users/coso/Documents/other/导入验收矩阵.xlsx";
    const readFilePreviewSpy = vi.spyOn(fileBrowserModule, "readFilePreview").mockResolvedValue({
      path: xlsxPath,
      content: "能力\t状态\nWebSearch\tcurrent",
      isBinary: false,
      size: 12048,
      error: null,
    });
    const setGeneralCanvasState = vi.fn();
    const setSelectedArtifactId = vi.fn();
    const setArtifactViewMode = vi.fn();
    const setLayoutMode = vi.fn();
    const upsertGeneralArtifact = vi.fn();
    const onRequestCanvasPreviewOpen = vi.fn();
    const { render, getValue } = renderHook({
      upsertGeneralArtifact,
      setGeneralCanvasState,
      setSelectedArtifactId,
      setArtifactViewMode,
      setLayoutMode,
      onRequestCanvasPreviewOpen,
    });

    await render();

    await act(async () => {
      getValue().handleFileClick(xlsxPath, "");
      await flushAsyncWork();
    });

    expect(readFilePreviewSpy).toHaveBeenCalledWith(xlsxPath, 64 * 1024);
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact).toEqual(
      expect.objectContaining({
        type: "document",
        title: "导入验收矩阵.xlsx",
        content: "能力\t状态\nWebSearch\tcurrent",
        meta: expect.objectContaining({
          previewArtifact: true,
          source: "file",
          sourceRef: xlsxPath,
          filename: "导入验收矩阵.xlsx",
          fileKind: "xlsx",
          contentKind: "document",
          renderMode: "document_text",
        }),
      }),
    );
    expect(onRequestCanvasPreviewOpen).toHaveBeenCalledWith({
      filePath: xlsxPath,
      selectionKey: `artifact:${artifact.id}`,
    });
  });

  it("点击占位任务文件时应按需读取会话内容并更新主题工作台画布", async () => {
    const readSessionFile = vi.fn(async () => "# 主题工作台内容");
    const setTaskFiles = vi.fn();
    const setSelectedFileId = vi.fn();
    const setCanvasState = vi.fn();
    const setLayoutMode = vi.fn();
    const placeholderFile = {
      id: "session-file:result.md",
      name: "result.md",
      type: "document" as const,
      version: 1,
      createdAt: 10,
      updatedAt: 10,
    };
    const { render, getValue } = renderHook({
      activeTheme: "article",
      mappedTheme: "general",
      layoutMode: "chat",
      isThemeWorkbench: true,
      taskFiles: [placeholderFile],
      sessionFiles: [
        {
          name: "result.md",
          fileType: "document",
          size: 20,
          createdAt: 10,
          updatedAt: 30,
        },
      ],
      readSessionFile,
      setTaskFiles,
      setSelectedFileId,
      setCanvasState,
      setLayoutMode,
    });

    await render();

    await act(async () => {
      getValue().handleTaskFileClick(placeholderFile);
      await flushAsyncWork();
    });

    expect(readSessionFile).toHaveBeenCalledWith("result.md");
    expect(setSelectedFileId).toHaveBeenCalledWith("session-file:result.md");
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");

    const taskFilesUpdater = setTaskFiles.mock.calls[0]?.[0];
    expect(typeof taskFilesUpdater).toBe("function");
    expect(taskFilesUpdater([placeholderFile])).toEqual([
      expect.objectContaining({
        id: "session-file:result.md",
        name: "result.md",
        content: "# 主题工作台内容",
        updatedAt: 30,
      }),
    ]);

    const canvasStateUpdater = setCanvasState.mock.calls[0]?.[0];
    expect(typeof canvasStateUpdater).toBe("function");
    expect(
      canvasStateUpdater({
        type: "document",
        content: "旧内容",
        platform: "markdown",
        versions: [],
        currentVersionId: "version-1",
        isEditing: true,
      }),
    ).toEqual(
      expect.objectContaining({
        type: "document",
        content: "# 主题工作台内容",
        platform: "markdown",
        currentVersionId: "version-1",
      }),
    );
  });
});
