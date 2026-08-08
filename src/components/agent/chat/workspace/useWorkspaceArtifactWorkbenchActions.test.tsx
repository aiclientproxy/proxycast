import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ArtifactDocumentV1 } from "@/lib/artifact-document";
import type { Artifact } from "@/lib/artifact/types";
import { useWorkspaceArtifactWorkbenchActions } from "./useWorkspaceArtifactWorkbenchActions";

const toastSuccess = vi.fn();
const toastError = vi.fn();
const mockSaveDialog = vi.fn();
const mockSaveExportedDocument = vi.fn();

vi.mock("sonner", () => ({
  toast: {
    success: (...args: unknown[]) => toastSuccess(...args),
    error: (...args: unknown[]) => toastError(...args),
  },
}));

vi.mock("@/lib/desktop-host/dialog", () => ({
  save: (...args: unknown[]) => mockSaveDialog(...args),
}));

vi.mock("@/lib/api/document-export", () => ({
  saveExportedDocument: (...args: unknown[]) =>
    mockSaveExportedDocument(...args),
}));

type HookProps = Parameters<typeof useWorkspaceArtifactWorkbenchActions>[0];

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

function createArtifact(): Artifact {
  return {
    id: "artifact-1",
    type: "document",
    title: "board-review.artifact.json",
    content: "",
    status: "complete",
    meta: {
      filePath: ".lime/artifacts/thread-1/board-review.artifact.json",
      filename: "board-review.artifact.json",
      language: "json",
    },
    position: { start: 0, end: 0 },
    createdAt: 1,
    updatedAt: 1,
  };
}

function createDocument(
  status: "ready" | "archived" = "ready",
): ArtifactDocumentV1 {
  return {
    schemaVersion: "artifact_document.v1",
    artifactId: "artifact-document:board-review",
    kind: "analysis",
    title: "董事会季度复盘",
    status,
    language: "zh-CN",
    blocks: [
      {
        id: "body-1",
        type: "rich_text",
        contentFormat: "markdown",
        content: "正文内容",
        markdown: "正文内容",
      },
    ],
    sources: [],
    metadata: {
      currentVersionId: "artifact-document:board-review:v2",
      currentVersionNo: 2,
      versionHistory: [
        {
          id: "artifact-document:board-review:v1",
          artifactId: "artifact-document:board-review",
          versionNo: 1,
          title: "董事会季度复盘",
          status: "ready",
        },
        {
          id: "artifact-document:board-review:v2",
          artifactId: "artifact-document:board-review",
          versionNo: 2,
          title: "董事会季度复盘",
          status,
        },
      ],
    },
  };
}

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  let latestValue: ReturnType<
    typeof useWorkspaceArtifactWorkbenchActions
  > | null = null;

  const defaultProps: HookProps = {
    activeTheme: "general",
    projectId: "project-1",
    syncGeneralArtifactToResource: vi.fn().mockResolvedValue({
      status: "uploaded",
    }),
    onSaveArtifactDocument: vi.fn().mockResolvedValue(undefined),
  };

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceArtifactWorkbenchActions(currentProps);
    return null;
  }

  const render = async (nextProps?: Partial<HookProps>) => {
    await act(async () => {
      root.render(<Probe {...defaultProps} {...props} {...nextProps} />);
      await Promise.resolve();
    });
  };

  mountedRoots.push({ root, container });

  return {
    render,
    getValue: () => {
      if (!latestValue) {
        throw new Error("hook 尚未初始化");
      }
      return latestValue;
    },
  };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  toastSuccess.mockReset();
  toastError.mockReset();
  mockSaveDialog.mockReset();
  mockSaveExportedDocument.mockReset();
});

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

describe("useWorkspaceArtifactWorkbenchActions", () => {
  it("项目复用动作应复用现有资源同步主线", async () => {
    const syncGeneralArtifactToResource = vi.fn().mockResolvedValue({
      status: "uploaded",
    });
    const { render, getValue } = renderHook({
      syncGeneralArtifactToResource,
    });
    await render();

    const state = getValue().getToolbarActionState(
      createArtifact(),
      createDocument(),
    );
    expect(state).not.toBeNull();

    await act(async () => {
      await state?.onSaveToProject();
    });

    expect(syncGeneralArtifactToResource).toHaveBeenCalledWith({
      rawFilePath: ".lime/artifacts/thread-1/board-review.artifact.json",
      preferredName: "董事会季度复盘",
    });
    expect(toastSuccess).toHaveBeenCalledWith("已保存到项目资源");
  });

  it("归档动作应回写同一份 ArtifactDocument 状态", async () => {
    const onSaveArtifactDocument = vi.fn().mockResolvedValue(undefined);
    const { render, getValue } = renderHook({
      onSaveArtifactDocument,
    });
    await render();

    const artifact = createArtifact();
    const document = createDocument();
    const state = getValue().getToolbarActionState(artifact, document);
    expect(state?.archiveLabel).toBe("归档");

    await act(async () => {
      await state?.onToggleArchive();
    });

    expect(onSaveArtifactDocument).toHaveBeenCalledWith(
      expect.objectContaining({ id: "artifact-1" }),
      expect.objectContaining({
        status: "archived",
        metadata: expect.objectContaining({
          versionHistory: expect.arrayContaining([
            expect.objectContaining({
              id: "artifact-document:board-review:v2",
              status: "archived",
            }),
          ]),
        }),
      }),
    );
    expect(toastSuccess).toHaveBeenCalledWith("已归档当前交付物");
  });

  it("导出 Markdown 应走统一桌面导出主链", async () => {
    mockSaveDialog.mockResolvedValue("/tmp/board-review");
    mockSaveExportedDocument.mockResolvedValue(undefined);
    const { render, getValue } = renderHook();
    await render();

    const state = getValue().getToolbarActionState(
      createArtifact(),
      createDocument(),
    );

    await act(async () => {
      await state?.onExportMarkdown();
    });

    expect(mockSaveDialog).toHaveBeenCalledWith({
      title: "导出 Markdown",
      defaultPath: "board-review.md",
      filters: [{ name: "Markdown", extensions: ["md"] }],
    });
    expect(mockSaveExportedDocument).toHaveBeenCalledWith(
      "/tmp/board-review.md",
      "# 董事会季度复盘\n\n正文内容",
    );
    expect(toastSuccess).toHaveBeenCalledWith("已导出 Markdown");
  });

  it("导出 HTML 应走统一桌面导出主链", async () => {
    mockSaveDialog.mockResolvedValue("/tmp/board-review");
    mockSaveExportedDocument.mockResolvedValue(undefined);
    const { render, getValue } = renderHook();
    await render();

    const state = getValue().getToolbarActionState(
      createArtifact(),
      createDocument(),
    );

    await act(async () => {
      await state?.onExportHtml();
    });

    expect(mockSaveDialog).toHaveBeenCalledWith({
      title: "导出 HTML",
      defaultPath: "board-review.html",
      filters: [{ name: "HTML", extensions: ["html"] }],
    });
    expect(mockSaveExportedDocument).toHaveBeenCalledWith(
      "/tmp/board-review.html",
      expect.stringContaining("<!doctype html>"),
    );
    expect(mockSaveExportedDocument).toHaveBeenCalledWith(
      "/tmp/board-review.html",
      expect.stringContaining("<title>董事会季度复盘</title>"),
    );
    expect(mockSaveExportedDocument).toHaveBeenCalledWith(
      "/tmp/board-review.html",
      expect.stringContaining("<h1>董事会季度复盘</h1>"),
    );
    expect(mockSaveExportedDocument).toHaveBeenCalledWith(
      "/tmp/board-review.html",
      expect.stringContaining("<p>正文内容</p>"),
    );
    expect(toastSuccess).toHaveBeenCalledWith("已导出 HTML");
  });

  it("导出 Artifact JSON 应保留 .artifact.json 后缀", async () => {
    mockSaveDialog.mockResolvedValue("/tmp/board-review");
    mockSaveExportedDocument.mockResolvedValue(undefined);
    const { render, getValue } = renderHook();
    await render();

    const state = getValue().getToolbarActionState(
      createArtifact(),
      createDocument(),
    );

    await act(async () => {
      await state?.onExportJson();
    });

    expect(mockSaveDialog).toHaveBeenCalledWith({
      title: "导出 Artifact JSON",
      defaultPath: "board-review.artifact.json",
      filters: [{ name: "Artifact JSON", extensions: ["json"] }],
    });
    expect(mockSaveExportedDocument).toHaveBeenCalledWith(
      "/tmp/board-review.artifact.json",
      JSON.stringify(createDocument(), null, 2),
    );
    expect(toastSuccess).toHaveBeenCalledWith("已导出 Artifact JSON");
  });
});
