import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Artifact } from "@/lib/artifact/types";
import {
  ArtifactWorkbenchPreview,
  WorkspaceLiveCanvasPreview,
} from "./workbenchPreview";

vi.mock("@/components/artifact", () => ({
  ArtifactCanvasOverlay: () => <div data-testid="artifact-overlay" />,
  ArtifactRenderer: ({ artifact }: { artifact: Artifact }) => (
    <div data-testid="artifact-renderer">{artifact.title}</div>
  ),
  ArtifactToolbar: ({ actionsSlot }: { actionsSlot?: ReactNode }) => (
    <div data-testid="artifact-toolbar">{actionsSlot}</div>
  ),
}));

vi.mock("./ArtifactWorkbenchShell", () => ({
  ArtifactWorkbenchShell: ({ artifact }: { artifact: Artifact }) => (
    <div data-testid="artifact-workbench-shell">{artifact.title}</div>
  ),
}));

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

function createArtifact(): Artifact {
  return {
    id: "artifact-live-1",
    type: "document",
    title: "live.artifact.json",
    content: '{"schemaVersion":"artifact_document.v1"}',
    status: "complete",
    meta: {
      filePath: ".lime/artifacts/thread-1/live.artifact.json",
      filename: "live.artifact.json",
      language: "json",
    },
    position: { start: 0, end: 38 },
    createdAt: 1,
    updatedAt: 1,
  };
}

function createArtifactVariant(
  id: string,
  type: Artifact["type"],
  content: string,
): Artifact {
  return {
    id,
    type,
    title: `${id}.artifact`,
    content,
    status: "complete",
    meta: {
      filename: `${id}.artifact`,
      language: type === "code" ? "ts" : "json",
    },
    position: { start: 0, end: content.length },
    createdAt: 1,
    updatedAt: 1,
  };
}

function createDocumentArtifact(): Artifact {
  return createArtifactVariant(
    "document-preview",
    "document",
    JSON.stringify({
      schemaVersion: "artifact_document.v1",
      title: "结构化文稿",
      blocks: [
        {
          id: "body-1",
          type: "rich_text",
          content: "正文内容",
        },
      ],
    }),
  );
}

function renderArtifactPreview(artifact: Artifact) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedRoots.push({ container, root });

  const commonProps = {
    currentCanvasArtifact: artifact,
    displayedCanvasArtifact: artifact,
    artifactOverlay: null,
    showPreviousVersionBadge: false,
    artifactViewMode: "preview" as const,
    onArtifactViewModeChange: vi.fn(),
    artifactPreviewSize: "desktop" as const,
    onArtifactPreviewSizeChange: vi.fn(),
    onCloseCanvas: vi.fn(),
  };

  act(() => {
    root.render(
      <ArtifactWorkbenchPreview artifact={artifact} {...commonProps} />,
    );
  });

  return {
    container,
    rerender(nextArtifact: Artifact) {
      act(() => {
        root.render(
          <ArtifactWorkbenchPreview
            artifact={nextArtifact}
            {...commonProps}
            currentCanvasArtifact={nextArtifact}
            displayedCanvasArtifact={nextArtifact}
          />,
        );
      });
    },
  };
}

describe("workbenchPreview", () => {
  afterEach(() => {
    while (mountedRoots.length > 0) {
      const mounted = mountedRoots.pop();
      if (!mounted) {
        continue;
      }
      act(() => {
        mounted.root.unmount();
      });
      mounted.container.remove();
    }
    vi.restoreAllMocks();
  });

  it("general live artifact 应强制走 canvas-only 文稿布局，避免重复 inspector", () => {
    const artifact = createArtifact();
    const renderArtifactPreview = vi.fn(() => null);

    WorkspaceLiveCanvasPreview({
      currentImageWorkbenchActive: false,
      imageWorkbenchProps: {} as never,
      onCloseCanvas: vi.fn(),
      canvasRenderTheme: "general",
      liveArtifact: artifact,
      hasDisplayedLiveArtifact: true,
      renderArtifactPreview,
      generalCanvasPanelProps: null,
      shouldShowCanvasLoadingState: false,
      canvasLoadingLabel: "loading",
      canvasFactoryProps: null,
      stackedWorkbenchTrigger: null,
    });

    expect(renderArtifactPreview).toHaveBeenCalledTimes(1);
    expect(renderArtifactPreview).toHaveBeenCalledWith(
      artifact,
      expect.objectContaining({
        stackedWorkbenchTrigger: null,
      }),
    );
  });

  it("ArtifactWorkbenchPreview 在文稿和普通 artifact 间切换时保持 Hook 顺序稳定", () => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const documentArtifact = createDocumentArtifact();
    const codeArtifact = createArtifactVariant(
      "code-preview",
      "code",
      "export const value = 1;",
    );

    const mounted = renderArtifactPreview(documentArtifact);
    mounted.rerender(codeArtifact);

    const hookOrderErrors = consoleError.mock.calls.filter(([message]) =>
      String(message).includes("Rendered fewer hooks"),
    );
    expect(hookOrderErrors).toHaveLength(0);
    expect(mounted.container.textContent).toContain("code-preview.artifact");
  });
});
