import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Artifact } from "@/lib/artifact/types";
import { useWorkspaceRightSurfaceArtifactOpenRuntime } from "./useWorkspaceRightSurfaceArtifactOpenRuntime";
import type { WorkspaceArticleWorkspace } from "./workspaceArticleWorkspaceModel";

type HookProps = Parameters<
  typeof useWorkspaceRightSurfaceArtifactOpenRuntime
>[0];

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

const currentArticleWorkspace: WorkspaceArticleWorkspace = {
  schemaVersion: "article-workspace.v1",
  appId: "content-factory-app",
  sessionId: "sess-history",
  workspaceId: "workspace-main",
  source: "threadRead",
  objectCount: 1,
  actionHistory: [],
  objects: [
    {
      ref: {
        appId: "content-factory-app",
        kind: "articleDraft",
        id: "article-1",
        sessionId: "sess-history",
        artifactIds: ["artifact-article-1"],
      },
      title: "公众号文章草稿",
      status: "needs_review",
      previewArtifactId: "artifact-article-1",
      source: {
        documentText: "# 公众号文章草稿\n\n正文",
      },
    },
  ],
  primaryObjectRef: {
    appId: "content-factory-app",
    kind: "articleDraft",
    id: "article-1",
    sessionId: "sess-history",
    artifactIds: ["artifact-article-1"],
  },
  selectedObjectRef: {
    appId: "content-factory-app",
    kind: "articleDraft",
    id: "article-1",
    sessionId: "sess-history",
  },
};

function artifact(overrides: Partial<Artifact> = {}): Artifact {
  return {
    id: "artifact-article-1",
    type: "document",
    title: "公众号文章草稿",
    content: "# 公众号文章草稿\n\n正文",
    status: "complete",
    meta: {},
    position: { start: 0, end: 0 },
    createdAt: 1,
    updatedAt: 1,
    ...overrides,
  };
}

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<
    typeof useWorkspaceRightSurfaceArtifactOpenRuntime
  > | null = null;

  const defaultProps: HookProps = {
    clearFocusedArtifactBlock: vi.fn(),
    fallbackOpenArtifact: vi.fn(),
    openArticleWorkspaceRightSurface: vi.fn(),
    setExpertInfoPanelCollapsed: vi.fn(),
    setHarnessPanelVisible: vi.fn(),
  };

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceRightSurfaceArtifactOpenRuntime(currentProps);
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
    props: { ...defaultProps, ...props },
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
});

describe("useWorkspaceRightSurfaceArtifactOpenRuntime", () => {
  it("Article Workspace artifact 应打开右侧 Article surface 并消费 pending request", async () => {
    const refreshRightSurfacePendingRequests = vi.fn().mockResolvedValue(null);
    const consumePendingRequestsForSurface = vi.fn().mockResolvedValue(null);
    const { render, getValue, props } = renderHook();

    await render();
    act(() => {
      getValue().bindArticleEditorRightSurface(currentArticleWorkspace);
      getValue().bindRightSurfacePendingActions({
        consumePendingRequestsForSurface,
        refreshRightSurfacePendingRequests,
      });
      getValue().handleWorkspaceArtifactClick(
        artifact({
          meta: {
            openedFrom: "right_surface_article_workspace",
            articleWorkspace: {
              appId: "content-factory-app",
              sessionId: "sess-history",
              objectKind: "articleDraft",
              objectId: "article-1",
            },
          },
        }),
      );
    });

    expect(props.clearFocusedArtifactBlock).toHaveBeenCalledTimes(1);
    expect(props.setHarnessPanelVisible).toHaveBeenCalledWith(false);
    expect(props.setExpertInfoPanelCollapsed).toHaveBeenCalledWith(true);
    expect(props.openArticleWorkspaceRightSurface).toHaveBeenCalledWith(
      currentArticleWorkspace,
    );
    expect(refreshRightSurfacePendingRequests).toHaveBeenCalledTimes(1);
    expect(consumePendingRequestsForSurface).toHaveBeenCalledOnce();
    expect(consumePendingRequestsForSurface).toHaveBeenCalledWith(
      "articleWorkspace",
    );
    expect(props.fallbackOpenArtifact).not.toHaveBeenCalled();
  });

  it("普通 artifact 应清理焦点后回落到原 artifact 打开动作", async () => {
    const consumePendingRequestsForSurface = vi.fn().mockResolvedValue(null);
    const { render, getValue, props } = renderHook();
    const ordinaryArtifact = artifact({ id: "artifact-html-1", meta: {} });

    await render();
    act(() => {
      getValue().bindArticleEditorRightSurface(currentArticleWorkspace);
      getValue().bindRightSurfacePendingActions({
        consumePendingRequestsForSurface,
      });
      getValue().handleWorkspaceArtifactClick(ordinaryArtifact);
    });

    expect(props.clearFocusedArtifactBlock).toHaveBeenCalledTimes(1);
    expect(props.fallbackOpenArtifact).toHaveBeenCalledWith(ordinaryArtifact);
    expect(props.openArticleWorkspaceRightSurface).not.toHaveBeenCalled();
    expect(consumePendingRequestsForSurface).not.toHaveBeenCalled();
  });
});
