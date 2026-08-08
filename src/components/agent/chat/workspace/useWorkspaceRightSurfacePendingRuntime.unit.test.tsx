import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { WorkspaceRightSurfacePendingRequest } from "@/lib/api/workspaceRightSurface";
import {
  applyWorkspaceRightSurfacePendingChanges,
  buildWorkspaceRightSurfacePendingListParams,
  useWorkspaceRightSurfacePendingRuntime,
} from "./useWorkspaceRightSurfacePendingRuntime";

type HookProps = Parameters<typeof useWorkspaceRightSurfacePendingRuntime>[0];

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

const pendingRequest: WorkspaceRightSurfacePendingRequest = {
  requestId: "right_surface_1",
  workspaceId: "workspace-main",
  workspaceRoot: "/workspace/project",
  sessionId: "session-main",
  surfaceKind: "objectCanvas",
  origin: "mcpTool",
  priority: "foreground",
  status: "pending",
  reason: "browser_assist_candidate",
  requestedAt: "2026-06-23T00:00:00.000Z",
};
const filesPendingRequest: WorkspaceRightSurfacePendingRequest = {
  ...pendingRequest,
  requestId: "right_surface_files_1",
  surfaceKind: "files",
  origin: "skill",
  reason: "file_preview_ready",
  candidateId: "docs/report.md",
};
const articleEditorPendingRequest: WorkspaceRightSurfacePendingRequest = {
  ...pendingRequest,
  requestId: "right_surface_article_workspace_1",
  surfaceKind: "articleWorkspace",
  origin: "runtime",
  reason: "article_workspace_ready",
  metadata: {
    workspacePatch: {
      schemaVersion: "article-workspace.v1",
      appId: "content-factory-app",
      sessionId: "session-main",
      workspaceId: "workspace-main",
      selectedObjectRef: {
        appId: "content-factory-app",
        kind: "articleDraft",
        id: "article-1",
        sessionId: "session-main",
      },
      objects: [
        {
          ref: {
            appId: "content-factory-app",
            kind: "articleDraft",
            id: "article-1",
            sessionId: "session-main",
          },
          title: "公众号文章草稿",
          status: "ready",
          summary: "已生成首版文章",
          source: {
            taskKind: "content.article.generate",
          },
        },
      ],
    },
  },
};
const articleEditorArtifactPendingRequest: WorkspaceRightSurfacePendingRequest = {
  ...articleEditorPendingRequest,
  requestId: "right_surface_article_workspace_artifact_1",
  metadata: {
    artifact: {
      artifactId: "artifact-workspace-patch-1",
      path: ".lime/artifacts/workspace-patch.json",
      title: "内容工厂工作区补丁",
      kind: "workspace_patch",
      status: "ready",
      metadata: articleEditorPendingRequest.metadata,
    },
  },
};
const pluginSurfacePendingRequest: WorkspaceRightSurfacePendingRequest = {
  ...pendingRequest,
  requestId: "right_surface_plugin_1",
  surfaceKind: "appSurface",
  origin: "runtime",
  reason: "plugin_surface_ready",
  candidateId: "content-factory-app",
  metadata: {
    appId: "content-factory-app",
    title: "内容工厂",
    surface: {
      activeStrategy: "controlledBrowserWindow",
      supportedStrategies: ["controlledBrowserWindow", "webContentsView"],
      entryUrl: "http://127.0.0.1:4199/dashboard",
      containerId: "plugin-shell-content-factory-app-standalone",
      embedding: {
        standaloneWindow: true,
        rightSurfaceDock: true,
        iframe: false,
        browserView: false,
      },
    },
  },
};
const browserPendingRequest: WorkspaceRightSurfacePendingRequest = {
  ...pendingRequest,
  requestId: "right_surface_browser_1",
  surfaceKind: "browser",
  origin: "runtime",
  priority: "foreground",
  reason: "browser_requirement",
  candidateId: "https://example.com/fallback",
  metadata: {
    title: "Example Browser",
    launchUrl: "https://example.com/dashboard",
    browserSessionId: "browser-session-1",
    profileKey: "task-profile",
    targetId: "target-1",
  },
};

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<
    typeof useWorkspaceRightSurfacePendingRuntime
  > | null = null;

  const defaultProps: HookProps = {
    enabled: true,
    workspaceId: "workspace-main",
    workspaceRoot: "/workspace/project",
    sessionId: "session-main",
    pollIntervalMs: 0,
    eventDrainIntervalMs: 0,
    isBridgeAvailable: () => true,
    listPending: vi.fn(async () => ({ pending: [pendingRequest] })),
    consumePending: vi.fn(async () => ({
      status: "consumed",
      consumedRequestIds: [pendingRequest.requestId],
      missingRequestIds: [],
    })),
    dismissPending: vi.fn(async () => ({
      status: "dismissed",
      dismissedRequestIds: [pendingRequest.requestId],
      missingRequestIds: [],
    })),
    drainPendingChanges: vi.fn(async () => []),
    now: () => 200,
  };

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceRightSurfacePendingRuntime(currentProps);
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
    props: { ...defaultProps, ...props },
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
  vi.restoreAllMocks();
});

describe("buildWorkspaceRightSurfacePendingListParams", () => {
  it("有 workspace 过滤时应优先按 workspace 查询，不把 session 作为唯一范围", () => {
    expect(
      buildWorkspaceRightSurfacePendingListParams({
        workspaceId: " workspace-main ",
        workspaceRoot: " /workspace/project ",
        sessionId: " session-main ",
      }),
    ).toEqual({
      limit: 50,
      workspaceId: "workspace-main",
      workspaceRoot: "/workspace/project",
    });
  });

  it("没有 workspace 过滤时可按 session 查询；没有上下文时返回 null", () => {
    expect(
      buildWorkspaceRightSurfacePendingListParams({
        sessionId: " session-main ",
      }),
    ).toEqual({
      limit: 50,
      sessionId: "session-main",
    });

    expect(buildWorkspaceRightSurfacePendingListParams({})).toBeNull();
  });
});

describe("applyWorkspaceRightSurfacePendingChanges", () => {
  it("requested notification 应按当前查询范围增量加入 pending 请求", () => {
    expect(
      applyWorkspaceRightSurfacePendingChanges(
        [],
        [
          {
            changeType: "requested",
            requestIds: [filesPendingRequest.requestId],
            pending: [
              filesPendingRequest,
              {
                ...filesPendingRequest,
                requestId: "right_surface_other_workspace",
                workspaceId: "workspace-other",
              },
            ],
          },
        ],
        {
          limit: 50,
          workspaceId: "workspace-main",
          workspaceRoot: "/workspace/project",
        },
      ),
    ).toEqual([filesPendingRequest]);
  });

  it("requested notification 应覆盖同 id 的旧 pending 快照", () => {
    const updatedRequest = {
      ...pendingRequest,
      reason: "updated_reason",
      metadata: { title: "updated" },
    };

    expect(
      applyWorkspaceRightSurfacePendingChanges(
        [pendingRequest],
        [
          {
            changeType: "requested",
            requestIds: [pendingRequest.requestId],
            pending: [updatedRequest],
          },
        ],
      ),
    ).toEqual([updatedRequest]);
  });

  it("consumed / dismissed notification 应只移除服务端确认变化的 request id", () => {
    expect(
      applyWorkspaceRightSurfacePendingChanges(
        [pendingRequest, filesPendingRequest],
        [
          {
            changeType: "consumed",
            requestIds: [pendingRequest.requestId],
            consumedRequestIds: [pendingRequest.requestId],
            missingRequestIds: ["right_surface_missing"],
          },
          {
            changeType: "dismissed",
            requestIds: [filesPendingRequest.requestId],
            dismissedRequestIds: [filesPendingRequest.requestId],
          },
        ],
      ),
    ).toEqual([]);
  });
});

describe("useWorkspaceRightSurfacePendingRuntime", () => {
  it("autoRefresh 关闭时不应首帧拉取或订阅右侧 surface pending", async () => {
    const listPending = vi.fn(async () => ({ pending: [pendingRequest] }));
    const drainPendingChanges = vi.fn(async () => []);
    const subscribePendingChanges = vi.fn(() => () => undefined);
    const { render, getValue } = renderHook({
      autoRefreshEnabled: false,
      listPending,
      drainPendingChanges,
      subscribePendingChanges,
    });

    await render();
    await act(async () => {
      await Promise.resolve();
    });

    expect(listPending).not.toHaveBeenCalled();
    expect(drainPendingChanges).not.toHaveBeenCalled();
    expect(subscribePendingChanges).not.toHaveBeenCalled();
    expect(getValue().pendingRequests).toEqual([]);
    expect(getValue().pendingIntents).toEqual([]);
  });

  it("autoRefresh 关闭后仍允许显式刷新 pending 请求", async () => {
    const listPending = vi.fn(async () => ({ pending: [pendingRequest] }));
    const { render, getValue } = renderHook({
      autoRefreshEnabled: false,
      listPending,
    });

    await render();
    await act(async () => {
      await getValue().refreshPendingRequests();
    });

    expect(listPending).toHaveBeenCalledWith({
      limit: 50,
      workspaceId: "workspace-main",
      workspaceRoot: "/workspace/project",
    });
    expect(getValue().pendingRequests).toEqual([pendingRequest]);
  });

  it("应通过 App Server pending/list 拉取请求并投影为 Right Surface intent", async () => {
    const listPending = vi.fn(async () => ({ pending: [pendingRequest] }));
    const { render, getValue } = renderHook({ listPending });

    await render();

    await vi.waitFor(() => {
      expect(listPending).toHaveBeenCalledWith({
        limit: 50,
        workspaceId: "workspace-main",
        workspaceRoot: "/workspace/project",
      });
    });
    await vi.waitFor(() => {
      expect(getValue().pendingRequests).toEqual([pendingRequest]);
    });
    expect(getValue().pendingIntents).toEqual([
      expect.objectContaining({
        id: "app-server:right_surface_1",
        priority: "foreground",
        command: expect.objectContaining({
          action: "open",
          kind: "objectCanvas",
          origin: "mcpTool",
          reason: "browser_assist_candidate",
        }),
      }),
    ]);
    expect(getValue().pendingObjectCanvasCandidate).toEqual(
      expect.objectContaining({
        candidateId: "right_surface_1",
        sessionId: "session-main",
        sourceKind: "rightSurfacePending",
        sourceRequestId: "right_surface_1",
      }),
    );
  });

  it("应把 Article Editor pending metadata 投影为文章编辑器", async () => {
    const listPending = vi.fn(async () => ({
      pending: [articleEditorPendingRequest],
    }));
    const { render, getValue } = renderHook({ listPending });

    await render();

    await vi.waitFor(() => {
      expect(getValue().pendingArticleWorkspace).toMatchObject({
        appId: "content-factory-app",
        sessionId: "session-main",
        source: "rightSurfacePending",
        objects: [
          {
            title: "公众号文章草稿",
            status: "ready",
          },
        ],
      });
    });
    expect(getValue().pendingIntents).toEqual([
      expect.objectContaining({
        id: "app-server:right_surface_article_workspace_1",
        command: expect.objectContaining({
          action: "open",
          kind: "articleWorkspace",
          reason: "article_workspace_ready",
        }),
      }),
    ]);
  });

  it("应把 artifact metadata 包装的内容工厂 workspace patch 投影为文章编辑器", async () => {
    const listPending = vi.fn(async () => ({
      pending: [articleEditorArtifactPendingRequest],
    }));
    const { render, getValue } = renderHook({ listPending });

    await render();

    await vi.waitFor(() => {
      expect(getValue().pendingArticleWorkspace).toMatchObject({
        appId: "content-factory-app",
        sessionId: "session-main",
        source: "rightSurfacePending",
        sourceArtifacts: [
          {
            requestId: "right_surface_article_workspace_artifact_1",
            origin: "runtime",
            artifactRef: "artifact-workspace-patch-1",
            kind: "workspace_patch",
          },
        ],
        objects: [
          {
            title: "公众号文章草稿",
            status: "ready",
          },
        ],
      });
    });
  });

  it("应把 appSurface pending metadata 投影为右侧 Plugin Surface", async () => {
    const listPending = vi.fn(async () => ({
      pending: [pluginSurfacePendingRequest],
    }));
    const { render, getValue } = renderHook({ listPending });

    await render();

    await vi.waitFor(() => {
      expect(getValue().pendingPluginSurface).toEqual({
        appId: "content-factory-app",
        title: "内容工厂",
        entryUrl: "http://127.0.0.1:4199/dashboard",
        containerId: "plugin-shell-content-factory-app-standalone",
        activeStrategy: "controlledBrowserWindow",
        supportedStrategies: ["controlledBrowserWindow", "webContentsView"],
        sourceRequestId: "right_surface_plugin_1",
      });
    });
    expect(getValue().pendingPluginSurfaces).toEqual([
      expect.objectContaining({
        appId: "content-factory-app",
        containerId: "plugin-shell-content-factory-app-standalone",
      }),
    ]);
    expect(getValue().pendingIntents).toEqual([
      expect.objectContaining({
        id: "app-server:right_surface_plugin_1",
        command: expect.objectContaining({
          action: "open",
          kind: "appSurface",
          reason: "plugin_surface_ready",
        }),
      }),
    ]);
  });

  it("应把 browser pending metadata 投影为 Right Surface Browser intent", async () => {
    const listPending = vi.fn(async () => ({
      pending: [browserPendingRequest],
    }));
    const { render, getValue } = renderHook({ listPending });

    await render();

    await vi.waitFor(() => {
      expect(getValue().pendingBrowserIntent).toEqual({
        source: "rightSurfacePending",
        sourceRequestId: "right_surface_browser_1",
        origin: "runtime",
        reason: "browser_requirement",
        priority: "foreground",
        browserSessionId: "browser-session-1",
        launchUrl: "https://example.com/dashboard",
        title: "Example Browser",
        profileKey: "task-profile",
        targetId: "target-1",
        lifecycleState: null,
        controlMode: null,
        sessionRef: {
          sourceRequestId: "right_surface_browser_1",
          browserSessionId: "browser-session-1",
          profileKey: "task-profile",
          adapterKind: "cdp",
          launchUrl: "https://example.com/dashboard",
          title: "Example Browser",
        },
      });
    });
    expect(getValue().pendingIntents).toEqual([
      expect.objectContaining({
        id: "app-server:right_surface_browser_1",
        priority: "foreground",
        command: expect.objectContaining({
          action: "open",
          kind: "browser",
          reason: "browser_requirement",
        }),
      }),
    ]);
  });

  it("bridge 不可用时不应调用 App Server，也不应保留旧 pending", async () => {
    const listPending = vi.fn(async () => ({ pending: [pendingRequest] }));
    const { render, getValue } = renderHook({
      isBridgeAvailable: () => false,
      listPending,
    });

    await render();

    expect(listPending).not.toHaveBeenCalled();
    expect(getValue().pendingRequests).toEqual([]);
    expect(getValue().pendingIntents).toEqual([]);
    expect(getValue().lastError).toBeNull();
  });

  it("App Server 查询失败时应 fail closed 并记录错误", async () => {
    const listPending = vi.fn(async () => {
      throw new Error("bridge failed");
    });
    const { render, getValue } = renderHook({ listPending });

    await render();

    await vi.waitFor(() => {
      expect(getValue().lastError?.message).toBe("bridge failed");
    });
    expect(getValue().pendingRequests).toEqual([]);
    expect(getValue().pendingIntents).toEqual([]);
  });

  it("应消费 pendingChanged notification 并实时更新本地 pending 请求", async () => {
    const drainPendingChanges = vi.fn(async () => [
      {
        changeType: "requested",
        requestIds: [filesPendingRequest.requestId],
        pending: [filesPendingRequest],
      },
    ]);
    const { render, getValue } = renderHook({
      eventDrainIntervalMs: 1_000,
      listPending: vi.fn(async () => ({ pending: [] })),
      drainPendingChanges,
    });

    await render();

    await vi.waitFor(() => {
      expect(drainPendingChanges).toHaveBeenCalledWith(20);
    });
    await vi.waitFor(() => {
      expect(getValue().pendingRequests).toEqual([filesPendingRequest]);
    });
    expect(getValue().pendingFileTarget).toEqual({
      relativePath: "docs/report.md",
      title: "report.md",
    });
  });

  it("默认应订阅共享 App Server event bus，而不是直接 drain pendingChanged 队列", async () => {
    let capturedSubscription:
      | Parameters<NonNullable<HookProps["subscribePendingChanges"]>>[0]
      | null = null;
    let capturedOptions:
      | Parameters<NonNullable<HookProps["subscribePendingChanges"]>>[1]
      | null = null;
    const unsubscribe = vi.fn();
    const subscribePendingChanges = vi.fn(
      (
        subscription: Parameters<
          NonNullable<HookProps["subscribePendingChanges"]>
        >[0],
        options: Parameters<
          NonNullable<HookProps["subscribePendingChanges"]>
        >[1],
      ) => {
        capturedSubscription = subscription;
        capturedOptions = options;
        return unsubscribe;
      },
    );
    const listPending = vi.fn(async () => ({ pending: [] }));
    const { render, getValue } = renderHook({
      eventDrainIntervalMs: 1_000,
      listPending,
      drainPendingChanges: undefined,
      subscribePendingChanges,
    });

    await render();

    expect(subscribePendingChanges).toHaveBeenCalledTimes(1);
    expect(capturedOptions).toEqual({
      intervalMs: 1_000,
      isBridgeAvailable: expect.any(Function),
      limit: 20,
    });
    expect(capturedSubscription).not.toBeNull();

    await act(async () => {
      capturedSubscription?.onChanges([
        {
          changeType: "requested",
          requestIds: [filesPendingRequest.requestId],
          pending: [filesPendingRequest],
        },
      ]);
    });

    expect(getValue().pendingRequests).toEqual([filesPendingRequest]);
    expect(getValue().pendingFileTarget).toEqual({
      relativePath: "docs/report.md",
      title: "report.md",
    });
  });

  it("pendingChanged drain 失败时应保留 polling 结果并记录错误", async () => {
    const drainPendingChanges = vi.fn(async () => {
      throw new Error("drain failed");
    });
    const { render, getValue } = renderHook({
      eventDrainIntervalMs: 1_000,
      drainPendingChanges,
    });

    await render();

    await vi.waitFor(() => {
      expect(getValue().pendingRequests).toEqual([pendingRequest]);
    });
    await vi.waitFor(() => {
      expect(getValue().lastError?.message).toBe("drain failed");
    });
  });

  it("打开某类 surface 后应只消费对应类别的 App Server pending 请求", async () => {
    const consumePending = vi.fn(async () => ({
      status: "consumed",
      consumedRequestIds: [pendingRequest.requestId],
      missingRequestIds: [],
    }));
    const { render, getValue } = renderHook({
      consumePending,
      listPending: vi.fn(async () => ({
        pending: [pendingRequest, filesPendingRequest],
      })),
    });

    await render();
    await vi.waitFor(() => {
      expect(getValue().pendingRequests).toHaveLength(2);
    });

    await act(async () => {
      await getValue().consumePendingRequestsForSurface("objectCanvas");
    });

    expect(consumePending).toHaveBeenCalledWith({
      requestIds: [pendingRequest.requestId],
    });
    expect(getValue().pendingRequests).toEqual([filesPendingRequest]);
    expect(getValue().pendingIntents).toEqual([
      expect.objectContaining({
        id: "app-server:right_surface_files_1",
        command: expect.objectContaining({
          kind: "files",
        }),
      }),
    ]);
  });

  it("bridge 不可用时不应消费 pending 请求", async () => {
    const consumePending = vi.fn(async () => ({
      status: "consumed",
      consumedRequestIds: [pendingRequest.requestId],
      missingRequestIds: [],
    }));
    const { render, getValue } = renderHook({
      consumePending,
      isBridgeAvailable: () => false,
    });

    await render();
    await act(async () => {
      await getValue().consumePendingRequestsForSurface("objectCanvas");
    });

    expect(consumePending).not.toHaveBeenCalled();
    expect(getValue().pendingRequests).toEqual([]);
    expect(getValue().lastError).toBeNull();
  });

  it("consume 失败时应保留 pending 并记录错误", async () => {
    const consumePending = vi.fn(async () => {
      throw new Error("consume failed");
    });
    const { render, getValue } = renderHook({ consumePending });

    await render();
    await vi.waitFor(() => {
      expect(getValue().pendingRequests).toEqual([pendingRequest]);
    });

    await act(async () => {
      await getValue().consumePendingRequestsForSurface("objectCanvas");
    });

    expect(getValue().pendingRequests).toEqual([pendingRequest]);
    expect(getValue().lastError?.message).toBe("consume failed");
  });

  it("显式关闭某类 surface 后应只 dismiss 对应类别的 App Server pending 请求", async () => {
    const dismissPending = vi.fn(async () => ({
      status: "dismissed",
      dismissedRequestIds: [pendingRequest.requestId],
      missingRequestIds: [],
    }));
    const { render, getValue } = renderHook({
      dismissPending,
      listPending: vi.fn(async () => ({
        pending: [pendingRequest, filesPendingRequest],
      })),
    });

    await render();
    await vi.waitFor(() => {
      expect(getValue().pendingRequests).toHaveLength(2);
    });

    await act(async () => {
      await getValue().dismissPendingRequestsForSurface(
        "objectCanvas",
        " user_closed_surface ",
      );
    });

    expect(dismissPending).toHaveBeenCalledWith({
      requestIds: [pendingRequest.requestId],
      reason: "user_closed_surface",
    });
    expect(getValue().pendingRequests).toEqual([filesPendingRequest]);
  });

  it("bridge 不可用时不应 dismiss pending 请求", async () => {
    const dismissPending = vi.fn(async () => ({
      status: "dismissed",
      dismissedRequestIds: [pendingRequest.requestId],
      missingRequestIds: [],
    }));
    const { render, getValue } = renderHook({
      dismissPending,
      isBridgeAvailable: () => false,
    });

    await render();
    await act(async () => {
      await getValue().dismissPendingRequestsForSurface("objectCanvas");
    });

    expect(dismissPending).not.toHaveBeenCalled();
    expect(getValue().pendingRequests).toEqual([]);
    expect(getValue().lastError).toBeNull();
  });

  it("dismiss 失败时应保留 pending 并记录错误", async () => {
    const dismissPending = vi.fn(async () => {
      throw new Error("dismiss failed");
    });
    const { render, getValue } = renderHook({ dismissPending });

    await render();
    await vi.waitFor(() => {
      expect(getValue().pendingRequests).toEqual([pendingRequest]);
    });

    await act(async () => {
      await getValue().dismissPendingRequestsForSurface("objectCanvas");
    });

    expect(getValue().pendingRequests).toEqual([pendingRequest]);
    expect(getValue().lastError?.message).toBe("dismiss failed");
  });
});
