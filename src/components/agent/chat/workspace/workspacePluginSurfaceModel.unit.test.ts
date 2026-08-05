import { describe, expect, it } from "vitest";
import type { WorkspaceRightSurfacePendingRequest } from "@/lib/api/workspaceRightSurface";
import {
  buildWorkspacePluginSurfaceFromPendingRequest,
  buildWorkspacePluginSurfaceFromPendingRequests,
  buildWorkspacePluginSurfacesFromPendingRequests,
  buildWorkspacePluginSurfacesFromThreadItems,
  closeWorkspacePluginSurfaceDescriptor,
  mergeWorkspacePluginSurfaceDescriptors,
  resolveWorkspacePluginSurfaceActiveContainerId,
  resolveWorkspacePluginSurfaceSessionEpoch,
  resolveWorkspacePluginSurfaceThreadId,
  selectWorkspacePluginSurfaceDescriptor,
} from "./workspacePluginSurfaceModel";

const basePending: WorkspaceRightSurfacePendingRequest = {
  requestId: "right_surface_app_1",
  workspaceId: "workspace-main",
  workspaceRoot: "/workspace/project",
  sessionId: "session-main",
  surfaceKind: "appSurface",
  origin: "runtime",
  priority: "foreground",
  status: "pending",
  reason: "plugin_surface_ready",
  requestedAt: "2026-06-24T00:00:00.000Z",
};

describe("workspacePluginSurfaceModel", () => {
  it("应从 canonical MCP tool item 生成可恢复的 MCP App surface", () => {
    const surfaces = buildWorkspacePluginSurfacesFromThreadItems([
      {
        id: "item-mcp-app-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        sequence: 3,
        status: "completed",
        started_at: "2026-08-04T00:00:00.000Z",
        updated_at: "2026-08-04T00:00:01.000Z",
        completed_at: "2026-08-04T00:00:01.000Z",
        type: "tool_call",
        tool_name: "mcp__plugin__demo__report",
        success: true,
        metadata: {
          canonical_type: "mcpToolCall",
          server: "plugin__demo__server",
          plugin_id: "demo-plugin",
          mcp_app_resource_uri: "ui://demo/report.html",
        },
      },
    ]);

    expect(surfaces).toEqual([
      {
        appId: "demo-plugin",
        title: "demo-plugin",
        containerId: "mcp-app-item-mcp-app-1",
        activeStrategy: "webContentsView",
        supportedStrategies: ["webContentsView"],
        mcpApp: {
          resourceUri: "ui://demo/report.html",
          serverName: "plugin__demo__server",
          toolItemId: "item-mcp-app-1",
        },
      },
    ]);
    expect(
      buildWorkspacePluginSurfacesFromThreadItems(
        [
          {
            id: "item-mcp-app-1",
            thread_id: "thread-1",
            turn_id: "turn-1",
            sequence: 3,
            status: "completed",
            started_at: "2026-08-04T00:00:00.000Z",
            updated_at: "2026-08-04T00:00:01.000Z",
            type: "tool_call",
            tool_name: "mcp__plugin__demo__report",
            metadata: {
              canonical_type: "mcpToolCall",
              server: "plugin__demo__server",
              plugin_id: "demo-plugin",
              mcp_app_resource_uri: "ui://demo/report.html",
            },
          },
        ],
        ["mcp-app-item-mcp-app-1"],
      ),
    ).toEqual([]);
  });

  it("应按 canonical thread identity 过滤旧会话 item", () => {
    const item = {
      id: "item-mcp-app-1",
      thread_id: "thread-old",
      turn_id: "turn-1",
      sequence: 3,
      status: "completed" as const,
      started_at: "2026-08-04T00:00:00.000Z",
      updated_at: "2026-08-04T00:00:01.000Z",
      type: "tool_call" as const,
      tool_name: "mcp__plugin__demo__report",
      metadata: {
        canonical_type: "mcpToolCall",
        server: "plugin__demo__server",
        plugin_id: "demo-plugin",
        mcp_app_resource_uri: "ui://demo/report.html",
      },
    };

    expect(
      buildWorkspacePluginSurfacesFromThreadItems([item], [], "thread-new"),
    ).toEqual([]);
    expect(
      buildWorkspacePluginSurfacesFromThreadItems([item], [], "thread-old")[0]
        ?.containerId,
    ).toBe("mcp-app-item-mcp-app-1");
  });

  it("session 切换期间应屏蔽旧 thread，目标 thread 到达后恢复同一 identity", () => {
    const initial = resolveWorkspacePluginSurfaceSessionEpoch({
      currentSessionId: "session-old",
      currentThreadId: "thread-old",
      previousEpoch: null,
    });
    const switching = resolveWorkspacePluginSurfaceSessionEpoch({
      currentSessionId: "session-new",
      currentThreadId: "thread-old",
      previousEpoch: initial.epoch,
    });
    const restored = resolveWorkspacePluginSurfaceSessionEpoch({
      currentSessionId: "session-new",
      currentThreadId: "thread-new",
      previousEpoch: initial.epoch,
    });

    expect(initial).toEqual({
      epoch: { sessionId: "session-old", threadId: "thread-old" },
      ready: true,
    });
    expect(switching).toEqual({
      epoch: { sessionId: "session-old", threadId: "thread-old" },
      ready: false,
    });
    expect(restored).toEqual({
      epoch: { sessionId: "session-new", threadId: "thread-new" },
      ready: true,
    });
  });

  it("read model 延迟时应从同一 scene items 恢复 thread identity", () => {
    const item = {
      id: "item-mcp-app-1",
      thread_id: "thread-scene",
      turn_id: "turn-1",
      sequence: 1,
      status: "completed" as const,
      started_at: "2026-08-04T00:00:00.000Z",
      updated_at: "2026-08-04T00:00:01.000Z",
      type: "tool_call" as const,
      tool_name: "mcp__plugin__demo__report",
    };

    expect(resolveWorkspacePluginSurfaceThreadId(null, [item])).toBe(
      "thread-scene",
    );
    expect(
      resolveWorkspacePluginSurfaceThreadId("thread-read-model", [item]),
    ).toBe("thread-read-model");
    expect(
      resolveWorkspacePluginSurfaceThreadId(null, [
        item,
        { ...item, id: "item-2", thread_id: "thread-other" },
      ]),
    ).toBeNull();
  });

  it("冷恢复应重建同一 surface identity，用户关闭后不应再次投影", () => {
    const item = {
      id: "item-mcp-app-stable",
      thread_id: "thread-stable",
      turn_id: "turn-stable",
      sequence: 1,
      status: "completed" as const,
      started_at: "2026-08-04T00:00:00.000Z",
      updated_at: "2026-08-04T00:00:01.000Z",
      type: "tool_call" as const,
      tool_name: "mcp__plugin__demo__report",
      metadata: {
        canonical_type: "mcpToolCall",
        server: "plugin__demo__server",
        plugin_id: "demo-plugin",
        mcp_app_resource_uri: "ui://demo/report.html",
      },
    };
    const first = buildWorkspacePluginSurfacesFromThreadItems(
      [item],
      [],
      "thread-stable",
    );
    const restored = buildWorkspacePluginSurfacesFromThreadItems(
      [item],
      [],
      "thread-stable",
    );

    expect(restored).toEqual(first);
    expect(
      buildWorkspacePluginSurfacesFromThreadItems(
        [item],
        [first[0]?.containerId ?? ""],
        "thread-stable",
      ),
    ).toEqual([]);
  });
  it("应从 Right Surface pending metadata 水合 Plugin Surface descriptor", () => {
    expect(
      buildWorkspacePluginSurfaceFromPendingRequests([
        {
          ...basePending,
          candidateId: "content-factory-app",
          metadata: {
            appId: "content-factory-app",
            title: "内容工厂",
            surface: {
              activeStrategy: "controlledBrowserWindow",
              supportedStrategies: [
                "controlledBrowserWindow",
                "webContentsView",
              ],
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
        },
      ]),
    ).toEqual({
      appId: "content-factory-app",
      title: "内容工厂",
      entryUrl: "http://127.0.0.1:4199/dashboard",
      containerId: "plugin-shell-content-factory-app-standalone",
      activeStrategy: "controlledBrowserWindow",
      supportedStrategies: ["controlledBrowserWindow", "webContentsView"],
      sourceRequestId: "right_surface_app_1",
    });
  });

  it("应从多个 pending 水合多个 Plugin Surface，并按 containerId 去重", () => {
    expect(
      buildWorkspacePluginSurfacesFromPendingRequests([
        {
          ...basePending,
          requestId: "right_surface_app_1",
          candidateId: "content-factory-app",
          metadata: {
            appId: "content-factory-app",
            title: "内容工厂",
            entryUrl: "http://127.0.0.1:4199/dashboard",
            containerId: "plugin-shell-content-factory-app",
            supportedStrategies: ["webContentsView"],
          },
        },
        {
          ...basePending,
          requestId: "right_surface_app_2",
          candidateId: "prompt-lab-app",
          metadata: {
            appId: "prompt-lab-app",
            title: "提示词实验室",
            entryUrl: "http://127.0.0.1:4201/",
            containerId: "plugin-shell-prompt-lab-app",
            supportedStrategies: ["webContentsView"],
          },
        },
        {
          ...basePending,
          requestId: "right_surface_app_3",
          candidateId: "content-factory-app",
          metadata: {
            appId: "content-factory-app",
            title: "内容工厂新窗口",
            entryUrl: "http://127.0.0.1:4199/profile",
            containerId: "plugin-shell-content-factory-app",
            supportedStrategies: ["webContentsView"],
          },
        },
      ]),
    ).toEqual([
      expect.objectContaining({
        appId: "content-factory-app",
        title: "内容工厂新窗口",
        entryUrl: "http://127.0.0.1:4199/profile",
        containerId: "plugin-shell-content-factory-app",
        sourceRequestId: "right_surface_app_3",
      }),
      expect.objectContaining({
        appId: "prompt-lab-app",
        title: "提示词实验室",
        entryUrl: "http://127.0.0.1:4201/",
        containerId: "plugin-shell-prompt-lab-app",
        sourceRequestId: "right_surface_app_2",
      }),
    ]);
  });

  it("应合并、聚焦和关闭 Plugin Surface 实例", () => {
    const contentFactory = {
      appId: "content-factory-app",
      title: "内容工厂",
      entryUrl: "http://127.0.0.1:4199/dashboard",
      containerId: "plugin-shell-content-factory-app",
      activeStrategy: "webContentsView" as const,
      supportedStrategies: ["webContentsView" as const],
    };
    const promptLab = {
      appId: "prompt-lab-app",
      title: "提示词实验室",
      entryUrl: "http://127.0.0.1:4201/",
      containerId: "plugin-shell-prompt-lab-app",
      activeStrategy: "webContentsView" as const,
      supportedStrategies: ["webContentsView" as const],
    };
    const merged = mergeWorkspacePluginSurfaceDescriptors(
      [contentFactory],
      [promptLab],
    );

    expect(merged).toEqual([contentFactory, promptLab]);
    expect(
      resolveWorkspacePluginSurfaceActiveContainerId({
        activeContainerId: contentFactory.containerId,
        preferredContainerId: promptLab.containerId,
        surfaces: merged,
      }),
    ).toBe(promptLab.containerId);
    expect(
      selectWorkspacePluginSurfaceDescriptor(merged, promptLab.containerId),
    ).toBe(promptLab);
    expect(
      closeWorkspacePluginSurfaceDescriptor({
        activeContainerId: promptLab.containerId,
        containerId: promptLab.containerId,
        surfaces: merged,
      }),
    ).toEqual({
      activeContainerId: contentFactory.containerId,
      surfaces: [contentFactory],
    });
  });

  it("应拒绝 iframe / BrowserView 合同回流到 appSurface", () => {
    expect(
      buildWorkspacePluginSurfaceFromPendingRequest({
        ...basePending,
        metadata: {
          appId: "legacy-content-factory",
          entryUrl: "http://127.0.0.1:4199/dashboard",
          supportedStrategies: ["webContentsView"],
          embedding: {
            rightSurfaceDock: true,
            iframe: true,
            browserView: false,
          },
        },
      }),
    ).toBeNull();

    expect(
      buildWorkspacePluginSurfaceFromPendingRequest({
        ...basePending,
        metadata: {
          appId: "legacy-content-factory",
          entryUrl: "http://127.0.0.1:4199/dashboard",
          supportedStrategies: ["controlledBrowserWindow"],
          embedding: {
            rightSurfaceDock: true,
            iframe: false,
            browserView: false,
          },
        },
      }),
    ).toBeNull();
  });
});
