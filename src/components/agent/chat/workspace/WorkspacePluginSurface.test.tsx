import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkspacePluginSurface } from "./WorkspacePluginSurface";
import type { WorkspacePluginSurfaceDescriptor } from "./workspacePluginSurfaceModel";

const embeddedBrowserMocks = vi.hoisted(() => ({
  destroyEmbeddedBrowserView: vi.fn(async () => undefined),
  isEmbeddedBrowserHostAvailable: vi.fn(() => true),
  listenEmbeddedBrowserViewLoadFailed: vi.fn(async () => vi.fn()),
  listenEmbeddedBrowserViewState: vi.fn(async () => vi.fn()),
  loadEmbeddedBrowserViewHtml: vi.fn(async (params) => ({
    viewId: params.viewId,
    url: params.sourceUri,
    title: "MCP App",
    canGoBack: false,
    canGoForward: false,
    isLoading: false,
  })),
  mountEmbeddedBrowserView: vi.fn(async () => ({
    viewId: "plugin-surface-plugin-shell-content-factory-app-standalone",
    url: "http://127.0.0.1:4199/dashboard",
    title: "内容工厂",
    canGoBack: false,
    canGoForward: false,
    isLoading: false,
  })),
  navigateEmbeddedBrowserView: vi.fn(async () => ({
    viewId: "plugin-surface-plugin-shell-content-factory-app-standalone",
    url: "http://127.0.0.1:4199/dashboard",
    title: "内容工厂",
    canGoBack: false,
    canGoForward: false,
    isLoading: false,
  })),
  setEmbeddedBrowserViewBounds: vi.fn(async () => ({
    viewId: "plugin-surface-plugin-shell-content-factory-app-standalone",
    url: "http://127.0.0.1:4199/dashboard",
    title: "内容工厂",
    canGoBack: false,
    canGoForward: false,
    isLoading: false,
  })),
}));

vi.mock("@/lib/api/embeddedBrowser", () => embeddedBrowserMocks);

const mcpApiMocks = vi.hoisted(() => ({
  readResource: vi.fn(),
}));

vi.mock("@/lib/api/mcp", () => ({
  mcpApi: mcpApiMocks,
}));

const pluginCatalogMocks = vi.hoisted(() => ({
  listInstalledPluginCatalog: vi.fn(),
}));

vi.mock("@/lib/api/pluginCatalog", () => pluginCatalogMocks);

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const copy: Record<string, string> = {
        "agentChat.pluginSurface.hostUnavailableBody":
          "当前环境没有可用的 Electron WebContentsView bridge。",
        "agentChat.pluginSurface.hostUnavailableTitle": "宿主不可用",
        "agentChat.pluginSurface.closeTab": "关闭",
        "agentChat.pluginSurface.historyUnavailableBody":
          "对话和运行结果仍会保留，但这个历史界面无法再次打开，也不会自动重跑原操作。",
        "agentChat.pluginSurface.historyUnavailableStatus": "仅保留历史",
        "agentChat.pluginSurface.historyUnavailableTitle": "Plugin 已卸载",
        "agentChat.pluginSurface.loading": "正在加载",
        "agentChat.pluginSurface.loadingBody": "正在连接 Plugin。",
        "agentChat.pluginSurface.loadingTitle": "正在打开 Plugin",
        "agentChat.pluginSurface.ready": "已连接",
        "agentChat.pluginSurface.mcpApp.invalidMimeType": "资源格式不受支持",
        "agentChat.pluginSurface.mcpApp.missingHtml": "资源没有可显示内容",
        "agentChat.pluginSurface.mcpApp.uriMismatch": "资源身份不一致",
      };
      return copy[key] ?? key;
    },
  }),
}));

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

const surface: WorkspacePluginSurfaceDescriptor = {
  appId: "content-factory-app",
  title: "内容工厂",
  entryUrl: "http://127.0.0.1:4199/dashboard",
  containerId: "plugin-shell-content-factory-app-standalone",
  activeStrategy: "controlledBrowserWindow",
  supportedStrategies: ["controlledBrowserWindow", "webContentsView"],
  sourceRequestId: "right_surface_plugin_1",
};

const promptLabSurface: WorkspacePluginSurfaceDescriptor = {
  appId: "prompt-lab-app",
  title: "提示词实验室",
  entryUrl: "http://127.0.0.1:4201/",
  containerId: "plugin-shell-prompt-lab-app",
  activeStrategy: "webContentsView",
  supportedStrategies: ["webContentsView"],
  sourceRequestId: "right_surface_plugin_2",
};

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  vi.clearAllMocks();
  embeddedBrowserMocks.isEmbeddedBrowserHostAvailable.mockReturnValue(true);
  pluginCatalogMocks.listInstalledPluginCatalog.mockResolvedValue({
    generatedAt: "now",
    plugins: [{ id: "demo-plugin", enabled: true, installed: true }],
  });
  mcpApiMocks.readResource.mockResolvedValue({
    uri: "ui://demo/report.html",
    mime_type: "text/html;profile=mcp-app",
    text: "<!doctype html><main>Plugin report</main>",
  });
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

async function renderSurface(
  props: Partial<React.ComponentProps<typeof WorkspacePluginSurface>> = {},
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  await act(async () => {
    root.render(<WorkspacePluginSurface surface={surface} {...props} />);
    await Promise.resolve();
    await Promise.resolve();
  });

  mountedRoots.push({ root, container });
  return container;
}

describe("WorkspacePluginSurface", () => {
  it("应通过 embedded browser host 挂载 Plugin entry URL", async () => {
    const container = await renderSurface();

    expect(
      container.querySelector('[data-testid="workspace-plugin-surface"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("内容工厂");
    expect(embeddedBrowserMocks.mountEmbeddedBrowserView).toHaveBeenCalledWith(
      expect.objectContaining({
        viewId: "plugin-surface-plugin-shell-content-factory-app-standalone",
        url: "http://127.0.0.1:4199/dashboard",
      }),
    );

    const mounted = mountedRoots.pop();
    act(() => {
      mounted?.root.unmount();
    });
    mounted?.container.remove();
    expect(
      embeddedBrowserMocks.destroyEmbeddedBrowserView,
    ).toHaveBeenCalledWith(
      "plugin-surface-plugin-shell-content-factory-app-standalone",
      embeddedBrowserMocks.mountEmbeddedBrowserView.mock.calls[0]?.[0].leaseId,
    );
  });

  it("Electron host 不可用时应 fail closed，不挂载 WebContentsView", async () => {
    embeddedBrowserMocks.isEmbeddedBrowserHostAvailable.mockReturnValue(false);
    const container = await renderSurface();
    const frame = container.querySelector(
      '[data-testid="workspace-plugin-surface-frame"]',
    );

    expect(container.textContent).toContain("宿主不可用");
    expect(frame?.getAttribute("data-host-available")).toBe("false");
    expect(frame?.getAttribute("data-mounted")).toBe("false");
    expect(
      embeddedBrowserMocks.mountEmbeddedBrowserView,
    ).not.toHaveBeenCalled();
  });

  it("同一 viewId 的实例应使用独立租约释放原生视图", async () => {
    const firstContainer = await renderSurface();
    const secondContainer = await renderSurface();
    expect(embeddedBrowserMocks.mountEmbeddedBrowserView).toHaveBeenCalledTimes(
      2,
    );

    const firstMounted = mountedRoots.find(
      (entry) => entry.container === firstContainer,
    );
    expect(firstMounted).toBeDefined();
    await act(async () => {
      firstMounted?.root.unmount();
      await Promise.resolve();
    });
    firstContainer.remove();
    const firstIndex = mountedRoots.findIndex(
      (entry) => entry.container === firstContainer,
    );
    if (firstIndex >= 0) {
      mountedRoots.splice(firstIndex, 1);
    }

    const mountLeases = embeddedBrowserMocks.mountEmbeddedBrowserView.mock.calls
      .map(([params]) => params.leaseId)
      .filter(Boolean);
    expect(new Set(mountLeases).size).toBe(2);
    expect(
      embeddedBrowserMocks.destroyEmbeddedBrowserView,
    ).toHaveBeenCalledWith(
      "plugin-surface-plugin-shell-content-factory-app-standalone",
      mountLeases[0],
    );

    const secondMounted = mountedRoots.find(
      (entry) => entry.container === secondContainer,
    );
    await act(async () => {
      secondMounted?.root.unmount();
      await Promise.resolve();
    });
    secondContainer.remove();
    const secondIndex = mountedRoots.findIndex(
      (entry) => entry.container === secondContainer,
    );
    if (secondIndex >= 0) {
      mountedRoots.splice(secondIndex, 1);
    }

    expect(
      embeddedBrowserMocks.destroyEmbeddedBrowserView,
    ).toHaveBeenCalledTimes(2);
    expect(
      embeddedBrowserMocks.destroyEmbeddedBrowserView,
    ).toHaveBeenCalledWith(
      "plugin-surface-plugin-shell-content-factory-app-standalone",
      mountLeases[1],
    );
  });

  it("应从 canonical MCP resource 读取并受控加载 MCP App HTML", async () => {
    const mcpSurface: WorkspacePluginSurfaceDescriptor = {
      appId: "demo-plugin",
      title: "demo-plugin",
      containerId: "mcp-app-item-1",
      activeStrategy: "webContentsView",
      supportedStrategies: ["webContentsView"],
      mcpApp: {
        resourceUri: "ui://demo/report.html",
        serverName: "plugin__demo__server",
        toolItemId: "item-1",
      },
    };

    await renderSurface({
      runtimeOwner: { sessionId: "session-1", threadId: "thread-1" },
      surface: mcpSurface,
    });

    expect(mcpApiMocks.readResource).toHaveBeenCalledWith(
      "plugin__demo__server",
      "ui://demo/report.html",
      { sessionId: "session-1", threadId: "thread-1" },
    );
    expect(
      embeddedBrowserMocks.loadEmbeddedBrowserViewHtml,
    ).toHaveBeenCalledWith(
      expect.objectContaining({
        html: "<!doctype html><main>Plugin report</main>",
        source: "mcpApp",
        sourceUri: "ui://demo/report.html",
        viewId: "plugin-surface-mcp-app-item-1",
      }),
    );
    expect(
      embeddedBrowserMocks.navigateEmbeddedBrowserView,
    ).not.toHaveBeenCalledWith(
      expect.objectContaining({ url: "ui://demo/report.html" }),
    );
  });

  it("Plugin 已卸载时应保留历史状态且不重启 MCP App", async () => {
    pluginCatalogMocks.listInstalledPluginCatalog.mockResolvedValueOnce({
      generatedAt: "now",
      plugins: [],
    });
    const mcpSurface: WorkspacePluginSurfaceDescriptor = {
      appId: "demo-plugin",
      title: "demo-plugin",
      containerId: "mcp-app-item-uninstalled",
      activeStrategy: "webContentsView",
      supportedStrategies: ["webContentsView"],
      mcpApp: {
        resourceUri: "ui://demo/report.html",
        serverName: "plugin__demo__server",
        toolItemId: "item-uninstalled",
      },
    };

    const container = await renderSurface({
      runtimeOwner: { sessionId: "session-1", threadId: "thread-1" },
      surface: mcpSurface,
    });
    const frame = container.querySelector(
      '[data-testid="workspace-plugin-surface-frame"]',
    );

    expect(frame?.getAttribute("data-plugin-availability")).toBe("uninstalled");
    expect(frame?.getAttribute("data-mounted")).toBe("false");
    expect(
      container.querySelector(
        '[data-testid="workspace-plugin-surface-history-unavailable"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("Plugin 已卸载");
    expect(container.textContent).toContain("不会自动重跑原操作");
    expect(
      embeddedBrowserMocks.mountEmbeddedBrowserView,
    ).not.toHaveBeenCalled();
    expect(mcpApiMocks.readResource).not.toHaveBeenCalled();
    expect(
      embeddedBrowserMocks.loadEmbeddedBrowserViewHtml,
    ).not.toHaveBeenCalled();
  });

  it("应在空 URL WebContentsView 异步挂载完成后读取 MCP App resource", async () => {
    let resolveMount:
      | ((value: {
          viewId: string;
          url: string;
          title: string;
          canGoBack: boolean;
          canGoForward: boolean;
          isLoading: boolean;
        }) => void)
      | undefined;
    embeddedBrowserMocks.mountEmbeddedBrowserView.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveMount = resolve;
        }),
    );
    const mcpSurface: WorkspacePluginSurfaceDescriptor = {
      appId: "demo-plugin",
      title: "demo-plugin",
      containerId: "mcp-app-item-async",
      activeStrategy: "webContentsView",
      supportedStrategies: ["webContentsView"],
      mcpApp: {
        resourceUri: "ui://demo/report.html",
        serverName: "plugin__demo__server",
        toolItemId: "item-async",
      },
    };

    await renderSurface({
      runtimeOwner: { sessionId: "session-1", threadId: "thread-1" },
      surface: mcpSurface,
    });

    const frame = document.querySelector(
      '[data-testid="workspace-plugin-surface-frame"]',
    );
    expect(frame).not.toBeNull();
    expect(frame?.getAttribute("data-host-available")).toBe("true");
    expect(frame?.getAttribute("data-mounted")).toBe("false");
    expect(frame?.getAttribute("data-runtime-session-id")).toBe("session-1");
    expect(frame?.getAttribute("data-runtime-thread-id")).toBe("thread-1");
    expect(frame?.getAttribute("data-mcp-server")).toBe("plugin__demo__server");
    expect(frame?.getAttribute("data-mcp-resource-uri")).toBe(
      "ui://demo/report.html",
    );
    expect(mcpApiMocks.readResource).not.toHaveBeenCalled();
    await act(async () => {
      resolveMount?.({
        viewId: "plugin-surface-mcp-app-item-async",
        url: "",
        title: "",
        canGoBack: false,
        canGoForward: false,
        isLoading: false,
      });
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(frame?.getAttribute("data-mounted")).toBe("true");
    expect(mcpApiMocks.readResource).toHaveBeenCalledWith(
      "plugin__demo__server",
      "ui://demo/report.html",
      { sessionId: "session-1", threadId: "thread-1" },
    );
    expect(
      embeddedBrowserMocks.loadEmbeddedBrowserViewHtml,
    ).toHaveBeenCalledWith(
      expect.objectContaining({
        sourceUri: "ui://demo/report.html",
        viewId: "plugin-surface-mcp-app-item-async",
      }),
    );
  });

  it("descriptor identity 变化时不应重复读取同一 MCP App resource", async () => {
    const mcpSurface: WorkspacePluginSurfaceDescriptor = {
      appId: "demo-plugin",
      title: "demo-plugin",
      containerId: "mcp-app-item-1",
      activeStrategy: "webContentsView",
      supportedStrategies: ["webContentsView"],
      mcpApp: {
        resourceUri: "ui://demo/report.html",
        serverName: "plugin__demo__server",
        toolItemId: "item-1",
      },
    };
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedRoots.push({ root, container });

    await act(async () => {
      root.render(<WorkspacePluginSurface surface={mcpSurface} />);
      await Promise.resolve();
      await Promise.resolve();
    });
    await act(async () => {
      root.render(
        <WorkspacePluginSurface
          surface={{
            ...mcpSurface,
            mcpApp: { ...mcpSurface.mcpApp! },
          }}
        />,
      );
      await Promise.resolve();
    });

    expect(mcpApiMocks.readResource).toHaveBeenCalledTimes(1);
    expect(
      embeddedBrowserMocks.loadEmbeddedBrowserViewHtml,
    ).toHaveBeenCalledTimes(1);
  });

  it("MCP App resource 读取期间重渲染不应取消同一 identity 的加载", async () => {
    let resolveResource:
      | ((value: { uri: string; mime_type: string; text: string }) => void)
      | undefined;
    mcpApiMocks.readResource.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveResource = resolve;
        }),
    );
    const mcpSurface: WorkspacePluginSurfaceDescriptor = {
      appId: "demo-plugin",
      title: "demo-plugin",
      containerId: "mcp-app-item-pending",
      activeStrategy: "webContentsView",
      supportedStrategies: ["webContentsView"],
      mcpApp: {
        resourceUri: "ui://demo/report.html",
        serverName: "plugin__demo__server",
        toolItemId: "item-pending",
      },
    };
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedRoots.push({ root, container });

    await act(async () => {
      root.render(<WorkspacePluginSurface surface={mcpSurface} />);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(mcpApiMocks.readResource).toHaveBeenCalledTimes(1);

    await act(async () => {
      root.render(
        <WorkspacePluginSurface
          surface={{ ...mcpSurface, title: "demo-plugin-app" }}
        />,
      );
      await Promise.resolve();
      resolveResource?.({
        uri: "ui://demo/report.html",
        mime_type: "text/html;profile=mcp-app",
        text: "<!doctype html><main>Plugin report</main>",
      });
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mcpApiMocks.readResource).toHaveBeenCalledTimes(1);
    expect(
      embeddedBrowserMocks.loadEmbeddedBrowserViewHtml,
    ).toHaveBeenCalledTimes(1);
  });

  it("应在右侧 appSurface 内渲染多个 Plugin 实例 tab，并支持聚焦和关闭", async () => {
    const onSelectSurface = vi.fn();
    const onCloseSurface = vi.fn();
    const container = await renderSurface({
      activeContainerId: promptLabSurface.containerId,
      surfaces: [surface, promptLabSurface],
      onSelectSurface,
      onCloseSurface,
    });

    expect(
      container.querySelector('[data-testid="workspace-plugin-surface-tabs"]'),
    ).not.toBeNull();
    expect(container.textContent).toContain("内容工厂");
    expect(container.textContent).toContain("提示词实验室");
    expect(embeddedBrowserMocks.mountEmbeddedBrowserView).toHaveBeenCalledWith(
      expect.objectContaining({
        viewId: "plugin-surface-plugin-shell-prompt-lab-app",
        url: "http://127.0.0.1:4201/",
      }),
    );

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          `[data-testid="workspace-plugin-surface-tab-${surface.containerId}"]`,
        )
        ?.click();
    });
    expect(onSelectSurface).toHaveBeenCalledWith(surface);

    act(() => {
      container
        .querySelector<HTMLButtonElement>(
          `[data-testid="workspace-plugin-surface-close-${promptLabSurface.containerId}"]`,
        )
        ?.click();
    });
    expect(onCloseSurface).toHaveBeenCalledWith(promptLabSurface);
  });

  it("切换 Plugin 实例时应保留已有 WebContentsView，不销毁重建", async () => {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mountedRoots.push({ root, container });

    await act(async () => {
      root.render(
        <WorkspacePluginSurface
          activeContainerId={promptLabSurface.containerId}
          surfaces={[surface, promptLabSurface]}
        />,
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(embeddedBrowserMocks.mountEmbeddedBrowserView).toHaveBeenCalledTimes(
      2,
    );
    embeddedBrowserMocks.mountEmbeddedBrowserView.mockClear();
    embeddedBrowserMocks.destroyEmbeddedBrowserView.mockClear();

    await act(async () => {
      root.render(
        <WorkspacePluginSurface
          activeContainerId={surface.containerId}
          surfaces={[surface, promptLabSurface]}
        />,
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(
      embeddedBrowserMocks.mountEmbeddedBrowserView,
    ).not.toHaveBeenCalled();
    expect(
      embeddedBrowserMocks.destroyEmbeddedBrowserView,
    ).not.toHaveBeenCalled();
    expect(
      embeddedBrowserMocks.setEmbeddedBrowserViewBounds,
    ).toHaveBeenCalledWith(
      expect.objectContaining({
        viewId: "plugin-surface-plugin-shell-prompt-lab-app",
        visible: false,
      }),
    );
    expect(
      container.querySelectorAll(
        '[data-testid="workspace-plugin-surface-frame"]',
      ),
    ).toHaveLength(2);
  });
});
