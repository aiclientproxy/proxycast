import { describe, expect, it, vi } from "vitest";
import { ElectronEmbeddedBrowserHost } from "./embeddedBrowserHost";

const {
  addChildViewMock,
  closeMock,
  findInPageMock,
  getSessionUserAgentMock,
  getZoomFactorMock,
  loadUrlMock,
  menuBuildFromTemplateMock,
  menuPopupMock,
  reloadMock,
  removeChildViewMock,
  sessionEventHandlers,
  sessionFromPartitionMock,
  setPermissionRequestHandlerMock,
  setBoundsMock,
  setSessionUserAgentMock,
  setWebContentsUserAgentMock,
  setWindowOpenHandlerMock,
  setZoomFactorMock,
  shellOpenExternalMock,
  stopMock,
  stopFindInPageMock,
  setVisibleMock,
  webContentsEventHandlers,
  webContentsState,
  webContentsViewCtorMock,
  clipboardWriteTextMock,
  copyImageAtMock,
  downloadUrlMock,
} = vi.hoisted(() => {
  const webContentsState = {
    destroyed: false,
    isLoading: false,
    title: "Example",
    url: "https://example.com/",
  };
  const webContentsEventHandlers = new Map<
    string,
    Array<(...args: unknown[]) => void>
  >();
  const sessionEventHandlers = new Map<
    string,
    Array<(...args: unknown[]) => void>
  >();
  const setPermissionRequestHandlerMock = vi.fn();
  const loadUrlMock = vi.fn(async () => undefined);
  const reloadMock = vi.fn();
  const stopMock = vi.fn();
  const findInPageMock = vi.fn(() => 7);
  const stopFindInPageMock = vi.fn();
  const getZoomFactorMock = vi.fn(() => 1);
  const setZoomFactorMock = vi.fn();
  const clipboardWriteTextMock = vi.fn();
  const closeMock = vi.fn();
  const copyImageAtMock = vi.fn();
  const downloadUrlMock = vi.fn();
  const getSessionUserAgentMock = vi.fn();
  const setSessionUserAgentMock = vi.fn();
  const sessionFromPartitionMock = vi.fn(() => ({
    getUserAgent: getSessionUserAgentMock,
    on: vi.fn((eventName: string, handler: (...args: unknown[]) => void) => {
      const handlers = sessionEventHandlers.get(eventName) || [];
      handlers.push(handler);
      sessionEventHandlers.set(eventName, handlers);
    }),
    setPermissionRequestHandler: setPermissionRequestHandlerMock,
    setUserAgent: setSessionUserAgentMock,
  }));
  const setWebContentsUserAgentMock = vi.fn();
  const setWindowOpenHandlerMock = vi.fn();
  const shellOpenExternalMock = vi.fn();
  const setBoundsMock = vi.fn();
  const setVisibleMock = vi.fn();
  const menuPopupMock = vi.fn();
  const menuBuildFromTemplateMock = vi.fn(() => ({
    popup: menuPopupMock,
  }));
  return {
    addChildViewMock: vi.fn(),
    closeMock,
    clipboardWriteTextMock,
    copyImageAtMock,
    downloadUrlMock,
    findInPageMock,
    getSessionUserAgentMock,
    getZoomFactorMock,
    loadUrlMock,
    reloadMock,
    removeChildViewMock: vi.fn(),
    sessionEventHandlers,
    sessionFromPartitionMock,
    setPermissionRequestHandlerMock,
    setBoundsMock,
    setSessionUserAgentMock,
    setWebContentsUserAgentMock,
    setZoomFactorMock,
    setVisibleMock,
    menuBuildFromTemplateMock,
    menuPopupMock,
    webContentsEventHandlers,
    webContentsState,
    webContentsViewCtorMock: vi.fn(() => ({
      setBackgroundColor: vi.fn(),
      setBounds: setBoundsMock,
      setVisible: setVisibleMock,
      webContents: {
        close: closeMock,
        copyImageAt: copyImageAtMock,
        downloadURL: downloadUrlMock,
        findInPage: findInPageMock,
        getTitle: () => webContentsState.title,
        getURL: () => webContentsState.url,
        getZoomFactor: getZoomFactorMock,
        isDestroyed: () => webContentsState.destroyed,
        isLoading: () => webContentsState.isLoading,
        loadURL: loadUrlMock,
        navigationHistory: {
          canGoBack: () => false,
          canGoForward: () => false,
          goBack: vi.fn(),
          goForward: vi.fn(),
        },
        on: vi.fn(
          (eventName: string, handler: (...args: unknown[]) => void) => {
            const handlers = webContentsEventHandlers.get(eventName) || [];
            handlers.push(handler);
            webContentsEventHandlers.set(eventName, handlers);
          },
        ),
        off: vi.fn(
          (eventName: string, handler: (...args: unknown[]) => void) => {
            const handlers = webContentsEventHandlers.get(eventName) || [];
            webContentsEventHandlers.set(
              eventName,
              handlers.filter((item) => item !== handler),
            );
          },
        ),
        reload: reloadMock,
        setZoomFactor: setZoomFactorMock,
        stop: stopMock,
        stopFindInPage: stopFindInPageMock,
        setUserAgent: setWebContentsUserAgentMock,
        setWindowOpenHandler: setWindowOpenHandlerMock,
      },
    })),
    setWindowOpenHandlerMock,
    shellOpenExternalMock,
    stopMock,
    stopFindInPageMock,
  };
});

vi.mock("./electronRuntime", () => ({
  app: {
    getLocale: () => "zh-CN",
  },
  clipboard: {
    writeText: clipboardWriteTextMock,
  },
  Menu: {
    buildFromTemplate: menuBuildFromTemplateMock,
  },
  session: {
    fromPartition: sessionFromPartitionMock,
  },
  shell: {
    openExternal: shellOpenExternalMock,
  },
  WebContentsView: webContentsViewCtorMock,
}));

function createWindow() {
  let destroyed = false;
  const windowEventHandlers = new Map<
    string,
    Array<(...args: unknown[]) => void>
  >();
  const windowWebContentsEventHandlers = new Map<
    string,
    Array<(...args: unknown[]) => void>
  >();
  return {
    contentView: {
      addChildView: addChildViewMock,
      removeChildView: removeChildViewMock,
    },
    destroyForTest: () => {
      destroyed = true;
    },
    emitWebContentsEventForTest: (eventName: string, ...args: unknown[]) => {
      for (const handler of windowWebContentsEventHandlers.get(eventName) ||
        []) {
        handler(...args);
      }
    },
    emitWindowEventForTest: (eventName: string, ...args: unknown[]) => {
      for (const handler of windowEventHandlers.get(eventName) || []) {
        handler(...args);
      }
    },
    isDestroyed: () => destroyed,
    off: vi.fn((eventName: string, handler: (...args: unknown[]) => void) => {
      const handlers = windowEventHandlers.get(eventName) || [];
      windowEventHandlers.set(
        eventName,
        handlers.filter((item) => item !== handler),
      );
    }),
    on: vi.fn((eventName: string, handler: (...args: unknown[]) => void) => {
      const handlers = windowEventHandlers.get(eventName) || [];
      handlers.push(handler);
      windowEventHandlers.set(eventName, handlers);
    }),
    webContents: {
      off: vi.fn((eventName: string, handler: (...args: unknown[]) => void) => {
        const handlers = windowWebContentsEventHandlers.get(eventName) || [];
        windowWebContentsEventHandlers.set(
          eventName,
          handlers.filter((item) => item !== handler),
        );
      }),
      on: vi.fn((eventName: string, handler: (...args: unknown[]) => void) => {
        const handlers = windowWebContentsEventHandlers.get(eventName) || [];
        handlers.push(handler);
        windowWebContentsEventHandlers.set(eventName, handlers);
      }),
    },
  };
}

function emitWebContentsEvent(eventName: string, ...args: unknown[]) {
  for (const handler of webContentsEventHandlers.get(eventName) || []) {
    handler(...args);
  }
}

describe("ElectronEmbeddedBrowserHost", () => {
  function resetMocks() {
    addChildViewMock.mockClear();
    closeMock.mockClear();
    clipboardWriteTextMock.mockClear();
    copyImageAtMock.mockClear();
    downloadUrlMock.mockClear();
    loadUrlMock.mockReset();
    loadUrlMock.mockResolvedValue(undefined);
    reloadMock.mockClear();
    stopMock.mockClear();
    findInPageMock.mockClear();
    stopFindInPageMock.mockClear();
    getZoomFactorMock.mockClear();
    getZoomFactorMock.mockReturnValue(1);
    setZoomFactorMock.mockClear();
    removeChildViewMock.mockClear();
    getSessionUserAgentMock.mockReset();
    getSessionUserAgentMock.mockReturnValue(
      "Mozilla/5.0 AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36 Electron/31.0.0 Lime/1.78.0",
    );
    sessionFromPartitionMock.mockClear();
    setPermissionRequestHandlerMock.mockClear();
    setBoundsMock.mockClear();
    setSessionUserAgentMock.mockClear();
    setWebContentsUserAgentMock.mockClear();
    setWindowOpenHandlerMock.mockClear();
    shellOpenExternalMock.mockClear();
    setVisibleMock.mockClear();
    menuBuildFromTemplateMock.mockClear();
    menuPopupMock.mockClear();
    webContentsViewCtorMock.mockClear();
    webContentsState.destroyed = false;
    webContentsState.isLoading = false;
    webContentsState.title = "Example";
    webContentsState.url = "https://example.com/";
    sessionEventHandlers.clear();
    webContentsEventHandlers.clear();
  }

  it("通过 WebContentsView 挂载、导航和销毁内嵌浏览器", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await expect(
      host.invoke(window as never, "embedded_browser_view_mount", {
        viewId: "browser-1",
        url: "https://example.com",
        bounds: { x: 10, y: 20, width: 300, height: 200 },
      }),
    ).resolves.toMatchObject({
      viewId: "browser-1",
      url: "https://example.com/",
    });

    expect(webContentsViewCtorMock).toHaveBeenCalledWith(
      expect.objectContaining({
        webPreferences: expect.objectContaining({
          contextIsolation: true,
          nodeIntegration: false,
          partition: "persist:embedded-browser",
          sandbox: true,
          webSecurity: true,
        }),
      }),
    );
    const normalizedUserAgent =
      "Mozilla/5.0 AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36";
    expect(sessionFromPartitionMock).toHaveBeenCalledWith(
      "persist:embedded-browser",
      { cache: true },
    );
    expect(setSessionUserAgentMock).toHaveBeenCalledWith(
      normalizedUserAgent,
      "zh-CN,zh,en-US,en",
    );
    expect(setWebContentsUserAgentMock).toHaveBeenCalledWith(
      normalizedUserAgent,
    );
    expect(addChildViewMock).toHaveBeenCalled();
    expect(setBoundsMock).toHaveBeenCalledWith({
      x: 10,
      y: 20,
      width: 300,
      height: 200,
    });
    expect(loadUrlMock).toHaveBeenCalledWith("https://example.com/");

    await host.invoke(window as never, "embedded_browser_view_navigate", {
      viewId: "browser-1",
      url: "https://openai.com",
    });
    expect(loadUrlMock).toHaveBeenCalledWith("https://openai.com/");

    await host.invoke(window as never, "embedded_browser_view_reload", {
      viewId: "browser-1",
    });
    expect(reloadMock).toHaveBeenCalled();

    webContentsState.isLoading = true;
    await host.invoke(window as never, "embedded_browser_view_stop", {
      viewId: "browser-1",
    });
    expect(stopMock).toHaveBeenCalled();

    await host.invoke(window as never, "embedded_browser_view_destroy", {
      viewId: "browser-1",
    });
    expect(removeChildViewMock).toHaveBeenCalled();
    expect(closeMock).toHaveBeenCalled();
    expect(
      emitted.some((item) => item.event === "embedded-browser-view-state"),
    ).toBe(true);
  });

  it("通过受控命令加载 MCP App HTML，并保持 ui source URI", async () => {
    resetMocks();
    const host = new ElectronEmbeddedBrowserHost();
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "mcp-app-1",
      bounds: { x: 0, y: 0, width: 480, height: 640 },
    });
    await expect(
      host.invoke(window as never, "embedded_browser_view_load_html", {
        viewId: "mcp-app-1",
        html: "<!doctype html><title>Plugin report</title>",
        source: "mcpApp",
        sourceUri: "ui://plugin/report.html",
      }),
    ).resolves.toMatchObject({
      viewId: "mcp-app-1",
      url: "ui://plugin/report.html",
    });

    expect(loadUrlMock).toHaveBeenLastCalledWith(
      expect.stringMatching(/^data:text\/html;charset=utf-8;base64,/),
    );
  });

  it("受控 HTML 命令拒绝普通网页来源，普通导航仍拒绝 ui/data", async () => {
    resetMocks();
    const host = new ElectronEmbeddedBrowserHost();
    const window = createWindow();

    await expect(
      host.invoke(window as never, "embedded_browser_view_load_html", {
        viewId: "mcp-app-1",
        html: "<p>bad</p>",
        source: "browser",
        sourceUri: "ui://plugin/report.html",
      }),
    ).rejects.toThrow("source 必须是 mcpApp");
    await expect(
      host.invoke(window as never, "embedded_browser_view_navigate", {
        viewId: "mcp-app-1",
        url: "ui://plugin/report.html",
      }),
    ).rejects.toThrow("只支持 http/https");
    await expect(
      host.invoke(window as never, "embedded_browser_view_navigate", {
        viewId: "mcp-app-1",
        url: "data:text/html,unsafe",
      }),
    ).rejects.toThrow("只支持 http/https");
  });

  it("MCP App 加载后拒绝页面内 http 导航和新窗口", async () => {
    resetMocks();
    const host = new ElectronEmbeddedBrowserHost();
    const window = createWindow();
    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "mcp-app-1",
    });
    await host.invoke(window as never, "embedded_browser_view_load_html", {
      viewId: "mcp-app-1",
      html: '<a href="https://example.com">blocked</a>',
      source: "mcpApp",
      sourceUri: "ui://plugin/report.html",
    });

    const preventDefault = vi.fn();
    emitWebContentsEvent(
      "will-navigate",
      { preventDefault },
      "https://example.com/",
    );
    expect(preventDefault).toHaveBeenCalledOnce();
    const openHandler = setWindowOpenHandlerMock.mock.calls.at(-1)?.[0];
    expect(openHandler?.({ url: "https://example.com/" })).toEqual({
      action: "deny",
    });
    expect(loadUrlMock).toHaveBeenCalledTimes(1);
  });

  it("host dispose 不访问已销毁 BrowserWindow", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });

    window.destroyForTest();

    expect(() => host.dispose()).not.toThrow();
    expect(removeChildViewMock).not.toHaveBeenCalled();
    expect(closeMock).toHaveBeenCalled();
    expect(emitted).toContainEqual({
      event: "embedded-browser-view-destroyed",
      payload: {
        viewId: "browser-1",
        reason: "host-dispose",
        requestedLeaseId: null,
        remainingLeaseCount: 0,
      },
    });
  });

  it("窗口 closed 事件应记录 window-closed 销毁原因", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      leaseId: "lease-1",
    });
    window.destroyForTest();
    window.emitWindowEventForTest("closed");

    expect(closeMock).toHaveBeenCalledOnce();
    expect(emitted).toContainEqual({
      event: "embedded-browser-view-destroyed",
      payload: {
        viewId: "browser-1",
        reason: "window-closed",
        requestedLeaseId: null,
        remainingLeaseCount: 1,
      },
    });
    expect(() => host.dispose()).not.toThrow();
    expect(closeMock).toHaveBeenCalledOnce();
  });

  it("陈旧租约释放不得销毁同 viewId 的当前原生视图", async () => {
    resetMocks();
    const host = new ElectronEmbeddedBrowserHost();
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "plugin-surface-1",
      leaseId: "lease-old",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });
    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "plugin-surface-1",
      leaseId: "lease-current",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });

    removeChildViewMock.mockClear();
    closeMock.mockClear();
    await host.invoke(window as never, "embedded_browser_view_destroy", {
      viewId: "plugin-surface-1",
      leaseId: "lease-old",
    });

    expect(removeChildViewMock).not.toHaveBeenCalled();
    expect(closeMock).not.toHaveBeenCalled();

    await host.invoke(window as never, "embedded_browser_view_destroy", {
      viewId: "plugin-surface-1",
      leaseId: "lease-current",
    });

    expect(removeChildViewMock).toHaveBeenCalledTimes(1);
    expect(closeMock).toHaveBeenCalledTimes(1);
  });

  it("只在主窗口开始完整导航时销毁内嵌浏览器原生视图", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });

    removeChildViewMock.mockClear();
    closeMock.mockClear();
    window.emitWebContentsEventForTest("did-start-loading");
    window.emitWebContentsEventForTest(
      "did-start-navigation",
      {},
      "http://127.0.0.1:1420/#settings",
      true,
      true,
    );
    window.emitWebContentsEventForTest(
      "did-start-navigation",
      {},
      "https://example.com/frame",
      false,
      false,
    );

    expect(removeChildViewMock).not.toHaveBeenCalled();
    expect(closeMock).not.toHaveBeenCalled();

    window.emitWebContentsEventForTest(
      "did-start-navigation",
      {},
      "http://127.0.0.1:1420/",
      false,
      true,
    );

    expect(removeChildViewMock).toHaveBeenCalled();
    expect(closeMock).toHaveBeenCalled();
    expect(emitted).toContainEqual({
      event: "embedded-browser-view-destroyed",
      payload: {
        viewId: "browser-1",
        reason: "renderer-navigation",
        requestedLeaseId: null,
        remainingLeaseCount: 0,
      },
    });
  });

  it("WebContents 已销毁时不会重复 close", async () => {
    resetMocks();
    const host = new ElectronEmbeddedBrowserHost();
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });

    webContentsState.destroyed = true;

    await expect(
      host.invoke(window as never, "embedded_browser_view_destroy", {
        viewId: "browser-1",
      }),
    ).resolves.toEqual({});
    expect(removeChildViewMock).toHaveBeenCalled();
    expect(closeMock).not.toHaveBeenCalled();
  });

  it("原生 WebContents destroyed 时会发出 Browser 关闭终态", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 0, y: 0, width: 300, height: 200 },
    });

    webContentsState.destroyed = true;
    emitWebContentsEvent("destroyed");

    expect(emitted).toContainEqual({
      event: "embedded-browser-view-destroyed",
      payload: expect.objectContaining({
        reason: "webcontents-destroyed",
        viewId: "browser-1",
      }),
    });
    expect(removeChildViewMock).toHaveBeenCalled();
    expect(closeMock).not.toHaveBeenCalled();
  });

  it("渲染进程退出时同样会回收 Browser route", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 0, y: 0, width: 300, height: 200 },
    });

    webContentsState.destroyed = true;
    emitWebContentsEvent("render-process-gone", {}, { reason: "crashed" });

    expect(emitted).toContainEqual({
      event: "embedded-browser-view-destroyed",
      payload: expect.objectContaining({
        reason: "webcontents-destroyed",
        viewId: "browser-1",
      }),
    });
  });

  it("拒绝非 http/https 地址", async () => {
    resetMocks();
    const host = new ElectronEmbeddedBrowserHost();
    await expect(
      host.invoke(createWindow() as never, "embedded_browser_view_mount", {
        viewId: "browser-1",
        url: "file:///tmp/token",
      }),
    ).rejects.toThrow("内嵌浏览器只支持 http/https 地址");
  });

  it("挂载外部网址时只发起导航，不等待页面加载完成", async () => {
    resetMocks();
    webContentsState.url = "";
    webContentsState.isLoading = true;
    loadUrlMock.mockReturnValue(new Promise(() => undefined));
    const host = new ElectronEmbeddedBrowserHost();
    const window = createWindow();

    await expect(
      host.invoke(window as never, "embedded_browser_view_mount", {
        viewId: "browser-1",
        url: "https://google.com",
      }),
    ).resolves.toMatchObject({
      viewId: "browser-1",
      url: "https://google.com/",
      isLoading: true,
    });

    expect(loadUrlMock).toHaveBeenCalledWith("https://google.com/");
  });

  it("同步 favicon、加载进度、页内查找和单标签缩放状态", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    webContentsState.url = "";
    webContentsState.isLoading = true;
    loadUrlMock.mockReturnValue(new Promise(() => undefined));
    setZoomFactorMock.mockImplementation((nextZoom: number) => {
      getZoomFactorMock.mockReturnValue(nextZoom);
    });
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await expect(
      host.invoke(window as never, "embedded_browser_view_mount", {
        viewId: "browser-1",
        url: "https://example.com",
      }),
    ).resolves.toMatchObject({
      viewId: "browser-1",
      loadProgress: 0.1,
      zoomFactor: 1,
    });

    emitWebContentsEvent("page-favicon-updated", {}, [
      "https://example.com/favicon.ico",
    ]);
    await host.invoke(window as never, "embedded_browser_view_find_in_page", {
      viewId: "browser-1",
      text: "Example",
    });
    expect(findInPageMock).toHaveBeenCalledWith("Example", {
      forward: true,
      findNext: false,
    });
    emitWebContentsEvent(
      "found-in-page",
      {},
      {
        requestId: 7,
        activeMatchOrdinal: 2,
        matches: 5,
        finalUpdate: true,
      },
    );

    await host.invoke(window as never, "embedded_browser_view_set_zoom", {
      viewId: "browser-1",
      zoomFactor: 1.25,
    });
    await host.invoke(
      window as never,
      "embedded_browser_view_stop_find_in_page",
      {
        viewId: "browser-1",
      },
    );

    expect(setZoomFactorMock).toHaveBeenCalledWith(1.25);
    expect(stopFindInPageMock).toHaveBeenCalledWith("clearSelection");
    expect(
      emitted.some(
        (item) =>
          item.event === "embedded-browser-view-state" &&
          (item.payload as { faviconUrl?: string }).faviconUrl ===
            "https://example.com/favicon.ico",
      ),
    ).toBe(true);
    expect(
      emitted.some(
        (item) =>
          item.event === "embedded-browser-view-state" &&
          (item.payload as { find?: { matches?: number } }).find?.matches === 5,
      ),
    ).toBe(true);
    expect(
      emitted.find(
        (item) =>
          item.event === "embedded-browser-view-state" &&
          (item.payload as { zoomFactor?: number }).zoomFactor === 1.25,
      ),
    ).toBeTruthy();
  });

  it("更新 bounds 时应按 visible 参数隐藏或恢复 WebContentsView", async () => {
    resetMocks();
    const host = new ElectronEmbeddedBrowserHost();
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });
    setVisibleMock.mockClear();

    await host.invoke(window as never, "embedded_browser_view_set_bounds", {
      viewId: "browser-1",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
      visible: false,
    });
    expect(setBoundsMock).toHaveBeenLastCalledWith({
      x: 10,
      y: 20,
      width: 300,
      height: 200,
    });
    expect(setVisibleMock).toHaveBeenLastCalledWith(false);

    await host.invoke(window as never, "embedded_browser_view_set_bounds", {
      viewId: "browser-1",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
      visible: true,
    });
    expect(setVisibleMock).toHaveBeenLastCalledWith(true);
  });

  it("导航失败时不让 IPC reject，并通过失败事件回传", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    loadUrlMock.mockRejectedValue(new Error("navigation failed"));
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await expect(
      host.invoke(window as never, "embedded_browser_view_navigate", {
        viewId: "browser-1",
        url: "https://blocked.example",
      }),
    ).resolves.toMatchObject({
      viewId: "browser-1",
      url: "https://blocked.example/",
      isLoading: true,
    });

    await vi.waitFor(() => {
      expect(
        emitted.find(
          (item) => item.event === "embedded-browser-view-load-failed",
        ),
      ).toMatchObject({
        payload: expect.objectContaining({
          viewId: "browser-1",
          url: "https://blocked.example/",
          errorCode: null,
          errorDescription: "navigation failed",
          failureCategory: "load_failed",
        }),
      });
    });
  });

  it("同一次导航的 did-fail-load 与 loadURL reject 不应重复回传失败事件", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    let rejectNavigation: (error: Error) => void = () => undefined;
    loadUrlMock.mockReturnValue(
      new Promise<undefined>((_resolve, reject) => {
        rejectNavigation = reject;
      }),
    );
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_navigate", {
      viewId: "browser-1",
      url: "https://blocked.example",
    });

    emitWebContentsEvent(
      "did-fail-load",
      {},
      -105,
      "NAME_NOT_RESOLVED",
      "https://blocked.example/",
    );
    rejectNavigation(new Error("navigation failed"));
    await Promise.resolve();
    await Promise.resolve();

    expect(
      emitted.filter(
        (item) => item.event === "embedded-browser-view-load-failed",
      ),
    ).toHaveLength(1);
    expect(
      emitted.find(
        (item) => item.event === "embedded-browser-view-load-failed",
      ),
    ).toMatchObject({
      payload: expect.objectContaining({
        url: "https://blocked.example/",
        errorCode: -105,
        errorDescription: "NAME_NOT_RESOLVED",
        failureCategory: "dns",
      }),
    });
  });

  it("按 Electron 加载失败原因分类错误页", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_navigate", {
      viewId: "browser-1",
      url: "https://cert.example",
    });
    emitWebContentsEvent(
      "did-fail-load",
      {},
      -202,
      "CERT_AUTHORITY_INVALID",
      "https://cert.example/",
    );

    await host.invoke(window as never, "embedded_browser_view_navigate", {
      viewId: "browser-1",
      url: "https://blocked.example",
    });
    emitWebContentsEvent(
      "did-fail-load",
      {},
      -20,
      "ERR_BLOCKED_BY_CLIENT",
      "https://blocked.example/",
    );

    await host.invoke(window as never, "embedded_browser_view_navigate", {
      viewId: "browser-1",
      url: "https://stopped.example",
    });
    emitWebContentsEvent(
      "did-fail-load",
      {},
      -3,
      "ERR_ABORTED",
      "https://stopped.example/",
    );

    expect(loadFailureFor(emitted, "https://cert.example/")).toMatchObject({
      failureCategory: "tls",
    });
    expect(loadFailureFor(emitted, "https://blocked.example/")).toMatchObject({
      failureCategory: "blocked",
    });
    expect(loadFailureFor(emitted, "https://stopped.example/")).toMatchObject({
      failureCategory: "aborted",
    });
  });

  it("target blank 应留在内嵌浏览器内导航，不弹系统浏览器", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });
    loadUrlMock.mockClear();

    const handler = setWindowOpenHandlerMock.mock.calls[0]?.[0] as
      | ((details: { url: string }) => { action: string })
      | undefined;
    expect(handler).toBeTypeOf("function");
    expect(handler?.({ url: "https://example.com/new" })).toEqual({
      action: "deny",
    });

    expect(shellOpenExternalMock).not.toHaveBeenCalled();
    expect(loadUrlMock).toHaveBeenCalledWith("https://example.com/new");
    expect(
      emitted.find(
        (item) =>
          item.event === "embedded-browser-view-state" &&
          (item.payload as { url?: string }).url === "https://example.com/new",
      ),
    ).toBeTruthy();
  });

  it("右键链接和图片时显示受控菜单，不弹出系统浏览器", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });
    loadUrlMock.mockClear();

    emitWebContentsEvent(
      "context-menu",
      {},
      {
        x: 12,
        y: 24,
        pageURL: "https://example.com/page",
        linkURL: "https://example.com/link",
        mediaType: "image",
        srcURL: "https://example.com/image.png",
      },
    );

    expect(menuBuildFromTemplateMock).toHaveBeenCalled();
    expect(menuPopupMock).toHaveBeenCalledWith({ window });
    const menuCalls = menuBuildFromTemplateMock.mock.calls as unknown as Array<
      [
        Array<{
          label?: string;
          click?: () => void;
        }>,
      ]
    >;
    const template = menuCalls[0]?.[0] ?? [];
    expect(template.map((item) => item.label).filter(Boolean)).toEqual(
      expect.arrayContaining([
        "在当前标签页打开链接",
        "在系统浏览器中打开链接",
        "复制链接地址",
        "复制图片",
        "复制图片地址",
        "图片另存为",
      ]),
    );

    template.find((item) => item.label === "在当前标签页打开链接")?.click?.();
    expect(loadUrlMock).toHaveBeenCalledWith("https://example.com/link");
    expect(shellOpenExternalMock).not.toHaveBeenCalled();
    expect(
      emitted.find(
        (item) =>
          item.event === "embedded-browser-view-state" &&
          (item.payload as { url?: string }).url === "https://example.com/link",
      ),
    ).toBeTruthy();

    template.find((item) => item.label === "复制链接地址")?.click?.();
    expect(clipboardWriteTextMock).toHaveBeenCalledWith(
      "https://example.com/link",
    );
    template.find((item) => item.label === "复制图片")?.click?.();
    expect(copyImageAtMock).toHaveBeenCalledWith(12, 24);
    template.find((item) => item.label === "图片另存为")?.click?.();
    expect(downloadUrlMock).toHaveBeenCalledWith(
      "https://example.com/image.png",
    );
  });

  it("下载事件应按 viewId 回传进度和完成状态，不暴露本地保存路径", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });

    const item = createDownloadItem();
    const webContents = webContentsViewCtorMock.mock.results[0]?.value
      .webContents as unknown;
    emitSessionEvent("will-download", {}, item, webContents);
    item.receivedBytes = 50;
    item.emit("updated", {}, "progressing");
    item.receivedBytes = 100;
    item.emit("done", {}, "completed");

    const downloads = emitted.filter(
      (entry) => entry.event === "embedded-browser-view-download",
    );
    expect(downloads).toHaveLength(3);
    expect(downloads[0]?.payload).toMatchObject({
      viewId: "browser-1",
      url: "https://example.com/report.pdf",
      filename: "report.pdf",
      mimeType: "application/pdf",
      state: "started",
      receivedBytes: 0,
      totalBytes: 100,
      canResume: false,
    });
    expect(downloads[1]?.payload).toMatchObject({
      state: "progressing",
      receivedBytes: 50,
    });
    expect(downloads[2]?.payload).toMatchObject({
      state: "completed",
      receivedBytes: 100,
    });
    expect(JSON.stringify(downloads)).not.toContain("/Users/");
  });

  it("页面权限请求默认拒绝，并向 Renderer 回传受控事件", async () => {
    resetMocks();
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const host = new ElectronEmbeddedBrowserHost((event, payload) => {
      emitted.push({ event, payload });
    });
    const window = createWindow();

    await host.invoke(window as never, "embedded_browser_view_mount", {
      viewId: "browser-1",
      url: "https://example.com",
      bounds: { x: 10, y: 20, width: 300, height: 200 },
    });

    const permissionHandler = setPermissionRequestHandlerMock.mock
      .calls[0]?.[0] as
      | ((
          webContents: unknown,
          permission: string,
          callback: (allowed: boolean) => void,
          details: {
            requestingUrl?: string;
            embeddingOrigin?: string;
          },
        ) => void)
      | undefined;
    expect(permissionHandler).toBeTypeOf("function");
    const callback = vi.fn();
    const webContents = webContentsViewCtorMock.mock.results[0]?.value
      .webContents as unknown;

    permissionHandler?.(webContents, "media", callback, {
      requestingUrl: "https://example.com/camera",
      embeddingOrigin: "https://example.com",
    });

    expect(callback).toHaveBeenCalledWith(false);
    expect(
      emitted.find(
        (entry) => entry.event === "embedded-browser-view-permission-request",
      ),
    ).toMatchObject({
      payload: expect.objectContaining({
        viewId: "browser-1",
        permission: "media",
        url: "https://example.com/",
        requestingUrl: "https://example.com/camera",
        embeddingOrigin: "https://example.com",
        decision: "blocked",
      }),
    });
  });
});

function loadFailureFor(
  emitted: Array<{ event: string; payload?: unknown }>,
  url: string,
) {
  return emitted.find(
    (item) =>
      item.event === "embedded-browser-view-load-failed" &&
      (item.payload as { url?: string }).url === url,
  )?.payload;
}

function emitSessionEvent(eventName: string, ...args: unknown[]) {
  for (const handler of sessionEventHandlers.get(eventName) || []) {
    handler(...args);
  }
}

function createDownloadItem() {
  const handlers = new Map<string, Array<(...args: unknown[]) => void>>();
  return {
    receivedBytes: 0,
    canResume: () => false,
    getFilename: () => "report.pdf",
    getMimeType: () => "application/pdf",
    getReceivedBytes() {
      return this.receivedBytes;
    },
    getTotalBytes: () => 100,
    getURL: () => "https://example.com/report.pdf",
    on(eventName: string, handler: (...args: unknown[]) => void) {
      const nextHandlers = handlers.get(eventName) || [];
      nextHandlers.push(handler);
      handlers.set(eventName, nextHandlers);
    },
    once(eventName: string, handler: (...args: unknown[]) => void) {
      this.on(eventName, handler);
    },
    emit(eventName: string, ...args: unknown[]) {
      for (const handler of handlers.get(eventName) || []) {
        handler(...args);
      }
    },
  };
}
