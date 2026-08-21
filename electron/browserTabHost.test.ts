import { describe, expect, it, vi } from "vitest";
import { ElectronBrowserTabHost, type BrowserToolCall } from "./browserTabHost";

function createHarness() {
  let nextWebContentsId = 100;
  const views = new Map<string, ReturnType<typeof createNativeView>>();
  const events: Array<{ event: string; payload: unknown }> = [];
  const embeddedHost = {
    invoke: vi.fn(async (window, command: string, args) => {
      const viewId = String(args?.viewId ?? "");
      if (command === "embedded_browser_view_destroy") {
        views.delete(viewId);
        return {};
      }
      let native = views.get(viewId);
      if (command === "embedded_browser_view_mount") {
        native ??= createNativeView(
          window,
          viewId,
          nextWebContentsId++,
          String(args?.url ?? "https://example.com/"),
        );
        views.set(viewId, native);
      }
      if (!native) {
        throw new Error(`missing native view: ${viewId}`);
      }
      if (command === "embedded_browser_view_navigate") {
        native.webContents.url = String(args?.url);
      }
      return state(native);
    }),
    resolveNativeView: vi.fn((viewId: string) => {
      const native = views.get(viewId);
      return native
        ? {
            state: state(native),
            view: native.view,
            window: native.window,
          }
        : null;
    }),
  };
  const window = {
    id: 7,
    isDestroyed: () => false,
    webContents: { id: 41 },
  };
  const browserHost = new ElectronBrowserTabHost(
    embeddedHost as never,
    (event, payload) => events.push({ event, payload }),
  );
  const host = {
    connectionLost: browserHost.connectionLost.bind(browserHost),
    executeTool: (
      call: Omit<BrowserToolCall, "phase"> &
        Partial<Pick<BrowserToolCall, "phase">>,
    ) => browserHost.executeTool({ phase: "preflight", ...call }),
    invoke: browserHost.invoke.bind(browserHost),
    observeEmbeddedEvent: browserHost.observeEmbeddedEvent.bind(browserHost),
    setTurnInterruptHandler:
      browserHost.setTurnInterruptHandler.bind(browserHost),
    turnEnded: browserHost.turnEnded.bind(browserHost),
  };
  return { embeddedHost, events, host, views, window };
}

function createNativeView(
  window: { id: number; webContents: { id: number } },
  viewId: string,
  webContentsId: number,
  url: string,
) {
  const listeners = new Map<string, Set<(...args: unknown[]) => void>>();
  const emit = (event: string, ...args: unknown[]) => {
    for (const listener of listeners.get(event) ?? []) {
      listener(...args);
    }
  };
  let debuggerAttached = false;
  const debuggerApi = {
    attach: vi.fn(() => {
      debuggerAttached = true;
    }),
    detach: vi.fn(() => {
      debuggerAttached = false;
    }),
    isAttached: vi.fn(() => debuggerAttached),
    sendCommand: vi.fn(
      async (method: string, params?: Record<string, unknown>) => {
        if (method === "Input.dispatchMouseEvent") {
          emit(
            "before-mouse-event",
            {},
            {
              button: "left",
              type: params?.type === "mousePressed" ? "mouseDown" : "mouseUp",
              x: Number(params?.x ?? 0),
              y: Number(params?.y ?? 0),
            },
          );
        }
        if (method === "Input.dispatchKeyEvent") {
          emit(
            "before-input-event",
            {},
            {
              key: params?.key,
              type: params?.type,
            },
          );
        }
        if (method === "Page.getNavigationHistory") {
          return { currentIndex: 0 };
        }
        if (method === "Accessibility.getFullAXTree") {
          return {
            nodes: [
              {
                backendDOMNodeId: 11,
                childIds: [],
                ignored: false,
                name: { value: "Open details" },
                nodeId: "ax-11",
                role: { value: "button" },
              },
            ],
          };
        }
        if (method === "DOM.describeNode") {
          return {
            node: {
              attributes: webContents.nodeAttributes,
              nodeName: "BUTTON",
              localName: "button",
            },
          };
        }
        if (method === "DOM.getBoxModel") {
          return {
            model: { content: [0, 0, 40, 0, 40, 20, 0, 20] },
          };
        }
        return {};
      },
    ),
  };
  const webContents = {
    debugger: debuggerApi,
    getTitle: () => "Example",
    getURL: () => webContents.url,
    id: webContentsId,
    nodeAttributes: [] as string[],
    emit,
    off: vi.fn((event: string, listener: (...args: unknown[]) => void) => {
      listeners.get(event)?.delete(listener);
    }),
    on: vi.fn((event: string, listener: (...args: unknown[]) => void) => {
      const eventListeners = listeners.get(event) ?? new Set();
      eventListeners.add(listener);
      listeners.set(event, eventListeners);
    }),
    once: vi.fn(),
    url,
  };
  const view = {
    getBounds: () => ({ x: 0, y: 0, width: 800, height: 600 }),
    webContents,
  };
  return { viewId, view, webContents, window };
}

function state(native: ReturnType<typeof createNativeView>) {
  return {
    canGoBack: false,
    canGoForward: false,
    error: null,
    find: null,
    isLoading: false,
    title: native.webContents.getTitle(),
    url: native.webContents.getURL(),
    viewId: native.viewId,
    zoomFactor: 1,
  };
}

async function mount(
  harness: ReturnType<typeof createHarness>,
  overrides: Record<string, unknown> = {},
) {
  return await harness.host.invoke(
    harness.window as never,
    "browser_tab_mount",
    {
      browserSessionId: "browser-session-1",
      bounds: { x: 0, y: 0, width: 800, height: 600 },
      selected: true,
      tabId: "tab-1",
      threadId: "thread-1",
      url: "https://example.com/",
      ...overrides,
    },
  );
}

describe("ElectronBrowserTabHost", () => {
  it("keeps Renderer and Agent actions on the same native WebContents", async () => {
    const harness = createHarness();
    const mounted = (await mount(harness)) as {
      viewId: string;
      webContentsId: number;
    };

    await harness.host.executeTool({
      arguments: {
        pageRevision: 0,
        tabId: "tab-1",
        title: "Example",
        url: "https://example.com/",
      },
      callId: "claim-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "claimTab",
      turnId: "turn-1",
    });
    const observed = await harness.host.executeTool({
      arguments: { tabId: "tab-1" },
      callId: "observe-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "observe",
      turnId: "turn-1",
    });

    expect(observed.state?.webContentsId).toBe(mounted.webContentsId);
    expect(observed.data).toMatchObject({
      pageRevision: 0,
      snapshotId: expect.any(String),
    });
    expect(observed.state).toMatchObject({
      browserSessionId: "browser-session-1",
      ownerWebContentsId: 41,
      tabId: "tab-1",
      threadId: "thread-1",
    });
    const native = harness.views.get(mounted.viewId);
    expect(native?.webContents.debugger.attach).toHaveBeenCalledWith("1.3");
    expect(native?.webContents.debugger.sendCommand).toHaveBeenCalledWith(
      "Accessibility.getFullAXTree",
    );
  });

  it("preserves the current URL when an existing user tab is remounted", async () => {
    const harness = createHarness();
    const first = (await mount(harness)) as { viewId: string; url: string };
    await harness.host.invoke(harness.window as never, "browser_tab_navigate", {
      tabId: "tab-1",
      url: "https://example.com/navigated",
    });

    await mount(harness, { url: "https://example.com/initial" });

    expect(harness.views.get(first.viewId)?.webContents.getURL()).toBe(
      "https://example.com/navigated",
    );
    const mountCalls = harness.embeddedHost.invoke.mock.calls.filter(
      ([, command]) => command === "embedded_browser_view_mount",
    );
    expect(mountCalls).toHaveLength(2);
    expect((mountCalls[1]?.[2] as { url?: unknown }).url).toBeUndefined();
  });

  it("preflights a sensitive click and executes it once after approval", async () => {
    const harness = createHarness();
    const mounted = (await mount(harness)) as { viewId: string };
    await harness.host.executeTool({
      arguments: {
        pageRevision: 0,
        tabId: "tab-1",
        title: "Example",
        url: "https://example.com/",
      },
      callId: "claim-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "claimTab",
      turnId: "turn-1",
    });
    const observed = await harness.host.executeTool({
      arguments: { tabId: "tab-1" },
      callId: "observe-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "observe",
      turnId: "turn-1",
    });
    const snapshotId = String(
      (observed.data as { snapshotId?: string }).snapshotId,
    );
    const native = harness.views.get(mounted.viewId);
    if (!native) {
      throw new Error("missing Browser test view");
    }
    native.webContents.nodeAttributes = ["aria-label", "Delete account"];
    const argumentsValue = {
      backendNodeId: 11,
      snapshotId,
      tabId: "tab-1",
    };

    const preflight = await harness.host.executeTool({
      arguments: argumentsValue,
      callId: "click-sensitive-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "click",
      turnId: "turn-1",
    });

    expect(preflight).toMatchObject({
      status: "approval_required",
      approval: {
        actionKind: "click",
        approvalToken: expect.any(String),
        backendNodeId: 11,
        riskClass: "high_impact_click",
        snapshotId,
        tabId: "tab-1",
        webContentsId: native.webContents.id,
      },
      state: { controlOwner: "agent", pageRevision: 0 },
    });
    expect(
      native.webContents.debugger.sendCommand.mock.calls.some(
        ([method]) => method === "Input.dispatchMouseEvent",
      ),
    ).toBe(false);

    const approved = await harness.host.executeTool({
      approvalToken: preflight.approval?.approvalToken,
      arguments: argumentsValue,
      callId: "click-sensitive-1",
      ownerWebContentsId: 41,
      phase: "approvedExecute",
      threadId: "thread-1",
      tool: "click",
      turnId: "turn-1",
    });

    expect(approved).toMatchObject({
      status: "completed",
      state: { controlOwner: "agent", pageRevision: 1 },
    });
    expect(
      native.webContents.debugger.sendCommand.mock.calls.filter(
        ([method]) => method === "Input.dispatchMouseEvent",
      ),
    ).toHaveLength(2);
    await expect(
      harness.host.executeTool({
        approvalToken: preflight.approval?.approvalToken,
        arguments: argumentsValue,
        callId: "click-sensitive-1",
        ownerWebContentsId: 41,
        phase: "approvedExecute",
        threadId: "thread-1",
        tool: "click",
        turnId: "turn-1",
      }),
    ).rejects.toThrow("approval token is stale or invalid");

    const observedAgain = await harness.host.executeTool({
      arguments: { tabId: "tab-1" },
      callId: "observe-2",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "observe",
      turnId: "turn-1",
    });
    const nextSnapshotId = String(
      (observedAgain.data as { snapshotId?: string }).snapshotId,
    );
    const stalePreflight = await harness.host.executeTool({
      arguments: {
        ...argumentsValue,
        snapshotId: nextSnapshotId,
      },
      callId: "click-sensitive-2",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "click",
      turnId: "turn-1",
    });
    await harness.host.invoke(harness.window as never, "browser_tab_reload", {
      tabId: "tab-1",
    });
    await expect(
      harness.host.executeTool({
        approvalToken: stalePreflight.approval?.approvalToken,
        arguments: {
          ...argumentsValue,
          snapshotId: nextSnapshotId,
        },
        callId: "click-sensitive-2",
        ownerWebContentsId: 41,
        phase: "approvedExecute",
        threadId: "thread-1",
        tool: "click",
        turnId: "turn-1",
      }),
    ).rejects.toThrow("approval token is stale or invalid");
  });

  it.each([
    [
      "mouse",
      "before-mouse-event",
      { button: "left", type: "mouseDown", x: 10, y: 10 },
    ],
    ["keyboard", "before-input-event", { key: "x", type: "keyDown" }],
  ])(
    "invalidates pending approval when native %s input takes control",
    async (_kind, eventName, input) => {
      const harness = createHarness();
      const mounted = (await mount(harness)) as { viewId: string };
      await harness.host.executeTool({
        arguments: {
          pageRevision: 0,
          tabId: "tab-1",
          title: "Example",
          url: "https://example.com/",
        },
        callId: "claim-1",
        ownerWebContentsId: 41,
        threadId: "thread-1",
        tool: "claimTab",
        turnId: "turn-1",
      });
      const observed = await harness.host.executeTool({
        arguments: { tabId: "tab-1" },
        callId: "observe-1",
        ownerWebContentsId: 41,
        threadId: "thread-1",
        tool: "observe",
        turnId: "turn-1",
      });
      const native = harness.views.get(mounted.viewId);
      if (!native) {
        throw new Error("missing Browser test view");
      }
      native.webContents.nodeAttributes = ["aria-label", "Delete account"];
      const argumentsValue = {
        backendNodeId: 11,
        snapshotId: String(
          (observed.data as { snapshotId?: string }).snapshotId,
        ),
        tabId: "tab-1",
      };
      const preflight = await harness.host.executeTool({
        arguments: argumentsValue,
        callId: "click-sensitive-1",
        ownerWebContentsId: 41,
        threadId: "thread-1",
        tool: "click",
        turnId: "turn-1",
      });

      native.webContents.emit(eventName, {}, input);

      expect(harness.events.at(-1)).toEqual({
        event: "browser-tab-state",
        payload: expect.objectContaining({
          activeTurnId: null,
          controlOwner: "user",
          pageRevision: 1,
          tabId: "tab-1",
        }),
      });
      expect(native.webContents.debugger.detach).toHaveBeenCalledOnce();
      await expect(
        harness.host.executeTool({
          approvalToken: preflight.approval?.approvalToken,
          arguments: argumentsValue,
          callId: "click-sensitive-1",
          ownerWebContentsId: 41,
          phase: "approvedExecute",
          threadId: "thread-1",
          tool: "click",
          turnId: "turn-1",
        }),
      ).rejects.toThrow("approval token is stale or invalid");
    },
  );

  it("在 native 用户接管时请求 canonical turn interrupt", async () => {
    const harness = createHarness();
    const interrupt = vi.fn(async () => undefined);
    harness.host.setTurnInterruptHandler(interrupt);
    const mounted = (await mount(harness)) as { viewId: string };
    await harness.host.executeTool({
      arguments: {
        pageRevision: 0,
        tabId: "tab-1",
        title: "Example",
        url: "https://example.com/",
      },
      callId: "claim-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "claimTab",
      turnId: "turn-1",
    });

    const native = harness.views.get(mounted.viewId);
    if (!native) {
      throw new Error("missing Browser test view");
    }
    native.webContents.emit("before-mouse-event", {}, {
      button: "left",
      type: "mouseDown",
      x: 10,
      y: 10,
    });

    await vi.waitFor(() => {
      expect(interrupt).toHaveBeenCalledWith("thread-1", "turn-1");
    });
  });

  it("rejects stale snapshots after an Agent action", async () => {
    const harness = createHarness();
    await mount(harness);
    await harness.host.executeTool({
      arguments: {
        pageRevision: 0,
        tabId: "tab-1",
        title: "Example",
        url: "https://example.com/",
      },
      callId: "claim-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "claimTab",
      turnId: "turn-1",
    });
    const observed = await harness.host.executeTool({
      arguments: { tabId: "tab-1" },
      callId: "observe-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "observe",
      turnId: "turn-1",
    });
    const snapshotId = String(
      (observed.data as { snapshotId?: string }).snapshotId,
    );

    const clicked = await harness.host.executeTool({
      arguments: { backendNodeId: 11, snapshotId, tabId: "tab-1" },
      callId: "click-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "click",
      turnId: "turn-1",
    });
    expect(clicked.state?.pageRevision).toBe(1);
    await expect(
      harness.host.executeTool({
        arguments: { backendNodeId: 11, snapshotId, tabId: "tab-1" },
        callId: "click-2",
        ownerWebContentsId: 41,
        threadId: "thread-1",
        tool: "click",
        turnId: "turn-1",
      }),
    ).rejects.toThrow("snapshot is stale");
  });

  it("fails route identity and stale turn mismatches closed", async () => {
    const harness = createHarness();
    await mount(harness);

    await expect(
      mount(harness, { browserSessionId: "forged-session" }),
    ).rejects.toThrow("Browser route identity mismatch");
    await harness.host.executeTool({
      arguments: {
        pageRevision: 0,
        tabId: "tab-1",
        title: "Example",
        url: "https://example.com/",
      },
      callId: "claim-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "claimTab",
      turnId: "turn-1",
    });
    await expect(
      harness.host.executeTool({
        arguments: {
          pageRevision: 0,
          tabId: "tab-1",
          title: "Example",
          url: "https://example.com/",
        },
        callId: "release-stale",
        ownerWebContentsId: 41,
        threadId: "thread-1",
        tool: "releaseTab",
        turnId: "turn-stale",
      }),
    ).rejects.toThrow("claimed by the active turn");
    await expect(
      harness.host.executeTool({
        arguments: { tabId: "tab-1" },
        callId: "claim-wrong-owner",
        ownerWebContentsId: 99,
        threadId: "thread-1",
        tool: "claimTab",
        turnId: "turn-1",
      }),
    ).rejects.toThrow("stale or unavailable");
  });

  it("forwards permission and download events with the canonical native tab identity", async () => {
    const harness = createHarness();
    const mounted = (await mount(harness)) as {
      viewId: string;
      webContentsId: number;
    };

    harness.host.observeEmbeddedEvent(
      "embedded-browser-view-permission-request",
      {
        decision: "blocked",
        embeddingOrigin: "https://example.com",
        permission: "geolocation",
        requestId: "permission-1",
        requestingUrl: "https://example.com/",
        url: "https://example.com/",
        viewId: mounted.viewId,
      },
    );
    harness.host.observeEmbeddedEvent("embedded-browser-view-download", {
      canResume: false,
      downloadId: "download-1",
      filename: "report.pdf",
      mimeType: "application/pdf",
      receivedBytes: 100,
      state: "completed",
      totalBytes: 100,
      url: "https://example.com/report.pdf",
      viewId: mounted.viewId,
    });

    for (const event of [
      "browser-tab-permission-request",
      "browser-tab-download",
    ]) {
      expect(harness.events).toContainEqual({
        event,
        payload: expect.objectContaining({
          browserSessionId: "browser-session-1",
          ownerWebContentsId: 41,
          tabId: "tab-1",
          threadId: "thread-1",
          viewId: mounted.viewId,
          webContentsId: mounted.webContentsId,
          windowId: 7,
        }),
      });
    }
  });

  it("releases retained user tabs and closes Agent tabs when the turn ends", async () => {
    const harness = createHarness();
    await mount(harness);
    await harness.host.executeTool({
      arguments: {
        pageRevision: 0,
        tabId: "tab-1",
        title: "Example",
        url: "https://example.com/",
      },
      callId: "claim-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "claimTab",
      turnId: "turn-1",
    });
    await harness.host.executeTool({
      arguments: {
        pageRevision: 0,
        tabId: "tab-1",
        title: "Example",
        url: "https://example.com/",
      },
      callId: "observe-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "observe",
      turnId: "turn-1",
    });
    await harness.host.executeTool({
      arguments: { url: "https://example.org/" },
      callId: "new-tab-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "newTab",
      turnId: "turn-1",
    });

    harness.host.turnEnded("thread-1", "turn-1");

    expect(harness.events).toContainEqual({
      event: "browser-tab-closed",
      payload: expect.objectContaining({ reason: "turn-ended" }),
    });
    expect(
      harness.events.some(
        ({ event, payload }) =>
          event === "browser-tab-state" &&
          (payload as { tabId?: string }).tabId === "tab-1" &&
          (payload as { controlOwner?: string }).controlOwner === "released",
      ),
    ).toBe(true);
    const retained = [...harness.views.values()].find(
      ({ webContents }) => webContents.id === 100,
    );
    expect(retained?.webContents.debugger.detach).toHaveBeenCalledOnce();
  });

  it("releases user tabs and closes Agent tabs when the App Server disconnects", async () => {
    const harness = createHarness();
    await mount(harness);
    await harness.host.executeTool({
      arguments: {
        pageRevision: 0,
        tabId: "tab-1",
        title: "Example",
        url: "https://example.com/",
      },
      callId: "claim-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "claimTab",
      turnId: "turn-1",
    });
    await harness.host.executeTool({
      arguments: { tabId: "tab-1" },
      callId: "observe-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "observe",
      turnId: "turn-1",
    });

    harness.host.connectionLost("app-server-disconnected");

    expect(
      harness.events.some(
        ({ event, payload }) =>
          event === "browser-tab-state" &&
          (payload as { tabId?: string }).tabId === "tab-1" &&
          (payload as { controlOwner?: string }).controlOwner === "released",
      ),
    ).toBe(true);
    const retained = [...harness.views.values()].find(
      ({ webContents }) => webContents.id === 100,
    );
    expect(retained?.webContents.debugger.detach).toHaveBeenCalledOnce();

    const agentTab = await harness.host.executeTool({
      arguments: { url: "https://example.org/" },
      callId: "new-tab-1",
      ownerWebContentsId: 41,
      threadId: "thread-1",
      tool: "newTab",
      turnId: "turn-2",
    });
    expect(agentTab.state?.origin).toBe("agent");
    harness.host.connectionLost("app-server-disconnected");
    expect(harness.events).toContainEqual({
      event: "browser-tab-closed",
      payload: expect.objectContaining({
        reason: "app-server-disconnected",
        tabId: "new-tab-1",
      }),
    });
  });
});
