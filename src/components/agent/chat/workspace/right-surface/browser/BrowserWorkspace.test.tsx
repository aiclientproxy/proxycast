import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  BrowserTabDownloadEvent,
  BrowserTabLoadFailedEvent,
  BrowserTabPermissionRequestEvent,
  BrowserTabState,
} from "@/lib/api/browserTab";

const mocks = vi.hoisted(() => ({
  close: vi.fn(async () => undefined),
  createIdentity: vi.fn(),
  downloadHandler: null as ((event: BrowserTabDownloadEvent) => void) | null,
  find: vi.fn(),
  goBack: vi.fn(),
  goForward: vi.fn(),
  loadFailedHandler: null as
    | ((event: BrowserTabLoadFailedEvent) => void)
    | null,
  mount: vi.fn(),
  navigate: vi.fn(),
  openIdentity: vi.fn(),
  permissionHandler: null as
    | ((event: BrowserTabPermissionRequestEvent) => void)
    | null,
  select: vi.fn(),
  setBounds: vi.fn(),
  setZoom: vi.fn(),
  stateHandler: null as ((state: BrowserTabState) => void) | null,
  stopFind: vi.fn(),
}));

vi.mock("react-i18next", () => {
  const t = (key: string, options?: Record<string, unknown>) => {
    const rendered = key.replace(/\{\{(\w+)\}\}/g, (_match, name: string) =>
      String(options?.[name] ?? `{{${name}}}`),
    );
    return options
      ? `${rendered} ${Object.values(options).join(" ")}`
      : rendered;
  };
  return { useTranslation: () => ({ t }) };
});

vi.mock("@/lib/api/browserTab", () => ({
  closeBrowserTab: mocks.close,
  findInBrowserTab: mocks.find,
  goBackBrowserTab: mocks.goBack,
  goForwardBrowserTab: mocks.goForward,
  isBrowserTabHostAvailable: () => true,
  listenBrowserTabClosed: vi.fn(async () => () => undefined),
  listenBrowserTabDownload: vi.fn(async (handler) => {
    mocks.downloadHandler = handler;
    return () => {
      if (mocks.downloadHandler === handler) {
        mocks.downloadHandler = null;
      }
    };
  }),
  listenBrowserTabLoadFailed: vi.fn(async (handler) => {
    mocks.loadFailedHandler = handler;
    return () => {
      if (mocks.loadFailedHandler === handler) {
        mocks.loadFailedHandler = null;
      }
    };
  }),
  listenBrowserTabPermissionRequest: vi.fn(async (handler) => {
    mocks.permissionHandler = handler;
    return () => {
      if (mocks.permissionHandler === handler) {
        mocks.permissionHandler = null;
      }
    };
  }),
  listenBrowserTabState: vi.fn(async (handler) => {
    mocks.stateHandler = handler;
    return () => {
      if (mocks.stateHandler === handler) {
        mocks.stateHandler = null;
      }
    };
  }),
  mountBrowserTab: mocks.mount,
  navigateBrowserTab: mocks.navigate,
  reloadBrowserTab: vi.fn(),
  selectBrowserTab: mocks.select,
  setBrowserTabBounds: mocks.setBounds,
  setBrowserTabZoom: mocks.setZoom,
  stopBrowserTab: vi.fn(),
  stopFindInBrowserTab: mocks.stopFind,
}));

vi.mock("@/lib/api/browserWorkspace", () => ({
  createBrowserWorkspaceTabIdentity: mocks.createIdentity,
  openBrowserWorkspaceIdentity: mocks.openIdentity,
}));

import { BrowserWorkspace } from "./BrowserWorkspace";

function browserState(
  overrides: Partial<BrowserTabState> = {},
): BrowserTabState {
  return {
    activeTurnId: null,
    browserSessionId: "browser-session-1",
    canGoBack: false,
    canGoForward: false,
    controlOwner: "user",
    find: {
      activeMatchOrdinal: 0,
      finalUpdate: true,
      matches: 0,
      text: "",
    },
    humanReason: null,
    isLoading: false,
    mark: null,
    origin: "user",
    ownerWebContentsId: 41,
    pageRevision: 0,
    selected: true,
    tabId: "browser-session-1:user:primary",
    threadId: "thread-1",
    title: "Example",
    url: "https://example.com/",
    viewId: "browser:browser-session-1:user:primary",
    webContentsId: 101,
    windowId: 7,
    zoomFactor: 1,
    ...overrides,
  };
}

describe("BrowserWorkspace", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    vi.stubGlobal(
      "requestAnimationFrame",
      (callback: (time: number) => void) => {
        callback(0);
        return 1;
      },
    );
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    container = document.createElement("div");
    container.style.width = "900px";
    container.style.height = "700px";
    document.body.appendChild(container);
    root = createRoot(container);
    mocks.mount.mockReset();
    mocks.navigate.mockReset();
    mocks.openIdentity.mockReset();
    mocks.createIdentity.mockReset();
    mocks.setBounds.mockReset();
    mocks.downloadHandler = null;
    mocks.loadFailedHandler = null;
    mocks.permissionHandler = null;
    mocks.openIdentity.mockResolvedValue({
      browserSessionId: "browser-session-1",
      tabId: "browser-session-1:user:primary",
    });
    mocks.createIdentity.mockResolvedValue({
      browserSessionId: "browser-session-1",
      tabId: "browser-session-1:user:secondary",
    });
    mocks.mount.mockImplementation(async (params) =>
      browserState({
        browserSessionId: params.browserSessionId,
        tabId: params.tabId,
        threadId: params.threadId,
      }),
    );
    mocks.navigate.mockImplementation(async (tabId: string, url: string) =>
      browserState({ tabId, url }),
    );
    mocks.setBounds.mockImplementation(async ({ tabId }) =>
      browserState(
        tabId === "tab-agent-1"
          ? {
              activeTurnId: "turn-1",
              controlOwner: "agent",
              origin: "agent",
              tabId,
              title: "Agent tab",
              viewId: "browser-tab-agent-1",
              webContentsId: 202,
            }
          : { tabId },
      ),
    );
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    vi.unstubAllGlobals();
    container.remove();
  });

  it("mounts a stable native tab identity and projects its WebContents id", async () => {
    await act(async () => {
      root.render(
        <BrowserWorkspace
          initialUrl="https://example.com"
          runtimeSessionId="session-1"
          threadId="thread-1"
        />,
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(mocks.mount).toHaveBeenCalledWith(
      expect.objectContaining({
        browserSessionId: "browser-session-1",
        tabId: "browser-session-1:user:primary",
        threadId: "thread-1",
      }),
    );
    expect(mocks.openIdentity).toHaveBeenCalledWith({
      runtimeSessionId: "session-1",
      threadId: "thread-1",
    });
    const workspace = container.querySelector(
      '[data-testid="browser-workspace"]',
    );
    expect(workspace?.getAttribute("data-browser-tab-id")).toBe(
      "browser-session-1:user:primary",
    );
    expect(workspace?.getAttribute("data-browser-active-turn-id")).toBe("");
    expect(workspace?.getAttribute("data-browser-control-owner")).toBe("user");
    expect(workspace?.getAttribute("data-browser-page-revision")).toBe("0");
    expect(workspace?.getAttribute("data-browser-view-id")).toBe(
      "browser:browser-session-1:user:primary",
    );
    expect(workspace?.getAttribute("data-browser-web-contents-id")).toBe("101");
  });

  it("accepts an Agent-created tab event and navigates that same tab", async () => {
    await act(async () => {
      root.render(
        <BrowserWorkspace runtimeSessionId="session-1" threadId="thread-1" />,
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    await act(async () => {
      mocks.stateHandler?.(
        browserState({
          activeTurnId: "turn-1",
          controlOwner: "agent",
          origin: "agent",
          tabId: "tab-agent-1",
          title: "Agent tab",
          viewId: "browser-tab-agent-1",
          webContentsId: 202,
        }),
      );
    });

    const workspace = container.querySelector(
      '[data-testid="browser-workspace"]',
    );
    expect(workspace?.getAttribute("data-browser-tab-id")).toBe("tab-agent-1");
    expect(workspace?.getAttribute("data-browser-active-turn-id")).toBe(
      "turn-1",
    );
    expect(workspace?.getAttribute("data-browser-control-owner")).toBe("agent");
    expect(workspace?.getAttribute("data-browser-page-revision")).toBe("0");
    expect(workspace?.getAttribute("data-browser-view-id")).toBe(
      "browser-tab-agent-1",
    );
    expect(workspace?.getAttribute("data-browser-web-contents-id")).toBe("202");

    const input = container.querySelector<HTMLInputElement>(
      'input[aria-label="agentChat.browserWorkspace.address"]',
    );
    const form = input?.closest("form");
    await act(async () => {
      if (input) {
        Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype,
          "value",
        )?.set?.call(input, "example.org");
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
      await Promise.resolve();
    });
    await act(async () => {
      form?.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true }),
      );
      await Promise.resolve();
    });

    expect(mocks.navigate).toHaveBeenCalledWith(
      "tab-agent-1",
      "https://example.org/",
    );
  });

  it("creates a canonical owner before mounting from an empty task", async () => {
    const ensureOwner = vi.fn(async () => ({
      runtimeSessionId: "session-created",
      threadId: "thread-created",
    }));

    await act(async () => {
      root.render(<BrowserWorkspace ensureOwner={ensureOwner} threadId="" />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(ensureOwner).toHaveBeenCalledOnce();
    expect(mocks.openIdentity).toHaveBeenCalledWith({
      runtimeSessionId: "session-created",
      threadId: "thread-created",
    });
    expect(mocks.mount).toHaveBeenCalledWith(
      expect.objectContaining({
        browserSessionId: "browser-session-1",
        tabId: "browser-session-1:user:primary",
        threadId: "thread-created",
      }),
    );
    expect(
      container.querySelector('[data-testid="browser-workspace-loading"]'),
    ).toBeNull();
  });

  it("只投影当前 native tab 的权限和下载事件，并把状态带放在 viewport 外", async () => {
    await act(async () => {
      root.render(
        <BrowserWorkspace runtimeSessionId="session-1" threadId="thread-1" />,
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const permission = permissionEvent();
    await act(async () => {
      mocks.permissionHandler?.({ ...permission, tabId: "other-tab" });
    });
    expect(
      container.querySelector('[data-testid="browser-workspace-permission"]'),
    ).toBeNull();

    await act(async () => {
      mocks.permissionHandler?.(permission);
    });
    const banner = container.querySelector(
      '[data-testid="browser-workspace-permission"]',
    );
    const viewport = container.querySelector(
      '[data-testid="browser-workspace-viewport"]',
    );
    expect(banner).not.toBeNull();
    expect(banner?.className).not.toContain("absolute");
    expect(
      Boolean(
        banner &&
        viewport &&
        banner.compareDocumentPosition(viewport) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ),
    ).toBe(true);

    await act(async () => {
      mocks.downloadHandler?.(downloadEvent());
    });
    const shelf = container.querySelector(
      '[data-testid="browser-workspace-download"]',
    );
    expect(shelf).not.toBeNull();
    expect(shelf?.className).not.toContain("absolute");

    await act(async () => {
      mocks.stateHandler?.(
        browserState({
          tabId: "tab-2",
          viewId: "view-2",
          webContentsId: 202,
        }),
      );
    });
    expect(
      container
        .querySelector('[data-testid="browser-workspace"]')
        ?.getAttribute("data-browser-tab-id"),
    ).toBe("tab-2");
    expect(
      container.querySelector('[data-testid="browser-workspace-permission"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="browser-workspace-download"]'),
    ).toBeNull();
  });

  it("把当前 native tab 的 load-failed 事件投影成可区分的错误状态", async () => {
    await act(async () => {
      root.render(
        <BrowserWorkspace runtimeSessionId="session-1" threadId="thread-1" />,
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    await act(async () => {
      mocks.loadFailedHandler?.({
        ...browserState({ isLoading: false }),
        errorCode: -105,
        errorDescription: "NAME_NOT_RESOLVED",
        failureCategory: "dns",
      });
    });

    const error = container.querySelector(
      '[data-testid="browser-workspace-error"]',
    );
    expect(error?.getAttribute("data-browser-workspace-status")).toBe(
      "load-error",
    );
    expect(error?.getAttribute("data-browser-error-source")).toBe("load");
    expect(error?.textContent).toContain(
      "agentChat.browserWorkspace.loadFailedDnsTitle",
    );
    expect(error?.textContent).toContain("NAME_NOT_RESOLVED");

    await act(async () => {
      mocks.loadFailedHandler?.({
        ...browserState({ tabId: "other-tab" }),
        errorCode: -105,
        errorDescription: "OTHER_TAB",
        failureCategory: "dns",
      });
    });
    expect(error?.textContent).not.toContain("OTHER_TAB");
  });
});

function permissionEvent(
  overrides: Partial<BrowserTabPermissionRequestEvent> = {},
): BrowserTabPermissionRequestEvent {
  return {
    browserSessionId: "browser-session-1",
    decision: "blocked",
    embeddingOrigin: "https://example.com",
    ownerWebContentsId: 41,
    permission: "geolocation",
    requestingUrl: "https://example.com/",
    requestId: "permission-1",
    tabId: "browser-session-1:user:primary",
    threadId: "thread-1",
    url: "https://example.com/",
    viewId: "browser:browser-session-1:user:primary",
    webContentsId: 101,
    windowId: 7,
    ...overrides,
  };
}

function downloadEvent(
  overrides: Partial<BrowserTabDownloadEvent> = {},
): BrowserTabDownloadEvent {
  return {
    browserSessionId: "browser-session-1",
    canResume: false,
    downloadId: "download-1",
    filename: "report.pdf",
    mimeType: "application/pdf",
    ownerWebContentsId: 41,
    receivedBytes: 100,
    state: "completed",
    tabId: "browser-session-1:user:primary",
    threadId: "thread-1",
    totalBytes: 100,
    url: "https://example.com/report.pdf",
    viewId: "browser:browser-session-1:user:primary",
    webContentsId: 101,
    windowId: 7,
    ...overrides,
  };
}
