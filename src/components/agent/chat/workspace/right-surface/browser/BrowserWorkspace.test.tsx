import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BrowserTabState } from "@/lib/api/browserTab";

const mocks = vi.hoisted(() => ({
  close: vi.fn(async () => undefined),
  createIdentity: vi.fn(),
  mount: vi.fn(),
  navigate: vi.fn(),
  openIdentity: vi.fn(),
  select: vi.fn(),
  setBounds: vi.fn(),
  stateHandler: null as ((state: BrowserTabState) => void) | null,
}));

vi.mock("react-i18next", () => {
  const t = (key: string) => key;
  return { useTranslation: () => ({ t }) };
});

vi.mock("@/lib/api/browserTab", () => ({
  closeBrowserTab: mocks.close,
  findInBrowserTab: vi.fn(),
  goBackBrowserTab: vi.fn(),
  goForwardBrowserTab: vi.fn(),
  isBrowserTabHostAvailable: () => true,
  listenBrowserTabClosed: vi.fn(async () => () => undefined),
  listenBrowserTabDownload: vi.fn(async () => () => undefined),
  listenBrowserTabLoadFailed: vi.fn(async () => () => undefined),
  listenBrowserTabPermissionRequest: vi.fn(async () => () => undefined),
  listenBrowserTabState: vi.fn(async (handler) => {
    mocks.stateHandler = handler;
    return () => {
      mocks.stateHandler = null;
    };
  }),
  mountBrowserTab: mocks.mount,
  navigateBrowserTab: mocks.navigate,
  reloadBrowserTab: vi.fn(),
  selectBrowserTab: mocks.select,
  setBrowserTabBounds: mocks.setBounds,
  setBrowserTabZoom: vi.fn(),
  stopBrowserTab: vi.fn(),
  stopFindInBrowserTab: vi.fn(),
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
    expect(workspace?.getAttribute("data-browser-active-turn-id")).toBe("turn-1");
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
});
