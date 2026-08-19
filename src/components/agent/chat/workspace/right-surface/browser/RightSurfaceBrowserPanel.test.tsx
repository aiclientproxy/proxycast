import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RightSurfaceBrowserPanel } from "./RightSurfaceBrowserPanel";

vi.mock("./BrowserWorkspace", () => ({
  BrowserWorkspace: ({
    initialUrl,
    runtimeSessionId,
    threadId,
    onNavigate,
  }: {
    initialUrl?: string | null;
    runtimeSessionId?: string | null;
    threadId: string;
    onNavigate?: (url: string, title?: string | null) => void;
  }) => (
    <button
      type="button"
      data-testid="mock-browser-workspace"
      data-initial-url={initialUrl ?? ""}
      data-runtime-session-id={runtimeSessionId ?? ""}
      data-thread-id={threadId}
      onClick={() => onNavigate?.("https://example.com/", "Example")}
    >
      browser panel
    </button>
  ),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

describe("RightSurfaceBrowserPanel", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("应在右侧 surface 内渲染内嵌浏览器 panel", () => {
    act(() => {
      root.render(
        <RightSurfaceBrowserPanel
          initialUrl="https://example.com"
          runtimeSessionId="session-1"
          controlMode="human_takeover"
          lifecycleState="human_controlling"
          threadId="thread-1"
        />,
      );
    });

    const panel = container.querySelector(
      '[data-testid="right-surface-browser-panel"]',
    );
    expect(panel).toBeTruthy();
    expect(panel?.getAttribute("data-browser-session-id")).toBe("");
    expect(panel?.getAttribute("data-browser-control-mode")).toBe(
      "human_takeover",
    );
    expect(panel?.getAttribute("data-browser-control-owner")).toBe("human");
    expect(panel?.getAttribute("data-browser-human-takeover")).toBe("true");
    expect(panel?.getAttribute("data-browser-lifecycle-state")).toBe(
      "human_controlling",
    );
    expect(panel?.getAttribute("data-browser-thread-id")).toBe("thread-1");
    expect(
      container.querySelector('[data-testid="right-surface-browser-panel"]'),
    ).toBeTruthy();
    expect(
      container
        .querySelector('[data-testid="mock-browser-workspace"]')
        ?.getAttribute("data-initial-url"),
    ).toBe("https://example.com");
    expect(
      container
        .querySelector('[data-testid="mock-browser-workspace"]')
        ?.getAttribute("data-runtime-session-id"),
    ).toBe("session-1");
    expect(
      container.querySelector(
        '[data-testid="right-surface-browser-control-overlay"]',
      )?.textContent,
    ).toContain("agentChat.rightSurface.browserControl.human.label");
  });

  it("未激活时不挂载浏览器 view", () => {
    act(() => {
      root.render(
        <RightSurfaceBrowserPanel active={false} threadId="thread-1" />,
      );
    });

    expect(
      container.querySelector('[data-testid="right-surface-browser-panel"]'),
    ).toBeNull();
  });

  it("Agent 控制常态只暴露 flags，不渲染接管 overlay", () => {
    act(() => {
      root.render(
        <RightSurfaceBrowserPanel
          runtimeSessionId="session-2"
          controlMode="agent"
          lifecycleState="live"
          threadId="thread-2"
        />,
      );
    });

    const panel = container.querySelector(
      '[data-testid="right-surface-browser-panel"]',
    );
    expect(panel?.getAttribute("data-browser-control-owner")).toBe("agent");
    expect(panel?.getAttribute("data-browser-human-takeover")).toBe("false");
    expect(
      container.querySelector(
        '[data-testid="right-surface-browser-control-overlay"]',
      ),
    ).toBeNull();
  });
});
