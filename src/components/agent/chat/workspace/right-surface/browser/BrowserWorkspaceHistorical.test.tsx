import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BrowserTabHistoricalProjection } from "@/lib/api/browserTab";
import { BrowserWorkspaceHistorical } from "./BrowserWorkspaceHistorical";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      `${key}:${String(options?.revision ?? options?.mark ?? "")}`,
  }),
}));

const projection: BrowserTabHistoricalProjection = {
  browserSessionId: "browser-session-1",
  tabId: "browser-tab-1",
  threadId: "thread-1",
  url: "https://example.com/history",
  title: "History",
  pageRevision: 3,
  mark: "handoff",
  origin: "agent",
  selected: true,
  snapshotId: "snapshot-3",
  replayedAt: "2026-08-23T00:00:00.000Z",
  readOnly: true,
};

describe("BrowserWorkspaceHistorical", () => {
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
    act(() => root.unmount());
    container.remove();
  });

  it("渲染历史事实但不暴露 native 运行态 identity", () => {
    act(() => {
      root.render(<BrowserWorkspaceHistorical projection={projection} />);
    });
    const element = container.querySelector(
      '[data-testid="browser-workspace-historical"]',
    );
    expect(element).toBeTruthy();
    expect(element?.getAttribute("data-browser-historical")).toBe("true");
    expect(element?.getAttribute("data-browser-page-revision")).toBe("3");
    expect(element?.getAttribute("data-browser-web-contents-id")).toBe("");
    expect(element?.getAttribute("data-browser-active-turn-id")).toBe("");
    expect(container.textContent).toContain("https://example.com/history");
    expect(container.textContent).toContain("historicalTitle");
  });
});
