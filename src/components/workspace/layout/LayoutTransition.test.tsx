import React from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { LayoutMode } from "@/lib/workspace/workflowTypes";
import { LayoutTransition } from "./LayoutTransition";
import {
  cleanupMountedRoots,
  clickElement,
  mountHarness,
  setupReactActEnvironment,
  type MountedRoot,
} from "../hooks/testUtils";

setupReactActEnvironment();

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function dispatchPointerLikeEvent(
  element: HTMLElement,
  type: string,
  options: { clientX: number; pointerId: number },
) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperties(event, {
    clientX: { value: options.clientX },
    pointerId: { value: options.pointerId },
  });
  element.dispatchEvent(event);
}

function LayoutHarness({
  mode,
  forceOpenChatPanel = false,
  chatPanelTopInset,
  canvasPanelTopInset,
}: {
  mode: LayoutMode;
  forceOpenChatPanel?: boolean;
  chatPanelTopInset?: string;
  canvasPanelTopInset?: string;
}) {
  return (
    <div style={{ width: "1200px", height: "720px" }}>
      <LayoutTransition
        mode={mode}
        chatContent={<div data-testid="layout-chat-content">chat</div>}
        canvasContent={<div data-testid="layout-canvas-content">canvas</div>}
        forceOpenChatPanel={forceOpenChatPanel}
        chatPanelTopInset={chatPanelTopInset}
        canvasPanelTopInset={canvasPanelTopInset}
      />
    </div>
  );
}

function EmptyCanvasLayoutHarness({ mode }: { mode: LayoutMode }) {
  return (
    <div style={{ width: "1200px", height: "720px" }}>
      <LayoutTransition
        mode={mode}
        chatContent={<div data-testid="layout-chat-content">chat</div>}
        canvasContent={null}
      />
    </div>
  );
}

function PlainChatLayoutHarness({ mode }: { mode: LayoutMode }) {
  return (
    <div style={{ width: "1200px", height: "720px" }}>
      <LayoutTransition
        mode={mode}
        chatContent={<div data-testid="layout-chat-content">chat</div>}
        canvasContent={null}
        chatPanelChrome="plain"
      />
    </div>
  );
}

describe("LayoutTransition", () => {
  const mountedRoots: MountedRoot[] = [];
  const originalInnerWidth = window.innerWidth;
  const originalInnerHeight = window.innerHeight;

  beforeEach(() => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 1440,
    });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      writable: true,
      value: 900,
    });
  });

  afterEach(() => {
    cleanupMountedRoots(mountedRoots);
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: originalInnerWidth,
    });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      writable: true,
      value: originalInnerHeight,
    });
  });

  it("chat-canvas 模式应为画布与对话保留分栏间距", () => {
    const { container } = mountHarness(
      LayoutHarness,
      { mode: "chat-canvas" },
      mountedRoots,
    );

    const root = container.querySelector<HTMLElement>(
      '[data-testid="layout-transition-root"]',
    );
    expect(root).not.toBeNull();

    const panelViewport = container.querySelector<HTMLElement>(
      '[data-testid="layout-panel-viewport"]',
    );
    const styles = Array.from(document.head.querySelectorAll("style"))
      .map((node) => node.textContent || "")
      .join("\n");

    const hasGapRule = Array.from(panelViewport?.classList ?? []).some(
      (className) =>
        new RegExp(`\\.${escapeRegExp(className)}\\{[^}]*gap:12px;`).test(
          styles,
        ),
    );

    expect(hasGapRule).toBe(true);
    expect(root?.getAttribute("data-layout-axis")).toBe("horizontal");
  });

  it("大屏 chat-canvas 模式应把对话列放在画布列之前", () => {
    const { container } = mountHarness(
      LayoutHarness,
      { mode: "chat-canvas" },
      mountedRoots,
    );

    const panelViewport = container.querySelector<HTMLElement>(
      '[data-testid="layout-panel-viewport"]',
    );
    const orderedPanels = Array.from(panelViewport?.children ?? [])
      .map((node) => node.getAttribute("data-testid"))
      .filter(Boolean);

    expect(orderedPanels).toEqual([
      "layout-chat-panel",
      "layout-chat-canvas-resize-handle",
      "layout-canvas-panel",
    ]);
    expect(
      container.querySelector('[data-testid="layout-chat-content"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="layout-canvas-content"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="layout-chat-canvas-resize-handle"]',
      ),
    ).not.toBeNull();
  });

  it("大屏 chat-canvas 模式应支持拖动分隔线调整对话区宽度", async () => {
    const { container } = mountHarness(
      LayoutHarness,
      { mode: "chat-canvas" },
      mountedRoots,
    );

    const root = container.querySelector<HTMLElement>(
      '[data-testid="layout-transition-root"]',
    );
    const chatPanel = container.querySelector<HTMLElement>(
      '[data-testid="layout-chat-panel"]',
    );
    const handle = container.querySelector<HTMLElement>(
      '[data-testid="layout-chat-canvas-resize-handle"]',
    );

    expect(root).not.toBeNull();
    expect(chatPanel).not.toBeNull();
    expect(handle).not.toBeNull();

    root!.getBoundingClientRect = () =>
      ({
        left: 0,
        top: 0,
        right: 1200,
        bottom: 720,
        width: 1200,
        height: 720,
        x: 0,
        y: 0,
        toJSON: () => undefined,
      }) as DOMRect;
    handle!.setPointerCapture = () => undefined;
    handle!.releasePointerCapture = () => undefined;
    handle!.hasPointerCapture = () => true;

    await act(async () => {
      dispatchPointerLikeEvent(handle!, "pointerdown", {
        clientX: 720,
        pointerId: 1,
      });
      dispatchPointerLikeEvent(handle!, "pointermove", {
        clientX: 760,
        pointerId: 1,
      });
      dispatchPointerLikeEvent(handle!, "pointerup", {
        clientX: 760,
        pointerId: 1,
      });
      await Promise.resolve();
    });

    expect(chatPanel?.getAttribute("data-chat-panel-width")).toBe("760px");
    expect(handle?.getAttribute("data-dragging")).toBe("false");
  });

  it("小屏 chat-canvas 模式仍应保持中右分栏比例一致", () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 1080,
    });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      writable: true,
      value: 720,
    });

    const { container } = mountHarness(
      LayoutHarness,
      { mode: "chat-canvas" },
      mountedRoots,
    );

    const root = container.querySelector<HTMLElement>(
      '[data-testid="layout-transition-root"]',
    );
    const panelViewport = container.querySelector<HTMLElement>(
      '[data-testid="layout-panel-viewport"]',
    );
    const orderedPanels = Array.from(panelViewport?.children ?? [])
      .map((node) => node.getAttribute("data-testid"))
      .filter(Boolean);

    expect(root?.getAttribute("data-layout-axis")).toBe("horizontal");
    expect(root?.getAttribute("data-chat-panel-placement")).toBe("inline");
    expect(orderedPanels).toEqual([
      "layout-chat-panel",
      "layout-chat-canvas-resize-handle",
      "layout-canvas-panel",
    ]);
    expect(
      container
        .querySelector<HTMLElement>('[data-testid="layout-chat-panel"]')
        ?.getAttribute("data-overlay-state"),
    ).toBe("inline");
    expect(
      container.querySelector('[data-testid="layout-chat-content"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="layout-canvas-content"]'),
    ).not.toBeNull();
    expect(
      container
        .querySelector<HTMLElement>('[data-testid="layout-compact-mode-bar"]')
        ?.getAttribute("data-visible"),
    ).toBe("false");
    expect(
      container.querySelector(
        '[data-testid="layout-chat-canvas-resize-handle"]',
      ),
    ).not.toBeNull();
  });

  it("极窄 chat-canvas 模式应默认用完整聊天主面板并允许切换工作台", async () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 820,
    });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      writable: true,
      value: 600,
    });

    const { container } = mountHarness(
      LayoutHarness,
      { mode: "chat-canvas" },
      mountedRoots,
    );

    const root = container.querySelector<HTMLElement>(
      '[data-testid="layout-transition-root"]',
    );

    expect(root?.getAttribute("data-layout-axis")).toBe("single");
    expect(root?.getAttribute("data-chat-panel-placement")).toBe(
      "single-panel",
    );
    expect(root?.getAttribute("data-compact-primary-panel")).toBe("chat");
    expect(
      container.querySelector('[data-testid="layout-compact-mode-bar"]'),
    ).not.toBeNull();
    expect(
      container.querySelector(
        '[data-testid="layout-chat-canvas-resize-handle"]',
      ),
    ).toBeNull();
    expect(
      container
        .querySelector<HTMLElement>('[data-testid="layout-chat-panel"]')
        ?.getAttribute("data-overlay-state"),
    ).toBe("single-active");
    expect(
      container.querySelector('[data-testid="layout-chat-content"]'),
    ).not.toBeNull();

    await act(async () => {
      clickElement(
        container.querySelector('[data-testid="layout-compact-canvas-tab"]'),
      );
      await Promise.resolve();
    });

    expect(root?.getAttribute("data-compact-primary-panel")).toBe("canvas");
    expect(
      container
        .querySelector<HTMLElement>('[data-testid="layout-chat-panel"]')
        ?.getAttribute("data-overlay-state"),
    ).toBe("single-hidden");
    expect(
      container.querySelector('[data-testid="layout-canvas-content"]'),
    ).not.toBeNull();

    await act(async () => {
      clickElement(
        container.querySelector('[data-testid="layout-compact-chat-tab"]'),
      );
      await Promise.resolve();
    });

    expect(root?.getAttribute("data-compact-primary-panel")).toBe("chat");
  });

  it("紧凑模式条应消费顶部页头占位且不在单面板内重复留白", () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 820,
    });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      writable: true,
      value: 720,
    });

    const { container } = mountHarness(
      LayoutHarness,
      {
        mode: "chat-canvas",
        chatPanelTopInset: "52px",
        canvasPanelTopInset: "52px",
      },
      mountedRoots,
    );

    const compactModeBar = container.querySelector<HTMLElement>(
      '[data-testid="layout-compact-mode-bar"]',
    );
    const chatPanel = container.querySelector<HTMLElement>(
      '[data-testid="layout-chat-panel-inner"]',
    );
    const canvasPanel = container.querySelector<HTMLElement>(
      '[data-testid="layout-canvas-panel"]',
    );

    expect(compactModeBar?.dataset.visible).toBe("true");
    expect(compactModeBar?.dataset.topInset).toBe("52px");
    expect(window.getComputedStyle(compactModeBar!).marginTop).toBe("52px");
    expect(chatPanel?.dataset.topInset).toBe("0px");
    expect(canvasPanel?.dataset.topInset).toBe("0px");
  });

  it("应按实际工作区容器宽度切换单面板，而不是只看窗口宽度", async () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 1200,
    });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      writable: true,
      value: 720,
    });

    const { container } = mountHarness(
      LayoutHarness,
      { mode: "chat-canvas" },
      mountedRoots,
    );

    const root = container.querySelector<HTMLElement>(
      '[data-testid="layout-transition-root"]',
    );
    expect(root?.getAttribute("data-layout-axis")).toBe("horizontal");

    root!.getBoundingClientRect = () =>
      ({
        left: 0,
        top: 0,
        right: 760,
        bottom: 720,
        width: 760,
        height: 720,
        x: 0,
        y: 0,
        toJSON: () => undefined,
      }) as DOMRect;

    await act(async () => {
      window.dispatchEvent(new Event("resize"));
      await Promise.resolve();
    });

    expect(root?.getAttribute("data-layout-axis")).toBe("single");
    expect(root?.getAttribute("data-compact-primary-panel")).toBe("chat");
  });

  it("紧凑单面板态存在待处理 A2UI 时应自动切回聊天区", async () => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: 820,
    });
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      writable: true,
      value: 600,
    });

    const { container, rerender } = mountHarness(
      LayoutHarness,
      { mode: "chat-canvas", forceOpenChatPanel: false },
      mountedRoots,
    );

    await act(async () => {
      clickElement(
        container.querySelector('[data-testid="layout-compact-canvas-tab"]'),
      );
      await Promise.resolve();
    });
    expect(
      container
        .querySelector<HTMLElement>('[data-testid="layout-transition-root"]')
        ?.getAttribute("data-compact-primary-panel"),
    ).toBe("canvas");

    rerender({ mode: "chat-canvas", forceOpenChatPanel: true });

    await act(async () => {
      await Promise.resolve();
    });

    expect(
      container
        .querySelector<HTMLElement>('[data-testid="layout-chat-panel"]')
        ?.getAttribute("data-overlay-state"),
    ).toBe("single-active");
    expect(
      container.querySelector('[data-testid="layout-chat-content"]'),
    ).not.toBeNull();
  });

  it("画布内容为空时应退回聊天布局，避免保留空白画布列", () => {
    const { container } = mountHarness(
      EmptyCanvasLayoutHarness,
      { mode: "chat-canvas" },
      mountedRoots,
    );

    const root = container.querySelector<HTMLElement>(
      '[data-testid="layout-transition-root"]',
    );

    expect(root?.getAttribute("data-effective-mode")).toBe("chat");
    expect(root?.getAttribute("data-has-canvas")).toBe("false");
    expect(
      container.querySelector('[data-testid="layout-chat-content"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="layout-canvas-content"]'),
    ).toBeNull();
  });

  it("plain 聊天壳模式不应再渲染额外面板背景层", () => {
    const { container } = mountHarness(
      PlainChatLayoutHarness,
      { mode: "chat" },
      mountedRoots,
    );

    expect(
      container.querySelector('[data-testid="layout-chat-panel-plain"]'),
    ).not.toBeNull();
    expect(
      container.querySelector('[data-testid="layout-chat-panel-inner"]'),
    ).toBeNull();
  });
});
