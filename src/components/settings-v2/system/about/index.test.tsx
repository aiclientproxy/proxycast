import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { UpdateInstallSession } from "@/lib/api/appUpdate";

const {
  mockCheckForUpdates,
  mockGetRuntimeAppVersion,
  mockGetUpdateInstallSession,
  mockListenUpdateInstallSession,
  mockStartUpdateInstallSession,
} = vi.hoisted(() => ({
  mockCheckForUpdates: vi.fn(),
  mockGetRuntimeAppVersion: vi.fn(),
  mockGetUpdateInstallSession: vi.fn(),
  mockListenUpdateInstallSession: vi.fn(),
  mockStartUpdateInstallSession: vi.fn(),
}));
const { mockOpenExternalUrlWithSystemBrowser } = vi.hoisted(() => ({
  mockOpenExternalUrlWithSystemBrowser: vi.fn(),
}));

vi.mock("@/lib/api/appUpdate", () => ({
  checkForUpdates: mockCheckForUpdates,
  getUpdateInstallSession: mockGetUpdateInstallSession,
  isUpdateInstallSessionActive: (
    session: { stage: string; isActive: boolean } | null | undefined,
  ) =>
    Boolean(
      session?.isActive &&
      ["checking", "downloading", "installing", "restarting"].includes(
        session.stage,
      ),
    ),
  listenUpdateInstallSession: mockListenUpdateInstallSession,
  startUpdateInstallSession: mockStartUpdateInstallSession,
}));

vi.mock("@/lib/api/externalUrl", () => ({
  openExternalUrlWithSystemBrowser: mockOpenExternalUrlWithSystemBrowser,
}));
vi.mock("@/lib/appVersion", () => ({
  getRuntimeAppVersion: mockGetRuntimeAppVersion,
}));

import { AboutSection } from ".";

interface Mounted {
  container: HTMLDivElement;
  root: Root;
}

const mounted: Mounted[] = [];
let originalUserAgent: PropertyDescriptor | undefined;

function renderComponent() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<AboutSection />);
  });

  mounted.push({ container, root });
  return container;
}

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
  });
}

async function waitForLoad() {
  await flushEffects();
  await flushEffects();
}

function findButton(container: HTMLElement, text: string): HTMLButtonElement {
  const target = Array.from(container.querySelectorAll("button")).find(
    (button) => button.textContent?.includes(text),
  );

  if (!target) {
    throw new Error(`未找到按钮: ${text}`);
  }

  return target as HTMLButtonElement;
}

function createInstallSession(
  overrides: Partial<UpdateInstallSession> = {},
): UpdateInstallSession {
  return {
    sessionId: "session-1",
    stage: "downloading",
    currentVersion: "1.10.0",
    latestVersion: "1.10.1",
    downloadUrl: "https://example.com/lime",
    downloadedBytes: 50,
    totalBytes: 100,
    percent: 0.5,
    message: "downloading",
    error: null,
    startedAt: 1,
    updatedAt: 2,
    completedAt: null,
    canCloseWindow: true,
    isActive: true,
    ...overrides,
  };
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;

  vi.clearAllMocks();
  await changeLimeLocale("en-US");
  originalUserAgent = Object.getOwnPropertyDescriptor(
    window.navigator,
    "userAgent",
  );
  Object.defineProperty(window.navigator, "userAgent", {
    configurable: true,
    value: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
  });

  mockGetRuntimeAppVersion.mockReturnValue("1.10.0");
  mockCheckForUpdates.mockResolvedValue({
    current: "1.10.0",
    latest: "1.10.1",
    hasUpdate: true,
    downloadUrl: "https://example.com/lime",
    releaseNotesUrl: "https://example.com/lime/releases",
    releaseNotes: "修复设置页视觉层级并优化更新体验。",
    pubDate: "2026-03-20T00:00:00.000Z",
    error: undefined,
  });
  mockGetUpdateInstallSession.mockResolvedValue(
    createInstallSession({
      sessionId: "idle",
      stage: "idle",
      latestVersion: null,
      totalBytes: null,
      percent: 0,
      message: "idle",
      isActive: false,
    }),
  );
  mockListenUpdateInstallSession.mockResolvedValue(vi.fn());
  mockStartUpdateInstallSession.mockResolvedValue(createInstallSession());
  mockOpenExternalUrlWithSystemBrowser.mockResolvedValue(undefined);
});

afterEach(async () => {
  while (mounted.length > 0) {
    const target = mounted.pop();
    if (!target) {
      break;
    }

    act(() => {
      target.root.unmount();
    });
    target.container.remove();
  }

  vi.clearAllMocks();

  if (originalUserAgent) {
    Object.defineProperty(window.navigator, "userAgent", originalUserAgent);
  } else {
    Reflect.deleteProperty(window.navigator, "userAgent");
  }

  await changeLimeLocale("zh-CN");
});

describe("AboutSection", () => {
  it("应只渲染必要的品牌、版本与更新信息", async () => {
    const container = renderComponent();
    await waitForLoad();

    const text = container.textContent ?? "";
    expect(container.querySelector("img[alt='Lime']")).toBeInstanceOf(
      HTMLImageElement,
    );
    expect(text).toContain("Lime");
    expect(text).toContain("Version 1.10.0 (1.10.0)");
    expect(text).toContain("Copyright © 2026 Lime");
    expect(text).toContain("Update available: 1.10.1");
    expect(text).toContain("Check for Updates");
    expect(text).toContain("Restart to Update");
    expect(text).not.toContain("Skill package opening");
    expect(text).not.toContain("可更新到 1.10.1");
    expect(text).not.toContain("settings.about");
  });

  it("下载完成事件应独立驱动可安装状态，不依赖检查请求返回时序", async () => {
    mockCheckForUpdates.mockResolvedValue({
      current: "1.10.0",
      latest: null,
      hasUpdate: false,
      downloadUrl: null,
      releaseNotesUrl: null,
      releaseNotes: null,
      pubDate: null,
      error: null,
    });
    mockGetUpdateInstallSession.mockResolvedValue(
      createInstallSession({
        stage: "completed",
        percent: 1,
        message: "ready",
        completedAt: 3,
        isActive: false,
      }),
    );

    const container = renderComponent();
    await waitForLoad();

    expect(container.textContent).toContain("Update available: 1.10.1");
    expect(container.textContent).toContain("Ready to install");
    expect(findButton(container, "Restart to Update")).toBeInstanceOf(
      HTMLButtonElement,
    );
  });

  it("应移除关于页里的营销与能力说明噪音", async () => {
    const container = renderComponent();
    await waitForLoad();

    const text = container.textContent ?? "";
    expect(text).not.toContain("产品定位");
    expect(text).not.toContain("3 步开始创作");
    expect(text).not.toContain("适合谁");
    expect(text).not.toContain("工作区主线");
    expect(text).not.toContain("可选能力");
    expect(text).not.toContain("Made for creators");
  });

  it("点击更新按钮时应重新检查并允许触发安装会话", async () => {
    mockStartUpdateInstallSession.mockResolvedValueOnce(
      createInstallSession({
        stage: "failed",
        downloadedBytes: 0,
        totalBytes: null,
        percent: 0,
        message: "failed",
        error: "signature mismatch",
        completedAt: 3,
        isActive: false,
      }),
    );
    const container = renderComponent();
    await waitForLoad();

    await act(async () => {
      findButton(container, "Check for Updates").click();
      await waitForLoad();
    });

    expect(mockCheckForUpdates).toHaveBeenCalledTimes(2);
    expect(mockCheckForUpdates).toHaveBeenNthCalledWith(1, {
      automatic: true,
    });
    expect(mockCheckForUpdates).toHaveBeenNthCalledWith(2);

    await act(async () => {
      findButton(container, "Restart to Update").click();
      await waitForLoad();
    });

    expect(mockStartUpdateInstallSession).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain(
      "Unable to install the update automatically. Please download the latest version from the web page.",
    );
    expect(container.textContent).not.toContain("signature mismatch");
  });

  it("手动下载链接应走 Desktop Host 外链网关", async () => {
    const container = renderComponent();
    await waitForLoad();

    const link = Array.from(container.querySelectorAll("a")).find((anchor) =>
      anchor.textContent?.includes("Download from Web"),
    );
    expect(link).toBeInstanceOf(HTMLAnchorElement);
    expect(link?.getAttribute("href")).toBe(
      "https://example.com/lime/releases",
    );
    expect(link?.getAttribute("target")).toBeNull();
    expect(link?.getAttribute("rel")).toBe("noreferrer noopener");

    const clickEvent = new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
    });
    await act(async () => {
      link?.dispatchEvent(clickEvent);
      await Promise.resolve();
    });

    expect(clickEvent.defaultPrevented).toBe(true);
    expect(mockOpenExternalUrlWithSystemBrowser).toHaveBeenCalledWith(
      "https://example.com/lime/releases",
    );
  });

  it("自动安装失败后的网页下载链接应走 Desktop Host 外链网关", async () => {
    mockStartUpdateInstallSession.mockResolvedValueOnce(
      createInstallSession({
        stage: "failed",
        downloadedBytes: 0,
        totalBytes: null,
        percent: 0,
        message: "failed",
        error: "signature mismatch",
        completedAt: 3,
        isActive: false,
      }),
    );
    const container = renderComponent();
    await waitForLoad();

    await act(async () => {
      findButton(container, "Restart to Update").click();
      await waitForLoad();
    });

    const link = Array.from(container.querySelectorAll("a")).find((anchor) =>
      anchor.textContent?.includes("Open Web Download"),
    );
    expect(link).toBeInstanceOf(HTMLAnchorElement);
    expect(link?.getAttribute("href")).toBe(
      "https://example.com/lime/releases",
    );
    expect(link?.getAttribute("target")).toBeNull();
    expect(link?.getAttribute("rel")).toBe("noreferrer noopener");

    const clickEvent = new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
    });
    await act(async () => {
      link?.dispatchEvent(clickEvent);
      await Promise.resolve();
    });

    expect(clickEvent.defaultPrevented).toBe(true);
    expect(mockOpenExternalUrlWithSystemBrowser).toHaveBeenCalledWith(
      "https://example.com/lime/releases",
    );
  });

  it("Windows 关于页应只提示单一 setup 安装包", async () => {
    const container = renderComponent();
    await waitForLoad();

    expect(container.textContent).toContain(
      "Windows provides a single setup installer. To upgrade manually or reinstall, use the latest version from the web download page.",
    );
    expect(container.textContent).not.toContain("在线安装包");
    expect(container.textContent).not.toContain("offline 安装包");
  });

  it("更新检查失败时应隐藏技术错误", async () => {
    const consoleWarnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined);
    try {
      mockCheckForUpdates.mockResolvedValueOnce({
        current: "1.10.0",
        latest: "1.10.1",
        hasUpdate: false,
        downloadUrl: undefined,
        releaseNotes: undefined,
        pubDate: undefined,
        error: "更新清单请求失败（HTTP 404 Not Found），已回退本地缓存",
      });

      const container = renderComponent();
      await waitForLoad();

      const text = container.textContent ?? "";
      expect(text).toContain(
        "Unable to check for updates right now. Please try again later.",
      );
      expect(text).not.toContain("HTTP 404");
      expect(text).not.toContain("已回退本地缓存");
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    } finally {
      consoleWarnSpy.mockRestore();
    }
  });

  it("更新通道不可用时应使用构建版本兜底，不显示双重读取中", async () => {
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    try {
      mockGetRuntimeAppVersion.mockReturnValue("1.60.0");
      mockCheckForUpdates.mockRejectedValueOnce(
        new Error("Electron updater unavailable"),
      );

      const container = renderComponent();
      await waitForLoad();

      const text = container.textContent ?? "";
      expect(text).toContain("Version 1.60.0 (1.60.0)");
      expect(text).toContain(
        "Unable to check for updates right now. Please try again later.",
      );
      expect(text).not.toContain("Version Loading (Loading)");
      expect(text).not.toContain("Electron updater unavailable");
    } finally {
      consoleErrorSpy.mockRestore();
    }
  });
});
