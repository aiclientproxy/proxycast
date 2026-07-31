import { act } from "react";
import { readFileSync } from "node:fs";
import path from "node:path";
import { cwd } from "node:process";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UpdateInstallSession } from "@/lib/api/appUpdate";

const {
  mockCloseUpdateWindow,
  mockDismissUpdateNotification,
  mockGetUpdateInstallSession,
  mockGetCurrentWindow,
  mockListenUpdateInstallSession,
  mockRecordUpdateNotificationAction,
  mockRemindUpdateLater,
  mockOpenExternalUrlWithSystemBrowser,
  mockStartUpdateInstallSession,
} = vi.hoisted(() => ({
  mockCloseUpdateWindow: vi.fn(async () => undefined),
  mockDismissUpdateNotification: vi.fn(async () => undefined),
  mockGetUpdateInstallSession: vi.fn(),
  mockGetCurrentWindow: vi.fn(() => ({
    close: vi.fn(async () => undefined),
    startDragging: vi.fn(async () => undefined),
  })),
  mockListenUpdateInstallSession: vi.fn(),
  mockOpenExternalUrlWithSystemBrowser: vi.fn(async () => undefined),
  mockRecordUpdateNotificationAction: vi.fn(async () => undefined),
  mockRemindUpdateLater: vi.fn(async () => undefined),
  mockStartUpdateInstallSession: vi.fn(),
}));

vi.mock("@/lib/desktop-host/window", () => ({
  getCurrentWindow: mockGetCurrentWindow,
}));

vi.mock("@/lib/api/externalUrl", () => ({
  openExternalUrlWithSystemBrowser: mockOpenExternalUrlWithSystemBrowser,
}));

vi.mock("@/lib/api/appUpdate", () => ({
  closeUpdateWindow: mockCloseUpdateWindow,
  dismissUpdateNotification: mockDismissUpdateNotification,
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
  recordUpdateNotificationAction: mockRecordUpdateNotificationAction,
  remindUpdateLater: mockRemindUpdateLater,
  startUpdateInstallSession: mockStartUpdateInstallSession,
}));

import { changeLimeLocale } from "@/i18n/createI18n";
import { UpdateNotificationPage } from "./update-notification";

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

async function flushEffects(times = 4) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

async function renderUpdateNotification(path: string) {
  window.history.pushState({}, "", path);
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(<UpdateNotificationPage />);
  });
  await flushEffects();

  mountedRoots.push({ root, container });
  return container;
}

function getText(container: HTMLElement) {
  return (container.textContent ?? "").replace(/\s+/g, " ").trim();
}

function findButtonByText(container: HTMLElement, text: string) {
  const button = Array.from(container.querySelectorAll("button")).find((item) =>
    item.textContent?.includes(text),
  );
  if (!button) {
    throw new Error(`未找到按钮文本: ${text}`);
  }
  return button as HTMLButtonElement;
}

function readUpdateNotificationCss() {
  return readFileSync(
    path.resolve(cwd(), "src/pages/update-notification.css"),
    "utf8",
  );
}

function createInstallSession(
  overrides: Partial<UpdateInstallSession> = {},
): UpdateInstallSession {
  return {
    sessionId: "session-1",
    stage: "downloading",
    currentVersion: "1.0.0",
    latestVersion: "1.2.0",
    downloadUrl: "https://example.com/release",
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
  mockOpenExternalUrlWithSystemBrowser.mockResolvedValue(undefined);
  mockStartUpdateInstallSession.mockResolvedValue(createInstallSession());
});

afterEach(async () => {
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

  window.history.pushState({}, "", "/");
  await changeLimeLocale("zh-CN");
  vi.clearAllMocks();
});

describe("UpdateNotificationPage", () => {
  it("更新独立窗口应只保留单层 toast 表面，不应再露出外层背景", () => {
    const css = readUpdateNotificationCss();

    expect(css).toContain(".update-container");
    expect(css).toContain("background: transparent;");
    expect(css).toContain("padding: 0;");
    expect(css).toContain("pointer-events: none;");
    expect(css).toContain(".update-toast");
    expect(css).toContain("width: 100%;");
    expect(css).toContain("min-height: 100%;");
    expect(css).not.toContain("background: rgba(255, 255, 255");
    expect(css).not.toContain("backdrop-filter");
    expect(css).not.toContain("backdrop-blur");
  });

  it("挂载期间应把独立更新窗口根背景标记为透明并在卸载后恢复", async () => {
    delete document.documentElement.dataset.limeWindow;
    await renderUpdateNotification(
      "/update-notification?current=1.0.0&latest=1.2.0",
    );

    expect(document.documentElement.dataset.limeWindow).toBe(
      "update-notification",
    );

    const mounted = mountedRoots.pop();
    expect(mounted).toBeDefined();
    act(() => {
      mounted!.root.unmount();
    });
    mounted!.container.remove();

    expect(document.documentElement.dataset.limeWindow).toBeUndefined();
  });

  it("应通过 common namespace 渲染英文更新提醒窗口文案", async () => {
    const container = await renderUpdateNotification(
      "/update-notification?current=1.0.0&latest=1.2.0&download_url=https%3A%2F%2Fexample.com%2Frelease",
    );

    const text = getText(container);
    expect(text).toContain("New version 1.2.0");
    expect(text).toContain("(current 1.0.0)");
    expect(text).toContain("Later");
    expect(text).toContain("Update now");
    expect(text).not.toContain("In 3 days");
    expect(text).not.toContain("Next week");
    expect(text).not.toContain("Skip this version");
    expect(text).not.toContain("View release page in browser");
    expect(text).not.toContain("发现新版本");
    expect(
      container.querySelector('button[aria-label="Close reminder"]'),
    ).toBeInstanceOf(HTMLButtonElement);
  });

  it("更新失败时应显示失败与重试，不再显示发现新版本或稍后", async () => {
    mockGetUpdateInstallSession.mockResolvedValueOnce(
      createInstallSession({
        stage: "failed",
        message: "signature mismatch",
        error: "signature mismatch",
        isActive: false,
      }),
    );

    const container = await renderUpdateNotification(
      "/update-notification?current=1.0.0&latest=1.2.0",
    );
    const text = getText(container);

    expect(text).toContain("Update failed");
    expect(text).toContain("Retry");
    expect(text).not.toContain("New version");
    expect(text).not.toContain("Later");
  });

  it("同版本状态应只显示已是最新和关闭操作", async () => {
    mockGetUpdateInstallSession.mockResolvedValueOnce(
      createInstallSession({
        stage: "up_to_date",
        latestVersion: "1.0.0",
        message: "up to date",
        error: null,
        isActive: false,
      }),
    );

    const container = await renderUpdateNotification(
      "/update-notification?current=1.0.0&latest=1.0.0",
    );
    const text = getText(container);

    expect(text).toContain("Already up to date");
    expect(text).not.toContain("New version");
    expect(text).not.toContain("Update now");
    expect(text).not.toContain("Later");
  });

  it("已下载的新版本应明确提示已准备好并要求重启更新", async () => {
    mockGetUpdateInstallSession.mockResolvedValueOnce(
      createInstallSession({
        stage: "completed",
        percent: 1,
        message: "ready",
        completedAt: 3,
        isActive: false,
      }),
    );

    const container = await renderUpdateNotification(
      "/update-notification?current=1.0.0&latest=1.2.0",
    );
    const text = getText(container);

    expect(text).toContain("New version 1.2.0");
    expect(text).toContain("Ready to install");
    expect(text).toContain("Restart to update");
  });

  it("立即更新过程中的按钮状态应使用 common namespace 文案", async () => {
    mockStartUpdateInstallSession.mockResolvedValueOnce(createInstallSession());
    const container = await renderUpdateNotification(
      "/update-notification?current=1.0.0&latest=1.2.0",
    );

    await act(async () => {
      findButtonByText(container, "Update now").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRecordUpdateNotificationAction).toHaveBeenCalledWith(
      "update_now",
    );
    expect(mockStartUpdateInstallSession).toHaveBeenCalledTimes(1);
    expect(getText(container)).toContain("Downloading 50%");
    expect(container.querySelector('[role="progressbar"]')).toBeInstanceOf(
      HTMLDivElement,
    );
  });

  it("启动更新安装失败时应通过 current 外链网关打开下载地址且不回退 window.open", async () => {
    mockStartUpdateInstallSession.mockRejectedValueOnce(
      new Error("install session unavailable"),
    );
    const windowOpenSpy = vi.spyOn(window, "open").mockReturnValue(null);
    const container = await renderUpdateNotification(
      "/update-notification?current=1.0.0&latest=1.2.0&download_url=https%3A%2F%2Fexample.com%2Frelease",
    );

    await act(async () => {
      findButtonByText(container, "Update now").dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockRecordUpdateNotificationAction).toHaveBeenCalledWith(
      "update_now",
    );
    expect(mockOpenExternalUrlWithSystemBrowser).toHaveBeenCalledWith(
      "https://example.com/release",
    );
    expect(windowOpenSpy).not.toHaveBeenCalled();
  });
});
