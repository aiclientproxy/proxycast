import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";

const { mockReadWindowsSandboxReadiness } = vi.hoisted(() => ({
  mockReadWindowsSandboxReadiness: vi.fn(),
}));

vi.mock("@/lib/api/windowsSandbox", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/windowsSandbox")
  >("@/lib/api/windowsSandbox");
  return {
    ...actual,
    readWindowsSandboxReadiness: mockReadWindowsSandboxReadiness,
  };
});

import { WindowsSandboxReadinessStatus } from "./WindowsSandboxReadinessStatus";

let container: HTMLDivElement;
let root: Root;

function setNavigator(platform: string, userAgent: string) {
  Object.defineProperty(window.navigator, "platform", {
    configurable: true,
    value: platform,
  });
  Object.defineProperty(window.navigator, "userAgent", {
    configurable: true,
    value: userAgent,
  });
}

async function renderStatus() {
  await act(async () => {
    root.render(<WindowsSandboxReadinessStatus />);
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  vi.clearAllMocks();
  await changeLimeLocale("zh-CN");
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  act(() => root.unmount());
  container.remove();
  setNavigator("MacIntel", "Mozilla/5.0 (Macintosh)");
  await changeLimeLocale("zh-CN");
});

describe("WindowsSandboxReadinessStatus", () => {
  it("仅在 Windows 展示 App Server 返回的真实状态", async () => {
    setNavigator("Win32", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
    mockReadWindowsSandboxReadiness.mockResolvedValue("updateRequired");

    await renderStatus();

    const status = container.querySelector(
      '[data-testid="windows-sandbox-readiness"]',
    );
    expect(mockReadWindowsSandboxReadiness).toHaveBeenCalledTimes(1);
    expect(status?.getAttribute("data-status")).toBe("updateRequired");
    expect(status?.textContent).toContain("需要更新");
    expect(status?.textContent).toContain("尚未提供 Windows 隔离运行器");
  });

  it("非 Windows 平台不请求也不展示 readiness", async () => {
    setNavigator("MacIntel", "Mozilla/5.0 (Macintosh)");

    await renderStatus();

    expect(mockReadWindowsSandboxReadiness).not.toHaveBeenCalled();
    expect(
      container.querySelector('[data-testid="windows-sandbox-readiness"]'),
    ).toBeNull();
  });

  it("读取失败后显示可重试状态", async () => {
    setNavigator("Win32", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
    mockReadWindowsSandboxReadiness
      .mockRejectedValueOnce(new Error("bridge unavailable"))
      .mockResolvedValueOnce("notConfigured");

    await renderStatus();

    const retry = container.querySelector<HTMLButtonElement>(
      'button[aria-label="重新检查 Windows 隔离状态"]',
    );
    expect(container.textContent).toContain("检查失败");
    expect(retry).not.toBeNull();

    await act(async () => {
      retry?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(mockReadWindowsSandboxReadiness).toHaveBeenCalledTimes(2);
    expect(
      container
        .querySelector('[data-testid="windows-sandbox-readiness"]')
        ?.getAttribute("data-status"),
    ).toBe("notConfigured");
  });
});
