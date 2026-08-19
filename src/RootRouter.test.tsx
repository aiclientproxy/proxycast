import { act } from "react";
import type { ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RootRouter } from "./RootRouter";

const {
  mockFinalizeCrashRecoveryAutoReload,
  mockFinalizeModuleImportAutoReload,
  mockGetRuntimeAppVersion,
} = vi.hoisted(() => ({
  mockFinalizeCrashRecoveryAutoReload: vi.fn(),
  mockFinalizeModuleImportAutoReload: vi.fn(),
  mockGetRuntimeAppVersion: vi.fn(() => "0.0.0-test"),
}));

vi.mock("./App", () => ({
  default: () => <div data-testid="main-app">主应用</div>,
}));

vi.mock("./pages/update-notification", () => ({
  UpdateNotificationPage: () => <div data-testid="update-notification-page" />,
}));

vi.mock("./features/resource-manager", () => ({
  ResourceManagerPage: () => <div data-testid="resource-manager-page" />,
}));

vi.mock("./components/ui/sonner", () => ({
  Toaster: () => <div data-testid="toaster" />,
}));

vi.mock("./components/layout/AppCrashBoundary", () => ({
  AppCrashBoundary: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("./components/layout/CrashRecoveryPanel.helpers", () => ({
  finalizeCrashRecoveryAutoReload: mockFinalizeCrashRecoveryAutoReload,
  finalizeModuleImportAutoReload: mockFinalizeModuleImportAutoReload,
}));

vi.mock("./lib/appVersion", () => ({
  getRuntimeAppVersion: mockGetRuntimeAppVersion,
}));

interface MountedRootRouter {
  container: HTMLDivElement;
  root: Root;
}

const mounted: MountedRootRouter[] = [];

async function renderRootRouter(pathname: string) {
  window.history.pushState({}, "", pathname);
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  await act(async () => {
    root.render(<RootRouter />);
    await Promise.resolve();
  });

  const page = { container, root };
  mounted.push(page);
  return page;
}

describe("RootRouter", () => {
  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  });

  afterEach(() => {
    for (const item of mounted.splice(0)) {
      act(() => item.root.unmount());
      item.container.remove();
    }
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("打开主应用时只渲染主应用，不应启动云端登录流程", async () => {
    const { container } = await renderRootRouter("/");

    expect(container.textContent).toContain("主应用");
    expect(container.querySelector('[data-testid="toaster"]')).not.toBeNull();
  });

  it("独立更新窗口从 index.html 入口启动时应映射到更新提醒页", async () => {
    const { container } = await renderRootRouter(
      "/index.html?lime_window=update-notification&latest=1.58.0",
    );

    expect(
      container.querySelector('[data-testid="update-notification-page"]'),
    ).not.toBeNull();
  });

  it("独立资源管理器窗口从 index.html 入口启动时应映射到资源管理器页", async () => {
    const { container } = await renderRootRouter(
      "/index.html?lime_window=resource-manager&session=resource-session-1",
    );

    expect(
      container.querySelector('[data-testid="resource-manager-page"]'),
    ).not.toBeNull();
    expect(container.querySelector('[data-testid="main-app"]')).toBeNull();
  });

  it("打包后的文件路径 index.html 入口也应映射到更新提醒页", async () => {
    const { container } = await renderRootRouter(
      "/Applications/Lime.app/Contents/Resources/app.asar/dist/index.html?lime_window=update-notification&latest=1.58.0",
    );

    expect(
      container.querySelector('[data-testid="update-notification-page"]'),
    ).not.toBeNull();
    expect(container.querySelector('[data-testid="main-app"]')).toBeNull();
  });
});
