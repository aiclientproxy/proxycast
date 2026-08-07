import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { AppServerPluginCatalogSummary } from "@/lib/api/appServerTypes";
import { PluginCatalogPage } from "./PluginCatalogPage";

const mocks = vi.hoisted(() => ({
  appsSubscription: null as null | {
    onUpdate: (apps: unknown[]) => void;
  },
  installPluginCatalog: vi.fn(),
  listPluginCatalog: vi.fn(),
  readAppsReadiness: vi.fn(),
  readPluginCatalog: vi.fn(),
  selectPluginCatalogSource: vi.fn(),
  setPluginCatalogEnabled: vi.fn(),
  subscribeAppsListUpdates: vi.fn(),
  uninstallPluginCatalog: vi.fn(),
  toastSuccess: vi.fn(),
}));

vi.mock("@/lib/api/apps", () => ({
  readAppsReadiness: (...args: unknown[]) => mocks.readAppsReadiness(...args),
  subscribeAppsListUpdates: (...args: unknown[]) =>
    mocks.subscribeAppsListUpdates(...args),
}));

vi.mock("sonner", () => ({
  toast: {
    success: (...args: unknown[]) => mocks.toastSuccess(...args),
  },
}));

vi.mock("@/lib/api/pluginCatalog", () => ({
  installPluginCatalog: (...args: unknown[]) =>
    mocks.installPluginCatalog(...args),
  listPluginCatalog: (...args: unknown[]) => mocks.listPluginCatalog(...args),
  readPluginCatalog: (...args: unknown[]) => mocks.readPluginCatalog(...args),
  selectPluginCatalogSource: (...args: unknown[]) =>
    mocks.selectPluginCatalogSource(...args),
  setPluginCatalogEnabled: (...args: unknown[]) =>
    mocks.setPluginCatalogEnabled(...args),
  uninstallPluginCatalog: (...args: unknown[]) =>
    mocks.uninstallPluginCatalog(...args),
}));

function summary(
  overrides: Partial<AppServerPluginCatalogSummary> = {},
): AppServerPluginCatalogSummary {
  return {
    appsCount: 1,
    authPolicy: "ON_USE",
    availability: "installed",
    description: "Writing and research tools",
    enabled: true,
    hooksCount: 1,
    id: "writer-plugin",
    marketplaceId: "personal",
    contentDigest: "sha256:test",
    installPolicy: "AVAILABLE",
    installed: true,
    localVersion: "1.2.3",
    mcpServersCount: 1,
    name: "Writer Plugin",
    skillsCount: 1,
    source: "local",
    sourceUri: "/tmp/writer-plugin",
    version: "1.2.3",
    ...overrides,
  };
}

const mounted: Array<{ container: HTMLDivElement; root: Root }> = [];

async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function renderPage() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => {
    root.render(<PluginCatalogPage />);
  });
  await flush();
  mounted.push({ container, root });
  return container;
}

async function click(element: Element | null) {
  expect(element).not.toBeNull();
  await act(async () => {
    element?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await Promise.resolve();
  });
  await flush();
}

describe("PluginCatalogPage", () => {
  beforeEach(async () => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
    await changeLimeLocale("zh-CN");
    vi.clearAllMocks();
    mocks.appsSubscription = null;
    mocks.listPluginCatalog.mockResolvedValue({
      generatedAt: "2026-08-04T00:00:00Z",
      plugins: [summary()],
    });
    mocks.readPluginCatalog.mockResolvedValue({
      plugin: {
        summary: summary(),
        skills: [
          {
            id: "article-writing",
            name: "Article Writing",
            description: "",
            requiresAuth: false,
          },
        ],
        mcpServers: [],
        apps: [
          {
            id: "writer-app",
            name: "Writer App",
            description: "",
            requiresAuth: false,
          },
        ],
        hooks: [],
        uiResources: [],
      },
    });
    mocks.setPluginCatalogEnabled.mockResolvedValue({
      plugin: summary({ enabled: false }),
    });
    mocks.readAppsReadiness.mockResolvedValue({
      apps: [
        {
          id: "writer-app",
          name: "Writer App",
          description: null,
          logoUrl: null,
          logoUrlDark: null,
          iconAssets: null,
          iconDarkAssets: null,
          distributionChannel: "local",
          branding: null,
          appMetadata: null,
          labels: null,
          installUrl: null,
          isAccessible: true,
          isEnabled: true,
          pluginDisplayNames: ["Writer Plugin"],
        },
      ],
      installed: [
        {
          id: "writer-app",
          runtimeName: "Writer App",
          enabled: true,
          callable: false,
        },
      ],
      ready: false,
    });
    mocks.subscribeAppsListUpdates.mockImplementation(
      (subscription: { onUpdate: (apps: unknown[]) => void }) => {
        mocks.appsSubscription = subscription;
        return vi.fn();
      },
    );
  });

  afterEach(async () => {
    while (mounted.length) {
      const entry = mounted.pop();
      if (!entry) {
        continue;
      }
      await act(async () => entry.root.unmount());
      entry.container.remove();
    }
  });

  it("从 v2 catalog 展示安装状态并读取详情", async () => {
    const container = await renderPage();

    expect(container.textContent).toContain("插件中心");
    expect(
      container.querySelector('[data-testid="plugin-v2-card-writer-plugin"]'),
    ).not.toBeNull();
    expect(mocks.listPluginCatalog).toHaveBeenCalledWith();
    expect(mocks.readPluginCatalog).toHaveBeenCalledWith({
      pluginId: "writer-plugin",
    });
    expect(container.textContent).toContain("Article Writing");
    expect(
      container
        .querySelector('[data-testid="plugin-v2-app-readiness-writer-app"]')
        ?.getAttribute("data-callable"),
    ).toBe("false");
    expect(container.textContent).toContain("宿主待接入");

    await click(
      container.querySelector(
        '[data-testid="plugin-v2-actions-writer-plugin"]',
      ),
    );
    await click(
      container.querySelector('[data-testid="plugin-v2-toggle-writer-plugin"]'),
    );
    expect(mocks.setPluginCatalogEnabled).toHaveBeenCalledWith({
      pluginId: "writer-plugin",
      enabled: false,
    });
  });

  it("收到 typed app/list/updated 后刷新 Apps readiness", async () => {
    const container = await renderPage();
    mocks.readAppsReadiness.mockResolvedValueOnce({
      apps: [],
      installed: [
        {
          id: "writer-app",
          runtimeName: "Writer App",
          enabled: false,
          callable: false,
        },
      ],
      ready: true,
    });

    await act(async () => {
      mocks.appsSubscription?.onUpdate([]);
      await Promise.resolve();
    });
    await flush();

    expect(mocks.readAppsReadiness).toHaveBeenCalledTimes(2);
    const row = container.querySelector(
      '[data-testid="plugin-v2-app-readiness-writer-app"]',
    );
    expect(row?.getAttribute("data-enabled")).toBe("false");
    expect(row?.getAttribute("data-status")).toBe("disabled");
    expect(row?.textContent).toContain("已停用");
  });

  it("通过原生目录选择和 App Server review 安装本地插件", async () => {
    const available = summary({ installed: false, enabled: false });
    mocks.listPluginCatalog.mockImplementation(
      async (params?: { marketplacePaths?: string[] }) =>
        params?.marketplacePaths
          ? { generatedAt: "now", plugins: [available] }
          : { generatedAt: "now", plugins: [] },
    );
    mocks.selectPluginCatalogSource.mockResolvedValue("/tmp/marketplace");
    mocks.installPluginCatalog.mockResolvedValue({
      plugin: summary(),
    });

    const container = await renderPage();
    await click(
      container.querySelector('[data-testid="plugin-v2-install-local"]'),
    );

    expect(mocks.listPluginCatalog).toHaveBeenLastCalledWith({
      marketplacePaths: ["/tmp/marketplace"],
    });
    expect(
      document.body.querySelector('[data-testid="plugin-v2-install-review"]'),
    ).not.toBeNull();

    await click(
      document.body.querySelector('[data-testid="plugin-v2-confirm-install"]'),
    );
    expect(mocks.installPluginCatalog).toHaveBeenCalledWith({
      sourcePath: "/tmp/writer-plugin",
      marketplaceId: "personal",
      source: "local",
      expectedDigest: "sha256:test",
    });
  });
});
