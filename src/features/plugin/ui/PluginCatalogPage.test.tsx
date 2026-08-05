import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { AppServerPluginCatalogSummary } from "@/lib/api/appServerTypes";
import { PluginCatalogPage } from "./PluginCatalogPage";

const mocks = vi.hoisted(() => ({
  installPluginCatalog: vi.fn(),
  listPluginCatalog: vi.fn(),
  readPluginCatalog: vi.fn(),
  selectPluginCatalogSource: vi.fn(),
  setPluginCatalogEnabled: vi.fn(),
  uninstallPluginCatalog: vi.fn(),
  toastSuccess: vi.fn(),
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
        apps: [],
        hooks: [],
        uiResources: [],
      },
    });
    mocks.setPluginCatalogEnabled.mockResolvedValue({
      plugin: summary({ enabled: false }),
    });
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
