import { describe, expect, it, vi } from "vitest";
import {
  PLUGIN_CATALOG_CHANGED_EVENT,
  installPluginCatalog,
  listInstalledPluginCatalog,
  listPluginCatalog,
  readPluginCatalog,
  searchPluginCatalog,
  setPluginCatalogEnabled,
  uninstallPluginCatalog,
  type PluginCatalogClient,
} from "./pluginCatalog";

function client(): PluginCatalogClient {
  return {
    listPluginCatalog: vi
      .fn()
      .mockResolvedValue({ result: { plugins: [], generatedAt: "now" } }),
    searchPlugins: vi
      .fn()
      .mockResolvedValue({ result: { data: [], nextCursor: null } }),
    setPluginCatalogEnabled: vi
      .fn()
      .mockResolvedValue({ result: { plugin: {} } }),
    readPluginCatalog: vi.fn().mockResolvedValue({ result: { plugin: {} } }),
    installPluginCatalog: vi.fn().mockResolvedValue({ result: { plugin: {} } }),
    uninstallPluginCatalog: vi
      .fn()
      .mockResolvedValue({ result: { pluginId: "demo", uninstalled: true } }),
    listInstalledPluginCatalog: vi
      .fn()
      .mockResolvedValue({ result: { plugins: [], generatedAt: "now" } }),
  };
}

describe("Plugin v2 catalog API", () => {
  it("只通过 typed App Server client 访问 catalog/install owner", async () => {
    const appServerClient = client();

    await listPluginCatalog({ query: "demo" }, appServerClient);
    await searchPluginCatalog({ searchTerm: "demo" }, appServerClient);
    await readPluginCatalog({ pluginId: "demo" }, appServerClient);
    await setPluginCatalogEnabled(
      { pluginId: "demo", enabled: true },
      appServerClient,
    );
    await installPluginCatalog(
      { sourcePath: "/tmp/demo-plugin" },
      appServerClient,
    );
    await listInstalledPluginCatalog({}, appServerClient);
    await uninstallPluginCatalog({ pluginId: "demo" }, appServerClient);

    expect(appServerClient.listPluginCatalog).toHaveBeenCalledWith({
      query: "demo",
    });
    expect(appServerClient.searchPlugins).toHaveBeenCalledWith({
      searchTerm: "demo",
    });
    expect(appServerClient.setPluginCatalogEnabled).toHaveBeenCalledWith({
      pluginId: "demo",
      enabled: true,
    });
    expect(appServerClient.readPluginCatalog).toHaveBeenCalledWith({
      pluginId: "demo",
    });
    expect(appServerClient.installPluginCatalog).toHaveBeenCalledWith({
      sourcePath: "/tmp/demo-plugin",
    });
    expect(appServerClient.listInstalledPluginCatalog).toHaveBeenCalledWith({});
    expect(appServerClient.uninstallPluginCatalog).toHaveBeenCalledWith({
      pluginId: "demo",
    });
  });

  it("安装、启停和卸载后应通知 v2 catalog 消费者刷新", async () => {
    const appServerClient = client();
    const listener = vi.fn();
    window.addEventListener(PLUGIN_CATALOG_CHANGED_EVENT, listener);

    try {
      await installPluginCatalog(
        { sourcePath: "/tmp/demo-plugin" },
        appServerClient,
      );
      await setPluginCatalogEnabled(
        { pluginId: "demo", enabled: false },
        appServerClient,
      );
      await uninstallPluginCatalog({ pluginId: "demo" }, appServerClient);
    } finally {
      window.removeEventListener(PLUGIN_CATALOG_CHANGED_EVENT, listener);
    }

    expect(listener).toHaveBeenCalledTimes(3);
  });
});
