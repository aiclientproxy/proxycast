import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { afterEach, describe, expect, it, vi } from "vitest";

const { openExternalMock } = vi.hoisted(() => ({
  openExternalMock: vi.fn(),
}));

vi.mock("./electronRuntime", () => ({
  app: {
    getName: () => "Lime",
    getVersion: () => "0.0.0-test",
    isPackaged: true,
  },
  systemPreferences: {
    isTrustedAccessibilityClient: () => false,
  },
  shell: {
    openExternal: openExternalMock,
  },
}));

import { SystemUtilityHost } from "./systemUtilityHost";

const tempDirs: string[] = [];

async function createTempUserDataDir() {
  const dir = await mkdtemp(path.join(os.tmpdir(), "lime-system-utility-"));
  tempDirs.push(dir);
  return dir;
}

function createHost(
  appDataRoot: string,
  config: Record<string, unknown> = {},
): SystemUtilityHost {
  return new SystemUtilityHost({
    appDataRoot,
    readConfig: async () => config,
  });
}

afterEach(async () => {
  vi.clearAllMocks();
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir) {
      await rm(dir, { recursive: true, force: true });
    }
  }
});

describe("SystemUtilityHost", () => {
  it("openExternalUrl 只允许 http/https 并通过系统浏览器打开", async () => {
    const host = createHost(await createTempUserDataDir());
    openExternalMock.mockResolvedValueOnce(undefined);

    await expect(
      host.openExternalUrl({ url: " https://user.limeai.run/login " }),
    ).resolves.toEqual({});
    await expect(
      host.openExternalUrl({ url: "file:///tmp/token" }),
    ).rejects.toThrow("外部链接只支持 http/https 地址");

    expect(openExternalMock).toHaveBeenCalledTimes(1);
    expect(openExternalMock).toHaveBeenCalledWith(
      "https://user.limeai.run/login",
    );
  });

  it("openSystemSettingsUrl 只允许系统设置 scheme", async () => {
    const host = createHost(await createTempUserDataDir());
    openExternalMock.mockResolvedValue(undefined);

    await expect(
      host.openSystemSettingsUrl({
        url: " x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility ",
      }),
    ).resolves.toEqual({});
    await expect(
      host.openSystemSettingsUrl({
        url: "ms-settings:clipboard",
      }),
    ).resolves.toEqual({});
    await expect(
      host.openSystemSettingsUrl({
        url: "https://example.com/settings",
      }),
    ).rejects.toThrow(
      "系统设置链接只支持 x-apple.systempreferences 或 ms-settings scheme",
    );

    expect(openExternalMock).toHaveBeenCalledTimes(2);
    expect(openExternalMock).toHaveBeenNthCalledWith(
      1,
      "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    );
    expect(openExternalMock).toHaveBeenNthCalledWith(
      2,
      "ms-settings:clipboard",
    );
  });

  it("getEnvironmentPreview 返回 current 环境预览且不暴露 diagnostic facade", async () => {
    const host = createHost(await createTempUserDataDir(), {
      server: {
        host: "127.0.0.1",
        port: 8910,
        api_key: "secret-key",
      },
    });

    const result = await host.getEnvironmentPreview();
    const shellImport = result.shellImport as Record<string, unknown>;
    const desktopCapabilities = result.desktopCapabilities as Record<
      string,
      unknown
    >;

    expect(result).not.toHaveProperty("diagnostic");
    expect(desktopCapabilities).toEqual(
      expect.objectContaining({
        applicationId: "com.limecloud.lime",
        packaged: true,
        capabilities: expect.objectContaining({
          accessibility: expect.objectContaining({
            status: process.platform === "darwin" ? "not_granted" : "unsupported",
          }),
          applicationGroups: expect.objectContaining({
            status: "not_configured",
            identifiers: [],
          }),
        }),
      }),
    );
    expect(JSON.stringify(desktopCapabilities)).not.toContain("openai");
    expect(shellImport).not.toHaveProperty("diagnostic");
    expect(shellImport).toEqual(
      expect.objectContaining({
        enabled: false,
        status: "disabled",
        importedCount: 0,
        durationMs: null,
      }),
    );
    expect(result.entries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: "LIME_API_BASE",
          value: "http://127.0.0.1:8910",
          maskedValue: "http://127.0.0.1:8910",
          sensitive: false,
        }),
        expect.objectContaining({
          key: "LIME_API_KEY",
          value: "secret-key",
          maskedValue: "********",
          sensitive: true,
        }),
      ]),
    );
  });

  it("浏览器诊断占位保持 degraded diagnostic 形态", async () => {
    const appDataRoot = await createTempUserDataDir();
    const host = createHost(appDataRoot);

    expect(host.getBrowserConnectorSettings()).toEqual(
      expect.objectContaining({
        enabled: true,
        install_root_dir: path.join(appDataRoot, "connectors", "browser"),
        diagnostic: expect.objectContaining({
          command: "get_browser_connector_settings_cmd",
        }),
      }),
    );
    expect(host.getBrowserConnectorInstallStatus()).toEqual(
      expect.objectContaining({
        status: "not_installed",
        bundled_version: "0.0.0-test",
        diagnostic: expect.objectContaining({
          command: "get_browser_connector_install_status_cmd",
        }),
      }),
    );
    expect(host.getChromeBridgeEndpointInfo()).toEqual(
      expect.objectContaining({
        server_running: false,
        diagnostic: expect.objectContaining({
          command: "get_chrome_bridge_endpoint_info",
        }),
      }),
    );
    expect(host.getBrowserBackendsStatus()).toEqual(
      expect.objectContaining({
        policy: expect.objectContaining({
          priority: ["lime_extension_bridge", "cdp_direct"],
        }),
        backends: expect.arrayContaining([
          expect.objectContaining({
            backend: "lime_extension_bridge",
            available: false,
          }),
        ]),
        diagnostic: expect.objectContaining({
          command: "get_browser_backends_status",
        }),
      }),
    );

    const profileSessions = host.getChromeProfileSessions();
    expect(profileSessions).toEqual([]);
    expect(
      Object.getOwnPropertyDescriptor(profileSessions, "__diagnostic"),
    ).toEqual(
      expect.objectContaining({
        enumerable: false,
        value: expect.objectContaining({
          command: "get_chrome_profile_sessions",
          status: "degraded",
        }),
      }),
    );
  });
});
