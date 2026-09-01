import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  DESKTOP_RESOURCES_MANIFEST_NAME,
  buildDesktopResourcesManifest,
  desktopResourcePlatformKey,
  preparePackagedDesktopResources,
} from "./electron-desktop-resources.mjs";

const roots = [];

afterEach(() => {
  while (roots.length > 0) {
    rmSync(roots.pop(), { recursive: true, force: true });
  }
});

function createWindowsPackage() {
  const root = mkdtempSync(path.join(tmpdir(), "lime-desktop-resources-"));
  roots.push(root);
  const resources = path.join(root, "resources");
  const sidecars = path.join(resources, "app-server", "win32-x64");
  mkdirSync(sidecars, { recursive: true });
  for (const name of [
    "app-server.exe",
    "code-mode-host.exe",
    "windows-sandbox-setup.exe",
    "windows-sandbox-runner.exe",
  ]) {
    writeFileSync(path.join(sidecars, name), name);
  }
  writeFileSync(
    path.join(resources, "app-server.release.json"),
    JSON.stringify({ artifacts: [{ platform: "win32-x64" }] }),
  );
  return { root, resources };
}

describe("electron desktop resource manifest", () => {
  it("Windows native host 使用 SDK UIA_HWND 类型接收窗口句柄", () => {
    const source = readFileSync(
      "electron/native/windows/windows-native-host.cpp",
      "utf8",
    );

    expect(source).toContain("UIA_HWND nativeWindowHandle = nullptr;");
    expect(source).not.toContain("int nativeWindowHandle = 0;");
    expect(source).not.toMatch(
      /^\s*HWND nativeWindowHandle = nullptr;\s*$/mu,
    );
  });

  it("按平台和架构生成稳定的资源清单", () => {
    expect(desktopResourcePlatformKey("darwin", "arm64")).toBe("darwin-arm64");
    expect(desktopResourcePlatformKey("win32", "x64")).toBe("win32-x64");
  });

  it("Windows packaged resources 包含 sidecar 安全资源组", () => {
    const { root, resources } = createWindowsPackage();
    const compileCalls = [];
    const result = preparePackagedDesktopResources({
      buildPath: root,
      platform: "win32",
      arch: "x64",
      version: "1.2.3",
      execFileSyncImpl: (command, args) => {
        compileCalls.push([command, args]);
        const outputArg = args.find((arg) => String(arg).startsWith("/Fe:"));
        const outputPath = outputArg
          ? String(outputArg).slice("/Fe:".length)
          : String(args.at(-1));
        writeFileSync(outputPath, "compiled windows native host");
      },
    });

    expect(result.manifestPath).toBe(
      path.join(resources, DESKTOP_RESOURCES_MANIFEST_NAME),
    );
    expect(result.manifest).toMatchObject({
      schemaVersion: 1,
      applicationId: "com.limecloud.lime",
      platform: "win32",
      arch: "x64",
      platformKey: "win32-x64",
    });
    expect(result.manifest.resources.map((resource) => resource.id)).toEqual([
      "app-server",
      "code-mode-host",
      "windows-sandbox-setup",
      "windows-sandbox-runner",
      "windows-native-host",
    ]);
    expect(result.manifest.native.windowsHelper).toMatchObject({
      id: "windows-native-host",
      path: "native/windows/windows-native-host.exe",
      api: expect.arrayContaining(["UIAutomation", "RawInput"]),
      readOnly: true,
      signedByForge: true,
    });
    expect(result.manifest.native.windowsHelper.api).toEqual(
      expect.arrayContaining([
        "WindowEnumeration",
        "DisplayEnumeration",
        "DisplayWatcher",
      ]),
    );
    expect(compileCalls).toHaveLength(1);
    expect(compileCalls[0][0]).toBe("cl");
    expect(compileCalls[0][1]).toEqual(
      expect.arrayContaining([
        expect.stringContaining(
          "electron/native/windows/windows-native-host.cpp",
        ),
        "uiautomationcore.lib",
      ]),
    );
  });

  it("macOS packaged resources 编译并登记原生 helper", () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-macos-resources-"));
    roots.push(root);
    const resources = path.join(root, "Lime.app", "Contents", "Resources");
    const sidecars = path.join(resources, "app-server", "darwin-arm64");
    mkdirSync(sidecars, { recursive: true });
    writeFileSync(path.join(sidecars, "app-server"), "app-server");
    writeFileSync(path.join(sidecars, "code-mode-host"), "code-mode-host");
    writeFileSync(
      path.join(resources, "app-server.release.json"),
      JSON.stringify({ artifacts: [{ platform: "darwin-arm64" }] }),
    );

    const compileCalls = [];
    const result = preparePackagedDesktopResources({
      buildPath: root,
      platform: "darwin",
      arch: "arm64",
      version: "1.2.3",
      execFileSyncImpl: (command, args) => {
        compileCalls.push([command, args]);
        writeFileSync(args.at(-1), "compiled native host");
      },
    });

    expect(
      readFileSync(
        path.join(
          resources,
          "native/macos/macos-native-host.app/Contents/MacOS/macos-native-host",
        ),
        "utf8",
      ),
    ).toBe("compiled native host");
    expect(
      readFileSync(
        path.join(
          resources,
          "native/macos/macos-native-host.app/Contents/Info.plist",
        ),
        "utf8",
      ),
    ).toContain("com.limecloud.lime.native-host");
    expect(
      readFileSync(
        path.join(
          resources,
          "native/macos/macos-native-host.app/Contents/Info.plist",
        ),
        "utf8",
      ),
    ).toContain("NSCameraUsageDescription");
    expect(
      readFileSync(
        path.join(
          resources,
          "native/macos/macos-native-host.app/Contents/Info.plist",
        ),
        "utf8",
      ),
    ).toContain("NSAppleEventsUsageDescription");
    expect(result.manifest.resources.map((resource) => resource.id)).toContain(
      "macos-native-host",
    );
    expect(result.manifest.native.helper).toMatchObject({
      id: "macos-native-host",
      signedByForge: true,
      frameworks: expect.arrayContaining([
        "AVFoundation",
        "CoreGraphics",
        "IOKit",
        "LocalAuthentication",
        "Security",
      ]),
    });
    expect(result.manifest.native.entitlements).toMatchObject({
      source: "lime-rs/entitlements.plist",
      applicationGroups: [],
      automationAppleEvents: true,
    });
    expect(compileCalls).toHaveLength(1);
    expect(compileCalls[0][0]).toBe("swiftc");
    expect(compileCalls[0][1]).toEqual(
      expect.arrayContaining([
        "AVFoundation",
        expect.stringContaining(
          "electron/native/macos/macos-native-host.swift",
        ),
        expect.stringContaining(
          "electron/native/macos/macos-screen-capture.swift",
        ),
        expect.stringContaining(
          "electron/native/macos/macos-window-orchestration.swift",
        ),
        expect.stringContaining(
          "electron/native/macos/macos-accessibility-tree.swift",
        ),
        expect.stringContaining(
          "electron/native/macos/macos-display-watcher.swift",
        ),
        expect.stringContaining(
          "electron/native/macos/macos-media-permissions.swift",
        ),
        expect.stringContaining(
          "electron/native/macos/macos-apple-events.swift",
        ),
      ]),
    );
  });

  it("拒绝越过 resources 根的资源路径", () => {
    const root = mkdtempSync(path.join(tmpdir(), "lime-desktop-manifest-"));
    roots.push(root);
    writeFileSync(path.join(root, "outside"), "outside");

    expect(() =>
      buildDesktopResourcesManifest({
        resourcesRoot: root,
        platform: "win32",
        arch: "x64",
        version: "1.2.3",
        files: { escape: "../outside" },
      }),
    ).toThrow(/must stay inside resources/u);
  });
});
