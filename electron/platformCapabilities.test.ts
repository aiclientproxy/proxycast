import { describe, expect, it } from "vitest";
import { buildDesktopCapabilities } from "./platformCapabilities";

describe("Desktop capability contract", () => {
  it("macOS 只报告真实的 Accessibility 状态，不伪造 OpenAI Application Group", () => {
    const result = buildDesktopCapabilities({
      platform: "darwin",
      arch: "arm64",
      applicationId: "com.limecloud.lime",
      appVersion: "1.2.3",
      packaged: true,
      accessibilityTrusted: false,
      nativeHostReady: false,
      resourceManifestReady: false,
    });

    expect(result).toMatchObject({
      schemaVersion: 1,
      platform: "darwin",
      arch: "arm64",
      applicationId: "com.limecloud.lime",
      appVersion: "1.2.3",
      packaged: true,
      capabilities: {
        accessibility: {
          status: "not_granted",
          settingsUrl:
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        },
        applicationGroups: {
          status: "not_configured",
          identifiers: [],
        },
        nativeModules: {
          status: "not_configured",
          identifiers: [],
        },
        resourceManifest: {
          status: "not_configured",
          identifiers: [],
          platformKey: null,
        },
        sidecars: {
          status: "not_configured",
          identifiers: [],
        },
      },
    });
    expect(result.capabilities.inputMonitoring.settingsUrl).toBe(
      "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    );
    expect(result.capabilities.mediaPermissions).toEqual({
      microphone: expect.objectContaining({ status: "not_configured" }),
      camera: expect.objectContaining({ status: "not_configured" }),
    });
    expect(JSON.stringify(result)).not.toContain("openai");
  });

  it("Windows 将平台 sandbox readiness 交还给 tool-runtime", () => {
    const result = buildDesktopCapabilities({
      platform: "win32",
      arch: "x64",
      applicationId: "com.limecloud.lime",
      appVersion: "1.2.3",
      packaged: true,
      accessibilityTrusted: null,
      resourceManifestReady: false,
    });

    expect(result.capabilities.accessibility.status).toBe("unsupported");
    expect(result.capabilities.inputMonitoring.status).toBe("unsupported");
    expect(result.capabilities.sandbox).toMatchObject({
      status: "unverified",
    });
    expect(result.capabilities.applicationGroups.identifiers).toEqual([]);
    expect(result.capabilities.resourceManifest.status).toBe("not_configured");
    expect(result.capabilities.codeMode.status).toBe("not_configured");
    expect(result.capabilities.updates.status).toBe("unverified");
    expect(result.capabilities.mediaPermissions).toMatchObject({
      microphone: {
        status: "unverified",
        settingsUrl: "ms-settings:privacy-microphone",
      },
      camera: {
        status: "unverified",
        settingsUrl: "ms-settings:privacy-webcam",
      },
    });
  });

  it("Windows native host 存在时只报告 UIA/Raw Input 未验证", () => {
    const result = buildDesktopCapabilities({
      platform: "win32",
      arch: "x64",
      packaged: true,
      resourceManifestReady: true,
      windowsNativeHostReady: true,
    });

    expect(result.capabilities.nativeModules).toMatchObject({
      status: "ready",
      identifiers: ["windows-native-host"],
    });
    expect(result.capabilities.uiAutomation.status).toBe("unverified");
    expect(result.capabilities.rawInput.status).toBe("unverified");
    expect(result.capabilities.windowHandles.status).toBe("unverified");
    expect(result.capabilities.displays.status).toBe("unverified");
    expect(result.capabilities.displayWatcher.status).toBe("unverified");
    expect(result.capabilities.uiAutomation.reason).toContain("read-only");
    expect(result.capabilities.displayWatcher.reason).toContain("WM_DISPLAYCHANGE");
  });

  it("macOS 原生窗口、显示、HID 和设备密钥能力保持真实未验证状态", () => {
    const result = buildDesktopCapabilities({
      platform: "darwin",
      arch: "arm64",
      packaged: true,
      nativeHostReady: true,
      accessibilityTrusted: true,
    });

    expect(result.capabilities.windowHandles.status).toBe("unverified");
    expect(result.capabilities.windowOrchestration.status).toBe("unverified");
    expect(result.capabilities.accessibilityTree.status).toBe("unverified");
    expect(result.capabilities.displays.status).toBe("unverified");
    expect(result.capabilities.displayWatcher.status).toBe("unverified");
    expect(result.capabilities.hidTopology.status).toBe("unverified");
    expect(result.capabilities.bareModifierMonitor.status).toBe("unverified");
    expect(result.capabilities.screenCapture.status).toBe("unverified");
    expect(result.capabilities.appleEvents.status).toBe("unverified");
    expect(result.capabilities.mediaPermissions).toMatchObject({
      microphone: { status: "unverified" },
      camera: { status: "unverified" },
    });
    expect(result.capabilities.securityScopedBookmarks.status).toBe(
      "unverified",
    );
    expect(result.capabilities.deviceKey.status).toBe("unverified");
    expect(result.capabilities.localAuthentication.status).toBe("unverified");
    expect(result.capabilities.deviceKey.reason).toContain("Secure Enclave");
  });

  it("非 Desktop 平台不声明原生 capability readiness", () => {
    const result = buildDesktopCapabilities({
      platform: "linux",
      arch: "x64",
      accessibilityTrusted: null,
    });

    expect(result.capabilities.sandbox.status).toBe("unsupported");
    expect(result.capabilities.nativeModules.status).toBe("not_configured");
    expect(result.capabilities.windowHandles.status).toBe("unsupported");
    expect(result.capabilities.windowOrchestration.status).toBe("unsupported");
    expect(result.capabilities.accessibilityTree.status).toBe("unsupported");
    expect(result.capabilities.bareModifierMonitor.status).toBe("unsupported");
    expect(result.capabilities.screenCapture.status).toBe("unsupported");
    expect(result.capabilities.appleEvents.status).toBe("unsupported");
    expect(result.capabilities.mediaPermissions).toMatchObject({
      microphone: { status: "unsupported" },
      camera: { status: "unsupported" },
    });
    expect(result.capabilities.displayWatcher.status).toBe("unsupported");
    expect(result.capabilities.securityScopedBookmarks.status).toBe(
      "unsupported",
    );
    expect(result.capabilities.deviceKey.status).toBe("unsupported");
    expect(result.capabilities.localAuthentication.status).toBe("unsupported");
    expect(result.capabilities.resourceManifest.status).toBe("unsupported");
    expect(result.capabilities.codeMode.status).toBe("unsupported");
    expect(result.capabilities.updates.status).toBe("unsupported");
  });
});
