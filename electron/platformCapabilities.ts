import { app, systemPreferences } from "./electronRuntime";
import {
  readDesktopResourceReadiness,
  type DesktopResourceReadiness,
} from "./desktopResourceReadiness";
import { resolveMacOSNativeHostPath } from "./macosNativeHost";
import { resolveWindowsNativeHostPath } from "./windowsNativeHost";

export const DESKTOP_CAPABILITIES_SCHEMA_VERSION = 1 as const;
export const LIME_APP_BUNDLE_ID = "com.limecloud.lime";

export type DesktopCapabilityStatus =
  | "ready"
  | "not_granted"
  | "not_configured"
  | "unverified"
  | "unsupported"
  | "unavailable";

export type DesktopCapability = {
  status: DesktopCapabilityStatus;
  reason: string;
  settingsUrl?: string | null;
};

export type DesktopCapabilities = {
  schemaVersion: typeof DESKTOP_CAPABILITIES_SCHEMA_VERSION;
  platform: string;
  arch: string;
  applicationId: string;
  appVersion: string;
  packaged: boolean;
  capabilities: {
    accessibility: DesktopCapability;
    inputMonitoring: DesktopCapability;
    applicationGroups: DesktopCapability & { identifiers: string[] };
    nativeModules: DesktopCapability & { identifiers: string[] };
    windowHandles: DesktopCapability;
    windowOrchestration: DesktopCapability;
    accessibilityTree: DesktopCapability;
    displays: DesktopCapability;
    displayWatcher: DesktopCapability;
    hidTopology: DesktopCapability;
    bareModifierMonitor: DesktopCapability;
    screenCapture: DesktopCapability;
    appleEvents: DesktopCapability;
    uiAutomation: DesktopCapability;
    rawInput: DesktopCapability;
    mediaPermissions: {
      microphone: DesktopCapability;
      camera: DesktopCapability;
    };
    securityScopedBookmarks: DesktopCapability;
    deviceKey: DesktopCapability;
    localAuthentication: DesktopCapability;
    resourceManifest: DesktopCapability & {
      identifiers: string[];
      platformKey: string | null;
    };
    sidecars: DesktopCapability & { identifiers: string[] };
    codeMode: DesktopCapability;
    updates: DesktopCapability;
    sandbox: DesktopCapability;
  };
};

export type DesktopCapabilitiesInput = {
  platform?: string;
  arch?: string;
  applicationId?: string;
  appVersion?: string;
  packaged?: boolean;
  accessibilityTrusted?: boolean | null;
  nativeHostReady?: boolean;
  windowsNativeHostReady?: boolean;
  resourceManifestReady?: boolean;
};

const ACCESSIBILITY_SETTINGS_URL =
  "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility";
const INPUT_MONITORING_SETTINGS_URL =
  "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent";

function accessibilityCapability(
  platform: string,
  accessibilityTrusted: boolean | null | undefined,
): DesktopCapability {
  if (platform !== "darwin") {
    return {
      status: "unsupported",
      reason: "Accessibility readiness is only queried on macOS.",
      settingsUrl: null,
    };
  }
  if (accessibilityTrusted === true) {
    return {
      status: "ready",
      reason: "macOS Accessibility permission is granted.",
      settingsUrl: ACCESSIBILITY_SETTINGS_URL,
    };
  }
  if (accessibilityTrusted === false) {
    return {
      status: "not_granted",
      reason: "macOS Accessibility permission is not granted.",
      settingsUrl: ACCESSIBILITY_SETTINGS_URL,
    };
  }
  return {
    status: "unavailable",
    reason: "macOS Accessibility permission could not be queried.",
    settingsUrl: ACCESSIBILITY_SETTINGS_URL,
  };
}

export function buildDesktopCapabilities(
  input: DesktopCapabilitiesInput = {},
): DesktopCapabilities {
  const platform = input.platform ?? process.platform;
  const isMacOS = platform === "darwin";
  const isWindows = platform === "win32";
  const accessibility = accessibilityCapability(
    platform,
    input.accessibilityTrusted,
  );
  const nativeHostReady =
    isMacOS && (input.nativeHostReady ?? Boolean(resolveMacOSNativeHostPath()));
  const windowsNativeHostReady =
    isWindows &&
    (input.windowsNativeHostReady ?? Boolean(resolveWindowsNativeHostPath()));
  const packaged = input.packaged ?? false;
  const resourceReadiness: DesktopResourceReadiness =
    input.resourceManifestReady === undefined
      ? readDesktopResourceReadiness({
          platform,
          arch: input.arch ?? process.arch,
          packaged,
        })
      : input.resourceManifestReady
        ? {
            status: "unverified",
            reason:
              "Desktop resource manifest readiness was supplied by the packaged test harness.",
            platformKey: `${platform}-${input.arch ?? process.arch}`,
            resourceIds: [],
          }
        : {
            status: "not_configured",
            reason:
              "Desktop resource manifest readiness is disabled for this capability evaluation.",
            platformKey: null,
            resourceIds: [],
          };
  const resourceManifest = {
    status: resourceReadiness.status,
    reason: resourceReadiness.reason,
    settingsUrl: null,
    identifiers: resourceReadiness.resourceIds,
    platformKey: resourceReadiness.platformKey,
  } satisfies DesktopCapabilities["capabilities"]["resourceManifest"];
  const sidecars = sidecarCapability(resourceReadiness);

  return {
    schemaVersion: DESKTOP_CAPABILITIES_SCHEMA_VERSION,
    platform,
    arch: input.arch ?? process.arch,
    applicationId: input.applicationId ?? LIME_APP_BUNDLE_ID,
    appVersion: input.appVersion ?? "unknown",
    packaged,
    capabilities: {
      accessibility,
      inputMonitoring: {
        status: isMacOS
          ? nativeHostReady
            ? "unverified"
            : "not_configured"
          : "unsupported",
        reason: isMacOS
          ? nativeHostReady
            ? "Input Monitoring is queried by the verified macOS native host."
            : "Lime has no verified macOS Input Monitoring helper."
          : "Input Monitoring is a macOS-specific capability.",
        settingsUrl: isMacOS ? INPUT_MONITORING_SETTINGS_URL : null,
      },
      applicationGroups: {
        status: "not_configured",
        reason:
          "No Lime native helper or extension currently requires an Application Group.",
        identifiers: [],
        settingsUrl: null,
      },
      nativeModules: {
        status:
          nativeHostReady || windowsNativeHostReady ? "ready" : "not_configured",
        reason: nativeHostReady
          ? "The signed macOS native host resource is present and hash-verified."
          : windowsNativeHostReady
            ? "The Windows native host resource is present and hash-verified."
            : "No verified Lime platform-native module is bundled for this capability contract.",
        identifiers: [
          ...(nativeHostReady ? ["macos-native-host"] : []),
          ...(windowsNativeHostReady ? ["windows-native-host"] : []),
        ],
        settingsUrl: null,
      },
      windowHandles: desktopNativeCapability(
        platform,
        nativeHostReady,
        windowsNativeHostReady,
        "Native window handle and CGWindow queries are provided by the macOS host.",
        "Native window enumeration and HWND geometry queries are provided by the Windows host.",
      ),
      windowOrchestration: nativeCapability(
        isMacOS,
        nativeHostReady,
        "Window anchoring, stacking and hide-for-task leases are provided by the macOS host.",
      ),
      accessibilityTree: nativeCapability(
        isMacOS,
        nativeHostReady,
        "Read-only Accessibility tree observation is provided by the macOS host.",
      ),
      displays: desktopNativeCapability(
        platform,
        nativeHostReady,
        windowsNativeHostReady,
        "Native display geometry and scale queries are provided by the macOS host.",
        "Native monitor geometry and work-area queries are provided by the Windows host.",
      ),
      displayWatcher: isMacOS
        ? nativeCapability(
            true,
            nativeHostReady,
            "CoreGraphics display reconfiguration events are provided by the macOS host.",
          )
        : windowsNativeCapability(
            isWindows,
            windowsNativeHostReady,
            "WM_DISPLAYCHANGE monitor topology events are provided by the Windows host.",
          ),
      hidTopology: nativeCapability(
        isMacOS,
        nativeHostReady,
        "IOHID topology queries and change events are provided by the macOS host.",
      ),
      bareModifierMonitor: nativeCapability(
        isMacOS,
        nativeHostReady,
        "Bare modifier monitoring is provided by the macOS host after Input Monitoring authorization.",
      ),
      screenCapture: nativeCapability(
        isMacOS,
        nativeHostReady,
        "Screen Recording permission is queried by the macOS host before capture is enabled.",
      ),
      appleEvents: nativeCapability(
        isMacOS,
        nativeHostReady,
        "Apple Events authorization queries and explicit consent requests are provided by the macOS host.",
      ),
      uiAutomation: windowsNativeCapability(
        isWindows,
        windowsNativeHostReady,
        "Windows UI Automation read-only tree observation is provided by the native host.",
      ),
      rawInput: windowsNativeCapability(
        isWindows,
        windowsNativeHostReady,
        "Windows Raw Input modifier observation is provided by the native host without input injection.",
      ),
      mediaPermissions: {
        microphone: mediaPermissionCapability(
          platform,
          nativeHostReady,
          "microphone",
        ),
        camera: mediaPermissionCapability(platform, nativeHostReady, "camera"),
      },
      securityScopedBookmarks: nativeCapability(
        isMacOS,
        nativeHostReady,
        "Security-scoped bookmark lifecycle is provided by the macOS host and managed storage owner.",
      ),
      deviceKey: {
        status:
          isMacOS && nativeHostReady
            ? "unverified"
            : isMacOS
              ? "not_configured"
              : "unsupported",
        reason: isMacOS
          ? nativeHostReady
            ? "Secure Enclave device-key operations require signed entitlements and compatible hardware."
            : "Lime has no verified macOS native host for device-key operations."
          : "Secure Enclave device keys are a macOS-specific capability.",
        settingsUrl: null,
      },
      localAuthentication: nativeCapability(
        isMacOS,
        nativeHostReady,
        "LocalAuthentication operations are provided by the macOS native host.",
      ),
      resourceManifest,
      sidecars,
      codeMode: codeModeCapability(platform, resourceReadiness),
      updates: updateCapability(platform, packaged),
      sandbox: {
        status: isWindows || isMacOS ? "unverified" : "unsupported",
        reason: isWindows
          ? "Windows sandbox readiness is owned by tool-runtime and packaged resource verification."
          : isMacOS
            ? "macOS seatbelt readiness is owned by tool-runtime and is not inferred from Electron."
            : "No platform sandbox readiness is declared by Desktop Host.",
        settingsUrl: null,
      },
    },
  };
}

function sidecarCapability(
  resourceReadiness: ReturnType<typeof readDesktopResourceReadiness>,
): DesktopCapabilities["capabilities"]["sidecars"] {
  if (resourceReadiness.status === "unsupported") {
    return {
      status: "unsupported",
      reason: resourceReadiness.reason,
      identifiers: [],
      settingsUrl: null,
    };
  }
  return {
    status: resourceReadiness.status,
    reason:
      resourceReadiness.status === "unverified"
        ? "Required desktop sidecars are present; digest, signature and runtime Gate B remain release checks."
        : resourceReadiness.reason,
    identifiers: resourceReadiness.resourceIds.filter(
      (id) => id !== "macos-native-host" && id !== "windows-native-host",
    ),
    settingsUrl: null,
  };
}

function codeModeCapability(
  platform: string,
  resourceReadiness: ReturnType<typeof readDesktopResourceReadiness>,
): DesktopCapability {
  if (platform !== "darwin" && platform !== "win32") {
    return {
      status: "unsupported",
      reason:
        "Code Mode packaged readiness is only defined for macOS and Windows.",
      settingsUrl: null,
    };
  }
  if (resourceReadiness.status === "unverified") {
    return {
      status: "unverified",
      reason:
        "The Code Mode host resource is present; sandbox/V8 execution requires packaged Gate B.",
      settingsUrl: null,
    };
  }
  return {
    status: resourceReadiness.status,
    reason: resourceReadiness.reason,
    settingsUrl: null,
  };
}

function updateCapability(
  platform: string,
  packaged: boolean,
): DesktopCapability {
  if (platform !== "darwin" && platform !== "win32") {
    return {
      status: "unsupported",
      reason: "Desktop update readiness is only defined for macOS and Windows.",
      settingsUrl: null,
    };
  }
  if (!packaged) {
    return {
      status: "not_configured",
      reason: "Packaged update feeds are not enabled for development builds.",
      settingsUrl: null,
    };
  }
  return {
    status: "unverified",
    reason:
      "Forge updater configuration is present; signed feed and install Gate B remain release checks.",
    settingsUrl: null,
  };
}

function nativeCapability(
  isMacOS: boolean,
  nativeHostReady: boolean,
  readyReason: string,
): DesktopCapability {
  if (!isMacOS) {
    return {
      status: "unsupported",
      reason: "This capability is only available on macOS.",
      settingsUrl: null,
    };
  }
  return {
    status: nativeHostReady ? "unverified" : "not_configured",
    reason: nativeHostReady
      ? readyReason
      : "Lime has no verified macOS native host for this capability.",
    settingsUrl: null,
  };
}

function desktopNativeCapability(
  platform: string,
  macOSHostReady: boolean,
  windowsHostReady: boolean,
  macOSReason: string,
  windowsReason: string,
): DesktopCapability {
  if (platform === "darwin") {
    return {
      status: macOSHostReady ? "unverified" : "not_configured",
      reason: macOSHostReady
        ? macOSReason
        : "Lime has no verified macOS native host for this capability.",
      settingsUrl: null,
    };
  }
  if (platform === "win32") {
    return {
      status: windowsHostReady ? "unverified" : "not_configured",
      reason: windowsHostReady
        ? windowsReason
        : "Lime has no verified Windows native host for this capability.",
      settingsUrl: null,
    };
  }
  return {
    status: "unsupported",
    reason: "This capability is only available on macOS and Windows.",
    settingsUrl: null,
  };
}

function mediaPermissionCapability(
  platform: string,
  nativeHostReady: boolean,
  kind: "microphone" | "camera",
): DesktopCapability {
  const settingsUrl =
    platform === "darwin"
      ? kind === "microphone"
        ? "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        : "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
      : platform === "win32"
        ? kind === "microphone"
          ? "ms-settings:privacy-microphone"
          : "ms-settings:privacy-webcam"
        : null;
  if (platform === "darwin") {
    return {
      status: nativeHostReady ? "unverified" : "not_configured",
      reason: nativeHostReady
        ? `${kind === "microphone" ? "Microphone" : "Camera"} permission query and request are provided by the macOS host.`
        : "Lime has no verified macOS native host for media permissions.",
      settingsUrl,
    };
  }
  if (platform === "win32") {
    return {
      status: "unverified",
      reason:
        "Windows media permission requests are handled by the Electron main-window permission owner; packaged OS authorization remains a release check.",
      settingsUrl,
    };
  }
  return {
    status: "unsupported",
    reason:
      "Desktop media permissions are only implemented for macOS and Windows.",
    settingsUrl: null,
  };
}

function windowsNativeCapability(
  isWindows: boolean,
  nativeHostReady: boolean,
  readyReason: string,
): DesktopCapability {
  if (!isWindows) {
    return {
      status: "unsupported",
      reason: "This capability is only available on Windows.",
      settingsUrl: null,
    };
  }
  return {
    status: nativeHostReady ? "unverified" : "not_configured",
    reason: nativeHostReady
      ? readyReason
      : "Lime has no verified Windows native host for this capability.",
    settingsUrl: null,
  };
}

function readAccessibilityTrusted(): boolean | null {
  if (process.platform !== "darwin") {
    return null;
  }
  try {
    return systemPreferences.isTrustedAccessibilityClient(false);
  } catch {
    return null;
  }
}

export function readDesktopCapabilities(): DesktopCapabilities {
  return buildDesktopCapabilities({
    appVersion: app.getVersion(),
    packaged: app.isPackaged,
    accessibilityTrusted: readAccessibilityTrusted(),
    nativeHostReady: Boolean(resolveMacOSNativeHostPath()),
    windowsNativeHostReady: Boolean(resolveWindowsNativeHostPath()),
  });
}
