import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const DESKTOP_RESOURCES_MANIFEST_NAME =
  "desktop-resources.manifest.json";
export const DESKTOP_RESOURCES_SCHEMA_VERSION = 1;
export const DESKTOP_APPLICATION_ID = "com.limecloud.lime";
const REPO_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const MAC_NATIVE_HOST_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/macos/macos-native-host.swift",
);
const MAC_SCREEN_CAPTURE_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/macos/macos-screen-capture.swift",
);
const MAC_SECURITY_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/macos/macos-security.swift",
);
const MAC_WINDOW_ORCHESTRATION_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/macos/macos-window-orchestration.swift",
);
const MAC_ACCESSIBILITY_TREE_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/macos/macos-accessibility-tree.swift",
);
const MAC_DISPLAY_WATCHER_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/macos/macos-display-watcher.swift",
);
const MAC_MEDIA_PERMISSIONS_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/macos/macos-media-permissions.swift",
);
const MAC_APPLE_EVENTS_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/macos/macos-apple-events.swift",
);
const WINDOWS_NATIVE_HOST_SOURCE = path.join(
  REPO_ROOT,
  "electron/native/windows/windows-native-host.cpp",
);
const MAC_NATIVE_HOST_BUNDLE_RELATIVE_PATH =
  "native/macos/macos-native-host.app";
const MAC_NATIVE_HOST_RELATIVE_PATH = `${MAC_NATIVE_HOST_BUNDLE_RELATIVE_PATH}/Contents/MacOS/macos-native-host`;
const WINDOWS_NATIVE_HOST_RELATIVE_PATH = "native/windows/windows-native-host.exe";

export function desktopResourcePlatformKey(
  platform = process.platform,
  arch = process.arch,
) {
  if (platform === "win32") {
    return "win32-x64";
  }
  if (platform === "darwin" && arch === "arm64") {
    return "darwin-arm64";
  }
  if (platform === "darwin") {
    return "darwin-x64";
  }
  if (platform === "linux") {
    return "linux-x64";
  }
  return `${platform}-${arch}`;
}

export function desktopResourcesManifestPath(resourcesRoot) {
  return path.join(resourcesRoot, DESKTOP_RESOURCES_MANIFEST_NAME);
}

export function sha256FileSync(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

export function resolvePackagedResourcesRoot(buildPath, platform, productName) {
  if (platform !== "darwin") {
    return path.resolve(buildPath, "resources");
  }

  const preferred = path.resolve(
    buildPath,
    `${productName || "Lime"}.app`,
    "Contents",
    "Resources",
  );
  if (existsSync(preferred)) {
    return preferred;
  }

  const appPath = readdirSync(buildPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.endsWith(".app"))
    .map((entry) => path.join(buildPath, entry.name))
    .sort()[0];
  if (!appPath) {
    throw new Error(`macOS app bundle is missing under ${buildPath}`);
  }
  return path.join(appPath, "Contents", "Resources");
}

export function buildDesktopResourcesManifest({
  resourcesRoot,
  platform = process.platform,
  arch = process.arch,
  version,
  applicationId = DESKTOP_APPLICATION_ID,
  files,
  native = {},
}) {
  const platformKey = desktopResourcePlatformKey(platform, arch);
  const normalizedVersion = String(version || "").trim();
  if (!normalizedVersion) {
    throw new Error("desktop resource manifest version is required");
  }
  const entries = Object.entries(files || {}).map(([id, relativePath]) => {
    const normalizedPath = normalizeRelativePath(relativePath, id);
    const absolutePath = path.resolve(resourcesRoot, normalizedPath);
    assertRegularFile(absolutePath, `desktop resource ${id}`);
    return {
      id,
      kind:
        id === "app-server" || id === "code-mode-host" ? "sidecar" : "helper",
      path: normalizedPath,
      sha256: sha256FileSync(absolutePath),
      required: true,
    };
  });

  return {
    schemaVersion: DESKTOP_RESOURCES_SCHEMA_VERSION,
    applicationId,
    version: normalizedVersion,
    platform,
    arch,
    platformKey,
    minimumOsVersion: platform === "darwin" ? "13.0" : null,
    resources: entries,
    native: {
      helper: native.helper || null,
      windowsHelper: native.windowsHelper || null,
      entitlements: native.entitlements || null,
    },
  };
}

export function preparePackagedDesktopResources({
  buildPath,
  platform,
  arch,
  version,
  applicationId = DESKTOP_APPLICATION_ID,
  productName = "Lime",
  execFileSyncImpl = execFileSync,
}) {
  const resourcesRoot = resolvePackagedResourcesRoot(
    buildPath,
    platform,
    productName,
  );
  return prepareDesktopResourcesFromRoot({
    resourcesRoot,
    platform,
    arch,
    version,
    applicationId,
    execFileSyncImpl,
    nativeSignedByForge: true,
  });
}

export function prepareDevelopmentDesktopResources({
  outputRoot = path.resolve("dist-electron"),
  platform = process.platform,
  arch = process.arch,
  version,
  applicationId = DESKTOP_APPLICATION_ID,
  execFileSyncImpl = execFileSync,
}) {
  return prepareDesktopResourcesFromRoot({
    resourcesRoot: path.resolve(outputRoot),
    platform,
    arch,
    version,
    applicationId,
    execFileSyncImpl,
    nativeSignedByForge: false,
  });
}

function prepareDesktopResourcesFromRoot({
  resourcesRoot,
  platform,
  arch,
  version,
  applicationId,
  execFileSyncImpl,
  nativeSignedByForge,
}) {
  const platformKey = desktopResourcePlatformKey(platform, arch);
  const appServerManifestPath = path.join(
    resourcesRoot,
    "app-server.release.json",
  );
  assertRegularFile(appServerManifestPath, "app-server release manifest");
  const appServerManifest = JSON.parse(
    readFileSync(appServerManifestPath, "utf8"),
  );
  const artifact = Array.isArray(appServerManifest.artifacts)
    ? appServerManifest.artifacts.find((item) => item?.platform === platformKey)
    : null;
  if (!artifact) {
    throw new Error(`app-server release manifest is missing ${platformKey}`);
  }
  const binaryName = platform === "win32" ? "app-server.exe" : "app-server";
  const codeModeHostName =
    platform === "win32" ? "code-mode-host.exe" : "code-mode-host";
  const files = {
    "app-server": path.join("app-server", platformKey, binaryName),
    "code-mode-host": path.join("app-server", platformKey, codeModeHostName),
  };
  if (platform === "win32") {
    files["windows-sandbox-setup"] = path.join(
      "app-server",
      platformKey,
      "windows-sandbox-setup.exe",
    );
    files["windows-sandbox-runner"] = path.join(
      "app-server",
      platformKey,
      "windows-sandbox-runner.exe",
    );
  }
  let native = {};
  if (platform === "darwin") {
    const helperBundlePath = path.join(
      resourcesRoot,
      MAC_NATIVE_HOST_BUNDLE_RELATIVE_PATH,
    );
    const helperPath = path.join(resourcesRoot, MAC_NATIVE_HOST_RELATIVE_PATH);
    mkdirSync(path.dirname(helperPath), { recursive: true });
    compileMacNativeHost({
      sourcePaths: [
        MAC_NATIVE_HOST_SOURCE,
        MAC_SCREEN_CAPTURE_SOURCE,
        MAC_SECURITY_SOURCE,
        MAC_WINDOW_ORCHESTRATION_SOURCE,
        MAC_ACCESSIBILITY_TREE_SOURCE,
        MAC_DISPLAY_WATCHER_SOURCE,
        MAC_MEDIA_PERMISSIONS_SOURCE,
        MAC_APPLE_EVENTS_SOURCE,
      ],
      outputPath: helperPath,
      arch,
      execFileSyncImpl,
    });
    writeFileSync(
      path.join(helperBundlePath, "Contents", "Info.plist"),
      macNativeHostInfoPlist(applicationId),
      "utf8",
    );
    files["macos-native-host"] = MAC_NATIVE_HOST_RELATIVE_PATH;
    native = {
      helper: {
        id: "macos-native-host",
        path: MAC_NATIVE_HOST_RELATIVE_PATH,
        bundlePath: MAC_NATIVE_HOST_BUNDLE_RELATIVE_PATH,
        bundleIdentifier: `${applicationId}.native-host`,
        frameworks: [
          "AppKit",
          "ApplicationServices",
          "CoreGraphics",
          "Foundation",
          "AVFoundation",
          "IOKit",
          "LocalAuthentication",
          "Security",
        ],
        signedByForge: nativeSignedByForge,
      },
      entitlements: {
        source: "lime-rs/entitlements.plist",
        applicationGroups: [],
        automationAppleEvents: true,
      },
    };
  } else if (platform === "win32") {
    const helperPath = path.join(resourcesRoot, WINDOWS_NATIVE_HOST_RELATIVE_PATH);
    mkdirSync(path.dirname(helperPath), { recursive: true });
    compileWindowsNativeHost({
      sourcePath: WINDOWS_NATIVE_HOST_SOURCE,
      outputPath: helperPath,
      execFileSyncImpl,
    });
    files["windows-native-host"] = WINDOWS_NATIVE_HOST_RELATIVE_PATH;
    native = {
      windowsHelper: {
        id: "windows-native-host",
        path: WINDOWS_NATIVE_HOST_RELATIVE_PATH,
        api: [
          "UIAutomation",
          "RawInput",
          "WindowEnumeration",
          "DisplayEnumeration",
          "DisplayWatcher",
        ],
        readOnly: true,
        signedByForge: nativeSignedByForge,
      },
    };
  }
  const manifest = buildDesktopResourcesManifest({
    resourcesRoot,
    platform,
    arch: platform === "win32" ? "x64" : arch,
    version,
    applicationId,
    files,
    native,
  });
  const manifestPath = desktopResourcesManifestPath(resourcesRoot);
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return { resourcesRoot, manifestPath, manifest, appServerArtifact: artifact };
}

function macNativeHostInfoPlist(applicationId) {
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0">',
    "<dict>",
    "  <key>CFBundleDisplayName</key>",
    "  <string>Lime Native Host</string>",
    "  <key>CFBundleExecutable</key>",
    "  <string>macos-native-host</string>",
    "  <key>CFBundleIdentifier</key>",
    `  <string>${applicationId}.native-host</string>`,
    "  <key>CFBundleName</key>",
    "  <string>Lime Native Host</string>",
    "  <key>CFBundlePackageType</key>",
    "  <string>APPL</string>",
    "  <key>CFBundleShortVersionString</key>",
    "  <string>1.0</string>",
    "  <key>CFBundleVersion</key>",
    "  <string>1</string>",
    "  <key>NSCameraUsageDescription</key>",
    "  <string>Lime 需要访问摄像头以使用视觉输入功能</string>",
    "  <key>NSMicrophoneUsageDescription</key>",
    "  <string>Lime 需要访问麦克风以使用语音输入功能</string>",
    "  <key>NSAppleEventsUsageDescription</key>",
    "  <string>Lime 需要获得授权以查询其他应用的自动化权限</string>",
    "</dict>",
    "</plist>",
    "",
  ].join("\n");
}

function compileMacNativeHost({
  sourcePaths,
  outputPath,
  arch,
  execFileSyncImpl,
}) {
  for (const sourcePath of sourcePaths) {
    assertRegularFile(sourcePath, "macOS native host source");
  }
  const targetArch = arch === "x64" ? "x86_64" : "arm64";
  execFileSyncImpl(
    "swiftc",
    [
      "-O",
      "-target",
      `${targetArch}-apple-macos13.0`,
      "-framework",
      "AppKit",
      "-framework",
      "ApplicationServices",
      "-framework",
      "CoreGraphics",
      "-framework",
      "Security",
      "-framework",
      "IOKit",
      "-framework",
      "LocalAuthentication",
      "-framework",
      "AVFoundation",
      ...sourcePaths,
      "-o",
      outputPath,
    ],
    { stdio: "pipe" },
  );
  chmodSync(outputPath, 0o755);
}

function compileWindowsNativeHost({
  sourcePath,
  outputPath,
  execFileSyncImpl,
}) {
  assertRegularFile(sourcePath, "Windows native host source");
  const compiler = String(process.env.CXX || "cl").trim() || "cl";
  const isMsvc =
    /(?:^|[\\/])(?:cl|clang-cl)(?:\.exe)?$/iu.test(compiler) ||
    compiler === "cl";
  const args = isMsvc
    ? [
        "/nologo",
        "/std:c++17",
        "/EHsc",
        "/O2",
        sourcePath,
        `/Fe:${outputPath}`,
        "ole32.lib",
        "oleaut32.lib",
        "user32.lib",
        "uiautomationcore.lib",
      ]
    : [
        "-std=c++17",
        "-O2",
        sourcePath,
        "-lole32",
        "-loleaut32",
        "-luser32",
        "-luiautomationcore",
        "-o",
        outputPath,
      ];
  execFileSyncImpl(compiler, args, { stdio: "pipe" });
  chmodSync(outputPath, 0o755);
}

function normalizeRelativePath(value, id) {
  const normalized = String(value || "").replace(/\\/g, "/");
  if (
    !normalized ||
    normalized.startsWith("/") ||
    normalized.split("/").some((part) => part === "..")
  ) {
    throw new Error(`desktop resource ${id} path must stay inside resources`);
  }
  return normalized;
}

function assertRegularFile(filePath, label) {
  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    throw new Error(`${label} is missing: ${filePath}`);
  }
}
