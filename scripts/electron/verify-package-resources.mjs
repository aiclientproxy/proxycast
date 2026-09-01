#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { builtinModules } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  DESKTOP_APPLICATION_ID,
  DESKTOP_RESOURCES_MANIFEST_NAME,
  DESKTOP_RESOURCES_SCHEMA_VERSION,
} from "../lib/electron-desktop-resources.mjs";

const DEFAULT_PACKAGE_ROOT = "release-electron";
const MAC_PRODUCT_NAME = "Lime";
const MAC_APP_ID = "com.limecloud.lime";
const ELECTRON_RUNTIME_BUNDLES = [
  {
    label: "Electron main bundle",
    relativePath: "dist-electron/main/main.js",
  },
  {
    label: "Electron preload bundle",
    relativePath: "dist-electron/preload/preload.cjs",
  },
];
const ALLOWED_BARE_RUNTIME_IMPORTS = new Set([
  "electron",
  ...builtinModules.map((moduleName) => moduleName.replace(/^node:/, "")),
]);

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith("--")) {
      continue;
    }
    const key = item.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      args[key] = "true";
      continue;
    }
    args[key] = next;
    index += 1;
  }
  return args;
}

function walkDirectories(rootDir) {
  if (!existsSync(rootDir)) {
    return [];
  }
  const result = [];
  const stack = [rootDir];
  while (stack.length > 0) {
    const current = stack.pop();
    result.push(current);
    for (const entry of safeReadDir(current)) {
      if (entry.isDirectory()) {
        stack.push(path.join(current, entry.name));
      }
    }
  }
  return result.sort();
}

function safeReadDir(dir) {
  try {
    return statSync(dir).isDirectory()
      ? Array.from(readdirSync(dir, { withFileTypes: true }))
      : [];
  } catch {
    return [];
  }
}

function findResourceRoots(packageRoot) {
  const roots = walkDirectories(packageRoot).filter((dir) => {
    const manifest = path.join(dir, "app-server.release.json");
    const desktopManifest = path.join(dir, DESKTOP_RESOURCES_MANIFEST_NAME);
    const assets = path.join(dir, "desktop-assets");
    return (
      existsSync(manifest) && existsSync(desktopManifest) && existsSync(assets)
    );
  });
  return roots.sort();
}

function findMacAppBundles(packageRoot) {
  return walkDirectories(packageRoot)
    .filter((dir) => dir.endsWith(".app"))
    .map((appPath) => ({
      appPath,
      infoPlistPath: path.join(appPath, "Contents", "Info.plist"),
    }))
    .filter(({ infoPlistPath }) => existsSync(infoPlistPath));
}

function platformKey(platform = process.platform, arch = process.arch) {
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

function binaryName(platform = process.platform) {
  return platform === "win32" ? "app-server.exe" : "app-server";
}

function codeModeHostBinaryName(platform = process.platform) {
  return platform === "win32" ? "code-mode-host.exe" : "code-mode-host";
}

function windowsSandboxSetupBinaryName(platform = process.platform) {
  return platform === "win32" ? "windows-sandbox-setup.exe" : null;
}

function windowsSandboxRunnerBinaryName(platform = process.platform) {
  return platform === "win32" ? "windows-sandbox-runner.exe" : null;
}

function sha256(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

function isMacCodeSigned(filePath, execFileSyncImpl = execFileSync) {
  try {
    execFileSyncImpl("codesign", ["--verify", "--strict", filePath], {
      stdio: "ignore",
    });
    return true;
  } catch {
    return false;
  }
}

function verifySidecarIntegrity(
  filePath,
  expectedSha256,
  { platform, label, execFileSyncImpl = execFileSync },
) {
  if (!/^[a-f0-9]{64}$/u.test(expectedSha256 ?? "")) {
    throw new Error(`${label} manifest sha256 is invalid`);
  }
  const packagedSha256 = sha256(filePath);
  const sha256Matches = expectedSha256 === packagedSha256;
  const signedMacSidecar =
    platform === "darwin" &&
    !sha256Matches &&
    isMacCodeSigned(filePath, execFileSyncImpl);
  if (!sha256Matches && !signedMacSidecar) {
    throw new Error(`${label} sha256 mismatch: ${filePath}`);
  }
  return {
    manifest: expectedSha256,
    packaged: packagedSha256,
    matches: sha256Matches,
    acceptedBecause: signedMacSidecar ? "macos-signed-sidecar" : "sha256",
  };
}

function assertFile(filePath, label) {
  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    throw new Error(`${label} is missing: ${filePath}`);
  }
}

export function verifyResourceRoot(
  root,
  { platform, arch, execFileSyncImpl = execFileSync },
) {
  const manifestPath = path.join(root, "app-server.release.json");
  const desktopManifestPath = path.join(root, DESKTOP_RESOURCES_MANIFEST_NAME);
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const desktopManifest = JSON.parse(readFileSync(desktopManifestPath, "utf8"));
  const key = platformKey(platform, arch);
  const deferredDesktopResourceIntegrity = verifyDesktopResourceManifest(
    desktopManifest,
    {
      root,
      platform,
      arch,
      key,
      execFileSyncImpl,
    },
  );
  const artifact = Array.isArray(manifest.artifacts)
    ? manifest.artifacts.find((item) => item?.platform === key)
    : null;
  if (!artifact) {
    throw new Error(`app-server manifest is missing platform ${key}`);
  }

  const sidecarPath = path.join(root, "app-server", key, binaryName(platform));
  assertFile(sidecarPath, "app-server sidecar");
  const codeModeHostPath = path.join(
    root,
    "app-server",
    key,
    codeModeHostBinaryName(platform),
  );
  assertFile(codeModeHostPath, "code-mode host sidecar");
  const codeModeHost = verifySidecarIntegrity(
    codeModeHostPath,
    artifact.codeModeHostSha256,
    {
      platform,
      label: "code-mode host sidecar",
      execFileSyncImpl,
    },
  );
  const appServer = verifySidecarIntegrity(sidecarPath, artifact.sha256, {
    platform,
    label: "app-server sidecar",
    execFileSyncImpl,
  });
  verifyDeferredDesktopResourceIntegrity(deferredDesktopResourceIntegrity, {
    appServer,
    codeModeHost,
  });
  let windowsSandboxSetupPath = null;
  let windowsSandboxSetupIntegrity = null;
  let windowsSandboxRunnerPath = null;
  let windowsSandboxRunnerIntegrity = null;
  if (platform === "win32") {
    windowsSandboxSetupPath = path.join(
      root,
      "app-server",
      key,
      windowsSandboxSetupBinaryName(platform),
    );
    assertFile(windowsSandboxSetupPath, "Windows sandbox setup helper");
    windowsSandboxSetupIntegrity = verifySidecarIntegrity(
      windowsSandboxSetupPath,
      artifact.windowsSandboxSetupSha256,
      {
        platform,
        label: "Windows sandbox setup helper",
        execFileSyncImpl,
      },
    );
    windowsSandboxRunnerPath = path.join(
      root,
      "app-server",
      key,
      windowsSandboxRunnerBinaryName(platform),
    );
    assertFile(windowsSandboxRunnerPath, "Windows sandbox runner");
    windowsSandboxRunnerIntegrity = verifySidecarIntegrity(
      windowsSandboxRunnerPath,
      artifact.windowsSandboxRunnerSha256,
      {
        platform,
        label: "Windows sandbox runner",
        execFileSyncImpl,
      },
    );
  }

  for (const name of [
    "icon.png",
    "trayTemplate.png",
    "trayTemplate@2x.png",
    "tray-running.png",
    "tray-stopped.png",
    "tray-warning.png",
    "tray-error.png",
  ]) {
    assertFile(
      path.join(root, "desktop-assets", name),
      `desktop asset ${name}`,
    );
  }

  return {
    platform: key,
    resourceRoot: root,
    sidecarPath,
    codeModeHostPath,
    codeModeHostSha256: codeModeHost.packaged,
    codeModeHostIntegrity: codeModeHost,
    windowsSandboxSetupPath,
    windowsSandboxSetupIntegrity,
    windowsSandboxRunnerPath,
    windowsSandboxRunnerIntegrity,
    sha256: appServer,
    desktopManifest,
  };
}

function verifyDesktopResourceManifest(
  manifest,
  { root, platform, arch, key, execFileSyncImpl },
) {
  if (manifest?.schemaVersion !== DESKTOP_RESOURCES_SCHEMA_VERSION) {
    throw new Error(
      `desktop resource manifest schema version is unsupported: ${manifest?.schemaVersion ?? "missing"}`,
    );
  }
  if (manifest.applicationId !== DESKTOP_APPLICATION_ID) {
    throw new Error(
      `desktop resource manifest applicationId mismatch: ${manifest.applicationId ?? "missing"}`,
    );
  }
  if (manifest.platformKey !== key || manifest.platform !== platform) {
    throw new Error(
      `desktop resource manifest platform mismatch: expected ${key}, got ${manifest.platformKey ?? "missing"}`,
    );
  }
  const expectedArch = platform === "win32" ? "x64" : arch;
  if (manifest.arch !== expectedArch) {
    throw new Error(
      `desktop resource manifest architecture mismatch: expected ${expectedArch}, got ${manifest.arch ?? "missing"}`,
    );
  }
  const expectedMinimumOsVersion = platform === "darwin" ? "13.0" : null;
  if (manifest.minimumOsVersion !== expectedMinimumOsVersion) {
    throw new Error(
      `desktop resource manifest minimum OS version mismatch: expected ${expectedMinimumOsVersion ?? "null"}, got ${manifest.minimumOsVersion ?? "null"}`,
    );
  }
  if (!Array.isArray(manifest.resources) || manifest.resources.length === 0) {
    throw new Error("desktop resource manifest has no resources");
  }

  const ids = new Set();
  const signedMacNativeHost =
    platform === "darwin" && manifest.native?.helper?.signedByForge === true;
  const macNativeHostBundlePath =
    platform === "darwin"
      ? resolveManifestPath(root, manifest.native?.helper?.bundlePath)
      : null;
  const signedMacNativeHostPath = signedMacNativeHost
    ? macNativeHostBundlePath
    : null;
  const deferredIntegrity = [];
  for (const resource of manifest.resources) {
    if (!resource || typeof resource !== "object") {
      throw new Error("desktop resource manifest contains an invalid resource");
    }
    const id = String(resource.id || "");
    const relativePath = String(resource.path || "").replace(/\\/g, "/");
    if (!id || ids.has(id)) {
      throw new Error(
        `desktop resource manifest has duplicate or missing id: ${id}`,
      );
    }
    ids.add(id);
    const expectedPath = expectedDesktopResourcePath(id, key, platform);
    if (expectedPath && relativePath !== expectedPath) {
      throw new Error(
        `desktop resource ${id} path mismatch: expected ${expectedPath}, got ${relativePath}`,
      );
    }
    if (
      !relativePath ||
      relativePath.startsWith("/") ||
      relativePath.split("/").some((part) => part === "..")
    ) {
      throw new Error(`desktop resource path escapes package resources: ${id}`);
    }
    const resourcePath = path.resolve(root, relativePath);
    if (!resourcePath.startsWith(`${path.resolve(root)}${path.sep}`)) {
      throw new Error(`desktop resource path escapes package resources: ${id}`);
    }
    assertFile(resourcePath, desktopResourceLabel(id));
    if (!/^[a-f0-9]{64}$/u.test(resource.sha256 ?? "")) {
      throw new Error(`desktop resource ${id} sha256 is invalid`);
    }
    const packagedSha256 = sha256(resourcePath);
    if (packagedSha256 !== resource.sha256) {
      if (
        !signedMacNativeHost ||
        id !== "macos-native-host" ||
        !isMacCodeSigned(
          signedMacNativeHostPath ?? resourcePath,
          execFileSyncImpl,
        )
      ) {
        if (
          platform === "darwin" &&
          (id === "app-server" || id === "code-mode-host")
        ) {
          deferredIntegrity.push({
            id,
            resourcePath,
            expected: resource.sha256,
          });
        } else {
          throw new Error(
            `desktop resource ${id} sha256 mismatch: ${resourcePath}`,
          );
        }
      }
    }
  }
  if (platform === "win32") {
    const helperMetadata = manifest.native?.windowsHelper;
    const helperResource = manifest.resources.find(
      (resource) => resource?.id === "windows-native-host",
    );
    if (helperMetadata || helperResource) {
      if (helperMetadata?.id !== "windows-native-host") {
        throw new Error(
          "desktop resource manifest Windows native helper metadata is invalid",
        );
      }
      if (
        typeof helperMetadata.path !== "string" ||
        String(helperMetadata.path).replace(/\\/g, "/") !==
          String(helperResource?.path || "").replace(/\\/g, "/")
      ) {
        throw new Error(
          "desktop resource manifest Windows native helper path does not match resource",
        );
      }
      if (
        String(helperMetadata.path).replace(/\\/g, "/") !==
        "native/windows/windows-native-host.exe"
      ) {
        throw new Error(
          "desktop resource manifest Windows native helper path mismatch",
        );
      }
      if (helperMetadata.readOnly !== true) {
        throw new Error(
          "desktop resource manifest Windows native helper must be read-only",
        );
      }
      if (!Array.isArray(helperMetadata.api) || helperMetadata.api.length === 0) {
        throw new Error(
          "desktop resource manifest Windows native helper API metadata is missing",
        );
      }
    }
  }

  const requiredIds = ["app-server", "code-mode-host"];
  if (platform === "win32") {
    requiredIds.push(
      "windows-sandbox-setup",
      "windows-sandbox-runner",
      "windows-native-host",
    );
  }
  if (platform === "darwin") {
    requiredIds.push("macos-native-host");
    const helperMetadata = manifest.native?.helper;
    const helperResource = manifest.resources.find(
      (resource) => resource?.id === "macos-native-host",
    );
    if (helperMetadata?.id !== "macos-native-host") {
      throw new Error(
        "desktop resource manifest is missing macOS native helper metadata",
      );
    }
    if (
      typeof helperMetadata.path !== "string" ||
      String(helperMetadata.path).replace(/\\/g, "/") !==
        String(helperResource?.path || "").replace(/\\/g, "/")
    ) {
      throw new Error(
        "desktop resource manifest macOS helper path does not match resource",
      );
    }
    if (
      helperMetadata.bundleIdentifier !==
      `${DESKTOP_APPLICATION_ID}.native-host`
    ) {
      throw new Error(
        "desktop resource manifest macOS helper bundle identifier mismatch",
      );
    }
    if (helperMetadata.bundlePath && !macNativeHostBundlePath) {
      throw new Error(
        "desktop resource manifest macOS helper bundle path is invalid",
      );
    }
    if (helperMetadata.bundlePath && macNativeHostBundlePath) {
      if (!statSync(macNativeHostBundlePath).isDirectory()) {
        throw new Error(
          "desktop resource manifest macOS helper bundle path must be a directory",
        );
      }
      const bundleInfoPlist = path.join(
        macNativeHostBundlePath,
        "Contents",
        "Info.plist",
      );
      if (!existsSync(bundleInfoPlist)) {
        throw new Error(
          "desktop resource manifest macOS helper bundle is missing Info.plist",
        );
      }
      const bundleInfoPlistContent = readFileSync(bundleInfoPlist, "utf8");
      if (
        !plistContainsString(
          bundleInfoPlistContent,
          "CFBundleIdentifier",
          helperMetadata.bundleIdentifier,
        )
      ) {
        throw new Error(
          "desktop resource manifest macOS helper Info.plist bundle identifier mismatch",
        );
      }
    }
    const applicationGroups = manifest.native?.entitlements?.applicationGroups;
    if (!Array.isArray(applicationGroups)) {
      throw new Error(
        "desktop resource manifest macOS entitlements applicationGroups must be an array",
      );
    }
    for (const group of applicationGroups) {
      if (
        typeof group !== "string" ||
        (group !== DESKTOP_APPLICATION_ID &&
          !group.startsWith(`${DESKTOP_APPLICATION_ID}.`))
      ) {
        throw new Error(
          "desktop resource manifest macOS Application Group must use the Lime namespace",
        );
      }
    }
    if (manifest.native?.entitlements?.automationAppleEvents !== true) {
      throw new Error(
        "desktop resource manifest macOS entitlements automationAppleEvents must be true",
      );
    }
    if (helperMetadata.signedByForge === true) {
      const signaturePath =
        resolveManifestPath(root, manifest.native.helper.bundlePath) ??
        (typeof helperResource?.path === "string"
          ? path.resolve(root, helperResource.path)
          : null);
      if (!signaturePath) {
        throw new Error(
          "desktop resource manifest is missing macOS native helper resource path",
        );
      }
      try {
        execFileSyncImpl("codesign", ["--verify", "--strict", signaturePath], {
          stdio: "ignore",
        });
      } catch (error) {
        throw new Error(
          `macOS native host signature is invalid: ${signaturePath}`,
          {
            cause: error,
          },
        );
      }
    }
  }
  for (const id of requiredIds) {
    if (!ids.has(id)) {
      throw new Error(
        `desktop resource manifest is missing required resource: ${id}`,
      );
    }
  }

  return deferredIntegrity;
}

function verifyDeferredDesktopResourceIntegrity(
  deferredIntegrity,
  { appServer, codeModeHost },
) {
  for (const entry of deferredIntegrity) {
    const sidecar = entry.id === "app-server" ? appServer : codeModeHost;
    if (
      !sidecar ||
      entry.expected !== sidecar.manifest ||
      (!sidecar.matches && sidecar.acceptedBecause !== "macos-signed-sidecar")
    ) {
      throw new Error(
        `desktop resource ${entry.id} sha256 mismatch: ${entry.resourcePath}`,
      );
    }
  }
}

function resolveManifestPath(root, value) {
  const relativePath = String(value || "").replace(/\\/g, "/");
  if (
    !relativePath ||
    relativePath.startsWith("/") ||
    relativePath.split("/").some((part) => part === "..")
  ) {
    return null;
  }
  const resolved = path.resolve(root, relativePath);
  return resolved.startsWith(`${path.resolve(root)}${path.sep}`) &&
    existsSync(resolved)
    ? resolved
    : null;
}

function desktopResourceLabel(id) {
  switch (id) {
    case "windows-sandbox-setup":
      return "Windows sandbox setup helper";
    case "windows-sandbox-runner":
      return "Windows sandbox runner";
    case "app-server":
      return "app-server sidecar";
    case "code-mode-host":
      return "code-mode host sidecar";
    case "macos-native-host":
      return "macOS native host helper";
    case "windows-native-host":
      return "Windows native host helper";
    default:
      return `desktop resource ${id}`;
  }
}

function expectedDesktopResourcePath(id, key, platform) {
  const suffix = platform === "win32" ? ".exe" : "";
  switch (id) {
    case "app-server":
      return `app-server/${key}/app-server${suffix}`;
    case "code-mode-host":
      return `app-server/${key}/code-mode-host${suffix}`;
    case "windows-sandbox-setup":
      return platform === "win32"
        ? `app-server/${key}/windows-sandbox-setup.exe`
        : null;
    case "windows-sandbox-runner":
      return platform === "win32"
        ? `app-server/${key}/windows-sandbox-runner.exe`
        : null;
    case "windows-native-host":
      return platform === "win32" ? "native/windows/windows-native-host.exe" : null;
    default:
      return null;
  }
}

function verifyMainBundle(repoRoot) {
  verifyElectronRuntimeBundles(repoRoot);
}

export function verifyElectronRuntimeBundles(repoRoot) {
  for (const bundle of ELECTRON_RUNTIME_BUNDLES) {
    const bundlePath = path.resolve(repoRoot, bundle.relativePath);
    assertFile(bundlePath, bundle.label);
    const content = readFileSync(bundlePath, "utf8");
    const bareImports = collectBareRuntimeImports(content).filter(
      (packageName) => !ALLOWED_BARE_RUNTIME_IMPORTS.has(packageName),
    );
    if (bareImports.length > 0) {
      throw new Error(
        `${bundle.label} still imports runtime package(s) outside app.asar bundle: ${bareImports.join(", ")}`,
      );
    }
  }
}

export function collectBareRuntimeImports(content) {
  const imports = new Set();
  for (const pattern of [
    /^\s*import\s+(?!type\b)[^;]*?\s+from\s+["']([^"']+)["']/gm,
    /^\s*import\s+["']([^"']+)["']/gm,
    /\brequire\(\s*["']([^"']+)["']\s*\)/g,
  ]) {
    for (const match of content.matchAll(pattern)) {
      const packageName = barePackageName(match[1]);
      if (packageName) {
        imports.add(packageName);
      }
    }
  }
  return [...imports].sort();
}

function barePackageName(specifier) {
  if (
    specifier.startsWith(".") ||
    specifier.startsWith("/") ||
    specifier.startsWith("#") ||
    specifier.startsWith("node:")
  ) {
    return null;
  }
  if (specifier.startsWith("@")) {
    return specifier.split("/").slice(0, 2).join("/");
  }
  return specifier.split("/")[0] ?? null;
}

export function verifyMacAppIdentity(packageRoot, { platform }) {
  if (platform !== "darwin") {
    return [];
  }

  const appBundles = findMacAppBundles(packageRoot);
  if (appBundles.length === 0) {
    return [];
  }

  return appBundles.map((bundle) => verifyMacAppBundleIdentity(bundle));
}

export function verifyMacAppSignatures(
  packageRoot,
  { platform, execFileSyncImpl = execFileSync },
) {
  if (platform !== "darwin") {
    return [];
  }

  const mainAppBundles = findMacAppBundles(packageRoot).filter(
    ({ appPath }) => path.basename(appPath) === `${MAC_PRODUCT_NAME}.app`,
  );
  if (mainAppBundles.length === 0) {
    throw new Error(
      `no macOS ${MAC_PRODUCT_NAME}.app bundle found under ${packageRoot}`,
    );
  }

  return mainAppBundles.map(({ appPath }) => {
    try {
      execFileSyncImpl(
        "codesign",
        ["--verify", "--deep", "--strict", appPath],
        { encoding: "utf8", stdio: "pipe" },
      );
    } catch (error) {
      const stderr = error?.stderr?.toString?.("utf8").trim();
      const detail = stderr || error?.message || String(error);
      throw new Error(
        `macOS app bundle signature is invalid: ${appPath}\n${detail}`,
        { cause: error },
      );
    }

    return {
      appPath,
      valid: true,
      verification: "codesign --verify --deep --strict",
    };
  });
}

function verifyMacAppBundleIdentity({ appPath, infoPlistPath }) {
  const content = readFileSync(infoPlistPath, "utf8");
  if (/<key>\d+<\/key>\s*<string>/.test(content)) {
    throw new Error(
      `macOS Info.plist contains numeric extendInfo keys; Forge packager extendInfo must be an object: ${infoPlistPath}`,
    );
  }

  const isMainApp = path.basename(appPath) === `${MAC_PRODUCT_NAME}.app`;
  if (isMainApp) {
    verifyMainMacAppInfoPlist(content, infoPlistPath);
  } else {
    verifyHelperMacAppInfoPlist(content, infoPlistPath);
  }

  return {
    appPath,
    infoPlistPath,
    kind: isMainApp ? "main" : "helper",
  };
}

function verifyMainMacAppInfoPlist(content, infoPlistPath) {
  const requiredPairs = new Map([
    ["CFBundleDisplayName", MAC_PRODUCT_NAME],
    ["CFBundleName", MAC_PRODUCT_NAME],
    ["CFBundleExecutable", MAC_PRODUCT_NAME],
    ["CFBundleIdentifier", MAC_APP_ID],
    ["CFBundleIconFile", "icon.icns"],
  ]);
  for (const [key, value] of requiredPairs) {
    if (!plistContainsString(content, key, value)) {
      throw new Error(
        `macOS app identity mismatch for ${key}: ${infoPlistPath}`,
      );
    }
  }

  rejectElectronBrandValue(content, infoPlistPath);
}

function verifyHelperMacAppInfoPlist(content, infoPlistPath) {
  for (const key of [
    "CFBundleDisplayName",
    "CFBundleName",
    "CFBundleExecutable",
  ]) {
    if (plistStringValueStartsWith(content, key, "Electron")) {
      throw new Error(
        `macOS helper app identity still uses Electron for ${key}: ${infoPlistPath}`,
      );
    }
  }
}

function rejectElectronBrandValue(content, infoPlistPath) {
  if (plistContainsString(content, "CFBundleDisplayName", "Electron")) {
    throw new Error(
      `macOS app display name still uses Electron: ${infoPlistPath}`,
    );
  }
  if (plistContainsString(content, "CFBundleName", "Electron")) {
    throw new Error(
      `macOS app bundle name still uses Electron: ${infoPlistPath}`,
    );
  }
  if (plistContainsString(content, "CFBundleExecutable", "Electron")) {
    throw new Error(
      `macOS app executable still uses Electron: ${infoPlistPath}`,
    );
  }
}

function plistContainsString(content, key, value) {
  const escapedKey = escapeRegExp(key);
  const escapedValue = escapeRegExp(value);
  return new RegExp(
    `<key>${escapedKey}</key>\\s*<string>${escapedValue}</string>`,
  ).test(content);
}

function plistStringValueStartsWith(content, key, valuePrefix) {
  const escapedKey = escapeRegExp(key);
  const escapedValuePrefix = escapeRegExp(valuePrefix);
  return new RegExp(
    `<key>${escapedKey}</key>\\s*<string>${escapedValuePrefix}`,
  ).test(content);
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const repoRoot = process.cwd();
  const packageRoot = path.resolve(
    args["package-root"] || DEFAULT_PACKAGE_ROOT,
  );
  const platform = args.platform || process.platform;
  const arch = args.arch || process.arch;

  verifyMainBundle(repoRoot);
  const macAppInfoPlists = verifyMacAppIdentity(packageRoot, { platform });
  const macAppSignatures = verifyMacAppSignatures(packageRoot, { platform });
  const resourceRoots = findResourceRoots(packageRoot);
  if (resourceRoots.length === 0) {
    throw new Error(
      `no Electron packaged resource root found under ${packageRoot}`,
    );
  }

  const verified = resourceRoots.map((root) =>
    verifyResourceRoot(root, { platform, arch }),
  );
  console.log(
    JSON.stringify(
      {
        packageRoot,
        macAppInfoPlists,
        macAppSignatures,
        verified,
      },
      null,
      2,
    ),
  );
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main();
}
