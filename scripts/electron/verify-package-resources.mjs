#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { builtinModules } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

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
    const assets = path.join(dir, "desktop-assets");
    return existsSync(manifest) && existsSync(assets);
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
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const key = platformKey(platform, arch);
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
    sha256: appServer,
  };
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
