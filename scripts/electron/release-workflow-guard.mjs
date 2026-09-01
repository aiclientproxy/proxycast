#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import YAML from "yaml";

const DEFAULT_WORKFLOW_PATH = ".github/workflows/release.yml";
const DEFAULT_FORGE_CONFIG_PATH = "forge.config.mjs";
const DEFAULT_STAGE_RELEASE_ASSETS_PATH =
  "scripts/electron/stage-release-assets.mjs";
const DEFAULT_UPDATE_FEED_UPLOAD_PLAN_PATH =
  "scripts/electron/update-feed-r2-upload-plan.mjs";
const DEFAULT_GITHUB_RELEASE_ASSETS_PATH =
  "scripts/electron/prepare-github-release-assets.mjs";
const DEFAULT_REPOSITORY_ROOT = ".";

const RETIRED_PACKAGING_FILE_PATTERNS = [
  /(?:^|\/)tauri(?:\.[^/]*)?\.conf\.json$/i,
  /(?:^|\/)latest(?:-mac)?\.ya?ml$/i,
  /\.blockmap$/i,
  /\.app\.tar\.gz$/i,
  /\.sig$/i,
];

const IGNORED_RETIRED_FILE_DIRS = new Set([
  ".codex",
  ".git",
  ".tmp",
  "coverage",
  "dist",
  "dist-electron",
  "node_modules",
  "release-assets",
  "release-electron",
  "release-github-assets",
  "target",
]);

function readWorkflow(workflowPath = DEFAULT_WORKFLOW_PATH) {
  return YAML.parse(fs.readFileSync(workflowPath, "utf8"));
}

function stepByName(steps, name) {
  return steps.find((step) => step?.name === name);
}

function assertEnvValueIncludes(env, key, needle, label) {
  if (!env || !(key in env)) {
    throw new Error(`${label} env missing ${key}`);
  }
  assertIncludes(env[key], needle, `${label} env ${key}`);
}

function assertIncludes(haystack, needle, label) {
  if (!String(haystack || "").includes(needle)) {
    throw new Error(`${label} must include ${needle}`);
  }
}

function assertNoLegacyUpdaterAssets(runScript, label) {
  for (const forbidden of [
    "*.app.tar.gz",
    "*.sig",
    "latest.json",
    "latest.yml",
    "latest-mac.yml",
    "*.blockmap",
  ]) {
    assertIncludes(runScript, forbidden, label);
  }
}

function assertMatrix(buildJob) {
  const matrix = buildJob?.strategy?.matrix?.include;
  if (!Array.isArray(matrix)) {
    throw new Error("release build job must define strategy.matrix.include");
  }

  const expected = new Map([
    [
      "macOS-arm64",
      {
        arch: "arm64",
        feed: "darwin-arm64",
        forge_targets: "dmg,zip",
        host_platform: "darwin",
        platform: "macos-15",
        target: "aarch64-apple-darwin",
      },
    ],
    [
      "macOS-x64",
      {
        arch: "x64",
        feed: "darwin-x64",
        forge_targets: "dmg,zip",
        host_platform: "darwin",
        platform: "macos-15-intel",
        target: "x86_64-apple-darwin",
      },
    ],
    [
      "Windows-x64",
      {
        arch: "x64",
        feed: "win32-x64",
        forge_targets: "squirrel",
        host_platform: "win32",
        platform: "windows-2022",
        target: "x86_64-pc-windows-msvc",
      },
    ],
  ]);

  for (const [name, fields] of expected) {
    const row = matrix.find((item) => item?.name === name);
    if (!row) {
      throw new Error(`release build matrix missing ${name}`);
    }
    for (const [key, value] of Object.entries(fields)) {
      if (row[key] !== value) {
        throw new Error(
          `release build matrix ${name}.${key} expected ${value}, got ${row[key]}`,
        );
      }
    }
  }
}

function assertBuildSteps(buildJob) {
  const steps = buildJob?.steps;
  if (!Array.isArray(steps)) {
    throw new Error("release build job must define steps");
  }

  const buildJobEnv = buildJob?.env || {};
  assertEnvValueIncludes(
    buildJobEnv,
    "LIME_ELECTRON_UPDATES_URL",
    "LIME_UPDATES_BASE_URL",
    "release build job",
  );
  assertEnvValueIncludes(
    buildJobEnv,
    "LIME_ELECTRON_UPDATES_URL",
    "matrix.feed",
    "release build job",
  );

  const pnpmSetupStep = stepByName(steps, "Setup pnpm");
  if (pnpmSetupStep?.uses !== "pnpm/action-setup@v4") {
    throw new Error("release build must set up pnpm with pnpm/action-setup@v4");
  }
  if (pnpmSetupStep?.with && Object.hasOwn(pnpmSetupStep.with, "version")) {
    throw new Error(
      "release build must read pnpm version from packageManager, not workflow version",
    );
  }

  const nodeSetupStep = stepByName(steps, "Setup Node.js");
  if (nodeSetupStep?.with?.cache !== "pnpm") {
    throw new Error("release build must cache pnpm dependencies");
  }
  if (nodeSetupStep?.with?.["cache-dependency-path"] !== "pnpm-lock.yaml") {
    throw new Error(
      "release build must use pnpm-lock.yaml as cache dependency path",
    );
  }

  const installStep = stepByName(steps, "Install dependencies");
  if (
    installStep?.run !==
    "pnpm install --frozen-lockfile --config.node-linker=hoisted"
  ) {
    throw new Error(
      "release build must install dependencies with pnpm install --frozen-lockfile --config.node-linker=hoisted",
    );
  }

  const macSecretStep = stepByName(
    steps,
    "Validate Electron macOS signing secrets",
  );
  const macSecretEnv = macSecretStep?.env || {};
  for (const secret of [
    "APPLE_CERTIFICATE",
    "APPLE_CERTIFICATE_PASSWORD",
    "APPLE_ID",
    "APPLE_PASSWORD",
    "APPLE_SIGNING_IDENTITY",
    "APPLE_TEAM_ID",
    "KEYCHAIN_PASSWORD",
  ]) {
    assertEnvValueIncludes(
      macSecretEnv,
      secret,
      `secrets.${secret}`,
      "macOS signing secret preflight",
    );
  }
  const macSecretRun = macSecretStep?.run || "";
  for (const secret of [
    "APPLE_CERTIFICATE",
    "APPLE_CERTIFICATE_PASSWORD",
    "APPLE_ID",
    "APPLE_PASSWORD",
    "APPLE_SIGNING_IDENTITY",
    "APPLE_TEAM_ID",
    "KEYCHAIN_PASSWORD",
  ]) {
    assertIncludes(macSecretRun, secret, "macOS signing secret preflight");
  }

  const macImportStep = stepByName(
    steps,
    "Import Electron macOS signing certificate",
  );
  const macImportEnv = macImportStep?.env || {};
  for (const secret of [
    "APPLE_CERTIFICATE",
    "APPLE_CERTIFICATE_PASSWORD",
    "KEYCHAIN_PASSWORD",
  ]) {
    assertEnvValueIncludes(
      macImportEnv,
      secret,
      `secrets.${secret}`,
      "macOS certificate import",
    );
  }
  const macImportRun = macImportStep?.run || "";
  for (const required of [
    "base64 --decode",
    "security create-keychain",
    "existing_keychains=()",
    "while IFS= read -r keychain",
    'security list-keychains -d user -s "$KEYCHAIN_PATH" "${existing_keychains[@]}"',
    "security import",
    "security set-key-partition-list",
    "security find-identity -v -p codesigning",
    "LIME_MACOS_KEYCHAIN",
    '>> "$GITHUB_ENV"',
  ]) {
    assertIncludes(macImportRun, required, "macOS certificate import");
  }

  const winSecretStep = stepByName(
    steps,
    "Validate Electron Windows signing secrets",
  );
  const winSecretEnv = winSecretStep?.env || {};
  for (const secret of [
    "WINDOWS_SIGNING_CERTIFICATE",
    "WINDOWS_SIGNING_CERTIFICATE_PASSWORD",
  ]) {
    assertEnvValueIncludes(
      winSecretEnv,
      secret,
      `secrets.${secret}`,
      "Windows signing secret preflight",
    );
  }
  const winSecretRun = winSecretStep?.run || "";
  for (const required of [
    "WINDOWS_SIGNING_CERTIFICATE",
    "WINDOWS_SIGNING_CERTIFICATE_PASSWORD",
    "Forge Squirrel will produce unsigned installer assets",
    "Incomplete Electron Windows signing secrets",
    "Forge Squirrel signing will be enabled",
    "has_certificate",
    "has_password",
  ]) {
    assertIncludes(winSecretRun, required, "Windows signing secret preflight");
  }

  for (const forbidden of [
    "Missing Electron Windows signing secrets",
    '[ -n "${WINDOWS_SIGNING_CERTIFICATE:-}" ] || missing+=("WINDOWS_SIGNING_CERTIFICATE")',
    '[ -n "${WINDOWS_SIGNING_CERTIFICATE_PASSWORD:-}" ] || missing+=("WINDOWS_SIGNING_CERTIFICATE_PASSWORD")',
  ]) {
    if (winSecretRun.includes(forbidden)) {
      throw new Error(
        `Windows signing secret preflight must not require optional Squirrel signing input: ${forbidden}`,
      );
    }
  }

  const winImportStep = stepByName(
    steps,
    "Prepare Electron Windows signing certificate",
  );
  const winImportEnv = winImportStep?.env || {};
  for (const secret of [
    "WINDOWS_SIGNING_CERTIFICATE",
    "WINDOWS_SIGNING_CERTIFICATE_PASSWORD",
  ]) {
    assertEnvValueIncludes(
      winImportEnv,
      secret,
      `secrets.${secret}`,
      "Windows certificate preparation",
    );
  }
  const winImportRun = winImportStep?.run || "";
  for (const required of [
    "WINDOWS_SIGNING_CERTIFICATE_PATH",
    'Buffer.from(certificate, "base64")',
    "Skipping Electron Windows signing certificate preparation",
    "LIME_WINDOWS_SIGNING_CERTIFICATE_FILE",
    "LIME_WINDOWS_SIGNING_CERTIFICATE_PASSWORD",
    '>> "$GITHUB_ENV"',
  ]) {
    assertIncludes(winImportRun, required, "Windows certificate preparation");
  }

  const buildStep = stepByName(steps, "Build Electron app");
  const buildRun = buildStep?.run || "";
  const rustyV8Step = stepByName(steps, "Refresh sandboxed rusty_v8 artifacts");
  const rustyV8Run = rustyV8Step?.run || "";
  for (const required of [
    "scripts/lib/rusty-v8-artifacts.mjs --github-env",
    "cargo clean --manifest-path lime-rs/Cargo.toml -p v8",
  ]) {
    assertIncludes(rustyV8Run, required, "rusty_v8 artifact refresh");
  }
  const sherpaStep = stepByName(steps, "Prepare sherpa-onnx runtime");
  const sherpaRun = sherpaStep?.run || "";
  for (const required of [
    "scripts/prepare-sherpa-onnx-runtime.mjs",
    '--target "${{ matrix.target }}"',
  ]) {
    assertIncludes(sherpaRun, required, "sherpa-onnx runtime preparation");
  }
  for (const required of [
    "npm run electron:build",
    "npx electron-forge package",
    "FORGE_PACKAGE_LOG",
    "run_forge_package",
    "is_transient_macos_package_error",
    "NSURLErrorDomain",
    "The Internet connection appears to be offline",
    "No network route",
    "HTTPError\\\\(statusCode: nil",
    "_NSURLErrorNWPathKey=unsatisfied",
    "Response code 502 \\\\(Bad Gateway\\\\)",
    "Response code 504 \\\\(Gateway Time-out\\\\)",
    "A timestamp was expected but was not found",
    "The timestamp service is not available",
    "retrying package attempt",
    "npx electron-forge make",
    "FORGE_MAKE_LOG",
    "run_forge_make",
    "--skip-package",
    '--platform "${{ matrix.host_platform }}"',
    '--arch "${{ matrix.arch }}"',
    '--targets "${{ matrix.forge_targets }}"',
    "PIPESTATUS[0]",
    "hdiutil detach /Volumes/Lime",
    "No such file or directory",
    'hdiutil detach "/Volumes/Lime" -force || true',
    "find release-electron -type f | sort",
    "codesign --display --verbose=4",
    "codesign --verify --deep --strict --verbose=4",
    "release-electron/make",
    "*.nupkg",
    "RELEASES.json",
    "Electron Forge make produced no release assets",
  ]) {
    assertIncludes(buildRun, required, "Electron Forge make step");
  }
  const buildEnv = buildStep?.env || {};
  assertEnvValueIncludes(
    buildEnv,
    "LIME_ELECTRON_SIGN",
    "matrix.host_platform == 'win32'",
    "Electron build",
  );
  assertEnvValueIncludes(
    buildEnv,
    "APPLE_SIGNING_IDENTITY",
    "secrets.APPLE_SIGNING_IDENTITY",
    "Electron build",
  );
  assertEnvValueIncludes(
    buildEnv,
    "APPLE_ID",
    "secrets.APPLE_ID",
    "Electron build",
  );
  assertEnvValueIncludes(
    buildEnv,
    "APPLE_APP_SPECIFIC_PASSWORD",
    "secrets.APPLE_PASSWORD",
    "Electron build",
  );
  assertEnvValueIncludes(
    buildEnv,
    "APPLE_TEAM_ID",
    "secrets.APPLE_TEAM_ID",
    "Electron build",
  );

  const verifyStep = stepByName(steps, "Verify Electron package resources");
  for (const required of [
    "scripts/electron/verify-package-resources.mjs",
    "--package-root release-electron",
    '--platform "${{ matrix.host_platform }}"',
    '--arch "${{ matrix.arch }}"',
    "desktop-resources.manifest.json",
  ]) {
    assertIncludes(
      verifyStep?.run,
      required,
      "Electron package resource verification",
    );
  }

  const stageStep = stepByName(steps, "Stage Electron release assets");
  assertIncludes(
    stageStep?.run,
    "scripts/electron/stage-release-assets.mjs",
    "Electron release staging",
  );

  const windowsNMinusOneStep = stepByName(
    steps,
    "Download Windows N-1 Squirrel installer",
  );
  assertIncludes(
    windowsNMinusOneStep?.if,
    "matrix.host_platform == 'win32'",
    "Windows Squirrel N-1 download condition",
  );
  for (const required of [
    "selectNMinusOneVersion",
    "gh release download",
    "Lime-$N_MINUS_ONE_VERSION.Setup.exe",
  ]) {
    assertIncludes(
      windowsNMinusOneStep?.run,
      required,
      "Windows Squirrel N-1 download",
    );
  }

  const windowsRcSmokeStep = stepByName(
    steps,
    "Smoke installed Windows Squirrel candidate",
  );
  assertIncludes(
    windowsRcSmokeStep?.if,
    "matrix.host_platform == 'win32'",
    "Windows Squirrel RC smoke condition",
  );
  for (const required of [
    "scripts/electron/windows-squirrel-rc-smoke.mjs",
    '--installer-dir "release-assets/${{ matrix.target }}"',
    '--candidate-feed-dir "release-assets/${{ matrix.target }}"',
    "--evidence-dir",
    "--n-minus-one-installer-dir",
    "--n-minus-one-version",
    "--version",
  ]) {
    assertIncludes(
      windowsRcSmokeStep?.run,
      required,
      "Windows Squirrel RC smoke",
    );
  }

  const windowsRcEvidenceStep = stepByName(
    steps,
    "Upload Windows Squirrel RC evidence",
  );
  assertIncludes(
    windowsRcEvidenceStep?.if,
    "always()",
    "Windows Squirrel RC evidence upload condition",
  );
  assertIncludes(
    windowsRcEvidenceStep?.if,
    "matrix.host_platform == 'win32'",
    "Windows Squirrel RC evidence upload condition",
  );
  if (windowsRcEvidenceStep?.uses !== "actions/upload-artifact@v4") {
    throw new Error(
      "Windows Squirrel RC evidence must use actions/upload-artifact@v4",
    );
  }
  assertIncludes(
    windowsRcEvidenceStep?.with?.path,
    ".lime/qc/windows-squirrel-rc",
    "Windows Squirrel RC evidence upload",
  );

  const windowsCodeModeStep = stepByName(
    steps,
    "Run installed Windows CodeMode Gate B",
  );
  assertIncludes(
    windowsCodeModeStep?.if,
    "matrix.host_platform == 'win32'",
    "Windows CodeMode Gate B condition",
  );
  for (const required of [
    "scripts/agent-runtime/code-mode-electron-gate-b.mjs",
    "--electron-executable",
    "code-mode-electron-gate-b-windows",
    "--timeout-ms 600000",
  ]) {
    assertIncludes(
      windowsCodeModeStep?.run,
      required,
      "Windows CodeMode Gate B",
    );
  }

  const windowsCodeModeEvidenceStep = stepByName(
    steps,
    "Upload Windows CodeMode Gate B evidence",
  );
  assertIncludes(
    windowsCodeModeEvidenceStep?.if,
    "always()",
    "Windows CodeMode Gate B evidence upload condition",
  );
  assertIncludes(
    windowsCodeModeEvidenceStep?.if,
    "matrix.host_platform == 'win32'",
    "Windows CodeMode Gate B evidence upload condition",
  );
  if (windowsCodeModeEvidenceStep?.uses !== "actions/upload-artifact@v4") {
    throw new Error(
      "Windows CodeMode Gate B evidence must use actions/upload-artifact@v4",
    );
  }
  assertIncludes(
    windowsCodeModeEvidenceStep?.with?.path,
    "code-mode-electron-gate-b-windows",
    "Windows CodeMode Gate B evidence upload",
  );

  const windowsNativeHostStep = stepByName(
    steps,
    "Run installed Windows native host Gate B",
  );
  assertIncludes(
    windowsNativeHostStep?.if,
    "matrix.host_platform == 'win32'",
    "Windows native host Gate B condition",
  );
  for (const required of [
    "scripts/electron/windows-native-host-gate-b.mjs",
    "--electron-executable",
    "windows-native-host-gate-b",
    "--timeout-ms 120000",
  ]) {
    assertIncludes(
      windowsNativeHostStep?.run,
      required,
      "Windows native host Gate B",
    );
  }

  const windowsNativeHostEvidenceStep = stepByName(
    steps,
    "Upload Windows native host Gate B evidence",
  );
  assertIncludes(
    windowsNativeHostEvidenceStep?.if,
    "always()",
    "Windows native host Gate B evidence upload condition",
  );
  assertIncludes(
    windowsNativeHostEvidenceStep?.if,
    "matrix.host_platform == 'win32'",
    "Windows native host Gate B evidence upload condition",
  );
  if (windowsNativeHostEvidenceStep?.uses !== "actions/upload-artifact@v4") {
    throw new Error(
      "Windows native host Gate B evidence must use actions/upload-artifact@v4",
    );
  }
  assertIncludes(
    windowsNativeHostEvidenceStep?.with?.path,
    "windows-native-host-gate-b",
    "Windows native host Gate B evidence upload",
  );
}

function assertPublishSteps(workflow) {
  const publishJob = workflow?.jobs?.publish_release_assets;
  const steps = publishJob?.steps;
  if (!Array.isArray(steps)) {
    throw new Error("publish_release_assets job must define steps");
  }

  const inspectStep = stepByName(steps, "Inspect Electron release assets");
  assertNoLegacyUpdaterAssets(
    inspectStep?.run || "",
    "GitHub Release asset inspection",
  );

  const prepareStep = stepByName(steps, "Prepare GitHub release upload assets");
  assertIncludes(
    prepareStep?.run,
    "scripts/electron/prepare-github-release-assets.mjs",
    "GitHub Release asset preparation",
  );

  const updaterJob = workflow?.jobs?.publish_updater_assets_r2;
  const updaterSteps = updaterJob?.steps;
  if (!Array.isArray(updaterSteps)) {
    throw new Error("publish_updater_assets_r2 job must define steps");
  }
  const uploadStep = stepByName(
    updaterSteps,
    "Upload Electron updater assets to Cloudflare R2",
  );
  assertIncludes(
    uploadStep?.run,
    "scripts/electron/update-feed-r2-upload-plan.mjs",
    "R2 updater upload plan",
  );
  const cleanupStep = stepByName(
    updaterSteps,
    "Clean old updater assets from Cloudflare R2",
  );
  assertIncludes(
    cleanupStep?.run,
    "scripts/electron/r2-release-cleanup-plan.mjs",
    "R2 updater cleanup plan",
  );
}

function assertNoRetiredPackagingInputs(content, label) {
  for (const forbidden of [
    "electron-builder",
    "electron-updater",
    "nsis",
    "plan-electron-updater-r2-upload",
  ]) {
    if (content.toLowerCase().includes(forbidden.toLowerCase())) {
      throw new Error(
        `${label} must not use retired packaging input: ${forbidden}`,
      );
    }
  }
}

function listRepositoryFiles(root) {
  const directories = [""];
  const files = [];

  while (directories.length > 0) {
    const currentRelativePath = directories.pop();
    const absoluteDir = path.join(root, currentRelativePath);
    let entries;
    try {
      entries = fs.readdirSync(absoluteDir, { withFileTypes: true });
    } catch (error) {
      if (error?.code === "ENOENT") {
        continue;
      }
      throw error;
    }

    for (const entry of entries) {
      const relativePath = currentRelativePath
        ? path.join(currentRelativePath, entry.name)
        : entry.name;
      const normalizedPath = relativePath.replace(/\\/g, "/");

      if (entry.isDirectory()) {
        if (IGNORED_RETIRED_FILE_DIRS.has(entry.name)) {
          continue;
        }
        directories.push(relativePath);
        continue;
      }
      if (entry.isFile()) {
        files.push(normalizedPath);
      }
    }
  }

  return files.sort();
}

function assertNoRetiredPackagingFiles(
  repositoryRoot = DEFAULT_REPOSITORY_ROOT,
) {
  const root = path.resolve(repositoryRoot);
  const retiredFiles = listRepositoryFiles(root).filter((filePath) =>
    RETIRED_PACKAGING_FILE_PATTERNS.some((pattern) => pattern.test(filePath)),
  );
  if (retiredFiles.length > 0) {
    throw new Error(
      `retired Electron packaging files must not exist in the repository: ${retiredFiles.join(", ")}`,
    );
  }
}

function assertForgeConfig(forgeConfigPath = DEFAULT_FORGE_CONFIG_PATH) {
  const forgeConfig = fs.readFileSync(forgeConfigPath, "utf8");

  assertNoRetiredPackagingInputs(forgeConfig, "Forge config");

  for (const required of [
    "@electron-forge/maker-dmg",
    "@electron-forge/maker-zip",
    "@electron-forge/maker-squirrel",
    "new MakerDMG",
    "new MakerZIP",
    "new MakerSquirrel",
    '["darwin"]',
    '["win32"]',
    "macUpdateManifestBaseUrl",
    'updateFeedUrl("darwin", arch',
    "RELEASE_OUTPUT_DIR",
    "LIME_ELECTRON_FORGE_OUT_DIR",
    "dist-electron/app-server.release.json",
    "preparePackagedDesktopResources",
    "dist-electron/app-server",
  ]) {
    assertIncludes(forgeConfig, required, "Forge current maker config");
  }

  if (forgeConfig.includes("afterComplete")) {
    throw new Error(
      "Forge macOS branding hook must run before signing via afterCopyExtraResources",
    );
  }

  for (const required of [
    "macSignOptions",
    "continueOnError: false",
    "identityValidation",
    "preAutoEntitlements: false",
    "preEmbedProvisioningProfile: false",
    "optionsForFile",
    "MACOS_APP_ENTITLEMENTS",
    "isTopLevelAppBundle",
    "MACOS_NON_CODE_RESOURCE_EXTENSIONS",
    "isMacOSNonCodeResource",
    '!normalized.includes(".app/")',
    "const releaseSigning =",
    'env.LIME_ELECTRON_SIGN === "1"',
    "Boolean(env.LIME_MACOS_KEYCHAIN)",
    "identityValidation: releaseSigning",
    "return releaseSigning",
    ': { hardenedRuntime: false, timestamp: "none" }',
    'return { timestamp: "none" }',
    "if (!releaseSigning)",
    'options.identity = "-"',
    "lime-rs/entitlements.plist",
    "hardenedRuntime: releaseSigning",
    "signatureFlags",
    "APPLE_SIGNING_IDENTITY",
    "LIME_MACOS_KEYCHAIN",
    "osxSign: macSignOptions()",
    "macNotarizeOptions",
    "APPLE_ID",
    "APPLE_APP_SPECIFIC_PASSWORD",
    "APPLE_PASSWORD",
    "APPLE_TEAM_ID",
    "osxNotarize: macNotarizeOptions()",
    "afterCopyExtraResources",
    "brandMacHelperApps",
  ]) {
    assertIncludes(forgeConfig, required, "Forge macOS signing config");
  }

  for (const required of [
    "windowsSigningOptions",
    "LIME_WINDOWS_SIGNING_CERTIFICATE_FILE",
    "LIME_WINDOWS_SIGNING_CERTIFICATE_PASSWORD",
    "certificateFile",
    "certificatePassword",
    "squirrelConfig",
    "SQUIRREL_PACKAGE_NAME",
    "noMsi: true",
    "setupExe",
    "Setup.exe",
    "setupIcon",
    "lime-rs/icons/icon.ico",
    "windowsSquirrelRemoteReleasesUrl",
    "windowsSquirrelRemoteReleasesOptions",
    "LIME_WINDOWS_SQUIRREL_REMOTE_RELEASES_URL",
    "...windowsSquirrelRemoteReleasesOptions(options)",
    "...windowsSigningOptions(options)",
  ]) {
    assertIncludes(forgeConfig, required, "Forge Windows Squirrel config");
  }

  if (forgeConfig.includes('remoteReleases: updateFeedUrl("win32"')) {
    throw new Error(
      "Forge Windows Squirrel config must not use runtime update feed as remoteReleases",
    );
  }
}

function assertReleaseAssetPipelineScripts({
  githubReleaseAssetsPath = DEFAULT_GITHUB_RELEASE_ASSETS_PATH,
  stageAssetsPath = DEFAULT_STAGE_RELEASE_ASSETS_PATH,
  updateFeedUploadPlanPath = DEFAULT_UPDATE_FEED_UPLOAD_PLAN_PATH,
} = {}) {
  const stageAssets = fs.readFileSync(stageAssetsPath, "utf8");
  for (const required of [
    '"aarch64-apple-darwin"',
    '"x86_64-apple-darwin"',
    '"x86_64-pc-windows-msvc"',
    'metadataNames: ["RELEASES.json"]',
    'metadataNames: ["RELEASES"]',
    'installerExtensions: [".dmg"]',
    'installerExtensions: [".exe"]',
    'installerBasenameIncludes: ["setup"]',
    'archiveExtensions: [".zip"]',
    'archiveExtensions: [".nupkg"]',
    "assertNoRetiredUpdaterAssets",
    "assertNoLocalMacUpdateManifest",
    "local Electron updater feed URLs are not allowed in release staging",
    "legacy updater assets are not allowed in Electron release staging",
  ]) {
    assertIncludes(stageAssets, required, "Electron release staging script");
  }

  const uploadPlan = fs.readFileSync(updateFeedUploadPlanPath, "utf8");
  for (const required of [
    '"aarch64-apple-darwin": "darwin-arm64"',
    '"x86_64-apple-darwin": "darwin-x64"',
    '"x86_64-pc-windows-msvc": "win32-x64"',
    'basename === "RELEASES.json"',
    'basename === "RELEASES"',
    "\\.(dmg|exe|nupkg|zip)",
    "assertNoRetiredUpdaterAssets",
    "legacy updater assets are not allowed in Electron release",
  ]) {
    assertIncludes(uploadPlan, required, "Electron R2 updater upload plan");
  }

  const githubReleaseAssets = fs.readFileSync(githubReleaseAssetsPath, "utf8");
  for (const required of [
    "duplicateTargetLabel",
    "macos-arm64",
    "macos-x64",
    "windows-x64",
    "assertNoRetiredUpdaterAssets",
    "legacy updater assets are not allowed in Electron GitHub release assets",
  ]) {
    assertIncludes(
      githubReleaseAssets,
      required,
      "Electron GitHub release asset preparation",
    );
  }
}

function validateReleaseWorkflow({
  forgeConfigPath = DEFAULT_FORGE_CONFIG_PATH,
  githubReleaseAssetsPath = DEFAULT_GITHUB_RELEASE_ASSETS_PATH,
  repositoryRoot = DEFAULT_REPOSITORY_ROOT,
  stageAssetsPath = DEFAULT_STAGE_RELEASE_ASSETS_PATH,
  updateFeedUploadPlanPath = DEFAULT_UPDATE_FEED_UPLOAD_PLAN_PATH,
  workflowPath = DEFAULT_WORKFLOW_PATH,
} = {}) {
  const workflowText = fs.readFileSync(workflowPath, "utf8");
  const workflow = YAML.parse(workflowText);
  const buildJob = workflow?.jobs?.build;
  if (!buildJob) {
    throw new Error("release workflow missing build job");
  }

  assertMatrix(buildJob);
  assertBuildSteps(buildJob);
  assertPublishSteps(workflow);
  assertNoRetiredPackagingInputs(workflowText, "release workflow");
  assertNoRetiredPackagingFiles(repositoryRoot);
  assertForgeConfig(forgeConfigPath);
  assertReleaseAssetPipelineScripts({
    githubReleaseAssetsPath,
    stageAssetsPath,
    updateFeedUploadPlanPath,
  });
}

function main() {
  const workflowPath = process.argv[2] || DEFAULT_WORKFLOW_PATH;
  validateReleaseWorkflow({ workflowPath });
  console.log("[electron-release-workflow] ok");
}

const isCli =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isCli) {
  main();
}

export { validateReleaseWorkflow };
