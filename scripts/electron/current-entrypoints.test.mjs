import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const bundledElectronRuntimePackages = new Set([
  "@limecloud/app-server-client",
]);

function readPackageJson() {
  return JSON.parse(fs.readFileSync("package.json", "utf8"));
}

function readPackageScripts() {
  return readPackageJson().scripts ?? {};
}

function readFile(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function retiredHostTopic() {
  return ["ta", "uri"].join("");
}

function retiredHostConfigStem() {
  return ["ta", "uri.conf"].join("");
}

function retiredRootDesktopHostFiles() {
  const configStem = retiredHostConfigStem();
  return [
    "lime-rs/build.rs",
    "lime-rs/src/main.rs",
    "lime-rs/src/lib.rs",
    ["lime-rs/", configStem, ".json"].join(""),
    ["lime-rs/", configStem, ".headless.json"].join(""),
    ["lime-rs/", retiredHostTopic(), ".windows.conf.json"].join(""),
  ];
}

function retiredHostCleanupCandidateFiles() {
  return [];
}

function retiredHostBuildInputTerms() {
  return [
    retiredHostConfigStem(),
    ["lime-rs/", retiredHostConfigStem()].join(""),
    ["lime-rs/", retiredHostTopic(), ".windows.conf"].join(""),
    ["src-", "ta", "uri"].join(""),
    ["@", "ta", "uri-apps/cli"].join(""),
    ["npm run ", "ta", "uri"].join(""),
    ["node_modules/.bin/", "ta", "uri"].join(""),
  ];
}

function retiredHostRuntimeTerms() {
  return [
    ...retiredHostBuildInputTerms(),
    ["__", "TA", "URI__"].join(""),
    ["__", "TA", "URI_INTERNALS__"].join(""),
    ["TA", "URI_"].join(""),
    ["ta", "uri::"].join(""),
    ["#[", "ta", "uri::command]"].join(""),
  ];
}

function expectNoRetiredHostBuildInput(content, label) {
  for (const term of retiredHostBuildInputTerms()) {
    expect(content, label).not.toContain(term);
  }
}

function expectNoRetiredHostRuntimeInput(content, label) {
  for (const term of retiredHostRuntimeTerms()) {
    expect(content, label).not.toContain(term);
  }
}

function expectNoElectronRuntimeEsmImport(content, label) {
  const runtimeImportLines = content
    .split("\n")
    .filter((line) => /^\s*import\s+(?!type\b).*from "electron";/.test(line));

  expect(runtimeImportLines, label).toEqual([]);
}

function barePackageName(specifier) {
  if (
    specifier.startsWith(".") ||
    specifier.startsWith("/") ||
    specifier.startsWith("#") ||
    specifier.startsWith("node:") ||
    specifier === "electron"
  ) {
    return null;
  }
  if (specifier.startsWith("@")) {
    return specifier.split("/").slice(0, 2).join("/");
  }
  return specifier.split("/")[0] ?? null;
}

function collectRuntimeBareImports(filePath) {
  const content = readFile(filePath);
  const imports = new Set();
  const importFromPattern =
    /^\s*import\s+(?!type\b)[^;]*?\s+from\s+["']([^"']+)["']/gm;
  const sideEffectImportPattern = /^\s*import\s+["']([^"']+)["']/gm;
  const requirePattern = /\brequire\(\s*["']([^"']+)["']\s*\)/g;

  for (const pattern of [
    importFromPattern,
    sideEffectImportPattern,
    requirePattern,
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

function listFiles(root, predicate = () => true) {
  const entries = fs.readdirSync(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const filePath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...listFiles(filePath, predicate));
      continue;
    }
    if (entry.isFile() && predicate(filePath)) {
      files.push(filePath);
    }
  }
  return files.sort();
}

function currentElectronEntrypointFiles() {
  const explicitFiles = [
    "package.json",
    "forge.config.mjs",
    "scripts/electron/brand-mac-helper-apps.mjs",
    "scripts/check-app-version-consistency.mjs",
    "scripts/electron/build-host.mjs",
    "scripts/electron/build-renderer.mjs",
    "scripts/electron/build-renderer-smoke.mjs",
    "scripts/electron/renderer-build-env.mjs",
    "scripts/electron/copy-desktop-assets.mjs",
    "scripts/electron/smoke.mjs",
    "scripts/electron/make-zip-local-feed.mjs",
    "scripts/electron/update-feed-r2-upload-plan.mjs",
    "scripts/electron/prepare-app-server-assets.mjs",
    "scripts/electron/run-dev.mjs",
    "scripts/electron/run-package-dir.mjs",
    "scripts/electron/run-preview.mjs",
    "scripts/electron/stage-release-assets.mjs",
    "scripts/electron/verify-package-resources.mjs",
    "scripts/electron/macos-native-host-gate-b.mjs",
    "scripts/electron/lib/macos-native-host-electron-gate-b.mjs",
    "scripts/electron/macos-window-fixture.swift",
    "scripts/electron/windows-native-host-gate-b.mjs",
    "scripts/electron/windows-packaged-evidence.mjs",
    "scripts/lib/electron-desktop-resources.mjs",
    "electron/native/macos/macos-native-host.swift",
    "electron/native/macos/macos-screen-capture.swift",
    "electron/native/macos/macos-security.swift",
    "electron/native/macos/macos-window-orchestration.swift",
    "electron/native/macos/macos-accessibility-tree.swift",
    "electron/native/macos/macos-display-watcher.swift",
    "electron/native/macos/macos-media-permissions.swift",
    "electron/native/macos/macos-apple-events.swift",
    "electron/native/windows/windows-native-host.cpp",
  ];
  return [
    ...explicitFiles,
    ...listFiles(".github", (filePath) =>
      /\.(ya?ml|json|md|sh|mjs|js|ts)$/i.test(filePath),
    ),
    ...listFiles("electron", (filePath) =>
      /\.(ya?ml|json|md|mjs|js|ts|tsx)$/i.test(filePath),
    ),
  ];
}

function electronProductionSourceFiles() {
  return listFiles("electron", (filePath) => {
    const normalized = filePath.replace(/\\/g, "/");
    return (
      /\.(?:mjs|js|ts)$/i.test(normalized) &&
      !normalized.endsWith(".d.ts") &&
      !normalized.endsWith(".test.ts") &&
      !normalized.endsWith(".test.mjs") &&
      !normalized.endsWith(".test.js")
    );
  });
}

describe("Electron current package entrypoints", () => {
  it("default desktop commands point to Electron current", () => {
    const scripts = readPackageScripts();

    expect(scripts.dev).toBe("npm run electron:dev");
    expect(scripts.build).toBe("npm run electron:build");
    expect(scripts.preview).toBe("npm run electron:start");
    expect(scripts["verify:gui-smoke"]).toBe("npm run smoke:electron");
    expect(scripts["smoke:electron"]).toContain("scripts/electron/smoke.mjs");
    expect(scripts["electron:make:zip-local-feed"]).toBe(
      "node scripts/electron/make-zip-local-feed.mjs",
    );
  });

  it("package scripts do not expose retired desktop host as a GUI current path", () => {
    const scripts = readPackageScripts();
    const retiredHostScripts = Object.entries(scripts).filter(
      ([name, command]) =>
        `${name} ${command}`.toLowerCase().includes("retired-host"),
    );

    expect(retiredHostScripts).toEqual([]);
  });

  it("does not keep retired root desktop host app entry files", () => {
    for (const filePath of retiredRootDesktopHostFiles()) {
      expect(fs.existsSync(filePath), filePath).toBe(false);
    }
  });

  it("keeps retired host cleanup candidates out of current entrypoints", () => {
    const currentEntrypointContent = currentElectronEntrypointFiles()
      .map((filePath) => readFile(filePath))
      .join("\n");

    for (const filePath of retiredHostCleanupCandidateFiles()) {
      expect(currentEntrypointContent, filePath).not.toContain(filePath);
    }
  });

  it("package scripts, Electron Forge, and version checks do not consume retired host config", () => {
    const scripts = readPackageScripts();
    const scriptEntrypoints = Object.entries(scripts)
      .map(([name, command]) => `${name} ${command}`)
      .join("\n");
    const forgeConfig = readFile("forge.config.mjs");
    const appVersionCheck = readFile(
      "scripts/check-app-version-consistency.mjs",
    );

    expectNoRetiredHostBuildInput(scriptEntrypoints, "package.json scripts");
    expectNoRetiredHostBuildInput(forgeConfig, "forge.config.mjs");
    expectNoRetiredHostBuildInput(
      appVersionCheck,
      "scripts/check-app-version-consistency.mjs",
    );

    expect(forgeConfig).toContain('const PRODUCT_NAME = "Lime"');
    expect(forgeConfig).toContain('const APP_ID = "com.limecloud.lime"');
    expect(forgeConfig).toContain("app-server.release.json");
    expect(forgeConfig).toContain("new MakerDMG");
    expect(forgeConfig).toContain("new MakerZIP");
    expect(forgeConfig).toContain("new MakerSquirrel");
    expect(scriptEntrypoints).toContain(
      "electron:make:zip-local-feed node scripts/electron/make-zip-local-feed.mjs",
    );
    expect(appVersionCheck).toContain("Cargo.toml");
    expect(appVersionCheck).toContain("packages");
    expect(appVersionCheck).toContain("packages/cli");
  });

  it("Electron current build, release, CI, and host files stay free of retired host runtime inputs", () => {
    for (const filePath of currentElectronEntrypointFiles()) {
      expectNoRetiredHostRuntimeInput(readFile(filePath), filePath);
    }
  });

  it("Electron main process avoids ESM runtime imports from electron", () => {
    const mainProcessFiles = [
      "electron/main.ts",
      "electron/updateHost.ts",
      "electron/appServerHost.ts",
      "electron/hostCommands.ts",
    ];
    for (const filePath of mainProcessFiles) {
      expectNoElectronRuntimeEsmImport(readFile(filePath), filePath);
      expect(readFile(filePath), filePath).toContain(
        'from "./electronRuntime"',
      );
    }

    const runtime = readFile("electron/electronRuntime.ts");
    expect(runtime).toContain("createRequire(import.meta.url)");
    expect(runtime).toContain('requireElectron("electron")');
    expectNoElectronRuntimeEsmImport(runtime, "electron/electronRuntime.ts");
  });

  it("Electron production runtime bare imports stay in packaged dependencies", () => {
    const packageJson = readPackageJson();
    const runtimeDependencies = new Set(
      Object.keys(packageJson.dependencies ?? {}),
    );
    const missing = [];

    for (const filePath of electronProductionSourceFiles()) {
      for (const packageName of collectRuntimeBareImports(filePath)) {
        if (
          !runtimeDependencies.has(packageName) &&
          !bundledElectronRuntimePackages.has(packageName)
        ) {
          missing.push(`${filePath}: ${packageName}`);
        }
      }
    }

    expect(
      missing,
      "Forge package uses prune=true, so Electron production runtime imports must be root dependencies.",
    ).toEqual([]);
  });

  it("legacy verify-gui-smoke script delegates to Electron smoke", () => {
    const content = readFile("scripts/verify-gui-smoke.mjs");

    expect(content).toContain('"smoke:electron"');
    expect(content).toContain("Electron Desktop Host current GUI smoke");
    expect(content).not.toMatch(/retired-host/i);
    expect(content).not.toContain("headless");
    expect(content).not.toContain("retired-host.conf");
    expect(content).not.toContain("node_modules/.bin");
  });

  it("Electron smoke gates Claw workbench shell and composer readiness", () => {
    const mainContent = readFile("electron/main.ts");
    const memorySmokeContent = readFile("electron/smokeMemorySettings.ts");
    const smokeChecksContent = readFile("electron/smokeChecks.ts");
    const smokeEvidenceContent = readFile("electron/smokeEvidence.ts");
    const smokeScript = readFile("scripts/electron/smoke.mjs");

    expect(mainContent).toContain("createElectronSmokeRunner");
    expect(mainContent).not.toContain("function runElectronSmokeChecks");
    expect(smokeChecksContent).toContain("waitForElectronSmokeWorkbenchReady");
    expect(smokeChecksContent).toContain(
      "waitForElectronSmokeMemorySettingsReady",
    );
    expect(mainContent).toContain("showMainWindowDuringStartup");
    expect(mainContent).toContain("shouldShowMainWindowDuringStartup");
    expect(mainContent).toContain("LIME_ELECTRON_SMOKE_VISIBLE");
    expect(mainContent).toContain("window.showInactive()");
    expect(smokeChecksContent).toContain(
      '[data-testid="workspace-shell-scene"]',
    );
    expect(smokeChecksContent).toContain(
      '[data-testid="inputbar-core-container"]',
    );
    expect(smokeChecksContent).toContain('textarea[name="agent-chat-message"]');
    expect(smokeChecksContent).toContain("claw workbench shell ready");
    expect(smokeChecksContent).toContain("memory settings ready");
    expect(smokeChecksContent).toContain("reloadElectronSmokeRenderer");
    expect(smokeChecksContent).toContain(
      "claw workbench shell ready after reload",
    );
    expect(smokeChecksContent).toContain("collectRendererPageErrorCount");
    expect(smokeChecksContent).toContain("collectElectronSmokeLayoutEvidence");
    expect(smokeChecksContent).toContain("ELECTRON_SMOKE_LAYOUT_VIEWPORTS");
    expect(smokeChecksContent).toContain("window.setSize");
    expect(smokeChecksContent).toContain("window.getSize");
    expect(memorySmokeContent).toContain(
      '[data-testid="app-sidebar-account-model-settings"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-sidebar-tab-memory"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-floating-nav-button"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-floating-tab-memory"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-memory-store-panel"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-memory-review-refresh"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-memory-index-rebuild"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-memory-consolidate"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-memory-rollout-refresh"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-memory-rollout-consolidate"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-memory-soul-panel"]',
    );
    expect(memorySmokeContent).toContain(
      '[data-testid="settings-memory-advanced-panel"]',
    );
    expect(memorySmokeContent).toContain("/灵感库/");
    expect(memorySmokeContent).toContain("/MemoryPage/");
    expect(mainContent).not.toContain('"turn/start"');
    expect(mainContent).not.toContain('"test_api_key_provider_chat"');
    expect(smokeChecksContent).toContain("LIME_GATE_RUN_ID");
    expect(smokeChecksContent).toContain("LIME_ELECTRON_SMOKE_EVIDENCE_DIR");
    expect(smokeChecksContent).toContain("app_server_handle_json_lines");
    expect(smokeChecksContent).toContain("electron-ipc");
    expect(smokeChecksContent).toContain("capturePage");
    expect(smokeChecksContent).toContain("legacyCommandHitCount");
    expect(smokeChecksContent).toContain("mockFallbackHitCount");
    expect(smokeChecksContent).toContain("pageErrorCount");
    expect(smokeChecksContent).toContain("rendererCrashCount");
    expect(smokeEvidenceContent).toContain(
      'ELECTRON_SMOKE_PROOF_LEVEL = "Gate B-F"',
    );
    expect(smokeEvidenceContent).toContain("currentAppServerMethodObserved");
    expect(smokeEvidenceContent).toContain("noMockFallbackHits");
    expect(smokeEvidenceContent).toContain("screenshotCaptured");
    expect(smokeEvidenceContent).toContain('surfaceId: "SHELL-01"');
    expect(smokeEvidenceContent).toContain('proof: "gate-b-f"');
    expect(smokeEvidenceContent).toContain('complete: result === "pass"');
    expect(smokeEvidenceContent).toContain("workbenchReloadReady");
    expect(smokeEvidenceContent).toContain("ELECTRON_SMOKE_LAYOUT_PROOF_LEVEL");
    expect(smokeEvidenceContent).toContain(
      '"electron-responsive-layout-contract"',
    );
    expect(smokeEvidenceContent).toContain("BrowserWindow setSize/getSize");
    expect(smokeEvidenceContent).toContain("layoutGeometryStable");
    expect(smokeScript).toContain("mkdtempSync");
    expect(smokeScript).toContain("ELECTRON_E2E_USER_DATA_DIR");
    expect(smokeScript).toContain('LIME_ELECTRON_E2E: "1"');
    expect(smokeScript).toContain(
      'packagedExecutable ? { LIME_ELECTRON_BRAND_DEV_APP: "0" } : {}',
    );
    expect(smokeScript).toContain("LIME_ELECTRON_SMOKE_VISIBLE");
    expect(smokeScript).toContain("LIME_GATE_RUN_ID");
    expect(smokeScript).toContain("LIME_ELECTRON_SMOKE_EVIDENCE_DIR");
    expect(smokeScript).toContain('path.join(evidenceDir, "summary.json")');
    expect(smokeScript).toContain("timed out waiting for renderer/workbench");
  });

  it("dev launcher can expose a guarded Electron CDP port for real GUI续测", () => {
    const runDev = readFile("scripts/electron/run-dev.mjs");

    expect(runDev).toContain("LIME_ELECTRON_REMOTE_DEBUGGING_PORT");
    expect(runDev).toContain("--remote-debugging-port=");
    expect(runDev).toContain("normalizeRemoteDebuggingPort");
    expect(runDev).toContain("must be between 1 and 65535");
    expect(runDev).toContain("resolveElectronDevLaunchEnv");
    expect(runDev).toContain("ELECTRON_E2E_USER_DATA_DIR");
    expect(runDev).toContain("LIME_ELECTRON_E2E");
    expect(runDev).toContain("LIME_ELECTRON_DEV_HTTP_BRIDGE");
  });

  it("Electron packaged renderer uses relative assets under file URLs", () => {
    const viteConfig = readFile("vite.config.ts");
    const buildRenderer = readFile("scripts/electron/build-renderer.mjs");
    const smokeBuildRenderer = readFile(
      "scripts/electron/build-renderer-smoke.mjs",
    );
    const rendererBuildEnv = readFile(
      "scripts/electron/renderer-build-env.mjs",
    );

    expect(buildRenderer).toContain("LIME_ELECTRON_RENDERER");
    expect(buildRenderer).toContain("acquireRendererBuildLock");
    expect(buildRenderer).toContain("rendererBuildEnv");
    expect(buildRenderer).toContain("startRendererBuildHeartbeat");
    expect(smokeBuildRenderer).toContain("LIME_ELECTRON_RENDERER");
    expect(smokeBuildRenderer).toContain('LIME_VITE_EMPTY_OUT_DIR = "0"');
    expect(smokeBuildRenderer).toContain("acquireRendererBuildLock");
    expect(smokeBuildRenderer).toContain("rendererBuildEnv");
    expect(smokeBuildRenderer).toContain("startRendererBuildHeartbeat");
    expect(rendererBuildEnv).toContain("--max-old-space-size=8192");
    expect(rendererBuildEnv).toContain("NODE_OPTIONS");
    expect(rendererBuildEnv).toContain("still running after");
    expect(viteConfig).toContain("base:");
    expect(viteConfig).toContain("emptyOutDir:");
    expect(viteConfig).toContain("keepExistingOutDir ? false : undefined");
    expect(viteConfig).toContain('isElectronRenderer ? "./" : undefined');
    expect(viteConfig).toContain('find: "@limecloud/app-server-client"');
    expect(viteConfig).toContain("./packages/app-server-client/src/browser.ts");
  });

  it("build monitor observes Electron package output instead of retired host bundles", () => {
    const content = readFile("scripts/monitor-build.sh");

    expect(content).toContain("Electron 打包进度监控");
    expect(content).toContain("/tmp/electron-build.log");
    expect(content).toContain("release-electron");
    expect(content).toContain(
      "electron-forge|electron:package:dir|electron:dist",
    );
    expect(content).not.toMatch(/retired-host/i);
    expect(content).not.toContain("lime-rs/target/release/bundle");
    expect(content).not.toContain("/tmp/retired-host-build.log");
  });

  it("local install script advertises Electron current desktop commands", () => {
    const content = readFile("lime-rs/install-local.sh");

    expect(content).toContain("Rust CLI");
    expect(content).toContain("Electron 命令接管");
    expect(content).toContain("npm run electron:dev");
    expect(content).toContain("npm run electron:build");
    expect(content).toContain("npm run verify:gui-smoke");
    expectNoRetiredHostBuildInput(content, "lime-rs/install-local.sh");
  });

  it("repository discovery metadata advertises Electron current host", () => {
    const content = readFile(".github/repository-metadata.md");

    expect(content).toContain("electron");
    expect(content).not.toContain(retiredHostTopic());
  });
});
