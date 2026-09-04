import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readFile(path) {
  return fs.readFileSync(path, "utf8");
}

function existingFiles(paths) {
  return paths.filter((path) => fs.existsSync(path));
}

function sectionBetween(content, startHeading, endHeadingPrefix) {
  const start = content.indexOf(startHeading);
  expect(start, `missing section ${startHeading}`).toBeGreaterThanOrEqual(0);
  const afterStart = content.slice(start);
  const end = afterStart.indexOf(endHeadingPrefix, startHeading.length);
  return end === -1 ? afterStart : afterStart.slice(0, end);
}

function retiredGuiCommandPattern() {
  return new RegExp(["npm run ", "ta", "uri(?::dev)?\\b"].join(""), "i");
}

function retiredCliCommandPattern() {
  return new RegExp(["\\b", "ta", "uri dev\\b"].join(""), "i");
}

function retiredHostTerms() {
  return [
    ["ta", "uri:"].join(""),
    ["src-", "ta", "uri"].join(""),
    ["@", "ta", "uri-apps"].join(""),
    ["__TA", "URI__"].join(""),
    ["TA", "URI_"].join(""),
    ["#[", "ta", "uri::command]"].join(""),
    "SC_DISABLE_SPEEDY",
  ];
}

function expectNoRetiredGuiStartupReference(content, label) {
  expect(content, label).not.toMatch(retiredGuiCommandPattern());
  expect(content, label).not.toMatch(retiredCliCommandPattern());
  expect(content, label).not.toContain(["headless ", "Ta", "uri"].join(""));
  expect(content, label).not.toContain(["验证 ", "Ta", "uri 壳"].join(""));
}

function expectNoRetiredCurrentHostReference(content, label) {
  expectNoRetiredGuiStartupReference(content, label);
  for (const term of retiredHostTerms()) {
    expect(content, label).not.toContain(term);
  }
}

function listMarkdownFiles(root) {
  const entries = fs.readdirSync(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = `${root}/${entry.name}`;
    if (entry.isDirectory()) {
      files.push(...listMarkdownFiles(path));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(path);
    }
  }
  return files;
}

function splitSentences(content) {
  return content
    .split(/(?<=[。！？!?])|\n+/u)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function hasCodexReference(sentence) {
  return (
    sentence.includes("/Users/coso/Documents/dev/rust/codex") ||
    sentence.includes("codex-rs") ||
    sentence.includes("Codex CLI")
  );
}

function hasRetiredCodexAppTarget(sentence) {
  return [
    "Codex App UI",
    "Codex App 前端",
    "Electron shell",
    "tray",
    "托盘",
    "Dock",
    "updater",
    "桌面壳",
  ].some((term) => sentence.includes(term));
}

function hasReferenceVerb(sentence) {
  return ["参考", "借鉴", "复用", "覆盖", "推断"].some((term) =>
    sentence.includes(term),
  );
}

function isNegatedReference(sentence) {
  return [
    "不参考",
    "不得把",
    "不得作为",
    "不覆盖",
    "不包含",
    "没有",
    "不能",
    "不是",
    "不从",
    "禁止",
  ].some((term) => sentence.includes(term));
}

function expectNoPositiveCodexAppUiReference(content, label) {
  const offendingSentences = splitSentences(content).filter(
    (sentence) =>
      hasCodexReference(sentence) &&
      hasRetiredCodexAppTarget(sentence) &&
      hasReferenceVerb(sentence) &&
      !isNegatedReference(sentence),
  );

  expect(
    offendingSentences,
    `${label} must not treat Codex CLI/codex-rs as Codex App UI or desktop shell reference`,
  ).toEqual([]);
}

function expectNoLegacyAgentUiCommandGatewayReference(content, label) {
  const forbiddenSnippets = [
    "Tauri Commands agent_runtime_*",
    "Tauri Command 层",
    "Tauri command 主入口",
    "RuntimeApi --> Tauri",
    "Api --> Tauri",
    "Tauri --> AgentCrate",
    "Tauri --> Services",
    "任何新增或修改 Tauri command 必须同步四侧",
    "runtime command + DAO",
    "Tauri command / bridge",
    "协议、Tauri command",
  ];

  for (const snippet of forbiddenSnippets) {
    expect(
      content,
      `${label} must not treat legacy Tauri as AgentUI current`,
    ).not.toContain(snippet);
  }
}

describe("Electron current testing docs guard", () => {
  it("does not recommend retired dev host as a current GUI startup path", () => {
    const requiredDocs = [
      "internal/testing/skills-e2e-testing.md",
      "internal/tests/lime-agent-qc-rollout-plan.md",
    ];
    const optionalDocs = [
      ".codex/skills/lime-playwright-e2e/references/playwright-e2e.md",
    ];
    const docs = [
      ...requiredDocs,
      ...optionalDocs.filter((path) => fs.existsSync(path)),
    ];

    for (const path of docs) {
      expectNoRetiredGuiStartupReference(readFile(path), path);
    }

    const qcloopOperations = readFile(
      "internal/tests/lime-agent-qc-qcloop-operations.md",
    );
    const currentStartupSection = sectionBetween(
      qcloopOperations,
      "## 3. qcloop server 启动环境",
      "\n## ",
    );

    expectNoRetiredGuiStartupReference(
      currentStartupSection,
      "internal/tests/lime-agent-qc-qcloop-operations.md#qcloop-server-startup",
    );
  });

  it("keeps GUI smoke fixtures on Electron current commands", () => {
    const agentQcReportTest = readFile(
      "scripts/lib/agent-qc-report-core.test.ts",
    );
    expect(agentQcReportTest).toContain(
      '"verify:gui-smoke": "npm run smoke:electron"',
    );
    expect(agentQcReportTest).toContain(
      '"smoke:electron": "node scripts/electron/smoke.mjs"',
    );
    expect(agentQcReportTest).not.toMatch(
      /"verify:gui-smoke"\s*:\s*"node scripts\/verify-gui-smoke\.mjs"/,
    );

    const runLockTest = readFile("scripts/lib/gui-smoke-run-lock.test.mjs");
    expect(runLockTest).toContain('command: "npm run smoke:electron"');
    expect(runLockTest).not.toContain(
      'command: "node scripts/verify-gui-smoke.mjs"',
    );
  });

  it("keeps internal testing entrypoints on Electron and App Server current", () => {
    const testingIndex = readFile("internal/test/README.md");
    expect(testingIndex).toContain(
      "Electron Desktop Host + App Server JSON-RPC",
    );
    expect(testingIndex).toContain("packages/app-server-client");
    expect(testingIndex).toContain("src/lib/desktop-host/");
    expect(testingIndex).toContain("smoke:electron");
    expect(testingIndex).toContain("verify:gui-smoke");
    expect(testingIndex).toContain(
      "Gate A Renderer 证据与真实 Electron Gate B fixture",
    );
    expect(testingIndex).toContain("生产路径不得使用其 mock 作为 fallback");

    const p0Scenarios = readFile("internal/tests/agent-qc-p0-scenarios.md");
    expect(p0Scenarios).toContain("npm run test:contracts");
    expect(p0Scenarios).toContain("npm run verify:gui-smoke");
    expect(p0Scenarios).toContain("release / GUI startup smoke");
    expectNoRetiredGuiStartupReference(
      p0Scenarios,
      "internal/tests/agent-qc-p0-scenarios.md",
    );

    const autonomousMatrix = readFile(
      "internal/tests/lime-agent-autonomous-test-execution-matrix.md",
    );
    expect(autonomousMatrix).toContain("Electron dev host");
    expect(autonomousMatrix).toContain("npm run test:contracts");
    expect(autonomousMatrix).toContain("npm run verify:gui-smoke");
    expect(autonomousMatrix).toContain("passive desktop runtime");
  });

  it("keeps E2E and testing strategy current sections on Electron evidence", () => {
    const e2eGuide = readFile("internal/test/e2e-tests.md");
    const e2eCurrentSection = sectionBetween(
      e2eGuide,
      "### current",
      "\n### supplement",
    );

    expect(e2eCurrentSection).toContain("npm run electron:dev");
    expect(e2eCurrentSection).toContain("npm run smoke:electron");
    expect(e2eCurrentSection).toContain("Electron GUI");
    expectNoRetiredGuiStartupReference(
      e2eCurrentSection,
      "internal/test/e2e-tests.md#current",
    );

    const testingStrategy = readFile("internal/test/testing-strategy-2026.md");
    const strategyCurrentSection = sectionBetween(
      testingStrategy,
      "### current",
      "\n### compat",
    );

    expect(strategyCurrentSection).toContain("Electron Desktop Host");
    expect(strategyCurrentSection).toContain("App Server JSON-RPC");
    expect(strategyCurrentSection).toContain("packages/app-server-client");
    expect(strategyCurrentSection).toContain("src/lib/desktop-host/");
    expect(strategyCurrentSection).toContain("npm run smoke:electron");
    expect(strategyCurrentSection).toContain("npm run verify:gui-smoke");
    expectNoRetiredGuiStartupReference(
      strategyCurrentSection,
      "internal/test/testing-strategy-2026.md#current",
    );
  });

  it("keeps high-weight current guidance free of retired host examples", () => {
    const currentGuides = [
      "AGENTS.md",
      "internal/aiprompts/README.md",
      "internal/aiprompts/commands.md",
      "internal/aiprompts/governance.md",
      "internal/aiprompts/hooks.md",
      "internal/aiprompts/mcp.md",
      "internal/aiprompts/overview.md",
      "internal/aiprompts/performance-profiling.md",
      "internal/aiprompts/playwright-e2e.md",
      "internal/aiprompts/quality-workflow.md",
      "internal/aiprompts/workspace.md",
      ...existingFiles([
        ".codex/skills/lime-command-boundary/SKILL.md",
        ".codex/skills/lime-governance/SKILL.md",
        ".codex/skills/lime-quality-workflow/SKILL.md",
        ".codex/skills/lime-playwright-e2e/SKILL.md",
      ]),
      "internal/roadmap/appserver/README.md",
      "internal/roadmap/appserver/architecture.md",
      "internal/roadmap/appserver/consumer-integration.md",
      "internal/roadmap/appserver/flowcharts.md",
      "internal/roadmap/appserver/frontend-electron-migration.md",
      "internal/roadmap/appserver/frontend-integration-matrix.md",
      "internal/roadmap/appserver/implementation-plan.md",
      "internal/roadmap/appserver/prd.md",
      "internal/roadmap/appserver/protocol.md",
      "internal/roadmap/appserver/release-updater.md",
      "internal/roadmap/appserver/sequences.md",
      "internal/roadmap/appserver/service-extraction.md",
      "index.html",
    ];

    for (const filePath of currentGuides) {
      expectNoRetiredCurrentHostReference(readFile(filePath), filePath);
    }
  });

  it("documents the codex-rs reference boundary without treating Codex CLI as Codex App UI", () => {
    const roadmap = readFile("internal/roadmap/appserver/README.md");
    expect(roadmap).toContain("/Users/coso/Documents/dev/rust/codex");
    expect(roadmap).toContain("codex-rs");
    expect(roadmap).toContain("Codex CLI");
    expect(roadmap).toContain("不包含 Codex App 前端实现");
    expect(roadmap).toContain("不得把它当作 Codex App UI");
    expect(roadmap).toContain("Electron shell、托盘、Dock、updater");

    const architecture = readFile("internal/roadmap/appserver/architecture.md");
    expect(architecture).toContain("Codex CLI");
    expect(architecture).toContain("Codex CLI 版本代码");
    expect(architecture).toContain("codex-rs");
    expect(architecture).toContain("不包含 Codex App 前端实现");
    expect(architecture).toContain("不参考 Codex App UI 或桌面壳实现");

    const implementationPlan = readFile(
      "internal/roadmap/appserver/implementation-plan.md",
    );
    expect(implementationPlan).toContain("Codex CLI");
    expect(implementationPlan).toContain("codex-rs");
    expect(implementationPlan).toContain("不覆盖 Codex App UI 或桌面壳实现");
    expect(implementationPlan).toContain(
      "Rust App Server / protocol / client / daemon",
    );

    const execPlan = readFile(
      "internal/exec-plans/app-server-implementation-plan.md",
    );
    expect(execPlan).toContain("Electron Desktop Host JSONL transport");
    expect(execPlan).toContain("App Server JSON-RPC");
    expect(execPlan).toContain("RuntimeCore / domain owner");
    expect(execPlan).not.toContain("Codex App 前端参考");
    expect(execPlan).not.toContain("Codex App UI 参考");
  });

  it("blocks positive Codex App UI or desktop shell references in current docs", () => {
    const scannedFiles = [
      ...listMarkdownFiles("internal/roadmap/appserver"),
      ...listMarkdownFiles("internal/aiprompts"),
      "internal/exec-plans/app-server-implementation-plan.md",
      "scripts/electron/current-docs-guard.test.mjs",
    ];

    for (const filePath of scannedFiles) {
      expectNoPositiveCodexAppUiReference(readFile(filePath), filePath);
    }
  });

  it("keeps Electron release and updater docs aligned with current packaging", () => {
    const releaseUpdater = readFile(
      "internal/roadmap/appserver/release-updater.md",
    );
    expect(releaseUpdater).toContain("forge.config.mjs");
    expect(releaseUpdater).toContain("Electron Forge");
    expect(releaseUpdater).toContain("electron/updateHost.ts");
    expect(releaseUpdater).toContain("Electron 内置 `autoUpdater`");
    expect(releaseUpdater).toContain("ElectronUpdateHost");
    expect(releaseUpdater).toContain("LIME_ELECTRON_UPDATES_URL");
    expect(releaseUpdater).toContain("RELEASES.json");
    expect(releaseUpdater).toContain("RELEASES");
    expect(releaseUpdater).toContain(".nupkg");
    expect(releaseUpdater).toContain("stage-electron-release-assets");
    expect(releaseUpdater).toContain("electron:make:zip-local-feed");
    expect(releaseUpdater).toContain("fail-fast");
    expect(releaseUpdater).toContain("*.app.tar.gz");
    expect(releaseUpdater).toContain("*.sig");
    expect(releaseUpdater).toContain("darwin-arm64");
    expect(releaseUpdater).toContain("darwin-x64");
    expect(releaseUpdater).toContain("win32-x64");
    expect(releaseUpdater).toContain("Cloudflare R2");
    expect(releaseUpdater).toContain("LIME_ELECTRON_SIGN");
    expect(releaseUpdater).toContain("LIME_MACOS_KEYCHAIN");
    expect(releaseUpdater).toContain("APPLE_APP_SPECIFIC_PASSWORD");
    expect(releaseUpdater).toContain("WINDOWS_SIGNING_CERTIFICATE");
    expect(releaseUpdater).toContain("LIME_WINDOWS_SIGNING_CERTIFICATE_FILE");
    expect(releaseUpdater).toContain(
      "LIME_WINDOWS_SIGNING_CERTIFICATE_PASSWORD",
    );
    expect(releaseUpdater).toContain(
      "LIME_WINDOWS_SQUIRREL_REMOTE_RELEASES_URL",
    );
    expect(releaseUpdater).toContain("可选但成对");
    expect(releaseUpdater).toContain("unsigned Forge Squirrel installer");
    expect(releaseUpdater).toContain("app-server.release.json");
    expect(releaseUpdater).toContain("不进入 App Server JSON-RPC");
    expect(releaseUpdater).toContain("不从 Codex App UI 推断");
    expectNoRetiredCurrentHostReference(
      releaseUpdater,
      "internal/roadmap/appserver/release-updater.md",
    );

    const forgeConfig = readFile("forge.config.mjs");
    expect(forgeConfig).toContain('const APP_ID = "com.limecloud.lime"');
    expect(forgeConfig).toContain('const PRODUCT_NAME = "Lime"');
    expect(forgeConfig).toContain("LIME_ELECTRON_FORGE_OUT_DIR");
    expect(forgeConfig).toContain("macUpdateManifestBaseUrl");
    expect(forgeConfig).toContain("remoteReleases");
    expect(forgeConfig).toContain("LIME_ELECTRON_UPDATES_URL");
    expect(forgeConfig).toContain("app-server.release.json");
    expect(forgeConfig).toContain("dist-electron/app-server");
    expect(forgeConfig).toContain("new MakerDMG");
    expect(forgeConfig).toContain("new MakerZIP");
    expect(forgeConfig).toContain("new MakerSquirrel");
    expect(forgeConfig).toContain("SQUIRREL_PACKAGE_NAME");
    expect(forgeConfig).toContain("certificateFile");
    expect(forgeConfig).toContain("certificatePassword");
    expect(forgeConfig).toContain("LIME_WINDOWS_SIGNING_CERTIFICATE_FILE");
    expect(forgeConfig).toContain("LIME_WINDOWS_SQUIRREL_REMOTE_RELEASES_URL");

    const electronRuntime = readFile("electron/electronRuntime.ts");
    expect(electronRuntime).toContain(
      'import type * as Electron from "electron"',
    );
    expect(electronRuntime).toContain('requireElectron("electron")');
    expect(electronRuntime).toContain("autoUpdater,");

    const updateHost = readFile("electron/updateHost.ts");
    expect(updateHost).toContain(
      'import { app, autoUpdater } from "./electronRuntime"',
    );
    expect(updateHost).toContain(
      'serverType: process.platform === "darwin" ? "json" : "default"',
    );
    expect(updateHost).toContain("RELEASES.json");
    expect(updateHost).toContain("LIME_UPDATES_BASE_URL");
    expect(updateHost).toContain("LIME_ELECTRON_UPDATES_URL");
    expect(updateHost).toContain("LIME_ELECTRON_ENABLE_DEV_UPDATER");
    expect(updateHost).toContain("check_for_updates");
    expect(updateHost).toContain("download_update");
    expect(updateHost).toContain("start_update_install_session");

    const releaseWorkflow = readFile(".github/workflows/release.yml");
    expect(releaseWorkflow).toContain("Build Electron");
    expect(releaseWorkflow).toContain("platform: macos-15");
    expect(releaseWorkflow).toContain("platform: macos-15-intel");
    expect(releaseWorkflow).toContain("platform: windows-2022");
    expect(releaseWorkflow).toContain("npx electron-forge make");
    expect(releaseWorkflow).toContain("LIME_ELECTRON_UPDATES_URL");
    expect(releaseWorkflow).toContain(
      "Validate Electron macOS signing secrets",
    );
    expect(releaseWorkflow).toContain("LIME_ELECTRON_SIGN");
    expect(releaseWorkflow).toContain("LIME_MACOS_KEYCHAIN");
    expect(releaseWorkflow).toContain("APPLE_APP_SPECIFIC_PASSWORD");
    expect(releaseWorkflow).toContain("APPLE_SIGNING_IDENTITY");
    expect(releaseWorkflow).toContain("KEYCHAIN_PASSWORD");
    expect(releaseWorkflow).toContain(
      "Validate Electron Windows signing secrets",
    );
    expect(releaseWorkflow).toContain(
      "Prepare Electron Windows signing certificate",
    );
    expect(releaseWorkflow).toContain("WINDOWS_SIGNING_CERTIFICATE");
    expect(releaseWorkflow).toContain(
      "Forge Squirrel will produce unsigned installer assets",
    );
    expect(releaseWorkflow).toContain(
      "Incomplete Electron Windows signing secrets",
    );
    expect(releaseWorkflow).toContain("LIME_WINDOWS_SIGNING_CERTIFICATE_FILE");
    expect(releaseWorkflow).toContain(
      "Publish Electron updater assets to Cloudflare R2",
    );
    expect(releaseWorkflow).toContain(
      "scripts/electron/update-feed-r2-upload-plan.mjs",
    );
    expect(releaseWorkflow).toContain(
      "Legacy updater assets must not be published by the Electron release workflow",
    );

    const packageScripts = JSON.parse(readFile("package.json")).scripts;
    expect(packageScripts["governance:electron-release-workflow"]).toBe(
      "node scripts/electron/release-workflow-guard.mjs",
    );
    expect(packageScripts["test:contracts"]).toContain(
      "npm run governance:electron-release-workflow",
    );
    expect(packageScripts["electron:make:zip-local-feed"]).toBe(
      "node scripts/electron/make-zip-local-feed.mjs",
    );

    const githubReleaseAssets = readFile(
      "scripts/electron/prepare-github-release-assets.mjs",
    );
    expect(githubReleaseAssets).toContain("assertNoRetiredUpdaterAssets");
    expect(githubReleaseAssets).toContain(
      "legacy updater assets are not allowed in Electron GitHub release assets",
    );

    const uploadPlan = readFile(
      "scripts/electron/update-feed-r2-upload-plan.mjs",
    );
    expect(uploadPlan).toContain('"aarch64-apple-darwin": "darwin-arm64"');
    expect(uploadPlan).toContain('"x86_64-apple-darwin": "darwin-x64"');
    expect(uploadPlan).toContain('"x86_64-pc-windows-msvc": "win32-x64"');
    expect(uploadPlan).toContain("RELEASES.json");
    expect(uploadPlan).toContain("RELEASES");
    expect(uploadPlan).toContain("latest(?:-mac)?");
    expect(uploadPlan).toContain("legacy updater assets are not allowed");

    const localFeedMake = readFile("scripts/electron/make-zip-local-feed.mjs");
    expect(localFeedMake).toContain("electron-forge.js");
    expect(localFeedMake).toContain(".tmp");
    expect(localFeedMake).toContain("electron-forge-local-feed");
    expect(localFeedMake).toContain("LIME_ELECTRON_FORGE_OUT_DIR");
    expect(localFeedMake).toContain("LIME_ELECTRON_UPDATES_URL");
    expect(localFeedMake).toContain("RELEASES.json");
    expect(localFeedMake).toContain("--targets");
    expect(localFeedMake).toContain("zip");
    expect(localFeedMake).not.toContain("electron-builder");
    expect(localFeedMake).not.toContain("latest-mac.yml");

    const workflowGuard = readFile(
      "scripts/electron/release-workflow-guard.mjs",
    );
    const workflowMatrixGuard = readFile(
      "scripts/electron/lib/release-workflow-matrix-guard.mjs",
    );
    expect(workflowGuard).toContain("validateReleaseWorkflow");
    expect(workflowGuard).toContain("validateReleaseMatrix");
    expect(workflowGuard).toContain(
      "./lib/release-workflow-matrix-guard.mjs",
    );
    expect(workflowMatrixGuard).toContain("macos-15");
    expect(workflowMatrixGuard).toContain("macos-15-intel");
    expect(workflowMatrixGuard).toContain("windows-2022");
    expect(workflowGuard).toContain("electron-forge make");
    expect(workflowGuard).toContain("APPLE_CERTIFICATE");
    expect(workflowGuard).toContain("WINDOWS_SIGNING_CERTIFICATE");
    expect(workflowGuard).toContain(
      "scripts/electron/update-feed-r2-upload-plan.mjs",
    );

    const stageAssets = readFile("scripts/electron/stage-release-assets.mjs");
    expect(stageAssets).toContain("assertNoRetiredUpdaterAssets");
    expect(stageAssets).toContain("assertNoLocalMacUpdateManifest");
    expect(stageAssets).toContain(
      "local Electron updater feed URLs are not allowed in release staging",
    );
    expect(stageAssets).toContain(
      "legacy updater assets are not allowed in Electron release staging",
    );
  });

  it("keeps Electron frontend host docs as current contract, not a future migration", () => {
    const frontendHost = readFile(
      "internal/roadmap/appserver/frontend-electron-migration.md",
    );
    expect(frontendHost).toContain("Electron Desktop Host Current");
    expect(frontendHost).toContain("已经由 Electron 全面接管");
    expect(frontendHost).toContain("不是“后续切换”计划，而是 current 契约");
    expect(frontendHost).toContain("Frontend");
    expect(frontendHost).toContain("Electron Desktop Host bridge");
    expect(frontendHost).toContain("app_server_handle_json_lines");
    expect(frontendHost).toContain("App Server JSON-RPC");
    expect(frontendHost).toContain("Electron 只负责 Desktop Host bridge");
    expect(frontendHost).toContain("不是第二套后端");
    expect(frontendHost).toContain("不是 Agent runtime adapter");
    expect(frontendHost).toContain("Lime 不参考 Codex App UI 或桌面壳实现");
    expect(frontendHost).toContain("生产路径不能靠 mock 成功");
    expect(frontendHost).toContain("npm run verify:gui-smoke");
    expectNoRetiredCurrentHostReference(
      frontendHost,
      "internal/roadmap/appserver/frontend-electron-migration.md",
    );

    const roadmap = readFile("internal/roadmap/appserver/README.md");
    expect(roadmap).toContain("Electron Desktop Host current 契约");
    expect(roadmap).not.toContain("Lime 前端切换到 Electron Desktop Host");
  });

  it("keeps AgentUI docs on Electron and App Server command gateway", () => {
    const agentUiDocs = [
      "internal/roadmap/agentui/README.md",
      "internal/roadmap/agentui/lime-agentui-target-architecture.md",
      "internal/roadmap/agentui/lime-agentui-code-map.md",
      "internal/roadmap/agentui/lime-agentui-backend-coordination.md",
      "internal/roadmap/agentui/lime-agentui-implementation-roadmap.md",
      "internal/roadmap/agentui/conversation-projection-implementation-plan.md",
    ];

    for (const filePath of agentUiDocs) {
      const content = readFile(filePath);
      expect(content, filePath).toContain("App Server");
      expectNoLegacyAgentUiCommandGatewayReference(content, filePath);
    }

    const targetArchitecture = readFile(
      "internal/roadmap/agentui/lime-agentui-target-architecture.md",
    );
    expect(targetArchitecture).toContain(
      "Electron Desktop Host bridge / App Server JSON-RPC",
    );
    expect(targetArchitecture).toContain("CommandGateway --> RuntimeQueue");

    const codeMap = readFile(
      "internal/roadmap/agentui/lime-agentui-code-map.md",
    );
    expect(codeMap).toContain("## 5. Command Gateway 层");
    expect(codeMap).toContain("App Server");
    expect(codeMap).toContain("Electron Desktop Host");
    expectNoLegacyAgentUiCommandGatewayReference(
      codeMap,
      "internal/roadmap/agentui/lime-agentui-code-map.md",
    );
  });

  it("keeps i18n app metadata workflow on Electron Forge current sources", () => {
    const evaluation = readFile(
      "internal/roadmap/i18n/app-metadata-workflow-evaluation.md",
    );
    expect(evaluation).toContain("forge.config.mjs");
    expect(evaluation).toContain("当前 Electron 发布元数据事实源");
    expect(evaluation).toContain("Electron Forge / 平台发布链路");
    expect(evaluation).toContain("已按 `dead` release / metadata surface 下线");
    expect(evaluation).toContain(
      "不是 current app metadata、installer、release、updater、签名或版本同步事实源",
    );
    expect(evaluation).toContain("不能作为 i18n evidence 输入回流");
    expect(evaluation).not.toContain("Tauri file association");
    expect(evaluation).not.toContain("真实 Tauri");
    expect(evaluation).not.toContain("手工复制多份 Tauri 配置");

    expect(evaluation).toContain("9 个 metadata entry");
    expect(evaluation).toContain("`9` 个真实 app / installer metadata 字段");
    expect(evaluation).not.toContain("implementation-progress.md");
  });

  it("keeps App Server protocol aligned with codex-rs initialize and in-process boundaries", () => {
    const protocol = readFile("internal/roadmap/appserver/protocol.md");
    expect(protocol).toContain("JSON-RPC-like");
    expect(protocol).toContain('不要求也不发送 `"jsonrpc":"2.0"` header');
    expect(protocol).toContain("experimentalApi");
    expect(protocol).toContain("optOutNotificationMethods");
    expect(protocol).toContain("stable / experimental schema");
    expect(protocol).not.toContain("eventMethods");
    expect(protocol).not.toMatch(/"experimental"\s*:/);

    const consumerIntegration = readFile(
      "internal/roadmap/appserver/consumer-integration.md",
    );
    expect(consumerIntegration).toContain("experimentalApi");
    expect(consumerIntegration).toContain("optOutNotificationMethods");
    expect(consumerIntegration).not.toContain("eventMethods");

    const architecture = readFile("internal/roadmap/appserver/architecture.md");
    expect(architecture).toContain("不引入第二响应合同");
    expect(architecture).toContain("App Server JSON-RPC result envelope");

    const frontendMatrix = readFile(
      "internal/roadmap/appserver/frontend-integration-matrix.md",
    );
    expect(frontendMatrix).toContain("不得引入第二响应合同");
    expect(frontendMatrix).toContain("不得引入第二响应合同、第二 read model");
  });
});
