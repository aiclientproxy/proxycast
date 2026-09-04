#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import process from "node:process";

import { withNativeSystemPath } from "./lib/native-executable-env.mjs";

const DEFAULT_GIT_COMMAND = process.platform === "win32" ? "git.exe" : "git";

const IGNORED_PREFIXES = [
  ".turbo/",
  "coverage/",
  "dist/",
  "docs/.output/",
  "node_modules/",
  "target/",
  "target-site-e2e/",
];

const IGNORED_FILES = new Set([".DS_Store"]);

const LOW_RISK_WORKFLOW_FILES = new Set([
  ".github/workflows/build-windows-test.yml",
]);

const FRONTEND_ROOT_FILES = new Set([
  "package.json",
  "package-lock.json",
  "vite.config.ts",
  "tsconfig.json",
  "tsconfig.node.json",
  "eslint.config.js",
  "tailwind.config.js",
  "postcss.config.js",
  "index.html",
]);

const FRONTEND_TOOLING_FILES = new Set([
  "scripts/local-ci.mjs",
  "scripts/ai-code-verify.ts",
  "scripts/quality-task-planner.mjs",
  "scripts/quality-task-selector.mjs",
]);

const I18N_CHECK_FILES = new Set([
  "package.json",
  "scripts/i18n/detect-missing-translations.ts",
  "scripts/i18n/detect-missing-translations.test.ts",
  "scripts/local-ci.mjs",
  "scripts/quality-task-planner.mjs",
  "scripts/quality-task-selector.mjs",
  "src/i18n/loadNamespace.ts",
  "src/i18n/locales.ts",
  "src/i18n/types.d.ts",
]);

const I18N_CHECK_PREFIXES = [
  "scripts/i18n/",
  "src/i18n/resources/",
  "src/i18n/__tests__/",
];

const I18N_TRANSLATION_REVIEW_PACK_RECOMMENDED_COMMANDS = [
  "npm run i18n:translation-pr-pack:json -- --output internal/roadmap/i18n/evidence/translation-pr-pack.json",
];

const I18N_BUNDLE_STRATEGY_RECOMMENDED_COMMANDS = [
  "npm run i18n:bundle-report:json -- --output internal/roadmap/i18n/evidence/bundle-strategy-report.json",
];

const I18N_BUNDLE_STRATEGY_FILES = new Set([
  "internal/roadmap/i18n/evidence/bundle-strategy-report.json",
  "scripts/i18n/i18n-bundle-report.test.ts",
  "scripts/i18n/i18n-bundle-report.ts",
  "src/i18n/bundledNamespaceParts.ts",
  "src/i18n/loadNamespace.ts",
]);

const I18N_PATCH_RETIREMENT_RECOMMENDED_COMMANDS = [
  "npm run i18n:patch-retirement-gate -- --check",
];

const I18N_RELEASE_DOCS_WORKFLOW_RECOMMENDED_COMMANDS = [
  "npm run i18n:docs-locale-manifest:json -- --output internal/roadmap/i18n/evidence/docs-locale-build-manifest.json",
  "npm run i18n:release-docs-report:json -- --output internal/roadmap/i18n/evidence/release-docs-workflow-inventory.json",
];

const I18N_CHROME_EXTENSION_WORKFLOW_RECOMMENDED_COMMANDS = [
  "npm run i18n:chrome-extension-report:json -- --output internal/roadmap/i18n/evidence/chrome-extension-workflow-inventory.json",
];

const I18N_APP_METADATA_WORKFLOW_RECOMMENDED_COMMANDS = [
  "npm run i18n:app-metadata-locale-manifest:json -- --output internal/roadmap/i18n/evidence/app-metadata-locale-build-manifest.json",
  "npm run i18n:app-metadata-report:json -- --output internal/roadmap/i18n/evidence/app-metadata-workflow-inventory.json",
];

const I18N_RTL_READINESS_RECOMMENDED_COMMANDS = [
  "npm run i18n:rtl-readiness-report:json -- --output internal/roadmap/i18n/evidence/rtl-readiness-inventory.json",
];

const I18N_RTL_SMOKE_RECOMMENDED_COMMANDS = ["npm run i18n:rtl-smoke"];

const I18N_P4_READINESS_RECOMMENDED_COMMANDS = [
  "npm run i18n:p4-readiness-report:json -- --output internal/roadmap/i18n/evidence/p4-readiness-report.json",
];

const I18N_ROADMAP_READINESS_RECOMMENDED_COMMANDS = [
  "npm run i18n:roadmap-readiness-report:json -- --output internal/roadmap/i18n/evidence/roadmap-readiness-report.json",
];

const I18N_PATCH_RETIREMENT_FILES = new Set([
  "scripts/i18n/i18n-patch-metrics-report.mjs",
  "scripts/i18n/i18n-patch-retirement-gate.mjs",
  "scripts/lib/i18n-patch-metrics-report-core.mjs",
  "scripts/lib/legacy-surface-report-core.mjs",
  "scripts/report-legacy-surfaces.mjs",
]);

const I18N_RELEASE_DOCS_WORKFLOW_FILES = new Set([
  "README.md",
  "README.en.md",
  "RELEASE_NOTES.md",
  "RELEASE_NOTES.en.md",
  "docs/README.md",
  "docs/nuxt.config.ts",
  "docs/package.json",
  "internal/roadmap/i18n/evidence/docs-locale-build-manifest.json",
  "internal/roadmap/i18n/evidence/release-docs-workflow-inventory.json",
  "internal/roadmap/i18n/release-docs-translation-scope.json",
  "internal/roadmap/i18n/release-docs-workflow-evaluation.md",
  "scripts/i18n/i18n-docs-locale-build-manifest.test.ts",
  "scripts/i18n/i18n-docs-locale-build-manifest.ts",
  "scripts/i18n/i18n-release-docs-workflow-report.test.ts",
  "scripts/i18n/i18n-release-docs-workflow-report.ts",
]);

const I18N_RELEASE_DOCS_WORKFLOW_PREFIXES = [
  "internal/bussniss/",
  "docs/content/",
  "internal/develop/",
  "internal/oem/",
  "internal/roadmap/i18n/companions/",
];

const I18N_CHROME_EXTENSION_WORKFLOW_FILES = new Set([
  "internal/roadmap/i18n/chrome-extension-evaluation.md",
  "internal/roadmap/i18n/evidence/chrome-extension-workflow-inventory.json",
  "extensions/lime-chrome/CHROME_WEB_STORE_SUBMISSION.md",
  "extensions/lime-chrome/README.md",
  "extensions/lime-chrome/manifest.json",
  "extensions/lime-chrome/pages/scripts/install-i18n.js",
  "extensions/lime-chrome/pages/scripts/options.js",
  "scripts/i18n/i18n-chrome-extension-workflow-report.test.ts",
  "scripts/i18n/i18n-chrome-extension-workflow-report.ts",
]);

const I18N_CHROME_EXTENSION_WORKFLOW_PREFIXES = [
  "extensions/lime-chrome/pages/",
];

const I18N_APP_METADATA_WORKFLOW_FILES = new Set([
  "internal/roadmap/i18n/app-metadata-translation-scope.json",
  "internal/roadmap/i18n/app-metadata-workflow-evaluation.md",
  "internal/roadmap/i18n/evidence/app-metadata-locale-build-manifest.json",
  "internal/roadmap/i18n/evidence/app-metadata-workflow-inventory.json",
  "package.json",
  "scripts/i18n/i18n-app-metadata-locale-build-manifest.test.ts",
  "scripts/i18n/i18n-app-metadata-locale-build-manifest.ts",
  "scripts/i18n/i18n-app-metadata-workflow-report.test.ts",
  "scripts/i18n/i18n-app-metadata-workflow-report.ts",
  "lime-rs/Cargo.toml",
  "forge.config.mjs",
]);

const I18N_RTL_READINESS_FILES = new Set([
  "internal/roadmap/i18n/evidence/rtl-readiness-inventory.json",
  "internal/roadmap/i18n/rtl-readiness-evaluation.md",
  "scripts/i18n/i18n-rtl-readiness-report.test.ts",
  "scripts/i18n/i18n-rtl-readiness-report.ts",
  "src/i18n/createI18n.ts",
  "src/i18n/locales.ts",
]);

const I18N_RTL_SMOKE_FILES = new Set([
  "scripts/i18n/i18n-rtl-playwright-smoke.mjs",
]);

const I18N_P4_READINESS_FILES = new Set([
  "internal/roadmap/i18n/evidence/p4-readiness-report.json",
  "internal/roadmap/i18n/prd.md",
  "scripts/i18n/i18n-p4-readiness-report.test.ts",
  "scripts/i18n/i18n-p4-readiness-report.ts",
]);

const I18N_ROADMAP_READINESS_FILES = new Set([
  "internal/roadmap/i18n/evidence/roadmap-readiness-report.json",
  "internal/roadmap/i18n/prd.md",
  "scripts/i18n/i18n-roadmap-readiness-report.test.ts",
  "scripts/i18n/i18n-roadmap-readiness-report.ts",
]);

const I18N_RTL_SURFACE_FILES = new Set([
  "src/App.tsx",
  "src/components/AppSidebar.tsx",
  "src/components/Modal.tsx",
  "src/components/agent/chat/workspace/WorkspaceConversationScene.tsx",
  "src/components/agent/chat/workspace/WorkspaceGeneralWorkbenchSidebar.tsx",
  "src/components/agent/chat/workspace/WorkspaceHarnessDialogs.tsx",
  "src/components/agent/chat/workspace/WorkspaceMainArea.tsx",
  "src/components/agent/chat/workspace/WorkspaceShellScene.tsx",
  "src/components/api-key-provider/ImportExportDialog.tsx",
  "src/components/app-sidebar/AppSidebarConversationShelf.tsx",
  "src/components/channels/ImConfigPage.tsx",
  "src/components/connect/ConnectConfirmDialog.tsx",
  "src/components/settings-v2/_layout/SettingsSidebar.tsx",
  "src/components/settings-v2/_layout/index.tsx",
  "src/components/settings-v2/agent/providers/index.tsx",
  "src/components/settings-v2/general/appearance/index.tsx",
  "src/components/settings-v2/general/memory/index.tsx",
  "src/components/settings-v2/system/about/index.tsx",
  "src/components/settings-v2/system/automation/index.tsx",
  "src/components/settings-v2/system/channels/ChannelLogTailPanel.tsx",
  "src/components/settings-v2/system/channels/ChannelsDebugWorkbench.tsx",
  "src/components/settings-v2/system/developer/index.tsx",
  "src/components/settings-v2/system/environment/index.tsx",
  "src/components/settings-v2/system/experimental/index.tsx",
  "src/components/settings-v2/system/web-search/index.tsx",
  "src/components/skills/SkillScaffoldDialog.tsx",
  "src/components/workspace/canvas/shared/CanvasBreadcrumbHeader.tsx",
  "src/components/workspace/document/DocumentToolbar.tsx",
  "src/components/workspace/video/VideoSidebar.tsx",
  "src/features/knowledge/KnowledgePage.tsx",
]);

const I18N_HARDCODED_SCAN_PREFIXES = [
  "src/components/",
  "src/features/",
  "src/pages/",
];

const BRIDGE_FILES = new Set([
  "vite.config.ts",
  "electron/appServerHost.ts",
  "electron/hostCommands.ts",
  "electron/ipcChannels.ts",
  "electron/main.ts",
  "electron/preload.ts",
  "scripts/check-command-contracts.mjs",
  "scripts/check-generated-slop-report.mjs",
  "scripts/check-dev-bridge-health.mjs",
  "scripts/harness/eval-history-record.mjs",
  "scripts/harness/eval-trend-report.mjs",
  "scripts/report-generated-slop.mjs",
  "scripts/social-workbench-e2e-smoke.mjs",
  "scripts/chrome-bridge-e2e.mjs",
  "scripts/lib/generated-slop-report-core.mjs",
  "scripts/lib/harness-dashboard-core.mjs",
  "internal/aiprompts/playwright-e2e.md",
]);

const HARNESS_CLEANUP_CONTRACT_FILES = new Set([
  "scripts/check-generated-slop-report.mjs",
  "scripts/harness/eval-history-record.mjs",
  "scripts/harness/eval-trend-report.mjs",
  "scripts/report-generated-slop.mjs",
  "scripts/lib/generated-slop-report-core.mjs",
  "scripts/lib/harness-dashboard-core.mjs",
]);

const INTEGRITY_FILES = new Set([
  "package.json",
  "packages/cli/package.json",
  "lime-rs/Cargo.toml",
  "forge.config.mjs",
  "scripts/check-app-version-consistency.mjs",
  "scripts/quality-task-planner.mjs",
  "scripts/quality-task-selector.mjs",
]);

const GUI_SMOKE_FILES = new Set([
  "src/App.tsx",
  "src/main.tsx",
  "electron/appServerHost.ts",
  "electron/hostCommands.ts",
  "electron/ipcChannels.ts",
  "electron/main.ts",
  "electron/preload.ts",
  "scripts/electron/build-renderer.mjs",
  "scripts/electron/build-host.mjs",
  "scripts/check-dev-bridge-health.mjs",
  "scripts/electron/copy-desktop-assets.mjs",
  "scripts/electron/smoke.mjs",
  "scripts/electron/run-dev.mjs",
  "scripts/electron/run-preview.mjs",
  "src/lib/electron-host.ts",
  "tsconfig.electron.json",
]);

const GUI_SMOKE_PREFIXES = [
  "electron/",
  "src/components/",
  "src/contexts/",
  "src/features/",
  "src/hooks/",
  "src/lib/dev-bridge/",
  "src/lib/navigation/",
  "src/lib/desktop-host/",
  "src/lib/workspace/",
  "src/pages/",
  "src/stores/",
];

const KNOWLEDGE_PRODUCT_E2E_FILES = new Set([
  "scripts/knowledge-gui-smoke.mjs",
  "scripts/knowledge-product-e2e.mjs",
  "src/components/agent/chat/AgentChatWorkspace.tsx",
]);

const KNOWLEDGE_PRODUCT_E2E_PREFIXES = [
  "src/features/knowledge/",
  "src/components/agent/chat/workspace/knowledge/",
];

const KNOWLEDGE_PRODUCT_E2E_RECOMMENDED_COMMANDS = [
  "npm run knowledge:product-e2e",
  "npm run verify:gui-smoke -- --include-knowledge-product-e2e --reuse-running",
];

function gitOutput({ cwd, gitCommand, args }) {
  try {
    return execFileSync(gitCommand, args, {
      cwd,
      encoding: "utf8",
      env: withNativeSystemPath(process.env),
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return "";
  }
}

function splitLines(value) {
  if (!value) {
    return [];
  }

  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

function uniquePaths(paths) {
  return Array.from(new Set(paths)).filter((file) => !isIgnoredPath(file));
}

function isIgnoredPath(file) {
  if (IGNORED_FILES.has(file)) {
    return true;
  }

  return IGNORED_PREFIXES.some((prefix) => file.startsWith(prefix));
}

function resolveDiffBase({ base = "", cwd, gitCommand = DEFAULT_GIT_COMMAND }) {
  if (base) {
    return base;
  }

  const upstream = gitOutput({
    cwd,
    gitCommand,
    args: ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
  });
  if (upstream) {
    return upstream;
  }

  for (const candidate of ["origin/main", "origin/master", "main", "master"]) {
    const exists = gitOutput({
      cwd,
      gitCommand,
      args: ["rev-parse", "--verify", candidate],
    });
    if (exists) {
      return candidate;
    }
  }

  return "";
}

function collectChangedFiles({
  full = false,
  staged = false,
  base = "",
  cwd = process.cwd(),
  gitCommand = DEFAULT_GIT_COMMAND,
} = {}) {
  if (full) {
    return [];
  }

  if (staged) {
    return uniquePaths(
      splitLines(
        gitOutput({
          cwd,
          gitCommand,
          args: ["diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        }),
      ),
    );
  }

  const diffBase = resolveDiffBase({ base, cwd, gitCommand });
  const candidates = [];

  if (diffBase) {
    candidates.push(
      ...splitLines(
        gitOutput({
          cwd,
          gitCommand,
          args: [
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            `${diffBase}...HEAD`,
          ],
        }),
      ),
    );
  }

  candidates.push(
    ...splitLines(
      gitOutput({
        cwd,
        gitCommand,
        args: ["diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
      }),
    ),
  );
  candidates.push(
    ...splitLines(
      gitOutput({
        cwd,
        gitCommand,
        args: ["ls-files", "--others", "--exclude-standard"],
      }),
    ),
  );

  return uniquePaths(candidates);
}

function isMarkdownLike(file) {
  return /\.(md|mdx)$/i.test(file);
}

function isWorkflowChange(file) {
  return (
    file.startsWith(".github/workflows/") && !LOW_RISK_WORKFLOW_FILES.has(file)
  );
}

function isDocsChange(file) {
  return (
    file.startsWith("docs/") ||
    file.startsWith("internal/tests/") ||
    file.startsWith("internal/test/agent-qc-") ||
    file.startsWith("internal/roadmap/i18n/") ||
    isMarkdownLike(file)
  );
}

function isDocsOnlyChange(files) {
  return files.length > 0 && files.every((file) => isDocsChange(file));
}

function isFrontendChange(file) {
  return (
    file.startsWith("src/") ||
    FRONTEND_ROOT_FILES.has(file) ||
    FRONTEND_TOOLING_FILES.has(file)
  );
}

function isI18nCheckChange(file) {
  return (
    I18N_CHECK_FILES.has(file) ||
    I18N_CHECK_PREFIXES.some((prefix) => file.startsWith(prefix))
  );
}

function isI18nTranslationReviewPackChange(file) {
  return (
    file.startsWith("src/i18n/resources/") ||
    file === "scripts/i18n/detect-missing-translations.ts" ||
    file === "scripts/i18n/i18n-source-locale-export.ts" ||
    file === "scripts/i18n/i18n-translation-pr-pack.ts"
  );
}

function isI18nBundleStrategyChange(file) {
  return (
    file.startsWith("src/i18n/resources/") ||
    I18N_BUNDLE_STRATEGY_FILES.has(file)
  );
}

function isI18nPatchRetirementChange(file) {
  return I18N_PATCH_RETIREMENT_FILES.has(file);
}

function isI18nReleaseDocsWorkflowChange(file) {
  return (
    I18N_RELEASE_DOCS_WORKFLOW_FILES.has(file) ||
    I18N_RELEASE_DOCS_WORKFLOW_PREFIXES.some((prefix) =>
      file.startsWith(prefix),
    )
  );
}

function isI18nChromeExtensionWorkflowChange(file) {
  return (
    I18N_CHROME_EXTENSION_WORKFLOW_FILES.has(file) ||
    I18N_CHROME_EXTENSION_WORKFLOW_PREFIXES.some((prefix) =>
      file.startsWith(prefix),
    )
  );
}

function isI18nAppMetadataWorkflowChange(file) {
  return I18N_APP_METADATA_WORKFLOW_FILES.has(file);
}

function isI18nRtlReadinessChange(file) {
  return (
    I18N_RTL_READINESS_FILES.has(file) ||
    I18N_RTL_SMOKE_FILES.has(file) ||
    I18N_RTL_SURFACE_FILES.has(file)
  );
}

function isI18nRtlSmokeChange(file) {
  return I18N_RTL_SMOKE_FILES.has(file) || I18N_RTL_SURFACE_FILES.has(file);
}

function isI18nP4ReadinessChange(file) {
  return (
    I18N_P4_READINESS_FILES.has(file) ||
    isI18nReleaseDocsWorkflowChange(file) ||
    isI18nChromeExtensionWorkflowChange(file) ||
    isI18nAppMetadataWorkflowChange(file) ||
    isI18nRtlReadinessChange(file)
  );
}

function isI18nRoadmapReadinessChange(file) {
  return (
    I18N_ROADMAP_READINESS_FILES.has(file) ||
    isI18nTranslationReviewPackChange(file) ||
    isI18nBundleStrategyChange(file) ||
    isI18nPatchRetirementChange(file) ||
    isI18nP4ReadinessChange(file) ||
    file ===
      "internal/roadmap/i18n/evidence/translation-coverage-report.json" ||
    file === "internal/roadmap/i18n/evidence/source-locale-export.json" ||
    file === "internal/roadmap/i18n/evidence/language-boundary-report.json" ||
    file ===
      "internal/roadmap/i18n/evidence/content-target-language-boundary-report.json" ||
    file ===
      "internal/roadmap/i18n/evidence/i18next-cli-parity-benchmark.json" ||
    file === "internal/roadmap/i18n/glossary.md" ||
    file === "internal/roadmap/i18n/implementation-progress.md" ||
    file === "internal/roadmap/i18n/language-boundary-evaluation.md" ||
    file ===
      "internal/roadmap/i18n/response-language-injection-evaluation.md" ||
    file === "internal/roadmap/i18n/toolchain-evaluation.md"
  );
}

function isI18nHardcodedScanChange(file) {
  return (
    file === "src/App.tsx" ||
    file === "src/main.tsx" ||
    I18N_HARDCODED_SCAN_PREFIXES.some((prefix) => file.startsWith(prefix))
  );
}

function isRustChange(file) {
  return file.startsWith("lime-rs/");
}

function isBridgeChange(file) {
  return (
    file.startsWith("src/lib/dev-bridge/") ||
    file.startsWith("src/lib/desktop-host/") ||
    BRIDGE_FILES.has(file)
  );
}

function isHarnessCleanupContractChange(file) {
  return HARNESS_CLEANUP_CONTRACT_FILES.has(file);
}

function collectBridgeReasons(
  changedFiles,
  { full = false, fallback = false, workflow = false } = {},
) {
  if (full) {
    return ["full_suite"];
  }

  if (fallback) {
    return ["fallback_full_suite"];
  }

  if (workflow) {
    return ["workflow_full_suite"];
  }

  const reasons = [];

  if (changedFiles.some(isHarnessCleanupContractChange)) {
    reasons.push("harness_cleanup_contract");
  }

  if (
    changedFiles.some(
      (file) => isBridgeChange(file) && !isHarnessCleanupContractChange(file),
    )
  ) {
    reasons.push("bridge_runtime");
  }

  if (reasons.length === 0 && changedFiles.some(isBridgeChange)) {
    reasons.push("bridge_contracts");
  }

  return reasons;
}

function isGuiSmokeChange(file) {
  return (
    GUI_SMOKE_FILES.has(file) ||
    GUI_SMOKE_PREFIXES.some((prefix) => file.startsWith(prefix)) ||
    isI18nPatchRetirementChange(file)
  );
}

function isKnowledgeProductE2eChange(file) {
  return (
    KNOWLEDGE_PRODUCT_E2E_FILES.has(file) ||
    KNOWLEDGE_PRODUCT_E2E_PREFIXES.some((prefix) => file.startsWith(prefix))
  );
}

function collectRecommendedCommands(changedFiles, { docsOnly = false } = {}) {
  if (docsOnly) {
    return [];
  }

  const commands = [];
  if (changedFiles.some(isI18nPatchRetirementChange)) {
    commands.push(...I18N_PATCH_RETIREMENT_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isKnowledgeProductE2eChange)) {
    commands.push(...KNOWLEDGE_PRODUCT_E2E_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nTranslationReviewPackChange)) {
    commands.push(...I18N_TRANSLATION_REVIEW_PACK_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nBundleStrategyChange)) {
    commands.push(...I18N_BUNDLE_STRATEGY_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nReleaseDocsWorkflowChange)) {
    commands.push(...I18N_RELEASE_DOCS_WORKFLOW_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nChromeExtensionWorkflowChange)) {
    commands.push(...I18N_CHROME_EXTENSION_WORKFLOW_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nAppMetadataWorkflowChange)) {
    commands.push(...I18N_APP_METADATA_WORKFLOW_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nRtlReadinessChange)) {
    commands.push(...I18N_RTL_READINESS_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nRtlSmokeChange)) {
    commands.push(...I18N_RTL_SMOKE_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nP4ReadinessChange)) {
    commands.push(...I18N_P4_READINESS_RECOMMENDED_COMMANDS);
  }

  if (changedFiles.some(isI18nRoadmapReadinessChange)) {
    commands.push(...I18N_ROADMAP_READINESS_RECOMMENDED_COMMANDS);
  }

  return Array.from(new Set(commands));
}

function isIntegrityChange(file) {
  return (
    INTEGRITY_FILES.has(file) ||
    isWorkflowChange(file) ||
    isFrontendChange(file) ||
    isRustChange(file)
  );
}

function detectTasks(changedFiles, { full = false } = {}) {
  const recommendedCommands = collectRecommendedCommands(changedFiles, {
    docsOnly: false,
  });

  if (full) {
    return {
      integrity: true,
      i18n: true,
      i18nHardcoded: true,
      i18nUnused: true,
      frontend: true,
      rust: true,
      bridge: true,
      bridgeReasons: collectBridgeReasons([], { full: true }),
      guiSmoke: true,
      docs: true,
      docsOnly: false,
      fallback: false,
      recommendedCommands,
      workflow: false,
    };
  }

  if (changedFiles.length === 0) {
    return {
      integrity: true,
      i18n: true,
      i18nHardcoded: true,
      i18nUnused: true,
      frontend: true,
      rust: true,
      bridge: true,
      bridgeReasons: collectBridgeReasons([], { fallback: true }),
      guiSmoke: true,
      docs: true,
      docsOnly: false,
      fallback: true,
      recommendedCommands,
      workflow: false,
    };
  }

  const workflow = changedFiles.some(isWorkflowChange);
  if (workflow) {
    return {
      integrity: true,
      i18n: true,
      i18nHardcoded: true,
      i18nUnused: true,
      frontend: true,
      rust: true,
      bridge: true,
      bridgeReasons: collectBridgeReasons([], { workflow: true }),
      guiSmoke: true,
      docs: true,
      docsOnly: false,
      fallback: false,
      recommendedCommands,
      workflow: true,
    };
  }

  if (isDocsOnlyChange(changedFiles)) {
    return {
      integrity: false,
      i18n: false,
      i18nHardcoded: false,
      i18nUnused: false,
      frontend: false,
      rust: false,
      bridge: false,
      bridgeReasons: [],
      guiSmoke: false,
      docs: true,
      docsOnly: true,
      fallback: false,
      recommendedCommands,
      workflow: false,
    };
  }

  const bridge = changedFiles.some(isBridgeChange);
  const docsOnly = false;

  return {
    integrity: changedFiles.some(isIntegrityChange),
    i18n: changedFiles.some(isI18nCheckChange),
    i18nHardcoded: changedFiles.some(isI18nHardcodedScanChange),
    i18nUnused:
      changedFiles.some(isI18nCheckChange) ||
      changedFiles.some(isFrontendChange),
    frontend: changedFiles.some(isFrontendChange),
    rust: changedFiles.some(isRustChange),
    bridge,
    bridgeReasons: bridge ? collectBridgeReasons(changedFiles) : [],
    guiSmoke: changedFiles.some(isGuiSmokeChange),
    docs: changedFiles.some(isDocsChange),
    docsOnly,
    fallback: false,
    recommendedCommands,
    workflow: false,
  };
}

function planQualityTasks({
  full = false,
  staged = false,
  base = "",
  cwd = process.cwd(),
  gitCommand = DEFAULT_GIT_COMMAND,
} = {}) {
  const changedFiles = collectChangedFiles({
    full,
    staged,
    base,
    cwd,
    gitCommand,
  });

  const tasks = detectTasks(changedFiles, { full });

  return {
    changedFiles,
    tasks,
  };
}

export {
  collectBridgeReasons,
  collectChangedFiles,
  collectRecommendedCommands,
  detectTasks,
  planQualityTasks,
  resolveDiffBase,
};
