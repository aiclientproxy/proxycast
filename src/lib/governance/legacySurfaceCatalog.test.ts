import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

import agentCommandCatalog from "./agentCommandCatalog.json";
import legacySurfaceCatalogJson from "./legacySurfaceCatalog.json";

const REPO_ROOT = process.cwd();
const AGENT_UI_PACKAGE_SOURCE_ROOTS = [
  "packages/agent-ui-contracts/src",
  "packages/agent-runtime-projection/src",
  "packages/agent-runtime-ui/src",
  "packages/agent-runtime-client/src",
];
const SOURCE_EXTENSIONS = new Set([".ts", ".tsx", ".js", ".jsx", ".mjs"]);

function collectSourceFiles(root: string): string[] {
  const absoluteRoot = join(REPO_ROOT, root);
  const entries = readdirSync(absoluteRoot);
  return entries.flatMap((entry) => {
    const absolutePath = join(absoluteRoot, entry);
    const stats = statSync(absolutePath);
    if (stats.isDirectory()) {
      return collectSourceFiles(relative(REPO_ROOT, absolutePath));
    }
    const dotIndex = entry.lastIndexOf(".");
    const extension = dotIndex >= 0 ? entry.slice(dotIndex) : "";
    return SOURCE_EXTENSIONS.has(extension)
      ? [relative(REPO_ROOT, absolutePath)]
      : [];
  });
}

describe("legacySurfaceCatalog", () => {
  it("应提供完整且无重复的治理扫描目录册", () => {
    const catalog = legacySurfaceCatalogJson;
    const groups = [
      catalog.imports,
      catalog.commands,
      catalog.frontendText,
      catalog.rustText,
      catalog.rustTextCounts,
    ];

    expect(groups.every(Array.isArray)).toBe(true);
    expect(catalog.imports.length).toBeGreaterThan(0);
    expect(catalog.commands.length).toBeGreaterThan(0);
    expect(catalog.frontendText.length).toBeGreaterThan(0);
    expect(catalog.rustText.length).toBeGreaterThan(0);
    expect(catalog.rustTextCounts.length).toBeGreaterThan(0);

    const ids = groups.flat().map((monitor) => monitor.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("命令目录册不应继续携带 legacy surface 扫描数据", () => {
    expect("legacyCommandSurfaceMonitors" in agentCommandCatalog).toBe(false);
    expect("legacyHelperSurfaceMonitors" in agentCommandCatalog).toBe(false);
  });

  it("应阻止已删除的 agentSession action replay RPC 回流", () => {
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "frontend-retired-agent-session-action-replay",
    );
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-agent-session-action-replay",
    );

    expect(frontendMonitor?.classification).toBe("dead");
    expect(frontendMonitor?.allowedPaths).toEqual([]);
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "agentSession/action/replay",
        "APP_SERVER_METHOD_AGENT_SESSION_ACTION_REPLAY",
        "replayAction",
      ]),
    );
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "agentSession/action/replay",
        "METHOD_AGENT_SESSION_ACTION_REPLAY",
        "AgentSessionActionReplayParams",
        "replay_action(",
      ]),
    );
  });

  it("应阻止第一批已删除前端空壳和重复 UI surface 回流", () => {
    const expectedTargetsById = {
      "frontend-image-search-retired-facade-surface": [
        "src/lib/api/imageSearch.ts",
        "src/lib/api/imageSearch.test.ts",
        "src/lib/api/imageSearch.diagnostic.test.ts",
      ],
      "frontend-open-url-redundant-helper-surface": [
        "src/lib/openUrl.ts",
        "src/lib/openUrl.test.ts",
      ],
      "sceneapp-context-layer-directory-surface": [
        "src/lib/context-layer/index.ts",
        "src/lib/context-layer/types.ts",
      ],
      "compact-right-panel-legacy-shell-surface": [
        "src/lib/compactRightPanelEvents.ts",
        "src/components/ui/compact-right-dock-button.tsx",
        "src/components/ui/compact-right-drawer-header.tsx",
      ],
      "agent-chat-generic-artifact-frame-shell-surface": [
        "src/components/agent/chat/components/ArtifactFrame.tsx",
      ],
    } as const;

    for (const [id, targets] of Object.entries(expectedTargetsById)) {
      const monitor = legacySurfaceCatalogJson.imports.find(
        (entry) => entry.id === id,
      );

      expect(monitor).toBeTruthy();
      expect(monitor?.classification).toBe("dead");
      expect(monitor?.allowedPaths).toEqual([]);
      expect(monitor?.targets).toEqual(targets);
    }
  });

  it("应阻止第二批 parser、静态 Provider、Voice 与 Relay surface 回流", () => {
    const expectedTargetsById = {
      "artifact-fenced-response-parser-surface": [
        "src/lib/artifact/parser.ts",
        "src/lib/artifact/parser.test.ts",
      ],
      "static-system-provider-config-surface": [
        "src/lib/config/README.md",
        "src/lib/config/providers.ts",
        "src/lib/config/providers.test.ts",
      ],
      "voice-unused-preview-device-ui-surface": [
        "src/lib/voiceLivePreview.ts",
        "src/lib/voiceLivePreview.test.ts",
        "src/components/voice/MicrophoneTest.tsx",
        "src/components/voice/VolumeWaveform.tsx",
      ],
      "connect-relay-registry-disabled-hook-surface": [
        "src/hooks/useRelayRegistry.ts",
      ],
    } as const;

    for (const [id, targets] of Object.entries(expectedTargetsById)) {
      const monitor = legacySurfaceCatalogJson.imports.find(
        (entry) => entry.id === id,
      );

      expect(monitor).toBeTruthy();
      expect(monitor?.classification).toBe("dead");
      expect(monitor?.allowedPaths).toEqual([]);
      expect(monitor?.targets).toEqual(targets);
    }

    const audioDeviceMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "voice-audio-device-list-helper-surface",
    );
    expect(audioDeviceMonitor?.classification).toBe("dead");
    expect(audioDeviceMonitor?.includePathPrefixes).toEqual([
      "src/lib/api/asrProvider.ts",
    ]);
    expect(audioDeviceMonitor?.patterns).toEqual(
      expect.arrayContaining(["listAudioDevices", "AudioDeviceInfo"]),
    );

    const relayCopyMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "connect-relay-registry-legacy-patch-copy",
    );
    expect(relayCopyMonitor?.classification).toBe("dead");
    expect(relayCopyMonitor?.includePathPrefixes).toEqual([
      "src/i18n/legacy-patch",
    ]);
    expect(relayCopyMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "hooks\\\\useRelayRegistry.ts",
        "加载中转商列表失败",
      ]),
    );
  });

  it("应阻止第三批 Hotkeys 页面、Shortcut 编辑器与 Renderer facade 回流", () => {
    const expectedTargetsById = {
      "settings-hotkeys-retired-page-surface": [
        "src/components/settings-v2/general/hotkeys/index.tsx",
        "src/components/settings-v2/general/hotkeys/index.test.tsx",
        "src/components/settings-v2/general/hotkeys/hotkeyCatalog.ts",
        "src/components/settings-v2/general/hotkeys/hotkeyCatalog.test.ts",
      ],
      "settings-shortcut-editor-retired-surface": [
        "src/components/settings-v2/shared/ShortcutSettings.tsx",
        "src/components/settings-v2/shared/ShortcutSettings.test.tsx",
      ],
      "renderer-hotkey-voice-shortcut-facade-surface": [
        "src/lib/api/hotkeys.ts",
        "src/lib/api/hotkeys.test.ts",
        "src/lib/api/voiceShortcutEvents.ts",
        "src/lib/api/voiceShortcutEvents.test.ts",
      ],
    } as const;

    for (const [id, targets] of Object.entries(expectedTargetsById)) {
      const monitor = legacySurfaceCatalogJson.imports.find(
        (entry) => entry.id === id,
      );

      expect(monitor).toBeTruthy();
      expect(monitor?.classification).toBe("dead");
      expect(monitor?.allowedPaths).toEqual([]);
      expect(monitor?.targets).toEqual(targets);
    }

    const commandMonitor = legacySurfaceCatalogJson.commands.find(
      (entry) => entry.id === "voice-shortcut-retired-host-commands",
    );
    expect(commandMonitor?.classification).toBe("dead");
    expect(commandMonitor?.commands).toEqual([
      "get_voice_shortcut_runtime_status",
      "validate_shortcut",
    ]);
    expect(commandMonitor?.allowedPaths).toEqual([
      "src/lib/desktop-host/voiceMocks.test.ts",
      "src/lib/desktop-host/core.unhandled-mock.test.ts",
    ]);
  });

  it("应阻止第四批 Agent UI 与 article orchestration dead surface 回流", () => {
    const expectedTargetsById = {
      "agent-ui-subagents-view-model-surface": [
        "src/components/agent/chat/projection/agentUiSubagentsViewModel.ts",
        "src/components/agent/chat/projection/agentUiSubagentsViewModel.test.ts",
      ],
      "workspace-article-workflow-read-model-surface": [
        "src/components/agent/chat/workspace/useWorkspaceArticleWorkflowReadModel.ts",
        "src/components/agent/chat/workspace/useWorkspaceArticleWorkflowReadModel.test.tsx",
      ],
      "workspace-article-editor-orchestration-model-surface": [
        "src/components/agent/chat/workspace/workspaceArticleEditorOrchestrationModel.ts",
        "src/components/agent/chat/workspace/workspaceArticleEditorOrchestrationModel.unit.test.ts",
      ],
    } as const;

    for (const [id, targets] of Object.entries(expectedTargetsById)) {
      const monitor = legacySurfaceCatalogJson.imports.find(
        (entry) => entry.id === id,
      );

      expect(monitor).toBeTruthy();
      expect(monitor?.classification).toBe("dead");
      expect(monitor?.allowedPaths).toEqual([]);
      expect(monitor?.targets).toEqual(targets);
    }
  });

  it("应阻止 Plugin HostDrawer 本地 fallback 模块回流", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "plugin-host-drawer-fallback-module-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/features/plugin/ui/AgentRunHostDrawerFallback.tsx",
    ]);
  });

  it("应阻止 Rust runtime queue 与旧 registry surface 回流", () => {
    const expectedTargetsById = {
      "rust-runtime-queue-legacy-surface": [
        "lime-rs/crates/agent-runtime/src/runtime_queue.rs",
        "lime-rs/crates/agent-runtime/src/runtime_queue/in_memory.rs",
        "lime-rs/crates/agent-runtime/src/runtime_queue/sqlite.rs",
      ],
      "rust-agent-queue-legacy-surface": [
        "lime-rs/crates/agent/src/queued_turn.rs",
        "lime-rs/crates/agent/src/runtime_queue.rs",
        "lime-rs/crates/agent/src/runtime_support.rs",
      ],
      "rust-app-server-agent-runtime-registry-legacy-surface": [
        "lime-rs/crates/app-server/src/agent_runtime_registry.rs",
      ],
    } as const;

    for (const [id, targets] of Object.entries(expectedTargetsById)) {
      const monitor = legacySurfaceCatalogJson.imports.find(
        (entry) => entry.id === id,
      );

      expect(monitor).toBeTruthy();
      expect(monitor?.classification).toBe("dead");
      expect(monitor?.allowedPaths).toEqual([]);
      expect(monitor?.targets).toEqual(targets);
    }
  });

  it("应阻止已删除的默认 Playwright MCP seed 回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-default-playwright-mcp-seed",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs"]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "@modelcontextprotocol/server-playwright",
        "migrated_playwright_mcp_server_v1",
        "migration_v3::migrate_playwright_mcp_server(",
        "pub mod migration_v3;",
      ]),
    );
  });

  it("应记录已删除的根目录一次性 Task Center 补丁脚本", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "root-task-center-patch-scripts",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "patch_dark.js",
      "patch_flare.js",
      "patch_navbar.js",
    ]);
  });

  it("应记录已删除的旧 PluginMarketplacePage 页面和详情弹窗 surface", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "plugin-marketplace-page-legacy-ui-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/features/plugin/PluginMarketplacePage.tsx",
      "src/features/plugin/PluginMarketplacePage.test.tsx",
      "src/features/plugin/PluginMarketplacePage.visibleBlockers.test.tsx",
      "src/features/plugin/PluginMarketplacePageNavigation.ts",
      "src/features/plugin/PluginMarketplaceDetailPanel.tsx",
      "src/features/plugin/PluginMarketplaceHistorySessionPanel.tsx",
      "src/features/plugin/PluginMarketplaceRegistrationPanel.tsx",
      "src/features/plugin/PluginMarketplaceRegistrationPanelModel.ts",
      "src/features/plugin/PluginMarketplaceSkillPanel.tsx",
    ]);
  });

  it("应记录已删除的旧 PluginMarketplacePage 专属 i18n key", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "plugin-marketplace-page-legacy-i18n-keys",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual(["src/i18n/resources"]);
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "plugin.marketplace.action.detail",
        "plugin.marketplace.action.disable",
        "plugin.marketplace.action.uninstallKeepData",
        "plugin.marketplace.detailActionTitle",
        "plugin.marketplace.detail.",
        "plugin.marketplace.openActionTitle",
        "plugin.marketplace.historyActionTitle",
        "plugin.marketplace.historySelection.",
        "plugin.marketplace.management.",
        "plugin.marketplace.registration.",
        "plugin.marketplace.history.entryBanner",
      ]),
    );
  });

  it("应为 AgentUI 标准防回流提供机械守卫", () => {
    const treeMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "agent-ui-nonstandard-tree-terminology",
    );
    const retiredTeamWorkbenchMonitor =
      legacySurfaceCatalogJson.frontendText.find(
        (entry) =>
          entry.id === "agent-ui-retired-team-workbench-standard-surface",
      );
    const localProcessMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "agent-ui-local-process-owner-terminology",
    );
    const hostDrawerFallbackMonitor =
      legacySurfaceCatalogJson.frontendText.find(
        (entry) => entry.id === "plugin-host-drawer-local-process-fallback",
      );
    const providerMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "agent-ui-direct-provider-runtime-surface",
    );

    expect(treeMonitor).toBeTruthy();
    expect(treeMonitor?.classification).toBe("dead");
    expect(treeMonitor?.patterns).toEqual(["ViewTree", "ProcessTree"]);
    expect(treeMonitor?.includePathPrefixes).toEqual([
      "src/components/agent",
      "src/features/plugin",
      "src/lib/api/agentRuntime",
    ]);
    expect(treeMonitor?.allowedPaths).toEqual([]);

    expect(retiredTeamWorkbenchMonitor).toBeTruthy();
    expect(retiredTeamWorkbenchMonitor?.classification).toBe("dead");
    expect(retiredTeamWorkbenchMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "TeamWorkbench",
        "teamWorkbench",
        "Team Workbench",
        "agent-team-workbench",
      ]),
    );
    expect(retiredTeamWorkbenchMonitor?.includePathPrefixes).toEqual([
      "packages/agent-ui-contracts",
      "packages/agent-runtime-projection",
      "packages/agent-runtime-ui",
      "packages/agent-runtime-client",
      "internal/aiprompts/agent-ui-runtime-standard.md",
      "internal/aiprompts/playwright-e2e.md",
      "internal/prd/next/implementation-roadmap.md",
      "internal/roadmap/agentworkbench",
      "src/components/agent/chat/projection",
      "src/components/agent/chat/workspace",
      "src/i18n/resources",
    ]);
    expect(retiredTeamWorkbenchMonitor?.allowedPaths).toEqual([]);

    expect(localProcessMonitor).toBeTruthy();
    expect(localProcessMonitor?.classification).toBe("dead");
    expect(localProcessMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "LocalProcessPanel",
        "LocalProcessTimeline",
        "AgentProcessTree",
        "ProcessComponent",
        "processComponent",
      ]),
    );
    expect(localProcessMonitor?.allowedPaths).toEqual([]);

    expect(hostDrawerFallbackMonitor).toBeTruthy();
    expect(hostDrawerFallbackMonitor?.classification).toBe("dead");
    expect(hostDrawerFallbackMonitor?.patterns).toEqual([
      "AgentRunHostDrawerFallback",
      "AgentRunLocalProcessFallback",
      "shouldRenderLocalProcessFallback",
      "shouldRenderProjection ? (",
    ]);
    expect(hostDrawerFallbackMonitor?.includePathPrefixes).toEqual([
      "src/features/plugin/ui/AgentRunHostDrawer.tsx",
    ]);
    expect(hostDrawerFallbackMonitor?.allowedPaths).toEqual([]);

    expect(providerMonitor).toBeTruthy();
    expect(providerMonitor?.classification).toBe("dead");
    expect(providerMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "new OpenAI(",
        "providerApiKey",
        "provider_api_key",
        "directProviderRuntime",
        "directProviderClient",
      ]),
    );
    expect(providerMonitor?.allowedPaths).toEqual([]);
  });

  it("AgentUI SDK seed 包不应恢复 ViewTree / ProcessTree 非标准术语", () => {
    const sourceFiles =
      AGENT_UI_PACKAGE_SOURCE_ROOTS.flatMap(collectSourceFiles);
    const offenders = sourceFiles.filter((file) => {
      const source = readFileSync(join(REPO_ROOT, file), "utf8");
      return source.includes("ViewTree") || source.includes("ProcessTree");
    });

    expect(offenders).toEqual([]);
  });

  it("AgentUI SDK seed 包不应恢复旧协作标准术语", () => {
    const sourceFiles =
      AGENT_UI_PACKAGE_SOURCE_ROOTS.flatMap(collectSourceFiles);
    const retiredTerms = [
      "TeamWorkbench",
      "teamWorkbench",
      "Team Workbench",
      "agent-team-workbench",
    ];
    const offenders = sourceFiles.filter((file) => {
      const source = readFileSync(join(REPO_ROOT, file), "utf8");
      return retiredTerms.some((term) => source.includes(term));
    });

    expect(offenders).toEqual([]);
  });

  it("应记录已删除的旧桥接与 New API 手动调试脚本", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) =>
        entry.id === "scripts-hardcoded-bridge-debug-and-newapi-image-smoke",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "scripts/check-chrome-bridge.mjs",
      "scripts/test-chrome-bridge.mjs",
      "scripts/test-ws-connection.mjs",
      "scripts/test-newapi-image.ts",
    ]);
  });

  it("应记录已删除的旧启动排版诊断脚本", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "scripts-startup-layout-diagnostics-legacy",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "scripts/startup-layout-e2e.mjs",
      "scripts/startup-layout-guide.mjs",
    ]);
  });

  it("应封住本地 Plugin workflow runtime profile 的生产回流", () => {
    const profileMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "plugin-local-workflow-runtime-profile-production-surface",
    );
    const adapterMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "plugin-adapter-local-workflow-entry-success-runner",
    );

    expect(profileMonitor).toBeTruthy();
    expect(profileMonitor?.classification).toBe("dead");
    expect(profileMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "buildWorkflowRuntimeCapabilityProfile",
        "workerRuntimeEnabled: true",
        "白名单 workflow DSL",
      ]),
    );
    expect(profileMonitor?.includePathPrefixes).toEqual([
      "src/features/plugin",
      "src/i18n/resources",
    ]);
    expect(profileMonitor?.allowedPaths).toEqual(
      expect.arrayContaining([
        "src/features/plugin/testing/workflowRuntimeCapabilityProfile.ts",
        "src/features/plugin/testing/capabilityDispatcherTestFixtures.ts",
        "src/features/plugin/ui/PluginsPage.runtime.test.tsx",
      ]),
    );

    expect(adapterMonitor).toBeTruthy();
    expect(adapterMonitor?.classification).toBe("dead");
    expect(adapterMonitor?.patterns).toEqual([
      'humanReview: entry.kind === "workflow"',
    ]);
    expect(adapterMonitor?.allowedPaths).toEqual([]);
  });

  it("应记录已删除的 SceneApp runtime context 历史转发壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "sceneapp-runtime-context-compat-barrel",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/sceneapp/types-runtime-context.ts",
    ]);
  });

  it("应将旧 MemoryPage 灵感库混合视图标记为 dead surface", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "memory-page-inspiration-library-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/memory/index.ts",
      "src/components/memory/inspirationProjection.ts",
      "src/components/memory/memoryLayerMetrics.ts",
      "src/components/memory/memoryLayerMetrics.test.ts",
      "src/components/memory/MemoryCuratedTaskSuggestionPanel.tsx",
      "src/components/memory/MemoryCuratedTaskSuggestionPanel.test.tsx",
      "src/components/memory/MemoryPage.tsx",
      "src/components/memory/MemoryPage.test.tsx",
      "src/components/agent/chat/utils/messageInspirationDraft.ts",
      "src/components/agent/chat/utils/messageInspirationDraft.test.ts",
      "src/components/agent/chat/utils/saveSceneAppExecutionAsInspiration.ts",
      "src/components/agent/chat/utils/saveSceneAppExecutionAsInspiration.test.ts",
      "src/components/agent/chat/utils/sceneAppExecutionInspirationDraft.ts",
      "src/components/agent/chat/utils/sceneAppExecutionInspirationDraft.test.ts",
      "src/lib/api/unifiedMemory.ts",
      "src/lib/api/unifiedMemory.test.ts",
    ]);
  });

  it("应将旧 memoryRuntime 网关和预取诊断面标记为 dead surface", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "memory-runtime-retired-gateway-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/api/memoryRuntime.ts",
      "src/lib/api/memoryRuntime.test.ts",
      "src/lib/api/memoryRuntimeTypes.ts",
      "src/lib/runtimeMemoryPrefetchHistory.ts",
      "src/lib/runtimeMemoryPrefetchHistory.test.ts",
      "src/components/agent/chat/components/AgentThreadMemoryPrefetchPreview.tsx",
      "src/components/agent/chat/components/AgentThreadMemoryPrefetchPreview.test.tsx",
      "src/components/agent/chat/components/AgentThreadMemoryPrefetchBaselineCard.tsx",
    ]);
  });

  it("应将旧 SQLite memory crate 标记为 dead surface", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "memory-crate-retired-sqlite-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "lime-rs/crates/memory/Cargo.toml",
      "lime-rs/crates/memory/migrations/001_unified_memory.sql",
      "lime-rs/crates/memory/migrations/003_feedback.sql",
      "lime-rs/crates/memory/src/extractor.rs",
      "lime-rs/crates/memory/src/feedback.rs",
      "lime-rs/crates/memory/src/gatekeeper.rs",
      "lime-rs/crates/memory/src/lib.rs",
      "lime-rs/crates/memory/src/migration.rs",
      "lime-rs/crates/memory/src/migrations/mod.rs",
      "lime-rs/crates/memory/src/migrations/v1_unified_memory.rs",
      "lime-rs/crates/memory/src/migrations/v1_unified_memory.sql",
      "lime-rs/crates/memory/src/models/mod.rs",
      "lime-rs/crates/memory/src/models/unified.rs",
      "lime-rs/crates/memory/src/search.rs",
    ]);
  });

  it("应将 App Server 旧 unified memory 处理器标记为 dead surface", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "memory-app-server-unified-retired-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "lime-rs/crates/app-server/src/local_data_source/unified_memory.rs",
      "lime-rs/crates/app-server/src/processor/unified.rs",
    ]);
  });

  it("应记录已删除的 Team selector/panel 与 TeamWorkspace 产品 UI 面", () => {
    const selectorImportMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "agent-chat-retired-team-selector-ui-surface",
    );
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "agent-chat-team-workspace-board-ui-surface",
    );
    const textMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "agent-chat-retired-team-workspace-board-terminology",
    );

    expect(selectorImportMonitor).toBeTruthy();
    expect(selectorImportMonitor?.classification).toBe("dead");
    expect(selectorImportMonitor?.allowedPaths).toEqual([]);
    expect(selectorImportMonitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/components/Inputbar/components/TeamSelector.tsx",
        "src/components/agent/chat/components/Inputbar/components/TeamSelectorPanel.tsx",
        "src/components/agent/chat/components/Inputbar/components/inputbarTeamSelectorCopy.ts",
        "src/components/agent/chat/components/TeamSuggestionBar.tsx",
        "src/components/agent/chat/utils/teamSuggestion.ts",
      ]),
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/components/AgentUiTeamWorkbenchSurfaceView.tsx",
        "src/components/agent/chat/components/TeamMemoryShadowCard.tsx",
        "src/components/agent/chat/components/TeamWorkbenchSummaryPanel.tsx",
        "src/components/agent/chat/components/TeamWorkspaceBoard.tsx",
        "src/components/agent/chat/components/TeamWorkspaceDock.tsx",
        "src/components/agent/chat/projection/agentUiTeamWorkbenchViewModel.ts",
        "src/components/agent/chat/components/team-workspace-board/TeamWorkspaceBoardShell.tsx",
        "src/components/agent/chat/components/team-workspace-board/useTeamWorkspaceBoardComposer.ts",
      ]),
    );

    expect(textMonitor).toBeTruthy();
    expect(textMonitor?.classification).toBe("dead");
    expect(textMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "TeamSelector",
        "TeamSelectorPanel",
        "inputbarTeamSelectorCopy",
        "TeamSuggestionBar",
        "teamSuggestion",
        "team-suggestion-bar",
        "agentChat.inputbar.teamSuggestion",
        "onEnableSuggestedTeam",
        "handleEnableSuggestedTeam",
        "enableSuggestedTeam",
        "cloneTeamDefinitionAsCustom",
        "saveCustomTeams",
        "loadCustomTeams(",
        "buildWorkspaceSettingsWithCustomTeams",
        "lime.chat.custom_teams.v1",
        "team-workspace-board/",
        "TeamWorkspaceBoard",
        "TeamWorkspaceDock",
        "TeamMemoryShadowCard",
        "TeamWorkbenchSummaryPanel",
        "AgentUiTeamWorkbenchSurfaceView",
        "teamWorkbenchView",
        "teamWorkbenchSurfaceProps",
        "teamWorkspaceDockProps",
        "teamWorkspaceEnabled",
        "dismissActiveTeamWorkbenchAutoOpen",
        "handleActivateTeamWorkbench",
        "team-workbench",
        "empty-state-team-selector",
        "team-selector-stub",
        "Team current tools",
        "Team Memory",
        "Team member",
        "Team policy",
        "Background teammate",
        "Remote teammate",
        "Team 状态",
        "团队控制",
        "团队编队",
        "创建团队",
        "删除团队",
        "当前团队",
      ]),
    );
    expect(textMonitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/README.md",
      "src/components/agent/chat/components",
      "src/components/agent/chat/projection",
      "src/components/agent/chat/utils",
      "src/components/agent/chat/workspace",
      "src/i18n/resources",
    ]);
    expect(textMonitor?.allowedPaths).toEqual([]);
  });

  it("应记录已删除的 SceneApps 独立工作流 rail", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "sceneapps-workflow-rail-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/sceneapps/SceneAppsWorkflowRail.tsx",
    ]);
  });

  it("应将 SceneApp 首页卡片与 Workspace 启动链标记为 dead surface", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "sceneapp-active-launch-surface",
    );
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "sceneapp-active-launch-frontend-surface",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/workspace/useWorkspaceSceneAppEntryActions.ts",
        "src/components/agent/chat/workspace/sceneAppLaunch.ts",
        "src/lib/sceneapp/launchBridge.ts",
        "src/lib/sceneapp/launcher.ts",
      ]),
    );

    expect(frontendMonitor).toBeTruthy();
    expect(frontendMonitor?.classification).toBe("dead");
    expect(frontendMonitor?.allowedPaths).toEqual([]);
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "useWorkspaceSceneAppEntryActions",
        "useSceneAppLaunchRuntime",
        "featuredSceneApps",
        "entry-sceneapp-",
      ]),
    );
  });

  it("应将 SceneApp 独立 catalog 投影与目录产品卡片标记为 dead surface", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "sceneapp-catalog-projection-surface",
    );
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "sceneapp-catalog-product-frontend-surface",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(["src/lib/sceneapp/catalog.ts"]);

    expect(frontendMonitor).toBeTruthy();
    expect(frontendMonitor?.classification).toBe("dead");
    expect(frontendMonitor?.allowedPaths).toEqual([]);
    expect(frontendMonitor?.includePathPrefixes).toEqual([
      "src/lib/api",
      "src/lib/sceneapp",
    ]);
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "compileSceneAppCatalogFromPackage",
        "listSceneAppCatalog",
        "SceneAppCatalog",
        "buildSceneAppCatalogCardViewModel",
      ]),
    );
  });

  it("应将 SceneApp 自动化详情运行面标记为 dead surface", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "sceneapp-automation-detail-runtime-surface",
    );
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "sceneapp-automation-detail-frontend-surface",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/settings-v2/system/automation/useAutomationSceneAppRuntime.ts",
        "src/components/sceneapps/SceneAppRunDetailPanel.tsx",
        "src/components/sceneapps/SceneAppReviewFeedbackBanner.tsx",
        "src/components/sceneapps/index.ts",
      ]),
    );

    expect(frontendMonitor).toBeTruthy();
    expect(frontendMonitor?.classification).toBe("dead");
    expect(frontendMonitor?.allowedPaths).toEqual([]);
    expect(frontendMonitor?.includePathPrefixes).toEqual([
      "src/components/settings-v2/system/automation",
      "src/components/sceneapps",
      "src/lib/sceneapp",
      "src/i18n",
    ]);
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "useAutomationSceneAppRuntime",
        "SceneAppRunDetailPanel",
        "sceneapp-run-detail-save-as-inspiration",
        "settings.automation.details.sceneApp.action.openDetail",
        "settings.automation.details.sceneApp.toast.",
      ]),
    );
  });

  it("应将 SceneApp 执行摘要实时 API 轮询链标记为 dead surface", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "sceneapp-api-runtime-polling-surface",
    );
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "sceneapp-execution-summary-live-api-frontend-surface",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(["src/lib/api/sceneapp.ts"]);

    expect(frontendMonitor).toBeTruthy();
    expect(frontendMonitor?.classification).toBe("dead");
    expect(frontendMonitor?.allowedPaths).toEqual([]);
    expect(frontendMonitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat",
      "src/lib/api",
    ]);
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "@/lib/api/sceneapp",
        "listSceneAppRuns",
        "prepareSceneAppRunGovernanceArtifact",
        "SCENEAPP_RUNTIME_POLL_INTERVAL_MS",
      ]),
    );
  });

  it("应将 src/lib/sceneapp 目录标记为 dead surface", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "sceneapp-library-directory-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(
      expect.arrayContaining([
        "src/lib/sceneapp/index.ts",
        "src/lib/sceneapp/types.ts",
        "src/lib/sceneapp/product.ts",
        "src/lib/sceneapp/automation.ts",
        "src/lib/sceneapp/executionPromptActions.ts",
        "src/lib/sceneapp/runEntryNavigation.ts",
      ]),
    );
  });

  it("应将 SceneApp 执行摘要运行详情构造侧链标记为 dead surface", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "sceneapp-execution-summary-run-detail-helper-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat",
      "src/lib/sceneapp",
    ]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "buildSceneAppExecutionSummaryRunDetailViewModel",
        "buildSceneAppExecutionPromptActions",
        "buildSceneAppExecutionPromptActionPayload",
        "findLatestSceneAppPackResultRun",
        "hasSceneAppRunDeliveryArtifacts",
        "resolveSceneAppRunEntryNavigationTarget",
        "SceneAppProjectPackRuntimePanel",
        "sceneAppExecutionPromptContinuation",
      ]),
    );
  });

  it("应将 SceneApps 独立页面与 sceneapp 命令面标记为 dead surface", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "sceneapps-independent-page-surface",
    );
    const commandMonitor = legacySurfaceCatalogJson.commands.find(
      (entry) => entry.id === "sceneapp-legacy-desktop-host-command-surface",
    );
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "sceneapps-page-route-frontend-surface",
    );
    const rustMonitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-sceneapp-command-module-surface",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/sceneapps/SceneAppsPage.tsx",
        "src/components/sceneapps/useSceneAppsPageRuntime.ts",
        "src/lib/sceneapp/navigation.ts",
      ]),
    );

    expect(commandMonitor).toBeTruthy();
    expect(commandMonitor?.classification).toBe("dead");
    expect(commandMonitor?.allowedPaths).toEqual([]);
    expect(commandMonitor?.commands).toEqual(
      expect.arrayContaining([
        "sceneapp_list_catalog",
        "sceneapp_plan_launch",
        "sceneapp_create_automation_job",
      ]),
    );

    expect(frontendMonitor).toBeTruthy();
    expect(frontendMonitor?.classification).toBe("dead");
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining([
        '"sceneapps"',
        "SceneAppsPageParams",
        "resolveSceneAppsPageEntryParams",
      ]),
    );

    expect(rustMonitor).toBeTruthy();
    expect(rustMonitor?.classification).toBe("dead");
    expect(rustMonitor?.patterns).toEqual(
      expect.arrayContaining(["commands::sceneapp_cmd::", "crate::sceneapp::"]),
    );
  });

  it("应记录已删除的旧 ASR 凭证管理 UI", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "voice-asr-provider-management-ui",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/voice/AddAsrCredentialModal.tsx",
      "src/components/voice/AsrCredentialCard.tsx",
      "src/components/voice/AsrProviderSection.tsx",
      "src/components/voice/index.ts",
    ]);
  });

  it("应将 smart-input、截图对话和初装引导旧面标记为 dead surface", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "smart-input-screenshot-onboarding-surface",
    );
    const commandMonitor = legacySurfaceCatalogJson.commands.find(
      (entry) => entry.id === "smart-input-screenshot-command-surface",
    );
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "smart-input-screenshot-onboarding-frontend-surface",
    );
    const rustMonitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-smart-input-screenshot-surface",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(
      expect.arrayContaining([
        "src/pages/smart-input.tsx",
        "src/lib/api/screenshotChat.ts",
        "src/components/onboarding/OnboardingWizard.tsx",
      ]),
    );

    expect(commandMonitor).toBeTruthy();
    expect(commandMonitor?.classification).toBe("dead");
    expect(commandMonitor?.allowedPaths).toEqual([]);
    expect(commandMonitor?.commands).toEqual(
      expect.arrayContaining([
        "send_screenshot_chat",
        "open_input_with_text",
        "update_screenshot_shortcut",
      ]),
    );

    expect(frontendMonitor).toBeTruthy();
    expect(frontendMonitor?.classification).toBe("dead");
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "@/components/smart-input",
        "settings.experimental.screenshot.",
        "common.smartInput.",
        "OnboardingWizard",
      ]),
    );

    expect(rustMonitor).toBeTruthy();
    expect(rustMonitor?.classification).toBe("dead");
    expect(rustMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "commands::screenshot_cmd::",
        "ScreenshotChatConfig",
        "open_floating_window",
      ]),
    );
  });

  it("应记录已删除的 KnowledgePage 旧拆分组件", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "knowledge-page-legacy-component-split",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/features/knowledge/components/FileEntryList.tsx",
      "src/features/knowledge/components/KnowledgePackCard.tsx",
      "src/features/knowledge/components/KnowledgeStatusRail.tsx",
      "src/features/knowledge/components/KnowledgeTroubleshootingPanel.tsx",
    ]);
  });

  it("应记录已删除的普通视觉 brief 确认旧 helper", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) =>
        entry.id === "agent-chat-plain-visual-brief-confirmation-helper",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/utils/plainVisualBriefConfirmation.ts",
    ]);
  });

  it("应记录已删除的首页空态 SceneApp 旧面板", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "empty-state-legacy-sceneapps-panel-entry",
    );
    const textMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "empty-state-legacy-sceneapps-panel-surface",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead-candidate");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual([
      "src/components/agent/chat/components/EmptyStateSceneAppsPanel.tsx",
    ]);

    expect(textMonitor).toBeTruthy();
    expect(textMonitor?.classification).toBe("dead-candidate");
    expect(textMonitor?.allowedPaths).toEqual([]);
    expect(textMonitor?.patterns).toEqual([
      "EmptyStateSceneAppsPanel",
      "sceneapps-home-directory",
    ]);
    expect(textMonitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat",
    ]);
  });

  it("应记录已删除的旧首页 entry 推荐方案工具", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "entry-recommended-solutions-legacy-helper",
    );
    const textMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "entry-recommended-solutions-legacy-runtime",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead-candidate");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual([
      "src/components/agent/chat/utils/entryRecommendedSolutions.ts",
    ]);

    expect(textMonitor).toBeTruthy();
    expect(textMonitor?.classification).toBe("dead-candidate");
    expect(textMonitor?.allowedPaths).toEqual([]);
    expect(textMonitor?.patterns).toEqual([
      "lime:entry-recommended-solution-usage:v1",
      "ENTRY_RECOMMENDED_SOLUTIONS",
      "listEntryRecommendedSolutions(",
      "recordEntryRecommendedSolutionUsage(",
    ]);
    expect(textMonitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat",
    ]);
  });

  it("应记录已删除的 Agent Chat 侧微信模型同步 Hook", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "agent-chat-wechat-runtime-model-sync-hook",
    );
    const textMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "agent-chat-wechat-runtime-model-sync-hook-usage",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead-candidate");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual([
      "src/components/agent/chat/hooks/useWechatRuntimeModelSync.ts",
    ]);

    expect(textMonitor).toBeTruthy();
    expect(textMonitor?.classification).toBe("dead-candidate");
    expect(textMonitor?.allowedPaths).toEqual([]);
    expect(textMonitor?.patterns).toEqual(["useWechatRuntimeModelSync"]);
    expect(textMonitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat",
    ]);
  });

  it("应禁止 desktop-host/index 恢复旧 mock 聚合导出", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "desktop-host-index-legacy-barrel-exports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      'export * from "./core"',
      'export * from "./event"',
      "openFileDialog",
      "openShell",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/lib/desktop-host/index.ts",
    ]);
  });

  it("应将旧海报素材命令与 helper 收敛到图库主链", () => {
    expect(agentCommandCatalog.deprecatedCommandReplacements).toMatchObject({
      create_poster_metadata: "galleryMaterialMetadata/create",
      get_poster_metadata: "galleryMaterialMetadata/get",
      get_poster_material: "galleryMaterial/get",
      update_poster_metadata: "galleryMaterialMetadata/update",
      delete_poster_metadata: "galleryMaterialMetadata/delete",
      list_by_image_category: "galleryMaterial/listByImageCategory",
      list_by_layout_category: "galleryMaterial/listByLayoutCategory",
      list_by_mood: "galleryMaterial/listByMood",
    });
    expect(agentCommandCatalog.deprecatedHelperReplacements).toMatchObject({
      getPosterMaterial: "getGalleryMaterial",
      createPosterMetadata: "createGalleryMetadata",
      updatePosterMetadata: "updateGalleryMetadata",
      deletePosterMetadata: "deleteGalleryMetadata",
      listPosterMaterialsByImageCategory: "listGalleryMaterialsByImageCategory",
      listPosterMaterialsByLayoutCategory:
        "listGalleryMaterialsByLayoutCategory",
      listPosterMaterialsByMood: "listGalleryMaterialsByMood",
      usePosterMaterial: "useGalleryMaterial",
    });
  });

  it("应记录已删除的旧 SubAgent scheduler Rust 模块路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "team-subagent-scheduler-rust-modules",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "lime-rs/src/commands/subagent_cmd.rs",
      "lime-rs/src/agent/subagent_scheduler.rs",
    ]);
  });

  it("应将旧 SubAgent scheduler 命令与事件总线标记为已删除 surface", () => {
    const commandMonitor = legacySurfaceCatalogJson.commands.find(
      (entry) => entry.id === "team-subagent-scheduler-commands",
    );
    const frontendEventMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "frontend-subagent-scheduler-event-bus",
    );
    const rustEventMonitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-subagent-scheduler-event-bus",
    );

    expect(commandMonitor).toBeTruthy();
    expect(commandMonitor?.classification).toBe("dead-candidate");
    expect(commandMonitor?.allowedPaths).toEqual([]);
    expect(commandMonitor?.commands).toEqual([
      "init_subagent_scheduler",
      "execute_subagent_tasks",
      "cancel_subagent_tasks",
    ]);

    expect(frontendEventMonitor).toBeTruthy();
    expect(frontendEventMonitor?.classification).toBe("dead-candidate");
    expect(frontendEventMonitor?.allowedPaths).toEqual([]);

    expect(rustEventMonitor).toBeTruthy();
    expect(rustEventMonitor?.classification).toBe("dead-candidate");
    expect(rustEventMonitor?.allowedPaths).toEqual([]);
  });

  it("应禁止 subagent metadata 直读重新回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-agent-subagent-metadata-direct-read",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual(["resolve_subagent_session_metadata("]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应将 channels_cmd 旧 CRUD stub 命令标记为 dead-candidate", () => {
    const monitor = legacySurfaceCatalogJson.commands.find(
      (entry) => entry.id === "channels-crud-stub-commands",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.commands).toEqual([
      "get_ai_channels",
      "get_ai_channel",
      "create_ai_channel",
      "update_ai_channel",
      "delete_ai_channel",
      "test_ai_channel",
      "get_notification_channels",
      "get_notification_channel",
      "create_notification_channel",
      "update_notification_channel",
      "delete_notification_channel",
      "test_notification_channel",
    ]);
  });

  it("应禁止 channels_cmd 旧 Rust 模块与注册面重新回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-channels-cmd-legacy-surfaces",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "commands::channels_cmd::",
      "pub mod channels_cmd;",
    ]);
  });

  it("应禁止 SkillSelectorPanel 旧面板路径重新回流", () => {
    const legacyPanelPath = `./${"SkillSelectorPanel"}`;
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-skill-selector-panel-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        `from "${legacyPanelPath}"`,
        `import('${legacyPanelPath}')`,
      ]),
    );
  });

  it("应记录 Inputbar 已删除的 A2UI 浮层桥接入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "inputbar-a2ui-overlay-bridge-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/components/Inputbar/components/A2UIFloatingForm.tsx",
        "src/components/agent/chat/components/Inputbar/hooks/useInputbarDisplayState.ts",
      ]),
    );
  });

  it("应记录已删除的 WorkspacePendingA2UIDialog 文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "workspace-pending-a2ui-dialog-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/workspace/WorkspacePendingA2UIDialog.tsx",
        "src/components/agent/chat/workspace/WorkspacePendingA2UIDialog.test.tsx",
      ]),
    );
  });

  it("应记录已删除的工作区创建确认策略 helper 文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "workspace-create-confirmation-policy-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/workspace/utils/createConfirmationPolicy.ts",
    ]);
  });

  it("应记录已迁出 Inputbar 的 A2UI 提示 helper", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "inputbar-a2ui-dialog-helper-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/components/Inputbar/components/A2UISubmissionNotice.tsx",
        "src/components/agent/chat/components/Inputbar/hooks/useA2UISubmissionNotice.ts",
        "src/components/agent/chat/components/Inputbar/hooks/useStickyA2UIForm.ts",
      ]),
    );
  });

  it("应记录已迁出 Inputbar 的工作流输入状态 helper", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "inputbar-workflow-input-state-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/components/Inputbar/hooks/useThemeWorkbenchInputState.ts",
        "src/components/agent/chat/components/Inputbar/hooks/useThemeWorkbenchInputState.test.ts",
      ]),
    );
  });

  it("应将旧 workspace useWorkflow Hook 标记为已删除 surface", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "workspace-use-workflow-legacy-hook-shell",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.targets).toEqual([
      "src/components/workspace/hooks/useWorkflow.ts",
      "src/components/workspace/hooks/useWorkflow.test.ts",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应将前端 WorkflowRuntimeHost DSL 文件标记为已删除 surface", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "workflow-runtime-host-legacy-dsl-files",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.targets).toEqual([
      "src/features/plugin/runtime/workflowRuntimeHost.ts",
      "src/features/plugin/runtime/workflowRuntimeHost.test.ts",
      "src/features/plugin/runtime/runtimePolicy.ts",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应记录已删除的旧工作区 runtime 文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "workspace-theme-workbench-runtime-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/workspace/useWorkspaceThemeWorkbenchRuntime.ts",
    ]);
  });

  it("应记录已删除的旧工作流布局 helper 文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "workflow-layout-legacy-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/utils/themeWorkbenchLayout.ts",
      "src/components/agent/chat/utils/themeWorkbenchLayout.test.ts",
    ]);
  });

  it("应记录已删除的旧工作区主题工作台 runtime 壳文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "workspace-theme-workbench-runtime-shell-entries",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/workspace/useWorkspaceThemeWorkbenchSidebarRuntime.ts",
      "src/components/agent/chat/workspace/useWorkspaceThemeWorkbenchSidebarRuntime.test.tsx",
      "src/components/agent/chat/workspace/useWorkspaceThemeWorkbenchScaffoldRuntime.ts",
      "src/components/agent/chat/workspace/useWorkspaceThemeWorkbenchScaffoldRuntime.test.tsx",
      "src/components/agent/chat/workspace/useWorkspaceThemeWorkbenchVersionStatusRuntime.ts",
      "src/components/agent/chat/workspace/useWorkspaceThemeWorkbenchDocumentPersistenceRuntime.ts",
      "src/components/agent/chat/workspace/useWorkspaceThemeWorkbenchShellRuntime.tsx",
    ]);
  });

  it("应记录已删除的旧工作区主题工作台 helper 与 sidebar 壳文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) =>
        entry.id ===
        "workspace-theme-workbench-helper-and-sidebar-shell-entries",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/workspace/themeWorkbenchHelpers.ts",
      "src/components/agent/chat/workspace/themeWorkbenchHelpers.test.ts",
      "src/components/agent/chat/workspace/ThemeWorkbenchSidebarSection.tsx",
      "src/components/agent/chat/workspace/useThemeWorkbenchSidebarPresentation.tsx",
    ]);
  });

  it("应记录已删除的 general workbench entry hooks 旧文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "general-workbench-entry-hook-legacy-paths",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/hooks/useThemeWorkbenchEntryPrompt.ts",
      "src/components/agent/chat/hooks/useThemeWorkbenchEntryPrompt.test.tsx",
      "src/components/agent/chat/hooks/useThemeWorkbenchEntryPromptActions.ts",
      "src/components/agent/chat/hooks/useThemeWorkbenchEntryPromptActions.test.tsx",
      "src/components/agent/chat/hooks/useThemeWorkbenchSendBoundary.ts",
      "src/components/agent/chat/hooks/useThemeWorkbenchSendBoundary.test.tsx",
    ]);
  });

  it("应记录已删除的 general workbench entry prompt accessory 旧文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) =>
        entry.id === "general-workbench-entry-prompt-accessory-legacy-paths",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/components/ThemeWorkbenchEntryPromptAccessory.tsx",
      "src/components/agent/chat/components/ThemeWorkbenchEntryPromptAccessory.test.tsx",
    ]);
  });

  it("应记录已删除的 general workbench sidebar 壳旧文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "general-workbench-sidebar-shell-legacy-paths",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/components/ThemeWorkbenchHarnessCard.tsx",
      "src/components/agent/chat/workspace/WorkspaceThemeSidebar.tsx",
    ]);
  });

  it("应记录已删除的 general workbench sidebar 展示壳旧文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "general-workbench-sidebar-display-legacy-paths",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/components/ThemeWorkbenchSidebar.tsx",
      "src/components/agent/chat/components/ThemeWorkbenchSidebar.test.tsx",
      "src/components/agent/chat/components/ThemeWorkbenchSidebarShell.tsx",
      "src/components/agent/chat/components/ThemeWorkbenchSidebarPanels.tsx",
    ]);
  });

  it("应记录已删除的 general workbench sidebar 支撑层旧文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "general-workbench-sidebar-support-legacy-paths",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/components/ThemeWorkbenchContextPanel.tsx",
      "src/components/agent/chat/components/ThemeWorkbenchExecLog.tsx",
      "src/components/agent/chat/components/ThemeWorkbenchWorkflowPanel.tsx",
      "src/components/agent/chat/components/buildThemeWorkbenchContextPanelProps.ts",
      "src/components/agent/chat/components/buildThemeWorkbenchExecLogProps.ts",
      "src/components/agent/chat/components/buildThemeWorkbenchWorkflowPanelProps.ts",
      "src/components/agent/chat/components/buildThemeWorkbenchSidebarOrchestrationSource.ts",
      "src/components/agent/chat/components/themeWorkbenchContextData.ts",
      "src/components/agent/chat/components/themeWorkbenchExecLogData.ts",
      "src/components/agent/chat/components/themeWorkbenchWorkflowData.ts",
      "src/components/agent/chat/components/themeWorkbenchWorkflowData.test.ts",
      "src/components/agent/chat/components/themeWorkbenchSidebarComparator.ts",
      "src/components/agent/chat/components/themeWorkbenchSidebarContentContract.ts",
      "src/components/agent/chat/components/themeWorkbenchSidebarContract.ts",
      "src/components/agent/chat/components/themeWorkbenchSidebarOrchestrationContract.ts",
      "src/components/agent/chat/components/themeWorkbenchSidebarShared.ts",
      "src/components/agent/chat/components/useThemeWorkbenchArtifactActions.ts",
      "src/components/agent/chat/components/useThemeWorkbenchContextPanelState.ts",
      "src/components/agent/chat/components/useThemeWorkbenchExecLogState.ts",
      "src/components/agent/chat/components/useThemeWorkbenchSidebarOrchestration.ts",
      "src/components/agent/chat/components/useThemeWorkbenchSidebarTelemetry.ts",
      "src/components/agent/chat/components/useThemeWorkbenchWorkflowPanelState.ts",
    ]);
  });

  it("应记录已删除的 ThemeWorkbenchSkillsPanel 孤岛组件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "theme-workbench-skills-panel-legacy-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/components/ThemeWorkbenchSkillsPanel.tsx",
      "src/components/agent/chat/components/ThemeWorkbenchSkillsPanel.test.tsx",
    ]);
  });

  it("应记录已删除的 WorkspaceSelector 旧入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "workspace-selector-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/workspace/WorkspaceSelector.tsx",
    ]);
  });

  it("应记录已删除的 LegacyChannelsWorkbench 旧路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "channels-legacy-debug-workbench-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/settings-v2/system/channels/LegacyChannelsWorkbench.tsx",
      "src/components/settings-v2/system/channels/LegacyChannelsWorkbench.test.tsx",
    ]);
  });

  it("应记录已删除的 API compatibility 旧前端网关", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "api-compatibility-gateway-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/api/apiCompatibility.ts",
      "src/lib/api/apiCompatibility.test.ts",
    ]);
  });

  it("应记录已删除的 provider-pool 旧模型库页面簇", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "provider-pool-model-registry-tabs",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/provider-pool/EnhancedModelsTab.tsx",
      "src/components/provider-pool/ModelRegistryTab.tsx",
    ]);
  });

  it("应记录已删除的 provider-pool 独立凭证表单与 barrel 入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "provider-pool-standalone-credential-forms",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/provider-pool/credential-forms/AntigravityFormStandalone.tsx",
      "src/components/provider-pool/credential-forms/ClaudeFormStandalone.tsx",
      "src/components/provider-pool/credential-forms/GeminiFormStandalone.tsx",
      "src/components/provider-pool/credential-forms/KiroFormStandalone.tsx",
      "src/components/provider-pool/credential-forms/index.ts",
    ]);
  });

  it("应记录已删除的 Agent Chat 固定后端 compat 配置壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "agent-chat-fixed-backend-config-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/components/agent/chat/config.ts"]);
  });

  it("应禁止 @代码 专用 parser 与测试入口重新回流", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "agent-chat-code-workbench-command-parser-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/utils/codeWorkbenchCommand.ts",
      "src/components/agent/chat/utils/codeWorkbenchCommand.test.ts",
    ]);
  });

  it("应记录已删除的稳定处理中提示组件与 Hook", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "stable-processing-notice-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/components/StableProcessingNotice.tsx",
      "src/components/agent/chat/hooks/useStableProcessingNotice.ts",
    ]);
  });

  it("应记录已删除的零入口 RadioGroup UI primitive", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "ui-radio-group-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/components/ui/radio-group.tsx"]);
  });

  it("应记录已删除的零入口 Separator UI primitive", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "ui-separator-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/components/ui/separator.tsx"]);
  });

  it("应记录已删除的前端 plugin-ui 渲染链", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "plugin-ui-frontend-runtime-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(
      expect.arrayContaining([
        "src/lib/plugin-ui/ComponentRegistry.ts",
        "src/lib/plugin-ui/DataStore.ts",
        "src/lib/plugin-ui/PluginUIContainer.tsx",
        "src/lib/plugin-ui/PluginUIRenderer.tsx",
        "src/lib/plugin-ui/SurfaceManager.ts",
        "src/lib/plugin-ui/index.ts",
        "src/lib/plugin-ui/usePluginUI.ts",
      ]),
    );
  });

  it("应记录已删除的零入口 Alert UI primitive", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "ui-alert-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/components/ui/alert.tsx"]);
  });

  it("应记录已删除的旧块系统与独立 workspace store", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "blocks-workspace-legacy-runtime-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/blocks/BlockFrame.tsx",
      "src/components/blocks/PreviewBlock.tsx",
      "src/components/blocks/TerminalBlock.tsx",
      "src/components/blocks/WebBlock.tsx",
      "src/components/preview/ImagePreview.tsx",
      "src/lib/blocks/blockStore.ts",
      "src/lib/blocks/index.ts",
      "src/lib/blocks/registry.ts",
      "src/lib/blocks/types.ts",
      "src/lib/workspace/types.ts",
      "src/lib/workspace/workspaceStore.ts",
    ]);
  });

  it("应记录已删除的零入口 barrel 导出文件", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "zero-entry-barrel-export-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/hooks/index.ts",
      "src/lib/artifact/index.ts",
      "src/types/index.ts",
    ]);
  });

  it("应记录已删除的 writeFile 旧解析模块", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "write-file-legacy-module",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/writeFile/index.ts",
      "src/lib/writeFile/parser.ts",
      "src/lib/writeFile/README.md",
    ]);
  });

  it("应记录已删除的零引用前端工具模块", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "unused-frontend-utility-modules",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/notifications.ts",
      "src/lib/legacy-desktop-host-event.ts",
      "src/lib/legacy-desktop-host-event.test.ts",
      "src/lib/utils/syntaxHighlight.ts",
    ]);
  });

  it("应记录已删除的独立 API Key 格式校验 helper", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "api-key-validation-helper-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/utils/apiKeyValidation.ts",
      "src/lib/utils/apiKeyValidation.test.ts",
    ]);
  });

  it("应记录已删除的零入口前端 persona / auto-fix / sysinfo API 包装壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "isolated-frontend-api-wrapper-shells",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/api/personas.ts",
      "src/lib/api/personas.test.ts",
      "src/lib/api/autoFix.ts",
      "src/lib/api/autoFix.test.ts",
      "src/lib/api/sysinfo.ts",
      "src/lib/api/sysinfo.test.ts",
    ]);
  });

  it("应记录已删除的旧 WebSocket 状态组件入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "websocket-status-widget-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/websocket/WebSocketStatus.tsx",
    ]);
  });

  it("应记录已删除的旧 SubAgent 展示壳入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "subagent-progress-display-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/subagent/SubAgentProgress.tsx",
      "src/components/subagent/index.ts",
    ]);
  });

  it("应记录已删除的 OpenClaw Dashboard 内嵌 frame 旧入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "openclaw-dashboard-frame-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/openclaw/OpenClawDashboardFrame.tsx",
    ]);
  });

  it("应记录已删除的 OpenClaw 兼容运行模块", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "openclaw-module-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toContain(
      "src/components/openclaw/OpenClawPage.tsx",
    );
    expect(monitor?.targets).toContain("lime-rs/src/commands/openclaw_cmd.rs");
  });

  it("应记录已删除的 OpenClaw 命令族", () => {
    const commandEntry = legacySurfaceCatalogJson.commands.find(
      (entry) => entry.id === "openclaw-command-surface",
    );

    expect(commandEntry).toBeTruthy();
    expect(commandEntry?.classification).toBe("dead");
    expect(commandEntry?.allowedPaths).toEqual([]);
    expect(commandEntry?.commands).toContain("openclaw_get_environment_status");
    expect(commandEntry?.commands).toContain("openclaw_install_event");
  });

  it("应记录已删除的电商差评回复旧前端页面与 API 网关", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "ecommerce-review-reply-frontend-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/solutions/ecommerce-review-reply/GuideStep.tsx",
      "src/solutions/ecommerce-review-reply/Results.tsx",
      "src/solutions/ecommerce-review-reply/Tasks.tsx",
      "src/solutions/ecommerce-review-reply/index.tsx",
      "src/lib/api/ecommerce-review-reply.ts",
    ]);
  });

  it("应记录已删除的记忆反馈旧前端侧链", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "memory-feedback-frontend-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/memory/MemoryFeedback.tsx",
      "src/components/memory/FeedbackStats.tsx",
      "src/lib/api/memoryFeedback.ts",
      "src/lib/api/memoryFeedback.test.ts",
    ]);
  });

  it("应记录已删除的思考模型切换推导 helper", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "thinking-model-resolver-helper-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/model/thinkingModelResolver.ts",
      "src/lib/model/thinkingModelResolver.test.ts",
    ]);
  });

  it("应记录已删除的桌宠快动作与对话 helper 侧链", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "companion-pet-quick-actions-helper-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/companion/petQuickActions.ts",
      "src/lib/companion/petQuickActions.test.ts",
    ]);
  });

  it("应记录已删除的零入口工作区 helper 与 prompt cache 提示壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "isolated-workspace-helper-notice-surfaces",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/memory/memoryEntryCreationSeed.ts",
      "src/components/memory/memoryEntryCreationSeed.test.ts",
      "src/components/agent/chat/utils/styleRuntime.ts",
      "src/components/agent/chat/utils/styleRuntime.test.ts",
      "src/components/agent/chat/components/Inputbar/components/InputbarPromptCacheNotice.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarPromptCacheNotice.test.tsx",
    ]);
  });

  it("应记录已删除的旧问卷 A2UI 模块命名入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "legacy-questionnaire-a2ui-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/utils/legacyQuestionnaireA2UI.ts",
      "src/components/agent/chat/utils/legacyQuestionnaireA2UI.test.ts",
    ]);
  });

  it("应记录已删除的 settings-v2 旧聊天外观兼容页入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "settings-chat-appearance-page-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/settings-v2/general/chat-appearance/index.tsx",
    ]);
  });

  it("应记录已删除的 settings-v2 旧渠道包装页入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "settings-channels-wrapper-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/settings-v2/system/channels/index.tsx",
      "src/components/settings-v2/system/channels/index.test.tsx",
    ]);
  });

  it("应记录已删除的 settings-v2 旧渠道列表页与弹窗组件簇", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "settings-channels-legacy-list-surfaces",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/settings-v2/system/channels/AIChannelsList.tsx",
      "src/components/settings-v2/system/channels/NotificationChannelsList.tsx",
      "src/components/settings-v2/system/channels/ConnectionTestButton.tsx",
      "src/components/settings-v2/system/channels/SendTestMessageButton.tsx",
      "src/components/settings-v2/system/channels/DeleteChannelDialog.tsx",
      "src/components/settings-v2/system/channels/AIChannelFormModal.tsx",
      "src/components/settings-v2/system/channels/NotificationChannelFormModal.tsx",
    ]);
  });

  it("应记录已删除的 API Key Provider 旧分组列表与完整表单组件簇", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "provider-pool-api-key-legacy-surfaces",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/provider-pool/api-key/AddCustomProviderModal.tsx",
      "src/components/provider-pool/api-key/AddCustomProviderModal.test.ts",
      "src/components/provider-pool/api-key/AddCustomProviderModal.ui.test.tsx",
      "src/components/provider-pool/api-key/ApiKeyItem.tsx",
      "src/components/provider-pool/api-key/ApiKeyList.tsx",
      "src/components/provider-pool/api-key/ApiKeyList.ui.test.tsx",
      "src/components/provider-pool/api-key/ConnectionTestButton.tsx",
      "src/components/provider-pool/api-key/DeleteProviderDialog.tsx",
      "src/components/provider-pool/api-key/DeleteProviderDialog.test.ts",
      "src/components/provider-pool/api-key/ProviderConfigForm.tsx",
      "src/components/provider-pool/api-key/ProviderConfigForm.test.ts",
      "src/components/provider-pool/api-key/ProviderConfigForm.ui.test.tsx",
      "src/components/provider-pool/api-key/ProviderConfigForm.utils.ts",
      "src/components/provider-pool/api-key/ProviderGroup.tsx",
      "src/components/provider-pool/api-key/ProviderList.tsx",
      "src/components/provider-pool/api-key/ProviderList.test.ts",
      "src/components/provider-pool/api-key/ProviderListItem.tsx",
      "src/components/provider-pool/api-key/ProviderListItem.test.ts",
      "src/components/provider-pool/api-key/ProviderListItem.ui.test.tsx",
      "src/components/provider-pool/api-key/ProviderModelList.tsx",
      "src/components/provider-pool/api-key/ProviderModelList.test.tsx",
      "src/components/provider-pool/api-key/providerModelListCache.ts",
      "src/components/provider-pool/api-key/SectionInfoButton.tsx",
    ]);
  });

  it("应记录 API Key Provider 旧 provider-pool 目录名已退役", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "provider-pool-api-key-old-current-directory",
    );
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "frontend-provider-pool-api-key-old-imports",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.classification).toBe("dead-candidate");
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/provider-pool/api-key/ApiKeyProviderSection.tsx",
        "src/components/provider-pool/api-key/ProviderSetting.tsx",
        "src/components/provider-pool/api-key/index.ts",
      ]),
    );

    expect(frontendMonitor).toBeTruthy();
    expect(frontendMonitor?.classification).toBe("dead-candidate");
    expect(frontendMonitor?.allowedPaths).toEqual([]);
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining(["@/components/provider-pool/api-key"]),
    );
  });

  it("应阻止 Provider 导出文件名恢复旧品牌前缀", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "frontend-provider-export-retired-brand-filename",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.patterns).toEqual(["lime-providers-"]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/api-key-provider",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应记录已删除的 settings-v2 执行轨迹独立页面壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "settings-execution-tracker-page-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/settings-v2/system/execution-tracker/index.tsx",
      "src/components/settings-v2/system/execution-tracker/index.test.tsx",
    ]);
  });

  it("应记录已删除的 settings-v2 旧代理配置页入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "settings-proxy-page-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/settings-v2/system/proxy/index.tsx",
    ]);
  });

  it("应记录已删除的 settings-v2 旧通用页头组件入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "settings-header-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/settings-v2/features/SettingHeader.tsx",
    ]);
  });

  it("应记录已删除的 settings-v2 旧共享语言选择器入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "settings-language-selector-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/settings-v2/shared/language/LanguageSelector.tsx",
    ]);
  });

  it("应记录已删除的 ExperimentalBanner 旧提示组件入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "experimental-banner-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/ui/ExperimentalBanner.tsx",
    ]);
  });

  it("应记录已删除的 PanelLayout 旧分屏布局入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "panel-layout-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/components/layout/PanelLayout.tsx"]);
  });

  it("应记录已删除的旧 SubAgent scheduler Hook 路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "team-subagent-scheduler-hook",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应记录已删除的旧 SubAgent scheduler 前端 API 路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "team-subagent-scheduler-api",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.targets).toEqual(["src/lib/api/subAgentScheduler.ts"]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应记录已删除的旧项目与工作区通用 Hook 壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "project-workspace-legacy-hooks",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/hooks/useProject.ts",
      "src/hooks/useWorkspace.ts",
    ]);
  });

  it("应记录已删除的旧项目上下文前端壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "project-context-legacy-frontend-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/hooks/useProjectContext.ts",
      "src/lib/api/projectContext.ts",
      "src/types/context.ts",
    ]);
  });

  it("应记录已删除的零入口通用 Hook 包装壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "unused-frontend-generic-hook-wrappers",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/hooks/useErrorHandler.ts",
      "src/hooks/useAutoFix.ts",
      "src/hooks/useConfigEvents.ts",
      "src/hooks/useFileMonitoring.ts",
      "src/hooks/usePersonas.ts",
      "src/hooks/useProviderState.ts",
    ]);
  });

  it("应记录已删除的零入口任务运行态胶囊卡壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "agent-task-runtime-card-shell",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/components/AgentTaskRuntimeCard.tsx",
    ]);
  });

  it("应记录已删除的零入口 sonner Hook 包装壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "unused-toast-hook-wrapper",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/hooks/use-toast.ts"]);
  });

  it("应记录已删除的统一 chat 类型壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "legacy-chat-shared-type-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/types/chat.ts"]);
  });

  it("应记录已删除的统一 platform 类型壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "legacy-platform-shared-type-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/types/platform.ts"]);
  });

  it("应记录已删除的统一 persona 类型壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "legacy-persona-shared-type-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/types/persona.ts"]);
  });

  it("应记录已删除的零入口 layout 样式表壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "legacy-layout-stylesheet-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/components/layout/layout.css"]);
  });

  it("应记录已删除的零入口连接管理前端网关壳", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "connection-management-legacy-frontend-gateway",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/lib/connection-api.ts"]);
  });

  it("应记录已删除的 i18n 动态模板与 barrel 旧入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "i18n-dynamic-template-legacy-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/i18n/index.ts",
      "src/i18n/dynamic-translation.ts",
      "src/i18n/I18nPatchProvider.tsx",
      "src/i18n/dom-replacer.ts",
      "src/i18n/text-map.ts",
      "src/i18n/patches/en.json",
      "src/i18n/patches/zh.json",
    ]);
  });

  it("应记录已删除的 compat subagent runtime 桥路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "team-subagent-runtime-compat-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/hooks/useCompatSubagentRuntime.ts",
      "src/components/agent/chat/utils/compatSubagentRuntime.ts",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应记录已删除的旧问卷转 A2UI compat 桥路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "legacy-questionnaire-a2ui-compat-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.targets).toEqual([
      "src/components/agent/chat/utils/compatQuestionnaireA2UI.ts",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应记录已删除的首页 entry task prompt composer 入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "empty-state-entry-task-prompt-composer-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/utils/entryPromptComposer.ts",
        "src/components/agent/chat/utils/entryPromptComposer.test.ts",
      ]),
    );
  });

  it("应禁止技能入口重新回到扁平 props 透传与扁平契约", () => {
    const parentMonitorIds = [
      "inputbar-composer-flat-skill-parent-props",
      "empty-state-composer-flat-skill-parent-props",
    ];
    const contractMonitorIds = [
      "inputbar-composer-flat-skill-prop-contract",
      "empty-state-composer-flat-skill-prop-contract",
    ];

    for (const monitorId of [...parentMonitorIds, ...contractMonitorIds]) {
      const monitor = legacySurfaceCatalogJson.frontendText.find(
        (entry) => entry.id === monitorId,
      );

      expect(monitor).toBeTruthy();
      expect(monitor?.classification).toBe("dead-candidate");
      expect(monitor?.allowedPaths).toEqual([]);
      expect(
        (monitor?.patterns?.length ?? 0) +
          (monitor?.regexPatterns?.length ?? 0),
      ).toBeGreaterThan(0);
    }
  });

  it("应禁止 settings-v2 恢复旧 chat-appearance 与 channels compat tab", () => {
    const monitorIds = [
      "settings-chat-appearance-legacy-tab",
      "settings-channels-legacy-tab",
    ];

    for (const monitorId of monitorIds) {
      const monitor = legacySurfaceCatalogJson.frontendText.find(
        (entry) => entry.id === monitorId,
      );

      expect(monitor).toBeTruthy();
      expect(monitor?.classification).toBe("dead-candidate");
      expect(monitor?.allowedPaths).toEqual([]);
      expect(
        (monitor?.patterns?.length ?? 0) +
          (monitor?.regexPatterns?.length ?? 0),
      ).toBeGreaterThan(0);
    }
  });

  it("应禁止实验设置恢复旧 UpdateNotification compat 空导出", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "settings-update-notification-compat-export",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(
      (monitor?.patterns?.length ?? 0) + (monitor?.regexPatterns?.length ?? 0),
    ).toBeGreaterThan(0);
  });

  it("应禁止设置中心恢复旧共享 LanguageSelector 组件表面", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "settings-language-selector-legacy-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(
      (monitor?.patterns?.length ?? 0) + (monitor?.regexPatterns?.length ?? 0),
    ).toBeGreaterThan(0);
  });

  it("应禁止 ChannelsDebugWorkbench 恢复旧表单重连预留数组", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "channels-debug-workbench-legacy-form-reserve",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(
      (monitor?.patterns?.length ?? 0) + (monitor?.regexPatterns?.length ?? 0),
    ).toBeGreaterThan(0);
  });

  it("应禁止 ChannelsDebugWorkbench 恢复已零调用的旧内联渠道表单壳", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "channels-debug-workbench-legacy-inline-forms",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(
      (monitor?.patterns?.length ?? 0) + (monitor?.regexPatterns?.length ?? 0),
    ).toBeGreaterThan(0);
  });

  it("应禁止 Inputbar 恢复 A2UI 浮层 props 透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-a2ui-panel-prop-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/index.tsx",
    ]);
    expect(monitor?.regexPatterns).toEqual([
      "\\bpendingA2UIForm\\s*\\?:",
      "\\bonA2UISubmit\\s*\\?:",
      "\\ba2uiSubmissionNotice\\s*\\?:",
    ]);
  });

  it("应禁止 WorkspacePendingA2UIPanel 回流到 Inputbar A2UI panel helper 路径", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "workspace-pending-a2ui-panel-inputbar-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/workspace/WorkspacePendingA2UIPanel.tsx",
      "src/components/agent/chat/workspace/WorkspacePendingA2UIPanel.test.tsx",
    ]);
    expect(monitor?.patterns).toEqual([
      "../components/Inputbar/components/A2UISubmissionNotice",
      "../components/Inputbar/hooks/useA2UISubmissionNotice",
      "../components/Inputbar/hooks/useStickyA2UIForm",
    ]);
  });

  it("应禁止工作流输入状态 helper 回流到 Inputbar hooks 路径", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "workflow-input-state-inputbar-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarController.ts",
      "src/components/agent/chat/components/Inputbar/index.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarWorkflowStatusPanel.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarComposerSection.tsx",
      "src/components/agent/chat/workspace/useWorkspaceGeneralWorkbenchRuntime.ts",
      "src/components/agent/chat/utils/workflowLayout.ts",
    ]);
    expect(monitor?.patterns).toEqual([
      "../components/Inputbar/hooks/useThemeWorkbenchInputState",
      "./hooks/useThemeWorkbenchInputState",
      "../hooks/useThemeWorkbenchInputState",
    ]);
  });

  it("应禁止 general workbench entry hooks 回流到旧 themeWorkbench 路径", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "general-workbench-entry-hook-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/AgentChatWorkspace.tsx",
      "src/components/agent/chat/components/GeneralWorkbenchEntryPromptAccessory.tsx",
      "src/components/agent/chat/components/GeneralWorkbenchEntryPromptAccessory.test.tsx",
      "src/components/agent/chat/hooks/useGeneralWorkbenchEntryPrompt.test.tsx",
      "src/components/agent/chat/hooks/useGeneralWorkbenchEntryPromptActions.ts",
      "src/components/agent/chat/hooks/useGeneralWorkbenchEntryPromptActions.test.tsx",
      "src/components/agent/chat/hooks/useGeneralWorkbenchSendBoundary.test.tsx",
      "src/components/agent/chat/workspace/useWorkspaceAutoGuideRuntime.ts",
      "src/components/agent/chat/workspace/useWorkspaceSendActions.ts",
    ]);
    expect(monitor?.patterns).toEqual([
      "./hooks/useThemeWorkbenchEntryPrompt",
      "./hooks/useThemeWorkbenchEntryPromptActions",
      "./hooks/useThemeWorkbenchSendBoundary",
      "./useThemeWorkbenchEntryPrompt",
      "./useThemeWorkbenchEntryPromptActions",
      "./useThemeWorkbenchSendBoundary",
      "../hooks/useThemeWorkbenchEntryPrompt",
      "../hooks/useThemeWorkbenchEntryPromptActions",
      "../hooks/useThemeWorkbenchSendBoundary",
    ]);
  });

  it("应禁止 general workbench entry prompt accessory 回流到旧 ThemeWorkbench 文件路径", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "general-workbench-entry-prompt-accessory-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/workspace/useWorkspaceInputbarPresentation.tsx",
      "src/components/agent/chat/components/GeneralWorkbenchEntryPromptAccessory.test.tsx",
    ]);
    expect(monitor?.patterns).toEqual([
      "../components/ThemeWorkbenchEntryPromptAccessory",
      "./ThemeWorkbenchEntryPromptAccessory",
    ]);
  });

  it("应禁止 general workbench sidebar 壳回流到旧 ThemeWorkbenchHarnessCard / WorkspaceThemeSidebar 路径", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "general-workbench-sidebar-shell-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/workspace/WorkspaceGeneralWorkbenchSidebar.tsx",
      "src/components/agent/chat/workspace/useGeneralWorkbenchSidebarPresentation.tsx",
    ]);
    expect(monitor?.patterns).toEqual([
      "../components/ThemeWorkbenchHarnessCard",
      "./WorkspaceThemeSidebar",
    ]);
  });

  it("应禁止 general workbench sidebar 展示壳回流到旧 ThemeWorkbenchSidebar / Shell / Panels 路径", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "general-workbench-sidebar-display-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/index.test.tsx",
      "src/components/agent/chat/components/GeneralWorkbenchSidebar.tsx",
      "src/components/agent/chat/components/GeneralWorkbenchSidebar.test.tsx",
      "src/components/agent/chat/components/useGeneralWorkbenchSidebarOrchestration.ts",
      "src/components/agent/chat/components/useGeneralWorkbenchSidebarTelemetry.ts",
      "src/components/agent/chat/workspace/GeneralWorkbenchSidebarSection.tsx",
    ]);
    expect(monitor?.patterns).toEqual([
      "./components/ThemeWorkbenchSidebar",
      "../components/ThemeWorkbenchSidebar",
      "./ThemeWorkbenchSidebar",
      "./ThemeWorkbenchSidebarShell",
      "./ThemeWorkbenchSidebarPanels",
    ]);
  });

  it("应禁止 general workbench sidebar 支撑层回流到旧 ThemeWorkbench 文件路径", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "general-workbench-sidebar-support-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components",
      "src/components/agent/chat/workspace/GeneralWorkbenchSidebarSection.tsx",
      "src/components/agent/chat/workspace/useWorkspaceGeneralWorkbenchScaffoldRuntime.ts",
    ]);
    expect(monitor?.patterns).toEqual([
      "./ThemeWorkbenchContextPanel",
      "./ThemeWorkbenchExecLog",
      "./ThemeWorkbenchWorkflowPanel",
      "./buildThemeWorkbenchContextPanelProps",
      "./buildThemeWorkbenchExecLogProps",
      "./buildThemeWorkbenchWorkflowPanelProps",
      "./buildThemeWorkbenchSidebarOrchestrationSource",
      "./themeWorkbenchContextData",
      "./themeWorkbenchExecLogData",
      "./themeWorkbenchWorkflowData",
      "./themeWorkbenchSidebarComparator",
      "./themeWorkbenchSidebarContentContract",
      "./themeWorkbenchSidebarContract",
      "./themeWorkbenchSidebarOrchestrationContract",
      "./themeWorkbenchSidebarShared",
      "./useThemeWorkbenchArtifactActions",
      "./useThemeWorkbenchContextPanelState",
      "./useThemeWorkbenchExecLogState",
      "./useThemeWorkbenchSidebarOrchestration",
      "./useThemeWorkbenchSidebarTelemetry",
      "./useThemeWorkbenchWorkflowPanelState",
      "../components/themeWorkbenchSidebarContract",
      "../components/themeWorkbenchWorkflowData",
    ]);
  });

  it("应禁止 ThemeWorkbenchSkillsPanel 回流到运行时代码或测试夹具", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "theme-workbench-skills-panel-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components",
      "src/components/agent/chat/index.test.tsx",
    ]);
    expect(monitor?.patterns).toEqual([
      "./ThemeWorkbenchSkillsPanel",
      "../components/ThemeWorkbenchSkillsPanel",
    ]);
  });

  it("应禁止运行时代码绕过共享绑定边界直接构造 skillSelection", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "skill-selection-direct-construction-runtime-usage",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual(["createSkillSelectionProps("]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components",
      "src/components/agent/chat/skill-selection",
    ]);
    expect(monitor?.allowedPaths).toEqual([
      "src/components/agent/chat/skill-selection/skillSelectionBindings.ts",
    ]);
  });

  it("应禁止共享技能选择能力继续从 Inputbar 旧目录导入", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "skill-selection-inputbar-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyState.tsx",
      "src/components/agent/chat/components/EmptyStateComposerPanel.tsx",
      "src/components/agent/chat/components/EmptyStateComposerPanel.test.tsx",
      "src/components/agent/chat/components/EmptyState.test.tsx",
      "src/components/agent/chat/components/Inputbar/index.tsx",
      "src/components/agent/chat/components/Inputbar/index.test.tsx",
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarController.ts",
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarSend.ts",
      "src/components/agent/chat/components/Inputbar/components/InputbarComposerSection.tsx",
      "src/components/agent/chat/components/Inputbar/components/BuiltinCommandBadge.tsx",
    ]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "./Inputbar/components/CharacterMention",
        "./Inputbar/components/SkillBadge",
        "./Inputbar/components/SkillSelector",
        "./Inputbar/components/skillSelectionBindings",
        "./Inputbar/hooks/useActiveSkill",
        "../components/SkillBadge",
        "./useActiveSkill",
        "./CharacterMention",
        "./SkillSelector",
        "./skillSelectionBindings",
        "./builtinCommands",
        "./useIdleModulePreload",
      ]),
    );
  });

  it("应记录已删除的独立终端页面与挂件 surface 文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "terminal-page-shell-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/terminal/index.ts",
      "src/components/terminal/TerminalWorkspace.tsx",
      "src/components/terminal/TerminalPanel.tsx",
      "src/components/terminal/TerminalView.tsx",
      "src/components/terminal/terminalPageHotkeys.ts",
      "src/components/terminal/widgets/FileBrowserView.tsx",
      "src/components/terminal/widgets/SysinfoView.tsx",
      "src/components/terminal/widgets/WebView.tsx",
      "src/components/terminal/ai/index.ts",
      "src/components/terminal/ai/TerminalAIInput.tsx",
    ]);
  });

  it("应记录已删除的终端状态与 VDOM 侧链文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "terminal-runtime-state-modules",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/lib/terminal/README.md",
      "src/lib/terminal/stickers/README.md",
      "src/lib/terminal/stickers/index.ts",
      "src/lib/terminal/stickers/store.ts",
      "src/lib/terminal/stickers/types.ts",
      "src/lib/terminal/store/README.md",
      "src/lib/terminal/store/atoms.ts",
      "src/lib/terminal/store/events.ts",
      "src/lib/terminal/store/hooks.ts",
      "src/lib/terminal/store/index.ts",
      "src/lib/terminal/store/multiInput.ts",
      "src/lib/terminal/store/types.ts",
      "src/lib/terminal/store/viewmodel.ts",
      "src/lib/terminal/themes.ts",
      "src/lib/terminal/vdom/README.md",
      "src/lib/terminal/vdom/index.ts",
      "src/lib/terminal/vdom/store.ts",
      "src/lib/terminal/vdom/types.ts",
    ]);
  });

  it("应记录已删除的工具箱页面与图像分析工具面文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "tools-page-shell-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/tools/ToolsPage.tsx",
      "src/components/tools/ToolCardContextMenu.tsx",
      "src/components/tools/image-analysis/index.ts",
      "src/components/tools/image-analysis/ImageAnalysisTool.tsx",
    ]);
  });

  it("应记录已删除的独立插图页面与旧搜图 surface 文件路径", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "image-page-shell-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/image-gen/ImageGenPage.tsx",
      "src/components/image-gen/ImageGenPage.test.tsx",
      "src/components/image-gen/tabs/AiImageGenTab.tsx",
      "src/components/image-gen/tabs/ImageSearchTab.tsx",
      "src/components/image-gen/tabs/ImageSearchTab.test.tsx",
      "src/components/image-gen/hooks/useImageSearch.ts",
      "src/components/image-gen/hooks/useImageSearch.test.tsx",
    ]);
  });

  it("应记录已删除的 image-gen 目录级 barrel 入口", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "image-generation-runtime-barrel-entry",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual(["src/components/image-gen/index.ts"]);
  });

  it("应记录已删除的 Renderer Provider 直连图片 executor 文件", () => {
    const monitor = legacySurfaceCatalogJson.imports.find(
      (entry) =>
        entry.id === "frontend-retired-direct-provider-image-executor-files",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.targets).toEqual([
      "src/components/image-gen/falImageExecutor.ts",
      "src/components/image-gen/falImageExecutor.test.ts",
      "src/components/image-gen/geminiImageExecutor.ts",
      "src/components/image-gen/geminiImageExecutor.test.ts",
      "src/components/image-gen/openAICompatibleImageExecutor.ts",
      "src/components/image-gen/openAICompatibleImageExecutor.test.ts",
      "src/components/image-gen/standardImageExecutor.ts",
      "src/components/image-gen/standardImageExecutor.test.ts",
      "src/components/image-gen/imageGenerationBatchRunner.ts",
      "src/components/image-gen/useImageGen.live.test.ts",
      "src/components/image-gen/localImageServerExecutor.ts",
      "src/components/image-gen/localImageServerExecutor.test.ts",
      "src/components/image-gen/useImageGen.transport.test.tsx",
      "src/components/image-gen/useImageGen.resource.test.tsx",
      "src/components/image-gen/imageGenLocalState.ts",
      "src/components/image-gen/types.ts",
      "src/components/image-gen/imageExecutorUtils.ts",
      "src/components/image-gen/imageErrorPresentation.ts",
      "src/components/image-gen/imageErrorPresentation.test.ts",
      "src/components/image-gen/imageResponseParsers.ts",
      "src/components/image-gen/imageResponseParsers.test.ts",
      "src/components/image-gen/localImageServerErrors.ts",
    ]);
  });

  it("应限制 AI 图片生成 runtime 入口继续扩散到 Claw 工作台之外", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "image-generation-runtime-entry-usage",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual(["useImageGen({"]);
    expect(monitor?.includePathPrefixes).toEqual(["src/components"]);
    expect(monitor?.allowedPaths).toEqual([
      "src/components/agent/chat/workspace/useWorkspaceImageWorkbenchProviderRuntime.ts",
    ]);
  });

  it("应冻结无生产消费者的 Renderer 本机图片 HTTP executor", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "frontend-deprecated-local-image-http-executor",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.patterns).toEqual(["requestImagesFromLocalImageServer"]);
    expect(monitor?.includePathPrefixes).toEqual(["src/components/image-gen"]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应禁止已删除的 Renderer Provider 直连图片 executors 回流", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "frontend-retired-direct-provider-image-executors",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.patterns).toEqual([
      "requestImageFromFal(",
      "requestImageFromFalQueue(",
      "requestImageFromGemini(",
      "requestImageFromNewApi(",
      "requestImageFromNewApiResponsesStream(",
      "requestImagesFromStandardImagesApi(",
      "runSingleImageGenerationBatch(",
    ]);
    expect(monitor?.includePathPrefixes).toEqual(["src"]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应禁止前端恢复 image-gen 目录级 barrel 导入", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "image-generation-runtime-barrel-imports",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual([
      'from "@/components/image-gen"',
      "from '@/components/image-gen'",
      'import("@/components/image-gen")',
      "import('@/components/image-gen')",
      'from "@/components/image-gen/index"',
      "from '@/components/image-gen/index'",
      'import("@/components/image-gen/index")',
      "import('@/components/image-gen/index')",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应禁止前端恢复独立工具箱页面 surface", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "frontend-tools-page-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual([]);
    expect(monitor?.regexPatterns).toEqual([
      "\\bcurrentPage\\s*===\\s*[\"']tools[\"']",
      "\\bcurrentPage\\s*===\\s*[\"']image-analysis[\"']",
      "\\bpage\\s*:\\s*[\"']tools[\"']",
      "\\bpage\\s*:\\s*[\"']image-analysis[\"']",
      "\\bonNavigate\\(\\s*[\"']tools[\"']",
      "\\bonNavigate\\(\\s*[\"']image-analysis[\"']",
    ]);
    expect(monitor?.includePathPrefixes).toEqual(["src"]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应禁止前端恢复独立终端页面 surface", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "frontend-terminal-page-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual([]);
    expect(monitor?.regexPatterns).toEqual([
      "\\bcurrentPage\\s*===\\s*[\"']terminal[\"']",
      "\\bcurrentPage\\s*===\\s*[\"']sysinfo[\"']",
      "\\bcurrentPage\\s*===\\s*[\"']files[\"']",
      "\\bcurrentPage\\s*===\\s*[\"']web[\"']",
      "\\bpage\\s*:\\s*[\"']terminal[\"']",
      "\\bpage\\s*:\\s*[\"']sysinfo[\"']",
      "\\bpage\\s*:\\s*[\"']files[\"']",
      "\\bpage\\s*:\\s*[\"']web[\"']",
    ]);
    expect(monitor?.includePathPrefixes).toEqual(["src"]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应禁止前端恢复独立插图页面 surface", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "frontend-image-page-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual([]);
    expect(monitor?.regexPatterns).toEqual([
      "\\bcurrentPage\\s*===\\s*[\"']image-gen[\"']",
      "\\bpage\\s*:\\s*[\"']image-gen[\"']",
      "\\brenderContent\\([\"']image-gen[\"']",
    ]);
    expect(monitor?.includePathPrefixes).toEqual(["src"]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应禁止首页恢复旧 entry task 发送链", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "empty-state-legacy-entry-task-runtime-usage",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyState.tsx",
      "src/components/agent/chat/components/EmptyStateComposerPanel.tsx",
      "src/components/agent/chat/utils/contextualRecommendations.ts",
    ]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "ENTRY_THEME_ID",
        "SOCIAL_MEDIA_ENTRY_TASKS",
        "composeEntryPrompt(",
        "validateEntryTaskSlots(",
      ]),
    );
  });

  it("应禁止首页类型定义恢复旧 entry task 契约", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "empty-state-legacy-entry-task-type-contract",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/types.ts",
    ]);
    expect(monitor?.patterns).toEqual([
      "export type EntryTaskType",
      "export interface EntryTaskSlotDefinition",
      "export interface EntryTaskTemplate",
      "export type EntryTaskSlotValues",
    ]);
  });

  it("应禁止 Inputbar 恢复旧 execution strategy 兼容 props 链", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-legacy-execution-strategy-prop-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(["showExecutionStrategy"]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyStateComposerPanel.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarComposerSection.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarCore.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarTools.tsx",
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarAdapter.ts",
      "src/components/input-kit/adapters/agentAdapter.ts",
      "src/components/input-kit/adapters/types.ts",
    ]);
  });

  it("应禁止 Inputbar 家族恢复旧空透传 props", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-legacy-dead-prop-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "isCanvasOpen",
      "isExecutionRuntimeActive",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/AgentChatWorkspace.tsx",
      "src/components/agent/chat/components/Inputbar/index.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarComposerSection.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarCore.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarTools.tsx",
    ]);
  });

  it("应禁止 Inputbar hooks 恢复旧本地 toggle 兜底链", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-legacy-local-tool-toggle-runtime",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      'case "execution_strategy"',
      'case "canvas"',
      'activeTools["execution_strategy"]',
      "onToggleCanvas",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarController.ts",
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarSend.ts",
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarToolState.ts",
      "src/components/agent/chat/components/Inputbar/index.tsx",
      "src/components/agent/chat/workspace/useWorkspaceInputbarSceneRuntime.tsx",
      "src/components/agent/chat/AgentChatWorkspace.tsx",
    ]);
  });

  it("应禁止 TeamSelector 调用端恢复旧 workspace 上下文透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "team-selector-legacy-context-prop-callsite",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(["workspaceId={workspaceId}"]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyStateComposerPanel.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarComposerSection.tsx",
    ]);
  });

  it("应禁止 TeamSelector 面板恢复未消费的旧运行时上下文透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "team-selector-panel-legacy-runtime-context-props",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "providerType={providerType}",
      "model={model}",
      "executionStrategy={executionStrategy}",
      "workspaceId?: string | null;",
      "providerType?: string;",
      "model?: string;",
      'executionStrategy?: "react" | "code_orchestrated" | "auto";',
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/TeamSelector.tsx",
      "src/components/agent/chat/components/Inputbar/components/TeamSelectorPanel.tsx",
    ]);
  });

  it("应禁止 TeamSelector 恢复旧触发器文案与样式透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "team-selector-legacy-trigger-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/TeamSelector.tsx",
    ]);
    expect(monitor?.regexPatterns).toEqual([
      "\\btriggerLabel\\s*\\?:",
      "\\bclassName\\s*\\?:",
    ]);
  });

  it("应禁止 Inputbar 恢复旧 workspace 上下文透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-legacy-workspace-context-prop-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "workspaceId?: string | null;",
      'projectId: InputbarParams["workspaceId"];',
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/index.tsx",
      "src/components/agent/chat/workspace/useWorkspaceInputbarSceneRuntime.tsx",
    ]);
  });

  it("应禁止 Inputbar 恢复旧 task 工具状态契约", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-legacy-task-tool-state-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "task: false",
      "task: true",
      "task?: boolean;",
      "toolStates?.task",
      "next.task",
      "prev.task",
      "taskEnabled",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarToolState.ts",
      "src/components/agent/chat/components/Inputbar/index.tsx",
      "src/components/agent/chat/components/Inputbar/index.test.tsx",
    ]);
  });

  it("应禁止 Inputbar 工具状态 runtime 恢复已删除的旧工具动作分支", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-legacy-dead-tool-action-runtime",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      'case "clear"',
      'case "new_topic"',
      'case "quick_action"',
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarToolState.ts",
    ]);
  });

  it("应禁止 Inputbar 恢复旧 onClearMessages 透传链", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-legacy-clear-messages-prop-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "onClearMessages?: () => void;",
      "onClearMessages: handleClearMessages,",
      'handleClearMessages: InputbarParams["onClearMessages"];',
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/index.tsx",
      "src/components/agent/chat/components/Inputbar/hooks/useInputbarController.ts",
      "src/components/agent/chat/workspace/useWorkspaceInputbarSceneRuntime.tsx",
      "src/components/agent/chat/AgentChatWorkspace.tsx",
    ]);
  });

  it("应禁止 InputbarModelExtra 恢复旧工作流 variant 透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-model-extra-legacy-variant-prop-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "isThemeWorkbenchVariant?: boolean;",
      "isThemeWorkbenchVariant={isThemeWorkbenchVariant}",
      "isThemeWorkbenchVariant = false,",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/InputbarComposerSection.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarModelExtra.tsx",
    ]);
  });

  it("应禁止 SkillSelector 恢复旧触发按钮自定义表面", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "skill-selector-legacy-trigger-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "triggerLabel?: string;",
      "className?: string;",
      'triggerLabel = "技能",',
      "className,",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/skill-selection/SkillSelector.tsx",
    ]);
  });

  it("应禁止 InputbarModelExtra 恢复旧模型 setter fallback", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-model-extra-legacy-setter-fallback",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "setProviderType?: (type: string) => void;",
      "setModel?: (model: string) => void;",
      "const NOOP_SET_PROVIDER_TYPE = (_type: string) => {};",
      "const NOOP_SET_MODEL = (_model: string) => {};",
      "setProviderType={setProviderType || NOOP_SET_PROVIDER_TYPE}",
      "setModel={setModel || NOOP_SET_MODEL}",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/InputbarModelExtra.tsx",
    ]);
  });

  it("应禁止 InputbarTools 恢复旧可选工具状态 props", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-tools-legacy-optional-state-props",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "onToolClick?: (tool: string) => void;",
      "activeTools?: Record<string, boolean>;",
      "activeTools = {}",
      'onToolClick?.("thinking")',
      'onToolClick?.("web_search")',
      'onToolClick?.("subagent_mode")',
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/InputbarTools.tsx",
    ]);
  });

  it("应禁止 InputbarVisionCapabilityNotice 恢复旧可选模型 props", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "inputbar-vision-notice-legacy-optional-model-props",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "providerType?: string;",
      "model?: string;",
      "Boolean(providerType?.trim())",
      "Boolean(model?.trim())",
      "if (!shouldInspectCapability || !model?.trim())",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/InputbarVisionCapabilityNotice.tsx",
    ]);
  });

  it("应禁止 InputbarWorkflowStatusPanel 恢复旧数组 fallback", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "inputbar-workflow-status-panel-legacy-array-fallbacks",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "quickActions?: ThemeWorkbenchQuickAction[];",
      "queueItems?: ThemeWorkbenchWorkflowStep[];",
      "quickActions = []",
      "queueItems = []",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/InputbarWorkflowStatusPanel.tsx",
    ]);
  });

  it("应禁止 WorkflowRuntimeHost 被生产 Agent runtime 重新引用", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "workflow-runtime-host-production-usage",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(["WorkflowRuntimeHost"]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent",
      "src/features/plugin/runtime",
      "src/lib/api/agentRuntime",
      "packages/agent-ui-contracts",
      "packages/agent-runtime-projection",
      "packages/agent-runtime-ui",
      "packages/agent-runtime-client",
    ]);
  });

  it("应禁止 InputbarCore 恢复零调用的 allowEmptySend 透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-core-legacy-allow-empty-send-prop",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "allowEmptySend?: boolean;",
      "allowEmptySend={allowEmptySend}",
      "allowEmptySend = false,",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/InputbarCore.tsx",
    ]);
  });

  it("应禁止 InputbarCore 恢复零调用的 rightExtra 右侧插槽", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "inputbar-core-legacy-right-extra-slot",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "rightExtra?: React.ReactNode;",
      "Boolean(rightExtra)",
      "<MetaSlot>{rightExtra}</MetaSlot>",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/Inputbar/components/InputbarCore.tsx",
    ]);
  });

  it("应禁止首页空态恢复旧主题 tabs 壳", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "empty-state-legacy-theme-tabs-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(["showThemeTabs", "themeTabs={"]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyState.tsx",
      "src/components/agent/chat/components/EmptyStateHero.tsx",
      "src/components/agent/chat/workspace/chatSurfaceProps.ts",
    ]);
  });

  it("应禁止首页空态恢复旧项目选择器扩展透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "empty-state-legacy-project-selector-overrides",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "skipProjectSelectorWorkspaceReadyCheck",
      "deferProjectSelectorListLoad",
      "skipDefaultWorkspaceReadyCheck={",
      "deferProjectListLoad={",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyState.tsx",
    ]);
  });

  it("应禁止首页空态恢复 supportingSlotOverride 注入口", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "empty-state-supporting-slot-override-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(["supportingSlotOverride"]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyState.tsx",
      "src/components/agent/chat/components/EmptyState.test.tsx",
    ]);
  });

  it("应禁止首页链路恢复模型选择器预加载 props 透传", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "empty-state-legacy-model-selector-preload-prop-bridge",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "modelSelectorBackgroundPreload",
      "backgroundPreload={backgroundPreload}",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyState.tsx",
      "src/components/agent/chat/components/EmptyStateComposerPanel.tsx",
      "src/components/agent/chat/components/Inputbar/components/InputbarModelExtra.tsx",
    ]);
  });

  it("应禁止 @代码 专用 parser 文件恢复 parser 符号", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "agent-chat-code-workbench-command-parser-symbols",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "parseCodeWorkbenchCommand",
      "ParsedCodeWorkbenchCommand",
      "ParseCodeWorkbenchCommandOptions",
      "mentionCommandPrefixKeyMap",
      "parseMentionCommand",
      "code_runtime",
      "code_orchestrated",
      "codeOrchestratedDefaults",
      "code_orchestrated_defaults",
      "applyCodeOrchestratedDefaults",
      "harness.code_command",
      "@代码",
      "@code",
      "@coding",
      "@开发",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/utils/codeWorkbenchCommand.ts",
      "src/components/agent/chat/utils/codeWorkbenchCommand.test.ts",
    ]);
  });

  it("应禁止 Workspace 发送主链恢复 @代码 专用硬编码或正文关键词路由", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "agent-chat-code-workbench-workspace-send-hardcode",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "parseCodeWorkbenchCommand",
      'commandKey: "code_runtime"',
      'commandKey === "code_runtime"',
      "codeOrchestratedDefaults",
      "code_orchestrated_defaults",
      "applyCodeOrchestratedDefaults",
      "harness.code_command",
      "code_command:",
      '.includes("代码")',
      ".includes('代码')",
      '.includes("修复")',
      ".includes('修复')",
      '.includes("重构")',
      ".includes('重构')",
      '.includes("评审")',
      ".includes('评审')",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/workspace/useWorkspaceSendActions.ts",
    ]);
  });

  it("应禁止 Coding Workbench current 运行时直接消费 legacy thread item fact", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "agent-chat-coding-workbench-legacy-thread-item-facts",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "file_artifact",
      "command_execution",
      "approval_request",
      "code_orchestrated",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/workspace/codingSessionOverviewProjection.ts",
      "src/components/agent/chat/workspace/workspaceConversationCodingViews.tsx",
      "src/components/agent/chat/workspace/workspaceConversationSessionViewModel.ts",
      "src/components/agent/chat/workspace/useWorkspaceConversationSceneRuntime.tsx",
      "src/components/agent/chat/workspace/CodingWorkbenchOutputPanel.tsx",
      "src/components/agent/chat/workspace/CodingWorkbenchLogPanel.tsx",
      "src/components/agent/chat/workspace/CodingWorkbenchActionPanel.tsx",
      "src/components/agent/chat/workspace/CodingWorkbenchDiagnosticPanel.tsx",
      "src/components/agent/chat/components/CanvasSessionOverviewPanel.tsx",
      "src/components/agent/chat/components/CanvasWorkbenchLayout.tsx",
      "src/components/agent/chat/components/canvas-workbench/",
    ]);
  });

  it("应禁止 current scene view model 恢复 legacy thread item adapter", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "agent-chat-coding-workbench-legacy-adapter-reexport",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "workspaceConversationLegacyThreadItemAdapter",
    ]);
    expect(monitor?.regexPatterns).toEqual([
      "\\bexport\\s*\\{\\s*buildCanvasWorkbenchChangeView\\s*,\\s*buildCodingRuntimeEventsFromThreadItems\\s*,\\s*buildCodingWorkbenchProjectionFromThreadItems\\s*,\\s*buildFileArtifactChangeItem\\s*\\}\\s*from\\s*[\"']\\./workspaceConversationLegacyThreadItemAdapter[\"']",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/workspace/workspaceConversationSceneViewModel.ts",
      "src/components/agent/chat/workspace/workspaceConversationWorkbenchViewModel.ts",
    ]);
  });

  it("应禁止 current 编程入口把 code_orchestrated 当成现役 execution strategy", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "agent-chat-code-orchestrated-current-entry-ban",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([
      "src/lib/api/agentProtocol.test.ts",
      "src/components/agent/chat/skill-selection/runtimeInputCapabilityCatalog.test.tsx",
      "src/components/agent/chat/hooks/agentSessionRefresh.test.ts",
      "src/components/agent/chat/hooks/agentChatStorage.test.ts",
      "src/components/agent/chat/hooks/agentSessionTopicViewModel.unit.test.ts",
      "src/components/agent/chat/utils/sessionExecutionRuntime.test.ts",
    ]);
    expect(monitor?.patterns).toEqual([
      'execution_strategy: "code_orchestrated"',
      'executionStrategy: "code_orchestrated"',
      'effectiveExecutionStrategy: "code_orchestrated"',
      "code_orchestrated_defaults",
      "codeOrchestratedDefaults",
    ]);
  });

  it("应禁止 execution strategy 历史值归一逻辑散落在多个 current 边界", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id ===
        "agent-runtime-execution-strategy-compat-helper-single-boundary",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual([
      "src/lib/api/agentProtocol.ts",
      "src/components/agent/chat/skill-selection/runtimeInputCapabilityCatalog.ts",
      "src/lib/api/agentRuntime/executionStrategyCompat.ts",
    ]);
    expect(monitor?.allowedPaths).toEqual([
      "src/lib/api/agentRuntime/executionStrategyCompat.ts",
      "src/lib/api/agentRuntime/executionStrategyCompat.test.ts",
      "src/components/agent/chat/skill-selection/runtimeInputCapabilityCatalog.test.tsx",
    ]);
    expect(monitor?.patterns).toEqual(["code_orchestrated"]);
    expect(monitor?.regexPatterns).toEqual([
      "\\bnormalizeLegacyExecutionStrategyToCurrentReact\\s*\\(",
      "\\bnormalizeCatalogExecutionStrategyCompat\\s*\\(",
      "\\bfn\\s+normalize_execution_strategy\\s*\\(",
      "\\bnormalizeLegacyExecutionStrategy\\s*\\(",
      "\\bnormalizeLegacyCatalogExecutionStrategy\\s*\\(",
    ]);
  });

  it("应阻止已删除的 lime-agent execution strategy compat 模块回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-execution-strategy-compat",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs/crates/agent/src"]);
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "mod execution_strategy_compat;",
        "normalize_execution_strategy_to_react(",
      ]),
    );
  });

  it("应阻止已删除的 lime-agent subagent sidecar 回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-agent-subagent-sidecars",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs/crates/agent/src"]);
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "mod subagent_control;",
        "load_subagent_runtime_status(",
        "SubagentCustomizationState",
        "subagent_control.v0",
      ]),
    );
  });

  it("应阻止已删除的 Team/collab_agent 工具面回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-team-tool-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/tool-runtime/src",
      "lime-rs/crates/agent/src",
      "lime-rs/crates/core/src",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "pub mod collab_agent;",
        "collab_agent_tool_definitions(",
        'name: "Agent",',
        'name: "TeamCreate",',
        'canonical_name: "ListPeers",',
        "SUBAGENT_TEAMMATE_ALLOWED_TOOL_NAMES",
        "SUBAGENT_ALLOWED_NATIVE_TOOL_NAMES",
        "SUBAGENT_ALLOWED_COORDINATION_TOOL_NAMES",
        "runtime_turn_tool_exposure_allows_tool_name(",
      ]),
    );
  });

  it("应阻止 Renderer 本地 Team formation 与 synthetic dispatch preview 回流", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "agent-chat-retired-local-team-formation-preview",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "useRuntimeTeamFormation",
        "prepareRuntimeTeamBeforeSend",
        "team_formation_projection",
        "runtime-team-dispatch:",
      ]),
    );
    expect(monitor?.includePathPrefixes).toEqual(
      expect.arrayContaining([
        "packages/agent-ui-contracts/src",
        "packages/agent-runtime-projection",
        "src/components/agent/chat",
        "src/i18n/resources",
      ]),
    );
  });

  it("应阻止 Renderer Team runtime sidecar 与本地 live map 回流", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "agent-chat-retired-team-runtime-sidecar",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "team-workspace-runtime/",
        "useTeamWorkspaceRuntime",
        "teamWorkspaceRuntime",
        "restoredTeamFactsProjection",
        "liveRuntimeBySessionId",
        "liveActivityBySessionId",
        "agentChat.teamWorkspace.control.",
      ]),
    );
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat",
      "src/i18n/resources",
    ]);
    expect(monitor?.allowedPaths).toEqual([
      "src/components/agent/chat/AgentChatWorkspace.teamRuntimeBoundaryGuard.test.ts",
    ]);
  });

  it("应阻止已删除的 lime-agent aggregate execution runtime 回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-agent-runtime-aggregate",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs/crates/agent/src"]);
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "mod runtime_payload;",
        "build_session_execution_runtime(",
        "reconcile_session_execution_runtime_permission_fallback(",
        "pub struct SessionExecutionRuntime {",
        "LIME_RUNTIME_METADATA_KEY",
      ]),
    );
  });

  it("应阻止已删除的 lime-agent execution runtime session query 回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-agent-runtime-session-query",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs/crates/agent/src"]);
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "mod session_execution_runtime_query;",
        "read_session_execution_runtime_session_projection(",
        "project_session_record_execution_runtime_session(",
      ]),
    );
  });

  it("应禁止 Agent Runtime current 不可用报错重新指向 legacy command 恢复路径", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id ===
        "agent-runtime-current-unavailable-legacy-command-recovery-text",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "current Agent runtime cannot use legacy agent_runtime_* commands",
      "legacy agent_runtime_* commands",
    ]);
    expect(monitor?.includePathPrefixes).toEqual(["src/lib/api/agentRuntime"]);
  });

  it("应禁止 agent_messages 继续承接生产 transcript 写入", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-agent-messages-production-write-leak",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("deprecated");
    expect(monitor?.patterns).toEqual([
      "INSERT INTO agent_messages",
      "UPDATE agent_messages",
      "DELETE FROM agent_messages",
    ]);
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs/crates"]);
    expect(monitor?.allowedPaths).toEqual([
      "lime-rs/crates/app-server/src/local_data_source/legacy_message_backfill_source.rs",
      "lime-rs/crates/app-server/src/local_data_source/tests.rs",
      "lime-rs/crates/core/src/database/dao/agent.rs",
      "lime-rs/crates/core/src/database/dao/chat.rs",
      "lime-rs/crates/core/src/database/migration/general_chat_migration.rs",
    ]);
  });

  it("应禁止 agent_messages 成为产品读回长期 fallback", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-agent-messages-product-read-fallback-leak",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("deprecated");
    expect(monitor?.patterns).toEqual([
      "FROM agent_messages",
      "JOIN agent_messages",
      "FROM agent_messages m",
      "JOIN agent_messages m",
    ]);
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs/crates"]);
    expect(monitor?.allowedPaths).toEqual([
      "lime-rs/crates/app-server/src/local_data_source/legacy_message_backfill_source.rs",
      "lime-rs/crates/app-server/src/local_data_source/tests.rs",
      "lime-rs/crates/core/src/database/dao/agent.rs",
      "lime-rs/crates/core/src/database/dao/chat.rs",
      "lime-rs/crates/core/src/database/migration/general_chat_migration.rs",
      "lime-rs/crates/agent/src/agent_session_store/legacy_conversation.rs",
      "lime-rs/crates/agent/src/agent_session_store_tests.rs",
    ]);
  });

  it("应禁止产品服务层通过 ChatDao 旧消息 API 回读 agent_messages", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-chat-dao-agent-messages-product-api-leak",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual([
      "ChatDao::add_message",
      "ChatDao::delete_messages",
      "ChatDao::get_messages",
      "ChatDao::get_message_count",
      "ChatDao::get_session_detail",
      "ChatDao::list_sessions",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/app-server/src",
      "lime-rs/crates/server/src",
      "lime-rs/crates/services/src",
      "lime-rs/crates/websocket/src",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应禁止产品层通过 AgentDao 旧消息 API 回读 agent_messages", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-agent-session-direct-record-access",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.patterns).toEqual([
      "AgentDao::add_message(",
      "AgentDao::delete_messages(",
      "AgentDao::update_latest_assistant_message_usage(",
      "AgentDao::get_message_window_info(",
      "AgentDao::get_message_timestamp_by_id(",
      "AgentDao::get_session_with_messages(",
      "AgentDao::get_message_count(",
      "AgentDao::get_messages(",
      "AgentDao::get_messages_tail(",
      "AgentDao::get_messages_tail_page(",
      "AgentDao::get_messages_before(",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/agent/src",
      "lime-rs/crates/services/src",
      "lime-rs/crates/app-server/src",
      "lime-rs/crates/websocket/src",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应只在测试编译图保留 AgentDao 旧消息 API", () => {
    const agentDaoSource = readFileSync(
      join(REPO_ROOT, "lime-rs/crates/core/src/database/dao/agent.rs"),
      "utf8",
    );
    const legacyMethods = [
      "add_message",
      "delete_messages",
      "update_latest_assistant_message_usage",
      "get_message_window_info",
      "get_message_timestamp_by_id",
      "get_session_with_messages",
      "get_message_count",
      "get_messages",
      "get_messages_tail",
      "get_messages_tail_page",
      "get_messages_before",
    ];

    for (const method of legacyMethods) {
      expect(agentDaoSource).toMatch(
        new RegExp(
          `#\\[cfg\\(test\\)\\][\\s\\S]{0,160}pub fn ${method}\\s*\\(`,
        ),
      );
    }
  });

  it("应阻止已删除的 lime-agent session_store family 回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-agent-session-store-family",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs/crates/agent/src"]);
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "mod session_store;",
        "pub use session_store::{",
        "create_session_sync(",
        "get_runtime_session_detail(",
      ]),
    );
  });

  it("应禁止 agent_thread_items.payload_json 继续作为事实源", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-agent-thread-items-payload-json-truth-leak",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("deprecated");
    expect(monitor?.allowedPaths).toEqual([
      "lime-rs/crates/core/src/database/schema.rs",
      "lime-rs/crates/core/src/database/dao/agent_timeline.rs",
      "lime-rs/crates/core/src/database/dao/agent_timeline_payload.rs",
    ]);
    expect(monitor?.regexPatterns).toEqual([
      "agent_thread_items[\\s\\S]{0,1600}payload_json",
      "payload_json[\\s\\S]{0,1600}agent_thread_items",
    ]);
    expect(monitor?.includePathPrefixes).toEqual(["lime-rs/crates"]);
  });

  it("应禁止 runtime snapshot 裸字段和 sidecarRef 规则散落到非边界文件", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-runtime-snapshot-sidecar-ref-boundary-leak",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("deprecated");
    expect(monitor?.allowedPaths).toEqual([
      "lime-rs/crates/app-server/src/file_checkpoint.rs",
      "lime-rs/crates/app-server/src/file_checkpoint_snapshot.rs",
      "lime-rs/crates/app-server/src/media_task.rs",
      "lime-rs/crates/app-server/src/media_task/sidecar.rs",
      "lime-rs/crates/app-server/src/media_task/tests.rs",
      "lime-rs/crates/app-server/src/runtime/artifact_reader.rs",
      "lime-rs/crates/app-server/src/runtime/artifact_projection.rs",
      "lime-rs/crates/app-server/src/runtime/artifact_sidecar.rs",
      "lime-rs/crates/app-server/src/runtime/context_media.rs",
      "lime-rs/crates/app-server/src/runtime/context_packet.rs",
      "lime-rs/crates/app-server/src/runtime/evidence_provider.rs",
      "lime-rs/crates/app-server/src/runtime/file_checkpoint_projection.rs",
      "lime-rs/crates/app-server/src/runtime/output_refs.rs",
      "lime-rs/crates/app-server/src/runtime/output_refs/tests.rs",
      "lime-rs/crates/app-server/src/runtime/plugin_worker_runtime/tests.rs",
      "lime-rs/crates/app-server/src/runtime/thread_item_projection/media_result.rs",
      "lime-rs/crates/app-server/src/runtime/session_media_reader.rs",
      "lime-rs/crates/app-server/src/runtime/session_media_refs.rs",
    ]);
    expect(monitor?.patterns).toEqual([
      '"outputSnapshotFile"',
      '"checkpointSnapshotFile"',
      '"sidecarRef"',
      '"contentSha256"',
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/app-server/src",
    ]);
  });

  it("应禁止 app-data session fallback 和 hydration helper 回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-runtime-session-app-data-fallback",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.patterns).toEqual([
      "load_app_data_session",
      "read_agent_session(",
      "mod session_hydration;",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/app-server/src/runtime",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应禁止 generated artifact 正文继续内联在 runtime event 中", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) =>
        entry.id === "rust-runtime-artifact-generated-content-inline-leak",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("deprecated");
    expect(monitor?.patterns).toEqual([
      "generatedContent",
      "generated_content",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/app-server/src",
    ]);
    expect(monitor?.allowedPaths).toEqual([
      "lime-rs/crates/app-server/src/runtime/artifact_sidecar.rs",
      "lime-rs/crates/app-server/src/runtime/artifact_reader.rs",
      "lime-rs/crates/app-server/src/runtime/plugin_worker_runtime.rs",
      "lime-rs/crates/app-server/src/runtime/plugin_worker_runtime/response.rs",
      "lime-rs/crates/app-server/src/runtime/event_store.rs",
      "lime-rs/crates/app-server/src/runtime/tests/artifacts.rs",
      "lime-rs/crates/app-server/src/runtime/tests/evidence_exports.rs",
    ]);
  });

  it("应禁止 file checkpoint 正文继续内联在 runtime event 中", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) =>
        entry.id === "rust-runtime-file-checkpoint-inline-content-leak",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("deprecated");
    expect(monitor?.patterns).toEqual([
      "previousContent",
      "beforeContent",
      "oldContent",
    ]);
    expect(monitor?.regexPatterns).toEqual([
      "checkpointSnapshotFile[\\s\\S]{0,1600}(previousContent|beforeContent|oldContent)",
      "(previousContent|beforeContent|oldContent)[\\s\\S]{0,1600}checkpointSnapshotFile",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/app-server/src",
    ]);
    expect(monitor?.allowedPaths).toEqual([
      "lime-rs/crates/app-server/src/file_checkpoint.rs",
      "lime-rs/crates/app-server/src/file_checkpoint_snapshot.rs",
      "lime-rs/crates/app-server/src/runtime/file_checkpoint_projection.rs",
      "lime-rs/crates/app-server/src/runtime/evidence_provider.rs",
      "lime-rs/crates/app-server/src/runtime/tests/coding_events.rs",
      "lime-rs/crates/app-server/src/runtime/tests/evidence_exports.rs",
    ]);
  });

  it("应禁止 runtime store 自行发现平台路径", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-runtime-store-hardcoded-platform-path-leak",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "dirs::home_dir(",
      "dirs::data_dir(",
      "dirs::data_local_dir(",
      '"~/.lime"',
      '"/tmp/lime',
      '"APPDATA"',
      '"LOCALAPPDATA"',
      '"Application Support"',
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/app-server/src/runtime",
      "lime-rs/crates/app-server/src/main.rs",
      "lime-rs/crates/app-server/src/file_checkpoint_snapshot.rs",
    ]);
  });

  it("应禁止首页空态恢复 configLoadStrategy 时序开关", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "empty-state-legacy-config-load-strategy-prop",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "configLoadStrategy",
      "scheduleDeferredConfigLoad",
    ]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyState.tsx",
    ]);
  });

  it("应禁止 EmptyState 重新直读本地 activeSkill hook 状态", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "empty-state-local-active-skill-hook-usage",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.regexPatterns).toEqual([
      "(?<!\\.)\\bactiveSkill\\s*[,}]",
      "(?<!\\.)\\bclearActiveSkill\\s*[,}]",
    ]);
  });

  it("应禁止技能入口旧展示文案重新回到页面层手写", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "skill-selection-legacy-display-copy",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead-candidate");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.includePathPrefixes).toEqual([
      "src/components/agent/chat/components/EmptyState.tsx",
      "src/components/agent/chat/skill-selection/SkillSelector.tsx",
    ]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "当前技能 ",
        "当前已启用 ",
        "为当前任务挂载额外能力",
        "按需挂载能力",
        "项技能可用",
      ]),
    );
  });

  it("应禁止旧海报素材入口与 Rust 符号重新回流", () => {
    const importMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "gallery-material-legacy-frontend-module",
    );
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "gallery-material-legacy-helper-usage",
    );
    const rustMonitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-gallery-material-legacy-symbols",
    );

    expect(importMonitor).toBeTruthy();
    expect(importMonitor?.allowedPaths).toEqual([]);
    expect(importMonitor?.targets).toEqual(
      expect.arrayContaining([
        "src/lib/api/posterMaterials.ts",
        "src/hooks/usePosterMaterial.ts",
        "src/types/poster-material.ts",
      ]),
    );

    expect(frontendMonitor).toBeTruthy();
    expect(frontendMonitor?.classification).toBe("dead-candidate");
    expect(frontendMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "getPosterMaterial(",
        "createPosterMetadata(",
        "usePosterMaterial(",
      ]),
    );

    expect(rustMonitor).toBeTruthy();
    expect(rustMonitor?.allowedPaths).toEqual([
      "lime-rs/crates/core/src/database/schema.rs",
    ]);
    expect(rustMonitor?.patterns).toEqual(
      expect.arrayContaining([
        "PosterMaterialDao",
        "poster_material_metadata",
        "idx_poster_material_metadata_",
      ]),
    );
  });

  it("应记录 API Key Provider current App Server 方法目录", () => {
    expect(agentCommandCatalog.appServerModelProviderMethods).toEqual([
      "modelProvider/list",
      "modelProvider/catalog/list",
      "modelProvider/read",
      "modelProvider/create",
      "modelProvider/update",
      "modelProvider/delete",
      "modelProvider/sortOrders/update",
      "modelProviderConfig/export",
      "modelProviderConfig/import",
      "modelProvider/testConnection",
      "modelProvider/testChat",
      "modelProvider/fetchModels",
      "modelProviderKey/create",
      "modelProviderKey/update",
      "modelProviderKey/delete",
      "modelProviderUiState/read",
      "modelProviderUiState/write",
    ]);
  });

  it("应记录 Plugin 应用中心 current App Server 方法目录", () => {
    expect(agentCommandCatalog.appServerPluginMethods).toEqual([
      "pluginLocalPackage/inspect",
      "pluginLocalPackage/export",
      "pluginPackage/fetchCloud",
      "pluginInstalled/save",
      "pluginInstalled/list",
      "pluginInstalled/disabled/set",
      "pluginInstalled/uninstall/rehearsal",
      "pluginInstalled/uninstall",
      "pluginHostLifecycle/list",
      "pluginShell/prepare",
      "pluginUiRuntime/start",
      "pluginUiRuntime/status",
      "pluginUiRuntime/stop",
    ]);
  });

  it("应记录 Browser Session current App Server 方法目录", () => {
    expect(agentCommandCatalog.appServerBrowserSessionMethods).toEqual([
      "browserSession/target/list",
      "browserSession/open",
      "browserSession/read",
      "browserSession/close",
      "browserSession/event/list",
      "browserSession/action/execute",
    ]);
  });

  it("旧 Plugin lifecycle Tauri 命令不应继续作为 runtime gateway current surface", () => {
    expect(agentCommandCatalog.runtimeGatewayCommands).not.toEqual(
      expect.arrayContaining([
        "plugin_inspect_local_package",
        "plugin_fetch_cloud_package",
        "plugin_save_installed_state",
        "plugin_list_installed",
        "plugin_set_disabled",
        "plugin_uninstall_rehearsal",
        "plugin_uninstall",
      ]),
    );
    expect(agentCommandCatalog.deprecatedCommandReplacements).toMatchObject({
      plugin_inspect_local_package: "pluginLocalPackage/inspect",
      plugin_fetch_cloud_package: "pluginPackage/fetchCloud",
      plugin_save_installed_state: "pluginInstalled/save",
      plugin_list_installed: "pluginInstalled/list",
      plugin_set_disabled: "pluginInstalled/disabled/set",
      plugin_uninstall_rehearsal: "pluginInstalled/uninstall/rehearsal",
      plugin_uninstall: "pluginInstalled/uninstall",
    });
  });

  it("应记录旧 API Key Provider 命令到 current 方法的替换关系", () => {
    expect(agentCommandCatalog.deprecatedCommandReplacements).toMatchObject({
      get_api_key_providers: "modelProvider/list",
      get_system_provider_catalog: "modelProvider/catalog/list",
      get_api_key_provider: "modelProvider/read",
      create_api_key_provider: "modelProvider/create",
      update_api_key_provider: "modelProvider/update",
      delete_api_key_provider: "modelProvider/delete",
      update_api_key_provider_sort_orders: "modelProvider/sortOrders/update",
      export_api_key_provider_config: "modelProviderConfig/export",
      import_api_key_provider_config: "modelProviderConfig/import",
      test_api_key_provider_connection: "modelProvider/testConnection",
      test_api_key_provider_chat: "modelProvider/testChat",
      fetch_provider_models_auto: "modelProvider/fetchModels",
      create_api_key_provider_key: "modelProviderKey/create",
      update_api_key_provider_key: "modelProviderKey/update",
      delete_api_key_provider_key: "modelProviderKey/delete",
      get_provider_ui_state: "modelProviderUiState/read",
      set_provider_ui_state: "modelProviderUiState/write",
    });
  });

  it("应将旧 API Key Provider command surface 标记为 dead", () => {
    const monitor = legacySurfaceCatalogJson.commands.find(
      (entry) => entry.id === "api-key-provider-legacy-command-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.commands).toEqual(
      expect.arrayContaining([
        "get_api_key_providers",
        "get_system_provider_catalog",
        "get_api_key_provider",
        "create_api_key_provider",
        "update_api_key_provider",
        "delete_api_key_provider",
        "update_api_key_provider_sort_orders",
        "export_api_key_provider_config",
        "import_api_key_provider_config",
        "test_api_key_provider_connection",
        "test_api_key_provider_chat",
        "fetch_provider_models_auto",
        "create_api_key_provider_key",
        "update_api_key_provider_key",
        "delete_api_key_provider_key",
        "next_api_key_provider_key",
        "record_api_key_provider_key_usage",
        "record_api_key_provider_key_error",
        "get_provider_ui_state",
        "set_provider_ui_state",
      ]),
    );
  });

  it("应阻止已删除的 Provider key telemetry JSON-RPC 回流", () => {
    const frontendMonitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "frontend-retired-model-provider-key-telemetry-methods",
    );
    const rustMonitor = legacySurfaceCatalogJson.rustText.find(
      (entry) =>
        entry.id === "rust-retired-model-provider-key-telemetry-methods",
    );

    for (const monitor of [frontendMonitor, rustMonitor]) {
      expect(monitor).toBeTruthy();
      expect(monitor?.classification).toBe("dead");
      expect(monitor?.allowedPaths).toEqual([]);
      expect(monitor?.patterns).toEqual([
        "modelProviderKey/next",
        "modelProviderKey/usage/record",
        "modelProviderKey/error/record",
      ]);
    }
  });

  it("应阻止旧视频生成 service、DAO 与 Product DB 表回流", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) => entry.id === "rust-retired-video-generation-database-surface",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/core",
      "lime-rs/crates/services",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual([
      "pub mod video_generation_service;",
      "pub mod video_generation_task_dao;",
      "VideoGenerationService",
      "VideoGenerationTaskDao",
      "video_generation_tasks",
    ]);
  });

  it("应阻止媒体任务恢复固定本机 HTTP broker execution owner", () => {
    const monitor = legacySurfaceCatalogJson.rustText.find(
      (entry) =>
        entry.id === "rust-retired-local-media-service-execution-contract",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.patterns).toEqual(["local_lime_service"]);
    expect(monitor?.includePathPrefixes).toEqual([
      "lime-rs/crates/app-server/src",
      "lime-rs/crates/media-runtime/src",
      "lime-rs/crates/media-runtime/tests",
    ]);
    expect(monitor?.allowedPaths).toEqual([]);
  });

  it("应将旧 content workflow 命令面固定为 dead", () => {
    const apiMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "content-workflow-legacy-api-gateway",
    );
    const rustMonitor = legacySurfaceCatalogJson.imports.find(
      (entry) => entry.id === "content-workflow-rust-command-module",
    );
    const commandMonitor = legacySurfaceCatalogJson.commands.find(
      (entry) => entry.id === "content-workflow-legacy-commands",
    );

    expect(apiMonitor).toBeTruthy();
    expect(apiMonitor?.classification).toBe("dead");
    expect(apiMonitor?.targets).toEqual(["src/lib/api/content-workflow.ts"]);
    expect(apiMonitor?.allowedPaths).toEqual([]);

    expect(rustMonitor).toBeTruthy();
    expect(rustMonitor?.classification).toBe("dead");
    expect(rustMonitor?.targets).toEqual([
      "lime-rs/src/commands/content_workflow_cmd.rs",
    ]);
    expect(rustMonitor?.allowedPaths).toEqual([]);

    expect(commandMonitor).toBeTruthy();
    expect(commandMonitor?.classification).toBe("dead");
    expect(commandMonitor?.commands).toEqual(
      expect.arrayContaining([
        "content_workflow_create",
        "content_workflow_get",
        "content_workflow_get_by_content",
        "content_workflow_advance",
        "content_workflow_retry",
        "content_workflow_cancel",
      ]),
    );
    expect(commandMonitor?.allowedPaths).toEqual([]);
  });

  it("应禁止 Coding Workbench current session overview 重新直读旧 thread item facts", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) =>
        entry.id === "agent-chat-coding-workbench-legacy-thread-item-facts",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.allowedPaths).toEqual([]);
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        "file_artifact",
        "command_execution",
        "approval_request",
        "code_orchestrated",
      ]),
    );
    expect(monitor?.includePathPrefixes).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/components/CanvasSessionOverviewPanel.tsx",
        "src/components/agent/chat/workspace/codingSessionOverviewProjection.ts",
        "src/components/agent/chat/workspace/workspaceConversationCodingViews.tsx",
      ]),
    );
  });

  it("Coding roadmap 不应再把 code_orchestrated 写成 current 入口", () => {
    const roadmapFiles = [
      "internal/roadmap/coding/README.md",
      "internal/roadmap/coding/architecture.md",
      "internal/roadmap/coding/implementation-plan.md",
    ];
    const forbiddenSnippets = [
      "compat/current 入口语义",
      "作为 coding profile 的现有入口语义",
      "作为 profile selection 输入",
    ];
    const offenders = roadmapFiles.flatMap((file) => {
      const source = readFileSync(join(REPO_ROOT, file), "utf8");
      return forbiddenSnippets
        .filter((snippet) => source.includes(snippet))
        .map((snippet) => `${file}: ${snippet}`);
    });

    expect(offenders).toEqual([]);
  });

  it("治理目录册不应把 code_orchestrated runtime 写成 current 编程底座", () => {
    const source = readFileSync(
      join(REPO_ROOT, "src/lib/governance/legacySurfaceCatalog.json"),
      "utf8",
    );

    expect(source).not.toContain("code_orchestrated runtime");
    expect(source).toContain(
      "legacy code_orchestrated 只能在兼容边界归一到 react",
    );
  });

  it("旧 Product Profile 右栏 key 不应回流到 Article Workspace current surface", () => {
    const monitor = legacySurfaceCatalogJson.frontendText.find(
      (entry) => entry.id === "agent-chat-product-profile-current-surface-key",
    );

    expect(monitor).toBeTruthy();
    expect(monitor?.classification).toBe("dead");
    expect(monitor?.patterns).toEqual(
      expect.arrayContaining([
        '"productProfile"',
        "productProfileAvailable",
        "productProfileEnabled",
        "PluginProductProfile",
        "agentChat.navbar.productProfile",
        "agentChat.rightSurface.tabs.productProfile",
        "plugin.apps.center.host.productProfile",
      ]),
    );
    expect(monitor?.allowedPaths).toEqual(
      expect.arrayContaining([
        "src/components/agent/chat/workspace/right-surface/rightSurfaceTypes.ts",
        "src/features/plugin/history/pluginHistoryRestore.ts",
      ]),
    );
  });

  it("Coding roadmap 不应把已完成的 P5/P7/P8 baseline 重新写成主线 blocker", () => {
    const roadmapFiles = [
      "internal/roadmap/coding/README.md",
      "internal/roadmap/coding/implementation-plan.md",
    ];
    const forbiddenSnippets = [
      "P5 多模型 slot、P7 GUI smoke 未完成前",
      "多模型 slot      | 文档定义完成，工程实现未完成",
      "继续补 log/action-submit 面板接线和 UI smoke evidence",
      "日志、审批提交动作继续迁到 projection adapter，并补 GUI smoke",
      "补 code artifact workbench fixture / GUI smoke",
      "接 Provider slot diagnostics",
      "P3/P4 UI evidence、Provider slot diagnostics 仍按后续主线推进",
      "当前最优先做 P8 收口，而不是继续重复 P2-P7 baseline",
      "最高价值下一刀转为 P8 legacy residual 收口",
    ];
    const offenders = roadmapFiles.flatMap((file) => {
      const source = readFileSync(join(REPO_ROOT, file), "utf8");
      return forbiddenSnippets
        .filter((snippet) => source.includes(snippet))
        .map((snippet) => `${file}: ${snippet}`);
    });
    const readmeSource = readFileSync(
      join(REPO_ROOT, "internal/roadmap/coding/README.md"),
      "utf8",
    );
    const implementationSource = readFileSync(
      join(REPO_ROOT, "internal/roadmap/coding/implementation-plan.md"),
      "utf8",
    );

    expect(offenders).toEqual([]);
    expect(readmeSource).toContain(
      "P1/P2/P3/P4/P5/P7/P8 的骨架闭环已经具备 current facts、projection、GUI smoke、evidence export 与生产 mock / legacy command 防回流守卫",
    );
    expect(readmeSource).toContain(
      "推进 Windows restricted token ACL / token enforcement，补齐 Windows 平台 sandbox 真实 enforce",
    );
    expect(implementationSource).toContain(
      "P8 residual 盘点结论：生产 `src / packages / electron` 主路径未发现 `agent_runtime_*` 直接命令调用",
    );
    expect(implementationSource).toContain(
      "P1/P2/P3/P4/P5/P7/P8 骨架已经进入 App Server / RuntimeCore current crates 与 Workbench current projection",
    );
  });
});
