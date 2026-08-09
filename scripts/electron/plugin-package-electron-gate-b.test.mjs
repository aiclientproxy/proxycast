import fs from "node:fs";
import { describe, expect, it } from "vitest";
import { pluginPackageMcpAppSurfaceReady } from "./plugin-mcp-app-gate-b.mjs";

describe("Plugin package current Electron fixture guard", () => {
  it("uses the real Plugin package install and runtime MCP lifecycle", () => {
    const wrapper = fs.readFileSync(
      "scripts/electron/plugin-package-electron-gate-b.mjs",
      "utf8",
    );
    const gate = fs.readFileSync(
      "scripts/electron/mcp-elicitation-gate-b.mjs",
      "utf8",
    );
    const fixture = fs.readFileSync(
      "scripts/electron/mcp-config-fixture-smoke.mjs",
      "utf8",
    );
    const mcpAppGate = fs.readFileSync(
      "scripts/electron/plugin-mcp-app-gate-b.mjs",
      "utf8",
    );

    expect(wrapper).toContain("run({ pluginPackage: true })");
    expect(gate).toContain("ensureElectronFixtureBuild");
    expect(gate).toContain("rootDir: process.cwd()");
    expect(gate).toContain("--electron-executable");
    expect(gate).toContain("options.electronExecutable");
    expect(gate).toContain('{ APP_SERVER_BIN: "" }');
    expect(gate).toContain("summary.electronPackaged");
    expect(gate).toContain("summary.packagedElectronRequested");
    expect(fixture).toContain("const packagedExecutable =");
    expect(fixture).toContain(
      "executablePath: packagedExecutable || electronPath",
    );
    expect(fixture).toMatch(
      /args:\s*packagedExecutable\s*\?\s*\["--use-mock-keychain"\]/u,
    );
    expect(gate).toContain(
      '"https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"',
    );
    expect(gate).toContain(
      '"https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"',
    );
    expect(gate).toContain('path.join(fixture.root, "plugin.json")');
    expect(gate).toContain('path.join(fixture.root, "mcp.json")');
    expect(gate).toContain('"skills",');
    expect(gate).toContain("PLUGIN_PACKAGE_SKILL_ID");
    expect(gate).not.toContain(".codex-plugin/plugin.json");
    expect(gate).not.toContain(".mcp.json");
    expect(gate).toContain("summary.standardManifestSeen");
    expect(gate).toContain("summary.standardMcpConfigSeen");
    expect(gate).toContain("summary.pluginSkillProjected");
    expect(gate).toContain("summary.pluginSkillContextSeen");
    expect(gate).toContain("PLUGIN_PACKAGE_SKILL_BODY_MARKER");
    expect(gate).toContain("dynamicToolRequired: !pluginPackage");
    expect(gate).toContain("if (!pluginPackage)");
    expect(gate).toContain('"plugin/list"');
    expect(gate).toContain('data-testid="app-sidebar-nav-plugins"');
    expect(gate).toContain('data-testid="plugin-catalog-install-local"');
    expect(gate).toContain('data-testid="plugin-catalog-install-review"');
    expect(gate).toContain('data-testid="plugin-catalog-confirm-install"');
    expect(gate).toContain("armControlledPluginSourceDialog");
    expect(gate).toContain("installPluginPackageFromAppCenter");
    expect(gate).toContain('"plugin/install"');
    expect(gate).toContain('"plugin/read"');
    expect(gate).toContain('"plugin/installed"');
    expect(gate).toContain('"plugin/enabled/set"');
    expect(gate).toContain('data-testid="inputbar-plus-plugins"');
    expect(gate).toContain('data-testid="inputbar-plugin-option"');
    expect(gate).toContain('data-testid="inputbar-plugin-badge"');
    expect(gate).toContain('data-testid="model-selector"');
    expect(gate).toContain('data-model-selector-popover="true"');
    expect(gate).toContain("selectRuntimeRouteInRenderer");
    expect(gate).toContain("findPluginPackageMentionIdentity");
    expect(gate).toContain("plugin://${PLUGIN_PACKAGE_CONFIG_NAME}");
    expect(gate).toContain('data-testid="plugin-catalog-uninstall-confirm"');
    expect(gate).toContain('"plugin/uninstall"');
    expect(gate).toContain("historyReadableAfterUninstall");
    expect(gate).toContain("mcpAppHistoryUnavailableAfterUninstall");
    expect(gate).toContain("historyMcpRuntimeNotRestartedAfterUninstall");
    expect(gate).toContain("historyProviderNotReexecutedAfterUninstall");
    expect(gate).not.toContain("async function installPluginPackage(");
    expect(gate).toContain("pluginPackageServerName()");
    expect(gate).toContain("managementConnectionAbsent");
    expect(gate).toContain("Plugin package MCP 不得通过管理面");
    expect(gate).toContain('"mcpServer/resource/read"');
    expect(gate).toContain('method === "resources/read"');
    expect(gate).toContain('mimeType: "text/html;profile=mcp-app"');
    expect(gate).toContain("waitForPluginPackageMcpAppSurface");
    expect(gate).toContain("mcpAppRestoredAfterReload");
    expect(gate).toContain("cold-restart-plugin-package-history");
    expect(gate).toContain("switchRuntimeThreadInGuiViaSidebar");
    expect(gate).toContain("waitForRendererHistoryHydrationAfterClick");
    expect(gate).toContain("waitForRendererSessionSwitchSuccessAfterClick");
    expect(gate).toContain('entry?.phase === "session.switch.success"');
    expect(gate).toContain('"thread/items/list"');
    expect(gate).toContain('"thread/turns/list"');
    expect(gate).toContain('data-testid="app-sidebar-conversation-open"');
    expect(gate).toContain("sourceSessionId: summary.disabledNewSessionId");
    expect(gate).toContain("summary.electronLaunchCount === 2");
    expect(gate).toContain("summary.coldRestoreCompleted");
    expect(gate).toContain("summary.coldRestoreIdentityStable");
    expect(gate).toContain("summary.coldRestoreExplicitSurfaceOpen");
    expect(gate).toContain("summary.coldRestoreCanonicalHydrationViaSidebar");
    expect(gate).toContain("summary.coldRestoreMcpProcessRestarted");
    expect(gate).toContain("summary.coldRestoreProviderNotReexecuted");
    expect(gate).toContain("summary.coldRestoreToolNotReexecuted");
    expect(gate).toContain("summary.mcpAppResourceReadCount >= 4");
    expect(gate).toContain(
      "summary.mcpAppHtmlLoadCount === summary.mcpAppResourceReadCount",
    );
    expect(gate).toContain("summary.mcpAppToolCallCount === 1");
    expect(gate).toContain("repositoryCommit: readRepositoryCommit");
    expect(gate).toContain("summary.appVersion = electronRuntime.appVersion");
    expect(gate).toContain("summary.pluginMarketplaceId");
    expect(gate).toContain("summary.pluginContentDigest");
    expect(gate).toContain("verifyPluginPackageDisabledNewThreadBoundary");
    expect(gate).toContain("summary.disabledBoundaryCompleted");
    expect(gate).toContain(
      "summary.disabledNewSessionId = disabledBoundary.sessionId",
    );
    expect(gate).toContain("summary.disabledPluginPickerBlocked");
    expect(gate).toContain("summary.disabledProviderToolAbsent");
    expect(gate).toContain("summary.disabledThreadPluginItemsAbsent");
    expect(gate).toContain("summary.enabledNewBoundaryCapabilityRestored");
    expect(gate).toContain("summary.sessionId = runtime.sessionId");
    expect(gate).toContain("summary.threadId = runtime.threadId");
    expect(gate).toContain("summary.turnId = runtime.turnId");
    expect(gate).toContain("summary.userItemId = runtime.userItemId");
    expect(gate).toContain("summary.toolItemId = runtime.mcpAppItemId");
    expect(gate).toContain("summary.toolCallId === MCP_TOOL_CALL_ID");
    expect(gate).toContain("summary.surfaceId = firstSurface.viewId");
    expect(gate).toContain("summary.completedAt = new Date().toISOString()");
    expect(mcpAppGate).toContain("embedded_browser_view_load_html");
    expect(mcpAppGate).toContain(
      "request?.params?.threadId === runtime.threadId",
    );
    expect(mcpAppGate).toContain("request?.params?.sessionId === undefined");
    expect(mcpAppGate).toContain("args_preview?.request?.lines");
    expect(mcpAppGate).toMatch(/webContents\s*\.getAllWebContents\(\)/u);
    expect(mcpAppGate).not.toContain(
      '.filter((entry) => entry.getType() !== "window")',
    );
    expect(mcpAppGate).toContain("BrowserWindow.getAllWindows()");
    expect(mcpAppGate).toContain("browserWindowWebContentsIds.has(entry.id)");
    expect(mcpAppGate).toContain("Promise.race([");
    expect(gate).toContain('backendMode: "runtime"');
    expect(gate).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
  });

  it("waits for the cumulative MCP App lifecycle before exact-count assertions", () => {
    expect(
      pluginPackageMcpAppSurfaceReady(
        { resourceReadCount: 1, htmlLoadCount: 1 },
        2,
        2,
      ),
    ).toBe(false);
    expect(
      pluginPackageMcpAppSurfaceReady(
        { resourceReadCount: 2, htmlLoadCount: 2 },
        2,
        2,
      ),
    ).toBe(true);
    expect(
      pluginPackageMcpAppSurfaceReady(
        { resourceReadCount: 3, htmlLoadCount: 2 },
        2,
        2,
      ),
    ).toBe(true);
  });
});
