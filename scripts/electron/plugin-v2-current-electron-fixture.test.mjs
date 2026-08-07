import fs from "node:fs";
import { describe, expect, it } from "vitest";

describe("Plugin v2 current Electron fixture guard", () => {
  it("uses the real Plugin v2 install and runtime MCP lifecycle", () => {
    const wrapper = fs.readFileSync(
      "scripts/electron/plugin-v2-current-electron-fixture.mjs",
      "utf8",
    );
    const gate = fs.readFileSync(
      "scripts/electron/mcp-elicitation-gate-b.mjs",
      "utf8",
    );
    const mcpAppGate = fs.readFileSync(
      "scripts/electron/plugin-v2-mcp-app-gate-b.mjs",
      "utf8",
    );

    expect(wrapper).toContain("run({ pluginV2: true })");
    expect(gate).toContain("ensureElectronFixtureBuild");
    expect(gate).toContain("rootDir: process.cwd()");
    expect(gate).toContain('mcpServers: "./.mcp.json"');
    expect(gate).toContain('"plugin/list"');
    expect(gate).toContain('data-testid="app-sidebar-nav-plugins"');
    expect(gate).toContain('data-testid="plugin-v2-install-local"');
    expect(gate).toContain('data-testid="plugin-v2-install-review"');
    expect(gate).toContain('data-testid="plugin-v2-confirm-install"');
    expect(gate).toContain("armControlledPluginSourceDialog");
    expect(gate).toContain("installPluginV2FromAppCenter");
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
    expect(gate).toContain("findPluginV2MentionIdentity");
    expect(gate).toContain("plugin://${PLUGIN_V2_ID}");
    expect(gate).toContain('data-testid="plugin-v2-uninstall-confirm"');
    expect(gate).toContain('"plugin/uninstall"');
    expect(gate).toContain("historyReadableAfterUninstall");
    expect(gate).toContain("mcpAppHistoryUnavailableAfterUninstall");
    expect(gate).toContain("historyMcpRuntimeNotRestartedAfterUninstall");
    expect(gate).toContain("historyProviderNotReexecutedAfterUninstall");
    expect(gate).not.toContain("async function installPluginV2(");
    expect(gate).toContain("pluginV2ServerName()");
    expect(gate).toContain("managementConnectionAbsent");
    expect(gate).toContain("Plugin v2 MCP 不得通过管理面");
    expect(gate).toContain('"mcpServer/resource/read"');
    expect(gate).toContain('method === "resources/read"');
    expect(gate).toContain('mimeType: "text/html;profile=mcp-app"');
    expect(gate).toContain("waitForPluginV2McpAppSurface");
    expect(gate).toContain("mcpAppRestoredAfterReload");
    expect(gate).toContain("cold-restart-plugin-v2-history");
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
    expect(gate).toContain("verifyPluginV2DisabledNewThreadBoundary");
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
    expect(mcpAppGate).toContain(
      "request?.params?.sessionId === undefined",
    );
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
});
