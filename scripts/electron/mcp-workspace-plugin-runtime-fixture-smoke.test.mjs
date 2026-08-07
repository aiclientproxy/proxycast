import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readSmokeScript() {
  return fs.readFileSync(
    "scripts/electron/mcp-workspace-plugin-runtime-fixture-smoke.mjs",
    "utf8",
  );
}

describe("MCP Workspace plugin runtime Electron fixture smoke guard", () => {
  it("keeps the smoke on real Electron Desktop Host IPC and App Server JSON-RPC", () => {
    const content = readSmokeScript();

    expect(content).toContain("launchElectronFixture");
    expect(content).toContain("APP_SERVER_HANDLE_JSON_LINES_COMMAND");
    expect(content).toContain("app_server_handle_json_lines");
    expect(content).toContain("window.electronAPI?.invoke");
    expect(content).toContain("electronPreloadBridge");
    expect(content).toContain('backendMode: "runtime"');
    expect(content).toContain("Gate B skeleton");
    expect(content).toContain("claimBoundary");
  });

  it("drives a Workspace Harness prepare click across current MCP methods", () => {
    const content = readSmokeScript();

    expect(content).toContain("harness-status-panel");
    expect(content).toContain("工具与权限");
    expect(content).toContain("准备 MCP");
    expect(content).toContain("plugin_runtime_capabilities");
    expect(content).toContain("appServerCommand: command");
    expect(content).toContain("requestMethods:");
    expect(content).toContain("workspace-harness-page-result");
    expect(content).toContain('const PLUGIN_ID = "mcp-current-plugin"');
    expect(content).toContain('"agentSession/toolInventory/read"');
    expect(content).toContain('"mcpServer/start"');
    expect(content).toContain('"mcpTool/listForContext"');
    expect(content).toContain('"mcpServer/tool/call"');
    expect(content).toContain("ephemeral: true");
    expect(content).not.toContain('modelProvider: "openai"');
    expect(content).toContain("defaultProofDidNotCallTool");
    expect(content).toContain("LEGACY_MCP_COMMANDS");
  });

  it("does not use production mock fallbacks or legacy MCP facade as positive evidence", () => {
    const content = readSmokeScript();
    const requiredMethodsBlock = content.slice(
      content.indexOf("const REQUIRED_METHODS"),
      content.indexOf("function printHelp"),
    );

    expect(requiredMethodsBlock).not.toContain("get_mcp_servers");
    expect(requiredMethodsBlock).not.toContain("mcp_list_tools");
    expect(requiredMethodsBlock).not.toContain("mcp_call_tool");
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
  });
});
