import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readSmokeScript() {
  return fs.readFileSync(
    "scripts/electron/mcp-context7-live-fixture-smoke.mjs",
    "utf8",
  );
}

describe("MCP Context7 live Electron fixture smoke guard", () => {
  it("keeps live Context7 validation on Electron IPC and App Server JSON-RPC", () => {
    const content = readSmokeScript();

    expect(content).toContain("launchElectronFixture");
    expect(content).toContain("openMcpConfigSettings");
    expect(content).toContain("createContext7ConfigFromGui");
    expect(content).toContain("APP_SERVER_HANDLE_JSON_LINES_COMMAND");
    expect(content).toContain("app_server_handle_json_lines");
    expect(content).toContain('"mcpServer/start"');
    expect(content).toContain('"mcpServerStatus/list"');
    expect(content).toContain('"mcpTool/search"');
    expect(content).toContain('"mcpServer/tool/call"');
  });

  it("calls Context7 tools without recording secrets or using mock fallback", () => {
    const content = readSmokeScript();

    expect(content).toContain("https://mcp.context7.com/mcp");
    expect(content).toContain("resolve-library-id");
    expect(content).toContain("query-docs");
    expect(content).toContain("CONTEXT7_API_KEY");
    expect(content).toContain("context7ApiKeyEnvPresent");
    expect(content).toContain("LEGACY_MCP_COMMANDS");
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
    expect(content).not.toContain("ctx7sk-");
  });

  it("does not make legacy MCP facade calls as positive evidence", () => {
    const content = readSmokeScript();
    const positiveCallSurface = content.slice(
      content.indexOf("const REQUIRED_METHODS"),
      content.indexOf("function printHelp"),
    );

    expect(positiveCallSurface).not.toContain("get_mcp_servers");
    expect(positiveCallSurface).not.toContain("mcp_start_server");
    expect(positiveCallSurface).not.toContain("add_mcp_server");
  });

  it("does not execute the live fixture when imported as a helper module", () => {
    const content = readSmokeScript();

    expect(content).toContain('import { pathToFileURL } from "node:url"');
    expect(content).toContain(
      'import.meta.url === pathToFileURL(process.argv[1] || "").href',
    );
  });
});
