import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readGateB() {
  return [
    "scripts/electron/orchestrator-skills-gate-b.mjs",
    "scripts/electron/lib/orchestrator-skills-gate-b-core.mjs",
  ]
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
}

describe("Orchestrator Skills/MCP Electron Gate B guard", () => {
  it("keeps remote Skill discovery and read on the real current product chain", () => {
    const content = readGateB();

    expect(content).toContain('backendMode: "runtime"');
    expect(content).toContain("ensureElectronFixtureBuild");
    expect(content).toContain("launchElectronFixture");
    expect(content).toContain("APP_SERVER_HANDLE_JSON_LINES_COMMAND");
    expect(content).toContain('"turn/start"');
    expect(content).toContain('"thread/read"');
    expect(content).toContain("guiFinalVisible: true");
    expect(content).toContain("read.guiFinalVisible === true");
    expect(content).toContain("startOpenAiCompatibleFixtureServer");
    expect(content).toContain("name: SKILL_SEARCH_TOOL_NAME");
    expect(content).toContain("name: SKILL_READ_TOOL_NAME");
    expect(content).toContain(
      'SKILL_PACKAGE_URI = "skill://delivery/release-notes"',
    );
    expect(content).toContain(
      "SKILL_RESOURCE_URI = `${SKILL_PACKAGE_URI}/SKILL.md`",
    );
    expect(content).toContain('method === "resources/list"');
    expect(content).toContain('method === "resources/read"');
    expect(content).toContain('mimeType: "mcp/skill"');
    expect(content).toContain("frozenTurnResourceListCount === 1");
  });

  it("proves the disabled Apps boundary without hiding ordinary MCP", () => {
    const content = readGateB();

    expect(content).toContain('APPS_SERVER_NAME = "codex_apps"');
    expect(content).toContain('ORDINARY_SERVER_NAME = "ordinary_fixture"');
    expect(content).toContain('"config/read"');
    expect(content).toContain('"config/batchWrite"');
    expect(content).toContain("updateConfigFromPage");
    expect(content).toContain("mcp: { enabled: false }");
    expect(content).toContain(
      "!disabledRequests[0].toolNames.includes(APPS_TOOL_NAME)",
    );
    expect(content).toContain(
      "disabledRequests[0].toolNames.includes(ORDINARY_TOOL_NAME)",
    );
    expect(content).toContain("name: ORDINARY_TOOL_NAME");
  });

  it("forbids mock and legacy success paths", () => {
    const content = readGateB();

    expect(content).toContain('backendMode: "runtime"');
    expect(content).toContain("LEGACY_MCP_COMMANDS");
    expect(content).toContain("mockFallbackHitCount === 0");
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
    expect(content).not.toContain("mcpTool/callWithCaller");
  });
});
