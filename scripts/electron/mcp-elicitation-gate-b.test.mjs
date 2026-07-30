import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readGateB() {
  return fs.readFileSync("scripts/electron/mcp-elicitation-gate-b.mjs", "utf8");
}

describe("MCP elicitation Gate B guard", () => {
  it("keeps the success path on real Electron, App Server runtime, and scoped MCP execution", () => {
    const content = readGateB();

    expect(content).toContain('backendMode: "runtime"');
    expect(content).toContain("launchElectronFixture");
    expect(content).toContain("app_server_handle_json_lines");
    expect(content).toContain('"turn/start"');
    expect(content).toContain('"mcpServer/create"');
    expect(content).toContain('"mcpServer/start"');
    expect(content).toContain("mcp__${serverName}__${TOOL_SUFFIX}");
    expect(content).toContain('method: "elicitation/create"');
    expect(content).toContain('type: "initialize"');
    expect(content).toContain("startOpenAiCompatibleFixtureServer");
    expect(content).toContain('"modelProvider/create"');
    expect(content).toContain('"modelProvider/update"');
    expect(content).toContain('"modelProviderKey/create"');
    expect(content).toContain('"model/list"');
    expect(content).toContain(
      "capability: fixture.provider.providerConfig.modelCapabilities",
    );
    expect(content).toContain("const capability = model.capabilitySnapshot");
    expect(content).toContain("NAVIGATION_RESTORE_STORAGE_KEY");
    expect(content).toContain("initialSessionId: activeSessionId");
  });

  it("uses only current v2 Thread/Turn request shapes", () => {
    const content = readGateB();

    expect(content).toContain(
      'const threadId = String(start.result?.thread?.id || "").trim()',
    );
    expect(content).toMatch(/input:\s*\[\s*\{\s*type:\s*"text",\s*text:/u);
    expect(content).toContain('kind: "application"');
    expect(content).toContain("threadId: runtime.threadId");
    expect(content).toContain("includeTurns: true");
    expect(content).toContain("workspaceRoot: workspace.rootPath");
    expect(content).not.toContain("runtimeEnv.workspaceRoot");
    expect(content).not.toContain("businessObjectRef:");
    expect(content).not.toContain("runtimeOptions:");
    expect(content).not.toContain("queueIfBusy:");
    expect(content).not.toContain("historyLimit:");
  });

  it("requires exact runtime capability advertisement and management capability absence", () => {
    const content = readGateB();

    expect(content).toContain("capabilityAdvertisementRequired: true");
    expect(content).toContain('initializedProtocolVersion === "2025-06-18"');
    expect(content).toContain(
      "Object.keys(initializedCapabilities).length === 1",
    );
    expect(content).toContain(
      "isExactEmptyObject(initializedCapabilities.elicitation)",
    );
    expect(content).toContain('type: "capability_missing"');
    expect(content).toContain("runtimeCapabilityExact");
    expect(content).toContain("managementElicitationCapabilityAbsent");
    expect(content).toContain("summary.capabilityMissingCount === 0");
  });

  it("requires dynamic-tool and MCP results before the final provider response", () => {
    const content = readGateB();

    expect(content).toContain('input[type="checkbox"]');
    expect(content).toContain("confirmed=true");
    expect(content).toContain("MCP_ELICITATION_GATE_B_DONE");
    expect(content).toContain("providerRequests.length >= 3");
    expect(content).toContain("dynamicToolProviderResultObserved");
    expect(content).toContain("dynamicToolCanonicalCompleted");
    expect(content).toContain("dynamicToolRequestHiddenFromRenderer");
    expect(content).toContain('entry?.action === "accept"');
    expect(content).toContain("content?.confirmed === true");
    expect(content).toContain('data-testid="pending-interaction-layer"');
    expect(content).toContain('data-testid="mcp-server-elicitation-form"');
    expect(content).toContain("formClosedAfterResolved");
    expect(content).toContain("rootDialogAbsent");
    expect(content).not.toContain("dialogClosedAfterResolved");
  });

  it("does not use generic action response, explicit management call proof, or mock fallback", () => {
    const content = readGateB();

    expect(content).not.toContain("mcpTool/callWithCaller");
    expect(content).not.toContain("agentSession/action/respond");
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
  });
});
