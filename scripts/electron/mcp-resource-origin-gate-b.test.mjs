import fs from "node:fs";
import { describe, expect, it } from "vitest";

import { findMcpResourceOriginItems } from "./mcp-resource-origin-gate-b.mjs";
import {
  APPS_CONNECTOR_ID,
  APPS_RESOURCE_URI,
  APPS_SERVER_NAME,
} from "./lib/orchestrator-skills-gate-b-core.mjs";

function readGateB() {
  return [
    "scripts/electron/mcp-resource-origin-gate-b.mjs",
    "scripts/electron/orchestrator-skills-gate-b.mjs",
    "scripts/electron/lib/orchestrator-skills-gate-b-core.mjs",
  ]
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
}

describe("MCP resource origin Electron Gate B", () => {
  it("finds only the exact codex_apps canonical resource item", () => {
    const matching = {
      type: "mcpToolCall",
      id: "item_call-mcp-resource-origin",
      server: APPS_SERVER_NAME,
      mcpAppResourceUri: APPS_RESOURCE_URI,
    };
    expect(
      findMcpResourceOriginItems({
        turns: [
          { items: [matching] },
          {
            items: [
              { ...matching, server: "ordinary_fixture" },
              { ...matching, mcpAppResourceUri: "ui://calendar/wrong.html" },
            ],
          },
        ],
      }),
    ).toEqual([matching]);
  });

  it("proves canonical origin authority, GUI content, and cold restore", () => {
    const content = readGateB();

    expect(content).toContain('backendMode: "runtime"');
    expect(content).toContain('"mcpServer/resource/read"');
    expect(content).toContain("originCallId: canonical.itemId");
    expect(content).toContain("connectorId: FORGED_CONNECTOR_ID");
    expect(content).toContain("canonicalAuthorityOnEveryRead");
    expect(content).toContain("selected_connector_ids");
    expect(content).toContain("distinctResourceProcessCount >= 2");
    expect(content).toContain("providerNotReexecuted: true");
    expect(content).toContain("toolNotReexecuted");
    expect(content).toContain("workspace-plugin-surface-frame");
    expect(content).toContain("embedded_browser_view_load_html");
    expect(content).toContain("webContentsMarkerVisible: true");
    expect(content).toContain(APPS_RESOURCE_URI);
    expect(content).toContain(APPS_CONNECTOR_ID);
  });

  it("forbids production mock and legacy success paths", () => {
    const content = readGateB();

    expect(content).toContain("LEGACY_MCP_COMMANDS");
    expect(content).toContain("mockFallbackHitCount === 0");
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
    expect(content).not.toContain("mcpTool/callWithCaller");
  });
});
