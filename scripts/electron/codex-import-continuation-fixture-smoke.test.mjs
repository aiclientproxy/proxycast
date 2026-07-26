import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readSmokeSources() {
  return [
    "scripts/electron/codex-import-continuation-fixture-smoke.mjs",
    "scripts/electron/lib/codex-import-continuation-fixture.mjs",
  ]
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
}

describe("codex import continuation Electron fixture smoke guard", () => {
  it("keeps the smoke on real Electron Desktop Host IPC and App Server JSON-RPC", () => {
    const content = readSmokeSources();

    expect(content).toContain("import { _electron as electron }");
    expect(content).toContain("resolveDevAppServerBinary");
    expect(content).toContain("APP_SERVER_BIN: appServerBinary");
    expect(content).toContain("electron.launch({");
    expect(content).toContain('args: ["--use-mock-keychain", "."]');
    expect(content).toContain("ELECTRON_E2E_USER_DATA_DIR");
    expect(content).toContain('LIME_ELECTRON_E2E: "1"');
    expect(content).toContain('LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"');
    expect(content).toContain("window.__LIME_ELECTRON__ === true");
    expect(content).toContain(
      'typeof window.electronAPI?.invoke === "function"',
    );
    expect(content).toContain("window.electronAPI.supportsCommand");
    expect(content).toContain("app_server_handle_json_lines");
    expect(content).toContain("window.__LIME_ELECTRON__ === true");
    expect(content).toContain("client.bridgeFacts.every");
    expect(content).toContain("turnStartCount === 2");
  });

  it("uses the runtime provider loop for imported and normal unified exec turns", () => {
    const content = readSmokeSources();

    expect(content).toContain("startOpenAiCompatibleFixtureServer");
    expect(content).toContain('APP_SERVER_BACKEND_MODE: "runtime"');
    expect(content).toContain('name: "exec_command"');
    expect(content).toContain('request.toolNames.includes("exec_command")');
    expect(content).toContain('request.toolNames.includes("write_stdin")');
    expect(content).toContain("providerRequestsAfterCommit === 0");
    expect(content).toContain("findCompletedCommand");
    expect(content).toContain("commandShapesIsomorphic");
    expect(content).toContain("importedCommandShape");
    expect(content).toContain("normalCommandShape");
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "external"');
    expect(content).not.toContain("APP_SERVER_BACKEND_COMMAND");
  });

  it("imports canonical history details without replaying historical tools", () => {
    const content = readSmokeSources();

    expect(content).toContain('"conversationImport/thread/commit"');
    expect(content).toContain('"modelProvider/create"');
    expect(content).toContain('"modelProvider/update"');
    expect(content).toContain('"modelProviderKey/create"');
    expect(content).toContain('"thread/read"');
    expect(content).toContain('"thread/start"');
    expect(content).toContain('"thread/settings/update"');
    expect(content).not.toContain('"agentSession/update"');
    expect(content).toContain('"turn/start"');
    expect(content).toContain("writeCodexRolloutFixture");
    expect(content).toContain('type: "reasoning"');
    expect(content).toContain('type: "function_call"');
    expect(content).toContain('type: "web_search_call"');
    expect(content).toContain('type: "patch_apply_end"');
    expect(content).toContain('type: "exec_approval_request"');
    expect(content).toContain("historical.hasReasoningItem");
    expect(content).toContain("historical.hasCommandItem");
    expect(content).toContain("historical.hasPatchItem");
    expect(content).toContain("historical.hasWebSearchItem");
    expect(content).toContain("historical.hasApprovalFidelity");
    expect(content).toContain("job.result?.session?.threadId");
    expect(content).toContain("threadId,");
    expect(content).toContain("includeTurns: true");
    expect(content).toContain('input: [{ type: "text", text }]');
    expect(content).toContain(
      "const importedTurnId = String(importedTurn?.turn?.id",
    );
    expect(content).not.toContain("runtimeOptions(");
  });

  it("keeps retired shell names negative-only and excludes production mock fallback", () => {
    const content = readSmokeSources();

    expect(content).toContain(
      'const retiredTools = ["Bash", "PowerShell", "BashTool", "PowerShellTool"]',
    );
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain("agent_runtime_");
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
  });
});
