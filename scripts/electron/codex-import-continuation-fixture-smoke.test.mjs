import fs from "node:fs";
import { describe, expect, it } from "vitest";
import {
  buildProviderScriptedResponses,
  TERMINAL_INTERACTION_CHARS,
} from "./lib/codex-import-continuation-fixture.mjs";

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
    expect(content).toContain("turnStartCount === 1");
    expect(content).toContain("rendererTurnStart");
  });

  it("uses the runtime provider loop for imported and normal unified exec turns", () => {
    const content = readSmokeSources();

    expect(content).toContain("startOpenAiCompatibleFixtureServer");
    expect(content).toContain('APP_SERVER_BACKEND_MODE: "runtime"');
    expect(content).toContain('name: "exec_command"');
    expect(content).toContain('name: "write_stdin"');
    expect(content).toContain("session_id");
    expect(content).toContain("TERMINAL_INTERACTION_SUMMARY");
    expect(content).toContain(
      'message?.method === "item/commandExecution/terminalInteraction"',
    );
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

  it("derives write_stdin session identity from the preceding real tool result", async () => {
    const scripted = buildProviderScriptedResponses({
      workspaceRoot: "/fixture-workspace",
    });

    expect(scripted.responses).toHaveLength(6);
    const importedWrite = await scripted.responses[1]({
      body: {
        messages: [
          {
            role: "tool",
            content: '{"output":"","session_id":1234}',
          },
        ],
      },
    });
    const normalWrite = await scripted.responses[4]({
      body: {
        messages: [
          {
            role: "tool",
            content: '{"session_id":5678,"output":""}',
          },
        ],
      },
    });

    expect(importedWrite).toMatchObject({
      type: "tool_call",
      name: "write_stdin",
      arguments: {
        session_id: 1234,
        chars: TERMINAL_INTERACTION_CHARS,
      },
    });
    expect(normalWrite).toMatchObject({
      type: "tool_call",
      name: "write_stdin",
      arguments: {
        session_id: 5678,
        chars: TERMINAL_INTERACTION_CHARS,
      },
    });
  });

  it("imports canonical history details without replaying historical tools", () => {
    const content = readSmokeSources();

    expect(content).toContain('"conversationImport/thread/commit"');
    expect(content).toContain('"modelProvider/create"');
    expect(content).toContain('"modelProvider/update"');
    expect(content).toContain('"modelProviderKey/create"');
    expect(content).toContain('"model/list"');
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
    expect(content).toContain("beforeCommitResult");
    expect(content).toContain("model: FIXTURE_MODEL");
    expect(content).toContain("createRepositoryProvider");
    expect(content).toContain("provider.providerConfig.modelCapabilities");
    expect(content).toContain("catalogModel.capabilitySnapshot");
    expect(content).toContain("threadId,");
    expect(content).toContain("includeTurns: true");
    expect(content).toContain('input: [{ type: "text", text }]');
    expect(content).toContain(
      "const importedTurnId = String(importedTurn?.turn?.id",
    );
    expect(content).not.toContain("runtimeOptions(");
  });

  it("requires the visible redacted summary before and after renderer reload", () => {
    const content = readSmokeSources();

    expect(content).toContain("open-normal-thread-before-live-turn");
    expect(content).toContain("select-normal-thread-fixture-model");
    expect(content).toContain("send-normal-turn-through-renderer");
    expect(content).toContain(
      "reload-and-assert-terminal-interaction-recovery",
    );
    expect(content).toContain("lime.appNavigation.restore.v1");
    expect(content).toContain("openSessionInRenderer");
    expect(content).toContain('data-testid="send-btn"');
    expect(content).toContain('data-testid="model-selector"');
    expect(content).toContain('data-model-selector-popover="true"');
    expect(content).toContain("selectRuntimeRouteInRenderer");
    expect(content).toContain("providerName: route.providerName");
    expect(content).toContain("trigger?.textContent?.includes(expectedModel)");
    expect(content).toContain("sendButton.click");
    expect(content).toContain("waitForGuiTerminalInteraction");
    expect(content).toContain("rawStdinProjected: false");
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
