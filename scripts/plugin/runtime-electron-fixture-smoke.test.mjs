import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readSmokeScript() {
  return fs.readFileSync(
    "scripts/plugin/runtime-electron-fixture-smoke.mjs",
    "utf8",
  );
}

describe("plugin runtime Electron fixture smoke guard", () => {
  it("keeps the proof on the real Electron Desktop Host IPC facade", () => {
    const content = readSmokeScript();

    expect(content).toContain("import { _electron as electron }");
    expect(content).toContain("electron.launch({");
    expect(content).toContain('"--use-mock-keychain"');
    expect(content).toContain("ELECTRON_E2E_USER_DATA_DIR");
    expect(content).toContain('LIME_ELECTRON_E2E: "1"');
    expect(content).toContain('LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"');
    expect(content).toContain("window.__LIME_ELECTRON__ === true");
    expect(content).toContain(
      'typeof window.electronAPI?.invoke === "function"',
    );
    expect(content).toContain("window.electronAPI.supportsCommand");
    expect(content).toContain("plugin_runtime_start_task");
    expect(content).toContain("plugin_runtime_get_task");
    expect(content).toContain("plugin_runtime_submit_host_response");
    expect(content).toContain("plugin_runtime_cancel_task");
  });

  it("uses an explicit external backend fixture and records App Server runtime requests", () => {
    const content = readSmokeScript();

    expect(content).toContain('APP_SERVER_BACKEND_MODE: "external"');
    expect(content).toContain("APP_SERVER_BACKEND_COMMAND: process.execPath");
    expect(content).toContain("APP_SERVER_BACKEND_ARGS: JSON.stringify");
    expect(content).toContain("writeFixtureBackend(");
    expect(content).toContain("readBackendLedger(");
    expect(content).toContain('input.kind === "turnStart"');
    expect(content).toContain('input.kind === "actionRespond"');
    expect(content).toContain('input.kind === "turnCancel"');
    expect(content).toContain("assertBackendLedger(");
    expect(content).toContain('"thread/start"');
    expect(content).toContain('"turn/start"');
    expect(content).toContain('"thread/read"');
    expect(content).toContain('"item/tool/requestUserInput"');
    expect(content).toContain('"serverRequest/resolved"');
    expect(content).not.toContain('"agentSession/action/respond"');
    expect(content).toContain('"turn/interrupt"');
  });

  it("locks typed runtime request evidence", () => {
    const content = readSmokeScript();

    expect(content).toContain("runtimeRequest: {");
    expect(content).toContain("providerConfig");
    expect(content).toContain("systemPrompt");
    expect(content).toContain("reasoningEffort");
    expect(content).toContain("approvalPolicy");
    expect(content).toContain("sandboxPolicy");
    expect(content).toContain("webSearch");
    expect(content).toContain("executionStrategy");
    expect(content).toContain("runtimeOptions?.runtimeRequest");
    expect(content).toContain("runtimeRequest.providerConfig");
    expect(content).toContain("runtimeRequest.providerPreference");
    expect(content).toContain("runtimeRequest.modelPreference");
  });

  it("does not use legacy runtime commands or mock fallback as success evidence", () => {
    const content = readSmokeScript();

    expect(content).not.toContain("agent_runtime_submit_turn");
    expect(content).not.toContain("agent_runtime_interrupt_turn");
    expect(content).not.toContain("agent_runtime_get_thread_read");
    expect(content).not.toContain("agent_runtime_respond_action");
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
    expect(content).not.toContain("pluginMocks");
    expect(content).not.toContain('backendMode: "mock"');
  });
});
