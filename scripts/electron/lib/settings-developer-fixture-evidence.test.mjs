import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  DEVELOPER_REQUIRED_APP_SERVER_METHODS,
  applyPassingSettingsDeveloperEvidence,
  createSettingsDeveloperEvidence,
  parseSettingsDeveloperFixtureArgs,
  summarizeSettingsDeveloperTrace,
} from "./settings-developer-fixture-evidence.mjs";

const RUN_ID = "standalone-settings-b-test";

function traceRaw(transport = "electron-ipc", methods = undefined) {
  return JSON.stringify([
    {
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: { lines: [JSON.stringify({ method: "config/read" })] },
      },
    },
    ...(methods ?? DEVELOPER_REQUIRED_APP_SERVER_METHODS).map((method) => ({
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({ jsonrpc: "2.0", id: method, method, params: {} }),
          ],
        },
      },
    })),
  ]);
}

function payloadShape() {
  return {
    generatedAt: true,
    desktopRuntime: true,
    persistedLogTail: true,
    serverDiagnostics: true,
    logStorageDiagnostics: true,
    currentLogPathAtAgentRoot: true,
    windowsStartupDiagnostics: true,
    runtimeSnapshot: true,
    configSummary: true,
    providerSummary: true,
    mcpSummary: true,
  };
}

function passingFacts() {
  return {
    completedAt: "2026-07-17T00:03:00.000Z",
    electronRenderer: true,
    preloadInvoke: true,
    isolatedUserData: true,
    developerTabActive: true,
    developerLabActive: true,
    copyJsonActionReady: true,
    diagnosticSuccess: true,
    clipboardSinkInstalled: true,
    clipboard: {
      writeCount: 1,
      textLength: 2048,
      jsonObject: true,
      payloadShape: payloadShape(),
    },
    loadingVisible: false,
    errorVisible: false,
    trace: summarizeSettingsDeveloperTrace(traceRaw()),
    consoleErrors: [],
    pageErrors: [],
    invokeErrorCount: 0,
    screenshotWritten: true,
  };
}

describe("Settings Developer Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsDeveloperFixtureArgs(["--run-id", RUN_ID], {
      defaults: {
        runId: null,
        evidenceDir: null,
        prefix: "settings-developer-fixture",
        timeoutMs: 120_000,
        intervalMs: 250,
        keepTemp: false,
      },
      cwd: "/repo",
    });
    expect(options.evidenceDir).toBe(
      path.join(
        "/repo",
        ".lime",
        "qc",
        "project-gates",
        RUN_ID,
        "settings-developer-current-diagnostics",
      ),
    );
  });

  it("requires every current diagnostic method and Host config read", () => {
    expect(summarizeSettingsDeveloperTrace(traceRaw())).toMatchObject({
      appServerIpcHitCount: DEVELOPER_REQUIRED_APP_SERVER_METHODS.length + 1,
      methods: ["config/read", ...DEVELOPER_REQUIRED_APP_SERVER_METHODS],
      missingMethods: [],
      missingConfigMethods: [],
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("completes only with real collection and privacy-minimal sink shape", () => {
    const summary = createSettingsDeveloperEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-developer-fixture",
    });
    applyPassingSettingsDeveloperEvidence(summary, passingFacts());
    expect(summary.result).toBe("pass");
    expect(summary.settingsScenarioProof).toEqual({
      scenarioId: "developer-current-diagnostics",
      complete: true,
    });
    expect(summary.clipboardSink).not.toHaveProperty("text");
  });

  it("rejects mock transport, missing method and incomplete payload shape", () => {
    const summary = createSettingsDeveloperEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-developer-fixture",
    });
    expect(() =>
      applyPassingSettingsDeveloperEvidence(summary, {
        ...passingFacts(),
        clipboard: {
          ...passingFacts().clipboard,
          payloadShape: {
            ...payloadShape(),
            currentLogPathAtAgentRoot: false,
          },
        },
        trace: summarizeSettingsDeveloperTrace(
          traceRaw("renderer-mock", ["log/list"]),
        ),
      }),
    ).toThrow(
      /appServerElectronIpc.*allCurrentDiagnosticMethods.*configCurrentRead.*diagnosticPayloadShape.*mockFallbackZero/,
    );
  });
});
