import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  applyPassingSettingsWebSearchEvidence,
  createSettingsWebSearchEvidence,
  parseSettingsWebSearchFixtureArgs,
  summarizeSettingsWebSearchTrace,
} from "./settings-web-search-fixture-evidence.mjs";

const RUN_ID = "standalone-settings-b-test";

function traceRaw(transport = "electron-ipc") {
  return JSON.stringify([
    {
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: 1,
              method: "config/read",
              params: {},
            }),
            JSON.stringify({
              jsonrpc: "2.0",
              id: 2,
              method: "config/batchWrite",
              params: {},
            }),
          ],
        },
      },
    },
  ]);
}

function passingFacts() {
  return {
    completedAt: "2026-07-17T00:03:00.000Z",
    electronLaunchCount: 3,
    preloadLaunchCount: 3,
    isolatedUserData: true,
    webSearchTabActive: true,
    routeControlReady: true,
    routeChanged: true,
    saveConfirmed: true,
    restartReadback: true,
    restorationSaveConfirmed: true,
    restorationReadback: true,
    loadingVisible: false,
    errorVisible: false,
    trace: summarizeSettingsWebSearchTrace([
      traceRaw(),
      traceRaw(),
      traceRaw(),
    ]),
    consoleErrors: [],
    pageErrors: [],
    invokeErrorCount: 0,
    savedScreenshotWritten: true,
    restartScreenshotWritten: true,
    restoredScreenshotWritten: true,
  };
}

describe("Settings Web Search Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsWebSearchFixtureArgs(["--run-id", RUN_ID], {
      defaults: {
        runId: null,
        evidenceDir: null,
        prefix: "settings-web-search-fixture",
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
        "settings-web-search-route",
      ),
    );
  });

  it("requires current config read/write and App Server IPC", () => {
    expect(summarizeSettingsWebSearchTrace(traceRaw())).toMatchObject({
      appServerIpcHitCount: 1,
      methods: ["config/read", "config/batchWrite"],
      configIpcHitCount: 1,
      missingConfigMethods: [],
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("completes only after changed and restored restart readback", () => {
    const summary = createSettingsWebSearchEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-web-search-fixture",
    });
    applyPassingSettingsWebSearchEvidence(summary, passingFacts());
    expect(summary.result).toBe("pass");
    expect(summary.settingsScenarioProof).toEqual({
      scenarioId: "web-search-route",
      complete: true,
    });
  });

  it("rejects mock transport and missing restoration readback", () => {
    const summary = createSettingsWebSearchEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-web-search-fixture",
    });
    expect(() =>
      applyPassingSettingsWebSearchEvidence(summary, {
        ...passingFacts(),
        restorationReadback: false,
        trace: summarizeSettingsWebSearchTrace(traceRaw("renderer-mock")),
      }),
    ).toThrow(
        /appServerElectronIpc.*appServerCurrentMethod.*configElectronIpc.*configCurrentReadWrite.*restorationReadback.*mockFallbackZero/,
    );
  });
});
