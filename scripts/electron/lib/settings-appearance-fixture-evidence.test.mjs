import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  applyPassingSettingsAppearanceEvidence,
  createSettingsAppearanceEvidence,
  parseSettingsAppearanceFixtureArgs,
  summarizeSettingsAppearanceTrace,
} from "./settings-appearance-fixture-evidence.mjs";

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
    appearanceTabActive: true,
    themeControlReady: true,
    behaviorControlReady: true,
    themeChanged: true,
    behaviorChanged: true,
    restartThemeReadback: true,
    restartBehaviorReadback: true,
    restorationSaved: true,
    restorationThemeReadback: true,
    restorationBehaviorReadback: true,
    loadingVisible: false,
    errorVisible: false,
    trace: summarizeSettingsAppearanceTrace([
      traceRaw(),
      traceRaw(),
      traceRaw(),
    ]),
    consoleErrors: [],
    pageErrors: [],
    invokeErrorCount: 0,
    savedScreenshotWritten: true,
    behaviorScreenshotWritten: true,
    restartScreenshotWritten: true,
    restoredScreenshotWritten: true,
  };
}

describe("Settings Appearance Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsAppearanceFixtureArgs(["--run-id", RUN_ID], {
      defaults: {
        runId: null,
        evidenceDir: null,
        prefix: "settings-appearance-fixture",
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
        "settings-appearance-persistence",
      ),
    );
  });

  it("requires current config read/write and App Server IPC", () => {
    expect(summarizeSettingsAppearanceTrace(traceRaw())).toMatchObject({
      appServerIpcHitCount: 1,
      methods: ["config/read", "config/batchWrite"],
      configIpcHitCount: 1,
      missingConfigMethods: [],
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("completes only after both owners restore across restart", () => {
    const summary = createSettingsAppearanceEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-appearance-fixture",
    });
    applyPassingSettingsAppearanceEvidence(summary, passingFacts());
    expect(summary.result).toBe("pass");
    expect(summary.settingsScenarioProof).toEqual({
      scenarioId: "appearance-persistence",
      complete: true,
    });
  });

  it("rejects mock transport and missing behavior restoration", () => {
    const summary = createSettingsAppearanceEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-appearance-fixture",
    });
    expect(() =>
      applyPassingSettingsAppearanceEvidence(summary, {
        ...passingFacts(),
        restorationBehaviorReadback: false,
        trace: summarizeSettingsAppearanceTrace(traceRaw("renderer-mock")),
      }),
    ).toThrow(
        /appServerElectronIpc.*appServerCurrentMethod.*configElectronIpc.*configCurrentReadWrite.*restorationBehaviorReadback.*mockFallbackZero/,
    );
  });
});
