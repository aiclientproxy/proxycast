import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  applyPassingSettingsProfileEvidence,
  createSettingsProfileEvidence,
  parseSettingsProfileFixtureArgs,
  summarizeSettingsProfileTrace,
} from "./settings-profile-fixture-evidence.mjs";

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
    localProfileMode: true,
    profileTabActive: true,
    profileEditorReady: true,
    profileChanged: true,
    saveConfirmed: true,
    restartReadback: true,
    restorationSaveConfirmed: true,
    restorationReadback: true,
    loadingVisible: false,
    errorVisible: false,
    trace: summarizeSettingsProfileTrace([traceRaw(), traceRaw(), traceRaw()]),
    consoleErrors: [],
    pageErrors: [],
    invokeErrorCount: 0,
    savedScreenshotWritten: true,
    restartScreenshotWritten: true,
    restoredScreenshotWritten: true,
  };
}

describe("Settings Profile Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsProfileFixtureArgs(["--run-id", RUN_ID], {
      defaults: {
        runId: null,
        evidenceDir: null,
        prefix: "settings-profile-fixture",
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
        "settings-profile-persistence",
      ),
    );
  });

  it("requires current config read/write and App Server IPC", () => {
    expect(summarizeSettingsProfileTrace(traceRaw())).toMatchObject({
      appServerIpcHitCount: 1,
      methods: ["config/read", "config/batchWrite"],
      configIpcHitCount: 1,
      missingConfigMethods: [],
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("completes only after changed and restored restart readback", () => {
    const summary = createSettingsProfileEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-profile-fixture",
    });
    applyPassingSettingsProfileEvidence(summary, passingFacts());
    expect(summary.result).toBe("pass");
    expect(summary.settingsScenarioProof).toEqual({
      scenarioId: "profile-persistence",
      complete: true,
    });
  });

  it("rejects mock transport and missing restoration readback", () => {
    const summary = createSettingsProfileEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-profile-fixture",
    });
    expect(() =>
      applyPassingSettingsProfileEvidence(summary, {
        ...passingFacts(),
        restorationReadback: false,
        trace: summarizeSettingsProfileTrace(traceRaw("renderer-mock")),
      }),
    ).toThrow(
        /appServerElectronIpc.*appServerCurrentMethod.*configElectronIpc.*configCurrentReadWrite.*restorationReadback.*mockFallbackZero/,
    );
  });
});
