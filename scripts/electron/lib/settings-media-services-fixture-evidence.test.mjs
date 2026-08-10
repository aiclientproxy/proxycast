import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  applyPassingSettingsMediaServicesEvidence,
  createSettingsMediaServicesEvidence,
  parseSettingsMediaServicesFixtureArgs,
  summarizeSettingsMediaServicesTrace,
} from "./settings-media-services-fixture-evidence.mjs";

const RUN_ID = "standalone-settings-b-test";

function traceRaw(transport = "electron-ipc") {
  const appServerEntries = [
    "model/list",
    "modelPreferences/list",
    "modelSyncState/read",
  ].map((method, index) => ({
    command: "app_server_handle_json_lines",
    transport,
    status: "success",
    args_preview: {
      request: {
        lines: [
          JSON.stringify({
            jsonrpc: "2.0",
            id: index + 1,
            method,
            params: {},
          }),
        ],
      },
    },
  }));
  return JSON.stringify([
    {
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [JSON.stringify({ method: "config/read", params: {} })],
        },
      },
    },
    {
      command: "voice_models_list_catalog",
      transport,
      status: "success",
    },
    ...appServerEntries,
  ]);
}

function passingFacts() {
  return {
    completedAt: "2026-07-17T00:01:00.000Z",
    electronRenderer: true,
    preloadInvoke: true,
    mediaServicesActive: true,
    serviceModelsVisible: true,
    imageServiceVisible: true,
    videoServiceVisible: true,
    voiceServiceVisible: true,
    configControlsReady: true,
    loadingVisible: false,
    errorVisible: false,
    trace: summarizeSettingsMediaServicesTrace(traceRaw()),
    consoleErrors: [],
    pageErrors: [],
    invokeErrorCount: 0,
    screenshotWritten: true,
    imageScreenshotWritten: true,
    videoScreenshotWritten: true,
    readinessScreenshotWritten: true,
  };
}

describe("Settings Media Services Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsMediaServicesFixtureArgs(
      ["--run-id", RUN_ID],
      {
        defaults: {
          runId: null,
          evidenceDir: null,
          prefix: "settings-media-services-fixture",
          timeoutMs: 120_000,
          intervalMs: 250,
          keepTemp: false,
        },
        cwd: "/repo",
      },
    );
    expect(options.evidenceDir).toBe(
      path.join(
        "/repo",
        ".lime",
        "qc",
        "project-gates",
        RUN_ID,
        "settings-media-services-readiness",
      ),
    );
  });

  it("requires model selector methods and both current Host reads", () => {
    expect(summarizeSettingsMediaServicesTrace(traceRaw())).toMatchObject({
      appServerIpcHitCount: 4,
      methods: ["config/read", "model/list", "modelPreferences/list", "modelSyncState/read"],
      configMethods: ["config/read", "model/list", "modelPreferences/list", "modelSyncState/read"],
      missingConfigMethods: [],
      missingMethods: [],
      hostIpcHitCount: 1,
      hostCommands: ["voice_models_list_catalog"],
      missingHostCommands: [],
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("completes only from all four visible service areas", () => {
    const summary = createSettingsMediaServicesEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-media-services-fixture",
    });
    applyPassingSettingsMediaServicesEvidence(summary, passingFacts());
    expect(summary.result).toBe("pass");
    expect(summary.settingsScenarioProof).toEqual({
      scenarioId: "media-services-readiness",
      complete: true,
    });
  });

  it("rejects mock transport and incomplete GUI readiness", () => {
    const summary = createSettingsMediaServicesEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-media-services-fixture",
    });
    expect(() =>
      applyPassingSettingsMediaServicesEvidence(summary, {
        ...passingFacts(),
        voiceServiceVisible: false,
        trace: summarizeSettingsMediaServicesTrace(traceRaw("renderer-mock")),
      }),
    ).toThrow(
      /appServerElectronIpc.*modelSelectorCurrentMethods.*hostElectronIpc.*hostCurrentReads.*voiceServiceVisible.*mockFallbackZero/,
    );
  });
});
