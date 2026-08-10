import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  applyPassingSettingsExecutionPolicyEvidence,
  createSettingsExecutionPolicyEvidence,
  parseSettingsExecutionPolicyFixtureArgs,
  summarizeSettingsExecutionPolicyTrace,
} from "./settings-execution-policy-fixture-evidence.mjs";

const RUN_ID = "standalone-settings-b-test";

function appServerEntry(method) {
  return {
    command: "app_server_handle_json_lines",
    transport: "electron-ipc",
    status: "success",
    args_preview: {
      request: {
        lines: [JSON.stringify({ jsonrpc: "2.0", id: 1, method, params: {} })],
      },
    },
  };
}

function traceRaw() {
  return JSON.stringify([
    {
      ...appServerEntry("config/read"),
    },
    {
      ...appServerEntry("config/batchWrite"),
    },
    {
      command: "app_server_handle_json_lines",
      transport: "electron-ipc",
      status: "error",
      error: "EISDIR: illegal operation on a directory",
    },
    appServerEntry("config/batchWrite"),
  ]);
}

function errorRaw() {
  return JSON.stringify([
    {
      command: "app_server_handle_json_lines",
      transport: "electron-ipc",
      error: "EISDIR: illegal operation on a directory",
    },
  ]);
}

function passingFacts() {
  return {
    completedAt: "2026-07-17T00:03:00.000Z",
    electronLaunchCount: 3,
    preloadLaunchCount: 3,
    isolatedUserData: true,
    executionPolicyTabActive: true,
    policyControlsReady: true,
    policyInputsChanged: true,
    strictRestrictionInput: true,
    warningBypassInput: true,
    restartReadback: true,
    expectedSaveFailureVisible: true,
    expectedSaveFailureRecovered: true,
    restorationSaved: true,
    finalRestorationReadback: true,
    loadingVisible: false,
    unexpectedErrorVisible: false,
    trace: summarizeSettingsExecutionPolicyTrace({
      traceRaws: [traceRaw()],
      errorRaws: [errorRaw()],
    }),
    consoleErrors: [],
    pageErrors: [],
    changedScreenshotWritten: true,
    restartScreenshotWritten: true,
    restoredScreenshotWritten: true,
  };
}

describe("Settings Execution Policy Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsExecutionPolicyFixtureArgs(
      ["--run-id", RUN_ID],
      {
        defaults: {
          runId: null,
          evidenceDir: null,
          prefix: "settings-execution-policy-fixture",
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
        "settings-execution-policy-allow-deny-error",
      ),
    );
  });

  it("classifies the expected Host save failure without hiding other errors", () => {
    expect(
      summarizeSettingsExecutionPolicyTrace({
        traceRaws: [traceRaw()],
        errorRaws: [errorRaw()],
      }),
    ).toMatchObject({
      appServerIpcHitCount: 4,
      methods: ["config/read", "config/batchWrite"],
      configMethods: ["config/read", "config/batchWrite"],
      missingConfigMethods: [],
      successfulSaveCount: 3,
      expectedSaveTraceErrorCount: 1,
      expectedSaveInvokeErrorCount: 1,
      unexpectedHostTraceErrorCount: 0,
      unexpectedInvokeErrorCount: 0,
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("completes only after persistence, expected error, recovery and restore", () => {
    const summary = createSettingsExecutionPolicyEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-execution-policy-fixture",
    });
    applyPassingSettingsExecutionPolicyEvidence(summary, passingFacts());
    expect(summary.result).toBe("pass");
    expect(summary.settingsScenarioProof).toEqual({
      scenarioId: "execution-policy-allow-deny-error",
      complete: true,
    });
    expect(summary.expectedFailure).not.toHaveProperty("error");
  });

  it("rejects unexpected invoke errors and missing final restoration", () => {
    const summary = createSettingsExecutionPolicyEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-execution-policy-fixture",
    });
    const unexpectedError = JSON.stringify([
      ...JSON.parse(errorRaw()),
      {
        command: "app_server_handle_json_lines",
        transport: "electron-ipc",
        error: "unexpected",
      },
    ]);
    expect(() =>
      applyPassingSettingsExecutionPolicyEvidence(summary, {
        ...passingFacts(),
        finalRestorationReadback: false,
        trace: summarizeSettingsExecutionPolicyTrace({
          traceRaws: [traceRaw()],
          errorRaws: [unexpectedError],
        }),
      }),
    ).toThrow(/finalRestorationReadback.*unexpectedInvokeErrorsZero/);
  });
});
