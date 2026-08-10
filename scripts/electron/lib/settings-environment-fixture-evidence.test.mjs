import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  applyPassingSettingsEnvironmentEvidence,
  createSettingsEnvironmentEvidence,
  parseSettingsEnvironmentFixtureArgs,
  summarizeSettingsEnvironmentTrace,
} from "./settings-environment-fixture-evidence.mjs";

const RUN_ID = "standalone-settings-b-test";

function traceRaw() {
  return JSON.stringify([
    {
      command: "get_environment_preview",
      transport: "electron-ipc",
      status: "success",
    },
    {
      command: "app_server_handle_json_lines",
      transport: "electron-ipc",
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({ id: 1, method: "thread/list", params: {} }),
          ],
        },
      },
    },
  ]);
}

describe("Settings Environment Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsEnvironmentFixtureArgs(["--run-id", RUN_ID], {
      defaults: {
        runId: null,
        evidenceDir: null,
        prefix: "settings-environment-fixture",
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
        "settings-environment-current-read",
      ),
    );
  });

  it("separates current Host reads from App Server bridge evidence", () => {
    expect(summarizeSettingsEnvironmentTrace(traceRaw())).toMatchObject({
      appServerIpcHitCount: 1,
      appServerMethods: ["thread/list"],
      hostIpcHitCount: 1,
      hostCommands: ["get_environment_preview"],
      missingHostCommands: [],
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("completes only from a terminal read state", () => {
    const summary = createSettingsEnvironmentEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-environment-fixture",
    });
    applyPassingSettingsEnvironmentEvidence(summary, {
      completedAt: "2026-07-17T00:01:00.000Z",
      electronRenderer: true,
      preloadInvoke: true,
      configShapeValid: true,
      environmentActive: true,
      loadingVisible: false,
      errorVisible: false,
      trace: summarizeSettingsEnvironmentTrace(traceRaw()),
      consoleErrors: [],
      pageErrors: [],
      invokeErrorCount: 0,
      screenshotWritten: true,
    });
    expect(summary.result).toBe("pass");
    expect(summary.settingsScenarioProof.complete).toBe(true);
  });

  it("rejects missing Host reads", () => {
    const summary = createSettingsEnvironmentEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-environment-fixture",
    });
    const trace = summarizeSettingsEnvironmentTrace("[]");
    expect(() =>
      applyPassingSettingsEnvironmentEvidence(summary, {
        completedAt: "2026-07-17T00:01:00.000Z",
        electronRenderer: true,
        preloadInvoke: true,
        configShapeValid: true,
        environmentActive: true,
        loadingVisible: false,
        errorVisible: false,
        trace,
        consoleErrors: [],
        pageErrors: [],
        invokeErrorCount: 0,
        screenshotWritten: true,
      }),
    ).toThrow(/appServerElectronIpc.*hostElectronIpc.*hostCurrentCommands/);
  });
});
