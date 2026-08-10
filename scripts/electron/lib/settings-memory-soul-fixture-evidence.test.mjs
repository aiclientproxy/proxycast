import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  MEMORY_SOUL_REQUIRED_METHODS,
  MEMORY_SOUL_RUNTIME_MARKERS,
  applyPassingSettingsMemorySoulEvidence,
  createSettingsMemorySoulEvidence,
  parseSettingsMemorySoulFixtureArgs,
  summarizeSettingsMemorySoulTrace,
} from "./settings-memory-soul-fixture-evidence.mjs";

const RUN_ID = "standalone-settings-b-test";

function traceRaw() {
  return JSON.stringify([
    {
      command: "app_server_handle_json_lines",
      transport: "electron-ipc",
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: "soulStylePack/list",
              method: "soulStylePack/list",
              params: {},
            }),
            JSON.stringify({ id: 2, method: "config/read", params: {} }),
            JSON.stringify({
              id: 3,
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
    electronLaunchCount: 2,
    preloadLaunchCount: 2,
    isolatedUserData: true,
    guiSaved: true,
    restartReadback: true,
    memoryEnabled: true,
    soulEnabled: true,
    profileSelected: true,
    runtime: {
      ok: true,
      scenario: "soul-style",
      proofLevel: "Gate B controlled fixture",
      soulStyleExpectation: { profileId: "cheeky_sassy_executor" },
      soulStyleConfig: {
        enabled: true,
        style_profile_id: "cheeky_sassy_executor",
      },
      soulStylePromptContextCoveredByRuntime: true,
      soulStylePromptContextMarkers: Object.fromEntries(
        MEMORY_SOUL_RUNTIME_MARKERS.map((key) => [key, true]),
      ),
    },
    trace: summarizeSettingsMemorySoulTrace(traceRaw()),
    consoleErrors: [],
    pageErrors: [],
    invokeErrorCount: 0,
    savedScreenshotWritten: true,
    recoveredScreenshotWritten: true,
  };
}

describe("Settings Memory Soul Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsMemorySoulFixtureArgs(["--run-id", RUN_ID], {
      defaults: {
        runId: null,
        evidenceDir: null,
        prefix: "settings-memory-soul-fixture",
        profileId: null,
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
        "settings-memory-soul-persistence",
      ),
    );
  });

  it("requires current style pack methods and host config writes", () => {
    expect(summarizeSettingsMemorySoulTrace(traceRaw())).toMatchObject({
      methods: [
        ...MEMORY_SOUL_REQUIRED_METHODS,
        "config/read",
        "config/batchWrite",
      ],
      missingMethods: [],
      configMethods: ["soulStylePack/list", "config/read", "config/batchWrite"],
      missingConfigMethods: [],
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("requires GUI persistence and runtime prompt markers", () => {
    const summary = createSettingsMemorySoulEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-memory-soul-fixture",
      profileId: "cheeky_sassy_executor",
    });
    applyPassingSettingsMemorySoulEvidence(summary, passingFacts());
    expect(summary).toMatchObject({
      result: "pass",
      settingsScenarioProof: {
        scenarioId: "memory-soul-persistence",
        complete: true,
      },
      runtime: { markersComplete: true, promptStored: false },
    });
  });

  it("rejects missing runtime markers", () => {
    const summary = createSettingsMemorySoulEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-memory-soul-fixture",
      profileId: "cheeky_sassy_executor",
    });
    expect(() =>
      applyPassingSettingsMemorySoulEvidence(summary, {
        ...passingFacts(),
        runtime: {
          ...passingFacts().runtime,
          soulStylePromptContextMarkers: {
            ...passingFacts().runtime.soulStylePromptContextMarkers,
            hasMemorySoulSchema: false,
          },
        },
      }),
    ).toThrow(/runtimePromptMarkers/);
  });
});
