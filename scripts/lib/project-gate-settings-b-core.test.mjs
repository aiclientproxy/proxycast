import { describe, expect, it } from "vitest";

import {
  SETTINGS_GATE_B_SCENARIOS,
  buildSettingsGateBFEvidence,
  validateSettingsGateBRunId,
} from "./project-gate-settings-b-core.mjs";

const RUN_ID = "standalone-settings-b-test";

function passingAssertions(details = {}) {
  return { total: 2, passed: 2, failed: [], details };
}

function shellMemorySummary(overrides = {}) {
  return {
    schemaVersion: 1,
    candidateRunId: RUN_ID,
    result: "pass",
    surfaceProof: { surfaceId: "SHELL-01", proof: "gate-b-f", complete: true },
    assertions: passingAssertions({ settingsMemoryReady: true }),
    routes: [{ stage: "settings-memory", ready: true }],
    bridge: {
      electron: true,
      preloadInvoke: true,
      transport: "electron-ipc",
      command: "app_server_handle_json_lines",
      appServerIpcHitCount: 3,
      methods: ["initialize", "memoryStore/status"],
      hostInitialized: true,
    },
    errors: {
      consoleErrorCount: 0,
      pageErrorCount: 0,
      invokeErrorCount: 0,
      traceErrorCount: 0,
      rendererCrashCount: 0,
      rendererUnresponsiveCount: 0,
      preloadErrorCount: 0,
      rendererLoadErrorCount: 0,
      legacyCommandHitCount: 0,
      legacyCommands: [],
      mockFallbackHitCount: 0,
    },
    artifacts: { screenshot: "memory.png", trace: "trace.json" },
    ...overrides,
  };
}

function genericScenarioSummary(scenarioId, overrides = {}) {
  return {
    schemaVersion: 1,
    candidateRunId: RUN_ID,
    result: "pass",
    proofLevel: "Gate B-F",
    settingsScenarioProof: { scenarioId, complete: true },
    assertions: passingAssertions(),
    bridge: {
      electron: true,
      preloadInvoke: true,
      transport: "electron-ipc",
      command: "app_server_handle_json_lines",
      appServerIpcHitCount: 1,
      methods: ["config/read"],
    },
    errors: {
      consoleErrorCount: 0,
      pageErrorCount: 0,
      invokeErrorCount: 0,
      legacyCommandHitCount: 0,
      legacyCommands: [],
      mockFallbackHitCount: 0,
    },
    artifacts: { screenshot: `${scenarioId}.png` },
    ...overrides,
  };
}

function source(kind, value, index = 0) {
  return {
    kind,
    value,
    file: `${kind}-${index}.json`,
    sha256: String(index).padStart(64, "0"),
  };
}

function build(sourceRecords) {
  return buildSettingsGateBFEvidence({
    candidateRunId: RUN_ID,
    startedAt: "2026-07-17T00:00:00.000Z",
    completedAt: "2026-07-17T00:01:00.000Z",
    sourceRecords,
  });
}

describe("project Gate SETTINGS-01 Gate B-F evidence", () => {
  it("aggregates only the exact claims proven by existing owner evidence", () => {
    const evidence = build([source("shell-memory", shellMemorySummary())]);

    expect(evidence.result).toBe("pass");
    expect(evidence.surfaceProof.complete).toBe(false);
    expect(evidence.coverage.completedScenarios).toEqual(["memory-ready"]);
    expect(evidence.missingScenarios).toContain("memory-soul-persistence");
    expect(evidence.missingScenarios).toContain("provider-crud-model-auth");
  });

  it("completes only when every required Settings scenario is present", () => {
    const sourceRecords = SETTINGS_GATE_B_SCENARIOS.map((scenario, index) =>
      source("settings-scenario", genericScenarioSummary(scenario.id), index),
    );
    const evidence = build(sourceRecords);

    expect(evidence.surfaceProof.complete).toBe(true);
    expect(evidence.missingScenarios).toEqual([]);
    expect(evidence.assertions).toMatchObject({
      total: SETTINGS_GATE_B_SCENARIOS.length,
      passed: SETTINGS_GATE_B_SCENARIOS.length,
      failed: [],
    });
  });

  it("rejects a source from another candidate run", () => {
    expect(() =>
      build([
        source(
          "shell-memory",
          shellMemorySummary({ candidateRunId: "other-run" }),
        ),
      ]),
    ).toThrow(/candidateRunId mismatch/);
  });

  it("rejects Shell evidence without explicit zero-error facts", () => {
    expect(() =>
      build([source("shell-memory", shellMemorySummary({ errors: {} }))]),
    ).toThrow(/incomplete real Electron Memory evidence/);
  });

  it("rejects duplicate scenario claims", () => {
    expect(() =>
      build([
        source("shell-memory", shellMemorySummary()),
        source("settings-scenario", genericScenarioSummary("memory-ready"), 1),
      ]),
    ).toThrow(/duplicate SETTINGS-01 Gate B scenarios/);
  });

  it("accepts stronger Gate B-R owner evidence for the browser lifecycle", () => {
    const evidence = build([
      source(
        "settings-scenario",
        genericScenarioSummary("chrome-relay-lifecycle", {
          proofLevel: "Gate B-R",
          bridge: {
            electron: true,
            preloadInvoke: true,
            transport: "electron-ipc",
            command: "app_server_handle_json_lines",
            appServerIpcHitCount: 4,
            methods: [
              "browserSession/target/list",
              "browserSession/open",
              "browserSession/read",
              "browserSession/close",
            ],
          },
        }),
      ),
    ]);

    expect(evidence.coverage.completedScenarios).toEqual([
      "chrome-relay-lifecycle",
    ]);
  });

  it("rejects unsafe run ids and empty source sets", () => {
    expect(() => validateSettingsGateBRunId("../escape")).toThrow(
      /invalid SETTINGS-01 Gate B run-id/,
    );
    expect(() => build([])).toThrow(/at least one source record/);
  });
});
