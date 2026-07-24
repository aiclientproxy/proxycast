import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  applyPassingSettingsArchivedLifecycleEvidence,
  createSettingsArchivedLifecycleEvidence,
  parseSettingsArchivedLifecycleArgs,
  summarizeSettingsArchivedLifecycleTrace,
} from "./settings-archived-lifecycle-fixture-evidence.mjs";

const RUN_ID = "standalone-settings-b-test";

function guiRequest(method) {
  return {
    command: "app_server_handle_json_lines",
    transport: "electron-ipc",
    status: "success",
    method,
    params: {},
  };
}

function rawEvidence() {
  return {
    sidebarGuiArchive: {
      requests: [guiRequest("thread/list")],
    },
    settingsGuiRestoreArchive: {
      requests: [guiRequest("agentSession/update")],
    },
    settingsGuiRestore: {
      requests: [guiRequest("agentSession/update")],
    },
    persistedArchive: {
      requests: [{ method: "thread/read" }],
    },
  };
}

function passingSourceSummary() {
  return {
    ok: true,
    electronPreloadBridge: true,
    sidecarRestartReadback: true,
    consoleErrors: [],
    pageErrors: [],
    persistedArchiveSummary: {
      archiveRequestSeen: true,
      archivedAfterSession: { archivedAt: "present" },
    },
    persistedArchiveReopenSummary: {
      archivedAfterRestartSession: { archivedAt: "present" },
    },
    sidebarGuiArchiveSummary: { updateRequestSeen: true },
    settingsGuiRestoreSummary: { updateRequestSeen: true },
    persistedUnarchiveSummary: {
      unarchiveRequestSeen: true,
      recentAfterSession: { archivedAt: null },
    },
    persistedUnarchiveReopenSummary: {
      recentAfterRestartSession: { archivedAt: null },
    },
  };
}

function passingFacts() {
  return {
    completedAt: "2026-07-17T00:03:00.000Z",
    sourceSummary: passingSourceSummary(),
    trace: summarizeSettingsArchivedLifecycleTrace(rawEvidence()),
    archivedScreenshotWritten: true,
    recoveredScreenshotWritten: true,
  };
}

describe("Settings archived lifecycle Gate B evidence", () => {
  it("uses the same project Gate run root", () => {
    const options = parseSettingsArchivedLifecycleArgs(["--run-id", RUN_ID], {
      defaults: {
        runId: null,
        evidenceDir: null,
        prefix: "settings-archived-lifecycle-fixture",
        timeoutMs: 240_000,
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
        "settings-archived-lifecycle-recovery",
      ),
    );
  });

  it("combines GUI IPC traces with persisted owner phase methods", () => {
    expect(
      summarizeSettingsArchivedLifecycleTrace(rawEvidence()),
    ).toMatchObject({
      appServerIpcHitCount: 3,
      methods: ["agentSession/update", "thread/list", "thread/read"],
      missingMethods: [],
      invokeErrorCount: 0,
      legacyCommands: [],
      mockFallbackHitCount: 0,
    });
  });

  it("requires archive, Settings restore, restart, and independent errors", () => {
    const summary = createSettingsArchivedLifecycleEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-archived-lifecycle-fixture",
    });
    applyPassingSettingsArchivedLifecycleEvidence(summary, passingFacts());
    expect(summary).toMatchObject({
      result: "pass",
      settingsScenarioProof: {
        scenarioId: "archived-lifecycle-recovery",
        complete: true,
      },
      errors: { pageErrorCount: 0 },
      lifecycle: {
        sidebarGuiArchive: true,
        settingsGuiRestore: true,
        archiveRestartReadback: true,
        unarchiveRestartReadback: true,
      },
    });
  });

  it("rejects a source that did not observe page errors", () => {
    const summary = createSettingsArchivedLifecycleEvidence({
      candidateRunId: RUN_ID,
      startedAt: "2026-07-17T00:00:00.000Z",
      prefix: "settings-archived-lifecycle-fixture",
    });
    expect(() =>
      applyPassingSettingsArchivedLifecycleEvidence(summary, {
        ...passingFacts(),
        sourceSummary: {
          ...passingSourceSummary(),
          pageErrors: undefined,
        },
      }),
    ).toThrow(/pageErrorsObservedAndZero/);
  });
});
