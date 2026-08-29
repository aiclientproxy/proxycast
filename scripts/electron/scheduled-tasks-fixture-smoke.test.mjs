import fs from "node:fs";

import { describe, expect, it } from "vitest";

import {
  REQUIRED_SCHEDULED_TASK_METHODS,
  parseScheduledTasksFixtureArgs,
  summarizeScheduledTasksTrace,
} from "./scheduled-tasks-fixture-smoke.mjs";

function traceRaw(
  transport = "electron-ipc",
  methods = REQUIRED_SCHEDULED_TASK_METHODS,
) {
  return JSON.stringify(
    methods.map((method) => ({
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [JSON.stringify({ jsonrpc: "2.0", id: method, method })],
        },
      },
    })),
  );
}

describe("Scheduled Tasks Electron Gate B", () => {
  it("summarizes the current Electron IPC method boundary", () => {
    expect(summarizeScheduledTasksTrace(traceRaw())).toEqual({
      appServerIpcHitCount: REQUIRED_SCHEDULED_TASK_METHODS.length,
      methods: REQUIRED_SCHEDULED_TASK_METHODS,
      missingMethods: [],
      legacyMethods: [],
      legacyCommands: [],
      mockFallbackHitCount: 0,
      invokeErrorCount: 0,
    });
  });

  it("fails evidence closed for mock transport and legacy Automation methods", () => {
    const summary = summarizeScheduledTasksTrace(
      traceRaw("renderer-mock", ["automationJob/list"]),
    );
    expect(summary.mockFallbackHitCount).toBe(1);
    expect(summary.missingMethods).toEqual(REQUIRED_SCHEDULED_TASK_METHODS);
    expect(summary.methods).toEqual([]);
  });

  it("validates fixture CLI bounds", () => {
    expect(
      parseScheduledTasksFixtureArgs(["--timeout-ms", "60000"], {
        evidenceDir: "/tmp/evidence",
        prefix: "scheduled-tasks",
        timeoutMs: 180_000,
        intervalMs: 250,
        keepTemp: false,
      }),
    ).toMatchObject({ timeoutMs: 60_000, prefix: "scheduled-tasks" });
    expect(() =>
      parseScheduledTasksFixtureArgs(["--prefix", "../unsafe"], {
        evidenceDir: "/tmp/evidence",
        prefix: "scheduled-tasks",
        timeoutMs: 180_000,
        intervalMs: 250,
        keepTemp: false,
      }),
    ).toThrow(/invalid evidence prefix/);
  });

  it("keeps the scenario on RuntimeCore and forbids mock success paths", () => {
    const content = fs.readFileSync(
      "scripts/electron/scheduled-tasks-fixture-smoke.mjs",
      "utf8",
    );
    expect(content).toContain('backendMode: "runtime"');
    expect(content).toContain("ensureElectronFixtureBuild");
    expect(content).toContain("startOpenAiCompatibleFixtureServer");
    expect(content).toContain('"workspace/default/ensure"');
    expect(content).toContain("agent_pref_provider_${selectedWorkspaceId}");
    expect(content).toContain("agent_pref_model_${selectedWorkspaceId}");
    expect(content).toContain("encodeModelRouteSelector");
    expect(content).toContain("persistedTaskRouteMatched: true");
    expect(content).toContain("Composer-selected provider/model route");
    expect(content).toContain("composerSelectedProviderModel: true");
    expect(content).not.toContain("inherited-model task");
    expect(content).not.toContain("inheritedComposerModel");
    expect(content).not.toContain("selected.isDefault === true");
    expect(content).not.toContain("sortOrder: -1");
    expect(content).toContain('app-sidebar-nav-scheduled-tasks');
    expect(content).toContain('"scheduledTask/run/start"');
    expect(content).toContain('"thread/read"');
    expect(content).toContain("canonicalConversationOpened: true");
    expect(content).toContain("mockFallbackHitCount === 0");
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
    expect(content).not.toContain('app-sidebar-nav-automation');
  });
});
