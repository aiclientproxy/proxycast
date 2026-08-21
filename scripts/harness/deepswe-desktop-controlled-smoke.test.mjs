import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  attachElectronHostLifecycleDiagnostics,
  buildControlledFailureEvidence,
  captureControlledPatch,
  classifyElectronHostLifecycleConsole,
  collectControlledFailureDiagnostics,
  parseArgs,
} from "./deepswe-desktop-controlled-smoke.mjs";
import {
  controlledFixtureForTask,
  controlledFixtureResponses,
  controlledFixtureTaskIds,
} from "./deepswe-desktop-controlled-fixtures.mjs";

function writeFiles(root, files) {
  for (const [relativePath, content] of Object.entries(files)) {
    const absolutePath = path.join(root, relativePath);
    fs.mkdirSync(path.dirname(absolutePath), { recursive: true });
    fs.writeFileSync(absolutePath, content);
  }
}

function runCommand(cwd, command) {
  return spawnSync("sh", ["-c", command], {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 120_000,
  });
}

describe("DeepSWE desktop controlled product smoke", () => {
  it("captures only allowlisted Electron App Server lifecycle console events", () => {
    let listener = null;
    const events = [];
    attachElectronHostLifecycleDiagnostics(
      {
        on(event, callback) {
          expect(event).toBe("console");
          listener = callback;
        },
      },
      events,
    );

    listener?.({
      text: () =>
        "[electron-host] app-server exited { stderrLines: ['secret'] }",
    });
    listener?.({ text: () => "unrelated warning with private prompt" });
    listener?.({
      text: () =>
        "[electron-host] app-server restart failed token=private-token",
    });

    expect(events).toHaveLength(2);
    expect(events.map((event) => event.event)).toEqual([
      "app-server-exited",
      "app-server-restart-failed",
    ]);
    expect(JSON.stringify(events)).not.toContain("secret");
    expect(JSON.stringify(events)).not.toContain("private-token");
    expect(
      classifyElectronHostLifecycleConsole("ordinary renderer warning"),
    ).toBeNull();
  });

  it("probes Electron, renderer, preload and thread/read on failure", async () => {
    let evaluationCount = 0;
    const diagnostics = await collectControlledFailureDiagnostics({
      electronHandle: {
        app: { process: () => ({ pid: process.pid }) },
        page: {
          async evaluate() {
            evaluationCount += 1;
            if (evaluationCount === 1) {
              return {
                url: "file:///fixture/index.html",
                electron: true,
                hasInvokeBridge: true,
                supportsAppServer: true,
              };
            }
            return {
              result: {
                thread: {
                  id: "thread-1",
                  turns: [{ id: "turn-1", status: "inProgress" }],
                },
              },
              messages: [],
            };
          },
        },
      },
      identity: { threadId: "thread-1" },
      probeTimeoutMs: 100,
    });

    expect(diagnostics).toEqual({
      schemaVersion: "deepswe-desktop-controlled-runtime-diagnostics-v1",
      probeTimeoutMs: 100,
      electron: {
        pid: process.pid,
        processAlive: true,
      },
      renderer: {
        status: "ok",
        url: "file:///fixture/index.html",
        electron: true,
        hasInvokeBridge: true,
        supportsAppServer: true,
      },
      bridge: {
        method: "thread/read",
        status: "ok",
        threadFound: true,
        threadIdMatches: true,
        latestTurn: {
          idPresent: true,
          status: "inprogress",
        },
      },
    });
  });

  it("captures provider connection diagnostics without retaining request secrets", () => {
    const runtimeDiagnostics = {
      electron: { pid: 123, processAlive: true },
      renderer: { status: "ok", electron: true, hasInvokeBridge: true },
      bridge: { method: "thread/read", status: "timeout", timeoutMs: 2_000 },
    };
    const failure = buildControlledFailureEvidence({
      error: new Error("terminal timeout"),
      fixtureServer: {
        requests: [
          {
            path: "/v1/chat/completions",
            authorization: "Bearer secret",
            body: {
              messages: [{ role: "user", content: "private prompt" }],
              stream: true,
              tools: [{ type: "function" }],
            },
            responseKind: "text",
          },
        ],
        connectionDiagnostics: [{ event: "response-finish", requestId: 1 }],
      },
      runId: "run-1",
      runtimeDiagnostics,
      sidecarLifecycleEvents: [
        { event: "app-server-exited", source: "electron-main-console" },
      ],
      stage: "terminal-wait",
      taskId: "task-1",
    });

    expect(failure).toMatchObject({
      schemaVersion: "deepswe-desktop-controlled-failure-v1",
      runId: "run-1",
      taskId: "task-1",
      stage: "terminal-wait",
      error: "terminal timeout",
      providerRequestCount: 1,
      providerRequests: [
        {
          index: 1,
          responseKind: "text",
          stream: true,
          messageCount: 1,
          toolCount: 1,
        },
      ],
      connectionDiagnostics: [{ event: "response-finish", requestId: 1 }],
      runtimeDiagnostics,
      sidecarLifecycleEvents: [
        { event: "app-server-exited", source: "electron-main-console" },
      ],
    });
    expect(JSON.stringify(failure)).not.toContain("secret");
    expect(JSON.stringify(failure)).not.toContain("private prompt");
  });

  it("parses one-task and all-task execution options", () => {
    expect(parseArgs([])).toMatchObject({
      task: "all",
      timeoutMs: 240_000,
      intervalMs: 250,
    });
    expect(
      parseArgs([
        "--task",
        "fd-deterministic-multi-key-sorting",
        "--timeout-ms",
        "60000",
        "--interval-ms",
        "100",
      ]),
    ).toMatchObject({
      task: "fd-deterministic-multi-key-sorting",
      timeoutMs: 60_000,
      intervalMs: 100,
    });
    expect(() => parseArgs(["--task", "missing"])).toThrow(
      "Desktop Smoke 5 task not found",
    );
    expect(() => parseArgs(["--timeout-ms", "1000"])).toThrow(
      "--timeout-ms must be >= 60000",
    );
  });

  it("covers exactly the five selected language tasks", () => {
    expect(controlledFixtureTaskIds()).toEqual([
      "happy-dom-abort-pending-body-reads",
      "go-genai-streamed-function-args",
      "httpx-multipart-response-parsing",
      "fd-deterministic-multi-key-sorting",
      "yjs-map-conflict-detection",
    ]);
  });

  it.each(controlledFixtureTaskIds())(
    "%s executes read/search/patch/native-test phases",
    (taskId) => {
      const responses = controlledFixtureResponses(taskId);
      expect(
        responses
          .filter((response) => !response.recovery)
          .map((response) => response.name)
          .filter(Boolean),
      ).toEqual(["Read", "Glob", "Grep", "apply_patch", "exec_command"]);
      expect(responses.filter((response) => response.recovery)).toHaveLength(2);
      expect(
        responses
          .filter((response) => response.recovery)
          .map((response) => response.name),
      ).toEqual(["exec_command", "exec_command"]);
      const finalResponse = responses.find(
        (response) => response.type === "text" && response.content,
      );
      expect(finalResponse).toBeTruthy();
      expect(finalResponse.content).toBe(
        controlledFixtureForTask(taskId).finalMarker,
      );
    },
  );

  it.each(controlledFixtureTaskIds())(
    "%s has a failing baseline and a passing native-language candidate",
    (taskId) => {
      const fixture = controlledFixtureForTask(taskId);
      const root = fs.mkdtempSync(path.join(os.tmpdir(), "desktop-language-"));
      try {
        writeFiles(root, fixture.files);
        const before = runCommand(root, fixture.testCommand);
        expect(before.status).not.toBe(0);

        writeFiles(root, fixture.finalFiles);
        const after = runCommand(root, fixture.testCommand);
        expect(
          `${after.stdout || ""}\n${after.stderr || ""}`,
          `${taskId} native test failed`,
        ).toContain(fixture.testMarker);
        expect(after.status).toBe(0);

        const patch = captureControlledPatch(root, fixture);
        expect(patch.patchBytes).toBeGreaterThan(0);
        expect(patch.patchSha256).toMatch(/^[a-f0-9]{64}$/u);
        expect(patch.changedFiles.length).toBeGreaterThan(0);
        expect(patch.patch).toContain("diff --git");
      } finally {
        fs.rmSync(root, { recursive: true, force: true });
      }
    },
    120_000,
  );
});
