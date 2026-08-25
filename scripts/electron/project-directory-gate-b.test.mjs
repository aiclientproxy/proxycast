import fs from "node:fs";

import { describe, expect, it } from "vitest";

import {
  assertProjectDirectoryEvidence,
  parseProjectDirectoryGateArgs,
  PROJECT_DIRECTORY_REQUIRED_METHODS,
  sanitizeProjectDirectoryFailure,
  summarizeProjectDirectoryEvidence,
} from "./project-directory-gate-b.mjs";

const THREAD_ID = "thread-project-gate-b";
const INITIAL_PROJECT_ID = "project-alpha";
const SELECTED_PROJECT_ID = "project-beta";

function request(method, params = {}, transport = "electron-ipc") {
  return {
    command: "app_server_handle_json_lines",
    method,
    params,
    status: "success",
    transport,
  };
}

function setupRequests(transport = "electron-ipc") {
  return [
    request("project/create", {}, transport),
    request("project/create", {}, transport),
    request("thread/start", {}, transport),
  ];
}

function traceRaw(transport = "electron-ipc", options = {}) {
  const threadId = options.threadId ?? THREAD_ID;
  const selectedProjectId = options.selectedProjectId ?? SELECTED_PROJECT_ID;
  const requests = [
    request("project/list", {}, transport),
    request("thread/read", { threadId }, transport),
    request(
      "thread/metadata/update",
      { threadId, projectId: selectedProjectId },
      transport,
    ),
    request("project/list", {}, transport),
    request("thread/read", { threadId }, transport),
  ];
  return JSON.stringify(
    requests.map((entry) => ({
      command: entry.command,
      transport: entry.transport,
      status: entry.status,
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: entry.method,
              method: entry.method,
              params: entry.params,
            }),
          ],
        },
      },
    })),
  );
}

function visibleDom() {
  return {
    conversationButtonVisible: true,
    selectorVisible: true,
    directoryVisible: true,
    initialProjectVisible: true,
    selectedProjectVisible: true,
    projectOptionCount: 2,
  };
}

function evidence(overrides = {}) {
  return summarizeProjectDirectoryEvidence({
    traceRaw: traceRaw(),
    errorRaw: "[]",
    dom: visibleDom(),
    threadId: THREAD_ID,
    selectedProjectId: SELECTED_PROJECT_ID,
    backendProjectId: SELECTED_PROJECT_ID,
    setupRequests: setupRequests(),
    ...overrides,
  });
}

describe("Project Directory Electron Gate B", () => {
  it("binds the visible Project selection to the canonical Thread over Electron IPC", () => {
    const result = evidence();

    expect(() => assertProjectDirectoryEvidence(result)).not.toThrow();
    expect(result.bridge.missingMethods).toEqual([]);
    expect(result.bridge.mockFallbackHitCount).toBe(0);
    expect(result.identity).toMatchObject({
      backendProjectMatchesSelection: true,
      metadataUpdateMatchesThreadAndProject: true,
    });
  });

  it("fails closed for mock transport or mismatched Thread and Project identity", () => {
    const mockResult = evidence({
      traceRaw: traceRaw("renderer-mock"),
    });
    expect(() => assertProjectDirectoryEvidence(mockResult)).toThrow(
      /缺少 Project current method/,
    );

    const mismatchedThread = evidence({
      traceRaw: traceRaw("electron-ipc", { threadId: "other-thread" }),
    });
    expect(() => assertProjectDirectoryEvidence(mismatchedThread)).toThrow(
      /canonical Thread\/Project identity 不一致/,
    );

    const mismatchedProject = evidence({
      backendProjectId: INITIAL_PROJECT_ID,
    });
    expect(() => assertProjectDirectoryEvidence(mismatchedProject)).toThrow(
      /thread\/read 未恢复 GUI 选择的 Project/,
    );
  });

  it("validates CLI bounds, redacts local paths, and excludes fallback owners", () => {
    expect(
      parseProjectDirectoryGateArgs(["--timeout-ms", "60000"], {
        evidenceDir: "/tmp/evidence",
        prefix: "project-directory",
        timeoutMs: 120_000,
        intervalMs: 250,
        keepTemp: false,
      }),
    ).toMatchObject({ timeoutMs: 60_000 });
    expect(() =>
      parseProjectDirectoryGateArgs(["--prefix", "../unsafe"], {
        evidenceDir: "/tmp/evidence",
        prefix: "project-directory",
        timeoutMs: 120_000,
        intervalMs: 250,
        keepTemp: false,
      }),
    ).toThrow(/invalid evidence prefix/);
    expect(
      sanitizeProjectDirectoryFailure(
        new Error("failed at /private/tmp/project-gate/workspace"),
        ["/private/tmp/project-gate"],
      ),
    ).toBe("failed at [local-path]/workspace");

    const content = fs.readFileSync(
      "scripts/electron/project-directory-gate-b.mjs",
      "utf8",
    );
    for (const method of PROJECT_DIRECTORY_REQUIRED_METHODS) {
      expect(content).toContain(method);
    }
    expect(content).toContain("ensureElectronFixtureBuild");
    expect(content).toContain('backendMode: "unavailable"');
    expect(content).toContain('data-testid="thread-project-directory"');
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain('"workspace/list"');
    expect(content).not.toContain('"workspace/create"');
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
  });
});
