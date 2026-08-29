import fs from "node:fs";

import { describe, expect, it } from "vitest";

import {
  assertThreadQueueEvidence,
  parseThreadQueueGateArgs,
  summarizeThreadQueueEvidence,
  THREAD_QUEUE_MARKER,
  THREAD_QUEUE_REQUIRED_METHODS,
  THREAD_QUEUE_TITLE,
} from "./thread-queue-gate-b.mjs";

const THREAD_ID = "thread-queue-gate-b";

function setupRequests(transport = "electron-ipc") {
  return [
    {
      command: "app_server_handle_json_lines",
      method: "thread/start",
      transport,
      status: "success",
      params: {},
    },
    {
      command: "app_server_handle_json_lines",
      method: "thread/queue/add",
      transport,
      status: "success",
      params: { threadId: THREAD_ID },
    },
  ];
}

function traceRaw(transport = "electron-ipc") {
  const methods = [
    "thread/start",
    "thread/queue/add",
    "thread/queue/list",
    "thread/read",
  ];
  return JSON.stringify(
    methods.map((method) => ({
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: method,
              method,
              params: method === "thread/start" ? {} : { threadId: THREAD_ID },
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
    headerTitle: THREAD_QUEUE_TITLE,
    queueStatusVisible: true,
    queueStatusText: "待处理 1",
    queueItemsPresent: false,
    queueStatusContainsMarker: false,
    timelineMarkerCount: 1,
  };
}

describe("Thread Queue Electron Gate B", () => {
  it("binds current Electron methods and visible queue to one Thread", () => {
    const evidence = summarizeThreadQueueEvidence({
      traceRaw: traceRaw(),
      errorRaw: "[]",
      dom: visibleDom(),
      threadId: THREAD_ID,
      queuedSubmissionId: "queued-1",
    });
    expect(() => assertThreadQueueEvidence(evidence)).not.toThrow();
    expect(evidence.bridge.missingMethods).toEqual([]);
    expect(evidence.identity.queueRequestsMatchThread).toBe(true);
  });

  it("combines preload setup evidence with GUI safeInvoke trace after reload", () => {
    const guiTrace = JSON.stringify(JSON.parse(traceRaw()).slice(2));
    const evidence = summarizeThreadQueueEvidence({
      traceRaw: guiTrace,
      errorRaw: "[]",
      dom: visibleDom(),
      threadId: THREAD_ID,
      queuedSubmissionId: "queued-1",
      setupRequests: setupRequests(),
    });
    expect(() => assertThreadQueueEvidence(evidence)).not.toThrow();
    expect(evidence.bridge.missingMethods).toEqual([]);
  });

  it("fails closed for mock transport or mismatched Thread identity", () => {
    const mockEvidence = summarizeThreadQueueEvidence({
      traceRaw: traceRaw("renderer-mock"),
      errorRaw: "[]",
      dom: visibleDom(),
      threadId: THREAD_ID,
      queuedSubmissionId: "queued-1",
    });
    expect(() => assertThreadQueueEvidence(mockEvidence)).toThrow(
      /缺少 Queue current method/,
    );

    const mismatched = summarizeThreadQueueEvidence({
      traceRaw: traceRaw(),
      errorRaw: "[]",
      dom: visibleDom(),
      threadId: "other-thread",
      queuedSubmissionId: "queued-1",
    });
    expect(() => assertThreadQueueEvidence(mismatched)).toThrow(
      /Thread identity 不一致/,
    );
  });

  it("fails closed when Queue status repeats canonical text or timeline is not unique", () => {
    const repeatedInStatus = summarizeThreadQueueEvidence({
      traceRaw: traceRaw(),
      errorRaw: "[]",
      dom: {
        ...visibleDom(),
        queueStatusContainsMarker: true,
        queueStatusText: `待处理 1 ${THREAD_QUEUE_MARKER}`,
      },
      threadId: THREAD_ID,
      queuedSubmissionId: "queued-1",
    });
    expect(() => assertThreadQueueEvidence(repeatedInStatus)).toThrow(
      /不得重复 canonical submission 正文/,
    );

    const duplicatedTimeline = summarizeThreadQueueEvidence({
      traceRaw: traceRaw(),
      errorRaw: "[]",
      dom: { ...visibleDom(), timelineMarkerCount: 2 },
      threadId: THREAD_ID,
      queuedSubmissionId: "queued-1",
    });
    expect(() => assertThreadQueueEvidence(duplicatedTimeline)).toThrow(
      /canonical 时间线中唯一可见/,
    );
  });

  it("validates CLI bounds and keeps retired queue paths absent", () => {
    expect(
      parseThreadQueueGateArgs(["--timeout-ms", "60000"], {
        evidenceDir: "/tmp/evidence",
        prefix: "thread-queue",
        timeoutMs: 120_000,
        intervalMs: 250,
        keepTemp: false,
      }),
    ).toMatchObject({ timeoutMs: 60_000 });
    expect(() =>
      parseThreadQueueGateArgs(["--prefix", "../unsafe"], {
        evidenceDir: "/tmp/evidence",
        prefix: "thread-queue",
        timeoutMs: 120_000,
        intervalMs: 250,
        keepTemp: false,
      }),
    ).toThrow(/invalid evidence prefix/);

    const content = fs.readFileSync(
      "scripts/electron/thread-queue-gate-b.mjs",
      "utf8",
    );
    const queueComponent = fs.readFileSync(
      "src/components/agent/chat/components/ThreadQueueStatus.tsx",
      "utf8",
    );
    for (const method of THREAD_QUEUE_REQUIRED_METHODS) {
      expect(content).toContain(method);
    }
    expect(content).toContain("ensureElectronFixtureBuild");
    expect(content).toContain('backendMode: "unavailable"');
    expect(content).toContain('data-testid="thread-queue-status"');
    expect(content).toContain('data-testid="thread-queue-items"');
    expect(queueComponent).not.toContain('data-testid="thread-queue-items"');
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain("queuedTurnSnapshot");
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
  });
});
