import { describe, expect, it } from "vitest";

import {
  assertThreadForkEvidence,
  parseThreadForkGateArgs,
  summarizeThreadForkEvidence,
  summarizeThreadForkFailure,
  THREAD_FORK_REQUIRED_METHODS,
  THREAD_FORK_TITLE,
} from "./thread-fork-gate-b.mjs";

const PARENT_THREAD_ID = "thread-parent";
const PARENT_SESSION_ID = "session-parent";
const CHILD_THREAD_ID = "thread-child";
const CHILD_SESSION_ID = "session-child";

function trace({ transport = "electron-ipc", includeResume = true } = {}) {
  const requests = [
    {
      method: "thread/fork",
      params: { threadId: PARENT_THREAD_ID },
    },
    { method: "thread/list", params: { archived: false } },
    {
      method: "thread/read",
      params: { threadId: CHILD_THREAD_ID },
    },
    ...(includeResume
      ? [
          {
            method: "thread/resume",
            params: { threadId: CHILD_THREAD_ID },
          },
        ]
      : []),
  ];
  return JSON.stringify([
    ...requests.map((request, index) => ({
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: `fork-${index}`,
              ...request,
            }),
          ],
        },
      },
    })),
    {
      command: "app_server_drain_events",
      transport,
      status: "success",
    },
  ]);
}

function fixtureEvidence(overrides = {}) {
  return summarizeThreadForkEvidence({
    traceRaw: trace(),
    errorRaw: "[]",
    parentThreadId: PARENT_THREAD_ID,
    parentSessionId: PARENT_SESSION_ID,
    childRead: {
      thread: {
        id: CHILD_THREAD_ID,
        sessionId: CHILD_SESSION_ID,
        forkedFromId: PARENT_THREAD_ID,
      },
    },
    listedThreads: [
      { id: PARENT_THREAD_ID },
      { id: CHILD_THREAD_ID, parentThreadId: PARENT_THREAD_ID },
    ],
    notifications: [
      {
        method: "thread/started",
        params: {
          thread: {
            id: CHILD_THREAD_ID,
            forkedFromId: PARENT_THREAD_ID,
          },
        },
      },
    ],
    beforeDom: {
      headerTitle: THREAD_FORK_TITLE,
      activeSessionId: PARENT_SESSION_ID,
      parentActive: true,
    },
    menuDom: {
      actionMenuVisible: true,
      forkActionVisible: true,
    },
    afterDom: {
      headerTitle: THREAD_FORK_TITLE,
      activeThreadId: CHILD_THREAD_ID,
      activeSessionId: CHILD_SESSION_ID,
      childActive: true,
      matchingConversationCount: 2,
      successToastVisible: true,
    },
    setupRequests: [
      {
        command: "app_server_handle_json_lines",
        method: "thread/start",
        transport: "electron-ipc",
        status: "success",
        params: {},
      },
    ],
    ...overrides,
  });
}

describe("Thread Fork Electron Gate B", () => {
  it("binds the GUI action, child read model and notification to one lineage", () => {
    const evidence = fixtureEvidence();

    expect(() => assertThreadForkEvidence(evidence)).not.toThrow();
    expect(evidence.bridge.missingMethods).toEqual([]);
    expect(evidence.bridge.methods).toEqual(
      expect.arrayContaining(THREAD_FORK_REQUIRED_METHODS),
    );
    expect(evidence.identity.childReadPreservesForkLineage).toBe(true);
    expect(evidence.identity.childStartedNotificationMatches).toBe(true);
    expect(evidence.gui.childActiveAfter).toBe(true);
  });

  it("retains action evidence after the trace ring evicts early requests", () => {
    const observedActionRequests = [
      ["thread/fork", PARENT_THREAD_ID],
      ["thread/list", null],
      ["thread/read", CHILD_THREAD_ID],
      ["thread/resume", CHILD_THREAD_ID],
    ].map(([method, threadId]) => ({
      command: "app_server_handle_json_lines",
      method,
      params: threadId ? { threadId } : { archived: false },
      transport: "electron-ipc",
      status: "success",
    }));
    const evidence = fixtureEvidence({
      traceRaw: JSON.stringify([
        {
          command: "app_server_drain_events",
          transport: "electron-ipc",
          status: "success",
        },
      ]),
      observedActionRequests,
    });

    expect(() => assertThreadForkEvidence(evidence)).not.toThrow();
    expect(evidence.identity.forkRequestMatchesParent).toBe(true);
    expect(evidence.identity.childOpenMethods).toEqual([
      "thread/read",
      "thread/resume",
    ]);
  });

  it("fails closed when lineage or active session identity drifts", () => {
    const wrongParent = fixtureEvidence({
      childRead: {
        thread: {
          id: CHILD_THREAD_ID,
          sessionId: CHILD_SESSION_ID,
          forkedFromId: "thread-other",
        },
      },
    });
    expect(() => assertThreadForkEvidence(wrongParent)).toThrow(
      /forkedFromId/u,
    );

    const wrongActiveSession = fixtureEvidence({
      afterDom: {
        headerTitle: THREAD_FORK_TITLE,
        activeThreadId: CHILD_THREAD_ID,
        activeSessionId: "session-other",
        childActive: true,
        matchingConversationCount: 2,
        successToastVisible: true,
      },
    });
    expect(() => assertThreadForkEvidence(wrongActiveSession)).toThrow(
      /active session/u,
    );
  });

  it("rejects a missing resume and non-Electron transport", () => {
    const missingResume = fixtureEvidence({
      traceRaw: trace({ includeResume: false }),
    });
    expect(() => assertThreadForkEvidence(missingResume)).toThrow(
      /thread\/resume/u,
    );

    const wrongTransport = fixtureEvidence({
      traceRaw: trace({ transport: "browser-mock" }),
    });
    expect(() => assertThreadForkEvidence(wrongTransport)).toThrow(
      /drain_events|electron-ipc/u,
    );
  });

  it("validates deterministic CLI arguments", () => {
    expect(
      parseThreadForkGateArgs([
        "--prefix",
        "thread-fork-candidate",
        "--timeout-ms",
        "60000",
        "--interval-ms",
        "200",
        "--keep-temp",
      ]),
    ).toMatchObject({
      prefix: "thread-fork-candidate",
      timeoutMs: 60_000,
      intervalMs: 200,
      keepTemp: true,
    });
    expect(() => parseThreadForkGateArgs(["--prefix", "../invalid"])).toThrow(
      /invalid evidence prefix/u,
    );
  });

  it("scrubs failure diagnostics to methods and bounded UI state", () => {
    const failure = summarizeThreadForkFailure({
      traceRaw: trace(),
      errorRaw: JSON.stringify([
        {
          command: "app_server_handle_json_lines",
          transport: "electron-ipc",
          status: "error",
          error: "failed for /Users/example/private/workspace",
        },
      ]),
      toasts: [{ type: "error", text: "分叉失败" }],
      dom: {
        headerTitle: THREAD_FORK_TITLE,
        activeConversationPresent: true,
        matchingConversationCount: 1,
      },
      observer: {
        requests: [{ method: "thread/fork" }, { method: "thread/read" }],
        forkSeen: true,
        childReadSeen: true,
        resumeSeen: false,
        successToastSeen: true,
        activeSessionChanged: true,
        headerSessionChanged: true,
        composerSessionChanged: true,
      },
      consoleErrors: ["thread/fork rejected"],
    });

    expect(failure.methods).toContain("thread/fork");
    expect(failure.invokeErrors).toHaveLength(1);
    expect(failure.invokeErrors[0].error).not.toContain(
      "/Users/example/private/workspace",
    );
    expect(failure.toasts).toEqual([{ type: "error", text: "分叉失败" }]);
    expect(failure.dom).toMatchObject({
      activeConversationPresent: true,
      matchingConversationCount: 1,
    });
    expect(failure.observer).toMatchObject({
      methods: ["thread/fork", "thread/read"],
      forkSeen: true,
      childReadSeen: true,
      resumeSeen: false,
      successToastSeen: true,
      activeSessionChanged: true,
      headerSessionChanged: true,
      composerSessionChanged: true,
    });
  });
});
