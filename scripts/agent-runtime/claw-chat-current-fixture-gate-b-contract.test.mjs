import fs from "node:fs";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  buildGateBContractAssertions,
  buildGateBContractEvidence,
} from "./claw-chat-current-fixture-gate-b-contract.mjs";

const RENDERER = {
  electron: true,
  hasInvokeBridge: true,
  supportsAppServer: true,
  url: "http://127.0.0.1:1420/?nativeStartup=1",
};

describe("claw chat Gate B contract evidence", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("records only sanitized current Electron bridge evidence", () => {
    vi.spyOn(fs, "existsSync").mockReturnValue(true);
    const evidence = buildGateBContractEvidence({
      rendererSnapshot: RENDERER,
      traceMessages: [
        {
          command: "app_server_handle_json_lines",
          transport: "electron-ipc",
          status: "success",
          args_preview: {
            request: {
              lines: [
                JSON.stringify({
                  jsonrpc: "2.0",
                  id: 1,
                  method: "turn/start",
                  params: {
                    sessionId: "session-1",
                    threadId: "thread-1",
                    turnId: "turn-1",
                    input: { text: "private prompt" },
                  },
                }),
              ],
            },
          },
        },
      ],
      pageErrors: [],
      pageLifecycleEvents: [],
      runId: "candidate-1",
      artifacts: {
        summary: "/tmp/candidate-1/scenario-summary.json",
        backendLedger: "/tmp/candidate-1/scenario-ledger.json",
        screenshot: "/tmp/candidate-1/scenario.png",
      },
      appServerRequests: [
        {
          method: "turn/start",
          response: {
            sessionId: "session-1",
            threadId: "thread-1",
            turnId: "turn-1",
            status: "in_progress",
          },
        },
        {
          method: "thread/read",
          response: {
            sessionId: "session-1",
            threadId: "thread-1",
            latestTurnStatus: "completed",
            turns: [{ turnId: "turn-1", status: "completed" }],
            items: [
              { itemId: "assistant-1", turnId: "turn-1", type: "message" },
            ],
          },
        },
      ],
      backendLedger: [
        { kind: "turnStart", sessionId: "session-1", turnId: "turn-1" },
        {
          kind: "backendEmit",
          sessionId: "session-1",
          turnId: "turn-1",
          eventTypes: ["message.delta", "turn.completed"],
        },
      ],
      guiEvidence: {
        sessionId: "session-1",
        turnId: "turn-1",
        itemId: "assistant-1",
        state: "terminal",
      },
      expectedIdentity: {
        sessionId: "session-1",
        threadId: "thread-1",
      },
    });

    expect(evidence.renderer).toEqual({
      electron: true,
      preloadInvoke: true,
      appServerCommandSupported: true,
      url: RENDERER.url,
    });
    expect(evidence.appServerIpcHitCount).toBe(1);
    expect(evidence.appServerIpcMethods).toEqual(["turn/start"]);
    expect(JSON.stringify(evidence)).not.toContain("private prompt");
    expect(evidence.legacyCommandHitCount).toBe(0);
    expect(evidence.mockFallbackHitCount).toBe(0);
    expect(evidence.pageErrorCount).toBe(0);
    expect(evidence.pageCrashCount).toBe(0);
    expect(evidence.identity.consistent).toBe(true);
    expect(evidence.outcome).toMatchObject({
      kind: "terminal",
      readModelStatus: "completed",
      explicit: true,
    });
    expect(Object.values(buildGateBContractAssertions(evidence))).not.toContain(
      false,
    );
  });

  it("does not infer Electron IPC usage from preload support and request logs", () => {
    vi.spyOn(fs, "existsSync").mockReturnValue(true);
    const evidence = buildGateBContractEvidence({
      rendererSnapshot: RENDERER,
      traceMessages: [],
      pageErrors: [],
      pageLifecycleEvents: [],
      runId: "candidate-without-ipc-trace",
      artifacts: {
        summary: "/tmp/candidate-without-ipc-trace/scenario-summary.json",
        backendLedger: "/tmp/candidate-without-ipc-trace/scenario-ledger.json",
        screenshot: "/tmp/candidate-without-ipc-trace/scenario.png",
      },
      appServerRequests: [{ method: "thread/read", response: {} }],
      backendLedger: [],
      guiEvidence: {},
      expectedIdentity: {},
    });

    expect(evidence.appServerRequestCount).toBe(1);
    expect(buildGateBContractAssertions(evidence)).toMatchObject({
      electronPreloadInvokeAvailable: true,
      appServerCommandSupported: true,
      electronIpcAppServerBridgeUsed: false,
    });
  });

  it("counts retired commands, mock fallback, page errors, and crashes", () => {
    vi.spyOn(fs, "existsSync").mockReturnValue(false);
    const evidence = buildGateBContractEvidence({
      rendererSnapshot: RENDERER,
      traceMessages: [
        {
          command: "agent_create_session",
          transport: "renderer-mock",
          status: "success",
        },
      ],
      pageErrors: ["boom"],
      pageLifecycleEvents: [{ type: "page-crash" }],
      runId: "",
      artifacts: {
        summary: "/tmp/candidate-1/scenario-summary.json",
        backendLedger: "/tmp/candidate-1/scenario-ledger.json",
      },
      appServerRequests: [
        {
          method: "turn/start",
          response: {
            sessionId: "server-session",
            turnId: "server-turn",
          },
        },
        {
          method: "thread/read",
          response: {
            sessionId: "read-session",
            latestTurnStatus: "running",
            turns: [{ turnId: "gui-turn", status: "running" }],
            items: [],
          },
        },
      ],
      backendLedger: [
        {
          kind: "turnStart",
          sessionId: "runtime-session",
          turnId: "runtime-turn",
        },
      ],
      guiEvidence: {
        sessionId: "gui-session",
        turnId: "gui-turn",
        state: "terminal",
      },
      expectedIdentity: { sessionId: "expected-session" },
    });

    expect(evidence.legacyCommands).toEqual(["agent_create_session"]);
    expect(evidence.legacyCommandHitCount).toBe(1);
    expect(evidence.mockFallbackHitCount).toBe(1);
    expect(evidence.pageErrorCount).toBe(1);
    expect(evidence.pageCrashCount).toBe(1);
    expect(buildGateBContractAssertions(evidence)).toMatchObject({
      runIdPresent: false,
      electronIpcAppServerBridgeUsed: false,
      screenshotCaptured: false,
      evidenceArtifactsComplete: false,
      evidenceArtifactsShareRunDirectory: true,
      noLegacyCommandHits: false,
      noMockFallbackHits: false,
      noPageErrors: false,
      noPageCrashes: false,
      identityConsistent: false,
      explicitTerminalOrPending: false,
    });
  });

  it("accepts an explicit pending GUI and read-model state", () => {
    vi.spyOn(fs, "existsSync").mockReturnValue(true);
    const evidence = buildGateBContractEvidence({
      rendererSnapshot: RENDERER,
      traceMessages: [
        {
          command: "app_server_handle_json_lines",
          transport: "electron-ipc",
          status: "success",
          args_preview: {
            request: {
              lines: [
                JSON.stringify({
                  jsonrpc: "2.0",
                  id: 1,
                  method: "turn/start",
                  params: { sessionId: "session-1", turnId: "turn-1" },
                }),
              ],
            },
          },
        },
      ],
      pageErrors: [],
      pageLifecycleEvents: [],
      runId: "candidate-1",
      artifacts: {
        summary: "/tmp/candidate-1/scenario-summary.json",
        backendLedger: "/tmp/candidate-1/scenario-ledger.json",
        screenshot: "/tmp/candidate-1/scenario.png",
      },
      appServerRequests: [
        {
          method: "turn/start",
          response: {
            sessionId: "session-1",
            turnId: "turn-1",
            status: "in_progress",
          },
        },
        {
          method: "thread/read",
          response: {
            sessionId: "session-1",
            latestTurnStatus: "in_progress",
            turns: [{ turnId: "turn-1", status: "in_progress" }],
            items: [],
          },
        },
      ],
      backendLedger: [
        { kind: "turnStart", sessionId: "session-1", turnId: "turn-1" },
      ],
      guiEvidence: {
        sessionId: "session-1",
        turnId: "turn-1",
        state: "pending",
      },
      expectedIdentity: { sessionId: "session-1" },
    });

    expect(evidence.identity.consistent).toBe(true);
    expect(evidence.outcome).toMatchObject({
      kind: "pending",
      readModelStatus: "in_progress",
      explicit: true,
    });
    expect(buildGateBContractAssertions(evidence)).toMatchObject({
      identityConsistent: true,
      explicitTerminalOrPending: true,
    });
  });

  it("matches trace identity by exact turn before same-session fallback", () => {
    vi.spyOn(fs, "existsSync").mockReturnValue(true);
    const trace = (turnId) => ({
      command: "app_server_handle_json_lines",
      transport: "electron-ipc",
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: turnId,
              method: "turn/start",
              params: { sessionId: "session-1", turnId },
            }),
          ],
        },
      },
    });
    const evidence = buildGateBContractEvidence({
      rendererSnapshot: RENDERER,
      traceMessages: [trace("turn-active"), trace("turn-queued")],
      pageErrors: [],
      pageLifecycleEvents: [],
      runId: "candidate-multi-turn",
      artifacts: {
        summary: "/tmp/candidate-multi-turn/scenario-summary.json",
        backendLedger: "/tmp/candidate-multi-turn/scenario-ledger.json",
        screenshot: "/tmp/candidate-multi-turn/scenario.png",
      },
      appServerRequests: [
        {
          method: "turn/start",
          response: {
            sessionId: "session-1",
            threadId: "thread-1",
            turnId: "event-read-probe",
            status: "completed",
          },
        },
        {
          method: "thread/read",
          response: {
            sessionId: "session-1",
            threadId: "thread-1",
            latestTurnStatus: "queued",
            turns: [
              { turnId: "turn-active", status: "completed" },
              { turnId: "turn-queued", status: "queued" },
            ],
            items: [
              {
                itemId: "item-active",
                turnId: "turn-active",
                type: "agent_message",
              },
            ],
          },
        },
      ],
      backendLedger: [
        { kind: "turnStart", sessionId: "session-1", turnId: "turn-active" },
        { kind: "turnStart", sessionId: "session-1", turnId: "turn-queued" },
        {
          kind: "backendEmit",
          sessionId: "session-1",
          turnId: "turn-active",
          eventTypes: ["turn.completed"],
        },
      ],
      guiEvidence: {
        sessionId: "session-1",
        turnId: "turn-active",
        itemId: "item-active",
        state: "terminal",
      },
      expectedIdentity: {
        sessionId: "session-1",
        threadId: "thread-1",
      },
    });

    expect(evidence.identity.sources.trace.turnId).toBe("turn-active");
    expect(evidence.identity.sources.appServer.turnId).toBeNull();
    expect(evidence.identity.consistent).toBe(true);
    expect(evidence.outcome.explicit).toBe(true);
  });
});
