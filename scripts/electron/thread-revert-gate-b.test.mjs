import fs from "node:fs";
import { describe, expect, it } from "vitest";

import {
  assertThreadRevertEvidence,
  parseThreadRevertGateArgs,
  summarizeThreadRevertEvidence,
  THREAD_REVERT_FIRST_ASSISTANT_MARKER,
  THREAD_REVERT_FIRST_USER_MARKER,
  THREAD_REVERT_REQUIRED_METHODS,
  THREAD_REVERT_SECOND_ASSISTANT_MARKER,
  THREAD_REVERT_SECOND_USER_MARKER,
  THREAD_REVERT_TITLE,
  THREAD_REVERT_WORKSPACE_CONTENT,
} from "./thread-revert-gate-b.mjs";

const THREAD_ID = "thread-revert-gate-b";
const FIRST_TURN_ID = "turn-revert-first";
const SECOND_TURN_ID = "turn-revert-second";

function setupRequests(transport = "electron-ipc") {
  return [
    {
      command: "app_server_handle_json_lines",
      method: "thread/start",
      transport,
      status: "success",
      params: {},
    },
    ...[FIRST_TURN_ID, SECOND_TURN_ID].map((turnId) => ({
      command: "app_server_handle_json_lines",
      method: "turn/start",
      transport,
      status: "success",
      params: { threadId: THREAD_ID, turnId },
    })),
  ];
}

function actionTrace({
  transport = "electron-ipc",
  beforeTurnId = SECOND_TURN_ID,
} = {}) {
  return JSON.stringify([
    {
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: "revert",
              method: "thread/revert",
              params: { threadId: THREAD_ID, beforeTurnId },
            }),
          ],
        },
      },
    },
    {
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: "read",
              method: "thread/read",
              params: { threadId: THREAD_ID },
            }),
          ],
        },
      },
    },
    {
      command: "app_server_drain_events",
      transport,
      status: "success",
    },
  ]);
}

function fixtureEvidence(overrides = {}) {
  return summarizeThreadRevertEvidence({
    traceRaw: actionTrace(),
    errorRaw: "[]",
    beforeDom: {
      headerTitle: THREAD_REVERT_TITLE,
      firstUserVisible: true,
      firstAssistantVisible: true,
      secondUserVisible: true,
      secondAssistantVisible: true,
      targetTriggerVisible: true,
      targetThreadId: THREAD_ID,
      targetBeforeTurnId: SECOND_TURN_ID,
    },
    dialogDom: {
      dialogVisible: true,
      explainsHistoryRemoval: true,
      explainsThreadPreserved: true,
      explainsFilesPreserved: true,
      confirmVisible: true,
    },
    afterDom: {
      headerTitle: THREAD_REVERT_TITLE,
      firstUserVisible: true,
      firstAssistantVisible: true,
      secondUserVisible: false,
      secondAssistantVisible: false,
      statusState: "success",
    },
    threadId: THREAD_ID,
    firstTurnId: FIRST_TURN_ID,
    secondTurnId: SECOND_TURN_ID,
    setupRequests: setupRequests(),
    backendLedger: [
      {
        kind: "turnStart",
        threadId: THREAD_ID,
        turnId: FIRST_TURN_ID,
        inputText: THREAD_REVERT_FIRST_USER_MARKER,
        assistantText: THREAD_REVERT_FIRST_ASSISTANT_MARKER,
        eventTypes: ["message.delta", "message.completed", "turn.completed"],
      },
      {
        kind: "turnStart",
        threadId: THREAD_ID,
        turnId: SECOND_TURN_ID,
        inputText: THREAD_REVERT_SECOND_USER_MARKER,
        assistantText: THREAD_REVERT_SECOND_ASSISTANT_MARKER,
        eventTypes: ["message.delta", "message.completed", "turn.completed"],
      },
    ],
    workspaceContentBefore: THREAD_REVERT_WORKSPACE_CONTENT,
    workspaceContentAfter: THREAD_REVERT_WORKSPACE_CONTENT,
    ...overrides,
  });
}

describe("Thread Revert Electron Gate B", () => {
  it("binds the GUI action, canonical refresh and preserved workspace to one Thread", () => {
    const evidence = fixtureEvidence();
    expect(() => assertThreadRevertEvidence(evidence)).not.toThrow();
    expect(evidence.bridge.missingMethods).toEqual([]);
    expect(evidence.identity.revertRequestMatchesSecondTurn).toBe(true);
    expect(evidence.gui.secondTurnRemovedAfter).toBe(true);
    expect(evidence.workspace.preserved).toBe(true);
  });

  it("fails closed when the second turn remains or the workspace file changes", () => {
    const historyMismatch = fixtureEvidence({
      afterDom: {
        headerTitle: THREAD_REVERT_TITLE,
        firstUserVisible: true,
        firstAssistantVisible: true,
        secondUserVisible: true,
        secondAssistantVisible: false,
        statusState: "success",
      },
    });
    expect(() => assertThreadRevertEvidence(historyMismatch)).toThrow(
      /未移除第二轮/u,
    );

    const workspaceMismatch = fixtureEvidence({
      workspaceContentAfter: "changed",
    });
    expect(() => assertThreadRevertEvidence(workspaceMismatch)).toThrow(
      /修改了工作区文件/u,
    );
  });

  it("rejects a mismatched Revert target and non-Electron transport", () => {
    const wrongTurn = fixtureEvidence({
      traceRaw: actionTrace({ beforeTurnId: FIRST_TURN_ID }),
    });
    expect(() => assertThreadRevertEvidence(wrongTurn)).toThrow(
      /未精确命中第二个 canonical Turn/u,
    );

    const wrongTransport = fixtureEvidence({
      traceRaw: actionTrace({ transport: "renderer-mock" }),
    });
    expect(() => assertThreadRevertEvidence(wrongTransport)).toThrow(
      /app_server_drain_events|缺少 Thread Revert current method/u,
    );
  });

  it("validates CLI bounds and keeps retired or mock paths absent", () => {
    expect(
      parseThreadRevertGateArgs(["--timeout-ms", "60000", "--keep-temp"]),
    ).toMatchObject({ timeoutMs: 60_000, keepTemp: true });
    expect(() => parseThreadRevertGateArgs(["--prefix", "../unsafe"])).toThrow(
      /invalid evidence prefix/u,
    );

    const source = fs.readFileSync(
      "scripts/electron/thread-revert-gate-b.mjs",
      "utf8",
    );
    for (const method of THREAD_REVERT_REQUIRED_METHODS) {
      expect(source).toContain(method);
    }
    expect(source).toContain("ensureElectronFixtureBuild");
    expect(source).toContain('backendMode: "external"');
    expect(source).toContain('type: "message.delta"');
    expect(source).toContain('type: "turn.completed"');
    expect(source).toContain('data-testid="thread-revert-trigger"');
    expect(source).toContain('data-testid="thread-revert-confirm"');
    expect(source).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(source).not.toContain("thread/rollback");
    expect(source).not.toContain("mockPriorityCommands");
    expect(source).not.toContain("defaultMocks");
    expect(source).not.toContain("invokeMockOnly");
    expect(source).not.toContain("agent_runtime_");
  });
});
