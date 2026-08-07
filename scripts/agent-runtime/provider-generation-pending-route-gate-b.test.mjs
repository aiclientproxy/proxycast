import fs from "node:fs";
import { describe, expect, it } from "vitest";

import {
  buildProviderScriptedResponse,
  buildV2ThreadReadParams,
  buildV2TurnStartParams,
  canonicalRecordsWithId,
  createProviderResponseController,
  deriveDurableIdentity,
  electronCallsFromRequestLog,
  hasExactlyOneCanonicalRecord,
  hasExactlyOneTerminalCanonicalRecord,
  parseArgs,
} from "./provider-generation-pending-route-gate-b.mjs";

describe("provider-generation PendingRoute Gate B fixture", () => {
  it("parses isolated fixture arguments without accepting undersized waits", () => {
    expect(
      parseArgs([
        "--output",
        ".lime/qc/p0-05.json",
        "--timeout-ms",
        "60000",
        "--interval-ms",
        "200",
        "--cleanup-temp",
      ]),
    ).toMatchObject({
      timeoutMs: 60_000,
      intervalMs: 200,
      cleanupTemp: true,
    });
    expect(() => parseArgs(["--timeout-ms", "1000"])).toThrow(
      "--timeout-ms must be >= 30000",
    );
  });

  it("routes parent spawn, parent completion, and child completion deterministically", () => {
    const initial = buildProviderScriptedResponse({
      body: {
        messages: [
          { role: "user", content: "P0_05_PROVIDER_GENERATION_PARENT" },
        ],
      },
    });
    expect(initial).toMatchObject({
      type: "tool_call",
      name: "spawn_agent",
      arguments: {
        task_name: "pending_route_child",
        fork_turns: "none",
      },
    });

    const parentDone = buildProviderScriptedResponse({
      body: {
        messages: [
          { role: "user", content: "P0_05_PROVIDER_GENERATION_PARENT" },
          {
            role: "assistant",
            tool_calls: [{ function: { name: "spawn_agent" } }],
          },
        ],
      },
    });
    expect(parentDone).toEqual({
      type: "text",
      content: "P0_05_PROVIDER_GENERATION_PARENT_DONE",
    });

    const childDone = buildProviderScriptedResponse({
      body: {
        messages: [
          { role: "user", content: "P0_05_PROVIDER_GENERATION_CHILD" },
        ],
      },
    });
    expect(childDone).toEqual({
      type: "text",
      content: "P0_05_PROVIDER_GENERATION_CHILD_DONE",
    });
  });

  it("pauses only the first parent provider response until key deletion releases it", async () => {
    const controller = createProviderResponseController();
    let settled = false;
    const response = controller
      .respond({
        body: {
          messages: [
            { role: "user", content: "P0_05_PROVIDER_GENERATION_PARENT" },
          ],
        },
      })
      .then((value) => {
        settled = true;
        return value;
      });

    await Promise.resolve();
    expect(controller.snapshot()).toEqual({
      parentRequestBlocked: true,
      parentRequestReleased: false,
      parentPauseCount: 1,
    });
    expect(settled).toBe(false);

    controller.releaseParentRequest();
    await expect(response).resolves.toMatchObject({
      type: "tool_call",
      name: "spawn_agent",
    });
    expect(controller.snapshot()).toEqual({
      parentRequestBlocked: true,
      parentRequestReleased: true,
      parentPauseCount: 1,
    });
  });

  it("uses only canonical v2 thread/read and turn/start request fields", () => {
    expect(buildV2ThreadReadParams("thread-v2")).toEqual({
      threadId: "thread-v2",
      includeTurns: true,
    });
    const turnStart = buildV2TurnStartParams({
      clientUserMessageId: "client-v2",
      threadId: "thread-v2",
      workspaceRoot: "/workspace",
    });
    expect(turnStart).toMatchObject({
      threadId: "thread-v2",
      clientUserMessageId: "client-v2",
      input: [
        {
          type: "text",
          text: expect.stringContaining("P0_05_PROVIDER_GENERATION_PARENT"),
        },
      ],
      cwd: "/workspace",
      runtimeWorkspaceRoots: ["/workspace"],
      model: "pending-route-fixture-model",
      approvalPolicy: "never",
      sandboxPolicy: "danger-full-access",
    });
    expect(turnStart).not.toHaveProperty("sessionId");
    expect(turnStart).not.toHaveProperty("turnId");
    expect(turnStart).not.toHaveProperty("runtimeOptions");
    expect(turnStart).not.toHaveProperty("providerConfig");
  });

  it("rejects duplicate canonical mailbox turn and item records", () => {
    const duplicated = {
      thread: {
        turns: [
          { id: "mailbox-turn-1", status: "completed" },
          { id: "mailbox-turn-1", status: "completed" },
        ],
        items: [
          { id: "mailbox-item-1", status: "completed" },
          { id: "mailbox-item-1", status: "completed" },
        ],
      },
    };
    expect(canonicalRecordsWithId(duplicated, "mailbox-turn-1")).toHaveLength(
      2,
    );
    expect(
      hasExactlyOneTerminalCanonicalRecord(duplicated, "mailbox-turn-1"),
    ).toBe(false);
    expect(hasExactlyOneCanonicalRecord(duplicated, "mailbox-item-1")).toBe(
      false,
    );
    expect(
      hasExactlyOneTerminalCanonicalRecord(
        { turn: { id: "mailbox-turn-1", status: "completed" } },
        "mailbox-turn-1",
      ),
    ).toBe(true);
  });

  it("derives stable child and mailbox identities without exposing source ids", () => {
    const first = deriveDurableIdentity({
      parentThreadId: "thread-parent",
      parentTurnId: "turn-parent",
    });
    const second = deriveDurableIdentity({
      parentThreadId: "thread-parent",
      parentTurnId: "turn-parent",
    });
    expect(first).toEqual(second);
    expect(first.childSessionId).toMatch(/^agent-[0-9a-f]{64}$/);
    expect(first.childThreadId).toMatch(/^thread-[0-9a-f]{64}$/);
    expect(first.messageId).toMatch(/^agent-control-message-[0-9a-f]{64}$/);
    expect(first.mailboxTurnId).toMatch(/^mailbox-turn-[0-9a-f]{64}$/);
    expect(first.mailboxItemId).toMatch(/^item_mailbox-item-[0-9a-f]{64}$/);
    expect(JSON.stringify(first)).not.toContain("thread-parent");
    expect(JSON.stringify(first)).not.toContain("turn-parent");
  });

  it("projects only completed direct Electron calls without exposing request params", () => {
    expect(
      electronCallsFromRequestLog([
        {
          method: "modelProviderKey/create",
          params: { apiKey: "fixture-secret" },
          response: { key: { id: "key-1" } },
        },
        {
          method: "modelProviderKey/delete",
          params: { keyId: "key-1" },
          response: { deleted: true },
        },
        { method: "initialized", params: {} },
        { method: "thread/read", error: "failed" },
      ]),
    ).toEqual([
      {
        method: "modelProviderKey/create",
        transport: "electron-ipc",
        status: "success",
      },
      {
        method: "modelProviderKey/delete",
        transport: "electron-ipc",
        status: "success",
      },
      {
        method: "thread/read",
        transport: "electron-ipc",
        status: "error",
      },
    ]);
  });

  it("keeps the executable skeleton on the real Electron and generation path", () => {
    const source = fs.readFileSync(
      "scripts/agent-runtime/provider-generation-pending-route-gate-b.mjs",
      "utf8",
    );
    for (const required of [
      'backendMode: "runtime"',
      '"modelProvider/create"',
      '"modelProvider/update"',
      'taskFamilies: ["chat"]',
      'runtimeFeatures: ["streaming", "tool_calling"]',
      '"modelProviderKey/create"',
      '"modelProviderKey/delete"',
      '"thread/list"',
      '"thread/read"',
      "stage=wait-parent-provider-pause",
      "stage=delete-first-key-over-electron-ipc",
      "stage=cold-restart-electron",
      "model_route_generation",
      "childProviderRequestExactlyOnce",
      "mailboxTurnTerminalExactlyOnce",
      "childTerminalVisibleInGui",
      "credentialCommitObservedThroughElectronIpc",
      "credentialDeleteObservedThroughElectronIpc",
      "mockFallbackClear",
      "consoleErrorsClear",
      'transport === "electron-ipc"',
    ]) {
      expect(source).toContain(required);
    }
    expect(source).not.toContain("internal/refactor/v1");
    expect(source).not.toContain("{ sessionId, historyLimit: 100 }");
    expect(source).not.toContain("runtimeOptions:");
  });
});
