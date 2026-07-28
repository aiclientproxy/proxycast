import assert from "node:assert/strict";
import { test } from "vitest";
import {
  isAgentMessageDeltaNotification,
  isCommandExecutionOutputDeltaNotification,
  isFileChangePatchUpdatedNotification,
  isItemCompletedNotification,
  isItemStartedNotification,
  isMcpToolCallProgressNotification,
  isModelListUpdatedNotification,
  isPlanDeltaNotification,
  isReasoningSummaryPartAddedNotification,
  isReasoningSummaryTextDeltaNotification,
  isReasoningTextDeltaNotification,
  isServerNotification,
  isThreadStartedNotification,
  isThreadSettingsUpdatedNotification,
  isTurnCompletedNotification,
  isTurnStartedNotification,
  modelListUpdatedServerNotification,
  serverNotification,
} from "../dist/index.js";

const threadId = "thread-1";
const turnId = "turn-1";

test("recognizes native v2 lifecycle and reasoning notifications", () => {
  const notifications = [
    {
      method: "thread/started",
      params: { thread: { id: threadId } },
    },
    {
      method: "turn/started",
      params: {
        threadId,
        turn: { id: turnId, status: "inProgress" },
      },
    },
    {
      method: "turn/completed",
      params: {
        threadId,
        turn: { id: turnId, status: "completed" },
      },
    },
    {
      method: "item/started",
      params: {
        item: { id: "item-1", text: "", type: "agentMessage" },
        startedAtMs: 1,
        threadId,
        turnId,
      },
    },
    {
      method: "item/completed",
      params: {
        completedAtMs: 2,
        item: { id: "item-1", text: "done", type: "agentMessage" },
        threadId,
        turnId,
      },
    },
    {
      method: "item/agentMessage/delta",
      params: { delta: "done", itemId: "item-1", threadId, turnId },
    },
    {
      method: "item/commandExecution/outputDelta",
      params: { delta: "stdout\n", itemId: "command-1", threadId, turnId },
    },
    {
      method: "item/fileChange/patchUpdated",
      params: {
        changes: [
          {
            diff: "-old\n+new",
            kind: { type: "update", move_path: "src/main.ts" },
            path: "src/index.ts",
          },
        ],
        itemId: "item_patch-1",
        threadId,
        turnId,
      },
    },
    {
      method: "item/plan/delta",
      params: { delta: "- [ ] verify", itemId: "plan-1", threadId, turnId },
    },
    {
      method: "item/mcpToolCall/progress",
      params: {
        itemId: "item_mcp-call-1",
        message: "正在读取文档索引",
        threadId,
        turnId,
      },
    },
    {
      method: "item/reasoning/summaryTextDelta",
      params: {
        delta: "summary",
        itemId: "reasoning-1",
        summaryIndex: 0,
        threadId,
        turnId,
      },
    },
    {
      method: "item/reasoning/summaryPartAdded",
      params: {
        itemId: "reasoning-1",
        summaryIndex: 1,
        threadId,
        turnId,
      },
    },
    {
      method: "item/reasoning/textDelta",
      params: {
        contentIndex: 0,
        delta: "raw reasoning",
        itemId: "reasoning-1",
        threadId,
        turnId,
      },
    },
  ];

  assert.equal(notifications.every(isServerNotification), true);
  assert.equal(isThreadStartedNotification(notifications[0]), true);
  assert.equal(isTurnStartedNotification(notifications[1]), true);
  assert.equal(isTurnCompletedNotification(notifications[2]), true);
  assert.equal(isItemStartedNotification(notifications[3]), true);
  assert.equal(isItemCompletedNotification(notifications[4]), true);
  assert.equal(isAgentMessageDeltaNotification(notifications[5]), true);
  assert.equal(
    isCommandExecutionOutputDeltaNotification(notifications[6]),
    true,
  );
  assert.equal(isFileChangePatchUpdatedNotification(notifications[7]), true);
  assert.equal(isPlanDeltaNotification(notifications[8]), true);
  assert.equal(isMcpToolCallProgressNotification(notifications[9]), true);
  assert.equal(
    isReasoningSummaryTextDeltaNotification(notifications[10]),
    true,
  );
  assert.equal(
    isReasoningSummaryPartAddedNotification(notifications[11]),
    true,
  );
  assert.equal(isReasoningTextDeltaNotification(notifications[12]), true);
  assert.equal(
    isThreadSettingsUpdatedNotification({
      method: "thread/settings/updated",
      params: {
        threadId,
        threadSettings: {
          cwd: "/tmp",
          model: "model-a",
          modelProvider: "provider-a",
        },
      },
    }),
    true,
  );
});

test("recognizes typed model list update notifications", () => {
  const providerUpdate = {
    method: "model/list/updated",
    params: { generation: 17, providerId: "openai" },
  };
  const globalUpdate = {
    method: "model/list/updated",
    params: { generation: 18, providerId: null },
  };

  assert.equal(isModelListUpdatedNotification(providerUpdate), true);
  assert.deepEqual(
    modelListUpdatedServerNotification(providerUpdate),
    providerUpdate,
  );
  assert.equal(isModelListUpdatedNotification(globalUpdate), true);
  assert.equal(isServerNotification(providerUpdate), false);
});

test("fails closed for malformed model list update notifications", () => {
  const malformed = [
    { method: "model/list/updated", params: {} },
    { method: "model/list/updated", params: { generation: -1 } },
    { method: "model/list/updated", params: { generation: 1.5 } },
    {
      method: "model/list/updated",
      params: { generation: 19, providerId: 42 },
    },
  ];

  assert.equal(
    malformed.every((message) => !isModelListUpdatedNotification(message)),
    true,
  );
});

test("fails closed for malformed or unknown notifications", () => {
  const malformed = {
    method: "turn/started",
    params: { threadId, turn: { status: "inProgress" } },
  };
  const retired = {
    method: "agentSession/event",
    params: { event: {} },
  };

  assert.equal(serverNotification(malformed), undefined);
  assert.equal(isServerNotification(malformed), false);
  assert.equal(serverNotification(retired), undefined);
  assert.equal(isServerNotification(retired), false);
});

test("fails closed for malformed reasoning notifications", () => {
  const malformed = [
    {
      method: "item/reasoning/summaryTextDelta",
      params: {
        delta: "summary",
        itemId: "reasoning-1",
        threadId,
        turnId,
      },
    },
    {
      method: "item/reasoning/summaryPartAdded",
      params: {
        itemId: "reasoning-1",
        summaryIndex: "0",
        threadId,
        turnId,
      },
    },
    {
      method: "item/reasoning/textDelta",
      params: {
        contentIndex: Number.POSITIVE_INFINITY,
        delta: "raw reasoning",
        itemId: "reasoning-1",
        threadId,
        turnId,
      },
    },
  ];

  assert.equal(
    malformed.every((message) => !isServerNotification(message)),
    true,
  );
});

test("fails closed for malformed file change patch updates", () => {
  const malformed = [
    {
      method: "item/fileChange/patchUpdated",
      params: { changes: {}, itemId: "item_patch-1", threadId, turnId },
    },
    {
      method: "item/fileChange/patchUpdated",
      params: {
        changes: [{ diff: "", kind: { type: "rename" }, path: "a.ts" }],
        itemId: "item_patch-1",
        threadId,
        turnId,
      },
    },
    {
      method: "item/fileChange/patchUpdated",
      params: {
        changes: [
          { diff: "", kind: { type: "update", move_path: 42 }, path: "a.ts" },
        ],
        itemId: "item_patch-1",
        threadId,
        turnId,
      },
    },
  ];

  assert.equal(
    malformed.every((message) => !isServerNotification(message)),
    true,
  );
});

test("fails closed for malformed MCP tool call progress", () => {
  const malformed = [
    { itemId: "item_mcp-call-1", message: "progress", threadId, turnId: "" },
    { itemId: "", message: "progress", threadId, turnId },
    { itemId: "item_mcp-call-1", message: "", threadId, turnId },
    { itemId: "item_mcp-call-1", message: "   ", threadId, turnId },
  ].map((params) => ({ method: "item/mcpToolCall/progress", params }));

  assert.equal(
    malformed.every((message) => !isServerNotification(message)),
    true,
  );
});
