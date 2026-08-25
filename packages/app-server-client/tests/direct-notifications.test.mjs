import assert from "node:assert/strict";
import { test } from "vitest";
import {
  isAgentMessageDeltaNotification,
  isCommandExecutionOutputDeltaNotification,
  isCommandExecutionTerminalInteractionNotification,
  isErrorNotification,
  isGuardianReviewCompletedNotification,
  isGuardianReviewStartedNotification,
  isStrictReviewRequiredNotification,
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
  isSkillsChangedNotification,
  isScheduledTaskChangedNotification,
  isScheduledTaskRunUpdatedNotification,
  isThreadStartedNotification,
  isThreadSettingsUpdatedNotification,
  isTurnPlanUpdatedNotification,
  isTurnCompletedNotification,
  isTurnStartedNotification,
  modelListUpdatedServerNotification,
  mcpServerOauthLoginCompletedServerNotification,
  mcpServerEventStreamServerNotification,
  mcpServerStatusUpdatedServerNotification,
  serverNotification,
  skillsChangedServerNotification,
  scheduledTaskChangedServerNotification,
  scheduledTaskRunUpdatedServerNotification,
} from "../dist/index.js";

const threadId = "thread-1";
const turnId = "turn-1";

test("recognizes strict skills/changed catalog invalidation", () => {
  const notification = { method: "skills/changed", params: {} };
  assert.deepEqual(skillsChangedServerNotification(notification), notification);
  assert.equal(isSkillsChangedNotification(notification), true);

  for (const malformed of [
    { method: "skills/changed" },
    { method: "skills/changed", params: null },
    { method: "skills/changed", params: { path: "/private/skill" } },
  ]) {
    assert.equal(skillsChangedServerNotification(malformed), undefined);
    assert.equal(isSkillsChangedNotification(malformed), false);
  }
});

test("recognizes strict scheduled task projection notifications", () => {
  const changed = {
    method: "scheduledTask/changed",
    params: { change: "updated", taskId: "task-1" },
  };
  const runUpdated = {
    method: "scheduledTask/run/updated",
    params: {
      attention: true,
      error: "provider unavailable",
      notificationPolicy: "failures",
      runId: "run-1",
      status: "error",
      taskId: "task-1",
      threadId: "thread-1",
      title: "Daily brief",
      turnId: "turn-1",
    },
  };

  assert.deepEqual(scheduledTaskChangedServerNotification(changed), changed);
  assert.equal(isScheduledTaskChangedNotification(changed), true);
  assert.deepEqual(
    scheduledTaskRunUpdatedServerNotification(runUpdated),
    runUpdated,
  );
  assert.equal(isScheduledTaskRunUpdatedNotification(runUpdated), true);

  for (const malformed of [
    { ...changed, params: { ...changed.params, taskId: "" } },
    { ...changed, params: { ...changed.params, change: "paused" } },
    { ...changed, params: { ...changed.params, source: "legacy" } },
  ]) {
    assert.equal(scheduledTaskChangedServerNotification(malformed), undefined);
    assert.equal(isScheduledTaskChangedNotification(malformed), false);
  }

  for (const malformed of [
    { ...runUpdated, params: { ...runUpdated.params, status: "running" } },
    {
      ...runUpdated,
      params: { ...runUpdated.params, notificationPolicy: "always" },
    },
    { ...runUpdated, params: { ...runUpdated.params, attention: "yes" } },
    { ...runUpdated, params: { ...runUpdated.params, source: "legacy" } },
  ]) {
    assert.equal(
      scheduledTaskRunUpdatedServerNotification(malformed),
      undefined,
    );
    assert.equal(isScheduledTaskRunUpdatedNotification(malformed), false);
  }
});

test("recognizes strict Guardian auto approval review lifecycle", () => {
  const action = {
    command: "git status --short",
    cwd: "/workspace",
    source: "shell",
    type: "command",
  };
  const started = {
    method: "item/autoApprovalReview/started",
    params: {
      action,
      review: { status: "inProgress" },
      reviewId: "guardian-1",
      startedAtMs: 1_783_814_400_100,
      targetItemId: "item-command",
      threadId,
      turnId,
    },
  };
  const completed = {
    method: "item/autoApprovalReview/completed",
    params: {
      action,
      completedAtMs: 1_783_814_401_100,
      decisionSource: "agent",
      review: {
        rationale: "workspace read only",
        riskLevel: "low",
        status: "approved",
        userAuthorization: "high",
      },
      reviewId: "guardian-1",
      startedAtMs: 1_783_814_400_100,
      targetItemId: "item-command",
      threadId,
      turnId,
    },
  };

  assert.deepEqual(isGuardianReviewStartedNotification(started), true);
  assert.deepEqual(isGuardianReviewCompletedNotification(completed), true);
  assert.deepEqual(serverNotification(started), started);
  assert.deepEqual(serverNotification(completed), completed);

  for (const malformed of [
    {
      ...started,
      params: { ...started.params, review: { status: "approved" } },
    },
    { ...completed, params: { ...completed.params, decisionSource: "user" } },
    { ...completed, params: { ...completed.params, extra: true } },
    {
      ...started,
      params: {
        ...started.params,
        action: { ...action, type: "unknown" },
      },
    },
  ]) {
    assert.equal(isGuardianReviewStartedNotification(malformed), false);
    assert.equal(isGuardianReviewCompletedNotification(malformed), false);
    assert.equal(serverNotification(malformed), undefined);
  }
});

test("recognizes strict review required as an independent notification", () => {
  const notification = {
    method: "autoApprovalReview/strictReviewRequired",
    params: {
      startedAtMs: 1_783_814_400_100,
      threadId,
      turnId,
    },
  };

  assert.equal(isStrictReviewRequiredNotification(notification), true);
  assert.deepEqual(serverNotification(notification), notification);

  for (const malformed of [
    { ...notification, params: { ...notification.params, turnId: "" } },
    {
      ...notification,
      params: { ...notification.params, startedAtMs: "1783814400100" },
    },
    { ...notification, params: { ...notification.params, extra: true } },
  ]) {
    assert.equal(isStrictReviewRequiredNotification(malformed), false);
    assert.equal(serverNotification(malformed), undefined);
  }
});

test("recognizes strict mcpServer/oauthLogin/completed notifications", () => {
  const success = {
    method: "mcpServer/oauthLogin/completed",
    params: {
      name: "remote-docs",
      threadId: null,
      success: true,
    },
  };
  const failure = {
    method: "mcpServer/oauthLogin/completed",
    params: {
      name: "remote-docs",
      threadId: "thread-1",
      success: false,
      error: "scope rejected",
    },
  };

  assert.deepEqual(
    mcpServerOauthLoginCompletedServerNotification(success),
    success,
  );
  assert.deepEqual(
    mcpServerOauthLoginCompletedServerNotification(failure),
    failure,
  );

  for (const malformed of [
    { method: "mcpServer/oauthLogin/completed" },
    {
      method: "mcpServer/oauthLogin/completed",
      params: { name: "remote-docs", success: true },
    },
    {
      method: "mcpServer/oauthLogin/completed",
      params: { name: "", threadId: null, success: true },
    },
    {
      method: "mcpServer/oauthLogin/completed",
      params: { name: "remote-docs", threadId: 7, success: true },
    },
    {
      method: "mcpServer/oauthLogin/completed",
      params: {
        name: "remote-docs",
        threadId: null,
        success: false,
        error: null,
      },
    },
    {
      method: "mcpServer/oauthLogin/completed",
      params: {
        name: "remote-docs",
        threadId: null,
        success: true,
        serverName: "legacy",
      },
    },
  ]) {
    assert.equal(
      mcpServerOauthLoginCompletedServerNotification(malformed),
      undefined,
    );
  }
});

test("recognizes strict mcpServer/startupStatus/updated notifications", () => {
  const ready = {
    method: "mcpServer/startupStatus/updated",
    params: {
      threadId: null,
      name: "remote-docs",
      status: "ready",
      error: null,
      failureReason: null,
    },
  };
  const failed = {
    method: "mcpServer/startupStatus/updated",
    params: {
      threadId: "thread-1",
      name: "remote-docs",
      status: "failed",
      error: "OAuth credentials expired",
      failureReason: "reauthenticationRequired",
    },
  };

  assert.deepEqual(mcpServerStatusUpdatedServerNotification(ready), ready);
  assert.deepEqual(mcpServerStatusUpdatedServerNotification(failed), failed);

  for (const malformed of [
    { method: "mcpServer/startupStatus/updated" },
    {
      method: "mcpServer/startupStatus/updated",
      params: {
        name: "remote-docs",
        status: "ready",
        error: null,
        failureReason: null,
      },
    },
    {
      method: "mcpServer/startupStatus/updated",
      params: {
        threadId: null,
        name: "",
        status: "ready",
        error: null,
        failureReason: null,
      },
    },
    {
      method: "mcpServer/startupStatus/updated",
      params: {
        threadId: null,
        name: "remote-docs",
        status: "stopped",
        error: null,
        failureReason: null,
      },
    },
    {
      method: "mcpServer/startupStatus/updated",
      params: {
        threadId: null,
        name: "remote-docs",
        status: "failed",
        error: null,
      },
    },
    {
      method: "mcpServer/startupStatus/updated",
      params: {
        threadId: null,
        name: "remote-docs",
        status: "ready",
        error: null,
        failureReason: null,
        serverName: "legacy",
      },
    },
  ]) {
    assert.equal(
      mcpServerStatusUpdatedServerNotification(malformed),
      undefined,
    );
  }
});

test("recognizes strict mcpServer/event/stream/notification notifications", () => {
  const active = {
    method: "mcpServer/event/stream/notification",
    params: {
      subscriptionId: "subscription-1",
      notification: {
        method: "notifications/events/active",
        params: { status: "active" },
      },
    },
  };
  const event = {
    method: "mcpServer/event/stream/notification",
    params: {
      subscriptionId: "subscription-1",
      notification: {
        method: "notifications/events/event",
        params: { name: "issue.updated", data: { issue: 42 } },
      },
    },
  };

  assert.deepEqual(mcpServerEventStreamServerNotification(active), active);
  assert.deepEqual(mcpServerEventStreamServerNotification(event), event);

  for (const malformed of [
    { ...active, params: { ...active.params, subscriptionId: "" } },
    { ...active, params: { ...active.params, notification: undefined } },
    {
      ...active,
      params: {
        ...active.params,
        notification: { method: "", params: {} },
      },
    },
    {
      ...active,
      params: {
        ...active.params,
        notification: { method: "notifications/events/active" },
      },
    },
    {
      ...active,
      params: { ...active.params, legacySubscriptionId: "legacy" },
    },
  ]) {
    assert.equal(mcpServerEventStreamServerNotification(malformed), undefined);
  }
});

test("recognizes native v2 lifecycle and reasoning notifications", () => {
  const notifications = [
    {
      method: "error",
      params: {
        error: {
          message: "provider stream reconnecting",
          codexErrorInfo: {
            responseStreamDisconnected: { httpStatusCode: null },
          },
          additionalDetails: null,
        },
        threadId,
        turnId,
        willRetry: true,
      },
    },
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
      method: "turn/plan/updated",
      params: {
        explanation: "继续执行",
        plan: [
          { step: "读取现状", status: "completed" },
          { step: "补齐主链", status: "inProgress" },
        ],
        threadId,
        turnId,
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
      method: "item/commandExecution/terminalInteraction",
      params: {
        itemId: "command-1",
        processId: "unified-exec-1000",
        stdin: "sent 9 chars",
        threadId,
        turnId,
      },
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
  assert.equal(isErrorNotification(notifications[0]), true);
  assert.equal(isThreadStartedNotification(notifications[1]), true);
  assert.equal(isTurnStartedNotification(notifications[2]), true);
  assert.equal(isTurnCompletedNotification(notifications[3]), true);
  assert.equal(isTurnPlanUpdatedNotification(notifications[4]), true);
  assert.equal(isItemStartedNotification(notifications[5]), true);
  assert.equal(isItemCompletedNotification(notifications[6]), true);
  assert.equal(isAgentMessageDeltaNotification(notifications[7]), true);
  assert.equal(
    isCommandExecutionOutputDeltaNotification(notifications[8]),
    true,
  );
  assert.equal(
    isCommandExecutionTerminalInteractionNotification(notifications[9]),
    true,
  );
  assert.equal(isFileChangePatchUpdatedNotification(notifications[10]), true);
  assert.equal(isPlanDeltaNotification(notifications[11]), true);
  assert.equal(isMcpToolCallProgressNotification(notifications[12]), true);
  assert.equal(
    isReasoningSummaryTextDeltaNotification(notifications[13]),
    true,
  );
  assert.equal(
    isReasoningSummaryPartAddedNotification(notifications[14]),
    true,
  );
  assert.equal(isReasoningTextDeltaNotification(notifications[15]), true);
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

test("fails closed for malformed terminal interaction notifications", () => {
  for (const params of [
    { itemId: "command-1", processId: "process-1", threadId, turnId },
    {
      itemId: "command-1",
      processId: "process-1",
      stdin: "raw-input-must-not-pass",
      threadId,
      turnId,
    },
    {
      itemId: "command-1",
      processId: "process-1",
      stdin: "sent 1 chars",
      threadId,
      turnId,
      raw: "x",
    },
  ]) {
    const notification = {
      method: "item/commandExecution/terminalInteraction",
      params,
    };
    assert.equal(serverNotification(notification), undefined);
    assert.equal(
      isCommandExecutionTerminalInteractionNotification(notification),
      false,
    );
  }
});

test("fails closed for malformed turn plan updates", () => {
  const malformed = [
    {
      method: "turn/plan/updated",
      params: {
        plan: [{ step: "补齐主链", status: "running" }],
        threadId,
        turnId,
      },
    },
    {
      method: "turn/plan/updated",
      params: {
        plan: [{ step: "补齐主链", status: "inProgress", extra: true }],
        threadId,
        turnId,
      },
    },
    {
      method: "turn/plan/updated",
      params: {
        plan: [{ step: "", status: "completed" }],
        threadId,
        turnId,
      },
    },
  ];

  assert.equal(
    malformed.every((message) => !isTurnPlanUpdatedNotification(message)),
    true,
  );
});

test("fails closed for malformed typed error notifications", () => {
  const malformed = [
    { error: { message: "failed" }, threadId, turnId },
    { error: { message: "failed" }, threadId, turnId, willRetry: "false" },
    { error: { message: "" }, threadId, turnId, willRetry: false },
    { error: { message: "failed" }, threadId: "", turnId, willRetry: false },
    { error: { message: "failed" }, threadId, turnId: "", willRetry: false },
    {
      error: { message: "failed", additionalDetails: 42 },
      threadId,
      turnId,
      willRetry: false,
    },
    {
      error: {
        message: "failed",
        codexErrorInfo: "responseStreamDisconnected",
      },
      threadId,
      turnId,
      willRetry: false,
    },
    {
      error: { message: "failed", legacyCode: "retryable" },
      threadId,
      turnId,
      willRetry: false,
    },
    {
      error: { message: "failed" },
      retryable: true,
      threadId,
      turnId,
      willRetry: false,
    },
  ].map((params) => ({ method: "error", params }));

  assert.equal(
    malformed.every((message) => !isServerNotification(message)),
    true,
  );
  assert.equal(
    malformed.every((message) => !isErrorNotification(message)),
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

test("recognizes environment connection lifecycle notifications", () => {
  for (const method of [
    "thread/environment/connected",
    "thread/environment/disconnected",
  ]) {
    const notification = {
      method,
      params: { threadId: "thread-1", environmentId: "remote-a" },
    };
    assert.deepEqual(serverNotification(notification), notification);
    assert.equal(isServerNotification(notification), true);
  }

  for (const params of [
    { threadId: "", environmentId: "remote-a" },
    { threadId: "thread-1", environmentId: "" },
    { threadId: "thread-1", environmentId: 42 },
  ]) {
    assert.equal(
      serverNotification({
        method: "thread/environment/connected",
        params,
      }),
      undefined,
    );
  }
});

test("recognizes thread queue changed notifications and fails closed", () => {
  const notification = {
    method: "thread/queue/changed",
    params: { threadId: "thread-1" },
  };
  assert.deepEqual(serverNotification(notification), notification);
  assert.equal(isServerNotification(notification), true);

  for (const params of [{}, { threadId: "" }, { threadId: 42 }]) {
    assert.equal(
      serverNotification({ method: "thread/queue/changed", params }),
      undefined,
    );
  }
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
