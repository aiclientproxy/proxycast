import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AppServerEventBusSubscription } from "@/lib/api/appServerEventBus";
import {
  APP_SERVER_METHOD_AGENT_SESSION_EVENT,
  type AppServerJsonRpcNotification,
} from "@/lib/api/appServer";
import { createAgentRuntimeEventListener } from "../agentRuntimeEvents";
import {
  AppServerAgentSessionEventDrainRouter,
  projectAppServerAgentEventPayload,
  publishAppServerAgentSessionNotificationsFromPipeline,
} from "./appServerEventStream";
import {
  projectAgentRuntimeSequenceGateNotifications,
  resetAgentRuntimeEventSequenceGatesForTests,
} from "./eventSequenceGate";
import {
  projectAppServerV2NotificationPayload,
  readAppServerV2NotificationRoute,
} from "./appServerV2Notification";
import {
  conversationProjectionEventFromPayload,
  createConversationProjectionReducer,
} from "./conversationProjection";

const threadId = "thread-v2";
const turnId = "turn-v2";

function directNotification(
  method: string,
  params: Record<string, unknown>,
): AppServerJsonRpcNotification {
  return { method, params };
}

function turn(status: string): Record<string, unknown> {
  return {
    id: turnId,
    items: [],
    itemsView: "full",
    status,
    startedAt: 1_783_814_400,
    ...(status === "inProgress" ? {} : { completedAt: 1_783_814_401 }),
    ...(status === "failed" ? { error: { message: "fixture failed" } } : {}),
  };
}

function tokenUsageBreakdown({
  inputTokens,
}: {
  inputTokens: number;
}): Record<string, number> {
  return {
    totalTokens: inputTokens,
    inputTokens,
    cachedInputTokens: 0,
    outputTokens: 0,
    reasoningOutputTokens: 0,
  };
}

describe("App Server v2 direct notifications", () => {
  beforeEach(() => {
    resetAgentRuntimeEventSequenceGatesForTests();
  });

  it("projects direct lifecycle notifications into the existing GUI payloads", () => {
    const notifications = [
      directNotification("thread/started", {
        thread: {
          id: threadId,
          createdAt: 1_783_814_399,
        },
      }),
      directNotification("turn/started", {
        threadId,
        turn: turn("inProgress"),
      }),
      directNotification("item/started", {
        item: {
          id: "item-v2",
          text: "",
          type: "agentMessage",
        },
        startedAtMs: 1_783_814_400_100,
        threadId,
        turnId,
      }),
      directNotification("item/agentMessage/delta", {
        delta: "hello",
        itemId: "item-v2",
        threadId,
        turnId,
      }),
      directNotification("item/completed", {
        completedAtMs: 1_783_814_400_900,
        item: {
          id: "item-v2",
          text: "hello",
          type: "agentMessage",
        },
        threadId,
        turnId,
      }),
      directNotification("turn/completed", {
        threadId,
        turn: turn("completed"),
      }),
    ];

    const projected = notifications.map(projectAppServerAgentEventPayload);

    expect(projected.map((payload) => payload?.type)).toEqual([
      "thread_started",
      "turn_started",
      "item_started",
      "text_delta",
      "item_completed",
      "turn_completed",
    ]);
    expect(projected[1]).toMatchObject({
      turn: {
        id: turnId,
        status: "running",
        thread_id: threadId,
      },
    });
    expect(projected[2]).toMatchObject({
      sequence: 0,
      sequence_provenance: "notification_order",
      item: {
        id: "item-v2",
        sequence: 0,
        status: "in_progress",
        text: "",
        type: "agent_message",
      },
    });
    expect(projected[3]).toMatchObject({
      item_id: "item-v2",
      text: "hello",
      thread_id: threadId,
      turn_id: turnId,
    });
  });

  it("uses first-seen notification order instead of timestamps for live Item sequence", () => {
    const firstStarted = projectAppServerV2NotificationPayload(
      directNotification("item/started", {
        item: { id: "item-first", text: "first", type: "agentMessage" },
        startedAtMs: 1_783_814_400_900,
        threadId,
        turnId,
      }),
    );
    const secondStarted = projectAppServerV2NotificationPayload(
      directNotification("item/started", {
        item: { id: "item-second", text: "second", type: "agentMessage" },
        startedAtMs: 1_783_814_400_100,
        threadId,
        turnId,
      }),
    );
    const firstCompleted = projectAppServerV2NotificationPayload(
      directNotification("item/completed", {
        completedAtMs: 1_783_814_401_500,
        item: { id: "item-first", text: "first done", type: "agentMessage" },
        threadId,
        turnId,
      }),
    );

    expect([
      (firstStarted?.item as { sequence?: number } | undefined)?.sequence,
      (secondStarted?.item as { sequence?: number } | undefined)?.sequence,
      (firstCompleted?.item as { sequence?: number } | undefined)?.sequence,
    ]).toEqual([0, 1, 0]);
    expect(firstStarted).toMatchObject({
      sequence: 0,
      sequence_provenance: "notification_order",
    });
  });

  it("projects paired hook notifications into one canonical item identity", () => {
    const hook = {
      completedAt: 1_783_814_401_500,
      displayOrder: 0,
      entries: [{ kind: "feedback", text: "检查完成" }],
      eventName: "preToolUse",
      executionMode: "sync",
      handlerType: "command",
      id: "hook-run-v2",
      scope: "turn",
      source: "project",
      sourcePath: "/workspace/.codex/hooks/check.sh",
      startedAt: 1_783_814_400_900,
      status: "completed",
      statusMessage: "检查完成",
    };
    const started = {
      ...hook,
      completedAt: null,
      status: "running",
    };
    const startedNotification = directNotification("hook/started", {
      run: started,
      threadId,
      turnId,
    });
    const completedNotification = directNotification("hook/completed", {
      run: hook,
      threadId,
      turnId,
    });

    expect(readAppServerV2NotificationRoute(startedNotification)).toEqual({
      itemId: "item_hook-run-v2",
      terminal: false,
      threadId,
      turnId,
    });
    expect(
      projectAppServerV2NotificationPayload(startedNotification),
    ).toMatchObject({
      item_id: "item_hook-run-v2",
      item: {
        id: "item_hook-run-v2",
        run_id: "hook-run-v2",
        status: "in_progress",
        type: "hook",
      },
      type: "item_started",
    });
    expect(readAppServerV2NotificationRoute(completedNotification)).toEqual({
      itemId: "item_hook-run-v2",
      terminal: true,
      threadId,
      turnId,
    });
    expect(
      projectAppServerV2NotificationPayload(completedNotification),
    ).toMatchObject({
      item_id: "item_hook-run-v2",
      item: {
        id: "item_hook-run-v2",
        run_id: "hook-run-v2",
        status: "completed",
        type: "hook",
      },
      type: "item_completed",
    });
  });

  it("projects typed warning and preserves the localization code", () => {
    const notification = directNotification("warning", {
      code: "skill_not_available",
      message: "技能不可用，已继续执行。",
      threadId,
    });

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      terminal: false,
      threadId,
    });
    expect(projectAppServerV2NotificationPayload(notification)).toMatchObject({
      code: "skill_not_available",
      message: "技能不可用，已继续执行。",
      protocol_method: "warning",
      session_id: threadId,
      thread_id: threadId,
      type: "warning",
    });
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2_warning",
        notification,
      ),
    ).toEqual([notification]);
  });

  it("projects guardianWarning as an independent high-priority notice", () => {
    const notification = directNotification("guardianWarning", {
      message: "Guardian review rejected too many requests; turn interrupted.",
      threadId,
    });

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      terminal: false,
      threadId,
    });
    const payload = projectAppServerV2NotificationPayload(notification);
    expect(payload).toMatchObject({
      code: "guardian_warning",
      message: "Guardian review rejected too many requests; turn interrupted.",
      protocol_method: "guardianWarning",
      type: "guardian_warning",
    });
    const event = conversationProjectionEventFromPayload(
      payload as Record<string, unknown>,
      "live",
      "guardian-warning-1",
    );
    expect(event).toMatchObject({
      type: "guardian_warning",
      thread_id: threadId,
      message: "Guardian review rejected too many requests; turn interrupted.",
    });
    const reducer = createConversationProjectionReducer({ threadId });
    reducer.dispatch(event!);
    expect(reducer.getProjection().notices).toContainEqual({
      id: "guardian-warning-1",
      thread_id: threadId,
      level: "warning",
      code: "guardian_warning",
      message: "Guardian review rejected too many requests; turn interrupted.",
    });
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2_guardian_warning",
        notification,
      ),
    ).toEqual([notification]);
  });

  it("rejects guardianWarning payloads with missing or extra fields", () => {
    expect(
      readAppServerV2NotificationRoute(
        directNotification("guardianWarning", {
          message: "missing thread",
        }),
      ),
    ).toBeNull();
    expect(
      readAppServerV2NotificationRoute(
        directNotification("guardianWarning", {
          message: "extra field",
          threadId,
          severity: "high",
        }),
      ),
    ).toBeNull();
  });

  it("unknown Item 只投影脱敏字段名，不保留原始字段值", () => {
    const secretValue = "secret-value-must-not-leak";
    const projected = projectAppServerV2NotificationPayload(
      directNotification("item/started", {
        item: {
          id: "future-item-v2",
          type: "unknownItem",
          upstreamType: "futureWidget",
          fieldNames: ["[redacted]", "displayName"],
          displayName: "safe display value",
          authorization: secretValue,
          apiKey: secretValue,
          "invalid field": secretValue,
        },
        startedAtMs: 1_783_814_400_100,
        threadId,
        turnId,
      }),
    );

    expect(projected).toMatchObject({
      type: "item_started",
      item: {
        id: "future-item-v2",
        type: "unknown_item",
        upstream_type: "futureWidget",
        field_names: ["[redacted]", "displayName"],
      },
    });
    expect(JSON.stringify(projected)).not.toContain(secretValue);
    expect(JSON.stringify(projected)).not.toContain("authorization");
    expect(JSON.stringify(projected)).not.toContain("apiKey");
  });

  it.each([
    ["completed", "turn_completed", "completed"],
    ["failed", "turn_failed", "failed"],
    ["interrupted", "turn_canceled", "interrupted"],
  ])("maps terminal turn status %s", (status, type, projectedStatus) => {
    expect(
      projectAppServerV2NotificationPayload(
        directNotification("turn/completed", {
          threadId,
          turn: turn(status),
        }),
      ),
    ).toMatchObject({
      type,
      turn: {
        id: turnId,
        status: projectedStatus,
      },
    });
  });

  it("projects the canonical final answer from a completed turn", () => {
    const completedTurn = turn("completed");
    completedTurn.items = [
      {
        id: "commentary-v2",
        phase: "commentary",
        text: "checking",
        type: "agentMessage",
      },
      {
        id: "final-v2",
        phase: "final_answer",
        text: "approval completed",
        type: "agentMessage",
      },
    ];

    expect(
      projectAppServerV2NotificationPayload(
        directNotification("turn/completed", {
          threadId,
          turn: completedTurn,
        }),
      ),
    ).toMatchObject({
      text: "approval completed",
      type: "turn_completed",
    });
  });

  it("projects current thread token usage onto the active turn", () => {
    const notification = directNotification("thread/tokenUsage/updated", {
      threadId,
      turnId,
      tokenUsage: {
        total: {
          totalTokens: 31_000,
          inputTokens: 31_000,
          cachedInputTokens: 0,
          cacheWriteInputTokens: 1_200,
          outputTokens: 0,
          reasoningOutputTokens: 0,
        },
        last: {
          totalTokens: 31_000,
          inputTokens: 31_000,
          cachedInputTokens: 0,
          cacheWriteInputTokens: 1_200,
          outputTokens: 0,
          reasoningOutputTokens: 0,
        },
        modelContextWindow: 128_000,
      },
    });

    expect(projectAppServerV2NotificationPayload(notification)).toMatchObject({
      type: "token_usage_updated",
      thread_id: threadId,
      turn_id: turnId,
      usage: {
        input_tokens: 31_000,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_input_tokens: 1_200,
      },
    });
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2_usage",
        notification,
      ),
    ).toEqual([notification]);
  });

  it("projects indexed reasoning notifications without collapsing their semantics", () => {
    const notifications = [
      directNotification("item/started", {
        item: {
          id: "reasoning-v2",
          summary: [],
          content: [],
          type: "reasoning",
        },
        startedAtMs: 1_783_814_400_100,
        threadId,
        turnId,
      }),
      directNotification("item/reasoning/summaryTextDelta", {
        delta: "summary",
        itemId: "reasoning-v2",
        summaryIndex: 0,
        threadId,
        turnId,
      }),
      directNotification("item/reasoning/summaryPartAdded", {
        itemId: "reasoning-v2",
        summaryIndex: 1,
        threadId,
        turnId,
      }),
      directNotification("item/reasoning/textDelta", {
        contentIndex: 0,
        delta: "raw",
        itemId: "reasoning-v2",
        threadId,
        turnId,
      }),
      directNotification("item/completed", {
        completedAtMs: 1_783_814_400_900,
        item: {
          id: "reasoning-v2",
          summary: ["summary"],
          content: ["raw"],
          type: "reasoning",
        },
        threadId,
        turnId,
      }),
    ];

    const projected = notifications.flatMap((notification) =>
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2_reasoning",
        notification,
      ).map(projectAppServerAgentEventPayload),
    );

    expect(projected.map((payload) => payload?.type)).toEqual([
      "item_started",
      "reasoning_summary_delta",
      "reasoning_summary_part_added",
      "reasoning_content_delta",
      "item_completed",
    ]);
    expect(projected[1]).toMatchObject({
      delta: "summary",
      itemId: "reasoning-v2",
      item_id: "reasoning-v2",
      reasoningId: "reasoning-v2",
      reasoning_id: "reasoning-v2",
      summaryIndex: 0,
      summary_index: 0,
      text: "summary",
    });
    expect(projected[2]).toMatchObject({
      item_id: "reasoning-v2",
      summaryIndex: 1,
      summary_index: 1,
    });
    expect(projected[3]).toMatchObject({
      contentIndex: 0,
      content_index: 0,
      delta: "raw",
      item_id: "reasoning-v2",
      text: "raw",
    });
    expect(projected[4]).toMatchObject({
      item: {
        id: "reasoning-v2",
        summary: ["summary"],
        content: ["raw"],
        status: "completed",
      },
    });
  });

  it("projects typed plan delta with canonical item identity", () => {
    const notification = directNotification("item/plan/delta", {
      delta: "\n- [ ] 接 GUI",
      itemId: "plan_turn-v2_proposed_plan:1",
      threadId,
      turnId,
    });

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      itemId: "plan_turn-v2_proposed_plan:1",
      terminal: false,
      threadId,
      turnId,
    });
    expect(projectAppServerV2NotificationPayload(notification)).toMatchObject({
      type: "plan_delta",
      text: "\n- [ ] 接 GUI",
      delta: "\n- [ ] 接 GUI",
      sourceItemId: "plan_turn-v2_proposed_plan:1",
      revisionId: "plan_turn-v2_proposed_plan:1",
      source: "app_server_v2",
      thread_id: threadId,
      turn_id: turnId,
    });
  });

  it("projects command execution output delta into the existing tool item shape", () => {
    const notification = directNotification(
      "item/commandExecution/outputDelta",
      {
        delta: "stdout\n",
        itemId: "command-v2",
        threadId,
        turnId,
      },
    );

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      itemId: "command-v2",
      terminal: false,
      threadId,
      turnId,
    });
    expect(projectAppServerV2NotificationPayload(notification)).toMatchObject({
      type: "tool_output_delta",
      tool_id: "command-v2",
      sourceItemId: "command-v2",
      source_item_id: "command-v2",
      delta: "stdout\n",
      thread_id: threadId,
      turn_id: turnId,
    });
  });

  it("projects a terminal interaction as a redacted command item delta", () => {
    const notification = directNotification(
      "item/commandExecution/terminalInteraction",
      {
        itemId: "command-v2",
        processId: "unified-exec-1000",
        stdin: "sent 9 chars",
        threadId,
        turnId,
      },
    );

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      itemId: "command-v2",
      terminal: false,
      threadId,
      turnId,
    });
    expect(projectAppServerV2NotificationPayload(notification)).toMatchObject({
      type: "terminal_interaction",
      item_id: "command-v2",
      process_id: "unified-exec-1000",
      stdin: "sent 9 chars",
      thread_id: threadId,
      turn_id: turnId,
    });
  });

  it("feeds direct command notifications through the projection reducer and lets completed snapshot win", () => {
    const reducer = createConversationProjectionReducer({ threadId });
    const dispatch = (
      eventId: string,
      method: string,
      params: Record<string, unknown>,
    ) => {
      const payload = projectAppServerV2NotificationPayload(
        directNotification(method, params),
      );
      expect(payload).not.toBeNull();
      const event = conversationProjectionEventFromPayload(
        payload ?? {},
        "live",
        eventId,
      );
      expect(event).not.toBeNull();
      if (event) {
        reducer.dispatch(event);
      }
    };

    dispatch("command-start", "item/started", {
      item: {
        id: "command-v2",
        type: "commandExecution",
        status: "inProgress",
        command: "printf first && printf second",
        cwd: "/workspace",
      },
      startedAtMs: 1_783_814_400_100,
      threadId,
      turnId,
    });
    dispatch("command-output-1", "item/commandExecution/outputDelta", {
      delta: "first\n",
      itemId: "command-v2",
      threadId,
      turnId,
    });
    dispatch("command-output-2", "item/commandExecution/outputDelta", {
      delta: "second\n",
      itemId: "command-v2",
      threadId,
      turnId,
    });
    dispatch("command-input-1", "item/commandExecution/terminalInteraction", {
      itemId: "command-v2",
      processId: "unified-exec-1000",
      stdin: "sent 9 chars",
      threadId,
      turnId,
    });

    expect(reducer.getProjection().items[0]).toMatchObject({
      id: "command-v2",
      type: "command_execution",
      status: "in_progress",
      aggregated_output: "first\nsecond\n",
      terminal_interactions: [
        { process_id: "unified-exec-1000", stdin: "sent 9 chars" },
      ],
    });

    dispatch("command-complete", "item/completed", {
      completedAtMs: 1_783_814_400_900,
      item: {
        id: "command-v2",
        type: "commandExecution",
        status: "completed",
        command: "printf first && printf second",
        cwd: "/workspace",
        aggregatedOutput: "authoritative snapshot\n",
        exitCode: 0,
      },
      threadId,
      turnId,
    });

    expect(reducer.getProjection().items[0]).toMatchObject({
      id: "command-v2",
      type: "command_execution",
      status: "completed",
      aggregated_output: "authoritative snapshot\n",
      exit_code: 0,
      terminal_interactions: [
        { process_id: "unified-exec-1000", stdin: "sent 9 chars" },
      ],
    });
  });

  it("projects file change patch updates into the existing patch item", () => {
    const changes = [
      {
        diff: "-old\n+new",
        kind: { type: "update", move_path: "src/main.ts" },
        path: "src/index.ts",
      },
    ];
    const notification = directNotification("item/fileChange/patchUpdated", {
      changes,
      itemId: "item_patch-1",
      threadId,
      turnId,
    });

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      itemId: "item_patch-1",
      terminal: false,
      threadId,
      turnId,
    });
    const projected = projectAppServerV2NotificationPayload(notification);
    expect(projected).toMatchObject({
      type: "item_updated",
      item: {
        changes,
        file_status: "inProgress",
        id: "item_patch-1",
        paths: ["src/index.ts"],
        status: "in_progress",
        success: false,
        type: "patch",
      },
      thread_id: threadId,
      turn_id: turnId,
    });
    const projectedItem = projected?.item as
      | { text?: string }
      | null
      | undefined;
    expect(JSON.parse(projectedItem?.text ?? "null")).toEqual(changes);
  });

  it("projects MCP progress onto the canonical MCP tool item identity", () => {
    const notification = directNotification("item/mcpToolCall/progress", {
      itemId: "item_mcp-call-1",
      message: "正在读取文档索引",
      threadId,
      turnId,
    });

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      itemId: "item_mcp-call-1",
      terminal: false,
      threadId,
      turnId,
    });
    expect(projectAppServerV2NotificationPayload(notification)).toMatchObject({
      type: "tool_progress",
      tool_id: "item_mcp-call-1",
      thread_id: threadId,
      turn_id: turnId,
      progress: {
        message: "正在读取文档索引",
        metadata: {
          notification_kind: "mcp_progress",
          source: "app_server_v2",
          source_item_id: "item_mcp-call-1",
        },
      },
    });
  });

  it.each([
    [
      "item/reasoning/summaryTextDelta",
      { delta: "missing index", itemId: "reasoning-v2", threadId, turnId },
    ],
    [
      "item/reasoning/summaryPartAdded",
      { itemId: "reasoning-v2", threadId, turnId },
    ],
    [
      "item/reasoning/textDelta",
      { contentIndex: 0, itemId: "reasoning-v2", threadId, turnId },
    ],
    ["item/plan/delta", { itemId: "plan-v2", threadId, turnId }],
    [
      "item/commandExecution/outputDelta",
      { itemId: "command-v2", threadId, turnId },
    ],
    [
      "item/fileChange/patchUpdated",
      {
        changes: [
          { diff: "", kind: { type: "update", move_path: 42 }, path: "a.ts" },
        ],
        itemId: "item_patch-1",
        threadId,
        turnId,
      },
    ],
    [
      "item/mcpToolCall/progress",
      { itemId: "item_mcp-call-1", message: "   ", threadId, turnId },
    ],
    ["warning", { message: "   ", threadId }],
    ["warning", { code: 42, message: "warning", threadId }],
  ])("fails closed for malformed %s", (method, params) => {
    const notification = directNotification(method, params);

    expect(readAppServerV2NotificationRoute(notification)).toBeNull();
    expect(projectAppServerV2NotificationPayload(notification)).toBeNull();
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2_malformed_reasoning",
        notification,
      ),
    ).toEqual([]);
  });

  it("accepts valid direct notifications through the current lifecycle verifier", () => {
    const notification = directNotification("turn/started", {
      threadId,
      turn: turn("inProgress"),
    });

    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2",
        notification,
      ),
    ).toEqual([notification]);
  });

  it("fails closed for malformed direct notification identity", () => {
    const notification = directNotification("turn/started", {
      threadId,
      turn: { status: "inProgress" },
    });

    expect(readAppServerV2NotificationRoute(notification)).toBeNull();
    expect(projectAppServerV2NotificationPayload(notification)).toBeNull();
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2_invalid",
        notification,
      ),
    ).toEqual([]);
  });

  it("routes drained direct notifications and closes on turn/completed", async () => {
    let subscription: AppServerEventBusSubscription | undefined;
    const eventBus = {
      subscribe(next: AppServerEventBusSubscription) {
        subscription = next;
        return vi.fn();
      },
    };
    const router = new AppServerAgentSessionEventDrainRouter(
      { drainEvents: () => [] },
      eventBus,
    );
    const received: unknown[] = [];
    const listen = createAgentRuntimeEventListener({
      listen: vi.fn().mockResolvedValue(vi.fn()),
    });
    const unlisten = await listen("agent_stream_direct_v2_route", (event) => {
      received.push(event.payload);
    });
    router.register({
      eventName: "agent_stream_direct_v2_route",
      sessionId: threadId,
    });

    subscription?.onNotifications?.([
      directNotification("turn/started", {
        threadId,
        turn: turn("inProgress"),
      }),
      directNotification("thread/tokenUsage/updated", {
        threadId,
        turnId,
        tokenUsage: {
          total: tokenUsageBreakdown({ inputTokens: 31_000 }),
          last: tokenUsageBreakdown({ inputTokens: 31_000 }),
          modelContextWindow: null,
        },
      }),
      directNotification("turn/completed", {
        threadId,
        turn: turn("completed"),
      }),
      directNotification("item/agentMessage/delta", {
        delta: "late",
        itemId: "item-v2",
        threadId,
        turnId,
      }),
    ]);

    expect(received).toMatchObject([
      { type: "turn_started" },
      {
        type: "token_usage_updated",
        turn_id: turnId,
        usage: { input_tokens: 31_000 },
      },
      { type: "turn_completed" },
    ]);
    unlisten();
  });

  it("取消回合关闭旧 route 后，旧 delta 不应绑定到下一回合", async () => {
    let subscription: AppServerEventBusSubscription | undefined;
    const eventBus = {
      subscribe(next: AppServerEventBusSubscription) {
        subscription = next;
        return vi.fn();
      },
    };
    const router = new AppServerAgentSessionEventDrainRouter(
      { drainEvents: () => [] },
      eventBus,
    );
    const oldEventName = "agent_stream_cancelled_old";
    const nextEventName = "agent_stream_cancelled_next";
    const oldTurnId = "turn-cancelled-old";
    const nextTurnId = "turn-cancelled-next";
    const received: unknown[] = [];
    const listen = createAgentRuntimeEventListener({
      listen: vi.fn().mockResolvedValue(vi.fn()),
    });
    const unlisten = await listen(nextEventName, (event) => {
      received.push(event.payload);
    });
    router.register({ eventName: oldEventName, sessionId: threadId });

    const oldDelta = directNotification("item/agentMessage/delta", {
      delta: "旧回合正文",
      itemId: "item-old",
      threadId,
      turnId: oldTurnId,
    });
    subscription?.onNotifications?.([oldDelta]);
    router.close({
      eventName: oldEventName,
      sessionId: threadId,
      turnId: oldTurnId,
    });
    router.register({ eventName: nextEventName, sessionId: threadId });

    subscription?.onNotifications?.([
      oldDelta,
      directNotification("item/agentMessage/delta", {
        delta: "新回合正文",
        itemId: "item-next",
        threadId,
        turnId: nextTurnId,
      }),
    ]);

    expect(received).toEqual([
      expect.objectContaining({
        text: "新回合正文",
        turn_id: nextTurnId,
        type: "text_delta",
      }),
    ]);
    unlisten();
  });

  it.each([
    ["response 先到", true],
    ["drain 先到", false],
  ])(
    "同一 direct delta 经 %s 时只应投影一次",
    async (_label, responseFirst) => {
      let subscription: AppServerEventBusSubscription | undefined;
      const eventBus = {
        subscribe(next: AppServerEventBusSubscription) {
          subscription = next;
          return vi.fn();
        },
      };
      const router = new AppServerAgentSessionEventDrainRouter(
        { drainEvents: () => [] },
        eventBus,
      );
      const received: unknown[] = [];
      const eventName = `agent_stream_direct_v2_mirror_${responseFirst}`;
      const listen = createAgentRuntimeEventListener({
        listen: vi.fn().mockResolvedValue(vi.fn()),
      });
      const unlisten = await listen(eventName, (event) => {
        received.push(event.payload);
      });
      const route = router.register({ eventName, sessionId: threadId });
      const notification = directNotification("item/agentMessage/delta", {
        delta: "唯一首字",
        itemId: "item-mirrored",
        threadId,
        turnId,
      });

      if (responseFirst) {
        route?.publish([notification]);
        subscription?.onNotifications?.([notification]);
      } else {
        subscription?.onNotifications?.([notification]);
        route?.publish([notification]);
      }

      expect(received).toEqual([
        expect.objectContaining({
          text: "唯一首字",
          thread_id: threadId,
          turn_id: turnId,
          type: "text_delta",
        }),
      ]);
      unlisten();
    },
  );

  it("字段顺序和顶层 envelope 不同的 direct delta 仍应识别为 response/drain 镜像", async () => {
    let subscription: AppServerEventBusSubscription | undefined;
    const eventBus = {
      subscribe(next: AppServerEventBusSubscription) {
        subscription = next;
        return vi.fn();
      },
    };
    const router = new AppServerAgentSessionEventDrainRouter(
      { drainEvents: () => [] },
      eventBus,
    );
    const received: unknown[] = [];
    const eventName = "agent_stream_direct_v2_stable_mirror";
    const listen = createAgentRuntimeEventListener({
      listen: vi.fn().mockResolvedValue(vi.fn()),
    });
    const unlisten = await listen(eventName, (event) => {
      received.push(event.payload);
    });
    const route = router.register({ eventName, sessionId: threadId });
    const responseNotification = {
      method: "item/agentMessage/delta",
      params: {
        delta: "唯一首字",
        itemId: "item-stable-mirrored",
        threadId,
        turnId,
      },
      transport: "response",
    } as AppServerJsonRpcNotification;
    const drainNotification = {
      metadata: { source: "drain" },
      method: "item/agentMessage/delta",
      params: {
        turnId,
        threadId,
        itemId: "item-stable-mirrored",
        delta: "唯一首字",
      },
    } as AppServerJsonRpcNotification;

    route?.publish([responseNotification]);
    subscription?.onNotifications?.([drainNotification]);

    expect(received).toHaveLength(1);
    expect(received[0]).toEqual(
      expect.objectContaining({ text: "唯一首字", type: "text_delta" }),
    );
    unlisten();
  });

  it("同一来源的相同 direct delta 不应被跨来源去重误删", async () => {
    const eventBus = {
      subscribe() {
        return vi.fn();
      },
    };
    const router = new AppServerAgentSessionEventDrainRouter(
      { drainEvents: () => [] },
      eventBus,
    );
    const received: unknown[] = [];
    const eventName = "agent_stream_direct_v2_same_source";
    const listen = createAgentRuntimeEventListener({
      listen: vi.fn().mockResolvedValue(vi.fn()),
    });
    const unlisten = await listen(eventName, (event) => {
      received.push(event.payload);
    });
    const route = router.register({ eventName, sessionId: threadId });
    const notification = directNotification("item/agentMessage/delta", {
      delta: "哈",
      itemId: "item-same-source",
      threadId,
      turnId,
    });

    route?.publish([notification, notification]);

    expect(received).toHaveLength(2);
    unlisten();
  });

  it("does not let a retired wrapped terminal close the direct v2 route", async () => {
    let subscription: AppServerEventBusSubscription | undefined;
    const eventBus = {
      subscribe(next: AppServerEventBusSubscription) {
        subscription = next;
        return vi.fn();
      },
    };
    const router = new AppServerAgentSessionEventDrainRouter(
      { drainEvents: () => [] },
      eventBus,
    );
    const received: unknown[] = [];
    const eventName = "agent_stream_direct_v2_after_retired_terminal";
    const listen = createAgentRuntimeEventListener({
      listen: vi.fn().mockResolvedValue(vi.fn()),
    });
    const unlisten = await listen(eventName, (event) => {
      received.push(event.payload);
    });
    router.register({ eventName, sessionId: threadId });

    subscription?.onNotifications?.([
      directNotification("turn/started", {
        threadId,
        turn: turn("inProgress"),
      }),
      directNotification(APP_SERVER_METHOD_AGENT_SESSION_EVENT, {
        event: {
          eventId: "retired-terminal",
          payload: {},
          sequence: 2,
          sessionId: threadId,
          threadId,
          timestamp: "2026-07-20T00:00:01.000Z",
          turnId,
          type: "turn.completed",
        },
      }),
      directNotification("thread/tokenUsage/updated", {
        threadId,
        turnId,
        tokenUsage: {
          total: tokenUsageBreakdown({ inputTokens: 31_000 }),
          last: tokenUsageBreakdown({ inputTokens: 31_000 }),
          modelContextWindow: null,
        },
      }),
      directNotification("turn/completed", {
        threadId,
        turn: turn("completed"),
      }),
      directNotification("item/agentMessage/delta", {
        delta: "late",
        itemId: "item-v2",
        threadId,
        turnId,
      }),
    ]);

    expect(received).toMatchObject([
      { type: "turn_started" },
      { type: "token_usage_updated", usage: { input_tokens: 31_000 } },
      { type: "turn_completed" },
    ]);
    unlisten();
  });

  it("binds a wildcard route to the first direct turn", async () => {
    let subscription: AppServerEventBusSubscription | undefined;
    const eventBus = {
      subscribe(next: AppServerEventBusSubscription) {
        subscription = next;
        return vi.fn();
      },
    };
    const router = new AppServerAgentSessionEventDrainRouter(
      { drainEvents: () => [] },
      eventBus,
    );
    const received: unknown[] = [];
    const listen = createAgentRuntimeEventListener({
      listen: vi.fn().mockResolvedValue(vi.fn()),
    });
    const eventName = "agent_stream_direct_v2_turn_binding";
    const unlisten = await listen(eventName, (event) => {
      received.push(event.payload);
    });
    router.register({ eventName, sessionId: threadId });

    subscription?.onNotifications?.([
      directNotification("item/agentMessage/delta", {
        delta: "first",
        itemId: "item-first",
        threadId,
        turnId,
      }),
      directNotification("item/agentMessage/delta", {
        delta: "other",
        itemId: "item-other",
        threadId,
        turnId: "turn-other",
      }),
    ]);

    expect(received).toEqual([
      expect.objectContaining({ text: "first", turn_id: turnId }),
    ]);
    unlisten();
  });

  it("routes current side-channel events by canonical thread identity", async () => {
    let subscription: AppServerEventBusSubscription | undefined;
    const eventBus = {
      subscribe(next: AppServerEventBusSubscription) {
        subscription = next;
        return vi.fn();
      },
    };
    const router = new AppServerAgentSessionEventDrainRouter(
      { drainEvents: () => [] },
      eventBus,
    );
    const received: unknown[] = [];
    const listen = createAgentRuntimeEventListener({
      listen: vi.fn().mockResolvedValue(vi.fn()),
    });
    const eventName = "agent_stream_image_side_channel";
    const unlisten = await listen(eventName, (event) => {
      received.push(event.payload);
    });
    router.register({ eventName, sessionId: threadId });

    subscription?.onNotifications?.([
      {
        method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        params: {
          event: {
            eventId: "event-image-task-created",
            payload: {
              response: {
                artifactPath: ".lime/tasks/image_generate/task.json",
                taskId: "task-image-1",
                taskType: "image_generate",
              },
              taskId: "task-image-1",
            },
            sequence: 9,
            sessionId: "session-v2",
            threadId,
            timestamp: "2026-07-20T00:00:01.000Z",
            turnId,
            type: "image_task.created",
          },
        },
      },
    ]);

    expect(received).toEqual([
      expect.objectContaining({
        task_id: "task-image-1",
        thread_id: threadId,
        turn_id: turnId,
        type: "image_task_created",
      }),
    ]);
    unlisten();
  });

  it("fails closed for the retired raw action side-channel", async () => {
    const received: unknown[] = [];
    const listen = createAgentRuntimeEventListener({
      listen: vi.fn().mockResolvedValue(vi.fn()),
    });
    const eventName = "agent_stream_action_side_channel";
    const unlisten = await listen(eventName, (event) => {
      received.push(event.payload);
    });

    publishAppServerAgentSessionNotificationsFromPipeline(eventName, [
      {
        method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
        params: {
          event: {
            eventId: "event-action-required",
            payload: {
              actionType: "tool_confirmation",
              prompt: "允许执行浏览器工具？",
              requestId: "approval-1",
            },
            sequence: 1,
            sessionId: threadId,
            threadId,
            timestamp: "2026-07-12T00:00:01.000Z",
            turnId,
            type: "action.required",
          },
        },
      },
    ]);

    expect(received).toEqual([]);
    unlisten();
  });

  it("projects turn/plan/updated as a strict checklist signal", () => {
    const notification = directNotification("turn/plan/updated", {
      explanation: "继续执行",
      plan: [
        { step: "读现状", status: "completed" },
        { step: "补主链", status: "inProgress" },
      ],
      threadId,
      turnId,
    });

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      terminal: false,
      threadId,
      turnId,
    });
    expect(projectAppServerV2NotificationPayload(notification)).toMatchObject({
      type: "turn_plan_updated",
      explanation: "继续执行",
      plan: [
        { step: "读现状", status: "completed" },
        { step: "补主链", status: "in_progress" },
      ],
    });
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2_turn_plan_updated",
        notification,
      ),
    ).toEqual([notification]);

    for (const params of [
      {
        explanation: "bad",
        plan: [{ step: "补主链", status: "running" }],
        threadId,
        turnId,
      },
      {
        plan: [{ step: "补主链", status: "inProgress", extra: true }],
        threadId,
        turnId,
      },
    ]) {
      const malformed = directNotification("turn/plan/updated", params);
      expect(readAppServerV2NotificationRoute(malformed)).toBeNull();
      expect(projectAppServerV2NotificationPayload(malformed)).toBeNull();
    }
  });

  it("projects turn/diff/updated into the canonical turn projection", () => {
    const notification = directNotification("turn/diff/updated", {
      diff: "diff --git a/src/a.ts b/src/a.ts\n",
      threadId,
      turnId,
    });

    expect(readAppServerV2NotificationRoute(notification)).toEqual({
      terminal: false,
      threadId,
      turnId,
    });
    const payload = projectAppServerV2NotificationPayload(notification);
    expect(payload).toMatchObject({
      type: "turn_diff_updated",
      thread_id: threadId,
      turn_id: turnId,
      unified_diff: "diff --git a/src/a.ts b/src/a.ts\n",
    });

    const reducer = createConversationProjectionReducer({ threadId });
    const turn = {
      id: turnId,
      thread_id: threadId,
      prompt_text: "",
      status: "running" as const,
      started_at: "2026-08-09T00:00:00.000Z",
      created_at: "2026-08-09T00:00:00.000Z",
      updated_at: "2026-08-09T00:00:00.000Z",
    };
    reducer.dispatch({
      type: "turn_started",
      source: "live",
      event_id: "turn-start",
      turn,
    });
    const event = conversationProjectionEventFromPayload(
      payload ?? {},
      "live",
      "turn-diff",
    );
    expect(event?.type).toBe("turn_diff_updated");
    if (event) reducer.dispatch(event);
    expect(reducer.getProjection().turns[0]?.unified_diff).toBe(
      "diff --git a/src/a.ts b/src/a.ts\n",
    );

    expect(
      projectAppServerV2NotificationPayload(
        directNotification("turn/diff/updated", {
          diff: "",
          threadId,
          turnId,
          extra: true,
        }),
      ),
    ).toBeNull();
  });

  it("projects turn/moderationMetadata as opaque last-write-wins turn state", () => {
    const firstNotification = directNotification("turn/moderationMetadata", {
      metadata: { presentation: "inline" },
      threadId,
      turnId,
    });
    expect(readAppServerV2NotificationRoute(firstNotification)).toEqual({
      terminal: false,
      threadId,
      turnId,
    });
    const firstPayload =
      projectAppServerV2NotificationPayload(firstNotification);
    expect(firstPayload).toMatchObject({
      type: "turn_moderation_metadata",
      thread_id: threadId,
      turn_id: turnId,
      moderation_metadata: { presentation: "inline" },
    });

    const reducer = createConversationProjectionReducer({ threadId });
    const runningTurn = {
      id: turnId,
      thread_id: threadId,
      prompt_text: "",
      status: "running" as const,
      started_at: "2026-08-09T00:00:00.000Z",
      created_at: "2026-08-09T00:00:00.000Z",
      updated_at: "2026-08-09T00:00:00.000Z",
    };
    reducer.dispatch({
      type: "turn_started",
      source: "live",
      event_id: "moderation-turn-start",
      turn: runningTurn,
    });
    const firstEvent = conversationProjectionEventFromPayload(
      firstPayload ?? {},
      "live",
      "moderation-first",
    );
    expect(firstEvent?.type).toBe("turn_moderation_metadata");
    if (firstEvent) reducer.dispatch(firstEvent);

    const updatedPayload = projectAppServerV2NotificationPayload(
      directNotification("turn/moderationMetadata", {
        metadata: null,
        threadId,
        turnId,
      }),
    );
    const updatedEvent = conversationProjectionEventFromPayload(
      updatedPayload ?? {},
      "live",
      "moderation-second",
    );
    if (updatedEvent) reducer.dispatch(updatedEvent);
    expect(reducer.getProjection().turns[0]?.moderation_metadata).toBeNull();

    reducer.dispatch({
      type: "turn_completed",
      source: "read",
      event_id: "moderation-terminal-snapshot",
      turn: { ...runningTurn, status: "completed" },
    });
    expect(reducer.getProjection().turns[0]?.moderation_metadata).toBeNull();

    for (const params of [
      { threadId, turnId },
      { metadata: {}, threadId, turnId, extra: true },
    ]) {
      expect(
        projectAppServerV2NotificationPayload(
          directNotification("turn/moderationMetadata", params),
        ),
      ).toBeNull();
    }
  });

  it("projects Guardian review lifecycle into pending interaction state", () => {
    const action = {
      command: "rm -rf /workspace/build",
      cwd: "/workspace",
      source: "shell",
      type: "command",
    };
    const startedNotification = directNotification(
      "item/autoApprovalReview/started",
      {
        action,
        review: { status: "inProgress" },
        reviewId: "guardian-1",
        startedAtMs: 1_783_814_400_100,
        targetItemId: "item-command",
        threadId,
        turnId,
      },
    );
    expect(readAppServerV2NotificationRoute(startedNotification)).toEqual({
      itemId: "item-command",
      terminal: false,
      threadId,
      turnId,
    });
    const startedPayload =
      projectAppServerV2NotificationPayload(startedNotification);
    expect(startedPayload).toMatchObject({
      type: "guardian_review_started",
      review_id: "guardian-1",
      target_item_id: "item-command",
      review: { status: "inProgress" },
    });

    const reducer = createConversationProjectionReducer({ threadId });
    const startedEvent = conversationProjectionEventFromPayload(
      startedPayload ?? {},
      "live",
      "guardian-start",
    );
    expect(startedEvent?.type).toBe("guardian_review_started");
    if (startedEvent) reducer.dispatch(startedEvent);
    expect(reducer.getProjection().pending_interactions).toMatchObject([
      {
        id: "guardian-1",
        kind: "guardian_review",
        status: "pending",
        item_id: "item-command",
      },
    ]);

    const strictNotification = directNotification(
      "autoApprovalReview/strictReviewRequired",
      {
        startedAtMs: 1_783_814_400_100,
        threadId,
        turnId,
      },
    );
    expect(readAppServerV2NotificationRoute(strictNotification)).toEqual({
      terminal: false,
      threadId,
      turnId,
    });
    const strictPayload =
      projectAppServerV2NotificationPayload(strictNotification);
    expect(strictPayload).toMatchObject({
      protocol_method: "autoApprovalReview/strictReviewRequired",
      server_event_emitted_at: 1_783_814_400_100,
      started_at_ms: 1_783_814_400_100,
      type: "strict_review_required",
    });
    const strictEvent = conversationProjectionEventFromPayload(
      strictPayload ?? {},
      "live",
      "guardian-strict",
    );
    expect(strictEvent?.type).toBe("strict_review_required");
    if (strictEvent) reducer.dispatch(strictEvent);
    expect(reducer.getProjection().strict_reviews).toEqual([
      {
        started_at_ms: 1_783_814_400_100,
        thread_id: threadId,
        turn_id: turnId,
      },
    ]);

    const completedNotification = directNotification(
      "item/autoApprovalReview/completed",
      {
        action,
        completedAtMs: 1_783_814_401_100,
        decisionSource: "agent",
        review: {
          rationale: "命令删除构建产物，拒绝执行。",
          riskLevel: "high",
          status: "denied",
          userAuthorization: "unknown",
        },
        reviewId: "guardian-1",
        startedAtMs: 1_783_814_400_100,
        targetItemId: "item-command",
        threadId,
        turnId,
      },
    );
    const completedPayload = projectAppServerV2NotificationPayload(
      completedNotification,
    );
    const completedEvent = conversationProjectionEventFromPayload(
      completedPayload ?? {},
      "live",
      "guardian-complete",
    );
    expect(completedEvent?.type).toBe("guardian_review_completed");
    if (completedEvent) reducer.dispatch(completedEvent);
    expect(reducer.getProjection().pending_interactions).toMatchObject([
      {
        id: "guardian-1",
        status: "declined",
        payload: {
          review: { status: "denied", risk_level: "high" },
        },
      },
    ]);
    expect(reducer.getProjection().strict_reviews).toEqual([]);
  });

  it("rejects malformed strict-review notifications", () => {
    for (const params of [
      { startedAtMs: "now", threadId, turnId },
      { startedAtMs: 1, threadId },
      { startedAtMs: 1, threadId, turnId, extra: true },
    ]) {
      const notification = directNotification(
        "autoApprovalReview/strictReviewRequired",
        params,
      );
      expect(readAppServerV2NotificationRoute(notification)).toBeNull();
      expect(projectAppServerV2NotificationPayload(notification)).toBeNull();
    }
  });

  it("rejects Guardian reviews with invalid optional field types", () => {
    const event = conversationProjectionEventFromPayload(
      {
        action: {},
        review: {
          riskLevel: 42,
          status: "inProgress",
        },
        reviewId: "guardian-invalid",
        threadId,
        turnId,
        type: "guardian_review_started",
      },
      "live",
    );

    expect(event).toBeNull();
  });
});
