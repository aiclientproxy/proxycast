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

  it("unknown Item 只投影脱敏字段名，不保留原始字段值", () => {
    const secretValue = "secret-value-must-not-leak";
    const projected = projectAppServerV2NotificationPayload(
      directNotification("item/started", {
        item: {
          id: "future-item-v2",
          type: "futureWidget",
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

    expect(reducer.getProjection().items[0]).toMatchObject({
      id: "command-v2",
      type: "command_execution",
      status: "in_progress",
      aggregated_output: "first\nsecond\n",
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
});
