import { beforeEach, describe, expect, it } from "vitest";
import {
  APP_SERVER_METHOD_AGENT_SESSION_EVENT,
  type AppServerJsonRpcNotification,
} from "@/lib/api/appServer";
import {
  projectAgentRuntimeSequenceGateNotifications,
  resetAgentRuntimeEventSequenceGatesForTests,
} from "./eventSequenceGate";
import { projectAppServerAgentEventPayload } from "./appServerEventStream";

function rawNotification(
  type: string,
  payload: Record<string, unknown> = {
    provider: "fixture-provider",
    model: "fixture-model",
    elapsed_ms: 90,
  },
): AppServerJsonRpcNotification {
  return {
    method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
    params: {
      event: {
        eventId: `evt-${type}`,
        sequence: 1,
        sessionId: "session-provider-trace",
        threadId: "thread-provider-trace",
        turnId: "turn-provider-trace",
        type,
        timestamp: "2026-07-12T00:00:00.000Z",
        payload,
      },
    },
  };
}

function directNotification(
  method: string,
  params: Record<string, unknown>,
): AppServerJsonRpcNotification {
  return { method, params };
}

function actionNotification(params: {
  payload: Record<string, unknown>;
  sequence: number;
  type: "action.required" | "action.resolved";
}): AppServerJsonRpcNotification {
  return {
    method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
    params: {
      event: {
        eventId: `evt-${params.sequence}`,
        sequence: params.sequence,
        sessionId: "thread-action",
        threadId: "thread-action",
        turnId: "turn-action",
        type: params.type,
        timestamp: `2026-07-12T00:00:0${params.sequence}.000Z`,
        payload: params.payload,
      },
    },
  };
}

describe("agent runtime event sequence gate", () => {
  beforeEach(() => {
    resetAgentRuntimeEventSequenceGatesForTests();
  });

  it("放行已知 provider raw diagnostic side-channel", () => {
    const notification = rawNotification("provider.first_text_delta.received");
    const accepted = projectAgentRuntimeSequenceGateNotifications(
      "agent_stream_provider_trace",
      notification,
    );

    expect(accepted).toEqual([notification]);
    expect(projectAppServerAgentEventPayload(accepted[0])).toMatchObject({
      type: "provider_trace",
      stage: "first_text_delta_received",
      elapsed_ms: 90,
    });
  });

  it("放行 current 图片任务非 Thread side-channel 并进入既有 GUI 投影", () => {
    const notification = rawNotification("image_task.created", {
      task_id: "task-image",
      artifact_path: ".lime/tasks/image_generate/task-image.json",
    });
    const accepted = projectAgentRuntimeSequenceGateNotifications(
      "agent_stream_image_task",
      notification,
    );

    expect(accepted).toEqual([notification]);
    expect(projectAppServerAgentEventPayload(accepted[0])).toMatchObject({
      type: "image_task_created",
      task_id: "task-image",
      artifact_path: ".lime/tasks/image_generate/task-image.json",
    });
  });

  it.each(["media.read.chunk", "media.read.completed"])(
    "放行 current media read side-channel: %s",
    (type) => {
      const notification = rawNotification(type, {
        streamId: "media-stream-1",
        done: type.endsWith("completed"),
      });

      expect(
        projectAgentRuntimeSequenceGateNotifications(
          "agent_stream_media_read",
          notification,
        ),
      ).toEqual([notification]);
    },
  );

  it("接受 direct-v2 Thread/Turn/Item 完整序列", () => {
    const threadId = "thread-direct";
    const turnId = "turn-direct";
    const notifications = [
      directNotification("thread/started", {
        thread: {
          createdAt: 1_783_814_399,
          id: threadId,
          preview: "Direct lifecycle fixture",
          updatedAt: 1_783_814_399,
        },
      }),
      directNotification("turn/started", {
        threadId,
        turn: {
          id: turnId,
          items: [],
          itemsView: "full",
          startedAt: 1_783_814_400,
          status: "inProgress",
        },
      }),
      directNotification("item/started", {
        item: { id: "item-direct", text: "", type: "agentMessage" },
        startedAtMs: 1_783_814_400_100,
        threadId,
        turnId,
      }),
      directNotification("item/agentMessage/delta", {
        delta: "第一段",
        itemId: "item-direct",
        threadId,
        turnId,
      }),
      directNotification("item/completed", {
        completedAtMs: 1_783_814_400_900,
        item: { id: "item-direct", text: "第一段", type: "agentMessage" },
        threadId,
        turnId,
      }),
      directNotification("turn/completed", {
        threadId,
        turn: {
          completedAt: 1_783_814_401,
          id: turnId,
          items: [],
          itemsView: "full",
          startedAt: 1_783_814_400,
          status: "completed",
        },
      }),
    ];
    const projected = notifications.flatMap((notification) =>
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_v2",
        notification,
      ).map(projectAppServerAgentEventPayload),
    );

    expect(projected.map((payload) => payload?.type)).toEqual([
      "thread_started",
      "turn_started",
      "item_started",
      "text_delta",
      "item_completed",
      "turn_completed",
    ]);
    expect(projected[2]).toMatchObject({
      type: "item_started",
      item: {
        id: "item-direct",
        type: "agent_message",
        status: "in_progress",
      },
    });
  });

  it("只在 plan item started 与 completed 之间放行 typed delta", () => {
    const threadId = "thread-plan";
    const turnId = "turn-plan";
    const itemId = "plan_turn-plan_proposed_plan:1";
    const started = directNotification("item/started", {
      item: { id: itemId, text: "- [ ] 读协议", type: "plan" },
      startedAtMs: 1_783_814_400_100,
      threadId,
      turnId,
    });
    const delta = directNotification("item/plan/delta", {
      delta: "- [ ] 读协议",
      itemId,
      threadId,
      turnId,
    });
    const completed = directNotification("item/completed", {
      completedAtMs: 1_783_814_400_900,
      item: { id: itemId, text: "- [ ] 读协议", type: "plan" },
      threadId,
      turnId,
    });
    const eventName = "agent_stream_direct_v2_plan";

    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, delta),
    ).toEqual([]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, started),
    ).toEqual([started]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, delta),
    ).toEqual([delta]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, completed),
    ).toEqual([completed]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, delta),
    ).toEqual([]);
  });

  it("只在 commandExecution item started 与 completed 之间放行 output delta", () => {
    const threadId = "thread-command";
    const turnId = "turn-command";
    const itemId = "command-v2";
    const started = directNotification("item/started", {
      item: {
        command: "printf ready",
        cwd: "/workspace",
        id: itemId,
        status: "inProgress",
        type: "commandExecution",
      },
      startedAtMs: 1_783_814_400_100,
      threadId,
      turnId,
    });
    const delta = directNotification("item/commandExecution/outputDelta", {
      delta: "ready\n",
      itemId,
      threadId,
      turnId,
    });
    const completed = directNotification("item/completed", {
      completedAtMs: 1_783_814_400_900,
      item: {
        command: "printf ready",
        cwd: "/workspace",
        id: itemId,
        status: "completed",
        type: "commandExecution",
      },
      threadId,
      turnId,
    });
    const eventName = "agent_stream_direct_v2_command";

    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, delta),
    ).toEqual([]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, started),
    ).toEqual([started]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, delta),
    ).toEqual([delta]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, completed),
    ).toEqual([completed]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, delta),
    ).toEqual([]);
  });

  it("只在 fileChange item started 与 completed 之间放行 patch update", () => {
    const threadId = "thread-file-change";
    const turnId = "turn-file-change";
    const itemId = "item_patch-1";
    const changes = [
      {
        diff: "-old\n+new",
        kind: { type: "update" },
        path: "src/index.ts",
      },
    ];
    const started = directNotification("item/started", {
      item: {
        changes,
        id: itemId,
        status: "inProgress",
        type: "fileChange",
      },
      startedAtMs: 1_783_814_400_100,
      threadId,
      turnId,
    });
    const updated = directNotification("item/fileChange/patchUpdated", {
      changes,
      itemId,
      threadId,
      turnId,
    });
    const completed = directNotification("item/completed", {
      completedAtMs: 1_783_814_400_900,
      item: {
        changes,
        id: itemId,
        status: "completed",
        type: "fileChange",
      },
      threadId,
      turnId,
    });
    const eventName = "agent_stream_direct_v2_file_change";

    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, updated),
    ).toEqual([]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, started),
    ).toEqual([started]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, updated),
    ).toEqual([updated]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, completed),
    ).toEqual([completed]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, updated),
    ).toEqual([]);
  });

  it("只把 MCP progress 投影到同一个 canonical MCP item", () => {
    const threadId = "thread-mcp";
    const turnId = "turn-mcp";
    const itemId = "item_mcp-call-1";
    const started = directNotification("item/started", {
      item: {
        arguments: { query: "codex" },
        id: itemId,
        server: "docs",
        status: "inProgress",
        tool: "search_docs",
        type: "mcpToolCall",
      },
      startedAtMs: 1_783_814_400_100,
      threadId,
      turnId,
    });
    const progress = directNotification("item/mcpToolCall/progress", {
      itemId,
      message: "正在读取文档索引",
      threadId,
      turnId,
    });
    const completed = directNotification("item/completed", {
      completedAtMs: 1_783_814_400_900,
      item: {
        arguments: { query: "codex" },
        id: itemId,
        result: { content: [{ text: "done", type: "text" }] },
        server: "docs",
        status: "completed",
        tool: "search_docs",
        type: "mcpToolCall",
      },
      threadId,
      turnId,
    });
    const eventName = "agent_stream_direct_v2_mcp_progress";

    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, progress),
    ).toEqual([]);
    const projected = [started, progress, completed].flatMap((notification) =>
      projectAgentRuntimeSequenceGateNotifications(eventName, notification).map(
        projectAppServerAgentEventPayload,
      ),
    );

    expect(projected.map((payload) => payload?.type)).toEqual([
      "item_started",
      "tool_progress",
      "item_completed",
    ]);
    expect(projected[0]).toMatchObject({ item: { id: itemId } });
    expect(projected[1]).toMatchObject({
      tool_id: itemId,
      progress: {
        message: "正在读取文档索引",
        metadata: { notification_kind: "mcp_progress" },
      },
    });
    expect(projected[2]).toMatchObject({ item: { id: itemId } });
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, progress),
    ).toEqual([]);
  });

  it("MCP progress 对缺失 identity、空消息和非 MCP item 均 fail closed", () => {
    const threadId = "thread-mcp-negative";
    const turnId = "turn-mcp-negative";
    const itemId = "item-not-mcp";
    const eventName = "agent_stream_direct_v2_mcp_negative";
    const dynamicStarted = directNotification("item/started", {
      item: {
        arguments: {},
        id: itemId,
        status: "inProgress",
        tool: "custom_tool",
        type: "dynamicToolCall",
      },
      startedAtMs: 1_783_814_400_100,
      threadId,
      turnId,
    });

    expect(
      projectAgentRuntimeSequenceGateNotifications(
        eventName,
        directNotification("item/mcpToolCall/progress", {
          message: "progress",
          threadId,
          turnId,
        }),
      ),
    ).toEqual([]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        eventName,
        directNotification("item/mcpToolCall/progress", {
          itemId,
          message: "   ",
          threadId,
          turnId,
        }),
      ),
    ).toEqual([]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(eventName, dynamicStarted),
    ).toEqual([dynamicStarted]);
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        eventName,
        directNotification("item/mcpToolCall/progress", {
          itemId,
          message: "progress",
          threadId,
          turnId,
        }),
      ),
    ).toEqual([]);
  });

  it("malformed direct lifecycle 应 fail closed 且不抛 TypeError", () => {
    const malformed = directNotification("turn/started", {
      threadId: "thread-malformed",
      turn: {
        id: "turn-malformed",
        items: [],
        itemsView: "full",
        status: "inProgress",
      },
    });
    let accepted: AppServerJsonRpcNotification[] | undefined;

    expect(() => {
      accepted = projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_direct_malformed",
        malformed,
      );
    }).not.toThrow();
    expect(accepted).toEqual([]);
  });

  it("不放行 wrapper action required/resolved", () => {
    const required = actionNotification({
      sequence: 1,
      type: "action.required",
      payload: {
        actionType: "tool_confirmation",
        prompt: "允许执行浏览器工具？",
        requestId: "approval-1",
      },
    });
    const resolved = actionNotification({
      sequence: 2,
      type: "action.resolved",
      payload: {
        actionType: "tool_confirmation",
        approved: true,
        requestId: "approval-1",
      },
    });

    const accepted = [required, resolved].flatMap((notification) =>
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_action",
        notification,
      ),
    );
    expect(accepted).toEqual([]);
  });

  it("不放行未知 raw event", () => {
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_unknown",
        rawNotification("runtime.unknown"),
      ),
    ).toEqual([]);
  });

  it("不让 wrapper lifecycle 回流", () => {
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_thread",
        rawNotification("turn.completed"),
      ),
    ).toEqual([]);
  });
});
