import { describe, expect, it } from "vitest";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import { projectAppServerAgentEventPayload } from "./appServerEventStream";

const COMPLETED_MESSAGE_TEXT =
  "The transport kept this.\nAnd dropped this.\n\n- Finding — /tmp/file.rs:1\n  Keep ::git-stage{cwd=/tmp} literal.";

function directToolNotification(
  type: "item.started" | "item.completed",
  status: "inProgress" | "completed",
  output: Record<string, unknown> | null,
) {
  const item = {
    id: "tool-call-1",
    type: "dynamicToolCall",
    tool: "WebFetch",
    arguments: [{ name: "url", value: "https://example.com/tool" }],
    status,
    ...(output ? { result: output } : {}),
    ...(typeof output?.durationMs === "number"
      ? { durationMs: output.durationMs }
      : {}),
    ...(type === "item.completed" ? { success: true } : {}),
  };
  return {
    method: type.split(".").join("/"),
    params: {
      threadId: "thread-1",
      turnId: "turn-1",
      item,
      ...(type === "item.started"
        ? { startedAtMs: 1_752_347_281_076 }
        : { completedAtMs: 1_752_347_281_113 }),
    },
  } satisfies AppServerJsonRpcNotification;
}

function directAgentMessageSnapshotNotification() {
  return {
    method: "item/started",
    params: {
      threadId: "thread-1",
      turnId: "turn-1",
      startedAtMs: 1_752_347_281_076,
      item: {
        id: "agent-message-1",
        type: "agentMessage",
        text: "先检查，再继续核对。",
        phase: "commentary",
      },
    },
  } satisfies AppServerJsonRpcNotification;
}

function directAgentMessageCompletedNotification() {
  return {
    method: "item/completed",
    params: {
      threadId: "thread-turn-stream-repair",
      turnId: "turn-turn-stream-repair",
      completedAtMs: 1_752_347_281_113,
      item: {
        id: "message-turn-stream-repair",
        type: "agentMessage",
        text: COMPLETED_MESSAGE_TEXT,
        phase: "final",
      },
    },
  } satisfies AppServerJsonRpcNotification;
}

describe("direct v2 App Server Item projection", () => {
  it("projects a running AgentMessage snapshot from item/started", () => {
    expect(
      projectAppServerAgentEventPayload(
        directAgentMessageSnapshotNotification(),
      ),
    ).toMatchObject({
      type: "item_started",
      item: {
        id: "agent-message-1",
        status: "in_progress",
        type: "agent_message",
        text: "先检查，再继续核对。",
        phase: "commentary",
      },
    });
  });

  it("projects a completed AgentMessage with full text for stream repair", () => {
    expect(
      projectAppServerAgentEventPayload(
        directAgentMessageCompletedNotification(),
      ),
    ).toMatchObject({
      type: "item_completed",
      thread_id: "thread-turn-stream-repair",
      turn_id: "turn-turn-stream-repair",
      item: {
        id: "message-turn-stream-repair",
        status: "completed",
        type: "agent_message",
        text: COMPLETED_MESSAGE_TEXT,
        phase: "final",
      },
    });
  });

  it("projects nested item.started Tool payload into a running tool_call", () => {
    const projected = projectAppServerAgentEventPayload(
      directToolNotification("item.started", "inProgress", null),
    );

    expect(projected).toMatchObject({
      type: "item_started",
      item: {
        id: "tool-call-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        status: "in_progress",
        type: "tool_call",
        tool_name: "WebFetch",
        arguments: [{ name: "url", value: "https://example.com/tool" }],
        metadata: { callId: "tool-call-1" },
      },
    });
  });

  it("projects nested item.completed Tool output and terminal state", () => {
    const projected = projectAppServerAgentEventPayload(
      directToolNotification("item.completed", "completed", {
        text: "fetched https://example.com/tool",
        structuredContent: { source: "fixture" },
        durationMs: 37,
        truncated: false,
      }),
    );

    expect(projected).toMatchObject({
      type: "item_completed",
      item: {
        id: "tool-call-1",
        status: "completed",
        type: "tool_call",
        tool_name: "WebFetch",
        output: "fetched https://example.com/tool",
        structured_content: { source: "fixture" },
        duration_ms: 37,
        success: true,
      },
    });
  });
});
