import { beforeEach, describe, expect, it } from "vitest";
import type { AgentEvent, AgentThreadItem } from "@/lib/api/agentProtocol";
import {
  clearConversationProjectionDiagnostics,
  conversationProjectionStore,
  selectConversationStreamDiagnostics,
} from "../projection/conversationProjectionStore";
import {
  applyAgentStreamConversationProjection,
  type AgentStreamConversationProjectionHost,
} from "./agentStreamConversationProjection";

const THREAD_ID = "thread-stream-projection";
const TURN_ID = "turn-stream-projection";

function commandItem(): AgentThreadItem {
  return {
    id: "command-stream-projection",
    thread_id: THREAD_ID,
    turn_id: TURN_ID,
    sequence: 1,
    status: "in_progress",
    started_at: "2026-07-29T00:00:00.000Z",
    updated_at: "2026-07-29T00:00:00.000Z",
    type: "command_execution",
    command: "printf stream",
    cwd: "/workspace",
  };
}

describe("agentStreamConversationProjection", () => {
  beforeEach(() => {
    clearConversationProjectionDiagnostics();
  });

  it("为无上游 event id 的到达事件分配稳定序号并累积到同一 canonical Item", () => {
    const host = {};
    const started = applyAgentStreamConversationProjection({
      event: {
        type: "item_started",
        protocol_method: "item/started",
        item: commandItem(),
      } as AgentEvent,
      existingItems: [],
      host,
      threadId: THREAD_ID,
    });
    const firstDelta = applyAgentStreamConversationProjection({
      event: {
        type: "tool_output_delta",
        thread_id: THREAD_ID,
        turn_id: TURN_ID,
        source_item_id: "command-stream-projection",
        protocol_method: "item/commandExecution/outputDelta",
        delta: "first\n",
      } as AgentEvent,
      existingItems: [],
      host,
      threadId: THREAD_ID,
    });
    const secondDelta = applyAgentStreamConversationProjection({
      event: {
        type: "tool_output_delta",
        thread_id: THREAD_ID,
        turn_id: TURN_ID,
        source_item_id: "command-stream-projection",
        protocol_method: "item/commandExecution/outputDelta",
        delta: "second\n",
      } as AgentEvent,
      existingItems: [],
      host,
      threadId: THREAD_ID,
    });

    expect(started?.event.event_id).toBe("live:1");
    expect(firstDelta?.event.event_id).toBe("live:2");
    expect(secondDelta?.event.event_id).toBe("live:3");
    expect(secondDelta?.projection.items[0]).toMatchObject({
      id: "command-stream-projection",
      type: "command_execution",
      aggregated_output: "first\nsecond\n",
    });
  });

  it("compat 事件不应抢占 current projection owner", () => {
    const host: AgentStreamConversationProjectionHost = {};
    const compatUpdate = applyAgentStreamConversationProjection({
      event: {
        type: "tool_output_delta",
        session_id: "session-stream-projection",
        turn_id: TURN_ID,
        source_item_id: "command-stream-projection",
        delta: "compat output",
      } as AgentEvent,
      existingItems: [],
      host,
      threadId: "session-stream-projection",
    });

    expect(compatUpdate).toBeNull();
    expect(host.conversationProjectionOwner).toBeUndefined();
  });

  it("以 canonical thread identity 建立 owner，不把 session id 当作 thread id", () => {
    const update = applyAgentStreamConversationProjection({
      event: {
        type: "item_started",
        protocol_method: "item/started",
        item: commandItem(),
      } as AgentEvent,
      existingItems: [],
      host: {},
      threadId: "session-stream-projection",
    });

    expect(update?.projection.thread_id).toBe(THREAD_ID);
    expect(update?.projection.items).toEqual([
      expect.objectContaining({
        id: "command-stream-projection",
        thread_id: THREAD_ID,
      }),
    ]);
    expect(update?.projection.diagnostics).toEqual([]);
  });

  it("以已有 canonical Item 建立 owner，并拒绝跨 Thread delta", () => {
    const host: AgentStreamConversationProjectionHost = {};
    const sameThreadUpdate = applyAgentStreamConversationProjection({
      event: {
        type: "tool_output_delta",
        thread_id: THREAD_ID,
        turn_id: TURN_ID,
        source_item_id: "command-stream-projection",
        protocol_method: "item/commandExecution/outputDelta",
        delta: "same-thread\n",
      } as AgentEvent,
      existingItems: [commandItem()],
      host,
      threadId: "session-stream-projection",
    });

    expect(sameThreadUpdate?.projection.items[0]).toMatchObject({
      id: "command-stream-projection",
      aggregated_output: "same-thread\n",
    });

    const crossThreadUpdate = applyAgentStreamConversationProjection({
      event: {
        type: "tool_output_delta",
        thread_id: "thread-other",
        turn_id: TURN_ID,
        source_item_id: "command-stream-projection",
        protocol_method: "item/commandExecution/outputDelta",
        delta: "must-not-apply",
      } as AgentEvent,
      existingItems: [],
      host,
      threadId: "session-stream-projection",
    });

    expect(crossThreadUpdate?.projection.items[0]).toMatchObject({
      id: "command-stream-projection",
      aggregated_output: "same-thread\n",
    });
    expect(crossThreadUpdate?.projection.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "thread_mismatch",
        thread_id: "thread-other",
      }),
    );
  });

  it("将 unknown Item drift 写入现有 diagnostics store 且不携带原始值", () => {
    const update = applyAgentStreamConversationProjection({
      event: {
        type: "item_started",
        protocol_method: "item/started",
        protocol_revision: "future-revision",
        item: {
          id: "future-item",
          thread_id: THREAD_ID,
          turn_id: TURN_ID,
          sequence: 2,
          status: "in_progress",
          started_at: "2026-07-29T00:00:00.000Z",
          updated_at: "2026-07-29T00:00:00.000Z",
          type: "unknown_item",
          upstream_type: "futureWidget",
          field_names: ["displayName", "[redacted]"],
        },
      } as AgentEvent,
      existingItems: [],
      host: {},
      threadId: THREAD_ID,
    });

    expect(update?.projection.diagnostics).toHaveLength(1);
    const diagnostics = selectConversationStreamDiagnostics(
      conversationProjectionStore.getSnapshot(),
    );
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toMatchObject({
      phase: "protocol_drift",
      sessionId: THREAD_ID,
      source: "conversation_projection",
      metrics: {
        fieldNames: "displayName,[redacted]",
        itemId: "future-item",
        method: "item/started",
        protocolRevision: "future-revision",
        source: "live",
        upstreamType: "futureWidget",
      },
    });
    expect(JSON.stringify(diagnostics[0])).not.toContain("secret-value");
  });

  it("diagnostics 环形缓冲满载后仍将新的 unknown Item drift 写入 store", () => {
    const host: AgentStreamConversationProjectionHost = {};
    applyAgentStreamConversationProjection({
      event: {
        type: "thread_started",
        thread_id: THREAD_ID,
        protocol_method: "thread/started",
      } as AgentEvent,
      existingItems: [],
      host,
      threadId: THREAD_ID,
    });
    const owner = host.conversationProjectionOwner;
    if (!owner) {
      throw new Error("expected conversation projection owner");
    }
    for (let index = 0; index < 200; index += 1) {
      owner.reducer.dispatch({
        type: "transport_disconnected",
        source: "live",
        event_id: `mismatch:${index}`,
        thread_id: `other-thread:${index}`,
      });
    }

    const update = applyAgentStreamConversationProjection({
      event: {
        type: "item_started",
        protocol_method: "item/started",
        protocol_revision: "future-revision",
        item: {
          id: "future-item-after-ring-full",
          thread_id: THREAD_ID,
          turn_id: TURN_ID,
          sequence: 201,
          status: "in_progress",
          started_at: "2026-07-29T00:00:00.000Z",
          updated_at: "2026-07-29T00:00:00.000Z",
          type: "unknown_item",
          upstream_type: "futureWidgetAfterRingFull",
          field_names: ["displayName"],
        },
      } as AgentEvent,
      existingItems: [],
      host,
      threadId: THREAD_ID,
    });

    expect(update?.projection.diagnostics).toHaveLength(200);
    expect(
      selectConversationStreamDiagnostics(
        conversationProjectionStore.getSnapshot(),
      ),
    ).toEqual([
      expect.objectContaining({
        phase: "protocol_drift",
        metrics: expect.objectContaining({
          itemId: "future-item-after-ring-full",
          upstreamType: "futureWidgetAfterRingFull",
        }),
      }),
    ]);
  });
});
