import { describe, expect, it } from "vitest";
import type { AgentThreadItem, AgentThreadTurn } from "../../agentProtocol";
import { projectAppServerAgentEventPayload } from "../appServerEventStream";
import { readCanonicalThreadDetail } from "../appServerCanonicalThreadProjection";
import {
  createConversationProjectionReducer,
  boundProjectionOutput,
  MAX_PROJECTION_OUTPUT_BYTES,
  PROJECTION_OUTPUT_TRUNCATION_MARKER,
  reduceConversationProjectionPayloads,
} from "./index";

const THREAD_ID = "thread-projection";
const TURN_ID = "turn-projection";

function agentMessage(
  status: "in_progress" | "completed",
  text: string,
): AgentThreadItem {
  return {
    id: "item-message",
    thread_id: THREAD_ID,
    turn_id: TURN_ID,
    sequence: 1,
    status,
    started_at: "2026-07-28T00:00:00.000Z",
    updated_at: "2026-07-28T00:00:00.100Z",
    ...(status === "completed"
      ? { completed_at: "2026-07-28T00:00:00.200Z" }
      : {}),
    type: "agent_message",
    text,
  };
}

function turn(status: AgentThreadTurn["status"]): AgentThreadTurn {
  return {
    id: TURN_ID,
    thread_id: THREAD_ID,
    prompt_text: "核对投影",
    status,
    started_at: "2026-07-28T00:00:00.000Z",
    created_at: "2026-07-28T00:00:00.000Z",
    updated_at: "2026-07-28T00:00:00.200Z",
    ...(status === "completed"
      ? { completed_at: "2026-07-28T00:00:00.200Z" }
      : {}),
  };
}

describe("ConversationProjection reducer", () => {
  it("缓冲 started 前 delta，completed snapshot 覆盖草稿，并拒绝 terminal 后 late delta", () => {
    const reducer = createConversationProjectionReducer({
      threadId: THREAD_ID,
    });
    reducer.dispatch({
      type: "turn_started",
      source: "live",
      event_id: "turn-start",
      turn: turn("running"),
    });
    reducer.dispatch({
      type: "item_delta",
      source: "live",
      event_id: "orphan-delta",
      thread_id: THREAD_ID,
      turn_id: TURN_ID,
      item_id: "item-message",
      sequence: 2,
      delta: { kind: "text", value: "先到的片段" },
    });
    reducer.dispatch({
      type: "item_started",
      source: "live",
      event_id: "item-start",
      item: agentMessage("in_progress", ""),
    });
    expect(reducer.getProjection().items[0]).toMatchObject({
      type: "agent_message",
      text: "先到的片段",
    });

    reducer.dispatch({
      type: "item_completed",
      source: "read",
      event_id: "item-complete",
      item: agentMessage("completed", "权威完整快照"),
    });
    reducer.dispatch({
      type: "item_delta",
      source: "live",
      event_id: "late-delta",
      thread_id: THREAD_ID,
      turn_id: TURN_ID,
      item_id: "item-message",
      sequence: 4,
      delta: { kind: "text", value: "不应追加" },
    });

    expect(reducer.getProjection().items[0]).toMatchObject({
      status: "completed",
      text: "权威完整快照",
    });
    expect(
      reducer.getProjection().diagnostics.map((item) => item.code),
    ).toEqual(["item_delta_before_start", "late_delta_rejected"]);
  });

  it("重复 event id 不改变 projection，且按 canonical sequence 保持 Item 顺序", () => {
    const reducer = createConversationProjectionReducer({
      threadId: THREAD_ID,
    });
    const first = agentMessage("completed", "第一条");
    const second = { ...first, id: "item-second", sequence: 2, text: "第二条" };
    reducer.dispatch({
      type: "item_completed",
      source: "replay",
      event_id: "same",
      item: second,
    });
    const firstProjection = reducer.getProjection();
    reducer.dispatch({
      type: "item_completed",
      source: "replay",
      event_id: "same",
      item: first,
    });
    expect(reducer.getProjection()).toEqual(firstProjection);
  });

  it("completed snapshot 保留 Item 首次 sequence 与 started_at", () => {
    const reducer = createConversationProjectionReducer({
      threadId: THREAD_ID,
    });
    reducer.dispatch({
      type: "item_started",
      source: "live",
      event_id: "identity-start",
      item: {
        ...agentMessage("in_progress", "draft"),
        sequence: 3,
        started_at: "2026-07-28T00:00:00.100Z",
      },
    });
    reducer.dispatch({
      type: "item_completed",
      source: "live",
      event_id: "identity-complete",
      item: {
        ...agentMessage("completed", "authoritative"),
        sequence: 99,
        started_at: "2026-07-28T00:00:09.900Z",
        completed_at: "2026-07-28T00:00:10.000Z",
      },
    });

    expect(reducer.getProjection().items[0]).toMatchObject({
      sequence: 3,
      started_at: "2026-07-28T00:00:00.100Z",
      completed_at: "2026-07-28T00:00:10.000Z",
      status: "completed",
      text: "authoritative",
    });
  });

  it("复用 direct notification 的 tool_output_delta 时，CommandExecution 输出仍进入同一 Item", () => {
    const reducer = createConversationProjectionReducer({
      threadId: THREAD_ID,
    });
    const command: AgentThreadItem = {
      id: "item-command",
      thread_id: THREAD_ID,
      turn_id: TURN_ID,
      sequence: 2,
      status: "in_progress",
      started_at: "2026-07-28T00:00:00.000Z",
      updated_at: "2026-07-28T00:00:00.000Z",
      type: "command_execution",
      command: "printf ready",
      cwd: "/workspace",
    };
    reducer.dispatch({
      type: "item_started",
      source: "live",
      event_id: "command-start",
      item: command,
    });
    reducer.dispatch({
      type: "item_delta",
      source: "live",
      event_id: "command-output",
      thread_id: THREAD_ID,
      turn_id: TURN_ID,
      item_id: command.id,
      sequence: 3,
      delta: { kind: "output", value: "ready\n" },
    });

    expect(reducer.getProjection().items[0]).toMatchObject({
      type: "command_execution",
      aggregated_output: "ready\n",
    });
  });

  it("CommandExecution 只保留最新 20 条 terminal interaction 摘要", () => {
    const reducer = createConversationProjectionReducer({
      threadId: THREAD_ID,
    });
    const command: AgentThreadItem = {
      id: "item-command-interactions",
      thread_id: THREAD_ID,
      turn_id: TURN_ID,
      sequence: 2,
      status: "in_progress",
      started_at: "2026-07-28T00:00:00.000Z",
      updated_at: "2026-07-28T00:00:00.000Z",
      type: "command_execution",
      command: "read input",
      cwd: "/workspace",
    };
    reducer.dispatch({
      type: "item_started",
      source: "live",
      event_id: "command-interactions-start",
      item: command,
    });
    for (let index = 0; index < 21; index += 1) {
      reducer.dispatch({
        type: "item_delta",
        source: "live",
        event_id: `command-interaction-${index}`,
        thread_id: THREAD_ID,
        turn_id: TURN_ID,
        item_id: command.id,
        sequence: 3 + index,
        delta: {
          kind: "terminal_interaction",
          process_id: "unified-exec-1000",
          stdin: `sent ${index} chars`,
        },
      });
    }

    const projected = reducer.getProjection().items[0];
    expect(projected?.type).toBe("command_execution");
    if (projected?.type !== "command_execution") {
      throw new Error("expected command_execution projection");
    }
    expect(projected.terminal_interactions).toHaveLength(20);
    expect(projected.terminal_interactions?.[0]?.stdin).toBe("sent 1 chars");
    expect(projected.terminal_interactions?.[19]?.stdin).toBe("sent 20 chars");
  });

  it("CommandExecution delta 与 completed snapshot 都应限制在 256 KiB 并保留尾部", () => {
    const reducer = createConversationProjectionReducer({
      threadId: THREAD_ID,
    });
    const oversizedDelta = `delta-head-${"d".repeat(
      MAX_PROJECTION_OUTPUT_BYTES + 128,
    )}-delta-tail`;
    const command: AgentThreadItem = {
      id: "item-command-bounded",
      thread_id: THREAD_ID,
      turn_id: TURN_ID,
      sequence: 2,
      status: "in_progress",
      started_at: "2026-07-28T00:00:00.000Z",
      updated_at: "2026-07-28T00:00:00.000Z",
      type: "command_execution",
      command: "generate-output",
      cwd: "/workspace",
    };
    reducer.dispatch({
      type: "item_started",
      source: "live",
      event_id: "bounded-command-start",
      item: command,
    });
    reducer.dispatch({
      type: "item_delta",
      source: "live",
      event_id: "bounded-command-delta",
      thread_id: THREAD_ID,
      turn_id: TURN_ID,
      item_id: command.id,
      sequence: 3,
      delta: { kind: "output", value: oversizedDelta },
    });

    const streamed = reducer.getProjection().items[0];
    expect(streamed?.type).toBe("command_execution");
    if (streamed?.type !== "command_execution") {
      throw new Error("expected command_execution projection");
    }
    expect(
      new TextEncoder().encode(streamed.aggregated_output).byteLength,
    ).toBe(MAX_PROJECTION_OUTPUT_BYTES);
    expect(streamed.aggregated_output).toMatch(
      new RegExp(`^\\${PROJECTION_OUTPUT_TRUNCATION_MARKER[0]}`),
    );
    expect(streamed.aggregated_output).toContain(
      PROJECTION_OUTPUT_TRUNCATION_MARKER,
    );
    expect(streamed.aggregated_output).toMatch(/delta-tail$/u);

    const oversizedSnapshot = `snapshot-head-${"s".repeat(
      MAX_PROJECTION_OUTPUT_BYTES + 128,
    )}-snapshot-tail`;
    reducer.dispatch({
      type: "item_completed",
      source: "read",
      event_id: "bounded-command-complete",
      item: {
        ...command,
        status: "completed",
        completed_at: "2026-07-28T00:00:00.200Z",
        updated_at: "2026-07-28T00:00:00.200Z",
        aggregated_output: oversizedSnapshot,
      },
    });

    const completed = reducer.getProjection().items[0];
    expect(completed?.type).toBe("command_execution");
    if (completed?.type !== "command_execution") {
      throw new Error("expected completed command_execution projection");
    }
    expect(
      new TextEncoder().encode(completed.aggregated_output).byteLength,
    ).toBe(MAX_PROJECTION_OUTPUT_BYTES);
    expect(completed.aggregated_output).toContain(
      PROJECTION_OUTPUT_TRUNCATION_MARKER,
    );
    expect(completed.aggregated_output).toMatch(/snapshot-tail$/u);
  });

  it.each([
    ["CJK", "界"],
    ["emoji", "😀"],
  ])("按 UTF-8 bytes 截断 %s 输出且不切断 Unicode 字符", (_label, unit) => {
    const bounded = boundProjectionOutput(
      `discarded-head-${unit.repeat(MAX_PROJECTION_OUTPUT_BYTES)}-kept-tail`,
    );
    const byteLength = new TextEncoder().encode(bounded).byteLength;
    const retained = bounded.slice(PROJECTION_OUTPUT_TRUNCATION_MARKER.length);
    const firstRetainedCodeUnit = retained.charCodeAt(0);

    expect(byteLength).toBeLessThanOrEqual(MAX_PROJECTION_OUTPUT_BYTES);
    expect(byteLength).toBeGreaterThan(MAX_PROJECTION_OUTPUT_BYTES - 4);
    expect(bounded.startsWith(PROJECTION_OUTPUT_TRUNCATION_MARKER)).toBe(true);
    expect(bounded).toMatch(/-kept-tail$/u);
    expect(
      firstRetainedCodeUnit < 0xdc00 || firstRetainedCodeUnit > 0xdfff,
    ).toBe(true);
  });

  it("unknown Item 只记录一次带 revision、method、type 与脱敏字段名的 drift", () => {
    const reducer = createConversationProjectionReducer({
      threadId: THREAD_ID,
    });
    const unknownItem: AgentThreadItem = {
      id: "item-future",
      thread_id: THREAD_ID,
      turn_id: TURN_ID,
      sequence: 7,
      status: "in_progress",
      started_at: "2026-07-28T00:00:00.000Z",
      updated_at: "2026-07-28T00:00:00.000Z",
      type: "unknown_item",
      upstream_type: "futureWidget",
      field_names: ["displayName", "[redacted]"],
    };
    reducer.dispatch({
      type: "item_started",
      source: "live",
      event_id: "future-item-start",
      protocol_revision: "upstream-revision",
      protocol_method: "item/started",
      item: unknownItem,
    });
    reducer.dispatch({
      type: "item_completed",
      source: "live",
      event_id: "future-item-complete",
      protocol_revision: "upstream-revision",
      protocol_method: "item/completed",
      item: {
        ...unknownItem,
        status: "completed",
        completed_at: "2026-07-28T00:00:00.200Z",
        updated_at: "2026-07-28T00:00:00.200Z",
      },
    });

    expect(
      reducer
        .getProjection()
        .diagnostics.filter(
          (diagnostic) => diagnostic.code === "protocol_drift",
        ),
    ).toEqual([
      expect.objectContaining({
        event_id: "future-item-start",
        field_names: ["displayName", "[redacted]"],
        item_id: "item-future",
        protocol_method: "item/started",
        protocol_revision: "upstream-revision",
        source: "live",
        upstream_type: "futureWidget",
      }),
    ]);
  });

  it("live、thread/read、replay 通过同一 reducer 得到等价 Item 投影", () => {
    const rawItem = {
      id: "item-message",
      type: "agentMessage",
      text: "权威完整快照",
      phase: "final",
      sequence: 1_753_132_800_000,
    };
    const read = readCanonicalThreadDetail({
      thread: {
        id: THREAD_ID,
        sessionId: "session-projection",
        status: { type: "idle" },
        createdAt: 1_753_132_800,
        updatedAt: 1_753_132_801,
        turns: [
          {
            id: TURN_ID,
            status: "completed",
            startedAt: 1_753_132_800,
            completedAt: 1_753_132_801,
            items: [rawItem],
          },
        ],
      },
    });
    expect(read).not.toBeNull();

    const notifications = [
      {
        method: "thread/started",
        params: {
          thread: {
            id: THREAD_ID,
            createdAt: 1_753_132_800,
            updatedAt: 1_753_132_800,
          },
        },
      },
      {
        method: "turn/started",
        params: {
          threadId: THREAD_ID,
          turn: {
            id: TURN_ID,
            status: "inProgress",
            startedAt: 1_753_132_800,
          },
        },
      },
      {
        method: "item/started",
        params: {
          threadId: THREAD_ID,
          turnId: TURN_ID,
          startedAtMs: 1_753_132_800_000,
          item: { ...rawItem, text: "" },
        },
      },
      {
        method: "item/agentMessage/delta",
        params: {
          threadId: THREAD_ID,
          turnId: TURN_ID,
          itemId: "item-message",
          delta: "权威完整快照",
        },
      },
      {
        method: "item/completed",
        params: {
          threadId: THREAD_ID,
          turnId: TURN_ID,
          completedAtMs: 1_753_132_800_000,
          item: rawItem,
        },
      },
      {
        method: "turn/completed",
        params: {
          threadId: THREAD_ID,
          turn: {
            id: TURN_ID,
            status: "completed",
            startedAt: 1_753_132_800,
            completedAt: 1_753_132_801,
          },
        },
      },
    ].map((notification) => projectAppServerAgentEventPayload(notification));
    const payloads = notifications.filter(
      (payload): payload is Record<string, unknown> => payload !== null,
    );
    const live = reduceConversationProjectionPayloads(
      payloads,
      "live",
      THREAD_ID,
    ).getProjection();
    const replay = reduceConversationProjectionPayloads(
      payloads,
      "replay",
      THREAD_ID,
    ).getProjection();

    expect(live.items).toEqual(read?.items);
    expect(replay.items).toEqual(read?.items);
    expect(live.turns.map(({ id, status }) => ({ id, status }))).toEqual([
      { id: TURN_ID, status: "completed" },
    ]);
  });
});
