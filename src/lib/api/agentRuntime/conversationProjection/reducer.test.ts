import { describe, expect, it } from "vitest";
import type { AgentThreadItem, AgentThreadTurn } from "../../agentProtocol";
import { projectAppServerAgentEventPayload } from "../appServerEventStream";
import { readCanonicalThreadDetail } from "../appServerCanonicalThreadProjection";
import {
  createConversationProjectionReducer,
  reduceConversationProjectionPayloads,
} from "./index";

const THREAD_ID = "thread-projection";
const TURN_ID = "turn-projection";

function agentMessage(status: "in_progress" | "completed", text: string): AgentThreadItem {
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
    const reducer = createConversationProjectionReducer({ threadId: THREAD_ID });
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
    expect(reducer.getProjection().diagnostics.map((item) => item.code)).toEqual([
      "item_delta_before_start",
      "late_delta_rejected",
    ]);
  });

  it("重复 event id 不改变 projection，且按 canonical sequence 保持 Item 顺序", () => {
    const reducer = createConversationProjectionReducer({ threadId: THREAD_ID });
    const first = agentMessage("completed", "第一条");
    const second = { ...first, id: "item-second", sequence: 2, text: "第二条" };
    reducer.dispatch({ type: "item_completed", source: "replay", event_id: "same", item: second });
    const firstProjection = reducer.getProjection();
    reducer.dispatch({ type: "item_completed", source: "replay", event_id: "same", item: first });
    expect(reducer.getProjection()).toEqual(firstProjection);
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
    const live = reduceConversationProjectionPayloads(payloads, "live", THREAD_ID).getProjection();
    const replay = reduceConversationProjectionPayloads(payloads, "replay", THREAD_ID).getProjection();

    expect(live.items).toEqual(read?.items);
    expect(replay.items).toEqual(read?.items);
    expect(live.turns.map(({ id, status }) => ({ id, status }))).toEqual([
      { id: TURN_ID, status: "completed" },
    ]);
  });
});
