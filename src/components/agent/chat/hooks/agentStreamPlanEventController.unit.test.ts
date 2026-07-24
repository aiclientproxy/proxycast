import { describe, expect, it } from "vitest";
import type {
  AgentEventEnvelope,
  AgentEventPlanDelta,
  AgentEventPlanFinal,
  AgentThreadItem,
} from "@/lib/api/agentProtocol";

import { buildAgentStreamPlanThreadItem } from "./agentStreamPlanEventController";

describe("agentStreamPlanEventController", () => {
  it("应把 plan.final live event 转成可恢复的 plan thread item", () => {
    const item = buildAgentStreamPlanThreadItem({
      activeSessionId: "session-1",
      fallbackTurnId: "turn-fallback",
      now: "2026-06-23T10:00:00.000Z",
      pendingItemKey: "pending-item",
      sequence: 7,
      event: {
        type: "plan_final",
        text: "- [x] 读现状\n- [ ] 打通主链",
        revisionId: "update_plan:tool-plan",
        toolCallId: "tool-plan",
        source: "update_plan",
        thread_id: "thread-1",
        turn_id: "turn-1",
        timestamp: "2026-06-23T09:59:59.000Z",
        plan: [{ step: "打通主链", status: "in_progress" }],
      } satisfies AgentEventPlanFinal & AgentEventEnvelope,
    });

    expect(item).toEqual({
      id: "plan:update_plan:tool-plan",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 7,
      status: "completed",
      started_at: "2026-06-23T09:59:59.000Z",
      completed_at: "2026-06-23T10:00:00.000Z",
      updated_at: "2026-06-23T10:00:00.000Z",
      type: "plan",
      text: "- [x] 读现状\n- [ ] 打通主链",
      metadata: {
        revisionId: "update_plan:tool-plan",
        source: "update_plan",
        plan: [{ step: "打通主链", status: "in_progress" }],
        tool_call_id: "tool-plan",
        pending_item_key: "pending-item",
      },
    });
  });

  it("plan.final 只有 structured plan 时也应生成可恢复 thread item", () => {
    const item = buildAgentStreamPlanThreadItem({
      activeSessionId: "session-1",
      fallbackTurnId: "turn-1",
      now: "2026-06-23T10:00:00.000Z",
      sequence: 8,
      event: {
        type: "plan_final",
        text: "",
        revisionId: "proposed_plan:fixture-1",
        source: "proposed_plan",
        thread_id: "thread-1",
        turn_id: "turn-1",
        plan: [
          { step: "确认计划模式请求进入 App Server", status: "completed" },
          { step: "输出 proposed_plan", status: "in_progress" },
        ],
      } satisfies AgentEventPlanFinal & AgentEventEnvelope,
    });

    expect(item).toMatchObject({
      id: "plan:proposed_plan:fixture-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      type: "plan",
      status: "completed",
      text: "- 确认计划模式请求进入 App Server\n" + "- 输出 proposed_plan",
      metadata: {
        revisionId: "proposed_plan:fixture-1",
        source: "proposed_plan",
        plan: [
          { step: "确认计划模式请求进入 App Server", status: "completed" },
          { step: "输出 proposed_plan", status: "in_progress" },
        ],
      },
    });
  });

  it("typed plan delta 应沿 canonical item identity 累积且不重复首段", () => {
    const previousItem = {
      id: "plan_turn-1_proposed_plan:1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 7,
      status: "in_progress",
      started_at: "2026-06-23T09:59:59.000Z",
      updated_at: "2026-06-23T09:59:59.000Z",
      type: "plan",
      text: "- [ ] 读协议",
      metadata: {},
    } satisfies AgentThreadItem;
    const baseEvent = {
      type: "plan_delta",
      text: "- [ ] 读协议",
      delta: "- [ ] 读协议",
      sourceItemId: previousItem.id,
      source: "app_server_v2",
      thread_id: "thread-1",
      turn_id: "turn-1",
    } satisfies AgentEventPlanDelta & AgentEventEnvelope;

    const first = buildAgentStreamPlanThreadItem({
      activeSessionId: "session-1",
      event: baseEvent,
      previousItem,
    });
    const second = buildAgentStreamPlanThreadItem({
      activeSessionId: "session-1",
      event: {
        ...baseEvent,
        text: "\n- [ ] 接 GUI",
        delta: "\n- [ ] 接 GUI",
      },
      previousItem: first ?? undefined,
    });

    expect(first).toMatchObject({
      id: previousItem.id,
      sequence: 7,
      text: "- [ ] 读协议",
    });
    expect(second).toMatchObject({
      id: previousItem.id,
      text: "- [ ] 读协议\n- [ ] 接 GUI",
    });
  });
});
