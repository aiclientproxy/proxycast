import { describe, expect, it } from "vitest";
import { parseAgentEvent } from "./agentProtocolEventParser";

describe("agentProtocolEventParser turn plan update", () => {
  it("解析严格的 checklist signal", () => {
    expect(
      parseAgentEvent({
        type: "turn_plan_updated",
        explanation: "继续执行",
        plan: [
          { step: "读现状", status: "completed" },
          { step: "补主链", status: "in_progress" },
        ],
      }),
    ).toMatchObject({
      type: "turn_plan_updated",
      explanation: "继续执行",
      plan: [
        { step: "读现状", status: "completed" },
        { step: "补主链", status: "in_progress" },
      ],
    });
  });

  it.each([
    { plan: [{ step: "补主链", status: "running" }] },
    { plan: [{ step: "补主链", status: "pending", extra: true }] },
    { explanation: 1, plan: [] },
  ])("拒绝非法 checklist signal: %o", (payload) => {
    expect(
      parseAgentEvent({ type: "turn_plan_updated", ...payload }),
    ).toBeNull();
  });
});
