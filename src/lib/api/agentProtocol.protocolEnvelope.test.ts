import { describe, expect, it } from "vitest";
import { parseAgentEvent } from "./agentProtocol";

describe("AgentEvent protocol envelope", () => {
  it("保留 App Server current notification 的 method 与 revision", () => {
    expect(
      parseAgentEvent({
        type: "item_started",
        protocol_method: "item/started",
        protocol_revision: "2026-07-29",
        item: {
          id: "item-1",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 1,
          status: "in_progress",
          started_at: "2026-07-29T00:00:00.000Z",
          updated_at: "2026-07-29T00:00:00.000Z",
          type: "command_execution",
          command: "printf current",
          cwd: "/workspace",
        },
      }),
    ).toMatchObject({
      type: "item_started",
      protocol_method: "item/started",
      protocol_revision: "2026-07-29",
    });
  });
});
