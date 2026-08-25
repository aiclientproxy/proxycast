import { describe, expect, it } from "vitest";
import { parseAgentEvent } from "./agentProtocol";

describe("parseAgentEvent strict review", () => {
  it("keeps strict-review and Guardian lifecycle events on the typed stream", () => {
    expect(
      parseAgentEvent({
        type: "strict_review_required",
        thread_id: "thread-1",
        turn_id: "turn-1",
        started_at_ms: 1_783_814_400_100,
      }),
    ).toMatchObject({
      type: "strict_review_required",
      thread_id: "thread-1",
      turn_id: "turn-1",
      started_at_ms: 1_783_814_400_100,
    });

    expect(
      parseAgentEvent({
        type: "guardian_review_started",
        review_id: "review-1",
        review: { status: "inProgress" },
        action: { type: "command" },
      }),
    ).toMatchObject({
      type: "guardian_review_started",
      review_id: "review-1",
    });

    expect(
      parseAgentEvent({
        type: "guardian_review_completed",
        review_id: "review-1",
        decision_source: "agent",
        review: { status: "approved" },
        action: { type: "command" },
      }),
    ).toMatchObject({
      type: "guardian_review_completed",
      review_id: "review-1",
      decision_source: "agent",
    });
  });

  it("fails closed on malformed strict-review payloads", () => {
    expect(
      parseAgentEvent({
        type: "strict_review_required",
        started_at_ms: "now",
      }),
    ).toBeNull();
    expect(
      parseAgentEvent({
        type: "guardian_review_completed",
        review_id: "review-1",
        decision_source: "user",
        review: { status: "approved" },
        action: { type: "command" },
      }),
    ).toBeNull();
  });
});
