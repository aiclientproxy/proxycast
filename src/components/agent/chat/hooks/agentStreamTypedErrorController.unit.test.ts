import { describe, expect, it } from "vitest";
import type { AgentEvent } from "@/lib/api/agentProtocol";
import { buildAgentStreamTypedErrorPlan } from "./agentStreamTypedErrorController";

function typedError(
  willRetry: boolean,
): Extract<AgentEvent, { type: "error" }> {
  return {
    message: willRetry ? "stream reconnecting" : "retry budget exhausted",
    protocol_method: "error",
    type: "error",
    will_retry: willRetry,
  };
}

describe("agentStreamTypedErrorController", () => {
  it("keeps a retrying error non-terminal", () => {
    expect(
      buildAgentStreamTypedErrorPlan({
        event: typedError(true),
        executionStrategy: "react",
      }),
    ).toMatchObject({
      kind: "retrying",
      status: {
        detail: "stream reconnecting",
        phase: "retrying",
      },
    });
  });

  it("waits for authoritative turn completion after a terminal error", () => {
    expect(
      buildAgentStreamTypedErrorPlan({
        event: typedError(false),
        executionStrategy: "react",
      }),
    ).toMatchObject({
      kind: "awaiting_terminal",
      status: {
        detail: "retry budget exhausted",
        phase: "failed",
      },
    });
  });

  it("does not take ownership of legacy synthetic errors", () => {
    expect(
      buildAgentStreamTypedErrorPlan({
        event: {
          message: "legacy error",
          type: "error",
        },
        executionStrategy: "react",
      }),
    ).toBeNull();
  });
});
