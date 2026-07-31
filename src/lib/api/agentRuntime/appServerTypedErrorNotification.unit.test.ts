import { beforeEach, describe, expect, it } from "vitest";
import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import { parseAgentEvent } from "@/lib/api/agentProtocol";
import { projectAppServerAgentEventPayload } from "./appServerEventStream";
import {
  projectAgentRuntimeSequenceGateNotifications,
  resetAgentRuntimeEventSequenceGatesForTests,
} from "./eventSequenceGate";

const threadId = "thread-typed-error";
const turnId = "turn-typed-error";

function errorNotification(
  params: Record<string, unknown>,
): AppServerJsonRpcNotification {
  return { method: "error", params };
}

describe("App Server typed error notification", () => {
  beforeEach(() => {
    resetAgentRuntimeEventSequenceGatesForTests();
  });

  it.each([true, false])(
    "projects willRetry=%s without changing terminal semantics",
    (willRetry) => {
      const notification = errorNotification({
        error: {
          additionalDetails: "provider request id: request-1",
          codexErrorInfo: {
            responseStreamDisconnected: { httpStatusCode: null },
          },
          message: "provider stream disconnected",
        },
        threadId,
        turnId,
        willRetry,
      });

      expect(
        projectAgentRuntimeSequenceGateNotifications(
          `agent_stream_typed_error_${willRetry}`,
          notification,
        ),
      ).toEqual([notification]);

      const payload = projectAppServerAgentEventPayload(notification);
      expect(payload).toMatchObject({
        additional_details: "provider request id: request-1",
        codex_error_info: {
          responseStreamDisconnected: { httpStatusCode: null },
        },
        message: "provider stream disconnected",
        protocol_method: "error",
        session_id: threadId,
        thread_id: threadId,
        turn_id: turnId,
        type: "error",
        will_retry: willRetry,
      });
      expect(parseAgentEvent(payload)).toMatchObject({
        additional_details: "provider request id: request-1",
        codex_error_info: {
          responseStreamDisconnected: { httpStatusCode: null },
        },
        message: "provider stream disconnected",
        protocol_method: "error",
        type: "error",
        will_retry: willRetry,
      });
    },
  );

  it("fails closed for malformed known direct errors instead of recording drift", () => {
    const malformed = errorNotification({
      error: { message: "provider failed" },
      threadId,
      turnId,
      willRetry: "false",
    });

    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "agent_stream_typed_error_malformed",
        malformed,
      ),
    ).toEqual([]);
    expect(projectAppServerAgentEventPayload(malformed)).toBeNull();
  });
});
