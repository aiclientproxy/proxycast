import type { AppServerJsonRpcNotification } from "@/lib/api/appServer";
import { describe, expect, it } from "vitest";
import {
  projectAppServerNotificationDriftPayload,
  readAppServerNotificationDrift,
  readAppServerNotificationDriftRoute,
} from "./appServerNotificationDrift";
import { projectAgentRuntimeSequenceGateNotifications } from "./eventSequenceGate";

function notification(
  method: string,
  params: Record<string, unknown>,
): AppServerJsonRpcNotification {
  return { method, params };
}

describe("App Server notification drift", () => {
  it("records planned notifications without retaining field values", () => {
    const source = notification("hook/started", {
      threadId: "thread-1",
      turnId: "turn-1",
      secret: "must-not-leak",
    });

    expect(readAppServerNotificationDrift(source)).toEqual({
      disposition: "known_projected",
      field_names: ["secret", "threadId", "turnId"],
      method: "hook/started",
      protocol_revision: expect.any(String),
      thread_id: "thread-1",
      turn_id: "turn-1",
    });
    expect(
      JSON.stringify(readAppServerNotificationDrift(source)),
    ).not.toContain("must-not-leak");
    expect(projectAppServerNotificationDriftPayload(source)).toBeNull();
    expect(
      projectAgentRuntimeSequenceGateNotifications(
        "notification-drift",
        source,
      ),
    ).toEqual([]);
  });

  it("keeps excluded and deprecated notifications diagnostic-only", () => {
    for (const method of [
      "rawResponse/completed",
      "process/outputDelta",
      "process/exited",
    ]) {
      const diagnostic = readAppServerNotificationDrift(
        notification(method, {
          threadId: "thread-1",
          turnId: "turn-1",
          diff: "raw unified diff must not reach the renderer",
          delta: "raw process output must not reach the renderer",
          response: { raw: "not-for-renderer" },
        }),
      );
      expect(diagnostic.disposition).toBe("known_diagnostic_only");
      expect(JSON.stringify(diagnostic)).not.toContain("must not reach");
      expect(
        projectAppServerNotificationDriftPayload(
          notification(method, {
            threadId: "thread-1",
            turnId: "turn-1",
            diff: "raw unified diff must not reach the renderer",
            delta: "raw process output must not reach the renderer",
            response: { raw: "not-for-renderer" },
          }),
        ),
      ).toBeNull();
    }
  });

  it("classifies turn diff as a projected current notification", () => {
    const source = notification("turn/diff/updated", {
      threadId: "thread-1",
      turnId: "turn-1",
      diff: "diff --git a/a.txt b/a.txt\n",
    });
    expect(readAppServerNotificationDrift(source).disposition).toBe(
      "known_projected",
    );
    expect(projectAppServerNotificationDriftPayload(source)).toBeNull();
  });

  it("classifies turn moderation metadata as projected without logging values", () => {
    const source = notification("turn/moderationMetadata", {
      threadId: "thread-1",
      turnId: "turn-1",
      metadata: { secret: "must-not-leak" },
    });
    const diagnostic = readAppServerNotificationDrift(source);
    expect(diagnostic.disposition).toBe("known_projected");
    expect(JSON.stringify(diagnostic)).not.toContain("must-not-leak");
    expect(projectAppServerNotificationDriftPayload(source)).toBeNull();
  });

  it("classifies guardianWarning as a projected current notification", () => {
    const source = notification("guardianWarning", {
      threadId: "thread-1",
      message: "Guardian circuit breaker interrupted the turn.",
    });
    expect(readAppServerNotificationDrift(source).disposition).toBe(
      "known_projected",
    );
    expect(projectAppServerNotificationDriftPayload(source)).toBeNull();
  });

  it("fails unknown notifications visibly when canonical identity is present", () => {
    const source = notification("future/itemChanged", {
      thread_id: "thread-future",
      turn_id: "turn-future",
      payload: { token: "hidden" },
    });

    expect(readAppServerNotificationDriftRoute(source)).toEqual({
      threadId: "thread-future",
      turnId: "turn-future",
    });
    expect(projectAppServerNotificationDriftPayload(source)).toMatchObject({
      code: "unknown_app_server_notification:future/itemChanged",
      field_names: ["payload", "thread_id", "turn_id"],
      protocol_method: "future/itemChanged",
      type: "warning",
    });
  });
});
