import { describe, expect, it } from "vitest";
import type { McpServerEventStreamNotification } from "@limecloud/app-server-client";
import {
  reduceMcpEventStreamNotification,
  selectMcpEventStreams,
} from "./eventStreamProjection";

function notification(
  method: string,
  params: unknown,
  subscriptionId = "subscription-1",
): McpServerEventStreamNotification {
  return {
    subscriptionId,
    notification: { method, params },
  };
}

describe("MCP event stream projection", () => {
  it("projects active, event and terminated lifecycle without dropping event counts", () => {
    let state = reduceMcpEventStreamNotification(
      {},
      notification("notifications/events/active", { status: "active" }),
      10,
    );
    state = reduceMcpEventStreamNotification(
      state,
      notification("notifications/events/event", {
        name: "issue.updated",
        data: { issue: 42 },
      }),
      20,
    );
    state = reduceMcpEventStreamNotification(
      state,
      notification("notifications/events/terminated", {}),
      30,
    );

    expect(state["subscription-1"]).toEqual({
      subscriptionId: "subscription-1",
      phase: "terminated",
      lastEventMethod: "notifications/events/terminated",
      lastEventName: null,
      eventCount: 1,
      activeCount: 1,
      reconnectCount: 0,
      updatedAt: 30,
    });
  });

  it("counts a second active barrier as a reconnect and sorts newest first", () => {
    let state = reduceMcpEventStreamNotification(
      {},
      notification("notifications/events/active", {}),
      10,
    );
    state = reduceMcpEventStreamNotification(
      state,
      notification("notifications/events/active", {}),
      30,
    );
    state = reduceMcpEventStreamNotification(
      state,
      notification("notifications/events/active", {}, "subscription-2"),
      20,
    );

    expect(state["subscription-1"].reconnectCount).toBe(1);
    expect(
      selectMcpEventStreams(state).map((item) => item.subscriptionId),
    ).toEqual(["subscription-1", "subscription-2"]);
  });
});
