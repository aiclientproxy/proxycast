import { describe, expect, it } from "vitest";
import {
  APP_SERVER_METHOD_AGENT_SESSION_EVENT,
  type AppServerJsonRpcNotification,
} from "@/lib/api/appServer";
import { projectAppServerAgentEventPayload } from "./appServerEventStream";

function rawNotification(
  type: string,
  payload: Record<string, unknown> = {},
): AppServerJsonRpcNotification {
  return {
    method: APP_SERVER_METHOD_AGENT_SESSION_EVENT,
    params: {
      event: {
        eventId: `event-${type}`,
        sessionId: "session-1",
        threadId: "thread-1",
        turnId: "turn-1",
        sequence: 1,
        timestamp: "2026-07-13T00:00:00.000Z",
        type,
        payload,
      },
    },
  };
}

describe("appServerEventStream", () => {
  it.each([
    "thread.started",
    "turn.started",
    "turn.completed",
    "item.started",
    "item.completed",
    "message.delta",
    "tool.started",
    "tool.result",
    "action.required",
    "action.resolved",
    "runtime.warning",
  ])("拒绝 retired agentSession/event wrapper: %s", (type) => {
    expect(projectAppServerAgentEventPayload(rawNotification(type))).toBeNull();
  });

  it("即使 wrapper 携带 canonicalEvent/typedEvent 也应 fail closed", () => {
    for (const key of ["canonicalEvent", "typedEvent"]) {
      const notification = rawNotification("turn.completed") as {
        method: string;
        params: Record<string, unknown>;
      };
      notification.params[key] = {
        method: "turn/updated",
        params: { id: "turn-1", status: "completed" },
      };

      expect(projectAppServerAgentEventPayload(notification)).toBeNull();
    }
  });

  it("保留 provider diagnostic raw side-channel", () => {
    expect(
      projectAppServerAgentEventPayload(
        rawNotification("provider.first_text_delta.received", {
          stage: "first_text_delta_received",
          provider: "openai",
          model: "gpt-5.3-codex",
          attempt: 1,
          elapsed_ms: 1_400,
          text_chars: 12,
          status: "running",
          runtime_provider_backend: "current",
          runtime_provider_selector: "codex",
          runtime_provider_protocol: "responses",
          runtime_provider_active_model: "gpt-5.3-codex",
        }),
      ),
    ).toMatchObject({
      type: "provider_trace",
      runtime_event_type: "provider.first_text_delta.received",
      stage: "first_text_delta_received",
      provider: "openai",
      model: "gpt-5.3-codex",
      elapsed_ms: 1_400,
      runtime_provider_backend: "current",
      runtime_provider_selector: "codex",
      runtime_provider_protocol: "responses",
      runtime_provider_active_model: "gpt-5.3-codex",
      event_id: "event-provider.first_text_delta.received",
      renderer_event_received_at: expect.any(Number),
      server_event_emitted_at: Date.parse("2026-07-13T00:00:00.000Z"),
    });
  });

  it("保留 runtime status raw side-channel", () => {
    expect(
      projectAppServerAgentEventPayload(
        rawNotification("runtime.status", {
          status: {
            phase: "retrying",
            title: "正在重试",
            detail: "provider retry",
          },
        }),
      ),
    ).toMatchObject({
      type: "runtime_status",
      status: { phase: "retrying", title: "正在重试" },
    });
  });

  it("保留图片任务创建与缺参 raw side-channel", () => {
    expect(
      projectAppServerAgentEventPayload(
        rawNotification("image_task.created", {
          task_id: "task-image",
          artifact_path: ".lime/tasks/image_generate/task-image.json",
        }),
      ),
    ).toMatchObject({
      type: "image_task_created",
      task_id: "task-image",
      artifact_path: ".lime/tasks/image_generate/task-image.json",
    });

    expect(
      projectAppServerAgentEventPayload(
        rawNotification("image_task.parameters.required", {
          missing: ["prompt", "size"],
        }),
      ),
    ).toMatchObject({
      type: "runtime_status",
      status: {
        phase: "routing",
        checkpoints: ["prompt", "size"],
      },
    });
  });
});
