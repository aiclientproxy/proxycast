import {
  APP_SERVER_METHOD_AGENT_SESSION_EVENT,
  type AppServerJsonRpcNotification,
} from "@/lib/api/appServer";
import {
  normalizeRecord,
  parseEventTimestampMs,
  providerTraceStageFromEventType,
  readAppServerAgentEvent,
  readBoolean,
  readFiniteNumber,
  readString,
  readStringArray,
} from "./appServerEventPayloadUtils";
import { projectAppServerV2NotificationPayload } from "./appServerV2Notification";
import { projectAppServerNotificationDriftPayload } from "./appServerNotificationDrift";

const RAW_SIDE_CHANNEL_TYPES = new Set([
  "image_task.created",
  "image_task.parameters.required",
  "image_task_parameters_required",
  "image_task.presentation.generated",
  "media.read.chunk",
  "media.read.completed",
  "runtime.status",
]);

export function projectAppServerAgentEventPayload(
  notification: AppServerJsonRpcNotification,
): Record<string, unknown> | null {
  const direct = projectAppServerV2NotificationPayload(notification);
  if (direct) {
    return direct;
  }
  const sideChannel = projectRawSideChannel(notification);
  if (sideChannel || notification.method === APP_SERVER_METHOD_AGENT_SESSION_EVENT) {
    return sideChannel;
  }
  return projectAppServerNotificationDriftPayload(notification);
}

function projectRawSideChannel(
  notification: AppServerJsonRpcNotification,
): Record<string, unknown> | null {
  if (notification.method !== APP_SERVER_METHOD_AGENT_SESSION_EVENT) {
    return null;
  }

  const event = readAppServerAgentEvent(notification.params);
  if (!event || !isRawSideChannelType(event.type)) {
    return null;
  }

  const payload = normalizeRecord(event.payload) ?? {};
  const receivedAtMs = Date.now();
  const basePayload = {
    ...payload,
    event_id: event.eventId,
    renderer_event_received_at: receivedAtMs,
    sequence: event.sequence,
    server_event_emitted_at: parseEventTimestampMs(event.timestamp) ?? null,
    session_id: event.sessionId,
    thread_id: event.threadId,
    turn_id: event.turnId,
    timestamp: event.timestamp,
  };

  switch (event.type) {
    case "provider.request.started":
    case "provider.first_event.received":
    case "provider.first_text_delta.received":
    case "provider.failed":
    case "provider.canceled":
      return {
        ...basePayload,
        type: "provider_trace",
        runtime_event_type: event.type,
        stage:
          readString(payload, "stage") ?? providerTraceStageFromEventType(event.type),
        provider: readString(payload, "provider", "providerId", "provider_id"),
        model: readString(payload, "model", "modelName", "model_name"),
        attempt: readFiniteNumber(payload, "attempt"),
        elapsed_ms: readFiniteNumber(payload, "elapsed_ms", "elapsedMs"),
        text_chars: readFiniteNumber(payload, "text_chars", "textChars"),
        status: readString(payload, "status"),
        failure_category: readString(
          payload,
          "failure_category",
          "failureCategory",
        ),
        retryable: readBoolean(payload, "retryable"),
        non_retryable_provider_rejection: readBoolean(
          payload,
          "non_retryable_provider_rejection",
          "nonRetryableProviderRejection",
        ),
        cancel_reason: readString(payload, "cancel_reason", "cancelReason"),
        provider_request_id: readString(
          payload,
          "provider_request_id",
          "providerRequestId",
        ),
        runtime_provider_backend: readString(
          payload,
          "runtime_provider_backend",
          "runtimeProviderBackend",
        ),
        runtime_provider_selector: readString(
          payload,
          "runtime_provider_selector",
          "runtimeProviderSelector",
        ),
        runtime_provider_protocol: readString(
          payload,
          "runtime_provider_protocol",
          "runtimeProviderProtocol",
        ),
        runtime_provider_active_model: readString(
          payload,
          "runtime_provider_active_model",
          "runtimeProviderActiveModel",
        ),
        provider_request_id_header: readString(
          payload,
          "provider_request_id_header",
          "providerRequestIdHeader",
        ),
      };
    case "image_task.created": {
      const response = normalizeRecord(payload.response);
      const responseRecord = response ?? {};
      const record = normalizeRecord(response?.record);
      const responsePayload = normalizeRecord(record?.payload);
      return {
        ...basePayload,
        type: "image_task_created",
        task_id:
          readString(payload, "task_id", "taskId") ??
          readString(responseRecord, "task_id", "taskId") ??
          "",
        task_type:
          readString(payload, "task_type", "taskType") ??
          readString(responseRecord, "task_type", "taskType"),
        task_family:
          readString(payload, "task_family", "taskFamily") ??
          readString(responseRecord, "task_family", "taskFamily"),
        status:
          readString(payload, "status") ?? readString(responseRecord, "status"),
        normalized_status:
          readString(payload, "normalized_status", "normalizedStatus") ??
          readString(responseRecord, "normalized_status", "normalizedStatus"),
        artifact_path:
          readString(payload, "artifact_path", "artifactPath") ??
          readString(responseRecord, "artifact_path", "artifactPath"),
        absolute_path:
          readString(payload, "absolute_path", "absolutePath") ??
          readString(responseRecord, "absolute_path", "absolutePath"),
        ...(response ? { response } : {}),
        ...(responsePayload ? { payload: responsePayload } : {}),
      };
    }
    case "image_task.parameters.required":
    case "image_task_parameters_required": {
      const missing =
        readStringArray(
          payload,
          "missing",
          "missingParameters",
          "missing_parameters",
        ) ?? [];
      const prompt =
        readString(payload, "prompt", "message", "reason") ??
        "图片生成还需要补充必要信息。";
      return {
        ...basePayload,
        type: "runtime_status",
        status: {
          phase: "routing",
          title: "图片生成需要补充信息",
          detail: missing.length > 0 ? `缺少: ${missing.join(", ")}` : prompt,
          checkpoints: missing,
          metadata: {
            source: readString(payload, "source") ?? "image_command_workflow",
            agentui: {
              workflow_key: "image_command_workflow",
              status_kind: "image_task_parameters_required",
              missing,
              missing_parameters: missing,
              image_task: normalizeRecord(payload.image_task),
            },
          },
        },
      };
    }
    case "runtime.status":
      return {
        ...basePayload,
        type: "runtime_status",
        status: normalizeRecord(payload.status) ?? payload,
      };
    default:
      return {
        ...basePayload,
        type: event.type.split(".").join("_"),
      };
  }
}

function isRawSideChannelType(type: string): boolean {
  return (
    RAW_SIDE_CHANNEL_TYPES.has(type) || providerTraceStageFromEventType(type) !== undefined
  );
}
