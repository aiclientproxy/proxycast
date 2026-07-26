import type { Dispatch, SetStateAction } from "react";
import type { AgentEvent } from "@/lib/api/agentProtocol";
import { activityLogger } from "@/lib/workspace/workbenchRuntime";
import type { Message } from "../types";
import { isRetainedSkillProcessMessage } from "../utils/skillInlineProcessRetention";
import {
  buildAgentStreamRequestLogFinishPlan,
  type AgentStreamRequestLogFinishPayload,
} from "./agentStreamRequestLogController";
import type { StreamRequestState } from "./agentStreamRuntimeHandlerTypes";

export function extractVisibleTextFromAgentMessage(
  message: AgentEvent extends never
    ? never
    : {
        content?: Array<{ type?: string; text?: string }>;
      },
): string {
  const parts = Array.isArray(message.content) ? message.content : [];
  return parts
    .filter(
      (part): part is { type: "text"; text: string } =>
        part?.type === "text" && typeof part.text === "string",
    )
    .map((part) => part.text)
    .join("");
}

export function sequenceFromAgentEvent(event: AgentEvent): number | null {
  return typeof event.sequence === "number" && Number.isFinite(event.sequence)
    ? event.sequence
    : null;
}

function hasOwnRecordKeys(value: unknown): value is Record<string, unknown> {
  return Boolean(
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value as Record<string, unknown>).length > 0,
  );
}

export function resolveActionResolvedUserData(
  event: Extract<AgentEvent, { type: "action_resolved" }>,
): unknown {
  if (hasOwnRecordKeys(event.data)) {
    return event.data;
  }
  if (event.feedback?.trim()) {
    return event.feedback.trim();
  }
  if (typeof event.approved === "boolean") {
    return { approved: event.approved };
  }
  return undefined;
}

export function stringifySubmittedActionResponse(
  value: unknown,
): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "string") {
    const text = value.trim();
    return text || undefined;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function resolveVisibleTextDeltaAfterSnapshotPrefill(params: {
  deltaText: string;
  prefilledSnapshotText?: string | null;
  replayOffset?: number | null;
}): {
  nextReplayOffset: number | null;
  textDelta: string;
} {
  const snapshotText = params.prefilledSnapshotText || "";
  if (!snapshotText) {
    return { nextReplayOffset: null, textDelta: params.deltaText };
  }

  const replayOffset = Math.max(0, params.replayOffset ?? 0);
  const replayRemainder = snapshotText.slice(replayOffset);
  if (!params.deltaText) {
    return { nextReplayOffset: replayOffset, textDelta: "" };
  }

  if (replayRemainder.startsWith(params.deltaText)) {
    return {
      nextReplayOffset: replayOffset + params.deltaText.length,
      textDelta: "",
    };
  }

  if (params.deltaText.startsWith(replayRemainder)) {
    return {
      nextReplayOffset: null,
      textDelta: params.deltaText.slice(replayRemainder.length),
    };
  }

  return { nextReplayOffset: null, textDelta: params.deltaText };
}

export function bindSubmissionMessagesToRuntimeTurn(
  setMessages: Dispatch<SetStateAction<Message[]>>,
  assistantMsgId: string,
  pendingTurnKey: string,
  turnId?: string | null,
) {
  const normalizedTurnId = turnId?.trim();
  const normalizedPendingTurnKey = pendingTurnKey.trim();
  if (!normalizedTurnId) {
    return;
  }

  setMessages((prev) => {
    let changed = false;
    const next = prev.map((message) => {
      if (
        message.id !== assistantMsgId &&
        (!normalizedPendingTurnKey ||
          message.runtimeTurnId !== normalizedPendingTurnKey)
      ) {
        return message;
      }
      if (message.runtimeTurnId === normalizedTurnId) {
        return message;
      }
      changed = true;
      return {
        ...message,
        runtimeTurnId: normalizedTurnId,
      };
    });
    return changed ? next : prev;
  });
}

export function hasRetainedSkillInlineProcess(message: Message): boolean {
  return (
    isRetainedSkillProcessMessage(message) &&
    (Boolean(message.thinkingContent?.trim()) ||
      Boolean(
        message.contentParts?.some(
          (part) => part.type === "thinking" && part.text.trim().length > 0,
        ),
      ))
  );
}

export function finishRequestLog(
  requestState: StreamRequestState,
  payload: AgentStreamRequestLogFinishPayload,
) {
  const requestLogPlan = buildAgentStreamRequestLogFinishPlan({
    requestLogId: requestState.requestLogId,
    requestFinished: requestState.requestFinished,
    requestStartedAt: requestState.requestStartedAt,
    finishedAt: Date.now(),
    payload,
  });
  if (
    !requestLogPlan.shouldUpdate ||
    !requestLogPlan.logId ||
    !requestLogPlan.updatePayload
  ) {
    return;
  }

  requestState.requestFinished = requestLogPlan.nextRequestFinished;
  activityLogger.updateLog(requestLogPlan.logId, requestLogPlan.updatePayload);
}
