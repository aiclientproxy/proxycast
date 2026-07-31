import type {
  AgentEvent,
  AgentRuntimeStatusPayload,
} from "@/lib/api/agentProtocol";
import type { AgentExecutionStrategy } from "@/lib/api/agentExecutionRuntime";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";
import {
  buildFailedAgentRuntimeStatus,
  buildWaitingAgentRuntimeStatus,
} from "../utils/agentRuntimeStatus";

export interface AgentStreamTypedErrorPlan {
  kind: "retrying" | "awaiting_terminal";
  status: AgentRuntimeStatusPayload;
}

export function buildAgentStreamTypedErrorPlan(params: {
  event: Extract<AgentEvent, { type: "error" }>;
  executionStrategy: AgentExecutionStrategy;
  soulCopy?: SoulInteractionCopy;
}): AgentStreamTypedErrorPlan | null {
  if (
    params.event.protocol_method !== "error" ||
    typeof params.event.will_retry !== "boolean"
  ) {
    return null;
  }

  if (!params.event.will_retry) {
    return {
      kind: "awaiting_terminal",
      status: buildFailedAgentRuntimeStatus(
        params.event.message,
        params.soulCopy,
      ),
    };
  }

  const waitingStatus = buildWaitingAgentRuntimeStatus({
    executionStrategy: params.executionStrategy,
    soulCopy: params.soulCopy,
  });
  return {
    kind: "retrying",
    status: {
      ...waitingStatus,
      phase: "retrying",
      detail: params.event.message,
    },
  };
}
