import type {
  AgentUiPhase,
  AgentUiProjectionEvent,
  AgentUiRuntimeEntity,
  AgentUiRuntimeStatus,
} from "@limecloud/agent-ui-contracts";

import { definedString } from "./normalization.js";

export type AgentRuntimeStatusPhase =
  | "context"
  | "failed"
  | "permission_review"
  | "preparing"
  | "routing"
  | string;

export interface AgentUiRuntimeEntityInferenceInput {
  runtimeEntity?: AgentUiRuntimeEntity | string | null;
  sourceType?: string | null;
  itemType?: string | null;
  runId?: string | null;
}

export interface AgentUiTeamRuntimeMetadata {
  team_phase?: string;
  team_parallel_budget?: number;
  team_active_count?: number;
  team_queued_count?: number;
  concurrency_phase?: string;
  concurrency_scope?: string;
  concurrency_active_count?: number;
  concurrency_queued_count?: number;
  concurrency_budget?: number;
  provider_concurrency_group?: string;
  provider_parallel_budget?: number;
  queue_reason?: string;
  retryable_overload?: boolean;
}

const AGENT_UI_RUNTIME_ENTITIES = new Set<AgentUiRuntimeEntity>([
  "agent_turn",
  "subagent_turn",
  "external_task",
  "work_item",
  "unknown",
]);

export function normalizeRuntimeEntity(
  value: AgentUiRuntimeEntity | string | null | undefined,
): AgentUiRuntimeEntity | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  return AGENT_UI_RUNTIME_ENTITIES.has(value as AgentUiRuntimeEntity)
    ? (value as AgentUiRuntimeEntity)
    : undefined;
}

export function inferAgentUiRuntimeEntity(
  input: AgentUiRuntimeEntityInferenceInput,
): AgentUiRuntimeEntity {
  const explicit = normalizeRuntimeEntity(input.runtimeEntity);
  if (explicit) {
    return explicit;
  }
  if (
    (input.sourceType === "item_started" ||
      input.sourceType === "item_updated" ||
      input.sourceType === "item_completed") &&
    input.itemType === "subagent_activity"
  ) {
    return "subagent_turn";
  }
  return "agent_turn";
}

export function normalizeRuntimeStatusFromRuntimePhase(
  phase: AgentRuntimeStatusPhase,
): AgentUiRuntimeStatus {
  switch (phase) {
    case "permission_review":
      return "waiting";
    case "failed":
      return "failed";
    case "context":
    case "preparing":
    case "routing":
    default:
      return "preparing";
  }
}

export function normalizeRuntimePhaseFromRuntimeStatusPhase(
  phase: AgentRuntimeStatusPhase,
): AgentUiPhase {
  switch (phase) {
    case "routing":
      return "routing";
    case "context":
      return "preparing";
    case "permission_review":
      return "waiting";
    case "failed":
      return "failed";
    case "preparing":
    default:
      return "preparing";
  }
}

export function buildTeamRuntimeFacts(
  metadata: AgentUiTeamRuntimeMetadata | undefined,
): Pick<
  AgentUiProjectionEvent,
  | "teamPhase"
  | "teamParallelBudget"
  | "teamActiveCount"
  | "teamQueuedCount"
  | "queuedTurnCount"
  | "providerConcurrencyGroup"
  | "providerParallelBudget"
  | "queueReason"
  | "retryableOverload"
> {
  return {
    teamPhase: definedString(
      metadata?.team_phase ?? metadata?.concurrency_phase,
    ),
    teamParallelBudget:
      metadata?.team_parallel_budget ?? metadata?.concurrency_budget,
    teamActiveCount:
      metadata?.team_active_count ?? metadata?.concurrency_active_count,
    teamQueuedCount:
      metadata?.team_queued_count ?? metadata?.concurrency_queued_count,
    queuedTurnCount:
      metadata?.team_queued_count ?? metadata?.concurrency_queued_count,
    providerConcurrencyGroup: definedString(
      metadata?.provider_concurrency_group,
    ),
    providerParallelBudget: metadata?.provider_parallel_budget,
    queueReason: definedString(metadata?.queue_reason),
    retryableOverload: metadata?.retryable_overload,
  };
}
