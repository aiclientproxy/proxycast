import { normalizeLegacyRuntimeStatusTitle } from "./agentTextNormalization";
import type {
  AgentSessionExecutionRuntimeCostState,
  AgentSessionExecutionRuntimeLimitEvent,
  AgentSessionExecutionRuntimeLimitState,
  AgentSessionExecutionRuntimeTaskProfile,
  AgentTurnOutputSchemaRuntime,
} from "./agentExecutionRuntime";
import { normalizeExecutionStrategyToReact } from "./agentRuntime/executionStrategyCompat";
import type {
  AgentContextTraceStep,
  AgentMessage,
  AgentTokenUsage,
} from "./agentProtocolCoreTypes";
import type {
  AgentEvent,
  AgentTurnContextSummary,
} from "./agentProtocolEventTypes";
import {
  normalizeRecord,
  pickStringField,
  routingDecisionFromEvent,
} from "./agentProtocolParserUtils";

export function parseAgentRuntimeEvent(
  type: string,
  event: Record<string, unknown>,
): AgentEvent | null {
  switch (type) {
    case "turn_context":
      return {
        type: "turn_context",
        session_id: (event.session_id as string) || "",
        thread_id: (event.thread_id as string) || "",
        turn_id: (event.turn_id as string) || "",
        execution_strategy: normalizeExecutionStrategyToReact(
          event.execution_strategy,
        ),
        output_schema_runtime:
          (event.output_schema_runtime as
            | AgentTurnOutputSchemaRuntime
            | null
            | undefined) || null,
        context_summary:
          (event.context_summary as
            | AgentTurnContextSummary
            | null
            | undefined) || null,
        approval_policy:
          typeof event.approval_policy === "string"
            ? event.approval_policy
            : null,
        sandbox_policy:
          typeof event.sandbox_policy === "string"
            ? event.sandbox_policy
            : null,
      };
    case "model_change":
      return {
        type: "model_change",
        model: (event.model as string) || "",
        mode: (event.mode as string) || "",
      };
    case "model_effective":
    case "model.effective": {
      const payload = normalizeRecord(event.payload);
      const source = payload ?? event;
      return {
        type: "model_effective",
        model: source.model,
        modelRef: source.modelRef ?? source.model_ref,
        provider: pickStringField(
          source,
          "provider",
          "providerId",
          "provider_id",
        ),
        modelName: pickStringField(
          source,
          "modelName",
          "model_name",
          "modelId",
          "model_id",
        ),
        source: pickStringField(source, "source"),
        serviceModelSlot: pickStringField(
          source,
          "serviceModelSlot",
          "service_model_slot",
        ),
        reasoning: source.reasoning,
        capability: source.capability,
        toolCalling: source.toolCalling ?? source.tool_calling,
        requestedReasoningEffort: pickStringField(
          source,
          "requestedReasoningEffort",
          "requested_reasoning_effort",
        ),
      };
    }
    case "context_trace":
      return {
        type: "context_trace",
        steps: Array.isArray(event.steps)
          ? (event.steps as AgentContextTraceStep[])
          : [],
      };
    case "runtime_status": {
      const status =
        event.status && typeof event.status === "object"
          ? (event.status as Record<string, unknown>)
          : null;
      const metadata =
        status?.metadata && typeof status.metadata === "object"
          ? (status.metadata as Record<string, unknown>)
          : null;
      const phase = status?.phase;
      return {
        type: "runtime_status",
        status: {
          phase:
            phase === "preparing" ||
            phase === "routing" ||
            phase === "context" ||
            phase === "permission_review" ||
            phase === "retrying" ||
            phase === "continuing" ||
            phase === "synthesizing" ||
            phase === "failed"
              ? phase
              : "routing",
          title:
            typeof status?.title === "string"
              ? normalizeLegacyRuntimeStatusTitle(status.title)
              : "",
          detail: typeof status?.detail === "string" ? status.detail : "",
          checkpoints: Array.isArray(status?.checkpoints)
            ? (status?.checkpoints as string[])
            : undefined,
          metadata: metadata
            ? {
                ...metadata,
                agentui:
                  metadata.agentui && typeof metadata.agentui === "object"
                    ? (metadata.agentui as Record<string, unknown>)
                    : undefined,
                agentUi:
                  metadata.agentUi && typeof metadata.agentUi === "object"
                    ? (metadata.agentUi as Record<string, unknown>)
                    : undefined,
                team_phase:
                  typeof metadata.team_phase === "string"
                    ? metadata.team_phase
                    : undefined,
                team_parallel_budget:
                  typeof metadata.team_parallel_budget === "number"
                    ? metadata.team_parallel_budget
                    : undefined,
                team_active_count:
                  typeof metadata.team_active_count === "number"
                    ? metadata.team_active_count
                    : undefined,
                team_queued_count:
                  typeof metadata.team_queued_count === "number"
                    ? metadata.team_queued_count
                    : undefined,
                concurrency_phase:
                  typeof metadata.concurrency_phase === "string"
                    ? metadata.concurrency_phase
                    : undefined,
                concurrency_scope:
                  typeof metadata.concurrency_scope === "string"
                    ? metadata.concurrency_scope
                    : undefined,
                concurrency_active_count:
                  typeof metadata.concurrency_active_count === "number"
                    ? metadata.concurrency_active_count
                    : undefined,
                concurrency_queued_count:
                  typeof metadata.concurrency_queued_count === "number"
                    ? metadata.concurrency_queued_count
                    : undefined,
                concurrency_budget:
                  typeof metadata.concurrency_budget === "number"
                    ? metadata.concurrency_budget
                    : undefined,
                provider_concurrency_group:
                  typeof metadata.provider_concurrency_group === "string"
                    ? metadata.provider_concurrency_group
                    : undefined,
                provider_parallel_budget:
                  typeof metadata.provider_parallel_budget === "number"
                    ? metadata.provider_parallel_budget
                    : undefined,
                queue_reason:
                  typeof metadata.queue_reason === "string"
                    ? metadata.queue_reason
                    : undefined,
                retryable_overload:
                  typeof metadata.retryable_overload === "boolean"
                    ? metadata.retryable_overload
                    : undefined,
                permission_status:
                  typeof metadata.permission_status === "string"
                    ? metadata.permission_status
                    : undefined,
                required_profile_keys: Array.isArray(
                  metadata.required_profile_keys,
                )
                  ? (metadata.required_profile_keys as string[])
                  : undefined,
                ask_profile_keys: Array.isArray(metadata.ask_profile_keys)
                  ? (metadata.ask_profile_keys as string[])
                  : undefined,
                blocking_profile_keys: Array.isArray(
                  metadata.blocking_profile_keys,
                )
                  ? (metadata.blocking_profile_keys as string[])
                  : undefined,
                decision_source:
                  typeof metadata.decision_source === "string"
                    ? metadata.decision_source
                    : undefined,
                decision_scope:
                  typeof metadata.decision_scope === "string"
                    ? metadata.decision_scope
                    : undefined,
                confirmation_status:
                  typeof metadata.confirmation_status === "string"
                    ? metadata.confirmation_status
                    : undefined,
                confirmation_request_id:
                  typeof metadata.confirmation_request_id === "string"
                    ? metadata.confirmation_request_id
                    : undefined,
                confirmation_source:
                  typeof metadata.confirmation_source === "string"
                    ? metadata.confirmation_source
                    : undefined,
                declared_only:
                  typeof metadata.declared_only === "boolean"
                    ? metadata.declared_only
                    : undefined,
                turn_gating:
                  typeof metadata.turn_gating === "boolean"
                    ? metadata.turn_gating
                    : undefined,
                limit_status:
                  typeof metadata.limit_status === "string"
                    ? metadata.limit_status
                    : undefined,
                capability_gap:
                  typeof metadata.capability_gap === "string"
                    ? metadata.capability_gap
                    : undefined,
                keepalive_kind:
                  typeof metadata.keepalive_kind === "string"
                    ? metadata.keepalive_kind
                    : undefined,
                keepalive_sequence:
                  typeof metadata.keepalive_sequence === "number"
                    ? metadata.keepalive_sequence
                    : undefined,
                keepalive_elapsed_ms:
                  typeof metadata.keepalive_elapsed_ms === "number"
                    ? metadata.keepalive_elapsed_ms
                    : undefined,
              }
            : undefined,
        },
      };
    }
    case "token_usage_updated":
      return {
        type: "token_usage_updated",
        turn_id:
          typeof event.turn_id === "string" && event.turn_id.trim()
            ? event.turn_id.trim()
            : undefined,
        usage: event.usage as AgentTokenUsage,
      };
    case "task_profile_resolved":
      return {
        type: "task_profile_resolved",
        task_profile:
          (event.task_profile as AgentSessionExecutionRuntimeTaskProfile) ||
          (event.taskProfile as AgentSessionExecutionRuntimeTaskProfile),
      };
    case "candidate_set_resolved":
      return {
        type: "candidate_set_resolved",
        routing_decision: routingDecisionFromEvent(event),
      };
    case "routing_decision_made":
      return {
        type: "routing_decision_made",
        routing_decision: routingDecisionFromEvent(event),
      };
    case "routing_fallback_applied":
      return {
        type: "routing_fallback_applied",
        routing_decision: routingDecisionFromEvent(event),
      };
    case "routing_not_possible":
      return {
        type: "routing_not_possible",
        routing_decision: routingDecisionFromEvent(event),
      };
    case "limit_state_updated":
      return {
        type: "limit_state_updated",
        limit_state:
          (event.limit_state as AgentSessionExecutionRuntimeLimitState) ||
          (event.limitState as AgentSessionExecutionRuntimeLimitState),
      };
    case "single_candidate_only":
      return {
        type: "single_candidate_only",
        limit_state:
          (event.limit_state as AgentSessionExecutionRuntimeLimitState) ||
          (event.limitState as AgentSessionExecutionRuntimeLimitState),
      };
    case "single_candidate_capability_gap":
      return {
        type: "single_candidate_capability_gap",
        limit_state:
          (event.limit_state as AgentSessionExecutionRuntimeLimitState) ||
          (event.limitState as AgentSessionExecutionRuntimeLimitState),
      };
    case "cost_estimated":
      return {
        type: "cost_estimated",
        cost_state:
          (event.cost_state as AgentSessionExecutionRuntimeCostState) ||
          (event.costState as AgentSessionExecutionRuntimeCostState),
      };
    case "cost_recorded":
      return {
        type: "cost_recorded",
        cost_state:
          (event.cost_state as AgentSessionExecutionRuntimeCostState) ||
          (event.costState as AgentSessionExecutionRuntimeCostState),
      };
    case "rate_limit_hit":
      return {
        type: "rate_limit_hit",
        limit_event:
          (event.limit_event as AgentSessionExecutionRuntimeLimitEvent) ||
          (event.limitEvent as AgentSessionExecutionRuntimeLimitEvent),
      };
    case "quota_low":
      return {
        type: "quota_low",
        limit_event:
          (event.limit_event as AgentSessionExecutionRuntimeLimitEvent) ||
          (event.limitEvent as AgentSessionExecutionRuntimeLimitEvent),
      };
    case "quota_blocked":
      return {
        type: "quota_blocked",
        limit_event:
          (event.limit_event as AgentSessionExecutionRuntimeLimitEvent) ||
          (event.limitEvent as AgentSessionExecutionRuntimeLimitEvent),
      };
    case "message":
      return {
        type: "message",
        message: event.message as AgentMessage,
      };
    case "error":
      return {
        type: "error",
        message: (event.message as string) || "Unknown error",
        will_retry:
          typeof event.will_retry === "boolean" ? event.will_retry : undefined,
        codex_error_info: event.codex_error_info,
        additional_details:
          typeof event.additional_details === "string" ||
          event.additional_details === null
            ? event.additional_details
            : undefined,
      };
    case "warning":
      return {
        type: "warning",
        code: event.code as string | undefined,
        message: (event.message as string) || "Unknown warning",
      };
    default:
      return null;
  }
}
