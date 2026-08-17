import type { AgentEvent } from "@/lib/api/agentProtocol";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";
import type {
  AgentUiProjectionContext,
  AgentUiProjectionEvent,
} from "@limecloud/agent-ui-contracts";
import { buildActionProjectionEvents } from "./actionProjection";
import { buildArtifactProjectionEvents } from "./artifactProjection";
import { buildConversationProjectionEvents } from "./conversationEventProjection";
import { buildTurnContextEvents } from "./contextProjection";
import { buildDiagnosticProjectionEvents } from "./diagnosticProjection";
import { sequenceProjectionEvents as sequenceEvents } from "./projectionBase";
import { buildRoutingProjectionEvents } from "./routingProjection";
import { buildRuntimeLifecycleEvents } from "./runtimeLifecycleProjection";
import { buildThreadItemProjectionEvents } from "./threadItemProjection";
import { buildToolProjectionEvents } from "./toolEventProjection";

export interface AgentUiProjectionOptions {
  soulCopy?: SoulInteractionCopy;
}

export type {
  AgentUiControl,
  AgentUiEventClass,
  AgentUiOwner,
  AgentUiPersistence,
  AgentUiPhase,
  AgentUiProjectionContext,
  AgentUiProjectionEvent,
  AgentUiProjectionRefs,
  AgentUiProjectionSourceType,
  AgentUiRuntimeEntity,
  AgentUiRuntimeStatus,
  AgentUiScope,
  AgentUiSurface,
  AgentUiTopology,
} from "@limecloud/agent-ui-contracts";

export type {
  AgentUiEvidenceProjectionInput,
  AgentUiHandoffProjectionInput,
  AgentUiRequestedFixExecutionResult,
  AgentUiRequestedFixExecutionStatus,
  AgentUiReviewProjectionInput,
} from "./evidenceProjection";

export {
  buildAgentUiEvidenceChangedEvent,
  buildAgentUiHandoffProjectionEvents,
  buildAgentUiReviewProjectionEvents,
} from "./evidenceProjection";

export type {
  AgentUiTeamControlProjectionAction,
  AgentUiTeamControlProjectionInput,
} from "./teamControlProjection";

export { buildAgentUiTeamControlProjectionEvents } from "./teamControlProjection";

export type { AgentUiMetricProjectionInput } from "./metricProjection";
export { buildAgentUiMetricChangedEvent } from "./metricProjection";

export type {
  AgentUiRemoteTeammateProjectionEvent,
  AgentUiRemoteTeammateProjectionInput,
} from "./remoteTeammateProjection";

export { buildAgentUiRemoteTeammateProjectionEvents } from "./remoteTeammateProjection";

export function buildAgentUiProjectionEvents(
  event: AgentEvent,
  context: AgentUiProjectionContext = {},
  options: AgentUiProjectionOptions = {},
): AgentUiProjectionEvent[] {
  const events: AgentUiProjectionEvent[] = (() => {
    switch (event.type) {
      case "thread_started":
        return buildRuntimeLifecycleEvents(event, context);
      case "turn_started":
        return buildRuntimeLifecycleEvents(event, context);
      case "item_started":
      case "item_updated":
      case "item_completed":
        return buildThreadItemProjectionEvents(event, context);
      case "turn_completed":
        return buildRuntimeLifecycleEvents(event, context);
      case "turn_canceled":
        return buildRuntimeLifecycleEvents(event, context);
      case "turn_failed":
      case "error":
        return buildRuntimeLifecycleEvents(event, context);
      case "message":
        return buildConversationProjectionEvents(event, context);
      case "text_delta":
      case "text_delta_batch":
        return buildConversationProjectionEvents(event, context);
      case "thinking_delta":
      case "reasoning_delta":
      case "reasoning_summary_delta":
      case "reasoning_summary_part_added":
      case "reasoning_content_delta":
      case "reasoning_final":
        return buildConversationProjectionEvents(event, context);
      case "reasoning_started":
      case "reasoning_ended":
        return buildConversationProjectionEvents(event, context);
      case "runtime_status":
        return buildRuntimeLifecycleEvents(event, context);
      case "tool_start":
        return buildToolProjectionEvents(event, context, options);
      case "tool_end":
        return buildToolProjectionEvents(event, context, options);
      case "tool_progress":
        return buildToolProjectionEvents(event, context, options);
      case "tool_output_delta":
        return buildToolProjectionEvents(event, context, options);
      case "tool_input_delta":
        return buildToolProjectionEvents(event, context, options);
      case "artifact_snapshot":
        return buildArtifactProjectionEvents(event, context);
      case "action_required":
        return buildActionProjectionEvents(event, context);
      case "action_resolved":
        return buildActionProjectionEvents(event, context);
      case "context_trace":
        return buildArtifactProjectionEvents(event, context);
      case "turn_context":
        return buildTurnContextEvents(event, context);
      case "model_change":
        return buildRuntimeLifecycleEvents(event, context);
      case "model_effective":
        return buildRuntimeLifecycleEvents(event, context);
      case "task_profile_resolved":
        return buildRuntimeLifecycleEvents(event, context);
      case "warning":
        return buildDiagnosticProjectionEvents(event, context);
      case "cost_estimated":
      case "cost_recorded":
        return buildDiagnosticProjectionEvents(event, context);
      case "candidate_set_resolved":
      case "routing_decision_made":
      case "routing_fallback_applied":
      case "routing_not_possible":
      case "limit_state_updated":
      case "single_candidate_only":
      case "single_candidate_capability_gap":
      case "rate_limit_hit":
      case "quota_low":
      case "quota_blocked":
        return buildRoutingProjectionEvents(event, context);
      default:
        return [];
    }
  })();

  return sequenceEvents(events, context.sequence);
}
