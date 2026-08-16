import type {
  AgentRuntimeHandoffBundle,
  AgentRuntimeAnalysisHandoff,
  AgentRuntimeReplayCase,
  AgentRuntimeReviewDecisionTemplate,
} from "./evidenceTypes";
import type { AgentRuntimeThreadReadModel } from "./sessionTypes";
export declare function normalizeAnalysisHandoff(
  value: unknown,
): AgentRuntimeAnalysisHandoff;
export declare function normalizeHandoffBundle(
  value: unknown,
): AgentRuntimeHandoffBundle;
export declare function normalizeReplayCase(
  value: unknown,
): AgentRuntimeReplayCase;
export declare function normalizeReviewDecisionTemplate(
  value: unknown,
): AgentRuntimeReviewDecisionTemplate;
export declare function normalizeThreadReadModel(
  threadRead?: AgentRuntimeThreadReadModel | null,
): AgentRuntimeThreadReadModel | null | undefined;
