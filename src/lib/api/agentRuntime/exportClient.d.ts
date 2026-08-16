import { AppServerClient } from "@/lib/api/appServer";
import type {
  AgentRuntimeAnalysisHandoff,
  AgentRuntimeHandoffBundle,
  AgentRuntimeReplayCase,
  AgentRuntimeReviewDecisionTemplate,
  AgentRuntimeSaveReviewDecisionRequest,
} from "./evidenceTypes";
export type AgentRuntimeExportAppServerClient = Pick<
  AppServerClient,
  | "exportHandoffBundle"
  | "exportReplayCase"
  | "exportAnalysisHandoff"
  | "exportReviewDecisionTemplate"
  | "saveReviewDecision"
>;
export interface AgentRuntimeExportClientDeps {
  appServerClient?: AgentRuntimeExportAppServerClient;
}
export interface AgentRuntimeExportOptions {
  locale?: string | null;
}
export declare function createExportClient({
  appServerClient,
}?: AgentRuntimeExportClientDeps): {
  exportAgentRuntimeAnalysisHandoff: (
    sessionId: string,
    options?: AgentRuntimeExportOptions,
  ) => Promise<AgentRuntimeAnalysisHandoff>;
  exportAgentRuntimeHandoffBundle: (
    sessionId: string,
    options?: AgentRuntimeExportOptions,
  ) => Promise<AgentRuntimeHandoffBundle>;
  exportAgentRuntimeReplayCase: (
    sessionId: string,
    options?: AgentRuntimeExportOptions,
  ) => Promise<AgentRuntimeReplayCase>;
  exportAgentRuntimeReviewDecisionTemplate: (
    sessionId: string,
    options?: AgentRuntimeExportOptions,
  ) => Promise<AgentRuntimeReviewDecisionTemplate>;
  saveAgentRuntimeReviewDecision: (
    request: AgentRuntimeSaveReviewDecisionRequest,
  ) => Promise<AgentRuntimeReviewDecisionTemplate>;
};
export declare const exportAgentRuntimeAnalysisHandoff: (
    sessionId: string,
    options?: AgentRuntimeExportOptions,
  ) => Promise<AgentRuntimeAnalysisHandoff>,
  exportAgentRuntimeHandoffBundle: (
    sessionId: string,
    options?: AgentRuntimeExportOptions,
  ) => Promise<AgentRuntimeHandoffBundle>,
  exportAgentRuntimeReplayCase: (
    sessionId: string,
    options?: AgentRuntimeExportOptions,
  ) => Promise<AgentRuntimeReplayCase>,
  exportAgentRuntimeReviewDecisionTemplate: (
    sessionId: string,
    options?: AgentRuntimeExportOptions,
  ) => Promise<AgentRuntimeReviewDecisionTemplate>,
  saveAgentRuntimeReviewDecision: (
    request: AgentRuntimeSaveReviewDecisionRequest,
  ) => Promise<AgentRuntimeReviewDecisionTemplate>;
