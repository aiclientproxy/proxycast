import {
  type AgentRuntimeBridgeInvoke,
  type AgentRuntimeCommandInvoke,
  type AgentRuntimeTransportDeps,
} from "./transport";
import type { AppServerSessionRpcClient } from "./appServerSessionClient";
import type { AgentRuntimeEvidenceExportAppServerClient } from "./exportClient";
import type { AgentRuntimeWorkspaceSkillBindingsAppServerClient } from "./inventoryClient";
import type { AgentRuntimeThreadClientDeps } from "./threadClient";
export type AgentRuntimeAppServerClient =
  AgentRuntimeThreadClientDeps["appServerClient"] &
    AppServerSessionRpcClient &
    AgentRuntimeEvidenceExportAppServerClient &
    AgentRuntimeWorkspaceSkillBindingsAppServerClient;
export interface AgentRuntimeClientDeps extends AgentRuntimeTransportDeps {
  bridgeInvoke?: AgentRuntimeBridgeInvoke;
  invokeCommand?: AgentRuntimeCommandInvoke;
  appServerClient?: AgentRuntimeAppServerClient;
  standardRuntimeClient?: AgentRuntimeThreadClientDeps["standardRuntimeClient"];
  isAppServerTurnLifecycleAvailable?: AgentRuntimeThreadClientDeps["isAppServerTurnLifecycleAvailable"];
}
export declare function createAgentRuntimeClient({
  appServerClient,
  bridgeInvoke,
  invoke,
  invokeCommand,
  isAppServerTurnLifecycleAvailable,
}?: AgentRuntimeClientDeps): {
  compactAgentRuntimeSession: (
    request: import("./requestTypes").AgentRuntimeCompactSessionRequest,
  ) => Promise<void>;
  diffAgentRuntimeFileCheckpoint: (
    request: import("./requestTypes").AgentRuntimeDiffFileCheckpointRequest,
  ) => Promise<import("./sessionTypes").AgentRuntimeFileCheckpointDiffResult>;
  getAgentRuntimeFileCheckpoint: (
    request: import("./requestTypes").AgentRuntimeGetFileCheckpointRequest,
  ) => Promise<import("./sessionTypes").AgentRuntimeFileCheckpointDetail>;
  getAgentRuntimeThreadRead: (
    sessionId: string,
  ) => Promise<import("./sessionTypes").AgentRuntimeThreadReadModel>;
  interruptAgentRuntimeTurn: (
    request: import("./requestTypes").AgentRuntimeInterruptTurnRequest,
  ) => Promise<boolean>;
  listAgentRuntimeFileCheckpoints: (
    request: import("./requestTypes").AgentRuntimeListFileCheckpointsRequest,
  ) => Promise<import("./sessionTypes").AgentRuntimeFileCheckpointListResult>;
  replayAgentRuntimeRequest: (
    request: import("./requestTypes").AgentRuntimeReplayRequestRequest,
  ) => Promise<
    import("./requestTypes").AgentRuntimeReplayedActionRequiredView | null
  >;
  respondAgentRuntimeAction: (
    request: import("./requestTypes").AgentRuntimeRespondActionRequest,
  ) => Promise<void>;
  resumeThread: (
    request: import("@limecloud/app-server-client").ThreadResumeParams,
  ) => Promise<
    import("@limecloud/app-server-client").AppServerRequestResult<
      import("@limecloud/app-server-client").ThreadResumeResponse
    >
  >;
  steerAgentRuntimeTurn: (
    request: import("@limecloud/app-server-client").TurnSteerParams,
  ) => Promise<
    import("@limecloud/app-server-client").AppServerRequestResult<
      import("@limecloud/app-server-client").TurnSteerResponse
    >
  >;
  submitAgentRuntimeTurn: (
    request: import("@limecloud/app-server-client").TurnStartParams,
  ) => Promise<void>;
  createAgentRuntimeSession: (
    workspaceId?: string,
    name?: string,
    executionStrategy?: import("../agentExecutionRuntime").AgentExecutionStrategy,
    options?: import("./requestTypes").AgentRuntimeCreateSessionOptions,
  ) => Promise<string>;
  deleteAgentRuntimeSession: (sessionId: string) => Promise<void>;
  getAgentRuntimeSession: (
    sessionId: string,
    options?: import("./requestTypes").AgentRuntimeGetSessionOptions,
  ) => Promise<import("./sessionTypes").AgentSessionDetail>;
  listAgentRuntimeSessions: (
    options?: import("./sessionTypes").AgentRuntimeListSessionsOptions,
  ) => Promise<import("./sessionTypes").AgentSessionInfo[]>;
  archiveAgentRuntimeSession: (sessionId: string) => Promise<void>;
  unarchiveAgentRuntimeSession: (sessionId: string) => Promise<void>;
  updateAgentRuntimeThreadToolPreferences: (
    sessionId: string,
    preferences: import("../agentExecutionRuntime").AgentSessionExecutionRuntimePreferences,
  ) => Promise<void>;
  updateAgentRuntimeThreadSettings: (
    request: import("../appServer").AppServerThreadSettingsUpdateParams,
  ) => Promise<
    import("@limecloud/app-server-client").AppServerRequestResult<
      import("../appServer").AppServerThreadSettingsUpdateResponse
    >
  >;
  setAgentRuntimeThreadName: (
    request: import("@limecloud/app-server-client").ThreadSetNameParams,
  ) => Promise<void>;
  getAgentRuntimeToolInventory: (
    request?: import("./toolInventoryTypes").AgentRuntimeToolInventoryRequest,
  ) => Promise<import("./toolInventoryTypes").AgentRuntimeToolInventory>;
  listWorkspaceSkillBindings: (
    request: import("./toolInventoryTypes").AgentRuntimeListWorkspaceSkillBindingsRequest,
  ) => Promise<
    import("./toolInventoryTypes").AgentRuntimeWorkspaceSkillBindings
  >;
  exportAgentRuntimeAnalysisHandoff: (
    sessionId: string,
    options?: import("./exportClient").AgentRuntimeExportOptions,
  ) => Promise<import("./evidenceTypes").AgentRuntimeAnalysisHandoff>;
  exportAgentRuntimeEvidencePack: (
    sessionId: string,
  ) => Promise<import("./evidenceTypes").AgentRuntimeEvidencePack>;
  exportAgentRuntimeHandoffBundle: (
    sessionId: string,
    options?: import("./exportClient").AgentRuntimeExportOptions,
  ) => Promise<import("./evidenceTypes").AgentRuntimeHandoffBundle>;
  exportAgentRuntimeReplayCase: (
    sessionId: string,
    options?: import("./exportClient").AgentRuntimeExportOptions,
  ) => Promise<import("./evidenceTypes").AgentRuntimeReplayCase>;
  exportAgentRuntimeReviewDecisionTemplate: (
    sessionId: string,
    options?: import("./exportClient").AgentRuntimeExportOptions,
  ) => Promise<import("./evidenceTypes").AgentRuntimeReviewDecisionTemplate>;
  saveAgentRuntimeReviewDecision: (
    request: import("./evidenceTypes").AgentRuntimeSaveReviewDecisionRequest,
  ) => Promise<import("./evidenceTypes").AgentRuntimeReviewDecisionTemplate>;
  generateAgentRuntimeTitleResult: (
    request: import("./agentClient").GenerateAgentRuntimeTitleRequest,
  ) => Promise<import("./sessionTypes").AgentRuntimeGeneratedTitleResult>;
  generateAgentRuntimeTitle: (
    request: import("./agentClient").GenerateAgentRuntimeTitleRequest,
  ) => Promise<string>;
  generateAgentRuntimeSessionTitle: (
    sessionId: string,
    previewText?: string,
  ) => Promise<string>;
  getRuntimeProviderSelection: () => Promise<
    import("./sessionTypes").RuntimeProviderSelection
  >;
};
export type AgentRuntimeClient = ReturnType<typeof createAgentRuntimeClient>;
