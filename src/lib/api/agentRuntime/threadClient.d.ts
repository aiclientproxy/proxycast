import {
  AppServerClient,
  type AppServerJsonRpcNotification,
  type AppServerThreadReadResponse,
  type AppServerReviewStartParams,
  type AppServerReviewStartResponse,
  type AppServerThreadSettingsUpdateParams,
  type AppServerThreadSettingsUpdateResponse,
  type AppServerThreadShellCommandParams,
} from "@/lib/api/appServer";
import type {
  AppServerRequestResult,
  ThreadResumeParams,
  ThreadResumeResponse,
  ThreadSetNameParams,
  TurnStartParams,
  TurnSteerParams,
  TurnSteerResponse,
} from "@limecloud/app-server-client";
import type { AgentRuntimeClient as StandardAgentRuntimeClient } from "@limecloud/agent-runtime-client";
import type { AgentRuntimeCapabilityManifest } from "@limecloud/agent-ui-contracts";
import { type AgentRuntimeCommandInvoke } from "./transport";
import type {
  AgentRuntimeCapabilityManifestRequest,
  AgentRuntimeCompactSessionRequest,
  AgentRuntimeDiffFileCheckpointRequest,
  AgentRuntimeGetFileCheckpointRequest,
  AgentRuntimeInterruptTurnRequest,
  AgentRuntimeListFileCheckpointsRequest,
  AgentRuntimeReplayRequestRequest,
  AgentRuntimeReplayedActionRequiredView,
  AgentRuntimeRespondActionRequest,
  AgentRuntimeRestoreFileCheckpointRequest,
} from "./requestTypes";
import type {
  AgentRuntimeFileCheckpointDetail,
  AgentRuntimeFileCheckpointDiffResult,
  AgentRuntimeFileCheckpointListResult,
  AgentRuntimeFileCheckpointRestoreResult,
  AgentRuntimeThreadReadModel,
} from "./sessionTypes";

export type AgentRuntimeAppServerClient = Pick<
  AppServerClient,
  | "readThread"
  | "setThreadName"
  | "updateThreadSettings"
  | "runThreadShellCommand"
  | "startTurn"
  | "startReview"
  | "steerTurn"
  | "cancelTurn"
  | "startThreadCompaction"
  | "resumeThread"
  | "drainEvents"
  | "listAgentSessionFileCheckpoints"
  | "getAgentSessionFileCheckpoint"
  | "diffAgentSessionFileCheckpoint"
  | "restoreAgentSessionFileCheckpoint"
  | "listCapabilities"
>;
export type AgentRuntimeLifecycleClient = Pick<
  StandardAgentRuntimeClient,
  "startTurn" | "steerTurn" | "cancelTurn" | "readThread"
>;
export interface AgentRuntimeThreadClientDeps {
  invokeCommand?: AgentRuntimeCommandInvoke;
  appServerClient?: AgentRuntimeAppServerClient;
  standardRuntimeClient?: AgentRuntimeLifecycleClient;
  isAppServerTurnLifecycleAvailable?: () => boolean;
  enableAppServerEventDrain?: boolean;
}
export declare function createThreadClient(
  deps?: AgentRuntimeThreadClientDeps,
): {
  compactAgentRuntimeSession: (
    request: AgentRuntimeCompactSessionRequest,
  ) => Promise<void>;
  diffAgentRuntimeFileCheckpoint: (
    request: AgentRuntimeDiffFileCheckpointRequest,
  ) => Promise<AgentRuntimeFileCheckpointDiffResult>;
  getAgentRuntimeFileCheckpoint: (
    request: AgentRuntimeGetFileCheckpointRequest,
  ) => Promise<AgentRuntimeFileCheckpointDetail>;
  getAgentRuntimeCapabilityManifest: (
    request?: AgentRuntimeCapabilityManifestRequest,
  ) => Promise<AgentRuntimeCapabilityManifest>;
  getAgentRuntimeThreadRead: (
    threadId: string,
  ) => Promise<AgentRuntimeThreadReadModel>;
  readAgentRuntimeThread: (
    threadId: string,
  ) => Promise<AppServerThreadReadResponse>;
  readThreadSessionId: (threadId: string) => Promise<string>;
  setAgentRuntimeThreadName: (request: ThreadSetNameParams) => Promise<void>;
  updateAgentRuntimeThreadSettings: (
    request: AppServerThreadSettingsUpdateParams,
  ) => Promise<AppServerRequestResult<AppServerThreadSettingsUpdateResponse>>;
  interruptAgentRuntimeTurn: (
    request: AgentRuntimeInterruptTurnRequest,
  ) => Promise<boolean>;
  listAgentRuntimeFileCheckpoints: (
    request: AgentRuntimeListFileCheckpointsRequest,
  ) => Promise<AgentRuntimeFileCheckpointListResult>;
  replayAgentRuntimeRequest: (
    request: AgentRuntimeReplayRequestRequest,
  ) => Promise<AgentRuntimeReplayedActionRequiredView | null>;
  respondAgentRuntimeAction: (
    request: AgentRuntimeRespondActionRequest,
  ) => Promise<void>;
  restoreAgentRuntimeFileCheckpoint: (
    request: AgentRuntimeRestoreFileCheckpointRequest,
  ) => Promise<AgentRuntimeFileCheckpointRestoreResult>;
  resumeThread: (
    request: ThreadResumeParams,
  ) => Promise<AppServerRequestResult<ThreadResumeResponse>>;
  runUserShellCommand: (
    request: AppServerThreadShellCommandParams,
    eventName: string,
  ) => Promise<void>;
  steerAgentRuntimeTurn: (
    request: TurnSteerParams,
  ) => Promise<AppServerRequestResult<TurnSteerResponse>>;
  submitAgentRuntimeReview: (
    request: AppServerReviewStartParams,
  ) => Promise<AppServerRequestResult<AppServerReviewStartResponse>>;
  submitAgentRuntimeTurn: (request: TurnStartParams) => Promise<void>;
};
export declare function publishAppServerAgentSessionNotifications(
  eventName: string | undefined,
  notifications: AppServerJsonRpcNotification[] | undefined,
): void;
export declare function projectAppServerAgentEventPayload(
  notification: AppServerJsonRpcNotification,
): Record<string, unknown> | null;
export declare const compactAgentRuntimeSession: ReturnType<
    typeof createThreadClient
  >["compactAgentRuntimeSession"],
  diffAgentRuntimeFileCheckpoint: ReturnType<
    typeof createThreadClient
  >["diffAgentRuntimeFileCheckpoint"],
  getAgentRuntimeCapabilityManifest: ReturnType<
    typeof createThreadClient
  >["getAgentRuntimeCapabilityManifest"],
  getAgentRuntimeFileCheckpoint: ReturnType<
    typeof createThreadClient
  >["getAgentRuntimeFileCheckpoint"],
  getAgentRuntimeThreadRead: ReturnType<
    typeof createThreadClient
  >["getAgentRuntimeThreadRead"],
  readAgentRuntimeThread: ReturnType<
    typeof createThreadClient
  >["readAgentRuntimeThread"],
  readThreadSessionId: ReturnType<
    typeof createThreadClient
  >["readThreadSessionId"],
  setAgentRuntimeThreadName: ReturnType<
    typeof createThreadClient
  >["setAgentRuntimeThreadName"],
  updateAgentRuntimeThreadSettings: ReturnType<
    typeof createThreadClient
  >["updateAgentRuntimeThreadSettings"],
  interruptAgentRuntimeTurn: ReturnType<
    typeof createThreadClient
  >["interruptAgentRuntimeTurn"],
  listAgentRuntimeFileCheckpoints: ReturnType<
    typeof createThreadClient
  >["listAgentRuntimeFileCheckpoints"],
  replayAgentRuntimeRequest: ReturnType<
    typeof createThreadClient
  >["replayAgentRuntimeRequest"],
  respondAgentRuntimeAction: ReturnType<
    typeof createThreadClient
  >["respondAgentRuntimeAction"],
  restoreAgentRuntimeFileCheckpoint: ReturnType<
    typeof createThreadClient
  >["restoreAgentRuntimeFileCheckpoint"],
  resumeThread: ReturnType<typeof createThreadClient>["resumeThread"],
  runUserShellCommand: ReturnType<
    typeof createThreadClient
  >["runUserShellCommand"],
  steerAgentRuntimeTurn: ReturnType<
    typeof createThreadClient
  >["steerAgentRuntimeTurn"],
  submitAgentRuntimeReview: ReturnType<
    typeof createThreadClient
  >["submitAgentRuntimeReview"],
  submitAgentRuntimeTurn: ReturnType<
    typeof createThreadClient
  >["submitAgentRuntimeTurn"];
