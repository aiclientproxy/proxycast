import {
  AppServerClient,
  type AppServerAgentSessionFileCheckpointDetail,
  type AppServerAgentSessionFileCheckpointDiffResponse,
  type AppServerAgentSessionFileCheckpointListResponse,
  type AppServerAgentSessionFileCheckpointRestoreResponse,
  type AppServerAgentSessionFileCheckpointSummary,
  type AppServerCapabilityListParams,
  type AppServerThreadReadParams,
  type AppServerThreadReadResponse,
  type AppServerThreadShellCommandParams,
} from "@/lib/api/appServer";
import type {
  AppServerRequestResult,
  ThreadResumeParams,
  ThreadResumeResponse,
  TurnInterruptParams,
  TurnStartParams,
  TurnSteerParams,
  TurnSteerResponse,
} from "@limecloud/app-server-client";
import { isAppServerBridgeAvailable } from "@/lib/api/appServerBridgeAvailability";
import type { AgentRuntimeClient as StandardAgentRuntimeClient } from "@limecloud/agent-runtime-client";
import {
  AppServerAgentSessionEventDrainRouter,
  publishAppServerAgentSessionNotifications,
  publishAppServerRpcErrorNotifications,
} from "./appServerEventStream";
import { projectAppServerThreadReadResult } from "./appServerReadModelClient";
import {
  invokeAgentRuntimeCommand,
  type AgentRuntimeCommandInvoke,
} from "./transport";
import { agentRuntimeCapabilityManifestFromAppServerResponse } from "./capabilityContract";
import { AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY } from "../agentProtocolOps";
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
  AgentRuntimeFileCheckpointSummary,
  AgentRuntimeThreadReadModel,
} from "./sessionTypes";
import type { AgentRuntimeCapabilityManifest } from "@limecloud/agent-ui-contracts";
import {
  findPendingTypedServerRequestAction,
  respondPendingTypedServerRequest,
  replayedActionViewFromPendingAction,
} from "./serverRequestReplay";

export type AgentRuntimeAppServerClient = Pick<
  AppServerClient,
  | "readThread"
  | "runThreadShellCommand"
  | "startTurn"
  | "steerTurn"
  | "cancelTurn"
  | "compactAgentSession"
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

type AgentRuntimeLifecycleStartTurnParams = Parameters<
  AgentRuntimeLifecycleClient["startTurn"]
>[0];
type AgentRuntimeLifecycleCancelTurnParams = Parameters<
  AgentRuntimeLifecycleClient["cancelTurn"]
>[0];
type AgentRuntimeLifecycleSteerTurnParams = Parameters<
  AgentRuntimeLifecycleClient["steerTurn"]
>[0];
type AgentRuntimeLifecycleReadThreadParams = Parameters<
  AgentRuntimeLifecycleClient["readThread"]
>[0];

export interface AgentRuntimeThreadClientDeps {
  invokeCommand?: AgentRuntimeCommandInvoke;
  appServerClient?: AgentRuntimeAppServerClient;
  standardRuntimeClient?: AgentRuntimeLifecycleClient;
  isAppServerTurnLifecycleAvailable?: () => boolean;
  enableAppServerEventDrain?: boolean;
}

export function createThreadClient(deps: AgentRuntimeThreadClientDeps = {}) {
  const {
    invokeCommand = invokeAgentRuntimeCommand,
    appServerClient = new AppServerClient(),
    isAppServerTurnLifecycleAvailable = defaultIsAppServerTurnLifecycleAvailable,
    enableAppServerEventDrain,
  } = deps;
  void invokeCommand;
  const standardRuntimeClient =
    deps.standardRuntimeClient ??
    createAppServerAgentRuntimeLifecycleClient(appServerClient);
  const appServerEventRouter = shouldEnableAppServerEventDrain(
    appServerClient,
    enableAppServerEventDrain,
    deps.standardRuntimeClient,
  )
    ? new AppServerAgentSessionEventDrainRouter(appServerClient)
    : null;

  async function submitAgentRuntimeTurn(
    request: TurnStartParams,
  ): Promise<void> {
    assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable);
    const threadId = request.threadId.trim();
    if (!threadId) {
      throw new Error("threadId is required to start App Server turn");
    }
    const { eventName, params } = prepareAppServerTurnStart(request, threadId);
    const route = appServerEventRouter?.register({
      eventName,
      sessionId: threadId,
    });
    try {
      const result = await standardRuntimeClient.startTurn(params);
      if (route) {
        route.publish(result.notifications);
      } else {
        publishAppServerAgentSessionNotifications(
          eventName,
          result.notifications,
        );
      }
    } catch (error) {
      publishAppServerRpcErrorNotifications(error, {
        eventName,
        sessionId: threadId,
      });
      throw error;
    }
  }

  async function interruptAgentRuntimeTurn(
    request: AgentRuntimeInterruptTurnRequest,
  ): Promise<boolean> {
    assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable);
    const result = await standardRuntimeClient.cancelTurn(
      appServerTurnCancelParamsFromRequest(request),
    );
    publishAppServerAgentSessionNotifications(
      request.event_name,
      result.notifications,
    );
    return true;
  }

  function steerAgentRuntimeTurn(
    request: TurnSteerParams,
  ): Promise<AppServerRequestResult<TurnSteerResponse>> {
    assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable);
    return standardRuntimeClient.steerTurn(request);
  }

  async function compactAgentRuntimeSession(
    request: AgentRuntimeCompactSessionRequest,
  ): Promise<void> {
    const result = await appServerClient.compactAgentSession({
      sessionId: request.session_id,
      eventName: request.event_name,
    });
    publishAppServerAgentSessionNotifications(
      request.event_name,
      result.notifications,
    );
  }

  async function resumeThread(
    request: ThreadResumeParams,
  ): Promise<AppServerRequestResult<ThreadResumeResponse>> {
    const threadId = request.threadId.trim();
    if (!threadId) {
      throw new Error("thread/resume requires a non-empty threadId");
    }
    const params: ThreadResumeParams = {
      ...request,
      threadId,
      excludeTurns: request.excludeTurns ?? true,
    };
    return appServerClient.resumeThread(params);
  }

  async function getAgentRuntimeCapabilityManifest(
    request: AgentRuntimeCapabilityManifestRequest = {},
  ): Promise<AgentRuntimeCapabilityManifest> {
    const params: AppServerCapabilityListParams = {
      ...(request.app_id ? { appId: request.app_id } : {}),
      ...(request.workspace_id ? { workspaceId: request.workspace_id } : {}),
      ...(request.session_id ? { sessionId: request.session_id } : {}),
      ...(request.cursor ? { cursor: request.cursor } : {}),
      ...(typeof request.limit === "number" ? { limit: request.limit } : {}),
    };
    const result = await appServerClient.listCapabilities(params);
    return agentRuntimeCapabilityManifestFromAppServerResponse(
      result.result.capabilities ?? [],
      result.result.runtimeCapabilityManifest,
      {
        sessionId: request.session_id,
      },
    );
  }

  async function replayAgentRuntimeRequest(
    request: AgentRuntimeReplayRequestRequest,
  ): Promise<AgentRuntimeReplayedActionRequiredView | null> {
    const action = findPendingTypedServerRequestAction(
      request.session_id,
      request.request_id,
    );
    return action ? replayedActionViewFromPendingAction(action) : null;
  }

  async function respondAgentRuntimeAction(
    request: AgentRuntimeRespondActionRequest,
  ): Promise<void> {
    assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable);
    if (respondPendingTypedServerRequest(request)) {
      return;
    }
    throw new Error(
      "Typed server request is no longer pending; generic agentSession/action/respond is retired.",
    );
  }

  async function getAgentRuntimeThreadRead(
    threadId: string,
  ): Promise<AgentRuntimeThreadReadModel> {
    assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable);
    const normalizedThreadId = threadId.trim();
    if (!normalizedThreadId) {
      throw new Error(
        "threadId is required to read canonical App Server thread",
      );
    }
    const response = await appServerClient.readThread({
      threadId: normalizedThreadId,
      includeTurns: true,
    });
    return projectAppServerThreadReadResult(response.result);
  }

  async function runUserShellCommand(
    request: AppServerThreadShellCommandParams,
    eventName: string,
  ): Promise<void> {
    const threadId = request.threadId.trim();
    const command = request.command.trim();
    const normalizedEventName = eventName.trim();
    if (!threadId) {
      throw new Error("threadId is required to run a shell command");
    }
    if (!command) {
      throw new Error("command is required to run a shell command");
    }
    if (!normalizedEventName) {
      throw new Error("eventName is required to route a shell command");
    }

    const route = appServerEventRouter?.register({
      eventName: normalizedEventName,
      sessionId: threadId,
    });
    try {
      const result = await appServerClient.runThreadShellCommand({
        threadId,
        command,
      });
      if (route) {
        route.publish(result.notifications);
      } else {
        publishAppServerAgentSessionNotifications(
          normalizedEventName,
          result.notifications,
        );
      }
    } catch (error) {
      publishAppServerRpcErrorNotifications(error, {
        eventName: normalizedEventName,
        sessionId: threadId,
      });
      throw error;
    }
  }

  async function readAgentRuntimeThread(
    threadId: string,
  ): Promise<AppServerThreadReadResponse> {
    assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable);
    const normalizedThreadId = threadId.trim();
    if (!normalizedThreadId) {
      throw new Error(
        "threadId is required to read canonical App Server thread",
      );
    }
    const response = await appServerClient.readThread({
      threadId: normalizedThreadId,
      includeTurns: true,
    } satisfies AppServerThreadReadParams);
    return response.result;
  }

  async function readThreadSessionId(threadId: string): Promise<string> {
    assertAppServerTurnLifecycleAvailable(isAppServerTurnLifecycleAvailable);
    const normalizedThreadId = threadId.trim();
    if (!normalizedThreadId) {
      throw new Error("threadId is required to resolve App Server session");
    }
    const response = await appServerClient.readThread({
      threadId: normalizedThreadId,
      includeTurns: false,
    } satisfies AppServerThreadReadParams);
    const resolvedThreadId = response.result.thread.id.trim();
    if (resolvedThreadId !== normalizedThreadId) {
      throw new Error(
        `thread/read returned mismatched threadId: expected ${normalizedThreadId}, received ${resolvedThreadId || "<empty>"}`,
      );
    }
    const sessionId = response.result.thread.sessionId.trim();
    if (!sessionId) {
      throw new Error(
        `thread/read returned an empty sessionId for ${normalizedThreadId}`,
      );
    }
    return sessionId;
  }

  async function listAgentRuntimeFileCheckpoints(
    request: AgentRuntimeListFileCheckpointsRequest,
  ): Promise<AgentRuntimeFileCheckpointListResult> {
    const command = "agentSession/fileCheckpoint/list";
    const result = await appServerClient.listAgentSessionFileCheckpoints({
      sessionId: request.session_id,
    });
    return projectAppServerFileCheckpointListResult(command, result.result);
  }

  async function getAgentRuntimeFileCheckpoint(
    request: AgentRuntimeGetFileCheckpointRequest,
  ): Promise<AgentRuntimeFileCheckpointDetail> {
    const command = "agentSession/fileCheckpoint/get";
    const result = await appServerClient.getAgentSessionFileCheckpoint({
      sessionId: request.session_id,
      checkpointId: request.checkpoint_id,
    });
    return projectAppServerFileCheckpointDetail(command, result.result);
  }

  async function diffAgentRuntimeFileCheckpoint(
    request: AgentRuntimeDiffFileCheckpointRequest,
  ): Promise<AgentRuntimeFileCheckpointDiffResult> {
    const command = "agentSession/fileCheckpoint/diff";
    const result = await appServerClient.diffAgentSessionFileCheckpoint({
      sessionId: request.session_id,
      checkpointId: request.checkpoint_id,
    });
    return projectAppServerFileCheckpointDiffResult(command, result.result);
  }

  async function restoreAgentRuntimeFileCheckpoint(
    request: AgentRuntimeRestoreFileCheckpointRequest,
  ): Promise<AgentRuntimeFileCheckpointRestoreResult> {
    const command = "agentSession/fileCheckpoint/restore";
    const result = await appServerClient.restoreAgentSessionFileCheckpoint({
      sessionId: request.session_id,
      checkpointId: request.checkpoint_id,
      confirmRestore: request.confirm_restore,
      createBackup: request.create_backup,
    });
    return projectAppServerFileCheckpointRestoreResult(command, result.result);
  }

  return {
    compactAgentRuntimeSession,
    diffAgentRuntimeFileCheckpoint,
    getAgentRuntimeFileCheckpoint,
    getAgentRuntimeCapabilityManifest,
    getAgentRuntimeThreadRead,
    readAgentRuntimeThread,
    readThreadSessionId,
    interruptAgentRuntimeTurn,
    listAgentRuntimeFileCheckpoints,
    replayAgentRuntimeRequest,
    respondAgentRuntimeAction,
    restoreAgentRuntimeFileCheckpoint,
    resumeThread,
    runUserShellCommand,
    steerAgentRuntimeTurn,
    submitAgentRuntimeTurn,
  };
}

function defaultIsAppServerTurnLifecycleAvailable(): boolean {
  return isAppServerBridgeAvailable();
}

function prepareAppServerTurnStart(
  request: TurnStartParams,
  threadId: string,
): { eventName: string; params: TurnStartParams } {
  const fallbackEventName = `agentSession/event/${threadId}`;
  const context = request.additionalContext;
  if (!isRecord(context)) {
    return { eventName: fallbackEventName, params: request };
  }

  const rendererEventName =
    context[AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY];
  if (
    !isRecord(rendererEventName) ||
    rendererEventName.kind !== "application" ||
    typeof rendererEventName.value !== "string"
  ) {
    return { eventName: fallbackEventName, params: request };
  }

  const {
    [AGENT_RUNTIME_RENDERER_EVENT_NAME_CONTEXT_KEY]: _rendererEventName,
    ...additionalContext
  } = context;
  const { additionalContext: _context, ...baseParams } = request;
  return {
    eventName: rendererEventName.value.trim() || fallbackEventName,
    params:
      Object.keys(additionalContext).length > 0
        ? { ...baseParams, additionalContext }
        : baseParams,
  };
}

function assertAppServerTurnLifecycleAvailable(
  isAvailable: () => boolean,
): void {
  if (!isAvailable()) {
    throw new Error(
      "App Server turn lifecycle is unavailable; Agent Runtime requires the App Server current lifecycle channel.",
    );
  }
}

function shouldEnableAppServerEventDrain(
  appServerClient: AgentRuntimeAppServerClient,
  override: boolean | undefined,
  standardRuntimeClient: AgentRuntimeLifecycleClient | undefined,
): boolean {
  if (override !== undefined) {
    return override;
  }
  if (standardRuntimeClient) {
    return false;
  }
  return appServerClient instanceof AppServerClient;
}

function createAppServerAgentRuntimeLifecycleClient(
  appServerClient: AgentRuntimeAppServerClient,
): AgentRuntimeLifecycleClient {
  return {
    startTurn: (params: AgentRuntimeLifecycleStartTurnParams) =>
      appServerClient.startTurn(params) as ReturnType<
        AgentRuntimeLifecycleClient["startTurn"]
      >,
    steerTurn: (params: AgentRuntimeLifecycleSteerTurnParams) =>
      appServerClient.steerTurn(params) as ReturnType<
        AgentRuntimeLifecycleClient["steerTurn"]
      >,
    cancelTurn: (params: AgentRuntimeLifecycleCancelTurnParams) =>
      appServerClient.cancelTurn(params) as ReturnType<
        AgentRuntimeLifecycleClient["cancelTurn"]
      >,
    readThread: (params: AgentRuntimeLifecycleReadThreadParams) =>
      appServerClient.readThread(
        params as unknown as AppServerThreadReadParams,
      ) as ReturnType<AgentRuntimeLifecycleClient["readThread"]>,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isRequiredString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isOptionalString(value: unknown): value is string | undefined {
  return value === undefined || typeof value === "string";
}

function isOptionalFiniteNumber(value: unknown): value is number | undefined {
  return (
    value === undefined || (typeof value === "number" && Number.isFinite(value))
  );
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

function isFileCheckpointSummary(
  value: unknown,
): value is AgentRuntimeFileCheckpointListResult["checkpoints"][number] {
  return (
    isRecord(value) &&
    isRequiredString(value.checkpoint_id) &&
    isRequiredString(value.turn_id) &&
    isRequiredString(value.path) &&
    isRequiredString(value.source) &&
    (typeof value.updated_at === "string" ||
      typeof value.updated_at === "number") &&
    isOptionalFiniteNumber(value.version_no) &&
    isOptionalString(value.version_id) &&
    isOptionalString(value.request_id) &&
    isOptionalString(value.title) &&
    isOptionalString(value.kind) &&
    isOptionalString(value.status) &&
    isOptionalString(value.preview_text) &&
    isOptionalString(value.snapshot_path) &&
    typeof value.validation_issue_count === "number" &&
    Number.isFinite(value.validation_issue_count)
  );
}

function isFileCheckpointListResult(
  value: unknown,
): value is AgentRuntimeFileCheckpointListResult {
  return (
    isRecord(value) &&
    isRequiredString(value.session_id) &&
    isRequiredString(value.thread_id) &&
    typeof value.checkpoint_count === "number" &&
    Number.isFinite(value.checkpoint_count) &&
    Array.isArray(value.checkpoints) &&
    value.checkpoints.every(isFileCheckpointSummary)
  );
}

function isFileCheckpointDetail(
  value: unknown,
): value is AgentRuntimeFileCheckpointDetail {
  return (
    isRecord(value) &&
    isRequiredString(value.session_id) &&
    isRequiredString(value.thread_id) &&
    isFileCheckpointSummary(value.checkpoint) &&
    isRequiredString(value.live_path) &&
    isRequiredString(value.snapshot_path) &&
    Array.isArray(value.version_history) &&
    isStringArray(value.validation_issues) &&
    (value.content === undefined || typeof value.content === "string")
  );
}

function isFileCheckpointDiffResult(
  value: unknown,
): value is AgentRuntimeFileCheckpointDiffResult {
  return (
    isRecord(value) &&
    isRequiredString(value.session_id) &&
    isRequiredString(value.thread_id) &&
    isFileCheckpointSummary(value.checkpoint) &&
    isOptionalString(value.current_version_id) &&
    isOptionalString(value.previous_version_id)
  );
}

function isFileCheckpointRestoreResult(
  value: unknown,
): value is AgentRuntimeFileCheckpointRestoreResult {
  return (
    isRecord(value) &&
    isRequiredString(value.session_id) &&
    isRequiredString(value.thread_id) &&
    isFileCheckpointSummary(value.checkpoint) &&
    isRequiredString(value.live_path) &&
    isRequiredString(value.snapshot_path) &&
    (value.backup_path === undefined ||
      value.backup_path === null ||
      typeof value.backup_path === "string") &&
    (typeof value.restored_at === "string" ||
      typeof value.restored_at === "number")
  );
}

function readField(
  record: Record<string, unknown>,
  camelKey: string,
  snakeKey?: string,
): unknown {
  if (Object.prototype.hasOwnProperty.call(record, camelKey)) {
    return record[camelKey];
  }
  return snakeKey ? record[snakeKey] : undefined;
}

function isOptionalNullableString(
  value: unknown,
): value is string | null | undefined {
  return value === undefined || value === null || typeof value === "string";
}

function isAppServerFileCheckpointSummary(
  value: unknown,
): value is AppServerAgentSessionFileCheckpointSummary {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isRequiredString(readField(value, "checkpointId", "checkpoint_id")) &&
    isRequiredString(readField(value, "turnId", "turn_id")) &&
    isRequiredString(readField(value, "path")) &&
    isRequiredString(readField(value, "source")) &&
    isRequiredString(readField(value, "updatedAt", "updated_at")) &&
    isOptionalFiniteNumber(readField(value, "versionNo", "version_no")) &&
    isOptionalString(readField(value, "versionId", "version_id")) &&
    isOptionalString(readField(value, "requestId", "request_id")) &&
    isOptionalString(readField(value, "title")) &&
    isOptionalString(readField(value, "kind")) &&
    isOptionalString(readField(value, "status")) &&
    isOptionalString(readField(value, "previewText", "preview_text")) &&
    isOptionalString(readField(value, "snapshotPath", "snapshot_path")) &&
    typeof readField(
      value,
      "validationIssueCount",
      "validation_issue_count",
    ) === "number" &&
    Number.isFinite(
      readField(value, "validationIssueCount", "validation_issue_count"),
    )
  );
}

function isAppServerFileCheckpointListResponse(
  value: unknown,
): value is AppServerAgentSessionFileCheckpointListResponse {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isRequiredString(readField(value, "sessionId", "session_id")) &&
    isRequiredString(readField(value, "threadId", "thread_id")) &&
    typeof readField(value, "checkpointCount", "checkpoint_count") ===
      "number" &&
    Number.isFinite(readField(value, "checkpointCount", "checkpoint_count")) &&
    Array.isArray(readField(value, "checkpoints")) &&
    (readField(value, "checkpoints") as unknown[]).every(
      isAppServerFileCheckpointSummary,
    )
  );
}

function isAppServerFileCheckpointDetail(
  value: unknown,
): value is AppServerAgentSessionFileCheckpointDetail {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isRequiredString(readField(value, "sessionId", "session_id")) &&
    isRequiredString(readField(value, "threadId", "thread_id")) &&
    isAppServerFileCheckpointSummary(readField(value, "checkpoint")) &&
    isRequiredString(readField(value, "livePath", "live_path")) &&
    isRequiredString(readField(value, "snapshotPath", "snapshot_path")) &&
    Array.isArray(readField(value, "versionHistory", "version_history")) &&
    isStringArray(readField(value, "validationIssues", "validation_issues")) &&
    (readField(value, "content") === undefined ||
      typeof readField(value, "content") === "string")
  );
}

function isAppServerFileCheckpointDiffResponse(
  value: unknown,
): value is AppServerAgentSessionFileCheckpointDiffResponse {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isRequiredString(readField(value, "sessionId", "session_id")) &&
    isRequiredString(readField(value, "threadId", "thread_id")) &&
    isAppServerFileCheckpointSummary(readField(value, "checkpoint")) &&
    isOptionalString(
      readField(value, "currentVersionId", "current_version_id"),
    ) &&
    isOptionalString(
      readField(value, "previousVersionId", "previous_version_id"),
    )
  );
}

function isAppServerFileCheckpointRestoreResponse(
  value: unknown,
): value is AppServerAgentSessionFileCheckpointRestoreResponse {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isRequiredString(readField(value, "sessionId", "session_id")) &&
    isRequiredString(readField(value, "threadId", "thread_id")) &&
    isAppServerFileCheckpointSummary(readField(value, "checkpoint")) &&
    isRequiredString(readField(value, "livePath", "live_path")) &&
    isRequiredString(readField(value, "snapshotPath", "snapshot_path")) &&
    isOptionalNullableString(readField(value, "backupPath", "backup_path")) &&
    isRequiredString(readField(value, "restoredAt", "restored_at"))
  );
}

function assignStringIfPresent<T extends object>(
  target: Partial<T>,
  key: keyof T,
  value: unknown,
): void {
  if (typeof value === "string") {
    target[key] = value as T[keyof T];
  }
}

function assignNumberIfPresent<T extends object>(
  target: Partial<T>,
  key: keyof T,
  value: unknown,
): void {
  if (typeof value === "number" && Number.isFinite(value)) {
    target[key] = value as T[keyof T];
  }
}

function assignNullableStringIfPresent<T extends object>(
  target: Partial<T>,
  key: keyof T,
  value: unknown,
): void {
  if (value === null || typeof value === "string") {
    target[key] = value as T[keyof T];
  }
}

function assignIfPresent<T extends object>(
  target: Partial<T>,
  key: keyof T,
  value: unknown,
): void {
  if (value !== undefined) {
    target[key] = value as T[keyof T];
  }
}

function projectAppServerFileCheckpointSummary(
  command: string,
  value: unknown,
): AgentRuntimeFileCheckpointSummary {
  if (!isAppServerFileCheckpointSummary(value)) {
    throw new Error(`${command} did not return file checkpoint summary`);
  }
  const record = value as unknown as Record<string, unknown>;
  const summary: AgentRuntimeFileCheckpointSummary = {
    checkpoint_id: readField(record, "checkpointId", "checkpoint_id") as string,
    turn_id: readField(record, "turnId", "turn_id") as string,
    path: readField(record, "path") as string,
    source: readField(record, "source") as string,
    updated_at: readField(record, "updatedAt", "updated_at") as string,
    validation_issue_count: readField(
      record,
      "validationIssueCount",
      "validation_issue_count",
    ) as number,
  };
  assignNumberIfPresent(
    summary,
    "version_no",
    readField(record, "versionNo", "version_no"),
  );
  assignStringIfPresent(
    summary,
    "version_id",
    readField(record, "versionId", "version_id"),
  );
  assignStringIfPresent(
    summary,
    "request_id",
    readField(record, "requestId", "request_id"),
  );
  assignStringIfPresent(summary, "title", readField(record, "title"));
  assignStringIfPresent(summary, "kind", readField(record, "kind"));
  assignStringIfPresent(summary, "status", readField(record, "status"));
  assignStringIfPresent(
    summary,
    "preview_text",
    readField(record, "previewText", "preview_text"),
  );
  assignStringIfPresent(
    summary,
    "snapshot_path",
    readField(record, "snapshotPath", "snapshot_path"),
  );
  if (!isFileCheckpointSummary(summary)) {
    throw new Error(`${command} did not return file checkpoint summary`);
  }
  return summary;
}

function projectAppServerFileCheckpointListResult(
  command: string,
  value: unknown,
): AgentRuntimeFileCheckpointListResult {
  if (!isAppServerFileCheckpointListResponse(value)) {
    throw new Error(`${command} did not return file checkpoint list`);
  }
  const record = value as unknown as Record<string, unknown>;
  const result: AgentRuntimeFileCheckpointListResult = {
    session_id: readField(record, "sessionId", "session_id") as string,
    thread_id: readField(record, "threadId", "thread_id") as string,
    checkpoint_count: readField(
      record,
      "checkpointCount",
      "checkpoint_count",
    ) as number,
    checkpoints: (readField(record, "checkpoints") as unknown[]).map(
      (checkpoint) =>
        projectAppServerFileCheckpointSummary(command, checkpoint),
    ),
  };
  assertFileCheckpointListResult(command, result);
  return result;
}

function projectAppServerFileCheckpointDetail(
  command: string,
  value: unknown,
): AgentRuntimeFileCheckpointDetail {
  if (!isAppServerFileCheckpointDetail(value)) {
    throw new Error(`${command} did not return file checkpoint detail`);
  }
  const record = value as unknown as Record<string, unknown>;
  const detail: AgentRuntimeFileCheckpointDetail = {
    session_id: readField(record, "sessionId", "session_id") as string,
    thread_id: readField(record, "threadId", "thread_id") as string,
    checkpoint: projectAppServerFileCheckpointSummary(
      command,
      readField(record, "checkpoint"),
    ),
    live_path: readField(record, "livePath", "live_path") as string,
    snapshot_path: readField(record, "snapshotPath", "snapshot_path") as string,
    version_history: readField(
      record,
      "versionHistory",
      "version_history",
    ) as unknown[],
    validation_issues: readField(
      record,
      "validationIssues",
      "validation_issues",
    ) as string[],
  };
  assignIfPresent(
    detail,
    "checkpoint_document",
    readField(record, "checkpointDocument", "checkpoint_document"),
  );
  assignIfPresent(
    detail,
    "live_document",
    readField(record, "liveDocument", "live_document"),
  );
  assignIfPresent(detail, "metadata", readField(record, "metadata"));
  assignStringIfPresent(detail, "content", readField(record, "content"));
  assertFileCheckpointDetail(command, detail);
  return detail;
}

function projectAppServerFileCheckpointDiffResult(
  command: string,
  value: unknown,
): AgentRuntimeFileCheckpointDiffResult {
  if (!isAppServerFileCheckpointDiffResponse(value)) {
    throw new Error(`${command} did not return file checkpoint diff`);
  }
  const record = value as unknown as Record<string, unknown>;
  const result: AgentRuntimeFileCheckpointDiffResult = {
    session_id: readField(record, "sessionId", "session_id") as string,
    thread_id: readField(record, "threadId", "thread_id") as string,
    checkpoint: projectAppServerFileCheckpointSummary(
      command,
      readField(record, "checkpoint"),
    ),
  };
  assignStringIfPresent(
    result,
    "current_version_id",
    readField(record, "currentVersionId", "current_version_id"),
  );
  assignStringIfPresent(
    result,
    "previous_version_id",
    readField(record, "previousVersionId", "previous_version_id"),
  );
  assignIfPresent(result, "diff", readField(record, "diff"));
  assertFileCheckpointDiffResult(command, result);
  return result;
}

function projectAppServerFileCheckpointRestoreResult(
  command: string,
  value: unknown,
): AgentRuntimeFileCheckpointRestoreResult {
  if (!isAppServerFileCheckpointRestoreResponse(value)) {
    throw new Error(`${command} did not return file checkpoint restore result`);
  }
  const record = value as unknown as Record<string, unknown>;
  const result: AgentRuntimeFileCheckpointRestoreResult = {
    session_id: readField(record, "sessionId", "session_id") as string,
    thread_id: readField(record, "threadId", "thread_id") as string,
    checkpoint: projectAppServerFileCheckpointSummary(
      command,
      readField(record, "checkpoint"),
    ),
    live_path: readField(record, "livePath", "live_path") as string,
    snapshot_path: readField(record, "snapshotPath", "snapshot_path") as string,
    restored_at: readField(record, "restoredAt", "restored_at") as string,
  };
  assignNullableStringIfPresent(
    result,
    "backup_path",
    readField(record, "backupPath", "backup_path"),
  );
  assertFileCheckpointRestoreResult(command, result);
  return result;
}

function assertFileCheckpointListResult(
  command: string,
  value: unknown,
): asserts value is AgentRuntimeFileCheckpointListResult {
  if (!isFileCheckpointListResult(value)) {
    throw new Error(`${command} did not return file checkpoint list`);
  }
}

function assertFileCheckpointDetail(
  command: string,
  value: unknown,
): asserts value is AgentRuntimeFileCheckpointDetail {
  if (!isFileCheckpointDetail(value)) {
    throw new Error(`${command} did not return file checkpoint detail`);
  }
}

function assertFileCheckpointDiffResult(
  command: string,
  value: unknown,
): asserts value is AgentRuntimeFileCheckpointDiffResult {
  if (!isFileCheckpointDiffResult(value)) {
    throw new Error(`${command} did not return file checkpoint diff`);
  }
}

function assertFileCheckpointRestoreResult(
  command: string,
  value: unknown,
): asserts value is AgentRuntimeFileCheckpointRestoreResult {
  if (!isFileCheckpointRestoreResult(value)) {
    throw new Error(`${command} did not return file checkpoint restore result`);
  }
}

export {
  projectAppServerAgentEventPayload,
  publishAppServerAgentSessionNotifications,
} from "./appServerEventStream";

function appServerTurnCancelParamsFromRequest(
  request: AgentRuntimeInterruptTurnRequest,
): TurnInterruptParams {
  if (!request.turn_id) {
    throw new Error("turn_id is required to cancel App Server turn");
  }
  return {
    threadId: request.session_id,
    turnId: request.turn_id,
  };
}

export const {
  compactAgentRuntimeSession,
  diffAgentRuntimeFileCheckpoint,
  getAgentRuntimeCapabilityManifest,
  getAgentRuntimeFileCheckpoint,
  getAgentRuntimeThreadRead,
  readAgentRuntimeThread,
  readThreadSessionId,
  interruptAgentRuntimeTurn,
  listAgentRuntimeFileCheckpoints,
  replayAgentRuntimeRequest,
  respondAgentRuntimeAction,
  restoreAgentRuntimeFileCheckpoint,
  resumeThread,
  runUserShellCommand,
  steerAgentRuntimeTurn,
  submitAgentRuntimeTurn,
} = createThreadClient();
