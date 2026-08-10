import { isCurrentSessionHydrationRequest } from "./sessionHydrationController";
import {
  executeSessionMetadataSync,
  type SessionMetadataSyncPlan,
  type SessionMetadataSyncRuntime,
} from "./sessionMetadataSyncController";

export interface SessionMetadataSyncScheduleEvent {
  logContext: Record<string, unknown>;
  logEvent: "switchTopic.metadataSyncSkipped";
  logOptions?: {
    throttleMs?: number;
  };
}

export type SessionMetadataSyncScheduleResult =
  | "scheduled"
  | "skipped_no_invoke"
  | "skipped_no_patch";

export interface SessionMetadataSyncSchedulerOptions {
  idleTimeoutMs: number;
  minimumDelayMs: number;
}

export interface SessionMetadataSyncScheduler {
  schedule(
    task: () => void,
    options: SessionMetadataSyncSchedulerOptions,
  ): () => void;
}

export function buildSessionMetadataSyncBrowserSkipEvent(params: {
  sessionId: string;
  workspaceId?: string | null;
}): SessionMetadataSyncScheduleEvent {
  return {
    logEvent: "switchTopic.metadataSyncSkipped",
    logContext: {
      reason: "browser_bridge_low_priority_backfill",
      topicId: params.sessionId,
      workspaceId: params.workspaceId,
    },
    logOptions: { throttleMs: 1000 },
  };
}

export function buildSessionMetadataSyncStaleSkipEvent(params: {
  currentSessionId?: string | null;
  sessionId: string;
  switchRequestVersion: number;
  workspaceId?: string | null;
}): SessionMetadataSyncScheduleEvent {
  return {
    logEvent: "switchTopic.metadataSyncSkipped",
    logContext: {
      currentSessionId: params.currentSessionId ?? null,
      switchRequestVersion: params.switchRequestVersion,
      topicId: params.sessionId,
      workspaceId: params.workspaceId,
    },
    logOptions: { throttleMs: 1000 },
  };
}

export function scheduleSessionMetadataSync(params: {
  getCurrentRequestVersion: () => number;
  getCurrentSessionId: () => string | null | undefined;
  hasRuntimeInvokeCapability: boolean;
  idleTimeoutMs: number;
  minimumDelayMs: number;
  onError: (error: unknown) => void;
  onSkipped: (event: SessionMetadataSyncScheduleEvent) => void;
  onSynced: (plan: SessionMetadataSyncPlan) => void;
  pendingCancel?: (() => void) | null;
  plan: SessionMetadataSyncPlan;
  runtime: SessionMetadataSyncRuntime;
  scheduler: SessionMetadataSyncScheduler;
  sessionId: string;
  setPendingCancel: (cancel: (() => void) | null) => void;
  switchRequestVersion: number;
  workspaceId?: string | null;
}): SessionMetadataSyncScheduleResult {
  if (!params.plan.hasPatch) {
    return "skipped_no_patch";
  }

  if (!params.hasRuntimeInvokeCapability) {
    params.onSkipped(
      buildSessionMetadataSyncBrowserSkipEvent({
        sessionId: params.sessionId,
        workspaceId: params.workspaceId,
      }),
    );
    return "skipped_no_invoke";
  }

  params.pendingCancel?.();

  let cancelled = false;
  const scheduledCancel = params.scheduler.schedule(
    () => {
      if (cancelled) {
        return;
      }
      params.setPendingCancel(null);

      const currentSessionId = params.getCurrentSessionId() ?? null;
      if (
        !isCurrentSessionHydrationRequest({
          currentRequestVersion: params.getCurrentRequestVersion(),
          requestVersion: params.switchRequestVersion,
          currentSessionId,
          targetSessionId: params.sessionId,
        })
      ) {
        params.onSkipped(
          buildSessionMetadataSyncStaleSkipEvent({
            currentSessionId,
            sessionId: params.sessionId,
            switchRequestVersion: params.switchRequestVersion,
            workspaceId: params.workspaceId,
          }),
        );
        return;
      }

      void executeSessionMetadataSync({
        fallbackExecutionStrategy: params.plan.fallbackExecutionStrategy,
        fallbackProviderPreference: params.plan.fallbackProviderPreference,
        patch: params.plan.patch,
        runtime: params.runtime,
        sessionId: params.sessionId,
      })
        .then(() => {
          params.onSynced(params.plan);
        })
        .catch(params.onError);
    },
    {
      minimumDelayMs: params.minimumDelayMs,
      idleTimeoutMs: params.idleTimeoutMs,
    },
  );
  const cancel = () => {
    cancelled = true;
    scheduledCancel();
  };
  params.setPendingCancel(cancel);
  return "scheduled";
}
