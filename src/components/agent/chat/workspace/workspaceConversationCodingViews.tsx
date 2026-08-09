import type { TFunction } from "i18next";
import { formatNumber } from "@/i18n/format";
import type {
  AgentRuntimeFileCheckpointThreadSummary,
  AgentRuntimeThreadReadModel,
} from "@/lib/api/agentRuntime/sessionTypes";
import { terminateBackgroundTerminalForItem } from "@/lib/api/backgroundTerminals";
import { projectCodingWorkbenchViewFromEvents } from "@limecloud/agent-runtime-projection";
import { CanvasSessionOverviewPanel } from "../components/CanvasSessionOverviewPanel";
import type {
  CanvasWorkbenchSessionView,
  CanvasWorkbenchUtilityView,
} from "../components/CanvasWorkbenchLayout";
import type { CanvasWorkbenchChangeView } from "../components/canvas-workbench";
import type {
  ActionRequired,
  AgentThreadItem,
  AgentThreadTurn,
  ConfirmResponse,
} from "../types";
import { CodingWorkbenchLogPanel } from "./CodingWorkbenchLogPanel";
import { CodingWorkbenchOutputPanel } from "./CodingWorkbenchOutputPanel";
import type { CodingWorkbenchRecoveryContext } from "./codingWorkbenchRecovery";
import { buildCodingSessionOverviewActivities } from "./codingSessionOverviewProjection";
import {
  buildCanvasWorkbenchChangeViewFromCodingProjection,
  buildOutputHeaderViewModel,
  buildSessionHeaderViewModel,
  buildSessionRuntimeCountersFromCodingProjection,
  resolveSessionStatusBadge,
  type SessionRuntimeCounters,
} from "./workspaceConversationSceneViewModel";

type AgentTranslate = TFunction<"agent", undefined>;

interface WorkspaceConversationCodingViewsParams {
  t: AgentTranslate;
  locale: string;
  turns: readonly AgentThreadTurn[];
  threadItems?: readonly AgentThreadItem[];
  currentTurnId?: string | null;
  threadRead?: AgentRuntimeThreadReadModel | null;
  pendingActions: readonly ActionRequired[];
  submittedActionsInFlight: readonly ActionRequired[];
  isSending?: boolean;
  focusedTimelineItemId?: string | null;
  onOpenFile?: (path: string) => void | Promise<void>;
  onRespondToAction?: (response: ConfirmResponse) => void | Promise<void>;
  onRefreshSessionReadModel?: () => void | Promise<unknown>;
  onStartReview?: () => Promise<unknown>;
  onSubmitRecoveryPrompt?: (
    prompt: string,
    context?: CodingWorkbenchRecoveryContext,
  ) => void | Promise<boolean> | boolean;
}

interface WorkspaceConversationCodingViews {
  currentSessionTurn: AgentThreadTurn | null;
  counters: SessionRuntimeCounters;
  fileCheckpointSummary: AgentRuntimeFileCheckpointThreadSummary | null;
  sessionView: CanvasWorkbenchSessionView | null;
  outputView: CanvasWorkbenchUtilityView | null;
  logView: CanvasWorkbenchUtilityView | null;
  changeView: CanvasWorkbenchChangeView | null;
}

function createEmptySessionRuntimeCounters(): SessionRuntimeCounters {
  return {
    outputItemCount: 0,
    failedOutputItemCount: 0,
    shouldUseRuntimeWorkbench: false,
    inProgressItemCount: 0,
    generatedFileCount: 0,
    hasRuntimeFileChanges: false,
    hasRuntimeOutputs: false,
    shouldExposeSessionProgress: false,
  };
}

export function buildWorkspaceConversationCodingViews({
  t,
  locale,
  turns,
  threadItems = [],
  currentTurnId,
  threadRead,
  pendingActions,
  submittedActionsInFlight,
  isSending,
  focusedTimelineItemId,
  onOpenFile,
  onRespondToAction,
  onRefreshSessionReadModel,
  onStartReview,
  onSubmitRecoveryPrompt,
}: WorkspaceConversationCodingViewsParams): WorkspaceConversationCodingViews {
  const currentSessionTurn =
    turns.find((turn) => turn.id === currentTurnId) || turns.at(-1) || null;
  const hasCodingWorkbenchEvidence =
    Boolean(threadRead) ||
    pendingActions.length > 0 ||
    submittedActionsInFlight.length > 0;
  if (!hasCodingWorkbenchEvidence) {
    return {
      currentSessionTurn,
      counters: createEmptySessionRuntimeCounters(),
      fileCheckpointSummary: null,
      sessionView: null,
      outputView: null,
      logView: null,
      changeView: null,
    };
  }

  const currentSessionStatus = resolveSessionStatusBadge(
    isSending ? "running" : currentSessionTurn?.status,
    t,
  );
  const fileCheckpointSummary = threadRead?.file_checkpoint_summary || null;
  const codingView = projectCodingWorkbenchViewFromEvents({
    executionEvents: [],
    threadItems,
    codingReadModel: threadRead,
  });
  const formatCount = (count: number) =>
    formatNumber(count, {
      locale,
    });
  const sessionOverviewActivities = buildCodingSessionOverviewActivities(
    codingView,
    {
      failedCount: (count) =>
        t("agentChat.canvasWorkbench.coding.sessionOverview.failedCount", {
          count,
          countLabel: formatCount(count),
        }),
      filesChanged: (count) =>
        t("agentChat.canvasWorkbench.coding.sessionOverview.filesChanged", {
          count,
          countLabel: formatCount(count),
        }),
      passedCount: (count) =>
        t("agentChat.canvasWorkbench.coding.sessionOverview.passedCount", {
          count,
          countLabel: formatCount(count),
        }),
      patchCount: (count) =>
        t("agentChat.canvasWorkbench.coding.sessionOverview.patchCount", {
          count,
          countLabel: formatCount(count),
        }),
      preparingResult: t(
        "agentChat.canvasWorkbench.coding.sessionOverview.preparingResult",
      ),
    },
  );
  const counters = buildSessionRuntimeCountersFromCodingProjection({
    codingView,
    fileCheckpointSummary,
    pendingActionCount: pendingActions.length,
  });
  const labels = {
    inProgressItemCountLabel: formatNumber(counters.inProgressItemCount, {
      locale,
    }),
    generatedFileCountLabel: formatNumber(counters.generatedFileCount, {
      locale,
    }),
    pendingActionCountLabel: formatNumber(pendingActions.length, {
      locale,
    }),
  };
  const sessionHeaderView = buildSessionHeaderViewModel({
    t,
    currentSessionTurn,
    currentSessionStatus,
    counters,
    labels,
    pendingActionCount: pendingActions.length,
  });
  const sessionView: CanvasWorkbenchSessionView | null = sessionHeaderView
    ? {
        ...sessionHeaderView,
        renderPanel: () => (
          <CanvasSessionOverviewPanel
            turns={turns}
            activityItems={sessionOverviewActivities}
            currentTurnId={currentTurnId}
            pendingActions={pendingActions}
            isSending={isSending}
            focusedItemId={focusedTimelineItemId}
          />
        ),
      }
    : null;
  const outputHeaderView = buildOutputHeaderViewModel({
    t,
    counters,
  });
  const outputView: CanvasWorkbenchUtilityView | null =
    counters.shouldUseRuntimeWorkbench
      ? {
          ...outputHeaderView,
          renderPanel: () => (
            <CodingWorkbenchOutputPanel
              codingView={codingView}
              fileCheckpointSummary={fileCheckpointSummary}
              submittedActionsInFlight={submittedActionsInFlight}
              processControls={
                threadRead
                  ? {
                      onTerminateBackgroundTerminal: withReadModelRefresh(
                        (itemId) =>
                          terminateBackgroundTerminalForItem({
                            threadId: threadRead.thread_id,
                            itemId,
                          }),
                        onRefreshSessionReadModel,
                      ),
                    }
                  : undefined
              }
              onRespondToAction={onRespondToAction}
              onSubmitRecoveryPrompt={onSubmitRecoveryPrompt}
            />
          ),
        }
      : null;
  const logView: CanvasWorkbenchUtilityView | null =
    counters.shouldUseRuntimeWorkbench
      ? {
          ...outputHeaderView,
          tabLabel: t("agentChat.workspaceSession.logView.tabLabel"),
          title: t("agentChat.workspaceSession.logView.title"),
          subtitle: t("agentChat.workspaceSession.logView.subtitle"),
          renderPanel: () => (
            <CodingWorkbenchLogPanel codingView={codingView} />
          ),
        }
      : null;
  const changeView = counters.shouldUseRuntimeWorkbench
    ? buildCanvasWorkbenchChangeViewFromCodingProjection({
        codingView,
        fileCheckpointSummary,
        onOpenFile,
        threadItems,
        reviewThreadId: threadRead?.thread_id,
        currentTurnId,
        turnDiff: currentSessionTurn?.unified_diff,
        canInterrupt: Boolean(isSending),
        onStartReview,
      })
    : null;

  return {
    currentSessionTurn,
    counters,
    fileCheckpointSummary,
    sessionView,
    outputView,
    logView,
    changeView,
  };
}

function withReadModelRefresh<TArgs extends readonly unknown[]>(
  handler: (...args: TArgs) => Promise<unknown>,
  onRefreshSessionReadModel?: () => void | Promise<unknown>,
): (...args: TArgs) => Promise<unknown> {
  return async (...args) => {
    const result = await handler(...args);
    await onRefreshSessionReadModel?.();
    return result;
  };
}
