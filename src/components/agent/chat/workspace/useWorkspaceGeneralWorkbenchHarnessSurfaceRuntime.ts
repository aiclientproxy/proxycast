import { useCallback, useMemo, type ComponentProps } from "react";
import { startReview } from "@/lib/api/review";
import { GeneralWorkbenchHarnessSurfaceSection } from "./WorkspaceHarnessDialogs";
import type { useWorkspaceContextHarnessRuntime } from "./useWorkspaceContextHarnessRuntime";
import type { useWorkspaceHarnessInventoryRuntime } from "./useWorkspaceHarnessInventoryRuntime";

export type WorkspaceGeneralWorkbenchHarnessPanelBaseProps = Omit<
  ComponentProps<typeof GeneralWorkbenchHarnessSurfaceSection>,
  "enabled" | "harnessState"
>;

type ContextHarnessRuntime = Pick<
  ReturnType<typeof useWorkspaceContextHarnessRuntime>,
  "harnessEnvironment"
>;
type HarnessInventoryRuntime = Pick<
  ReturnType<typeof useWorkspaceHarnessInventoryRuntime>,
  | "toolInventory"
  | "toolInventoryLoading"
  | "toolInventoryError"
  | "refreshToolInventory"
>;
type ReplayPendingRequestHandler = NonNullable<
  WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onReplayPendingRequest"]
>;

interface UseWorkspaceGeneralWorkbenchHarnessSurfaceRuntimeParams {
  activeExecutionRuntime?: {
    provider_selector?: string | null;
    model_name?: string | null;
  } | null;
  activeTheme?: string | null;
  canInterrupt: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["canInterrupt"];
  canonicalChildren: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["canonicalChildren"];
  contextHarnessRuntime: ContextHarnessRuntime;
  currentTurnId: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["currentTurnId"];
  executionStrategy?: string | null;
  harnessInventoryRuntime: HarnessInventoryRuntime;
  latestAssistantMessageId?: string | null;
  messages: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["messages"];
  model?: string | null;
  onInterruptCurrentTurn: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onInterruptCurrentTurn"];
  onLoadFilePreview: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onLoadFilePreview"];
  onManageProviders: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onManageProviders"];
  onOpenExecutionPolicySettings: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onOpenExecutionPolicySettings"];
  onOpenFile: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onOpenFile"];
  onOpenSubagentSession: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onOpenSubagentSession"];
  onRespondToAction: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onRespondToAction"];
  onSubmitCodeFixPrompt: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onSubmitCodeFixPrompt"];
  onStartReview?: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["onStartReview"];
  pendingActions: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["pendingActions"];
  projectId?: string | null;
  providerType?: string | null;
  replayPendingAction?: (
    requestId: string,
    latestAssistantMessageId: string,
  ) => ReturnType<ReplayPendingRequestHandler>;
  sessionId?: string | null;
  submittedActionsInFlight: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["submittedActionsInFlight"];
  threadItems: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["threadItems"];
  threadGoal: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["threadGoal"];
  threadGoalError: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["threadGoalError"];
  threadGoalLoading: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["threadGoalLoading"];
  threadRead: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["threadRead"];
  turns: WorkspaceGeneralWorkbenchHarnessPanelBaseProps["turns"];
  workingDir?: string | null;
  refreshSessionReadModel?: (targetSessionId?: string) => Promise<unknown>;
}

export function useWorkspaceGeneralWorkbenchHarnessSurfaceRuntime({
  activeExecutionRuntime,
  activeTheme,
  canInterrupt,
  canonicalChildren,
  contextHarnessRuntime,
  currentTurnId,
  executionStrategy,
  harnessInventoryRuntime,
  latestAssistantMessageId,
  messages,
  model,
  onInterruptCurrentTurn,
  onLoadFilePreview,
  onManageProviders,
  onOpenExecutionPolicySettings,
  onOpenFile,
  onOpenSubagentSession,
  onRespondToAction,
  onSubmitCodeFixPrompt,
  onStartReview: injectedOnStartReview,
  pendingActions,
  projectId,
  providerType,
  replayPendingAction,
  sessionId,
  submittedActionsInFlight,
  threadItems,
  threadGoal,
  threadGoalError,
  threadGoalLoading,
  threadRead,
  turns,
  workingDir,
  refreshSessionReadModel,
}: UseWorkspaceGeneralWorkbenchHarnessSurfaceRuntimeParams): WorkspaceGeneralWorkbenchHarnessPanelBaseProps {
  const reviewThreadId = threadRead?.thread_id?.trim() || "";
  const onStartReview = useCallback(async () => {
    if (injectedOnStartReview) {
      return injectedOnStartReview();
    }
    if (!reviewThreadId) {
      throw new Error("Review requires an active thread");
    }
    const result = await startReview(
      {
        threadId: reviewThreadId,
        delivery: "inline",
        target: { type: "uncommittedChanges" },
      },
      {
        onTerminal: async () => {
          await refreshSessionReadModel?.(sessionId || undefined);
        },
      },
    );
    await refreshSessionReadModel?.(sessionId || undefined);
    return result;
  }, [
    injectedOnStartReview,
    refreshSessionReadModel,
    reviewThreadId,
    sessionId,
  ]);

  return useMemo(
    () => ({
      environment: contextHarnessRuntime.harnessEnvironment,
      canonicalChildren,
      threadRead,
      threadGoal,
      threadGoalError,
      threadGoalLoading,
      turns,
      threadItems,
      currentTurnId,
      pendingActions,
      submittedActionsInFlight,
      onRespondToAction,
      canInterrupt,
      onInterruptCurrentTurn,
      onReplayPendingRequest:
        latestAssistantMessageId && replayPendingAction
          ? (requestId: string) =>
              replayPendingAction(requestId, latestAssistantMessageId)
          : undefined,
      onManageProviders,
      onOpenExecutionPolicySettings,
      messages,
      diagnosticRuntimeContext: {
        sessionId: sessionId || null,
        workspaceId: projectId,
        workingDir: workingDir || null,
        providerType:
          activeExecutionRuntime?.provider_selector || providerType || null,
        model: activeExecutionRuntime?.model_name || model || null,
        executionStrategy: executionStrategy || null,
        activeTheme: activeTheme || null,
      },
      toolInventory: harnessInventoryRuntime.toolInventory,
      toolInventoryLoading: harnessInventoryRuntime.toolInventoryLoading,
      toolInventoryError: harnessInventoryRuntime.toolInventoryError,
      onRefreshToolInventory: harnessInventoryRuntime.refreshToolInventory,
      onOpenSubagentSession,
      onLoadFilePreview,
      onOpenFile,
      onSubmitCodeFixPrompt,
      onStartReview,
    }),
    [
      activeExecutionRuntime?.model_name,
      activeExecutionRuntime?.provider_selector,
      activeTheme,
      canInterrupt,
      canonicalChildren,
      contextHarnessRuntime.harnessEnvironment,
      currentTurnId,
      executionStrategy,
      harnessInventoryRuntime.refreshToolInventory,
      harnessInventoryRuntime.toolInventory,
      harnessInventoryRuntime.toolInventoryError,
      harnessInventoryRuntime.toolInventoryLoading,
      latestAssistantMessageId,
      messages,
      model,
      onInterruptCurrentTurn,
      onLoadFilePreview,
      onManageProviders,
      onOpenExecutionPolicySettings,
      onOpenFile,
      onOpenSubagentSession,
      onRespondToAction,
      onSubmitCodeFixPrompt,
      onStartReview,
      pendingActions,
      projectId,
      providerType,
      replayPendingAction,
      sessionId,
      submittedActionsInFlight,
      threadItems,
      threadGoal,
      threadGoalError,
      threadGoalLoading,
      threadRead,
      turns,
      workingDir,
    ],
  );
}
