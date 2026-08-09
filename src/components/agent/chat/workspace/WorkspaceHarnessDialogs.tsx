import { useState, type ComponentProps } from "react";
import { useTranslation } from "react-i18next";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { AgentThreadFileCheckpointDialog } from "../components/AgentThreadFileCheckpointDialog";
import { AgentRuntimeStrip } from "../components/AgentRuntimeStrip";
import { CodeReviewSummaryPanel } from "../components/CodeReviewSummaryPanel";
import {
  CodeWorkbenchGuide,
  type CodeWorkbenchGuideTarget,
} from "../components/CodeWorkbenchGuide";
import {
  HarnessStatusPanel,
  type HarnessFileChangeReviewSummary,
} from "../components/HarnessStatusPanel";
import { countFailedHarnessOutputSignals } from "../utils/harnessOutputSignals";

type HarnessPanelBaseProps = Pick<
  ComponentProps<typeof HarnessStatusPanel>,
  | "harnessState"
  | "environment"
  | "canonicalChildren"
  | "threadRead"
  | "threadGoal"
  | "threadGoalError"
  | "threadGoalLoading"
  | "turns"
  | "threadItems"
  | "currentTurnId"
  | "pendingActions"
  | "submittedActionsInFlight"
  | "onRespondToAction"
  | "canInterrupt"
  | "onInterruptCurrentTurn"
  | "onReplayPendingRequest"
  | "onOpenMemoryWorkbench"
  | "onManageProviders"
  | "onOpenExecutionPolicySettings"
  | "messages"
  | "diagnosticRuntimeContext"
  | "toolInventory"
  | "toolInventoryLoading"
  | "toolInventoryError"
  | "onRefreshToolInventory"
  | "onOpenSubagentSession"
  | "onLoadFilePreview"
  | "onOpenFile"
> & {
  onStartReview?: () => Promise<unknown>;
};

function isReviewableCodeFileAction(
  action: HarnessPanelBaseProps["harnessState"]["recentFileEvents"][number]["action"],
): boolean {
  return action === "write" || action === "edit";
}

function resolveCodeWorkbenchGuideMetrics(
  harnessState: HarnessPanelBaseProps["harnessState"],
) {
  const fileChangePaths = new Set<string>();
  let latestFileName: string | null = null;

  for (const activeWrite of harnessState.activeFileWrites) {
    const path = activeWrite.path.trim();
    if (path) {
      fileChangePaths.add(path);
      latestFileName ||= activeWrite.displayName || path;
    }
  }

  for (const event of harnessState.recentFileEvents) {
    const path = event.path.trim();
    if (path && isReviewableCodeFileAction(event.action)) {
      fileChangePaths.add(path);
      latestFileName ||= event.displayName || path;
    }
  }

  return {
    pendingFileChangeCount: fileChangePaths.size,
    totalFileChangeCount: fileChangePaths.size,
    latestFileName,
  };
}

function openHarnessSection(target: CodeWorkbenchGuideTarget): void {
  if (typeof document === "undefined") {
    return;
  }

  const section = document.querySelector(`[data-harness-section="${target}"]`);
  section?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderCodeWorkbenchGuide({
  panelBaseProps,
  hasRuntimeWorkbenchSignals,
  hasFileCheckpoints,
}: {
  panelBaseProps: HarnessPanelBaseProps;
  hasRuntimeWorkbenchSignals: boolean;
  hasFileCheckpoints: boolean;
}) {
  if (!hasRuntimeWorkbenchSignals) {
    return null;
  }

  const metrics = resolveCodeWorkbenchGuideMetrics(panelBaseProps.harnessState);

  return (
    <CodeWorkbenchGuide
      pendingApprovalsCount={
        panelBaseProps.harnessState.pendingApprovals.length
      }
      activeWriteCount={panelBaseProps.harnessState.activeFileWrites.length}
      outputSignalCount={panelBaseProps.harnessState.outputSignals.length}
      failedOutputSignalCount={countFailedHarnessOutputSignals(
        panelBaseProps.harnessState.outputSignals,
      )}
      pendingFileChangeCount={metrics.pendingFileChangeCount}
      totalFileChangeCount={metrics.totalFileChangeCount}
      latestFileName={metrics.latestFileName}
      hasRuntimeStatus={Boolean(panelBaseProps.harnessState.runtimeStatus)}
      hasFileCheckpoints={hasFileCheckpoints}
      onOpenSection={openHarnessSection}
    />
  );
}

function hasRuntimeWorkbenchSignals({
  harnessState,
  fileCheckpointSummary,
}: {
  harnessState: HarnessPanelBaseProps["harnessState"];
  fileCheckpointSummary:
    | NonNullable<
        HarnessPanelBaseProps["threadRead"]
      >["file_checkpoint_summary"]
    | null
    | undefined;
}): boolean {
  return (
    (fileCheckpointSummary?.count ?? 0) > 0 ||
    harnessState.activeFileWrites.length > 0 ||
    harnessState.recentFileEvents.length > 0 ||
    harnessState.outputSignals.length > 0
  );
}

function renderCodeReviewSummaryPanel({
  panelBaseProps,
  hasRuntimeWorkbenchSignals,
  fileCheckpointSummary,
  fileChangeReviewSummary,
  onOpenFileCheckpoints,
  onSubmitCodeFixPrompt,
  onStartReview,
}: {
  panelBaseProps: HarnessPanelBaseProps;
  hasRuntimeWorkbenchSignals: boolean;
  fileCheckpointSummary:
    | NonNullable<
        HarnessPanelBaseProps["threadRead"]
      >["file_checkpoint_summary"]
    | null;
  fileChangeReviewSummary?: HarnessFileChangeReviewSummary | null;
  onOpenFileCheckpoints?: () => void;
  onSubmitCodeFixPrompt?: (prompt: string) => void | Promise<void>;
  onStartReview?: () => Promise<unknown>;
}) {
  if (!hasRuntimeWorkbenchSignals) {
    return null;
  }

  return (
    <CodeReviewSummaryPanel
      harnessState={panelBaseProps.harnessState}
      fileCheckpointSummary={fileCheckpointSummary}
      fileChangeReviewSummary={fileChangeReviewSummary}
      onOpenSection={openHarnessSection}
      onOpenFileCheckpoints={onOpenFileCheckpoints}
      onSubmitCodeFixPrompt={onSubmitCodeFixPrompt}
      threadItems={panelBaseProps.threadItems}
      reviewThreadId={panelBaseProps.threadRead?.thread_id}
      currentTurnId={panelBaseProps.currentTurnId}
      canInterrupt={panelBaseProps.canInterrupt}
      onStartReview={onStartReview}
    />
  );
}

interface UseGeneralWorkbenchHarnessSurfaceParams {
  panelBaseProps: HarnessPanelBaseProps;
  onSubmitCodeFixPrompt?: (prompt: string) => void | Promise<void>;
  onStartReview?: () => Promise<unknown>;
}

function useGeneralWorkbenchHarnessSurface({
  panelBaseProps,
  onSubmitCodeFixPrompt,
  onStartReview,
}: UseGeneralWorkbenchHarnessSurfaceParams) {
  const [fileCheckpointDialogOpen, setFileCheckpointDialogOpen] =
    useState(false);
  const diagnosticSessionId =
    panelBaseProps.diagnosticRuntimeContext?.sessionId?.trim() || "";
  const diagnosticWorkingDir =
    panelBaseProps.diagnosticRuntimeContext?.workingDir?.trim() || "";
  const fileCheckpointSummary =
    panelBaseProps.threadRead?.file_checkpoint_summary || null;
  const latestFileCheckpoint = fileCheckpointSummary?.latest_checkpoint || null;
  const shouldShowRuntimeWorkbench = hasRuntimeWorkbenchSignals({
    harnessState: panelBaseProps.harnessState,
    fileCheckpointSummary,
  });
  const codeWorkbenchGuide = renderCodeWorkbenchGuide({
    panelBaseProps,
    hasRuntimeWorkbenchSignals: shouldShowRuntimeWorkbench,
    hasFileCheckpoints: (fileCheckpointSummary?.count ?? 0) > 0,
  });
  const openFileCheckpoints = diagnosticSessionId
    ? () => setFileCheckpointDialogOpen(true)
    : undefined;
  const leadContent =
    codeWorkbenchGuide || shouldShowRuntimeWorkbench
      ? ({
          fileChangeReviewSummary,
        }: {
          fileChangeReviewSummary: HarnessFileChangeReviewSummary;
        }) => (
          <div className="space-y-3">
            {codeWorkbenchGuide}
            {renderCodeReviewSummaryPanel({
              panelBaseProps,
              hasRuntimeWorkbenchSignals: shouldShowRuntimeWorkbench,
              fileCheckpointSummary,
              fileChangeReviewSummary,
              onOpenFileCheckpoints: openFileCheckpoints,
              onSubmitCodeFixPrompt,
              onStartReview,
            })}
          </div>
        )
      : undefined;

  return {
    diagnosticSessionId,
    diagnosticWorkingDir,
    fileCheckpointDialogOpen,
    latestFileCheckpoint,
    leadContent,
    openFileCheckpoints,
    setFileCheckpointDialogOpen,
  };
}

interface GeneralWorkbenchHarnessSurfaceSectionProps extends HarnessPanelBaseProps {
  enabled: boolean;
  onSubmitCodeFixPrompt?: (prompt: string) => void | Promise<void>;
  onStartReview?: () => Promise<unknown>;
}

export function GeneralWorkbenchHarnessSurfaceSection({
  enabled,
  onSubmitCodeFixPrompt,
  onStartReview,
  ...panelBaseProps
}: GeneralWorkbenchHarnessSurfaceSectionProps) {
  const { t } = useTranslation("agent");
  const {
    diagnosticSessionId,
    diagnosticWorkingDir,
    fileCheckpointDialogOpen,
    latestFileCheckpoint,
    leadContent,
    openFileCheckpoints,
    setFileCheckpointDialogOpen,
  } = useGeneralWorkbenchHarnessSurface({
    panelBaseProps,
    onSubmitCodeFixPrompt,
    onStartReview,
  });

  if (!enabled) {
    return null;
  }

  return (
    <>
      <div
        className="lime-workbench-theme-scope lime-workbench-surface-scope flex h-full min-h-0 flex-col overflow-hidden bg-[color:var(--lime-surface)] text-[color:var(--lime-text)]"
        data-testid="general-workbench-harness-surface"
      >
        <HarnessStatusPanel
          {...panelBaseProps}
          layout="dialog"
          onOpenFileCheckpoints={openFileCheckpoints}
          leadContent={leadContent}
          title={t("agentChat.workspaceHarnessDialog.title")}
          description={t("agentChat.workspaceHarnessDialog.description")}
          toggleLabel={t("agentChat.workspaceHarnessDialog.toggleLabel")}
        />
      </div>

      {diagnosticSessionId ? (
        <AgentThreadFileCheckpointDialog
          open={fileCheckpointDialogOpen}
          onOpenChange={setFileCheckpointDialogOpen}
          sessionId={diagnosticSessionId}
          workingDir={diagnosticWorkingDir || null}
          defaultCheckpointId={latestFileCheckpoint?.checkpoint_id || null}
        />
      ) : null}
    </>
  );
}

interface GeneralWorkbenchDialogSectionProps extends HarnessPanelBaseProps {
  enabled: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  activeTheme: ComponentProps<typeof AgentRuntimeStrip>["activeTheme"];
  toolPreferences: ComponentProps<typeof AgentRuntimeStrip>["toolPreferences"];
  runtimeToolAvailability: ComponentProps<
    typeof AgentRuntimeStrip
  >["runtimeToolAvailability"];
  isSending: ComponentProps<typeof AgentRuntimeStrip>["isSending"];
  executionRuntime: ComponentProps<
    typeof AgentRuntimeStrip
  >["executionRuntime"];
  runtimeStatusTitle: ComponentProps<
    typeof AgentRuntimeStrip
  >["runtimeStatusTitle"];
  onSubmitCodeFixPrompt?: (prompt: string) => void | Promise<void>;
  onStartReview?: () => Promise<unknown>;
}

export function GeneralWorkbenchDialogSection({
  enabled,
  open,
  onOpenChange,
  activeTheme,
  toolPreferences,
  runtimeToolAvailability,
  isSending,
  executionRuntime,
  runtimeStatusTitle,
  onSubmitCodeFixPrompt,
  onStartReview,
  ...panelBaseProps
}: GeneralWorkbenchDialogSectionProps) {
  const { t } = useTranslation("agent");
  const [fileCheckpointDialogOpen, setFileCheckpointDialogOpen] =
    useState(false);
  const diagnosticSessionId =
    panelBaseProps.diagnosticRuntimeContext?.sessionId?.trim() || "";
  const diagnosticWorkingDir =
    panelBaseProps.diagnosticRuntimeContext?.workingDir?.trim() || "";
  const fileCheckpointSummary =
    panelBaseProps.threadRead?.file_checkpoint_summary || null;
  const latestFileCheckpoint = fileCheckpointSummary?.latest_checkpoint || null;
  const shouldShowRuntimeWorkbench = hasRuntimeWorkbenchSignals({
    harnessState: panelBaseProps.harnessState,
    fileCheckpointSummary,
  });
  const codeWorkbenchGuide = renderCodeWorkbenchGuide({
    panelBaseProps,
    hasRuntimeWorkbenchSignals: shouldShowRuntimeWorkbench,
    hasFileCheckpoints: (fileCheckpointSummary?.count ?? 0) > 0,
  });
  const openFileCheckpoints = diagnosticSessionId
    ? () => setFileCheckpointDialogOpen(true)
    : undefined;

  if (!enabled) {
    return null;
  }

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent
          maxWidth="max-w-6xl"
          className="lime-workbench-theme-scope lime-workbench-surface-scope flex h-[90vh] max-h-[90vh] flex-col overflow-hidden border border-[color:var(--lime-surface-border)] bg-[color:var(--lime-surface)] p-0 text-[color:var(--lime-text)]"
          draggable={true}
          dragHandleSelector='[data-harness-drag-handle="true"]'
        >
          <HarnessStatusPanel
            {...panelBaseProps}
            layout="dialog"
            onOpenFileCheckpoints={openFileCheckpoints}
            title={t("agentChat.workspaceHarnessDialog.title")}
            description={t("agentChat.workspaceHarnessDialog.description")}
            toggleLabel={t("agentChat.workspaceHarnessDialog.toggleLabel")}
            leadContent={({ fileChangeReviewSummary }) => (
              <div className="space-y-3">
                {codeWorkbenchGuide}
                {renderCodeReviewSummaryPanel({
                  panelBaseProps,
                  hasRuntimeWorkbenchSignals: shouldShowRuntimeWorkbench,
                  fileCheckpointSummary,
                  fileChangeReviewSummary,
                  onOpenFileCheckpoints: openFileCheckpoints,
                  onSubmitCodeFixPrompt,
                  onStartReview,
                })}
                <AgentRuntimeStrip
                  activeTheme={activeTheme}
                  toolPreferences={toolPreferences}
                  runtimeToolAvailability={runtimeToolAvailability}
                  harnessState={panelBaseProps.harnessState}
                  canonicalChildren={panelBaseProps.canonicalChildren}
                  variant="embedded"
                  isSending={isSending}
                  executionRuntime={executionRuntime}
                  runtimeStatusTitle={runtimeStatusTitle}
                  fileCheckpointSummary={fileCheckpointSummary}
                  onOpenFileCheckpoints={openFileCheckpoints}
                />
              </div>
            )}
          />
        </DialogContent>
      </Dialog>

      {diagnosticSessionId ? (
        <AgentThreadFileCheckpointDialog
          open={fileCheckpointDialogOpen}
          onOpenChange={setFileCheckpointDialogOpen}
          sessionId={diagnosticSessionId}
          workingDir={diagnosticWorkingDir || null}
          defaultCheckpointId={latestFileCheckpoint?.checkpoint_id || null}
        />
      ) : null}
    </>
  );
}
