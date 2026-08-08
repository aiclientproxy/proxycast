import { toast } from "sonner";
import type {
  AgentUiProjectionSummary,
  AgentUiProjectionTranslation,
} from "../projection/agentUiProjectionSummary";
import type { buildThreadReliabilityView } from "../utils/threadReliabilityView";
import type { TranslationFunction } from "./HarnessActivityTypes";
import type { HarnessStatusPanelProps } from "./HarnessStatusPanelTypes";
import type { AgentTranslation } from "./HarnessStatusPanelPrimitives";
import type { HarnessStatusPanelSectionsProps } from "./HarnessStatusPanelSections";
import {
  THREAD_GOAL_COPY,
  THREAD_GOAL_STATUS_LABEL_KEYS,
} from "./threadGoalCopy";
import type { useHarnessActivityModel } from "./useHarnessActivityModel";
import type { useHarnessFileReviewState } from "./useHarnessFileReviewState";
import type { useHarnessEvidencePackExport } from "./useHarnessEvidencePackExport";
import type { useHarnessPreviewDialog } from "./useHarnessPreviewDialog";
import type { useHarnessToolInventoryModel } from "./useHarnessToolInventoryModel";
import type {
  ChildSubagentSessionSummary,
  HarnessRuntimeFactSummary,
  RuntimeTaskPresentation,
} from "./harnessStatusPanelViewModel";

type ActivityModel = ReturnType<typeof useHarnessActivityModel>;
type FileReviewState = ReturnType<typeof useHarnessFileReviewState>;
type EvidencePackExport = ReturnType<typeof useHarnessEvidencePackExport>;
type PreviewModel = ReturnType<typeof useHarnessPreviewDialog>;
type ToolInventoryModel = ReturnType<typeof useHarnessToolInventoryModel>;
type ThreadReliabilityView = ReturnType<typeof buildThreadReliabilityView>;

interface BuildHarnessStatusPanelSectionModelsInput {
  activityModel: ActivityModel;
  agentUiProjectionSummary: AgentUiProjectionSummary;
  canInterrupt: boolean;
  canonicalChildren: NonNullable<HarnessStatusPanelProps["canonicalChildren"]>;
  currentSessionId: string | null;
  currentTurnId: HarnessStatusPanelProps["currentTurnId"];
  diagnosticRuntimeContext: HarnessStatusPanelProps["diagnosticRuntimeContext"];
  environment: HarnessStatusPanelProps["environment"];
  fileReviewState: FileReviewState;
  handleOpenExternalLink: (url: string) => void | Promise<void>;
  evidencePackExport: EvidencePackExport;
  hasAgentUiProjectionSection: boolean;
  hasHandoffSection: boolean;
  harnessState: HarnessStatusPanelProps["harnessState"];
  messages: NonNullable<HarnessStatusPanelProps["messages"]>;
  onInterruptCurrentTurn: HarnessStatusPanelProps["onInterruptCurrentTurn"];
  onManageProviders: HarnessStatusPanelProps["onManageProviders"];
  onOpenExecutionPolicySettings: HarnessStatusPanelProps["onOpenExecutionPolicySettings"];
  onOpenFileCheckpoints: HarnessStatusPanelProps["onOpenFileCheckpoints"];
  onOpenSubagentSession: HarnessStatusPanelProps["onOpenSubagentSession"];
  onRefreshToolInventory: HarnessStatusPanelProps["onRefreshToolInventory"];
  onReplayPendingRequest: HarnessStatusPanelProps["onReplayPendingRequest"];
  onRespondToAction: HarnessStatusPanelProps["onRespondToAction"];
  pendingActions: NonNullable<HarnessStatusPanelProps["pendingActions"]>;
  previewModel: PreviewModel;
  realTeamSummary: ChildSubagentSessionSummary;
  registerSectionRef: HarnessStatusPanelSectionsProps["registerSectionRef"];
  runtimeFactSummary: HarnessRuntimeFactSummary | null;
  runtimeTaskPresentation: RuntimeTaskPresentation | null;
  submittedActionsInFlight: NonNullable<
    HarnessStatusPanelProps["submittedActionsInFlight"]
  >;
  t: TranslationFunction;
  threadItems: NonNullable<HarnessStatusPanelProps["threadItems"]>;
  threadGoal: HarnessStatusPanelProps["threadGoal"];
  threadGoalError: HarnessStatusPanelProps["threadGoalError"];
  threadGoalLoading: boolean;
  threadRead: HarnessStatusPanelProps["threadRead"];
  threadReliabilityView: ThreadReliabilityView | null;
  toolInventory: HarnessStatusPanelProps["toolInventory"];
  toolInventoryError: string | null;
  toolInventoryLoading: boolean;
  toolInventoryModel: ToolInventoryModel;
  translateAgent: AgentTranslation;
  translateProjection: AgentUiProjectionTranslation;
  turns: NonNullable<HarnessStatusPanelProps["turns"]>;
}

export function buildHarnessStatusPanelSectionModels({
  activityModel,
  agentUiProjectionSummary,
  canInterrupt,
  canonicalChildren,
  currentSessionId,
  currentTurnId,
  diagnosticRuntimeContext,
  environment,
  fileReviewState,
  handleOpenExternalLink,
  evidencePackExport,
  hasAgentUiProjectionSection,
  hasHandoffSection,
  harnessState,
  messages,
  onInterruptCurrentTurn,
  onManageProviders,
  onOpenExecutionPolicySettings,
  onOpenFileCheckpoints,
  onOpenSubagentSession,
  onRefreshToolInventory,
  onReplayPendingRequest,
  onRespondToAction: _onRespondToAction,
  pendingActions,
  previewModel,
  realTeamSummary,
  registerSectionRef,
  runtimeFactSummary,
  runtimeTaskPresentation,
  submittedActionsInFlight,
  t,
  threadItems,
  threadGoal,
  threadGoalError,
  threadGoalLoading,
  threadRead,
  threadReliabilityView,
  toolInventory,
  toolInventoryError,
  toolInventoryLoading,
  toolInventoryModel,
  translateAgent,
  translateProjection,
  turns,
}: BuildHarnessStatusPanelSectionModelsInput): HarnessStatusPanelSectionsProps {
  const {
    fileDisplayMode,
    fileFilter,
    fileFilterOptions,
    filteredFileEvents,
    filteredOutputSignals,
    groupedFileEvents,
    groupedOutputEntries,
    outputFilter,
    outputFilterOptions,
    setFileDisplayMode,
    setFileFilter,
    setOutputFilter,
  } = activityModel;
  const {
    allFileChangesSelected,
    fileChangeReviewEntries,
    fileChangeStatusCounts,
    selectableFileChangeKeys,
    selectedFileChangeCount,
    selectedFileChangeEntries,
    selectedFileChangeSet,
    setFileChangeDecisions,
    setSelectedFileChangeKeys,
  } = fileReviewState;
  const {
    filteredCatalogTools,
    hasToolInventorySection,
    runtimeToolAvailability,
    runtimeToolCapabilityGaps,
    runtimeToolTotal,
    runtimeToolVisibleTotal,
    setToolInventoryFilter,
    toolInventoryCatalogTools,
    toolInventoryExtensionSurfaces,
    toolInventoryExtensionTools,
    toolInventoryFilter,
    toolInventoryMcpTools,
    toolInventoryNativeTools,
    toolInventoryRuntimeTools,
    toolInventorySourceStats,
    toolInventoryWarnings,
  } = toolInventoryModel;
  const {
    evidenceExportError,
    evidenceExporting,
    evidencePack,
    handleExportEvidencePack,
  } = evidencePackExport;
  const { handleOpenPathValue, openBrowserReplayPreview, openPreview } =
    previewModel;

  return {
    registerSectionRef,
    runtimeTaskSectionProps: runtimeTaskPresentation
      ? {
          runtimeTaskPresentation,
          handleOpenExternalLink,
          registerSectionRef,
        }
      : null,
    handoffSectionProps:
      hasHandoffSection && currentSessionId
        ? {
            evidencePack,
            evidenceExporting,
            evidenceExportError,
            registerSectionRef,
            handleExportEvidencePack,
            handleOpenPathValue,
            openPreview,
            openBrowserReplayPreview,
          }
        : null,
    objectiveSection:
      threadRead?.thread_id &&
      (!threadGoal || threadGoal.threadId === threadRead.thread_id)
        ? {
            title: String(t(THREAD_GOAL_COPY.sectionTitle as never)),
            badge: threadGoal
              ? String(
                  t(THREAD_GOAL_STATUS_LABEL_KEYS[threadGoal.status] as never),
                )
              : String(t(THREAD_GOAL_COPY.badgeEmpty as never)),
            panelProps: {
              threadId: threadRead.thread_id,
              threadGoal,
              threadGoalError,
              threadGoalLoading,
              runtimeBusy:
                threadRead?.status === "running" ||
                threadRead?.status === "queued" ||
                canInterrupt,
            },
          }
        : null,
    reliabilitySection: {
      shouldRender: Boolean(threadReliabilityView?.shouldRender),
      statusLabel: threadReliabilityView?.statusLabel ?? "",
      panelProps: {
        threadRead,
        turns,
        threadItems,
        currentTurnId,
        pendingActions,
        submittedActionsInFlight,
        canInterrupt,
        onInterruptCurrentTurn,
        onReplayPendingRequest,
        onManageProviders,
        onOpenExecutionPolicySettings,
        harnessState,
        messages,
        diagnosticRuntimeContext,
      },
    },
    fileReviewSection:
      fileChangeReviewEntries.length > 0
        ? {
            title: String(t("agentChat.harness.fileReview.title" as never)),
            badge: String(
              t(
                "agentChat.harness.fileReview.badge" as never,
                {
                  pending: fileChangeStatusCounts.pending,
                  total: fileChangeReviewEntries.length,
                } as never,
              ),
            ),
            props: {
              entries: fileChangeReviewEntries,
              statusCounts: fileChangeStatusCounts,
              selectableKeys: selectableFileChangeKeys,
              selectedSet: selectedFileChangeSet,
              selectedEntries: selectedFileChangeEntries,
              selectedCount: selectedFileChangeCount,
              allSelected: allFileChangesSelected,
              setSelectedKeys: setSelectedFileChangeKeys,
              setDecisions: setFileChangeDecisions,
              onOpenFileCheckpoints,
              onOpenPreview: openPreview,
              onOpenPath: handleOpenPathValue,
              onOpenUrl: handleOpenExternalLink,
              onRejectedWithoutCheckpoint: () =>
                toast.info(
                  translateAgent(
                    "agentChat.harness.fileReview.toast.rejectedNoCheckpoint",
                  ),
                ),
              onAppliedSelection: (count) =>
                toast.success(
                  translateAgent("agentChat.harness.fileReview.toast.applied", {
                    count,
                  }),
                ),
              translate: translateAgent,
            },
          }
        : null,
    agentUiProjectionSectionProps: hasAgentUiProjectionSection
      ? {
          agentUiProjectionSummary,
          translateProjection,
          registerSectionRef,
        }
      : null,
    runtimeFactsSectionProps: runtimeFactSummary
      ? {
          runtimeFactSummary,
          registerSectionRef,
        }
      : null,
    activeWritesSection:
      harnessState.activeFileWrites.length > 0
        ? {
            count: harnessState.activeFileWrites.length,
            props: {
              writes: harnessState.activeFileWrites,
              onOpenPreview: openPreview,
              onOpenPath: handleOpenPathValue,
              onOpenUrl: handleOpenExternalLink,
            },
          }
        : null,
    outputSignalsSection:
      harnessState.outputSignals.length > 0
        ? {
            totalCount: harnessState.outputSignals.length,
            filteredCount: filteredOutputSignals.length,
            props: {
              signals: harnessState.outputSignals,
              filteredSignals: filteredOutputSignals,
              groupedEntries: groupedOutputEntries,
              filter: outputFilter,
              filterOptions: outputFilterOptions,
              setFilter: setOutputFilter,
              onOpenPreview: openPreview,
              onOpenPath: handleOpenPathValue,
              onOpenUrl: handleOpenExternalLink,
              translate: translateAgent,
            },
          }
        : null,
    toolInventorySectionProps: {
      hasToolInventorySection,
      toolInventory,
      toolInventoryLoading,
      toolInventoryError,
      runtimeToolVisibleTotal,
      runtimeToolTotal,
      registerSectionRef,
      onRefreshToolInventory,
      toolInventorySourceStats,
      toolInventoryWarnings,
      runtimeToolAvailability,
      runtimeToolCapabilityGaps,
      toolInventoryRuntimeTools,
      toolInventoryCatalogTools,
      filteredCatalogTools,
      toolInventoryFilter,
      setToolInventoryFilter,
      toolInventoryNativeTools,
      toolInventoryExtensionSurfaces,
      toolInventoryExtensionTools,
      toolInventoryMcpTools,
    },
    activitySectionsProps: {
      harnessState,
      registerSectionRef,
      t,
      handleOpenExternalLink,
      handleOpenPathValue,
      fileFilterOptions,
      fileFilter,
      setFileFilter,
      fileDisplayMode,
      setFileDisplayMode,
      filteredFileEvents,
      groupedFileEvents,
      openPreview,
      realTeamSummary,
      canonicalChildren,
      onOpenSubagentSession,
      environment,
    },
  };
}
