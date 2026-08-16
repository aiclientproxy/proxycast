import { useCallback, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { openExternalUrlWithSystemBrowser } from "@/lib/api/externalUrl";
import {
  EMPTY_AGENT_UI_PROJECTION_SUMMARY,
  summarizeAgentUiProjectionEvents,
  type AgentUiProjectionTranslation,
} from "../projection/agentUiProjectionSummary";
import type { AgentUiProjectionScopeFilter } from "../projection/conversationProjectionStore";
import { useAgentUiProjectionEvents } from "../projection/useConversationProjectionStore";
import {
  buildThreadReliabilitySummary,
  buildThreadReliabilityView,
} from "../utils/threadReliabilityView";
import { HarnessPreviewDialog } from "./HarnessPreviewDialog";
import { HarnessStatusPanelSections } from "./HarnessStatusPanelSections";
import { HarnessStatusPanelShell } from "./HarnessStatusPanelShell";
import type { HarnessStatusPanelProps } from "./HarnessStatusPanelTypes";
import { type AgentTranslation } from "./HarnessStatusPanelPrimitives";
import type { HarnessSectionKey } from "./HarnessStatusSectionFrame";
import { useHarnessPreviewDialog } from "./useHarnessPreviewDialog";
import {
  buildRuntimeFactSummary,
  buildRuntimeTaskPresentation,
} from "./harnessStatusPanelViewModel";
import { summarizeCanonicalChildThreads } from "../projection/canonicalChildThreadSummary";
import {
  buildHarnessPanelSectionNavItems,
  buildHarnessSummaryCards,
} from "./harnessStatusPanelSummary";
import { buildHarnessStatusPanelSectionModels } from "./harnessStatusPanelSectionModels";
import { useHarnessFileReviewState } from "./useHarnessFileReviewState";
import { useHarnessActivityModel } from "./useHarnessActivityModel";
import { useHarnessToolInventoryModel } from "./useHarnessToolInventoryModel";

export type { HarnessFilePreviewResult } from "./useHarnessPreviewDialog";
export type { HarnessFileChangeReviewSummary } from "./useHarnessFileReviewState";

async function openExternalUrl(url: string): Promise<void> {
  await openExternalUrlWithSystemBrowser(url);
}

export function HarnessStatusPanel({
  harnessState,
  environment,
  layout = "default",
  onLoadFilePreview,
  onOpenFile,
  onRevealPath,
  onOpenPath,
  onOpenFileCheckpoints,
  canonicalChildren = [],
  onOpenSubagentSession,
  toolInventory,
  toolInventoryLoading = false,
  toolInventoryError = null,
  onRefreshToolInventory,
  title = "处理工作台",
  description = "集中查看最新进展、文件变更、处理结果和待确认事项。",
  toggleLabel = "详情",
  leadContent,
  threadRead = null,
  threadGoal = null,
  threadGoalError = null,
  threadGoalLoading = false,
  turns = [],
  threadItems = [],
  currentTurnId = null,
  pendingActions = [],
  submittedActionsInFlight = [],
  onRespondToAction,
  canInterrupt = false,
  onInterruptCurrentTurn,
  onReplayPendingRequest,
  onManageProviders,
  onOpenExecutionPolicySettings,
  messages = [],
  diagnosticRuntimeContext = null,
}: HarnessStatusPanelProps) {
  const { t, i18n } = useTranslation("agent");
  const translateAgent = useCallback<AgentTranslation>(
    (key, options) => String(t(key as never, options as never)),
    [t],
  );
  const translateProjection = useCallback<AgentUiProjectionTranslation>(
    (key, options) => String(t(key as never, options as never)),
    [t],
  );
  const locale = i18n.resolvedLanguage || i18n.language;
  const isDialogLayout = layout === "dialog";
  const [expanded, setExpanded] = useState(isDialogLayout);
  const isDetailsExpanded = isDialogLayout ? true : expanded;
  const activityModel = useHarnessActivityModel(
    harnessState,
    isDetailsExpanded,
  );
  const toolInventoryModel = useHarnessToolInventoryModel({
    toolInventory,
    toolInventoryError,
    toolInventoryLoading,
  });
  const fileReviewState = useHarnessFileReviewState(
    harnessState,
    isDetailsExpanded,
  );
  const { hasToolInventorySection, runtimeToolTotal, runtimeToolVisibleTotal } =
    toolInventoryModel;
  const {
    fileChangeReviewEntries,
    fileChangeReviewSummary,
    fileChangeStatusCounts,
  } = fileReviewState;
  const sectionRefs = useRef<
    Partial<Record<HarnessSectionKey, HTMLElement | null>>
  >({});
  const currentSessionId = diagnosticRuntimeContext?.sessionId?.trim() || null;
  const previewModel = useHarnessPreviewDialog({
    onLoadFilePreview,
    onOpenFile,
    onOpenPath,
    onRevealPath,
  });
  const {
    previewDialog,
    handlePreviewOpenChange,
    handleOpenFile,
    handleCopyPath,
    handleCopyContent,
    handleOpenPathValue,
    handleRevealPath,
    handleOpenPath,
  } = previewModel;
  const agentUiProjectionFilter = useMemo<AgentUiProjectionScopeFilter | null>(
    () => (currentSessionId ? { sessionId: currentSessionId } : null),
    [currentSessionId],
  );
  const agentUiProjectionEvents = useAgentUiProjectionEvents(
    agentUiProjectionFilter,
    { enabled: isDetailsExpanded },
  );
  const agentUiProjectionSummary = useMemo(
    () =>
      isDetailsExpanded && currentSessionId
        ? summarizeAgentUiProjectionEvents(agentUiProjectionEvents)
        : EMPTY_AGENT_UI_PROJECTION_SUMMARY,
    [agentUiProjectionEvents, currentSessionId, isDetailsExpanded],
  );
  const hasAgentUiProjectionSection = agentUiProjectionSummary.total > 0;

  const registerSectionRef = useCallback(
    (key: HarnessSectionKey, node: HTMLElement | null) => {
      sectionRefs.current[key] = node;
    },
    [],
  );

  const scrollToSection = useCallback((key: HarnessSectionKey) => {
    const target = sectionRefs.current[key];
    if (!target) {
      return;
    }
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);
  const handleToggleExpanded = useCallback(() => {
    setExpanded((value) => !value);
  }, []);

  const runtimeTaskPresentation = useMemo(
    () => buildRuntimeTaskPresentation(harnessState.runtimeStatus),
    [harnessState.runtimeStatus],
  );
  const realTeamSummary = useMemo(
    () => summarizeCanonicalChildThreads(canonicalChildren),
    [canonicalChildren],
  );
  const threadReliabilitySummary = useMemo(
    () =>
      buildThreadReliabilitySummary({
        threadRead,
        turns,
        currentTurnId,
        pendingActions,
        submittedActionsInFlight,
        t: translateProjection,
        locale,
      }),
    [
      currentTurnId,
      locale,
      pendingActions,
      submittedActionsInFlight,
      threadRead,
      translateProjection,
      turns,
    ],
  );
  const threadReliabilityView = useMemo(
    () =>
      isDetailsExpanded
        ? buildThreadReliabilityView({
            threadRead,
            turns,
            threadItems,
            currentTurnId,
            pendingActions,
            submittedActionsInFlight,
            t: translateProjection,
            locale,
          })
        : null,
    [
      currentTurnId,
      isDetailsExpanded,
      locale,
      pendingActions,
      submittedActionsInFlight,
      threadItems,
      threadRead,
      translateProjection,
      turns,
    ],
  );
  const runtimeFactSummary = useMemo(
    () => buildRuntimeFactSummary(threadRead),
    [threadRead],
  );
  const resolvedLeadContent =
    typeof leadContent === "function"
      ? leadContent({ fileChangeReviewSummary })
      : leadContent;

  const fileReviewTitle = String(
    t("agentChat.harness.fileReview.title" as never),
  );
  const fileReviewSummaryValue = String(
    t(
      "agentChat.harness.fileReview.summaryValue" as never,
      {
        pending: fileChangeStatusCounts.pending,
        total: fileChangeReviewEntries.length,
      } as never,
    ),
  );
  const fileReviewEmptyHint = String(
    t("agentChat.harness.fileReview.emptyHint" as never),
  );

  const availableSections = useMemo(
    () =>
      buildHarnessPanelSectionNavItems({
        environment,
        fileChangeReviewEntriesLength: fileChangeReviewEntries.length,
        hasAgentUiProjectionSection,
        hasToolInventorySection,
        harnessState,
        realTeamSummary,
        runtimeTaskPresentation,
        threadReliability: {
          shouldRender: threadReliabilitySummary.shouldRender,
          statusLabel: threadReliabilitySummary.statusLabel,
          summary: threadReliabilitySummary.summary,
        },
        fileReviewTitle,
        runtimeFactSummary,
      }),
    [
      environment,
      fileChangeReviewEntries.length,
      fileReviewTitle,
      hasAgentUiProjectionSection,
      hasToolInventorySection,
      harnessState,
      realTeamSummary,
      runtimeTaskPresentation,
      threadReliabilitySummary.shouldRender,
      threadReliabilitySummary.statusLabel,
      threadReliabilitySummary.summary,
      runtimeFactSummary,
    ],
  );

  const summaryCards = useMemo(
    () =>
      buildHarnessSummaryCards({
        agentUiProjectionSummary,
        environment,
        fileChangeReviewEntries,
        fileReviewCopy: {
          title: fileReviewTitle,
          summaryValue: fileReviewSummaryValue,
          emptyHint: fileReviewEmptyHint,
        },
        runtimeFactSummary,
        hasAgentUiProjectionSection,
        hasToolInventorySection,
        harnessState,
        realTeamSummary,
        runtimeTaskPresentation,
        runtimeToolTotal,
        runtimeToolVisibleTotal,
        threadReliability: {
          shouldRender: threadReliabilitySummary.shouldRender,
          statusLabel: threadReliabilitySummary.statusLabel,
          summary: threadReliabilitySummary.summary,
        },
        toolInventory: toolInventory ?? null,
        toolInventoryError,
        toolInventoryLoading,
        translateProjection,
      }),
    [
      agentUiProjectionSummary,
      environment,
      fileChangeReviewEntries,
      fileReviewEmptyHint,
      fileReviewSummaryValue,
      fileReviewTitle,
      hasAgentUiProjectionSection,
      hasToolInventorySection,
      harnessState,
      realTeamSummary,
      runtimeTaskPresentation,
      runtimeToolTotal,
      runtimeToolVisibleTotal,
      threadReliabilitySummary.shouldRender,
      threadReliabilitySummary.statusLabel,
      threadReliabilitySummary.summary,
      runtimeFactSummary,
      toolInventory,
      toolInventoryError,
      toolInventoryLoading,
      translateProjection,
    ],
  );

  const handleOpenExternalLink = useCallback(async (url: string) => {
    const normalizedUrl = url.trim();
    if (!normalizedUrl) {
      toast.error("当前没有可打开的链接");
      return;
    }

    try {
      await openExternalUrl(normalizedUrl);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "打开链接失败");
    }
  }, []);
  const sectionModels = useMemo(
    () =>
      isDetailsExpanded
        ? buildHarnessStatusPanelSectionModels({
            activityModel,
            agentUiProjectionSummary,
            canInterrupt,
            canonicalChildren,
            currentTurnId,
            diagnosticRuntimeContext,
            environment,
            fileReviewState,
            handleOpenExternalLink,
            hasAgentUiProjectionSection,
            harnessState,
            messages,
            onInterruptCurrentTurn,
            onOpenFileCheckpoints,
            onManageProviders,
            onOpenExecutionPolicySettings,
            onOpenSubagentSession,
            onRefreshToolInventory,
            onReplayPendingRequest,
            onRespondToAction,
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
          })
        : null,
    [
      activityModel,
      agentUiProjectionSummary,
      canInterrupt,
      canonicalChildren,
      currentTurnId,
      diagnosticRuntimeContext,
      environment,
      fileReviewState,
      handleOpenExternalLink,
      hasAgentUiProjectionSection,
      harnessState,
      isDetailsExpanded,
      messages,
      onInterruptCurrentTurn,
      onManageProviders,
      onOpenExecutionPolicySettings,
      onOpenFileCheckpoints,
      onOpenSubagentSession,
      onRefreshToolInventory,
      onReplayPendingRequest,
      onRespondToAction,
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
    ],
  );

  return (
    <>
      <HarnessStatusPanelShell
        layout={layout}
        title={title}
        description={description}
        toggleLabel={toggleLabel}
        isDetailsExpanded={isDetailsExpanded}
        realTeamActive={realTeamSummary.active > 0}
        leadContent={resolvedLeadContent}
        summaryCards={summaryCards}
        availableSections={availableSections}
        onToggleExpanded={handleToggleExpanded}
        onScrollToSection={scrollToSection}
      >
        {sectionModels ? (
          <HarnessStatusPanelSections {...sectionModels} />
        ) : null}
      </HarnessStatusPanelShell>

      <HarnessPreviewDialog
        previewDialog={previewDialog}
        canOpenInConversation={Boolean(onOpenFile)}
        onOpenChange={handlePreviewOpenChange}
        onOpenFile={handleOpenFile}
        onCopyPath={handleCopyPath}
        onCopyContent={handleCopyContent}
        onRevealPath={handleRevealPath}
        onOpenPath={handleOpenPath}
        onOpenPathValue={handleOpenPathValue}
        onOpenUrl={handleOpenExternalLink}
      />
    </>
  );
}
