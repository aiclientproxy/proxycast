import React, { memo, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Circle,
  Clock3,
  FileText,
  GitBranch,
  Image as ImageIcon,
  Plus,
  Trash2,
} from "lucide-react";
import type { StepStatus } from "@/lib/workspace/workbenchContract";
import type { AgentRun } from "@/lib/api/executionRun";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import type {
  TopicBranchItem,
  TopicBranchStatus,
} from "../hooks/useTopicBranchBoard";
import {
  listCuratedTaskRecommendationSignals,
  subscribeCuratedTaskRecommendationSignalsChanged,
} from "../utils/curatedTaskRecommendationSignals";
import {
  formatReviewFeedbackTemplate,
  type ReviewFeedbackProjection,
} from "../utils/reviewFeedbackProjection";
import type { CuratedTaskResultBaselineSnapshot } from "../utils/curatedTaskResultBaseline";
import type {
  GeneralWorkbenchActivityLogGroup,
  GeneralWorkbenchCreationTaskGroup,
  GeneralWorkbenchRunMetadataSummary,
} from "./generalWorkbenchWorkflowData";
import type { GeneralWorkbenchFollowUpActionPayload } from "./generalWorkbenchSidebarContract";
import { GeneralWorkbenchWorkflowControlBar } from "./GeneralWorkbenchWorkflowControlBar";
import {
  buildActivitySectionSummary,
  buildGeneralWorkbenchFollowUpProjection,
  buildGeneralWorkbenchActivitySectionProjection,
  buildGeneralWorkbenchBranchSectionProjection,
  buildGeneralWorkbenchCreationTaskSectionProjection,
  buildGeneralWorkbenchRunDetailProjection,
  buildGeneralWorkbenchWorkflowCurrentProjection,
  buildCreationTaskSectionSummary,
  buildCuratedTaskFollowUpActionItems,
  buildWorkflowResultHandoffText,
  selectLatestReviewFeedbackSignal,
} from "./generalWorkbenchWorkflowPanelViewModel";
import type { WorkspaceWorkflowControlItem } from "../workspace/workspaceWorkflowControls";

interface GeneralWorkbenchWorkflowPanelProps {
  isVersionMode: boolean;
  projectId?: string | null;
  sessionId?: string | null;
  onNewTopic: () => void;
  onSwitchTopic: (topicId: string) => void;
  onDeleteTopic: (topicId: string) => void;
  branchItems: TopicBranchItem[];
  onSetBranchStatus: (topicId: string, status: TopicBranchStatus) => void;
  onApplyFollowUpAction?: (
    payload: GeneralWorkbenchFollowUpActionPayload,
  ) => void;
  workflowSteps: Array<{ id: string; title: string; status: StepStatus }>;
  completedSteps: number;
  progressPercent: number;
  onAddImage?: () => Promise<void> | void;
  onImportDocument?: () => Promise<void> | void;
  showBranchRecords: boolean;
  onToggleBranchRecords: () => void;
  creationTaskEventsCount: number;
  showCreationTasks: boolean;
  onToggleCreationTasks: () => void;
  groupedCreationTaskEvents: GeneralWorkbenchCreationTaskGroup[];
  showActivityLogs: boolean;
  onToggleActivityLogs: () => void;
  groupedActivityLogs: GeneralWorkbenchActivityLogGroup[];
  onViewRunDetail?: (runId: string) => void;
  activeRunDetail?: AgentRun | null;
  activeRunDetailLoading?: boolean;
  activeRunStagesLabel?: string | null;
  workflowControlItems?: WorkspaceWorkflowControlItem[];
  workflowControlPendingItemId?: string | null;
  onTriggerWorkflowControl?: (
    item: WorkspaceWorkflowControlItem,
  ) => Promise<void> | void;
  runMetadataText: string;
  runMetadataSummary: GeneralWorkbenchRunMetadataSummary;
  onCopyText: (text: string) => Promise<void> | void;
  onRevealArtifactInFinder: (
    artifactPath: string,
    sessionId?: string | null,
  ) => Promise<void> | void;
  onOpenArtifactWithDefaultApp: (
    artifactPath: string,
    sessionId?: string | null,
  ) => Promise<void> | void;
}

const WORKFLOW_SECTION_CLASSNAME = "border-b border-slate-200/70 px-4 py-3";

const WORKFLOW_SECTION_TITLE_CLASSNAME =
  "mb-2.5 flex items-center justify-between text-[11px] font-semibold text-slate-500";

const WORKFLOW_SECTION_BADGE_CLASSNAME =
  "inline-flex min-h-4 min-w-4 items-center justify-center rounded-full border border-slate-200 bg-white px-1.5 text-[10px] font-semibold text-slate-500";

const WORKFLOW_NEW_TOPIC_BUTTON_CLASSNAME =
  "inline-flex h-7 shrink-0 items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-2.5 text-[11px] font-medium text-slate-500 transition-colors hover:border-slate-300 hover:bg-white hover:text-slate-900";

const WORKFLOW_TASK_SUMMARY_CLASSNAME =
  "mt-2 rounded-[14px] border border-slate-200 bg-slate-50 px-3 py-2.5";

const WORKFLOW_TASK_SUMMARY_PILL_CLASSNAME =
  "inline-flex items-center rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-500";

const WORKFLOW_RESULT_HANDOFF_HINT_CLASSNAME =
  "mt-2 rounded-[12px] border border-slate-200/80 bg-white/90 px-3 py-2";

const TOGGLE_BUTTON_CLASSNAME =
  "inline-flex items-center text-slate-500 transition-colors hover:text-slate-900";

const WORKFLOW_INLINE_LABEL_CLASSNAME =
  "text-[10px] font-semibold text-slate-500";

function createDiv(baseClassName: string) {
  return function ClassedDiv({
    className,
    ...props
  }: React.ComponentPropsWithoutRef<"div">) {
    return <div className={cn(baseClassName, className)} {...props} />;
  };
}

function createButton(baseClassName: string) {
  return function ClassedButton({
    className,
    ...props
  }: React.ComponentPropsWithoutRef<"button">) {
    return <button className={cn(baseClassName, className)} {...props} />;
  };
}

function createCode(baseClassName: string) {
  return function ClassedCode({
    className,
    ...props
  }: React.ComponentPropsWithoutRef<"code">) {
    return <code className={cn(baseClassName, className)} {...props} />;
  };
}

const BranchList = createDiv("flex flex-col gap-1.5");

const BranchSectionSummary = createDiv(
  "mt-2 text-[11px] leading-5 text-slate-500",
);

function BranchItem({
  $active,
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div"> & {
  $active: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-xl border px-3 py-2",
        $active
          ? "border-sky-200/80 bg-white"
          : "border-slate-200/80 bg-slate-50/60",
        className,
      )}
      {...props}
    />
  );
}

const BranchHead = createDiv("flex items-start gap-2");

const BranchTitleButton = createButton(
  "flex-1 truncate border-0 bg-transparent p-0 text-left text-[12px] font-medium leading-5 text-slate-900",
);

function StatusBadge({
  $status,
  className,
  ...props
}: React.ComponentPropsWithoutRef<"span"> & {
  $status: TopicBranchStatus;
}) {
  return (
    <span
      className={cn(
        "inline-flex rounded-full border px-1.5 py-0.5 text-[10px] font-medium",
        $status === "merged" &&
          "border-emerald-200 bg-emerald-50 text-emerald-700",
        $status === "in_progress" && "border-sky-200 bg-sky-50 text-sky-700",
        $status === "pending" && "border-amber-200 bg-amber-50 text-amber-700",
        $status !== "merged" &&
          $status !== "in_progress" &&
          $status !== "pending" &&
          "border-slate-200 bg-slate-100 text-slate-500",
        className,
      )}
      {...props}
    />
  );
}

const BranchMeta = createDiv("mt-1 text-[10px] leading-4 text-slate-500");

const BranchHint = createDiv("mt-1 text-[10px] leading-4 text-slate-400");

const ActionRow = createDiv("mt-2 flex flex-wrap gap-1.5");

const TinyButton = createButton(
  "rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-500 transition-colors hover:border-slate-300 hover:bg-slate-50 hover:text-slate-900",
);

const DeleteButton = createButton(
  "rounded-full p-1 text-slate-400 transition-colors hover:bg-rose-50 hover:text-rose-600",
);

const ActivityList = createDiv("flex flex-col gap-[5px]");

const ActivityMeta = createDiv(
  "mt-2 rounded-lg bg-slate-50/90 px-3 py-2 text-[11px] leading-6 text-slate-500",
);

const SecondarySectionSummaryCard = createDiv(
  "mt-1 rounded-[12px] border border-slate-200/70 bg-slate-50/70 px-3 py-2",
);

const SecondarySectionSummaryTitle = createDiv(
  "text-[11px] font-medium leading-5 text-slate-700",
);

const SecondarySectionSummaryMeta = createDiv(
  "mt-0.5 text-[10px] leading-5 text-slate-500",
);

const CreationTaskGroupCard = createDiv(
  "rounded-[14px] border border-slate-200/80 bg-white px-3 py-2.5",
);

const CreationTaskGroupHeader = createDiv(
  "flex items-center gap-2 text-slate-900",
);

const CreationTaskGroupCount = createDiv(
  "inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-500",
);

const CreationTaskList = createDiv("mt-2 flex flex-col gap-1.5");

const CreationTaskRow = createDiv(
  "flex items-start gap-2 rounded-xl border border-slate-200/80 bg-slate-50/70 px-2.5 py-2",
);

const CreationTaskContent = createDiv("min-w-0 flex-1");

const CreationTaskTitleRow = createDiv("flex items-start gap-2");

const CreationTaskTitle = createDiv(
  "min-w-0 flex-1 truncate text-[11px] font-medium leading-5 text-slate-900",
);

const CreationTaskTime = createDiv(
  "shrink-0 text-[10px] leading-5 text-slate-400",
);

const CreationTaskPath = createCode(
  "mt-1 block truncate rounded-md bg-white px-1.5 py-1 font-mono text-[10px] text-slate-500",
);

const ActivityLogCard = createDiv(
  "rounded-[14px] border border-slate-200/80 bg-white px-3 py-2.5",
);

const ActivityLogHeader = createDiv("flex items-start gap-2");

const ActivityLogTitleBlock = createDiv("min-w-0 flex-1");

const ActivityLogTitle = createDiv(
  "truncate text-[12px] font-medium leading-5 text-slate-900",
);

const ActivityLogMetaRow = createDiv(
  "mt-1 flex flex-wrap items-center gap-1.5",
);

const ActivityLogBadge = createDiv(
  "inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-500",
);

const ActivityLogSummary = createDiv(
  "mt-2 text-[11px] leading-5 text-slate-500",
);

const ActivityLogSteps = createDiv("mt-2 flex flex-col gap-1.5");

const ActivityLogStepRow = createDiv(
  "rounded-xl border border-slate-200/80 bg-slate-50/70 px-2.5 py-2",
);

const ActivityLogStepHead = createDiv(
  "flex items-start gap-2 text-[11px] leading-5 text-slate-900",
);

const ActivityLogStepSummary = createDiv(
  "mt-1 text-[10px] leading-4 text-slate-500",
);

const RunLinkButton = createButton(
  "border-0 bg-transparent p-0 text-[11px] leading-[1.35] text-sky-700 transition-colors hover:text-slate-900 disabled:cursor-default disabled:text-slate-400",
);

const RunDetailPanel = createDiv(
  "mt-2 rounded-[14px] border border-slate-200/80 bg-slate-50/80 px-3 py-2.5",
);

const RunDetailHeader = createDiv("flex items-start gap-2");

const RunDetailTitleBlock = createDiv("min-w-0 flex-1");

const RunDetailMetaRow = createDiv("mt-1 flex flex-wrap items-center gap-1.5");

const RunDetailSummary = createDiv("mt-2 text-[11px] leading-5 text-slate-500");

const RunDetailBadge = createDiv(
  "inline-flex items-center rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-500",
);

const RunDetailTitle = createDiv(
  "text-[12px] font-medium leading-5 text-slate-900",
);

const RunDetailRow = createDiv(
  "break-all text-[11px] leading-[1.45] text-slate-500",
);

const RunDetailArtifacts = createDiv("mt-2 flex flex-col gap-2");

const RunDetailArtifactsTitle = createDiv(
  "text-[10px] font-semibold text-slate-500",
);

const RunDetailArtifactRow = createDiv(
  "rounded-xl border border-slate-200/80 bg-white px-2.5 py-2",
);

const RunDetailArtifactPath = createCode(
  "block truncate rounded-md bg-slate-50 px-1.5 py-1 font-mono text-[10px] text-slate-900",
);

const RunDetailArtifactActions = createDiv("mt-1.5 flex flex-wrap gap-1.5");

const RunDetailArtifactActionButton = createButton(
  "rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-medium text-slate-500 transition-colors hover:border-slate-300 hover:bg-white hover:text-slate-900",
);

const RunDetailActions = createDiv("mt-1.5 flex gap-1.5");

const RunDetailActionButton = createButton(
  "rounded-md border border-slate-200 bg-white px-[7px] py-[3px] text-[11px] text-slate-700 transition-colors hover:border-slate-300 hover:bg-slate-50 hover:text-slate-900 disabled:cursor-default disabled:text-slate-400",
);

function getStepIcon(status: StepStatus) {
  if (status === "completed") {
    return <CheckCircle2 size={13} />;
  }
  if (status === "error") {
    return <AlertCircle size={13} />;
  }
  if (status === "active") {
    return <Clock3 size={13} />;
  }
  return <Circle size={11} />;
}

function getStatusBadgeClassName(status: StepStatus) {
  return cn(
    "inline-flex rounded-full border px-2 py-0.5 text-[10px] font-semibold",
    status === "completed" &&
      "border-emerald-200 bg-emerald-50 text-emerald-700",
    status === "error" && "border-rose-200 bg-rose-50 text-rose-700",
    status === "active" && "border-sky-200 bg-sky-50 text-sky-700",
    status === "pending" && "border-amber-200 bg-amber-50 text-amber-700",
    status === "skipped" && "border-slate-200 bg-slate-50 text-slate-500",
  );
}

function getWorkflowStepIconClassName(status: StepStatus) {
  return cn(
    status === "completed" && "text-emerald-600",
    status === "error" && "text-rose-600",
    status === "active" && "text-sky-600",
    status !== "completed" &&
      status !== "error" &&
      status !== "active" &&
      "text-slate-400",
  );
}

function ReviewFeedbackProjectionCard({
  projection,
  dataTestId,
  className,
  onApplyAction,
}: {
  projection: ReviewFeedbackProjection;
  dataTestId?: string;
  className?: string;
  onApplyAction?: () => void;
}) {
  const { t } = useTranslation("agent");
  const primarySuggestedTask = projection.suggestedTasks[0] ?? null;

  return (
    <div
      className={cn(
        "mt-2 rounded-[12px] border border-sky-200/80 bg-sky-50/70 px-3 py-2",
        className,
      )}
      data-testid={dataTestId}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center rounded-full border border-emerald-200 bg-white px-2 py-0.5 text-[10px] font-medium text-emerald-700">
          {t("reviewFeedback.badge")}
        </span>
        {!projection.matchedCurrentTask &&
        projection.suggestedTaskTitles.length > 0 ? (
          <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-500">
            {projection.suggestedTaskTitles.join(" / ")}
          </span>
        ) : null}
      </div>
      <div className="mt-1 text-[11px] font-medium leading-5 text-slate-900">
        {formatReviewFeedbackTemplate(
          t("reviewFeedback.title", {
            title: projection.signal.title,
          }),
          { title: projection.signal.title },
        )}
      </div>
      <div className="mt-1 text-[11px] leading-5 text-slate-500">
        {projection.signal.summary}
      </div>
      <div className="mt-1 text-[11px] leading-5 text-slate-500">
        {projection.suggestionText}
      </div>
      {primarySuggestedTask && onApplyAction ? (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="rounded-full border border-sky-200 bg-white px-3 py-1.5 text-[11px] font-medium text-slate-700 transition-colors hover:border-sky-300 hover:bg-sky-50"
            data-testid={dataTestId ? `${dataTestId}-action` : undefined}
            onClick={onApplyAction}
          >
            {formatReviewFeedbackTemplate(
              t("reviewFeedback.action", {
                title: primarySuggestedTask.title,
              }),
              { title: primarySuggestedTask.title },
            )}
          </button>
          <span className="text-[10px] leading-5 text-slate-500">
            {t("reviewFeedback.helper.generalWorkbench")}
          </span>
        </div>
      ) : null}
    </div>
  );
}

function ResultBaselineCard({
  snapshot,
  highlights,
  dataTestId,
  className,
}: {
  snapshot: CuratedTaskResultBaselineSnapshot;
  highlights: string[];
  dataTestId?: string;
  className?: string;
}) {
  const { t } = useTranslation("agent");

  return (
    <div
      className={cn(
        "mt-2 rounded-[12px] border border-emerald-200/80 bg-emerald-50/70 px-3 py-2",
        className,
      )}
      data-testid={dataTestId}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center rounded-full border border-emerald-200 bg-white px-2 py-0.5 text-[10px] font-medium text-emerald-700">
          {t("generalWorkbench.workflow.baseline.badge")}
        </span>
        <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-500">
          {t("generalWorkbench.workflow.baseline.projectResult")}
        </span>
      </div>
      <div className="mt-1 text-[11px] font-medium leading-5 text-slate-900">
        {snapshot.sourceTitle}
      </div>
      {highlights.length > 0 ? (
        <div className="mt-1 space-y-1 text-[11px] leading-5 text-emerald-900">
          {highlights.map((item) => (
            <div key={`result-baseline-${item}`}>{item}</div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function getActivityStatusBadgeClassName(
  status: GeneralWorkbenchActivityLogGroup["status"],
) {
  return cn(
    "inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium",
    status === "running" && "border-sky-200 bg-sky-50 text-sky-700",
    status === "failed" && "border-rose-200 bg-rose-50 text-rose-700",
    status === "completed" &&
      "border-emerald-200 bg-emerald-50 text-emerald-700",
  );
}

function getRunDetailStatusBadgeClassName(status: AgentRun["status"]) {
  return cn(
    "inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium",
    status === "running" && "border-sky-200 bg-sky-50 text-sky-700",
    status === "success" && "border-emerald-200 bg-emerald-50 text-emerald-700",
    status === "error" && "border-rose-200 bg-rose-50 text-rose-700",
    status === "queued" && "border-amber-200 bg-amber-50 text-amber-700",
    status === "canceled" && "border-slate-200 bg-slate-100 text-slate-500",
    status === "timeout" && "border-rose-200 bg-rose-50 text-rose-700",
  );
}

function renderActivityLogItem(
  projection: ReturnType<
    typeof buildGeneralWorkbenchActivitySectionProjection
  >["logs"][number],
  onViewRunDetail: GeneralWorkbenchWorkflowPanelProps["onViewRunDetail"],
  onRevealArtifactInFinder: GeneralWorkbenchWorkflowPanelProps["onRevealArtifactInFinder"],
  onOpenArtifactWithDefaultApp: GeneralWorkbenchWorkflowPanelProps["onOpenArtifactWithDefaultApp"],
) {
  return (
    <ActivityLogCard key={`activity-${projection.key}`}>
      <ActivityLogHeader>
        <div className={getActivityStatusBadgeClassName(projection.status)}>
          {projection.statusLabel}
        </div>
        <ActivityLogTitleBlock>
          <ActivityLogTitle>{projection.title}</ActivityLogTitle>
          <ActivityLogMetaRow>
            {projection.sourceLabel ? (
              <ActivityLogBadge>{projection.sourceLabel}</ActivityLogBadge>
            ) : null}
            {projection.gateLabel ? (
              <ActivityLogBadge>{projection.gateLabel}</ActivityLogBadge>
            ) : null}
            {projection.stepCountLabel ? (
              <ActivityLogBadge>{projection.stepCountLabel}</ActivityLogBadge>
            ) : null}
            {projection.artifactCountLabel ? (
              <ActivityLogBadge>
                {projection.artifactCountLabel}
              </ActivityLogBadge>
            ) : null}
          </ActivityLogMetaRow>
        </ActivityLogTitleBlock>
        <div className="shrink-0 text-[10px] leading-5 text-slate-400">
          {projection.timeLabel}
        </div>
      </ActivityLogHeader>
      {projection.summary ? (
        <ActivityLogSummary>{projection.summary}</ActivityLogSummary>
      ) : null}
      <ActivityLogSteps>
        {projection.steps.map((step) => (
          <ActivityLogStepRow key={step.id}>
            <ActivityLogStepHead>
              <span className="text-slate-400">•</span>
              <span className="min-w-0 flex-1 break-words">{step.name}</span>
              <span className="shrink-0 text-[10px] text-slate-400">
                {step.timeLabel}
              </span>
            </ActivityLogStepHead>
            {step.summary ? (
              <ActivityLogStepSummary>{step.summary}</ActivityLogStepSummary>
            ) : null}
          </ActivityLogStepRow>
        ))}
      </ActivityLogSteps>
      <ActionRow>
        {projection.runAction && onViewRunDetail ? (
          <RunLinkButton
            type="button"
            onClick={() => onViewRunDetail(projection.runAction!.runId)}
          >
            {projection.runAction.label}
          </RunLinkButton>
        ) : null}
        {projection.artifactActions.length > 0
          ? projection.artifactActions.map((artifactActionGroup) => (
              <ActivityMetaFragment
                key={`${projection.key}-${artifactActionGroup.path}`}
                actionGroup={artifactActionGroup}
                onRevealArtifactInFinder={onRevealArtifactInFinder}
                onOpenArtifactWithDefaultApp={onOpenArtifactWithDefaultApp}
              />
            ))
          : null}
      </ActionRow>
    </ActivityLogCard>
  );
}

function ActivityMetaFragment({
  actionGroup,
  onRevealArtifactInFinder,
  onOpenArtifactWithDefaultApp,
}: {
  actionGroup: ReturnType<
    typeof buildGeneralWorkbenchActivitySectionProjection
  >["logs"][number]["artifactActions"][number];
  onRevealArtifactInFinder: GeneralWorkbenchWorkflowPanelProps["onRevealArtifactInFinder"];
  onOpenArtifactWithDefaultApp: GeneralWorkbenchWorkflowPanelProps["onOpenArtifactWithDefaultApp"];
}) {
  return (
    <>
      {actionGroup.actions.map((action) => (
        <TinyButton
          key={`${actionGroup.path}-${action.kind}`}
          type="button"
          aria-label={action.ariaLabel}
          onClick={() => {
            if (action.kind === "reveal") {
              void onRevealArtifactInFinder(
                action.targetPath,
                actionGroup.sessionId,
              );
              return;
            }
            void onOpenArtifactWithDefaultApp(
              action.targetPath,
              actionGroup.sessionId,
            );
          }}
        >
          {action.label}
        </TinyButton>
      ))}
    </>
  );
}

function CuratedTaskFollowUpActions({
  curatedTask,
  onApplyFollowUpAction,
}: {
  curatedTask: GeneralWorkbenchRunMetadataSummary["curatedTask"];
  onApplyFollowUpAction?: GeneralWorkbenchWorkflowPanelProps["onApplyFollowUpAction"];
}) {
  const { t } = useTranslation("agent");
  const actionItems = buildCuratedTaskFollowUpActionItems({
    curatedTask,
    t,
  });
  if (!onApplyFollowUpAction || actionItems.length === 0) {
    return null;
  }

  return (
    <ActionRow data-testid="workflow-sidebar-follow-up-actions">
      {actionItems.map((item) => (
        <TinyButton
          key={item.action}
          type="button"
          aria-label={item.ariaLabel}
          onClick={() => {
            onApplyFollowUpAction(item.payload);
          }}
        >
          {item.action}
        </TinyButton>
      ))}
    </ActionRow>
  );
}

function GeneralWorkbenchWorkflowPanelComponent({
  isVersionMode,
  projectId,
  sessionId,
  onNewTopic,
  onSwitchTopic,
  onDeleteTopic,
  branchItems,
  onSetBranchStatus,
  onApplyFollowUpAction,
  workflowSteps,
  completedSteps,
  progressPercent,
  onAddImage,
  onImportDocument,
  showBranchRecords,
  onToggleBranchRecords,
  creationTaskEventsCount,
  showCreationTasks,
  onToggleCreationTasks,
  groupedCreationTaskEvents,
  showActivityLogs,
  onToggleActivityLogs,
  groupedActivityLogs,
  onViewRunDetail,
  activeRunDetail,
  activeRunDetailLoading = false,
  activeRunStagesLabel,
  workflowControlItems = [],
  workflowControlPendingItemId = null,
  onTriggerWorkflowControl,
  runMetadataText,
  runMetadataSummary,
  onCopyText,
  onRevealArtifactInFinder,
  onOpenArtifactWithDefaultApp,
}: GeneralWorkbenchWorkflowPanelProps) {
  const { t } = useTranslation("agent");
  const [recommendationSignalsVersion, setRecommendationSignalsVersion] =
    useState(0);

  useEffect(() => {
    return subscribeCuratedTaskRecommendationSignalsChanged(() => {
      setRecommendationSignalsVersion((previous) => previous + 1);
    });
  }, []);

  const workflowCurrentProjection =
    buildGeneralWorkbenchWorkflowCurrentProjection({
      workflowSteps,
      completedSteps,
      progressPercent,
      t,
    });
  const branchProjection = buildGeneralWorkbenchBranchSectionProjection({
    branchItems,
    isVersionMode,
    t,
  });
  const creationTaskSectionSummary = buildCreationTaskSectionSummary({
    groups: groupedCreationTaskEvents,
    totalCount: creationTaskEventsCount,
    t,
  });
  const creationTaskSectionProjection =
    buildGeneralWorkbenchCreationTaskSectionProjection({
      groups: groupedCreationTaskEvents,
      t,
    });
  const workflowResultHandoffText = buildWorkflowResultHandoffText({
    branchSectionTitle: branchProjection.sectionTitle,
    hasRecordedOutputs:
      creationTaskEventsCount > 0 ||
      groupedActivityLogs.length > 0 ||
      runMetadataSummary.artifactPaths.length > 0,
    resultDestination: runMetadataSummary.curatedTask?.resultDestination,
    t,
  });
  const latestReviewSignal = useMemo(() => {
    void recommendationSignalsVersion;
    return selectLatestReviewFeedbackSignal(
      listCuratedTaskRecommendationSignals({
        projectId,
        sessionId,
      }),
    );
  }, [projectId, recommendationSignalsVersion, sessionId]);
  const followUpProjection = buildGeneralWorkbenchFollowUpProjection({
    latestReviewSignal,
    runMetadataSummary,
    t,
  });
  const activitySectionSummary = buildActivitySectionSummary({
    groups: groupedActivityLogs,
    activeRunDetail,
    t,
  });
  const activitySectionProjection =
    buildGeneralWorkbenchActivitySectionProjection({
      groups: groupedActivityLogs,
      t,
    });
  const runDetailProjection = activeRunDetail
    ? buildGeneralWorkbenchRunDetailProjection({
        activeRunDetail,
        runMetadataSummary,
        runMetadataText,
        activeRunStagesLabel,
        t,
      })
    : null;

  return (
    <>
      <section
        className={WORKFLOW_SECTION_CLASSNAME}
        data-testid="workflow-sidebar-task-section"
      >
        <div className={WORKFLOW_SECTION_TITLE_CLASSNAME}>
          <span>{t("generalWorkbench.workflow.current.title")}</span>
        </div>
        <div
          className={WORKFLOW_TASK_SUMMARY_CLASSNAME}
          data-testid="workflow-sidebar-task-summary"
        >
          <div className="flex items-start gap-2.5">
            <span
              className={cn(
                "mt-0.5",
                getWorkflowStepIconClassName(
                  workflowCurrentProjection.currentStepIconStatus,
                ),
              )}
            >
              {getStepIcon(workflowCurrentProjection.currentStepIconStatus)}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className={WORKFLOW_INLINE_LABEL_CLASSNAME}>
                  {t("generalWorkbench.workflow.current.focus")}
                </span>
                <span
                  className="break-words text-sm font-semibold leading-5 text-slate-900"
                  data-testid="workflow-sidebar-current-step"
                >
                  {workflowCurrentProjection.currentStepTitle}
                </span>
                <span
                  className={getStatusBadgeClassName(
                    workflowCurrentProjection.currentStepStatus,
                  )}
                >
                  {workflowCurrentProjection.currentStepStatusLabel}
                </span>
              </div>
              <div className="mt-1 text-[11px] leading-5 text-slate-500">
                {workflowCurrentProjection.workflowSummaryText}
              </div>
              <GeneralWorkbenchWorkflowControlBar
                items={workflowControlItems}
                pendingItemId={workflowControlPendingItemId}
                onTrigger={onTriggerWorkflowControl}
                translate={t}
              />
              <div
                className={WORKFLOW_RESULT_HANDOFF_HINT_CLASSNAME}
                data-testid="workflow-sidebar-result-destination-hint"
              >
                <div className="text-[10px] font-semibold text-slate-500">
                  {t("generalWorkbench.workflow.current.resultDestination")}
                </div>
                <div className="mt-1 text-[11px] leading-5 text-slate-500">
                  {workflowResultHandoffText}
                </div>
              </div>
              {followUpProjection.shouldShowFollowUpHint ? (
                <div
                  className={WORKFLOW_RESULT_HANDOFF_HINT_CLASSNAME}
                  data-testid="workflow-sidebar-follow-up-hint"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="text-[10px] font-semibold text-slate-500">
                      {t("generalWorkbench.workflow.current.suggestedNext")}
                    </div>
                    {runMetadataSummary.curatedTask?.taskTitle ? (
                      <span className={WORKFLOW_TASK_SUMMARY_PILL_CLASSNAME}>
                        {runMetadataSummary.curatedTask.taskTitle}
                      </span>
                    ) : null}
                  </div>
                  {followUpProjection.resultBaselineSnapshot ? (
                    <ResultBaselineCard
                      snapshot={followUpProjection.resultBaselineSnapshot}
                      highlights={followUpProjection.resultBaselineHighlights}
                      dataTestId="workflow-sidebar-result-baseline-card"
                    />
                  ) : null}
                  {followUpProjection.reviewFeedbackProjection ? (
                    <ReviewFeedbackProjectionCard
                      projection={followUpProjection.reviewFeedbackProjection}
                      dataTestId="workflow-sidebar-review-feedback-banner"
                      onApplyAction={(() => {
                        const payload =
                          followUpProjection.reviewFeedbackFollowUpActionPayload;
                        if (!payload || !onApplyFollowUpAction) {
                          return undefined;
                        }
                        return () => onApplyFollowUpAction(payload);
                      })()}
                    />
                  ) : null}
                  {followUpProjection.curatedTaskFollowUpHintText ? (
                    <>
                      <div className="mt-1 text-[11px] leading-5 text-slate-500">
                        {followUpProjection.curatedTaskFollowUpHintText}
                      </div>
                      <CuratedTaskFollowUpActions
                        curatedTask={runMetadataSummary.curatedTask}
                        onApplyFollowUpAction={onApplyFollowUpAction}
                      />
                    </>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </section>

      <section
        className={cn(WORKFLOW_SECTION_CLASSNAME, "relative z-10")}
        data-testid="workflow-sidebar-branch-section"
      >
        <div className={WORKFLOW_SECTION_TITLE_CLASSNAME}>
          <span>{branchProjection.sectionTitle}</span>
          <span className="inline-flex items-center gap-2">
            <span className={WORKFLOW_SECTION_BADGE_CLASSNAME}>
              {branchItems.length}
            </span>
            <button
              type="button"
              aria-label={t("generalWorkbench.workflow.branch.toggleAria")}
              className={TOGGLE_BUTTON_CLASSNAME}
              onClick={onToggleBranchRecords}
            >
              {showBranchRecords ? (
                <ChevronDown size={13} />
              ) : (
                <ChevronRight size={13} />
              )}
            </button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  className={WORKFLOW_NEW_TOPIC_BUTTON_CLASSNAME}
                >
                  <Plus size={13} />
                  {branchProjection.createLabel}
                  <ChevronDown size={11} />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" style={{ width: "260px" }}>
                <DropdownMenuItem onClick={onNewTopic}>
                  <GitBranch size={14} />
                  <span>{branchProjection.createLabel}</span>
                </DropdownMenuItem>
                {onAddImage ? (
                  <DropdownMenuItem onClick={onAddImage}>
                    <ImageIcon size={14} />
                    <span>
                      {t("generalWorkbench.workflow.branch.addImage")}
                    </span>
                  </DropdownMenuItem>
                ) : null}
                {onImportDocument ? (
                  <DropdownMenuItem onClick={onImportDocument}>
                    <FileText size={14} />
                    <span>
                      {t("generalWorkbench.workflow.branch.importDraft")}
                    </span>
                  </DropdownMenuItem>
                ) : null}
              </DropdownMenuContent>
            </DropdownMenu>
          </span>
        </div>
        {branchItems.length === 0 ? (
          <ActivityMeta>{branchProjection.emptyText}</ActivityMeta>
        ) : (
          <>
            <BranchSectionSummary data-testid="workflow-sidebar-branch-summary">
              {branchProjection.summaryText}
            </BranchSectionSummary>
            {showBranchRecords ? (
              <BranchList className="mt-2 custom-scrollbar">
                {branchProjection.itemProjections.map((item) => (
                  <BranchItem key={item.id} $active={item.isCurrent}>
                    <BranchHead>
                      <GitBranch
                        size={13}
                        className={cn(
                          "mt-0.5 shrink-0",
                          item.isCurrent ? "text-sky-600" : "text-slate-400",
                        )}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start gap-2">
                          <BranchTitleButton
                            onClick={() => onSwitchTopic(item.id)}
                          >
                            {item.title}
                          </BranchTitleButton>
                          <StatusBadge $status={item.status}>
                            {item.statusLabel}
                          </StatusBadge>
                          {item.deleteAriaLabel ? (
                            <DeleteButton
                              onClick={() => onDeleteTopic(item.id)}
                              aria-label={item.deleteAriaLabel}
                            >
                              <Trash2 size={12} />
                            </DeleteButton>
                          ) : null}
                        </div>
                        <BranchMeta>{item.metaText}</BranchMeta>
                        {item.actionItems.length > 0 ? (
                          <ActionRow>
                            {item.actionItems.map((action) => (
                              <TinyButton
                                key={`${item.id}-${action.kind}`}
                                onClick={() =>
                                  onSetBranchStatus(item.id, action.status)
                                }
                              >
                                {action.label}
                              </TinyButton>
                            ))}
                          </ActionRow>
                        ) : null}
                        {item.actionItems.length === 0 && item.hintText ? (
                          <BranchHint>{item.hintText}</BranchHint>
                        ) : null}
                      </div>
                    </BranchHead>
                  </BranchItem>
                ))}
              </BranchList>
            ) : null}
          </>
        )}
      </section>

      <section className={WORKFLOW_SECTION_CLASSNAME}>
        <div className={WORKFLOW_SECTION_TITLE_CLASSNAME}>
          <span>{t("generalWorkbench.workflow.outputs.title")}</span>
          <span className="inline-flex items-center gap-1.5">
            <span className={WORKFLOW_SECTION_BADGE_CLASSNAME}>
              {creationTaskEventsCount}
            </span>
            <button
              type="button"
              aria-label={t("generalWorkbench.workflow.outputs.toggleAria")}
              className={TOGGLE_BUTTON_CLASSNAME}
              onClick={onToggleCreationTasks}
            >
              {showCreationTasks ? (
                <ChevronDown size={13} />
              ) : (
                <ChevronRight size={13} />
              )}
            </button>
          </span>
        </div>
        <SecondarySectionSummaryCard data-testid="workflow-sidebar-creation-summary">
          <SecondarySectionSummaryTitle>
            {creationTaskSectionSummary.title}
          </SecondarySectionSummaryTitle>
          <SecondarySectionSummaryMeta>
            {creationTaskSectionSummary.meta}
          </SecondarySectionSummaryMeta>
        </SecondarySectionSummaryCard>
        {showCreationTasks ? (
          <ActivityList className="custom-scrollbar">
            {creationTaskSectionProjection.groups.length === 0 ? (
              <ActivityMeta>
                {creationTaskSectionProjection.emptyText}
              </ActivityMeta>
            ) : (
              creationTaskSectionProjection.groups.map((projection) => (
                <CreationTaskGroupCard key={`creation-task-${projection.key}`}>
                  <CreationTaskGroupHeader>
                    <span>{projection.label}</span>
                    <CreationTaskGroupCount>
                      {projection.countLabel}
                    </CreationTaskGroupCount>
                    <span className="ml-auto text-[10px] text-slate-400">
                      {projection.latestTimeLabel}
                    </span>
                  </CreationTaskGroupHeader>
                  <CreationTaskList>
                    {projection.tasks.map((task) => (
                      <CreationTaskRow key={task.key}>
                        <CreationTaskContent>
                          <CreationTaskTitleRow>
                            <CreationTaskTitle>{task.title}</CreationTaskTitle>
                            <CreationTaskTime>
                              {task.timeLabel}
                            </CreationTaskTime>
                          </CreationTaskTitleRow>
                          <CreationTaskPath>{task.path}</CreationTaskPath>
                        </CreationTaskContent>
                        <RunDetailActionButton
                          type="button"
                          aria-label={task.copyAriaLabel}
                          onClick={() => {
                            void onCopyText(task.copyTarget);
                          }}
                        >
                          {creationTaskSectionProjection.copyLabel}
                        </RunDetailActionButton>
                      </CreationTaskRow>
                    ))}
                  </CreationTaskList>
                </CreationTaskGroupCard>
              ))
            )}
          </ActivityList>
        ) : null}
      </section>

      <section className={WORKFLOW_SECTION_CLASSNAME}>
        <div className={WORKFLOW_SECTION_TITLE_CLASSNAME}>
          <span>{t("generalWorkbench.workflow.activity.title")}</span>
          <span className="inline-flex items-center gap-1.5">
            <span className={WORKFLOW_SECTION_BADGE_CLASSNAME}>
              {groupedActivityLogs.length}
            </span>
            <button
              type="button"
              aria-label={t("generalWorkbench.workflow.activity.toggleAria")}
              className={TOGGLE_BUTTON_CLASSNAME}
              onClick={onToggleActivityLogs}
            >
              {showActivityLogs ? (
                <ChevronDown size={13} />
              ) : (
                <ChevronRight size={13} />
              )}
            </button>
          </span>
        </div>
        <SecondarySectionSummaryCard data-testid="workflow-sidebar-activity-summary">
          <SecondarySectionSummaryTitle>
            {activitySectionSummary.title}
          </SecondarySectionSummaryTitle>
          <SecondarySectionSummaryMeta>
            {activitySectionSummary.meta}
          </SecondarySectionSummaryMeta>
        </SecondarySectionSummaryCard>
        {showActivityLogs ? (
          <>
            <ActivityList className="custom-scrollbar">
              {activitySectionProjection.logs.length === 0 ? (
                <ActivityMeta>
                  {activitySectionProjection.emptyText}
                </ActivityMeta>
              ) : (
                activitySectionProjection.logs.map((projection) =>
                  renderActivityLogItem(
                    projection,
                    onViewRunDetail,
                    onRevealArtifactInFinder,
                    onOpenArtifactWithDefaultApp,
                  ),
                )
              )}
            </ActivityList>
            {activeRunDetailLoading ? (
              <ActivityMeta>
                {activitySectionProjection.loadingText}
              </ActivityMeta>
            ) : activeRunDetail && runDetailProjection ? (
              <RunDetailPanel>
                <RunDetailHeader>
                  <div
                    className={getRunDetailStatusBadgeClassName(
                      runDetailProjection.status,
                    )}
                  >
                    {runDetailProjection.statusLabel}
                  </div>
                  <RunDetailTitleBlock>
                    <RunDetailTitle>
                      {activitySectionProjection.runDetailTitle}
                    </RunDetailTitle>
                    <RunDetailMetaRow>
                      {runDetailProjection.badges.map((badge) => (
                        <RunDetailBadge key={`run-detail-badge-${badge}`}>
                          {badge}
                        </RunDetailBadge>
                      ))}
                    </RunDetailMetaRow>
                  </RunDetailTitleBlock>
                </RunDetailHeader>
                <RunDetailSummary>
                  {runDetailProjection.summary}
                </RunDetailSummary>
                {runDetailProjection.detailRows.length > 0 ? (
                  <div className="mt-2 flex flex-col gap-1.5">
                    {runDetailProjection.detailRows.map((row) => (
                      <RunDetailRow key={row.key}>
                        {row.label}：{row.value}
                      </RunDetailRow>
                    ))}
                  </div>
                ) : null}
                {followUpProjection.resultBaselineSnapshot ? (
                  <ResultBaselineCard
                    snapshot={followUpProjection.resultBaselineSnapshot}
                    highlights={followUpProjection.resultBaselineHighlights}
                    dataTestId="workflow-run-detail-result-baseline-card"
                  />
                ) : null}
                {followUpProjection.reviewFeedbackProjection ? (
                  <ReviewFeedbackProjectionCard
                    projection={followUpProjection.reviewFeedbackProjection}
                    dataTestId="workflow-run-detail-review-feedback-banner"
                    onApplyAction={(() => {
                      const payload =
                        followUpProjection.reviewFeedbackFollowUpActionPayload;
                      if (!payload || !onApplyFollowUpAction) {
                        return undefined;
                      }
                      return () => onApplyFollowUpAction(payload);
                    })()}
                  />
                ) : null}
                {followUpProjection.curatedTaskFollowUpHintText ? (
                  <>
                    <RunDetailSummary>
                      {followUpProjection.curatedTaskFollowUpHintText}
                    </RunDetailSummary>
                    <CuratedTaskFollowUpActions
                      curatedTask={runMetadataSummary.curatedTask}
                      onApplyFollowUpAction={onApplyFollowUpAction}
                    />
                  </>
                ) : null}
                <RunDetailRow>
                  {t("generalWorkbench.workflow.runDetail.id", {
                    id: runDetailProjection.id,
                  })}
                </RunDetailRow>
                <RunDetailActions>
                  {runDetailProjection.actions.map((action) => (
                    <RunDetailActionButton
                      key={action.kind}
                      type="button"
                      aria-label={action.ariaLabel}
                      onClick={() => {
                        void onCopyText(action.copyTarget);
                      }}
                    >
                      {action.label}
                    </RunDetailActionButton>
                  ))}
                </RunDetailActions>
                {runDetailProjection.artifacts.length > 0 ? (
                  <RunDetailArtifacts>
                    <RunDetailArtifactsTitle>
                      {t("generalWorkbench.workflow.runDetail.artifactsTitle")}
                    </RunDetailArtifactsTitle>
                    {runDetailProjection.artifacts.map((artifact) => (
                      <RunDetailArtifactRow key={`run-detail-${artifact.path}`}>
                        <RunDetailArtifactPath>
                          {artifact.path}
                        </RunDetailArtifactPath>
                        <RunDetailArtifactActions>
                          {artifact.actions.map((action) => (
                            <RunDetailArtifactActionButton
                              key={`${artifact.path}-${action.kind}`}
                              type="button"
                              aria-label={action.ariaLabel}
                              onClick={() => {
                                if (action.kind === "copy") {
                                  void onCopyText(action.targetPath);
                                  return;
                                }
                                if (action.kind === "reveal") {
                                  void onRevealArtifactInFinder(
                                    action.targetPath,
                                  );
                                  return;
                                }
                                void onOpenArtifactWithDefaultApp(
                                  action.targetPath,
                                );
                              }}
                            >
                              {action.label}
                            </RunDetailArtifactActionButton>
                          ))}
                        </RunDetailArtifactActions>
                      </RunDetailArtifactRow>
                    ))}
                  </RunDetailArtifacts>
                ) : null}
              </RunDetailPanel>
            ) : null}
          </>
        ) : null}
      </section>
    </>
  );
}

export const GeneralWorkbenchWorkflowPanel = memo(
  GeneralWorkbenchWorkflowPanelComponent,
);
