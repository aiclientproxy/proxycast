import type { TFunction } from "i18next";
import type { AgentRun } from "@/lib/api/executionRun";
import type { StepStatus } from "@/lib/workspace/workbenchContract";
import type {
  TopicBranchItem,
  TopicBranchStatus,
} from "../hooks/useTopicBranchBoard";
import type { ReviewFeedbackProjection } from "../utils/reviewFeedbackProjection";
import type { CuratedTaskResultBaselineSnapshot } from "../utils/curatedTaskResultBaseline";
import type { GeneralWorkbenchFollowUpActionPayload } from "./generalWorkbenchSidebarContract";
import type {
  buildGeneralWorkbenchActivityLogGroups,
  buildGeneralWorkbenchCreationTaskGroups,
  GeneralWorkbenchActivityLogGroup,
  GeneralWorkbenchRunMetadataSummary,
} from "./generalWorkbenchWorkflowData";

export type GeneralWorkbenchWorkflowPanelTranslate = TFunction<
  "agent",
  undefined
>;

export interface GeneralWorkbenchWorkflowStepInput {
  id: string;
  title: string;
  status: StepStatus;
}

export interface GeneralWorkbenchWorkflowPanelViewModel {
  completedSteps: number;
  progressPercent: number;
  groupedActivityLogs: ReturnType<
    typeof buildGeneralWorkbenchActivityLogGroups
  >;
  groupedCreationTaskEvents: ReturnType<
    typeof buildGeneralWorkbenchCreationTaskGroups
  >;
  activeRunStagesLabel: string | null;
  runMetadataText: string;
  runMetadataSummary: GeneralWorkbenchRunMetadataSummary;
}

export interface GeneralWorkbenchWorkflowCurrentProjection {
  currentWorkflowStep: GeneralWorkbenchWorkflowStepInput | null;
  currentStepTitle: string;
  currentStepIconStatus: StepStatus;
  currentStepStatus: StepStatus;
  currentStepStatusLabel: string;
  remainingSteps: number;
  visibleQueueSteps: GeneralWorkbenchWorkflowStepInput[];
  queueItems: GeneralWorkbenchWorkflowQueueItemProjection[];
  hiddenQueueCount: number;
  completedWorkflowSteps: number;
  workflowSummaryText: string;
  workflowProgressLabel: string;
  remainingText: string;
  progressBarPercent: number;
  progressPercentLabel: string;
  queueHeaderText: string | null;
  completedCountText: string | null;
  completedHintText: string | null;
}

export interface GeneralWorkbenchWorkflowQueueItemProjection {
  id: string;
  title: string;
  status: StepStatus;
  indexLabel: string;
  statusLabel: string;
}

export interface GeneralWorkbenchBranchSectionProjection {
  sectionTitle: string;
  createLabel: string;
  primaryActionLabel: string;
  secondaryActionLabel: string;
  sortedBranchItems: TopicBranchItem[];
  itemProjections: GeneralWorkbenchBranchItemProjection[];
  currentBranchItem: TopicBranchItem | null;
  secondaryBranchCount: number;
  summaryText: string;
  emptyText: string;
}

export interface GeneralWorkbenchBranchItemProjection {
  id: string;
  title: string;
  status: TopicBranchStatus;
  isCurrent: boolean;
  statusLabel: string;
  metaText: string;
  deleteAriaLabel: string | null;
  hintText: string | null;
  actionItems: Array<{
    kind: "primary" | "secondary";
    status: TopicBranchStatus;
    label: string;
  }>;
  item: TopicBranchItem;
}

export interface GeneralWorkbenchActivityLogProjection {
  key: string;
  status: GeneralWorkbenchActivityLogGroup["status"];
  statusLabel: string;
  title: string;
  timeLabel: string;
  sourceLabel: string | null;
  gateLabel: string | null;
  stepCountLabel: string | null;
  artifactCountLabel: string | null;
  summary: string | null;
  runId: string | null;
  runLabel: string | null;
  runAction: GeneralWorkbenchActivityRunAction | null;
  artifactPaths: string[];
  artifactActions: GeneralWorkbenchActivityArtifactActionGroup[];
  sessionId: string | null;
  steps: Array<{
    id: string;
    name: string;
    timeLabel: string;
    summary: string | null;
  }>;
}

export interface GeneralWorkbenchActivitySectionProjection {
  emptyText: string;
  loadingText: string;
  runDetailTitle: string;
  logs: GeneralWorkbenchActivityLogProjection[];
}

export interface GeneralWorkbenchActivityRunAction {
  runId: string;
  label: string;
}

export type GeneralWorkbenchActivityArtifactActionKind = "reveal" | "open";

export interface GeneralWorkbenchActivityArtifactActionItem {
  kind: GeneralWorkbenchActivityArtifactActionKind;
  label: string;
  ariaLabel: string;
  targetPath: string;
}

export interface GeneralWorkbenchActivityArtifactActionGroup {
  path: string;
  sessionId: string | null;
  actions: GeneralWorkbenchActivityArtifactActionItem[];
}

export interface GeneralWorkbenchRunDetailProjection {
  id: string;
  status: AgentRun["status"];
  statusLabel: string;
  sourceLabel: string;
  badges: string[];
  summary: string;
  detailRows: GeneralWorkbenchRunDetailFactRow[];
  actions: GeneralWorkbenchRunDetailActionItem[];
  artifactPaths: string[];
  artifacts: GeneralWorkbenchRunDetailArtifactProjection[];
}

export interface GeneralWorkbenchRunDetailFactRow {
  key: string;
  label: string;
  value: string;
}

export type GeneralWorkbenchRunDetailActionKind = "copy_id" | "copy_raw";

export interface GeneralWorkbenchRunDetailActionItem {
  kind: GeneralWorkbenchRunDetailActionKind;
  label: string;
  ariaLabel: string;
  copyTarget: string;
}

export type GeneralWorkbenchRunDetailArtifactActionKind =
  | "copy"
  | "reveal"
  | "open";

export interface GeneralWorkbenchRunDetailArtifactActionItem {
  kind: GeneralWorkbenchRunDetailArtifactActionKind;
  label: string;
  ariaLabel: string;
  targetPath: string;
}

export interface GeneralWorkbenchRunDetailArtifactProjection {
  path: string;
  actions: GeneralWorkbenchRunDetailArtifactActionItem[];
}

export interface GeneralWorkbenchFollowUpProjection {
  reviewFeedbackProjection: ReviewFeedbackProjection | null;
  reviewFeedbackFollowUpActionPayload: GeneralWorkbenchFollowUpActionPayload | null;
  resultBaselineSnapshot: CuratedTaskResultBaselineSnapshot | null;
  resultBaselineHighlights: string[];
  curatedTaskFollowUpHintText: string | null;
  shouldShowFollowUpHint: boolean;
}

export interface GeneralWorkbenchCuratedTaskFollowUpActionItem {
  action: string;
  ariaLabel: string;
  payload: GeneralWorkbenchFollowUpActionPayload;
}

export interface GeneralWorkbenchCreationTaskGroupProjection {
  key: string;
  label: string;
  countLabel: string;
  latestTimeLabel: string;
  tasks: Array<{
    key: string;
    title: string;
    timeLabel: string;
    path: string;
    copyTarget: string;
    copyAriaLabel: string;
  }>;
}

export interface GeneralWorkbenchCreationTaskSectionProjection {
  emptyText: string;
  copyLabel: string;
  groups: GeneralWorkbenchCreationTaskGroupProjection[];
}
