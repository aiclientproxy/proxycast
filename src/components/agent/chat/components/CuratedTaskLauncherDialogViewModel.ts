import {
  summarizeCuratedTaskFollowUpActions,
  summarizeCuratedTaskOutputContract,
  summarizeCuratedTaskRequiredInputs,
  findCuratedTaskTemplateById,
  hasFilledAllCuratedTaskRequiredInputs,
  type CuratedTaskPresentationCopy,
  type CuratedTaskInputValues,
  type CuratedTaskTemplateItem,
} from "@/components/agent/chat/utils/curatedTaskTemplates";
import type { CuratedTaskRecommendationSignal } from "@/components/agent/chat/utils/curatedTaskRecommendationSignals";
import { buildReviewFeedbackProjection } from "@/components/agent/chat/utils/reviewFeedbackProjection";
import type { CuratedTaskReferenceEntry } from "@/components/agent/chat/utils/curatedTaskReferenceSelection";
import {
  buildCuratedTaskResultBaselineHighlights,
  buildCuratedTaskResultBaselineSnapshot,
} from "@/components/agent/chat/utils/curatedTaskResultBaseline";

export const MAX_REFERENCE_SELECTION_COUNT = 3;

export interface CuratedTaskLauncherCopy {
  readinessReady: string;
  readinessMissing: (count: number) => string;
  outcomeWithFollowUp: (values: {
    followUp: string;
    outputHint: string;
  }) => string;
  outcomeDefault: (values: { outputHint: string }) => string;
  contractRequiredEmpty: string;
  carryFieldSeparator: string;
  carryReview: (fields: string) => string;
  carryDefault: (fields: string) => string;
}

export interface CuratedTaskLauncherReadiness {
  isLaunchDisabled: boolean;
  requiredFieldCount: number;
  filledRequiredFieldCount: number;
  remainingRequiredFieldCount: number;
  launcherReadinessLabel: string;
}

export interface CuratedTaskLauncherStarterContract {
  requiredSummary: string;
  outputSummary: string;
  followUpSummary: string;
}

export function buildCuratedTaskLauncherReadiness({
  task,
  inputValues,
  copy,
}: {
  task: CuratedTaskTemplateItem | null;
  inputValues: CuratedTaskInputValues;
  copy: Pick<CuratedTaskLauncherCopy, "readinessReady" | "readinessMissing">;
}): CuratedTaskLauncherReadiness {
  if (!task) {
    return {
      isLaunchDisabled: true,
      requiredFieldCount: 0,
      filledRequiredFieldCount: 0,
      remainingRequiredFieldCount: 0,
      launcherReadinessLabel: copy.readinessMissing(0),
    };
  }

  const requiredFieldCount = task.requiredInputFields.length;
  const filledRequiredFieldCount = task.requiredInputFields.filter((field) => {
    const value = inputValues[field.key];
    return typeof value === "string" && value.trim().length > 0;
  }).length;
  const remainingRequiredFieldCount = Math.max(
    requiredFieldCount - filledRequiredFieldCount,
    0,
  );

  return {
    isLaunchDisabled: !hasFilledAllCuratedTaskRequiredInputs({
      task,
      inputValues,
    }),
    requiredFieldCount,
    filledRequiredFieldCount,
    remainingRequiredFieldCount,
    launcherReadinessLabel:
      remainingRequiredFieldCount === 0
        ? copy.readinessReady
        : copy.readinessMissing(remainingRequiredFieldCount),
  };
}

export function resolveSelectedReferenceEntries({
  referenceEntries,
  selectedReferenceEntryIds,
}: {
  referenceEntries: CuratedTaskReferenceEntry[];
  selectedReferenceEntryIds: string[];
}): {
  selectedReferenceEntries: CuratedTaskReferenceEntry[];
  missingSelectedReferenceCount: number;
} {
  const referenceEntryMap = new Map(
    referenceEntries.map((entry) => [entry.id, entry]),
  );
  const selectedReferenceEntries = selectedReferenceEntryIds
    .map((id) => referenceEntryMap.get(id))
    .filter((entry): entry is CuratedTaskReferenceEntry => Boolean(entry));

  return {
    selectedReferenceEntries,
    missingSelectedReferenceCount:
      selectedReferenceEntryIds.length - selectedReferenceEntries.length,
  };
}

export function planReferenceEntrySelection({
  currentIds,
  entryId,
  maxSelectionCount = MAX_REFERENCE_SELECTION_COUNT,
}: {
  currentIds: string[];
  entryId: string;
  maxSelectionCount?: number;
}): string[] {
  if (currentIds.includes(entryId)) {
    return currentIds.filter((id) => id !== entryId);
  }

  if (currentIds.length >= maxSelectionCount) {
    return currentIds;
  }

  return [...currentIds, entryId];
}

export function buildLauncherOutcomeSummary({
  task,
  copy,
}: {
  task: CuratedTaskTemplateItem | null;
  copy: Pick<CuratedTaskLauncherCopy, "outcomeWithFollowUp" | "outcomeDefault">;
}): string {
  if (!task) {
    return "";
  }

  const followUp = task.followUpActions[0];
  if (followUp) {
    return copy.outcomeWithFollowUp({
      followUp,
      outputHint: task.outputHint,
    });
  }

  return copy.outcomeDefault({
    outputHint: task.outputHint,
  });
}

export function buildLauncherStarterContract({
  task,
  presentationCopy,
  copy,
}: {
  task: CuratedTaskTemplateItem | null;
  presentationCopy: CuratedTaskPresentationCopy;
  copy: Pick<CuratedTaskLauncherCopy, "contractRequiredEmpty">;
}): CuratedTaskLauncherStarterContract | null {
  if (!task) {
    return null;
  }

  return {
    requiredSummary:
      summarizeCuratedTaskRequiredInputs(task, 2, presentationCopy) ||
      copy.contractRequiredEmpty,
    outputSummary:
      summarizeCuratedTaskOutputContract(task, 2, presentationCopy) ||
      task.outputHint,
    followUpSummary: summarizeCuratedTaskFollowUpActions(
      task,
      2,
      presentationCopy,
    ),
  };
}

export function selectLatestReviewTaskSignal(
  signals: CuratedTaskRecommendationSignal[],
): CuratedTaskRecommendationSignal | null {
  return (
    signals
      .filter((signal) => signal.source === "review_feedback")
      .sort((left, right) => right.createdAt - left.createdAt)[0] ?? null
  );
}

export function resolvePrimarySuggestedTask({
  currentTask,
  latestReviewTaskSignal,
  curatedTaskTemplateCopy,
}: {
  currentTask: CuratedTaskTemplateItem | null;
  latestReviewTaskSignal: CuratedTaskRecommendationSignal | null;
  curatedTaskTemplateCopy: Parameters<typeof findCuratedTaskTemplateById>[1];
}): {
  reviewFeedbackProjection: ReturnType<
    typeof buildReviewFeedbackProjection
  > | null;
  primarySuggestedTask: CuratedTaskTemplateItem | null;
} {
  if (!currentTask) {
    return {
      reviewFeedbackProjection: null,
      primarySuggestedTask: null,
    };
  }

  const reviewFeedbackProjection = buildReviewFeedbackProjection({
    signal: latestReviewTaskSignal,
    currentTaskId: currentTask.id,
    currentTaskTitle: currentTask.title,
  });
  if (
    !reviewFeedbackProjection ||
    reviewFeedbackProjection.matchedCurrentTask
  ) {
    return {
      reviewFeedbackProjection,
      primarySuggestedTask: null,
    };
  }

  const suggestedTaskId = reviewFeedbackProjection.suggestedTasks[0]?.taskId;
  return {
    reviewFeedbackProjection,
    primarySuggestedTask: suggestedTaskId
      ? findCuratedTaskTemplateById(suggestedTaskId, curatedTaskTemplateCopy)
      : null,
  };
}

export function buildActiveReviewBaselineModel({
  task,
  selectedReferenceEntries,
  seededReferenceEntries,
  inputValues,
  copy,
}: {
  task: CuratedTaskTemplateItem | null;
  selectedReferenceEntries: CuratedTaskReferenceEntry[];
  seededReferenceEntries: CuratedTaskReferenceEntry[];
  inputValues: CuratedTaskInputValues;
  copy: Pick<
    CuratedTaskLauncherCopy,
    "carryFieldSeparator" | "carryReview" | "carryDefault"
  >;
}): {
  activeReviewBaselineSnapshot: ReturnType<
    typeof buildCuratedTaskResultBaselineSnapshot
  > | null;
  activeReviewBaselineHighlights: string[];
  activeReviewBaselineCarryHint: string | null;
} {
  if (!task) {
    return {
      activeReviewBaselineSnapshot: null,
      activeReviewBaselineHighlights: [],
      activeReviewBaselineCarryHint: null,
    };
  }

  const activeReferenceEntries =
    selectedReferenceEntries.length > 0
      ? selectedReferenceEntries
      : seededReferenceEntries;
  const activeReviewBaselineSnapshot = buildCuratedTaskResultBaselineSnapshot({
    referenceEntries: activeReferenceEntries,
    taskId: task.id,
  });
  const activeReviewBaselineHighlights =
    buildCuratedTaskResultBaselineHighlights(activeReviewBaselineSnapshot);

  if (!activeReviewBaselineSnapshot) {
    return {
      activeReviewBaselineSnapshot,
      activeReviewBaselineHighlights,
      activeReviewBaselineCarryHint: null,
    };
  }

  const carriedFields = task.requiredInputFields
    .filter((field) => (inputValues[field.key] ?? "").trim())
    .map((field) => field.label);
  if (carriedFields.length === 0) {
    return {
      activeReviewBaselineSnapshot,
      activeReviewBaselineHighlights,
      activeReviewBaselineCarryHint: null,
    };
  }

  const fields = carriedFields.join(copy.carryFieldSeparator);
  return {
    activeReviewBaselineSnapshot,
    activeReviewBaselineHighlights,
    activeReviewBaselineCarryHint:
      task.id === "account-project-review"
        ? copy.carryReview(fields)
        : copy.carryDefault(fields),
  };
}
