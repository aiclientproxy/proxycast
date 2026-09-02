import {
  buildCuratedTaskFollowUpDescription,
  buildCuratedTaskLaunchPrompt,
  findCuratedTaskTemplateById,
  resolveCuratedTaskFollowUpActionTarget,
} from "../utils/curatedTaskTemplates";
import { buildCuratedTaskLaunchInputPrefillFromReferenceEntries } from "../utils/curatedTaskReferenceSelection";
import {
  buildReviewFeedbackProjection,
  buildReviewFeedbackProjectionCopy,
  type ReviewFeedbackProjection,
} from "../utils/reviewFeedbackProjection";
import type { CuratedTaskRecommendationSignal } from "../utils/curatedTaskRecommendationSignals";
import {
  buildCuratedTaskFollowUpActionFromReferences,
  buildCuratedTaskResultBaselineHighlights,
  buildCuratedTaskResultBaselineSnapshot,
} from "../utils/curatedTaskResultBaseline";
import type { GeneralWorkbenchFollowUpActionPayload } from "./generalWorkbenchSidebarContract";
import type { GeneralWorkbenchRunMetadataSummary } from "./generalWorkbenchWorkflowData";
import type {
  GeneralWorkbenchCuratedTaskFollowUpActionItem,
  GeneralWorkbenchFollowUpProjection,
  GeneralWorkbenchWorkflowPanelTranslate,
} from "./generalWorkbenchWorkflowPanelTypes";

export function buildWorkflowResultHandoffText(params: {
  branchSectionTitle: string;
  hasRecordedOutputs: boolean;
  resultDestination?: string | null;
  t: GeneralWorkbenchWorkflowPanelTranslate;
}): string {
  const { branchSectionTitle, hasRecordedOutputs, resultDestination, t } =
    params;
  const normalizedResultDestination = resultDestination?.trim() || null;
  if (normalizedResultDestination) {
    return t("generalWorkbench.workflow.handoff.withDestination", {
      destination: normalizedResultDestination,
      branchTitle: branchSectionTitle,
    });
  }
  return hasRecordedOutputs
    ? t("generalWorkbench.workflow.handoff.defaultContinuing", {
        branchTitle: branchSectionTitle,
      })
    : t("generalWorkbench.workflow.handoff.defaultInitial", {
        branchTitle: branchSectionTitle,
      });
}

export function buildCuratedTaskFollowUpHintText(
  curatedTask: GeneralWorkbenchRunMetadataSummary["curatedTask"],
  t: GeneralWorkbenchWorkflowPanelTranslate,
): string | null {
  if (!curatedTask) {
    return null;
  }

  const followUpSummary = buildCuratedTaskFollowUpDescription(
    {
      followUpActions: curatedTask.followUpActions,
    },
    {
      limit: 2,
      prefix: "",
    },
  );

  if (followUpSummary) {
    return curatedTask.taskTitle
      ? t("generalWorkbench.workflow.followUp.hint.withTaskAndSummary", {
          title: curatedTask.taskTitle,
          summary: followUpSummary,
        })
      : t("generalWorkbench.workflow.followUp.hint.summaryOnly", {
          summary: followUpSummary,
        });
  }

  if (curatedTask.taskTitle) {
    return t("generalWorkbench.workflow.followUp.hint.taskOnly", {
      title: curatedTask.taskTitle,
    });
  }

  return null;
}

export function selectLatestReviewFeedbackSignal(
  signals: CuratedTaskRecommendationSignal[],
): CuratedTaskRecommendationSignal | null {
  return (
    signals
      .filter((signal) => signal.source === "review_feedback")
      .sort((left, right) => right.createdAt - left.createdAt)[0] ?? null
  );
}

export function listVisibleCuratedTaskFollowUpActions(
  curatedTask: GeneralWorkbenchRunMetadataSummary["curatedTask"],
  limit = 2,
): string[] {
  if (!curatedTask) {
    return [];
  }

  return Array.from(
    new Set(
      curatedTask.followUpActions
        .map((action) => action.trim())
        .filter((action) => action.length > 0),
    ),
  ).slice(0, Math.max(0, limit));
}

export function buildCuratedTaskFollowUpActionItems({
  curatedTask,
  t,
}: {
  curatedTask: GeneralWorkbenchRunMetadataSummary["curatedTask"];
  t: GeneralWorkbenchWorkflowPanelTranslate;
}): GeneralWorkbenchCuratedTaskFollowUpActionItem[] {
  return listVisibleCuratedTaskFollowUpActions(curatedTask)
    .map((action) => {
      const payload = buildCuratedTaskFollowUpActionPayload({
        action,
        curatedTask,
        t,
      });
      if (!payload) {
        return null;
      }

      return {
        action,
        ariaLabel: t("generalWorkbench.workflow.followUp.applyAria", {
          action,
        }),
        payload,
      };
    })
    .filter((item): item is GeneralWorkbenchCuratedTaskFollowUpActionItem =>
      Boolean(item),
    );
}

function buildCuratedTaskFollowUpPrompt(params: {
  action: string;
  curatedTask: GeneralWorkbenchRunMetadataSummary["curatedTask"];
  t: GeneralWorkbenchWorkflowPanelTranslate;
}): string {
  const normalizedAction = params.action.trim();
  if (!normalizedAction) {
    return "";
  }

  const taskTitle = params.curatedTask?.taskTitle?.trim();
  if (taskTitle) {
    return params.t("generalWorkbench.workflow.followUp.prompt.withTask", {
      title: taskTitle,
      action: normalizedAction,
    });
  }
  return params.t("generalWorkbench.workflow.followUp.prompt.current", {
    action: normalizedAction,
  });
}

function buildCuratedTaskFollowUpBannerMessage(params: {
  action: string;
  curatedTask: GeneralWorkbenchRunMetadataSummary["curatedTask"];
  targetTask?: { title: string } | null;
  t: GeneralWorkbenchWorkflowPanelTranslate;
}): string | undefined {
  const normalizedAction = params.action.trim();
  if (!normalizedAction) {
    return undefined;
  }

  const targetTaskTitle = params.targetTask?.title?.trim();
  if (targetTaskTitle) {
    return params.t("generalWorkbench.workflow.followUp.banner.withTarget", {
      title: targetTaskTitle,
    });
  }

  const currentTaskTitle = params.curatedTask?.taskTitle?.trim();
  if (currentTaskTitle) {
    return params.t(
      "generalWorkbench.workflow.followUp.banner.withCurrentTask",
      {
        action: normalizedAction,
        title: currentTaskTitle,
      },
    );
  }

  return params.t("generalWorkbench.workflow.followUp.banner.current", {
    action: normalizedAction,
  });
}

export function buildCuratedTaskFollowUpActionPayload(params: {
  action: string;
  curatedTask: GeneralWorkbenchRunMetadataSummary["curatedTask"];
  t: GeneralWorkbenchWorkflowPanelTranslate;
}): GeneralWorkbenchFollowUpActionPayload | null {
  const normalizedAction = params.action.trim();
  if (!normalizedAction) {
    return null;
  }

  const currentTaskId = params.curatedTask?.taskId?.trim();
  const currentTaskTitle = params.curatedTask?.taskTitle?.trim();
  const mappedTarget = resolveCuratedTaskFollowUpActionTarget({
    taskId: currentTaskId,
    action: normalizedAction,
  });
  const targetTask = mappedTarget?.task ?? null;
  const referenceEntries = params.curatedTask?.referenceEntries;
  const referenceMemoryIds = params.curatedTask?.referenceMemoryIds;
  const launchInputValues = targetTask
    ? buildCuratedTaskLaunchInputPrefillFromReferenceEntries({
        taskId: targetTask.id,
        inputValues: params.curatedTask?.launchInputValues,
        referenceEntries,
      })
    : params.curatedTask?.launchInputValues;
  const targetPrompt = targetTask
    ? buildCuratedTaskLaunchPrompt({
        task: targetTask,
        inputValues: launchInputValues ?? {},
        referenceEntries,
      }).trim()
    : "";
  const prompt =
    targetTask && targetPrompt
      ? [mappedTarget?.promptHint?.trim(), targetPrompt]
          .filter((segment): segment is string => Boolean(segment))
          .join("\n\n")
      : buildCuratedTaskFollowUpPrompt({
          action: normalizedAction,
          curatedTask: params.curatedTask,
          t: params.t,
        });
  if (!prompt) {
    return null;
  }

  const capabilityRoute =
    (targetTask?.id && targetTask.title.trim()) ||
    (currentTaskId && currentTaskTitle)
      ? {
          kind: "curated_task" as const,
          taskId: targetTask?.id ?? currentTaskId!,
          taskTitle: targetTask?.title.trim() || currentTaskTitle!,
          prompt,
          ...(launchInputValues
            ? {
                launchInputValues,
              }
            : {}),
          ...(referenceMemoryIds
            ? {
                referenceMemoryIds,
              }
            : {}),
          ...(referenceEntries
            ? {
                referenceEntries,
              }
            : {}),
        }
      : undefined;

  return {
    prompt,
    bannerMessage: buildCuratedTaskFollowUpBannerMessage({
      action: normalizedAction,
      curatedTask: params.curatedTask,
      targetTask,
      t: params.t,
    }),
    ...(capabilityRoute
      ? {
          capabilityRoute,
        }
      : {}),
  };
}

export function buildReviewFeedbackFollowUpActionPayload(params: {
  projection: ReviewFeedbackProjection;
  curatedTask: GeneralWorkbenchRunMetadataSummary["curatedTask"];
  t: GeneralWorkbenchWorkflowPanelTranslate;
}): GeneralWorkbenchFollowUpActionPayload | null {
  const primarySuggestedTask = params.projection.suggestedTasks[0];
  if (!primarySuggestedTask) {
    return null;
  }

  const referenceEntries = params.curatedTask?.referenceEntries;
  const referenceMemoryIds = params.curatedTask?.referenceMemoryIds;
  const launchInputValues = params.curatedTask?.launchInputValues;
  const referencePayload = buildCuratedTaskFollowUpActionFromReferences({
    taskId: primarySuggestedTask.taskId,
    inputValues: launchInputValues,
    referenceEntries,
  });
  if (referencePayload) {
    return referencePayload;
  }

  const targetTask = findCuratedTaskTemplateById(primarySuggestedTask.taskId);
  if (!targetTask) {
    return null;
  }

  const prompt = buildCuratedTaskLaunchPrompt({
    task: targetTask,
    inputValues: launchInputValues ?? {},
    referenceEntries,
  }).trim();
  if (!prompt) {
    return null;
  }

  return {
    prompt,
    bannerMessage: params.t(
      "generalWorkbench.workflow.reviewFeedback.followUpBanner",
      {
        title: targetTask.title,
      },
    ),
    capabilityRoute: {
      kind: "curated_task",
      taskId: targetTask.id,
      taskTitle: targetTask.title,
      prompt,
      ...(launchInputValues
        ? {
            launchInputValues,
          }
        : {}),
      ...(referenceMemoryIds
        ? {
            referenceMemoryIds,
          }
        : {}),
      ...(referenceEntries
        ? {
            referenceEntries,
          }
        : {}),
    },
  };
}

export function buildGeneralWorkbenchFollowUpProjection({
  latestReviewSignal,
  runMetadataSummary,
  t,
}: {
  latestReviewSignal: CuratedTaskRecommendationSignal | null;
  runMetadataSummary: GeneralWorkbenchRunMetadataSummary;
  t: GeneralWorkbenchWorkflowPanelTranslate;
}): GeneralWorkbenchFollowUpProjection {
  const reviewFeedbackProjection = buildReviewFeedbackProjection({
    copy: buildReviewFeedbackProjectionCopy(t),
    signal: latestReviewSignal,
    currentTaskId: runMetadataSummary.curatedTask?.taskId,
    currentTaskTitle: runMetadataSummary.curatedTask?.taskTitle,
  });
  const reviewFeedbackFollowUpActionPayload = reviewFeedbackProjection
    ? buildReviewFeedbackFollowUpActionPayload({
        projection: reviewFeedbackProjection,
        curatedTask: runMetadataSummary.curatedTask,
        t,
      })
    : null;
  const resultBaselineSnapshot = runMetadataSummary.curatedTask?.taskId
    ? buildCuratedTaskResultBaselineSnapshot({
        referenceEntries: runMetadataSummary.curatedTask.referenceEntries,
        taskId: runMetadataSummary.curatedTask.taskId,
      })
    : null;
  const resultBaselineHighlights = buildCuratedTaskResultBaselineHighlights(
    resultBaselineSnapshot,
  );
  const curatedTaskFollowUpHintText = buildCuratedTaskFollowUpHintText(
    runMetadataSummary.curatedTask,
    t,
  );

  return {
    reviewFeedbackProjection,
    reviewFeedbackFollowUpActionPayload,
    resultBaselineSnapshot,
    resultBaselineHighlights,
    curatedTaskFollowUpHintText,
    shouldShowFollowUpHint: Boolean(
      reviewFeedbackProjection ||
      curatedTaskFollowUpHintText ||
      resultBaselineSnapshot,
    ),
  };
}
