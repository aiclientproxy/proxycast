import React, { useEffect, useMemo, useState } from "react";
import { ListChecks, PencilLine, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  buildCuratedTaskCapabilityDescription,
  buildCuratedTaskFollowUpDescription,
  findCuratedTaskTemplateById,
  type CuratedTaskTemplateItem,
} from "../utils/curatedTaskTemplates";
import {
  listCuratedTaskRecommendationSignals,
  subscribeCuratedTaskRecommendationSignalsChanged,
} from "../utils/curatedTaskRecommendationSignals";
import { buildReviewFeedbackProjection } from "../utils/reviewFeedbackProjection";
import type { CuratedTaskReferenceEntry } from "../utils/curatedTaskReferenceSelection";
import {
  buildCuratedTaskResultBaselineHighlights,
  buildCuratedTaskResultBaselineSnapshot,
} from "../utils/curatedTaskResultBaseline";

interface CuratedTaskBadgeProps {
  task: CuratedTaskTemplateItem;
  projectId?: string | null;
  sessionId?: string | null;
  referenceEntries?: CuratedTaskReferenceEntry[] | null;
  onEdit?: () => void;
  onApplyReviewSuggestion?: (task: CuratedTaskTemplateItem) => void;
  onClear: () => void;
}

function truncateBadgeReviewText(value: string, maxLength = 28): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength).trimEnd()}...`;
}

export const CuratedTaskBadge: React.FC<CuratedTaskBadgeProps> = ({
  task,
  projectId,
  sessionId,
  referenceEntries,
  onEdit,
  onApplyReviewSuggestion,
  onClear,
}) => {
  const { t } = useTranslation("agent");
  const [recommendationSignalsVersion, setRecommendationSignalsVersion] =
    useState(0);

  useEffect(() => {
    return subscribeCuratedTaskRecommendationSignalsChanged(() => {
      setRecommendationSignalsVersion((previous) => previous + 1);
    });
  }, []);

  const followUpSummary = buildCuratedTaskFollowUpDescription(task, {
    limit: 2,
  });
  const badgeTitle = buildCuratedTaskCapabilityDescription(task, {
    includeSummary: false,
    includeResultDestination: true,
    includeFollowUpActions: true,
    followUpLimit: 2,
  });
  const resultBaselineSnapshot = useMemo(() => {
    return buildCuratedTaskResultBaselineSnapshot({
      referenceEntries,
      taskId: task.id,
    });
  }, [referenceEntries, task.id]);
  const resultBaselineHighlights = useMemo(
    () => buildCuratedTaskResultBaselineHighlights(resultBaselineSnapshot),
    [resultBaselineSnapshot],
  );
  const latestReviewSignal = useMemo(() => {
    void recommendationSignalsVersion;
    return (
      listCuratedTaskRecommendationSignals({
        projectId,
        sessionId,
      })
        .filter((signal) => signal.source === "review_feedback")
        .sort((left, right) => right.createdAt - left.createdAt)[0] ?? null
    );
  }, [projectId, recommendationSignalsVersion, sessionId]);
  const reviewProjection = useMemo(
    () =>
      buildReviewFeedbackProjection({
        signal: latestReviewSignal,
        currentTaskId: task.id,
        currentTaskTitle: task.title,
      }),
    [latestReviewSignal, task.id, task.title],
  );
  const visibleReviewProjection =
    reviewProjection &&
    (reviewProjection.matchedCurrentTask || Boolean(onApplyReviewSuggestion))
      ? reviewProjection
      : null;
  const recentReviewLabel = t("curatedTask.launcher.review.badge");
  const recentReviewTitlePrefix = t("curatedTask.badge.review.titlePrefix");
  const primarySuggestedTask = useMemo(() => {
    if (
      !visibleReviewProjection ||
      visibleReviewProjection.matchedCurrentTask
    ) {
      return null;
    }

    const suggestedTaskId = visibleReviewProjection.suggestedTasks[0]?.taskId;
    if (!suggestedTaskId) {
      return null;
    }

    return findCuratedTaskTemplateById(suggestedTaskId);
  }, [visibleReviewProjection]);
  const reviewSummary = visibleReviewProjection
    ? truncateBadgeReviewText(
        visibleReviewProjection.matchedCurrentTask
          ? visibleReviewProjection.signal.title.startsWith(
              recentReviewTitlePrefix,
            )
            ? visibleReviewProjection.signal.title
            : t("curatedTask.templates.recommendation.reviewReasonSummary", {
                title: visibleReviewProjection.signal.title,
              })
          : t("curatedTask.badge.review.suggestedTask", {
              title:
                primarySuggestedTask?.title ||
                visibleReviewProjection.signal.title,
            }),
      )
    : null;
  const resultStatusSummary = resultBaselineSnapshot?.statusLabel
    ? truncateBadgeReviewText(
        t("inputCapabilities.baseline.status", {
          value: resultBaselineSnapshot.statusLabel,
        }),
        34,
      )
    : resultBaselineSnapshot?.failureSignalLabel
      ? truncateBadgeReviewText(
          t("curatedTask.badge.result.failureSignal", {
            value: resultBaselineSnapshot.failureSignalLabel,
          }),
          34,
        )
      : null;
  const resultNextSummary = resultBaselineSnapshot?.destinationsLabel
    ? truncateBadgeReviewText(
        t("inputCapabilities.baseline.destination", {
          value: resultBaselineSnapshot.destinationsLabel,
        }),
        28,
      )
    : resultBaselineSnapshot?.operatingAction
      ? truncateBadgeReviewText(
          t("inputCapabilities.baseline.operatingAction", {
            value: resultBaselineSnapshot.operatingAction,
          }),
          28,
        )
      : null;
  const resultSummaryTitle =
    resultBaselineSnapshot && resultBaselineHighlights.length > 0
      ? [
          t("inputCapabilities.baseline.title", {
            title: resultBaselineSnapshot.sourceTitle,
          }),
          ...resultBaselineHighlights,
        ].join(t("skills.workspace.curatedTask.segmentSeparator"))
      : resultBaselineSnapshot?.sourceTitle
        ? t("inputCapabilities.baseline.title", {
            title: resultBaselineSnapshot.sourceTitle,
          })
        : null;

  return (
    <div
      data-testid="curated-task-badge"
      className="mx-1 mt-1 inline-flex max-w-full flex-wrap items-center gap-1.5 rounded-md bg-amber-500/10 px-2 py-1.5 text-xs font-medium text-amber-700"
      title={badgeTitle || task.title}
    >
      <ListChecks className="h-3 w-3" />
      <span>{task.title}</span>
      {reviewSummary ? (
        <span
          data-testid="curated-task-badge-review-signal"
          className="inline-flex max-w-[240px] items-center rounded-full border border-emerald-300/70 bg-white/90 px-2 py-0.5 text-[11px] leading-4 text-emerald-700"
          title={t("curatedTask.badge.review.signalTitle", {
            label: recentReviewLabel,
            summary:
              visibleReviewProjection?.signal.summary ||
              visibleReviewProjection?.signal.title ||
              "",
          })}
        >
          <span className="truncate">
            {visibleReviewProjection?.matchedCurrentTask
              ? t("curatedTask.badge.review.matchedSummary", {
                  label: recentReviewLabel,
                  summary: reviewSummary,
                })
              : reviewSummary}
          </span>
        </span>
      ) : null}
      {primarySuggestedTask && onApplyReviewSuggestion ? (
        <button
          type="button"
          data-testid="curated-task-badge-review-action"
          className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-white/90 px-2 py-0.5 text-[11px] leading-4 text-sky-700 transition hover:bg-white"
          title={t("curatedTask.badge.review.switchTitle", {
            title: primarySuggestedTask.title,
          })}
          onClick={() => onApplyReviewSuggestion(primarySuggestedTask)}
        >
          <span className="truncate">
            {t("curatedTask.launcher.review.switchAction", {
              title: primarySuggestedTask.title,
            })}
          </span>
        </button>
      ) : null}
      {resultStatusSummary ? (
        <span
          data-testid="curated-task-badge-result-status"
          className="inline-flex max-w-[240px] items-center rounded-full border border-sky-300/70 bg-white/90 px-2 py-0.5 text-[11px] leading-4 text-sky-700"
          title={resultSummaryTitle || undefined}
        >
          <span className="truncate">{resultStatusSummary}</span>
        </span>
      ) : null}
      {resultNextSummary ? (
        <span
          data-testid="curated-task-badge-result-next"
          className="inline-flex max-w-[220px] items-center rounded-full border border-emerald-300/70 bg-white/90 px-2 py-0.5 text-[11px] leading-4 text-emerald-700"
          title={resultSummaryTitle || undefined}
        >
          <span className="truncate">{resultNextSummary}</span>
        </span>
      ) : null}
      {followUpSummary ? (
        <span className="inline-flex max-w-[320px] items-center rounded-full border border-amber-300/70 bg-white/80 px-2 py-0.5 text-[11px] leading-4 text-amber-700">
          <span className="truncate">{followUpSummary}</span>
        </span>
      ) : null}
      {onEdit ? (
        <button
          type="button"
          onClick={onEdit}
          className="ml-0.5 inline-flex items-center gap-1 rounded-full border border-amber-300/80 bg-white/80 px-1.5 py-0.5 text-[11px] text-amber-700 transition hover:bg-white"
          aria-label={t("curatedTask.badge.action.editAria", {
            title: task.title,
          })}
          title={t("curatedTask.badge.action.editTitle")}
        >
          <PencilLine className="h-3 w-3" />
          <span>{t("curatedTask.badge.action.edit")}</span>
        </button>
      ) : null}
      <button
        type="button"
        onClick={onClear}
        className="ml-0.5 hover:opacity-70"
        aria-label={t("curatedTask.badge.action.clearAria", {
          title: task.title,
        })}
        title={t("curatedTask.badge.action.clearTitle")}
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
};
