import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  AlertCircle,
  ClipboardCheck,
  FileText,
  GitCompare,
  Loader2,
  RotateCcw,
  TerminalSquare,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import type { AgentThreadItem } from "@/lib/api/agentProtocolCoreTypes";
import type { AgentRuntimeFileCheckpointThreadSummary } from "@/lib/api/agentRuntime/sessionTypes";
import { cn } from "@/lib/utils";
import type { HarnessSessionState } from "../utils/harnessState";
import {
  countFailedHarnessOutputSignals,
  countPassingHarnessOutputSignals,
  resolvePriorityHarnessOutputSignal,
} from "../utils/harnessOutputSignals";
import { buildCodeFixPromptFromHarnessSignal } from "../utils/codeFixPrompt";
import type { CodeWorkbenchGuideTarget } from "./CodeWorkbenchGuide";
import type { HarnessFileChangeReviewSummary } from "./HarnessStatusPanel";
import {
  buildOutputDetailText,
  buildOutputPreviewText,
  hasCodeReviewSurface,
  normalizeReviewSummary,
  outputMentionsFile,
  rankFileChangesForOutput,
  resolveConfirmedFileChangeCount,
  resolveFocusDescriptionKey,
  resolveLatestReviewBoundary,
  resolvePrimaryActionKey,
  resolveReviewFocusTone,
  resolveReviewStatusPresentation,
  resolveReviewableFileChanges,
  selectLatestCheckpoint,
} from "./CodeReviewSummaryPanelViewModel";

export interface CodeReviewSummaryPanelProps {
  harnessState: HarnessSessionState;
  fileCheckpointSummary?: AgentRuntimeFileCheckpointThreadSummary | null;
  fileChangeReviewSummary?: HarnessFileChangeReviewSummary | null;
  onOpenSection: (target: CodeWorkbenchGuideTarget) => void;
  onOpenFileCheckpoints?: () => void;
  onSubmitCodeFixPrompt?: (prompt: string) => void | Promise<void>;
  threadItems?: readonly AgentThreadItem[];
  reviewThreadId?: string | null;
  currentTurnId?: string | null;
  canInterrupt?: boolean;
  onStartReview?: () => Promise<unknown>;
}

export function CodeReviewSummaryPanel({
  harnessState,
  fileCheckpointSummary,
  fileChangeReviewSummary,
  onOpenSection,
  onOpenFileCheckpoints,
  onSubmitCodeFixPrompt,
  threadItems,
  reviewThreadId,
  currentTurnId,
  canInterrupt = false,
  onStartReview,
}: CodeReviewSummaryPanelProps) {
  const { t } = useTranslation("agent");
  const [fixSubmitting, setFixSubmitting] = useState(false);
  const [reviewActionState, setReviewActionState] = useState<
    "idle" | "starting" | "failed"
  >("idle");
  const latestReviewBoundary = useMemo(
    () => resolveLatestReviewBoundary(threadItems),
    [threadItems],
  );
  const fileChanges = useMemo(
    () => resolveReviewableFileChanges(harnessState),
    [harnessState],
  );
  const passingOutputCount = countPassingHarnessOutputSignals(
    harnessState.outputSignals,
  );
  const failedOutputCount = countFailedHarnessOutputSignals(
    harnessState.outputSignals,
  );
  const outputSignalForDetail = resolvePriorityHarnessOutputSignal(
    harnessState.outputSignals,
  );
  const prioritizedFileChanges = useMemo(
    () => rankFileChangesForOutput(fileChanges, outputSignalForDetail),
    [fileChanges, outputSignalForDetail],
  );
  const outputDetail = buildOutputDetailText(outputSignalForDetail);
  const outputPreviewText = buildOutputPreviewText(outputSignalForDetail);
  const checkpointCount = fileCheckpointSummary?.count ?? 0;
  const latestCheckpoint = selectLatestCheckpoint(fileCheckpointSummary);
  const focusOutputLabel =
    outputDetail || t("agentChat.harness.codeReview.focus.outputFallback");
  const visibleFileChanges = prioritizedFileChanges.slice(0, 3);
  const hiddenFileChangeCount = Math.max(
    prioritizedFileChanges.length - visibleFileChanges.length,
    0,
  );
  const focusFileLabel =
    visibleFileChanges[0]?.displayName ||
    t("agentChat.harness.codeReview.focus.fileFallback");
  const shouldShowReviewPair =
    harnessState.outputSignals.length > 0 && visibleFileChanges.length > 0;
  const reviewPairFile = visibleFileChanges[0] || null;
  const reviewPairFileMentioned = outputMentionsFile(
    outputSignalForDetail,
    reviewPairFile,
  );
  const reviewSummary = normalizeReviewSummary(fileChangeReviewSummary);
  const confirmedFileChangeCount =
    resolveConfirmedFileChangeCount(reviewSummary);
  const primaryActionKey = resolvePrimaryActionKey({
    failedOutputCount,
    fileChangeCount: fileChanges.length,
    outputSignalCount: harnessState.outputSignals.length,
    checkpointCount,
    reviewSummary,
  });
  const hasReviewSurface = hasCodeReviewSurface({
    failedOutputCount,
    fileChangeCount: fileChanges.length,
    outputSignalCount: harnessState.outputSignals.length,
    checkpointCount,
    reviewSummary,
  });
  const codeFixPrompt = useMemo(() => {
    if (failedOutputCount === 0 || !outputSignalForDetail) {
      return null;
    }
    return buildCodeFixPromptFromHarnessSignal({
      signal: outputSignalForDetail,
      fileChanges: prioritizedFileChanges,
      fileCheckpointSummary,
      copy: {
        intro: t("agentChat.harness.codeReview.fixPrompt.intro"),
        requirements: t("agentChat.harness.codeReview.fixPrompt.requirements"),
        failedTool: t("agentChat.harness.codeReview.fixPrompt.failedTool"),
        failedTitle: t("agentChat.harness.codeReview.fixPrompt.failedTitle"),
        failedSummary: t(
          "agentChat.harness.codeReview.fixPrompt.failedSummary",
        ),
        failedPreview: t(
          "agentChat.harness.codeReview.fixPrompt.failedPreview",
        ),
        relatedFiles: t("agentChat.harness.codeReview.fixPrompt.relatedFiles"),
        latestCheckpoint: t(
          "agentChat.harness.codeReview.fixPrompt.latestCheckpoint",
        ),
      },
    });
  }, [
    failedOutputCount,
    prioritizedFileChanges,
    fileCheckpointSummary,
    outputSignalForDetail,
    t,
  ]);
  const canSubmitCodeFix = Boolean(onSubmitCodeFixPrompt && codeFixPrompt);
  const reviewStatus = resolveReviewStatusPresentation({
    failedOutputCount,
    fileChangeCount: fileChanges.length,
    outputSignalCount: harnessState.outputSignals.length,
    checkpointCount,
    reviewSummary,
  });
  const focusTone = resolveReviewFocusTone({
    failedOutputCount,
    fileChangeCount: fileChanges.length,
    outputSignalCount: harnessState.outputSignals.length,
    checkpointCount,
    reviewSummary,
  });
  const focusDescriptionKey = resolveFocusDescriptionKey({
    failedOutputCount,
    fileChangeCount: fileChanges.length,
    outputSignalCount: harnessState.outputSignals.length,
    checkpointCount,
    reviewSummary,
  });
  const normalizedReviewThreadId = reviewThreadId?.trim() || "";
  const reviewBlockedReason = !normalizedReviewThreadId
    ? "missingThread"
    : canInterrupt && currentTurnId
      ? "activeTurn"
      : null;
  const reviewRunStatus =
    reviewActionState === "starting"
      ? "starting"
      : reviewActionState === "failed"
        ? "failed"
        : latestReviewBoundary?.status || "idle";
  const reviewStatusClassName =
    reviewBlockedReason
      ? "border-amber-200 bg-amber-50 text-amber-700"
      : reviewRunStatus === "inProgress"
      ? "border-sky-200 bg-sky-50 text-sky-700"
      : reviewRunStatus === "completed"
        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
        : reviewRunStatus === "failed"
          ? "border-rose-200 bg-rose-50 text-rose-700"
          : "border-slate-200 bg-slate-50 text-slate-600";
  const reviewStatusLabelKey =
    reviewBlockedReason === "missingThread"
      ? "agentChat.harness.codeReview.status.blocked"
      : reviewBlockedReason === "activeTurn"
        ? "agentChat.harness.codeReview.status.blocked"
        : reviewRunStatus === "starting"
          ? "agentChat.harness.codeReview.status.starting"
          : reviewRunStatus === "inProgress"
            ? "agentChat.harness.codeReview.status.inProgress"
            : reviewRunStatus === "completed"
              ? "agentChat.harness.codeReview.status.completed"
              : reviewRunStatus === "failed"
                ? "agentChat.harness.codeReview.status.failed"
                : null;
  const reviewActionLabelKey =
    reviewRunStatus === "starting" || reviewRunStatus === "inProgress"
      ? "agentChat.harness.codeReview.action.starting"
      : reviewRunStatus === "completed" || reviewRunStatus === "failed"
        ? "agentChat.harness.codeReview.action.retry"
        : "agentChat.harness.codeReview.action.start";
  const reviewActionDisabled =
    !onStartReview ||
    Boolean(reviewBlockedReason) ||
    reviewRunStatus === "starting" ||
    reviewRunStatus === "inProgress";

  if (!hasReviewSurface) {
    return null;
  }

  return (
    <div
      className="rounded-lg border border-slate-200 bg-white px-3 py-3 text-slate-950"
      data-testid="code-review-summary-panel"
      data-status={reviewStatus.key}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-slate-200 bg-slate-50">
              <GitCompare className="h-4 w-4" />
            </span>
            <div className="text-sm font-semibold">
              {t("agentChat.harness.codeReview.title")}
            </div>
            <span
              className={cn(
                "inline-flex rounded-full border px-2 py-0.5 text-xs font-medium",
                reviewStatus.className,
              )}
              data-testid="code-review-summary-status"
            >
              {t(reviewStatus.labelKey)}
            </span>
          </div>
          <div className="mt-1 text-xs leading-5 text-slate-600">
            {t("agentChat.harness.codeReview.description", {
              files: fileChanges.length,
              outputs: harnessState.outputSignals.length,
              checkpoints: checkpointCount,
            })}
            {reviewSummary ? (
              <span
                className="mt-1 block"
                data-testid="code-review-summary-review-state"
              >
                {t("agentChat.harness.codeReview.reviewState", {
                  pending: reviewSummary.pending,
                  applied: reviewSummary.applied,
                  rejected: reviewSummary.rejected,
                })}
              </span>
            ) : null}
            {onStartReview ? (
              <div
                className="mt-2 flex flex-wrap items-center gap-2"
                data-testid="code-review-summary-review-control"
              >
                {reviewStatusLabelKey ? (
                  <span
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium",
                      reviewStatusClassName,
                    )}
                    data-testid="code-review-summary-review-status"
                  >
                    {reviewRunStatus === "starting" ||
                    reviewRunStatus === "inProgress" ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : reviewRunStatus === "failed" ? (
                      <AlertCircle className="h-3 w-3" />
                    ) : null}
                    {t(reviewStatusLabelKey)}
                  </span>
                ) : null}
                <span className="text-xs text-slate-500">
                  {reviewBlockedReason === "missingThread"
                    ? t("agentChat.harness.codeReview.reviewMissingThread")
                    : reviewBlockedReason === "activeTurn"
                      ? t("agentChat.harness.codeReview.reviewActiveTurn")
                      : t("agentChat.harness.codeReview.reviewTarget")}
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant={reviewRunStatus === "completed" ? "outline" : "default"}
                  disabled={reviewActionDisabled}
                  onClick={async () => {
                    if (reviewActionDisabled || !onStartReview) {
                      return;
                    }
                    setReviewActionState("starting");
                    try {
                      await onStartReview();
                      setReviewActionState("idle");
                    } catch {
                      setReviewActionState("failed");
                    }
                  }}
                  data-testid="code-review-summary-start-review"
                >
                  {reviewRunStatus === "starting" ||
                  reviewRunStatus === "inProgress" ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <ClipboardCheck className="h-3.5 w-3.5" />
                  )}
                  {t(reviewActionLabelKey)}
                </Button>
              </div>
            ) : null}
            {latestReviewBoundary?.status === "completed" &&
            latestReviewBoundary.review ? (
              <div
                className="mt-1 whitespace-pre-wrap break-words text-xs text-slate-600"
                data-testid="code-review-summary-review-result"
              >
                {t("agentChat.harness.codeReview.reviewResult", {
                  review: latestReviewBoundary.review,
                })}
              </div>
            ) : null}
            {reviewRunStatus === "failed" ? (
              <div
                className="mt-1 text-xs text-rose-700"
                data-testid="code-review-summary-review-error"
              >
                {t("agentChat.harness.codeReview.reviewError")}
              </div>
            ) : null}
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap justify-end gap-2">
          {canSubmitCodeFix ? (
            <Button
              type="button"
              size="sm"
              disabled={fixSubmitting}
              onClick={async () => {
                if (!codeFixPrompt || !onSubmitCodeFixPrompt) {
                  return;
                }
                setFixSubmitting(true);
                try {
                  await onSubmitCodeFixPrompt(codeFixPrompt);
                } finally {
                  setFixSubmitting(false);
                }
              }}
              data-testid="code-review-summary-fix-action"
            >
              {fixSubmitting
                ? t("agentChat.harness.codeReview.action.fixFailedOutputBusy")
                : t("agentChat.harness.codeReview.action.fixFailedOutput")}
            </Button>
          ) : null}
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="shrink-0"
            onClick={() => {
              if (
                failedOutputCount > 0 &&
                harnessState.outputSignals.length > 0
              ) {
                onOpenSection("outputs");
                return;
              }
              if (fileChanges.length > 0) {
                onOpenSection("file_review");
                return;
              }
              if (harnessState.outputSignals.length > 0) {
                onOpenSection("outputs");
                return;
              }
              if (checkpointCount > 0 && onOpenFileCheckpoints) {
                onOpenFileCheckpoints?.();
                return;
              }
              onOpenSection("runtime");
            }}
            data-testid="code-review-summary-primary-action"
          >
            {t(primaryActionKey)}
          </Button>
        </div>
      </div>

      <div
        className={cn(
          "mt-3 rounded-lg border px-3 py-2",
          focusTone === "danger"
            ? "border-rose-200 bg-rose-50 text-rose-950"
            : focusTone === "review"
              ? "border-sky-200 bg-sky-50 text-sky-950"
              : focusTone === "success"
                ? "border-emerald-200 bg-emerald-50 text-emerald-950"
                : "border-amber-200 bg-amber-50 text-amber-950",
        )}
        data-testid="code-review-summary-focus"
        data-tone={focusTone}
      >
        <div className="text-[11px] font-medium">
          {t("agentChat.harness.codeReview.focus.title")}
        </div>
        <div className="mt-1 text-xs leading-5 text-current/75">
          {t(focusDescriptionKey, {
            output: focusOutputLabel,
            file: focusFileLabel,
            files: fileChanges.length,
            checkpoints: checkpointCount,
          })}
        </div>
        <div className="mt-2 flex flex-wrap gap-2">
          {harnessState.outputSignals.length > 0 ? (
            <button
              type="button"
              className="inline-flex min-h-8 items-center gap-1.5 rounded-md border border-current/15 bg-white/75 px-2.5 py-1 text-xs font-medium transition-colors hover:bg-white"
              onClick={() => onOpenSection("outputs")}
              data-testid="code-review-summary-focus-output"
            >
              <TerminalSquare className="h-3.5 w-3.5" />
              {t("agentChat.harness.codeReview.focus.openOutput")}
            </button>
          ) : null}
          {fileChanges.length > 0 ? (
            <button
              type="button"
              className="inline-flex min-h-8 items-center gap-1.5 rounded-md border border-current/15 bg-white/75 px-2.5 py-1 text-xs font-medium transition-colors hover:bg-white"
              onClick={() => onOpenSection("file_review")}
              data-testid="code-review-summary-focus-files"
            >
              <FileText className="h-3.5 w-3.5" />
              {t("agentChat.harness.codeReview.focus.openFiles")}
            </button>
          ) : null}
          {checkpointCount > 0 && onOpenFileCheckpoints ? (
            <button
              type="button"
              className="inline-flex min-h-8 items-center gap-1.5 rounded-md border border-current/15 bg-white/75 px-2.5 py-1 text-xs font-medium transition-colors hover:bg-white"
              onClick={onOpenFileCheckpoints}
              data-testid="code-review-summary-focus-snapshots"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              {t("agentChat.harness.codeReview.focus.openSnapshots")}
            </button>
          ) : null}
        </div>
      </div>

      {shouldShowReviewPair ? (
        <div
          className="mt-3 grid min-w-0 gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr))]"
          data-testid="code-review-summary-pair"
        >
          <button
            type="button"
            className={cn(
              "rounded-lg border px-3 py-2 text-left transition-colors",
              failedOutputCount > 0
                ? "border-rose-200 bg-rose-50 hover:bg-rose-100"
                : "border-emerald-200 bg-emerald-50 hover:bg-emerald-100",
            )}
            onClick={() => onOpenSection("outputs")}
            data-testid="code-review-summary-pair-output"
          >
            <div className="flex items-center gap-2 text-xs font-semibold">
              <TerminalSquare className="h-4 w-4" />
              {t(
                failedOutputCount > 0
                  ? "agentChat.harness.codeReview.pair.failedOutput"
                  : "agentChat.harness.codeReview.pair.output",
              )}
            </div>
            <div className="mt-1 truncate text-xs text-current/75">
              {focusOutputLabel}
            </div>
            {outputPreviewText ? (
              <div className="mt-2 max-h-20 overflow-hidden whitespace-pre-wrap break-words rounded-md border border-current/15 bg-white/75 px-2 py-1.5 font-mono text-[11px] leading-5">
                {outputPreviewText}
              </div>
            ) : null}
          </button>
          <button
            type="button"
            className="rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-left text-sky-950 transition-colors hover:bg-sky-100"
            onClick={() => onOpenSection("file_review")}
            data-testid="code-review-summary-pair-file"
          >
            <div className="flex items-center gap-2 text-xs font-semibold">
              <FileText className="h-4 w-4" />
              {t("agentChat.harness.codeReview.pair.file")}
            </div>
            <div className="mt-1 truncate text-xs text-current/75">
              {reviewPairFile?.displayName ||
                t("agentChat.harness.codeReview.focus.fileFallback")}
            </div>
            <div className="mt-2 text-xs leading-5 text-current/65">
              {t(
                reviewPairFileMentioned
                  ? "agentChat.harness.codeReview.pair.fileMentionedHint"
                  : "agentChat.harness.codeReview.pair.fileHint",
                {
                  files: fileChanges.length,
                },
              )}
            </div>
          </button>
        </div>
      ) : null}

      <div className="mt-3 grid min-w-0 gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,11rem),1fr))]">
        <button
          type="button"
          className={cn(
            "rounded-lg border px-3 py-2 text-left transition-colors",
            fileChanges.length > 0
              ? "border-sky-200 bg-sky-50 hover:bg-sky-100"
              : "border-slate-200 bg-slate-50 text-slate-500",
          )}
          disabled={fileChanges.length === 0}
          onClick={() => onOpenSection("file_review")}
          data-testid="code-review-summary-files"
          data-confirmed={String(confirmedFileChangeCount)}
        >
          <div className="flex items-center gap-2 text-xs font-medium">
            <FileText className="h-4 w-4" />
            {t("agentChat.harness.codeReview.metric.files", {
              count: fileChanges.length,
            })}
          </div>
          <div className="mt-1 text-xs leading-5">
            {visibleFileChanges.length > 0
              ? visibleFileChanges.map((item) => item.displayName).join(", ")
              : t("agentChat.harness.codeReview.metric.filesEmpty")}
            {hiddenFileChangeCount > 0 ? (
              <div className="mt-1 text-current/70">
                {t("agentChat.harness.codeReview.metric.filesMore", {
                  count: hiddenFileChangeCount,
                })}
              </div>
            ) : null}
            {reviewSummary ? (
              <div className="mt-1 text-current/70">
                {t("agentChat.harness.codeReview.metric.filesReviewState", {
                  pending: reviewSummary.pending,
                  applied: reviewSummary.applied,
                  rejected: reviewSummary.rejected,
                })}
              </div>
            ) : null}
          </div>
        </button>

        <button
          type="button"
          className={cn(
            "rounded-lg border px-3 py-2 text-left transition-colors",
            failedOutputCount > 0
              ? "border-rose-200 bg-rose-50 hover:bg-rose-100"
              : harnessState.outputSignals.length > 0
                ? "border-emerald-200 bg-emerald-50 hover:bg-emerald-100"
                : "border-slate-200 bg-slate-50 text-slate-500",
          )}
          disabled={harnessState.outputSignals.length === 0}
          onClick={() => onOpenSection("outputs")}
          data-testid="code-review-summary-outputs"
          data-tone={
            failedOutputCount > 0
              ? "danger"
              : harnessState.outputSignals.length > 0
                ? "success"
                : "muted"
          }
        >
          <div className="flex items-center gap-2 text-xs font-medium">
            <TerminalSquare className="h-4 w-4" />
            {t("agentChat.harness.codeReview.metric.outputs", {
              count: harnessState.outputSignals.length,
            })}
          </div>
          <div className="mt-1 text-xs leading-5">
            <div>
              {failedOutputCount > 0
                ? t("agentChat.harness.codeReview.metric.outputsFailed", {
                    count: failedOutputCount,
                  })
                : passingOutputCount > 0
                  ? t("agentChat.harness.codeReview.metric.outputsPassing", {
                      count: passingOutputCount,
                    })
                  : t("agentChat.harness.codeReview.metric.outputsEmpty")}
            </div>
            {outputDetail ? (
              <div className="mt-1 truncate text-current/75">
                {outputDetail}
              </div>
            ) : null}
            {outputPreviewText ? (
              <div
                className={cn(
                  "mt-2 rounded-md border px-2 py-1.5",
                  "max-h-24 overflow-hidden whitespace-pre-wrap break-words font-mono text-[11px] leading-5",
                  failedOutputCount > 0
                    ? "border-rose-200 bg-white/75 text-rose-950"
                    : "border-emerald-200 bg-white/75 text-emerald-950",
                )}
                data-testid="code-review-summary-output-preview"
              >
                <div className="mb-0.5 font-sans text-[10px] font-medium text-current/65">
                  {t(
                    failedOutputCount > 0
                      ? "agentChat.harness.codeReview.metric.outputPreviewFailed"
                      : "agentChat.harness.codeReview.metric.outputPreview",
                  )}
                </div>
                {outputPreviewText}
              </div>
            ) : null}
          </div>
        </button>

        <button
          type="button"
          className={cn(
            "rounded-lg border px-3 py-2 text-left transition-colors",
            checkpointCount > 0
              ? "border-amber-200 bg-amber-50 hover:bg-amber-100"
              : "border-slate-200 bg-slate-50 text-slate-500",
          )}
          disabled={!onOpenFileCheckpoints || checkpointCount === 0}
          onClick={onOpenFileCheckpoints}
          data-testid="code-review-summary-checkpoints"
        >
          <div className="flex items-center gap-2 text-xs font-medium">
            <RotateCcw className="h-4 w-4" />
            {t("agentChat.harness.codeReview.metric.checkpoints", {
              count: checkpointCount,
            })}
          </div>
          <div className="mt-1 text-xs leading-5">
            {latestCheckpoint?.path ||
              t("agentChat.harness.codeReview.metric.checkpointsEmpty")}
          </div>
        </button>
      </div>
    </div>
  );
}
