import type { AgentI18nKey } from "@/i18n/agentResources";
import type { AgentRuntimeFileCheckpointThreadSummary } from "@/lib/api/agentRuntime/sessionTypes";
import type { AgentThreadItem } from "@/lib/api/agentProtocolCoreTypes";
import type { HarnessSessionState } from "../utils/harnessState";
import type { CodeFixPromptFileChange } from "../utils/codeFixPrompt";
import type { HarnessFileChangeReviewSummary } from "./HarnessStatusPanel";

export interface ReviewableFileChange extends CodeFixPromptFileChange {
  path: string;
  displayName: string;
}

export interface ReviewStatusPresentation {
  key: "failed_output" | "file_review" | "outputs" | "snapshots";
  labelKey: AgentI18nKey;
  className: string;
}

export type ReviewFocusTone = "danger" | "review" | "success" | "snapshot";

export type CodeReviewRunStatus =
  | "idle"
  | "starting"
  | "inProgress"
  | "completed"
  | "failed"
  | "blocked";

export interface LatestReviewBoundary {
  status: "inProgress" | "completed";
  turnId: string;
  review: string;
}

export interface CodeReviewSummaryPresentationInput {
  failedOutputCount: number;
  fileChangeCount: number;
  outputSignalCount: number;
  checkpointCount: number;
  reviewSummary: HarnessFileChangeReviewSummary | null;
}

type HarnessOutputSignal = HarnessSessionState["outputSignals"][number];
type HarnessFileEvent = HarnessSessionState["recentFileEvents"][number];

const OUTPUT_PREVIEW_MAX_CHARS = 220;

export function resolveLatestReviewBoundary(
  threadItems: readonly AgentThreadItem[] | undefined,
): LatestReviewBoundary | null {
  const boundaries = (threadItems ?? []).filter(
    (item): item is Extract<AgentThreadItem, { type: "review_boundary" }> =>
      item.type === "review_boundary",
  );
  const latest = boundaries.reduce<
    Extract<AgentThreadItem, { type: "review_boundary" }> | null
  >((current, candidate) => {
    if (!current) {
      return candidate;
    }
    if (candidate.sequence !== current.sequence) {
      return candidate.sequence > current.sequence ? candidate : current;
    }
    if (candidate.updated_at !== current.updated_at) {
      return candidate.updated_at > current.updated_at ? candidate : current;
    }
    return (candidate.ordinal ?? 0) > (current.ordinal ?? 0)
      ? candidate
      : current;
  }, null);

  if (!latest) {
    return null;
  }
  return {
    status: latest.boundary === "entered" ? "inProgress" : "completed",
    turnId: latest.turn_id,
    review: latest.review.trim(),
  };
}

export function outputMentionsFile(
  signal: HarnessOutputSignal | null,
  file: ReviewableFileChange | null,
): boolean {
  if (!signal || !file) {
    return false;
  }

  const haystack = [
    signal.title,
    signal.summary,
    signal.preview,
    signal.content,
  ]
    .join("\n")
    .toLowerCase();

  return (
    haystack.includes(file.displayName.toLowerCase()) ||
    haystack.includes(file.path.toLowerCase())
  );
}

export function isReviewableFileEvent(event: HarnessFileEvent): boolean {
  if (event.action === "write" || event.action === "edit") {
    return true;
  }
  if (event.action !== "persist") {
    return false;
  }
  return event.kind !== "log" && event.kind !== "offload";
}

export function resolveReviewableFileChanges(
  harnessState: HarnessSessionState,
): ReviewableFileChange[] {
  const changesByPath = new Map<string, ReviewableFileChange>();

  for (const write of harnessState.activeFileWrites) {
    const path = write.path.trim();
    if (path) {
      changesByPath.set(path, {
        path,
        displayName: write.displayName || path,
      });
    }
  }

  for (const event of harnessState.recentFileEvents) {
    const path = event.path.trim();
    if (path && isReviewableFileEvent(event)) {
      changesByPath.set(path, {
        path,
        displayName: event.displayName || path,
      });
    }
  }

  return [...changesByPath.values()];
}

export function rankFileChangesForOutput(
  fileChanges: ReviewableFileChange[],
  signal: HarnessOutputSignal | null,
): ReviewableFileChange[] {
  if (!signal || fileChanges.length < 2) {
    return fileChanges;
  }

  return [...fileChanges].sort((left, right) => {
    const leftMentioned = outputMentionsFile(signal, left);
    const rightMentioned = outputMentionsFile(signal, right);

    if (leftMentioned === rightMentioned) {
      return 0;
    }
    return leftMentioned ? -1 : 1;
  });
}

export function buildOutputPreviewText(
  signal: HarnessOutputSignal | null,
): string | null {
  const raw = signal?.content || signal?.preview || signal?.summary || "";
  const text = raw
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter(Boolean)
    .slice(0, 4)
    .join("\n")
    .trim();

  if (!text) {
    return null;
  }

  if (text.length <= OUTPUT_PREVIEW_MAX_CHARS) {
    return text;
  }

  return `${text.slice(0, OUTPUT_PREVIEW_MAX_CHARS).trimEnd()}...`;
}

export function buildOutputDetailText(
  signal: HarnessOutputSignal | null,
): string | null {
  if (!signal) {
    return null;
  }

  const parts = [signal.title, signal.summary, signal.preview]
    .map((part) => part?.trim())
    .filter((part): part is string => Boolean(part));
  return [...new Set(parts)].slice(0, 2).join(" · ") || null;
}

export function normalizeReviewSummary(
  reviewSummary: HarnessFileChangeReviewSummary | null | undefined,
): HarnessFileChangeReviewSummary | null {
  return reviewSummary?.total ? reviewSummary : null;
}

export function resolveConfirmedFileChangeCount(
  reviewSummary: HarnessFileChangeReviewSummary | null,
): number {
  return reviewSummary ? reviewSummary.applied + reviewSummary.rejected : 0;
}

export function resolvePrimaryActionKey({
  failedOutputCount,
  fileChangeCount,
  outputSignalCount,
  checkpointCount,
}: CodeReviewSummaryPresentationInput): AgentI18nKey {
  if (failedOutputCount > 0 && outputSignalCount > 0) {
    return "agentChat.harness.codeReview.action.viewFailedOutput";
  }
  if (fileChangeCount > 0) {
    return "agentChat.harness.codeReview.action.review";
  }
  if (outputSignalCount > 0) {
    return "agentChat.harness.codeReview.action.viewOutput";
  }
  if (checkpointCount > 0) {
    return "agentChat.harness.codeReview.action.viewSnapshots";
  }
  return "agentChat.harness.codeReview.action.review";
}

export function hasCodeReviewSurface({
  fileChangeCount,
  outputSignalCount,
  checkpointCount,
}: CodeReviewSummaryPresentationInput): boolean {
  return fileChangeCount > 0 || outputSignalCount > 0 || checkpointCount > 0;
}

export function resolveReviewStatusPresentation({
  failedOutputCount,
  fileChangeCount,
  outputSignalCount,
  reviewSummary,
}: CodeReviewSummaryPresentationInput): ReviewStatusPresentation {
  if (failedOutputCount > 0) {
    return {
      key: "failed_output",
      labelKey: "agentChat.harness.codeReview.status.failedOutput",
      className: "border-rose-200 bg-rose-50 text-rose-700",
    };
  }

  if (reviewSummary && reviewSummary.pending === 0) {
    return reviewSummary.rejected > 0
      ? {
          key: "snapshots",
          labelKey: "agentChat.harness.codeReview.status.rollbackReviewed",
          className: "border-amber-200 bg-amber-50 text-amber-700",
        }
      : {
          key: "outputs",
          labelKey: "agentChat.harness.codeReview.status.reviewed",
          className: "border-emerald-200 bg-emerald-50 text-emerald-700",
        };
  }

  if (fileChangeCount > 0) {
    return {
      key: "file_review",
      labelKey: "agentChat.harness.codeReview.status.fileReview",
      className: "border-sky-200 bg-sky-50 text-sky-700",
    };
  }

  if (outputSignalCount > 0) {
    return {
      key: "outputs",
      labelKey: "agentChat.harness.codeReview.status.outputsReady",
      className: "border-emerald-200 bg-emerald-50 text-emerald-700",
    };
  }

  return {
    key: "snapshots",
    labelKey: "agentChat.harness.codeReview.status.snapshotsReady",
    className: "border-amber-200 bg-amber-50 text-amber-700",
  };
}

export function resolveReviewFocusTone({
  failedOutputCount,
  fileChangeCount,
  outputSignalCount,
}: CodeReviewSummaryPresentationInput): ReviewFocusTone {
  if (failedOutputCount > 0) {
    return "danger";
  }
  if (fileChangeCount > 0) {
    return "review";
  }
  if (outputSignalCount > 0) {
    return "success";
  }
  return "snapshot";
}

export function resolveFocusDescriptionKey({
  failedOutputCount,
  fileChangeCount,
  outputSignalCount,
}: CodeReviewSummaryPresentationInput): AgentI18nKey {
  if (failedOutputCount > 0 && fileChangeCount > 0) {
    return "agentChat.harness.codeReview.focus.failedWithFiles";
  }
  if (failedOutputCount > 0) {
    return "agentChat.harness.codeReview.focus.failed";
  }
  if (fileChangeCount > 0) {
    return "agentChat.harness.codeReview.focus.files";
  }
  if (outputSignalCount > 0) {
    return "agentChat.harness.codeReview.focus.outputs";
  }
  return "agentChat.harness.codeReview.focus.snapshots";
}

export function selectLatestCheckpoint(
  fileCheckpointSummary?: AgentRuntimeFileCheckpointThreadSummary | null,
): AgentRuntimeFileCheckpointThreadSummary["latest_checkpoint"] | null {
  return fileCheckpointSummary?.latest_checkpoint || null;
}
