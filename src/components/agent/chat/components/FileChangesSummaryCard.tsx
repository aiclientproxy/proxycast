import React, { useState } from "react";
import {
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileDiff,
  FileText,
  Loader2,
  Pencil,
  RotateCcw,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import type {
  FileChangesAggregate,
  FileChangeSummary,
} from "../utils/fileChangeSummary";
import {
  buildDiffFileCanvasContent,
  type DiffReviewFile,
  type DiffReviewLine,
} from "../utils/diffReview";

interface FileChangesSummaryCardProps {
  aggregate: FileChangesAggregate;
  variant?: "default" | "timeline";
  isStreaming?: boolean;
  onFileClick?: (path: string, content: string) => void;
  onOpenFile?: (file: FileChangeSummary) => void;
  onUndo?: () =>
    | void
    | { restoredCount?: number }
    | Promise<void | { restoredCount?: number }>;
}

const COLLAPSED_FILE_COUNT = 3;
type UndoState = "idle" | "confirming" | "loading" | "success" | "error";
const UNDO_ERROR_CODES = new Set([
  "emptyAggregate",
  "missingSession",
  "noMatchingCheckpoints",
]);
type AgentNamespaceTranslation = (key: string) => string;

function resolveDisplayPath(path: string): string {
  const normalized = path.replace(/\\/g, "/").trim();
  const limeIndex = normalized.indexOf("/.lime/");
  if (limeIndex >= 0) {
    return normalized.slice(limeIndex + 1);
  }
  const srcIndex = normalized.lastIndexOf("/src/");
  if (srcIndex >= 0) {
    return normalized.slice(srcIndex + 1);
  }
  const internalIndex = normalized.lastIndexOf("/internal/");
  if (internalIndex >= 0) {
    return normalized.slice(internalIndex + 1);
  }
  return normalized || path;
}

function resolveDisplayFilePath(file: FileChangeSummary): string {
  const path = resolveDisplayPath(file.path);
  return file.movePath
    ? `${path} -> ${resolveDisplayPath(file.movePath)}`
    : path;
}

function resolveDiffStatus(file: FileChangeSummary): DiffReviewFile["status"] {
  if (file.kind === "add") return "added";
  if (file.kind === "delete") return "deleted";
  if (file.kind === "update") return "modified";
  return "unknown";
}

function buildDiffReviewFile(file: FileChangeSummary): DiffReviewFile {
  const lines: DiffReviewLine[] = file.diff.map((line) => ({
    kind: line.kind,
    text: line.value,
    ...(line.oldLine !== undefined ? { oldLine: line.oldLine } : {}),
    ...(line.newLine !== undefined ? { newLine: line.newLine } : {}),
  }));

  return {
    id: file.path,
    path: file.path,
    status: resolveDiffStatus(file),
    additions: file.linesAdded,
    deletions: file.linesRemoved,
    hunks: lines.length > 0 ? 1 : 0,
    previewLines: lines.slice(0, 8),
    lines,
  };
}

function resolveUndoErrorMessage(
  error: unknown,
  t: AgentNamespaceTranslation,
): string {
  const code =
    error &&
    typeof error === "object" &&
    "code" in error &&
    typeof error.code === "string" &&
    UNDO_ERROR_CODES.has(error.code)
      ? error.code
      : null;
  if (code) {
    return t(`agentChat.fileChangesSummary.undoError.${code}`);
  }

  return error instanceof Error
    ? error.message
    : t("agentChat.fileChangesSummary.undoError.unknown");
}

export function FileChangesSummaryCard({
  aggregate,
  variant = "default",
  isStreaming = false,
  onFileClick,
  onOpenFile,
  onUndo,
}: FileChangesSummaryCardProps) {
  const { t } = useTranslation("agent");
  const { files, totalAdded, totalRemoved, fileCount } = aggregate;
  const [isFileListExpanded, setIsFileListExpanded] = useState(
    variant === "timeline",
  );
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [undoState, setUndoState] = useState<UndoState>("idle");
  const [undoError, setUndoError] = useState<string | null>(null);
  const [restoredCount, setRestoredCount] = useState<number | null>(null);

  const visibleFiles = isFileListExpanded
    ? files
    : files.slice(0, COLLAPSED_FILE_COUNT);
  const firstFilePath = files[0]?.path;
  const hasCollapsibleFileList = files.length > COLLAPSED_FILE_COUNT;
  const canUndo = Boolean(onUndo) && !isStreaming && undoState !== "success";
  const primaryOpenLabel = onOpenFile
    ? t("agentChat.fileChangesSummary.openFile")
    : t("agentChat.fileChangesSummary.review");

  const openFileReview = (file: FileChangeSummary) => {
    setSelectedPath(file.path);
    if (onOpenFile) {
      onOpenFile(file);
      return;
    }
    const reviewFile = buildDiffReviewFile(file);
    const content = buildDiffFileCanvasContent({
      file: reviewFile,
      title: t("agentChat.fileChangesSummary.reviewCanvasTitle", {
        path: file.path,
      }),
      statusLabel: t("agentChat.fileChangesSummary.reviewCanvasStatus", {
        status: t(
          `agentChat.fileChangesSummary.reviewStatus.${reviewFile.status}`,
        ),
      }),
      additionsLabel: t("agentChat.fileChangesSummary.reviewAdditions", {
        count: reviewFile.additions,
      }),
      deletionsLabel: t("agentChat.fileChangesSummary.reviewDeletions", {
        count: reviewFile.deletions,
      }),
      hunksLabel: t("agentChat.fileChangesSummary.reviewHunks", {
        count: reviewFile.hunks,
      }),
    });

    onFileClick?.(file.path, content);
  };

  if (variant === "timeline") {
    return (
      <div
        data-testid="file-changes-summary-card"
        data-file-status={files[0]?.fileStatus}
        className="space-y-0.5 py-1 text-slate-500"
      >
        <button
          type="button"
          data-testid="file-changes-summary-timeline-toggle"
          aria-expanded={isFileListExpanded}
          className="flex w-full items-center gap-2 rounded-md px-1 py-1 text-left text-sm transition-colors hover:bg-slate-50 hover:text-slate-700"
          onClick={() => setIsFileListExpanded((current) => !current)}
        >
          <Pencil className="h-3.5 w-3.5 shrink-0 text-slate-400" aria-hidden="true" />
          <span className="min-w-0 flex-1 truncate">
            {t("agentChat.messageList.historicalTimeline.fileChangesTitle")}
            <span className="sr-only">
              {t("agentChat.fileChangesSummary.summary", { count: fileCount })}
              {` +${totalAdded} -${totalRemoved}`}
            </span>
          </span>
          {isFileListExpanded ? (
            <ChevronUp className="h-3.5 w-3.5 shrink-0 text-slate-400" aria-hidden="true" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5 shrink-0 text-slate-400" aria-hidden="true" />
          )}
        </button>
        {isFileListExpanded ? (
          <div className="space-y-0.5 pl-1">
            {visibleFiles.map((file) => {
              const statusKey =
                file.kind === "add"
                  ? "agentChat.toolCall.actionOverride.create.completed"
                  : file.kind === "delete"
                    ? "agentChat.toolCall.actionOverride.delete.completed"
                    : "agentChat.toolCall.action.edit.completed";
              const isSelected = file.path === selectedPath;
              return (
                <button
                  key={file.path}
                  type="button"
                  data-testid="file-changes-summary-file-row"
                  data-file-status={file.fileStatus}
                  aria-expanded={isSelected}
                  className={cn(
                    "flex w-full min-w-0 items-center gap-2 rounded-md px-1 py-0.5 text-left text-xs leading-5 transition-colors hover:bg-slate-50",
                    isSelected && "bg-slate-50",
                  )}
                  onClick={() => openFileReview(file)}
                  title={
                    file.movePath
                      ? `${file.path} -> ${file.movePath}`
                      : file.path
                  }
                >
                  <Pencil
                    className="h-3 w-3 shrink-0 text-slate-400"
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1 truncate text-slate-500">
                    <span>{t(statusKey)}</span>{" "}
                    <span className="underline decoration-slate-300 underline-offset-2">
                      {resolveDisplayFilePath(file)}
                    </span>
                  </span>
                  <span className="flex shrink-0 items-center gap-1 font-mono text-[11px] text-slate-400">
                    <span className="text-emerald-600">+{file.linesAdded}</span>
                    <span className="text-red-500">-{file.linesRemoved}</span>
                    <span
                      className="h-1.5 w-1.5 rounded-full bg-sky-500"
                      aria-hidden="true"
                    />
                  </span>
                </button>
              );
            })}
          </div>
        ) : null}
        {!isStreaming && hasCollapsibleFileList && isFileListExpanded ? (
          <button
            type="button"
            data-testid="file-changes-summary-toggle"
            aria-expanded={isFileListExpanded}
            className="flex items-center gap-1 px-1 py-1 text-xs font-medium text-slate-500 hover:text-slate-700"
            onClick={() => setIsFileListExpanded(false)}
          >
            {t("agentChat.fileChangesSummary.collapseFiles")}
            <ChevronUp className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        ) : null}
      </div>
    );
  }

  const requestUndo = () => {
    if (!canUndo) {
      return;
    }
    setUndoError(null);
    setUndoState("confirming");
  };

  const confirmUndo = async () => {
    if (!onUndo || undoState === "loading") {
      return;
    }
    setUndoError(null);
    setUndoState("loading");

    try {
      const result = await onUndo();
      setRestoredCount(result?.restoredCount ?? fileCount);
      setUndoState("success");
    } catch (error) {
      setUndoError(
        resolveUndoErrorMessage(
          error,
          t as unknown as AgentNamespaceTranslation,
        ),
      );
      setUndoState("error");
    }
  };

  return (
    <div
      data-testid="file-changes-summary-card"
      data-file-status={files[0]?.fileStatus}
      className="overflow-hidden rounded-lg border border-slate-200 bg-white text-slate-900"
    >
      <div
        data-testid="file-changes-summary-card-header"
        className="flex min-h-16 items-center justify-between gap-3 border-b border-slate-200 px-3 py-2.5"
      >
        <div className="flex min-w-0 items-center gap-2">
          <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-slate-50 text-slate-600">
            <FileDiff className="h-4 w-4" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold leading-5 text-slate-900">
              {isStreaming
                ? t("agentChat.fileChangesSummary.writing")
                : t("agentChat.fileChangesSummary.summary", {
                    count: fileCount,
                  })}
            </div>
            {!isStreaming ? (
              <div className="mt-0.5 font-mono text-[12px] leading-4">
                <span className="text-emerald-600">+{totalAdded}</span>
                <span className="ml-2 text-red-500">-{totalRemoved}</span>
              </div>
            ) : null}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            disabled={!canUndo || undoState === "loading"}
            title={
              canUndo
                ? t("agentChat.fileChangesSummary.undo")
                : t("agentChat.fileChangesSummary.undoUnavailable")
            }
            className={cn(
              "inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-[13px] font-medium transition-colors",
              canUndo
                ? "text-slate-700 hover:bg-slate-50 hover:text-slate-900"
                : "text-slate-400",
            )}
            onClick={requestUndo}
          >
            {t("agentChat.fileChangesSummary.undo")}
            {undoState === "loading" ? (
              <Loader2
                className="h-3.5 w-3.5 animate-spin"
                aria-hidden="true"
              />
            ) : (
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            )}
          </button>
          {!isStreaming && firstFilePath ? (
            <button
              type="button"
              className="inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-[13px] font-medium text-slate-900 transition-colors hover:border-slate-300 hover:bg-slate-50"
              onClick={() => {
                const firstFile = files.find(
                  (file) => file.path === firstFilePath,
                );
                if (firstFile) {
                  openFileReview(firstFile);
                }
              }}
            >
              {onOpenFile ? (
                <FileText className="h-3.5 w-3.5" aria-hidden="true" />
              ) : (
                <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              {primaryOpenLabel}
            </button>
          ) : null}
        </div>
      </div>

      {undoState === "confirming" ? (
        <div
          data-testid="file-changes-summary-undo-confirmation"
          className="border-b border-amber-200 bg-amber-50 px-3 py-3 text-sm leading-6 text-amber-900"
        >
          <div className="font-medium">
            {t("agentChat.fileChangesSummary.undoConfirmTitle")}
          </div>
          <div className="mt-1">
            {t("agentChat.fileChangesSummary.undoConfirmDescription", {
              count: fileCount,
            })}
          </div>
          <div className="mt-3 flex flex-wrap justify-end gap-2">
            <button
              type="button"
              className="h-8 rounded-md border border-amber-200 bg-white px-2.5 text-[13px] font-medium text-slate-700 transition hover:bg-amber-100"
              onClick={() => setUndoState("idle")}
            >
              {t("agentChat.fileChangesSummary.undoCancel")}
            </button>
            <button
              type="button"
              data-testid="file-changes-summary-undo-confirm"
              className="h-8 rounded-md border border-red-200 bg-red-600 px-2.5 text-[13px] font-medium text-white transition hover:bg-red-700"
              onClick={() => {
                void confirmUndo();
              }}
            >
              {t("agentChat.fileChangesSummary.undoConfirmAction")}
            </button>
          </div>
        </div>
      ) : undoState === "loading" ? (
        <div
          data-testid="file-changes-summary-undo-status"
          className="border-b border-slate-200 bg-slate-50 px-3 py-2 text-[13px] leading-5 text-slate-600"
        >
          {t("agentChat.fileChangesSummary.undoRestoring")}
        </div>
      ) : undoState === "success" ? (
        <div
          data-testid="file-changes-summary-undo-status"
          className="border-b border-emerald-200 bg-emerald-50 px-3 py-2 text-[13px] leading-5 text-emerald-700"
        >
          {t("agentChat.fileChangesSummary.undoSuccess", {
            count: restoredCount ?? fileCount,
          })}
        </div>
      ) : undoState === "error" ? (
        <div
          data-testid="file-changes-summary-undo-status"
          className="border-b border-red-200 bg-red-50 px-3 py-2 text-[13px] leading-5 text-red-700"
        >
          {t("agentChat.fileChangesSummary.undoFailed", {
            error:
              undoError || t("agentChat.fileChangesSummary.undoError.unknown"),
          })}
        </div>
      ) : null}

      {!isStreaming && visibleFiles.length > 0 ? (
        <div className="divide-y divide-slate-100">
          {visibleFiles.map((file) => {
            const isSelected = file.path === selectedPath;
            return (
              <button
                key={file.path}
                type="button"
                data-testid="file-changes-summary-file-row"
                data-file-status={file.fileStatus}
                aria-expanded={isSelected}
                className={cn(
                  "grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-slate-50",
                  isSelected && "bg-slate-50",
                )}
                onClick={() => openFileReview(file)}
                title={
                  file.movePath ? `${file.path} -> ${file.movePath}` : file.path
                }
              >
                <span className="min-w-0 truncate text-[13px] leading-5 text-slate-800">
                  {resolveDisplayFilePath(file)}
                </span>
                <span className="shrink-0 text-[12px] leading-5">
                  <span className="text-emerald-600">+{file.linesAdded}</span>
                  <span className="ml-2 text-red-500">
                    -{file.linesRemoved}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      ) : null}

      {!isStreaming && hasCollapsibleFileList ? (
        <button
          type="button"
          data-testid="file-changes-summary-toggle"
          aria-expanded={isFileListExpanded}
          className="flex w-full items-center justify-start gap-1 border-t border-slate-100 px-3 py-2 text-[13px] font-medium text-slate-700 transition-colors hover:bg-slate-50 hover:text-slate-900"
          onClick={() => setIsFileListExpanded((current) => !current)}
        >
          {isFileListExpanded
            ? t("agentChat.fileChangesSummary.collapseFiles")
            : t("agentChat.fileChangesSummary.expandFiles", {
                count: files.length - COLLAPSED_FILE_COUNT,
              })}
          {isFileListExpanded ? (
            <ChevronUp className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
          )}
        </button>
      ) : null}
    </div>
  );
}
