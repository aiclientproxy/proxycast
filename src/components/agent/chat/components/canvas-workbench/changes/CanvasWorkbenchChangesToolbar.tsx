import {
  Check,
  ClipboardCheck,
  ChevronDown,
  ChevronRight,
  FileDiff,
  FolderOpen,
  GitCompareArrows,
  ListChecks,
  Loader2,
  MoreHorizontal,
  GitBranch,
} from "lucide-react";
import type { CanvasWorkbenchChangeDiffStats } from "./CanvasWorkbenchChangesPanelViewModel";
import { DiffStats } from "./CanvasWorkbenchChangeStats";
import {
  CanvasWorkbenchReviewMenu,
  type CanvasWorkbenchReviewMenuModel,
} from "./CanvasWorkbenchReviewMenu";
import {
  CANVAS_WORKBENCH_REVIEW_BASE_OPTIONS,
  type CanvasWorkbenchReviewBase,
  type CanvasWorkbenchReviewCommitOption,
  type CanvasWorkbenchTranslation,
} from "./CanvasWorkbenchChangesTypes";

export interface CanvasWorkbenchChangesToolbarProps extends CanvasWorkbenchReviewMenuModel {
  translateWorkbench: CanvasWorkbenchTranslation;
  diffStats: CanvasWorkbenchChangeDiffStats;
  checkpointCount?: number;
  reviewMenuOpen: boolean;
  baseMenuOpen: boolean;
  selectedBase: CanvasWorkbenchReviewBase;
  branchCompareLabel?: string | null;
  filesPanelOpen?: boolean;
  showFilesToggle?: boolean;
  actionsDisabled?: boolean;
  baseSelectorDisabled?: boolean;
  reviewMenuDisabled?: boolean;
  diffViewToggleDisabled?: boolean;
  reviewActionBusy?: boolean;
  copyGitApplyDisabled?: boolean;
  loadFullFileDisabled?: boolean;
  commitOptions?: CanvasWorkbenchReviewCommitOption[];
  commitsLoading?: boolean;
  selectedCommitSha?: string | null;
  diffVariant: "inline" | "split";
  onToggleReviewMenu: () => void;
  onToggleBaseMenu: () => void;
  onSelectBase: (base: CanvasWorkbenchReviewBase) => void;
  onOpenCommitMenu?: () => void | Promise<void>;
  onSelectCommit?: (commit: CanvasWorkbenchReviewCommitOption) => void;
  onRefreshChanges?: () => void | Promise<void>;
  onCopyGitApply?: () => void | Promise<void>;
  onToggleDiffVariant?: () => void;
  onToggleFilesPanel?: () => void;
  reviewRunStatus?: "idle" | "starting" | "inProgress" | "completed" | "failed";
  reviewStatusLabelKey?: string | null;
  reviewStartDisabled?: boolean;
  onStartReview?: () => void | Promise<void>;
}

export function CanvasWorkbenchChangesToolbar({
  translateWorkbench,
  diffStats,
  checkpointCount,
  reviewMenuOpen,
  baseMenuOpen,
  selectedBase,
  branchCompareLabel,
  filesPanelOpen = true,
  showFilesToggle = true,
  actionsDisabled = false,
  baseSelectorDisabled,
  reviewMenuDisabled,
  diffViewToggleDisabled,
  reviewActionBusy = false,
  copyGitApplyDisabled = false,
  loadFullFileDisabled = false,
  commitOptions = [],
  commitsLoading = false,
  selectedCommitSha = null,
  autoExecuteEnabled,
  collapseDiffContext,
  loadFullFile,
  richPreviewEnabled,
  textDiffEnabled,
  showWhitespace,
  diffVariant,
  onToggleReviewMenu,
  onToggleBaseMenu,
  onSelectBase,
  onOpenCommitMenu,
  onSelectCommit,
  onRefreshChanges,
  onCopyGitApply,
  onToggleAutoExecute,
  onToggleCollapseContext,
  onToggleLoadFullFile,
  onToggleRichPreview,
  onToggleTextDiff,
  onToggleWhitespace,
  onToggleDiffVariant,
  onToggleFilesPanel,
  reviewRunStatus = "idle",
  reviewStatusLabelKey,
  reviewStartDisabled = true,
  onStartReview,
}: CanvasWorkbenchChangesToolbarProps) {
  const filesPanelToggleLabel = translateWorkbench(
    filesPanelOpen
      ? "agentChat.canvasWorkbench.coding.changes.hideFiles"
      : "agentChat.canvasWorkbench.coding.changes.showFiles",
  );
  const diffViewToggleLabel = translateWorkbench(
    diffVariant === "inline"
      ? "agentChat.canvasWorkbench.coding.changes.switchToSplitDiff"
      : "agentChat.canvasWorkbench.coding.changes.switchToInlineDiff",
  );
  const selectedBaseLabel = translateWorkbench(
    CANVAS_WORKBENCH_REVIEW_BASE_OPTIONS.find(
      (option) => option.key === selectedBase,
    )?.labelKey ||
      "agentChat.canvasWorkbench.coding.changes.base.previousConversation",
  );
  const isBaseSelectorDisabled = baseSelectorDisabled ?? actionsDisabled;
  const isReviewMenuDisabled = reviewMenuDisabled ?? actionsDisabled;
  const isDiffViewToggleDisabled =
    diffViewToggleDisabled ?? (actionsDisabled || !onToggleDiffVariant);
  const toolbarButtonClassName =
    "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[7px] border border-transparent bg-white text-slate-500 transition-colors hover:border-slate-200 hover:bg-slate-50 hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300";
  const activeToolbarButtonClassName =
      "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[7px] border border-slate-200 bg-slate-50 text-slate-700 transition-colors hover:border-slate-300 hover:bg-white hover:text-slate-900 disabled:cursor-not-allowed disabled:text-slate-300";
  const reviewButtonLabelKey =
    reviewRunStatus === "completed" || reviewRunStatus === "failed"
      ? "agentChat.harness.codeReview.action.retry"
      : reviewStatusLabelKey || "agentChat.harness.codeReview.action.start";

  return (
    <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_auto] gap-x-2 gap-y-1 border-b border-slate-200 bg-white px-3 py-2">
      <div className="flex min-w-0 items-center gap-2">
        <div className="relative">
          <button
            type="button"
            aria-label={translateWorkbench(
              "agentChat.canvasWorkbench.coding.changes.base.selectorAria",
            )}
            aria-haspopup="menu"
            aria-expanded={baseMenuOpen}
            className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[7px] border border-slate-200 bg-slate-50 px-2.5 text-[13px] font-medium text-slate-900 transition-colors hover:border-slate-300 hover:bg-white disabled:opacity-70"
            disabled={isBaseSelectorDisabled}
            onClick={onToggleBaseMenu}
          >
            <span className="truncate">{selectedBaseLabel}</span>
            <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
          </button>
          {baseMenuOpen && !isBaseSelectorDisabled ? (
            <div
              role="menu"
              data-testid="canvas-workbench-changes-base-menu"
              className="absolute left-0 top-[calc(100%+6px)] z-[80] w-[202px] rounded-[12px] border border-slate-200/90 bg-white p-1.5 shadow-xl shadow-slate-950/10"
            >
              {CANVAS_WORKBENCH_REVIEW_BASE_OPTIONS.map((option) => {
                const active = option.key === selectedBase;
                const isCommit = option.key === "commit";
                const selectedCommit = commitOptions.find(
                  (commit) => commit.sha === selectedCommitSha,
                );
                return (
                  <div key={option.key} className="relative">
                    <button
                      type="button"
                      role="menuitemradio"
                      aria-checked={active}
                      data-testid={`canvas-workbench-changes-base-option-${option.key}`}
                      onClick={() => {
                        if (isCommit) {
                          void onOpenCommitMenu?.();
                        } else {
                          onSelectBase(option.key);
                        }
                      }}
                      className={
                        "flex h-8 w-full items-center justify-between gap-2 rounded-[7px] px-2 text-left text-[13px] font-medium text-slate-700 transition-colors hover:bg-slate-50 hover:text-slate-950"
                      }
                    >
                      <span className="min-w-0 truncate">
                        {translateWorkbench(option.labelKey)}
                      </span>
                      {isCommit ? (
                        <ChevronRight className="h-3.5 w-3.5 shrink-0 text-slate-400" />
                      ) : active ? (
                        <Check className="h-3.5 w-3.5 shrink-0 text-slate-600" />
                      ) : null}
                    </button>
                    {isCommit ? (
                      <div
                        role="menu"
                        data-testid="canvas-workbench-changes-base-commit-submenu"
                        className="absolute left-[calc(100%+6px)] top-0 z-[90] max-h-[320px] w-[280px] overflow-y-auto rounded-[10px] border border-slate-200/90 bg-white p-1.5 shadow-xl shadow-slate-950/10"
                      >
                        {commitsLoading ? (
                          <div className="px-2 py-2 text-[13px] font-medium text-slate-400">
                            {translateWorkbench(
                              "agentChat.canvasWorkbench.coding.changes.base.loadingCommits",
                            )}
                          </div>
                        ) : commitOptions.length > 0 ? (
                          commitOptions.map((commit) => {
                            const commitActive =
                              commit.sha === selectedCommit?.sha;
                            return (
                              <button
                                key={commit.sha}
                                type="button"
                                role="menuitemradio"
                                aria-checked={commitActive}
                                data-testid={`canvas-workbench-changes-base-commit-option-${commit.shortSha}`}
                                onClick={() => onSelectCommit?.(commit)}
                                className="flex w-full items-start justify-between gap-2 rounded-[7px] px-2 py-1.5 text-left transition-colors hover:bg-slate-50"
                              >
                                <span className="min-w-0">
                                  <span className="block truncate text-[13px] font-medium text-slate-800">
                                    {commit.subject ||
                                      translateWorkbench(
                                        "agentChat.canvasWorkbench.coding.changes.base.untitledCommit",
                                      )}
                                  </span>
                                  <span className="block truncate font-mono text-[11px] text-slate-400">
                                    {commit.shortSha}
                                  </span>
                                </span>
                                {commitActive ? (
                                  <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-600" />
                                ) : null}
                              </button>
                            );
                          })
                        ) : (
                          <div className="px-2 py-2 text-[13px] font-medium text-slate-400">
                            {translateWorkbench(
                              "agentChat.canvasWorkbench.coding.changes.base.emptyCommits",
                            )}
                          </div>
                        )}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
        <span
          className="inline-flex items-center gap-1 font-mono text-[13px] font-semibold"
          data-testid="canvas-workbench-changes-diff-stats"
        >
          <DiffStats stats={diffStats} />
        </span>
        {checkpointCount ? (
          <span
            className="inline-flex items-center gap-1.5 rounded-[7px] border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700"
            data-testid="canvas-workbench-changes-checkpoints"
          >
            <ListChecks className="h-3.5 w-3.5" />
            {translateWorkbench(
              "agentChat.canvasWorkbench.coding.changes.checkpointBadge",
              { count: checkpointCount },
            )}
          </span>
        ) : null}
      </div>

      <div className="flex shrink-0 items-center gap-1.5">
        {onStartReview ? (
          <button
            type="button"
            data-testid="code-review-summary-start-review"
            aria-label={
              translateWorkbench(reviewButtonLabelKey)
            }
            title={
              translateWorkbench(reviewButtonLabelKey)
            }
            disabled={reviewStartDisabled}
            onClick={() => void onStartReview()}
            className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[7px] border border-sky-200 bg-sky-50 px-2.5 text-[12px] font-medium text-sky-700 transition-colors hover:border-sky-300 hover:bg-sky-100 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-50 disabled:text-slate-300"
          >
            {reviewRunStatus === "starting" || reviewRunStatus === "inProgress" ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <ClipboardCheck className="h-3.5 w-3.5" />
            )}
            <span className="hidden lg:inline">
              {translateWorkbench(reviewButtonLabelKey)}
            </span>
          </button>
        ) : null}
        {reviewStatusLabelKey && reviewRunStatus !== "idle" ? (
          <span
            className="hidden rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] font-medium text-slate-600 xl:inline"
            data-testid="code-review-summary-review-status"
          >
            {translateWorkbench(reviewStatusLabelKey)}
          </span>
        ) : null}
        <div className="relative">
          <button
            type="button"
            aria-label={translateWorkbench(
              "agentChat.canvasWorkbench.coding.changes.more",
            )}
            title={translateWorkbench(
              "agentChat.canvasWorkbench.coding.changes.more",
            )}
            disabled={isReviewMenuDisabled}
            onClick={onToggleReviewMenu}
            className={toolbarButtonClassName}
          >
            <MoreHorizontal className="h-4 w-4" />
          </button>
          {reviewMenuOpen && !isReviewMenuDisabled ? (
            <CanvasWorkbenchReviewMenu
              translateWorkbench={translateWorkbench}
              reviewActionBusy={reviewActionBusy}
              copyGitApplyDisabled={copyGitApplyDisabled}
              loadFullFileDisabled={loadFullFileDisabled}
              autoExecuteEnabled={autoExecuteEnabled}
              collapseDiffContext={collapseDiffContext}
              loadFullFile={loadFullFile}
              richPreviewEnabled={richPreviewEnabled}
              textDiffEnabled={textDiffEnabled}
              showWhitespace={showWhitespace}
              onRefreshChanges={onRefreshChanges}
              onCopyGitApply={onCopyGitApply}
              onToggleAutoExecute={onToggleAutoExecute}
              onToggleCollapseContext={onToggleCollapseContext}
              onToggleLoadFullFile={
                loadFullFileDisabled ? undefined : onToggleLoadFullFile
              }
              onToggleRichPreview={onToggleRichPreview}
              onToggleTextDiff={onToggleTextDiff}
              onToggleWhitespace={onToggleWhitespace}
              onRequestClose={onToggleReviewMenu}
              className="absolute right-0 z-[80] mt-2"
            />
          ) : null}
        </div>
        <button
          type="button"
          aria-label={diffViewToggleLabel}
          title={diffViewToggleLabel}
          disabled={isDiffViewToggleDisabled}
          onClick={onToggleDiffVariant}
          className={
            diffVariant === "split"
              ? activeToolbarButtonClassName
              : toolbarButtonClassName
          }
        >
          {diffVariant === "split" ? (
            <GitCompareArrows className="h-4 w-4" />
          ) : (
            <FileDiff className="h-4 w-4" />
          )}
        </button>
        {showFilesToggle ? (
          <button
            type="button"
            data-testid="canvas-workbench-changes-files-toggle"
            aria-label={filesPanelToggleLabel}
            aria-expanded={filesPanelOpen}
            title={filesPanelToggleLabel}
            disabled={!onToggleFilesPanel}
            onClick={onToggleFilesPanel}
            className={
              filesPanelOpen
                ? activeToolbarButtonClassName
                : toolbarButtonClassName
            }
          >
            <FolderOpen className="h-4 w-4" />
          </button>
        ) : null}
        <button
          type="button"
          aria-label={translateWorkbench(
            "agentChat.canvasWorkbench.coding.changes.submit",
          )}
          title={translateWorkbench(
            "agentChat.canvasWorkbench.coding.changes.submit",
          )}
          disabled={copyGitApplyDisabled || reviewActionBusy || !onCopyGitApply}
          onClick={onCopyGitApply}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-[7px] border border-slate-200 bg-white px-2.5 text-[12px] font-medium text-slate-700 transition-colors hover:border-slate-300 hover:bg-slate-50 hover:text-slate-950 disabled:cursor-not-allowed disabled:text-slate-300"
        >
          <GitCompareArrows className="h-3.5 w-3.5" />
          <span className="hidden lg:inline">
            {translateWorkbench(
              "agentChat.canvasWorkbench.coding.changes.submit",
            )}
          </span>
        </button>
      </div>
      {selectedBase === "branch" && branchCompareLabel ? (
        <div
          className="col-span-2 flex min-w-0 items-center gap-1.5 pl-1 font-mono text-[12px] text-slate-500"
          data-testid="canvas-workbench-changes-branch-compare"
        >
          <GitBranch className="h-3.5 w-3.5 shrink-0 text-slate-400" />
          <span className="min-w-0 truncate">{branchCompareLabel}</span>
        </div>
      ) : null}
    </div>
  );
}
