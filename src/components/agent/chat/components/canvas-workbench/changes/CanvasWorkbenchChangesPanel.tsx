import { useEffect, useMemo, useRef, useState } from "react";
import { collapseCanvasWorkbenchDiffContext } from "../../../utils/canvasWorkbenchDiff";
import type { HarnessFilePreviewResult } from "../../HarnessStatusPanel";
import { resolveLatestReviewBoundary } from "../../CodeReviewSummaryPanelViewModel";
import type { CanvasWorkbenchDiffLine } from "../../../utils/canvasWorkbenchDiff";
import type { CanvasWorkbenchResolvedSelection } from "../../CanvasWorkbenchLayoutViewModel";
import {
  buildCanvasWorkbenchChangeFileTree,
  buildCanvasWorkbenchChangeDiffLines,
  countCanvasWorkbenchChangeItemsStats,
  countCanvasWorkbenchDiffStats,
  findChangeItemForSelection,
  flattenCanvasWorkbenchChangeFileTree,
  resolveCanvasWorkbenchSelectedChangeItem,
} from "./CanvasWorkbenchChangesPanelViewModel";
import type {
  CanvasWorkbenchChangeItem,
  CanvasWorkbenchChangeView,
} from "./CanvasWorkbenchChangesPanelViewModel";
import { CanvasWorkbenchChangeDetailPanel } from "./CanvasWorkbenchChangeDetailPanel";
import { CanvasWorkbenchChangeDiffList } from "./CanvasWorkbenchChangeDiffList";
import type { CanvasWorkbenchChangesToolbarProps } from "./CanvasWorkbenchChangesToolbar";
import { CanvasWorkbenchEmptyDiffPanel } from "./CanvasWorkbenchEmptyDiffPanel";
import { CanvasWorkbenchReviewSurface } from "./CanvasWorkbenchReviewSurface";
import { useCanvasWorkbenchChangesFilesPanelResize } from "./useCanvasWorkbenchChangesFilesPanelResize";
import { useCanvasWorkbenchReviewState } from "./useCanvasWorkbenchReviewState";
import type { CanvasWorkbenchTranslation } from "./CanvasWorkbenchChangesTypes";

interface CanvasWorkbenchChangesPanelProps {
  changeView: CanvasWorkbenchChangeView | null | undefined;
  documentContext: CanvasWorkbenchResolvedSelection | null;
  documentDiffLines: CanvasWorkbenchDiffLine[];
  translateWorkbench: CanvasWorkbenchTranslation;
  panelClassName: string;
  mutedPanelClassName: string;
  workspaceRoot?: string | null;
  loadFilePreview?: (path: string) => Promise<HarnessFilePreviewResult>;
  filesPanelOpen?: boolean;
  onToggleFilesPanel?: () => void;
}

export function CanvasWorkbenchChangesPanel({
  changeView,
  documentContext,
  documentDiffLines,
  translateWorkbench,
  panelClassName,
  mutedPanelClassName,
  workspaceRoot,
  loadFilePreview,
  filesPanelOpen = true,
  onToggleFilesPanel,
}: CanvasWorkbenchChangesPanelProps) {
  const [selectedChangeId, setSelectedChangeId] = useState<string | null>(null);
  const [reviewMenuOpen, setReviewMenuOpen] = useState(false);
  const [baseMenuOpen, setBaseMenuOpen] = useState(false);
  const [collapseDiffContext, setCollapseDiffContext] = useState(false);
  const [showWhitespace, setShowWhitespace] = useState(false);
  const wordWrapEnabled = true;
  const [autoExecuteEnabled, setAutoExecuteEnabled] = useState(false);
  const [richPreviewEnabled, setRichPreviewEnabled] = useState(true);
  const [textDiffEnabled, setTextDiffEnabled] = useState(true);
  const [diffVariant, setDiffVariant] = useState<"inline" | "split">("inline");
  const [fileFilter, setFileFilter] = useState("");
  const [reviewStartState, setReviewStartState] = useState<
    "idle" | "starting" | "failed"
  >("idle");
  const {
    filesPanelGridStyle,
    handleFilesPanelResizeStart,
    handleFilesPanelResizeMove,
    handleFilesPanelResizeEnd,
    handleFilesPanelResizeKeyDown,
  } = useCanvasWorkbenchChangesFilesPanelResize(filesPanelOpen);
  const {
    backendHasGitRepository,
    backendPatch,
    branchCompareLabel,
    changeItems,
    commitOptions,
    commitsLoading,
    copyGitApply,
    fallbackPatch,
    fullFileContentById,
    loadFullFile,
    openCommitMenu,
    refreshChanges,
    reviewActionBusy,
    selectBase,
    selectCommit,
    selectedBase,
    selectedBaseUsesGit,
    selectedCommitSha,
    toggleLoadFullFile,
  } = useCanvasWorkbenchReviewState({
    changeView,
    workspaceRoot,
    loadFilePreview,
    translateWorkbench,
  });
  const changeItemCount = changeItems.length;
  const activeSelectionChangeItem = useMemo(
    () => findChangeItemForSelection(changeItems, documentContext),
    [changeItems, documentContext],
  );
  const selectedChangeItem = useMemo(
    () =>
      resolveCanvasWorkbenchSelectedChangeItem({
        items: changeItems,
        selectedChangeId,
        activeSelectionChangeItem,
        preferRuntimeEvidenceDefault: true,
      }),
    [activeSelectionChangeItem, changeItems, selectedChangeId],
  );
  const selectedChangeItemRef = useRef<CanvasWorkbenchChangeItem | undefined>(
    selectedChangeItem,
  );
  selectedChangeItemRef.current = selectedChangeItem;
  const selectedLoadedContent =
    loadFullFile && selectedChangeItem
      ? fullFileContentById[selectedChangeItem.id]
      : undefined;
  const selectedLoadFullFile = Boolean(
    loadFullFile && selectedLoadedContent != null,
  );
  const emptyDiffMessageKey =
    selectedBaseUsesGit && backendHasGitRepository === false
      ? "agentChat.canvasWorkbench.coding.changes.notGitRepository"
      : "agentChat.canvasWorkbench.coding.changes.empty";
  const latestReviewBoundary = resolveLatestReviewBoundary(
    changeView?.threadItems,
  );
  const reviewThreadId = changeView?.reviewThreadId?.trim() || "";
  const reviewBlocked = !reviewThreadId || Boolean(changeView?.canInterrupt);
  const reviewRunStatus =
    reviewStartState === "starting"
      ? "starting"
      : reviewStartState === "failed"
        ? "failed"
        : latestReviewBoundary?.status || "idle";
  const reviewStatusLabelKey =
    reviewBlocked
      ? "agentChat.harness.codeReview.status.blocked"
      : reviewRunStatus === "starting"
        ? "agentChat.harness.codeReview.status.starting"
          : reviewRunStatus === "inProgress"
            ? "agentChat.harness.codeReview.status.inProgress"
          : reviewRunStatus === "completed"
            ? "agentChat.harness.codeReview.status.completed"
            : reviewRunStatus === "failed"
              ? "agentChat.harness.codeReview.status.failed"
              : "agentChat.harness.codeReview.action.start";
  const reviewStartDisabled =
    reviewBlocked ||
    !changeView?.onStartReview ||
    reviewRunStatus === "starting" ||
    reviewRunStatus === "inProgress";
  const handleStartReview = async () => {
    if (reviewStartDisabled || !changeView?.onStartReview) {
      return;
    }
    setReviewStartState("starting");
    try {
      await changeView.onStartReview();
      setReviewStartState("idle");
    } catch {
      setReviewStartState("failed");
    }
  };

  useEffect(() => {
    if (
      selectedChangeId !== null &&
      !changeItems.some((item) => item.id === selectedChangeId)
    ) {
      setSelectedChangeId(null);
    }
  }, [changeItems, selectedChangeId]);

  const handleSelectBase = (base: typeof selectedBase) => {
    if (base === "commit") {
      return;
    }
    void selectBase(base).then((changed) => {
      if (changed) {
        setBaseMenuOpen(false);
      }
    });
  };
  const handleSelectCommit = (commit: Parameters<typeof selectCommit>[0]) => {
    selectCommit(commit);
    setBaseMenuOpen(false);
  };
  const handleToggleReviewMenu = () => {
    setBaseMenuOpen(false);
    setReviewMenuOpen((open) => !open);
  };
  const handleToggleBaseMenu = () => {
    setReviewMenuOpen(false);
    setBaseMenuOpen((open) => !open);
  };
  const handleToggleDiffVariant = () => {
    setBaseMenuOpen(false);
    setReviewMenuOpen(false);
    setDiffVariant((variant) => (variant === "inline" ? "split" : "inline"));
  };
  const handleToggleFilesPanel = () => {
    setBaseMenuOpen(false);
    setReviewMenuOpen(false);
    onToggleFilesPanel?.();
  };
  const buildToolbarProps = (
    props: Pick<
      CanvasWorkbenchChangesToolbarProps,
      | "diffStats"
      | "checkpointCount"
      | "copyGitApplyDisabled"
      | "diffViewToggleDisabled"
      | "loadFullFile"
      | "loadFullFileDisabled"
      | "onCopyGitApply"
      | "onToggleDiffVariant"
      | "onToggleLoadFullFile"
    >,
  ): CanvasWorkbenchChangesToolbarProps => ({
    translateWorkbench,
    reviewMenuOpen,
    baseMenuOpen,
    selectedBase,
    branchCompareLabel,
    filesPanelOpen,
    reviewActionBusy,
    autoExecuteEnabled,
    commitOptions,
    commitsLoading,
    selectedCommitSha,
    collapseDiffContext,
    richPreviewEnabled,
    textDiffEnabled,
    showWhitespace,
    diffVariant,
    onToggleReviewMenu: handleToggleReviewMenu,
    onToggleBaseMenu: handleToggleBaseMenu,
    onSelectBase: handleSelectBase,
    onOpenCommitMenu: openCommitMenu,
    onSelectCommit: handleSelectCommit,
    onRefreshChanges: refreshChanges,
    onToggleAutoExecute: () => setAutoExecuteEnabled((enabled) => !enabled),
    onToggleCollapseContext: () =>
      setCollapseDiffContext((enabled) => !enabled),
    onToggleRichPreview: () => setRichPreviewEnabled((enabled) => !enabled),
    onToggleTextDiff: () => setTextDiffEnabled((enabled) => !enabled),
    onToggleWhitespace: () => setShowWhitespace((enabled) => !enabled),
    onToggleFilesPanel: handleToggleFilesPanel,
    ...props,
    reviewRunStatus,
    reviewStatusLabelKey,
    reviewStartDisabled,
    onStartReview: changeView?.onStartReview ? handleStartReview : undefined,
  });

  const filesPanelResizeHandle = filesPanelOpen ? (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={translateWorkbench(
        "agentChat.canvasWorkbench.coding.changes.resizeFilesPanel",
      )}
      tabIndex={0}
      data-testid="canvas-workbench-changes-files-resizer"
      className="absolute inset-y-0 right-0 z-10 w-2 translate-x-1 cursor-col-resize touch-none"
      onPointerDown={handleFilesPanelResizeStart}
      onPointerMove={handleFilesPanelResizeMove}
      onPointerUp={handleFilesPanelResizeEnd}
      onPointerCancel={handleFilesPanelResizeEnd}
      onKeyDown={handleFilesPanelResizeKeyDown}
    >
      <span className="mx-auto block h-full w-px bg-slate-200 transition-colors group-hover:bg-slate-300" />
    </div>
  ) : null;

  if (changeItemCount > 0) {
    const selectedDiffLines = buildCanvasWorkbenchChangeDiffLines(
      selectedChangeItem,
      selectedLoadedContent,
    );
    const visibleDiffLines = collapseDiffContext
      ? collapseCanvasWorkbenchDiffContext(selectedDiffLines)
      : textDiffEnabled
        ? selectedDiffLines
        : selectedDiffLines.filter((line) => line.type !== "context");
    const filteredFileTree = buildCanvasWorkbenchChangeFileTree(
      changeItems,
      fileFilter,
    );
    const filteredChangeItems =
      flattenCanvasWorkbenchChangeFileTree(filteredFileTree);
    const selectedItemStats = selectedChangeItem
      ? countCanvasWorkbenchChangeItemsStats([selectedChangeItem])
      : { additions: 0, removals: 0 };
    const diffStats = selectedBaseUsesGit
      ? countCanvasWorkbenchChangeItemsStats(filteredChangeItems)
      : selectedItemStats;
    const latestCheckpointPath =
      changeView?.latestCheckpointPath ||
      selectedChangeItem?.checkpointPath ||
      null;
    const handleSelectChangeItem = (item: CanvasWorkbenchChangeItem) => {
      setSelectedChangeId(item.id);
      if (item.absolutePath || item.path) {
        void changeView?.onOpenFile?.(item.absolutePath || item.path);
      }
    };
    return (
      <CanvasWorkbenchReviewSurface
        toolbar={buildToolbarProps({
          diffStats,
          checkpointCount: changeView?.checkpointCount,
          copyGitApplyDisabled: !backendPatch?.trim() && !fallbackPatch.trim(),
          loadFullFileDisabled: !loadFilePreview || !selectedChangeItem,
          loadFullFile: selectedLoadFullFile,
          onCopyGitApply: copyGitApply,
          onToggleLoadFullFile: () => {
            void toggleLoadFullFile(selectedChangeItemRef.current);
          },
          onToggleDiffVariant: handleToggleDiffVariant,
        })}
        filesPanelOpen={filesPanelOpen}
        filesPanelGridStyle={filesPanelGridStyle}
        filesPanelResizeHandle={filesPanelResizeHandle}
        fileTree={filteredFileTree}
        selectedChangeItem={selectedChangeItem}
        fileFilter={fileFilter}
        translateWorkbench={translateWorkbench}
        onFileFilterChange={setFileFilter}
        onSelectChangeItem={handleSelectChangeItem}
        detail={
          selectedBaseUsesGit ? (
            <CanvasWorkbenchChangeDiffList
              items={filteredChangeItems}
              selectedChangeItem={selectedChangeItem}
              fullFileContentById={fullFileContentById}
              loadFullFile={loadFullFile}
              collapseDiffContext={collapseDiffContext}
              textDiffEnabled={textDiffEnabled}
              panelClassName={panelClassName}
              diffVariant={diffVariant}
              showWhitespace={showWhitespace}
              wordWrapEnabled={wordWrapEnabled}
              translateWorkbench={translateWorkbench}
              onSelectChangeItem={handleSelectChangeItem}
            />
          ) : (
            <CanvasWorkbenchChangeDetailPanel
              selectedChangeItem={selectedChangeItem}
              selectedItemStats={selectedItemStats}
              latestCheckpointPath={latestCheckpointPath}
              visibleDiffLines={visibleDiffLines}
              selectedDiffLines={selectedDiffLines}
              panelClassName={panelClassName}
              mutedPanelClassName={mutedPanelClassName}
              diffVariant={diffVariant}
              showWhitespace={showWhitespace}
              wordWrapEnabled={wordWrapEnabled}
              translateWorkbench={translateWorkbench}
            />
          )
        }
      />
    );
  }

  if ((changeView || selectedBaseUsesGit) && changeItemCount === 0) {
    return (
      <CanvasWorkbenchReviewSurface
        toolbar={buildToolbarProps({
          diffStats: { additions: 0, removals: 0 },
          copyGitApplyDisabled: !backendPatch?.trim(),
          diffViewToggleDisabled: true,
          loadFullFileDisabled: true,
          loadFullFile,
        })}
        filesPanelOpen={filesPanelOpen}
        filesPanelGridStyle={filesPanelGridStyle}
        filesPanelResizeHandle={filesPanelResizeHandle}
        fileTree={[]}
        fileFilter=""
        fileListDisabled
        translateWorkbench={translateWorkbench}
        detail={
          <CanvasWorkbenchEmptyDiffPanel
            messageKey={emptyDiffMessageKey}
            translateWorkbench={translateWorkbench}
          />
        }
      />
    );
  }

  if (!documentContext) {
    return (
      <div data-testid="canvas-workbench-panel-changes" className="p-5">
        <div className={mutedPanelClassName}>
          {translateWorkbench("agentChat.canvasWorkbench.coding.changes.empty")}
        </div>
      </div>
    );
  }

  if (documentContext.previousContent === null) {
    return (
      <div data-testid="canvas-workbench-panel-changes" className="p-5">
        <div className={mutedPanelClassName}>
          {translateWorkbench(
            "agentChat.canvasWorkbench.coding.changes.noBaseline",
          )}
        </div>
      </div>
    );
  }

  const documentChangeItem: CanvasWorkbenchChangeItem = {
    id: documentContext.selectionKey || `document:${documentContext.title}`,
    path:
      documentContext.selectionPath ||
      documentContext.subtitle ||
      documentContext.title,
    absolutePath: documentContext.selectionPath || null,
    displayName: documentContext.title,
    source: "document",
    status: "completed",
    changeKind: "modified",
    previousContent: documentContext.previousContent,
    currentContent: documentContext.content,
  };
  const documentDiffStats = countCanvasWorkbenchDiffStats(documentDiffLines);
  const visibleDocumentDiffLines = collapseDiffContext
    ? collapseCanvasWorkbenchDiffContext(documentDiffLines)
    : textDiffEnabled
      ? documentDiffLines
      : documentDiffLines.filter((line) => line.type !== "context");
  const filteredDocumentFileTree = buildCanvasWorkbenchChangeFileTree(
    [documentChangeItem],
    fileFilter,
  );

  return (
    <CanvasWorkbenchReviewSurface
      toolbar={buildToolbarProps({
        diffStats: documentDiffStats,
        copyGitApplyDisabled: true,
        loadFullFileDisabled: true,
        loadFullFile: false,
        onToggleDiffVariant: handleToggleDiffVariant,
      })}
      filesPanelOpen={filesPanelOpen}
      filesPanelGridStyle={filesPanelGridStyle}
      filesPanelResizeHandle={filesPanelResizeHandle}
      fileTree={filteredDocumentFileTree}
      selectedChangeItem={documentChangeItem}
      fileFilter={fileFilter}
      translateWorkbench={translateWorkbench}
      onFileFilterChange={setFileFilter}
      detail={
        <CanvasWorkbenchChangeDetailPanel
          selectedChangeItem={documentChangeItem}
          selectedItemStats={documentDiffStats}
          latestCheckpointPath={null}
          visibleDiffLines={visibleDocumentDiffLines}
          selectedDiffLines={documentDiffLines}
          panelClassName={panelClassName}
          mutedPanelClassName={mutedPanelClassName}
          diffVariant={diffVariant}
          showWhitespace={showWhitespace}
          wordWrapEnabled={wordWrapEnabled}
          translateWorkbench={translateWorkbench}
        />
      }
    />
  );
}
