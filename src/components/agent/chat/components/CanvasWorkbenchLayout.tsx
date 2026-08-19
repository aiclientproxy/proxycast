import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import type { Artifact } from "@/lib/artifact/types";
import type { CanvasStateUnion } from "@/components/workspace/canvas/canvasUtils";
import type { TaskFile } from "./TaskFiles";
import type { HarnessFilePreviewResult } from "./HarnessStatusPanel";
import { buildCanvasWorkbenchDiff } from "../utils/canvasWorkbenchDiff";
import { extractFileNameFromPath } from "../workspace/workspacePath";
import {
  type CanvasWorkbenchCopy,
  type CanvasWorkbenchPreviewTarget,
} from "./CanvasWorkbenchLayoutViewModel";
import {
  CanvasWorkbenchShell,
  type CanvasWorkbenchChangeView,
} from "./canvas-workbench";
import {
  WORKBENCH_GHOST_BUTTON_CLASSNAME,
  WORKBENCH_MUTED_PANEL_CLASSNAME,
  WORKBENCH_PANEL_CLASSNAME,
  WORKBENCH_SHELL_CLASSNAME,
  downloadCanvasWorkbenchText,
  translateCanvasWorkbenchText,
  type CanvasWorkbenchDefaultPreview,
  type CanvasWorkbenchLayoutMode,
  type CanvasWorkbenchMode,
  type CanvasWorkbenchNewToolTab,
  type CanvasWorkbenchTab,
  type CanvasWorkbenchTranslation,
} from "./canvas-workbench/CanvasWorkbenchLayoutState";
import { useCanvasWorkbenchDocumentState } from "./canvas-workbench/useCanvasWorkbenchDocumentState";
import { buildCanvasWorkbenchToolTabProjection } from "./canvas-workbench/tabs/CanvasWorkbenchToolTabsViewModel";
import { useCanvasWorkbenchToolTabsState } from "./canvas-workbench/tabs/useCanvasWorkbenchToolTabsState";

export type {
  CanvasWorkbenchDefaultPreview,
  CanvasWorkbenchLayoutMode,
  CanvasWorkbenchMode,
  CanvasWorkbenchNewToolTab,
  CanvasWorkbenchTab,
};

export type CanvasWorkbenchHeaderBadgeTone = "default" | "accent" | "success";

export interface CanvasWorkbenchHeaderBadge {
  key: string;
  label: string;
  tone?: CanvasWorkbenchHeaderBadgeTone;
}

export interface CanvasWorkbenchSummaryStat {
  key: string;
  label: string;
  value: string;
  detail: string;
  tone?: CanvasWorkbenchHeaderBadgeTone;
}

export interface CanvasWorkbenchPanelCopy {
  introText?: string;
  emptyText?: string;
  unavailableText?: string;
  sectionEyebrow?: string;
  loadingText?: string;
  emptyDirectoryText?: string;
}

export interface CanvasWorkbenchHeaderView {
  eyebrow?: string;
  title?: string;
  subtitle?: string;
  badges?: CanvasWorkbenchHeaderBadge[];
  summaryStats?: CanvasWorkbenchSummaryStat[];
  tabLabel?: string;
  tabBadge?: string;
  tabBadgeTone?: "slate" | "sky" | "rose";
  panelCopy?: CanvasWorkbenchPanelCopy;
}

export type { CanvasWorkbenchPreviewTarget };

export interface CanvasWorkbenchPreviewOpenRequest {
  requestKey: string | number;
  selectionKey?: string | null;
  filePath?: string | null;
}

export interface CanvasWorkbenchSessionView extends CanvasWorkbenchHeaderView {
  renderPanel: () => ReactNode;
}

export interface CanvasWorkbenchUtilityView extends CanvasWorkbenchHeaderView {
  enabled?: boolean;
  leadContent?: ReactNode;
  renderPanel: () => ReactNode;
}

function getPreviewTargetPath(
  target: CanvasWorkbenchPreviewTarget | null | undefined,
  key: "filePath" | "absolutePath",
): string | undefined {
  if (!target || target.kind === "empty") {
    return undefined;
  }
  return key === "filePath" ? target.filePath : target.absolutePath;
}

export interface CanvasWorkbenchLayoutProps {
  artifacts: Artifact[];
  canvasState: CanvasStateUnion | null;
  taskFiles: TaskFile[];
  selectedFileId?: string;
  workspaceRoot?: string | null;
  workspaceUnavailable?: boolean;
  defaultPreview: CanvasWorkbenchDefaultPreview | null;
  loadFilePreview: (path: string) => Promise<HarnessFilePreviewResult>;
  onOpenPath: (path: string) => Promise<void>;
  onRevealPath: (path: string) => Promise<void>;
  onClose?: () => void;
  onLayoutModeChange?: (mode: CanvasWorkbenchLayoutMode) => void;
  workbenchMode?: CanvasWorkbenchMode;
  workspaceView?: CanvasWorkbenchHeaderView | null;
  sessionView?: CanvasWorkbenchSessionView | null;
  outputView?: CanvasWorkbenchUtilityView | null;
  logView?: CanvasWorkbenchUtilityView | null;
  changeView?: CanvasWorkbenchChangeView | null;
  topRightTools?: ReactNode;
  previewOpenRequest?: CanvasWorkbenchPreviewOpenRequest | null;
  onPreviewOpenRequestHandled?: (requestKey: string | number) => void;
}

const STACKED_LAYOUT_BREAKPOINT = 1040;

function normalizePreviewOnlyPath(path: string | null | undefined): string {
  return path?.replace(/\\/g, "/").trim() || "";
}

function doesPreviewOnlyPathMatch(
  requestedPath: string | null | undefined,
  candidatePath: string | null | undefined,
): boolean {
  const requested = normalizePreviewOnlyPath(requestedPath);
  const candidate = normalizePreviewOnlyPath(candidatePath);
  if (!requested || !candidate) {
    return false;
  }
  return (
    requested === candidate ||
    requested.endsWith(`/${candidate}`) ||
    candidate.endsWith(`/${requested}`)
  );
}

export const CanvasWorkbenchLayout = memo(function CanvasWorkbenchLayout({
  artifacts,
  canvasState,
  taskFiles,
  selectedFileId,
  workspaceRoot,
  workspaceUnavailable = false,
  defaultPreview,
  loadFilePreview,
  onOpenPath,
  onRevealPath,
  onClose,
  onLayoutModeChange,
  workbenchMode = "default",
  workspaceView = null,
  sessionView = null,
  outputView = null,
  logView = null,
  changeView = null,
  topRightTools = null,
  previewOpenRequest = null,
  onPreviewOpenRequestHandled,
}: CanvasWorkbenchLayoutProps) {
  const { i18n, t } = useTranslation("agent");
  const locale = i18n.language || "zh-CN";
  const canvasT = t as unknown as CanvasWorkbenchTranslation;
  const [shellNodeVersion, setShellNodeVersion] = useState(0);
  const canvasTRef = useRef(canvasT);
  canvasTRef.current = canvasT;
  const handleShellRef = useCallback((node: HTMLElement | null) => {
    shellRef.current = node;
    if (node) {
      setShellNodeVersion((version) => version + 1);
    }
  }, []);
  const translateWorkbench = useCallback(
    (key: string, options?: Record<string, unknown>) =>
      translateCanvasWorkbenchText(canvasTRef.current, key, options),
    [],
  );
  const workbenchCopy = useMemo<CanvasWorkbenchCopy>(
    () => ({
      kind: {
        artifact: translateWorkbench("agentChat.canvasWorkbench.kind.artifact"),
        currentDraft: translateWorkbench(
          "agentChat.canvasWorkbench.kind.currentDraft",
        ),
        currentVersion: translateWorkbench(
          "agentChat.canvasWorkbench.kind.currentVersion",
        ),
        defaultDraft: translateWorkbench(
          "agentChat.canvasWorkbench.kind.defaultDraft",
        ),
        taskDocument: translateWorkbench(
          "agentChat.canvasWorkbench.kind.taskDocument",
        ),
        taskFile: translateWorkbench("agentChat.canvasWorkbench.kind.taskFile"),
        version: translateWorkbench("agentChat.canvasWorkbench.kind.version"),
        versionTitle: (count: number) =>
          translateWorkbench("agentChat.canvasWorkbench.kind.versionTitle", {
            count,
          }),
        workspaceFile: translateWorkbench(
          "agentChat.canvasWorkbench.kind.workspaceFile",
        ),
      },
      tab: {
        files: translateWorkbench("agentChat.canvasWorkbench.tabs.files"),
        generated: translateWorkbench(
          "agentChat.canvasWorkbench.tabs.generated",
        ),
        tasks: translateWorkbench("agentChat.canvasWorkbench.tabs.tasks"),
        sessionMain: translateWorkbench(
          "agentChat.canvasWorkbench.tabs.sessionMain",
        ),
      },
      workspaceFile: {
        binaryUnsupported: translateWorkbench(
          "agentChat.canvasWorkbench.workspaceFile.binaryUnsupported",
        ),
        readFailed: translateWorkbench(
          "agentChat.canvasWorkbench.workspaceFile.readFailed",
        ),
      },
    }),
    [translateWorkbench],
  );
  const isCodingWorkbench = workbenchMode === "coding";
  const hasDefaultPreviewContent = Boolean(defaultPreview?.content.trim());
  const hasCustomSessionView = Boolean(sessionView?.renderPanel);
  const hasCustomOutputView = Boolean(outputView?.renderPanel);
  const shellRef = useRef<HTMLElement | null>(null);
  const [isStackedLayout, setIsStackedLayout] = useState(false);
  const {
    activeTab,
    setActiveTab,
    activePreviewContext,
    documentContext,
    documentSelectionKey,
    previewModeState,
    projectFilesPreviewMode,
    setProjectFilesPreviewMode,
    workspacePanelRootPath,
    workspacePanelDisplayPath,
    directoryCache,
    expandedDirectories,
    loadingDirectories,
    toggleDirectory,
    refreshDirectorySubtree,
    openDocumentSelection,
    handleSelectWorkspaceFile,
  } = useCanvasWorkbenchDocumentState({
    artifacts,
    canvasState,
    taskFiles,
    selectedFileId,
    workspaceRoot,
    workspaceUnavailable,
    defaultPreview,
    loadFilePreview,
    workbenchCopy,
    translateWorkbench,
    hasDefaultPreviewContent,
    hasCustomSessionView,
    hasCustomOutputView,
    isCodingWorkbench,
    changeView,
  });
  const { openedToolTabs, openNewToolTab, closeToolTab, resolveToolTabKind } =
    useCanvasWorkbenchToolTabsState({
      activeTab,
      setActiveTab,
    });
  const [changesFilesPanelOpen, setChangesFilesPanelOpen] = useState(false);
  const [previewOnlyFilePath, setPreviewOnlyFilePath] = useState<string | null>(
    null,
  );
  const handledPreviewOpenRequestKeyRef = useRef<string | number | null>(null);
  const hasAutoFocusedInitialDocumentTabRef = useRef(
    Boolean(documentSelectionKey),
  );

  useEffect(() => {
    if (hasAutoFocusedInitialDocumentTabRef.current) {
      return;
    }
    if (!sessionView?.renderPanel || activeTab !== "outputs") {
      return;
    }
    if (!documentSelectionKey) {
      return;
    }
    hasAutoFocusedInitialDocumentTabRef.current = true;
    setActiveTab(previewModeState.defaultMode);
  }, [
    activeTab,
    documentSelectionKey,
    previewModeState.defaultMode,
    sessionView?.renderPanel,
    setActiveTab,
  ]);

  useEffect(() => {
    const node = shellRef.current;
    if (!node) {
      return;
    }

    const updateLayout = (width: number) => {
      if (width <= 0) {
        return;
      }
      setIsStackedLayout(width < STACKED_LAYOUT_BREAKPOINT);
    };

    const fallbackWidth =
      node.getBoundingClientRect().width ||
      node.clientWidth ||
      window.innerWidth;
    updateLayout(fallbackWidth);

    if (typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver((observerEntries) => {
      const contentRect = observerEntries[0]?.contentRect;
      const nextWidth =
        contentRect?.width ||
        node.getBoundingClientRect().width ||
        node.clientWidth;
      updateLayout(nextWidth);
    });

    observer.observe(node);
    return () => {
      observer.disconnect();
    };
  }, [shellNodeVersion]);

  useEffect(() => {
    onLayoutModeChange?.(isStackedLayout ? "stacked" : "split");
  }, [isStackedLayout, onLayoutModeChange]);

  useEffect(() => {
    const node = shellRef.current;
    if (!node) {
      return;
    }

    node.dataset.documentSelectionKey = documentSelectionKey || "";
    node.dataset.documentContextKey = documentContext?.selectionKey || "";
    node.dataset.previewRequestKey = String(
      previewOpenRequest?.requestKey ?? "",
    );
    node.dataset.previewRequestSelectionKey =
      previewOpenRequest?.selectionKey || "";
    node.dataset.artifactSelectionKeys = artifacts
      .map((artifact) => `artifact:${artifact.id}`)
      .join(",");
  }, [
    artifacts,
    documentContext?.selectionKey,
    documentSelectionKey,
    previewOpenRequest?.requestKey,
    previewOpenRequest?.selectionKey,
  ]);

  const activeSelectionPath = activePreviewContext?.selectionPath;
  const activeContent = activePreviewContext?.content || "";
  const closeWorkbenchLabel = translateCanvasWorkbenchText(
    canvasT,
    "agentChat.canvasWorkbench.close",
  );
  const headerActionsVisible = Boolean(
    activePreviewContext && (activeSelectionPath || activeContent.trim()),
  );

  const documentDiffLines = useMemo(
    () =>
      documentContext && documentContext.previousContent !== null
        ? buildCanvasWorkbenchDiff(
            documentContext.previousContent,
            documentContext.content,
          )
        : [],
    [documentContext],
  );
  const activeDocumentPaths = useMemo(
    () => [
      documentContext?.selectionPath,
      documentContext?.subtitle,
      getPreviewTargetPath(documentContext?.target, "filePath"),
      getPreviewTargetPath(documentContext?.target, "absolutePath"),
      documentContext?.title,
    ],
    [
      documentContext?.selectionPath,
      documentContext?.subtitle,
      documentContext?.target,
      documentContext?.title,
    ],
  );
  const isPreviewOnlyFileOpen = Boolean(
    previewOnlyFilePath &&
    activeDocumentPaths.some((candidatePath) =>
      doesPreviewOnlyPathMatch(previewOnlyFilePath, candidatePath),
    ),
  );
  const hasReviewSurface =
    isCodingWorkbench || Boolean(changeView) || documentDiffLines.length > 0;
  const resolvedChangeView = useMemo<CanvasWorkbenchChangeView | null>(() => {
    if (changeView) {
      return changeView;
    }
    if (!hasReviewSurface) {
      return null;
    }
    if (documentDiffLines.length > 0) {
      return null;
    }
    return {
      items: [],
      checkpointCount: 0,
      latestCheckpointPath: null,
    };
  }, [changeView, documentDiffLines.length, hasReviewSurface]);
  const changeItems = useMemo(() => changeView?.items ?? [], [changeView]);
  const changeItemCount = changeItems.length;
  const failedChangeItemCount = useMemo(
    () => changeItems.filter((item) => item.status === "failed").length,
    [changeItems],
  );

  const hasAutoFocusedCodingReviewRef = useRef(
    (changeView?.items?.length || 0) > 0 ||
      Boolean(defaultPreview?.previousContent),
  );

  useEffect(() => {
    if (!hasReviewSurface) {
      return;
    }
    if (isPreviewOnlyFileOpen) {
      return;
    }
    if (hasAutoFocusedCodingReviewRef.current) {
      return;
    }
    if (changeItemCount <= 0 && documentDiffLines.length <= 0) {
      return;
    }
    hasAutoFocusedCodingReviewRef.current = true;
    setActiveTab("changes");
  }, [
    changeItemCount,
    documentDiffLines.length,
    hasReviewSurface,
    isPreviewOnlyFileOpen,
    setActiveTab,
  ]);

  useEffect(() => {
    if (!previewOpenRequest) {
      return;
    }
    if (
      handledPreviewOpenRequestKeyRef.current === previewOpenRequest.requestKey
    ) {
      return;
    }
    const selectionKey = previewOpenRequest.selectionKey?.trim();
    if (selectionKey) {
      if (documentSelectionKey !== selectionKey) {
        setPreviewOnlyFilePath(previewOpenRequest.filePath?.trim() || null);
        openDocumentSelection(selectionKey);
        return;
      }
      if (documentContext?.selectionKey !== selectionKey) {
        return;
      }
      setActiveTab(previewModeState.defaultMode);
      setPreviewOnlyFilePath(previewOpenRequest.filePath?.trim() || null);
      handledPreviewOpenRequestKeyRef.current = previewOpenRequest.requestKey;
      onPreviewOpenRequestHandled?.(previewOpenRequest.requestKey);
      return;
    }
    setPreviewOnlyFilePath(previewOpenRequest.filePath?.trim() || null);
    setActiveTab(previewModeState.defaultMode);
    handledPreviewOpenRequestKeyRef.current = previewOpenRequest.requestKey;
    onPreviewOpenRequestHandled?.(previewOpenRequest.requestKey);
  }, [
    onPreviewOpenRequestHandled,
    openDocumentSelection,
    documentContext?.selectionKey,
    documentSelectionKey,
    previewModeState.defaultMode,
    previewOpenRequest,
    setActiveTab,
  ]);

  const { primaryTabs: basePrimaryTabs, newTabActions } = useMemo(
    () =>
      buildCanvasWorkbenchToolTabProjection({
        changeItemCount,
        documentDiffLineCount: documentDiffLines.length,
        failedChangeItemCount,
        utilityTabs: {
          outputs: hasCustomOutputView,
          logs: Boolean(logView?.renderPanel),
        },
        openedToolTabs,
        translateWorkbench: canvasT,
      }),
    [
      changeItemCount,
      documentDiffLines.length,
      failedChangeItemCount,
      logView?.renderPanel,
      openedToolTabs,
      hasCustomOutputView,
      canvasT,
    ],
  );
  const primaryTabs = useMemo(() => {
    if (!documentContext) {
      return basePrimaryTabs;
    }

    return [
      {
        key: previewModeState.defaultMode,
        label: documentContext.tabLabel || documentContext.title,
      },
      ...basePrimaryTabs,
    ];
  }, [basePrimaryTabs, documentContext, previewModeState.defaultMode]);

  const handleCopyPath = useCallback(async () => {
    if (!activeSelectionPath) {
      return;
    }
    try {
      if (!navigator.clipboard?.writeText) {
        throw new Error(
          translateWorkbench("agentChat.canvasWorkbench.clipboard.unsupported"),
        );
      }
      await navigator.clipboard.writeText(activeSelectionPath);
      toast.success(
        translateWorkbench("agentChat.canvasWorkbench.clipboard.copied"),
      );
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : translateWorkbench(
              "agentChat.canvasWorkbench.clipboard.copyFailed",
            ),
      );
    }
  }, [activeSelectionPath, translateWorkbench]);

  const handleDownload = useCallback(() => {
    if (!activeContent.trim()) {
      return;
    }
    const filename = extractFileNameFromPath(
      activeSelectionPath || activePreviewContext?.title || "canvas.md",
    );
    downloadCanvasWorkbenchText(filename, activeContent);
  }, [activeContent, activePreviewContext?.title, activeSelectionPath]);

  const activeToolTabKind = resolveToolTabKind(activeTab);
  const topActiveTab: CanvasWorkbenchTab =
    documentContext && activeTab === "code"
      ? previewModeState.defaultMode
      : primaryTabs.some((tab) => tab.key === activeTab) ||
          (activeTab === "changes" && hasReviewSurface)
        ? activeTab
        : hasReviewSurface
          ? "changes"
          : primaryTabs[0]?.key || activeTab;

  return (
    <CanvasWorkbenchShell
      shellRef={handleShellRef}
      layoutMode={isStackedLayout ? "stacked" : "split"}
      activeTab={topActiveTab}
      contentTab={activeTab}
      tabs={primaryTabs}
      newTabActions={newTabActions}
      topRightTools={topRightTools}
      detailHeaderVisible={
        activeTab !== "changes" &&
        activeToolTabKind !== "terminal" &&
        (activeToolTabKind !== "project-files" || Boolean(activePreviewContext))
      }
      translateWorkbench={translateWorkbench}
      onSelectTab={setActiveTab}
      onNewToolTab={openNewToolTab}
      onCloseTab={closeToolTab}
      onClose={onClose}
      closeWorkbenchLabel={closeWorkbenchLabel}
      headerActionsVisible={headerActionsVisible}
      activeSelectionPath={activeSelectionPath}
      activeContent={activeContent}
      onCopyPath={() => {
        void handleCopyPath();
      }}
      onOpenPath={(path) => {
        void onOpenPath(path);
      }}
      onRevealPath={(path) => {
        void onRevealPath(path);
      }}
      onDownload={handleDownload}
      documentContext={documentContext}
      documentSelectionKey={documentSelectionKey}
      documentDiffLines={documentDiffLines}
      previewModeState={previewModeState}
      changeView={resolvedChangeView}
      changesFilesPanelOpen={changesFilesPanelOpen}
      loadFilePreview={loadFilePreview}
      workspaceUnavailable={workspaceUnavailable}
      workspaceRoot={workspaceRoot}
      workspacePanelRootPath={workspacePanelRootPath}
      workspacePanelDisplayPath={workspacePanelDisplayPath}
      projectFilesPreviewMode={projectFilesPreviewMode}
      directoryCache={directoryCache}
      expandedDirectories={expandedDirectories}
      loadingDirectories={loadingDirectories}
      workspacePanelCopy={workspaceView?.panelCopy}
      locale={locale}
      outputView={outputView}
      logView={logView}
      sessionView={sessionView}
      shellClassName={WORKBENCH_SHELL_CLASSNAME}
      panelClassName={WORKBENCH_PANEL_CLASSNAME}
      mutedPanelClassName={WORKBENCH_MUTED_PANEL_CLASSNAME}
      ghostButtonClassName={WORKBENCH_GHOST_BUTTON_CLASSNAME}
      onToggleDirectory={toggleDirectory}
      onSelectFile={(path) => {
        void handleSelectWorkspaceFile(path);
      }}
      onRefreshDirectory={(path) => {
        void refreshDirectorySubtree(path);
      }}
      onSelectPreviewMode={setActiveTab}
      onSelectProjectFilesPreviewMode={setProjectFilesPreviewMode}
      onToggleChangesFilesPanel={() =>
        setChangesFilesPanelOpen((open) => !open)
      }
    />
  );
});
