import { memo, type ReactNode } from "react";
import { Copy, Download, ExternalLink, FolderOpen, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { DirectoryListing } from "@/lib/api/fileBrowser";
import type { CanvasWorkbenchDiffLine } from "../../utils/canvasWorkbenchDiff";
import type {
  CanvasWorkbenchLayoutMode,
  CanvasWorkbenchNewToolTab,
  CanvasWorkbenchSessionView,
  CanvasWorkbenchTab,
  CanvasWorkbenchUtilityView,
} from "../CanvasWorkbenchLayout";
import { resolveCanvasWorkbenchToolTabKind } from "./CanvasWorkbenchLayoutState";
import type { HarnessFilePreviewResult } from "../HarnessStatusPanel";
import type { CanvasWorkbenchResolvedSelection } from "../CanvasWorkbenchLayoutViewModel";
import { CanvasWorkbenchChangesPanel } from "./changes/CanvasWorkbenchChangesPanel";
import type { CanvasWorkbenchChangeView } from "./changes/CanvasWorkbenchChangesPanelViewModel";
import { CanvasWorkbenchPreviewModePanel } from "./CanvasWorkbenchPreviewModePanel";
import {
  isCanvasWorkbenchPreviewMode,
  type CanvasWorkbenchPreviewMode,
  type CanvasWorkbenchPreviewModeState,
} from "./CanvasWorkbenchPreviewModeViewModel";
import {
  CanvasWorkbenchTopTabs,
  type CanvasWorkbenchNewTabAction,
  type CanvasWorkbenchTopTab,
} from "./tabs/CanvasWorkbenchTopTabs";
import { CanvasWorkbenchUtilityPanel } from "./CanvasWorkbenchUtilityPanel";
import {
  CanvasWorkbenchWorkspacePanel,
  type CanvasWorkbenchWorkspacePanelProps,
} from "./CanvasWorkbenchWorkspacePanel";
import { CanvasWorkbenchProjectFilesPanel } from "./files/CanvasWorkbenchProjectFilesPanel";
import { ProjectShellSurface } from "../ProjectShellSurface";

type CanvasWorkbenchTranslation = (
  key: string,
  options?: Record<string, unknown>,
) => string;

interface CanvasWorkbenchShellProps {
  shellRef: (node: HTMLElement | null) => void;
  layoutMode: CanvasWorkbenchLayoutMode;
  activeTab: CanvasWorkbenchTab;
  contentTab: CanvasWorkbenchTab;
  tabs: CanvasWorkbenchTopTab[];
  newTabActions: CanvasWorkbenchNewTabAction[];
  topRightTools?: ReactNode;
  detailHeaderVisible?: boolean;
  translateWorkbench: CanvasWorkbenchTranslation;
  onSelectTab: (tab: CanvasWorkbenchTab) => void;
  onNewToolTab: (tab: CanvasWorkbenchNewToolTab) => void;
  onCloseTab?: (tab: CanvasWorkbenchTab) => void;
  onClose?: () => void;
  closeWorkbenchLabel: string;
  headerActionsVisible: boolean;
  activeSelectionPath?: string;
  activeContent: string;
  onCopyPath: () => void;
  onOpenPath: (path: string) => void;
  onRevealPath: (path: string) => void;
  onDownload: () => void;
  documentContext: CanvasWorkbenchResolvedSelection | null;
  documentSelectionKey: string | null;
  documentDiffLines: CanvasWorkbenchDiffLine[];
  previewModeState: CanvasWorkbenchPreviewModeState;
  changeView?: CanvasWorkbenchChangeView | null;
  changesFilesPanelOpen?: boolean;
  loadFilePreview?: (path: string) => Promise<HarnessFilePreviewResult>;
  workspaceUnavailable: boolean;
  workspaceRoot?: string | null;
  workspacePanelRootPath: string | null;
  workspacePanelDisplayPath?: string | null;
  projectFilesPreviewMode: CanvasWorkbenchPreviewMode;
  directoryCache: Record<string, DirectoryListing>;
  expandedDirectories: Record<string, boolean>;
  loadingDirectories: Record<string, boolean>;
  workspacePanelCopy?: CanvasWorkbenchWorkspacePanelProps["panelCopy"];
  locale: string;
  outputView?: CanvasWorkbenchUtilityView | null;
  logView?: CanvasWorkbenchUtilityView | null;
  sessionView?: CanvasWorkbenchSessionView | null;
  shellClassName: string;
  panelClassName: string;
  mutedPanelClassName: string;
  ghostButtonClassName: string;
  onToggleDirectory: (path: string) => void;
  onSelectFile: (path: string) => void;
  onRefreshDirectory: (path: string) => void;
  onSelectPreviewMode: (mode: CanvasWorkbenchPreviewMode) => void;
  onSelectProjectFilesPreviewMode: (mode: CanvasWorkbenchPreviewMode) => void;
  onToggleChangesFilesPanel?: () => void;
}

function HeaderActionButton({
  label,
  onClick,
  disabled,
  icon,
  ghostButtonClassName,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  icon: ReactNode;
  ghostButtonClassName: string;
}) {
  if (disabled) {
    return null;
  }

  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className={cn(
        "inline-flex h-8 w-8 items-center justify-center rounded-[7px] border text-[color:var(--lime-text-muted)] transition-colors",
        ghostButtonClassName,
      )}
    >
      {icon}
    </button>
  );
}

export const CanvasWorkbenchShell = memo(function CanvasWorkbenchShell({
  shellRef,
  layoutMode,
  activeTab,
  contentTab,
  tabs,
  newTabActions,
  topRightTools = null,
  detailHeaderVisible = true,
  translateWorkbench,
  onSelectTab,
  onNewToolTab,
  onCloseTab,
  onClose,
  closeWorkbenchLabel,
  headerActionsVisible,
  activeSelectionPath,
  activeContent,
  onCopyPath,
  onOpenPath,
  onRevealPath,
  onDownload,
  documentContext,
  documentSelectionKey,
  documentDiffLines,
  previewModeState,
  changeView,
  changesFilesPanelOpen = true,
  loadFilePreview,
  workspaceUnavailable,
  workspaceRoot,
  workspacePanelRootPath,
  workspacePanelDisplayPath,
  projectFilesPreviewMode,
  directoryCache,
  expandedDirectories,
  loadingDirectories,
  workspacePanelCopy,
  locale,
  outputView,
  logView,
  sessionView,
  shellClassName,
  panelClassName,
  mutedPanelClassName,
  ghostButtonClassName,
  onToggleDirectory,
  onSelectFile,
  onRefreshDirectory,
  onSelectPreviewMode,
  onSelectProjectFilesPreviewMode,
  onToggleChangesFilesPanel,
}: CanvasWorkbenchShellProps) {
  const contentToolTabKind = resolveCanvasWorkbenchToolTabKind(contentTab);
  const workspacePanel = (
    <CanvasWorkbenchWorkspacePanel
      workspaceUnavailable={workspaceUnavailable}
      workspaceRoot={workspaceRoot}
      workspacePanelRootPath={workspacePanelRootPath}
      workspacePanelDisplayPath={workspacePanelDisplayPath}
      selectedFileKey={documentSelectionKey}
      directoryCache={directoryCache}
      expandedDirectories={expandedDirectories}
      loadingDirectories={loadingDirectories}
      panelCopy={workspacePanelCopy}
      locale={locale}
      ghostButtonClassName={ghostButtonClassName}
      mutedPanelClassName={mutedPanelClassName}
      translateWorkbench={translateWorkbench}
      onToggleDirectory={onToggleDirectory}
      onSelectFile={onSelectFile}
      onRefreshDirectory={onRefreshDirectory}
    />
  );

  return (
    <section
      ref={shellRef}
      data-testid="canvas-workbench-shell"
      data-layout-mode={layoutMode}
      className={cn(
        "lime-workbench-theme-scope",
        "lime-workbench-surface-scope",
        shellClassName,
        "relative flex h-full min-h-0 flex-col overflow-hidden",
      )}
    >
      <header className="relative z-[90] shrink-0 overflow-visible border-b border-slate-200 bg-white">
        <div
          data-testid="canvas-workbench-header-row"
          className="flex h-10 min-w-0 items-center justify-between gap-2 px-2.5"
        >
          <div
            className="min-w-0 flex-1 overflow-visible"
            data-testid="canvas-workbench-top-tabs-slot"
          >
            <CanvasWorkbenchTopTabs
              activeTab={activeTab}
              tabs={tabs}
              newTabActions={newTabActions}
              translateWorkbench={translateWorkbench}
              onSelectTab={onSelectTab}
              onNewToolTab={onNewToolTab}
              onCloseTab={onCloseTab}
            />
          </div>

          <div
            className="flex shrink-0 items-center gap-1"
            data-testid="canvas-workbench-top-right-tools"
          >
            {topRightTools}
          </div>
        </div>

        {detailHeaderVisible ? (
          <div className="flex h-10 min-w-0 items-center justify-between gap-2 border-t border-slate-100 bg-slate-50/70 px-3">
            <div className="flex min-w-0 items-center gap-2">
              <span className="truncate px-1 font-mono text-[12px] font-medium text-slate-600">
                {documentContext?.title ||
                  translateWorkbench(
                    "agentChat.canvasWorkbench.title.fallback",
                  )}
              </span>
            </div>
            {headerActionsVisible || onClose ? (
              <div className="flex shrink-0 items-center justify-end gap-1">
                {headerActionsVisible ? (
                  <div
                    className="flex shrink-0 items-center justify-end gap-1"
                    data-testid="canvas-workbench-header-actions"
                  >
                    <HeaderActionButton
                      label={translateWorkbench(
                        "agentChat.canvasWorkbench.actions.copyPath",
                      )}
                      disabled={!activeSelectionPath}
                      onClick={onCopyPath}
                      icon={<Copy className="h-4 w-4" />}
                      ghostButtonClassName={ghostButtonClassName}
                    />
                    <HeaderActionButton
                      label={translateWorkbench(
                        "agentChat.canvasWorkbench.actions.revealPath",
                      )}
                      disabled={!activeSelectionPath}
                      onClick={() => {
                        if (activeSelectionPath) {
                          onRevealPath(activeSelectionPath);
                        }
                      }}
                      icon={<FolderOpen className="h-4 w-4" />}
                      ghostButtonClassName={ghostButtonClassName}
                    />
                    <HeaderActionButton
                      label={translateWorkbench(
                        "agentChat.canvasWorkbench.actions.openPath",
                      )}
                      disabled={!activeSelectionPath}
                      onClick={() => {
                        if (activeSelectionPath) {
                          onOpenPath(activeSelectionPath);
                        }
                      }}
                      icon={<ExternalLink className="h-4 w-4" />}
                      ghostButtonClassName={ghostButtonClassName}
                    />
                    <HeaderActionButton
                      label={translateWorkbench(
                        "agentChat.canvasWorkbench.actions.download",
                      )}
                      disabled={!activeContent.trim()}
                      onClick={onDownload}
                      icon={<Download className="h-4 w-4" />}
                      ghostButtonClassName={ghostButtonClassName}
                    />
                  </div>
                ) : null}

                {onClose ? (
                  <button
                    type="button"
                    aria-label={closeWorkbenchLabel}
                    title={closeWorkbenchLabel}
                    onClick={onClose}
                    className={cn(
                      "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[7px] border transition-colors",
                      ghostButtonClassName,
                    )}
                  >
                    <X className="h-4 w-4" />
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}
      </header>

      <div
        data-testid="canvas-workbench-layout"
        data-panel-placement="canvas"
        className="relative z-0 min-h-0 flex-1 overflow-hidden bg-[image:var(--lime-stage-surface-soft)]"
      >
        {isCanvasWorkbenchPreviewMode(contentTab) ? (
          <CanvasWorkbenchPreviewModePanel
            context={documentContext}
            mode={contentTab}
            modeState={previewModeState}
            translateWorkbench={translateWorkbench}
            onSelectMode={onSelectPreviewMode}
          />
        ) : contentTab === "workspace" ? (
          workspacePanel
        ) : contentToolTabKind === "project-files" ? (
          <CanvasWorkbenchProjectFilesPanel
            workspaceUnavailable={workspaceUnavailable}
            workspaceRoot={workspaceRoot}
            selectedFileKey={documentSelectionKey}
            documentContext={documentContext}
            previewMode={projectFilesPreviewMode}
            directoryCache={directoryCache}
            expandedDirectories={expandedDirectories}
            loadingDirectories={loadingDirectories}
            panelCopy={workspacePanelCopy}
            locale={locale}
            ghostButtonClassName={ghostButtonClassName}
            mutedPanelClassName={mutedPanelClassName}
            translateWorkbench={translateWorkbench}
            onToggleDirectory={onToggleDirectory}
            onSelectFile={onSelectFile}
            onRefreshDirectory={onRefreshDirectory}
            onSelectPreviewMode={onSelectProjectFilesPreviewMode}
          />
        ) : contentTab === "changes" ? (
          <CanvasWorkbenchChangesPanel
            changeView={changeView}
            documentContext={documentContext}
            documentDiffLines={documentDiffLines}
            translateWorkbench={translateWorkbench}
            panelClassName={panelClassName}
            mutedPanelClassName={mutedPanelClassName}
            workspaceRoot={workspaceRoot}
            loadFilePreview={loadFilePreview}
            filesPanelOpen={changesFilesPanelOpen}
            onToggleFilesPanel={onToggleChangesFilesPanel}
          />
        ) : contentTab === "outputs" ? (
          <CanvasWorkbenchUtilityPanel
            view={outputView || sessionView}
            testId="canvas-workbench-panel-outputs"
            fallbackTextKey="agentChat.canvasWorkbench.coding.outputs.empty"
            mutedPanelClassName={mutedPanelClassName}
            translateWorkbench={translateWorkbench}
          />
        ) : contentTab === "logs" ? (
          <CanvasWorkbenchUtilityPanel
            view={logView}
            testId="canvas-workbench-panel-logs"
            fallbackTextKey="agentChat.canvasWorkbench.coding.logs.empty"
            mutedPanelClassName={mutedPanelClassName}
            translateWorkbench={translateWorkbench}
          />
        ) : contentToolTabKind === "terminal" ? (
          <section
            data-testid="canvas-workbench-panel-terminal"
            className="h-full min-h-0 bg-white"
          >
            <ProjectShellSurface
              projectRootPath={workspaceRoot}
              testIdPrefix="canvas-workbench-shell"
              className="h-full"
              bodyClassName="px-3 py-2"
            />
          </section>
        ) : (
          workspacePanel
        )}
      </div>
    </section>
  );
});

CanvasWorkbenchShell.displayName = "CanvasWorkbenchShell";
