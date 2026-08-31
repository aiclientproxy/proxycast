import { Children, useState, type CSSProperties, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import {
  AutoHideNavbarBackdrop,
  AutoHideNavbarHandle,
  AutoHideNavbarHost,
  AutoHideNavbarPanel,
  LayoutTransitionRenderGate,
  MainArea,
  GeneralWorkbenchInputOverlay,
  GeneralWorkbenchLayoutShell,
  RIGHT_SURFACE_CHAT_PANEL_MIN_WIDTH,
  RIGHT_SURFACE_CHAT_PANEL_WIDTH,
} from "./WorkspaceStyles";
import { TASK_CENTER_CHROME_RAIL_SURFACE } from "./taskCenterChromeTokens";

const TASK_CENTER_SPLIT_CHROME_TOP_INSET = "84px";
const TASK_CENTER_SPLIT_WORKBENCH_TOP_INSET = "42px";
const TASK_CENTER_DETACHED_TOOLBAR_TOP_INSET = "42px";
const THREAD_WORKSPACE_HEADER_TOP_INSET = "52px";
const TASK_CENTER_SPLIT_CHROME_BREAKPOINT_WIDTH = 900;
const TASK_CENTER_SPLIT_CHROME_BREAKPOINT_HEIGHT = 620;
const TASK_CENTER_DETACHED_TOOLBAR_BREAKPOINT_WIDTH = 1024;
const TASK_CENTER_TOP_TOOLBAR_MAX_WIDTH = "100%";
const TASK_CENTER_TOP_TOOLBAR_RESERVE_WIDTH = "min(360px, 34%)";
const DEFAULT_CHAT_CANVAS_PANEL_WIDTH = "min(100%, clamp(640px, 54%, 1180px))";
const DEFAULT_TASK_CENTER_HOME_CHROME_WIDTH = `calc(100% - ${TASK_CENTER_TOP_TOOLBAR_RESERVE_WIDTH})`;

function shouldUseTaskCenterSplitChrome(mode: LayoutMode): boolean {
  if (mode !== "chat-canvas" || typeof window === "undefined") {
    return false;
  }

  return (
    window.innerWidth > TASK_CENTER_SPLIT_CHROME_BREAKPOINT_WIDTH &&
    window.innerHeight > TASK_CENTER_SPLIT_CHROME_BREAKPOINT_HEIGHT
  );
}

function shouldDetachTaskCenterHomeToolbar(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return window.innerWidth >= TASK_CENTER_DETACHED_TOOLBAR_BREAKPOINT_WIDTH;
}

interface WorkspaceMainAreaProps {
  compactChrome: boolean;
  navbarNode: ReactNode;
  threadHeaderNode?: ReactNode;
  autoHideTaskCenterNavbar?: boolean;
  taskCenterUtilityToolbarNode?: ReactNode;
  taskCenterTabsNode?: ReactNode;
  taskCenterShellPanelNode?: ReactNode;
  contentSyncNoticeNode: ReactNode;
  shellBottomInset: string;
  layoutMode: LayoutMode;
  forceCanvasMode: boolean;
  chatContent: ReactNode;
  canvasContent: ReactNode;
  rightSurfaceContent?: ReactNode;
  chatPanelWidth?: string;
  chatPanelMinWidth?: string;
  generalWorkbenchDialog: ReactNode;
  showFloatingInputOverlay: boolean;
  hasPendingA2UIForm: boolean;
  inputbarNode: ReactNode;
}

export function WorkspaceMainArea({
  compactChrome,
  navbarNode,
  threadHeaderNode,
  autoHideTaskCenterNavbar = false,
  taskCenterUtilityToolbarNode,
  taskCenterTabsNode,
  taskCenterShellPanelNode,
  contentSyncNoticeNode,
  shellBottomInset,
  layoutMode,
  forceCanvasMode,
  chatContent,
  canvasContent,
  rightSurfaceContent,
  chatPanelWidth,
  chatPanelMinWidth,
  generalWorkbenchDialog,
  showFloatingInputOverlay,
  hasPendingA2UIForm,
  inputbarNode,
}: WorkspaceMainAreaProps) {
  const { t } = useTranslation("agent");
  const [navbarOpen, setNavbarOpen] = useState(false);
  const hasRightSurfaceContent = Children.count(rightSurfaceContent) > 0;
  const hasThreadHeader = Children.count(threadHeaderNode) > 0;
  const shouldUseRightSurfaceChatWidth =
    hasRightSurfaceContent && !chatPanelWidth;
  const effectiveChatPanelWidth = shouldUseRightSurfaceChatWidth
    ? RIGHT_SURFACE_CHAT_PANEL_WIDTH
    : chatPanelWidth;
  const effectiveChatPanelMinWidth =
    shouldUseRightSurfaceChatWidth && !chatPanelMinWidth
      ? RIGHT_SURFACE_CHAT_PANEL_MIN_WIDTH
      : chatPanelMinWidth;
  const effectiveCanvasContent = hasRightSurfaceContent
    ? rightSurfaceContent
    : canvasContent;
  const effectiveLayoutMode = hasPendingA2UIForm
    ? "chat"
    : hasRightSurfaceContent
      ? "chat-canvas"
      : forceCanvasMode
        ? "canvas"
        : layoutMode;
  const shouldAutoHideNavbar =
    autoHideTaskCenterNavbar &&
    Boolean(navbarNode) &&
    Boolean(taskCenterTabsNode);
  const isAutoHideNavbarVisible = shouldAutoHideNavbar && navbarOpen;
  const shouldRenderRevealHandle =
    shouldAutoHideNavbar && !isAutoHideNavbarVisible;
  const hasCanvasContent = Children.count(effectiveCanvasContent) > 0;
  const shouldRenderTaskCenterChrome =
    !hasThreadHeader &&
    !taskCenterUtilityToolbarNode &&
    !shouldAutoHideNavbar &&
    Boolean(taskCenterTabsNode);
  const shouldSplitTaskCenterChrome =
    shouldRenderTaskCenterChrome &&
    hasCanvasContent &&
    shouldUseTaskCenterSplitChrome(effectiveLayoutMode);
  const shouldDetachTaskCenterToolbar =
    !hasThreadHeader &&
    !shouldSplitTaskCenterChrome &&
    Boolean(taskCenterUtilityToolbarNode) &&
    (!shouldRenderTaskCenterChrome || shouldDetachTaskCenterHomeToolbar());
  const splitTaskCenterChromeWidth =
    effectiveChatPanelWidth || DEFAULT_CHAT_CANVAS_PANEL_WIDTH;
  const splitTaskCenterChromeMinWidth = effectiveChatPanelMinWidth || "560px";
  const taskCenterChromeStyle = shouldSplitTaskCenterChrome
    ? ({
        background: TASK_CENTER_CHROME_RAIL_SURFACE,
        width: splitTaskCenterChromeWidth,
        minWidth: splitTaskCenterChromeMinWidth,
      } as const)
    : shouldDetachTaskCenterToolbar
      ? ({
          background: TASK_CENTER_CHROME_RAIL_SURFACE,
          width: DEFAULT_TASK_CENTER_HOME_CHROME_WIDTH,
        } as const)
      : ({ background: TASK_CENTER_CHROME_RAIL_SURFACE } as const);
  const taskCenterChromeLayout = shouldSplitTaskCenterChrome
    ? "split"
    : shouldDetachTaskCenterToolbar
      ? "detached"
      : "stacked";
  const chatPanelTopInset = hasThreadHeader
    ? THREAD_WORKSPACE_HEADER_TOP_INSET
    : shouldSplitTaskCenterChrome
      ? TASK_CENTER_SPLIT_CHROME_TOP_INSET
      : shouldDetachTaskCenterToolbar
        ? TASK_CENTER_DETACHED_TOOLBAR_TOP_INSET
        : "0px";
  const canvasPanelTopInset = hasThreadHeader
    ? THREAD_WORKSPACE_HEADER_TOP_INSET
    : shouldSplitTaskCenterChrome || shouldDetachTaskCenterToolbar
      ? TASK_CENTER_SPLIT_WORKBENCH_TOP_INSET
      : "0px";
  const taskCenterChromeNode = shouldRenderTaskCenterChrome ? (
    <div
      className={
        shouldSplitTaskCenterChrome
          ? "absolute left-0 top-0 z-20 shrink-0 overflow-visible bg-[color:var(--lime-chrome-rail)] dark:bg-slate-900"
          : "relative z-20 shrink-0 overflow-visible bg-[color:var(--lime-chrome-rail)] dark:bg-slate-900"
      }
      data-testid="task-center-chrome-shell"
      data-layout={taskCenterChromeLayout}
      data-split-left-width={
        shouldSplitTaskCenterChrome ? splitTaskCenterChromeWidth : undefined
      }
      data-split-left-min-width={
        shouldSplitTaskCenterChrome ? splitTaskCenterChromeMinWidth : undefined
      }
      data-detached-right-reserve={
        shouldDetachTaskCenterToolbar
          ? TASK_CENTER_TOP_TOOLBAR_RESERVE_WIDTH
          : undefined
      }
      data-detached-left-width={
        shouldDetachTaskCenterToolbar
          ? DEFAULT_TASK_CENTER_HOME_CHROME_WIDTH
          : undefined
      }
      style={taskCenterChromeStyle}
    >
      {navbarNode}
      <div className="relative z-10 flex min-h-[42px] shrink-0 items-stretch overflow-hidden border-b border-[color:var(--lime-chrome-divider)] bg-[color:var(--lime-chrome-tab-active-surface)]">
        <div className="min-w-0 flex-1">{taskCenterTabsNode}</div>
        {taskCenterUtilityToolbarNode &&
        !shouldSplitTaskCenterChrome &&
        !shouldDetachTaskCenterToolbar ? (
          <div
            className="flex min-w-0 max-w-[42%] shrink-0 items-center justify-end overflow-hidden border-l border-[color:var(--lime-chrome-divider)] bg-[color:var(--lime-chrome-tab-active-surface)] px-2 lg:hidden"
            data-testid="task-center-utility-toolbar-host"
          >
            {taskCenterUtilityToolbarNode}
          </div>
        ) : null}
      </div>
    </div>
  ) : null;

  return (
    <MainArea
      $compact={compactChrome}
      $taskCenterSurface={Boolean(taskCenterChromeNode) || hasThreadHeader}
      data-testid="workspace-main-area"
      data-layout-mode={effectiveLayoutMode}
      data-has-right-surface={hasRightSurfaceContent ? "true" : "false"}
      data-floating-input-overlay={showFloatingInputOverlay ? "true" : "false"}
    >
      {hasThreadHeader ? (
        <div
          className="absolute inset-x-0 top-0 z-20 h-[52px] min-w-0"
          data-testid="thread-workspace-header-host"
        >
          {threadHeaderNode}
        </div>
      ) : shouldAutoHideNavbar ? (
        <AutoHideNavbarBackdrop
          type="button"
          $visible={isAutoHideNavbarVisible}
          data-testid="workspace-navbar-backdrop"
          data-visible={isAutoHideNavbarVisible ? "true" : "false"}
          aria-label={t("agentChat.navbar.autoHide.closeTopTools")}
          onClick={() => {
            setNavbarOpen(false);
          }}
        />
      ) : null}
      {hasThreadHeader ? null : shouldAutoHideNavbar ? (
        <AutoHideNavbarHost
          data-testid="workspace-navbar-auto-hide-shell"
          data-visible={isAutoHideNavbarVisible ? "true" : "false"}
        >
          {shouldRenderRevealHandle ? (
            <AutoHideNavbarHandle
              type="button"
              $visible={isAutoHideNavbarVisible}
              data-testid="workspace-navbar-reveal-handle"
              aria-label={t("agentChat.navbar.autoHide.expandTopTools")}
              aria-expanded={isAutoHideNavbarVisible}
              onClick={() => {
                setNavbarOpen(true);
              }}
            >
              <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
            </AutoHideNavbarHandle>
          ) : null}
          <AutoHideNavbarPanel
            $visible={isAutoHideNavbarVisible}
            data-testid="workspace-navbar-auto-hide-panel"
            data-visible={isAutoHideNavbarVisible ? "true" : "false"}
          >
            {navbarNode}
          </AutoHideNavbarPanel>
        </AutoHideNavbarHost>
      ) : taskCenterChromeNode ? (
        taskCenterChromeNode
      ) : taskCenterUtilityToolbarNode ? null : (
        navbarNode
      )}
      {!hasThreadHeader && shouldAutoHideNavbar ? taskCenterTabsNode : null}
      {contentSyncNoticeNode}
      {!shouldAutoHideNavbar &&
      taskCenterUtilityToolbarNode &&
      shouldDetachTaskCenterToolbar ? (
        <div
          className="absolute right-0 top-0 z-10 flex h-[42px] min-w-0 items-end justify-end overflow-visible px-4 pb-1"
          data-testid="task-center-home-top-toolbar-host"
          data-width-policy="content-adaptive"
          style={{ maxWidth: TASK_CENTER_TOP_TOOLBAR_MAX_WIDTH }}
        >
          {taskCenterUtilityToolbarNode}
        </div>
      ) : null}
      {shouldSplitTaskCenterChrome && taskCenterUtilityToolbarNode ? (
        <div
          className="absolute right-0 top-0 z-20 flex h-[42px] min-w-0 items-end justify-end overflow-visible px-4 pb-1 [left:var(--task-center-split-left)]"
          data-testid="task-center-workbench-top-toolbar-host"
          style={
            {
              "--task-center-split-left": splitTaskCenterChromeWidth,
            } as CSSProperties
          }
        >
          {taskCenterUtilityToolbarNode}
        </div>
      ) : null}
      <GeneralWorkbenchLayoutShell
        $bottomInset={shellBottomInset}
        $taskCenterSurface={Boolean(taskCenterChromeNode) || hasThreadHeader}
      >
        <LayoutTransitionRenderGate
          mode={effectiveLayoutMode}
          chatContent={chatContent}
          canvasContent={effectiveCanvasContent}
          chatPanelWidth={effectiveChatPanelWidth}
          chatPanelMinWidth={effectiveChatPanelMinWidth}
          chatPanelTopInset={chatPanelTopInset}
          canvasPanelTopInset={canvasPanelTopInset}
          forceOpenChatPanel={hasPendingA2UIForm}
        />
      </GeneralWorkbenchLayoutShell>
      {generalWorkbenchDialog}
      {showFloatingInputOverlay ? (
        <GeneralWorkbenchInputOverlay
          $bottomInset={shellBottomInset}
          data-testid="general-workbench-input-overlay"
          data-action-owner="composer"
          data-bottom-inset={shellBottomInset}
        >
          {inputbarNode}
        </GeneralWorkbenchInputOverlay>
      ) : null}
      {taskCenterShellPanelNode}
    </MainArea>
  );
}
