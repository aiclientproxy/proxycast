import { useState, type ComponentProps, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { ThreadGoal } from "@limecloud/app-server-client";
import { AlertTriangle, CheckCircle2, Info, Loader2 } from "lucide-react";
import { StepProgress } from "@/components/workspace/layout/StepProgress";
import type {
  A2UIFormData,
  A2UIResponse,
} from "@/components/workspace/a2ui/types";
import { CanvasWorkbenchLayout } from "../components/CanvasWorkbenchLayout";
import { ChatNavbar } from "../components/ChatNavbar";
import {
  TaskCenterShellPanel,
  TASK_CENTER_SHELL_PANEL_DEFAULT_HEIGHT_PX,
  TASK_CENTER_SHELL_PANEL_MAX_HEIGHT_RATIO,
} from "../components/TaskCenterShellPanel";
import { TaskCenterUtilityToolbar } from "../components/TaskCenterUtilityToolbar";
import type { GeneralWorkbenchWorkflowStepInput } from "../components/generalWorkbenchWorkflowPanelViewModel";
import type { SidebarActivityLog } from "../hooks/useThemeContextWorkspace";
import type { GeneralWorkbenchCreationTaskEvent } from "../components/generalWorkbenchWorkflowData";
import type { GeneralWorkbenchTaskRailContextInput } from "../components/generalWorkbenchTaskRailViewModel";
import type { ConfirmResponse } from "../types";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import { CreationReplaySurfaceBanner } from "../components/CreationReplaySurfaceBanner";
import { EmptyState } from "../components/EmptyState";
import { MessageList } from "../components/MessageList";
import { WorkspaceMainArea } from "./WorkspaceMainArea";
import { WorkspacePendingA2UIPanel } from "./WorkspacePendingA2UIPanel";
import { buildWorkspaceNavbarProps } from "./chatSurfaceProps";
import type { SyncStatus } from "../hooks/useContentSync";
import type { A2UISubmissionNoticeData } from "./A2UISubmissionNotice";
import type { WorkspaceConversationLandingSurfaceRuntime } from "./useWorkspaceConversationLandingSurfaceRuntime";
import {
  ChatContainer,
  ChatContainerInner,
  ChatContent,
  ChatInputSlot,
  ContentSyncNotice,
  ContentSyncNoticeText,
  EntryBanner,
  EntryBannerClose,
  MessageViewport,
} from "./WorkspaceStyles";
import { ThreadWorkspaceHeader } from "./ThreadWorkspaceHeader";
import type { ThreadWorkspaceHeaderViewModel } from "./threadWorkspaceHeaderViewModel";

type WorkspaceMainAreaProps = Omit<
  ComponentProps<typeof WorkspaceMainArea>,
  | "navbarNode"
  | "contentSyncNoticeNode"
  | "forceCanvasMode"
  | "chatContent"
  | "canvasContent"
>;
type CanvasWorkbenchLayoutProps = ComponentProps<typeof CanvasWorkbenchLayout>;
type StepProgressProps = ComponentProps<typeof StepProgress>;
type MessageListProps = ComponentProps<typeof MessageList>;
type AgentNamespaceTranslation = (
  key: string,
  options?: Record<string, unknown>,
) => unknown;

interface WorkspaceChatContentParams {
  landingSurface: WorkspaceConversationLandingSurfaceRuntime;
  stepProgressProps?: StepProgressProps | null;
  showChatLayout: boolean;
  compactChrome: boolean;
  taskCenterSurface: boolean;
  contextWorkspaceEnabled: boolean;
  showFloatingInputOverlay: boolean;
  generalWorkbenchMessageViewportBottomPadding?: string;
  messageListProps: MessageListProps;
  showWorkspaceAlert: boolean;
  onSelectWorkspaceDirectory: () => void;
  onDismissWorkspaceAlert: () => void;
  pendingA2UIForm?: A2UIResponse | null;
  onPendingA2UISubmit?: (formData: A2UIFormData) => void;
  a2uiSubmissionNotice?: A2UISubmissionNoticeData | null;
  showInlineInputbar: boolean;
  inputbarNode: ReactNode;
  copy: WorkspaceChatContentCopy;
}

interface WorkspaceChatContentCopy {
  entryBannerClose: string;
  entryBannerCloseAria: string;
  workspaceMissing: string;
  workspaceReselect: string;
  workspaceDismissAria: string;
}

function resolveContentSyncNoticeMeta(status: Exclude<SyncStatus, "idle">): {
  label: string;
  Icon: typeof Loader2;
  animated?: boolean;
} {
  switch (status) {
    case "syncing":
      return {
        label: "正在同步到当前内容…",
        Icon: Loader2,
        animated: true,
      };
    case "success":
      return {
        label: "内容已同步",
        Icon: CheckCircle2,
      };
    case "error":
    default:
      return {
        label: "同步失败，将自动重试",
        Icon: AlertTriangle,
      };
  }
}

function renderWorkspaceChatContent({
  landingSurface,
  stepProgressProps,
  showChatLayout,
  compactChrome,
  taskCenterSurface,
  contextWorkspaceEnabled,
  showFloatingInputOverlay,
  generalWorkbenchMessageViewportBottomPadding,
  messageListProps,
  showWorkspaceAlert,
  onSelectWorkspaceDirectory,
  onDismissWorkspaceAlert,
  pendingA2UIForm,
  onPendingA2UISubmit,
  a2uiSubmissionNotice,
  showInlineInputbar,
  inputbarNode,
  copy,
}: WorkspaceChatContentParams): ReactNode {
  const pendingA2UISource = messageListProps.activePendingA2UISource ?? null;
  const hasPendingA2UIMessageTailPayload = Boolean(
    pendingA2UIForm || a2uiSubmissionNotice,
  );
  const shouldRenderPendingA2UIAsMessageTail =
    hasPendingA2UIMessageTailPayload &&
    (!pendingA2UISource ||
      pendingA2UISource.kind === "scene_gate" ||
      pendingA2UISource.kind === "service_skill" ||
      Boolean(a2uiSubmissionNotice));
  const leadingMessageContent =
    landingSurface.sceneAppExecutionSummaryCard ||
    stepProgressProps ||
    landingSurface.serviceSkillExecutionCard ? (
      <>
        {landingSurface.sceneAppExecutionSummaryCard}
        {stepProgressProps ? <StepProgress {...stepProgressProps} /> : null}
        {landingSurface.serviceSkillExecutionCard}
      </>
    ) : null;
  const pendingA2UIMessageTail = shouldRenderPendingA2UIAsMessageTail ? (
    <WorkspacePendingA2UIPanel
      pendingA2UIForm={pendingA2UIForm}
      onA2UISubmit={onPendingA2UISubmit}
      a2uiSubmissionNotice={a2uiSubmissionNotice}
      placement="message"
    />
  ) : null;
  const floatingInputTailSpacer =
    contextWorkspaceEnabled && showFloatingInputOverlay ? (
      <div
        aria-hidden="true"
        className="h-8 shrink-0 max-[960px]:h-16"
        data-testid="message-list-floating-input-spacer"
      />
    ) : null;
  const trailingMessageContent =
    messageListProps.trailingContent ||
    pendingA2UIMessageTail ||
    floatingInputTailSpacer ? (
      <>
        {messageListProps.trailingContent}
        {pendingA2UIMessageTail}
        {floatingInputTailSpacer}
      </>
    ) : null;

  const messageListNode = (
    <MessageList
      {...messageListProps}
      leadingContent={leadingMessageContent}
      trailingContent={trailingMessageContent}
      compactLeadingSpacing={contextWorkspaceEnabled}
    />
  );

  return (
    <ChatContainer>
      <ChatContainerInner $taskCenterSurface={taskCenterSurface}>
        {landingSurface.entryBannerVisible &&
        landingSurface.entryBannerMessage ? (
          <EntryBanner data-testid="workspace-entry-banner">
            <Info className="h-4 w-4 shrink-0" />
            <span data-testid="workspace-entry-banner-text">
              {landingSurface.entryBannerMessage}
            </span>
            <EntryBannerClose
              type="button"
              onClick={landingSurface.onDismissEntryBanner}
              aria-label={copy.entryBannerCloseAria}
            >
              {copy.entryBannerClose}
            </EntryBannerClose>
          </EntryBanner>
        ) : null}

        {showChatLayout && landingSurface.creationReplaySurface ? (
          <CreationReplaySurfaceBanner
            surface={landingSurface.creationReplaySurface}
            className="mx-4 mb-2"
          />
        ) : null}

        {showChatLayout ? (
          <ChatContent $compact={compactChrome}>
            <>
              {contextWorkspaceEnabled ? (
                <MessageViewport
                  $bottomPadding={generalWorkbenchMessageViewportBottomPadding}
                >
                  {messageListNode}
                </MessageViewport>
              ) : (
                messageListNode
              )}
              {showWorkspaceAlert ? (
                <div className="mx-4 mb-2 flex items-center gap-2 rounded-[18px] border border-amber-200/90 bg-amber-50/86 px-3.5 py-2.5 text-sm text-amber-800 shadow-sm shadow-amber-950/5 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-300">
                  <span className="flex-1">{copy.workspaceMissing}</span>
                  <button
                    type="button"
                    onClick={onSelectWorkspaceDirectory}
                    className="shrink-0 rounded-xl border border-amber-200 bg-white/84 px-2.5 py-1 text-xs font-medium text-amber-900 transition hover:border-amber-300 hover:bg-white dark:bg-amber-800 dark:text-amber-100 dark:hover:bg-amber-700"
                  >
                    {copy.workspaceReselect}
                  </button>
                  <button
                    type="button"
                    onClick={onDismissWorkspaceAlert}
                    className="shrink-0 text-amber-600 hover:text-amber-900 dark:text-amber-400 dark:hover:text-amber-200"
                    aria-label={copy.workspaceDismissAria}
                  >
                    ✕
                  </button>
                </div>
              ) : null}
              {showInlineInputbar ? (
                <ChatInputSlot data-testid="workspace-inline-input-slot">
                  {inputbarNode}
                </ChatInputSlot>
              ) : null}
            </>
          </ChatContent>
        ) : (
          <EmptyState {...landingSurface.emptyStateProps} />
        )}
      </ChatContainerInner>
    </ChatContainer>
  );
}

interface WorkspaceConversationSceneProps extends WorkspaceMainAreaProps {
  threadHeader?: ThreadWorkspaceHeaderViewModel | null;
  landingSurface: WorkspaceConversationLandingSurfaceRuntime;
  stepProgressProps?: WorkspaceChatContentParams["stepProgressProps"];
  showChatLayout: boolean;
  contextWorkspaceEnabled: boolean;
  generalWorkbenchMessageViewportBottomPadding?: string;
  messageListProps: WorkspaceChatContentParams["messageListProps"];
  taskRail?: {
    sessionId?: string | null;
    workflowSteps: GeneralWorkbenchWorkflowStepInput[];
    messages: MessageListProps["messages"];
    activityLogs?: SidebarActivityLog[];
    creationTaskEvents?: GeneralWorkbenchCreationTaskEvent[];
    pendingActions?: MessageListProps["pendingActions"];
    submittedActionsInFlight?: MessageListProps["submittedActionsInFlight"];
    threadItems?: MessageListProps["threadItems"];
    threadGoal?: ThreadGoal | null;
    threadRead?: MessageListProps["threadRead"];
    executionRuntime?: AgentSessionExecutionRuntime | null;
    canonicalChildren?: MessageListProps["canonicalChildren"];
    context?: GeneralWorkbenchTaskRailContextInput;
    onOpenOutput?: (path: string) => void | Promise<void>;
    onRespondToAction?: (response: ConfirmResponse) => void | Promise<void>;
  };
  workspaceAlertVisible: boolean;
  onSelectWorkspaceDirectory: () => void;
  onDismissWorkspaceAlert: () => void;
  pendingA2UIForm?: WorkspaceChatContentParams["pendingA2UIForm"];
  onPendingA2UISubmit?: WorkspaceChatContentParams["onPendingA2UISubmit"];
  a2uiSubmissionNotice?: WorkspaceChatContentParams["a2uiSubmissionNotice"];
  shouldHideGeneralWorkbenchInputForTheme: boolean;
  projectId: string | null;
  openedProjects?: ComponentProps<typeof ChatNavbar>["openedProjects"];
  projectRootPath?: string | null;
  onProjectChange?: (projectId: string | null) => void;
  onCloseProject?: ComponentProps<typeof ChatNavbar>["onCloseProject"];
  workspaceType?: ComponentProps<typeof ChatNavbar>["workspaceType"];
  deferWorkspaceListLoad?: ComponentProps<
    typeof ChatNavbar
  >["deferWorkspaceListLoad"];
  onOpenSettings?: () => void;
  taskCenterTabsNode?: ReactNode;
  navbarVisible: boolean;
  isRunning: boolean;
  navbarChrome: ComponentProps<typeof ChatNavbar>["chrome"];
  navbarContextVariant?: "default" | "task-center";
  onBackToProjectManagement?: ComponentProps<
    typeof ChatNavbar
  >["onBackToProjectManagement"];
  onBackToResources?: ComponentProps<typeof ChatNavbar>["onBackToResources"];
  onToggleCanvas?: ComponentProps<typeof ChatNavbar>["onToggleCanvas"];
  onBackHome?: ComponentProps<typeof ChatNavbar>["onBackHome"];
  showHarnessToggle: boolean;
  harnessPanelVisible: boolean;
  onToggleHarnessPanel?: ComponentProps<
    typeof ChatNavbar
  >["onToggleHarnessPanel"];
  showExpertInfoToggle?: boolean;
  expertInfoPanelVisible?: boolean;
  onToggleExpertInfoPanel?: () => void;
  harnessPendingCount: number;
  harnessAttentionLevel: ComponentProps<
    typeof ChatNavbar
  >["harnessAttentionLevel"];
  harnessToggleLabel?: ComponentProps<typeof ChatNavbar>["harnessToggleLabel"];
  showContextCompactionAction?: ComponentProps<
    typeof ChatNavbar
  >["showContextCompactionAction"];
  contextCompactionRunning?: ComponentProps<
    typeof ChatNavbar
  >["contextCompactionRunning"];
  onCompactContext?: ComponentProps<typeof ChatNavbar>["onCompactContext"];
  isThemeWorkbench: boolean;
  contentId?: string;
  syncStatus: SyncStatus;
  hasLiveCanvasPreviewContent: boolean;
  liveCanvasPreview: ReactNode;
  rightSurfaceContent?: ReactNode;
  rightSurfaceLaunchers?: ComponentProps<
    typeof TaskCenterUtilityToolbar
  >["rightSurfaceLaunchers"];
  rightSurfaceObjectCanvasOpen?: boolean;
  onToggleRightSurfaceObjectCanvas?: () => void;
  rightSurfaceBrowserOpen?: boolean;
  onToggleRightSurfaceBrowser?: () => void;
  rightSurfaceFilesOpen?: boolean;
  onToggleRightSurfaceFiles?: () => void;
  rightSurfaceTraceOpen?: boolean;
  onToggleRightSurfaceTrace?: () => void;
  rightSurfaceShellOpen?: boolean;
  onToggleRightSurfaceShell?: () => void;
  currentImageWorkbenchActive: boolean;
  shouldShowCanvasLoadingState: boolean;
  canvasWorkbenchLayoutProps: CanvasWorkbenchLayoutProps;
}

export function WorkspaceConversationScene({
  threadHeader,
  landingSurface,
  stepProgressProps,
  showChatLayout,
  compactChrome,
  contextWorkspaceEnabled,
  generalWorkbenchMessageViewportBottomPadding,
  messageListProps,
  taskRail,
  workspaceAlertVisible,
  onSelectWorkspaceDirectory,
  onDismissWorkspaceAlert,
  pendingA2UIForm,
  onPendingA2UISubmit,
  a2uiSubmissionNotice,
  shouldHideGeneralWorkbenchInputForTheme,
  inputbarNode,
  contentId,
  projectId,
  openedProjects,
  projectRootPath,
  onProjectChange,
  onCloseProject,
  workspaceType,
  deferWorkspaceListLoad,
  onOpenSettings,
  taskCenterTabsNode,
  navbarVisible,
  isRunning,
  navbarChrome,
  navbarContextVariant = "default",
  onBackToProjectManagement,
  onBackToResources,
  isThemeWorkbench,
  layoutMode,
  onToggleCanvas,
  onBackHome,
  showHarnessToggle,
  harnessPanelVisible,
  onToggleHarnessPanel,
  showExpertInfoToggle,
  expertInfoPanelVisible,
  onToggleExpertInfoPanel,
  harnessPendingCount,
  harnessAttentionLevel,
  harnessToggleLabel,
  showContextCompactionAction,
  contextCompactionRunning,
  onCompactContext,
  syncStatus,
  hasLiveCanvasPreviewContent,
  liveCanvasPreview,
  rightSurfaceContent,
  rightSurfaceLaunchers,
  rightSurfaceObjectCanvasOpen,
  onToggleRightSurfaceObjectCanvas,
  rightSurfaceBrowserOpen,
  onToggleRightSurfaceBrowser,
  rightSurfaceFilesOpen,
  onToggleRightSurfaceFiles,
  rightSurfaceTraceOpen,
  onToggleRightSurfaceTrace,
  rightSurfaceShellOpen,
  onToggleRightSurfaceShell,
  currentImageWorkbenchActive,
  shouldShowCanvasLoadingState,
  canvasWorkbenchLayoutProps,
  shellBottomInset,
  chatPanelWidth,
  chatPanelMinWidth,
  generalWorkbenchDialog,
  showFloatingInputOverlay,
  hasPendingA2UIForm,
}: WorkspaceConversationSceneProps) {
  const { t } = useTranslation("agent");
  const agentT = t as unknown as AgentNamespaceTranslation;
  const text = (key: string) =>
    String(agentT(`agentChat.workspaceConversation.${key}`));
  const chatContent = renderWorkspaceChatContent({
    landingSurface,
    stepProgressProps,
    showChatLayout,
    compactChrome,
    taskCenterSurface: navbarContextVariant === "task-center",
    contextWorkspaceEnabled,
    showFloatingInputOverlay,
    generalWorkbenchMessageViewportBottomPadding,
    messageListProps,
    showWorkspaceAlert: workspaceAlertVisible,
    onSelectWorkspaceDirectory,
    onDismissWorkspaceAlert,
    pendingA2UIForm,
    onPendingA2UISubmit,
    a2uiSubmissionNotice,
    showInlineInputbar:
      !contextWorkspaceEnabled && !shouldHideGeneralWorkbenchInputForTheme,
    inputbarNode,
    copy: {
      entryBannerClose: text("entryBanner.close"),
      entryBannerCloseAria: text("entryBanner.closeAria"),
      workspaceMissing: text("workspaceAlert.missing"),
      workspaceReselect: text("workspaceAlert.reselect"),
      workspaceDismissAria: text("workspaceAlert.dismissAria"),
    },
  });
  const [localShellPanelOpen, setLocalShellPanelOpen] = useState(false);
  const [shellPanelHeightPx, setShellPanelHeightPx] = useState(
    TASK_CENTER_SHELL_PANEL_DEFAULT_HEIGHT_PX,
  );
  const [shellPanelMaximized, setShellPanelMaximized] = useState(false);
  const shouldUseTaskCenterUtilityToolbar =
    navbarContextVariant === "task-center" && Boolean(taskCenterTabsNode);
  const shellManagedByRightSurface = Boolean(onToggleRightSurfaceShell);
  const shellPanelOpen = shellManagedByRightSurface
    ? Boolean(rightSurfaceShellOpen)
    : localShellPanelOpen;
  const bottomShellPanelOpen =
    localShellPanelOpen &&
    shouldUseTaskCenterUtilityToolbar &&
    !shellManagedByRightSurface;
  const effectiveShellPanelHeightPx = bottomShellPanelOpen
    ? shellPanelMaximized && typeof window !== "undefined"
      ? Math.max(
          TASK_CENTER_SHELL_PANEL_DEFAULT_HEIGHT_PX,
          Math.floor(
            window.innerHeight * TASK_CENTER_SHELL_PANEL_MAX_HEIGHT_RATIO,
          ),
        )
      : shellPanelHeightPx
    : 0;
  const effectiveShellBottomInset = bottomShellPanelOpen
    ? `calc(${shellBottomInset} + ${effectiveShellPanelHeightPx}px)`
    : shellBottomInset;
  const taskCenterUtilityToolbarNode = shouldUseTaskCenterUtilityToolbar ? (
    <TaskCenterUtilityToolbar
      projectRootPath={projectRootPath}
      onProjectChange={onProjectChange}
      taskRail={taskRail}
      placement={layoutMode !== "chat" ? "workbench-header" : "task-strip"}
      showCanvasToggle={!isThemeWorkbench}
      isCanvasOpen={layoutMode !== "chat"}
      onToggleCanvas={onToggleCanvas}
      showHarnessToggle={showHarnessToggle}
      harnessPanelVisible={harnessPanelVisible}
      onToggleHarnessPanel={onToggleHarnessPanel}
      showExpertInfoToggle={showExpertInfoToggle}
      expertInfoPanelVisible={expertInfoPanelVisible}
      onToggleExpertInfoPanel={onToggleExpertInfoPanel}
      harnessPendingCount={harnessPendingCount}
      harnessAttentionLevel={harnessAttentionLevel ?? "idle"}
      harnessToggleLabel={harnessToggleLabel ?? "Harness"}
      shellPanelOpen={shellPanelOpen}
      onToggleObjectCanvasPanel={
        rightSurfaceObjectCanvasOpen || onToggleRightSurfaceObjectCanvas
          ? onToggleRightSurfaceObjectCanvas
          : undefined
      }
      onToggleBrowserPanel={
        rightSurfaceBrowserOpen || onToggleRightSurfaceBrowser
          ? onToggleRightSurfaceBrowser
          : undefined
      }
      onToggleFilesPanel={
        rightSurfaceFilesOpen || onToggleRightSurfaceFiles
          ? onToggleRightSurfaceFiles
          : undefined
      }
      onToggleTracePanel={
        rightSurfaceTraceOpen || onToggleRightSurfaceTrace
          ? onToggleRightSurfaceTrace
          : undefined
      }
      rightSurfaceLaunchers={rightSurfaceLaunchers}
      onToggleShellPanel={() => {
        if (onToggleRightSurfaceShell) {
          onToggleRightSurfaceShell();
          return;
        }
        setLocalShellPanelOpen((current) => !current);
      }}
    />
  ) : null;
  const shouldRenderThreadHeader = Boolean(
    threadHeader && navbarContextVariant === "task-center" && showChatLayout,
  );
  const threadHeaderNode =
    shouldRenderThreadHeader && threadHeader ? (
      <ThreadWorkspaceHeader
        {...threadHeader}
        actions={taskCenterUtilityToolbarNode}
      />
    ) : null;
  const chatNavbarProps = buildWorkspaceNavbarProps({
    visible: navbarVisible,
    isRunning,
    chrome: navbarChrome,
    navbarContextVariant,
    onBackToProjectManagement,
    onBackToResources,
    showCanvasToggle: !isThemeWorkbench,
    isCanvasOpen: layoutMode !== "chat",
    onToggleCanvas,
    projectId,
    openedProjects,
    onProjectChange,
    onCloseProject,
    deferWorkspaceListLoad,
    workspaceType,
    onBackHome,
    showHarnessToggle,
    harnessPanelVisible,
    onToggleHarnessPanel,
    harnessPendingCount,
    harnessAttentionLevel,
    harnessToggleLabel,
    showContextCompactionAction,
    contextCompactionRunning,
    onCompactContext,
    onOpenSettings,
  });

  const navbarNode = chatNavbarProps ? (
    <ChatNavbar {...chatNavbarProps} />
  ) : null;
  const shouldShowContentSyncNotice =
    !isThemeWorkbench &&
    Boolean(contentId) &&
    (syncStatus === "syncing" ||
      syncStatus === "success" ||
      syncStatus === "error");
  const contentSyncNoticeNode = shouldShowContentSyncNotice
    ? (() => {
        const notice = resolveContentSyncNoticeMeta(syncStatus);
        const NoticeIcon = notice.Icon;

        return (
          <ContentSyncNotice $status={syncStatus}>
            <NoticeIcon
              className={
                notice.animated ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"
              }
            />
            <ContentSyncNoticeText>{notice.label}</ContentSyncNoticeText>
          </ContentSyncNotice>
        );
      })()
    : null;
  const taskCenterShellPanelNode = bottomShellPanelOpen ? (
    <TaskCenterShellPanel
      heightPx={effectiveShellPanelHeightPx}
      maximized={shellPanelMaximized}
      projectRootPath={projectRootPath}
      onClose={() => {
        setLocalShellPanelOpen(false);
      }}
      onHeightChange={(heightPx) => {
        setShellPanelMaximized(false);
        setShellPanelHeightPx(heightPx);
      }}
      onToggleMaximize={() => {
        setShellPanelMaximized((current) => !current);
      }}
    />
  ) : null;
  const canvasContent =
    !liveCanvasPreview ? null : currentImageWorkbenchActive ||
      shouldShowCanvasLoadingState ? (
      liveCanvasPreview
    ) : (
      <CanvasWorkbenchLayout
        {...canvasWorkbenchLayoutProps}
        topRightTools={null}
      />
    );
  const forceCanvasMode = Boolean(
    isThemeWorkbench && hasLiveCanvasPreviewContent,
  );

  return (
    <>
      <WorkspaceMainArea
        compactChrome={compactChrome}
        navbarNode={navbarNode}
        threadHeaderNode={threadHeaderNode}
        taskCenterUtilityToolbarNode={
          shouldRenderThreadHeader ? null : taskCenterUtilityToolbarNode
        }
        taskCenterTabsNode={taskCenterTabsNode}
        taskCenterShellPanelNode={taskCenterShellPanelNode}
        contentSyncNoticeNode={contentSyncNoticeNode}
        shellBottomInset={effectiveShellBottomInset}
        layoutMode={layoutMode}
        forceCanvasMode={forceCanvasMode}
        chatContent={chatContent}
        canvasContent={canvasContent}
        rightSurfaceContent={rightSurfaceContent}
        chatPanelWidth={chatPanelWidth}
        chatPanelMinWidth={chatPanelMinWidth}
        generalWorkbenchDialog={generalWorkbenchDialog}
        showFloatingInputOverlay={showFloatingInputOverlay}
        hasPendingA2UIForm={hasPendingA2UIForm}
        inputbarNode={inputbarNode}
      />
    </>
  );
}
