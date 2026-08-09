import type { ComponentProps, ReactNode } from "react";
import { useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { startReview } from "@/lib/api/review";
import type { ThreadGoal } from "@limecloud/app-server-client";
import { StepProgress } from "@/components/workspace/layout/StepProgress";
import { useWorkspaceNavigationActions } from "./useWorkspaceNavigationActions";
import { useWorkspaceInputbarSceneRuntime } from "./useWorkspaceInputbarSceneRuntime";
import { useWorkspaceCanvasSceneRuntime } from "./useWorkspaceCanvasSceneRuntime";
import { MessageList } from "../components/MessageList";
import type { WriteArtifactContext } from "../types";
import type { PendingA2UISource } from "../types";
import type { LayoutMode } from "@/lib/workspace/workbenchContract";
import type { TaskFile } from "../components/TaskFiles";
import type { InputbarSendHandler } from "../components/Inputbar/inputbarSendPayload";
import type { WorkspacePathMissingState } from "../hooks/agentChatShared";
import type { SidebarActivityLog } from "../hooks/useThemeContextWorkspace";
import type { SyncStatus } from "../hooks/useContentSync";
import type { ArtifactTimelineOpenTarget } from "../utils/artifactTimelineNavigation";
import { buildStepProgressProps } from "./chatSurfaceProps";
import { WorkspaceConversationScene } from "./WorkspaceConversationScene";
import { buildWorkspaceConversationCodingViews } from "./workspaceConversationCodingViews";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type { AgentTodoItem } from "@/lib/api/agentRuntime/sessionTypes";
import type { CodingWorkbenchRecoveryContext } from "./codingWorkbenchRecovery";
import type { GeneralWorkbenchCreationTaskEvent } from "../components/generalWorkbenchWorkflowData";
import type { GeneralWorkbenchTaskRailContextInput } from "../components/generalWorkbenchTaskRailViewModel";
import type { CanonicalChildThreadSummary } from "../projection/canonicalChildThreadSummary";
import { useWorkspaceTaskRailRuntime } from "./useWorkspaceTaskRailRuntime";
import { useSessionRuntimeProjectionDeferral } from "./useSessionRuntimeProjectionDeferral";
import {
  buildQuotedReplyText,
  buildWorkspaceHeaderView,
} from "./workspaceConversationSceneViewModel";
import {
  useWorkspaceConversationLandingSessionRuntime,
  type WorkspaceConversationLandingSurfaceRuntime,
} from "./useWorkspaceConversationLandingSurfaceRuntime";
import {
  buildWorkspaceConversationRightSurfaceSceneProps,
  type WorkspaceConversationRightSurfaceChromeRuntime,
} from "./workspaceConversationRightSurfaceChrome";

type InputbarScene = Pick<
  ReturnType<typeof useWorkspaceInputbarSceneRuntime>,
  | "inputbarNode"
  | "generalWorkbenchDialog"
  | "runtimeToolAvailability"
  | "knowledgePackSelection"
  | "knowledgePackOptions"
  | "onToggleKnowledgePack"
  | "onSelectKnowledgePack"
  | "onToggleKnowledgeCompanionPack"
  | "onStartKnowledgeOrganize"
  | "onManageKnowledgePacks"
>;
type CanvasScene = Pick<
  ReturnType<typeof useWorkspaceCanvasSceneRuntime>,
  | "hasLiveCanvasPreviewContent"
  | "liveCanvasPreview"
  | "shouldShowCanvasLoadingState"
  | "canvasWorkbenchDefaultPreview"
  | "handleOpenCanvasWorkbenchPath"
  | "handleRevealCanvasWorkbenchPath"
  | "handleCloseCanvasWorkbench"
>;
type WorkspaceConversationSceneProps = ComponentProps<
  typeof WorkspaceConversationScene
>;
type CanvasWorkbenchLayoutProps = NonNullable<
  WorkspaceConversationSceneProps["canvasWorkbenchLayoutProps"]
>;
type MessageListProps = ComponentProps<typeof MessageList>;
interface ConversationScenePresentationParams {
  scene: Omit<
    WorkspaceConversationSceneProps,
    | "workspaceAlertVisible"
    | "projectId"
    | "canvasWorkbenchLayoutProps"
    | "stepProgressProps"
    | "messageListProps"
  > & {
    projectId: string | null | undefined;
  };
  stepProgress: {
    hidden: boolean;
    isSpecializedThemeMode: boolean;
    hasMessages: boolean;
    steps: ComponentProps<typeof StepProgress>["steps"];
    currentIndex: ComponentProps<typeof StepProgress>["currentIndex"];
    onStepClick: NonNullable<
      ComponentProps<typeof StepProgress>["onStepClick"]
    >;
  };
  messageList: MessageListProps;
  workspaceAlert: {
    workspacePathMissing: boolean;
    workspaceHealthError: boolean;
  };
  canvasWorkbenchLayout: Omit<
    CanvasWorkbenchLayoutProps,
    "workspaceUnavailable"
  >;
}
interface WorkspaceConversationScenePresentationResult {
  workspaceAlertVisible: boolean;
  mainAreaNode: ReactNode;
}
type NavigationActions = Pick<
  ReturnType<typeof useWorkspaceNavigationActions>,
  | "handleDismissEntryBanner"
  | "handleWorkspaceAlertSelectDirectory"
  | "handleDismissWorkspaceAlert"
  | "handleManageProviders"
  | "handleOpenExecutionPolicySettings"
  | "handleProjectChange"
  | "handleOpenAppearanceSettings"
  | "handleOpenRuntimeMemoryWorkbench"
  | "handleOpenChannels"
  | "handleOpenChromeRelay"
  | "handleBackToResources"
  | "handleCompactContext"
>;
interface ShellChromeRuntime {
  showChatLayout: boolean;
  isWorkspaceCompactChrome: boolean;
  workflowLayoutBottomSpacing: {
    shellBottomInset: string;
    messageViewportBottomPadding: string;
  };
  shouldHideGeneralWorkbenchInputForTheme: boolean;
  shouldRenderTopBar: boolean;
  layoutTransitionChatPanelWidth?: string;
  layoutTransitionChatPanelMinWidth?: string;
  shouldShowGeneralWorkbenchFloatingInputOverlay: boolean;
  shouldRenderInlineA2UI: boolean;
}

export interface WorkspaceConversationMessageListRuntime {
  emptyStateVariant?: "none" | "task-center";
  quoteInput: string;
  onQuoteInputChange: (value: string) => void;
  providerType: MessageListProps["providerType"];
  model: string | null | undefined;
  reasoningEffort?: string | null;
  accessMode: GeneralWorkbenchTaskRailContextInput["accessMode"];
  messages: ConversationScenePresentationParams["messageList"]["messages"];
  turns: ConversationScenePresentationParams["messageList"]["turns"];
  threadItems: ConversationScenePresentationParams["messageList"]["threadItems"];
  todoItems?: AgentTodoItem[];
  currentTurnId: ConversationScenePresentationParams["messageList"]["currentTurnId"];
  threadRead: ConversationScenePresentationParams["messageList"]["threadRead"];
  executionRuntime?: AgentSessionExecutionRuntime | null;
  pendingActions: NonNullable<
    ConversationScenePresentationParams["messageList"]["pendingActions"]
  >;
  submittedActionsInFlight: NonNullable<
    ConversationScenePresentationParams["messageList"]["submittedActionsInFlight"]
  >;
  canonicalChildren?: NonNullable<
    ConversationScenePresentationParams["messageList"]["canonicalChildren"]
  >;
  sessionHistoryWindow?: ConversationScenePresentationParams["messageList"]["sessionHistoryWindow"];
  onLoadFullHistory?: ConversationScenePresentationParams["messageList"]["onLoadFullHistory"];
  isSending: ConversationScenePresentationParams["messageList"]["isSending"];
  onInterruptCurrentTurn: ConversationScenePresentationParams["messageList"]["onInterruptCurrentTurn"];
  onReplayPendingRequest: ConversationScenePresentationParams["messageList"]["onReplayPendingRequest"];
  onDeleteMessage: ConversationScenePresentationParams["messageList"]["onDeleteMessage"];
  onEditMessage: ConversationScenePresentationParams["messageList"]["onEditMessage"];
  onA2UISubmit: ConversationScenePresentationParams["messageList"]["onA2UISubmit"];
  onWriteFile: (
    content: string,
    fileName: string,
    context?: WriteArtifactContext,
  ) => void | Promise<void>;
  onFileClick: ConversationScenePresentationParams["messageList"]["onFileClick"];
  onOpenArtifactFromTimeline: (target: ArtifactTimelineOpenTarget) => void;
  onOpenSavedSiteContent: ConversationScenePresentationParams["messageList"]["onOpenSavedSiteContent"];
  onArtifactClick: ConversationScenePresentationParams["messageList"]["onArtifactClick"];
  onOpenUrlPreview?: ConversationScenePresentationParams["messageList"]["onOpenUrlPreview"];
  onOpenMessagePreview?: ConversationScenePresentationParams["messageList"]["onOpenMessagePreview"];
  onSaveMessageAsSkill?: ConversationScenePresentationParams["messageList"]["onSaveMessageAsSkill"];
  onSaveMessageAsKnowledge?: ConversationScenePresentationParams["messageList"]["onSaveMessageAsKnowledge"];
  onOpenSubagentSession: ConversationScenePresentationParams["messageList"]["onOpenSubagentSession"];
  onPermissionResponse: ConversationScenePresentationParams["messageList"]["onPermissionResponse"];
  onRefreshSessionReadModel?: () => void | Promise<unknown>;
  pendingPromotedA2UIActionRequest: unknown;
  collapseCodeBlocks: ConversationScenePresentationParams["messageList"]["collapseCodeBlocks"];
  shouldCollapseCodeBlock: ConversationScenePresentationParams["messageList"]["shouldCollapseCodeBlock"];
  onCodeBlockClick: ConversationScenePresentationParams["messageList"]["onCodeBlockClick"];
  focusedTimelineItemId: string | null;
  timelineFocusRequestKey: number;
}

function renderWorkspaceConversationScene({
  scene,
  stepProgress,
  messageList,
  workspaceAlert,
  canvasWorkbenchLayout,
}: ConversationScenePresentationParams): WorkspaceConversationScenePresentationResult {
  const stepProgressProps = buildStepProgressProps(stepProgress);
  const workspaceAlertVisible = Boolean(
    workspaceAlert.workspacePathMissing || workspaceAlert.workspaceHealthError,
  );

  const canvasWorkbenchLayoutProps: CanvasWorkbenchLayoutProps = {
    ...canvasWorkbenchLayout,
    workspaceUnavailable: workspaceAlertVisible,
  };

  return {
    workspaceAlertVisible,
    mainAreaNode: (
      <WorkspaceConversationScene
        {...scene}
        stepProgressProps={stepProgressProps}
        messageListProps={messageList}
        workspaceAlertVisible={workspaceAlertVisible}
        projectId={scene.projectId ?? null}
        canvasWorkbenchLayoutProps={canvasWorkbenchLayoutProps}
      />
    ),
  };
}

const EMPTY_PROJECTED_TURNS: NonNullable<
  ConversationScenePresentationParams["messageList"]["turns"]
> = [];
const EMPTY_PROJECTED_THREAD_ITEMS: NonNullable<
  ConversationScenePresentationParams["messageList"]["threadItems"]
> = [];
const EMPTY_PROJECTED_PENDING_ACTIONS: NonNullable<
  ConversationScenePresentationParams["messageList"]["pendingActions"]
> = [];
const EMPTY_PROJECTED_SUBMITTED_ACTIONS: NonNullable<
  ConversationScenePresentationParams["messageList"]["submittedActionsInFlight"]
> = [];
interface UseWorkspaceConversationSceneRuntimeParams {
  threadHeader?: WorkspaceConversationSceneProps["threadHeader"];
  navbarContextVariant?: "default" | "task-center";
  navigationActions: NavigationActions;
  inputbarScene: InputbarScene;
  canvasScene: CanvasScene;
  handleSendFromEmptyState: InputbarSendHandler;
  landingSurface: WorkspaceConversationLandingSurfaceRuntime;
  shellChromeRuntime: ShellChromeRuntime;
  contextWorkspaceEnabled: boolean;
  activeTheme: string;
  contentId: ConversationScenePresentationParams["scene"]["contentId"];
  projectId: string | null;
  openedProjects?: ConversationScenePresentationParams["scene"]["openedProjects"];
  onCloseProject?: ConversationScenePresentationParams["scene"]["onCloseProject"];
  deferWorkspaceListLoad?: ConversationScenePresentationParams["scene"]["deferWorkspaceListLoad"];
  taskCenterTabsNode?: ConversationScenePresentationParams["scene"]["taskCenterTabsNode"];
  suppressNavbarUtilityActions?: boolean;
  topBarChrome: ConversationScenePresentationParams["scene"]["navbarChrome"];
  onBackToProjectManagement?: ConversationScenePresentationParams["scene"]["onBackToProjectManagement"];
  fromResources: boolean;
  handleBackHome: ConversationScenePresentationParams["scene"]["onBackHome"];
  rightSurfaceChrome: WorkspaceConversationRightSurfaceChromeRuntime;
  isRestoringSession: boolean;
  sessionId: string | null | undefined;
  threadGoal?: ThreadGoal | null;
  syncStatus: SyncStatus;
  pendingA2UIForm: ConversationScenePresentationParams["scene"]["pendingA2UIForm"];
  pendingA2UISource: PendingA2UISource | null;
  a2uiSubmissionNotice: ConversationScenePresentationParams["scene"]["a2uiSubmissionNotice"];
  handlePendingA2UISubmit: NonNullable<
    ConversationScenePresentationParams["scene"]["onPendingA2UISubmit"]
  >;
  handleToggleCanvas: ConversationScenePresentationParams["scene"]["onToggleCanvas"];
  currentImageWorkbenchActive: ConversationScenePresentationParams["scene"]["currentImageWorkbenchActive"];
  browserWorkbenchOpenRequest?: ConversationScenePresentationParams["canvasWorkbenchLayout"]["browserOpenRequest"];
  onBrowserWorkbenchOpenRequestHandled?: ConversationScenePresentationParams["canvasWorkbenchLayout"]["onBrowserOpenRequestHandled"];
  canvasWorkbenchPreviewOpenRequest?: ConversationScenePresentationParams["canvasWorkbenchLayout"]["previewOpenRequest"];
  onCanvasWorkbenchPreviewOpenRequestHandled?: ConversationScenePresentationParams["canvasWorkbenchLayout"]["onPreviewOpenRequestHandled"];
  hideInlineStepProgress: ConversationScenePresentationParams["stepProgress"]["hidden"];
  isSpecializedThemeMode: ConversationScenePresentationParams["stepProgress"]["isSpecializedThemeMode"];
  hasMessages: ConversationScenePresentationParams["stepProgress"]["hasMessages"];
  steps: ConversationScenePresentationParams["stepProgress"]["steps"];
  activityLogs?: SidebarActivityLog[];
  creationTaskEvents?: GeneralWorkbenchCreationTaskEvent[];
  canonicalChildren?: CanonicalChildThreadSummary[];
  currentStepIndex: ConversationScenePresentationParams["stepProgress"]["currentIndex"];
  goToStep: ConversationScenePresentationParams["stepProgress"]["onStepClick"];
  messageListRuntime: WorkspaceConversationMessageListRuntime;
  layoutMode: LayoutMode;
  isThemeWorkbench: boolean;
  settledWorkbenchArtifacts: ConversationScenePresentationParams["canvasWorkbenchLayout"]["artifacts"];
  resolvedCanvasState: ConversationScenePresentationParams["canvasWorkbenchLayout"]["canvasState"];
  taskFiles: TaskFile[];
  selectedFileId: string | undefined;
  projectRootPath: string | null;
  canvasWorkbenchRootPath?: string | null;
  handleHarnessLoadFilePreview: ConversationScenePresentationParams["canvasWorkbenchLayout"]["loadFilePreview"];
  setCanvasWorkbenchLayoutMode: ConversationScenePresentationParams["canvasWorkbenchLayout"]["onLayoutModeChange"];
  workspacePathMissing: WorkspacePathMissingState | boolean | null;
  workspaceHealthError: boolean;
}

export function useWorkspaceConversationSceneRuntime({
  threadHeader,
  navbarContextVariant = "default",
  navigationActions,
  inputbarScene,
  canvasScene,
  handleSendFromEmptyState,
  landingSurface,
  shellChromeRuntime,
  contextWorkspaceEnabled,
  activeTheme,
  contentId,
  projectId,
  openedProjects,
  onCloseProject,
  deferWorkspaceListLoad,
  taskCenterTabsNode,
  suppressNavbarUtilityActions = false,
  topBarChrome,
  onBackToProjectManagement,
  fromResources,
  handleBackHome,
  rightSurfaceChrome,
  isRestoringSession,
  sessionId,
  threadGoal,
  syncStatus,
  pendingA2UIForm,
  pendingA2UISource,
  a2uiSubmissionNotice,
  handlePendingA2UISubmit,
  handleToggleCanvas,
  currentImageWorkbenchActive,
  browserWorkbenchOpenRequest,
  onBrowserWorkbenchOpenRequestHandled,
  canvasWorkbenchPreviewOpenRequest,
  onCanvasWorkbenchPreviewOpenRequestHandled,
  hideInlineStepProgress,
  isSpecializedThemeMode,
  hasMessages,
  steps,
  activityLogs = [],
  creationTaskEvents = [],
  canonicalChildren = [],
  currentStepIndex,
  goToStep,
  messageListRuntime,
  layoutMode,
  isThemeWorkbench,
  settledWorkbenchArtifacts,
  resolvedCanvasState,
  taskFiles,
  selectedFileId,
  projectRootPath,
  canvasWorkbenchRootPath,
  handleHarnessLoadFilePreview,
  setCanvasWorkbenchLayoutMode,
  workspacePathMissing,
  workspaceHealthError,
}: UseWorkspaceConversationSceneRuntimeParams) {
  const { i18n, t } = useTranslation("agent");
  const locale = i18n.language;
  const {
    emptyStateVariant: messageListEmptyStateVariant = "none",
    quoteInput,
    onQuoteInputChange,
    providerType,
    model,
    reasoningEffort,
    accessMode,
    messages: displayMessages,
    turns = EMPTY_PROJECTED_TURNS,
    threadItems: effectiveThreadItems = EMPTY_PROJECTED_THREAD_ITEMS,
    todoItems = [],
    currentTurnId,
    threadRead,
    executionRuntime,
    pendingActions = EMPTY_PROJECTED_PENDING_ACTIONS,
    submittedActionsInFlight = EMPTY_PROJECTED_SUBMITTED_ACTIONS,
    sessionHistoryWindow = null,
    onLoadFullHistory,
    isSending,
    onInterruptCurrentTurn,
    onReplayPendingRequest,
    onDeleteMessage,
    onEditMessage,
    onA2UISubmit,
    onWriteFile,
    onFileClick,
    onOpenArtifactFromTimeline,
    onOpenSavedSiteContent,
    onArtifactClick,
    onOpenUrlPreview,
    onOpenMessagePreview,
    onSaveMessageAsSkill,
    onSaveMessageAsKnowledge,
    onOpenSubagentSession,
    onPermissionResponse,
    onRefreshSessionReadModel,
    pendingPromotedA2UIActionRequest,
    collapseCodeBlocks,
    shouldCollapseCodeBlock,
    onCodeBlockClick,
    focusedTimelineItemId,
    timelineFocusRequestKey,
  } = messageListRuntime;
  const projectedRuntime = useSessionRuntimeProjectionDeferral({
    sessionId,
    messages: displayMessages,
    turns,
    threadItems: effectiveThreadItems,
    currentTurnId,
    threadRead,
    pendingActions,
    submittedActionsInFlight,
    isRestoringSession,
    isSending: Boolean(isSending),
    focusedTimelineItemId,
    pendingA2UIForm,
  });
  const projectedTurns = projectedRuntime.turns;
  const projectedThreadItems = projectedRuntime.threadItems;
  const projectedCurrentTurnId = projectedRuntime.currentTurnId;
  const projectedThreadRead = projectedRuntime.threadRead;
  const projectedPendingActions = projectedRuntime.pendingActions;
  const projectedSubmittedActionsInFlight =
    projectedRuntime.submittedActionsInFlight;
  const handleQuoteMessage = (content: string) => {
    const quotedText = buildQuotedReplyText({
      content,
      input: quoteInput,
    });
    if (!quotedText) {
      return;
    }
    onQuoteInputChange(quotedText);
  };

  const navbarUtilityActionsVisible = !suppressNavbarUtilityActions;
  const taskCenterUtilityActionsVisible =
    navbarUtilityActionsVisible || navbarContextVariant === "task-center";
  const rightSurfaceSceneProps =
    buildWorkspaceConversationRightSurfaceSceneProps({
      rightSurfaceChrome,
      utilityActionsVisible: taskCenterUtilityActionsVisible,
    });
  const shouldSyncCanvasWorkbenchLayoutMode =
    !isThemeWorkbench &&
    activeTheme === "general" &&
    layoutMode === "chat-canvas";
  const onStartReview = useCallback(async () => {
    const reviewThreadId = projectedThreadRead?.thread_id?.trim() || "";
    if (!reviewThreadId) {
      throw new Error("Review requires an active thread");
    }
    const result = await startReview(
      {
        threadId: reviewThreadId,
        delivery: "inline",
        target: { type: "uncommittedChanges" },
      },
      {
        onTerminal: async () => {
          await onRefreshSessionReadModel?.();
        },
      },
    );
    await onRefreshSessionReadModel?.();
    return result;
  }, [onRefreshSessionReadModel, projectedThreadRead?.thread_id]);
  const codingWorkbenchViews = useMemo(
    () =>
      buildWorkspaceConversationCodingViews({
        t,
        locale,
        turns: projectedTurns,
        threadItems: projectedThreadItems,
        currentTurnId: projectedCurrentTurnId,
        threadRead: projectedThreadRead,
        pendingActions: projectedPendingActions,
        submittedActionsInFlight: projectedSubmittedActionsInFlight,
        isSending,
        focusedTimelineItemId,
        onOpenFile: canvasScene.handleOpenCanvasWorkbenchPath,
        onRespondToAction: onPermissionResponse,
        onRefreshSessionReadModel,
        onStartReview,
        onSubmitRecoveryPrompt: (
          prompt,
          recoveryContext?: CodingWorkbenchRecoveryContext,
        ) =>
          handleSendFromEmptyState({
            textOverride: prompt,
            sendOptions: recoveryContext
              ? {
                  requestMetadata: {
                    harness: {
                      coding_workbench_recovery: recoveryContext,
                    },
                  },
                }
              : undefined,
          }),
      }),
    [
      canvasScene.handleOpenCanvasWorkbenchPath,
      focusedTimelineItemId,
      handleSendFromEmptyState,
      onPermissionResponse,
      onRefreshSessionReadModel,
      onStartReview,
      isSending,
      locale,
      projectedCurrentTurnId,
      projectedPendingActions,
      projectedSubmittedActionsInFlight,
      projectedThreadItems,
      projectedThreadRead,
      projectedTurns,
      t,
    ],
  );
  const landingSurfaceWithSessionRuntime =
    useWorkspaceConversationLandingSessionRuntime({
      landingSurface,
      showChatLayout: shellChromeRuntime.showChatLayout,
      messages: displayMessages,
      turns: projectedTurns,
      threadItems: projectedThreadItems,
      currentTurnId: projectedCurrentTurnId,
      threadRead: projectedThreadRead,
      pendingActions: projectedPendingActions,
      submittedActionsInFlight: projectedSubmittedActionsInFlight,
      canonicalChildren,
      isSending,
      sessionId,
      projectRootPath,
      currentUserMessage:
        codingWorkbenchViews.currentSessionTurn?.prompt_text || null,
      onOpenMemoryWorkbench: navigationActions.handleOpenRuntimeMemoryWorkbench,
      onOpenChannels: navigationActions.handleOpenChannels,
      onOpenChromeRelay: navigationActions.handleOpenChromeRelay,
    });
  const sessionRuntimeCounters = codingWorkbenchViews.counters;
  const shouldUseCodingWorkbenchChrome =
    sessionRuntimeCounters.shouldUseRuntimeWorkbench ||
    navbarContextVariant === "task-center";
  const effectiveCanvasWorkbenchRootPath =
    projectRootPath?.trim() || canvasWorkbenchRootPath?.trim() || null;
  const taskRailProps = useWorkspaceTaskRailRuntime({
    sessionId,
    workflowSteps: steps,
    messages: displayMessages,
    activityLogs,
    creationTaskEvents,
    pendingActions: projectedPendingActions,
    submittedActionsInFlight: projectedSubmittedActionsInFlight,
    threadItems: projectedThreadItems,
    todoItems,
    threadGoal,
    threadRead: projectedThreadRead,
    executionRuntime,
    canonicalChildren,
    providerType,
    model,
    accessMode,
    reasoningEffort,
    projectRootPath,
    canvasWorkbenchRootPath,
    onOpenWorkspacePath: canvasScene.handleOpenCanvasWorkbenchPath,
    onRespondToAction: onPermissionResponse,
  });
  const workspaceView = buildWorkspaceHeaderView({
    projectRootPath: effectiveCanvasWorkbenchRootPath,
    workspacePathMissing: Boolean(workspacePathMissing),
    workspaceHealthError,
  });

  return renderWorkspaceConversationScene({
    scene: {
      threadHeader,
      landingSurface: landingSurfaceWithSessionRuntime,
      showChatLayout: shellChromeRuntime.showChatLayout,
      compactChrome: shellChromeRuntime.isWorkspaceCompactChrome,
      contextWorkspaceEnabled,
      generalWorkbenchMessageViewportBottomPadding:
        shellChromeRuntime.workflowLayoutBottomSpacing
          .messageViewportBottomPadding,
      taskRail: taskRailProps,
      onSelectWorkspaceDirectory:
        navigationActions.handleWorkspaceAlertSelectDirectory,
      onDismissWorkspaceAlert: navigationActions.handleDismissWorkspaceAlert,
      shouldHideGeneralWorkbenchInputForTheme:
        shellChromeRuntime.shouldHideGeneralWorkbenchInputForTheme,
      inputbarNode: inputbarScene.inputbarNode,
      contentId,
      projectId,
      openedProjects,
      projectRootPath,
      deferWorkspaceListLoad,
      onProjectChange: navigationActions.handleProjectChange,
      onCloseProject,
      onOpenSettings: navbarUtilityActionsVisible
        ? navigationActions.handleOpenAppearanceSettings
        : undefined,
      taskCenterTabsNode,
      workspaceType: activeTheme,
      navbarVisible: shellChromeRuntime.shouldRenderTopBar,
      isRunning: Boolean(isSending),
      navbarChrome: topBarChrome,
      navbarContextVariant,
      onBackToProjectManagement,
      onBackToResources: fromResources
        ? navigationActions.handleBackToResources
        : undefined,
      isThemeWorkbench,
      layoutMode,
      onToggleCanvas: handleToggleCanvas,
      onBackHome: handleBackHome,
      ...rightSurfaceSceneProps,
      showContextCompactionAction:
        navbarUtilityActionsVisible &&
        Boolean(sessionId) &&
        projectedThreadRead?.can_accept_direct_input !== false,
      contextCompactionRunning: navbarUtilityActionsVisible && isSending,
      onCompactContext:
        navbarUtilityActionsVisible &&
        projectedThreadRead?.can_accept_direct_input !== false
          ? navigationActions.handleCompactContext
          : undefined,
      syncStatus,
      pendingA2UIForm,
      onPendingA2UISubmit: handlePendingA2UISubmit,
      a2uiSubmissionNotice,
      hasLiveCanvasPreviewContent: canvasScene.hasLiveCanvasPreviewContent,
      liveCanvasPreview: canvasScene.liveCanvasPreview,
      currentImageWorkbenchActive,
      shouldShowCanvasLoadingState: canvasScene.shouldShowCanvasLoadingState,
      shellBottomInset:
        shellChromeRuntime.workflowLayoutBottomSpacing.shellBottomInset,
      chatPanelWidth: shellChromeRuntime.layoutTransitionChatPanelWidth,
      chatPanelMinWidth: shellChromeRuntime.layoutTransitionChatPanelMinWidth,
      generalWorkbenchDialog: inputbarScene.generalWorkbenchDialog,
      showFloatingInputOverlay:
        shellChromeRuntime.shouldShowGeneralWorkbenchFloatingInputOverlay,
      hasPendingA2UIForm: Boolean(pendingA2UIForm),
    },
    stepProgress: {
      hidden: hideInlineStepProgress,
      isSpecializedThemeMode,
      hasMessages,
      steps,
      currentIndex: currentStepIndex,
      onStepClick: goToStep,
    },
    messageList: {
      sessionId,
      messages: displayMessages,
      emptyStateVariant: messageListEmptyStateVariant,
      providerType,
      turns: projectedTurns,
      threadItems: projectedThreadItems,
      currentTurnId: projectedCurrentTurnId,
      threadRead: projectedThreadRead,
      pendingActions: projectedPendingActions,
      submittedActionsInFlight: projectedSubmittedActionsInFlight,
      canonicalChildren,
      sessionHistoryWindow,
      onLoadFullHistory,
      isRestoringSession,
      isSending,
      onInterruptCurrentTurn,
      onReplayPendingRequest,
      onDeleteMessage,
      onEditMessage,
      onQuoteMessage: handleQuoteMessage,
      onA2UISubmit,
      onWriteFile,
      onFileClick,
      onOpenArtifactFromTimeline,
      onOpenSavedSiteContent,
      onArtifactClick,
      onOpenUrlPreview,
      onOpenMessagePreview,
      onSaveMessageAsSkill,
      onSaveMessageAsKnowledge,
      onOpenSubagentSession,
      onPermissionResponse,
      promoteActionRequestsToA2UI: Boolean(pendingPromotedA2UIActionRequest),
      renderA2UIInline: shellChromeRuntime.shouldRenderInlineA2UI,
      activePendingA2UISource: pendingA2UISource,
      collapseCodeBlocks,
      shouldCollapseCodeBlock,
      onCodeBlockClick,
      focusedTimelineItemId,
      timelineFocusRequestKey,
    },
    workspaceAlert: {
      workspacePathMissing: Boolean(workspacePathMissing),
      workspaceHealthError,
    },
    canvasWorkbenchLayout: {
      artifacts: settledWorkbenchArtifacts,
      canvasState: resolvedCanvasState,
      taskFiles,
      selectedFileId,
      workspaceRoot: effectiveCanvasWorkbenchRootPath,
      defaultPreview: canvasScene.canvasWorkbenchDefaultPreview,
      loadFilePreview: handleHarnessLoadFilePreview,
      onOpenPath: canvasScene.handleOpenCanvasWorkbenchPath,
      onRevealPath: canvasScene.handleRevealCanvasWorkbenchPath,
      onClose: canvasScene.handleCloseCanvasWorkbench,
      workbenchMode: shouldUseCodingWorkbenchChrome ? "coding" : "default",
      workspaceView,
      sessionView: codingWorkbenchViews.sessionView,
      outputView: codingWorkbenchViews.outputView,
      logView: codingWorkbenchViews.logView,
      changeView: codingWorkbenchViews.changeView,
      onLayoutModeChange: shouldSyncCanvasWorkbenchLayoutMode
        ? setCanvasWorkbenchLayoutMode
        : undefined,
      browserOpenRequest: browserWorkbenchOpenRequest,
      onBrowserOpenRequestHandled: onBrowserWorkbenchOpenRequestHandled,
      previewOpenRequest: canvasWorkbenchPreviewOpenRequest,
      onPreviewOpenRequestHandled: onCanvasWorkbenchPreviewOpenRequestHandled,
    },
  });
}
