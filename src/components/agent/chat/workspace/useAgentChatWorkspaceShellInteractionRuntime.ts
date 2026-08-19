import type { MutableRefObject } from "react";
import type { useSoulInteractionCopy } from "@/hooks/useSoulInteractionCopy";
import type { CanvasState as GeneralCanvasState } from "@/components/general-chat/bridge";
import type { AgentChatWorkspaceProps } from "../agentChatWorkspaceContract";
import type { TaskFile } from "../components/TaskFiles";
import type { useAgentChatUnified } from "../hooks";
import type { useWorkspaceProjectSelection } from "../hooks/useWorkspaceProjectSelection";
import { resolveAgentChatWorkspaceShellViewModel } from "../agentChatWorkspaceShellViewModel";
import type { useAgentChatWorkspaceCommandWiring } from "./useAgentChatWorkspaceCommandWiring";
import type { useAgentChatWorkspaceLocalDisplayState } from "./useAgentChatWorkspaceLocalDisplayState";
import type { useWorkspaceArtifactCanvasRuntime } from "./useWorkspaceArtifactCanvasRuntime";
import { useWorkspaceCanvasSurfaceRuntime } from "./useWorkspaceCanvasSurfaceRuntime";
import type { useWorkspaceContextSurfaceRuntime } from "./useWorkspaceContextSurfaceRuntime";
import type { useWorkspaceGeneralWorkbenchRuntime } from "./useWorkspaceGeneralWorkbenchRuntime";
import type { useWorkspaceImageWorkbenchSessionRuntime } from "./useWorkspaceImageWorkbenchSessionRuntime";
import type { useWorkspaceOpenedProjectsRuntime } from "./useWorkspaceOpenedProjectsRuntime";
import { useWorkspaceTaskCenterInteractionRuntime } from "./useWorkspaceTaskCenterInteractionRuntime";
import { useWorkspaceTaskCenterSurfaceRuntime } from "./useWorkspaceTaskCenterSurfaceRuntime";

type CommandCompositionScope = ReturnType<
  typeof useAgentChatWorkspaceCommandWiring
>["compositionScope"];
type ContextSurfaceRuntime = ReturnType<
  typeof useWorkspaceContextSurfaceRuntime
>;
type ArtifactCanvasRuntime = ReturnType<
  typeof useWorkspaceArtifactCanvasRuntime
>;
type ImageWorkbenchSessionRuntime = ReturnType<
  typeof useWorkspaceImageWorkbenchSessionRuntime
>;
type OpenedProjectsRuntime = ReturnType<
  typeof useWorkspaceOpenedProjectsRuntime
>;
type GeneralWorkbenchRuntime = ReturnType<
  typeof useWorkspaceGeneralWorkbenchRuntime
>;

interface UseAgentChatWorkspaceShellInteractionRuntimeParams {
  scope: CommandCompositionScope;
  agentChatRuntime: ReturnType<typeof useAgentChatUnified>;
  localDisplayRuntime: ReturnType<
    typeof useAgentChatWorkspaceLocalDisplayState
  >;
  projectSelectionRuntime: ReturnType<typeof useWorkspaceProjectSelection>;
  contextSurfaceRuntime: ContextSurfaceRuntime;
  artifactCanvasRuntime: ArtifactCanvasRuntime;
  imageWorkbenchSessionRuntime: ImageWorkbenchSessionRuntime;
  openedProjects: OpenedProjectsRuntime["openedProjects"];
  documentEditorFocusedRef: GeneralWorkbenchRuntime["documentEditorFocusedRef"];
  soulInteractionCopy: ReturnType<typeof useSoulInteractionCopy>;
  activeSessionIdRef: MutableRefObject<string | null>;
  agentEntry: NonNullable<AgentChatWorkspaceProps["agentEntry"]>;
  clearEntryPendingA2UI: () => void;
  generalCanvasState: GeneralCanvasState;
  hasPendingA2UIForm: boolean;
  homeMountedAt: number;
  initialPendingServiceSkillLaunchSignature: string | null;
  initialSessionName?: string;
  newConversationLabel: string;
  renamePromptLabel: string;
  sessionRestorePresentation: "foreground" | "background";
  selectedFileId?: string;
  shouldBootstrapCanvasOnEntry: boolean;
  shouldPreserveBlankHomeSurface: boolean;
  shouldUseBrowserWorkspaceHomeChrome: boolean;
  showChatPanel: boolean;
  taskCenterWorkspaceId: string | null;
  taskFiles: TaskFile[];
  onHasMessagesChange?: (hasMessages: boolean) => void;
  untitledTaskLabel: string;
}

/** 组合 Workspace 壳层交互，不持有 Thread/Turn/Item 或 Task Center 状态事实源。 */
export function useAgentChatWorkspaceShellInteractionRuntime({
  scope,
  agentChatRuntime,
  localDisplayRuntime,
  projectSelectionRuntime,
  contextSurfaceRuntime,
  artifactCanvasRuntime,
  imageWorkbenchSessionRuntime,
  openedProjects,
  documentEditorFocusedRef,
  soulInteractionCopy,
  activeSessionIdRef,
  agentEntry,
  clearEntryPendingA2UI,
  generalCanvasState,
  hasPendingA2UIForm,
  homeMountedAt,
  initialPendingServiceSkillLaunchSignature,
  initialSessionName,
  newConversationLabel,
  renamePromptLabel,
  sessionRestorePresentation,
  selectedFileId,
  shouldBootstrapCanvasOnEntry,
  shouldPreserveBlankHomeSurface,
  shouldUseBrowserWorkspaceHomeChrome,
  showChatPanel,
  taskCenterWorkspaceId,
  taskFiles,
  onHasMessagesChange,
  untitledTaskLabel,
}: UseAgentChatWorkspaceShellInteractionRuntimeParams) {
  const {
    activeTheme,
    autoCollapsedTopicSidebarRef,
    bootstrapDispatchPreview,
    canvasState,
    clearMessages,
    clearTaskCenterEmbeddedHomeSession,
    contentId,
    defaultTopicSidebarVisible,
    displayMessages,
    executionStrategy,
    externalProjectId,
    homePendingPreviewRequest,
    initialDispatchKey,
    initialSessionId,
    input,
    isAutoRestoringSession,
    isBootstrapDispatchPending,
    isPreparingSend,
    isSending,
    isTaskCenterEntry,
    lockTheme,
    mappedTheme,
    markTaskCenterEmbeddedHomeSession,
    markTaskCenterLocalSessionOverride,
    messages,
    newChatAt,
    normalizedInitialSessionId,
    onNavigate,
    originalSwitchTopic,
    projectId,
    queuedTurnCount,
    replaceTaskCenterOpenTabs,
    resetProjectSelection,
    resetTopicLocalState,
    sessionId,
    setActiveTaskCenterDraftTabId,
    setCanvasState,
    setGeneralCanvasState,
    setHomePendingPreviewRequest,
    setInput,
    setLayoutMode,
    setMentionedCharacters,
    setSelectedFileId,
    setSelectedText,
    setShowSidebar,
    setTaskCenterDetachedTopicId,
    setTaskCenterDraftSendRequest,
    setTaskCenterDraftTabs,
    setTaskCenterLocalSessionOverride,
    setTaskCenterOpenTabMap,
    setTaskCenterTransitionTopicId,
    shouldUseCompactGeneralWorkbench,
    switchTopic,
    taskCenterDetachedTopicId,
    taskCenterDraftSendRequest,
    taskCenterDraftSurfaceActiveRef,
    taskCenterDraftTabs,
    taskCenterEmbeddedHomeSessionIds,
    taskCenterFallbackRestoreRef,
    taskCenterLocalSessionOverride,
    taskCenterOpenTabIds,
    taskCenterOpenTabIdsRef,
    taskCenterTransitionTopicId,
    topicById,
    topics,
    upsertTaskCenterOpenTab,
  } = scope;
  const { createFreshSession, renameTopic } = agentChatRuntime;
  const {
    canvasWorkbenchLayoutMode,
    layoutMode,
    setCanvasWorkbenchLayoutMode,
    showSidebar,
  } = localDisplayRuntime;
  const { applyProjectSelection } = projectSelectionRuntime;
  const { contextHarnessRuntime, effectiveThreadItems } = contextSurfaceRuntime;
  const { harnessPanelVisible, setHarnessPanelVisible } = contextHarnessRuntime;
  const { currentCanvasArtifact } = artifactCanvasRuntime;
  const { resetLocalImageWorkbenchSessionScope } = imageWorkbenchSessionRuntime;

  const taskCenterSurfaceRuntime = useWorkspaceTaskCenterSurfaceRuntime({
    activeTheme,
    bootstrapDispatchPreview,
    draftSurface: {
      agentEntry,
      isTaskCenterEntry,
      activeDraftTabId: scope.activeTaskCenterDraftTabId,
      draftTabs: taskCenterDraftTabs,
      draftSurfaceActive: taskCenterDraftSurfaceActiveRef.current,
      initialSessionId: normalizedInitialSessionId,
      sessionId,
      draftSendRequest: taskCenterDraftSendRequest,
      displayMessageCount: displayMessages.length,
      threadItemCount: effectiveThreadItems.length,
      hasLocalSessionOverride: taskCenterLocalSessionOverride !== null,
      hasPendingA2UIForm,
      isPreparingSend,
      isSending,
      queuedTurnCount,
    },
    homePendingPreview: {
      homePendingPreviewRequest,
      displayMessagesLength: displayMessages.length,
      executionStrategy,
      workspaceId: taskCenterWorkspaceId,
      soulCopy: soulInteractionCopy,
    },
    lockTheme,
    onNavigate,
    taskCenterWorkspaceId,
  });
  const {
    isTaskCenterDraftTabActive,
    isTaskCenterDraftSurfaceActive,
    shouldSuppressTaskCenterDraftContent,
    isHomePendingPreviewActive,
    isHomeSendStarting,
    persistTaskCenterMaterializedSessionNavigation,
  } = taskCenterSurfaceRuntime;

  const hasCanvasWorkbenchContent = layoutMode !== "chat";
  const shellViewModel = resolveAgentChatWorkspaceShellViewModel({
    agentEntry,
    showChatPanel,
    contentId,
    initialSessionId,
    displayMessageCount: displayMessages.length,
    threadItemCount: effectiveThreadItems.length,
    isHomePendingPreviewActive,
    shouldSuppressTaskCenterDraftContent,
    hasCanvasWorkbenchContent,
    isThemeWorkbench: contextHarnessRuntime.isThemeWorkbench,
    shouldUseCompactGeneralWorkbench,
    isBootstrapDispatchPending,
    isSending,
    queuedTurnCount,
  });
  const { hasDisplayMessages, hasMessages, effectiveShowChatPanel } =
    shellViewModel;

  const canvasSurfaceRuntime = useWorkspaceCanvasSurfaceRuntime({
    layout: {
      activeTheme,
      isThemeWorkbench: contextHarnessRuntime.isThemeWorkbench,
      hasPendingA2UIForm,
      layoutMode,
      showChatPanel: effectiveShowChatPanel,
      showSidebar,
      defaultTopicSidebarVisible,
      hasMessages,
      canvasWorkbenchLayoutMode,
      autoCollapsedTopicSidebarRef,
      mappedTheme,
      normalizedEntryTheme: scope.normalizedInitialTheme,
      shouldPreserveBlankHomeSurface,
      shouldBootstrapCanvasOnEntry,
      canvasState,
      generalCanvasState,
      hasCurrentCanvasArtifact: Boolean(currentCanvasArtifact),
      currentImageWorkbenchActive:
        imageWorkbenchSessionRuntime.currentImageWorkbenchState.active,
      onHasMessagesChange,
      setShowSidebar,
      setLayoutMode,
      setGeneralCanvasState,
      setCanvasState,
      setCanvasWorkbenchLayoutMode,
    },
    selection: {
      activeTheme,
      contentId,
      setSelectedText,
    },
    taskFileSync: {
      taskFiles,
      isThemeWorkbench: contextHarnessRuntime.isThemeWorkbench,
      selectedFileId,
      canvasState,
      mappedTheme,
      documentEditorFocusedRef,
      setSelectedFileId,
      setCanvasState,
    },
  });

  const taskCenterInteractionRuntime = useWorkspaceTaskCenterInteractionRuntime(
    {
      chromeNavigation: {
        agentEntry,
        applyProjectSelection,
        clearEmbeddedHomeSession: clearTaskCenterEmbeddedHomeSession,
        detachedTopicId: taskCenterDetachedTopicId,
        displayMessageCount: displayMessages.length,
        draftSendRequest: taskCenterDraftSendRequest,
        embeddedHomeSessionIds: taskCenterEmbeddedHomeSessionIds,
        externalProjectId,
        fallbackRestoreRef: taskCenterFallbackRestoreRef,
        hasDisplayMessages,
        hasLocalSessionOverride: taskCenterLocalSessionOverride !== null,
        hasPendingA2UIForm,
        harnessPanelVisible,
        homeMountedAt,
        initialDispatchKey,
        initialSessionName,
        initialPendingServiceSkillLaunchSignature,
        isAutoRestoringSession,
        isBootstrapDispatchPending,
        isHomeSessionBackgroundRecovery:
          sessionRestorePresentation === "background" &&
          !normalizedInitialSessionId,
        isHomePendingPreviewActive,
        isHomeSendStarting,
        isPreparingSend,
        isSending,
        isSessionHydrating: scope.isSessionHydrating,
        isThemeWorkbench: contextHarnessRuntime.isThemeWorkbench,
        layoutMode,
        messagesLength: messages.length,
        newChatAt,
        newConversationLabel,
        normalizedInitialSessionId,
        onNavigate,
        onToggleWorkbench: canvasSurfaceRuntime.handleToggleCanvas,
        openTabIds: taskCenterOpenTabIds,
        openedProjects,
        projectId,
        queuedTurnsLength: queuedTurnCount,
        renamePromptLabel,
        renameTopic,
        resetProjectSelection,
        sessionId,
        setHarnessPanelVisible,
        shouldSuppressDraftContent: shouldSuppressTaskCenterDraftContent,
        shouldUseBrowserWorkspaceHomeChrome,
        taskCenterWorkspaceId,
        threadItemCount: effectiveThreadItems.length,
        topicById,
        topics,
        transitionTopicId: taskCenterTransitionTopicId,
        untitledTaskLabel,
      },
      draftMaterialization: {
        activeTaskCenterDraftTabId: scope.activeTaskCenterDraftTabId,
        agentEntry,
        clearMessages,
        createFreshSession,
        input,
        isPreparingSend,
        isSending,
        markTaskCenterEmbeddedHomeSession,
        markTaskCenterLocalSessionOverride,
        resetLocalImageWorkbenchSessionScope,
        resetTopicLocalState,
        setActiveTaskCenterDraftTabId,
        setHomePendingPreviewRequest,
        setInput,
        setMentionedCharacters,
        setSelectedText,
        setTaskCenterDetachedTopicId,
        setTaskCenterDraftSendRequest,
        setTaskCenterDraftTabs,
        setTaskCenterTransitionTopicId,
        taskCenterDraftSurfaceActiveRef,
        taskCenterDraftTabs,
        taskCenterWorkspaceId,
        upsertTaskCenterOpenTab,
      },
      persistMaterializedSessionNavigation:
        persistTaskCenterMaterializedSessionNavigation,
      taskCenterSurface: {
        isTaskCenterDraftSurfaceActive,
        isTaskCenterDraftTabActive,
      },
      topicNavigation: {
        activeSessionIdRef,
        agentEntry,
        clearEntryPendingA2UI,
        clearMessages,
        clearTaskCenterEmbeddedHomeSession,
        messagesLength: messages.length,
        replaceTaskCenterOpenTabs,
        resetLocalImageWorkbenchSessionScope,
        resetTopicLocalState,
        sessionId,
        setActiveTaskCenterDraftTabId,
        setHomePendingPreviewRequest,
        setInput,
        setMentionedCharacters,
        setSelectedText,
        setTaskCenterDetachedTopicId,
        setTaskCenterDraftSendRequest,
        setTaskCenterDraftTabs,
        setTaskCenterLocalSessionOverride,
        setTaskCenterOpenTabMap,
        setTaskCenterTransitionTopicId,
        switchTopic,
        taskCenterDetachedTopicId,
        taskCenterDraftSurfaceActiveRef,
        taskCenterOpenTabIdsRef,
        taskCenterTransitionTopicId,
        taskCenterWorkspaceId,
        topicById,
        upsertTaskCenterOpenTab,
        markTaskCenterLocalSessionOverride,
      },
      switchTopic: originalSwitchTopic,
    },
  );

  return {
    ...shellViewModel,
    ...canvasSurfaceRuntime,
    ...taskCenterSurfaceRuntime,
    ...taskCenterInteractionRuntime,
    hasCanvasWorkbenchContent,
  };
}
