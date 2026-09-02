/** Workspace scene/projection current owner。 */

/** Workspace scene/projection current owner。 */

import { useCallback } from "react";
import { useWorkspaceCanvasSceneRuntime } from "./useWorkspaceCanvasSceneRuntime";
import { useWorkspaceInputbarSceneRuntime } from "./useWorkspaceInputbarSceneRuntime";
import { useWorkspaceTaskCenterSendRuntime } from "./useWorkspaceTaskCenterSendRuntime";
import { useWorkspaceMessageKnowledgeSaveRuntime } from "./useWorkspaceMessageKnowledgeSaveRuntime";
import { ScheduledTaskDialog } from "@/components/scheduled-tasks/ScheduledTaskDialog";
import {
  EMPTY_WORKSPACE_WORKFLOW_STEPS,
  HIDDEN_WORKSPACE_WORKFLOW_STEP_INDEX,
  ignoreHiddenWorkspaceWorkflowStepClick,
} from "./useWorkspaceHiddenWorkflowProgressRuntime";
import { useAgentChatWorkspaceSceneComposition } from "./useAgentChatWorkspaceSceneComposition";
import { buildThreadWorkspaceHeaderViewModel } from "./threadWorkspaceHeaderViewModel";
import type { AgentChatWorkspaceProps } from "../agentChatWorkspaceContract";
import type { useAgentChatWorkspaceEntryRuntime } from "./useAgentChatWorkspaceEntryRuntime";
import type { useAgentChatWorkspaceSetupRuntime } from "./useAgentChatWorkspaceSetupRuntime";
import type { useAgentChatWorkspaceCommandRuntime } from "./useAgentChatWorkspaceCommandRuntime";

type EntryRuntime = ReturnType<typeof useAgentChatWorkspaceEntryRuntime>;
type SetupRuntime = ReturnType<typeof useAgentChatWorkspaceSetupRuntime>;
type CommandRuntime = ReturnType<typeof useAgentChatWorkspaceCommandRuntime>;

interface UseAgentChatWorkspaceSceneRuntimeParams {
  props: AgentChatWorkspaceProps;
  entryRuntime: EntryRuntime;
  setupRuntime: SetupRuntime;
  commandRuntime: CommandRuntime;
}

export function useAgentChatWorkspaceSceneRuntime({
  props,
  entryRuntime,
  setupRuntime,
  commandRuntime,
}: UseAgentChatWorkspaceSceneRuntimeParams) {
  // prettier-ignore
  const {
    onNavigate: _onNavigate, contentId, agentEntry, lockTheme, fromResources, topBarChrome = "full", onBackToProjectManagement,
    hideInlineStepProgress, initialSessionName, initialKnowledgePackSelection, newChatAt, onBackgroundSessionRuntimeChange, preferContentReviewInRightRail,
  } = props;
  const resolvedAgentEntry = agentEntry ?? "claw";
  const resolvedLockTheme = lockTheme ?? false;
  const resolvedFromResources = fromResources ?? false;
  const resolvedHideInlineStepProgress = hideInlineStepProgress ?? false;
  const resolvedPreferContentReviewInRightRail =
    preferContentReviewInRightRail ?? false;
  // prettier-ignore
  const {
    untitledTaskLabel, normalizedEntryTheme, shouldBootstrapCanvasOnEntry, activeTheme, artifactPreviewSize, creationMode, effectiveEntryBannerMessage, entryBannerVisible,
    expertInfoPanelCollapsed, handleInputRestoreRequestHandled, input, inputbarObjectiveModeEnabled, inputRestoreRequest, layoutMode, selectedText, setActiveTheme,
    setArtifactPreviewSize, setCanvasWorkbenchLayoutMode, setCreationMode, setExpertInfoPanelCollapsed, setInput, setInputbarObjectiveModeEnabled, setLayoutMode, pathReferences,
    handleAddPathReferences, handleRemovePathReference, handleClearPathReferences, fileManagerSidebar, activeSessionIdRef, setChatToolPreferences, handleInstallSkillPackageFromFileManager,
    handleOpenSkillsManageFromExpertPanel, projectId, markNewChatRequestHandled, taskCenterWorkspaceId, normalizedInitialSessionId, shouldUseBrowserWorkspaceHomeChrome, shouldDeferWorkspaceAuxiliaryLoads, project,
    projectMemory, isInitialContentLoading, initialContentLoadError, canvasState, setCanvasState, initialCreationReplaySurface, runtimeWorkspaceId, generalCanvasState,
    setGeneralCanvasState, taskFiles, selectedFileId, openedProjects, handleCloseOpenedProject, clawTraceEnabled, soulArtifactVoiceGenerationBrief, soulArtifactVoiceEnabledForTurn,
    setSoulArtifactVoiceEnabledForTurn, inputCompletionEnabled, imageGenerationSelectionReady, imageGenerationSelectionWarning, imageWorkbenchGenerationRuntime, imageWorkbenchPreferenceSummary, imageWorkbenchPreferenceWarning, setMentionedCharacters,
    skills, serviceSkills, serviceSkillGroups, handleRefreshSkills, handleSkillSuggestionsNeeded, mappedTheme, syncStatus,
    isSpecializedThemeMode, workbenchRequests, handleNavigateToSkillSettings,
  } = entryRuntime;
  // prettier-ignore
  const {
    generalHarnessEntryEnabled, providerType, setProviderType, model, setModel, reasoningEffort, setReasoningEffort, accessMode,
    setAccessMode, messages, setChatMessages, currentTurnId, turns, todoItems,
    threadRead, threadGoal, threadGoalError, isThreadGoalLoading, executionRuntime, sessionWorkingDir, isSending, stopSending, replayPendingAction,
    clearMessages, deleteMessage, editMessage, handlePermissionResponse, submittedActionsInFlight, sessionHistoryWindow,
    isAutoRestoringSession, isSessionHydrating, sessionId, ensureSession, getThreadIdForSubmit, originalSwitchTopic, loadFullSessionHistory, refreshSessionReadModel, renameTopic, archiveTopic, forkTopic, workspacePathMissing, workspaceHealthError,
    combinedSkillsLoading, expertPanelRequestMetadata, expertPanelRuntimeKey, expertSkillRefsOverride, expertWorkspaceSkillRuntimeEnableBindings, expertWorkspaceSkillRuntimeEnableRefs, handleEnableExpertWorkspaceSkillRuntime, handleExpertSkillRefsChange,
    handlePluginSuggestionsNeeded, handleThreadExpertProfileSwitch, workspacePluginInputSuggestions, workspacePluginSuggestionsError, workspacePluginSuggestionsLoading, workspaceSkillBindings, topicById, effectiveChatToolPreferences, canonicalChildren,
    currentSessionTitle, handleStopSending, handleOpenSubagentSession, currentImageWorkbenchState, imageWorkbenchSessionKey, updateCurrentImageWorkbenchState, artifacts, artifactDisplayState,
    artifactViewMode, currentCanvasArtifact, displayedCanvasArtifact, handleArtifactViewModeChange,
    handleOpenBrowserRuntimeForBrowserAssist, settledWorkbenchArtifacts, contextHarnessRuntime, effectiveThreadItems, harnessState, inputbarIsSending, rightSurfaceLocalState, contextWorkspace,
    isThemeWorkbench, setHarnessPanelVisible, harnessPendingCount, showHarnessToggle, harnessAttentionLevel, harnessToggleLabel, generalWorkbenchScaffoldRuntime, workspaceServiceSkillEntryActions,
    a2uiSubmissionNotice, effectivePendingA2UIForm, effectivePendingA2UISource, handleMessageA2UISubmit, handlePendingA2UISubmit, pendingPromotedA2UIActionRequest, currentGate, themeWorkbenchRunState,
  } = setupRuntime;
  // prettier-ignore
  const {
    shellChromeRuntime, showGeneralWorkbenchLeftExpandButton, effectiveInitialInputCapability, generalWorkbenchHarnessPanelBaseProps, navigationActions, homePendingPreviewRequest, setActiveTaskCenterDraftTabId, setHomePendingPreviewRequest,
    setTaskCenterDraftSendRequest, setTaskCenterDraftTabs, taskCenterDraftSendRequest, taskCenterDraftSurfaceActiveRef, generalWorkbenchEntryPrompt, handleBackHome, markTaskCenterLocalSessionOverride, setTaskCenterDetachedTopicId,
    setTaskCenterTransitionTopicId, upsertTaskCenterOpenTab, handleSend, handleRecommendationClick, handleSendRef, isPreparingSend, displayMessages, handleImageWorkbenchCommand,
    imageWorkbenchActionRuntime, handleContinueGeneralWorkbenchEntryPrompt, handleRestartGeneralWorkbenchEntryPrompt, handleDocumentAutoContinueRun, handleArtifactBlockRewriteRun, handleDocumentContentReviewRun, handleDocumentTextStylizeRun, handleAddImage,
    handleImportDocument, activeTaskCenterDraftTabIdRef, bootstrapPendingPreviewMessages, browserWorkspaceHomeTabsNode, commitMaterializedTaskCenterDraftTab, handleCanvasSelectionTextChange, handleCloseCanvas, handleOpenProjectConversation,
    handleOpenTaskTopic, handleResumeRecentSession, handleToggleCanvas, hasDisplayMessages, hasMessages, homePendingPreviewMessages, isTaskCenterDraftSendPending, materializeTaskCenterDraftTab,
    projectConversationGroups, recentSessionActionLabel, recentSessionTopic, resolvedCanvasState, shouldRenderTaskCenterEmbeddedHome, shouldRenderTaskCenterTabStrip, shouldRestoreImageTasksFromWorkspace, shouldSuppressTaskCenterDraftContent,
    suppressHomeNavbarUtilityActions, taskCenterDraftMaterializedSessionIdsRef, taskCenterDraftWarmupSessionIdsRef, taskCenterHomeSurfaceState, taskCenterTabsNode, bindArticleEditorRightSurface, bindRightSurfacePendingActions, handleCodeBlockClick,
    handleHarnessLoadFilePreview, handleOpenArtifactFromTimeline, handleOpenMessagePreview, handleOpenSavedSiteContent, handleOpenServiceSkillResultFile, handleOpenUrlPreview, handleSaveArtifactDocument, handleWorkspaceArtifactClick,
    handleWorkspaceFileClick, openProjectFilePreviewInCanvas, openWorkspaceArtifactInWorkbench, preferredServiceSkillResultFileTarget, renderArtifactWorkbenchToolbarActions, shouldCollapseCodeBlockInChat, shouldCollapseCodeBlocks, handleWriteFile,
    handleJumpToTimelineItem, serviceSkillExecutionCard, generalWorkbenchActivityLogs,
    generalWorkbenchWorkflowSteps, handleExpandGeneralWorkbenchSidebar, renderGeneralWorkbenchSidebarNode, planComposerPendingActions, planDecisionAccessory, handleSaveMessageAsSkill,
  } = commandRuntime;
  const inputbarScene = useWorkspaceInputbarSceneRuntime({
    contextVariant: agentEntry === "claw" ? "task-center" : "default",
    setMentionedCharacters,
    taskFiles,
    selectedFileId,
    isThemeWorkbench,
    sessionId,
    threadId: threadRead?.thread_id ?? null,
    canAcceptDirectInput: threadRead?.can_accept_direct_input,
    threadGoal,
    threadGoalError,
    threadGoalLoading: isThreadGoalLoading,
    generalWorkbenchHarnessPanelBaseProps,
    currentSessionTitle,
    handleStopSending,
    input,
    setInput,
    currentGate,
    generalWorkbenchWorkflowSteps,
    steps: EMPTY_WORKSPACE_WORKFLOW_STEPS,
    workflowRunState: themeWorkbenchRunState,
    handleSend,
    isPreparingSend,
    isSending: inputbarIsSending,
    isSessionRestoring:
      isAutoRestoringSession ||
      isSessionHydrating ||
      taskCenterHomeSurfaceState.isRestoringSession,
    providerType,
    setProviderType,
    model,
    setModel,
    reasoningEffort,
    setReasoningEffort,
    sessionExecutionRuntime: executionRuntime,
    projectId: projectId ?? null,
    openedProjects,
    projectRootPath: project?.rootPath || null,
    accessMode,
    setAccessMode,
    activeTheme,
    navigationActions,
    characters: projectMemory?.characters || [],
    skills,
    serviceSkills: activeTheme === "general" ? serviceSkills : [],
    serviceSkillGroups: activeTheme === "general" ? serviceSkillGroups : [],
    skillsLoading: combinedSkillsLoading,
    onSelectServiceSkill:
      workspaceServiceSkillEntryActions.handleServiceSkillSelect,
    onSkillSuggestionsNeeded: handleSkillSuggestionsNeeded,
    initialInputCapability: effectiveInitialInputCapability,
    initialKnowledgePackSelection,
    pluginSuggestions: workspacePluginInputSuggestions,
    pluginSuggestionsError: workspacePluginSuggestionsError?.message ?? null,
    pluginSuggestionsLoading: workspacePluginSuggestionsLoading,
    onPluginSuggestionsNeeded: handlePluginSuggestionsNeeded,
    setChatToolPreferences,
    objectiveEnabled: inputbarObjectiveModeEnabled,
    onObjectiveEnabledChange: setInputbarObjectiveModeEnabled,
    handleNavigateToSkillSettings,
    handleRefreshSkills,
    soulArtifactVoiceGenerationBrief,
    soulArtifactVoiceEnabledForTurn,
    onSoulArtifactVoiceEnabledForTurnChange: setSoulArtifactVoiceEnabledForTurn,
    generalWorkbenchEntryPrompt,
    handleRestartGeneralWorkbenchEntryPrompt,
    handleContinueGeneralWorkbenchEntryPrompt,
    planDecisionAccessory,
    generalWorkbenchEnabled:
      generalHarnessEntryEnabled && !suppressHomeNavbarUtilityActions,
    harnessPanelVisible:
      !suppressHomeNavbarUtilityActions &&
      contextHarnessRuntime.harnessPanelVisible,
    setHarnessPanelVisible: contextHarnessRuntime.setHarnessPanelVisible,
    harnessState,
    mappedTheme,
    activeRuntimeStatusTitle: contextHarnessRuntime.activeRuntimeStatusTitle,
    chatToolPreferences: effectiveChatToolPreferences,
    defaultCuratedTaskReferenceMemoryIds:
      initialCreationReplaySurface?.defaultReferenceMemoryIds,
    defaultCuratedTaskReferenceEntries:
      initialCreationReplaySurface?.defaultReferenceEntries,
    pathReferences,
    onAddPathReferences: handleAddPathReferences,
    inputRestoreRequest,
    onInputRestoreRequestHandled: handleInputRestoreRequestHandled,
    onRemovePathReference: handleRemovePathReference,
    onClearPathReferences: handleClearPathReferences,
    fileManagerOpen: fileManagerSidebar.fileManagerOpen,
    onToggleFileManager: fileManagerSidebar.fileManagerAvailable
      ? fileManagerSidebar.toggleFileManagerSidebar
      : undefined,
    inputCompletionEnabled,
  });
  const { handleSaveMessageAsKnowledge } =
    useWorkspaceMessageKnowledgeSaveRuntime({
      currentSessionTitle,
      importTextAsKnowledge: inputbarScene.onImportTextAsKnowledge,
      knowledgeSelectionWorkingDir:
        inputbarScene.knowledgePackSelection?.workingDir,
      onNavigate: _onNavigate,
      projectRootPath: project?.rootPath,
      selectedPackName: inputbarScene.knowledgePackSelection?.packName,
    });

  const canvasScene = useWorkspaceCanvasSceneRuntime({
    shouldBootstrapCanvasOnEntry,
    normalizedEntryTheme,
    mappedTheme,
    canvasState,
    resolvedCanvasState,
    isInitialContentLoading,
    initialContentLoadError,
    imageWorkbenchGenerationRuntime,
    imageWorkbenchActionRuntime,
    inputbarScene,
    projectRootPath: project?.rootPath || null,
    generalCanvasState,
    setGeneralCanvasState,
    currentCanvasArtifact,
    displayedCanvasArtifact,
    artifactDisplayState,
    artifactViewMode,
    setArtifactViewMode: handleArtifactViewModeChange,
    artifactPreviewSize,
    setArtifactPreviewSize,
    onSaveArtifactDocument: handleSaveArtifactDocument,
    onArtifactBlockRewriteRun: handleArtifactBlockRewriteRun,
    renderArtifactWorkbenchToolbarActions,
    threadItems: effectiveThreadItems,
    focusedBlockId: workbenchRequests.focusedArtifactBlockId,
    blockFocusRequestKey: workbenchRequests.artifactBlockFocusRequestKey,
    onJumpToTimelineItem: handleJumpToTimelineItem,
    handleCloseCanvas,
    currentImageWorkbenchState,
    imageWorkbenchPreferenceSummary,
    imageWorkbenchPreferenceWarning,
    setCanvasState,
    handleBackHome,
    isSending,
    handleCanvasSelectionTextChange,
    projectId: projectId ?? null,
    contentId: contentId ?? null,
    imageGenerationSelectionReady,
    imageGenerationSelectionWarning,
    sourceThreadId: sessionId ?? null,
    providerType,
    setProviderType,
    model,
    setModel,
    handleDocumentAutoContinueRun,
    handleAddImage,
    handleImportDocument,
    handleDocumentContentReviewRun,
    handleDocumentTextStylizeRun,
    preferContentReviewInRightRail: resolvedPreferContentReviewInRightRail,
  });

  const taskCenterSendRuntime = useWorkspaceTaskCenterSendRuntime({
    activeDraftTabIdRef: activeTaskCenterDraftTabIdRef,
    activeSessionIdRef,
    agentEntry: resolvedAgentEntry,
    clearMessages,
    commitMaterializedDraftTab: commitMaterializedTaskCenterDraftTab,
    currentSessionId: sessionId,
    currentTurnId,
    displayMessages,
    effectiveThreadItems,
    executionRuntime,
    handleSend,
    hasDisplayMessages,
    homePendingPreviewMessages,
    bootstrapPendingPreviewMessages,
    input,
    isPreparingSend,
    isSending,
    isTaskCenterDraftSendPending,
    layoutMode,
    markNewChatRequestHandled,
    markTaskCenterLocalSessionOverride,
    materializedSessionIdsRef: taskCenterDraftMaterializedSessionIdsRef,
    materializeDraftTab: materializeTaskCenterDraftTab,
    messagesLength: messages.length,
    newChatAt,
    normalizedInitialSessionId,
    planComposerPendingActions,
    prewarmedDraftSessionIdsRef: taskCenterDraftWarmupSessionIdsRef,
    sendRef: handleSendRef,
    setActiveDraftTabId: setActiveTaskCenterDraftTabId,
    setDetachedTopicId: setTaskCenterDetachedTopicId,
    setHomePendingPreviewRequest,
    setInput,
    setTaskCenterDraftSendRequest,
    setTaskCenterDraftTabs,
    setTransitionTopicId: setTaskCenterTransitionTopicId,
    shouldRenderTaskCenterEmbeddedHome,
    shouldSuppressTaskCenterDraftContent,
    submittedActionsInFlight,
    taskCenterDraftSendRequest,
    taskCenterDraftSurfaceActiveRef,
    taskCenterHomeSurfaceState,
    taskCenterWorkspaceId,
    threadRead,
    turns,
    switchToReadySession: originalSwitchTopic,
    upsertTaskCenterOpenTab,
  });
  const {
    handleSendFromEmptyState,
    sceneDisplayMessages,
    sceneTurns,
    sceneThreadItems,
    sceneCurrentTurnId,
    sceneThreadRead,
    sceneExecutionRuntime,
    scenePendingActions,
    sceneSubmittedActionsInFlight,
    sceneIsPreparingSend,
    sceneIsSending,
    sceneIsRestoringSession,
    sceneLayoutMode,
    sceneMessageListEmptyStateVariant,
    sceneSessionId,
    shouldHideCurrentSessionContent,
  } = taskCenterSendRuntime;
  const ensureBrowserWorkspaceOwner = useCallback(async () => {
    const runtimeSessionId = (await ensureSession())?.trim();
    if (!runtimeSessionId) {
      return null;
    }
    let threadId = getThreadIdForSubmit(runtimeSessionId)?.trim();
    if (!threadId) {
      await refreshSessionReadModel();
      threadId = getThreadIdForSubmit(runtimeSessionId)?.trim();
    }
    return threadId ? { runtimeSessionId, threadId } : null;
  }, [ensureSession, getThreadIdForSubmit, refreshSessionReadModel]);
  const workspaceSceneNode = useAgentChatWorkspaceSceneComposition({
    expertPanel: {
      canOpenSkillsManage: Boolean(_onNavigate),
      combinedSkillsLoading,
      effectiveThreadItems,
      expertInfoPanelCollapsed,
      expertPanelRequestMetadata,
      expertPanelRuntimeKey,
      expertSkillRefsOverride,
      expertWorkspaceSkillRuntimeEnableBindings,
      expertWorkspaceSkillRuntimeEnableRefs,
      handleEnableExpertWorkspaceSkillRuntime,
      handleExpertSkillRefsChange,
      handleOpenSkillsManageFromExpertPanel,
      handleThreadExpertProfileSwitch,
      localSkills: skills,
      sceneLayoutMode,
      serviceSkills,
      setExpertInfoPanelCollapsed,
      workspaceSkillBindings,
    },
    rightSurface: {
      articleEditor: {
        activeArticleWorkspace: rightSurfaceLocalState.activeArticleWorkspace,
        canvasState,
        contentId,
        currentImageWorkbenchState,
        imageWorkbenchSessionKey,
        messages,
        projectId,
        runtimeWorkspaceId,
        sceneDisplayMessages,
        sceneIsPreparingSend,
        sceneIsSending,
        sceneSessionId,
        sceneThreadRead,
        setCanvasState,
        setChatMessages,
        shouldDeferWorkspaceAuxiliaryLoads,
        shouldHideCurrentSessionContent,
        shouldRestoreImageTasksFromWorkspace,
        updateCurrentImageWorkbenchState,
      },
      bindArticleEditorRightSurface,
      chrome: {
        showHarnessToggle,
        harnessPendingCount,
        harnessAttentionLevel,
        harnessToggleLabel,
        suppressHarnessChrome: suppressHomeNavbarUtilityActions,
      },
      coordinator: {
        bindRightSurfacePendingActions,
        clawTraceEnabled,
        expertInfoPanelCollapsed,
        handleToggleCanvas,
        harnessPendingCount,
        localState: rightSurfaceLocalState,
        preferredServiceSkillResultFileTarget,
        runtimeWorkspaceId,
        sceneIsPreparingSend,
        sceneIsSending,
        sceneLayoutMode,
        sceneSessionId,
        sceneThreadId: sceneThreadRead?.thread_id ?? null,
        activityAvailable: Boolean(sceneThreadRead?.thread_id),
        sessionId,
        showHarnessToggle,
        suppressHomeNavbarUtilityActions,
        taskCenterHomeHotpathActive:
          shouldRenderTaskCenterEmbeddedHome ||
          Boolean(taskCenterDraftSendRequest || homePendingPreviewRequest),
        threadItems: sceneThreadItems,
        setExpertInfoPanelCollapsed,
        setHarnessPanelVisible,
        setLayoutMode,
      },
      host: {
        ensureBrowserWorkspaceOwner,
        generalWorkbenchHarnessPanelBaseProps,
        activityPanelProps: {
          canonicalChildren,
          threadItems: sceneThreadItems,
          toolInventory: generalWorkbenchHarnessPanelBaseProps.toolInventory,
          onOpenSubagentSession: handleOpenSubagentSession,
        },
        harnessState,
        preferredServiceSkillResultFileTarget,
        runtimeWorkspaceId,
        sceneSessionId,
        sceneThreadId: sceneThreadRead?.thread_id ?? null,
        onOpenArticlePreviewArtifact: openWorkspaceArtifactInWorkbench,
        onOpenServiceSkillResultFile: handleOpenServiceSkillResultFile,
        handleSendRef,
        restoreInput: setInput,
        setLayoutMode,
      },
      imageSlot: {
        contentId,
        handleImageWorkbenchCommand,
        projectId,
        setLayoutMode,
      },
      projectRootPath: project?.rootPath,
      renderGeneralWorkbenchSidebarNode,
      sessionWorkingDir,
    },
    homeRecovery: {
      onBackgroundSessionRuntimeChange,
      onNavigate: _onNavigate,
      onOpenTaskTopic: handleOpenTaskTopic,
      onResumeRecentSession: handleResumeRecentSession,
      projectId,
      recentSessionTopic,
    },
    conversation: {
      landing: {
        accessMode,
        activeTheme,
        artifacts,
        browserAssistLoading: false,
        chatToolPreferences: effectiveChatToolPreferences,
        contentId,
        creationMode,
        creationReplaySurface: initialCreationReplaySurface,
        defaultCuratedTaskReferenceEntries:
          initialCreationReplaySurface?.defaultReferenceEntries,
        defaultCuratedTaskReferenceMemoryIds:
          initialCreationReplaySurface?.defaultReferenceMemoryIds,
        emptyStateDisabled: sceneIsPreparingSend || sceneIsSending,
        emptyStateIsLoading: sceneIsPreparingSend || sceneIsSending,
        emptyStateSendOnPointerDown: true,
        entryBannerMessage: effectiveEntryBannerMessage,
        entryBannerVisible,
        fileManagerOpen: fileManagerSidebar.fileManagerOpen,
        generalCanvasContent: generalCanvasState.content,
        handleSendFromEmptyState,
        initialInputCapability: effectiveInitialInputCapability,
        input,
        inputbarScene,
        inputRestoreRequest,
        lockTheme: resolvedLockTheme,
        model,
        objectiveEnabled: inputbarObjectiveModeEnabled,
        onAddPathReferences: handleAddPathReferences,
        onClearPathReferences: handleClearPathReferences,
        onDismissEntryBanner: navigationActions.handleDismissEntryBanner,
        onInputRestoreRequestHandled: handleInputRestoreRequestHandled,
        onLaunchBrowserAssist: handleOpenBrowserRuntimeForBrowserAssist,
        onManageProviders: navigationActions.handleManageProviders,
        onNavigateToSettings: handleNavigateToSkillSettings,
        onObjectiveEnabledChange: setInputbarObjectiveModeEnabled,
        onOpenProjectConversation: handleOpenProjectConversation,
        onPluginSuggestionsNeeded: handlePluginSuggestionsNeeded,
        onProjectChange: navigationActions.handleProjectChange,
        onRecommendationClick: handleRecommendationClick,
        onRefreshSkills: handleRefreshSkills,
        onRemovePathReference: handleRemovePathReference,
        onSelectServiceSkill:
          workspaceServiceSkillEntryActions.handleServiceSkillSelect,
        onStopSending: stopSending,
        onToggleFileManager: fileManagerSidebar.fileManagerAvailable
          ? fileManagerSidebar.toggleFileManagerSidebar
          : undefined,
        openedProjects,
        pathReferences,
        pluginSuggestions: workspacePluginInputSuggestions,
        pluginSuggestionsError:
          workspacePluginSuggestionsError?.message ?? null,
        pluginSuggestionsLoading: workspacePluginSuggestionsLoading,
        projectCharacters: projectMemory?.characters || [],
        projectConversationGroups,
        projectId: projectId ?? null,
        providerType,
        reasoningEffort,
        recentSessionActionLabel,
        recentSessionSummary: recentSessionTopic?.lastPreview ?? null,
        recentSessionTitle: recentSessionTopic?.title ?? null,
        resolvedCanvasState,
        serviceSkillExecutionCard,
        serviceSkillGroups: activeTheme === "general" ? serviceSkillGroups : [],
        serviceSkills: activeTheme === "general" ? serviceSkills : [],
        sessionId: sceneSessionId,
        setAccessMode,
        setActiveTheme,
        setChatToolPreferences,
        setCreationMode,
        setInput,
        setModel,
        setProviderType,
        setReasoningEffort,
        skills,
        skillsLoading: combinedSkillsLoading,
        selectedText,
        suppressRecentSessionRecovery:
          sceneMessageListEmptyStateVariant === "task-center",
      },
      messageList: {
        actions: {
          onA2UISubmit: handleMessageA2UISubmit,
          onArtifactClick: handleWorkspaceArtifactClick,
          onCodeBlockClick: handleCodeBlockClick,
          onDeleteMessage: deleteMessage,
          onEditMessage: editMessage,
          onFileClick: handleWorkspaceFileClick,
          onInterruptCurrentTurn: stopSending,
          onLoadFullHistory: () => {
            void loadFullSessionHistory();
          },
          onOpenArtifactFromTimeline: handleOpenArtifactFromTimeline,
          onOpenMessagePreview: handleOpenMessagePreview,
          onOpenSavedSiteContent: handleOpenSavedSiteContent,
          onOpenSubagentSession: handleOpenSubagentSession,
          onOpenUrlPreview: handleOpenUrlPreview,
          onPermissionResponse: handlePermissionResponse,
          onReplayPendingRequest: replayPendingAction,
          onSaveMessageAsKnowledge: handleSaveMessageAsKnowledge,
          onSaveMessageAsSkill: handleSaveMessageAsSkill,
          onWriteFile: handleWriteFile,
        },
        collapseCodeBlocks: shouldCollapseCodeBlocks,
        emptyStateVariant: sceneMessageListEmptyStateVariant,
        focus: {
          focusedTimelineItemId: workbenchRequests.focusedTimelineItemId,
          timelineFocusRequestKey: workbenchRequests.timelineFocusRequestKey,
        },
        input: {
          quoteInput: input,
          onQuoteInputChange: setInput,
        },
        pendingPromotedA2UIActionRequest,
        projection: {
          currentTurnId: sceneCurrentTurnId,
          executionRuntime: sceneExecutionRuntime,
          isSending: sceneIsSending,
          pendingActions: scenePendingActions,
          sessionHistoryWindow,
          submittedActionsInFlight: sceneSubmittedActionsInFlight,
          threadItems: sceneThreadItems,
          threadRead: sceneThreadRead,
          todoItems,
          turns: sceneTurns,
        },
        provider: {
          accessMode,
          model,
          providerType,
          reasoningEffort,
        },
        refreshSessionReadModel,
        sceneSessionId,
        shouldCollapseCodeBlock: shouldCollapseCodeBlockInChat,
      },
      scene: {
        threadHeader: buildThreadWorkspaceHeaderViewModel({
          sessionId: sceneSessionId,
          currentSessionTitle,
          initialSessionId: normalizedInitialSessionId,
          initialSessionName,
          topic: sceneSessionId ? topicById.get(sceneSessionId) : null,
          sessionWorkingDirectory: sessionWorkingDir,
          projectRootPath: project?.rootPath,
          canonicalThreadStatus: threadRead?.status,
          canAcceptDirectInput: threadRead?.can_accept_direct_input,
          onRename: sceneSessionId
            ? (nextTitle) => renameTopic(sceneSessionId, nextTitle)
            : undefined,
          onArchive: sceneSessionId
            ? async () => {
                const archived = await archiveTopic(sceneSessionId);
                if (archived) {
                  handleBackHome();
                }
              }
            : undefined,
          onFork: sceneSessionId
            ? async () => {
                const forkedSessionId = await forkTopic(sceneSessionId);
                if (forkedSessionId) {
                  await originalSwitchTopic(forkedSessionId);
                }
              }
            : undefined,
          isSending: sceneIsSending,
          pendingActionCount: scenePendingActions.length,
          untitledTaskLabel,
        }),
        navbarContextVariant:
          agentEntry === "claw" || shouldUseBrowserWorkspaceHomeChrome
            ? "task-center"
            : "default",
        navigationActions,
        inputbarScene,
        canvasScene,
        handleSendFromEmptyState,
        shellChromeRuntime,
        currentImageWorkbenchActive: currentImageWorkbenchState.active,
        canvasWorkbenchPreviewOpenRequest:
          workbenchRequests.canvasWorkbenchPreviewOpenRequest,
        onCanvasWorkbenchPreviewOpenRequestHandled:
          workbenchRequests.handleCanvasWorkbenchPreviewOpenRequestHandled,
        projectId: projectId ?? null,
        openedProjects,
        onCloseProject: handleCloseOpenedProject,
        deferWorkspaceListLoad: shouldUseBrowserWorkspaceHomeChrome,
        projectRootPath: project?.rootPath || null,
        contextWorkspaceEnabled: contextWorkspace.generalWorkbenchEnabled,
        activeTheme,
        contentId,
        taskCenterTabsNode: shouldRenderTaskCenterTabStrip
          ? taskCenterTabsNode
          : browserWorkspaceHomeTabsNode,
        suppressNavbarUtilityActions: suppressHomeNavbarUtilityActions,
        topBarChrome,
        onBackToProjectManagement,
        fromResources: resolvedFromResources,
        handleBackHome,
        isRestoringSession: sceneIsRestoringSession,
        sessionId: sceneSessionId,
        threadGoal,
        syncStatus,
        pendingA2UIForm: effectivePendingA2UIForm,
        pendingA2UISource: effectivePendingA2UISource,
        a2uiSubmissionNotice,
        handlePendingA2UISubmit,
        hideInlineStepProgress: resolvedHideInlineStepProgress,
        isSpecializedThemeMode,
        hasMessages,
        steps: EMPTY_WORKSPACE_WORKFLOW_STEPS,
        activityLogs: generalWorkbenchActivityLogs,
        creationTaskEvents:
          generalWorkbenchScaffoldRuntime.generalWorkbenchCreationTaskEvents,
        canonicalChildren,
        currentStepIndex: HIDDEN_WORKSPACE_WORKFLOW_STEP_INDEX,
        goToStep: ignoreHiddenWorkspaceWorkflowStepClick,
        layoutMode: sceneLayoutMode,
        isThemeWorkbench,
        settledWorkbenchArtifacts,
        resolvedCanvasState,
        taskFiles,
        selectedFileId,
        handleHarnessLoadFilePreview,
        setCanvasWorkbenchLayoutMode,
        workspacePathMissing: Boolean(workspacePathMissing),
        workspaceHealthError,
      },
    },
    fileManager: {
      fileManagerSidebar,
      initialDirectory: project?.rootPath || null,
      onAddPathReferences: handleAddPathReferences,
      onImportAsKnowledge: inputbarScene.onImportPathReferenceAsKnowledge,
      onInstallSkillPackage: _onNavigate
        ? handleInstallSkillPackageFromFileManager
        : undefined,
      onOpenWorkspaceFile: (absolutePath) => {
        void openProjectFilePreviewInCanvas({
          absolutePath,
        });
      },
    },
    shell: {
      compactChrome: shellChromeRuntime.isWorkspaceCompactChrome,
      isThemeWorkbench,
      showGeneralWorkbenchLeftExpandButton,
      onExpandGeneralWorkbenchSidebar: handleExpandGeneralWorkbenchSidebar,
    },
  });
  return (
    <>
      {workspaceSceneNode}
      <ScheduledTaskDialog
        open={workspaceServiceSkillEntryActions.automationDialogOpen}
        initialForm={
          workspaceServiceSkillEntryActions.automationDialogInitialValues
        }
        saving={workspaceServiceSkillEntryActions.automationJobSaving}
        onOpenChange={
          workspaceServiceSkillEntryActions.handleAutomationDialogOpenChange
        }
        onSubmit={
          workspaceServiceSkillEntryActions.handleAutomationDialogSubmit
        }
      />
    </>
  );
}
