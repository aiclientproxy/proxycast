/** Workspace command/runtime current owner。 */

import { useMemo } from "react";

import {
  cancelMediaTaskArtifact,
  createImageGenerationTaskArtifact,
  getMediaTaskArtifact,
} from "@/lib/api/mediaTasks";
import { useWorkspaceNavigationActions } from "./useWorkspaceNavigationActions";
import { useWorkspaceWorkbenchSideEffectRuntime } from "./useWorkspaceWorkbenchSideEffectRuntime";
import { useWorkspaceShellChromeRuntime } from "./useWorkspaceShellChromeRuntime";
import { useWorkspaceGeneralWorkbenchSidebarHostRuntime } from "./useWorkspaceGeneralWorkbenchSidebarHostRuntime";
import { useWorkspaceHarnessNavigationRuntime } from "./useWorkspaceHarnessNavigationRuntime";
import { useWorkspaceGeneralWorkbenchHarnessSurfaceRuntime } from "./useWorkspaceGeneralWorkbenchHarnessSurfaceRuntime";
import { useWorkspaceMessageSkillSaveRuntime } from "./useWorkspaceMessageSkillSaveRuntime";
import { useWorkspaceHiddenWorkflowProgressRuntime } from "./useWorkspaceHiddenWorkflowProgressRuntime";
import { useWorkspacePlanDecisionRuntime } from "./useWorkspacePlanDecisionRuntime";
import { GENERAL_WORKBENCH_HISTORY_PAGE_SIZE } from "./generalWorkbenchHelpers";
import { useAgentChatWorkspaceCommandWiring } from "./useAgentChatWorkspaceCommandWiring";
import { useAgentChatWorkspaceShellInteractionRuntime } from "./useAgentChatWorkspaceShellInteractionRuntime";
import { useAgentChatWorkspaceArtifactInteractionRuntime } from "./useAgentChatWorkspaceArtifactInteractionRuntime";
import { resolveEffectiveInitialInputCapability } from "../utils/inputCapabilityBootstrap";
import type { AgentChatWorkspaceProps } from "../agentChatWorkspaceContract";
import type { useAgentChatWorkspaceEntryRuntime } from "./useAgentChatWorkspaceEntryRuntime";
import type { useAgentChatWorkspaceSetupRuntime } from "./useAgentChatWorkspaceSetupRuntime";

type EntryRuntime = ReturnType<typeof useAgentChatWorkspaceEntryRuntime>;
type SetupRuntime = ReturnType<typeof useAgentChatWorkspaceSetupRuntime>;

interface UseAgentChatWorkspaceCommandRuntimeParams {
  props: AgentChatWorkspaceProps;
  entryRuntime: EntryRuntime;
  setupRuntime: SetupRuntime;
}

export function useAgentChatWorkspaceCommandRuntime({
  props,
  entryRuntime,
  setupRuntime,
}: UseAgentChatWorkspaceCommandRuntimeParams) {
  // prettier-ignore
  const {
    onNavigate: _onNavigate, projectId: externalProjectId, contentId, initialSessionId, initialSceneAppExecutionSummary, initialAutoSendRequestMetadata, autoRunInitialPromptOnMount, agentEntry,
    theme: initialTheme, initialCreationMode, lockTheme, showChatPanel, hideTopBar, topBarChrome, onWorkflowProgressChange, initialUserPrompt,
    initialUserImages, initialSessionName, initialInputCapability, initialProjectFileOpenTarget, onInitialUserPromptConsumed, newChatAt, onHasMessagesChange: _onHasMessagesChange,
  } = props;
  const resolvedAutoRunInitialPromptOnMount =
    autoRunInitialPromptOnMount ?? false;
  const resolvedAgentEntry = agentEntry ?? "claw";
  const resolvedLockTheme = lockTheme ?? false;
  const resolvedShowChatPanel = showChatPanel ?? true;
  const resolvedHideTopBar = hideTopBar ?? false;
  const resolvedTopBarChrome = topBarChrome ?? "full";
  // prettier-ignore
  const {
    t, untitledTaskLabel, taskCenterRenamePromptLabel, newConversationLabel, workspaceRenderT0, normalizedEntryTheme, defaultTopicSidebarVisible, shouldBootstrapCanvasOnEntry,
    shouldKeepNewTaskHomeSessionRestoreDisabled, localDisplayRuntime, activeTheme, layoutMode, runtimeInitialInputCapability, setCanvasWorkbenchLayoutMode, setEntryBannerVisible, setExpertInfoPanelCollapsed,
    setInput, setLayoutMode, showSidebar, activeSessionIdRef, deferSessionRecentMetadataSyncForNavigation, setChatToolPreferences, projectSelectionRuntime,
    projectId, shouldDisableSessionRestore, applyProjectSelection, taskCenterWorkspaceId, normalizedInitialSessionId, sessionRestorePresentation, shouldPreserveBlankHomeSurface, shouldUseBrowserWorkspaceHomeChrome,
    shouldDeferWorkspaceAuxiliaryLoads, project, isInitialContentLoading, canvasState, setCanvasState, setDocumentVersionStatusMap, lastCanvasSyncRequestRef, initialAutoSendAllowsDetachedSession,
    initialCreationReplay, initialCreationReplaySurface, initialPendingServiceSkillLaunchSignature, validatedRuntimeProjectId, generalCanvasState, setGeneralCanvasState, taskFiles, setTaskFiles,
    selectedFileId, setSelectedFileId, taskFilesRef, socialStageLogRef, openedProjects, serviceModels, agentResponseLanguage, refreshServiceModelsConfig,
    soulArtifactVoiceGenerationBrief, soulInteractionCopy, soulArtifactVoiceEnabledForTurn, setSoulArtifactVoiceEnabledForTurn, effectiveImageWorkbenchPreference, imageWorkbenchGenerationRuntime, setOnDemandMediaDefaults, imageWorkbenchSelectedProviderId,
    imageWorkbenchSelectedModelId, imageWorkbenchSelectedSize, setImageWorkbenchSelectedSize, imageWorkbenchPreferredProviderUnavailable, ensureImageWorkbenchProvidersLoaded, imageWorkbenchProvidersLoading, mentionedCharacters,
    setMentionedCharacters, serviceSkills, pendingSkillKey, consumePendingSkill, processedMessageIds, handleWriteFileRef, sceneGateResumeHandlerRef, mappedTheme,
    syncContent, isSpecializedThemeMode, workbenchRequests,
  } = entryRuntime;
  // prettier-ignore
  const {
    systemPrompt, agentChatRuntime, providerType, model, executionStrategy, accessMode, messages, setChatMessages,
    currentTurnId, turns, threadItems, queuedTurnCount, threadRead, threadGoal, threadGoalError, isThreadGoalLoading, activeExecutionRuntime, isSending, compactSession, refreshSessionReadModel,
    stopSending, replayPendingAction, handlePermissionResponse, pendingActions, submittedActionsInFlight, triggerAIGuide,
    topics, isAutoRestoringSession, isSessionHydrating, sessionId, ensureSession, originalSwitchTopic, workspacePathMissing,
    fixWorkspacePathAndRetry, dismissWorkspacePathError, setWorkspaceHealthError, expertWorkspaceSkillRuntimeEnableInput, workspaceRequestMetadataWithExpertSkills, workspaceSkillBindings, topicById,
    autoCollapsedTopicSidebarRef, effectiveChatToolPreferences, canonicalChildren, hasRuntimeSessions, subagentsRuntimeVisible, handleOpenSubagentSession, imageWorkbenchSessionRuntime, currentImageWorkbenchState,
    imageWorkbenchSessionKey, updateCurrentImageWorkbenchState, artifactCanvasRuntime, artifacts, browserAssistArtifactOpenControl, applyAutoArtifactViewMode, browserAssistRequestAutoLaunch, browserAssistRequestPreferredBackend,
    browserAssistRequestProfileKey, browserAssistSessionState, currentCanvasArtifact, ensureBrowserAssistCanvas, setSelectedArtifactId, siteSkillExecutionState, upsertGeneralArtifact, handleOpenBrowserRuntimeForSiteSkillExecution,
    contextSurfaceRuntime, contextHarnessRuntime, effectiveThreadItems, harnessShellState, inputbarIsSending, openArticleWorkspaceRightSurface, contextWorkspace, isThemeWorkbench,
    setHarnessPanelVisible, generalWorkbenchScaffoldRuntime, shouldUseCompactGeneralWorkbench, shouldSkipGeneralWorkbenchAutoGuideWithoutPrompt, setTopicStatus, workspaceServiceSkillEntryActions, clearEntryPendingA2UI, hasPendingA2UIForm,
    openRuntimeSceneGate, currentGate, documentEditorFocusedRef, themeWorkbenchActiveQueueItem, themeWorkbenchBackendRunState, themeWorkbenchRunState, handleViewContextDetail, harnessInventoryRuntime,
  } = setupRuntime;
  const commandWiring = useAgentChatWorkspaceCommandWiring({
    navigationProjectId: validatedRuntimeProjectId ?? undefined,
    trayActiveTheme: mappedTheme,
    consumePendingSkill,
    pendingSkillKey,
    sceneGateResumeHandlerRef,
    scope: {
      ...agentChatRuntime,
      ...localDisplayRuntime,
      ...projectSelectionRuntime,
      accessMode,
      agentResponseLanguage,
      agentEntry: resolvedAgentEntry,
      autoCollapsedTopicSidebarRef,
      autoRunInitialPromptOnMount: resolvedAutoRunInitialPromptOnMount,
      autoSyncEnabled: false,
      browserAssistAutoLaunch: browserAssistRequestAutoLaunch,
      browserAssistPreferredBackend: browserAssistRequestPreferredBackend,
      browserAssistProfileKey: browserAssistRequestProfileKey,
      browserAssistSessionState,
      cancelImageTask: cancelMediaTaskArtifact,
      canvasState,
      chatToolPreferences: effectiveChatToolPreferences,
      clearPendingEntryA2UI: clearEntryPendingA2UI,
      contentId,
      contextWorkspace: {
        enabled: contextWorkspace.generalWorkbenchEnabled,
        activeContextPrompt: contextWorkspace.activeContextPrompt,
        prepareActiveContextPrompt: contextWorkspace.prepareActiveContextPrompt,
      },
      createImageGenerationTask: createImageGenerationTaskArtifact,
      currentGateKey: currentGate.key,
      currentImageWorkbenchState,
      currentSessionId: sessionId,
      defaultTopicSidebarVisible,
      deferInitialSync: true,
      deferSessionRecentMetadataSyncForNavigation,
      effectiveThreadItemCount: threadItems.length,
      externalProjectId,
      getImageTask: getMediaTaskArtifact,
      hasInitialSessionTopic: normalizedInitialSessionId
        ? topicById.has(normalizedInitialSessionId)
        : false,
      initialCreationMode,
      initialSessionId,
      initialSessionMessagesCount: normalizedInitialSessionId
        ? (topicById.get(normalizedInitialSessionId)?.messagesCount ?? null)
        : null,
      initialTheme,
      initialUserImages,
      initialUserPrompt,
      isAutoRestoringSession,
      isSessionHydrating,
      isSpecializedThemeMode,
      isThemeWorkbench,
      imageWorkbenchPreferredModelId:
        effectiveImageWorkbenchPreference.preferredModelId,
      imageWorkbenchPreferredProviderId:
        effectiveImageWorkbenchPreference.preferredProviderId,
      imageWorkbenchPreferredProviderUnavailable,
      imageWorkbenchProvidersLoading,
      imageWorkbenchSelectedModelId,
      imageWorkbenchSelectedProviderId,
      imageWorkbenchSelectedSize,
      imageWorkbenchSessionKey,
      ensureImageWorkbenchProvidersLoaded,
      ensureBrowserAssistCanvas,
      ensureSessionForCommandMetadata: ensureSession,
      executionStrategy,
      handleAutoLaunchMatchedSiteSkill:
        workspaceServiceSkillEntryActions.handleAutoLaunchMatchedSiteSkill,
      lastCanvasSyncRequestRef,
      lockTheme: resolvedLockTheme,
      mappedTheme,
      mentionedCharacters,
      messages,
      messagesLength: messages.length,
      newChatAt,
      normalizedInitialSessionId,
      normalizedInitialTheme: normalizedEntryTheme,
      onInitialUserPromptConsumed,
      onNavigate: _onNavigate,
      openRuntimeSceneGate,
      originalSwitchTopic,
      preserveSessionRestoreOnNewChat:
        shouldKeepNewTaskHomeSessionRestoreDisabled &&
        !shouldDisableSessionRestore,
      processedMessageIdsRef: processedMessageIds,
      projectImageGenerationPreference: project?.settings?.imageGeneration,
      projectName: project?.name,
      projectRootPath: project?.rootPath || null,
      queuedTurnCount,
      queuedTurnsLength: queuedTurnCount,
      resolveServiceModelsBeforeSend: shouldDeferWorkspaceAuxiliaryLoads
        ? refreshServiceModelsConfig
        : undefined,
      savedSoulArtifactVoiceGenerationBrief: soulArtifactVoiceGenerationBrief,
      serviceModels,
      serviceSkills: activeTheme === "general" ? serviceSkills : [],
      setCanvasState,
      setDocumentVersionStatusMap,
      setChatMessages,
      setChatToolPreferences,
      setGeneralCanvasState,
      setMentionedCharacters,
      setOnDemandMediaDefaults,
      setSelectedFileId,
      setSoulArtifactVoiceEnabledForTurn,
      setTaskFiles,
      setTopicStatus,
      shouldUseCompactGeneralWorkbench,
      syncContent,
      taskCenterWorkspaceId,
      taskFilesLength: taskFiles.length,
      themeWorkbenchLatestTerminal:
        themeWorkbenchBackendRunState?.latest_terminal ?? null,
      themeWorkbenchRunState,
      themeWorkbenchActiveQueueTitle: themeWorkbenchActiveQueueItem?.title,
      threadItemsLength: threadItems.length,
      topicById,
      topics,
      turnsLength: turns.length,
      updateCurrentImageWorkbenchState,
      workspaceRequestMetadataBase:
        workspaceRequestMetadataWithExpertSkills ?? undefined,
      workspaceSkillBindings,
      workspaceSkillRuntimeEnable: expertWorkspaceSkillRuntimeEnableInput,
      soulArtifactVoiceEnabledForTurn,
    },
  });
  const {
    homePendingPreviewRequest,
    setActiveTaskCenterDraftTabId,
    setHomePendingPreviewRequest,
    setTaskCenterDraftSendRequest,
    setTaskCenterDraftTabs,
    taskCenterDraftSendRequest,
    taskCenterDraftSurfaceActiveRef,
  } = commandWiring.taskCenterDraftState;
  const {
    readSessionFile,
    saveSessionFile,
    sessionFiles,
    syncGeneralArtifactToResource,
  } = commandWiring.persistence;
  const {
    consumedInitialPromptRef,
    generalWorkbenchEntryCheckPending,
    generalWorkbenchEntryPrompt,
    hasTriggeredGuideRef,
    initialDispatchKey,
    isBootstrapDispatchPending,
  } = commandWiring.initialDispatch;
  const { handleBackHome } = commandWiring.reset;
  const {
    markTaskCenterLocalSessionOverride,
    setTaskCenterDetachedTopicId,
    setTaskCenterTransitionTopicId,
    upsertTaskCenterOpenTab,
  } = commandWiring.taskCenterNavigation;

  const {
    handleSend,
    handleRecommendationClick,
    handleSendRef,
    isPreparingSend,
    displayMessages,
    handleImageWorkbenchCommand,
    imageWorkbenchActionRuntime,
    latestAssistantMessageId,
  } = commandWiring.send;

  const {
    applyWorkbenchFollowUpActionPayload,
    handleContinueGeneralWorkbenchEntryPrompt,
    handleRestartGeneralWorkbenchEntryPrompt,
    handleDocumentAutoContinueRun,
    handleArtifactBlockRewriteRun,
    handleDocumentContentReviewRun,
    handleDocumentTextStylizeRun,
    handleSwitchBranchVersion,
    handleCreateVersionSnapshot,
    handleSetBranchStatus,
    handleAddImage,
    handleImportDocument,
  } = commandWiring.workbenchActions;

  const shellInteractionRuntime = useAgentChatWorkspaceShellInteractionRuntime({
    scope: commandWiring.compositionScope,
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
    agentEntry: resolvedAgentEntry,
    clearEntryPendingA2UI,
    generalCanvasState,
    hasPendingA2UIForm,
    homeMountedAt: workspaceRenderT0.current,
    initialPendingServiceSkillLaunchSignature,
    initialSessionName,
    newConversationLabel,
    renamePromptLabel: taskCenterRenamePromptLabel,
    sessionRestorePresentation,
    selectedFileId,
    shouldBootstrapCanvasOnEntry,
    shouldPreserveBlankHomeSurface,
    shouldUseBrowserWorkspaceHomeChrome,
    showChatPanel: resolvedShowChatPanel,
    taskCenterWorkspaceId,
    taskFiles,
    onHasMessagesChange: _onHasMessagesChange,
    untitledTaskLabel,
  });
  const {
    activeTaskCenterDraftTabIdRef,
    bootstrapPendingPreviewMessages,
    browserWorkspaceHomeTabsNode,
    commitMaterializedTaskCenterDraftTab,
    effectiveShowChatPanel,
    handleCanvasSelectionTextChange,
    handleCloseCanvas,
    handleOpenProjectConversation,
    handleOpenTaskTopic,
    handleResumeRecentSession,
    handleToggleCanvas,
    hasCanvasWorkbenchContent,
    hasDisplayMessages,
    hasHomeConversationActivity,
    hasMessages,
    homePendingPreviewMessages,
    isTaskCenterDraftSendPending,
    materializeTaskCenterDraftTab,
    projectConversationGroups,
    recentSessionActionLabel,
    recentSessionTopic,
    resolvedCanvasState,
    shouldRenderTaskCenterEmbeddedHome,
    shouldRenderTaskCenterTabStrip,
    shouldRestoreImageTasksFromWorkspace,
    shouldSuppressTaskCenterDraftContent,
    suppressHomeNavbarUtilityActions,
    taskCenterDraftMaterializedSessionIdsRef,
    taskCenterDraftWarmupSessionIdsRef,
    taskCenterHomeSurfaceState,
    taskCenterTabsNode,
  } = shellInteractionRuntime;
  const artifactInteractionRuntime =
    useAgentChatWorkspaceArtifactInteractionRuntime({
      action: {
        activeTheme,
        artifacts,
        contentId,
        currentCanvasArtifact,
        currentGateKey: currentGate.key,
        currentTurnId,
        effectiveThreadItems,
        generalCanvasState,
        browserAssistArtifactOpenControl,
        handleToggleCanvas,
        handleWriteFileRef,
        initialProjectFileOpenTarget,
        isInitialContentLoading,
        isThemeWorkbench,
        layoutMode,
        mappedTheme,
        messages,
        onNavigate: _onNavigate,
        openArticleWorkspaceRightSurface,
        projectId,
        projectRootPath: project?.rootPath || null,
        readSessionFile,
        saveSessionFile,
        sessionFiles,
        sessionId,
        threadId: threadRead?.thread_id ?? null,
        setArtifactViewMode: applyAutoArtifactViewMode,
        setCanvasState,
        setCanvasWorkbenchLayoutMode,
        setDocumentVersionStatusMap,
        setExpertInfoPanelCollapsed,
        setGeneralCanvasState,
        setHarnessPanelVisible,
        setLayoutMode,
        setSelectedArtifactId,
        setSelectedFileId,
        setTaskFiles,
        siteSkillExecutionState,
        socialStageLogRef,
        suppressCanvasAutoOpen: hasPendingA2UIForm,
        syncGeneralArtifactToResource,
        taskFiles,
        taskFilesRef,
        themeWorkbenchActiveQueueItem,
        updateCurrentImageWorkbenchState,
        upsertGeneralArtifact,
        workbenchRequests,
      },
      surface: {
        serviceSkillExecution: {
          onOpenBrowserRuntime: handleOpenBrowserRuntimeForSiteSkillExecution,
          state: siteSkillExecutionState,
        },
        sceneAppExecution: {
          artifacts,
          initialSummary: initialSceneAppExecutionSummary,
          isSending,
          onApplyFollowUpAction: applyWorkbenchFollowUpActionPayload,
          onNavigate: _onNavigate,
          projectId,
          readSessionFile,
          replayReferenceEntries:
            initialCreationReplaySurface?.defaultReferenceEntries,
          replayReferenceMemoryIds:
            initialCreationReplaySurface?.defaultReferenceMemoryIds,
          sessionFiles,
          sessionId,
          taskFiles,
        },
        setLayoutMode,
        workbenchRequests,
      },
    });
  const {
    bindArticleEditorRightSurface,
    bindRightSurfacePendingActions,
    handleCodeBlockClick,
    handleHarnessLoadFilePreview,
    handleOpenArtifactFromTimeline,
    handleOpenMessagePreview,
    handleOpenSavedSiteContent,
    handleOpenServiceSkillResultFile,
    handleOpenUrlPreview,
    handleSaveArtifactDocument,
    handleWorkspaceArtifactClick,
    handleWorkspaceFileClick,
    openProjectFilePreviewInCanvas,
    openWorkspaceArtifactInWorkbench,
    preferredServiceSkillResultFileTarget,
    renderArtifactWorkbenchToolbarActions,
    shouldCollapseCodeBlockInChat,
    shouldCollapseCodeBlocks,
    handleWriteFile,
    defaultCuratedTaskReferenceEntries,
    defaultCuratedTaskReferenceMemoryIds,
    handleJumpToTimelineItem,
    sceneAppExecutionSummaryCard,
    sceneAppReviewDecisionDialogNode,
    serviceSkillExecutionCard,
  } = artifactInteractionRuntime;
  useWorkspaceWorkbenchSideEffectRuntime({
    autoGuide: {
      autoRunInitialPromptOnMount: resolvedAutoRunInitialPromptOnMount,
      canvasState,
      contentId,
      consumedInitialPromptRef,
      generalWorkbenchEntryCheckPending,
      generalWorkbenchEntryPrompt,
      handleSend,
      hasProject: Boolean(project),
      hasTriggeredGuideRef,
      initialAutoSendAllowsDetachedSession,
      initialAutoSendRequestMetadata,
      initialDispatchKey,
      initialUserPrompt,
      initialUserImages,
      isSending,
      isThemeWorkbench,
      mappedTheme,
      messagesLength: messages.length,
      onInitialUserPromptConsumed,
      projectId,
      sessionId,
      setInput,
      shouldSkipGeneralWorkbenchAutoGuideWithoutPrompt,
      shouldUseCompactGeneralWorkbench,
      systemPrompt,
    },
    mediaTask: {
      canvasState,
      contentId,
      handleImageWorkbenchCommand,
      messages,
      onImageWorkbenchRequested:
        imageWorkbenchGenerationRuntime.ensureProvidersLoaded,
      projectId,
      projectRootPath: project?.rootPath || null,
      setCanvasState,
      setChatMessages,
      setImageWorkbenchSelectedSize,
      updateCurrentImageWorkbenchState,
    },
    triggerAIGuide,
  });

  const shellChromeRuntime = useWorkspaceShellChromeRuntime({
    activeTheme,
    agentEntry: resolvedAgentEntry,
    contextWorkspaceEnabled: contextWorkspace.generalWorkbenchEnabled,
    effectiveShowChatPanel,
    gateStatus: currentGate.status,
    generalWorkbenchPanelCollapseEnabled:
      generalWorkbenchScaffoldRuntime.enableGeneralWorkbenchPanelCollapse,
    generalWorkbenchSidebarCollapsed:
      generalWorkbenchScaffoldRuntime.generalWorkbenchSidebarCollapsed,
    hasCanvasWorkbenchContent,
    hasDisplayMessages,
    hasHomeConversationActivity,
    hasPendingA2UIForm,
    hideTopBar: resolvedHideTopBar,
    isBootstrapDispatchPending,
    isPreparingSend,
    isSending,
    isTaskCenterDraftSendPending,
    isThemeWorkbench,
    layoutMode,
    normalizedInitialSessionId,
    queuedTurnCount,
    sessionId,
    shouldRenderTaskCenterEmbeddedHome,
    shouldSuppressTaskCenterDraftContent,
    shouldUseBrowserWorkspaceHomeChrome,
    shouldUseCompactGeneralWorkbench,
    showSidebar,
    subagentsRuntimeVisible,
    hasRuntimeSessions,
    themeWorkbenchRunState,
    topBarChrome: resolvedTopBarChrome,
  });
  const showGeneralWorkbenchSidebar =
    shellChromeRuntime.showGeneralWorkbenchSidebar;
  const showGeneralWorkbenchLeftExpandButton =
    shellChromeRuntime.showGeneralWorkbenchLeftExpandButton;
  const generalWorkbenchSidebarHostRuntime =
    useWorkspaceGeneralWorkbenchSidebarHostRuntime({
      contextActivityLogs: contextWorkspace.activityLogs,
      contextWorkspace: contextHarnessRuntime.contextWorkspace,
      generalWorkbenchHarnessSummary:
        harnessInventoryRuntime.generalWorkbenchHarnessSummary,
      generalWorkbenchScaffoldRuntime,
      handleSendRef,
      historyPageSize: GENERAL_WORKBENCH_HISTORY_PAGE_SIZE,
      isSending,
      isThemeWorkbench,
      messages,
      onAddImage: handleAddImage,
      onApplyFollowUpAction: applyWorkbenchFollowUpActionPayload,
      onCreateVersionSnapshot: handleCreateVersionSnapshot,
      onImportDocument: handleImportDocument,
      onSetBranchStatus: handleSetBranchStatus,
      onSwitchBranchVersion: handleSwitchBranchVersion,
      onViewContextDetail: handleViewContextDetail,
      projectId,
      sessionId,
      sidebarVisible: showGeneralWorkbenchSidebar,
      themeWorkbenchBackendRunState,
    });
  const {
    generalWorkbenchActivityLogs,
    generalWorkbenchWorkflowSteps,
    handleExpandGeneralWorkbenchSidebar,
    handleSubmitCodeFixPrompt,
    renderGeneralWorkbenchSidebarNode,
  } = generalWorkbenchSidebarHostRuntime;
  const {
    handleManageProvidersFromHarness,
    handleOpenExecutionPolicySettingsFromHarness,
  } = useWorkspaceHarnessNavigationRuntime({ onNavigate: _onNavigate });
  const effectiveInitialInputCapability = useMemo(
    () =>
      resolveEffectiveInitialInputCapability({
        bootstrap: initialInputCapability,
        runtime: runtimeInitialInputCapability,
      }),
    [initialInputCapability, runtimeInitialInputCapability],
  );
  const { planComposerPendingActions, planDecisionAccessory } =
    useWorkspacePlanDecisionRuntime({
      acceptedLabel: t("agentChat.planComposerDecision.option.accept"),
      displayMessages,
      effectiveChatToolPreferences,
      effectiveThreadItems,
      handlePermissionResponse,
      handleSendRef,
      isSending,
      pendingActions,
      planState: harnessShellState.plan,
      sessionId,
      submittedActionsInFlight,
    });
  const generalWorkbenchHarnessPanelBaseProps =
    useWorkspaceGeneralWorkbenchHarnessSurfaceRuntime({
      activeExecutionRuntime,
      activeTheme,
      canInterrupt: inputbarIsSending,
      canonicalChildren,
      contextHarnessRuntime,
      currentTurnId,
      executionStrategy,
      harnessInventoryRuntime,
      latestAssistantMessageId,
      messages: displayMessages,
      model,
      onInterruptCurrentTurn: stopSending,
      onLoadFilePreview: handleHarnessLoadFilePreview,
      onManageProviders: handleManageProvidersFromHarness,
      onOpenExecutionPolicySettings:
        handleOpenExecutionPolicySettingsFromHarness,
      onOpenFile: handleWorkspaceFileClick,
      onOpenSubagentSession: handleOpenSubagentSession,
      onRespondToAction: handlePermissionResponse,
      onSubmitCodeFixPrompt: handleSubmitCodeFixPrompt,
      refreshSessionReadModel,
      pendingActions: planComposerPendingActions,
      projectId,
      providerType,
      replayPendingAction,
      sessionId,
      submittedActionsInFlight,
      threadItems: effectiveThreadItems,
      threadGoal,
      threadGoalError,
      threadGoalLoading: isThreadGoalLoading,
      threadRead,
      turns,
      workingDir: project?.rootPath || null,
    });
  useWorkspaceHiddenWorkflowProgressRuntime({
    hasMessages,
    isSpecializedThemeMode,
    onWorkflowProgressChange,
  });
  const navigationActions = useWorkspaceNavigationActions({
    applyProjectSelection,
    compactSession,
    dismissWorkspacePathError,
    fixWorkspacePathAndRetry,
    agentEntry: resolvedAgentEntry,
    externalProjectId,
    onNavigate: _onNavigate,
    projectId: projectId || undefined,
    setEntryBannerVisible,
    setWorkspaceHealthError,
    workspacePathMissing,
  });
  const { handleSaveMessageAsSkill } = useWorkspaceMessageSkillSaveRuntime({
    creationProjectId: projectId,
    creationReplay: initialCreationReplay,
    onNavigate: _onNavigate,
  });

  // prettier-ignore
  return {
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
    defaultCuratedTaskReferenceEntries, defaultCuratedTaskReferenceMemoryIds, handleJumpToTimelineItem, sceneAppExecutionSummaryCard, sceneAppReviewDecisionDialogNode, serviceSkillExecutionCard, generalWorkbenchActivityLogs,
    generalWorkbenchWorkflowSteps, handleExpandGeneralWorkbenchSidebar, renderGeneralWorkbenchSidebarNode, planComposerPendingActions, planDecisionAccessory, handleSaveMessageAsSkill,
  };
}
