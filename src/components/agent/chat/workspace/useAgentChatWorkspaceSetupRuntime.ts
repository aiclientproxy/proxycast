/** Agent runtime/read-model bootstrap current owner。 */

import { useCallback, useEffect, useMemo, useRef } from "react";
import { useAgentChatUnified } from "../hooks";
import type { AgentChatWorkspaceProps } from "../agentChatWorkspaceContract";
import { useWorkspaceHarnessInventoryRuntime } from "./useWorkspaceHarnessInventoryRuntime";
import { useWorkspaceImageWorkbenchSessionRuntime } from "./useWorkspaceImageWorkbenchSessionRuntime";
import { useWorkspaceSystemPromptRuntime } from "./useWorkspaceSystemPromptRuntime";
import { useWorkspaceGeneralWorkbenchScaffoldRuntime } from "./useWorkspaceGeneralWorkbenchScaffoldRuntime";
import { useWorkspaceGeneralWorkbenchRuntime } from "./useWorkspaceGeneralWorkbenchRuntime";
import { useWorkspaceTeamRuntime } from "./useWorkspaceTeamRuntime";
import { useWorkspaceGeneralWorkbenchDocumentPersistenceRuntime } from "./useWorkspaceGeneralWorkbenchDocumentPersistenceRuntime";
import { useWorkspaceServiceSkillEntryActions } from "./useWorkspaceServiceSkillEntryActions";
import { useWorkspaceHealthRuntime } from "./useWorkspaceHealthRuntime";
import { resolveBrowserRuntimeNavigationFromSiteSkill } from "./workspaceBrowserRuntimeNavigation";
import { useWorkspaceContextSurfaceRuntime } from "./useWorkspaceContextSurfaceRuntime";
import {
  createRestoredInteractiveMessageSnapshot,
  resolveReadOnlyInteractiveMessageIds,
} from "./workspaceRestoredInteractiveMessages";
import { useWorkspaceDebugRuntime } from "./useWorkspaceDebugRuntime";
import { useWorkspaceClassicClawSidebarRuntime } from "./useWorkspaceClassicClawSidebarRuntime";
import { useWorkspaceChatToolPreferencesRuntime } from "./useWorkspaceChatToolPreferencesRuntime";
import { useWorkspaceArtifactCanvasRuntime } from "./useWorkspaceArtifactCanvasRuntime";
import { useWorkspaceExpertSkillPanelRuntime } from "./useWorkspaceExpertSkillPanelRuntime";
import { useWorkspaceSubagentNavigationRuntime } from "./useWorkspaceSubagentNavigationRuntime";
import { useWorkspaceContextDetailRuntime } from "./useWorkspaceContextDetailRuntime";
import { useWorkspaceHarnessRequestMetadataRuntime } from "./useWorkspaceHarnessRequestMetadataRuntime";
import { useWorkspacePendingInputRuntime } from "./useWorkspacePendingInputRuntime";
import {
  GENERAL_BROWSER_ASSIST_PROFILE_KEY,
  NOOP_SET_CHAT_MESSAGES,
} from "./agentChatWorkspaceHelpers";

import type { useAgentChatWorkspaceEntryRuntime } from "./useAgentChatWorkspaceEntryRuntime";

type EntryRuntime = ReturnType<typeof useAgentChatWorkspaceEntryRuntime>;

interface UseAgentChatWorkspaceSetupRuntimeParams {
  props: AgentChatWorkspaceProps;
  entryRuntime: EntryRuntime;
}

export function useAgentChatWorkspaceSetupRuntime({
  props,
  entryRuntime,
}: UseAgentChatWorkspaceSetupRuntimeParams) {
  const {
    onNavigate: _onNavigate,
    projectId: externalProjectId,
    contentId,
    initialRequestMetadata,
    initialAutoSendRequestMetadata,
    agentEntry = "claw",
    theme: initialTheme,
    initialCreationMode,
    lockTheme = false,
    initialUserPrompt,
    initialPendingServiceSkillLaunch,
    newChatAt,
    expertAgentLaunch,
    onSessionChange,
    onAgentStreamingChange,
    openBrowserAssistOnMount = false,
    initialSiteSkillLaunch,
  } = props;

  const {
    t,
    normalizedEntryTheme,
    shouldAutoCollapseClassicClawSidebar,
    activeTheme,
    creationMode,
    handleRestoreInterruptedInput,
    input,
    layoutMode,
    setLayoutMode,
    setShowSidebar,
    activeSessionIdRef,
    deferSessionRecentMetadataSyncForNavigation,
    syncSessionRecentPreferences,
    chatToolPreferences,
    setChatToolPreferences,
    syncChatToolPreferencesSource,
    getSyncedSessionRecentPreferences,
    handleOpenSkillsManageFromExpertPanel,
    handleOpenSubagents,
    projectId,
    shouldDisableSessionRestore,
    applyProjectSelection,
    normalizedInitialSessionId,
    sessionRestorePresentation,
    shouldDeferWorkspaceAuxiliaryLoads,
    shouldDeferInitialTopicsLoad,
    shouldDeferInitialRuntimeWarmup,
    deferredWorkspaceAuxiliaryLoadMs,
    deferredInitialTopicsLoadMs,
    deferredInitialRuntimeWarmupMs,
    project,
    projectMemory,
    isInitialContentLoading,
    initialContentLoadError,
    canvasState,
    setCanvasState,
    documentVersionStatusMap,
    setDocumentVersionStatusMap,
    contentMetadataRef,
    persistedWorkbenchSnapshotRef,
    dismissedInitialPendingServiceSkillLaunchSignatureRef,
    handledInitialPendingServiceSkillLaunchSignatureRef,
    initialCreationReplay,
    initialPendingServiceSkillLaunchSignature,
    runtimeWorkspaceId,
    generalCanvasState,
    clawTraceEnabled,
    workspaceHarnessEnabled,
    agentResponseLanguage,
    soulInteractionCopy,
    ensureImageWorkbenchProvidersLoaded,
    skills,
    skillsLoading,
    serviceSkills,
    serviceSkillsLoading,
    serviceSkillsError,
    recordServiceSkillUsage,
    clearThemeSkillsRailState,
    handleWriteFileRef,
    sceneGateResumeHandlerRef,
    mappedTheme,
    isSpecializedThemeMode,
    workbenchRequests,
  } = entryRuntime;

  const { chatMode, generalHarnessEntryEnabled, systemPrompt } =
    useWorkspaceSystemPromptRuntime({
      chatToolPreferences,
      contentId,
      creationMode,
      isSpecializedThemeMode,
      mappedTheme,
      projectMemory,
    });

  // 使用 Agent Chat Hook（传递系统提示词）
  const agentChatRuntime = useAgentChatUnified({
    systemPrompt,
    onWriteFile: (content, fileName, context) => {
      // 使用 ref 调用最新的 handleWriteFile
      handleWriteFileRef.current?.(content, fileName, context);
    },
    workspaceId: runtimeWorkspaceId,
    workingDir: project?.rootPath || null,
    disableSessionRestore: shouldDisableSessionRestore,
    sessionRestorePresentation,
    initialTopicsLoadMode: shouldDeferInitialTopicsLoad
      ? "deferred"
      : "immediate",
    initialTopicsDeferredDelayMs: shouldDeferInitialTopicsLoad
      ? deferredInitialTopicsLoadMs
      : undefined,
    initialRuntimeWarmupLoadMode: shouldDeferInitialRuntimeWarmup
      ? "deferred"
      : "immediate",
    initialRuntimeWarmupDeferredDelayMs: shouldDeferInitialRuntimeWarmup
      ? deferredInitialRuntimeWarmupMs
      : undefined,
    getSyncedSessionRecentPreferences,
    onOpenSubagents: handleOpenSubagents,
    onRestoreInterruptedInput: handleRestoreInterruptedInput,
    clawTraceEnabled,
    soulCopy: soulInteractionCopy,
  });
  const {
    providerType,
    setProviderType,
    model,
    setModel,
    reasoningEffort,
    setReasoningEffort,
    executionStrategy,
    accessMode,
    setAccessMode,
    messages = [],
    setMessages: setChatMessages = NOOP_SET_CHAT_MESSAGES,
    currentTurnId,
    turns = [],
    threadItems = [],
    todoItems = [],
    queuedTurnCount = 0,
    threadRead = null,
    threadGoal = null,
    threadGoalError = null,
    isThreadGoalLoading = false,
    executionRuntime = null,
    sessionWorkingDir = null,
    activeExecutionRuntime = null,
    isSending,
    sendMessage,
    compactSession = async () => undefined,
    stopSending,
    replayPendingAction = async () => false,
    clearMessages,
    deleteMessage,
    editMessage,
    handlePermissionResponse,
    pendingActions = [],
    submittedActionsInFlight = [],
    triggerAIGuide,
    topics = [],
    sessionHistoryWindow = null,
    isAutoRestoringSession = false,
    isSessionHydrating = false,
    sessionId,
    ensureSession = async () => null,
    switchTopic: originalSwitchTopic,
    loadFullSessionHistory = async () => false,
    refreshSessionReadModel = async () => false,
    workspacePathMissing = false,
    fixWorkspacePathAndRetry,
    dismissWorkspacePathError,
  } = agentChatRuntime;
  const { workspaceHealthError, setWorkspaceHealthError } =
    useWorkspaceHealthRuntime({
      enabled:
        !shouldDeferWorkspaceAuxiliaryLoads || Boolean(workspacePathMissing),
      project,
      projectId,
      workspacePathMissing,
      shouldDeferWorkspaceAuxiliaryLoads,
      deferredWorkspaceAuxiliaryLoadMs,
    });
  const activeSessionKey = sessionId?.trim() || null;
  const {
    combinedSkillsLoading,
    expertPanelRequestMetadata,
    expertPanelRuntimeKey,
    expertSkillRefsOverride,
    expertWorkspaceSkillRuntimeEnableBindings,
    expertWorkspaceSkillRuntimeEnableInput,
    expertWorkspaceSkillRuntimeEnableRefs,
    handleEnableExpertWorkspaceSkillRuntime,
    handleExpertSkillRefsChange,
    handlePluginSuggestionsNeeded,
    handleThreadExpertProfileSwitch,
    workspacePluginInputSuggestions,
    workspacePluginSuggestionsError,
    workspacePluginSuggestionsLoading,
    workspaceRequestMetadataWithExpertSkills,
    workspaceSkillBindings,
  } = useWorkspaceExpertSkillPanelRuntime({
    activeSessionKey,
    activeTheme,
    deferredDelayMs: shouldDeferWorkspaceAuxiliaryLoads
      ? deferredWorkspaceAuxiliaryLoadMs
      : undefined,
    expertAgentLaunch,
    initialAutoSendRequestMetadata,
    initialRequestMetadata,
    newChatAt,
    onOpenSkillsManage: _onNavigate
      ? handleOpenSkillsManageFromExpertPanel
      : undefined,
    serviceSkillsLoading,
    skillsLoading,
    threadRead,
    workspaceRoot: project?.rootPath,
  });
  const restoredInteractiveMessageSnapshotRef = useRef(
    createRestoredInteractiveMessageSnapshot(),
  );
  const readOnlyInteractiveMessageIds = useMemo<ReadonlySet<string>>(() => {
    return resolveReadOnlyInteractiveMessageIds({
      snapshot: restoredInteractiveMessageSnapshotRef.current,
      activeSessionKey,
      messages,
      normalizedInitialSessionId,
      isAutoRestoringSession,
      isSessionHydrating,
      isLoadingFullSessionHistory: sessionHistoryWindow?.isLoadingFull === true,
    });
  }, [
    activeSessionKey,
    isAutoRestoringSession,
    isSessionHydrating,
    messages,
    normalizedInitialSessionId,
    sessionHistoryWindow?.isLoadingFull,
  ]);
  const topicById = useMemo(
    () => new Map(topics.map((topic) => [topic.id, topic])),
    [topics],
  );
  activeSessionIdRef.current = sessionId;
  const { autoCollapsedTopicSidebarRef } =
    useWorkspaceClassicClawSidebarRuntime({
      contentId,
      externalProjectId,
      newChatAt,
      normalizedEntryTheme,
      sessionId,
      shouldAutoCollapseClassicClawSidebar,
      setShowSidebar,
    });
  const effectiveChatToolPreferences = useWorkspaceChatToolPreferencesRuntime({
    activeTheme,
    chatToolPreferences,
    executionRuntime,
    executionStrategy,
    sessionId,
    setChatToolPreferences,
    syncChatToolPreferencesSource,
    syncSessionRecentPreferences,
  });

  const {
    canonicalChildren,
    currentSessionTitle,
    handleStopSending,
    hasRuntimeSessions,
    subagentsRuntimeVisible,
  } = useWorkspaceTeamRuntime({
    canonicalRefreshKey: threadItems
      .map((item) => `${item.id}:${item.status}:${item.updated_at}`)
      .join("|"),
    referencedChildThreadIds: threadItems.flatMap((item) =>
      item.type === "subagent_activity" && item.session_id?.trim()
        ? [item.session_id.trim()]
        : [],
    ),
    session: {
      currentTopicId: sessionId,
      parentThreadId: threadRead?.thread_id ?? sessionId,
      topics,
      subagentEnabled: effectiveChatToolPreferences.subagent,
    },
    stopSending,
  });
  const { handleOpenSubagentSession } = useWorkspaceSubagentNavigationRuntime({
    canonicalChildren,
    deferSessionRecentMetadataSyncForNavigation,
    switchTopic: originalSwitchTopic,
  });
  const imageWorkbenchSessionRuntime = useWorkspaceImageWorkbenchSessionRuntime(
    {
      contentId,
      messages,
      projectId,
      sessionId,
    },
  );
  const {
    currentImageWorkbenchState,
    imageWorkbenchSessionKey,
    updateCurrentImageWorkbenchState,
  } = imageWorkbenchSessionRuntime;
  useEffect(() => {
    if (!shouldDeferWorkspaceAuxiliaryLoads) {
      return;
    }
    if (
      !currentImageWorkbenchState.active &&
      currentImageWorkbenchState.tasks.length === 0
    ) {
      return;
    }

    ensureImageWorkbenchProvidersLoaded();
  }, [
    currentImageWorkbenchState.active,
    currentImageWorkbenchState.tasks.length,
    ensureImageWorkbenchProvidersLoaded,
    shouldDeferWorkspaceAuxiliaryLoads,
  ]);
  useWorkspaceDebugRuntime({
    agentEntry,
    contentId,
    externalProjectId,
    initialCreationMode,
    initialTheme,
    lockTheme,
    stateSnapshot: {
      activeTheme,
      contentId: contentId ?? null,
      initialContentLoadError: initialContentLoadError ?? null,
      isAutoRestoringSession,
      isInitialContentLoading,
      isSessionHydrating,
      isSending,
      layoutMode,
      messagesCount: messages.length,
      projectId: projectId ?? null,
      sessionId: sessionId ?? null,
      skillsCount: skills.length,
      skillsLoading: combinedSkillsLoading,
      topicsCount: topics.length,
      workspaceHealthError,
    },
  });
  const artifactCanvasRuntime = useWorkspaceArtifactCanvasRuntime({
    activeTheme,
    contentId,
    generalBrowserAssistProfileKey: GENERAL_BROWSER_ASSIST_PROFILE_KEY,
    generalCanvasState,
    input,
    initialAutoSendRequestMetadata,
    initialSiteSkillLaunch,
    initialUserPrompt,
    mappedTheme,
    messages,
    onNavigate: _onNavigate,
    openBrowserAssistOnMount,
    projectId,
    sessionId,
    setLayoutMode,
    siteSkillLaunchNonce: newChatAt,
    isSending,
    workbenchRequests,
  });
  const {
    artifacts,
    artifactDisplayState,
    artifactOpenControl: browserAssistArtifactOpenControl,
    artifactViewMode,
    applyAutoArtifactViewMode,
    browserAssistLaunching,
    browserAssistRequestAutoLaunch,
    browserAssistRequestPreferredBackend,
    browserAssistRequestProfileKey,
    browserAssistSessionRef,
    browserAssistSessionState,
    currentCanvasArtifact,
    currentBrowserAssistScopeKey,
    displayedCanvasArtifact,
    ensureBrowserAssistCanvas,
    handleArtifactViewModeChange,
    handleOpenBrowserRuntimeForBrowserAssist,
    setSelectedArtifactId,
    settledWorkbenchArtifacts,
    siteSkillExecutionState,
    upsertGeneralArtifact,
  } = artifactCanvasRuntime;
  const handleOpenBrowserRuntimeForSiteSkillExecution = useCallback(() => {
    if (!_onNavigate || !initialSiteSkillLaunch?.adapterName?.trim()) {
      return;
    }

    _onNavigate(
      "browser-runtime",
      resolveBrowserRuntimeNavigationFromSiteSkill({
        contentId,
        initialSiteSkillLaunch,
        projectId,
        siteSkillExecutionState,
      }),
    );
  }, [
    contentId,
    initialSiteSkillLaunch,
    _onNavigate,
    projectId,
    siteSkillExecutionState,
  ]);
  const contextSurfaceRuntime = useWorkspaceContextSurfaceRuntime({
    activeTheme,
    generalHarnessEntryEnabled,
    isSending,
    layoutMode,
    mappedTheme,
    messages,
    model,
    onAgentStreamingChange,
    onSessionChange,
    pendingActions,
    projectId,
    projectMemory,
    providerType,
    sessionId,
    threadItems,
    threadRead,
    todoItems,
    workspaceHarnessEnabled,
  });
  const {
    contextHarnessRuntime,
    effectiveThreadItems,
    harnessState,
    harnessRuntimeVisible,
    harnessShellState,
    inputbarIsSending,
    rightSurfaceLocalState,
  } = contextSurfaceRuntime;
  const { openArticleWorkspaceRightSurface } = rightSurfaceLocalState;
  const {
    contextWorkspace,
    isThemeWorkbench,
    setHarnessPanelVisible,
    harnessPendingCount,
    showHarnessToggle,
    harnessAttentionLevel,
    harnessToggleLabel,
  } = contextHarnessRuntime;
  const generalWorkbenchScaffoldRuntime =
    useWorkspaceGeneralWorkbenchScaffoldRuntime({
      isGeneralWorkbench: isThemeWorkbench,
      mappedTheme,
      sessionId,
      projectId,
      canvasState,
      documentVersionStatusMap,
      setDocumentVersionStatusMap,
      clearThemeSkillsRailState,
      setCanvasState,
      setLayoutMode,
    });
  const {
    shouldUseCompactGeneralWorkbench,
    shouldSkipGeneralWorkbenchAutoGuideWithoutPrompt,
    setTopicStatus,
  } = generalWorkbenchScaffoldRuntime;

  useWorkspaceGeneralWorkbenchDocumentPersistenceRuntime({
    isThemeWorkbench,
    contentId,
    canvasState,
    documentVersionStatusMap,
    contentMetadataRef,
    persistedWorkbenchSnapshotRef,
  });

  const workspaceServiceSkillEntryActions =
    useWorkspaceServiceSkillEntryActions({
      activeTheme,
      creationMode,
      projectId,
      contentId,
      sessionId,
      threadId: threadRead?.thread_id ?? sessionId,
      ensureSessionForThreadLineage: ensureSession,
      input,
      chatToolPreferences: effectiveChatToolPreferences,
      creationReplay: initialCreationReplay,
      onNavigate: _onNavigate,
      recordServiceSkillUsage,
    });
  const {
    a2uiSubmissionNotice,
    clearEntryPendingA2UI,
    effectivePendingA2UIForm,
    effectivePendingA2UISource,
    handleMessageA2UISubmit,
    handlePendingA2UISubmit,
    hasPendingA2UIForm,
    openRuntimeSceneGate,
    pendingActionRequest,
    pendingPromotedA2UIActionRequest,
  } = useWorkspacePendingInputRuntime({
    activeTheme,
    applyProjectSelection,
    clearPendingServiceSkillLaunch:
      workspaceServiceSkillEntryActions.clearPendingServiceSkillLaunch,
    contentId,
    creationReplay: initialCreationReplay,
    dismissedInitialPendingServiceSkillLaunchSignatureRef,
    handlePendingServiceSkillLaunchSubmit:
      workspaceServiceSkillEntryActions.handlePendingServiceSkillLaunchSubmit,
    handlePermissionResponse,
    handledInitialPendingServiceSkillLaunchSignatureRef,
    initialPendingServiceSkillLaunch,
    initialPendingServiceSkillLaunchSignature,
    messages,
    onSelectServiceSkill:
      workspaceServiceSkillEntryActions.handleServiceSkillSelect,
    pendingActions,
    pendingServiceSkillLaunchForm:
      workspaceServiceSkillEntryActions.pendingServiceSkillLaunchForm,
    pendingServiceSkillLaunchSource:
      workspaceServiceSkillEntryActions.pendingServiceSkillLaunchSource,
    projectId,
    readOnlyInteractiveMessageIds,
    sceneGateResumeHandlerRef,
    sendMessage,
    serviceSkills,
    serviceSkillsError,
    serviceSkillsLoading,
    submittedActionsInFlight,
  });

  const {
    currentGate,
    documentEditorFocusedRef,
    themeWorkbenchActiveQueueItem,
    themeWorkbenchBackendRunState,
    themeWorkbenchRunState,
  } = useWorkspaceGeneralWorkbenchRuntime({
    isThemeWorkbench,
    sessionId,
    isSending,
    pendingActionRequest,
  });

  const handleViewContextDetail = useWorkspaceContextDetailRuntime({
    contextWorkspace,
    t,
  });

  const harnessRequestMetadata = useWorkspaceHarnessRequestMetadataRuntime({
    enabled: workspaceHarnessEnabled && harnessRuntimeVisible,
    agentResponseLanguage,
    browserAssistAutoLaunch: browserAssistRequestAutoLaunch,
    browserAssistPreferredBackend: browserAssistRequestPreferredBackend,
    browserAssistProfileKey: browserAssistRequestProfileKey,
    contentId,
    currentGateKey: currentGate.key,
    effectiveChatToolPreferences,
    isThemeWorkbench,
    mappedTheme,
    themeWorkbenchActiveQueueTitle: themeWorkbenchActiveQueueItem?.title,
    workspaceSkillBindings,
    workspaceSkillRuntimeEnable: expertWorkspaceSkillRuntimeEnableInput,
  });
  const harnessInventoryRuntime = useWorkspaceHarnessInventoryRuntime({
    enabled: workspaceHarnessEnabled,
    chatMode,
    mappedTheme,
    harnessPanelVisible: harnessRuntimeVisible,
    harnessRequestMetadata,
    isThemeWorkbench,
    themeWorkbenchRunState,
    currentGate,
    themeWorkbenchBackendRunState,
    themeWorkbenchActiveQueueItem,
    harnessPendingCount,
  });

  return {
    chatMode,
    generalHarnessEntryEnabled,
    systemPrompt,
    agentChatRuntime,
    providerType,
    setProviderType,
    model,
    setModel,
    reasoningEffort,
    setReasoningEffort,
    executionStrategy,
    accessMode,
    setAccessMode,
    messages,
    setChatMessages,
    currentTurnId,
    turns,
    threadItems,
    todoItems,
    queuedTurnCount,
    threadRead,
    threadGoal,
    threadGoalError,
    isThreadGoalLoading,
    executionRuntime,
    sessionWorkingDir,
    activeExecutionRuntime,
    isSending,
    sendMessage,
    compactSession,
    stopSending,
    replayPendingAction,
    clearMessages,
    deleteMessage,
    editMessage,
    handlePermissionResponse,
    pendingActions,
    submittedActionsInFlight,
    triggerAIGuide,
    topics,
    sessionHistoryWindow,
    isAutoRestoringSession,
    isSessionHydrating,
    sessionId,
    ensureSession,
    originalSwitchTopic,
    loadFullSessionHistory,
    refreshSessionReadModel,
    workspacePathMissing,
    fixWorkspacePathAndRetry,
    dismissWorkspacePathError,
    workspaceHealthError,
    setWorkspaceHealthError,
    activeSessionKey,
    combinedSkillsLoading,
    expertPanelRequestMetadata,
    expertPanelRuntimeKey,
    expertSkillRefsOverride,
    expertWorkspaceSkillRuntimeEnableBindings,
    expertWorkspaceSkillRuntimeEnableInput,
    expertWorkspaceSkillRuntimeEnableRefs,
    handleEnableExpertWorkspaceSkillRuntime,
    handleExpertSkillRefsChange,
    handlePluginSuggestionsNeeded,
    handleThreadExpertProfileSwitch,
    workspacePluginInputSuggestions,
    workspacePluginSuggestionsError,
    workspacePluginSuggestionsLoading,
    workspaceRequestMetadataWithExpertSkills,
    workspaceSkillBindings,
    restoredInteractiveMessageSnapshotRef,
    readOnlyInteractiveMessageIds,
    topicById,
    autoCollapsedTopicSidebarRef,
    effectiveChatToolPreferences,
    canonicalChildren,
    currentSessionTitle,
    handleStopSending,
    hasRuntimeSessions,
    subagentsRuntimeVisible,
    handleOpenSubagentSession,
    imageWorkbenchSessionRuntime,
    currentImageWorkbenchState,
    imageWorkbenchSessionKey,
    updateCurrentImageWorkbenchState,
    artifactCanvasRuntime,
    artifacts,
    artifactDisplayState,
    browserAssistArtifactOpenControl,
    artifactViewMode,
    applyAutoArtifactViewMode,
    browserAssistLaunching,
    browserAssistRequestAutoLaunch,
    browserAssistRequestPreferredBackend,
    browserAssistRequestProfileKey,
    browserAssistSessionRef,
    browserAssistSessionState,
    currentCanvasArtifact,
    currentBrowserAssistScopeKey,
    displayedCanvasArtifact,
    ensureBrowserAssistCanvas,
    handleArtifactViewModeChange,
    handleOpenBrowserRuntimeForBrowserAssist,
    setSelectedArtifactId,
    settledWorkbenchArtifacts,
    siteSkillExecutionState,
    upsertGeneralArtifact,
    handleOpenBrowserRuntimeForSiteSkillExecution,
    contextSurfaceRuntime,
    contextHarnessRuntime,
    effectiveThreadItems,
    harnessState,
    harnessRuntimeVisible,
    harnessShellState,
    inputbarIsSending,
    rightSurfaceLocalState,
    openArticleWorkspaceRightSurface,
    contextWorkspace,
    isThemeWorkbench,
    setHarnessPanelVisible,
    harnessPendingCount,
    showHarnessToggle,
    harnessAttentionLevel,
    harnessToggleLabel,
    generalWorkbenchScaffoldRuntime,
    shouldUseCompactGeneralWorkbench,
    shouldSkipGeneralWorkbenchAutoGuideWithoutPrompt,
    setTopicStatus,
    workspaceServiceSkillEntryActions,
    a2uiSubmissionNotice,
    clearEntryPendingA2UI,
    effectivePendingA2UIForm,
    effectivePendingA2UISource,
    handleMessageA2UISubmit,
    handlePendingA2UISubmit,
    hasPendingA2UIForm,
    openRuntimeSceneGate,
    pendingActionRequest,
    pendingPromotedA2UIActionRequest,
    currentGate,
    documentEditorFocusedRef,
    themeWorkbenchActiveQueueItem,
    themeWorkbenchBackendRunState,
    themeWorkbenchRunState,
    handleViewContextDetail,
    harnessRequestMetadata,
    harnessInventoryRuntime,
  };
}
