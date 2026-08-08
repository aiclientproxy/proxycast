import {
  useCallback,
  useMemo,
  type ComponentProps,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import type { CanvasStateUnion } from "@/components/workspace/canvas/canvasUtils";
import type { Artifact } from "@/lib/artifact/types";
import type { Character } from "@/lib/api/projectMemory";
import type { ThemeType } from "@/lib/workspace/workbenchContract";
import { EmptyState } from "../components/EmptyState";
import type { InputbarSendHandler } from "../components/Inputbar/inputbarSendPayload";
import type { CreationMode } from "../components/types";
import type { InterruptedInputRestoreRequest } from "../hooks/agentStreamInputRestoreTypes";
import type { HomeRecoverySession } from "../home/homeSurfaceTypes";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { CreationReplaySurfaceModel } from "../utils/creationReplaySurface";
import { buildAgentTaskRuntimeCardModel } from "../utils/agentTaskRuntime";
import { isCanvasStateEmpty } from "./generalWorkbenchHelpers";
import { buildWorkspaceEmptyStateProps } from "./chatSurfaceProps";
import type { useWorkspaceInputbarSceneRuntime } from "./useWorkspaceInputbarSceneRuntime";

export interface WorkspaceConversationLandingSurfaceRuntime {
  entryBannerVisible: boolean;
  entryBannerMessage?: string;
  onDismissEntryBanner: () => void;
  creationReplaySurface?: CreationReplaySurfaceModel | null;
  sceneAppExecutionSummaryCard?: ReactNode;
  serviceSkillExecutionCard?: ReactNode;
  emptyStateProps: ComponentProps<typeof EmptyState>;
}

interface OpenRuntimeMemoryWorkbenchParams {
  sessionId?: string | null;
  workingDir?: string | null;
  userMessage?: string | null;
}

type WorkspaceConversationLandingInputbarSceneRuntime = Pick<
  ReturnType<typeof useWorkspaceInputbarSceneRuntime>,
  | "runtimeToolAvailability"
  | "knowledgePackSelection"
  | "knowledgePackOptions"
  | "onToggleKnowledgePack"
  | "onSelectKnowledgePack"
  | "onToggleKnowledgeCompanionPack"
  | "onStartKnowledgeOrganize"
  | "onManageKnowledgePacks"
  | "onImportPathReferenceAsKnowledge"
>;

type UseWorkspaceConversationLandingSessionRuntimeParams = Parameters<
  typeof buildAgentTaskRuntimeCardModel
>[0] & {
  landingSurface: WorkspaceConversationLandingSurfaceRuntime;
  showChatLayout: boolean;
  sessionId?: string | null;
  projectRootPath?: string | null;
  currentUserMessage?: string | null;
  onOpenMemoryWorkbench: (params: OpenRuntimeMemoryWorkbenchParams) => void;
  onOpenChannels: () => void;
  onOpenChromeRelay: () => void;
};

interface UseWorkspaceConversationLandingSurfaceRuntimeParams {
  accessMode: ComponentProps<typeof EmptyState>["accessMode"];
  activeTheme: string;
  artifacts: Artifact[];
  browserAssistLoading: boolean;
  chatToolPreferences: ChatToolPreferences;
  contentId?: string | null;
  creationMode: CreationMode;
  creationReplaySurface?: CreationReplaySurfaceModel | null;
  defaultCuratedTaskReferenceEntries?: ComponentProps<
    typeof EmptyState
  >["defaultCuratedTaskReferenceEntries"];
  defaultCuratedTaskReferenceMemoryIds?: ComponentProps<
    typeof EmptyState
  >["defaultCuratedTaskReferenceMemoryIds"];
  emptyStateDisabled?: ComponentProps<typeof EmptyState>["disabled"];
  emptyStateIsLoading?: ComponentProps<typeof EmptyState>["isLoading"];
  emptyStateSendOnPointerDown?: ComponentProps<
    typeof EmptyState
  >["sendOnPointerDown"];
  entryBannerMessage?: string;
  entryBannerVisible: boolean;
  fileManagerOpen?: ComponentProps<typeof EmptyState>["fileManagerOpen"];
  generalCanvasContent?: string | null;
  handleSendFromEmptyState: InputbarSendHandler;
  homeRecoverySession?: HomeRecoverySession | null;
  initialInputCapability?: ComponentProps<
    typeof EmptyState
  >["initialInputCapability"];
  input: ComponentProps<typeof EmptyState>["input"];
  inputbarScene: WorkspaceConversationLandingInputbarSceneRuntime;
  inputRestoreRequest?: InterruptedInputRestoreRequest | null;
  lockTheme: boolean;
  model: ComponentProps<typeof EmptyState>["model"];
  objectiveEnabled?: ComponentProps<typeof EmptyState>["objectiveEnabled"];
  onAddPathReferences?: ComponentProps<
    typeof EmptyState
  >["onAddPathReferences"];
  onClearPathReferences?: ComponentProps<
    typeof EmptyState
  >["onClearPathReferences"];
  onDismissEntryBanner: () => void;
  onInputRestoreRequestHandled?: ComponentProps<
    typeof EmptyState
  >["onInputRestoreRequestHandled"];
  onLaunchBrowserAssist?: ComponentProps<
    typeof EmptyState
  >["onLaunchBrowserAssist"];
  onManageProviders?: ComponentProps<typeof EmptyState>["onManageProviders"];
  onNavigateToSettings?: ComponentProps<
    typeof EmptyState
  >["onNavigateToSettings"];
  onObjectiveEnabledChange?: ComponentProps<
    typeof EmptyState
  >["onObjectiveEnabledChange"];
  onOpenProjectConversation?: ComponentProps<
    typeof EmptyState
  >["onOpenProjectConversation"];
  onPluginSuggestionsNeeded?: ComponentProps<
    typeof EmptyState
  >["onPluginSuggestionsNeeded"];
  onProjectChange?: ComponentProps<typeof EmptyState>["onProjectContextChange"];
  onRecommendationClick?: ComponentProps<
    typeof EmptyState
  >["onRecommendationClick"];
  onRefreshSkills?: ComponentProps<typeof EmptyState>["onRefreshSkills"];
  onRemovePathReference?: ComponentProps<
    typeof EmptyState
  >["onRemovePathReference"];
  onResumeRecentSession?: ComponentProps<
    typeof EmptyState
  >["onResumeRecentSession"];
  onSelectServiceSkill?: ComponentProps<
    typeof EmptyState
  >["onSelectServiceSkill"];
  onStopSending?: ComponentProps<typeof EmptyState>["onStop"];
  onToggleFileManager?: ComponentProps<
    typeof EmptyState
  >["onToggleFileManager"];
  openedProjects?: ComponentProps<typeof EmptyState>["openedProjects"];
  pathReferences?: ComponentProps<typeof EmptyState>["pathReferences"];
  pluginSuggestions?: ComponentProps<typeof EmptyState>["pluginSuggestions"];
  pluginSuggestionsError?: ComponentProps<
    typeof EmptyState
  >["pluginSuggestionsError"];
  pluginSuggestionsLoading?: ComponentProps<
    typeof EmptyState
  >["pluginSuggestionsLoading"];
  projectCharacters: Character[];
  projectConversationGroups?: ComponentProps<
    typeof EmptyState
  >["projectConversationGroups"];
  projectId: string | null;
  providerType: ComponentProps<typeof EmptyState>["providerType"];
  reasoningEffort?: ComponentProps<typeof EmptyState>["reasoningEffort"];
  recentSessionActionLabel?: ComponentProps<
    typeof EmptyState
  >["recentSessionActionLabel"];
  recentSessionSummary?: ComponentProps<
    typeof EmptyState
  >["recentSessionSummary"];
  recentSessionTitle?: ComponentProps<typeof EmptyState>["recentSessionTitle"];
  resolvedCanvasState: CanvasStateUnion | null;
  sceneAppExecutionSummaryCard?: ReactNode;
  serviceSkillExecutionCard?: ReactNode;
  serviceSkillGroups: NonNullable<
    ComponentProps<typeof EmptyState>["serviceSkillGroups"]
  >;
  serviceSkills: NonNullable<
    ComponentProps<typeof EmptyState>["serviceSkills"]
  >;
  sessionId?: ComponentProps<typeof EmptyState>["sessionId"];
  setAccessMode?: ComponentProps<typeof EmptyState>["setAccessMode"];
  setActiveTheme: Dispatch<SetStateAction<string>>;
  setChatToolPreferences: Dispatch<SetStateAction<ChatToolPreferences>>;
  setCreationMode: Dispatch<SetStateAction<CreationMode>>;
  setInput: ComponentProps<typeof EmptyState>["setInput"];
  setModel: ComponentProps<typeof EmptyState>["setModel"];
  setProviderType: ComponentProps<typeof EmptyState>["setProviderType"];
  setReasoningEffort?: ComponentProps<typeof EmptyState>["setReasoningEffort"];
  skills: NonNullable<ComponentProps<typeof EmptyState>["skills"]>;
  skillsLoading: boolean;
  selectedText: ComponentProps<typeof EmptyState>["selectedText"];
  suppressRecentSessionRecovery?: boolean;
}

export function useWorkspaceConversationLandingSessionRuntime({
  landingSurface,
  showChatLayout,
  messages,
  turns,
  threadItems,
  currentTurnId,
  threadRead,
  pendingActions,
  submittedActionsInFlight,
  canonicalChildren,
  isSending,
  sessionId,
  projectRootPath,
  currentUserMessage,
  onOpenMemoryWorkbench,
  onOpenChannels,
  onOpenChromeRelay,
}: UseWorkspaceConversationLandingSessionRuntimeParams): WorkspaceConversationLandingSurfaceRuntime {
  const runtimeTaskCard = useMemo(
    () =>
      showChatLayout
        ? null
        : buildAgentTaskRuntimeCardModel({
            messages,
            turns,
            threadItems,
            currentTurnId,
            threadRead,
            pendingActions,
            submittedActionsInFlight,
            canonicalChildren,
            isSending,
          }),
    [
      canonicalChildren,
      currentTurnId,
      isSending,
      messages,
      pendingActions,
      showChatLayout,
      submittedActionsInFlight,
      threadItems,
      threadRead,
      turns,
    ],
  );
  const handleOpenMemoryWorkbench = useCallback(
    () =>
      onOpenMemoryWorkbench({
        sessionId,
        workingDir: projectRootPath,
        userMessage: currentUserMessage,
      }),
    [currentUserMessage, onOpenMemoryWorkbench, projectRootPath, sessionId],
  );

  return useMemo(
    () => ({
      ...landingSurface,
      emptyStateProps: {
        ...landingSurface.emptyStateProps,
        runtimeTaskCard,
        onOpenMemoryWorkbench: handleOpenMemoryWorkbench,
        onOpenChannels,
        onOpenChromeRelay,
      },
    }),
    [
      handleOpenMemoryWorkbench,
      landingSurface,
      onOpenChannels,
      onOpenChromeRelay,
      runtimeTaskCard,
    ],
  );
}

export function useWorkspaceConversationLandingSurfaceRuntime({
  accessMode,
  activeTheme,
  artifacts,
  browserAssistLoading,
  chatToolPreferences,
  contentId,
  creationMode,
  creationReplaySurface,
  defaultCuratedTaskReferenceEntries,
  defaultCuratedTaskReferenceMemoryIds,
  emptyStateDisabled = false,
  emptyStateIsLoading = false,
  emptyStateSendOnPointerDown = false,
  entryBannerMessage,
  entryBannerVisible,
  fileManagerOpen,
  generalCanvasContent,
  handleSendFromEmptyState,
  homeRecoverySession,
  initialInputCapability,
  input,
  inputbarScene,
  inputRestoreRequest,
  lockTheme,
  model,
  objectiveEnabled,
  onAddPathReferences,
  onClearPathReferences,
  onDismissEntryBanner,
  onInputRestoreRequestHandled,
  onLaunchBrowserAssist,
  onManageProviders,
  onNavigateToSettings,
  onObjectiveEnabledChange,
  onOpenProjectConversation,
  onPluginSuggestionsNeeded,
  onProjectChange,
  onRecommendationClick,
  onRefreshSkills,
  onRemovePathReference,
  onResumeRecentSession,
  onSelectServiceSkill,
  onStopSending,
  onToggleFileManager,
  openedProjects,
  pathReferences,
  pluginSuggestions,
  pluginSuggestionsError,
  pluginSuggestionsLoading,
  projectCharacters,
  projectConversationGroups,
  projectId,
  providerType,
  reasoningEffort,
  recentSessionActionLabel,
  recentSessionSummary,
  recentSessionTitle,
  resolvedCanvasState,
  sceneAppExecutionSummaryCard,
  serviceSkillExecutionCard,
  serviceSkillGroups,
  serviceSkills,
  sessionId,
  setAccessMode,
  setActiveTheme,
  setChatToolPreferences,
  setCreationMode,
  setInput,
  setModel,
  setProviderType,
  setReasoningEffort,
  skills,
  skillsLoading,
  selectedText,
  suppressRecentSessionRecovery = false,
}: UseWorkspaceConversationLandingSurfaceRuntimeParams): WorkspaceConversationLandingSurfaceRuntime {
  const {
    knowledgePackOptions,
    knowledgePackSelection,
    onImportPathReferenceAsKnowledge,
    onManageKnowledgePacks,
    onSelectKnowledgePack,
    onStartKnowledgeOrganize,
    onToggleKnowledgeCompanionPack,
    onToggleKnowledgePack,
    runtimeToolAvailability,
  } = inputbarScene;
  const handleToolPreferenceChange = useCallback(
    (key: keyof ChatToolPreferences, enabled: boolean) =>
      setChatToolPreferences((previous) => ({
        ...previous,
        [key]: enabled,
      })),
    [setChatToolPreferences],
  );

  const emptyStateProps = useMemo(
    () =>
      buildWorkspaceEmptyStateProps({
        input,
        setInput,
        onSendMessage: handleSendFromEmptyState,
        onStopSending,
        sendOnPointerDown: emptyStateSendOnPointerDown,
        isLoading: emptyStateIsLoading,
        disabled: emptyStateDisabled,
        providerType,
        setProviderType,
        model,
        setModel,
        reasoningEffort,
        setReasoningEffort,
        accessMode,
        setAccessMode,
        onManageProviders,
        toolPreferences: chatToolPreferences,
        onToolPreferenceChange: handleToolPreferenceChange,
        objectiveEnabled,
        onObjectiveEnabledChange,
        creationMode,
        onCreationModeChange: setCreationMode,
        activeTheme: activeTheme as ThemeType,
        onThemeChange: setActiveTheme,
        themeLocked: lockTheme,
        hasCanvasContent:
          activeTheme === "general"
            ? artifacts.length > 0 || Boolean(generalCanvasContent?.trim())
            : !isCanvasStateEmpty(resolvedCanvasState),
        hasContentId: Boolean(contentId),
        selectedText,
        onRecommendationClick,
        characters: projectCharacters,
        skills,
        serviceSkills,
        serviceSkillGroups,
        isSkillsLoading: skillsLoading,
        onSelectServiceSkill,
        onNavigateToSettings,
        onRefreshSkills,
        onLaunchBrowserAssist,
        browserAssistLoading,
        recentSessionTitle: suppressRecentSessionRecovery
          ? undefined
          : recentSessionTitle,
        recentSessionSummary: suppressRecentSessionRecovery
          ? undefined
          : recentSessionSummary,
        recentSessionActionLabel: suppressRecentSessionRecovery
          ? undefined
          : recentSessionActionLabel,
        homeRecoverySession: suppressRecentSessionRecovery
          ? undefined
          : homeRecoverySession,
        onResumeRecentSession: suppressRecentSessionRecovery
          ? undefined
          : onResumeRecentSession,
        projectConversationGroups: suppressRecentSessionRecovery
          ? undefined
          : projectConversationGroups,
        onOpenProjectConversation: suppressRecentSessionRecovery
          ? undefined
          : onOpenProjectConversation,
        projectId,
        openedProjects,
        onProjectChange,
        sessionId,
        runtimeToolAvailability,
        pluginSuggestions,
        pluginSuggestionsError,
        pluginSuggestionsLoading,
        onPluginSuggestionsNeeded,
        initialInputCapability,
        knowledgePackSelection,
        knowledgePackOptions,
        onToggleKnowledgePack,
        onSelectKnowledgePack,
        onToggleKnowledgeCompanionPack,
        onStartKnowledgeOrganize,
        onManageKnowledgePacks,
        runtimeTaskCard: null,
        onOpenMemoryWorkbench: undefined,
        onOpenChannels: undefined,
        onOpenChromeRelay: undefined,
        creationReplaySurface,
        defaultCuratedTaskReferenceMemoryIds,
        defaultCuratedTaskReferenceEntries,
        pathReferences,
        onAddPathReferences,
        inputRestoreRequest,
        onInputRestoreRequestHandled,
        onImportPathReferenceAsKnowledge,
        onRemovePathReference,
        onClearPathReferences,
        fileManagerOpen,
        onToggleFileManager,
      }),
    [
      accessMode,
      activeTheme,
      artifacts,
      browserAssistLoading,
      chatToolPreferences,
      contentId,
      creationMode,
      creationReplaySurface,
      defaultCuratedTaskReferenceEntries,
      defaultCuratedTaskReferenceMemoryIds,
      emptyStateDisabled,
      emptyStateIsLoading,
      emptyStateSendOnPointerDown,
      fileManagerOpen,
      generalCanvasContent,
      handleSendFromEmptyState,
      handleToolPreferenceChange,
      homeRecoverySession,
      initialInputCapability,
      input,
      inputRestoreRequest,
      lockTheme,
      model,
      objectiveEnabled,
      onAddPathReferences,
      onClearPathReferences,
      onImportPathReferenceAsKnowledge,
      onInputRestoreRequestHandled,
      onLaunchBrowserAssist,
      onManageKnowledgePacks,
      onManageProviders,
      onNavigateToSettings,
      onObjectiveEnabledChange,
      onOpenProjectConversation,
      onPluginSuggestionsNeeded,
      onProjectChange,
      onRecommendationClick,
      onRefreshSkills,
      onRemovePathReference,
      onResumeRecentSession,
      onSelectKnowledgePack,
      onSelectServiceSkill,
      onStartKnowledgeOrganize,
      onStopSending,
      onToggleFileManager,
      onToggleKnowledgeCompanionPack,
      onToggleKnowledgePack,
      openedProjects,
      pathReferences,
      pluginSuggestions,
      pluginSuggestionsError,
      pluginSuggestionsLoading,
      projectCharacters,
      projectConversationGroups,
      projectId,
      providerType,
      reasoningEffort,
      recentSessionActionLabel,
      recentSessionSummary,
      recentSessionTitle,
      resolvedCanvasState,
      runtimeToolAvailability,
      serviceSkillGroups,
      serviceSkills,
      sessionId,
      setAccessMode,
      setActiveTheme,
      setCreationMode,
      setInput,
      setModel,
      setProviderType,
      setReasoningEffort,
      skills,
      skillsLoading,
      selectedText,
      suppressRecentSessionRecovery,
      knowledgePackOptions,
      knowledgePackSelection,
    ],
  );

  return useMemo(
    () => ({
      entryBannerVisible,
      entryBannerMessage,
      onDismissEntryBanner,
      creationReplaySurface,
      sceneAppExecutionSummaryCard,
      serviceSkillExecutionCard,
      emptyStateProps,
    }),
    [
      creationReplaySurface,
      emptyStateProps,
      entryBannerMessage,
      entryBannerVisible,
      onDismissEntryBanner,
      sceneAppExecutionSummaryCard,
      serviceSkillExecutionCard,
    ],
  );
}
