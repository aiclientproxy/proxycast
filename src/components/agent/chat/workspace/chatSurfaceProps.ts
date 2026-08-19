import type { ComponentProps } from "react";
import { StepProgress } from "@/components/workspace/layout/StepProgress";
import { ChatNavbar } from "../components/ChatNavbar";
import { EmptyState } from "../components/EmptyState";
import type { InputbarSendHandler } from "../components/Inputbar/inputbarSendPayload";
import type { CreationReplaySurfaceModel } from "../utils/creationReplaySurface";

type ChatToolPreferences = {
  task: boolean;
  subagent: boolean;
};

type ChatToolPreferenceKey = keyof ChatToolPreferences;

interface BuildStepProgressPropsParams {
  hidden: boolean;
  isSpecializedThemeMode: boolean;
  hasMessages: boolean;
  steps: ComponentProps<typeof StepProgress>["steps"];
  currentIndex: ComponentProps<typeof StepProgress>["currentIndex"];
  onStepClick: NonNullable<ComponentProps<typeof StepProgress>["onStepClick"]>;
}

export function buildStepProgressProps({
  hidden,
  isSpecializedThemeMode,
  hasMessages,
  steps,
  currentIndex,
  onStepClick,
}: BuildStepProgressPropsParams): ComponentProps<typeof StepProgress> | null {
  if (hidden || !isSpecializedThemeMode || !hasMessages || steps.length === 0) {
    return null;
  }

  return {
    steps,
    currentIndex,
    onStepClick,
  };
}

interface BuildWorkspaceEmptyStatePropsParams {
  input: ComponentProps<typeof EmptyState>["input"];
  setInput: ComponentProps<typeof EmptyState>["setInput"];
  onSendMessage: InputbarSendHandler;
  onStopSending?: ComponentProps<typeof EmptyState>["onStop"];
  sendOnPointerDown?: ComponentProps<typeof EmptyState>["sendOnPointerDown"];
  isLoading: ComponentProps<typeof EmptyState>["isLoading"];
  disabled: ComponentProps<typeof EmptyState>["disabled"];
  providerType: ComponentProps<typeof EmptyState>["providerType"];
  setProviderType: ComponentProps<typeof EmptyState>["setProviderType"];
  model: ComponentProps<typeof EmptyState>["model"];
  setModel: ComponentProps<typeof EmptyState>["setModel"];
  reasoningEffort?: ComponentProps<typeof EmptyState>["reasoningEffort"];
  setReasoningEffort?: ComponentProps<typeof EmptyState>["setReasoningEffort"];
  accessMode: ComponentProps<typeof EmptyState>["accessMode"];
  setAccessMode: ComponentProps<typeof EmptyState>["setAccessMode"];
  onManageProviders?: ComponentProps<typeof EmptyState>["onManageProviders"];
  toolPreferences: ChatToolPreferences;
  onToolPreferenceChange: (
    key: ChatToolPreferenceKey,
    enabled: boolean,
  ) => void;
  objectiveEnabled?: ComponentProps<typeof EmptyState>["objectiveEnabled"];
  onObjectiveEnabledChange?: ComponentProps<
    typeof EmptyState
  >["onObjectiveEnabledChange"];
  creationMode: ComponentProps<typeof EmptyState>["creationMode"];
  onCreationModeChange?: ComponentProps<
    typeof EmptyState
  >["onCreationModeChange"];
  activeTheme: ComponentProps<typeof EmptyState>["activeTheme"];
  onThemeChange?: NonNullable<
    ComponentProps<typeof EmptyState>["onThemeChange"]
  >;
  themeLocked: boolean;
  hasCanvasContent: boolean;
  hasContentId: boolean;
  selectedText: ComponentProps<typeof EmptyState>["selectedText"];
  onRecommendationClick?: ComponentProps<
    typeof EmptyState
  >["onRecommendationClick"];
  characters: NonNullable<ComponentProps<typeof EmptyState>["characters"]>;
  skills: NonNullable<ComponentProps<typeof EmptyState>["skills"]>;
  serviceSkills: NonNullable<
    ComponentProps<typeof EmptyState>["serviceSkills"]
  >;
  serviceSkillGroups: NonNullable<
    ComponentProps<typeof EmptyState>["serviceSkillGroups"]
  >;
  isSkillsLoading: boolean;
  onSelectServiceSkill?: ComponentProps<
    typeof EmptyState
  >["onSelectServiceSkill"];
  onNavigateToSettings?: ComponentProps<
    typeof EmptyState
  >["onNavigateToSettings"];
  onRefreshSkills?: ComponentProps<typeof EmptyState>["onRefreshSkills"];
  onLaunchBrowserAssist?: ComponentProps<
    typeof EmptyState
  >["onLaunchBrowserAssist"];
  browserAssistLoading: boolean;
  recentSessionTitle?: ComponentProps<typeof EmptyState>["recentSessionTitle"];
  recentSessionSummary?: ComponentProps<
    typeof EmptyState
  >["recentSessionSummary"];
  recentSessionActionLabel?: ComponentProps<
    typeof EmptyState
  >["recentSessionActionLabel"];
  homeRecoverySession?: ComponentProps<
    typeof EmptyState
  >["homeRecoverySession"];
  onResumeRecentSession?: ComponentProps<
    typeof EmptyState
  >["onResumeRecentSession"];
  projectConversationGroups?: ComponentProps<
    typeof EmptyState
  >["projectConversationGroups"];
  onOpenProjectConversation?: ComponentProps<
    typeof EmptyState
  >["onOpenProjectConversation"];
  projectId: string | null;
  openedProjects?: ComponentProps<typeof EmptyState>["openedProjects"];
  sessionId?: string | null;
  onProjectChange?: ComponentProps<typeof EmptyState>["onProjectContextChange"];
  runtimeToolAvailability?: ComponentProps<
    typeof EmptyState
  >["runtimeToolAvailability"];
  pluginSuggestions?: ComponentProps<typeof EmptyState>["pluginSuggestions"];
  pluginSuggestionsError?: ComponentProps<
    typeof EmptyState
  >["pluginSuggestionsError"];
  pluginSuggestionsLoading?: ComponentProps<
    typeof EmptyState
  >["pluginSuggestionsLoading"];
  onPluginSuggestionsNeeded?: ComponentProps<
    typeof EmptyState
  >["onPluginSuggestionsNeeded"];
  initialInputCapability?: ComponentProps<
    typeof EmptyState
  >["initialInputCapability"];
  knowledgePackSelection?: ComponentProps<
    typeof EmptyState
  >["knowledgePackSelection"];
  knowledgePackOptions?: ComponentProps<
    typeof EmptyState
  >["knowledgePackOptions"];
  onToggleKnowledgePack?: ComponentProps<
    typeof EmptyState
  >["onToggleKnowledgePack"];
  onSelectKnowledgePack?: ComponentProps<
    typeof EmptyState
  >["onSelectKnowledgePack"];
  onToggleKnowledgeCompanionPack?: ComponentProps<
    typeof EmptyState
  >["onToggleKnowledgeCompanionPack"];
  onStartKnowledgeOrganize?: ComponentProps<
    typeof EmptyState
  >["onStartKnowledgeOrganize"];
  onManageKnowledgePacks?: ComponentProps<
    typeof EmptyState
  >["onManageKnowledgePacks"];
  runtimeTaskCard?: ComponentProps<typeof EmptyState>["runtimeTaskCard"];
  onOpenMemoryWorkbench?: ComponentProps<
    typeof EmptyState
  >["onOpenMemoryWorkbench"];
  onOpenChannels?: ComponentProps<typeof EmptyState>["onOpenChannels"];
  creationReplaySurface?: CreationReplaySurfaceModel | null;
  defaultCuratedTaskReferenceMemoryIds?: ComponentProps<
    typeof EmptyState
  >["defaultCuratedTaskReferenceMemoryIds"];
  defaultCuratedTaskReferenceEntries?: ComponentProps<
    typeof EmptyState
  >["defaultCuratedTaskReferenceEntries"];
  pathReferences?: ComponentProps<typeof EmptyState>["pathReferences"];
  onAddPathReferences?: ComponentProps<
    typeof EmptyState
  >["onAddPathReferences"];
  inputRestoreRequest?: ComponentProps<
    typeof EmptyState
  >["inputRestoreRequest"];
  onInputRestoreRequestHandled?: ComponentProps<
    typeof EmptyState
  >["onInputRestoreRequestHandled"];
  onImportPathReferenceAsKnowledge?: ComponentProps<
    typeof EmptyState
  >["onImportPathReferenceAsKnowledge"];
  onRemovePathReference?: ComponentProps<
    typeof EmptyState
  >["onRemovePathReference"];
  onClearPathReferences?: ComponentProps<
    typeof EmptyState
  >["onClearPathReferences"];
  fileManagerOpen?: ComponentProps<typeof EmptyState>["fileManagerOpen"];
  onToggleFileManager?: ComponentProps<
    typeof EmptyState
  >["onToggleFileManager"];
}

export function buildWorkspaceEmptyStateProps({
  input,
  setInput,
  onSendMessage,
  onStopSending,
  sendOnPointerDown,
  isLoading,
  disabled,
  providerType,
  setProviderType,
  model,
  setModel,
  reasoningEffort,
  setReasoningEffort,
  accessMode,
  setAccessMode,
  onManageProviders,
  toolPreferences,
  onToolPreferenceChange,
  objectiveEnabled,
  onObjectiveEnabledChange,
  creationMode,
  onCreationModeChange,
  activeTheme,
  onThemeChange,
  themeLocked,
  hasCanvasContent,
  hasContentId,
  selectedText,
  onRecommendationClick,
  characters,
  skills,
  serviceSkills,
  serviceSkillGroups,
  isSkillsLoading,
  onSelectServiceSkill,
  onNavigateToSettings,
  onRefreshSkills,
  onLaunchBrowserAssist,
  browserAssistLoading,
  recentSessionTitle,
  recentSessionSummary,
  recentSessionActionLabel,
  homeRecoverySession,
  onResumeRecentSession,
  projectConversationGroups,
  onOpenProjectConversation,
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
  runtimeTaskCard,
  onOpenMemoryWorkbench,
  onOpenChannels,
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
}: BuildWorkspaceEmptyStatePropsParams): ComponentProps<typeof EmptyState> {
  const handleEmptyStateSend: InputbarSendHandler = (payload = {}) =>
    onSendMessage({
      ...payload,
      textOverride: payload.textOverride ?? input,
    });

  return {
    input,
    setInput,
    onSend: handleEmptyStateSend,
    onStop: onStopSending,
    sendOnPointerDown,
    isLoading,
    disabled,
    providerType,
    setProviderType,
    model,
    setModel,
    reasoningEffort,
    setReasoningEffort,
    accessMode,
    setAccessMode,
    onManageProviders,
    taskEnabled: toolPreferences.task,
    onTaskEnabledChange: (enabled) => onToolPreferenceChange("task", enabled),
    objectiveEnabled,
    onObjectiveEnabledChange,
    subagentEnabled: toolPreferences.subagent,
    onSubagentEnabledChange: (enabled) =>
      onToolPreferenceChange("subagent", enabled),
    creationMode,
    onCreationModeChange,
    activeTheme,
    onThemeChange: themeLocked
      ? undefined
      : (theme) => {
          onThemeChange?.(theme);
        },
    hasCanvasContent,
    hasContentId,
    selectedText,
    onRecommendationClick,
    characters,
    skills,
    serviceSkills,
    serviceSkillGroups,
    isSkillsLoading,
    onSelectServiceSkill,
    onNavigateToSettings,
    onRefreshSkills,
    onLaunchBrowserAssist,
    browserAssistLoading,
    recentSessionTitle,
    recentSessionSummary,
    recentSessionActionLabel,
    homeRecoverySession,
    onResumeRecentSession,
    projectConversationGroups,
    onOpenProjectConversation,
    projectId,
    openedProjects,
    onProjectContextChange: onProjectChange,
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
    runtimeTaskCard,
    onOpenMemoryWorkbench,
    onOpenChannels,
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
  };
}

interface BuildWorkspaceNavbarPropsParams {
  visible: boolean;
  isRunning: boolean;
  chrome: ComponentProps<typeof ChatNavbar>["chrome"];
  collapseChrome?: boolean;
  navbarContextVariant?: "default" | "task-center";
  collapseEntryContext?: boolean;
  onBackToProjectManagement?: ComponentProps<
    typeof ChatNavbar
  >["onBackToProjectManagement"];
  onBackToResources?: ComponentProps<typeof ChatNavbar>["onBackToResources"];
  showCanvasToggle: boolean;
  isCanvasOpen: boolean;
  onToggleCanvas?: ComponentProps<typeof ChatNavbar>["onToggleCanvas"];
  projectId: string | null;
  openedProjects?: ComponentProps<typeof ChatNavbar>["openedProjects"];
  onProjectChange?: ComponentProps<typeof ChatNavbar>["onProjectChange"];
  onCloseProject?: ComponentProps<typeof ChatNavbar>["onCloseProject"];
  workspaceType?: ComponentProps<typeof ChatNavbar>["workspaceType"];
  deferWorkspaceListLoad?: ComponentProps<
    typeof ChatNavbar
  >["deferWorkspaceListLoad"];
  onBackHome?: ComponentProps<typeof ChatNavbar>["onBackHome"];
  showHarnessToggle: boolean;
  harnessPanelVisible: boolean;
  onToggleHarnessPanel?: ComponentProps<
    typeof ChatNavbar
  >["onToggleHarnessPanel"];
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
  onOpenSettings?: () => void;
}

export function buildWorkspaceNavbarProps({
  visible,
  isRunning,
  chrome,
  collapseChrome = false,
  navbarContextVariant = "default",
  collapseEntryContext = false,
  onBackToProjectManagement,
  onBackToResources,
  showCanvasToggle,
  isCanvasOpen,
  onToggleCanvas,
  projectId,
  openedProjects,
  onProjectChange,
  onCloseProject,
  workspaceType,
  deferWorkspaceListLoad,
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
}: BuildWorkspaceNavbarPropsParams): ComponentProps<typeof ChatNavbar> | null {
  if (!visible) {
    return null;
  }

  return {
    isRunning,
    chrome,
    collapseChrome,
    contextVariant:
      navbarContextVariant === "task-center" && !collapseEntryContext
        ? "task-center"
        : "default",
    onToggleFullscreen: () => undefined,
    onBackToProjectManagement,
    onBackToResources,
    showCanvasToggle,
    isCanvasOpen,
    onToggleCanvas,
    projectId,
    openedProjects,
    onProjectChange,
    onCloseProject,
    workspaceType,
    deferWorkspaceListLoad,
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
    onToggleSettings: onOpenSettings,
  };
}
