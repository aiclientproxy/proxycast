import {
  useCallback,
  useMemo,
  type Dispatch,
  type SetStateAction,
} from "react";
import { useTranslation } from "react-i18next";
import type {
  AgentInitialInputCapabilityParams,
  AgentInitialKnowledgePackSelectionParams,
} from "@/types/page";
import { InputbarApprovalPrompt } from "../components/Inputbar/components/InputbarApprovalPrompt";
import { useWorkspaceNavigationActions } from "./useWorkspaceNavigationActions";
import type { ConfirmResponse } from "../types";
import {
  DEFAULT_CHAT_TOOL_PREFERENCES,
  type ChatToolPreferences,
} from "../utils/chatToolPreferences";
import { deriveRuntimeToolAvailability } from "../utils/runtimeToolAvailability";
import type { WorkspaceGeneralWorkbenchHarnessPanelBaseProps } from "./useWorkspaceGeneralWorkbenchHarnessSurfaceRuntime";
import type { InputbarSendHandler } from "../components/Inputbar/inputbarSendPayload";
import type { WorkspaceHandleSend } from "./useWorkspaceSendActions";
import type { CuratedTaskReferenceEntry } from "../utils/curatedTaskReferenceSelection";
import { useWorkspaceKnowledgeRuntime } from "./knowledge/useWorkspaceKnowledgeRuntime";
import { selectPendingInputbarApprovalAction } from "./inputbarApprovalAction";
import {
  useWorkspaceInputbarScenePresentationRuntime,
  type GeneralWorkbenchDialogParams,
  type InputbarParams,
  type InputbarPresentationParams,
  type InputbarScenePresentationParams,
  type WorkspaceInputbarToolStates,
} from "./useWorkspaceInputbarScenePresentationRuntime";

type NavigationActions = ReturnType<typeof useWorkspaceNavigationActions>;

interface UseWorkspaceInputbarSceneRuntimeParams {
  contextVariant?: "default" | "task-center";
  setMentionedCharacters: InputbarScenePresentationParams["setMentionedCharacters"];
  taskFiles?: InputbarScenePresentationParams["taskFiles"];
  selectedFileId?: InputbarScenePresentationParams["selectedFileId"];
  isThemeWorkbench: InputbarScenePresentationParams["isThemeWorkbench"];
  sessionId: InputbarParams["sessionId"];
  threadId: InputbarParams["threadId"];
  canAcceptDirectInput?: boolean | null;
  threadGoal: InputbarParams["threadGoal"];
  threadGoalError: InputbarParams["threadGoalError"];
  threadGoalLoading: InputbarParams["threadGoalLoading"];
  generalWorkbenchHarnessPanelBaseProps: WorkspaceGeneralWorkbenchHarnessPanelBaseProps;
  currentSessionTitle: string | null | undefined;
  handleStopSending: InputbarParams["onStop"];
  input: InputbarParams["input"];
  setInput: InputbarParams["setInput"];
  currentGate: InputbarParams["workflowGate"];
  generalWorkbenchWorkflowSteps: InputbarParams["workflowSteps"];
  steps: InputbarParams["workflowSteps"];
  workflowRunState: InputbarParams["workflowRunState"];
  handleSend: WorkspaceHandleSend;
  isPreparingSend: boolean;
  isSending: boolean;
  isSessionRestoring?: boolean;
  providerType: InputbarParams["providerType"];
  setProviderType: InputbarParams["setProviderType"];
  model: InputbarParams["model"];
  setModel: InputbarParams["setModel"];
  reasoningEffort: InputbarParams["reasoningEffort"];
  setReasoningEffort: InputbarParams["setReasoningEffort"];
  sessionExecutionRuntime: InputbarParams["executionRuntime"];
  projectId: string | null | undefined;
  openedProjects: InputbarParams["openedProjects"];
  projectRootPath: string | null | undefined;
  accessMode: InputbarParams["accessMode"];
  setAccessMode: InputbarParams["setAccessMode"];
  activeTheme: InputbarParams["activeTheme"];
  navigationActions: Pick<
    NavigationActions,
    | "handleManageProviders"
    | "handleOpenExecutionPolicySettings"
    | "handleOpenRuntimeMemoryWorkbench"
    | "handleOpenKnowledgeManagement"
    | "handleProjectChange"
  >;
  characters: InputbarParams["characters"];
  skills: InputbarParams["skills"];
  serviceSkills: InputbarParams["serviceSkills"];
  serviceSkillGroups: InputbarParams["serviceSkillGroups"];
  skillsLoading: InputbarParams["isSkillsLoading"];
  onSelectServiceSkill: InputbarParams["onSelectServiceSkill"];
  onSkillSuggestionsNeeded?: InputbarParams["onSkillSuggestionsNeeded"];
  initialInputCapability?: AgentInitialInputCapabilityParams;
  initialKnowledgePackSelection?: AgentInitialKnowledgePackSelectionParams;
  pluginSuggestions?: InputbarParams["pluginSuggestions"];
  pluginSuggestionsError?: string | null;
  pluginSuggestionsLoading?: boolean;
  onPluginSuggestionsNeeded?: InputbarParams["onPluginSuggestionsNeeded"];
  setChatToolPreferences: Dispatch<SetStateAction<ChatToolPreferences>>;
  objectiveEnabled?: boolean;
  onObjectiveEnabledChange?: (enabled: boolean) => void;
  handleNavigateToSkillSettings: InputbarParams["onNavigateToSettings"];
  handleRefreshSkills: InputbarParams["onRefreshSkills"];
  soulArtifactVoiceGenerationBrief?: Record<string, unknown> | null;
  soulArtifactVoiceEnabledForTurn: boolean;
  onSoulArtifactVoiceEnabledForTurnChange: (enabled: boolean) => void;
  generalWorkbenchEntryPrompt: InputbarPresentationParams["generalWorkbenchEntryPrompt"];
  handleRestartGeneralWorkbenchEntryPrompt: InputbarPresentationParams["onRestartGeneralWorkbenchEntryPrompt"];
  handleContinueGeneralWorkbenchEntryPrompt: InputbarPresentationParams["onContinueGeneralWorkbenchEntryPrompt"];
  planDecisionAccessory?: InputbarPresentationParams["planDecisionAccessory"];
  generalWorkbenchEnabled: boolean;
  harnessPanelVisible: GeneralWorkbenchDialogParams["open"];
  setHarnessPanelVisible: GeneralWorkbenchDialogParams["onOpenChange"];
  harnessState: GeneralWorkbenchDialogParams["harnessState"];
  mappedTheme: GeneralWorkbenchDialogParams["activeTheme"];
  activeRuntimeStatusTitle: GeneralWorkbenchDialogParams["runtimeStatusTitle"];
  chatToolPreferences?: ChatToolPreferences;
  defaultCuratedTaskReferenceMemoryIds?: string[];
  defaultCuratedTaskReferenceEntries?: CuratedTaskReferenceEntry[];
  pathReferences: InputbarParams["pathReferences"];
  onAddPathReferences: InputbarParams["onAddPathReferences"];
  inputRestoreRequest?: InputbarParams["inputRestoreRequest"];
  onInputRestoreRequestHandled?: InputbarParams["onInputRestoreRequestHandled"];
  onRemovePathReference: InputbarParams["onRemovePathReference"];
  onClearPathReferences: InputbarParams["onClearPathReferences"];
  fileManagerOpen: InputbarParams["fileManagerOpen"];
  onToggleFileManager: InputbarParams["onToggleFileManager"];
  inputCompletionEnabled?: boolean;
}

export function useWorkspaceInputbarSceneRuntime({
  contextVariant = "default",
  setMentionedCharacters,
  isThemeWorkbench,
  sessionId,
  threadId,
  canAcceptDirectInput,
  threadGoal,
  threadGoalError,
  threadGoalLoading,
  generalWorkbenchHarnessPanelBaseProps,
  currentSessionTitle,
  handleStopSending,
  input,
  setInput,
  currentGate,
  generalWorkbenchWorkflowSteps,
  steps,
  workflowRunState,
  handleSend,
  isPreparingSend,
  isSending,
  isSessionRestoring = false,
  providerType,
  setProviderType,
  model,
  setModel,
  reasoningEffort,
  setReasoningEffort,
  sessionExecutionRuntime,
  projectId,
  openedProjects,
  projectRootPath,
  accessMode,
  setAccessMode,
  activeTheme,
  navigationActions,
  characters,
  skills,
  serviceSkills,
  serviceSkillGroups,
  skillsLoading,
  onSelectServiceSkill,
  onSkillSuggestionsNeeded,
  initialInputCapability,
  initialKnowledgePackSelection,
  pluginSuggestions,
  pluginSuggestionsError,
  pluginSuggestionsLoading,
  onPluginSuggestionsNeeded,
  setChatToolPreferences,
  objectiveEnabled = false,
  onObjectiveEnabledChange,
  handleNavigateToSkillSettings,
  handleRefreshSkills,
  soulArtifactVoiceGenerationBrief,
  soulArtifactVoiceEnabledForTurn,
  onSoulArtifactVoiceEnabledForTurnChange,
  generalWorkbenchEntryPrompt,
  handleRestartGeneralWorkbenchEntryPrompt,
  handleContinueGeneralWorkbenchEntryPrompt,
  planDecisionAccessory,
  generalWorkbenchEnabled,
  harnessPanelVisible,
  setHarnessPanelVisible,
  harnessState,
  mappedTheme,
  activeRuntimeStatusTitle,
  chatToolPreferences,
  defaultCuratedTaskReferenceMemoryIds = [],
  defaultCuratedTaskReferenceEntries = [],
  pathReferences,
  onAddPathReferences,
  inputRestoreRequest,
  onInputRestoreRequestHandled,
  onRemovePathReference,
  onClearPathReferences,
  fileManagerOpen,
  onToggleFileManager,
  inputCompletionEnabled = true,
}: UseWorkspaceInputbarSceneRuntimeParams) {
  const { t } = useTranslation("agent");
  const directInputBlocked = canAcceptDirectInput === false;
  const knowledgeRuntime = useWorkspaceKnowledgeRuntime({
    projectRootPath,
    currentSessionTitle,
    input,
    setInput,
    handleSend,
    onOpenKnowledgeManagement: navigationActions.handleOpenKnowledgeManagement,
    initialKnowledgePackSelection,
  });
  const resolvedChatToolPreferences =
    chatToolPreferences ?? DEFAULT_CHAT_TOOL_PREFERENCES;
  const inputbarApprovalAction = useMemo(
    () =>
      selectPendingInputbarApprovalAction(
        generalWorkbenchHarnessPanelBaseProps.pendingActions ?? [],
        generalWorkbenchHarnessPanelBaseProps.submittedActionsInFlight ?? [],
      ),
    [
      generalWorkbenchHarnessPanelBaseProps.pendingActions,
      generalWorkbenchHarnessPanelBaseProps.submittedActionsInFlight,
    ],
  );
  const runtimeToolAvailability = useMemo(
    () =>
      deriveRuntimeToolAvailability(
        generalWorkbenchHarnessPanelBaseProps.toolInventory,
      ),
    [generalWorkbenchHarnessPanelBaseProps.toolInventory],
  );
  const inputbarTargetSessionId = sessionId?.trim() || undefined;
  const handleInputbarSend = useCallback<InputbarSendHandler>(
    (payload = {}) => {
      if (directInputBlocked) {
        return false;
      }
      const payloadTargetSessionId =
        payload.sendOptions?.targetSessionId?.trim() || undefined;
      const sendOptions =
        inputbarTargetSessionId || payloadTargetSessionId
          ? {
              ...(payload.sendOptions || {}),
              targetSessionId:
                payloadTargetSessionId ?? inputbarTargetSessionId,
            }
          : payload.sendOptions;
      return handleSend(
        payload.images,
        undefined,
        undefined,
        payload.textOverride,
        "react",
        payload.autoContinuePayload,
        sendOptions,
      );
    },
    [directInputBlocked, handleSend, inputbarTargetSessionId],
  );
  const handleInputbarToolStatesChange = useCallback(
    (nextToolStates: WorkspaceInputbarToolStates) => {
      if (directInputBlocked) {
        return;
      }
      const hasPlanChange = typeof nextToolStates.plan === "boolean";
      const hasSubagentChange = typeof nextToolStates.subagent === "boolean";
      const hasObjectiveChange = typeof nextToolStates.objective === "boolean";
      if (!hasPlanChange && !hasSubagentChange && !hasObjectiveChange) {
        return;
      }

      if (hasPlanChange || hasSubagentChange) {
        setChatToolPreferences((previous) => ({
          ...previous,
          ...(hasPlanChange ? { task: nextToolStates.plan } : {}),
          ...(hasSubagentChange ? { subagent: nextToolStates.subagent } : {}),
        }));
      }
      if (hasObjectiveChange) {
        onObjectiveEnabledChange?.(nextToolStates.objective === true);
      }
    },
    [directInputBlocked, onObjectiveEnabledChange, setChatToolPreferences],
  );
  const resolvedTurns = useMemo(
    () => generalWorkbenchHarnessPanelBaseProps.turns ?? [],
    [generalWorkbenchHarnessPanelBaseProps.turns],
  );
  const latestTurnPrompt =
    resolvedTurns
      .find(
        (turn) =>
          turn.id === generalWorkbenchHarnessPanelBaseProps.currentTurnId,
      )
      ?.prompt_text?.trim() ||
    resolvedTurns[resolvedTurns.length - 1]?.prompt_text?.trim() ||
    "";
  const approvalAccessory = inputbarApprovalAction ? (
    <InputbarApprovalPrompt
      request={inputbarApprovalAction}
      onSubmit={
        generalWorkbenchHarnessPanelBaseProps.onRespondToAction as
          | ((response: ConfirmResponse) => void | Promise<void>)
          | undefined
      }
    />
  ) : null;

  const presentationRuntime = useWorkspaceInputbarScenePresentationRuntime({
    setMentionedCharacters,
    isThemeWorkbench,
    inputbarPresentation: {
      inputbar: {
        input,
        setInput,
        contextVariant,
        variant: isThemeWorkbench ? "workspace" : "default",
        projectId,
        openedProjects,
        onProjectContextChange: navigationActions.handleProjectChange,
        sessionId,
        threadId,
        threadGoal,
        threadGoalError,
        threadGoalLoading,
        workflowGate: isThemeWorkbench ? currentGate : null,
        workflowSteps: isThemeWorkbench ? generalWorkbenchWorkflowSteps : steps,
        workflowRunState,
        onSend: handleInputbarSend,
        onStop: handleStopSending,
        isLoading: isSending,
        knowledgePackSelection: knowledgeRuntime.knowledgePackSelection,
        knowledgePackOptions: knowledgeRuntime.knowledgePackOptions,
        onKnowledgePacksNeeded: knowledgeRuntime.onKnowledgePacksNeeded,
        onToggleKnowledgePack: knowledgeRuntime.onToggleKnowledgePack,
        onSelectKnowledgePack: knowledgeRuntime.onSelectKnowledgePack,
        onToggleKnowledgeCompanionPack:
          knowledgeRuntime.onToggleKnowledgeCompanionPack,
        onStartKnowledgeOrganize: knowledgeRuntime.onStartKnowledgeOrganize,
        onManageKnowledgePacks: knowledgeRuntime.onManageKnowledgePacks,
        providerType,
        setProviderType,
        model,
        setModel,
        reasoningEffort,
        setReasoningEffort,
        executionRuntime: sessionExecutionRuntime,
        accessMode,
        setAccessMode,
        activeTheme,
        onManageProviders: navigationActions.handleManageProviders,
        disabled:
          directInputBlocked ||
          isSessionRestoring ||
          isPreparingSend ||
          (contextVariant !== "task-center" && !projectId && !sessionId),
        disabledPlaceholder: directInputBlocked
          ? t("agentChat.inputbar.parentOwned.placeholder")
          : undefined,
        characters,
        skills,
        serviceSkills,
        serviceSkillGroups,
        isSkillsLoading: skillsLoading,
        onSelectServiceSkill,
        onSkillSuggestionsNeeded,
        initialInputCapability,
        pluginSuggestions,
        pluginSuggestionsError,
        pluginSuggestionsLoading,
        onPluginSuggestionsNeeded,
        toolStates: {
          objective: objectiveEnabled,
          plan: resolvedChatToolPreferences.task,
          subagent: resolvedChatToolPreferences.subagent,
        },
        onToolStatesChange: handleInputbarToolStatesChange,
        onNavigateToSettings: handleNavigateToSkillSettings,
        onRefreshSkills: handleRefreshSkills,
        defaultCuratedTaskReferenceMemoryIds,
        defaultCuratedTaskReferenceEntries,
        pathReferences,
        onAddPathReferences,
        inputRestoreRequest,
        onInputRestoreRequestHandled,
        onImportPathReferenceAsKnowledge:
          knowledgeRuntime.onImportPathReferenceAsKnowledge,
        onRemovePathReference,
        onClearPathReferences,
        fileManagerOpen,
        onToggleFileManager,
        inputCompletionEnabled,
      },
      generalWorkbenchEntryPrompt,
      onRestartGeneralWorkbenchEntryPrompt:
        handleRestartGeneralWorkbenchEntryPrompt,
      onContinueGeneralWorkbenchEntryPrompt:
        handleContinueGeneralWorkbenchEntryPrompt,
      approvalAccessory,
      planDecisionAccessory,
      soulArtifactVoiceGenerationBrief,
      soulArtifactVoiceEnabledForTurn,
      onSoulArtifactVoiceEnabledForTurnChange,
      generalWorkbenchDialog: {
        enabled: generalWorkbenchEnabled && !isThemeWorkbench,
        open: harnessPanelVisible,
        onOpenChange: setHarnessPanelVisible,
        harnessState,
        ...generalWorkbenchHarnessPanelBaseProps,
        onInterruptCurrentTurn: handleStopSending,
        onManageProviders:
          generalWorkbenchHarnessPanelBaseProps.onManageProviders ??
          navigationActions.handleManageProviders,
        onOpenExecutionPolicySettings:
          generalWorkbenchHarnessPanelBaseProps.onOpenExecutionPolicySettings ??
          navigationActions.handleOpenExecutionPolicySettings,
        onOpenMemoryWorkbench:
          generalWorkbenchHarnessPanelBaseProps.diagnosticRuntimeContext
            ?.sessionId && projectRootPath
            ? () =>
                navigationActions.handleOpenRuntimeMemoryWorkbench({
                  sessionId:
                    generalWorkbenchHarnessPanelBaseProps
                      .diagnosticRuntimeContext?.sessionId || "",
                  workingDir: projectRootPath,
                  userMessage: latestTurnPrompt,
                })
            : undefined,
        activeTheme: mappedTheme,
        toolPreferences: resolvedChatToolPreferences,
        runtimeToolAvailability,
        isSending,
        executionRuntime: sessionExecutionRuntime,
        runtimeStatusTitle: activeRuntimeStatusTitle,
      },
    },
  });

  return {
    ...presentationRuntime,
    knowledgePackSelection: knowledgeRuntime.knowledgePackSelection,
    knowledgePackOptions: knowledgeRuntime.knowledgePackOptions,
    onToggleKnowledgePack: knowledgeRuntime.onToggleKnowledgePack,
    onSelectKnowledgePack: knowledgeRuntime.onSelectKnowledgePack,
    onToggleKnowledgeCompanionPack:
      knowledgeRuntime.onToggleKnowledgeCompanionPack,
    onKnowledgePacksNeeded: knowledgeRuntime.onKnowledgePacksNeeded,
    onStartKnowledgeOrganize: knowledgeRuntime.onStartKnowledgeOrganize,
    onManageKnowledgePacks: knowledgeRuntime.onManageKnowledgePacks,
    onImportPathReferenceAsKnowledge:
      knowledgeRuntime.onImportPathReferenceAsKnowledge,
    onImportTextAsKnowledge: knowledgeRuntime.onImportTextAsKnowledge,
  };
}
