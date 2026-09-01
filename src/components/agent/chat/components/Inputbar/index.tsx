import React from "react";
import styled from "styled-components";
import { useTranslation } from "react-i18next";
import type { MessagePathReference } from "../../types";
import type { Character } from "@/lib/api/projectMemory";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import { InputbarComposerSection } from "./components/InputbarComposerSection";
import type { InputbarOpenedProject } from "./components/InputbarProjectContextBar";
import { HintRoutePopup } from "./components/HintRoutePopup";
import { InputbarSurface } from "./components/InputbarSurface";
import type { SkillSelectionSourceProps } from "../../skill-selection/skillSelectionBindings";
import type {
  WorkflowGateState,
  WorkflowStep,
} from "../../utils/workflowInputState";
import { type InputbarToolStates } from "./hooks/useInputbarToolState";
import { useInputbarController } from "./hooks/useInputbarController";
import type { AgentAccessMode } from "../../hooks/agentChatStorage";
import type { AgentInitialInputCapabilityParams } from "@/types/page";
import type { CuratedTaskReferenceEntry } from "../../utils/curatedTaskReferenceSelection";
import type {
  InputbarKnowledgePackOption,
  InputbarKnowledgePackSelection,
} from "./types";
import type { InputbarPluginCapability } from "./pluginInputCapability";
import { buildInputbarComposerSectionCopy } from "./components/inputbarComposerSectionCopy";
import { buildInputbarCoreCopy } from "./components/inputbarCoreCopy";
import { buildInputbarWorkflowPanelCopy } from "./inputbarWorkflowCopy";
import type { InputbarSendHandler } from "./inputbarSendPayload";
import type { InterruptedInputRestoreRequest } from "../../hooks/agentStreamInputRestoreTypes";
import type { ModelReasoningEffortLevel } from "@/lib/types/modelRegistry";
import type { ThreadGoal } from "@limecloud/app-server-client";
import type { ComposerSubmitTarget } from "@/components/input-kit";

const SecondaryControlsRow = styled.div`
  position: absolute;
  right: 8px;
  bottom: calc(100% + 8px);
  left: 8px;
  display: flex;
  flex-wrap: wrap;
  justify-content: stretch;
  align-items: flex-end;
  gap: 8px;
  pointer-events: none;
  z-index: 80;

  > * {
    flex: 1 1 100%;
    pointer-events: auto;
    max-width: 100%;
  }
`;

interface InputbarProps extends SkillSelectionSourceProps {
  input: string;
  setInput: (value: string) => void;
  onSend: InputbarSendHandler;
  /** 停止生成回调 */
  onStop?: () => void;
  isLoading: boolean;
  disabled?: boolean;
  disabledPlaceholder?: string;
  /** 输入区上方并排浮层控件 */
  overlayAccessory?: React.ReactNode;
  /** 角色列表（用于 @ 引用） */
  characters?: Character[];
  /** 选择角色回调 */
  onSelectCharacter?: (character: Character) => void;
  providerType?: string;
  setProviderType?: (type: string) => void;
  model?: string;
  setModel?: (model: string) => void;
  reasoningEffort?: ModelReasoningEffortLevel | "";
  setReasoningEffort?: (value: ModelReasoningEffortLevel | "") => void;
  executionRuntime?: AgentSessionExecutionRuntime | null;
  accessMode?: AgentAccessMode;
  setAccessMode?: (mode: AgentAccessMode) => void;
  toolStates?: Partial<InputbarToolStates>;
  onToolStatesChange?: (states: Partial<InputbarToolStates>) => void;
  activeTheme?: string;
  onManageProviders?: () => void;
  initialInputCapability?: AgentInitialInputCapabilityParams;
  variant?: "default" | "workspace";
  workflowGate?: WorkflowGateState | null;
  workflowSteps?: WorkflowStep[];
  workflowRunState?: "idle" | "auto_running" | "await_user_decision";
  knowledgePackSelection?: InputbarKnowledgePackSelection | null;
  knowledgePackOptions?: InputbarKnowledgePackOption[];
  pluginSuggestions?: InputbarPluginCapability[];
  pluginSuggestionsError?: string | null;
  pluginSuggestionsLoading?: boolean;
  onSkillSuggestionsNeeded?: () => void;
  onPluginSuggestionsNeeded?: () => void;
  onKnowledgePacksNeeded?: () => void;
  onToggleKnowledgePack?: (enabled: boolean) => void;
  onSelectKnowledgePack?: (packName: string) => void;
  onToggleKnowledgeCompanionPack?: (packName: string, enabled: boolean) => void;
  onStartKnowledgeOrganize?: () => void;
  onManageKnowledgePacks?: () => void;
  contextVariant?: "default" | "task-center";
  projectId?: string | null;
  openedProjects?: InputbarOpenedProject[];
  onProjectContextChange?: (projectId: string | null) => void;
  projectContextModeLabel?: string;
  projectContextBranchLabel?: string;
  sessionId?: string | null;
  threadId?: string | null;
  threadGoal?: ThreadGoal | null;
  threadGoalError?: unknown;
  threadGoalLoading?: boolean;
  pathReferences?: MessagePathReference[];
  onAddPathReferences?: (references: MessagePathReference[]) => void;
  inputRestoreRequest?: InterruptedInputRestoreRequest | null;
  onInputRestoreRequestHandled?: (requestId: string) => void;
  onImportPathReferenceAsKnowledge?: (reference: MessagePathReference) => void;
  onRemovePathReference?: (id: string) => void;
  onClearPathReferences?: () => void;
  fileManagerOpen?: boolean;
  onToggleFileManager?: () => void;
  defaultCuratedTaskReferenceMemoryIds?: string[];
  defaultCuratedTaskReferenceEntries?: CuratedTaskReferenceEntry[];
  inputCompletionEnabled?: boolean;
  submitTarget?: ComposerSubmitTarget;
}

export const Inputbar: React.FC<InputbarProps> = ({
  input,
  setInput,
  onSend,
  onStop,
  isLoading,
  disabled,
  disabledPlaceholder,
  overlayAccessory,
  characters = [],
  skills,
  serviceSkills,
  serviceSkillGroups,
  isSkillsLoading,
  onSelectCharacter,
  onSelectServiceSkill,
  onNavigateToSettings,
  onImportSkill,
  onRefreshSkills,
  providerType,
  setProviderType,
  model,
  setModel,
  reasoningEffort,
  setReasoningEffort,
  executionRuntime,
  accessMode,
  setAccessMode,
  toolStates,
  onToolStatesChange,
  activeTheme,
  onManageProviders,
  initialInputCapability,
  variant = "default",
  workflowGate,
  workflowSteps = [],
  workflowRunState,
  knowledgePackSelection = null,
  knowledgePackOptions = [],
  pluginSuggestions = [],
  pluginSuggestionsError = null,
  pluginSuggestionsLoading = false,
  onSkillSuggestionsNeeded,
  onPluginSuggestionsNeeded,
  onKnowledgePacksNeeded,
  onToggleKnowledgePack,
  onSelectKnowledgePack,
  onToggleKnowledgeCompanionPack,
  onStartKnowledgeOrganize,
  onManageKnowledgePacks,
  contextVariant = "default",
  projectId = null,
  sessionId = null,
  threadId = null,
  threadGoal = null,
  threadGoalError = null,
  threadGoalLoading = false,
  pathReferences = [],
  onAddPathReferences,
  inputRestoreRequest = null,
  onInputRestoreRequestHandled,
  onImportPathReferenceAsKnowledge,
  onRemovePathReference,
  onClearPathReferences,
  fileManagerOpen = false,
  onToggleFileManager,
  defaultCuratedTaskReferenceMemoryIds = [],
  defaultCuratedTaskReferenceEntries = [],
  inputCompletionEnabled = true,
  submitTarget,
}) => {
  const { t } = useTranslation("agent");
  const inputbarCopy = React.useMemo(
    () => buildInputbarCoreCopy((key, values) => t(key, values ?? {})),
    [t],
  );
  const inputbarComposerCopy = React.useMemo(
    () =>
      buildInputbarComposerSectionCopy((key, values) => t(key, values ?? {})),
    [t],
  );
  const workflowPanelCopy = React.useMemo(
    () => buildInputbarWorkflowPanelCopy((key, values) => t(key, values ?? {})),
    [t],
  );
  const showModelControls = Boolean(
    providerType ||
    model ||
    setProviderType ||
    setModel ||
    onManageProviders ||
    executionRuntime,
  );
  const {
    textareaRef,
    composerController,
    isWorkspaceVariant,
    pendingImages,
    fileInputRef,
    handleFileSelect,
    handlePaste,
    handleDragOver,
    handleDrop,
    handleRemoveImage,
    showHintPopup,
    hintRoutes,
    hintIndex,
    handleHintSelect,
    handleHintKeyDown,
    activeTools,
    handleToolClick,
    isFullscreen,
    handleSend,
    inputAdapter,
    topExtra,
    dialogLayer,
    workflowQuickActions,
    workflowQueueItems,
    workflowActiveItem,
    workflowQueueTotalCount,
    workflowCompletedCount,
    workflowTotalCount,
    workflowProgressLabel,
    workflowSummaryLabel,
    renderWorkflowGeneratingPanel,
    skillSelection,
    handleSelectInputCapability,
    activeCapability,
    pluginSuggestions: resolvedPluginSuggestions,
    activePluginSelection,
    handleSelectPlugin,
    knowledgeHubOpenRequestKey,
  } = useInputbarController({
    input,
    setInput,
    onSend,
    onStop,
    isLoading,
    disabled,
    providerType,
    setProviderType,
    model,
    setModel,
    reasoningEffort,
    setReasoningEffort,
    toolStates,
    onToolStatesChange,
    initialInputCapability,
    variant,
    workflowGate,
    workflowSteps,
    workflowRunState,
    knowledgePackSelection,
    pluginSuggestions,
    onKnowledgePacksNeeded,
    onStartKnowledgeOrganize,
    onManageKnowledgePacks,
    projectId,
    sessionId,
    pathReferences,
    onAddPathReferences,
    inputRestoreRequest,
    onInputRestoreRequestHandled,
    onClearPathReferences,
    skills,
    serviceSkills,
    serviceSkillGroups,
    isSkillsLoading,
    onSelectServiceSkill,
    onNavigateToSettings,
    onImportSkill,
    onRefreshSkills,
    submitTarget,
  });

  return (
    <InputbarSurface
      isFullscreen={isFullscreen}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onKeyDown={handleHintKeyDown}
    >
      {showHintPopup ? (
        <HintRoutePopup
          routes={hintRoutes}
          activeIndex={hintIndex}
          onSelect={handleHintSelect}
        />
      ) : null}
      {overlayAccessory ? (
        <SecondaryControlsRow data-testid="inputbar-secondary-controls">
          {overlayAccessory}
        </SecondaryControlsRow>
      ) : null}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        style={{ display: "none" }}
        onChange={handleFileSelect}
      />
      <InputbarComposerSection
        renderWorkflowGeneratingPanel={renderWorkflowGeneratingPanel}
        workflowGate={workflowGate}
        workflowQuickActions={workflowQuickActions}
        workflowQueueItems={workflowQueueItems}
        workflowActiveItem={workflowActiveItem}
        workflowQueueTotalCount={workflowQueueTotalCount}
        workflowCompletedCount={workflowCompletedCount}
        workflowTotalCount={workflowTotalCount}
        workflowProgressLabel={workflowProgressLabel}
        workflowSummaryLabel={workflowSummaryLabel}
        inputAdapter={inputAdapter}
        composerController={composerController}
        characters={characters}
        skillSelection={skillSelection}
        onSkillSuggestionsNeeded={onSkillSuggestionsNeeded}
        textareaRef={textareaRef}
        input={input}
        disabledPlaceholder={disabledPlaceholder}
        onSelectCharacter={onSelectCharacter}
        onSelectInputCapability={handleSelectInputCapability}
        activeCapability={activeCapability}
        activePluginSelection={activePluginSelection}
        pluginSuggestions={resolvedPluginSuggestions}
        pluginSuggestionsError={pluginSuggestionsError}
        pluginSuggestionsLoading={pluginSuggestionsLoading}
        onPluginSuggestionsNeeded={onPluginSuggestionsNeeded}
        onSelectPlugin={handleSelectPlugin}
        projectId={projectId}
        sessionId={sessionId}
        threadId={threadId}
        threadGoal={threadGoal}
        threadGoalError={threadGoalError}
        threadGoalLoading={threadGoalLoading}
        defaultCuratedTaskReferenceMemoryIds={
          defaultCuratedTaskReferenceMemoryIds
        }
        defaultCuratedTaskReferenceEntries={defaultCuratedTaskReferenceEntries}
        knowledgePackSelection={knowledgePackSelection}
        knowledgePackOptions={knowledgePackOptions}
        knowledgeHubOpenRequestKey={knowledgeHubOpenRequestKey}
        onKnowledgePacksNeeded={onKnowledgePacksNeeded}
        onToggleKnowledgePack={onToggleKnowledgePack}
        onSelectKnowledgePack={onSelectKnowledgePack}
        onToggleKnowledgeCompanionPack={onToggleKnowledgeCompanionPack}
        onStartKnowledgeOrganize={onStartKnowledgeOrganize}
        onManageKnowledgePacks={onManageKnowledgePacks}
        onSend={handleSend}
        onToolClick={handleToolClick}
        activeTools={activeTools}
        pendingImages={pendingImages}
        onRemoveImage={handleRemoveImage}
        pathReferences={pathReferences}
        onImportPathReferenceAsKnowledge={onImportPathReferenceAsKnowledge}
        onRemovePathReference={onRemovePathReference}
        fileManagerOpen={fileManagerOpen}
        onToggleFileManager={onToggleFileManager}
        onPaste={handlePaste}
        isFullscreen={isFullscreen}
        isWorkspaceVariant={isWorkspaceVariant}
        activeTheme={activeTheme}
        onManageProviders={onManageProviders}
        reasoningEffort={reasoningEffort}
        setReasoningEffort={setReasoningEffort}
        executionRuntime={executionRuntime}
        accessMode={accessMode}
        setAccessMode={setAccessMode}
        showModelControls={showModelControls}
        topExtra={topExtra}
        contextVariant={contextVariant}
        inputCompletionEnabled={inputCompletionEnabled}
        copy={inputbarComposerCopy}
        inputbarCopy={inputbarCopy}
        workflowPanelCopy={workflowPanelCopy}
      />
      {dialogLayer}
    </InputbarSurface>
  );
};
