import React from "react";
import { FolderOpen } from "lucide-react";
import type { ChatInputAdapter } from "@/components/input-kit/adapters/types";
import type { Character } from "@/lib/api/projectMemory";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type { MessageImage, MessagePathReference } from "../../../types";
import { CharacterMention } from "../../../skill-selection/CharacterMention";
import { InputbarCore } from "./InputbarCore";
import { SkillSelector } from "../../../skill-selection/SkillSelector";
import { InputbarWorkflowStatusPanel } from "./InputbarWorkflowStatusPanel";
import { InputbarModelExtra } from "./InputbarModelExtra";
import { InputbarVisionCapabilityNotice } from "./InputbarVisionCapabilityNotice";
import { InputbarAccessModeSelect } from "./InputbarAccessModeSelect";
import { InputbarModeStatusChip } from "./InputbarModeStatusChip";
import { ThreadGoalPanel } from "../../ThreadGoalPanel";
import { InputbarPluginSelector } from "./InputbarPluginSelector";
import { isGeneralResearchTheme } from "../../../utils/generalAgentPrompt";
import {
  buildSkillSelectionBindings,
  type SkillSelectionProps,
} from "../../../skill-selection/skillSelectionBindings";
import type {
  InputCapabilitySelection,
  SelectInputCapabilityHandler,
} from "../../../skill-selection/inputCapabilitySelection";
import type { AgentAccessMode } from "../../../hooks/agentChatStorage";
import type { CuratedTaskReferenceEntry } from "../../../utils/curatedTaskReferenceSelection";
import type {
  InputbarKnowledgePackOption,
  InputbarKnowledgePackSelection,
} from "../types";
import { InputbarKnowledgeControl } from "../knowledge/InputbarKnowledgeControl";
import type { InputbarComposerSectionCopy } from "./inputbarComposerSectionCopy";
import type { InputbarCoreCopy } from "./inputbarCoreCopy";
import type { InputbarWorkflowPanelCopy } from "../inputbarWorkflowCopy";
import type {
  WorkflowGateState,
  WorkflowQuickAction,
  WorkflowStep,
} from "../../../utils/workflowInputState";
import {
  MetaIconButton,
  PlanModeContext,
  PlanModeContextSegment,
  PlanModeContextSeparator,
} from "../styles";
import type { ModelReasoningEffortLevel } from "@/lib/types/modelRegistry";
import type {
  InputbarPluginCapability,
  InputbarPluginSelection,
  InputbarPluginSelectionOptions,
  InputbarPluginSkillCapability,
} from "../pluginInputCapability";
import type { BaseComposerSendMetadata } from "@/components/input-kit";
import type { ThreadGoal } from "@limecloud/app-server-client";

interface InputbarComposerSectionProps {
  renderWorkflowGeneratingPanel: boolean;
  workflowGate?: WorkflowGateState | null;
  workflowQuickActions: WorkflowQuickAction[];
  workflowQueueItems: WorkflowStep[];
  workflowActiveItem: WorkflowStep | null;
  workflowQueueTotalCount: number;
  workflowCompletedCount: number;
  workflowTotalCount: number;
  workflowProgressLabel: string;
  workflowSummaryLabel: string;
  inputAdapter: ChatInputAdapter;
  characters: Character[];
  skillSelection: SkillSelectionProps;
  onSkillSuggestionsNeeded?: () => void;
  textareaRef: React.RefObject<HTMLTextAreaElement>;
  input: string;
  disabledPlaceholder?: string;
  onSelectCharacter?: (character: Character) => void;
  onSelectInputCapability: SelectInputCapabilityHandler;
  activeCapability?: InputCapabilitySelection | null;
  activePluginSelection?: InputbarPluginSelection | null;
  pluginSuggestions?: readonly InputbarPluginCapability[];
  pluginSuggestionsError?: string | null;
  pluginSuggestionsLoading?: boolean;
  onPluginSuggestionsNeeded?: () => void;
  onSelectPlugin?: (
    plugin: InputbarPluginCapability,
    skill?: InputbarPluginSkillCapability,
    options?: InputbarPluginSelectionOptions,
  ) => void;
  defaultCuratedTaskReferenceMemoryIds?: string[];
  defaultCuratedTaskReferenceEntries?: CuratedTaskReferenceEntry[];
  knowledgePackSelection?: InputbarKnowledgePackSelection | null;
  knowledgePackOptions?: InputbarKnowledgePackOption[];
  knowledgeHubOpenRequestKey?: number;
  onKnowledgePacksNeeded?: () => void;
  onToggleKnowledgePack?: (enabled: boolean) => void;
  onSelectKnowledgePack?: (packName: string) => void;
  onToggleKnowledgeCompanionPack?: (packName: string, enabled: boolean) => void;
  onStartKnowledgeOrganize?: () => void;
  onManageKnowledgePacks?: () => void;
  onSend: (metadata?: BaseComposerSendMetadata) => void;
  onToolClick: (tool: string) => void;
  activeTools: Record<string, boolean>;
  pendingImages: MessageImage[];
  onRemoveImage: (index: number) => void;
  pathReferences?: MessagePathReference[];
  onImportPathReferenceAsKnowledge?: (reference: MessagePathReference) => void;
  onRemovePathReference?: (id: string) => void;
  fileManagerOpen?: boolean;
  onToggleFileManager?: () => void;
  onPaste: (event: React.ClipboardEvent) => void;
  isFullscreen: boolean;
  isWorkspaceVariant: boolean;
  activeTheme?: string;
  onManageProviders?: () => void;
  reasoningEffort?: ModelReasoningEffortLevel | "";
  setReasoningEffort?: (value: ModelReasoningEffortLevel | "") => void;
  executionRuntime?: AgentSessionExecutionRuntime | null;
  accessMode?: AgentAccessMode;
  setAccessMode?: (mode: AgentAccessMode) => void;
  showModelControls?: boolean;
  topExtra?: React.ReactNode;
  contextVariant?: "default" | "task-center";
  projectId?: string | null;
  sessionId?: string | null;
  threadId?: string | null;
  threadGoal?: ThreadGoal | null;
  threadGoalError?: unknown;
  threadGoalLoading?: boolean;
  inputCompletionEnabled?: boolean;
  copy: InputbarComposerSectionCopy;
  inputbarCopy: InputbarCoreCopy;
  workflowPanelCopy: InputbarWorkflowPanelCopy;
}

export const InputbarComposerSection: React.FC<
  InputbarComposerSectionProps
> = ({
  renderWorkflowGeneratingPanel,
  workflowGate,
  workflowQuickActions,
  workflowQueueItems,
  workflowActiveItem,
  workflowQueueTotalCount,
  workflowCompletedCount,
  workflowTotalCount,
  workflowProgressLabel,
  workflowSummaryLabel,
  inputAdapter,
  characters,
  skillSelection,
  onSkillSuggestionsNeeded,
  textareaRef,
  input,
  disabledPlaceholder,
  onSelectCharacter,
  onSelectInputCapability,
  activeCapability,
  activePluginSelection = null,
  pluginSuggestions = [],
  pluginSuggestionsError = null,
  pluginSuggestionsLoading = false,
  onPluginSuggestionsNeeded,
  onSelectPlugin,
  defaultCuratedTaskReferenceMemoryIds = [],
  defaultCuratedTaskReferenceEntries = [],
  knowledgePackSelection,
  knowledgePackOptions = [],
  knowledgeHubOpenRequestKey,
  onKnowledgePacksNeeded,
  onToggleKnowledgePack,
  onSelectKnowledgePack,
  onToggleKnowledgeCompanionPack,
  onStartKnowledgeOrganize,
  onManageKnowledgePacks,
  onSend,
  onToolClick,
  activeTools,
  pendingImages,
  onRemoveImage,
  pathReferences = [],
  onImportPathReferenceAsKnowledge,
  onRemovePathReference,
  fileManagerOpen = false,
  onToggleFileManager,
  onPaste,
  isFullscreen,
  isWorkspaceVariant,
  activeTheme,
  onManageProviders,
  reasoningEffort,
  setReasoningEffort,
  executionRuntime,
  accessMode,
  setAccessMode,
  showModelControls = false,
  topExtra,
  contextVariant = "default",
  projectId = null,
  sessionId = null,
  threadId = null,
  threadGoal = null,
  threadGoalError = null,
  threadGoalLoading = false,
  inputCompletionEnabled = true,
  copy,
  inputbarCopy,
  workflowPanelCopy,
}) => {
  const showSkillSelector = isGeneralResearchTheme(activeTheme);
  const currentPendingImages =
    (inputAdapter.state.attachments as MessageImage[] | undefined) ||
    pendingImages;
  const { mentionProps: mentionSkillProps, selectorProps: skillSelectorProps } =
    buildSkillSelectionBindings(skillSelection);
  const resolvedProviderType = inputAdapter.model?.providerType;
  const resolvedModel = inputAdapter.model?.model;
  const resolvedReasoningEffort =
    inputAdapter.model?.reasoningEffort ?? reasoningEffort ?? "";
  const shouldShowModelControls = showModelControls;
  const resolvedSetProviderType =
    inputAdapter.actions.setProviderType || (() => undefined);
  const resolvedSetModel = inputAdapter.actions.setModel || (() => undefined);
  const resolvedSetReasoningEffort =
    inputAdapter.actions.setReasoningEffort || setReasoningEffort;
  const shouldShowVisionNotice =
    currentPendingImages.length > 0 &&
    Boolean(resolvedProviderType?.trim()) &&
    Boolean(resolvedModel?.trim());
  const inputbarDisabled = Boolean(inputAdapter.state.disabled);
  const activeKnowledgeStatusControl =
    knowledgePackSelection?.enabled &&
    knowledgePackSelection.packName.trim() &&
    knowledgePackSelection.workingDir.trim() ? (
      <InputbarKnowledgeControl
        knowledgePackSelection={knowledgePackSelection}
        knowledgePackOptions={knowledgePackOptions}
        inputText={input}
        openKnowledgeHubRequestKey={knowledgeHubOpenRequestKey}
        onToggleKnowledgePack={onToggleKnowledgePack}
        onSelectKnowledgePack={onSelectKnowledgePack}
        onToggleKnowledgeCompanionPack={onToggleKnowledgeCompanionPack}
        onStartKnowledgeOrganize={onStartKnowledgeOrganize}
        onManageKnowledgePacks={onManageKnowledgePacks}
      />
    ) : null;
  const objectiveInlinePanel =
    threadId && (threadGoal || threadGoalLoading) ? (
      <ThreadGoalPanel
        runtimeBusy={inputAdapter.state.isSending}
        threadGoal={threadGoal}
        threadGoalError={threadGoalError}
        threadGoalLoading={threadGoalLoading}
        threadId={threadId}
        variant="inline"
      />
    ) : null;
  const resolvedTopExtra =
    activeKnowledgeStatusControl ||
    objectiveInlinePanel ||
    topExtra ||
    shouldShowVisionNotice ? (
      <>
        {activeKnowledgeStatusControl}
        {objectiveInlinePanel}
        {topExtra}
        {shouldShowVisionNotice && resolvedProviderType && resolvedModel ? (
          <InputbarVisionCapabilityNotice
            providerType={resolvedProviderType}
            model={resolvedModel}
            hasPendingImages={currentPendingImages.length > 0}
          />
        ) : null}
      </>
    ) : undefined;
  const handleToolAction = (tool: string) => {
    onToolClick(tool);
  };
  const planModeStatusLabel = copy.planStatus.label;
  const objectiveStatusLabel = copy.plusMenu.objective;
  const planStatusModelLabel = resolvedModel?.trim()
    ? copy.planStatus.model(resolvedModel.trim())
    : copy.planStatus.modelFallback;
  const planStatusReasoningLevel = resolvedReasoningEffort
    ? (copy.planStatus.reasoningLevels[resolvedReasoningEffort] ??
      resolvedReasoningEffort)
    : copy.planStatus.reasoningDefault;
  const planStatusReasoningLabel = copy.planStatus.reasoning(
    planStatusReasoningLevel,
  );
  const planStatusTitle = [
    copy.plusMenu.planMode,
    planStatusModelLabel,
    planStatusReasoningLabel,
  ].join(" · ");
  const plusMenuKnowledgePanel =
    knowledgePackSelection || onStartKnowledgeOrganize ? (
      <InputbarKnowledgeControl
        renderMode="inline"
        knowledgePackSelection={knowledgePackSelection}
        knowledgePackOptions={knowledgePackOptions}
        inputText={input}
        openKnowledgeHubRequestKey={knowledgeHubOpenRequestKey}
        onToggleKnowledgePack={onToggleKnowledgePack}
        onSelectKnowledgePack={onSelectKnowledgePack}
        onToggleKnowledgeCompanionPack={onToggleKnowledgeCompanionPack}
        onStartKnowledgeOrganize={onStartKnowledgeOrganize}
        onManageKnowledgePacks={onManageKnowledgePacks}
      />
    ) : undefined;
  const plusMenuSkillsPanel = showSkillSelector ? (
    <SkillSelector {...skillSelectorProps} renderMode="inline" />
  ) : undefined;
  const plusMenuPluginsPanel = onSelectPlugin ? (
    <InputbarPluginSelector
      plugins={pluginSuggestions}
      labels={{
        empty: copy.pluginChip.empty,
        error: copy.pluginChip.error,
        loading: copy.pluginChip.loading,
        skillPrefix: copy.pluginChip.skillPrefix,
        title: copy.pluginChip.selectorTitle,
        unavailable: copy.pluginChip.unavailable,
      }}
      loading={pluginSuggestionsLoading}
      error={pluginSuggestionsError}
      onSelectPlugin={onSelectPlugin}
    />
  ) : undefined;
  const fileManagerLabel = fileManagerOpen
    ? copy.fileManager.close
    : copy.fileManager.open;
  const workspacePlaceholder =
    workflowGate?.status === "waiting"
      ? copy.workspacePlaceholder.waiting
      : contextVariant === "task-center"
        ? copy.workspacePlaceholder.taskCenter
        : copy.workspacePlaceholder.default;
  const plusMenu = {
    labels: copy.plusMenu,
    taskEnabled: Boolean(activeTools["task_mode"]),
    knowledgeOpenRequestKey: knowledgeHubOpenRequestKey,
    subagentEnabled: Boolean(activeTools["subagent_mode"]),
    knowledgeActive: Boolean(knowledgePackSelection?.enabled),
    objectiveActive: Boolean(activeTools["objective_mode"]),
    pluginsActive: Boolean(activePluginSelection),
    skillsActive: Boolean(skillSelection.activeSkill),
    knowledgePanel: plusMenuKnowledgePanel,
    pluginsPanel: plusMenuPluginsPanel,
    skillsPanel: plusMenuSkillsPanel,
    onPanelOpen: (panelId: "knowledge" | "plugins" | "skills") => {
      if (panelId === "knowledge") {
        onKnowledgePacksNeeded?.();
      } else if (panelId === "plugins") {
        onPluginSuggestionsNeeded?.();
      } else if (panelId === "skills") {
        onSkillSuggestionsNeeded?.();
      }
    },
    onAddFiles: () => handleToolAction("attach"),
    onToggleTask: () => handleToolAction("task_mode"),
    onToggleObjective: () => handleToolAction("objective_mode"),
    onToggleSubagent: showSkillSelector
      ? () => handleToolAction("subagent_mode")
      : undefined,
  };
  const shouldShowLeftExtra =
    Boolean(activeTools["task_mode"]) ||
    Boolean(activeTools["objective_mode"]) ||
    Boolean(setAccessMode) ||
    Boolean(onToggleFileManager);
  const leftExtra = shouldShowLeftExtra ? (
    <>
      <InputbarAccessModeSelect
        isFullscreen={isFullscreen}
        disabled={inputbarDisabled}
        accessMode={accessMode}
        setAccessMode={setAccessMode}
      />

      {activeTools["task_mode"] ? (
        <>
          <InputbarModeStatusChip
            label={planModeStatusLabel}
            testId="inputbar-task-mode-status"
            disabled={inputbarDisabled}
            onRemove={() => handleToolAction("task_mode")}
          />
          <PlanModeContext
            data-testid="inputbar-plan-mode-context"
            title={planStatusTitle}
          >
            <PlanModeContextSegment>
              {planStatusModelLabel}
            </PlanModeContextSegment>
            <PlanModeContextSeparator aria-hidden>·</PlanModeContextSeparator>
            <PlanModeContextSegment>
              {planStatusReasoningLabel}
            </PlanModeContextSegment>
          </PlanModeContext>
        </>
      ) : null}

      {activeTools["objective_mode"] ? (
        <InputbarModeStatusChip
          label={objectiveStatusLabel}
          testId="inputbar-objective-status"
          disabled={inputbarDisabled}
          onRemove={() => handleToolAction("objective_mode")}
        />
      ) : null}

      {onToggleFileManager ? (
        <MetaIconButton
          type="button"
          $active={fileManagerOpen}
          aria-label={fileManagerLabel}
          title={fileManagerLabel}
          data-testid="inputbar-file-manager-toggle"
          onClick={onToggleFileManager}
        >
          <FolderOpen className="h-4 w-4" aria-hidden />
        </MetaIconButton>
      ) : null}
    </>
  ) : undefined;
  const trailingMeta = shouldShowModelControls ? (
    <InputbarModelExtra
      isFullscreen={isFullscreen}
      disabled={inputbarDisabled}
      providerType={resolvedProviderType}
      setProviderType={resolvedSetProviderType}
      model={resolvedModel}
      setModel={resolvedSetModel}
      reasoningEffort={resolvedReasoningEffort}
      setReasoningEffort={resolvedSetReasoningEffort}
      activeTheme={activeTheme}
      onManageProviders={onManageProviders}
      executionRuntime={executionRuntime}
    />
  ) : undefined;
  if (renderWorkflowGeneratingPanel) {
    return (
      <InputbarWorkflowStatusPanel
        gate={workflowGate}
        quickActions={workflowQuickActions}
        queueItems={workflowQueueItems}
        activeItem={workflowActiveItem}
        queueTotalCount={workflowQueueTotalCount}
        completedCount={workflowCompletedCount}
        totalCount={workflowTotalCount}
        progressLabel={workflowProgressLabel}
        summaryLabel={workflowSummaryLabel}
        renderGeneratingPanel
        onQuickAction={inputAdapter.actions.setText}
        onStop={inputAdapter.actions.stop}
        copy={workflowPanelCopy}
      />
    );
  }

  return (
    <>
      <InputbarWorkflowStatusPanel
        gate={workflowGate}
        quickActions={workflowQuickActions}
        queueItems={workflowQueueItems}
        activeItem={workflowActiveItem}
        queueTotalCount={workflowQueueTotalCount}
        completedCount={workflowCompletedCount}
        totalCount={workflowTotalCount}
        progressLabel={workflowProgressLabel}
        summaryLabel={workflowSummaryLabel}
        renderGeneratingPanel={false}
        onQuickAction={inputAdapter.actions.setText}
        onStop={inputAdapter.actions.stop}
        copy={workflowPanelCopy}
      />
      <CharacterMention
        {...mentionSkillProps}
        characters={characters}
        inputRef={textareaRef}
        value={input}
        onChange={inputAdapter.actions.setText}
        onSelectCharacter={onSelectCharacter}
        onSelectInputCapability={onSelectInputCapability}
        onSkillSuggestionsNeeded={onSkillSuggestionsNeeded}
        pluginSuggestions={pluginSuggestions}
        onPluginSuggestionsNeeded={onPluginSuggestionsNeeded}
        onSelectPlugin={onSelectPlugin}
        projectId={projectId}
        sessionId={sessionId}
        defaultCuratedTaskReferenceMemoryIds={
          activeCapability?.kind === "curated_task"
            ? activeCapability.referenceMemoryIds ||
              defaultCuratedTaskReferenceMemoryIds
            : defaultCuratedTaskReferenceMemoryIds
        }
        defaultCuratedTaskReferenceEntries={
          activeCapability?.kind === "curated_task"
            ? activeCapability.referenceEntries ||
              defaultCuratedTaskReferenceEntries
            : defaultCuratedTaskReferenceEntries
        }
        inputCompletionEnabled={inputCompletionEnabled}
      />
      <InputbarCore
        uiCopy={inputbarCopy}
        textareaRef={textareaRef}
        text={inputAdapter.state.text}
        setText={inputAdapter.actions.setText}
        onSend={onSend}
        onStop={inputAdapter.actions.stop}
        isLoading={inputAdapter.state.isSending}
        disabled={inputbarDisabled}
        sessionId={sessionId}
        onToolClick={handleToolAction}
        activeTools={activeTools}
        pendingImages={currentPendingImages}
        onRemoveImage={onRemoveImage}
        pathReferences={pathReferences}
        onImportPathReferenceAsKnowledge={onImportPathReferenceAsKnowledge}
        onRemovePathReference={onRemovePathReference}
        onPaste={onPaste}
        isFullscreen={isFullscreen}
        placeholder={
          disabledPlaceholder ||
          (isWorkspaceVariant ? workspacePlaceholder : undefined)
        }
        toolMode={isWorkspaceVariant ? "attach-only" : "default"}
        showDragHandle={!isWorkspaceVariant}
        visualVariant={isWorkspaceVariant ? "floating" : "default"}
        connectedContextBar={false}
        topExtra={resolvedTopExtra}
        activeTheme={activeTheme}
        leftExtra={leftExtra}
        trailingMeta={trailingMeta}
        showMetaTools={false}
        plusMenu={plusMenu}
      />
    </>
  );
};
