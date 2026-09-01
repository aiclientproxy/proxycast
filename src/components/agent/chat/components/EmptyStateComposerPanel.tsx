import React, { useEffect, useRef, useState } from "react";
import styled from "styled-components";
import { FolderOpen, Lightbulb, X } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { CharacterMention } from "../skill-selection/CharacterMention";
import { BuiltinCommandBadge } from "./Inputbar/components/BuiltinCommandBadge";
import { InputbarAccessModeSelect } from "./Inputbar/components/InputbarAccessModeSelect";
import { InputbarCore } from "./Inputbar/components/InputbarCore";
import { InputbarModeStatusChip } from "./Inputbar/components/InputbarModeStatusChip";
import { InputbarPluginBadge } from "./Inputbar/components/InputbarPluginBadge";
import { InputbarPluginSelector } from "./Inputbar/components/InputbarPluginSelector";
import {
  InputbarProjectContextBar,
  type InputbarOpenedProject,
} from "./Inputbar/components/InputbarProjectContextBar";
import { InputbarKnowledgeControl } from "./Inputbar/knowledge/InputbarKnowledgeControl";
import { InputbarModelExtra } from "./Inputbar/components/InputbarModelExtra";
import { RuntimeSceneBadge } from "./Inputbar/components/RuntimeSceneBadge";
import { CuratedTaskBadge } from "../skill-selection/CuratedTaskBadge";
import { SkillBadge } from "../skill-selection/SkillBadge";
import { SkillSelector } from "../skill-selection/SkillSelector";
import { CREATION_MODE_CONFIG } from "./constants";
import type { CreationMode } from "./types";
import type { Character } from "@/lib/api/projectMemory";
import type { MessageImage, MessagePathReference } from "../types";
import {
  EMPTY_STATE_PASSIVE_BADGE_CLASSNAME,
  EMPTY_STATE_SELECT_TRIGGER_CLASSNAME,
} from "./emptyStateSurfaceTokens";
import { resolveEmptyStateActiveCapability } from "./EmptyStateComposerPanelViewModel";
import { MetaIconButton } from "./Inputbar/styles";
import {
  buildSkillSelectionBindings,
  type SkillSelectionProps,
} from "../skill-selection/skillSelectionBindings";
import type { HomeSurfaceComposerCopy } from "../home/homeSurfaceCopy";
import type { AgentAccessMode } from "../hooks/agentChatStorage";
import type {
  InputCapabilitySelection,
  SelectInputCapabilityHandler,
} from "../skill-selection/inputCapabilitySelection";
import type { CuratedTaskTemplateItem } from "../utils/curatedTaskTemplates";
import type { CuratedTaskReferenceEntry } from "../utils/curatedTaskReferenceSelection";
import type { CreationReplaySurfaceModel } from "../utils/creationReplaySurface";
import type {
  InputbarKnowledgePackOption,
  InputbarKnowledgePackSelection,
} from "./Inputbar/types";
import {
  applyInputbarPluginSelection,
  removeInputbarPluginSelection,
  resolveInputbarPluginDisplayName,
  resolveInputbarPluginSubmissionText,
  type InputbarPluginCapability,
  type InputbarPluginSelection,
  type InputbarPluginSelectionOptions,
  type InputbarPluginSkillCapability,
} from "./Inputbar/pluginInputCapability";
import type { InputbarCoreCopy } from "./Inputbar/components/inputbarCoreCopy";
import type { ModelReasoningEffortLevel } from "@/lib/types/modelRegistry";
import type { BaseComposerSendMetadata } from "@/components/input-kit";
import type { ComposerController } from "@/components/input-kit";
import {
  HomeComposerAccent,
  HomeComposerDecorationFrame,
} from "./HomeComposerDecorations";
import { useHomeSkinPresentation } from "./homeSkinPresentation";

const ConnectedComposerShell = styled.div`
  width: 100%;
  border: 1px solid var(--lime-composer-border, rgba(110, 231, 183, 0.84));
  border-radius: 22px;
  background: var(--lime-composer-surface);
  box-shadow:
    0 12px 30px var(--lime-shadow-color),
    inset 0 1px 0 rgba(255, 255, 255, 0.9);
  transition:
    border-color 160ms ease,
    box-shadow 160ms ease;

  > div:first-child {
    padding: 0;
  }

  [data-testid="inputbar-core-container"] {
    border: 0;
    border-radius: 21px 21px 0 0;
    box-shadow: none;
  }

  textarea {
    min-height: 88px;
    padding-top: 7px;
    font-size: 16px;
    line-height: 1.55;
  }

  [data-testid="inputbar-context-bar-slot"] {
    border: 0;
    border-top: 1px solid var(--lime-surface-border, rgba(226, 232, 240, 0.9));
    border-radius: 0 0 21px 21px;
  }

  &:focus-within {
    border-color: var(--lime-composer-border-focus, rgba(74, 222, 128, 0.7));
    box-shadow:
      0 0 0 3px var(--lime-focus-ring, rgba(74, 222, 128, 0.24)),
      0 14px 32px var(--lime-shadow-color),
      inset 0 1px 0 rgba(255, 255, 255, 0.92);
  }

  &:focus-within [data-testid="inputbar-core-container"] {
    border: 0;
    border-bottom-right-radius: 0;
    border-bottom-left-radius: 0;
    box-shadow: none;
  }

  &:focus-within [data-testid="inputbar-context-bar-slot"] {
    border-top-color: var(
      --lime-surface-border-strong,
      rgba(203, 213, 225, 0.96)
    );
    background: var(
      --lime-composer-context-surface-focus,
      var(--lime-composer-surface-focus)
    );
    box-shadow: none;
  }
`;

interface EmptyStateComposerPanelProps {
  input: string;
  setInput: (value: string) => void;
  controller?: ComposerController;
  placeholder: string;
  onSend: (
    inputOverride?: string,
    modeState?: {
      goalEnabled?: boolean;
      planEnabled?: boolean;
      subagentEnabled?: boolean;
    },
    triggerMetadata?: BaseComposerSendMetadata,
  ) => void | boolean | Promise<boolean>;
  onStop?: () => void;
  sendOnPointerDown?: boolean;
  isLoading?: boolean;
  disabled?: boolean;
  activeTheme: string;
  providerType: string;
  setProviderType: (type: string) => void;
  model: string;
  setModel: (model: string) => void;
  reasoningEffort?: ModelReasoningEffortLevel | "";
  setReasoningEffort?: (value: ModelReasoningEffortLevel | "") => void;
  accessMode?: AgentAccessMode;
  setAccessMode?: (mode: AgentAccessMode) => void;
  onManageProviders?: () => void;
  isGeneralTheme: boolean;
  characters: Character[];
  skillSelection: SkillSelectionProps;
  activeCapability?: InputCapabilitySelection | null;
  onSelectInputCapability?: SelectInputCapabilityHandler;
  onClearInputCapability?: () => void;
  onEditCuratedTask?: () => void;
  onApplyCuratedTaskReviewSuggestion?: (task: CuratedTaskTemplateItem) => void;
  creationReplaySurface?: CreationReplaySurfaceModel | null;
  projectId?: string | null;
  openedProjects?: InputbarOpenedProject[];
  onProjectContextChange?: (projectId: string | null) => void;
  projectContextModeLabel?: string;
  projectContextBranchLabel?: string;
  sessionId?: string | null;
  defaultCuratedTaskReferenceMemoryIds?: string[];
  defaultCuratedTaskReferenceEntries?: CuratedTaskReferenceEntry[];
  knowledgePackSelection?: InputbarKnowledgePackSelection | null;
  knowledgePackOptions?: InputbarKnowledgePackOption[];
  knowledgeHubOpenRequestKey?: number;
  onToggleKnowledgePack?: (enabled: boolean) => void;
  onSelectKnowledgePack?: (packName: string) => void;
  onToggleKnowledgeCompanionPack?: (packName: string, enabled: boolean) => void;
  onStartKnowledgeOrganize?: () => void;
  onManageKnowledgePacks?: () => void;
  copy: HomeSurfaceComposerCopy;
  inputbarCopy: InputbarCoreCopy;
  pluginSuggestions?: readonly InputbarPluginCapability[];
  pluginSuggestionsError?: string | null;
  pluginSuggestionsLoading?: boolean;
  onPluginSuggestionsNeeded?: () => void;
  showCreationModeSelector: boolean;
  creationMode: CreationMode;
  onCreationModeChange?: (mode: CreationMode) => void;
  taskEnabled?: boolean;
  onTaskEnabledChange?: (enabled: boolean) => void;
  objectiveEnabled?: boolean;
  onObjectiveEnabledChange?: (enabled: boolean) => void;
  subagentEnabled: boolean;
  onSubagentEnabledChange?: (enabled: boolean) => void;
  pendingImages: MessageImage[];
  onFileSelect: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onPaste?: (event: React.ClipboardEvent<HTMLTextAreaElement>) => void;
  onDragOver?: (event: React.DragEvent) => void;
  onDrop?: (event: React.DragEvent) => void;
  onRemoveImage?: (index: number) => void;
  pathReferences?: MessagePathReference[];
  onImportPathReferenceAsKnowledge?: (reference: MessagePathReference) => void;
  onRemovePathReference?: (id: string) => void;
  fileManagerOpen?: boolean;
  onToggleFileManager?: () => void;
  guideHelpActive?: boolean;
  guideHelpLabel?: string;
  onClearGuideHelp?: () => void;
}

function GuideHelpBadge({
  label,
  closeLabel,
  closeTitle,
  onClear,
}: {
  label: string;
  closeLabel: string;
  closeTitle: string;
  onClear: () => void;
}) {
  return (
    <div
      data-testid="home-guide-help-active-badge"
      className="mx-1 mt-1 inline-flex w-fit max-w-full items-center gap-1.5 rounded-full border border-emerald-200/80 bg-emerald-50/90 px-2.5 py-1.5 text-xs font-semibold text-emerald-900 shadow-sm shadow-emerald-950/5"
      title={label}
    >
      <Lightbulb className="h-3 w-3" strokeWidth={1.9} />
      <span className="truncate">{label}</span>
      <button
        type="button"
        onClick={onClear}
        className="ml-0.5 text-emerald-800/70 hover:opacity-70"
        aria-label={closeLabel}
        title={closeTitle}
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}

function GuideHelpToolbarBadge({
  label,
  title,
  onClear,
}: {
  label: string;
  title: string;
  onClear: () => void;
}) {
  return (
    <button
      type="button"
      data-testid="home-guide-help-toolbar-badge"
      className="inline-flex min-h-8 max-w-full items-center gap-1.5 rounded-full border border-emerald-200/80 bg-emerald-50/90 px-3 text-xs font-semibold text-emerald-900 shadow-sm shadow-emerald-950/5 transition hover:bg-white hover:text-emerald-950"
      title={title}
      onClick={onClear}
    >
      <Lightbulb className="h-3.5 w-3.5" strokeWidth={1.9} />
      <span className="truncate">{label.replace(/^Lime\s+/, "")}</span>
      <X className="h-3.5 w-3.5 opacity-70" aria-hidden />
    </button>
  );
}

export function EmptyStateComposerPanel({
  input,
  setInput,
  controller,
  placeholder,
  onSend,
  onStop,
  isLoading = false,
  disabled = false,
  activeTheme,
  providerType,
  setProviderType,
  model,
  setModel,
  reasoningEffort,
  setReasoningEffort,
  accessMode,
  setAccessMode,
  onManageProviders,
  isGeneralTheme,
  characters,
  skillSelection,
  activeCapability = null,
  onSelectInputCapability,
  onClearInputCapability,
  onEditCuratedTask,
  onApplyCuratedTaskReviewSuggestion,
  creationReplaySurface = null,
  projectId = null,
  openedProjects = [],
  onProjectContextChange,
  projectContextModeLabel,
  projectContextBranchLabel,
  sessionId = null,
  defaultCuratedTaskReferenceMemoryIds = [],
  defaultCuratedTaskReferenceEntries = [],
  knowledgePackSelection = null,
  knowledgePackOptions = [],
  knowledgeHubOpenRequestKey,
  onToggleKnowledgePack,
  onSelectKnowledgePack,
  onToggleKnowledgeCompanionPack,
  onStartKnowledgeOrganize,
  onManageKnowledgePacks,
  copy,
  inputbarCopy,
  pluginSuggestions = [],
  pluginSuggestionsError = null,
  pluginSuggestionsLoading = false,
  onPluginSuggestionsNeeded,
  showCreationModeSelector,
  creationMode,
  onCreationModeChange,
  taskEnabled = false,
  onTaskEnabledChange,
  objectiveEnabled: objectiveEnabledProp,
  onObjectiveEnabledChange,
  subagentEnabled,
  onSubagentEnabledChange,
  pendingImages,
  onFileSelect,
  onPaste,
  onDragOver,
  onDrop,
  onRemoveImage,
  pathReferences = [],
  onImportPathReferenceAsKnowledge,
  onRemovePathReference,
  fileManagerOpen = false,
  onToggleFileManager,
  guideHelpActive = false,
  guideHelpLabel,
  onClearGuideHelp,
}: EmptyStateComposerPanelProps) {
  const { composerDecorations } = useHomeSkinPresentation();
  const [activePluginSelection, setActivePluginSelection] =
    useState<InputbarPluginSelection | null>(null);
  const pluginSelectionInputSyncedRef = useRef(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const [localObjectiveEnabled, setLocalObjectiveEnabled] = useState(false);
  const objectiveEnabled = objectiveEnabledProp ?? localObjectiveEnabled;
  const activeCapabilityState = resolveEmptyStateActiveCapability({
    activeCapability,
    fallbackActiveSkill: skillSelection.activeSkill ?? null,
  });
  const {
    activeBuiltinCommand,
    activeRuntimeScene,
    activeCuratedTask,
    activeCuratedTaskReferenceEntries,
    activeSkill,
  } = activeCapabilityState;
  const clearActiveSkill = skillSelection.onClearSkill;
  const { mentionProps: mentionSkillProps, selectorProps: skillSelectorProps } =
    buildSkillSelectionBindings(skillSelection);
  useEffect(() => {
    if (!activePluginSelection) {
      pluginSelectionInputSyncedRef.current = false;
      return;
    }

    if (activePluginSelection.preserveInput) {
      pluginSelectionInputSyncedRef.current = false;
      return;
    }

    const inputText = input.trimStart();
    const trigger = activePluginSelection.trigger.trim();
    if (inputText === trigger || inputText.startsWith(`${trigger} `)) {
      pluginSelectionInputSyncedRef.current = true;
      return;
    }

    if (!pluginSelectionInputSyncedRef.current) {
      return;
    }

    pluginSelectionInputSyncedRef.current = false;
    setActivePluginSelection(null);
  }, [activePluginSelection, input]);

  const handleSendDraft = (triggerMetadata?: BaseComposerSendMetadata) => {
    const composerInput = controller?.getDocument().text ?? input;
    const submittedInput = resolveInputbarPluginSubmissionText({
      input: composerInput,
      selection: activePluginSelection,
    });
    const result = onSend(
      submittedInput,
      {
        goalEnabled: objectiveEnabled,
        planEnabled: taskEnabled,
        subagentEnabled,
      },
      triggerMetadata,
    );
    if (result === false || controller) {
      return;
    }
    if (result && typeof result === "object" && "then" in result) {
      void result.then((accepted) => {
        if (accepted !== false) {
          setInput("");
        }
      });
      return;
    }
    setInput("");
  };

  const handleToggleSubagentMode = () => {
    onSubagentEnabledChange?.(!subagentEnabled);
  };

  const handleToggleTaskMode = () => {
    onTaskEnabledChange?.(!taskEnabled);
  };

  const handleToggleObjectiveMode = () => {
    const nextObjectiveEnabled = !objectiveEnabled;
    if (onObjectiveEnabledChange) {
      onObjectiveEnabledChange(nextObjectiveEnabled);
      return;
    }
    setLocalObjectiveEnabled(nextObjectiveEnabled);
  };

  const handleSelectPlugin = (
    plugin: InputbarPluginCapability,
    skill?: InputbarPluginSkillCapability,
    options?: InputbarPluginSelectionOptions,
  ) => {
    const blocked =
      plugin.disabled ||
      (plugin.blockerCodes?.length ?? 0) > 0 ||
      skill?.disabled ||
      (skill?.blockerCodes?.length ?? 0) > 0;
    if (blocked) {
      return;
    }
    const selection = applyInputbarPluginSelection({
      input: options?.inputOverride ?? input,
      plugin,
      skill,
      preserveInput: options?.preserveInputOverride === true,
    });
    pluginSelectionInputSyncedRef.current = false;
    setInput(selection.text);
    setActivePluginSelection(selection);
    window.requestAnimationFrame(() => {
      textareaRef.current?.focus();
      textareaRef.current?.setSelectionRange(
        selection.text.length,
        selection.text.length,
      );
    });
  };

  const handleClearPluginSelection = () => {
    if (!activePluginSelection) {
      return;
    }
    const nextInput = removeInputbarPluginSelection({
      input,
      selection: activePluginSelection,
    });
    setInput(nextInput);
    setActivePluginSelection(null);
  };

  const handleToolAction = (tool: string) => {
    switch (tool) {
      case "attach":
        imageInputRef.current?.click();
        return;
      case "task_mode":
        handleToggleTaskMode();
        return;
      case "objective_mode":
        handleToggleObjectiveMode();
        return;
      case "subagent_mode":
        handleToggleSubagentMode();
        return;
      default:
        return;
    }
  };

  const effectiveGuideHelpLabel = guideHelpLabel ?? copy.guideHelpDefaultLabel;
  const topExtra =
    guideHelpActive ||
    activePluginSelection ||
    activeBuiltinCommand ||
    activeRuntimeScene ||
    activeCuratedTask ||
    activeSkill ||
    creationReplaySurface ? (
      <>
        {guideHelpActive ? (
          <GuideHelpBadge
            label={effectiveGuideHelpLabel}
            closeLabel={copy.guideHelpCloseWithLabel(effectiveGuideHelpLabel)}
            closeTitle={copy.guideHelpClose}
            onClear={onClearGuideHelp ?? (() => undefined)}
          />
        ) : null}

        {activePluginSelection ? (
          <InputbarPluginBadge
            selection={activePluginSelection}
            removeLabel={copy.pluginChip.remove(
              resolveInputbarPluginDisplayName(activePluginSelection.plugin),
            )}
            onClear={handleClearPluginSelection}
          />
        ) : null}

        {activeBuiltinCommand ? (
          <BuiltinCommandBadge
            command={activeBuiltinCommand}
            onClear={onClearInputCapability ?? (() => undefined)}
          />
        ) : null}

        {activeRuntimeScene ? (
          <RuntimeSceneBadge
            command={activeRuntimeScene}
            onClear={onClearInputCapability ?? (() => undefined)}
          />
        ) : null}

        {activeSkill ? (
          <SkillBadge
            skill={activeSkill}
            onClear={
              onClearInputCapability || clearActiveSkill || (() => undefined)
            }
          />
        ) : null}

        {activeCuratedTask ? (
          <CuratedTaskBadge
            task={activeCuratedTask}
            projectId={projectId}
            sessionId={sessionId}
            referenceEntries={activeCuratedTaskReferenceEntries}
            onEdit={onEditCuratedTask}
            onApplyReviewSuggestion={onApplyCuratedTaskReviewSuggestion}
            onClear={onClearInputCapability ?? (() => undefined)}
          />
        ) : null}

        {creationReplaySurface ? (
          <Badge
            className={`${EMPTY_STATE_PASSIVE_BADGE_CLASSNAME} max-w-[320px] justify-start gap-1.5`}
            title={`${creationReplaySurface.eyebrow} · ${creationReplaySurface.title} · ${creationReplaySurface.summary}`}
          >
            <span className="shrink-0 text-[color:var(--lime-brand-strong)]">
              {creationReplaySurface.badgeLabel}
            </span>
            <span className="truncate">{creationReplaySurface.title}</span>
          </Badge>
        ) : null}
      </>
    ) : undefined;

  const hasKnowledgePackControl = Boolean(
    knowledgePackSelection || onStartKnowledgeOrganize,
  );
  const plusMenuKnowledgePanel = hasKnowledgePackControl ? (
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
  const plusMenuSkillsPanel = isGeneralTheme ? (
    <SkillSelector {...skillSelectorProps} renderMode="inline" />
  ) : undefined;
  const plusMenuPluginsPanel = (
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
      onSelectPlugin={handleSelectPlugin}
    />
  );
  const plusMenuLabels = copy.plusMenu;
  const planModeStatusLabel = plusMenuLabels.planMode
    .replace(/模式$/, "")
    .replace(/ mode$/i, "");
  const objectiveStatusLabel = plusMenuLabels.objective;
  const fileManagerToggleLabel = fileManagerOpen
    ? copy.fileManager.close
    : copy.fileManager.open;
  const plusMenu = {
    labels: plusMenuLabels,
    taskEnabled,
    knowledgeOpenRequestKey: knowledgeHubOpenRequestKey,
    subagentEnabled,
    knowledgeActive: Boolean(knowledgePackSelection?.enabled),
    objectiveActive: objectiveEnabled,
    pluginsActive: Boolean(activePluginSelection),
    skillsActive: Boolean(skillSelection.activeSkill),
    knowledgePanel: plusMenuKnowledgePanel,
    pluginsPanel: plusMenuPluginsPanel,
    skillsPanel: plusMenuSkillsPanel,
    onPanelOpen: (panelId: "knowledge" | "plugins" | "skills") => {
      if (panelId === "plugins") {
        onPluginSuggestionsNeeded?.();
      }
    },
    onAddFiles: () => handleToolAction("attach"),
    onToggleTask: () => handleToolAction("task_mode"),
    onToggleObjective: () => handleToolAction("objective_mode"),
    onToggleSubagent:
      isGeneralTheme && onSubagentEnabledChange
        ? () => handleToolAction("subagent_mode")
        : undefined,
  };
  const leftExtra = (
    <>
      <InputbarAccessModeSelect
        accessMode={accessMode}
        setAccessMode={setAccessMode}
      />
      <HomeComposerAccent icon={composerDecorations?.accentIcon} />

      {taskEnabled ? (
        <InputbarModeStatusChip
          label={planModeStatusLabel}
          testId="empty-state-task-mode-status"
          onRemove={() => handleToolAction("task_mode")}
        />
      ) : null}

      {objectiveEnabled ? (
        <InputbarModeStatusChip
          label={objectiveStatusLabel}
          testId="empty-state-objective-status"
          onRemove={() => handleToolAction("objective_mode")}
        />
      ) : null}

      {guideHelpActive ? (
        <GuideHelpToolbarBadge
          label={effectiveGuideHelpLabel}
          title={copy.guideHelpClose}
          onClear={onClearGuideHelp ?? (() => undefined)}
        />
      ) : null}

      {showCreationModeSelector ? (
        <Select
          value={creationMode}
          onValueChange={(value) =>
            onCreationModeChange?.(value as CreationMode)
          }
        >
          <SelectTrigger
            className={`${EMPTY_STATE_SELECT_TRIGGER_CLASSNAME} min-w-[120px]`}
          >
            <div className="flex items-center gap-2">
              {CREATION_MODE_CONFIG[creationMode].icon}
              <span>{CREATION_MODE_CONFIG[creationMode].name}</span>
            </div>
          </SelectTrigger>
          <SelectContent className="min-w-[200px] p-1" side="top">
            <div className="px-2 py-1.5 text-xs font-medium text-slate-500">
              {copy.creationMode.label}
            </div>
            {(
              Object.entries(CREATION_MODE_CONFIG) as [
                CreationMode,
                (typeof CREATION_MODE_CONFIG)[CreationMode],
              ][]
            ).map(([key, config]) => (
              <SelectItem key={key} value={key}>
                <div className="flex items-center gap-3">
                  <span className="flex-shrink-0">{config.icon}</span>
                  <span className="font-medium">{config.name}</span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : null}

      {onToggleFileManager ? (
        <MetaIconButton
          type="button"
          $active={fileManagerOpen}
          aria-label={fileManagerToggleLabel}
          title={fileManagerToggleLabel}
          data-testid="inputbar-file-manager-toggle"
          onClick={onToggleFileManager}
        >
          <FolderOpen className="h-4 w-4" aria-hidden />
        </MetaIconButton>
      ) : null}
    </>
  );
  const trailingMeta = (
    <InputbarModelExtra
      providerType={providerType}
      setProviderType={setProviderType}
      model={model}
      setModel={setModel}
      reasoningEffort={reasoningEffort}
      setReasoningEffort={setReasoningEffort}
      activeTheme={activeTheme}
      onManageProviders={onManageProviders}
    />
  );
  const projectContextBar = (
    <div
      data-testid="inputbar-context-bar-slot"
      className="flex min-h-10 w-full items-center bg-[image:var(--lime-composer-context-surface,var(--lime-composer-surface))] px-4 py-1.5"
    >
      <InputbarProjectContextBar
        projectId={projectId}
        openedProjects={openedProjects}
        onProjectChange={onProjectContextChange}
        modeLabel={projectContextModeLabel}
        branchLabel={projectContextBranchLabel}
        copy={inputbarCopy.projectContext}
      />
    </div>
  );
  return (
    <>
      <CharacterMention
        {...mentionSkillProps}
        characters={characters}
        inputRef={textareaRef}
        value={input}
        onChange={setInput}
        onSelectInputCapability={onSelectInputCapability}
        pluginSuggestions={pluginSuggestions}
        onPluginSuggestionsNeeded={onPluginSuggestionsNeeded}
        onSelectPlugin={handleSelectPlugin}
        projectId={projectId}
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
      />

      <input
        ref={imageInputRef}
        type="file"
        accept="image/*"
        multiple
        style={{ display: "none" }}
        onChange={onFileSelect}
      />

      <HomeComposerDecorationFrame>
        <ConnectedComposerShell data-testid="inputbar-connected-composer">
          <InputbarCore
            uiCopy={inputbarCopy}
            controller={controller}
            textareaRef={textareaRef}
            text={input}
            setText={setInput}
            onSend={handleSendDraft}
            onStop={onStop}
            isLoading={isLoading}
            disabled={disabled}
            onToolClick={handleToolAction}
            activeTools={{
              objective_mode: objectiveEnabled,
              task_mode: taskEnabled,
              subagent_mode: subagentEnabled,
            }}
            pendingImages={pendingImages}
            onRemoveImage={onRemoveImage}
            onPaste={
              onPaste
                ? (event) =>
                    onPaste(event as React.ClipboardEvent<HTMLTextAreaElement>)
                : undefined
            }
            onDragOver={onDragOver}
            onDrop={onDrop}
            placeholder={placeholder}
            activeTheme={activeTheme}
            showDragHandle={false}
            visualVariant="default"
            connectedContextBar
            deferSendOnEnter
            sendOnPointerDown
            topExtra={topExtra}
            leftExtra={leftExtra}
            trailingMeta={trailingMeta}
            pathReferences={pathReferences}
            onImportPathReferenceAsKnowledge={onImportPathReferenceAsKnowledge}
            onRemovePathReference={onRemovePathReference}
            showMetaTools={false}
            showTextareaExpandButton={false}
            plusMenu={plusMenu}
          />
          {projectContextBar}
        </ConnectedComposerShell>
      </HomeComposerDecorationFrame>
    </>
  );
}
