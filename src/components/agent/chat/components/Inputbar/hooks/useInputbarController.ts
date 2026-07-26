import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import { BuiltinCommandBadge } from "../components/BuiltinCommandBadge";
import { InputbarPluginBadge } from "../components/InputbarPluginBadge";
import { RuntimeSceneBadge } from "../components/RuntimeSceneBadge";
import { CuratedTaskBadge } from "../../../skill-selection/CuratedTaskBadge";
import { SkillBadge } from "../../../skill-selection/SkillBadge";
import { CuratedTaskLauncherDialog } from "../../CuratedTaskLauncherDialog";
import { useHintRoutes } from "./useHintRoutes";
import { useImageAttachments } from "./useImageAttachments";
import { useInputbarAdapter } from "./useInputbarAdapter";
import { useInputbarSend } from "./useInputbarSend";
import {
  type InputbarToolStates,
  useInputbarToolState,
} from "./useInputbarToolState";
import {
  buildSkillSelectionProps,
  type SkillSelectionSourceProps,
} from "../../../skill-selection/skillSelectionBindings";
import type {
  WorkflowGateState,
  WorkflowStep,
} from "../../../utils/workflowInputState";
import { useWorkflowInputState } from "../../../utils/workflowInputState";
import type { ServiceSkillHomeItem } from "@/components/agent/chat/service-skills/types";
import type { MessagePathReference } from "../../../types";
import {
  resolveInputCapabilitySelectionFromRoute,
  type InputCapabilitySelection,
} from "../../../skill-selection/inputCapabilitySelection";
import type { InputbarKnowledgePackSelection } from "../types";
import type { InputbarSendHandler } from "../inputbarSendPayload";
import type { InterruptedInputRestoreRequest } from "../../../hooks/agentStreamInputRestoreTypes";
import type { AgentInitialInputCapabilityParams } from "@/types/page";
import {
  buildCuratedTaskLaunchPrompt,
  findCuratedTaskTemplateById,
  hasFilledAllCuratedTaskRequiredInputs,
  replaceCuratedTaskLaunchPromptInInput,
  type CuratedTaskInputValues,
} from "../../../utils/curatedTaskTemplates";
import {
  mergeCuratedTaskReferenceEntries,
  normalizeCuratedTaskReferenceMemoryIds,
  type CuratedTaskReferenceSelection,
} from "../../../utils/curatedTaskReferenceSelection";
import {
  readCustomPathReferencesFromDataTransfer,
  readSystemPathReferencesFromFiles,
} from "../../../utils/pathReferences";
import { toast } from "sonner";
import { buildInputbarControllerCopy } from "./inputbarControllerCopy";
import { buildInputbarWorkflowStateCopy } from "../inputbarWorkflowCopy";
import { logAgentDebug } from "@/lib/agentDebug";
import type { ModelReasoningEffortLevel } from "@/lib/types/modelRegistry";
import {
  applyInputbarPluginSelection,
  removeInputbarPluginSelection,
  resolveInputbarPluginDisplayName,
  type InputbarPluginCapability,
  type InputbarPluginSelection,
  type InputbarPluginSelectionOptions,
  type InputbarPluginSkillCapability,
} from "../pluginInputCapability";

interface UseInputbarControllerParams {
  input: string;
  setInput: (value: string) => void;
  onSend: InputbarSendHandler;
  onStop?: () => void;
  isLoading: boolean;
  disabled?: boolean;
  providerType?: string;
  setProviderType?: (type: string) => void;
  model?: string;
  setModel?: (model: string) => void;
  reasoningEffort?: ModelReasoningEffortLevel | "";
  setReasoningEffort?: (value: ModelReasoningEffortLevel | "") => void;
  toolStates?: Partial<InputbarToolStates>;
  onToolStatesChange?: (states: Partial<InputbarToolStates>) => void;
  initialInputCapability?: AgentInitialInputCapabilityParams;
  variant?: "default" | "workspace";
  workflowGate?: WorkflowGateState | null;
  workflowSteps?: WorkflowStep[];
  workflowRunState?: "idle" | "auto_running" | "await_user_decision";
  knowledgePackSelection?: InputbarKnowledgePackSelection | null;
  onKnowledgePacksNeeded?: () => void;
  onStartKnowledgeOrganize?: () => void;
  onManageKnowledgePacks?: () => void;
  pluginSuggestions?: InputbarPluginCapability[];
  projectId?: string | null;
  sessionId?: string | null;
  pathReferences?: MessagePathReference[];
  onAddPathReferences?: (references: MessagePathReference[]) => void;
  onClearPathReferences?: () => void;
  inputRestoreRequest?: InterruptedInputRestoreRequest | null;
  onInputRestoreRequestHandled?: (requestId: string) => void;
}

export function useInputbarController({
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
  variant = "default",
  workflowGate,
  workflowSteps = [],
  workflowRunState,
  knowledgePackSelection,
  onKnowledgePacksNeeded,
  onStartKnowledgeOrganize,
  onManageKnowledgePacks,
  pluginSuggestions = [],
  projectId = null,
  sessionId = null,
  pathReferences = [],
  onAddPathReferences,
  onClearPathReferences,
  inputRestoreRequest = null,
  onInputRestoreRequestHandled,
  skills,
  serviceSkills,
  serviceSkillGroups,
  isSkillsLoading,
  onSelectServiceSkill,
  onNavigateToSettings,
  onImportSkill,
  onRefreshSkills,
}: UseInputbarControllerParams & SkillSelectionSourceProps) {
  const { t } = useTranslation("agent");
  const copy = useMemo(() => buildInputbarControllerCopy((key) => t(key)), [t]);
  const workflowCopy = useMemo(
    () => buildInputbarWorkflowStateCopy((key, values) => t(key, values ?? {})),
    [t],
  );
  const [activeCapability, setActiveCapability] =
    useState<InputCapabilitySelection | null>(null);
  const [activePluginSelection, setActivePluginSelection] =
    useState<InputbarPluginSelection | null>(null);
  const [knowledgeHubOpenRequestKey, setKnowledgeHubOpenRequestKey] =
    useState(0);
  const [editingCuratedTaskCapability, setEditingCuratedTaskCapability] =
    useState<Extract<
      InputCapabilitySelection,
      { kind: "curated_task" }
    > | null>(null);
  const [curatedTaskEditorPrefillHint, setCuratedTaskEditorPrefillHint] =
    useState<string | null>(null);
  const handledInitialInputCapabilitySignatureRef = useRef("");
  const pluginSelectionInputSyncedRef = useRef(false);
  const inputRestoreEpochRef = useRef(0);
  const {
    pendingImages,
    fileInputRef,
    handleFileSelect,
    handlePaste,
    handleDragOver,
    handleDrop: handleImageDrop,
    handleRemoveImage,
    clearPendingImages,
    replacePendingImages,
    openFileDialog,
  } = useImageAttachments();
  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      const customReferences = readCustomPathReferencesFromDataTransfer(
        event.dataTransfer,
      );
      if (customReferences.length > 0) {
        event.preventDefault();
        event.stopPropagation();
        onAddPathReferences?.(customReferences);
        return;
      }

      const files = event.dataTransfer.files;
      const systemReferences =
        files && files.length > 0
          ? readSystemPathReferencesFromFiles(files)
          : [];
      if (systemReferences.length > 0) {
        event.preventDefault();
        event.stopPropagation();
        onAddPathReferences?.(systemReferences);
        return;
      }

      if (files && files.length > 0) {
        const hasImageFile = Array.from(files).some((file) =>
          file.type.startsWith("image/"),
        );
        if (!hasImageFile) {
          toast.error(copy.systemPathDropUnsupported);
          event.preventDefault();
          event.stopPropagation();
          return;
        }
      }

      handleImageDrop(event);
    },
    [copy.systemPathDropUnsupported, handleImageDrop, onAddPathReferences],
  );
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isWorkspaceVariant = variant === "workspace";
  const activeSkill =
    activeCapability?.kind === "installed_skill"
      ? activeCapability.skill
      : null;
  const activeBuiltinCommand =
    activeCapability?.kind === "builtin_command"
      ? activeCapability.command
      : null;
  const activeRuntimeScene =
    activeCapability?.kind === "runtime_scene"
      ? activeCapability.command
      : null;
  const activeCuratedTask =
    activeCapability?.kind === "curated_task" ? activeCapability.task : null;
  const activeCuratedTaskReferenceEntries =
    activeCapability?.kind === "curated_task"
      ? activeCapability.referenceEntries
      : undefined;

  useEffect(() => {
    if (activeBuiltinCommand?.key !== "knowledge_pack") {
      return;
    }
    setActiveCapability(null);
    if (!knowledgePackSelection && !onStartKnowledgeOrganize) {
      onManageKnowledgePacks?.();
      return;
    }
    onKnowledgePacksNeeded?.();
    setKnowledgeHubOpenRequestKey((current) => current + 1);
  }, [
    activeBuiltinCommand?.key,
    knowledgePackSelection,
    onKnowledgePacksNeeded,
    onManageKnowledgePacks,
    onStartKnowledgeOrganize,
  ]);
  const initialInputCapabilitySignature = useMemo(() => {
    const route = initialInputCapability?.capabilityRoute;
    if (!route) {
      return "";
    }

    return JSON.stringify({
      requestKey: initialInputCapability.requestKey ?? 0,
      route,
    });
  }, [initialInputCapability]);

  useEffect(() => {
    if (!inputRestoreRequest) {
      return;
    }

    inputRestoreEpochRef.current += 1;
    const restoreEpoch = inputRestoreEpochRef.current;
    const { draft, requestId } = inputRestoreRequest;
    const restoredPathReferences = [...(draft.pathReferences ?? [])];
    logAgentDebug("Inputbar", "inputRestoreRequest.apply", {
      draftImageCount: draft.images?.length ?? 0,
      draftPathReferenceCount: restoredPathReferences.length,
      draftTextLength: draft.text.trim().length,
      hasCapabilityRoute: Boolean(draft.inputCapabilityRoute),
      requestId,
      restoreEpoch,
    });
    setInput(draft.text);
    replacePendingImages([...(draft.images ?? [])]);
    onClearPathReferences?.();
    if (restoredPathReferences.length > 0) {
      onAddPathReferences?.(restoredPathReferences);
    }
    setActivePluginSelection(null);
    setActiveCapability(
      draft.inputCapabilityRoute
        ? resolveInputCapabilitySelectionFromRoute({
            route: draft.inputCapabilityRoute,
            skills,
          })
        : null,
    );
    onInputRestoreRequestHandled?.(requestId);
  }, [
    inputRestoreRequest,
    onAddPathReferences,
    onClearPathReferences,
    onInputRestoreRequestHandled,
    replacePendingImages,
    setInput,
    skills,
  ]);

  useEffect(() => {
    if (!initialInputCapabilitySignature) {
      handledInitialInputCapabilitySignatureRef.current = "";
      return;
    }

    if (
      handledInitialInputCapabilitySignatureRef.current ===
      initialInputCapabilitySignature
    ) {
      return;
    }

    const route = initialInputCapability?.capabilityRoute;
    if (!route) {
      return;
    }

    handledInitialInputCapabilitySignatureRef.current =
      initialInputCapabilitySignature;
    const resolvedCapability = resolveInputCapabilitySelectionFromRoute({
      route,
      skills,
    });
    const shouldOpenCuratedTaskLauncherOnBootstrap =
      resolvedCapability.kind === "curated_task" &&
      resolvedCapability.task.requiredInputFields.length > 0 &&
      !hasFilledAllCuratedTaskRequiredInputs({
        task: resolvedCapability.task,
        inputValues: resolvedCapability.launchInputValues ?? {},
      });

    if (
      route.kind === "curated_task" &&
      !shouldOpenCuratedTaskLauncherOnBootstrap &&
      !input.trim() &&
      route.prompt.trim().length > 0
    ) {
      setInput(route.prompt);
    }

    setActiveCapability(resolvedCapability);
    if (
      shouldOpenCuratedTaskLauncherOnBootstrap &&
      resolvedCapability.kind === "curated_task"
    ) {
      setCuratedTaskEditorPrefillHint(null);
      setEditingCuratedTaskCapability(resolvedCapability);
    }
  }, [
    initialInputCapability,
    initialInputCapabilitySignature,
    input,
    setInput,
    skills,
  ]);

  const { activeTools, handleToolClick, isFullscreen } = useInputbarToolState({
    toolStates,
    onToolStatesChange,
    openFileDialog,
  });

  const {
    showHintPopup,
    hintRoutes,
    hintIndex,
    handleSetInput,
    handleHintSelect,
    handleHintKeyDown,
  } = useHintRoutes({
    setInput,
    textareaRef,
  });

  const handleSend = useInputbarSend({
    input,
    pendingImages,
    pathReferences,
    activeCapability,
    activePluginSelection,
    knowledgePackSelection,
    activeTools,
    projectId,
    sessionId,
    onSend,
    clearPendingImages,
    clearPathReferences: onClearPathReferences,
    clearActiveCapability: () => setActiveCapability(null),
    getInputRestoreEpoch: () => inputRestoreEpochRef.current,
  });
  const inputAdapter = useInputbarAdapter({
    input,
    setInput: handleSetInput,
    isLoading,
    disabled,
    providerType,
    setProviderType,
    model,
    setModel,
    reasoningEffort,
    setReasoningEffort,
    handleSend,
    onStop,
    pendingImages,
  });

  const {
    workflowQuickActions,
    workflowQueueItems,
    workflowActiveItem,
    workflowQueueTotalCount,
    workflowCompletedCount,
    workflowTotalCount,
    workflowProgressLabel,
    workflowSummaryLabel,
    renderWorkflowGeneratingPanel,
  } = useWorkflowInputState({
    isWorkspaceVariant,
    workflowGate,
    workflowSteps,
    workflowRunState,
    isSending: inputAdapter.state.isSending,
    copy: workflowCopy,
  });

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

  const topExtra =
    activeSkill ||
    activeBuiltinCommand ||
    activeRuntimeScene ||
    activeCuratedTask ||
    activePluginSelection
      ? React.createElement(
          React.Fragment,
          null,
          activePluginSelection
            ? React.createElement(InputbarPluginBadge, {
                selection: activePluginSelection,
                removeLabel: t("agentChat.inputbar.pluginChip.remove", {
                  name: resolveInputbarPluginDisplayName(
                    activePluginSelection.plugin,
                  ),
                }),
                onClear: () => {
                  setInput(
                    removeInputbarPluginSelection({
                      input,
                      selection: activePluginSelection,
                    }),
                  );
                  setActivePluginSelection(null);
                },
              })
            : null,
          activeBuiltinCommand
            ? React.createElement(BuiltinCommandBadge, {
                command: activeBuiltinCommand,
                onClear: () => setActiveCapability(null),
              })
            : null,
          activeRuntimeScene
            ? React.createElement(RuntimeSceneBadge, {
                command: activeRuntimeScene,
                onClear: () => setActiveCapability(null),
              })
            : null,
          activeSkill
            ? React.createElement(SkillBadge, {
                skill: activeSkill,
                onClear: () => setActiveCapability(null),
              })
            : null,
          activeCuratedTask
            ? React.createElement(CuratedTaskBadge, {
                task: activeCuratedTask,
                projectId,
                sessionId,
                referenceEntries: activeCuratedTaskReferenceEntries,
                onEdit: () => {
                  if (activeCapability?.kind !== "curated_task") {
                    return;
                  }
                  setCuratedTaskEditorPrefillHint(null);
                  setEditingCuratedTaskCapability(activeCapability);
                },
                onApplyReviewSuggestion: (task) => {
                  if (activeCapability?.kind !== "curated_task") {
                    return;
                  }

                  const resolvedTask =
                    findCuratedTaskTemplateById(task.id) ?? task;
                  const nextReferenceEntries = mergeCuratedTaskReferenceEntries(
                    activeCapability.referenceEntries ?? [],
                  );
                  const nextLaunchInputValues =
                    activeCapability.launchInputValues ?? {};
                  const nextReferenceMemoryIds =
                    normalizeCuratedTaskReferenceMemoryIds(
                      activeCapability.referenceMemoryIds,
                    ) ?? [];
                  const nextPrompt = buildCuratedTaskLaunchPrompt({
                    task: resolvedTask,
                    inputValues: nextLaunchInputValues,
                    referenceEntries: nextReferenceEntries,
                  });

                  setInput(
                    replaceCuratedTaskLaunchPromptInInput({
                      currentInput: input,
                      previousPrompt: activeCapability.task.prompt,
                      nextPrompt,
                    }),
                  );
                  setActiveCapability({
                    kind: "curated_task",
                    task: {
                      ...resolvedTask,
                      prompt: nextPrompt,
                    },
                    launchInputValues: nextLaunchInputValues,
                    referenceMemoryIds: nextReferenceMemoryIds,
                    referenceEntries: nextReferenceEntries,
                  });
                },
                onClear: () => setActiveCapability(null),
              })
            : null,
        )
      : undefined;
  const handleCuratedTaskEditorOpenChange = useCallback((open: boolean) => {
    if (!open) {
      setCuratedTaskEditorPrefillHint(null);
      setEditingCuratedTaskCapability(null);
    }
  }, []);
  const handleApplyCuratedTaskEditorReviewSuggestion = useCallback(
    (
      task: NonNullable<typeof editingCuratedTaskCapability>["task"],
      options: {
        inputValues: CuratedTaskInputValues;
        referenceSelection: CuratedTaskReferenceSelection;
      },
    ) => {
      const resolvedTask = findCuratedTaskTemplateById(task.id) ?? task;
      setEditingCuratedTaskCapability((current) => {
        if (!current) {
          return current;
        }

        return {
          ...current,
          task: resolvedTask,
          launchInputValues: options.inputValues,
          referenceMemoryIds:
            normalizeCuratedTaskReferenceMemoryIds(
              options.referenceSelection.referenceMemoryIds,
            ) ?? [],
          referenceEntries: mergeCuratedTaskReferenceEntries(
            options.referenceSelection.referenceEntries,
          ),
        };
      });
      setCuratedTaskEditorPrefillHint(
        copy.curatedTaskReviewSuggestionPrefillHint,
      );
    },
    [copy.curatedTaskReviewSuggestionPrefillHint],
  );
  const handleConfirmCuratedTaskEdit = useCallback(
    (
      task: NonNullable<typeof editingCuratedTaskCapability>["task"],
      inputValues: CuratedTaskInputValues,
      referenceSelection: CuratedTaskReferenceSelection,
    ) => {
      const previousPrompt = editingCuratedTaskCapability?.task.prompt;
      const resolvedTask = findCuratedTaskTemplateById(task.id) ?? task;
      const nextPrompt = buildCuratedTaskLaunchPrompt({
        task: resolvedTask,
        inputValues,
        referenceEntries: referenceSelection.referenceEntries,
      });
      setInput(
        replaceCuratedTaskLaunchPromptInInput({
          currentInput: input,
          previousPrompt,
          nextPrompt,
        }),
      );
      setActiveCapability({
        kind: "curated_task",
        task: {
          ...resolvedTask,
          prompt: nextPrompt,
        },
        launchInputValues: inputValues,
        referenceMemoryIds: referenceSelection.referenceMemoryIds,
        referenceEntries: referenceSelection.referenceEntries,
      });
      setCuratedTaskEditorPrefillHint(null);
      setEditingCuratedTaskCapability(null);
    },
    [editingCuratedTaskCapability, input, setInput],
  );
  const dialogLayer = editingCuratedTaskCapability
    ? React.createElement(CuratedTaskLauncherDialog, {
        open: true,
        task: editingCuratedTaskCapability.task,
        projectId,
        sessionId,
        initialInputValues: editingCuratedTaskCapability.launchInputValues,
        initialReferenceMemoryIds:
          editingCuratedTaskCapability.referenceMemoryIds,
        initialReferenceEntries: editingCuratedTaskCapability.referenceEntries,
        prefillHint: curatedTaskEditorPrefillHint,
        onOpenChange: handleCuratedTaskEditorOpenChange,
        onApplyReviewSuggestion: handleApplyCuratedTaskEditorReviewSuggestion,
        onConfirm: handleConfirmCuratedTaskEdit,
      })
    : undefined;
  const handleSelectServiceSkill = (skill: ServiceSkillHomeItem) => {
    setActiveCapability(null);
    onSelectServiceSkill?.(skill);
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
  };
  const handleSelectInputCapability = (
    capability: InputCapabilitySelection,
  ) => {
    if (capability.kind === "service_skill") {
      handleSelectServiceSkill(capability.skill);
      return;
    }
    if (capability.kind === "builtin_command") {
      if (capability.command.key === "knowledge_pack") {
        if (!knowledgePackSelection && !onStartKnowledgeOrganize) {
          onManageKnowledgePacks?.();
        } else {
          onKnowledgePacksNeeded?.();
          setKnowledgeHubOpenRequestKey((current) => current + 1);
        }
        setActiveCapability(null);
        return;
      }

      if (capability.command.key === "knowledge_settle") {
        onStartKnowledgeOrganize?.();
        setActiveCapability(null);
        return;
      }
    }
    setActiveCapability(capability);
  };
  const skillSelection = buildSkillSelectionProps({
    skills,
    serviceSkills,
    serviceSkillGroups,
    activeSkill,
    isSkillsLoading,
    onSelectInputCapability: handleSelectInputCapability,
    onClearSkill: () => setActiveCapability(null),
    onNavigateToSettings,
    onImportSkill,
    onRefreshSkills,
  });

  return {
    textareaRef,
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
    pluginSuggestions,
    activePluginSelection,
    handleSelectPlugin,
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
    knowledgeHubOpenRequestKey,
  };
}
