import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import {
  buildRecommendationPrompt,
  getContextualRecommendations,
  isTeamRuntimeRecommendation,
} from "../utils/contextualRecommendations";
import {
  buildCuratedTaskLaunchPrompt,
  buildCuratedTaskTemplateCopy,
  findCuratedTaskTemplateById,
  listCuratedTaskTemplates,
  recordCuratedTaskTemplateUsage,
  replaceCuratedTaskLaunchPromptInInput,
  subscribeCuratedTaskTemplateUsageChanged,
  type CuratedTaskInputValues,
  type CuratedTaskTemplateItem,
} from "../utils/curatedTaskTemplates";
import { subscribeCuratedTaskRecommendationSignalsChanged } from "../utils/curatedTaskRecommendationSignals";
import type {
  CuratedTaskReferenceEntry,
  CuratedTaskReferenceSelection,
} from "../utils/curatedTaskReferenceSelection";
import { CuratedTaskLauncherDialog } from "./CuratedTaskLauncherDialog";
import { EmptyStateComposerPanel } from "./EmptyStateComposerPanel";
import { EmptyStateQuickActions } from "./EmptyStateQuickActions";
import {
  EmptyStateComposerFrame,
  EmptyStatePrimaryStack,
  EmptyStateLayout,
} from "./EmptyStateLayout";
import { buildSkillSelectionProps } from "../skill-selection/skillSelectionBindings";
import { isGeneralResearchTheme } from "../utils/generalAgentPrompt";
import { buildPathReferenceRequestMetadata } from "../utils/pathReferences";
import { buildKnowledgeRequestMetadata } from "@/features/knowledge/agent/knowledgeMetadata";
import {
  resolveInputCapabilityDispatch,
  resolveInputCapabilitySendRoute,
  resolveInputCapabilitySelectionFromRoute,
  type InputCapabilitySelection,
} from "../skill-selection/inputCapabilitySelection";
import { buildServiceSkillHomeCopy } from "../service-skills/homeCopy";
import { HomeStartSurface } from "../home/HomeStartSurface";
import { buildHomeSurfaceCopy } from "../home/homeSurfaceCopy";
import { buildInputbarCoreCopy } from "./Inputbar/components/inputbarCoreCopy";
import { buildInputbarToolPreferencesOverride } from "./Inputbar/utils/inputbarModeRequestMetadata";
import {
  buildEmptyStateQuickActionItems,
  resolveEffectiveCuratedTaskReferences,
} from "./EmptyStateViewModel";
import { useEmptyStateAttachments } from "./useEmptyStateAttachments";
import { useEmptyStateRecommendationPreferences } from "./useEmptyStateRecommendationPreferences";
import { useHomeSkillSurface } from "./useHomeSkillSurface";
import { useCuratedTaskLauncherState } from "./useCuratedTaskLauncherState";
import { useEmptyStateHomeActions } from "./useEmptyStateHomeActions";
import type { AgentI18nKey, EmptyStateProps } from "./EmptyState.types";
import {
  type BaseComposerSendMetadata,
  useComposerController,
  type ComposerSubmitTarget,
} from "@/components/input-kit";
import { AppServerClient } from "@/lib/api/appServer";
import { readPromptHistory } from "@/lib/api/promptHistory";
import { recordAgentUiPerformanceMetric } from "@/lib/agentUiPerformanceMetrics";
import { getLimeSkinCopy } from "@/lib/appearance/skinContent";
import { useHomeSkinPresentation } from "./homeSkinPresentation";

const CREATION_THEMES: string[] = [];

export const EmptyState: React.FC<EmptyStateProps> = ({
  input,
  setInput,
  onSend,
  onStop,
  sendOnPointerDown = false,
  creationMode = "guided",
  onCreationModeChange,
  activeTheme = "general",
  onThemeChange,
  onRecommendationClick,
  providerType,
  setProviderType,
  model,
  setModel,
  reasoningEffort,
  setReasoningEffort,
  accessMode,
  setAccessMode,
  onManageProviders,
  taskEnabled = false,
  onTaskEnabledChange,
  objectiveEnabled: objectiveEnabledProp,
  onObjectiveEnabledChange: onObjectiveEnabledChangeProp,
  subagentEnabled = false,
  onSubagentEnabledChange,
  hasCanvasContent = false,
  hasContentId = false,
  selectedText = "",
  characters = [],
  skills,
  serviceSkills,
  serviceSkillGroups,
  isSkillsLoading,
  onSelectServiceSkill,
  onNavigateToSettings,
  onImportSkill,
  onRefreshSkills,
  onLaunchBrowserAssist,
  recentSessionTitle = null,
  recentSessionSummary = null,
  recentSessionActionLabel,
  homeRecoverySession = null,
  onResumeRecentSession,
  projectConversationGroups = [],
  onOpenProjectConversation,
  projectId = null,
  openedProjects = [],
  onProjectContextChange,
  sessionId = null,
  pluginSuggestions = [],
  pluginSuggestionsError = null,
  pluginSuggestionsLoading = false,
  onPluginSuggestionsNeeded,
  isLoading = false,
  disabled = false,
  initialInputCapability,
  creationReplaySurface = null,
  defaultCuratedTaskReferenceMemoryIds,
  defaultCuratedTaskReferenceEntries,
  knowledgePackSelection = null,
  knowledgePackOptions = [],
  onToggleKnowledgePack,
  onSelectKnowledgePack,
  onToggleKnowledgeCompanionPack,
  onStartKnowledgeOrganize,
  onManageKnowledgePacks,
  pathReferences = [],
  onAddPathReferences,
  inputRestoreRequest = null,
  onInputRestoreRequestHandled,
  onImportPathReferenceAsKnowledge,
  onRemovePathReference,
  onClearPathReferences,
  fileManagerOpen = false,
  onToggleFileManager,
}) => {
  const { t, i18n } = useTranslation("agent");
  const {
    controller: composerController,
    snapshot: composerSnapshot,
    setText: setComposerControllerText,
    setAttachments: setComposerControllerAttachments,
    setPathReferences: setComposerControllerPathReferences,
  } = useComposerController({
    initialDocument: {
      text: input,
      pathReferences,
    },
  });
  const promptHistoryClientRef = useRef<AppServerClient | null>(null);
  if (!promptHistoryClientRef.current) {
    promptHistoryClientRef.current = new AppServerClient();
  }
  useEffect(() => {
    let cancelled = false;
    void readPromptHistory(promptHistoryClientRef.current ?? undefined)
      .then((entries) => {
        if (!cancelled) {
          composerController.mergeHistory(
            entries.reverse().map((entry) => ({ text: entry.text })),
          );
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [composerController]);
  const setComposerText = useCallback(
    (value: string) => {
      setComposerControllerText(value);
      setInput(value);
    },
    [setComposerControllerText, setInput],
  );
  useEffect(() => {
    setComposerControllerText(input);
  }, [input, setComposerControllerText]);
  const homeSkinPresentation = useHomeSkinPresentation();
  const translateAgentCopyKey = useCallback(
    (key: string, values?: Record<string, number | string>) =>
      t(key as AgentI18nKey, values ?? {}),
    [t],
  );
  const curatedTaskTemplateCopy = useMemo(
    () => buildCuratedTaskTemplateCopy(translateAgentCopyKey),
    [translateAgentCopyKey],
  );
  const serviceSkillHomeCopy = useMemo(
    () => buildServiceSkillHomeCopy(translateAgentCopyKey),
    [translateAgentCopyKey],
  );
  const skinCopy = useMemo(
    () => getLimeSkinCopy(homeSkinPresentation.skinId, i18n.language),
    [homeSkinPresentation.skinId, i18n.language],
  );
  const homeSurfaceCopy = useMemo(
    () => buildHomeSurfaceCopy(translateAgentCopyKey, skinCopy),
    [skinCopy, translateAgentCopyKey],
  );
  const inputbarCoreCopy = useMemo(
    () => buildInputbarCoreCopy(translateAgentCopyKey),
    [translateAgentCopyKey],
  );
  const handledInitialInputCapabilitySignatureRef = useRef("");
  const inputRestoreEpochRef = useRef(0);
  const [activeCapability, setActiveCapability] =
    useState<InputCapabilitySelection | null>(null);
  const [knowledgeHubOpenRequestKey, setKnowledgeHubOpenRequestKey] =
    useState(0);
  const [localObjectiveEnabled, setLocalObjectiveEnabled] = useState(false);
  const objectiveEnabled = objectiveEnabledProp ?? localObjectiveEnabled;
  const handleObjectiveEnabledChange = useCallback(
    (enabled: boolean) => {
      if (onObjectiveEnabledChangeProp) {
        onObjectiveEnabledChangeProp(enabled);
        return;
      }
      setLocalObjectiveEnabled(enabled);
    },
    [onObjectiveEnabledChangeProp],
  );
  const activeCuratedTaskCapability =
    activeCapability?.kind === "curated_task" ? activeCapability : null;
  const activeCuratedTask = activeCuratedTaskCapability?.task ?? null;
  const activeCuratedTaskLaunchInputValues =
    activeCuratedTaskCapability?.launchInputValues;
  const activeCuratedTaskReferenceMemoryIds =
    activeCuratedTaskCapability?.referenceMemoryIds;
  const activeCuratedTaskReferenceEntries =
    activeCuratedTaskCapability?.referenceEntries;
  const {
    effectiveDefaultCuratedTaskReferenceMemoryIds,
    effectiveDefaultCuratedTaskReferenceEntries,
  } = resolveEffectiveCuratedTaskReferences({
    defaultCuratedTaskReferenceMemoryIds,
    defaultCuratedTaskReferenceEntries,
    creationReplaySurface,
  });
  const currentSkill =
    activeCapability?.kind === "installed_skill"
      ? activeCapability.skill
      : null;
  const activeBuiltinCommandKey =
    activeCapability?.kind === "builtin_command"
      ? activeCapability.command.key
      : null;
  const clearSelectedSkill = useCallback(() => {
    setActiveCapability(null);
  }, []);
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

    if (
      route.kind === "curated_task" &&
      !input.trim() &&
      route.prompt.trim().length > 0
    ) {
      setComposerText(route.prompt);
    }

    setActiveCapability(resolvedCapability);
  }, [
    initialInputCapability,
    initialInputCapabilitySignature,
    input,
    setComposerText,
    skills,
  ]);
  const handleSelectInputCapability = useCallback(
    (capability: InputCapabilitySelection) => {
      if (capability.kind === "service_skill") {
        setActiveCapability(null);
        onSelectServiceSkill?.(capability.skill);
        return;
      }
      if (capability.kind === "builtin_command") {
        if (capability.command.key === "knowledge_pack") {
          if (!knowledgePackSelection && !onStartKnowledgeOrganize) {
            onManageKnowledgePacks?.();
          } else {
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
    },
    [
      knowledgePackSelection,
      onManageKnowledgePacks,
      onSelectServiceSkill,
      onStartKnowledgeOrganize,
    ],
  );
  useEffect(() => {
    if (activeBuiltinCommandKey !== "knowledge_pack") {
      return;
    }
    setActiveCapability(null);
    if (!knowledgePackSelection && !onStartKnowledgeOrganize) {
      onManageKnowledgePacks?.();
      return;
    }
    setKnowledgeHubOpenRequestKey((current) => current + 1);
  }, [
    activeBuiltinCommandKey,
    knowledgePackSelection,
    onManageKnowledgePacks,
    onStartKnowledgeOrganize,
  ]);
  const skillSelection = buildSkillSelectionProps({
    skills,
    serviceSkills,
    serviceSkillGroups,
    isSkillsLoading,
    activeSkill: currentSkill,
    onSelectInputCapability: handleSelectInputCapability,
    onClearSkill: clearSelectedSkill,
    onNavigateToSettings,
    onImportSkill,
    onRefreshSkills,
  });
  const [curatedTaskTemplatesVersion, setCuratedTaskTemplatesVersion] =
    useState(0);
  const [
    curatedTaskRecommendationSignalsVersion,
    setCuratedTaskRecommendationSignalsVersion,
  ] = useState(0);
  const { appendSelectedTextToRecommendation } =
    useEmptyStateRecommendationPreferences();
  const {
    applyReviewSuggestion: applyLauncherReviewSuggestion,
    handleOpenChange: handleCuratedTaskLauncherOpenChange,
    initialInputValues: curatedTaskLauncherInitialInputValues,
    initialReferenceEntries: curatedTaskLauncherInitialReferenceEntries,
    initialReferenceMemoryIds: curatedTaskLauncherInitialReferenceMemoryIds,
    open: openCuratedTaskLauncher,
    prefillHint: curatedTaskLauncherPrefillHint,
    reset: resetCuratedTaskLauncher,
    task: curatedTaskLauncherTask,
  } = useCuratedTaskLauncherState({
    effectiveDefaultCuratedTaskReferenceEntries,
    effectiveDefaultCuratedTaskReferenceMemoryIds,
    reviewSuggestionPrefillHint:
      homeSurfaceCopy.curatedTaskReviewSuggestionPrefillHint,
  });

  useEffect(() => {
    return subscribeCuratedTaskTemplateUsageChanged(() => {
      setCuratedTaskTemplatesVersion((previous) => previous + 1);
    });
  }, []);

  useEffect(() => {
    return subscribeCuratedTaskRecommendationSignalsChanged(() => {
      setCuratedTaskRecommendationSignalsVersion((previous) => previous + 1);
    });
  }, []);

  // 使用外部传入的 activeTheme，如果有 onThemeChange 则使用受控模式
  const handleThemeChange = useCallback(
    (theme: string) => {
      if (onThemeChange) {
        onThemeChange(theme === "general" ? theme : "general");
      }
    },
    [onThemeChange],
  );

  // 判断当前主题是否需要显示创作模式选择器
  const showCreationModeSelector = CREATION_THEMES.includes(activeTheme);

  const isGeneralTheme = isGeneralResearchTheme(activeTheme);
  const isComposerBusy = isLoading || disabled;
  const {
    clearPendingImages,
    handleDragOver,
    handleDrop,
    handleFileSelect,
    handlePaste,
    handleRemoveImage,
    pendingImages,
    replacePendingImages,
  } = useEmptyStateAttachments({
    toastCopy: homeSurfaceCopy.toast,
    onAddPathReferences,
  });

  useEffect(() => {
    setComposerControllerAttachments(pendingImages);
  }, [pendingImages, setComposerControllerAttachments]);
  useEffect(() => {
    setComposerControllerPathReferences(pathReferences);
  }, [pathReferences, setComposerControllerPathReferences]);
  useEffect(() => {
    composerController.setInputCapabilityRoute(
      resolveInputCapabilitySendRoute(activeCapability),
    );
  }, [activeCapability, composerController]);

  useEffect(() => {
    if (!inputRestoreRequest || isComposerBusy) {
      return;
    }

    inputRestoreEpochRef.current += 1;
    const { draft, requestId } = inputRestoreRequest;
    const restoredPathReferences = [...(draft.pathReferences ?? [])];
    setComposerText(draft.text);
    replacePendingImages([...(draft.images ?? [])]);
    onClearPathReferences?.();
    if (restoredPathReferences.length > 0) {
      onAddPathReferences?.(restoredPathReferences);
    }
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
    isComposerBusy,
    onAddPathReferences,
    onClearPathReferences,
    onInputRestoreRequestHandled,
    replacePendingImages,
    setComposerText,
    skills,
  ]);

  const recommendationSelectedText = appendSelectedTextToRecommendation
    ? selectedText
    : "";

  const currentRecommendations = useMemo(() => {
    return getContextualRecommendations({
      activeTheme,
      input,
      creationMode,
      hasCanvasContent,
      hasContentId,
      selectedText: recommendationSelectedText,
      subagentEnabled,
    });
  }, [
    activeTheme,
    input,
    creationMode,
    hasCanvasContent,
    hasContentId,
    recommendationSelectedText,
    subagentEnabled,
  ]);

  const curatedTaskTemplates = useMemo(() => {
    void curatedTaskTemplatesVersion;
    void curatedTaskRecommendationSignalsVersion;
    return listCuratedTaskTemplates(curatedTaskTemplateCopy);
  }, [
    curatedTaskRecommendationSignalsVersion,
    curatedTaskTemplateCopy,
    curatedTaskTemplatesVersion,
  ]);

  const selectedTextPreview = useMemo(() => {
    const normalized = (recommendationSelectedText || "")
      .trim()
      .replace(/\s+/g, " ");
    if (!normalized) {
      return "";
    }

    return normalized.length > 56
      ? `${normalized.slice(0, 56).trim()}…`
      : normalized;
  }, [recommendationSelectedText]);

  const handleSend = (
    inputOverride = input,
    modeState?: {
      goalEnabled?: boolean;
      planEnabled?: boolean;
      subagentEnabled?: boolean;
    },
    triggerMetadata?: BaseComposerSendMetadata,
  ) => {
    const composerInputOverride =
      inputOverride === input
        ? composerController.getDocument().text
        : inputOverride;
    const handlerEnteredAt = Date.now();
    const triggeredAt =
      typeof triggerMetadata?.triggeredAt === "number" &&
      Number.isFinite(triggerMetadata.triggeredAt)
        ? triggerMetadata.triggeredAt
        : handlerEnteredAt;
    const triggerSource = triggerMetadata?.triggerSource ?? "adapter";
    recordAgentUiPerformanceMetric("inputbar.send.enter", {
      durationMs: Math.max(0, handlerEnteredAt - triggeredAt),
      hasTriggerMetadata: Boolean(triggerMetadata),
      imageCount: pendingImages.length,
      inputLength: composerInputOverride.trim().length,
      pathReferenceCount: pathReferences.length,
      sessionId: sessionId ?? null,
      source: "empty-state",
      triggerSource,
      workspaceId: projectId ?? null,
    });
    const sendRestoreEpoch = inputRestoreEpochRef.current;
    const hasPathReferences = pathReferences.length > 0;
    if (
      isComposerBusy ||
      (!composerInputOverride.trim() &&
        pendingImages.length === 0 &&
        !hasPathReferences)
    ) {
      return;
    }
    const imagesToSend = pendingImages.length > 0 ? pendingImages : undefined;
    const capabilityDispatch = resolveInputCapabilityDispatch(
      activeCapability,
      composerInputOverride,
    );
    const submitTarget: ComposerSubmitTarget =
      triggerMetadata?.submitTarget ?? (isLoading ? "steer" : "start");
    setComposerControllerText(composerInputOverride);
    setComposerControllerAttachments(pendingImages);
    setComposerControllerPathReferences(pathReferences);
    composerController.setInputCapabilityRoute(
      capabilityDispatch.capabilityRoute,
    );
    const composerReceipt = composerController.submit(submitTarget);
    if (composerReceipt.kind === "empty") {
      return;
    }
    const baseRequestMetadata = buildPathReferenceRequestMetadata(
      capabilityDispatch.requestMetadata,
      pathReferences,
    );
    const knowledgeRequestMetadata =
      knowledgePackSelection?.enabled &&
      knowledgePackSelection.packName.trim() &&
      knowledgePackSelection.workingDir.trim()
        ? {
            ...(baseRequestMetadata || {}),
            ...buildKnowledgeRequestMetadata({
              workingDir: knowledgePackSelection.workingDir.trim(),
              packName: knowledgePackSelection.packName.trim(),
              packs: knowledgePackSelection.companionPacks,
              source: "inputbar",
            }),
          }
        : baseRequestMetadata;
    const inputbarModeState = {
      planEnabled: modeState?.planEnabled ?? taskEnabled,
      subagentEnabled: modeState?.subagentEnabled ?? subagentEnabled,
    };
    const requestMetadata = knowledgeRequestMetadata;
    const threadGoal =
      (modeState?.goalEnabled ?? objectiveEnabled) && composerInputOverride.trim()
        ? { objective: composerInputOverride.trim() }
        : undefined;
    const toolPreferencesOverride =
      buildInputbarToolPreferencesOverride(inputbarModeState);
    const collaborationMode = inputbarModeState.planEnabled
      ? ("plan" as const)
      : undefined;
    const effectiveInput = composerInputOverride.trim()
      ? composerInputOverride
      : hasPathReferences
        ? homeSurfaceCopy.composerPathReferenceFallbackPrompt
        : composerInputOverride;
    const inputRestoreDraft = {
      text: composerInputOverride.trim() ? composerInputOverride : "",
      images: [...pendingImages],
      pathReferences: [...pathReferences],
      inputCapabilityRoute: capabilityDispatch.capabilityRoute,
    };
    const shouldAttachInputRestoreDraft =
      pendingImages.length > 0 || pathReferences.length > 0;
    const sendOptions =
      capabilityDispatch.capabilityRoute ||
      capabilityDispatch.displayContent ||
      requestMetadata ||
      threadGoal ||
      toolPreferencesOverride ||
      collaborationMode ||
      shouldAttachInputRestoreDraft
        ? {
            ...(capabilityDispatch.capabilityRoute
              ? { capabilityRoute: capabilityDispatch.capabilityRoute }
              : {}),
            ...(shouldAttachInputRestoreDraft ? { inputRestoreDraft } : {}),
            ...(capabilityDispatch.displayContent
              ? { displayContent: capabilityDispatch.displayContent }
              : {}),
            ...(requestMetadata ? { requestMetadata } : {}),
            ...(threadGoal ? { threadGoal } : {}),
            ...(toolPreferencesOverride ? { toolPreferencesOverride } : {}),
            ...(collaborationMode ? { collaborationMode } : {}),
          }
        : undefined;

    const sendResult = onSend({
      images: imagesToSend,
      textOverride: effectiveInput,
      sendOptions,
      composerIntent: composerReceipt.intent,
      composerTarget: composerReceipt.target,
      composerDraft: composerReceipt.draft,
      ...(triggerMetadata
        ? {
            triggeredAt: triggerMetadata.triggeredAt,
            triggerSource: triggerMetadata.triggerSource,
          }
        : {}),
    });
    const clearAcceptedSubmissionState = () => {
      if (inputRestoreEpochRef.current !== sendRestoreEpoch) {
        return;
      }
      if (!composerController.commit(composerReceipt)) {
        return;
      }
      if (sessionId && composerReceipt.draft.text.trim()) {
        void promptHistoryClientRef.current
          ?.appendPromptHistory({
            sessionId,
            text: composerReceipt.draft.text,
          })
          .catch(() => undefined);
      }
      setComposerText("");
      clearPendingImages();
      onClearPathReferences?.();
      clearSelectedSkill?.();
    };
    if (sendResult === false) {
      return false;
    }
    if (sendResult && typeof sendResult === "object" && "then" in sendResult) {
      return sendResult.then((accepted) => {
        if (accepted !== false) {
          clearAcceptedSubmissionState();
        }
        return accepted;
      });
    }
    clearAcceptedSubmissionState();
    return sendResult;
  };

  const getPlaceholder = () => homeSurfaceCopy.composerPlaceholder;

  const handleApplyRecommendation = useCallback(
    (shortLabel: string, fullPrompt: string) => {
      const looksLikeTeamRuntimePrompt =
        activeTheme === "general" &&
        isTeamRuntimeRecommendation(shortLabel, fullPrompt);
      if (looksLikeTeamRuntimePrompt) {
        onSubagentEnabledChange?.(true);
      }

      const promptWithSelection = buildRecommendationPrompt(
        fullPrompt,
        selectedText,
        appendSelectedTextToRecommendation,
      );
      if (onRecommendationClick) {
        onRecommendationClick(shortLabel, promptWithSelection);
        return;
      }
      setComposerText(promptWithSelection);
    },
    [
      activeTheme,
      appendSelectedTextToRecommendation,
      onRecommendationClick,
      onSubagentEnabledChange,
      selectedText,
      setComposerText,
    ],
  );

  const handleCuratedTaskLauncherRequest = useCallback(
    (
      template: CuratedTaskTemplateItem,
      initialInputValues?: CuratedTaskInputValues | null,
      initialReferenceMemoryIds?: string[] | null,
      initialReferenceEntries?: CuratedTaskReferenceEntry[] | null,
      prefillHint?: string | null,
    ) => {
      openCuratedTaskLauncher(
        template,
        initialInputValues,
        initialReferenceMemoryIds,
        initialReferenceEntries,
        prefillHint,
      );
    },
    [openCuratedTaskLauncher],
  );

  const handleApplyLauncherReviewSuggestion = useCallback(
    (
      template: CuratedTaskTemplateItem,
      options: {
        inputValues: CuratedTaskInputValues;
        referenceSelection: CuratedTaskReferenceSelection;
      },
    ) => {
      applyLauncherReviewSuggestion(template, options);
    },
    [applyLauncherReviewSuggestion],
  );

  const handleApplyCuratedTaskTemplate = useCallback(
    (
      template: CuratedTaskTemplateItem,
      inputValues: CuratedTaskInputValues,
      referenceSelection: CuratedTaskReferenceSelection,
    ) => {
      recordCuratedTaskTemplateUsage({
        templateId: template.id,
        launchInputValues: inputValues,
        referenceMemoryIds: referenceSelection.referenceMemoryIds,
        referenceEntries: referenceSelection.referenceEntries,
      });
      setCuratedTaskTemplatesVersion((previous) => previous + 1);
      resetCuratedTaskLauncher();

      if (template.shouldEnableTeamMode && !subagentEnabled) {
        onSubagentEnabledChange?.(true);
      }

      if (template.themeTarget) {
        handleThemeChange(template.themeTarget);
      }

      if (template.shouldLaunchBrowserAssist) {
        void onLaunchBrowserAssist?.();
      }

      const resolvedTemplate =
        findCuratedTaskTemplateById(template.id, curatedTaskTemplateCopy) ??
        template;
      const launchPrompt = buildCuratedTaskLaunchPrompt({
        task: resolvedTemplate,
        inputValues,
        referenceEntries: referenceSelection.referenceEntries,
      });
      const nextPrompt = buildRecommendationPrompt(
        launchPrompt,
        selectedText,
        appendSelectedTextToRecommendation,
      );
      const promptWithSelection = replaceCuratedTaskLaunchPromptInInput({
        currentInput: input,
        previousPrompt:
          activeCuratedTask?.id === template.id
            ? activeCuratedTask.prompt
            : null,
        nextPrompt,
      });
      setActiveCapability({
        kind: "curated_task",
        task: {
          ...resolvedTemplate,
          prompt: nextPrompt,
        },
        launchInputValues: inputValues,
        referenceMemoryIds: referenceSelection.referenceMemoryIds,
        referenceEntries: referenceSelection.referenceEntries,
      });
      setComposerText(promptWithSelection);
    },
    [
      activeCuratedTask,
      appendSelectedTextToRecommendation,
      curatedTaskTemplateCopy,
      handleThemeChange,
      input,
      onLaunchBrowserAssist,
      onSubagentEnabledChange,
      resetCuratedTaskLauncher,
      selectedText,
      setComposerText,
      subagentEnabled,
    ],
  );

  const quickActionItems = useMemo(
    () =>
      buildEmptyStateQuickActionItems({
        activeTheme,
        recommendations: currentRecommendations,
        resolveBadge: homeSurfaceCopy.quickActions.badge,
      }),
    [activeTheme, currentRecommendations, homeSurfaceCopy.quickActions],
  );

  const quickStartPresets = homeSurfaceCopy.quickActions.presets;

  const {
    galleryItems: homeGalleryItems,
    guideCards: homeGuideCards,
    serviceSkillItems: homeServiceSkillItems,
    skillItems: homeSkillItems,
    skillSections: homeSkillSections,
    starterChips: homeStarterChips,
  } = useHomeSkillSurface({
    copy: homeSurfaceCopy,
    curatedTasks: curatedTaskTemplates,
    installedSkills: skillSelection.skills ?? [],
    serviceSkillHomeCopy,
    serviceSkills: serviceSkills ?? [],
  });
  const handleOpenKnowledgeHub = useCallback(
    () => setKnowledgeHubOpenRequestKey((current) => current + 1),
    [],
  );
  const {
    guideHelpActive,
    guideHelpLabel,
    handleSelectHomeGuideCard,
    handleSelectHomeSkillItem,
    handleSelectHomeStarterChip,
    homeSupplementalActions,
    setGuideHelpActive,
  } = useEmptyStateHomeActions({
    curatedTaskTemplateCopy,
    effectiveDefaultCuratedTaskReferenceEntries,
    effectiveDefaultCuratedTaskReferenceMemoryIds,
    guideHelpContextLabel: homeSurfaceCopy.guideHelpContextLabel,
    guideHelpContextLabelWithStarter:
      homeSurfaceCopy.guideHelpContextLabelWithStarter,
    handleCuratedTaskLauncherRequest,
    handleSelectInputCapability,
    homeServiceSkillItems,
    homeSkillItems,
    homeStarterChips,
    isGeneralTheme,
    knowledgePackSelection,
    onManageKnowledgePacks,
    onOpenKnowledgeHub: handleOpenKnowledgeHub,
    onResumeRecentSession,
    onStartKnowledgeOrganize,
    recentSessionActionLabel,
    recentSessionDefaultActionLabel:
      homeSurfaceCopy.chrome.recentSessionDefaultActionLabel,
    recentSessionSummary,
    recentSessionTitle,
    serviceSkills,
    setActiveCapability,
    setInput: setComposerText,
    skillSelectionSkills: skillSelection.skills,
  });

  const handleStop = useCallback(() => {
    composerController.submit("interrupt", { allowEmpty: true });
    onStop?.();
  }, [composerController, onStop]);

  const composerPanel = (
    <EmptyStateComposerFrame>
      <EmptyStateComposerPanel
        input={composerSnapshot.document.text}
        setInput={setComposerText}
        controller={composerController}
        placeholder={
          guideHelpActive
            ? homeSurfaceCopy.guideHelpPlaceholder
            : getPlaceholder()
        }
        onSend={handleSend}
        onStop={handleStop}
        sendOnPointerDown={sendOnPointerDown}
        activeTheme={activeTheme}
        providerType={providerType}
        setProviderType={setProviderType}
        model={model}
        setModel={setModel}
        reasoningEffort={reasoningEffort}
        setReasoningEffort={setReasoningEffort}
        accessMode={accessMode}
        setAccessMode={setAccessMode}
        onManageProviders={onManageProviders}
        isLoading={isComposerBusy}
        disabled={isComposerBusy}
        isGeneralTheme={isGeneralTheme}
        characters={characters}
        skillSelection={skillSelection}
        activeCapability={activeCapability}
        onSelectInputCapability={handleSelectInputCapability}
        onClearInputCapability={clearSelectedSkill}
        onEditCuratedTask={
          activeCuratedTask
            ? () =>
                handleCuratedTaskLauncherRequest(
                  activeCuratedTask,
                  activeCuratedTaskLaunchInputValues,
                  activeCuratedTaskReferenceMemoryIds ||
                    effectiveDefaultCuratedTaskReferenceMemoryIds,
                  activeCuratedTaskReferenceEntries ||
                    effectiveDefaultCuratedTaskReferenceEntries,
                )
            : undefined
        }
        onApplyCuratedTaskReviewSuggestion={
          activeCuratedTask
            ? (task) =>
                handleCuratedTaskLauncherRequest(
                  task,
                  activeCuratedTaskLaunchInputValues,
                  activeCuratedTaskReferenceMemoryIds ||
                    effectiveDefaultCuratedTaskReferenceMemoryIds,
                  activeCuratedTaskReferenceEntries ||
                    effectiveDefaultCuratedTaskReferenceEntries,
                  homeSurfaceCopy.curatedTaskReviewSuggestionPrefillHint,
                )
            : undefined
        }
        creationReplaySurface={creationReplaySurface}
        projectId={projectId}
        openedProjects={openedProjects}
        onProjectContextChange={onProjectContextChange}
        sessionId={sessionId}
        defaultCuratedTaskReferenceMemoryIds={
          effectiveDefaultCuratedTaskReferenceMemoryIds
        }
        defaultCuratedTaskReferenceEntries={
          effectiveDefaultCuratedTaskReferenceEntries
        }
        knowledgePackSelection={knowledgePackSelection}
        knowledgePackOptions={knowledgePackOptions}
        knowledgeHubOpenRequestKey={knowledgeHubOpenRequestKey}
        onToggleKnowledgePack={onToggleKnowledgePack}
        onSelectKnowledgePack={onSelectKnowledgePack}
        onToggleKnowledgeCompanionPack={onToggleKnowledgeCompanionPack}
        onStartKnowledgeOrganize={onStartKnowledgeOrganize}
        onManageKnowledgePacks={onManageKnowledgePacks}
        copy={homeSurfaceCopy.composer}
        inputbarCopy={inputbarCoreCopy}
        pluginSuggestions={pluginSuggestions}
        pluginSuggestionsError={pluginSuggestionsError}
        pluginSuggestionsLoading={pluginSuggestionsLoading}
        onPluginSuggestionsNeeded={onPluginSuggestionsNeeded}
        showCreationModeSelector={showCreationModeSelector}
        creationMode={creationMode}
        onCreationModeChange={onCreationModeChange}
        taskEnabled={taskEnabled}
        onTaskEnabledChange={onTaskEnabledChange}
        objectiveEnabled={objectiveEnabled}
        onObjectiveEnabledChange={handleObjectiveEnabledChange}
        subagentEnabled={subagentEnabled}
        onSubagentEnabledChange={onSubagentEnabledChange}
        pendingImages={pendingImages}
        onFileSelect={handleFileSelect}
        onPaste={handlePaste}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onRemoveImage={handleRemoveImage}
        pathReferences={pathReferences}
        onImportPathReferenceAsKnowledge={onImportPathReferenceAsKnowledge}
        onRemovePathReference={onRemovePathReference}
        fileManagerOpen={fileManagerOpen}
        onToggleFileManager={onToggleFileManager}
        guideHelpActive={guideHelpActive}
        guideHelpLabel={guideHelpLabel}
        onClearGuideHelp={() => setGuideHelpActive(false)}
      />
    </EmptyStateComposerFrame>
  );

  const defaultQuickActionsPanel = (
    <EmptyStateQuickActions
      title={homeSurfaceCopy.quickActions.title}
      description={homeSurfaceCopy.quickActions.description}
      selectedTextPreview={selectedTextPreview}
      presets={quickStartPresets}
      items={quickActionItems}
      embedded
      onPresetAction={(item) =>
        handleApplyRecommendation(item.label, item.prompt)
      }
      onAction={(item) => handleApplyRecommendation(item.title, item.prompt)}
    />
  );

  const homeStartSurfacePanel = (
    <HomeStartSurface
      starterChips={homeStarterChips}
      copy={homeSurfaceCopy.chrome}
      guideCards={homeGuideCards}
      guideOpen={guideHelpActive}
      recoverySession={homeRecoverySession}
      sections={homeSkillSections}
      conversationGroups={projectConversationGroups}
      supplementalActions={homeSupplementalActions}
      onGuideOpenChange={setGuideHelpActive}
      onSelectRecoverySession={onResumeRecentSession}
      onSelectConversation={onOpenProjectConversation}
      onSelectStarterChip={handleSelectHomeStarterChip}
      onSelectGuideCard={handleSelectHomeGuideCard}
      onSelectSkillItem={handleSelectHomeSkillItem}
    />
  );

  const priorityPanel = isGeneralTheme ? (
    <EmptyStatePrimaryStack>
      {composerPanel}
      {homeStartSurfacePanel}
    </EmptyStatePrimaryStack>
  ) : (
    composerPanel
  );

  return (
    <EmptyStateLayout
      heroCopy={homeSurfaceCopy.hero}
      chromeCopy={homeSurfaceCopy.chrome}
      prioritySlot={priorityPanel}
      supportingSlot={isGeneralTheme ? null : defaultQuickActionsPanel}
      isGeneralTheme={isGeneralTheme}
      galleryItems={homeGalleryItems}
      onSelectGalleryItem={handleSelectHomeSkillItem}
    >
      <CuratedTaskLauncherDialog
        open={Boolean(curatedTaskLauncherTask)}
        task={curatedTaskLauncherTask}
        projectId={projectId}
        sessionId={sessionId}
        initialInputValues={curatedTaskLauncherInitialInputValues}
        initialReferenceMemoryIds={curatedTaskLauncherInitialReferenceMemoryIds}
        initialReferenceEntries={curatedTaskLauncherInitialReferenceEntries}
        prefillHint={curatedTaskLauncherPrefillHint}
        onOpenChange={handleCuratedTaskLauncherOpenChange}
        onApplyReviewSuggestion={handleApplyLauncherReviewSuggestion}
        onConfirm={handleApplyCuratedTaskTemplate}
      />
    </EmptyStateLayout>
  );
};
