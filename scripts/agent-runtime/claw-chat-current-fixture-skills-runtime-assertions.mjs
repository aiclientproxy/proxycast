import fs from "node:fs";
import {
  EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF,
  EXPERT_SKILLS_RUNTIME_ID,
  EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  EXPERT_SKILLS_RUNTIME_PROMPT,
  EXPERT_SKILLS_RUNTIME_SESSION_TITLE,
  EXPERT_SKILLS_RUNTIME_SKILL_REF,
  EXPERT_SKILLS_RUNTIME_TITLE,
  SKILLS_RUNTIME_EXPLICIT_PROMPT,
  SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
  SKILLS_RUNTIME_PROMPT,
  SKILLS_RUNTIME_QUERY,
  SKILLS_RUNTIME_SKILL_NAME,
} from "./claw-chat-current-fixture-constants.mjs";
import { EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF } from "./claw-chat-current-fixture-expert-actions.mjs";

function skillRuntimeEvidence(summary) {
  return summary?.runtimeStatus ?? summary ?? {};
}

export function buildSkillsRuntimeScenarioAssertions({
  explicitSkillsRuntimeTurnStart,
  manualEnableRuntimeBinding,
  manualEnableRuntimeMetadata,
  manualEnableSkillsRuntimeTurnStart,
  skillsRuntimeTurnStart,
  summary,
  workspace,
}) {
  return {
    initialCurrentSkillListObserved:
      summary.skillsChangedCatalogRefresh?.initialCatalog?.afterPanelOpen
        ?.method === "skills/list" &&
      summary.skillsChangedCatalogRefresh?.initialCatalog?.afterPanelOpen
        ?.electronIpcSuccessCount >
        summary.skillsChangedCatalogRefresh?.initialCatalog?.beforeOpen
          ?.electronIpcSuccessCount,
    typedSkillsChangedConsumed:
      summary.skillsChangedCatalogRefresh?.notification?.method ===
        "skills/changed" &&
      summary.skillsChangedCatalogRefresh?.notification?.marker ===
        "skillsChanged.received" &&
      summary.skillsChangedCatalogRefresh?.notification?.markerCount > 0,
    automaticSkillListRefreshObserved:
      summary.skillsChangedCatalogRefresh?.automaticRefresh?.method ===
        "skills/list" &&
      summary.skillsChangedCatalogRefresh?.automaticRefresh?.transport ===
        "electron-ipc" &&
      summary.skillsChangedCatalogRefresh?.automaticRefresh?.increment > 0,
    guiSkillCatalogUpdated:
      summary.skillsChangedCatalogRefresh?.gui?.panelVisible === true &&
      summary.skillsChangedCatalogRefresh?.gui?.selectorVisible === true &&
      summary.skillsChangedCatalogRefresh?.gui?.skillVisible === true,
    skillsCatalogRefreshDidNotUseManualRefresh:
      summary.skillsChangedCatalogRefresh?.manualRefresh?.clickCount === 0,
    skillsRuntimePromptReachedBackend:
      skillsRuntimeTurnStart?.inputText === SKILLS_RUNTIME_PROMPT,
    guiSkillsRuntimeInputSubmitted:
      summary.skillsRuntimeInputSend?.afterFill?.promptVisibleInTextarea ===
        true && summary.skillsRuntimeInputSend?.clicked?.clicked === true,
    guiSkillsRuntimeCompleted:
      summary.guiSkillsRuntimeCompleted?.hasPrompt === true &&
      (summary.guiSkillsRuntimeCompleted?.hasAssistantSummary === true ||
        summary.guiSkillsRuntimeCompleted?.hasDoneText === true) &&
      summary.guiSkillsRuntimeCompleted?.textareaVisible === true &&
      summary.guiSkillsRuntimeCompleted?.textareaDisabled === false &&
      summary.guiSkillsRuntimeCompleted?.stopButtonVisible === false,
    readModelSkillsRuntimeCompleted:
      summary.readModelSkillsRuntimeCompleted?.includesPrompt === true &&
      (summary.readModelSkillsRuntimeCompleted?.includesAssistantDone ===
        true ||
        summary.readModelSkillsRuntimeCompleted?.includesAssistantSummary ===
          true),
    readModelSkillSearchObserved:
      summary.readModelSkillsRuntimeCompleted?.includesSkillSearchTool === true,
    readModelSkillInvocationObserved:
      summary.readModelSkillsRuntimeCompleted?.includesSkillTool === true &&
      summary.readModelSkillsRuntimeCompleted?.includesSkillName === true,
    readModelSkillBodyReadObserved:
      skillRuntimeEvidence(summary.readModelSkillsRuntime)
        .skillBodyReadObserved === true,
    readModelSkillGateObserved:
      skillRuntimeEvidence(summary.readModelSkillsRuntime).skillGateObserved ===
      true,
    readModelSkillSearchObserved:
      summary.readModelSkillsRuntime?.hasSkillSearchSummary === true &&
      summary.readModelSkillsRuntime?.searchQuery === SKILLS_RUNTIME_QUERY,
    readModelSkillInvocationObserved:
      summary.readModelSkillsRuntime?.hasSkillInvocationSummary === true &&
      summary.readModelSkillsRuntime?.invocationSkillName ===
        SKILLS_RUNTIME_SKILL_NAME,
    skillSearchBeforeSkillInvocation:
      summary.readModelSkillsRuntime?.skillSearchBeforeSkillInvocation ===
      true,
    explicitSkillsRuntimePromptReachedBackend:
      explicitSkillsRuntimeTurnStart?.inputText ===
      SKILLS_RUNTIME_EXPLICIT_PROMPT,
    guiExplicitSkillsRuntimeInputSubmitted:
      summary.explicitSkillsRuntimeInputSend?.afterFill
        ?.promptVisibleInTextarea === true &&
      summary.explicitSkillsRuntimeInputSend?.clicked?.clicked === true,
    guiExplicitSkillsRuntimeCompleted:
      summary.guiExplicitSkillsRuntimeCompleted?.hasPrompt === true &&
      (summary.guiExplicitSkillsRuntimeCompleted?.hasAssistantSummary ===
        true ||
        summary.guiExplicitSkillsRuntimeCompleted?.hasDoneText === true) &&
      summary.guiExplicitSkillsRuntimeCompleted?.textareaVisible === true &&
      summary.guiExplicitSkillsRuntimeCompleted?.textareaDisabled === false &&
      summary.guiExplicitSkillsRuntimeCompleted?.stopButtonVisible === false,
    readModelExplicitSkillsRuntimeCompleted:
      summary.readModelExplicitSkillsRuntimeCompleted?.includesPrompt ===
        true &&
      (summary.readModelExplicitSkillsRuntimeCompleted
        ?.includesAssistantDone === true ||
        summary.readModelExplicitSkillsRuntimeCompleted
          ?.includesAssistantSummary === true),
    readModelExplicitSkillSearchObserved:
      summary.readModelExplicitSkillsRuntimeCompleted
        ?.includesSkillSearchTool === true,
    readModelExplicitSkillInvocationObserved:
      summary.readModelExplicitSkillsRuntimeCompleted?.includesSkillTool ===
        true &&
      summary.readModelExplicitSkillsRuntimeCompleted?.includesSkillName ===
        true,
    readModelExplicitSkillBodyReadObserved:
      skillRuntimeEvidence(summary.readModelExplicitSkillsRuntime)
        .skillBodyReadObserved === true,
    readModelExplicitSkillGateObserved:
      skillRuntimeEvidence(summary.readModelExplicitSkillsRuntime)
        .skillGateObserved === true,
    readModelExplicitSkillSearchObserved:
      summary.readModelExplicitSkillsRuntime?.hasSkillSearchSummary ===
        true &&
      summary.readModelExplicitSkillsRuntime?.searchQuery ===
        SKILLS_RUNTIME_QUERY,
    readModelExplicitSkillInvocationObserved:
      summary.readModelExplicitSkillsRuntime?.hasSkillInvocationSummary ===
        true &&
      summary.readModelExplicitSkillsRuntime?.invocationSkillName ===
        SKILLS_RUNTIME_SKILL_NAME,
    explicitSkillSearchBeforeSkillInvocation:
      summary.readModelExplicitSkillsRuntime
        ?.skillSearchBeforeSkillInvocation === true,
    manualEnableSkillsRuntimePromptReachedBackend:
      manualEnableSkillsRuntimeTurnStart?.inputText ===
      SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
    manualEnableSkillsRuntimeMetadataReachedBackend:
      manualEnableRuntimeMetadata?.source === "manual_session_enable" &&
      manualEnableRuntimeMetadata?.approval === "manual" &&
      manualEnableRuntimeMetadata?.workspace_root === workspace.rootPath &&
      manualEnableRuntimeBinding?.directory === "capability-report" &&
      manualEnableRuntimeBinding?.skill === SKILLS_RUNTIME_SKILL_NAME &&
      manualEnableRuntimeBinding?.registered_skill_directory ===
        summary.manualEnableSkillsRuntimeSkill?.skillDirectory &&
      manualEnableRuntimeBinding?.source_draft_id ===
        "capdraft-fixture-capability-report" &&
      manualEnableRuntimeBinding?.source_verification_report_id ===
        "capver-fixture-capability-report",
    manualEnableSkillsRuntimeLaunchedFromSkillsWorkspace:
      summary.manualEnableSkillsRuntimeTurnStart?.launch?.clicked === true &&
      summary.manualEnableSkillsRuntimeTurnStart?.launch
        ?.registeredPanelVisible === true &&
      summary.manualEnableSkillsRuntimeTurnStart?.launch
        ?.enableButtonVisible === true &&
      summary.manualEnableSkillsRuntimeTurnStart?.launch
        ?.enableButtonDisabled === false,
    manualEnableSkillsRuntimeUsedAgentSession:
      typeof summary.manualEnableSkillsRuntimeTurnStart?.backend?.sessionId ===
        "string" &&
      summary.manualEnableSkillsRuntimeTurnStart.backend.sessionId.length > 0 &&
      typeof summary.manualEnableSkillsRuntimeTurnStart?.backend?.turnId ===
        "string" &&
      summary.manualEnableSkillsRuntimeTurnStart.backend.turnId.length > 0,
    manualEnableSkillsRuntimeSkillDirectoryPrepared:
      typeof summary.manualEnableSkillsRuntimeSkill?.skillFilePath ===
        "string" &&
      fs.existsSync(summary.manualEnableSkillsRuntimeSkill.skillFilePath),
    guiManualEnableSkillsRuntimeCompleted:
      summary.guiManualEnableSkillsRuntimeCompleted?.hasPrompt === true &&
      (summary.guiManualEnableSkillsRuntimeCompleted?.hasAssistantSummary ===
        true ||
        summary.guiManualEnableSkillsRuntimeCompleted?.hasDoneText === true) &&
      summary.guiManualEnableSkillsRuntimeCompleted?.textareaVisible === true &&
      summary.guiManualEnableSkillsRuntimeCompleted?.textareaDisabled ===
        false &&
      summary.guiManualEnableSkillsRuntimeCompleted?.stopButtonVisible ===
        false,
    readModelManualEnableSkillsRuntimeCompleted:
      summary.readModelManualEnableSkillsRuntimeCompleted?.includesPrompt ===
        true &&
      (summary.readModelManualEnableSkillsRuntimeCompleted
        ?.includesAssistantDone === true ||
        summary.readModelManualEnableSkillsRuntimeCompleted
          ?.includesAssistantSummary === true),
    readModelManualEnableSkillSearchObserved:
      summary.readModelManualEnableSkillsRuntimeCompleted
        ?.includesSkillSearchTool === true,
    readModelManualEnableSkillInvocationObserved:
      summary.readModelManualEnableSkillsRuntimeCompleted?.includesSkillTool ===
        true &&
      summary.readModelManualEnableSkillsRuntimeCompleted?.includesSkillName ===
        true,
    readModelManualEnableSkillBodyReadObserved:
      skillRuntimeEvidence(summary.readModelManualEnableSkillsRuntime)
        .skillBodyReadObserved === true,
    readModelManualEnableSkillGateObserved:
      skillRuntimeEvidence(summary.readModelManualEnableSkillsRuntime)
        .skillGateObserved === true &&
      skillRuntimeEvidence(summary.readModelManualEnableSkillsRuntime)
        .skillGateMode === "workspace_runtime_enable",
    readModelManualEnableWorkspaceRuntimeEnableObserved:
      skillRuntimeEvidence(summary.readModelManualEnableSkillsRuntime)
        .skillGateWorkspaceRuntimeEnable === true &&
      skillRuntimeEvidence(summary.readModelManualEnableSkillsRuntime)
        .skillGateSourceAllowlist?.includes(SKILLS_RUNTIME_SKILL_NAME) === true,
    readModelManualEnableSkillSearchObserved:
      summary.readModelManualEnableSkillsRuntime?.hasSkillSearchSummary ===
        true &&
      summary.readModelManualEnableSkillsRuntime?.searchQuery ===
        SKILLS_RUNTIME_QUERY,
    readModelManualEnableSkillInvocationObserved:
      summary.readModelManualEnableSkillsRuntime
        ?.hasSkillInvocationSummary === true &&
      summary.readModelManualEnableSkillsRuntime?.invocationSkillName ===
        SKILLS_RUNTIME_SKILL_NAME,
    manualEnableSkillSearchBeforeSkillInvocation:
      summary.readModelManualEnableSkillsRuntime
        ?.skillSearchBeforeSkillInvocation === true,
  };
}

export function buildExpertSkillsRuntimeScenarioAssertions({
  expectedExpertHarnessSkillRef,
  expertHarnessMetadata,
  expertHarnessSkillRefs,
  expertPanelSkillsRuntimeTurnStart,
  expertRuntimeMetadata,
  expertSkillsRuntimeTurnStart,
  isExpertPanelSkillsRuntimeScenario,
  isExpertPlazaSkillsRuntimeScenario,
  summary,
}) {
  return {
    ...(isExpertPanelSkillsRuntimeScenario
      ? {}
      : {
          expertSkillsRuntimePromptReachedBackend:
            expertSkillsRuntimeTurnStart?.inputText?.includes(
              EXPERT_SKILLS_RUNTIME_PROMPT,
            ) === true,
          expertSkillsRuntimeMetadataReachedBackend:
            (expertRuntimeMetadata?.expertId === EXPERT_SKILLS_RUNTIME_ID ||
              expertRuntimeMetadata?.expert_id === EXPERT_SKILLS_RUNTIME_ID) &&
            (expertHarnessMetadata?.expert_id === EXPERT_SKILLS_RUNTIME_ID ||
              expertHarnessMetadata?.expertId === EXPERT_SKILLS_RUNTIME_ID) &&
            expertHarnessSkillRefs.includes(expectedExpertHarnessSkillRef) ===
              true,
          expertDeclaredSkillRefsObserved:
            skillRuntimeEvidence(summary.readModelExpertSkillsRuntime)
              .expertDeclaredObserved === true &&
            skillRuntimeEvidence(
              summary.readModelExpertSkillsRuntime,
            ).expertDeclaredSkillRefs?.includes(
              EXPERT_SKILLS_RUNTIME_SKILL_REF,
            ) === true,
          expertSelectedSkillObserved:
            skillRuntimeEvidence(summary.readModelExpertSkillsRuntime)
              .expertSelectedObserved === true &&
            skillRuntimeEvidence(summary.readModelExpertSkillsRuntime)
              .expertSelectedSkill === SKILLS_RUNTIME_SKILL_NAME,
          expertInvokedSkillObserved:
            skillRuntimeEvidence(summary.readModelExpertSkillsRuntime)
              .expertInvokedObserved === true &&
            skillRuntimeEvidence(summary.readModelExpertSkillsRuntime)
              .expertInvokedSkill === SKILLS_RUNTIME_SKILL_NAME,
          guiExpertSkillsRuntimeSessionVisible:
            summary.guiExpertSkillsRuntimeSessionVisible?.hasSessionTitle ===
              true ||
            summary.guiExpertSkillsRuntimeCompleted?.bodyText?.includes(
              EXPERT_SKILLS_RUNTIME_SESSION_TITLE,
            ) === true ||
            summary.guiExpertSkillsRuntimeCompleted?.bodyText?.includes(
              EXPERT_SKILLS_RUNTIME_TITLE,
            ) === true,
          readModelExpertSkillsRuntimeCompleted:
            summary.readModelExpertSkillsRuntimeCompleted?.includesPrompt ===
              true &&
            (summary.readModelExpertSkillsRuntimeCompleted
              ?.includesAssistantDone === true ||
              summary.readModelExpertSkillsRuntimeCompleted
                ?.includesAssistantSummary === true),
          readModelExpertSkillSearchObserved:
            summary.readModelExpertSkillsRuntimeCompleted
              ?.includesSkillSearchTool === true,
          readModelExpertSkillInvocationObserved:
            summary.readModelExpertSkillsRuntimeCompleted?.includesSkillTool ===
              true &&
            summary.readModelExpertSkillsRuntimeCompleted?.includesSkillName ===
              true,
          readModelExpertSkillBodyReadObserved:
            skillRuntimeEvidence(summary.readModelExpertSkillsRuntime)
              .skillBodyReadObserved === true,
          readModelExpertSkillGateObserved:
            skillRuntimeEvidence(summary.readModelExpertSkillsRuntime)
              .skillGateObserved === true &&
            skillRuntimeEvidence(summary.readModelExpertSkillsRuntime)
              .skillGateMode === "selected_skills",
          readModelExpertSkillSearchObserved:
            summary.readModelExpertSkillsRuntime?.hasSkillSearchSummary ===
              true &&
            summary.readModelExpertSkillsRuntime?.searchQuery ===
              SKILLS_RUNTIME_QUERY,
          readModelExpertSkillInvocationObserved:
            summary.readModelExpertSkillsRuntime
              ?.hasSkillInvocationSummary === true &&
            summary.readModelExpertSkillsRuntime?.invocationSkillName ===
              SKILLS_RUNTIME_SKILL_NAME,
          expertSkillSearchBeforeSkillInvocation:
            summary.readModelExpertSkillsRuntime
              ?.skillSearchBeforeSkillInvocation === true,
        }),
    ...(isExpertPlazaSkillsRuntimeScenario || isExpertPanelSkillsRuntimeScenario
      ? {
          expertPlazaCatalogInjected:
            summary.expertPlazaSkillsRuntimeCatalog?.expertId ===
              EXPERT_SKILLS_RUNTIME_ID &&
            summary.expertPlazaSkillsRuntimeCatalog?.skillRefs?.includes(
              isExpertPanelSkillsRuntimeScenario
                ? EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF
                : EXPERT_SKILLS_RUNTIME_SKILL_REF,
            ) === true &&
            summary.expertPlazaSkillsRuntimeCatalog?.promptStarter ===
              EXPERT_SKILLS_RUNTIME_PROMPT,
          expertPlazaCardClicked:
            summary.expertPlazaSkillsRuntimeLaunch?.clicked === true &&
            summary.expertPlazaSkillsRuntimeLaunch?.plazaVisible === true &&
            summary.expertPlazaSkillsRuntimeLaunch?.cardVisible === true &&
            summary.expertPlazaSkillsRuntimeLaunch?.startButtonVisible === true,
          expertPlazaAutoSendTurnStarted:
            typeof summary.expertSkillsRuntimeTurnStart?.sessionId ===
              "string" &&
            summary.expertSkillsRuntimeTurnStart.sessionId.length > 0 &&
            summary.expertSkillsRuntimeTurnStart?.inputText?.includes(
              EXPERT_SKILLS_RUNTIME_PROMPT,
            ) === true,
        }
      : {}),
    ...(isExpertPanelSkillsRuntimeScenario
      ? {
          expertPanelSkillPickerOpened:
            summary.expertPanelSkillsRuntimeAddSkill?.pickerOpened
              ?.dialogVisible === true,
          expertPanelSkillAdded:
            summary.expertPanelSkillsRuntimeAddSkill?.candidate
              ?.addButtonVisible === true &&
            summary.expertPanelSkillsRuntimeAddSkill?.candidate
              ?.addButtonDisabled === false,
          expertPanelAddedSkillVisible:
            summary.expertPanelSkillsRuntimeAddSkill?.added
              ?.baseSkillVisible === true &&
            summary.expertPanelSkillsRuntimeAddSkill?.added
              ?.addedSkillVisible === true,
          expertPanelSecondTurnPromptReachedBackend:
            expertPanelSkillsRuntimeTurnStart?.inputText ===
            EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
          expertPanelSkillRefsOverrideReachedBackend:
            skillRuntimeEvidence(summary.readModelExpertPanelSkillsRuntime)
              .expertDeclaredSkillRefs?.includes(
              EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF,
            ) === true,
          expertPanelReadModelCompleted:
            summary.readModelExpertPanelSkillsRuntimeCompleted
              ?.includesPrompt === true &&
            (summary.readModelExpertPanelSkillsRuntimeCompleted
              ?.includesAssistantDone === true ||
              summary.readModelExpertPanelSkillsRuntimeCompleted
                ?.includesAssistantSummary === true),
          expertPanelReadModelSkillBodyReadObserved:
            skillRuntimeEvidence(summary.readModelExpertPanelSkillsRuntime)
              .skillBodyReadObserved === true,
          expertPanelReadModelSkillGateObserved:
            skillRuntimeEvidence(summary.readModelExpertPanelSkillsRuntime)
              .skillGateObserved === true &&
            skillRuntimeEvidence(summary.readModelExpertPanelSkillsRuntime)
              .skillGateMode === "selected_skills",
          expertPanelReadModelSkillSearchObserved:
            summary.readModelExpertPanelSkillsRuntime
              ?.hasSkillSearchSummary === true &&
            summary.readModelExpertPanelSkillsRuntime?.searchQuery ===
              SKILLS_RUNTIME_QUERY,
          expertPanelReadModelSkillInvocationObserved:
            summary.readModelExpertPanelSkillsRuntime
              ?.hasSkillInvocationSummary === true &&
            summary.readModelExpertPanelSkillsRuntime
              ?.invocationSkillName === SKILLS_RUNTIME_SKILL_NAME,
          expertPanelSkillSearchBeforeSkillInvocation:
            summary.readModelExpertPanelSkillsRuntime
              ?.skillSearchBeforeSkillInvocation === true,
        }
      : {}),
  };
}
