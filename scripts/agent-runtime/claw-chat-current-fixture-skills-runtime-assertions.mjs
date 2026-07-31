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
        ?.method === "skill/list" &&
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
        "skill/list" &&
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
    evidenceSkillBodyReadObserved:
      summary.evidencePackSkillsRuntime?.skillBodyReadObserved === true,
    evidenceSkillGateObserved:
      summary.evidencePackSkillsRuntime?.skillGateObserved === true,
    evidencePackSkillSearchObserved:
      summary.evidencePackSkillsRuntime?.hasSkillSearchSummary === true &&
      summary.evidencePackSkillsRuntime?.searchQuery === SKILLS_RUNTIME_QUERY,
    evidencePackSkillInvocationObserved:
      summary.evidencePackSkillsRuntime?.hasSkillInvocationSummary === true &&
      summary.evidencePackSkillsRuntime?.invocationSkillName ===
        SKILLS_RUNTIME_SKILL_NAME,
    skillSearchBeforeSkillInvocation:
      summary.evidencePackSkillsRuntime?.skillSearchBeforeSkillInvocation ===
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
    evidenceExplicitSkillBodyReadObserved:
      summary.evidencePackExplicitSkillsRuntime?.skillBodyReadObserved === true,
    evidenceExplicitSkillGateObserved:
      summary.evidencePackExplicitSkillsRuntime?.skillGateObserved === true,
    evidencePackExplicitSkillSearchObserved:
      summary.evidencePackExplicitSkillsRuntime?.hasSkillSearchSummary ===
        true &&
      summary.evidencePackExplicitSkillsRuntime?.searchQuery ===
        SKILLS_RUNTIME_QUERY,
    evidencePackExplicitSkillInvocationObserved:
      summary.evidencePackExplicitSkillsRuntime?.hasSkillInvocationSummary ===
        true &&
      summary.evidencePackExplicitSkillsRuntime?.invocationSkillName ===
        SKILLS_RUNTIME_SKILL_NAME,
    explicitSkillSearchBeforeSkillInvocation:
      summary.evidencePackExplicitSkillsRuntime
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
    evidenceManualEnableSkillBodyReadObserved:
      summary.evidencePackManualEnableSkillsRuntime?.skillBodyReadObserved ===
      true,
    evidenceManualEnableSkillGateObserved:
      summary.evidencePackManualEnableSkillsRuntime?.skillGateObserved ===
        true &&
      summary.evidencePackManualEnableSkillsRuntime?.skillGateMode ===
        "workspace_runtime_enable",
    evidenceManualEnableWorkspaceRuntimeEnableObserved:
      summary.evidencePackManualEnableSkillsRuntime
        ?.skillGateWorkspaceRuntimeEnable === true &&
      summary.evidencePackManualEnableSkillsRuntime?.skillGateSourceAllowlist?.includes(
        SKILLS_RUNTIME_SKILL_NAME,
      ) === true,
    evidencePackManualEnableSkillSearchObserved:
      summary.evidencePackManualEnableSkillsRuntime?.hasSkillSearchSummary ===
        true &&
      summary.evidencePackManualEnableSkillsRuntime?.searchQuery ===
        SKILLS_RUNTIME_QUERY,
    evidencePackManualEnableSkillInvocationObserved:
      summary.evidencePackManualEnableSkillsRuntime
        ?.hasSkillInvocationSummary === true &&
      summary.evidencePackManualEnableSkillsRuntime?.invocationSkillName ===
        SKILLS_RUNTIME_SKILL_NAME,
    manualEnableSkillSearchBeforeSkillInvocation:
      summary.evidencePackManualEnableSkillsRuntime
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
            summary.evidencePackExpertSkillsRuntime?.expertDeclaredObserved ===
              true &&
            summary.evidencePackExpertSkillsRuntime?.expertDeclaredSkillRefs?.includes(
              EXPERT_SKILLS_RUNTIME_SKILL_REF,
            ) === true,
          expertSelectedSkillObserved:
            summary.evidencePackExpertSkillsRuntime?.expertSelectedObserved ===
              true &&
            summary.evidencePackExpertSkillsRuntime?.expertSelectedSkill ===
              SKILLS_RUNTIME_SKILL_NAME,
          expertInvokedSkillObserved:
            summary.evidencePackExpertSkillsRuntime?.expertInvokedObserved ===
              true &&
            summary.evidencePackExpertSkillsRuntime?.expertInvokedSkill ===
              SKILLS_RUNTIME_SKILL_NAME,
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
          evidenceExpertSkillBodyReadObserved:
            summary.evidencePackExpertSkillsRuntime?.skillBodyReadObserved ===
            true,
          evidenceExpertSkillGateObserved:
            summary.evidencePackExpertSkillsRuntime?.skillGateObserved ===
              true &&
            summary.evidencePackExpertSkillsRuntime?.skillGateMode ===
              "selected_skills",
          evidencePackExpertSkillSearchObserved:
            summary.evidencePackExpertSkillsRuntime?.hasSkillSearchSummary ===
              true &&
            summary.evidencePackExpertSkillsRuntime?.searchQuery ===
              SKILLS_RUNTIME_QUERY,
          evidencePackExpertSkillInvocationObserved:
            summary.evidencePackExpertSkillsRuntime
              ?.hasSkillInvocationSummary === true &&
            summary.evidencePackExpertSkillsRuntime?.invocationSkillName ===
              SKILLS_RUNTIME_SKILL_NAME,
          expertSkillSearchBeforeSkillInvocation:
            summary.evidencePackExpertSkillsRuntime
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
            summary.evidencePackExpertPanelSkillsRuntime?.expertDeclaredSkillRefs?.includes(
              EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF,
            ) === true,
          expertPanelReadModelCompleted:
            summary.readModelExpertPanelSkillsRuntimeCompleted
              ?.includesPrompt === true &&
            (summary.readModelExpertPanelSkillsRuntimeCompleted
              ?.includesAssistantDone === true ||
              summary.readModelExpertPanelSkillsRuntimeCompleted
                ?.includesAssistantSummary === true),
          expertPanelEvidenceSkillBodyReadObserved:
            summary.evidencePackExpertPanelSkillsRuntime
              ?.skillBodyReadObserved === true,
          expertPanelEvidenceSkillGateObserved:
            summary.evidencePackExpertPanelSkillsRuntime?.skillGateObserved ===
              true &&
            summary.evidencePackExpertPanelSkillsRuntime?.skillGateMode ===
              "selected_skills",
          expertPanelEvidenceSkillSearchObserved:
            summary.evidencePackExpertPanelSkillsRuntime
              ?.hasSkillSearchSummary === true &&
            summary.evidencePackExpertPanelSkillsRuntime?.searchQuery ===
              SKILLS_RUNTIME_QUERY,
          expertPanelEvidenceSkillInvocationObserved:
            summary.evidencePackExpertPanelSkillsRuntime
              ?.hasSkillInvocationSummary === true &&
            summary.evidencePackExpertPanelSkillsRuntime
              ?.invocationSkillName === SKILLS_RUNTIME_SKILL_NAME,
          expertPanelSkillSearchBeforeSkillInvocation:
            summary.evidencePackExpertPanelSkillsRuntime
              ?.skillSearchBeforeSkillInvocation === true,
          expertPanelEvidencePackExportedFromHarnessPanel:
            summary.expertPanelEvidencePackGuiExport?.clicked?.clicked ===
              true &&
            summary.expertPanelEvidencePackGuiExport?.exported
              ?.hasExportedPack === true,
        }
      : {}),
  };
}
