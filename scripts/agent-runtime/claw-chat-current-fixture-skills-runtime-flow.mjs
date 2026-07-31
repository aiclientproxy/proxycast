import {
  EXPERT_PANEL_SKILLS_RUNTIME_SCENARIO,
  EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF,
  EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  EXPERT_SKILLS_RUNTIME_PROMPT,
  EXPERT_SKILLS_RUNTIME_SCENARIO,
  EXPERT_SKILLS_RUNTIME_SESSION_ID,
  SESSION_ID,
  SKILLS_RUNTIME_EXPLICIT_PROMPT,
  SKILLS_RUNTIME_EXPLICIT_SCENARIO,
  SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
  SKILLS_RUNTIME_MANUAL_ENABLE_SCENARIO,
  SKILLS_RUNTIME_PROMPT,
  SKILLS_RUNTIME_SCENARIO,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  addExpertSkillsRuntimeSkillFromInfoPanel,
  exportExpertPanelEvidencePackFromHarnessPanel,
  launchExpertSkillsRuntimeFromExpertPlaza,
  reloadRendererAfterExpertPanelSkillCatalogInjection,
  selectExpertPanelSkillsRuntimeSessionId,
  summarizeExpertPanelSkillsRuntimeTurnStart,
} from "./claw-chat-current-fixture-expert-actions.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import { waitForGuiSkillsRuntimeCompleted } from "./claw-chat-current-fixture-gui-completion-waits.mjs";
import {
  injectExpertSkillsRuntimeCatalog,
  openSessionFromSidebar,
} from "./claw-chat-current-fixture-session.mjs";
import {
  ensureManualEnableWorkspaceSkill,
  ensureUserVisibleCapabilityReportSkill,
  launchSkillsRuntimeFromWorkspacePanel,
  verifySkillsChangedCatalogRefresh,
  waitForExpertPanelSkillsRuntimeSessionReady,
} from "./claw-chat-current-fixture-skills-workspace.mjs";
import {
  clearInvokeBuffers,
  waitForRendererReady,
} from "./claw-chat-current-fixture-rpc.mjs";
import {
  exportSkillsRuntimeEvidencePack,
  waitForCanonicalThreadIdBySessionId,
  waitForSessionReadSkillsRuntimeCompleted,
} from "./claw-chat-current-fixture-read-model-waits.mjs";
import {
  waitForBackendLedgerTurnStart,
  waitForBackendLedgerTurnStartContaining,
} from "./claw-chat-current-fixture-backend-ledger.mjs";
import { sanitizeJson } from "./claw-chat-current-fixture-utils.mjs";

export async function runSkillsRuntimeScenario({
  page,
  options,
  workspace,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  const result = {};

  logStage("verify-skills-changed-catalog-refresh");
  result.skillsChangedCatalogRefresh = await verifySkillsChangedCatalogRefresh(
    page,
    options,
    runtimeEnv,
  );

  logStage("send-skills-runtime-prompt-from-gui");
  result.skillsRuntimeInputSend = sanitizeJson(
    await sendPromptFromGui(page, options, SKILLS_RUNTIME_PROMPT),
  );

  logStage("wait-gui-skills-runtime-completed");
  result.guiSkillsRuntimeCompleted = sanitizeJson(
    await waitForGuiSkillsRuntimeCompleted(page, options),
  );

  logStage("wait-read-model-skills-runtime-completed");
  const readModelSkillsRuntimeCompleted =
    await waitForSessionReadSkillsRuntimeCompleted(
      page,
      options,
      appServerRequests,
    );
  result.readModelSkillsRuntimeCompleted =
    readModelSkillsRuntimeCompleted.summary;

  logStage("export-skills-runtime-evidence-pack");
  const evidencePackSkillsRuntime = await exportSkillsRuntimeEvidencePack(
    page,
    appServerRequests,
    SKILLS_RUNTIME_SCENARIO,
    options.sessionId,
  );
  result.evidencePackSkillsRuntime = evidencePackSkillsRuntime.summary;

  logStage("send-explicit-skills-runtime-prompt-from-gui");
  result.explicitSkillsRuntimeInputSend = sanitizeJson(
    await sendPromptFromGui(page, options, SKILLS_RUNTIME_EXPLICIT_PROMPT),
  );

  logStage("wait-gui-explicit-skills-runtime-completed");
  result.guiExplicitSkillsRuntimeCompleted = sanitizeJson(
    await waitForGuiSkillsRuntimeCompleted(
      page,
      options,
      SKILLS_RUNTIME_EXPLICIT_SCENARIO,
    ),
  );

  logStage("wait-read-model-explicit-skills-runtime-completed");
  const readModelExplicitSkillsRuntimeCompleted =
    await waitForSessionReadSkillsRuntimeCompleted(
      page,
      options,
      appServerRequests,
      SKILLS_RUNTIME_EXPLICIT_SCENARIO,
    );
  result.readModelExplicitSkillsRuntimeCompleted =
    readModelExplicitSkillsRuntimeCompleted.summary;

  logStage("export-explicit-skills-runtime-evidence-pack");
  const evidencePackExplicitSkillsRuntime =
    await exportSkillsRuntimeEvidencePack(
      page,
      appServerRequests,
      SKILLS_RUNTIME_EXPLICIT_SCENARIO,
      options.sessionId,
    );
  result.evidencePackExplicitSkillsRuntime =
    evidencePackExplicitSkillsRuntime.summary;

  logStage("launch-manual-enable-skills-runtime-from-workspace-panel");
  const manualEnableSkillsRuntimeLaunch =
    await launchSkillsRuntimeFromWorkspacePanel(page, options, workspace);
  result.manualEnableSkillsRuntimeTurnStart = sanitizeJson({
    launch: manualEnableSkillsRuntimeLaunch,
  });
  result.manualEnableSkillsRuntimeSkill =
    manualEnableSkillsRuntimeLaunch.workspaceSkill;

  logStage("wait-manual-enable-skills-runtime-backend-turn-start");
  const manualEnableSkillsRuntimeBackendTurn =
    await waitForBackendLedgerTurnStart(
      runtimeEnv.backendLedgerPath,
      SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
      options,
    );
  const manualEnableSkillsRuntimeSessionId =
    manualEnableSkillsRuntimeBackendTurn.entry.sessionId ?? SESSION_ID;
  const manualEnableSkillsRuntimeThreadId =
    await waitForCanonicalThreadIdBySessionId(
      page,
      options,
      appServerRequests,
      manualEnableSkillsRuntimeSessionId,
    );
  result.manualEnableSkillsRuntimeTurnStart = sanitizeJson({
    ...result.manualEnableSkillsRuntimeTurnStart,
    backend: {
      sessionId: manualEnableSkillsRuntimeSessionId,
      turnId: manualEnableSkillsRuntimeBackendTurn.entry.turnId ?? null,
      inputText: manualEnableSkillsRuntimeBackendTurn.entry.inputText ?? null,
    },
  });

  logStage("wait-gui-manual-enable-skills-runtime-completed");
  result.guiManualEnableSkillsRuntimeCompleted = sanitizeJson(
    await waitForGuiSkillsRuntimeCompleted(
      page,
      options,
      SKILLS_RUNTIME_MANUAL_ENABLE_SCENARIO,
    ),
  );

  logStage("wait-read-model-manual-enable-skills-runtime-completed");
  const readModelManualEnableSkillsRuntimeCompleted =
    await waitForSessionReadSkillsRuntimeCompleted(
      page,
      options,
      appServerRequests,
      SKILLS_RUNTIME_MANUAL_ENABLE_SCENARIO,
      manualEnableSkillsRuntimeThreadId,
    );
  result.readModelManualEnableSkillsRuntimeCompleted =
    readModelManualEnableSkillsRuntimeCompleted.summary;

  logStage("export-manual-enable-skills-runtime-evidence-pack");
  const evidencePackManualEnableSkillsRuntime =
    await exportSkillsRuntimeEvidencePack(
      page,
      appServerRequests,
      SKILLS_RUNTIME_MANUAL_ENABLE_SCENARIO,
      manualEnableSkillsRuntimeSessionId,
    );
  result.evidencePackManualEnableSkillsRuntime =
    evidencePackManualEnableSkillsRuntime.summary;

  return result;
}

export async function runExpertSkillsRuntimeScenario({
  page,
  options,
  workspace,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  return runPlazaOrPanelExpertSkillsRuntimeScenario({
    page,
    options,
    workspace,
    appServerRequests,
    runtimeEnv,
    logStage,
  });
}

async function runPlazaOrPanelExpertSkillsRuntimeScenario({
  page,
  options,
  workspace,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  const result = {};
  const isExpertPanelSkillsRuntimeScenario =
    options.scenario === "expert-panel-skills-runtime";

  logStage(
    isExpertPanelSkillsRuntimeScenario
      ? "prepare-expert-panel-skills-runtime-workspace-skill"
      : "prepare-expert-plaza-skills-runtime-workspace-skill",
  );
  const expertSkillsRuntimeSkill = ensureManualEnableWorkspaceSkill(
    workspace.rootPath,
  );
  result.expertSkillsRuntimeSkill = sanitizeJson(expertSkillsRuntimeSkill);
  if (isExpertPanelSkillsRuntimeScenario) {
    result.expertPanelSkillsRuntimeUserSkill = sanitizeJson(
      ensureUserVisibleCapabilityReportSkill(runtimeEnv),
    );
  }

  logStage(
    isExpertPanelSkillsRuntimeScenario
      ? "inject-expert-panel-skills-runtime-catalog"
      : "inject-expert-plaza-skills-runtime-catalog",
  );
  result.expertPlazaSkillsRuntimeCatalog = sanitizeJson(
    await injectExpertSkillsRuntimeCatalog(page, {
      ...(isExpertPanelSkillsRuntimeScenario
        ? {
            releaseSkillRefs: [EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF],
          }
        : {}),
      workspaceSkill: expertSkillsRuntimeSkill,
    }),
  );
  if (isExpertPanelSkillsRuntimeScenario) {
    logStage("reload-expert-panel-skills-runtime-catalog");
    result.expertPanelSkillsRuntimeCatalogReload = sanitizeJson(
      await reloadRendererAfterExpertPanelSkillCatalogInjection(
        page,
        options,
        waitForRendererReady,
        clearInvokeBuffers,
      ),
    );
  }

  logStage("launch-expert-skills-runtime-from-expert-plaza");
  result.expertPlazaSkillsRuntimeLaunch = sanitizeJson(
    await launchExpertSkillsRuntimeFromExpertPlaza(page, options),
  );

  logStage("wait-expert-plaza-skills-runtime-backend-turn-start");
  const expertPlazaSkillsRuntimeBackendTurn =
    await waitForBackendLedgerTurnStartContaining(
      runtimeEnv.backendLedgerPath,
      EXPERT_SKILLS_RUNTIME_PROMPT,
      options,
    );
  const expertPlazaSkillsRuntimeSessionId =
    expertPlazaSkillsRuntimeBackendTurn.entry.sessionId ??
    EXPERT_SKILLS_RUNTIME_SESSION_ID;
  const expertPlazaSkillsRuntimeThreadId =
    await waitForCanonicalThreadIdBySessionId(
      page,
      options,
      appServerRequests,
      expertPlazaSkillsRuntimeSessionId,
    );
  result.expertSkillsRuntimeTurnStart = sanitizeJson({
    sessionId: expertPlazaSkillsRuntimeSessionId,
    turnId: expertPlazaSkillsRuntimeBackendTurn.entry.turnId ?? null,
    inputText: expertPlazaSkillsRuntimeBackendTurn.entry.inputText ?? null,
  });

  logStage("wait-read-model-expert-plaza-skills-runtime-completed");
  const readModelExpertSkillsRuntimeCompleted =
    await waitForSessionReadSkillsRuntimeCompleted(
      page,
      options,
      appServerRequests,
      EXPERT_SKILLS_RUNTIME_SCENARIO,
      expertPlazaSkillsRuntimeThreadId,
    );
  result.readModelExpertSkillsRuntimeCompleted =
    readModelExpertSkillsRuntimeCompleted.summary;

  logStage("export-expert-plaza-skills-runtime-evidence-pack");
  const evidencePackExpertSkillsRuntime = await exportSkillsRuntimeEvidencePack(
    page,
    appServerRequests,
    EXPERT_SKILLS_RUNTIME_SCENARIO,
    expertPlazaSkillsRuntimeSessionId,
  );
  result.evidencePackExpertSkillsRuntime =
    evidencePackExpertSkillsRuntime.summary;

  logStage("wait-gui-expert-plaza-skills-runtime-completed");
  result.guiExpertSkillsRuntimeCompleted = sanitizeJson(
    await waitForGuiSkillsRuntimeCompleted(
      page,
      options,
      EXPERT_SKILLS_RUNTIME_SCENARIO,
      {
        sessionId: expertPlazaSkillsRuntimeSessionId,
        turnId: expertPlazaSkillsRuntimeBackendTurn.entry.turnId,
      },
    ),
  );

  if (isExpertPanelSkillsRuntimeScenario) {
    await runExpertPanelSkillsRuntimeFollowup({
      page,
      options,
      appServerRequests,
      runtimeEnv,
      logStage,
      result,
      expertPlazaSkillsRuntimeSessionId,
    });
  }

  return result;
}

async function runExpertPanelSkillsRuntimeFollowup({
  page,
  options,
  appServerRequests,
  runtimeEnv,
  logStage,
  result,
  expertPlazaSkillsRuntimeSessionId,
}) {
  logStage("add-expert-panel-skills-runtime-skill");
  result.expertPanelSkillsRuntimeAddSkill = sanitizeJson(
    await addExpertSkillsRuntimeSkillFromInfoPanel(page, options),
  );

  logStage("wait-expert-panel-skills-runtime-session-ready");
  result.guiExpertPanelSkillsRuntimeSessionReady = sanitizeJson(
    await waitForExpertPanelSkillsRuntimeSessionReady(
      page,
      options,
      expertPlazaSkillsRuntimeSessionId,
    ),
  );

  const expertPanelReadySnapshot =
    result.guiExpertPanelSkillsRuntimeSessionReady ?? {};
  const expertPanelInputMatchesSession =
    expertPanelReadySnapshot.textareaVisible === true &&
    expertPanelReadySnapshot.textareaDisabled === false &&
    expertPanelReadySnapshot.textareaSessionId ===
      expertPlazaSkillsRuntimeSessionId;
  const expertPanelFallbackInputReady =
    expertPanelReadySnapshot.fallbackTextareaVisible === true &&
    expertPanelReadySnapshot.fallbackTextareaDisabled === false;
  let expertPanelFollowupConstraints = expertPanelInputMatchesSession
    ? { expectedSessionId: expertPlazaSkillsRuntimeSessionId }
    : {};

  if (!expertPanelInputMatchesSession && !expertPanelFallbackInputReady) {
    logStage("reopen-expert-panel-skills-runtime-session");
    result.guiExpertPanelSkillsRuntimeSessionReopened = sanitizeJson(
      await openSessionFromSidebar(page, options, appServerRequests, {
        sessionId: expertPlazaSkillsRuntimeSessionId,
        title: "请以「代码文学专家」专家身份工作。",
      }),
    );
    expertPanelFollowupConstraints = {
      expectedSessionId: expertPlazaSkillsRuntimeSessionId,
    };
  } else {
    result.guiExpertPanelSkillsRuntimeSessionReopened = sanitizeJson({
      skipped: true,
      reason: expertPanelInputMatchesSession
        ? "expert panel session input already ready"
        : "expert panel current page input ready without session marker",
      sessionId: expertPlazaSkillsRuntimeSessionId,
      textareaSessionId:
        expertPanelReadySnapshot.textareaSessionId ??
        expertPanelReadySnapshot.fallbackTextareaSessionId ??
        null,
    });
  }

  logStage("send-expert-panel-skills-runtime-followup");
  result.expertPanelSkillsRuntimeInputSend = sanitizeJson(
    await sendPromptFromGui(
      page,
      options,
      EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
      expertPanelFollowupConstraints,
    ),
  );

  logStage("wait-expert-panel-skills-runtime-backend-turn-start");
  const expertPanelSkillsRuntimeBackendTurn =
    await waitForBackendLedgerTurnStart(
      runtimeEnv.backendLedgerPath,
      EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
      options,
    );
  const expertPanelSkillsRuntimeSessionId =
    selectExpertPanelSkillsRuntimeSessionId(
      expertPanelSkillsRuntimeBackendTurn,
      expertPlazaSkillsRuntimeSessionId,
    );
  const expertPanelSkillsRuntimeThreadId =
    await waitForCanonicalThreadIdBySessionId(
      page,
      options,
      appServerRequests,
      expertPanelSkillsRuntimeSessionId,
    );
  result.expertPanelSkillsRuntimeTurnStart =
    summarizeExpertPanelSkillsRuntimeTurnStart(
      expertPanelSkillsRuntimeBackendTurn,
    );
  result.expertPanelSkillsRuntimeSessionId = expertPanelSkillsRuntimeSessionId;

  logStage("wait-read-model-expert-panel-skills-runtime-completed");
  const readModelExpertPanelSkillsRuntimeCompleted =
    await waitForSessionReadSkillsRuntimeCompleted(
      page,
      options,
      appServerRequests,
      EXPERT_PANEL_SKILLS_RUNTIME_SCENARIO,
      expertPanelSkillsRuntimeThreadId,
    );
  result.readModelExpertPanelSkillsRuntimeCompleted =
    readModelExpertPanelSkillsRuntimeCompleted.summary;

  logStage("export-expert-panel-skills-runtime-evidence-pack");
  const evidencePackExpertPanelSkillsRuntime =
    await exportSkillsRuntimeEvidencePack(
      page,
      appServerRequests,
      EXPERT_PANEL_SKILLS_RUNTIME_SCENARIO,
      expertPanelSkillsRuntimeSessionId,
    );
  result.evidencePackExpertPanelSkillsRuntime =
    evidencePackExpertPanelSkillsRuntime.summary;

  logStage("wait-gui-expert-panel-skills-runtime-completed");
  result.guiExpertPanelSkillsRuntimeCompleted = sanitizeJson(
    await waitForGuiSkillsRuntimeCompleted(
      page,
      options,
      EXPERT_PANEL_SKILLS_RUNTIME_SCENARIO,
      {
        sessionId: expertPanelSkillsRuntimeSessionId,
        turnId: expertPanelSkillsRuntimeBackendTurn.entry.turnId,
      },
    ),
  );

  logStage("export-expert-panel-evidence-pack-from-harness-panel");
  result.expertPanelEvidencePackGuiExport = sanitizeJson(
    await exportExpertPanelEvidencePackFromHarnessPanel(page, options),
  );
}
