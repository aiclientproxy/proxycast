import fs from "node:fs";
import {
  createExpertSkillsRuntimeFixtureScenario,
  createManualEnableSkillsRuntimeFixtureScenario,
  createSkillsRuntimeFixtureScenario,
  EXPERT_PANEL_SKILLS_RUNTIME_ASSERTION_KEYS,
  EXPERT_PLAZA_SKILLS_RUNTIME_ASSERTION_KEYS,
  EXPERT_SKILLS_RUNTIME_DONE_TEXT,
  EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT,
  EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
  EXPERT_SKILLS_RUNTIME_PROMPT,
  EXPERT_SKILLS_RUNTIME_SKILL_REF,
  SKILLS_RUNTIME_EXPLICIT_DONE_TEXT,
  SKILLS_RUNTIME_EXPLICIT_PROMPT,
  SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT,
  SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
  SKILLS_RUNTIME_QUERY,
  SKILLS_RUNTIME_SKILL_NAME,
  summarizeSkillsRuntimeEvidenceExport,
} from "./skills-runtime-fixture-scenario.mjs";

function readSkillsRuntimeFixtureScenario() {
  return fs.readFileSync(
    "scripts/agent-runtime/skills-runtime-fixture-scenario.mjs",
    "utf8",
  );
}

function readSessionScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-session.mjs",
    "utf8",
  );
}

function readBackendScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-backend-script.mjs",
    "utf8",
  );
}

function expectAllToContain(expect, content, fragments) {
  for (const fragment of fragments) expect(content).toContain(fragment);
}

function expectAllNotToContain(expect, content, fragments) {
  for (const fragment of fragments) expect(content).not.toContain(fragment);
}

function canonicalToolCompletedEvent(
  toolCallId,
  { eventTypeKey = "type", callIdKey = "call_id" } = {},
) {
  return {
    [eventTypeKey]: "item.completed",
    payload: {
      item: {
        itemId: toolCallId,
        kind: "tool",
        status: "completed",
        payload: {
          type: "tool",
          [callIdKey]: toolCallId,
        },
        metadata: {},
      },
    },
  };
}

export function registerSkillsRuntimeSmokeGuards({
  expect,
  it,
  readSmokeScript,
  readCurrentFixtureRegressionSmokeScript,
  readExpertActionsScript,
  readGuiActionsScript,
}) {
  it("covers Skills runtime search, on-demand body load, gate, and Evidence Pack in the real Electron fixture", () => {
    const content = readSmokeScript();
    const backendScriptContent = readBackendScript();
    const scenarioContent = readSkillsRuntimeFixtureScenario();
    const expertActionsContent = readExpertActionsScript();
    const sessionContent = readSessionScript();
    const guiActionsContent = readGuiActionsScript();
    const expertRuntimeContent = `${content}\n${expertActionsContent}\n${guiActionsContent}`;

    expectAllToContain(expect, content, [
      "skills-runtime",
      "skills-runtime-fixture-scenario.mjs",
      "createSkillsRuntimeFixtureScenario",
      "renderSkillsRuntimeBackendEvents",
      "SKILLS_RUNTIME_PROMPT",
      "SKILLS_RUNTIME_DONE_TEXT",
      "SKILLS_RUNTIME_EXPLICIT_PROMPT",
      "SKILLS_RUNTIME_EXPLICIT_DONE_TEXT",
      "SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT",
      "SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT",
      "expert-plaza-skills-runtime",
      "expert-panel-skills-runtime",
      'options.scenario !== "expert-panel-skills-runtime"',
      "createExpertSkillsRuntimeFixtureScenario",
      "createExpertPanelSkillsRuntimeFixtureScenario",
      "buildExpertSkillsRuntimeMetadata",
      "buildExpertSkillsRuntimeCatalog",
      "EXPERT_SKILLS_RUNTIME_ASSERTION_KEYS",
      "EXPERT_PLAZA_SKILLS_RUNTIME_ASSERTION_KEYS",
      "EXPERT_PANEL_SKILLS_RUNTIME_ASSERTION_KEYS",
      "EXPERT_SKILLS_RUNTIME_PROMPT",
      "EXPERT_SKILLS_RUNTIME_PANEL_PROMPT",
      "EXPERT_SKILLS_RUNTIME_DONE_TEXT",
      "EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT",
      "EXPERT_SKILLS_RUNTIME_SKILL_REF",
      "EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF",
      "injectExpertSkillsRuntimeCatalog",
      "buildExpertPanelWorkspaceSkillCatalog",
      "expertPanelSkillsRuntimeCatalogReload",
      "reload-expert-panel-skills-runtime-catalog",
      "workspaceSkillCatalog",
      "workspaceSkill: expertSkillsRuntimeSkill",
      '"native_skill"',
      '"skill:capability-report"',
      "selectExpertPanelSkillsRuntimeSessionId",
      "result.expertPanelSkillsRuntimeSessionId",
      "expertPanelSkillsRuntimeSessionId",
      "reopen-expert-panel-skills-runtime-session",
      "guiExpertPanelSkillsRuntimeSessionReopened",
      "openSessionFromSidebar(page, options, appServerRequests",
      "expectedSessionId",
      "{ expectedSessionId: expertPlazaSkillsRuntimeSessionId }",
      "expertPlazaCatalogInjected",
      "expertPlazaCardClicked",
      "expertPlazaAutoSendTurnStarted",
      "expertPanelSkillPickerOpened",
      "expertPanelSkillAdded",
      "expertPanelAddedSkillVisible",
      "expertPanelEvidencePackGuiExport",
      "expertPanelEvidencePackExportedFromHarnessPanel",
      "expertPanelSkillRefsOverrideReachedBackend",
      "waitForBackendLedgerTurnStartContaining",
      "launchSkillsRuntimeFromWorkspacePanel",
      "createExpertSkillsRuntimeSession",
      "expertSkillsRuntimeTurnStart",
      "expertPanelSkillsRuntimeTurnStart",
      "{ title }",
      "waitForBackendLedgerTurnStart",
      "manualEnableSkillsRuntimeSessionId",
      "ensureManualEnableWorkspaceSkill",
      '".lime"',
      '"registration.json"',
      "workspace-registered-skill-enable-runtime",
      "app-sidebar-nav-plugins",
      "plugin-workspace-tab-skills",
      "sanitizeBackendLedgerForEvidence",
      "isIgnorableConsoleError",
      "actionableConsoleErrors",
      "workspaceSkillRuntimeEnable",
      "SKILLS_RUNTIME_QUERY",
      "SKILLS_RUNTIME_SKILL_NAME",
      '"evidence/export"',
      "includeEvidencePack: true",
      "waitForGuiSkillsRuntimeCompleted",
      "verifySkillsChangedCatalogRefresh",
      "verify-skills-changed-catalog-refresh",
      "inputbar-plus-trigger",
      "inputbar-plus-skills",
      "inputbar-plus-panel-skills",
      "skill-selector-inline",
      "skill-selector-refresh",
      '"skills/list"',
      '"skills/changed"',
      '"skillsChanged.received"',
      "initialCurrentSkillListObserved",
      "typedSkillsChangedConsumed",
      "automaticSkillListRefreshObserved",
      "guiSkillCatalogUpdated",
      "skillsCatalogRefreshDidNotUseManualRefresh",
      'transport: "electron-ipc"',
      "scenario.guiSummaryText ?? scenario.summaryText",
      "waitForSessionReadSkillsRuntimeCompleted",
      "summarizeSkillsRuntimeReadModel",
      "readModelTurnTerminal",
      "exportSkillsRuntimeEvidencePack",
      "summarizeSkillsRuntimeEvidenceExport",
      "skillsRuntimePromptReachedBackend",
      "readModelSkillSearchObserved",
      "readModelSkillInvocationObserved",
      "evidenceSkillBodyReadObserved",
      "evidenceSkillGateObserved",
      "evidencePackSkillSearchObserved",
      "evidencePackSkillInvocationObserved",
      "skillSearchBeforeSkillInvocation",
      "explicitSkillsRuntimePromptReachedBackend",
      "guiExplicitSkillsRuntimeInputSubmitted",
      "readModelExplicitSkillSearchObserved",
      "evidenceExplicitSkillBodyReadObserved",
      "explicitSkillSearchBeforeSkillInvocation",
      "manualEnableSkillsRuntimePromptReachedBackend",
      "manualEnableSkillsRuntimeMetadataReachedBackend",
      "manualEnableSkillsRuntimeSkillDirectoryPrepared",
      "manualEnableSkillsRuntimeLaunchedFromSkillsWorkspace",
      "manualEnableSkillsRuntimeUsedAgentSession",
      "expertSkillsRuntimeMetadataReachedBackend",
      "expert_declared_skill_refs",
      "expert_selected_skill",
      "expert_invoked_skill",
      "expertDeclaredSkillRefsObserved",
      "expertSelectedSkillObserved",
      "expertInvokedSkillObserved",
      "evidencePackExpertSkillSearchObserved",
      "evidencePackExpertSkillInvocationObserved",
      "expertSkillSearchBeforeSkillInvocation",
      "guiManualEnableSkillsRuntimeCompleted",
      "readModelManualEnableSkillSearchObserved",
      "evidenceManualEnableWorkspaceRuntimeEnableObserved",
      "manualEnableSkillSearchBeforeSkillInvocation",
      "SKILLS_RUNTIME_ASSERTION_KEYS",
    ]);
    expectAllNotToContain(expect, content, [
      "startExpertSkillsRuntimeTurn",
      "EXPERT_SKILLS_RUNTIME_TURN_ID",
      "async function runManualEnableSkillsRuntimeTurn",
      "agent_runtime_",
      "app-sidebar-nav-skills",
    ]);
    expectAllToContain(expect, expertRuntimeContent, [
      "reloadRendererAfterExpertPanelSkillCatalogInjection",
      "lime:skill-catalog:v1",
      "launchExpertSkillsRuntimeFromExpertPlaza",
      "addExpertSkillsRuntimeSkillFromInfoPanel",
      "EXPERT_PANEL_SKILLS_RUNTIME_UI_SKILL_REF",
      "EXPERT_PANEL_SKILLS_RUNTIME_UI_ADD_TEST_ID",
      "EXPERT_PANEL_SKILLS_RUNTIME_UI_CHIP_TEST_ID",
      "data-session-id",
      "hasAddedSkill",
    ]);
    expectAllToContain(expect, expertActionsContent, [
      "waitForExpertSkillPickerState",
      "clickExpertSkillPickerTrigger",
      "expert-info-skills-runtime-action-skill-code-review",
      "mapping-action",
      "setExpertSkillPickerQuery",
      "pickerSearch",
      "waitForExpertPanelAddedSkill",
      "exportExpertPanelEvidencePackFromHarnessPanel",
      "missing-visible-trigger",
      "visibleElementSnapshot(candidate).visible",
      "导出问题证据包",
      "刷新证据包",
      "app-sidebar-nav-plugins",
      "plugin-workspace-tab-experts",
      "expert-start-${EXPERT_SKILLS_RUNTIME_ID}",
      "expert-info-skills-add",
      "EXPERT_SKILLS_RUNTIME_SKILL_REF",
    ]);
    expect(expertActionsContent).not.toContain("skill:local:capability-report");
    expect(expertActionsContent).not.toContain("agent_runtime_");
    expect(expertActionsContent).not.toContain("app-sidebar-nav-experts");

    expectAllToContain(expect, sessionContent, [
      "lime:skill-catalog-changed",
      'source: "manual_override"',
      "window.__LIME_OEM_CLOUD__?.tenantId",
      "buildExpertPanelWorkspaceSkillCatalog",
      "options.workspaceSkill",
      "tenantId",
      "EXPERT_SKILLS_RUNTIME_TENANT_ID",
    ]);

    expectAllToContain(expect, scenarioContent, [
      "EXPERT_SKILLS_RUNTIME_TENANT_ID",
      "createExplicitSkillsRuntimeFixtureScenario",
      "createManualEnableSkillsRuntimeFixtureScenario",
      "buildManualEnableSkillsRuntimeMetadata",
      "createExpertSkillsRuntimeFixtureScenario",
      "createExpertPanelSkillsRuntimeFixtureScenario",
      "buildExpertSkillsRuntimeMetadata",
      "buildExpertSkillsRuntimeCatalog",
      SKILLS_RUNTIME_EXPLICIT_PROMPT,
      SKILLS_RUNTIME_EXPLICIT_DONE_TEXT,
      SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
      SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT,
      EXPERT_SKILLS_RUNTIME_PROMPT,
      EXPERT_SKILLS_RUNTIME_DONE_TEXT,
      EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
      EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT,
      EXPERT_SKILLS_RUNTIME_SKILL_REF,
      'trigger: "explicit"',
      "explicit skill mention",
      'trigger: "workspace_panel_manual_enable"',
      "launched from Skills workspace panel",
      'gateMode: "workspace_runtime_enable"',
      "sourceAllowlist",
      "searchToolCallId",
      "skillToolCallId",
      'name: "skill_search"',
      'tool_family: "skill_search"',
      "skill_search_query",
      "skill_search_snapshot_skill_count",
      "skill_search_result_count",
      "skillRuntime",
      "skill_body_read",
      "skill_gate_decision",
      'name: "Skill"',
      'tool_family: "skill"',
      "workspace_skill_runtime_enable",
      "expertSkillsRuntime",
      "expert_skills_runtime",
      "guiSummaryText",
      "专家面板新增 Skill 后的下一轮 runtime 证据已完成",
      "expert_declared_skill_refs",
      "expert_selected_skill",
      "expert_invoked_skill",
      "promptStarters",
      "EXPERT_PLAZA_SKILLS_RUNTIME_ASSERTION_KEYS",
      "expertDeclaredObserved",
      "expertSelectedObserved",
      "expertInvokedObserved",
      "export function summarizeSkillsRuntimeEvidenceExport",
    ]);
    for (const assertionKey of EXPERT_PLAZA_SKILLS_RUNTIME_ASSERTION_KEYS) {
      expect(content).toContain(assertionKey);
      expect(scenarioContent).toContain(assertionKey);
    }
    for (const assertionKey of EXPERT_PANEL_SKILLS_RUNTIME_ASSERTION_KEYS) {
      expect(content).toContain(assertionKey);
      expect(scenarioContent).toContain(assertionKey);
    }
    expectAllToContain(expect, scenarioContent, [
      'type: "item.started"',
      'type: "item.completed"',
      "buildCanonicalToolItem({",
    ]);
    expectAllNotToContain(expect, scenarioContent, [
      'type: "tool.started"',
      'type: "tool.result"',
    ]);
    expectAllToContain(expect, backendScriptContent, [
      "if (hasProcessPrelude && initialMessageText.length > 0)",
      'type: "message.completed"',
      '"commentary",',
      "commentaryItemId",
    ]);
    const commentaryCompletedIndex = backendScriptContent.indexOf(
      "if (hasProcessPrelude && initialMessageText.length > 0)",
    );
    const skillEventsIndex = backendScriptContent.indexOf(
      "${renderBackendToolAndSkillEventScript({",
    );
    const finalMessageIndex = backendScriptContent.indexOf(
      'messageDeltaPayload(followupText, "final_answer", finalAnswerItemId)',
      skillEventsIndex,
    );
    expect(commentaryCompletedIndex).toBeGreaterThan(-1);
    expect(skillEventsIndex).toBeGreaterThan(commentaryCompletedIndex);
    expect(finalMessageIndex).toBeGreaterThan(skillEventsIndex);
    expect(scenarioContent).not.toContain("agent_runtime_");
  });

  it("summarizes Skills runtime evidence with mixed camelCase and snake_case fields", () => {
    const scenario = createSkillsRuntimeFixtureScenario(
      "skills-runtime-unit-session",
    );
    const evidenceExportResult = {
      evidencePack: {
        observability_summary: {
          skillSearches: [
            {
              query: SKILLS_RUNTIME_QUERY,
              tool_call_id: scenario.searchToolCallId,
            },
          ],
          skill_invocations: [
            {
              skill_name: SKILLS_RUNTIME_SKILL_NAME,
              toolCallId: scenario.skillToolCallId,
              workspaceSkillRuntimeEnable: {
                source: "manual_session_enable",
                authorization_scope: "session",
              },
            },
          ],
        },
      },
      events: [
        canonicalToolCompletedEvent(scenario.searchToolCallId, {
          eventTypeKey: "event_type",
          callIdKey: "callId",
        }),
        {
          type: "runtime.status",
          payload: {
            metadata: {
              skillRuntime: {
                event: "skill_body_read",
              },
            },
          },
        },
        {
          type: "runtime.status",
          payload: {
            metadata: {
              skill_runtime: {
                event: "skill_gate_decision",
                mode: "selected_skills",
              },
            },
          },
        },
        canonicalToolCompletedEvent(scenario.skillToolCallId, {
          eventTypeKey: "eventType",
        }),
      ],
    };

    expect(
      summarizeSkillsRuntimeEvidenceExport(evidenceExportResult, scenario),
    ).toMatchObject({
      hasEvidencePack: true,
      eventCount: 4,
      skillSearchCount: 1,
      skillInvocationCount: 1,
      hasSkillSearchSummary: true,
      hasSkillInvocationSummary: true,
      skillBodyReadObserved: true,
      skillGateObserved: true,
      skillBodyReadBeforeGate: true,
      skillGateMode: "selected_skills",
      skillGateWorkspaceRuntimeEnable: null,
      skillGateSourceAllowlist: [],
      skillSearchEventIndex: 0,
      skillBodyReadEventIndex: 1,
      skillGateEventIndex: 2,
      skillInvocationEventIndex: 3,
      skillSearchBeforeSkillInvocation: true,
      searchQuery: SKILLS_RUNTIME_QUERY,
      invocationSkillName: SKILLS_RUNTIME_SKILL_NAME,
    });
  });

  it("rejects Skills runtime gate evidence that precedes the skill body read", () => {
    const scenario = createSkillsRuntimeFixtureScenario(
      "skills-runtime-unit-session",
    );
    const evidenceExportResult = {
      evidencePack: {
        observabilitySummary: {
          skillSearches: [
            {
              query: SKILLS_RUNTIME_QUERY,
              toolCallId: scenario.searchToolCallId,
            },
          ],
          skillInvocations: [
            {
              skillName: SKILLS_RUNTIME_SKILL_NAME,
              toolCallId: scenario.skillToolCallId,
              workspaceSkillRuntimeEnable: { source: "manual_session_enable" },
            },
          ],
        },
      },
      events: [
        canonicalToolCompletedEvent(scenario.searchToolCallId),
        {
          type: "runtime.status",
          payload: {
            metadata: { skillRuntime: { event: "skill_gate_decision" } },
          },
        },
        {
          type: "runtime.status",
          payload: {
            metadata: { skillRuntime: { event: "skill_body_read" } },
          },
        },
        canonicalToolCompletedEvent(scenario.skillToolCallId),
      ],
    };

    expect(
      summarizeSkillsRuntimeEvidenceExport(evidenceExportResult, scenario),
    ).toMatchObject({
      skillBodyReadObserved: true,
      skillGateObserved: false,
      skillBodyReadBeforeGate: false,
      skillBodyReadEventIndex: 2,
      skillGateEventIndex: 1,
    });
  });

  it("ties Skills runtime body and gate evidence to the selected tool-call pair", () => {
    const natural = createSkillsRuntimeFixtureScenario(
      "skills-runtime-unit-session",
    );
    const explicit = createSkillsRuntimeFixtureScenario(
      "skills-runtime-unit-session",
      { variant: "explicit" },
    );
    const evidenceExportResult = {
      evidencePack: {
        observabilitySummary: {
          skillSearches: [
            {
              query: SKILLS_RUNTIME_QUERY,
              toolCallId: natural.searchToolCallId,
            },
            {
              query: SKILLS_RUNTIME_QUERY,
              toolCallId: explicit.searchToolCallId,
            },
          ],
          skillInvocations: [
            {
              skillName: SKILLS_RUNTIME_SKILL_NAME,
              toolCallId: natural.skillToolCallId,
              workspaceSkillRuntimeEnable: { source: "manual_session_enable" },
            },
          ],
        },
      },
      events: [
        canonicalToolCompletedEvent(natural.searchToolCallId),
        {
          type: "runtime.status",
          payload: {
            metadata: { skillRuntime: { event: "skill_body_read" } },
          },
        },
        {
          type: "runtime.status",
          payload: {
            metadata: { skillRuntime: { event: "skill_gate_decision" } },
          },
        },
        canonicalToolCompletedEvent(natural.skillToolCallId),
        canonicalToolCompletedEvent(explicit.searchToolCallId),
        canonicalToolCompletedEvent(explicit.skillToolCallId),
      ],
    };

    expect(
      summarizeSkillsRuntimeEvidenceExport(evidenceExportResult, natural),
    ).toMatchObject({
      skillBodyReadObserved: true,
      skillGateObserved: true,
      skillBodyReadBeforeGate: true,
      skillSearchBeforeSkillInvocation: true,
    });
    expect(
      summarizeSkillsRuntimeEvidenceExport(evidenceExportResult, explicit),
    ).toMatchObject({
      hasSkillSearchSummary: true,
      hasSkillInvocationSummary: false,
      skillBodyReadObserved: false,
      skillGateObserved: false,
      skillSearchBeforeSkillInvocation: true,
    });
  });

  it("summarizes the manual-enable Skills runtime gate mode and allowlist", () => {
    const scenario = createManualEnableSkillsRuntimeFixtureScenario(
      "skills-runtime-unit-session",
    );
    const evidenceExportResult = {
      evidencePack: {
        observabilitySummary: {
          skillSearches: [
            {
              query: SKILLS_RUNTIME_QUERY,
              toolCallId: scenario.searchToolCallId,
            },
          ],
          skillInvocations: [
            {
              skillName: SKILLS_RUNTIME_SKILL_NAME,
              toolCallId: scenario.skillToolCallId,
              workspaceSkillRuntimeEnable: { source: "manual_session_enable" },
            },
          ],
        },
      },
      events: [
        canonicalToolCompletedEvent(scenario.searchToolCallId),
        {
          type: "runtime.status",
          payload: {
            metadata: { skillRuntime: { event: "skill_body_read" } },
          },
        },
        {
          type: "runtime.status",
          payload: {
            metadata: {
              skill_runtime: {
                event: "skill_gate_decision",
                mode: "workspace_runtime_enable",
                workspace_runtime_enable: true,
                source_allowlist: [SKILLS_RUNTIME_SKILL_NAME],
              },
            },
          },
        },
        canonicalToolCompletedEvent(scenario.skillToolCallId),
      ],
    };

    expect(
      summarizeSkillsRuntimeEvidenceExport(evidenceExportResult, scenario),
    ).toMatchObject({
      hasSkillSearchSummary: true,
      hasSkillInvocationSummary: true,
      skillBodyReadObserved: true,
      skillGateObserved: true,
      skillBodyReadBeforeGate: true,
      skillGateMode: "workspace_runtime_enable",
      skillGateWorkspaceRuntimeEnable: true,
      skillGateSourceAllowlist: [SKILLS_RUNTIME_SKILL_NAME],
      skillSearchBeforeSkillInvocation: true,
      searchQuery: SKILLS_RUNTIME_QUERY,
      invocationSkillName: SKILLS_RUNTIME_SKILL_NAME,
    });
  });

  it("summarizes expert Skills runtime declaration, selection, and invocation evidence", () => {
    const scenario = createExpertSkillsRuntimeFixtureScenario(
      "expert-skills-runtime-unit-session",
    );
    const evidenceExportResult = {
      evidencePack: {
        observability_summary: {
          skill_searches: [
            {
              query: SKILLS_RUNTIME_QUERY,
              toolCallId: scenario.searchToolCallId,
            },
          ],
          skillInvocations: [
            {
              skill_name: SKILLS_RUNTIME_SKILL_NAME,
              tool_call_id: scenario.skillToolCallId,
              workspace_skill_runtime_enable: {
                source: "manual_session_enable",
              },
            },
          ],
        },
      },
      events: [
        {
          type: "runtime.status",
          payload: {
            metadata: {
              expertSkillsRuntime: {
                event: "expert_declared_skill_refs",
                skillRefs: [EXPERT_SKILLS_RUNTIME_SKILL_REF],
              },
            },
          },
        },
        canonicalToolCompletedEvent(scenario.searchToolCallId, {
          eventTypeKey: "event_type",
        }),
        {
          type: "runtime.status",
          payload: {
            metadata: {
              skillRuntime: { event: "skill_body_read" },
            },
          },
        },
        {
          type: "runtime.status",
          payload: {
            metadata: {
              skill_runtime: {
                event: "skill_gate_decision",
                mode: "selected_skills",
              },
            },
          },
        },
        {
          type: "runtime.status",
          payload: {
            metadata: {
              expert_skills_runtime: {
                event: "expert_selected_skill",
                skill_name: SKILLS_RUNTIME_SKILL_NAME,
              },
            },
          },
        },
        canonicalToolCompletedEvent(scenario.skillToolCallId, {
          eventTypeKey: "eventType",
          callIdKey: "callId",
        }),
        {
          type: "runtime.status",
          payload: {
            metadata: {
              expertSkillsRuntime: {
                event: "expert_invoked_skill",
                skillName: SKILLS_RUNTIME_SKILL_NAME,
              },
            },
          },
        },
      ],
    };

    expect(
      summarizeSkillsRuntimeEvidenceExport(evidenceExportResult, scenario),
    ).toMatchObject({
      hasEvidencePack: true,
      eventCount: 7,
      hasSkillSearchSummary: true,
      hasSkillInvocationSummary: true,
      skillBodyReadObserved: true,
      skillGateObserved: true,
      skillBodyReadBeforeGate: true,
      skillGateMode: "selected_skills",
      expertDeclaredObserved: true,
      expertSelectedObserved: true,
      expertInvokedObserved: true,
      expertDeclaredSkillRefs: [EXPERT_SKILLS_RUNTIME_SKILL_REF],
      expertSelectedSkill: SKILLS_RUNTIME_SKILL_NAME,
      expertInvokedSkill: SKILLS_RUNTIME_SKILL_NAME,
      skillSearchBeforeSkillInvocation: true,
      searchQuery: SKILLS_RUNTIME_QUERY,
      invocationSkillName: SKILLS_RUNTIME_SKILL_NAME,
    });
  });

  it("keeps the Skills runtime fixture in the current Agent Runtime regression smoke", () => {
    const content = readCurrentFixtureRegressionSmokeScript();

    expectAllToContain(expect, content, [
      "Claw Skills Runtime natural + explicit $skill + Skills workspace try Electron fixture",
      "claw-chat-current-fixture-smoke.mjs",
      '"skills-runtime"',
      "claw-chat-current-fixture-skills-runtime-regression",
      "Skills Runtime natural + 显式 $skill + 技能中心试用入口三入口按需加载 Electron fixture",
      "Claw MCP structuredContent Agent Chat GUI Electron fixture",
      '"mcp-structured-content"',
      "claw-chat-current-fixture-mcp-structured-content-regression",
      "MCP structuredContent 到 Agent Chat GUI 可见 Electron fixture",
      "Claw Expert Plaza Skills Runtime click-through Electron fixture",
      '"expert-plaza-skills-runtime"',
      "claw-chat-current-fixture-expert-plaza-skills-runtime-regression",
      "Expert Plaza 点击专家卡片进入同一 Skills Runtime 闭环 Electron fixture",
      "Claw Expert Panel Skills Runtime override Electron fixture",
      '"expert-panel-skills-runtime"',
      "claw-chat-current-fixture-expert-panel-skills-runtime-regression",
      "ExpertInfoPanel 调整 skillRefs 后下一轮继承同一 Skills Runtime 闭环并展示 Evidence Pack 复盘 Electron fixture",
      'LIME_ALLOW_LIVE_PROVIDER_SMOKE: "0"',
      'LIME_REAL_API_TEST: "0"',
    ]);
    expectAllNotToContain(expect, content, [
      "Claw Expert Skills Runtime declared + selected + invoked Electron fixture",
      '"expert-skills-runtime"',
      "claw-chat-current-fixture-expert-skills-runtime-regression",
    ]);
  });
}
