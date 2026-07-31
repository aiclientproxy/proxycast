export const SKILLS_RUNTIME_PROMPT = "验证 Skills 按需加载";
export const SKILLS_RUNTIME_DONE_TEXT = "CLAW_SKILLS_RUNTIME_DONE";
export const SKILLS_RUNTIME_QUERY = "capability report";
export const SKILLS_RUNTIME_SKILL_NAME = "project:capability-report";
export const SKILLS_RUNTIME_EXPLICIT_PROMPT =
  "使用 $project:capability-report 验证 Skills 按需加载";
export const SKILLS_RUNTIME_EXPLICIT_DONE_TEXT =
  "CLAW_SKILLS_RUNTIME_EXPLICIT_DONE";
export const SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT =
  "先试用一次「Capability Report」技能。请按需读取它的 SKILL.md 和必要引用，说明会使用哪些能力，然后执行一次最小验证。";
export const SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT =
  "CLAW_SKILLS_RUNTIME_MANUAL_ENABLE_DONE";
export const EXPERT_SKILLS_RUNTIME_PROMPT =
  "请以「代码文学专家」身份，使用绑定技能完成一次最小代码审查。";
export const EXPERT_SKILLS_RUNTIME_DONE_TEXT =
  "CLAW_EXPERT_SKILLS_RUNTIME_DONE";
export const EXPERT_SKILLS_RUNTIME_ID = "code-literature";
export const EXPERT_SKILLS_RUNTIME_TITLE = "代码文学专家";
export const EXPERT_SKILLS_RUNTIME_RELEASE_ID = "rel-code-literature-20260515";
export const EXPERT_SKILLS_RUNTIME_TENANT_ID = "local-seeded";
export const EXPERT_SKILLS_RUNTIME_SKILL_REF = "skill:capability-report";
export const EXPERT_SKILLS_RUNTIME_BASE_SKILL_REF = "skill:code-review";
export const EXPERT_SKILLS_RUNTIME_PANEL_PROMPT =
  "请继续以「代码文学专家」身份，使用刚添加的技能再做一次最小代码审查。";
export const EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT =
  "CLAW_EXPERT_SKILLS_RUNTIME_PANEL_DONE";
export const SKILLS_RUNTIME_ASSERTION_KEYS = [
  "initialCurrentSkillListObserved",
  "typedSkillsChangedConsumed",
  "automaticSkillListRefreshObserved",
  "guiSkillCatalogUpdated",
  "skillsCatalogRefreshDidNotUseManualRefresh",
  "skillsRuntimePromptReachedBackend",
  "guiSkillsRuntimeInputSubmitted",
  "guiSkillsRuntimeCompleted",
  "readModelSkillsRuntimeCompleted",
  "readModelSkillSearchObserved",
  "readModelSkillInvocationObserved",
  "evidenceSkillBodyReadObserved",
  "evidenceSkillGateObserved",
  "evidencePackSkillSearchObserved",
  "evidencePackSkillInvocationObserved",
  "skillSearchBeforeSkillInvocation",
  "explicitSkillsRuntimePromptReachedBackend",
  "guiExplicitSkillsRuntimeInputSubmitted",
  "guiExplicitSkillsRuntimeCompleted",
  "readModelExplicitSkillsRuntimeCompleted",
  "readModelExplicitSkillSearchObserved",
  "readModelExplicitSkillInvocationObserved",
  "evidenceExplicitSkillBodyReadObserved",
  "evidenceExplicitSkillGateObserved",
  "evidencePackExplicitSkillSearchObserved",
  "evidencePackExplicitSkillInvocationObserved",
  "explicitSkillSearchBeforeSkillInvocation",
  "manualEnableSkillsRuntimePromptReachedBackend",
  "manualEnableSkillsRuntimeMetadataReachedBackend",
  "manualEnableSkillsRuntimeSkillDirectoryPrepared",
  "manualEnableSkillsRuntimeLaunchedFromSkillsWorkspace",
  "manualEnableSkillsRuntimeUsedAgentSession",
  "guiManualEnableSkillsRuntimeCompleted",
  "readModelManualEnableSkillsRuntimeCompleted",
  "readModelManualEnableSkillSearchObserved",
  "readModelManualEnableSkillInvocationObserved",
  "evidenceManualEnableSkillBodyReadObserved",
  "evidenceManualEnableSkillGateObserved",
  "evidenceManualEnableWorkspaceRuntimeEnableObserved",
  "evidencePackManualEnableSkillSearchObserved",
  "evidencePackManualEnableSkillInvocationObserved",
  "manualEnableSkillSearchBeforeSkillInvocation",
];
export const EXPERT_SKILLS_RUNTIME_ASSERTION_KEYS = [
  "expertSkillsRuntimePromptReachedBackend",
  "expertSkillsRuntimeMetadataReachedBackend",
  "expertDeclaredSkillRefsObserved",
  "expertSelectedSkillObserved",
  "expertInvokedSkillObserved",
  "guiExpertSkillsRuntimeSessionVisible",
  "readModelExpertSkillsRuntimeCompleted",
  "readModelExpertSkillSearchObserved",
  "readModelExpertSkillInvocationObserved",
  "evidenceExpertSkillBodyReadObserved",
  "evidenceExpertSkillGateObserved",
  "evidencePackExpertSkillSearchObserved",
  "evidencePackExpertSkillInvocationObserved",
  "expertSkillSearchBeforeSkillInvocation",
];
export const EXPERT_PLAZA_SKILLS_RUNTIME_ASSERTION_KEYS = [
  "expertPlazaCatalogInjected",
  "expertPlazaCardClicked",
  "expertPlazaAutoSendTurnStarted",
];
export const EXPERT_PANEL_SKILLS_RUNTIME_ASSERTION_KEYS = [
  "expertPanelSkillPickerOpened",
  "expertPanelSkillAdded",
  "expertPanelAddedSkillVisible",
  "expertPanelSecondTurnPromptReachedBackend",
  "expertPanelSkillRefsOverrideReachedBackend",
  "expertPanelReadModelCompleted",
  "expertPanelEvidenceSkillBodyReadObserved",
  "expertPanelEvidenceSkillGateObserved",
  "expertPanelEvidenceSkillSearchObserved",
  "expertPanelEvidenceSkillInvocationObserved",
  "expertPanelSkillSearchBeforeSkillInvocation",
  "expertPanelEvidencePackExportedFromHarnessPanel",
];

export function createSkillsRuntimeFixtureScenario(sessionId, options = {}) {
  const variant = options.variant ?? "natural";
  const idSuffix = variant === "natural" ? "" : `-${variant}`;
  const prompt = options.prompt ?? SKILLS_RUNTIME_PROMPT;
  const doneText = options.doneText ?? SKILLS_RUNTIME_DONE_TEXT;
  const summaryText = options.summaryText ?? "Skills runtime 证据已完成";
  const guiSummaryText = options.guiSummaryText ?? summaryText;
  const trigger = options.trigger ?? "runtime_suggested";
  const selectionReason =
    options.selectionReason ?? "skill_search selected capability report";
  const gateMode = options.gateMode ?? "selected_skills";
  const workspaceRuntimeEnable = options.workspaceRuntimeEnable ?? null;
  const sourceAllowlist = options.sourceAllowlist ?? [];
  const dedupeGuardTexts = options.dedupeGuardTexts ?? [];
  const disallowedVisibleTexts = options.disallowedVisibleTexts ?? [];
  return {
    variant,
    prompt,
    doneText,
    summaryText,
    guiSummaryText,
    dedupeGuardTexts,
    disallowedVisibleTexts,
    trigger,
    selectionReason,
    gateMode,
    workspaceRuntimeEnable,
    sourceAllowlist,
    searchToolCallId: `${sessionId}:tool:skill-search${idSuffix}`,
    skillToolCallId: `${sessionId}:tool:skill-capability-report${idSuffix}`,
    fixtureText:
      options.fixtureText ??
      `${summaryText}：先搜索 capability report，再加载 capability-report，并只授权本轮选中 Skill。\n`,
    searchOutput: JSON.stringify({
      query: SKILLS_RUNTIME_QUERY,
      results: [
        {
          name: SKILLS_RUNTIME_SKILL_NAME,
          description: "Generate a capability report from repository facts.",
          locator: "skill://project/capability-report/SKILL.md",
          score: 0.94,
        },
      ],
    }),
    skillOutput: JSON.stringify({
      skill: SKILLS_RUNTIME_SKILL_NAME,
      artifact: "capability-report.md",
      summary:
        "Capability report generated after turn-scoped Skill authorization.",
    }),
  };
}

export function createExplicitSkillsRuntimeFixtureScenario(sessionId) {
  return createSkillsRuntimeFixtureScenario(sessionId, {
    variant: "explicit",
    prompt: SKILLS_RUNTIME_EXPLICIT_PROMPT,
    doneText: SKILLS_RUNTIME_EXPLICIT_DONE_TEXT,
    summaryText: "Skills runtime 显式触发证据已完成",
    trigger: "explicit",
    selectionReason: "$project:capability-report explicit skill mention",
  });
}

export function createManualEnableSkillsRuntimeFixtureScenario(sessionId) {
  return createSkillsRuntimeFixtureScenario(sessionId, {
    variant: "manual-enable",
    prompt: SKILLS_RUNTIME_MANUAL_ENABLE_PROMPT,
    doneText: SKILLS_RUNTIME_MANUAL_ENABLE_DONE_TEXT,
    summaryText: "Skills runtime 技能中心试用入口证据已完成",
    trigger: "workspace_panel_manual_enable",
    selectionReason:
      "workspace_skill_runtime_enable launched from Skills workspace panel",
    gateMode: "workspace_runtime_enable",
    workspaceRuntimeEnable: true,
    sourceAllowlist: [SKILLS_RUNTIME_SKILL_NAME],
  });
}

export function createExpertSkillsRuntimeFixtureScenario(sessionId) {
  return createSkillsRuntimeFixtureScenario(sessionId, {
    variant: "expert",
    prompt: EXPERT_SKILLS_RUNTIME_PROMPT,
    doneText: EXPERT_SKILLS_RUNTIME_DONE_TEXT,
    summaryText: "专家 Skills runtime 证据已完成",
    trigger: "expert_declared_skill_refs",
    selectionReason:
      "expert skillRefs declared skill:capability-report; selector still used skill_search before invocation",
    gateMode: "selected_skills",
    fixtureText:
      "专家 Skills runtime 证据已完成：专家声明 skillRefs 只作为候选提示，实际执行仍经过 skill_search、SKILL.md 按需读取、gate 和 Skill 调用。\n",
  });
}

export function createExpertPanelSkillsRuntimeFixtureScenario(sessionId) {
  return createSkillsRuntimeFixtureScenario(sessionId, {
    variant: "expert-panel",
    prompt: EXPERT_SKILLS_RUNTIME_PANEL_PROMPT,
    doneText: EXPERT_SKILLS_RUNTIME_PANEL_DONE_TEXT,
    summaryText: "专家面板新增 Skill 后的下一轮 runtime 证据已完成",
    dedupeGuardTexts: ["专家 Skills runtime 证据已完成"],
    disallowedVisibleTexts: [
      "专家 Skills runtime 证据已完成",
      "我识别到专家绑定的 skillRefs",
    ],
    trigger: "expert_panel_skill_refs_override",
    selectionReason:
      "ExpertInfoPanel added skill:capability-report; next turn inherited overridden expert skillRefs before skill_search and invocation",
    gateMode: "selected_skills",
    fixtureText:
      "专家面板新增 Skill 后的下一轮 runtime 证据已完成：右侧面板调整 skillRefs 后，下一轮请求继续通过 skill_search、SKILL.md 按需读取、gate 和 Skill 调用。\n",
  });
}

export function buildExpertSkillsRuntimeMetadata() {
  const skillRefs = [EXPERT_SKILLS_RUNTIME_SKILL_REF];
  return {
    expert: {
      expertId: EXPERT_SKILLS_RUNTIME_ID,
      releaseId: EXPERT_SKILLS_RUNTIME_RELEASE_ID,
      title: EXPERT_SKILLS_RUNTIME_TITLE,
      category: "engineering",
      source: "fixture",
      catalogVersion: "fixture-experts-2026-06-21",
      tenantId: EXPERT_SKILLS_RUNTIME_TENANT_ID,
      personaRef: "expert-persona:code-literature@1.0.0",
      skillRefs,
      workflowRefs: [],
      memoryEnabled: true,
      workflowEnabled: false,
    },
    harness: {
      source: "smoke:claw-chat-current-fixture:expert-skills-runtime",
      expert: {
        expert_id: EXPERT_SKILLS_RUNTIME_ID,
        release_id: EXPERT_SKILLS_RUNTIME_RELEASE_ID,
        title: EXPERT_SKILLS_RUNTIME_TITLE,
        category: "engineering",
        source: "fixture",
        catalog_version: "fixture-experts-2026-06-21",
        tenant_id: EXPERT_SKILLS_RUNTIME_TENANT_ID,
        persona_ref: "expert-persona:code-literature@1.0.0",
        skill_refs: skillRefs,
        workflow_refs: [],
        memory_enabled: true,
        workflow_enabled: false,
      },
    },
  };
}

export function buildExpertSkillsRuntimeCatalog(options = {}) {
  const releaseSkillRefs = options.releaseSkillRefs ?? [
    EXPERT_SKILLS_RUNTIME_SKILL_REF,
  ];
  const tenantId = options.tenantId ?? EXPERT_SKILLS_RUNTIME_TENANT_ID;
  const syncedAt = "2026-06-21T00:00:00.000Z";
  return {
    version: "fixture-experts-2026-06-21",
    tenantId,
    syncedAt,
    categories: [
      { key: "all", title: "全部", sort: 0 },
      { key: "engineering", title: "工程部", sort: 30 },
    ],
    rankings: [
      {
        key: "personal_picks",
        title: "为你推荐",
        summary: "用于 Expert Plaza -> Skills Runtime 点击穿透 fixture。",
        category: "engineering",
        items: [EXPERT_SKILLS_RUNTIME_ID],
        generatedAt: syncedAt,
      },
    ],
    items: [
      {
        id: EXPERT_SKILLS_RUNTIME_ID,
        slug: EXPERT_SKILLS_RUNTIME_ID,
        title: EXPERT_SKILLS_RUNTIME_TITLE,
        summary: "读取代码上下文，按需选择技能完成最小审查。",
        avatar: { kind: "emoji", value: "CL" },
        category: "engineering",
        tags: ["code", "review", "skills-runtime"],
        source: "seeded_fallback",
        stats: {
          usageCount: 55000,
          likeCount: 6700,
          hotScore: 0.94,
          freshReleasedAt: syncedAt,
        },
        release: {
          releaseId: EXPERT_SKILLS_RUNTIME_RELEASE_ID,
          version: "1.0.0",
          personaRef: "expert-persona:code-literature@1.0.0",
          personaHash: "sha256:fixture-code-literature",
          memoryTemplateRef: "memory-template:code-literature@1.0.0",
          skillRefs: releaseSkillRefs,
          workflowRefs: ["workflow:code-explain-review"],
          readiness: { requiresModel: true, requiresProject: true },
          releasedAt: syncedAt,
        },
        promptStarters: [EXPERT_SKILLS_RUNTIME_PROMPT],
        showcase: [
          {
            title: "最小代码审查",
            body: "先声明专家技能引用，再通过 selector 和 gate 选择本轮 Skill。",
          },
        ],
      },
    ],
  };
}

export function buildManualEnableSkillsRuntimeMetadata(
  workspaceRoot,
  registeredSkillDirectory,
) {
  return {
    harness: {
      source: "smoke:claw-chat-current-fixture:skills-runtime-manual-enable",
      workspace_skill_runtime_enable: {
        source: "manual_session_enable",
        approval: "manual",
        workspace_root: workspaceRoot,
        bindings: [
          {
            directory: "capability-report",
            skill: SKILLS_RUNTIME_SKILL_NAME,
            registered_skill_directory: registeredSkillDirectory,
            source_draft_id: "capdraft-fixture-capability-report",
            source_verification_report_id: "capver-fixture-capability-report",
            permission_summary: ["Level 0 read-only fixture"],
          },
        ],
      },
    },
  };
}

export function renderSkillsRuntimeBackendEvents({
  promptFlagName = "isSkillsRuntimePrompt",
  searchToolCallId,
  skillToolCallId,
  searchOutput,
  skillOutput,
  trigger,
  selectionReason,
  gateMode = "selected_skills",
  workspaceRuntimeEnable = null,
  sourceAllowlist = [],
}) {
  return `
  if (${promptFlagName}) {
    emitEvents([
      {
        type: "item.started",
        payload: buildCanonicalToolItem({
          sessionId: input.request?.session?.sessionId,
          threadId: currentThreadId(),
          turnId: currentTurnId(),
          itemId: "${searchToolCallId}",
          ordinal: 2,
          callId: "${searchToolCallId}",
          name: "skill_search",
          arguments: {
            query: "${SKILLS_RUNTIME_QUERY}"
          },
          status: "inProgress"
        })
      }
    ]);
    await sleep(80);
    emitEvents([
      {
        type: "item.completed",
        payload: buildCanonicalToolItem({
          sessionId: input.request?.session?.sessionId,
          threadId: currentThreadId(),
          turnId: currentTurnId(),
          itemId: "${searchToolCallId}",
          ordinal: 2,
          callId: "${searchToolCallId}",
          name: "skill_search",
          status: "completed",
          output: {
            text: ${JSON.stringify(searchOutput)}
          },
          metadata: {
            tool_family: "skill_search",
            skill_search_query: "${SKILLS_RUNTIME_QUERY}",
            skill_search_snapshot_skill_count: 9,
            skill_search_result_count: 1
          }
        })
      }
    ]);
    await sleep(80);
    emitEvents([
      {
        type: "runtime.status",
        payload: {
          status: "loaded",
          text: "已按需读取 capability-report/SKILL.md",
          metadata: {
            skillRuntime: {
              event: "skill_body_read",
              skillName: "${SKILLS_RUNTIME_SKILL_NAME}",
              trigger: "${trigger}",
              reason: "${selectionReason}",
              skillFilePath: ".agents/skills/capability-report/SKILL.md",
              bodyChars: 512,
              status: "loaded"
            },
            skill_runtime: {
              event: "skill_body_read",
              skill_name: "${SKILLS_RUNTIME_SKILL_NAME}",
              trigger: "${trigger}",
              reason: "${selectionReason}",
              skill_file_path: ".agents/skills/capability-report/SKILL.md",
              body_chars: 512,
              status: "loaded"
            }
          }
        }
      }
    ]);
    await sleep(80);
    emitEvents([
      {
        type: "runtime.status",
        payload: {
          status: "allowed",
          text: "已把 SkillTool 裁剪到本轮选中的 capability-report",
          metadata: {
            skillRuntime: {
              event: "skill_gate_decision",
              mode: "${gateMode}",
              selectedSkills: ["${SKILLS_RUNTIME_SKILL_NAME}"],
              sourceAllowlist: ${JSON.stringify(sourceAllowlist)},
              workspaceRuntimeEnable: ${JSON.stringify(workspaceRuntimeEnable)}
            },
            skill_runtime: {
              event: "skill_gate_decision",
              mode: "${gateMode}",
              selected_skills: ["${SKILLS_RUNTIME_SKILL_NAME}"],
              source_allowlist: ${JSON.stringify(sourceAllowlist)},
              workspace_runtime_enable: ${JSON.stringify(workspaceRuntimeEnable)}
            }
          }
        }
      }
    ]);
    await sleep(80);
    emitEvents([
      {
        type: "item.started",
        payload: buildCanonicalToolItem({
          sessionId: input.request?.session?.sessionId,
          threadId: currentThreadId(),
          turnId: currentTurnId(),
          itemId: "${skillToolCallId}",
          ordinal: 3,
          callId: "${skillToolCallId}",
          name: "Skill",
          arguments: {
            skill: "${SKILLS_RUNTIME_SKILL_NAME}"
          },
          status: "inProgress"
        })
      }
    ]);
    await sleep(80);
    emitEvents([
      {
        type: "item.completed",
        payload: buildCanonicalToolItem({
          sessionId: input.request?.session?.sessionId,
          threadId: currentThreadId(),
          turnId: currentTurnId(),
          itemId: "${skillToolCallId}",
          ordinal: 3,
          callId: "${skillToolCallId}",
          name: "Skill",
          status: "completed",
          output: {
            text: ${JSON.stringify(skillOutput)}
          },
          metadata: {
            tool_family: "skill",
            skill_name: "${SKILLS_RUNTIME_SKILL_NAME}",
            workspace_skill_runtime_enable: {
              source: "manual_session_enable",
              approval: "manual",
              authorization_scope: "session",
              directory: "capability-report",
              skill: "${SKILLS_RUNTIME_SKILL_NAME}"
            }
          }
        })
      }
    ]);
    await sleep(120);
  }
`;
}

function readRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function collectEvidenceExportEvents(evidenceExportResult) {
  return Array.isArray(evidenceExportResult?.events)
    ? evidenceExportResult.events
    : [];
}

function evidenceExportEventType(event) {
  return String(event?.eventType ?? event?.event_type ?? event?.type ?? "");
}

function evidenceExportEventPayload(event) {
  return readRecord(event?.payload) ?? {};
}

function evidenceExportEventItem(event) {
  return readRecord(evidenceExportEventPayload(event).item);
}

function evidenceExportPayloadMetadata(event) {
  const payload = evidenceExportEventPayload(event);
  return (
    readRecord(evidenceExportEventItem(event)?.metadata) ??
    readRecord(payload.metadata) ??
    {}
  );
}

function evidenceExportEventToolCallId(event) {
  const payload = evidenceExportEventPayload(event);
  const item = evidenceExportEventItem(event);
  const itemPayload = readRecord(item?.payload) ?? {};
  return String(
    itemPayload.call_id ??
      itemPayload.callId ??
      item?.itemId ??
      item?.item_id ??
      payload.toolCallId ??
      payload.tool_call_id ??
      payload.toolId ??
      payload.tool_id ??
      payload.id ??
      "",
  );
}

function evidenceExportEventHasSkillRuntimeEvent(event, runtimeEvent) {
  const metadata = evidenceExportPayloadMetadata(event);
  const skillRuntime =
    readRecord(metadata.skillRuntime) ??
    readRecord(metadata.skill_runtime) ??
    {};
  return String(skillRuntime.event ?? "") === runtimeEvent;
}

function evidenceExportEventHasExpertRuntimeEvent(event, runtimeEvent) {
  const metadata = evidenceExportPayloadMetadata(event);
  const expertRuntime =
    readRecord(metadata.expertSkillsRuntime) ??
    readRecord(metadata.expert_skills_runtime) ??
    {};
  return String(expertRuntime.event ?? "") === runtimeEvent;
}

function evidenceExportEventExpertRuntime(event) {
  const metadata = evidenceExportPayloadMetadata(event);
  return (
    readRecord(metadata.expertSkillsRuntime) ??
    readRecord(metadata.expert_skills_runtime) ??
    {}
  );
}

function evidenceExportEventSkillRuntime(event) {
  const metadata = evidenceExportPayloadMetadata(event);
  return (
    readRecord(metadata.skillRuntime) ??
    readRecord(metadata.skill_runtime) ??
    {}
  );
}

export function summarizeSkillsRuntimeEvidenceExport(
  evidenceExportResult,
  { searchToolCallId, skillToolCallId },
) {
  const evidencePack = evidenceExportResult?.evidencePack;
  const observabilitySummary =
    evidencePack?.observabilitySummary ??
    evidencePack?.observability_summary ??
    {};
  const skillSearches = Array.isArray(
    observabilitySummary.skillSearches ?? observabilitySummary.skill_searches,
  )
    ? (observabilitySummary.skillSearches ??
      observabilitySummary.skill_searches)
    : [];
  const skillInvocations = Array.isArray(
    observabilitySummary.skillInvocations ??
      observabilitySummary.skill_invocations,
  )
    ? (observabilitySummary.skillInvocations ??
      observabilitySummary.skill_invocations)
    : [];
  const events = collectEvidenceExportEvents(evidenceExportResult);
  const skillSearchEventIndex = events.findIndex(
    (event) =>
      evidenceExportEventType(event) === "item.completed" &&
      evidenceExportEventToolCallId(event) === searchToolCallId,
  );
  const skillInvocationEventIndex = events.findIndex(
    (event) =>
      evidenceExportEventType(event) === "item.completed" &&
      evidenceExportEventToolCallId(event) === skillToolCallId,
  );
  const eventBelongsToCurrentSkillInvocation = (_event, eventIndex) =>
    skillSearchEventIndex >= 0 &&
    skillInvocationEventIndex >= 0 &&
    eventIndex > skillSearchEventIndex &&
    eventIndex < skillInvocationEventIndex;
  const skillBodyReadEventIndex = events.findIndex(
    (event, eventIndex) =>
      eventBelongsToCurrentSkillInvocation(event, eventIndex) &&
      evidenceExportEventHasSkillRuntimeEvent(event, "skill_body_read"),
  );
  const skillGateEventIndex = events.findIndex(
    (event, eventIndex) =>
      eventBelongsToCurrentSkillInvocation(event, eventIndex) &&
      evidenceExportEventHasSkillRuntimeEvent(event, "skill_gate_decision"),
  );
  const skillGateEvent =
    skillGateEventIndex >= 0 ? events[skillGateEventIndex] : null;
  const skillBodyReadBeforeGate =
    skillBodyReadEventIndex >= 0 &&
    skillGateEventIndex > skillBodyReadEventIndex;
  const skillGateRuntime = skillGateEvent
    ? evidenceExportEventSkillRuntime(skillGateEvent)
    : {};
  const expertDeclaredEventIndex = events.findIndex((event) =>
    evidenceExportEventHasExpertRuntimeEvent(
      event,
      "expert_declared_skill_refs",
    ),
  );
  const expertSelectedEventIndex = events.findIndex((event) =>
    evidenceExportEventHasExpertRuntimeEvent(event, "expert_selected_skill"),
  );
  const expertInvokedEventIndex = events.findIndex((event) =>
    evidenceExportEventHasExpertRuntimeEvent(event, "expert_invoked_skill"),
  );
  const expertDeclaredRuntime =
    expertDeclaredEventIndex >= 0
      ? evidenceExportEventExpertRuntime(events[expertDeclaredEventIndex])
      : {};
  const expertSelectedRuntime =
    expertSelectedEventIndex >= 0
      ? evidenceExportEventExpertRuntime(events[expertSelectedEventIndex])
      : {};
  const expertInvokedRuntime =
    expertInvokedEventIndex >= 0
      ? evidenceExportEventExpertRuntime(events[expertInvokedEventIndex])
      : {};
  const skillGateSourceAllowlist =
    skillGateRuntime.sourceAllowlist ?? skillGateRuntime.source_allowlist;
  const hasSkillSearchSummary = skillSearches.some((entry) => {
    const query = entry?.query;
    const toolCallId = entry?.toolCallId ?? entry?.tool_call_id;
    return query === SKILLS_RUNTIME_QUERY && toolCallId === searchToolCallId;
  });
  const hasSkillInvocationSummary = skillInvocations.some((entry) => {
    const skillName = entry?.skillName ?? entry?.skill_name;
    const toolCallId = entry?.toolCallId ?? entry?.tool_call_id;
    const runtimeEnable =
      entry?.workspaceSkillRuntimeEnable ??
      entry?.workspace_skill_runtime_enable;
    return (
      skillName === SKILLS_RUNTIME_SKILL_NAME &&
      toolCallId === skillToolCallId &&
      runtimeEnable &&
      typeof runtimeEnable === "object"
    );
  });
  const matchingInvocation = skillInvocations.find((entry) => {
    const skillName = entry?.skillName ?? entry?.skill_name;
    const toolCallId = entry?.toolCallId ?? entry?.tool_call_id;
    return (
      skillName === SKILLS_RUNTIME_SKILL_NAME && toolCallId === skillToolCallId
    );
  });

  return {
    hasEvidencePack: Boolean(evidencePack),
    eventCount: events.length,
    skillSearchCount: skillSearches.length,
    skillInvocationCount: skillInvocations.length,
    hasSkillSearchSummary,
    hasSkillInvocationSummary,
    skillBodyReadObserved: skillBodyReadEventIndex >= 0,
    skillGateObserved: skillBodyReadBeforeGate,
    skillBodyReadBeforeGate,
    skillGateMode: skillGateRuntime.mode ?? null,
    skillGateWorkspaceRuntimeEnable:
      skillGateRuntime.workspaceRuntimeEnable ??
      skillGateRuntime.workspace_runtime_enable ??
      null,
    skillGateSourceAllowlist: Array.isArray(skillGateSourceAllowlist)
      ? skillGateSourceAllowlist
      : [],
    skillSearchEventIndex,
    skillBodyReadEventIndex,
    skillGateEventIndex,
    skillInvocationEventIndex,
    expertDeclaredObserved: expertDeclaredEventIndex >= 0,
    expertSelectedObserved: expertSelectedEventIndex >= 0,
    expertInvokedObserved: expertInvokedEventIndex >= 0,
    expertDeclaredSkillRefs: Array.isArray(
      expertDeclaredRuntime.skillRefs ?? expertDeclaredRuntime.skill_refs,
    )
      ? (expertDeclaredRuntime.skillRefs ?? expertDeclaredRuntime.skill_refs)
      : [],
    expertSelectedSkill:
      expertSelectedRuntime.skillName ??
      expertSelectedRuntime.skill_name ??
      null,
    expertInvokedSkill:
      expertInvokedRuntime.skillName ?? expertInvokedRuntime.skill_name ?? null,
    skillSearchBeforeSkillInvocation:
      skillSearchEventIndex >= 0 &&
      skillInvocationEventIndex >= 0 &&
      skillSearchEventIndex < skillInvocationEventIndex,
    searchQuery:
      skillSearches.find((entry) => {
        const toolCallId = entry?.toolCallId ?? entry?.tool_call_id;
        return (
          entry?.query === SKILLS_RUNTIME_QUERY &&
          toolCallId === searchToolCallId
        );
      })?.query ?? null,
    invocationSkillName:
      matchingInvocation?.skillName ?? matchingInvocation?.skill_name ?? null,
  };
}
