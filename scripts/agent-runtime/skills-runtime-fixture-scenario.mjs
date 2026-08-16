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
  "readModelSkillBodyReadObserved",
  "readModelSkillGateObserved",
  "skillSearchBeforeSkillInvocation",
  "explicitSkillsRuntimePromptReachedBackend",
  "guiExplicitSkillsRuntimeInputSubmitted",
  "guiExplicitSkillsRuntimeCompleted",
  "readModelExplicitSkillsRuntimeCompleted",
  "readModelExplicitSkillSearchObserved",
  "readModelExplicitSkillInvocationObserved",
  "readModelExplicitSkillBodyReadObserved",
  "readModelExplicitSkillGateObserved",
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
  "readModelManualEnableSkillBodyReadObserved",
  "readModelManualEnableSkillGateObserved",
  "readModelManualEnableWorkspaceRuntimeEnableObserved",
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
  "readModelExpertSkillBodyReadObserved",
  "readModelExpertSkillGateObserved",
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
  "expertPanelReadModelSkillBodyReadObserved",
  "expertPanelReadModelSkillGateObserved",
  "expertPanelReadModelSkillSearchObserved",
  "expertPanelReadModelSkillInvocationObserved",
  "expertPanelSkillSearchBeforeSkillInvocation",
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

function eventPayloadSkillRuntime(event) {
  return eventPayloadRuntime(event, ["skillRuntime", "skill_runtime"]);
}

function eventPayloadRuntime(event, keys) {
  const payload = readRecord(event?.payload);
  const status = readRecord(payload?.status);
  const metadata =
    readRecord(payload?.metadata) ?? readRecord(status?.metadata);
  for (const key of keys) {
    const runtime = readRecord(metadata?.[key]);
    if (runtime) {
      return runtime;
    }
  }
  return null;
}

export function summarizeSkillsRuntimeStatusEvents(
  messages,
  { sessionId, threadId, turnId } = {},
) {
  const expectedSessionId = typeof sessionId === "string" ? sessionId : null;
  const expectedThreadId = typeof threadId === "string" ? threadId : null;
  const expectedTurnId = typeof turnId === "string" ? turnId : null;
  const candidateEvents = (Array.isArray(messages) ? messages : [])
    .filter((message) => message?.method === "agentSession/event")
    .map((message) => readRecord(message.params)?.event)
    .map((event) => readRecord(event))
    .filter((event) => event?.type === "runtime.status")
    .map((event) => ({
      eventId: event.eventId ?? event.event_id ?? null,
      sessionId: event.sessionId ?? event.session_id ?? null,
      threadId: event.threadId ?? event.thread_id ?? null,
      turnId: event.turnId ?? event.turn_id ?? null,
      runtime: eventPayloadSkillRuntime(event),
    }));
  const events = candidateEvents
    .filter((event) => {
      if (expectedSessionId && event.sessionId !== expectedSessionId) {
        return false;
      }
      if (expectedThreadId && event.threadId !== expectedThreadId) {
        return false;
      }
      if (expectedTurnId && event.turnId !== expectedTurnId) {
        return false;
      }
      return Boolean(event.runtime);
    })
    .map((event) => ({
      event,
      runtime: event.runtime,
    }));
  const bodyReadIndex = events.findIndex(
    ({ runtime }) => runtime?.event === "skill_body_read",
  );
  const gateIndex = events.findIndex(
    ({ runtime }) => runtime?.event === "skill_gate_decision",
  );
  const gateRuntime = gateIndex >= 0 ? events[gateIndex].runtime : null;
  const expertEvents = (Array.isArray(messages) ? messages : [])
    .filter((message) => message?.method === "agentSession/event")
    .map((message) => readRecord(message.params)?.event)
    .map((event) => readRecord(event))
    .filter((event) => event?.type === "runtime.status")
    .map((event) => ({
      sessionId: event.sessionId ?? event.session_id ?? null,
      threadId: event.threadId ?? event.thread_id ?? null,
      turnId: event.turnId ?? event.turn_id ?? null,
      runtime: eventPayloadRuntime(event, [
        "expertSkillsRuntime",
        "expert_skills_runtime",
      ]),
    }))
    .filter((event) => {
      if (expectedSessionId && event.sessionId !== expectedSessionId) {
        return false;
      }
      if (expectedThreadId && event.threadId !== expectedThreadId) {
        return false;
      }
      if (expectedTurnId && event.turnId !== expectedTurnId) {
        return false;
      }
      return Boolean(event.runtime);
    });
  const expertDeclaredEntry = expertEvents.find(
    ({ runtime }) => runtime?.event === "expert_declared_skill_refs",
  );
  const expertSelectedEntry = expertEvents.find(
    ({ runtime }) => runtime?.event === "expert_selected_skill",
  );
  const expertInvokedEntry = expertEvents.find(
    ({ runtime }) => runtime?.event === "expert_invoked_skill",
  );
  const expertDeclaredRuntime = expertDeclaredEntry?.runtime ?? {};
  const expertSelectedRuntime = expertSelectedEntry?.runtime ?? {};
  const expertInvokedRuntime = expertInvokedEntry?.runtime ?? {};
  const expertSkillRefs =
    expertDeclaredRuntime.skillRefs ?? expertDeclaredRuntime.skill_refs;
  return {
    eventCount: events.length,
    skillBodyReadObserved: bodyReadIndex >= 0,
    skillGateObserved: bodyReadIndex >= 0 && gateIndex > bodyReadIndex,
    skillBodyReadEventIndex: bodyReadIndex,
    skillGateEventIndex: gateIndex,
    skillGateMode: gateRuntime?.mode ?? null,
    skillGateWorkspaceRuntimeEnable:
      gateRuntime?.workspaceRuntimeEnable ??
      gateRuntime?.workspace_runtime_enable ??
      null,
    skillGateSourceAllowlist: Array.isArray(
      gateRuntime?.sourceAllowlist ?? gateRuntime?.source_allowlist,
    )
      ? gateRuntime.sourceAllowlist ?? gateRuntime.source_allowlist
      : [],
    expertDeclaredObserved: Boolean(expertDeclaredEntry),
    expertSelectedObserved: Boolean(expertSelectedEntry),
    expertInvokedObserved: Boolean(expertInvokedEntry),
    expertDeclaredSkillRefs: Array.isArray(expertSkillRefs)
      ? expertSkillRefs
      : [],
    expertSelectedSkill:
      expertSelectedRuntime.skillName ??
      expertSelectedRuntime.skill_name ??
      null,
    expertInvokedSkill:
      expertInvokedRuntime.skillName ??
      expertInvokedRuntime.skill_name ??
      null,
  };
}

export function summarizeSkillsRuntimeThreadRead(
  readModelResult,
  { searchToolCallId, skillToolCallId },
) {
  const canonicalTurns = Array.isArray(readModelResult?.thread?.turns)
    ? readModelResult.thread.turns
    : Array.isArray(readModelResult?.turns)
      ? readModelResult.turns
      : Array.isArray(readModelResult?.detail?.turns)
        ? readModelResult.detail.turns
        : null;
  const matchingTurn = canonicalTurns?.find((turn) => {
    const items = Array.isArray(turn?.items) ? turn.items : [];
    const serialized = JSON.stringify(items);
    return (
      serialized.includes(searchToolCallId) ||
      serialized.includes(skillToolCallId)
    );
  });
  const canonicalItems = matchingTurn
    ? Array.isArray(matchingTurn.items)
      ? matchingTurn.items
      : []
    : canonicalTurns
      ? canonicalTurns.flatMap((turn) =>
          Array.isArray(turn?.items) ? turn.items : [],
        )
      : [];
  if (canonicalTurns) {
    const itemText = canonicalItems.map((item) => JSON.stringify(item)).join("\n");
    const skillSearchEventIndex = canonicalItems.findIndex((item) =>
      JSON.stringify(item).includes(searchToolCallId),
    );
    const skillInvocationEventIndex = canonicalItems.findIndex((item) =>
      JSON.stringify(item).includes(skillToolCallId),
    );
    const searchObserved = skillSearchEventIndex >= 0;
    const invocationObserved = skillInvocationEventIndex >= 0;
    const skillBodyReadObserved =
      itemText.includes("skill_body_read") ||
      itemText.includes("skillBodyRead") ||
      itemText.includes("SKILL.md");
    const runtimeMetadata = canonicalItems
      .map((item, index) => ({
        index,
        metadata:
          readRecord(item?.metadata) ?? readRecord(item?.payload?.metadata),
      }))
      .filter((entry) => entry.metadata);
    const skillRuntimeEntries = runtimeMetadata
      .map(({ index, metadata }) => ({
        index,
        runtime:
          readRecord(metadata.skillRuntime) ??
          readRecord(metadata.skill_runtime),
      }))
      .filter((entry) => entry.runtime);
    const expertRuntimeEntries = runtimeMetadata
      .map(({ index, metadata }) => ({
        index,
        runtime:
          readRecord(metadata.expertSkillsRuntime) ??
          readRecord(metadata.expert_skills_runtime),
      }))
      .filter((entry) => entry.runtime);
    const skillBodyReadEntry = skillRuntimeEntries.find(
      ({ runtime }) => runtime.event === "skill_body_read",
    );
    const skillGateEntry = skillRuntimeEntries.find(
      ({ runtime }) => runtime.event === "skill_gate_decision",
    );
    const skillBodyReadEventIndex = skillBodyReadEntry?.index ?? -1;
    const skillGateEventIndex = skillGateEntry?.index ?? -1;
    const skillBodyReadBeforeGate =
      skillBodyReadEventIndex >= 0 &&
      skillGateEventIndex > skillBodyReadEventIndex;
    const skillGateRuntime = skillGateEntry?.runtime ?? {};
    const expertDeclaredEntry = expertRuntimeEntries.find(
      ({ runtime }) => runtime.event === "expert_declared_skill_refs",
    );
    const expertSelectedEntry = expertRuntimeEntries.find(
      ({ runtime }) => runtime.event === "expert_selected_skill",
    );
    const expertInvokedEntry = expertRuntimeEntries.find(
      ({ runtime }) => runtime.event === "expert_invoked_skill",
    );
    const expertDeclaredRuntime = expertDeclaredEntry?.runtime ?? {};
    const expertSelectedRuntime = expertSelectedEntry?.runtime ?? {};
    const expertInvokedRuntime = expertInvokedEntry?.runtime ?? {};
    const skillRefs =
      expertDeclaredRuntime.skillRefs ?? expertDeclaredRuntime.skill_refs;
    return {
      hasThreadRead: true,
      itemCount: canonicalItems.length,
      skillSearchCount: searchObserved ? 1 : 0,
      skillInvocationCount: invocationObserved ? 1 : 0,
      hasSkillSearchSummary: searchObserved,
      hasSkillInvocationSummary: invocationObserved,
      skillBodyReadObserved,
      skillBodyReadBeforeGate,
      skillGateObserved: skillBodyReadBeforeGate,
      skillGateMode: skillGateRuntime.mode ?? null,
      skillGateWorkspaceRuntimeEnable:
        skillGateRuntime.workspaceRuntimeEnable ??
        skillGateRuntime.workspace_runtime_enable ??
        null,
      skillGateSourceAllowlist: Array.isArray(
        skillGateRuntime.sourceAllowlist ?? skillGateRuntime.source_allowlist,
      )
        ? skillGateRuntime.sourceAllowlist ?? skillGateRuntime.source_allowlist
        : [],
      skillSearchEventIndex,
      skillBodyReadEventIndex,
      skillGateEventIndex,
      skillInvocationEventIndex,
      expertDeclaredObserved: Boolean(expertDeclaredEntry),
      expertSelectedObserved: Boolean(expertSelectedEntry),
      expertInvokedObserved: Boolean(expertInvokedEntry),
      expertDeclaredSkillRefs: Array.isArray(skillRefs) ? skillRefs : [],
      expertSelectedSkill:
        expertSelectedRuntime.skillName ??
        expertSelectedRuntime.skill_name ??
        null,
      expertInvokedSkill:
        expertInvokedRuntime.skillName ??
        expertInvokedRuntime.skill_name ??
        null,
      skillSearchBeforeSkillInvocation:
        skillSearchEventIndex >= 0 &&
        skillInvocationEventIndex >= 0 &&
        skillSearchEventIndex < skillInvocationEventIndex,
      searchQuery: searchObserved ? SKILLS_RUNTIME_QUERY : null,
      invocationSkillName: invocationObserved ? SKILLS_RUNTIME_SKILL_NAME : null,
    };
  }

  return {
    hasThreadRead: false,
    itemCount: 0,
    skillSearchCount: 0,
    skillInvocationCount: 0,
    hasSkillSearchSummary: false,
    hasSkillInvocationSummary: false,
    skillBodyReadObserved: false,
    skillGateObserved: false,
    skillBodyReadBeforeGate: false,
    skillGateMode: null,
    skillGateWorkspaceRuntimeEnable: null,
    skillGateSourceAllowlist: [],
    skillSearchEventIndex: -1,
    skillBodyReadEventIndex: -1,
    skillGateEventIndex: -1,
    skillInvocationEventIndex: -1,
    expertDeclaredObserved: false,
    expertSelectedObserved: false,
    expertInvokedObserved: false,
    expertDeclaredSkillRefs: [],
    expertSelectedSkill: null,
    expertInvokedSkill: null,
    skillSearchBeforeSkillInvocation: false,
    searchQuery: null,
    invocationSkillName: null,
  };
}
