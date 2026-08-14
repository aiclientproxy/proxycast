export const AGENT_CONTROL_VISIBLE_DOM_GATE_B_BATCH_ID = "agent-control-tools";
export const AGENT_CONTROL_CAPACITY_GATE_B_BATCH_ID =
  "agent-control-capacity-gate-b";
export const AGENT_CONTROL_RESIDENCY_GATE_B_BATCH_ID =
  "agent-control-residency-gate-b";
export const AGENT_CONTROL_FINAL_TEXT =
  "AGENT_RUNTIME_AGENT_CONTROL_TOOLS_DONE";
export const AGENT_CONTROL_CAPACITY_FINAL_TEXT =
  "AGENT_RUNTIME_AGENT_CAPACITY_GATE_B_DONE";
export const AGENT_CONTROL_RESIDENCY_FINAL_TEXT =
  "AGENT_RUNTIME_AGENT_RESIDENCY_GATE_B_DONE";
export const AGENT_CONTROL_TOOL_NAMES = [
  "spawn_agent",
  "list_agents",
  "send_message",
  "followup_task",
  "interrupt_agent",
  "wait_agent",
];
export const AGENT_CONTROL_SUBAGENT_ACTIVITY_KINDS = [
  "started",
  "interacted",
  "interrupted",
];
export const PARENT_OWNED_DIRECT_INPUT_ERROR =
  "direct app-server input is not allowed for parent-owned threads";

export const PARENT_OWNED_PLACEHOLDERS = [
  "此子线程由父线程管理，无法直接输入",
  "此子執行緒由父執行緒管理，無法直接輸入",
  "This child thread is managed by its parent and cannot accept direct input",
  "この子スレッドは親スレッドによって管理されているため、直接入力できません",
  "이 하위 스레드는 상위 스레드에서 관리하므로 직접 입력할 수 없습니다",
];

const RETIRED_TEAM_TOOL_NAMES = new Set([
  "Agent",
  "TeamCreate",
  "TeamDelete",
  "SendMessage",
  "ListPeers",
  "SendUserMessage",
]);

function toolIdentity(rows) {
  return rows
    .filter((row) => AGENT_CONTROL_TOOL_NAMES.includes(String(row?.name || "")))
    .map((row) => [
      String(row?.id || ""),
      String(row?.name || ""),
      String(row?.status || ""),
    ])
    .sort((left, right) =>
      JSON.stringify(left).localeCompare(JSON.stringify(right)),
    );
}

function subagentActivityIdentity(rows) {
  return rows
    .map((row) => [
      String(row?.itemId || ""),
      String(row?.activityKind || ""),
      String(row?.threadId || ""),
    ])
    .sort((left, right) =>
      JSON.stringify(left).localeCompare(JSON.stringify(right)),
    );
}

function agentStateIdentity(states) {
  return states
    .map((state) => [
      String(state?.thread_id || state?.threadId || ""),
      String(state?.status || ""),
      state?.message ?? null,
    ])
    .sort((left, right) =>
      JSON.stringify(left).localeCompare(JSON.stringify(right)),
    );
}

function sameIdentity(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

export function buildAgentControlVisibleDomAssertions({ evidence, snapshot }) {
  const matrix = Array.isArray(evidence?.runtime?.matrix)
    ? evidence.runtime.matrix
    : [];
  const completedRuntimeTools = new Set(
    matrix
      .filter(
        (entry) => entry?.status === "completed" && entry?.success !== false,
      )
      .map((entry) => String(entry?.tool || "").trim())
      .filter(Boolean),
  );
  const expectedWaitAgentStates = matrix.find(
    (entry) => entry?.tool === "wait_agent",
  )?.agentStates;
  const restoredWaitAgentStates = Array.isArray(snapshot?.waitAgentStates)
    ? snapshot.waitAgentStates
    : [];
  const typedToolRows = Array.isArray(snapshot?.typedToolRows)
    ? snapshot.typedToolRows
    : [];
  const agentControlRows = typedToolRows.filter((row) =>
    AGENT_CONTROL_TOOL_NAMES.includes(String(row?.name || "")),
  );
  const rowCountByName = new Map(
    AGENT_CONTROL_TOOL_NAMES.map((toolName) => [
      toolName,
      agentControlRows.filter((row) => row?.name === toolName).length,
    ]),
  );
  const subagentActivityRows = Array.isArray(snapshot?.subagentActivityRows)
    ? snapshot.subagentActivityRows
    : [];
  const visibleActivityKinds = new Set(
    subagentActivityRows
      .filter(
        (row) => row?.visible === true && String(row?.threadId || "").trim(),
      )
      .map((row) =>
        String(row?.activityKind || "")
          .trim()
          .toLowerCase(),
      )
      .filter(Boolean),
  );
  const appServerCalls = Array.isArray(snapshot?.appServerCalls)
    ? snapshot.appServerCalls
    : [];
  const preRestartToolRows = Array.isArray(snapshot?.preRestart?.typedToolRows)
    ? snapshot.preRestart.typedToolRows
    : [];
  const preRestartActivityRows = Array.isArray(
    snapshot?.preRestart?.subagentActivityRows,
  )
    ? snapshot.preRestart.subagentActivityRows
    : [];
  const preRestartChildThreadIds = new Set(
    preRestartActivityRows
      .map((row) => String(row?.threadId || ""))
      .filter(Boolean),
  );
  const restoredChildThreadIds = new Set(
    subagentActivityRows
      .map((row) => String(row?.threadId || ""))
      .filter(Boolean),
  );
  const parentOwnedChild = snapshot?.parentOwnedChild ?? {};
  const canonicalChildThread = parentOwnedChild.canonicalThread ?? {};
  const parentOwnedDom = parentOwnedChild.dom ?? {};
  const parentOwnedControls = parentOwnedDom.controls ?? {};
  const serverRejection = parentOwnedChild.serverRejection ?? {};

  return {
    visibleDomUsesRealElectronHost:
      snapshot?.electron === true &&
      snapshot?.hasInvokeBridge === true &&
      snapshot?.supportsAppServer === true,
    visibleDomRestoredAfterColdRestart:
      snapshot?.coldRestart?.electronProcessReplaced === true,
    visibleDomNavigatedToTargetSession:
      Boolean(snapshot?.sessionId) &&
      snapshot?.activeSessionId === snapshot?.sessionId,
    visibleDomCurrentReadModelObserved: appServerCalls.some(
      (call) =>
        call?.method === "thread/read" &&
        call?.transport === "electron-ipc" &&
        call?.status === "success",
    ),
    visibleDomCurrentThreadListObserved: appServerCalls.some(
      (call) =>
        call?.method === "thread/list" &&
        call?.transport === "electron-ipc" &&
        call?.status === "success",
    ),
    visibleDomToolIdentityStableAcrossRestart:
      preRestartToolRows.length > 0 &&
      sameIdentity(
        toolIdentity(preRestartToolRows),
        toolIdentity(typedToolRows),
      ),
    visibleDomSubAgentIdentityStableAcrossRestart:
      preRestartActivityRows.length > 0 &&
      sameIdentity(
        subagentActivityIdentity(preRestartActivityRows),
        subagentActivityIdentity(subagentActivityRows),
      ),
    visibleDomChildThreadStableAcrossRestart:
      preRestartChildThreadIds.size === 1 &&
      restoredChildThreadIds.size === 1 &&
      [...preRestartChildThreadIds][0] === [...restoredChildThreadIds][0],
    visibleDomWaitAgentStatesStableAcrossRestart:
      Array.isArray(expectedWaitAgentStates) &&
      expectedWaitAgentStates.length > 0 &&
      restoredWaitAgentStates.length > 0 &&
      sameIdentity(
        agentStateIdentity(expectedWaitAgentStates),
        agentStateIdentity(restoredWaitAgentStates),
      ),
    visibleDomAllAgentControlToolsCompletedInReadModel:
      AGENT_CONTROL_TOOL_NAMES.every((toolName) =>
        completedRuntimeTools.has(toolName),
      ),
    visibleDomAllAgentControlToolRowsPresentOnce:
      AGENT_CONTROL_TOOL_NAMES.every(
        (toolName) => rowCountByName.get(toolName) === 1,
      ),
    visibleDomAllAgentControlToolRowsCompleted: agentControlRows.every(
      (row) => row?.status === "completed",
    ),
    visibleDomAllAgentControlToolRowsVisible:
      agentControlRows.length === AGENT_CONTROL_TOOL_NAMES.length &&
      agentControlRows.every((row) => row?.visible === true),
    visibleDomCanonicalSubAgentActivitiesVisible:
      AGENT_CONTROL_SUBAGENT_ACTIVITY_KINDS.every((activityKind) =>
        visibleActivityKinds.has(activityKind),
      ),
    visibleDomSubAgentActivitiesUseCanonicalIdentity:
      subagentActivityRows.length >=
        AGENT_CONTROL_SUBAGENT_ACTIVITY_KINDS.length &&
      subagentActivityRows.every(
        (row) =>
          String(row?.itemId || "").trim().length > 0 &&
          String(row?.threadId || "").trim().length > 0,
      ),
    visibleDomParentOwnedChildUsesCanonicalThreadFact:
      Boolean(parentOwnedChild.childThreadId) &&
      canonicalChildThread.id === parentOwnedChild.childThreadId &&
      canonicalChildThread.parentThreadId === snapshot?.sessionId &&
      canonicalChildThread.canAcceptDirectInput === false,
    visibleDomParentOwnedChildRouteOpened:
      Boolean(canonicalChildThread.sessionId) &&
      parentOwnedDom.activeSessionId === canonicalChildThread.sessionId &&
      parentOwnedDom.childThreadId === parentOwnedChild.childThreadId,
    visibleDomParentOwnedComposerDisabled:
      parentOwnedDom.textareaVisible === true &&
      parentOwnedDom.textareaDisabled === true &&
      PARENT_OWNED_PLACEHOLDERS.includes(parentOwnedDom.placeholder) &&
      parentOwnedControls.sendUnavailable === true &&
      parentOwnedControls.accessModeDisabled === true &&
      parentOwnedControls.modelSelectorCount > 0 &&
      parentOwnedControls.modelSelectorsDisabled === true &&
      parentOwnedControls.taskModeDisabled !== false,
    visibleDomParentOwnedUiAttemptDidNotStartTurn:
      parentOwnedChild.uiAttempt?.dispatchedEnter === true &&
      parentOwnedChild.uiAttempt?.sendUnavailable === true &&
      parentOwnedChild.uiAttempt?.turnStartCountBefore ===
        parentOwnedChild.uiAttempt?.turnStartCountAfter,
    visibleDomParentOwnedServerRejectsDirectTurn:
      serverRejection.code === -32600 &&
      serverRejection.message === PARENT_OWNED_DIRECT_INPUT_ERROR &&
      serverRejection.hasResult === false,
    visibleDomRetiredTeamToolsAbsent: !typedToolRows.some((row) =>
      RETIRED_TEAM_TOOL_NAMES.has(String(row?.name || "")),
    ),
    visibleDomFinalAssistantTextVisible:
      snapshot?.finalAssistantTextVisible === true,
    visibleDomInvokeErrorsClear: snapshot?.invokeErrorCount === 0,
    visibleDomConsoleErrorsClear: snapshot?.consoleErrorCount === 0,
  };
}

export function buildAgentControlCapacityVisibleDomAssertions({ snapshot }) {
  const typedToolRows = Array.isArray(snapshot?.typedToolRows)
    ? snapshot.typedToolRows
    : [];
  const spawnRows = typedToolRows.filter(
    (row) => String(row?.name || "") === "spawn_agent",
  );
  const subagentThreadIds = new Set(
    (Array.isArray(snapshot?.subagentActivityRows)
      ? snapshot.subagentActivityRows
      : []
    )
      .map((row) => String(row?.threadId || "").trim())
      .filter(Boolean),
  );
  const appServerCalls = Array.isArray(snapshot?.appServerCalls)
    ? snapshot.appServerCalls
    : [];
  return {
    visibleDomUsesRealElectronHost:
      snapshot?.electron === true &&
      snapshot?.hasInvokeBridge === true &&
      snapshot?.supportsAppServer === true,
    visibleDomCurrentReadModelObserved: appServerCalls.some(
      (call) =>
        call?.method === "thread/read" &&
        call?.transport === "electron-ipc" &&
        call?.status === "success",
    ),
    visibleDomAllParallelSpawnRowsPresent: spawnRows.length >= 4,
    visibleDomCapacityRejectionVisible: spawnRows.some(
      (row) =>
        String(row?.status || "").toLowerCase() === "failed" ||
        String(row?.text || "").includes("agent_limit_reached"),
    ),
    visibleDomThreeChildIdentitiesVisible: subagentThreadIds.size >= 3,
    visibleDomFinalAssistantTextVisible:
      snapshot?.finalAssistantTextVisible === true,
    visibleDomInvokeErrorsClear: snapshot?.invokeErrorCount === 0,
    visibleDomConsoleErrorsClear: snapshot?.consoleErrorCount === 0,
  };
}

export function buildAgentControlResidencyVisibleDomAssertions({ snapshot }) {
  const typedToolRows = Array.isArray(snapshot?.typedToolRows)
    ? snapshot.typedToolRows
    : [];
  const spawnRows = typedToolRows.filter(
    (row) => String(row?.name || "") === "spawn_agent",
  );
  const followupRows = typedToolRows.filter(
    (row) => String(row?.name || "") === "followup_task",
  );
  const activityRows = Array.isArray(snapshot?.subagentActivityRows)
    ? snapshot.subagentActivityRows
    : [];
  const activityByThread = new Map();
  for (const row of activityRows) {
    const threadId = String(row?.threadId || "").trim();
    if (!threadId) continue;
    const kinds = activityByThread.get(threadId) || new Set();
    kinds.add(String(row?.activityKind || "").trim().toLowerCase());
    activityByThread.set(threadId, kinds);
  }
  const appServerCalls = Array.isArray(snapshot?.appServerCalls)
    ? snapshot.appServerCalls
    : [];
  const reusedChildIdentity = [...activityByThread.values()].some(
    (kinds) => kinds.has("started") && kinds.has("interacted"),
  );
  return {
    visibleDomUsesRealElectronHost:
      snapshot?.electron === true &&
      snapshot?.hasInvokeBridge === true &&
      snapshot?.supportsAppServer === true,
    visibleDomCurrentReadModelObserved: appServerCalls.some(
      (call) =>
        call?.method === "thread/read" &&
        call?.transport === "electron-ipc" &&
        call?.status === "success",
    ),
    visibleDomFourChildIdentitiesVisible:
      new Set(
        activityRows
          .map((row) => String(row?.threadId || "").trim())
          .filter(Boolean),
      ).size >= 4,
    visibleDomTerminalSlotReused:
      spawnRows.length >= 4 &&
      spawnRows.every((row) => row?.status === "completed") &&
      snapshot?.residency?.terminalSlotReused === true,
    visibleDomLruColdReloadVisible:
      followupRows.some((row) => row?.status === "completed") &&
      reusedChildIdentity &&
      snapshot?.residency?.lruColdReload === true,
    visibleDomFollowupUsesExistingChildIdentity: reusedChildIdentity,
    visibleDomFinalAssistantTextVisible:
      snapshot?.finalAssistantTextVisible === true,
    visibleDomInvokeErrorsClear: snapshot?.invokeErrorCount === 0,
    visibleDomConsoleErrorsClear: snapshot?.consoleErrorCount === 0,
  };
}
