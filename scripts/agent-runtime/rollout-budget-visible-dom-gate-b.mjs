export const ROLLOUT_BUDGET_GATE_B_BATCH_ID =
  "agent-rollout-budget-gate-b";
export const ROLLOUT_BUDGET_FINAL_TEXT =
  "AGENT_RUNTIME_ROLLOUT_BUDGET_GATE_B_DONE";

function turnStatuses(thread) {
  return (Array.isArray(thread?.turns) ? thread.turns : []).map((turn) =>
    String(turn?.status || turn?.status?.type || "")
      .trim()
      .toLowerCase(),
  );
}

export function buildRolloutBudgetVisibleDomAssertions({ snapshot }) {
  const appServerCalls = Array.isArray(snapshot?.appServerCalls)
    ? snapshot.appServerCalls
    : [];
  const restartRejection = snapshot?.restartRejection ?? {};
  const statuses = turnStatuses(snapshot?.thread);
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
    visibleDomBudgetExhaustionProjected:
      (statuses.includes("failed") &&
        String(snapshot?.latestTurnError?.message || "").includes(
          "rollout budget",
        )) ||
      restartRejection?.error?.data?.reason === "rollout_budget_exhausted",
    visibleDomRestartRejectionUsesStableReason:
      restartRejection?.error?.data?.reason === "rollout_budget_exhausted" &&
      restartRejection?.error?.data?.retryable === false,
    visibleDomElectronRestartReplaced:
      snapshot?.coldRestart?.electronProcessReplaced === true,
    visibleDomFinalFailureVisible: snapshot?.failureVisible === true,
    visibleDomInvokeErrorsClear: snapshot?.invokeErrorCount === 0,
    visibleDomConsoleErrorsClear: snapshot?.consoleErrorCount === 0,
  };
}
