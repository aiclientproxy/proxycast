export function currentChainFromError(error) {
  return error instanceof Error && error.currentChain
    ? error.currentChain
    : null;
}

function isBudgetExhausted(currentChain) {
  if (!currentChain || typeof currentChain !== "object") return false;
  if (
    currentChain.providerStepExhaustion ||
    currentChain.budgetCancellation ||
    currentChain.providerSteps?.budgets?.exhausted
  ) {
    return true;
  }
  return /(?:provider )?budget exhausted/i.test(
    String(currentChain.terminalMessage || ""),
  );
}

function patchBytes(patch) {
  const bytes = Number(patch?.bytes);
  return Number.isFinite(bytes) ? bytes : 0;
}

/**
 * A verifier result is complete only when the current chain, candidate patch,
 * budget state, and Pier reward all agree. The string form remains supported
 * for the small status-only callers and their existing failure semantics.
 */
export function verifierCompletionStatus(input) {
  if (typeof input === "string" || input == null) {
    return input === "completed" ? "verified" : "verified_with_product_failure";
  }
  const currentChain = input.currentChain || input;
  const reward =
    input.reward ?? input.verification?.reward ?? input.verification?.score;
  const complete =
    currentChain.status === "completed" &&
    !isBudgetExhausted(currentChain) &&
    patchBytes(input.patch) > 0 &&
    reward === 1;
  return complete ? "verified" : "verified_with_product_failure";
}

export function classifyFailure(stage, error) {
  const message = error instanceof Error ? error.message : String(error);
  let owner = "environment";
  if (
    /unsupported workspace_type|spawnSync git ENOBUFS|workspace HEAD contains non-candidate commits/i.test(
      message,
    )
  ) {
    owner = "harness";
  } else if (
    /fetch failed|ECONNRESET|ECONNREFUSED|DevBridge health/i.test(message)
  ) {
    owner = "transport";
  } else if (
    /budget|token|cost|DeepSWE turn timeout|timed out waiting for app-server message/i.test(
      message,
    )
  ) {
    owner = "budget";
  } else if (
    /provider|model|api key|authentication|rate.limit/i.test(message)
  ) {
    owner = "model";
  } else if (/empty patch|produced no candidate|\bno[- ]op\b/i.test(message)) {
    owner = "model";
  } else if (/tool|sandbox|approval/i.test(message)) {
    owner = "tool-runtime";
  } else if (
    /app server|agentSession|thread\/(?:start|read)|turn\/(?:start|interrupt)|workspace\/ensure|evidence\/export|DevBridge/i.test(
      message,
    )
  ) {
    owner = "app-server";
  } else if (/Pier|verifier|reward\.json|ctrf\.json/i.test(message)) {
    owner = "verifier";
  } else if (stage === "transport") {
    owner = "transport";
  } else if (stage.startsWith("agent") || /terminal status/i.test(message)) {
    owner = "agent-runtime";
  }
  return {
    schemaVersion: "deepswe-failure-classification-v1",
    generatedAt: new Date().toISOString(),
    status: "failed",
    stage,
    owner,
    message,
  };
}
