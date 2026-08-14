export const TOOL_ORCHESTRATOR_SANDBOX_RETRY_GATE_B_BATCH_ID =
  "tool-orchestrator-sandbox-retry-gate-b";
export const TOOL_ORCHESTRATOR_SANDBOX_RETRY_FINAL_TEXT =
  "TOOL_ORCHESTRATOR_SANDBOX_RETRY_DONE";
export const TOOL_ORCHESTRATOR_SANDBOX_RETRY_CALL_ID =
  "call-tool-orchestrator-sandbox-retry";
export const TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_GATE_B_BATCH_ID =
  "tool-orchestrator-managed-network-retry-gate-b";
export const TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_FINAL_TEXT =
  "TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_DONE";
export const TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_CALL_ID =
  "call-tool-orchestrator-managed-network-retry";

function record(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function optionalNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function optionalString(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function extractToolOrchestratorAttemptEvidence({
  callId,
  evidenceExport,
}) {
  const events = Array.isArray(evidenceExport?.events)
    ? evidenceExport.events
    : [];
  for (const event of [...events].reverse()) {
    const eventType = String(
      event?.type ?? event?.eventType ?? event?.event_type ?? "",
    );
    if (eventType !== "item.completed") {
      continue;
    }
    const item = record(record(event?.payload)?.item);
    const itemId = String(item?.itemId ?? item?.item_id ?? item?.id ?? "");
    if (!itemId || !itemId.endsWith(callId)) {
      continue;
    }
    const metadata = record(item?.metadata) ?? {};
    const attemptCount = optionalNumber(
      metadata.toolAttemptCount ?? metadata.tool_attempt_count,
    );
    if (attemptCount == null) {
      continue;
    }
    return {
      itemId,
      toolAttemptCount: attemptCount,
      toolAttemptNumber: optionalNumber(
        metadata.toolAttemptNumber ?? metadata.tool_attempt_number,
      ),
      toolEscalated:
        typeof (metadata.toolEscalated ?? metadata.tool_escalated) === "boolean"
          ? (metadata.toolEscalated ?? metadata.tool_escalated)
          : null,
      approvalSource: optionalString(
        metadata.approvalSource ?? metadata.approval_source,
      ),
      requestedSandboxPolicy: optionalString(
        metadata.requestedSandboxPolicy ??
          metadata.requested_sandbox_policy,
      ),
      effectiveSandboxPolicy: optionalString(
        metadata.effectiveSandboxPolicy ??
          metadata.effective_sandbox_policy,
      ),
      firstAttemptOutcome: optionalString(
        metadata.firstAttemptOutcome ?? metadata.first_attempt_outcome,
      ),
    };
  }
  return null;
}

export function buildToolOrchestratorSandboxRetryVisibleDomAssertions({
  evidence,
  snapshot,
}) {
  const runtimeAssertions = evidence?.assertions ?? {};
  const policies = evidence?.runtime?.policies ?? {};
  const respondedRequests = Array.isArray(
    evidence?.runtime?.finalSnapshot?.respondedRequests,
  )
    ? evidence.runtime.finalSnapshot.respondedRequests
    : [];
  const appServerCalls = Array.isArray(snapshot?.appServerCalls)
    ? snapshot.appServerCalls
    : [];
  const fileChangeGroups = Array.isArray(snapshot?.fileChangeGroups)
    ? snapshot.fileChangeGroups
    : [];
  const approval = respondedRequests[0] ?? null;

  return {
    sandboxRetryVisibleDomUsesRealElectronHost:
      snapshot?.electron === true &&
      snapshot?.hasInvokeBridge === true &&
      snapshot?.supportsAppServer === true,
    sandboxRetryVisibleDomNavigatedToTargetSession:
      Boolean(snapshot?.sessionId) &&
      snapshot?.activeSessionId === snapshot?.sessionId,
    sandboxRetryVisibleDomCurrentReadModelObserved: appServerCalls.some(
      (call) =>
        call?.method === "thread/read" &&
        call?.transport === "electron-ipc" &&
        call?.status === "success",
    ),
    sandboxRetryUsedRestrictedInitialPolicy:
      policies.approvalPolicy === "on-request" &&
      policies.sandboxPolicy === "workspace-write",
    sandboxRetryApprovalUsedTypedFileChangeRequest:
      respondedRequests.length === 1 &&
      approval?.method === "item/fileChange/requestApproval" &&
      String(approval?.outerRequestId || "").length > 0 &&
      approval?.itemId === TOOL_ORCHESTRATOR_SANDBOX_RETRY_CALL_ID,
    sandboxRetryRuntimeProofsPassed:
      runtimeAssertions.sandboxRetryKeptOneCanonicalToolIdentity === true &&
      runtimeAssertions.sandboxRetryApprovalResponded === true &&
      runtimeAssertions.sandboxRetryCompletedCanonicalTool === true &&
      runtimeAssertions.sandboxRetryWroteOutsideWorkspaceAfterApproval === true,
    sandboxRetryVisibleDomHasOneFileChangeGroup:
      fileChangeGroups.length === 1 && fileChangeGroups[0]?.fileRowCount > 0,
    sandboxRetryVisibleDomFileChangeGroupCompleted:
      fileChangeGroups.length === 1 &&
      fileChangeGroups[0]?.status === "completed",
    sandboxRetryVisibleDomFileChangeGroupVisible:
      fileChangeGroups.length === 1 && fileChangeGroups[0]?.visible === true,
    sandboxRetryVisibleDomFinalAssistantTextVisible:
      snapshot?.finalAssistantTextVisible === true,
    sandboxRetryVisibleDomInvokeErrorsClear: snapshot?.invokeErrorCount === 0,
    sandboxRetryVisibleDomConsoleErrorsClear: snapshot?.consoleErrorCount === 0,
  };
}

export function buildToolOrchestratorManagedNetworkRetryVisibleDomAssertions({
  evidence,
  snapshot,
}) {
  const runtimeAssertions = evidence?.assertions ?? {};
  const policies = evidence?.runtime?.policies ?? {};
  const respondedRequests = Array.isArray(
    evidence?.runtime?.finalSnapshot?.respondedRequests,
  )
    ? evidence.runtime.finalSnapshot.respondedRequests
    : [];
  const appServerCalls = Array.isArray(snapshot?.appServerCalls)
    ? snapshot.appServerCalls
    : [];
  const approval = respondedRequests[0] ?? null;
  const typedContext = approval?.networkApprovalContext ?? null;

  return {
    managedNetworkRetryVisibleDomUsesRealElectronHost:
      snapshot?.electron === true &&
      snapshot?.hasInvokeBridge === true &&
      snapshot?.supportsAppServer === true,
    managedNetworkRetryVisibleDomNavigatedToTargetSession:
      Boolean(snapshot?.sessionId) &&
      snapshot?.activeSessionId === snapshot?.sessionId,
    managedNetworkRetryVisibleDomCurrentReadModelObserved: appServerCalls.some(
      (call) =>
        call?.method === "thread/read" &&
        call?.transport === "electron-ipc" &&
        call?.status === "success",
    ),
    managedNetworkRetryUsedWorkspaceWriteInitialPolicy:
      policies.approvalPolicy === "on-request" &&
      policies.sandboxPolicy === "workspace-write",
    managedNetworkRetryApprovalUsedTypedNetworkContext:
      respondedRequests.length === 1 &&
      approval?.method === "item/commandExecution/requestApproval" &&
      approval?.itemId === TOOL_ORCHESTRATOR_MANAGED_NETWORK_RETRY_CALL_ID &&
      typedContext?.host === "127.0.0.1" &&
      typedContext?.protocol === "http",
    managedNetworkRetryRuntimeProofsPassed:
      runtimeAssertions.managedNetworkRetryKeptOneCanonicalToolIdentity === true &&
      runtimeAssertions.managedNetworkRetryApprovalRespondedOnce === true &&
      runtimeAssertions.managedNetworkRetryUsedTypedNetworkApproval === true &&
      runtimeAssertions.managedNetworkRetryRecordedTwoAttempts === true &&
      runtimeAssertions.managedNetworkRetryPreservedWorkspaceSandbox === true &&
      runtimeAssertions.managedNetworkRetryReachedRealEndpointOnce === true,
    managedNetworkRetryVisibleDomFinalAssistantTextVisible:
      snapshot?.finalAssistantTextVisible === true,
    managedNetworkRetryVisibleDomEndpointProofVisible:
      snapshot?.endpointProofVisible === true,
    managedNetworkRetryVisibleDomInvokeErrorsClear:
      snapshot?.invokeErrorCount === 0,
    managedNetworkRetryVisibleDomConsoleErrorsClear:
      snapshot?.consoleErrorCount === 0,
  };
}
