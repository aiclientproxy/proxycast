import fs from "node:fs";
import { describe, expect, it } from "vitest";

function readSmokeScript() {
  return fs.readFileSync(
    "scripts/electron/code-artifact-workbench-fixture-smoke.mjs",
    "utf8",
  );
}

describe("code artifact workbench Electron fixture smoke guard", () => {
  it("keeps the smoke on real Electron Desktop Host IPC and App Server JSON-RPC", () => {
    const content = readSmokeScript();

    expect(content).toContain("import { _electron as electron }");
    expect(content).toContain("electron.launch({");
    expect(content).toContain("waitForAppUrlReady");
    expect(content).toContain('logStage("wait-app-url")');
    expect(content).toContain('"--use-mock-keychain"');
    expect(content).toContain("ELECTRON_E2E_USER_DATA_DIR");
    expect(content).toContain('LIME_ELECTRON_E2E: "1"');
    expect(content).toContain('LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"');
    expect(content).toContain("window.__LIME_ELECTRON__ === true");
    expect(content).toContain(
      'typeof window.electronAPI?.invoke === "function"',
    );
    expect(content).toContain("window.electronAPI.supportsCommand");
    expect(content).toContain("app_server_handle_json_lines");
  });

  it("uses a local external fixture backend instead of a live provider or mock backend", () => {
    const content = readSmokeScript();

    expect(content).toContain('APP_SERVER_BACKEND_MODE: "external"');
    expect(content).toContain("APP_SERVER_BACKEND_COMMAND: process.execPath");
    expect(content).toContain("writeFixtureBackend");
    expect(content).toContain('modelProvider: "fixture-provider"');
    expect(content).toContain('model: "fixture-model"');
    expect(content).toContain(
      "requestMetadata = readApplicationMetadata(request.metadata)",
    );
    expect(content).toContain("inputText = readRuntimeInputText(request)");
    expect(content).toContain("function readRuntimeInputText(request)");
    expect(content).toContain("request?.input?.parts");
    expect(content).toContain("threadId = String(session.threadId");
    expect(content).toContain("liveProviderNotUsed");
    expect(content).toContain('entry.kind !== "turnStart"');
    expect(content).toContain("!entry.providerPreference");
    expect(content).toContain("!entry.modelPreference");
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain('backendMode: "mock"');
    expect(content).not.toContain("--allow-live-provider");
    expect(content).not.toContain("APP_SERVER_BACKEND_COMMAND: undefined");
  });

  it("creates a code artifact session and opens the GUI workbench", () => {
    const content = readSmokeScript();

    expect(content).toContain('"thread/start"');
    expect(content).toContain('"turn/start"');
    expect(content).toContain('"thread/read"');
    expect(content).toContain("thread/start 未返回 canonical sessionId");
    expect(content).toContain("thread/start 未返回 canonical thread.id");
    expect(content).toContain('input: [{ type: "text", text: USER_PROMPT }]');
    expect(content).not.toContain('"agentSession/update"');
    expect(content).toContain('type: "item.started"');
    expect(content).toContain('type: "item.completed"');
    expect(content).toContain('type: "message.delta"');
    expect(content).toContain('type: "message.completed"');
    expect(content).not.toContain("canonicalAssistantItem");
    expect(content).toContain("itemId: assistantItemId");
    expect(content).toContain('phase: "final_answer"');
    expect(content).not.toContain('kind: "agentMessage"');
    expect(content).not.toContain('type: "agentMessage"');
    expect(content).toContain("canonicalToolItem");
    expect(content).toContain("call_id: toolCallId");
    expect(content).toContain("changes: [");
    expect(content).toContain('kind: { type: "update" }');
    expect(content).toContain(
      'status: status === "completed" ? "applied" : "proposed"',
    );
    expect(content).not.toContain('callId: "${TOOL_CALL_ID}"');
    expect(content).toContain("ordinal: 4");
    expect(content).not.toContain('type: "tool.started"');
    expect(content).not.toContain('type: "tool.result"');
    expect(content).toContain("TOOL_CALL_ID");
    expect(content).toContain("TOOL_OUTPUT_PREVIEW");
    expect(content).toContain("collectToolCalls");
    expect(content).toContain("hasToolTimelineProjection");
    expect(content).toContain("toolTimelineProjectionPersisted");
    expect(content).toContain("codingProjectionPersisted");
    expect(content).toContain("inspectHistoricalTimelineSummary");
    expect(content).toContain(
      'logStage("inspect-historical-timeline-summary")',
    );
    expect(content).toContain("timelineProcessEvidence");
    expect(content).toContain("hasHistoricalOperationalDetailsHidden");
    expect(content).toContain(
      '[data-testid^="message-list-historical-timeline-preview:"]',
    );
    expect(content).toContain('[data-testid="tool-call-row"]');
    expect(content).toContain("collectCodingWorkbenchGuiEvidence");
    expect(content).toContain("codingWorkbenchGuiEvidence");
    expect(content).toContain("gui-coding-input");
    expect(content).toContain("GUI_CODING_PROMPT");
    expect(content).toContain("sendPromptFromGui");
    expect(content).toContain("guiPromptSubmitted");
    expect(content).toContain("guiSessionOpenAfterInputClick");
    expect(content).toContain("clickCodingWorkbenchRecovery");
    expect(content).toContain("codingRecoveryEvidence");
    expect(content).toContain("waitForCodingRecoveryGuiTerminal");
    expect(content).toContain('logStage("wait-gui-recovery-terminal")');
    expect(content).toContain('logStage("verify-session-after-recovery")');
    expect(content).toContain("open-workbench-after-recovery");
    expect(content).toContain("guiSessionOpenAfterRecovery");
    expect(content).toContain("guiRecoveryTerminal");
    expect(content).toContain("workbenchAfterRecovery");
    const recoveryFlow = content.slice(
      content.indexOf('logStage("wait-recovery-read-model")'),
      content.indexOf('logStage("open-workbench-after-recovery")'),
    );
    expect(recoveryFlow).not.toContain("openFixtureSessionFromSidebar");
    expect(content).toContain("coding-workbench-recovery-submit");
    expect(content).toContain("coding_workbench_recovery");
    expect(content).toContain("CODING_RECOVERY_PROMPT_INTRO");
    expect(content).toContain("isCodingRecoveryPromptText");
    expect(content).toContain("const isRecoveryTurn =");
    expect(content).not.toContain(
      'if (metadata.harness && typeof metadata.harness === "object")',
    );
    expect(content).toContain("readTurnStartApplicationMetadata");
    expect(content).toContain("readTurnStartInputText");
    expect(content).toContain("readApplicationContextValue(");
    expect(content).toContain("message?.params?.additionalContext");
    expect(content).toContain('"metadata"');
    expect(content).toContain('entry.kind !== "application"');
    expect(content).not.toContain("runtimeOptions");
    expect(content).not.toContain("runtimeRequest");
    const codingProjectionHelpers = content.slice(
      content.indexOf("function hasCodingProjection(readResult)"),
      content.indexOf("async function ensureDefaultWorkspace"),
    );
    expect(codingProjectionHelpers).not.toContain(
      "JSON.stringify(readResult || {})",
    );
    expect(content).toContain('item?.type === "fileChange"');
    expect(content).toContain('item?.type === "commandExecution"');
    expect(content).toContain('item.status === "completed"');
    expect(content).toContain("item.exitCode === 0");
    expect(content).toContain("codingRecoveryGuiSubmitted");
    expect(content).toContain("codingRecoveryReachedBackend");
    expect(content).toContain("codingRecoveryTraceWire");
    expect(content).toContain("codingRecoveryReadCompleted");
    expect(content).not.toContain("capturedRecoveryContext");
    expect(content).toContain("appServerJsonRpcObserved");
    expect(content).toContain("backendTurnStartObserved");
    expect(content).toContain("canonicalSessionIdentity:");
    expect(content).toContain("codingRecoveryCanonicalIdentity:");
    expect(content).toContain("codingRecoveryTraceThreadIdentity:");
    expect(content).toContain("assertNoRendererErrors");
    expect(content).toContain("Electron renderer console error");
    expect(content).toContain("Electron renderer page error");
    expect(content).toContain(
      "assertNoRendererErrors(consoleErrors, pageErrors)",
    );
    expect(content).toContain("lime:agent-runtime-sessions-changed");
    expect(content).toContain('reason: "external"');
    expect(content).toContain("workspaceId");
    expect(content).toContain("codingChangesEvidencePresent");
    expect(content).toContain("codingOutputsEvidencePresent");
    expect(content).toContain("codingLogsEvidencePresent");
    expect(content).toContain("data-canvas-tab-key");
    expect(content).toContain("canvas-workbench-panel-changes");
    expect(content).toContain("canvas-workbench-panel-outputs");
    expect(content).toContain("canvas-workbench-panel-logs");
    expect(content).toContain("canonicalFileItem");
    expect(content).toContain("canonicalCommandItem");
    expect(content).not.toContain('type: "file.changed"');
    expect(content).not.toContain('type: "patch.started"');
    expect(content).not.toContain('type: "patch.applied"');
    expect(content).not.toContain('type: "command.started"');
    expect(content).not.toContain('type: "command.output"');
    expect(content).not.toContain('type: "command.exited"');
    expect(content).not.toContain('type: "test.started"');
    expect(content).not.toContain('type: "test.completed"');
    expect(content).toContain("historicalOperationalDetailsHidden");
    const historicalDetailsAssertion = content.slice(
      content.indexOf("historicalOperationalDetailsHidden,"),
      content.indexOf(
        "noInvokeErrors:",
        content.indexOf("historicalOperationalDetailsHidden,"),
      ),
    );
    expect(historicalDetailsAssertion).not.toContain(
      "toolTimelineProjectionPersisted",
    );
    expect(content).toContain(
      "timelineProcessEvidence?.toolCallRowCount === 0",
    );
    expect(content).toContain('type: "artifact.snapshot"');
    expect(content).toContain('type: "turn.completed"');
    expect(content).toContain(
      'const APP_SERVER_METHOD_ARTIFACT_WRITE = "artifact/write"',
    );
    expect(content).toContain("waitForCodeArtifactTurnTerminal");
    expect(content).toContain("writeArtifactFromCurrentRenderer");
    expect(content).toContain('"/src/lib/api/appServer.ts"');
    expect(content).toContain(".createAppServerClient()");
    expect(content).toContain(".writeArtifact(params)");
    expect(content).toContain("findArtifactWriteElectronIpcTrace");
    expect(content).toContain("waitForArtifactWriteElectronIpcTrace");
    expect(content).toContain(
      "entry?.command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND",
    );
    expect(content).toContain('entry?.transport !== "electron-ipc"');
    expect(content).toContain(
      "message?.method !== APP_SERVER_METHOD_ARTIFACT_WRITE",
    );
    expect(content).toContain("typedArtifactWriteUsed:");
    expect(content).toContain("typedArtifactWriteElectronIpcTrace:");
    expect(content).toContain(
      "summary.sessionCreation?.artifactWriteIpcTrace?.threadId ===",
    );
    expect(content).toContain(
      "summary.sessionCreation?.artifactWriteIpcTrace?.turnId === TURN_ID",
    );
    expect(content).toContain(
      "summary.sessionCreation?.artifactWriteIpcTrace?.artifactRef ===",
    );
    expect(content).toContain("directSessionBackendDidNotInjectArtifact:");
    expect(content).toContain(
      'process.env.CODE_ARTIFACT_WORKBENCH_FIXTURE_SCENARIO !== "direct-session"',
    );
    expect(content.indexOf("waitForCodeArtifactTurnTerminal")).toBeLessThan(
      content.indexOf(
        "const artifactWrite = await writeArtifactFromCurrentRenderer(",
      ),
    );
    expect(content).not.toContain('type: "turn.final_done"');
    expect(content).toContain('kind: "backendEvents"');
    expect(content).toContain("backendEmittedEventTypes");
    expect(content).toContain("backendEmittedCurrentTerminal");
    expect(content).toContain("backendDidNotEmitLegacyTerminal");
    expect(content).toContain("canonicalItemLifecycleClean");
    expect(content).toContain(
      'items.filter((item) => item?.type === "agentMessage").length === 1',
    );
    expect(content).toContain("new Set(itemIds).size === itemIds.length");
    expect(content).toContain(
      'items.every((item) => item?.status !== "inProgress")',
    );
    expect(content).toContain("Hello Lime Workbench");
    expect(content).toContain("CODE_ARTIFACT_WORKBENCH_DONE");
    expect(content).toContain("waitForFixtureSessionOpenedFromSidebar");
    expect(content).not.toContain('"lime:task-center:open-task"');
    expect(content).toContain("openWorkbench");
    expect(content).toContain("hasUserPrompt");
    expect(content).toContain("function hasHydratedSessionSnapshot(snapshot)");
    expect(content).not.toContain("hasGuiCodingInputHydratedSession");
    expect(content).toContain(
      "guiHydratedSession: hasHydratedSessionSnapshot(",
    );
    expect(content).not.toContain("hasToolTimelineText");
    expect(content).toContain("hasTaskCenterShell");
    expect(content).toContain("hasTaskCenterWorkbenchTab");
    expect(content).toContain("task-center-chrome-shell");
    expect(content).toContain("task-center-tab-workbench");
    expect(content).toContain("theme-workbench-harness-toggle");
    expect(content).toContain("general-workbench-sidebar");
    expect(content).toContain("hasWorkbenchEntry");
    expect(content).toContain("artifact-workbench-shell");
    expect(content).toContain("canvas-workbench-shell");
    expect(content).toContain("canvas-workbench-layout");
    expect(content).toContain("代码产物会话未在 GUI 中完成 hydrate:");
    expect(content).toContain("canvas-workbench-panel-");
    expect(content).toContain("visibleWorkbenchRoot");
    expect(content).toContain(
      'root.querySelectorAll(`[data-canvas-tab-key="${key}"]`)',
    );
    expect(content).toContain(
      "root.querySelectorAll('[data-canvas-tab-key=\"outputs\"]')",
    );
    expect(content).toContain(
      "root.querySelectorAll('[data-testid=\"canvas-workbench-panel-outputs\"]')",
    );
    expect(content).not.toContain(
      'document.querySelectorAll(`[data-canvas-tab-key="${key}"]`)',
    );
    expect(content).not.toContain(
      "document.querySelectorAll('[data-canvas-tab-key=\"outputs\"]')",
    );
    expect(content).not.toContain("const sourceText =");
    expect(content).toContain('window.dispatchEvent(new Event("focus"))');
  });

  it("keeps recovery execution identities turn-scoped while preserving failure source refs", () => {
    const content = readSmokeScript();

    expect(content).toContain("const turnScopedExecutionId = (baseId) =>");
    expect(content).toContain(
      'isRecoveryTurn ? baseId + ":" + turnId : baseId',
    );
    expect(content).toContain(
      'const assistantItemId = turnScopedExecutionId("code-artifact-workbench-electron:assistant")',
    );
    expect(content).toContain(
      'const toolCallId = turnScopedExecutionId("${TOOL_CALL_ID}")',
    );
    expect(content).toContain(
      'const fileChangeItemId = turnScopedExecutionId("${CODING_ARTIFACT_ID}")',
    );
    expect(content).toContain(
      'const commandId = turnScopedExecutionId("${CODING_COMMAND_ID}")',
    );
    expect(content).not.toContain("const patchId = turnScopedExecutionId");
    expect(content).not.toContain("const testRunId = turnScopedExecutionId");
    expect(content).toContain("assistantItemId,");
    expect(content).toContain("itemId: toolCallId");
    expect(content).toContain("call_id: toolCallId");
    expect(content).toContain("itemId: fileChangeItemId");
    expect(content).toContain("executionIds: {");
    expect(content).toContain("recoveryExecutionIdsTurnScoped:");
    expect(content).toContain("recoveryExecutionIds.length === 4");
    expect(content).toContain(
      "executionId.endsWith(`:${backendRecoveryTurnStart.turnId}`)",
    );
    expect(content).not.toContain('itemId: "${TOOL_CALL_ID}"');
    expect(content).not.toContain('patchId: "${CODING_PATCH_ID}"');
    expect(content).not.toContain('commandId: "${CODING_COMMAND_ID}"');
    expect(content).not.toContain('testRunId: "${CODING_TEST_RUN_ID}"');
    expect(content).toContain("guiRecoveryCommandId.length > 0");
    expect(content).toContain("guiRecoveryTestRunId.length > 0");
    expect(content).toContain("guiRecoveryCommandId");
    expect(content).toContain("guiRecoveryTestRunId");
  });

  it("drives FileChange decline and cancel through the typed server request", () => {
    const content = readSmokeScript();

    expect(content).toContain("FILE_CHANGE_BATCH_SCENARIO");
    expect(content).toContain("renderFileChangeGateBBackendScript()");
    expect(content).toContain("startFileChangeBatchTurnFromGui");
    expect(content).toContain('"lime:debug:claw-trace-enabled:v1"');
    expect(content).toContain(
      '"lime:debug:app-server-server-request-lifecycle:v1"',
    );
    expect(content).toContain("fileChangeBatchLifecycleTraceEnabled");
    expect(content).toContain('logStage("file-change-batch-decline")');
    expect(content).toContain('logStage("file-change-batch-cancel")');
    expect(content).toContain("waitForFileChangeApprovalPending");
    expect(content).toContain("clickFileChangeApprovalDecision");
    expect(content).toContain("waitForFileChangeTerminalReadModel");
    expect(content).toContain("waitForFileChangeTerminalGui");
    expect(content).toContain("declineTurnCompleted");
    expect(content).toContain("cancelTurnInterrupted");
    expect(content).toContain("pendingCleared");
  });

  it("always replaces backend ledger evidence on a failed Gate B run", () => {
    const content = readSmokeScript();
    const catchStart = content.lastIndexOf("  } catch (error) {");
    const finallyStart = content.indexOf("  } finally {", catchStart);
    const failureBranch = content.slice(catchStart, finallyStart);

    expect(content).toContain("function persistBackendLedgerEvidence(");
    expect(failureBranch).toContain("persistBackendLedgerEvidence(");
    expect(failureBranch).toContain("summary.backendKinds =");
    expect(failureBranch).toContain("summary.backendEmittedEventTypes =");
    expect(failureBranch.indexOf("persistBackendLedgerEvidence(")).toBeLessThan(
      failureBranch.indexOf("writeJsonFile(summaryPath, summary)"),
    );
  });

  it("does not use legacy commands or renderer mock fallback as success evidence", () => {
    const content = readSmokeScript();

    expect(content).not.toContain("agent_runtime_");
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
    expect(content).not.toContain("safeInvoke(");
  });
});
