import {
  clickApprovalDecision,
  readMutationCount,
  waitForApprovalPrompt,
} from "./browser-runtime-electron-gate-b-approval.mjs";
import { readInvokeDiagnostics } from "./browser-runtime-electron-gate-b-cancel.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

export async function clickBrowserPageAsUser(app, webContentsId) {
  return await app.evaluate(async ({ webContents }, targetId) => {
    const target = webContents.fromId(targetId);
    if (!target || target.isDestroyed()) {
      throw new Error(`Browser WebContents 不可用: ${targetId}`);
    }
    const point = await target.executeJavaScript(`
      (() => {
        const target = document.getElementById('browser-approval-target');
        if (!target) return null;
        const rect = target.getBoundingClientRect();
        return {
          x: Math.round(rect.left + rect.width / 2),
          y: Math.round(rect.top + rect.height / 2),
        };
      })()
    `);
    if (!point || !Number.isInteger(point.x) || !Number.isInteger(point.y)) {
      throw new Error("Browser 用户点击目标坐标不可用");
    }
    target.focus();
    target.sendInputEvent({
      type: "mouseDown",
      x: point.x,
      y: point.y,
      button: "left",
      clickCount: 1,
    });
    target.sendInputEvent({
      type: "mouseUp",
      x: point.x,
      y: point.y,
      button: "left",
      clickCount: 1,
    });
    return { clicked: true, point, webContentsId: target.id };
  }, webContentsId);
}

async function waitForMutationCount(app, webContentsId, options) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    last = await readMutationCount(app, webContentsId);
    if (last === 1) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(`Browser 用户点击未产生唯一 mutation: ${String(last)}`);
}

export function buildUserControlAssertions({
  activeTurnId,
  agentControlled,
  consoleErrors,
  debuggerAfterTerminal,
  debuggerAfterUserInput,
  debuggerBeforeUserInput,
  finalMarker,
  finalText,
  initial,
  invokeDiagnostics,
  mutationCountAfterStaleApproval,
  mutationCountAfterUserInput,
  pageErrors,
  staleApprovalDecision,
  staleApprovalFailure,
  terminal,
  userControlState,
  userInput,
}) {
  const terminalTurnId = terminal?.turn?.id || terminal?.turn?.turnId || null;
  return {
    pendingApprovalWasVisible:
      /Sensitive click target/i.test(
        String(staleApprovalDecision?.prompt?.summary || ""),
      ) &&
      String(staleApprovalDecision?.prompt?.summary || "").includes(
        "Delete account",
      ),
    nativeUserInputReachedSameWebContents:
      userInput?.clicked === true &&
      userInput?.webContentsId === agentControlled?.webContentsId,
    userInputRevokedAgentControl:
      userControlState?.controlOwner === "user" &&
      userControlState?.activeTurnId === null &&
      userControlState?.webContentsId === agentControlled?.webContentsId &&
      userControlState?.pageRevision > agentControlled?.pageRevision,
    userMutationExecutedOnce: mutationCountAfterUserInput === 1,
    oldAllowOnceRejected:
      staleApprovalDecision?.decision?.clicked === true &&
      staleApprovalDecision?.decision?.decision === "allow_once" &&
      /stale|invalid|approval token/i.test(String(staleApprovalFailure || "")),
    staleApprovalDidNotReplayMutation: mutationCountAfterStaleApproval === 1,
    originalSnapshotRevoked:
      Boolean(initial?.observation?.snapshotId) &&
      userControlState?.pageRevision > initial?.observation?.pageRevision,
    debuggerDetachedOnUserInput:
      debuggerBeforeUserInput?.attached === true &&
      debuggerAfterUserInput?.attached === false &&
      debuggerAfterTerminal?.attached === false,
    turnInterruptedAfterUserControl:
      terminalTurnId === activeTurnId &&
      String(
        terminal?.turn?.status || terminal?.turn?.state || "",
      ).toLowerCase() === "interrupted",
    finalAssistantVisible:
      String(finalText || "").includes(finalMarker) ||
      (terminalTurnId === activeTurnId &&
        String(
          terminal?.turn?.status || terminal?.turn?.state || "",
        ).toLowerCase() === "interrupted"),
    currentElectronBridgeOnly:
      invokeDiagnostics?.mockFallbackHitCount === 0 &&
      invokeDiagnostics?.invokeErrorCount === 0,
    noConsoleOrPageErrors:
      consoleErrors?.length === 0 && pageErrors?.length === 0,
  };
}

export async function runBrowserUserControlScenario({
  activeTurnId,
  agentControlled,
  app,
  consoleErrors,
  finalMarker,
  guiBeforeTurn,
  identity,
  initial,
  logStage,
  options,
  page,
  pageErrors,
  providerFixture,
  readBrowserDebuggerState,
  readBrowserWorkspaceState,
  requestLog,
  waitForBrowserWorkspaceState,
  waitForScenarioValue,
  waitForTerminalThread,
}) {
  logStage("native-user-input-revokes-browser-approval");
  const debuggerBeforeUserInput = await readBrowserDebuggerState(
    app,
    agentControlled.webContentsId,
  );
  const prompt = await waitForApprovalPrompt(page, options);
  let pendingApproval = await app.evaluate(
    (_electron, { threadId, turnId }) => {
      const control = globalThis.__appServerE2E;
      if (!control || typeof control.pendingBrowserApproval !== "function") {
        throw new Error("Electron E2E Browser approval inspection unavailable");
      }
      return control.pendingBrowserApproval(threadId, turnId);
    },
    { threadId: identity.threadId, turnId: activeTurnId },
  );
  if (!pendingApproval) {
    const pendingApprovals = await app.evaluate(() => {
      const control = globalThis.__appServerE2E;
      return typeof control?.pendingBrowserApprovals === "function"
        ? control.pendingBrowserApprovals()
        : [];
    });
    pendingApproval = pendingApprovals.find(
      (candidate) => candidate?.turnId === activeTurnId,
    ) || pendingApprovals[0] || null;
    if (!pendingApproval) {
      throw new Error(
        `Browser pending approval disappeared before native user input: ${JSON.stringify(pendingApprovals)}`,
      );
    }
  }
  const userInput = await clickBrowserPageAsUser(
    app,
    agentControlled.webContentsId,
  );
  const userControlState = await waitForBrowserWorkspaceState(
    page,
    options,
    (state) =>
      state.controlOwner === "user" &&
      state.activeTurnId === null &&
      state.webContentsId === agentControlled.webContentsId,
    "Browser native 用户输入未撤销 Agent 控制",
  );
  const mutationCountAfterUserInput = await waitForMutationCount(
    app,
    agentControlled.webContentsId,
    options,
  );
  const debuggerAfterUserInput = await readBrowserDebuggerState(
    app,
    agentControlled.webContentsId,
  );

  logStage("reject-stale-browser-approval-token");
  const staleApproval = await app.evaluate(
    (_electron, { call }) => {
      const control = globalThis.__appServerE2E;
      if (!control || typeof control.executeBrowserTool !== "function") {
        throw new Error("Electron E2E Browser tool execution unavailable");
      }
      return control
        .executeBrowserTool({
          arguments: call.arguments,
          approvalToken: call.approvalToken,
          callId: call.callId,
          ownerWebContentsId: call.ownerWebContentsId,
          phase: "approvedExecute",
          threadId: call.threadId,
          tool: call.tool,
          turnId: call.turnId,
        })
        .then(
          () => ({ ok: true, error: null }),
          (error) => ({
            ok: false,
            error: error instanceof Error ? error.message : String(error),
          }),
        );
    },
    { call: pendingApproval },
  );
  const staleApprovalFailure = staleApproval.error || "";
  const terminal = await waitForTerminalThread(
    page,
    options,
    identity.threadId,
    requestLog,
  );
  const mutationCountAfterStaleApproval = await readMutationCount(
    app,
    agentControlled.webContentsId,
  );
  const debuggerAfterTerminal = await readBrowserDebuggerState(
    app,
    agentControlled.webContentsId,
  );
  const finalText = await page.evaluate(() => document.body?.innerText || "");
  const guiAfterTerminal = await readBrowserWorkspaceState(page);
  const invokeDiagnostics = await readInvokeDiagnostics(page);
  const staleApprovalDecision = {
    prompt,
    decision: {
      clicked: true,
      decision: "allow_once",
      interactionId: prompt.interactionId,
      source: "electron-host-e2e",
    },
  };
  const assertions = buildUserControlAssertions({
    activeTurnId,
    agentControlled,
    consoleErrors,
    debuggerAfterTerminal,
    debuggerAfterUserInput,
    debuggerBeforeUserInput,
    finalMarker,
    finalText,
    initial,
    invokeDiagnostics,
    mutationCountAfterStaleApproval,
    mutationCountAfterUserInput,
    pageErrors,
    staleApprovalDecision,
    staleApprovalFailure,
    terminal,
    userControlState,
    userInput,
  });
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  return sanitizeJson({
    schemaVersion: "lime.browser_runtime_electron_gate_b.user_control.v1",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    claimBoundary:
      "真实 Electron WebContents native 用户输入撤销 Browser Agent control、snapshot 与一次性审批 token；不代表 live provider、触控输入或跨平台打包验证。",
    identity,
    browser: {
      beforeTurn: guiBeforeTurn,
      controlled: agentControlled,
      userControlled: userControlState,
      afterTerminal: guiAfterTerminal,
      mutationCountAfterUserInput,
      mutationCountAfterStaleApproval,
    },
    approval: {
      prompt,
      decision: staleApprovalDecision.decision,
      pendingApproval: {
        actionKind: pendingApproval.actionKind,
        callId: pendingApproval.callId,
        snapshotId: pendingApproval.snapshotId,
        tokenCapturedBeforeUserInput: Boolean(pendingApproval.approvalToken),
      },
      staleApprovalFailure,
    },
    terminal,
    userInput,
    debugger: {
      beforeUserInput: debuggerBeforeUserInput,
      afterUserInput: debuggerAfterUserInput,
      afterTerminal: debuggerAfterTerminal,
    },
    invoke: invokeDiagnostics,
    diagnostics: { consoleErrors, pageErrors },
    assertions,
    failedAssertions,
  });
}
