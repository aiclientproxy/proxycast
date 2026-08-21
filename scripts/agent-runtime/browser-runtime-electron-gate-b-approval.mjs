import { readInvokeDiagnostics } from "./browser-runtime-electron-gate-b-cancel.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

export async function installBrowserApprovalPage(app, webContentsId) {
  return await app.evaluate(async ({ webContents }, targetId) => {
    const target = webContents.fromId(targetId);
    if (!target || target.isDestroyed()) {
      throw new Error(`Browser WebContents 不可用: ${targetId}`);
    }
    return await target.executeJavaScript(`
      (() => {
        document.title = "Browser Approval Gate B";
        document.body.innerHTML = [
          '<main style="font-family: sans-serif; padding: 32px">',
          '<h1>Account controls</h1>',
          '<button id="browser-approval-target" aria-label="Delete account">Delete account</button>',
          '<output id="browser-approval-count">0</output>',
          '</main>',
        ].join('');
        const button = document.getElementById('browser-approval-target');
        const output = document.getElementById('browser-approval-count');
        button.addEventListener('click', () => {
          const next = Number(output.textContent || '0') + 1;
          output.textContent = String(next);
          document.documentElement.dataset.browserApprovalMutationCount = String(next);
        });
        document.documentElement.dataset.browserApprovalMutationCount = '0';
        return {
          mutationCount: 0,
          title: document.title,
          url: location.href,
        };
      })()
    `);
  }, webContentsId);
}

export async function readMutationCount(app, webContentsId) {
  return await app.evaluate(async ({ webContents }, targetId) => {
    const target = webContents.fromId(targetId);
    if (!target || target.isDestroyed()) {
      return null;
    }
    return await target.executeJavaScript(
      "Number(document.documentElement.dataset.browserApprovalMutationCount || '0')",
    );
  }, webContentsId);
}

export async function waitForApprovalPrompt(
  page,
  options,
  previousInteractionId = null,
) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    last = await page.evaluate(() => {
      const layers = Array.from(
        document.querySelectorAll(
          '[data-testid="pending-interaction-layer"][data-interaction-kind="approval"]',
        ),
      );
      const layer = layers.at(-1) || null;
      const prompt = layer?.querySelector(
        '[data-testid="inputbar-approval-prompt"]',
      );
      const buttons = Array.from(layer?.querySelectorAll("button") || []).map(
        (button) => ({
          decision: button.getAttribute("data-decision") || "",
          disabled: button.disabled,
          text: button.textContent || "",
        }),
      );
      return {
        interactionId: layer?.getAttribute("data-interaction-id") || null,
        interactionKind: layer?.getAttribute("data-interaction-kind") || null,
        summary:
          prompt?.querySelector('[data-testid="inputbar-approval-summary"]')
            ?.textContent ||
          prompt?.textContent ||
          "",
        decisions: buttons.map((button) => button.decision).filter(Boolean),
        buttons,
      };
    });
    if (
      last?.interactionId &&
      last.interactionId !== previousInteractionId &&
      last.interactionKind === "approval" &&
      last.buttons.some(
        (button) => button.decision === "allow_once" && !button.disabled,
      ) &&
      last.buttons.some(
        (button) => button.decision === "decline" && !button.disabled,
      )
    ) {
      return sanitizeJson(last);
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Browser approval pending UI 未出现: ${JSON.stringify(sanitizeJson(last))}`,
  );
}

export async function clickApprovalDecision(page, interactionId, decision) {
  const result = await page.evaluate(
    ({ decision, interactionId }) => {
      const layer = Array.from(
        document.querySelectorAll(
          '[data-testid="pending-interaction-layer"][data-interaction-kind="approval"]',
        ),
      ).find(
        (candidate) =>
          candidate.getAttribute("data-interaction-id") === interactionId,
      );
      const button = Array.from(layer?.querySelectorAll("button") || []).find(
        (candidate) =>
          candidate.getAttribute("data-decision") === decision &&
          !candidate.disabled,
      );
      if (!button) {
        return { clicked: false, decision, interactionId };
      }
      button.click();
      return { clicked: true, decision, interactionId };
    },
    { decision, interactionId },
  );
  if (!result.clicked) {
    throw new Error(`Browser approval ${decision} 按钮不可用`);
  }
  return result;
}

export function buildApprovalAssertions({
  activeTurnId,
  agentControlled,
  approvedMutation,
  consoleErrors,
  debuggerAfterTerminal,
  debuggerBeforeApproval,
  declinedMutationFailure,
  firstDecision,
  firstPrompt,
  finalMarker,
  finalText,
  initial,
  invokeDiagnostics,
  mutationCountAfterApproval,
  mutationCountAfterDecline,
  pageErrors,
  released,
  secondDecision,
  secondObservation,
  secondPrompt,
  terminal,
}) {
  const firstDecisions = new Set(firstPrompt?.decisions || []);
  const secondDecisions = new Set(secondPrompt?.decisions || []);
  const onceOnly = (decisions) =>
    decisions.has("allow_once") &&
    decisions.has("decline") &&
    decisions.has("cancel") &&
    !decisions.has("allow_for_session");
  const terminalTurnId = terminal?.turn?.id || terminal?.turn?.turnId || null;
  return {
    dangerousTargetObserved:
      initial?.target?.name === "Delete account" &&
      Number.isInteger(initial?.target?.backendNodeId),
    canonicalApprovalPromptVisible:
      /Sensitive click target/i.test(String(firstPrompt?.summary || "")) &&
      String(firstPrompt?.summary || "").includes("Delete account"),
    browserApprovalIsOnceOnly: onceOnly(firstDecisions),
    approvedWithAllowOnce:
      firstDecision?.clicked === true &&
      firstDecision?.decision === "allow_once",
    approvedActionResumedSameTab:
      approvedMutation?.state?.tabId === agentControlled?.tabId &&
      approvedMutation?.state?.webContentsId ===
        agentControlled?.webContentsId &&
      approvedMutation?.state?.activeTurnId === activeTurnId,
    approvedMutationExecutedOnce: mutationCountAfterApproval === 1,
    freshSnapshotUsedForSecondAction:
      secondObservation?.observation?.snapshotId &&
      secondObservation.observation.snapshotId !==
        initial?.observation?.snapshotId &&
      secondObservation?.state?.webContentsId ===
        agentControlled?.webContentsId,
    secondApprovalHasNewIdentity:
      Boolean(secondPrompt?.interactionId) &&
      secondPrompt.interactionId !== firstPrompt?.interactionId &&
      onceOnly(secondDecisions),
    declinedWithoutMutation:
      secondDecision?.clicked === true &&
      secondDecision?.decision === "decline" &&
      /拒绝|declin|approval/i.test(String(declinedMutationFailure || "")) &&
      mutationCountAfterDecline === 1,
    turnCompletedAfterDecline:
      terminalTurnId === activeTurnId &&
      String(
        terminal?.turn?.status || terminal?.turn?.state || "",
      ).toLowerCase() === "completed",
    browserReleasedAfterTerminal:
      released?.controlOwner === "released" &&
      released?.activeTurnId === null &&
      released?.webContentsId === agentControlled?.webContentsId,
    debuggerLifecycleClosed:
      debuggerBeforeApproval?.attached === true &&
      debuggerAfterTerminal?.attached === false,
    finalAssistantVisible: String(finalText || "").includes(finalMarker),
    currentElectronBridgeOnly:
      invokeDiagnostics?.mockFallbackHitCount === 0 &&
      invokeDiagnostics?.invokeErrorCount === 0,
    noConsoleOrPageErrors:
      consoleErrors?.length === 0 && pageErrors?.length === 0,
  };
}

export async function runBrowserApprovalScenario({
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
  logStage("approve-sensitive-browser-action");
  const debuggerBeforeApproval = await readBrowserDebuggerState(
    app,
    agentControlled.webContentsId,
  );
  const firstPrompt = await waitForApprovalPrompt(page, options);
  const firstDecision = await clickApprovalDecision(
    page,
    firstPrompt.interactionId,
    "allow_once",
  );
  const approvedMutation = await waitForScenarioValue(
    options,
    () => providerFixture.scenario.approvedMutation,
    "Browser action approval resume",
  );
  const mutationCountAfterApproval = await readMutationCount(
    app,
    agentControlled.webContentsId,
  );

  logStage("decline-second-sensitive-browser-action");
  const secondObservation = await waitForScenarioValue(
    options,
    () => providerFixture.scenario.secondObservation,
    "Browser fresh snapshot before decline",
  );
  const secondPrompt = await waitForApprovalPrompt(
    page,
    options,
    firstPrompt.interactionId,
  );
  const secondDecision = await clickApprovalDecision(
    page,
    secondPrompt.interactionId,
    "decline",
  );
  const declinedMutationFailure = await waitForScenarioValue(
    options,
    () => providerFixture.scenario.declinedMutationFailure,
    "Browser decline terminal",
  );
  const terminal = await waitForTerminalThread(
    page,
    options,
    identity.threadId,
    requestLog,
  );
  const released = await waitForBrowserWorkspaceState(
    page,
    options,
    (state) =>
      state.controlOwner === "released" &&
      state.activeTurnId === null &&
      state.webContentsId === agentControlled.webContentsId,
    "Browser approval turn terminal 后未 release",
  );
  const mutationCountAfterDecline = await readMutationCount(
    app,
    released.webContentsId,
  );
  const debuggerAfterTerminal = await readBrowserDebuggerState(
    app,
    released.webContentsId,
  );
  await page.waitForFunction(
    (marker) => document.body?.innerText.includes(marker),
    finalMarker,
    { timeout: options.timeoutMs },
  );
  const finalText = await page.evaluate(() => document.body?.innerText || "");
  const guiAfterTerminal = await readBrowserWorkspaceState(page);
  const invokeDiagnostics = await readInvokeDiagnostics(page);
  const assertions = buildApprovalAssertions({
    activeTurnId,
    agentControlled,
    approvedMutation,
    consoleErrors,
    debuggerAfterTerminal,
    debuggerBeforeApproval,
    declinedMutationFailure,
    firstDecision,
    firstPrompt,
    finalMarker,
    finalText,
    initial,
    invokeDiagnostics,
    mutationCountAfterApproval,
    mutationCountAfterDecline,
    pageErrors,
    released,
    secondDecision,
    secondObservation,
    secondPrompt,
    terminal,
  });
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  return sanitizeJson({
    schemaVersion: "lime.browser_runtime_electron_gate_b.approval.v1",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    claimBoundary:
      "真实 Electron WebContentsView、App Server dynamic-tool 两阶段请求、canonical action_required 与 GUI allow_once/decline；不代表 live provider 或跨平台打包验证。",
    identity,
    browser: {
      beforeTurn: guiBeforeTurn,
      controlled: agentControlled,
      released,
      afterTerminal: guiAfterTerminal,
      mutationCountAfterApproval,
      mutationCountAfterDecline,
    },
    approval: {
      firstPrompt,
      firstDecision,
      secondPrompt,
      secondDecision,
      declinedMutationFailure,
    },
    terminal,
    debugger: {
      beforeApproval: debuggerBeforeApproval,
      afterTerminal: debuggerAfterTerminal,
    },
    invoke: invokeDiagnostics,
    diagnostics: { consoleErrors, pageErrors },
    assertions,
    failedAssertions,
  });
}
