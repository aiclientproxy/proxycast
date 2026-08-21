import {
  providerRequestCount,
  readInvokeDiagnostics,
  waitForProviderCancellation,
} from "./browser-runtime-electron-gate-b-cancel.mjs";
import { sanitizeJson } from "./claw-chat-current-fixture-utils.mjs";

async function terminateAppServerSidecar(app) {
  return await app.evaluate(() => {
    const control = globalThis.__appServerE2E;
    if (!control || typeof control.terminateSidecar !== "function") {
      return { available: false };
    }
    return { available: true, ...control.terminateSidecar() };
  });
}

export function buildDisconnectAssertions({
  activeTurnId,
  agentControlled,
  consoleErrors,
  debuggerAfterDisconnect,
  debuggerBeforeDisconnect,
  finalMarker,
  guiAfterDisconnect,
  invokeDiagnostics,
  pageErrors,
  providerAfterDisconnect,
  providerBeforeDisconnect,
  providerCancellation,
  released,
  termination,
}) {
  return {
    agentControlledBeforeDisconnect:
      agentControlled?.controlOwner === "agent" &&
      agentControlled?.activeTurnId === activeTurnId,
    debuggerAttachedBeforeDisconnect:
      debuggerBeforeDisconnect?.exists === true &&
      debuggerBeforeDisconnect?.attached === true &&
      debuggerBeforeDisconnect?.webContentsId ===
        agentControlled?.webContentsId,
    realSidecarTerminationRequested:
      termination?.available === true &&
      termination?.requested === true &&
      termination?.signal === "SIGTERM" &&
      Number.isInteger(termination?.pid) &&
      termination.pid > 0,
    browserReleasedAfterDisconnect:
      released?.controlOwner === "released" &&
      released?.activeTurnId === null &&
      released?.sessionId === agentControlled?.sessionId &&
      released?.tabId === agentControlled?.tabId,
    sameWebContentsRetainedForUser:
      released?.webContentsId === agentControlled?.webContentsId &&
      guiAfterDisconnect?.webContentsId === agentControlled?.webContentsId,
    debuggerDetachedAfterDisconnect:
      debuggerAfterDisconnect?.exists === true &&
      debuggerAfterDisconnect?.attached === false &&
      debuggerAfterDisconnect?.webContentsId === agentControlled?.webContentsId,
    providerRequestPendingAtDisconnect:
      providerBeforeDisconnect?.requestCount >= 4 &&
      providerBeforeDisconnect?.unfinishedResponseCloseCount === 0,
    pendingProviderRequestCanceled:
      providerCancellation?.event === "response-close" &&
      providerCancellation?.responseFinished === false,
    noProviderContinuationAfterDisconnect:
      providerAfterDisconnect?.requestCount ===
      providerBeforeDisconnect?.requestCount,
    disconnectSkippedLifecycleTail:
      invokeDiagnostics?.browserNavigateCount === 0 &&
      !String(guiAfterDisconnect?.bodyText || "").includes(finalMarker),
    productionMockFallbackZero: invokeDiagnostics?.mockFallbackHitCount === 0,
    consoleAndPageErrorsZero:
      consoleErrors?.length === 0 && pageErrors?.length === 0,
  };
}

export async function runBrowserDisconnectScenario({
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
  waitForBrowserWorkspaceState,
}) {
  logStage("terminate-app-server-sidecar");
  const debuggerBeforeDisconnect = await readBrowserDebuggerState(
    app,
    agentControlled.webContentsId,
  );
  const providerBeforeDisconnect = {
    requestCount: providerRequestCount(providerFixture),
    unfinishedResponseCloseCount: providerFixture.connectionDiagnostics.filter(
      (entry) =>
        entry?.event === "response-close" &&
        entry?.path === "/v1/chat/completions" &&
        entry?.responseFinished === false,
    ).length,
  };
  const termination = await terminateAppServerSidecar(app);

  logStage("wait-browser-release-after-sidecar-disconnect");
  const released = await waitForBrowserWorkspaceState(
    page,
    options,
    (state) =>
      state.controlOwner === "released" &&
      state.activeTurnId === null &&
      state.webContentsId === agentControlled.webContentsId,
    "App Server sidecar 断连后 Browser 用户 tab 未 release",
  );
  const debuggerAfterDisconnect = await readBrowserDebuggerState(
    app,
    released.webContentsId,
  );
  const providerCancellation = await waitForProviderCancellation(
    providerFixture,
    options,
  );
  const providerAfterDisconnect = {
    requestCount: providerRequestCount(providerFixture),
  };
  const invokeDiagnostics = await readInvokeDiagnostics(page);
  const guiAfterDisconnect = {
    ...(await readBrowserWorkspaceState(page)),
    bodyText: await page.evaluate(() => document.body?.innerText || ""),
  };
  const assertions = buildDisconnectAssertions({
    activeTurnId,
    agentControlled,
    consoleErrors,
    debuggerAfterDisconnect,
    debuggerBeforeDisconnect,
    finalMarker,
    guiAfterDisconnect,
    invokeDiagnostics,
    pageErrors,
    providerAfterDisconnect,
    providerBeforeDisconnect,
    providerCancellation,
    released,
    termination,
  });
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  const { bodyText: _bodyText, ...guiAfterDisconnectEvidence } =
    guiAfterDisconnect;

  return sanitizeJson({
    schemaVersion: "lime.browser_runtime_electron_gate_b.disconnect.v1",
    scenario: "disconnect",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    claimBoundary:
      "真实 Electron App Server sidecar 进程退出、AppServerHost connectionLost、Browser 用户 tab release 与 debugger detach 的本地闭环；不代表 restart 后 turn 恢复、权限/下载、live provider 或跨平台打包验证。",
    identity,
    gui: {
      beforeTurn: guiBeforeTurn,
      agentControlled,
      afterDisconnect: guiAfterDisconnectEvidence,
    },
    sidecar: { termination },
    provider: {
      model: providerFixture.provider?.model || null,
      beforeDisconnect: providerBeforeDisconnect,
      afterDisconnect: providerAfterDisconnect,
      cancellation: providerCancellation,
    },
    lifecycle: {
      initial: {
        activeTurnId: initial.state.activeTurnId,
        pageRevision: initial.observation.pageRevision,
        tabId: initial.state.tabId,
        webContentsId: initial.state.webContentsId,
      },
      disconnected: {
        controlOwner: released.controlOwner,
        debuggerAttachedBefore: debuggerBeforeDisconnect.attached,
        debuggerAttachedAfter: debuggerAfterDisconnect.attached,
        tabId: released.tabId,
        webContentsId: released.webContentsId,
      },
    },
    invoke: invokeDiagnostics,
    assertions,
    failedAssertions,
    diagnostics: { consoleErrors, pageErrors },
  });
}
