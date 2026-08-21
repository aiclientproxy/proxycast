import { APP_SERVER_METHOD_SESSION_TURN_CANCEL } from "./claw-chat-current-fixture-constants.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

const CANCELED_TURN_STATUSES = new Set([
  "interrupted",
  "canceled",
  "cancelled",
]);

export function providerRequestCount(providerFixture) {
  return providerFixture.requests.filter(
    (request) => request?.path === "/v1/chat/completions",
  ).length;
}

function unfinishedProviderResponseClose(providerFixture) {
  return providerFixture.connectionDiagnostics.find(
    (entry) =>
      entry?.event === "response-close" &&
      entry?.path === "/v1/chat/completions" &&
      entry?.responseFinished === false,
  );
}

export async function waitForProviderCancellation(providerFixture, options) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < options.timeoutMs) {
    const closed = unfinishedProviderResponseClose(providerFixture);
    if (closed) {
      return {
        event: closed.event,
        requestId: closed.requestId,
        requestComplete: closed.requestComplete,
        responseFinished: closed.responseFinished,
      };
    }
    await sleep(options.intervalMs);
  }
  return null;
}

export async function readInvokeDiagnostics(page) {
  return await page.evaluate(() => {
    const readArray = (key) => {
      try {
        const value = JSON.parse(window.localStorage.getItem(key) || "[]");
        return Array.isArray(value) ? value : [];
      } catch {
        return [];
      }
    };
    const traces = readArray("lime_invoke_trace_buffer_v1");
    const calls = traces.flatMap((entry) => {
      if (entry?.command !== "app_server_handle_json_lines") return [];
      const lines = Array.isArray(entry?.args_preview?.request?.lines)
        ? entry.args_preview.request.lines
        : [];
      return lines.flatMap((line) => {
        try {
          const message = typeof line === "string" ? JSON.parse(line) : line;
          if (typeof message?.method !== "string") return [];
          return [
            {
              method: message.method,
              threadId: String(message?.params?.threadId || ""),
              turnId: String(message?.params?.turnId || ""),
              transport: String(entry?.transport || ""),
              status: String(entry?.status || ""),
            },
          ];
        } catch {
          return [];
        }
      });
    });
    const mockFallbackHitCount = traces.filter((entry) => {
      if (entry?.mock === true || entry?.mockFallback === true) return true;
      return [
        entry?.transport,
        entry?.source,
        entry?.fallback,
        entry?.fallbackMode,
      ].some(
        (value) =>
          typeof value === "string" && value.toLowerCase().includes("mock"),
      );
    }).length;
    return {
      calls,
      invokeErrorCount: readArray("lime_invoke_error_buffer_v1").length,
      mockFallbackHitCount,
      browserNavigateCount: traces.filter(
        (entry) => entry?.command === "browser_tab_navigate",
      ).length,
    };
  });
}

async function interruptActiveTurnFromGui(page, options, identity) {
  const stopButton = page
    .locator('[data-testid="inputbar-core-container"] button')
    .filter({ has: page.locator("svg.lucide-square") });
  await stopButton
    .first()
    .waitFor({ state: "visible", timeout: options.timeoutMs });
  if (
    (await stopButton.count()) !== 1 ||
    !(await stopButton.first().isEnabled())
  ) {
    throw new Error("运行中的 Agent 输入栏未显示唯一可用的停止按钮");
  }
  await stopButton.first().click();

  const startedAt = Date.now();
  while (Date.now() - startedAt < options.timeoutMs) {
    const diagnostics = await readInvokeDiagnostics(page);
    const call = diagnostics.calls.find(
      (candidate) =>
        candidate.method === APP_SERVER_METHOD_SESSION_TURN_CANCEL &&
        candidate.threadId === identity.threadId &&
        candidate.turnId === identity.turnId,
    );
    if (call) return { call, diagnostics };
    await sleep(options.intervalMs);
  }
  throw new Error("GUI 停止操作未产生 turn/interrupt Electron IPC trace");
}

export function buildCancelAssertions({
  activeTurnId,
  agentControlled,
  consoleErrors,
  debuggerAfterInterrupt,
  debuggerBeforeInterrupt,
  finalMarker,
  guiAfterInterrupt,
  identity,
  interruptCall,
  invokeDiagnostics,
  pageErrors,
  providerAfterInterrupt,
  providerBeforeInterrupt,
  providerCancellation,
  released,
  terminalStatus,
  terminalTurnId,
}) {
  return {
    agentControlledBeforeInterrupt:
      agentControlled?.controlOwner === "agent" &&
      agentControlled?.activeTurnId === activeTurnId,
    debuggerAttachedBeforeInterrupt:
      debuggerBeforeInterrupt?.exists === true &&
      debuggerBeforeInterrupt?.attached === true &&
      debuggerBeforeInterrupt?.webContentsId === agentControlled?.webContentsId,
    interruptRequestedWithCanonicalIdentity:
      interruptCall?.method === APP_SERVER_METHOD_SESSION_TURN_CANCEL &&
      interruptCall?.threadId === identity?.threadId &&
      interruptCall?.turnId === activeTurnId,
    interruptUsedCurrentElectronBridge:
      interruptCall?.transport === "electron-ipc" &&
      interruptCall?.status === "success" &&
      interruptCall?.threadId === identity?.threadId &&
      interruptCall?.turnId === activeTurnId,
    interruptedTurnProjected:
      terminalTurnId === activeTurnId &&
      CANCELED_TURN_STATUSES.has(String(terminalStatus || "").toLowerCase()),
    browserReleasedAfterInterrupt:
      released?.controlOwner === "released" &&
      released?.activeTurnId === null &&
      released?.sessionId === agentControlled?.sessionId &&
      released?.tabId === agentControlled?.tabId,
    sameWebContentsRetainedForUser:
      released?.webContentsId === agentControlled?.webContentsId &&
      guiAfterInterrupt?.webContentsId === agentControlled?.webContentsId,
    debuggerDetachedAfterInterrupt:
      debuggerAfterInterrupt?.exists === true &&
      debuggerAfterInterrupt?.attached === false &&
      debuggerAfterInterrupt?.webContentsId === agentControlled?.webContentsId,
    providerRequestPendingAtInterrupt:
      providerBeforeInterrupt?.requestCount >= 4 &&
      providerBeforeInterrupt?.unfinishedResponseCloseCount === 0,
    pendingProviderRequestCanceled:
      providerCancellation?.event === "response-close" &&
      providerCancellation?.responseFinished === false,
    noProviderContinuationAfterInterrupt:
      providerAfterInterrupt?.requestCount ===
      providerBeforeInterrupt?.requestCount,
    cancelSkippedLifecycleTail:
      invokeDiagnostics?.browserNavigateCount === 0 &&
      !String(guiAfterInterrupt?.bodyText || "").includes(finalMarker),
    productionMockFallbackZero: invokeDiagnostics?.mockFallbackHitCount === 0,
    invokeErrorsZero: invokeDiagnostics?.invokeErrorCount === 0,
    consoleAndPageErrorsZero:
      consoleErrors?.length === 0 && pageErrors?.length === 0,
  };
}

export async function runBrowserCancelScenario({
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
  waitForTerminalThread,
}) {
  logStage("interrupt-active-browser-turn");
  const debuggerBeforeInterrupt = await readBrowserDebuggerState(
    app,
    agentControlled.webContentsId,
  );
  const providerBeforeInterrupt = {
    requestCount: providerRequestCount(providerFixture),
    unfinishedResponseCloseCount: providerFixture.connectionDiagnostics.filter(
      (entry) =>
        entry?.event === "response-close" &&
        entry?.path === "/v1/chat/completions" &&
        entry?.responseFinished === false,
    ).length,
  };
  const interruptedFromGui = await interruptActiveTurnFromGui(page, options, {
    threadId: identity.threadId,
    turnId: activeTurnId,
  });
  const interruptCall = interruptedFromGui.call;

  logStage("wait-interrupted-read-model-and-browser-release");
  const terminal = await waitForTerminalThread(
    page,
    options,
    identity.threadId,
    requestLog,
  );
  const terminalTurnId = terminal.turn?.id || terminal.turn?.turnId || null;
  const terminalStatus = terminal.turn?.status || terminal.turn?.state || null;
  const released = await waitForBrowserWorkspaceState(
    page,
    options,
    (state) =>
      state.controlOwner === "released" &&
      state.activeTurnId === null &&
      state.webContentsId === agentControlled.webContentsId,
    "turn/interrupt 后 Browser 用户 tab 未 release",
  );
  const debuggerAfterInterrupt = await readBrowserDebuggerState(
    app,
    released.webContentsId,
  );
  const providerCancellation = await waitForProviderCancellation(
    providerFixture,
    options,
  );
  const providerAfterInterrupt = {
    requestCount: providerRequestCount(providerFixture),
  };
  const invokeDiagnostics = await readInvokeDiagnostics(page);
  const guiAfterInterrupt = {
    ...(await readBrowserWorkspaceState(page)),
    bodyText: await page.evaluate(() => document.body?.innerText || ""),
  };
  const assertions = buildCancelAssertions({
    activeTurnId,
    agentControlled,
    consoleErrors,
    debuggerAfterInterrupt,
    debuggerBeforeInterrupt,
    finalMarker,
    guiAfterInterrupt,
    identity,
    interruptCall,
    invokeDiagnostics,
    pageErrors,
    providerAfterInterrupt,
    providerBeforeInterrupt,
    providerCancellation,
    released,
    terminalStatus,
    terminalTurnId,
  });
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  const { bodyText: _bodyText, ...guiAfterInterruptEvidence } =
    guiAfterInterrupt;

  return sanitizeJson({
    schemaVersion: "lime.browser_runtime_electron_gate_b.cancel.v1",
    scenario: "cancel",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    claimBoundary:
      "真实 Electron WebContentsView、preload IPC、App Server turn/interrupt、RuntimeCore/read model 与 Browser terminal cleanup 的本地闭环；不代表 live provider、disconnect、window close 或跨平台打包验证。",
    identity,
    gui: {
      beforeTurn: guiBeforeTurn,
      agentControlled,
      afterInterrupt: guiAfterInterruptEvidence,
    },
    interrupt: {
      method: interruptCall?.method || null,
      params: {
        threadId: interruptCall?.threadId || null,
        turnId: interruptCall?.turnId || null,
      },
      bridgeCall: interruptCall || null,
    },
    terminal: {
      status: terminalStatus,
      turnId: terminalTurnId,
    },
    provider: {
      model: providerFixture.provider?.model || null,
      beforeInterrupt: providerBeforeInterrupt,
      afterInterrupt: providerAfterInterrupt,
      cancellation: providerCancellation,
    },
    lifecycle: {
      initial: {
        activeTurnId: initial.state.activeTurnId,
        pageRevision: initial.observation.pageRevision,
        tabId: initial.state.tabId,
        webContentsId: initial.state.webContentsId,
      },
      interrupted: {
        controlOwner: released.controlOwner,
        debuggerAttachedBefore: debuggerBeforeInterrupt.attached,
        debuggerAttachedAfter: debuggerAfterInterrupt.attached,
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
