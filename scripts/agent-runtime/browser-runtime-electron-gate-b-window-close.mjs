import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

async function destroyBrowserOwnerWindow(app, windowId) {
  return await app.evaluate(
    async ({ BrowserWindow, ipcMain }, targetWindowId) => {
      const target = BrowserWindow.fromId(targetWindowId);
      if (!target) {
        return { exists: false, windowId: targetWindowId };
      }
      const observerChannel = `browser-gate-window-close-${targetWindowId}-${Date.now()}`;
      const observer = new BrowserWindow({
        show: false,
        webPreferences: {
          contextIsolation: false,
          nodeIntegration: true,
          sandbox: false,
        },
      });
      observer.__browserGateClosedEvent = null;
      ipcMain.once(observerChannel, (_event, payload) => {
        observer.__browserGateClosedEvent = payload;
      });
      const observerSource = [
        "<script>",
        'const { ipcRenderer } = require("electron");',
        `ipcRenderer.on("evt:browser-tab-closed", (_event, payload) => ipcRenderer.send(${JSON.stringify(observerChannel)}, payload));`,
        "</script>",
      ].join("");
      await observer.loadURL(
        `data:text/html;charset=utf-8,${encodeURIComponent(observerSource)}`,
      );
      const rendererWebContentsId = target.webContents.id;
      setImmediate(() => {
        if (!target.isDestroyed()) {
          target.destroy();
        }
      });
      return {
        exists: true,
        observerWindowId: observer.id,
        rendererWebContentsId,
        trigger: "browser-window-destroy",
        windowId: targetWindowId,
      };
    },
    windowId,
  );
}

async function readNativeWindowState(
  app,
  { browserWebContentsId, observerWindowId, windowId },
) {
  return await app.evaluate(
    ({ BrowserWindow, webContents }, ids) => {
      const window = BrowserWindow.fromId(ids.windowId);
      const browserWebContents = webContents.fromId(ids.browserWebContentsId);
      const observer = BrowserWindow.fromId(ids.observerWindowId);
      return {
        browserWebContentsDestroyed:
          !browserWebContents || browserWebContents.isDestroyed(),
        browserWebContentsExists: Boolean(browserWebContents),
        windowDestroyed: !window || window.isDestroyed(),
        windowExists: Boolean(window),
        routeClosedEvent: observer?.__browserGateClosedEvent ?? null,
      };
    },
    { browserWebContentsId, observerWindowId, windowId },
  );
}

async function waitForNativeWindowClosed(app, options, ids) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    last = await readNativeWindowState(app, ids);
    if (
      !last.windowExists &&
      !last.browserWebContentsExists &&
      last.windowDestroyed &&
      last.browserWebContentsDestroyed &&
      last.routeClosedEvent?.event === "browser-tab-closed"
    ) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Browser owner window/native WebContents 未完成销毁: ${JSON.stringify(
      sanitizeJson(last),
    )}`,
  );
}

export function buildWindowCloseAssertions({
  agentControlled,
  closeRequest,
  consoleErrors,
  debuggerBeforeClose,
  guiBeforeTurn,
  identity,
  nativeAfterClose,
  pageClosed,
  pageErrors,
  providerRequestsBeforeClose,
}) {
  return {
    browserSurfaceVisibleBeforeClose:
      guiBeforeTurn?.panelVisible === true &&
      guiBeforeTurn?.activeSurface === "browser",
    canonicalIdentityPresent:
      identity?.threadId === agentControlled?.threadId &&
      Boolean(identity?.sessionId),
    agentControlledBeforeClose:
      agentControlled?.controlOwner === "agent" &&
      Boolean(agentControlled?.activeTurnId),
    debuggerAttachedBeforeClose:
      debuggerBeforeClose?.exists === true &&
      debuggerBeforeClose?.attached === true &&
      debuggerBeforeClose?.webContentsId === agentControlled?.webContentsId,
    windowCloseTriggeredInElectron:
      closeRequest?.exists === true &&
      closeRequest?.trigger === "browser-window-destroy" &&
      closeRequest?.windowId === agentControlled?.windowId,
    rendererPageClosed: pageClosed === true,
    ownerWindowDestroyed:
      nativeAfterClose?.windowExists === false &&
      nativeAfterClose?.windowDestroyed === true,
    browserWebContentsDestroyed:
      nativeAfterClose?.browserWebContentsExists === false &&
      nativeAfterClose?.browserWebContentsDestroyed === true,
    browserRouteClosedByWindow:
      nativeAfterClose?.routeClosedEvent?.event === "browser-tab-closed" &&
      nativeAfterClose?.routeClosedEvent?.payload?.reason === "window-closed" &&
      nativeAfterClose?.routeClosedEvent?.payload?.tabId ===
        agentControlled?.tabId &&
      nativeAfterClose?.routeClosedEvent?.payload?.threadId ===
        agentControlled?.threadId &&
      nativeAfterClose?.routeClosedEvent?.payload?.viewId ===
        agentControlled?.viewId,
    nativeIdentityBoundBeforeClose:
      Number.isInteger(agentControlled?.windowId) &&
      agentControlled.windowId > 0 &&
      Number.isInteger(agentControlled?.webContentsId) &&
      agentControlled.webContentsId > 0 &&
      Number.isInteger(agentControlled?.ownerWebContentsId) &&
      agentControlled.ownerWebContentsId > 0,
    noProviderContinuationAfterWindowClose:
      providerRequestsBeforeClose?.after ===
      providerRequestsBeforeClose?.before,
    consoleAndPageErrorsZero:
      consoleErrors?.length === 0 && pageErrors?.length === 0,
  };
}

export async function runBrowserWindowCloseScenario({
  agentControlled,
  app,
  consoleErrors,
  guiBeforeTurn,
  identity,
  initial,
  options,
  page,
  pageErrors,
  providerFixture,
  readBrowserDebuggerState,
}) {
  const controlledIdentity = {
    ...agentControlled,
    ownerWebContentsId: initial.state.ownerWebContentsId,
    windowId: initial.state.windowId,
  };
  const debuggerBeforeClose = await readBrowserDebuggerState(
    app,
    controlledIdentity.webContentsId,
  );
  const providerRequestsBeforeClose = {
    before: providerFixture.requests.length,
    after: null,
  };
  const pageClosedPromise = new Promise((resolve) => {
    page.once("close", () => resolve(true));
  });

  console.log(
    "[smoke:browser-runtime-electron-gate-b] stage=close-browser-window",
  );
  const closeRequest = await destroyBrowserOwnerWindow(
    app,
    controlledIdentity.windowId,
  );
  const pageClosed = await Promise.race([
    pageClosedPromise,
    sleep(options.timeoutMs).then(() => false),
  ]);
  const nativeAfterClose = await waitForNativeWindowClosed(app, options, {
    browserWebContentsId: controlledIdentity.webContentsId,
    observerWindowId: closeRequest.observerWindowId,
    windowId: controlledIdentity.windowId,
  });
  providerRequestsBeforeClose.after = providerFixture.requests.length;
  const assertions = buildWindowCloseAssertions({
    agentControlled: controlledIdentity,
    closeRequest,
    consoleErrors,
    debuggerBeforeClose,
    guiBeforeTurn,
    identity,
    nativeAfterClose,
    pageClosed,
    pageErrors,
    providerRequestsBeforeClose,
  });
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);

  return sanitizeJson({
    schemaVersion: "lime.browser_runtime_electron_gate_b.window_close.v1",
    scenario: "window-close",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    claimBoundary:
      "真实 Electron BrowserWindow closed 事件、BrowserTabHost/EmbeddedBrowserHost native WebContents 清理与 renderer page close 的本地闭环；不代表 sidecar disconnect、权限/下载、live provider 或跨平台打包验证。",
    identity,
    gui: {
      beforeTurn: guiBeforeTurn,
      agentControlled: controlledIdentity,
    },
    close: {
      request: closeRequest,
      pageClosed,
      nativeAfterClose,
      debuggerBeforeClose,
    },
    provider: providerRequestsBeforeClose,
    assertions,
    failedAssertions,
    diagnostics: { consoleErrors, pageErrors },
  });
}
