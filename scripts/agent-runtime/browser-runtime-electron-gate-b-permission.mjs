import { runBrowserCancelScenario } from "./browser-runtime-electron-gate-b-cancel.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

const PERMISSION_EVENT = "browser-tab-permission-request";

function rectanglesDoNotOverlapVertically(upper, lower) {
  return (
    upper &&
    lower &&
    Number.isFinite(upper.bottom) &&
    Number.isFinite(lower.y) &&
    upper.bottom <= lower.y + 1
  );
}

export function buildPermissionAssertions({
  agentControlled,
  cancelEvidence,
  event,
  gui,
  initial,
  native,
  permissionResult,
}) {
  const eventIdentity = event?.payload || {};
  const nativeBounds = native?.bounds || {};
  const viewport = gui?.viewport || {};
  return {
    pagePermissionRequestBlocked:
      permissionResult?.outcome === "blocked" &&
      permissionResult?.code === 1 &&
      /denied|blocked/i.test(String(permissionResult?.message || "")),
    realPreloadPermissionEvent:
      event?.event === PERMISSION_EVENT &&
      eventIdentity.permission === "geolocation" &&
      eventIdentity.decision === "blocked",
    canonicalPermissionIdentity:
      eventIdentity.browserSessionId === initial?.state?.browserSessionId &&
      eventIdentity.browserSessionId === agentControlled?.sessionId &&
      eventIdentity.threadId === initial?.state?.threadId &&
      eventIdentity.tabId === agentControlled?.tabId &&
      eventIdentity.viewId === agentControlled?.viewId &&
      eventIdentity.webContentsId === agentControlled?.webContentsId &&
      eventIdentity.ownerWebContentsId === initial?.state?.ownerWebContentsId &&
      eventIdentity.windowId === initial?.state?.windowId,
    permissionSourceMatchesNativePage:
      eventIdentity.url === permissionResult?.url &&
      String(eventIdentity.requestingUrl || "").startsWith(
        permissionResult?.origin || "<missing-origin>",
      ),
    permissionBannerVisible:
      gui?.banner?.visible === true &&
      gui?.banner?.width > 0 &&
      gui?.banner?.height > 0 &&
      String(gui?.banner?.text || "").includes("geolocation"),
    permissionBannerBoundToSelectedTab:
      gui?.workspace?.sessionId === eventIdentity.browserSessionId &&
      gui?.workspace?.threadId === eventIdentity.threadId &&
      gui?.workspace?.tabId === eventIdentity.tabId &&
      gui?.workspace?.viewId === eventIdentity.viewId &&
      gui?.workspace?.webContentsId === eventIdentity.webContentsId,
    permissionBannerOutsideNativeView: rectanglesDoNotOverlapVertically(
      gui?.banner,
      nativeBounds,
    ),
    nativeViewTracksRendererViewport:
      Math.abs(Number(nativeBounds.x) - Number(viewport.x)) <= 1 &&
      Math.abs(Number(nativeBounds.y) - Number(viewport.y)) <= 1 &&
      Math.abs(Number(nativeBounds.width) - Number(viewport.width)) <= 1 &&
      Math.abs(Number(nativeBounds.height) - Number(viewport.height)) <= 1,
    permissionTurnReachedExplicitTerminal:
      cancelEvidence?.status === "pass" &&
      cancelEvidence?.failedAssertions?.length === 0,
    noProductionMockFallback:
      cancelEvidence?.invoke?.mockFallbackHitCount === 0,
    noConsoleOrPageErrors:
      cancelEvidence?.diagnostics?.consoleErrors?.length === 0 &&
      cancelEvidence?.diagnostics?.pageErrors?.length === 0,
  };
}

async function installPermissionEventCapture(page) {
  await page.evaluate((eventName) => {
    const state = {
      events: [],
      stop: null,
    };
    state.stop = window.electronAPI.listen(eventName, (event) => {
      state.events.push(event);
    });
    window.__browserPermissionGateB = state;
  }, PERMISSION_EVENT);
}

async function stopPermissionEventCapture(page) {
  await page
    .evaluate(() => {
      window.__browserPermissionGateB?.stop?.();
      if (window.__browserPermissionGateB) {
        window.__browserPermissionGateB.stop = null;
      }
    })
    .catch(() => undefined);
}

async function requestGeolocationPermission(app, webContentsId) {
  return await app.evaluate(async ({ webContents }, targetId) => {
    const target = webContents.fromId(targetId);
    if (!target || target.isDestroyed()) {
      throw new Error(`Browser WebContents 不可用: ${targetId}`);
    }
    return await target.executeJavaScript(`
      (async () => {
        const readPermission = async () => {
          try {
            return (await navigator.permissions.query({ name: "geolocation" })).state;
          } catch {
            return "unsupported";
          }
        };
        const permissionBefore = await readPermission();
        const result = await new Promise((resolve) => {
          navigator.geolocation.getCurrentPosition(
            () => resolve({ outcome: "granted", code: 0 }),
            (error) => resolve({
              outcome: "blocked",
              code: error.code,
              message: String(error.message || ""),
            }),
            { enableHighAccuracy: false, maximumAge: 0, timeout: 10000 },
          );
        });
        return {
          ...result,
          origin: location.origin,
          permissionAfter: await readPermission(),
          permissionBefore,
          secureContext: window.isSecureContext,
          url: location.href,
        };
      })()
    `);
  }, webContentsId);
}

async function readPermissionProjection(page) {
  return await page.evaluate(() => {
    const workspace = document.querySelector(
      '[data-testid="browser-workspace"]',
    );
    const banner = document.querySelector(
      '[data-testid="browser-workspace-permission"]',
    );
    const viewport = document.querySelector(
      '[data-testid="browser-workspace-viewport"]',
    );
    const bannerRect = banner?.getBoundingClientRect();
    const viewportRect = viewport?.getBoundingClientRect();
    return {
      event: window.__browserPermissionGateB?.events?.at(-1) || null,
      banner: bannerRect
        ? {
            bottom: bannerRect.bottom,
            height: bannerRect.height,
            text: banner?.textContent || "",
            visible:
              bannerRect.width > 0 &&
              bannerRect.height > 0 &&
              getComputedStyle(banner).visibility !== "hidden",
            width: bannerRect.width,
            x: bannerRect.x,
            y: bannerRect.y,
          }
        : null,
      viewport: viewportRect
        ? {
            height: viewportRect.height,
            width: viewportRect.width,
            x: viewportRect.x,
            y: viewportRect.y,
          }
        : null,
      workspace: {
        sessionId: workspace?.getAttribute("data-browser-session-id") || null,
        tabId: workspace?.getAttribute("data-browser-tab-id") || null,
        threadId: workspace?.getAttribute("data-browser-thread-id") || null,
        viewId: workspace?.getAttribute("data-browser-view-id") || null,
        webContentsId:
          Number(workspace?.getAttribute("data-browser-web-contents-id")) ||
          null,
      },
    };
  });
}

async function readNativeViewBounds(app, webContentsId) {
  return await app.evaluate(({ BrowserWindow, webContents }, targetId) => {
    const target = webContents.fromId(targetId);
    const owner = BrowserWindow.getAllWindows().find((window) =>
      window.contentView.children.some(
        (child) => child.webContents?.id === targetId,
      ),
    );
    const view = owner?.contentView.children.find(
      (child) => child.webContents?.id === targetId,
    );
    return {
      bounds: view?.getBounds?.() || null,
      ownerWebContentsId: owner?.webContents.id || null,
      webContentsId: target?.id || null,
      windowId: owner?.id || null,
    };
  }, webContentsId);
}

async function waitForPermissionProjection({
  app,
  options,
  page,
  webContentsId,
}) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const gui = await readPermissionProjection(page);
    const native = await readNativeViewBounds(app, webContentsId);
    last = { gui, native };
    if (
      gui.event?.payload?.decision === "blocked" &&
      gui.banner?.visible === true &&
      rectanglesDoNotOverlapVertically(gui.banner, native.bounds) &&
      Math.abs(Number(native.bounds?.y) - Number(gui.viewport?.y)) <= 1
    ) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Browser permission GUI/native 投影未对齐: ${JSON.stringify(sanitizeJson(last))}`,
  );
}

export async function runBrowserPermissionScenario({
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
  let permissionResult;
  let projection;
  await installPermissionEventCapture(page);
  try {
    logStage("request-native-geolocation-permission");
    permissionResult = await requestGeolocationPermission(
      app,
      agentControlled.webContentsId,
    );
    projection = await waitForPermissionProjection({
      app,
      options,
      page,
      webContentsId: agentControlled.webContentsId,
    });
  } finally {
    await stopPermissionEventCapture(page);
  }

  logStage("terminate-permission-scenario-turn");
  const cancelEvidence = await runBrowserCancelScenario({
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
  });
  const assertions = buildPermissionAssertions({
    agentControlled,
    cancelEvidence,
    event: projection.gui.event,
    gui: projection.gui,
    initial,
    native: projection.native,
    permissionResult,
  });
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  return sanitizeJson({
    schemaVersion: "lime.browser_runtime_electron_gate_b.permission.v1",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    claimBoundary:
      "真实 Electron WebContentsView 页面权限请求、默认拒绝、preload IPC identity、GUI 可见状态带和 turn terminal；不代表权限授予、live provider 或跨平台打包验证。",
    identity,
    browser: {
      agentControlled,
      initial: initial.state,
      native: projection.native,
    },
    permission: {
      event: projection.gui.event,
      gui: projection.gui,
      result: permissionResult,
    },
    terminal: cancelEvidence.terminal,
    invokeTrace: cancelEvidence.invokeTrace,
    assertions,
    failedAssertions,
    diagnostics: cancelEvidence.diagnostics,
  });
}
