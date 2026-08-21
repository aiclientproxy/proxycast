import { runBrowserCancelScenario } from "./browser-runtime-electron-gate-b-cancel.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

const DOWNLOAD_EVENT = "browser-tab-download";
const DOWNLOAD_FILENAME = "browser-gate-b.txt";

export function buildDownloadAssertions({
  agentControlled,
  cancelEvidence,
  events,
  gui,
  initial,
  native,
  trigger,
}) {
  const started = events.find((event) => event?.payload?.state === "started");
  const cancelled = events.find(
    (event) => event?.payload?.state === "cancelled",
  );
  const identity = cancelled?.payload || started?.payload || {};
  const serializedEvents = JSON.stringify(events);
  return {
    pageDownloadTriggered:
      trigger?.triggered === true && trigger?.filename === DOWNLOAD_FILENAME,
    realPreloadDownloadLifecycle:
      started?.event === DOWNLOAD_EVENT &&
      cancelled?.event === DOWNLOAD_EVENT &&
      started?.payload?.downloadId === cancelled?.payload?.downloadId &&
      cancelled?.payload?.filename === DOWNLOAD_FILENAME,
    canonicalDownloadIdentity:
      identity.browserSessionId === initial?.state?.browserSessionId &&
      identity.browserSessionId === agentControlled?.sessionId &&
      identity.threadId === initial?.state?.threadId &&
      identity.tabId === agentControlled?.tabId &&
      identity.viewId === agentControlled?.viewId &&
      identity.webContentsId === agentControlled?.webContentsId &&
      identity.ownerWebContentsId === initial?.state?.ownerWebContentsId &&
      identity.windowId === initial?.state?.windowId,
    downloadBannerVisible:
      gui?.banner?.visible === true &&
      gui?.banner?.width > 0 &&
      gui?.banner?.height > 0 &&
      String(gui?.banner?.text || "").includes(DOWNLOAD_FILENAME),
    downloadBannerBoundToSelectedTab:
      gui?.workspace?.sessionId === identity.browserSessionId &&
      gui?.workspace?.threadId === identity.threadId &&
      gui?.workspace?.tabId === identity.tabId &&
      gui?.workspace?.viewId === identity.viewId &&
      gui?.workspace?.webContentsId === identity.webContentsId,
    downloadBannerOutsideNativeView:
      Number(gui?.banner?.bottom) <= Number(native?.bounds?.y) + 1,
    nativeViewTracksRendererViewport:
      Math.abs(Number(native?.bounds?.x) - Number(gui?.viewport?.x)) <= 1 &&
      Math.abs(Number(native?.bounds?.y) - Number(gui?.viewport?.y)) <= 1 &&
      Math.abs(Number(native?.bounds?.width) - Number(gui?.viewport?.width)) <=
        1 &&
      Math.abs(
        Number(native?.bounds?.height) - Number(gui?.viewport?.height),
      ) <= 1,
    noLocalSavePathExposed:
      !serializedEvents.includes("savePath") &&
      !serializedEvents.includes("/Users/") &&
      !serializedEvents.includes("\\Users\\"),
    downloadTurnReachedExplicitTerminal:
      cancelEvidence?.status === "pass" &&
      cancelEvidence?.failedAssertions?.length === 0,
    noProductionMockFallback:
      cancelEvidence?.invoke?.mockFallbackHitCount === 0,
    noConsoleOrPageErrors:
      cancelEvidence?.diagnostics?.consoleErrors?.length === 0 &&
      cancelEvidence?.diagnostics?.pageErrors?.length === 0,
  };
}

async function installDownloadEventCapture(page) {
  await page.evaluate((eventName) => {
    const state = { events: [], stop: null };
    state.stop = window.electronAPI.listen(eventName, (event) => {
      state.events.push(event);
    });
    window.__browserDownloadGateB = state;
  }, DOWNLOAD_EVENT);
}

async function stopDownloadEventCapture(page) {
  await page
    .evaluate(() => {
      window.__browserDownloadGateB?.stop?.();
      if (window.__browserDownloadGateB) {
        window.__browserDownloadGateB.stop = null;
      }
    })
    .catch(() => undefined);
}

async function triggerCancelledDownload(app, webContentsId) {
  return await app.evaluate(
    async ({ webContents }, { filename, targetId }) => {
      const target = webContents.fromId(targetId);
      if (!target || target.isDestroyed()) {
        throw new Error(`Browser WebContents 不可用: ${targetId}`);
      }
      target.session.once("will-download", (_event, item, source) => {
        if (source?.id === targetId) {
          item.cancel();
        }
      });
      await target.executeJavaScript(`
        (() => {
          const anchor = document.createElement("a");
          anchor.href = "data:text/plain;charset=utf-8,browser-gate-b";
          anchor.download = ${JSON.stringify(filename)};
          document.body.appendChild(anchor);
          anchor.click();
          anchor.remove();
          return true;
        })()
      `);
      return {
        filename,
        triggered: true,
        url: target.getURL(),
        webContentsId: target.id,
      };
    },
    { filename: DOWNLOAD_FILENAME, targetId: webContentsId },
  );
}

async function readDownloadProjection(page) {
  return await page.evaluate(() => {
    const workspace = document.querySelector(
      '[data-testid="browser-workspace"]',
    );
    const banner = document.querySelector(
      '[data-testid="browser-workspace-download"]',
    );
    const viewport = document.querySelector(
      '[data-testid="browser-workspace-viewport"]',
    );
    const bannerRect = banner?.getBoundingClientRect();
    const viewportRect = viewport?.getBoundingClientRect();
    return {
      events: window.__browserDownloadGateB?.events || [],
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

async function waitForDownloadProjection({
  app,
  options,
  page,
  webContentsId,
}) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const gui = await readDownloadProjection(page);
    const native = await readNativeViewBounds(app, webContentsId);
    last = { gui, native };
    if (
      gui.events.some((event) => event?.payload?.state === "started") &&
      gui.events.some((event) => event?.payload?.state === "cancelled") &&
      gui.banner?.visible === true &&
      Number(gui.banner.bottom) <= Number(native.bounds?.y) + 1 &&
      Math.abs(Number(native.bounds?.y) - Number(gui.viewport?.y)) <= 1
    ) {
      return last;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Browser download GUI/native 投影未对齐: ${JSON.stringify(sanitizeJson(last))}`,
  );
}

export async function runBrowserDownloadScenario({
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
  let trigger;
  let projection;
  await installDownloadEventCapture(page);
  try {
    logStage("trigger-and-cancel-native-download");
    trigger = await triggerCancelledDownload(
      app,
      agentControlled.webContentsId,
    );
    projection = await waitForDownloadProjection({
      app,
      options,
      page,
      webContentsId: agentControlled.webContentsId,
    });
  } finally {
    await stopDownloadEventCapture(page);
  }

  logStage("terminate-download-scenario-turn");
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
  const assertions = buildDownloadAssertions({
    agentControlled,
    cancelEvidence,
    events: projection.gui.events,
    gui: projection.gui,
    initial,
    native: projection.native,
    trigger,
  });
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  return sanitizeJson({
    schemaVersion: "lime.browser_runtime_electron_gate_b.download.v1",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    claimBoundary:
      "真实 Electron WebContentsView 页面下载触发、will-download started/cancelled、preload IPC identity、GUI 可见状态带和 turn terminal；测试主动取消下载，不代表文件落盘、artifact ref 或跨平台打包验证。",
    identity,
    browser: {
      agentControlled,
      initial: initial.state,
      native: projection.native,
    },
    download: {
      events: projection.gui.events,
      gui: projection.gui,
      trigger,
    },
    terminal: cancelEvidence.terminal,
    invoke: cancelEvidence.invoke,
    assertions,
    failedAssertions,
    diagnostics: cancelEvidence.diagnostics,
  });
}
