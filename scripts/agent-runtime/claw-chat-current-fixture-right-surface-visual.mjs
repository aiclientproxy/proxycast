import path from "node:path";
import {
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
  EXPERT_SKILLS_RUNTIME_SESSION_ID,
} from "./claw-chat-current-fixture-constants.mjs";
import { invokeAppServerFromPage } from "./claw-chat-current-fixture-rpc.mjs";
import {
  assert,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

const RIGHT_SURFACE_ROOTS = {
  appSurface: "workspace-plugin-surface",
  browser: "right-surface-browser-panel",
  expertInfo: "expert-info-panel",
  files: "workspace-files-surface",
  objectCanvas: "workspace-object-canvas-surface",
  articleWorkspace: "workspace-article-editor-surface",
  activity: "thread-activity-panel",
};

export async function runRightSurfaceVisualMatrix({
  page,
  options,
  workspace,
  appServerRequests,
  sessionId = EXPERT_SKILLS_RUNTIME_SESSION_ID,
}) {
  assert(
    workspace?.workspaceId,
    "Right Surface visual matrix 缺少 workspaceId",
  );
  assert(
    workspace?.rootPath,
    "Right Surface visual matrix 缺少 workspace rootPath",
  );

  const requests = {
    files: await requestRightSurfacePending(page, appServerRequests, {
      workspace,
      sessionId,
      surfaceKind: "files",
      candidateId: "internal/roadmap/rightsurface/README.md",
      metadata: {
        relativePath: "internal/roadmap/rightsurface/README.md",
        title: "Right Surface roadmap",
      },
    }),
    objectCanvas: await requestRightSurfacePending(page, appServerRequests, {
      workspace,
      sessionId,
      surfaceKind: "objectCanvas",
      candidateId: "right-surface-visual-browser-session",
      metadata: {
        candidateId: "right-surface-visual-browser-session",
        title: "Right Surface visual browser session",
        url: "https://example.com/right-surface-object-canvas",
        sessionId: "fixture-browser-session",
        profileKey: "fixture-profile",
        targetId: "fixture-target",
        lifecycleState: "ready",
        controlMode: "inspect",
        transportKind: "cdp",
      },
    }),
    browser: await requestRightSurfacePending(page, appServerRequests, {
      workspace,
      sessionId,
      surfaceKind: "browser",
      candidateId: "right-surface-visual-browser",
      metadata: {
        browserSessionId: "fixture-browser-session",
        profileKey: "fixture-profile",
        targetId: "fixture-target",
        launchUrl: "https://example.com/right-surface-browser",
        title: "Right Surface visual browser",
        adapterKind: "cdp",
        lifecycleState: "ready",
        controlMode: "inspect",
      },
    }),
    appSurfaceContentFactory: await requestRightSurfacePending(
      page,
      appServerRequests,
      {
        workspace,
        sessionId,
        surfaceKind: "appSurface",
        candidateId: "plugin-shell-content-factory-app-main",
        metadata: {
          appId: "content-factory-app",
          title: "内容工厂",
          surface: {
            entryUrl:
              "https://example.com/lime-plugins/content-factory/right-surface",
            containerId: "plugin-shell-content-factory-app-main",
            activeStrategy: "webContentsView",
            supportedStrategies: ["controlledBrowserWindow", "webContentsView"],
            embedding: {
              rightSurfaceDock: true,
              iframe: false,
              browserView: false,
            },
          },
        },
      },
    ),
    appSurfacePromptLab: await requestRightSurfacePending(
      page,
      appServerRequests,
      {
        workspace,
        sessionId,
        surfaceKind: "appSurface",
        candidateId: "plugin-shell-prompt-lab-app",
        metadata: {
          appId: "prompt-lab-app",
          title: "Prompt Lab",
          surface: {
            entryUrl:
              "https://example.com/lime-plugins/prompt-lab/right-surface",
            containerId: "plugin-shell-prompt-lab-app",
            activeStrategy: "webContentsView",
            supportedStrategies: ["controlledBrowserWindow", "webContentsView"],
            embedding: {
              rightSurfaceDock: true,
              iframe: false,
              browserView: false,
            },
          },
        },
      },
    ),
  };

  const pendingBeforeClicks = await listRightSurfacePending(
    page,
    appServerRequests,
    {
      workspace,
      sessionId,
    },
  );
  const toolbarBeforeClicks = await captureRightSurfaceToolbarDiagnostics(page);

  let captures;
  try {
    captures = {
      files: await clickAndAssertRightSurface(page, options, {
        surfaceKind: "files",
        toggleTestId: "task-center-files-toggle",
        rootTestId: RIGHT_SURFACE_ROOTS.files,
      }),
      objectCanvas: await clickAndAssertRightSurface(page, options, {
        surfaceKind: "objectCanvas",
        toggleTestId: "task-center-object-canvas-toggle",
        rootTestId: RIGHT_SURFACE_ROOTS.objectCanvas,
      }),
      expertInfo: await clickAndAssertRightSurface(page, options, {
        surfaceKind: "expertInfo",
        toggleTestId: "task-center-expert-info-toggle",
        rootTestId: RIGHT_SURFACE_ROOTS.expertInfo,
      }),
      browser: await clickAndAssertRightSurface(page, options, {
        surfaceKind: "browser",
        toggleTestId: "task-center-browser-toggle",
        rootTestId: RIGHT_SURFACE_ROOTS.browser,
        screenshotName: "right-surface-browser",
      }),
      appSurface: await clickAndAssertRightSurface(page, options, {
        surfaceKind: "appSurface",
        toggleTestId: "workspace-right-surface-tab-appSurface",
        rootTestId: RIGHT_SURFACE_ROOTS.appSurface,
      }),
      activity: await clickAndAssertRightSurface(page, options, {
        surfaceKind: "activity",
        toggleTestId: "task-center-activity-toggle",
        rootTestId: RIGHT_SURFACE_ROOTS.activity,
        requireCanvasPanelFill: false,
      }),
    };
  } catch (error) {
    const toolbarAfterFailure = await captureRightSurfaceToolbarDiagnostics(
      page,
    ).catch((diagnosticError) => ({
      error:
        diagnosticError instanceof Error
          ? diagnosticError.message
          : String(diagnosticError),
    }));
    const diagnostic = sanitizeJson({
      requests: summarizeRightSurfaceRequests(requests),
      pendingBeforeClicks: summarizePendingList(pendingBeforeClicks.result),
      toolbarBeforeClicks,
      toolbarAfterFailure,
    });
    const message = error instanceof Error ? error.message : String(error);
    return sanitizeJson({
      sessionId,
      error: message,
      ...diagnostic,
      captures: {},
    });
  }

  const pendingAfterClicks = await listRightSurfacePending(
    page,
    appServerRequests,
    {
      workspace,
      sessionId,
    },
  );

  return sanitizeJson({
    sessionId,
    requests: summarizeRightSurfaceRequests(requests),
    pendingBeforeClicks: summarizePendingList(pendingBeforeClicks.result),
    toolbarBeforeClicks,
    captures,
    pendingAfterClicks: summarizePendingList(pendingAfterClicks.result),
  });
}

export async function runThreadActivityPanelSmoke({
  page,
  options,
  sessionId,
}) {
  const toggle = await waitForRightSurfaceToggle(
    page,
    options,
    "task-center-activity-toggle",
  );
  await clickRightSurfaceToggle(page, "task-center-activity-toggle");
  const opened = await waitForRightSurfaceSnapshot(page, options, {
    activeSurfaceKind: "activity",
    requireCanvasPanelFill: false,
    surfaceKind: "activity",
    rootTestId: RIGHT_SURFACE_ROOTS.activity,
  });
  await sleep(1_000);
  const stable = await captureRightSurfaceSnapshot(page, {
    activeSurfaceKind: "activity",
    surfaceKind: "activity",
    rootTestId: RIGHT_SURFACE_ROOTS.activity,
  });
  assertRightSurfaceSnapshot(stable, "activity", "activity", false);
  return sanitizeJson({
    activityOnly: true,
    sessionId,
    captures: { activity: { toggle, opened, stable } },
  });
}

async function requestRightSurfacePending(
  page,
  appServerRequests,
  { workspace, sessionId, surfaceKind, candidateId, metadata },
) {
  return await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
    {
      workspaceId: workspace.workspaceId,
      workspaceRoot: workspace.rootPath,
      sessionId,
      surfaceKind,
      origin: "runtime",
      priority: "normal",
      candidateId,
      ttlMs: 120_000,
      metadata,
    },
    appServerRequests,
  );
}

async function listRightSurfacePending(
  page,
  appServerRequests,
  { workspace, sessionId },
) {
  return await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST,
    {
      workspaceId: workspace.workspaceId,
      workspaceRoot: workspace.rootPath,
      sessionId,
      limit: 20,
    },
    appServerRequests,
  );
}

async function captureRightSurfaceToolbarDiagnostics(page) {
  return await page.evaluate(() => {
    const toggleByKind = {
      appSurface: "workspace-right-surface-tab-appSurface",
      browser: "task-center-browser-toggle",
      expertInfo: "task-center-expert-info-toggle",
      files: "task-center-files-toggle",
      objectCanvas: "task-center-object-canvas-toggle",
      shell: "task-center-shell-toggle",
      trace: "task-center-trace-toggle",
      workbench: "task-center-workbench-toggle",
    };
    const toolbar = document.querySelector(
      '[data-testid="task-center-utility-toolbar"]',
    );
    const panelGroup = document.querySelector(
      '[data-testid="task-center-tool-group-panels"]',
    );
    const host = document.querySelector(
      '[data-testid="workspace-right-surface-host"]',
    );

    return {
      electron: window.__LIME_ELECTRON__ === true,
      hasElectronInvoke: Boolean(window.electronAPI?.invoke),
      hasElectronSupportsCommand:
        typeof window.electronAPI?.supportsCommand === "function",
      appServerCommandAvailable:
        typeof window.electronAPI?.supportsCommand === "function"
          ? window.electronAPI.supportsCommand("app_server_handle_json_lines")
          : null,
      toolbar: visibleInfo(toolbar),
      panelGroup: visibleInfo(panelGroup),
      activeSurface: host?.getAttribute("data-surface") ?? null,
      toggles: Object.fromEntries(
        Object.entries(toggleByKind).map(([kind, testId]) => {
          const element = document.querySelector(`[data-testid="${testId}"]`);
          const button =
            element instanceof HTMLButtonElement
              ? element
              : element?.closest("button");
          return [
            kind,
            {
              ...visibleInfo(element),
              disabled:
                button instanceof HTMLButtonElement
                  ? button.disabled ||
                    button.getAttribute("aria-disabled") === "true"
                  : null,
              ariaExpanded: button?.getAttribute("aria-expanded") ?? null,
              title: button?.getAttribute("title") ?? "",
              text: (button?.textContent ?? "").trim().slice(0, 80),
            },
          ];
        }),
      ),
      panelButtonKinds: Array.from(
        document.querySelectorAll(
          '[data-testid="task-center-tool-group-panels"] [data-testid]',
        ),
      ).map((element) => {
        const testId = element.getAttribute("data-testid") ?? "";
        return testId
          .replace(/^task-center-/, "")
          .replace(/^workspace-right-surface-tab-/, "surface-tab-")
          .replace(/-toggle$/, "");
      }),
      bodyTextSample: (document.body?.innerText ?? "").slice(0, 1000),
    };

    function visibleInfo(node) {
      const rect = node?.getBoundingClientRect();
      const style = node ? window.getComputedStyle(node) : null;
      return {
        exists: Boolean(node),
        visible: Boolean(
          node &&
          rect &&
          rect.width > 8 &&
          rect.height > 8 &&
          style?.display !== "none" &&
          style?.visibility !== "hidden" &&
          Number(style?.opacity ?? "1") > 0,
        ),
        rect: rect ? rectToJson(rect) : null,
      };
    }

    function rectToJson(rect) {
      return {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        top: Math.round(rect.top),
        left: Math.round(rect.left),
        right: Math.round(rect.right),
        bottom: Math.round(rect.bottom),
      };
    }
  });
}

export async function clickAndAssertRightSurface(
  page,
  options,
  {
    surfaceKind,
    activeSurfaceKind = surfaceKind,
    toggleTestId,
    rootTestId,
    screenshotName = null,
    requireCanvasPanelFill = true,
  },
) {
  const toggle = await waitForRightSurfaceToggle(page, options, toggleTestId);
  await clickRightSurfaceToggle(page, toggleTestId);
  const opened = await waitForRightSurfaceSnapshot(page, options, {
    activeSurfaceKind,
    requireCanvasPanelFill,
    surfaceKind,
    rootTestId,
  });

  await sleep(1_000);
  const stable = await captureRightSurfaceSnapshot(page, {
    activeSurfaceKind,
    surfaceKind,
    rootTestId,
  });
  assertRightSurfaceSnapshot(
    stable,
    surfaceKind,
    activeSurfaceKind,
    requireCanvasPanelFill,
  );
  const screenshot = screenshotName
    ? await captureRightSurfaceScreenshot(page, options, screenshotName)
    : null;

  return {
    toggle,
    opened,
    stable,
    screenshot,
  };
}

async function captureRightSurfaceScreenshot(page, options, screenshotName) {
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-${screenshotName}.png`,
  );
  await page.screenshot({
    path: screenshotPath,
    fullPage: false,
    timeout: 15_000,
  });
  return screenshotPath;
}

async function waitForRightSurfaceToggle(page, options, testId) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await page.evaluate(
      ({ testId }) => {
        const element = document.querySelector(`[data-testid="${testId}"]`);
        const rect = element?.getBoundingClientRect();
        const style = element ? window.getComputedStyle(element) : null;
        const button =
          element instanceof HTMLButtonElement
            ? element
            : element?.closest("button");
        const visible = Boolean(
          element &&
          rect &&
          rect.width > 8 &&
          rect.height > 8 &&
          style?.display !== "none" &&
          style?.visibility !== "hidden",
        );
        return {
          testId,
          visible,
          disabled:
            button instanceof HTMLButtonElement
              ? button.disabled ||
                button.getAttribute("aria-disabled") === "true"
              : null,
          ariaExpanded: button?.getAttribute("aria-expanded") ?? null,
          title: button?.getAttribute("title") ?? "",
          text: button?.textContent ?? "",
          rect: rect ? rectToJson(rect) : null,
        };

        function rectToJson(rect) {
          return {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            top: Math.round(rect.top),
            left: Math.round(rect.left),
            right: Math.round(rect.right),
            bottom: Math.round(rect.bottom),
          };
        }
      },
      { testId },
    );
    lastSnapshot = snapshot;
    if (snapshot.visible && snapshot.disabled === false) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `Right Surface 顶部按钮不可用: ${testId}; snapshot=${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function clickRightSurfaceToggle(page, testId) {
  const clicked = await page.evaluate(
    ({ testId }) => {
      const element = document.querySelector(`[data-testid="${testId}"]`);
      const button =
        element instanceof HTMLButtonElement
          ? element
          : element?.closest("button");
      if (!button) {
        return { clicked: false, reason: "missing-button" };
      }
      button.click();
      return {
        clicked: true,
        testId,
        ariaExpanded: button.getAttribute("aria-expanded"),
        title: button.getAttribute("title") || "",
      };
    },
    { testId },
  );
  assert(clicked.clicked, `Right Surface 顶部按钮点击失败: ${testId}`);
  return clicked;
}

async function waitForRightSurfaceSnapshot(
  page,
  options,
  {
    surfaceKind,
    activeSurfaceKind = surfaceKind,
    requireCanvasPanelFill = true,
    rootTestId,
  },
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await captureRightSurfaceSnapshot(page, {
      surfaceKind,
      rootTestId,
    });
    lastSnapshot = snapshot;
    if (
      isRightSurfaceSnapshotReady(
        snapshot,
        surfaceKind,
        activeSurfaceKind,
        requireCanvasPanelFill,
      )
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `Right Surface 未打开目标 surface: ${surfaceKind}; snapshot=${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function captureRightSurfaceSnapshot(
  page,
  { surfaceKind, activeSurfaceKind = surfaceKind, rootTestId },
) {
  return await page.evaluate(
    ({ surfaceKind, activeSurfaceKind, rootTestId, roots }) => {
      const host = document.querySelector(
        '[data-testid="workspace-right-surface-host"]',
      );
      const canvasPanel = document.querySelector(
        '[data-testid="layout-canvas-panel"]',
      );
      const activePane = document.querySelector(
        '[data-testid="workspace-right-surface-active-pane"]',
      );
      const layoutRoot = document.querySelector(
        '[data-testid="layout-transition-root"]',
      );
      const root = document.querySelector(`[data-testid="${rootTestId}"]`);
      const pluginTabs = document.querySelector(
        '[data-testid="workspace-plugin-surface-tabs"]',
      );
      const pluginTabButtons = Array.from(
        document.querySelectorAll(
          '[data-testid^="workspace-plugin-surface-tab-"]',
        ),
      );
      const pluginFrames = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-plugin-surface-frame"]',
        ),
      );
      const pluginViewports = Array.from(
        document.querySelectorAll(
          '[data-testid="workspace-plugin-surface-viewport"]',
        ),
      );
      const browserPanel = document.querySelector(
        '[data-testid="right-surface-browser-panel"]',
      );
      const hostRect = rectFor(host);
      const canvasPanelRect = rectFor(canvasPanel);
      const activePaneRect = rectFor(activePane);
      const rootRect = rectFor(root);
      const rootVisibility = Object.fromEntries(
        Object.entries(roots).map(([kind, testId]) => {
          const node = document.querySelector(`[data-testid="${testId}"]`);
          return [kind, visibleInfo(node)];
        }),
      );
      const visibleRootKinds = Object.entries(rootVisibility)
        .filter(([, value]) => value.visible)
        .map(([kind]) => kind);
      const hostCanvasWidthDelta =
        hostRect && canvasPanelRect
          ? Math.abs(hostRect.width - canvasPanelRect.width)
          : null;
      const hostCanvasHeightDelta =
        hostRect && canvasPanelRect
          ? Math.abs(hostRect.height - canvasPanelRect.height)
          : null;
      const rootHostWidthDelta =
        hostRect && rootRect ? Math.abs(hostRect.width - rootRect.width) : null;
      const rootHostHeightDelta =
        hostRect && rootRect
          ? Math.abs(hostRect.height - rootRect.height)
          : null;
      const rootActivePaneWidthDelta =
        activePaneRect && rootRect
          ? Math.abs(activePaneRect.width - rootRect.width)
          : null;
      const rootActivePaneHeightDelta =
        activePaneRect && rootRect
          ? Math.abs(activePaneRect.height - rootRect.height)
          : null;
      const rootFillsHost =
        Boolean(hostRect && rootRect) &&
        rootRect.width >= hostRect.width - 8 &&
        rootRect.height >= hostRect.height - 8;
      const rootFillsActivePane =
        Boolean(activePaneRect && rootRect) &&
        rootRect.width >= activePaneRect.width - 8 &&
        rootRect.height >= activePaneRect.height - 8;

      return {
        expectedSurface: surfaceKind,
        expectedActiveSurface: activeSurfaceKind,
        activeSurface: host?.getAttribute("data-surface") ?? null,
        layoutMode: layoutRoot?.getAttribute("data-effective-mode") ?? null,
        hostVisible: visibleInfo(host).visible,
        rootVisible: visibleInfo(root).visible,
        visibleRootKinds,
        rootVisibility,
        rects: {
          host: hostRect,
          canvasPanel: canvasPanelRect,
          activePane: activePaneRect,
          root: rootRect,
        },
        geometry: {
          hostCanvasWidthDelta,
          hostCanvasHeightDelta,
          rootHostWidthDelta,
          rootHostHeightDelta,
          rootActivePaneWidthDelta,
          rootActivePaneHeightDelta,
          hostFillsCanvasPanel:
            Boolean(hostRect && canvasPanelRect) &&
            hostRect.width >= 360 &&
            hostRect.height >= 500 &&
            hostCanvasWidthDelta <= 8 &&
            hostCanvasHeightDelta <= 8,
          rootFillsHost,
          rootFillsActivePane,
          rootFillsSurfaceViewport: rootFillsHost || rootFillsActivePane,
        },
        pluginSurface: {
          tabs: visibleInfo(pluginTabs),
          tabCount: pluginTabButtons.length,
          tabLabels: pluginTabButtons.map((button) =>
            (button.textContent || "").trim(),
          ),
          activeTabLabels: pluginTabButtons
            .filter((button) => button.getAttribute("aria-selected") === "true")
            .map((button) => (button.textContent || "").trim()),
          frameCount: pluginFrames.length,
          visibleFrameCount: pluginFrames.filter(
            (frame) => visibleInfo(frame).visible,
          ).length,
          viewportCount: pluginViewports.length,
          visibleViewportCount: pluginViewports.filter(
            (viewport) => visibleInfo(viewport).visible,
          ).length,
        },
        browserSurface: browserPanel
          ? {
              adapterKind:
                browserPanel.getAttribute("data-browser-adapter-kind") ?? "",
              controlMode:
                browserPanel.getAttribute("data-browser-control-mode") ?? "",
              controlOwner:
                browserPanel.getAttribute("data-browser-control-owner") ?? "",
              humanTakeover:
                browserPanel.getAttribute("data-browser-human-takeover") ?? "",
              lifecycleState:
                browserPanel.getAttribute("data-browser-lifecycle-state") ?? "",
              profileKey:
                browserPanel.getAttribute("data-browser-profile-key") ?? "",
              sessionId:
                browserPanel.getAttribute("data-browser-session-id") ?? "",
              hasControlOverlay: Boolean(
                document.querySelector(
                  '[data-testid="right-surface-browser-control-overlay"]',
                ),
              ),
            }
          : null,
        bodyTextSample: (document.body?.innerText || "").slice(0, 2000),
      };

      function visibleInfo(node) {
        const rect = node?.getBoundingClientRect();
        const style = node ? window.getComputedStyle(node) : null;
        return {
          exists: Boolean(node),
          visible: Boolean(
            node &&
            rect &&
            rect.width > 8 &&
            rect.height > 8 &&
            style?.display !== "none" &&
            style?.visibility !== "hidden" &&
            Number(style?.opacity ?? "1") > 0,
          ),
          rect: rect ? rectToJson(rect) : null,
        };
      }

      function rectFor(node) {
        const rect = node?.getBoundingClientRect();
        return rect ? rectToJson(rect) : null;
      }

      function rectToJson(rect) {
        return {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
          width: Math.round(rect.width),
          height: Math.round(rect.height),
          top: Math.round(rect.top),
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          bottom: Math.round(rect.bottom),
        };
      }
    },
    {
      activeSurfaceKind,
      surfaceKind,
      rootTestId,
      roots: RIGHT_SURFACE_ROOTS,
    },
  );
}

export function isRightSurfaceSnapshotReady(
  snapshot,
  surfaceKind,
  activeSurfaceKind = surfaceKind,
  requireCanvasPanelFill = true,
) {
  const surfaceViewportReady =
    snapshot?.geometry?.rootFillsSurfaceViewport === true &&
    (surfaceKind === "articleWorkspace" ||
      requireCanvasPanelFill === false ||
      snapshot?.geometry?.hostFillsCanvasPanel === true ||
      snapshot?.geometry?.rootFillsActivePane === true);

  return (
    snapshot?.activeSurface === activeSurfaceKind &&
    snapshot?.hostVisible === true &&
    snapshot?.rootVisible === true &&
    Array.isArray(snapshot?.visibleRootKinds) &&
    snapshot.visibleRootKinds.length === 1 &&
    snapshot.visibleRootKinds[0] === surfaceKind &&
    surfaceViewportReady
  );
}

function assertRightSurfaceSnapshot(
  snapshot,
  surfaceKind,
  activeSurfaceKind = surfaceKind,
  requireCanvasPanelFill = true,
) {
  assert(
    isRightSurfaceSnapshotReady(
      snapshot,
      surfaceKind,
      activeSurfaceKind,
      requireCanvasPanelFill,
    ),
    `Right Surface stable snapshot 未保持目标 surface: ${surfaceKind}; snapshot=${JSON.stringify(
      sanitizeJson(snapshot),
    )}`,
  );
}

function summarizeRightSurfaceRequest(invocation) {
  const pending = invocation?.result?.pending ?? {};
  return sanitizeJson({
    requestId: invocation?.result?.requestId ?? pending.requestId ?? null,
    status: invocation?.result?.status ?? pending.status ?? null,
    surfaceKind: pending.surfaceKind ?? null,
    candidateId: pending.candidateId ?? null,
    origin: pending.origin ?? null,
  });
}

function summarizeRightSurfaceRequests(requests) {
  return {
    files: summarizeRightSurfaceRequest(requests.files),
    objectCanvas: summarizeRightSurfaceRequest(requests.objectCanvas),
    browser: summarizeRightSurfaceRequest(requests.browser),
    appSurfaceContentFactory: summarizeRightSurfaceRequest(
      requests.appSurfaceContentFactory,
    ),
    appSurfacePromptLab: summarizeRightSurfaceRequest(
      requests.appSurfacePromptLab,
    ),
  };
}

function summarizePendingList(result) {
  const pending = Array.isArray(result?.pending) ? result.pending : [];
  return sanitizeJson({
    count: pending.length,
    surfaces: pending.map((request) => ({
      requestId: request?.requestId ?? null,
      surfaceKind: request?.surfaceKind ?? null,
      status: request?.status ?? null,
      candidateId: request?.candidateId ?? null,
    })),
  });
}
