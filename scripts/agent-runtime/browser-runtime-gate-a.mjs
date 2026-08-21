import path from "node:path";

import {
  assert,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

const NAVIGATION_URL = "https://example.com/browser-runtime-gate-a-nav";

export async function runBrowserProjectionGateA({
  guiBeforeTurn,
  identity,
  options,
  page,
  readBrowserWorkspaceState,
}) {
  const screenshots = [];
  const stage = (name) => console.log(`[browser-runtime-gate-a] stage=${name}`);
  stage("initial");
  const initial = await readProjection(page);
  assert(initial.panelVisible, "Gate A Browser panel 未显示");
  assert(initial.workspaceVisible, "Gate A Browser workspace 未显示");
  assert(initial.tabs.length === 1, "Gate A 首屏应只有一个选中 tab");
  assert(
    initial.selectedTabId === guiBeforeTurn.tabId,
    "Gate A 首屏 tab identity 漂移",
  );

  stage("navigate");
  const address = page.locator('[data-testid="browser-workspace-address"]');
  await address.fill(NAVIGATION_URL);
  await address.press("Enter");
  const navigated = await waitForProjection(
    page,
    (state) => state.address === NAVIGATION_URL && state.loading === false,
  );
  assert(
    navigated.selectedTabId === initial.selectedTabId,
    "导航创建了替代 tab",
  );
  assert(
    navigated.sessionId === initial.sessionId,
    "导航改变了 Browser session identity",
  );

  stage("find");
  await page.locator('[data-testid="browser-workspace-find"]').click();
  const findInput = page.locator(
    '[data-testid="browser-workspace-find-input"]',
  );
  await findInput.fill("Example");
  await findInput.press("Enter");
  const findVisible = await findInput.isVisible();
  assert(findVisible, "查找栏未保持可见");

  stage("zoom");
  const zoomBefore = await readProjection(page);
  await page.locator('[data-testid="browser-workspace-zoom-in"]').click();
  const zoomed = await waitForProjection(
    page,
    (state) => state.zoomLabel !== zoomBefore.zoomLabel,
  );
  assert(
    zoomed.zoomLabel === "110%",
    `缩放步进不稳定: ${zoomBefore.zoomLabel} -> ${zoomed.zoomLabel}`,
  );
  await page.locator('[data-testid="browser-workspace-zoom-reset"]').click();
  const zoomReset = await waitForProjection(
    page,
    (state) => state.zoomLabel === "100%",
  );

  stage("new-tab");
  await page.locator('[data-testid="browser-workspace-new-tab"]').click();
  const secondTab = await waitForProjection(
    page,
    (state) => state.tabs.length === 2,
  );
  assert(
    secondTab.tabs.every((tab) => tab.browserSessionId === initial.sessionId),
    "新 tab 脱离 canonical session",
  );
  assert(
    new Set(secondTab.tabs.map((tab) => tab.tabId)).size === 2,
    "新 tab 复用了已有 tabId",
  );
  const secondaryTabId = secondTab.tabs.find(
    (tab) => tab.tabId !== initial.selectedTabId,
  )?.tabId;
  assert(secondaryTabId, "Gate A 未找到第二个 tab");

  stage("select-primary");
  await page
    .locator(
      `[data-browser-tab-id="${initial.selectedTabId}"] [data-testid="browser-workspace-tab-select"]`,
    )
    .click();
  const selectedFirst = await waitForProjection(
    page,
    (state) => state.selectedTabId === initial.selectedTabId,
  );
  assert(
    selectedFirst.threadId === identity.threadId,
    "选择 tab 改变了 canonical thread",
  );

  stage("close-secondary");
  await page
    .locator(
      `[data-browser-tab-id="${secondaryTabId}"] [data-testid="browser-workspace-tab-close"]`,
    )
    .click();
  const closed = await waitForProjection(
    page,
    (state) => state.tabs.length === 1,
  );
  assert(
    closed.selectedTabId === initial.selectedTabId,
    "关闭次 tab 后未恢复原选中 tab",
  );

  stage("collapse-restore");
  const browserToggle = page.locator(
    '[data-testid="task-center-browser-toggle"]',
  );
  assert(
    (await browserToggle.count()) === 1,
    "Gate A 未找到 Browser Right Surface 收起控件",
  );
  await browserToggle.click();
  const collapsed = await waitForProjection(
    page,
    (state) => !state.panelVisible && !state.workspaceVisible,
  );
  assert(!collapsed.panelVisible, "Browser 收起后 Right Surface 仍可见");
  await browserToggle.click();
  const restored = await waitForProjection(
    page,
    (state) =>
      state.panelVisible &&
      state.workspaceVisible &&
      state.selectedTabId === initial.selectedTabId,
  );
  assert(
    restored.sessionId === initial.sessionId &&
      restored.threadId === identity.threadId &&
      restored.address === closed.address,
    "Browser 恢复后 session/thread/address identity 漂移",
  );

  stage("resize");
  const desktop = restored;
  await page.setViewportSize({ width: 1440, height: 900 });
  const resized = await waitForProjection(
    page,
    (state) =>
      state.panelVisible &&
      state.viewport.width >= 200 &&
      state.viewport.height >= 200,
  );
  assert(
    resized.viewport.width !== desktop.viewport.width ||
      resized.viewport.height !== desktop.viewport.height,
    "resize 未改变 Browser viewport 投影",
  );
  assert(resized.panelVisible, "resize 后 Browser panel 不可见");
  assert(
    resized.viewport.width >= 200 && resized.viewport.height >= 200,
    "resize 后 Browser viewport 不可用",
  );
  assert(
    resized.documentOverflow === false,
    "Browser chrome 在窄视口产生横向溢出",
  );

  const screenshotPath = path.join(
    path.dirname(options.output),
    "browser-runtime-gate-a.png",
  );
  await page.screenshot({ path: screenshotPath, fullPage: false });
  screenshots.push(screenshotPath);

  const finalState = await readBrowserWorkspaceState(page);
  const assertions = {
    panelVisible: initial.panelVisible,
    workspaceVisible: initial.workspaceVisible,
    oneInitialTab: initial.tabs.length === 1,
    sameTabAfterNavigation: navigated.selectedTabId === initial.selectedTabId,
    sameSessionAfterNavigation: navigated.sessionId === initial.sessionId,
    findChromeVisible: findVisible,
    zoomInAndReset:
      zoomed.zoomLabel === "110%" && zoomReset.zoomLabel === "100%",
    secondTabUsesCanonicalSession: secondTab.tabs.every(
      (tab) => tab.browserSessionId === initial.sessionId,
    ),
    oneSelectedTabAfterCreate:
      secondTab.tabs.filter((tab) => tab.selected).length === 1,
    secondTabHasDistinctTabIds:
      new Set(secondTab.tabs.map((tab) => tab.tabId)).size === 2,
    selectAndCloseRestoresPrimary:
      closed.tabs.length === 1 &&
      closed.selectedTabId === initial.selectedTabId,
    collapseHidesRightSurface:
      !collapsed.panelVisible && !collapsed.workspaceVisible,
    restoreKeepsIdentity:
      restored.panelVisible &&
      restored.sessionId === initial.sessionId &&
      restored.threadId === identity.threadId &&
      restored.selectedTabId === initial.selectedTabId &&
      restored.address === closed.address,
    resizeKeepsUsableViewport:
      resized.panelVisible &&
      resized.viewport.width >= 200 &&
      resized.viewport.height >= 200,
    resizeHasNoOverflow: resized.documentOverflow === false,
    stableFinalIdentity:
      finalState.sessionId === initial.sessionId &&
      finalState.tabId === initial.selectedTabId,
  };
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  return sanitizeJson({
    schemaVersion: "browser-runtime-gate-a.v1",
    scenarioId: "browser-runtime-gate-a",
    proofLevel: "Gate A",
    proofScope:
      "Renderer BrowserWorkspace projection and chrome only; does not prove Agent same-WebContents execution.",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    identity: {
      browserSessionId: initial.sessionId,
      threadId: initial.threadId,
      primaryTabId: initial.selectedTabId,
    },
    captures: {
      initial,
      navigated,
      zoomed,
      zoomReset,
      secondTab,
      selectedFirst,
      closed,
      collapsed,
      restored,
      resized,
    },
    screenshots,
    assertions,
    failedAssertions,
    diagnostics: {
      consoleErrors: [],
      pageErrors: [],
    },
  });
}

async function waitForProjection(page, predicate, timeoutMs = 30_000) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < timeoutMs) {
    last = await readProjection(page);
    if (predicate(last)) {
      return last;
    }
    await sleep(250);
  }
  const invokeTrace = await page.evaluate(() => {
    try {
      return JSON.parse(
        window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "[]",
      )
        .filter(
          (entry) =>
            entry &&
            typeof entry.command === "string" &&
            entry.command.startsWith("browser_tab_"),
        )
        .slice(-12)
        .map((entry) => ({
          command: entry.command,
          transport: entry.transport,
          status: entry.status,
          args: entry.args_preview ?? null,
        }));
    } catch {
      return [];
    }
  });
  throw new Error(
    `Browser Gate A 投影等待超时: ${JSON.stringify(
      sanitizeJson({ state: last, invokeTrace }),
    )}`,
  );
}

async function readProjection(page) {
  return await page.evaluate(() => {
    const panel = document.querySelector(
      '[data-testid="right-surface-browser-panel"]',
    );
    const workspace = document.querySelector(
      '[data-testid="browser-workspace"]',
    );
    const viewport = document.querySelector(
      '[data-testid="browser-workspace-viewport"]',
    );
    const address = document.querySelector(
      '[data-testid="browser-workspace-address"]',
    );
    const zoom = document.querySelector(
      '[data-testid="browser-workspace-zoom-reset"]',
    );
    const error = document.querySelector(
      '[data-testid="browser-workspace-error"]',
    );
    const tabs = Array.from(
      document.querySelectorAll(
        '[data-testid="browser-workspace-tabs"] [data-browser-tab-id]',
      ),
    ).map((tab) => ({
      tabId: tab.getAttribute("data-browser-tab-id"),
      selected:
        tab
          .querySelector('[data-testid="browser-workspace-tab-select"]')
          ?.getAttribute("aria-selected") === "true",
      browserSessionId: tab.getAttribute("data-browser-tab-session-id") || null,
    }));
    const selected = tabs.find((tab) => tab.selected) ?? null;
    const workspaceRect = workspace?.getBoundingClientRect();
    const viewportRect = viewport?.getBoundingClientRect();
    return {
      panelVisible: Boolean(
        panel &&
        workspaceRect &&
        workspaceRect.width > 200 &&
        workspaceRect.height > 200,
      ),
      workspaceVisible: Boolean(
        workspaceRect && workspaceRect.width > 0 && workspaceRect.height > 0,
      ),
      sessionId: workspace?.getAttribute("data-browser-session-id") || null,
      threadId: workspace?.getAttribute("data-browser-thread-id") || null,
      loading: workspace?.getAttribute("data-browser-loading") === "true",
      selectedTabId:
        selected?.tabId ??
        workspace?.getAttribute("data-browser-tab-id") ??
        null,
      address: address instanceof HTMLInputElement ? address.value : null,
      zoomLabel: zoom?.textContent?.trim() || null,
      zoomDisabled: zoom instanceof HTMLButtonElement ? zoom.disabled : null,
      errorText: error?.textContent?.trim() || null,
      tabs,
      viewport: viewportRect
        ? {
            width: Math.round(viewportRect.width),
            height: Math.round(viewportRect.height),
          }
        : { width: 0, height: 0 },
      documentOverflow:
        document.documentElement.scrollWidth > window.innerWidth + 2,
    };
  });
}
