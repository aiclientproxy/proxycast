import path from "node:path";
import {
  APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
  LIVE_TAIL_COMMIT_DONE_TEXT,
  LIVE_TAIL_COMMIT_FIRST_TEXT,
  LIVE_TAIL_COMMIT_OVERFLOW_MARKER,
  LIVE_TAIL_COMMIT_PROMPT,
  LIVE_TAIL_COMMIT_TABLE_HEADER,
  LIVE_TAIL_COMMIT_TABLE_TAIL,
} from "./claw-chat-current-fixture-constants.mjs";
import { waitForBackendLedgerEntry } from "./claw-chat-current-fixture-backend-ledger.mjs";
import { sendPromptFromGui } from "./claw-chat-current-fixture-gui-actions.mjs";
import { waitForGuiChatCompleted } from "./claw-chat-current-fixture-gui-completion-waits.mjs";
import { readModelLatestTurnStatus } from "./claw-chat-current-fixture-read-model-core.mjs";
import { waitForSessionReadCompleted } from "./claw-chat-current-fixture-read-model-waits.mjs";
import {
  evaluatePageSnapshot,
  invokeAppServerFromPage,
} from "./claw-chat-current-fixture-rpc.mjs";
import { clickAndAssertRightSurface } from "./claw-chat-current-fixture-right-surface-visual.mjs";
import {
  assert,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

const RESIZE_REFLOW_VIEWPORTS = {
  wide: { width: 1280, height: 820 },
  compact: { width: 880, height: 720 },
  restored: { width: 1280, height: 820 },
};

function markdownTableRowCells(row) {
  return row
    .split("|")
    .map((cell) => cell.trim())
    .filter(Boolean);
}

const LIVE_TAIL_COMMIT_TABLE_HEADER_CELLS = markdownTableRowCells(
  LIVE_TAIL_COMMIT_TABLE_HEADER,
);
const LIVE_TAIL_COMMIT_TABLE_TAIL_CELLS = markdownTableRowCells(
  LIVE_TAIL_COMMIT_TABLE_TAIL,
);

function summarizeResizeReflowReadModel(readModel) {
  const serialized = JSON.stringify(readModel || {});
  return {
    detailItemCount: Array.isArray(readModel?.detail?.items)
      ? readModel.detail.items.length
      : null,
    threadReadItemCount: Array.isArray(
      readModel?.detail?.thread_read?.thread_items,
    )
      ? readModel.detail.thread_read.thread_items.length
      : null,
    latestTurnStatus: readModelLatestTurnStatus(readModel),
    includesPrompt: serialized.includes(LIVE_TAIL_COMMIT_PROMPT),
    includesFirstText: serialized.includes(LIVE_TAIL_COMMIT_FIRST_TEXT),
    includesOverflowMarker: serialized.includes(
      LIVE_TAIL_COMMIT_OVERFLOW_MARKER,
    ),
    includesTableHeader: serialized.includes(LIVE_TAIL_COMMIT_TABLE_HEADER),
    includesTableTail: serialized.includes(LIVE_TAIL_COMMIT_TABLE_TAIL),
    includesAssistantDone: serialized.includes(LIVE_TAIL_COMMIT_DONE_TEXT),
  };
}

async function scrollResizeReflowTailIntoView(page) {
  return await page.evaluate(async () => {
    const scrollRoot = document.querySelector(
      '[data-testid="message-list-scroll-container"]',
    );
    if (scrollRoot) {
      let previousScrollHeight = -1;
      let stableFrameCount = 0;
      for (let frame = 0; frame < 12 && stableFrameCount < 2; frame += 1) {
        await new Promise((resolve) => window.requestAnimationFrame(resolve));
        const currentScrollHeight = scrollRoot.scrollHeight;
        scrollRoot.scrollTop = currentScrollHeight;
        stableFrameCount =
          currentScrollHeight === previousScrollHeight
            ? stableFrameCount + 1
            : 0;
        previousScrollHeight = currentScrollHeight;
      }
      scrollRoot.dispatchEvent(new Event("scroll"));
    }
    return {
      scrolled: Boolean(scrollRoot),
      scrollRootTestId: scrollRoot?.getAttribute?.("data-testid") ?? null,
      distanceToBottom: scrollRoot
        ? Math.max(
            0,
            scrollRoot.scrollHeight -
              scrollRoot.clientHeight -
              scrollRoot.scrollTop,
          )
        : null,
    };
  });
}

async function captureResizeScreenshot(page, options, name) {
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-${name}.png`,
  );
  await page.screenshot({
    path: screenshotPath,
    fullPage: false,
  });
  return screenshotPath;
}

async function requestResizeReflowFilesSurface({
  page,
  appServerRequests,
  workspace,
  sessionId,
}) {
  assert(workspace?.workspaceId, "resize/reflow 缺少 workspaceId");
  assert(workspace?.rootPath, "resize/reflow 缺少 workspace rootPath");
  assert(sessionId, "resize/reflow 缺少 sessionId");

  return await invokeAppServerFromPage(
    page,
    APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST,
    {
      workspaceId: workspace.workspaceId,
      workspaceRoot: workspace.rootPath,
      sessionId,
      surfaceKind: "files",
      origin: "runtime",
      priority: "normal",
      candidateId: "internal/roadmap/test/clawstream/scenario-ledger.md",
      ttlMs: 120_000,
      metadata: {
        relativePath: "internal/roadmap/test/clawstream/scenario-ledger.md",
        title: "Clawstream scenario ledger",
      },
    },
    appServerRequests,
  );
}

async function evaluateResizeReflowSnapshot(page, label) {
  return await evaluatePageSnapshot(
    page,
    ({
      doneText,
      firstText,
      label,
      overflowMarker,
      prompt,
      tableHeaderCells,
      tableTailCells,
    }) => {
      const bodyText = document.body?.innerText || "";
      const messageListScope =
        document.querySelector('[data-testid="message-list-column"]') ||
        document.querySelector('[data-testid="message-list"]') ||
        document.querySelector('[data-testid="message-list-frame"]') ||
        document.querySelector("main") ||
        document.body;
      const turnGroups = Array.from(
        document.querySelectorAll('[data-testid="message-turn-group"]'),
      );
      const matchingTurnGroups = turnGroups.filter((group) =>
        (group.innerText || "").includes(prompt),
      );
      const scopedTurnGroup = matchingTurnGroups.at(-1) ?? null;
      const assistantBubbles = Array.from(
        (scopedTurnGroup || messageListScope).querySelectorAll(
          '[data-message-role="assistant"]',
        ),
      );
      const assistantScope =
        assistantBubbles.at(-1) ?? scopedTurnGroup ?? messageListScope;
      const assistantText = assistantScope?.innerText || "";
      const renderedTables = Array.from(
        assistantScope?.querySelectorAll("table") || [],
      );
      const renderedTableRows = renderedTables.flatMap((table) =>
        Array.from(table.querySelectorAll("tr")).map((row) => ({
          row,
          cells: Array.from(row.querySelectorAll("th, td")).map((cell) =>
            (cell.textContent || "").replace(/\s+/gu, " ").trim(),
          ),
        })),
      );
      const matchingRows = (expectedCells) =>
        renderedTableRows.filter(
          ({ cells }) =>
            cells.length === expectedCells.length &&
            cells.every((cell, index) => cell === expectedCells[index]),
        );
      const tableHeaderRows = matchingRows(tableHeaderCells);
      const tableTailRows = matchingRows(tableTailCells);
      const renderedTable = tableTailRows[0]?.row.closest("table") ?? null;
      const tableOverflowHost = findTableOverflowHost(
        renderedTable,
        assistantScope,
      );
      const textarea = document.querySelector(
        'textarea[name="agent-chat-message"]',
      );
      const inputbar =
        textarea?.closest('[data-testid="inputbar-core-container"]') ??
        document.querySelector('[data-testid="inputbar-core-container"]');
      const rightHost = document.querySelector(
        '[data-testid="workspace-right-surface-host"]',
      );
      const filesRoot = document.querySelector(
        '[data-testid="workspace-files-surface"]',
      );
      const activePane = document.querySelector(
        '[data-testid="workspace-right-surface-active-pane"]',
      );
      const threadHeader = document.querySelector(
        '[data-testid="thread-workspace-header"]',
      );
      const threadHeaderContext =
        document.querySelector(
          '[data-testid="thread-workspace-header-context"]',
        ) || threadHeader?.firstElementChild;
      const threadHeaderTitle = document.querySelector(
        '[data-testid="thread-workspace-header-title"]',
      );
      const threadHeaderStatus = document.querySelector(
        '[data-testid="thread-workspace-header-status"]',
      );
      const threadHeaderActions = document.querySelector(
        '[data-testid="thread-workspace-header-actions"]',
      );
      const taskCenterUtilityToolbar = document.querySelector(
        '[data-testid="task-center-utility-toolbar"]',
      );
      const taskCenterChromeShell = document.querySelector(
        '[data-testid="task-center-chrome-shell"]',
      );
      const taskCenterTabStrip = document.querySelector(
        '[data-testid="task-center-tab-strip"]',
      );
      const taskCenterWorkspaceBar = document.querySelector(
        '[data-testid="task-center-workspace-bar"]',
      );
      const compactModeBar = document.querySelector(
        '[data-testid="layout-compact-mode-bar"]',
      );
      const stopButtonVisible = Array.from(document.querySelectorAll("button"))
        .filter((button) => !button.disabled)
        .some((button) => {
          const label = [
            button.getAttribute("title") || "",
            button.textContent || "",
            button.getAttribute("aria-label") || "",
          ].join("\n");
          return (
            label.includes("停止") ||
            label.includes("终止") ||
            /\bStop\b/i.test(label)
          );
        });
      const viewport = {
        width: Math.round(window.innerWidth),
        height: Math.round(window.innerHeight),
      };
      const messageList = visibleInfo(messageListScope);
      const inputbarInfo = visibleInfo(inputbar);
      const textareaInfo = visibleInfo(textarea);
      const rightHostInfo = visibleInfo(rightHost);
      const filesRootInfo = visibleInfo(filesRoot);
      const activePaneInfo = visibleInfo(activePane);
      const threadHeaderInfo = visibleInfo(threadHeader);
      const threadHeaderContextInfo = visibleInfo(threadHeaderContext);
      const threadHeaderTitleInfo = visibleInfo(threadHeaderTitle);
      const threadHeaderStatusInfo = visibleInfo(threadHeaderStatus);
      const threadHeaderActionsInfo = visibleInfo(threadHeaderActions);
      const taskCenterUtilityToolbarInfo = visibleInfo(
        taskCenterUtilityToolbar,
      );
      const taskCenterUtilityToolbarVisualRect = visualBounds(
        taskCenterUtilityToolbar,
      );
      const taskCenterChromeShellInfo = visibleInfo(taskCenterChromeShell);
      const taskCenterTabStripInfo = visibleInfo(taskCenterTabStrip);
      const taskCenterWorkspaceBarInfo = visibleInfo(taskCenterWorkspaceBar);
      const compactModeBarInfo = visibleInfo(compactModeBar);
      const compactModeBarVisualRect = visualBounds(compactModeBar);
      const activeSurface = rightHost?.getAttribute("data-surface") ?? null;
      const tableInfo = visibleInfo(renderedTable);
      const tableOverflowHostInfo = visibleInfo(tableOverflowHost);
      const tableTailRange = tableTailRows[0]?.row
        ? rectToJson(tableTailRows[0].row.getBoundingClientRect())
        : null;
      const doneTextRange = textRangeRect(assistantScope, doneText);
      const turnGroupRect = scopedTurnGroup
        ? rectToJson(scopedTurnGroup.getBoundingClientRect())
        : null;
      const markerRect = doneTextRange ?? tableTailRange;
      const scrollRoot =
        document.querySelector(
          '[data-testid="message-list-scroll-container"]',
        ) ||
        document.querySelector('[data-testid="message-list-frame"]') ||
        document.querySelector('[data-testid="message-list"]') ||
        document.scrollingElement;
      const scroll = scrollMetrics(scrollRoot);
      const noTableTailInputOverlap =
        tableTailRange != null &&
        inputbarInfo.rect != null &&
        tableTailRange.bottom <= inputbarInfo.rect.top - 2;
      const noDoneTextInputOverlap =
        doneTextRange != null &&
        inputbarInfo.rect != null &&
        doneTextRange.bottom <= inputbarInfo.rect.top - 2;
      const noTurnGroupInputOverlap =
        turnGroupRect != null &&
        inputbarInfo.rect != null &&
        turnGroupRect.bottom <= inputbarInfo.rect.top - 2;
      const noTailInputOverlap =
        noTableTailInputOverlap &&
        noDoneTextInputOverlap &&
        markerRect != null &&
        markerRect.top >= 0 &&
        markerRect.bottom <= viewport.height;
      const noMessageRightOverlap =
        !rightHostInfo.visible ||
        messageList.rect == null ||
        rightHostInfo.rect == null ||
        messageList.rect.right <= rightHostInfo.rect.left + 8 ||
        rightHostInfo.rect.right <= messageList.rect.left + 8;
      const noInputRightOverlap =
        !rightHostInfo.visible ||
        inputbarInfo.rect == null ||
        rightHostInfo.rect == null ||
        inputbarInfo.rect.right <= rightHostInfo.rect.left + 8 ||
        rightHostInfo.rect.right <= inputbarInfo.rect.left + 8;
      const noTableRightOverlap =
        !rightHostInfo.visible ||
        tableOverflowHostInfo.rect == null ||
        rightHostInfo.rect == null ||
        tableOverflowHostInfo.rect.right <= rightHostInfo.rect.left + 8 ||
        rightHostInfo.rect.right <= tableOverflowHostInfo.rect.left + 8;
      const noDocumentHorizontalOverflow =
        document.documentElement.scrollWidth <= viewport.width + 1 &&
        document.body.scrollWidth <= viewport.width + 1;
      const tableOverflowed = Boolean(
        tableOverflowHost &&
        tableOverflowHost.scrollWidth > tableOverflowHost.clientWidth + 2,
      );
      const tableHostContained = Boolean(
        tableOverflowHostInfo.rect &&
        messageList.rect &&
        tableOverflowHostInfo.rect.left >= messageList.rect.left - 2 &&
        tableOverflowHostInfo.rect.right <= messageList.rect.right + 2 &&
        tableOverflowHostInfo.rect.left >= -1 &&
        tableOverflowHostInfo.rect.right <= viewport.width + 1,
      );
      const tableOverflowHandled =
        renderedTables.length === 1 &&
        tableHeaderRows.length === 1 &&
        tableTailRows.length === 1 &&
        tableInfo.visible === true &&
        tableOverflowHostInfo.visible === true &&
        tableHostContained &&
        noDocumentHorizontalOverflow;
      const inputbarAnchored =
        inputbarInfo.visible === true &&
        textareaInfo.visible === true &&
        inputbarInfo.rect != null &&
        inputbarInfo.rect.bottom <= viewport.height - 2 &&
        inputbarInfo.rect.top >= Math.round(viewport.height * 0.45);
      const rightSurfaceExpectedVisibility =
        label === "compact"
          ? rightHostInfo.visible === false && filesRootInfo.visible === false
          : rightHostInfo.visible === true && filesRootInfo.visible === true;
      const rightSurfaceStable =
        activeSurface === "files" && rightSurfaceExpectedVisibility;
      const headerContextRect =
        threadHeaderContextInfo.rect ??
        unionRects(
          threadHeaderTitleInfo.rect,
          threadHeaderStatusInfo.visible ? threadHeaderStatusInfo.rect : null,
        );
      const headerActionsRect =
        taskCenterUtilityToolbarVisualRect ?? threadHeaderActionsInfo.rect;
      const noHeaderContextActionsOverlap = Boolean(
        headerContextRect &&
        headerActionsRect &&
        !rectsOverlap(headerContextRect, headerActionsRect),
      );
      const headerChildrenContained = Boolean(
        threadHeaderInfo.rect &&
        headerContextRect &&
        headerActionsRect &&
        rectContains(threadHeaderInfo.rect, headerContextRect, 1) &&
        rectContains(threadHeaderInfo.rect, headerActionsRect, 1),
      );
      const compactModeBarExpectedVisible = label === "compact";
      const noHeaderCompactModeBarOverlap = Boolean(
        threadHeaderInfo.rect &&
        (!compactModeBarInfo.visible ||
          (compactModeBarVisualRect &&
            !rectsOverlap(threadHeaderInfo.rect, compactModeBarVisualRect))),
      );
      const compactModeBarPositionStable = compactModeBarExpectedVisible
        ? Boolean(
            compactModeBarInfo.visible &&
            compactModeBarVisualRect &&
            threadHeaderInfo.rect &&
            compactModeBarVisualRect.top >= threadHeaderInfo.rect.bottom - 1 &&
            compactModeBarVisualRect.left >= threadHeaderInfo.rect.left - 1 &&
            compactModeBarVisualRect.right <= threadHeaderInfo.rect.right + 1,
          )
        : compactModeBarInfo.visible === false;
      const activeThreadHeaderStable =
        threadHeaderInfo.visible === true &&
        threadHeaderTitleInfo.visible === true &&
        threadHeaderContextInfo.visible === true &&
        threadHeaderActionsInfo.visible === true &&
        taskCenterUtilityToolbarInfo.visible === true &&
        taskCenterChromeShellInfo.visible === false &&
        taskCenterTabStripInfo.visible === false &&
        taskCenterWorkspaceBarInfo.visible === false &&
        noHeaderContextActionsOverlap &&
        headerChildrenContained &&
        noHeaderCompactModeBarOverlap &&
        compactModeBarPositionStable;

      return {
        label,
        url: window.location.href,
        viewport,
        hasPrompt: bodyText.includes(prompt),
        turnGroupCountWithPrompt: matchingTurnGroups.length,
        assistantTextIncludesPrompt: assistantText.includes(prompt),
        hasFirstText: assistantText.includes(firstText),
        hasOverflowMarker: assistantText.includes(overflowMarker),
        hasTableHeader: tableHeaderRows.length === 1,
        hasTableTail: tableTailRows.length === 1,
        hasDoneText: assistantText.includes(doneText),
        renderedTableCount: renderedTables.length,
        tableHeaderOccurrenceCount: tableHeaderRows.length,
        tableTailOccurrenceCount: tableTailRows.length,
        startupNoteVisible: [
          "启动处理流程",
          "启动说明",
          "已接收请求",
          "正在启动",
        ].some((fragment) => assistantText.includes(fragment)),
        textareaDisabled:
          textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
        stopButtonVisible,
        markerRect,
        tableTailRange,
        doneTextRange,
        turnGroupRect,
        messageList,
        inputbar: inputbarInfo,
        textarea: textareaInfo,
        rightSurface: {
          activeSurface,
          expectedVisibility: rightSurfaceExpectedVisibility,
          host: rightHostInfo,
          activePane: activePaneInfo,
          filesRoot: filesRootInfo,
        },
        activeThreadHeader: {
          stable: activeThreadHeaderStable,
          header: threadHeaderInfo,
          context: threadHeaderContextInfo,
          title: threadHeaderTitleInfo,
          status: threadHeaderStatusInfo,
          actions: threadHeaderActionsInfo,
          toolbar: {
            ...taskCenterUtilityToolbarInfo,
            visualRect: taskCenterUtilityToolbarVisualRect,
          },
          taskCenterChromeShell: taskCenterChromeShellInfo,
          taskCenterTabStrip: taskCenterTabStripInfo,
          taskCenterWorkspaceBar: taskCenterWorkspaceBarInfo,
          compactModeBar: {
            ...compactModeBarInfo,
            visualRect: compactModeBarVisualRect,
            expectedVisible: compactModeBarExpectedVisible,
            positionStable: compactModeBarPositionStable,
          },
          noContextActionsOverlap: noHeaderContextActionsOverlap,
          childrenContained: headerChildrenContained,
          noCompactModeBarOverlap: noHeaderCompactModeBarOverlap,
        },
        table: {
          rendered: tableInfo.visible,
          rect: tableInfo.rect,
          overflowHost: tableOverflowHostInfo,
          overflowed: tableOverflowed,
          hostContained: tableHostContained,
          overflowHandled: tableOverflowHandled,
          noDocumentHorizontalOverflow,
        },
        scroll,
        messageAnchorStable:
          markerRect != null &&
          noTailInputOverlap &&
          (scroll == null || scroll.nearBottom === true),
        inputbarAnchored,
        rightSurfaceStable,
        activeThreadHeaderStable,
        noTailInputOverlap,
        noTableTailInputOverlap,
        noDoneTextInputOverlap,
        noTurnGroupInputOverlap,
        noMessageRightOverlap,
        noInputRightOverlap,
        noTableRightOverlap,
        noOverlap:
          noTailInputOverlap &&
          noTurnGroupInputOverlap &&
          noMessageRightOverlap &&
          noInputRightOverlap &&
          noTableRightOverlap &&
          noDocumentHorizontalOverflow,
        assistantTextLength: assistantText.length,
        assistantTextPreview: assistantText.slice(0, 240),
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

      function visualBounds(node) {
        if (!node) {
          return null;
        }
        const visibleRects = [node, ...node.querySelectorAll("*")]
          .map((element) => {
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            return style.display !== "none" &&
              style.visibility !== "hidden" &&
              Number(style.opacity || "1") > 0 &&
              rect.width > 0 &&
              rect.height > 0
              ? rectToJson(rect)
              : null;
          })
          .filter(Boolean);
        return visibleRects.reduce(
          (bounds, rect) => unionRects(bounds, rect),
          null,
        );
      }

      function unionRects(left, right) {
        if (!left) return right;
        if (!right) return left;
        const top = Math.min(left.top, right.top);
        const leftEdge = Math.min(left.left, right.left);
        const rightEdge = Math.max(left.right, right.right);
        const bottom = Math.max(left.bottom, right.bottom);
        return {
          x: leftEdge,
          y: top,
          width: rightEdge - leftEdge,
          height: bottom - top,
          top,
          left: leftEdge,
          right: rightEdge,
          bottom,
        };
      }

      function rectsOverlap(left, right) {
        return !(
          left.right <= right.left ||
          right.right <= left.left ||
          left.bottom <= right.top ||
          right.bottom <= left.top
        );
      }

      function rectContains(outer, inner, tolerance = 0) {
        return (
          inner.left >= outer.left - tolerance &&
          inner.top >= outer.top - tolerance &&
          inner.right <= outer.right + tolerance &&
          inner.bottom <= outer.bottom + tolerance
        );
      }

      function scrollMetrics(node) {
        if (!node) {
          return null;
        }
        const rect = node.getBoundingClientRect();
        const scrollHeight = Math.round(node.scrollHeight || 0);
        const clientHeight = Math.round(node.clientHeight || rect.height || 0);
        const scrollTop = Math.round(node.scrollTop || 0);
        const distanceToBottom = Math.round(
          Math.max(0, scrollHeight - clientHeight - scrollTop),
        );
        return {
          testId: node.getAttribute?.("data-testid") || node.tagName,
          scrollHeight,
          clientHeight,
          scrollTop,
          distanceToBottom,
          overflowed: scrollHeight > clientHeight + 4,
          nearBottom: !scrollHeight || distanceToBottom <= 220,
          rect: rectToJson(rect),
        };
      }

      function findTableOverflowHost(table, scope) {
        let node = table?.parentElement ?? null;
        while (node) {
          const style = window.getComputedStyle(node);
          if (
            ["auto", "scroll"].includes(style.overflowX) ||
            node.scrollWidth > node.clientWidth + 2
          ) {
            return node;
          }
          if (node === scope) {
            break;
          }
          node = node.parentElement;
        }
        return table?.parentElement ?? null;
      }

      function textRangeRect(root, needle) {
        if (!root || !needle) {
          return null;
        }
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        let node = walker.nextNode();
        while (node) {
          const index = String(node.nodeValue || "").indexOf(needle);
          if (index >= 0) {
            const range = document.createRange();
            range.setStart(node, index);
            range.setEnd(node, index + needle.length);
            const rect = range.getBoundingClientRect();
            range.detach();
            return rect ? rectToJson(rect) : null;
          }
          node = walker.nextNode();
        }
        return null;
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
      doneText: LIVE_TAIL_COMMIT_DONE_TEXT,
      firstText: LIVE_TAIL_COMMIT_FIRST_TEXT,
      label,
      overflowMarker: LIVE_TAIL_COMMIT_OVERFLOW_MARKER,
      prompt: LIVE_TAIL_COMMIT_PROMPT,
      tableHeaderCells: LIVE_TAIL_COMMIT_TABLE_HEADER_CELLS,
      tableTailCells: LIVE_TAIL_COMMIT_TABLE_TAIL_CELLS,
    },
  );
}

function isResizeReflowSnapshotReady(snapshot, expectedViewport) {
  return (
    snapshot?.viewport?.width === expectedViewport.width &&
    snapshot?.viewport?.height === expectedViewport.height &&
    snapshot.hasPrompt === true &&
    snapshot.turnGroupCountWithPrompt === 1 &&
    snapshot.hasFirstText === true &&
    snapshot.hasOverflowMarker === true &&
    snapshot.hasTableHeader === true &&
    snapshot.hasTableTail === true &&
    snapshot.hasDoneText === true &&
    snapshot.renderedTableCount === 1 &&
    snapshot.tableHeaderOccurrenceCount === 1 &&
    snapshot.tableTailOccurrenceCount === 1 &&
    snapshot.table?.overflowHandled === true &&
    snapshot.startupNoteVisible === false &&
    snapshot.textareaDisabled === false &&
    snapshot.stopButtonVisible === false &&
    snapshot.messageAnchorStable === true &&
    snapshot.inputbarAnchored === true &&
    snapshot.activeThreadHeaderStable === true &&
    snapshot.rightSurfaceStable === true &&
    snapshot.noOverlap === true
  );
}

async function waitForResizeReflowSnapshot(page, options, { label, viewport }) {
  await page.setViewportSize(viewport);
  await page.evaluate(() => window.dispatchEvent(new Event("resize")));
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluateResizeReflowSnapshot(page, label);
    lastSnapshot = snapshot;
    if (isResizeReflowSnapshotReady(snapshot, viewport)) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `Electron resize/reflow snapshot 未稳定: ${label}; snapshot=${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export async function runElectronResizeReflowScenario({
  page,
  options,
  workspace,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  const result = {};

  logStage("send-electron-resize-reflow-prompt-from-gui");
  result.electronResizeReflowInputSend = sanitizeJson(
    await sendPromptFromGui(page, options, LIVE_TAIL_COMMIT_PROMPT),
  );

  logStage("wait-electron-resize-reflow-backend-turn-start");
  const backendTurnStart = await waitForBackendLedgerEntry(
    runtimeEnv.backendLedgerPath,
    (entry) =>
      entry.kind === "turnStart" && entry.inputText === LIVE_TAIL_COMMIT_PROMPT,
    options,
  );
  result.electronResizeReflowBackendTurnStart = sanitizeJson({
    sessionId: backendTurnStart.entry.sessionId,
    turnId: backendTurnStart.entry.turnId,
    inputText: backendTurnStart.entry.inputText,
    ledgerCount: backendTurnStart.ledger.length,
  });

  logStage("wait-gui-electron-resize-reflow-completed");
  result.guiElectronResizeReflowCompleted = sanitizeJson(
    await waitForGuiChatCompleted(page, options, {
      prompt: LIVE_TAIL_COMMIT_PROMPT,
      doneText: LIVE_TAIL_COMMIT_DONE_TEXT,
      summaryText: LIVE_TAIL_COMMIT_FIRST_TEXT,
      requiredVisibleTexts: [
        LIVE_TAIL_COMMIT_OVERFLOW_MARKER,
        LIVE_TAIL_COMMIT_TABLE_HEADER_CELLS[0],
        LIVE_TAIL_COMMIT_TABLE_TAIL_CELLS[1],
      ],
    }),
  );

  logStage("wait-read-model-electron-resize-reflow-completed");
  const readModel = await waitForSessionReadCompleted(
    page,
    options,
    appServerRequests,
    {
      prompt: LIVE_TAIL_COMMIT_PROMPT,
      doneText: LIVE_TAIL_COMMIT_DONE_TEXT,
      summaryText: LIVE_TAIL_COMMIT_FIRST_TEXT,
    },
  );
  result.readModelElectronResizeReflowCompleted = sanitizeJson(
    summarizeResizeReflowReadModel(readModel),
  );

  const liveTailLedger = await waitForBackendLedgerEntry(
    runtimeEnv.backendLedgerPath,
    (entry) => entry.kind === "liveTailCommitCompleted",
    options,
  );
  result.electronResizeReflowBackendCompleted = sanitizeJson({
    threadId: liveTailLedger.entry.threadId,
    turnId: liveTailLedger.entry.turnId,
    itemId: liveTailLedger.entry.itemId,
    droppedEventType: liveTailLedger.entry.droppedEventType,
    repairEventType: liveTailLedger.entry.repairEventType,
    terminalEventType: liveTailLedger.entry.terminalEventType,
    emittedEventTypes: liveTailLedger.entry.emittedEventTypes,
    firstText: liveTailLedger.entry.firstText,
    overflowMarker: liveTailLedger.entry.overflowMarker,
    tableHeader: liveTailLedger.entry.tableHeader,
    tableTail: liveTailLedger.entry.tableTail,
    ledgerCount: liveTailLedger.ledger.length,
  });

  logStage("request-electron-resize-reflow-files-surface");
  result.electronResizeReflowFilesSurfaceRequest = sanitizeJson(
    await requestResizeReflowFilesSurface({
      page,
      appServerRequests,
      workspace,
      sessionId: backendTurnStart.entry.sessionId,
    }),
  );

  logStage("open-electron-resize-reflow-files-surface");
  result.electronResizeReflowFilesSurface = sanitizeJson(
    await clickAndAssertRightSurface(page, options, {
      surfaceKind: "files",
      toggleTestId: "task-center-files-toggle",
      rootTestId: "workspace-files-surface",
      requireCanvasPanelFill: false,
    }),
  );

  await page.setViewportSize(RESIZE_REFLOW_VIEWPORTS.wide);
  logStage("scroll-electron-resize-reflow-tail-into-view");
  result.electronResizeReflowTailScroll = sanitizeJson(
    await scrollResizeReflowTailIntoView(page),
  );

  const snapshots = {};
  const screenshots = {};
  for (const [label, viewport] of Object.entries(RESIZE_REFLOW_VIEWPORTS)) {
    logStage(`capture-electron-resize-reflow-${label}`);
    snapshots[label] = sanitizeJson(
      await waitForResizeReflowSnapshot(page, options, {
        label,
        viewport,
      }),
    );
    screenshots[label] = await captureResizeScreenshot(
      page,
      options,
      `electron-resize-reflow-${label}`,
    );
  }

  assert(
    snapshots.wide?.rightSurface?.activeSurface ===
      snapshots.compact?.rightSurface?.activeSurface &&
      snapshots.compact?.rightSurface?.activeSurface ===
        snapshots.restored?.rightSurface?.activeSurface,
    `Electron resize/reflow right surface owner 不稳定: ${JSON.stringify(
      sanitizeJson({
        wide: snapshots.wide?.rightSurface,
        compact: snapshots.compact?.rightSurface,
        restored: snapshots.restored?.rightSurface,
      }),
    )}`,
  );

  result.electronResizeReflowLayout = sanitizeJson({
    viewports: RESIZE_REFLOW_VIEWPORTS,
    snapshots,
    screenshots,
    stableViewportCount: Object.values(snapshots).filter(Boolean).length,
    screenshotCount: Object.values(screenshots).filter(Boolean).length,
  });

  return result;
}
