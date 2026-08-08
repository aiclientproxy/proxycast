import path from "node:path";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  assert,
  evaluatePageSnapshot,
  sanitizeJson,
  sleep,
} from "./local-history-import-smoke-utils.mjs";
import {
  CONTINUE_ASSISTANT_TEXT,
  CONTINUE_USER_TEXT,
  IMPORTED_ASSISTANT_SUMMARY_TEXT,
  IMPORTED_MARKDOWN_HEADING_TEXT,
  IMPORTED_PREVIEW_DOCX_FILE,
  IMPORTED_PREVIEW_DOCX_TEXT,
  IMPORTED_PREVIEW_HTML_FILE,
  IMPORTED_PREVIEW_HTML_TEXT,
  IMPORTED_PREVIEW_MARKDOWN_FILE,
  IMPORTED_PREVIEW_MARKDOWN_TEXT,
  IMPORTED_PREVIEW_PPTX_FILE,
  IMPORTED_PREVIEW_PPTX_TEXT,
  IMPORTED_PREVIEW_PDF_FILE,
  IMPORTED_PREVIEW_PDF_TEXT,
  IMPORTED_PREVIEW_XLSX_FILE,
  IMPORTED_PREVIEW_XLSX_TEXT,
  IMPORTED_REASONING_TEXT,
  IMPORTED_USER_TEXT,
  IMPORTED_WEB_SEARCH_QUERY,
  IMPORTED_WEB_SEARCH_SOURCE_LABEL,
  IMPORTED_WEB_SEARCH_TITLE,
  IMPORTED_WEB_SEARCH_URL,
  LEGACY_CONTINUATION_SENTINEL,
  SOURCE_THREAD_ID,
} from "./local-history-import-click-through-fixture.mjs";

const VISUAL_AUDIT_VIEWPORTS = [
  { label: "desktop", width: 1440, height: 1000 },
  { label: "compact", width: 1100, height: 820 },
  { label: "narrow", width: 820, height: 900 },
];

export async function clearInvokeBuffers(page) {
  await evaluatePageSnapshot(page, () => {
    window.localStorage.removeItem("lime_invoke_error_buffer_v1");
    window.localStorage.removeItem("lime_invoke_trace_buffer_v1");
  });
}

async function waitForUiSnapshot(page, options, predicate, failureLabel) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluatePageSnapshot(page, () => {
      const bodyText = document.body?.innerText || "";
      const dialog = document.querySelector(
        '[data-testid="app-sidebar-conversation-import-dialog"]',
      );
      const textarea = document.querySelector(
        'textarea[name="agent-chat-message"]',
      );
      const sendButton = Array.from(document.querySelectorAll("button")).find(
        (button) => {
          const label = [
            button.getAttribute("aria-label") || "",
            button.getAttribute("title") || "",
            button.textContent || "",
          ].join("\n");
          return label.includes("发送") || /\bSend\b/i.test(label);
        },
      );
      return {
        url: window.location.href,
        title: document.title || "",
        bodyText,
        dialogVisible: Boolean(dialog),
        importButtonVisible: Boolean(
          document.querySelector(
            '[data-testid="app-sidebar-import-conversation-button"]',
          ),
        ),
        importConfirmVisible: Boolean(
          document.querySelector(
            '[data-testid="app-sidebar-conversation-import-confirm"]',
          ),
        ),
        importConfirmDisabled:
          document.querySelector(
            '[data-testid="app-sidebar-conversation-import-confirm"]',
          ) instanceof HTMLButtonElement
            ? document.querySelector(
                '[data-testid="app-sidebar-conversation-import-confirm"]',
              ).disabled
            : null,
        importProgressVisible: Boolean(
          document.querySelector(
            '[data-testid="app-sidebar-conversation-import-progress"]',
          ),
        ),
        importCloseVisible: Boolean(
          document.querySelector(
            '[data-testid="app-sidebar-conversation-import-close"]',
          ),
        ),
        textareaVisible: textarea instanceof HTMLTextAreaElement,
        textareaDisabled:
          textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
        textareaValue:
          textarea instanceof HTMLTextAreaElement ? textarea.value : null,
        sendButtonVisible: sendButton instanceof HTMLButtonElement,
        sendButtonDisabled:
          sendButton instanceof HTMLButtonElement ? sendButton.disabled : null,
        operationalDetailRowCount: document.querySelectorAll(
          '[data-testid="tool-call-row"]',
        ).length,
        operationalTimelineDetailsCount: Array.from(
          document.querySelectorAll(
            'details[data-testid^="agent-thread-block:"][data-testid$=":process"], details[data-testid^="agent-thread-block:"][data-testid$=":approval"]',
          ),
        ).filter((details) => details.open).length,
        deferredHistoricalPreviewCount: document.querySelectorAll(
          '[data-testid="message-list-historical-markdown-preview"], [data-testid="message-list-historical-assistant-preview"], [data-testid="message-list-long-history-preview"]',
        ).length,
        environmentTriggerVisible: Boolean(
          document.querySelector(
            '[data-testid="task-center-environment-trigger"]',
          ),
        ),
        environmentPopoverVisible: Boolean(
          document.querySelector(
            '[data-testid="task-center-environment-popover"]',
          ),
        ),
        traceRaw: window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        errorRaw: window.localStorage.getItem("lime_invoke_error_buffer_v1"),
      };
    });
    if (!snapshot) {
      await sleep(options.intervalMs);
      continue;
    }
    lastSnapshot = snapshot;
    if (predicate(snapshot)) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `${failureLabel}: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

export async function clickSidebarImport(page, options) {
  await waitForUiSnapshot(
    page,
    options,
    (snapshot) => snapshot.importButtonVisible,
    "侧边栏导入按钮未出现",
  );
  await page
    .locator('[data-testid="app-sidebar-import-conversation-button"]')
    .click();
}

function extractPreviewTraceMethods(rawTrace) {
  return Array.from(new Set(extractPreviewTraceMethodCalls(rawTrace)));
}

function extractPreviewTraceMethodCalls(rawTrace) {
  const methods = [];
  let entries = [];
  try {
    const parsed = JSON.parse(rawTrace || "[]");
    entries = Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
  for (const entry of entries) {
    if (entry?.command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
      continue;
    }
    const lines = Array.isArray(entry?.args_preview?.request?.lines)
      ? entry.args_preview.request.lines
      : [];
    for (const line of lines) {
      try {
        const message = JSON.parse(line);
        if (typeof message?.method === "string") {
          methods.push(message.method);
        }
      } catch {
        // ignore non JSON trace line
      }
    }
  }
  return methods;
}

export async function waitForImportPreview(page, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  let reopenedAfterPreviewTrace = false;
  const isReady = (snapshot) =>
    snapshot.dialogVisible &&
    snapshot.importConfirmVisible &&
    snapshot.importConfirmDisabled === false &&
    snapshot.bodyText.includes("本地历史导入点击闭环") &&
    snapshot.bodyText.includes(IMPORTED_USER_TEXT) &&
    snapshot.bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT) &&
    snapshot.bodyText.includes("将保留工具、命令、补丁、确认与思考记录") &&
    snapshot.bodyText.includes("命令") &&
    snapshot.bodyText.includes("补丁");

  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await waitForUiSnapshot(
      page,
      { ...options, timeoutMs: Math.max(options.intervalMs * 2, 1000) },
      () => true,
      "读取导入弹窗快照失败",
    );
    lastSnapshot = snapshot;
    if (isReady(snapshot)) {
      return snapshot;
    }

    const traceMethods = extractPreviewTraceMethods(snapshot.traceRaw);
    const previewRpcCompleted =
      traceMethods.includes("conversationImport/source/scan") &&
      traceMethods.includes("conversationImport/thread/preview");
    if (
      previewRpcCompleted &&
      !snapshot.dialogVisible &&
      snapshot.importButtonVisible &&
      !reopenedAfterPreviewTrace
    ) {
      reopenedAfterPreviewTrace = true;
      await page
        .locator('[data-testid="app-sidebar-import-conversation-button"]')
        .click();
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `本地历史导入弹窗预览未完成: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

export async function confirmImport(page, options) {
  await page
    .locator('[data-testid="app-sidebar-conversation-import-confirm"]')
    .click();
  const runningSnapshot = await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.dialogVisible &&
      snapshot.importProgressVisible &&
      snapshot.importCloseVisible &&
      snapshot.bodyText.includes("导入将在后台继续"),
    "后台导入启动后未显示可关闭进度",
  );

  await page
    .locator('[data-testid="app-sidebar-conversation-import-close"]')
    .click();
  const closedSnapshot = await waitForUiSnapshot(
    page,
    options,
    (snapshot) => !snapshot.dialogVisible && snapshot.importButtonVisible,
    "关闭导入弹窗后侧栏入口未恢复",
  );

  await page
    .locator('[data-testid="app-sidebar-import-conversation-button"]')
    .click();
  const resumedSnapshot = await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.dialogVisible &&
      snapshot.importConfirmVisible &&
      snapshot.importConfirmDisabled === false &&
      snapshot.bodyText.includes("正在导入") &&
      snapshot.bodyText.includes("查看进度"),
    "重新打开导入弹窗后未附着后台 job",
  );

  await page
    .locator('[data-testid="app-sidebar-conversation-import-confirm"]')
    .click();
  const importedSnapshot = await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      !snapshot.dialogVisible &&
      snapshot.textareaVisible &&
      snapshot.bodyText.includes(IMPORTED_USER_TEXT) &&
      snapshot.bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT),
    "确认导入后未进入可继续对话的会话页",
  );
  const traceMethodCalls = extractPreviewTraceMethodCalls(
    importedSnapshot.traceRaw,
  );
  return {
    ...importedSnapshot,
    backgroundResume: {
      started: runningSnapshot.importProgressVisible,
      closed: !closedSnapshot.dialogVisible,
      reattached: resumedSnapshot.bodyText.includes("查看进度"),
      commitRequestCount: traceMethodCalls.filter(
        (method) => method === "conversationImport/thread/commit",
      ).length,
      jobReadRequestCount: traceMethodCalls.filter(
        (method) => method === "conversationImport/job/read",
      ).length,
    },
  };
}

export async function inspectImportedHistoricalTimelineSummary(page, options) {
  const selector =
    '[data-testid="message-list-historical-timeline-preview:leading"]';
  const processDetailsSelector =
    'details[data-testid$=":process"][data-details-available="true"]';
  const processSummarySelector = `${processDetailsSelector} > summary`;
  const startedAt = Date.now();
  let lastSnapshot = null;
  const readTimelineSnapshot = async () =>
    page.evaluate(
      ({ processDetailsSelector, timelineSelector }) => {
        const preview = document.querySelector(timelineSelector);
        const processDetails = Array.from(
          document.querySelectorAll(processDetailsSelector),
        );
        return {
          previewVisible: preview instanceof HTMLElement,
          historicalSummaryVisible: preview instanceof HTMLElement,
          interactive:
            preview instanceof HTMLButtonElement ||
            preview instanceof HTMLAnchorElement ||
            preview?.getAttribute("role") === "button",
          previewAriaExpanded: preview?.getAttribute("aria-expanded") || null,
          previewText: preview?.textContent || "",
          timelineVisible: Boolean(
            document.querySelector('[data-testid="agent-thread-flow"]'),
          ),
          expandableProcessCount: processDetails.length,
          openExpandableProcessCount: processDetails.filter(
            (details) => details.open,
          ).length,
          expandedProcessDetailsVisible: processDetails.some(
            (details) =>
              details.open &&
              details.querySelector(
                '[data-testid^="agent-thread-block:"][data-testid$=":details"]',
              ) instanceof HTMLElement,
          ),
          fileChangeRowCount: document.querySelectorAll(
            '[data-testid="file-changes-summary-file-row"]',
          ).length,
          toolRowSamples: Array.from(
            document.querySelectorAll('[data-testid="tool-call-row"]'),
          )
            .slice(0, 3)
            .map((row) => ({
              text: row.textContent || "",
              outer: row.outerHTML.slice(0, 500),
              parentTestId:
                row.parentElement?.getAttribute("data-testid") || null,
            })),
        };
      },
      { processDetailsSelector, timelineSelector: selector },
    );

  while (Date.now() - startedAt < options.timeoutMs) {
    lastSnapshot = await readTimelineSnapshot();
    if (!lastSnapshot?.previewVisible) {
      await sleep(options.intervalMs);
      continue;
    }
    if (!lastSnapshot.interactive) {
      return {
        expanded: false,
        previewVisible: lastSnapshot.previewVisible,
        historicalSummaryVisible: lastSnapshot.historicalSummaryVisible,
        interactive: false,
        previewAriaExpanded: lastSnapshot.previewAriaExpanded,
        operationalDetailRowCount: await page
          .locator('[data-testid="tool-call-row"]')
          .count(),
        operationalTimelineDetailsCount: await page
          .locator(
            'details[data-testid$=":process"][data-details-available="true"], details[data-testid$=":approval"][data-details-available="true"]',
          )
          .count(),
        deferredHistoricalPreviewCount: await page
          .locator(
            '[data-testid="message-list-historical-markdown-preview"], [data-testid="message-list-historical-assistant-preview"], [data-testid="message-list-long-history-preview"]',
          )
          .count(),
        previewText: lastSnapshot.previewText,
        toolRowSamples: lastSnapshot.toolRowSamples,
      };
    }

    const initialSnapshot = lastSnapshot;
    await page.locator(selector).click();
    await page
      .locator('[data-testid="agent-thread-flow"]')
      .first()
      .waitFor({ state: "visible", timeout: options.timeoutMs });
    const expandedSnapshot = await readTimelineSnapshot();

    const processSummary = page.locator(processSummarySelector).first();
    const expandableProcessSummaryCount = await page
      .locator(processSummarySelector)
      .count();
    let processDetailsSnapshot = expandedSnapshot;
    if (
      expandableProcessSummaryCount > 0 &&
      expandedSnapshot.openExpandableProcessCount === 0
    ) {
      if (await processSummary.count()) {
        await processSummary.click();
      } else {
        await page.locator(processSummarySelector).first().click();
      }
      await page.waitForFunction(
        (detailsSelector) =>
          Array.from(document.querySelectorAll(detailsSelector)).some(
            (details) =>
              details.open &&
              (details.querySelector('[data-testid$=":details"]') ||
                details.querySelector('[data-testid="tool-call-row"]')),
          ),
        processDetailsSelector,
        { timeout: options.timeoutMs },
      );
      processDetailsSnapshot = await readTimelineSnapshot();
    }

    if (
      expandableProcessSummaryCount > 0 &&
      processDetailsSnapshot.openExpandableProcessCount > 0
    ) {
      const openProcessDetailsSelector = `${processDetailsSelector}[open]`;
      while (
        (await page.locator(openProcessDetailsSelector).count()) > 0
      ) {
        const openCount = await page
          .locator(openProcessDetailsSelector)
          .count();
        await page
          .locator(`${openProcessDetailsSelector} > summary`)
          .first()
          .click();
        await page.waitForFunction(
          ({ detailsSelector, previousCount }) =>
            document.querySelectorAll(detailsSelector).length < previousCount,
          { detailsSelector: openProcessDetailsSelector, previousCount: openCount },
          { timeout: options.timeoutMs },
        );
      }
    }

    const collapsedAfterInteraction = await waitForUiSnapshot(
      page,
      options,
      (snapshot) =>
        snapshot.dialogVisible === false &&
        snapshot.operationalDetailRowCount === 0 &&
        snapshot.operationalTimelineDetailsCount === 0 &&
        snapshot.deferredHistoricalPreviewCount === 0,
      "展开历史步骤后未能恢复折叠态",
    );

    return {
      expanded: true,
      previewVisible: initialSnapshot.previewVisible,
      historicalSummaryVisible: initialSnapshot.historicalSummaryVisible,
      interactive: initialSnapshot.interactive === true,
      previewAriaExpanded: initialSnapshot.previewAriaExpanded,
      timelineVisible: expandedSnapshot.timelineVisible,
      expandableProcessCount: expandedSnapshot.expandableProcessCount,
      expandedProcessDetailsVisible:
        processDetailsSnapshot.expandedProcessDetailsVisible,
      fileChangeRowCount: expandedSnapshot.fileChangeRowCount,
      operationalDetailRowCount:
        collapsedAfterInteraction.operationalDetailRowCount,
      operationalTimelineDetailsCount:
        collapsedAfterInteraction.operationalTimelineDetailsCount,
      deferredHistoricalPreviewCount:
        collapsedAfterInteraction.deferredHistoricalPreviewCount,
      previewText: initialSnapshot.previewText,
      toolRowSamples: processDetailsSnapshot.toolRowSamples,
    };
  }
  throw new Error(
    `导入历史摘要未出现: ${JSON.stringify(sanitizeJson(lastSnapshot))}`,
  );
}

function hasAnyText(snapshot, values) {
  return values.some((value) => snapshot.bodyText.includes(value));
}

function hasPatchEvidenceText(bodyText) {
  const currentFileArtifactVisible =
    bodyText.includes("lib.rs") && bodyText.includes("打开文件");
  return (
    (hasAnyText({ bodyText }, ["补丁", "Patch", "patch", "已编辑"]) ||
      currentFileArtifactVisible) &&
    hasAnyText({ bodyText }, ["src/lib.rs", "lib.rs", "文件"])
  );
}

export function summarizeImportedDetailsSnapshot(
  snapshot,
  readModelSummary = null,
) {
  const bodyText = snapshot?.bodyText || "";
  const hasOperationalDomSnapshot =
    Number.isFinite(snapshot?.operationalDetailRowCount) &&
    Number.isFinite(snapshot?.operationalTimelineDetailsCount);
  const hasHistoricalReasoningText = bodyText.includes(IMPORTED_REASONING_TEXT);
  const hasHistoricalCommandText = bodyText.includes("npm test");
  const hasHistoricalCommandOutput = bodyText.includes("ok");
  const hasHistoricalApprovalText = hasAnyText({ bodyText }, [
    "导入的权限记录",
    "已导入，只读记录",
    "权限记录",
    "审批",
    "确认",
    "权限请求",
    "Approval",
    "approval",
  ]);
  return {
    hasImportedUserMessage: bodyText.includes(IMPORTED_USER_TEXT),
    hasImportedAssistantMessage: bodyText.includes(
      IMPORTED_ASSISTANT_SUMMARY_TEXT,
    ),
    hasReasoningItem: readModelSummary?.hasReasoningItem === true,
    hasCommandText: readModelSummary?.hasCommandItem === true,
    hasCommandItem: readModelSummary?.hasCommandItem === true,
    hasSearchItem: readModelSummary?.hasWebSearchItem === true,
    hasApprovalItem: readModelSummary?.hasApprovalItem === true,
    hasPatchText: hasPatchEvidenceText(bodyText),
    historicalOperationalDetailsHidden: hasOperationalDomSnapshot
      ? snapshot.operationalDetailRowCount === 0 &&
        snapshot.operationalTimelineDetailsCount === 0 &&
        snapshot.deferredHistoricalPreviewCount === 0
      : !hasHistoricalReasoningText &&
        !hasHistoricalCommandText &&
        !hasHistoricalCommandOutput &&
        !bodyText.includes(IMPORTED_WEB_SEARCH_QUERY) &&
        !hasHistoricalApprovalText,
    hidesRawImportedCommand:
      !bodyText.includes("Approve imported command") &&
      !bodyText.includes("imported_read_only"),
  };
}

export function summarizeContinuationSnapshot(snapshot) {
  const bodyText = snapshot?.bodyText || "";
  return {
    hasContinueUserMessage: bodyText.includes(CONTINUE_USER_TEXT),
    hasContinueAssistantMessage: bodyText.includes(CONTINUE_ASSISTANT_TEXT),
    hidesFixtureSentinel: !bodyText.includes(LEGACY_CONTINUATION_SENTINEL),
  };
}

export function summarizeImportPreviewSnapshot(snapshot) {
  const bodyText = snapshot?.bodyText || "";
  return {
    hidesRawSourceEventNames:
      !bodyText.includes("event_msg") &&
      !bodyText.includes("agent_message") &&
      !bodyText.includes("user_message") &&
      !bodyText.includes("response_item"),
    hasReadableSourceLabels:
      bodyText.includes("来源行 #") &&
      bodyText.includes("用户消息") &&
      bodyText.includes("助手回复"),
  };
}

export async function inspectEnvironmentPopoverImportBoundary(page, options) {
  await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.textareaVisible &&
      snapshot.bodyText.includes(IMPORTED_USER_TEXT) &&
      snapshot.bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT),
    "导入会话页未稳定，无法检查环境信息边界",
  );
  const forbiddenEnvironmentTexts = [
    "本地历史导入",
    "已还原",
    "Approve Codex command",
    "thread-codex",
    SOURCE_THREAD_ID,
  ];
  const triggerSnapshot = await evaluatePageSnapshot(
    page,
    (forbiddenTexts) => {
      const importedRunControl = document.querySelector(
        '[data-testid="task-center-run-control-imported"]',
      );
      const taskRail = document.querySelector(
        '[data-testid="task-center-task-rail"]',
      );
      const taskRailText = taskRail?.textContent || "";
      return {
        triggerVisible: Boolean(
          document.querySelector(
            '[data-testid="task-center-environment-trigger"]',
          ),
        ),
        importedRunControlVisible: Boolean(importedRunControl),
        taskRailText,
        hidesImportedRunControlCard:
          !importedRunControl &&
          !forbiddenTexts.some((text) => taskRailText.includes(text)) &&
          !/\bcodex\b/i.test(taskRailText),
      };
    },
    forbiddenEnvironmentTexts,
  );
  if (!triggerSnapshot?.triggerVisible) {
    return {
      popoverVisible: false,
      popoverText: "",
      taskRailText: triggerSnapshot?.taskRailText || "",
      importedRunControlVisible:
        triggerSnapshot?.importedRunControlVisible === true,
      hidesImportedRunControlCard:
        triggerSnapshot?.hidesImportedRunControlCard === true,
      checkedMode: "environment-trigger-not-rendered",
    };
  }

  await page.locator('[data-testid="task-center-environment-trigger"]').click();
  const snapshot = await waitForUiSnapshot(
    page,
    { ...options, timeoutMs: Math.min(options.timeoutMs, 10_000) },
    (current) => current.environmentPopoverVisible === true,
    "环境信息弹层未打开",
  );
  const result = await page.evaluate((forbiddenTexts) => {
    const popover = document.querySelector(
      '[data-testid="task-center-environment-popover"]',
    );
    const taskRail = popover?.querySelector(
      '[data-testid="task-center-task-rail"]',
    );
    const importedRunControl = popover?.querySelector(
      '[data-testid="task-center-run-control-imported"]',
    );
    const popoverText = popover?.textContent || "";
    const taskRailText = taskRail?.textContent || "";
    return {
      popoverVisible: Boolean(popover),
      popoverText,
      taskRailText,
      importedRunControlVisible: Boolean(importedRunControl),
      hidesImportedRunControlCard:
        !importedRunControl &&
        !forbiddenTexts.some((text) => taskRailText.includes(text)) &&
        !/\bcodex\b/i.test(taskRailText),
    };
  }, forbiddenEnvironmentTexts);
  return {
    ...result,
    snapshotUrl: snapshot.url,
    checkedMode: "environment-popover",
  };
}

export async function inspectImportedHistoryBanner(page, options) {
  await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.textareaVisible &&
      snapshot.bodyText.includes(IMPORTED_USER_TEXT) &&
      snapshot.bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT),
    "导入会话页未稳定，无法检查本地历史摘要",
  );

  return await page.evaluate(() => {
    const banner = document.querySelector(
      '[data-testid="imported-source-banner"]',
    );
    const text = banner?.textContent || "";
    return {
      visible: Boolean(banner),
      text,
      hiddenFromMainTimeline: !banner,
    };
  });
}

export async function inspectSidebarImportDiscoverability(page, options) {
  await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.textareaVisible &&
      snapshot.bodyText.includes(IMPORTED_USER_TEXT) &&
      snapshot.bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT),
    "导入会话页未稳定，无法检查侧栏会话入口",
  );

  return await page.evaluate(() => {
    const shelf = document.querySelector(
      '[data-testid="app-sidebar-conversation-shelf"]',
    );
    const projectList = document.querySelector(
      '[data-testid="app-sidebar-project-conversations"]',
    );
    const recentList = document.querySelector(
      '[data-testid="app-sidebar-recent-conversations"]',
    );
    const shelfText = shelf?.textContent || "";
    const projectText = projectList?.textContent || "";
    const recentText = recentList?.textContent || "";
    const discoverableLabels = ["本地历史导入点击闭环", "请运行测试并修复失败"];
    const importedEntryVisible = discoverableLabels.some((label) =>
      shelfText.includes(label),
    );
    return {
      visible: Boolean(shelf),
      shelfText,
      projectText,
      recentText,
      importedEntryVisible,
      emptyStateOnly:
        !importedEntryVisible &&
        (projectText.includes("暂无聊天") || recentText.includes("暂无聊天")),
    };
  });
}

export async function inspectImportedAttachmentPreview(page, options) {
  await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.textareaVisible &&
      snapshot.bodyText.includes(IMPORTED_USER_TEXT) &&
      snapshot.bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT),
    "导入会话页未稳定，无法检查导入图片附件",
  );

  const before = await page.evaluate(() => {
    const attachments = Array.from(
      document.querySelectorAll('[data-testid^="message-image-attachment-"]'),
    ).filter(
      (node) =>
        node instanceof HTMLImageElement &&
        node.getAttribute("data-testid") !== null,
    );
    const openButtons = Array.from(
      document.querySelectorAll(
        '[data-testid^="message-image-attachment-open-"]',
      ),
    );
    return {
      attachmentCount: attachments.length,
      openButtonCount: openButtons.length,
      firstAttachmentSrc:
        attachments[0] instanceof HTMLImageElement ? attachments[0].src : "",
      firstOpenButtonText:
        openButtons[0] instanceof HTMLElement ? openButtons[0].textContent : "",
    };
  });

  assert(
    before.attachmentCount > 0,
    `导入会话未渲染图片附件: ${JSON.stringify(sanitizeJson(before))}`,
  );
  assert(
    before.openButtonCount > 0,
    `导入图片附件没有打开入口: ${JSON.stringify(sanitizeJson(before))}`,
  );

  await page
    .locator('[data-testid^="message-image-attachment-open-"]')
    .first()
    .click();

  const startedAt = Date.now();
  let result = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 15_000)) {
    result = await page.evaluate(() => {
      const media = document.querySelector(
        '[data-testid="preview-artifact-image"]',
      );
      const workbench = document.querySelector(
        '[data-testid="artifact-workbench-shell"]',
      );
      const emptySurface = document.querySelector(
        '[data-testid="artifact-empty-surface"]',
      );
      const canvasPreview = document.querySelector(
        '[data-testid^="canvas-workbench-preview-"]',
      );
      const attachmentImages = Array.from(
        document.querySelectorAll('[data-testid^="message-image-attachment-"]'),
      ).filter((node) => node instanceof HTMLImageElement);
      return {
        previewImageVisible: media instanceof HTMLImageElement,
        previewImageSrc: media instanceof HTMLImageElement ? media.src : "",
        artifactWorkbenchVisible: Boolean(workbench),
        artifactWorkbenchText: workbench?.textContent || "",
        artifactEmptySurfaceVisible: Boolean(emptySurface),
        canvasWorkbenchPreviewVisible: Boolean(canvasPreview),
        attachmentCountAfterClick: attachmentImages.length,
        selectedArtifactText: document.body?.innerText || "",
      };
    });
    if (result.previewImageVisible) {
      break;
    }
    await sleep(options.intervalMs);
  }

  assert(
    result?.previewImageVisible,
    `导入图片附件未进入媒体 preview artifact: ${JSON.stringify(
      sanitizeJson(result),
    )}`,
  );

  return {
    ...before,
    ...result,
  };
}

async function waitForTimelineToolFileButton(page, options, fileName) {
  const toolRowSelector = '[data-testid="tool-call-row"]';
  const artifactCardSelector =
    '[data-testid="timeline-file-attachment-card"], [data-testid="timeline-file-artifact-card"]';
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 20_000)) {
    lastSnapshot = await page.evaluate(
      ({ artifactCardSelector, fileName, toolRowSelector }) => {
        const toolRows = Array.from(
          document.querySelectorAll(toolRowSelector),
        ).map((row, index) => ({
          index,
          text: row.textContent || "",
          buttons: Array.from(row.querySelectorAll("button")).map(
            (button, buttonIndex) => ({
              index: buttonIndex,
              title: button.getAttribute("title") || "",
              ariaLabel: button.getAttribute("aria-label") || "",
            }),
          ),
        }));
        const matchingToolRow = toolRows.find((row) =>
          row.text.includes(fileName),
        );
        const openButton = matchingToolRow?.buttons.find((button) =>
          /(?:画布|canvas)/i.test(`${button.title}\n${button.ariaLabel}`),
        );
        const artifactCards = Array.from(
          document.querySelectorAll(artifactCardSelector),
        ).filter((card) => (card.textContent || "").includes(fileName));
        return {
          matchingToolRow,
          openButton,
          artifactCardCount: artifactCards.length,
          toolRows,
          bodyText: document.body?.innerText || "",
        };
      },
      { artifactCardSelector, fileName, toolRowSelector },
    );
    if (lastSnapshot.matchingToolRow && lastSnapshot.openButton) {
      return {
        kind: "timeline-tool",
        selector: toolRowSelector,
        rowIndex: lastSnapshot.matchingToolRow.index,
        buttonIndex: lastSnapshot.openButton.index,
      };
    }
    if (lastSnapshot.artifactCardCount > 0) {
      return { kind: "artifact-card", selector: artifactCardSelector };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `导入工具轨缺少文件打开按钮 ${fileName}: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForWorkbenchFilePreview(page, options, expected) {
  const startedAt = Date.now();
  let result = null;
  const selectionHistory = [];
  let lastSelectionSignature = "";
  while (Date.now() - startedAt < Math.min(options.timeoutMs, 20_000)) {
    result = await page.evaluate((expectedFileName) => {
      const artifactWorkbench = document.querySelector(
        '[data-testid="artifact-workbench-shell"]',
      );
      const canvasWorkbench = document.querySelector(
        '[data-testid="canvas-workbench-shell"]',
      );
      const canvasWorkbenchLayout = document.querySelector(
        '[data-testid="canvas-workbench-layout"]',
      );
      const previewPanel = document.querySelector(
        '[data-testid="canvas-workbench-preview-mode-panel"]',
      );
      const markdownPreview = document.querySelector(
        '[data-testid="canvas-workbench-markdown-preview"]',
      );
      const htmlPreview = document.querySelector(
        '[data-testid="canvas-workbench-html-preview"]',
      );
      const codePreview = document.querySelector(
        '[data-testid="canvas-workbench-code-preview"]',
      );
      const emptySurface = document.querySelector(
        '[data-testid="artifact-empty-surface"]',
      );
      const fallbackSurface = document.querySelector(
        '[data-testid="preview-artifact-fallback-surface"]',
      );
      const bodyText = document.body?.innerText || "";
      const previewPanelText = previewPanel?.textContent || "";
      const workbenchText =
        artifactWorkbench?.textContent ||
        canvasWorkbench?.textContent ||
        canvasWorkbenchLayout?.textContent ||
        previewPanelText ||
        "";
      const canvasWorkbenchDataset =
        canvasWorkbench instanceof HTMLElement
          ? { ...canvasWorkbench.dataset }
          : {};
      const timelineToolRows = Array.from(
        document.querySelectorAll('[data-testid="tool-call-row"]'),
      ).map((row) => ({
        text: row.textContent || "",
        buttons: Array.from(row.querySelectorAll("button")).map((button) => ({
          title: button.getAttribute("title") || "",
          ariaLabel: button.getAttribute("aria-label") || "",
        })),
      }));
      const matchingTimelineToolRows = timelineToolRows.filter((row) =>
        row.text.includes(expectedFileName),
      );
      const traceRaw =
        window.localStorage.getItem("lime_invoke_trace_buffer_v1") || "";
      const errorRaw =
        window.localStorage.getItem("lime_invoke_error_buffer_v1") || "";
      const readJsonArray = (raw) => {
        try {
          const value = JSON.parse(raw || "[]");
          return Array.isArray(value) ? value : [];
        } catch {
          return [];
        }
      };
      const filePreviewTraceEntries = readJsonArray(traceRaw)
        .filter(
          (entry) =>
            JSON.stringify(entry).includes("fs/readFile") &&
            JSON.stringify(entry).includes(expectedFileName),
        )
        .slice(-4);
      const filePreviewInvokeErrors = readJsonArray(errorRaw)
        .filter(
          (entry) =>
            JSON.stringify(entry).includes("fs/readFile") &&
            JSON.stringify(entry).includes(expectedFileName),
        )
        .slice(-4);
      return {
        bodyText,
        workbenchVisible: Boolean(
          artifactWorkbench ||
          canvasWorkbench ||
          canvasWorkbenchLayout ||
          previewPanel,
        ),
        workbenchText,
        previewPanelText,
        canvasWorkbenchDataset,
        artifactWorkbenchVisible: Boolean(artifactWorkbench),
        canvasWorkbenchVisible: Boolean(canvasWorkbench),
        canvasWorkbenchLayoutVisible: Boolean(canvasWorkbenchLayout),
        previewPanelVisible: Boolean(previewPanel),
        markdownPreviewVisible: Boolean(markdownPreview),
        htmlPreviewVisible: Boolean(htmlPreview),
        codePreviewVisible: Boolean(codePreview),
        emptySurfaceVisible: Boolean(emptySurface),
        fallbackSurfaceVisible: Boolean(fallbackSurface),
        fallbackRenderMode:
          fallbackSurface?.getAttribute("data-preview-render-mode") || "",
        fallbackSurfaceText: fallbackSurface?.textContent || "",
        htmlPreviewSrc:
          htmlPreview instanceof HTMLIFrameElement ? htmlPreview.src : "",
        htmlPreviewSrcDoc:
          htmlPreview instanceof HTMLIFrameElement ? htmlPreview.srcdoc : "",
        timelineToolRows,
        matchingTimelineToolRows,
        traceHasFsReadFile: traceRaw.includes("fs/readFile"),
        traceHasExpectedFile: traceRaw.includes(expectedFileName),
        filePreviewTraceEntries,
        filePreviewInvokeErrors,
      };
    }, expected.fileName);
    const selectionState = {
      canvasWorkbenchDataset: result.canvasWorkbenchDataset,
      previewPanelText: result.previewPanelText,
      markdownPreviewVisible: result.markdownPreviewVisible,
      htmlPreviewVisible: result.htmlPreviewVisible,
      fallbackSurfaceVisible: result.fallbackSurfaceVisible,
    };
    const selectionSignature = JSON.stringify(selectionState);
    if (selectionSignature !== lastSelectionSignature) {
      selectionHistory.push({
        elapsedMs: Date.now() - startedAt,
        ...selectionState,
      });
      lastSelectionSignature = selectionSignature;
    }
    result.selectionHistory = selectionHistory;
    if (
      expected.kind !== "html" &&
      expected.kind !== "system_open" &&
      result.workbenchVisible &&
      result.previewPanelVisible &&
      (result.previewPanelText || result.workbenchText).includes(
        expected.fileName,
      ) &&
      (result.previewPanelText || result.workbenchText).includes(expected.text)
    ) {
      return result;
    }
    if (
      expected.kind === "system_open" &&
      result.workbenchVisible &&
      result.previewPanelVisible &&
      result.fallbackSurfaceVisible &&
      result.fallbackRenderMode === "system_open" &&
      result.fallbackSurfaceText.includes(expected.fileName) &&
      result.fallbackSurfaceText.includes(expected.text)
    ) {
      return result;
    }
    if (
      expected.kind === "html" &&
      result.workbenchVisible &&
      result.previewPanelVisible &&
      result.htmlPreviewVisible &&
      (result.previewPanelText || result.workbenchText).includes(
        expected.fileName,
      ) &&
      (result.bodyText.includes(expected.text) ||
        result.htmlPreviewSrc ||
        result.htmlPreviewSrcDoc.includes(expected.text))
    ) {
      return result;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Workbench 未显示导入文件预览 ${expected.fileName}: ${JSON.stringify(
      sanitizeJson(result),
    )}`,
  );
}

async function clickImportedToolFilePreview(page, options, expected) {
  const selector = await waitForTimelineToolFileButton(
    page,
    options,
    expected.fileName,
  );
  if (selector.kind === "timeline-tool") {
    await page
      .locator(selector.selector)
      .nth(selector.rowIndex)
      .locator("button")
      .nth(selector.buttonIndex)
      .click();
  } else {
    await page
      .locator(selector.selector)
      .filter({ hasText: expected.fileName })
      .first()
      .locator("button")
      .first()
      .click();
  }
  const result = await waitForWorkbenchFilePreview(page, options, expected);
  assert(
    result.workbenchVisible && result.previewPanelVisible,
    `${expected.fileName} 未进入 Artifact Workbench`,
  );
  assert(
    !result.emptySurfaceVisible,
    `${expected.fileName} 打开后仍显示空预览`,
  );
  const previewText = result.previewPanelText || result.workbenchText;
  if (expected.kind === "html") {
    assert(
      result.htmlPreviewVisible &&
        Boolean(result.htmlPreviewSrc || result.htmlPreviewSrcDoc) &&
        previewText.includes(expected.fileName),
      `${expected.fileName} 未进入 HTML iframe 预览`,
    );
  } else if (expected.kind === "system_open") {
    assert(
      result.fallbackSurfaceVisible &&
        result.fallbackRenderMode === "system_open" &&
        previewText.includes(expected.fileName) &&
        previewText.includes(expected.text),
      `${expected.fileName} 未进入 system_open preview fallback`,
    );
  } else {
    assert(
      previewText.includes(expected.fileName) &&
        previewText.includes(expected.text),
      `${expected.fileName} 未显示预期内容`,
    );
  }
  if (expected.kind === "docx") {
    const forbiddenTexts = [
      "PK\u0003\u0004",
      "word/document.xml",
      "[Content_Types].xml",
    ];
    assert(
      !forbiddenTexts.some((text) => result.bodyText.includes(text)),
      `${expected.fileName} 预览暴露了 DOCX ZIP/OpenXML 噪音`,
    );
  }
  if (expected.kind === "xlsx" || expected.kind === "pptx") {
    const forbiddenTexts = [
      "PK\u0003\u0004",
      "xl/worksheets/sheet1.xml",
      "xl/sharedStrings.xml",
      "ppt/slides/slide1.xml",
      "[Content_Types].xml",
    ];
    assert(
      !forbiddenTexts.some((text) => result.bodyText.includes(text)),
      `${expected.fileName} 预览暴露了 Office ZIP/OpenXML 噪音`,
    );
  }
  return {
    fileName: expected.fileName,
    kind: expected.kind,
    selector,
    ...result,
  };
}

export async function inspectImportedFilePreviewArtifacts(page, options) {
  await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.textareaVisible &&
      snapshot.bodyText.includes(IMPORTED_USER_TEXT) &&
      snapshot.bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT),
    "导入会话页未稳定，无法检查导入文件预览",
  );

  const markdown = await clickImportedToolFilePreview(page, options, {
    fileName: IMPORTED_PREVIEW_MARKDOWN_FILE,
    kind: "markdown",
    text: IMPORTED_PREVIEW_MARKDOWN_TEXT,
  });
  const html = await clickImportedToolFilePreview(page, options, {
    fileName: IMPORTED_PREVIEW_HTML_FILE,
    kind: "html",
    text: IMPORTED_PREVIEW_HTML_TEXT,
  });
  const docx = await clickImportedToolFilePreview(page, options, {
    fileName: IMPORTED_PREVIEW_DOCX_FILE,
    kind: "docx",
    text: IMPORTED_PREVIEW_DOCX_TEXT,
  });
  const xlsx = await clickImportedToolFilePreview(page, options, {
    fileName: IMPORTED_PREVIEW_XLSX_FILE,
    kind: "xlsx",
    text: IMPORTED_PREVIEW_XLSX_TEXT,
  });
  const pptx = await clickImportedToolFilePreview(page, options, {
    fileName: IMPORTED_PREVIEW_PPTX_FILE,
    kind: "pptx",
    text: IMPORTED_PREVIEW_PPTX_TEXT,
  });
  const pdf = await clickImportedToolFilePreview(page, options, {
    fileName: IMPORTED_PREVIEW_PDF_FILE,
    kind: "pdf",
    text: IMPORTED_PREVIEW_PDF_TEXT,
  });

  return {
    markdown,
    html,
    docx,
    xlsx,
    pptx,
    pdf,
    openedAllImportedPreviewArtifacts: true,
  };
}

async function inspectImportedSessionVisualViewport(
  page,
  options,
  viewport,
  screenshotPath,
) {
  await page.setViewportSize({
    width: viewport.width,
    height: viewport.height,
  });

  const isStableImportedVisualState = (current) => {
    const bodyText = current.bodyText || "";
    return (
      current.textareaVisible &&
      current.textareaDisabled === false &&
      bodyText.includes(IMPORTED_USER_TEXT) &&
      bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT) &&
      bodyText.includes(CONTINUE_USER_TEXT) &&
      bodyText.includes(CONTINUE_ASSISTANT_TEXT) &&
      hasPatchEvidenceText(bodyText) &&
      bodyText.includes(IMPORTED_PREVIEW_MARKDOWN_FILE) &&
      !bodyText.includes("imported_read_only") &&
      !bodyText.includes("thread-codex") &&
      !bodyText.includes("Approve Codex command")
    );
  };

  const snapshot = await waitForUiSnapshot(
    page,
    { ...options, timeoutMs: Math.min(options.timeoutMs, 15_000) },
    isStableImportedVisualState,
    `导入会话 ${viewport.label} 视口未稳定`,
  );
  const audit = await page.evaluate(
    ({
      importedUserText,
      importedAssistantText,
      importedMarkdownHeadingText,
      importedWebSearchQuery,
      importedWebSearchSourceLabel,
      importedWebSearchTitle,
      importedWebSearchUrl,
      importedReasoningText,
      continueUserText,
      continueAssistantText,
    }) => {
      const bodyText = document.body?.innerText || "";
      const textarea = document.querySelector(
        'textarea[name="agent-chat-message"]',
      );
      const importedBanner = document.querySelector(
        '[data-testid="imported-source-banner"]',
      );
      const importedRunControl = document.querySelector(
        '[data-testid="task-center-run-control-imported"]',
      );
      const messageList =
        document.querySelector('[data-testid="message-list"]') ||
        document.querySelector('[data-testid="message-list-stub"]') ||
        document.querySelector('[data-testid="agent-message-list"]') ||
        document.querySelector(
          '[data-testid="message-list-scroll-container"]',
        ) ||
        document.querySelector('[data-testid="message-list-column"]') ||
        document.querySelector('[data-testid="message-list-frame"]');
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const textareaRect =
        textarea instanceof HTMLElement
          ? textarea.getBoundingClientRect()
          : null;
      const messageRect =
        messageList instanceof HTMLElement
          ? messageList.getBoundingClientRect()
          : null;
      const compactModeControl = document.querySelector(
        '[data-testid="layout-compact-mode-bar"] [role="tablist"]',
      );
      const compactModeControlRect =
        compactModeControl instanceof HTMLElement
          ? compactModeControl.getBoundingClientRect()
          : null;
      const clippedVisibleRect = (element) => {
        const source = element.getBoundingClientRect();
        let left = Math.max(0, source.left);
        let top = Math.max(0, source.top);
        let right = Math.min(window.innerWidth, source.right);
        let bottom = Math.min(window.innerHeight, source.bottom);
        for (
          let ancestor = element.parentElement;
          ancestor;
          ancestor = ancestor.parentElement
        ) {
          const style = window.getComputedStyle(ancestor);
          const ancestorRect = ancestor.getBoundingClientRect();
          if (["auto", "clip", "hidden", "scroll"].includes(style.overflowX)) {
            left = Math.max(left, ancestorRect.left);
            right = Math.min(right, ancestorRect.right);
          }
          if (["auto", "clip", "hidden", "scroll"].includes(style.overflowY)) {
            top = Math.max(top, ancestorRect.top);
            bottom = Math.min(bottom, ancestorRect.bottom);
          }
        }
        return right > left && bottom > top
          ? { left, top, right, bottom }
          : null;
      };
      const compactModeOverlapButtons = compactModeControlRect
        ? Array.from(document.querySelectorAll("button"))
            .filter((button) => {
              if (
                compactModeControl.contains(button) ||
                !button.isConnected ||
                button.hidden
              ) {
                return false;
              }
              const rect = clippedVisibleRect(button);
              return (
                rect &&
                rect.left < compactModeControlRect.right &&
                rect.right > compactModeControlRect.left &&
                rect.top < compactModeControlRect.bottom &&
                rect.bottom > compactModeControlRect.top
              );
            })
            .slice(0, 5)
            .map((button) => ({
              text: button.textContent || "",
              testId: button.getAttribute("data-testid") || null,
              rect: (() => {
                const value = button.getBoundingClientRect();
                return {
                  left: value.left,
                  top: value.top,
                  width: value.width,
                  height: value.height,
                };
              })(),
            }))
        : [];
      const compactModeControlOverlapsOtherButton = Boolean(
        compactModeControlRect &&
        compactModeControlRect.width > 0 &&
        compactModeControlRect.height > 0 &&
        compactModeOverlapButtons.length > 0,
      );
      const forbiddenMainTexts = [
        "imported_read_only",
        "thread-codex",
        "Approve Codex command",
      ];
      const requiredTexts = [
        importedUserText,
        importedAssistantText,
        continueUserText,
        continueAssistantText,
      ];
      const rawMarkdownTexts = [
        `${importedMarkdownHeadingText}###`,
        "####如果历史会话",
        "**推荐 做法 **",
        "| 场景 | 过程 |",
      ];
      const searchNoiseTexts = [
        "Help",
        "Sign In",
        "Yahoo Scout",
        "https://help.yahoo.com/kb/search-for-desktop",
        "https://login.yahoo.com/",
      ];
      const headingVisible = Array.from(
        document.querySelectorAll("h1,h2,h3,h4,h5,h6"),
      ).some((node) =>
        (node.textContent || "").includes(importedMarkdownHeadingText),
      );
      const strongTexts = Array.from(document.querySelectorAll("strong")).map(
        (node) => node.textContent || "",
      );
      const timelineProcessBlocks = Array.from(
        document.querySelectorAll(
          'details[data-testid^="agent-thread-block:"][data-testid$=":process"]',
        ),
      );
      const operationalDetailRows = Array.from(
        document.querySelectorAll('[data-testid="tool-call-row"]'),
      );
      const searchToolContainer = operationalDetailRows
        .map((row) => row.parentElement)
        .find((container) =>
          (container?.textContent || "").includes(importedWebSearchQuery),
        );
      const searchRegionText = searchToolContainer?.textContent || "";
      const importedContextTexts = [
        importedWebSearchTitle,
        importedWebSearchSourceLabel,
      ].filter(Boolean);
      const bodyTextWithoutImportedContext = importedContextTexts.reduce(
        (text, allowedText) => text.split(allowedText).join(""),
        bodyText,
      );
      return {
        viewportWidth,
        viewportHeight,
        bodyTextLength: bodyText.length,
        hasImportedUserMessage: bodyText.includes(importedUserText),
        hasImportedAssistantMessage: bodyText.includes(importedAssistantText),
        markdownHeadingVisible: headingVisible,
        markdownStrongVisible:
          strongTexts.some((text) => text.includes("推荐 做法")) &&
          strongTexts.some((text) => text.includes("理由")),
        markdownTableVisible: Boolean(
          document.querySelector('[data-testid="markdown-table-scroll"] table'),
        ),
        rawMarkdownVisible: rawMarkdownTexts.some((text) =>
          bodyText.includes(text),
        ),
        hasImportedSearchResult:
          searchRegionText.includes(importedWebSearchTitle) ||
          searchRegionText.includes(importedWebSearchSourceLabel),
        searchTimelineVisible:
          timelineProcessBlocks.length > 0 && Boolean(searchToolContainer),
        searchNoiseVisible:
          Boolean(searchRegionText) &&
          searchNoiseTexts.some((text) => searchRegionText.includes(text)),
        hasFullSearchUrlVisible:
          Boolean(searchRegionText) &&
          searchRegionText.includes(importedWebSearchUrl),
        operationalDetailRowCount: operationalDetailRows.length,
        historicalOperationalDetailsHidden:
          operationalDetailRows.length === 0 && !searchToolContainer,
        hasContinueUserMessage: bodyText.includes(continueUserText),
        hasContinueAssistantMessage: bodyText.includes(continueAssistantText),
        historicalTimelinePreviewCount: document.querySelectorAll(
          '[data-testid^="message-list-historical-timeline-preview:"]',
        ).length,
        deferredHistoricalPreviewCount: document.querySelectorAll(
          [
            '[data-testid="message-list-historical-markdown-preview"]',
            '[data-testid="message-list-historical-assistant-preview"]',
            '[data-testid="message-list-long-history-preview"]',
          ].join(","),
        ).length,
        hasReasoningVisible: bodyText.includes(importedReasoningText),
        hasReasoningStatusVisible: bodyText.includes("已完成思考"),
        hasCommandExecutionVisible: bodyText.includes("npm test"),
        hasCommandOutput: bodyText.includes("ok"),
        hasPatchText:
          (bodyText.includes("补丁") ||
            bodyText.includes("Patch") ||
            bodyText.includes("已编辑") ||
            (bodyText.includes("lib.rs") && bodyText.includes("打开文件"))) &&
          (bodyText.includes("src/lib.rs") ||
            bodyText.includes("lib.rs") ||
            bodyText.includes("文件")),
        hasSearchEvidence: searchRegionText.includes(importedWebSearchQuery),
        hasApprovalText:
          bodyText.includes("导入的权限记录") ||
          bodyText.includes("已导入，只读记录") ||
          bodyText.includes("权限记录") ||
          bodyText.includes("审批") ||
          bodyText.includes("Approval"),
        importedBannerVisible: Boolean(importedBanner),
        importedRunControlVisible: Boolean(importedRunControl),
        inputbarVisible:
          textarea instanceof HTMLTextAreaElement &&
          textareaRect !== null &&
          textareaRect.width > 120 &&
          textareaRect.height >= 24 &&
          textareaRect.bottom > 0 &&
          textareaRect.top < viewportHeight,
        inputbarDisabled:
          textarea instanceof HTMLTextAreaElement ? textarea.disabled : true,
        inputbarOccludesMainContent:
          Boolean(textareaRect && messageRect) &&
          textareaRect.top < messageRect.top + 80,
        messageListVisible:
          messageRect !== null &&
          messageRect.width > 240 &&
          messageRect.height > 120 &&
          messageRect.bottom > 0 &&
          messageRect.top < viewportHeight,
        compactModeControlVisible: Boolean(
          compactModeControlRect &&
          compactModeControlRect.width > 0 &&
          compactModeControlRect.height > 0,
        ),
        compactModeControlOverlapsOtherButton,
        compactModeOverlapButtons,
        hidesRawImportedCommand: !forbiddenMainTexts.some((text) =>
          bodyText.includes(text),
        ),
        hidesSourceBrandText: !/\bcodex\b/i.test(
          bodyTextWithoutImportedContext,
        ),
        missingRequiredTexts: requiredTexts.filter(
          (text) => !bodyText.includes(text),
        ),
      };
    },
    {
      importedUserText: IMPORTED_USER_TEXT,
      importedAssistantText: IMPORTED_ASSISTANT_SUMMARY_TEXT,
      importedMarkdownHeadingText: IMPORTED_MARKDOWN_HEADING_TEXT,
      importedWebSearchSourceLabel: IMPORTED_WEB_SEARCH_SOURCE_LABEL,
      importedWebSearchTitle: IMPORTED_WEB_SEARCH_TITLE,
      importedWebSearchUrl: IMPORTED_WEB_SEARCH_URL,
      importedWebSearchQuery: IMPORTED_WEB_SEARCH_QUERY,
      importedReasoningText: IMPORTED_REASONING_TEXT,
      continueUserText: CONTINUE_USER_TEXT,
      continueAssistantText: CONTINUE_ASSISTANT_TEXT,
    },
  );
  await page.screenshot({ path: screenshotPath, fullPage: true });
  return {
    label: viewport.label,
    width: viewport.width,
    height: viewport.height,
    screenshot: screenshotPath,
    snapshotUrl: snapshot.url,
    ...audit,
  };
}

export async function collectImportedSessionVisualAudit(
  page,
  options,
  screenshotDir,
) {
  const audits = [];
  for (const viewport of VISUAL_AUDIT_VIEWPORTS) {
    const screenshotPath = path.join(
      screenshotDir,
      `${options.prefix}-${viewport.label}.png`,
    );
    const audit = await inspectImportedSessionVisualViewport(
      page,
      options,
      viewport,
      screenshotPath,
    );
    assert(
      audit.hasImportedUserMessage,
      `${viewport.label} 视口缺少导入用户消息`,
    );
    assert(
      audit.hasImportedAssistantMessage,
      `${viewport.label} 视口缺少导入助手消息`,
    );
    assert(
      audit.markdownHeadingVisible,
      `${viewport.label} 视口未把导入 Markdown 标题渲染为 heading`,
    );
    assert(
      audit.markdownStrongVisible,
      `${viewport.label} 视口未把导入 Markdown 加粗渲染为 strong`,
    );
    assert(
      audit.markdownTableVisible,
      `${viewport.label} 视口未把导入 Markdown 表格渲染为 table`,
    );
    assert(
      !audit.rawMarkdownVisible,
      `${viewport.label} 视口暴露了导入原始 Markdown 语法`,
    );
    assert(
      audit.historicalOperationalDetailsHidden,
      `${viewport.label} 视口铺开了历史运行期工具明细`,
    );
    assert(
      !audit.searchNoiseVisible,
      `${viewport.label} 视口暴露了搜索导航噪音`,
    );
    assert(
      !audit.hasFullSearchUrlVisible,
      `${viewport.label} 视口暴露了完整搜索 URL`,
    );
    assert(
      audit.hasContinueUserMessage,
      `${viewport.label} 视口缺少续聊用户消息`,
    );
    assert(
      audit.hasContinueAssistantMessage,
      `${viewport.label} 视口缺少续聊助手消息`,
    );
    assert(audit.hasPatchText, `${viewport.label} 视口缺少导入补丁记录`);
    assert(
      !audit.importedBannerVisible,
      `${viewport.label} 视口不应展示导入主线 banner`,
    );
    assert(
      !audit.importedRunControlVisible,
      `${viewport.label} 视口不应展示导入运行控制卡`,
    );
    assert(audit.inputbarVisible, `${viewport.label} 视口输入框不可见`);
    assert(!audit.inputbarDisabled, `${viewport.label} 视口输入框不可用`);
    assert(
      !audit.inputbarOccludesMainContent,
      `${viewport.label} 视口输入框遮挡消息主内容`,
    );
    assert(audit.messageListVisible, `${viewport.label} 视口消息列表不可见`);
    assert(
      !audit.compactModeControlOverlapsOtherButton,
      `${viewport.label} 视口聊天/工作台模式控件与其他按钮重叠: ${JSON.stringify(audit.compactModeOverlapButtons)}`,
    );
    assert(
      audit.hidesRawImportedCommand,
      `${viewport.label} 视口暴露了原始导入命令或内部字段`,
    );
    assert(
      audit.hidesSourceBrandText,
      `${viewport.label} 视口暴露了非导入语境来源品牌字眼`,
    );
    audits.push(audit);
  }
  return audits;
}

export async function inspectImportedMarkdownAndSearchRendering(page, options) {
  const snapshot = await waitForUiSnapshot(
    page,
    options,
    (current) =>
      current.textareaVisible &&
      current.bodyText.includes(IMPORTED_USER_TEXT) &&
      current.bodyText.includes(IMPORTED_ASSISTANT_SUMMARY_TEXT),
    "导入会话页未稳定，无法检查 Markdown 与搜索渲染",
  );
  const rendering = await page.evaluate(
    ({
      importedMarkdownHeadingText,
      importedWebSearchTitle,
      importedWebSearchQuery,
      importedWebSearchSourceLabel,
      importedWebSearchUrl,
    }) => {
      const bodyText = document.body?.innerText || "";
      const headings = Array.from(
        document.querySelectorAll("h1,h2,h3,h4,h5,h6"),
      ).map((node) => node.textContent || "");
      const strongTexts = Array.from(document.querySelectorAll("strong")).map(
        (node) => node.textContent || "",
      );
      const timelineProcessBlocks = Array.from(
        document.querySelectorAll(
          'details[data-testid^="agent-thread-block:"][data-testid$=":process"]',
        ),
      );
      const timelineProcessText = timelineProcessBlocks
        .map((node) => node.textContent || "")
        .join("\n");
      const toolRows = Array.from(
        document.querySelectorAll('[data-testid="tool-call-row"]'),
      );
      const searchToolContainer = toolRows
        .map((row) => row.parentElement)
        .find((container) =>
          (container?.textContent || "").includes(importedWebSearchQuery),
        );
      const searchTimelineText = searchToolContainer?.textContent || "";
      const toolRowText = toolRows
        .map((node) => node.textContent || "")
        .join("\n");
      const rawMarkdownTexts = [
        `${importedMarkdownHeadingText}###`,
        "####如果历史会话",
        "**推荐 做法 **",
        "| 场景 | 过程 |",
      ];
      const searchNoiseTexts = [
        "Help",
        "Sign In",
        "Yahoo Scout",
        "https://help.yahoo.com/kb/search-for-desktop",
        "https://login.yahoo.com/",
      ];
      return {
        snapshotTextLength: bodyText.length,
        headings,
        strongTexts,
        timelineProcessText,
        toolRowText,
        markdownHeadingVisible: headings.some((text) =>
          text.includes(importedMarkdownHeadingText),
        ),
        markdownStrongVisible:
          strongTexts.some((text) => text.includes("推荐 做法")) &&
          strongTexts.some((text) => text.includes("理由")),
        markdownTableVisible: Boolean(
          document.querySelector('[data-testid="markdown-table-scroll"] table'),
        ),
        rawMarkdownVisible: rawMarkdownTexts.some((text) =>
          bodyText.includes(text),
        ),
        hasImportedSearchResult:
          searchTimelineText.includes(importedWebSearchTitle) ||
          searchTimelineText.includes(importedWebSearchSourceLabel),
        searchTimelineVisible:
          timelineProcessBlocks.length > 0 && Boolean(searchToolContainer),
        searchQueryVisible: searchTimelineText.includes(importedWebSearchQuery),
        searchNoiseVisible: searchNoiseTexts.some((text) =>
          searchTimelineText.includes(text),
        ),
        hasFullSearchUrlVisible:
          searchTimelineText.includes(importedWebSearchUrl),
        operationalDetailRowCount: toolRows.length,
        historicalOperationalDetailsHidden:
          toolRows.length === 0 && !searchToolContainer,
      };
    },
    {
      importedMarkdownHeadingText: IMPORTED_MARKDOWN_HEADING_TEXT,
      importedWebSearchTitle: IMPORTED_WEB_SEARCH_TITLE,
      importedWebSearchQuery: IMPORTED_WEB_SEARCH_QUERY,
      importedWebSearchSourceLabel: IMPORTED_WEB_SEARCH_SOURCE_LABEL,
      importedWebSearchUrl: IMPORTED_WEB_SEARCH_URL,
    },
  );

  assert(
    rendering.markdownHeadingVisible,
    `导入 Markdown 标题未渲染为 heading: ${JSON.stringify(
      sanitizeJson(rendering),
    )}`,
  );
  assert(
    rendering.markdownStrongVisible,
    `导入 Markdown 加粗未渲染为 strong: ${JSON.stringify(
      sanitizeJson(rendering),
    )}`,
  );
  assert(
    rendering.markdownTableVisible,
    `导入 Markdown 表格未渲染为 table: ${JSON.stringify(
      sanitizeJson(rendering),
    )}`,
  );
  assert(
    !rendering.rawMarkdownVisible,
    `导入正文暴露了原始 Markdown 语法: ${JSON.stringify(
      sanitizeJson(rendering),
    )}`,
  );
  assert(
    rendering.historicalOperationalDetailsHidden,
    `导入历史铺开了运行期工具明细: ${JSON.stringify(sanitizeJson(rendering))}`,
  );
  assert(
    !rendering.searchNoiseVisible,
    `导入搜索结果暴露了搜索导航噪音: ${JSON.stringify(
      sanitizeJson(rendering),
    )}`,
  );
  assert(
    !rendering.hasFullSearchUrlVisible,
    `导入搜索结果暴露了完整 URL: ${JSON.stringify(sanitizeJson(rendering))}`,
  );

  return {
    ...rendering,
    snapshotUrl: snapshot.url,
  };
}

export async function waitForImportedSessionDetails(page, options) {
  return await waitForUiSnapshot(
    page,
    options,
    (snapshot) => {
      const summary = summarizeImportedDetailsSnapshot(snapshot);
      return (
        snapshot.textareaVisible &&
        summary.hasImportedUserMessage &&
        summary.hasImportedAssistantMessage &&
        summary.hasPatchText &&
        summary.historicalOperationalDetailsHidden &&
        summary.hidesRawImportedCommand
      );
    },
    "导入后的会话页未稳定为历史摘要形态",
  );
}

export async function sendFollowUpFromGui(page, options) {
  await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.textareaVisible && snapshot.textareaDisabled === false,
    "续聊输入框未就绪",
  );
  const textarea = page.locator('textarea[name="agent-chat-message"]').first();
  await textarea.fill(CONTINUE_USER_TEXT);
  await waitForUiSnapshot(
    page,
    options,
    (snapshot) => snapshot.textareaValue === CONTINUE_USER_TEXT,
    "续聊输入未进入 textarea",
  );
  const clicked = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll("button"));
    const sendButton = buttons.find((button) => {
      const label = [
        button.getAttribute("aria-label") || "",
        button.getAttribute("title") || "",
        button.textContent || "",
      ].join("\n");
      return (
        (label.includes("发送") || /\bSend\b/i.test(label)) && !button.disabled
      );
    });
    if (sendButton instanceof HTMLElement) {
      sendButton.click();
      return {
        clicked: true,
        label:
          sendButton.getAttribute("aria-label") ||
          sendButton.getAttribute("title") ||
          sendButton.textContent ||
          "send",
      };
    }
    return {
      clicked: false,
      labels: buttons.map((button) =>
        [
          button.getAttribute("aria-label") || "",
          button.getAttribute("title") || "",
          button.textContent || "",
        ].join(" / "),
      ),
    };
  });
  assert(
    clicked?.clicked,
    `未找到可点击发送按钮: ${JSON.stringify(sanitizeJson(clicked))}`,
  );
  return clicked;
}

export async function waitForContinuationVisible(page, options) {
  return await waitForUiSnapshot(
    page,
    options,
    (snapshot) =>
      snapshot.textareaVisible &&
      snapshot.bodyText.includes(CONTINUE_USER_TEXT) &&
      snapshot.bodyText.includes(CONTINUE_ASSISTANT_TEXT),
    "续聊消息未在同一会话页完成",
  );
}
