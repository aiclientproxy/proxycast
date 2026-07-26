import { createHash } from "node:crypto";
import {
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
import {
  collectReadModelItems,
  collectReadModelTurns,
  readModelLatestTurnStatus,
  readModelTurnId,
} from "./claw-chat-current-fixture-read-model-core.mjs";
import { waitForSessionReadCompleted } from "./claw-chat-current-fixture-read-model-waits.mjs";
import { evaluatePageSnapshot } from "./claw-chat-current-fixture-rpc.mjs";
import {
  readString,
  readArray,
  sanitizeJson,
  sleep,
} from "./claw-chat-current-fixture-utils.mjs";

function readModelItemCount(readModel) {
  return {
    detailItemCount: Array.isArray(readModel?.detail?.items)
      ? readModel.detail.items.length
      : null,
    threadReadItemCount: Array.isArray(
      readModel?.detail?.thread_read?.thread_items,
    )
      ? readModel.detail.thread_read.thread_items.length
      : null,
  };
}

function countOccurrences(text, fragment) {
  if (!fragment) {
    return 0;
  }
  return text.split(fragment).length - 1;
}

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

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

function normalizeItemType(item) {
  return String(readString(item, "type", "kind", "item_type", "itemType") ?? "")
    .replaceAll("-", "_")
    .toLowerCase();
}

function summarizeLiveTailCommitReadModel(readModel, expected) {
  const serialized = JSON.stringify(readModel || {});
  const matchedItems = collectReadModelItems(readModel).filter(
    (item) => readString(item, "id", "item_id", "itemId") === expected.itemId,
  );
  const agentMessageItems = matchedItems.filter((item) =>
    [
      "agentmessage",
      "agent_message",
      "assistantmessage",
      "assistant_message",
    ].includes(normalizeItemType(item)),
  );
  const canonicalTexts = [
    ...new Set(
      agentMessageItems
        .map((item) => readString(item, "text"))
        .filter((text) => typeof text === "string"),
    ),
  ];
  const itemThreadIds = [
    ...new Set(
      agentMessageItems
        .map((item) => readString(item, "thread_id", "threadId"))
        .filter(Boolean),
    ),
  ];
  const itemTurnIds = [
    ...new Set(
      agentMessageItems
        .map((item) => readString(item, "turn_id", "turnId"))
        .filter(Boolean),
    ),
  ];
  const matchedTurnIds = [
    ...new Set(
      collectReadModelTurns(readModel)
        .filter((turn) =>
          readArray(turn, "items").some(
            (item) =>
              readString(item, "id", "item_id", "itemId") === expected.itemId,
          ),
        )
        .map((turn) => readModelTurnId(turn))
        .filter(Boolean),
    ),
  ];
  const readModelThreadId =
    readString(readModel?.thread, "id", "threadId", "thread_id") ??
    readString(readModel, "threadId", "thread_id") ??
    null;
  return {
    ...readModelItemCount(readModel),
    latestTurnStatus: readModelLatestTurnStatus(readModel),
    includesPrompt: serialized.includes(LIVE_TAIL_COMMIT_PROMPT),
    includesFirstText: serialized.includes(LIVE_TAIL_COMMIT_FIRST_TEXT),
    includesOverflowMarker: serialized.includes(
      LIVE_TAIL_COMMIT_OVERFLOW_MARKER,
    ),
    includesTableHeader: serialized.includes(LIVE_TAIL_COMMIT_TABLE_HEADER),
    includesTableTail: serialized.includes(LIVE_TAIL_COMMIT_TABLE_TAIL),
    includesAssistantDone: serialized.includes(LIVE_TAIL_COMMIT_DONE_TEXT),
    canonicalItemId: expected.itemId,
    canonicalItemRepresentationCount: agentMessageItems.length,
    canonicalTextVariantCount: canonicalTexts.length,
    canonicalTextLength: canonicalTexts[0]?.length ?? null,
    canonicalTextSha256: canonicalTexts[0] ? sha256(canonicalTexts[0]) : null,
    canonicalTextMatchesCompletedItem:
      canonicalTexts.length === 1 &&
      canonicalTexts[0].length === expected.completedTextLength &&
      sha256(canonicalTexts[0]) === expected.completedTextSha256,
    canonicalMarkerExactOnce:
      canonicalTexts.length === 1 &&
      [
        LIVE_TAIL_COMMIT_FIRST_TEXT,
        LIVE_TAIL_COMMIT_OVERFLOW_MARKER,
        LIVE_TAIL_COMMIT_TABLE_HEADER,
        LIVE_TAIL_COMMIT_TABLE_TAIL,
        LIVE_TAIL_COMMIT_DONE_TEXT,
      ].every(
        (fragment) => countOccurrences(canonicalTexts[0], fragment) === 1,
      ),
    itemThreadIds,
    itemTurnIds,
    matchedTurnIds,
    readModelThreadId,
    identityMatches:
      readModelThreadId === expected.threadId &&
      matchedTurnIds.length === 1 &&
      matchedTurnIds[0] === expected.turnId &&
      (itemThreadIds.length === 0 ||
        (itemThreadIds.length === 1 &&
          itemThreadIds[0] === expected.threadId)) &&
      (itemTurnIds.length === 0 ||
        (itemTurnIds.length === 1 && itemTurnIds[0] === expected.turnId)),
  };
}

async function evaluateLiveTailSnapshot(page, expectedItemId) {
  return await evaluatePageSnapshot(
    page,
    ({
      doneText,
      expectedItemId,
      firstText,
      overflowMarker,
      prompt,
      tableHeaderCells,
      tableTailCells,
    }) => {
      const text = document.body?.innerText || "";
      const mainText = document.querySelector("main")?.innerText || text;
      const messageListScope =
        document.querySelector('[data-testid="message-list-column"]') ||
        document.querySelector('[data-testid="message-list"]') ||
        document.querySelector('[data-testid="message-list-frame"]') ||
        document.querySelector("main") ||
        document.body;
      const turnGroups = Array.from(
        document.querySelectorAll('[data-testid="message-turn-group"]'),
      );
      const scopedTurnGroup =
        [...turnGroups]
          .reverse()
          .find((group) => (group.innerText || "").includes(prompt)) ?? null;
      const assistantBubbles = Array.from(
        (scopedTurnGroup || messageListScope).querySelectorAll(
          '[data-message-role="assistant"]',
        ),
      );
      const assistantScope =
        assistantBubbles[assistantBubbles.length - 1] ?? scopedTurnGroup;
      const scopedText = scopedTurnGroup?.innerText || mainText;
      const assistantText = assistantScope?.innerText || scopedText;
      const assistantTranscriptText =
        assistantBubbles.length > 0
          ? assistantBubbles.map((bubble) => bubble.innerText || "").join("\n")
          : assistantText;
      const assistantIdentities = assistantBubbles.map((bubble) => ({
        messageId: bubble.getAttribute("data-message-id"),
        runtimeTurnId: bubble.getAttribute("data-runtime-turn-id"),
        threadItemId: bubble.getAttribute("data-thread-item-id"),
        timelineItems: bubble.getAttribute("data-timeline-items") || "",
      }));
      const assistantMessageIds = assistantIdentities
        .map((identity) => identity.messageId)
        .filter(Boolean);
      const assistantThreadItemIds = assistantIdentities
        .map((identity) => identity.threadItemId)
        .filter(Boolean);
      const assistantRuntimeTurnIds = assistantIdentities
        .map((identity) => identity.runtimeTurnId)
        .filter(Boolean);
      const timelineAgentMessageIds = assistantIdentities.flatMap((identity) =>
        identity.timelineItems
          .split("|")
          .map((entry) => entry.trim())
          .filter((entry) => entry.startsWith("agent_message:"))
          .map((entry) => entry.slice("agent_message:".length).trim())
          .filter(Boolean),
      );
      const occurrenceCount = (fragment) =>
        fragment ? assistantTranscriptText.split(fragment).length - 1 : 0;
      const textarea = document.querySelector(
        'textarea[name="agent-chat-message"]',
      );
      const textareaRect = textarea?.getBoundingClientRect();
      const textareaStyle = textarea ? window.getComputedStyle(textarea) : null;
      const textareaVisible = Boolean(
        textarea &&
        textareaRect &&
        textareaRect.width > 16 &&
        textareaRect.height > 16 &&
        textareaStyle?.visibility !== "hidden" &&
        textareaStyle?.display !== "none",
      );
      const buttons = Array.from(document.querySelectorAll("button")).map(
        (button) => ({
          title: button.getAttribute("title") || "",
          text: button.textContent || "",
          aria: button.getAttribute("aria-label") || "",
          disabled: button.disabled,
        }),
      );
      const stopButtonVisible = buttons.some((button) => {
        const label = [button.title, button.text, button.aria].join("\n");
        return (
          !button.disabled &&
          (label.includes("停止") ||
            label.includes("终止") ||
            /\bStop\b/i.test(label))
        );
      });
      const runningStatusVisible =
        stopButtonVisible ||
        scopedText.includes("正在输出") ||
        scopedText.includes("正在生成") ||
        scopedText.includes("生成中") ||
        /\bStreaming\b/i.test(scopedText);
      const startupNoteVisible = [
        "启动处理流程",
        "启动说明",
        "已接收请求",
        "正在启动",
      ].some((fragment) => scopedText.includes(fragment));
      const readScrollMetrics = (node) => {
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
          nearBottom: !scrollHeight || distanceToBottom <= 180,
          rectHeight: Math.round(rect.height || 0),
        };
      };
      const scrollCandidates = [
        messageListScope,
        document.querySelector('[data-testid="message-list-frame"]'),
        document.querySelector("main"),
        document.scrollingElement,
      ]
        .map(readScrollMetrics)
        .filter(Boolean);
      const scrollRoot =
        scrollCandidates.find((candidate) => candidate.overflowed) ??
        scrollCandidates[0] ??
        null;
      const renderedTables = assistantScope
        ? Array.from(assistantScope.querySelectorAll("table"))
        : [];
      const renderedTableRows = renderedTables.flatMap((table) =>
        Array.from(table.querySelectorAll("tr")).map((row) =>
          Array.from(row.querySelectorAll("th, td")).map((cell) =>
            (cell.textContent || "").replace(/\s+/gu, " ").trim(),
          ),
        ),
      );
      const countMatchingTableRows = (expectedCells) =>
        renderedTableRows.filter(
          (cells) =>
            cells.length === expectedCells.length &&
            cells.every((cell, index) => cell === expectedCells[index]),
        ).length;
      const tableHeaderOccurrenceCount =
        countMatchingTableRows(tableHeaderCells);
      const tableTailOccurrenceCount = countMatchingTableRows(tableTailCells);
      const renderedTableCount = renderedTables.length;
      const firstTextIndex = assistantText.indexOf(firstText);
      const overflowMarkerIndex = assistantText.indexOf(overflowMarker);
      const tableTailIndex = assistantText.indexOf(tableTailCells[1] || "");
      const doneTextIndex = assistantText.indexOf(doneText);
      return {
        url: window.location.href,
        hasPrompt: scopedText.includes(prompt),
        hasFirstText: assistantText.includes(firstText),
        hasOverflowMarker: assistantText.includes(overflowMarker),
        hasTableHeader: tableHeaderOccurrenceCount > 0,
        hasTableTail: tableTailOccurrenceCount > 0,
        hasDoneText: assistantText.includes(doneText),
        expectedItemId,
        assistantMessageIds,
        assistantThreadItemIds,
        assistantRuntimeTurnIds,
        timelineAgentMessageIds,
        expectedItemIdentityVisible:
          typeof expectedItemId === "string" &&
          (assistantThreadItemIds.includes(expectedItemId) ||
            timelineAgentMessageIds.includes(expectedItemId)),
        firstTextOccurrenceCount: occurrenceCount(firstText),
        overflowMarkerOccurrenceCount: occurrenceCount(overflowMarker),
        tableHeaderOccurrenceCount,
        tableTailOccurrenceCount,
        doneTextOccurrenceCount: occurrenceCount(doneText),
        canonicalMarkersExactOnce:
          [firstText, overflowMarker, doneText].every(
            (fragment) => occurrenceCount(fragment) === 1,
          ) &&
          tableHeaderOccurrenceCount === 1 &&
          tableTailOccurrenceCount === 1,
        firstTextBeforeOverflow:
          firstTextIndex >= 0 &&
          overflowMarkerIndex >= 0 &&
          firstTextIndex < overflowMarkerIndex,
        firstTextBeforeTableTail:
          firstTextIndex >= 0 &&
          tableTailIndex >= 0 &&
          firstTextIndex < tableTailIndex,
        firstTextBeforeDone:
          firstTextIndex >= 0 &&
          doneTextIndex >= 0 &&
          firstTextIndex < doneTextIndex,
        renderedTableCount,
        markdownTableRendered: renderedTableCount > 0,
        textareaVisible,
        textareaDisabled:
          textarea instanceof HTMLTextAreaElement ? textarea.disabled : null,
        stopButtonVisible,
        runningStatusVisible,
        startupNoteVisible,
        scrollRoot,
        scrollCandidates,
        scrollAnchorStable:
          scrollRoot == null || scrollRoot.nearBottom === true,
        overflowCommitted:
          assistantText.includes(overflowMarker) &&
          (scrollRoot?.overflowed === true || assistantText.length > 1400),
        assistantTextPreview: assistantText.slice(0, 240),
        assistantTextLength: assistantText.length,
      };
    },
    {
      doneText: LIVE_TAIL_COMMIT_DONE_TEXT,
      expectedItemId,
      firstText: LIVE_TAIL_COMMIT_FIRST_TEXT,
      overflowMarker: LIVE_TAIL_COMMIT_OVERFLOW_MARKER,
      prompt: LIVE_TAIL_COMMIT_PROMPT,
      tableHeaderCells: LIVE_TAIL_COMMIT_TABLE_HEADER_CELLS,
      tableTailCells: LIVE_TAIL_COMMIT_TABLE_TAIL_CELLS,
    },
  );
}

async function waitForGuiLiveTailFirstVisibleBeforeCommit(
  page,
  options,
  expectedItemId,
) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluateLiveTailSnapshot(page, expectedItemId);
    lastSnapshot = snapshot;
    if (
      snapshot?.hasPrompt === true &&
      snapshot.hasFirstText === true &&
      snapshot.hasOverflowMarker === false &&
      snapshot.hasTableTail === false &&
      snapshot.hasDoneText === false &&
      snapshot.firstTextOccurrenceCount === 1 &&
      snapshot.runningStatusVisible === true &&
      snapshot.startupNoteVisible === false
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未捕获 live-tail 首字完成前可见: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function waitForGuiLiveTailVisualOracle(page, options, expectedItemId) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const snapshot = await evaluateLiveTailSnapshot(page, expectedItemId);
    lastSnapshot = snapshot;
    if (
      snapshot?.hasPrompt === true &&
      snapshot.hasFirstText === true &&
      snapshot.hasOverflowMarker === true &&
      snapshot.hasTableHeader === true &&
      snapshot.hasTableTail === true &&
      snapshot.hasDoneText === true &&
      snapshot.expectedItemIdentityVisible === true &&
      snapshot.canonicalMarkersExactOnce === true &&
      snapshot.markdownTableRendered === true &&
      snapshot.overflowCommitted === true &&
      snapshot.scrollAnchorStable === true &&
      snapshot.firstTextBeforeOverflow === true &&
      snapshot.firstTextBeforeTableTail === true &&
      snapshot.firstTextBeforeDone === true &&
      snapshot.textareaVisible === true &&
      snapshot.textareaDisabled === false &&
      snapshot.stopButtonVisible === false &&
      snapshot.startupNoteVisible === false
    ) {
      return snapshot;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Claw GUI 未完成 live-tail visual oracle: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

export async function runLiveTailCommitScenario({
  page,
  options,
  appServerRequests,
  runtimeEnv,
  logStage,
}) {
  const result = {};

  logStage("send-live-tail-commit-prompt-from-gui");
  result.liveTailCommitInputSend = sanitizeJson(
    await sendPromptFromGui(page, options, LIVE_TAIL_COMMIT_PROMPT),
  );

  logStage("wait-live-tail-commit-backend-turn-start");
  const backendTurnStart = await waitForBackendLedgerEntry(
    runtimeEnv.backendLedgerPath,
    (entry) =>
      entry.kind === "turnStart" && entry.inputText === LIVE_TAIL_COMMIT_PROMPT,
    options,
  );
  result.liveTailCommitBackendTurnStart = sanitizeJson({
    sessionId: backendTurnStart.entry.sessionId,
    turnId: backendTurnStart.entry.turnId,
    inputText: backendTurnStart.entry.inputText,
    ledgerCount: backendTurnStart.ledger.length,
  });
  const expectedItemId = `agent-message-final-${backendTurnStart.entry.turnId}`;

  logStage("wait-gui-live-tail-first-visible-before-commit");
  result.guiLiveTailFirstVisibleBeforeCommit = sanitizeJson(
    await waitForGuiLiveTailFirstVisibleBeforeCommit(
      page,
      options,
      expectedItemId,
    ),
  );

  logStage("wait-gui-live-tail-completed");
  result.guiLiveTailCompleted = sanitizeJson(
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

  logStage("wait-gui-live-tail-visual-oracle");
  result.guiLiveTailVisualOracle = sanitizeJson(
    await waitForGuiLiveTailVisualOracle(page, options, expectedItemId),
  );

  const liveTailLedger = await waitForBackendLedgerEntry(
    runtimeEnv.backendLedgerPath,
    (entry) => entry.kind === "liveTailCommitCompleted",
    options,
  );
  result.liveTailCommitBackendCompleted = sanitizeJson({
    threadId: liveTailLedger.entry.threadId,
    turnId: liveTailLedger.entry.turnId,
    itemId: liveTailLedger.entry.itemId,
    droppedEventType: liveTailLedger.entry.droppedEventType,
    repairEventType: liveTailLedger.entry.repairEventType,
    terminalEventType: liveTailLedger.entry.terminalEventType,
    emittedEventTypes: liveTailLedger.entry.emittedEventTypes,
    completedTextLength: liveTailLedger.entry.completedTextLength,
    completedTextSha256: liveTailLedger.entry.completedTextSha256,
    firstText: liveTailLedger.entry.firstText,
    overflowMarker: liveTailLedger.entry.overflowMarker,
    tableHeader: liveTailLedger.entry.tableHeader,
    tableTail: liveTailLedger.entry.tableTail,
    ledgerCount: liveTailLedger.ledger.length,
  });

  logStage("wait-read-model-live-tail-completed");
  const readModelLiveTailCommitCompleted = await waitForSessionReadCompleted(
    page,
    options,
    appServerRequests,
    {
      prompt: LIVE_TAIL_COMMIT_PROMPT,
      doneText: LIVE_TAIL_COMMIT_DONE_TEXT,
      summaryText: LIVE_TAIL_COMMIT_FIRST_TEXT,
    },
  );
  result.readModelLiveTailCommitCompleted = sanitizeJson(
    summarizeLiveTailCommitReadModel(readModelLiveTailCommitCompleted, {
      threadId: liveTailLedger.entry.threadId,
      turnId: liveTailLedger.entry.turnId,
      itemId: liveTailLedger.entry.itemId,
      completedTextLength: liveTailLedger.entry.completedTextLength,
      completedTextSha256: liveTailLedger.entry.completedTextSha256,
    }),
  );

  return result;
}
