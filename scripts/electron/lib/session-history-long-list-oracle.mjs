import {
  THREAD_READ_LONG_LIST,
  THREAD_READ_LONG_LIST_ITEM_COUNT,
  THREAD_READ_LONG_LIST_TURN_COUNT,
} from "./session-history-long-list-fixture.mjs";

const MAX_INITIAL_DOM_TURN_GROUPS = 12;
const MAX_INITIAL_DOM_TEXT_LENGTH = 40_000;
const MAX_INITIAL_THREAD_ITEMS = MAX_INITIAL_DOM_TURN_GROUPS * 3;
const MAX_CLICK_TO_FIRST_PAINT_MS = 8_000;
const MAX_MESSAGE_LIST_COMPUTE_MS = 500;
const REQUIRED_METHODS = [
  "thread/read",
  "thread/turns/list",
  "thread/items/list",
];
const FORBIDDEN_METHODS = [
  "agentSession/get",
  "agentSession/list",
  "agentSession/history",
];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

async function clickLongListConversation(page, options) {
  const startedAt = Date.now();
  let lastRows = [];
  while (Date.now() - startedAt < options.timeoutMs) {
    const result = await page.evaluate((fixture) => {
      const buttons = Array.from(
        document.querySelectorAll(
          '[data-testid="app-sidebar-conversation-open"]',
        ),
      );
      const rows = buttons.map((button) => ({
        title: button.getAttribute("title") || "",
        text: button.textContent || "",
      }));
      const target = buttons.find((button) => {
        const title = button.getAttribute("title") || "";
        const text = button.textContent || "";
        return title === fixture.title || text.includes(fixture.title);
      });
      if (!(target instanceof HTMLButtonElement))
        return { clicked: false, rows };
      target.click();
      return { clicked: true, rows };
    }, THREAD_READ_LONG_LIST);
    if (result?.clicked) return result;
    lastRows = result?.rows ?? [];
    await sleep(options.intervalMs);
  }
  throw new Error(
    `long-list sidebar conversation 不可见: ${JSON.stringify(lastRows)}`,
  );
}

async function readLongListSnapshot(page) {
  return await page.evaluate((fixture) => {
    const frame = document.querySelector('[data-testid="message-list-frame"]');
    const column = document.querySelector(
      '[data-testid="message-list-column"]',
    );
    const groups = Array.from(
      document.querySelectorAll('[data-testid="message-turn-group"]'),
    );
    const historyWindow = document.querySelector(
      '[data-testid="message-list-history-window"]',
    );
    const persistedWindow = document.querySelector(
      '[data-testid="message-list-persisted-history-window"]',
    );
    const bodyText = column?.textContent || "";
    const latestTurn = fixture.turns.at(-1);
    const oldestTurn = fixture.turns[0];
    const perf = window.__LIME_AGENTUI_PERF__?.summary?.() ?? null;
    const performanceSession =
      perf?.sessions?.find(
        (session) => session.sessionId === fixture.sessionId,
      ) ?? null;
    let traceEntries = [];
    try {
      const raw = window.localStorage.getItem("lime_invoke_trace_buffer_v1");
      const parsed = raw ? JSON.parse(raw) : [];
      traceEntries = Array.isArray(parsed) ? parsed : [];
    } catch {
      traceEntries = [];
    }
    const requestMethods = traceEntries.flatMap((entry) => {
      if (entry?.command !== "app_server_handle_json_lines") return [];
      const lines = entry?.args_preview?.request?.lines;
      if (!Array.isArray(lines)) return [];
      return lines.flatMap((line) => {
        try {
          const message = JSON.parse(String(line));
          return typeof message?.method === "string" ? [message.method] : [];
        } catch {
          return [];
        }
      });
    });
    return {
      frameSessionId: frame?.getAttribute("data-session-id") || null,
      turnGroupCount: groups.length,
      canonicalTurnGroupCount: groups.filter(
        (group) =>
          group.getAttribute("data-render-entry-kind") === "canonical_turn",
      ).length,
      residualMessageGroupCount: groups.filter(
        (group) =>
          group.getAttribute("data-render-entry-kind") === "message_group",
      ).length,
      firstRenderedTurnId:
        groups[0]?.getAttribute("data-runtime-turn-id") || null,
      lastRenderedTurnId:
        groups.at(-1)?.getAttribute("data-runtime-turn-id") || null,
      historyWindowVisible: Boolean(historyWindow),
      persistedWindowVisible: Boolean(persistedWindow),
      hiddenHistoryCount: Number(
        historyWindow?.getAttribute("data-hidden-history-count") || 0,
      ),
      renderedEntriesCount: Number(
        historyWindow?.getAttribute("data-rendered-entries-count") ||
          groups.length,
      ),
      restoredHistoryWindow:
        historyWindow?.getAttribute("data-restored-history-window") === "true",
      latestUserVisible: bodyText.includes(latestTurn.userText),
      latestAssistantHeadingVisible: bodyText.includes(
        `Long history terminal answer ${fixture.turns.length}`,
      ),
      oldestUserVisible: bodyText.includes(oldestTurn.userText),
      terminalMarkerVisible: bodyText.includes("LONG_HISTORY_TERMINAL_MARKER"),
      longPreviewVisible: Boolean(
        column?.querySelector(
          '[data-testid="message-list-long-history-preview"]',
        ),
      ),
      bodyTextLength: bodyText.length,
      requestMethods: [...new Set(requestMethods)],
      performanceSession,
    };
  }, THREAD_READ_LONG_LIST);
}

export async function runThreadReadLongListDomOracle(page, options) {
  const click = await clickLongListConversation(page, options);
  const startedAt = Date.now();
  let snapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    snapshot = await readLongListSnapshot(page);
    if (
      snapshot.frameSessionId === THREAD_READ_LONG_LIST.sessionId &&
      snapshot.turnGroupCount > 0 &&
      snapshot.latestUserVisible &&
      snapshot.latestAssistantHeadingVisible &&
      snapshot.longPreviewVisible &&
      snapshot.performanceSession?.messageListPaintCount > 0
    ) {
      return { click, snapshot };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `long-list DOM/telemetry 未稳定: ${JSON.stringify(snapshot)}`,
  );
}

export function assertThreadReadLongListDomOracle(result) {
  const snapshot = result?.snapshot ?? {};
  const performance = snapshot.performanceSession ?? {};
  assert(
    THREAD_READ_LONG_LIST.turns.length === THREAD_READ_LONG_LIST_TURN_COUNT,
    "long-list fixture Turn 数漂移",
  );
  assert(
    THREAD_READ_LONG_LIST_ITEM_COUNT === THREAD_READ_LONG_LIST_TURN_COUNT * 3,
    "long-list fixture Item 数漂移",
  );
  assert(
    snapshot.frameSessionId === THREAD_READ_LONG_LIST.sessionId,
    "long-list message list session identity 不正确",
  );
  assert(
    snapshot.turnGroupCount > 0 &&
      snapshot.turnGroupCount <= MAX_INITIAL_DOM_TURN_GROUPS,
    `long-list 首帧 DOM Turn 未受限: ${snapshot.turnGroupCount}`,
  );
  assert(
    snapshot.canonicalTurnGroupCount === snapshot.turnGroupCount &&
      snapshot.residualMessageGroupCount === 0,
    "long-list 首帧未保持 direct canonical Turn owner",
  );
  assert(
    snapshot.historyWindowVisible || snapshot.persistedWindowVisible,
    "long-list 首帧未展示历史窗口边界",
  );
  assert(
    snapshot.hiddenHistoryCount > 0 || snapshot.persistedWindowVisible,
    "long-list 首帧没有隐藏或持久化的更早历史",
  );
  assert(snapshot.latestUserVisible, "long-list 最新 UserMessage 不可见");
  assert(
    snapshot.latestAssistantHeadingVisible,
    "long-list 最新 AgentMessage 预览不可见",
  );
  assert(!snapshot.oldestUserVisible, "long-list 最旧 Turn 不应进入首帧 DOM");
  assert(snapshot.longPreviewVisible, "long-list 长正文未使用有界预览");
  assert(
    !snapshot.terminalMarkerVisible,
    "long-list 长正文尾部不应在显式展开前进入 DOM",
  );
  assert(
    snapshot.bodyTextLength <= MAX_INITIAL_DOM_TEXT_LENGTH,
    `long-list 首帧 DOM 文本过大: ${snapshot.bodyTextLength}`,
  );
  assert(
    REQUIRED_METHODS.every((method) =>
      snapshot.requestMethods.includes(method),
    ),
    `long-list 缺少 current method: ${REQUIRED_METHODS.filter(
      (method) => !snapshot.requestMethods.includes(method),
    ).join(", ")}`,
  );
  assert(
    FORBIDDEN_METHODS.every(
      (method) => !snapshot.requestMethods.includes(method),
    ),
    `long-list 命中 legacy method: ${snapshot.requestMethods.join(", ")}`,
  );
  const clickToFirstPaintMs = finiteNumber(
    performance.clickToFirstMessageListPaintMs,
  );
  const messageListComputeMaxMs = finiteNumber(
    performance.messageListComputeMaxMs,
  );
  assert(
    clickToFirstPaintMs !== null &&
      clickToFirstPaintMs <= MAX_CLICK_TO_FIRST_PAINT_MS,
    `long-list 首次 paint 超限或缺失: ${clickToFirstPaintMs}`,
  );
  assert(
    messageListComputeMaxMs !== null &&
      messageListComputeMaxMs <= MAX_MESSAGE_LIST_COMPUTE_MS,
    `long-list MessageList compute 超限或缺失: ${messageListComputeMaxMs}`,
  );
  assert(
    finiteNumber(performance.finalThreadItemsCount) !== null &&
      performance.finalThreadItemsCount <= MAX_INITIAL_THREAD_ITEMS,
    `long-list 首帧 Item 扫描未受 Turn 窗口约束: ${performance.finalThreadItemsCount}`,
  );

  return {
    proofLevel: "Gate B controlled fixture",
    sessionId: snapshot.frameSessionId,
    threadId: THREAD_READ_LONG_LIST.threadId,
    seededTurnCount: THREAD_READ_LONG_LIST_TURN_COUNT,
    seededItemCount: THREAD_READ_LONG_LIST_ITEM_COUNT,
    dom: {
      turnGroupCount: snapshot.turnGroupCount,
      canonicalTurnGroupCount: snapshot.canonicalTurnGroupCount,
      residualMessageGroupCount: snapshot.residualMessageGroupCount,
      hiddenHistoryCount: snapshot.hiddenHistoryCount,
      renderedEntriesCount: snapshot.renderedEntriesCount,
      persistedWindowVisible: snapshot.persistedWindowVisible,
      longPreviewVisible: snapshot.longPreviewVisible,
      bodyTextLength: snapshot.bodyTextLength,
      oldestUserVisible: snapshot.oldestUserVisible,
      terminalMarkerVisible: snapshot.terminalMarkerVisible,
    },
    performance: {
      clickToFirstMessageListPaintMs: clickToFirstPaintMs,
      clickToMessageListPaintMs: finiteNumber(
        performance.clickToMessageListPaintMs,
      ),
      messageListComputeMaxMs,
      messageListTimelineBuildMaxMs: finiteNumber(
        performance.messageListTimelineBuildMaxMs,
      ),
      messageListThreadItemsScanMaxMs: finiteNumber(
        performance.messageListThreadItemsScanMaxMs,
      ),
      finalThreadItemsCount: finiteNumber(performance.finalThreadItemsCount),
      hiddenHistoryCount: finiteNumber(performance.hiddenHistoryCount),
      persistedHiddenHistoryCount: finiteNumber(
        performance.persistedHiddenHistoryCount,
      ),
      longTaskCount: finiteNumber(performance.longTaskCount),
      longTaskMaxMs: finiteNumber(performance.longTaskMaxMs),
    },
    requestMethods: snapshot.requestMethods,
  };
}
