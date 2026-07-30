import { seedCanonicalHistoryThread } from "./session-history-canonical-thread-seed.mjs";

export const THREAD_READ_LONG_LIST_TURN_COUNT = 240;
export const THREAD_READ_LONG_LIST_ITEM_COUNT =
  THREAD_READ_LONG_LIST_TURN_COUNT * 3;

function buildLongMarkdown(turnNumber) {
  const rows = Array.from({ length: 420 }, (_, index) => {
    const row = index + 1;
    return `| ${row} | canonical turn ${turnNumber} | bounded historical markdown row ${row} |`;
  });
  return [
    `# Long history terminal answer ${turnNumber}`,
    "",
    "| Row | Turn | Evidence |",
    "| ---: | --- | --- |",
    ...rows,
    "",
    "LONG_HISTORY_TERMINAL_MARKER",
  ].join("\n");
}

export const THREAD_READ_LONG_LIST = {
  sessionId: "",
  threadId: "",
  workspaceId: null,
  title: "Electron canonical long list performance fixture",
  turns: Array.from(
    { length: THREAD_READ_LONG_LIST_TURN_COUNT },
    (_, index) => {
      const ordinal = index + 1;
      const turnId = `thread-read-long-list-turn-${String(ordinal).padStart(3, "0")}`;
      const userText = `长历史第 ${ordinal} 轮：验证 canonical Turn 窗口。`;
      return {
        turnId,
        reasoningItemId: `item_thread-read-long-list-reasoning-${ordinal}`,
        assistantItemId: `item_thread-read-long-list-assistant-${ordinal}`,
        userText,
        userInputs: [{ type: "text", text: userText }],
        reasoningText: `长历史第 ${ordinal} 轮 reasoning：保持 Item sequence 与折叠边界。`,
        assistantText:
          ordinal === THREAD_READ_LONG_LIST_TURN_COUNT
            ? buildLongMarkdown(ordinal)
            : `长历史第 ${ordinal} 轮结果：direct TurnTimeline 顺序稳定。`,
      };
    },
  ),
};

export function seedThreadReadLongListCanonicalThread({
  runtimeEnv,
  runSqlite,
  sqlLiteral,
  thread,
}) {
  return seedCanonicalHistoryThread({
    runtimeEnv,
    runSqlite,
    sqlLiteral,
    thread,
    fixture: THREAD_READ_LONG_LIST,
    metadataSource: "thread_read_long_list_performance",
  });
}
