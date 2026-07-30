import { seedCanonicalHistoryThread } from "./session-history-canonical-thread-seed.mjs";

export const THREAD_READ_PAGE_ISOMORPHIC = {
  sessionId: "",
  threadId: "",
  workspaceId: null,
  title: "Electron thread read page isomorphic fixture",
  turns: [
    {
      turnId: "thread-read-isomorphic-turn-1",
      reasoningItemId: "item_thread-read-isomorphic-reasoning-1",
      assistantItemId: "item_thread-read-isomorphic-assistant-1",
      userText: "第一轮：建立 thread read 同构基线。",
      userInputs: [
        {
          type: "text",
          text: "第一轮：建立 thread read 同构基线。",
        },
        {
          type: "image",
          uri: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
          detail: "low",
        },
        {
          type: "image",
          uri: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC",
          detail: "high",
        },
      ],
      reasoningText: "第一轮 reasoning：先确认 read page 的 thread id。",
      assistantText: "第一轮结果：read/list/resume 使用同一 thread。",
    },
    {
      turnId: "thread-read-isomorphic-turn-2",
      reasoningItemId: "item_thread-read-isomorphic-reasoning-2",
      assistantItemId: "item_thread-read-isomorphic-assistant-2",
      userText: "第二轮：验证分页不会重排。",
      userInputs: [
        {
          type: "text",
          text: "第二轮：验证分页不会重排。",
        },
      ],
      reasoningText: "第二轮 reasoning：分页窗口只能移动 cursor。",
      assistantText: "第二轮结果：分页窗口保持稳定顺序。",
    },
    {
      turnId: "thread-read-isomorphic-turn-3",
      reasoningItemId: "item_thread-read-isomorphic-reasoning-3",
      assistantItemId: "item_thread-read-isomorphic-assistant-3",
      userText: "第三轮：恢复后 DOM 要和 read model 同源。",
      userInputs: [
        {
          type: "text",
          text: "第三轮：恢复后 DOM 要和 read model 同源。",
        },
      ],
      reasoningText: "第三轮 reasoning：hydrate 必须使用 read model。",
      assistantText: "第三轮结果：hydrate 不再二次拼装 timeline。",
    },
  ],
};

export function seedThreadReadPageIsomorphicCanonicalThread({
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
    fixture: THREAD_READ_PAGE_ISOMORPHIC,
    metadataSource: "thread_read_page_isomorphic",
  });
}
