import { describe, expect, it } from "vitest";

import type { AgentThreadItem } from "../types";
import { appendInterruptedPlaceholderToThreadItem } from "./agentInterruptedMessageContent";

describe("agentInterruptedMessageContent", () => {
  it("停止 ThreadItem 应让 partial 正文与 marker 同时进入 contentParts", () => {
    const item: AgentThreadItem = {
      id: "agent-message-interrupted-partial",
      type: "agent_message",
      thread_id: "thread-interrupted-partial",
      turn_id: "turn-interrupted-partial",
      sequence: 1,
      status: "in_progress",
      started_at: "2026-08-21T00:00:00.000Z",
      updated_at: "2026-08-21T00:00:01.000Z",
      text: "以下是今日国际新闻简要整理：",
      phase: "final_answer",
    };

    expect(appendInterruptedPlaceholderToThreadItem(item)).toMatchObject({
      text: "以下是今日国际新闻简要整理：\n\n(已停止)",
      contentParts: [
        { type: "text", text: "以下是今日国际新闻简要整理：" },
        { type: "text", text: "(已停止)" },
      ],
    });
  });
});
