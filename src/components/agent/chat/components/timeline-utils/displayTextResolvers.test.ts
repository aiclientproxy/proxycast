import { describe, expect, it } from "vitest";

import type { AgentThreadItem } from "../../types";
import { resolveThinkingDisplayText } from "./displayTextResolvers";

function reasoningItem(
  overrides: Partial<Extract<AgentThreadItem, { type: "reasoning" }>>,
): Extract<AgentThreadItem, { type: "reasoning" }> {
  return {
    id: "reasoning-1",
    thread_id: "thread-1",
    turn_id: "turn-1",
    sequence: 1,
    status: "completed",
    started_at: "2026-05-30T00:00:00.000Z",
    updated_at: "2026-05-30T00:00:00.000Z",
    type: "reasoning",
    text: "",
    ...overrides,
  };
}

describe("resolveReasoningDisplayText", () => {
  it("timeline reasoning 应显示 provider summary 而不显示 raw text", () => {
    const text = resolveThinkingDisplayText(
      reasoningItem({
        summary: ["Finding latest news.", "正在核对来源可信度。"],
        text: "Investigating tool calls for WebSearch.\n已整理可用来源。",
      }),
    );

    expect(text).toContain("Finding latest news");
    expect(text).toContain("正在核对来源可信度。");
    expect(text).not.toContain("Investigating tool calls");
    expect(text).not.toContain("已整理可用来源。");
  });

  it("英文 summary 应保留，但不应追加 raw text", () => {
    const text = resolveThinkingDisplayText(
      reasoningItem({
        summary: ["Finding latest news."],
        text: "I'm thinking about available tools.",
      }),
    );

    expect(text).toContain("Finding latest news.");
    expect(text).not.toContain("I'm thinking about available tools.");
  });

  it("canonical reasoning 只显示 summary，不显示 raw content", () => {
    const text = resolveThinkingDisplayText(
      reasoningItem({
        summary: ["先确认用户意图。"],
        content: ["完整推理第一段。", "完整推理第二段。"],
        text: "先确认用户意图。",
      }),
    );

    expect(text).toBe("先确认用户意图。");
  });

  it("没有 summary 的 canonical reasoning 不应回退显示 raw content", () => {
    expect(
      resolveThinkingDisplayText(
        reasoningItem({
          text: "模型返回的完整思考。",
          content: ["模型返回的完整思考。"],
        }),
      ),
    ).toBe("");
  });
});
