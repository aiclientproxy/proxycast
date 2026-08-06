import { describe, expect, it } from "vitest";
import type { AgentThreadItem } from "../types";
import {
  formatHistoricalTimelineDuration,
  resolveHistoricalTimelineDurationMs,
  summarizeHistoricalTimelineItems,
} from "./messageListHistoricalPreviewText";

function toolItem(startedAt: string, completedAt: string): AgentThreadItem {
  return {
    id: "tool-1",
    thread_id: "thread-1",
    turn_id: "turn-1",
    sequence: 1,
    status: "completed",
    started_at: startedAt,
    completed_at: completedAt,
    updated_at: completedAt,
    type: "tool_call",
    tool_name: "read_file",
  } as AgentThreadItem;
}

function reasoningItem(
  id: string,
  sequence: number,
  text: string,
  summary?: string[],
): AgentThreadItem {
  return {
    id,
    thread_id: "thread-1",
    turn_id: "turn-1",
    sequence,
    status: "completed",
    started_at: "2026-08-05T12:00:00.000Z",
    updated_at: "2026-08-05T12:00:01.000Z",
    type: "reasoning",
    text,
    ...(summary ? { summary } : {}),
  } as AgentThreadItem;
}

describe("messageListHistoricalPreviewText", () => {
  it("turn 时间戳相同但过程 item 有时间范围时应使用 item 耗时", () => {
    const items = [
      toolItem("2026-08-02T06:00:00.000Z", "2026-08-02T06:00:04.000Z"),
    ];

    expect(
      resolveHistoricalTimelineDurationMs(
        items,
        "2026-08-02T06:00:00.000Z",
        "2026-08-02T06:00:00.000Z",
      ),
    ).toBe(4_000);
    expect(formatHistoricalTimelineDuration(4_000)).toBe("4s");
  });

  it("异常的长 turn 范围不应覆盖实际过程 item 的短耗时", () => {
    const items = [
      toolItem("2026-08-02T06:00:00.000Z", "2026-08-02T06:00:09.000Z"),
    ];

    expect(
      resolveHistoricalTimelineDurationMs(
        items,
        "2026-08-02T05:33:44.000Z",
        "2026-08-02T06:00:00.000Z",
      ),
    ).toBe(9_000);
    expect(formatHistoricalTimelineDuration(9_000)).toBe("9s");
  });

  it("零耗时不应显示误导性的 0s", () => {
    expect(formatHistoricalTimelineDuration(0)).toBeNull();
    expect(formatHistoricalTimelineDuration(null)).toBeNull();
  });

  it("正的小于一秒耗时至少显示 1s", () => {
    expect(formatHistoricalTimelineDuration(250)).toBe("1s");
  });
});

describe("summarizeHistoricalTimelineItems", () => {
  it("返回去重后的 reasoning 摘要，并限制 compact 预览数量", () => {
    const summary = summarizeHistoricalTimelineItems([
      reasoningItem("reasoning-1", 1, "完整正文不应直接进入 compact 预览。", [
        "先确认用户要验证展示时序",
      ]),
      reasoningItem("reasoning-duplicate", 2, "另一个 owner 的完整正文。", [
        "先确认用户要验证展示时序",
      ]),
      reasoningItem("reasoning-2", 3, "第二条完整正文。", ["第二条独立思路。"]),
      reasoningItem("reasoning-3", 4, "第三条完整正文。", [
        "第三条不应进入 compact 预览。",
      ]),
    ]);

    expect(summary.thinkingStepsCount).toBe(4);
    expect(summary.thinkingPreviews).toEqual([
      "先确认用户要验证展示时序",
      "第二条独立思路。",
    ]);
  });
});
