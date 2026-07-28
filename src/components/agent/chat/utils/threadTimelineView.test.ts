import { describe, expect, it } from "vitest";

import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";
import {
  buildMessageTurnTimeline,
  mergeThreadTurns,
  mergeThreadItems,
} from "./threadTimelineView";

describe("threadTimelineView", () => {
  it("应优先使用 assistant 消息上的 runtimeTurnId 精确绑定执行过程", () => {
    const messages: Message[] = [
      {
        id: "user-1",
        role: "user",
        content: "第一次",
        timestamp: new Date("2026-03-13T10:00:00Z"),
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "余额不足",
        timestamp: new Date("2026-03-13T10:00:01Z"),
        runtimeTurnId: "turn-1",
      },
      {
        id: "user-2",
        role: "user",
        content: "继续",
        timestamp: new Date("2026-03-13T10:01:00Z"),
      },
      {
        id: "assistant-2",
        role: "assistant",
        content: "继续执行",
        timestamp: new Date("2026-03-13T10:01:01Z"),
        runtimeTurnId: "turn-2",
      },
    ];
    const turns: AgentThreadTurn[] = [
      {
        id: "turn-1",
        thread_id: "thread-1",
        prompt_text: "第一次",
        status: "failed",
        started_at: "2026-03-13T10:00:00Z",
        completed_at: "2026-03-13T10:00:02Z",
        created_at: "2026-03-13T10:00:00Z",
        updated_at: "2026-03-13T10:00:02Z",
      },
      {
        id: "turn-2",
        thread_id: "thread-1",
        prompt_text: "继续",
        status: "running",
        started_at: "2026-03-13T10:01:00Z",
        created_at: "2026-03-13T10:01:00Z",
        updated_at: "2026-03-13T10:01:02Z",
      },
    ];
    const items: AgentThreadItem[] = [
      {
        id: "error-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        sequence: 1,
        status: "failed",
        started_at: "2026-03-13T10:00:01Z",
        updated_at: "2026-03-13T10:00:02Z",
        type: "error",
        message: "402 Payment Required",
      },
      {
        id: "process-2",
        thread_id: "thread-1",
        turn_id: "turn-2",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-03-13T10:01:01Z",
        updated_at: "2026-03-13T10:01:02Z",
        type: "reasoning",
        text: "继续处理",
      },
    ];

    const timeline = buildMessageTurnTimeline(messages, turns, items);

    expect(timeline.get("assistant-1")?.turn.id).toBe("turn-1");
    expect(timeline.get("assistant-1")?.items).toEqual([
      expect.objectContaining({ id: "error-1" }),
    ]);
    expect(timeline.get("assistant-2")?.turn.id).toBe("turn-2");
    expect(timeline.get("assistant-2")?.items).toEqual([
      expect.objectContaining({ id: "process-2" }),
    ]);
  });

  it("按时间兜底时不应把新回合执行过程跨到上一轮 assistant", () => {
    const messages: Message[] = [
      {
        id: "user-1",
        role: "user",
        content: "第一次",
        timestamp: new Date("2026-03-13T10:00:00Z"),
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "余额不足",
        timestamp: new Date("2026-03-13T10:00:01Z"),
      },
      {
        id: "user-2",
        role: "user",
        content: "继续",
        timestamp: new Date("2026-03-13T10:01:00Z"),
      },
    ];
    const turns: AgentThreadTurn[] = [
      {
        id: "turn-2",
        thread_id: "thread-1",
        prompt_text: "继续",
        status: "running",
        started_at: "2026-03-13T10:01:00Z",
        created_at: "2026-03-13T10:01:00Z",
        updated_at: "2026-03-13T10:01:02Z",
      },
    ];
    const items: AgentThreadItem[] = [
      {
        id: "process-2",
        thread_id: "thread-1",
        turn_id: "turn-2",
        sequence: 1,
        status: "in_progress",
        started_at: "2026-03-13T10:01:01Z",
        updated_at: "2026-03-13T10:01:02Z",
        type: "reasoning",
        text: "继续处理",
      },
    ];

    const timeline = buildMessageTurnTimeline(messages, turns, items);

    expect(timeline.has("assistant-1")).toBe(false);
  });

  it("应将 turn 对齐到最近的 assistant 消息", () => {
    const messages: Message[] = [
      {
        id: "user-1",
        role: "user",
        content: "旧问题",
        timestamp: new Date("2026-03-13T10:00:00Z"),
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "旧回答",
        timestamp: new Date("2026-03-13T10:00:01Z"),
      },
      {
        id: "user-2",
        role: "user",
        content: "新问题",
        timestamp: new Date("2026-03-13T10:01:00Z"),
      },
      {
        id: "assistant-2",
        role: "assistant",
        content: "新回答",
        timestamp: new Date("2026-03-13T10:01:01Z"),
      },
    ];
    const turns: AgentThreadTurn[] = [
      {
        id: "turn-2",
        thread_id: "thread-1",
        prompt_text: "新问题",
        status: "completed",
        started_at: "2026-03-13T10:01:00Z",
        completed_at: "2026-03-13T10:01:05Z",
        created_at: "2026-03-13T10:01:00Z",
        updated_at: "2026-03-13T10:01:05Z",
      },
    ];
    const items: AgentThreadItem[] = [
      {
        id: "plan-1",
        thread_id: "thread-1",
        turn_id: "turn-2",
        sequence: 2,
        status: "completed",
        started_at: "2026-03-13T10:01:02Z",
        completed_at: "2026-03-13T10:01:03Z",
        updated_at: "2026-03-13T10:01:03Z",
        type: "plan",
        text: "1. 总结\n2. 输出",
      },
    ];

    const timeline = buildMessageTurnTimeline(messages, turns, items);

    expect(timeline.has("assistant-1")).toBe(false);
    expect(timeline.get("assistant-2")?.turn.id).toBe("turn-2");
    expect(timeline.get("assistant-2")?.items).toHaveLength(1);
  });

  it("应合并并排序真实与临时 thread items", () => {
    const persistedItem: AgentThreadItem = {
      id: "item-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      status: "completed",
      started_at: "2026-03-13T10:00:00Z",
      completed_at: "2026-03-13T10:00:01Z",
      updated_at: "2026-03-13T10:00:01Z",
      type: "plan",
      text: "旧计划",
    };
    const syntheticItem: AgentThreadItem = {
      id: "item-2",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 10000,
      status: "in_progress",
      started_at: "2026-03-13T10:00:02Z",
      updated_at: "2026-03-13T10:00:02Z",
      type: "subagent_activity",
      status_label: "running",
      title: "子代理协作",
      summary: "执行中",
    };

    const items = mergeThreadItems([persistedItem], [syntheticItem]);

    expect(items.map((item) => item.id)).toEqual(["item-1", "item-2"]);
  });

  it("同一 turn 内应优先按 sequence 排序而不是按时间戳猜顺序", () => {
    const laterTimestampFirstSequenceItem: AgentThreadItem = {
      id: "item-seq-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      status: "completed",
      started_at: "2026-03-13T10:00:10Z",
      completed_at: "2026-03-13T10:00:11Z",
      updated_at: "2026-03-13T10:00:11Z",
      type: "reasoning",
      text: "先分析问题",
    };
    const earlierTimestampSecondSequenceItem: AgentThreadItem = {
      id: "item-seq-2",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 2,
      status: "completed",
      started_at: "2026-03-13T10:00:01Z",
      completed_at: "2026-03-13T10:00:02Z",
      updated_at: "2026-03-13T10:00:02Z",
      type: "tool_call",
      tool_name: "web_search",
      arguments: { query: "Lime streaming output order" },
      output: "ok",
      success: true,
    };

    const items = mergeThreadItems([
      earlierTimestampSecondSequenceItem,
      laterTimestampFirstSequenceItem,
    ]);

    expect(items.map((item) => item.id)).toEqual(["item-seq-1", "item-seq-2"]);
  });

  it("应忽略仅用于内部恢复成功的 warning 项", () => {
    const repairedWarning: AgentThreadItem = {
      id: "warning-artifact-repaired",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      status: "completed",
      started_at: "2026-03-13T10:00:00Z",
      completed_at: "2026-03-13T10:00:01Z",
      updated_at: "2026-03-13T10:00:01Z",
      type: "warning",
      code: "artifact_document_repaired",
      message:
        "ArtifactDocument 已落盘: 已根据正文整理出一份可继续编辑的草稿。",
    };
    const planItem: AgentThreadItem = {
      id: "plan-2",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 2,
      status: "completed",
      started_at: "2026-03-13T10:00:02Z",
      completed_at: "2026-03-13T10:00:03Z",
      updated_at: "2026-03-13T10:00:03Z",
      type: "plan",
      text: "继续整理结构",
    };

    const items = mergeThreadItems([repairedWarning], [planItem]);

    expect(items).toEqual([planItem]);
  });

  it("应隐藏诊断态摘要并保留上下文整理项", () => {
    const internalRuntimeSummary: AgentThreadItem = {
      id: "summary-runtime-status",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      status: "in_progress",
      started_at: "2026-05-11T10:00:00Z",
      updated_at: "2026-05-11T10:00:01Z",
      type: "turn_summary",
      text: "runtime status should not be rendered as conversation prose",
      metadata: {
        sourceType: "runtime_status",
        surface: "runtime_status",
        visibility: "diagnostics",
      },
    };
    const contextCompaction: AgentThreadItem = {
      id: "context-compaction",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 2,
      status: "completed",
      started_at: "2026-05-11T10:00:01Z",
      completed_at: "2026-05-11T10:00:02Z",
      updated_at: "2026-05-11T10:00:02Z",
      type: "context_compaction",
      stage: "completed",
      trigger: "token_budget",
      detail: "压缩历史上下文",
    };
    const toolItem: AgentThreadItem = {
      id: "tool-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 3,
      status: "completed",
      started_at: "2026-05-11T10:00:03Z",
      completed_at: "2026-05-11T10:00:04Z",
      updated_at: "2026-05-11T10:00:04Z",
      type: "tool_call",
      tool_name: "web_search",
      arguments: { query: "Agent UI" },
      output: "ok",
      success: true,
    };

    expect(
      mergeThreadItems([internalRuntimeSummary, contextCompaction, toolItem]),
    ).toEqual([contextCompaction, toolItem]);
  });

  it("不应再靠 turn_summary 文案内容猜测内部过程", () => {
    const summary: AgentThreadItem = {
      id: "summary-user-visible",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      status: "completed",
      started_at: "2026-05-11T10:00:00Z",
      completed_at: "2026-05-11T10:00:01Z",
      updated_at: "2026-05-11T10:00:01Z",
      type: "turn_summary",
      text: "正在理解意图",
    };

    expect(mergeThreadItems([summary])).toEqual([summary]);
  });

  it("应隐藏辅助运行时投影，避免挤占后续真实 reasoning", () => {
    const messages: Message[] = [
      {
        id: "user-1",
        role: "user",
        content: "先回答一个问题",
        timestamp: new Date("2026-05-06T10:00:00.000Z"),
      },
      {
        id: "assistant-1",
        role: "assistant",
        content: "第一轮回答",
        timestamp: new Date("2026-05-06T10:00:08.000Z"),
      },
      {
        id: "user-2",
        role: "user",
        content: "请继续认真思考",
        timestamp: new Date("2026-05-06T10:00:20.000Z"),
      },
      {
        id: "assistant-2",
        role: "assistant",
        content: "第二轮回答",
        timestamp: new Date("2026-05-06T10:00:25.000Z"),
      },
    ];
    const turns: AgentThreadTurn[] = [
      {
        id: "turn-1",
        thread_id: "thread-1",
        prompt_text: "先回答一个问题",
        status: "completed",
        started_at: "2026-05-06T10:00:00.000Z",
        completed_at: "2026-05-06T10:00:08.000Z",
        created_at: "2026-05-06T10:00:00.000Z",
        updated_at: "2026-05-06T10:00:08.000Z",
      },
      {
        id: "auxiliary-runtime-projection-title-1",
        thread_id: "thread-1",
        prompt_text: "辅助标题生成 · 第一轮回答",
        status: "completed",
        started_at: "2026-05-06T10:00:09.000Z",
        completed_at: "2026-05-06T10:00:10.000Z",
        created_at: "2026-05-06T10:00:09.000Z",
        updated_at: "2026-05-06T10:00:10.000Z",
      },
      {
        id: "turn-2",
        thread_id: "thread-1",
        prompt_text: "请继续认真思考",
        status: "completed",
        started_at: "2026-05-06T10:00:20.000Z",
        completed_at: "2026-05-06T10:00:25.000Z",
        created_at: "2026-05-06T10:00:20.000Z",
        updated_at: "2026-05-06T10:00:25.000Z",
      },
    ];
    const auxiliaryArtifact: AgentThreadItem = {
      id: "auxiliary-title-artifact",
      thread_id: "thread-1",
      turn_id: "auxiliary-runtime-projection-title-1",
      sequence: 1,
      status: "completed",
      started_at: "2026-05-06T10:00:09.000Z",
      completed_at: "2026-05-06T10:00:10.000Z",
      updated_at: "2026-05-06T10:00:10.000Z",
      type: "file_artifact",
      path: ".lime/harness/sessions/session-1/auxiliary-runtime/title-generation-aux-1.json",
      source: "auxiliary.title_generation_result",
      metadata: {
        task_type: "auxiliary_runtime_projection",
        projection_kind: "title_generation",
      },
    };
    const reasoningItem: AgentThreadItem = {
      id: "reasoning-2",
      thread_id: "thread-1",
      turn_id: "turn-2",
      sequence: 1,
      status: "completed",
      started_at: "2026-05-06T10:00:21.000Z",
      completed_at: "2026-05-06T10:00:24.000Z",
      updated_at: "2026-05-06T10:00:24.000Z",
      type: "reasoning",
      text: "第二轮真实思考过程",
    };

    const timeline = buildMessageTurnTimeline(messages, turns, [
      auxiliaryArtifact,
      reasoningItem,
    ]);
    const mergedItems = mergeThreadItems([auxiliaryArtifact, reasoningItem]);

    expect(mergedItems).toEqual([reasoningItem]);
    expect(timeline.has("assistant-1")).toBe(true);
    expect(timeline.get("assistant-2")?.turn.id).toBe("turn-2");
    expect(timeline.get("assistant-2")?.items).toEqual([reasoningItem]);
    expect(
      [...timeline.values()].some((entry) =>
        entry.turn.id.startsWith("auxiliary-runtime-projection-"),
      ),
    ).toBe(false);
  });

  it("应优先将 turn 关联到最接近完成时刻的 assistant 消息", () => {
    const messages: Message[] = [
      {
        id: "user-1",
        role: "user",
        content: "请生成图片",
        timestamp: new Date("2026-03-18T08:35:41.000Z"),
      },
      {
        id: "assistant-thinking",
        role: "assistant",
        content: "我先整理思路。",
        timestamp: new Date("2026-03-18T08:35:43.000Z"),
      },
      {
        id: "assistant-final",
        role: "assistant",
        content: "这是最终可用的 Prompt。",
        timestamp: new Date("2026-03-18T08:36:05.000Z"),
      },
      {
        id: "assistant-followup",
        role: "assistant",
        content: "如果你需要，我还可以继续细化风格。",
        timestamp: new Date("2026-03-18T08:36:20.000Z"),
      },
    ];
    const turns: AgentThreadTurn[] = [
      {
        id: "turn-1",
        thread_id: "thread-1",
        prompt_text: "请生成图片",
        status: "completed",
        started_at: "2026-03-18T08:35:41.000Z",
        completed_at: "2026-03-18T08:36:06.000Z",
        created_at: "2026-03-18T08:35:41.000Z",
        updated_at: "2026-03-18T08:36:06.000Z",
      },
    ];
    const items: AgentThreadItem[] = [
      {
        id: "reasoning-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        sequence: 1,
        status: "completed",
        started_at: "2026-03-18T08:35:43.000Z",
        completed_at: "2026-03-18T08:36:05.000Z",
        updated_at: "2026-03-18T08:36:05.000Z",
        type: "reasoning",
        text: "先理解主题，再组织 Prompt。",
      },
    ];

    const timeline = buildMessageTurnTimeline(messages, turns, items);

    expect(timeline.has("assistant-thinking")).toBe(false);
    expect(timeline.get("assistant-final")?.turn.id).toBe("turn-1");
    expect(timeline.has("assistant-followup")).toBe(false);
  });

  it("相同距离下应优先绑定到有实际内容的 assistant 消息", () => {
    const messages: Message[] = [
      {
        id: "assistant-empty",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-03-18T08:36:05.000Z"),
        contentParts: [],
      },
      {
        id: "assistant-substantive",
        role: "assistant",
        content: "这是最终答复",
        timestamp: new Date("2026-03-18T08:36:05.000Z"),
      },
    ];
    const turns: AgentThreadTurn[] = [
      {
        id: "turn-1",
        thread_id: "thread-1",
        prompt_text: "帮我处理",
        status: "completed",
        started_at: "2026-03-18T08:35:41.000Z",
        completed_at: "2026-03-18T08:36:05.000Z",
        created_at: "2026-03-18T08:35:41.000Z",
        updated_at: "2026-03-18T08:36:05.000Z",
      },
    ];

    const timeline = buildMessageTurnTimeline(messages, turns, []);

    expect(timeline.has("assistant-empty")).toBe(false);
    expect(timeline.get("assistant-substantive")?.turn.id).toBe("turn-1");
  });

  it("应在增量刷新时合并并覆盖同 id 的 turn", () => {
    const previousTurns: AgentThreadTurn[] = [
      {
        id: "turn-1",
        thread_id: "thread-1",
        prompt_text: "帮我处理",
        status: "running",
        started_at: "2026-03-18T08:35:41.000Z",
        created_at: "2026-03-18T08:35:41.000Z",
        updated_at: "2026-03-18T08:35:41.000Z",
      },
    ];
    const incomingTurns: AgentThreadTurn[] = [
      {
        id: "turn-1",
        thread_id: "thread-1",
        prompt_text: "帮我处理",
        status: "completed",
        started_at: "2026-03-18T08:35:41.000Z",
        completed_at: "2026-03-18T08:36:05.000Z",
        created_at: "2026-03-18T08:35:41.000Z",
        updated_at: "2026-03-18T08:36:05.000Z",
      },
    ];

    const mergedTurns = mergeThreadTurns(previousTurns, incomingTurns);

    expect(mergedTurns).toHaveLength(1);
    expect(mergedTurns[0]?.status).toBe("completed");
    expect(mergedTurns[0]?.completed_at).toBe("2026-03-18T08:36:05.000Z");
  });

  it("历史 turn 缺少 prompt_text 时不应中断会话水合", () => {
    const persistedTurn = {
      id: "turn-from-app-server-detail",
      thread_id: "thread-1",
      status: "completed",
      started_at: "2026-06-07T06:17:13.000Z",
      completed_at: "2026-06-07T06:17:14.000Z",
      created_at: "2026-06-07T06:17:13.000Z",
      updated_at: "2026-06-07T06:17:14.000Z",
    } as AgentThreadTurn;

    expect(() => mergeThreadTurns([persistedTurn])).not.toThrow();
    expect(mergeThreadTurns([persistedTurn])).toEqual([persistedTurn]);
  });

  it("历史 turn 或 item 缺少 started_at 时不应中断排序", () => {
    const turnWithoutStart = {
      id: "turn-without-start",
      thread_id: "thread-1",
      prompt_text: "整理今天的国际新闻",
      status: "failed",
      created_at: "2026-06-07T07:53:20.000Z",
      updated_at: "2026-06-07T07:55:20.000Z",
    } as AgentThreadTurn;
    const itemWithoutStart = {
      id: "item-without-start",
      thread_id: "thread-1",
      turn_id: "turn-without-start",
      sequence: 1,
      status: "failed",
      updated_at: "2026-06-07T07:55:20.000Z",
      type: "error",
      message: "执行已中断",
    } as AgentThreadItem;

    expect(() => mergeThreadTurns([turnWithoutStart])).not.toThrow();
    expect(() => mergeThreadItems([itemWithoutStart])).not.toThrow();
    expect(mergeThreadTurns([turnWithoutStart])).toEqual([turnWithoutStart]);
    expect(mergeThreadItems([itemWithoutStart])).toEqual([itemWithoutStart]);
  });
});
