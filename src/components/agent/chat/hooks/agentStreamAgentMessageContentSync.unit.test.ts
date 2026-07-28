import { describe, expect, it, vi } from "vitest";
import type { Dispatch, SetStateAction } from "react";
import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import type { Message } from "../types";
import {
  mergeAssistantAgentMessageContentPartsFromThreadItems,
  syncAssistantAgentMessageContentPartFromThreadItem,
} from "./agentStreamAgentMessageContentSync";

describe("agentStreamAgentMessageContentSync", () => {
  it("同一 commentary item 的前缀快照应原位替换并保持在后续工具之前", () => {
    const commentaryText =
      "第一层已经露出几个可疑点：仓库根目录有 `.DS_Store`、`tsconfig.node.tsbuildinfo`，甚至还有 `lime.db`。先别急着给它们判刑——我继续核对是否被 Git 跟踪、是否有合理用途，再看真正的大头。";
    const prefixItem: AgentThreadItem = {
      id: "agent-message-commentary",
      thread_id: "thread-1",
      turn_id: "turn-1",
      type: "agent_message",
      status: "in_progress",
      ordinal: 103,
      sequence: 150,
      text: commentaryText.slice(0, 42),
      phase: "commentary",
      started_at: "2026-07-18T12:15:54.122Z",
      updated_at: "2026-07-18T12:15:58.000Z",
    };
    const completedItem: AgentThreadItem = {
      ...prefixItem,
      status: "completed",
      sequence: 171,
      text: commentaryText,
      updated_at: "2026-07-18T12:16:01.150Z",
      completed_at: "2026-07-18T12:16:01.150Z",
    };
    const toolItem: AgentThreadItem = {
      id: "tool-read",
      thread_id: "thread-1",
      turn_id: "turn-1",
      type: "tool_call",
      tool_name: "Read",
      arguments: { file_path: "README.md" },
      status: "completed",
      ordinal: 104,
      sequence: 160,
      started_at: "2026-07-18T12:15:59.000Z",
      updated_at: "2026-07-18T12:16:00.000Z",
      completed_at: "2026-07-18T12:16:00.000Z",
    };
    let messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-18T12:15:54.122Z"),
        contentParts: [
          {
            type: "tool_use",
            toolCall: {
              id: "tool-read",
              name: "Read",
              arguments: { file_path: "README.md" },
              status: "completed",
            },
          },
        ],
      },
    ];
    const setMessages: Dispatch<SetStateAction<Message[]>> = vi.fn(
      (value: SetStateAction<Message[]>) => {
        messages = typeof value === "function" ? value(messages) : value;
      },
    );

    syncAssistantAgentMessageContentPartFromThreadItem({
      assistantMsgId: "assistant-1",
      item: prefixItem,
      threadItems: [prefixItem, toolItem],
      setMessages,
    });
    syncAssistantAgentMessageContentPartFromThreadItem({
      assistantMsgId: "assistant-1",
      item: completedItem,
      threadItems: [completedItem, toolItem],
      setMessages,
    });

    expect(messages[0]?.contentParts?.map((part) => part.type)).toEqual([
      "text",
      "tool_use",
    ]);
    expect(
      messages[0]?.contentParts?.filter((part) => part.type === "text"),
    ).toEqual([
      expect.objectContaining({
        text: commentaryText,
        metadata: expect.objectContaining({
          itemId: "agent-message-commentary",
          phase: "commentary",
          sequence: 103,
        }),
      }),
    ]);
  });

  it("完成态合并 agent_message content part 时不应跨 runtime turn 串线", () => {
    const previousTurnItem: AgentThreadItem = {
      id: "agent-message-final-turn-old",
      thread_id: "thread-1",
      turn_id: "turn-old",
      type: "agent_message",
      status: "completed",
      sequence: 10,
      text: "上一轮专家 Skills runtime 证据已完成",
      phase: "final_answer",
      started_at: "2026-06-26T10:00:00.000Z",
      updated_at: "2026-06-26T10:00:01.000Z",
      completed_at: "2026-06-26T10:00:01.000Z",
    };
    const currentTurnItem: AgentThreadItem = {
      id: "agent-message-final-turn-current",
      thread_id: "thread-1",
      turn_id: "turn-current",
      type: "agent_message",
      status: "completed",
      sequence: 20,
      text: "当前轮专家面板新增 Skill 后的下一轮 runtime 证据已完成",
      phase: "final_answer",
      started_at: "2026-06-26T10:01:00.000Z",
      updated_at: "2026-06-26T10:01:01.000Z",
      completed_at: "2026-06-26T10:01:01.000Z",
    };

    const parts = mergeAssistantAgentMessageContentPartsFromThreadItems({
      items: [previousTurnItem, currentTurnItem],
      turnId: "turn-current",
    });

    expect(parts).toEqual([
      expect.objectContaining({
        type: "text",
        text: "当前轮专家面板新增 Skill 后的下一轮 runtime 证据已完成",
        metadata: expect.objectContaining({
          itemId: "agent-message-final-turn-current",
          phase: "final_answer",
          turnId: "turn-current",
        }),
      }),
    ]);
  });

  it("合并 agent_message media contentParts 时保留引用元数据", () => {
    const mediaItem: AgentThreadItem = {
      id: "agent-message-media",
      thread_id: "thread-1",
      turn_id: "turn-media",
      type: "agent_message",
      status: "completed",
      sequence: 20,
      text: "",
      phase: "final_answer",
      contentParts: [
        {
          type: "media",
          kind: "image",
          caption: "结果图",
          reference: {
            uri: "sidecar://media/image-1",
            mime_type: "image/png",
            title: "image-1.png",
          },
        },
      ],
      started_at: "2026-06-26T10:01:00.000Z",
      updated_at: "2026-06-26T10:01:01.000Z",
      completed_at: "2026-06-26T10:01:01.000Z",
    };

    const parts = mergeAssistantAgentMessageContentPartsFromThreadItems({
      items: [mediaItem],
      turnId: "turn-media",
    });

    expect(parts).toEqual([
      expect.objectContaining({
        type: "media_reference",
        reference: expect.objectContaining({
          caption: "结果图",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          kind: "image",
        }),
        metadata: expect.objectContaining({
          source: "agent_media_reference",
          itemId: "agent-message-media",
          referenceUri: "sidecar://media/image-1",
          mimeType: "image/png",
          mediaKind: "image",
        }),
      }),
    ]);
  });

  it("hosted image completed item 应实时且 exactly once 合并到当前 Turn", () => {
    const item: AgentThreadItem = {
      id: "image-generation-hosted",
      thread_id: "thread-1",
      turn_id: "turn-image",
      type: "image_generation",
      status: "completed",
      generation_status: "completed",
      sequence: 21,
      revised_prompt: "a green circle",
      result: "aW1hZ2U=",
      saved_path: "/tmp/green-circle.png",
      started_at: "2026-07-27T10:01:00.000Z",
      updated_at: "2026-07-27T10:01:01.000Z",
      completed_at: "2026-07-27T10:01:01.000Z",
    };
    const initial: Message["contentParts"] = [
      { type: "text", text: "图片已生成。" },
    ];
    const first = mergeAssistantAgentMessageContentPartsFromThreadItems({
      items: [item],
      parts: initial,
      turnId: "turn-image",
    });
    const second = mergeAssistantAgentMessageContentPartsFromThreadItems({
      items: [item],
      parts: first,
      turnId: "turn-image",
    });

    expect(second?.filter((part) => part.type === "media_reference")).toEqual([
      expect.objectContaining({
        reference: expect.objectContaining({
          uri: "data:image/png;base64,aW1hZ2U=",
          caption: "a green circle",
          sourcePath: "/tmp/green-circle.png",
        }),
        metadata: expect.objectContaining({
          source: "hosted_image_generation",
          threadItemId: "image-generation-hosted",
        }),
      }),
    ]);
  });

  it("同步 content part 无实际变化时应保留 messages 数组引用", () => {
    const item: AgentThreadItem = {
      id: "agent-message-final",
      thread_id: "thread-1",
      turn_id: "turn-current",
      type: "agent_message",
      status: "completed",
      sequence: 20,
      text: "最终摘要",
      phase: "final_answer",
      started_at: "2026-06-26T10:01:00.000Z",
      updated_at: "2026-06-26T10:01:01.000Z",
      completed_at: "2026-06-26T10:01:01.000Z",
    };
    const messages: Message[] = [
      {
        id: "assistant-1",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-26T10:01:01.000Z"),
        contentParts: [
          {
            type: "text",
            text: "最终摘要",
            metadata: {
              itemId: "agent-message-final",
              phase: "final_answer",
              sequence: 20,
              source: "agent_text_delta",
              turnId: "turn-current",
            },
          },
        ],
      },
    ];
    let nextMessages: Message[] | undefined;
    const setMessages: Dispatch<SetStateAction<Message[]>> = vi.fn(
      (value: SetStateAction<Message[]>) => {
        nextMessages = typeof value === "function" ? value(messages) : value;
      },
    );

    syncAssistantAgentMessageContentPartFromThreadItem({
      assistantMsgId: "assistant-1",
      item,
      setMessages,
    });

    expect(setMessages).toHaveBeenCalledTimes(1);
    expect(nextMessages).toBe(messages);
  });

  it("同步图片任务 agent_message content part 时保留后端原文", () => {
    const item: AgentThreadItem = {
      id: "agent-message-image",
      thread_id: "thread-1",
      turn_id: "turn-image",
      type: "agent_message",
      status: "completed",
      sequence: 20,
      text: "好啊，先来Generate深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
      phase: "final_answer",
      started_at: "2026-06-26T10:01:00.000Z",
      updated_at: "2026-06-26T10:01:01.000Z",
      completed_at: "2026-06-26T10:01:01.000Z",
    };
    const messages: Message[] = [
      {
        id: "assistant-image",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-06-26T10:01:01.000Z"),
        contentParts: [],
      },
    ];
    let nextMessages: Message[] | undefined;
    const setMessages: Dispatch<SetStateAction<Message[]>> = vi.fn(
      (value: SetStateAction<Message[]>) => {
        nextMessages = typeof value === "function" ? value(messages) : value;
      },
    );

    syncAssistantAgentMessageContentPartFromThreadItem({
      assistantMsgId: "assistant-image",
      item,
      setMessages,
    });

    const textPart = nextMessages?.[0]?.contentParts?.find(
      (
        part,
      ): part is Extract<
        NonNullable<Message["contentParts"]>[number],
        { type: "text" }
      > => part.type === "text",
    );
    expect(textPart?.text).toBe(
      "好啊，先来Generate深圳夏day午后的城市照片，阳光明亮，真实摄影Style。",
    );
  });
});
