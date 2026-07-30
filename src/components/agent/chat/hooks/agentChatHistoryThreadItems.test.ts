import { describe, expect, it } from "vitest";

import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";

import {
  collectDetailThreadItems,
  hydrateSessionDetailMessagesFromThreadItems,
} from "./agentChatHistoryThreadItems";

describe("agentChatHistoryThreadItems", () => {
  it("canonical process Items 不应再合成为 Message", () => {
    const detail = {
      id: "canonical-process-owner",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-1",
          thread_id: "thread-1",
          status: "completed",
        },
      ],
      items: [
        {
          id: "reasoning-1",
          type: "reasoning",
          text: "canonical reasoning",
        },
        {
          id: "plan-1",
          type: "plan",
          text: "canonical plan",
          metadata: { revision_id: "revision-1" },
        },
        {
          id: "tool-1",
          type: "tool_call",
          tool_name: "Read",
          arguments: { path: "README.md" },
        },
        {
          id: "patch-1",
          type: "patch",
          text: "canonical patch",
          paths: ["src/example.ts"],
        },
        {
          id: "media-1",
          type: "media",
          media_type: "image",
          uri: "sidecar://media/image-1",
        },
        {
          id: "image-1",
          type: "image_generation",
          generation_status: "completed",
          result: "aW1hZ2U=",
        },
        {
          id: "approval-1",
          type: "approval_request",
          request_id: "approval-1",
          action_type: "command_execution",
        },
      ].map((item, index) => ({
        ...item,
        thread_id: "thread-1",
        turn_id: "turn-1",
        sequence: index + 1,
        status: "completed",
        started_at: `2026-07-29T10:00:0${index}.000Z`,
        updated_at: `2026-07-29T10:00:0${index}.000Z`,
      })),
    } as unknown as AgentSessionDetail;

    expect(
      hydrateSessionDetailMessagesFromThreadItems(
        detail,
        "canonical-process-owner",
      ),
    ).toEqual([]);
  });

  it("冷历史只保留 canonical User 与最终 AgentMessage 临时锚点", () => {
    const detail = {
      id: "canonical-message-anchors",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-1",
          thread_id: "thread-1",
          status: "completed",
        },
      ],
      items: [
        {
          id: "user-1",
          type: "user_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 1,
          status: "completed",
          content: "继续",
          started_at: "2026-07-29T10:00:00.000Z",
          updated_at: "2026-07-29T10:00:00.000Z",
        },
        {
          id: "agent-process",
          type: "agent_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 2,
          status: "completed",
          phase: "commentary",
          text: "我先核对实现。",
          started_at: "2026-07-29T10:00:01.000Z",
          updated_at: "2026-07-29T10:00:01.000Z",
        },
        {
          id: "agent-final",
          type: "agent_message",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 3,
          status: "completed",
          phase: "final_answer",
          text: "最终答复。",
          started_at: "2026-07-29T10:00:02.000Z",
          updated_at: "2026-07-29T10:00:02.000Z",
          completed_at: "2026-07-29T10:00:02.000Z",
        },
      ],
    } as unknown as AgentSessionDetail;

    const messages = hydrateSessionDetailMessagesFromThreadItems(
      detail,
      "canonical-message-anchors",
    );

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      id: "canonical-message-anchors-timeline-user-1",
      role: "user",
      content: "继续",
      runtimeTurnId: "turn-1",
    });
    expect(messages[1]).toMatchObject({
      id: "canonical-message-anchors-timeline-agent-final",
      role: "assistant",
      content: "最终答复。",
      contentParts: [{ type: "text", text: "最终答复。" }],
      runtimeTurnId: "turn-1",
    });
    expect(JSON.stringify(messages)).not.toContain("我先核对实现");
  });

  it("同一 Item 出现在 detail 与 thread/read 时应只保留一次", () => {
    const item = {
      id: "user-shared",
      type: "user_message",
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      status: "completed",
      content: "hello",
      started_at: "2026-07-29T10:00:00.000Z",
      updated_at: "2026-07-29T10:00:00.000Z",
    };
    const detail = {
      id: "dedupe-items",
      created_at: 1,
      updated_at: 2,
      messages: [],
      items: [item],
      thread_read: {
        thread_id: "thread-1",
        thread_items: [item],
      },
    } as unknown as AgentSessionDetail;

    expect(collectDetailThreadItems(detail)).toHaveLength(1);
  });
});
