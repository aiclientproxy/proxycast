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

  it("canonical user content_parts 中的图片应进入历史 Message，图片-only 也不能丢失", () => {
    const detail = {
      id: "history-user-images",
      created_at: 1,
      updated_at: 2,
      messages: [],
      turns: [
        {
          id: "turn-images",
          thread_id: "thread-images",
          status: "completed",
        },
      ],
      items: [
        {
          id: "user-images",
          type: "user_message",
          thread_id: "thread-images",
          turn_id: "turn-images",
          sequence: 1,
          status: "completed",
          content: "",
          content_parts: [
            {
              type: "image",
              mime_type: "image/png",
              data: "aW1hZ2U=",
            },
            {
              type: "image",
              mime_type: "image/jpeg",
              data: "",
              source_path: "/tmp/imported-photo.jpg",
            },
          ],
          started_at: "2026-07-29T10:00:00.000Z",
          updated_at: "2026-07-29T10:00:00.000Z",
        },
      ],
    } as unknown as AgentSessionDetail;

    const messages = hydrateSessionDetailMessagesFromThreadItems(
      detail,
      "history-user-images",
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({
      role: "user",
      content: "",
      images: [
        expect.objectContaining({
          mediaType: "image/png",
          data: "aW1hZ2U=",
        }),
        expect.objectContaining({
          mediaType: "image/jpeg",
          sourcePath: "/tmp/imported-photo.jpg",
        }),
      ],
    });
  });

  it("Codex response_item 与 event_msg 图片输入跨 turn 只保留一条且优先内联图片", () => {
    const sourceThreadId = "codex-thread-duplicate-image";
    const responseItem = {
      id: "imported-user-response",
      type: "user_message",
      thread_id: "thread-images",
      turn_id: "turn-response",
      sequence: 1,
      status: "completed",
      content:
        '<image name=[Image #1] path="/tmp/imported-photo.jpg">\n</image>\n[Image #1] 请检查这张图',
      content_parts: [
        {
          type: "image",
          mime_type: "image/png",
          data: "inline-image-data",
        },
      ],
      metadata: {
        imported: true,
        source_event_seq: 9,
        source_provenance: {
          sourceEventType: "message",
          sourceEventSeq: 9,
          sourceThreadId,
        },
      },
      started_at: "2026-07-29T10:00:00.000Z",
      updated_at: "2026-07-29T10:00:00.000Z",
    };
    const eventItem = {
      id: "imported-user-event",
      type: "user_message",
      thread_id: "thread-images",
      turn_id: "turn-event",
      sequence: 1,
      status: "completed",
      content: "[Image #1] 请检查这张图",
      content_parts: [
        {
          type: "image",
          mime_type: "image/png",
          data: "",
          source_path: "/tmp/imported-photo.jpg",
        },
      ],
      metadata: {
        imported: true,
        source_event_seq: 10,
        source_provenance: {
          sourceEventType: "user_message",
          sourceEventSeq: 10,
          sourceThreadId,
        },
      },
      started_at: "2026-07-29T10:00:00.100Z",
      updated_at: "2026-07-29T10:00:00.100Z",
    };
    const detail = {
      id: "duplicate-imported-image",
      created_at: 1,
      updated_at: 2,
      messages: [],
      items: [responseItem, eventItem],
    } as unknown as AgentSessionDetail;

    const items = collectDetailThreadItems(detail);
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      id: "imported-user-event",
      turn_id: "turn-event",
      content: "[Image #1] 请检查这张图",
      content_parts: [
        expect.objectContaining({
          type: "image",
          data: "inline-image-data",
        }),
      ],
    });
    expect(items[0]).not.toMatchObject({
      content_parts: [
        expect.objectContaining({ source_path: "/tmp/imported-photo.jpg" }),
      ],
    });

    const messages = hydrateSessionDetailMessagesFromThreadItems(
      detail,
      "duplicate-imported-image",
    );
    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({
      role: "user",
      content: "图片请检查这张图",
      images: [expect.objectContaining({ data: "inline-image-data" })],
    });
  });
});
