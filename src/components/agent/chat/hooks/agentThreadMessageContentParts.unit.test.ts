import { describe, expect, it } from "vitest";
import type { AgentThreadItem } from "@/lib/api/agentProtocol";
import {
  imageGenerationContentPartFromThreadItem,
  mediaReferenceContentPartFromThreadItem,
  messageContentPartsFromAgentThreadItem,
} from "./agentThreadMessageContentParts";

type AgentMessageItem = Extract<AgentThreadItem, { type: "agent_message" }>;

function agentMessageItem(
  overrides: Partial<AgentMessageItem> = {},
): AgentMessageItem {
  return {
    id: "agent-message-1",
    thread_id: "thread-1",
    turn_id: "turn-1",
    type: "agent_message",
    status: "completed",
    sequence: 7,
    text: "",
    phase: "final_answer",
    started_at: "2026-07-07T10:00:00.000Z",
    updated_at: "2026-07-07T10:00:01.000Z",
    completed_at: "2026-07-07T10:00:01.000Z",
    ...overrides,
  };
}

describe("messageContentPartsFromAgentThreadItem", () => {
  it("把 App Server media contentParts 转成 GUI media reference", () => {
    const parts = messageContentPartsFromAgentThreadItem(
      agentMessageItem({
        contentParts: [
          {
            type: "text",
            text: "图片已生成。",
          },
          {
            type: "media",
            kind: "image",
            caption: "结果图",
            reference: {
              uri: "sidecar://media/image-1",
              ref_id: "media-ref-image-1",
              mime_type: "image/png",
              title: "image-1.png",
              source_uri: "sidecar://media/image-1",
              source_path: "/tmp/lime-media/image-1.png",
              preview_url: "asset:///tmp/lime-media/image-1.png",
              sha256: "sha256-image-1",
              byte_size: 2048,
            },
          },
        ],
      }),
    );

    expect(parts).toEqual([
      expect.objectContaining({
        type: "text",
        text: "图片已生成。",
        metadata: expect.objectContaining({
          source: "agent_text_delta",
          itemId: "agent-message-1",
          turnId: "turn-1",
          sequence: 7,
          contentPartIndex: 0,
        }),
      }),
      expect.objectContaining({
        type: "media_reference",
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          refId: "media-ref-image-1",
          mimeType: "image/png",
          title: "image-1.png",
          caption: "结果图",
          sourceUri: "sidecar://media/image-1",
          sourcePath: "/tmp/lime-media/image-1.png",
          previewUrl: "asset:///tmp/lime-media/image-1.png",
          sha256: "sha256-image-1",
          byteSize: 2048,
        },
        metadata: expect.objectContaining({
          source: "agent_media_reference",
          itemId: "agent-message-1",
          threadItemId: "agent-message-1",
          turnId: "turn-1",
          sequence: 7,
          contentPartIndex: 1,
          referenceUri: "sidecar://media/image-1",
          mediaKind: "image",
          refId: "media-ref-image-1",
          mimeType: "image/png",
          sourceUri: "sidecar://media/image-1",
          sourcePath: "/tmp/lime-media/image-1.png",
          previewUrl: "asset:///tmp/lime-media/image-1.png",
        }),
      }),
    ]);
  });

  it("丢弃 inline data source owner，避免 GUI 消费 provider payload", () => {
    const parts = messageContentPartsFromAgentThreadItem(
      agentMessageItem({
        contentParts: [
          {
            type: "media",
            kind: "image",
            caption: "结果图",
            reference: {
              uri: "sidecar://media/image-1",
              mime_type: "image/png",
              source_uri: "data:image/png;base64,AAAA",
              preview_url: "data:image/png;base64,BBBB",
            },
          },
        ],
      }),
    );

    expect(parts).toEqual([
      expect.objectContaining({
        type: "media_reference",
        reference: expect.not.objectContaining({
          sourceUri: expect.any(String),
          previewUrl: expect.any(String),
        }),
        metadata: expect.not.objectContaining({
          sourceUri: expect.any(String),
          previewUrl: expect.any(String),
        }),
      }),
    ]);
  });

  it("拒绝 inline data URI，避免 GUI 消费 provider wire payload", () => {
    const parts = messageContentPartsFromAgentThreadItem(
      agentMessageItem({
        contentParts: [
          {
            type: "media",
            kind: "image",
            caption: "不应展示",
            reference: {
              uri: "data:image/png;base64,AAAA",
              mime_type: "image/png",
            },
          },
        ],
      }),
    );

    expect(parts).toEqual([]);
  });
});

describe("imageGenerationContentPartFromThreadItem", () => {
  const item = {
    id: "image-generation-1",
    thread_id: "thread-1",
    turn_id: "turn-1",
    type: "image_generation",
    status: "completed",
    generation_status: "completed",
    sequence: 9,
    revised_prompt: "a blue square",
    result: "Zm9v",
    saved_path: "/tmp/blue-square.png",
    started_at: "2026-07-27T10:00:00.000Z",
    updated_at: "2026-07-27T10:00:01.000Z",
    completed_at: "2026-07-27T10:00:01.000Z",
  } satisfies Extract<AgentThreadItem, { type: "image_generation" }>;

  it("把 Codex hosted result 投影为可展示的 PNG media reference", () => {
    expect(imageGenerationContentPartFromThreadItem(item)).toEqual({
      type: "media_reference",
      reference: {
        kind: "image",
        uri: "data:image/png;base64,Zm9v",
        mimeType: "image/png",
        title: "blue-square.png",
        caption: "a blue square",
        sourcePath: "/tmp/blue-square.png",
      },
      metadata: {
        itemId: "image-generation-1",
        threadItemId: "image-generation-1",
        turnId: "turn-1",
        sequence: 9,
        contentPartIndex: 0,
        source: "hosted_image_generation",
        generationStatus: "completed",
        mediaKind: "image",
        mimeType: "image/png",
        caption: "a blue square",
        sourcePath: "/tmp/blue-square.png",
      },
    });
  });

  it.each([
    ["in_progress item", { status: "in_progress", result: "" }],
    [
      "failed item",
      { status: "failed", generation_status: "failed", result: "" },
    ],
    ["empty completed result", { result: "" }],
  ])("%s 不生成破图", (_label, overrides) => {
    expect(
      imageGenerationContentPartFromThreadItem({ ...item, ...overrides }),
    ).toBeNull();
  });
});

describe("mediaReferenceContentPartFromThreadItem", () => {
  it("解码 sidecar URI 的可见文件名并保留安全 handle", () => {
    const item = {
      id: "media-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      type: "media",
      status: "completed",
      sequence: 10,
      uri: "sidecar://media/output-deadbeef/fixture%20result.png",
      mime_type: "image/png",
      started_at: "2026-07-27T10:00:00.000Z",
      updated_at: "2026-07-27T10:00:01.000Z",
      completed_at: "2026-07-27T10:00:01.000Z",
    } satisfies Extract<AgentThreadItem, { type: "media" }>;

    expect(mediaReferenceContentPartFromThreadItem(item)).toMatchObject({
      type: "media_reference",
      reference: {
        uri: item.uri,
        title: "fixture result.png",
        mimeType: "image/png",
      },
      metadata: {
        referenceUri: item.uri,
        mimeType: "image/png",
      },
    });
  });
});
