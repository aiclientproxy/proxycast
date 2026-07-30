import { describe, expect, it, vi } from "vitest";
import type {
  AppServerMediaReadParams,
  AppServerMediaReadResponse,
  AppServerJsonRpcNotification,
} from "@/lib/api/appServer";
import {
  buildMediaReadParams,
  createMediaReferenceBinaryPreviewArtifact,
  createMediaReferenceChunkedObjectUrlPreviewArtifact,
  createMediaReferenceObjectUrlPreviewArtifact,
  createMediaReferenceProgressPreviewArtifact,
  createMediaReferencePreviewArtifact,
  emitStreamingMediaReadProgress,
} from "./mediaReferencePreviewArtifacts";
import { createMediaReferencePagedPreviewArtifact } from "./mediaReferencePreviewPagination";

const t = (key: string, options?: Record<string, unknown>) => {
  if (key === "agentChat.mediaReferencePreview.fallbackTitle") {
    return `媒体引用 ${options?.index ?? ""}`;
  }
  if (key === "agentChat.mediaReferencePreview.previewUnavailable") {
    return "该媒体当前以引用形式保存，完整预览需要 media sidecar source 接管。";
  }
  if (key === "agentChat.mediaReferencePreview.reference") {
    return `引用：${options?.value ?? ""}`;
  }
  if (key === "agentChat.mediaReferencePreview.kind") {
    return `类型：${options?.value ?? ""}`;
  }
  if (key === "agentChat.mediaReferencePreview.loading") {
    return "正在读取媒体预览...";
  }
  if (key === "agentChat.mediaReferencePreview.loadingProgress") {
    return `已读取 ${options?.loaded ?? ""} / ${options?.total ?? ""} 字节。`;
  }
  if (key === "agentChat.mediaReferencePreview.mime") {
    return `MIME：${options?.value ?? ""}`;
  }
  if (key === "agentChat.mediaReferencePreview.byteSize") {
    return `大小：${options?.value ?? ""} 字节`;
  }
  if (key === "agentChat.mediaReferencePreview.sha256") {
    return `SHA-256：${options?.value ?? ""}`;
  }
  return key;
};

function encodeBase64(value: string): string {
  return globalThis.btoa(value);
}

function createMediaReadResponse(
  overrides: Partial<AppServerMediaReadResponse> = {},
): AppServerMediaReadResponse {
  return {
    threadId: "thread-media",
    uri: "sidecar://media/image-1",
    mimeType: "image/png",
    bytes: 4,
    totalBytes: 8,
    offset: 0,
    length: 4,
    contentRange: "bytes 0-3/8",
    hasMore: true,
    sha256: "sha256-image-1",
    contentBase64: encodeBase64("ABCD"),
    sidecarRef: {
      ref: "sidecar://media/image-1",
      kind: "media",
      relativePath: "sessions/session-media/media/image-1.png",
    },
    ...overrides,
  };
}

function createMediaReadChunkNotification(params: {
  bytes: number;
  chunkIndex: number;
  contentBase64: string;
  contentRange: string;
  eventId?: string;
  hasMore: boolean;
  offset: number;
  threadId?: string;
  streamId?: string;
  totalBytes: number;
  uri?: string;
}): AppServerJsonRpcNotification {
  const threadId = params.threadId ?? "thread-media";
  return {
    method: "agentSession/event",
    params: {
      event: {
        eventId: params.eventId ?? `evt-media-read-chunk-${params.chunkIndex}`,
        sequence: params.chunkIndex,
        threadId,
        type: "media.read.chunk",
        timestamp: "2026-07-07T00:00:00.000Z",
        payload: {
          streamId: params.streamId ?? "media-read-stream-1",
          chunkIndex: params.chunkIndex,
          done: false,
          chunk: {
            threadId,
            uri: params.uri ?? "sidecar://media/image-1",
            mimeType: "image/png",
            bytes: params.bytes,
            totalBytes: params.totalBytes,
            offset: params.offset,
            length: params.bytes,
            contentRange: params.contentRange,
            hasMore: params.hasMore,
            contentBase64: params.contentBase64,
          },
        },
      },
    },
  };
}

function createMediaReadCompletedNotification(): AppServerJsonRpcNotification {
  return {
    method: "agentSession/event",
    params: {
      event: {
        eventId: "evt-media-read-completed",
        sequence: 3,
        threadId: "thread-media",
        type: "media.read.completed",
        timestamp: "2026-07-07T00:00:01.000Z",
        payload: {
          streamId: "media-read-stream-1",
          chunkCount: 2,
          done: true,
          media: {
            threadId: "thread-media",
            uri: "sidecar://media/image-1",
            mimeType: "image/png",
            bytes: 8,
            totalBytes: 8,
            offset: 0,
            length: 8,
            contentRange: "bytes 0-7/8",
            hasMore: false,
            sha256: "sha256-image-1",
          },
        },
      },
    },
  };
}

describe("createMediaReferencePreviewArtifact", () => {
  it("sidecar media reference 应生成 metadata fallback artifact，不展开媒体 payload", () => {
    const artifact = createMediaReferencePreviewArtifact({
      message: {
        id: "assistant-media",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          caption: "结果图",
          byteSize: 2048,
          sha256: "sha256-image-1",
        },
      },
      t,
    });

    expect(artifact.title).toBe("结果图");
    expect(artifact.content).toContain("media sidecar source");
    expect(artifact.content).toContain("sidecar://media/image-1");
    expect(artifact.meta).toMatchObject({
      openedFrom: "message-media-reference",
      messageId: "assistant-media",
      mediaUri: "sidecar://media/image-1",
      mediaPreviewPolicySchema: "lime.media_reference.preview_policy.v1",
      mediaPreviewPolicy: "sidecar_metadata_fallback",
      mediaPreviewRequiresPagination: false,
      mediaReferenceSoulSchema: "lime.media_reference.soul_surface.v1",
      mediaReferenceSoulStyleLevels: {
        title: "L0",
        referenceFacts: "L0",
        loadingStatus: "L1",
        previewCaption: "L2",
        mediaArtifact: "L3",
      },
      mediaArtifactBoundary: "source_owned_media_payload",
      contentKind: "markdown",
      renderMode: "canvas",
    });
  });

  it("sidecar media reference 可构造 App Server media read params", () => {
    const params = buildMediaReadParams({
      threadId: " thread-media ",
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          sidecarRef: {
            ref: "sidecar://media/image-1",
            kind: "media",
            relativePath: "sessions/session-media/media/image-1.png",
          },
        },
      },
    });

    expect(params).toEqual({
      threadId: "thread-media",
      uri: "sidecar://media/image-1",
      sidecarRef: {
        ref: "sidecar://media/image-1",
        kind: "media",
        relativePath: "sessions/session-media/media/image-1.png",
      },
      maxBytes: 25 * 1024 * 1024,
      offset: 0,
      length: 25 * 1024 * 1024,
    });
  });

  it("已有 source owner 的 media reference 不应重复读取 sidecar bytes", () => {
    const params = buildMediaReadParams({
      threadId: "thread-media",
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          sourcePath: "/tmp/lime-media/image-1.png",
        },
      },
    });

    expect(params).toBeNull();
  });

  it("非 sidecar 展示 URI 可通过 refId 进入 App Server media read", () => {
    const params = buildMediaReadParams({
      threadId: "thread-media",
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "artifact://message/image-1",
          refId: "sidecar://media/image-1",
          mimeType: "image/png",
        },
      },
      maxBytes: 1024,
      offset: 128,
      length: 512,
    });

    expect(params).toEqual({
      threadId: "thread-media",
      refId: "sidecar://media/image-1",
      maxBytes: 1024,
      offset: 128,
      length: 512,
    });
  });

  it("非 sidecar 展示 URI 可通过 sidecar sourceUri 进入 App Server media read", () => {
    const params = buildMediaReadParams({
      threadId: "thread-media",
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "artifact://message/image-1",
          sourceUri: "sidecar://media/image-1",
          mimeType: "image/png",
        },
      },
    });

    expect(params).toEqual({
      threadId: "thread-media",
      uri: "sidecar://media/image-1",
      maxBytes: 25 * 1024 * 1024,
      offset: 0,
      length: 25 * 1024 * 1024,
    });
  });

  it("relativePath-only sidecarRef 可进入 App Server media read", () => {
    const sidecarRef = {
      kind: "media",
      relativePath: "sessions/session-media/media/image-1.png",
      sha256: "sha256-image-1",
      bytes: 8,
      mimeType: "image/png",
    };
    const params = buildMediaReadParams({
      threadId: "thread-media",
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "artifact://message/image-1",
          sidecarRef,
          mimeType: "image/png",
        },
      },
    });

    expect(params).toEqual({
      threadId: "thread-media",
      sidecarRef,
      maxBytes: 25 * 1024 * 1024,
      offset: 0,
      length: 25 * 1024 * 1024,
    });
  });

  it("sidecar bytes 读取结果应生成 media preview artifact", () => {
    const artifact = createMediaReferenceBinaryPreviewArtifact({
      message: {
        id: "assistant-media-sidecar",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          title: "image-1.png",
          sha256: "sha256-image-1",
          byteSize: 2048,
        },
      },
      media: {
        threadId: "thread-media",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        bytes: 8,
        totalBytes: 8,
        offset: 0,
        length: 8,
        contentRange: "bytes 0-7/8",
        hasMore: false,
        sha256: "sha256-image-1",
        contentBase64: "iVBORw0K",
        sidecarRef: {
          ref: "sidecar://media/image-1",
          kind: "media",
          relativePath: "sessions/session-media/media/image-1.png",
        },
      },
      t,
    });

    expect(artifact.title).toBe("image-1.png");
    expect(artifact.content).toBe("data:image/png;base64,iVBORw0K");
    expect(artifact.meta).toMatchObject({
      openedFrom: "message-media-reference",
      mediaUri: "sidecar://media/image-1",
      mediaReadUri: "sidecar://media/image-1",
      mediaPreviewPolicy: "sidecar_read",
      mediaPreviewRequiresPagination: false,
      mediaPreviewSource: "sidecar_read",
      mediaReadOffset: 0,
      mediaReadLength: 8,
      mediaReadTotalBytes: 8,
      mediaReadContentRange: "bytes 0-7/8",
      mediaReadHasMore: false,
      mediaMimeType: "image/png",
      mediaSha256: "sha256-image-1",
      mediaByteSize: 8,
      contentKind: "image",
      renderMode: "media",
    });
    expect(artifact.content).not.toContain("media sidecar source");
  });

  it("sidecar range progress 应生成 loading preview artifact", () => {
    const artifact = createMediaReferenceProgressPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-progress",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          title: "image-1.png",
          sha256: "sha256-image-1",
          byteSize: 8,
        },
      },
      progress: {
        contentRange: "bytes 0-3/8",
        hasMore: true,
        loadedBytes: 4,
        mimeType: "image/png",
        sha256: "sha256-image-1",
        totalBytes: 8,
      },
      t,
    });

    expect(artifact.title).toBe("image-1.png");
    expect(artifact.content).toContain("正在读取媒体预览");
    expect(artifact.content).toContain("已读取 4 / 8 字节");
    expect(artifact.content).toContain("sidecar://media/image-1");
    expect(artifact.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_progress",
      mediaPreviewRequiresPagination: false,
      mediaPreviewSource: "sidecar_progress",
      mediaPreviewStatus: "loading",
      mediaReadContentRange: "bytes 0-3/8",
      mediaReadHasMore: true,
      mediaReadLength: 4,
      mediaReadTotalBytes: 8,
      contentKind: "markdown",
      renderMode: "canvas",
    });
  });

  it("完整 sidecar bytes 读取结果应生成 object URL media preview artifact", () => {
    const createObjectUrl = vi.fn(() => "blob:media-sidecar-preview");
    const artifact = createMediaReferenceObjectUrlPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-object-url",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          title: "image-1.png",
        },
      },
      media: {
        threadId: "thread-media",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        bytes: 8,
        totalBytes: 8,
        offset: 0,
        length: 8,
        contentRange: "bytes 0-7/8",
        hasMore: false,
        sha256: "sha256-image-1",
        contentBase64: "iVBORw0K",
      },
      t,
      createObjectUrl,
    });

    expect(createObjectUrl).toHaveBeenCalledTimes(1);
    expect(artifact?.content).toBe("blob:media-sidecar-preview");
    expect(artifact?.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_object_url",
      mediaPreviewRequiresPagination: false,
      mediaPreviewSource: "sidecar_object_url",
      mediaPreviewObjectUrl: "blob:media-sidecar-preview",
      mediaReadHasMore: false,
      contentKind: "image",
      renderMode: "media",
    });
  });

  it("partial range sidecar 读取结果不应伪装成完整媒体预览", () => {
    const createObjectUrl = vi.fn(() => "blob:partial-media");
    const artifact = createMediaReferenceObjectUrlPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-partial",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          title: "image-1.png",
        },
      },
      media: {
        threadId: "thread-media",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        bytes: 4,
        totalBytes: 8,
        offset: 0,
        length: 4,
        contentRange: "bytes 0-3/8",
        hasMore: true,
        sha256: "sha256-image-1",
        contentBase64: "iVBO",
      },
      t,
      createObjectUrl,
    });

    expect(artifact).toBeNull();
    expect(createObjectUrl).not.toHaveBeenCalled();
  });

  it("sidecar media reference 可按 range window 组装 object URL 预览", async () => {
    const message = {
      id: "assistant-media-sidecar-chunked",
      role: "assistant" as const,
      content: "",
      timestamp: new Date("2026-07-07T00:00:00.000Z"),
    };
    const target = {
      kind: "media_reference" as const,
      index: 0,
      reference: {
        kind: "image",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        title: "image-1.png",
        sha256: "sha256-image-1",
        byteSize: 8,
        sidecarRef: {
          ref: "sidecar://media/image-1",
          kind: "media",
          relativePath: "sessions/session-media/media/image-1.png",
        },
      },
    };
    const responses = [
      createMediaReadResponse(),
      createMediaReadResponse({
        bytes: 4,
        offset: 4,
        length: 4,
        contentRange: "bytes 4-7/8",
        hasMore: false,
        contentBase64: encodeBase64("EFGH"),
      }),
    ];
    const requests: AppServerMediaReadParams[] = [];
    const readMedia = vi.fn(
      async (
        request: AppServerMediaReadParams,
      ): Promise<AppServerMediaReadResponse> => {
        requests.push(request);
        const response = responses[requests.length - 1];
        if (!response) {
          throw new Error("unexpected extra media read");
        }
        return response;
      },
    );
    const createObjectUrl = vi.fn((blob: Blob) => {
      expect(blob.size).toBe(8);
      expect(blob.type).toBe("image/png");
      return "blob:chunked-media";
    });
    const onProgress = vi.fn();

    const artifact = await createMediaReferenceChunkedObjectUrlPreviewArtifact({
      message,
      target,
      threadId: " thread-media ",
      t,
      readMedia,
      createObjectUrl,
      onProgress,
      chunkBytes: 4,
      maxBytes: 16,
    });

    expect(readMedia).toHaveBeenCalledTimes(2);
    expect(requests).toEqual([
      {
        threadId: "thread-media",
        uri: "sidecar://media/image-1",
        sidecarRef: target.reference.sidecarRef,
        maxBytes: 4,
        offset: 0,
        length: 4,
        stream: true,
      },
      {
        threadId: "thread-media",
        uri: "sidecar://media/image-1",
        sidecarRef: target.reference.sidecarRef,
        maxBytes: 4,
        offset: 4,
        length: 4,
        stream: true,
      },
    ]);
    expect(createObjectUrl).toHaveBeenCalledTimes(1);
    expect(onProgress).toHaveBeenCalledTimes(1);
    expect(onProgress).toHaveBeenCalledWith({
      contentRange: "bytes 0-3/8",
      hasMore: true,
      loadedBytes: 4,
      mimeType: "image/png",
      sha256: "sha256-image-1",
      totalBytes: 8,
    });
    expect(artifact?.content).toBe("blob:chunked-media");
    expect(artifact?.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_object_url",
      mediaPreviewMaxBytes: 16,
      mediaPreviewChunkBytes: 4,
      mediaPreviewRequiresPagination: false,
      mediaPreviewSource: "sidecar_object_url",
      mediaPreviewObjectUrl: "blob:chunked-media",
      mediaReadOffset: 0,
      mediaReadLength: 8,
      mediaReadTotalBytes: 8,
      mediaReadContentRange: "bytes 0-7/8",
      mediaReadHasMore: false,
      mediaByteSize: 8,
      contentKind: "image",
      renderMode: "media",
    });
  });

  it("stream=true media.read.chunk notification 应驱动 progress artifact", async () => {
    const target = {
      kind: "media_reference" as const,
      index: 0,
      reference: {
        kind: "image",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        title: "image-1.png",
        sidecarRef: {
          ref: "sidecar://media/image-1",
          kind: "media",
          relativePath: "sessions/session-media/media/image-1.png",
        },
      },
    };
    const responses = [
      {
        media: createMediaReadResponse(),
        notifications: [
          createMediaReadChunkNotification({
            bytes: 4,
            chunkIndex: 1,
            contentBase64: encodeBase64("ABCD"),
            contentRange: "bytes 0-3/8",
            hasMore: true,
            offset: 0,
            totalBytes: 8,
          }),
        ],
      },
      {
        media: createMediaReadResponse({
          bytes: 4,
          offset: 4,
          length: 4,
          contentRange: "bytes 4-7/8",
          hasMore: false,
          contentBase64: encodeBase64("EFGH"),
        }),
        notifications: [
          createMediaReadChunkNotification({
            bytes: 4,
            chunkIndex: 2,
            contentBase64: encodeBase64("EFGH"),
            contentRange: "bytes 4-7/8",
            hasMore: false,
            offset: 4,
            totalBytes: 8,
          }),
          createMediaReadCompletedNotification(),
        ],
      },
    ];
    const readMedia = vi.fn(async () => {
      const response = responses.shift();
      if (!response) {
        throw new Error("unexpected extra media read");
      }
      return response;
    });
    const onProgress = vi.fn();

    const artifact = await createMediaReferenceChunkedObjectUrlPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-stream",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target,
      threadId: "thread-media",
      t,
      readMedia,
      createObjectUrl: vi.fn(() => "blob:streamed-media"),
      onProgress,
      chunkBytes: 4,
      maxBytes: 16,
    });

    expect(readMedia).toHaveBeenCalledTimes(2);
    expect(readMedia.mock.calls[0]?.[0]).toMatchObject({ stream: true });
    expect(readMedia.mock.calls[1]?.[0]).toMatchObject({ stream: true });
    expect(onProgress).toHaveBeenCalledTimes(1);
    expect(onProgress).toHaveBeenCalledWith({
      contentRange: "bytes 0-3/8",
      hasMore: true,
      loadedBytes: 4,
      mimeType: "image/png",
      sha256: undefined,
      totalBytes: 8,
    });
    expect(artifact?.content).toBe("blob:streamed-media");
  });

  it("media.read.chunk progress 应按 threadId / streamId / uri / offset fail closed", () => {
    const onProgress = vi.fn();
    const seenEventIds = new Set<string>();

    expect(
      emitStreamingMediaReadProgress({
        expectedOffset: 0,
        expectedStreamId: "media-read-stream-expected",
        expectedUri: "sidecar://media/image-1",
        notifications: [
          createMediaReadChunkNotification({
            bytes: 4,
            chunkIndex: 1,
            contentBase64: encodeBase64("ABCD"),
            contentRange: "bytes 0-3/8",
            hasMore: true,
            offset: 0,
            streamId: "media-read-stream-other",
            totalBytes: 8,
          }),
          createMediaReadChunkNotification({
            bytes: 4,
            chunkIndex: 2,
            contentBase64: encodeBase64("ABCD"),
            contentRange: "bytes 0-3/8",
            hasMore: true,
            offset: 0,
            threadId: "session-other",
            streamId: "media-read-stream-expected",
            totalBytes: 8,
          }),
          createMediaReadChunkNotification({
            bytes: 4,
            chunkIndex: 3,
            contentBase64: encodeBase64("ABCD"),
            contentRange: "bytes 4-7/8",
            hasMore: true,
            offset: 4,
            streamId: "media-read-stream-expected",
            totalBytes: 8,
          }),
          createMediaReadChunkNotification({
            bytes: 4,
            chunkIndex: 4,
            contentBase64: encodeBase64("ABCD"),
            contentRange: "bytes 0-3/8",
            hasMore: true,
            offset: 0,
            streamId: "media-read-stream-expected",
            totalBytes: 8,
            uri: "sidecar://media/image-other",
          }),
        ],
        onProgress,
        seenEventIds,
        threadId: "thread-media",
      }).emitted,
    ).toBe(false);
    expect(onProgress).not.toHaveBeenCalled();

    const firstResult = emitStreamingMediaReadProgress({
      expectedOffset: 0,
      expectedUri: "sidecar://media/image-1",
      notifications: [
        createMediaReadChunkNotification({
          bytes: 4,
          chunkIndex: 5,
          contentBase64: encodeBase64("ABCD"),
          contentRange: "bytes 0-3/8",
          eventId: "evt-media-live-1",
          hasMore: true,
          offset: 0,
          streamId: "media-read-stream-bound",
          totalBytes: 8,
        }),
      ],
      onProgress,
      seenEventIds,
      threadId: "thread-media",
    });

    expect(firstResult).toEqual({
      emitted: true,
      streamId: "media-read-stream-bound",
    });
    expect(onProgress).toHaveBeenCalledTimes(1);

    const duplicateResult = emitStreamingMediaReadProgress({
      expectedStreamId: firstResult.streamId,
      notifications: [
        createMediaReadChunkNotification({
          bytes: 4,
          chunkIndex: 5,
          contentBase64: encodeBase64("ABCD"),
          contentRange: "bytes 0-3/8",
          eventId: "evt-media-live-1",
          hasMore: true,
          offset: 0,
          streamId: "media-read-stream-bound",
          totalBytes: 8,
        }),
      ],
      onProgress,
      seenEventIds,
      threadId: "thread-media",
    });

    expect(duplicateResult).toEqual({
      emitted: false,
      streamId: "media-read-stream-bound",
    });
    expect(onProgress).toHaveBeenCalledTimes(1);
  });

  it("range window 不连续时应放弃 chunked object URL 预览", async () => {
    const target = {
      kind: "media_reference" as const,
      index: 0,
      reference: {
        kind: "image",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        sidecarRef: {
          ref: "sidecar://media/image-1",
          kind: "media",
          relativePath: "sessions/session-media/media/image-1.png",
        },
      },
    };
    const responses = [
      createMediaReadResponse(),
      createMediaReadResponse({
        bytes: 3,
        offset: 5,
        length: 3,
        contentRange: "bytes 5-7/8",
        hasMore: false,
        contentBase64: encodeBase64("FGH"),
      }),
    ];
    const readMedia = vi.fn(
      async (): Promise<AppServerMediaReadResponse> =>
        responses.shift() ?? createMediaReadResponse(),
    );
    const createObjectUrl = vi.fn(() => "blob:invalid-gap");

    const artifact = await createMediaReferenceChunkedObjectUrlPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-gap",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target,
      threadId: "thread-media",
      t,
      readMedia,
      createObjectUrl,
      chunkBytes: 4,
      maxBytes: 16,
    });

    expect(artifact).toBeNull();
    expect(readMedia).toHaveBeenCalledTimes(2);
    expect(createObjectUrl).not.toHaveBeenCalled();
  });

  it("shouldContinue 停止后不应继续组装 chunked object URL 预览", async () => {
    const target = {
      kind: "media_reference" as const,
      index: 0,
      reference: {
        kind: "image",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        sidecarRef: {
          ref: "sidecar://media/image-1",
          kind: "media",
          relativePath: "sessions/session-media/media/image-1.png",
        },
      },
    };
    let canContinue = true;
    const readMedia = vi.fn(async () => {
      canContinue = false;
      return createMediaReadResponse();
    });
    const createObjectUrl = vi.fn(() => "blob:cancelled-media");

    const artifact = await createMediaReferenceChunkedObjectUrlPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-cancelled",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target,
      threadId: "thread-media",
      t,
      readMedia,
      createObjectUrl,
      shouldContinue: () => canContinue,
      chunkBytes: 4,
      maxBytes: 16,
    });

    expect(artifact).toBeNull();
    expect(readMedia).toHaveBeenCalledTimes(1);
    expect(createObjectUrl).not.toHaveBeenCalled();
  });

  it("range window sha256 不一致时应放弃 chunked object URL 预览", async () => {
    const target = {
      kind: "media_reference" as const,
      index: 0,
      reference: {
        kind: "image",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        sidecarRef: {
          ref: "sidecar://media/image-1",
          kind: "media",
          relativePath: "sessions/session-media/media/image-1.png",
        },
      },
    };
    const responses = [
      createMediaReadResponse(),
      createMediaReadResponse({
        bytes: 4,
        offset: 4,
        length: 4,
        contentRange: "bytes 4-7/8",
        hasMore: false,
        sha256: "sha256-other",
        contentBase64: encodeBase64("EFGH"),
      }),
    ];
    const readMedia = vi.fn(
      async (): Promise<AppServerMediaReadResponse> =>
        responses.shift() ?? createMediaReadResponse(),
    );
    const createObjectUrl = vi.fn(() => "blob:invalid-sha");

    const artifact = await createMediaReferenceChunkedObjectUrlPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-sha",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target,
      threadId: "thread-media",
      t,
      readMedia,
      createObjectUrl,
      chunkBytes: 4,
      maxBytes: 16,
    });

    expect(artifact).toBeNull();
    expect(readMedia).toHaveBeenCalledTimes(2);
    expect(createObjectUrl).not.toHaveBeenCalled();
  });

  it("media 总大小超过前端预览上限时不继续分片读取", async () => {
    const target = {
      kind: "media_reference" as const,
      index: 0,
      reference: {
        kind: "image",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        sidecarRef: {
          ref: "sidecar://media/image-1",
          kind: "media",
          relativePath: "sessions/session-media/media/image-1.png",
        },
      },
    };
    const readMedia = vi.fn(async () =>
      createMediaReadResponse({
        totalBytes: 12,
        contentRange: "bytes 0-3/12",
      }),
    );
    const createObjectUrl = vi.fn(() => "blob:too-large");

    const artifact = await createMediaReferenceChunkedObjectUrlPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-too-large",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target,
      threadId: "thread-media",
      t,
      readMedia,
      createObjectUrl,
      chunkBytes: 4,
      maxBytes: 8,
    });

    expect(artifact?.content).toContain("media sidecar source");
    expect(artifact?.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_preview_budget_exceeded",
      mediaPreviewMaxBytes: 8,
      mediaPreviewChunkBytes: 4,
      mediaPreviewRequiresPagination: true,
      mediaPreviewLoadedBytes: 4,
      mediaPreviewNextOffset: 4,
      mediaPreviewTotalBytes: 12,
      mediaReadContentRange: "bytes 0-3/12",
      mediaReadHasMore: true,
      mediaReadLength: 4,
      mediaReadTotalBytes: 12,
      contentKind: "markdown",
      renderMode: "canvas",
    });
    expect(readMedia).toHaveBeenCalledTimes(1);
    expect(createObjectUrl).not.toHaveBeenCalled();
  });

  it("可按 offset 读取大型媒体 page window 并生成分页 metadata artifact", async () => {
    const target = {
      kind: "media_reference" as const,
      index: 0,
      reference: {
        kind: "image",
        uri: "sidecar://media/image-1",
        mimeType: "image/png",
        sidecarRef: {
          ref: "sidecar://media/image-1",
          kind: "media",
          relativePath: "sessions/session-media/media/image-1.png",
        },
      },
    };
    const requests: AppServerMediaReadParams[] = [];
    const readMedia = vi.fn(async (request) => {
      requests.push(request);
      return createMediaReadResponse({
        bytes: 4,
        offset: 4,
        length: 4,
        totalBytes: 12,
        contentRange: "bytes 4-7/12",
        hasMore: true,
        contentBase64: encodeBase64("EFGH"),
      });
    });

    const artifact = await createMediaReferencePagedPreviewArtifact({
      message: {
        id: "assistant-media-sidecar-page",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target,
      threadId: "thread-media",
      t,
      readMedia,
      offset: 4,
      length: 4,
      maxBytes: 8,
    });

    expect(readMedia).toHaveBeenCalledTimes(1);
    expect(requests).toEqual([
      {
        threadId: "thread-media",
        uri: "sidecar://media/image-1",
        sidecarRef: target.reference.sidecarRef,
        maxBytes: 4,
        offset: 4,
        length: 4,
      },
    ]);
    expect(artifact?.content).toContain("media sidecar source");
    expect(artifact?.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_page_window",
      mediaPreviewMaxBytes: 8,
      mediaPreviewChunkBytes: 4,
      mediaPreviewRequiresPagination: true,
      mediaPreviewLoadedBytes: 8,
      mediaPreviewPreviousOffset: 0,
      mediaPreviewNextOffset: 8,
      mediaPreviewTotalBytes: 12,
      mediaPreviewPageOffset: 4,
      mediaPreviewPageLength: 4,
      mediaPreviewPageIndex: 2,
      mediaPreviewCanReadNextPage: true,
      mediaPreviewCanReadPreviousPage: true,
      mediaReadContentRange: "bytes 4-7/12",
      mediaReadHasMore: true,
      mediaReadLength: 8,
      mediaReadTotalBytes: 12,
      contentKind: "markdown",
      renderMode: "canvas",
    });
  });

  it("带 sourcePath owner 的 sidecar media reference 应生成 media artifact", () => {
    const artifact = createMediaReferencePreviewArtifact({
      message: {
        id: "assistant-media-source",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          title: "image-1.png",
          sourcePath: "/tmp/lime-media/image-1.png",
        },
      },
      t,
    });

    expect(artifact.title).toBe("image-1.png");
    expect(artifact.content).not.toContain("media sidecar source");
    expect(artifact.meta).toMatchObject({
      openedFrom: "message-media-reference",
      mediaUri: "sidecar://media/image-1",
      mediaSourcePath: "/tmp/lime-media/image-1.png",
      mediaPreviewPolicy: "source_path_owner",
      mediaPreviewSource: "source_path",
      sourcePath: "/tmp/lime-media/image-1.png",
      contentKind: "image",
      renderMode: "media",
    });
  });

  it("inline data previewUrl 应 fail closed 到 metadata fallback", () => {
    const artifact = createMediaReferencePreviewArtifact({
      message: {
        id: "assistant-media-inline-preview",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target: {
        kind: "media_reference",
        index: 0,
        reference: {
          kind: "image",
          uri: "sidecar://media/image-1",
          mimeType: "image/png",
          title: "image-1.png",
          previewUrl: "data:image/png;base64,AAAA",
        },
      },
      t,
    });

    expect(artifact.content).toContain("media sidecar source");
    expect(artifact.content).toContain("sidecar://media/image-1");
    expect(artifact.content).not.toContain("data:image");
    expect(artifact.meta).toMatchObject({
      contentKind: "markdown",
      renderMode: "canvas",
    });
    expect(artifact.meta.mediaPreviewUrl).toBeUndefined();
  });

  it("可直接预览的媒体 URI 应生成 media artifact", () => {
    const artifact = createMediaReferencePreviewArtifact({
      message: {
        id: "assistant-media-direct",
        role: "assistant",
        content: "",
        timestamp: new Date("2026-07-07T00:00:00.000Z"),
      },
      target: {
        kind: "media_reference",
        index: 1,
        reference: {
          kind: "image",
          uri: "https://example.com/image.png",
          mimeType: "image/png",
          title: "image.png",
        },
      },
      t,
    });

    expect(artifact.title).toBe("image.png");
    expect(artifact.content).toBe("https://example.com/image.png");
    expect(artifact.meta).toMatchObject({
      openedFrom: "message-media-reference",
      contentKind: "image",
      renderMode: "media",
      mediaPreviewPolicy: "direct_owner",
      previewUrl: "https://example.com/image.png",
    });
  });
});
