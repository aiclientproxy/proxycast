import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createAppServerClient } from "@/lib/api/appServer";
import type { AppServerMediaReadResponse } from "@/lib/api/appServer";
import { logAgentDebug } from "@/lib/agentDebug";
import type { Artifact } from "@/lib/artifact/types";
import type { Message, MessagePreviewTarget } from "../types";
import {
  WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY,
  useWorkspaceMediaReferencePreviewRuntime,
  type WorkspaceMediaReferencePreviewRuntime,
} from "./useWorkspaceMediaReferencePreviewRuntime";

vi.mock("@/lib/api/appServer", () => ({
  createAppServerClient: vi.fn(),
}));

vi.mock("@/lib/agentDebug", () => ({
  logAgentDebug: vi.fn(),
}));

type HookProps = Parameters<typeof useWorkspaceMediaReferencePreviewRuntime>[0];

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];
const restoreObjectUrlMocks: Array<() => void> = [];

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

function createMessage(id = "assistant-media"): Message {
  return {
    id,
    role: "assistant",
    content: "",
    timestamp: new Date("2026-07-07T00:00:00.000Z"),
  };
}

function createDirectMediaTarget(): Extract<
  MessagePreviewTarget,
  { kind: "media_reference" }
> {
  return {
    kind: "media_reference",
    index: 0,
    reference: {
      kind: "image",
      uri: "https://example.com/image.png",
      mimeType: "image/png",
      title: "image.png",
    },
  };
}

function createSidecarMediaTarget(
  overrides: Partial<
    Extract<MessagePreviewTarget, { kind: "media_reference" }>["reference"]
  > = {},
): Extract<MessagePreviewTarget, { kind: "media_reference" }> {
  const uri = overrides.uri?.trim() || "sidecar://media/image-1";
  const title = overrides.title?.trim() || "image-1.png";
  return {
    kind: "media_reference",
    index: 0,
    reference: {
      kind: "image",
      uri,
      mimeType: "image/png",
      title,
      sha256: "sha256-image-1",
      byteSize: 3,
      sidecarRef: {
        ref: uri,
        kind: "media",
        relativePath: `sessions/session-media/media/${title}`,
      },
      ...overrides,
    },
  };
}

function createMediaReadResponse(
  overrides: Partial<AppServerMediaReadResponse> = {},
): AppServerMediaReadResponse {
  return {
    threadId: "thread-media",
    uri: "sidecar://media/image-1",
    mimeType: "image/png",
    bytes: 3,
    totalBytes: 3,
    offset: 0,
    length: 3,
    contentRange: "bytes 0-2/3",
    hasMore: false,
    sha256: "sha256-image-1",
    contentBase64: globalThis.btoa("abc"),
    sidecarRef: createSidecarMediaTarget().reference.sidecarRef,
    ...overrides,
  };
}

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

function renderHook(props?: Partial<HookProps>) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: WorkspaceMediaReferencePreviewRuntime | null = null;

  const defaultProps: HookProps = {
    artifacts: [],
    handleWorkspaceArtifactClick: vi.fn(),
    requestCanvasWorkbenchPreviewOpen: vi.fn(),
    threadId: "thread-media",
    setCanvasWorkbenchLayoutMode: vi.fn(),
    setLayoutMode: vi.fn(),
    t,
    upsertGeneralArtifact: vi.fn(),
  };

  function Probe(currentProps: HookProps) {
    latestValue = useWorkspaceMediaReferencePreviewRuntime(currentProps);
    return null;
  }

  const render = async (nextProps?: Partial<HookProps>) => {
    await act(async () => {
      root.render(<Probe {...defaultProps} {...props} {...nextProps} />);
      await Promise.resolve();
    });
  };

  mountedRoots.push({ container, root });

  return {
    render,
    getValue: () => {
      if (!latestValue) {
        throw new Error("hook 尚未初始化");
      }
      return latestValue;
    },
  };
}

function installObjectUrlMocks() {
  const originalCreateObjectUrl = Object.getOwnPropertyDescriptor(
    URL,
    "createObjectURL",
  );
  const originalRevokeObjectUrl = Object.getOwnPropertyDescriptor(
    URL,
    "revokeObjectURL",
  );
  let objectUrlIndex = 0;
  const createObjectUrl = vi.fn().mockImplementation(() => {
    objectUrlIndex += 1;
    return `blob:media-${objectUrlIndex}`;
  });
  const revokeObjectUrl = vi.fn();
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: createObjectUrl,
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    value: revokeObjectUrl,
  });

  const restore = () => {
    if (originalCreateObjectUrl) {
      Object.defineProperty(URL, "createObjectURL", originalCreateObjectUrl);
    } else {
      Reflect.deleteProperty(URL, "createObjectURL");
    }
    if (originalRevokeObjectUrl) {
      Object.defineProperty(URL, "revokeObjectURL", originalRevokeObjectUrl);
    } else {
      Reflect.deleteProperty(URL, "revokeObjectURL");
    }
  };
  restoreObjectUrlMocks.push(restore);

  return {
    createObjectUrl,
    restore,
    revokeObjectUrl,
  };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  vi.mocked(createAppServerClient).mockReset();
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
  while (restoreObjectUrlMocks.length > 0) {
    restoreObjectUrlMocks.pop()?.();
  }
  vi.restoreAllMocks();
});

describe("useWorkspaceMediaReferencePreviewRuntime", () => {
  it("直接媒体 URL 预览不应实例化 App Server client", async () => {
    const upsertGeneralArtifact = vi.fn();
    const handleWorkspaceArtifactClick = vi.fn();
    const requestCanvasWorkbenchPreviewOpen = vi.fn();
    const setLayoutMode = vi.fn();
    const setCanvasWorkbenchLayoutMode = vi.fn();
    const { render, getValue } = renderHook({
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      setCanvasWorkbenchLayoutMode,
      setLayoutMode,
      upsertGeneralArtifact,
    });

    await render();
    await act(async () => {
      await getValue().openMediaReferencePreview(
        createDirectMediaTarget(),
        createMessage("assistant-direct-media"),
      );
    });

    expect(createAppServerClient).not.toHaveBeenCalled();
    expect(setLayoutMode).toHaveBeenCalledWith("chat-canvas");
    expect(setCanvasWorkbenchLayoutMode).toHaveBeenCalledWith("split");
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact.meta).toMatchObject({
      mediaPreviewSource: "direct_uri",
      mediaUri: "https://example.com/image.png",
      mediaMimeType: "image/png",
      contentKind: "image",
      renderMode: "media",
    });
    expect(handleWorkspaceArtifactClick).toHaveBeenCalledWith(artifact);
    expect(requestCanvasWorkbenchPreviewOpen).toHaveBeenCalledWith({
      filePath: artifact.meta.filePath,
      selectionKey: `artifact:${artifact.id}`,
    });
  });

  it("sidecar 媒体预览应创建 object URL，并在替换和卸载时释放", async () => {
    const upsertGeneralArtifact = vi.fn();
    const readMedia = vi
      .fn()
      .mockResolvedValueOnce({ result: createMediaReadResponse() })
      .mockResolvedValueOnce({ result: createMediaReadResponse() });
    vi.mocked(createAppServerClient).mockReturnValue({
      readMedia,
    } as ReturnType<typeof createAppServerClient>);
    const { createObjectUrl, revokeObjectUrl } = installObjectUrlMocks();
    const { render, getValue } = renderHook({
      upsertGeneralArtifact,
    });

    await render();
    await act(async () => {
      await getValue().openMediaReferencePreview(
        createSidecarMediaTarget(),
        createMessage("assistant-sidecar-media"),
      );
    });
    const firstArtifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;

    await render({ artifacts: [firstArtifact] });
    await act(async () => {
      await getValue().openMediaReferencePreview(
        createSidecarMediaTarget(),
        createMessage("assistant-sidecar-media"),
      );
    });

    expect(createAppServerClient).toHaveBeenCalledTimes(2);
    expect(readMedia).toHaveBeenCalledTimes(2);
    expect(createObjectUrl).toHaveBeenCalledTimes(2);
    const secondArtifact = upsertGeneralArtifact.mock.calls[1]?.[0] as Artifact;
    expect(firstArtifact.id).toBe(secondArtifact.id);
    expect(firstArtifact.meta.mediaPreviewObjectUrl).toBe("blob:media-1");
    expect(secondArtifact.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_object_url",
      mediaPreviewRequiresPagination: false,
      mediaPreviewObjectUrl: "blob:media-2",
      mediaPreviewSource: "sidecar_object_url",
      mediaReadContentRange: "bytes 0-2/3",
      mediaReadTotalBytes: 3,
      contentKind: "image",
      renderMode: "media",
    });
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:media-1");

    act(() => {
      const mounted = mountedRoots.pop();
      if (!mounted) {
        throw new Error("缺少已挂载 hook");
      }
      mounted.root.unmount();
      mounted.container.remove();
    });

    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:media-2");
  });

  it("sidecar 读取被权限拒绝时应保留可见 metadata fallback", async () => {
    const upsertGeneralArtifact = vi.fn();
    const handleWorkspaceArtifactClick = vi.fn();
    const requestCanvasWorkbenchPreviewOpen = vi.fn();
    const readMedia = vi
      .fn()
      .mockRejectedValue(new Error("permission denied: private sidecar path"));
    vi.mocked(createAppServerClient).mockReturnValue({
      readMedia,
    } as ReturnType<typeof createAppServerClient>);
    const { createObjectUrl } = installObjectUrlMocks();
    const { render, getValue } = renderHook({
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      upsertGeneralArtifact,
    });

    await render();
    await act(async () => {
      await getValue().openMediaReferencePreview(
        createSidecarMediaTarget(),
        createMessage("assistant-sidecar-media-permission-denied"),
      );
    });

    expect(readMedia).toHaveBeenCalledOnce();
    expect(createObjectUrl).not.toHaveBeenCalled();
    expect(upsertGeneralArtifact).toHaveBeenCalledOnce();
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact.content).toContain("media sidecar source");
    expect(artifact.content).not.toContain("permission denied");
    expect(artifact.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_metadata_fallback",
      contentKind: "markdown",
      renderMode: "canvas",
    });
    expect(handleWorkspaceArtifactClick).toHaveBeenCalledWith(artifact);
    expect(requestCanvasWorkbenchPreviewOpen).toHaveBeenCalledWith({
      filePath: artifact.meta.filePath,
      selectionKey: `artifact:${artifact.id}`,
    });
  });

  it("chunked sidecar 媒体预览应先写入 progress artifact，再替换为 object URL", async () => {
    const upsertGeneralArtifact = vi.fn();
    const handleWorkspaceArtifactClick = vi.fn();
    const requestCanvasWorkbenchPreviewOpen = vi.fn();
    const readMedia = vi
      .fn()
      .mockResolvedValueOnce({
        result: createMediaReadResponse({
          bytes: 2,
          totalBytes: 4,
          offset: 0,
          length: 2,
          contentRange: "bytes 0-1/4",
          hasMore: true,
          contentBase64: globalThis.btoa("ab"),
        }),
      })
      .mockResolvedValueOnce({
        result: createMediaReadResponse({
          bytes: 2,
          totalBytes: 4,
          offset: 2,
          length: 2,
          contentRange: "bytes 2-3/4",
          hasMore: false,
          contentBase64: globalThis.btoa("cd"),
        }),
      });
    vi.mocked(createAppServerClient).mockReturnValue({
      readMedia,
    } as ReturnType<typeof createAppServerClient>);
    const { createObjectUrl } = installObjectUrlMocks();
    const { render, getValue } = renderHook({
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      upsertGeneralArtifact,
    });

    await render();
    await act(async () => {
      await getValue().openMediaReferencePreview(
        createSidecarMediaTarget({ byteSize: 4 }),
        createMessage("assistant-sidecar-media-progress"),
      );
    });

    expect(readMedia).toHaveBeenCalledTimes(2);
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(2);
    const progressArtifact = upsertGeneralArtifact.mock
      .calls[0]?.[0] as Artifact;
    const finalArtifact = upsertGeneralArtifact.mock.calls[1]?.[0] as Artifact;
    expect(progressArtifact.id).toBe(finalArtifact.id);
    expect(progressArtifact.content).toContain("正在读取媒体预览");
    expect(progressArtifact.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_progress",
      mediaPreviewRequiresPagination: false,
      mediaPreviewSource: "sidecar_progress",
      mediaPreviewStatus: "loading",
      mediaReadContentRange: "bytes 0-1/4",
      mediaReadHasMore: true,
      mediaReadLength: 2,
      mediaReadTotalBytes: 4,
      contentKind: "markdown",
      renderMode: "canvas",
    });
    expect(finalArtifact.content).toBe("blob:media-1");
    expect(finalArtifact.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_object_url",
      mediaPreviewRequiresPagination: false,
      mediaPreviewObjectUrl: "blob:media-1",
      mediaPreviewSource: "sidecar_object_url",
      mediaReadContentRange: "bytes 0-3/4",
      mediaReadTotalBytes: 4,
      contentKind: "image",
      renderMode: "media",
    });
    expect(createObjectUrl).toHaveBeenCalledTimes(1);
    expect(handleWorkspaceArtifactClick).toHaveBeenCalledTimes(2);
    expect(requestCanvasWorkbenchPreviewOpen).toHaveBeenCalledTimes(2);
  });

  it("超过前端预览预算的大媒体应打开带分页策略的 fallback artifact", async () => {
    const upsertGeneralArtifact = vi.fn();
    const handleWorkspaceArtifactClick = vi.fn();
    const requestCanvasWorkbenchPreviewOpen = vi.fn();
    const readMedia = vi.fn().mockResolvedValue({
      result: createMediaReadResponse({
        bytes: 3,
        totalBytes:
          WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY.objectUrlMaxBytes +
          1,
        contentRange: `bytes 0-2/${
          WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY.objectUrlMaxBytes + 1
        }`,
        hasMore: true,
      }),
    });
    vi.mocked(createAppServerClient).mockReturnValue({
      readMedia,
    } as ReturnType<typeof createAppServerClient>);
    const { createObjectUrl } = installObjectUrlMocks();
    const { render, getValue } = renderHook({
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      upsertGeneralArtifact,
    });

    await render();
    await act(async () => {
      await getValue().openMediaReferencePreview(
        createSidecarMediaTarget({
          byteSize:
            WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY.objectUrlMaxBytes +
            1,
        }),
        createMessage("assistant-sidecar-media-too-large"),
      );
    });

    expect(readMedia).toHaveBeenCalledTimes(1);
    expect(createObjectUrl).not.toHaveBeenCalled();
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact.content).toContain("media sidecar source");
    expect(artifact.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_preview_budget_exceeded",
      mediaPreviewRequiresPagination: true,
      mediaPreviewLoadedBytes: 3,
      mediaPreviewNextOffset: 3,
      mediaPreviewTotalBytes:
        WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY.objectUrlMaxBytes + 1,
      mediaReadHasMore: true,
      mediaReadLength: 3,
      contentKind: "markdown",
      renderMode: "canvas",
    });
    expect(handleWorkspaceArtifactClick).toHaveBeenCalledWith(artifact);
    expect(requestCanvasWorkbenchPreviewOpen).toHaveBeenCalledWith({
      filePath: artifact.meta.filePath,
      selectionKey: `artifact:${artifact.id}`,
    });
  });

  it("可打开大型媒体指定 page window artifact", async () => {
    const upsertGeneralArtifact = vi.fn();
    const handleWorkspaceArtifactClick = vi.fn();
    const requestCanvasWorkbenchPreviewOpen = vi.fn();
    const readMedia = vi.fn().mockResolvedValue({
      result: createMediaReadResponse({
        bytes: 3,
        offset: 3,
        length: 3,
        totalBytes: 9,
        contentRange: "bytes 3-5/9",
        hasMore: true,
      }),
    });
    vi.mocked(createAppServerClient).mockReturnValue({
      readMedia,
    } as ReturnType<typeof createAppServerClient>);
    const { createObjectUrl } = installObjectUrlMocks();
    const { render, getValue } = renderHook({
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      upsertGeneralArtifact,
    });

    await render();
    await act(async () => {
      await getValue().openMediaReferencePreviewPage(
        createSidecarMediaTarget({ byteSize: 9 }),
        createMessage("assistant-sidecar-media-page"),
        { offset: 3, length: 3 },
      );
    });

    expect(readMedia).toHaveBeenCalledTimes(1);
    expect(readMedia).toHaveBeenCalledWith(
      expect.objectContaining({
        length: 3,
        maxBytes: 3,
        offset: 3,
        threadId: "thread-media",
        uri: "sidecar://media/image-1",
      }),
      expect.objectContaining({
        signal: expect.any(AbortSignal),
      }),
    );
    expect(createObjectUrl).not.toHaveBeenCalled();
    const artifact = upsertGeneralArtifact.mock.calls[0]?.[0] as Artifact;
    expect(artifact.meta).toMatchObject({
      mediaPreviewPolicy: "sidecar_page_window",
      mediaPreviewRequiresPagination: true,
      mediaPreviewLoadedBytes: 6,
      mediaPreviewPreviousOffset: 0,
      mediaPreviewNextOffset: 6,
      mediaPreviewTotalBytes: 9,
      mediaPreviewPageOffset: 3,
      mediaPreviewPageLength: 3,
      mediaPreviewPageIndex: 2,
      mediaPreviewCanReadNextPage: true,
      mediaPreviewCanReadPreviousPage: true,
      mediaReadContentRange: "bytes 3-5/9",
      mediaReadHasMore: true,
      mediaReadLength: 6,
      contentKind: "markdown",
      renderMode: "canvas",
    });
    expect(handleWorkspaceArtifactClick).toHaveBeenCalledWith(artifact);
    expect(requestCanvasWorkbenchPreviewOpen).toHaveBeenCalledWith({
      filePath: artifact.meta.filePath,
      selectionKey: `artifact:${artifact.id}`,
    });
  });

  it("后发 media preview 请求应压过先发慢请求，迟到结果不写回 UI", async () => {
    const upsertGeneralArtifact = vi.fn();
    const handleWorkspaceArtifactClick = vi.fn();
    const requestCanvasWorkbenchPreviewOpen = vi.fn();
    const firstRead = createDeferred<{
      result: AppServerMediaReadResponse;
    }>();
    const readMedia = vi
      .fn()
      .mockReturnValueOnce(firstRead.promise)
      .mockResolvedValueOnce({
        result: createMediaReadResponse({
          uri: "sidecar://media/image-2",
          sidecarRef: {
            ref: "sidecar://media/image-2",
            kind: "media",
            relativePath: "sessions/session-media/media/image-2.png",
          },
        }),
      });
    vi.mocked(createAppServerClient).mockReturnValue({
      readMedia,
    } as ReturnType<typeof createAppServerClient>);
    const { createObjectUrl } = installObjectUrlMocks();
    const { render, getValue } = renderHook({
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      upsertGeneralArtifact,
    });

    await render();
    const firstPreview = getValue().openMediaReferencePreview(
      createSidecarMediaTarget(),
      createMessage("assistant-sidecar-media-first"),
    );
    await Promise.resolve();
    await act(async () => {
      await getValue().openMediaReferencePreview(
        createSidecarMediaTarget({
          uri: "sidecar://media/image-2",
          title: "image-2.png",
          sidecarRef: {
            ref: "sidecar://media/image-2",
            kind: "media",
            relativePath: "sessions/session-media/media/image-2.png",
          },
        }),
        createMessage("assistant-sidecar-media-second"),
      );
    });

    const firstReadOptions = readMedia.mock.calls[0]?.[1] as
      | { signal?: AbortSignal }
      | undefined;
    const secondReadOptions = readMedia.mock.calls[1]?.[1] as
      | { signal?: AbortSignal }
      | undefined;
    expect(firstReadOptions?.signal).toBeInstanceOf(AbortSignal);
    expect(firstReadOptions?.signal?.aborted).toBe(true);
    expect(firstReadOptions?.signal?.reason).toBe(
      "media preview request superseded",
    );
    expect(secondReadOptions?.signal).toBeInstanceOf(AbortSignal);
    expect(secondReadOptions?.signal?.aborted).toBe(false);
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    const visibleArtifact = upsertGeneralArtifact.mock
      .calls[0]?.[0] as Artifact;
    expect(visibleArtifact.meta.mediaUri).toBe("sidecar://media/image-2");
    expect(visibleArtifact.content).toBe("blob:media-1");

    await act(async () => {
      firstRead.resolve({ result: createMediaReadResponse() });
      await firstPreview;
    });

    expect(readMedia).toHaveBeenCalledTimes(2);
    expect(createObjectUrl).toHaveBeenCalledTimes(1);
    expect(upsertGeneralArtifact).toHaveBeenCalledTimes(1);
    expect(handleWorkspaceArtifactClick).toHaveBeenCalledTimes(1);
    expect(requestCanvasWorkbenchPreviewOpen).toHaveBeenCalledTimes(1);
    expect(logAgentDebug).not.toHaveBeenCalled();
  });

  it("卸载后迟到 media read 结果不应写 UI 或创建 object URL", async () => {
    const upsertGeneralArtifact = vi.fn();
    const handleWorkspaceArtifactClick = vi.fn();
    const requestCanvasWorkbenchPreviewOpen = vi.fn();
    const pendingRead = createDeferred<{
      result: AppServerMediaReadResponse;
    }>();
    const readMedia = vi.fn().mockReturnValue(pendingRead.promise);
    vi.mocked(createAppServerClient).mockReturnValue({
      readMedia,
    } as ReturnType<typeof createAppServerClient>);
    const { createObjectUrl, revokeObjectUrl } = installObjectUrlMocks();
    const { render, getValue } = renderHook({
      handleWorkspaceArtifactClick,
      requestCanvasWorkbenchPreviewOpen,
      upsertGeneralArtifact,
    });

    await render();
    const preview = getValue().openMediaReferencePreview(
      createSidecarMediaTarget(),
      createMessage("assistant-sidecar-media-unmount"),
    );
    await Promise.resolve();

    act(() => {
      const mounted = mountedRoots.pop();
      if (!mounted) {
        throw new Error("缺少已挂载 hook");
      }
      mounted.root.unmount();
      mounted.container.remove();
    });

    const readOptions = readMedia.mock.calls[0]?.[1] as
      | { signal?: AbortSignal }
      | undefined;
    expect(readOptions?.signal).toBeInstanceOf(AbortSignal);
    expect(readOptions?.signal?.aborted).toBe(true);
    expect(readOptions?.signal?.reason).toBe("media preview runtime disposed");

    await act(async () => {
      pendingRead.resolve({ result: createMediaReadResponse() });
      await preview;
    });

    expect(readMedia).toHaveBeenCalledTimes(1);
    expect(createObjectUrl).not.toHaveBeenCalled();
    expect(revokeObjectUrl).not.toHaveBeenCalled();
    expect(upsertGeneralArtifact).not.toHaveBeenCalled();
    expect(handleWorkspaceArtifactClick).not.toHaveBeenCalled();
    expect(requestCanvasWorkbenchPreviewOpen).not.toHaveBeenCalled();
    expect(logAgentDebug).not.toHaveBeenCalled();
  });

  it("object URL 数量超过 runtime 预算时应释放最旧 preview URL", async () => {
    const readMedia = vi.fn(async () => ({
      result: createMediaReadResponse(),
    }));
    vi.mocked(createAppServerClient).mockReturnValue({
      readMedia,
    } as ReturnType<typeof createAppServerClient>);
    const { revokeObjectUrl } = installObjectUrlMocks();
    const { render, getValue } = renderHook();

    await render();
    for (
      let index = 0;
      index <=
      WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY.objectUrlMaxCount;
      index += 1
    ) {
      await act(async () => {
        await getValue().openMediaReferencePreview(
          createSidecarMediaTarget({
            uri: `sidecar://media/image-${index + 1}`,
            title: `image-${index + 1}.png`,
          }),
          createMessage(`assistant-sidecar-media-budget-${index + 1}`),
        );
      });
    }

    expect(readMedia).toHaveBeenCalledTimes(
      WORKSPACE_MEDIA_REFERENCE_PREVIEW_RUNTIME_POLICY.objectUrlMaxCount + 1,
    );
    expect(revokeObjectUrl).toHaveBeenCalledTimes(1);
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:media-1");
  });
});
