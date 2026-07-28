import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { MessageImage } from "../types";
import { MessageImageAttachments } from "./MessageImageAttachments";

const appServerMocks = vi.hoisted(() => ({
  readAgentSessionMedia: vi.fn(),
}));

vi.mock("@/lib/api/appServer", () => ({
  createAppServerClient: () => appServerMocks,
}));

vi.mock("@/lib/api/fileSystem", () => ({
  resolveLocalFilePreviewUrl: (path: string) => `asset://${path}`,
}));

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  appServerMocks.readAgentSessionMedia.mockReset();
  await changeLimeLocale("zh-CN");
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      continue;
    }
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
});

function renderAttachments(images: MessageImage[], sessionId?: string) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <MessageImageAttachments images={images} sessionId={sessionId} />,
    );
  });

  mountedRoots.push({ container, root });
  return container;
}

describe("MessageImageAttachments", () => {
  it("图片加载失败时应隐藏浏览器 alt 文本，只展示一处受控占位", () => {
    const container = renderAttachments([
      {
        data: "",
        mediaType: "image/png",
        sourceUri: "asset://missing-image.png",
      },
    ]);
    const image = container.querySelector(
      '[data-testid="message-image-attachment-0"]',
    );

    expect(image).not.toBeNull();

    act(() => {
      image?.dispatchEvent(new Event("error"));
    });

    expect(
      container.querySelector(
        '[data-testid="message-image-attachment-unavailable-0"]',
      ),
    ).not.toBeNull();
    expect(container.textContent).toContain("图片暂时无法显示");
    expect(container.textContent).not.toContain("图片附件");
  });

  it("sidecar 图片应通过 App Server 读取后再交给浏览器", async () => {
    appServerMocks.readAgentSessionMedia.mockResolvedValueOnce({
      result: {
        sessionId: "session-1",
        uri: "sidecar://media/input-demo.png",
        mimeType: "image/png",
        bytes: 4,
        totalBytes: 4,
        offset: 0,
        length: 4,
        contentRange: "bytes 0-3/4",
        hasMore: false,
        sha256: "sha256:demo",
        contentBase64: "iVBORw==",
      },
    });

    const container = renderAttachments(
      [
        {
          data: "",
          mediaType: "image/png",
          sourceUri: "sidecar://media/input-demo.png",
          previewUrl: "sidecar://media/input-demo.png",
        },
      ],
      "session-1",
    );

    expect(container.querySelector("img")).toBeNull();
    expect(appServerMocks.readAgentSessionMedia).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: "session-1",
        uri: "sidecar://media/input-demo.png",
      }),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );

    await act(async () => {
      await Promise.resolve();
    });

    const image = container.querySelector(
      '[data-testid="message-image-attachment-0"]',
    );
    expect(image?.getAttribute("src")).toBe("data:image/png;base64,iVBORw==");
    expect(container.querySelector('img[src^="sidecar://"]')).toBeNull();
  });
});
