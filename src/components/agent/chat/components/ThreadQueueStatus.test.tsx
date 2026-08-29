import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ThreadQueueStatus } from "./ThreadQueueStatus";

const queueHook = vi.hoisted(() => ({
  useAgentSessionThreadQueue: vi.fn(),
}));

vi.mock("../hooks/useAgentSessionThreadQueue", () => queueHook);
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { count?: number }) =>
      options?.count === undefined ? key : `${key}:${options.count}`,
  }),
}));

describe("ThreadQueueStatus", () => {
  const mounted: Array<{
    container: HTMLDivElement;
    root: ReturnType<typeof createRoot>;
  }> = [];

  afterEach(() => {
    while (mounted.length > 0) {
      const instance = mounted.pop();
      if (!instance) {
        break;
      }
      act(() => instance.root.unmount());
      instance.container.remove();
    }
    vi.clearAllMocks();
  });

  function renderQueue() {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
      root.render(<ThreadQueueStatus threadId="thread-1" />);
    });
    mounted.push({ container, root });
    return container;
  }

  it("Queue 为空时不占用会话空间", () => {
    queueHook.useAgentSessionThreadQueue.mockReturnValue({
      error: null,
      loading: false,
      submissions: [],
    });

    expect(renderQueue().textContent).toBe("");
  });

  it("只展示 durable submission 数量，不重复 canonical 时间线正文", () => {
    queueHook.useAgentSessionThreadQueue.mockReturnValue({
      error: null,
      loading: false,
      submissions: [
        {
          clientUserMessageId: "client-1",
          id: "queued-1",
          input: [{ type: "text", text: "修复测试" }],
        },
        {
          clientUserMessageId: "client-2",
          id: "queued-2",
          input: [{ type: "image", url: "sidecar://media/input-2" }],
        },
        {
          clientUserMessageId: "client-3",
          id: "queued-3",
          input: [{ type: "text", text: "更新文档" }],
        },
        {
          clientUserMessageId: "client-4",
          id: "queued-4",
          input: [{ type: "text", text: "运行验证" }],
        },
      ],
    });

    const container = renderQueue();
    expect(container.textContent).toContain("agentChat.threadQueue.title");
    expect(container.textContent).toContain(
      "agentChat.threadQueue.pendingCount:4",
    );
    expect(container.textContent).not.toContain("修复测试");
    expect(container.textContent).not.toContain("更新文档");
    expect(
      container.querySelector("[data-testid=thread-queue-items]"),
    ).toBeNull();
  });

  it("读取失败时保留数量并显示收敛状态，不重复已有正文", () => {
    queueHook.useAgentSessionThreadQueue.mockReturnValue({
      error: new Error("offline"),
      loading: false,
      submissions: [
        {
          clientUserMessageId: "client-1",
          id: "queued-1",
          input: [{ type: "text", text: "保留中的消息" }],
        },
      ],
    });

    const container = renderQueue();
    expect(container.textContent).toContain(
      "agentChat.threadQueue.pendingCount:1",
    );
    expect(container.textContent).not.toContain("保留中的消息");
    expect(container.textContent).toContain(
      "agentChat.threadQueue.refreshFailed",
    );
  });

  it("读取中只显示稳定状态，不提前显示数量或正文", () => {
    queueHook.useAgentSessionThreadQueue.mockReturnValue({
      error: null,
      loading: true,
      submissions: [
        {
          clientUserMessageId: "client-1",
          id: "queued-1",
          input: [{ type: "text", text: "读取中的消息" }],
        },
      ],
    });

    const container = renderQueue();
    expect(
      container.querySelector('[aria-label="agentChat.threadQueue.loading"]'),
    ).not.toBeNull();
    expect(container.textContent).not.toContain(
      "agentChat.threadQueue.pendingCount",
    );
    expect(container.textContent).not.toContain("读取中的消息");
  });
});
