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

  it("展示前三条 durable submission 和剩余数量", () => {
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
    expect(
      container.querySelector("[data-testid=thread-queue-items]")?.children,
    ).toHaveLength(4);
    expect(container.textContent).toContain("修复测试");
    expect(container.textContent).toContain(
      "agentChat.threadQueue.itemFallback",
    );
    expect(container.textContent).toContain("agentChat.threadQueue.more:1");
  });

  it("读取失败时保留已有 Queue 并显示收敛状态", () => {
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
    expect(container.textContent).toContain("保留中的消息");
    expect(container.textContent).toContain(
      "agentChat.threadQueue.refreshFailed",
    );
  });
});
