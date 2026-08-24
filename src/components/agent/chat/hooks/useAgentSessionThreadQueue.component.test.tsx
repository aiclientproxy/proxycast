import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { QueuedSubmission } from "@limecloud/app-server-client";
import type { AppServerEventBusSubscription } from "@/lib/api/appServerEventBus";
import { useAgentSessionThreadQueue } from "./useAgentSessionThreadQueue";

const initial: QueuedSubmission = {
  clientUserMessageId: "client-initial",
  id: "queued-initial",
  input: [{ type: "text", text: "initial" }],
};
const updated: QueuedSubmission = {
  clientUserMessageId: "client-updated",
  id: "queued-updated",
  input: [{ type: "text", text: "updated" }],
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

describe("useAgentSessionThreadQueue component", () => {
  beforeEach(() => {
    (
      globalThis as typeof globalThis & {
        IS_REACT_ACT_ENVIRONMENT?: boolean;
      }
    ).IS_REACT_ACT_ENVIRONMENT = true;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Queue changed 通知应刷新当前 Thread，切换 Thread 应立即清空旧数据", async () => {
    const firstRead = deferred<QueuedSubmission[]>();
    const updatedRead = deferred<QueuedSubmission[]>();
    const secondThreadRead = deferred<QueuedSubmission[]>();
    const readQueue = vi
      .fn<(threadId: string) => Promise<QueuedSubmission[]>>()
      .mockImplementationOnce(() => firstRead.promise)
      .mockImplementationOnce(() => updatedRead.promise)
      .mockImplementationOnce(() => secondThreadRead.promise);
    let subscription: AppServerEventBusSubscription | null = null;
    const subscribeNotifications = vi.fn(
      (nextSubscription: AppServerEventBusSubscription) => {
        subscription = nextSubscription;
        return vi.fn();
      },
    );
    let current: ReturnType<typeof useAgentSessionThreadQueue> | null = null;
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    function TestComponent({ threadId }: { threadId: string }) {
      current = useAgentSessionThreadQueue({
        readQueue,
        subscribeNotifications,
        threadId,
      });
      return null;
    }

    try {
      await act(async () => {
        root.render(<TestComponent threadId="thread-1" />);
        await Promise.resolve();
      });
      expect(current).toMatchObject({ loading: true, submissions: [] });

      await act(async () => {
        subscription?.onNotifications?.([
          {
            jsonrpc: "2.0",
            method: "thread/queue/changed",
            params: { threadId: "thread-other" },
          },
          {
            jsonrpc: "2.0",
            method: "thread/queue/changed",
            params: { threadId: "thread-1" },
          },
        ]);
        await Promise.resolve();
      });
      expect(readQueue).toHaveBeenCalledTimes(2);

      await act(async () => {
        updatedRead.resolve([updated]);
        await updatedRead.promise;
      });
      expect(current?.submissions).toEqual([updated]);

      await act(async () => {
        firstRead.resolve([initial]);
        await firstRead.promise;
      });
      expect(current?.submissions).toEqual([updated]);

      await act(async () => {
        root.render(<TestComponent threadId="thread-2" />);
        await Promise.resolve();
      });
      expect(current).toMatchObject({ loading: true, submissions: [] });
    } finally {
      await act(async () => root.unmount());
      container.remove();
      secondThreadRead.resolve([]);
    }
  });
});
