import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Project } from "@limecloud/app-server-client";
import type { AppServerEventBusSubscription } from "@/lib/api/appServerEventBus";
import {
  isThreadProjectDirectoryNotification,
  useThreadProjectDirectory,
} from "./useThreadProjectDirectory";

const firstProject: Project = {
  id: "project-1",
  name: "Lime",
  roots: [{ path: "/workspace/lime" }],
  metadata: {},
  position: 0,
  createdAt: 1,
  updatedAt: 1,
};

describe("useThreadProjectDirectory", () => {
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

  it("应消费 project/changed 与当前 thread/project/updated 并在切换 Thread 时恢复归属", async () => {
    const readDirectory = vi
      .fn()
      .mockResolvedValueOnce({ projects: [firstProject], projectId: null })
      .mockResolvedValueOnce({
        projects: [firstProject],
        projectId: "project-1",
      })
      .mockResolvedValueOnce({ projects: [firstProject], projectId: null });
    let subscription: AppServerEventBusSubscription | null = null;
    const subscribeNotifications = vi.fn(
      (next: AppServerEventBusSubscription) => {
        subscription = next;
        return vi.fn();
      },
    );
    let current: ReturnType<typeof useThreadProjectDirectory> | null = null;
    const container = document.createElement("div");
    const root = createRoot(container);

    function Harness({ threadId }: { threadId: string }) {
      current = useThreadProjectDirectory({
        threadId,
        readDirectory,
        subscribeNotifications,
      });
      return null;
    }

    try {
      await act(async () => {
        root.render(<Harness threadId="thread-1" />);
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(current).toMatchObject({
        loading: false,
        projectId: null,
        projects: [firstProject],
      });

      await act(async () => {
        subscription?.onNotifications?.([
          {
            jsonrpc: "2.0",
            method: "thread/project/updated",
            params: { threadId: "thread-other", projectId: "project-1" },
          },
          {
            jsonrpc: "2.0",
            method: "project/changed",
            params: { projectId: "project-1", changeType: "updated" },
          },
        ]);
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(readDirectory).toHaveBeenCalledTimes(2);
      expect(current?.projectId).toBe("project-1");

      await act(async () => {
        root.render(<Harness threadId="thread-2" />);
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(readDirectory).toHaveBeenLastCalledWith("thread-2");
      expect(current?.projectId).toBeNull();
    } finally {
      await act(async () => root.unmount());
    }
  });

  it("通知过滤应只接受目录变更或当前 Thread 归属变更", () => {
    expect(
      isThreadProjectDirectoryNotification(
        {
          jsonrpc: "2.0",
          method: "project/changed",
          params: { projectId: "project-1", changeType: "created" },
        },
        "thread-1",
      ),
    ).toBe(true);
    expect(
      isThreadProjectDirectoryNotification(
        {
          jsonrpc: "2.0",
          method: "thread/project/updated",
          params: { threadId: "thread-1", projectId: null },
        },
        "thread-1",
      ),
    ).toBe(true);
    expect(
      isThreadProjectDirectoryNotification(
        {
          jsonrpc: "2.0",
          method: "thread/project/updated",
          params: { threadId: "thread-2", projectId: null },
        },
        "thread-1",
      ),
    ).toBe(false);
  });
});
