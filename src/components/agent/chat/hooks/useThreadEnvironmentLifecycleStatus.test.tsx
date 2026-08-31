import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AppServerEventBusSubscription } from "@/lib/api/appServerEventBus";
import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import {
  projectThreadEnvironmentLifecycleStatus,
  readThreadEnvironmentIds,
  useThreadEnvironmentLifecycleStatus,
} from "./useThreadEnvironmentLifecycleStatus";

const threadRead: AgentRuntimeThreadReadModel = {
  thread_id: "thread-1",
  session_business_object_ref_metadata: {
    environments: [
      { environmentId: "local", cwd: "/workspace" },
      { environmentId: "remote-a", cwd: "/remote/workspace" },
    ],
  },
};

describe("useThreadEnvironmentLifecycleStatus", () => {
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

  it("从 Thread metadata 读取唯一 Environment identity", () => {
    expect(
      readThreadEnvironmentIds({
        ...threadRead,
        session_business_object_ref_metadata: {
          environments: [
            { environmentId: "local" },
            { environment_id: "remote-a" },
            { id: "remote-a" },
            { environmentId: "" },
          ],
        },
      }),
    ).toEqual(["local", "remote-a"]);
  });

  it("只投影当前 Thread 的 connected/disconnected 通知", () => {
    const disconnected = projectThreadEnvironmentLifecycleStatus(
      [{ environmentId: "remote-a", status: "connected" }],
      {
        jsonrpc: "2.0",
        method: "thread/environment/disconnected",
        params: { threadId: "thread-1", environmentId: "remote-a" },
      },
      "thread-1",
    );
    expect(disconnected).toEqual([
      { environmentId: "remote-a", status: "disconnected" },
    ]);
    expect(
      projectThreadEnvironmentLifecycleStatus(
        disconnected,
        {
          jsonrpc: "2.0",
          method: "thread/environment/connected",
          params: { threadId: "thread-other", environmentId: "remote-a" },
        },
        "thread-1",
      ),
    ).toEqual(disconnected);
  });

  it("首次读取状态并由通知完成断线到重连闭环", async () => {
    let subscription: AppServerEventBusSubscription | null = null;
    const unsubscribe = vi.fn();
    const subscribeNotifications = vi.fn(
      (next: AppServerEventBusSubscription) => {
        subscription = next;
        return unsubscribe;
      },
    );
    const readStatuses = vi.fn(async () => [
      { environmentId: "local", status: "connected" as const },
      { environmentId: "remote-a", status: "connected" as const },
    ]);
    let current: ReturnType<typeof useThreadEnvironmentLifecycleStatus> = [];
    const container = document.createElement("div");
    const root = createRoot(container);

    function Harness({ threadId }: { threadId: string }) {
      current = useThreadEnvironmentLifecycleStatus({
        readStatuses,
        subscribeNotifications,
        threadId,
        threadRead,
      });
      return null;
    }

    try {
      await act(async () => {
        root.render(<Harness threadId="thread-1" />);
        await Promise.resolve();
      });
      expect(readStatuses).toHaveBeenCalledWith(["local", "remote-a"]);
      expect(subscription?.getDrainOptions?.()).toEqual({
        includeRecent: true,
      });
      expect(current).toEqual([
        { environmentId: "local", status: "connected" },
        { environmentId: "remote-a", status: "connected" },
      ]);

      act(() =>
        subscription?.onNotifications?.([
          {
            jsonrpc: "2.0",
            method: "thread/environment/disconnected",
            params: { threadId: "thread-1", environmentId: "remote-a" },
          },
        ]),
      );
      expect(current.at(1)?.status).toBe("disconnected");

      act(() =>
        subscription?.onNotifications?.([
          {
            jsonrpc: "2.0",
            method: "thread/environment/connected",
            params: { threadId: "thread-1", environmentId: "remote-a" },
          },
        ]),
      );
      expect(current.at(1)?.status).toBe("connected");
    } finally {
      await act(async () => root.unmount());
      expect(unsubscribe).toHaveBeenCalledTimes(1);
    }
  });

  it("优先使用 canonical Thread environment snapshot 作为首帧状态", async () => {
    const readStatuses = vi.fn(async () => [
      {
        environmentId: "remote-a",
        status: "disconnected" as const,
        error: "exec-server unavailable",
      },
    ]);
    let current: ReturnType<typeof useThreadEnvironmentLifecycleStatus> = [];
    const container = document.createElement("div");
    const root = createRoot(container);
    const canonicalThreadRead: AgentRuntimeThreadReadModel = {
      thread_id: "thread-1",
      environment_selections: [
        {
          environment_id: "remote-a",
          cwd: "/remote/workspace",
          status: "disconnected",
          error: "exec-server unavailable",
        },
      ],
    };

    function Harness() {
      current = useThreadEnvironmentLifecycleStatus({
        readStatuses,
        threadId: "thread-1",
        threadRead: canonicalThreadRead,
      });
      return null;
    }

    try {
      await act(async () => {
        root.render(<Harness />);
        await Promise.resolve();
      });
      expect(current).toEqual([
        {
          environmentId: "remote-a",
          status: "disconnected",
          error: "exec-server unavailable",
        },
      ]);
      expect(readStatuses).toHaveBeenCalledWith(["remote-a"]);
    } finally {
      await act(async () => root.unmount());
    }
  });
});
