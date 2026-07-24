import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  AppServerThread,
  AppServerThreadListResponse,
} from "@/lib/api/appServer";
import type { CanonicalThreadListClient } from "@/lib/api/agentRuntime/canonicalThreadClient";
import {
  deriveWorkspaceSubagentRuntime,
  useWorkspaceTeamRuntime,
} from "./useWorkspaceTeamRuntime";

type HookProps = Parameters<typeof useWorkspaceTeamRuntime>[0];

const mountedRoots: Array<{ container: HTMLDivElement; root: Root }> = [];

function thread(
  threadId: string,
  status: AppServerThread["status"] = { type: "idle" },
): AppServerThread {
  return {
    cliVersion: "1.111.0",
    createdAt: 100,
    cwd: "/workspace",
    ephemeral: false,
    id: threadId,
    modelProvider: "openai",
    parentThreadId: "root-1",
    preview: "",
    sessionId: `session-${threadId}`,
    source: "app_server",
    status,
    turns: [],
    updatedAt: 200,
  };
}

function response(result: AppServerThreadListResponse) {
  return {
    configWarnings: [],
    id: 1,
    messages: [],
    notifications: [],
    response: { id: 1, result },
    result,
  };
}

function renderTeamRuntimeHook(initialProps: HookProps) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let latestValue: ReturnType<typeof useWorkspaceTeamRuntime> | null = null;

  function Probe(props: HookProps) {
    latestValue = useWorkspaceTeamRuntime(props);
    return null;
  }

  const render = async (props: HookProps = initialProps) => {
    await act(async () => {
      root.render(React.createElement(Probe, props));
      await Promise.resolve();
      await Promise.resolve();
    });
  };

  mountedRoots.push({ container, root });
  return {
    getValue: () => {
      if (!latestValue) {
        throw new Error("workspace team runtime hook 尚未初始化");
      }
      return latestValue;
    },
    render,
  };
}

beforeEach(() => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) {
      break;
    }
    act(() => mounted.root.unmount());
    mounted.container.remove();
  }
});

describe("workspace canonical SubAgent runtime", () => {
  it("只从 canonical child facts 派生会话标题和可见性", () => {
    expect(
      deriveWorkspaceSubagentRuntime({
        canonicalChildren: [
          {
            name: "Reviewer",
            parentThreadId: "root-1",
            sessionId: "session-child-1",
            status: "running",
            threadId: "child-1",
            updatedAtMs: 2,
          },
        ],
        currentTopicId: "root-1",
        hasParentThread: false,
        subagentEnabled: false,
        topics: [{ id: "root-1", title: "Canonical thread" }],
      }),
    ).toEqual({
      currentSessionTitle: "Canonical thread",
      hasRuntimeSessions: true,
      subagentsRuntimeVisible: true,
    });
  });

  it("读取 canonical child Thread，并在 refresh key 变化后刷新状态", async () => {
    const listThreads = vi
      .fn<CanonicalThreadListClient["listThreads"]>()
      .mockResolvedValueOnce(
        response({
          data: [thread("child-1", { type: "active", activeFlags: [] })],
        }),
      )
      .mockResolvedValueOnce(
        response({
          data: [
            {
              ...thread("child-1"),
              turns: [
                {
                  completedAt: 300,
                  items: [],
                  status: "completed",
                  id: "turn-1",
                  startedAt: 100,
                },
              ],
            },
          ],
        }),
      );
    const stopSending = vi.fn(async () => undefined);
    const baseProps: HookProps = {
      canonicalClient: { listThreads },
      canonicalRefreshKey: 1,
      session: {
        currentTopicId: "root-1",
        parentThreadId: "root-1",
        subagentEnabled: false,
        topics: [{ id: "root-1", title: "Canonical thread" }],
      },
      stopSending,
    };
    const harness = renderTeamRuntimeHook(baseProps);

    await harness.render();

    expect(harness.getValue()).toMatchObject({
      canonicalChildCounts: { active: 1, running: 1, total: 1 },
      canonicalChildren: [expect.objectContaining({ status: "running" })],
      canonicalChildrenLoading: false,
      handleStopSending: stopSending,
    });

    await harness.render({ ...baseProps, canonicalRefreshKey: 2 });

    expect(harness.getValue()).toMatchObject({
      canonicalChildCounts: { active: 0, settled: 1, total: 1 },
      canonicalChildren: [expect.objectContaining({ status: "completed" })],
      canonicalChildrenLoading: false,
    });
    expect(listThreads).toHaveBeenCalledTimes(2);
  });

  it("没有 parent Thread 时不访问 App Server", async () => {
    const listThreads = vi.fn<CanonicalThreadListClient["listThreads"]>();
    const stopSending = vi.fn(async () => undefined);
    const harness = renderTeamRuntimeHook({
      canonicalClient: { listThreads },
      session: {
        currentTopicId: null,
        parentThreadId: null,
        subagentEnabled: true,
        topics: [],
      },
      stopSending,
    });

    await harness.render();

    expect(harness.getValue()).toMatchObject({
      canonicalChildren: [],
      currentSessionTitle: null,
      hasRuntimeSessions: false,
      subagentsRuntimeVisible: true,
    });
    expect(listThreads).not.toHaveBeenCalled();
  });

  it("当前 canonical Thread 有 parent 时保持 Subagents runtime 可见", async () => {
    const listThreads = vi
      .fn<CanonicalThreadListClient["listThreads"]>()
      .mockResolvedValue(response({ data: [thread("child-1")] }));
    const harness = renderTeamRuntimeHook({
      canonicalClient: { listThreads },
      session: {
        currentTopicId: "child-1",
        parentThreadId: "child-1",
        subagentEnabled: false,
        topics: [{ id: "child-1", title: "Child thread" }],
      },
      stopSending: vi.fn(async () => undefined),
    });

    await harness.render();

    expect(harness.getValue()).toMatchObject({
      canonicalChildren: [],
      currentSessionTitle: "Child thread",
      hasRuntimeSessions: true,
      subagentsRuntimeVisible: true,
    });
  });
});
