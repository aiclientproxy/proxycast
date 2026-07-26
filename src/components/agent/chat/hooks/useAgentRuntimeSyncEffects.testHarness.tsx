import { act } from "react";
import { createRoot } from "react-dom/client";
import { vi } from "vitest";
import type { AgentRuntimeAppServerClient } from "@/lib/api/agentRuntime/threadClient";
import type { AgentThreadTurn } from "../types";
import { useAgentRuntimeSyncEffects } from "./useAgentRuntimeSyncEffects";

const runtimeSyncEffectMocks = vi.hoisted(() => ({
  mockIsAppServerBridgeAvailable: vi.fn(() => false),
  mockHasDevBridgeEventListenerCapability: vi.fn(() => false),
  mockHasDesktopHostEventListenerCapability: vi.fn(() => true),
  mockSafeInvoke: vi.fn(),
  mockSafeListen: vi.fn(async () => () => {}),
}));

export const {
  mockIsAppServerBridgeAvailable,
  mockHasDevBridgeEventListenerCapability,
  mockHasDesktopHostEventListenerCapability,
  mockSafeInvoke,
  mockSafeListen,
} = runtimeSyncEffectMocks;

vi.mock("@/lib/api/appServerBridgeAvailability", () => ({
  isAppServerBridgeAvailable:
    runtimeSyncEffectMocks.mockIsAppServerBridgeAvailable,
}));

vi.mock("@/lib/dev-bridge", () => ({
  hasDevBridgeEventListenerCapability:
    runtimeSyncEffectMocks.mockHasDevBridgeEventListenerCapability,
  safeInvoke: runtimeSyncEffectMocks.mockSafeInvoke,
  safeListen: runtimeSyncEffectMocks.mockSafeListen,
}));

vi.mock("@/lib/desktop-runtime", () => ({
  hasDesktopHostEventListenerCapability:
    runtimeSyncEffectMocks.mockHasDesktopHostEventListenerCapability,
}));

type HookProps = Parameters<typeof useAgentRuntimeSyncEffects>[0];

interface HookHarness {
  render: (nextProps?: Partial<HookProps>) => Promise<void>;
  unmount: () => void;
}

const mountedRoots: Array<{
  container: HTMLDivElement;
  root: ReturnType<typeof createRoot>;
}> = [];

export async function flushCoalescedRefresh(): Promise<void> {
  await act(async () => {
    vi.advanceTimersByTime(120);
    await Promise.resolve();
  });
}

export function createThreadTurn(
  overrides?: Partial<AgentThreadTurn>,
): AgentThreadTurn {
  return {
    id: "turn-1",
    thread_id: "thread-1",
    prompt_text: "继续执行",
    status: "completed",
    started_at: "2026-03-29T00:00:00.000Z",
    created_at: "2026-03-29T00:00:00.000Z",
    updated_at: "2026-03-29T00:00:00.000Z",
    ...overrides,
  };
}

export function runtimeSyncRefreshRequest(source: string) {
  return {
    source,
    detailMergeMode: "runtime_sync",
  };
}

export function terminalRefreshRequest(source: string) {
  return {
    source,
    detailMergeMode: "terminal_reconcile",
  };
}

export function createAppServerThreadClientMock(): AgentRuntimeAppServerClient {
  return {
    readSession: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        session: {
          sessionId: "session-1",
          threadId: "thread-1",
          appId: "agent-chat",
          status: "idle",
          createdAt: "2026-06-06T00:00:00.000Z",
          updatedAt: "2026-06-06T00:00:00.000Z",
        },
        turns: [],
      },
      response: {
        id: 1,
        result: {
          session: {
            sessionId: "session-1",
            threadId: "thread-1",
            appId: "agent-chat",
            status: "idle",
            createdAt: "2026-06-06T00:00:00.000Z",
            updatedAt: "2026-06-06T00:00:00.000Z",
          },
          turns: [],
        },
      },
      messages: [],
      notifications: [],
    }),
    runThreadShellCommand: vi.fn().mockResolvedValue({
      id: 4,
      result: {},
      response: { id: 4, result: {} },
      messages: [],
      notifications: [],
    }),
    startTurn: vi.fn().mockResolvedValue({}),
    cancelTurn: vi.fn().mockResolvedValue({}),
    startThreadCompaction: vi.fn().mockResolvedValue({
      id: 1,
      result: {},
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    resumeThread: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        thread: { id: "thread-1", sessionId: "session-1", turns: [] },
        model: "gpt-5.4",
        modelProvider: "openai",
        cwd: "/tmp/workspace",
      },
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    respondAction: vi.fn().mockResolvedValue({}),
    listAgentSessionFileCheckpoints: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        sessionId: "session-1",
        threadId: "thread-1",
        checkpointCount: 0,
        checkpoints: [],
      },
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    getAgentSessionFileCheckpoint: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        sessionId: "session-1",
        threadId: "thread-1",
        checkpoint: {
          checkpointId: "checkpoint-1",
          turnId: "turn-1",
          path: "src/App.tsx",
          source: "tool_result",
          updatedAt: "2026-06-06T00:00:00.000Z",
          validationIssueCount: 0,
        },
        livePath: "/tmp/work/src/App.tsx",
        snapshotPath: "/tmp/work/.lime/checkpoints/checkpoint-1/App.tsx",
        versionHistory: [],
        validationIssues: [],
      },
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    diffAgentSessionFileCheckpoint: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        sessionId: "session-1",
        threadId: "thread-1",
        checkpoint: {
          checkpointId: "checkpoint-1",
          turnId: "turn-1",
          path: "src/App.tsx",
          source: "tool_result",
          updatedAt: "2026-06-06T00:00:00.000Z",
          validationIssueCount: 0,
        },
        diff: [],
      },
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    restoreAgentSessionFileCheckpoint: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        sessionId: "session-1",
        threadId: "thread-1",
        checkpoint: {
          checkpointId: "checkpoint-1",
          turnId: "turn-1",
          path: "src/App.tsx",
          source: "tool_result",
          updatedAt: "2026-06-06T00:00:00.000Z",
          validationIssueCount: 0,
        },
        livePath: "src/App.tsx",
        snapshotPath: ".lime/checkpoints/checkpoint-1/App.tsx",
        backupPath: null,
        restoredAt: "2026-06-06T00:00:01.000Z",
      },
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    listCapabilities: vi.fn().mockResolvedValue({
      id: 1,
      result: {
        capabilities: [],
        runtimeCapabilityManifest: {
          schemaVersion: "lime-runtime-capability-manifest/v0.1",
          runtimeId: "app-server",
          generatedAt: "2026-06-12T00:00:00.000Z",
          capabilities: [],
        },
      },
      response: {
        id: 1,
        result: {},
      },
      messages: [],
      notifications: [],
    }),
    drainEvents: vi.fn().mockResolvedValue([]),
  };
}

export async function mountHook(
  props?: Partial<HookProps>,
): Promise<HookHarness> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const defaultProps: HookProps = {
    runtime: {
      listenToTurnEvents: vi.fn(async () => () => {}),
    },
    sessionIdRef: { current: "session-1" },
    sessionId: "session-1",
    currentTurnEventName: null,
    isSending: false,
    threadRead: null,
    threadReadStatus: null,
    queuedTurnCount: 0,
    threadTurns: [],
    refreshSessionDetail: vi.fn(async () => true),
    refreshSessionReadModel: vi.fn(async () => true),
  };

  function TestComponent(currentProps: HookProps) {
    useAgentRuntimeSyncEffects(currentProps);
    return null;
  }

  const render = async (nextProps?: Partial<HookProps>) => {
    const mergedProps = {
      ...defaultProps,
      ...props,
      ...nextProps,
    };
    await act(async () => {
      root.render(<TestComponent {...mergedProps} />);
      await Promise.resolve();
    });
  };

  await render();
  const mounted = { container, root };
  mountedRoots.push(mounted);

  return {
    render,
    unmount: () => {
      const index = mountedRoots.indexOf(mounted);
      if (index >= 0) {
        mountedRoots.splice(index, 1);
      }
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

export function setupRuntimeSyncEffectsTestHarness() {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-03-29T00:05:00.000Z"));
  mockIsAppServerBridgeAvailable.mockReturnValue(false);
  mockHasDevBridgeEventListenerCapability.mockReturnValue(false);
  mockHasDesktopHostEventListenerCapability.mockReturnValue(true);
  mockSafeListen.mockResolvedValue(() => {});
}

export function cleanupRuntimeSyncEffectsTestHarness() {
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
  vi.useRealTimers();
  vi.clearAllMocks();
}
