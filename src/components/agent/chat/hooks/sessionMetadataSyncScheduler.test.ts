import { describe, expect, it, vi } from "vitest";
import type { SessionMetadataSyncPlan } from "./sessionMetadataSyncController";
import {
  buildSessionMetadataSyncBrowserSkipEvent,
  buildSessionMetadataSyncStaleSkipEvent,
  scheduleSessionMetadataSync,
  type SessionMetadataSyncScheduler,
} from "./sessionMetadataSyncScheduler";

const flushPromises = async () => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

function metadataPlan(
  overrides: Partial<SessionMetadataSyncPlan> = {},
): SessionMetadataSyncPlan {
  return {
    accessMode: "current",
    accessModeSource: "execution_runtime",
    fallbackExecutionStrategy: null,
    fallbackProviderPreference: null,
    hasPatch: true,
    modelPreferenceSource: null,
    patch: { accessMode: "current" },
    providerPreferenceToApply: null,
    shouldPersistAccessMode: true,
    ...overrides,
  };
}

function createScheduler() {
  const scheduledTasks: Array<{
    options: { idleTimeoutMs: number; minimumDelayMs: number };
    task: () => void;
  }> = [];
  const cancel = vi.fn();
  const scheduler: SessionMetadataSyncScheduler = {
    schedule: vi.fn((task, options) => {
      scheduledTasks.push({ task, options });
      return cancel;
    }),
  };

  return { cancel, scheduledTasks, scheduler };
}

describe("sessionMetadataSyncScheduler", () => {
  it("无 patch 时不应调度或触发 skip 日志", () => {
    const { scheduler } = createScheduler();
    const onSkipped = vi.fn();
    const result = scheduleSessionMetadataSync({
      getCurrentRequestVersion: () => 1,
      getCurrentSessionId: () => "topic-a",
      hasRuntimeInvokeCapability: true,
      idleTimeoutMs: 15_000,
      minimumDelayMs: 8_000,
      onError: vi.fn(),
      onSkipped,
      onSynced: vi.fn(),
      plan: metadataPlan({ hasPatch: false, patch: {} }),
      runtime: {
        setSessionExecutionStrategy: vi.fn(),
        setSessionProviderSelection: vi.fn(),
      },
      scheduler,
      sessionId: "topic-a",
      setPendingCancel: vi.fn(),
      switchRequestVersion: 1,
      workspaceId: "workspace-a",
    });

    expect(result).toBe("skipped_no_patch");
    expect(scheduler.schedule).not.toHaveBeenCalled();
    expect(onSkipped).not.toHaveBeenCalled();
  });

  it("浏览器桥接低优先级回填时应跳过 runtime invoke", () => {
    const { scheduler } = createScheduler();
    const onSkipped = vi.fn();
    const result = scheduleSessionMetadataSync({
      getCurrentRequestVersion: () => 1,
      getCurrentSessionId: () => "topic-a",
      hasRuntimeInvokeCapability: false,
      idleTimeoutMs: 15_000,
      minimumDelayMs: 8_000,
      onError: vi.fn(),
      onSkipped,
      onSynced: vi.fn(),
      plan: metadataPlan(),
      runtime: {
        setSessionExecutionStrategy: vi.fn(),
        setSessionProviderSelection: vi.fn(),
      },
      scheduler,
      sessionId: "topic-a",
      setPendingCancel: vi.fn(),
      switchRequestVersion: 1,
      workspaceId: "workspace-a",
    });

    expect(result).toBe("skipped_no_invoke");
    expect(scheduler.schedule).not.toHaveBeenCalled();
    expect(onSkipped).toHaveBeenCalledWith(
      buildSessionMetadataSyncBrowserSkipEvent({
        sessionId: "topic-a",
        workspaceId: "workspace-a",
      }),
    );
  });

  it("应取消上一次调度并按 idle 参数安排 metadata sync", async () => {
    const { scheduledTasks, scheduler } = createScheduler();
    const previousCancel = vi.fn();
    const setPendingCancel = vi.fn();
    const runtime = {
      updateSessionMetadata: vi.fn().mockResolvedValue(undefined),
      setSessionExecutionStrategy: vi.fn(),
      setSessionProviderSelection: vi.fn(),
    };
    const plan = metadataPlan({
      fallbackExecutionStrategy: "react",
      patch: {
        accessMode: "current",
        executionStrategy: "react",
      },
    });

    const result = scheduleSessionMetadataSync({
      getCurrentRequestVersion: () => 7,
      getCurrentSessionId: () => "topic-a",
      hasRuntimeInvokeCapability: true,
      idleTimeoutMs: 15_000,
      minimumDelayMs: 8_000,
      onError: vi.fn(),
      onSkipped: vi.fn(),
      onSynced: vi.fn(),
      pendingCancel: previousCancel,
      plan,
      runtime,
      scheduler,
      sessionId: "topic-a",
      setPendingCancel,
      switchRequestVersion: 7,
      workspaceId: "workspace-a",
    });

    expect(result).toBe("scheduled");
    expect(previousCancel).toHaveBeenCalledTimes(1);
    expect(scheduler.schedule).toHaveBeenCalledTimes(1);
    expect(scheduledTasks[0]?.options).toEqual({
      idleTimeoutMs: 15_000,
      minimumDelayMs: 8_000,
    });
    const scheduledCancel = setPendingCancel.mock.lastCall?.[0];
    expect(scheduledCancel).toEqual(expect.any(Function));

    scheduledTasks[0]?.task();
    await flushPromises();

    expect(setPendingCancel).toHaveBeenCalledWith(null);
    expect(runtime.updateSessionMetadata).toHaveBeenCalledWith("topic-a", {
      accessMode: "current",
      executionStrategy: "react",
    });
  });

  it("取消后即使 scheduler 回调任务也不应触发 metadata sync", async () => {
    const { scheduledTasks, scheduler } = createScheduler();
    const setPendingCancel = vi.fn();
    const runtime = {
      updateSessionMetadata: vi.fn().mockResolvedValue(undefined),
      setSessionExecutionStrategy: vi.fn(),
      setSessionProviderSelection: vi.fn(),
    };

    scheduleSessionMetadataSync({
      getCurrentRequestVersion: () => 1,
      getCurrentSessionId: () => "topic-a",
      hasRuntimeInvokeCapability: true,
      idleTimeoutMs: 15_000,
      minimumDelayMs: 8_000,
      onError: vi.fn(),
      onSkipped: vi.fn(),
      onSynced: vi.fn(),
      plan: metadataPlan(),
      runtime,
      scheduler,
      sessionId: "topic-a",
      setPendingCancel,
      switchRequestVersion: 1,
    });

    const cancel = setPendingCancel.mock.lastCall?.[0] as
      | (() => void)
      | undefined;
    cancel?.();
    scheduledTasks[0]?.task();
    await flushPromises();

    expect(runtime.updateSessionMetadata).not.toHaveBeenCalled();
  });

  it("调度执行时如果会话已过期，应跳过 metadata sync", () => {
    const { scheduledTasks, scheduler } = createScheduler();
    const runtime = {
      updateSessionMetadata: vi.fn(),
      setSessionExecutionStrategy: vi.fn(),
      setSessionProviderSelection: vi.fn(),
    };
    const onSkipped = vi.fn();

    scheduleSessionMetadataSync({
      getCurrentRequestVersion: () => 8,
      getCurrentSessionId: () => "topic-b",
      hasRuntimeInvokeCapability: true,
      idleTimeoutMs: 15_000,
      minimumDelayMs: 8_000,
      onError: vi.fn(),
      onSkipped,
      onSynced: vi.fn(),
      plan: metadataPlan(),
      runtime,
      scheduler,
      sessionId: "topic-a",
      setPendingCancel: vi.fn(),
      switchRequestVersion: 7,
      workspaceId: "workspace-a",
    });

    scheduledTasks[0]?.task();

    expect(runtime.updateSessionMetadata).not.toHaveBeenCalled();
    expect(onSkipped).toHaveBeenCalledWith(
      buildSessionMetadataSyncStaleSkipEvent({
        currentSessionId: "topic-b",
        sessionId: "topic-a",
        switchRequestVersion: 7,
        workspaceId: "workspace-a",
      }),
    );
  });

  it("metadata sync 失败时应回调错误处理", async () => {
    const { scheduledTasks, scheduler } = createScheduler();
    const error = new Error("sync failed");
    const onError = vi.fn();

    scheduleSessionMetadataSync({
      getCurrentRequestVersion: () => 1,
      getCurrentSessionId: () => "topic-a",
      hasRuntimeInvokeCapability: true,
      idleTimeoutMs: 15_000,
      minimumDelayMs: 8_000,
      onError,
      onSkipped: vi.fn(),
      onSynced: vi.fn(),
      plan: metadataPlan(),
      runtime: {
        updateSessionMetadata: vi.fn().mockRejectedValue(error),
        setSessionExecutionStrategy: vi.fn(),
        setSessionProviderSelection: vi.fn(),
      },
      scheduler,
      sessionId: "topic-a",
      setPendingCancel: vi.fn(),
      switchRequestVersion: 1,
    });

    scheduledTasks[0]?.task();
    await flushPromises();

    expect(onError).toHaveBeenCalledWith(error);
  });
});
