import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  rememberInitialSessionNavigationStart,
  requestExplicitInitialSessionNavigation,
  resetInitialSessionNavigationDeduplicationForTests,
  useWorkspaceInitialSessionNavigation,
} from "./useWorkspaceInitialSessionNavigation";

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

interface ProbeProps {
  currentSessionId?: string | null;
  initialSessionId?: string | null;
  shouldAllowResolvedForceMatchedHydration?: boolean;
  shouldCancelPausedInitialSessionNavigationOnCurrentSessionChange?: boolean;
  shouldPauseInitialSessionNavigation?: boolean;
  shouldHydrateMatchedInitialSession?: boolean;
  switchTopic: (
    topicId: string,
    options?: {
      forceRefresh?: boolean;
      resumeSessionStartHooks?: boolean;
      allowDetachedSession?: boolean;
    },
  ) => Promise<unknown>;
  resolveInitialSessionSwitch?: (sessionId: string) => {
    forceRefresh?: boolean;
    resumeSessionStartHooks?: boolean;
    allowDetachedSession?: boolean;
    waitForResolution?: boolean;
  } | null;
}

function renderHook(props: ProbeProps) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  function Probe(currentProps: ProbeProps) {
    useWorkspaceInitialSessionNavigation(currentProps);
    return null;
  }

  mountedRoots.push({ root, container });

  act(() => {
    root.render(<Probe {...props} />);
  });

  return {
    rerender(nextProps: ProbeProps) {
      act(() => {
        root.render(<Probe {...nextProps} />);
      });
    },
  };
}

async function flushEffects(times = 3) {
  await act(async () => {
    for (let index = 0; index < times; index += 1) {
      await Promise.resolve();
    }
  });
}

describe("useWorkspaceInitialSessionNavigation", () => {
  beforeEach(() => {
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
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

    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    resetInitialSessionNavigationDeduplicationForTests();
  });

  it("应在初始会话和当前会话不一致时触发恢复", async () => {
    const switchTopic = vi.fn(async () => undefined);

    renderHook({
      initialSessionId: "session-42",
      currentSessionId: null,
      switchTopic,
    });

    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith("session-42", undefined);
  });

  it("当前已经在目标会话时不应重复切换", async () => {
    const switchTopic = vi.fn(async () => undefined);
    renderHook({
      initialSessionId: "session-42",
      currentSessionId: "session-42",
      switchTopic,
    });

    await flushEffects();

    expect(switchTopic).not.toHaveBeenCalled();
  });

  it("当前会话已命中但仍在恢复态时应继续触发 hydrate", async () => {
    const switchTopic = vi.fn(async () => undefined);
    renderHook({
      initialSessionId: "session-42",
      currentSessionId: "session-42",
      shouldHydrateMatchedInitialSession: true,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
      }),
    });

    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith("session-42", {
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("当前会话已命中但 resolver 要求刷新时应继续触发 hydrate", async () => {
    const switchTopic = vi.fn(async () => undefined);
    renderHook({
      initialSessionId: "session-42",
      currentSessionId: "session-42",
      shouldHydrateMatchedInitialSession: false,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
        forceRefresh: true,
      }),
    });

    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith("session-42", {
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("显式关闭 resolver force-refresh 时不应刷新已匹配会话", async () => {
    const switchTopic = vi.fn(async () => undefined);
    renderHook({
      initialSessionId: "session-42",
      currentSessionId: "session-42",
      shouldAllowResolvedForceMatchedHydration: false,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
        forceRefresh: true,
      }),
    });

    await flushEffects();

    expect(switchTopic).not.toHaveBeenCalled();
  });

  it("初始导航进入同一会话恢复壳后应复用在途 hydrate", async () => {
    let resolveSwitch: (result: unknown) => void = () => {};
    const switchTopic = vi.fn(
      () =>
        new Promise<unknown>((resolve) => {
          resolveSwitch = resolve;
        }),
    );
    const mounted = renderHook({
      initialSessionId: "session-42",
      currentSessionId: null,
      shouldHydrateMatchedInitialSession: false,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
      }),
    });

    await flushEffects();

    mounted.rerender({
      initialSessionId: "session-42",
      currentSessionId: "session-42",
      shouldHydrateMatchedInitialSession: true,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
      }),
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledTimes(1);
    expect(switchTopic).toHaveBeenNthCalledWith(1, "session-42", {
      allowDetachedSession: true,
    });
    await act(async () => {
      resolveSwitch("success");
    });
  });

  it("初始会话水合失败后不应自动重复恢复", async () => {
    let resolveSwitch: (result: unknown) => void = () => {};
    const switchTopic = vi.fn(
      () =>
        new Promise<unknown>((resolve) => {
          resolveSwitch = resolve;
        }),
    );
    const mounted = renderHook({
      initialSessionId: "session-invalid-import",
      currentSessionId: null,
      switchTopic,
    });

    await flushEffects();
    mounted.rerender({
      initialSessionId: "session-invalid-import",
      currentSessionId: "session-invalid-import",
      shouldHydrateMatchedInitialSession: true,
      switchTopic,
    });
    await flushEffects();
    expect(switchTopic).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveSwitch("error");
    });
    mounted.rerender({
      initialSessionId: "session-invalid-import",
      currentSessionId: null,
      switchTopic,
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledTimes(1);
  });

  it("切换到新的初始会话时应重新触发恢复", async () => {
    const switchTopic = vi.fn(async () => undefined);
    const mounted = renderHook({
      initialSessionId: "session-42",
      currentSessionId: null,
      switchTopic,
    });

    await flushEffects();

    mounted.rerender({
      initialSessionId: "session-99",
      currentSessionId: "session-42",
      switchTopic,
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenNthCalledWith(1, "session-42", undefined);
    expect(switchTopic).toHaveBeenNthCalledWith(2, "session-99", undefined);
  });

  it("应透传解析后的初始会话切换选项", async () => {
    const switchTopic = vi.fn(async () => undefined);

    renderHook({
      initialSessionId: "session-archived",
      currentSessionId: null,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
        forceRefresh: true,
        resumeSessionStartHooks: false,
      }),
    });

    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith("session-archived", {
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("等待会话解析期间不应提前切换", async () => {
    const switchTopic = vi.fn(async () => undefined);
    const mounted = renderHook({
      initialSessionId: "session-42",
      currentSessionId: null,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        waitForResolution: true,
      }),
    });

    await flushEffects();
    expect(switchTopic).not.toHaveBeenCalled();

    mounted.rerender({
      initialSessionId: "session-42",
      currentSessionId: null,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        forceRefresh: true,
        resumeSessionStartHooks: true,
      }),
    });

    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith("session-42", {
      forceRefresh: true,
      resumeSessionStartHooks: true,
    });
  });

  it("任务中心草稿 surface 暂停时不应触发 matched hydrate 抢回旧空会话", async () => {
    const switchTopic = vi.fn(async () => undefined);
    const mounted = renderHook({
      initialSessionId: "session-42",
      currentSessionId: "session-42",
      shouldPauseInitialSessionNavigation: true,
      shouldHydrateMatchedInitialSession: true,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
        forceRefresh: true,
      }),
    });

    await flushEffects();
    expect(switchTopic).not.toHaveBeenCalled();

    mounted.rerender({
      initialSessionId: "session-42",
      currentSessionId: "session-42",
      shouldPauseInitialSessionNavigation: false,
      shouldHydrateMatchedInitialSession: true,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
        forceRefresh: true,
      }),
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith("session-42", {
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("用户显式打开历史会话时不应被遗留草稿 surface 永久暂停", async () => {
    const switchTopic = vi.fn(async () => undefined);

    renderHook({
      initialSessionId: "session-from-history",
      currentSessionId: null,
      shouldPauseInitialSessionNavigation: true,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
        forceRefresh: true,
      }),
    });

    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith("session-from-history", {
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("新建任务过渡中旧 initialSessionId 未变化时应继续暂停恢复", async () => {
    const switchTopic = vi.fn(async () => undefined);
    const mounted = renderHook({
      initialSessionId: "session-current",
      currentSessionId: "session-current",
      shouldPauseInitialSessionNavigation: false,
      switchTopic,
    });

    await flushEffects();
    expect(switchTopic).not.toHaveBeenCalled();

    mounted.rerender({
      initialSessionId: "session-current",
      currentSessionId: null,
      shouldPauseInitialSessionNavigation: true,
      switchTopic,
    });
    await flushEffects();

    expect(switchTopic).not.toHaveBeenCalled();
  });

  it("用户从新建任务重新打开同一历史会话时应显式恢复", async () => {
    const switchTopic = vi.fn(async () => undefined);
    const mounted = renderHook({
      initialSessionId: "session-current",
      currentSessionId: "session-current",
      shouldPauseInitialSessionNavigation: false,
      switchTopic,
    });

    await flushEffects();
    expect(switchTopic).not.toHaveBeenCalled();

    mounted.rerender({
      initialSessionId: "session-current",
      currentSessionId: null,
      shouldPauseInitialSessionNavigation: true,
      switchTopic,
    });
    await flushEffects();
    expect(switchTopic).not.toHaveBeenCalled();

    act(() => {
      requestExplicitInitialSessionNavigation("session-current");
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith("session-current", undefined);
  });

  it("任务中心首发已切到新会话后不应恢复旧 initialSessionId", async () => {
    const switchTopic = vi.fn(async () => undefined);
    const mounted = renderHook({
      initialSessionId: "session-old",
      currentSessionId: "session-old",
      shouldCancelPausedInitialSessionNavigationOnCurrentSessionChange: true,
      shouldPauseInitialSessionNavigation: true,
      shouldHydrateMatchedInitialSession: true,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
        forceRefresh: true,
      }),
    });

    await flushEffects();
    expect(switchTopic).not.toHaveBeenCalled();

    mounted.rerender({
      initialSessionId: "session-old",
      currentSessionId: "session-new",
      shouldCancelPausedInitialSessionNavigationOnCurrentSessionChange: true,
      shouldPauseInitialSessionNavigation: false,
      shouldHydrateMatchedInitialSession: true,
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
        forceRefresh: true,
      }),
    });
    await flushEffects();

    expect(switchTopic).not.toHaveBeenCalled();
  });

  it("已水合 initialSessionId 后切入 detached session 不应被旧目标抢回", async () => {
    const switchTopic = vi.fn(async () => undefined);
    const mounted = renderHook({
      initialSessionId: "session-parent",
      currentSessionId: "session-parent",
      shouldHydrateMatchedInitialSession: true,
      switchTopic,
    });

    await flushEffects();
    expect(switchTopic).toHaveBeenCalledWith("session-parent", {
      forceRefresh: true,
    });

    switchTopic.mockClear();
    mounted.rerender({
      initialSessionId: "session-parent",
      currentSessionId: "session-child",
      shouldHydrateMatchedInitialSession: false,
      switchTopic,
    });
    await flushEffects();

    expect(switchTopic).not.toHaveBeenCalled();
  });

  it("短时间内重复挂载相同初始会话时应去重", async () => {
    const switchTopic = vi.fn(async () => undefined);

    renderHook({
      initialSessionId: "session-dedupe",
      currentSessionId: null,
      switchTopic,
    });
    await flushEffects();

    renderHook({
      initialSessionId: "session-dedupe",
      currentSessionId: null,
      switchTopic,
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledTimes(1);
    expect(switchTopic).toHaveBeenCalledWith("session-dedupe", undefined);
  });

  it("外部发送路径已登记导航时不应立即触发初始会话切换", async () => {
    const switchTopic = vi.fn(async () => undefined);

    rememberInitialSessionNavigationStart("session-materialized-send");
    renderHook({
      initialSessionId: "session-materialized-send",
      currentSessionId: null,
      shouldPauseInitialSessionNavigation: true,
      switchTopic,
    });
    await flushEffects();

    expect(switchTopic).not.toHaveBeenCalled();
  });

  it("外部发送路径已登记导航后不应在去重窗口后补发初始会话切换", async () => {
    const switchTopic = vi.fn(async () => undefined);
    const nowSpy = vi.spyOn(Date, "now");
    nowSpy.mockReturnValue(10_000);

    rememberInitialSessionNavigationStart("session-materialized-send");
    const mounted = renderHook({
      initialSessionId: "session-materialized-send",
      currentSessionId: null,
      switchTopic,
    });
    await flushEffects();
    expect(switchTopic).not.toHaveBeenCalled();

    nowSpy.mockReturnValue(13_000);
    mounted.rerender({
      initialSessionId: "session-materialized-send",
      currentSessionId: "previous-session",
      switchTopic,
      resolveInitialSessionSwitch: () => ({
        allowDetachedSession: true,
      }),
    });
    await flushEffects();

    expect(switchTopic).not.toHaveBeenCalled();
  });

  it("不同工作区实例恢复同一初始会话时不应互相去重", async () => {
    const firstSwitchTopic = vi.fn(async () => undefined);
    const secondSwitchTopic = vi.fn(async () => undefined);

    renderHook({
      initialSessionId: "session-shared",
      currentSessionId: "session-current-a",
      switchTopic: firstSwitchTopic,
    });
    await flushEffects();

    renderHook({
      initialSessionId: "session-shared",
      currentSessionId: "session-current-b",
      switchTopic: secondSwitchTopic,
    });
    await flushEffects();

    expect(firstSwitchTopic).toHaveBeenCalledWith("session-shared", undefined);
    expect(secondSwitchTopic).toHaveBeenCalledWith("session-shared", undefined);
  });
});
