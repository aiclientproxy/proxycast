import { act } from "react";
import { describe, expect, it, vi } from "vitest";

const threadRevertMocks = vi.hoisted(() => ({
  revertThreadHistory: vi.fn(async () => ({
    threadId: "thread-revert",
    turnsBackwardsCursor: null,
    itemsBackwardsCursor: null,
  })),
}));

vi.mock("@/lib/api/threadRevert", () => threadRevertMocks);

import {
  buildHeavySessionRuntimeFixture,
  createBaseParams,
  getRenderedSceneProps,
  renderHook,
} from "./useWorkspaceConversationSceneRuntime.testFixtures";

describe("useWorkspaceConversationSceneRuntime message list projection", () => {
  it("恢复历史后应等待同一 canonical Thread 的 read model 刷新", async () => {
    const onRefreshSessionReadModel = vi.fn(async () => true);
    const sceneProps = getRenderedSceneProps(
      createBaseParams({
        threadRead: {
          thread_id: "thread-revert",
          can_accept_direct_input: true,
        },
        onRefreshSessionReadModel,
      }),
    );

    await sceneProps.messageListProps.onRevertThread?.({
      threadId: "thread-revert",
      beforeTurnId: "turn-revert",
    });

    expect(threadRevertMocks.revertThreadHistory).toHaveBeenCalledWith({
      threadId: "thread-revert",
      beforeTurnId: "turn-revert",
    });
    expect(onRefreshSessionReadModel).toHaveBeenCalledWith({
      timelineMode: "replace",
    });
  });

  it("会话切换后应拒绝旧 Thread 的恢复请求", async () => {
    const sceneProps = getRenderedSceneProps(
      createBaseParams({
        threadRead: {
          thread_id: "thread-current",
          can_accept_direct_input: true,
        },
      }),
    );

    await expect(
      sceneProps.messageListProps.onRevertThread?.({
        threadId: "thread-stale",
        beforeTurnId: "turn-stale",
      }),
    ).rejects.toThrow("no longer active");
  });

  it("应将 canonical child roster 透传给消息列表", () => {
    const canonicalChildren = [
      {
        name: "实现",
        parentThreadId: "thread-parent",
        sessionId: "session-child",
        status: "running" as const,
        threadId: "thread-child",
        updatedAtMs: 1,
      },
    ];
    const sceneProps = getRenderedSceneProps(
      createBaseParams({ canonicalChildren }),
    );

    expect(sceneProps.messageListProps.canonicalChildren).toBe(
      canonicalChildren,
    );
  });

  it("messageListRuntime 应作为消息列表 current 契约来源", () => {
    const legacyFlatMessages = [
      {
        id: "legacy-flat-message",
        role: "assistant",
        content: "不应进入当前消息列表",
      },
    ];
    const runtimeMessages = [
      {
        id: "runtime-message",
        role: "assistant",
        content: "来自 current messageListRuntime",
      },
    ];
    const params = createBaseParams({
      displayMessages: legacyFlatMessages,
      messageListRuntime: {
        messages: runtimeMessages,
      },
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.messageListProps.messages).toBe(runtimeMessages);
    expect(sceneProps.taskRail?.messages).toBe(runtimeMessages);
  });

  it("应把 URL 来源预览入口透传给消息列表", () => {
    const handleOpenUrlPreview = vi.fn();
    const params = createBaseParams({
      handleOpenUrlPreview,
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.messageListProps.onOpenUrlPreview).toBe(
      handleOpenUrlPreview,
    );
  });

  it("恢复旧会话首帧应先透传消息，并延迟运行轨迹投影", () => {
    vi.useFakeTimers();
    const { messages, turns, threadItems } =
      buildHeavySessionRuntimeFixture("restore");
    const params = createBaseParams({
      displayMessages: messages,
      turns,
      currentTurnId: "restore-turn-5",
      effectiveThreadItems: threadItems,
      pendingActions: [
        {
          requestId: "req-1",
          actionType: "elicitation",
          prompt: "补充信息",
          status: "pending",
        },
      ],
      isRestoringSession: true,
    });

    const harness = renderHook(params);
    let sceneProps = (harness.getValue().mainAreaNode as any).props;

    expect(sceneProps.messageListProps.messages).toBe(messages);
    expect(sceneProps.messageListProps.turns).toEqual([]);
    expect(sceneProps.messageListProps.threadItems).toEqual([]);
    expect(sceneProps.messageListProps.currentTurnId).toBeNull();
    expect(sceneProps.messageListProps.pendingActions).toEqual([]);
    expect(sceneProps.canvasWorkbenchLayoutProps.sessionView).toBeNull();

    act(() => {
      vi.advanceTimersByTime(700);
    });

    sceneProps = (harness.getValue().mainAreaNode as any).props;
    expect(sceneProps.messageListProps.turns).toBe(turns);
    expect(sceneProps.messageListProps.threadItems).toBe(threadItems);
    expect(sceneProps.messageListProps.currentTurnId).toBe("restore-turn-5");
    expect(sceneProps.messageListProps.pendingActions).toHaveLength(1);
    vi.useRealTimers();
  });

  it("历史窗口 hydrate 完成后应直接透传消息和运行轨迹投影", () => {
    const { messages, turns, threadItems } =
      buildHeavySessionRuntimeFixture("history-window");
    const params = createBaseParams({
      displayMessages: messages,
      turns,
      currentTurnId: "history-window-turn-5",
      effectiveThreadItems: threadItems,
      isRestoringSession: false,
      sessionHistoryWindow: {
        loadedMessages: 40,
        totalMessages: 320,
        isLoadingFull: false,
        error: null,
      },
    });

    const harness = renderHook(params);
    const sceneProps = (harness.getValue().mainAreaNode as any).props;

    expect(sceneProps.messageListProps.messages).toBe(messages);
    expect(sceneProps.messageListProps.turns).toBe(turns);
    expect(sceneProps.messageListProps.threadItems).toBe(threadItems);
    expect(sceneProps.messageListProps.currentTurnId).toBe(
      "history-window-turn-5",
    );
  });

  it("发送中会话不应延迟运行轨迹投影", () => {
    vi.useFakeTimers();
    const { messages, turns, threadItems } =
      buildHeavySessionRuntimeFixture("sending");
    const params = createBaseParams({
      displayMessages: messages,
      turns,
      currentTurnId: "sending-turn-5",
      effectiveThreadItems: threadItems,
      isRestoringSession: true,
      isSending: true,
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.messageListProps.turns).toBe(turns);
    expect(sceneProps.messageListProps.threadItems).toBe(threadItems);
    expect(sceneProps.messageListProps.currentTurnId).toBe("sending-turn-5");
    vi.useRealTimers();
  });

  it("聚焦 timeline 或存在 A2UI 表单时不应延迟运行轨迹投影", () => {
    const { messages, turns, threadItems } =
      buildHeavySessionRuntimeFixture("interactive");
    const focusedSceneProps = getRenderedSceneProps(
      createBaseParams({
        displayMessages: messages,
        turns,
        currentTurnId: "interactive-turn-5",
        effectiveThreadItems: threadItems,
        isRestoringSession: true,
        focusedTimelineItemId: "interactive-item-1",
      }),
    );
    expect(focusedSceneProps.messageListProps.turns).toBe(turns);
    expect(focusedSceneProps.messageListProps.threadItems).toBe(threadItems);

    const pendingA2UISceneProps = getRenderedSceneProps(
      createBaseParams({
        displayMessages: messages,
        turns,
        currentTurnId: "interactive-turn-5",
        effectiveThreadItems: threadItems,
        isRestoringSession: true,
        pendingA2UIForm: {
          id: "form-1",
          title: "补充信息",
          schema: {},
        },
      }),
    );
    expect(pendingA2UISceneProps.messageListProps.turns).toBe(turns);
    expect(pendingA2UISceneProps.messageListProps.threadItems).toBe(
      threadItems,
    );
  });

  it("切换到另一条同长度旧会话时应重新延迟运行轨迹投影", () => {
    vi.useFakeTimers();
    const buildSession = (sessionId: string) => {
      const messages = Array.from({ length: 24 }, (_, index) => ({
        id: `${sessionId}-msg-${index}`,
        role: index % 2 === 0 ? "user" : "assistant",
        content: `${sessionId} 消息 ${index}`,
        timestamp: new Date(2026, 3, 30, 11, index),
      }));
      const turns = Array.from({ length: 6 }, (_, index) => ({
        id: `${sessionId}-turn-${index}`,
        thread_id: `${sessionId}-thread`,
        prompt_text: `${sessionId} 任务 ${index}`,
        status: "completed",
        started_at: `2026-04-30T11:0${index}:00.000Z`,
        created_at: `2026-04-30T11:0${index}:00.000Z`,
        updated_at: `2026-04-30T11:0${index}:01.000Z`,
      }));
      const threadItems = Array.from({ length: 28 }, (_, index) => ({
        id: `${sessionId}-item-${index}`,
        thread_id: `${sessionId}-thread`,
        turn_id: `${sessionId}-turn-${Math.min(5, Math.floor(index / 5))}`,
        sequence: index + 1,
        status: "completed",
        started_at: `2026-04-30T11:00:${String(index).padStart(2, "0")}.000Z`,
        updated_at: `2026-04-30T11:00:${String(index).padStart(2, "0")}.500Z`,
        type: "tool_call",
        tool_name: "Read",
        arguments: { index },
      }));

      return { messages, turns, threadItems };
    };
    const sessionA = buildSession("session-a");
    const sessionB = buildSession("session-b");
    const buildParams = (
      sessionId: string,
      session: ReturnType<typeof buildSession>,
    ) =>
      createBaseParams({
        sessionId,
        displayMessages: session.messages,
        turns: session.turns,
        currentTurnId: session.turns.at(-1)?.id ?? null,
        effectiveThreadItems: session.threadItems,
        isRestoringSession: true,
      });

    const harness = renderHook(buildParams("session-a", sessionA));
    act(() => {
      vi.advanceTimersByTime(700);
    });

    let sceneProps = (harness.getValue().mainAreaNode as any).props;
    expect(sceneProps.messageListProps.turns).toBe(sessionA.turns);
    expect(sceneProps.messageListProps.threadItems).toBe(sessionA.threadItems);

    harness.render(buildParams("session-b", sessionB));
    sceneProps = (harness.getValue().mainAreaNode as any).props;

    expect(sceneProps.messageListProps.messages).toBe(sessionB.messages);
    expect(sceneProps.messageListProps.turns).toEqual([]);
    expect(sceneProps.messageListProps.threadItems).toEqual([]);
    expect(sceneProps.messageListProps.currentTurnId).toBeNull();

    act(() => {
      vi.advanceTimersByTime(700);
    });

    sceneProps = (harness.getValue().mainAreaNode as any).props;
    expect(sceneProps.messageListProps.turns).toBe(sessionB.turns);
    expect(sceneProps.messageListProps.threadItems).toBe(sessionB.threadItems);
    expect(sceneProps.messageListProps.currentTurnId).toBe("session-b-turn-5");
    vi.useRealTimers();
  });
});
