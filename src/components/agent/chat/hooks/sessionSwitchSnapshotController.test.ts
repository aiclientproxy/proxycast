import { describe, expect, it } from "vitest";
import {
  buildPendingSessionShellMetricContext,
  buildSessionSwitchCachedSnapshotPlan,
  buildSessionSwitchDeferHydrationPlan,
  buildSessionSwitchPendingShellPlan,
  buildSessionSwitchStartStatePlan,
  buildSessionSwitchDeferHydrationMetricContext,
  buildSessionSwitchLocalSnapshotOverride,
  buildSessionSwitchStartMetricContext,
  shouldApplyCachedTopicSnapshot,
  shouldApplyPendingSessionShell,
  shouldLoadCachedTopicSnapshot,
  shouldRefreshCachedSnapshotImmediately,
  shouldReuseHydratedCurrentSession,
  shouldReusePendingCurrentSessionHydration,
  shouldReuseActiveSessionSwitch,
} from "./sessionSwitchSnapshotController";
import type { AgentSessionCachedSnapshot } from "./agentSessionScopedStorage";
import type { AgentThreadItem, AgentThreadTurn, Message } from "../types";

function message(id: string): Message {
  return {
    id,
    role: "assistant",
    content: id,
    timestamp: new Date("2026-05-05T00:00:00.000Z"),
  };
}

function snapshot(
  messages: Message[] = [message("message-a")],
): AgentSessionCachedSnapshot {
  return {
    messages,
    threadTurns: [{ id: "turn-a" }] as AgentThreadTurn[],
    threadItems: [{ id: "item-a", turn_id: "turn-a" }] as AgentThreadItem[],
    currentTurnId: null,
    cacheMetadata: {
      storageKind: "transient",
      freshness: "fresh",
      updatedAt: 1,
      lastAccessedAt: 1,
      expiresAt: 2,
      staleUntil: 3,
      sessionUpdatedAt: 1,
      messagesCount: messages.length,
      historyTruncated: false,
    },
  };
}

describe("sessionSwitchSnapshotController", () => {
  it("仅 canonical detail 已应用时才复用当前会话", () => {
    expect(
      shouldReuseHydratedCurrentSession({
        currentSessionId: "topic-a",
        hydratedSessionId: null,
        messagesCount: 2,
        topicId: "topic-a",
      }),
    ).toBe(false);
    expect(
      shouldReuseHydratedCurrentSession({
        currentSessionId: "topic-a",
        hydratedSessionId: "topic-a",
        messagesCount: 2,
        topicId: "topic-a",
      }),
    ).toBe(true);
    expect(
      shouldReuseHydratedCurrentSession({
        currentSessionId: "topic-a",
        forceRefresh: true,
        hydratedSessionId: "topic-a",
        messagesCount: 2,
        topicId: "topic-a",
      }),
    ).toBe(false);
  });

  it("当前会话已在后台水合时应复用 pending 请求", () => {
    expect(
      shouldReusePendingCurrentSessionHydration({
        currentSessionId: "topic-a",
        pendingSessionId: "topic-a",
        topicId: "topic-a",
      }),
    ).toBe(true);
    expect(
      shouldReusePendingCurrentSessionHydration({
        currentSessionId: "topic-a",
        forceRefresh: true,
        pendingSessionId: "topic-a",
        topicId: "topic-a",
      }),
    ).toBe(false);
    expect(
      shouldReusePendingCurrentSessionHydration({
        currentSessionId: "topic-a",
        pendingSessionId: "topic-b",
        topicId: "topic-a",
      }),
    ).toBe(false);
  });

  it("当前会话不应重复加载或应用 cached snapshot", () => {
    expect(
      shouldLoadCachedTopicSnapshot({
        currentSessionId: "topic-a",
        topicId: "topic-a",
      }),
    ).toBe(false);
    expect(
      shouldApplyCachedTopicSnapshot({
        currentSessionId: "topic-a",
        topicId: "topic-a",
      }),
    ).toBe(false);
  });

  it("stale / running / waiting 话题应立即刷新 cached snapshot", () => {
    expect(
      shouldRefreshCachedSnapshotImmediately({
        cacheFreshness: "stale",
        topicStatus: "idle",
      }),
    ).toBe(true);
    expect(
      shouldRefreshCachedSnapshotImmediately({
        cacheFreshness: "fresh",
        topicStatus: "running",
      }),
    ).toBe(true);
    expect(
      shouldRefreshCachedSnapshotImmediately({
        cacheFreshness: "fresh",
        topicStatus: "completed",
      }),
    ).toBe(false);
  });

  it("无 cached snapshot 且切换到其他会话时应先应用 pending shell", () => {
    expect(
      shouldApplyPendingSessionShell({
        currentSessionId: "topic-a",
        topicId: "topic-b",
        cachedSnapshot: null,
      }),
    ).toBe(true);
    expect(
      shouldApplyPendingSessionShell({
        currentSessionId: "topic-a",
        topicId: "topic-b",
        cachedSnapshot: snapshot(),
      }),
    ).toBe(false);
  });

  it("仅普通同 topic 切换请求才应复用 in-flight switch", () => {
    expect(
      shouldReuseActiveSessionSwitch({
        activeTopicId: "topic-a",
        topicId: "topic-a",
      }),
    ).toBe(true);
    expect(
      shouldReuseActiveSessionSwitch({
        activeTopicId: "topic-b",
        topicId: "topic-a",
      }),
    ).toBe(false);
    expect(
      shouldReuseActiveSessionSwitch({
        activeTopicId: "topic-a",
        forceRefresh: true,
        topicId: "topic-a",
      }),
    ).toBe(false);
    expect(
      shouldReuseActiveSessionSwitch({
        activeTopicId: "topic-a",
        restoreSource: "auto",
        topicId: "topic-a",
      }),
    ).toBe(false);
    expect(
      shouldReuseActiveSessionSwitch({
        activeTopicId: "topic-a",
        resumeSessionStartHooks: true,
        topicId: "topic-a",
      }),
    ).toBe(false);
    expect(
      shouldReuseActiveSessionSwitch({
        activeTopicId: "topic-a",
        allowDetachedSession: true,
        topicId: "topic-a",
      }),
    ).toBe(false);
  });

  it("应把 switchTopic 开始阶段状态重置计划收敛为纯决策", () => {
    expect(
      buildSessionSwitchStartStatePlan({
        currentSessionId: "topic-a",
        topicId: "topic-b",
      }),
    ).toEqual({
      currentSessionIdToPersist: "topic-a",
      detachedSessionId: null,
      shouldClearAutoRestoringSession: true,
      shouldResetSessionHydrating: true,
    });
    expect(
      buildSessionSwitchStartStatePlan({
        allowDetachedSession: true,
        currentSessionId: null,
        restoreSource: "auto",
        topicId: "topic-b",
      }),
    ).toEqual({
      currentSessionIdToPersist: null,
      detachedSessionId: "topic-b",
      shouldClearAutoRestoringSession: false,
      shouldResetSessionHydrating: true,
    });
  });

  it("应构造 switchTopic cached snapshot 策略计划", () => {
    expect(
      buildSessionSwitchCachedSnapshotPlan({
        cachedSnapshot: snapshot(),
        currentSessionId: "topic-a",
        topicId: "topic-b",
        topicStatus: "running",
      }),
    ).toEqual({
      shouldApplyCachedSnapshot: true,
      shouldLoadCachedSnapshot: true,
      shouldRefreshCachedSnapshotImmediately: true,
    });
    expect(
      buildSessionSwitchCachedSnapshotPlan({
        cachedSnapshot: snapshot(),
        currentSessionId: "topic-a",
        forceRefresh: true,
        topicId: "topic-b",
        topicStatus: "completed",
      }),
    ).toEqual({
      shouldApplyCachedSnapshot: true,
      shouldLoadCachedSnapshot: false,
      shouldRefreshCachedSnapshotImmediately: false,
    });
    expect(
      buildSessionSwitchCachedSnapshotPlan({
        currentSessionId: "topic-a",
        topicId: "topic-a",
      }),
    ).toEqual({
      shouldApplyCachedSnapshot: false,
      shouldLoadCachedSnapshot: false,
      shouldRefreshCachedSnapshotImmediately: false,
    });
  });

  it("应构造 switchTopic defer hydration 状态计划", () => {
    expect(
      buildSessionSwitchDeferHydrationPlan({
        refreshCachedSnapshotImmediately: true,
        topicId: "topic-a",
      }),
    ).toEqual({
      detailLoadMode: "direct",
      restoreCandidateSessionId: "topic-a",
      shouldApplyCachedTopicChromeState: true,
      shouldClearAutoRestoringSession: true,
      shouldResetSessionHydrating: true,
    });
    expect(
      buildSessionSwitchDeferHydrationPlan({
        refreshCachedSnapshotImmediately: false,
        topicId: "topic-a",
      }).detailLoadMode,
    ).toBe("deferred");
  });

  it("应构造 switchTopic pending shell 状态计划", () => {
    expect(
      buildSessionSwitchPendingShellPlan({
        topicId: "topic-a",
      }),
    ).toEqual({
      restoreCandidateSessionId: "topic-a",
      sessionId: "topic-a",
      shouldApplyCachedTopicChromeState: true,
      shouldApplyEmptySessionSnapshot: true,
      shouldResetHistoryWindow: true,
      shouldSetSessionHydrating: true,
    });
  });

  it("应构造 switch start / defer / pending shell 指标上下文", () => {
    const cachedSnapshot = snapshot([
      message("message-a"),
      message("message-b"),
    ]);

    expect(
      buildSessionSwitchStartMetricContext({
        cachedSnapshot,
        currentSessionId: "topic-a",
        messagesCount: 7,
        refreshCachedSnapshotImmediately: true,
        topicId: "topic-b",
        workspaceId: "workspace-a",
      }),
    ).toMatchObject({
      cacheFreshness: "fresh",
      cacheStorageKind: "transient",
      cachedLocalMessagesCount: 2,
      currentSessionId: "topic-a",
      messagesCount: 7,
      refreshCachedSnapshotImmediately: true,
      sessionId: "topic-b",
      workspaceId: "workspace-a",
    });
    expect(
      buildSessionSwitchDeferHydrationMetricContext({
        cachedSnapshot,
        currentSessionId: "topic-a",
        refreshImmediately: false,
        topicId: "topic-b",
      }),
    ).toMatchObject({
      cachedLocalMessagesCount: 2,
      refreshImmediately: false,
      sessionId: "topic-b",
    });
    expect(
      buildPendingSessionShellMetricContext({
        currentSessionId: "topic-a",
        topicId: "topic-b",
      }),
    ).toEqual({
      currentSessionId: "topic-a",
      sessionId: "topic-b",
      topicId: "topic-b",
      workspaceId: undefined,
    });
  });

  it("仅当前 UI 仍持有同一 cached snapshot 引用时才返回 localSnapshotOverride", () => {
    const cachedSnapshot = snapshot();

    expect(
      buildSessionSwitchLocalSnapshotOverride({
        cachedSnapshot,
        currentSessionId: "topic-a",
        messages: cachedSnapshot.messages,
        threadTurns: cachedSnapshot.threadTurns,
        threadItems: cachedSnapshot.threadItems,
        topicId: "topic-a",
      }),
    ).toEqual({
      sessionId: "topic-a",
      messages: cachedSnapshot.messages,
      threadTurns: cachedSnapshot.threadTurns,
      threadItems: cachedSnapshot.threadItems,
    });
    expect(
      buildSessionSwitchLocalSnapshotOverride({
        cachedSnapshot,
        currentSessionId: "topic-a",
        messages: cachedSnapshot.messages.slice(),
        threadTurns: cachedSnapshot.threadTurns,
        threadItems: cachedSnapshot.threadItems,
        topicId: "topic-a",
      }),
    ).toBeNull();
  });
});
