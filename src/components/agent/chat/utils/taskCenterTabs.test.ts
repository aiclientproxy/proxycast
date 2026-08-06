import { describe, expect, it } from "vitest";
import type { Topic } from "../hooks/agentChatShared";
import {
  applyTaskCenterRouteTabSyncToMap,
  areTaskCenterTabIdsEqual,
  clearTaskCenterLocalSessionOverrideForTopic,
  clearTaskCenterTransitionTopicForTopic,
  initializeTaskCenterOpenTabMap,
  isTaskCenterTopicSwitchPending,
  normalizeTaskCenterWorkspaceTabMap,
  reconcileTaskCenterTabIds,
  resolveTaskCenterTopicClosePlan,
  resolveTaskCenterTopicSwitchOptions,
  resolveTaskCenterReconcileCurrentTopicId,
  resolveInitialTaskSessionSwitchOptions,
  replaceTaskCenterTabIdsForWorkspace,
  rollbackTaskCenterOpenTabMapForFailedSwitch,
  resolveTaskCenterFallbackRestorePlan,
  resolveTaskCenterFallbackTopicId,
  shouldSkipTaskCenterActiveTopicReopen,
  shouldHideTaskCenterTabsForDetachedSession,
  resolveTaskCenterPreviewTopicId,
  resolveTaskCenterRouteTabSyncIntent,
  resolveTaskCenterVisibleTabIds,
  resolveTaskCenterTabIdsForWorkspace,
  shouldRespectTaskCenterLocalSessionOverride,
  shouldResumeTaskSession,
  shouldWaitForTaskCenterInitialSessionTopic,
  updateTaskCenterTabIdsForWorkspace,
} from "./taskCenterTabs";

function createTopic(id: string, overrides?: Partial<Topic>): Topic {
  return {
    id,
    title: id,
    createdAt: new Date("2026-04-20T00:00:00.000Z"),
    updatedAt: new Date("2026-04-20T00:00:00.000Z"),
    messagesCount: 1,
    executionStrategy: "react",
    status: "done",
    statusReason: "default",
    lastPreview: `${id} preview`,
    hasUnread: false,
    sourceSessionId: id,
    ...overrides,
  };
}

function createFallbackRestoreParams(
  overrides: Partial<
    Parameters<typeof resolveTaskCenterFallbackRestorePlan>[0]
  > = {},
): Parameters<typeof resolveTaskCenterFallbackRestorePlan>[0] {
  return {
    agentEntry: "claw",
    workspaceId: "workspace-test",
    isAutoRestoringSession: false,
    isSessionHydrating: false,
    draftSurfaceActive: false,
    draftTabActive: false,
    initialPendingServiceSkillLaunchSignature: null,
    initialDispatchKey: null,
    isBootstrapDispatchPending: false,
    isHomeSessionBackgroundRecovery: false,
    messagesLength: 1,
    isSending: false,
    queuedTurnsLength: 0,
    shouldHideDetachedTaskCenterTabs: false,
    normalizedInitialSessionId: null,
    sessionId: null,
    currentSessionIsKnownTopic: false,
    hasDisplayMessages: false,
    switchingTopicId: null,
    openTabIds: ["topic-old"],
    topics: [createTopic("topic-old")],
    previousRestore: null,
    now: 2_026,
    ...overrides,
  };
}

describe("taskCenterTabs", () => {
  it("reconcile 应过滤失效 id 并把当前任务前置", () => {
    const topics = [
      createTopic("topic-a"),
      createTopic("topic-b", { status: "running" }),
      createTopic("topic-c"),
    ];

    expect(
      reconcileTaskCenterTabIds({
        existingIds: ["topic-a", "missing", "topic-c"],
        topics,
        currentTopicId: "topic-b",
      }),
    ).toEqual(["topic-b", "topic-a", "topic-c"]);
  });

  it("reconcile 在没有历史标签时应保持为空，避免自动打开项目历史会话", () => {
    const topics = [
      createTopic("topic-a", { status: "waiting" }),
      createTopic("topic-b"),
    ];

    expect(
      reconcileTaskCenterTabIds({
        existingIds: [],
        topics,
        currentTopicId: null,
      }),
    ).toEqual([]);
  });

  it("reconcile 没有历史标签但存在当前任务时，只打开当前任务", () => {
    const topics = [
      createTopic("topic-a", { status: "waiting" }),
      createTopic("topic-b"),
    ];

    expect(
      reconcileTaskCenterTabIds({
        existingIds: [],
        topics,
        currentTopicId: "topic-b",
      }),
    ).toEqual(["topic-b"]);
  });

  it("应按 workspace 读取和更新标签列表", () => {
    const currentMap = {
      "workspace-a": ["topic-a", "topic-b"],
      "workspace-b": ["topic-c"],
    };

    expect(
      resolveTaskCenterTabIdsForWorkspace(currentMap, "workspace-a"),
    ).toEqual(["topic-a", "topic-b"]);

    expect(
      updateTaskCenterTabIdsForWorkspace(
        currentMap,
        "workspace-a",
        (currentIds) => ["topic-d", ...currentIds],
      ),
    ).toEqual({
      "workspace-a": ["topic-d", "topic-a", "topic-b"],
      "workspace-b": ["topic-c"],
    });
  });

  it("standalone 会话应使用 legacy bucket 保存任务标签", () => {
    const currentMap = {
      "__legacy__": ["topic-standalone-a"],
      "workspace-a": ["topic-project"],
    };

    expect(resolveTaskCenterTabIdsForWorkspace(currentMap, null)).toEqual([
      "topic-standalone-a",
    ]);

    expect(
      updateTaskCenterTabIdsForWorkspace(
        currentMap,
        null,
        (currentIds) => ["topic-standalone-b", ...currentIds],
      ),
    ).toEqual({
      "__legacy__": ["topic-standalone-b", "topic-standalone-a"],
      "workspace-a": ["topic-project"],
    });
  });

  it("导航栏打开单个任务时，应覆盖当前 workspace 的旧多标签状态", () => {
    const currentMap = {
      "workspace-a": ["topic-a", "topic-b", "topic-c"],
      "workspace-b": ["topic-d"],
    };

    expect(
      replaceTaskCenterTabIdsForWorkspace(
        currentMap,
        "workspace-a",
        "topic-selected",
      ),
    ).toEqual({
      "workspace-a": ["topic-selected"],
      "workspace-b": ["topic-d"],
    });
  });

  it("初始化任务中心时应在 task center 路由会话下覆盖当前 workspace 标签", () => {
    const currentMap = {
      "workspace-a": ["topic-a", "topic-b"],
      "workspace-b": ["topic-c"],
    };

    expect(
      initializeTaskCenterOpenTabMap({
        initialTabMap: currentMap,
        agentEntry: "claw",
        workspaceId: "workspace-a",
        normalizedInitialSessionId: "topic-selected",
      }),
    ).toEqual({
      "workspace-a": ["topic-selected"],
      "workspace-b": ["topic-c"],
    });

    expect(
      initializeTaskCenterOpenTabMap({
        initialTabMap: currentMap,
        agentEntry: "new-task",
        workspaceId: "workspace-a",
        normalizedInitialSessionId: "topic-selected",
      }),
    ).toEqual({
      "workspace-a": ["topic-selected"],
      "workspace-b": ["topic-c"],
    });
  });

  it("初始化任务中心时 standalone 路由会话应进入 legacy 标签桶", () => {
    const currentMap = {
      "workspace-a": ["topic-a"],
    };

    expect(
      initializeTaskCenterOpenTabMap({
        initialTabMap: currentMap,
        agentEntry: "claw",
        workspaceId: null,
        normalizedInitialSessionId: "topic-imported",
      }),
    ).toEqual({
      "__legacy__": ["topic-imported"],
      "workspace-a": ["topic-a"],
    });
  });

  it("路由同步应区分外部直达覆盖与任务中心本地切换追平", () => {
    const currentMap = {
      "workspace-a": ["topic-current", "topic-old"],
      "workspace-b": ["topic-other"],
    };

    expect(
      resolveTaskCenterRouteTabSyncIntent({
        agentEntry: "claw",
        workspaceId: "workspace-a",
        normalizedInitialSessionId: "topic-selected",
        lastSyncedInitialSessionId: "topic-current",
        shouldRespectLocalSession: false,
      }),
    ).toEqual({
      shouldSync: true,
      routeChanged: true,
      shouldClearActiveDraft: true,
      shouldClearTransitionAndDetached: false,
      nextRouteSyncSessionId: "topic-selected",
    });

    expect(
      applyTaskCenterRouteTabSyncToMap({
        currentMap,
        workspaceId: "workspace-a",
        normalizedInitialSessionId: "topic-selected",
        shouldRespectLocalSession: false,
      }),
    ).toEqual({
      "workspace-a": ["topic-selected"],
      "workspace-b": ["topic-other"],
    });

    expect(
      resolveTaskCenterRouteTabSyncIntent({
        agentEntry: "claw",
        workspaceId: "workspace-a",
        normalizedInitialSessionId: "topic-selected",
        lastSyncedInitialSessionId: "topic-selected",
        shouldRespectLocalSession: true,
      }),
    ).toMatchObject({
      shouldSync: true,
      routeChanged: false,
      shouldClearActiveDraft: true,
      shouldClearTransitionAndDetached: true,
    });

    expect(
      applyTaskCenterRouteTabSyncToMap({
        currentMap,
        workspaceId: "workspace-a",
        normalizedInitialSessionId: "topic-selected",
        shouldRespectLocalSession: true,
      }),
    ).toEqual({
      "workspace-a": ["topic-selected", "topic-current", "topic-old"],
      "workspace-b": ["topic-other"],
    });
  });

  it("standalone 路由同步应覆盖 legacy 标签桶", () => {
    const currentMap = {
      "__legacy__": ["topic-old"],
      "workspace-a": ["topic-project"],
    };

    expect(
      resolveTaskCenterRouteTabSyncIntent({
        agentEntry: "claw",
        workspaceId: null,
        normalizedInitialSessionId: "topic-imported",
        lastSyncedInitialSessionId: null,
        shouldRespectLocalSession: false,
      }),
    ).toMatchObject({
      shouldSync: true,
      routeChanged: true,
      shouldClearActiveDraft: true,
    });

    expect(
      applyTaskCenterRouteTabSyncToMap({
        currentMap,
        workspaceId: null,
        normalizedInitialSessionId: "topic-imported",
        shouldRespectLocalSession: false,
      }),
    ).toEqual({
      "__legacy__": ["topic-imported"],
      "workspace-a": ["topic-project"],
    });
  });

  it("new-task 路由同步也应维护 task center 标签", () => {
    expect(
      resolveTaskCenterRouteTabSyncIntent({
        agentEntry: "new-task",
        workspaceId: "workspace-a",
        normalizedInitialSessionId: "topic-selected",
        lastSyncedInitialSessionId: null,
        shouldRespectLocalSession: false,
      }),
    ).toMatchObject({
      shouldSync: true,
      routeChanged: true,
      shouldClearActiveDraft: true,
    });
  });

  it("任务中心本地会话覆盖只应在路由或当前会话匹配时生效", () => {
    expect(
      shouldRespectTaskCenterLocalSessionOverride({
        localSessionOverride: {
          sessionId: "topic-a",
          routeSessionId: "topic-current",
        },
        normalizedInitialSessionId: "topic-a",
        sessionId: "topic-a",
      }),
    ).toBe(true);

    expect(
      shouldRespectTaskCenterLocalSessionOverride({
        localSessionOverride: {
          sessionId: "topic-a",
          routeSessionId: "topic-current",
        },
        normalizedInitialSessionId: "topic-b",
        sessionId: "topic-b",
      }),
    ).toBe(false);
  });

  it("应兼容旧的全局数组存储并迁移到当前 workspace", () => {
    expect(
      normalizeTaskCenterWorkspaceTabMap(
        ["topic-a", "title-gen-1", "topic-b"],
        {
          workspaceId: "workspace-a",
        },
      ),
    ).toEqual({
      "workspace-a": ["topic-a", "topic-b"],
    });

    expect(
      updateTaskCenterTabIdsForWorkspace(
        {
          __legacy__: ["topic-a", "topic-b"],
        },
        "workspace-b",
        (currentIds) => currentIds,
      ),
    ).toEqual({
      "workspace-b": ["topic-a", "topic-b"],
    });
  });

  it("应正确识别需要恢复 start hooks 的任务", () => {
    expect(
      shouldResumeTaskSession({
        status: "running",
        statusReason: "default",
      }),
    ).toBe(true);
    expect(
      shouldResumeTaskSession({
        status: "waiting",
        statusReason: "user_action",
      }),
    ).toBe(true);
    expect(
      shouldResumeTaskSession({
        status: "failed",
        statusReason: "workspace_error",
      }),
    ).toBe(true);
    expect(
      shouldResumeTaskSession({
        status: "done",
        statusReason: "default",
      }),
    ).toBe(false);
  });

  it("打开任务会话时应把恢复态与强刷选项收敛成 switch 参数", () => {
    expect(
      resolveTaskCenterTopicSwitchOptions({
        shouldResume: false,
        forceRefresh: false,
      }),
    ).toBeUndefined();

    expect(
      resolveTaskCenterTopicSwitchOptions({
        shouldResume: true,
        forceRefresh: true,
      }),
    ).toEqual({
      forceRefresh: true,
      resumeSessionStartHooks: true,
    });

    expect(
      resolveTaskCenterTopicSwitchOptions({
        allowDetachedSession: true,
        shouldResume: false,
        forceRefresh: false,
      }),
    ).toEqual({
      allowDetachedSession: true,
    });
  });

  it("重复打开当前活跃任务时应跳过 switch，避免清空已渲染消息", () => {
    expect(
      shouldSkipTaskCenterActiveTopicReopen({
        topicId: "topic-current",
        activeSessionId: "topic-current",
        messagesLength: 1,
        activeDraftTabId: null,
        draftSurfaceActive: false,
        detachedTopicId: null,
        shouldResume: false,
      }),
    ).toBe(true);

    expect(
      shouldSkipTaskCenterActiveTopicReopen({
        topicId: "topic-current",
        activeSessionId: "topic-current",
        messagesLength: 1,
        activeDraftTabId: null,
        draftSurfaceActive: false,
        detachedTopicId: null,
        shouldResume: false,
        preferResume: true,
      }),
    ).toBe(false);
  });

  it("切换失败回滚时只移除本次临时打开的任务标签和本地覆盖", () => {
    const currentMap = {
      "workspace-a": ["topic-new", "topic-old"],
      "workspace-b": ["topic-other"],
    };

    expect(
      rollbackTaskCenterOpenTabMapForFailedSwitch({
        currentMap,
        workspaceId: "workspace-a",
        topicId: "topic-new",
        wasOpenInTaskCenter: false,
      }),
    ).toEqual({
      "workspace-a": ["topic-old"],
      "workspace-b": ["topic-other"],
    });

    expect(
      rollbackTaskCenterOpenTabMapForFailedSwitch({
        currentMap,
        workspaceId: "workspace-a",
        topicId: "topic-old",
        wasOpenInTaskCenter: true,
      }),
    ).toBe(currentMap);

    expect(
      clearTaskCenterLocalSessionOverrideForTopic(
        { sessionId: "topic-new", routeSessionId: "topic-route" },
        "topic-new",
      ),
    ).toBeNull();
    expect(
      clearTaskCenterTransitionTopicForTopic("topic-new", "topic-new"),
    ).toBeNull();
  });

  it("关闭活跃任务标签时应按相邻顺序选择 fallback", () => {
    expect(
      resolveTaskCenterTopicClosePlan({
        closingTopicId: "topic-b",
        currentOpenTabIds: ["topic-a", "topic-b", "topic-c"],
        sessionId: "topic-b",
        detachedTopicId: "topic-b",
        transitionTopicId: "topic-b",
      }),
    ).toEqual({
      remainingIds: ["topic-a", "topic-c"],
      isActiveTab: true,
      shouldClearDetachedTopic: true,
      shouldClearTransitionTopic: true,
      fallbackTopicId: "topic-c",
    });

    expect(
      resolveTaskCenterTopicClosePlan({
        closingTopicId: "topic-a",
        currentOpenTabIds: ["topic-a"],
        sessionId: "topic-a",
      }),
    ).toMatchObject({
      remainingIds: [],
      isActiveTab: true,
      fallbackTopicId: null,
    });
  });

  it("初始会话恢复默认应允许 detached 会话，不强制刷新", () => {
    expect(
      resolveInitialTaskSessionSwitchOptions({
        status: "done",
        statusReason: "default",
      }),
    ).toEqual({
      allowDetachedSession: true,
    });

    expect(resolveInitialTaskSessionSwitchOptions(null)).toEqual({
      allowDetachedSession: true,
    });
  });

  it("初始会话恢复遇到等待态任务时应恢复启动钩子", () => {
    expect(
      resolveInitialTaskSessionSwitchOptions({
        status: "waiting",
        statusReason: "user_action",
      }),
    ).toEqual({
      allowDetachedSession: true,
      resumeSessionStartHooks: true,
    });
  });

  it("初始会话恢复遇到已有历史消息时应强制读取详情", () => {
    expect(
      resolveInitialTaskSessionSwitchOptions({
        status: "done",
        statusReason: "default",
        messagesCount: 6,
      }),
    ).toEqual({
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("初始会话恢复遇到工作区错误任务时应强制刷新并恢复启动钩子", () => {
    expect(
      resolveInitialTaskSessionSwitchOptions({
        status: "failed",
        statusReason: "workspace_error",
      }),
    ).toEqual({
      allowDetachedSession: true,
      forceRefresh: true,
      resumeSessionStartHooks: true,
    });
  });

  it("应正确比较标签 id 列表是否一致", () => {
    expect(areTaskCenterTabIdsEqual(["a", "b"], ["a", "b"])).toBe(true);
    expect(areTaskCenterTabIdsEqual(["a", "b"], ["b", "a"])).toBe(false);
  });

  it("初始路由会话缺失时应延后 reconcile，避免回灌旧普通任务标签", () => {
    expect(
      shouldWaitForTaskCenterInitialSessionTopic({
        normalizedInitialSessionId: "topic-archived",
        hasInitialSessionTopic: false,
      }),
    ).toBe(true);

    expect(
      shouldWaitForTaskCenterInitialSessionTopic({
        normalizedInitialSessionId: "topic-open",
        hasInitialSessionTopic: true,
      }),
    ).toBe(false);

    expect(
      resolveTaskCenterReconcileCurrentTopicId({
        normalizedInitialSessionId: "topic-archived",
        sessionId: "topic-current",
        shouldRespectLocalSession: false,
        detachedTopicId: null,
      }),
    ).toBeNull();
  });

  it("reconcile 当前任务应保留本地追平目标，并跳过 detached 预览", () => {
    expect(
      resolveTaskCenterReconcileCurrentTopicId({
        normalizedInitialSessionId: "topic-a",
        sessionId: "topic-a",
        shouldRespectLocalSession: true,
        localSessionOverride: {
          sessionId: "topic-a",
          routeSessionId: "topic-current",
        },
        detachedTopicId: null,
      }),
    ).toBe("topic-a");

    expect(
      resolveTaskCenterReconcileCurrentTopicId({
        normalizedInitialSessionId: "topic-archived",
        sessionId: "topic-archived",
        shouldRespectLocalSession: false,
        detachedTopicId: "topic-archived",
      }),
    ).toBeNull();
  });

  it("打开不在 open tabs 中的归档对话时，顶部只应展示当前对话", () => {
    const topics = [
      createTopic("topic-open-a"),
      createTopic("topic-open-b"),
      createTopic("topic-archived-preview", {
        updatedAt: new Date("2026-04-10T00:00:00.000Z"),
      }),
    ];

    expect(
      resolveTaskCenterVisibleTabIds({
        openTabIds: ["topic-open-a", "topic-open-b"],
        topics,
        currentTopicId: "topic-archived-preview",
      }),
    ).toEqual(["topic-archived-preview"]);
  });

  it("项目会话目录加载多条历史 topics 时，顶部不应突然打开所有对话", () => {
    const topics = [
      createTopic("topic-project-a"),
      createTopic("topic-project-b"),
      createTopic("topic-project-c"),
    ];

    expect(
      resolveTaskCenterVisibleTabIds({
        openTabIds: [],
        topics,
        currentTopicId: null,
      }),
    ).toEqual([]);
  });

  it("当前对话已在 open tabs 中时，应继续展示原有任务标签", () => {
    const topics = [
      createTopic("topic-open-a"),
      createTopic("topic-open-b"),
      createTopic("title-gen-1"),
    ];

    expect(
      resolveTaskCenterVisibleTabIds({
        openTabIds: ["topic-open-a", "title-gen-1", "topic-open-b"],
        topics,
        currentTopicId: "topic-open-b",
      }),
    ).toEqual(["topic-open-a", "topic-open-b"]);
  });

  it("当前会话不在任务列表时，只应在没有切换中任务时恢复 fallback", () => {
    const topics = [createTopic("topic-open-a"), createTopic("topic-open-b")];

    expect(
      resolveTaskCenterFallbackTopicId({
        sessionId: null,
        switchingTopicId: null,
        openTabIds: ["topic-open-a", "topic-open-b"],
        topics,
      }),
    ).toBe("topic-open-a");

    expect(
      resolveTaskCenterFallbackTopicId({
        sessionId: null,
        switchingTopicId: "topic-open-a",
        openTabIds: ["topic-open-a", "topic-open-b"],
        topics,
      }),
    ).toBeNull();
  });

  it("当前会话已在任务列表中时，不应触发 fallback 恢复", () => {
    const topics = [createTopic("topic-open-a"), createTopic("topic-open-b")];

    expect(
      resolveTaskCenterFallbackTopicId({
        sessionId: "topic-open-b",
        switchingTopicId: null,
        openTabIds: ["topic-open-a", "topic-open-b"],
        topics,
      }),
    ).toBeNull();
  });

  it("新建草稿 surface 激活时，应跳过旧任务标签 fallback 恢复", () => {
    expect(
      resolveTaskCenterFallbackRestorePlan(
        createFallbackRestoreParams({
          draftSurfaceActive: true,
        }),
      ),
    ).toEqual({
      action: "skip",
      reason: "draft-surface-active",
    });
  });

  it("自动首条提交未完成时，应跳过旧任务标签 fallback 恢复", () => {
    expect(
      resolveTaskCenterFallbackRestorePlan(
        createFallbackRestoreParams({
          initialDispatchKey: "initial-skill-submit",
          messagesLength: 0,
        }),
      ),
    ).toEqual({
      action: "skip",
      reason: "initial-dispatch-pending",
    });
  });

  it("路由 initial session 尚未切换完成时，应跳过旧任务标签 fallback 恢复", () => {
    expect(
      resolveTaskCenterFallbackRestorePlan(
        createFallbackRestoreParams({
          normalizedInitialSessionId: "topic-imported",
          sessionId: "topic-old-home",
          openTabIds: ["topic-old"],
          topics: [createTopic("topic-old")],
        }),
      ),
    ).toEqual({
      action: "skip",
      reason: "initial-session-pending",
    });
  });

  it("当前没有有效任务会话时，应恢复第一个可见旧任务标签", () => {
    expect(
      resolveTaskCenterFallbackRestorePlan(
        createFallbackRestoreParams({
          now: 10_000,
        }),
      ),
    ).toEqual({
      action: "restore",
      fallbackTopicId: "topic-old",
      nextRestore: {
        topicId: "topic-old",
        startedAt: 10_000,
      },
    });
  });

  it("new-task 首页当前没有有效任务会话时，不应自动恢复旧任务标签", () => {
    expect(
      resolveTaskCenterFallbackRestorePlan(
        createFallbackRestoreParams({
          agentEntry: "new-task",
          now: 10_000,
        }),
      ),
    ).toEqual({
      action: "skip",
      reason: "new-task-home",
    });
  });

  it("new-task 首页后台恢复运行候选时，应跳过旧任务标签 fallback 恢复", () => {
    expect(
      resolveTaskCenterFallbackRestorePlan(
        createFallbackRestoreParams({
          agentEntry: "new-task",
          isHomeSessionBackgroundRecovery: true,
          now: 10_000,
        }),
      ),
    ).toEqual({
      action: "skip",
      reason: "home-background-recovery",
    });
  });

  it("new-task 首页后台恢复运行候选时，即使旧 sessionId 尚未清空也不应恢复旧详情", () => {
    expect(
      resolveTaskCenterFallbackRestorePlan(
        createFallbackRestoreParams({
          agentEntry: "new-task",
          isHomeSessionBackgroundRecovery: true,
          sessionId: "topic-old",
          currentSessionIsKnownTopic: true,
          hasDisplayMessages: true,
          now: 10_000,
        }),
      ),
    ).toEqual({
      action: "skip",
      reason: "home-background-recovery",
    });
  });

  it("短时间内已恢复同一任务标签时，应跳过重复 fallback 恢复", () => {
    expect(
      resolveTaskCenterFallbackRestorePlan(
        createFallbackRestoreParams({
          previousRestore: {
            topicId: "topic-old",
            startedAt: 10_000,
          },
          now: 11_999,
        }),
      ),
    ).toEqual({
      action: "skip",
      reason: "recently-restored",
    });
  });

  it("辅助运行时会话不应进入任务中心标签", () => {
    const topics = [
      createTopic("topic-open-a"),
      createTopic("title-gen-1"),
      createTopic("persona-gen-1"),
      createTopic("topic-open-b"),
    ];

    expect(
      reconcileTaskCenterTabIds({
        existingIds: ["topic-open-a", "title-gen-1", "persona-gen-1"],
        topics,
        currentTopicId: "persona-gen-1",
      }),
    ).toEqual(["topic-open-a"]);
  });

  it("切换归档会话时，应立即把目标会话作为预览焦点", () => {
    expect(
      resolveTaskCenterPreviewTopicId({
        sessionId: "topic-open-a",
        detachedTopicId: "topic-archived",
        switchingTopicId: "topic-archived",
      }),
    ).toBe("topic-archived");
  });

  it("归档会话完成切换后，应继续保持 detached 会话焦点", () => {
    expect(
      resolveTaskCenterPreviewTopicId({
        sessionId: "topic-archived",
        detachedTopicId: "topic-archived",
        switchingTopicId: null,
      }),
    ).toBe("topic-archived");
  });

  it("切换中的目标会话尚未成为当前会话时，应标记为待恢复态", () => {
    expect(
      isTaskCenterTopicSwitchPending({
        sessionId: "topic-open-a",
        switchingTopicId: "topic-archived",
      }),
    ).toBe(true);

    expect(
      isTaskCenterTopicSwitchPending({
        sessionId: "topic-archived",
        switchingTopicId: "topic-archived",
      }),
    ).toBe(false);
  });

  it("detached 会话处于当前预览时，应隐藏顶部任务标签", () => {
    expect(
      shouldHideTaskCenterTabsForDetachedSession({
        sessionId: "topic-archived",
        detachedTopicId: "topic-archived",
        openTabIds: ["topic-open-a", "topic-open-b"],
      }),
    ).toBe(true);
  });

  it("从导航栏直达且不在 open tabs 中的会话，应隐藏顶部任务标签", () => {
    expect(
      shouldHideTaskCenterTabsForDetachedSession({
        sessionId: "topic-archived",
        initialSessionId: "topic-archived",
        openTabIds: ["topic-open-a", "topic-open-b"],
      }),
    ).toBe(true);
  });

  it("当前会话已进入 open tabs 时，不应隐藏顶部任务标签", () => {
    expect(
      shouldHideTaskCenterTabsForDetachedSession({
        sessionId: "topic-open-b",
        initialSessionId: "topic-open-b",
        openTabIds: ["topic-open-a", "topic-open-b"],
      }),
    ).toBe(false);
  });
});
