import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  createMockAgentChatUnifiedState,
  FIXED_TOPIC_UPDATED_AT,
  flushEffects,
  installMockAgentChatUnifiedState,
  mountPage,
} from "./index.testFixtures";
import { buildHomeAgentParams } from "@/lib/workspace/navigation";
import { TASK_CENTER_OPEN_TAB_IDS_STORAGE_KEY } from "./utils/taskCenterTabs";
import { requestTaskCenterDraftTask } from "./taskCenterDraftTaskEvents";

describe("AgentChatPage 任务中心初始会话标签", { timeout: 20_000 }, () => {
  it("切到另一条旧会话后仍应继续在当前页新建本地草稿", async () => {
    const onNavigate = vi.fn();
    vi.mocked(buildHomeAgentParams).mockClear();
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-current",
      topics: [
        {
          id: "topic-current",
          title: "旧会话 A",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT - 1_000),
          workspaceId: "workspace-test",
        },
        {
          id: "topic-next",
          title: "旧会话 B",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    state.createFreshSession = vi.fn(async () => "new-topic");
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
      onNavigate,
    });
    await flushEffects();

    expect(requestTaskCenterDraftTask({ source: "sidebar" })).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(
      mounted.container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();
    expect(
      mounted.container.querySelector(
        '[data-testid^="task-center-tab-task-draft-"]',
      ),
    ).toBeNull();
    expect(
      mounted.container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();

    state.sessionId = "topic-next";
    mounted.rerender({ initialSessionId: "topic-next" });
    await flushEffects();

    expect(requestTaskCenterDraftTask({ source: "sidebar" })).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(state.createFreshSession).not.toHaveBeenCalled();
    expect(buildHomeAgentParams).not.toHaveBeenCalled();
    expect(onNavigate).not.toHaveBeenCalled();
    expect(
      mounted.container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();
    expect(
      mounted.container.querySelector('[data-testid="message-list"]'),
    ).toBeNull();
  });

  it("从任务中心外部入口切到非路由会话时，不应被初始路由抢回", async () => {
    let resolveSwitchTopic: (() => void) | null = null;
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-current",
      topics: [
        {
          id: "topic-current",
          title: "当前会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT - 1_000),
          workspaceId: "workspace-test",
        },
        {
          id: "topic-a",
          title: "目标会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    const switchTopic = vi.fn(async (topicId: string) => {
      state.sessionId = topicId;
      return new Promise<void>((resolve) => {
        resolveSwitchTopic = resolve;
      });
    });
    state.switchTopic = switchTopic;
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    mounted.rerender({
      initialSessionId: "topic-a",
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith(
      "topic-a",
      expect.objectContaining({ allowDetachedSession: true }),
    );
    expect(
      mounted.container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();

    act(() => {
      resolveSwitchTopic?.();
    });
    await flushEffects();
  });

  it("外层侧边栏打开历史会话后，路由追平应只保留目标会话", async () => {
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-current",
      topics: [
        {
          id: "topic-current",
          title: "旧会话 A",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT - 1_000),
          workspaceId: "workspace-test",
        },
        {
          id: "topic-next",
          title: "旧会话 B",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    const switchTopic = vi.fn(async (topicId: string) => {
      state.sessionId = topicId;
    });
    state.switchTopic = switchTopic;
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    mounted.rerender({
      initialSessionId: "topic-next",
    });
    await flushEffects();
    expect(switchTopic).toHaveBeenCalledWith(
      "topic-next",
      expect.objectContaining({ allowDetachedSession: true }),
    );

    mounted.rerender();
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledTimes(1);
    expect(
      mounted.container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();
    expect(
      JSON.parse(
        localStorage.getItem(TASK_CENTER_OPEN_TAB_IDS_STORAGE_KEY) ?? "{}",
      )["workspace-test"],
    ).toEqual(["topic-next"]);
  });

  it("路由会话已匹配但消息投影为空时应强制 hydrate 历史详情", async () => {
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-history",
      messages: [],
      turns: [],
      threadItems: [],
      topics: [],
    });
    const switchTopic = vi.fn(async (topicId: string) => {
      state.sessionId = topicId;
    });
    state.switchTopic = switchTopic;
    installMockAgentChatUnifiedState(state);

    mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-history",
      projectId: "workspace-test",
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith(
      "topic-history",
      expect.objectContaining({
        allowDetachedSession: true,
        forceRefresh: true,
      }),
    );
  });

  it("打开历史会话时不应先清空当前消息再切换", async () => {
    const setMessages = vi.fn();
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: "topic-current",
      messages: [
        {
          id: "msg-current",
          role: "assistant",
          content: "当前会话已有内容",
          timestamp: new Date(FIXED_TOPIC_UPDATED_AT),
        },
      ],
      topics: [
        {
          id: "topic-current",
          title: "当前会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT - 1_000),
          workspaceId: "workspace-test",
        },
        {
          id: "topic-a",
          title: "历史会话",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
      setMessages,
    });
    const switchTopic = vi.fn(async (topicId: string) => {
      state.sessionId = topicId;
    });
    state.switchTopic = switchTopic;
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "claw",
      initialSessionId: "topic-current",
      projectId: "workspace-test",
    });
    await flushEffects();

    mounted.rerender({
      initialSessionId: "topic-a",
    });
    await flushEffects();

    expect(switchTopic).toHaveBeenNthCalledWith(
      1,
      "topic-a",
      expect.objectContaining({ allowDetachedSession: true }),
    );
    expect(setMessages).not.toHaveBeenCalledWith([]);
  });

  it("new-task 首页连续新建任务事件应保留在当前页", async () => {
    const onNavigate = vi.fn();
    vi.mocked(buildHomeAgentParams).mockClear();
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: null,
      topics: [],
    });
    state.createFreshSession = vi.fn(async () => "new-topic");
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "new-task",
      projectId: "workspace-test",
      onNavigate,
    });
    await flushEffects();

    expect(requestTaskCenterDraftTask({ source: "sidebar" })).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(requestTaskCenterDraftTask({ source: "sidebar" })).toBe(true);
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(
      mounted.container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();
    expect(
      mounted.container.querySelector(
        '[data-testid^="task-center-tab-task-draft-"]',
      ),
    ).toBeNull();
    expect(
      mounted.container.querySelector('[data-testid="empty-state"]'),
    ).not.toBeNull();
    expect(state.createFreshSession).not.toHaveBeenCalled();
    expect(buildHomeAgentParams).not.toHaveBeenCalled();
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("new-task 首页收到 current route 历史会话时应立即切换会话", async () => {
    const state: Record<string, unknown> = createMockAgentChatUnifiedState({
      sessionId: null,
      topics: [
        {
          id: "topic-next",
          title: "旧会话 B",
          updatedAt: new Date(FIXED_TOPIC_UPDATED_AT),
          workspaceId: "workspace-test",
        },
      ],
    });
    const switchTopic = vi.fn(async (topicId: string) => {
      state.sessionId = topicId;
    });
    state.switchTopic = switchTopic;
    installMockAgentChatUnifiedState(state);

    const mounted = mountPage({
      agentEntry: "new-task",
      projectId: "workspace-test",
    });
    await flushEffects();

    mounted.rerender({
      agentEntry: "claw",
      initialSessionId: "topic-next",
    });
    await flushEffects();
    mounted.rerender();
    await flushEffects();

    expect(switchTopic).toHaveBeenCalledWith(
      "topic-next",
      expect.objectContaining({ allowDetachedSession: true }),
    );
    expect(
      mounted.container.querySelector('[data-testid="task-center-tab-strip"]'),
    ).toBeNull();
  });

  it("外部路由带项目时不再渲染顶部项目切换入口", async () => {
    const onNavigate = vi.fn();
    installMockAgentChatUnifiedState(createMockAgentChatUnifiedState());

    const mounted = mountPage({
      agentEntry: "claw",
      projectId: "project-default",
      onNavigate,
    });
    await flushEffects();

    expect(
      mounted.container.querySelector('[data-testid="chat-navbar"]'),
    ).toBeNull();
    expect(
      mounted.container.querySelector(
        '[data-testid="inputbar-project-context-project-trigger"]',
      ),
    ).toBeNull();
    expect(onNavigate).not.toHaveBeenCalled();
  });

});
