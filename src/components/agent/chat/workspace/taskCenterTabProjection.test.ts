import { describe, expect, it } from "vitest";
import type { Topic } from "../hooks/agentChatShared";
import {
  buildBrowserWorkspaceHomeTabItem,
  buildTaskCenterTabItems,
  resolveTaskCenterTopicTitle,
  shouldRenderTaskCenterTabStrip,
  TASK_CENTER_HOME_TAB_ID,
} from "./taskCenterTabProjection";
import type { TaskCenterDraftTab } from "./agentChatWorkspaceHelpers";

function createTopic(id: string, overrides?: Partial<Topic>): Topic {
  return {
    id,
    title: id,
    createdAt: new Date("2026-04-20T00:00:00.000Z"),
    updatedAt: new Date("2026-04-20T01:00:00.000Z"),
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

describe("taskCenterTabProjection", () => {
  it("应把已有任务排在新对话草稿前面", () => {
    const draft: TaskCenterDraftTab = {
      id: "task-draft-a",
      title: "新对话",
      createdAt: new Date("2026-04-20T02:00:00.000Z"),
      updatedAt: new Date("2026-04-20T02:00:00.000Z"),
      status: "draft",
    };
    const topicById = new Map<string, Topic>([
      [
        "topic-a",
        createTopic("topic-a", {
          title: "",
          hasUnread: true,
        }),
      ],
      [
        "topic-b",
        createTopic("topic-b", {
          title: "任务 B",
          status: "running",
          updatedAt: "2026-04-20T03:00:00.000Z" as unknown as Date,
        }),
      ],
    ]);

    const items = buildTaskCenterTabItems({
      draftTabs: [draft],
      activeDraftTabId: draft.id,
      isDraftTabActive: true,
      sessionId: "topic-b",
      previewTopicId: null,
      visibleTabIds: ["topic-a", "topic-b"],
      topicById,
      untitledTaskLabel: "未命名任务",
    });

    expect(items.map((item) => item.id)).toEqual([
      "topic-a",
      "topic-b",
      "task-draft-a",
    ]);
    expect(items[0]).toMatchObject({
      title: "未命名任务",
      hasUnread: true,
      isActive: false,
    });
    expect(items[1]).toMatchObject({
      title: "任务 B",
      renamable: true,
      isActive: false,
      status: "running",
    });
    expect(items[1]?.updatedAt).toBeInstanceOf(Date);
    expect(items[2]).toMatchObject({
      isActive: true,
      renamable: false,
      status: "draft",
    });
  });

  it("达到标签上限时仍应把活跃新对话保留在最后", () => {
    const draft: TaskCenterDraftTab = {
      id: "task-draft-a",
      title: "新对话",
      createdAt: new Date("2026-04-20T04:00:00.000Z"),
      updatedAt: new Date("2026-04-20T04:00:00.000Z"),
      status: "draft",
    };
    const topicById = new Map<string, Topic>(
      ["topic-a", "topic-b", "topic-c"].map((topicId) => [
        topicId,
        createTopic(topicId),
      ]),
    );

    const items = buildTaskCenterTabItems({
      draftTabs: [draft],
      activeDraftTabId: draft.id,
      isDraftTabActive: true,
      sessionId: "topic-a",
      previewTopicId: null,
      visibleTabIds: ["topic-a", "topic-b", "topic-c"],
      topicById,
      untitledTaskLabel: "未命名任务",
      maxCount: 3,
    });

    expect(items.map((item) => item.id)).toEqual([
      "topic-a",
      "topic-b",
      "task-draft-a",
    ]);
    expect(items[2]).toMatchObject({
      isActive: true,
      status: "draft",
    });
  });

  it("应构建不可关闭的浏览器首页标签", () => {
    expect(
      buildBrowserWorkspaceHomeTabItem({
        title: "新对话",
        updatedAtMs: Date.UTC(2026, 3, 20, 4, 0, 0),
      }),
    ).toMatchObject({
      id: TASK_CENTER_HOME_TAB_ID,
      title: "新对话",
      status: "draft",
      isActive: true,
      renamable: false,
      closable: false,
    });
  });

  it("应清理任务标签里的运行时错误标题并保留附件标题", () => {
    const runtimeErrorEnvelope = [
      "Ran into this error: Server error: upstream temporarily unavailable.",
      "",
      "Please retry if you think this is a transient or recoverable error.",
    ].join("\n");

    expect(resolveTaskCenterTopicTitle(runtimeErrorEnvelope, "未命名任务")).toBe(
      "未命名任务",
    );
    expect(resolveTaskCenterTopicTitle("Ran into this erro...", "未命名任务")).toBe(
      "未命名任务",
    );
    expect(resolveTaskCenterTopicTitle("[Image #4]", "未命名任务")).toBe(
      "图片任务 4",
    );
  });

  it("初始 Thread 标题尚未进入 Topic 投影时应使用路由元数据兜底", () => {
    const topic = createTopic("thread-active", { title: "" });

    const items = buildTaskCenterTabItems({
      draftTabs: [],
      activeDraftTabId: null,
      isDraftTabActive: false,
      sessionId: "thread-active",
      initialSessionId: "thread-active",
      initialSessionName: "修复恢复链路",
      previewTopicId: null,
      visibleTabIds: ["thread-active"],
      topicById: new Map([[topic.id, topic]]),
      untitledTaskLabel: "未命名任务",
    });

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      id: "thread-active",
      title: "修复恢复链路",
      isActive: true,
    });
  });

  it("应在 claw 或 new-task 有标签项时渲染任务标签栏", () => {
    expect(
      shouldRenderTaskCenterTabStrip({
        agentEntry: "claw",
        hasLocalSessionOverride: false,
        tabItemCount: 0,
      }),
    ).toBe(true);
    expect(
      shouldRenderTaskCenterTabStrip({
        agentEntry: "new-task",
        hasLocalSessionOverride: true,
        tabItemCount: 1,
      }),
    ).toBe(true);
    expect(
      shouldRenderTaskCenterTabStrip({
        agentEntry: "new-task",
        hasLocalSessionOverride: false,
        tabItemCount: 1,
      }),
    ).toBe(true);
    expect(
      shouldRenderTaskCenterTabStrip({
        agentEntry: "new-task",
        hasLocalSessionOverride: false,
        tabItemCount: 0,
      }),
    ).toBe(false);
  });
});
