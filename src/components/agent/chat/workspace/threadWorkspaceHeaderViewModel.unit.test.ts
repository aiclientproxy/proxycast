import { describe, expect, it } from "vitest";
import type { Topic } from "../hooks/agentChatShared";
import { buildThreadWorkspaceHeaderViewModel } from "./threadWorkspaceHeaderViewModel";

function topic(overrides?: Partial<Topic>): Topic {
  return {
    id: "thread-1",
    title: "整理 GUI 对齐计划",
    createdAt: new Date("2026-07-18T08:00:00.000Z"),
    updatedAt: new Date("2026-07-18T09:00:00.000Z"),
    workingDir: "/workspace/lime",
    messagesCount: 4,
    executionStrategy: "react",
    status: "running",
    statusReason: "default",
    lastPreview: "继续整理界面",
    hasUnread: false,
    sourceSessionId: "thread-1",
    ...overrides,
  };
}

describe("buildThreadWorkspaceHeaderViewModel", () => {
  it("应优先使用 active Topic 的标题、状态和当前 session 工作目录", () => {
    expect(
      buildThreadWorkspaceHeaderViewModel({
        sessionId: "thread-1",
        currentSessionTitle: "旧标题",
        topic: topic(),
        sessionWorkingDirectory: "/workspace/current",
        projectRootPath: "/workspace/project",
        untitledTaskLabel: "未命名任务",
      }),
    ).toEqual({
      sessionId: "thread-1",
      title: "整理 GUI 对齐计划",
      status: "running",
      workingDirectory: "/workspace/current",
    });
  });

  it("Topic 标题尚未恢复时应使用入口 Thread 名称兜底", () => {
    expect(
      buildThreadWorkspaceHeaderViewModel({
        sessionId: "thread-1",
        initialSessionId: "thread-1",
        initialSessionName: "恢复历史任务",
        topic: topic({ title: "", workingDir: null, status: "done" }),
        projectRootPath: "/workspace/project",
        untitledTaskLabel: "未命名任务",
      }),
    ).toMatchObject({
      title: "恢复历史任务",
      status: "done",
      workingDirectory: "/workspace/project",
    });
  });

  it.each([
    {
      name: "待用户操作优先于运行态",
      isSending: true,
      pendingActionCount: 1,
      expected: "waiting",
    },
    {
      name: "运行态优先于 Topic 缓存状态",
      isSending: true,
      pendingActionCount: 0,
      expected: "running",
    },
  ])("应使用 current scene 投影状态：$name", (projection) => {
    expect(
      buildThreadWorkspaceHeaderViewModel({
        sessionId: "thread-1",
        topic: topic({ status: "done" }),
        isSending: projection.isSending,
        pendingActionCount: projection.pendingActionCount,
        untitledTaskLabel: "未命名任务",
      })?.status,
    ).toBe(projection.expected);
  });

  it("没有 active Thread 时不应生成页头", () => {
    expect(
      buildThreadWorkspaceHeaderViewModel({
        sessionId: null,
        untitledTaskLabel: "未命名任务",
      }),
    ).toBeNull();
  });
});
