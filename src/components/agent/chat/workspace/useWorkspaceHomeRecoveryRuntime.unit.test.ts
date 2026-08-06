import { describe, expect, it } from "vitest";
import type { Topic } from "../hooks/agentChatShared";
import { resolveWorkspaceHomeRecoverySession } from "./useWorkspaceHomeRecoveryRuntime";

function topic(overrides: Partial<Topic> = {}): Topic {
  return {
    id: "topic-1",
    title: "  后台任务  ",
    createdAt: new Date("2026-01-01T00:00:00.000Z"),
    updatedAt: new Date("2026-01-01T00:00:00.000Z"),
    messagesCount: 1,
    executionStrategy: "react",
    status: "running",
    lastPreview: "  正在整理资料  ",
    hasUnread: false,
    sourceSessionId: "session-1",
    ...overrides,
  };
}

describe("resolveWorkspaceHomeRecoverySession", () => {
  it("只投影可恢复的后台会话状态", () => {
    expect(
      resolveWorkspaceHomeRecoverySession(topic({ status: "running" })),
    ).toMatchObject({ status: "running" });
    expect(
      resolveWorkspaceHomeRecoverySession(topic({ status: "queued" })),
    ).toMatchObject({ status: "queued" });
    expect(
      resolveWorkspaceHomeRecoverySession(topic({ status: "waiting" })),
    ).toMatchObject({ status: "waiting" });
    expect(
      resolveWorkspaceHomeRecoverySession(topic({ status: "done" })),
    ).toBeNull();
  });

  it("标题为空时不展示恢复入口", () => {
    expect(
      resolveWorkspaceHomeRecoverySession(topic({ title: "   " })),
    ).toBeNull();
  });

  it("裁剪标题和摘要，并优先使用 sourceSessionId", () => {
    expect(resolveWorkspaceHomeRecoverySession(topic())).toEqual({
      sessionId: "session-1",
      title: "后台任务",
      summary: "正在整理资料",
      status: "running",
    });
  });

  it("没有摘要和 sourceSessionId 时回退到 topic id", () => {
    expect(
      resolveWorkspaceHomeRecoverySession(
        topic({ id: "topic-fallback", sourceSessionId: "", lastPreview: " " }),
      ),
    ).toEqual({
      sessionId: "topic-fallback",
      title: "后台任务",
      status: "running",
    });
  });
});
