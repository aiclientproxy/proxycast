import { describe, expect, it, vi } from "vitest";
import type { CanonicalChildThreadSummary } from "../projection/canonicalChildThreadSummary";
import { openWorkspaceSubagentTarget } from "./useWorkspaceSubagentNavigationRuntime";

function canonicalChild(
  overrides: Partial<CanonicalChildThreadSummary> = {},
): CanonicalChildThreadSummary {
  return {
    name: "Child",
    parentThreadId: "thread-parent",
    sessionId: "agent-child",
    status: "running",
    threadId: "thread-child",
    updatedAtMs: 1,
    ...overrides,
  };
}

describe("openWorkspaceSubagentTarget", () => {
  it("canonical ThreadId 命中已知 sessionId 时应直接导航", async () => {
    const readSessionId = vi.fn();
    const deferSessionRecentMetadataSyncForNavigation = vi.fn();
    const switchTopic = vi.fn();

    await openWorkspaceSubagentTarget({
      canonicalChildren: [canonicalChild()],
      deferSessionRecentMetadataSyncForNavigation,
      readSessionId,
      switchTopic,
      targetId: " thread-child ",
    });

    expect(readSessionId).not.toHaveBeenCalled();
    expect(deferSessionRecentMetadataSyncForNavigation).toHaveBeenCalledWith(
      "agent-child",
    );
    expect(switchTopic).toHaveBeenCalledWith("agent-child", {
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("canonical child 缺少 sessionId 时应读取 Thread 映射", async () => {
    const readSessionId = vi.fn().mockResolvedValue("agent-resolved");
    const deferSessionRecentMetadataSyncForNavigation = vi.fn();
    const switchTopic = vi.fn();

    await openWorkspaceSubagentTarget({
      canonicalChildren: [canonicalChild({ sessionId: null })],
      deferSessionRecentMetadataSyncForNavigation,
      readSessionId,
      switchTopic,
      targetId: "thread-child",
    });

    expect(readSessionId).toHaveBeenCalledWith("thread-child");
    expect(deferSessionRecentMetadataSyncForNavigation).toHaveBeenCalledWith(
      "agent-resolved",
    );
    expect(switchTopic).toHaveBeenCalledWith("agent-resolved", {
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("未知 canonical ThreadId 应读取真实 sessionId 后导航", async () => {
    const readSessionId = vi.fn().mockResolvedValue("agent-child");
    const deferSessionRecentMetadataSyncForNavigation = vi.fn();
    const switchTopic = vi.fn();

    await openWorkspaceSubagentTarget({
      canonicalChildren: [],
      deferSessionRecentMetadataSyncForNavigation,
      readSessionId,
      switchTopic,
      targetId: "thread-child",
    });

    expect(readSessionId).toHaveBeenCalledWith("thread-child");
    expect(deferSessionRecentMetadataSyncForNavigation).toHaveBeenCalledWith(
      "agent-child",
    );
    expect(switchTopic).toHaveBeenCalledWith("agent-child", {
      allowDetachedSession: true,
      forceRefresh: true,
    });
  });

  it("Thread 身份解析失败时不得切换 session", async () => {
    const deferSessionRecentMetadataSyncForNavigation = vi.fn();
    const switchTopic = vi.fn();

    await expect(
      openWorkspaceSubagentTarget({
        canonicalChildren: [],
        deferSessionRecentMetadataSyncForNavigation,
        readSessionId: vi.fn().mockRejectedValue(new Error("mismatch")),
        switchTopic,
        targetId: "thread-child",
      }),
    ).rejects.toThrow("mismatch");

    expect(deferSessionRecentMetadataSyncForNavigation).not.toHaveBeenCalled();
    expect(switchTopic).not.toHaveBeenCalled();
  });
});
