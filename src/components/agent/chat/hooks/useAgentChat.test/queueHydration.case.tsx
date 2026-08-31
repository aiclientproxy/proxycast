import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  flushEffects,
  mockGetAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockResumeAgentRuntimeThread,
  mockSubmitAgentRuntimeTurn,
  mountHook,
} from "../useAgentChat.testUtils";

describe("useAgentChat canonical queue status", () => {
  it("切换话题时应消费 session list 的 canonical queue count", async () => {
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: "session-queue",
        name: "带队列的话题",
        created_at: 1,
        updated_at: 2,
        thread_status: "queued",
        queued_turn_count: 1,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "session-queue",
      messages: [],
      turns: [],
      items: [],
      thread_read: {
        thread_id: "thread-queue",
        status: "queued",
      },
    });

    const harness = mountHook("ws-queue-hydration");

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("session-queue");
      });
      await flushEffects();

      expect(harness.getValue().queuedTurnCount).toBe(1);
      expect(harness.getValue()).not.toHaveProperty("queuedTurns");
      expect(
        harness.getValue().topics.find((topic) => topic.id === "session-queue"),
      ).toMatchObject({
        status: "queued",
      });
      expect(mockResumeAgentRuntimeThread).toHaveBeenCalledWith({
        threadId: "thread-queue",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换到 running thread_read 时重新建立 canonical Thread 订阅", async () => {
    const sessionId = "session-running-auto-resume";
    mockListAgentRuntimeSessions.mockResolvedValue([
      {
        id: sessionId,
        name: "运行中的话题",
        created_at: 1,
        updated_at: 2,
      },
    ]);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [],
      turns: [],
      items: [],
      thread_read: {
        thread_id: "thread-running-auto-resume",
        status: "running",
        active_turn_id: "turn-running-1",
        pending_requests: [],
        incidents: [],
      },
    });

    const harness = mountHook("ws-running-auto-resume");

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(sessionId);
      });
      await flushEffects();

      expect(mockResumeAgentRuntimeThread).toHaveBeenCalledWith({
        threadId: "thread-running-auto-resume",
      });
      expect(mockSubmitAgentRuntimeTurn).not.toHaveBeenCalled();
      expect(harness.getValue().isSending).toBe(false);
      expect(harness.getValue().threadRead).toMatchObject({
        thread_id: "thread-running-auto-resume",
        status: "running",
        active_turn_id: "turn-running-1",
      });
      expect(
        harness.getValue().topics.find((topic) => topic.id === sessionId),
      ).toMatchObject({
        status: "running",
      });
    } finally {
      harness.unmount();
    }
  });
});
