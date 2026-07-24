import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  captureContextCompactionStream,
  captureTurnStream,
  completedTurn,
  createDeferred,
  flushEffects,
  mockCompactAgentRuntimeSession,
  mockSubmitAgentRuntimeTurn,
  mockToast,
  mountHook,
  seedSession,
} from "../useAgentChat.testUtils";

describe("useAgentChat thread timeline", () => {
  it("sendMessage 后在首个流事件前应先注入本地回合占位", async () => {
    const workspaceId = "ws-thread-optimistic";
    seedSession(workspaceId, "session-thread-optimistic");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("帮我先开始处理", [], false, false, false, "react");
      });
      await flushEffects();

      const pendingTurnKey = harness.getValue().currentTurnId;
      expect(pendingTurnKey).toContain("pending-turn:");
      expect(harness.getValue().turns).toHaveLength(1);
      expect(harness.getValue().turns[0]?.id).toBe(pendingTurnKey);
      expect(harness.getValue().turns[0]?.status).toBe("running");
      expect(harness.getValue().threadItems).toHaveLength(1);
      expect(harness.getValue().threadItems[0]?.id).toBe(
        `pending-item:${pendingTurnKey}`,
      );
      expect(harness.getValue().threadItems[0]?.type).toBe("turn_summary");
      expect(harness.getValue().threadItems[0]?.status).toBe("in_progress");

      act(() => {
        stream.emit({
          type: "turn_started",
          turn: {
            id: "turn-real-1",
            thread_id: "session-thread-optimistic",
            prompt_text: "帮我先开始处理",
            status: "running",
            started_at: "2026-03-13T11:00:00.000Z",
            created_at: "2026-03-13T11:00:00.000Z",
            updated_at: "2026-03-13T11:00:00.000Z",
          },
        });
      });

      expect(harness.getValue().currentTurnId).toBe("turn-real-1");
      expect(harness.getValue().turns).toHaveLength(1);
      expect(harness.getValue().turns[0]?.id).toBe("turn-real-1");
      expect(harness.getValue().threadItems).toEqual([
        expect.objectContaining({
          id: `pending-item:${pendingTurnKey}`,
          type: "turn_summary",
          status: "in_progress",
          turn_id: "turn-real-1",
          thread_id: "session-thread-optimistic",
        }),
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("submitTurn 失败时应保留失败回合与失败消息，而不是清空当前过程", async () => {
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    const workspaceId = "ws-thread-submit-failed";
    seedSession(workspaceId, "session-thread-submit-failed");
    mockSubmitAgentRuntimeTurn.mockRejectedValueOnce(
      new Error("429 rate limit"),
    );
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("帮我开始执行", [], false, false, false, "react");
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.content).toContain("执行失败：429 rate limit");
      expect(assistantMessage?.runtimeStatus).toMatchObject({
        phase: "failed",
        title: "当前处理失败",
      });
      expect(harness.getValue().turns).toEqual([
        expect.objectContaining({
          status: "failed",
          error_message: "429 rate limit",
        }),
      ]);
      expect(harness.getValue().threadItems).toEqual([
        expect.objectContaining({
          type: "turn_summary",
          status: "failed",
        }),
      ]);
      expect(mockToast.warning).toHaveBeenCalledWith(
        "请求过于频繁，请稍后重试",
      );
    } finally {
      consoleErrorSpy.mockRestore();
      harness.unmount();
    }
  });

  it("应接收 turn/item 生命周期事件并写入运行态", async () => {
    const workspaceId = "ws-thread-timeline";
    seedSession(workspaceId, "session-thread-timeline");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("帮我整理一个计划", [], false, false, false, "react");
      });

      act(() => {
        stream.emit({
          type: "turn_started",
          turn: {
            id: "turn-1",
            thread_id: "session-thread-timeline",
            prompt_text: "帮我整理一个计划",
            status: "running",
            started_at: "2026-03-13T10:00:00.000Z",
            created_at: "2026-03-13T10:00:00.000Z",
            updated_at: "2026-03-13T10:00:00.000Z",
          },
        });
        stream.emit({
          type: "item_started",
          item: {
            id: "plan-1",
            thread_id: "session-thread-timeline",
            turn_id: "turn-1",
            sequence: 1,
            status: "in_progress",
            started_at: "2026-03-13T10:00:01.000Z",
            updated_at: "2026-03-13T10:00:01.000Z",
            type: "plan",
            text: "1. 收集资料\n2. 输出结论",
          },
        });
        stream.emit({
          type: "item_completed",
          item: {
            id: "plan-1",
            thread_id: "session-thread-timeline",
            turn_id: "turn-1",
            sequence: 1,
            status: "completed",
            started_at: "2026-03-13T10:00:01.000Z",
            completed_at: "2026-03-13T10:00:03.000Z",
            updated_at: "2026-03-13T10:00:03.000Z",
            type: "plan",
            text: "1. 收集资料\n2. 输出结论",
          },
        });
        stream.emit({
          type: "turn_completed",
          turn: {
            id: "turn-1",
            thread_id: "session-thread-timeline",
            prompt_text: "帮我整理一个计划",
            status: "completed",
            started_at: "2026-03-13T10:00:00.000Z",
            completed_at: "2026-03-13T10:00:04.000Z",
            created_at: "2026-03-13T10:00:00.000Z",
            updated_at: "2026-03-13T10:00:04.000Z",
          },
        });
      });

      expect(harness.getValue().currentTurnId).toBe("turn-1");
      expect(harness.getValue().turns).toHaveLength(1);
      expect(harness.getValue().turns[0]?.status).toBe("completed");
      expect(harness.getValue().threadItems).toHaveLength(1);
      expect(harness.getValue().threadItems[0]?.type).toBe("plan");
      expect(harness.getValue().threadItems[0]?.status).toBe("completed");
    } finally {
      harness.unmount();
    }
  });

  it("stream error 事件时应保留失败消息与失败回合", async () => {
    const workspaceId = "ws-thread-stream-error";
    seedSession(workspaceId, "session-thread-stream-error");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("请开始处理", [], false, false, false, "react");
      });

      act(() => {
        stream.emit({
          type: "turn_started",
          turn: {
            id: "turn-stream-error-1",
            thread_id: "session-thread-stream-error",
            prompt_text: "请开始处理",
            status: "running",
            started_at: "2026-03-20T10:00:00.000Z",
            created_at: "2026-03-20T10:00:00.000Z",
            updated_at: "2026-03-20T10:00:00.000Z",
          },
        });
        stream.emit({
          type: "error",
          message: "模型执行失败",
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.content).toContain("执行失败：模型执行失败");
      expect(assistantMessage?.runtimeStatus).toMatchObject({
        phase: "failed",
        title: "当前处理失败",
      });
      expect(harness.getValue().turns).toEqual([
        expect.objectContaining({
          id: "turn-stream-error-1",
          status: "failed",
          error_message: "模型执行失败",
        }),
      ]);
      expect(harness.getValue().threadItems).toEqual([
        expect.objectContaining({
          id: expect.stringContaining("pending-item:"),
          type: "turn_summary",
          status: "failed",
          turn_id: "turn-stream-error-1",
        }),
      ]);
      expect(mockToast.error).toHaveBeenCalledWith("响应错误: 模型执行失败");
    } finally {
      harness.unmount();
    }
  });

  it("stream error 命中 Provider schema 兼容问题时应展示友好提示", async () => {
    const workspaceId = "ws-thread-provider-session-expired";
    const rawErrorMessage =
      "Agent provider execution failed: Request failed: Bad request (400): Invalid schema for function 'SendMessage': In context=('properties', 'message', 'oneOf', '2'), array schema missing items";
    const friendlyErrorMessage =
      "当前模型通道返回了不兼容的工具 schema，请前往设置 -> AI 服务商检查 Provider 配置或切换模型后重试。";
    seedSession(workspaceId, "session-thread-provider-session-expired");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("请继续处理", [], false, false, false, "react");
      });

      act(() => {
        stream.emit({
          type: "turn_started",
          turn: {
            id: "turn-provider-session-expired-1",
            thread_id: "session-thread-provider-session-expired",
            prompt_text: "请继续处理",
            status: "running",
            started_at: "2026-04-07T10:00:00.000Z",
            created_at: "2026-04-07T10:00:00.000Z",
            updated_at: "2026-04-07T10:00:00.000Z",
          },
        });
        stream.emit({
          type: "error",
          message: rawErrorMessage,
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.content).toContain(
        `执行失败：${friendlyErrorMessage}`,
      );
      expect(assistantMessage?.runtimeStatus).toMatchObject({
        phase: "failed",
        title: "当前处理失败",
        detail: friendlyErrorMessage,
      });
      expect(harness.getValue().turns).toEqual([
        expect.objectContaining({
          id: "turn-provider-session-expired-1",
          status: "failed",
          error_message: rawErrorMessage,
        }),
      ]);
      expect(mockToast.error).toHaveBeenCalledWith(friendlyErrorMessage);
    } finally {
      harness.unmount();
    }
  });

  it("手动压缩上下文时即使没有 assistant 正文也应完成时间线更新", async () => {
    const workspaceId = "ws-context-compaction";
    seedSession(workspaceId, "session-context-compaction");
    const harness = mountHook(workspaceId);
    const stream = captureContextCompactionStream();
    const deferredCompaction = createDeferred<void>();
    mockCompactAgentRuntimeSession.mockReturnValueOnce(
      deferredCompaction.promise,
    );

    try {
      await flushEffects();

      let compactionPromise!: Promise<void>;
      act(() => {
        compactionPromise = harness.getValue().compactSession();
      });
      await flushEffects();

      expect(mockCompactAgentRuntimeSession).toHaveBeenCalledWith({
        session_id: "session-context-compaction",
        event_name: stream.getEventName(),
      });

      act(() => {
        stream.emitBridge({
          type: "turn_started",
          turn: {
            id: "turn-compact-1",
            thread_id: "session-context-compaction",
            prompt_text: "压缩上下文",
            status: "running",
            started_at: "2026-03-23T09:00:00.000Z",
            created_at: "2026-03-23T09:00:00.000Z",
            updated_at: "2026-03-23T09:00:00.000Z",
          },
        });
        stream.emitBridge({
          type: "item_started",
          item: {
            id: "compact-1",
            thread_id: "session-context-compaction",
            turn_id: "turn-compact-1",
            sequence: 1,
            status: "in_progress",
            started_at: "2026-03-23T09:00:01.000Z",
            updated_at: "2026-03-23T09:00:01.000Z",
            type: "context_compaction",
            stage: "started",
            trigger: "manual",
            detail: "压缩当前会话上下文",
          },
        });
        stream.emitBridge({
          type: "item_completed",
          item: {
            id: "compact-1",
            thread_id: "session-context-compaction",
            turn_id: "turn-compact-1",
            sequence: 1,
            status: "completed",
            started_at: "2026-03-23T09:00:01.000Z",
            completed_at: "2026-03-23T09:00:03.000Z",
            updated_at: "2026-03-23T09:00:03.000Z",
            type: "context_compaction",
            stage: "completed",
            trigger: "manual",
            detail: "已生成摘要并替换旧上下文",
          },
        });
        stream.emitBridge({
          type: "turn_completed",
          turn: {
            id: "turn-compact-1",
            thread_id: "session-context-compaction",
            prompt_text: "压缩上下文",
            status: "completed",
            started_at: "2026-03-23T09:00:00.000Z",
            completed_at: "2026-03-23T09:00:04.000Z",
            created_at: "2026-03-23T09:00:00.000Z",
            updated_at: "2026-03-23T09:00:04.000Z",
          },
        });
        stream.emitBridge({
          type: "warning",
          code: "context_compaction_accuracy_manual_timeline",
          message:
            "长对话和多次上下文压缩会降低模型准确性；如果后续结果开始漂移，建议新开会话。",
        });
      });

      deferredCompaction.resolve();
      await act(async () => {
        await compactionPromise;
      });
      await flushEffects();

      expect(mockToast.error).not.toHaveBeenCalledWith(
        expect.stringContaining("压缩上下文失败"),
      );
    } finally {
      harness.unmount();
    }
  });

  it("手动压缩上下文返回字符串错误时应透出真实原因", async () => {
    const workspaceId = "ws-context-compaction-error";
    seedSession(workspaceId, "session-context-compaction-error");
    const harness = mountHook(workspaceId);

    mockCompactAgentRuntimeSession.mockRejectedValueOnce(
      "当前会话上下文尚未准备完成，请稍后再试",
    );

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().compactSession();
      });

      expect(mockToast.error).toHaveBeenCalledWith(
        "当前会话上下文尚未准备完成，请稍后再试",
      );
    } finally {
      harness.unmount();
    }
  });

  it("Artifact 恢复提示不应打断为全局 toast", async () => {
    const workspaceId = "ws-artifact-warning-tone";
    seedSession(workspaceId, "session-artifact-warning-tone");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "帮我整理成结构化文稿",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      act(() => {
        stream.emit({
          type: "warning",
          code: "artifact_document_repaired",
          message:
            "ArtifactDocument 已落盘: 已根据正文整理出一份可继续编辑的草稿。",
        });
        stream.emit({
          type: "turn_completed",
          turn: completedTurn(),
        });
      });

      expect(mockToast.info).not.toHaveBeenCalled();
      expect(mockToast.warning).not.toHaveBeenCalledWith(
        expect.stringContaining("ArtifactDocument"),
      );
    } finally {
      harness.unmount();
    }
  });
});
