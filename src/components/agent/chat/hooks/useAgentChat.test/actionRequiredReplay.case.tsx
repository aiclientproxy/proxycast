import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  captureTurnStream,
  flushEffects,
  mockReplayAgentRuntimeRequest,
  mockRespondAgentRuntimeAction,
  mockToast,
  mockTypedActionResponseHandled,
  mountHook,
  seedSession,
} from "../useAgentChat.testUtils";

describe("useAgentChat action_required 渲染链路 - reply / replay", () => {
  it("replayPendingAction 应调用 replay request 命令并恢复 pendingActions", async () => {
    const workspaceId = "ws-replay-action-required";
    seedSession(workspaceId, "session-replay-action-required");
    const respond = mockTypedActionResponseHandled("ask_user");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("请继续", [], false, false, false, "react");
      });

      act(() => {
        stream.emit({
          type: "action_required",
          request_id: "req-replay-1",
          action_type: "ask_user",
          prompt: "请选择执行模式",
          questions: [
            {
              question: "请选择执行模式",
              options: ["自动执行", "确认后执行"],
            },
          ],
        });
      });

      await act(async () => {
        await harness.getValue().confirmAction({
          requestId: "req-replay-1",
          confirmed: true,
          actionType: "ask_user",
          response: '{"answer":"自动执行"}',
        });
      });

      let assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.actionRequests?.[0]).toMatchObject({
        requestId: "req-replay-1",
        status: "submitted",
      });

      mockReplayAgentRuntimeRequest.mockResolvedValueOnce({
        type: "action_required",
        request_id: "req-replay-1",
        action_type: "ask_user",
        prompt: "请选择执行模式",
        questions: [
          {
            question: "请选择执行模式",
            options: ["自动执行", "确认后执行"],
          },
        ],
        scope: {
          session_id: "session-replay-action-required",
          thread_id: "thread-replay-action-required",
          turn_id: "turn-replay-action-required",
        },
      });

      await act(async () => {
        await expect(
          harness
            .getValue()
            .replayPendingAction("req-replay-1", assistantMessage?.id || ""),
        ).resolves.toBe(true);
      });

      assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(mockReplayAgentRuntimeRequest).toHaveBeenCalledWith({
        session_id: "session-replay-action-required",
        request_id: "req-replay-1",
      });
      expect(harness.getValue().pendingActions).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            requestId: "req-replay-1",
            actionType: "ask_user",
            status: "pending",
            scope: {
              sessionId: "session-replay-action-required",
              threadId: "thread-replay-action-required",
              turnId: "turn-replay-action-required",
            },
          }),
        ]),
      );
      expect(assistantMessage?.actionRequests).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            requestId: "req-replay-1",
            actionType: "ask_user",
            status: "pending",
          }),
        ]),
      );
      expect(
        assistantMessage?.contentParts?.some(
          (part) =>
            part.type === "action_required" &&
            part.actionRequired.requestId === "req-replay-1" &&
            part.actionRequired.status === "pending",
        ),
      ).toBe(true);
      expect(mockToast.success).toHaveBeenCalledWith("已重新拉起待处理请求");
    } finally {
      respond.mockRestore();
      harness.unmount();
    }
  });

  it("fallback ask 缺少 typed server request 时应 fail closed", async () => {
    const workspaceId = "ws-ask-fallback-pending";
    seedSession(workspaceId, "session-ask-fallback-pending");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("请继续", [], false, false, false, "react");
      });

      act(() => {
        stream.emit({
          type: "tool_start",
          tool_id: "tool-fallback-only",
          tool_name: "Ask",
          arguments: JSON.stringify({
            question: "请选择您喜欢的科技风格类型",
            options: ["网络矩阵", "极简未来"],
          }),
        });
      });

      await act(async () => {
        await harness.getValue().confirmAction({
          requestId: "fallback:tool-fallback-only",
          confirmed: true,
          actionType: "ask_user",
          response: '{"answer":"网络矩阵"}',
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(mockRespondAgentRuntimeAction).not.toHaveBeenCalled();
      expect(mockToast.error).toHaveBeenCalledWith(
        "用户输入请求已失效，请等待当前 typed request",
      );
      expect(assistantMessage?.actionRequests?.[0]).toMatchObject({
        requestId: "fallback:tool-fallback-only",
        status: "pending",
      });
    } finally {
      harness.unmount();
    }
  });

  it("current react 不应自动确认 tool_confirmation", async () => {
    const workspaceId = "ws-auto-confirm";
    seedSession(workspaceId, "session-auto-confirm");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("执行命令", [], false, false, false, "react");
      });

      act(() => {
        stream.emit({
          type: "action_required",
          request_id: "req-auto-1",
          action_type: "tool_confirmation",
          tool_name: "bash",
          arguments: { command: "ls" },
          prompt: "是否执行命令",
        });
      });

      await flushEffects();

      expect(mockRespondAgentRuntimeAction).not.toHaveBeenCalled();

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");
      expect(assistantMessage?.actionRequests?.length ?? 0).toBe(1);
    } finally {
      harness.unmount();
    }
  });
});
