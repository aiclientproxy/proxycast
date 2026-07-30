import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  captureTurnStream,
  flushEffects,
  mockRespondAgentRuntimeAction,
  mockResolveClawWorkspaceProviderSelection,
  mockTypedActionResponseHandled,
  mountHook,
  seedSession,
} from "../useAgentChat.testUtils";

describe("useAgentChat action_required 渲染链路 - ask / elicitation", () => {
  it("仅收到 Ask 工具调用时应兜底渲染提问面板", async () => {
    const workspaceId = "ws-ask-fallback";
    seedSession(workspaceId, "session-ask-fallback");
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
          tool_id: "tool-ask-1",
          tool_name: "Ask",
          arguments: JSON.stringify({
            question: "你希望海报主色调是什么？",
            options: ["蓝紫", "赛博绿"],
          }),
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage).toBeDefined();
      expect(assistantMessage?.actionRequests?.[0]?.actionType).toBe(
        "ask_user",
      );
      expect(
        assistantMessage?.actionRequests?.[0]?.questions?.[0]?.question,
      ).toBe("你希望海报主色调是什么？");
      expect(
        assistantMessage?.actionRequests?.[0]?.questions?.[0]?.options?.map(
          (item) => item.label,
        ),
      ).toEqual(["蓝紫", "赛博绿"]);
    } finally {
      harness.unmount();
    }
  });

  it("Ask fallback 应优先使用参数中的 id 作为 requestId", async () => {
    const workspaceId = "ws-ask-fallback-id";
    seedSession(workspaceId, "session-ask-fallback-id");
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
          tool_id: "tool-ask-fallback-id",
          tool_name: "Ask",
          arguments: JSON.stringify({
            id: "req-from-ask-arg",
            question: "你希望主色调是什么？",
          }),
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.actionRequests?.[0]?.requestId).toBe(
        "req-from-ask-arg",
      );
    } finally {
      harness.unmount();
    }
  });

  it("收到 action_required 后应写入消息 actionRequests 与 contentParts", async () => {
    const workspaceId = "ws-action-required";
    seedSession(workspaceId, "session-action-required");
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
          request_id: "req-ar-1",
          action_type: "elicitation",
          prompt: "请选择一个方案",
          requested_schema: {
            type: "object",
            properties: {
              answer: {
                type: "string",
                enum: ["A", "B"],
              },
            },
          },
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage).toBeDefined();
      expect(assistantMessage?.actionRequests?.[0]?.requestId).toBe("req-ar-1");
      expect(
        assistantMessage?.contentParts?.some(
          (part) =>
            part.type === "action_required" &&
            part.actionRequired.requestId === "req-ar-1",
        ),
      ).toBe(true);
    } finally {
      harness.unmount();
    }
  });

  it("新一轮普通发送后不应继续暴露上一轮 pendingActions", async () => {
    const workspaceId = "ws-action-required-current-tail";
    seedSession(workspaceId, "session-action-required-current-tail");
    const harness = mountHook(workspaceId);
    const stream = captureTurnStream();

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("第一次请求", [], false, false, false, "react");
      });

      act(() => {
        stream.emit({
          type: "action_required",
          request_id: "req-stale-tail-1",
          action_type: "ask_user",
          prompt: "请补充继续执行所需信息",
          questions: [
            {
              question: "请补充继续执行所需信息",
            },
          ],
        });
      });

      expect(harness.getValue().pendingActions).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            requestId: "req-stale-tail-1",
            sourceMessageId: expect.any(String),
          }),
        ]),
      );

      act(() => {
        stream.emit({
          type: "error",
          message: "Provider 402 Payment Required",
        });
      });
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("第二次继续", [], false, false, false, "react");
      });

      expect(harness.getValue().pendingActions).toEqual([]);
      const staleAssistantMessage = harness
        .getValue()
        .messages.find((message) =>
          message.actionRequests?.some(
            (request) => request.requestId === "req-stale-tail-1",
          ),
        );
      expect(staleAssistantMessage?.actionRequests?.[0]).toMatchObject({
        requestId: "req-stale-tail-1",
        status: "pending",
      });
    } finally {
      harness.unmount();
    }
  });

  it("收到带 scope 的 action_required 后应保留作用域，并在提交时透传 action_scope", async () => {
    const workspaceId = "ws-action-required-scope";
    seedSession(workspaceId, "session-action-required-scope");
    const respond = mockTypedActionResponseHandled(
      "ask_user",
      "req-ar-scope-1",
    );
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
          request_id: "req-ar-scope-1",
          action_type: "ask_user",
          prompt: "请选择执行模式",
          scope: {
            session_id: "session-action-required-scope",
            thread_id: "thread-action-required-scope",
            turn_id: "turn-action-required-scope",
          },
          questions: [
            {
              question: "请选择执行模式",
              options: ["自动执行", "确认后执行"],
            },
          ],
        });
      });

      let assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.actionRequests?.[0]?.scope).toEqual({
        sessionId: "session-action-required-scope",
        threadId: "thread-action-required-scope",
        turnId: "turn-action-required-scope",
      });

      await act(async () => {
        await harness.getValue().confirmAction({
          requestId: "req-ar-scope-1",
          confirmed: true,
          actionType: "ask_user",
          response: '{"answer":"自动执行"}',
        });
      });

      assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(respond).toHaveBeenCalledWith({
        confirmed: true,
        interactionId: "request_user_input:test:req-ar-scope-1",
        kind: "request_user_input",
        response: '{"answer":"自动执行"}',
        userData: { answer: "自动执行" },
      });
      expect(mockRespondAgentRuntimeAction).not.toHaveBeenCalled();
      expect(assistantMessage?.actionRequests?.[0]).toMatchObject({
        requestId: "req-ar-scope-1",
        status: "submitted",
        scope: {
          sessionId: "session-action-required-scope",
          threadId: "thread-action-required-scope",
          turnId: "turn-action-required-scope",
        },
      });
    } finally {
      respond.mockRestore();
      harness.unmount();
    }
  });

  it("tool_confirmation 提交时应透传 action_scope 以恢复原 turn", async () => {
    const workspaceId = "ws-tool-confirmation-scope";
    seedSession(workspaceId, "session-tool-confirmation-scope");
    mockResolveClawWorkspaceProviderSelection.mockResolvedValue({
      providerType: "openai",
      model: "gpt-5.4-mini",
    });
    const respond = mockTypedActionResponseHandled(
      "tool_confirmation",
      "req-tool-confirm-scope-1",
    );
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
          request_id: "req-tool-confirm-scope-1",
          action_type: "tool_confirmation",
          prompt: "允许执行测试命令？",
          tool_name: "Shell",
          arguments: {
            command: "echo approval-resume",
          },
          scope: {
            session_id: "session-tool-confirmation-scope",
            thread_id: "thread-tool-confirmation-scope",
            turn_id: "turn-tool-confirmation-scope",
          },
        });
      });

      const sourceEventName = stream.getEventName();
      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.actionRequests?.[0]?.scope).toEqual({
        sessionId: "session-tool-confirmation-scope",
        threadId: "thread-tool-confirmation-scope",
        turnId: "turn-tool-confirmation-scope",
      });

      await act(async () => {
        await harness.getValue().confirmAction({
          requestId: "req-tool-confirm-scope-1",
          confirmed: true,
          actionType: "tool_confirmation",
          response: "允许",
        });
      });

      expect(respond).toHaveBeenCalledWith({
        interactionId: "approval:test:req-tool-confirm-scope-1",
        kind: "approval",
        response: {
          requestId: "req-tool-confirm-scope-1",
          confirmed: true,
          actionType: "tool_confirmation",
          response: "允许",
        },
      });
      expect(mockRespondAgentRuntimeAction).not.toHaveBeenCalled();
      expect(sourceEventName).toMatch(/^agent_stream_/);
    } finally {
      respond.mockRestore();
      harness.unmount();
    }
  });

  it("runtime 权限确认在 stream 结束后提交时仍应透传来源 event_name", async () => {
    const workspaceId = "ws-runtime-permission-event-name";
    seedSession(workspaceId, "session-runtime-permission-event-name");
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
          request_id: "runtime_permission_confirmation:turn-1",
          action_type: "elicitation",
          prompt: "当前执行需要确认运行时权限：web_search。",
          scope: {
            session_id: "session-runtime-permission-event-name",
            thread_id: "thread-runtime-permission-event-name",
            turn_id: "turn-runtime-permission-event-name",
          },
          questions: [
            {
              header: "运行时权限确认",
              question: "当前执行需要确认运行时权限：web_search。",
              options: ["允许本次执行", "拒绝"],
            },
          ],
        });
      });

      const sourceEventName = stream.getEventName();

      act(() => {
        stream.emit({
          type: "error",
          message:
            "运行时权限声明需要真实确认，当前 turn 已在模型执行前等待用户确认：confirmationStatus=not_requested，askProfileKeys=web_search。已创建真实权限确认请求；请确认后重试或恢复本轮执行。",
        });
      });

      await flushEffects();

      await act(async () => {
        await harness.getValue().confirmAction({
          requestId: "runtime_permission_confirmation:turn-1",
          confirmed: true,
          actionType: "elicitation",
          response: '{"answer":"允许本次执行"}',
        });
      });

      expect(mockRespondAgentRuntimeAction).toHaveBeenCalledWith(
        expect.objectContaining({
          session_id: "session-runtime-permission-event-name",
          request_id: "runtime_permission_confirmation:turn-1",
          action_type: "elicitation",
          confirmed: true,
          event_name: sourceEventName,
          action_scope: {
            session_id: "session-runtime-permission-event-name",
            thread_id: "thread-runtime-permission-event-name",
            turn_id: "turn-runtime-permission-event-name",
          },
        }),
      );
    } finally {
      harness.unmount();
    }
  });

  it("action_required 的字符串 options 应归一化为可展示选项", async () => {
    const workspaceId = "ws-action-required-options";
    seedSession(workspaceId, "session-action-required-options");
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
          request_id: "req-ar-options-1",
          action_type: "ask_user",
          prompt: "请选择执行模式",
          questions: [
            {
              question: "请选择执行模式",
              options: ["自动执行（Auto）", "确认后执行（Ask）"],
            },
          ],
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(
        assistantMessage?.actionRequests?.[0]?.questions?.[0]?.options?.map(
          (option) => option.label,
        ),
      ).toEqual(["自动执行（Auto）", "确认后执行（Ask）"]);
    } finally {
      harness.unmount();
    }
  });

  it("ask_user 多问题时应在进入聊天状态前裁剪为单轮单问", async () => {
    const workspaceId = "ws-action-required-governed-ask";
    seedSession(workspaceId, "session-action-required-governed-ask");
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
          request_id: "req-ar-governed-ask-1",
          action_type: "ask_user",
          prompt: "继续前先确认几个点",
          questions: [
            {
              question: "你希望我先聚焦哪一部分？",
            },
            {
              question: "这一步更看重速度还是完整度？",
            },
          ],
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.actionRequests?.[0]).toMatchObject({
        requestId: "req-ar-governed-ask-1",
        governance: {
          originalQuestionCount: 2,
          deferredQuestionCount: 1,
        },
      });
      expect(assistantMessage?.actionRequests?.[0]?.questions).toEqual([
        {
          question: "你希望我先聚焦哪一部分？",
          multiSelect: false,
        },
      ]);
    } finally {
      harness.unmount();
    }
  });

  it("elicitation 多字段时应在进入聊天状态前裁剪为单轮单字段", async () => {
    const workspaceId = "ws-action-required-governed-elicitation";
    seedSession(workspaceId, "session-action-required-governed-elicitation");
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
          request_id: "req-ar-governed-elicitation-1",
          action_type: "elicitation",
          prompt: "补充创作约束",
          requested_schema: {
            type: "object",
            required: ["topic", "style"],
            properties: {
              topic: {
                type: "string",
                title: "主题",
              },
              style: {
                type: "string",
                title: "风格",
              },
            },
          },
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.actionRequests?.[0]).toMatchObject({
        requestId: "req-ar-governed-elicitation-1",
        governance: {
          originalFieldCount: 2,
          retainedFieldKey: "topic",
          deferredFieldCount: 1,
        },
        requestedSchema: {
          type: "object",
          required: ["topic"],
          properties: {
            topic: {
              type: "string",
              title: "主题",
            },
          },
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("elicitation 缺少 questions 时应从 requested_schema 扩展恢复 rich question，并在治理后只保留当前一问", async () => {
    const workspaceId = "ws-action-required-rich-elicitation";
    seedSession(workspaceId, "session-action-required-rich-elicitation");
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
          request_id: "req-ar-rich-elicitation-1",
          action_type: "elicitation",
          prompt: "继续前请确认执行模式和范围",
          requested_schema: {
            type: "object",
            required: ["mode", "scope"],
            properties: {
              mode: {
                type: "string",
                title: "执行模式",
              },
              scope: {
                type: "string",
                title: "执行范围",
              },
            },
            "x-lime-ask-user-questions": [
              {
                question: "请选择执行模式",
                header: "mode",
                options: [
                  {
                    label: "自动执行",
                    description: "直接继续推进",
                  },
                  {
                    value: "confirm",
                    label: "确认后执行",
                    description: "每一步都等我确认",
                  },
                ],
                multiSelect: false,
              },
              {
                question: "请选择执行范围",
                header: "scope",
                options: ["仅修改 ask", "顺手整理上下游"],
                multiSelect: false,
              },
            ],
          },
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(assistantMessage?.actionRequests?.[0]).toMatchObject({
        requestId: "req-ar-rich-elicitation-1",
        actionType: "elicitation",
        questions: [
          {
            question: "请选择执行模式",
            header: "mode",
            options: [
              {
                label: "自动执行",
                description: "直接继续推进",
              },
              {
                label: "确认后执行",
                description: "每一步都等我确认",
              },
            ],
            multiSelect: false,
          },
        ],
        governance: {
          originalQuestionCount: 2,
          deferredQuestionCount: 1,
          originalFieldCount: 2,
          retainedFieldKey: "mode",
          deferredFieldCount: 1,
        },
        requestedSchema: {
          type: "object",
          required: ["mode"],
          properties: {
            mode: {
              type: "string",
              title: "执行模式",
            },
          },
          "x-lime-ask-user-questions": [
            {
              question: "请选择执行模式",
              header: "mode",
              options: [
                {
                  label: "自动执行",
                  description: "直接继续推进",
                },
                {
                  label: "确认后执行",
                  description: "每一步都等我确认",
                },
              ],
              multiSelect: false,
            },
          ],
        },
      });
    } finally {
      harness.unmount();
    }
  });

  it("ask_user 提交后应保留只读回显，避免面板消失", async () => {
    const workspaceId = "ws-ask-submit-keep";
    seedSession(workspaceId, "session-ask-submit-keep");
    const respond = mockTypedActionResponseHandled(
      "ask_user",
      "req-ask-submit-1",
    );
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
          request_id: "req-ask-submit-1",
          action_type: "ask_user",
          prompt: "请选择执行模式",
          questions: [{ question: "你希望如何执行？" }],
        });
      });

      await act(async () => {
        await harness.getValue().confirmAction({
          requestId: "req-ask-submit-1",
          confirmed: true,
          actionType: "ask_user",
          response: '{"answer":"自动执行（Auto）"}',
        });
      });

      const assistantMessage = [...harness.getValue().messages]
        .reverse()
        .find((msg) => msg.role === "assistant");

      expect(respond).toHaveBeenCalledWith({
        confirmed: true,
        interactionId: "request_user_input:test:req-ask-submit-1",
        kind: "request_user_input",
        response: '{"answer":"自动执行（Auto）"}',
        userData: { answer: "自动执行（Auto）" },
      });
      expect(mockRespondAgentRuntimeAction).not.toHaveBeenCalled();
      expect(assistantMessage?.actionRequests?.[0]).toMatchObject({
        requestId: "req-ask-submit-1",
        actionType: "ask_user",
        status: "submitted",
        submittedResponse: '{"answer":"自动执行（Auto）"}',
        submittedUserData: { answer: "自动执行（Auto）" },
      });
      expect(assistantMessage?.runtimeStatus).toMatchObject({
        phase: "routing",
        title: "已收到补充信息，继续处理中",
      });
      expect(
        assistantMessage?.contentParts?.some(
          (part) =>
            part.type === "action_required" &&
            part.actionRequired.requestId === "req-ask-submit-1" &&
            part.actionRequired.status === "submitted",
        ),
      ).toBe(true);
    } finally {
      respond.mockRestore();
      harness.unmount();
    }
  });
});
