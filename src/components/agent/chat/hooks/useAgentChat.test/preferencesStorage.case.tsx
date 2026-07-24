import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import {
  flushEffects,
  mockCreateAgentRuntimeSession,
  mockGetAgentRuntimeSession,
  mockSubmitAgentRuntimeTurn,
  mountHook,
  seedSession,
} from "../useAgentChat.testUtils";

const expectCurrentProviderRoutingMetadata = () =>
  expect.objectContaining({
    executionRuntime: expect.objectContaining({
      modelName: "gpt-5.4-mini",
      providerSelector: "openai",
    }),
    extensionData: expect.objectContaining({
      "lime_provider_routing.v0": expect.objectContaining({
        providerSelector: "openai",
      }),
    }),
    modelName: "gpt-5.4-mini",
    providerSelector: "openai",
  });

describe("useAgentChat 偏好持久化 - storage cleanup", () => {
  it("初始化时应清理 sessionStorage 中空白 user 消息", async () => {
    const workspaceId = "ws-clean-blank-user";
    sessionStorage.setItem(
      `agent_messages_${workspaceId}`,
      JSON.stringify([
        {
          id: "blank-user",
          role: "user",
          content: "",
          timestamp: new Date().toISOString(),
        },
        {
          id: "assistant-text",
          role: "assistant",
          content: "hello",
          timestamp: new Date().toISOString(),
        },
      ]),
    );

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      const value = harness.getValue();
      expect(value.messages).toHaveLength(1);
      expect(value.messages[0]?.role).toBe("assistant");
      expect(value.messages[0]?.content).toBe("hello");
    } finally {
      harness.unmount();
    }
  });

  it("初始化时应将仅含工具轨迹的空白 user 消息归一为 assistant", async () => {
    const workspaceId = "ws-normalize-tool-user";
    sessionStorage.setItem(
      `agent_messages_${workspaceId}`,
      JSON.stringify([
        {
          id: "legacy-user-tool",
          role: "user",
          content: "",
          toolCalls: [
            {
              id: "tool_1",
              name: "bash",
              status: "completed",
              result: {
                success: true,
                output: "ok",
              },
              startTime: new Date().toISOString(),
              endTime: new Date().toISOString(),
            },
          ],
          timestamp: new Date().toISOString(),
        },
      ]),
    );

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      const value = harness.getValue();
      expect(value.messages).toHaveLength(1);
      expect(value.messages[0]?.role).toBe("assistant");
      expect(value.messages[0]?.toolCalls?.[0]).toMatchObject({
        id: "tool_1",
        status: "completed",
      });
    } finally {
      harness.unmount();
    }
  });

  it("初始化时应丢弃带 fallback 工具名的旧缓存消息并触发回源", async () => {
    const workspaceId = "ws-drop-fallback-tool-name-cache";
    sessionStorage.setItem(
      `agent_messages_${workspaceId}`,
      JSON.stringify([
        {
          id: "legacy-fallback-tool-name",
          role: "assistant",
          content: "",
          toolCalls: [
            {
              id: "call_324abc",
              name: "工具调用 call_324abc",
              status: "completed",
              result: {
                success: true,
                output: "Launching skill: canvas-design",
              },
              startTime: new Date().toISOString(),
              endTime: new Date().toISOString(),
            },
          ],
          timestamp: new Date().toISOString(),
        },
      ]),
    );

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      const value = harness.getValue();
      expect(value.messages).toHaveLength(0);
    } finally {
      harness.unmount();
    }
  });

  it("应将旧全局偏好迁移到当前工作区", async () => {
    localStorage.setItem("agent_pref_provider", JSON.stringify("gemini"));
    localStorage.setItem("agent_pref_model", JSON.stringify("gemini-2.5-pro"));

    const workspaceId = "ws-migrate";
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      const value = harness.getValue();
      expect(value.providerType).toBe("gemini");
      expect(value.model).toBe("gemini-2.5-pro");
      expect(
        JSON.parse(
          localStorage.getItem(`agent_pref_provider_${workspaceId}`) || "null",
        ),
      ).toBe("gemini");
      expect(
        JSON.parse(
          localStorage.getItem(`agent_pref_model_${workspaceId}`) || "null",
        ),
      ).toBe("gemini-2.5-pro");
      expect(
        JSON.parse(
          localStorage.getItem(`agent_pref_migrated_${workspaceId}`) || "false",
        ),
      ).toBe(true);
    } finally {
      harness.unmount();
    }
  });

  it("应优先使用工作区偏好而不是旧全局偏好", async () => {
    localStorage.setItem("agent_pref_provider", JSON.stringify("claude"));
    localStorage.setItem("agent_pref_model", JSON.stringify("claude-legacy"));
    localStorage.setItem(
      "agent_pref_provider_ws-prefer-scoped",
      JSON.stringify("deepseek"),
    );
    localStorage.setItem(
      "agent_pref_model_ws-prefer-scoped",
      JSON.stringify("deepseek-reasoner"),
    );

    const harness = mountHook("ws-prefer-scoped");

    try {
      await flushEffects();

      const value = harness.getValue();
      expect(value.providerType).toBe("deepseek");
      expect(value.model).toBe("deepseek-reasoner");
    } finally {
      harness.unmount();
    }
  });

  it("无工作区时应保留全局模型偏好（切主题不丢失）", async () => {
    const firstMount = mountHook("");

    try {
      await flushEffects();
      act(() => {
        firstMount.getValue().setProviderType("gemini");
        firstMount.getValue().setModel("gemini-2.5-pro");
      });
      await flushEffects();
    } finally {
      firstMount.unmount();
    }

    const secondMount = mountHook("");
    try {
      await flushEffects();
      const value = secondMount.getValue();
      expect(value.providerType).toBe("gemini");
      expect(value.model).toBe("gemini-2.5-pro");
      expect(
        JSON.parse(
          localStorage.getItem("agent_pref_provider_global") || "null",
        ),
      ).toBe("gemini");
      expect(
        JSON.parse(localStorage.getItem("agent_pref_model_global") || "null"),
      ).toBe("gemini-2.5-pro");
    } finally {
      secondMount.unmount();
    }
  });

  it("会话已绑定其他工作区时不应覆盖 agent_session_workspace 映射", async () => {
    const consoleWarnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => {});
    const workspaceId = "ws-current";
    const sessionId = "session-conflict";
    seedSession(workspaceId, sessionId);
    localStorage.setItem(
      `agent_session_workspace_${sessionId}`,
      JSON.stringify("ws-legacy"),
    );

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      expect(
        JSON.parse(
          localStorage.getItem(`agent_session_workspace_${sessionId}`) ||
            "null",
        ),
      ).toBe("ws-legacy");
    } finally {
      consoleWarnSpy.mockRestore();
      harness.unmount();
    }
  });

  it("会话映射为空占位时应写入当前工作区", async () => {
    const workspaceId = "ws-current";
    const sessionId = "session-invalid-placeholder";
    seedSession(workspaceId, sessionId);
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: sessionId,
      messages: [],
      turns: [],
      items: [],
      queued_turns: [],
    });
    localStorage.setItem(
      `agent_session_workspace_${sessionId}`,
      JSON.stringify("__invalid__"),
    );

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();
      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        sessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(
        JSON.parse(
          localStorage.getItem(`agent_session_workspace_${sessionId}`) ||
            "null",
        ),
      ).toBe(workspaceId);
    } finally {
      harness.unmount();
    }
  });

  it("legacy workspace-default 映射不应再触发旧会话恢复", async () => {
    const workspaceId = "ws-current";
    const sessionId = "session-legacy-workspace-default";
    seedSession(workspaceId, sessionId);
    localStorage.setItem(
      `agent_session_workspace_${sessionId}`,
      JSON.stringify("workspace-default"),
    );

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      expect(harness.getValue().sessionId).toBeNull();
      let ensuredSessionId: string | null = null;
      await act(async () => {
        ensuredSessionId = await harness.getValue().ensureSession();
      });
      expect(ensuredSessionId).toBe("created-session");
      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledWith(
        workspaceId,
        undefined,
        "react",
        {
          runStartHooks: true,
          workingDir: null,
          metadata: expectCurrentProviderRoutingMetadata(),
        },
      );
    } finally {
      harness.unmount();
    }
  });

  it("发送前应丢弃 App Server 已不存在的本地恢复会话并新建", async () => {
    const workspaceId = "ws-stale-restore-submit";
    const staleSessionId = "session-stale-restore-submit";
    const freshSessionId = "session-fresh-after-stale-restore";
    seedSession(workspaceId, staleSessionId);
    mockCreateAgentRuntimeSession.mockResolvedValue(freshSessionId);
    mockGetAgentRuntimeSession.mockImplementation(async (sessionId: string) => {
      if (sessionId === staleSessionId) {
        throw new Error(`session not found: ${staleSessionId}`);
      }

      return {
        id: sessionId,
        workspace_id: workspaceId,
        messages: [],
        turns: [],
        items: [],
        queued_turns: [],
      };
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await flushEffects();

      expect(mockGetAgentRuntimeSession).toHaveBeenCalledWith(
        staleSessionId,
        expect.objectContaining({ historyLimit: 40 }),
      );
      expect(harness.getValue().sessionId).toBeNull();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("继续处理这个任务", [], false, false, false, "react");
      });

      expect(mockCreateAgentRuntimeSession).toHaveBeenCalledWith(
        workspaceId,
        undefined,
        "react",
        {
          runStartHooks: true,
          workingDir: null,
          metadata: expectCurrentProviderRoutingMetadata(),
        },
      );
      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledWith(
        expect.objectContaining({
          input: [{ type: "text", text: "继续处理这个任务" }],
          threadId: freshSessionId,
        }),
      );
      expect(harness.getValue().sessionId).toBe(freshSessionId);
      expect(
        sessionStorage.getItem(`agent_curr_sessionId_${workspaceId}`),
      ).toBe(JSON.stringify(freshSessionId));
    } finally {
      harness.unmount();
    }
  });
});
