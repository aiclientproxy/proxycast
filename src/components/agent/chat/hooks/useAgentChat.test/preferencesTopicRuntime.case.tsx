import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  flushEffects,
  mockGetAgentRuntimeSession,
  mockUpdateAgentRuntimeThreadSettings,
  mountHook,
} from "../useAgentChat.testUtils";

describe("useAgentChat 偏好持久化 - topic runtime fallback", () => {
  it("切换话题时应优先从 execution_runtime 恢复 provider/model", async () => {
    const workspaceId = "ws-topic-runtime-priority";
    const topicId = "topic-runtime-priority";
    localStorage.setItem(
      `agent_pref_provider_${workspaceId}`,
      JSON.stringify("deepseek"),
    );
    localStorage.setItem(
      `agent_pref_model_${workspaceId}`,
      JSON.stringify("deepseek-chat"),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      messages: [],
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        provider_selector: "openai",
        provider_name: "openai",
        model_name: "gpt-5.4-mini",
        source: "session",
      },
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      await flushEffects();

      const value = harness.getValue();
      expect(value.providerType).toBe("openai");
      expect(value.model).toBe("gpt-5.4-mini");
      expect(
        JSON.parse(
          localStorage.getItem(
            `agent_topic_model_pref_${workspaceId}_${topicId}`,
          ) || "null",
        ),
      ).toEqual({
        providerType: "openai",
        model: "gpt-5.4-mini",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时 execution_runtime 缺失应回退本地 session preference", async () => {
    const workspaceId = "ws-topic-runtime-fallback";
    const topicId = "topic-runtime-fallback";
    localStorage.setItem(
      `agent_pref_provider_${workspaceId}`,
      JSON.stringify("openai"),
    );
    localStorage.setItem(
      `agent_pref_model_${workspaceId}`,
      JSON.stringify("gpt-5.4-mini"),
    );
    localStorage.setItem(
      `agent_topic_model_pref_${workspaceId}_${topicId}`,
      JSON.stringify({
        providerType: "gemini",
        model: "gemini-2.5-pro",
      }),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      messages: [],
      execution_strategy: "react",
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      await flushEffects();

      const value = harness.getValue();
      expect(value.providerType).toBe("gemini");
      expect(value.model).toBe("gemini-2.5-pro");
      expect(mockUpdateAgentRuntimeThreadSettings).toHaveBeenCalledWith({
        threadId: topicId,
        approvalPolicy: "never",
        permissions: ":danger-full-access",
        modelProvider: "gemini",
        model: "gemini-2.5-pro",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时 execution_strategy 缺失应回退工作区影子缓存并回写 session", async () => {
    const workspaceId = "ws-topic-strategy-fallback";
    const topicId = "topic-strategy-fallback";
    localStorage.setItem(
      `agent_execution_strategy_${workspaceId}`,
      JSON.stringify("auto"),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      messages: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      await flushEffects();

      expect(harness.getValue().executionStrategy).toBe("react");
      expect(mockUpdateAgentRuntimeThreadSettings).toHaveBeenCalledWith({
        threadId: topicId,
        approvalPolicy: "never",
        permissions: ":danger-full-access",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时 recent_access_mode 缺失应回退本地 session access shadow 并回写 session", async () => {
    const workspaceId = "ws-topic-access-fallback";
    const topicId = "topic-access-fallback";
    localStorage.setItem(
      `agent_session_access_mode_${workspaceId}_${topicId}`,
      JSON.stringify("full-access"),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      messages: [],
      execution_strategy: "react",
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      await flushEffects();

      expect(harness.getValue().accessMode).toBe("full-access");
      expect(mockUpdateAgentRuntimeThreadSettings).toHaveBeenCalledWith({
        threadId: topicId,
        approvalPolicy: "never",
        permissions: ":danger-full-access",
      });
    } finally {
      harness.unmount();
    }
  });

  it("切换话题时 recent_access_mode 与 session access shadow 都缺失应回退工作区默认并回写 session", async () => {
    const workspaceId = "ws-topic-access-workspace-default";
    const topicId = "topic-access-workspace-default";
    localStorage.setItem(
      `agent_access_mode_${workspaceId}`,
      JSON.stringify("read-only"),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      messages: [],
      execution_strategy: "react",
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      await flushEffects();

      expect(harness.getValue().accessMode).toBe("read-only");
      expect(mockUpdateAgentRuntimeThreadSettings).toHaveBeenCalledWith({
        threadId: topicId,
        approvalPolicy: "on-request",
        permissions: ":read-only",
      });
    } finally {
      harness.unmount();
    }
  });
});
