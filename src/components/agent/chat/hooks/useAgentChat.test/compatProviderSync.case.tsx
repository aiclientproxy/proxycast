import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  flushEffects,
  getSubmittedTurnStart,
  mockGetAgentRuntimeSession,
  mockScheduleMinimumDelayIdleTask,
  mockSubmitAgentRuntimeTurn,
  mockUpdateAgentRuntimeThreadSettings,
  mountHook,
} from "../useAgentChat.testUtils";

describe("useAgentChat 兼容接口 - provider sync", () => {
  it("已有 executionRuntime 且 provider/model 未变化时不应重复提交偏好", async () => {
    const workspaceId = "ws-runtime-model-reuse";
    const selectedProvider = "openai";
    const selectedModel = "gpt-5.4-mini";
    localStorage.setItem(
      `agent_pref_provider_${workspaceId}`,
      JSON.stringify(selectedProvider),
    );
    localStorage.setItem(
      `agent_pref_model_${workspaceId}`,
      JSON.stringify(selectedModel),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-runtime-model-reuse",
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: "topic-runtime-model-reuse",
        provider_selector: selectedProvider,
        provider_name: "openai",
        model_name: selectedModel,
        source: "session",
      },
      messages: [],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic("topic-runtime-model-reuse");
      });

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续沿用当前模型处理",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("provider");
      expect(getSubmittedTurnStart()).not.toHaveProperty("model");
    } finally {
      harness.unmount();
    }
  });

  it("同 provider 切模型且 session 已同步时不应重复提交 model 偏好", async () => {
    const workspaceId = "ws-runtime-model-switch-same-provider";
    const selectedProvider = "openai";
    const currentModel = "gpt-5.4-mini";
    const nextModel = "gpt-5.4";
    localStorage.setItem(
      `agent_pref_provider_${workspaceId}`,
      JSON.stringify(selectedProvider),
    );
    localStorage.setItem(
      `agent_pref_model_${workspaceId}`,
      JSON.stringify(currentModel),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-runtime-model-switch-same-provider",
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: "topic-runtime-model-switch-same-provider",
        provider_selector: selectedProvider,
        provider_name: "openai",
        model_name: currentModel,
        source: "session",
      },
      messages: [],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .switchTopic("topic-runtime-model-switch-same-provider");
      });

      act(() => {
        harness.getValue().setModel(nextModel);
      });
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "切换到同 provider 的另一个模型",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("provider");
      expect(getSubmittedTurnStart()).not.toHaveProperty("model");
    } finally {
      harness.unmount();
    }
  });

  it("同 provider 切模型时应只通过 thread settings 同步，不随 turn 重复提交", async () => {
    const workspaceId = "ws-runtime-model-switch-pending-sync";
    const selectedProvider = "openai";
    const currentModel = "gpt-5.4-mini";
    const nextModel = "gpt-5.4";
    let resolveProviderSync: (() => void) | null = null;
    localStorage.setItem(
      `agent_pref_provider_${workspaceId}`,
      JSON.stringify(selectedProvider),
    );
    localStorage.setItem(
      `agent_pref_model_${workspaceId}`,
      JSON.stringify(currentModel),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: "topic-runtime-model-switch-pending-sync",
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: "topic-runtime-model-switch-pending-sync",
        provider_selector: selectedProvider,
        provider_name: "openai",
        model_name: currentModel,
        source: "session",
      },
      messages: [],
      turns: [],
      items: [],
    });
    mockUpdateAgentRuntimeThreadSettings.mockImplementation((request) => {
      if (request?.modelProvider || request?.model) {
        return new Promise<void>((resolve) => {
          resolveProviderSync = resolve;
        });
      }
      return Promise.resolve();
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness
          .getValue()
          .switchTopic("topic-runtime-model-switch-pending-sync");
      });

      act(() => {
        harness.getValue().setModel(nextModel);
      });
      await flushEffects();

      let sendPromise: Promise<void>;
      await act(async () => {
        sendPromise = harness
          .getValue()
          .sendMessage(
            "切换到同 provider 的另一个模型，但 session 还没同步完",
            [],
            false,
            false,
            false,
            "react",
          );
        await Promise.resolve();
      });

      expect(mockUpdateAgentRuntimeThreadSettings).toHaveBeenCalledWith({
        threadId: "topic-runtime-model-switch-pending-sync",
        modelProvider: selectedProvider,
        model: nextModel,
        effort: null,
      });
      expect(mockSubmitAgentRuntimeTurn).not.toHaveBeenCalled();

      await act(async () => {
        (resolveProviderSync as (() => void) | null)?.();
        await sendPromise!;
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("provider");
      expect(getSubmittedTurnStart()).not.toHaveProperty("model");
    } finally {
      (resolveProviderSync as (() => void) | null)?.();
      harness.unmount();
    }
  });

  it("execution_runtime 缺失但 session provider/model 已迁移回写后，不应重复随 turn 提交", async () => {
    const workspaceId = "ws-runtime-model-shadow-reuse";
    const topicId = "topic-runtime-model-shadow-reuse";
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
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续沿用刚迁移回写的模型处理",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("provider");
      expect(getSubmittedTurnStart()).not.toHaveProperty("model");
    } finally {
      harness.unmount();
    }
  });

  it("execution_runtime 缺失且 session provider/model 回写未完成时，应随 turn 提交偏好", async () => {
    const workspaceId = "ws-runtime-model-shadow-pending-sync";
    const topicId = "topic-runtime-model-shadow-pending-sync";
    const selectedProvider = "custom-cb381b4f-d2fa-4eff-ba22-c867c38ba8d3";
    const selectedModel = "gpt-5.5";
    const scheduledTasks: Array<() => void> = [];
    localStorage.setItem(
      `agent_topic_model_pref_${workspaceId}_${topicId}`,
      JSON.stringify({
        providerType: selectedProvider,
        model: selectedModel,
      }),
    );
    mockScheduleMinimumDelayIdleTask.mockImplementation((task: () => void) => {
      scheduledTasks.push(task);
      return () => undefined;
    });
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
      expect(scheduledTasks.length).toBeGreaterThan(0);
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续沿用本地话题模型处理",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(harness.getValue().providerType).toBe(selectedProvider);
      expect(getSubmittedTurnStart()?.model).toBe(selectedModel);
    } finally {
      harness.unmount();
    }
  });

  it("导入会话的来源模型不应阻止续聊提交当前 Lime provider/model", async () => {
    const workspaceId = "ws-imported-source-runtime-model";
    const topicId = "topic-imported-source-runtime-model";
    const selectedProvider = "custom-current-provider";
    const selectedModel = "gpt-5.5";
    localStorage.setItem(
      `agent_pref_provider_${workspaceId}`,
      JSON.stringify(selectedProvider),
    );
    localStorage.setItem(
      `agent_pref_model_${workspaceId}`,
      JSON.stringify(selectedModel),
    );
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      messages: [],
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        source: "session",
        provider_name: "openai",
        model_name: "gpt-5.4",
        source_client: "codex",
        imported_continuation: {
          modelProvider: "openai",
          model: "gpt-5.4",
        },
      },
    });

    const harness = mountHook(workspaceId);

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      await flushEffects();
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "基于导入历史继续处理",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(harness.getValue().providerType).toBe(selectedProvider);
      expect(getSubmittedTurnStart()?.model).toBe(selectedModel);
    } finally {
      harness.unmount();
    }
  });
});
