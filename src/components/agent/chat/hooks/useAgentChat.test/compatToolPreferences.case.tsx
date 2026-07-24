import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  flushEffects,
  getSubmittedTurnMetadata,
  getSubmittedTurnStart,
  mockGetAgentRuntimeSession,
  mockSubmitAgentRuntimeTurn,
  mountHook,
} from "../useAgentChat.testUtils";

describe("useAgentChat 兼容接口 - tool preferences", () => {
  it("已有 recent_preferences.thinking 时不应重复随 turn 提交 thinking_enabled", async () => {
    const workspaceId = "ws-runtime-thinking-reuse";
    const topicId = "topic-runtime-thinking-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: false,
          thinking: true,
          task: false,
          subagent: false,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage("继续沿用深度思考配置", [], false, true, false, "react");
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("thinkingEnabled");
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { preferences?: { thinking?: boolean } };
          } | null
        )?.harness?.preferences?.thinking,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("execution_runtime 缺失但 session recent_preferences 已同步时，不应重复随 turn 提交 thinking_enabled", async () => {
    const workspaceId = "ws-runtime-thinking-shadow-reuse";
    const topicId = "topic-runtime-thinking-shadow-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      messages: [],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId, {
      getSyncedSessionRecentPreferences: (sessionId) =>
        sessionId === topicId
          ? {
              webSearch: false,
              thinking: true,
              task: false,
              subagent: false,
            }
          : null,
    });

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续沿用已同步的 thinking",
            [],
            false,
            true,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("thinkingEnabled");
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { preferences?: { thinking?: boolean } };
          } | null
        )?.harness?.preferences?.thinking,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("旧 thinking positional 开关已变更但 session 仍是旧值时也不随 turn 提交 thinking_enabled", async () => {
    const workspaceId = "ws-runtime-thinking-pending-sync";
    const topicId = "topic-runtime-thinking-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: false,
          thinking: false,
          task: false,
          subagent: false,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "切换 thinking 后立即发送",
            [],
            false,
            true,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("thinkingEnabled");
    } finally {
      harness.unmount();
    }
  });

  it("已有 recent_preferences.thinking 且 metadata 显式携带时，不应重复保留 thinking 偏好", async () => {
    const workspaceId = "ws-runtime-thinking-metadata-reuse";
    const topicId = "topic-runtime-thinking-metadata-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: false,
          thinking: true,
          task: false,
          subagent: false,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续沿用已保存的 thinking metadata 偏好",
            [],
            false,
            true,
            false,
            "react",
            undefined,
            undefined,
            {
              requestMetadata: {
                harness: {
                  preferences: {
                    thinking: true,
                  },
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { preferences?: { thinking?: boolean } };
          } | null
        )?.harness?.preferences?.thinking,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("thinking 已变更且 metadata 显式携带时也应裁掉 retired turn 偏好", async () => {
    const workspaceId = "ws-runtime-thinking-metadata-pending-sync";
    const topicId = "topic-runtime-thinking-metadata-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: false,
          thinking: false,
          task: false,
          subagent: false,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "切换 thinking metadata 后立即发送",
            [],
            false,
            true,
            false,
            "react",
            undefined,
            undefined,
            {
              requestMetadata: {
                harness: {
                  preferences: {
                    thinking: true,
                  },
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("thinkingEnabled");
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { preferences?: { thinking?: boolean } };
          } | null
        )?.harness?.preferences?.thinking,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("已有 recent_preferences.webSearch 时不应重复随 turn 提交 web_search", async () => {
    const workspaceId = "ws-runtime-websearch-reuse";
    const topicId = "topic-runtime-websearch-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: true,
          thinking: false,
          task: false,
          subagent: false,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续按已保存的联网偏好处理",
            [],
            true,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("webSearch");
      expect(getSubmittedTurnStart()).not.toHaveProperty("searchMode");
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { preferences?: { web_search?: boolean } };
          } | null
        )?.harness?.preferences?.web_search,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("execution_runtime 缺失但 session recent_preferences 已同步时，不应重复随 turn 提交 web_search", async () => {
    const workspaceId = "ws-runtime-websearch-shadow-reuse";
    const topicId = "topic-runtime-websearch-shadow-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      messages: [],
      turns: [],
      items: [],
    });

    const harness = mountHook(workspaceId, {
      getSyncedSessionRecentPreferences: (sessionId) =>
        sessionId === topicId
          ? {
              webSearch: true,
              thinking: false,
              task: false,
              subagent: false,
            }
          : null,
    });

    try {
      await flushEffects();
      await act(async () => {
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续沿用已同步的联网偏好",
            [],
            true,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("webSearch");
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { preferences?: { web_search?: boolean } };
          } | null
        )?.harness?.preferences?.web_search,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("旧 webSearch positional 开关已变更但 session 仍是旧值时也不随 turn 提交 web_search", async () => {
    const workspaceId = "ws-runtime-websearch-pending-sync";
    const topicId = "topic-runtime-websearch-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: false,
          thinking: false,
          task: false,
          subagent: false,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "切换联网偏好后立即发送",
            [],
            true,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("webSearch");
    } finally {
      harness.unmount();
    }
  });

  it("webSearch 已变更且 metadata 显式携带时也应裁掉 retired turn 偏好", async () => {
    const workspaceId = "ws-runtime-websearch-metadata-pending-sync";
    const topicId = "topic-runtime-websearch-metadata-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: false,
          thinking: false,
          task: false,
          subagent: false,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "切换 webSearch metadata 后立即发送",
            [],
            false,
            false,
            false,
            "react",
            undefined,
            undefined,
            {
              requestMetadata: {
                harness: {
                  preferences: {
                    web_search: true,
                  },
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(getSubmittedTurnStart()).not.toHaveProperty("webSearch");
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { preferences?: { web_search?: boolean } };
          } | null
        )?.harness?.preferences?.web_search,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("已有 recent_preferences.task/subagent 时不应重复随 turn 提交 metadata 偏好", async () => {
    const workspaceId = "ws-runtime-task-subagent-reuse";
    const topicId = "topic-runtime-task-subagent-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: false,
          thinking: false,
          task: true,
          subagent: true,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "继续沿用已保存的 task/subagent 偏好",
            [],
            false,
            false,
            false,
            "react",
            undefined,
            undefined,
            {
              requestMetadata: {
                harness: {
                  preferences: {
                    task: true,
                    subagent: true,
                  },
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: {
              preferences?: { task?: boolean; subagent?: boolean };
            };
          } | null
        )?.harness?.preferences?.task,
      ).toBeUndefined();
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: {
              preferences?: { task?: boolean; subagent?: boolean };
            };
          } | null
        )?.harness?.preferences?.subagent,
      ).toBeUndefined();
    } finally {
      harness.unmount();
    }
  });

  it("task/subagent 已变更但 session 仍是旧值时，仍应保留 metadata 偏好", async () => {
    const workspaceId = "ws-runtime-task-subagent-pending-sync";
    const topicId = "topic-runtime-task-subagent-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_preferences: {
          webSearch: false,
          thinking: false,
          task: false,
          subagent: false,
        },
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
        await harness.getValue().switchTopic(topicId);
      });
      mockSubmitAgentRuntimeTurn.mockClear();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "切换 task/subagent 后立即发送",
            [],
            false,
            false,
            false,
            "react",
            undefined,
            undefined,
            {
              requestMetadata: {
                harness: {
                  preferences: {
                    task: true,
                    subagent: true,
                  },
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: {
              preferences?: { task?: boolean; subagent?: boolean };
            };
          } | null
        )?.harness?.preferences?.task,
      ).toBe(true);
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: {
              preferences?: { task?: boolean; subagent?: boolean };
            };
          } | null
        )?.harness?.preferences?.subagent,
      ).toBe(true);
    } finally {
      harness.unmount();
    }
  });
});
