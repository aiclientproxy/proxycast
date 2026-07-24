import { act } from "react";
import { describe, expect, it } from "vitest";
import {
  flushEffects,
  getSubmittedTurnMetadata,
  mockGetAgentRuntimeSession,
  mockSubmitAgentRuntimeTurn,
  mountHook,
} from "../useAgentChat.testUtils";

const expectHarnessMetadataRemoved = () => {
  const metadata = getSubmittedTurnMetadata();
  expect(metadata?.harness).toBeUndefined();
  expect(metadata?.agentUiPerformanceTrace).toEqual(expect.any(Object));
};

describe("useAgentChat 兼容接口 - runtime metadata", () => {
  it("已有会话时不应重复随 turn 提交 workspace_id", async () => {
    const workspaceId = "ws-runtime-workspace-reuse";
    const topicId = "topic-runtime-workspace-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      workspace_id: workspaceId,
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
            "继续沿用当前会话工作区",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]).not.toHaveProperty(
        "workspace_id",
      );
    } finally {
      harness.unmount();
    }
  });

  it("首次创建新会话发送时应通过建会话绑定 workspace", async () => {
    const workspaceId = "ws-runtime-workspace-bootstrap";
    const harness = mountHook(workspaceId);

    try {
      await flushEffects();

      await act(async () => {
        await harness
          .getValue()
          .sendMessage(
            "首条消息需要绑定工作区",
            [],
            false,
            false,
            false,
            "react",
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(mockSubmitAgentRuntimeTurn.mock.calls[0]?.[0]).not.toHaveProperty(
        "workspace_id",
      );
    } finally {
      harness.unmount();
    }
  });

  it("已有 recent_content_id 且 metadata 显式携带时，不应重复保留 content_id metadata", async () => {
    const workspaceId = "ws-runtime-content-id-reuse";
    const topicId = "topic-runtime-content-id-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_content_id: "content-current-1",
        source: "runtime_snapshot",
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
            "继续写当前主稿",
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
                  content_id: "content-current-1",
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expectHarnessMetadataRemoved();
    } finally {
      harness.unmount();
    }
  });

  it("content_id 已变更但 session 仍是旧值时，应保留 content_id metadata", async () => {
    const workspaceId = "ws-runtime-content-id-pending-sync";
    const topicId = "topic-runtime-content-id-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_content_id: "content-old-1",
        source: "runtime_snapshot",
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
            "切到新主稿后立即发送",
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
                  content_id: "content-new-1",
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { content_id?: string };
          } | null
        )?.harness?.content_id,
      ).toBe("content-new-1");
    } finally {
      harness.unmount();
    }
  });

  it("已有 recent_theme/recent_session_mode 且 metadata 显式携带时，不应重复保留 theme/session_mode metadata", async () => {
    const workspaceId = "ws-runtime-theme-reuse";
    const topicId = "topic-runtime-theme-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_theme: "general",
        recent_session_mode: "default",
        source: "runtime_snapshot",
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
            "继续沿用当前主题会话",
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
                  theme: "general",
                  session_mode: "default",
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expectHarnessMetadataRemoved();
    } finally {
      harness.unmount();
    }
  });

  it("theme/session_mode 已变更但 session 仍是旧值时，应保留 theme/session_mode metadata", async () => {
    const workspaceId = "ws-runtime-theme-pending-sync";
    const topicId = "topic-runtime-theme-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_theme: "general",
        recent_session_mode: "default",
        source: "runtime_snapshot",
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
            "切到工作台后立即发送",
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
                  theme: "general",
                  session_mode: "general_workbench",
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { theme?: string; session_mode?: string };
          } | null
        )?.harness,
      ).toEqual({
        session_mode: "general_workbench",
      });
    } finally {
      harness.unmount();
    }
  });

  it("已有 recent_gate_key/recent_run_title 且 metadata 显式携带时，不应重复保留 gate/run metadata", async () => {
    const workspaceId = "ws-runtime-gate-reuse";
    const topicId = "topic-runtime-gate-reuse";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_gate_key: "write_mode",
        recent_run_title: "社媒初稿",
        source: "runtime_snapshot",
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
            "继续当前社媒运行",
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
                  gate_key: "write_mode",
                  run_title: "社媒初稿",
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expectHarnessMetadataRemoved();
    } finally {
      harness.unmount();
    }
  });

  it("gate_key/run_title 已变更但 session 仍是旧值时，应保留 gate/run metadata", async () => {
    const workspaceId = "ws-runtime-gate-pending-sync";
    const topicId = "topic-runtime-gate-pending-sync";
    mockGetAgentRuntimeSession.mockResolvedValue({
      id: topicId,
      created_at: Date.now(),
      updated_at: Date.now(),
      execution_strategy: "react",
      execution_runtime: {
        session_id: topicId,
        execution_strategy: "react",
        recent_gate_key: "topic_select",
        recent_run_title: "旧任务标题",
        source: "runtime_snapshot",
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
            "切到新 gate 后立即发送",
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
                  gate_key: "publish_confirm",
                  run_title: "发布确认",
                },
              },
            },
          );
      });

      expect(mockSubmitAgentRuntimeTurn).toHaveBeenCalledTimes(1);
      expect(
        (
          getSubmittedTurnMetadata() as {
            harness?: { gate_key?: string; run_title?: string };
          } | null
        )?.harness,
      ).toEqual({
        gate_key: "publish_confirm",
        run_title: "发布确认",
      });
    } finally {
      harness.unmount();
    }
  });
});
