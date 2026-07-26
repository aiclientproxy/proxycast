import { describe, expect, it, vi } from "vitest";
import type { MutableRefObject } from "react";
import type { AgentSessionDetail } from "@/lib/api/agentRuntime/sessionTypes";
import {
  createAgentSessionReadModelSnapshot,
  hydrateFreshAgentSessionReadModel,
  refreshAgentSessionDetailState,
  refreshAgentSessionReadModelState,
  resolveDefaultAgentSessionDetailMergeMode,
} from "./agentSessionRefresh";

describe("agentSessionRefresh", () => {
  it("默认 detail merge mode 应只用于历史 hydrate 兜底", () => {
    expect(resolveDefaultAgentSessionDetailMergeMode()).toBe("history_hydrate");
  });

  it("应把 canonical thread_read 作为唯一刷新快照", () => {
    const snapshot = createAgentSessionReadModelSnapshot({
      thread_id: "thread-1",
      status: "queued",
      pending_requests: [],
      incidents: [],
    } as never);

    expect(snapshot.threadRead).toMatchObject({
      thread_id: "thread-1",
      status: "queued",
    });
  });

  it("新会话应在 submit 前 hydrate canonical threadId", async () => {
    const getSessionReadModel = vi.fn(async () => ({
      thread_id: " thread-1 ",
      status: "idle" as const,
      pending_requests: [],
      incidents: [],
    }));

    await expect(
      hydrateFreshAgentSessionReadModel({ getSessionReadModel }, " session-1 "),
    ).resolves.toMatchObject({ thread_id: "thread-1", status: "idle" });
    expect(getSessionReadModel).toHaveBeenCalledWith("session-1");
  });

  it("新会话缺少 canonical threadId 时应 fail closed", async () => {
    await expect(
      hydrateFreshAgentSessionReadModel(
        {
          getSessionReadModel: vi.fn(async () => ({
            thread_id: "",
          })),
        },
        "session-1",
      ),
    ).rejects.toThrow(
      "fresh session read model did not include a canonical threadId",
    );
  });

  it("刷新 detail 时应应用 detail 并把 legacy executionStrategy 归一后同步", async () => {
    const applySessionDetail = vi.fn();
    const markSynced = vi.fn();
    const detail: AgentSessionDetail = {
      id: "session-1",
      messages: [],
      created_at: 1,
      updated_at: 2,
      execution_strategy: "code_orchestrated" as never,
    };
    const getSession = vi.fn(async () => detail);

    await expect(
      refreshAgentSessionDetailState({
        runtime: {
          getSession,
        },
        sessionIdRef: {
          current: "session-1",
        } as MutableRefObject<string | null>,
        applySessionDetail,
        markSessionExecutionStrategySynced: markSynced,
        source: "runtimeSync.event",
        detailMergeMode: "terminal_reconcile",
      }),
    ).resolves.toBe(true);

    expect(getSession).toHaveBeenCalledWith("session-1", {
      historyLimit: 40,
      source: "runtimeSync.event",
    });
    expect(applySessionDetail).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({
        id: "session-1",
        execution_strategy: "code_orchestrated",
      }),
      {
        preserveExecutionStrategyOnMissingDetail: true,
        detailMergeMode: "terminal_reconcile",
      },
    );
    expect(markSynced).toHaveBeenCalledWith("session-1", "react");
  });

  it("刷新 detail 时应把 recent_access_mode 同步到当前 accessMode 与 session shadow", async () => {
    const applySessionDetail = vi.fn();
    const markSynced = vi.fn();
    const persistSessionAccessMode = vi.fn();
    const setAccessModeState = vi.fn();
    const detail: AgentSessionDetail = {
      id: "session-1",
      messages: [],
      created_at: 1,
      updated_at: 2,
      execution_strategy: "react",
      execution_runtime: {
        session_id: "session-1",
        execution_strategy: "react",
        recent_access_mode: "current",
        source: "session",
      },
    };
    const getSession = vi.fn(async () => detail);

    await expect(
      refreshAgentSessionDetailState({
        runtime: {
          getSession,
        },
        sessionIdRef: {
          current: "session-1",
        } as MutableRefObject<string | null>,
        applySessionDetail,
        markSessionExecutionStrategySynced: markSynced,
        persistSessionAccessMode,
        setAccessModeState,
      }),
    ).resolves.toBe(true);

    expect(getSession).toHaveBeenCalledWith("session-1", {
      historyLimit: 40,
    });
    expect(persistSessionAccessMode).toHaveBeenCalledWith(
      "session-1",
      "current",
    );
    expect(setAccessModeState).toHaveBeenCalledWith("current");
  });

  it("刷新 read model 时应在会话仍匹配时应用 snapshot", async () => {
    const applyReadModelSnapshot = vi.fn();

    await expect(
      refreshAgentSessionReadModelState({
        runtime: {
          getSessionReadModel: vi.fn(async () => ({
            thread_id: "thread-1",
            status: "idle",
            pending_requests: [],
            incidents: [],
          })),
        },
        sessionIdRef: {
          current: "session-1",
        } as MutableRefObject<string | null>,
        applyReadModelSnapshot,
      }),
    ).resolves.toBe(true);

    expect(applyReadModelSnapshot).toHaveBeenCalledWith({
      threadRead: expect.objectContaining({
        thread_id: "thread-1",
        status: "idle",
      }),
    });
  });
});
