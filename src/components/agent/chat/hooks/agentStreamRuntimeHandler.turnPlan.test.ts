import { describe, expect, it, vi } from "vitest";
import { parseAgentEvent } from "@/lib/api/agentProtocol";
import { handleTurnStreamEvent } from "./agentStreamRuntimeHandler";

describe("agentStreamRuntimeHandler turn plan signal", () => {
  it("只更新 checklist，不创建 ThreadItem 或结束当前流", () => {
    const requestState = {
      accumulatedContent: "",
      requestLogId: null,
      requestStartedAt: 0,
      requestFinished: false,
    };
    const activateStream = vi.fn();
    const setThreadItems = vi.fn();
    const setTodoItems = vi.fn();

    const data = parseAgentEvent({
      type: "turn_plan_updated",
      explanation: "继续执行",
      plan: [
        { step: "读现状", status: "completed" },
        { step: "补主链", status: "in_progress" },
      ],
    });
    expect(data).not.toBeNull();

    handleTurnStreamEvent({
      data: data!,
      requestState,
      callbacks: {
        activateStream,
        isStreamActivated: () => true,
        clearOptimisticItem: vi.fn(),
        clearOptimisticTurn: vi.fn(),
        disposeListener: vi.fn(),
        clearActiveStreamIfMatch: () => true,
        appendThinkingToParts: (parts) => parts,
      },
      eventName: "turn-plan-signal-test",
      pendingTurnKey: "pending-turn",
      pendingItemKey: "pending-item",
      assistantMsgId: "assistant-1",
      activeSessionId: "session-1",
      resolvedWorkspaceId: "workspace-1",
      effectiveExecutionStrategy: "react",
      content: "",
      runtime: {} as never,
      warnedKeysRef: { current: new Set<string>() },
      actionLoggedKeys: new Set<string>(),
      toolLogIdByToolId: new Map<string, string>(),
      toolStartedAtByToolId: new Map<string, number>(),
      toolNameByToolId: new Map<string, string>(),
      setMessages: vi.fn() as never,
      setPendingActions: vi.fn() as never,
      setThreadItems: setThreadItems as never,
      setTodoItems: setTodoItems as never,
      setThreadTurns: vi.fn() as never,
      setCurrentTurnId: vi.fn() as never,
      setExecutionRuntime: vi.fn() as never,
      setIsSending: vi.fn() as never,
    });

    expect(activateStream).toHaveBeenCalledOnce();
    expect(setTodoItems).toHaveBeenCalledWith([
      { content: "读现状", status: "completed" },
      { content: "补主链", status: "in_progress" },
    ]);
    expect(setThreadItems).not.toHaveBeenCalled();
    expect(requestState.requestFinished).toBe(false);
  });
});
