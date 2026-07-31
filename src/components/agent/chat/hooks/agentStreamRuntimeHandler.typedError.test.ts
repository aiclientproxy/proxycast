import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentEvent, AgentThreadItem } from "@/lib/api/agentProtocol";
import type { Message } from "../types";
import { clearAgentUiProjectionEvents } from "../projection/conversationProjectionStore";
import { handleTurnStreamEvent } from "./agentStreamRuntimeHandler";
import { clearAllAgentStreamTextOverlays } from "./agentStreamTextOverlayStore";

const { mockToast } = vi.hoisted(() => ({
  mockToast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}));

vi.mock("sonner", () => ({ toast: mockToast }));

function runTypedError(willRetry: boolean) {
  let messages: Message[] = [
    {
      content: "partial answer",
      contentParts: [{ text: "partial answer", type: "text" }],
      id: "assistant-typed-error",
      isThinking: true,
      role: "assistant",
      timestamp: new Date("2026-07-31T00:00:00.000Z"),
    },
  ];
  let threadItems: AgentThreadItem[] = [
    {
      id: "pending-item",
      sequence: 0,
      started_at: "2026-07-31T00:00:00.000Z",
      status: "in_progress",
      text: "running",
      thread_id: "thread-typed-error",
      turn_id: "turn-typed-error",
      type: "turn_summary",
      updated_at: "2026-07-31T00:00:00.000Z",
    },
  ];
  const callbacks = {
    activateStream: vi.fn(),
    appendThinkingToParts: vi.fn((parts) => parts),
    clearActiveStreamIfMatch: vi.fn(() => true),
    clearOptimisticItem: vi.fn(),
    clearOptimisticTurn: vi.fn(),
    disposeListener: vi.fn(),
    isStreamActivated: vi.fn(() => true),
  };
  const observer = {
    onComplete: vi.fn(),
    onError: vi.fn(),
    onTextDelta: vi.fn(),
  };
  const setIsSending = vi.fn();
  const setMessages = vi.fn(
    (value: Message[] | ((current: Message[]) => Message[])) => {
      messages = typeof value === "function" ? value(messages) : value;
    },
  );
  const setThreadItems = vi.fn(
    (
      value:
        | AgentThreadItem[]
        | ((current: AgentThreadItem[]) => AgentThreadItem[]),
    ) => {
      threadItems = typeof value === "function" ? value(threadItems) : value;
    },
  );

  handleTurnStreamEvent({
    activeSessionId: "thread-typed-error",
    actionLoggedKeys: new Set<string>(),
    assistantMsgId: "assistant-typed-error",
    callbacks,
    content: "user prompt",
    data: {
      message: willRetry ? "stream reconnecting" : "retry budget exhausted",
      protocol_method: "error",
      session_id: "thread-typed-error",
      thread_id: "thread-typed-error",
      turn_id: "turn-typed-error",
      type: "error",
      will_retry: willRetry,
    } as AgentEvent,
    effectiveExecutionStrategy: "react",
    eventName: "agent-stream-typed-error",
    observer,
    pendingItemKey: "pending-item",
    pendingTurnKey: "pending-turn",
    requestState: {
      accumulatedContent: "partial answer",
      currentTurnId: "turn-typed-error",
      requestFinished: false,
      requestLogId: null,
      requestStartedAt: 0,
    },
    resolvedWorkspaceId: "workspace-typed-error",
    runtime: {} as never,
    setCurrentTurnId: vi.fn() as never,
    setExecutionRuntime: vi.fn() as never,
    setIsSending: setIsSending as never,
    setMessages: setMessages as never,
    setPendingActions: vi.fn() as never,
    setThreadItems: setThreadItems as never,
    setThreadTurns: vi.fn() as never,
    toolLogIdByToolId: new Map<string, string>(),
    toolNameByToolId: new Map<string, string>(),
    toolStartedAtByToolId: new Map<string, number>(),
    warnedKeysRef: { current: new Set<string>() },
  });

  return { callbacks, messages, observer, setIsSending, threadItems };
}

describe("agentStreamRuntimeHandler typed error", () => {
  afterEach(() => {
    clearAgentUiProjectionEvents();
    clearAllAgentStreamTextOverlays();
    for (const toast of Object.values(mockToast)) {
      toast.mockReset();
    }
  });

  it.each([
    [true, "retrying", "stream reconnecting"],
    [false, "failed", "retry budget exhausted"],
  ] as const)(
    "willRetry=%s only updates runtime status and waits for turn/completed",
    (willRetry, phase, detail) => {
      const result = runTypedError(willRetry);

      expect(result.callbacks.activateStream).toHaveBeenCalledTimes(1);
      expect(result.messages[0]).toMatchObject({
        content: "partial answer",
        isThinking: true,
        runtimeStatus: { detail, phase },
      });
      expect(result.threadItems[0]).toMatchObject({
        status: "in_progress",
        text: expect.stringContaining(detail),
      });
      expect(result.callbacks.clearOptimisticItem).not.toHaveBeenCalled();
      expect(result.callbacks.clearOptimisticTurn).not.toHaveBeenCalled();
      expect(result.callbacks.disposeListener).not.toHaveBeenCalled();
      expect(result.callbacks.clearActiveStreamIfMatch).not.toHaveBeenCalled();
      expect(result.setIsSending).not.toHaveBeenCalled();
      expect(result.observer.onComplete).not.toHaveBeenCalled();
      expect(result.observer.onError).not.toHaveBeenCalled();
      expect(mockToast.error).not.toHaveBeenCalled();
      expect(mockToast.warning).not.toHaveBeenCalled();
    },
  );
});
