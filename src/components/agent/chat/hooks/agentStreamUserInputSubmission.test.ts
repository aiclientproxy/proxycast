import { afterEach, describe, expect, it, vi } from "vitest";
import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import { activityLogger } from "@/lib/workspace/workbenchRuntime";
import type { AgentThreadItem, AgentThreadTurn } from "@/lib/api/agentProtocol";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type { ActionRequired, Message } from "../types";
import type { WorkspacePathMissingState } from "./agentChatShared";
import type { ActiveStreamState } from "./agentStreamSubmissionLifecycle";
import type { AgentStreamPreparedSendEnv } from "./agentStreamPreparedSendEnv";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import { submitAgentStreamUserInput } from "./agentStreamUserInputSubmission";
import { buildWaitingAgentRuntimeStatus } from "../utils/agentRuntimeStatus";

const { resolveAllowedPermissionProfileMock } = vi.hoisted(() => ({
  resolveAllowedPermissionProfileMock: vi.fn(async (id: string) => ({
    id,
    allowed: true,
  })),
}));

vi.mock("@/lib/api/permissionProfiles", () => ({
  resolveAllowedPermissionProfile: resolveAllowedPermissionProfileMock,
}));

function createStateSetter<T>(getValue: () => T, setValue: (value: T) => void) {
  return (next: T | ((prev: T) => T)) => {
    setValue(
      typeof next === "function" ? (next as (prev: T) => T)(getValue()) : next,
    );
  };
}

function noopDispatch<T>() {
  return vi.fn() as unknown as Dispatch<SetStateAction<T>>;
}

describe("agentStreamUserInputSubmission", () => {
  afterEach(() => {
    activityLogger.clear();
  });

  it("应串起 lifecycle 与 execute helper，完成一次 user_input 提交", async () => {
    const assistantMsg: Message = {
      id: "assistant-1",
      role: "assistant",
      content: "",
      timestamp: new Date("2026-03-27T01:00:00.000Z"),
      isThinking: true,
      contentParts: [],
      runtimeStatus: buildWaitingAgentRuntimeStatus({
        executionStrategy: "react",
      }),
    };

    const activeStreamRefState: { current: ActiveStreamState | null } = {
      current: null,
    };
    let messages: Message[] = [assistantMsg];
    let threadItems: AgentThreadItem[] = [];
    let threadTurns: AgentThreadTurn[] = [];
    let currentTurnId: string | null = null;
    const runtime = {
      listenToTurnEvents: vi.fn(async () => vi.fn()),
      getThreadTurnControl: vi.fn(async () => ({
        threadId: "thread-1",
        updatedAtMs: Date.now(),
        activeTurnId: null,
        queuedTurnIds: [],
      })),
      submitOp: vi.fn(async () => {}),
    } as unknown as AgentRuntimeAdapter;
    const env: AgentStreamPreparedSendEnv = {
      runtime,
      ensureSession: async () => "session-1",
      attemptSilentTurnRecovery: async () => false,
      refreshSessionReadModel: async () => true,
      executionStrategy: "react",
      accessMode: "current",
      clawTraceEnabled: false,
      providerTypeRef: { current: "openai" } as MutableRefObject<string>,
      modelRef: { current: "gpt-5.4" } as MutableRefObject<string>,
      reasoningEffortRef: { current: "" } as MutableRefObject<string>,
      sessionIdRef: { current: null } as MutableRefObject<string | null>,
      runPreparedSubmit: async (task) => task(),
      getWorkspaceIdForSubmit: () => "workspace-1",
      waitForSessionProviderSelectionSync: async () => undefined,
      getThreadIdForSubmit: () => "thread-1",
      getSyncedSessionModelPreference: () => null,
      getSyncedSessionExecutionStrategy: () => "react",
      warnedKeysRef: { current: new Set<string>() },
      listenerMapRef: { current: new Map() },
      activeStreamRef: {
        current: null,
      } as MutableRefObject<ActiveStreamState | null>,
      setActiveStream: (next) => {
        activeStreamRefState.current = next;
      },
      clearActiveStreamIfMatch: () => false,
      setMessages: createStateSetter(
        () => messages,
        (value) => {
          messages = value;
        },
      ),
      setThreadItems: createStateSetter(
        () => threadItems,
        (value) => {
          threadItems = value;
        },
      ),
      setThreadTurns: createStateSetter(
        () => threadTurns,
        (value) => {
          threadTurns = value;
        },
      ),
      setCurrentTurnId: createStateSetter(
        () => currentTurnId,
        (value) => {
          currentTurnId = value;
        },
      ),
      setExecutionRuntime: noopDispatch<AgentSessionExecutionRuntime | null>(),
      setPendingActions: noopDispatch<ActionRequired[]>(),
      setWorkspacePathMissing: noopDispatch<WorkspacePathMissingState | null>(),
      setIsSending: noopDispatch<boolean>(),
      appendThinkingToParts: (
        parts: NonNullable<Message["contentParts"]>,
        _textDelta: string,
      ) => parts,
    };

    await submitAgentStreamUserInput({
      preparedSend: {
        content: "继续生成提纲",
        images: [],
        skipUserMessage: false,
        effectiveExecutionStrategy: "react",
        effectiveProviderType: "openai",
        effectiveModel: "gpt-5.4",
        syncedSessionModelPreference: null,
        assistantMsgId: "assistant-1",
        userMsgId: "user-1",
        userMsg: {
          id: "user-1",
          role: "user",
          content: "继续生成提纲",
          timestamp: new Date("2026-03-27T01:00:00.000Z"),
        },
        assistantMsg,
      },
      env,
    });

    expect(runtime.submitOp).toHaveBeenCalledTimes(1);
    expect(runtime.submitOp).toHaveBeenCalledWith(
      expect.objectContaining({
        turn: expect.objectContaining({
          threadId: "thread-1",
          input: [{ type: "text", text: "继续生成提纲" }],
          approvalPolicy: "on-request",
          permissions: ":workspace",
        }),
      }),
    );
    expect(activeStreamRefState.current?.sessionId).toBe("session-1");
    expect(threadTurns).toHaveLength(1);
    expect(threadItems).toHaveLength(1);
    expect(currentTurnId).toBeTruthy();
    expect(activityLogger.getLogs()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          eventType: "chat_request_start",
          status: "pending",
        }),
      ]),
    );
  });
});
