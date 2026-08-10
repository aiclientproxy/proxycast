import { describe, expect, it } from "vitest";
import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import type { AgentThreadItem, AgentThreadTurn } from "@/lib/api/agentProtocol";
import type { AgentSessionExecutionRuntime } from "@/lib/api/agentExecutionRuntime";
import type { ActiveStreamState } from "./agentStreamSubmissionLifecycle";
import type { WorkspacePathMissingState } from "./agentChatShared";
import type { ActionRequired, Message } from "../types";
import { createAgentStreamPreparedSendEnv } from "./agentStreamPreparedSendEnv";

function noopDispatch<T>() {
  return (() => undefined) as unknown as Dispatch<SetStateAction<T>>;
}

describe("createAgentStreamPreparedSendEnv", () => {
  it("应保留 current prepared send env 绑定", () => {
    const options = {
      runtime: {} as never,
      ensureSession: async () => "session-1",
      attemptSilentTurnRecovery: async () => false,
      refreshSessionReadModel: async () => true,
      executionStrategy: "react",
      accessMode: "current",
      providerTypeRef: { current: "openai" } as MutableRefObject<string>,
      modelRef: { current: "gpt-5.4" } as MutableRefObject<string>,
      reasoningEffortRef: { current: "" } as MutableRefObject<string>,
      sessionIdRef: { current: "session-1" } as MutableRefObject<string | null>,
      getThreadIdForSubmit: () => "thread-1",
      runPreparedSubmit: async (task) => task(),
      getWorkspaceIdForSubmit: () => "workspace-1",
      waitForSessionProviderSelectionSync: async () => undefined,
      getSyncedSessionModelPreference: () => null,
      getSyncedSessionExecutionStrategy: () => "react",
      listenerMapRef: { current: new Map() },
      clawTraceEnabled: false,
      activeStreamRef: {
        current: null,
      } as MutableRefObject<ActiveStreamState | null>,
      warnedKeysRef: { current: new Set<string>() },
      setActiveStream: () => undefined,
      clearActiveStreamIfMatch: () => false,
      setMessages: noopDispatch<Message[]>(),
      setThreadItems: noopDispatch<AgentThreadItem[]>(),
      setThreadTurns: noopDispatch<AgentThreadTurn[]>(),
      setCurrentTurnId: noopDispatch<string | null>(),
      setExecutionRuntime: noopDispatch<AgentSessionExecutionRuntime | null>(),
      setPendingActions: noopDispatch<ActionRequired[]>(),
      setWorkspacePathMissing: noopDispatch<WorkspacePathMissingState | null>(),
      setIsSending: noopDispatch<boolean>(),
      appendThinkingToParts: (parts) => parts,
    } satisfies Parameters<typeof createAgentStreamPreparedSendEnv>[0];
    const env = createAgentStreamPreparedSendEnv(options);

    expect(env).toBe(options);
    expect(env.getThreadIdForSubmit()).toBe("thread-1");
  });
});
