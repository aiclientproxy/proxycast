import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import type { AgentThreadItem, AgentThreadTurn } from "@/lib/api/agentProtocol";
import type { AgentTodoItem } from "@/lib/api/agentRuntime/sessionTypes";
import type {
  AgentExecutionStrategy,
  AgentSessionExecutionRuntime,
} from "@/lib/api/agentExecutionRuntime";
import type {
  SessionModelPreference,
  WorkspacePathMissingState,
} from "./agentChatShared";
import type { AgentRuntimeAdapter } from "./agentRuntimeAdapter";
import type { AgentAccessMode } from "./agentChatStorage";
import type { ActiveStreamState } from "./agentStreamSubmissionLifecycle";
import type { ActionRequired, Message } from "../types";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";
import type { SoulInteractionCopy } from "@/lib/soul/interactionCopy";

export type AppendThinkingToPartsFn = (
  parts: NonNullable<Message["contentParts"]>,
  textDelta: string,
) => NonNullable<Message["contentParts"]>;

export interface AgentStreamPreparedSendEnv {
  runtime: AgentRuntimeAdapter;
  ensureSession: (options?: {
    targetSessionId?: string;
    skipSessionRestore?: boolean;
    skipSessionStartHooks?: boolean;
  }) => Promise<string | null>;
  attemptSilentTurnRecovery: (
    sessionId: string,
    requestStartedAt: number,
    promptText: string,
    options?: { requireTerminal?: boolean; turnId?: string | null },
  ) => Promise<boolean>;
  refreshSessionReadModel: (targetSessionId?: string) => Promise<boolean>;
  executionStrategy: AgentExecutionStrategy;
  accessMode: AgentAccessMode;
  providerTypeRef: MutableRefObject<string>;
  modelRef: MutableRefObject<string>;
  reasoningEffortRef: MutableRefObject<string>;
  sessionIdRef: MutableRefObject<string | null>;
  runPreparedSubmit: <T>(task: () => Promise<T>) => Promise<T>;
  getWorkspaceIdForSubmit: () => string | undefined;
  waitForSessionProviderSelectionSync?: (
    sessionId: string,
  ) => Promise<void>;
  getThreadIdForSubmit: (targetSessionId?: string) => string | undefined;
  getSyncedSessionModelPreference: (
    sessionId: string,
  ) => SessionModelPreference | null;
  getSyncedSessionExecutionStrategy: (
    sessionId: string,
  ) => AgentExecutionStrategy | null;
  getSyncedSessionRecentPreferences?: (
    sessionId: string,
  ) => ChatToolPreferences | null;
  listenerMapRef: MutableRefObject<Map<string, () => void>>;
  activeStreamRef: MutableRefObject<ActiveStreamState | null>;
  warnedKeysRef: MutableRefObject<Set<string>>;
  onWriteFile?: (
    content: string,
    fileName: string,
    context?: import("../types").WriteArtifactContext,
  ) => void;
  executionRuntime?: AgentSessionExecutionRuntime | null;
  clawTraceEnabled: boolean;
  soulCopy?: SoulInteractionCopy;
  setActiveStream: (nextActive: ActiveStreamState | null) => void;
  clearActiveStreamIfMatch: (eventName: string) => boolean;
  setMessages: Dispatch<SetStateAction<Message[]>>;
  getThreadItems?: () => readonly AgentThreadItem[];
  setThreadItems: Dispatch<SetStateAction<AgentThreadItem[]>>;
  setTodoItems?: Dispatch<SetStateAction<AgentTodoItem[]>>;
  setThreadTurns: Dispatch<SetStateAction<AgentThreadTurn[]>>;
  setCurrentTurnId: Dispatch<SetStateAction<string | null>>;
  setExecutionRuntime: Dispatch<
    SetStateAction<AgentSessionExecutionRuntime | null>
  >;
  setPendingActions: Dispatch<SetStateAction<ActionRequired[]>>;
  setWorkspacePathMissing: Dispatch<
    SetStateAction<WorkspacePathMissingState | null>
  >;
  setIsSending: Dispatch<SetStateAction<boolean>>;
  appendThinkingToParts: AppendThinkingToPartsFn;
}

export type CreateAgentStreamPreparedSendEnvOptions =
  AgentStreamPreparedSendEnv;

export function createAgentStreamPreparedSendEnv(
  options: CreateAgentStreamPreparedSendEnvOptions,
): AgentStreamPreparedSendEnv {
  return options;
}
