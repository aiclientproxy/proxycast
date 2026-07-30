import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, vi } from "vitest";
import type { TurnStartParams } from "@limecloud/app-server-client";
import { getDefaultPendingInteractionController } from "@/lib/api/agentRuntime/pendingInteractionController";
import type { WriteArtifactContext } from "../types";
import type { ChatToolPreferences } from "../utils/chatToolPreferences";

const {
  mockGetRuntimeProviderSelection,
  mockSubmitAgentRuntimeTurn,
  mockCreateAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockGetAgentRuntimeSession,
  mockGetAgentRuntimeThreadRead,
  mockReadAgentRuntimeThread,
  mockGenerateAgentRuntimeSessionTitle,
  mockSetAgentRuntimeThreadName,
  mockUpdateAgentRuntimeThreadSettings,
  mockDeleteAgentRuntimeSession,
  mockCompactAgentRuntimeSession,
  mockInterruptAgentRuntimeTurn,
  mockResumeAgentRuntimeThread,
  mockReplayAgentRuntimeRequest,
  mockRespondAgentRuntimeAction,
  mockParseAgentEvent,
  mockSafeListen,
  mockToast,
  mockWechatChannelSetRuntimeModel,
  mockResolveClawWorkspaceProviderSelection,
  mockScheduleMinimumDelayIdleTask,
  mockAgentSessionDetailByThreadId,
  resolveMockAgentRuntimeSession,
} = vi.hoisted(() => {
  const mockGetAgentRuntimeSession = vi.fn();
  const mockAgentSessionDetailByThreadId = new Map<
    string,
    { thread_read?: unknown }
  >();
  const resolveMockAgentRuntimeSession = async (
    sessionId: string,
    ...args: unknown[]
  ) => {
    const detail = await mockGetAgentRuntimeSession(sessionId, ...args);
    const threadId =
      detail.thread_id?.trim() ||
      detail.thread_read?.thread_id?.trim() ||
      sessionId;
    const normalizedDetail = { ...detail, thread_id: threadId };
    mockAgentSessionDetailByThreadId.set(threadId, normalizedDetail);
    return normalizedDetail;
  };

  return {
    mockGetRuntimeProviderSelection: vi.fn(),
    mockSubmitAgentRuntimeTurn: vi.fn(),
    mockCreateAgentRuntimeSession: vi.fn(),
    mockListAgentRuntimeSessions: vi.fn(),
    mockGetAgentRuntimeSession,
    mockGetAgentRuntimeThreadRead: vi.fn(),
    mockReadAgentRuntimeThread: vi.fn(),
    mockGenerateAgentRuntimeSessionTitle: vi.fn(),
    mockSetAgentRuntimeThreadName: vi.fn(),
    mockUpdateAgentRuntimeThreadSettings: vi.fn(),
    mockDeleteAgentRuntimeSession: vi.fn(),
    mockCompactAgentRuntimeSession: vi.fn(),
    mockInterruptAgentRuntimeTurn: vi.fn(),
    mockResumeAgentRuntimeThread: vi.fn(),
    mockReplayAgentRuntimeRequest: vi.fn(),
    mockRespondAgentRuntimeAction: vi.fn(),
    mockParseAgentEvent: vi.fn((payload: unknown) => payload),
    mockSafeListen: vi.fn(),
    mockToast: {
      success: vi.fn(),
      error: vi.fn(),
      info: vi.fn(),
      warning: vi.fn(),
    },
    mockWechatChannelSetRuntimeModel: vi.fn(async () => undefined),
    mockResolveClawWorkspaceProviderSelection: vi.fn(),
    mockScheduleMinimumDelayIdleTask: vi.fn((task: () => void) => {
      task();
      return () => undefined;
    }),
    mockAgentSessionDetailByThreadId,
    resolveMockAgentRuntimeSession,
  };
});

export {
  mockGetRuntimeProviderSelection,
  mockSubmitAgentRuntimeTurn,
  mockCreateAgentRuntimeSession,
  mockListAgentRuntimeSessions,
  mockGetAgentRuntimeSession,
  mockGetAgentRuntimeThreadRead,
  mockReadAgentRuntimeThread,
  mockGenerateAgentRuntimeSessionTitle,
  mockSetAgentRuntimeThreadName,
  mockUpdateAgentRuntimeThreadSettings,
  mockDeleteAgentRuntimeSession,
  mockCompactAgentRuntimeSession,
  mockInterruptAgentRuntimeTurn,
  mockResumeAgentRuntimeThread,
  mockReplayAgentRuntimeRequest,
  mockRespondAgentRuntimeAction,
  mockParseAgentEvent,
  mockSafeListen,
  mockToast,
  mockWechatChannelSetRuntimeModel,
  mockResolveClawWorkspaceProviderSelection,
  mockScheduleMinimumDelayIdleTask,
};

vi.mock("@/lib/api/agentRuntime/clientFactory", () => ({
  createAgentRuntimeClient: () => ({
    getRuntimeProviderSelection: mockGetRuntimeProviderSelection,
    submitAgentRuntimeTurn: mockSubmitAgentRuntimeTurn,
    createAgentRuntimeSession: mockCreateAgentRuntimeSession,
    listAgentRuntimeSessions: mockListAgentRuntimeSessions,
    getAgentRuntimeSession: resolveMockAgentRuntimeSession,
    getAgentRuntimeThreadRead: mockGetAgentRuntimeThreadRead,
    readAgentRuntimeThread: mockReadAgentRuntimeThread,
    generateAgentRuntimeSessionTitle: mockGenerateAgentRuntimeSessionTitle,
    setAgentRuntimeThreadName: mockSetAgentRuntimeThreadName,
    updateAgentRuntimeThreadSettings: mockUpdateAgentRuntimeThreadSettings,
    deleteAgentRuntimeSession: mockDeleteAgentRuntimeSession,
    compactAgentRuntimeSession: mockCompactAgentRuntimeSession,
    interruptAgentRuntimeTurn: mockInterruptAgentRuntimeTurn,
    resumeThread: mockResumeAgentRuntimeThread,
    replayAgentRuntimeRequest: mockReplayAgentRuntimeRequest,
    respondAgentRuntimeAction: mockRespondAgentRuntimeAction,
  }),
  getRuntimeProviderSelection: mockGetRuntimeProviderSelection,
  submitAgentRuntimeTurn: mockSubmitAgentRuntimeTurn,
  createAgentRuntimeSession: mockCreateAgentRuntimeSession,
  listAgentRuntimeSessions: mockListAgentRuntimeSessions,
  getAgentRuntimeSession: resolveMockAgentRuntimeSession,
  getAgentRuntimeThreadRead: mockGetAgentRuntimeThreadRead,
  generateAgentRuntimeSessionTitle: mockGenerateAgentRuntimeSessionTitle,
  setAgentRuntimeThreadName: mockSetAgentRuntimeThreadName,
  updateAgentRuntimeThreadSettings: mockUpdateAgentRuntimeThreadSettings,
  deleteAgentRuntimeSession: mockDeleteAgentRuntimeSession,
  compactAgentRuntimeSession: mockCompactAgentRuntimeSession,
  interruptAgentRuntimeTurn: mockInterruptAgentRuntimeTurn,
  resumeThread: mockResumeAgentRuntimeThread,
  replayAgentRuntimeRequest: mockReplayAgentRuntimeRequest,
  respondAgentRuntimeAction: mockRespondAgentRuntimeAction,
}));

vi.mock("@/lib/api/agentProtocol", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/api/agentProtocol")
  >("@/lib/api/agentProtocol");
  return {
    ...actual,
    parseAgentEvent: mockParseAgentEvent,
  };
});

vi.mock("@/lib/dev-bridge", () => ({
  safeListen: mockSafeListen,
  hasDevBridgeEventListenerCapability: vi.fn(() => false),
  isDevBridgeAvailable: vi.fn(() => false),
}));

vi.mock("sonner", () => ({
  toast: mockToast,
}));

vi.mock("@/lib/api/channelsRuntime", () => ({
  wechatChannelSetRuntimeModel: mockWechatChannelSetRuntimeModel,
}));

vi.mock("@/lib/desktop-runtime", () => ({
  getLegacyDesktopHostGlobal: vi.fn(() => null),
  hasDesktopHostEventCapability: vi.fn(() => true),
  hasDesktopHostEventListenerCapability: vi.fn(() => true),
  hasDesktopHostInvokeCapability: vi.fn(() => true),
  hasDesktopHostRuntimeMarkers: vi.fn(() => true),
}));

vi.mock("@/lib/utils/scheduleMinimumDelayIdleTask", () => ({
  scheduleMinimumDelayIdleTask: mockScheduleMinimumDelayIdleTask,
}));

vi.mock("../utils/clawWorkspaceProviderSelection", () => ({
  resolveClawWorkspaceProviderSelection:
    mockResolveClawWorkspaceProviderSelection,
}));

import { useAgentChat } from "./useAgentChat";
import { publishAgentRuntimeEvent } from "@/lib/api/agentRuntimeEvents";
import { changeLimeLocale } from "@/i18n/createI18n";
import {
  clearAllAgentStreamTextOverlays,
  getAgentStreamTextOverlay as readAgentStreamTextOverlay,
} from "./agentStreamTextOverlayStore";

export interface HookHarness {
  getValue: () => ReturnType<typeof useAgentChat>;
  getRenderCount: () => number;
  unmount: () => void;
}

export function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

export function getSubmittedTurnStart(
  callIndex = 0,
): TurnStartParams | undefined {
  return mockSubmitAgentRuntimeTurn.mock.calls[callIndex]?.[0] as
    | TurnStartParams
    | undefined;
}

export function getSubmittedTurnMetadata(
  callIndex = 0,
): Record<string, unknown> | undefined {
  const entry = getSubmittedTurnStart(callIndex)?.additionalContext?.metadata;
  if (entry?.kind !== "application" || typeof entry.value !== "string") {
    return undefined;
  }
  const parsed = JSON.parse(entry.value) as unknown;
  return parsed && typeof parsed === "object" && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>)
    : undefined;
}

export function mockTypedActionResponseHandled(
  actionType: "ask_user" | "tool_confirmation",
  requestId: string,
) {
  const controller = getDefaultPendingInteractionController();
  const kind = actionType === "ask_user" ? "request_user_input" : "approval";
  const interactionId = `${kind}:test:${encodeURIComponent(requestId)}`;
  const getSnapshot = vi.spyOn(controller, "getSnapshot").mockReturnValue([
    {
      id: interactionId,
      thread_id: "test-thread",
      kind,
      status: "pending",
      payload: {
        request: { requestId, actionType, status: "pending" },
      },
    },
  ]);
  const respond = vi
    .spyOn(controller, "respond")
    .mockReturnValue({ accepted: true });
  const restoreRespond = respond.mockRestore.bind(respond);
  respond.mockRestore = () => {
    restoreRespond();
    getSnapshot.mockRestore();
  };
  return respond;
}

export function mountHook(
  workspaceId = "ws-test",
  currentOptions: {
    onWriteFile?: (
      content: string,
      fileName: string,
      context?: WriteArtifactContext,
    ) => void;
    getSyncedSessionRecentPreferences?: (
      sessionId: string,
    ) => ChatToolPreferences | null;
    initialTopicsLoadMode?: "immediate" | "deferred";
    initialTopicsDeferredDelayMs?: number;
    initialRuntimeWarmupLoadMode?: "immediate" | "deferred";
    initialRuntimeWarmupDeferredDelayMs?: number;
    sessionRestorePresentation?: "foreground" | "background";
  } = {},
): HookHarness {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  let hookValue: ReturnType<typeof useAgentChat> | null = null;
  let renderCount = 0;

  function TestComponent() {
    renderCount += 1;
    hookValue = useAgentChat({
      workspaceId,
      onWriteFile: currentOptions.onWriteFile,
      getSyncedSessionRecentPreferences:
        currentOptions.getSyncedSessionRecentPreferences,
      initialTopicsLoadMode: currentOptions.initialTopicsLoadMode,
      initialTopicsDeferredDelayMs: currentOptions.initialTopicsDeferredDelayMs,
      initialRuntimeWarmupLoadMode: currentOptions.initialRuntimeWarmupLoadMode,
      initialRuntimeWarmupDeferredDelayMs:
        currentOptions.initialRuntimeWarmupDeferredDelayMs,
      sessionRestorePresentation: currentOptions.sessionRestorePresentation,
    });
    return null;
  }

  act(() => {
    root.render(<TestComponent />);
  });

  return {
    getValue: () => {
      if (!hookValue) {
        throw new Error("hook 尚未初始化");
      }
      return hookValue;
    },
    getRenderCount: () => renderCount,
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

export async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
  });
}

export async function flushRuntimeDetailRefresh() {
  await act(async () => {
    await new Promise((resolve) => window.setTimeout(resolve, 140));
  });
  await flushEffects();
}

export function captureTurnStream() {
  return captureRuntimeStream((eventName) => {
    return (
      typeof eventName === "string" && eventName.startsWith("agent_stream_")
    );
  });
}

export function captureContextCompactionStream() {
  return captureRuntimeStream((eventName) => {
    return (
      typeof eventName === "string" &&
      eventName.startsWith("agent_context_compaction_")
    );
  });
}

function captureRuntimeStream(matcher: (eventName: unknown) => boolean) {
  let streamHandler: ((event: { payload: unknown }) => void) | null = null;
  let activeEventName: string | null = null;

  mockSafeListen.mockImplementation(async (eventName, handler) => {
    if (matcher(eventName)) {
      streamHandler = handler as (event: { payload: unknown }) => void;
      activeEventName =
        typeof eventName === "string" ? eventName : String(eventName);
      return () => {
        if (streamHandler === handler) {
          streamHandler = null;
        }
        if (activeEventName === eventName) {
          activeEventName = null;
        }
      };
    }
    return () => {};
  });

  return {
    emit(payload: unknown) {
      const eventName = resolveCapturedRuntimeEventName({
        activeEventName,
        matcher,
      });
      if (eventName) {
        publishAgentRuntimeEvent(eventName, payload);
        return;
      }
      streamHandler?.({ payload });
    },
    emitBridge(payload: unknown) {
      streamHandler?.({ payload });
    },
    getEventName() {
      return resolveCapturedRuntimeEventName({
        activeEventName,
        matcher,
      });
    },
  };
}

function resolveCapturedRuntimeEventName({
  activeEventName,
  matcher,
}: {
  activeEventName: string | null;
  matcher: (eventName: unknown) => boolean;
}) {
  if (activeEventName && matcher(activeEventName)) {
    return activeEventName;
  }

  const runtimeCalls = [
    ...collectRuntimeEventNameCalls(mockSubmitAgentRuntimeTurn),
    ...collectRuntimeEventNameCalls(mockCompactAgentRuntimeSession),
  ].sort((left, right) => right.order - left.order);

  for (const call of runtimeCalls) {
    const eventName = readRuntimeEventNameFromCall(call.args);
    if (eventName && matcher(eventName)) {
      return eventName;
    }
  }

  return null;
}

function collectRuntimeEventNameCalls(mock: typeof mockSubmitAgentRuntimeTurn) {
  return mock.mock.calls.map((args, index) => ({
    args,
    order: mock.mock.invocationCallOrder[index] ?? index,
  }));
}

function readRuntimeEventNameFromCall(args: unknown[]) {
  const request = args[0];
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    return null;
  }
  const runtimeOptions = (request as { runtimeOptions?: unknown })
    .runtimeOptions;
  if (
    !runtimeOptions ||
    typeof runtimeOptions !== "object" ||
    Array.isArray(runtimeOptions)
  ) {
    return null;
  }
  const eventName = (runtimeOptions as { eventName?: unknown }).eventName;
  return typeof eventName === "string" && eventName.trim() ? eventName : null;
}

export function seedSession(workspaceId: string, sessionId: string) {
  localStorage.setItem(
    `agent_session_workspace_${sessionId}`,
    JSON.stringify(workspaceId),
  );
  sessionStorage.setItem(
    `agent_curr_sessionId_${workspaceId}`,
    JSON.stringify(sessionId),
  );
  sessionStorage.setItem(
    `agent_messages_${workspaceId}`,
    JSON.stringify([
      {
        id: "m-1",
        role: "assistant",
        content: "hello",
        timestamp: new Date().toISOString(),
      },
    ]),
  );
}

export function seedSessionSnapshots(
  workspaceId: string,
  snapshots: Record<string, unknown>,
) {
  sessionStorage.setItem(
    `agent_session_snapshots_${workspaceId}`,
    JSON.stringify(snapshots),
  );
}

export function completedTurn(id = "turn-completed") {
  return {
    id,
    thread_id: "thread-completed",
    prompt_text: "",
    status: "completed" as const,
    started_at: "2026-06-12T00:00:00.000Z",
    completed_at: "2026-06-12T00:00:01.000Z",
    created_at: "2026-06-12T00:00:00.000Z",
    updated_at: "2026-06-12T00:00:01.000Z",
  };
}

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");

  mockGetRuntimeProviderSelection.mockReset();
  mockSubmitAgentRuntimeTurn.mockReset();
  mockCreateAgentRuntimeSession.mockReset();
  mockListAgentRuntimeSessions.mockReset();
  mockGetAgentRuntimeSession.mockReset();
  mockGetAgentRuntimeThreadRead.mockReset();
  mockReadAgentRuntimeThread.mockReset();
  mockGenerateAgentRuntimeSessionTitle.mockReset();
  mockSetAgentRuntimeThreadName.mockReset();
  mockUpdateAgentRuntimeThreadSettings.mockReset();
  mockDeleteAgentRuntimeSession.mockReset();
  mockCompactAgentRuntimeSession.mockReset();
  mockInterruptAgentRuntimeTurn.mockReset();
  mockResumeAgentRuntimeThread.mockReset();
  mockReplayAgentRuntimeRequest.mockReset();
  mockRespondAgentRuntimeAction.mockReset();
  mockParseAgentEvent.mockReset();
  mockSafeListen.mockReset();
  mockWechatChannelSetRuntimeModel.mockReset();
  mockResolveClawWorkspaceProviderSelection.mockReset();
  mockScheduleMinimumDelayIdleTask.mockReset();
  mockScheduleMinimumDelayIdleTask.mockImplementation((task: () => void) => {
    task();
    return () => undefined;
  });
  mockToast.success.mockReset();
  mockToast.error.mockReset();
  mockToast.info.mockReset();
  mockToast.warning.mockReset();
  localStorage.clear();
  sessionStorage.clear();
  mockAgentSessionDetailByThreadId.clear();

  mockGetRuntimeProviderSelection.mockResolvedValue({
    provider_configured: false,
  });
  mockSubmitAgentRuntimeTurn.mockResolvedValue(undefined);
  mockCreateAgentRuntimeSession.mockResolvedValue("created-session");
  mockListAgentRuntimeSessions.mockResolvedValue([]);
  mockGetAgentRuntimeSession.mockImplementation(async (sessionId: string) => ({
    id: sessionId,
    thread_id: sessionId,
    messages: [],
  }));
  mockGetAgentRuntimeThreadRead.mockImplementation(async (threadId: string) => {
    const threadRead =
      mockAgentSessionDetailByThreadId.get(threadId)?.thread_read;
    return threadRead &&
      typeof threadRead === "object" &&
      !Array.isArray(threadRead)
      ? { ...threadRead, thread_id: threadId }
      : { thread_id: threadId, status: "idle" };
  });
  mockReadAgentRuntimeThread.mockImplementation(async (threadId: string) => ({
    thread: {
      createdAt: 0.001,
      id: threadId,
      sessionId: threadId,
      status: { type: "idle" },
      turns: [],
      updatedAt: 0.001,
    },
  }));
  mockGenerateAgentRuntimeSessionTitle.mockResolvedValue("");
  mockSetAgentRuntimeThreadName.mockResolvedValue(undefined);
  mockUpdateAgentRuntimeThreadSettings.mockResolvedValue(undefined);
  mockDeleteAgentRuntimeSession.mockResolvedValue(undefined);
  mockCompactAgentRuntimeSession.mockResolvedValue(undefined);
  mockInterruptAgentRuntimeTurn.mockResolvedValue(undefined);
  mockResumeAgentRuntimeThread.mockImplementation(
    async ({ threadId }: { threadId: string }) => ({
      result: { thread: { id: threadId } },
    }),
  );
  mockReplayAgentRuntimeRequest.mockResolvedValue(null);
  mockRespondAgentRuntimeAction.mockResolvedValue(undefined);
  mockParseAgentEvent.mockImplementation((payload: unknown) => payload);
  mockSafeListen.mockResolvedValue(() => {});
  mockResolveClawWorkspaceProviderSelection.mockImplementation(
    async (input?: {
      currentProviderType?: string | null;
      currentModel?: string | null;
    }) => ({
      providerType: input?.currentProviderType?.trim() || "openai",
      model: input?.currentModel?.trim() || "gpt-5.4-mini",
    }),
  );
  clearAllAgentStreamTextOverlays();
});

afterEach(() => {
  clearAllAgentStreamTextOverlays();
  localStorage.clear();
  sessionStorage.clear();
});

export function getAgentStreamTextOverlay(
  messageId: string | null | undefined,
) {
  return readAgentStreamTextOverlay(messageId);
}
