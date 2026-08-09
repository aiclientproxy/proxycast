import {
  type AppServerClient,
  type AppServerReviewStartParams,
  type AppServerReviewStartResponse,
  type AppServerReviewTarget,
} from "./appServer";
import { parseAgentEvent, type AgentEvent } from "./agentProtocol";
import {
  listenAgentRuntimeEvent,
  type AgentRuntimeEventListener,
} from "./agentRuntimeEvents";
import { submitAgentRuntimeReview } from "./agentRuntime/threadClient";

export type ReviewClient = Pick<AppServerClient, "startReview">;
export type ReviewTerminalEvent = Extract<
  AgentEvent,
  { type: "turn_completed" | "turn_failed" | "turn_canceled" }
>;

export interface ReviewStartOptions {
  onTerminal?: (event: ReviewTerminalEvent) => void | Promise<void>;
}

export interface ReviewGatewayDeps {
  client?: ReviewClient;
  listenRuntimeEvent?: AgentRuntimeEventListener;
  terminalTimeoutMs?: number;
}

const DEFAULT_REVIEW_TERMINAL_TIMEOUT_MS = 30 * 60 * 1000;

export async function startReview(
  params: AppServerReviewStartParams,
  options: ReviewStartOptions = {},
  deps: ReviewGatewayDeps = {},
): Promise<AppServerReviewStartResponse> {
  const normalizedParams = normalizeReviewParams(params);
  const terminalObserver = options.onTerminal
    ? await createReviewTerminalObserver({
        threadId: normalizedParams.threadId,
        onTerminal: options.onTerminal,
        listenRuntimeEvent:
          deps.listenRuntimeEvent ?? listenAgentRuntimeEvent,
        timeoutMs:
          deps.terminalTimeoutMs ?? DEFAULT_REVIEW_TERMINAL_TIMEOUT_MS,
      })
    : null;

  try {
    const response = deps.client
      ? await deps.client.startReview(normalizedParams)
      : await submitAgentRuntimeReview(normalizedParams);
    terminalObserver?.bindTurn(response.result.turn.id);
    return response.result;
  } catch (error) {
    terminalObserver?.dispose();
    throw error;
  }
}

interface ReviewTerminalObserver {
  bindTurn: (turnId: string) => void;
  dispose: () => void;
}

async function createReviewTerminalObserver({
  threadId,
  onTerminal,
  listenRuntimeEvent,
  timeoutMs,
}: {
  threadId: string;
  onTerminal: NonNullable<ReviewStartOptions["onTerminal"]>;
  listenRuntimeEvent: AgentRuntimeEventListener;
  timeoutMs: number;
}): Promise<ReviewTerminalObserver> {
  let expectedTurnId: string | null = null;
  let pendingTerminalEvent: ReviewTerminalEvent | null = null;
  let disposed = false;
  let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
  let unlisten: (() => void) | null = null;

  const dispose = () => {
    if (disposed) {
      return;
    }
    disposed = true;
    if (timeoutHandle !== null) {
      clearTimeout(timeoutHandle);
      timeoutHandle = null;
    }
    try {
      unlisten?.();
    } catch {
      // Cleanup must not replace the review request result.
    }
    unlisten = null;
  };

  const notifyTerminal = (event: ReviewTerminalEvent) => {
    dispose();
    void Promise.resolve()
      .then(() => onTerminal(event))
      .catch(() => undefined);
  };

  const handleTerminal = (event: ReviewTerminalEvent) => {
    const turnId = terminalTurnId(event);
    if (!turnId || disposed) {
      return;
    }
    if (!expectedTurnId) {
      pendingTerminalEvent = event;
      return;
    }
    if (turnId === expectedTurnId) {
      notifyTerminal(event);
    }
  };

  unlisten = await listenRuntimeEvent(
    `agentSession/event/${threadId}`,
    ({ payload }) => {
      const event = parseAgentEvent(payload);
      if (isReviewTerminalEvent(event)) {
        handleTerminal(event);
      }
    },
  );
  timeoutHandle = setTimeout(dispose, Math.max(0, timeoutMs));

  return {
    bindTurn(turnId) {
      expectedTurnId = turnId.trim();
      if (
        expectedTurnId &&
        pendingTerminalEvent &&
        terminalTurnId(pendingTerminalEvent) === expectedTurnId
      ) {
        notifyTerminal(pendingTerminalEvent);
      }
      pendingTerminalEvent = null;
    },
    dispose,
  };
}

function isReviewTerminalEvent(
  event: AgentEvent | null,
): event is ReviewTerminalEvent {
  return (
    event?.type === "turn_completed" ||
    event?.type === "turn_failed" ||
    event?.type === "turn_canceled"
  );
}

function terminalTurnId(event: ReviewTerminalEvent): string {
  return typeof event.turn?.id === "string" ? event.turn.id.trim() : "";
}

function normalizeReviewParams(
  params: AppServerReviewStartParams,
): AppServerReviewStartParams {
  const threadId = requiredField(params.threadId, "threadId");
  return {
    ...params,
    threadId,
    target: normalizeReviewTarget(params.target),
  };
}

function normalizeReviewTarget(
  target: AppServerReviewTarget,
): AppServerReviewTarget {
  switch (target.type) {
    case "uncommittedChanges":
      return target;
    case "baseBranch":
      return {
        ...target,
        branch: requiredField(target.branch, "branch"),
      };
    case "commit": {
      const title = target.title?.trim();
      return {
        ...target,
        sha: requiredField(target.sha, "sha"),
        title: title || null,
      };
    }
    case "custom":
      return {
        ...target,
        instructions: requiredField(target.instructions, "instructions"),
      };
  }
}

function requiredField(value: string, field: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`review/start ${field} must not be empty`);
  }
  return normalized;
}
