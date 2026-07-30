import { useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import { logAgentDebug } from "@/lib/agentDebug";

const INITIAL_SESSION_NAVIGATION_DEDUPE_MS = 2_000;
type InitialSessionSwitchTopic = (
  topicId: string,
  options?: InitialSessionSwitchOptions,
) => Promise<unknown>;

let switchTopicScopedNavigationStarts = new WeakMap<
  InitialSessionSwitchTopic,
  Map<string, number>
>();
const externalNavigationStartsBySessionId = new Map<string, number>();
const explicitNavigationRequestVersionsBySessionId = new Map<string, number>();
const explicitNavigationRequestListeners = new Set<() => void>();
let explicitNavigationRequestSequence = 0;

interface InitialSessionSwitchOptions {
  forceRefresh?: boolean;
  resumeSessionStartHooks?: boolean;
  allowDetachedSession?: boolean;
}

interface InitialSessionSwitchResolution extends InitialSessionSwitchOptions {
  waitForResolution?: boolean;
}

interface UseWorkspaceInitialSessionNavigationParams {
  initialSessionId?: string | null;
  currentSessionId?: string | null;
  shouldAllowResolvedForceMatchedHydration?: boolean;
  shouldSkipInitialSessionNavigation?: boolean;
  shouldCancelPausedInitialSessionNavigationOnCurrentSessionChange?: boolean;
  shouldPauseInitialSessionNavigation?: boolean;
  shouldHydrateMatchedInitialSession?: boolean;
  switchTopic: InitialSessionSwitchTopic;
  resolveInitialSessionSwitch?: (
    sessionId: string,
  ) => InitialSessionSwitchResolution | null | undefined;
}

function normalizeSessionId(value?: string | null): string | null {
  if (typeof value !== "string") {
    return null;
  }

  const normalized = value.trim();
  return normalized ? normalized : null;
}

export function resetInitialSessionNavigationDeduplicationForTests() {
  switchTopicScopedNavigationStarts = new WeakMap();
  externalNavigationStartsBySessionId.clear();
  explicitNavigationRequestVersionsBySessionId.clear();
  explicitNavigationRequestSequence = 0;
}

export function requestExplicitInitialSessionNavigation(sessionId: string) {
  const normalizedSessionId = normalizeSessionId(sessionId);
  if (!normalizedSessionId) {
    return;
  }

  explicitNavigationRequestSequence += 1;
  explicitNavigationRequestVersionsBySessionId.set(
    normalizedSessionId,
    explicitNavigationRequestSequence,
  );
  explicitNavigationRequestListeners.forEach((listener) => listener());
}

function subscribeExplicitInitialSessionNavigationRequests(
  listener: () => void,
) {
  explicitNavigationRequestListeners.add(listener);
  return () => explicitNavigationRequestListeners.delete(listener);
}

export function rememberInitialSessionNavigationStart(sessionId: string) {
  const normalizedSessionId = normalizeSessionId(sessionId);
  if (!normalizedSessionId) {
    return;
  }

  externalNavigationStartsBySessionId.set(normalizedSessionId, Date.now());
}

export function useWorkspaceInitialSessionNavigation({
  initialSessionId,
  currentSessionId,
  shouldAllowResolvedForceMatchedHydration = true,
  shouldSkipInitialSessionNavigation = false,
  shouldCancelPausedInitialSessionNavigationOnCurrentSessionChange = false,
  shouldPauseInitialSessionNavigation = false,
  shouldHydrateMatchedInitialSession = false,
  switchTopic,
  resolveInitialSessionSwitch,
}: UseWorkspaceInitialSessionNavigationParams) {
  const appliedInitialSessionIdRef = useRef<string | null>(null);
  const activeInitialSessionNavigationRef = useRef<{
    sessionId: string;
    promise: Promise<unknown>;
  } | null>(null);
  const failedInitialSessionIdRef = useRef<string | null>(null);
  const handledExplicitNavigationRequestVersionRef = useRef(0);
  const previousInitialSessionIdRef = useRef<string | null>(null);
  const pausedInitialSessionIdRef = useRef<string | null>(null);
  const normalizedInitialSessionId = normalizeSessionId(initialSessionId);
  const normalizedCurrentSessionId = normalizeSessionId(currentSessionId);
  const explicitNavigationRequestVersion = useSyncExternalStore(
    subscribeExplicitInitialSessionNavigationRequests,
    () =>
      normalizedInitialSessionId
        ? (explicitNavigationRequestVersionsBySessionId.get(
            normalizedInitialSessionId,
          ) ?? 0)
        : 0,
    () => 0,
  );
  const shouldRunMatchedHydration =
    normalizedCurrentSessionId === normalizedInitialSessionId &&
    shouldHydrateMatchedInitialSession;
  const resolvedSwitchOptions = useMemo(
    () =>
      normalizedInitialSessionId
        ? (resolveInitialSessionSwitch?.(normalizedInitialSessionId) ?? null)
        : null,
    [normalizedInitialSessionId, resolveInitialSessionSwitch],
  );
  const shouldForceMatchedHydration =
    normalizedCurrentSessionId === normalizedInitialSessionId &&
    shouldAllowResolvedForceMatchedHydration &&
    resolvedSwitchOptions?.forceRefresh === true;
  const shouldHydrateMatchedSession =
    shouldRunMatchedHydration || shouldForceMatchedHydration;
  const navigationMode = shouldHydrateMatchedSession
    ? "matched-hydrate"
    : "navigate";
  const appliedNavigationKey = normalizedInitialSessionId
    ? `${normalizedInitialSessionId}:${navigationMode}`
    : null;

  useEffect(() => {
    const previousInitialSessionId = previousInitialSessionIdRef.current;
    previousInitialSessionIdRef.current = normalizedInitialSessionId;

    if (!normalizedInitialSessionId) {
      appliedInitialSessionIdRef.current = null;
      pausedInitialSessionIdRef.current = null;
      failedInitialSessionIdRef.current = null;
      return;
    }

    const navigationStartedAt = Date.now();
    const registeredNavigationStartedAt =
      externalNavigationStartsBySessionId.get(normalizedInitialSessionId) ?? 0;
    const hasRegisteredLocalNavigation =
      navigationStartedAt - registeredNavigationStartedAt <
      INITIAL_SESSION_NAVIGATION_DEDUPE_MS;
    const initialSessionTargetChanged =
      previousInitialSessionId !== normalizedInitialSessionId;
    const hasExplicitNavigationRequest =
      explicitNavigationRequestVersion >
      handledExplicitNavigationRequestVersionRef.current;
    if (
      previousInitialSessionId &&
      previousInitialSessionId !== normalizedInitialSessionId
    ) {
      failedInitialSessionIdRef.current = null;
    }
    const shouldPauseForLocalDraftTransition =
      shouldPauseInitialSessionNavigation &&
      !hasExplicitNavigationRequest &&
      (!initialSessionTargetChanged ||
        normalizedCurrentSessionId === normalizedInitialSessionId ||
        hasRegisteredLocalNavigation);

    if (shouldPauseForLocalDraftTransition) {
      pausedInitialSessionIdRef.current = normalizedInitialSessionId;
      logAgentDebug(
        "AgentChatPage",
        "initialSessionNavigation.paused",
        {
          currentSessionId: normalizedCurrentSessionId,
          hasExplicitNavigationRequest,
          hasRegisteredLocalNavigation,
          initialSessionId: normalizedInitialSessionId,
          initialSessionTargetChanged,
        },
        {
          dedupeKey: `initialSessionNavigation.paused:${normalizedInitialSessionId}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    const isExplicitResumeOfPausedNavigation =
      hasExplicitNavigationRequest &&
      pausedInitialSessionIdRef.current === normalizedInitialSessionId;
    const consumeExplicitNavigationRequest = () => {
      if (!hasExplicitNavigationRequest) {
        return;
      }
      handledExplicitNavigationRequestVersionRef.current =
        explicitNavigationRequestVersion;
    };

    if (
      !hasExplicitNavigationRequest &&
      shouldCancelPausedInitialSessionNavigationOnCurrentSessionChange &&
      pausedInitialSessionIdRef.current === normalizedInitialSessionId &&
      normalizedCurrentSessionId &&
      normalizedCurrentSessionId !== normalizedInitialSessionId
    ) {
      appliedInitialSessionIdRef.current = `${normalizedInitialSessionId}:cancelled-after-current-session-change`;
      logAgentDebug(
        "AgentChatPage",
        "initialSessionNavigation.cancelledAfterCurrentSessionChange",
        {
          currentSessionId: normalizedCurrentSessionId,
          initialSessionId: normalizedInitialSessionId,
        },
        {
          dedupeKey: `initialSessionNavigation.cancelledAfterCurrentSessionChange:${normalizedInitialSessionId}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    if (shouldSkipInitialSessionNavigation) {
      consumeExplicitNavigationRequest();
      appliedInitialSessionIdRef.current = `${normalizedInitialSessionId}:skipped`;
      logAgentDebug(
        "AgentChatPage",
        "initialSessionNavigation.skipped",
        {
          currentSessionId: normalizedCurrentSessionId,
          initialSessionId: normalizedInitialSessionId,
        },
        {
          dedupeKey: `initialSessionNavigation.skipped:${normalizedInitialSessionId}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    if (
      normalizedCurrentSessionId === normalizedInitialSessionId &&
      !shouldHydrateMatchedSession
    ) {
      consumeExplicitNavigationRequest();
      appliedInitialSessionIdRef.current = `${normalizedInitialSessionId}:matched-current`;
      pausedInitialSessionIdRef.current = null;
      externalNavigationStartsBySessionId.delete(normalizedInitialSessionId);
      return;
    }

    if (
      !hasExplicitNavigationRequest &&
      normalizedCurrentSessionId &&
      normalizedCurrentSessionId !== normalizedInitialSessionId &&
      appliedInitialSessionIdRef.current?.startsWith(
        `${normalizedInitialSessionId}:`,
      )
    ) {
      logAgentDebug(
        "AgentChatPage",
        "initialSessionNavigation.preserveCurrent",
        {
          currentSessionId: normalizedCurrentSessionId,
          initialSessionId: normalizedInitialSessionId,
        },
        {
          dedupeKey: `initialSessionNavigation.preserveCurrent:${normalizedInitialSessionId}:${normalizedCurrentSessionId}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    if (
      !hasExplicitNavigationRequest &&
      appliedInitialSessionIdRef.current === appliedNavigationKey
    ) {
      return;
    }

    if (
      !hasExplicitNavigationRequest &&
      failedInitialSessionIdRef.current === normalizedInitialSessionId
    ) {
      logAgentDebug(
        "AgentChatPage",
        "initialSessionNavigation.skipFailedSession",
        {
          currentSessionId: normalizedCurrentSessionId,
          initialSessionId: normalizedInitialSessionId,
        },
        {
          dedupeKey: `initialSessionNavigation.failed:${normalizedInitialSessionId}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    const activeInitialNavigation = activeInitialSessionNavigationRef.current;
    if (activeInitialNavigation?.sessionId === normalizedInitialSessionId) {
      consumeExplicitNavigationRequest();
      appliedInitialSessionIdRef.current = appliedNavigationKey;
      logAgentDebug(
        "AgentChatPage",
        "initialSessionNavigation.reuseInFlight",
        {
          currentSessionId: normalizedCurrentSessionId,
          initialSessionId: normalizedInitialSessionId,
          matchedHydration: shouldHydrateMatchedSession,
        },
        {
          dedupeKey: `initialSessionNavigation.inFlight:${normalizedInitialSessionId}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    if (hasExplicitNavigationRequest) {
      failedInitialSessionIdRef.current = null;
    }

    if (resolvedSwitchOptions?.waitForResolution) {
      logAgentDebug(
        "AgentChatPage",
        "initialSessionNavigation.waitForResolution",
        {
          currentSessionId: normalizedCurrentSessionId,
          initialSessionId: normalizedInitialSessionId,
        },
        {
          dedupeKey: `initialSessionNavigation.wait:${normalizedInitialSessionId}`,
          throttleMs: 1000,
        },
      );
      return;
    }

    const dedupeKey = `${normalizedCurrentSessionId ?? ""}->${normalizedInitialSessionId}:${navigationMode}`;
    const sessionDedupeKey = `*->${normalizedInitialSessionId}:${navigationMode}`;
    let switchTopicStarts = switchTopicScopedNavigationStarts.get(switchTopic);
    if (!switchTopicStarts) {
      switchTopicStarts = new Map();
      switchTopicScopedNavigationStarts.set(switchTopic, switchTopicStarts);
    }

    const startedAt = navigationStartedAt;
    const scopedLastStartedAt = switchTopicStarts.get(dedupeKey) ?? 0;
    const scopedSessionLastStartedAt =
      switchTopicStarts.get(sessionDedupeKey) ?? 0;
    const externalLastStartedAt =
      externalNavigationStartsBySessionId.get(normalizedInitialSessionId) ?? 0;
    const lastStartedAt = Math.max(
      scopedLastStartedAt,
      scopedSessionLastStartedAt,
      externalLastStartedAt,
    );
    if (
      !isExplicitResumeOfPausedNavigation &&
      startedAt - lastStartedAt < INITIAL_SESSION_NAVIGATION_DEDUPE_MS
    ) {
      consumeExplicitNavigationRequest();
      appliedInitialSessionIdRef.current = appliedNavigationKey;
      logAgentDebug(
        "AgentChatPage",
        "initialSessionNavigation.deduped",
        {
          currentSessionId: normalizedCurrentSessionId,
          initialSessionId: normalizedInitialSessionId,
          elapsedSinceLastStartMs: startedAt - lastStartedAt,
        },
        {
          dedupeKey: `initialSessionNavigation.deduped:${normalizedInitialSessionId}`,
          throttleMs: INITIAL_SESSION_NAVIGATION_DEDUPE_MS,
        },
      );
      return;
    }

    consumeExplicitNavigationRequest();
    appliedInitialSessionIdRef.current = appliedNavigationKey;
    pausedInitialSessionIdRef.current = null;
    switchTopicStarts.set(dedupeKey, startedAt);
    switchTopicStarts.set(sessionDedupeKey, startedAt);
    logAgentDebug("AgentChatPage", "initialSessionNavigation.start", {
      currentSessionId: normalizedCurrentSessionId,
      forceRefresh:
        shouldHydrateMatchedSession ||
        resolvedSwitchOptions?.forceRefresh === true,
      initialSessionId: normalizedInitialSessionId,
      matchedHydration: shouldHydrateMatchedSession,
      resumeSessionStartHooks:
        resolvedSwitchOptions?.resumeSessionStartHooks === true,
      allowDetachedSession:
        resolvedSwitchOptions?.allowDetachedSession === true,
    });

    const switchOptions: InitialSessionSwitchOptions = {
      ...(shouldHydrateMatchedSession ||
      resolvedSwitchOptions?.forceRefresh === true
        ? { forceRefresh: true }
        : {}),
      ...(resolvedSwitchOptions?.resumeSessionStartHooks === true
        ? { resumeSessionStartHooks: true }
        : {}),
      ...(resolvedSwitchOptions?.allowDetachedSession === true
        ? { allowDetachedSession: true }
        : {}),
    };
    const hasSwitchOptions = Object.keys(switchOptions).length > 0;

    const switchPromise = switchTopic(
      normalizedInitialSessionId,
      hasSwitchOptions ? switchOptions : undefined,
    );
    activeInitialSessionNavigationRef.current = {
      sessionId: normalizedInitialSessionId,
      promise: switchPromise,
    };
    void switchPromise
      .then((result) => {
        if (
          activeInitialSessionNavigationRef.current?.promise !== switchPromise
        ) {
          return;
        }
        if (result === "error") {
          failedInitialSessionIdRef.current = normalizedInitialSessionId;
          logAgentDebug(
            "AgentChatPage",
            "initialSessionNavigation.error",
            {
              initialSessionId: normalizedInitialSessionId,
              switchResult: result,
            },
            { level: "error" },
          );
          return;
        }
        failedInitialSessionIdRef.current = null;
      })
      .catch((error) => {
        if (
          activeInitialSessionNavigationRef.current?.promise !== switchPromise
        ) {
          return;
        }
        failedInitialSessionIdRef.current = normalizedInitialSessionId;
        switchTopicStarts.delete(dedupeKey);
        externalNavigationStartsBySessionId.delete(normalizedInitialSessionId);
        logAgentDebug(
          "AgentChatPage",
          "initialSessionNavigation.error",
          {
            error,
            initialSessionId: normalizedInitialSessionId,
          },
          { level: "error" },
        );
        console.error("[AgentChatPage] 恢复初始会话失败:", error);
      })
      .finally(() => {
        if (
          activeInitialSessionNavigationRef.current?.promise === switchPromise
        ) {
          activeInitialSessionNavigationRef.current = null;
        }
      });
  }, [
    normalizedCurrentSessionId,
    normalizedInitialSessionId,
    appliedNavigationKey,
    explicitNavigationRequestVersion,
    navigationMode,
    resolveInitialSessionSwitch,
    resolvedSwitchOptions,
    shouldAllowResolvedForceMatchedHydration,
    shouldSkipInitialSessionNavigation,
    shouldCancelPausedInitialSessionNavigationOnCurrentSessionChange,
    shouldPauseInitialSessionNavigation,
    shouldHydrateMatchedSession,
    shouldHydrateMatchedInitialSession,
    shouldRunMatchedHydration,
    switchTopic,
  ]);
}
