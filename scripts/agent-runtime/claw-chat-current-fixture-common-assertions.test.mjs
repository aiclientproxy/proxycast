import { describe, expect, it } from "vitest";
import { buildCommonAssertions } from "./claw-chat-current-fixture-common-assertions.mjs";

const SCENARIO_FLAGS = [
  "isCancelOnlyScenario",
  "isCancelThenContinueScenario",
  "isPlanScenario",
  "isGoalScenario",
  "isHomeHotpathScenario",
  "isImageCommandScenario",
  "isInputbarRichRestoreScenario",
  "isActiveSteerScenario",
  "isWebToolsRenderingScenario",
  "isLiveTailCommitScenario",
  "isElectronResizeReflowScenario",
  "isMcpStructuredContentScenario",
  "isMediaReferenceScenario",
  "isReasoningFirstVisibleScenario",
  "isTerminalCanceledAfterAnswerScenario",
  "isTerminalFailedAfterAnswerScenario",
  "isTerminalStaleGuardScenario",
  "isTypedErrorRetryScenario",
  "isTypedErrorRetrySuccessScenario",
  "isSkillsRuntimeScenario",
  "isSoulStyleScenario",
  "isRightSurfaceVisualMatrixScenario",
  "isContentFactoryArticleWorkspaceScenario",
  "isAnyExpertSkillsRuntimeScenario",
  "isApprovalRequestResumeScenario",
  "isApprovalRequestDeclineScenario",
  "isApprovalRequestCancelScenario",
  "isApprovalRequestDecisionScenario",
  "isApprovalRequestFullAccessScenario",
  "isExpertPlazaSkillsRuntimeScenario",
  "isExpertPanelSkillsRuntimeScenario",
  "hasCancelPhase",
];

function buildContext({ typedError = false, summary = {} } = {}) {
  return {
    rendererSnapshot: {},
    appServerRequestMethods: [],
    guiTurnStartReachedBackend: false,
    backendLedger: [],
    runtimeRequest: {},
    pageText: "",
    errorRaw: null,
    actionableConsoleErrors: [],
    summary,
    ...Object.fromEntries(SCENARIO_FLAGS.map((flag) => [flag, false])),
    isTypedErrorRetryScenario: typedError,
    isTypedErrorRetrySuccessScenario: typedError,
  };
}

function buildTraceSummary({
  hasClientLocalOutputMs = true,
  hasFirstVisibleOutputMs = true,
  hasFirstTextDeltaToFirstTextPaintMs = true,
  hasAppServerMessageDelta = true,
  hasServerEventEmittedAt = true,
  hasW3cTraceContext = true,
} = {}) {
  return {
    agentUiPerformanceTrace: {
      available: true,
      hasProviderWaitMs: false,
      hasClientLocalOutputMs,
      hasFirstVisibleOutputMs,
      hasFirstTextDeltaToFirstTextPaintMs,
      rawEntriesExported: false,
      forbiddenFragmentPresent: false,
    },
    appServerTraceEvidence: {
      available: true,
      hasProviderFirstTextDelta: false,
      hasAppServerMessageDelta,
      hasProviderWaitMs: false,
      hasServerEventEmittedAt,
      hasW3cTraceContext,
      forbiddenFragmentPresent: false,
    },
  };
}

describe("claw chat current fixture common trace assertions", () => {
  it("requires provider first-text evidence for ordinary text streaming", () => {
    const assertions = buildCommonAssertions(
      buildContext({ summary: buildTraceSummary() }),
    );

    expect(assertions.agentUiPerformanceTraceSeparatesProviderAndClient).toBe(
      false,
    );
    expect(assertions.appServerTraceEvidenceSeparatesProviderAndServer).toBe(
      false,
    );
    expect(
      assertions.appServerTraceEvidenceHasProviderFirstTextCheckpoint,
    ).toBe(false);
    expect(assertions.appServerTraceEvidenceHasProviderWaitMs).toBe(false);
    expect(assertions.appServerTraceEvidenceHasMessageDeltaCheckpoint).toBe(
      true,
    );
    expect(
      assertions.appServerTraceEvidenceHasServerEmissionTimestamp,
    ).toBe(true);
  });

  it("does not require provider first-text evidence for typed error retries", () => {
    const assertions = buildCommonAssertions(
      buildContext({ typedError: true, summary: buildTraceSummary() }),
    );

    expect(assertions.agentUiPerformanceTraceSeparatesProviderAndClient).toBe(
      true,
    );
    expect(assertions.appServerTraceEvidenceSeparatesProviderAndServer).toBe(
      true,
    );
    expect(
      assertions.appServerTraceEvidenceHasProviderFirstTextCheckpoint,
    ).toBe(true);
    expect(assertions.appServerTraceEvidenceHasProviderWaitMs).toBe(true);
    expect(assertions.appServerTraceEvidenceHasMessageDeltaCheckpoint).toBe(
      true,
    );
    expect(
      assertions.appServerTraceEvidenceHasServerEmissionTimestamp,
    ).toBe(true);
  });

  it("keeps non-provider trace evidence mandatory for typed error retries", () => {
    const assertions = buildCommonAssertions(
      buildContext({
        typedError: true,
        summary: buildTraceSummary({
          hasClientLocalOutputMs: false,
          hasFirstVisibleOutputMs: false,
          hasFirstTextDeltaToFirstTextPaintMs: false,
          hasAppServerMessageDelta: false,
          hasServerEventEmittedAt: false,
          hasW3cTraceContext: false,
        }),
      }),
    );

    expect(assertions.agentUiPerformanceTraceSeparatesProviderAndClient).toBe(
      false,
    );
    expect(assertions.agentUiPerformanceTraceHasFirstVisibleTextPaint).toBe(
      false,
    );
    expect(assertions.appServerTraceEvidenceSeparatesProviderAndServer).toBe(
      false,
    );
    expect(assertions.appServerTraceEvidenceHasMessageDeltaCheckpoint).toBe(
      false,
    );
    expect(
      assertions.appServerTraceEvidenceHasServerEmissionTimestamp,
    ).toBe(false);
    expect(assertions.appServerTraceEvidenceUsesCurrentMethods).toBe(false);
    expect(assertions.appServerTraceEvidenceHasW3cCarrier).toBe(false);
  });
});
