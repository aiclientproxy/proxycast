import fs from "node:fs";
import { describe, expect, it } from "vitest";
import { containsForbiddenTraceEvidenceFragment } from "./claw-chat-current-fixture-agent-ui-trace.mjs";
import {
  APP_SERVER_METHOD_SESSION_TURN_CANCEL,
  APPROVAL_REQUEST_RESUME_REQUEST_ID,
  HOME_HOTPATH_GREETING_SCENARIO,
  HOME_HOTPATH_SCENARIO,
  IMAGE_FIXTURE_MODEL,
  TEXT_PROVIDER_FIXTURE_API_KEY,
  TURN_PLAN_UPDATE_DONE_TEXT,
  TURN_PLAN_UPDATE_PROMPT,
  TURN_PLAN_UPDATE_STEPS,
} from "./claw-chat-current-fixture-constants.mjs";
import {
  readTraceTurnStartInputText,
  resolveGateBExpectedIdentity,
} from "./claw-chat-current-fixture-assertion-context.mjs";
import {
  buildCanonicalToolItem,
  runtimeInputText,
  summarizeRequestInput,
} from "./claw-chat-current-fixture-backend-script.mjs";
import {
  summarizeApprovalDecisionReadModel,
  summarizeApprovalSessionCacheReadModel,
} from "./claw-chat-current-fixture-approval-read-model.mjs";
import { summarizeHostInterruptCanonicalEventSequence } from "./claw-chat-current-fixture-approval-resume.mjs";
import {
  buildApprovalRequestDecisionScenarioAssertions,
  buildApprovalRequestHostInterruptScenarioAssertions,
  buildApprovalRequestResumeScenarioAssertions,
} from "./claw-chat-current-fixture-approval-assertions.mjs";
import {
  summarizeApprovalHostInterruptLifecycle,
  summarizeApprovalServerRequestLifecycle,
} from "./claw-chat-current-fixture-approval-trace.mjs";
import { isRightSurfaceSnapshotReady } from "./claw-chat-current-fixture-right-surface-visual.mjs";
import { serializeReadModelSummary } from "./claw-chat-current-fixture-image-command.mjs";
import { summarizeReadModelMediaReference } from "./claw-chat-current-fixture-media-reference.mjs";
import { buildScenarioAssertions } from "./claw-chat-current-fixture-scenario-assertions.mjs";
import { registerImageContentSmokeGuards } from "./claw-chat-current-fixture-smoke-domain-guards.mjs";
import { registerSkillsRuntimeSmokeGuards } from "./claw-chat-current-fixture-smoke-skills-runtime-guards.mjs";
import {
  buildSoulStyleTranscriptGoldenReport,
  SOUL_STYLE_TRANSCRIPT_GOLDENS,
  SOUL_STYLE_TRANSCRIPT_SURFACES,
} from "./claw-chat-current-fixture-soul-style-transcript-golden.mjs";
import { SOUL_STYLE_FIXTURE_PROFILE_IDS } from "./claw-chat-current-fixture-soul-style.mjs";
import {
  analyzeHomeHotpathSubmitToConversationSamples,
  summarizeHomeHotpathPreTurnTrace,
} from "./claw-chat-current-fixture-home-hotpath.mjs";
import { runtimeEventFromDirectNotification } from "./claw-chat-current-fixture-rpc.mjs";
import { readModelLatestTurnStatus } from "./claw-chat-current-fixture-read-model-core.mjs";
import { isGuiChatCompletedSnapshotReady } from "./claw-chat-current-fixture-gui-completion-waits.mjs";
import {
  IMAGE_PROVIDER_FIXTURE_API_KEY,
  startImageProviderFixtureServer,
  startTextProviderFixtureServer,
} from "./claw-chat-current-fixture-backend-file.mjs";
import {
  ACTIVE_STEER_DONE_TEXT,
  ACTIVE_STEER_FINAL_TEXT,
  ACTIVE_STEER_INITIAL_PROMPT,
  ACTIVE_STEER_INPUT,
  ACTIVE_STEER_SCENARIO,
  buildActiveSteerScenarioAssertions,
} from "./claw-chat-current-fixture-active-steer.mjs";

const fixtureSourceFiles = [
  "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
  "scripts/lib/electron-fixture-build.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-constants.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-backend-file.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-backend-script.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-approval-backend-events.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-backend-tool-skill-events.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-backend-ledger.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-rpc.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-invoke-trace.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-agent-ui-trace.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-read-model-core.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-read-model-waits.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-session.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-gui-completion-waits.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-gui-input-modes.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-gui-tool-waits.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-gui-web-tools-waits.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-turn-plan-update.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-approval-resume.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-approval-gui.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-approval-read-model.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-approval-trace.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-approval-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-image-command.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-media-reference.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-image-command-workflow-read.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-skills-workspace.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-inputbar-rich-restore.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-active-steer.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-home-hotpath.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-skills-runtime-flow.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-terminal-after-answer.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-terminal-stale-guard.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-typed-error.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-live-tail.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-resize-reflow.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-web-tools-rendering.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-plan-history.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-right-surface-visual.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-content-factory-article-workspace.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-inline-image-article-workspace.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-content-factory-workspace-patches.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-scenario-flow.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-common-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-scenario-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-resize-reflow-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-runtime-surface-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-skills-runtime-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-terminal-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-web-tools-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-soul-style-transcript-golden.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-soul-style.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-content-factory-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-not-applicable-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-assertion-context.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-assertions.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-gate-b-contract.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-gate-b-execution-evidence.mjs",
];

describe("Claw GUI canonical completion guard", () => {
  const completedSnapshot = {
    hasPrompt: true,
    hasAssistantSummary: true,
    hasDoneText: false,
    requiredVisibleTextHits: [],
    assistantScopeDedupeGuardHits: [],
    disallowedVisibleTextHits: [],
    textareaVisible: true,
    textareaDisabled: false,
    textareaSessionId: "session-expert",
    stopButtonVisible: false,
    completionScope: {
      runtimeTurnId: "turn-expert",
      runtimeTurnStatus: "completed",
      assistantRuntimeTurnId: "turn-expert",
    },
  };

  it("rejects visible expert output until the assistant leaves its pending turn", () => {
    expect(
      isGuiChatCompletedSnapshotReady(
        {
          ...completedSnapshot,
          completionScope: {
            ...completedSnapshot.completionScope,
            assistantRuntimeTurnId: "pending-turn:expert",
          },
        },
        {
          expectedIdentity: {
            sessionId: "session-expert",
            turnId: "turn-expert",
          },
        },
      ),
    ).toBe(false);
    expect(
      isGuiChatCompletedSnapshotReady(completedSnapshot, {
        expectedIdentity: {
          sessionId: "session-expert",
          turnId: "turn-expert",
        },
      }),
    ).toBe(true);
  });
});

describe("Claw read model status guard", () => {
  it("reads terminal status from canonical top-level turns", () => {
    expect(
      readModelLatestTurnStatus({
        turns: [{ id: "turn-typed-error", status: "completed" }],
        detail: { status: "running" },
      }),
    ).toBe("completed");
  });
});

function readSmokeScript() {
  return fixtureSourceFiles
    .map((filePath) => fs.readFileSync(filePath, "utf8"))
    .join("\n");
}

function removeContentFactoryForbiddenMarkerGuard(content) {
  return content.replace(
    /const (?:FORBIDDEN_CONTENT_FACTORY_ARTICLE_TEMPLATE_MARKERS|forbiddenArticleTemplateMarkers) = \[[\s\S]*?\];/g,
    "",
  );
}

function readCurrentFixtureRegressionSmokeScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/current-fixture-regression-smoke.mjs",
    "utf8",
  );
}

function readExpertActionsScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-expert-actions.mjs",
    "utf8",
  );
}

function readGuiActionsScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-gui-actions.mjs",
    "utf8",
  );
}

function readFixtureUtilsScript() {
  return fs.readFileSync(
    "scripts/agent-runtime/claw-chat-current-fixture-utils.mjs",
    "utf8",
  );
}

describe("claw chat current Electron fixture smoke guard", () => {
  it.each([HOME_HOTPATH_SCENARIO, HOME_HOTPATH_GREETING_SCENARIO])(
    "%s Gate B identity binds to the home submission turn",
    (scenario) => {
      const identity = resolveGateBExpectedIdentity({
        summary: {
          sessionId: "precreated-session",
          threadId: "precreated-thread",
          homeHotpath: {
            backendTurnStart: {
              sessionId: "home-session",
              turnId: "home-turn",
            },
          },
        },
        options: { scenario },
        backendLedger: [
          {
            kind: "turnStart",
            sessionId: "precreated-session",
            threadId: "precreated-thread",
            turnId: "precreated-turn",
          },
          {
            kind: "turnStart",
            sessionId: "other-session",
            threadId: "other-thread",
            turnId: "other-turn",
          },
          {
            kind: "turnStart",
            sessionId: "home-session",
            threadId: null,
            turnId: "home-turn",
          },
          {
            kind: "turnStart",
            sessionId: "latest-session",
            threadId: "latest-thread",
            turnId: "latest-turn",
          },
        ],
        appServerRequests: [
          {
            method: "thread/read",
            params: { threadId: "wrong-thread" },
            response: {
              sessionId: "different-session",
              threadId: "wrong-thread",
              turns: [{ turnId: "home-turn" }],
            },
          },
          {
            method: "thread/read",
            params: { threadId: "other-turn-thread" },
            response: {
              sessionId: "home-session",
              threadId: "other-turn-thread",
              turns: [{ turnId: "other-turn" }],
            },
          },
          {
            method: "thread/read",
            params: { threadId: "" },
            response: {
              sessionId: "home-session",
              threadId: null,
              turns: [{ turnId: "home-turn" }],
            },
          },
          {
            method: "thread/read",
            params: { threadId: "home-thread" },
            response: {
              sessionId: "home-session",
              threadId: "home-thread",
              turns: [{ turnId: "home-turn" }],
            },
          },
          {
            method: "thread/read",
            params: { threadId: "latest-thread" },
            response: {
              sessionId: "latest-session",
              threadId: "latest-thread",
              turns: [{ turnId: "latest-turn" }],
            },
          },
        ],
      });

      expect(identity).toEqual({
        sessionId: "home-session",
        threadId: "home-thread",
        turnId: "home-turn",
      });
    },
  );

  it("keeps the precreated Gate B identity for non-home scenarios", () => {
    expect(
      resolveGateBExpectedIdentity({
        summary: {
          sessionId: "precreated-session",
          threadId: "precreated-thread",
        },
        options: { scenario: "complete" },
        appServerRequests: [],
        backendLedger: [
          {
            kind: "turnStart",
            sessionId: "runtime-session",
            threadId: "runtime-thread",
            turnId: "runtime-turn",
          },
        ],
      }),
    ).toEqual({
      sessionId: "precreated-session",
      threadId: "precreated-thread",
    });
  });

  it("binds skills runtime Gate B identity to the final manual-enable turn", () => {
    expect(
      resolveGateBExpectedIdentity({
        summary: {
          sessionId: "precreated-session",
          threadId: "precreated-thread",
          manualEnableSkillsRuntimeTurnStart: {
            backend: {
              sessionId: "skills-session",
              turnId: "skills-turn",
            },
          },
        },
        options: { scenario: "skills-runtime" },
        backendLedger: [
          {
            kind: "turnStart",
            sessionId: "skills-session",
            turnId: "skills-turn",
          },
        ],
        appServerRequests: [
          {
            method: "thread/read",
            params: { threadId: "skills-thread" },
            response: {
              sessionId: "skills-session",
              threadId: "skills-thread",
              turns: [{ turnId: "skills-turn" }],
            },
          },
        ],
      }),
    ).toEqual({
      sessionId: "skills-session",
      threadId: "skills-thread",
      turnId: "skills-turn",
    });
  });

  it("fails closed when the home submission turn has no matching ledger identity", () => {
    expect(() =>
      resolveGateBExpectedIdentity({
        summary: {
          sessionId: "precreated-session",
          threadId: "precreated-thread",
          homeHotpath: {
            backendTurnStart: {
              sessionId: "home-session",
              turnId: "home-turn",
            },
          },
        },
        options: { scenario: HOME_HOTPATH_SCENARIO },
        appServerRequests: [
          {
            method: "thread/read",
            params: { threadId: "home-thread" },
            response: {
              sessionId: "home-session",
              threadId: "home-thread",
              turns: [{ turnId: "home-turn" }],
            },
          },
        ],
        backendLedger: [
          {
            kind: "turnStart",
            sessionId: "home-session",
            threadId: "wrong-thread",
            turnId: "different-turn",
          },
          {
            kind: "turnStart",
            sessionId: "different-session",
            threadId: "wrong-thread",
            turnId: "home-turn",
          },
        ],
      }),
    ).toThrow(/matching backend turnStart and thread\/read evidence/);
  });

  it("fails closed when no session read binds the home session and turn", () => {
    expect(() =>
      resolveGateBExpectedIdentity({
        summary: {
          homeHotpath: {
            backendTurnStart: {
              sessionId: "home-session",
              turnId: "home-turn",
            },
          },
        },
        options: { scenario: HOME_HOTPATH_SCENARIO },
        backendLedger: [
          {
            kind: "turnStart",
            sessionId: "home-session",
            threadId: null,
            turnId: "home-turn",
          },
        ],
        appServerRequests: [
          {
            method: "thread/read",
            params: { threadId: "wrong-thread" },
            response: {
              sessionId: "home-session",
              threadId: "wrong-thread",
              turns: [{ turnId: "different-turn" }],
            },
          },
          {
            method: "thread/read",
            params: { threadId: "different-thread" },
            response: {
              sessionId: "home-session",
              threadId: "wrong-thread",
              turns: [{ turnId: "home-turn" }],
            },
          },
        ],
      }),
    ).toThrow(/matching backend turnStart and thread\/read evidence/);
  });

  it("binds Gate B artifacts to one run before assertions", () => {
    const content = readSmokeScript();
    const screenshotIndex = content.indexOf("path: screenshotPath");
    const assertionIndex = content.indexOf(
      "const assertionReport = buildFixtureAssertionReport({",
    );

    expect(content).toContain("LIME_GATE_RUN_ID");
    expect(content).toContain('arg === "--run-id"');
    expect(content).toContain("runId: options.runId");
    expect(content).toContain("screenshotCaptured");
    expect(content).toContain("collectGateBGuiEvidence");
    expect(content).toContain("identityConsistent");
    expect(content).toContain("explicitTerminalOrPending");
    expect(screenshotIndex).toBeGreaterThan(-1);
    expect(assertionIndex).toBeGreaterThan(screenshotIndex);
  });

  it("首页首发采样只允许从完整首页单向切换到 conversation", () => {
    const mainAreaBounds = { left: 294, top: 8, width: 1134, height: 980 };
    const homeSample = (elapsedMs) => ({
      elapsedMs,
      hasConnectedComposer: true,
      hasEmptyConversationText: false,
      hasEmptyStateFirstScreen: true,
      hasMessageList: false,
      hasNoAvailableModelText: false,
      hasTaskCenterHomeText: true,
      imperativePendingShellCount: 0,
      mainAreaBounds,
      promptInBody: false,
    });
    const conversationSample = (elapsedMs) => ({
      elapsedMs,
      hasConnectedComposer: false,
      hasEmptyConversationText: false,
      hasEmptyStateFirstScreen: false,
      hasMessageList: true,
      hasNoAvailableModelText: false,
      hasTaskCenterHomeText: false,
      imperativePendingShellCount: 0,
      mainAreaBounds,
      promptInBody: true,
    });

    const stable = analyzeHomeHotpathSubmitToConversationSamples(
      [
        homeSample(16),
        homeSample(32),
        conversationSample(48),
        conversationSample(64),
      ],
      mainAreaBounds,
    );
    expect(stable).toMatchObject({
      stable: true,
      conversationStartedAtMs: 48,
      beforeConversationSampleCount: 2,
      conversationSampleCount: 2,
      unstableCount: 0,
    });

    const returnedHome = analyzeHomeHotpathSubmitToConversationSamples(
      [homeSample(16), conversationSample(32), homeSample(48)],
      mainAreaBounds,
    );
    expect(returnedHome.stable).toBe(false);
    expect(returnedHome.firstUnstableConversationSamples).toHaveLength(1);

    const blankIntermediate = analyzeHomeHotpathSubmitToConversationSamples(
      [
        homeSample(16),
        {
          ...homeSample(32),
          hasConnectedComposer: false,
          hasEmptyStateFirstScreen: false,
          hasTaskCenterHomeText: false,
        },
        conversationSample(48),
      ],
      mainAreaBounds,
    );
    expect(blankIntermediate.stable).toBe(false);
    expect(
      blankIntermediate.firstInvalidBeforeConversationSamples,
    ).toHaveLength(1);
  });

  it("drives the real Electron Desktop Host bridge and App Server JSON-RPC", () => {
    const content = readSmokeScript();

    expect(content).toContain("import { _electron as electron, chromium }");
    expect(content).toContain("electron.launch({");
    expect(content).toContain("--cdp-port");
    expect(content).toContain("--remote-debugging-port=");
    expect(content).toContain("chromium.connectOverCDP");
    expect(content).toContain("findElectronCdpPage");
    expect(content).toContain("LIME_ELECTRON_REMOTE_DEBUGGING_PORT");
    expect(content).toContain("ensureElectronFixtureBuild");
    expect(content).toContain("../lib/electron-fixture-build.mjs");
    expect(content).toContain("rebuilding stale packaged fixture assets");
    expect(content).toContain("buildStaleElectronFixtureSegments");
    expect(content).toContain("waitForAppUrlReady");
    expect(content).toContain('logStage("wait-app-url")');
    expect(content).toContain('"--use-mock-keychain"');
    expect(content).toContain("ELECTRON_E2E_USER_DATA_DIR");
    expect(content).toContain('LIME_ELECTRON_E2E: "1"');
    expect(content).toContain('LIME_ELECTRON_DEV_HTTP_BRIDGE: "0"');
    expect(content).toContain("window.__LIME_ELECTRON__ === true");
    expect(content).toContain(
      'typeof window.electronAPI?.invoke === "function"',
    );
    expect(content).toContain("window.electronAPI.supportsCommand");
    expect(content).toContain("app_server_handle_json_lines");
    expect(content).toContain("startInvokeTraceEvidenceCollector");
    expect(content).toContain('"initialize"');
    expect(content).toContain('"initialized"');
  });

  it("uses GUI input to submit the news prompt instead of calling turn/start directly", () => {
    const content = readSmokeScript();
    const guiActionsContent = readGuiActionsScript();
    const regressionContent = readCurrentFixtureRegressionSmokeScript();

    expect(guiActionsContent).toContain('textarea[name="agent-chat-message"]');
    expect(content + guiActionsContent).toContain("waitForInputReady");
    expect(guiActionsContent).toContain("waitForSendButtonReady");
    expect(content).toContain("sendNewsPromptFromGui");
    expect(guiActionsContent).toContain("setControlledTextareaValue");
    expect(guiActionsContent).toContain('new InputEvent("input"');
    expect(content).toContain("整理今天的国际新闻");
    expect(content).toContain("promptVisibleInTextarea");
    expect(content).toContain("hasPrompt");
    expect(content).toContain('[data-testid="message-turn-group"]');
    expect(content).toContain("assistantScopeText");
    expect(content).toContain("completionScope");
    expect(content).toContain("assistantScopeDedupeGuardHits");
    expect(content).toContain("scenario.disallowedVisibleTexts");
    expect(content).toContain("今日国际新闻简要整理");
    expect(content).toContain("CLAW_NEWS_FIXTURE_DONE");
    expect(content).toContain("guiInputRemainsReady");
    expect(content).toContain("guiNotStuckStreaming");
    expect(content).toContain("noEpochFallbackTitle");
    expect(content).toContain("agentUiPerformanceTraceEvidenceAvailable");
    expect(content).toContain(
      "agentUiPerformanceTraceSeparatesProviderAndClient",
    );
    expect(content).toContain(
      "agentUiPerformanceTraceHasFirstVisibleTextPaint",
    );
    expect(content).toContain("hasFirstVisibleOutputMs");
    expect(content).toContain("hasHomeInputToFirstTextPaintMs");
    expect(content).toContain("hasStreamRequestStartToFirstTextPaintMs");
    expect(content).toContain("hasSubmitAcceptedToFirstTextPaintMs");
    expect(content).toContain("hasFirstTextDeltaToFirstTextPaintMs");
    expect(content).toContain("agentUiPerformanceTraceNoRawPayload");
    expect(content).toContain("collectAppServerTraceEvidence");
    expect(content).toContain("appServerTraceEvidenceAvailable");
    expect(content).toContain("appServerTraceEvidenceUsesCurrentMethods");
    expect(content).toContain(
      "appServerTraceEvidenceHasProviderFirstTextCheckpoint",
    );
    expect(content).toContain(
      "appServerTraceEvidenceHasMessageDeltaCheckpoint",
    );
    expect(content).toContain("appServerTraceEvidenceHasProviderWaitMs");
    expect(content).toContain(
      "appServerTraceEvidenceHasServerEmissionTimestamp",
    );
    expect(content).toContain("serverEventEmittedAt");
    expect(content).toContain("providerWaitMs");
    expect(content).toContain("appServerTraceEvidenceHasW3cCarrier");
    expect(content).toContain("appServerTraceEvidenceExportedSummaryOnly");
    expect(content).toContain(
      "appServerTraceSupportBundleOptInUsesCurrentMethod",
    );
    expect(content).toContain("appServerTraceSupportBundleOptInSummaryOnly");
    expect(content).toContain('"diagnostics/supportBundle/export"');
    expect(content).toContain("includeTraceExport");
    expect(content).toContain("LIME_SUPPORT_BUNDLE_OUTPUT_DIR");
    expect(content).toContain('"diagnostics/trace/export"');
    expect(content).toContain("recordAgentUiPerformanceTraceEvidence");
    expect(content).toContain("agentUiPerformanceTraceLatest");
    expect(content).toContain("traceEvidenceHasProviderAndClient(evidence)");
    expect(content).toContain("HOME_HOTPATH_SCENARIO");
    expect(content).toContain('"home-hotpath"');
    expect(content).toContain("HOME_HOTPATH_GREETING_SCENARIO");
    expect(content).toContain('"home-hotpath-greeting"');
    expect(content).toContain("GREETING_PROMPT");
    expect(content).toContain("GREETING_DONE_TEXT");
    expect(content).toContain("GREETING_SUMMARY_TEXT");
    expect(content).toContain("runHomeHotpathScenario");
    expect(content).toContain("scenarioConfig");
    expect(content).toContain("allowTaskCenterHomeInput");
    expect(content).toContain("homeHotpathNoBlankConversationAfterSubmit");
    expect(content).toContain("collectAfterFillStability");
    expect(content).toContain("afterFillStability");
    expect(content).toContain("postCompletionStability");
    expect(content).toContain("homeHotpathNoTransientFallbackAfterInputFill");
    expect(content).toContain("homeHotpathNoPostCompletionRefreshFlicker");
    expect(content).toContain("homeHotpathNoImperativePendingShell");
    expect(content).toContain("homeHotpathMainAreaBoundsStable");
    expect(content).toContain("data-home-hotpath-pending-shell");
    expect(content).toContain("workspace-main-area");
    expect(content).toContain("HOME_HOTPATH_BLOCKED_PRE_TURN_METHODS");
    expect(content).toContain("blockedAuxiliaryMethodsBeforeTurnStart");
    expect(content).toContain("homeHotpathPreTurnTraceWindowAvailable");
    expect(content).toContain("homeHotpathNoAuxiliaryAppServerBeforeTurnStart");
    expect(content).toContain('"sessionFile/getOrCreate"');
    expect(content).toContain('"workspaceRightSurface/pending/list"');
    expect(content).toMatch(
      /HOME_HOTPATH_BLOCKED_PRE_TURN_METHODS[\s\S]*"modelPreferences\/list"[\s\S]*"modelSyncState\/read"/,
    );
    expect(content).toContain("homeHotpathPendingPreviewPaintWithinBudget");
    expect(content).toContain("HOME_HOTPATH_PENDING_PREVIEW_PAINT_BUDGET_MS");
    expect(content).toContain("HOME_HOTPATH_SEND_DISPATCH_BUDGET_MS");
    expect(content).toContain("HOME_HOTPATH_SUBMIT_ACCEPTED_BUDGET_MS");
    expect(content).toContain(
      "const HOME_HOTPATH_SUBMIT_ACCEPTED_BUDGET_MS = 1800;",
    );
    expect(content).toContain("HOME_HOTPATH_TEXT_DELTA_TO_PAINT_BUDGET_MS");
    expect(content).toContain("homeHotpathSendDispatchWithinBudget");
    expect(content).toContain("homeHotpathTraceHasSubmitAccepted");
    expect(content).toContain("homeHotpathSubmitAcceptedWithinBudget");
    expect(content).toContain("homeHotpathTextDeltaToPaintWithinBudget");
    expect(content).toContain("inputbarTriggerToPendingPreviewPaintMs");
    expect(content).toContain("homeInputToPendingPreviewPaintMs");
    expect(content).toContain("inputbarTriggerToSubmitAcceptedMs");
    expect(content).toContain("homeInputToSubmitAcceptedMs");
    expect(content).toContain("sendDispatchToSubmitAcceptedMs");
    expect(content).toContain("firstTextDeltaToFirstTextPaintMs");
    expect(content).toContain("isGreetingPrompt");
    expect(content).toContain("CLAW_GREETING_FIXTURE_DONE");
    expect(regressionContent).toContain("home-hotpath-greeting");
    expect(regressionContent).toContain(
      "claw-chat-current-fixture-home-hotpath-greeting-regression",
    );
  });

  it("rebuilds the home hotpath pre-turn window from collected invoke trace", () => {
    const prompt = "整理今天的国际新闻";
    const inputSend = {
      clicked: { clickedAt: "2026-07-20T12:00:00.100Z" },
    };
    const preTurnTrace = summarizeHomeHotpathPreTurnTrace(
      [
        {
          command: "app_server_handle_json_lines",
          timestamp: "2026-07-20T12:00:01.000Z",
          duration_ms: 500,
          status: "success",
          transport: "electron-ipc",
          args_preview: {
            request: {
              lines: [
                JSON.stringify({
                  jsonrpc: "2.0",
                  id: 1,
                  method: "turn/start",
                  params: {
                    threadId: "thread-current",
                    input: { text: prompt },
                  },
                }),
              ],
            },
          },
        },
      ],
      inputSend,
      prompt,
    );

    expect(preTurnTrace).toMatchObject({
      clickAt: "2026-07-20T12:00:00.100Z",
      turnStartAt: "2026-07-20T12:00:01.000Z",
      turnStartMs: Date.parse("2026-07-20T12:00:00.500Z"),
      turnStartSource: "renderer-safe-invoke",
      traceRequestCount: 1,
      blockedAuxiliaryMethodsBeforeTurnStart: [],
    });
    expect(
      summarizeHomeHotpathPreTurnTrace(
        [],
        inputSend,
        prompt,
        "2026-07-20T12:00:02.000Z",
      ),
    ).toMatchObject({
      turnStartAt: "2026-07-20T12:00:02.000Z",
      turnStartMs: Date.parse("2026-07-20T12:00:02.000Z"),
      turnStartSource: "external-backend-ledger",
    });
    expect(
      summarizeHomeHotpathPreTurnTrace([], inputSend, prompt).turnStartAt,
    ).toBeNull();

    const assertions = buildScenarioAssertions({
      appServerRequestMethods: [],
      isHomeHotpathScenario: true,
      summary: {
        agentUiPerformanceTraceDeferred: { reason: "unrelated trace gap" },
        homeHotpath: {
          preTurnTrace: {
            clickAt: inputSend.clicked.clickedAt,
            turnStartAt: null,
            blockedAuxiliaryMethodsBeforeTurnStart: [],
          },
        },
      },
    });
    expect(assertions.homeHotpathPreTurnTraceWindowAvailable).toBe(false);
  });

  it("reads typed turn/start text parts from the renderer trace", () => {
    expect(
      readTraceTurnStartInputText([
        { type: "text", text: "整理今天的国际新闻" },
        { type: "image", url: "data:image/png;base64,ignored" },
      ]),
    ).toBe("整理今天的国际新闻");
  });

  it("passes home hotpath text matchers into page.evaluate instead of closing over Node imports", () => {
    const content = fs.readFileSync(
      "scripts/agent-runtime/claw-chat-current-fixture-home-hotpath.mjs",
      "utf8",
    );
    const snapshotFunction = content.slice(
      content.indexOf("function readHomeHotpathSnapshot"),
      content.indexOf("async function waitForHomeHotpathReady"),
    );

    expect(content).toContain("HOME_HOTPATH_MATCHERS");
    expect(content).toContain("prompt: NEWS_PROMPT");
    expect(content).toContain("doneText: ASSISTANT_DONE_TEXT");
    expect(content).toContain("normalizeHomeHotpathScenarioConfig");
    expect(content).toContain("config.prompt");
    expect(content).toContain("config.doneText");
    expect(content).toContain("config.summaryText");
    expect(content).toContain(
      "promptInBody: prompt ? bodyText.includes(prompt) : false",
    );
    expect(content).toContain(
      "doneInBody: doneText ? bodyText.includes(doneText) : false",
    );
    expect(content).toContain("readHomeHotpathSnapshot");
    expect(content).toContain("HOME_HOTPATH_MATCHERS");
    expect(snapshotFunction).not.toContain("NEWS_PROMPT");
    expect(snapshotFunction).not.toContain("ASSISTANT_DONE_TEXT");
  });

  it("does not run default news completion waits after scenario-owned flows complete", () => {
    const content = fs.readFileSync(
      "scripts/agent-runtime/claw-chat-current-fixture-scenario-flow.mjs",
      "utf8",
    );

    expect(content).toContain(
      "options.scenario === ELECTRON_RESIZE_REFLOW_SCENARIO",
    );
    expect(content).toContain(
      "options.scenario !== ELECTRON_RESIZE_REFLOW_SCENARIO",
    );
    expect(content).toContain("options.scenario === HOME_HOTPATH_SCENARIO");
    expect(content).toContain(
      "options.scenario === HOME_HOTPATH_GREETING_SCENARIO",
    );
    expect(content).toContain("!isHomeHotpathScenario");
    expect(content).toContain(
      "options.scenario !== APPROVAL_REQUEST_HOST_INTERRUPT_SCENARIO",
    );
    expect(content.indexOf("runElectronResizeReflowScenario")).toBeLessThan(
      content.indexOf("wait-gui-completed"),
    );
    expect(content.indexOf("runHomeHotpathScenario")).toBeLessThan(
      content.indexOf("wait-gui-completed"),
    );
  });

  it("does not classify redacted task ids as leaked API keys in trace evidence", () => {
    expect(
      containsForbiddenTraceEvidenceFragment({
        sessionId: "task-[redacted]",
        source: "task-[redacted]",
      }),
    ).toBe(false);
    expect(
      containsForbiddenTraceEvidenceFragment({
        authorization: "Bearer [redacted]",
      }),
    ).toBe(true);
    expect(
      containsForbiddenTraceEvidenceFragment({
        token: "sk-1234567890abcdef",
      }),
    ).toBe(true);
  });

  it("uses a local external fixture backend and current direct v2 methods", () => {
    const content = readSmokeScript();

    expect(content).toContain('"app_server_drain_events"');
    expect(content).toContain('"turn/started"');
    expect(content).toContain('"turn/completed"');
    expect(content).toContain('"item/started"');
    expect(content).toContain('"item/completed"');
    expect(content).toContain('"item/agentMessage/delta"');
    expect(content).not.toContain('"agentSession/event"');
    expect(content).toContain('APP_SERVER_BACKEND_MODE: "external"');
    expect(content).toContain('APP_SERVER_BACKEND_MODE: "runtime"');
    expect(content).toContain("resolveScenarioBackendEnv");
    expect(content).toContain("APP_SERVER_BACKEND_COMMAND: process.execPath");
    expect(content).toContain("writeFixtureBackend");
    expect(content).toContain('const FIXTURE_PROVIDER = "fixture-provider"');
    expect(content).toContain('const FIXTURE_MODEL = "fixture-model"');
    expect(content).toContain('"thread/start"');
    expect(content).not.toContain('"agentSession/update"');
    expect(content).toContain('"thread/settings/update"');
    expect(content).toContain('"turn/start"');
    expect(content).toContain('"turn/interrupt"');
    expect(content).toContain('"thread/read"');
    expect(content).toContain('"thread/list"');
    expect(content).toContain('"workspace/default/ensure"');
    expect(content).toContain('kind === "turnStart"');
    expect(content).toContain('kind === "turnCancel"');
    expect(content).toContain('type: "message.delta"');
    expect(content).toContain('type: "message.completed"');
    expect(content).toContain('type: "turn.completed"');
    expect(content).toContain('type: "turn.canceled"');
    expect(content).not.toContain('type: "turn.final_done"');
  });

  it("clears Electron renderer cache for packaged file fixture runs", () => {
    const content = readSmokeScript();

    expect(content).toContain('LIME_ELECTRON_CLEAR_RENDERER_CACHE: "1"');
    expect(content).not.toContain('LIME_ELECTRON_CLEAR_RENDERER_CACHE: "0"');
  });

  it("covers stream parser completed full-text dedupe in the default Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("streamParserCompletedFullTextObserved");
    expect(content).toContain("guiStreamParserNoDuplicateFinalText");
    expect(content).toContain("readModelStreamParserNoDuplicateFinalText");
    expect(content).toContain("STREAM_PARSER_BOUNDARY_DEDUPE_GUARDS");
    expect(content).toContain("message.completed");
    expect(content).toContain("flattenBackendEmitTypesForPrompt");
    expect(content).toContain("includesOrderedEventTypes");
  });

  it("uses a local image provider stub for @配图 fixture execution", () => {
    const content = readSmokeScript();

    expect(content).toContain("startImageProviderFixtureServer");
    expect(content).toContain("LOCAL_IMAGE_SERVER_API_KEY");
    expect(content).toContain("imageProviderFixtureServer.baseUrl");
    expect(content).toContain("localImageServerApiKey");
    expect(content).toContain("/v1/images/generations");
    expect(content).toContain("IMAGE_PROVIDER_FIXTURE_DATA_URL");
    expect(content).toContain("ADAAAAAbCAMAAAANt/x");
    expect(content).not.toContain("AAAAEAAAABCAQAAAC1HAw");
  });

  it("reads canonical image task output from dynamicToolCall contentItems", () => {
    const summary = serializeReadModelSummary({
      thread: {
        turns: [
          {
            status: "completed",
            items: [
              {
                type: "dynamicToolCall",
                name: "lime_create_image_generation_task",
                status: "completed",
                contentItems: [
                  {
                    type: "inputText",
                    inputText: {
                      text: JSON.stringify({
                        task_id: "image-task-1",
                        artifact_path:
                          ".lime/tasks/image_generate/image-task-1.json",
                      }),
                    },
                  },
                ],
              },
            ],
          },
        ],
      },
    });

    expect(summary.createTaskOutputContainsTaskId).toBe(true);
    expect(summary.createTaskOutputContainsTaskFile).toBe(true);
  });

  it("binds image workflow audit to the current runtime session", () => {
    const workflowRead = {
      sessionId: "sess-current",
      activeWorkflowRunId: "",
      matchedRun: {
        workflowKey: "image_command_workflow",
        status: "completed",
        stepCounts: { total: 5 },
      },
    };
    const buildAssertions = (sessionId) =>
      buildScenarioAssertions({
        appServerRequestMethods: ["workflow/read"],
        isImageCommandScenario: true,
        pageText: "",
        summary: {
          sessionId,
          imageCommandWorkflowRead: workflowRead,
        },
      });

    expect(
      buildAssertions("sess-current")
        .imageCommandWorkflowAuditReadModelProjected,
    ).toBe(true);
    expect(
      buildAssertions("sess-stale").imageCommandWorkflowAuditReadModelProjected,
    ).toBe(false);
  });

  it("proves runtime notifications align with the same turn read model", () => {
    const content = readSmokeScript();

    expect(content).toContain("runEventReadProbe");
    expect(content).toContain("waitForRuntimeEventsForTurn");
    expect(content).toContain("waitForSessionReadContainsTurn");
    expect(content).toContain("drainAppServerEventsFromPage");
    expect(content).toContain("clientUserMessageId");
    expect(content).toContain("EVENT_READ_PROBE_DONE_TEXT");
    expect(content).toContain("EVENT_READ_PROBE_TOOL_CALL_ID");
    expect(content).toContain('const EVENT_READ_PROBE_TOOL_NAME = "WebFetch"');
    expect(content).toContain("EVENT_READ_PROBE_TOOL_OUTPUT");
    expect(content).toContain('type: "item.started"');
    expect(content).toContain('type: "item.completed"');
    expect(content).toContain("hasToolStarted");
    expect(content).toContain("hasToolResult");
    expect(content).toContain("collectReadModelToolCalls");
    expect(content).toContain("findReadModelToolCall");
    expect(content).toContain("eventReadProbeObserved");
    expect(content).toContain("readModelEventReadAligned");
    expect(content).toContain("readModelToolCallAligned");
    expect(content).toContain("containsToolOutput");
    expect(content).toContain(
      "direct v2 notification 与 read model 同 turn 对齐",
    );
    expect(content).not.toContain("eventReadProbe?.deferred === true");
  });

  it("normalizes direct v2 lifecycle notifications for the event/read probe", () => {
    expect(
      runtimeEventFromDirectNotification({
        method: "item/agentMessage/delta",
        params: {
          threadId: "thread-1",
          turnId: "turn-1",
          itemId: "item-1",
          delta: "hello",
        },
      }),
    ).toMatchObject({
      threadId: "thread-1",
      turnId: "turn-1",
      type: "message.delta",
    });
    expect(
      runtimeEventFromDirectNotification({
        method: "item/completed",
        params: {
          threadId: "thread-1",
          turnId: "turn-1",
          item: { id: "tool-1", type: "commandExecution" },
        },
      }),
    ).toMatchObject({
      threadId: "thread-1",
      turnId: "turn-1",
      type: "item.completed",
    });
    expect(
      runtimeEventFromDirectNotification({
        method: "turn/plan/updated",
        params: {
          threadId: "thread-1",
          turnId: "turn-1",
          explanation: "继续执行",
          plan: [{ step: "补主链", status: "inProgress" }],
        },
      }),
    ).toMatchObject({
      threadId: "thread-1",
      turnId: "turn-1",
      type: "turn.plan.updated",
    });
    expect(
      runtimeEventFromDirectNotification({
        method: "turn/completed",
        params: {
          threadId: "thread-1",
          turn: { id: "turn-1", status: "completed" },
        },
      }),
    ).toMatchObject({
      threadId: "thread-1",
      turnId: "turn-1",
      type: "turn.completed",
    });
  });

  it("keeps scenario-specific assertions out of unrelated evidence", () => {
    const content = readSmokeScript();

    expect(content).toContain("const commonAssertions = {");
    expect(content).toContain("const scenarioAssertions =");
    expect(content).toContain("const notApplicableAssertions =");
    expect(content).toContain(
      "summary.commonAssertions = assertionReport.commonAssertions",
    );
    expect(content).toContain(
      "summary.scenarioAssertions = assertionReport.scenarioAssertions",
    );
    expect(content).toContain(
      "summary.notApplicableAssertions = assertionReport.notApplicableAssertions",
    );
    expect(content).toContain('"readModelCanceled"');
    expect(content).toContain('"eventReadProbeObserved"');
    expect(content).toContain('"readModelToolCallAligned"');
    expect(content).toContain("isCancelThenContinueScenario");
    expect(content).toContain("hasCancelPhase");
    expect(content).toContain("continuePromptReachedBackend");
    expect(content).toContain("backendRecordedCancelThenContinue");
  });

  it("covers approval decline and cancel decision semantics through current action/respond", () => {
    const content = readSmokeScript();
    const backendScript = fs.readFileSync(
      "scripts/agent-runtime/claw-chat-current-fixture-backend-script.mjs",
      "utf8",
    );

    expect(content).toContain("approval-request-resume");
    expect(content).toContain("approval-request-decline");
    expect(content).toContain("approval-request-cancel");
    expect(content).toContain("runApprovalRequestDecisionScenario");
    expect(content).toContain("clickApprovalDecisionButton");
    expect(content).toContain('decision === "cancel"');
    expect(content).toContain('decision === "decline"');
    const approvalBackendEvents = fs.readFileSync(
      "scripts/agent-runtime/claw-chat-current-fixture-approval-backend-events.mjs",
      "utf8",
    );
    const approvalCancelBranchStart = approvalBackendEvents.indexOf(
      "const completionEvents = approvalCanceled",
    );
    const approvalCancelBranchEnd = approvalBackendEvents.indexOf(
      "\n    : approvalAllowed",
      approvalCancelBranchStart,
    );
    expect(approvalCancelBranchStart).toBeGreaterThanOrEqual(0);
    expect(approvalCancelBranchEnd).toBeGreaterThan(approvalCancelBranchStart);
    const approvalCancelCompletionBranch = approvalBackendEvents.slice(
      approvalCancelBranchStart,
      approvalCancelBranchEnd,
    );
    expect(approvalCancelCompletionBranch).toContain('type: "item.completed"');
    expect(approvalCancelCompletionBranch).toContain('status: "failed"');
    expect(approvalCancelCompletionBranch).toContain('type: "turn.canceled"');
    expect(approvalCancelCompletionBranch).toContain(
      'reason: "approval_request_cancelled"',
    );
    expect(content).toContain("approvalRequestDeclineNoToolExecuted");
    expect(content).toContain("approvalRequestCancelNoToolExecuted");
    expect(content).toContain("readModelApprovalRequestCancelCanceled");
    expect(content).toContain("APPROVAL_REQUEST_DECISION_ASSERTION_KEYS");
    expect(backendScript).toContain(
      "input.request?.runtimeOptions?.runtimeRequest",
    );
    expect(backendScript).not.toContain(
      "input.request?.runtimeOptions?.hostOptions",
    );
  });

  it("requires the resumed turn to clear its own pending approval status", () => {
    const content = readSmokeScript();

    expect(content).toContain("waitForGuiApprovalPromptAbsentAfterSecondTurn");
    expect(content).toContain('[data-testid="message-turn-group"]');
    expect(content).toContain("hasPendingApprovalStatus");
    expect(content).toMatch(
      /hasPendingApprovalStatus:\s*\(secondTurnGroup\?\.innerText \|\| ""\)\.includes\(\s*"待确认"/u,
    );
    expect(content).toContain("snapshot?.hasPendingApprovalStatus === false");
    expect(content).toContain("?.hasPendingApprovalStatus === false");
  });

  it("fails the resumed approval assertion when its turn remains pending", () => {
    const buildAssertions = (hasPendingApprovalStatus) =>
      buildApprovalRequestResumeScenarioAssertions({
        appServerRequestMethods: [],
        approvalRequestResumeTurnStart: null,
        pageText: "",
        summary: {
          guiApprovalRequestResumeSecondNoApprovalPrompt: {
            approvalPromptVisible: false,
            includesRuntimePermissionPrompt: false,
            hasPendingApprovalStatus,
            textareaVisible: true,
            textareaDisabled: false,
          },
        },
      });

    expect(
      buildAssertions(false).approvalRequestResumeSecondNoPendingApproval,
    ).toBe(true);
    expect(
      buildAssertions(true).approvalRequestResumeSecondNoPendingApproval,
    ).toBe(false);
  });

  it("binds approval response scope to the canonical generated session", () => {
    const sessionId = "sess_generated";
    const turnId = "turn_generated";
    const buildAssertions = (requestSessionId) =>
      buildApprovalRequestResumeScenarioAssertions({
        appServerRequestMethods: [],
        approvalRequestResumeTurnStart: { sessionId, turnId },
        pageText: "",
        summary: {
          sessionId,
          approvalRequestResumeRespondActionRequest: {
            params: {
              sessionId: requestSessionId,
              requestId: APPROVAL_REQUEST_RESUME_REQUEST_ID,
              actionType: "tool_confirmation",
              decision: "allow_for_session",
              actionScope: { turnId },
            },
          },
        },
      });

    expect(
      buildAssertions(sessionId).approvalRequestResumeRespondPayloadScoped,
    ).toBe(true);
    expect(
      buildAssertions("sess_other").approvalRequestResumeRespondPayloadScoped,
    ).toBe(false);
  });

  it("binds decline response scope to the canonical generated session", () => {
    const sessionId = "sess_generated";
    const turnId = "turn_generated";
    const buildAssertions = (requestSessionId) =>
      buildApprovalRequestDecisionScenarioAssertions({
        appServerRequestMethods: [],
        approvalRequestResumeTurnStart: { sessionId, turnId },
        backendLedger: [],
        isApprovalRequestCancelScenario: false,
        isApprovalRequestDeclineScenario: true,
        summary: {
          sessionId,
          approvalRequestDecisionRespondActionRequest: {
            params: {
              sessionId: requestSessionId,
              requestId: APPROVAL_REQUEST_RESUME_REQUEST_ID,
              actionType: "tool_confirmation",
              decision: "decline",
              actionScope: { turnId },
            },
          },
        },
      });

    expect(
      buildAssertions(sessionId).approvalRequestDecisionRespondPayloadScoped,
    ).toBe(true);
    expect(
      buildAssertions("sess_other").approvalRequestDecisionRespondPayloadScoped,
    ).toBe(false);
  });

  it("keeps full-access approval out of prompts, records, and action/respond", () => {
    const content = readSmokeScript();
    const regressionContent = readCurrentFixtureRegressionSmokeScript();

    expect(content).toContain("approval-request-full-access");
    expect(content).toContain("APPROVAL_REQUEST_FULL_ACCESS_PROMPT");
    expect(content).toContain("APPROVAL_REQUEST_FULL_ACCESS_RESULT_TEXT");
    expect(content).toContain("APPROVAL_REQUEST_FULL_ACCESS_DONE_TEXT");
    expect(content).toContain("runApprovalRequestFullAccessScenario");
    expect(content).toContain(
      'setInputbarAccessMode(\n    page,\n    options,\n    "full-access"',
    );
    expect(content).toContain("waitForGuiApprovalPromptAbsent");
    expect(content).toContain("approvalPromptVisible");
    expect(content).toContain("approvalRecordCount");
    expect(content).toContain("approvalRecordShape");
    expect(content).toContain("turnStartApprovalPolicy");
    expect(content).toContain("turnStartSandboxPolicy");
    expect(content).toContain('"never"');
    expect(content).toContain('"danger-full-access"');
    expect(content).toContain("approvalRequestFullAccessNoActionRespond");
    expect(content).toContain(
      "approvalRequestFullAccessNoLegacyRuntimeRespond",
    );
    expect(content).toContain("APPROVAL_REQUEST_FULL_ACCESS_ASSERTION_KEYS");
    expect(content).toMatch(
      /approvalRequestFullAccessNoActionRespond:\s*!appServerRequestMethods\.includes\(\s*APP_SERVER_METHOD_AGENT_SESSION_ACTION_RESPOND/u,
    );
    expect(regressionContent).toContain(
      "Claw approval full-access no prompt Electron fixture",
    );
    expect(regressionContent).toContain("approval-request-full-access");
    expect(regressionContent).toContain(
      "claw-chat-current-fixture-approval-request-full-access-regression",
    );
  });

  it("keeps pending approval above an enabled composer without tool argument details", () => {
    const content = readSmokeScript();

    expect(content).toContain('[data-testid="inputbar-approval-prompt"]');
    expect(content).toContain('[data-testid="inputbar-approval-summary"]');
    expect(content).toContain('approvalPrompt?.querySelector("details")');
    expect(content).toContain('approvalPrompt?.querySelector("pre")');
    expect(content).toContain("snapshot.textareaVisible === true");
    expect(content).toContain("snapshot.textareaDisabled === false");
    expect(content).toContain("snapshot.singleLine === true");
    expect(content).toContain(
      "approvalRequestResumePendingGui?.hasToolName === false",
    );
    expect(content).toContain(
      "approvalRequestResumePendingGui?.hasCommand === false",
    );
  });

  it("covers the current cancel flow through GUI stop and App Server read model", () => {
    const content = readSmokeScript();

    expect(content).toContain("--scenario <name>");
    expect(content).toContain('scenario: "complete"');
    expect(content).toContain("CLAW_CHAT_FIXTURE_SCENARIO: options.scenario");
    expect(content).toContain("waitForStopButtonVisibleAndClick");
    expect(content).toContain("requireVisibleOutput: true");
    expect(content).toContain("waitForGuiChatCanceled");
    expect(content).toContain("waitForSessionReadCanceled");
    expect(content).toContain("click-stop-from-gui");
    expect(content).toContain("wait-read-model-canceled");
    expect(content).toContain("usedCurrentTurnCancel");
    expect(content).toContain("externalFixtureCancelUsed");
    expect(content).toContain("fixtureCancelReachedBackend");
    expect(content).toContain("readModelCanceled");
    expect(content).toContain("guiStopClicked");
    expect(content).toContain("guiRunningStatusPreservedBeforeStop");
    expect(content).toContain("hasVisibleAssistantOutput");
    expect(content).toContain("hasRunningStatus");
    expect(content).toContain("statusSnapshots");
  });

  it("proves a stopped Claw turn can continue in the same current session", () => {
    const content = readSmokeScript();

    expect(content).toContain('const CONTINUE_PROMPT = "继续输出"');
    expect(content).toContain(
      'const CONTINUE_DONE_TEXT = "CLAW_CONTINUE_FIXTURE_DONE"',
    );
    expect(content).toContain("send-continue-prompt-from-gui");
    expect(content).toContain("wait-gui-continue-completed");
    expect(content).toContain("wait-read-model-continue-completed");
    expect(content).toContain("continueInputSend");
    expect(content).toContain("guiContinueCompleted");
    expect(content).toContain("readModelContinueCompleted");
    expect(content).toContain("continuePromptReachedBackend");
    expect(content).toContain("guiContinueInputSubmitted");
    expect(content).toContain("guiContinueCompleted");
    expect(content).toContain("readModelContinueCompleted");
    expect(content).toContain("backendRecordedCancelThenContinue");
    expect(content).toContain("停止后的同一会话已经可以继续输出");
  });

  it("covers Inputbar rich draft restore after an output-free cancel", () => {
    const content = readSmokeScript();
    const regressionContent = readCurrentFixtureRegressionSmokeScript();

    expect(content).toContain("inputbar-rich-restore");
    expect(content).toContain("INPUTBAR_RICH_RESTORE_PROMPT");
    expect(content).toContain("runInputbarRichRestoreScenario");
    expect(content).toContain("ensureUserVisibleCapabilityReportSkill");
    expect(content).toContain("application/x-lime-path-reference");
    expect(content).toContain("DataTransfer");
    expect(content).toContain("DragEvent");
    expect(content).toContain("inputbar-path-reference-chip");
    expect(content).toContain("input-skill-badge");
    expect(content).toContain("RICH_RESTORE_IMAGE_BASE64");
    expect(content).toContain("attachFixtureImage");
    expect(content).toContain("dropPathReference");
    expect(content).toContain("selectCapabilityReportSkill");
    expect(content).toContain("normalizeMentionText");
    expect(content).toContain("toLocaleLowerCase()");
    expect(content.split('.replace(/[\\s_-]+/g, "")')).toHaveLength(3);
    expect(content).toContain("expectedSkillName");
    expect(content).toContain("lastSelectionResult");
    expect(content).toContain("sendPromptFromGui");
    expect(content).toContain("waitForBackendLedgerTurnStart");
    expect(content).toContain("waitForStopButtonVisibleAndClick");
    expect(content).toContain("waitForInputbarRichRestoreReadModelCanceled");
    expect(content).toContain("provider.first_event.received");
    expect(content).toContain("turn.canceled");
    expect(content).toContain("CLAW_INPUTBAR_RICH_RESTORE_DONE");
    expect(content).toContain("INPUTBAR_RICH_RESTORE_FORBIDDEN_ASSISTANT_TEXT");
    expect(content).toContain("inputbarRichRestoreDraftPrepared");
    expect(content).toContain("inputbarRichRestoreInputSubmitted");
    expect(content).toContain("inputbarRichRestoreBackendInputSummaryReached");
    expect(content).toContain("inputbarRichRestoreUsedCurrentTurnCancel");
    expect(content).toContain("inputbarRichRestoreGuiCanceled");
    expect(content).toContain("inputbarRichRestoreTextRestored");
    expect(content).toContain("inputbarRichRestoreImageRestored");
    expect(content).toContain("inputbarRichRestorePathRestored");
    expect(content).toContain("inputbarRichRestoreSkillRestored");
    expect(content).toContain("inputbarRichRestoreNoVisibleAssistantOutput");
    expect(content).toContain("inputbarRichRestoreReadModelCanceled");
    expect(content).toContain("INPUTBAR_RICH_RESTORE_ASSERTION_KEYS");
    expect(content).toContain("shouldUseTextProviderFixture");
    expect(content).toContain("modelProvider/fetchModels");
    expect(content).toContain("models: []");
    expect(content).toContain('["/models", "/v1/models"]');
    expect(content).toContain('input_modalities: ["text", "image"]');
    expect(content).toContain("fixtureModelInputModalities");
    expect(content).toContain(
      "options.scenario !== INPUTBAR_RICH_RESTORE_SCENARIO",
    );
    expect(regressionContent).toContain(
      "Claw Inputbar rich draft restore output-free cancel Electron fixture",
    );
    expect(regressionContent).toContain('"inputbar-rich-restore"');
    expect(regressionContent).toContain(
      "claw-chat-current-fixture-inputbar-rich-restore-regression",
    );
    expect(regressionContent).toContain(
      "Inputbar rich draft restore output-free cancel Electron fixture",
    );
  });

  it("extracts current ordered runtime input parts", () => {
    expect(
      runtimeInputText({
        input: {
          parts: [
            { Text: { text: "first", text_elements: [] } },
            { Skill: { name: "demo", path: "/skills/demo" } },
            { Text: { text: "second", text_elements: [] } },
          ],
          agent_only: false,
        },
      }),
    ).toBe("firstsecond");
  });

  it("counts current runtime image parts by protocol variant", () => {
    expect(
      summarizeRequestInput({
        input: {
          parts: [
            {
              Image: {
                uri: "data:image/png;base64,abc",
                media_type: "image/png",
                provider_data: null,
                detail: null,
              },
            },
          ],
          agent_only: false,
        },
      }),
    ).toMatchObject({
      attachmentCount: 1,
      imageAttachmentCount: 1,
    });
  });

  it("builds typed canonical tool items for the external fixture", () => {
    expect(
      buildCanonicalToolItem({
        sessionId: "session-1",
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: "tool-1",
        ordinal: 2,
        callId: "tool-1",
        name: "browser_control",
        arguments: { command: "open https://example.com" },
      }),
    ).toMatchObject({
      item: {
        sessionId: "session-1",
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: "tool-1",
        ordinal: 2,
        kind: "tool",
        status: "inProgress",
        payload: {
          type: "tool",
          call_id: "tool-1",
          name: "browser_control",
          arguments: [{ name: "command", value: "open https://example.com" }],
          output: null,
        },
      },
    });
  });

  it("keeps retired raw tool wire out of external backend emissions", () => {
    const backendEventSources = [
      "scripts/agent-runtime/claw-chat-current-fixture-approval-backend-events.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-backend-tool-skill-events.mjs",
      "scripts/agent-runtime/skills-runtime-fixture-scenario.mjs",
    ].map((file) => fs.readFileSync(file, "utf8"));
    const retiredToolWire =
      /type: "tool\.(?:started|args|result|failed|args\.delta|input\.delta)"/u;

    for (const source of backendEventSources) {
      expect(source).not.toMatch(retiredToolWire);
    }
  });

  it("keeps session-cache control audit out of the current v2 read model", () => {
    expect(
      summarizeApprovalSessionCacheReadModel(
        {
          thread: {
            turns: [{ id: "turn-2", items: [], status: "completed" }],
          },
        },
        "turn-2",
      ),
    ).toMatchObject({
      includesApprovalSessionCacheHit: false,
      includesAllowForSession: false,
      includesSecondPermissionRequestId: false,
      includesActionRequiredForSecondPermission: false,
      includesActionResolvedForSecondPermission: false,
    });
  });

  it("proves session-cache auto-resolution with backend audit and filtered read model", () => {
    const secondTurnId = "turn-2";
    const assertions = buildApprovalRequestResumeScenarioAssertions({
      appServerRequestMethods: [],
      approvalRequestResumeTurnStart: null,
      backendLedger: [
        {
          kind: "backendEmit",
          turnId: secondTurnId,
          eventTypes: ["approval.session_cache.hit", "action.resolved"],
        },
      ],
      pageText: "",
      summary: {
        approvalRequestResumeSecondBackendTurnStart: { turnId: secondTurnId },
        readModelApprovalRequestResumeSecondCompleted: {
          pendingRequestCount: 0,
          includesApprovalSessionCacheHit: false,
          includesAllowForSession: false,
          includesSecondPermissionRequestId: false,
          includesActionRequiredForSecondPermission: false,
          includesActionResolvedForSecondPermission: false,
        },
      },
    });

    expect(assertions.approvalRequestResumeSecondReadModelAutoResolved).toBe(
      true,
    );
  });

  it("matches approval reverse response with resolved before runtime terminal", () => {
    const lifecycle = summarizeApprovalServerRequestLifecycle({
      traceMessages: [],
      lifecycleEntries: [
        {
          sequence: 1,
          kind: "terminal",
          method: "item/completed",
          threadId: "thread-1",
          turnId: "turn-1",
        },
        {
          sequence: 2,
          kind: "request",
          id: "app-server-request:boot:1",
          method: "item/commandExecution/requestApproval",
          threadId: "thread-1",
          turnId: "turn-1",
        },
        {
          sequence: 3,
          kind: "response",
          id: "app-server-request:boot:1",
          decision: "acceptForSession",
        },
        {
          sequence: 4,
          kind: "resolved",
          method: "serverRequest/resolved",
          id: "app-server-request:boot:1",
          threadId: "thread-1",
        },
        {
          sequence: 5,
          kind: "terminal",
          method: "item/completed",
          threadId: "thread-1",
          turnId: "turn-1",
        },
      ],
      notifications: [
        {
          jsonrpc: "2.0",
          method: "serverRequest/resolved",
          params: {
            requestId: "app-server-request:boot:1",
            threadId: "thread-1",
          },
        },
        {
          jsonrpc: "2.0",
          method: "item/completed",
          params: { threadId: "thread-1", turnId: "turn-1" },
        },
      ],
      wireDecision: "acceptForSession",
      threadId: "thread-1",
      turnId: "turn-1",
    });

    expect(lifecycle).toMatchObject({
      request: {
        id: "app-server-request:boot:1",
        method: "item/commandExecution/requestApproval",
        threadId: "thread-1",
        turnId: "turn-1",
      },
      response: {
        id: "app-server-request:boot:1",
        decision: "acceptForSession",
      },
      resolved: {
        method: "serverRequest/resolved",
        requestId: "app-server-request:boot:1",
        threadId: "thread-1",
      },
      responseMatchesResolved: true,
      responseBeforeResolved: true,
      runtimeTerminalMethod: "item/completed",
      resolvedBeforeRuntimeTerminal: true,
    });
  });

  it("aborts a pending approval through turn/interrupt without a renderer response", () => {
    const lifecycle = summarizeApprovalHostInterruptLifecycle({
      lifecycleEntries: [
        {
          sequence: 1,
          kind: "request",
          id: "app-server-request:boot:1",
          method: "item/commandExecution/requestApproval",
          threadId: "thread-1",
          turnId: "turn-1",
        },
        {
          sequence: 2,
          kind: "resolved",
          method: "serverRequest/resolved",
          id: "app-server-request:boot:1",
          threadId: "thread-1",
        },
        {
          sequence: 3,
          kind: "terminal",
          method: "item/completed",
          threadId: "thread-1",
          turnId: "turn-1",
        },
      ],
      threadId: "thread-1",
      turnId: "turn-1",
    });

    expect(lifecycle).toMatchObject({
      request: {
        id: "app-server-request:boot:1",
        method: "item/commandExecution/requestApproval",
      },
      resolved: {
        method: "serverRequest/resolved",
        requestId: "app-server-request:boot:1",
        threadId: "thread-1",
      },
      responseCount: 0,
      noRendererResponse: true,
      resolvedBeforeRuntimeTerminal: true,
    });

    const assertions = buildApprovalRequestHostInterruptScenarioAssertions({
      appServerRequestMethods: [APP_SERVER_METHOD_SESSION_TURN_CANCEL],
      approvalRequestResumeTurnStart: {
        inputText: "验证审批请求 hydrate 后允许继续",
        turnId: "turn-1",
      },
      summary: {
        threadId: "thread-1",
        approvalRequestDecisionInputSend: {
          afterFill: { promptVisibleInTextarea: true },
          clicked: { clicked: true },
        },
        approvalRequestDecisionPendingGui: {
          hasSection: true,
          hasApprovalContent: true,
          hasPrompt: true,
          textareaVisible: true,
          textareaDisabled: false,
          singleLine: true,
        },
        approvalRequestDecisionPendingReadModel: {
          hasPendingRequest: true,
          payloadActionType: "tool_confirmation",
          includesRequestId: true,
          includesToolCallId: true,
        },
        approvalRequestHostInterruptRequest: {
          method: APP_SERVER_METHOD_SESSION_TURN_CANCEL,
          params: { threadId: "thread-1", turnId: "turn-1" },
        },
        approvalRequestHostInterruptLifecycle: lifecycle,
        approvalRequestHostInterruptBackend: { actionRespondCount: 0 },
        guiApprovalRequestCancelCompleted: {
          hasPrompt: true,
          textareaVisible: true,
          textareaDisabled: false,
          stopButtonVisible: false,
          completionScope: { assistantText: "" },
        },
        readModelApprovalRequestCancelCanceled: {
          pendingRequestCount: 0,
          latestTurnCanceled: true,
          includesPrompt: true,
          includesToolCallId: true,
          includesCanceled: true,
          includesToolResult: false,
        },
        approvalRequestHostInterruptCanonicalEvents: { ordered: true },
      },
    });

    expect(Object.values(assertions).every(Boolean)).toBe(true);
  });

  it("requires canonical host interrupt events in canceled item order", () => {
    expect(
      summarizeHostInterruptCanonicalEventSequence([
        { type: "action.canceled", payload: {} },
        {
          type: "item.completed",
          payload: { item: { status: "cancelled" } },
        },
        { type: "turn.canceled", payload: {} },
      ]),
    ).toMatchObject({
      actionCanceledIndex: 0,
      itemCompletedCanceledIndex: 1,
      turnCanceledIndex: 2,
      ordered: true,
    });
  });

  it("hides terminal approval details behind a non-interactive history summary", () => {
    const buildAssertions = ({ previewCount, recordCount }) =>
      buildApprovalRequestDecisionScenarioAssertions({
        appServerRequestMethods: [],
        approvalRequestResumeTurnStart: { turnId: "turn-1" },
        backendLedger: [],
        isApprovalRequestCancelScenario: false,
        isApprovalRequestDeclineScenario: true,
        summary: {
          guiApprovalRequestDeclineCompleted: {
            compactTimelinePreviewCount: previewCount,
            approvalRecordShape: {
              recordCount,
              texts: recordCount === 0 ? [] : ["approval record"],
            },
          },
        },
      });

    expect(
      buildAssertions({ previewCount: 1, recordCount: 0 })
        .guiApprovalRequestDeclineHistoricalDetailsHidden,
    ).toBe(true);
    expect(
      buildAssertions({ previewCount: 0, recordCount: 0 })
        .guiApprovalRequestDeclineHistoricalDetailsHidden,
    ).toBe(true);
    expect(
      buildAssertions({ previewCount: 1, recordCount: 1 })
        .guiApprovalRequestDeclineHistoricalDetailsHidden,
    ).toBe(false);
    expect(
      buildAssertions({ previewCount: 0, recordCount: 0 })
        .guiApprovalRequestDeclineHistoricalDetailsHidden,
    ).toBe(true);
  });

  it("keeps retired pending-steer GUI fixtures deleted", () => {
    const content = readSmokeScript();
    const regressionContent = readCurrentFixtureRegressionSmokeScript();
    const retiredFixtureFiles = [
      "scripts/agent-runtime/claw-chat-current-fixture-inputbar-pending-steer.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-gui-actions.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-read-model.mjs",
      "scripts/agent-runtime/claw-chat-current-fixture-pending-steer-assertions.mjs",
    ];
    const retiredScenarios = [
      "inputbar-pending-steer-rich-restore",
      "inputbar-pending-steer-multi-queue",
      "inputbar-pending-steer-pop-front-resume",
    ];

    for (const filePath of retiredFixtureFiles) {
      expect(fs.existsSync(filePath)).toBe(false);
      expect(fixtureSourceFiles).not.toContain(filePath);
    }
    for (const scenario of retiredScenarios) {
      expect(content).not.toContain(scenario);
      expect(regressionContent).not.toContain(scenario);
    }
    expect(content).not.toContain("INPUTBAR_PENDING_STEER");
    expect(content).not.toContain("inputbarPendingSteer");
    expect(content).not.toContain("InputbarPendingSteer");
  });

  it("covers active steer through thread/read and turn/steer without public queue", () => {
    const content = readSmokeScript();
    const regressionContent = readCurrentFixtureRegressionSmokeScript();

    expect(content).toContain(ACTIVE_STEER_SCENARIO);
    expect(content).toContain("runActiveSteerScenario");
    expect(content).toContain('request.method === "turn/steer"');
    expect(content).toContain('request.method === "thread/read"');
    expect(content).toContain("expectedTurnId");
    expect(content).toContain("activeSteerProviderReceivedSecondStep");
    expect(content).toContain("activeSteerSingleTurnInReadModel");
    expect(content).toContain("activeSteerSingleTurnInGui");
    expect(content).toContain("activeSteerNoSecondTurnStart");
    expect(content).toContain("activeSteerNoPublicQueueMethod");
    expect(content).toContain("activeSteerNoQueuedTurnGui");
    expect(content).toContain('APP_SERVER_BACKEND_MODE: "runtime"');
    expect(regressionContent).toContain(
      "Claw active turn steer same identity Electron fixture",
    );
    expect(regressionContent).toContain(`"${ACTIVE_STEER_SCENARIO}"`);
    expect(regressionContent).toContain(
      "claw-chat-current-fixture-active-steer-regression",
    );
  });

  it("binds active steer Gate B identity without an external backend ledger", () => {
    expect(
      resolveGateBExpectedIdentity({
        summary: {
          sessionId: "session-active-steer",
          threadId: "thread-active-steer",
          activeSteer: { activeTurnId: "turn-active-steer" },
        },
        options: { scenario: ACTIVE_STEER_SCENARIO },
        backendLedger: [],
        appServerRequests: [],
      }),
    ).toEqual({
      sessionId: "session-active-steer",
      threadId: "thread-active-steer",
      turnId: "turn-active-steer",
    });
  });

  it("requires one active Turn across steer trace, provider, read model and GUI", () => {
    const summary = {
      guiCompleted: {
        completionScope: {
          runtimeTurnId: "turn-active-steer",
          assistantRuntimeTurnId: "turn-active-steer",
        },
      },
      activeSteer: {
        activeTurnId: "turn-active-steer",
        steerTrace: {
          threadReadBeforeSteer: true,
          steer: {
            expectedTurnId: "turn-active-steer",
            transport: "electron-ipc",
            status: "success",
            inputIsTyped: true,
            inputText: ACTIVE_STEER_INPUT,
          },
        },
        provider: {
          chatRequestCount: 2,
          initialRequestObserved: true,
          steerRequestObserved: true,
        },
        readModel: {
          turnIds: ["turn-active-steer"],
          relevantItemTurnIds: [],
          includesInitialPrompt: true,
          includesSteerInput: true,
          includesFirstText: true,
          includesFinalText: true,
          includesDoneText: true,
          includesTurnSteerSource: false,
        },
        gui: {
          scenarioGroupCount: 4,
          scenarioRuntimeTurnIds: ["turn-active-steer"],
          includesInitialPrompt: true,
          includesSteerInput: true,
          includesFirstText: true,
          includesFinalText: true,
          includesDoneText: true,
          scenarioGroups: [
            {
              runtimeTurnId: "turn-active-steer",
              includesInitialPrompt: true,
              includesSteerInput: false,
              includesFirstText: false,
              includesFinalText: false,
              includesDoneText: false,
            },
            {
              runtimeTurnId: "turn-active-steer",
              includesInitialPrompt: false,
              includesSteerInput: true,
              includesFirstText: false,
              includesFinalText: false,
              includesDoneText: false,
            },
            {
              runtimeTurnId: "turn-active-steer",
              includesInitialPrompt: false,
              includesSteerInput: false,
              includesFirstText: true,
              includesFinalText: false,
              includesDoneText: false,
            },
            {
              runtimeTurnId: "turn-active-steer",
              includesInitialPrompt: false,
              includesSteerInput: false,
              includesFirstText: false,
              includesFinalText: true,
              includesDoneText: true,
            },
          ],
          textareaVisible: true,
          textareaDisabled: false,
          stopButtonVisible: false,
          queuedTurnGuiCount: 0,
        },
        trace: {
          scenarioTurnStartCount: 1,
          initialTurnStartCount: 1,
          steerInputTurnStartCount: 0,
          postBaselineTurnStartCount: 0,
          publicQueueMethodHits: [],
        },
      },
    };

    expect(
      Object.values(buildActiveSteerScenarioAssertions({ summary })),
    ).toEqual(Array(10).fill(true));
    summary.activeSteer.readModel.turnIds.push("turn-forbidden-second");
    expect(
      buildActiveSteerScenarioAssertions({ summary })
        .activeSteerSingleTurnInReadModel,
    ).toBe(false);
  });

  it("streams the active steer first step and records only request markers", async () => {
    const fixture = await startTextProviderFixtureServer({
      activeSteerDelayMs: 10,
    });
    const headers = {
      Authorization: `Bearer ${TEXT_PROVIDER_FIXTURE_API_KEY}`,
      "content-type": "application/json",
    };
    try {
      const firstResponse = await fetch(`${fixture.baseUrl}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          stream: true,
          messages: [{ role: "user", content: ACTIVE_STEER_INITIAL_PROMPT }],
        }),
      });
      expect(await firstResponse.text()).toContain(
        "ACTIVE_STEER_FIRST_VISIBLE",
      );

      const secondResponse = await fetch(
        `${fixture.baseUrl}/chat/completions`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({
            stream: true,
            messages: [
              { role: "user", content: ACTIVE_STEER_INITIAL_PROMPT },
              { role: "user", content: ACTIVE_STEER_INPUT },
            ],
          }),
        },
      );
      const secondBody = await secondResponse.text();
      expect(secondBody).toContain(ACTIVE_STEER_FINAL_TEXT);
      expect(secondBody).toContain(ACTIVE_STEER_DONE_TEXT);

      const requests = fixture
        .requests()
        .filter((request) => request.method === "POST");
      expect(requests).toHaveLength(2);
      expect(requests[0].bodySummary).toMatchObject({
        bodyIncludesActiveSteerInitialPrompt: true,
        bodyIncludesActiveSteerInput: false,
      });
      expect(requests[1].bodySummary).toMatchObject({
        bodyIncludesActiveSteerInitialPrompt: true,
        bodyIncludesActiveSteerInput: true,
      });
      expect(requests.every((request) => !("body" in request))).toBe(true);
    } finally {
      await fixture.close();
    }
  });

  it("returns update_plan first and final text only after the tool result", async () => {
    const fixture = await startTextProviderFixtureServer({
      turnPlanUpdate: true,
    });
    const headers = {
      Authorization: `Bearer ${TEXT_PROVIDER_FIXTURE_API_KEY}`,
      "content-type": "application/json",
    };
    const tools = [
      {
        type: "function",
        function: { name: "update_plan", parameters: { type: "object" } },
      },
    ];
    try {
      const firstResponse = await fetch(`${fixture.baseUrl}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          stream: true,
          tools,
          messages: [{ role: "user", content: TURN_PLAN_UPDATE_PROMPT }],
        }),
      });
      const firstBody = await firstResponse.text();
      expect(firstBody).toContain("update_plan");
      for (const step of TURN_PLAN_UPDATE_STEPS) {
        expect(firstBody).toContain(step.step);
      }
      expect(firstBody).not.toContain(TURN_PLAN_UPDATE_DONE_TEXT);

      const secondResponse = await fetch(
        `${fixture.baseUrl}/chat/completions`,
        {
          method: "POST",
          headers,
          body: JSON.stringify({
            stream: true,
            tools,
            messages: [
              { role: "user", content: TURN_PLAN_UPDATE_PROMPT },
              {
                role: "assistant",
                tool_calls: [
                  {
                    id: "call_turn_plan_update",
                    type: "function",
                    function: { name: "update_plan", arguments: "{}" },
                  },
                ],
              },
              {
                role: "tool",
                tool_call_id: "call_turn_plan_update",
                content: "Plan updated",
              },
            ],
          }),
        },
      );
      expect(await secondResponse.text()).toContain(TURN_PLAN_UPDATE_DONE_TEXT);

      const requests = fixture
        .requests()
        .filter((request) => request.method === "POST");
      expect(requests).toHaveLength(2);
      expect(requests[0].bodySummary).toMatchObject({
        bodyIncludesTurnPlanUpdatePrompt: true,
        bodyIncludesUpdatePlanToolDefinition: true,
        bodyIncludesPlanUpdatedToolResult: false,
      });
      expect(requests[1].bodySummary).toMatchObject({
        bodyIncludesTurnPlanUpdatePrompt: true,
        bodyIncludesUpdatePlanToolDefinition: true,
        bodyIncludesPlanUpdatedToolResult: true,
      });
      expect(requests.every((request) => !("body" in request))).toBe(true);
    } finally {
      await fixture.close();
    }
  });

  it("requires scoped credential for text provider model discovery", async () => {
    const fixture = await startTextProviderFixtureServer();
    try {
      const unauthorized = await fetch(`${fixture.baseUrl}/models`);
      expect(unauthorized.status).toBe(401);
      expect(await unauthorized.json()).toMatchObject({
        error: { type: "authentication_error" },
      });

      const authorized = await fetch(`${fixture.baseUrl}/models`, {
        headers: {
          Authorization: `Bearer ${TEXT_PROVIDER_FIXTURE_API_KEY}`,
        },
      });
      expect(authorized.status).toBe(200);
      expect(await authorized.json()).toMatchObject({
        data: [{ id: expect.any(String) }],
      });
      expect(fixture.requests()).toEqual([
        expect.objectContaining({ authorization: "missing" }),
        expect.objectContaining({ authorization: "present" }),
      ]);
    } finally {
      await fixture.close();
    }
  });

  it("advertises explicit image capabilities for provider discovery", async () => {
    const fixture = await startImageProviderFixtureServer();
    try {
      const response = await fetch(`${fixture.baseUrl}/models`, {
        headers: {
          Authorization: `Bearer ${IMAGE_PROVIDER_FIXTURE_API_KEY}`,
        },
      });

      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({
        data: [
          {
            id: IMAGE_FIXTURE_MODEL,
            task_families: ["image_generation", "image_edit"],
            input_modalities: ["text", "image"],
            output_modalities: ["image"],
            runtime_features: ["images_api"],
          },
        ],
      });
      expect(fixture.requests()).toEqual([
        expect.objectContaining({
          method: "GET",
          url: "/v1/models",
          authorization: "present",
        }),
      ]);
    } finally {
      await fixture.close();
    }
  });

  it("covers Plan mode revisioned thread item and history hydrate in the real Electron fixture", () => {
    const content = readSmokeScript();
    const planHistoryContent = fs.readFileSync(
      "scripts/agent-runtime/claw-chat-current-fixture-plan-history.mjs",
      "utf8",
    );

    expect(content).toContain('options.scenario === "plan"');
    expect(content).toContain("enablePlanModeFromGui");
    expect(content).toContain("waitForGuiPlanCompleted");
    expect(content).toContain("waitForSessionReadPlanCompleted");
    expect(content).toContain("verifyPlanHistoryHydrate");
    expect(content).toContain("verify-plan-history-hydrate-from-sidebar");
    expect(content).toContain("allowPlanDecision: true");
    expect(content).toContain("lime:agent-runtime-sessions-changed");
    expect(content).toContain('reason: "external"');
    expect(content).toContain("hasPlanDecisionPanel");
    expect(content).toContain("hasPlanDecisionTitle");
    expect(content).toContain("readModelPlanThreadItem");
    expect(content).toContain("guiPlanHistoryHydrateCompleted");
    expect(content).toContain("readModelPlanHistoryHydrate");
    expect(content).toContain(
      "summary.guiPlanCompleted?.hasPlanSection === true",
    );
    expect(content).toContain(
      "summary.guiPlanCompleted?.planOwnerHasAllSteps === true",
    );
    expect(content).toContain("planOwnerKindsWithAllSteps");
    expect(content).toContain("planDecisionRevisionBound");
    expect(content).toContain("planOwnerRevisionIds");
    expect(content).toContain("readModelPlanThreadItemRevisioned");
    expect(content).toContain("readModelPlanHistoryHydratePreserved");
    expect(content).toContain("legacyUpdatePlanToolHidden");
    expect(content).toContain("planUiAbsentWithoutProposedPlan");
    expect(content).toContain("guiNoPlanUiWithoutProposedPlan");
    expect(content).toContain("revisionId");
    expect(content).toContain("proposed_plan");
    expect(content).toContain("UpdatePlanTool");
    expect(content).toContain("update_plan");
    expect(content).toContain("legacyUpdatePlanToolVisible");
    expect(planHistoryContent).not.toContain("clearInvokeBuffers");
  });

  it("locks the news WebSearch policy to model-visible auto choice, not keyword required", () => {
    const content = readSmokeScript();

    expect(content).toContain("newsRequestDidNotForceRequiredSearch");
    expect(content).toContain("newsRequestDidNotPassLegacyWebSearchFlag");
    expect(content).toContain('search_mode !== "required"');
    expect(content).toContain('"web_search"');
  });

  it("covers Soul style config through the real Electron fixture without exposing prompt payload", () => {
    const content = readSmokeScript();

    expect(content).toContain("soul-style");
    expect(content).toContain("SOUL_STYLE_SCENARIO");
    expect(content).toContain("--soul-style-profile");
    expect(content).toContain("SOUL_STYLE_FIXTURE_PROFILES");
    expect(content).toContain("createSoulStyleFixtureSelection");
    expect(content).toContain("buildSoulStyleFixtureAssistantText");
    expect(content).toContain("resolveSoulStyleFixtureExpectedTexts");
    expect(content).toContain("soulStyleExpectation");
    expect(content).toContain("enable-soul-style-config");
    expect(content).toContain("soulStyleConfig");
    expect(content).toContain("soulStyleConfigEnabled");
    expect(content).toContain("soulStylePromptReachedBackend");
    expect(content).toContain("soulStyleRuntimeProviderReached");
    expect(content).toContain("soulStylePromptContextCoveredByRuntime");
    expect(content).toContain("soulStylePromptContextMarkers");
    expect(content).toContain("applySoulStyleProviderMarkerSummary");
    expect(content).toContain("summarizeSoulPromptMarkers");
    expect(content).toContain("hasInteractionSoul");
    expect(content).toContain("hasMemorySoulSchema");
    expect(content).toContain("hasProfileId");
    expect(content).toContain("hasStylePack");
    expect(content).toContain("hasToolLifecycleSurfaceContracts");
    expect(content).toContain("closing_suggestion:");
    expect(content).toContain("buildSoulStyleScenarioAssertions");
    expect(content).toContain("soulStyleTranscriptMatchesExpectedProfile");
    expect(content).toContain("SOUL_STYLE_TRANSCRIPT_GOLDENS");
    expect(content).toContain("buildSoulStyleTranscriptGoldenReport");
    expect(content).toContain("soulStyleReadModelCompleted");
    expect(content).toContain("soulStyleGuiCompleted");
    expect(content).toContain("!isSoulStyleScenario");
    expect(content).toContain("options.scenario === SOUL_STYLE_SCENARIO");
    expect(content).toContain("options.scenario !== SOUL_STYLE_SCENARIO");
    expect(content).toContain('APP_SERVER_BACKEND_MODE: "runtime"');
    expect(readGuiActionsScript()).toContain(
      "/^(system_prompt|systemPrompt)$/u",
    );
    expect(readFixtureUtilsScript()).toContain(
      "/^(system_prompt|systemPrompt)$/u",
    );
    expect(readGuiActionsScript()).toContain("[redacted-prompt]");
    expect(readFixtureUtilsScript()).toContain("[redacted-prompt]");
    expect(content).not.toContain("soulStyleContextReachedBackend");
    expect(content).not.toContain("includesMemorySoulPromptContext");
    expect(content).not.toContain("requestContains");
    expect(content).not.toContain("bodyPreview");
    expect(content).not.toContain("contentPreview");
    expect(content).not.toContain("SOUL_STYLE_PROFILE_ID");
    expect(content).not.toContain("SOUL_STYLE_PACK_ID");
    expect(content).not.toContain("SOUL_STYLE_INTENSITY");
  });

  it("locks Soul transcript golden to four different styles over the same facts", () => {
    const report = buildSoulStyleTranscriptGoldenReport();

    expect(report.profiles).toEqual([
      "cheeky_sassy_executor",
      "warm_supportive_companion",
      "cool_confident_operator",
      "calm_professional_partner",
    ]);
    expect(SOUL_STYLE_FIXTURE_PROFILE_IDS).toEqual(report.profiles);
    expect(report.surfaces).toEqual(SOUL_STYLE_TRANSCRIPT_SURFACES);
    expect(SOUL_STYLE_TRANSCRIPT_GOLDENS).toHaveLength(4);

    for (const check of report.checks) {
      expect(check.textCount, `${check.surface} text count`).toBe(4);
      expect(check.uniqueTextCount, `${check.surface} style collapse`).toBe(4);
      expect(check.factSignatureCount, `${check.surface} fact drift`).toBe(1);
      expect(
        check.missingFactsByProfile,
        `${check.surface} missing facts`,
      ).toEqual({});
    }
  });

  it("covers web tool WebSearch/WebFetch rendering in the real Electron fixture", () => {
    const content = readSmokeScript();
    const backendContent = fs.readFileSync(
      "scripts/agent-runtime/claw-chat-current-fixture-backend-tool-skill-events.mjs",
      "utf8",
    );
    const webToolsBranchStart = backendContent.indexOf(
      "if (isWebToolsRenderingPrompt)",
    );
    const webToolsBranchEnd = backendContent.indexOf(
      "if (isMcpStructuredContentPrompt)",
      webToolsBranchStart,
    );
    const webToolsBranch = backendContent.slice(
      webToolsBranchStart,
      webToolsBranchEnd,
    );
    const reasoningBuilderIndex = webToolsBranch.indexOf(
      "const buildWebToolsReasoningItem",
    );
    const reasoningStartedIndex = webToolsBranch.indexOf(
      'type: "item.started"',
      reasoningBuilderIndex,
    );
    const reasoningUpdatedIndex = webToolsBranch.indexOf(
      'type: "item.updated"',
      reasoningStartedIndex,
    );
    const reasoningCompletedIndex = webToolsBranch.lastIndexOf(
      'type: "item.completed"',
    );

    expect(content).toContain("web-tools-rendering");
    expect(content).toContain("WEB_TOOLS_RENDERING_PROMPT");
    expect(content).toContain("WEB_TOOLS_SEARCH_TITLE");
    expect(content).toContain("WEB_TOOLS_SEARCH_URL");
    expect(content).toContain("WEB_TOOLS_MID_THINKING_TEXT");
    expect(content).toContain("WEB_TOOLS_REASONING_ITEM_ID");
    expect(content).toContain("waitForGuiWebToolsRenderingCompleted");
    expect(content).toContain("startupNoteVisible");
    expect(content).toContain("webProcessGroupExpanded");
    expect(content).toContain("webProcessGroupRunning");
    expect(content).toContain("hasSearchSourceSection");
    expect(content).toContain("hasFetchPageSection");
    expect(content).toContain("hasFetchPageUrl");
    expect(content).toContain("hasFetchMarkdownHidden");
    expect(content).toContain("rawJsonEnvelopeVisible");
    expect(content).toContain("guiWebToolsLiveRunningStateCaptured");
    expect(content).toContain("guiWebToolsLiveNoLegacyTextAfterProcess");
    expect(content).toContain("guiWebToolsLiveSourcesVisible");
    expect(content).toContain("guiWebToolsLiveReadPagesVisible");
    expect(content).toContain("guiWebToolsLiveTimelineOrderPreserved");
    expect(content).toContain("guiWebToolsCompletedProcessCompacted");
    expect(content).toContain("guiWebToolsFinalTextVisibleAfterCompletion");
    expect(content).toContain("historicalTimelinePreviewVisible");
    expect(content).toContain(
      "${WEB_TOOLS_BROKEN_MARKDOWN_TEXT}\\n${WEB_TOOLS_RENDERING_DONE_TEXT}\\n",
    );
    expect(content).toContain(
      "text: isWebToolsRenderingPrompt ? followupText : assistantDoneText",
    );
    expect(content).not.toContain("expandAndInspectGuiWebToolsProcess");
    expect(content).not.toContain("fastCompletedBeforeLiveCapture");
    expect(content).not.toContain("guiWebSearchProcessShowsSourcesAfterExpand");
    expect(content).not.toContain(
      "guiWebFetchProcessShowsReadPagesAfterExpand",
    );
    expect(content).not.toContain("expandedDetails?.hasSearchTitle");
    expect(content).toContain("guiWebFetchTransportEnvelopeHidden");
    expect(content).toContain('type: "item.updated"');
    expect(content).toContain('type: "item.completed"');
    expect(webToolsBranchStart).toBeGreaterThan(-1);
    expect(webToolsBranchEnd).toBeGreaterThan(webToolsBranchStart);
    expect(reasoningBuilderIndex).toBeGreaterThan(-1);
    expect(reasoningStartedIndex).toBeGreaterThan(reasoningBuilderIndex);
    expect(reasoningUpdatedIndex).toBeGreaterThan(reasoningStartedIndex);
    expect(reasoningCompletedIndex).toBeGreaterThan(reasoningUpdatedIndex);
    expect(webToolsBranch).not.toContain('type: "reasoning.final"');
    expect(content).not.toContain("WEB_TOOLS_REASONING_FINAL_ID");
    expect(content).not.toContain("WEB_TOOLS_REASONING_FINAL_SIGNATURE");
    expect(content).toContain("hasTimelineOrderPreserved");
    expect(content).toContain("WEB_TOOLS_RENDERING_ASSERTION_KEYS");
    expect(content).toContain("bytes: 2048");
    expect(content).toContain('codeText: "OK"');
    expect(content).toContain("forbiddenTransportFragments");
    expect(content).not.toContain("agent_runtime_");
    expect(content).not.toContain(
      "guiWebToolsReasoningVisibleBeforeFinalAnswer",
    );
  });

  it("covers reasoning-first visibility in a dedicated real Electron fixture", () => {
    const content = readSmokeScript();
    const backendContent = fs.readFileSync(
      "scripts/agent-runtime/claw-chat-current-fixture-backend-script.mjs",
      "utf8",
    );
    const reasoningBranchStart = backendContent.indexOf(
      "if (isReasoningFirstVisiblePrompt)",
    );
    const reasoningBranchEnd = backendContent.indexOf(
      "if (isLiveTailCommitPrompt)",
      reasoningBranchStart,
    );
    const reasoningBranch = backendContent.slice(
      reasoningBranchStart,
      reasoningBranchEnd,
    );

    expect(content).toContain("reasoning-first-visible");
    expect(content).toContain("REASONING_FIRST_VISIBLE_PROMPT");
    expect(content).toContain("REASONING_FIRST_VISIBLE_TEXT");
    expect(content).toContain("REASONING_FIRST_VISIBLE_CONTENT_TEXT");
    expect(content).toContain("REASONING_FIRST_VISIBLE_FINAL_TEXT");
    expect(content).toContain("REASONING_FIRST_VISIBLE_DONE_TEXT");
    expect(content).toContain("waitForGuiReasoningFirstVisibleBeforeAnswer");
    expect(content).toContain("waitForGuiReasoningFirstVisibleCompleted");
    expect(content).toContain("guiReasoningFirstVisibleBeforeAnswer");
    expect(content).toContain("guiReasoningFirstVisibleCompleted");
    expect(content).toContain("historicalReasoningPreviewExpanded");
    expect(content).toContain("reasoningDetailsAvailable");
    expect(content).toContain("reasoningOpenedByClick");
    expect(content).toContain("reasoningContentExpandedAfterCompletion");
    expect(content).toContain("readModelReasoningFirstVisibleCompleted");
    expect(content).toContain("readModelReasoningFirstVisibleItemObserved");
    expect(content).toContain("REASONING_FIRST_VISIBLE_ASSERTION_KEYS");
    expect(content).toContain('type: "reasoning"');
    expect(content).toContain('status: "in_progress"');
    expect(content).toContain('type: "message.delta"');
    expect(content).not.toContain("agent_runtime_");
    expect(reasoningBranchStart).toBeGreaterThan(-1);
    expect(reasoningBranchEnd).toBeGreaterThan(reasoningBranchStart);
    expect(
      reasoningBranch.indexOf('type: "reasoning.started"'),
    ).toBeGreaterThan(-1);
    expect(reasoningBranch.indexOf('type: "reasoning.final"')).toBeGreaterThan(
      reasoningBranch.indexOf('type: "reasoning.started"'),
    );
    expect(reasoningBranch.indexOf('type: "reasoning.ended"')).toBeGreaterThan(
      reasoningBranch.indexOf('type: "reasoning.final"'),
    );
    expect(reasoningBranch).toContain('type: "item.updated"');
    expect(reasoningBranch).toContain(
      'content: ["${REASONING_FIRST_VISIBLE_CONTENT_TEXT}"]',
    );
    expect(reasoningBranch).toContain(
      'canonicalLifecycle: "runtime_message_reasoning.v1"',
    );
    expect(reasoningBranch).not.toContain('type: "item.completed"');
  });

  it("covers live-tail commit in a dedicated real Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("live-tail-commit");
    expect(content).toContain("LIVE_TAIL_COMMIT_PROMPT");
    expect(content).toContain("LIVE_TAIL_COMMIT_FIRST_TEXT");
    expect(content).toContain("LIVE_TAIL_COMMIT_OVERFLOW_MARKER");
    expect(content).toContain("LIVE_TAIL_COMMIT_TABLE_HEADER");
    expect(content).toContain("LIVE_TAIL_COMMIT_TABLE_TAIL");
    expect(content).toContain("LIVE_TAIL_COMMIT_DONE_TEXT");
    expect(content).toContain("runLiveTailCommitScenario");
    expect(content).toContain("wait-gui-live-tail-first-visible-before-commit");
    expect(content).toContain("guiLiveTailFirstVisibleBeforeCommit");
    expect(content).toContain("runningStatusVisible");
    expect(content).toContain("startupNoteVisible");
    expect(content).toContain("overflowCommitted");
    expect(content).toContain("scrollAnchorStable");
    expect(content).toContain("markdownTableRendered");
    expect(content).toContain("readModelLiveTailCommitCompleted");
    expect(content).toContain("liveTailCommitCompleted");
    expect(content).toContain("LIVE_TAIL_COMMIT_ASSERTION_KEYS");
    expect(content).toContain('eventType: "turn.completed"');
    expect(content).toContain('type: "turn.completed"');
    expect(content).toContain("options.scenario !== LIVE_TAIL_COMMIT_SCENARIO");
    expect(content).not.toContain('type: "turn.final_done"');
    expect(content).not.toContain("agent_runtime_");
  });

  it("covers Electron resize/reflow in a dedicated real Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("electron-resize-reflow");
    expect(content).toContain("ELECTRON_RESIZE_REFLOW_SCENARIO");
    expect(content).toContain("runElectronResizeReflowScenario");
    expect(content).toContain(
      "APP_SERVER_METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST",
    );
    expect(content).toContain("request-electron-resize-reflow-files-surface");
    expect(content).toContain("wait-electron-resize-reflow-backend-turn-start");
    expect(content).toContain("capture-electron-resize-reflow-${label}");
    expect(content).toContain("wide: { width: 1280, height: 820 }");
    expect(content).toContain("compact: { width: 880, height: 720 }");
    expect(content).toContain("restored: { width: 1280, height: 820 }");
    expect(content).toContain("tableOverflowHandled");
    expect(content).toContain("noDocumentHorizontalOverflow");
    expect(content).toContain("noTableRightOverlap");
    expect(content).toContain('label === "compact"');
    expect(content).toContain("rightSurfaceExpectedVisibility");
    expect(content).not.toContain(
      'snapshot.label !== "compact" || snapshot.table?.overflowed === true',
    );
    expect(content).toContain("captureResizeScreenshot");
    expect(content).toContain("screenshotCount");
    expect(content).toContain(
      "page.setViewportSize(RESIZE_REFLOW_VIEWPORTS.wide)",
    );
    expect(content).toContain('window.dispatchEvent(new Event("resize"))');
    expect(content).toContain("stableFrameCount < 2");
    expect(content).toContain('scrollRoot.dispatchEvent(new Event("scroll"))');
    expect(content).toContain("workspace-files-surface");
    expect(content).toContain("task-center-files-toggle");
    expect(content).toContain("textRangeRect");
    expect(content).toContain("noTailInputOverlap");
    expect(content).toContain("noTableTailInputOverlap");
    expect(content).toContain("noDoneTextInputOverlap");
    expect(content).toContain("noTurnGroupInputOverlap");
    expect(content).toContain("doneTextRange ?? tableTailRange");
    expect(content).toContain("noMessageRightOverlap");
    expect(content).toContain("guiElectronResizeReflowNoOverlap");
    expect(content).toContain("ELECTRON_RESIZE_REFLOW_ASSERTION_KEYS");
    expect(content).toContain('eventType: "turn.completed"');
    expect(content).toContain('type: "turn.completed"');
    expect(content).not.toContain('type: "turn.final_done"');
    expect(content).not.toContain("agent_runtime_");
  });

  it("covers stale terminal guard in a dedicated real Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("terminal-stale-guard");
    expect(content).toContain("TERMINAL_STALE_GUARD_FIRST_PROMPT");
    expect(content).toContain("TERMINAL_STALE_GUARD_SECOND_PROMPT");
    expect(content).toContain("TERMINAL_STALE_GUARD_FIRST_DONE_TEXT");
    expect(content).toContain("TERMINAL_STALE_GUARD_DONE_TEXT");
    expect(content).toContain("TERMINAL_STALE_GUARD_STALE_DONE_TEXT");
    expect(content).toContain("terminalStaleGuardStaleTerminal");
    expect(content).toContain("wait-gui-terminal-stale-guard-first-completed");
    expect(content).toContain("wait-gui-terminal-stale-guard-second-completed");
    expect(content).toContain("readModelTerminalStaleGuardSecondCompleted");
    expect(content).toContain("terminalStaleGuardStaleTerminalIgnored");
    expect(content).toContain("TERMINAL_STALE_GUARD_ASSERTION_KEYS");
    expect(content).toContain(
      "options.scenario !== TERMINAL_STALE_GUARD_SCENARIO",
    );
    expect(content).toContain('staleEventType: "turn.completed"');
    expect(content).toContain('type: "turn.completed"');
    expect(content).not.toContain("agent_runtime_");
  });

  it("covers failed terminal after visible answer in a dedicated real Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("terminal-failed-after-answer");
    expect(content).toContain("TERMINAL_FAILED_AFTER_ANSWER_PROMPT");
    expect(content).toContain("TERMINAL_FAILED_AFTER_ANSWER_PARTIAL_TEXT");
    expect(content).toContain("TERMINAL_FAILED_AFTER_ANSWER_FAILURE_TEXT");
    expect(content).toContain("terminalFailedAfterAnswerTurnFailed");
    expect(content).toContain("wait-gui-terminal-failed-after-answer-failed");
    expect(content).toContain(
      "wait-read-model-terminal-failed-after-answer-failed",
    );
    expect(content).toContain("waitForSessionReadFailedAfterAnswer");
    expect(content).toContain("readModelTerminalFailedAfterAnswerFailed");
    expect(content).toContain("backendTerminalFailedAfterAnswerRecorded");
    expect(content).toContain("TERMINAL_FAILED_AFTER_ANSWER_ASSERTION_KEYS");
    expect(content).toContain(
      "options.scenario !== TERMINAL_FAILED_AFTER_ANSWER_SCENARIO",
    );
    expect(content).toContain('eventType: "turn.failed"');
    expect(content).toContain('type: "turn.failed"');
    expect(content).not.toContain("agent_runtime_");
  });

  it("covers canceled terminal after visible answer in a dedicated real Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("terminal-canceled-after-answer");
    expect(content).toContain("TERMINAL_CANCELED_AFTER_ANSWER_PROMPT");
    expect(content).toContain("TERMINAL_CANCELED_AFTER_ANSWER_PARTIAL_TEXT");
    expect(content).toContain("TERMINAL_CANCELED_AFTER_ANSWER_CANCELED_TEXT");
    expect(content).toContain("terminalCanceledAfterAnswerTurnCanceled");
    expect(content).toContain(
      "click-stop-after-terminal-canceled-partial-from-gui",
    );
    expect(content).toContain(
      "wait-read-model-terminal-canceled-after-answer-canceled",
    );
    expect(content).toContain("waitForSessionReadCanceled");
    expect(content).toContain("readModelTerminalCanceledAfterAnswerCanceled");
    expect(content).toContain("backendTerminalCanceledAfterAnswerRecorded");
    expect(content).toContain("TERMINAL_CANCELED_AFTER_ANSWER_ASSERTION_KEYS");
    expect(content).toContain(
      "options.scenario !== TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO",
    );
    expect(content).toContain('eventType: "turn.canceled"');
    expect(content).toContain('type: "turn.canceled"');
    expect(content).not.toContain("agent_runtime_");
  });

  it("covers typed error retry success and failure in real Electron fixtures", () => {
    const content = readSmokeScript();
    const regressionContent = readCurrentFixtureRegressionSmokeScript();
    const typedErrorBranchStart = content.indexOf(
      "if (isTypedErrorRetrySuccessPrompt || isTypedErrorRetryFailurePrompt)",
    );
    const typedErrorBranch = content.slice(
      typedErrorBranchStart,
      content.indexOf(
        "if (isImageTaskPresentationPrompt)",
        typedErrorBranchStart,
      ),
    );

    expect(content).toContain("typed-error-retry-success");
    expect(content).toContain("typed-error-retry-failure");
    expect(content).toContain('type: "runtime.error"');
    expect(content).toContain("willRetry: false");
    expect(content).toContain('type: "turn.failed"');
    expect(content).toContain('type: "turn.completed"');
    expect(content).toContain("wait-typed-error-retrying-gui");
    expect(content).toContain("wait-typed-error-awaiting-terminal-gui");
    expect(content).toContain("hasExpectedPhase");
    expect(content).toContain('phase: "retrying"');
    expect(content).toContain('phase: "failed"');
    expect(content).toContain("typedErrorSignalPath");
    expect(content).toContain("readModelTypedErrorPending");
    expect(content).toContain("TYPED_ERROR_RETRY_ASSERTION_KEYS");
    expect(content).toContain("typedErrorRetryFailureNoPrematureTerminal");
    expect(content).toContain("typedErrorRetryIdentityConsistent");
    expect(
      typedErrorBranch.match(/type: "provider\.first_text_delta\.received"/g),
    ).toHaveLength(2);
    expect(regressionContent).toContain(
      "Claw typed error retry success Electron fixture",
    );
    expect(regressionContent).toContain(
      "Claw typed error retry failure Electron fixture",
    );
    expect(regressionContent).toContain(
      "agentStreamRuntimeHandler.typedError.test.ts",
    );
    expect(content).not.toContain("agent_runtime_");
  });

  it("covers MCP structuredContent rendering in the real Electron fixture", () => {
    const content = readSmokeScript();

    expect(content).toContain("mcp-structured-content");
    expect(content).toContain("MCP_STRUCTURED_CONTENT_PROMPT");
    expect(content).toContain("MCP_STRUCTURED_CONTENT_DONE_TEXT");
    expect(content).toContain("MCP_STRUCTURED_CONTENT_TOOL_CALL_ID");
    expect(content).toContain(
      'const MCP_STRUCTURED_CONTENT_TOOL_NAME = "mcp__docs__diagnostic_probe"',
    );
    expect(content).toContain("MCP_STRUCTURED_CONTENT_TOOL_DISPLAY_LABEL");
    expect(content).toContain('"docs / diagnostic probe"');
    expect(content).toContain("MCP_STRUCTURED_CONTENT_ANSWER");
    expect(content).toContain("MCP_STRUCTURED_CONTENT_REFERENCE_ID");
    expect(content).toContain("MCP_STRUCTURED_CONTENT_PROTOCOL_OUTPUT");
    expect(content).toContain("MCP_STRUCTURED_CONTENT_RESULT");
    expect(content).toContain('tool_family: "mcp"');
    expect(content).toContain('mcp_server: "docs"');
    expect(content).toContain('mcp_tool: "diagnostic_probe"');
    expect(content).toContain("buildCanonicalToolItem({");
    expect(content).toContain('status: "completed"');
    expect(content).toContain("structuredContent:");
    expect(content).toContain(
      "text: ${JSON.stringify(MCP_STRUCTURED_CONTENT_PROTOCOL_OUTPUT)}",
    );
    expect(content).toContain("waitForGuiMcpStructuredContentCompleted");
    expect(content).toContain(
      "waitForSessionReadMcpStructuredContentCompleted",
    );
    expect(content).toContain("forbiddenEnvelopeFragments");
    expect(content).toContain("request_metadata");
    expect(content).toContain("mcp_tool_result_projection");
    expect(content).toContain("diagnostics");
    expect(content).toContain("raw_transport_payload");
    expect(content).toContain("doc-hidden-envelope");
    expect(content).toContain("guiMcpStructuredContentEnvelopeHidden");
    expect(content).toContain("guiMcpStructuredContentVisible");
    expect(content).toContain("readModelMcpStructuredContentCompleted");
    expect(content).toContain("readModelMcpStructuredContentObserved");
    expect(content).toContain("MCP_STRUCTURED_CONTENT_ASSERTION_KEYS");
    expect(content).not.toContain("server__diagnostic_probe");
    expect(content).not.toContain("agent_runtime_");
  });

  it("covers current media items in Agent Chat and Workbench preview", () => {
    const content = readSmokeScript();
    const backendContent = fs.readFileSync(
      "scripts/agent-runtime/claw-chat-current-fixture-backend-script.mjs",
      "utf8",
    );

    expect(content).toContain("media-reference");
    expect(content).toContain("MEDIA_REFERENCE_PROMPT");
    expect(content).toContain("验证媒体引用展示");
    expect(content).toContain("CLAW_MEDIA_REFERENCE_FIXTURE_DONE");
    expect(content).toContain("runtimeEnv.mediaReferenceSourcePath");
    expect(backendContent).toContain('type: "item.completed"');
    expect(backendContent).toContain('type: "media"');
    expect(backendContent).toContain("mediaReferenceSourcePath");
    expect(content).toContain("streaming-media-reference-card");
    expect(content).toContain("runMediaReferenceScenario");
    expect(content).toContain("summarizeGuiMediaReferenceSnapshot");
    expect(content).toContain("openGuiMediaReferencePreview");
    expect(content).toContain("openGuiMediaReferenceUnavailableFallback");
    expect(content).toContain("waitForSessionReadMediaReferenceCompleted");
    expect(content).toContain("mediaReferencePromptReachedBackend");
    expect(content).toContain("guiMediaReferenceCardVisible");
    expect(content).toContain("guiMediaReferenceDoesNotExposeInlinePayload");
    expect(content).toContain("guiMediaReferenceUsesSafeSidecarHandle");
    expect(content).toContain("guiMediaReferenceSourcePathNotExposed");
    expect(content).toContain("MEDIA_REFERENCE_SAFE_URI_PREFIX");
    expect(content).toContain("cardTextIncludesSafeHandle");
    expect(content).toContain("sourcePathNotExposed");
    expect(content).toContain("guiMediaReferencePreviewOpened");
    expect(content).toContain("appServerMediaReadV2Succeeded");
    expect(content).toContain("appServerMediaReadThreadScoped");
    expect(content).toContain("guiMediaReferenceUnavailableFallbackVisible");
    expect(content).toContain('APP_SERVER_METHOD_MEDIA_READ = "media/read"');
    expect(content).toContain("mediaReadV2Success");
    expect(content).toContain("mediaReadV2Trace");
    expect(content).toContain("noLegacySessionIdentity");
    expect(content).toContain("makeMediaSidecarUnavailable");
    expect(content).toContain("markdownPreviewVisible");
    expect(content).toContain("readModelMediaReferenceObserved");
    expect(content).toContain("imageViewPaths");
    expect(content).toContain("hasSourceOwner");
    expect(content).toContain("preview-artifact-image");
    const mediaBranchIndex = backendContent.indexOf(
      "if (isMediaReferencePrompt)",
    );
    const mediaCompletedIndex = backendContent.indexOf(
      'type: "item.completed"',
      mediaBranchIndex,
    );
    const turnCompletedIndex = backendContent.indexOf(
      'type: "turn.completed"',
      mediaCompletedIndex,
    );
    expect(mediaBranchIndex).toBeGreaterThan(-1);
    expect(mediaCompletedIndex).toBeGreaterThan(mediaBranchIndex);
    expect(turnCompletedIndex).toBeGreaterThan(mediaCompletedIndex);
    expect(
      backendContent.slice(mediaCompletedIndex, turnCompletedIndex),
    ).toContain('id: "agent-media-reference-1"');
    expect(content).toContain("!isMediaReferenceScenario");
    expect(content).not.toContain("data:image/png;base64,fixture-image-1");
  });

  it("summarizes current imageView media and v2 turn status", () => {
    const sourcePath = "/tmp/fixture-media-reference.png";
    const referenceUri =
      "sidecar://media/output-deadbeef/fixture-media-reference.png";
    expect(
      summarizeReadModelMediaReference(
        {
          thread: {
            turns: [
              {
                status: "completed",
                items: [{ type: "imageView", path: referenceUri }],
              },
            ],
          },
        },
        referenceUri,
        sourcePath,
      ),
    ).toMatchObject({
      latestTurnStatus: "completed",
      imageViewCount: 1,
      matchingImageViewCount: 1,
      hasMediaReference: true,
      hasSourceOwner: true,
      usesSafeSidecarHandle: true,
      sourcePathNotExposed: true,
      contentPartsKeyObserved: false,
      noInlinePayload: true,
    });
  });

  registerImageContentSmokeGuards({
    expect,
    it,
    readSmokeScript,
    removeContentFactoryForbiddenMarkerGuard,
  });
  registerSkillsRuntimeSmokeGuards({
    expect,
    it,
    readSmokeScript,
    readCurrentFixtureRegressionSmokeScript,
    readExpertActionsScript,
    readGuiActionsScript,
  });

  it("covers the Right Surface visual matrix without a model turn", () => {
    const content = readSmokeScript();

    expect(content).toContain("right-surface-visual-matrix");
    expect(content).toContain("runRightSurfaceVisualMatrix");
    expect(content).toContain("create-right-surface-visual-expert-session");
    expect(content).toContain("run-right-surface-visual-matrix");
    expect(content).toContain("workspaceRightSurface/request");
    expect(content).toContain('origin: "runtime"');
    expect(content).not.toContain("fixture:right-surface-visual-matrix");
    expect(content).toContain("workspaceRightSurface/pending/list");
    expect(content).toContain("task-center-files-toggle");
    expect(content).toContain("task-center-object-canvas-toggle");
    expect(content).toContain("task-center-browser-toggle");
    expect(content).toContain("task-center-expert-info-toggle");
    expect(content).toContain("workspace-right-surface-tab-appSurface");
    expect(content).toContain("right-surface-browser.png");
    expect(content).toContain("workspace-right-surface-host");
    expect(content).toContain("workspace-files-surface");
    expect(content).toContain("workspace-object-canvas-surface");
    expect(content).toContain("right-surface-browser-panel");
    expect(content).toContain("summarizeRightSurfaceRequest(requests.browser)");
    expect(content).toMatch(
      /fs\.existsSync\(\s*rightSurfaceVisualCaptures\.browser\.screenshot,?\s*\)/u,
    );
    expect(content).toContain("expert-info-panel");
    expect(content).toContain("workspace-plugin-surface");
    expect(content).toContain("workspace-plugin-surface-tabs");
    expect(content).toContain("workspace-plugin-surface-frame");
    expect(content).toContain("workspace-plugin-surface-viewport");
    expect(content).toContain("plugin-shell-content-factory-app-main");
    expect(content).toContain("plugin-shell-prompt-lab-app");
    expect(content).toContain("webContentsView");
    expect(content).toContain("iframe: false");
    expect(content).toContain("browserView: false");
    expect(content).toContain("rightSurfaceVisualMatrixHostsFillRightSide");
    expect(content).toContain(
      "rightSurfaceVisualMatrixObjectCanvasRailVisible",
    );
    expect(content).toContain("rightSurfaceVisualMatrixBrowserSurfaceVisible");
    expect(content).toContain("rightSurfaceVisualMatrixAppSurfaceVisible");
    expect(content).toContain(
      "rightSurfaceVisualMatrixAppSurfaceMultiInstanceTabs",
    );
    expect(content).toContain(
      "rightSurfaceVisualMatrixPendingConsumeKeepsSurfaceOpen",
    );
    expect(content).toContain("rightSurfaceVisualMatrixDoesNotUseModelTurn");
    expect(content).not.toContain("agent_runtime_");
  });

  it("accepts Article Editor right rail snapshots without canvas-panel fill", () => {
    expect(
      isRightSurfaceSnapshotReady(
        {
          activeSurface: "articleWorkspace",
          hostVisible: true,
          rootVisible: true,
          visibleRootKinds: ["articleWorkspace"],
          geometry: {
            hostFillsCanvasPanel: false,
            rootFillsSurfaceViewport: true,
          },
        },
        "articleWorkspace",
      ),
    ).toBe(true);

    expect(
      isRightSurfaceSnapshotReady(
        {
          activeSurface: "files",
          hostVisible: true,
          rootVisible: true,
          visibleRootKinds: ["files"],
          geometry: {
            hostFillsCanvasPanel: false,
            rootFillsSurfaceViewport: true,
          },
        },
        "files",
      ),
    ).toBe(false);

    expect(
      isRightSurfaceSnapshotReady(
        {
          activeSurface: "files",
          hostVisible: true,
          rootVisible: true,
          visibleRootKinds: ["files"],
          geometry: {
            hostFillsCanvasPanel: false,
            rootFillsSurfaceViewport: true,
          },
        },
        "files",
        "files",
        false,
      ),
    ).toBe(true);
  });

  it("does not use live providers, App Server mock backend, renderer mocks, or legacy commands", () => {
    const content = readSmokeScript();

    expect(content).toContain('LIME_ALLOW_LIVE_PROVIDER_SMOKE: "0"');
    expect(content).toContain('LIME_REAL_API_TEST: "0"');
    expect(content).toContain("liveProviderNotUsed");
    expect(content).toContain("invokeErrorBufferClearedBeforeScenario");
    expect(content).toContain('removeItem("lime_invoke_error_buffer_v1")');
    expect(content).not.toContain("--allow-live-provider");
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain('backendMode: "mock"');
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
    expect(content).not.toContain("explicitMockFallback");
    expect(content).not.toContain("safeInvoke(");
    expect(content).not.toContain("agent_runtime_");
  });
});
