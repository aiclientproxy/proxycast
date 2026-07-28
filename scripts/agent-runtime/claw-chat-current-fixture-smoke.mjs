#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import electronPath from "electron";
import { _electron as electron, chromium } from "playwright";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";
import { ensureElectronFixtureBuild } from "../lib/electron-fixture-build.mjs";
import {
  APP_SERVER_METHOD_SESSION_LIST,
  APPROVAL_REQUEST_CANCEL_SCENARIO,
  APPROVAL_REQUEST_DECLINE_SCENARIO,
  APPROVAL_REQUEST_FULL_ACCESS_PROMPT,
  APPROVAL_REQUEST_FULL_ACCESS_SCENARIO,
  APPROVAL_REQUEST_HOST_INTERRUPT_SCENARIO,
  APPROVAL_REQUEST_RESUME_PROMPT,
  APPROVAL_REQUEST_RESUME_SCENARIO,
  CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO,
  CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO,
  DEFAULTS,
  ELECTRON_RESIZE_REFLOW_SCENARIO,
  FIXTURE_MODEL,
  FIXTURE_PROVIDER,
  GREETING_PROMPT,
  HOME_HOTPATH_GREETING_SCENARIO,
  HOME_HOTPATH_SCENARIO,
  IMAGE_COMMAND_SCENARIO,
  INPUTBAR_RICH_RESTORE_PROMPT,
  INPUTBAR_RICH_RESTORE_SCENARIO,
  LIVE_TAIL_COMMIT_PROMPT,
  LIVE_TAIL_COMMIT_SCENARIO,
  LOG_PREFIX,
  NEWS_PROMPT,
  PLAIN_IMAGE_INTENT_SCENARIO,
  REASONING_FIRST_VISIBLE_PROMPT,
  REASONING_FIRST_VISIBLE_SCENARIO,
  RIGHT_SURFACE_VISUAL_MATRIX_SCENARIO,
  SOUL_STYLE_SCENARIO,
  TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO,
  TERMINAL_FAILED_AFTER_ANSWER_SCENARIO,
  TERMINAL_STALE_GUARD_SCENARIO,
} from "./claw-chat-current-fixture-constants.mjs";
import { buildFixtureAssertionReport } from "./claw-chat-current-fixture-assertions.mjs";
import { resolveGateBExpectedIdentity } from "./claw-chat-current-fixture-assertion-context.mjs";
import { collectGateBGuiEvidence } from "./claw-chat-current-fixture-gate-b-execution-evidence.mjs";
import {
  collectAppServerTraceEvidence,
  collectAgentUiPerformanceTraceEvidence,
  enableClawTraceDebugOverride,
} from "./claw-chat-current-fixture-agent-ui-trace.mjs";
import {
  createTempRuntimeEnv,
  LOCAL_IMAGE_SERVER_API_KEY,
  startImageProviderFixtureServer,
  startTextProviderFixtureServer,
} from "./claw-chat-current-fixture-backend-file.mjs";
import {
  sanitizeBackendLedgerForEvidence,
  summarizeBackendLedger,
} from "./claw-chat-current-fixture-backend-ledger.mjs";
import {
  bindGuiWorkspaceAndModelPreferences,
  clearInvokeBuffers,
  ensureDefaultWorkspace,
  ensureFixtureImageProvider,
  ensureFixtureTextProvider,
  initializeAppServer,
  invokeAppServerFromPage,
  readTraceMessages,
  waitForAppUrlReady,
  waitForRendererReady,
} from "./claw-chat-current-fixture-rpc.mjs";
import {
  mergeInvokeTraceEvidence,
  startInvokeTraceEvidenceCollector,
} from "./claw-chat-current-fixture-invoke-trace.mjs";
import { summarizeHomeHotpathPreTurnTrace } from "./claw-chat-current-fixture-home-hotpath.mjs";
import { executeScenarioFlow } from "./claw-chat-current-fixture-scenario-flow.mjs";
import {
  MEDIA_REFERENCE_PROMPT,
  MEDIA_REFERENCE_SCENARIO,
} from "./claw-chat-current-fixture-media-reference.mjs";
import {
  ACTIVE_STEER_INITIAL_PROMPT,
  ACTIVE_STEER_SCENARIO,
} from "./claw-chat-current-fixture-active-steer.mjs";
import {
  DEFAULT_SOUL_STYLE_FIXTURE_PROFILE_ID,
  createSoulStyleFixtureOverrides,
  createSoulStyleFixtureSelection,
  isSoulStylePromptContextCoveredByRuntime,
  pickLatestSoulStylePromptMarkers,
} from "./claw-chat-current-fixture-soul-style.mjs";
import {
  createFixtureSession,
  navigateGuiToWorkspaceScopedAgent,
  openFixtureSessionFromSidebar,
  waitForGuiSessionVisible,
} from "./claw-chat-current-fixture-session.mjs";
import {
  cleanupTempRoot,
  isIgnorableConsoleError,
  logStage,
  readJsonl,
  sanitizeJson,
  sanitizeText,
  writeJsonFile,
} from "./claw-chat-current-fixture-utils.mjs";

function printHelp() {
  console.log(`
Claw Chat Current Electron Fixture Smoke

用途:
  启动真实 Electron Desktop Host，通过 GUI 输入框发送“${NEWS_PROMPT}”，
  并验证 Frontend -> Electron IPC -> App Server JSON-RPC -> current runtime
  的主链可以完成用户消息、assistant 输出和 read model 收尾。

边界:
  普通聊天场景使用一次性本地 external backend fixture；图片命令场景使用
  APP_SERVER_BACKEND_MODE=runtime 以验证 App Server ImageCommandWorkflow。
  全部场景均不调用正式模型后端，不使用 APP_SERVER_BACKEND_MODE=mock，
  不走 Tauri / legacy runtime command / renderer mock fallback 作为成功证据。

用法:
  node scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs

选项:
  --app-url <url>        可选 renderer dev server，例如 http://127.0.0.1:1420/
  --evidence-dir <path>  证据目录
  --prefix <name>        证据文件前缀
  --run-id <id>          Gate 项目 run-id；也可通过 LIME_GATE_RUN_ID 注入
  --scenario <name>      complete | home-hotpath | home-hotpath-greeting | cancel | cancel-then-continue | inputbar-rich-restore | inputbar-active-steer | plan | goal | soul-style | image-command | plain-image-intent | media-reference | reasoning-first-visible | live-tail-commit | electron-resize-reflow | approval-request-resume | approval-request-decline | approval-request-cancel | approval-request-host-interrupt | approval-request-full-access | terminal-failed-after-answer | terminal-canceled-after-answer | terminal-stale-guard | web-tools-rendering | mcp-structured-content | skills-runtime | expert-plaza-skills-runtime | expert-panel-skills-runtime | right-surface-visual-matrix | content-factory-article-workspace | content-factory-inline-image-article-workspace，默认 complete
  --prompt <text>        仅 home-hotpath 场景可用，覆盖默认新闻输入
  --soul-style-profile <id>   soul-style 场景使用的 profile，默认 ${DEFAULT_SOUL_STYLE_FIXTURE_PROFILE_ID}
  --cdp-port <port>      可选 Electron remote debugging port；传入后通过 CDP renderer 执行 GUI 动作
  --timeout-ms <ms>      总超时，默认 180000
  --interval-ms <ms>     轮询间隔，默认 500
  --keep-temp            保留临时目录便于调试
  -h, --help             显示帮助
`);
}

function parseArgs(argv) {
  const options = {
    ...DEFAULTS,
    cdpPort: null,
    cdpUrl: null,
    promptOverride: null,
    runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
    soulStyleProfileId: DEFAULT_SOUL_STYLE_FIXTURE_PROFILE_ID,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--app-url" && next) {
      options.appUrl = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--evidence-dir" && next) {
      options.evidenceDir = path.resolve(next.trim());
      index += 1;
      continue;
    }
    if (arg === "--prefix" && next) {
      options.prefix = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--run-id" && next) {
      options.runId = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--scenario" && next) {
      options.scenario = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--prompt" && next) {
      options.promptOverride = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--soul-style-profile" && next) {
      options.soulStyleProfileId = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--cdp-port" && next) {
      options.cdpPort = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && next) {
      options.timeoutMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && next) {
      options.intervalMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--keep-temp") {
      options.keepTemp = true;
      continue;
    }
    throw new Error(`未知参数: ${arg}`);
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (
    options.cdpPort !== null &&
    (!Number.isFinite(options.cdpPort) ||
      options.cdpPort < 1 ||
      options.cdpPort > 65535)
  ) {
    throw new Error("--cdp-port 必须是 1 到 65535 的数字");
  }
  if (options.cdpPort !== null) {
    options.cdpUrl = `http://127.0.0.1:${options.cdpPort}`;
  }
  if (!options.evidenceDir || !options.prefix) {
    throw new Error("--evidence-dir / --prefix 均不能为空");
  }
  options.runId ||= `standalone-${options.prefix}`;
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(options.runId)) {
    throw new Error(
      "--run-id / LIME_GATE_RUN_ID 只能包含字母、数字、点、下划线和连字符，且长度不超过 128",
    );
  }
  const allowedScenarios = [
    "complete",
    HOME_HOTPATH_SCENARIO,
    HOME_HOTPATH_GREETING_SCENARIO,
    "cancel",
    "cancel-then-continue",
    INPUTBAR_RICH_RESTORE_SCENARIO,
    ACTIVE_STEER_SCENARIO,
    "plan",
    "goal",
    SOUL_STYLE_SCENARIO,
    IMAGE_COMMAND_SCENARIO,
    PLAIN_IMAGE_INTENT_SCENARIO,
    MEDIA_REFERENCE_SCENARIO,
    REASONING_FIRST_VISIBLE_SCENARIO,
    LIVE_TAIL_COMMIT_SCENARIO,
    ELECTRON_RESIZE_REFLOW_SCENARIO,
    APPROVAL_REQUEST_RESUME_SCENARIO,
    APPROVAL_REQUEST_DECLINE_SCENARIO,
    APPROVAL_REQUEST_CANCEL_SCENARIO,
    APPROVAL_REQUEST_HOST_INTERRUPT_SCENARIO,
    APPROVAL_REQUEST_FULL_ACCESS_SCENARIO,
    TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO,
    TERMINAL_FAILED_AFTER_ANSWER_SCENARIO,
    TERMINAL_STALE_GUARD_SCENARIO,
    "web-tools-rendering",
    "mcp-structured-content",
    "skills-runtime",
    "expert-plaza-skills-runtime",
    "expert-panel-skills-runtime",
    RIGHT_SURFACE_VISUAL_MATRIX_SCENARIO,
    CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO,
    CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO,
  ];
  if (!allowedScenarios.includes(options.scenario)) {
    throw new Error(`--scenario 只能是 ${allowedScenarios.join("、")}`);
  }
  if (options.promptOverride && options.scenario !== HOME_HOTPATH_SCENARIO) {
    throw new Error("--prompt 仅支持 --scenario home-hotpath");
  }
  createSoulStyleFixtureSelection({
    profileId: options.soulStyleProfileId,
  });
  return options;
}

function traceEvidenceHasProviderAndClient(evidence) {
  return (
    evidence?.hasProviderWaitMs === true &&
    evidence?.hasClientLocalOutputMs === true &&
    evidence?.hasFirstVisibleOutputMs === true &&
    evidence?.hasFirstTextDeltaToFirstTextPaintMs === true
  );
}

function applySoulStyleProviderMarkerSummary(summary, textProviderRequests) {
  const soulMarkers = pickLatestSoulStylePromptMarkers(textProviderRequests);
  summary.soulStylePromptContextCoveredByRuntime =
    isSoulStylePromptContextCoveredByRuntime(soulMarkers);
  summary.soulStylePromptContextMarkers = sanitizeJson(soulMarkers ?? null);
}

function isImageWorkflowScenario(scenario) {
  return (
    scenario === IMAGE_COMMAND_SCENARIO ||
    scenario === PLAIN_IMAGE_INTENT_SCENARIO
  );
}

function shouldUseTextProviderFixture(scenario) {
  return (
    isImageWorkflowScenario(scenario) ||
    scenario === SOUL_STYLE_SCENARIO ||
    scenario === INPUTBAR_RICH_RESTORE_SCENARIO ||
    scenario === ACTIVE_STEER_SCENARIO ||
    scenario === CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO ||
    scenario === CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO
  );
}

function scenarioWaitsForExternalBackendCancel(scenario) {
  return (
    scenario === "cancel" ||
    scenario === "cancel-then-continue" ||
    scenario === INPUTBAR_RICH_RESTORE_SCENARIO ||
    scenario === TERMINAL_CANCELED_AFTER_ANSWER_SCENARIO ||
    scenario === APPROVAL_REQUEST_HOST_INTERRUPT_SCENARIO
  );
}

function resolveScenarioBackendEnv(options, runtimeEnv) {
  if (isImageWorkflowScenario(options.scenario)) {
    return {
      APP_SERVER_BACKEND_MODE: "runtime",
      APP_SERVER_BACKEND_COMMAND: "",
      APP_SERVER_BACKEND_ARGS: "",
      APP_SERVER_BACKEND_TIMEOUT_MS: "10000",
    };
  }
  if (
    options.scenario === SOUL_STYLE_SCENARIO ||
    options.scenario === ACTIVE_STEER_SCENARIO ||
    options.scenario === CONTENT_FACTORY_ARTICLE_WORKSPACE_SCENARIO ||
    options.scenario === CONTENT_FACTORY_INLINE_IMAGE_ARTICLE_WORKSPACE_SCENARIO
  ) {
    return {
      APP_SERVER_BACKEND_MODE: "runtime",
      APP_SERVER_BACKEND_COMMAND: "",
      APP_SERVER_BACKEND_ARGS: "",
      APP_SERVER_BACKEND_TIMEOUT_MS: "10000",
    };
  }

  const backendTimeoutMs = scenarioWaitsForExternalBackendCancel(
    options.scenario,
  )
    ? String(Math.max(options.timeoutMs, 130_000))
    : "10000";

  return {
    APP_SERVER_BACKEND_MODE: "external",
    APP_SERVER_BACKEND_COMMAND: process.execPath,
    APP_SERVER_BACKEND_ARGS: JSON.stringify([
      runtimeEnv.backendPath,
      runtimeEnv.backendLedgerPath,
      runtimeEnv.cancelSignalPath,
    ]),
    APP_SERVER_BACKEND_TIMEOUT_MS: backendTimeoutMs,
  };
}

async function fetchJson(url) {
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}`);
  }
  return await response.json();
}

async function waitForCdpEndpoint(options) {
  const startedAt = Date.now();
  let lastError = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const [version, targets] = await Promise.all([
        fetchJson(`${options.cdpUrl}/json/version`),
        fetchJson(`${options.cdpUrl}/json/list`),
      ]);
      return sanitizeJson({
        url: options.cdpUrl,
        waitedMs: Date.now() - startedAt,
        version: {
          browser: version?.Browser ?? null,
          protocolVersion: version?.["Protocol-Version"] ?? null,
          userAgent: version?.["User-Agent"] ?? null,
          webSocketDebuggerUrl: version?.webSocketDebuggerUrl
            ? "present"
            : null,
        },
        targets: Array.isArray(targets)
          ? targets.map((target) => ({
              id: target?.id ?? null,
              type: target?.type ?? null,
              title: target?.title ?? null,
              url: target?.url ?? null,
            }))
          : [],
      });
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      await new Promise((resolve) => setTimeout(resolve, options.intervalMs));
    }
  }
  throw new Error(`Electron CDP endpoint 未就绪: ${lastError}`);
}

async function findElectronCdpPage(browser, options) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    const pages = browser.contexts().flatMap((context) => context.pages());
    for (const page of pages) {
      const snapshot = await page
        .evaluate(() => ({
          url: window.location.href,
          title: document.title || "",
          electron: window.__LIME_ELECTRON__ === true,
          hasInvokeBridge: typeof window.electronAPI?.invoke === "function",
          supportsAppServer:
            typeof window.electronAPI?.supportsCommand === "function" &&
            window.electronAPI.supportsCommand("app_server_handle_json_lines"),
          startupVisible: Boolean(
            document.querySelector("[data-lime-startup-shell]"),
          ),
          appSidebarVisible: Boolean(
            document.querySelector('[data-testid="app-sidebar"]'),
          ),
        }))
        .catch(() => null);
      lastSnapshot = snapshot;
      if (
        snapshot?.electron &&
        snapshot.hasInvokeBridge &&
        snapshot.supportsAppServer
      ) {
        return page;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, options.intervalMs));
  }
  throw new Error(
    `未找到真实 Electron renderer CDP 页签: ${JSON.stringify(
      sanitizeJson(lastSnapshot),
    )}`,
  );
}

async function updateAgentUiPerformanceTraceEvidence(summary, page) {
  const evidence = sanitizeJson(
    await collectAgentUiPerformanceTraceEvidence(page),
  );
  summary.agentUiPerformanceTraceLatest = evidence;
  if (
    !summary.agentUiPerformanceTrace ||
    !traceEvidenceHasProviderAndClient(summary.agentUiPerformanceTrace) ||
    traceEvidenceHasProviderAndClient(evidence)
  ) {
    summary.agentUiPerformanceTrace = evidence;
  }
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  ensureElectronFixtureBuild({
    appUrl: options.appUrl,
    logPrefix: LOG_PREFIX,
    rootDir: process.cwd(),
  });
  const soulStyleSelection =
    options.scenario === SOUL_STYLE_SCENARIO
      ? createSoulStyleFixtureSelection({
          profileId: options.soulStyleProfileId,
        })
      : null;
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    options.prefix + "-summary.json",
  );
  const backendLedgerEvidencePath = path.join(
    options.evidenceDir,
    options.prefix + "-backend-ledger.json",
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    options.prefix + "-chat.png",
  );
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    options.prefix + "-failure.png",
  );

  const runtimeEnv = createTempRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: {
      ...runtimeEnv.env,
      APP_SERVER_BIN: appServerBinary,
    },
  });
  const scenarioBackendEnv = resolveScenarioBackendEnv(options, runtimeEnv);
  const appServerRequests = [];
  const summary = {
    ok: false,
    runId: options.runId,
    scenarioId: "claw-chat-current-fixture",
    scenario: options.scenario,
    prompt:
      options.promptOverride ||
      (options.scenario === INPUTBAR_RICH_RESTORE_SCENARIO
        ? INPUTBAR_RICH_RESTORE_PROMPT
        : options.scenario === ACTIVE_STEER_SCENARIO
          ? ACTIVE_STEER_INITIAL_PROMPT
          : options.scenario === MEDIA_REFERENCE_SCENARIO
            ? MEDIA_REFERENCE_PROMPT
            : options.scenario === REASONING_FIRST_VISIBLE_SCENARIO
              ? REASONING_FIRST_VISIBLE_PROMPT
              : options.scenario === LIVE_TAIL_COMMIT_SCENARIO
                ? LIVE_TAIL_COMMIT_PROMPT
                : options.scenario === APPROVAL_REQUEST_RESUME_SCENARIO
                  ? APPROVAL_REQUEST_RESUME_PROMPT
                  : options.scenario ===
                      APPROVAL_REQUEST_HOST_INTERRUPT_SCENARIO
                    ? APPROVAL_REQUEST_RESUME_PROMPT
                    : options.scenario === APPROVAL_REQUEST_FULL_ACCESS_SCENARIO
                      ? APPROVAL_REQUEST_FULL_ACCESS_PROMPT
                      : options.scenario === HOME_HOTPATH_GREETING_SCENARIO
                        ? GREETING_PROMPT
                        : NEWS_PROMPT),
    sessionId: null,
    threadId: null,
    workspaceId: null,
    workspace: null,
    provider: FIXTURE_PROVIDER,
    model: FIXTURE_MODEL,
    backendMode: scenarioBackendEnv.APP_SERVER_BACKEND_MODE,
    appUrl: options.appUrl || null,
    proofLevel: options.cdpPort
      ? "Gate B CDP controlled fixture"
      : "Gate B controlled fixture",
    cdpUrl: options.cdpUrl,
    cdpEndpoint: null,
    cdpPage: null,
    checkedAt: new Date().toISOString(),
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    electronUserDataDir: options.keepTemp
      ? runtimeEnv.electronUserDataDir
      : null,
    backendPath: options.keepTemp ? runtimeEnv.backendPath : null,
    backendLedgerPath: options.keepTemp ? runtimeEnv.backendLedgerPath : null,
    backendLedger: backendLedgerEvidencePath,
    screenshot: null,
    consoleErrors: [],
    pageErrors: [],
    rendererSnapshot: null,
    initialize: null,
    imageFixtureProvider: null,
    soulStyleExpectation: soulStyleSelection
      ? sanitizeJson(soulStyleSelection)
      : null,
    soulStyleConfig: null,
    soulStylePromptContextCoveredByRuntime: false,
    guiWorkspaceBinding: null,
    sessionCreation: null,
    guiWorkspaceNavigation: null,
    guiSessionVisible: null,
    guiSessionOpened: null,
    inputSend: null,
    guiCompleted: null,
    stopClick: null,
    guiCanceled: null,
    inputbarRichRestoreSkill: null,
    inputbarRichRestoreDraftPrepared: null,
    inputbarRichRestoreInputSend: null,
    inputbarRichRestoreBackendTurnStart: null,
    inputbarRichRestoreStopClick: null,
    inputbarRichRestoreGuiCanceled: null,
    inputbarRichRestoreReadModelCanceled: null,
    activeSteer: null,
    homeHotpath: null,
    continueInputSend: null,
    guiContinueCompleted: null,
    planModeEnabled: null,
    planInputSend: null,
    guiPlanCompleted: null,
    goalModeEnabled: null,
    goalInputSend: null,
    guiGoalCompleted: null,
    reasoningFirstVisibleInputSend: null,
    guiReasoningFirstVisibleBeforeAnswer: null,
    guiReasoningFirstVisibleCompleted: null,
    approvalRequestResumeInputSend: null,
    approvalRequestResumeBackendTurnStart: null,
    approvalRequestResumePendingGui: null,
    approvalRequestResumePendingReadModel: null,
    approvalRequestResumeApproveClick: null,
    approvalRequestResumeRespondActionRequest: null,
    approvalRequestResumeBackendActionRespond: null,
    guiApprovalRequestResumeCompleted: null,
    readModelApprovalRequestResumeCompleted: null,
    approvalRequestResumeSecondAccessModeSet: null,
    approvalRequestResumeSecondInputSend: null,
    approvalRequestResumeSecondBackendTurnStart: null,
    guiApprovalRequestResumeSecondCompleted: null,
    guiApprovalRequestResumeSecondNoApprovalPrompt: null,
    readModelApprovalRequestResumeSecondCompleted: null,
    webToolsRenderingInputSend: null,
    guiWebToolsRenderingCompleted: null,
    skillsRuntimeInputSend: null,
    guiSkillsRuntimeCompleted: null,
    explicitSkillsRuntimeInputSend: null,
    guiExplicitSkillsRuntimeCompleted: null,
    manualEnableSkillsRuntimeTurnStart: null,
    manualEnableSkillsRuntimeSkill: null,
    guiManualEnableSkillsRuntimeCompleted: null,
    readModelCompleted: null,
    readModelCanceled: null,
    readModelContinueCompleted: null,
    readModelPlanCompleted: null,
    readModelGoalCompleted: null,
    readModelReasoningFirstVisibleCompleted: null,
    readModelWebToolsRenderingCompleted: null,
    imageCommandInputSend: null,
    imageCommandBackendTurnStart: null,
    imageCommandTaskCreateRequest: null,
    imageCommandTaskArtifact: null,
    imageCommandTaskArtifactTerminalPatch: null,
    imageCommandTaskArtifactTerminal: null,
    guiImageCommandCompleted: null,
    guiImageCommandTerminal: null,
    guiImageCommandReload: null,
    guiImageCommandRestoredAfterReload: null,
    imageCommandTaskArtifactAfterReload: null,
    readModelImageCommandCompleted: null,
    mediaReferenceInputSend: null,
    guiMediaReferenceCompleted: null,
    guiMediaReferenceSnapshot: null,
    guiMediaReferencePreview: null,
    readModelMediaReferenceCompleted: null,
    readModelSkillsRuntimeCompleted: null,
    readModelExplicitSkillsRuntimeCompleted: null,
    readModelManualEnableSkillsRuntimeCompleted: null,
    rightSurfaceVisualMatrixSessionCreation: null,
    guiRightSurfaceVisualMatrixSessionVisible: null,
    guiRightSurfaceVisualMatrixSessionOpened: null,
    rightSurfaceVisualMatrix: null,
    contentFactoryArticleWorkspaceSessionCreation: null,
    contentFactoryArticleWorkspaceInstalledStateSave: null,
    contentFactoryArticleWorkspaceArtifactWrite: null,
    contentFactoryArticleWorkspaceActionResultArtifactWrite: null,
    contentFactoryArticleWorkspaceWorkerTurnStart: null,
    contentFactoryArticleWorkspaceWorkerHostGenerationFixture: null,
    contentFactoryArticleWorkspaceRightSurfaceRequest: null,
    guiContentFactoryArticleWorkspaceSessionVisible: null,
    guiContentFactoryArticleWorkspaceSessionOpened: null,
    contentFactoryArticleWorkspaceRightSurface: null,
    contentFactoryArticleWorkspaceGui: null,
    contentFactoryArticleWorkspaceEditedDraftUpdate: null,
    contentFactoryArticleWorkspaceEditedDraftReload: null,
    contentFactoryArticleWorkspaceEditedDraftSessionReopened: null,
    contentFactoryArticleWorkspaceEditedDraftArtifactFrame: null,
    contentFactoryArticleWorkspaceEditedDraftRestored: null,
    contentFactoryArticleWorkspaceReadModel: null,
    contentFactoryArticleWorkspaceArtifactRead: null,
    contentFactoryInlineImageEditedDraftUpdate: null,
    contentFactoryInlineImageTaskCreated: null,
    contentFactoryInlineImageTaskSubmittedEvent: null,
    contentFactoryInlineImageTaskCompleted: null,
    contentFactoryInlineImageReload: null,
    contentFactoryInlineImageCanvas: null,
    expertSkillsRuntimeSkill: null,
    expertSkillsRuntimeTurnStart: null,
    expertPlazaSkillsRuntimeCatalog: null,
    expertPlazaSkillsRuntimeLaunch: null,
    guiExpertSkillsRuntimeCompleted: null,
    readModelExpertSkillsRuntimeCompleted: null,
    evidencePackSkillsRuntime: null,
    evidencePackExplicitSkillsRuntime: null,
    evidencePackManualEnableSkillsRuntime: null,
    evidencePackExpertSkillsRuntime: null,
    eventReadProbe: null,
    clawTraceDebugOverride: null,
    agentUiPerformanceTrace: null,
    agentUiPerformanceTraceLatest: null,
    appServerTraceEvidence: null,
    gateBGuiEvidence: null,
    assertions: {},
    summary: summaryPath,
  };

  let app = null;
  let cdpBrowser = null;
  let page = null;
  let imageProviderFixtureServer = null;
  let textProviderFixtureServer = null;
  let invokeTraceCollector = null;
  let collectedInvokeTraceMessages = [];
  let chatProviderPreference = {
    provider: FIXTURE_PROVIDER,
    model: FIXTURE_MODEL,
  };
  const consoleErrors = [];
  const actionableConsoleErrors = [];
  const pageErrors = [];
  const agentDebugLogs = [];
  const pageLifecycleEvents = [];
  const fixtureConfigSoulOverrides =
    createSoulStyleFixtureOverrides(soulStyleSelection);
  const collectConsoleMessage = (message) => {
    const text = sanitizeText(message.text());
    if (text.includes("[AgentDebug]")) {
      agentDebugLogs.push({ type: message.type(), text });
    }
    if (message.type() === "error") {
      consoleErrors.push(text);
      if (!isIgnorableConsoleError(text)) {
        actionableConsoleErrors.push(text);
      }
    }
  };
  const collectPageError = (error) => {
    pageErrors.push(sanitizeText(error));
  };

  try {
    imageProviderFixtureServer = await startImageProviderFixtureServer();
    if (shouldUseTextProviderFixture(options.scenario)) {
      textProviderFixtureServer = await startTextProviderFixtureServer({
        soulStyleExpectation: soulStyleSelection,
        activeSteerDelayMs:
          options.scenario === ACTIVE_STEER_SCENARIO ? 30_000 : 0,
      });
      summary.textProviderFixtureServer = sanitizeJson({
        baseUrl: textProviderFixtureServer.baseUrl,
        requestCount: 0,
      });
    }
    runtimeEnv.writeFixtureConfig?.({
      serverHost: imageProviderFixtureServer.host,
      serverPort: imageProviderFixtureServer.port,
      serverApiKey: LOCAL_IMAGE_SERVER_API_KEY,
      ...fixtureConfigSoulOverrides,
    });
    summary.imageProviderFixtureServer = sanitizeJson({
      baseUrl: imageProviderFixtureServer.baseUrl,
      requestCount: 0,
    });

    if (options.appUrl) {
      logStage("wait-app-url");
      summary.rendererDevServer = sanitizeJson(
        await waitForAppUrlReady(options),
      );
    }

    logStage("launch-electron");
    app = await electron.launch({
      executablePath: electronPath,
      args: [
        ...(options.cdpPort
          ? [`--remote-debugging-port=${options.cdpPort}`]
          : []),
        "--use-mock-keychain",
        ".",
      ],
      cwd: process.cwd(),
      env: {
        ...runtimeEnv.env,
        ...appServerEnv,
        ...scenarioBackendEnv,
        CLAW_CHAT_FIXTURE_SCENARIO: options.scenario,
        ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
        LIME_ALLOW_LIVE_PROVIDER_SMOKE: "0",
        LIME_REAL_API_TEST: "0",
        LIME_ELECTRON_E2E: "1",
        LIME_ELECTRON_BRAND_DEV_APP: "0",
        LIME_ELECTRON_CLEAR_RENDERER_CACHE: "1",
        LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
        ...(options.cdpPort
          ? {
              LIME_ELECTRON_REMOTE_DEBUGGING_PORT: String(options.cdpPort),
            }
          : {}),
        LIME_TRACE_EXPORT_OUTPUT_DIR: path.join(
          runtimeEnv.tempRoot,
          "trace-exports",
        ),
        LIME_SUPPORT_BUNDLE_OUTPUT_DIR: path.join(
          runtimeEnv.tempRoot,
          "support-bundles",
        ),
        ...(options.appUrl ? { VITE_DEV_SERVER_URL: options.appUrl } : {}),
      },
      timeout: options.timeoutMs,
    });

    app.on("console", collectConsoleMessage);
    app.on("close", () => {
      pageLifecycleEvents.push({
        type: "electron-app-close",
        timestamp: new Date().toISOString(),
      });
    });

    page = await app.firstWindow({ timeout: options.timeoutMs });
    page.on("console", collectConsoleMessage);
    page.on("pageerror", collectPageError);
    page.on("close", () => {
      pageLifecycleEvents.push({
        type: "page-close",
        timestamp: new Date().toISOString(),
      });
    });
    page.on("crash", () => {
      pageLifecycleEvents.push({
        type: "page-crash",
        timestamp: new Date().toISOString(),
      });
    });
    page.setDefaultTimeout(options.timeoutMs);
    await page.setViewportSize({ width: 1440, height: 1000 });

    if (options.cdpPort) {
      logStage("wait-cdp-endpoint");
      summary.cdpEndpoint = await waitForCdpEndpoint(options);

      logStage("connect-over-cdp");
      cdpBrowser = await chromium.connectOverCDP(options.cdpUrl);
      page = await findElectronCdpPage(cdpBrowser, options);
      page.on("console", collectConsoleMessage);
      page.on("pageerror", collectPageError);
      page.on("close", () => {
        pageLifecycleEvents.push({
          type: "cdp-page-close",
          timestamp: new Date().toISOString(),
        });
      });
      page.on("crash", () => {
        pageLifecycleEvents.push({
          type: "cdp-page-crash",
          timestamp: new Date().toISOString(),
        });
      });
      page.setDefaultTimeout(options.timeoutMs);
      await page.setViewportSize({ width: 1440, height: 1000 });
      summary.cdpPage = sanitizeJson({
        url: page.url(),
        title: await page.title().catch(() => ""),
      });
    }

    logStage("wait-renderer");
    const rendererSnapshot = await waitForRendererReady(
      page,
      options,
      (snapshot) => {
        summary.rendererSnapshot = sanitizeJson(snapshot);
      },
    );
    summary.rendererSnapshot = sanitizeJson(rendererSnapshot);

    logStage("enable-claw-trace-debug-override");
    summary.clawTraceDebugOverride = sanitizeJson(
      await enableClawTraceDebugOverride(page, options),
    );
    await clearInvokeBuffers(page);
    invokeTraceCollector = startInvokeTraceEvidenceCollector(page);

    logStage("initialize-app-server");
    summary.initialize = sanitizeJson(
      await initializeAppServer(page, appServerRequests),
    );

    logStage("ensure-default-workspace");
    const workspace = await ensureDefaultWorkspace(page, appServerRequests);
    summary.workspaceId = workspace.workspaceId;
    summary.workspace = sanitizeJson(workspace);

    if (options.scenario === SOUL_STYLE_SCENARIO) {
      logStage("enable-soul-style-config");
      runtimeEnv.writeFixtureConfig(fixtureConfigSoulOverrides);
      summary.soulStyleConfig = sanitizeJson(
        await page.evaluate(
          async ({ profileId }) => {
            const currentConfig = await window.electronAPI.invoke("get_config");
            const nextConfig = {
              ...(currentConfig || {}),
              memory: {
                ...(currentConfig?.memory || {}),
                enabled: true,
                soul: {
                  ...(currentConfig?.memory?.soul || {}),
                  enabled: true,
                  style_profile_id: profileId,
                  imported_from: "manual",
                },
              },
            };
            await window.electronAPI.invoke("save_config", {
              config: nextConfig,
            });
            window.localStorage.setItem(
              "lime.app-config.changed-at",
              String(Date.now()),
            );
            window.dispatchEvent(new Event("lime:app-config-changed"));
            return nextConfig.memory.soul;
          },
          {
            profileId: soulStyleSelection.profileId,
          },
        ),
      );
    }

    logStage("ensure-fixture-image-provider");
    const imageFixtureProvider = await ensureFixtureImageProvider(
      page,
      appServerRequests,
      {
        apiHost: imageProviderFixtureServer.baseUrl,
        localImageServerApiKey: LOCAL_IMAGE_SERVER_API_KEY,
        localImageServerHost: imageProviderFixtureServer.host,
        localImageServerPort: imageProviderFixtureServer.port,
      },
    );
    runtimeEnv.writeFixtureConfig({
      serverHost: imageProviderFixtureServer.host,
      serverPort: imageProviderFixtureServer.port,
      serverApiKey: LOCAL_IMAGE_SERVER_API_KEY,
      imageProviderId: imageFixtureProvider.providerId,
      imageModelId: imageFixtureProvider.modelId,
      ...fixtureConfigSoulOverrides,
    });
    summary.imageFixtureProvider = sanitizeJson({
      ...imageFixtureProvider,
      appServerConfigBinding: {
        providerId: imageFixtureProvider.providerId,
        modelId: imageFixtureProvider.modelId,
        configPath: runtimeEnv.configPath,
        macConfigPath: runtimeEnv.macConfigPath,
      },
    });

    if (textProviderFixtureServer) {
      logStage("ensure-fixture-text-provider");
      const textFixtureProvider = await ensureFixtureTextProvider(
        page,
        appServerRequests,
        {
          apiHost: textProviderFixtureServer.baseUrl,
        },
      );
      chatProviderPreference = {
        provider: textFixtureProvider.providerId,
        model: textFixtureProvider.modelId,
      };
      summary.textFixtureProvider = sanitizeJson(textFixtureProvider);
    }

    logStage("bind-gui-workspace-model");
    summary.guiWorkspaceBinding = sanitizeJson(
      await bindGuiWorkspaceAndModelPreferences(
        page,
        workspace.workspaceId,
        chatProviderPreference,
      ),
    );

    logStage("create-fixture-session");
    const sessionCreation = await createFixtureSession(
      page,
      workspace,
      appServerRequests,
      chatProviderPreference,
    );
    const fixtureIdentity = sessionCreation.identity;
    options.sessionId = fixtureIdentity.sessionId;
    options.threadId = fixtureIdentity.threadId;
    summary.sessionId = fixtureIdentity.sessionId;
    summary.threadId = fixtureIdentity.threadId;
    summary.sessionCreation = sanitizeJson({
      sessionId: fixtureIdentity.sessionId,
      threadId: fixtureIdentity.threadId,
    });

    logStage("verify-session-list");
    const sessionList = await invokeAppServerFromPage(
      page,
      APP_SERVER_METHOD_SESSION_LIST,
      { archived: false, cwd: workspace.rootPath, limit: 20 },
      appServerRequests,
    );
    const listedThreads = Array.isArray(sessionList.result?.data)
      ? sessionList.result.data
      : [];
    summary.sessionListVisibility = sanitizeJson({
      count: listedThreads.length,
      containsFixtureSession: listedThreads.some(
        (thread) =>
          thread?.id === fixtureIdentity.threadId &&
          thread?.sessionId === fixtureIdentity.sessionId,
      ),
    });

    logStage("navigate-gui-workspace");
    summary.guiWorkspaceNavigation = sanitizeJson(
      await navigateGuiToWorkspaceScopedAgent(
        page,
        options,
        workspace.workspaceId,
      ),
    );

    logStage("open-session-from-sidebar");
    summary.guiSessionVisible = sanitizeJson(
      await waitForGuiSessionVisible(page, options),
    );
    summary.guiSessionOpened = sanitizeJson(
      await openFixtureSessionFromSidebar(page, options, appServerRequests),
    );

    summary.invokeErrorBufferClearedBeforeScenario = sanitizeJson(
      await page.evaluate(() => {
        window.localStorage.removeItem("lime_invoke_error_buffer_v1");
        return {
          cleared: true,
          clearedAt: new Date().toISOString(),
        };
      }),
    );

    await executeScenarioFlow({
      page,
      options,
      workspace,
      summary,
      appServerRequests,
      runtimeEnv,
      readTextProviderRequests: textProviderFixtureServer
        ? () => textProviderFixtureServer.requests()
        : null,
    });
    collectedInvokeTraceMessages = await invokeTraceCollector.stop();
    invokeTraceCollector = null;

    const backendLedger = readJsonl(runtimeEnv.backendLedgerPath);
    writeJsonFile(
      backendLedgerEvidencePath,
      sanitizeBackendLedgerForEvidence(backendLedger),
    );
    const pageText = await page.evaluate(() => document.body?.innerText || "");
    summary.appServerTraceEvidence = sanitizeJson(
      await collectAppServerTraceEvidence(page, appServerRequests),
    );
    if (summary.appServerTraceEvidence?.traceCount === 0) {
      summary.appServerTraceEvidenceDeferred = {
        reason:
          "当前 Electron/App Server trace export 仍未产生 diagnostics trace rows；主链已由真实 IPC、current JSON-RPC、GUI 与 read model 证据覆盖。",
      };
    }
    const traceRaw = await page.evaluate(() =>
      window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
    );
    const errorRaw = await page.evaluate(() =>
      window.localStorage.getItem("lime_invoke_error_buffer_v1"),
    );
    const traceMessages = mergeInvokeTraceEvidence(
      collectedInvokeTraceMessages,
      readTraceMessages(traceRaw),
    );
    if (
      (options.scenario === HOME_HOTPATH_SCENARIO ||
        options.scenario === HOME_HOTPATH_GREETING_SCENARIO) &&
      summary.homeHotpath
    ) {
      summary.homeHotpath.preTurnTrace = sanitizeJson(
        summarizeHomeHotpathPreTurnTrace(
          traceMessages,
          summary.homeHotpath.inputSend,
          summary.homeHotpath.prompt,
          summary.homeHotpath.preTurnTrace?.turnStartAt ?? null,
        ),
      );
    }
    await updateAgentUiPerformanceTraceEvidence(summary, page);
    if (!traceEvidenceHasProviderAndClient(summary.agentUiPerformanceTrace)) {
      summary.agentUiPerformanceTraceDeferred = {
        reason:
          "v2 direct lifecycle 已记录 firstTextDelta、firstTextPaint 与 clientLocalOutput；providerWait/server emission timestamp 仍归 App Server diagnostics trace，当前未接回 Renderer projection。",
      };
    }
    summary.imageProviderFixtureServer = sanitizeJson({
      baseUrl: imageProviderFixtureServer.baseUrl,
      requestCount: imageProviderFixtureServer.requestCount(),
      requests: imageProviderFixtureServer.requests(),
    });
    if (textProviderFixtureServer) {
      const textProviderRequests = textProviderFixtureServer.requests();
      summary.textProviderFixtureServer = sanitizeJson({
        baseUrl: textProviderFixtureServer.baseUrl,
        requestCount: textProviderFixtureServer.requestCount(),
        requests: textProviderRequests,
      });
      if (options.scenario === SOUL_STYLE_SCENARIO) {
        applySoulStyleProviderMarkerSummary(summary, textProviderRequests);
      }
    }
    summary.pageErrors = pageErrors;
    summary.pageLifecycleEvents = pageLifecycleEvents;
    const gateBExpectedIdentity = resolveGateBExpectedIdentity({
      summary,
      options,
      backendLedger,
      appServerRequests,
    });
    summary.gateBGuiEvidence = sanitizeJson(
      await collectGateBGuiEvidence(page, gateBExpectedIdentity),
    );
    summary.gateBAppServerRequests = sanitizeJson(
      appServerRequests.map(({ method, response, error }) => ({
        method,
        response,
        error,
      })),
    );
    try {
      await page.screenshot({
        path: screenshotPath,
        fullPage: true,
        timeout: 15_000,
      });
      summary.screenshot = screenshotPath;
    } catch (screenshotError) {
      summary.screenshotError = sanitizeText(screenshotError);
    }
    const assertionReport = buildFixtureAssertionReport({
      backendLedger,
      traceMessages,
      appServerRequests,
      rendererSnapshot,
      summary,
      pageText,
      errorRaw,
      actionableConsoleErrors,
      workspace,
      options,
    });

    summary.consoleErrors = consoleErrors;
    summary.actionableConsoleErrors = actionableConsoleErrors;
    summary.agentDebugLogs = agentDebugLogs.slice(-200);
    summary.agentStreamDebugLogs = agentDebugLogs
      .filter((entry) => entry.text.includes("[AgentDebug] AgentStream."))
      .slice(-200);
    summary.pageLifecycleEvents = pageLifecycleEvents;
    summary.appServerRequestMethods = assertionReport.appServerRequestMethods;
    summary.backend = sanitizeJson(assertionReport.backendSummary);
    summary.gateBContract = sanitizeJson(assertionReport.gateBContract);
    summary.assertions = assertionReport.assertions;
    summary.commonAssertions = assertionReport.commonAssertions;
    summary.scenarioAssertions = assertionReport.scenarioAssertions;
    summary.notApplicableAssertions = assertionReport.notApplicableAssertions;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(summaryPath, summary);
    console.log(LOG_PREFIX + " summary=" + summaryPath);
    console.log(LOG_PREFIX + " pass session=" + summary.sessionId);
  } catch (error) {
    try {
      if (page) {
        const traceRaw = await page.evaluate(() =>
          window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
        );
        const errorRaw = await page.evaluate(() =>
          window.localStorage.getItem("lime_invoke_error_buffer_v1"),
        );
        if (invokeTraceCollector) {
          collectedInvokeTraceMessages = await invokeTraceCollector.stop();
          invokeTraceCollector = null;
        }
        summary.invokeTrace = sanitizeJson(
          mergeInvokeTraceEvidence(
            collectedInvokeTraceMessages,
            readTraceMessages(traceRaw),
          ),
        );
        summary.invokeErrors = sanitizeJson(
          (() => {
            try {
              return JSON.parse(errorRaw || "[]");
            } catch {
              return errorRaw;
            }
          })(),
        );
        await updateAgentUiPerformanceTraceEvidence(summary, page);
        summary.appServerTraceEvidence = sanitizeJson(
          await collectAppServerTraceEvidence(page, appServerRequests),
        );
      }
    } catch (traceError) {
      summary.invokeTraceError = sanitizeText(traceError);
    }
    try {
      const backendLedger = readJsonl(runtimeEnv.backendLedgerPath);
      writeJsonFile(
        backendLedgerEvidencePath,
        sanitizeBackendLedgerForEvidence(backendLedger),
      );
      summary.backend = sanitizeJson(summarizeBackendLedger(backendLedger));
    } catch (ledgerError) {
      summary.backendLedgerError = sanitizeText(ledgerError);
    }
    summary.error = sanitizeText(
      error instanceof Error ? error.stack || error.message : String(error),
    );
    summary.consoleErrors = consoleErrors;
    summary.pageErrors = pageErrors;
    summary.actionableConsoleErrors = actionableConsoleErrors;
    if (imageProviderFixtureServer) {
      summary.imageProviderFixtureServer = sanitizeJson({
        baseUrl: imageProviderFixtureServer.baseUrl,
        requestCount: imageProviderFixtureServer.requestCount(),
        requests: imageProviderFixtureServer.requests(),
      });
    }
    if (textProviderFixtureServer) {
      const textProviderRequests = textProviderFixtureServer.requests();
      summary.textProviderFixtureServer = sanitizeJson({
        baseUrl: textProviderFixtureServer.baseUrl,
        requestCount: textProviderFixtureServer.requestCount(),
        requests: textProviderRequests,
      });
      if (options.scenario === SOUL_STYLE_SCENARIO) {
        applySoulStyleProviderMarkerSummary(summary, textProviderRequests);
      }
    }
    summary.agentDebugLogs = agentDebugLogs.slice(-200);
    summary.agentStreamDebugLogs = agentDebugLogs
      .filter((entry) => entry.text.includes("[AgentDebug] AgentStream."))
      .slice(-200);
    summary.pageLifecycleEvents = pageLifecycleEvents;
    try {
      if (page) {
        await page.screenshot({
          path: failureScreenshotPath,
          fullPage: true,
          timeout: 15_000,
        });
        summary.screenshot = failureScreenshotPath;
      }
    } catch (screenshotError) {
      summary.screenshotError = sanitizeText(screenshotError);
    }
    writeJsonFile(summaryPath, summary);
    console.error(summary.error);
    console.error(LOG_PREFIX + " failureSummary=" + summaryPath);
    process.exitCode = 1;
  } finally {
    if (invokeTraceCollector) {
      await invokeTraceCollector.stop().catch(() => undefined);
    }
    if (cdpBrowser) {
      await cdpBrowser.close().catch(() => undefined);
    }
    if (app) {
      await app.close().catch(() => undefined);
    }
    if (imageProviderFixtureServer) {
      await imageProviderFixtureServer.close().catch(() => undefined);
    }
    if (textProviderFixtureServer) {
      await textProviderFixtureServer.close().catch(() => undefined);
    }
    if (!options.keepTemp) {
      cleanupTempRoot(runtimeEnv.tempRoot);
    }
  }
}

await run();
