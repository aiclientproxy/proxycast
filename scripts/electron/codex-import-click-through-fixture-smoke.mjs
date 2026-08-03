#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import electronPath from "electron";
import { _electron as electron } from "playwright";
import {
  CONTINUE_ASSISTANT_TEXT,
  CONTINUE_USER_TEXT,
  IMPORTED_ASSISTANT_SUMMARY_TEXT,
  IMPORTED_ASSISTANT_TEXT,
  IMPORTED_CWD,
  IMPORTED_REASONING_TEXT,
  IMPORTED_USER_TEXT,
  IMPORTED_WEB_SEARCH_QUERY,
  LEGACY_CONTINUATION_SENTINEL,
  REQUIRED_BACKEND_METHODS,
  SOURCE_THREAD_ID,
  createClickThroughFixtureRuntimeEnv,
  readBackendLedger,
} from "./lib/local-history-import-click-through-fixture.mjs";
import {
  clearInvokeBuffers,
  clickSidebarImport,
  collectImportedSessionVisualAudit,
  confirmImport,
  inspectImportedHistoricalTimelineSummary,
  inspectEnvironmentPopoverImportBoundary,
  inspectImportedAttachmentPreview,
  inspectImportedFilePreviewArtifacts,
  inspectImportedHistoryBanner,
  inspectImportedMarkdownAndSearchRendering,
  inspectSidebarImportDiscoverability,
  sendFollowUpFromGui,
  summarizeContinuationSnapshot,
  summarizeImportPreviewSnapshot,
  summarizeImportedDetailsSnapshot,
  waitForContinuationVisible,
  waitForImportPreview,
  waitForImportedSessionDetails,
} from "./lib/local-history-import-click-through-gui.mjs";
import {
  APP_SERVER_HANDLE_JSON_LINES_COMMAND,
  assert,
  initializeAppServer,
  invokeAppServerFromPage,
  sanitizeJson,
  sanitizeText,
  sleep,
  waitForRendererReady,
  writeJsonFile,
} from "./lib/local-history-import-smoke-utils.mjs";
import { resolveElectronAppServerRuntimeEnv } from "../lib/electron-app-server-assets.mjs";
import { resolveDevAppServerBinary } from "../lib/electron-dev-sidecar.mjs";

const DEFAULTS = {
  appUrl: "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "codex-import-click-through-fixture",
  ),
  prefix: "codex-import-click-through-fixture",
  timeoutMs: 120_000,
  intervalMs: 250,
  keepTemp: false,
};

const LOG_PREFIX = "[smoke:codex-import-click-through-fixture]";
const RPC_ID_PREFIX = "codex-import-click-through";
const APP_SERVER_CLIENT_INFO = {
  name: "codex-import-click-through-fixture",
  version: "1.0.0",
};

function printHelp() {
  console.log(`
Local History Import Click-through Electron Fixture Smoke

用途:
  启动真实 Electron Desktop Host，从侧边栏点击“本地历史导入”，
  在确认弹窗中预览临时本地历史会话，点击确认导入后进入 Lime 会话页，
  验证导入消息、导入细节还原和 task rail 上下文可见，再通过真实
  输入框发送 follow-up，证明同一导入 session 可继续对话。

边界:
  使用临时 CODEX_HOME fixture，不读取或修改真实 ~/.codex；
  external backend 只作为本脚本一次性可观测执行器，不调用正式模型后端；
  不使用 legacy runtime commands、renderer mock fallback 或 App Server mock
  backend 作为成功证据。

用法:
  node scripts/electron/codex-import-click-through-fixture-smoke.mjs

选项:
  --app-url <url>        可选 renderer dev server，例如 http://127.0.0.1:1420/
  --evidence-dir <path>  证据目录
  --prefix <name>        证据文件前缀
  --timeout-ms <ms>      总超时，默认 120000
  --interval-ms <ms>     轮询间隔，默认 250
  --keep-temp            保留临时目录便于调试
  -h, --help             显示帮助
`);
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };
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
  if (!options.evidenceDir || !options.prefix) {
    throw new Error("--evidence-dir / --prefix 均不能为空");
  }
  return options;
}

function logStage(stage) {
  console.log(`${LOG_PREFIX} stage=${stage}`);
}

function contentTextFromMessage(message) {
  if (typeof message?.text === "string") {
    return message.text.trim();
  }
  return (Array.isArray(message?.content) ? message.content : [])
    .map((part) => {
      if (!part || typeof part !== "object") {
        return "";
      }
      return typeof part.text === "string" ? part.text : "";
    })
    .join("")
    .trim();
}

function normalizeComparablePath(value) {
  return String(value ?? "").replace(/\\/g, "/");
}

function canonicalItemText(value) {
  if (typeof value === "string") {
    return value;
  }
  if (value && typeof value === "object" && typeof value.text === "string") {
    return value.text;
  }
  return "";
}

function canonicalDynamicToolArgument(item, name) {
  const argumentsValue = item?.arguments;
  if (Array.isArray(argumentsValue)) {
    const argument = argumentsValue.find(
      (candidate) => candidate?.name === name,
    );
    return argument?.value;
  }
  if (argumentsValue && typeof argumentsValue === "object") {
    return argumentsValue[name];
  }
  return undefined;
}

function canonicalThreadReadProjection(readResult) {
  const thread = readResult?.thread ?? null;
  const turns = Array.isArray(thread?.turns) ? thread.turns : [];
  const canonicalItems = turns.flatMap((turn) =>
    Array.isArray(turn?.items) ? turn.items : [],
  );
  const messages = canonicalItems.flatMap((item) => {
    if (item?.type === "userMessage") {
      return [
        {
          role: "user",
          content: Array.isArray(item.content) ? item.content : [],
        },
      ];
    }
    if (item?.type === "agentMessage") {
      return [
        {
          role: "assistant",
          content: [{ text: typeof item.text === "string" ? item.text : "" }],
        },
      ];
    }
    return [];
  });

  const items = canonicalItems.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }
    switch (item.type) {
      case "userMessage":
        return [{ ...item, type: "user_message" }];
      case "agentMessage":
        return [{ ...item, type: "agent_message" }];
      case "commandExecution":
        return [{ ...item, type: "command_execution" }];
      case "reasoning":
        return [
          {
            ...item,
            type: "reasoning",
            text: [
              ...(Array.isArray(item.summary) ? item.summary : []),
              ...(Array.isArray(item.content) ? item.content : []),
            ]
              .map(canonicalItemText)
              .filter(Boolean)
              .join("\n\n"),
          },
        ];
      case "fileChange": {
        const metadata = item.metadata ?? {};
        const sourceEventType = String(metadata.sourceEventType ?? "");
        const sourceCallId = String(metadata.sourceCallId ?? "");
        const readOnly =
          metadata.importedReadOnly === true ||
          sourceEventType === "file.read" ||
          sourceCallId.startsWith("call_read_");
        const changes = Array.isArray(item.changes) ? item.changes : [];
        return [
          {
            ...item,
            type: readOnly ? "file_artifact" : "patch",
            paths: changes
              .map((change) => change?.path)
              .filter((filePath) => typeof filePath === "string"),
          },
        ];
      }
      case "dynamicToolCall": {
        const toolName = String(item.tool ?? "").trim().toLowerCase();
        if (toolName !== "web_search") {
          return [{ ...item, type: "dynamic_tool_call" }];
        }
        return [
          {
            ...item,
            type: "web_search",
            query: canonicalDynamicToolArgument(item, "query"),
            action: canonicalDynamicToolArgument(item, "action"),
          },
        ];
      }
      case "webSearch":
        return [{ ...item, type: "web_search" }];
      default:
        return [{ ...item }];
    }
  });

  const importFidelity =
    thread?.extra && typeof thread.extra === "object"
      ? thread.extra.codexImportFidelity
      : null;
  const hasApprovalRecord = Boolean(
    items.some(
      (item) =>
        (item?.type === "approvalRequest" ||
          item?.type === "approval_request") &&
        (item?.requestId === "call_exec" || item?.request_id === "call_exec"),
    ) ||
    (importFidelity &&
      typeof importFidelity === "object" &&
      Number(importFidelity.approvals) > 0),
  );

  return {
    thread,
    turns,
    messages,
    items,
    hasApprovalRecord,
  };
}

function summarizeReadModel(readResult) {
  const projection = canonicalThreadReadProjection(readResult);
  const messages = projection.messages;
  const items = projection.items;
  return {
    sessionId: readResult?.thread?.sessionId ?? readResult?.session?.sessionId ?? null,
    messagesLength: messages.length,
    itemsLength: items.length,
    itemTypes: [...new Set(items.map((item) => item?.type).filter(Boolean))],
    itemToolNames: [
      ...new Set(items.map((item) => item?.tool_name).filter(Boolean)),
    ],
    hasImportedUserMessage: messages.some(
      (message) =>
        message?.role === "user" &&
        contentTextFromMessage(message) === IMPORTED_USER_TEXT,
    ),
    hasImportedAssistantMessage: messages.some(
      (message) =>
        message?.role === "assistant" &&
        contentTextFromMessage(message).includes(IMPORTED_ASSISTANT_TEXT),
    ),
    hasContinueUserMessage: messages.some(
      (message) =>
        message?.role === "user" &&
        contentTextFromMessage(message) === CONTINUE_USER_TEXT,
    ),
    hasContinueAssistantMessage: messages.some(
      (message) =>
        message?.role === "assistant" &&
        contentTextFromMessage(message).includes(CONTINUE_ASSISTANT_TEXT),
    ),
    hasReasoningItem: items.some(
      (item) =>
        item?.type === "reasoning" && item?.text === IMPORTED_REASONING_TEXT,
    ),
    hasCommandItem: items.some(
      (item) =>
        item?.type === "command_execution" &&
        String(item?.command || "").includes("npm test"),
    ),
    hasPatchItem: items.some((item) => {
      const expectedPath = normalizeComparablePath(
        path.join(IMPORTED_CWD, "src", "lib.rs"),
      );
      if (item?.type === "patch" && Array.isArray(item?.paths)) {
        return item.paths.some(
          (itemPath) => normalizeComparablePath(itemPath) === expectedPath,
        );
      }
      return (
        item?.type === "file_artifact" &&
        normalizeComparablePath(item?.path) === expectedPath
      );
    }),
    hasWebSearchItem: items.some((item) => {
      if (item?.type !== "web_search") {
        return false;
      }
      const sourceCallId =
        item?.call_id ??
        item?.metadata?.source_call_id ??
        item?.metadata?.sourceCallId ??
        item?.metadata?.source_provenance?.source_call_id ??
        item?.metadata?.sourceProvenance?.sourceCallId;
      const hasSourceCallId =
        sourceCallId === "call_search" ||
        String(item?.id ?? "").includes("call_search");
      return (
        hasSourceCallId &&
        item?.query === IMPORTED_WEB_SEARCH_QUERY &&
        (String(item?.action || "").includes("search_query") ||
          String(item?.action || "").includes("search") ||
          String(item?.tool_name || "").includes("web_search") ||
          String(item?.output || item?.output_preview || "").includes(
            "search_query",
          ) ||
          String(item?.tool || "").includes("web_search"))
      );
    }),
    hasApprovalItem: projection.hasApprovalRecord,
    hasImportedAttachment: messages.some(
      (message) =>
        message?.role === "user" &&
        Array.isArray(message?.content) &&
        message.content.some(
          (part) =>
            part?.type === "image" || part?.type === "localImage",
        ),
    ),
  };
}

async function waitForImportedReadModel(page, options, threadId) {
  const startedAt = Date.now();
  let latest = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    latest = await invokeAppServerFromPage(
      page,
      "thread/read",
      {
        threadId,
        includeTurns: true,
      },
      { idPrefix: RPC_ID_PREFIX },
    );
    const summary = summarizeReadModel(latest.result);
    if (
      summary.hasImportedUserMessage &&
      summary.hasImportedAssistantMessage &&
      summary.hasContinueUserMessage &&
      summary.hasContinueAssistantMessage &&
      summary.hasReasoningItem &&
      summary.hasCommandItem &&
      summary.hasPatchItem &&
      summary.hasWebSearchItem &&
      summary.hasApprovalItem
    ) {
      return {
        read: latest.result,
        messages: latest.messages,
        summary,
      };
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `导入 session read model 未收敛: ${JSON.stringify(
      sanitizeJson({
        latest: latest?.result ?? null,
        summary: latest?.result ? summarizeReadModel(latest.result) : null,
      }),
    )}`,
  );
}

function extractInvokeTraceMethods(rawTrace) {
  const methods = [];
  let entries = [];
  try {
    const parsed = JSON.parse(rawTrace || "[]");
    entries = Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
  for (const entry of entries) {
    if (entry?.command !== APP_SERVER_HANDLE_JSON_LINES_COMMAND) {
      continue;
    }
    const lines = Array.isArray(entry?.args_preview?.request?.lines)
      ? entry.args_preview.request.lines
      : [];
    for (const line of lines) {
      try {
        const message = JSON.parse(line);
        if (typeof message?.method === "string") {
          methods.push(message.method);
        }
      } catch {
        // ignore non JSON trace line
      }
    }
  }
  return Array.from(new Set(methods));
}

function summarizeBackendLedger(backendLedger, sessionId) {
  const backendTurnStart = backendLedger.find(
    (entry) => entry.kind === "turnStart",
  );
  const backendRuntimeOptions =
    backendTurnStart?.request?.runtimeOptions ??
    backendTurnStart?.request?.runtime_options ??
    null;
  const input = backendTurnStart?.request?.input;
  const backendInputText =
    typeof input?.text === "string"
      ? input.text
      : Array.isArray(input?.parts)
        ? input.parts
            .map((part) =>
              typeof part?.text === "string"
                ? part.text
                : typeof part?.Text?.text === "string"
                  ? part.Text.text
                  : "",
            )
            .join("")
        : null;
  return {
    backendTurnStartSeen: Boolean(backendTurnStart),
    backendSessionId: backendTurnStart?.request?.session?.sessionId ?? null,
    backendTurnId: backendTurnStart?.request?.turn?.turnId ?? null,
    backendInputText,
    backendMetadataImported:
      backendRuntimeOptions?.runtimeRequest?.metadata?.imported === true,
    backendCwd: backendRuntimeOptions?.runtimeRequest?.workingDir ?? null,
    backendSessionMatches:
      backendTurnStart?.request?.session?.sessionId === sessionId,
  };
}

function summarizeImportedFilePreviewArtifacts(previewSummary) {
  if (!previewSummary) {
    return null;
  }
  const summarize = (item) =>
    item
      ? {
          fileName: item.fileName,
          kind: item.kind,
          selector: item.selector,
          workbenchVisible: item.workbenchVisible === true,
          previewPanelVisible: item.previewPanelVisible === true,
          markdownPreviewVisible: item.markdownPreviewVisible === true,
          htmlPreviewVisible: item.htmlPreviewVisible === true,
          codePreviewVisible: item.codePreviewVisible === true,
          emptySurfaceVisible: item.emptySurfaceVisible === true,
          fallbackSurfaceVisible: item.fallbackSurfaceVisible === true,
          fallbackRenderMode: item.fallbackRenderMode || "",
          htmlPreviewSrc: item.htmlPreviewSrc || "",
          bodyTextLength:
            typeof item.bodyText === "string" ? item.bodyText.length : 0,
          workbenchTextLength:
            typeof item.workbenchText === "string"
              ? item.workbenchText.length
              : 0,
        }
      : null;
  return {
    openedAllImportedPreviewArtifacts:
      previewSummary.openedAllImportedPreviewArtifacts === true,
    markdown: summarize(previewSummary.markdown),
    html: summarize(previewSummary.html),
    docx: summarize(previewSummary.docx),
    xlsx: summarize(previewSummary.xlsx),
    pptx: summarize(previewSummary.pptx),
    pdf: summarize(previewSummary.pdf),
  };
}

async function extractClickThroughSummary(
  page,
  readModelSummary,
  backendLedger,
) {
  return await page.evaluate(
    ({
      requiredMethods,
      sourceThreadId,
      importedUserText,
      importedAssistantText,
      importedReasoningText,
      importedWebSearchQuery,
      continueUserText,
      continueAssistantText,
      readModelSummary,
      backendLedger,
    }) => {
      const traceRaw = window.localStorage.getItem(
        "lime_invoke_trace_buffer_v1",
      );
      const errorRaw = window.localStorage.getItem(
        "lime_invoke_error_buffer_v1",
      );
      const bodyText = document.body?.innerText || "";
      return {
        url: window.location.href,
        title: document.title || "",
        sourceThreadId,
        traceRaw,
        errorRaw,
        bodyTextLength: bodyText.length,
        hasDialogPreview: bodyText.includes(
          "将保留工具、命令、补丁、确认与思考记录",
        ),
        hasImportedSourceTaskRail: bodyText.includes(sourceThreadId),
        hasImportedUserMessage: bodyText.includes(importedUserText),
        hasImportedAssistantMessage: bodyText.includes(importedAssistantText),
        hasHistoricalReasoningVisible: bodyText.includes(importedReasoningText),
        hasReasoningItem: readModelSummary?.hasReasoningItem === true,
        hasHistoricalCommandExecutionVisible: bodyText.includes("npm test"),
        hasCommandText: readModelSummary?.hasCommandItem === true,
        hasHistoricalCommandOutput: bodyText.includes("ok"),
        hidesRawImportedCommand:
          !bodyText.includes("Approve imported command") &&
          !bodyText.includes("imported_read_only"),
        hasPatchText:
          bodyText.includes("补丁") ||
          bodyText.includes("已编辑") ||
          bodyText.includes("Patch") ||
          bodyText.includes("patch") ||
          (bodyText.includes("lib.rs") && bodyText.includes("打开文件")),
        hasSearchItem: readModelSummary?.hasWebSearchItem === true,
        hasApprovalItem: readModelSummary?.hasApprovalItem === true,
        hasHistoricalApprovalText:
          bodyText.includes("导入的权限记录") ||
          bodyText.includes("已导入，只读记录") ||
          bodyText.includes("权限记录") ||
          bodyText.includes("审批") ||
          bodyText.includes("确认") ||
          bodyText.includes("权限请求") ||
          bodyText.includes("Approval") ||
          bodyText.includes("approval"),
        hasContinueUserMessage: bodyText.includes(continueUserText),
        hasContinueAssistantMessage: bodyText.includes(continueAssistantText),
        hidesFixtureSentinel: !bodyText.includes(
          "CODEX_IMPORT_CLICK_THROUGH_DONE",
        ),
        readModelSummary,
        backendLedgerLength: Array.isArray(backendLedger)
          ? backendLedger.length
          : 0,
        requiredMethods,
        historicalOperationalDetailsHidden:
          !bodyText.includes(importedReasoningText) &&
          !bodyText.includes("npm test") &&
          !bodyText.includes("ok") &&
          !bodyText.includes(importedWebSearchQuery) &&
          !bodyText.includes("导入的权限记录") &&
          !bodyText.includes("已导入，只读记录") &&
          !bodyText.includes("权限记录") &&
          !bodyText.includes("审批") &&
          !bodyText.includes("确认") &&
          !bodyText.includes("权限请求") &&
          !bodyText.includes("Approval") &&
          !bodyText.includes("approval"),
      };
    },
    {
      requiredMethods: REQUIRED_BACKEND_METHODS,
      sourceThreadId: SOURCE_THREAD_ID,
      importedUserText: IMPORTED_USER_TEXT,
      importedAssistantText: IMPORTED_ASSISTANT_SUMMARY_TEXT,
      importedReasoningText: IMPORTED_REASONING_TEXT,
      importedWebSearchQuery: IMPORTED_WEB_SEARCH_QUERY,
      continueUserText: CONTINUE_USER_TEXT,
      continueAssistantText: CONTINUE_ASSISTANT_TEXT,
      readModelSummary,
      backendLedger,
    },
  );
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });

  const summaryPath = path.join(
    options.evidenceDir,
    `${options.prefix}-summary.json`,
  );
  const rawEvidencePath = path.join(
    options.evidenceDir,
    `${options.prefix}-raw.json`,
  );
  const backendLedgerEvidencePath = path.join(
    options.evidenceDir,
    `${options.prefix}-backend-ledger.json`,
  );
  const screenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}.png`,
  );
  const visualScreenshotDir = path.join(options.evidenceDir, "visual-audit");
  const failureScreenshotPath = path.join(
    options.evidenceDir,
    `${options.prefix}-failure.png`,
  );

  const runtimeEnv = createClickThroughFixtureRuntimeEnv();
  const appServerBinary = resolveDevAppServerBinary({
    env: runtimeEnv.env,
    repoRoot: process.cwd(),
    forceBuild: false,
  });
  const appServerEnv = resolveElectronAppServerRuntimeEnv({
    env: {
      ...runtimeEnv.env,
      APP_SERVER_BIN: appServerBinary,
    },
  });
  const summary = {
    ok: false,
    checkedAt: new Date().toISOString(),
    appUrl: options.appUrl || null,
    sourceThreadId: SOURCE_THREAD_ID,
    tempRoot: options.keepTemp ? runtimeEnv.tempRoot : null,
    electronUserDataDir: options.keepTemp
      ? runtimeEnv.electronUserDataDir
      : null,
    sourceRoot: options.keepTemp ? runtimeEnv.sourceRoot : null,
    rolloutPath: options.keepTemp ? runtimeEnv.rolloutPath : null,
    sessionIndexPath: options.keepTemp ? runtimeEnv.sessionIndexPath : null,
    appServerBinary,
    backendPath: options.keepTemp ? runtimeEnv.backendPath : null,
    backendLedgerPath: options.keepTemp ? runtimeEnv.backendLedgerPath : null,
    backendMode: "external",
    requiredMethods: REQUIRED_BACKEND_METHODS,
    electronPreloadBridge: false,
    sessionId: null,
    clickThroughSummary: null,
    backendSummary: null,
    consoleErrors: [],
    screenshot: null,
    visualAudit: null,
    rawEvidence: rawEvidencePath,
    backendLedgerEvidence: backendLedgerEvidencePath,
    summary: summaryPath,
  };

  let app = null;
  let page = null;
  const consoleErrors = [];
  const rendererSnapshots = [];
  const appServerRequests = [];
  let previewSnapshot = null;
  let previewSummary = null;
  let importedPageSnapshot = null;
  let importedHistoricalTimelineExpansion = null;
  let importedDetailsSnapshot = null;
  let importedDetailsSummary = null;
  let importedMarkdownAndSearchRenderingSummary = null;
  let importedHistoryBannerSummary = null;
  let importedAttachmentPreviewSummary = null;
  let importedFilePreviewArtifactsSummary = null;
  let sidebarImportDiscoverabilitySummary = null;
  let environmentPopoverSummary = null;
  let visualAuditSummary = null;
  let sendClick = null;
  let continuationSnapshot = null;
  let continuationSummary = null;
  let readModel = null;
  let clickThroughSummary = null;

  try {
    logStage("launch-electron");
    app = await electron.launch({
      executablePath: electronPath,
      args: ["--use-mock-keychain", "."],
      cwd: process.cwd(),
      env: {
        ...runtimeEnv.env,
        ...appServerEnv,
        APP_SERVER_BACKEND_MODE: "external",
        APP_SERVER_BACKEND_COMMAND: process.execPath,
        APP_SERVER_BACKEND_ARGS: JSON.stringify([
          runtimeEnv.backendPath,
          runtimeEnv.backendLedgerPath,
        ]),
        APP_SERVER_BACKEND_TIMEOUT_MS: "5000",
        ELECTRON_E2E_USER_DATA_DIR: runtimeEnv.electronUserDataDir,
        LIME_ELECTRON_E2E: "1",
        LIME_ELECTRON_BRAND_DEV_APP: "0",
        LIME_ELECTRON_CLEAR_RENDERER_CACHE: "0",
        LIME_ELECTRON_DEV_HTTP_BRIDGE: "0",
        ...(options.appUrl ? { VITE_DEV_SERVER_URL: options.appUrl } : {}),
      },
      timeout: options.timeoutMs,
    });

    app.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(sanitizeText(message.text()));
      }
    });

    page = await app.firstWindow({ timeout: options.timeoutMs });
    page.setDefaultTimeout(options.timeoutMs);
    await page.setViewportSize({ width: 1440, height: 1000 });

    logStage("wait-renderer");
    const rendererSnapshot = await waitForRendererReady(
      page,
      options,
      (snapshot) => {
        rendererSnapshots.push(sanitizeJson(snapshot));
      },
    );
    summary.electronPreloadBridge =
      rendererSnapshot.electron && rendererSnapshot.hasInvokeBridge;
    await clearInvokeBuffers(page);
    appServerRequests.push({ method: "initialize", source: "script-probe" });
    await initializeAppServer(page, APP_SERVER_CLIENT_INFO, {
      eventMethods: ["agentSession/event"],
    });

    logStage("click-sidebar-import");
    await clickSidebarImport(page, options);

    logStage("wait-import-preview");
    previewSnapshot = await waitForImportPreview(page, options);
    previewSummary = summarizeImportPreviewSnapshot(previewSnapshot);
    assert(
      previewSummary.hidesRawSourceEventNames,
      "导入预览暴露了 source event / payload 内部字段",
    );
    assert(
      previewSummary.hasReadableSourceLabels,
      "导入预览未展示可读的来源行和消息类型",
    );

    logStage("confirm-import");
    importedPageSnapshot = await confirmImport(page, options);
    assert(
      importedPageSnapshot.backgroundResume?.started === true &&
        importedPageSnapshot.backgroundResume?.closed === true &&
        importedPageSnapshot.backgroundResume?.reattached === true &&
        importedPageSnapshot.backgroundResume?.commitRequestCount === 1 &&
        importedPageSnapshot.backgroundResume?.jobReadRequestCount > 0,
      "后台导入未完成关闭弹窗并重新附着同一 job 的闭环",
    );
    summary.backgroundImportResume = sanitizeJson(
      importedPageSnapshot.backgroundResume,
    );

    logStage("inspect-imported-historical-timeline-summary");
    importedHistoricalTimelineExpansion =
      await inspectImportedHistoricalTimelineSummary(page, options);
    assert(
      importedHistoricalTimelineExpansion.historicalSummaryVisible,
      "导入历史未显示 terminal 摘要",
    );
    assert(
      importedHistoricalTimelineExpansion.interactive,
      "导入历史摘要应支持点击展开步骤",
    );
    assert(
      importedHistoricalTimelineExpansion.previewAriaExpanded === "false",
      "导入历史摘要初始状态应保持折叠",
    );
    assert(
      importedHistoricalTimelineExpansion.expanded === true &&
        importedHistoricalTimelineExpansion.timelineVisible,
      "导入历史摘要点击后未显示真实时间线",
    );
    assert(
      importedHistoricalTimelineExpansion.expandableProcessCount > 0 &&
        importedHistoricalTimelineExpansion.expandedProcessDetailsVisible,
      "导入历史步骤不可继续展开",
    );
    assert(
      importedHistoricalTimelineExpansion.previewText.includes("9s"),
      `导入历史过程耗时未使用源事件时间: ${importedHistoricalTimelineExpansion.previewText}`,
    );
    assert(
      importedHistoricalTimelineExpansion.operationalDetailRowCount === 0,
      "导入历史仍挂载运行期工具明细",
    );
    assert(
      importedHistoricalTimelineExpansion.operationalTimelineDetailsCount === 0,
      "导入历史仍挂载 operational details",
    );
    assert(
      importedHistoricalTimelineExpansion.deferredHistoricalPreviewCount === 0,
      "导入历史仍挂载 deferred preview",
    );

    logStage("wait-imported-details");
    importedDetailsSnapshot = await waitForImportedSessionDetails(
      page,
      options,
    );

    logStage("inspect-imported-markdown-and-search-rendering");
    importedMarkdownAndSearchRenderingSummary =
      await inspectImportedMarkdownAndSearchRendering(page, options);

    logStage("inspect-imported-history-banner");
    importedHistoryBannerSummary = await inspectImportedHistoryBanner(
      page,
      options,
    );
    assert(
      importedHistoryBannerSummary.hiddenFromMainTimeline,
      "导入会话主线不应展示本地历史摘要",
    );

    logStage("inspect-imported-attachment-preview");
    importedAttachmentPreviewSummary = await inspectImportedAttachmentPreview(
      page,
      options,
    );

    logStage("inspect-imported-file-preview-artifacts");
    importedFilePreviewArtifactsSummary =
      await inspectImportedFilePreviewArtifacts(page, options);

    logStage("inspect-sidebar-import-discoverability");
    sidebarImportDiscoverabilitySummary =
      await inspectSidebarImportDiscoverability(page, options);
    assert(sidebarImportDiscoverabilitySummary.visible, "侧栏会话入口不可见");
    assert(
      sidebarImportDiscoverabilitySummary.importedEntryVisible,
      "导入会话未出现在侧栏会话入口",
    );
    assert(
      !sidebarImportDiscoverabilitySummary.emptyStateOnly,
      "侧栏导入后仍只显示空态",
    );

    logStage("inspect-environment-popover-import-boundary");
    environmentPopoverSummary = await inspectEnvironmentPopoverImportBoundary(
      page,
      options,
    );

    logStage("send-follow-up");
    sendClick = await sendFollowUpFromGui(page, options);

    logStage("wait-continuation");
    continuationSnapshot = await waitForContinuationVisible(page, options);

    const backendLedger = readBackendLedger(runtimeEnv.backendLedgerPath);
    const backendTurnStart = backendLedger.find(
      (entry) => entry.kind === "turnStart",
    );
    const sessionId = backendTurnStart?.request?.session?.sessionId ?? null;
    const threadId =
      backendTurnStart?.request?.session?.threadId ??
      backendTurnStart?.request?.turn?.threadId ??
      null;
    assert(sessionId, "backend ledger 未记录导入 sessionId");
    assert(threadId, "backend ledger 未记录导入 threadId");
    summary.sessionId = sessionId;
    summary.threadId = threadId;
    appServerRequests.push({
      method: "thread/read",
      source: "script-probe",
      sessionId,
      threadId,
    });
    readModel = await waitForImportedReadModel(page, options, threadId);
    const backendSummary = summarizeBackendLedger(backendLedger, sessionId);
    summary.backendSummary = sanitizeJson(backendSummary);
    assert(
      backendSummary.backendTurnStartSeen,
      "external backend 未收到 turnStart",
    );
    assert(
      backendSummary.backendSessionMatches,
      "backend turnStart 不属于导入 session",
    );
    assert(
      typeof backendSummary.backendTurnId === "string" &&
        backendSummary.backendTurnId.length > 0,
      "backend 未收到有效 turnId",
    );
    assert(
      backendSummary.backendInputText === CONTINUE_USER_TEXT,
      "backend 收到的续聊输入不正确",
    );
    assert(
      backendSummary.backendMetadataImported,
      "续聊 runtimeOptions 未携带 imported session metadata",
    );
    assert(
      backendSummary.backendCwd === IMPORTED_CWD,
      "续聊 runtimeOptions 未继承导入 cwd",
    );

    clickThroughSummary = await extractClickThroughSummary(
      page,
      readModel.summary,
      backendLedger,
    );
    importedDetailsSummary = summarizeImportedDetailsSnapshot(
      importedDetailsSnapshot,
      readModel.summary,
    );
    continuationSummary = summarizeContinuationSnapshot(continuationSnapshot);
    assert(
      continuationSummary.hidesFixtureSentinel,
      "续聊输出暴露了 fixture 哨兵文本",
    );
    const traceMethods = Array.from(
      new Set([
        ...extractInvokeTraceMethods(clickThroughSummary.traceRaw),
        ...appServerRequests.map((request) => request.method),
      ]),
    );
    const missingRequiredMethods = REQUIRED_BACKEND_METHODS.filter(
      (method) => !traceMethods.includes(method),
    );
    assert(
      missingRequiredMethods.length === 0,
      `GUI 点击链路缺少 App Server method trace: ${missingRequiredMethods.join(", ")}`,
    );
    assert(
      importedDetailsSummary.hasImportedUserMessage,
      "页面未显示导入用户消息",
    );
    assert(
      importedDetailsSummary.hasImportedAssistantMessage,
      "页面未显示导入助手消息",
    );
    assert(
      importedDetailsSummary.historicalOperationalDetailsHidden,
      "历史页面铺开了 reasoning / command / search / approval 运行期明细",
    );
    assert(
      readModel.summary.hasReasoningItem,
      "read model 未保留导入 reasoning",
    );
    summary.readModelSummary = sanitizeJson(readModel.summary);
    assert(readModel.summary.hasCommandItem, "read model 未保留导入 command");
    assert(
      readModel.summary.hasImportedAttachment,
      "read model 未保留导入图片附件",
    );
    assert(
      importedDetailsSummary.hidesRawImportedCommand,
      "页面暴露了原始审批命令或导入内部字段",
    );
    assert(importedDetailsSummary.hasPatchText, "页面未显示导入 patch");
    assert(
      readModel.summary.hasWebSearchItem,
      "read model 未保留导入 web search",
    );
    assert(readModel.summary.hasApprovalItem, "read model 未保留导入 approval");
    assert(
      continuationSummary.hasContinueUserMessage,
      "页面未显示续聊用户消息",
    );
    assert(
      continuationSummary.hasContinueAssistantMessage,
      "页面未显示续聊助手消息",
    );

    logStage("collect-visual-audit");
    fs.mkdirSync(visualScreenshotDir, { recursive: true });
    visualAuditSummary = await collectImportedSessionVisualAudit(
      page,
      options,
      visualScreenshotDir,
    );
    await page.setViewportSize({ width: 1440, height: 1000 });

    assert(
      !clickThroughSummary.errorRaw,
      `invoke error buffer 非空: ${clickThroughSummary.errorRaw}`,
    );
    assert(
      consoleErrors.length === 0,
      `观察到 console error: ${consoleErrors.join(" | ")}`,
    );

    summary.clickThroughSummary = sanitizeJson({
      ...clickThroughSummary,
      importedDetailsSummary,
      importedMarkdownAndSearchRenderingSummary,
      importedHistoryBannerSummary,
      importedAttachmentPreviewSummary,
      importedFilePreviewArtifactsSummary:
        summarizeImportedFilePreviewArtifacts(
          importedFilePreviewArtifactsSummary,
        ),
      sidebarImportDiscoverabilitySummary,
      continuationSummary,
      environmentPopoverSummary,
      visualAuditSummary,
      traceMethods,
      missingRequiredMethods,
      traceRaw: undefined,
      errorRaw: clickThroughSummary.errorRaw,
    });

    writeJsonFile(
      rawEvidencePath,
      sanitizeJson({
        rendererSnapshots,
        previewSnapshot,
        previewSummary,
        importedPageSnapshot,
        importedHistoricalTimelineExpansion,
        importedDetailsSnapshot,
        importedDetailsSummary,
        importedMarkdownAndSearchRenderingSummary,
        importedHistoryBannerSummary,
        importedAttachmentPreviewSummary,
        importedFilePreviewArtifactsSummary,
        environmentPopoverSummary,
        visualAuditSummary,
        sendClick,
        continuationSnapshot,
        continuationSummary,
        appServerRequests,
        readModel,
        clickThroughSummary: {
          ...clickThroughSummary,
          importedDetailsSummary,
          importedMarkdownAndSearchRenderingSummary,
          importedHistoryBannerSummary,
          importedAttachmentPreviewSummary,
          importedFilePreviewArtifactsSummary:
            summarizeImportedFilePreviewArtifacts(
              importedFilePreviewArtifactsSummary,
            ),
          sidebarImportDiscoverabilitySummary,
          continuationSummary,
          environmentPopoverSummary,
          visualAuditSummary,
          traceMethods,
        },
      }),
    );
    writeJsonFile(backendLedgerEvidencePath, backendLedger.map(sanitizeJson));

    await page.screenshot({ path: screenshotPath, fullPage: true });
    summary.screenshot = screenshotPath;
    summary.visualAudit = sanitizeJson(visualAuditSummary);
    summary.consoleErrors = consoleErrors;
    summary.ok = true;
    summary.completedAt = new Date().toISOString();
    writeJsonFile(summaryPath, summary);
    console.log(`${LOG_PREFIX} summary=${summaryPath}`);
    console.log(
      `${LOG_PREFIX} session=${sessionId} importedItems=${readModel.summary.itemsLength} messages=${readModel.summary.messagesLength}`,
    );
  } catch (error) {
    summary.error = error instanceof Error ? error.message : String(error);
    summary.consoleErrors = consoleErrors;
    writeJsonFile(
      rawEvidencePath,
      sanitizeJson({
        rendererSnapshots,
        previewSnapshot,
        previewSummary,
        importedPageSnapshot,
        importedHistoricalTimelineExpansion,
        importedDetailsSnapshot,
        importedDetailsSummary,
        importedMarkdownAndSearchRenderingSummary,
        importedHistoryBannerSummary,
        importedAttachmentPreviewSummary,
        importedFilePreviewArtifactsSummary,
        sidebarImportDiscoverabilitySummary,
        environmentPopoverSummary,
        visualAuditSummary,
        sendClick,
        continuationSnapshot,
        continuationSummary,
        readModel,
        clickThroughSummary,
        appServerRequests,
        error: summary.error,
      }),
    );
    try {
      const backendLedger = readBackendLedger(runtimeEnv.backendLedgerPath);
      writeJsonFile(backendLedgerEvidencePath, backendLedger.map(sanitizeJson));
    } catch {
      // ignore failure evidence write
    }
    writeJsonFile(summaryPath, summary);
    if (page) {
      try {
        await page.screenshot({
          path: failureScreenshotPath,
          fullPage: true,
        });
        summary.failureScreenshot = failureScreenshotPath;
        writeJsonFile(summaryPath, summary);
      } catch {
        // ignore screenshot failure
      }
    }
    throw error;
  } finally {
    if (app) {
      await app.close().catch(() => undefined);
    }
    if (!options.keepTemp) {
      try {
        fs.rmSync(runtimeEnv.tempRoot, {
          recursive: true,
          force: true,
          maxRetries: 5,
          retryDelay: 200,
        });
      } catch (cleanupError) {
        console.warn(
          `${LOG_PREFIX} cleanup warning: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`,
        );
      }
    }
  }
}

run().catch((error) => {
  console.error(
    `${LOG_PREFIX} failed: ${
      error instanceof Error ? error.message : String(error)
    }`,
  );
  process.exitCode = 1;
});
