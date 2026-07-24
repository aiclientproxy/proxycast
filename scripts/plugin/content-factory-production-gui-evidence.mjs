#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

import { readProductionTurnStartTrace } from "../lib/content-factory-production-turn-start-trace.mjs";
import {
  projectAppServerParamsForEvidence,
  readWorkflowJsonlEvents,
  summarizeWorkflowResumeLifecycle,
  workflowResumeBindingsFromTrace,
  workflowResumeEventBinding,
} from "../lib/content-factory-production-workflow-evidence.mjs";

const APP_ID = "content-factory-app";
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const ARTICLE_EDITOR_WORKFLOW_FACT_TEST_IDS = {
  detail: "workspace-article-editor-workflow-detail",
  step: "workspace-article-editor-workflow-step",
  sidePanel: "workspace-article-editor-side-panel",
};
const DEFAULTS = {
  cdpUrl: process.env.LIME_ELECTRON_CDP_URL || "",
  evidenceDir: path.join(
    process.cwd(),
    ".lime",
    "qc",
    "gui-evidence",
    "agent-apps",
  ),
  prefix: "content-factory-production-gui-evidence",
  sessionId: process.env.CONTENT_FACTORY_PRODUCTION_SESSION_ID || "",
  threadId: process.env.CONTENT_FACTORY_PRODUCTION_THREAD_ID || "",
  workflowJsonl: process.env.CONTENT_FACTORY_PRODUCTION_WORKFLOW_JSONL || "",
  turnStartTrace: process.env.CONTENT_FACTORY_PRODUCTION_TURN_START_TRACE || "",
  timeoutMs: 30_000,
};

function printHelp() {
  console.log(`Usage:
  node scripts/plugin/content-factory-production-gui-evidence.mjs [options]

Options:
  --cdp-url <url>          Electron CDP endpoint, or LIME_ELECTRON_CDP_URL.
  --thread-id <id>         Target canonical thread id. If omitted, inferred from the turn/start trace.
  --session-id <id>        Internal session id for evidence/export. If omitted, read from thread/read.
  --turn-start-trace <path>  Real CDP turn-start trace JSON proving Electron IPC -> App Server turn/start.
  --workflow-jsonl <path>  workflow-events.jsonl path copied from the real runtime evidence.
  --evidence-dir <dir>     Evidence output directory.
  --prefix <name>          Evidence filename prefix.
  --timeout-ms <ms>        CDP timeout, default 30000.
  -h, --help               Show help.

This collector only reads a running real Electron page and App Server state.
It does not install plugins, does not run a Provider, and does not create mock
workflow evidence. Missing production facts stay visible as failed assertions.`);
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
    if (arg === "--cdp-url" && next) {
      options.cdpUrl = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--session-id" && next) {
      options.sessionId = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--thread-id" && next) {
      options.threadId = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--workflow-jsonl" && next) {
      options.workflowJsonl = path.resolve(next.trim());
      index += 1;
      continue;
    }
    if (arg === "--turn-start-trace" && next) {
      options.turnStartTrace = path.resolve(next.trim());
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
    throw new Error(`Unsupported argument: ${arg}`);
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 5_000) {
    throw new Error("--timeout-ms must be >= 5000");
  }
  if (!options.cdpUrl) {
    throw new Error("--cdp-url or LIME_ELECTRON_CDP_URL is required");
  }
  return options;
}

function sanitizeText(value) {
  const text = String(value ?? "")
    .replace(/\bsk-[A-Za-z0-9_-]{16,}\b/g, "[redacted-secret]")
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]{16,}/gi, "Bearer [redacted]")
    .replace(
      /((?:api[_-]?key|authorization|password|secret|token)[^=\s:]*\s*[:=]\s*)(["']?)[^\s"',}]+/gi,
      "$1$2[redacted]",
    );
  return text.length > 2_000
    ? `${text.slice(0, 2_000)}... [truncated ${text.length - 2_000} chars]`
    : text;
}

function sanitizeJson(value, depth = 0) {
  if (depth > 8) return "[truncated-depth]";
  if (typeof value === "string") return sanitizeText(value);
  if (
    value === null ||
    value === undefined ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value ?? null;
  }
  if (Array.isArray(value)) {
    return value.slice(0, 80).map((item) => sanitizeJson(item, depth + 1));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 160)
        .map(([key, item]) => [key, sanitizeJson(item, depth + 1)]),
    );
  }
  return sanitizeText(value);
}

function isRecord(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function readString(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function readBool(...values) {
  for (const value of values) {
    if (typeof value === "boolean") return value;
  }
  return false;
}

function parseJson(raw, fallback = null) {
  try {
    return JSON.parse(String(raw ?? ""));
  } catch {
    return fallback;
  }
}

function decodeJsonRpcLines(lines) {
  if (!Array.isArray(lines)) return [];
  return lines
    .map((line) => parseJson(line, null))
    .filter((message) => isRecord(message));
}

function collectTraceEntries(traceRaw) {
  const entries = parseJson(traceRaw, []);
  if (!Array.isArray(entries)) return [];
  return entries.map((entry) => {
    const requestMessages = decodeJsonRpcLines(
      entry?.args_preview?.request?.lines,
    );
    return {
      command: entry?.command ?? null,
      transport: entry?.transport ?? null,
      status: entry?.status ?? null,
      error: entry?.error ? sanitizeText(entry.error) : null,
      appServerRequests: requestMessages
        .filter((message) => typeof message.method === "string")
        .map((message) => ({
          id: message.id ?? null,
          method: message.method,
          params: projectAppServerParamsForEvidence(message.params ?? {}),
        })),
    };
  });
}

function appServerMethods(traceEntries) {
  return Array.from(
    new Set(
      traceEntries
        .flatMap((entry) => entry.appServerRequests || [])
        .map((request) => request.method)
        .filter(Boolean),
    ),
  ).sort();
}

function appServerMethodsWithTurnStartTrace(traceEntries, turnStartTrace) {
  const methods = new Set(appServerMethods(traceEntries));
  if (turnStartTrace?.matched && turnStartTrace?.method) {
    methods.add(turnStartTrace.method);
  }
  return Array.from(methods).sort();
}

function inferThreadId(options, traceEntries, turnStartTrace) {
  if (options.threadId) return options.threadId;
  const requests = traceEntries.flatMap(
    (entry) => entry.appServerRequests || [],
  );
  const preferred = [...requests]
    .reverse()
    .find(
      (request) =>
        (request.method === "thread/read" || request.method === "turn/start") &&
        readString(request.params?.threadId),
    );
  return readString(preferred?.params?.threadId, turnStartTrace?.threadId);
}

function inferSessionId(options, traceEntries, readResult) {
  if (options.sessionId) return options.sessionId;
  const requests = traceEntries.flatMap(
    (entry) => entry.appServerRequests || [],
  );
  const preferred = [...requests]
    .reverse()
    .find(
      (request) =>
        request.method === "workflow/respond" &&
        readString(request.params?.sessionId),
    );
  return readString(
    readResult?.thread?.sessionId,
    preferred?.params?.sessionId,
  );
}

function findContentFactoryInstalledState(listResult) {
  const states = Array.isArray(listResult?.states) ? listResult.states : [];
  return states.find(
    (state) =>
      readString(
        state?.appId,
        state?.app_id,
        state?.identity?.appId,
        state?.identity?.app_id,
      ) === APP_ID,
  );
}

export function summarizeInstalledState(state) {
  const identity = state?.identity || {};
  const setup = state?.setup || {};
  const evidence =
    setup.cloudReleaseEvidence ||
    setup.cloud_release_evidence ||
    state?.cloudReleaseEvidence ||
    state?.cloud_release_evidence ||
    {};
  return {
    appId: readString(
      state?.appId,
      state?.app_id,
      identity.appId,
      identity.app_id,
    ),
    appVersion: readString(
      state?.appVersion,
      state?.app_version,
      identity.appVersion,
      identity.app_version,
      state?.manifest?.version,
    ),
    sourceKind: readString(
      identity.sourceKind,
      identity.source_kind,
      state?.sourceKind,
      state?.source_kind,
    ),
    sourceUriConfigured: Boolean(
      readString(
        identity.sourceUri,
        identity.source_uri,
        state?.sourceUri,
        state?.source_uri,
        evidence.sourceUri,
        evidence.source_uri,
      ),
    ),
    packageHash: readString(
      identity.packageHash,
      identity.package_hash,
      state?.packageHash,
      state?.package_hash,
      evidence.packageHash,
      evidence.package_hash,
    ),
    manifestHash: readString(
      identity.manifestHash,
      identity.manifest_hash,
      state?.manifestHash,
      state?.manifest_hash,
      evidence.manifestHash,
      evidence.manifest_hash,
    ),
    releaseId: readString(
      identity.releaseId,
      identity.release_id,
      state?.releaseId,
      state?.release_id,
    ),
    signatureRef: readString(
      identity.signatureRef,
      identity.signature_ref,
      state?.signatureRef,
      state?.signature_ref,
    ),
    signaturePolicy: readString(
      evidence.signaturePolicy,
      evidence.signature_policy,
      state?.signaturePolicy,
      state?.signature_policy,
    ),
    signatureVerificationStatus: readString(
      evidence.signatureVerificationStatus,
      evidence.signature_verification_status,
      state?.signatureVerificationStatus,
      state?.signature_verification_status,
    ),
    packageVerificationStatus: readString(
      evidence.packageVerificationStatus,
      evidence.package_verification_status,
    ),
    packageHashMatched: readBool(
      evidence.packageHashMatched,
      evidence.package_hash_matched,
    ),
    manifestHashMatched: readBool(
      evidence.manifestHashMatched,
      evidence.manifest_hash_matched,
    ),
    cloudReleaseEvidenceStatus: readString(evidence.status),
  };
}

function articleObjectsFromReadModel(readResult) {
  const detail = readResult?.detail || {};
  const articleWorkspace =
    detail.article_workspace || detail.articleWorkspace || {};
  const objects = Array.isArray(articleWorkspace.objects)
    ? articleWorkspace.objects
    : [];
  return objects.filter((object) => object?.ref?.kind === "articleDraft");
}

function summarizeReadModel(readResult) {
  const article = articleObjectsFromReadModel(readResult)[0] || null;
  const text = readString(article?.source?.documentText);
  const hostManagedGeneration = article?.source?.hostManagedGeneration || {};
  const hostManagedGenerationStatus = readString(hostManagedGeneration.status);
  const generatedArticleMarkerClean = !text.includes(
    "fixturePromptFingerprint:",
  );
  const hostManagedGenerationOutputIds = [
    "article-draft-document",
    ...(Array.isArray(hostManagedGeneration.outputs)
      ? hostManagedGeneration.outputs
          .map((output) => readString(output?.id, output?.outputId))
          .filter(Boolean)
      : []),
  ];
  return {
    articleDraftDocumentPresent: Boolean(text),
    articleDraftDocumentLength: text.length,
    generatedArticleMarkerClean,
    hostManagedGenerationOutputIds: Array.from(
      new Set(hostManagedGenerationOutputIds),
    ),
    hostManagedGenerationStatus: hostManagedGenerationStatus || null,
  };
}

export function inferLiveProviderUsed({ installedState, readModel }) {
  return (
    installedState.sourceKind === "cloud_release" &&
    Boolean(installedState.packageHash) &&
    Boolean(installedState.manifestHash) &&
    Boolean(installedState.releaseId) &&
    Boolean(installedState.signatureRef) &&
    installedState.signaturePolicy === "required" &&
    installedState.signatureVerificationStatus === "verified" &&
    installedState.cloudReleaseEvidenceStatus === "ready" &&
    installedState.packageVerificationStatus === "verified" &&
    installedState.packageHashMatched === true &&
    installedState.manifestHashMatched === true &&
    readModel.hostManagedGenerationStatus === "completed" &&
    readModel.articleDraftDocumentPresent &&
    readModel.generatedArticleMarkerClean === true
  );
}

function summarizeEvidenceExport(evidenceResult) {
  const workflowAudit =
    evidenceResult?.observabilitySummary?.workflow_audit ||
    evidenceResult?.evidencePack?.observabilitySummary?.workflow_audit ||
    null;
  return {
    workflowAudit: workflowAudit
      ? {
          eventCount: workflowAudit.eventCount ?? null,
          metadataOnly: workflowAudit.metadataOnly ?? null,
          rawContentIncluded: workflowAudit.rawContentIncluded ?? null,
          redactionPolicyEventCount:
            workflowAudit.redactionPolicyEventCount ?? null,
          redactionPolicy: workflowAudit.redactionPolicy ?? null,
          source: workflowAudit.source ?? null,
          status: workflowAudit.status ?? null,
        }
      : null,
  };
}

export function summarizeWorkflowFactsDom(snapshot) {
  const workflowDetailCount = Number(snapshot?.workflowDetailCount) || 0;
  const workflowStepCount = Number(snapshot?.workflowStepCount) || 0;
  const sidePanelWorkflowFactMentioned = Boolean(
    snapshot?.sidePanelWorkflowFactMentioned,
  );
  return {
    hidden:
      workflowDetailCount === 0 &&
      workflowStepCount === 0 &&
      !sidePanelWorkflowFactMentioned,
    sidePanelWorkflowFactMentioned,
    workflowDetailCount,
    workflowStepCount,
  };
}

async function inspectArticleEditorWorkflowFacts(page) {
  const snapshot = await page.evaluate((testIds) => {
    const countByTestId = (testId) =>
      document.querySelectorAll(`[data-testid="${testId}"]`).length;
    const sidePanelText = Array.from(
      document.querySelectorAll(`[data-testid="${testIds.sidePanel}"]`),
    )
      .map((element) => element.textContent || "")
      .join("\n");
    return {
      workflowDetailCount: countByTestId(testIds.detail),
      workflowStepCount: countByTestId(testIds.step),
      sidePanelWorkflowFactMentioned:
        /content_article_workflow|workflow-events\.jsonl/i.test(sidePanelText),
    };
  }, ARTICLE_EDITOR_WORKFLOW_FACT_TEST_IDS);
  return summarizeWorkflowFactsDom(snapshot);
}

async function waitForElectronPage(options) {
  const browser = await chromium.connectOverCDP(options.cdpUrl, {
    timeout: options.timeoutMs,
  });
  const pages = browser.contexts().flatMap((context) => context.pages());
  const page =
    pages.find(
      (candidate) =>
        candidate.url().includes("nativeStartup=1") ||
        candidate.url().includes(":1420"),
    ) || pages.find((candidate) => candidate.url().startsWith("http"));
  if (!page) {
    throw new Error("未找到可用 Electron renderer 页签");
  }
  return { browser, page };
}

async function callAppServer(page, method, params = {}) {
  const response = await page.evaluate(
    async ({ method, params }) => {
      const line = JSON.stringify({
        jsonrpc: "2.0",
        id: `${method}:${Date.now()}`,
        method,
        params,
      });
      return await window.electronAPI.invoke("app_server_handle_json_lines", {
        request: { lines: [line] },
      });
    },
    { method, params },
  );
  const envelope = response?.result ?? response;
  const lines = Array.isArray(envelope?.lines) ? envelope.lines : [];
  const messages = decodeJsonRpcLines(lines);
  const result = messages.find((message) => message.result || message.error);
  if (result?.error) {
    throw new Error(`${method} failed: ${JSON.stringify(result.error)}`);
  }
  return result?.result;
}

function buildAssertions({
  runtime,
  traceEntries,
  turnStartTrace,
  installedState,
  readModel,
  workflowJsonlEvents,
  workflowResumeLifecycle,
  evidenceExport,
  workflowFactsDom,
}) {
  const methods = appServerMethodsWithTurnStartTrace(
    traceEntries,
    turnStartTrace,
  );
  const liveProviderUsed = inferLiveProviderUsed({ installedState, readModel });
  const localTurnStartViaElectronIpc = traceEntries.some(
    (entry) =>
      entry.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
      entry.transport === "electron-ipc" &&
      entry.status === "success" &&
      (entry.appServerRequests || []).some(
        (request) => request.method === "turn/start",
      ),
  );
  const tracedTurnStartViaElectronIpc =
    turnStartTrace?.matched === true &&
    turnStartTrace?.threadMatched === true &&
    turnStartTrace?.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND &&
    turnStartTrace?.transport === "electron-ipc" &&
    turnStartTrace?.status === "success" &&
    turnStartTrace?.method === "turn/start";
  return {
    articleDraftDocumentPresent: readModel.articleDraftDocumentPresent,
    contentFactoryArticleWorkspaceWorkflowFactsHidden:
      workflowFactsDom.hidden === true,
    electronBridgePresent: runtime.electron && runtime.hasInvoke,
    liveProviderUsed,
    manifestHashMatched: installedState.manifestHashMatched === true,
    releaseIdPresent: Boolean(installedState.releaseId),
    signatureRefPresent: Boolean(installedState.signatureRef),
    packageHashMatched: installedState.packageHashMatched === true,
    packageHashPresent: Boolean(installedState.packageHash),
    manifestHashPresent: Boolean(installedState.manifestHash),
    packageVerified: installedState.packageVerificationStatus === "verified",
    releaseEvidenceReady: installedState.cloudReleaseEvidenceStatus === "ready",
    signaturePolicyRequired: installedState.signaturePolicy === "required",
    sourceKindCloudRelease: installedState.sourceKind === "cloud_release",
    signatureVerified:
      installedState.signatureVerificationStatus === "verified",
    turnStartViaElectronIpc:
      localTurnStartViaElectronIpc || tracedTurnStartViaElectronIpc,
    workflowJsonlPresent: workflowJsonlEvents.length > 0,
    workflowAuditExported:
      evidenceExport.workflowAudit?.status === "exported" &&
      evidenceExport.workflowAudit?.source === "workflow-events.jsonl" &&
      evidenceExport.workflowAudit?.eventCount > 0,
    workflowAuditMetadataOnly:
      evidenceExport.workflowAudit?.metadataOnly === true,
    workflowAuditRawContentExcluded:
      evidenceExport.workflowAudit?.rawContentIncluded === false,
    workflowAuditRedactionPolicyPresent:
      evidenceExport.workflowAudit?.redactionPolicy ===
        "workflow_audit_metadata_only" &&
      evidenceExport.workflowAudit?.redactionPolicyEventCount > 0,
    workflowResumeLifecyclePresent:
      workflowResumeLifecycle.actionMetadataPresent &&
      workflowResumeLifecycle.auditEventsPresent,
    appServerMethodsSeen: methods,
  };
}

function pageUrlKind(href) {
  const value = String(href || "");
  if (value.includes(":1420") || value.includes("nativeStartup=1")) {
    return "electron_dev_renderer";
  }
  if (value.startsWith("file:")) {
    return "electron_packaged_renderer";
  }
  return value ? "electron_renderer" : "unknown";
}

function statusFromAssertions(assertions, readModel) {
  return assertions.electronBridgePresent &&
    assertions.contentFactoryArticleWorkspaceWorkflowFactsHidden &&
    assertions.turnStartViaElectronIpc &&
    assertions.sourceKindCloudRelease &&
    assertions.signaturePolicyRequired &&
    assertions.signatureVerified &&
    assertions.releaseEvidenceReady &&
    assertions.packageVerified &&
    assertions.packageHashPresent &&
    assertions.packageHashMatched &&
    assertions.manifestHashPresent &&
    assertions.manifestHashMatched &&
    assertions.releaseIdPresent &&
    assertions.signatureRefPresent &&
    assertions.liveProviderUsed &&
    assertions.workflowJsonlPresent &&
    assertions.workflowAuditExported &&
    assertions.workflowAuditMetadataOnly &&
    assertions.workflowAuditRawContentExcluded &&
    assertions.workflowAuditRedactionPolicyPresent &&
    assertions.workflowResumeLifecyclePresent &&
    readModel.hostManagedGenerationStatus === "completed" &&
    assertions.articleDraftDocumentPresent
    ? "passed"
    : "failed";
}

function missingAssertionKeys(assertions) {
  return Object.entries(assertions)
    .filter(([, value]) => value === false)
    .map(([key]) => key);
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function run() {
  const options = parseArgs(process.argv.slice(2));
  fs.mkdirSync(options.evidenceDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputPath = path.join(
    options.evidenceDir,
    `${options.prefix}-${stamp}.json`,
  );
  const tracePath = path.join(
    options.evidenceDir,
    `${options.prefix}-${stamp}.trace.json`,
  );

  const connected = await waitForElectronPage(options);
  const browser = connected.browser;
  try {
    const page = connected.page;
    const runtime = await page.evaluate(() => ({
      electron: window.__LIME_ELECTRON__ === true,
      hasInvoke: typeof window.electronAPI?.invoke === "function",
      href: window.location.href,
      supportsAppServer:
        typeof window.electronAPI?.supportsCommand === "function"
          ? window.electronAPI.supportsCommand("app_server_handle_json_lines")
          : null,
      userAgent: navigator.userAgent,
    }));
    if (!runtime.electron || !runtime.hasInvoke) {
      throw new Error("CDP 连接的不是 Lime Electron renderer");
    }

    const traceEntriesBefore = collectTraceEntries(
      await page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      ),
    );
    const initialTurnStartTrace = readProductionTurnStartTrace(
      options.turnStartTrace,
      { expectedThreadId: options.threadId },
    );
    const installedList = await callAppServer(page, "pluginInstalled/list", {});
    const installedState = summarizeInstalledState(
      findContentFactoryInstalledState(installedList) || {},
    );
    const threadId = inferThreadId(
      options,
      traceEntriesBefore,
      initialTurnStartTrace,
    );
    if (!threadId) {
      throw new Error(
        "缺少 --thread-id，且无法从 turn/start trace 推断目标 thread",
      );
    }
    const readResult = await callAppServer(page, "thread/read", {
      threadId,
      includeTurns: true,
    });
    const sessionId = inferSessionId(options, traceEntriesBefore, readResult);
    if (!sessionId) {
      throw new Error("thread/read 未返回 thread.sessionId");
    }
    const turnStartTrace = readProductionTurnStartTrace(
      options.turnStartTrace,
      { expectedThreadId: threadId },
    );
    const evidenceResult = await callAppServer(page, "evidence/export", {
      sessionId,
      includeArtifacts: true,
      includeEvents: true,
      includeEvidencePack: true,
    });
    const traceEntries = collectTraceEntries(
      await page.evaluate(() =>
        window.localStorage.getItem("lime_invoke_trace_buffer_v1"),
      ),
    );
    const workflowJsonlEvents = readWorkflowJsonlEvents(options.workflowJsonl);
    const workflowResumeEventBindings = workflowJsonlEvents
      .map(workflowResumeEventBinding)
      .filter(Boolean);
    const workflowResumeTraceBindings =
      workflowResumeBindingsFromTrace(traceEntries);
    const workflowResumeLifecycle = summarizeWorkflowResumeLifecycle(
      workflowResumeTraceBindings,
      workflowResumeEventBindings,
    );
    const readModel = summarizeReadModel(readResult);
    const evidenceExport = summarizeEvidenceExport(evidenceResult);
    const workflowFactsDom = await inspectArticleEditorWorkflowFacts(page);
    const assertions = buildAssertions({
      runtime,
      traceEntries,
      turnStartTrace,
      installedState,
      readModel,
      workflowJsonlEvents,
      workflowResumeLifecycle,
      evidenceExport,
      workflowFactsDom,
    });
    const status = statusFromAssertions(assertions, readModel);
    const missingAssertions = missingAssertionKeys(assertions);
    const evidence = {
      schemaVersion: "content-factory-production-gui-evidence.v1",
      appId: APP_ID,
      generatedAt: new Date().toISOString(),
      sessionId,
      status,
      threadId,
      assertions,
      missingAssertions,
      cdp: {
        attached: true,
        endpointConfigured: Boolean(options.cdpUrl),
        pageUrlKind: pageUrlKind(runtime.href),
        usedRealElectron: runtime.electron,
        supportsAppServer: runtime.supportsAppServer,
      },
      eventLogs: {
        workflowJsonl: options.workflowJsonl || null,
        workflowJsonlEventCount: workflowJsonlEvents.length,
        workflowJsonlEventTypes: Array.from(
          new Set(
            workflowJsonlEvents
              .map((event) => event?.eventType)
              .filter(Boolean),
          ),
        ).sort(),
        workflowResumeEvents: workflowResumeEventBindings.map(sanitizeJson),
      },
      installedState,
      liveProvider: {
        apiKeyConfigured: null,
        apiKeyEnv: null,
        inference:
          "cloud_release required/verified/ready + hash matched + hostManagedGeneration completed + generated article marker clean",
        liveProviderUsed: assertions.liveProviderUsed,
        note: "Collector records GUI/read-model markers only; Provider secret values and raw provider requests are not read.",
      },
      providerEvidence: {
        liveProviderUsed: assertions.liveProviderUsed,
        productionRoute: true,
      },
      readModel,
      ui: {
        articleEditorWorkflowFacts: workflowFactsDom,
      },
      runtimeActionResponse:
        workflowResumeTraceBindings.length > 0
          ? {
              actionId: workflowResumeTraceBindings[0].actionId,
              confirmed:
                workflowResumeTraceBindings[0].decision === "approved"
                  ? true
                  : null,
              decision: workflowResumeTraceBindings[0].decision,
              method: workflowResumeTraceBindings[0].method,
              metadata: {
                workflowResume: {
                  stepId: workflowResumeTraceBindings[0].stepId,
                  workflowKey: workflowResumeTraceBindings[0].workflowKey,
                  workflowRunId: workflowResumeTraceBindings[0].workflowRunId,
                },
              },
            }
          : null,
      signatureVerificationStatus:
        installedState.signatureVerificationStatus || null,
      sourceKind: installedState.sourceKind || null,
      trace: {
        file: tracePath,
        appServerHandleJsonLinesSeen:
          assertions.turnStartViaElectronIpc ||
          traceEntries.some(
            (entry) => entry.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND,
          ),
        appServerMethodsSeen: appServerMethodsWithTurnStartTrace(
          traceEntries,
          turnStartTrace,
        ),
        turnStartTrace,
        workflowResumeBindingCount: workflowResumeTraceBindings.length,
      },
      evidenceExport,
      notes: [
        "This file is collected from a running real Electron page via CDP.",
        "It intentionally fails closed when cloud_release, signature verification, live provider read model, workflow JSONL, or workflowResume lifecycle evidence is missing.",
        "No Provider API key, bearer token, raw provider request, or full article text is written.",
      ],
    };
    writeJson(tracePath, {
      appServerInvokeEntries: traceEntries
        .filter(
          (entry) => entry.command === APP_SERVER_HANDLE_JSON_LINES_COMMAND,
        )
        .map(sanitizeJson),
    });
    writeJson(outputPath, evidence);
    console.log(
      `[content-factory-production-gui-evidence] status=${status} output=${outputPath}`,
    );
    if (status !== "passed") {
      console.log(
        `[content-factory-production-gui-evidence] missingAssertions=${missingAssertions.join(
          ",",
        )}`,
      );
    }
  } finally {
    await browser.close().catch(() => undefined);
  }
}

const isDirectRun =
  process.argv[1] &&
  fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);

if (isDirectRun) {
  run().catch((error) => {
    console.error(
      `[content-factory-production-gui-evidence] failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    process.exitCode = 1;
  });
}
