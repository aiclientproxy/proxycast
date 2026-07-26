#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import {
  EXPERT_SKILLS_RUNTIME_PROMPT,
  SKILLS_RUNTIME_SKILL_NAME,
  buildExpertSkillsRuntimeMetadata,
} from "./skills-runtime-fixture-scenario.mjs";
import { buildExpertSkillsLiveGateReport } from "./expert-skills-live-gate.mjs";
import {
  assertLiveProviderSmokeAllowed,
  liveProviderSmokeAllowed,
} from "../lib/live-provider-smoke-gate.mjs";
import {
  createAgentSessionCurrent,
  invokeAppServerMethod,
  invokeDevBridge,
  readAgentRuntimeThreadCurrent,
  resolveProviderPreference,
  sleep,
  startAgentSessionTurnCurrent,
  summarizeEvidencePack,
  summarizeThreadRead,
  threadSettled,
  updateAgentThreadSettingsCurrent,
  waitForHealth,
} from "../lib/agent-runtime-smoke-core.mjs";
import { workspaceIdFromDefaultProject } from "../lib/agent-runtime-smoke-support.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "../..");

const DEFAULT_OUTPUT = path.join(
  rootDir,
  ".lime/qc/expert-skills-live-runner-summary.json",
);
const DEFAULT_LIVE_WORKSPACE_ROOT = path.join(
  rootDir,
  ".lime/qc/expert-skills-live-workspace",
);
const DEFAULT_DETERMINISTIC_SUMMARY =
  ".lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-expert-panel-skills-runtime-regression-summary.json";
const DEFAULT_HEALTH_URL = "http://127.0.0.1:3030/health";
const DEFAULT_INVOKE_URL = "http://127.0.0.1:3030/invoke";
const DEFAULT_TIMEOUT_MS = 180_000;
const DEFAULT_INTERVAL_MS = 1_000;
const DEFAULT_SETTLED_GRACE_MS = 30_000;
const LOG_PREFIX = "[smoke:expert-skills-live-runner]";
const APP_SERVER_METHOD_EVIDENCE_EXPORT = "evidence/export";
const LIVE_SKILL_DIRECTORY = "capability-report";
const LIVE_SKILL_SOURCE_DRAFT_ID = "capdraft-live-capability-report";
const LIVE_SKILL_VERIFICATION_REPORT_ID = "capver-live-capability-report";

const CORE_ASSERTION_KEYS = [
  "expertSkillsRuntimePromptReachedBackend",
  "expertSkillsRuntimeMetadataReachedBackend",
  "expertDeclaredSkillRefsObserved",
  "expertSelectedSkillObserved",
  "expertInvokedSkillObserved",
  "readModelExpertSkillsRuntimeCompleted",
  "readModelExpertSkillSearchObserved",
  "readModelExpertSkillInvocationObserved",
  "evidenceExpertSkillBodyReadObserved",
  "evidenceExpertSkillGateObserved",
  "evidencePackExpertSkillSearchObserved",
  "evidencePackExpertSkillInvocationObserved",
  "expertSkillSearchBeforeSkillInvocation",
];

function isRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function resolvePath(inputPath) {
  return path.isAbsolute(inputPath) ? inputPath : path.join(rootDir, inputPath);
}

function readJsonFile(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJsonFile(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function jsonText(value) {
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return String(value ?? "");
  }
}

function valueAt(value, keys) {
  let cursor = value;
  for (const key of keys) {
    if (!isRecord(cursor)) {
      return undefined;
    }
    cursor = cursor[key];
  }
  return cursor;
}

function boolAt(value, keys) {
  return valueAt(value, keys) === true;
}

function hasLiveProviderStatement(summary) {
  return (
    summary?.liveProviderUsed === true ||
    boolAt(summary, ["assertions", "liveProviderUsed"]) ||
    boolAt(summary, ["commonAssertions", "liveProviderUsed"]) ||
    boolAt(summary, ["scenarioAssertions", "liveProviderUsed"]) ||
    boolAt(summary, ["liveProvider", "used"])
  );
}

function providerLooksLikeFixture(summary) {
  return (
    summary?.provider === "fixture-provider" ||
    summary?.model === "fixture-model" ||
    summary?.liveProvider?.provider === "fixture-provider" ||
    summary?.liveProvider?.model === "fixture-model"
  );
}

function sourceProvider(summary) {
  return String(
    summary?.provider ||
      summary?.liveProvider?.provider ||
      summary?.runtime?.providerPreference ||
      "",
  ).trim();
}

function sourceModel(summary) {
  return String(
    summary?.model ||
      summary?.liveProvider?.model ||
      summary?.runtime?.modelPreference ||
      "",
  ).trim();
}

function assertLiveSummarySource(summary, sourcePath) {
  if (!isRecord(summary)) {
    throw new Error(`live summary 不是 JSON object: ${sourcePath}`);
  }
  if (!hasLiveProviderStatement(summary)) {
    throw new Error(
      `live summary 缺少 liveProviderUsed=true 声明，不能作为真实 Provider 证据: ${sourcePath}`,
    );
  }
  if (providerLooksLikeFixture(summary)) {
    throw new Error(
      `live summary 仍是 fixture provider/model，不能作为真实 Provider 证据: ${sourcePath}`,
    );
  }
  if (!sourceProvider(summary) || !sourceModel(summary)) {
    throw new Error(`live summary 缺少真实 provider/model 字段: ${sourcePath}`);
  }
}

export function normalizeLiveSummaryFromSource(summary, sourcePath) {
  assertLiveSummarySource(summary, sourcePath);
  const provider = sourceProvider(summary);
  const model = sourceModel(summary);
  return {
    ...summary,
    ok: summary.ok === true,
    scenario: summary.scenario || "expert-skills-runtime-live",
    provider,
    model,
    liveProviderUsed: true,
    liveProvider: {
      ...(isRecord(summary.liveProvider) ? summary.liveProvider : {}),
      used: true,
      provider,
      model,
    },
    runner: {
      ...(isRecord(summary.runner) ? summary.runner : {}),
      source: "smoke:expert-skills-live-runner",
      sourceSummary: sourcePath,
      normalizedAt: new Date().toISOString(),
      executionMode: "provided-live-summary",
    },
  };
}

function indexOfAny(text, patterns) {
  const indexes = patterns
    .map((pattern) => text.indexOf(pattern))
    .filter((index) => index >= 0);
  return indexes.length > 0 ? Math.min(...indexes) : -1;
}

function countAny(text, patterns) {
  return patterns.reduce((count, pattern) => {
    let nextCount = count;
    let index = text.indexOf(pattern);
    while (index >= 0) {
      nextCount += 1;
      index = text.indexOf(pattern, index + pattern.length);
    }
    return nextCount;
  }, 0);
}

function pickString(target, keys) {
  for (const key of keys) {
    const value = target?.[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function visitRecords(value, visitor, seen = new Set()) {
  if (!value || typeof value !== "object") {
    return;
  }
  if (seen.has(value)) {
    return;
  }
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) {
      visitRecords(item, visitor, seen);
    }
    return;
  }
  visitor(value);
  for (const item of Object.values(value)) {
    visitRecords(item, visitor, seen);
  }
}

function toolRecordPayload(record) {
  return isRecord(record?.payload) ? record.payload : record;
}

function toolRecordName(record) {
  const payload = toolRecordPayload(record);
  return pickString(payload, ["toolName", "tool_name", "name"]);
}

function toolRecordMetadata(record) {
  const payload = toolRecordPayload(record);
  return isRecord(payload?.metadata) ? payload.metadata : {};
}

function toolRecordType(record) {
  return pickString(record, ["type", "eventType", "event_type"]);
}

function isStructuredToolRecord(record) {
  const payload = toolRecordPayload(record);
  const type = toolRecordType(record);
  return (
    type.startsWith("tool.") ||
    Object.hasOwn(payload, "toolName") ||
    Object.hasOwn(payload, "tool_name") ||
    Object.hasOwn(payload, "toolCallId") ||
    Object.hasOwn(payload, "tool_call_id") ||
    Object.hasOwn(payload, "toolId") ||
    Object.hasOwn(payload, "tool_id")
  );
}

function collectStructuredToolMarkers(...sources) {
  const markers = [];
  for (const source of sources) {
    visitRecords(source, (record) => {
      if (!isStructuredToolRecord(record)) {
        return;
      }
      const payload = toolRecordPayload(record);
      const metadata = toolRecordMetadata(record);
      const toolName = toolRecordName(record);
      const toolFamily = pickString(metadata, ["tool_family", "toolFamily"]);
      const payloadText = jsonText(payload);
      if (toolName === "skill_search" || toolFamily === "skill_search") {
        markers.push({ kind: "skill_search" });
        return;
      }
      if (
        toolName === "Skill" ||
        toolName === "skill" ||
        toolFamily === "skill" ||
        payloadText.includes("skill_invocation")
      ) {
        markers.push({ kind: "skill_invocation" });
      }
    });
  }
  return markers;
}

function latestStatusFromThreadRead(threadRead) {
  return String(
    threadRead?.diagnostics?.latest_turn_status ||
      threadRead?.diagnostics?.latestTurnStatus ||
      threadRead?.runtime_summary?.latestTurnStatus ||
      threadRead?.runtimeSummary?.latestTurnStatus ||
      threadRead?.status ||
      valueAt(threadRead, ["result", "session", "status"]) ||
      "",
  ).toLowerCase();
}

function threadCompleted(threadRead) {
  const latestStatus = latestStatusFromThreadRead(threadRead);
  if (latestStatus === "completed" || latestStatus === "succeeded") {
    return true;
  }
  const turnStatuses = Array.isArray(threadRead?.turns)
    ? threadRead.turns.map((turn) => String(turn?.status || "").toLowerCase())
    : [];
  return turnStatuses.length > 0
    ? turnStatuses.every(
        (status) => status === "completed" || status === "succeeded",
      )
    : false;
}

function providerFailureMessage(threadRead) {
  return (
    pickString(threadRead?.diagnostics, [
      "latest_turn_error_message",
      "latestTurnErrorMessage",
    ]) ||
    pickString(threadRead?.runtime_summary, [
      "latestTurnErrorMessage",
      "latest_turn_error_message",
    ]) ||
    pickString(threadRead?.runtimeSummary, [
      "latestTurnErrorMessage",
      "latest_turn_error_message",
    ]) ||
    ""
  );
}

async function exportAgentSessionEvidenceCurrent(
  options,
  { sessionId, turnId },
) {
  return invokeAppServerMethod(options, APP_SERVER_METHOD_EVIDENCE_EXPORT, {
    sessionId,
    turnId,
    includeEvents: true,
    includeArtifacts: true,
    includeEvidencePack: true,
  });
}

function writeLiveSkillFileIfMissing(skillFilePath) {
  if (fs.existsSync(skillFilePath)) {
    return false;
  }
  fs.writeFileSync(
    skillFilePath,
    [
      "---",
      "name: Capability Report",
      "description: Read local context and produce a minimal capability/code-review report for live runtime verification.",
      "allowed-tools: Read",
      "---",
      "",
      "# Capability Report",
      "",
      "Use this skill when the current expert needs a small, evidence-backed code capability or review report.",
      "",
      "## Steps",
      "",
      "1. Read only the immediately relevant local context.",
      "2. Identify one concrete capability, risk, or improvement.",
      "3. Return the checked scope, evidence used, and one minimal recommendation.",
      "",
      "Do not claim that external systems were changed unless the evidence shows it.",
      "",
    ].join("\n"),
  );
  return true;
}

export function ensureExpertSkillsLiveWorkspaceSkill(workspaceRoot) {
  const resolvedWorkspaceRoot = resolvePath(workspaceRoot);
  if (!path.isAbsolute(resolvedWorkspaceRoot)) {
    throw new Error(`live workspace root 必须是绝对路径: ${workspaceRoot}`);
  }
  const skillDirectory = path.join(
    resolvedWorkspaceRoot,
    ".agents",
    "skills",
    LIVE_SKILL_DIRECTORY,
  );
  const skillFilePath = path.join(skillDirectory, "SKILL.md");
  const registrationDirectory = path.join(skillDirectory, ".lime");
  const registrationFilePath = path.join(
    registrationDirectory,
    "registration.json",
  );
  fs.mkdirSync(registrationDirectory, { recursive: true });
  const skillFileCreated = writeLiveSkillFileIfMissing(skillFilePath);
  const registration = {
    registrationId: "capreg-live-capability-report",
    registeredAt: new Date().toISOString(),
    skillDirectory: LIVE_SKILL_DIRECTORY,
    registeredSkillDirectory: skillDirectory,
    sourceDraftId: LIVE_SKILL_SOURCE_DRAFT_ID,
    sourceVerificationReportId: LIVE_SKILL_VERIFICATION_REPORT_ID,
    generatedFileCount: 1,
    permissionSummary: ["Level 0 read-only live smoke"],
  };
  writeJsonFile(registrationFilePath, registration);
  return {
    workspaceRoot: resolvedWorkspaceRoot,
    directory: LIVE_SKILL_DIRECTORY,
    skillName: SKILLS_RUNTIME_SKILL_NAME,
    skillDirectory,
    skillFilePath,
    registrationFilePath,
    registration,
    skillFileCreated,
  };
}

export function buildExpertSkillsLiveRuntimeMetadata(workspaceSkill) {
  const metadata = buildExpertSkillsRuntimeMetadata();
  return {
    ...metadata,
    workspaceRoot: workspaceSkill.workspaceRoot,
    harness: {
      ...metadata.harness,
      source: "smoke:expert-skills-live-runner",
      workspace_root: workspaceSkill.workspaceRoot,
      working_directory: workspaceSkill.workspaceRoot,
      workspace_skill_runtime_enable: {
        source: "manual_session_enable",
        approval: "manual",
        workspace_root: workspaceSkill.workspaceRoot,
        bindings: [
          {
            directory: workspaceSkill.directory,
            skill: workspaceSkill.skillName,
            registered_skill_directory: workspaceSkill.skillDirectory,
            source_draft_id:
              workspaceSkill.registration.sourceDraftId ||
              LIVE_SKILL_SOURCE_DRAFT_ID,
            source_verification_report_id:
              workspaceSkill.registration.sourceVerificationReportId ||
              LIVE_SKILL_VERIFICATION_REPORT_ID,
            permission_summary:
              workspaceSkill.registration.permissionSummary || [],
          },
        ],
      },
    },
  };
}

export function buildLiveRuntimeSummary({
  provider,
  model,
  threadRead,
  evidencePack,
  evidenceExport,
  sessionId,
  turnId,
  workspaceSkill,
}) {
  const threadText = jsonText(threadRead);
  const evidenceText = jsonText(evidencePack);
  const evidenceExportText = jsonText(evidenceExport);
  const combinedText = `${threadText}\n${evidenceText}\n${evidenceExportText}`;
  const toolMarkers = collectStructuredToolMarkers(
    threadRead,
    evidencePack,
    evidenceExport,
  );
  const searchIndex = toolMarkers.findIndex(
    (marker) => marker.kind === "skill_search",
  );
  const invocationIndex = toolMarkers.findIndex(
    (marker) => marker.kind === "skill_invocation",
  );
  const skillSearchCount = toolMarkers.filter(
    (marker) => marker.kind === "skill_search",
  ).length;
  const skillInvocationCount = toolMarkers.filter(
    (marker) => marker.kind === "skill_invocation",
  ).length;
  const skillBodyReadObserved =
    combinedText.includes("skill_body_read") ||
    combinedText.includes("skillBodyRead") ||
    combinedText.includes("SKILL.md");
  const skillGateObserved =
    combinedText.includes("skill_gate_decision") ||
    combinedText.includes("skillGate") ||
    combinedText.includes("Skill gate");
  const expertDeclaredObserved =
    combinedText.includes("expert_binding") ||
    combinedText.includes("skill:capability-report") ||
    combinedText.includes("expert.skillRefs");
  const expertSelectedObserved =
    combinedText.includes("capability-report") &&
    (combinedText.includes("selected") ||
      combinedText.includes("selection") ||
      combinedText.includes("expert_binding"));
  const expertInvokedObserved =
    skillInvocationCount > 0 && combinedText.includes("capability-report");
  const skillSearchBeforeSkillInvocation =
    searchIndex >= 0 && invocationIndex >= 0 && searchIndex < invocationIndex;

  const assertions = {
    liveProviderUsed: true,
    liveProviderNotUsed: false,
    expertSkillsRuntimePromptReachedBackend: true,
    expertSkillsRuntimeMetadataReachedBackend: expertDeclaredObserved,
    expertDeclaredSkillRefsObserved: expertDeclaredObserved,
    expertSelectedSkillObserved: expertSelectedObserved,
    expertInvokedSkillObserved: expertInvokedObserved,
    readModelExpertSkillsRuntimeCompleted: threadCompleted(threadRead),
    readModelExpertSkillSearchObserved: skillSearchCount > 0,
    readModelExpertSkillInvocationObserved: skillInvocationCount > 0,
    evidenceExpertSkillBodyReadObserved: skillBodyReadObserved,
    evidenceExpertSkillGateObserved: skillGateObserved,
    evidencePackExpertSkillSearchObserved: skillSearchCount > 0,
    evidencePackExpertSkillInvocationObserved: skillInvocationCount > 0,
    expertSkillSearchBeforeSkillInvocation: skillSearchBeforeSkillInvocation,
  };
  const ok = CORE_ASSERTION_KEYS.every((key) => assertions[key] === true);

  return {
    ok,
    scenario: "expert-skills-runtime-live",
    provider,
    model,
    liveProviderUsed: true,
    assertions,
    liveProvider: {
      used: true,
      provider,
      model,
    },
    runtime: {
      sessionId,
      turnId,
      thread: summarizeThreadRead(threadRead),
      evidencePack: summarizeEvidencePack(evidencePack),
      evidenceExport: evidenceExport
        ? {
            eventCount: Array.isArray(evidenceExport.events)
              ? evidenceExport.events.length
              : Array.isArray(evidenceExport.agentEvents)
                ? evidenceExport.agentEvents.length
                : null,
            artifactCount: Array.isArray(evidenceExport.artifacts)
              ? evidenceExport.artifacts.length
              : null,
          }
        : null,
      providerFailureMessage: providerFailureMessage(threadRead) || null,
      workspaceSkill: workspaceSkill
        ? {
            workspaceRoot: workspaceSkill.workspaceRoot,
            directory: workspaceSkill.directory,
            skillName: workspaceSkill.skillName,
            skillFilePath: workspaceSkill.skillFilePath,
            registrationFilePath: workspaceSkill.registrationFilePath,
            skillFileCreated: workspaceSkill.skillFileCreated,
          }
        : null,
    },
    evidencePackExpertSkillsRuntime: {
      hasEvidencePack: Boolean(evidencePack),
      skillSearchCount,
      skillInvocationCount,
      skillBodyReadObserved,
      skillGateObserved,
      expertDeclaredObserved,
      expertSelectedObserved,
      expertInvokedObserved,
      skillSearchBeforeSkillInvocation,
    },
    runner: {
      source: "smoke:expert-skills-live-runner",
      executionMode: "live-runtime",
      generatedAt: new Date().toISOString(),
    },
  };
}

async function waitForRuntimeCompletion(options, sessionId) {
  const startedAt = Date.now();
  let lastSnapshot = null;
  const deadline = startedAt + options.timeoutMs;
  const graceDeadline = deadline + options.settledGraceMs;
  while (Date.now() < graceDeadline) {
    const threadRead = await readAgentRuntimeThreadCurrent(options, sessionId, {
      historyLimit: 80,
    });
    lastSnapshot = summarizeThreadRead(threadRead);
    if (threadSettled(threadRead)) {
      return threadRead;
    }
    const remainingMs =
      Date.now() < deadline
        ? deadline - Date.now()
        : graceDeadline - Date.now();
    if (remainingMs <= 0) {
      break;
    }
    await sleep(options.intervalMs);
  }
  throw new Error(
    `${LOG_PREFIX} live runtime timeout; timeoutMs=${options.timeoutMs} settledGraceMs=${options.settledGraceMs} last=${JSON.stringify(lastSnapshot)}`,
  );
}

async function executeLiveRuntime(options) {
  await waitForHealth(options);
  const workspace = await invokeDevBridge(
    options,
    "get_or_create_default_project",
    {},
    30_000,
  );
  const workspaceId = workspaceIdFromDefaultProject(workspace);
  if (!workspaceId) {
    throw new Error(`${LOG_PREFIX} 默认 workspace 缺少 id`);
  }

  const provider = await resolveProviderPreference(options);
  const workspaceSkill = ensureExpertSkillsLiveWorkspaceSkill(
    options.liveWorkspaceRoot,
  );
  const metadata = buildExpertSkillsLiveRuntimeMetadata(workspaceSkill);
  const sessionId = await createAgentSessionCurrent(options, {
    workspaceId,
    title: `Expert Skills live ${new Date().toISOString()}`,
    executionStrategy: "react",
    metadata: {
      ...metadata,
      harness: {
        ...metadata.harness,
        hiddenFromUserRecents: true,
        source: "smoke:expert-skills-live-runner",
      },
    },
  });
  await updateAgentThreadSettingsCurrent(options, {
    threadId: sessionId,
    provider,
  });

  const turnId = `expert-skills-live-${Date.now()}-${process.pid}`;
  const eventName = `app_server_expert_skills_live_${turnId}`;
  await startAgentSessionTurnCurrent(options, {
    sessionId,
    workspaceId,
    message: options.prompt,
    eventName,
    turnId,
    runtimeRequest: {
      providerPreference: provider.providerPreference,
      modelPreference: provider.modelPreference,
      approvalPolicy: "never",
      sandboxPolicy: "workspace-write",
      metadata,
    },
    skipPreSubmitResume: true,
  });

  const threadRead = await waitForRuntimeCompletion(options, sessionId);
  const evidenceExport = await exportAgentSessionEvidenceCurrent(options, {
    sessionId,
    turnId,
  });
  const evidencePack =
    evidenceExport?.evidencePack ?? evidenceExport?.evidence_pack ?? null;
  return buildLiveRuntimeSummary({
    provider: provider.providerPreference,
    model: provider.modelPreference,
    threadRead,
    evidencePack,
    evidenceExport,
    sessionId,
    turnId,
    workspaceSkill,
  });
}

function printHelp() {
  console.log(`
Expert Skills Live Runner

用途:
  为专家 Skills Runtime 生成可被 smoke:expert-skills-live-gate 消费的 live Provider summary。
  默认禁止真实模型调用；必须显式传 --allow-live-provider 或设置 live Provider smoke 环境变量。

用法:
  npm run smoke:expert-skills-live-runner -- --allow-live-provider --live-summary .lime/qc/live-summary.json
  npm run smoke:expert-skills-live-runner -- --allow-live-provider --execute-live-runtime

选项:
  --allow-live-provider       明确允许本脚本处理 live Provider 验收；真实执行还需要 --execute-live-runtime
  --live-summary <path>       读取已有 live Provider 专家 Skills summary 并归一化输出
  --execute-live-runtime      通过 App Server current JSON-RPC 提交真实 Provider turn
  --deterministic-summary <path>
                              deterministic Electron summary，用于最终 gate 校验
  --output <path>             live summary 输出路径，默认 .lime/qc/expert-skills-live-runner-summary.json
  --health-url <url>          DevBridge health 地址，默认 ${DEFAULT_HEALTH_URL}
  --invoke-url <url>          DevBridge invoke 地址，默认 ${DEFAULT_INVOKE_URL}
  --live-workspace-root <path>
                              live smoke 专用 workspace root，默认 ${DEFAULT_LIVE_WORKSPACE_ROOT}
  --timeout-ms <ms>           live runtime 等待超时，默认 ${DEFAULT_TIMEOUT_MS}
  --settled-grace-ms <ms>     timeout 后继续短暂轮询稳定 read model，默认 ${DEFAULT_SETTLED_GRACE_MS}
  --interval-ms <ms>          轮询间隔，默认 ${DEFAULT_INTERVAL_MS}
  --provider-preference <id>  live runtime provider 偏好
  --model-preference <name>   live runtime model 偏好
  --format json|text          输出格式
  -h, --help                  显示帮助
`);
}

export function parseArgs(argv) {
  const options = {
    allowLiveProvider: liveProviderSmokeAllowed(),
    liveSummary: "",
    executeLiveRuntime: false,
    deterministicSummary: DEFAULT_DETERMINISTIC_SUMMARY,
    output: DEFAULT_OUTPUT,
    healthUrl: DEFAULT_HEALTH_URL,
    invokeUrl: DEFAULT_INVOKE_URL,
    liveWorkspaceRoot: DEFAULT_LIVE_WORKSPACE_ROOT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    settledGraceMs: DEFAULT_SETTLED_GRACE_MS,
    intervalMs: DEFAULT_INTERVAL_MS,
    providerPreference:
      process.env.LIME_AGENT_QC_PROVIDER ||
      process.env.LIME_E2E_PROVIDER ||
      process.env.LIME_DEFAULT_PROVIDER ||
      "",
    modelPreference:
      process.env.LIME_AGENT_QC_MODEL ||
      process.env.LIME_E2E_MODEL ||
      process.env.LIME_DEFAULT_MODEL ||
      "",
    prompt: EXPERT_SKILLS_RUNTIME_PROMPT,
    format: "text",
    logPrefix: LOG_PREFIX,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--live-summary" && next) {
      options.liveSummary = next;
      index += 1;
      continue;
    }
    if (arg === "--execute-live-runtime") {
      options.executeLiveRuntime = true;
      continue;
    }
    if (arg === "--deterministic-summary" && next) {
      options.deterministicSummary = next;
      index += 1;
      continue;
    }
    if (arg === "--output" && next) {
      options.output = resolvePath(next);
      index += 1;
      continue;
    }
    if (arg === "--health-url" && next) {
      options.healthUrl = String(next).trim();
      index += 1;
      continue;
    }
    if (arg === "--invoke-url" && next) {
      options.invokeUrl = String(next).trim();
      index += 1;
      continue;
    }
    if (arg === "--live-workspace-root" && next) {
      options.liveWorkspaceRoot = resolvePath(next);
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && next) {
      options.timeoutMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--settled-grace-ms" && next) {
      options.settledGraceMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && next) {
      options.intervalMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--provider-preference" && next) {
      options.providerPreference = String(next).trim();
      index += 1;
      continue;
    }
    if (arg === "--model-preference" && next) {
      options.modelPreference = String(next).trim();
      index += 1;
      continue;
    }
    if (arg === "--prompt" && next) {
      options.prompt = String(next);
      index += 1;
      continue;
    }
    if (arg === "--format" && next) {
      options.format = String(next).trim();
      index += 1;
      continue;
    }
    throw new Error(`未知参数: ${arg}`);
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms 必须是 >= 30000 的数字");
  }
  if (!Number.isFinite(options.settledGraceMs) || options.settledGraceMs < 0) {
    throw new Error("--settled-grace-ms 必须是 >= 0 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!["json", "text"].includes(options.format)) {
    throw new Error("--format 只支持 json 或 text");
  }
  return options;
}

export async function runExpertSkillsLiveRunner(options) {
  assertLiveProviderSmokeAllowed({
    allowed: options.allowLiveProvider,
    scriptName: "smoke:expert-skills-live-runner",
  });

  if (options.liveSummary && options.executeLiveRuntime) {
    throw new Error("--live-summary 与 --execute-live-runtime 只能二选一");
  }
  if (!options.liveSummary && !options.executeLiveRuntime) {
    throw new Error(
      "缺少 live 验收输入：请传 --live-summary <path>，或显式传 --execute-live-runtime 真实执行 Provider turn。",
    );
  }

  const liveSummary = options.liveSummary
    ? normalizeLiveSummaryFromSource(
        readJsonFile(resolvePath(options.liveSummary)),
        resolvePath(options.liveSummary),
      )
    : await executeLiveRuntime(options);

  writeJsonFile(options.output, liveSummary);
  const gateReport = buildExpertSkillsLiveGateReport({
    deterministicSummary: options.deterministicSummary,
    liveSummary: options.output,
  });
  return {
    summary: liveSummary,
    output: options.output,
    gateReport,
  };
}

function printTextResult(result) {
  console.log(`EXPERT_SKILLS_LIVE_RUNNER_RESULT=${result.gateReport.status}`);
  console.log(`output=${result.output}`);
  console.log(`provider=${result.summary.provider}`);
  console.log(`model=${result.summary.model}`);
  if (result.gateReport.nextRequired) {
    console.log(`next=${result.gateReport.nextRequired}`);
  }
  for (const issue of result.gateReport.live.issues ?? []) {
    console.log(`live: ${issue}`);
  }
}

async function main() {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
      printHelp();
      return;
    }
    const result = await runExpertSkillsLiveRunner(options);
    if (options.format === "json") {
      console.log(JSON.stringify(result, null, 2));
    } else {
      printTextResult(result);
    }
    if (result.gateReport.status !== "pass") {
      process.exitCode = 1;
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

const currentFile = fileURLToPath(import.meta.url);
if (process.argv[1] && path.resolve(process.argv[1]) === currentFile) {
  main();
}
