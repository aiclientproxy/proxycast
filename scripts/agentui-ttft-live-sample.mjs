#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import {
  assertLiveProviderSmokeAllowed,
  liveProviderSmokeAllowed,
} from "./lib/live-provider-smoke-gate.mjs";
import {
  createAgentSessionCurrent,
  readAgentRuntimeThreadCurrent,
  startAgentSessionTurnCurrent,
} from "./lib/agent-runtime-smoke-core.mjs";

const DEFAULT_HEALTH_URL = "http://127.0.0.1:3030/health";
const DEFAULT_INVOKE_URL = "http://127.0.0.1:3030/invoke";
const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_INTERVAL_MS = 1_000;
const DEFAULT_MESSAGE = "Reply with exactly: OK";
const DEFAULT_MODE = "responsive-auto";
const MAX_SAMPLES_PER_RUN = 6;
const TERMINAL_RUN_STATUSES = new Set([
  "completed",
  "success",
  "error",
  "failed",
  "timeout",
  "idle",
  "aborted",
  "cancelled",
  "canceled",
]);

function printHelp() {
  console.log(`
Lime AgentUI TTFT Live Sample

用途:
  通过 DevBridge 提交真实 Agent Runtime 回合，采集脱敏 routing / TTFT 证据。
  默认走 responsive_chat 自动路由，不输出 prompt、assistant 正文、error_message、密钥或 run id。

用法:
  node scripts/agentui-ttft-live-sample.mjs
  node scripts/agentui-ttft-live-sample.mjs --samples 3 --output /tmp/agentui-ttft-live-sample.json

选项:
  --mode responsive-auto|request-override
                       默认 responsive-auto；request-override 仅用于对照样本
  --provider-preference ID
                       request-override 模式下的 provider
  --model-preference ID
                       request-override 模式下的 model
  --samples N          本次采样条数，默认 1，单次最多 ${MAX_SAMPLES_PER_RUN}
  --message TEXT       提交给模型的最短测试消息；不会写入脚本输出，默认 ${DEFAULT_MESSAGE}
  --output PATH        写入脱敏 JSON；默认 stdout
  --allow-live-provider
                       确认允许调用真实模型 Provider；默认禁止以避免消耗额度
  --trace              打印脱敏阶段耗时，便于定位 health / turn start / first text
  --health-url URL     DevBridge health 地址，默认 ${DEFAULT_HEALTH_URL}
  --invoke-url URL     DevBridge invoke 地址，默认 ${DEFAULT_INVOKE_URL}
  --timeout-ms MS      单条样本等待超时，默认 ${DEFAULT_TIMEOUT_MS}
  --interval-ms MS     轮询间隔，默认 ${DEFAULT_INTERVAL_MS}
  -h, --help           显示帮助
`);
}

function parseArgs(argv) {
  const options = {
    healthUrl: DEFAULT_HEALTH_URL,
    invokeUrl: DEFAULT_INVOKE_URL,
    intervalMs: DEFAULT_INTERVAL_MS,
    message: DEFAULT_MESSAGE,
    mode: DEFAULT_MODE,
    modelPreference: "",
    outputPath: "",
    providerPreference: "",
    samples: 1,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    allowLiveProvider: liveProviderSmokeAllowed(),
    trace: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
    if (arg === "--health-url" && argv[index + 1]) {
      options.healthUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--invoke-url" && argv[index + 1]) {
      options.invokeUrl = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && argv[index + 1]) {
      options.timeoutMs = Number(String(argv[index + 1]).trim());
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && argv[index + 1]) {
      options.intervalMs = Number(String(argv[index + 1]).trim());
      index += 1;
      continue;
    }
    if (arg === "--samples" && argv[index + 1]) {
      options.samples = Number.parseInt(String(argv[index + 1]).trim(), 10);
      index += 1;
      continue;
    }
    if (arg === "--mode" && argv[index + 1]) {
      options.mode = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--message" && argv[index + 1]) {
      options.message = String(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--output" && argv[index + 1]) {
      options.outputPath = path.resolve(String(argv[index + 1]).trim());
      index += 1;
      continue;
    }
    if (arg === "--allow-live-provider") {
      options.allowLiveProvider = true;
      continue;
    }
    if (arg === "--trace") {
      options.trace = true;
      continue;
    }
    if (arg === "--provider-preference" && argv[index + 1]) {
      options.providerPreference = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--model-preference" && argv[index + 1]) {
      options.modelPreference = String(argv[index + 1]).trim();
      index += 1;
    }
  }

  if (!["responsive-auto", "request-override"].includes(options.mode)) {
    throw new Error("--mode 只能是 responsive-auto 或 request-override");
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 10_000) {
    throw new Error("--timeout-ms 必须是 >= 10000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }
  if (!Number.isFinite(options.samples) || options.samples < 1) {
    throw new Error("--samples 必须是 >= 1 的数字");
  }
  options.samples = Math.min(Math.floor(options.samples), MAX_SAMPLES_PER_RUN);
  if (options.mode === "request-override") {
    if (!options.providerPreference || !options.modelPreference) {
      throw new Error(
        "request-override 模式必须同时提供 --provider-preference 和 --model-preference",
      );
    }
  }

  return options;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function trace(options, stage, fields = {}) {
  if (!options.trace) {
    return;
  }
  const suffix = Object.entries(fields)
    .filter(
      ([, value]) => value !== undefined && value !== null && value !== "",
    )
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(" ");
  console.log(
    `[agentui-ttft-live] stage=${stage}${suffix ? ` ${suffix}` : ""}`,
  );
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}

function stringField(record, keys) {
  const source = asRecord(record);
  if (!source) {
    return null;
  }
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function numberField(record, keys) {
  const source = asRecord(record);
  if (!source) {
    return null;
  }
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim()) {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return null;
}

function stringArrayField(record, keys) {
  const source = asRecord(record);
  if (!source) {
    return [];
  }
  for (const key of keys) {
    const value = source[key];
    if (Array.isArray(value)) {
      return value.filter((item) => typeof item === "string" && item.trim());
    }
  }
  return [];
}

async function waitForHealth(options) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    try {
      const response = await fetch(options.healthUrl, {
        method: "GET",
        signal: AbortSignal.timeout(Math.min(options.intervalMs, 5_000)),
      });
      const text = await response.text();
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      trace(options, "health-ready", {
        elapsedMs: Date.now() - startedAt,
      });
      return text ? JSON.parse(text) : {};
    } catch (error) {
      lastError = error;
      await sleep(options.intervalMs);
    }
  }

  throw new Error(
    `DevBridge health timeout: ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }`,
  );
}

async function fetchWithHardTimeout(url, init, timeoutMs, label) {
  const controller = new AbortController();
  let rejectTimeout;
  const abortTimeout = setTimeout(() => controller.abort(), timeoutMs);
  const timeoutPromise = new Promise((_, reject) => {
    rejectTimeout = setTimeout(() => {
      reject(new Error(`${label} timeout after ${timeoutMs}ms`));
    }, timeoutMs + 100);
  });
  try {
    return await Promise.race([
      fetch(url, { ...init, signal: controller.signal }),
      timeoutPromise,
    ]);
  } finally {
    clearTimeout(abortTimeout);
    clearTimeout(rejectTimeout);
  }
}

async function invoke(options, cmd, args, timeoutMs = options.timeoutMs) {
  const response = await fetchWithHardTimeout(
    options.invokeUrl,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ cmd, args }),
    },
    timeoutMs,
    cmd,
  );
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${cmd} HTTP ${response.status}`);
  }
  const payload = text ? JSON.parse(text) : null;
  if (payload?.error) {
    throw new Error(`${cmd} error (redacted)`);
  }
  return payload?.result;
}

function buildRuntimeRequest(options) {
  const runtimeRequest = {
    searchMode: "disabled",
    webSearch: false,
    thinkingEnabled: false,
  };

  if (options.mode === "request-override") {
    runtimeRequest.providerPreference = options.providerPreference;
    runtimeRequest.modelPreference = options.modelPreference;
  }

  return runtimeRequest;
}

function latestTurnStatus(threadRead) {
  const diagnostics = asRecord(threadRead?.diagnostics);
  const runtimeSummary = asRecord(
    threadRead?.runtime_summary ?? threadRead?.runtimeSummary,
  );
  return (
    stringField(diagnostics, ["latest_turn_status", "latestTurnStatus"]) ||
    stringField(runtimeSummary, ["latestTurnStatus", "latest_turn_status"]) ||
    stringField(threadRead, ["status"])
  );
}

function summarizeThreadRead(threadRead) {
  const modelRouting = asRecord(
    threadRead?.model_routing ?? threadRead?.modelRouting,
  );
  const latestTiming = asRecord(
    modelRouting?.latestModelDeltaTiming ??
      modelRouting?.latest_model_delta_timing,
  );
  const latestRouting = asRecord(latestTiming?.routing);
  const runtimeSummary = asRecord(
    threadRead?.runtime_summary ?? threadRead?.runtimeSummary,
  );

  const routing = {
    decisionSource:
      stringField(modelRouting, ["decisionSource", "decision_source"]) ||
      stringField(latestRouting, ["decisionSource", "decision_source"]) ||
      stringField(runtimeSummary, ["decisionSource", "decision_source"]),
    settingsSource:
      stringField(modelRouting, ["settingsSource", "settings_source"]) ||
      stringField(latestRouting, ["settingsSource", "settings_source"]),
    serviceModelSlot:
      stringField(modelRouting, ["serviceModelSlot", "service_model_slot"]) ||
      stringField(latestRouting, ["serviceModelSlot", "service_model_slot"]),
    selectedProvider:
      stringField(modelRouting, ["selectedProvider", "selected_provider"]) ||
      stringField(latestRouting, ["selectedProvider", "selected_provider"]),
    selectedModel:
      stringField(modelRouting, ["selectedModel", "selected_model"]) ||
      stringField(latestRouting, ["selectedModel", "selected_model"]),
    decisionReason:
      stringField(modelRouting, ["decisionReason", "decision_reason"]) ||
      stringField(latestRouting, ["decisionReason", "decision_reason"]),
    fallbackChain: [
      ...new Set([
        ...stringArrayField(modelRouting, ["fallbackChain", "fallback_chain"]),
        ...stringArrayField(latestRouting, ["fallbackChain", "fallback_chain"]),
      ]),
    ],
  };

  const timing = {
    source: stringField(latestTiming, ["source"]),
    runStatus: stringField(latestTiming, ["runStatus", "run_status"]),
    durationMs: numberField(latestTiming, ["durationMs", "duration_ms"]),
    firstVisibleDeltaMs: numberField(latestTiming, [
      "firstVisibleDeltaMs",
      "first_visible_delta_ms",
    ]),
    firstThinkingDeltaMs: numberField(latestTiming, [
      "firstThinkingDeltaMs",
      "first_thinking_delta_ms",
    ]),
    firstTextDeltaMs: numberField(latestTiming, [
      "firstTextDeltaMs",
      "first_text_delta_ms",
    ]),
  };

  return {
    status: {
      threadStatus: stringField(threadRead, ["status"]),
      latestTurnStatus: latestTurnStatus(threadRead),
      latestRunStatus: timing.runStatus,
    },
    routing,
    timing,
    assertions: {
      hasFirstTextDelta: timing.firstTextDeltaMs !== null,
      hasRoutingEvidence: Boolean(
        routing.decisionSource ||
        routing.selectedProvider ||
        routing.selectedModel ||
        routing.decisionReason,
      ),
      isResponsiveChatAuto: routing.decisionSource === "responsive_chat_auto",
    },
  };
}

function isTerminalSample(summary) {
  if (summary.assertions.hasFirstTextDelta) {
    return true;
  }
  if (TERMINAL_RUN_STATUSES.has(String(summary.status.latestRunStatus || ""))) {
    return true;
  }
  if (
    TERMINAL_RUN_STATUSES.has(String(summary.status.latestTurnStatus || ""))
  ) {
    return true;
  }
  return false;
}

async function waitForSampleSummary(options, sessionId) {
  const startedAt = Date.now();
  let lastSummary = null;
  let pollCount = 0;
  let lastTraceKey = "";

  while (Date.now() - startedAt < options.timeoutMs) {
    const threadRead = await readAgentRuntimeThreadCurrent(options, sessionId);
    lastSummary = summarizeThreadRead(threadRead);
    pollCount += 1;
    const traceKey = [
      lastSummary.status?.latestRunStatus,
      lastSummary.status?.latestTurnStatus,
      lastSummary.timing?.firstTextDeltaMs,
    ].join(":");
    if (
      options.trace &&
      (traceKey !== lastTraceKey ||
        lastSummary.assertions?.hasFirstTextDelta ||
        pollCount % 10 === 0)
    ) {
      lastTraceKey = traceKey;
      trace(options, "poll-thread", {
        elapsedMs: Date.now() - startedAt,
        poll: pollCount,
        turnStatus: lastSummary.status?.latestTurnStatus,
        runStatus: lastSummary.status?.latestRunStatus,
        firstVisibleDeltaMs: lastSummary.timing?.firstVisibleDeltaMs,
        firstTextDeltaMs: lastSummary.timing?.firstTextDeltaMs,
      });
    }
    if (isTerminalSample(lastSummary)) {
      return lastSummary;
    }
    await sleep(options.intervalMs);
  }

  return {
    ...(lastSummary || summarizeThreadRead(null)),
    timedOut: true,
  };
}

async function runSample(options, workspaceId, sampleIndex) {
  const sampleStartedAt = Date.now();
  const stamp = `${Date.now()}-${process.pid}-${sampleIndex}`;
  trace(options, "sample-start", {
    sampleIndex,
    mode: options.mode,
    provider: options.providerPreference,
    model: options.modelPreference,
  });
  const sessionId = await createAgentSessionCurrent(options, {
    workspaceId,
    title: `AgentUI TTFT live sample ${options.mode} ${stamp}`,
    metadata: {
      harness: {
        hiddenFromUserRecents: true,
        source: "sample:agentui-ttft-live",
        mode: options.mode,
      },
    },
  });
  trace(options, "session-created", {
    sampleIndex,
    elapsedMs: Date.now() - sampleStartedAt,
  });
  const turnId = `agentui-ttft-${options.mode}-${stamp}`;
  const eventName = `agent_stream_${sessionId}_${turnId}`;

  await startAgentSessionTurnCurrent(options, {
    message: options.message,
    sessionId,
    workspaceId,
    eventName,
    turnId,
    runtimeRequest: buildRuntimeRequest(options),
    skipPreSubmitResume: true,
  });
  trace(options, "turn-started", {
    sampleIndex,
    elapsedMs: Date.now() - sampleStartedAt,
  });

  const summary = await waitForSampleSummary(options, sessionId);
  trace(options, "sample-complete", {
    sampleIndex,
    elapsedMs: Date.now() - sampleStartedAt,
    turnStatus: summary.status?.latestTurnStatus,
    runStatus: summary.status?.latestRunStatus,
    firstTextDeltaMs: summary.timing?.firstTextDeltaMs,
  });
  return {
    sampleIndex,
    sessionId,
    turnId,
    evidenceEventCount: 0,
    firstTextDeltaAt: null,
    firstTextDeltaMs: summary.timing?.firstTextDeltaMs ?? null,
    ...summary,
  };
}

function buildSummary(samples) {
  return {
    samples: samples.length,
    firstTextSamples: samples.filter(
      (sample) => sample.assertions?.hasFirstTextDelta,
    ).length,
    responsiveAutoSamples: samples.filter(
      (sample) => sample.assertions?.isResponsiveChatAuto,
    ).length,
    routingEvidenceSamples: samples.filter(
      (sample) => sample.assertions?.hasRoutingEvidence,
    ).length,
    selectedGroups: samples.map((sample) => ({
      decisionSource: sample.routing?.decisionSource || null,
      provider: sample.routing?.selectedProvider || null,
      model: sample.routing?.selectedModel || null,
      firstTextDeltaMs: sample.timing?.firstTextDeltaMs ?? null,
      runStatus: sample.status?.latestRunStatus || null,
    })),
  };
}

function writeOutput(options, payload) {
  const text = `${JSON.stringify(payload, null, 2)}\n`;
  if (!options.outputPath) {
    process.stdout.write(text);
    return;
  }
  fs.mkdirSync(path.dirname(options.outputPath), { recursive: true });
  fs.writeFileSync(options.outputPath, text);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  assertLiveProviderSmokeAllowed({
    allowed: options.allowLiveProvider,
    scriptName: "scripts/agentui-ttft-live-sample.mjs",
  });
  trace(options, "start", {
    mode: options.mode,
    samples: options.samples,
  });
  const health = await waitForHealth(options);
  const workspace = await invoke(options, "get_or_create_default_project", {});
  const workspaceId = workspace?.id;
  if (!workspaceId) {
    throw new Error("get_or_create_default_project 缺少 workspace id");
  }
  trace(options, "workspace-ready", {
    workspaceId,
  });

  const samples = [];
  for (let index = 0; index < options.samples; index += 1) {
    samples.push(await runSample(options, workspaceId, index + 1));
  }

  writeOutput(options, {
    schemaVersion: "v1",
    generatedAt: new Date().toISOString(),
    mode: options.mode,
    health: {
      status: health?.status || null,
      service: health?.service || null,
      version: health?.version || null,
    },
    workspaceId,
    summary: buildSummary(samples),
    samples,
    privacy: {
      exportedFields: [
        "session/turn id",
        "routing decision",
        "first visible/thinking/text TTFT",
        "status summary",
      ],
      omittedFields: [
        "prompt",
        "assistant response",
        "error_message",
        "secrets",
        "run id",
      ],
    },
  });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
