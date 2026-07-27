#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import {
  buildApprovalSandboxSmokeEvidence,
  renderApprovalSandboxTranscriptLines,
} from "../lib/agent-runtime-approval-sandbox-smoke-core.mjs";
import {
  createAgentSessionCurrent,
  readAgentRuntimeThreadCurrent,
  respondAgentServerRequestCurrent,
  startAgentSessionTurnCurrent,
} from "../lib/agent-runtime-smoke-core.mjs";
import {
  assertLiveProviderSmokeAllowed,
  liveProviderSmokeAllowed,
} from "../lib/live-provider-smoke-gate.mjs";
import { runVitestSmoke } from "../lib/vitest-smoke-runner.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "../..");
const DEFAULT_OUTPUT = path.join(
  rootDir,
  ".lime/qc/runtime-approval-sandbox-smoke.json",
);
const DEFAULT_HEALTH_URL = "http://127.0.0.1:3030/health";
const DEFAULT_INVOKE_URL = "http://127.0.0.1:3030/invoke";
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_INTERVAL_MS = 1_000;
const DEFAULT_PROVIDER_PREFERENCE =
  process.env.LIME_AGENT_QC_PROVIDER ||
  process.env.LIME_E2E_PROVIDER ||
  process.env.LIME_DEFAULT_PROVIDER ||
  "";
const DEFAULT_MODEL_PREFERENCE =
  process.env.LIME_AGENT_QC_MODEL ||
  process.env.LIME_E2E_MODEL ||
  process.env.LIME_DEFAULT_MODEL ||
  "";
const APPROVAL_POLICY = "on-request";
const SANDBOX_POLICY = "workspace-write";
const PROVIDER_PICK_ORDER = ["deepseek", "doubao", "lime-hub"];
const APP_SERVER_HANDLE_JSON_LINES_COMMAND = "app_server_handle_json_lines";
const APP_SERVER_METHOD_MODEL_PROVIDER_LIST = "modelProvider/list";
const APP_SERVER_METHOD_MODEL_PROVIDER_READ = "modelProvider/read";

function printHelp() {
  console.log(`
Lime Agent Runtime Approval / Sandbox Smoke

用途:
  生成 tool / approval / sandbox 边界的确定性 smoke 证据，补齐 qcloop P0 场景的可审查摘要。

用法:
  node scripts/agent-runtime/approval-sandbox-smoke.mjs [选项]

选项:
  --output <path>   写入 evidence JSON，默认 ./.lime/qc/runtime-approval-sandbox-smoke.json
  --health-url <url> DevBridge health 地址，默认 ${DEFAULT_HEALTH_URL}
  --invoke-url <url> DevBridge invoke 地址，默认 ${DEFAULT_INVOKE_URL}
  --timeout-ms <ms>  live runtime 等待超时，默认 ${DEFAULT_TIMEOUT_MS}
  --interval-ms <ms> live runtime 轮询间隔，默认 ${DEFAULT_INTERVAL_MS}
  --provider-preference <id> live runtime 使用的 provider 偏好；默认读取 LIME_AGENT_QC_PROVIDER / LIME_E2E_PROVIDER，未设置时自动选本地启用 provider
  --model-preference <name>  live runtime 使用的 model 偏好；默认读取 LIME_AGENT_QC_MODEL / LIME_E2E_MODEL，未设置时自动选 provider 的首个自定义模型
  --devbridge-denied-runtime
                      采集真实 DevBridge denied-only runtime 权限确认 transcript；只拒绝权限，不继续模型执行，不要求真实 Provider
  --allow-live-provider / --live-runtime
                      确认允许调用真实模型 Provider 并采集 live runtime transcript；默认仅生成确定性投影摘要
  --skip-live-runtime 跳过 DevBridge live runtime transcript，仅生成确定性投影摘要
  --no-write        只运行校验并打印摘要，不写 evidence JSON
  -h, --help        显示帮助
`);
}

function parseArgs(argv) {
  const options = {
    output: DEFAULT_OUTPUT,
    healthUrl: DEFAULT_HEALTH_URL,
    invokeUrl: DEFAULT_INVOKE_URL,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    intervalMs: DEFAULT_INTERVAL_MS,
    providerPreference: DEFAULT_PROVIDER_PREFERENCE,
    modelPreference: DEFAULT_MODEL_PREFERENCE,
    devBridgeDeniedRuntime: false,
    liveRuntime: liveProviderSmokeAllowed(),
    liveProviderExplicitlyAllowed: false,
    write: true,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--output" && argv[index + 1]) {
      options.output = path.resolve(rootDir, String(argv[index + 1]));
      index += 1;
      continue;
    }
    if (arg === "--no-write") {
      options.write = false;
      continue;
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
      options.timeoutMs = Number(argv[index + 1]);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && argv[index + 1]) {
      options.intervalMs = Number(argv[index + 1]);
      index += 1;
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
      continue;
    }
    if (arg === "--skip-live-runtime" || arg === "--no-live-runtime") {
      options.liveRuntime = false;
      continue;
    }
    if (arg === "--devbridge-denied-runtime") {
      options.devBridgeDeniedRuntime = true;
      continue;
    }
    if (arg === "--allow-live-provider" || arg === "--live-runtime") {
      options.liveRuntime = true;
      options.liveProviderExplicitlyAllowed = true;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 10_000) {
    throw new Error("--timeout-ms 必须是 >= 10000 的数字");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms 必须是 >= 100 的数字");
  }

  return options;
}

function runVitest(label, args) {
  return runVitestSmoke({
    rootDir,
    label,
    args,
    logPrefix: "smoke:agent-runtime-approval-sandbox",
  });
}

function writeEvidence(outputPath, evidence) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(evidence, null, 2)}\n`);
  return outputPath;
}

function writeEvidenceWithFallback(outputPath, evidence) {
  try {
    return writeEvidence(outputPath, evidence);
  } catch (error) {
    if (outputPath !== DEFAULT_OUTPUT) {
      throw error;
    }
    const fallbackPath = path.join(
      os.tmpdir(),
      "lime-runtime-approval-sandbox-smoke.json",
    );
    const writtenPath = writeEvidence(fallbackPath, evidence);
    console.warn(
      `[smoke:agent-runtime-approval-sandbox] 默认 evidence 写入失败，已回退到 ${writtenPath}: ${error.message}`,
    );
    return writtenPath;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
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
      const payload = text ? JSON.parse(text) : {};
      console.log(
        `[smoke:agent-runtime-approval-sandbox] DevBridge 已就绪 (${Date.now() - startedAt}ms)${
          payload?.status ? ` status=${payload.status}` : ""
        }`,
      );
      return payload;
    } catch (error) {
      lastError = error;
      await sleep(options.intervalMs);
    }
  }

  const detail =
    lastError instanceof Error
      ? lastError.message
      : String(lastError || "unknown error");
  throw new Error(
    `[smoke:agent-runtime-approval-sandbox] DevBridge 未就绪，无法采集 live runtime transcript: ${detail}`,
  );
}

async function invoke(options, cmd, args, timeoutMs = options.timeoutMs) {
  const response = await fetch(options.invokeUrl, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({ cmd, args }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${cmd} HTTP ${response.status}: ${text}`);
  }
  const payload = text ? JSON.parse(text) : null;
  if (payload?.error) {
    throw new Error(`${cmd} error: ${payload.error}`);
  }
  return payload?.result;
}

let appServerRequestId = 1;

async function invokeAppServerMethod(
  options,
  method,
  params,
  timeoutMs = options.timeoutMs,
) {
  const id = `approval-sandbox-${appServerRequestId++}`;
  const request =
    params === undefined ? { id, method } : { id, method, params };
  const result = await invoke(
    options,
    APP_SERVER_HANDLE_JSON_LINES_COMMAND,
    { request: { lines: [`${JSON.stringify(request)}\n`] } },
    timeoutMs,
  );
  const messages = (Array.isArray(result?.lines) ? result.lines : [])
    .map((line) => {
      try {
        return JSON.parse(String(line));
      } catch {
        return null;
      }
    })
    .filter(Boolean);
  const error = messages.find((message) => message.id === id && message.error);
  if (error) {
    throw new Error(
      `${method} error: ${error.error?.message || "App Server JSON-RPC error"}`,
    );
  }
  const response = messages.find(
    (message) => message.id === id && Object.hasOwn(message, "result"),
  );
  if (!response) {
    throw new Error(`${method} missing App Server response`);
  }
  return response.result;
}

function normalizeProviderId(provider) {
  return String(
    provider?.id || provider?.provider_id || provider?.providerId || "",
  ).trim();
}

function providerEnabled(provider) {
  return provider?.enabled !== false;
}

function pickModelPreference(provider) {
  const candidates = [
    ...(Array.isArray(provider?.models) ? provider.models : []),
  ]
    .map((value) =>
      typeof value === "string"
        ? value
        : String(value?.name || value?.id || value?.model || "").trim(),
    )
    .filter(Boolean);

  return (
    candidates.find((value) => /flash|mini|lite/i.test(value)) ||
    candidates[0] ||
    ""
  );
}

function pickProvider(providers, preferredProviderId) {
  const enabled = providers.filter((provider) => providerEnabled(provider));
  if (preferredProviderId) {
    return (
      enabled.find(
        (provider) => normalizeProviderId(provider) === preferredProviderId,
      ) ||
      providers.find(
        (provider) => normalizeProviderId(provider) === preferredProviderId,
      ) ||
      null
    );
  }

  for (const providerId of PROVIDER_PICK_ORDER) {
    const match = enabled.find(
      (provider) => normalizeProviderId(provider) === providerId,
    );
    if (match) {
      return match;
    }
  }

  return enabled[0] || null;
}

async function resolveProviderPreference(options) {
  const explicitProvider = String(options.providerPreference || "").trim();
  const explicitModel = String(options.modelPreference || "").trim();
  if (explicitProvider && explicitModel) {
    return {
      providerPreference: explicitProvider,
      modelPreference: explicitModel,
      source: "explicit",
    };
  }

  const providerList = await invokeAppServerMethod(
    options,
    APP_SERVER_METHOD_MODEL_PROVIDER_LIST,
    {},
    30_000,
  );
  const providers = providerList?.providers;
  const selected = pickProvider(
    Array.isArray(providers) ? providers : [],
    explicitProvider,
  );
  const providerId = normalizeProviderId(selected);
  if (!providerId) {
    throw new Error(
      "[smoke:agent-runtime-approval-sandbox] 未找到可用 provider；请传 --provider-preference / --model-preference 或先配置本地 provider",
    );
  }

  let providerDetail = selected;
  try {
    providerDetail =
      (
        await invokeAppServerMethod(
          options,
          APP_SERVER_METHOD_MODEL_PROVIDER_READ,
          { providerId },
          30_000,
        )
      )?.provider || selected;
  } catch (error) {
    console.warn(
      `[smoke:agent-runtime-approval-sandbox] 读取 provider 详情失败，使用列表摘要继续: ${error.message}`,
    );
  }

  const modelPreference = explicitModel || pickModelPreference(providerDetail);
  if (!modelPreference) {
    throw new Error(
      `[smoke:agent-runtime-approval-sandbox] provider ${providerId} 缺少可用模型；请传 --model-preference`,
    );
  }

  return {
    providerPreference: providerId,
    modelPreference,
    source:
      explicitProvider || explicitModel
        ? "partial-explicit"
        : "auto-enabled-provider",
  };
}

function pickPermissionState(threadRead) {
  return threadRead?.permission_state || threadRead?.permissionState || {};
}

function pendingRequests(threadRead) {
  return threadRead?.pending_requests || threadRead?.pendingRequests || [];
}

function latestTurnStatus(threadRead) {
  return (
    threadRead?.diagnostics?.latest_turn_status ||
    threadRead?.diagnostics?.latestTurnStatus ||
    threadRead?.runtime_summary?.latestTurnStatus ||
    threadRead?.runtimeSummary?.latestTurnStatus ||
    threadRead?.status ||
    null
  );
}

function permissionField(permissionState, snakeKey, camelKey) {
  return permissionState?.[snakeKey] ?? permissionState?.[camelKey] ?? null;
}

function summarizeThreadRead(threadRead) {
  const permissionState = pickPermissionState(threadRead);
  const requests = pendingRequests(threadRead);
  return {
    threadStatus: threadRead?.status || null,
    latestTurnStatus: latestTurnStatus(threadRead),
    primaryBlockingKind:
      threadRead?.diagnostics?.primary_blocking_kind ||
      threadRead?.diagnostics?.primaryBlockingKind ||
      null,
    primaryBlockingSummary:
      threadRead?.diagnostics?.primary_blocking_summary ||
      threadRead?.diagnostics?.primaryBlockingSummary ||
      null,
    pendingRequestCount: requests.length,
    permissionStatus: permissionField(permissionState, "status", "status"),
    confirmationStatus: permissionField(
      permissionState,
      "confirmation_status",
      "confirmationStatus",
    ),
    confirmationRequestId: permissionField(
      permissionState,
      "confirmation_request_id",
      "confirmationRequestId",
    ),
    confirmationSource: permissionField(
      permissionState,
      "confirmation_source",
      "confirmationSource",
    ),
    askProfileKeys:
      permissionField(permissionState, "ask_profile_keys", "askProfileKeys") ||
      [],
    requiredProfileKeys:
      permissionField(
        permissionState,
        "required_profile_keys",
        "requiredProfileKeys",
      ) || [],
  };
}

async function waitForThreadRead(options, sessionId, predicate, label) {
  const startedAt = Date.now();
  let lastSummary = null;

  while (Date.now() - startedAt < options.timeoutMs) {
    const threadRead = await readAgentRuntimeThreadCurrent(options, sessionId);
    lastSummary = summarizeThreadRead(threadRead);
    if (predicate(threadRead, lastSummary)) {
      return { threadRead, summary: lastSummary };
    }
    await sleep(options.intervalMs);
  }

  throw new Error(
    `[smoke:agent-runtime-approval-sandbox] ${label} 超时，最后线程摘要: ${JSON.stringify(lastSummary)}`,
  );
}

function buildRuntimeContractMetadata() {
  return {
    harness: {
      browser_assist: {
        runtime_contract: {
          contract_key: "browser_control",
          routing_slot: "browser_reasoning_model",
          execution_profile: {
            profile_key: "browser_control_profile",
          },
          executor_adapter: {
            adapter_key: "browser:browser_assist",
          },
          executor_binding: {
            executor_kind: "browser",
            binding_key: "browser_assist",
          },
        },
      },
    },
  };
}

async function runPermissionDecisionFlow(
  options,
  workspaceId,
  providerPreference,
  decision,
) {
  const confirmed = decision === "resolved";
  const responseLabel = confirmed ? "允许本次执行" : "拒绝";
  const submittedStrategy = "react";
  const sessionId = await createAgentSessionCurrent(options, {
    workspaceId,
    title: `Agent QC approval ${decision} ${Date.now()}`,
    executionStrategy: submittedStrategy,
    metadata: {
      harness: {
        hiddenFromUserRecents: true,
        source: "smoke:agent-runtime-approval-sandbox",
        decision,
      },
    },
  });
  const turnId = `qc-approval-${decision}-${Date.now()}-${process.pid}`;
  const eventName = `agent_stream_${sessionId}_${turnId}`;
  const submittedPolicies = {
    approvalPolicy: APPROVAL_POLICY,
    sandboxPolicy: SANDBOX_POLICY,
  };

  const runtimeRequest = {
    executionStrategy: submittedStrategy,
    approvalPolicy: APPROVAL_POLICY,
    sandboxPolicy: SANDBOX_POLICY,
    metadata: buildRuntimeContractMetadata(),
  };
  if (providerPreference?.providerPreference) {
    runtimeRequest.providerPreference = providerPreference.providerPreference;
  }
  if (providerPreference?.modelPreference) {
    runtimeRequest.modelPreference = providerPreference.modelPreference;
  }

  await startAgentSessionTurnCurrent(options, {
    message:
      "Agent QC runtime permission confirmation smoke：应在模型执行前创建真实权限确认请求。",
    sessionId,
    workspaceId,
    eventName,
    turnId,
    runtimeRequest,
    skipPreSubmitResume: true,
  });

  const requested = await waitForThreadRead(
    options,
    sessionId,
    (_threadRead, summary) =>
      summary.confirmationStatus === "requested" &&
      summary.pendingRequestCount > 0 &&
      String(summary.confirmationRequestId || "").includes(turnId),
    `等待 ${decision} flow 进入权限确认 requested`,
  );

  const requestId = requested.summary.confirmationRequestId;
  assert(requestId, `${decision} flow 缺少 confirmationRequestId`);

  await respondAgentServerRequestCurrent(options, {
    requestId,
    confirmed,
    userData: { answer: responseLabel },
    actionScope: {
      threadId: sessionId,
      turnId,
    },
  });

  const completed = await waitForThreadRead(
    options,
    sessionId,
    (_threadRead, summary) =>
      summary.pendingRequestCount === 0 &&
      summary.confirmationStatus === decision,
    `等待 ${decision} flow 写回确认结果`,
  );

  return {
    decision: confirmed ? "resolved" : "denied",
    sessionId,
    turnId,
    requestId,
    providerPreference,
    submittedStrategy,
    submittedPolicies,
    before: requested.summary,
    respond: {
      confirmed,
      responseLabel,
    },
    after: completed.summary,
  };
}

async function collectDevBridgeDeniedRuntimeTranscript(options) {
  const health = await waitForHealth(options);
  const workspace = await invoke(options, "get_or_create_default_project");
  const workspaceId = workspace?.id;
  assert(workspaceId, "get_or_create_default_project 缺少 workspace id");

  const deniedFlow = await runPermissionDecisionFlow(
    options,
    workspaceId,
    null,
    "denied",
  );
  const flows = [deniedFlow];

  return {
    kind: "devbridge-runtime-permission-confirmation-denied-only",
    health,
    workspaceId,
    providerPreference: null,
    flows,
    assertions: {
      devBridgeHealthy: health?.status === "ok" || Boolean(health),
      permissionRequestCreatedBeforeModel: flows.every(
        (flow) =>
          flow.before.permissionStatus === "requires_confirmation" &&
          flow.before.confirmationStatus === "requested" &&
          flow.before.pendingRequestCount > 0 &&
          flow.before.latestTurnStatus === "waitingAction" &&
          String(flow.before.confirmationRequestId || "").includes(flow.turnId),
      ),
      deniedDecisionClearsPendingRequest:
        deniedFlow.after.confirmationStatus === "denied" &&
        deniedFlow.after.pendingRequestCount === 0,
      approvalPolicySubmitted: flows.every(
        (flow) => flow.submittedPolicies.approvalPolicy === APPROVAL_POLICY,
      ),
      sandboxPolicySubmitted: flows.every(
        (flow) => flow.submittedPolicies.sandboxPolicy === SANDBOX_POLICY,
      ),
      reactRuntimeSubmitted: flows.every(
        (flow) => flow.submittedStrategy === "react",
      ),
      providerNotRequired: flows.every((flow) => !flow.providerPreference),
    },
  };
}

async function collectLiveRuntimeTranscript(options) {
  const health = await waitForHealth(options);
  const providerPreference = await resolveProviderPreference(options);
  console.log(
    `[smoke:agent-runtime-approval-sandbox] live runtime provider: provider=${providerPreference.providerPreference} model=${providerPreference.modelPreference} source=${providerPreference.source}`,
  );
  const workspace = await invoke(options, "get_or_create_default_project");
  const workspaceId = workspace?.id;
  assert(workspaceId, "get_or_create_default_project 缺少 workspace id");

  const deniedFlow = await runPermissionDecisionFlow(
    options,
    workspaceId,
    providerPreference,
    "denied",
  );
  const resolvedFlow = await runPermissionDecisionFlow(
    options,
    workspaceId,
    providerPreference,
    "resolved",
  );
  const flows = [deniedFlow, resolvedFlow];

  return {
    kind: "devbridge-runtime-permission-confirmation",
    health,
    workspaceId,
    providerPreference,
    flows,
    assertions: {
      devBridgeHealthy: health?.status === "ok" || Boolean(health),
      permissionRequestCreatedBeforeModel: flows.every(
        (flow) =>
          flow.before.permissionStatus === "requires_confirmation" &&
          flow.before.confirmationStatus === "requested" &&
          flow.before.pendingRequestCount > 0 &&
          flow.before.latestTurnStatus === "waitingAction" &&
          String(flow.before.confirmationRequestId || "").includes(flow.turnId),
      ),
      deniedDecisionClearsPendingRequest:
        deniedFlow.after.confirmationStatus === "denied" &&
        deniedFlow.after.pendingRequestCount === 0,
      resolvedDecisionClearsPendingRequest:
        resolvedFlow.after.confirmationStatus === "resolved" &&
        resolvedFlow.after.pendingRequestCount === 0,
      approvalPolicySubmitted: flows.every(
        (flow) => flow.submittedPolicies.approvalPolicy === APPROVAL_POLICY,
      ),
      sandboxPolicySubmitted: flows.every(
        (flow) => flow.submittedPolicies.sandboxPolicy === SANDBOX_POLICY,
      ),
    },
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const commandResults = [];

  commandResults.push(
    runVitest("submit preferences 应透传 approval / sandbox policy", [
      "src/components/agent/chat/utils/buildUserInputSubmitOp.test.ts",
      "src/components/agent/chat/hooks/agentStreamUserInputSubmission.test.ts",
      "src/components/agent/chat/hooks/agentStreamSubmitExecution.test.ts",
      "src/components/agent/chat/hooks/agentStreamSubmitOpController.test.ts",
    ]),
  );

  commandResults.push(
    runVitest("runtime projection 应保留 permission / tool lifecycle", [
      "src/components/agent/chat/projection/agentUiEventProjection.test.ts",
      "-t",
      "runtime permission metadata|工具输入、成功输出和失败输出|plan approval",
    ]),
  );

  commandResults.push(
    runVitest("消息与时间线不应把权限确认误渲染为失败", [
      "src/components/agent/chat/components/AgentThreadTimeline.test.tsx",
      "src/components/agent/chat/components/MessageList.test.tsx",
      "-t",
      "运行时权限确认",
    ]),
  );

  commandResults.push(
    runVitest("Harness 面板应展示工具权限、待审批与 sandbox 来源", [
      "src/components/agent/chat/components/HarnessStatusPanel.test.tsx",
      "-t",
      "存在工具库存时应展示工具与权限区块及来源统计|待审批区块应通过 artifact protocol",
    ]),
  );

  if (options.liveRuntime) {
    assertLiveProviderSmokeAllowed({
      allowed:
        liveProviderSmokeAllowed() || options.liveProviderExplicitlyAllowed,
      scriptName: "smoke:agent-runtime-approval-sandbox",
      flag: "--allow-live-provider",
    });
  } else {
    console.log(
      "[smoke:agent-runtime-approval-sandbox] 默认跳过 live runtime transcript，避免调用真实模型 Provider；需要时请显式传入 --allow-live-provider。",
    );
  }

  const liveRuntimeTranscript = options.liveRuntime
    ? await collectLiveRuntimeTranscript(options)
    : null;
  const devBridgeDeniedRuntimeTranscript = options.devBridgeDeniedRuntime
    ? await collectDevBridgeDeniedRuntimeTranscript(options)
    : null;

  const evidence = buildApprovalSandboxSmokeEvidence({
    commandResults,
    generatedAt: new Date().toISOString(),
    devBridgeDeniedRuntimeTranscript,
    liveRuntimeTranscript,
  });

  if (
    options.devBridgeDeniedRuntime &&
    !evidence.coverage.devBridgeDeniedRuntimeTranscript
  ) {
    throw new Error(
      "[smoke:agent-runtime-approval-sandbox] DevBridge denied-only runtime transcript 未满足门禁断言",
    );
  }

  if (options.liveRuntime && !evidence.coverage.liveRuntimeTranscript) {
    throw new Error(
      "[smoke:agent-runtime-approval-sandbox] live runtime transcript 未满足发布门禁断言",
    );
  }

  for (const line of renderApprovalSandboxTranscriptLines(evidence)) {
    console.log(line);
  }

  if (options.write) {
    const evidencePath = writeEvidenceWithFallback(options.output, evidence);
    console.log(
      `\n[smoke:agent-runtime-approval-sandbox] evidence: ${evidencePath}`,
    );
  }

  console.log("\n[smoke:agent-runtime-approval-sandbox] 通过");
}

main().catch((error) => {
  console.error(
    error instanceof Error ? error.message : String(error || "unknown error"),
  );
  process.exit(1);
});
