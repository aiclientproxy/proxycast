import { describe, expect, it } from "vitest";

import type { AgentRuntimeThreadReadModel } from "@/lib/api/agentRuntime/sessionTypes";
import {
  buildRuntimeRoutingEvidenceLines,
  formatDiagnosticDurationMs,
  resolveRuntimeRoutingEvidence,
  type RuntimeRoutingEvidenceLineText,
} from "./runtimeRoutingEvidence";

const EN_LINE_TEXT: RuntimeRoutingEvidenceLineText = {
  appliedFallback: "Automatic fallback",
  decisionReason: "Decision reason",
  decisionSource: "Decision source",
  durationUnknown: "Duration unknown",
  evidenceSource: "Evidence source",
  fallbackChain: "Fallback chain",
  firstText: "First text",
  firstThinking: "First thinking",
  firstTokenMetrics: "First-token metrics",
  firstVisible: "First visible",
  none: "None",
  network: "Network rule",
  networkDecision: "Network decision",
  policy: "Policy decision",
  policyFailure: "Policy failure",
  policySources: "Policy sources",
  sandbox: "Sandbox policy",
  sandboxBackend: "Sandbox backend",
  policyProfile: "Policy profile",
  policySummary: "Policy summary",
  policyTitle: "Policy and network facts",
  policyOpenSettings: "Open execution policy settings",
  requestedModel: "Requested model",
  run: "Run",
  runUnknown: "Run unknown",
  selectedModel: "Response model",
  serviceModelSlot: "Service slot",
  settingsSource: "Settings source",
  providerReadiness: "Provider readiness",
  providerReadinessKeys: "Provider keys",
  providerReadinessProviderType: "Provider type",
  providerReadinessRecovery: "Recovery action",
  routingAttempts: "Routing attempts",
  modelRegistry: "Model registry facts",
  modelRegistryAlias: "Model alias",
  modelRegistryCapabilities: "Model capabilities",
  modelRegistryReasoning: "Reasoning support",
  unknown: "Unknown",
  unset: "Unset",
};

function threadRead(
  value: Partial<AgentRuntimeThreadReadModel>,
): AgentRuntimeThreadReadModel {
  return value as AgentRuntimeThreadReadModel;
}

describe("runtimeRoutingEvidence", () => {
  it("应从 latestModelDeltaTiming.routing 读取自动回退原因", () => {
    const evidence = resolveRuntimeRoutingEvidence(
      threadRead({
        model_routing: {
          latestModelDeltaTiming: {
            source: "agent_runs.metadata",
            runStatus: "success",
            durationMs: 1300,
            firstVisibleDeltaMs: "842",
            firstThinkingDeltaMs: 842,
            firstTextDeltaMs: 1299,
            routing: {
              decisionSource: "responsive_chat_auto",
              decisionReason:
                "service_models.responsive_chat 历史样本不满足低延迟目标。",
              fallbackChain: [
                "deepseek:deepseek-v4-pro",
                "deepseek:deepseek-v4-flash",
              ],
              serviceModelSlot: "responsive_chat",
              selectedProvider: "deepseek",
              selectedModel: "deepseek-v4-flash",
            },
          },
        },
      }),
    );

    expect(evidence).toMatchObject({
      shouldRender: true,
      decisionSource: "responsive_chat_auto",
      serviceModelSlot: "responsive_chat",
      selectedProvider: "deepseek",
      selectedModel: "deepseek-v4-flash",
      firstVisibleDeltaMs: 842,
      firstThinkingDeltaMs: 842,
      firstTextDeltaMs: 1299,
      runDurationMs: 1300,
      runStatus: "success",
      timingSource: "agent_runs.metadata",
    });
    expect(evidence.decisionReason).toContain("低延迟目标");
    expect(evidence.fallbackChain).toEqual([
      "deepseek:deepseek-v4-pro",
      "deepseek:deepseek-v4-flash",
    ]);
  });

  it("应优先使用 thread_read 顶层路由事实并生成复制文本", () => {
    const evidence = resolveRuntimeRoutingEvidence(
      threadRead({
        decision_source: "session_default",
        service_model_slot: "responsive_chat",
        model_routing: {
          decisionSource: "responsive_chat_auto",
          selectedProvider: "deepseek",
          selectedModel: "deepseek-v4-flash",
          latestModelDeltaTiming: {
            source: "agent_runs.metadata",
            firstTextDeltaMs: 1377,
            routing: {
              decisionSource: "request_override",
              selectedProvider: "other",
              selectedModel: "slow-model",
            },
          },
        },
      }),
    );

    expect(evidence.decisionSource).toBe("responsive_chat_auto");
    expect(evidence.serviceModelSlot).toBe("responsive_chat");
    expect(evidence.selectedProvider).toBe("deepseek");
    expect(evidence.selectedModel).toBe("deepseek-v4-flash");

    const lines = buildRuntimeRoutingEvidenceLines(evidence, EN_LINE_TEXT);
    expect(lines.join("\n")).toContain("Decision source: responsive_chat_auto");
    expect(lines.join("\n")).toContain(
      "Response model: deepseek/deepseek-v4-flash",
    );
    expect(lines.join("\n")).toContain("First text=1.38s");
    expect(lines.join("\n")).toContain("Evidence source: agent_runs.metadata");
  });

  it("应从 thread_read.model_routing 解析模型注册事实并生成复制文本", () => {
    const evidence = resolveRuntimeRoutingEvidence(
      threadRead({
        model_routing: {
          serviceModelSlot: "coding",
          decisionSource: "coding_profile_slot",
          selectedProvider: "custom-coder",
          selectedModel: "coder-large",
          modelRegistry: {
            source: "provider_declared_model",
            status: "matched",
            reasonCode: "matched_provider_models",
            matchedModelId: "coder-large",
            modelCapabilities: {
              capabilities: {
                tools: true,
                streaming: true,
                reasoning: true,
                image: false,
              },
              taskFamilies: ["chat", "reasoning"],
              runtimeFeatures: ["streaming", "tool_calling", "reasoning"],
            },
            modelAlias: {
              canonicalModelId: "coder-large",
              providerModelId: "provider/coder-large",
              aliasSource: "local",
            },
            reasoning: {
              supported: true,
              reasoningEffort: {
                supported: true,
                levels: ["low", "medium", "high"],
                default: "medium",
              },
            },
          },
        },
      }),
    );

    expect(evidence).toMatchObject({
      shouldRender: true,
      modelRegistrySource: "provider_declared_model",
      modelRegistryStatus: "matched",
      modelRegistryReason: "matched_provider_models",
      modelRegistryMatchedModel: "coder-large",
      modelRegistryCapabilityTags: [
        "tools",
        "streaming",
        "reasoning",
        "chat",
        "tool_calling",
      ],
      modelRegistryAlias:
        "canonical=coder-large · provider=provider/coder-large · source=local",
      modelRegistryReasoning:
        "supported=true · effort=true · levels=low/medium/high · default=medium",
    });

    const lines = buildRuntimeRoutingEvidenceLines(evidence, EN_LINE_TEXT);
    const text = lines.join("\n");
    expect(text).toContain(
      "Model registry facts: provider_declared_model · matched · matched_provider_models · coder-large",
    );
    expect(text).toContain(
      "Model capabilities: tools, streaming, reasoning, chat, tool_calling",
    );
    expect(text).toContain(
      "Model alias: canonical=coder-large · provider=provider/coder-large · source=local",
    );
    expect(text).toContain(
      "Reasoning support: supported=true · effort=true · levels=low/medium/high · default=medium",
    );
  });

  it("应从 thread_read.model_routing 解析 provider readiness 阻断事实", () => {
    const evidence = resolveRuntimeRoutingEvidence(
      threadRead({
        model_routing: {
          serviceModelSlot: "coding",
          decisionSource: "coding_profile_slot",
          selectedProvider: "custom-coder",
          selectedModel: "coder-large",
          providerReadiness: {
            ready: false,
            status: "needs_setup",
            source: "provider_store",
            reasonCode: "missing_enabled_api_key",
            providerType: "openai-compatible",
            enabled: true,
            enabledKeyCount: 0,
            totalKeyCount: 2,
          },
        },
      }),
    );

    expect(evidence).toMatchObject({
      shouldRender: true,
      providerReadinessSource: "provider_store",
      providerReadinessStatus: "needs_setup",
      providerReadinessReason: "missing_enabled_api_key",
      providerReadinessProviderType: "openai-compatible",
      providerReadinessKeySummary: "0/2",
      providerReadinessRecoveryAction: "add_enabled_api_key",
    });

    const text = buildRuntimeRoutingEvidenceLines(evidence, EN_LINE_TEXT).join(
      "\n",
    );
    expect(text).toContain(
      "Provider readiness: provider_store · needs_setup · missing_enabled_api_key · openai-compatible",
    );
    expect(text).toContain("Provider keys: 0/2");
    expect(text).toContain("Recovery action: add_enabled_api_key");
  });

  it("应解析 coding 槽位不可用后的自动回退尝试链", () => {
    const evidence = resolveRuntimeRoutingEvidence(
      threadRead({
        model_routing: {
          serviceModelSlot: "base",
          decisionSource: "profile_model_slot",
          selectedProvider: "openai",
          selectedModel: "gpt-4.1-mini",
          fallbackApplied: true,
          requestedSelection: {
            provider: "custom-coding",
            model: "coder-large",
            source: "profile_model_slot",
          },
          routingAttempts: [
            {
              slot: "coding",
              provider: "custom-coding",
              model: "coder-large",
              source: "profile_model_slot",
              providerReadiness: {
                status: "needs_setup",
                reasonCode: "missing_enabled_api_key",
              },
            },
            {
              slot: "base",
              provider: "openai",
              model: "gpt-4.1-mini",
              source: "profile_model_slot",
              providerReadiness: {
                status: "ready",
              },
            },
          ],
        },
      }),
    );

    expect(evidence).toMatchObject({
      shouldRender: true,
      fallbackApplied: true,
      requestedSelectionProvider: "custom-coding",
      requestedSelectionModel: "coder-large",
      requestedSelectionSource: "profile_model_slot",
      routingAttempts: [
        {
          slot: "coding",
          provider: "custom-coding",
          model: "coder-large",
          source: "profile_model_slot",
          providerReadinessStatus: "needs_setup",
          providerReadinessReason: "missing_enabled_api_key",
        },
        {
          slot: "base",
          provider: "openai",
          model: "gpt-4.1-mini",
          source: "profile_model_slot",
          providerReadinessStatus: "ready",
          providerReadinessReason: null,
        },
      ],
    });

    const text = buildRuntimeRoutingEvidenceLines(evidence, EN_LINE_TEXT).join(
      "\n",
    );
    expect(text).toContain("Automatic fallback: true");
    expect(text).toContain("Requested model: custom-coding/coder-large");
    expect(text).toContain(
      "Routing attempts: coding · custom-coding/coder-large · needs_setup · missing_enabled_api_key -> base · openai/gpt-4.1-mini · ready",
    );
  });

  it("应格式化毫秒和秒级耗时", () => {
    expect(formatDiagnosticDurationMs(842)).toBe("842ms");
    expect(formatDiagnosticDurationMs(1377)).toBe("1.38s");
    expect(formatDiagnosticDurationMs(12_345)).toBe("12.3s");
    expect(formatDiagnosticDurationMs(null)).toBeNull();
  });
});
