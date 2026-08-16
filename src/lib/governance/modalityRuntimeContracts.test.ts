import { describe, expect, it } from "vitest";
import {
  resolveBrowserControlRuntimeContractBinding,
  resolveBrowserControlEntrySource,
  isImageGenerationBoundEntrySource,
  resolveAudioTranscriptionRuntimeContractBinding,
  resolveImageGenerationRuntimeContractBinding,
  resolvePdfExtractRuntimeContractBinding,
  resolveTextTransformRuntimeContractBinding,
  resolveVoiceGenerationRuntimeContractBinding,
  resolveWebResearchRuntimeContractBinding,
} from "./modalityRuntimeContracts";

describe("modalityRuntimeContracts", () => {
  it("image_generation contract 应提供底层运行字段与上层入口绑定", () => {
    const contract = resolveImageGenerationRuntimeContractBinding();

    expect(contract).toMatchObject({
      contractKey: "image_generation",
      modality: "image",
      routingSlot: "image_generation_model",
      runtimeContract: expect.objectContaining({
        contract_key: "image_generation",
        routing_slot: "image_generation_model",
        limecore_policy_refs: [
          "model_catalog",
          "provider_offer",
          "tenant_feature_flags",
        ],
        limecore_policy_snapshot: expect.objectContaining({
          status: "local_defaults_evaluated",
          decision: "allow",
          source: "modality_runtime_contract",
          decision_source: "local_default_policy",
          decision_scope: "local_defaults_only",
          decision_reason: "declared_policy_refs_with_no_local_deny_rule",
          refs: ["model_catalog", "provider_offer", "tenant_feature_flags"],
          unresolved_refs: [
            "model_catalog",
            "provider_offer",
            "tenant_feature_flags",
          ],
          missing_inputs: [
            "model_catalog",
            "provider_offer",
            "tenant_feature_flags",
          ],
          policy_inputs: expect.arrayContaining([
            expect.objectContaining({
              ref_key: "model_catalog",
              status: "declared_only",
              value_source: "limecore_pending",
            }),
          ]),
          pending_hit_refs: [
            "model_catalog",
            "provider_offer",
            "tenant_feature_flags",
          ],
          policy_value_hits: [],
          policy_value_hit_count: 0,
        }),
      }),
    });
    expect(contract.limecorePolicyRefs).toEqual([
      "model_catalog",
      "provider_offer",
      "tenant_feature_flags",
    ]);
    expect(contract.requiredCapabilities).toEqual(
      expect.arrayContaining([
        "text_generation",
        "image_generation",
        "vision_input",
      ]),
    );
    expect(contract.boundEntrySources).toEqual(
      expect.arrayContaining(["at_image_command", "at_poster_command"]),
    );
  });

  it("image_generation entry source 只从 contract registry 判定", () => {
    expect(isImageGenerationBoundEntrySource("at_image_command")).toBe(true);
    expect(isImageGenerationBoundEntrySource("at_poster_command")).toBe(true);
    expect(isImageGenerationBoundEntrySource("at_video_command")).toBe(false);
  });

  it("policy value hit seam 应把真实命中值写回同一 snapshot 字段", () => {
    const contract = resolveImageGenerationRuntimeContractBinding({
      policyValueHits: [
        {
          ref_key: "model_catalog",
          status: "resolved",
          source: "limecore_policy_hit_resolver",
          value_source: "local_model_catalog",
          summary: "命中 gpt-image-1 的 image_generation capability",
          value: {
            model_id: "gpt-image-1",
            capability: "image_generation",
          },
        },
      ],
    });

    expect(contract.limecorePolicySnapshot).toMatchObject({
      evaluated_refs: ["model_catalog"],
      unresolved_refs: ["provider_offer", "tenant_feature_flags"],
      missing_inputs: ["provider_offer", "tenant_feature_flags"],
      pending_hit_refs: ["provider_offer", "tenant_feature_flags"],
      policy_value_hit_count: 1,
      policy_value_hits: [
        expect.objectContaining({
          ref_key: "model_catalog",
          status: "resolved",
          value_source: "local_model_catalog",
        }),
      ],
    });
    expect(contract.limecorePolicySnapshot.policy_inputs).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          ref_key: "model_catalog",
          status: "resolved",
          value_source: "local_model_catalog",
        }),
        expect.objectContaining({
          ref_key: "provider_offer",
          status: "declared_only",
          value_source: "limecore_pending",
        }),
      ]),
    );
    expect(contract.runtimeContract.limecore_policy_snapshot).toBe(
      contract.limecorePolicySnapshot,
    );
  });

  it("policy evaluator seam 应在所有输入命中后给出 resolved allow 决策", () => {
    const contract = resolveBrowserControlRuntimeContractBinding({
      policyValueHits: [
        {
          ref_key: "tenant_feature_flags",
          status: "resolved",
          source: "harness_tenant_feature_flags",
          value_source: "oem_cloud_bootstrap_features",
          value: {
            tenant_id: "tenant-1",
            flags: {
              gatewayEnabled: true,
            },
          },
        },
        {
          ref_key: "gateway_policy",
          status: "resolved",
          source: "harness_oem_routing",
          value_source: "request_oem_routing",
          value: {
            tenant_id: "tenant-1",
            can_invoke: true,
            quota_low: false,
          },
        },
      ],
    });

    expect(contract.limecorePolicySnapshot).toMatchObject({
      status: "policy_inputs_evaluated",
      decision: "allow",
      decision_source: "policy_input_evaluator",
      decision_scope: "resolved_policy_inputs",
      decision_reason: "resolved_policy_inputs_with_no_deny_or_ask_signal",
      missing_inputs: [],
      pending_hit_refs: [],
      policy_evaluation: expect.objectContaining({
        status: "evaluated",
        decision: "allow",
        blocking_refs: [],
        ask_refs: [],
        pending_refs: [],
      }),
    });
  });

  it("policy evaluator seam 应把 gateway deny 信号转成可审计 deny 决策", () => {
    const contract = resolveBrowserControlRuntimeContractBinding({
      policyValueHits: [
        {
          ref_key: "tenant_feature_flags",
          status: "resolved",
          source: "harness_tenant_feature_flags",
          value_source: "oem_cloud_bootstrap_features",
          value: {
            tenant_id: "tenant-1",
            flags: {
              gatewayEnabled: true,
            },
          },
        },
        {
          ref_key: "gateway_policy",
          status: "resolved",
          source: "harness_oem_routing",
          value_source: "request_oem_routing",
          value: {
            tenant_id: "tenant-1",
            can_invoke: false,
          },
        },
      ],
    });

    expect(contract.limecorePolicySnapshot).toMatchObject({
      status: "policy_inputs_evaluated",
      decision: "deny",
      decision_source: "policy_input_evaluator",
      decision_reason: "resolved_policy_inputs_contain_deny_signal",
      policy_evaluation: expect.objectContaining({
        status: "evaluated",
        decision: "deny",
        blocking_refs: ["gateway_policy"],
      }),
    });
  });

  it("browser_control contract 应提供 Browser Assist 底层运行字段", () => {
    const contract = resolveBrowserControlRuntimeContractBinding();

    expect(contract).toMatchObject({
      contractKey: "browser_control",
      modality: "browser",
      routingSlot: "browser_reasoning_model",
      runtimeContract: expect.objectContaining({
        contract_key: "browser_control",
        routing_slot: "browser_reasoning_model",
        executor_binding: expect.objectContaining({
          executor_kind: "browser",
          binding_key: "browser_assist",
        }),
      }),
    });
    expect(contract.requiredCapabilities).toEqual(
      expect.arrayContaining([
        "text_generation",
        "browser_reasoning",
        "browser_control_planning",
      ]),
    );
    expect(contract.boundEntrySources).toEqual(
      expect.arrayContaining([
        "at_browser_command",
        "at_browser_agent_command",
        "at_mini_tester_command",
      ]),
    );
  });

  it("browser_control entry source 应从 contract registry 派生", () => {
    expect(resolveBrowserControlEntrySource("@浏览器")).toBe(
      "at_browser_command",
    );
    expect(resolveBrowserControlEntrySource("@Browser Agent")).toBe(
      "at_browser_agent_command",
    );
    expect(resolveBrowserControlEntrySource("@Mini Tester")).toBe(
      "at_mini_tester_command",
    );
    expect(resolveBrowserControlEntrySource("@unknown")).toBe(
      "at_browser_command",
    );
  });

  it("pdf_extract contract 应提供 Skill(pdf_read) 底层运行字段与上层入口绑定", () => {
    const contract = resolvePdfExtractRuntimeContractBinding();

    expect(contract).toMatchObject({
      contractKey: "pdf_extract",
      modality: "document",
      routingSlot: "base_model",
      runtimeContract: expect.objectContaining({
        contract_key: "pdf_extract",
        routing_slot: "base_model",
        executor_binding: expect.objectContaining({
          executor_kind: "skill",
          binding_key: "pdf_read",
        }),
      }),
    });
    expect(contract.requiredCapabilities).toEqual(
      expect.arrayContaining([
        "text_generation",
        "local_file_read",
        "long_context",
      ]),
    );
    expect(contract.boundEntrySources).toEqual(
      expect.arrayContaining(["at_pdf_read_command"]),
    );
  });

  it("voice_generation contract 应提供 App Server audio worker 入口元数据", () => {
    const contract = resolveVoiceGenerationRuntimeContractBinding();

    expect(contract).toMatchObject({
      contractKey: "voice_generation",
      modality: "audio",
      routingSlot: "voice_generation_model",
      executionProfileKey: "voice_generation_profile",
      executorAdapterKey: "app_server:mediaTaskArtifact/audio/create",
      runtimeContract: expect.objectContaining({
        contract_key: "voice_generation",
        routing_slot: "voice_generation_model",
        route_execution_status: "executable",
        executor_binding: expect.objectContaining({
          executor_kind: "app_server",
          binding_key: "mediaTaskArtifact/audio/create",
        }),
        executor_adapter: expect.objectContaining({
          adapter_key: "app_server:mediaTaskArtifact/audio/create",
        }),
      }),
    });
    expect(contract.executionProfile?.lifecycle).toBe("current");
    expect(contract.executorAdapter).toMatchObject({
      adapter_key: "app_server:mediaTaskArtifact/audio/create",
    });
    expect(contract.requiredCapabilities).toEqual(
      expect.arrayContaining(["text_generation", "voice_generation"]),
    );
    expect(contract.boundEntrySources).toEqual(
      expect.arrayContaining(["at_voice_command"]),
    );
  });

  it("audio_transcription contract 应提供 Skill(transcription_generate) 底层运行字段与上层入口绑定", () => {
    const contract = resolveAudioTranscriptionRuntimeContractBinding();

    expect(contract).toMatchObject({
      contractKey: "audio_transcription",
      modality: "audio",
      routingSlot: "audio_transcription_model",
      runtimeContract: expect.objectContaining({
        contract_key: "audio_transcription",
        routing_slot: "audio_transcription_model",
        executor_binding: expect.objectContaining({
          executor_kind: "skill",
          binding_key: "transcription_generate",
        }),
      }),
    });
    expect(contract.requiredCapabilities).toEqual(
      expect.arrayContaining(["text_generation", "audio_transcription"]),
    );
    expect(contract.boundEntrySources).toEqual(
      expect.arrayContaining(["at_transcription_command"]),
    );
  });

  it("web_research contract 应提供 Skill(research) 底层运行字段与上层入口绑定", () => {
    const contract = resolveWebResearchRuntimeContractBinding();

    expect(contract).toMatchObject({
      contractKey: "web_research",
      modality: "mixed",
      routingSlot: "report_generation_model",
      runtimeContract: expect.objectContaining({
        contract_key: "web_research",
        routing_slot: "report_generation_model",
        executor_binding: expect.objectContaining({
          executor_kind: "skill",
          binding_key: "research",
        }),
      }),
    });
    expect(contract.requiredCapabilities).toEqual(
      expect.arrayContaining([
        "text_generation",
        "web_search",
        "structured_document_generation",
        "long_context",
      ]),
    );
    expect(contract.boundEntrySources).toEqual(
      expect.arrayContaining([
        "at_search_command",
        "at_deep_search_command",
        "at_site_search_command",
        "at_report_command",
        "at_competitor_command",
      ]),
    );
  });

  it("text_transform contract 应提供文本转换底层运行字段与上层入口绑定", () => {
    const contract = resolveTextTransformRuntimeContractBinding();

    expect(contract).toMatchObject({
      contractKey: "text_transform",
      modality: "document",
      routingSlot: "base_model",
      runtimeContract: expect.objectContaining({
        contract_key: "text_transform",
        routing_slot: "base_model",
        executor_binding: expect.objectContaining({
          executor_kind: "skill",
          binding_key: "text_transform",
        }),
      }),
    });
    expect(contract.requiredCapabilities).toEqual(
      expect.arrayContaining([
        "text_generation",
        "local_file_read",
        "long_context",
      ]),
    );
    expect(contract.boundEntrySources).toEqual(
      expect.arrayContaining([
        "at_file_read_command",
        "at_summary_command",
        "at_translation_command",
        "at_analysis_command",
        "at_publish_compliance_command",
      ]),
    );
  });
});
