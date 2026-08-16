import { describe, expect, it } from "vitest";
import {
  resolveBrowserControlRuntimeContractBinding,
  resolveImageGenerationRuntimeContractBinding,
  resolveTextTransformRuntimeContractBinding,
  resolveVoiceGenerationRuntimeContractBinding,
} from "./modalityRuntimeContracts";
import {
  resolveExecutorAdapterKey,
  resolveModalityExecutionProfileBinding,
} from "./modalityExecutionProfiles";

describe("modalityExecutionProfiles", () => {
  it("应从 executor binding 解析标准 adapter key", () => {
    expect(
      resolveExecutorAdapterKey({
        executor_kind: "workflow",
        binding_key: "image_command",
      }),
    ).toBe("workflow:image_command");
    expect(resolveExecutorAdapterKey({ executor_kind: "skill" })).toBeNull();
  });

  it("image_generation 应解析到 profile 与 executor adapter", () => {
    const binding = resolveImageGenerationRuntimeContractBinding();

    expect(binding).toMatchObject({
      executionProfileKey: "image_generation_profile",
      executorAdapterKey: "workflow:image_command",
      runtimeContract: {
        execution_profile: expect.objectContaining({
          profile_key: "image_generation_profile",
          model_role_slots: ["image_generation_model"],
          artifact_policy: expect.objectContaining({
            artifact_kinds: ["image_task", "image_output"],
            viewer_surfaces: ["image_workbench"],
          }),
        }),
        executor_adapter: expect.objectContaining({
          adapter_key: "workflow:image_command",
          supports_progress: true,
          supports_cancel: true,
          supports_resume: false,
          artifact_output_kinds: ["image_task", "image_output"],
        }),
      },
    });
  });

  it("browser_control profile 应保留浏览器权限与 resume 能力", () => {
    const binding = resolveBrowserControlRuntimeContractBinding();

    expect(binding.executionProfile).toEqual(
      expect.objectContaining({
        profile_key: "browser_control_profile",
        permission_profile_keys: expect.arrayContaining([
          "browser_control",
          "web_search",
        ]),
        fallback_behavior: expect.arrayContaining([
          "do_not_fallback_to_web_search",
        ]),
      }),
    );
    expect(binding.executorAdapter).toEqual(
      expect.objectContaining({
        adapter_key: "browser:browser_assist",
        supports_resume: true,
        artifact_output_kinds: expect.arrayContaining([
          "browser_session",
          "browser_snapshot",
        ]),
      }),
    );
  });

  it("text_transform 只能把 generic_file 作为 compat fallback", () => {
    const binding = resolveTextTransformRuntimeContractBinding();

    expect(binding.executionProfile).toEqual(
      expect.objectContaining({
        profile_key: "text_transform_profile",
        artifact_policy: expect.objectContaining({
          artifact_kinds: ["report_document", "generic_file"],
        }),
        fallback_behavior: expect.arrayContaining([
          "keep_generic_file_as_compat_only",
        ]),
      }),
    );
  });

  it("voice_generation profile 应指向 current App Server audio worker", () => {
    const binding = resolveVoiceGenerationRuntimeContractBinding();

    expect(binding.runtimeContract).toEqual(
      expect.objectContaining({
        route_execution_status: "executable",
      }),
    );
    expect(binding.executionProfile).toEqual(
      expect.objectContaining({
        lifecycle: "current",
        profile_key: "voice_generation_profile",
        fallback_behavior: expect.arrayContaining([
          "do_not_fallback_to_legacy_tts_test_command",
        ]),
      }),
    );
    expect(binding.executionProfile?.fallback_behavior).toEqual(
      expect.arrayContaining(["do_not_fallback_to_legacy_tts_test_command"]),
    );
    expect(binding.executorAdapterKey).toBe(
      "app_server:mediaTaskArtifact/audio/create",
    );
    expect(binding.executorAdapter).toMatchObject({
      executor_kind: "app_server",
      binding_key: "mediaTaskArtifact/audio/create",
      lifecycle: "current",
    });
  });

  it("直接 resolver 应按 contract key 返回同一份 profile 绑定", () => {
    const binding = resolveModalityExecutionProfileBinding({
      contractKey: "audio_transcription",
      executorBinding: {
        executor_kind: "app_server",
        binding_key: "mediaTaskArtifact/transcription/create",
      },
    });

    expect(binding).toEqual(
      expect.objectContaining({
        profileKey: "audio_transcription_profile",
        executorAdapterKey: "app_server:mediaTaskArtifact/transcription/create",
        executionProfile: expect.objectContaining({
          model_role_slots: ["audio_transcription_model"],
          permission_profile_keys: expect.arrayContaining([
            "read_files",
            "write_artifacts",
            "service_api_call",
          ]),
        }),
        executorAdapter: expect.objectContaining({
          artifact_output_kinds: ["transcript"],
        }),
      }),
    );
  });
});
