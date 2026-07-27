import { describe, expect, it } from "vitest";
import type { EnhancedModelMetadata } from "@/lib/types/modelRegistry";
import { resolveVisionModel } from "./visionModelResolver";

function createModel(
  id: string,
  overrides: Partial<EnhancedModelMetadata> = {},
): EnhancedModelMetadata {
  return {
    id,
    display_name: id,
    provider_id: "zhipuai",
    provider_name: "Zhipu AI",
    family: id,
    tier: "pro",
    capabilities: {
      vision: false,
      tools: true,
      streaming: true,
      json_mode: true,
      function_calling: true,
      reasoning: false,
    },
    pricing: null,
    limits: {
      context_length: null,
      max_output_tokens: null,
      requests_per_minute: null,
      tokens_per_minute: null,
    },
    status: "active",
    release_date: "2026-01-01",
    is_latest: false,
    description: null,
    source: "local",
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

describe("resolveVisionModel", () => {
  it("当前模型已支持视觉时应保持不变", () => {
    const models = [
      createModel("glm-4.6v-flash", {
        capabilities: {
          vision: true,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: false,
        },
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "glm-4.6v-flash",
      models,
    });

    expect(result).toEqual({
      targetModelId: "glm-4.6v-flash",
      switched: false,
      reason: "already_vision",
    });
  });

  it("当前模型未收录但模型名可推断支持视觉时应保持不变", () => {
    const models = [
      createModel("gpt-5.3-codex", {
        provider_id: "codex",
        provider_name: "Local CLI",
        capabilities: {
          vision: true,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: true,
        },
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "gpt-5.4",
      models,
    });

    expect(result).toEqual({
      targetModelId: "gpt-5.4",
      switched: false,
      reason: "already_vision",
    });
  });

  it("当前模型未收录但属于现代图片输入模型时应保持不变", () => {
    const models = [
      createModel("gpt-5.4", {
        capabilities: {
          vision: true,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: true,
        },
      }),
    ];

    for (const currentModelId of ["o3", "o4-mini", "grok-4.3", "qwen3.5-27b"]) {
      expect(
        resolveVisionModel({
          currentModelId,
          models,
        }),
        currentModelId,
      ).toEqual({
        targetModelId: currentModelId,
        switched: false,
        reason: "already_vision",
      });
    }
  });

  it("当前模型来自旧缓存且缺少 vision 标记时应按已知视觉模型名保持不变", () => {
    const models = [
      createModel("o3", {
        provider_id: "openai",
        provider_name: "OpenAI",
        task_families: ["chat"],
        input_modalities: ["text"],
        capabilities: {
          vision: false,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: false,
        },
      }),
      createModel("gpt-5.4", {
        provider_id: "openai",
        provider_name: "OpenAI",
        capabilities: {
          vision: true,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: true,
        },
      }),
    ];

    expect(
      resolveVisionModel({
        currentModelId: "o3",
        models,
      }),
    ).toEqual({
      targetModelId: "o3",
      switched: false,
      reason: "already_vision",
    });
  });

  it("当前模型未收录且属于无图片输入同系列模型时仍应切换", () => {
    const models = [
      createModel("gpt-5.4", {
        capabilities: {
          vision: true,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: true,
        },
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "o3-mini",
      models,
    });

    expect(result.targetModelId).toBe("gpt-5.4");
    expect(result.switched).toBe(true);
  });

  it("显式输入模态包含 image 时应视为支持图片输入", () => {
    const models = [
      createModel("provider-vlm-chat", {
        capabilities: {
          vision: false,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: false,
        },
        task_families: ["chat"],
        input_modalities: ["text", "image"],
        output_modalities: ["text"],
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "provider-vlm-chat",
      models,
    });

    expect(result).toEqual({
      targetModelId: "provider-vlm-chat",
      switched: false,
      reason: "already_vision",
    });
  });

  it("未声明工具能力时不应把未知误当成显式 false", () => {
    const models = [
      createModel("provider-image-preview", {
        capability_provenance: "inferred_hint",
        capabilities: {
          vision: true,
          reasoning: false,
        },
        input_modalities: ["text", "image"],
        output_modalities: ["text"],
      }),
    ];

    expect(
      resolveVisionModel({
        currentModelId: "provider-image-preview",
        models,
      }),
    ).toEqual({
      targetModelId: "provider-image-preview",
      switched: false,
      reason: "already_vision",
    });
  });

  it("显式 vision_understanding 任务族应优先于 capabilities.vision 的缺省值", () => {
    const models = [
      createModel("hub-vlm", {
        capabilities: {
          vision: false,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: false,
        },
        task_families: ["chat", "vision_understanding"],
        input_modalities: ["text"],
        output_modalities: ["text"],
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "hub-vlm",
      models,
    });

    expect(result.reason).toBe("already_vision");
  });

  it("应优先选择支持视觉的聊天模型，而不是纯生图模型", () => {
    const models = [
      createModel("gemini-3-pro-image-preview", {
        family: "gemini-3-pro-image",
        capabilities: {
          vision: true,
          tools: false,
          streaming: true,
          json_mode: false,
          function_calling: false,
          reasoning: false,
        },
        description: "image generation model",
        is_latest: true,
      }),
      createModel("glm-4.6v-flash", {
        family: "glm-4.6v",
        tier: "mini",
        capabilities: {
          vision: true,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: false,
        },
        release_date: "2026-02-01",
        is_latest: true,
      }),
      createModel("glm-4.7", {
        family: "glm-4.7",
        capabilities: {
          vision: false,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: true,
        },
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "glm-4.7",
      models,
    });

    expect(result.targetModelId).toBe("glm-4.6v-flash");
    expect(result.switched).toBe(true);
    expect(result.reason).toBe("fallback_latest");
  });

  it("应复用图片 matcher，避免 OpenAI 兼容中转图片模型被当作视觉理解候选", () => {
    const models = [
      createModel("agnes-image-2.1-flash", {
        family: "agnes-image",
        capabilities: {
          vision: true,
          tools: false,
          streaming: true,
          json_mode: false,
          function_calling: false,
          reasoning: false,
        },
        description: "OpenAI-compatible image generation model",
        is_latest: true,
      }),
      createModel("glm-4.6v-flash", {
        family: "glm-4.6v",
        tier: "mini",
        capabilities: {
          vision: true,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: false,
        },
        release_date: "2026-02-01",
        is_latest: true,
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "deepseek-chat",
      models,
    });

    expect(result.targetModelId).toBe("glm-4.6v-flash");
    expect(result.reason).toBe("fallback_latest");
  });

  it("应允许显式同时支持视觉理解和生图的文本模型作为图片理解候选", () => {
    const models = [
      createModel("relay-vision-image-pro", {
        family: "relay-vision-image",
        capabilities: {
          vision: false,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: false,
        },
        task_families: ["chat", "vision_understanding", "image_generation"],
        input_modalities: ["text", "image"],
        output_modalities: ["text", "image"],
        is_latest: true,
      }),
      createModel("glm-4.7", {
        family: "glm-4.7",
        capabilities: {
          vision: false,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: true,
        },
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "glm-4.7",
      models,
    });

    expect(result.targetModelId).toBe("relay-vision-image-pro");
    expect(result.reason).toBe("fallback_latest");
  });

  it("没有可用视觉聊天模型时应返回 no_vision_model", () => {
    const models = [
      createModel("glm-4.7"),
      createModel("gemini-3-pro-image-preview", {
        capabilities: {
          vision: true,
          tools: false,
          streaming: true,
          json_mode: false,
          function_calling: false,
          reasoning: false,
        },
        description: "image generation model",
      }),
    ];

    const result = resolveVisionModel({
      currentModelId: "glm-4.7",
      models,
    });

    expect(result).toEqual({
      targetModelId: "glm-4.7",
      switched: false,
      reason: "no_vision_model",
    });
  });
});
