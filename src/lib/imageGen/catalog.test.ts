import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";
import {
  findDefaultImageCapabilityProvider,
  isImageCapabilityProvider,
  resolveImageCapabilityModelIds,
  resolveImageCapabilityModels,
  resolveImageCapabilityProviderEntry,
  type ImageCapabilitySelectionCandidate,
} from "./catalog";

function readSource(relativePath: string): string {
  return readFileSync(resolve(cwd(), relativePath), "utf8");
}

function expectNoLegacyImageCapabilityHeuristic(source: string): void {
  expect(source).not.toContain("@/lib/imageGeneration");
  expect(source).not.toContain("IMAGE_GEN_MODELS");
  expect(source).not.toContain('from "@/lib/imageGen/models"');
  expect(source).not.toContain("isImageProvider(");
  expect(source).not.toContain("getImageModelIdsForProvider");
  expect(source).not.toContain("function isFalProviderLike");
  expect(source).not.toContain('includes("fal.run")');
  expect(source).not.toContain('selectedProvider.id === "new-api"');
  expect(source).not.toContain('selectedProvider.type === "new-api"');
  expect(source).not.toContain('selectedProvider.type === "NewApi"');
  expect(source).not.toContain('selectedProvider.id === "gemini"');
  expect(source).not.toContain('selectedProvider.id === "google"');
  expect(source).not.toContain('selectedProvider.id === "vertexai"');
}

function expectNoInlineFalProviderMatcher(source: string): void {
  expect(source).toContain("@/lib/imageGen/providerMatchers");
  expect(source).not.toContain('includes("fal.run")');
  expect(source).not.toContain('includes("queue.fal.run")');
  expect(source).not.toContain('providerType === "fal"');
  expect(source).not.toContain('providerId === "fal"');
  expect(source).not.toContain('providerId.startsWith("fal-")');
  expect(source).not.toContain('providerId.includes("fal.ai")');
}

const providers: ImageCapabilitySelectionCandidate[] = [
  { id: "new-api", type: "openai", models: ["gpt-images-2"] },
  {
    id: "gemini",
    type: "gemini",
    models: ["gemini-3.1-flash-image"],
    api_host: "https://generativelanguage.googleapis.com",
  },
  { id: "zhipuai", type: "zhipuai", models: ["glm-image"] },
  {
    id: "alibaba",
    type: "openai",
    models: ["qwen-image-plus"],
    api_host: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
  { id: "fal", type: "fal", models: ["fal-ai/nano-banana-pro"] },
];

describe("imageGen/catalog", () => {
  it("应识别图片能力 Provider", () => {
    expect(isImageCapabilityProvider(providers[0]!)).toBe(true);
    expect(isImageCapabilityProvider(providers[1]!)).toBe(true);
    expect(isImageCapabilityProvider({ id: "tts", type: "audio" })).toBe(false);
  });

  it("应解析 Provider 条目", () => {
    expect(
      resolveImageCapabilityProviderEntry(providers[0]!)?.providerKey,
    ).toBe("openai-compatible-responses");
    expect(resolveImageCapabilityProviderEntry(providers[0]!)?.transport).toBe(
      "openai_responses_image",
    );
    expect(
      resolveImageCapabilityProviderEntry(providers[1]!)?.providerKey,
    ).toBe("gemini");
    expect(
      resolveImageCapabilityProviderEntry(providers[2]!)?.providerKey,
    ).toBe("zhipu");
    expect(resolveImageCapabilityModelIds(providers[2]!)).toContain(
      "glm-image",
    );
    expect(
      resolveImageCapabilityProviderEntry(providers[3]!)?.providerKey,
    ).toBe("dashscope");
    expect(resolveImageCapabilityModelIds(providers[3]!)).toContain(
      "qwen-image-plus",
    );
    expect(
      resolveImageCapabilityProviderEntry({
        id: "fal",
        type: "openai",
        api_host: "https://fal.run/fal-ai",
      })?.providerKey,
    ).toBe("fal");
  });

  it("应优先选择 openai-like 默认 Provider", () => {
    expect(findDefaultImageCapabilityProvider(providers)?.id).toBe("new-api");
  });

  it("应解析 Provider 模型列表", () => {
    expect(resolveImageCapabilityModelIds(providers[0]!)).toContain(
      "gpt-images-2",
    );
    expect(resolveImageCapabilityModelIds(providers[1]!)).toContain(
      "gemini-3.1-flash-image",
    );
    expect(resolveImageCapabilityModels(providers[4]!)?.[0]?.id).toBe(
      "fal-ai/nano-banana-pro",
    );
  });

  it("用户声明的图片模型应优先于内置 fallback 模型", () => {
    expect(
      resolveImageCapabilityModels({
        id: "airgate-openai-images",
        type: "openai",
        api_host: "https://airgate.k8ray.com/v1",
        models: ["gpt-images-2"],
      })[0]?.id,
    ).toBe("gpt-images-2");
  });

  it("OpenAI 兼容中转拉取的新图片模型应进入图片默认模型候选", () => {
    expect(
      resolveImageCapabilityModelIds({
        id: "agnes",
        type: "openai",
        api_host: "https://agnes.example.test/v1",
        models: [
          "agnes-2.0-flash",
          "agnes-image-2.1-flash",
          "agnes-image-2.0-flash",
        ],
      }),
    ).toEqual(
      expect.arrayContaining([
        "agnes-image-2.1-flash",
        "agnes-image-2.0-flash",
      ]),
    );
  });

  it("Agnes 应作为一等图片 Provider 对齐官方图片接口", () => {
    const entry = resolveImageCapabilityProviderEntry({
      id: "agnes",
      type: "openai",
      api_host: "https://apihub.agnes-ai.com/v1",
    });

    expect(entry?.providerKey).toBe("agnes");
    expect(entry?.transport).toBe("openai_images");
    expect(entry?.endpointPath).toBe("/v1/images/generations");
    expect(
      resolveImageCapabilityModelIds({ id: "agnes", type: "openai" }),
    ).toContain("agnes-image-2.1-flash");
    expect(
      resolveImageCapabilityProviderEntry({
        id: "custom-openai",
        type: "openai",
        api_host: "https://apihub.agnes-ai.com/v1",
      })?.providerKey,
    ).toBe("agnes");
  });

  it("DashScope 拉取的新 Qwen-Image 模型应进入图片默认模型候选", () => {
    expect(
      resolveImageCapabilityModelIds({
        id: "custom-alibaba-provider",
        type: "openai",
        api_host: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        models: ["qwen3.5-72b", "qwen-image-2.0", "qwen-image-plus"],
      }),
    ).toEqual(expect.arrayContaining(["qwen-image-2.0", "qwen-image-plus"]));
  });

  it("settings 与 workspace 图片命令应共享图片能力目录入口", () => {
    const settingsSource = readSource(
      "src/components/settings-v2/agent/image-gen/index.tsx",
    );
    expect(settingsSource).toContain('from "@/lib/imageGen/catalog"');
    expect(settingsSource).toContain("isImageCapabilityProvider");
    expect(settingsSource).toContain("resolveImageCapabilityModelIds");
    expect(settingsSource).toContain("resolveImageCapabilityModels");
    expect(settingsSource).toContain("isImageCapabilityModelMetadata");
    expect(settingsSource).not.toContain(
      "buildProviderModelsFromBackendModelIds",
    );
    expectNoLegacyImageCapabilityHeuristic(settingsSource);

    const seededCommandSource = readSource(
      "src/lib/base-setup/seededCommandPackage.ts",
    );
    expect(seededCommandSource).toContain('from "@/lib/imageGen/catalog"');
    expect(seededCommandSource).toContain("requireSeededImageCommandModelId");
    expectNoLegacyImageCapabilityHeuristic(seededCommandSource);
    expect(seededCommandSource).not.toContain(
      'model: "fal-ai/nano-banana-pro"',
    );
    expect(seededCommandSource).not.toContain('model: "gpt-images-2"');

    const workspaceLaunchSource = readSource(
      "src/components/agent/chat/workspace/imageCommandIntent.ts",
    );
    expect(workspaceLaunchSource).toContain(
      'from "@/lib/governance/modalityRuntimeContracts"',
    );
    expect(workspaceLaunchSource).toContain(
      "resolveImageGenerationRuntimeContractBinding",
    );
    expectNoLegacyImageCapabilityHeuristic(workspaceLaunchSource);
  });

  it("本机图片服务端点应允许前端锁定图片 Provider", () => {
    const serverSource = readSource("lime-rs/crates/server/src/lib.rs");
    const imageHandlerSource = readSource(
      "lime-rs/crates/server/src/handlers/image_handler.rs",
    );

    expect(serverSource).toContain('HeaderName::from_static("x-provider-id")');
    expect(imageHandlerSource).toContain('get("x-provider-id")');
  });

  it("Provider 模型拉取不应内联 Fal provider keyword 判断", () => {
    const providerSettingSource = readSource(
      "src/components/api-key-provider/ProviderSetting.tsx",
    );
    expect(providerSettingSource).toContain("@/lib/imageGen/providerMatchers");
    expect(providerSettingSource).toContain(
      "@/lib/layered-design/imageModelCapabilities",
    );
    expect(providerSettingSource).not.toContain(
      "function isLikelyImageModelCommandModel(modelId: string): boolean {\n  const normalized",
    );
    expect(providerSettingSource).not.toContain(
      "nano-banana|banana|flux|seedream|kontext|recraft|ideogram|sdxl|stable-diffusion",
    );

    const modelAddPanelSource = readSource(
      "src/components/api-key-provider/ModelAddPanel.tsx",
    );
    expect(modelAddPanelSource).toContain("./providerModelFetchHelpers");
    expect(modelAddPanelSource).not.toContain('includes("fal.run")');
    expect(modelAddPanelSource).not.toContain('includes("queue.fal.run")');
    expect(modelAddPanelSource).not.toContain('providerType === "fal"');
    expect(modelAddPanelSource).not.toContain('providerId === "fal"');
    expect(modelAddPanelSource).not.toContain('providerId.startsWith("fal-")');
    expect(modelAddPanelSource).not.toContain('providerId.includes("fal.ai")');

    const fetchHelpersSource = readSource(
      "src/components/api-key-provider/providerModelFetchHelpers.ts",
    );
    expectNoInlineFalProviderMatcher(fetchHelpersSource);

    const fetchSupportSource = readSource(
      "src/lib/model/providerModelFetchSupport.ts",
    );
    expectNoInlineFalProviderMatcher(fetchSupportSource);
  });

  it("设计画布图片任务入口不应重新维护图片 Provider / 模型 heuristics", () => {
    const canvasRuntimeSource = readSource(
      "src/components/agent/chat/workspace/useWorkspaceCanvasSceneRuntime.tsx",
    );
    expect(canvasRuntimeSource).toContain(
      "imageWorkbenchGenerationRuntime.selectedProviderId",
    );
    expect(canvasRuntimeSource).toContain(
      "imageWorkbenchGenerationRuntime.selectedModelId",
    );
    expectNoLegacyImageCapabilityHeuristic(canvasRuntimeSource);

    const canvasFactorySource = readSource(
      "src/components/workspace/canvas/CanvasFactory.tsx",
    );
    expect(canvasFactorySource).toContain("imageGenerationProviderId");
    expect(canvasFactorySource).toContain("imageGenerationModelId");
    expectNoLegacyImageCapabilityHeuristic(canvasFactorySource);

    const designCanvasSource = readSource(
      "src/components/workspace/design/DesignCanvas.tsx",
    );
    expect(designCanvasSource).toContain(
      "createLayeredDesignImageTaskArtifacts",
    );
    expect(designCanvasSource).toContain(
      "providerId: imageGenerationProviderId",
    );
    expect(designCanvasSource).toContain("model: imageGenerationModelId");
    expectNoLegacyImageCapabilityHeuristic(designCanvasSource);

    const imageTaskSource = readSource("src/lib/layered-design/imageTasks.ts");
    expect(imageTaskSource).toContain(
      'modalityContractKey: "image_generation"',
    );
    expect(imageTaskSource).toContain('routingSlot: "image_generation_model"');
    expectNoLegacyImageCapabilityHeuristic(imageTaskSource);
  });

  it("useImageGen 生产消费者只应复用图片 Provider 与模型选择", () => {
    const hookSource = readSource("src/components/image-gen/useImageGen.ts");
    const providerRuntimeSource = readSource(
      "src/components/agent/chat/workspace/useWorkspaceImageWorkbenchProviderRuntime.ts",
    );

    expect(hookSource).toContain("resolveImageCapabilityModels");
    expect(providerRuntimeSource).toContain("useImageGen({");
    expect(providerRuntimeSource).toContain(
      "imageWorkbenchGenerationRuntime.selectedProvider",
    );
    expect(providerRuntimeSource).toContain(
      "imageWorkbenchGenerationRuntime.selectedModel",
    );
    expect(providerRuntimeSource).not.toContain("generateImage");
    expect(providerRuntimeSource).not.toContain("cancelGeneration");
    expect(providerRuntimeSource).not.toContain("saveImagesToResource");
    expect(providerRuntimeSource).not.toContain(
      "requestImagesFromLocalImageServer",
    );
    expectNoLegacyImageCapabilityHeuristic(hookSource);
    expectNoLegacyImageCapabilityHeuristic(providerRuntimeSource);

    const layeredImageModelCapabilitySource = readSource(
      "src/lib/layered-design/imageModelCapabilities.ts",
    );
    expect(layeredImageModelCapabilitySource).toContain(
      "@/lib/imageGen/providerMatchers",
    );
    expect(layeredImageModelCapabilitySource).not.toContain(
      'normalized === "gpt-image-2"',
    );
    expect(layeredImageModelCapabilitySource).not.toContain(
      'normalized === "gpt-images-2"',
    );
    expect(layeredImageModelCapabilitySource).not.toContain(
      'normalized.endsWith("/gpt-image-2")',
    );
    expect(layeredImageModelCapabilitySource).not.toContain(
      'normalized.endsWith("/gpt-images-2")',
    );

    const modelThemePolicySource = readSource(
      "src/components/agent/chat/utils/modelThemePolicy.ts",
    );
    expect(modelThemePolicySource).toContain("@/lib/imageGen/providerMatchers");
    expect(modelThemePolicySource).not.toContain("IMAGE_INCLUDE_KEYWORDS");
    expect(modelThemePolicySource).not.toContain("midjourney");
    expect(modelThemePolicySource).not.toContain("生图");

    const visionModelResolverSource = readSource(
      "src/lib/model/visionModelResolver.ts",
    );
    expect(visionModelResolverSource).toContain(
      "@/lib/imageGen/providerMatchers",
    );
    expect(visionModelResolverSource).not.toContain(
      "IMAGE_GENERATION_KEYWORDS",
    );
    expect(visionModelResolverSource).not.toContain("midjourney");
    expect(visionModelResolverSource).not.toContain("stable diffusion");

    const inferModelCapabilitiesSource = readSource(
      "src/lib/model/inferModelCapabilities.ts",
    );
    expect(inferModelCapabilitiesSource).toContain(
      "@/lib/imageGen/providerMatchers",
    );
    expect(inferModelCapabilitiesSource).not.toContain(
      "IMAGE_GENERATION_PATTERN",
    );
    expect(inferModelCapabilitiesSource).not.toContain("image-preview");
    expect(inferModelCapabilitiesSource).not.toContain("nano-banana");
  });
});
