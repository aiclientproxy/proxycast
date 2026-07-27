import { describe, expect, it } from "vitest";
import {
  buildPersistedMediaGenerationPreference,
  findMediaProviderById,
  findTtsProviderForSelection,
  findVideoProviderForSelection,
  getTtsModelsForProvider,
  getVideoModelsForProvider,
  hasMediaGenerationPreferenceOverride,
  isTtsProvider,
  isVideoProvider,
  pickTtsModel,
  pickVideoModelByVersion,
  resolveMediaGenerationPreference,
} from "./mediaGeneration";

const providers = [
  { id: "doubao-video", type: "openai", models: [] },
  { id: "openai-tts", type: "openai", models: ["gpt-4o-mini-tts"] },
  { id: "qwen-video", type: "openai", models: [] },
];

describe("mediaGeneration", () => {
  it("应按 ID 查找媒体 Provider", () => {
    expect(findMediaProviderById(providers, "doubao-video")?.id).toBe(
      "doubao-video",
    );
    expect(findMediaProviderById(providers, "missing")).toBeNull();
    expect(findMediaProviderById(undefined, "doubao-video")).toBeNull();
  });

  it("应识别视频和语音 Provider", () => {
    expect(isVideoProvider("doubao-video")).toBe(true);
    expect(isVideoProvider("openai-tts")).toBe(false);
    expect(isTtsProvider("openai-tts", "openai")).toBe(true);
    expect(isTtsProvider("runway-video", "video")).toBe(false);
  });

  it("应解析视频 Provider 模型并按版本挑选", () => {
    expect(getVideoModelsForProvider("doubao-video")[0]).toBe(
      "seedance-1-5-pro-251215",
    );
    expect(
      pickVideoModelByVersion(
        ["seedance-1-5-pro-251215", "seedance-1-5-lite-250428"],
        "v2-1-master",
      ),
    ).toBe("seedance-1-5-pro-251215");
  });

  it("应解析语音 Provider 模型并挑选 TTS 模型", () => {
    expect(getTtsModelsForProvider()[0]).toBe("gpt-4o-mini-tts");
    expect(pickTtsModel(["gpt-4o-mini-tts", "gpt-image-1"])).toBe(
      "gpt-4o-mini-tts",
    );
  });

  it("应按默认策略选择视频与语音 Provider", () => {
    expect(findVideoProviderForSelection(providers, "jimeng")?.id).toBe(
      "doubao-video",
    );
    expect(findTtsProviderForSelection([providers[1]])?.id).toBe("openai-tts");
    expect(findVideoProviderForSelection(undefined, "jimeng")).toBeNull();
    expect(findTtsProviderForSelection(undefined)).toBeNull();
  });

  it("应优先使用项目覆盖，否则回退到全局默认", () => {
    expect(
      resolveMediaGenerationPreference(
        {
          preferredProviderId: "project-provider",
          preferredModelId: "project-model",
          allowFallback: false,
        },
        {
          preferredProviderId: "global-provider",
          preferredModelId: "global-model",
          allowFallback: true,
        },
      ),
    ).toEqual({
      preferredProviderId: "project-provider",
      preferredModelId: "project-model",
      allowFallback: false,
      source: "project",
    });

    expect(
      resolveMediaGenerationPreference(undefined, {
        preferredProviderId: "global-provider",
        preferredModelId: "global-model",
        allowFallback: false,
      }),
    ).toEqual({
      preferredProviderId: "global-provider",
      preferredModelId: "global-model",
      allowFallback: false,
      source: "global",
    });
  });

  it("应在无有效覆盖时不持久化默认值", () => {
    expect(hasMediaGenerationPreferenceOverride(undefined)).toBe(false);
    expect(
      buildPersistedMediaGenerationPreference({
        preferredProviderId: "",
        preferredModelId: "model-only",
        allowFallback: true,
      }),
    ).toBeUndefined();
    expect(
      buildPersistedMediaGenerationPreference({
        allowFallback: false,
      }),
    ).toEqual({
      preferredProviderId: undefined,
      preferredModelId: undefined,
      allowFallback: false,
    });
  });
});
