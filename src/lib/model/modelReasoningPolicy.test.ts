import { describe, expect, it } from "vitest";
import {
  buildModelReasoningPolicy,
  normalizeModelReasoningEffort,
  resolveModelReasoningEffortForModelSwitch,
  resolveModelReasoningEffortForRequest,
} from "./modelReasoningPolicy";

describe("modelReasoningPolicy", () => {
  it("默认 fail-closed，不发送 reasoning effort", () => {
    expect(buildModelReasoningPolicy(null)).toEqual({
      supports_reasoning_summaries: false,
      default_reasoning_level: null,
      supported_reasoning_levels: [],
      supported_reasoning_efforts: [],
      can_set_reasoning_effort: false,
    });
  });

  it("归一 Codex reasoning levels，并保留未来自定义非空 effort", () => {
    expect(normalizeModelReasoningEffort("ultra")).toBe("ultra");
    expect(normalizeModelReasoningEffort("future-effort")).toBe(
      "future-effort",
    );
    expect(normalizeModelReasoningEffort("")).toBeNull();

    expect(
      buildModelReasoningPolicy({
        supports_reasoning_summary_parameter: true,
        default_reasoning_level: "medium",
        supported_reasoning_levels: [
          {
            id: "low",
            value: "low",
            label: "",
            description: "Small budget",
          },
          {
            id: "low",
            value: "low",
            label: "",
            description: "duplicate",
          },
          {
            id: "max",
            value: "max",
            label: "",
            description: "Max budget",
          },
          "future-effort",
          { id: "invalid", value: "", label: "", description: "invalid" },
        ],
      }),
    ).toEqual({
      supports_reasoning_summaries: true,
      default_reasoning_level: "medium",
      supported_reasoning_levels: [
        {
          id: "low",
          label: "",
          value: "low",
          description: "Small budget",
        },
        {
          id: "max",
          label: "",
          value: "max",
          description: "Max budget",
        },
        {
          id: "future-effort",
          label: "",
          value: "future-effort",
          description: "",
        },
      ],
      supported_reasoning_efforts: ["low", "max", "future-effort"],
      can_set_reasoning_effort: true,
    });
  });

  it("保留 grok-build 菜单 id/label，并把 default 与可提交值归一到 canonical wire value", () => {
    const policy = buildModelReasoningPolicy({
      defaultReasoningLevel: "deep",
      supportedReasoningLevels: [
        {
          id: "deep",
          label: "Deep",
          value: "xhigh",
          description: "Maximum reasoning",
        },
        {
          id: "balanced",
          label: "Balanced",
          value: "medium",
        },
      ],
    });

    expect(policy).toEqual({
      supports_reasoning_summaries: false,
      default_reasoning_level: "xhigh",
      supported_reasoning_levels: [
        {
          id: "deep",
          label: "Deep",
          value: "xhigh",
          description: "Maximum reasoning",
        },
        {
          id: "balanced",
          label: "Balanced",
          value: "medium",
          description: "",
        },
      ],
      supported_reasoning_efforts: ["xhigh", "medium"],
      can_set_reasoning_effort: true,
    });
    expect(resolveModelReasoningEffortForRequest(policy, "xhigh")).toBe(
      "xhigh",
    );
    expect(resolveModelReasoningEffortForRequest(policy, "deep")).toBe(
      "xhigh",
    );
  });

  it("request effort 只有请求档位受支持时才透传，缺省走 model default", () => {
    const policy = buildModelReasoningPolicy({
      supportsReasoningSummaryParameter: false,
      defaultReasoningLevel: "medium",
      supportedReasoningLevels: [
        { id: "low", value: "low", label: "", description: "" },
        { id: "medium", value: "medium", label: "", description: "" },
        { id: "high", value: "high", label: "", description: "" },
      ],
    });

    expect(resolveModelReasoningEffortForRequest(policy, "high")).toBe("high");
    expect(resolveModelReasoningEffortForRequest(policy, "ultra")).toBe(
      "medium",
    );
    expect(resolveModelReasoningEffortForRequest(policy, undefined)).toBe(
      "medium",
    );
    expect(
      resolveModelReasoningEffortForRequest(
        { ...policy, supports_reasoning_summaries: false },
        "high",
      ),
    ).toBe("high");
  });

  it("切模型时沿用 Codex 语义：保留受支持 current，否则取 supported 中位数，再 fallback default", () => {
    const policy = buildModelReasoningPolicy({
      supports_reasoning_summary_parameter: false,
      default_reasoning_level: "medium",
      supported_reasoning_levels: [
        { id: "minimal", value: "minimal", label: "", description: "" },
        { id: "low", value: "low", label: "", description: "" },
        { id: "medium", value: "medium", label: "", description: "" },
        { id: "high", value: "high", label: "", description: "" },
      ],
    });

    expect(resolveModelReasoningEffortForModelSwitch(policy, "high")).toBe(
      "high",
    );
    expect(resolveModelReasoningEffortForModelSwitch(policy, "ultra")).toBe(
      "low",
    );
    expect(
      resolveModelReasoningEffortForModelSwitch(
        {
          supports_reasoning_summaries: true,
          default_reasoning_level: "medium",
          supported_reasoning_efforts: [],
        },
        "high",
      ),
    ).toBe("medium");
  });

  it("不会从 capabilities、runtime features 或 picker/catalog 字段推断 reasoning policy", () => {
    expect(
      buildModelReasoningPolicy({
        capabilities: { reasoning: true },
        runtime_features: ["reasoning"],
        input_modalities: ["text", "image"],
        tier: "max",
        status: "active",
        pricing: { input_per_million: 1 },
        provider_name: "OpenAI",
      } as Record<string, unknown>),
    ).toEqual({
      supports_reasoning_summaries: false,
      default_reasoning_level: null,
      supported_reasoning_levels: [],
      supported_reasoning_efforts: [],
      can_set_reasoning_effort: false,
    });
  });
});
