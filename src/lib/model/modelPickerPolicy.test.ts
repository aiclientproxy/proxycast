import { describe, expect, it } from "vitest";
import {
  buildCatalogModelPickerPolicy,
  buildModelPickerPolicy,
  resolveModelServiceTierForRequest,
} from "./modelPickerPolicy";

describe("modelPickerPolicy", () => {
  it("默认 fail-closed，不展示到 picker，也不生成服务 tier", () => {
    expect(buildModelPickerPolicy(null)).toEqual({
      visibility: "none",
      show_in_picker: false,
      service_tiers: [],
      supported_service_tier_ids: [],
      default_service_tier: null,
    });
  });

  it("只在 visibility=list 时展示到 picker", () => {
    expect(
      buildModelPickerPolicy({
        visibility: "list",
      }),
    ).toMatchObject({
      visibility: "list",
      show_in_picker: true,
    });

    expect(buildModelPickerPolicy({ visibility: "hide" })).toMatchObject({
      visibility: "hide",
      show_in_picker: false,
    });

    expect(buildModelPickerPolicy({ visibility: "unknown" })).toMatchObject({
      visibility: "none",
      show_in_picker: false,
    });
  });

  it("在 policy owner 内把 catalog hidden 字段映射为 visibility", () => {
    expect(buildCatalogModelPickerPolicy(false)).toMatchObject({
      visibility: "list",
      show_in_picker: true,
    });
    expect(buildCatalogModelPickerPolicy(true)).toMatchObject({
      visibility: "hide",
      show_in_picker: false,
    });
  });

  it("归一 service_tiers，过滤无 id 项和重复 id", () => {
    expect(
      buildModelPickerPolicy({
        service_tiers: [
          { id: "priority", name: "Fast", description: "Priority lane" },
          { id: "priority", name: "Duplicate", description: "ignored" },
          { id: "", name: "Invalid" },
          { id: "flex" },
        ],
        default_service_tier: "priority",
      }),
    ).toMatchObject({
      service_tiers: [
        { id: "priority", name: "Fast", description: "Priority lane" },
        { id: "flex", name: "flex", description: "" },
      ],
      supported_service_tier_ids: ["priority", "flex"],
      default_service_tier: "priority",
    });
  });

  it("request service tier 只透传显式且受支持的 tier，不自动应用 catalog default", () => {
    const policy = buildModelPickerPolicy({
      serviceTiers: [{ id: "priority", name: "Fast", description: "" }],
      defaultServiceTier: "priority",
    });

    expect(resolveModelServiceTierForRequest(policy, undefined)).toBeNull();
    expect(resolveModelServiceTierForRequest(policy, "default")).toBeNull();
    expect(resolveModelServiceTierForRequest(policy, "unsupported")).toBeNull();
    expect(resolveModelServiceTierForRequest(policy, "priority")).toBe(
      "priority",
    );
  });

  it("不会从 capability、pricing 或 provider catalog 字段推断 picker policy", () => {
    expect(
      buildModelPickerPolicy({
        tier: "max",
        status: "active",
        pricing: { input_per_million: 1 },
        provider_name: "OpenAI",
        input_modalities: ["text", "image"],
      } as Record<string, unknown>),
    ).toEqual({
      visibility: "none",
      show_in_picker: false,
      service_tiers: [],
      supported_service_tier_ids: [],
      default_service_tier: null,
    });
  });
});
