import React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { changeLimeLocale } from "@/i18n/createI18n";
import { ModelCapabilityBadges } from "./ModelCapabilityBadges";
import type { EnhancedModelMetadata } from "@/lib/types/modelRegistry";

const mountedRoots: Array<{ root: Root; container: HTMLDivElement }> = [];

beforeEach(async () => {
  (
    globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT?: boolean;
    }
  ).IS_REACT_ACT_ENVIRONMENT = true;
  await changeLimeLocale("zh-CN");
});

afterEach(() => {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
});

function renderBadges(
  props?: Partial<React.ComponentProps<typeof ModelCapabilityBadges>>,
) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  act(() => {
    root.render(
      <ModelCapabilityBadges
        capabilities={{
          vision: false,
          tools: true,
          streaming: true,
          json_mode: true,
          function_calling: true,
          reasoning: false,
        }}
        {...props}
      />,
    );
  });

  mountedRoots.push({ root, container });
  return container;
}

function createModel(
  id: string,
  overrides: Partial<EnhancedModelMetadata> = {},
): EnhancedModelMetadata {
  return {
    id,
    display_name: id,
    provider_id: "openai",
    provider_name: "OpenAI",
    family: null,
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
    release_date: null,
    is_latest: false,
    description: null,
    source: "local",
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

describe("ModelCapabilityBadges", () => {
  it("显式 image 输入模态应展示为支持多模态", () => {
    const container = renderBadges({
      model: createModel("provider-vlm-chat", {
        task_families: ["chat"],
        input_modalities: ["text", "image"],
      }),
    });

    expect(container.textContent).toContain("支持多模态");
    expect(container.textContent).not.toContain("无多模态");
  });

  it("已知视觉模型即使旧缓存未写入 vision 标记也应展示为支持多模态", () => {
    const container = renderBadges({
      model: createModel("o3", {
        task_families: ["chat"],
        input_modalities: ["text"],
      }),
    });

    expect(container.textContent).toContain("支持多模态");
    expect(container.textContent).not.toContain("无多模态");
  });

  it("支持 reasoning summary 但不可调 effort 时不应展示为无思考", () => {
    const container = renderBadges({
      model: createModel("provider-reasoning-summary", {
        reasoning_policy: {
          supports_reasoning_summaries: true,
          default_reasoning_level: null,
          supported_reasoning_levels: [],
          supported_reasoning_efforts: [],
          can_set_reasoning_effort: false,
        },
      }),
    });

    expect(container.textContent).toContain("支持思考");
    expect(container.textContent).not.toContain("无思考");
  });

  it("未显式声明 reasoning 时不应根据模型名推断思考能力", () => {
    const container = renderBadges({
      model: createModel("gpt-5.4-mini", {
        provider_id: "lime",
        provider_name: "lime",
      }),
    });

    expect(container.textContent).toContain("无思考");
    expect(container.textContent).not.toContain("支持思考");
  });
});
