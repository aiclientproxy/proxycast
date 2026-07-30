import React, { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { vi } from "vitest";
import type {
  EnhancedModelMetadata,
  ModelCapabilities,
} from "@/lib/types/modelRegistry";
import { ModelSelector, type ModelSelectorProps } from "./ModelSelector";

interface MountedRoot {
  root: Root;
  container: HTMLDivElement;
}

const mountedRoots: MountedRoot[] = [];

const defaultCapabilities: ModelCapabilities = {
  vision: false,
  tools: false,
  streaming: false,
  json_mode: false,
  function_calling: false,
  reasoning: false,
};

const reasoningVisionCapabilities: ModelCapabilities = {
  vision: true,
  tools: true,
  streaming: true,
  json_mode: true,
  function_calling: true,
  reasoning: true,
};

const textOnlyCapabilities: ModelCapabilities = {
  vision: false,
  tools: true,
  streaming: true,
  json_mode: true,
  function_calling: true,
  reasoning: false,
};

export function createModelMetadata(
  id: string,
  overrides: Omit<Partial<EnhancedModelMetadata>, "capabilities"> & {
    capabilities?: Partial<ModelCapabilities>;
  } = {},
): EnhancedModelMetadata {
  const { capabilities, ...metadataOverrides } = overrides;
  return {
    id,
    display_name: id,
    provider_id: "fal",
    provider_name: "Fal",
    family: null,
    tier: "pro",
    capabilities: {
      ...defaultCapabilities,
      ...capabilities,
    },
    capability_provenance: "provider_explicit",
    pricing: null,
    limits: {
      context_length: null,
      max_output_tokens: null,
      requests_per_minute: null,
      tokens_per_minute: null,
      ...overrides.limits,
    },
    status: "active",
    release_date: null,
    is_latest: false,
    description: null,
    source: "custom",
    created_at: 0,
    updated_at: 0,
    ...metadataOverrides,
  };
}

export function createReasoningModelMetadata(id: string) {
  return createModelMetadata(id, {
    capabilities: reasoningVisionCapabilities,
  });
}

export function createTextOnlyModelMetadata(id: string) {
  return createModelMetadata(id, {
    capabilities: textOnlyCapabilities,
  });
}

export function renderModelSelector(props: Partial<ModelSelectorProps> = {}) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);

  const mergedProps: ModelSelectorProps = {
    providerType: "custom-local-cli",
    setProviderType: vi.fn(),
    model: "gpt-5.3-codex",
    setModel: vi.fn(),
    activeTheme: "general",
    ...props,
  };

  act(() => {
    root.render(React.createElement(ModelSelector, mergedProps));
  });

  mountedRoots.push({ root, container });
  return { container, props: mergedProps };
}

export function cleanupMountedModelSelectors() {
  while (mountedRoots.length > 0) {
    const mounted = mountedRoots.pop();
    if (!mounted) break;
    act(() => {
      mounted.root.unmount();
    });
    mounted.container.remove();
  }
}

export function getModelSelectorTrigger(
  container: HTMLElement,
): HTMLButtonElement {
  const trigger = container.querySelector<HTMLButtonElement>(
    'button[role="combobox"]',
  );
  if (!trigger) {
    throw new Error("未找到模型选择触发器");
  }
  return trigger;
}

export function clickModelSelectorTrigger(container: HTMLElement) {
  act(() => {
    getModelSelectorTrigger(container).click();
  });
}

export function findBodyButtonByText(text: string): HTMLButtonElement {
  const button = Array.from(
    document.body.querySelectorAll<HTMLButtonElement>("button"),
  ).find((item) => item.textContent?.includes(text));
  if (!button) {
    throw new Error(`未找到包含 ${text} 的按钮`);
  }
  return button;
}

export function clickBodyButtonByText(text: string) {
  act(() => {
    findBodyButtonByText(text).click();
  });
}

export function clickNoProviderGuideDismissButton(container: HTMLElement) {
  const dismissButton = container.querySelector<HTMLButtonElement>(
    'button[aria-label="关闭工具模型未配置提示"]',
  );
  if (!dismissButton) {
    throw new Error("未找到关闭引导按钮");
  }

  act(() => {
    dismissButton.click();
  });
}

export function getBodyText() {
  return document.body.textContent || "";
}
