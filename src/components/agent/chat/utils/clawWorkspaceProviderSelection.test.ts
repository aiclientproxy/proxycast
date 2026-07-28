import { beforeEach, describe, expect, it, vi } from "vitest";
import type { EnhancedModelMetadata } from "@/lib/types/modelRegistry";
import { resolveClawWorkspaceProviderSelection } from "./clawWorkspaceProviderSelection";

const { mockGetModelRegistry, mockFilterModelsByTheme } = vi.hoisted(() => ({
  mockGetModelRegistry: vi.fn(),
  mockFilterModelsByTheme: vi.fn(),
}));

vi.mock("@/lib/api/modelRegistry", () => ({
  modelRegistryApi: { getModelRegistry: mockGetModelRegistry },
}));

vi.mock("./modelThemePolicy", () => ({
  filterModelsByTheme: mockFilterModelsByTheme,
}));

function createModel(
  providerId: string,
  id: string,
  overrides: Partial<EnhancedModelMetadata> = {},
): EnhancedModelMetadata {
  return {
    id,
    display_name: id,
    provider_id: providerId,
    provider_name: providerId,
    family: null,
    tier: "pro",
    capabilities: { vision: false, reasoning: false },
    task_families: ["chat"],
    input_modalities: ["text"],
    output_modalities: ["text"],
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
    source: "api",
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

describe("resolveClawWorkspaceProviderSelection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetModelRegistry.mockResolvedValue([]);
    mockFilterModelsByTheme.mockImplementation(
      (_theme: string | undefined, models: EnhancedModelMetadata[]) => ({
        models,
        usedFallback: false,
        filteredOutCount: 0,
        policyName: "mock",
      }),
    );
  });

  it("应保留 App Server ready catalog 中的当前精确模型", async () => {
    mockGetModelRegistry.mockResolvedValue([
      createModel("provider-a", "model-a"),
      createModel("provider-b", "model-b", { is_default: true }),
    ]);

    await expect(
      resolveClawWorkspaceProviderSelection({
        currentProviderType: "provider-a",
        currentModel: "model-a",
        theme: "general",
      }),
    ).resolves.toEqual({ providerType: "provider-a", model: "model-a" });
  });

  it("当前模型消失时应只在同一 ready provider 内选择", async () => {
    mockGetModelRegistry.mockResolvedValue([
      createModel("provider-a", "model-a2"),
      createModel("provider-b", "model-b", { is_default: true }),
    ]);

    await expect(
      resolveClawWorkspaceProviderSelection({
        currentProviderType: "provider-a",
        currentModel: "removed-model",
      }),
    ).resolves.toEqual({ providerType: "provider-a", model: "model-a2" });
  });

  it("关闭 provider fallback 时不得跨到其他 provider", async () => {
    mockGetModelRegistry.mockResolvedValue([
      createModel("provider-b", "model-b", { is_default: true }),
    ]);

    await expect(
      resolveClawWorkspaceProviderSelection({
        currentProviderType: "provider-a",
        currentModel: "model-a",
        allowProviderFallback: false,
      }),
    ).resolves.toBeNull();
  });

  it("当前 provider 不可执行时应消费 App Server 唯一默认模型", async () => {
    mockGetModelRegistry.mockResolvedValue([
      createModel("provider-a", "model-a"),
      createModel("provider-b", "model-b", { is_default: true }),
    ]);

    await expect(
      resolveClawWorkspaceProviderSelection({
        currentProviderType: "provider-missing",
        currentModel: "model-missing",
      }),
    ).resolves.toEqual({ providerType: "provider-b", model: "model-b" });
  });

  it("未指定 provider 时应消费 App Server 唯一默认模型", async () => {
    mockGetModelRegistry.mockResolvedValue([
      createModel("provider-a", "model-a"),
      createModel("provider-b", "model-b", { is_default: true }),
    ]);

    await expect(resolveClawWorkspaceProviderSelection({})).resolves.toEqual({
      providerType: "provider-b",
      model: "model-b",
    });
  });

  it("普通聊天选择应继续拒绝纯图片输出模型", async () => {
    mockGetModelRegistry.mockResolvedValue([
      createModel("provider-image", "image-model", {
        is_default: true,
        task_families: ["image_generation"],
        output_modalities: ["image"],
      }),
      createModel("provider-chat", "chat-model"),
    ]);

    await expect(resolveClawWorkspaceProviderSelection({})).resolves.toEqual({
      providerType: "provider-chat",
      model: "chat-model",
    });
  });

  it("App Server ready catalog 为空时应返回 null", async () => {
    await expect(resolveClawWorkspaceProviderSelection({})).resolves.toBeNull();
  });
});
