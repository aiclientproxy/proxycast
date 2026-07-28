import { useEffect } from "react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanupMountedRoots,
  flushEffects,
  mountHarness,
  setupReactActEnvironment,
  type MountedRoot,
} from "@/components/workspace/hooks/testUtils";
import { useModelRegistry } from "./useModelRegistry";

const modelRegistryMocks = vi.hoisted(() => ({
  getModelRegistry: vi.fn(),
  getModelPreferences: vi.fn(),
  getModelSyncState: vi.fn(),
  refreshModelRegistry: vi.fn(),
  subscribeModelRegistryUpdates: vi.fn(),
  toggleModelFavorite: vi.fn(),
  hideModel: vi.fn(),
}));

vi.mock("@/lib/api/modelRegistry", () => ({
  modelRegistryApi: modelRegistryMocks,
}));

type HookValue = ReturnType<typeof useModelRegistry>;

function HookHarness({ onReady }: { onReady: (value: HookValue) => void }) {
  const value = useModelRegistry();
  useEffect(() => onReady(value), [onReady, value]);
  return null;
}

function model(id: string) {
  return {
    id,
    display_name: id,
    provider_id: "openai",
    provider_name: "OpenAI",
    tier: "pro",
    status: "active",
    is_latest: true,
  };
}

const mountedRoots: MountedRoot[] = [];

describe("useModelRegistry", () => {
  let latestValue: HookValue | null = null;
  let modelUpdateCallback: (() => void) | undefined;
  let unsubscribe: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    setupReactActEnvironment();
    vi.clearAllMocks();
    latestValue = null;
    modelUpdateCallback = undefined;
    unsubscribe = vi.fn();
    modelRegistryMocks.getModelPreferences.mockResolvedValue([]);
    modelRegistryMocks.getModelSyncState.mockResolvedValue({
      last_sync_at: 1,
    });
    modelRegistryMocks.subscribeModelRegistryUpdates.mockImplementation(
      (callback) => {
        modelUpdateCallback = callback;
        return unsubscribe;
      },
    );
  });

  afterEach(() => {
    cleanupMountedRoots(mountedRoots);
  });

  it("收到 model/list/updated 后应强制重读并更新已挂载模型列表", async () => {
    modelRegistryMocks.getModelRegistry
      .mockResolvedValueOnce([model("gpt-4.1")])
      .mockResolvedValueOnce([model("gpt-5")]);
    mountHarness(
      HookHarness,
      {
        onReady: (value) => {
          latestValue = value;
        },
      },
      mountedRoots,
    );
    await flushEffects(6);

    expect(latestValue?.models.map((item) => item.id)).toEqual(["gpt-4.1"]);
    expect(modelUpdateCallback).toBeTypeOf("function");

    await act(async () => {
      modelUpdateCallback?.();
    });
    await flushEffects(6);

    expect(modelRegistryMocks.getModelRegistry).toHaveBeenNthCalledWith(2, {
      forceRefresh: true,
      includeHidden: false,
    });
    expect(latestValue?.models.map((item) => item.id)).toEqual(["gpt-5"]);

    cleanupMountedRoots(mountedRoots);
    expect(unsubscribe).toHaveBeenCalledOnce();
  });
});
