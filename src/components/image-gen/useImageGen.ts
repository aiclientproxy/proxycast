/**
 * @file 图片 Provider / 模型选择 Hook
 * @description 只负责图片能力选择；任务执行归工作台 current media task runtime。
 * @module components/image-gen/useImageGen
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApiKeyProvider } from "@/hooks/useApiKeyProvider";
import {
  providerModelIds,
  type ProviderModelConfig,
} from "@/lib/api/apiKeyProvider";
import {
  findDefaultImageCapabilityProvider,
  isImageCapabilityProvider,
  resolveImageCapabilityModels,
} from "@/lib/imageGen/catalog";
import { isDebugFlagEnabled } from "@/lib/perfDebug";
import { scheduleMinimumDelayIdleTask } from "@/lib/utils/scheduleMinimumDelayIdleTask";

interface UseImageGenOptions {
  preferredProviderId?: string;
  preferredModelId?: string;
  allowFallback?: boolean;
  providerLoadEnabled?: boolean;
  providerLoadMode?: "immediate" | "deferred";
  providerDeferredDelayMs?: number;
  selectionScopeKey?: string;
}

const PROVIDER_DEBUG_KEY = "lime:provider-debug";
const IMAGE_GEN_PROVIDER_IDLE_TIMEOUT_MS = 1_500;

function imageGenDebugLog(...args: unknown[]): void {
  if (isDebugFlagEnabled(PROVIDER_DEBUG_KEY)) {
    console.debug("[image-gen]", ...args);
  }
}

function isImageGenProvider(provider: {
  id: string;
  type: string;
  models?: ProviderModelConfig[];
  api_host?: string;
}): boolean {
  return isImageCapabilityProvider({
    id: provider.id,
    type: provider.type,
    models: providerModelIds(provider.models),
    api_host: provider.api_host,
  });
}

export function useImageGen(options: UseImageGenOptions = {}) {
  const providerLoadEnabled = options.providerLoadEnabled ?? true;
  const providerLoadMode = options.providerLoadMode ?? "immediate";
  const providerDeferredDelayMs =
    options.providerDeferredDelayMs ?? IMAGE_GEN_PROVIDER_IDLE_TIMEOUT_MS;
  const [providerLoadReady, setProviderLoadReady] = useState(
    providerLoadEnabled && providerLoadMode !== "deferred",
  );
  const {
    providers,
    loading: providersLoading,
    refresh,
  } = useApiKeyProvider({
    autoLoad: providerLoadReady,
  });
  const preferredProviderId = options.preferredProviderId?.trim() || "";
  const preferredModelId = options.preferredModelId?.trim() || "";
  const allowFallback = options.allowFallback ?? true;
  const selectionScopeKey = options.selectionScopeKey?.trim() || "";
  const [selectedProviderId, setSelectedProviderId] = useState("");
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedSize, setSelectedSize] = useState("1024x1024");
  const syncedPreferredProviderIdRef = useRef("");
  const syncedPreferredModelIdRef = useRef("");
  const hasManualProviderSelectionRef = useRef(false);
  const syncedSelectionScopeKeyRef = useRef("");

  useEffect(() => {
    if (syncedSelectionScopeKeyRef.current === selectionScopeKey) {
      return;
    }

    syncedSelectionScopeKeyRef.current = selectionScopeKey;
    hasManualProviderSelectionRef.current = false;
    syncedPreferredProviderIdRef.current = "";
    syncedPreferredModelIdRef.current = "";
    setSelectedProviderId("");
    setSelectedModelId("");
  }, [selectionScopeKey]);

  useEffect(() => {
    if (!providerLoadEnabled || providerLoadMode !== "deferred") {
      if (providerLoadEnabled) {
        setProviderLoadReady(true);
      }
      return;
    }

    if (providerLoadReady) {
      return;
    }

    return scheduleMinimumDelayIdleTask(() => setProviderLoadReady(true), {
      minimumDelayMs: providerDeferredDelayMs,
      idleTimeoutMs: IMAGE_GEN_PROVIDER_IDLE_TIMEOUT_MS,
    });
  }, [
    providerDeferredDelayMs,
    providerLoadEnabled,
    providerLoadMode,
    providerLoadReady,
  ]);

  const ensureProvidersLoaded = useCallback(async () => {
    setProviderLoadReady(true);
    await refresh();
  }, [refresh]);

  const availableProviders = useMemo(() => {
    const imageProviders = providers.filter(
      (provider) =>
        provider.enabled &&
        provider.api_key_count > 0 &&
        isImageGenProvider(provider),
    );
    imageGenDebugLog(
      "[useImageGen] 图片 Provider:",
      imageProviders.map((provider) => provider.id),
    );
    return imageProviders;
  }, [providers]);

  const selectedProvider = useMemo(
    () =>
      availableProviders.find((provider) => provider.id === selectedProviderId),
    [availableProviders, selectedProviderId],
  );
  const preferredProviderUnavailable = useMemo(
    () =>
      Boolean(preferredProviderId) &&
      !providersLoading &&
      !availableProviders.some(
        (provider) => provider.id === preferredProviderId,
      ),
    [availableProviders, preferredProviderId, providersLoading],
  );
  const availableModels = useMemo(
    () =>
      selectedProvider
        ? resolveImageCapabilityModels({
            id: selectedProvider.id,
            type: selectedProvider.type,
            models: providerModelIds(selectedProvider.models),
            api_host: selectedProvider.api_host,
          })
        : [],
    [selectedProvider],
  );
  const selectedModel = useMemo(
    () => availableModels.find((model) => model.id === selectedModelId),
    [availableModels, selectedModelId],
  );

  useEffect(() => {
    if (availableProviders.length === 0) {
      if (selectedProviderId) {
        setSelectedProviderId("");
      }
      return;
    }

    const preferredProvider = preferredProviderId
      ? availableProviders.find(
          (provider) => provider.id === preferredProviderId,
        )
      : undefined;
    const preferredProviderChanged =
      syncedPreferredProviderIdRef.current !== preferredProviderId;
    const currentProviderAvailable = availableProviders.some(
      (provider) => provider.id === selectedProviderId,
    );

    if (preferredProviderChanged) {
      syncedPreferredProviderIdRef.current = preferredProviderId;
      if (preferredProvider && selectedProviderId !== preferredProvider.id) {
        hasManualProviderSelectionRef.current = false;
        setSelectedProviderId(preferredProvider.id);
        return;
      }
    }

    if (preferredProviderId && !preferredProvider && !allowFallback) {
      if (
        selectedProviderId &&
        currentProviderAvailable &&
        hasManualProviderSelectionRef.current
      ) {
        return;
      }
      if (selectedProviderId) {
        hasManualProviderSelectionRef.current = false;
        setSelectedProviderId("");
      }
      return;
    }

    if (selectedProviderId && currentProviderAvailable) {
      return;
    }

    const nextProvider = preferredProvider
      ? preferredProvider
      : allowFallback
        ? findDefaultImageCapabilityProvider(
            availableProviders.map((provider) => ({
              id: provider.id,
              type: provider.type,
              models: providerModelIds(provider.models),
              api_host: provider.api_host,
            })),
          )
        : undefined;
    if (nextProvider) {
      hasManualProviderSelectionRef.current = false;
      setSelectedProviderId(nextProvider.id);
    }
  }, [
    allowFallback,
    availableProviders,
    preferredProviderId,
    selectedProviderId,
  ]);

  useEffect(() => {
    if (availableModels.length === 0) {
      if (selectedModelId) {
        setSelectedModelId("");
      }
      return;
    }

    const preferredModelChanged =
      syncedPreferredModelIdRef.current !== preferredModelId;
    if (preferredModelChanged) {
      syncedPreferredModelIdRef.current = preferredModelId;
    }

    if (
      preferredModelId &&
      availableModels.some((model) => model.id === preferredModelId)
    ) {
      if (selectedModelId !== preferredModelId) {
        setSelectedModelId(preferredModelId);
      }
      return;
    }

    if (!availableModels.some((model) => model.id === selectedModelId)) {
      setSelectedModelId(availableModels[0]?.id ?? "");
    }
  }, [availableModels, preferredModelId, selectedModelId]);

  const handleProviderChange = useCallback(
    (providerId: string) => {
      hasManualProviderSelectionRef.current = true;
      setSelectedProviderId(providerId);
      const provider = availableProviders.find(
        (item) => item.id === providerId,
      );
      const firstModel = provider
        ? resolveImageCapabilityModels({
            id: provider.id,
            type: provider.type,
            models: providerModelIds(provider.models),
            api_host: provider.api_host,
          })[0]
        : undefined;
      if (firstModel) {
        setSelectedModelId(firstModel.id);
      }
    },
    [availableProviders],
  );

  return {
    availableProviders,
    selectedProvider,
    selectedProviderId,
    setSelectedProviderId: handleProviderChange,
    ensureProvidersLoaded,
    providersLoading,
    preferredProviderUnavailable,
    availableModels,
    selectedModel,
    selectedModelId,
    setSelectedModelId,
    selectedSize,
    setSelectedSize,
  };
}
