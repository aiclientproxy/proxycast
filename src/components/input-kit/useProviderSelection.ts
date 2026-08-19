import { useCallback, useEffect, useState } from "react";
import type { ConfiguredProvider } from "@/hooks/useConfiguredProviders";
import type {
  EnhancedModelMetadata,
  ModelReasoningEffortLevel,
} from "@/lib/types/modelRegistry";

interface ProviderModelOption {
  id: string;
  metadata: EnhancedModelMetadata;
  compatibilityIssue: unknown;
}

function modelBelongsToProvider(
  model: EnhancedModelMetadata,
  provider: ConfiguredProvider,
): boolean {
  const modelProviderId = model.provider_id?.trim().toLowerCase();
  if (!modelProviderId) {
    return true;
  }

  return [
    provider.key,
    provider.providerId,
    provider.registryId,
    provider.fallbackRegistryId,
  ]
    .filter((value): value is string => Boolean(value?.trim()))
    .some((value) => value.trim().toLowerCase() === modelProviderId);
}

export function resolvePendingProviderModel(params: {
  pendingProviderSelection: string | null;
  selectedProvider: ConfiguredProvider | null;
  modelOptions: ProviderModelOption[];
}): string {
  const selectedProvider = params.selectedProvider;
  if (!params.pendingProviderSelection || !selectedProvider) {
    return "";
  }

  return (
    params.modelOptions.find(
      (item) =>
        !item.compatibilityIssue &&
        modelBelongsToProvider(item.metadata, selectedProvider),
    )?.id ?? ""
  );
}

export function useProviderSelection(params: {
  providerType: string;
  commitProviderAndModel: (providerType: string, model: string) => void;
}) {
  const { providerType, commitProviderAndModel } = params;
  const [pendingProviderSelection, setPendingProviderSelection] = useState<
    string | null
  >(null);
  const queueOrCommitProviderSelection = useCallback(
    (nextProviderType: string, nextModel: string) => {
      if (nextModel.trim()) {
        setPendingProviderSelection(null);
        commitProviderAndModel(nextProviderType, nextModel);
        return true;
      }

      setPendingProviderSelection(nextProviderType);
      return false;
    },
    [commitProviderAndModel],
  );
  const clearPendingProviderSelection = useCallback(() => {
    setPendingProviderSelection(null);
  }, []);

  return {
    pendingProviderSelection,
    effectiveProviderType: pendingProviderSelection ?? providerType,
    queueOrCommitProviderSelection,
    clearPendingProviderSelection,
  };
}

export function useCommitPendingProviderSelection(params: {
  pendingProviderSelection: string | null;
  pendingProviderModel: string;
  ready: boolean;
  commitProviderAndModel: (providerType: string, model: string) => void;
  clearPendingProviderSelection: () => void;
  setReasoningEffort?: (value: ModelReasoningEffortLevel | "") => void;
}) {
  const {
    pendingProviderSelection,
    pendingProviderModel,
    ready,
    commitProviderAndModel,
    clearPendingProviderSelection,
    setReasoningEffort,
  } = params;

  useEffect(() => {
    if (!pendingProviderSelection) return;
    if (!ready) return;
    if (!pendingProviderModel) return;

    clearPendingProviderSelection();
    commitProviderAndModel(pendingProviderSelection, pendingProviderModel);
    setReasoningEffort?.("");
  }, [
    clearPendingProviderSelection,
    commitProviderAndModel,
    pendingProviderModel,
    pendingProviderSelection,
    ready,
    setReasoningEffort,
  ]);
}
