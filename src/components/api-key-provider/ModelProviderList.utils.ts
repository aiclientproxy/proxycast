import type { ProviderWithKeysDisplay } from "@/lib/api/apiKeyProvider";
import { isOemManagedHubProvider } from "@/lib/oemLimeHubProvider";

export interface EnabledModelListOptions {
  exposeOemLoginPrompt?: boolean;
}

export interface EnabledModelItem {
  id: string;
  provider: ProviderWithKeysDisplay;
  providerName: string;
  modelId: string | null;
  isDefault: boolean;
  enabledApiKeyCount: number;
  status: "ready" | "login_required";
}

function getEnabledApiKeyCount(provider: ProviderWithKeysDisplay): number {
  return provider.api_keys?.filter((apiKey) => apiKey.enabled).length ?? 0;
}

function getProviderDefaultModel(
  provider: ProviderWithKeysDisplay,
): string | null {
  return (
    provider.models?.find((model) => model.id.trim().length > 0)?.id.trim() ??
    null
  );
}

function isKeylessLocalProvider(provider: ProviderWithKeysDisplay): boolean {
  return (
    provider.enabled &&
    provider.type === "ollama" &&
    provider.api_host.trim().length > 0
  );
}

export function isProviderVisibleInEnabledModelList(
  provider: ProviderWithKeysDisplay,
  options: EnabledModelListOptions = {},
): boolean {
  if (!provider.enabled) {
    return false;
  }

  if (isOemManagedHubProvider(provider)) {
    return Boolean(
      options.exposeOemLoginPrompt &&
      getEnabledApiKeyCount(provider) === 0 &&
      !getProviderDefaultModel(provider),
    );
  }

  return (
    getEnabledApiKeyCount(provider) > 0 ||
    Boolean(getProviderDefaultModel(provider)) ||
    isKeylessLocalProvider(provider)
  );
}

export function buildEnabledModelItems(
  providers: ProviderWithKeysDisplay[],
  options: EnabledModelListOptions = {},
): EnabledModelItem[] {
  const items = providers
    .filter((provider) =>
      isProviderVisibleInEnabledModelList(provider, options),
    )
    .sort((a, b) => a.sort_order - b.sort_order)
    .map((provider): EnabledModelItem => {
      const modelId = getProviderDefaultModel(provider);
      const enabledApiKeyCount = getEnabledApiKeyCount(provider);
      const loginRequired =
        isOemManagedHubProvider(provider) &&
        enabledApiKeyCount === 0 &&
        !modelId;

      return {
        id: provider.id,
        provider,
        providerName: provider.name,
        modelId,
        isDefault: false,
        enabledApiKeyCount,
        status: loginRequired ? "login_required" : "ready",
      };
    });

  const defaultReadyIndex = items.findIndex((item) => item.status === "ready");
  if (defaultReadyIndex >= 0) {
    items[defaultReadyIndex] = {
      ...items[defaultReadyIndex],
      isDefault: true,
    };
  }

  return items;
}
