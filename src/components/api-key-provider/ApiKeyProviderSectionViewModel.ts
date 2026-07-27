import type {
  ProviderWithKeysDisplay,
  UpdateProviderRequest,
} from "@/lib/api/apiKeyProvider";
import { isOemManagedHubProvider } from "@/lib/oemLimeHubProvider";
import {
  buildEnabledModelItems,
  type EnabledModelItem,
} from "./ModelProviderList.utils";

export interface ApiKeyProviderSectionViewModel {
  enabledModelItems: EnabledModelItem[];
  selectedProviderLoginRequired: boolean;
}

export type ProviderSelectionPlan =
  | { type: "none" }
  | { type: "select"; providerId: string | null };

export type DeleteProviderConfigPlan =
  | { type: "missing" }
  | { type: "delete-custom"; providerId: string }
  | {
      type: "reset-system";
      providerId: string;
      apiKeyIds: string[];
      update: Pick<UpdateProviderRequest, "enabled" | "models">;
      clearSelection: boolean;
    };

function hasEnabledApiKey(provider: ProviderWithKeysDisplay): boolean {
  return (provider.api_keys ?? []).some((apiKey) => apiKey.enabled);
}

function hasAnyApiKey(provider: ProviderWithKeysDisplay): boolean {
  return (
    (typeof provider.api_key_count === "number" && provider.api_key_count > 0) ||
    (provider.api_keys ?? []).length > 0
  );
}

function hasConfiguredModel(provider: ProviderWithKeysDisplay): boolean {
  return (provider.models ?? []).some(
    (model) => model.id.trim().length > 0,
  );
}

export function isSelectedProviderLoginRequired({
  provider,
  exposeOemLoginPrompt,
}: {
  provider: ProviderWithKeysDisplay | null;
  exposeOemLoginPrompt: boolean;
}): boolean {
  return Boolean(
    provider &&
      exposeOemLoginPrompt &&
      isOemManagedHubProvider(provider) &&
      !hasAnyApiKey(provider) &&
      !hasEnabledApiKey(provider) &&
      !hasConfiguredModel(provider),
  );
}

export function buildApiKeyProviderSectionViewModel({
  providers,
  selectedProvider,
  exposeOemLoginPrompt,
}: {
  providers: ProviderWithKeysDisplay[];
  selectedProvider: ProviderWithKeysDisplay | null;
  exposeOemLoginPrompt: boolean;
}): ApiKeyProviderSectionViewModel {
  return {
    enabledModelItems: buildEnabledModelItems(providers, {
      exposeOemLoginPrompt,
    }),
    selectedProviderLoginRequired: isSelectedProviderLoginRequired({
      provider: selectedProvider,
      exposeOemLoginPrompt,
    }),
  };
}

export function planEnabledModelSelection({
  enabledModelItems,
  selectedProviderId,
  showAddModelFlow,
}: {
  enabledModelItems: Array<Pick<EnabledModelItem, "id" | "status">>;
  selectedProviderId: string | null;
  showAddModelFlow: boolean;
}): ProviderSelectionPlan {
  if (showAddModelFlow) {
    return { type: "none" };
  }

  if (enabledModelItems.length === 0) {
    return selectedProviderId
      ? { type: "select", providerId: null }
      : { type: "none" };
  }

  const loginRequiredItem = enabledModelItems.find(
    (item) => item.status === "login_required",
  );
  if (loginRequiredItem && selectedProviderId !== loginRequiredItem.id) {
    return { type: "select", providerId: loginRequiredItem.id };
  }

  if (
    selectedProviderId &&
    enabledModelItems.some((item) => item.id === selectedProviderId)
  ) {
    return { type: "none" };
  }

  return { type: "select", providerId: enabledModelItems[0]!.id };
}

export function planDeleteProviderConfig({
  providers,
  providerId,
  selectedProviderId,
}: {
  providers: ProviderWithKeysDisplay[];
  providerId: string;
  selectedProviderId: string | null;
}): DeleteProviderConfigPlan {
  const targetProvider = providers.find((provider) => provider.id === providerId);
  if (!targetProvider) {
    return { type: "missing" };
  }

  if (!targetProvider.is_system) {
    return { type: "delete-custom", providerId };
  }

  return {
    type: "reset-system",
    providerId,
    apiKeyIds: (targetProvider.api_keys ?? []).map((apiKey) => apiKey.id),
    update: {
      enabled: false,
      models: [],
    },
    clearSelection: selectedProviderId === providerId,
  };
}
