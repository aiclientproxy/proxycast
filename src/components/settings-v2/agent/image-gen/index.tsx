import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { getConfig, type Config } from "@/lib/api/appConfig";
import {
  findConfiguredProviderBySelection,
  type ConfiguredProvider,
  useConfiguredProviders,
} from "@/hooks/useConfiguredProviders";
import { useProviderModels } from "@/hooks/useProviderModels";
import { cn } from "@/lib/utils";
import {
  inferInputModalities,
  inferModelAliasSource,
  inferModelCapabilities,
  inferModelDeploymentSource,
  inferModelManagementPlane,
  inferModelTaskFamilies,
  inferOutputModalities,
  inferRuntimeFeatures,
} from "@/lib/model/inferModelCapabilities";
import type { EnhancedModelMetadata } from "@/lib/types/modelRegistry";
import { resolveProviderModelLoadOptions } from "@/lib/model/providerModelLoadOptions";
import {
  hasMediaGenerationPreferenceOverride,
  type MediaGenerationPreference,
} from "@/lib/mediaGeneration";
import {
  resolveImageCapabilityModelIds,
  resolveImageCapabilityModels,
  isImageCapabilityProvider,
  isImageCapabilityModelMetadata,
  type ImageGenModel,
} from "@/lib/imageGen/catalog";
import { MediaPreferenceSection } from "../shared/MediaPreferenceSection";
import { updateMediaPreference } from "../shared/mediaPreferencePersistence";

const DEFAULT_MEDIA_PREFERENCE: MediaGenerationPreference = {
  allowFallback: true,
};

function buildImageModelMetadata(
  provider: ConfiguredProvider,
  model: ImageGenModel,
): EnhancedModelMetadata {
  const providerId = provider.providerId ?? provider.key;
  const taxonomyParams = {
    modelId: model.id,
    providerId,
    providerModelId: model.id,
  };
  return {
    id: model.id,
    display_name: model.name || model.id,
    provider_id: providerId,
    provider_name: provider.label,
    family: null,
    tier: "pro",
    capabilities: inferModelCapabilities(taxonomyParams),
    task_families: inferModelTaskFamilies(taxonomyParams),
    input_modalities: inferInputModalities(taxonomyParams),
    output_modalities: inferOutputModalities(taxonomyParams),
    runtime_features: inferRuntimeFeatures(taxonomyParams),
    deployment_source: inferModelDeploymentSource(taxonomyParams),
    management_plane: inferModelManagementPlane(taxonomyParams),
    canonical_model_id: null,
    provider_model_id: model.id,
    alias_source: inferModelAliasSource(taxonomyParams),
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
    source: "embedded",
    created_at: 0,
    updated_at: 0,
  };
}

export function ImageGenSettings() {
  const { t } = useTranslation("settings");
  const [config, setConfig] = useState<Config | null>(null);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const [globalImagePreference, setGlobalImagePreference] =
    useState<MediaGenerationPreference>(DEFAULT_MEDIA_PREFERENCE);
  const { providers, loading: providersLoading } = useConfiguredProviders();

  useEffect(() => {
    void (async () => {
      try {
        const nextConfig = await getConfig();
        setConfig(nextConfig);
        setGlobalImagePreference(
          nextConfig.workspace_preferences?.media_defaults?.image ??
            DEFAULT_MEDIA_PREFERENCE,
        );
      } catch (error) {
        console.error("加载图片服务配置失败:", error);
      }
    })();
  }, []);

  const imageProviders = useMemo(
    () =>
      providers.filter(
        (provider) =>
          provider.authStatus === "login_required" ||
          isImageCapabilityProvider({
            id: provider.providerId ?? provider.key,
            type: provider.type,
            models: provider.models,
            api_host: provider.apiHost,
          }),
      ),
    [providers],
  );

  const selectedProvider = useMemo(
    () =>
      findConfiguredProviderBySelection(
        imageProviders,
        globalImagePreference.preferredProviderId,
      ),
    [globalImagePreference.preferredProviderId, imageProviders],
  );

  const availableModelIds = useMemo(() => {
    if (!selectedProvider) {
      return [];
    }

    return resolveImageCapabilityModelIds({
      id: selectedProvider.providerId ?? selectedProvider.key,
      type: selectedProvider.type,
      models: selectedProvider.models,
      api_host: selectedProvider.apiHost,
    });
  }, [selectedProvider]);
  const selectedProviderModelLoadOptions = useMemo(
    () =>
      resolveProviderModelLoadOptions({
        providerId: selectedProvider?.providerId,
        providerType: selectedProvider?.type,
        apiHost: selectedProvider?.apiHost,
        hasApiKey: selectedProvider?.hasApiKey,
        hasDeclaredModels: Boolean(
          selectedProvider?.models?.some((modelId) => modelId.trim()),
        ),
      }),
    [
      selectedProvider?.apiHost,
      selectedProvider?.models,
      selectedProvider?.hasApiKey,
      selectedProvider?.providerId,
      selectedProvider?.type,
    ],
  );
  const { models: selectedProviderModels } = useProviderModels(
    selectedProvider,
    {
      returnFullMetadata: true,
      autoLoad: Boolean(selectedProvider),
      ...selectedProviderModelLoadOptions,
    },
  );
  const imageSelectableModelIds = useMemo(() => {
    if (!selectedProvider) {
      return availableModelIds;
    }

    const modelIds = new Set(availableModelIds);
    for (const model of selectedProviderModels) {
      if (
        isImageCapabilityModelMetadata(
          {
            id: selectedProvider.providerId ?? selectedProvider.key,
            type: selectedProvider.type,
            models: selectedProvider.models,
            api_host: selectedProvider.apiHost,
          },
          model,
        )
      ) {
        modelIds.add(model.id);
      }
    }

    return Array.from(modelIds);
  }, [availableModelIds, selectedProvider, selectedProviderModels]);

  const providerUnavailableLabel =
    globalImagePreference.preferredProviderId && !selectedProvider
      ? t("settings.mediaGeneration.warning.unavailable", {
          id: globalImagePreference.preferredProviderId,
        })
      : undefined;

  const modelUnavailableLabel =
    globalImagePreference.preferredModelId &&
    !imageSelectableModelIds.includes(globalImagePreference.preferredModelId)
      ? t("settings.mediaGeneration.warning.unavailable", {
          id: globalImagePreference.preferredModelId,
        })
      : undefined;

  const showMessage = (type: "success" | "error", text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 3000);
  };

  const savePreference = async (
    updater: (current: MediaGenerationPreference) => MediaGenerationPreference,
  ) => {
    try {
      const { config: updatedConfig, preference: nextPreference } =
        await updateMediaPreference("image", updater);
      setConfig(updatedConfig);
      setGlobalImagePreference(nextPreference);
      showMessage("success", t("settings.mediaGeneration.message.saved"));
    } catch (error) {
      console.error("保存图片服务配置失败:", error);
      showMessage("error", t("settings.mediaGeneration.message.saveFailed"));
    }
  };

  const handleProviderChange = (value: string) => {
    const preferredProviderId = value.trim() || undefined;
    const nextProvider = findConfiguredProviderBySelection(
      imageProviders,
      preferredProviderId,
    );
    const nextModelIds = nextProvider
      ? resolveImageCapabilityModelIds({
          id: nextProvider.providerId ?? nextProvider.key,
          type: nextProvider.type,
          models: nextProvider.models,
          api_host: nextProvider.apiHost,
        })
      : [];
    void savePreference((current) => ({
      preferredProviderId,
      preferredModelId: preferredProviderId
        ? nextModelIds.includes(current.preferredModelId || "")
          ? current.preferredModelId
          : undefined
        : undefined,
      allowFallback: current.allowFallback ?? true,
    }));
  };

  const handleModelChange = (value: string) => {
    void savePreference((current) => ({
      ...current,
      preferredModelId: value.trim() || undefined,
      allowFallback: current.allowFallback ?? true,
    }));
  };

  const handleProviderAndModelChange = (
    providerValue: string,
    modelValue: string,
  ) => {
    void savePreference((current) => ({
      preferredProviderId: providerValue.trim() || undefined,
      preferredModelId:
        providerValue.trim() && modelValue.trim()
          ? modelValue.trim()
          : undefined,
      allowFallback: current.allowFallback ?? true,
    }));
  };

  const handleFallbackChange = (value: boolean) => {
    void savePreference((current) => ({
      ...current,
      allowFallback: value,
    }));
  };

  const handleResetPreference = () => {
    void savePreference(() => DEFAULT_MEDIA_PREFERENCE);
  };

  const getImageFallbackModels = useCallback((provider: ConfiguredProvider) => {
    const catalogModels = resolveImageCapabilityModels({
      id: provider.providerId ?? provider.key,
      type: provider.type,
      models: provider.models,
      api_host: provider.apiHost,
    });
    return catalogModels.map((model) =>
      buildImageModelMetadata(provider, model),
    );
  }, []);

  const providerHint = providersLoading
    ? t("settings.mediaGeneration.image.hint.loading")
    : imageProviders.length === 0
      ? t("settings.mediaGeneration.image.hint.empty")
      : t("settings.mediaGeneration.image.hint.ready");

  return (
    <div className="max-w-[820px] space-y-4">
      <MediaPreferenceSection
        title={t("settings.mediaGeneration.image.title")}
        description={t("settings.mediaGeneration.image.description")}
        selectorLabel={t("settings.mediaGeneration.selector.label")}
        selectorDescription={t("settings.mediaGeneration.selector.description")}
        selectionWarningText={providerUnavailableLabel ?? modelUnavailableLabel}
        providerType={globalImagePreference.preferredProviderId ?? ""}
        setProviderType={handleProviderChange}
        model={globalImagePreference.preferredModelId ?? ""}
        setModel={handleModelChange}
        setProviderAndModel={handleProviderAndModelChange}
        providerFilter={(provider) =>
          provider.authStatus === "login_required" ||
          isImageCapabilityProvider({
            id: provider.providerId ?? provider.key,
            type: provider.type,
            models: provider.models,
            api_host: provider.apiHost,
          })
        }
        modelFilter={(model, provider) =>
          isImageCapabilityModelMetadata(
            {
              id: provider.providerId ?? provider.key,
              type: provider.type,
              models: provider.models,
              api_host: provider.apiHost,
            },
            model,
          )
        }
        getFallbackModels={getImageFallbackModels}
        allowFallback={globalImagePreference.allowFallback ?? true}
        onAllowFallbackChange={handleFallbackChange}
        fallbackTitle={t("settings.mediaGeneration.fallback.title")}
        fallbackDescription={t(
          "settings.mediaGeneration.image.fallback.description",
        )}
        emptyStateTitle={t("settings.mediaGeneration.image.empty.title")}
        emptyStateDescription={providerHint}
        disabled={!config}
        onReset={handleResetPreference}
        resetLabel={t("settings.mediaGeneration.action.reset")}
        resetDisabled={
          !hasMediaGenerationPreferenceOverride(globalImagePreference)
        }
      />

      {message ? (
        <div
          className={cn(
            "flex items-center gap-2 rounded-[20px] border px-4 py-3",
            message.type === "success"
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : "border-rose-200 bg-rose-50 text-rose-700",
          )}
        >
          {message.type === "success" ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <span className="text-sm">{message.text}</span>
        </div>
      ) : null}
    </div>
  );
}
