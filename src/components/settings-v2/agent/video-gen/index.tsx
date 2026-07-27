import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  findConfiguredProviderBySelection,
  useConfiguredProviders,
} from "@/hooks/useConfiguredProviders";
import { getConfig, type Config } from "@/lib/api/appConfig";
import { cn } from "@/lib/utils";
import {
  getVideoModelsForProvider,
  hasMediaGenerationPreferenceOverride,
  isVideoProvider,
  type MediaGenerationPreference,
} from "@/lib/mediaGeneration";
import { MediaPreferenceSection } from "../shared/MediaPreferenceSection";
import { updateMediaPreference } from "../shared/mediaPreferencePersistence";

const DEFAULT_PREFERENCE: MediaGenerationPreference = {
  allowFallback: true,
};

export function VideoGenSettings() {
  const { t } = useTranslation("settings");
  const { providers, loading: providersLoading } = useConfiguredProviders();
  const [config, setConfig] = useState<Config | null>(null);
  const [videoPreference, setVideoPreference] =
    useState<MediaGenerationPreference>(DEFAULT_PREFERENCE);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const nextConfig = await getConfig();
        setConfig(nextConfig);
        setVideoPreference(
          nextConfig.workspace_preferences?.media_defaults?.video ??
            DEFAULT_PREFERENCE,
        );
      } catch (error) {
        console.error("加载视频服务配置失败:", error);
      }
    })();
  }, []);

  const videoProviders = useMemo(
    () =>
      providers.filter((provider) =>
        isVideoProvider(provider.providerId ?? provider.key),
      ),
    [providers],
  );

  const selectedProvider = useMemo(
    () =>
      findConfiguredProviderBySelection(
        videoProviders,
        videoPreference.preferredProviderId,
      ),
    [videoPreference.preferredProviderId, videoProviders],
  );

  const availableModels = useMemo(() => {
    if (!selectedProvider) {
      return [];
    }
    return getVideoModelsForProvider(
      selectedProvider.providerId ?? selectedProvider.key,
      selectedProvider.models,
    );
  }, [selectedProvider]);

  const providerUnavailableLabel =
    videoPreference.preferredProviderId && !selectedProvider
      ? t("settings.mediaGeneration.warning.unavailable", {
          id: videoPreference.preferredProviderId,
        })
      : undefined;

  const modelUnavailableLabel =
    videoPreference.preferredModelId &&
    !availableModels.includes(videoPreference.preferredModelId)
      ? t("settings.mediaGeneration.warning.unavailable", {
          id: videoPreference.preferredModelId,
        })
      : undefined;

  const savePreference = async (
    updater: (current: MediaGenerationPreference) => MediaGenerationPreference,
  ) => {
    try {
      const { config: updatedConfig, preference: nextPreference } =
        await updateMediaPreference("video", updater);
      setConfig(updatedConfig);
      setVideoPreference(nextPreference);
      setMessage({
        type: "success",
        text: t("settings.mediaGeneration.message.saved"),
      });
      setTimeout(() => setMessage(null), 3000);
    } catch (error) {
      console.error("保存视频服务配置失败:", error);
      setMessage({
        type: "error",
        text: t("settings.mediaGeneration.message.saveFailed"),
      });
      setTimeout(() => setMessage(null), 3000);
    }
  };

  const handleProviderChange = (value: string) => {
    const preferredProviderId = value.trim() || undefined;
    const nextProvider = findConfiguredProviderBySelection(
      videoProviders,
      preferredProviderId,
    );
    const nextModels = nextProvider
      ? getVideoModelsForProvider(
          nextProvider.providerId ?? nextProvider.key,
          nextProvider.models,
        )
      : [];
    void savePreference((current) => ({
      preferredProviderId,
      preferredModelId: preferredProviderId
        ? nextModels.includes(current.preferredModelId || "")
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
    void savePreference(() => DEFAULT_PREFERENCE);
  };

  const providerHint = providersLoading
    ? t("settings.mediaGeneration.video.hint.loading")
    : videoProviders.length === 0
      ? t("settings.mediaGeneration.video.hint.empty")
      : t("settings.mediaGeneration.video.hint.ready");

  return (
    <div className="max-w-[820px] space-y-4">
      <MediaPreferenceSection
        title={t("settings.mediaGeneration.video.title")}
        description={t("settings.mediaGeneration.video.description")}
        selectorLabel={t("settings.mediaGeneration.selector.label")}
        selectorDescription={t("settings.mediaGeneration.selector.description")}
        selectionWarningText={providerUnavailableLabel ?? modelUnavailableLabel}
        providerType={videoPreference.preferredProviderId ?? ""}
        setProviderType={handleProviderChange}
        model={videoPreference.preferredModelId ?? ""}
        setModel={handleModelChange}
        setProviderAndModel={handleProviderAndModelChange}
        providerFilter={(provider) =>
          isVideoProvider(provider.providerId ?? provider.key)
        }
        modelFilter={(model, provider) =>
          getVideoModelsForProvider(
            provider.providerId ?? provider.key,
            provider.models,
          ).includes(model.id)
        }
        allowFallback={videoPreference.allowFallback ?? true}
        onAllowFallbackChange={handleFallbackChange}
        fallbackTitle={t("settings.mediaGeneration.fallback.title")}
        fallbackDescription={t(
          "settings.mediaGeneration.video.fallback.description",
        )}
        emptyStateTitle={t("settings.mediaGeneration.video.empty.title")}
        emptyStateDescription={providerHint}
        disabled={!config}
        onReset={handleResetPreference}
        resetLabel={t("settings.mediaGeneration.action.reset")}
        resetDisabled={!hasMediaGenerationPreferenceOverride(videoPreference)}
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
