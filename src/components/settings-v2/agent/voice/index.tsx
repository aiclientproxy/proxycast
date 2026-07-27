import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";
import { open as openDialog } from "@/lib/desktop-host/plugin-dialog";
import type { TFunction } from "i18next";
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FolderOpen,
  HardDrive,
  Loader2,
  Trash2,
  Wand2,
  type LucideIcon,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { WorkbenchInfoTip } from "@/components/media/WorkbenchInfoTip";
import { InstructionEditor } from "@/components/voice/InstructionEditor";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { getConfig, type Config } from "@/lib/api/appConfig";
import {
  getVoiceInputConfig,
  saveVoiceInputConfig,
  type VoiceInputConfig,
  type VoiceInstruction,
} from "@/lib/api/asrProvider";
import {
  deleteVoiceModel,
  downloadVoiceModel,
  getVoiceModelInstallState,
  listenVoiceModelDownloadProgress,
  listVoiceModelCatalog,
  setDefaultVoiceModel,
  testTranscribeVoiceModelFile,
  type VoiceModelCatalogEntry,
  type VoiceModelDownloadProgressEvent,
  type VoiceModelInstallState,
  type VoiceModelTestTranscribeResult,
} from "@/lib/api/voiceModels";
import {
  consumeVoiceModelSettingsFocusRequest,
  VOICE_MODEL_SETTINGS_SECTION_ID,
} from "@/lib/voiceModelSettingsNavigation";
import {
  getTtsModelsForProvider,
  hasMediaGenerationPreferenceOverride,
  isTtsProvider,
  type MediaGenerationPreference,
} from "@/lib/mediaGeneration";
import { modelSupportsTaskFamily } from "@/lib/model/inferModelCapabilities";
import { cn } from "@/lib/utils";
import {
  findConfiguredProviderBySelection,
  useConfiguredProviders,
} from "@/hooks/useConfiguredProviders";
import { MediaPreferenceSection } from "../shared/MediaPreferenceSection";
import { updateMediaPreference } from "../shared/mediaPreferencePersistence";
import { SettingModelSelectorField } from "../shared/SettingModelSelectorField";
import { useAutoDismissMessage } from "./useAutoDismissMessage";

const DEFAULT_MEDIA_PREFERENCE: MediaGenerationPreference = {
  allowFallback: true,
};

type PillTone = "neutral" | "success" | "warning";
type VoiceModelAction = "download" | "delete" | "default" | "test";
type VoiceSettingsTranslate = TFunction<"settings", undefined>;

function normalizeOptionalText(value?: string | null): string | undefined {
  const normalized = value?.trim();
  return normalized ? normalized : undefined;
}

function ensureValidVoiceInstructionSelection(
  config: VoiceInputConfig,
): VoiceInputConfig {
  if (config.instructions.length === 0) {
    return config;
  }

  const hasInstruction = (id?: string | null) =>
    Boolean(
      id && config.instructions.some((instruction) => instruction.id === id),
    );

  const fallbackDefaultInstructionId = hasInstruction("default")
    ? "default"
    : (config.instructions[0]?.id ?? config.processor.default_instruction_id);

  const nextDefaultInstructionId = hasInstruction(
    config.processor.default_instruction_id,
  )
    ? config.processor.default_instruction_id
    : fallbackDefaultInstructionId;

  const fallbackTranslateInstructionId = hasInstruction("translate_en")
    ? "translate_en"
    : nextDefaultInstructionId;

  const nextTranslateInstructionId = hasInstruction(
    config.translate_instruction_id,
  )
    ? config.translate_instruction_id
    : fallbackTranslateInstructionId;

  if (
    nextDefaultInstructionId === config.processor.default_instruction_id &&
    nextTranslateInstructionId === config.translate_instruction_id
  ) {
    return config;
  }

  return {
    ...config,
    processor: {
      ...config.processor,
      default_instruction_id: nextDefaultInstructionId,
    },
    translate_instruction_id: nextTranslateInstructionId,
  };
}

function StatusPill({
  tone,
  children,
}: {
  tone: PillTone;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-medium",
        tone === "success" &&
          "border-emerald-200 bg-emerald-50 text-emerald-700",
        tone === "warning" && "border-amber-200 bg-amber-50 text-amber-700",
        tone === "neutral" && "border-slate-200 bg-slate-50 text-slate-600",
      )}
    >
      {children}
    </span>
  );
}

function SettingCard({
  title,
  description,
  icon: Icon,
  children,
  sectionId,
  tipAriaLabel,
}: {
  title: string;
  description: string;
  icon: LucideIcon;
  children: ReactNode;
  sectionId?: string;
  tipAriaLabel: string;
}) {
  return (
    <section
      id={sectionId}
      tabIndex={sectionId ? -1 : undefined}
      className={cn(
        "overflow-visible rounded-[24px] border border-slate-200/80 bg-white shadow-sm shadow-slate-950/5",
        sectionId && "scroll-mt-6 outline-none",
      )}
    >
      <div className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-sky-600" />
          <h3 className="text-base font-semibold tracking-tight text-slate-900">
            {title}
          </h3>
          <WorkbenchInfoTip
            ariaLabel={tipAriaLabel}
            content={description}
            tone="slate"
          />
        </div>
      </div>
      <div className="divide-y divide-slate-200/80 border-t border-slate-200/80">
        {children}
      </div>
    </section>
  );
}

function SettingRow({
  label,
  description,
  children,
}: {
  label: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-3 px-5 py-4 md:grid-cols-[220px_minmax(0,1fr)] md:items-start">
      <div className="space-y-1.5">
        <Label className="text-sm font-medium text-slate-800">{label}</Label>
        <p className="text-xs leading-5 text-slate-500">{description}</p>
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="max-w-[820px] space-y-4">
      <div className="h-[220px] animate-pulse rounded-[24px] border border-slate-200/80 bg-slate-50" />
      <div className="h-[260px] animate-pulse rounded-[24px] border border-slate-200/80 bg-white" />
      <div className="h-[200px] animate-pulse rounded-[24px] border border-slate-200/80 bg-white" />
    </div>
  );
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 MB";
  }

  const mb = value / 1024 / 1024;
  if (mb < 1024) {
    return `${mb.toFixed(0)} MB`;
  }
  return `${(mb / 1024).toFixed(1)} GB`;
}

function getVoiceModelDisplayName(entry: VoiceModelCatalogEntry): string {
  return entry.name.replace(/\s+INT8$/i, "").trim() || entry.name;
}

function getVoiceModelInstallStatusText(
  t: VoiceSettingsTranslate,
  entry: VoiceModelCatalogEntry,
  state: VoiceModelInstallState | null,
  action: VoiceModelAction | null,
  progress?: VoiceModelDownloadProgressEvent | null,
): string {
  if (action === "download") {
    return (
      progress?.message || t("settings.voice.model.status.prepareDownload")
    );
  }

  const modelSize = entry.size_bytes
    ? t("settings.voice.model.status.approxSize", {
        size: formatBytes(entry.size_bytes),
      })
    : t("settings.voice.model.status.sizePending");

  if (state?.installed) {
    return t("settings.voice.model.status.installed", {
      size: modelSize,
    });
  }

  return t("settings.voice.model.status.notInstalled", {
    size: modelSize,
  });
}

function getVoiceModelDownloadPercent(
  progress?: VoiceModelDownloadProgressEvent | null,
): number {
  if (!progress || !Number.isFinite(progress.overall_progress)) {
    return 0;
  }

  return Math.min(
    100,
    Math.max(0, Math.round(progress.overall_progress * 100)),
  );
}

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  if (typeof error === "string" && error.trim()) {
    return error;
  }
  return fallback;
}

export function VoiceSettings() {
  const { t } = useTranslation("settings");
  const [config, setConfig] = useState<Config | null>(null);
  const [voiceConfig, setVoiceConfig] = useState<VoiceInputConfig | null>(null);
  const [voiceModelCatalog, setVoiceModelCatalog] = useState<
    VoiceModelCatalogEntry[]
  >([]);
  const [voiceModelState, setVoiceModelState] =
    useState<VoiceModelInstallState | null>(null);
  const [voiceModelAction, setVoiceModelAction] =
    useState<VoiceModelAction | null>(null);
  const [voiceModelDownloadProgress, setVoiceModelDownloadProgress] =
    useState<VoiceModelDownloadProgressEvent | null>(null);
  const [voiceModelTestPath, setVoiceModelTestPath] = useState("");
  const [voiceModelTestResult, setVoiceModelTestResult] =
    useState<VoiceModelTestTranscribeResult | null>(null);
  const [voiceModelTestError, setVoiceModelTestError] = useState<string | null>(
    null,
  );
  const { message, showMessage } = useAutoDismissMessage();
  const [loading, setLoading] = useState(true);
  const [globalVoicePreference, setGlobalVoicePreference] =
    useState<MediaGenerationPreference>(DEFAULT_MEDIA_PREFERENCE);
  const { providers, loading: providersLoading } = useConfiguredProviders();

  const loadVoiceSettings = useCallback(async () => {
    setLoading(true);

    try {
      const [nextConfig, nextVoiceConfig, nextVoiceModelCatalog] =
        await Promise.all([
          getConfig(),
          getVoiceInputConfig(),
          listVoiceModelCatalog().catch(() => []),
        ]);
      const primaryVoiceModel = nextVoiceModelCatalog[0] ?? null;
      const nextVoiceModelState = primaryVoiceModel
        ? await getVoiceModelInstallState(primaryVoiceModel.id).catch(
            () => null,
          )
        : null;

      const normalizedVoiceConfig =
        ensureValidVoiceInstructionSelection(nextVoiceConfig);

      setConfig(nextConfig);
      setVoiceConfig(normalizedVoiceConfig);
      setVoiceModelCatalog(nextVoiceModelCatalog);
      setVoiceModelState(nextVoiceModelState);
      setGlobalVoicePreference(
        nextConfig.workspace_preferences?.media_defaults?.voice ??
          DEFAULT_MEDIA_PREFERENCE,
      );
    } catch (error) {
      console.error("加载语音设置失败:", error);
      showMessage("error", t("settings.voice.message.loadFailed"));
    } finally {
      setLoading(false);
    }
  }, [showMessage, t]);

  useEffect(() => {
    void loadVoiceSettings();
  }, [loadVoiceSettings]);

  useEffect(() => {
    if (loading) {
      return;
    }

    const focusRequest = consumeVoiceModelSettingsFocusRequest();
    if (!focusRequest) {
      return;
    }

    const focusVoiceModelSection = () => {
      const target = document.getElementById(VOICE_MODEL_SETTINGS_SECTION_ID);
      if (!target) {
        return false;
      }
      target.scrollIntoView?.({ block: "start", behavior: "smooth" });
      target.focus?.({ preventScroll: true });
      return true;
    };

    if (focusVoiceModelSection()) {
      return;
    }

    let attempts = 0;
    const retryFocusVoiceModelSection = () => {
      if (focusVoiceModelSection()) {
        return;
      }
      if (attempts < 5) {
        attempts += 1;
        window.setTimeout(retryFocusVoiceModelSection, 0);
      }
    };

    window.requestAnimationFrame(retryFocusVoiceModelSection);
  }, [loading]);

  const persistVoiceConfig = useCallback(
    async (updater: (current: VoiceInputConfig) => VoiceInputConfig) => {
      if (!voiceConfig) {
        return;
      }

      try {
        const nextVoiceConfig = ensureValidVoiceInstructionSelection(
          updater(voiceConfig),
        );
        await saveVoiceInputConfig(nextVoiceConfig);
        setVoiceConfig(nextVoiceConfig);
        showMessage("success", t("settings.voice.message.saved"));
      } catch (error) {
        console.error("保存语音设置失败:", error);
        showMessage("error", t("settings.voice.message.saveFailed"));
      }
    },
    [showMessage, t, voiceConfig],
  );

  const persistGlobalVoicePreference = useCallback(
    async (
      updater: (
        current: MediaGenerationPreference,
      ) => MediaGenerationPreference,
    ) => {
      try {
        const { config: nextConfig, preference: nextPreference } =
          await updateMediaPreference("voice", updater);
        setConfig(nextConfig);
        setGlobalVoicePreference(nextPreference);
        showMessage(
          "success",
          t("settings.voice.message.mediaPreferenceSaved"),
        );
      } catch (error) {
        console.error("保存语音生成偏好失败:", error);
        showMessage(
          "error",
          t("settings.voice.message.mediaPreferenceSaveFailed"),
        );
      }
    },
    [showMessage, t],
  );

  const voiceProviders = useMemo(
    () =>
      providers.filter((provider) =>
        isTtsProvider(provider.providerId ?? provider.key, provider.type),
      ),
    [providers],
  );

  const polishProvider = useMemo(
    () =>
      voiceConfig
        ? findConfiguredProviderBySelection(
            providers,
            voiceConfig.processor.polish_provider,
          )
        : null,
    [providers, voiceConfig],
  );

  const polishProviderWarning = voiceConfig?.processor.polish_provider
    ? !polishProvider
      ? t("settings.voice.processing.warning.providerUnavailable", {
          provider: voiceConfig.processor.polish_provider,
        })
      : undefined
    : undefined;

  const polishModelWarning =
    voiceConfig?.processor.polish_model &&
    polishProvider?.models?.length &&
    !polishProvider.models.includes(voiceConfig.processor.polish_model)
      ? t("settings.voice.processing.warning.modelUnavailable", {
          provider: polishProvider.label,
          model: voiceConfig.processor.polish_model,
        })
      : undefined;

  const primaryVoiceModel = voiceModelCatalog[0] ?? null;
  const isVoiceModelDefault = Boolean(voiceModelState?.default_credential_id);
  const voiceModelDownloadReady = Boolean(
    primaryVoiceModel?.download_url?.trim() &&
    primaryVoiceModel?.vad_download_url?.trim(),
  );
  const voiceModelDownloadPercent = getVoiceModelDownloadPercent(
    voiceModelDownloadProgress,
  );

  const voiceInstructions = voiceConfig?.instructions ?? [];
  const defaultInstructionId =
    voiceConfig?.processor.default_instruction_id ?? "";
  const translateInstructionId = voiceConfig?.translate_instruction_id ?? "";

  const defaultInstructionLabel =
    voiceInstructions.find(
      (instruction) => instruction.id === defaultInstructionId,
    )?.name ?? t("settings.voice.processing.instruction.defaultPlaceholder");

  const translateInstructionLabel =
    voiceInstructions.find(
      (instruction) => instruction.id === translateInstructionId,
    )?.name ?? t("settings.voice.processing.instruction.translatePlaceholder");

  const providerHint = providersLoading
    ? t("settings.voice.media.providerHint.loading")
    : voiceProviders.length === 0
      ? t("settings.voice.media.providerHint.empty")
      : t("settings.voice.media.providerHint.ready");

  const llmModelHint = providersLoading
    ? t("settings.voice.processing.modelHint.loading")
    : providers.length === 0
      ? t("settings.voice.processing.modelHint.empty")
      : t("settings.voice.processing.modelHint.ready");

  const handlePolishEnabledChange = (enabled: boolean) => {
    void persistVoiceConfig((current) => ({
      ...current,
      processor: {
        ...current.processor,
        polish_enabled: enabled,
      },
    }));
  };

  const handlePolishProviderChange = (value: string) => {
    const nextProviderId = normalizeOptionalText(value);

    void persistVoiceConfig((current) => ({
      ...current,
      processor: {
        ...current.processor,
        polish_provider: nextProviderId,
        polish_model:
          nextProviderId === current.processor.polish_provider
            ? current.processor.polish_model
            : undefined,
      },
    }));
  };

  const handlePolishModelChange = (value: string) => {
    void persistVoiceConfig((current) => ({
      ...current,
      processor: {
        ...current.processor,
        polish_model: normalizeOptionalText(value),
      },
    }));
  };

  const handleDefaultInstructionChange = (instructionId: string) => {
    void persistVoiceConfig((current) => ({
      ...current,
      processor: {
        ...current.processor,
        default_instruction_id: instructionId,
      },
    }));
  };

  const handleTranslateInstructionChange = (
    event: ChangeEvent<HTMLSelectElement>,
  ) => {
    const instructionId = event.target.value;
    void persistVoiceConfig((current) => ({
      ...current,
      translate_instruction_id: instructionId,
    }));
  };

  const handleInstructionSnapshot = (instructions: VoiceInstruction[]) => {
    setVoiceConfig((current) => {
      if (!current) {
        return current;
      }

      return ensureValidVoiceInstructionSelection({
        ...current,
        instructions,
      });
    });
  };

  const handleDefaultInstructionSelect = (
    event: ChangeEvent<HTMLSelectElement>,
  ) => {
    const instructionId = event.target.value;
    handleDefaultInstructionChange(instructionId);
  };

  const handleMediaProviderChange = (value: string) => {
    const preferredProviderId = normalizeOptionalText(value);
    const nextProvider = findConfiguredProviderBySelection(
      voiceProviders,
      preferredProviderId,
    );
    const nextModels = nextProvider
      ? getTtsModelsForProvider(nextProvider.models)
      : [];
    void persistGlobalVoicePreference((current) => ({
      preferredProviderId,
      preferredModelId: preferredProviderId
        ? nextModels.includes(current.preferredModelId || "")
          ? current.preferredModelId
          : undefined
        : undefined,
      allowFallback: current.allowFallback ?? true,
    }));
  };

  const handleMediaModelChange = (value: string) => {
    void persistGlobalVoicePreference((current) => ({
      ...current,
      preferredModelId: normalizeOptionalText(value),
      allowFallback: current.allowFallback ?? true,
    }));
  };

  const handleMediaProviderAndModelChange = (
    providerValue: string,
    modelValue: string,
  ) => {
    void persistGlobalVoicePreference((current) => ({
      preferredProviderId: normalizeOptionalText(providerValue),
      preferredModelId:
        normalizeOptionalText(providerValue) &&
        normalizeOptionalText(modelValue)
          ? normalizeOptionalText(modelValue)
          : undefined,
      allowFallback: current.allowFallback ?? true,
    }));
  };

  const handleFallbackChange = (value: boolean) => {
    void persistGlobalVoicePreference((current) => ({
      ...current,
      allowFallback: value,
    }));
  };

  const handleResetPreference = () => {
    void persistGlobalVoicePreference(() => DEFAULT_MEDIA_PREFERENCE);
  };

  const handleDownloadVoiceModel = async () => {
    if (!primaryVoiceModel) {
      return;
    }

    setVoiceModelAction("download");
    setVoiceModelDownloadProgress({
      model_id: primaryVoiceModel.id,
      phase: "preparing",
      downloaded_bytes: 0,
      total_bytes: primaryVoiceModel.size_bytes || null,
      overall_progress: 0,
      message: t("settings.voice.model.status.prepareDownload"),
    });
    let unlistenProgress: (() => void) | null = null;
    try {
      unlistenProgress = await listenVoiceModelDownloadProgress((event) => {
        if (event.model_id !== primaryVoiceModel.id) {
          return;
        }
        setVoiceModelDownloadProgress(event);
      }).catch((error) => {
        console.warn("监听语音模型下载进度失败:", error);
        return null;
      });
      const result = await downloadVoiceModel(primaryVoiceModel.id);
      setVoiceModelState(result.state);
      setVoiceModelTestError(null);
      setVoiceModelTestResult(null);
      showMessage(
        "success",
        t("settings.voice.model.message.downloaded", {
          model: getVoiceModelDisplayName(primaryVoiceModel),
        }),
      );
    } catch (error) {
      console.error("下载 SenseVoice Small 模型失败:", error);
      showMessage(
        "error",
        t("settings.voice.model.message.downloadFailed", {
          model: getVoiceModelDisplayName(primaryVoiceModel),
        }),
      );
    } finally {
      unlistenProgress?.();
      setVoiceModelAction(null);
      setVoiceModelDownloadProgress(null);
    }
  };

  const handleDeleteVoiceModel = async () => {
    if (!primaryVoiceModel) {
      return;
    }

    setVoiceModelAction("delete");
    try {
      const state = await deleteVoiceModel(primaryVoiceModel.id);
      setVoiceModelState(state);
      setVoiceModelTestError(null);
      setVoiceModelTestResult(null);
      await loadVoiceSettings();
      showMessage(
        "success",
        t("settings.voice.model.message.deleted", {
          model: getVoiceModelDisplayName(primaryVoiceModel),
        }),
      );
    } catch (error) {
      console.error("删除 SenseVoice Small 模型失败:", error);
      showMessage(
        "error",
        t("settings.voice.model.message.deleteFailed", {
          model: getVoiceModelDisplayName(primaryVoiceModel),
        }),
      );
    } finally {
      setVoiceModelAction(null);
    }
  };

  const handleSetDefaultVoiceModel = async () => {
    if (!primaryVoiceModel) {
      return;
    }

    setVoiceModelAction("default");
    try {
      await setDefaultVoiceModel(primaryVoiceModel.id);
      await loadVoiceSettings();
      showMessage(
        "success",
        t("settings.voice.model.message.defaultSet", {
          model: getVoiceModelDisplayName(primaryVoiceModel),
        }),
      );
    } catch (error) {
      console.error("设置 SenseVoice Small 默认模型失败:", error);
      showMessage(
        "error",
        t("settings.voice.model.message.defaultSetFailed", {
          model: getVoiceModelDisplayName(primaryVoiceModel),
        }),
      );
    } finally {
      setVoiceModelAction(null);
    }
  };

  const handleSelectVoiceModelTestFile = async () => {
    if (!voiceModelState?.installed || voiceModelAction !== null) {
      return;
    }

    try {
      const selected = await openDialog({
        title: t("settings.voice.model.test.dialogTitle"),
        multiple: false,
        directory: false,
        filters: [
          {
            name: t("settings.voice.model.test.dialogFilter"),
            extensions: ["wav"],
          },
        ],
      });
      const filePath = Array.isArray(selected) ? selected[0] : selected;
      if (!filePath) {
        return;
      }
      setVoiceModelTestPath(filePath);
      setVoiceModelTestError(null);
      setVoiceModelTestResult(null);
    } catch (error) {
      console.error("选择 SenseVoice Small 测试文件失败:", error);
      const errorMessage = getErrorMessage(
        error,
        t("settings.voice.model.test.selectFailed"),
      );
      setVoiceModelTestError(errorMessage);
      showMessage("error", t("settings.voice.model.test.selectFailed"));
    }
  };

  const handleTestVoiceModel = async () => {
    if (!primaryVoiceModel || !voiceModelState?.installed) {
      return;
    }

    const filePath = voiceModelTestPath.trim();
    if (!filePath) {
      setVoiceModelTestError(t("settings.voice.model.test.pathRequired"));
      return;
    }

    setVoiceModelAction("test");
    setVoiceModelTestError(null);
    setVoiceModelTestResult(null);
    try {
      const result = await testTranscribeVoiceModelFile(
        primaryVoiceModel.id,
        filePath,
      );
      setVoiceModelTestResult(result);
      showMessage(
        "success",
        t("settings.voice.model.test.completed", {
          model: getVoiceModelDisplayName(primaryVoiceModel),
        }),
      );
    } catch (error) {
      console.error("SenseVoice Small 测试转写失败:", error);
      const errorMessage = getErrorMessage(
        error,
        t("settings.voice.model.test.failed"),
      );
      setVoiceModelTestError(errorMessage);
      showMessage("error", t("settings.voice.model.test.failed"));
    } finally {
      setVoiceModelAction(null);
    }
  };

  if (loading) {
    return <LoadingSkeleton />;
  }

  return (
    <div className="max-w-[820px] space-y-4">
      <SettingCard
        title={t("settings.voice.model.title")}
        description={t("settings.voice.model.description")}
        icon={HardDrive}
        sectionId={VOICE_MODEL_SETTINGS_SECTION_ID}
        tipAriaLabel={t("settings.voice.card.tipAria", {
          title: t("settings.voice.model.title"),
        })}
      >
        <SettingRow
          label={t("settings.voice.model.senseVoice.label")}
          description={t("settings.voice.model.senseVoice.description")}
        >
          {primaryVoiceModel ? (
            <div className="space-y-4 rounded-[22px] bg-[#f7fbf7] px-4 py-4">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div className="flex min-w-0 gap-3">
                  <div
                    className={cn(
                      "flex h-10 w-10 shrink-0 items-center justify-center rounded-[14px] border",
                      voiceModelState?.installed
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-slate-200 bg-white text-slate-500",
                    )}
                  >
                    <HardDrive className="h-4 w-4" />
                  </div>
                  <div className="min-w-0 space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-sm font-semibold text-slate-900">
                        {getVoiceModelDisplayName(primaryVoiceModel)}
                      </p>
                      <span className="inline-flex items-center rounded-full border border-emerald-200 bg-white px-2.5 py-1 text-[11px] font-semibold text-emerald-700">
                        {t("settings.voice.model.badge.local")}
                      </span>
                      {isVoiceModelDefault ? (
                        <StatusPill tone="success">
                          {t("settings.voice.model.badge.defaultAsr")}
                        </StatusPill>
                      ) : null}
                    </div>
                    <p className="text-xs leading-5 text-slate-500">
                      {primaryVoiceModel.description ||
                        t(
                          "settings.voice.model.senseVoice.fallbackDescription",
                        )}
                    </p>
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs leading-5 text-slate-500">
                      <span>
                        {getVoiceModelInstallStatusText(
                          t,
                          primaryVoiceModel,
                          voiceModelState,
                          voiceModelAction,
                          voiceModelDownloadProgress,
                        )}
                      </span>
                      <span className="text-slate-300">·</span>
                      <span>{primaryVoiceModel.runtime}</span>
                      {primaryVoiceModel.version ? (
                        <>
                          <span className="text-slate-300">·</span>
                          <span>{primaryVoiceModel.version}</span>
                        </>
                      ) : null}
                      {primaryVoiceModel.languages.length ? (
                        <>
                          <span className="text-slate-300">·</span>
                          <span>{primaryVoiceModel.languages.join(" / ")}</span>
                        </>
                      ) : null}
                    </div>
                    {voiceModelAction === "download" ? (
                      <div className="space-y-2">
                        <div className="h-1.5 overflow-hidden rounded-full bg-emerald-100">
                          <div
                            aria-label={t("settings.voice.model.progressAria")}
                            aria-valuemax={100}
                            aria-valuemin={0}
                            aria-valuenow={voiceModelDownloadPercent}
                            className="h-full rounded-full bg-emerald-700 transition-[width] duration-200"
                            role="progressbar"
                            style={{ width: `${voiceModelDownloadPercent}%` }}
                          />
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] leading-4 text-slate-500">
                          <span>{voiceModelDownloadPercent}%</span>
                        </div>
                      </div>
                    ) : null}
                    <p className="break-all text-xs leading-5 text-slate-500">
                      {voiceModelState?.installed
                        ? voiceModelState.install_dir
                        : t("settings.voice.model.install.onDemand")}
                    </p>
                    {voiceModelState?.installed ? (
                      <p className="text-xs leading-5 text-slate-500">
                        {t("settings.voice.model.install.installedSize", {
                          size: formatBytes(voiceModelState.installed_bytes),
                        })}
                      </p>
                    ) : null}
                  </div>
                </div>

                <div className="flex shrink-0 flex-wrap gap-2 lg:justify-end">
                  {voiceModelState?.installed ? (
                    <>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => void handleSetDefaultVoiceModel()}
                        disabled={
                          voiceModelAction !== null || isVoiceModelDefault
                        }
                      >
                        {t("settings.voice.model.action.setDefault")}
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => void handleDeleteVoiceModel()}
                        disabled={voiceModelAction !== null}
                      >
                        <Trash2 className="mr-1 h-4 w-4" />
                        {t("settings.voice.model.action.delete")}
                      </Button>
                    </>
                  ) : (
                    <Button
                      type="button"
                      size="sm"
                      onClick={() => void handleDownloadVoiceModel()}
                      disabled={
                        voiceModelAction !== null || !voiceModelDownloadReady
                      }
                      className="!h-10 !rounded-[10px] !border-cyan-500 !bg-gradient-to-r !from-sky-500 !to-emerald-500 !px-4 text-base font-semibold text-white shadow-md shadow-emerald-900/15 hover:!opacity-95 disabled:!border-slate-200 disabled:!bg-none disabled:!bg-slate-100 disabled:text-slate-400 disabled:shadow-none"
                    >
                      {voiceModelAction === "download" ? (
                        <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                      ) : (
                        <Download className="mr-1 h-4 w-4" />
                      )}
                      {voiceModelAction === "download"
                        ? t("settings.voice.model.action.downloading")
                        : t("settings.voice.model.action.download")}
                    </Button>
                  )}
                </div>
              </div>
              {!voiceModelDownloadReady && !voiceModelState?.installed ? (
                <p className="text-xs leading-5 text-amber-700">
                  {t("settings.voice.model.assetBaseMissing")}
                </p>
              ) : null}

              <div className="space-y-3 border-t border-slate-200/80 pt-4">
                <div className="space-y-1">
                  <p className="text-sm font-medium text-slate-800">
                    {t("settings.voice.model.test.title")}
                  </p>
                  <p className="text-xs leading-5 text-slate-500">
                    {t("settings.voice.model.test.description")}
                  </p>
                </div>
                <div className="flex flex-col gap-2 sm:flex-row">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => void handleSelectVoiceModelTestFile()}
                    disabled={
                      !voiceModelState?.installed || voiceModelAction !== null
                    }
                  >
                    <FolderOpen className="mr-1 h-4 w-4" />
                    {t("settings.voice.model.test.selectWav")}
                  </Button>
                  <input
                    aria-label={t("settings.voice.model.test.pathAria")}
                    className="h-10 min-w-0 flex-1 rounded-[14px] border border-slate-200 bg-white px-3 text-sm text-slate-700 shadow-sm outline-none transition placeholder:text-slate-400 focus:border-sky-400"
                    value={voiceModelTestPath}
                    onChange={(event) =>
                      setVoiceModelTestPath(event.target.value)
                    }
                    placeholder={t("settings.voice.model.test.pathPlaceholder")}
                    disabled={
                      !voiceModelState?.installed || voiceModelAction !== null
                    }
                  />
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => void handleTestVoiceModel()}
                    disabled={
                      !voiceModelState?.installed ||
                      voiceModelAction !== null ||
                      !voiceModelTestPath.trim()
                    }
                  >
                    {voiceModelAction === "test"
                      ? t("settings.voice.model.test.testing")
                      : t("settings.voice.model.test.action")}
                  </Button>
                </div>
                {!voiceModelState?.installed ? (
                  <p className="text-xs leading-5 text-amber-700">
                    {t("settings.voice.model.test.needsInstall")}
                  </p>
                ) : null}
                {voiceModelTestError ? (
                  <div className="rounded-[16px] border border-rose-200 bg-rose-50 px-3 py-2 text-xs leading-5 text-rose-700">
                    {voiceModelTestError}
                  </div>
                ) : null}
                {voiceModelTestResult ? (
                  <div className="space-y-2 rounded-[16px] border border-emerald-200 bg-emerald-50 px-3 py-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <StatusPill tone="success">
                        {t("settings.voice.model.test.completedBadge")}
                      </StatusPill>
                      <span className="text-xs text-emerald-700">
                        {t("settings.voice.model.test.resultMeta", {
                          duration: t(
                            "settings.voice.model.test.durationSeconds",
                            {
                              seconds:
                                voiceModelTestResult.duration_secs.toFixed(2),
                            },
                          ),
                          language:
                            voiceModelTestResult.language ||
                            t("settings.voice.model.test.languageAuto"),
                          sampleRate: voiceModelTestResult.sample_rate,
                        })}
                      </span>
                    </div>
                    <p className="whitespace-pre-wrap text-sm leading-6 text-slate-800">
                      {voiceModelTestResult.text ||
                        t("settings.voice.model.test.emptyText")}
                    </p>
                  </div>
                ) : null}
              </div>
            </div>
          ) : (
            <div className="rounded-[18px] border border-slate-200/80 bg-slate-50/80 px-4 py-3 text-sm text-slate-500">
              {t("settings.voice.model.emptyCatalog")}
            </div>
          )}
        </SettingRow>
      </SettingCard>

      <SettingCard
        title={t("settings.voice.processing.title")}
        description={t("settings.voice.processing.description")}
        icon={Wand2}
        tipAriaLabel={t("settings.voice.card.tipAria", {
          title: t("settings.voice.processing.title"),
        })}
      >
        <SettingRow
          label={t("settings.voice.processing.polishEnabled.label")}
          description={t("settings.voice.processing.polishEnabled.description")}
        >
          <div className="flex items-center justify-end">
            <Switch
              checked={voiceConfig?.processor.polish_enabled ?? true}
              onCheckedChange={handlePolishEnabledChange}
              disabled={!voiceConfig}
              aria-label={t(
                "settings.voice.processing.polishEnabled.toggleAria",
              )}
            />
          </div>
        </SettingRow>

        <SettingModelSelectorField
          label={t("settings.voice.processing.model.label")}
          description={llmModelHint}
          warningText={polishProviderWarning ?? polishModelWarning}
          providerType={voiceConfig?.processor.polish_provider ?? ""}
          setProviderType={handlePolishProviderChange}
          model={voiceConfig?.processor.polish_model ?? ""}
          setModel={handlePolishModelChange}
          providerFilter={() => true}
          modelFilter={(model) =>
            modelSupportsTaskFamily(model, "chat") ||
            modelSupportsTaskFamily(model, "reasoning")
          }
          emptyStateTitle={t("settings.voice.processing.model.emptyTitle")}
          emptyStateDescription={llmModelHint}
          disabled={!voiceConfig}
        />

        <SettingRow
          label={t("settings.voice.processing.defaultInstruction.label")}
          description={t(
            "settings.voice.processing.defaultInstruction.description",
          )}
        >
          <select
            aria-label={t("settings.voice.processing.defaultInstruction.aria")}
            className="h-11 w-full rounded-[16px] border border-slate-200 bg-white px-3 text-sm text-slate-700 shadow-sm outline-none transition focus:border-sky-400"
            value={defaultInstructionId}
            onChange={handleDefaultInstructionSelect}
            disabled={!voiceConfig || voiceInstructions.length === 0}
          >
            <option value="" disabled>
              {defaultInstructionLabel}
            </option>
            {voiceInstructions.map((instruction) => (
              <option key={instruction.id} value={instruction.id}>
                {instruction.name}
              </option>
            ))}
          </select>
        </SettingRow>

        <SettingRow
          label={t("settings.voice.processing.translateInstruction.label")}
          description={t(
            "settings.voice.processing.translateInstruction.description",
          )}
        >
          <select
            aria-label={t(
              "settings.voice.processing.translateInstruction.aria",
            )}
            className="h-11 w-full rounded-[16px] border border-slate-200 bg-white px-3 text-sm text-slate-700 shadow-sm outline-none transition focus:border-sky-400"
            value={translateInstructionId}
            onChange={handleTranslateInstructionChange}
            disabled={!voiceConfig || voiceInstructions.length === 0}
          >
            <option value="" disabled>
              {translateInstructionLabel}
            </option>
            {voiceInstructions.map((instruction) => (
              <option key={instruction.id} value={instruction.id}>
                {instruction.name}
              </option>
            ))}
          </select>
        </SettingRow>

        <div className="px-5 py-4">
          <InstructionEditor
            defaultInstructionId={defaultInstructionId}
            onDefaultChange={handleDefaultInstructionChange}
            onInstructionsChange={handleInstructionSnapshot}
            disabled={!voiceConfig}
          />
        </div>
      </SettingCard>

      <MediaPreferenceSection
        title={t("settings.voice.media.title")}
        description={t("settings.voice.media.description")}
        selectorLabel={t("settings.voice.media.selector.label")}
        selectorDescription={t("settings.voice.media.selector.description")}
        providerType={globalVoicePreference.preferredProviderId ?? ""}
        setProviderType={handleMediaProviderChange}
        model={globalVoicePreference.preferredModelId ?? ""}
        setModel={handleMediaModelChange}
        setProviderAndModel={handleMediaProviderAndModelChange}
        providerFilter={(provider) =>
          isTtsProvider(provider.providerId ?? provider.key, provider.type)
        }
        modelFilter={(model, provider) =>
          getTtsModelsForProvider(provider.models).includes(model.id)
        }
        allowFallback={globalVoicePreference.allowFallback ?? true}
        onAllowFallbackChange={handleFallbackChange}
        fallbackTitle={t("settings.voice.media.fallback.title")}
        fallbackDescription={t("settings.voice.media.fallback.description")}
        emptyStateTitle={t("settings.voice.media.emptyTitle")}
        emptyStateDescription={providerHint}
        disabled={!config}
        onReset={handleResetPreference}
        resetLabel={t("settings.voice.media.action.reset")}
        resetDisabled={
          !hasMediaGenerationPreferenceOverride(globalVoicePreference)
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
