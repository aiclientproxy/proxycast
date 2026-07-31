import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import {
  AlertCircle,
  ArrowRight,
  Bot,
  Check,
  ChevronDown,
  Settings2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { ProviderIcon } from "@/icons/providers";
import {
  type ConfiguredProvider,
  findConfiguredProviderBySelection,
  useConfiguredProviders,
} from "@/hooks/useConfiguredProviders";
import { useProviderModels } from "@/hooks/useProviderModels";
import { filterModelsByTheme } from "@/components/agent/chat/utils/modelThemePolicy";
import { getProviderModelCompatibilityIssue } from "@/components/agent/chat/utils/providerModelCompatibility";
import { getProviderLabel } from "@/lib/constants/providerMappings";
import {
  getProviderPromptCacheMode,
  resolvePromptCacheSupportNotice,
} from "@/lib/model/providerPromptCacheSupport";
import { buildProviderModelsFromBackendModelIds } from "@/lib/model/providerModelsCatalog";
import { ModelCapabilityBadges } from "@/components/model/ModelCapabilityBadges";
import { resolveOemCloudRuntimeContext } from "@/lib/api/oemCloudRuntime";
import { resolveOemLimeHubProviderName } from "@/lib/oemLimeHubProvider";
import { resolveProviderModelLoadOptions } from "@/lib/model/providerModelLoadOptions";
import {
  resolveModelReasoningEffortForModelSwitch,
  type ModelReasoningEffortPreset,
} from "@/lib/model/modelReasoningPolicy";
import type {
  EnhancedModelMetadata,
  ModelReasoningEffortLevel,
} from "@/lib/types/modelRegistry";

const compactTriggerClassName =
  "h-8 min-w-[104px] max-w-[168px] justify-start gap-1.5 rounded-full border-slate-200/80 bg-white/92 px-2.5 text-slate-600 shadow-none transition-colors hover:border-slate-300 hover:bg-white hover:text-slate-800";

const defaultTriggerClassName =
  "h-9 w-full min-w-0 justify-start gap-2 rounded-full border-slate-200/80 bg-white/92 px-3 font-normal text-slate-700 shadow-none transition-colors hover:border-slate-300 hover:bg-white";

const itemClassName =
  "flex w-full items-center justify-between rounded-xl border border-transparent px-2.5 py-2 text-left text-sm transition-colors";

const BACKGROUND_PRELOAD_IDLE_TIMEOUT_MS = 1_500;
const BACKGROUND_PRELOAD_FALLBACK_DELAY_MS = 180;
const NO_PROVIDER_GUIDE_DISMISSED_STORAGE_KEY =
  "lime_model_selector_no_provider_guide_dismissed_v1";

function resolveProviderSelectionValue(provider: ConfiguredProvider): string {
  return provider.providerId ?? provider.key;
}

function resolveInitialProviderModel(provider: ConfiguredProvider): string {
  return provider.models?.find((modelId) => modelId.trim().length > 0) ?? "";
}

function resolveInitialProviderModelFromOptions(
  provider: ConfiguredProvider,
  modelOptions: Array<{ id: string; compatibilityIssue: unknown }>,
): string {
  const firstAvailableModel = modelOptions.find(
    (item) => !item.compatibilityIssue,
  )?.id;

  return firstAvailableModel ?? resolveInitialProviderModel(provider);
}

function resolveProviderSelectionModel(params: {
  provider: ConfiguredProvider;
  modelFilter?: ModelSelectorProps["modelFilter"];
  getFallbackModels?: ModelSelectorProps["getFallbackModels"];
  isSelected: boolean;
  currentModelOptions: Array<{ id: string; compatibilityIssue: unknown }>;
}): string {
  if (params.isSelected) {
    return resolveInitialProviderModelFromOptions(
      params.provider,
      params.currentModelOptions,
    );
  }

  if (!params.modelFilter) {
    return resolveInitialProviderModel(params.provider);
  }

  const fallbackModels =
    params.getFallbackModels?.(params.provider) ??
    buildProviderModelsFromBackendModelIds(
      params.provider,
      [],
      params.provider.models ?? [],
    );
  const compatibleModels = fallbackModels.filter((model) =>
    params.modelFilter?.(model, params.provider),
  );

  return (
    compatibleModels.find((model) => model.id.trim().length > 0)?.id ??
    resolveInitialProviderModel(params.provider)
  );
}

function hasProviderDeclaredModel(provider: ConfiguredProvider): boolean {
  return Boolean(resolveInitialProviderModel(provider));
}

function isProviderDeclaredModel(
  provider: ConfiguredProvider | null | undefined,
  modelId: string,
): boolean {
  const normalizedModelId = modelId.trim().toLowerCase();
  return Boolean(
    normalizedModelId &&
    provider?.models?.some(
      (declaredModelId) =>
        declaredModelId.trim().toLowerCase() === normalizedModelId,
    ),
  );
}

function resolveReasoningEffortOptions(
  model: EnhancedModelMetadata | null | undefined,
): ModelReasoningEffortPreset[] {
  const policy = model?.reasoning_policy;
  if (!policy?.can_set_reasoning_effort) {
    return [];
  }

  return policy.supported_reasoning_levels;
}

function resolveReasoningEffortForModelSwitch(
  model: EnhancedModelMetadata | null | undefined,
  currentReasoningEffort: ModelReasoningEffortLevel | "",
): ModelReasoningEffortLevel | "" {
  const policy = model?.reasoning_policy;
  if (!policy?.can_set_reasoning_effort) {
    return "";
  }
  return (
    resolveModelReasoningEffortForModelSwitch(policy, currentReasoningEffort) ??
    ""
  );
}

export interface ModelSelectorProps {
  providerType: string;
  setProviderType: (type: string) => void;
  model: string;
  setModel: (model: string) => void;
  setProviderAndModel?: (providerType: string, model: string) => void;
  reasoningEffort?: ModelReasoningEffortLevel | "";
  setReasoningEffort?: (value: ModelReasoningEffortLevel | "") => void;
  activeTheme?: string;
  className?: string;
  compactTrigger?: boolean;
  onManageProviders?: () => void;
  popoverSide?: "top" | "bottom";
  disabled?: boolean;
  backgroundPreload?: "immediate" | "idle" | "disabled";
  allowAutoProvider?: boolean;
  allowAutoModel?: boolean;
  autoProviderLabel?: string;
  autoModelLabel?: string;
  placeholderLabel?: string;
  suppressAutoSelection?: boolean;
  providerFilter?: (provider: ConfiguredProvider) => boolean;
  modelFilter?: (
    model: EnhancedModelMetadata,
    provider: ConfiguredProvider,
  ) => boolean;
  getFallbackModels?: (provider: ConfiguredProvider) => EnhancedModelMetadata[];
  preserveUnknownModelSelection?: boolean;
  emptyStateTitle?: string;
  emptyStateDescription?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  providerType,
  setProviderType,
  model,
  setModel,
  setProviderAndModel,
  reasoningEffort = "",
  setReasoningEffort,
  activeTheme,
  className,
  compactTrigger = false,
  onManageProviders,
  popoverSide = "top",
  disabled = false,
  backgroundPreload = "immediate",
  allowAutoProvider = false,
  allowAutoModel = false,
  autoProviderLabel,
  autoModelLabel,
  placeholderLabel,
  suppressAutoSelection = false,
  providerFilter,
  modelFilter,
  getFallbackModels,
  preserveUnknownModelSelection = false,
  emptyStateTitle,
  emptyStateDescription,
}) => {
  const { t } = useTranslation("common");
  const [open, setOpen] = useState(false);
  const [noProviderGuideDismissed, setNoProviderGuideDismissed] = useState(
    () => {
      if (typeof window === "undefined") {
        return false;
      }
      return (
        window.localStorage.getItem(NO_PROVIDER_GUIDE_DISMISSED_STORAGE_KEY) ===
        "1"
      );
    },
  );
  const [backgroundProviderLoadReady, setBackgroundProviderLoadReady] =
    useState(false);
  const hasInitialized = useRef(false);
  const modelRef = useRef(model);
  modelRef.current = model;
  const commitProviderAndModel = useCallback(
    (nextProviderType: string, nextModel: string) => {
      if (setProviderAndModel) {
        setProviderAndModel(nextProviderType, nextModel);
        return;
      }
      setProviderType(nextProviderType);
      setModel(nextModel);
    },
    [setModel, setProviderAndModel, setProviderType],
  );
  const shouldBackgroundLoadModels =
    backgroundPreload === "immediate" ||
    backgroundProviderLoadReady ||
    !providerType.trim() ||
    !model.trim();
  const shouldLoadProviders =
    open ||
    backgroundPreload === "immediate" ||
    backgroundProviderLoadReady ||
    !providerType.trim() ||
    !model.trim();
  const shouldLoadModels = open || shouldBackgroundLoadModels;

  useEffect(() => {
    if (backgroundPreload !== "idle" || open || backgroundProviderLoadReady) {
      return;
    }

    if (typeof window === "undefined") {
      setBackgroundProviderLoadReady(true);
      return;
    }

    if (typeof window.requestIdleCallback === "function") {
      const idleId = window.requestIdleCallback(
        () => {
          setBackgroundProviderLoadReady(true);
        },
        {
          timeout: BACKGROUND_PRELOAD_IDLE_TIMEOUT_MS,
        },
      );

      return () => {
        if (typeof window.cancelIdleCallback === "function") {
          window.cancelIdleCallback(idleId);
        }
      };
    }

    const timeoutId = window.setTimeout(() => {
      setBackgroundProviderLoadReady(true);
    }, BACKGROUND_PRELOAD_FALLBACK_DELAY_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [backgroundPreload, backgroundProviderLoadReady, open]);

  const { providers: configuredProviders, loading: providersLoading } =
    useConfiguredProviders({ autoLoad: shouldLoadProviders });

  const visibleProviders = useMemo(
    () =>
      providerFilter
        ? configuredProviders.filter((provider) => providerFilter(provider))
        : configuredProviders,
    [configuredProviders, providerFilter],
  );
  const cloudProviders = useMemo(
    () =>
      visibleProviders.filter(
        (provider) =>
          provider.key === "lime-hub" || provider.providerId === "lime-hub",
      ),
    [visibleProviders],
  );
  const localProviders = useMemo(
    () =>
      visibleProviders.filter(
        (provider) =>
          provider.key !== "lime-hub" && provider.providerId !== "lime-hub",
      ),
    [visibleProviders],
  );
  const selectedProvider = useMemo(() => {
    return findConfiguredProviderBySelection(configuredProviders, providerType);
  }, [configuredProviders, providerType]);
  const selectedProviderLoginRequired =
    selectedProvider?.authStatus === "login_required";
  const selectedProviderHasDeclaredModel = selectedProvider
    ? hasProviderDeclaredModel(selectedProvider)
    : false;
  const selectedProviderBlocksModelLoad =
    selectedProviderLoginRequired && !selectedProviderHasDeclaredModel;
  const autoSelectableProviders = useMemo(
    () =>
      visibleProviders.filter(
        (provider) =>
          provider.authStatus !== "login_required" ||
          hasProviderDeclaredModel(provider),
      ),
    [visibleProviders],
  );
  const selectedProviderVisible = useMemo(
    () =>
      selectedProvider
        ? visibleProviders.some(
            (provider) => provider.key === selectedProvider.key,
          )
        : false,
    [selectedProvider, visibleProviders],
  );
  const providerModelLoadOptions = useMemo(
    () =>
      resolveProviderModelLoadOptions({
        providerId: selectedProvider?.providerId,
        providerType: selectedProvider?.type,
        apiHost: selectedProvider?.apiHost,
        hasApiKey: selectedProvider?.hasApiKey,
        hasDeclaredModels: selectedProvider
          ? hasProviderDeclaredModel(selectedProvider)
          : false,
      }),
    [selectedProvider],
  );

  const { models: providerModels, loading: modelsLoading } = useProviderModels(
    selectedProvider,
    {
      returnFullMetadata: true,
      autoLoad: shouldLoadModels && !selectedProviderBlocksModelLoad,
      ...providerModelLoadOptions,
    },
  );

  const filteredResult = useMemo(() => {
    return filterModelsByTheme(activeTheme, providerModels);
  }, [activeTheme, providerModels]);
  const visibleModels = useMemo(() => {
    const baseModels =
      selectedProvider && modelFilter
        ? filteredResult.models.filter((item) =>
            modelFilter(item, selectedProvider),
          )
        : filteredResult.models;

    if (baseModels.length > 0 || !selectedProvider || !getFallbackModels) {
      return baseModels;
    }

    const fallbackModels = getFallbackModels(selectedProvider);
    if (!selectedProvider || !modelFilter) {
      return fallbackModels;
    }

    return fallbackModels.filter((item) => modelFilter(item, selectedProvider));
  }, [filteredResult.models, getFallbackModels, modelFilter, selectedProvider]);

  const modelOptions = useMemo(
    () =>
      visibleModels.flatMap((item) => {
        const compatibilityIssue = getProviderModelCompatibilityIssue({
          providerType,
          configuredProviderType: selectedProvider?.type,
          model: item.id,
          capabilityProvenance: item.capability_provenance,
          providerDeclaredModel: isProviderDeclaredModel(
            selectedProvider,
            item.id,
          ),
          enforceExecutableCapability: true,
        });

        if (
          compatibilityIssue?.code === "runtime_capability_not_authoritative"
        ) {
          return [];
        }

        return [
          {
            id: item.id,
            metadata: item,
            compatibilityIssue,
          },
        ];
      }),
    [providerType, selectedProvider, visibleModels],
  );

  const currentModels = useMemo(
    () =>
      modelOptions
        .filter((item) => !item.compatibilityIssue)
        .map((item) => item.id),
    [modelOptions],
  );
  const selectedPromptCacheNotice = useMemo(
    () =>
      resolvePromptCacheSupportNotice({
        providerType,
        configuredProviderType: selectedProvider?.type,
        configuredApiHost: selectedProvider?.apiHost,
        configuredPromptCacheMode: selectedProvider?.promptCacheMode,
      }),
    [
      providerType,
      selectedProvider?.apiHost,
      selectedProvider?.promptCacheMode,
      selectedProvider?.type,
    ],
  );

  const incompatibleModelCount = useMemo(
    () => modelOptions.filter((item) => item.compatibilityIssue).length,
    [modelOptions],
  );
  const selectedModelOption = useMemo(
    () =>
      modelOptions.find(
        (item) => item.id === model && !item.compatibilityIssue,
      ) ?? null,
    [model, modelOptions],
  );
  const selectedModelKnownOption = useMemo(
    () => modelOptions.find((item) => item.id === model) ?? null,
    [model, modelOptions],
  );
  const selectedReasoningEffortOptions = useMemo(
    () => resolveReasoningEffortOptions(selectedModelOption?.metadata),
    [selectedModelOption?.metadata],
  );
  const selectedReasoningEffortOption =
    selectedReasoningEffortOptions.find(
      (option) => option.value === reasoningEffort,
    ) ?? null;
  const selectedReasoningEffort = selectedReasoningEffortOption?.value ?? "";
  const selectedReasoningEffortLabel = selectedReasoningEffortOption
    ? selectedReasoningEffortOption.label ||
      t(
        `common.modelSelector.reasoning.level.${selectedReasoningEffortOption.value}`,
        { defaultValue: selectedReasoningEffortOption.id },
      )
    : "";

  useEffect(() => {
    if (!setReasoningEffort) {
      return;
    }
    if (!shouldLoadModels) {
      return;
    }

    if (!selectedModelOption || selectedReasoningEffortOptions.length === 0) {
      if (reasoningEffort) {
        setReasoningEffort("");
      }
      return;
    }

    const nextReasoningEffort = resolveReasoningEffortForModelSwitch(
      selectedModelOption.metadata,
      reasoningEffort,
    );
    if (nextReasoningEffort !== reasoningEffort) {
      setReasoningEffort(nextReasoningEffort);
    }
  }, [
    reasoningEffort,
    selectedModelOption,
    selectedReasoningEffortOptions,
    setReasoningEffort,
    shouldLoadModels,
  ]);
  useEffect(() => {
    if (hasInitialized.current) return;
    if (!shouldLoadProviders) return;
    if (providersLoading) return;
    if (visibleProviders.length === 0) return;
    if (allowAutoProvider || suppressAutoSelection) return;

    hasInitialized.current = true;

    if (!providerType.trim()) {
      const nextProvider = autoSelectableProviders[0];
      if (nextProvider) {
        const nextModel = resolveInitialProviderModel(nextProvider);
        commitProviderAndModel(
          nextProvider.providerId ?? nextProvider.key,
          nextModel,
        );
      } else {
        hasInitialized.current = false;
      }
    }
  }, [
    allowAutoProvider,
    autoSelectableProviders,
    commitProviderAndModel,
    providerType,
    providersLoading,
    shouldLoadProviders,
    suppressAutoSelection,
    visibleProviders,
  ]);

  useEffect(() => {
    if (!shouldLoadModels) return;
    if (!selectedProvider) return;
    if (selectedProviderBlocksModelLoad) return;
    if (modelsLoading) return;
    if (allowAutoModel || suppressAutoSelection) return;

    const currentModel = modelRef.current;
    const currentModelKnownButIncompatible = Boolean(
      currentModel &&
      selectedModelKnownOption?.id === currentModel &&
      selectedModelKnownOption.compatibilityIssue,
    );
    const currentModelMissingFromOptions = Boolean(
      currentModel && !selectedModelKnownOption,
    );
    if (
      currentModels.length > 0 &&
      (!currentModel ||
        currentModelKnownButIncompatible ||
        (currentModelMissingFromOptions && !preserveUnknownModelSelection))
    ) {
      setModel(currentModels[0]);
    }
  }, [
    allowAutoModel,
    currentModels,
    modelsLoading,
    preserveUnknownModelSelection,
    selectedProvider,
    selectedModelKnownOption,
    selectedProviderBlocksModelLoad,
    setModel,
    shouldLoadModels,
    suppressAutoSelection,
  ]);

  useEffect(() => {
    if (!import.meta.env.DEV) return;
    if (!selectedProvider) return;
    if (!activeTheme) return;
    if (!filteredResult.usedFallback && filteredResult.filteredOutCount === 0) {
      return;
    }

    console.debug("[ModelSelector] 主题模型过滤结果", {
      theme: activeTheme,
      provider: selectedProvider.key,
      policyName: filteredResult.policyName,
      filteredOutCount: filteredResult.filteredOutCount,
      usedFallback: filteredResult.usedFallback,
    });
  }, [
    activeTheme,
    filteredResult.filteredOutCount,
    filteredResult.policyName,
    filteredResult.usedFallback,
    selectedProvider,
  ]);

  useEffect(() => {
    if (!disabled) return;
    if (!open) return;
    setOpen(false);
  }, [disabled, open]);

  const defaultHubProviderLabel = resolveOemLimeHubProviderName(
    resolveOemCloudRuntimeContext(),
  );
  const selectedProviderUnavailable = Boolean(
    providerType.trim() &&
      shouldLoadProviders &&
      !providersLoading &&
      !selectedProvider,
  );
  const compactProviderType =
    selectedProvider?.key || providerType || "lime-hub";
  const fallbackProviderLabel =
    compactProviderType.toLowerCase() === "lime-hub"
      ? defaultHubProviderLabel
      : getProviderLabel(compactProviderType);
  const hasModelWithoutProvider = !providerType.trim() && Boolean(model.trim());
  const isAutoSelection = !providerType.trim() && !model.trim();
  const showPlaceholderSelection =
    hasModelWithoutProvider ||
    (isAutoSelection && (allowAutoProvider || suppressAutoSelection));
  const resolvedAutoProviderLabel =
    autoProviderLabel ?? t("common.modelSelector.autoSelect");
  const resolvedAutoModelLabel =
    autoModelLabel ?? t("common.modelSelector.autoSelect");
  const resolvedPlaceholderLabel =
    placeholderLabel ?? t("common.modelSelector.placeholder");
  const resolvedEmptyStateTitle =
    emptyStateTitle ?? t("common.modelSelector.noProvider.title");
  const resolvedEmptyStateDescription =
    emptyStateDescription ?? t("common.modelSelector.noProvider.description");
  const selectedProviderLabel = selectedProviderUnavailable
    ? t("common.modelSelector.provider.selectedUnavailable")
    : showPlaceholderSelection
    ? resolvedPlaceholderLabel
    : selectedProvider?.label ||
      (allowAutoProvider && !providerType.trim()
        ? resolvedAutoProviderLabel
        : fallbackProviderLabel);
  const compactProviderLabel = selectedProviderUnavailable
    ? t("common.modelSelector.provider.selectedUnavailable")
    : showPlaceholderSelection
    ? resolvedPlaceholderLabel
    : selectedProvider?.label ||
      (allowAutoProvider && !providerType.trim()
        ? resolvedAutoProviderLabel
        : fallbackProviderLabel);
  const selectedModelLabel =
    selectedProviderUnavailable
      ? t("common.modelSelector.model.selectedUnavailable")
      : !showPlaceholderSelection && selectedProviderBlocksModelLoad
      ? t("common.modelSelector.state.loginRequired")
      : showPlaceholderSelection
        ? resolvedPlaceholderLabel
        : model ||
          (allowAutoModel
            ? resolvedAutoModelLabel
            : t("common.modelSelector.placeholder"));
  const selectedModelWithReasoningLabel =
    selectedReasoningEffortLabel && !showPlaceholderSelection
      ? `${selectedModelLabel} ${selectedReasoningEffortLabel}`
      : selectedModelLabel;
  const compactModelLabel = selectedModelWithReasoningLabel;
  const normalizedTheme = (activeTheme || "").toLowerCase();
  const activeThemeLabel =
    normalizedTheme === "general"
      ? t("common.modelSelector.theme.general")
      : activeTheme || t("common.modelSelector.theme.current");
  const showThemeFilterHint =
    normalizedTheme !== "" &&
    normalizedTheme !== "general" &&
    !filteredResult.usedFallback &&
    filteredResult.filteredOutCount > 0;
  const showNoProviderGuide =
    shouldLoadProviders &&
    !providersLoading &&
    visibleProviders.length === 0 &&
    !noProviderGuideDismissed;

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (visibleProviders.length === 0) {
      return;
    }
    window.localStorage.removeItem(NO_PROVIDER_GUIDE_DISMISSED_STORAGE_KEY);
    setNoProviderGuideDismissed(false);
  }, [visibleProviders.length]);

  const handleDismissNoProviderGuide = () => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(NO_PROVIDER_GUIDE_DISMISSED_STORAGE_KEY, "1");
    }
    setNoProviderGuideDismissed(true);
  };

  if (showNoProviderGuide) {
    return (
      <div
        className={cn(
          "w-full rounded-lg border border-amber-200 bg-amber-50/60 p-3",
          className,
        )}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-start gap-2 min-w-0">
            <AlertCircle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
            <div className="min-w-0">
              <div className="text-sm font-medium text-amber-900">
                {resolvedEmptyStateTitle}
              </div>
              <div className="text-xs text-amber-700 leading-5">
                {resolvedEmptyStateDescription}
              </div>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {onManageProviders && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 shrink-0 border-amber-300 bg-white text-amber-800 hover:bg-amber-100 hover:text-amber-900"
                onClick={onManageProviders}
              >
                {t("common.modelSelector.action.configure")}
              </Button>
            )}
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0 rounded-full text-amber-700 hover:bg-amber-100 hover:text-amber-900"
              aria-label={t("common.modelSelector.noProvider.dismissAria")}
              onClick={handleDismissNoProviderGuide}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("flex items-center min-w-0", className)}>
      <Popover
        modal={false}
        open={open}
        onOpenChange={(nextOpen) => {
          if (disabled) {
            return;
          }
          setOpen(nextOpen);
        }}
      >
        <PopoverTrigger asChild>
          {compactTrigger ? (
            <Button
              variant="outline"
              role="combobox"
              data-testid="model-selector"
              aria-expanded={open}
              disabled={disabled}
              className={cn(
                compactTriggerClassName,
                open && "border-slate-300 bg-white text-slate-700",
              )}
              title={
                showPlaceholderSelection
                  ? resolvedPlaceholderLabel
                  : `${selectedProviderLabel} / ${selectedModelWithReasoningLabel}`
              }
            >
              <ProviderIcon
                providerType={compactProviderType}
                fallbackText={compactProviderLabel}
                size={15}
              />
              <span className="min-w-0 flex-1 truncate text-xs font-medium">
                {compactModelLabel}
              </span>
              <ChevronDown className="h-3 w-3 shrink-0 text-slate-400 opacity-80" />
            </Button>
          ) : (
            <Button
              variant="outline"
              role="combobox"
              data-testid="model-selector"
              aria-expanded={open}
              disabled={disabled}
              className={defaultTriggerClassName}
            >
              <Bot size={16} className="text-slate-500" />
              {showPlaceholderSelection ? (
                <span className="min-w-0 flex-1 font-medium text-left text-slate-700">
                  {resolvedPlaceholderLabel}
                </span>
              ) : (
                <span className="min-w-0 flex-1 flex items-center gap-1.5">
                  <span className="font-medium truncate">
                    {selectedProviderLabel}
                  </span>
                  <span className="text-slate-300 shrink-0">/</span>
                  <span className="text-sm text-slate-500 truncate">
                    {selectedModelWithReasoningLabel}
                  </span>
                </span>
              )}
              <ChevronDown className="ml-1 h-3 w-3 text-slate-400 opacity-70" />
            </Button>
          )}
        </PopoverTrigger>

        <PopoverContent
          data-model-selector-popover="true"
          className="z-[80] w-[440px] max-w-[calc(100vw-24px)] overflow-hidden rounded-[22px] border border-slate-200/80 bg-white p-0 shadow-xl shadow-slate-950/8 opacity-100"
          align="start"
          side={popoverSide}
          sideOffset={8}
          avoidCollisions
          collisionPadding={8}
        >
          <div className="border-b border-slate-200/80 bg-[linear-gradient(180deg,rgb(255,255,255)_0%,rgb(248,250,252)_100%)] px-4 py-3">
            <div className="text-[11px] font-semibold tracking-[0.08em] text-slate-500">
              {t("common.modelSelector.header.title")}
            </div>
            {showPlaceholderSelection ? (
              <div className="mt-1 text-sm font-medium text-slate-700">
                {resolvedPlaceholderLabel}
              </div>
            ) : (
              <div className="mt-1 flex items-center gap-2 text-sm text-slate-700">
                <span className="font-medium">{selectedProviderLabel}</span>
                <span className="text-slate-300">/</span>
                <span className="truncate text-slate-500">
                  {selectedModelWithReasoningLabel}
                </span>
              </div>
            )}
            {activeTheme ? (
              <div className="mt-1 text-xs text-slate-500">
                {t("common.modelSelector.header.themeFilter", {
                  theme: activeThemeLabel,
                })}
              </div>
            ) : null}
            {selectedPromptCacheNotice ? (
              <div className="mt-1 text-xs text-amber-700">
                {t("common.modelSelector.notice.promptCacheExplicit")}
              </div>
            ) : null}
          </div>

          <div className="flex h-[336px]">
            <div className="flex w-[156px] flex-col gap-1 overflow-y-auto border-r border-slate-200/80 bg-slate-50 p-2">
              <div className="mb-1 px-2 py-1 text-[11px] font-semibold tracking-[0.08em] text-slate-500">
                {t("common.modelSelector.provider.title")}
              </div>

              {visibleProviders.length === 0 ? (
                <div className="px-2 py-3 text-xs leading-5 text-slate-500">
                  {t("common.modelSelector.provider.empty")}
                </div>
              ) : (
                <>
                  {allowAutoProvider ? (
                    <button
                      onClick={() => {
                        commitProviderAndModel("", "");
                        setReasoningEffort?.("");
                        setOpen(false);
                      }}
                      className={cn(
                        itemClassName,
                        !providerType.trim()
                          ? "border-slate-200 bg-white text-slate-900 shadow-sm shadow-slate-950/5"
                          : "text-slate-500 hover:border-slate-200 hover:bg-white hover:text-slate-900",
                      )}
                    >
                      <span className="truncate">
                        {resolvedAutoProviderLabel}
                      </span>
                      {!providerType.trim() ? (
                        <div className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                      ) : null}
                    </button>
                  ) : null}
                  {selectedProviderUnavailable ||
                  (selectedProvider && !selectedProviderVisible) ? (
                    <div className="rounded-xl border border-amber-200 bg-amber-50 px-2.5 py-2 text-[11px] leading-5 text-amber-700">
                      {t("common.modelSelector.provider.selectedUnavailable")}
                    </div>
                  ) : null}
                  {[
                    {
                      key: "cloud",
                      title: t("common.modelSelector.provider.section.cloud"),
                      providers: cloudProviders,
                    },
                    {
                      key: "local",
                      title: t("common.modelSelector.provider.section.local"),
                      providers: localProviders,
                    },
                  ]
                    .filter((section) => section.providers.length > 0)
                    .map((section) => (
                      <div key={section.key} className="space-y-1">
                        <div className="px-2 pt-2 text-[10px] font-semibold tracking-[0.08em] text-slate-400">
                          {section.title}
                        </div>
                        {section.providers.map((provider) => {
                          const isSelected =
                            selectedProvider?.key === provider.key;
                          const providerPromptCacheMode =
                            getProviderPromptCacheMode(
                              provider.type,
                              provider.promptCacheMode,
                              provider.apiHost,
                            );

                          return (
                            <button
                              key={provider.key}
                              onClick={() => {
                                const nextModel = resolveProviderSelectionModel(
                                  {
                                    provider,
                                    modelFilter,
                                    getFallbackModels,
                                    isSelected,
                                    currentModelOptions: modelOptions,
                                  },
                                );
                                commitProviderAndModel(
                                  resolveProviderSelectionValue(provider),
                                  nextModel,
                                );
                                setReasoningEffort?.("");
                              }}
                              className={cn(
                                itemClassName,
                                isSelected
                                  ? "border-slate-200 bg-white text-slate-900 shadow-sm shadow-slate-950/5"
                                  : "text-slate-500 hover:border-slate-200 hover:bg-white hover:text-slate-900",
                              )}
                            >
                              <span className="flex min-w-0 items-center gap-2">
                                <ProviderIcon
                                  providerType={provider.key}
                                  fallbackText={provider.label}
                                  size={15}
                                />
                                <span className="flex min-w-0 flex-col gap-1">
                                  <span className="truncate">
                                    {provider.label}
                                  </span>
                                  {provider.authStatus === "login_required" ? (
                                    <span className="text-[10px] leading-4 text-amber-700">
                                      {t(
                                        "common.modelSelector.provider.badge.loginRequired",
                                      )}
                                    </span>
                                  ) : providerPromptCacheMode ===
                                    "explicit_only" ? (
                                    <span className="text-[10px] leading-4 text-amber-700">
                                      {t(
                                        "common.modelSelector.provider.badge.explicitCache",
                                      )}
                                    </span>
                                  ) : null}
                                </span>
                              </span>
                              {isSelected && (
                                <div className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                              )}
                            </button>
                          );
                        })}
                      </div>
                    ))}
                </>
              )}
            </div>

            <div className="flex flex-1 flex-col overflow-hidden p-2.5">
              <div className="mb-1 px-2 py-1 text-[11px] font-semibold tracking-[0.08em] text-slate-500">
                {t("common.modelSelector.model.title")}
              </div>
              {showThemeFilterHint ||
              (normalizedTheme !== "general" && filteredResult.usedFallback) ||
              incompatibleModelCount > 0 ? (
                <div className="mb-2 space-y-1 rounded-2xl border border-slate-200/80 bg-slate-50 px-3 py-2">
                  {showThemeFilterHint ? (
                    <div className="text-[11px] leading-5 text-slate-500">
                      {t("common.modelSelector.model.themeFiltered", {
                        theme: activeThemeLabel,
                      })}
                    </div>
                  ) : null}
                  {normalizedTheme !== "general" &&
                  filteredResult.usedFallback ? (
                    <div className="text-[11px] leading-5 text-amber-700">
                      {t("common.modelSelector.model.themeFallback", {
                        theme: activeThemeLabel,
                      })}
                    </div>
                  ) : null}
                  {incompatibleModelCount > 0 ? (
                    <div className="text-[11px] leading-5 text-amber-700">
                      {t("common.modelSelector.model.incompatibleHidden", {
                        count: incompatibleModelCount,
                      })}
                    </div>
                  ) : null}
                </div>
              ) : null}

              {selectedReasoningEffortOptions.length > 0 ? (
                <div
                  className="mb-2 rounded-2xl border border-slate-200/80 bg-slate-50 px-2.5 py-2"
                  data-testid="model-selector-reasoning-effort"
                >
                  <div className="mb-1.5 flex items-center justify-between gap-2 px-1">
                    <div className="text-[11px] font-semibold tracking-[0.08em] text-slate-500">
                      {t("common.modelSelector.reasoning.title")}
                    </div>
                    <div className="truncate text-[11px] text-slate-400">
                      {selectedModelOption?.metadata.source === "api"
                        ? t("common.modelSelector.reasoning.source.api")
                        : null}
                    </div>
                  </div>
                  <div
                    className="grid gap-1"
                    style={{
                      gridTemplateColumns: `repeat(${selectedReasoningEffortOptions.length}, minmax(0, 1fr))`,
                    }}
                  >
                    {selectedReasoningEffortOptions.map((option) => {
                      const label =
                        option.label ||
                        t(
                          `common.modelSelector.reasoning.level.${option.value}`,
                          { defaultValue: option.id },
                        );
                      const selected =
                        selectedReasoningEffort === option.value ||
                        (!selectedReasoningEffort &&
                          option.value ===
                            resolveReasoningEffortForModelSwitch(
                              selectedModelOption?.metadata,
                              reasoningEffort,
                            ));
                      return (
                        <button
                          key={option.id}
                          type="button"
                          aria-pressed={selected}
                          aria-label={t("common.modelSelector.reasoning.aria", {
                            level: label,
                          })}
                          className={cn(
                            "inline-flex h-7 min-w-0 items-center justify-center gap-1 rounded-full px-2 text-xs font-medium transition-colors",
                            selected
                              ? "bg-slate-900 text-white shadow-sm shadow-slate-950/10"
                              : "text-slate-500 hover:bg-white hover:text-slate-900",
                          )}
                          onClick={() => setReasoningEffort?.(option.value)}
                        >
                          <span className="truncate">{label}</span>
                          {selected ? (
                            <Check size={12} className="shrink-0 opacity-80" />
                          ) : null}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : null}

              <ScrollArea className="flex-1">
                <div className="space-y-1 p-1">
                  {!selectedProvider &&
                  (allowAutoProvider || suppressAutoSelection) ? (
                    <div className="rounded-2xl border border-dashed border-slate-200 px-3 py-6 text-center text-xs text-slate-500">
                      {t("common.modelSelector.model.selectProviderFirst")}
                    </div>
                  ) : selectedProviderBlocksModelLoad ? (
                    <div className="rounded-2xl border border-amber-200 bg-amber-50 px-3 py-4 text-xs leading-5 text-amber-800">
                      <div className="flex items-start gap-2 text-left">
                        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                        <div className="min-w-0">
                          <div className="font-medium text-amber-900">
                            {t(
                              "common.modelSelector.model.loginRequiredTitle",
                              {
                                provider: selectedProvider?.label ?? "Lime Hub",
                              },
                            )}
                          </div>
                          <div className="mt-1 text-amber-700">
                            {t(
                              "common.modelSelector.model.loginRequiredDescription",
                            )}
                          </div>
                          {onManageProviders ? (
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="mt-3 h-8 border-amber-300 bg-white text-amber-800 hover:bg-amber-100 hover:text-amber-900"
                              onClick={() => {
                                setOpen(false);
                                onManageProviders();
                              }}
                            >
                              {t("common.modelSelector.action.signIn")}
                            </Button>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  ) : modelOptions.length === 0 ? (
                    <div className="rounded-2xl border border-dashed border-slate-200 px-3 py-6 text-center text-xs text-slate-500">
                      {t("common.modelSelector.model.empty")}
                    </div>
                  ) : (
                    <>
                      {allowAutoModel && selectedProvider ? (
                        <button
                          onClick={() => {
                            setModel("");
                            setReasoningEffort?.("");
                            setOpen(false);
                          }}
                          className={cn(
                            `${itemClassName} group`,
                            !model.trim()
                              ? "border-slate-200 bg-slate-50 text-slate-900"
                              : "text-slate-500 hover:border-slate-200 hover:bg-slate-50 hover:text-slate-900",
                          )}
                        >
                          <span className="truncate">
                            {resolvedAutoModelLabel}
                          </span>
                          {!model.trim() ? (
                            <Check size={14} className="text-slate-900" />
                          ) : null}
                        </button>
                      ) : null}
                      {selectedProvider &&
                      model.trim() &&
                      !currentModels.includes(model) ? (
                        <div className="rounded-xl border border-amber-200 bg-amber-50 px-2.5 py-2 text-[11px] leading-5 text-amber-700">
                          {t("common.modelSelector.model.selectedUnavailable")}
                        </div>
                      ) : null}
                      {modelOptions.map((currentModelItem) => (
                        <button
                          key={currentModelItem.id}
                          disabled={Boolean(
                            currentModelItem.compatibilityIssue,
                          )}
                          onClick={() => {
                            if (currentModelItem.compatibilityIssue) {
                              return;
                            }
                            setModel(currentModelItem.id);
                            setReasoningEffort?.(
                              resolveReasoningEffortForModelSwitch(
                                currentModelItem.metadata,
                                reasoningEffort,
                              ),
                            );
                            setOpen(false);
                          }}
                          className={cn(
                            `${itemClassName} group`,
                            currentModelItem.compatibilityIssue
                              ? "cursor-not-allowed border-transparent bg-transparent text-slate-400 opacity-70"
                              : model === currentModelItem.id
                                ? "border-slate-200 bg-slate-50 text-slate-900"
                                : "text-slate-500 hover:border-slate-200 hover:bg-slate-50 hover:text-slate-900",
                          )}
                          title={currentModelItem.compatibilityIssue?.message}
                        >
                          <span className="flex items-center gap-2 min-w-0">
                            {selectedProvider && (
                              <ProviderIcon
                                providerType={selectedProvider.key}
                                fallbackText={selectedProvider.label}
                                size={15}
                              />
                            )}
                            <span className="min-w-0 flex flex-col gap-1">
                              <span className="truncate">
                                {currentModelItem.id}
                              </span>
                              <ModelCapabilityBadges
                                capabilities={
                                  currentModelItem.metadata.capabilities
                                }
                                model={currentModelItem.metadata}
                                compact
                              />
                              {currentModelItem.compatibilityIssue ? (
                                <span className="truncate text-[11px] text-amber-700">
                                  {currentModelItem.compatibilityIssue.message}
                                </span>
                              ) : null}
                            </span>
                          </span>
                          {currentModelItem.compatibilityIssue ? (
                            <AlertCircle size={14} className="text-amber-500" />
                          ) : model === currentModelItem.id ? (
                            <Check size={14} className="text-slate-900" />
                          ) : null}
                        </button>
                      ))}
                    </>
                  )}
                </div>
              </ScrollArea>
            </div>
          </div>

          {onManageProviders && (
            <button
              type="button"
              className="flex h-11 w-full items-center justify-between border-t border-slate-200/80 px-4 text-sm text-slate-600 transition-colors hover:bg-slate-50 hover:text-slate-900"
              onClick={() => {
                setOpen(false);
                onManageProviders();
              }}
            >
              <span className="inline-flex items-center gap-2">
                <Settings2 size={14} className="text-slate-400" />
                {t("common.modelSelector.action.manageProviders")}
              </span>
              <ArrowRight size={14} className="text-slate-400" />
            </button>
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
};
