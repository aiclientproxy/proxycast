/**
 * @file ProviderSetting 组件
 * @description Provider 的简洁配置页，只保留密钥、模型优先级和连接测试。
 * @module components/api-key-provider/ProviderSetting
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  Eye,
  Loader2,
  Plus,
  RefreshCw,
  Save,
  Sparkles,
  Star,
  Trash2,
  X,
} from "lucide-react";
import { ProviderIcon } from "@/icons/providers";
import {
  apiKeyProviderApi,
  providerModelIds,
  reconcileProviderModels,
  type ProviderWithKeysDisplay,
  type UpdateProviderRequest,
} from "@/lib/api/apiKeyProvider";
import {
  fetchProviderModelsAuto,
  normalizeFetchProviderModelsSource,
} from "@/lib/api/modelRegistry";
import { isLikelyImageGenerationModelId } from "@/lib/imageGen/providerMatchers";
import {
  findModelBoundImageCommandEntryForModel,
  getCurrentSkillCatalogSnapshot,
  subscribeSkillCatalogChanged,
  upsertLocalModelBoundImageCommandBinding,
} from "@/lib/api/skillCatalog";
import { resolveLayeredDesignImageExecutorMode } from "@/lib/layered-design/imageModelCapabilities";
import { getProviderModelAutoFetchCapability } from "@/lib/model/providerModelFetchSupport";
import { getProviderPromptCacheMode } from "@/lib/model/providerPromptCacheSupport";
import {
  interceptHttpExternalLinkClick,
  resolveHttpExternalHref,
} from "@/lib/markdown/externalLinks";
import { getProviderAccessHelp } from "@/lib/provider/providerAccessHelp";
import { dedupeModelIds, getProviderTypeLabel } from "./providerConfigUtils";
import type {
  ConnectionTestOptions,
  ConnectionTestResult,
} from "./connectionTestTypes";
import {
  buildFalModelFetchStatus,
  buildResponsesModelFetchStatus,
  extractApiModelIds,
  isFalProviderLike,
  isLikelyFalImageModel,
  isProviderApiKeyRequired,
  type ProviderModelFetchStatusCopy,
} from "./providerModelFetchHelpers";
import type { ProviderSettingsFocusContext } from "@/types/page";
import { formatProviderConnectionError } from "./providerConnectionError";

// ============================================================================
// 类型定义
// ============================================================================

export interface ProviderSettingProps {
  /** Provider 数据（包含 API Keys） */
  provider: ProviderWithKeysDisplay | null;
  /** 从运行诊断进入设置页时的焦点 */
  focus?: ProviderSettingsFocusContext | null;
  /** 当前授权状态 */
  authStatus?: "ready" | "login_required";
  /** 触发登录或授权 */
  onLogin?: () => void | Promise<void>;
  /** 更新 Provider 配置回调 */
  onUpdate?: (id: string, request: UpdateProviderRequest) => Promise<void>;
  /** 添加 API Key 回调 */
  onAddApiKey?: (
    providerId: string,
    apiKey: string,
    alias?: string,
    options?: { replaceExisting?: boolean },
  ) => Promise<void>;
  /** 测试连接回调 */
  onTestConnection?: (
    providerId: string,
    options?: ConnectionTestOptions,
  ) => Promise<ConnectionTestResult>;
  /** 删除或停用 Provider 配置回调 */
  onDeleteProvider?: (providerId: string) => Promise<boolean | void>;
  /** 是否正在加载 */
  loading?: boolean;
  /** 额外的 CSS 类名 */
  className?: string;
}

interface ProviderSettingBodyProps extends ProviderSettingProps {
  provider: ProviderWithKeysDisplay;
}

type InlineStatusTone = "success" | "error" | "info";

interface InlineStatus {
  tone: InlineStatusTone;
  message: string;
}

type ImageModelCommandExecutorMode =
  | "images_api"
  | "responses_image_generation";

// ============================================================================
// 辅助函数
// ============================================================================

function normalizeProviderHostValue(apiHost: unknown): string {
  return typeof apiHost === "string" ? apiHost : "";
}

function formatProviderHost(apiHost: unknown): string {
  const normalizedApiHost = normalizeProviderHostValue(apiHost);
  try {
    const url = new URL(normalizedApiHost);
    return `${url.host}${url.pathname === "/" ? "" : url.pathname}`;
  } catch {
    return normalizedApiHost;
  }
}

function getFirstVisibleApiKey(provider: ProviderWithKeysDisplay): string {
  const key =
    provider.api_keys?.find((apiKey) => apiKey.enabled) ??
    provider.api_keys?.[0];
  return key?.api_key_masked ?? "";
}

function hasConfiguredApiKey(provider: ProviderWithKeysDisplay): boolean {
  if (provider.api_keys && provider.api_keys.length > 0) {
    return provider.api_keys.some((apiKey) => apiKey.enabled);
  }

  return provider.api_key_count > 0;
}

function isLikelyImageModelCommandModel(modelId: string): boolean {
  return isLikelyImageGenerationModelId(modelId);
}

function titleCaseAsciiWords(value: string): string {
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((segment) =>
      segment.length <= 3
        ? segment.toUpperCase()
        : segment[0]!.toUpperCase() + segment.slice(1),
    )
    .join(" ");
}

function buildSuggestedImageCommandTrigger(modelId: string): string {
  const normalized = modelId.trim();
  const lower = normalized.toLowerCase();
  if (lower === "gpt-images-2" || lower === "gpt-image-2") {
    return "@GPT Images 2";
  }
  if (lower.includes("nano-banana-2")) {
    return "@Nano Banana 2";
  }
  if (lower.includes("nano-banana-pro")) {
    return "@Nanobanana Pro";
  }

  const visibleModelName = normalized.split("/").pop() || normalized;
  return `@${titleCaseAsciiWords(visibleModelName) || "Image Model"}`;
}

function resolveSuggestedImageCommandExecutorMode(
  provider: ProviderWithKeysDisplay,
  modelId: string,
): ImageModelCommandExecutorMode {
  return resolveLayeredDesignImageExecutorMode({
    providerId: provider.id,
    model: modelId,
  });
}

function readCommandTriggerLabel(entry: {
  triggers?: Array<{ prefix: string }>;
}): string {
  return entry.triggers?.[0]?.prefix ?? "@图片模型";
}

function parseModelDraft(value: string): string[] {
  return dedupeModelIds(
    value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
  );
}

function buildStatusClass(tone: InlineStatusTone): string {
  switch (tone) {
    case "success":
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    case "error":
      return "border-rose-200 bg-rose-50 text-rose-700";
    case "info":
    default:
      return "border-sky-200 bg-sky-50 text-sky-700";
  }
}

function getStatusIcon(tone: InlineStatusTone) {
  if (tone === "success") {
    return <CheckCircle2 className="h-4 w-4" />;
  }
  if (tone === "error") {
    return <AlertCircle className="h-4 w-4" />;
  }
  return <Sparkles className="h-4 w-4" />;
}

function readActionErrorMessage(error: unknown): string | null {
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }

  if (typeof error === "string" && error.trim()) {
    return error;
  }

  if (error && typeof error === "object" && "message" in error) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }

  return null;
}

function formatActionError(error: unknown, fallback: string): string {
  return readActionErrorMessage(error) ?? fallback;
}

function isMissingApiKeyFocus(
  focus: ProviderSettingsFocusContext | null | undefined,
): boolean {
  return (
    focus?.recoveryAction === "add_enabled_api_key" ||
    focus?.reasonCode === "missing_enabled_api_key"
  );
}

// ============================================================================
// 组件实现
// ============================================================================

/**
 * Provider 设置面板组件
 *
 * 只保留最常用路径：API Key、模型优先级、接口获取模型、手动添加模型、测试连接。
 */
export const ProviderSetting: React.FC<ProviderSettingProps> = (props) => {
  const { t } = useTranslation("settings");

  if (!props.provider) {
    return (
      <div
        className={cn(
          "flex h-full items-center justify-center bg-slate-50 px-6",
          props.className,
        )}
        data-testid="provider-setting-empty"
      >
        <div className="w-full max-w-[720px] rounded-[28px] border border-slate-200/80 bg-white p-8 shadow-sm shadow-slate-950/5">
          <div className="flex items-center gap-3 text-slate-900">
            <Sparkles className="h-5 w-5" />
            <p className="text-lg font-semibold">
              {t("settings.providers.setting.empty.title", "选择或添加模型")}
            </p>
          </div>
          <p className="mt-3 text-sm leading-6 text-slate-500">
            {t(
              "settings.providers.setting.empty.description",
              "左侧选择一个已启用模型后，这里只展示密钥、模型优先级和测试连接。",
            )}
          </p>
        </div>
      </div>
    );
  }

  if (props.authStatus === "login_required") {
    return (
      <div
        className={cn("flex h-full flex-col bg-slate-50", props.className)}
        data-testid="provider-login-required"
        data-provider-id={props.provider.id}
      >
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 lg:px-6">
          <section className="mx-auto w-full max-w-[820px] rounded-[30px] border border-amber-200 bg-amber-50 p-5 shadow-sm shadow-slate-950/5 lg:p-6">
            <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex min-w-0 items-start gap-3">
                <ProviderIcon
                  providerType={props.provider.id}
                  fallbackText={props.provider.name}
                  size={48}
                  className="flex-shrink-0"
                  data-testid="provider-icon"
                />
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="min-w-0 truncate text-xl font-semibold tracking-tight text-amber-950">
                      {props.provider.name}
                    </h3>
                    <Badge
                      variant="outline"
                      className="border-amber-300 bg-white text-amber-800"
                    >
                      {t(
                        "settings.providers.setting.loginRequired.badge",
                        "需要登录",
                      )}
                    </Badge>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-amber-800">
                    {t(
                      "settings.providers.setting.loginRequired.description",
                      "登录后会自动同步 Lime Hub 的可用模型和本地托管访问凭证；未登录时不会展示本地兜底模型。",
                    )}
                  </p>
                </div>
              </div>

              {props.onLogin ? (
                <Button
                  type="button"
                  variant="outline"
                  className="h-10 shrink-0 rounded-full border-amber-300 bg-white px-4 text-amber-900 hover:bg-amber-100 hover:text-amber-950"
                  onClick={() => {
                    void props.onLogin?.();
                  }}
                  disabled={props.loading}
                  data-testid="provider-login-button"
                >
                  <ExternalLink className="mr-1.5 h-4 w-4" />
                  {t(
                    "settings.providers.setting.loginRequired.action.login",
                    "去登录",
                  )}
                </Button>
              ) : null}
            </div>
          </section>
        </div>
      </div>
    );
  }

  return <ProviderSettingBody {...props} provider={props.provider} />;
};

const ProviderSettingBody: React.FC<ProviderSettingBodyProps> = ({
  provider,
  focus,
  onUpdate,
  onAddApiKey,
  onTestConnection,
  onDeleteProvider,
  loading = false,
  className,
}) => {
  const { t } = useTranslation("settings");
  const [modelList, setModelList] = useState<string[]>(
    providerModelIds(provider.models),
  );
  const [modelDraft, setModelDraft] = useState("");
  const [apiModelIds, setApiModelIds] = useState<string[]>([]);
  const [apiModelQuery, setApiModelQuery] = useState("");
  const [fetchingModels, setFetchingModels] = useState(false);
  const [modelFetchStatus, setModelFetchStatus] = useState<InlineStatus | null>(
    null,
  );
  const [providerActionStatus, setProviderActionStatus] =
    useState<InlineStatus | null>(null);
  const [deletingProvider, setDeletingProvider] = useState(false);
  const [apiKeyDraft, setApiKeyDraft] = useState("");
  const [apiKeyDirty, setApiKeyDirty] = useState(false);
  const [savingApiKey, setSavingApiKey] = useState(false);
  const [apiKeyStatus, setApiKeyStatus] = useState<InlineStatus | null>(null);
  const [apiHostDraft, setApiHostDraft] = useState(() =>
    normalizeProviderHostValue(provider.api_host),
  );
  const [apiHostDirty, setApiHostDirty] = useState(false);
  const [savingApiHost, setSavingApiHost] = useState(false);
  const [apiHostStatus, setApiHostStatus] = useState<InlineStatus | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<InlineStatus | null>(
    null,
  );
  const [imageCommandDraft, setImageCommandDraft] = useState<{
    modelId: string;
    trigger: string;
  } | null>(null);
  const [imageCommandStatus, setImageCommandStatus] =
    useState<InlineStatus | null>(null);
  const [skillCatalogRevision, setSkillCatalogRevision] = useState(0);

  useEffect(() => {
    setModelList(providerModelIds(provider.models));
    setModelDraft("");
    setApiModelIds([]);
    setApiModelQuery("");
    setModelFetchStatus(null);
    setProviderActionStatus(null);
    setConnectionStatus(null);
    setApiKeyDraft("");
    setApiKeyDirty(false);
    setSavingApiKey(false);
    setApiKeyStatus(null);
    setApiHostDraft(normalizeProviderHostValue(provider.api_host));
    setApiHostDirty(false);
    setSavingApiHost(false);
    setApiHostStatus(null);
    setShowApiKey(false);
    setImageCommandDraft(null);
    setImageCommandStatus(null);
  }, [provider.id, provider.api_host, provider.models]);

  useEffect(() => {
    return subscribeSkillCatalogChanged(() => {
      setSkillCatalogRevision((revision) => revision + 1);
    });
  }, []);

  const normalizedProviderApiHost = normalizeProviderHostValue(
    provider.api_host,
  );
  const providerHostLabel = formatProviderHost(normalizedProviderApiHost);
  const apiKeyMask = getFirstVisibleApiKey(provider);
  const hasApiKey = hasConfiguredApiKey(provider);
  const effectiveApiHost = apiHostDirty
    ? apiHostDraft
    : normalizedProviderApiHost;
  const trimmedApiHostDraft = apiHostDraft.trim();
  const hasPendingApiHost =
    apiHostDirty && trimmedApiHostDraft !== normalizedProviderApiHost.trim();
  const accessHelp = getProviderAccessHelp({
    providerId: provider.id,
    providerName: provider.name,
    apiHost: effectiveApiHost,
  });
  const accessHelpRel = resolveHttpExternalHref(accessHelp.url)
    ? "noreferrer noopener"
    : undefined;
  const modelAutoFetchCapability = getProviderModelAutoFetchCapability({
    providerId: provider.id,
    providerType: provider.type,
    apiHost: effectiveApiHost,
  });
  const modelFetchApiKeyRequired = modelAutoFetchCapability.requiresApiKey;
  const providerApiKeyRequired = isProviderApiKeyRequired(
    provider,
    modelFetchApiKeyRequired,
  );
  const canUseDraftApiKey = apiKeyDirty && apiKeyDraft.trim().length > 0;
  const hasPendingApiKey = canUseDraftApiKey;
  const canReadModelsFromApi =
    modelAutoFetchCapability.supported &&
    (!modelFetchApiKeyRequired || hasApiKey || canUseDraftApiKey);
  const apiKeyInputValue = apiKeyDirty ? apiKeyDraft : apiKeyMask;
  const primaryModel = modelList[0] ?? null;
  const shouldRunChatReadinessTest = Boolean(
    primaryModel && !isLikelyImageModelCommandModel(primaryModel),
  );
  const normalizedModelSet = useMemo(
    () => new Set(modelList.map((model) => model.toLowerCase())),
    [modelList],
  );
  const availableApiModels = useMemo(
    () =>
      apiModelIds.filter(
        (modelId) => !normalizedModelSet.has(modelId.toLowerCase()),
      ),
    [apiModelIds, normalizedModelSet],
  );
  const normalizedApiModelQuery = apiModelQuery.trim().toLowerCase();
  const suggestedApiModels = useMemo(
    () =>
      normalizedApiModelQuery
        ? availableApiModels.filter((modelId) =>
            modelId.toLowerCase().includes(normalizedApiModelQuery),
          )
        : availableApiModels,
    [availableApiModels, normalizedApiModelQuery],
  );
  const focusedModelId = focus?.modelId?.trim() || "";
  const focusModelMissing = Boolean(
    focusedModelId && !normalizedModelSet.has(focusedModelId.toLowerCase()),
  );
  const showProviderFocusBanner = Boolean(
    focus?.providerId ||
    focus?.modelId ||
    focus?.reasonCode ||
    focus?.recoveryAction,
  );
  const needsApiKeyFromFocus = isMissingApiKeyFocus(focus);
  const imageModelCommandByModel = useMemo(() => {
    void skillCatalogRevision;
    const catalog = getCurrentSkillCatalogSnapshot();
    const entries = new Map<
      string,
      ReturnType<typeof findModelBoundImageCommandEntryForModel>
    >();
    for (const modelId of modelList) {
      if (!isLikelyImageModelCommandModel(modelId)) {
        continue;
      }
      const entry = findModelBoundImageCommandEntryForModel(
        catalog,
        provider.id,
        modelId,
      );
      if (entry) {
        entries.set(modelId, entry);
      }
    }
    return entries;
  }, [modelList, provider.id, skillCatalogRevision]);
  const canDeleteProvider = Boolean(onDeleteProvider);
  const showExplicitPromptCacheBadge =
    getProviderPromptCacheMode(
      provider.type,
      provider.prompt_cache_mode,
      normalizedProviderApiHost,
    ) === "explicit_only";
  const canSaveApiKey =
    !loading &&
    !savingApiHost &&
    !savingApiKey &&
    !testingConnection &&
    !fetchingModels &&
    hasPendingApiKey;
  const canSaveApiHost =
    !loading &&
    !savingApiHost &&
    !savingApiKey &&
    !testingConnection &&
    !fetchingModels &&
    hasPendingApiHost &&
    Boolean(trimmedApiHostDraft);
  const canTestConnection =
    !loading &&
    !testingConnection &&
    !savingApiHost &&
    !savingApiKey &&
    modelList.length > 0 &&
    (!providerApiKeyRequired || hasApiKey || canUseDraftApiKey);
  const modelFetchUnsupportedMessage = useMemo(() => {
    switch (provider.type) {
      case "azure-openai":
        return t(
          "settings.providers.setting.feedback.modelFetch.unsupported.azureOpenai",
        );
      case "vertexai":
        return t(
          "settings.providers.setting.feedback.modelFetch.unsupported.vertexAi",
        );
      case "aws-bedrock":
        return t(
          "settings.providers.setting.feedback.modelFetch.unsupported.awsBedrock",
        );
      default:
        return t(
          "settings.providers.setting.feedback.modelFetch.unsupported.default",
        );
    }
  }, [provider.type, t]);
  const modelFetchStatusCopy = useMemo<ProviderModelFetchStatusCopy>(
    () => ({
      responsesConfirmedImage: (imageModel) =>
        t(
          "settings.providers.setting.feedback.modelFetch.responses.confirmedImage",
          {
            imageModel,
          },
        ),
      responsesManualImage: t(
        "settings.providers.setting.feedback.modelFetch.responses.manualImage",
      ),
      falConfirmedModel: (modelId) =>
        t("settings.providers.setting.feedback.modelFetch.fal.confirmedModel", {
          modelId,
        }),
      falManualModel: t(
        "settings.providers.setting.feedback.modelFetch.fal.manualModel",
      ),
    }),
    [t],
  );

  const persistDraftApiHost = useCallback(async () => {
    const nextApiHost = apiHostDraft.trim();
    if (!apiHostDirty || nextApiHost === normalizedProviderApiHost.trim()) {
      return;
    }

    if (!nextApiHost) {
      throw new Error(
        t("settings.providers.setting.feedback.apiHost.required"),
      );
    }

    if (!onUpdate) {
      throw new Error(
        t(
          "settings.providers.setting.feedback.apiHost.updateMissingCapability",
        ),
      );
    }

    await onUpdate(provider.id, { api_host: nextApiHost });
    setApiHostDraft(nextApiHost);
    setApiHostDirty(false);
  }, [
    apiHostDirty,
    apiHostDraft,
    normalizedProviderApiHost,
    onUpdate,
    provider.id,
    t,
  ]);

  const handleSaveApiHost = useCallback(async () => {
    if (!apiHostDirty) {
      return;
    }

    if (!apiHostDraft.trim()) {
      setApiHostStatus({
        tone: "error",
        message: t("settings.providers.setting.feedback.apiHost.required"),
      });
      return;
    }

    setSavingApiHost(true);
    setApiHostStatus(null);
    setConnectionStatus(null);
    setModelFetchStatus(null);

    try {
      await persistDraftApiHost();
      setApiHostStatus({
        tone: "success",
        message: t("settings.providers.setting.feedback.apiHost.saved"),
      });
    } catch (error) {
      setApiHostStatus({
        tone: "error",
        message: formatActionError(
          error,
          t("settings.providers.setting.feedback.apiHost.saveFailed"),
        ),
      });
    } finally {
      setSavingApiHost(false);
    }
  }, [apiHostDirty, apiHostDraft, persistDraftApiHost, t]);

  const persistDraftApiKey = useCallback(async () => {
    const nextApiKey = apiKeyDraft.trim();
    if (!apiKeyDirty || !nextApiKey) {
      return;
    }

    if (!onAddApiKey) {
      throw new Error(
        t("settings.providers.setting.feedback.apiKey.addMissingCapability"),
      );
    }

    await onAddApiKey(provider.id, nextApiKey, undefined, {
      replaceExisting: true,
    });
    setApiKeyDraft("");
    setApiKeyDirty(false);
  }, [apiKeyDirty, apiKeyDraft, onAddApiKey, provider.id, t]);

  const handleSaveApiKey = useCallback(async () => {
    if (!apiKeyDirty) {
      return;
    }

    if (!apiKeyDraft.trim()) {
      setApiKeyStatus({
        tone: "error",
        message: t("settings.providers.setting.feedback.apiKey.required"),
      });
      return;
    }

    setSavingApiKey(true);
    setApiKeyStatus(null);
    setConnectionStatus(null);

    try {
      await persistDraftApiKey();
      setApiKeyStatus({
        tone: "success",
        message: t("settings.providers.setting.feedback.apiKey.saved"),
      });
    } catch (error) {
      setApiKeyStatus({
        tone: "error",
        message: formatActionError(
          error,
          t("settings.providers.setting.feedback.apiKey.saveFailed"),
        ),
      });
    } finally {
      setSavingApiKey(false);
    }
  }, [apiKeyDirty, apiKeyDraft, persistDraftApiKey, t]);

  const applyModels = useCallback(
    async (nextModels: string[]) => {
      const dedupedModels = dedupeModelIds(nextModels);
      setModelList(dedupedModels);
      setConnectionStatus(null);

      if (onUpdate) {
        await onUpdate(provider.id, {
          models: reconcileProviderModels(provider.models, dedupedModels),
        });
      }
    },
    [onUpdate, provider.id, provider.models],
  );

  const addModels = useCallback(
    async (models: string[]) => {
      const nextModels = dedupeModelIds([...modelList, ...models]);
      await applyModels(nextModels);
    },
    [applyModels, modelList],
  );

  const handleAddModelDraft = useCallback(async () => {
    const nextModels = parseModelDraft(modelDraft);
    if (nextModels.length === 0) {
      return;
    }

    await addModels(nextModels);
    setModelDraft("");
  }, [addModels, modelDraft]);

  const handleRemoveModel = useCallback(
    async (modelId: string) => {
      await applyModels(
        modelList.filter(
          (currentModel) =>
            currentModel.toLowerCase() !== modelId.toLowerCase(),
        ),
      );
    },
    [applyModels, modelList],
  );

  const handleSetMainModel = useCallback(
    async (modelId: string) => {
      await applyModels([
        modelId,
        ...modelList.filter(
          (currentModel) =>
            currentModel.toLowerCase() !== modelId.toLowerCase(),
        ),
      ]);
    },
    [applyModels, modelList],
  );

  const handleOpenImageCommandDraft = useCallback((modelId: string) => {
    setImageCommandStatus(null);
    setImageCommandDraft({
      modelId,
      trigger: buildSuggestedImageCommandTrigger(modelId),
    });
  }, []);

  const handleSaveImageCommandDraft = useCallback(() => {
    if (!imageCommandDraft) {
      return;
    }

    try {
      const entry = upsertLocalModelBoundImageCommandBinding({
        trigger: imageCommandDraft.trigger,
        providerId: provider.id,
        modelId: imageCommandDraft.modelId,
        executorMode: resolveSuggestedImageCommandExecutorMode(
          provider,
          imageCommandDraft.modelId,
        ),
      });
      setImageCommandDraft(null);
      setImageCommandStatus({
        tone: "success",
        message: t("settings.providers.setting.feedback.imageCommand.created", {
          trigger: readCommandTriggerLabel(entry),
          model: imageCommandDraft.modelId,
        }),
      });
      setSkillCatalogRevision((revision) => revision + 1);
    } catch {
      setImageCommandStatus({
        tone: "error",
        message: t(
          "settings.providers.setting.feedback.imageCommand.invalid",
          "请填写有效的 @命令名称。",
        ),
      });
    }
  }, [imageCommandDraft, provider, t]);

  const handleFetchModelsFromApi = useCallback(async () => {
    if (!modelAutoFetchCapability.supported) {
      setModelFetchStatus({
        tone: "info",
        message: modelFetchUnsupportedMessage,
      });
      return;
    }

    if (!canReadModelsFromApi) {
      setModelFetchStatus({
        tone: "error",
        message: t("settings.providers.setting.feedback.modelFetch.needApiKey"),
      });
      return;
    }

    setFetchingModels(true);
    setModelFetchStatus(null);

    try {
      await persistDraftApiHost();
      await persistDraftApiKey();
      const result = await fetchProviderModelsAuto(provider.id);
      const source = normalizeFetchProviderModelsSource(result);
      const fetchedModelIds = extractApiModelIds(result.models ?? []);

      if (source !== "Api") {
        setApiModelIds([]);
        const responsesStatus = buildResponsesModelFetchStatus(
          result,
          modelList,
          modelFetchStatusCopy,
        );
        const falStatus = buildFalModelFetchStatus(
          provider,
          result,
          modelList,
          modelFetchStatusCopy,
        );
        setModelFetchStatus({
          tone:
            responsesStatus?.tone ??
            falStatus?.tone ??
            (source === "Error" ? "error" : "info"),
          message:
            responsesStatus?.message ??
            falStatus?.message ??
            (source === "Error"
              ? (result.error ??
                t(
                  "settings.providers.setting.feedback.modelFetch.errorWithManualFallback",
                ))
              : t(
                  "settings.providers.setting.feedback.modelFetch.emptyRealtime",
                )),
        });
        return;
      }

      const effectiveFetchedModelIds = isFalProviderLike(provider)
        ? fetchedModelIds.filter(isLikelyFalImageModel)
        : fetchedModelIds;

      if (effectiveFetchedModelIds.length === 0) {
        setApiModelIds([]);
        setModelFetchStatus({
          tone: "info",
          message: isFalProviderLike(provider)
            ? t("settings.providers.setting.feedback.modelFetch.emptyFal")
            : t("settings.providers.setting.feedback.modelFetch.empty"),
        });
        return;
      }

      setApiModelIds(effectiveFetchedModelIds);
      setApiModelQuery("");
      const existingFetchedModelCount = effectiveFetchedModelIds.filter(
        (modelId) => normalizedModelSet.has(modelId.toLowerCase()),
      ).length;
      if (existingFetchedModelCount === effectiveFetchedModelIds.length) {
        setModelFetchStatus({
          tone: "success",
          message: t(
            "settings.providers.setting.feedback.modelFetch.confirmedAll",
            {
              count: effectiveFetchedModelIds.length,
            },
          ),
        });
        return;
      }

      setModelFetchStatus({
        tone: "success",
        message: t("settings.providers.setting.feedback.modelFetch.fetched", {
          count: effectiveFetchedModelIds.length,
        }),
      });
    } catch (error) {
      setModelFetchStatus({
        tone: "error",
        message: formatActionError(
          error,
          t("settings.providers.setting.feedback.modelFetch.errorDefault"),
        ),
      });
    } finally {
      setFetchingModels(false);
    }
  }, [
    canReadModelsFromApi,
    modelFetchStatusCopy,
    modelFetchUnsupportedMessage,
    modelList,
    modelAutoFetchCapability.supported,
    normalizedModelSet,
    persistDraftApiHost,
    persistDraftApiKey,
    provider,
    t,
  ]);

  const handleDeleteProvider = useCallback(async () => {
    if (!onDeleteProvider) {
      return;
    }

    const confirmed = window.confirm(
      provider.is_system
        ? t("settings.providers.setting.feedback.delete.confirm.system", {
            providerName: provider.name,
          })
        : t("settings.providers.setting.feedback.delete.confirm.custom", {
            providerName: provider.name,
          }),
    );
    if (!confirmed) {
      return;
    }

    setDeletingProvider(true);
    setProviderActionStatus(null);

    try {
      await onDeleteProvider(provider.id);
      setProviderActionStatus({
        tone: "success",
        message: t("settings.providers.setting.feedback.delete.success"),
      });
    } catch (error) {
      setProviderActionStatus({
        tone: "error",
        message: formatActionError(
          error,
          t("settings.providers.setting.feedback.delete.errorDefault"),
        ),
      });
    } finally {
      setDeletingProvider(false);
    }
  }, [onDeleteProvider, provider.id, provider.is_system, provider.name, t]);

  const handleTestConnection = useCallback(async () => {
    if (modelList.length === 0) {
      setConnectionStatus({
        tone: "error",
        message: t("settings.providers.setting.feedback.connection.needModel"),
      });
      return;
    }

    if (providerApiKeyRequired && !hasApiKey && !canUseDraftApiKey) {
      setConnectionStatus({
        tone: "error",
        message: t("settings.providers.setting.feedback.connection.needApiKey"),
      });
      return;
    }

    setTestingConnection(true);
    setConnectionStatus(null);

    try {
      await persistDraftApiHost();
      await persistDraftApiKey();
      const testPrompt = t(
        "settings.providers.setting.feedback.connection.chatReadyPrompt",
        "请用一句话回复：连接测试通过。",
      );
      const result = onTestConnection
        ? await onTestConnection(provider.id, {
            modelName: primaryModel ?? undefined,
            requireChatReady: shouldRunChatReadinessTest,
            prompt: testPrompt,
          })
        : shouldRunChatReadinessTest
          ? await apiKeyProviderApi
              .testChat(provider.id, primaryModel ?? undefined, testPrompt)
              .then((response) => ({
                success: response.success,
                latencyMs: response.latency_ms,
                error: response.error,
              }))
          : await apiKeyProviderApi
              .testConnection(provider.id, primaryModel ?? undefined)
              .then((response) => ({
                success: response.success,
                latencyMs: response.latency_ms,
                error: response.error,
                models: response.models,
              }));

      if (result.success) {
        setConnectionStatus({
          tone: "success",
          message:
            result.latencyMs !== undefined
              ? shouldRunChatReadinessTest
                ? t(
                    "settings.providers.setting.feedback.connection.chatReadyWithLatency",
                    {
                      latencyMs: result.latencyMs,
                    },
                  )
                : t(
                    "settings.providers.setting.feedback.connection.successWithLatency",
                    {
                      latencyMs: result.latencyMs,
                    },
                  )
              : shouldRunChatReadinessTest
                ? t("settings.providers.setting.feedback.connection.chatReady")
                : t("settings.providers.setting.feedback.connection.success"),
        });
      } else {
        setConnectionStatus({
          tone: "error",
          message: formatProviderConnectionError(result.error, {
            fallback: t(
              "settings.providers.setting.feedback.connection.failureDefault",
            ),
            timeout: t(
              "settings.providers.feedback.connection.timeout",
              "连接测试超时。请确认 API Base URL 可访问后重试。",
            ),
          }),
        });
      }
    } catch (error) {
      setConnectionStatus({
        tone: "error",
        message: formatProviderConnectionError(error, {
          fallback: t(
            "settings.providers.setting.feedback.connection.errorDefault",
          ),
          timeout: t(
            "settings.providers.feedback.connection.timeout",
            "连接测试超时。请确认 API Base URL 可访问后重试。",
          ),
        }),
      });
    } finally {
      setTestingConnection(false);
    }
  }, [
    canUseDraftApiKey,
    hasApiKey,
    modelList.length,
    onTestConnection,
    persistDraftApiHost,
    persistDraftApiKey,
    primaryModel,
    providerApiKeyRequired,
    provider.id,
    shouldRunChatReadinessTest,
    t,
  ]);

  return (
    <div
      className={cn("flex h-full min-h-0 flex-col bg-slate-50", className)}
      data-testid="provider-setting"
      data-provider-id={provider.id}
    >
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 lg:px-6">
        <section
          className="mx-auto w-full max-w-[820px] rounded-[30px] border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-950/5 lg:p-6"
          data-testid="provider-simple-card"
        >
          <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex min-w-0 items-center gap-3">
              <ProviderIcon
                providerType={provider.id}
                fallbackText={provider.name}
                size={48}
                className="flex-shrink-0"
                data-testid="provider-icon"
              />
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3
                    className="min-w-0 truncate text-xl font-semibold tracking-tight text-slate-900"
                    data-testid="provider-name"
                  >
                    {provider.name}
                  </h3>
                  <Badge
                    variant="outline"
                    className="border-slate-200 bg-slate-50 text-slate-600"
                  >
                    {getProviderTypeLabel(provider.type)}
                  </Badge>
                  {showExplicitPromptCacheBadge ? (
                    <Badge
                      variant="outline"
                      className="border-amber-200 bg-amber-50 text-amber-700"
                      data-testid="provider-prompt-cache-badge"
                    >
                      {t(
                        "settings.providers.setting.body.badge.explicitCache",
                        "显式缓存",
                      )}
                    </Badge>
                  ) : null}
                </div>
                <p className="mt-1 break-all text-sm text-slate-500">
                  {providerHostLabel}
                </p>
              </div>
            </div>

            {accessHelp.url || canDeleteProvider ? (
              <div className="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">
                {accessHelp.url ? (
                  <a
                    href={accessHelp.url}
                    rel={accessHelpRel}
                    onAuxClick={(event) => {
                      interceptHttpExternalLinkClick(event, accessHelp.url);
                    }}
                    onClick={(event) => {
                      interceptHttpExternalLinkClick(event, accessHelp.url);
                    }}
                    className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50 hover:text-slate-900"
                    data-testid="provider-api-key-link"
                  >
                    {t(
                      "settings.providers.setting.body.action.getApiKey",
                      "去获取 API 密钥",
                    )}
                    <ExternalLink className="h-3.5 w-3.5" />
                  </a>
                ) : null}
                {canDeleteProvider ? (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-9 rounded-full border-rose-200 bg-white px-3 text-rose-600 hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700"
                    onClick={() => {
                      void handleDeleteProvider();
                    }}
                    disabled={loading || deletingProvider}
                    data-testid="provider-delete-button"
                  >
                    {deletingProvider ? (
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Trash2 className="mr-1.5 h-3.5 w-3.5" />
                    )}
                    {deletingProvider
                      ? t(
                          "settings.providers.setting.body.action.deleting",
                          "删除中...",
                        )
                      : t(
                          "settings.providers.setting.body.action.delete",
                          "删除配置",
                        )}
                  </Button>
                ) : null}
              </div>
            ) : null}
          </header>

          <div className="mt-6 space-y-6">
            {showProviderFocusBanner ? (
              <div
                className={cn(
                  "rounded-[18px] border px-4 py-3 text-sm",
                  needsApiKeyFromFocus || focusModelMissing
                    ? "border-amber-200 bg-amber-50 text-amber-900"
                    : "border-sky-200 bg-sky-50 text-sky-900",
                )}
                data-testid="provider-focus-banner"
              >
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div className="min-w-0">
                    <div className="font-semibold">
                      {needsApiKeyFromFocus
                        ? t(
                            "settings.providers.setting.focus.needApiKey.title",
                            "补齐 API 密钥后可继续运行",
                          )
                        : focusModelMissing
                          ? t(
                              "settings.providers.setting.focus.needModel.title",
                              "补齐模型后可继续运行",
                            )
                          : t(
                              "settings.providers.setting.focus.title",
                              "运行诊断定位到这里",
                            )}
                    </div>
                    <p className="mt-1 leading-5">
                      {needsApiKeyFromFocus
                        ? t(
                            "settings.providers.setting.focus.needApiKey.description",
                            "请在下方填写并保存可用的 API 密钥，然后试跑当前模型。",
                          )
                        : focusModelMissing
                          ? t(
                              "settings.providers.setting.focus.needModel.description",
                              {
                                model: focusedModelId,
                              },
                            )
                          : t(
                              "settings.providers.setting.focus.description",
                              "请检查该服务商的密钥、模型优先级和连接状态。",
                            )}
                    </p>
                    {focus?.reasonCode ? (
                      <p className="mt-1 break-all text-xs opacity-70">
                        {t("settings.providers.setting.focus.reason", {
                          reason: focus.reasonCode,
                          defaultValue: "诊断原因：{{reason}}",
                        })}
                      </p>
                    ) : null}
                  </div>
                  {focusModelMissing ? (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-9 shrink-0 rounded-full border-amber-300 bg-white px-3 text-amber-900 hover:bg-amber-100"
                      onClick={() => {
                        void addModels([focusedModelId]);
                      }}
                      disabled={loading || !focusedModelId}
                      data-testid="provider-focus-add-model-button"
                    >
                      <Plus className="mr-1 h-3.5 w-3.5" />
                      {t(
                        "settings.providers.setting.focus.action.addModel",
                        "加入模型优先级",
                      )}
                    </Button>
                  ) : null}
                </div>
              </div>
            ) : null}

            {providerActionStatus ? (
              <div
                className={cn(
                  "flex items-start gap-2 rounded-[16px] border px-3 py-2 text-sm",
                  buildStatusClass(providerActionStatus.tone),
                )}
                data-testid="provider-action-status"
              >
                {getStatusIcon(providerActionStatus.tone)}
                <span className="leading-5">
                  {providerActionStatus.message}
                </span>
              </div>
            ) : null}

            <div className="space-y-2" data-testid="api-host-section">
              <div className="flex items-center justify-between gap-3">
                <Label
                  htmlFor="provider-api-host"
                  className="text-sm text-slate-600"
                >
                  {t(
                    "settings.providers.setting.body.apiHost.label",
                    "API Host",
                  )}
                </Label>
                {hasPendingApiHost ? (
                  <span className="text-xs text-amber-600">
                    {t(
                      "settings.providers.setting.body.apiHost.pending",
                      "未保存",
                    )}
                  </span>
                ) : null}
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <Input
                  id="provider-api-host"
                  type="url"
                  value={apiHostDraft}
                  onChange={(event) => {
                    setApiHostDirty(true);
                    setApiHostDraft(event.target.value);
                    setApiHostStatus(null);
                    setConnectionStatus(null);
                    setModelFetchStatus(null);
                  }}
                  placeholder={t(
                    "settings.providers.setting.body.apiHost.placeholder",
                    "输入 API Host",
                  )}
                  className="h-12 min-w-0 flex-1 rounded-[18px] border-slate-200 bg-white px-4"
                  autoCapitalize="none"
                  autoCorrect="off"
                  spellCheck={false}
                  disabled={
                    loading ||
                    savingApiHost ||
                    savingApiKey ||
                    testingConnection ||
                    fetchingModels
                  }
                  data-testid="provider-api-host-input"
                />
                <Button
                  type="button"
                  variant="outline"
                  className="h-12 rounded-[18px] border-slate-200 bg-white px-4 text-slate-700 hover:bg-slate-50"
                  onClick={() => {
                    void handleSaveApiHost();
                  }}
                  disabled={!canSaveApiHost}
                  data-testid="provider-api-host-save-button"
                >
                  {savingApiHost ? (
                    <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                  ) : (
                    <Save className="mr-1.5 h-4 w-4" />
                  )}
                  {savingApiHost
                    ? t(
                        "settings.providers.setting.body.apiHost.action.saving",
                        "保存中...",
                      )
                    : t(
                        "settings.providers.setting.body.apiHost.action.save",
                        "保存 Host",
                      )}
                </Button>
              </div>
              {apiHostStatus ? (
                <div
                  className={cn(
                    "flex items-start gap-2 rounded-[16px] border px-3 py-2 text-sm",
                    buildStatusClass(apiHostStatus.tone),
                  )}
                  data-testid="api-host-status"
                >
                  {getStatusIcon(apiHostStatus.tone)}
                  <span className="leading-5">{apiHostStatus.message}</span>
                </div>
              ) : null}
            </div>

            <div className="space-y-2" data-testid="api-key-section">
              <div className="flex items-center justify-between gap-3">
                <Label
                  htmlFor="provider-api-key"
                  className="text-sm text-slate-600"
                >
                  {providerApiKeyRequired
                    ? t(
                        "settings.providers.setting.body.apiKey.label",
                        "API 密钥",
                      )
                    : t(
                        "settings.providers.setting.body.apiKey.optionalLabel",
                        "API 密钥（可选）",
                      )}
                </Label>
                {hasPendingApiKey ? (
                  <span className="text-xs text-amber-600">
                    {t(
                      "settings.providers.setting.body.apiKey.pending",
                      "未保存",
                    )}
                  </span>
                ) : hasApiKey ? (
                  <span className="text-xs text-emerald-600">
                    {t(
                      "settings.providers.setting.body.apiKey.configured",
                      "已配置",
                    )}
                  </span>
                ) : null}
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <div className="relative min-w-0 flex-1">
                  <Input
                    id="provider-api-key"
                    type={showApiKey ? "text" : "password"}
                    value={apiKeyInputValue}
                    onFocus={() => {
                      if (!apiKeyDirty && apiKeyMask) {
                        setApiKeyDraft("");
                        setApiKeyDirty(true);
                      }
                    }}
                    onChange={(event) => {
                      setApiKeyDirty(true);
                      setApiKeyDraft(event.target.value);
                      setApiKeyStatus(null);
                      setConnectionStatus(null);
                    }}
                    placeholder={
                      providerApiKeyRequired
                        ? t(
                            "settings.providers.setting.body.apiKey.placeholder",
                            "输入 API 密钥",
                          )
                        : t(
                            "settings.providers.setting.body.apiKey.optionalPlaceholder",
                            "本地服务可留空",
                          )
                    }
                    className="h-12 rounded-[18px] border-slate-200 bg-white px-4 pr-11"
                    autoCapitalize="none"
                    autoCorrect="off"
                    spellCheck={false}
                    disabled={
                      loading ||
                      savingApiKey ||
                      testingConnection ||
                      fetchingModels
                    }
                    data-testid="provider-api-key-input"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey((previous) => !previous)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 transition hover:text-slate-700"
                    aria-label={t(
                      "settings.providers.setting.body.apiKey.toggleVisibility",
                      "显示或隐藏 API 密钥",
                    )}
                    data-testid="provider-api-key-eye-button"
                  >
                    <Eye className="h-4 w-4" />
                  </button>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  className="h-12 rounded-[18px] border-slate-200 bg-white px-4 text-slate-700 hover:bg-slate-50"
                  onClick={() => {
                    void handleSaveApiKey();
                  }}
                  disabled={!canSaveApiKey}
                  data-testid="provider-api-key-save-button"
                >
                  {savingApiKey ? (
                    <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                  ) : (
                    <Save className="mr-1.5 h-4 w-4" />
                  )}
                  {savingApiKey
                    ? t(
                        "settings.providers.setting.body.apiKey.action.saving",
                        "保存中...",
                      )
                    : t(
                        "settings.providers.setting.body.apiKey.action.save",
                        "保存密钥",
                      )}
                </Button>
              </div>
              {apiKeyStatus ? (
                <div
                  className={cn(
                    "flex items-start gap-2 rounded-[16px] border px-3 py-2 text-sm",
                    buildStatusClass(apiKeyStatus.tone),
                  )}
                  data-testid="api-key-status"
                >
                  {getStatusIcon(apiKeyStatus.tone)}
                  <span className="leading-5">{apiKeyStatus.message}</span>
                </div>
              ) : null}
              {accessHelp.keylessHint ? (
                <p className="text-xs leading-5 text-slate-500">
                  {accessHelp.keylessHint}
                </p>
              ) : null}
            </div>

            <div className="space-y-3" data-testid="model-priority-section">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <Label className="text-sm text-slate-600">
                    {t(
                      "settings.providers.setting.body.models.title",
                      "模型优先级",
                    )}
                  </Label>
                  <p className="mt-1 text-xs text-slate-500">
                    {t(
                      "settings.providers.setting.body.models.description",
                      "只使用接口返回或你手动添加的模型，不再显示本地兜底模型。",
                    )}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-9 rounded-full border-slate-200 bg-white"
                  onClick={() => {
                    void handleFetchModelsFromApi();
                  }}
                  disabled={loading || fetchingModels || !canReadModelsFromApi}
                  data-testid="fetch-models-button"
                >
                  {fetchingModels ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  {t(
                    "settings.providers.setting.body.models.action.fetch",
                    "从接口获取",
                  )}
                </Button>
              </div>

              {!modelAutoFetchCapability.supported &&
              modelAutoFetchCapability.unsupportedReason ? (
                <p className="rounded-[16px] border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 text-slate-600">
                  {modelAutoFetchCapability.unsupportedReason}
                </p>
              ) : null}

              {modelFetchStatus ? (
                <div
                  className={cn(
                    "flex items-start gap-2 rounded-[16px] border px-3 py-2 text-sm",
                    buildStatusClass(modelFetchStatus.tone),
                  )}
                  data-testid="model-fetch-status"
                >
                  {getStatusIcon(modelFetchStatus.tone)}
                  <span className="leading-5">{modelFetchStatus.message}</span>
                </div>
              ) : null}

              {availableApiModels.length > 0 ? (
                <div
                  className="rounded-[18px] border border-slate-200/80 bg-slate-50 p-3"
                  data-testid="api-model-suggestions"
                >
                  <div className="mb-2 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                    <div className="text-xs font-medium text-slate-500">
                      {t(
                        "settings.providers.setting.body.models.apiSuggestions",
                        {
                          visible: suggestedApiModels.length,
                          total: availableApiModels.length,
                        },
                      )}
                    </div>
                    <Input
                      value={apiModelQuery}
                      onChange={(event) => setApiModelQuery(event.target.value)}
                      placeholder={t(
                        "settings.providers.setting.body.models.filterPlaceholder",
                        "筛选接口模型",
                      )}
                      className="h-8 rounded-full border-slate-200 bg-white px-3 text-xs normal-case sm:w-[220px]"
                      autoCapitalize="none"
                      autoCorrect="off"
                      spellCheck={false}
                      data-testid="api-model-filter-input"
                    />
                  </div>
                  <div
                    className="max-h-56 overflow-y-auto pr-1"
                    data-testid="api-model-suggestion-list"
                  >
                    {suggestedApiModels.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {suggestedApiModels.map((modelId) => (
                          <button
                            key={modelId}
                            type="button"
                            onClick={() => {
                              void addModels([modelId]);
                            }}
                            className="max-w-full rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-700 transition hover:border-slate-300 hover:text-slate-900"
                            data-testid="api-model-suggestion"
                          >
                            <span className="block max-w-[220px] truncate normal-case">
                              {modelId}
                            </span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="rounded-[14px] border border-dashed border-slate-200 bg-white px-3 py-4 text-center text-xs text-slate-500">
                        {t(
                          "settings.providers.setting.body.models.noApiMatches",
                          "没有匹配的接口模型。",
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ) : null}

              <div
                className="rounded-[22px] bg-slate-100 p-3"
                data-testid="model-priority-list"
              >
                <input
                  id="custom-models"
                  type="hidden"
                  value={modelList.join(", ")}
                  readOnly
                />

                {modelList.length > 0 ? (
                  <div className="space-y-2">
                    {modelList.map((modelId, index) => {
                      const imageModelCommand =
                        imageModelCommandByModel.get(modelId);
                      const canBindImageCommand =
                        isLikelyImageModelCommandModel(modelId);

                      return (
                        <div
                          key={modelId}
                          className="flex items-center gap-3 rounded-[16px] bg-white px-3 py-2 text-sm text-slate-800"
                          data-testid="model-priority-item"
                        >
                          <span className="text-slate-400">::</span>
                          {index === 0 ? (
                            <Badge className="border border-amber-200 bg-amber-50 px-2 py-0 text-[11px] text-amber-700 hover:bg-amber-50">
                              {t(
                                "settings.providers.setting.body.models.primaryBadge",
                                "主模型",
                              )}
                            </Badge>
                          ) : null}
                          <span className="min-w-0 flex-1 truncate normal-case">
                            {modelId}
                          </span>
                          {imageModelCommand ? (
                            <Badge
                              className="border border-emerald-200 bg-emerald-50 px-2 py-0 text-[11px] text-emerald-700 hover:bg-emerald-50"
                              data-testid="image-command-bound-badge"
                            >
                              {t(
                                "settings.providers.setting.body.models.imageCommand.bound",
                                {
                                  trigger:
                                    readCommandTriggerLabel(imageModelCommand),
                                },
                              )}
                            </Badge>
                          ) : canBindImageCommand ? (
                            <button
                              type="button"
                              onClick={() => {
                                handleOpenImageCommandDraft(modelId);
                              }}
                              className="rounded-full border border-sky-200 bg-sky-50 px-2.5 py-1 text-xs font-medium text-sky-700 transition hover:border-sky-300 hover:bg-sky-100"
                              data-testid="create-image-command-button"
                            >
                              <Sparkles className="mr-1 inline h-3 w-3" />
                              {t(
                                "settings.providers.setting.body.models.action.createImageCommand",
                                "创建 @命令",
                              )}
                            </button>
                          ) : null}
                          {index > 0 ? (
                            <button
                              type="button"
                              onClick={() => {
                                void handleSetMainModel(modelId);
                              }}
                              className="text-xs font-medium text-slate-500 hover:text-slate-900"
                            >
                              <Star className="mr-1 inline h-3 w-3" />
                              {t(
                                "settings.providers.setting.body.models.action.setPrimary",
                                "设为主模型",
                              )}
                            </button>
                          ) : null}
                          <button
                            type="button"
                            onClick={() => {
                              void handleRemoveModel(modelId);
                            }}
                            className="text-slate-400 hover:text-rose-600"
                            aria-label={t(
                              "settings.providers.setting.body.models.action.removeAria",
                              {
                                model: modelId,
                              },
                            )}
                          >
                            <X className="h-4 w-4" />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="rounded-[18px] border border-dashed border-slate-200 bg-white px-4 py-5 text-sm text-slate-500">
                    {t(
                      "settings.providers.setting.body.models.empty",
                      "暂无模型。请从接口获取后选择，或手动添加模型 ID。",
                    )}
                  </div>
                )}

                {imageCommandStatus ? (
                  <div
                    className={cn(
                      "mt-3 flex items-start gap-2 rounded-[16px] border px-3 py-2 text-sm",
                      buildStatusClass(imageCommandStatus.tone),
                    )}
                    data-testid="image-command-status"
                  >
                    {getStatusIcon(imageCommandStatus.tone)}
                    <span className="leading-5">
                      {imageCommandStatus.message}
                    </span>
                  </div>
                ) : null}

                {imageCommandDraft ? (
                  <div
                    className="mt-3 rounded-[18px] border border-sky-200 bg-sky-50 p-3"
                    data-testid="image-command-binding-panel"
                  >
                    <div className="mb-2">
                      <div className="text-sm font-medium text-slate-800">
                        {t(
                          "settings.providers.setting.body.models.imageCommand.title",
                          "创建图片 @命令",
                        )}
                      </div>
                      <p className="mt-1 text-xs leading-5 text-slate-600">
                        {t(
                          "settings.providers.setting.body.models.imageCommand.description",
                          {
                            model: imageCommandDraft.modelId,
                          },
                        )}
                      </p>
                    </div>
                    <div className="flex flex-col gap-2 sm:flex-row">
                      <Input
                        value={imageCommandDraft.trigger}
                        onChange={(event) =>
                          setImageCommandDraft((draft) =>
                            draft
                              ? { ...draft, trigger: event.target.value }
                              : draft,
                          )
                        }
                        className="h-10 rounded-[14px] border-sky-200 bg-white px-3 normal-case"
                        autoCapitalize="none"
                        autoCorrect="off"
                        spellCheck={false}
                        data-testid="image-command-trigger-input"
                      />
                      <Button
                        type="button"
                        className="h-10 rounded-[14px] bg-slate-950 px-4 text-white hover:bg-slate-800"
                        onClick={handleSaveImageCommandDraft}
                        disabled={!imageCommandDraft.trigger.trim()}
                        data-testid="image-command-save-button"
                      >
                        {t(
                          "settings.providers.setting.body.models.imageCommand.action.save",
                          "保存命令",
                        )}
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        className="h-10 rounded-[14px] border-slate-200 bg-white px-4"
                        onClick={() => {
                          setImageCommandDraft(null);
                        }}
                        data-testid="image-command-cancel-button"
                      >
                        {t(
                          "settings.providers.setting.body.models.imageCommand.action.cancel",
                          "取消",
                        )}
                      </Button>
                    </div>
                  </div>
                ) : null}

                <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                  <Input
                    value={modelDraft}
                    onChange={(event) => setModelDraft(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === ",") {
                        event.preventDefault();
                        void handleAddModelDraft();
                      }
                    }}
                    placeholder={t(
                      "settings.providers.setting.body.models.draftPlaceholder",
                      "输入模型 ID，按 Enter 添加",
                    )}
                    className="h-11 rounded-[16px] border-slate-200 bg-white px-4 normal-case"
                    autoCapitalize="none"
                    autoCorrect="off"
                    spellCheck={false}
                    disabled={loading}
                    data-testid="model-draft-input"
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    className="h-11 rounded-[16px] px-4"
                    onClick={() => {
                      void handleAddModelDraft();
                    }}
                    disabled={loading || !modelDraft.trim()}
                    data-testid="model-draft-add-button"
                  >
                    <Plus className="mr-1 h-4 w-4" />
                    {t(
                      "settings.providers.setting.body.models.action.add",
                      "添加模型",
                    )}
                  </Button>
                </div>
              </div>
            </div>

            <div className="space-y-3" data-testid="connection-test-section">
              <Button
                type="button"
                onClick={() => {
                  void handleTestConnection();
                }}
                disabled={!canTestConnection}
                className="h-12 w-full rounded-full bg-slate-950 text-sm font-semibold text-white hover:bg-slate-800"
                data-testid="provider-test-connection-button"
              >
                {testingConnection ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                )}
                {testingConnection
                  ? t(
                      "settings.providers.setting.body.connection.testing",
                      "测试中...",
                    )
                  : shouldRunChatReadinessTest
                    ? t(
                        "settings.providers.setting.body.connection.testChat",
                        "试跑当前模型",
                      )
                    : t(
                        "settings.providers.setting.body.connection.test",
                        "测试连接",
                      )}
              </Button>

              {!canTestConnection && !testingConnection ? (
                <p className="text-center text-xs text-slate-500">
                  {modelList.length === 0
                    ? t(
                        "settings.providers.setting.body.connection.needModel",
                        "先添加一个模型，再测试连接。",
                      )
                    : providerApiKeyRequired && !hasApiKey && !canUseDraftApiKey
                      ? t(
                          "settings.providers.setting.body.connection.needApiKey",
                          "先填写 API 密钥，再测试连接。",
                        )
                      : t(
                          "settings.providers.setting.body.connection.unavailable",
                          "当前暂不可测试。",
                        )}
                </p>
              ) : null}

              {connectionStatus ? (
                <div
                  className={cn(
                    "flex items-start gap-2 rounded-[16px] border px-3 py-2 text-sm",
                    buildStatusClass(connectionStatus.tone),
                  )}
                  data-testid="connection-status"
                >
                  {getStatusIcon(connectionStatus.tone)}
                  <span className="leading-5">{connectionStatus.message}</span>
                </div>
              ) : null}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

// ============================================================================
// 辅助函数（用于测试）
// ============================================================================

/**
 * 从 Provider 数据中提取简洁设置页所需的信息。
 */
export function extractProviderSettingInfo(
  provider: ProviderWithKeysDisplay | null,
): {
  hasProvider: boolean;
  hasIcon: boolean;
  hasName: boolean;
  hasApiKeyInput: boolean;
  hasModelPriority: boolean;
  hasApiModelFetch: boolean;
  hasConnectionTest: boolean;
} {
  if (!provider) {
    return {
      hasProvider: false,
      hasIcon: false,
      hasName: false,
      hasApiKeyInput: false,
      hasModelPriority: false,
      hasApiModelFetch: false,
      hasConnectionTest: false,
    };
  }

  return {
    hasProvider: true,
    hasIcon: typeof provider.id === "string" && provider.id.length > 0,
    hasName: typeof provider.name === "string" && provider.name.length > 0,
    hasApiKeyInput: true,
    hasModelPriority: true,
    hasApiModelFetch: true,
    hasConnectionTest: true,
  };
}
