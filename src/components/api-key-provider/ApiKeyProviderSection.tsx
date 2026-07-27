/**
 * @file ApiKeyProviderSection 组件
 * @description API Key Provider 管理区域，实现左右分栏布局
 * @module components/api-key-provider/ApiKeyProviderSection
 *
 * **Feature: provider-ui-refactor**
 * **Validates: Requirements 1.1, 1.3, 1.4, 6.3, 6.4, 9.4, 9.5**
 */

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/utils";
import { useApiKeyProvider } from "@/hooks/useApiKeyProvider";
import {
  apiKeyProviderApi,
  providerModelIds,
  type ProviderModelConfig,
  type UpdateProviderRequest,
} from "@/lib/api/apiKeyProvider";
import { ProviderSetting } from "./ProviderSetting";
import { ImportExportDialog } from "./ImportExportDialog";
import type {
  ConnectionTestOptions,
  ConnectionTestResult,
} from "./connectionTestTypes";
import { resolveProviderTestModel } from "./ApiKeyProviderSection.helpers";
import { ModelAddPanel } from "./ModelAddPanel";
import { ModelProviderList } from "./ModelProviderList";
import {
  buildApiKeyProviderSectionViewModel,
  planDeleteProviderConfig,
  planEnabledModelSelection,
} from "./ApiKeyProviderSectionViewModel";
import type { ProviderSettingsFocusContext } from "@/types/page";

// ============================================================================
// 类型定义
// ============================================================================

export interface ApiKeyProviderSectionProps {
  /** 额外的 CSS 类名 */
  className?: string;
  /** 设置页深链焦点 */
  initialFocus?: ProviderSettingsFocusContext | null;
  /** 是否展示 OEM Lime Hub 未登录提示入口 */
  exposeOemLoginPrompt?: boolean;
  /** 触发 OEM Lime Hub 登录 */
  onOemLogin?: () => void | Promise<void>;
}

export interface ApiKeyProviderSectionRef {
  /** 刷新 Provider 列表 */
  refresh: () => Promise<void>;
}

function buildProviderFocusKey(
  focus: ProviderSettingsFocusContext | null | undefined,
): string | null {
  if (!focus) {
    return null;
  }

  return [
    focus.providerId ?? "",
    focus.modelId ?? "",
    focus.reasonCode ?? "",
    focus.recoveryAction ?? "",
    focus.requestKey ?? 0,
  ].join(":");
}

function resolveProviderFocusTargetId(
  providers: Array<{ id: string; models?: ProviderModelConfig[] | null }>,
  focus: ProviderSettingsFocusContext | null | undefined,
): string | null {
  if (!focus) {
    return null;
  }

  if (focus.providerId) {
    return providers.some((provider) => provider.id === focus.providerId)
      ? focus.providerId
      : null;
  }

  const normalizedModelId = focus.modelId?.trim().toLowerCase();
  if (!normalizedModelId) {
    return null;
  }

  return (
    providers.find((provider) =>
      provider.models?.some(
        (model) => model.id.toLowerCase() === normalizedModelId,
      ),
    )?.id ?? null
  );
}

// ============================================================================
// 组件实现
// ============================================================================

/**
 * API Key Provider 管理区域组件
 *
 * 实现左右分栏布局：
 * - 左侧：Provider 列表（固定宽度 240px）
 * - 右侧：Provider 设置面板（填充剩余空间）
 *
 * 当用户点击左侧列表中的 Provider 时，右侧面板同步显示该 Provider 的配置。
 *
 * @example
 * ```tsx
 * <ApiKeyProviderSection ref={apiKeyProviderRef} />
 * ```
 */
export const ApiKeyProviderSection = forwardRef<
  ApiKeyProviderSectionRef,
  ApiKeyProviderSectionProps
>(
  (
    { className, initialFocus, exposeOemLoginPrompt = false, onOemLogin },
    ref,
  ) => {
    const { t } = useTranslation("settings");
    // 使用 Hook 管理状态
    const {
      providers,
      selectedProviderId,
      selectedProvider,
      loading,
      error,
      selectProvider,
      addCustomProvider,
      updateProvider,
      addApiKey,
      deleteApiKey,
      deleteCustomProvider,
      exportConfig,
      importConfig,
      refresh,
    } = useApiKeyProvider({
      allowOemManagedSelection: exposeOemLoginPrompt,
    });

    // 暴露 refresh 方法给父组件
    useImperativeHandle(
      ref,
      () => ({
        refresh,
      }),
      [refresh],
    );

    // 导入导出对话框状态
    const [showImportExportDialog, setShowImportExportDialog] = useState(false);
    const [showAddModelFlow, setShowAddModelFlow] = useState(false);
    const appliedFocusKeyRef = useRef<string | null>(null);
    const focusKey = useMemo(
      () => buildProviderFocusKey(initialFocus),
      [initialFocus],
    );
    const focusedProviderId = useMemo(
      () => resolveProviderFocusTargetId(providers, initialFocus),
      [initialFocus, providers],
    );
    const hasUnresolvedInitialFocus = Boolean(
      initialFocus && providers.length > 0 && !focusedProviderId,
    );
    const shouldForwardFocusToSelectedProvider = Boolean(
      focusedProviderId && selectedProvider?.id === focusedProviderId,
    );
    const selectedProviderFocus =
      initialFocus &&
      (shouldForwardFocusToSelectedProvider ||
        (hasUnresolvedInitialFocus && !selectedProvider))
        ? initialFocus
        : null;
    const viewModel = useMemo(
      () =>
        buildApiKeyProviderSectionViewModel({
          providers,
          selectedProvider,
          exposeOemLoginPrompt,
        }),
      [exposeOemLoginPrompt, providers, selectedProvider],
    );

    useEffect(() => {
      if (hasUnresolvedInitialFocus) {
        return;
      }

      const plan = planEnabledModelSelection({
        enabledModelItems: viewModel.enabledModelItems,
        selectedProviderId,
        showAddModelFlow,
      });
      if (plan.type === "select") {
        selectProvider(plan.providerId);
      }
    }, [
      hasUnresolvedInitialFocus,
      selectProvider,
      selectedProviderId,
      showAddModelFlow,
      viewModel.enabledModelItems,
    ]);

    useEffect(() => {
      if (!initialFocus || !focusKey || providers.length === 0) {
        return;
      }

      if (appliedFocusKeyRef.current === focusKey) {
        return;
      }

      appliedFocusKeyRef.current = focusKey;
      setShowAddModelFlow(false);
      selectProvider(focusedProviderId);
    }, [
      focusKey,
      focusedProviderId,
      initialFocus,
      providers.length,
      selectProvider,
    ]);

    const resolveCurrentTestModel = useCallback(() => {
      const input = document.getElementById(
        "custom-models",
      ) as HTMLInputElement | null;
      return resolveProviderTestModel(
        providerModelIds(selectedProvider?.models),
        input?.value ?? "",
      );
    }, [selectedProvider?.models]);

    // ===== 包装回调函数以匹配 ProviderSetting 的类型要求 =====

    const handleUpdateProvider = useCallback(
      async (id: string, request: UpdateProviderRequest): Promise<void> => {
        await updateProvider(id, request);
      },
      [updateProvider],
    );

    const handleAddApiKey = useCallback(
      async (
        providerId: string,
        apiKey: string,
        alias?: string,
        options?: { replaceExisting?: boolean },
      ): Promise<void> => {
        if (options) {
          await addApiKey(providerId, apiKey, alias, options);
          return;
        }

        await addApiKey(providerId, apiKey, alias);
      },
      [addApiKey],
    );

    const handleDeleteProviderConfig = useCallback(
      async (providerId: string): Promise<boolean> => {
        const plan = planDeleteProviderConfig({
          providers,
          providerId,
          selectedProviderId,
        });

        if (plan.type === "missing") {
          return false;
        }

        if (plan.type === "delete-custom") {
          return deleteCustomProvider(providerId);
        }

        for (const apiKeyId of plan.apiKeyIds) {
          await deleteApiKey(apiKeyId);
        }
        await updateProvider(providerId, plan.update);
        if (plan.clearSelection) {
          selectProvider(null);
        }

        return true;
      },
      [
        deleteApiKey,
        deleteCustomProvider,
        providers,
        selectProvider,
        selectedProviderId,
        updateProvider,
      ],
    );

    // ===== 连接测试 =====
    const handleTestConnection = useCallback(
      async (
        providerId: string,
        options?: ConnectionTestOptions,
      ): Promise<ConnectionTestResult> => {
        try {
          const modelName = options?.modelName ?? resolveCurrentTestModel();
          const result = options?.requireChatReady
            ? await apiKeyProviderApi.testChat(
                providerId,
                modelName,
                options.prompt ?? "请用一句话回复：连接测试通过。",
              )
            : await apiKeyProviderApi.testConnection(providerId, modelName);

          return {
            success: result.success,
            latencyMs: result.latency_ms,
            error: result.error,
          };
        } catch (e) {
          return {
            success: false,
            error: e instanceof Error ? e.message : "连接测试失败",
          };
        }
      },
      [resolveCurrentTestModel],
    );

    const handleModelActivated = useCallback(
      (providerId: string) => {
        selectProvider(providerId);
        setShowAddModelFlow(false);
      },
      [selectProvider],
    );

    const handleSelectEnabledModel = useCallback(
      (providerId: string) => {
        setShowAddModelFlow(false);
        selectProvider(providerId);
      },
      [selectProvider],
    );

    return (
      <div
        className={cn(
          "relative flex h-full min-h-0 min-w-0 overflow-hidden rounded-[24px] border border-slate-200/80 bg-white shadow-sm shadow-slate-950/5",
          className,
        )}
        data-testid="api-key-provider-section"
      >
        {error && providers.length === 0 ? (
          <div
            className="flex min-h-0 flex-1 items-center justify-center bg-white px-8 py-10"
            data-testid="provider-load-error"
            role="alert"
            aria-live="polite"
          >
            <div className="flex max-w-md flex-col items-center text-center">
              <div className="mb-4 flex size-11 items-center justify-center rounded-full bg-rose-50 text-rose-700">
                <AlertCircle aria-hidden="true" className="size-5" />
              </div>
              <h3 className="text-base font-semibold text-slate-950">
                {t("settings.providers.loadError.title", {
                  defaultValue: "本地模型设置暂时不可用",
                })}
              </h3>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                {t("settings.providers.loadError.description", {
                  defaultValue:
                    "无法读取本地配置。请检查应用数据目录权限后重试。",
                })}
              </p>
              <button
                type="button"
                className="mt-5 inline-flex h-9 items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 text-sm font-medium text-slate-900 shadow-sm transition-colors hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                data-testid="provider-load-retry"
                disabled={loading}
                onClick={() => {
                  void refresh();
                }}
              >
                <RefreshCw
                  aria-hidden="true"
                  className={cn("size-4", loading && "animate-spin")}
                />
                {t("settings.providers.loadError.action.retry", {
                  defaultValue: "重试",
                })}
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* 左侧：已启用模型列表 */}
            <ModelProviderList
              providers={providers}
              options={{ exposeOemLoginPrompt }}
              selectedProviderId={selectedProviderId}
              onProviderSelect={handleSelectEnabledModel}
              onAddModel={() => setShowAddModelFlow(true)}
              onImportExport={() => setShowImportExportDialog(true)}
              className="flex-shrink-0"
            />

            {/* 右侧：Provider 设置面板 / 添加模型流程 */}
            <div
              className="relative min-h-0 flex-1 overflow-hidden bg-white"
              data-testid="api-key-provider-detail"
            >
              {showAddModelFlow ? (
                <ModelAddPanel
                  providers={providers}
                  onAddProvider={addCustomProvider}
                  onUpdateProvider={updateProvider}
                  onAddApiKey={handleAddApiKey}
                  onActivated={handleModelActivated}
                  onCancel={() => setShowAddModelFlow(false)}
                  className="h-full"
                />
              ) : (
                <ProviderSetting
                  provider={selectedProvider}
                  focus={selectedProviderFocus}
                  onUpdate={handleUpdateProvider}
                  onAddApiKey={handleAddApiKey}
                  onTestConnection={handleTestConnection}
                  onDeleteProvider={handleDeleteProviderConfig}
                  authStatus={
                    viewModel.selectedProviderLoginRequired
                      ? "login_required"
                      : "ready"
                  }
                  onLogin={onOemLogin}
                  loading={loading}
                  className="h-full"
                />
              )}
            </div>
          </>
        )}

        {/* 导入导出对话框 */}
        <ImportExportDialog
          isOpen={showImportExportDialog}
          onClose={() => setShowImportExportDialog(false)}
          onExport={exportConfig}
          onImport={importConfig}
        />
      </div>
    );
  },
);

ApiKeyProviderSection.displayName = "ApiKeyProviderSection";

// ============================================================================
// 辅助函数（用于测试）
// ============================================================================
