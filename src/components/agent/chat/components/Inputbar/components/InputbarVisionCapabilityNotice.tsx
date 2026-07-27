import React, { useEffect, useMemo } from "react";
import { AlertCircle } from "lucide-react";
import {
  findConfiguredProviderBySelection,
  useConfiguredProviders,
} from "@/hooks/useConfiguredProviders";
import { useProviderModels } from "@/hooks/useProviderModels";
import { resolveVisionModel } from "@/lib/model/visionModelResolver";
import { resolveProviderModelLoadOptions } from "@/lib/model/providerModelLoadOptions";
import {
  buildModelCapabilitySendGateInput,
  evaluateModelInputCapability,
  resolveModelCapabilitySummaryForSelection,
} from "@/lib/model/modelCapabilitySendGate";
import {
  buildModelInputSendPolicy,
  type ModelInputSendPolicy,
} from "@/lib/model/modelInputSendPolicy";

function hasDeclaredProviderModels(
  provider: { models?: string[] } | null | undefined,
): boolean {
  return Boolean(provider?.models?.some((modelId) => modelId.trim()));
}

interface InputbarVisionCapabilityNoticeProps {
  providerType: string;
  model: string;
  hasPendingImages: boolean;
  onPolicyChange?: (policy: ModelInputSendPolicy) => void;
}

export const InputbarVisionCapabilityNotice: React.FC<
  InputbarVisionCapabilityNoticeProps
> = ({ providerType, model, hasPendingImages, onPolicyChange }) => {
  const shouldInspectCapability = hasPendingImages;

  const { providers, loading: providersLoading } = useConfiguredProviders({
    autoLoad: shouldInspectCapability,
  });

  const selectedProvider = useMemo(
    () => findConfiguredProviderBySelection(providers, providerType),
    [providerType, providers],
  );
  const providerModelLoadOptions = useMemo(
    () =>
      resolveProviderModelLoadOptions({
        providerId: selectedProvider?.providerId,
        providerType: selectedProvider?.type,
        apiHost: selectedProvider?.apiHost,
        hasApiKey: selectedProvider?.hasApiKey,
        hasDeclaredModels: hasDeclaredProviderModels(selectedProvider),
      }),
    [selectedProvider],
  );

  const { models, loading: modelsLoading } = useProviderModels(
    selectedProvider,
    {
      returnFullMetadata: true,
      autoLoad: shouldInspectCapability && Boolean(selectedProvider),
      ...providerModelLoadOptions,
    },
  );

  const capabilityInspectionPending = providersLoading || modelsLoading;

  const policy = useMemo<ModelInputSendPolicy>(() => {
    const gateInput = buildModelCapabilitySendGateInput({
      imageCount: shouldInspectCapability ? 1 : 0,
    });
    const summary =
      !capabilityInspectionPending && selectedProvider && model.trim()
        ? resolveModelCapabilitySummaryForSelection({
            models,
            providerType,
            model,
          })
        : null;

    return buildModelInputSendPolicy(
      evaluateModelInputCapability(summary, gateInput),
      {
        failClosedOnUnknownMedia: !capabilityInspectionPending,
      },
    );
  }, [
    capabilityInspectionPending,
    model,
    models,
    providerType,
    selectedProvider,
    shouldInspectCapability,
  ]);

  const warningMessage = useMemo(() => {
    if (
      !shouldInspectCapability ||
      !model.trim() ||
      capabilityInspectionPending ||
      !selectedProvider
    ) {
      return null;
    }

    const visionResult = resolveVisionModel({
      currentModelId: model,
      models,
    });

    if (visionResult.reason === "already_vision") {
      return null;
    }

    if (visionResult.reason === "no_vision_model") {
      return "当前 Provider 暂无可用的多模态模型，请切换到支持多模态的 Provider 或模型后再发送图片";
    }

    const suggestedModel = visionResult.targetModelId.trim();
    return suggestedModel
      ? `当前模型 ${model} 不支持多模态图片理解，建议切换到 ${suggestedModel} 后再发送图片`
      : `当前模型 ${model} 不支持多模态图片理解，请切换到支持多模态的模型后再发送图片`;
  }, [
    capabilityInspectionPending,
    model,
    models,
    selectedProvider,
    shouldInspectCapability,
  ]);

  useEffect(() => {
    onPolicyChange?.(policy);
  }, [onPolicyChange, policy]);

  if (!warningMessage) {
    return null;
  }

  return (
    <div
      data-testid="inputbar-vision-warning"
      data-policy={policy.status}
      className="mx-3 mb-2 flex items-start gap-2 rounded-2xl border border-amber-200/80 bg-amber-50/90 px-3 py-2 text-[11px] leading-5 text-amber-800"
    >
      <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
      <span>{warningMessage}</span>
    </div>
  );
};
