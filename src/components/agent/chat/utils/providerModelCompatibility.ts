import type { ModelCapabilityProvenance } from "@/lib/types/modelRegistry";

export interface ProviderModelCompatibilityInput {
  providerType: string;
  configuredProviderType?: string | null;
  model: string;
  capabilityProvenance?: ModelCapabilityProvenance;
  enforceExecutableCapability?: boolean;
}

export interface ProviderModelCompatibilityResult {
  model: string;
  changed: boolean;
  reason?: string;
}

export interface ProviderModelCompatibilityIssue {
  code:
    | "local_cli_account_model_unsupported"
    | "runtime_capability_not_authoritative";
  message: string;
  suggestedModel?: string;
}

function normalize(value?: string | null): string {
  return (value || "").trim().toLowerCase();
}

export function getProviderModelCompatibilityIssue({
  providerType,
  configuredProviderType,
  model,
  capabilityProvenance,
  enforceExecutableCapability = false,
}: ProviderModelCompatibilityInput): ProviderModelCompatibilityIssue | null {
  const normalizedProviderType = normalize(providerType);
  const normalizedConfiguredType = normalize(configuredProviderType);
  const normalizedModel = normalize(model);

  const isLocalCliProvider =
    normalizedProviderType === "codex" || normalizedConfiguredType === "codex";

  if (isLocalCliProvider && normalizedModel === "gpt-5.3-codex") {
    return {
      code: "local_cli_account_model_unsupported",
      message: "当前本地 CLI 登录态不支持该模型",
      suggestedModel: "gpt-5.2-codex",
    };
  }

  if (
    enforceExecutableCapability &&
    capabilityProvenance !== "canonical" &&
    capabilityProvenance !== "provider_explicit"
  ) {
    return {
      code: "runtime_capability_not_authoritative",
      message: "模型缺少可执行的权威能力声明",
    };
  }

  return null;
}

export function resolveProviderModelCompatibility({
  providerType,
  configuredProviderType,
  model,
}: ProviderModelCompatibilityInput): ProviderModelCompatibilityResult {
  const issue = getProviderModelCompatibilityIssue({
    providerType,
    configuredProviderType,
    model,
  });
  if (issue?.suggestedModel) {
    return {
      model: issue.suggestedModel,
      changed: true,
      reason: `当前本地 CLI 登录态与 ${model} 兼容性不足，已自动切换到 ${issue.suggestedModel}。`,
    };
  }

  return {
    model,
    changed: false,
  };
}

export function filterProviderModelsByCompatibility(
  input: Pick<
    ProviderModelCompatibilityInput,
    "providerType" | "configuredProviderType"
  >,
  models: string[],
): {
  compatibleModels: string[];
  incompatibleModels: Array<{
    model: string;
    issue: ProviderModelCompatibilityIssue;
  }>;
} {
  const compatibleModels: string[] = [];
  const incompatibleModels: Array<{
    model: string;
    issue: ProviderModelCompatibilityIssue;
  }> = [];

  models.forEach((model) => {
    const issue = getProviderModelCompatibilityIssue({
      ...input,
      model,
    });
    if (issue) {
      incompatibleModels.push({ model, issue });
    } else {
      compatibleModels.push(model);
    }
  });

  return {
    compatibleModels,
    incompatibleModels,
  };
}
