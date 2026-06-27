import type { ProviderType } from "@/lib/types/provider";
const LOCAL_OPENAI_LIKE_PROVIDER_IDS = new Set([
  "ollama",
  "lmstudio",
  "gpustack",
  "ovms",
]);

const PUBLIC_MODEL_CATALOG_PROVIDER_IDS = new Set(["nearai"]);

function normalize(value?: string | null): string {
  return (value || "").trim().toLowerCase();
}

function isLikelyLocalHost(apiHost?: string | null): boolean {
  const host = normalize(apiHost);
  if (!host) {
    return false;
  }

  return (
    host.includes("://localhost") ||
    host.includes("://127.0.0.1") ||
    host.includes("://0.0.0.0") ||
    host.includes("://host.docker.internal")
  );
}

function isLikelyFalProvider(input: {
  providerId?: string | null;
  providerType?: string | null;
  apiHost?: string | null;
}): boolean {
  const providerId = normalize(input.providerId);
  const providerType = normalize(input.providerType);
  const apiHost = normalize(input.apiHost);

  return (
    providerType === "fal" ||
    providerId === "fal" ||
    providerId.startsWith("fal-") ||
    providerId.includes("fal.ai") ||
    apiHost.includes("fal.run") ||
    apiHost.includes("queue.fal.run")
  );
}

function hasPublicModelCatalog(input: {
  providerId?: string | null;
  apiHost?: string | null;
}): boolean {
  const providerId = normalize(input.providerId);
  const apiHost = normalize(input.apiHost);

  return (
    PUBLIC_MODEL_CATALOG_PROVIDER_IDS.has(providerId) ||
    apiHost.includes("cloud-api.near.ai")
  );
}

interface ProviderModelAutoFetchCapability {
  supported: boolean;
  requiresApiKey: boolean;
  requiresLiveModelTruth: boolean;
  unsupportedReason?: string;
}

export function getProviderModelAutoFetchCapability(input: {
  providerId?: string | null;
  providerType?: string | null;
  apiHost?: string | null;
}): ProviderModelAutoFetchCapability {
  const providerId = normalize(input.providerId);
  const providerType = normalize(input.providerType) as ProviderType | "";
  const localHost = isLikelyLocalHost(input.apiHost);

  if (isLikelyFalProvider(input)) {
    return {
      supported: true,
      requiresApiKey: false,
      requiresLiveModelTruth: false,
    };
  }

  switch (providerType) {
    case "openai":
    case "openai-response":
    case "codex":
    case "new-api":
    case "gateway":
      return {
        supported: true,
        requiresApiKey:
          !LOCAL_OPENAI_LIKE_PROVIDER_IDS.has(providerId) &&
          !hasPublicModelCatalog(input) &&
          !localHost,
        requiresLiveModelTruth: true,
      };
    case "anthropic":
      return {
        supported: true,
        requiresApiKey: true,
        requiresLiveModelTruth: true,
      };
    case "anthropic-compatible":
      return {
        supported: true,
        requiresApiKey: true,
        requiresLiveModelTruth: false,
      };
    case "gemini":
      return {
        supported: true,
        requiresApiKey: true,
        requiresLiveModelTruth: true,
      };
    case "ollama":
      return {
        supported: true,
        requiresApiKey: false,
        requiresLiveModelTruth: true,
      };
    case "azure-openai":
      return {
        supported: false,
        requiresApiKey: true,
        requiresLiveModelTruth: false,
        unsupportedReason:
          "Azure OpenAI 的模型枚举仍需单独适配资源端点与 API Version，当前不展示自动获取入口。",
      };
    case "vertexai":
      return {
        supported: false,
        requiresApiKey: false,
        requiresLiveModelTruth: false,
        unsupportedReason:
          "Vertex AI 需要单独的云端认证与项目上下文，当前不展示自动获取入口。",
      };
    case "aws-bedrock":
      return {
        supported: false,
        requiresApiKey: false,
        requiresLiveModelTruth: false,
        unsupportedReason:
          "AWS Bedrock 需要专门的云凭证签名流程，当前不展示自动获取入口。",
      };
    default:
      return {
        supported: false,
        requiresApiKey: true,
        requiresLiveModelTruth: false,
        unsupportedReason: "当前协议暂不支持自动获取最新模型。",
      };
  }
}
