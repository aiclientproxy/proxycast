import { resolveOemCloudRuntimeContext } from "./oemCloudRuntime";
import type {
  ModelAliasSource,
  ModelDeploymentSource,
  ModelManagementPlane,
  ModelModality,
  ModelRuntimeFeature,
  ModelTaskFamily,
} from "@/lib/types/modelRegistry";
import type {
  OemCloudPartnerHubAccessMode,
  OemCloudPartnerHubConfigMode,
  OemCloudPartnerHubModelsSource,
} from "./oemCloudControlPlaneTypes";

interface OemCloudEnvelope<T> {
  code?: number;
  message?: string;
  data?: T;
}

export class OemCloudControlPlaneError extends Error {
  status: number;
  code?: number;

  constructor(message: string, options?: { status?: number; code?: number }) {
    super(message);
    this.name = "OemCloudControlPlaneError";
    this.status = options?.status ?? 0;
    this.code = options?.code;
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

export function normalizeText(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  const normalized = value.trim();
  return normalized ? normalized : undefined;
}

export function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => normalizeText(item))
    .filter((item): item is string => Boolean(item));
}

export function normalizeStringMap(value: unknown): Record<string, string> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value)
      .map(([key, item]) => [key.trim(), normalizeText(item)] as const)
      .filter((entry): entry is readonly [string, string] =>
        Boolean(entry[0] && entry[1]),
      ),
  );
}

export function normalizeTypedStringArray<T extends string>(
  value: unknown,
  acceptedValues: Set<T>,
): T[] {
  return Array.from(
    new Set(
      normalizeStringArray(value).filter((item): item is T =>
        acceptedValues.has(item as T),
      ),
    ),
  );
}

export function normalizeBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

export function normalizeNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

export const COMPAT_REFERRAL_BRAND_NAME = "Lime";
export const COMPAT_REFERRAL_DOWNLOAD_URL = "https://limeai.run";

const PARTNER_HUB_ACCESS_MODE_SET = new Set<OemCloudPartnerHubAccessMode>([
  "session",
  "hub_token",
  "api_key",
]);

const PARTNER_HUB_CONFIG_MODE_SET = new Set<OemCloudPartnerHubConfigMode>([
  "managed",
  "hybrid",
  "developer",
]);

const PARTNER_HUB_MODELS_SOURCE_SET = new Set<OemCloudPartnerHubModelsSource>([
  "hub_catalog",
  "manual",
]);

export const MODEL_TASK_FAMILY_SET = new Set<ModelTaskFamily>([
  "chat",
  "reasoning",
  "vision_understanding",
  "image_generation",
  "image_edit",
  "video_generation",
  "speech_to_text",
  "text_to_speech",
  "embedding",
  "rerank",
  "moderation",
]);

export const MODEL_MODALITY_SET = new Set<ModelModality>([
  "text",
  "image",
  "audio",
  "video",
  "file",
  "embedding",
  "json",
]);

export const MODEL_RUNTIME_FEATURE_SET = new Set<ModelRuntimeFeature>([
  "streaming",
  "tool_calling",
  "json_schema",
  "reasoning",
  "prompt_cache",
  "responses_api",
  "chat_completions_api",
  "images_api",
]);

const MODEL_DEPLOYMENT_SOURCE_SET = new Set<ModelDeploymentSource>([
  "local",
  "user_cloud",
  "oem_cloud",
]);

const MODEL_MANAGEMENT_PLANE_SET = new Set<ModelManagementPlane>([
  "local_settings",
  "oem_control_plane",
  "hybrid",
]);

const MODEL_ALIAS_SOURCE_SET = new Set<ModelAliasSource>([
  "official",
  "relay",
  "oem",
  "local",
]);

export function parsePartnerHubAccessMode(
  value: unknown,
  fallback?: OemCloudPartnerHubAccessMode,
): OemCloudPartnerHubAccessMode {
  const accessMode = normalizeText(value) as
    | OemCloudPartnerHubAccessMode
    | undefined;
  if (accessMode && PARTNER_HUB_ACCESS_MODE_SET.has(accessMode)) {
    return accessMode;
  }

  if (fallback) {
    return fallback;
  }

  throw new OemCloudControlPlaneError("服务商接入模式格式非法");
}

export function parsePartnerHubConfigMode(
  value: unknown,
  fallback?: OemCloudPartnerHubConfigMode,
): OemCloudPartnerHubConfigMode {
  const configMode = normalizeText(value) as
    | OemCloudPartnerHubConfigMode
    | undefined;
  if (configMode && PARTNER_HUB_CONFIG_MODE_SET.has(configMode)) {
    return configMode;
  }

  if (fallback) {
    return fallback;
  }

  throw new OemCloudControlPlaneError("服务商配置模式格式非法");
}

export function parsePartnerHubModelsSource(
  value: unknown,
  fallback?: OemCloudPartnerHubModelsSource,
): OemCloudPartnerHubModelsSource {
  const modelsSource = normalizeText(value) as
    | OemCloudPartnerHubModelsSource
    | undefined;
  if (modelsSource && PARTNER_HUB_MODELS_SOURCE_SET.has(modelsSource)) {
    return modelsSource;
  }

  if (fallback) {
    return fallback;
  }

  throw new OemCloudControlPlaneError("服务商模型来源格式非法");
}

export function parseOptionalModelDeploymentSource(
  value: unknown,
): ModelDeploymentSource | undefined {
  const normalized = normalizeText(value) as ModelDeploymentSource | undefined;
  return normalized && MODEL_DEPLOYMENT_SOURCE_SET.has(normalized)
    ? normalized
    : undefined;
}

export function parseOptionalModelManagementPlane(
  value: unknown,
): ModelManagementPlane | undefined {
  const normalized = normalizeText(value) as ModelManagementPlane | undefined;
  return normalized && MODEL_MANAGEMENT_PLANE_SET.has(normalized)
    ? normalized
    : undefined;
}

export function parseOptionalModelAliasSource(
  value: unknown,
): ModelAliasSource | undefined {
  const normalized = normalizeText(value) as ModelAliasSource | undefined;
  return normalized && MODEL_ALIAS_SOURCE_SET.has(normalized)
    ? normalized
    : undefined;
}

function unwrapEnvelope<T>(payload: unknown): {
  data: T | undefined;
  message: string;
  code: number | undefined;
} {
  if (isRecord(payload)) {
    const code = typeof payload.code === "number" ? payload.code : undefined;
    const message = normalizeText(payload.message) ?? "";
    const data = payload.data as T | undefined;
    return {
      data,
      message,
      code,
    };
  }

  return {
    data: payload as T,
    message: "",
    code: undefined,
  };
}

export function ensureRuntime() {
  const runtime = resolveOemCloudRuntimeContext();
  if (!runtime) {
    throw new OemCloudControlPlaneError(
      "缺少品牌云端配置，请先配置域名与租户。",
    );
  }
  return runtime;
}

export async function requestControlPlane<T>(
  path: string,
  options?: {
    method?: "GET" | "POST" | "PUT" | "DELETE";
    payload?: unknown;
    auth?: boolean;
  },
): Promise<T> {
  const runtime = ensureRuntime();
  const method = options?.method ?? "GET";
  const headers: Record<string, string> = {
    Accept: "application/json",
  };

  if (options?.payload !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  if (options?.auth) {
    const token = normalizeText(runtime.sessionToken);
    if (!token) {
      throw new OemCloudControlPlaneError(
        "缺少品牌云端 Session Token，请先完成登录。",
      );
    }
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${runtime.controlPlaneBaseUrl}${path}`, {
    method,
    headers,
    body:
      options?.payload === undefined
        ? undefined
        : JSON.stringify(options.payload),
  });

  let payload: OemCloudEnvelope<T> | unknown = null;
  try {
    payload = (await response.json()) as OemCloudEnvelope<T>;
  } catch {
    payload = null;
  }

  const { data, message, code } = unwrapEnvelope<T>(payload);
  if (!response.ok) {
    throw new OemCloudControlPlaneError(
      message || `请求失败 (${response.status})`,
      {
        status: response.status,
        code,
      },
    );
  }

  if (data === undefined) {
    throw new OemCloudControlPlaneError(message || "服务端返回格式非法", {
      status: response.status,
      code,
    });
  }

  return data;
}
