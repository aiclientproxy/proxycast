import { safeInvoke } from "@/lib/dev-bridge";
import {
  METHOD_CONFIG_BATCH_WRITE,
  METHOD_CONFIG_READ,
  type ConfigBatchWriteParams,
  type ConfigReadResponse,
  type ConfigWriteResponse,
} from "@limecloud/app-server-client";
import {
  CURRENT_SIDEBAR_NAV_SCHEMA_VERSION,
  type Config,
  type EnvironmentPreview,
} from "./appConfigTypes";
import { AppServerClient } from "./appServer";
import { assertNotDiagnosticFacade } from "./diagnosticFacade";

const APP_CONFIG_CHANGE_STAMP_KEY = "lime.app-config.changed-at";
const APP_CONFIG_CHANGED_EVENT = "lime:app-config-changed";

let configCache: Config | null = null;
let configLoadingPromise: Promise<Config> | null = null;
let configCacheStamp: string | null = null;
let configCacheVersion: string | null = null;
let configMutationTail: Promise<void> = Promise.resolve();

export type ConfigUpdater = (current: Config) => Config;

export type {
  ClawTraceConfig,
  ClawTraceLevelConfig,
  Config,
  DesktopCapabilities,
  DesktopCapability,
  CrashReportingConfig,
  ChatAppearanceConfig,
  DeveloperConfig,
  EnvironmentConfig,
  EnvironmentPreview,
  EnvironmentPreviewEntry,
  EnvironmentVariableOverride,
  ImageGenConfig,
  MultiSearchConfig,
  MultiSearchEngineEntryConfig,
  NavigationConfig,
  NativeAgentConfig,
  OrchestratorConfig,
  OrchestratorFeatureConfig,
  QuotaExceededConfig,
  RemoteManagementConfig,
  ResponseCacheConfig,
  RolloutBudgetConfig,
  ServiceModelPreferenceConfig,
  ServiceModelsConfig,
  ShellImportPreview,
  TlsConfig,
  ToolCallingConfig,
  ToolExecutionCommandRiskLevelConfig,
  ToolExecutionCommandRuleConfig,
  ToolExecutionCommandRuleMatchTypeConfig,
  ToolExecutionNetworkRuleConfig,
  ToolExecutionNetworkRuleTargetConfig,
  ToolExecutionOverrideConfig,
  ToolExecutionPolicyConfig,
  ToolExecutionRestrictionProfileConfig,
  ToolExecutionSandboxProfileConfig,
  ToolExecutionWarningPolicyConfig,
  UserProfile,
  WorkspacePreferencesConfig,
  WorkspaceSandboxConfig,
} from "./appConfigTypes";

interface GetConfigOptions {
  forceRefresh?: boolean;
}

function cloneConfig(config: Config): Config {
  if (typeof structuredClone === "function") {
    return structuredClone(config);
  }
  return JSON.parse(JSON.stringify(config)) as Config;
}

function normalizeConfig(config: Config): Config {
  const nextConfig = cloneConfig(config);
  const navigation = nextConfig.navigation;

  if (navigation) {
    nextConfig.navigation = {
      ...navigation,
      schema_version: CURRENT_SIDEBAR_NAV_SCHEMA_VERSION,
      enabled_items: [],
    };
  }

  return nextConfig;
}

function readAppConfigChangeStamp(): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    return window.localStorage.getItem(APP_CONFIG_CHANGE_STAMP_KEY);
  } catch {
    return null;
  }
}

function markAppConfigChanged(): string | null {
  const nextStamp = String(Date.now());

  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(APP_CONFIG_CHANGE_STAMP_KEY, nextStamp);
    } catch {
      // ignore
    }

    try {
      window.dispatchEvent(new CustomEvent(APP_CONFIG_CHANGED_EVENT));
    } catch {
      // ignore
    }
  }

  return nextStamp;
}

function invalidateConfigCache(): void {
  configCache = null;
  configLoadingPromise = null;
  configCacheStamp = null;
  configCacheVersion = null;
}

function assertNonEmptyString(
  command: string,
  value: unknown,
  label: string,
): asserts value is string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${command} 未返回有效${label}`);
  }
}

function assertConfigShape(value: unknown): asserts value is Config {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("config/read 未返回有效配置");
  }

  const defaultProvider = (value as { default_provider?: unknown })
    .default_provider;
  if (
    typeof defaultProvider !== "string" ||
    defaultProvider.trim().length === 0
  ) {
    throw new Error("config/read 未返回有效配置");
  }
}

function readConfigVersion(response: ConfigReadResponse): string {
  const layers = response.layers;
  if (!Array.isArray(layers) || layers.length !== 1) {
    throw new Error("config/read 未返回唯一 Desktop 用户配置层");
  }
  const version = layers[0]?.version;
  if (typeof version !== "string" || version.length === 0) {
    throw new Error("config/read 未返回有效配置版本");
  }
  return version;
}

function configValuesEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function buildConfigEdits(
  current: Config,
  next: Config,
): ConfigBatchWriteParams["edits"] {
  return Object.entries(next)
    .filter(([key, value]) => {
      return !configValuesEqual(
        (current as unknown as Record<string, unknown>)[key],
        value,
      );
    })
    .map(([key, value]) => ({
      keyPath: key,
      value,
      mergeStrategy: "replace" as const,
    }));
}

function assertConfigWriteResponse(
  value: unknown,
): asserts value is ConfigWriteResponse {
  const response = value as Partial<ConfigWriteResponse> | null;
  if (
    !response ||
    response.status !== "ok" ||
    typeof response.version !== "string" ||
    response.version.length === 0 ||
    typeof response.filePath !== "string" ||
    response.filePath.length === 0
  ) {
    throw new Error("config/batchWrite 未返回有效写入结果");
  }
}

export function invalidateAppConfigCache(): void {
  invalidateConfigCache();
}

export function subscribeAppConfigChanged(listener: () => void): () => void {
  if (typeof window === "undefined") {
    return () => undefined;
  }

  const handleCustomChange = () => {
    listener();
  };
  const handleStorage = (event: StorageEvent) => {
    if (event.key === APP_CONFIG_CHANGE_STAMP_KEY) {
      listener();
    }
  };

  window.addEventListener(APP_CONFIG_CHANGED_EVENT, handleCustomChange);
  window.addEventListener("storage", handleStorage);

  return () => {
    window.removeEventListener(APP_CONFIG_CHANGED_EVENT, handleCustomChange);
    window.removeEventListener("storage", handleStorage);
  };
}

export async function getConfig(
  options: GetConfigOptions = {},
): Promise<Config> {
  if (options.forceRefresh) {
    invalidateConfigCache();
  }

  const currentStamp = readAppConfigChangeStamp();
  if (configCache && configCacheStamp !== currentStamp) {
    invalidateConfigCache();
  }

  if (configCache) {
    return cloneConfig(configCache);
  }

  if (!configLoadingPromise) {
    configLoadingPromise = new AppServerClient()
      .request<ConfigReadResponse>(METHOD_CONFIG_READ, {
        includeLayers: true,
      })
      .then((response) => {
        assertConfigShape(response.result.config);
        configCache = normalizeConfig(response.result.config);
        configCacheVersion = readConfigVersion(response.result);
        configCacheStamp = readAppConfigChangeStamp();
        return configCache;
      })
      .finally(() => {
        configLoadingPromise = null;
      });
  }

  return cloneConfig(await configLoadingPromise);
}

export async function saveConfig(config: Config): Promise<void> {
  const normalizedConfig = normalizeConfig(config);
  const currentConfig = configCache
    ? cloneConfig(configCache)
    : await getConfig();
  const expectedVersion = configCacheVersion;
  if (!expectedVersion) {
    invalidateConfigCache();
    throw new Error("config/read 未提供保存所需的配置版本");
  }
  const edits = buildConfigEdits(currentConfig, normalizedConfig);
  if (edits.length === 0) {
    return;
  }

  let result: ConfigWriteResponse;
  try {
    const response = await new AppServerClient().request<ConfigWriteResponse>(
      METHOD_CONFIG_BATCH_WRITE,
      {
        edits,
        expectedVersion,
        reloadUserConfig: true,
      },
    );
    assertConfigWriteResponse(response.result);
    result = response.result;
  } catch (error) {
    invalidateConfigCache();
    throw error;
  }
  configCache = cloneConfig({
    ...currentConfig,
    ...normalizedConfig,
  });
  configCacheVersion = result.version;
  configCacheStamp = markAppConfigChanged();
}

export function updateConfig(updater: ConfigUpdater): Promise<Config> {
  const mutation = configMutationTail.then(async () => {
    const currentConfig = await getConfig();
    await saveConfig(updater(currentConfig));
    return await getConfig();
  });

  configMutationTail = mutation.then(
    () => undefined,
    () => undefined,
  );
  return mutation;
}

export async function getEnvironmentPreview(): Promise<EnvironmentPreview> {
  const result = await safeInvoke<EnvironmentPreview>(
    "get_environment_preview",
  );
  assertNotDiagnosticFacade(
    "get_environment_preview",
    result,
    "真实环境预览 current 通道",
  );
  return result;
}

export async function getDefaultProvider(): Promise<string> {
  const result = await safeInvoke<unknown>("get_default_provider");
  assertNotDiagnosticFacade(
    "get_default_provider",
    result,
    "真实默认 Provider current 通道",
  );
  assertNonEmptyString("get_default_provider", result, "默认 Provider");
  return result;
}
