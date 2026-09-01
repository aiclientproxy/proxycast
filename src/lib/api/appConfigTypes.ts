import type { ChannelsConfig, GatewayConfig } from "./channelsRuntimeTypes";
import type {
  ExperimentalFeatures,
  ToolCallingConfig,
} from "./experimentalFeatureTypes";
import type { MemoryConfig } from "./memoryConfigTypes";

export type { ToolCallingConfig } from "./experimentalFeatureTypes";

export interface AmpModelMapping {
  from: string;
  to: string;
}

export interface AmpConfig {
  upstream_url: string | null;
  model_mappings: AmpModelMapping[];
  restrict_management_to_localhost: boolean;
}

export interface TlsConfig {
  enable: boolean;
  cert_path: string | null;
  key_path: string | null;
}

export interface ResponseCacheConfig {
  enabled: boolean;
  ttl_secs: number;
  max_entries: number;
  max_body_bytes: number;
  cacheable_status_codes: number[];
}

export interface RemoteManagementConfig {
  allow_remote: boolean;
  secret_key: string | null;
  disable_control_panel: boolean;
}

export interface QuotaExceededConfig {
  switch_project: boolean;
  switch_preview_model: boolean;
  cooldown_seconds: number;
}

export interface MultiSearchEngineEntryConfig {
  name: string;
  url_template: string;
  enabled: boolean;
}

export interface MultiSearchConfig {
  priority?: string[];
  engines?: MultiSearchEngineEntryConfig[];
  max_results_per_engine?: number;
  max_total_results?: number;
  timeout_ms?: number;
}

export interface MediaGenerationPreferenceConfig {
  preferredProviderId?: string;
  preferredModelId?: string;
  allowFallback?: boolean;
}

export interface MediaGenerationDefaultsConfig {
  image?: MediaGenerationPreferenceConfig;
  video?: MediaGenerationPreferenceConfig;
  voice?: MediaGenerationPreferenceConfig;
}

export interface ServiceModelPreferenceConfig {
  preferredProviderId?: string;
  preferredModelId?: string;
  enabled?: boolean;
  customPrompt?: string;
}

export interface ServiceModelsConfig {
  responsive_chat?: ServiceModelPreferenceConfig;
  topic?: ServiceModelPreferenceConfig;
  generation_topic?: ServiceModelPreferenceConfig;
  translation?: ServiceModelPreferenceConfig;
  history_compress?: ServiceModelPreferenceConfig;
  agent_meta?: ServiceModelPreferenceConfig;
  input_completion?: ServiceModelPreferenceConfig;
  prompt_rewrite?: ServiceModelPreferenceConfig;
  resource_prompt_rewrite?: ServiceModelPreferenceConfig;
}

export interface WorkspacePreferencesConfig {
  schema_version?: number;
  agent_response_language?: string;
  media_defaults?: MediaGenerationDefaultsConfig;
  service_models?: ServiceModelsConfig;
}

export const CURRENT_SIDEBAR_NAV_SCHEMA_VERSION = 3;

export type NavigationEnabledItemId = never;

export interface NavigationConfig {
  schema_version?: number;
  enabled_items: NavigationEnabledItemId[];
}

export interface ChatAppearanceConfig {
  fontSize?: number;
  transitionMode?: "none" | "fadeIn" | "smooth";
  bubbleStyle?: "default" | "minimal" | "colorful";
  showAvatar?: boolean;
  showTimestamp?: boolean;
  append_selected_text_to_recommendation?: boolean;
}

export type ClawTraceLevelConfig = "summary" | "debug";

export interface ClawTraceConfig {
  alert_enabled?: boolean;
  alert_notification_enabled?: boolean;
  enabled?: boolean;
  level?: ClawTraceLevelConfig;
  sample_rate?: number;
}

export interface DeveloperConfig {
  workspace_harness_enabled?: boolean;
  claw_trace?: ClawTraceConfig;
}

export type ToolExecutionWarningPolicyConfig = "none" | "shell_command_risk";

export type ToolExecutionRestrictionProfileConfig =
  | "none"
  | "workspace_path_required"
  | "workspace_path_optional"
  | "workspace_absolute_path_required"
  | "workspace_shell_command"
  | "analyze_image_input"
  | "safe_https_url_required";

export type ToolExecutionSandboxProfileConfig = "none" | "workspace_command";

export type ToolExecutionCommandRiskLevelConfig = "low" | "medium" | "high";

export type ToolExecutionCommandRuleMatchTypeConfig =
  | "regex"
  | "prefix"
  | "exact";

export type ToolExecutionNetworkRuleTargetConfig = "url" | "host";

export interface ToolExecutionOverrideConfig {
  warning_policy?: ToolExecutionWarningPolicyConfig;
  restriction_profile?: ToolExecutionRestrictionProfileConfig;
  sandbox_profile?: ToolExecutionSandboxProfileConfig;
}

export interface ToolExecutionCommandRuleConfig {
  rule_id: string;
  match_type?: ToolExecutionCommandRuleMatchTypeConfig;
  pattern: string;
  risk_level?: ToolExecutionCommandRiskLevelConfig;
  reason_code?: string;
  reason?: string;
}

export interface ToolExecutionNetworkRuleConfig {
  rule_id: string;
  match_type?: ToolExecutionCommandRuleMatchTypeConfig;
  target?: ToolExecutionNetworkRuleTargetConfig;
  pattern: string;
  risk_level?: ToolExecutionCommandRiskLevelConfig;
  reason_code?: string;
  reason?: string;
}

export interface ToolExecutionPolicyConfig {
  tool_overrides?: Record<string, ToolExecutionOverrideConfig>;
  shell_command_rules?: ToolExecutionCommandRuleConfig[];
  network_rules?: ToolExecutionNetworkRuleConfig[];
}

export interface WorkspaceSandboxConfig {
  enabled: boolean;
  strict: boolean;
  notify_on_fallback: boolean;
}

export interface RolloutBudgetConfig {
  limit_tokens: number;
  reminder_at_remaining_tokens?: number[];
  sampling_token_weight?: number;
  prefill_token_weight?: number;
}

export interface NativeAgentConfig {
  use_default_system_prompt?: boolean;
  custom_system_prompt?: string | null;
  system_prompt_file?: string | null;
  default_model?: string;
  temperature?: number;
  max_tokens?: number;
  workspace_sandbox?: WorkspaceSandboxConfig;
  tool_execution?: ToolExecutionPolicyConfig;
  rollout_budget?: RolloutBudgetConfig;
}

export interface OrchestratorFeatureConfig {
  enabled: boolean;
}

export interface OrchestratorConfig {
  skills: OrchestratorFeatureConfig;
  mcp: OrchestratorFeatureConfig;
}

export interface ImageGenConfig {
  default_service?: "dall_e" | "midjourney" | "stable_diffusion" | "flux";
  default_count?: number;
  default_size?:
    | "256x256"
    | "512x512"
    | "1024x1024"
    | "1792x1024"
    | "1024x1792";
  default_quality?: "standard" | "hd";
  default_style?: "vivid" | "natural";
  enable_enhancement?: boolean;
  auto_download?: boolean;
  image_search_pexels_api_key?: string;
  image_search_pixabay_api_key?: string;
}

export interface UserProfile {
  avatar_url?: string;
  nickname?: string;
  bio?: string;
  email?: string;
  tags?: string[];
}

export interface CrashReportingConfig {
  enabled: boolean;
  dsn?: string | null;
  environment?: string;
  sample_rate?: number;
  send_pii?: boolean;
}

export interface ShellEnvironmentImportConfig {
  enabled: boolean;
  timeout_ms: number;
}

export interface EnvironmentVariableOverride {
  key: string;
  value: string;
  enabled: boolean;
}

export interface EnvironmentConfig {
  shell_import: ShellEnvironmentImportConfig;
  variables: EnvironmentVariableOverride[];
}

export interface EnvironmentPreviewEntry {
  key: string;
  value: string;
  maskedValue: string;
  source: string;
  sourceLabel: string;
  sensitive: boolean;
  overriddenSources: string[];
}

export interface ShellImportPreview {
  enabled: boolean;
  status: string;
  message: string;
  importedCount: number;
  durationMs?: number | null;
}

export interface DesktopCapability {
  status:
    | "ready"
    | "not_granted"
    | "not_configured"
    | "unverified"
    | "unsupported"
    | "unavailable";
  reason: string;
  settingsUrl?: string | null;
}

export interface DesktopCapabilities {
  schemaVersion: number;
  platform: string;
  arch: string;
  applicationId: string;
  appVersion: string;
  packaged: boolean;
  capabilities: {
    accessibility: DesktopCapability;
    inputMonitoring: DesktopCapability;
    applicationGroups: DesktopCapability & { identifiers: string[] };
    nativeModules: DesktopCapability & { identifiers: string[] };
    windowHandles: DesktopCapability;
    windowOrchestration: DesktopCapability;
    accessibilityTree: DesktopCapability;
    displays: DesktopCapability;
    displayWatcher: DesktopCapability;
    hidTopology: DesktopCapability;
    bareModifierMonitor: DesktopCapability;
    screenCapture: DesktopCapability;
    appleEvents: DesktopCapability;
    uiAutomation: DesktopCapability;
    rawInput: DesktopCapability;
    mediaPermissions: {
      microphone: DesktopCapability;
      camera: DesktopCapability;
    };
    securityScopedBookmarks: DesktopCapability;
    deviceKey: DesktopCapability;
    localAuthentication: DesktopCapability;
    resourceManifest: DesktopCapability & {
      identifiers: string[];
      platformKey: string | null;
    };
    sidecars: DesktopCapability & { identifiers: string[] };
    codeMode: DesktopCapability;
    updates: DesktopCapability;
    sandbox: DesktopCapability;
  };
}

export interface EnvironmentPreview {
  desktopCapabilities?: DesktopCapabilities;
  shellImport: ShellImportPreview;
  entries: EnvironmentPreviewEntry[];
}

export interface Config {
  server: {
    host: string;
    port: number;
    api_key: string;
    tls: TlsConfig;
    response_cache: ResponseCacheConfig;
  };
  default_provider: string;
  remote_management: RemoteManagementConfig;
  quota_exceeded: QuotaExceededConfig;
  ampcode: AmpConfig;
  proxy_url: string | null;
  minimize_to_tray: boolean;
  /**
   * UI locale preference. New values use "auto" or BCP 47 locale tags
   * such as "zh-CN" / "en-US"; legacy "zh" / "en" are normalized on read.
   */
  language: string;
  experimental?: ExperimentalFeatures;
  tool_calling?: ToolCallingConfig;
  workspace_preferences?: WorkspacePreferencesConfig;
  navigation?: NavigationConfig;
  chat_appearance?: ChatAppearanceConfig;
  environment?: EnvironmentConfig;
  web_search?: {
    engine: "google" | "xiaohongshu";
    provider?:
      | "tavily"
      | "multi_search_engine"
      | "duckduckgo_instant"
      | "bing_search_api"
      | "google_custom_search";
    provider_priority?: Array<
      | "tavily"
      | "multi_search_engine"
      | "duckduckgo_instant"
      | "bing_search_api"
      | "google_custom_search"
    >;
    tavily_api_key?: string | null;
    bing_search_api_key?: string | null;
    google_search_api_key?: string | null;
    google_search_engine_id?: string | null;
    multi_search?: MultiSearchConfig;
  };
  memory?: MemoryConfig;
  image_gen?: ImageGenConfig;
  user_profile?: UserProfile;
  developer?: DeveloperConfig;
  agent?: NativeAgentConfig;
  orchestrator?: OrchestratorConfig;
  gateway?: GatewayConfig;
  channels?: ChannelsConfig;
  crash_reporting?: CrashReportingConfig;
}
