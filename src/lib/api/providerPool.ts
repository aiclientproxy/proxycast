import { invoke } from "@tauri-apps/api/core";

// Provider types supported by the pool
export type PoolProviderType =
  | "kiro"
  | "gemini"
  | "qwen"
  | "antigravity"
  | "openai"
  | "claude"
  | "codex"
  | "claude_oauth"
  | "iflow"
  | "gemini_api_key";

// Credential data types
export interface KiroOAuthCredential {
  type: "kiro_oauth";
  creds_file_path: string;
}

export interface GeminiOAuthCredential {
  type: "gemini_oauth";
  creds_file_path: string;
  project_id?: string;
}

export interface QwenOAuthCredential {
  type: "qwen_oauth";
  creds_file_path: string;
}

export interface AntigravityOAuthCredential {
  type: "antigravity_oauth";
  creds_file_path: string;
  project_id?: string;
}

export interface OpenAIKeyCredential {
  type: "openai_key";
  api_key: string;
  base_url?: string;
}

export interface ClaudeKeyCredential {
  type: "claude_key";
  api_key: string;
  base_url?: string;
}

export interface GeminiApiKeyCredential {
  type: "gemini_api_key";
  api_key: string;
  base_url?: string;
  excluded_models?: string[];
}

export interface CodexOAuthCredential {
  type: "codex_oauth";
  creds_file_path: string;
}

export interface ClaudeOAuthCredential {
  type: "claude_oauth";
  creds_file_path: string;
}

export interface IFlowOAuthCredential {
  type: "iflow_oauth";
  creds_file_path: string;
}

export interface IFlowCookieCredential {
  type: "iflow_cookie";
  creds_file_path: string;
}

export type CredentialData =
  | KiroOAuthCredential
  | GeminiOAuthCredential
  | QwenOAuthCredential
  | AntigravityOAuthCredential
  | OpenAIKeyCredential
  | ClaudeKeyCredential
  | GeminiApiKeyCredential
  | CodexOAuthCredential
  | ClaudeOAuthCredential
  | IFlowOAuthCredential
  | IFlowCookieCredential;

// Provider credential
export interface ProviderCredential {
  uuid: string;
  provider_type: PoolProviderType;
  credential: CredentialData;
  name?: string;
  is_healthy: boolean;
  is_disabled: boolean;
  check_health: boolean;
  check_model_name?: string;
  not_supported_models: string[];
  usage_count: number;
  error_count: number;
  last_used?: string;
  last_error_time?: string;
  last_error_message?: string;
  last_health_check_time?: string;
  last_health_check_model?: string;
  created_at: string;
  updated_at: string;
}

// Credential source type
export type CredentialSource = "manual" | "imported" | "private";

// Credential display (for UI, hides sensitive data)
export interface CredentialDisplay {
  uuid: string;
  provider_type: PoolProviderType;
  credential_type: string;
  name?: string;
  display_credential: string;
  is_healthy: boolean;
  is_disabled: boolean;
  check_health: boolean;
  check_model_name?: string;
  not_supported_models: string[];
  usage_count: number;
  error_count: number;
  last_used?: string;
  last_error_time?: string;
  last_error_message?: string;
  last_health_check_time?: string;
  last_health_check_model?: string;
  oauth_status?: OAuthStatus;
  token_cache_status?: TokenCacheStatus;
  created_at: string;
  updated_at: string;
  // 凭证来源（手动添加/导入/私有）
  source: CredentialSource;
  // API Key 凭证的 base_url（仅用于 OpenAI/Claude API Key 类型）
  base_url?: string;
  // API Key 凭证的完整 api_key（仅用于 OpenAI/Claude API Key 类型，用于编辑）
  api_key?: string;
  // 凭证级代理 URL（可覆盖全局代理设置）
  proxy_url?: string;
}

// Pool statistics
export interface PoolStats {
  total: number;
  healthy: number;
  unhealthy: number;
  disabled: number;
  total_usage: number;
  total_errors: number;
}

// Provider pool overview
export interface ProviderPoolOverview {
  provider_type: string;
  stats: PoolStats;
  credentials: CredentialDisplay[];
}

// Health check result
export interface HealthCheckResult {
  uuid: string;
  success: boolean;
  model?: string;
  message?: string;
  duration_ms: number;
}

// OAuth status
export interface OAuthStatus {
  has_access_token: boolean;
  has_refresh_token: boolean;
  is_token_valid: boolean;
  expiry_info?: string;
  creds_path: string;
}

// Token cache status (from database cache)
export interface TokenCacheStatus {
  has_cached_token: boolean;
  is_valid: boolean;
  is_expiring_soon: boolean;
  expiry_time?: string;
  last_refresh?: string;
  refresh_error_count: number;
  last_refresh_error?: string;
}

// Request types
export interface AddCredentialRequest {
  provider_type: string;
  credential: CredentialData;
  name?: string;
  check_health?: boolean;
  check_model_name?: string;
}

export interface UpdateCredentialRequest {
  name?: string;
  is_disabled?: boolean;
  check_health?: boolean;
  check_model_name?: string;
  not_supported_models?: string[];
  /// 新的凭证文件路径（仅适用于OAuth凭证，用于重新上传文件）
  new_creds_file_path?: string;
  /// OAuth相关：新的project_id（仅适用于Gemini）
  new_project_id?: string;
  /// API Key 相关：新的 base_url（仅适用于 API Key 凭证）
  new_base_url?: string;
  /// API Key 相关：新的 api_key（仅适用于 API Key 凭证）
  new_api_key?: string;
  /// 新的代理 URL（可覆盖全局代理设置）
  new_proxy_url?: string;
}

export const providerPoolApi = {
  // Get overview of all provider pools
  async getOverview(): Promise<ProviderPoolOverview[]> {
    return invoke("get_provider_pool_overview");
  },

  // Get credentials for a specific provider type
  async getCredentials(
    providerType: PoolProviderType,
  ): Promise<CredentialDisplay[]> {
    return invoke("get_provider_pool_credentials", { providerType });
  },

  // Add a generic credential
  async addCredential(
    request: AddCredentialRequest,
  ): Promise<ProviderCredential> {
    return invoke("add_provider_pool_credential", { request });
  },

  // Update a credential
  async updateCredential(
    uuid: string,
    request: UpdateCredentialRequest,
  ): Promise<ProviderCredential> {
    return invoke("update_provider_pool_credential", { uuid, request });
  },

  // Delete a credential
  async deleteCredential(
    uuid: string,
    providerType?: PoolProviderType,
  ): Promise<boolean> {
    return invoke("delete_provider_pool_credential", { uuid, providerType });
  },

  // Toggle credential enabled/disabled
  async toggleCredential(
    uuid: string,
    isDisabled: boolean,
  ): Promise<ProviderCredential> {
    return invoke("toggle_provider_pool_credential", { uuid, isDisabled });
  },

  // Reset credential counters
  async resetCredential(uuid: string): Promise<void> {
    return invoke("reset_provider_pool_credential", { uuid });
  },

  // Reset health status for all credentials of a type
  async resetHealth(providerType: PoolProviderType): Promise<number> {
    return invoke("reset_provider_pool_health", { providerType });
  },

  // Check health of a single credential
  async checkCredentialHealth(uuid: string): Promise<HealthCheckResult> {
    return invoke("check_provider_pool_credential_health", { uuid });
  },

  // Check health of all credentials of a type
  async checkTypeHealth(
    providerType: PoolProviderType,
  ): Promise<HealthCheckResult[]> {
    return invoke("check_provider_pool_type_health", { providerType });
  },

  // Provider-specific add methods
  async addKiroOAuth(
    credsFilePath: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_kiro_oauth_credential", { credsFilePath, name });
  },

  // 从 JSON 内容添加 Kiro 凭证（直接粘贴 JSON）
  async addKiroFromJson(
    jsonContent: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_kiro_from_json", { jsonContent, name });
  },

  async addGeminiOAuth(
    credsFilePath: string,
    projectId?: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_gemini_oauth_credential", {
      credsFilePath,
      projectId,
      name,
    });
  },

  async addQwenOAuth(
    credsFilePath: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_qwen_oauth_credential", { credsFilePath, name });
  },

  async addOpenAIKey(
    apiKey: string,
    baseUrl?: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_openai_key_credential", { apiKey, baseUrl, name });
  },

  async addClaudeKey(
    apiKey: string,
    baseUrl?: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_claude_key_credential", { apiKey, baseUrl, name });
  },

  async addGeminiApiKey(
    apiKey: string,
    baseUrl?: string,
    excludedModels?: string[],
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_gemini_api_key_credential", {
      apiKey,
      baseUrl,
      excludedModels,
      name,
    });
  },

  async addAntigravityOAuth(
    credsFilePath: string,
    projectId?: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_antigravity_oauth_credential", {
      credsFilePath,
      projectId,
      name,
    });
  },

  async addCodexOAuth(
    credsFilePath: string,
    apiBaseUrl?: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_codex_oauth_credential", {
      credsFilePath,
      apiBaseUrl,
      name,
    });
  },

  async addClaudeOAuth(
    credsFilePath: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_claude_oauth_credential", { credsFilePath, name });
  },

  async addIFlowOAuth(
    credsFilePath: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_iflow_oauth_credential", { credsFilePath, name });
  },

  async addIFlowCookie(
    credsFilePath: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("add_iflow_cookie_credential", { credsFilePath, name });
  },

  // Antigravity OAuth 登录（打开浏览器授权）
  async startAntigravityOAuthLogin(
    name?: string,
    skipProjectIdFetch?: boolean,
  ): Promise<ProviderCredential> {
    return invoke("start_antigravity_oauth_login", {
      name,
      skipProjectIdFetch,
    });
  },

  // 获取 Antigravity OAuth 授权 URL 并等待回调（不自动打开浏览器）
  // 服务器会在后台等待回调，成功后返回凭证
  // 如果需要显示 URL，错误信息会包含 AUTH_URL: 前缀
  async getAntigravityAuthUrlAndWait(
    name?: string,
    skipProjectIdFetch?: boolean,
  ): Promise<ProviderCredential> {
    return invoke("get_antigravity_auth_url_and_wait", {
      name,
      skipProjectIdFetch,
    });
  },

  // Codex OAuth 登录（打开浏览器授权）
  async startCodexOAuthLogin(name?: string): Promise<ProviderCredential> {
    return invoke("start_codex_oauth_login", { name });
  },

  // 获取 Codex OAuth 授权 URL 并等待回调（不自动打开浏览器）
  // 服务器会在后台等待回调，成功后返回凭证
  async getCodexAuthUrlAndWait(name?: string): Promise<ProviderCredential> {
    return invoke("get_codex_auth_url_and_wait", { name });
  },

  // Claude OAuth 登录（打开浏览器授权）
  async startClaudeOAuthLogin(name?: string): Promise<ProviderCredential> {
    return invoke("start_claude_oauth_login", { name });
  },

  // 获取 Claude OAuth 授权 URL 并等待回调（不自动打开浏览器）
  // 服务器会在后台等待回调，成功后返回凭证
  async getClaudeOAuthAuthUrlAndWait(
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("get_claude_oauth_auth_url_and_wait", { name });
  },

  // Claude Cookie 自动授权（使用 sessionKey 自动完成 OAuth 流程）
  // 这是一个更便捷的授权方式，无需手动复制授权码
  async claudeOAuthWithCookie(
    sessionKey: string,
    isSetupToken?: boolean,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("claude_oauth_with_cookie", {
      sessionKey,
      isSetupToken,
      name,
    });
  },

  // Qwen Device Code Flow 登录（打开浏览器授权）
  async startQwenDeviceCodeLogin(name?: string): Promise<ProviderCredential> {
    return invoke("start_qwen_device_code_login", { name });
  },

  // 获取 Qwen Device Code 并等待用户授权（不自动打开浏览器）
  // 服务器会在后台轮询等待授权，成功后返回凭证
  async getQwenDeviceCodeAndWait(name?: string): Promise<ProviderCredential> {
    return invoke("get_qwen_device_code_and_wait", { name });
  },

  // iFlow OAuth 登录（打开浏览器授权）
  async startIFlowOAuthLogin(name?: string): Promise<ProviderCredential> {
    return invoke("start_iflow_oauth_login", { name });
  },

  // 获取 iFlow OAuth 授权 URL 并等待回调（不自动打开浏览器）
  // 服务器会在后台等待回调，成功后返回凭证
  async getIFlowAuthUrlAndWait(name?: string): Promise<ProviderCredential> {
    return invoke("get_iflow_auth_url_and_wait", { name });
  },

  // Gemini OAuth 登录（打开浏览器授权）
  async startGeminiOAuthLogin(name?: string): Promise<ProviderCredential> {
    return invoke("start_gemini_oauth_login", { name });
  },

  // 获取 Gemini OAuth 授权 URL 并等待回调（不自动打开浏览器）
  // 服务器会在后台等待回调，成功后返回凭证
  async getGeminiAuthUrlAndWait(name?: string): Promise<ProviderCredential> {
    return invoke("get_gemini_auth_url_and_wait", { name });
  },

  // 用 Gemini 授权码交换 token
  async exchangeGeminiCode(
    code: string,
    sessionId?: string,
    name?: string,
  ): Promise<ProviderCredential> {
    return invoke("exchange_gemini_code", { code, sessionId, name });
  },

  // ============ Kiro Builder ID 登录 ============

  // 启动 Kiro Builder ID 登录（OIDC Device Authorization Flow）
  async startKiroBuilderIdLogin(
    region?: string,
  ): Promise<KiroBuilderIdLoginResponse> {
    return invoke("start_kiro_builder_id_login", { region });
  },

  // 轮询 Kiro Builder ID 授权状态
  async pollKiroBuilderIdAuth(): Promise<KiroBuilderIdPollResponse> {
    return invoke("poll_kiro_builder_id_auth");
  },

  // 取消 Kiro Builder ID 登录
  async cancelKiroBuilderIdLogin(): Promise<boolean> {
    return invoke("cancel_kiro_builder_id_login");
  },

  // 从 Builder ID 授权结果添加 Kiro 凭证
  async addKiroFromBuilderIdAuth(name?: string): Promise<ProviderCredential> {
    return invoke("add_kiro_from_builder_id_auth", { name });
  },

  // ============ Kiro Social Auth 登录 (Google/GitHub) ============

  // 启动 Kiro Social Auth 登录
  async startKiroSocialAuthLogin(
    provider: "Google" | "Github",
  ): Promise<KiroSocialAuthLoginResponse> {
    return invoke("start_kiro_social_auth_login", { provider });
  },

  // 交换 Kiro Social Auth Token
  async exchangeKiroSocialAuthToken(
    code: string,
    state: string,
  ): Promise<KiroSocialAuthTokenResponse> {
    return invoke("exchange_kiro_social_auth_token", { code, state });
  },

  // 取消 Kiro Social Auth 登录
  async cancelKiroSocialAuthLogin(): Promise<boolean> {
    return invoke("cancel_kiro_social_auth_login");
  },

  // 启动 Kiro Social Auth 回调服务器
  async startKiroSocialAuthCallbackServer(): Promise<boolean> {
    return invoke("start_kiro_social_auth_callback_server");
  },

  // OAuth token management
  async refreshCredentialToken(uuid: string): Promise<string> {
    return invoke("refresh_pool_credential_token", { uuid });
  },

  async getCredentialOAuthStatus(uuid: string): Promise<OAuthStatus> {
    return invoke("get_pool_credential_oauth_status", { uuid });
  },

  // Migration API
  async migratePrivateConfig(config: unknown): Promise<MigrationResult> {
    return invoke("migrate_private_config_to_pool", { config });
  },

  // 获取单个凭证的健康状态
  // Requirements: 4.4
  async getCredentialHealth(
    uuid: string,
  ): Promise<CredentialHealthInfo | null> {
    return invoke("get_credential_health", { uuid });
  },

  // 获取所有凭证的健康状态
  // Requirements: 4.4
  async getAllCredentialHealth(): Promise<CredentialHealthInfo[]> {
    return invoke("get_all_credential_health");
  },
};

// Migration result
export interface MigrationResult {
  migrated_count: number;
  skipped_count: number;
  errors: string[];
}

// Kiro Builder ID 登录响应
export interface KiroBuilderIdLoginResponse {
  success: boolean;
  userCode?: string;
  verificationUri?: string;
  expiresIn?: number;
  interval?: number;
  error?: string;
}

// Kiro Builder ID 轮询响应
export interface KiroBuilderIdPollResponse {
  success: boolean;
  completed: boolean;
  status?: string;
  error?: string;
}

// Kiro Social Auth 登录响应
export interface KiroSocialAuthLoginResponse {
  success: boolean;
  loginUrl?: string;
  state?: string;
  error?: string;
}

// Kiro Social Auth Token 交换响应
export interface KiroSocialAuthTokenResponse {
  success: boolean;
  error?: string;
}

// Kiro 凭证指纹信息
export interface KiroFingerprintInfo {
  /** Machine ID（SHA256 哈希，64 字符） */
  machine_id: string;
  /** Machine ID 的短格式（前 16 字符） */
  machine_id_short: string;
  /** 指纹来源（profileArn / clientId / system） */
  source: string;
  /** 认证方式 */
  auth_method: string;
}

// 凭证健康状态信息
// Requirements: 4.4
export interface CredentialHealthInfo {
  /** 凭证 UUID */
  uuid: string;
  /** 凭证名称 */
  name?: string;
  /** Provider 类型 */
  provider_type: string;
  /** 是否健康 */
  is_healthy: boolean;
  /** 最后错误信息 */
  last_error?: string;
  /** 最后错误时间（RFC3339 格式） */
  last_error_time?: string;
  /** 错误次数 */
  failure_count: number;
  /** 是否需要重新授权 */
  requires_reauth: boolean;
}

// Playwright 状态
export interface PlaywrightStatus {
  /** 浏览器是否可用 */
  available: boolean;
  /** 浏览器可执行文件路径 */
  browserPath?: string;
  /** 浏览器来源: "system" 或 "playwright" */
  browserSource?: "system" | "playwright";
  /** 错误信息 */
  error?: string;
}

// 获取 Kiro 凭证的指纹信息
export async function getKiroCredentialFingerprint(
  uuid: string,
): Promise<KiroFingerprintInfo> {
  return invoke("get_kiro_credential_fingerprint", { uuid });
}

// 切换到本地的结果
export interface SwitchToLocalResult {
  /** 是否成功 */
  success: boolean;
  /** 结果消息 */
  message: string;
  /** 是否需要用户操作（如需管理员权限） */
  requires_action: boolean;
  /** 切换的 Machine ID */
  machine_id?: string;
  /** 是否需要重启 Kiro IDE */
  requires_kiro_restart: boolean;
}

// 切换 Kiro 凭证到本地
export async function switchKiroToLocal(
  uuid: string,
): Promise<SwitchToLocalResult> {
  return invoke("switch_kiro_to_local", { uuid });
}

// 获取当前本地使用的 Kiro 凭证 UUID
export async function getLocalKiroCredentialUuid(): Promise<string | null> {
  return invoke("get_local_kiro_credential_uuid");
}

// ============ Kiro 凭证池管理 HTTP API ============

/** 可用凭证信息 */
export interface AvailableCredential {
  /** 凭证UUID */
  uuid: string;
  /** 凭证名称 */
  name: string;
  /** 是否可用 */
  available: boolean;
  /** Token过期时间 */
  expires_at?: string;
  /** 最后使用时间 */
  last_used?: string;
  /** 健康状态分数 (0-100) */
  health_score: number;
  /** 错误计数 */
  error_count: number;
  /** 最后错误信息 */
  last_error?: string;
}

/** 获取可用凭证列表的响应 */
export interface AvailableCredentialsResponse {
  /** 可用凭证列表 */
  credentials: AvailableCredential[];
  /** 总凭证数 */
  total: number;
  /** 可用凭证数 */
  available: number;
  /** 系统状态 */
  status: string;
}

/** 选择凭证请求参数 */
export interface SelectCredentialRequest {
  /** 指定模型（可选） */
  model?: string;
  /** 强制选择特定UUID（可选） */
  force_uuid?: string;
}

/** 选择凭证响应 */
export interface SelectCredentialResponse {
  /** 选中��凭证UUID */
  uuid: string;
  /** 凭证名称 */
  name: string;
  /** Access Token（脱敏显示） */
  access_token_preview: string;
  /** Token过期时间 */
  expires_at?: string;
  /** 选择原因 */
  selection_reason: string;
}

/** 刷新凭证响应 */
export interface RefreshCredentialResponse {
  /** 凭证UUID */
  uuid: string;
  /** 刷新是否成功 */
  success: boolean;
  /** 新的过期时间 */
  new_expires_at?: string;
  /** 刷新结果信息 */
  message: string;
  /** 错误信息（如果有） */
  error?: string;
}

/** Kiro 凭证池管理 API */
export const kiroCredentialApi = {
  /** 获取可用凭证列表 */
  async getAvailableCredentials(): Promise<AvailableCredentialsResponse> {
    const response = await fetch("/api/kiro/credentials/available");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  },

  /** 智能选择凭证 */
  async selectCredential(
    request: SelectCredentialRequest,
  ): Promise<SelectCredentialResponse> {
    const response = await fetch("/api/kiro/credentials/select", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.message || `HTTP ${response.status}: ${response.statusText}`,
      );
    }
    return response.json();
  },

  /** 手动刷新指定凭证 */
  async refreshCredential(uuid: string): Promise<RefreshCredentialResponse> {
    const response = await fetch(`/api/kiro/credentials/${uuid}/refresh`, {
      method: "PUT",
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.message || `HTTP ${response.status}: ${response.statusText}`,
      );
    }
    return response.json();
  },

  /** 获取凭证详细状态 */
  async getCredentialStatus(uuid: string): Promise<Record<string, any>> {
    const response = await fetch(`/api/kiro/credentials/${uuid}/status`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.message || `HTTP ${response.status}: ${response.statusText}`,
      );
    }
    return response.json();
  },
};

// ============ Playwright 指纹浏览器登录 ============

/**
 * 检查 Playwright 是否可用
 * Requirements: 2.1
 */
export async function checkPlaywrightAvailable(): Promise<PlaywrightStatus> {
  return invoke("check_playwright_available");
}

/**
 * 安装 Playwright Chromium 浏览器
 * Requirements: 6.1, 6.2
 *
 * 执行 npm install playwright && npx playwright install chromium
 * 会发送 playwright-install-progress 事件通知安装进度
 */
export async function installPlaywright(): Promise<PlaywrightStatus> {
  return invoke("install_playwright");
}

/**
 * 使用 Playwright 指纹浏览器启动 Kiro 登录
 * Requirements: 3.1
 *
 * @param provider 登录提供商: Google, Github, BuilderId
 * @param name 可选的凭证名称
 */
export async function startKiroPlaywrightLogin(
  provider: "Google" | "Github" | "BuilderId",
  name?: string,
): Promise<ProviderCredential> {
  return invoke("start_kiro_playwright_login", { provider, name });
}

/**
 * 取消 Playwright 登录
 * Requirements: 5.3
 */
export async function cancelKiroPlaywrightLogin(): Promise<boolean> {
  return invoke("cancel_kiro_playwright_login");
}
