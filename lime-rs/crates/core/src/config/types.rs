//! 配置类型定义
//!
//! 定义 Lime 的配置结构，支持 YAML 和 JSON 序列化/反序列化
//! 保持与旧版 JSON 配置的向后兼容性

use super::tool_execution::ToolExecutionPolicyConfig;
use crate::models::injection_types::{InjectionMode, InjectionRule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// ============ 已退役的旧凭证池配置类型 ============

/// 旧凭证池配置。
///
/// 仅用于读取历史配置和启动期清理，不再作为运行时凭证来源。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CredentialPoolConfig {
    /// Kiro 凭证列表（OAuth）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kiro: Vec<CredentialEntry>,
    /// Gemini 凭证列表（OAuth）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gemini: Vec<CredentialEntry>,
    /// Qwen 凭证列表（OAuth）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub qwen: Vec<CredentialEntry>,
    /// OpenAI 凭证列表（API Key）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub openai: Vec<ApiKeyEntry>,
    /// Claude 凭证列表（API Key）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub claude: Vec<ApiKeyEntry>,
    /// Gemini API Key 凭证列表（多账号负载均衡）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gemini_api_keys: Vec<GeminiApiKeyEntry>,
    /// Vertex AI 凭证列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vertex_api_keys: Vec<VertexApiKeyEntry>,
    /// Codex OAuth 凭证列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub codex: Vec<CredentialEntry>,
}

// ============ ASR 语音服务配置类型 ============

/// ASR Provider 类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AsrProviderType {
    /// 本地 Whisper（离线）
    #[default]
    WhisperLocal,
    /// 本地 SenseVoice Small（离线）
    SenseVoiceLocal,
    /// 讯飞语音识别
    Xunfei,
    /// 百度语音识别
    Baidu,
    /// OpenAI Whisper API
    OpenAI,
}

/// Whisper 模型大小
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WhisperModelSize {
    /// tiny - 最小，最快（~75MB）
    Tiny,
    /// base - 基础（~142MB）
    #[default]
    Base,
    /// small - 小型（~466MB）
    Small,
    /// medium - 中型（~1.5GB）
    Medium,
}

/// ASR 凭证条目
///
/// 用于语音识别服务的凭证管理
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrCredentialEntry {
    /// 凭证 ID
    pub id: String,
    /// Provider 类型
    pub provider: AsrProviderType,
    /// 显示名称
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 是否为默认凭证
    #[serde(default)]
    pub is_default: bool,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
    /// 识别语言（如 "zh", "en", "auto"）
    #[serde(default = "default_asr_language")]
    pub language: String,
    /// Whisper 本地配置（仅 WhisperLocal）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whisper_config: Option<WhisperLocalConfig>,
    /// SenseVoice 本地配置（仅 SenseVoiceLocal）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensevoice_config: Option<SenseVoiceLocalConfig>,
    /// 讯飞配置（仅 Xunfei）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xunfei_config: Option<XunfeiConfig>,
    /// 百度配置（仅 Baidu）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baidu_config: Option<BaiduConfig>,
    /// OpenAI 配置（仅 OpenAI）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_config: Option<OpenAIAsrConfig>,
}

fn default_asr_language() -> String {
    "zh".to_string()
}

/// Whisper 本地配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WhisperLocalConfig {
    /// 模型大小
    #[serde(default)]
    pub model: WhisperModelSize,
    /// 模型文件路径（可选，默认自动下载）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
}

/// SenseVoice 本地配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SenseVoiceLocalConfig {
    /// 模型 ID
    #[serde(default = "default_sensevoice_model_id")]
    pub model_id: String,
    /// 模型目录（可选，默认使用应用数据目录）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<String>,
    /// 是否启用逆文本标准化
    #[serde(default = "default_sensevoice_use_itn")]
    pub use_itn: bool,
    /// 推理线程数
    #[serde(default = "default_sensevoice_num_threads")]
    pub num_threads: u16,
    /// VAD 模型 ID（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vad_model_id: Option<String>,
}

fn default_sensevoice_model_id() -> String {
    "sensevoice-small-int8-2024-07-17".to_string()
}

fn default_sensevoice_use_itn() -> bool {
    true
}

fn default_sensevoice_num_threads() -> u16 {
    4
}

impl Default for SenseVoiceLocalConfig {
    fn default() -> Self {
        Self {
            model_id: default_sensevoice_model_id(),
            model_dir: None,
            use_itn: default_sensevoice_use_itn(),
            num_threads: default_sensevoice_num_threads(),
            vad_model_id: Some("silero-vad-onnx".to_string()),
        }
    }
}

/// 讯飞语音配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XunfeiConfig {
    /// App ID
    pub app_id: String,
    /// API Key
    pub api_key: String,
    /// API Secret
    pub api_secret: String,
}

/// 百度语音配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaiduConfig {
    /// API Key
    pub api_key: String,
    /// Secret Key
    pub secret_key: String,
}

/// OpenAI ASR 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIAsrConfig {
    /// API Key
    pub api_key: String,
    /// 自定义 Base URL（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// 代理 URL（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
}

/// Gemini API Key 凭证条目
///
/// 用于 Gemini API Key 多账号负载均衡
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeminiApiKeyEntry {
    /// 凭证 ID
    pub id: String,
    /// API Key
    pub api_key: String,
    /// 自定义 Base URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// 单独的代理 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    /// 排除的模型列表（支持通配符）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_models: Vec<String>,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
}

/// Vertex AI 模型别名映射
// Vertex AI 类型从 core crate 重新导出
pub use crate::models::vertex_model::{VertexApiKeyEntry, VertexModelAlias};

#[allow(dead_code)]
fn default_auth_type() -> String {
    "oauth".to_string()
}

/// OAuth 凭证条目
///
/// 用于 Kiro、Gemini、Qwen 等 OAuth 认证的 Provider
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CredentialEntry {
    /// 凭证 ID
    pub id: String,
    /// Token 文件路径（相对于 auth_dir）
    pub token_file: String,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
    /// 单独的代理 URL（覆盖全局代理）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
}

/// API Key 凭证条目
///
/// 用于 OpenAI、Claude 等 API Key 认证的 Provider
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiKeyEntry {
    /// 凭证 ID
    pub id: String,
    /// API Key
    pub api_key: String,
    /// 自定义 Base URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// 是否禁用
    #[serde(default)]
    pub disabled: bool,
    /// 单独的代理 URL（覆盖全局代理）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
}

/// 默认 auth_dir 路径
fn default_auth_dir() -> String {
    "~/.lime/auth".to_string()
}

/// 端点 Provider 配置
///
/// 允许为不同的客户端端点配置不同的 Provider
/// 例如：Cursor 使用 Qwen，Claude Code 使用 Kiro，Codex 使用 Codex
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct EndpointProvidersConfig {
    /// Cursor 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// Claude Code 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claude_code: Option<String>,
    /// Codex 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex: Option<String>,
    /// Windsurf 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windsurf: Option<String>,
    /// Kiro 客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kiro: Option<String>,
    /// 其他客户端使用的 Provider
    /// 如果为空，则使用 default_provider
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub other: Option<String>,
}

impl EndpointProvidersConfig {
    /// 根据客户端类型获取配置的 Provider
    ///
    /// # 参数
    /// - `client_type`: 客户端类型的配置键名（cursor, claude_code, codex, windsurf, kiro, other）
    ///
    /// # 返回
    /// 如果配置了对应的 Provider，返回 Some(&String)；否则返回 None
    pub fn get_provider(&self, client_type: &str) -> Option<&String> {
        match client_type {
            "cursor" => self.cursor.as_ref(),
            "claude_code" => self.claude_code.as_ref(),
            "codex" => self.codex.as_ref(),
            "windsurf" => self.windsurf.as_ref(),
            "kiro" => self.kiro.as_ref(),
            "other" => self.other.as_ref(),
            _ => None,
        }
    }

    /// 设置客户端类型的 Provider
    ///
    /// # 参数
    /// - `client_type`: 客户端类型的配置键名（cursor, claude_code, codex, windsurf, kiro, other）
    /// - `provider`: 要设置的 Provider 名称，None 或空字符串表示清除配置
    ///
    /// # 返回
    /// 如果客户端类型有效，返回 true；否则返回 false
    pub fn set_provider(&mut self, client_type: &str, provider: Option<String>) -> bool {
        let provider = provider.filter(|p| !p.is_empty());
        match client_type {
            "cursor" => {
                self.cursor = provider;
                true
            }
            "claude_code" => {
                self.claude_code = provider;
                true
            }
            "codex" => {
                self.codex = provider;
                true
            }
            "windsurf" => {
                self.windsurf = provider;
                true
            }
            "kiro" => {
                self.kiro = provider;
                true
            }
            "other" => {
                self.other = provider;
                true
            }
            _ => false,
        }
    }
}

/// 主配置结构
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    /// 服务器配置
    #[serde(default)]
    pub server: ServerConfig,
    /// Provider 配置
    #[serde(default)]
    pub providers: ProvidersConfig,
    /// 当前默认 Provider
    #[serde(default = "default_provider")]
    pub default_provider: String,
    /// 路由辅助配置
    #[serde(default)]
    pub routing: RoutingConfig,
    /// 重试配置
    #[serde(default)]
    pub retry: RetrySettings,
    /// 日志配置
    #[serde(default)]
    pub logging: LoggingConfig,
    /// 参数注入配置
    #[serde(default)]
    pub injection: InjectionSettings,
    /// 认证目录路径（存储 OAuth Token 文件，支持 ~ 展开）
    #[serde(default = "default_auth_dir")]
    pub auth_dir: String,
    /// 已退役的旧凭证池配置，仅用于读取历史配置和启动期清理，不再序列化新配置。
    #[serde(default, skip_serializing)]
    pub credential_pool: CredentialPoolConfig,
    /// 远程管理配置
    #[serde(default)]
    pub remote_management: RemoteManagementConfig,
    /// 配额超限配置
    #[serde(default)]
    pub quota_exceeded: QuotaExceededConfig,
    /// 全局代理 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url: Option<String>,
    /// Amp CLI 配置
    #[serde(default)]
    pub ampcode: AmpConfig,
    /// 端点 Provider 配置
    /// 允许为不同的客户端端点（CC/Codex）配置不同的 Provider
    #[serde(default)]
    pub endpoint_providers: EndpointProvidersConfig,
    /// 关闭时最小化到托盘（而不是退出应用）
    #[serde(default = "default_minimize_to_tray")]
    pub minimize_to_tray: bool,
    /// 用户界面语言。
    ///
    /// 新配置使用 BCP 47 风格 locale（如 "zh-CN"、"en-US"）或 "auto"；
    /// 历史配置中的 "zh" / "en" 仅作为前端读取兼容值保留。
    #[serde(default = "default_language")]
    pub language: String,
    /// 模型配置（动态加载 Provider 和模型列表）
    #[serde(default)]
    pub models: ModelsConfig,
    /// Native Agent 配置
    #[serde(default)]
    pub agent: NativeAgentConfig,
    /// Agent Skills 用户级启停配置
    #[serde(default, skip_serializing_if = "SkillsConfig::is_default")]
    pub skills: SkillsConfig,
    /// Orchestrator-owned Skills 与 Apps MCP 暴露开关。
    #[serde(default, skip_serializing_if = "OrchestratorConfig::is_default")]
    pub orchestrator: OrchestratorConfig,
    /// 实验室功能配置
    #[serde(default)]
    pub experimental: ExperimentalFeatures,
    /// Tool Calling 2.0 配置
    #[serde(default)]
    pub tool_calling: ToolCallingConfig,
    /// 工作区偏好配置
    #[serde(default, rename = "workspace_preferences")]
    pub workspace_preferences: WorkspacePreferencesConfig,
    /// 导航栏配置
    #[serde(default)]
    pub navigation: NavigationConfig,
    /// 聊天外观配置
    #[serde(default)]
    pub chat_appearance: ChatAppearanceConfig,
    /// 统一环境变量配置
    #[serde(default, skip_serializing_if = "EnvironmentConfig::is_default")]
    pub environment: EnvironmentConfig,
    /// 网络搜索偏好配置
    #[serde(default)]
    pub web_search: WebSearchConfig,
    /// 记忆管理配置
    #[serde(default)]
    pub memory: MemoryConfig,
    /// 图像生成服务配置
    #[serde(default)]
    pub image_gen: ImageGenConfig,
    /// 用户资料
    #[serde(default)]
    pub user_profile: UserProfile,
    /// 开发者功能开关
    #[serde(default)]
    pub developer: DeveloperConfig,
    /// 速率限制配置
    #[serde(default)]
    pub rate_limit: RateLimitSettings,
    /// 崩溃上报配置（Sentry 协议兼容）
    #[serde(default)]
    pub crash_reporting: CrashReportingConfig,
    /// 对话管理配置
    #[serde(default)]
    pub conversation: ConversationSettings,
    /// 提示路由配置
    #[serde(default)]
    pub hint_router: HintRouterSettings,
    /// 配对认证配置
    #[serde(default)]
    pub pairing: PairingSettings,
    /// 自动化调度配置
    #[serde(default)]
    pub automation: AutomationSettings,
    /// Gateway 全局配置（隧道/公网接入）
    #[serde(default)]
    pub gateway: GatewayConfig,
    /// 渠道配置（Telegram / Discord / 飞书 Bot）
    #[serde(default)]
    pub channels: ChannelsConfig,
}

// ============ Native Agent 配置类型 ============

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SkillsConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config: Vec<SkillConfig>,
}

impl SkillsConfig {
    pub fn is_default(value: &Self) -> bool {
        value == &Self::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OrchestratorFeatureConfig {
    #[serde(default = "default_orchestrator_feature_enabled")]
    pub enabled: bool,
}

impl Default for OrchestratorFeatureConfig {
    fn default() -> Self {
        Self {
            enabled: default_orchestrator_feature_enabled(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct OrchestratorConfig {
    #[serde(default)]
    pub skills: OrchestratorFeatureConfig,
    #[serde(default)]
    pub mcp: OrchestratorFeatureConfig,
}

impl OrchestratorConfig {
    pub fn is_default(value: &Self) -> bool {
        value == &Self::default()
    }
}

fn default_orchestrator_feature_enabled() -> bool {
    true
}

/// Native Agent 配置
///
/// 配置内置 Agent 的行为，包括系统提示词、工具使用规则等
/// 参考 Manus Agent 的模块化设计，支持灵活配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkspaceSandboxConfig {
    /// 是否启用 workspace 本地 sandbox
    #[serde(default = "default_workspace_sandbox_enabled")]
    pub enabled: bool,
    /// 严格模式：若 sandbox 不可用则直接失败
    #[serde(default = "default_workspace_sandbox_strict")]
    pub strict: bool,
    /// 发生降级时是否提醒用户
    #[serde(default = "default_workspace_sandbox_notify_on_fallback")]
    pub notify_on_fallback: bool,
}

impl WorkspaceSandboxConfig {
    pub fn is_default(value: &Self) -> bool {
        value == &Self::default()
    }
}

fn default_workspace_sandbox_enabled() -> bool {
    false
}

fn default_workspace_sandbox_strict() -> bool {
    false
}

fn default_workspace_sandbox_notify_on_fallback() -> bool {
    true
}

impl Default for WorkspaceSandboxConfig {
    fn default() -> Self {
        Self {
            enabled: default_workspace_sandbox_enabled(),
            strict: default_workspace_sandbox_strict(),
            notify_on_fallback: default_workspace_sandbox_notify_on_fallback(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NativeAgentConfig {
    /// 是否使用默认系统提示词
    /// 当 custom_system_prompt 为空时，如果此项为 true 则使用内置默认提示词
    #[serde(default = "default_use_default_prompt")]
    pub use_default_system_prompt: bool,
    /// 自定义系统提示词
    /// 如果设置了此项，将覆盖默认系统提示词
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_system_prompt: Option<String>,
    /// 系统提示词模板文件路径（支持 ~ 展开）
    /// 可以将系统提示词存储在外部文件中，便于管理和版本控制
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_file: Option<String>,
    /// 默认模型
    #[serde(default = "default_agent_model")]
    pub default_model: String,
    /// 默认温度参数
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// 默认最大 token 数
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// workspace 本地 sandbox 配置（可选安全增强）
    #[serde(default, skip_serializing_if = "WorkspaceSandboxConfig::is_default")]
    pub workspace_sandbox: WorkspaceSandboxConfig,
    /// 工具执行权限覆盖配置（默认策略之上的持久化覆盖）
    #[serde(default, skip_serializing_if = "ToolExecutionPolicyConfig::is_default")]
    pub tool_execution: ToolExecutionPolicyConfig,
    /// root thread 与所有子 Agent 共享的 rollout token 预算。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_budget: Option<RolloutBudgetConfig>,
}

fn default_use_default_prompt() -> bool {
    true
}

fn default_agent_model() -> String {
    "claude-sonnet-4-20250514".to_string()
}

fn default_temperature() -> f32 {
    0.7
}

fn default_max_tokens() -> u32 {
    4096
}

impl Default for NativeAgentConfig {
    fn default() -> Self {
        Self {
            use_default_system_prompt: default_use_default_prompt(),
            custom_system_prompt: None,
            system_prompt_file: None,
            default_model: default_agent_model(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            workspace_sandbox: WorkspaceSandboxConfig::default(),
            tool_execution: ToolExecutionPolicyConfig::default(),
            rollout_budget: None,
        }
    }
}

/// Codex-compatible shared rollout budget configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RolloutBudgetConfig {
    pub limit_tokens: i64,
    #[serde(default)]
    pub reminder_at_remaining_tokens: Vec<i64>,
    #[serde(default = "default_rollout_budget_weight")]
    pub sampling_token_weight: f64,
    #[serde(default = "default_rollout_budget_weight")]
    pub prefill_token_weight: f64,
}

fn default_rollout_budget_weight() -> f64 {
    1.0
}

impl RolloutBudgetConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.limit_tokens <= 0 {
            return Err("rollout budget limit_tokens must be positive".to_string());
        }
        if self
            .reminder_at_remaining_tokens
            .iter()
            .any(|threshold| *threshold <= 0 || *threshold >= self.limit_tokens)
        {
            return Err("rollout budget reminders must be in (0, limit_tokens)".to_string());
        }
        if !self.sampling_token_weight.is_finite() || self.sampling_token_weight < 0.0 {
            return Err(
                "rollout budget sampling_token_weight must be finite and non-negative".to_string(),
            );
        }
        if !self.prefill_token_weight.is_finite() || self.prefill_token_weight < 0.0 {
            return Err(
                "rollout budget prefill_token_weight must be finite and non-negative".to_string(),
            );
        }
        Ok(())
    }
}

// ============ 工作区偏好配置类型 ============

/// 工作区偏好配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkspacePreferencesConfig {
    /// 工作区偏好配置版本
    #[serde(default)]
    pub schema_version: u8,
    /// Agent 默认回复语言偏好；不等同于 UI locale 或内容产物语言
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_response_language: Option<String>,
    /// 全局媒体生成默认设置
    #[serde(default)]
    pub media_defaults: MediaGenerationDefaultsConfig,
    /// 服务模型设置
    #[serde(default, skip_serializing_if = "ServiceModelsConfig::is_default")]
    pub service_models: ServiceModelsConfig,
}

fn current_workspace_preferences_schema_version() -> u8 {
    3
}

impl Default for WorkspacePreferencesConfig {
    fn default() -> Self {
        Self {
            schema_version: current_workspace_preferences_schema_version(),
            agent_response_language: None,
            media_defaults: MediaGenerationDefaultsConfig::default(),
            service_models: ServiceModelsConfig::default(),
        }
    }
}

fn default_media_generation_allow_fallback() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct MediaGenerationPreferenceConfig {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "preferred_provider_id"
    )]
    pub preferred_provider_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "preferred_model_id"
    )]
    pub preferred_model_id: Option<String>,
    #[serde(
        default = "default_media_generation_allow_fallback",
        alias = "allow_fallback"
    )]
    pub allow_fallback: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MediaGenerationDefaultsConfig {
    #[serde(default)]
    pub image: MediaGenerationPreferenceConfig,
    #[serde(default)]
    pub video: MediaGenerationPreferenceConfig,
    #[serde(default)]
    pub voice: MediaGenerationPreferenceConfig,
}

fn default_service_model_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ServiceModelPreferenceConfig {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "preferred_provider_id"
    )]
    pub preferred_provider_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "preferred_model_id"
    )]
    pub preferred_model_id: Option<String>,
    #[serde(default = "default_service_model_enabled", alias = "enabled")]
    pub enabled: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "custom_prompt"
    )]
    pub custom_prompt: Option<String>,
}

impl ServiceModelPreferenceConfig {
    pub fn is_default(value: &Self) -> bool {
        value == &Self::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ServiceModelsConfig {
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub responsive_chat: ServiceModelPreferenceConfig,
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub topic: ServiceModelPreferenceConfig,
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub generation_topic: ServiceModelPreferenceConfig,
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub translation: ServiceModelPreferenceConfig,
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub history_compress: ServiceModelPreferenceConfig,
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub agent_meta: ServiceModelPreferenceConfig,
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub input_completion: ServiceModelPreferenceConfig,
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub prompt_rewrite: ServiceModelPreferenceConfig,
    #[serde(
        default,
        skip_serializing_if = "ServiceModelPreferenceConfig::is_default"
    )]
    pub resource_prompt_rewrite: ServiceModelPreferenceConfig,
}

impl ServiceModelsConfig {
    pub fn is_default(value: &Self) -> bool {
        value == &Self::default()
    }
}

// ============ 导航栏配置类型 ============

/// 导航栏模块配置
///
/// 保留历史导航配置字段；当前不再提供可显式开启的隐藏系统入口
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NavigationConfig {
    /// 工作区导航默认值版本
    #[serde(default)]
    pub schema_version: u8,
    /// 历史隐藏系统入口列表；归一化时清空，固定主导航不写入配置
    #[serde(default = "default_enabled_nav_items")]
    pub enabled_items: Vec<String>,
}

fn default_enabled_nav_items() -> Vec<String> {
    Vec::new()
}

impl Default for NavigationConfig {
    fn default() -> Self {
        Self {
            schema_version: current_workspace_preferences_schema_version(),
            enabled_items: default_enabled_nav_items(),
        }
    }
}

const CURRENT_CONFIGURABLE_NAV_ITEM_IDS: &[&str] = &[];

fn normalize_navigation_enabled_items(items: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();

    for item in items {
        let item_str = item.as_str();
        if !CURRENT_CONFIGURABLE_NAV_ITEM_IDS.contains(&item_str) {
            continue;
        }
        if normalized.iter().any(|current| current == item) {
            continue;
        }
        normalized.push(item.clone());
    }

    normalized
}

// ============ 实验室功能配置类型 ============

/// 自动更新检查配置
///
/// 配置自动检查更新的行为，符合 macOS/Windows 平台规范
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpdateCheckConfig {
    /// 是否启用自动检查更新
    #[serde(default = "default_update_check_enabled")]
    pub enabled: bool,
    /// 检查间隔（小时），默认 24 小时
    #[serde(default = "default_check_interval_hours")]
    pub check_interval_hours: u32,
    /// 是否显示系统通知
    #[serde(default = "default_show_notification")]
    pub show_notification: bool,
    /// 上次检查时间（Unix 时间戳，秒）
    #[serde(default)]
    pub last_check_timestamp: u64,
    /// 已跳过的版本（用户选择"跳过此版本"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skipped_version: Option<String>,
    /// 稍后提醒截止时间（Unix 时间戳，秒）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remind_later_until: Option<u64>,
    /// 最近一次通知的版本号
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_notified_version: Option<String>,
    /// 最近一次通知时间（Unix 时间戳，秒）
    #[serde(default)]
    pub last_notified_at: u64,
    /// 下次允许通知时间（Unix 时间戳，秒）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_notify_after: Option<u64>,
    /// 连续关闭提醒次数（用于退避）
    #[serde(default)]
    pub dismiss_streak: u32,
    /// 更新提醒展示次数
    #[serde(default)]
    pub notification_shown_count: u64,
    /// 点击“立即更新”次数
    #[serde(default)]
    pub action_update_now_count: u64,
    /// 点击“稍后提醒”次数
    #[serde(default)]
    pub action_remind_later_count: u64,
    /// 点击“跳过版本”次数
    #[serde(default)]
    pub action_skip_version_count: u64,
    /// 关闭提醒次数（ESC/关闭按钮）
    #[serde(default)]
    pub action_dismiss_count: u64,
}

fn default_update_check_enabled() -> bool {
    true
}

fn default_check_interval_hours() -> u32 {
    24
}

fn default_show_notification() -> bool {
    true
}

impl Default for UpdateCheckConfig {
    fn default() -> Self {
        Self {
            enabled: default_update_check_enabled(),
            check_interval_hours: default_check_interval_hours(),
            show_notification: default_show_notification(),
            last_check_timestamp: 0,
            skipped_version: None,
            remind_later_until: None,
            last_notified_version: None,
            last_notified_at: 0,
            next_notify_after: None,
            dismiss_streak: 0,
            notification_shown_count: 0,
            action_update_now_count: 0,
            action_remind_later_count: 0,
            action_skip_version_count: 0,
            action_dismiss_count: 0,
        }
    }
}

/// WebMCP 预留配置
///
/// 当前仅作为实验开关预留，不参与实际执行链。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WebMcpConfig {
    /// 是否允许未来接入 WebMCP 实验能力
    #[serde(default)]
    pub enabled: bool,
}

/// 实验室功能配置
///
/// 管理所有实验性功能的开关和配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ExperimentalFeatures {
    /// WebMCP 预留配置
    #[serde(default)]
    pub webmcp: WebMcpConfig,
    /// 自动更新检查配置
    #[serde(default)]
    pub update_check: UpdateCheckConfig,
    /// 语音输入功能配置
    #[serde(default)]
    pub voice_input: VoiceInputConfig,
}

/// Tool Calling 2.0 配置
///
/// 统一控制编程式工具调用、动态过滤与 input examples 透传行为。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallingConfig {
    /// 是否启用 Tool Calling 2.0 能力
    #[serde(default = "default_tool_calling_enabled")]
    pub enabled: bool,
    /// 是否启用动态过滤（优先过滤网页抓取噪音）
    #[serde(default = "default_tool_calling_dynamic_filtering_enabled")]
    pub dynamic_filtering: bool,
    /// 是否启用原生 input_examples 透传
    #[serde(default)]
    pub native_input_examples: bool,
}

fn default_tool_calling_enabled() -> bool {
    true
}

fn default_tool_calling_dynamic_filtering_enabled() -> bool {
    true
}

impl Default for ToolCallingConfig {
    fn default() -> Self {
        Self {
            enabled: default_tool_calling_enabled(),
            dynamic_filtering: default_tool_calling_dynamic_filtering_enabled(),
            native_input_examples: false,
        }
    }
}

// ============ 语音输入功能配置类型 ============

/// 语音输入功能配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceInputConfig {
    /// 是否启用语音输入功能
    #[serde(default)]
    pub enabled: bool,
    /// 触发语音输入的全局快捷键
    #[serde(default = "default_voice_shortcut")]
    pub shortcut: String,
    /// 语音处理配置
    #[serde(default)]
    pub processor: VoiceProcessorConfig,
    /// 输出配置
    #[serde(default)]
    pub output: VoiceOutputConfig,
    /// 自定义指令列表
    #[serde(default)]
    pub instructions: Vec<VoiceInstruction>,
    /// ASR 语音服务凭证列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub asr_credentials: Vec<AsrCredentialEntry>,
    /// 选择的麦克风设备 ID（为空时使用系统默认设备）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_device_id: Option<String>,
    /// 是否启用交互音效
    #[serde(default = "default_sound_enabled")]
    pub sound_enabled: bool,
    /// 翻译模式使用的指令 ID
    #[serde(default = "default_translate_instruction_id")]
    pub translate_instruction_id: String,
}

fn default_voice_shortcut() -> String {
    "CommandOrControl+Shift+V".to_string()
}

fn default_sound_enabled() -> bool {
    true
}

fn default_translate_instruction_id() -> String {
    "translate_en".to_string()
}

impl Default for VoiceInputConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            shortcut: default_voice_shortcut(),
            processor: VoiceProcessorConfig::default(),
            output: VoiceOutputConfig::default(),
            instructions: default_instructions(),
            asr_credentials: Vec::new(),
            selected_device_id: None,
            sound_enabled: default_sound_enabled(),
            translate_instruction_id: default_translate_instruction_id(),
        }
    }
}

impl CredentialPoolConfig {
    pub fn is_empty(&self) -> bool {
        self.kiro.is_empty()
            && self.gemini.is_empty()
            && self.qwen.is_empty()
            && self.openai.is_empty()
            && self.claude.is_empty()
            && self.gemini_api_keys.is_empty()
            && self.vertex_api_keys.is_empty()
            && self.codex.is_empty()
    }
}

/// 语音处理配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceProcessorConfig {
    /// 是否启用 AI 润色
    #[serde(default = "default_polish_enabled")]
    pub polish_enabled: bool,
    /// 润色使用的 LLM Provider（使用现有 Provider 系统）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polish_provider: Option<String>,
    /// 润色使用的模型
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polish_model: Option<String>,
    /// 默认指令 ID
    #[serde(default = "default_instruction_id")]
    pub default_instruction_id: String,
}

fn default_polish_enabled() -> bool {
    true
}

fn default_instruction_id() -> String {
    "default".to_string()
}

impl Default for VoiceProcessorConfig {
    fn default() -> Self {
        Self {
            polish_enabled: default_polish_enabled(),
            polish_provider: None,
            polish_model: None,
            default_instruction_id: default_instruction_id(),
        }
    }
}

/// 语音输出配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceOutputConfig {
    /// 输出模式
    #[serde(default)]
    pub mode: VoiceOutputMode,
    /// 输入延迟（毫秒），用于模拟键盘输入
    #[serde(default = "default_type_delay_ms")]
    pub type_delay_ms: u32,
}

fn default_type_delay_ms() -> u32 {
    10
}

impl Default for VoiceOutputConfig {
    fn default() -> Self {
        Self {
            mode: VoiceOutputMode::default(),
            type_delay_ms: default_type_delay_ms(),
        }
    }
}

/// 语音输出模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VoiceOutputMode {
    /// 模拟键盘输入
    #[default]
    Type,
    /// 复制到剪贴板
    Clipboard,
    /// 两者都做
    Both,
}

/// 语音处理指令
///
/// 定义不同的文本处理模式，如默认润色、翻译、邮件格式等
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceInstruction {
    /// 指令 ID
    pub id: String,
    /// 显示名称
    pub name: String,
    /// 指令描述
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Prompt 模板（使用 {{text}} 作为占位符）
    pub prompt: String,
    /// 快捷键（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shortcut: Option<String>,
    /// 是否为系统预设（不可删除）
    #[serde(default)]
    pub is_preset: bool,
    /// 图标（可选，用于 UI 显示）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
}

/// 默认指令列表
fn default_instructions() -> Vec<VoiceInstruction> {
    vec![
        VoiceInstruction {
            id: "default".to_string(),
            name: "默认润色".to_string(),
            description: Some("去除语气词、添加标点、修正语法".to_string()),
            prompt: "请对以下语音转文字内容进行润色，去除语气词（如「嗯」「啊」「那个」等），添加合适的标点符号，修正明显的语法错误，但保持原意不变。只输出润色后的文本，不要添加任何解释：\n\n{{text}}".to_string(),
            shortcut: None,
            is_preset: true,
            icon: Some("sparkles".to_string()),
        },
        VoiceInstruction {
            id: "translate_en".to_string(),
            name: "翻译为英文".to_string(),
            description: Some("将中文翻译为英文".to_string()),
            prompt: "请将以下中文内容翻译为英文，保持专业、自然的表达。只输出翻译结果，不要添加任何解释：\n\n{{text}}".to_string(),
            shortcut: None,
            is_preset: true,
            icon: Some("globe".to_string()),
        },
        VoiceInstruction {
            id: "email".to_string(),
            name: "邮件格式".to_string(),
            description: Some("整理为正式邮件格式".to_string()),
            prompt: "请将以下内容整理为正式的邮件格式，包含适当的问候语和结束语，语气专业礼貌。只输出邮件内容，不要添加任何解释：\n\n{{text}}".to_string(),
            shortcut: None,
            is_preset: true,
            icon: Some("mail".to_string()),
        },
        VoiceInstruction {
            id: "summary".to_string(),
            name: "总结要点".to_string(),
            description: Some("提取关键信息，生成简洁要点".to_string()),
            prompt: "请总结以下内容的要点，用简洁的条目列出关键信息：\n\n{{text}}".to_string(),
            shortcut: None,
            is_preset: true,
            icon: Some("list".to_string()),
        },
        VoiceInstruction {
            id: "raw".to_string(),
            name: "原始输出".to_string(),
            description: Some("不做任何处理，直接输出识别结果".to_string()),
            prompt: "{{text}}".to_string(),
            shortcut: None,
            is_preset: true,
            icon: Some("type".to_string()),
        },
    ]
}

impl NativeAgentConfig {
    /// 获取有效的系统提示词
    ///
    /// 优先级：
    /// 1. system_prompt_file（外部文件）
    /// 2. custom_system_prompt（配置中的自定义提示词）
    /// 3. 如果 use_default_system_prompt 为 true，返回 None 让调用方使用默认提示词
    /// 4. 否则返回 None（不使用任何系统提示词）
    pub fn get_effective_system_prompt(&self) -> Option<String> {
        // 优先从文件加载
        if let Some(file_path) = &self.system_prompt_file {
            let expanded_path = super::path_utils::expand_tilde(file_path);
            if let Ok(content) = std::fs::read_to_string(&expanded_path) {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }

        // 其次使用配置中的自定义提示词
        if let Some(prompt) = &self.custom_system_prompt {
            let trimmed = prompt.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }

        // 返回 None，让调用方根据 use_default_system_prompt 决定是否使用默认提示词
        None
    }
}

fn default_minimize_to_tray() -> bool {
    true
}

fn default_language() -> String {
    "zh-CN".to_string()
}

/// 服务器配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerConfig {
    /// 监听地址
    #[serde(default = "default_host")]
    pub host: String,
    /// 监听端口
    #[serde(default = "default_port")]
    pub port: u16,
    /// API 密钥
    #[serde(default = "default_api_key")]
    pub api_key: String,
    /// TLS 配置
    #[serde(default)]
    pub tls: TlsConfig,
    /// 响应缓存配置（仅影响非流式请求）
    #[serde(default)]
    pub response_cache: ResponseCacheSettings,
}

/// 响应缓存配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseCacheSettings {
    /// 是否启用响应缓存
    #[serde(default = "default_response_cache_enabled")]
    pub enabled: bool,
    /// 缓存 TTL（秒）
    #[serde(default = "default_response_cache_ttl_secs")]
    pub ttl_secs: u64,
    /// 最大缓存条目数
    #[serde(default = "default_response_cache_max_entries")]
    pub max_entries: usize,
    /// 单响应最大缓存字节数
    #[serde(default = "default_response_cache_max_body_bytes")]
    pub max_body_bytes: usize,
    /// 可缓存的 HTTP 状态码列表（默认仅 200）
    #[serde(default = "default_response_cache_cacheable_status_codes")]
    pub cacheable_status_codes: Vec<u16>,
}

fn default_response_cache_enabled() -> bool {
    true
}

fn default_response_cache_ttl_secs() -> u64 {
    600
}

fn default_response_cache_max_entries() -> usize {
    200
}

fn default_response_cache_max_body_bytes() -> usize {
    1_048_576
}

fn default_response_cache_cacheable_status_codes() -> Vec<u16> {
    vec![200]
}

impl Default for ResponseCacheSettings {
    fn default() -> Self {
        Self {
            enabled: default_response_cache_enabled(),
            ttl_secs: default_response_cache_ttl_secs(),
            max_entries: default_response_cache_max_entries(),
            max_body_bytes: default_response_cache_max_body_bytes(),
            cacheable_status_codes: default_response_cache_cacheable_status_codes(),
        }
    }
}

/// TLS 配置
///
/// 用于启用 HTTPS 支持
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TlsConfig {
    /// 是否启用 TLS
    #[serde(default)]
    pub enable: bool,
    /// 证书文件路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cert_path: Option<String>,
    /// 私钥文件路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_path: Option<String>,
}

/// 远程管理配置
///
/// 用于配置远程管理 API 的访问控制
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RemoteManagementConfig {
    /// 是否允许远程访问（非 localhost）
    #[serde(default)]
    pub allow_remote: bool,
    /// 管理 API 密钥（为空时禁用管理 API）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secret_key: Option<String>,
    /// 是否禁用控制面板
    #[serde(default)]
    pub disable_control_panel: bool,
}

/// 配额超限配置
///
/// 用于配置配额超限时的自动切换策略
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuotaExceededConfig {
    /// 是否自动切换到下一个凭证
    #[serde(default = "default_switch_project")]
    pub switch_project: bool,
    /// 是否尝试使用预览模型
    #[serde(default = "default_switch_preview_model")]
    pub switch_preview_model: bool,
    /// 冷却时间（秒）
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
}

fn default_switch_project() -> bool {
    true
}

fn default_switch_preview_model() -> bool {
    true
}

fn default_cooldown_seconds() -> u64 {
    300
}

impl Default for QuotaExceededConfig {
    fn default() -> Self {
        Self {
            switch_project: default_switch_project(),
            switch_preview_model: default_switch_preview_model(),
            cooldown_seconds: default_cooldown_seconds(),
        }
    }
}

/// Amp CLI 模型映射
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AmpModelMapping {
    /// 源模型名称
    pub from: String,
    /// 目标模型名称
    pub to: String,
}

/// Amp CLI 配置
///
/// 用于 Amp CLI 集成
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AmpConfig {
    /// 上游 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_url: Option<String>,
    /// 模型映射列表
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_mappings: Vec<AmpModelMapping>,
    /// 是否限制管理端点只能从 localhost 访问
    #[serde(default)]
    pub restrict_management_to_localhost: bool,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8999
}

pub const DEFAULT_API_KEY: &str = "proxy_cast";

fn default_api_key() -> String {
    DEFAULT_API_KEY.to_string()
}

/// 生成安全 API Key（32 字节随机）
pub fn generate_secure_api_key() -> String {
    use rand::distributions::Alphanumeric;
    use rand::Rng;

    let token: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    format!("pc_{token}")
}

/// 是否为默认 API Key
pub fn is_default_api_key(api_key: &str) -> bool {
    api_key == DEFAULT_API_KEY
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key: default_api_key(),
            tls: TlsConfig::default(),
            response_cache: ResponseCacheSettings::default(),
        }
    }
}

/// Provider 配置集合
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvidersConfig {
    /// Kiro Provider 配置
    #[serde(default)]
    pub kiro: ProviderConfig,
    /// Gemini Provider 配置
    #[serde(default)]
    pub gemini: ProviderConfig,
    /// Qwen Provider 配置
    #[serde(default)]
    pub qwen: ProviderConfig,
    /// OpenAI 自定义 Provider 配置
    #[serde(default)]
    pub openai: CustomProviderConfig,
    /// Claude 自定义 Provider 配置
    #[serde(default)]
    pub claude: CustomProviderConfig,
}

impl Default for ProvidersConfig {
    fn default() -> Self {
        Self {
            kiro: ProviderConfig::default(),
            gemini: ProviderConfig {
                enabled: false,
                credentials_path: Some("~/.gemini/oauth_creds.json".to_string()),
                region: None,
                project_id: None,
            },
            qwen: ProviderConfig {
                enabled: false,
                credentials_path: Some("~/.qwen/oauth_creds.json".to_string()),
                region: None,
                project_id: None,
            },
            openai: CustomProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("https://api.openai.com/v1".to_string()),
            },
            claude: CustomProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("https://api.anthropic.com".to_string()),
            },
        }
    }
}

/// OAuth Provider 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProviderConfig {
    /// 是否启用
    #[serde(default)]
    pub enabled: bool,
    /// 凭证文件路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials_path: Option<String>,
    /// 区域
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// 项目 ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
}

/// 自定义 Provider 配置（API Key 方式）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CustomProviderConfig {
    /// 是否启用
    #[serde(default)]
    pub enabled: bool,
    /// API 密钥
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// 基础 URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

/// 路由辅助配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct RoutingConfig {
    /// 模型别名映射
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,
}

fn default_provider() -> String {
    "openai".to_string()
}

/// 重试配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrySettings {
    /// 最大重试次数
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// 基础延迟（毫秒）
    #[serde(default = "default_base_delay_ms")]
    pub base_delay_ms: u64,
    /// 最大延迟（毫秒）
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,
    /// 是否自动切换 Provider
    #[serde(default = "default_auto_switch")]
    pub auto_switch_provider: bool,
}

fn default_max_retries() -> u32 {
    3
}

fn default_base_delay_ms() -> u64 {
    1000
}

fn default_max_delay_ms() -> u64 {
    30000
}

fn default_auto_switch() -> bool {
    true
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            base_delay_ms: default_base_delay_ms(),
            max_delay_ms: default_max_delay_ms(),
            auto_switch_provider: default_auto_switch(),
        }
    }
}

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoggingConfig {
    /// 是否启用日志
    #[serde(default = "default_logging_enabled")]
    pub enabled: bool,
    /// 日志级别
    #[serde(default = "default_log_level")]
    pub level: String,
    /// 日志保留天数
    #[serde(default = "default_retention_days")]
    pub retention_days: u32,
    /// 是否包含请求体
    #[serde(default)]
    pub include_request_body: bool,
}

/// 崩溃上报配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CrashReportingConfig {
    /// 是否启用崩溃上报
    #[serde(default = "default_crash_reporting_enabled")]
    pub enabled: bool,
    /// Sentry DSN（为空时不发送远端）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsn: Option<String>,
    /// 上报环境（如 prod/dev）
    #[serde(default = "default_crash_reporting_environment")]
    pub environment: String,
    /// 采样率（0.0 - 1.0）
    #[serde(default = "default_crash_reporting_sample_rate")]
    pub sample_rate: f64,
    /// 是否发送可能包含 PII 的默认字段
    #[serde(default)]
    pub send_pii: bool,
}

fn default_crash_reporting_enabled() -> bool {
    true
}

fn default_crash_reporting_environment() -> String {
    "production".to_string()
}

fn default_crash_reporting_sample_rate() -> f64 {
    1.0
}

impl Default for CrashReportingConfig {
    fn default() -> Self {
        Self {
            enabled: default_crash_reporting_enabled(),
            dsn: None,
            environment: default_crash_reporting_environment(),
            sample_rate: default_crash_reporting_sample_rate(),
            send_pii: false,
        }
    }
}

fn default_logging_enabled() -> bool {
    true
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_retention_days() -> u32 {
    7
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enabled: default_logging_enabled(),
            level: default_log_level(),
            retention_days: default_retention_days(),
            include_request_body: false,
        }
    }
}

// ============ 模型配置类型 ============

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// 模型 ID
    pub id: String,
    /// 模型显示名称（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 是否启用
    #[serde(default = "default_model_enabled")]
    pub enabled: bool,
}

fn default_model_enabled() -> bool {
    true
}

/// Provider 模型配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderModelsConfig {
    /// Provider 显示标签
    pub label: String,
    /// 模型列表
    #[serde(default)]
    pub models: Vec<ModelInfo>,
}

/// 模型配置（顶层）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelsConfig {
    /// 是否从 models.dev 获取模型列表（预留功能）
    #[serde(default)]
    pub fetch_from_models_dev: bool,
    /// models.dev 缓存 TTL（秒）
    #[serde(default = "default_cache_ttl_secs")]
    pub cache_ttl_secs: u64,
    /// Provider 模型配置
    #[serde(default)]
    pub providers: HashMap<String, ProviderModelsConfig>,
}

fn default_cache_ttl_secs() -> u64 {
    3600
}

impl Default for ModelsConfig {
    fn default() -> Self {
        let mut providers = HashMap::new();

        // Claude (直连 Anthropic API)
        providers.insert(
            "claude".to_string(),
            ProviderModelsConfig {
                label: "Claude".to_string(),
                models: vec![
                    ModelInfo {
                        id: "claude-opus-4-5-20251101".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "claude-sonnet-4-5-20250929".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "claude-sonnet-4-20250514".to_string(),
                        name: None,
                        enabled: true,
                    },
                ],
            },
        );

        // Anthropic (API Key Provider)
        providers.insert(
            "anthropic".to_string(),
            ProviderModelsConfig {
                label: "Anthropic".to_string(),
                models: vec![
                    ModelInfo {
                        id: "claude-opus-4-5-20251101".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "claude-sonnet-4-5-20250929".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "claude-sonnet-4-20250514".to_string(),
                        name: None,
                        enabled: true,
                    },
                ],
            },
        );

        // OpenAI
        providers.insert(
            "openai".to_string(),
            ProviderModelsConfig {
                label: "OpenAI".to_string(),
                models: vec![
                    ModelInfo {
                        id: "gpt-4o".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "gpt-4o-mini".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "gpt-4-turbo".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "o1".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "o1-mini".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "o3".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "o3-mini".to_string(),
                        name: None,
                        enabled: true,
                    },
                ],
            },
        );

        // Gemini
        providers.insert(
            "gemini".to_string(),
            ProviderModelsConfig {
                label: "Gemini".to_string(),
                models: vec![
                    ModelInfo {
                        id: "gemini-2.0-flash-exp".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "gemini-1.5-pro".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "gemini-1.5-flash".to_string(),
                        name: None,
                        enabled: true,
                    },
                ],
            },
        );

        // Qwen
        providers.insert(
            "qwen".to_string(),
            ProviderModelsConfig {
                label: "通义千问".to_string(),
                models: vec![
                    ModelInfo {
                        id: "qwen-max".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "qwen-plus".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "qwen-turbo".to_string(),
                        name: None,
                        enabled: true,
                    },
                ],
            },
        );

        // DeepSeek
        providers.insert(
            "deepseek".to_string(),
            ProviderModelsConfig {
                label: "DeepSeek".to_string(),
                models: vec![
                    ModelInfo {
                        id: "deepseek-reasoner".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "deepseek-chat".to_string(),
                        name: None,
                        enabled: true,
                    },
                ],
            },
        );

        // Codex - 模型列表从别名配置动态加载
        providers.insert(
            "codex".to_string(),
            ProviderModelsConfig {
                label: "Codex".to_string(),
                models: vec![], // 从 aliases/codex.json 动态加载
            },
        );

        // Claude OAuth
        providers.insert(
            "claude_oauth".to_string(),
            ProviderModelsConfig {
                label: "Claude OAuth".to_string(),
                models: vec![
                    ModelInfo {
                        id: "claude-sonnet-4-5-20250929".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "claude-3-5-sonnet-20241022".to_string(),
                        name: None,
                        enabled: true,
                    },
                ],
            },
        );

        // Submodel
        providers.insert(
            "submodel".to_string(),
            ProviderModelsConfig {
                label: "Submodel".to_string(),
                models: vec![
                    ModelInfo {
                        id: "openai/gpt-oss-120b".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "Qwen/Qwen3-235B-A22B-Instruct-2507".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "Qwen/Qwen3-235B-A22B-Thinking-2507".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "deepseek-ai/DeepSeek-R1-0528".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "deepseek-ai/DeepSeek-V3.1".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "deepseek-ai/DeepSeek-V3-0324".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "zai-org/GLM-4.5-FP8".to_string(),
                        name: None,
                        enabled: true,
                    },
                    ModelInfo {
                        id: "zai-org/GLM-4.5-Air".to_string(),
                        name: None,
                        enabled: true,
                    },
                ],
            },
        );

        Self {
            fetch_from_models_dev: false,
            cache_ttl_secs: default_cache_ttl_secs(),
            providers,
        }
    }
}

/// 参数注入配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InjectionSettings {
    /// 是否启用参数注入
    #[serde(default = "default_injection_enabled")]
    pub enabled: bool,
    /// 注入规则列表
    #[serde(default)]
    pub rules: Vec<InjectionRuleConfig>,
}

fn default_injection_enabled() -> bool {
    false
}

impl Default for InjectionSettings {
    fn default() -> Self {
        Self {
            enabled: default_injection_enabled(),
            rules: Vec::new(),
        }
    }
}

/// 注入规则配置（用于 YAML/JSON 序列化）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InjectionRuleConfig {
    /// 规则 ID
    pub id: String,
    /// 模型匹配模式（支持通配符）
    pub pattern: String,
    /// 要注入的参数
    pub parameters: serde_json::Value,
    /// 注入模式
    #[serde(default)]
    pub mode: InjectionMode,
    /// 优先级（数字越小优先级越高）
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// 是否启用
    #[serde(default = "default_rule_enabled")]
    pub enabled: bool,
}

fn default_rule_enabled() -> bool {
    true
}

fn default_priority() -> i32 {
    100
}

impl From<InjectionRuleConfig> for InjectionRule {
    fn from(config: InjectionRuleConfig) -> Self {
        let mut rule = InjectionRule::new(&config.id, &config.pattern, config.parameters);
        rule.mode = config.mode;
        rule.priority = config.priority;
        rule.enabled = config.enabled;
        rule
    }
}

impl From<&InjectionRule> for InjectionRuleConfig {
    fn from(rule: &InjectionRule) -> Self {
        Self {
            id: rule.id.clone(),
            pattern: rule.pattern.clone(),
            parameters: rule.parameters.clone(),
            mode: rule.mode,
            priority: rule.priority,
            enabled: rule.enabled,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            providers: ProvidersConfig::default(),
            default_provider: default_provider(),
            routing: RoutingConfig::default(),
            retry: RetrySettings::default(),
            logging: LoggingConfig::default(),
            injection: InjectionSettings::default(),
            auth_dir: default_auth_dir(),
            credential_pool: CredentialPoolConfig::default(),
            remote_management: RemoteManagementConfig::default(),
            quota_exceeded: QuotaExceededConfig::default(),
            proxy_url: None,
            ampcode: AmpConfig::default(),
            endpoint_providers: EndpointProvidersConfig::default(),
            minimize_to_tray: default_minimize_to_tray(),
            language: default_language(),
            models: ModelsConfig::default(),
            agent: NativeAgentConfig::default(),
            skills: SkillsConfig::default(),
            orchestrator: OrchestratorConfig::default(),
            experimental: ExperimentalFeatures::default(),
            tool_calling: ToolCallingConfig::default(),
            workspace_preferences: WorkspacePreferencesConfig::default(),
            navigation: NavigationConfig::default(),
            chat_appearance: ChatAppearanceConfig::default(),
            environment: EnvironmentConfig::default(),
            web_search: WebSearchConfig::default(),
            memory: MemoryConfig::default(),
            image_gen: ImageGenConfig::default(),
            user_profile: UserProfile::default(),
            developer: DeveloperConfig::default(),
            rate_limit: RateLimitSettings::default(),
            crash_reporting: CrashReportingConfig::default(),
            conversation: ConversationSettings::default(),
            hint_router: HintRouterSettings::default(),
            pairing: PairingSettings::default(),
            automation: AutomationSettings::default(),
            gateway: GatewayConfig::default(),
            channels: ChannelsConfig::default(),
        }
    }
}

impl Config {
    pub fn normalize_local_server_surface(&mut self) -> bool {
        let mut changed = false;

        if self.server.host != default_host() {
            self.server.host = default_host();
            changed = true;
        }

        if self.remote_management.allow_remote {
            self.remote_management.allow_remote = false;
            changed = true;
        }

        changed
    }

    pub fn normalize_workspace_preferences(&mut self) -> bool {
        let mut changed = false;
        let current_version = current_workspace_preferences_schema_version();
        let is_legacy_navigation_schema = self.navigation.schema_version < current_version;

        let normalized_navigation_items = if is_legacy_navigation_schema {
            Vec::new()
        } else {
            normalize_navigation_enabled_items(&self.navigation.enabled_items)
        };
        if self.navigation.enabled_items != normalized_navigation_items {
            self.navigation.enabled_items = normalized_navigation_items;
            changed = true;
        }

        if self.workspace_preferences.schema_version < current_version {
            self.workspace_preferences.schema_version = current_version;
            changed = true;
        }

        if self.navigation.schema_version < current_version {
            self.navigation.schema_version = current_version;
            changed = true;
        }

        changed
    }
}

// ============ 设置页面配置类型 ============

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShellEnvironmentImportConfig {
    /// 是否启用登录 Shell 环境导入
    #[serde(default)]
    pub enabled: bool,
    /// Shell 环境解析超时时间（毫秒）
    #[serde(default = "default_shell_import_timeout_ms")]
    pub timeout_ms: u64,
}

fn default_shell_import_timeout_ms() -> u64 {
    1500
}

impl Default for ShellEnvironmentImportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout_ms: default_shell_import_timeout_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvironmentVariableOverride {
    /// 环境变量名
    #[serde(default)]
    pub key: String,
    /// 环境变量值
    #[serde(default)]
    pub value: String,
    /// 是否启用
    #[serde(default = "default_environment_variable_enabled")]
    pub enabled: bool,
}

fn default_environment_variable_enabled() -> bool {
    true
}

impl Default for EnvironmentVariableOverride {
    fn default() -> Self {
        Self {
            key: String::new(),
            value: String::new(),
            enabled: default_environment_variable_enabled(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct EnvironmentConfig {
    /// Shell 环境导入配置
    #[serde(default)]
    pub shell_import: ShellEnvironmentImportConfig,
    /// 显式环境变量覆盖
    #[serde(default)]
    pub variables: Vec<EnvironmentVariableOverride>,
}

impl EnvironmentConfig {
    pub fn is_default(value: &Self) -> bool {
        value == &Self::default()
    }
}

/// 网络搜索引擎类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchEngine {
    /// Google 搜索（通用网页检索）
    #[default]
    Google,
    /// 小红书搜索（中文生活方式内容）
    Xiaohongshu,
}

/// 联网搜索提供商类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchProvider {
    /// Tavily Search API
    Tavily,
    /// Multi Search Engine v2.0.1
    MultiSearchEngine,
    /// DuckDuckGo Instant Answer API（无需 Key）
    #[default]
    DuckduckgoInstant,
    /// Bing Search API
    BingSearchApi,
    /// Google Custom Search API
    GoogleCustomSearch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultiSearchEngineEntryConfig {
    /// 引擎标识名
    pub name: String,
    /// 搜索 URL 模板，必须包含 {query}
    pub url_template: String,
    /// 是否启用该引擎
    #[serde(default = "default_mse_engine_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultiSearchConfig {
    /// 引擎优先级（按名称）
    #[serde(default)]
    pub priority: Vec<String>,
    /// 自定义/覆盖引擎列表
    #[serde(default = "default_multi_search_engines")]
    pub engines: Vec<MultiSearchEngineEntryConfig>,
    /// 每个引擎最大结果数
    #[serde(default = "default_mse_max_results_per_engine")]
    pub max_results_per_engine: usize,
    /// 最终聚合最大结果数
    #[serde(default = "default_mse_max_total_results")]
    pub max_total_results: usize,
    /// 每个引擎请求超时（毫秒）
    #[serde(default = "default_mse_timeout_ms")]
    pub timeout_ms: u64,
}

impl Default for MultiSearchConfig {
    fn default() -> Self {
        Self {
            priority: vec![],
            engines: default_multi_search_engines(),
            max_results_per_engine: default_mse_max_results_per_engine(),
            max_total_results: default_mse_max_total_results(),
            timeout_ms: default_mse_timeout_ms(),
        }
    }
}

fn default_mse_engine_enabled() -> bool {
    true
}

fn default_mse_max_results_per_engine() -> usize {
    5
}

fn default_mse_max_total_results() -> usize {
    20
}

fn default_mse_timeout_ms() -> u64 {
    4000
}

fn default_multi_search_engines() -> Vec<MultiSearchEngineEntryConfig> {
    vec![
        ("google", "https://www.google.com/search?q={query}"),
        ("bing", "https://www.bing.com/search?q={query}"),
        ("duckduckgo", "https://duckduckgo.com/?q={query}"),
        ("yahoo", "https://search.yahoo.com/search?p={query}"),
        ("baidu", "https://www.baidu.com/s?wd={query}"),
        ("yandex", "https://yandex.com/search/?text={query}"),
        ("ecosia", "https://www.ecosia.org/search?q={query}"),
        ("brave", "https://search.brave.com/search?q={query}"),
        (
            "startpage",
            "https://www.startpage.com/do/search?query={query}",
        ),
        ("qwant", "https://www.qwant.com/?q={query}&t=web"),
        ("sogou", "https://www.sogou.com/web?query={query}"),
        ("so360", "https://www.so.com/s?q={query}"),
        ("aol", "https://search.aol.com/aol/search?q={query}"),
        ("ask", "https://www.ask.com/web?q={query}"),
        (
            "naver",
            "https://search.naver.com/search.naver?query={query}",
        ),
        ("seznam", "https://search.seznam.cz/?q={query}"),
        ("dogpile", "https://www.dogpile.com/serp?q={query}"),
    ]
    .into_iter()
    .map(|(name, url_template)| MultiSearchEngineEntryConfig {
        name: name.to_string(),
        url_template: url_template.to_string(),
        enabled: true,
    })
    .collect()
}

/// 网络搜索配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct WebSearchConfig {
    /// 默认搜索引擎偏好
    #[serde(default)]
    pub engine: SearchEngine,
    /// 联网搜索提供商
    #[serde(default)]
    pub provider: WebSearchProvider,
    /// 提供商回退优先级
    #[serde(default)]
    pub provider_priority: Vec<WebSearchProvider>,
    /// Tavily Search API Key
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tavily_api_key: Option<String>,
    /// Bing Search API Key
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bing_search_api_key: Option<String>,
    /// Google Search API Key
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub google_search_api_key: Option<String>,
    /// Google Search Engine ID (CSE CX)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub google_search_engine_id: Option<String>,
    /// Multi Search Engine 配置
    #[serde(default)]
    pub multi_search: MultiSearchConfig,
}

/// 聊天外观配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ChatAppearanceConfig {
    /// 字体大小 (12-18)
    #[serde(default)]
    pub font_size: Option<i32>,
    /// 消息过渡模式
    #[serde(default)]
    pub transition_mode: Option<String>,
    /// 气泡样式
    #[serde(default)]
    pub bubble_style: Option<String>,
    /// 显示头像
    #[serde(default)]
    pub show_avatar: Option<bool>,
    /// 显示时间戳
    #[serde(default)]
    pub show_timestamp: Option<bool>,
    /// 推荐点击时自动附带当前选中文本上下文
    #[serde(default)]
    pub append_selected_text_to_recommendation: Option<bool>,
}

/// 开发者能力配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DeveloperConfig {
    /// 是否允许在工作区启用处理工作台与信息收集
    #[serde(default)]
    pub workspace_harness_enabled: bool,
}

/// 记忆管理配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MemoryProfileConfig {
    /// 当前学习/工作状态（单选）
    #[serde(default)]
    pub current_status: Option<String>,
    /// 擅长领域（多选）
    #[serde(default)]
    pub strengths: Vec<String>,
    /// 偏好的解释风格（多选）
    #[serde(default)]
    pub explanation_style: Vec<String>,
    /// 遇到难题时的偏好（多选）
    #[serde(default)]
    pub challenge_preference: Vec<String>,
}

/// Soul 导入来源
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemorySoulImportSource {
    /// 用户手动配置
    #[default]
    Manual,
    /// 从 SOUL.md 导入
    SoulMd,
}

/// 正式产物创作声线来源
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemorySoulArtifactVoiceSource {
    /// 使用用户显式配置的创作者声线
    #[default]
    CreatorVoice,
    /// 使用用户显式配置的品牌声线
    BrandVoice,
}

/// 全局交互口吻预设 ID。
///
/// 具体风格由 Style Pack Registry 解析，配置层只保存稳定 ID，不枚举内置 seed。
pub type MemorySoulStyleProfileId = String;

/// 正式产物创作声线配置
///
/// 该配置只作为 Generation Brief 的显式输入，默认不影响正式产物。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MemorySoulArtifactVoiceConfig {
    /// 是否允许本配置进入正式产物 Generation Brief
    #[serde(default)]
    pub enabled: bool,
    /// 声线来源
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_source: Option<MemorySoulArtifactVoiceSource>,
    /// 创作者声线 ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub creator_voice_id: Option<String>,
    /// 品牌声线 ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brand_voice_id: Option<String>,
    /// Evidence Pack ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_pack_id: Option<String>,
    /// 可追踪 evidence 引用
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

/// 全局交互人格配置
///
/// 该配置只表达 Lime 与用户互动时的语气、解释节奏和追问方式。
/// 正式创作声线必须通过 Generation Brief，不从这里默认进入 artifact。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MemorySoulConfig {
    /// 是否启用全局交互人格
    #[serde(default)]
    pub enabled: bool,
    /// 人格名称
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// 一句话风格摘要
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// 交互口吻预设
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style_profile_id: Option<MemorySoulStyleProfileId>,
    /// 语气标签
    #[serde(default)]
    pub tone: Vec<String>,
    /// 沟通方式
    #[serde(default)]
    pub communication_style: Vec<String>,
    /// 解释深度
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation_depth: Option<String>,
    /// 遇到弱假设或分歧时的处理方式
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub challenge_style: Option<String>,
    /// 避免事项
    #[serde(default)]
    pub avoid: Vec<String>,
    /// 正式产物创作声线配置
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_voice: Option<MemorySoulArtifactVoiceConfig>,
    /// 导入来源
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_from: Option<MemorySoulImportSource>,
    /// 更新时间
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

/// 记忆来源配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemorySourcesConfig {
    /// 组织级策略文件（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub managed_policy_path: Option<String>,
    /// 项目级记忆文件相对路径列表
    ///
    /// 默认优先使用标准 `AGENTS.md` / `AGENTS.override.md`，缺失时回退 `.lime/AGENTS.md`；
    /// 解析当前 workspace 根目录到工作目录的目录层级。
    #[serde(default)]
    pub project_memory_paths: Vec<String>,
    /// 项目规则目录相对路径列表（会按目录层级向上查找）
    #[serde(default)]
    pub project_rule_dirs: Vec<String>,
    /// 用户级记忆文件（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_memory_path: Option<String>,
}

impl Default for MemorySourcesConfig {
    fn default() -> Self {
        Self {
            managed_policy_path: None,
            project_memory_paths: vec![".lime/AGENTS.md".to_string()],
            project_rule_dirs: vec![".agents/rules".to_string()],
            user_memory_path: None,
        }
    }
}

/// 自动记忆配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryAutoConfig {
    /// 是否启用自动记忆
    #[serde(default = "default_memory_auto_enabled")]
    pub enabled: bool,
    /// MEMORY 入口文件名
    #[serde(default = "default_memory_auto_entrypoint")]
    pub entrypoint: String,
    /// 启动时加载 MEMORY 入口的最大行数
    #[serde(default = "default_memory_auto_max_loaded_lines")]
    pub max_loaded_lines: u32,
    /// 自动记忆根目录（可选，默认位于应用数据目录下的 projects/<project>/memory）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_dir: Option<String>,
}

fn default_memory_auto_enabled() -> bool {
    true
}

fn default_memory_auto_entrypoint() -> String {
    "MEMORY.md".to_string()
}

fn default_memory_auto_max_loaded_lines() -> u32 {
    200
}

impl Default for MemoryAutoConfig {
    fn default() -> Self {
        Self {
            enabled: default_memory_auto_enabled(),
            entrypoint: default_memory_auto_entrypoint(),
            max_loaded_lines: default_memory_auto_max_loaded_lines(),
            root_dir: None,
        }
    }
}

/// 记忆解析行为配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryResolveConfig {
    /// 额外参与记忆解析的目录
    #[serde(default)]
    pub additional_dirs: Vec<String>,
    /// 是否跟随 @import 引用
    #[serde(default = "default_memory_follow_imports")]
    pub follow_imports: bool,
    /// 最大导入深度
    #[serde(default = "default_memory_import_max_depth")]
    pub import_max_depth: u8,
    /// 是否从 additional_dirs 加载记忆文件
    #[serde(default)]
    pub load_additional_dirs_memory: bool,
}

fn default_memory_follow_imports() -> bool {
    true
}

fn default_memory_import_max_depth() -> u8 {
    5
}

impl Default for MemoryResolveConfig {
    fn default() -> Self {
        Self {
            additional_dirs: Vec::new(),
            follow_imports: default_memory_follow_imports(),
            import_max_depth: default_memory_import_max_depth(),
            load_additional_dirs_memory: false,
        }
    }
}

/// 记忆嵌入模型提供方式
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEmbeddingProvider {
    /// 自动选择可用提供方：内置模型优先，其次 OpenAI API
    #[default]
    Auto,
    /// 本地 ONNX 嵌入模型
    LocalOnnx,
    /// Lime 内置模型通道
    Builtin,
    /// OpenAI API Key Provider
    OpenaiApi,
    /// 指定 API Key Provider
    Provider,
    /// 关闭向量搜索，仅保留全文检索
    Disabled,
}

/// 记忆嵌入模型配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryEmbeddingConfig {
    /// 嵌入模型提供方式
    #[serde(default)]
    pub provider: MemoryEmbeddingProvider,
    /// 指定 Provider ID，仅 provider 模式使用
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_id: Option<String>,
    /// 嵌入模型 ID
    #[serde(default = "default_memory_embedding_model")]
    pub model: String,
}

fn default_memory_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

impl Default for MemoryEmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: MemoryEmbeddingProvider::Auto,
            provider_id: None,
            model: default_memory_embedding_model(),
        }
    }
}

/// 记忆管理配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MemoryConfig {
    /// 是否启用记忆功能
    #[serde(default)]
    pub enabled: bool,
    /// 最大记忆条数
    #[serde(default)]
    pub max_entries: Option<u32>,
    /// 记忆保留天数
    #[serde(default)]
    pub retention_days: Option<u32>,
    /// 自动清理过期记忆
    #[serde(default)]
    pub auto_cleanup: Option<bool>,
    /// 记忆偏好画像
    #[serde(default)]
    pub profile: Option<MemoryProfileConfig>,
    /// 全局交互人格
    #[serde(default)]
    pub soul: Option<MemorySoulConfig>,
    /// 记忆来源配置
    #[serde(default)]
    pub sources: MemorySourcesConfig,
    /// 自动记忆配置
    #[serde(default)]
    pub auto: MemoryAutoConfig,
    /// 记忆解析行为配置
    #[serde(default)]
    pub resolve: MemoryResolveConfig,
    /// 向量嵌入模型配置
    #[serde(default)]
    pub embedding: MemoryEmbeddingConfig,
}

/// 图像生成服务配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ImageGenConfig {
    /// 默认图像生成服务
    #[serde(default)]
    pub default_service: Option<String>,
    /// 默认图像数量
    #[serde(default)]
    pub default_count: Option<u32>,
    /// 默认图像尺寸
    #[serde(default)]
    pub default_size: Option<String>,
    /// 默认图像质量
    #[serde(default)]
    pub default_quality: Option<String>,
    /// 默认图像风格
    #[serde(default)]
    pub default_style: Option<String>,
    /// 启用图像增强
    #[serde(default)]
    pub enable_enhancement: Option<bool>,
    /// 自动下载生成的图像
    #[serde(default)]
    pub auto_download: Option<bool>,
    /// 图片搜索（Pexels）API Key
    #[serde(default)]
    pub image_search_pexels_api_key: Option<String>,
    /// 图片搜索（Pixabay）API Key
    #[serde(default)]
    pub image_search_pixabay_api_key: Option<String>,
}

/// 用户资料
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct UserProfile {
    /// 用户头像 URL
    #[serde(default)]
    pub avatar_url: Option<String>,
    /// 昵称
    #[serde(default)]
    pub nickname: Option<String>,
    /// 个人简介
    #[serde(default)]
    pub bio: Option<String>,
    /// 邮箱
    #[serde(default)]
    pub email: Option<String>,
    /// 偏好标签
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

#[cfg(test)]
mod unit_tests {
    use super::super::tool_execution::{
        ToolExecutionCommandRiskLevelConfig, ToolExecutionCommandRuleConfig,
        ToolExecutionCommandRuleMatchTypeConfig, ToolExecutionNetworkRuleConfig,
        ToolExecutionNetworkRuleTargetConfig, ToolExecutionOverrideConfig,
        ToolExecutionRestrictionProfileConfig, ToolExecutionSandboxProfileConfig,
        ToolExecutionWarningPolicyConfig,
    };
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8999);
        assert_eq!(config.server.api_key, "proxy_cast");
        assert!(config.server.response_cache.enabled);
        assert_eq!(
            config.server.response_cache.cacheable_status_codes,
            vec![200]
        );
        assert!(!config.providers.kiro.enabled);
        assert!(config.providers.kiro.credentials_path.is_none());
        assert!(!config.providers.gemini.enabled);
        assert_eq!(config.default_provider, "openai");
        assert_eq!(config.retry.max_retries, 3);
        assert!(config.logging.enabled);
        assert!(!config.injection.enabled);
        assert!(config.injection.rules.is_empty());
        // 新增字段测试
        assert_eq!(config.auth_dir, "~/.lime/auth");
        assert!(config.credential_pool.kiro.is_empty());
        assert!(config.credential_pool.openai.is_empty());
        assert!(config.crash_reporting.enabled);
        assert!(config.crash_reporting.dsn.is_none());
        assert_eq!(config.crash_reporting.environment, "production");
        assert_eq!(config.crash_reporting.sample_rate, 1.0);
        assert!(!config.crash_reporting.send_pii);
        assert_eq!(
            config.workspace_preferences.schema_version,
            current_workspace_preferences_schema_version()
        );
        assert_eq!(
            config.navigation.schema_version,
            current_workspace_preferences_schema_version()
        );
        assert!(config.navigation.enabled_items.is_empty());
        assert!(config.agent.tool_execution.tool_overrides.is_empty());
        assert!(config.orchestrator.skills.enabled);
        assert!(config.orchestrator.mcp.enabled);
    }

    #[test]
    fn orchestrator_features_default_enabled_and_preserve_partial_override() {
        let defaults: OrchestratorConfig =
            serde_json::from_value(serde_json::json!({})).expect("orchestrator defaults");
        assert!(defaults.skills.enabled);
        assert!(defaults.mcp.enabled);

        let partial: OrchestratorConfig = serde_json::from_value(serde_json::json!({
            "skills": { "enabled": false }
        }))
        .expect("orchestrator partial override");
        assert!(!partial.skills.enabled);
        assert!(partial.mcp.enabled);

        let serialized = serde_json::to_value(Config::default()).expect("serialize default config");
        assert!(serialized.get("orchestrator").is_none());
    }

    #[test]
    fn memory_soul_style_profile_accepts_registry_profile_id() {
        let yaml = r#"
memory:
  soul:
    enabled: true
    style_profile_id: cheeky_sassy_executor
"#;

        let config: Config = serde_yaml::from_str(yaml).expect("style profile id");

        assert_eq!(
            config.memory.soul.and_then(|soul| soul.style_profile_id),
            Some("cheeky_sassy_executor".to_string())
        );
    }

    #[test]
    fn memory_soul_style_profile_does_not_alias_legacy_profile_id() {
        let yaml = r#"
memory:
  soul:
    enabled: true
    style_profile_id: sassy_cute_executor
"#;

        let config: Config = serde_yaml::from_str(yaml).expect("style profile id");

        assert_eq!(
            config.memory.soul.and_then(|soul| soul.style_profile_id),
            Some("sassy_cute_executor".to_string())
        );
    }

    #[test]
    fn test_tool_execution_policy_config_supports_camel_case_runtime_shape() {
        let value = serde_json::json!({
            "toolOverrides": {
                "bash": {
                    "warningPolicy": "none",
                    "restrictionProfile": "workspace_path_required",
                    "sandboxProfile": "none"
                },
                "Task": {
                    "warning_policy": "shell_command_risk"
                }
            },
            "shellCommandRules": [
                {
                    "ruleId": "block_release_publish",
                    "matchType": "prefix",
                    "pattern": "cargo publish",
                    "riskLevel": "high",
                    "reasonCode": "release_publish_command",
                    "reason": "命令会发布 crate"
                }
            ],
            "networkRules": [
                {
                    "ruleId": "block_internal_download",
                    "matchType": "prefix",
                    "target": "host",
                    "pattern": "internal.",
                    "riskLevel": "high",
                    "reasonCode": "internal_network_target",
                    "reason": "请求访问内部网络域名"
                }
            ]
        });

        let parsed: ToolExecutionPolicyConfig =
            serde_json::from_value(value).expect("tool execution config should deserialize");

        assert_eq!(
            parsed
                .tool_overrides
                .get("bash")
                .and_then(|item| item.warning_policy),
            Some(ToolExecutionWarningPolicyConfig::None)
        );
        assert_eq!(
            parsed
                .tool_overrides
                .get("bash")
                .and_then(|item| item.restriction_profile),
            Some(ToolExecutionRestrictionProfileConfig::WorkspacePathRequired)
        );
        assert_eq!(
            parsed
                .tool_overrides
                .get("Task")
                .and_then(|item| item.warning_policy),
            Some(ToolExecutionWarningPolicyConfig::ShellCommandRisk)
        );
        assert_eq!(parsed.shell_command_rules.len(), 1);
        assert_eq!(
            parsed.shell_command_rules[0].rule_id,
            "block_release_publish"
        );
        assert_eq!(
            parsed.shell_command_rules[0].match_type,
            ToolExecutionCommandRuleMatchTypeConfig::Prefix
        );
        assert_eq!(
            parsed.shell_command_rules[0].risk_level,
            ToolExecutionCommandRiskLevelConfig::High
        );
        assert_eq!(parsed.network_rules.len(), 1);
        assert_eq!(parsed.network_rules[0].rule_id, "block_internal_download");
        assert_eq!(
            parsed.network_rules[0].target,
            ToolExecutionNetworkRuleTargetConfig::Host
        );
        assert_eq!(
            parsed.network_rules[0].match_type,
            ToolExecutionCommandRuleMatchTypeConfig::Prefix
        );
    }

    #[test]
    fn test_tool_execution_policy_config_roundtrip_preserves_non_default_policy() {
        let config = ToolExecutionPolicyConfig {
            update_plan_enabled: true,
            tool_overrides: HashMap::from([(
                "bash".to_string(),
                ToolExecutionOverrideConfig {
                    warning_policy: Some(ToolExecutionWarningPolicyConfig::None),
                    restriction_profile: Some(
                        ToolExecutionRestrictionProfileConfig::WorkspacePathRequired,
                    ),
                    sandbox_profile: Some(ToolExecutionSandboxProfileConfig::None),
                },
            )]),
            shell_command_rules: vec![ToolExecutionCommandRuleConfig {
                rule_id: "block_release_publish".to_string(),
                match_type: ToolExecutionCommandRuleMatchTypeConfig::Prefix,
                pattern: "cargo publish".to_string(),
                risk_level: ToolExecutionCommandRiskLevelConfig::High,
                reason_code: "release_publish_command".to_string(),
                reason: "命令会发布 crate".to_string(),
            }],
            network_rules: vec![ToolExecutionNetworkRuleConfig {
                rule_id: "block_internal_download".to_string(),
                match_type: ToolExecutionCommandRuleMatchTypeConfig::Prefix,
                target: ToolExecutionNetworkRuleTargetConfig::Host,
                pattern: "internal.".to_string(),
                risk_level: ToolExecutionCommandRiskLevelConfig::High,
                reason_code: "internal_network_target".to_string(),
                reason: "请求访问内部网络域名".to_string(),
            }],
        };

        let value = serde_json::to_value(&config).expect("tool execution config should serialize");
        let parsed: ToolExecutionPolicyConfig =
            serde_json::from_value(value).expect("tool execution config should deserialize");

        assert_eq!(parsed, config);
    }

    #[test]
    fn test_tool_execution_policy_config_omits_disabled_update_plan_by_default() {
        let value = serde_json::to_value(ToolExecutionPolicyConfig::default())
            .expect("default tool execution config should serialize");
        assert_eq!(value, serde_json::json!({}));

        let enabled = ToolExecutionPolicyConfig {
            update_plan_enabled: true,
            ..Default::default()
        };
        let value = serde_json::to_value(enabled).expect("enabled update_plan should serialize");
        assert_eq!(value["update_plan_enabled"], true);
    }

    #[test]
    fn test_normalize_workspace_preferences_upgrades_legacy_defaults() {
        let mut config = Config::default();
        config.workspace_preferences.schema_version = 0;
        config.navigation.schema_version = 0;
        config.navigation.enabled_items = vec![
            "home-general".to_string(),
            "claw".to_string(),
            "video".to_string(),
            "image-gen".to_string(),
        ];

        let changed = config.normalize_workspace_preferences();

        assert!(changed);
        assert_eq!(
            config.workspace_preferences.schema_version,
            current_workspace_preferences_schema_version()
        );
        assert_eq!(
            config.navigation.schema_version,
            current_workspace_preferences_schema_version()
        );
        assert!(config.navigation.enabled_items.is_empty());
    }

    #[test]
    fn test_normalize_workspace_preferences_clears_legacy_companion_default() {
        let mut config = Config::default();
        config.workspace_preferences.schema_version = 0;
        config.navigation.schema_version = 0;
        config.navigation.enabled_items = vec!["plugins".to_string(), "companion".to_string()];

        let changed = config.normalize_workspace_preferences();

        assert!(changed);
        assert_eq!(
            config.workspace_preferences.schema_version,
            current_workspace_preferences_schema_version()
        );
        assert_eq!(
            config.navigation.schema_version,
            current_workspace_preferences_schema_version()
        );
        assert!(config.navigation.enabled_items.is_empty());
    }

    #[test]
    fn test_normalize_workspace_preferences_clears_current_custom_values() {
        let mut config = Config::default();
        config.navigation.enabled_items = vec!["plugins".to_string(), "companion".to_string()];

        let changed = config.normalize_workspace_preferences();

        assert!(changed);
        assert!(config.navigation.enabled_items.is_empty());
    }

    #[test]
    fn test_normalize_workspace_preferences_clears_stale_sidebar_defaults_on_current_schema() {
        let mut config = Config::default();
        config.navigation.enabled_items = vec![
            "home-general".to_string(),
            "claw".to_string(),
            "video".to_string(),
            "image-gen".to_string(),
            "automation".to_string(),
            "resources".to_string(),
            "memory".to_string(),
        ];

        let changed = config.normalize_workspace_preferences();

        assert!(changed);
        assert_eq!(config.navigation.enabled_items, Vec::<String>::new());
    }

    #[test]
    fn test_normalize_workspace_preferences_filters_legacy_fixed_and_unknown_entries() {
        let mut config = Config::default();
        config.navigation.enabled_items = vec![
            "plugins".to_string(),
            "home-general".to_string(),
            "api-server".to_string(),
            "companion".to_string(),
            "plugins".to_string(),
            "settings".to_string(),
        ];

        let changed = config.normalize_workspace_preferences();

        assert!(changed);
        assert!(config.navigation.enabled_items.is_empty());
    }

    #[test]
    fn test_workspace_preferences_supports_agent_response_language_roundtrip() {
        let mut config = Config::default();
        config.workspace_preferences.agent_response_language = Some("en-US".to_string());

        let value =
            serde_json::to_value(&config).expect("config should serialize response language");
        let parsed: Config =
            serde_json::from_value(value).expect("config should deserialize response language");

        assert_eq!(
            parsed
                .workspace_preferences
                .agent_response_language
                .as_deref(),
            Some("en-US")
        );
    }

    #[test]
    fn test_workspace_preferences_supports_service_models_roundtrip() {
        let mut config = Config::default();
        config
            .workspace_preferences
            .service_models
            .responsive_chat
            .preferred_provider_id = Some("gemini".to_string());
        config
            .workspace_preferences
            .service_models
            .responsive_chat
            .preferred_model_id = Some("gemini-2.5-flash".to_string());
        config
            .workspace_preferences
            .service_models
            .topic
            .preferred_provider_id = Some("openai".to_string());
        config
            .workspace_preferences
            .service_models
            .topic
            .preferred_model_id = Some("gpt-5.4-mini".to_string());
        config
            .workspace_preferences
            .service_models
            .prompt_rewrite
            .enabled = false;
        config
            .workspace_preferences
            .service_models
            .resource_prompt_rewrite
            .custom_prompt = Some("请优先按资料库上下文重写提示词".to_string());

        let value = serde_json::to_value(&config)
            .expect("config should serialize service model preferences");
        let parsed: Config = serde_json::from_value(value)
            .expect("config should deserialize service model preferences");

        assert_eq!(
            parsed
                .workspace_preferences
                .service_models
                .responsive_chat
                .preferred_provider_id
                .as_deref(),
            Some("gemini")
        );
        assert_eq!(
            parsed
                .workspace_preferences
                .service_models
                .responsive_chat
                .preferred_model_id
                .as_deref(),
            Some("gemini-2.5-flash")
        );
        assert_eq!(
            parsed
                .workspace_preferences
                .service_models
                .topic
                .preferred_provider_id
                .as_deref(),
            Some("openai")
        );
        assert_eq!(
            parsed
                .workspace_preferences
                .service_models
                .topic
                .preferred_model_id
                .as_deref(),
            Some("gpt-5.4-mini")
        );
        assert!(
            !parsed
                .workspace_preferences
                .service_models
                .prompt_rewrite
                .enabled
        );
        assert_eq!(
            parsed
                .workspace_preferences
                .service_models
                .resource_prompt_rewrite
                .custom_prompt
                .as_deref(),
            Some("请优先按资料库上下文重写提示词")
        );
    }

    #[test]
    fn test_normalize_local_server_surface_forces_loopback_and_disables_remote_management() {
        let mut config = Config::default();
        config.server.host = "192.168.1.9".to_string();
        config.remote_management.allow_remote = true;

        let changed = config.normalize_local_server_surface();

        assert!(changed);
        assert_eq!(config.server.host, "127.0.0.1");
        assert!(!config.remote_management.allow_remote);
    }

    #[test]
    fn test_credential_pool_config_default() {
        let pool = CredentialPoolConfig::default();
        assert!(pool.kiro.is_empty());
        assert!(pool.gemini.is_empty());
        assert!(pool.qwen.is_empty());
        assert!(pool.openai.is_empty());
        assert!(pool.claude.is_empty());
    }

    #[test]
    fn test_credential_entry_serialization() {
        let entry = CredentialEntry {
            id: "kiro-main".to_string(),
            token_file: "kiro/main-token.json".to_string(),
            disabled: false,
            proxy_url: None,
        };
        let yaml = serde_yaml::to_string(&entry).unwrap();
        assert!(yaml.contains("id: kiro-main"));
        assert!(yaml.contains("token_file: kiro/main-token.json"));

        let parsed: CredentialEntry = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_api_key_entry_serialization() {
        let entry = ApiKeyEntry {
            id: "openai-main".to_string(),
            api_key: "sk-test-key".to_string(),
            base_url: Some("https://api.openai.com/v1".to_string()),
            disabled: false,
            proxy_url: None,
        };
        let yaml = serde_yaml::to_string(&entry).unwrap();
        assert!(yaml.contains("id: openai-main"));
        assert!(yaml.contains("api_key: sk-test-key"));

        let parsed: ApiKeyEntry = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_api_key_entry_without_base_url() {
        let entry = ApiKeyEntry {
            id: "claude-main".to_string(),
            api_key: "sk-ant-test".to_string(),
            base_url: None,
            disabled: true,
            proxy_url: None,
        };
        let yaml = serde_yaml::to_string(&entry).unwrap();
        // base_url should be skipped when None
        assert!(!yaml.contains("base_url"));
        assert!(yaml.contains("disabled: true"));

        let parsed: ApiKeyEntry = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_credential_pool_config_serialization() {
        let pool = CredentialPoolConfig {
            kiro: vec![CredentialEntry {
                id: "kiro-1".to_string(),
                token_file: "kiro/token-1.json".to_string(),
                disabled: false,
                proxy_url: None,
            }],
            gemini: vec![],
            qwen: vec![],
            openai: vec![ApiKeyEntry {
                id: "openai-1".to_string(),
                api_key: "sk-xxx".to_string(),
                base_url: None,
                disabled: false,
                proxy_url: None,
            }],
            claude: vec![],
            gemini_api_keys: vec![],
            vertex_api_keys: vec![],
            codex: vec![],
        };

        let yaml = serde_yaml::to_string(&pool).unwrap();
        // Empty vecs should be skipped
        assert!(!yaml.contains("gemini"));
        assert!(!yaml.contains("qwen"));
        assert!(!yaml.contains("claude"));
        assert!(yaml.contains("kiro"));
        assert!(yaml.contains("openai"));

        let parsed: CredentialPoolConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, pool);
    }

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8999);
        assert_eq!(config.api_key, "proxy_cast");
        assert!(config.response_cache.enabled);
        assert_eq!(config.response_cache.ttl_secs, 600);
        assert_eq!(config.response_cache.max_entries, 200);
        assert_eq!(config.response_cache.max_body_bytes, 1_048_576);
        assert_eq!(config.response_cache.cacheable_status_codes, vec![200]);
    }

    #[test]
    fn test_response_cache_settings_default() {
        let config = ResponseCacheSettings::default();
        assert!(config.enabled);
        assert_eq!(config.ttl_secs, 600);
        assert_eq!(config.max_entries, 200);
        assert_eq!(config.max_body_bytes, 1_048_576);
        assert_eq!(config.cacheable_status_codes, vec![200]);
    }

    #[test]
    fn test_retry_settings_default() {
        let config = RetrySettings::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay_ms, 1000);
        assert_eq!(config.max_delay_ms, 30000);
        assert!(config.auto_switch_provider);
    }

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.level, "info");
        assert_eq!(config.retention_days, 7);
        assert!(!config.include_request_body);
    }

    #[test]
    fn test_routing_config_default() {
        let config = RoutingConfig::default();
        assert!(config.model_aliases.is_empty());
    }

    #[test]
    fn test_endpoint_providers_config_default() {
        let config = EndpointProvidersConfig::default();
        assert!(config.cursor.is_none());
        assert!(config.claude_code.is_none());
        assert!(config.codex.is_none());
        assert!(config.windsurf.is_none());
        assert!(config.kiro.is_none());
        assert!(config.other.is_none());
    }

    #[test]
    fn test_endpoint_providers_config_get_provider() {
        let config = EndpointProvidersConfig {
            cursor: Some("qwen".to_string()),
            claude_code: Some("kiro".to_string()),
            codex: Some("codex".to_string()),
            windsurf: None,
            kiro: Some("gemini".to_string()),
            other: None,
        };

        assert_eq!(config.get_provider("cursor"), Some(&"qwen".to_string()));
        assert_eq!(
            config.get_provider("claude_code"),
            Some(&"kiro".to_string())
        );
        assert_eq!(config.get_provider("codex"), Some(&"codex".to_string()));
        assert_eq!(config.get_provider("windsurf"), None);
        assert_eq!(config.get_provider("kiro"), Some(&"gemini".to_string()));
        assert_eq!(config.get_provider("other"), None);
        assert_eq!(config.get_provider("invalid"), None);
    }

    #[test]
    fn test_endpoint_providers_config_set_provider() {
        let mut config = EndpointProvidersConfig::default();

        // 设置有效的客户端类型
        assert!(config.set_provider("cursor", Some("qwen".to_string())));
        assert_eq!(config.cursor, Some("qwen".to_string()));

        assert!(config.set_provider("claude_code", Some("kiro".to_string())));
        assert_eq!(config.claude_code, Some("kiro".to_string()));

        assert!(config.set_provider("codex", Some("codex".to_string())));
        assert_eq!(config.codex, Some("codex".to_string()));

        assert!(config.set_provider("windsurf", Some("gemini".to_string())));
        assert_eq!(config.windsurf, Some("gemini".to_string()));

        assert!(config.set_provider("kiro", Some("openai".to_string())));
        assert_eq!(config.kiro, Some("openai".to_string()));

        assert!(config.set_provider("other", Some("claude".to_string())));
        assert_eq!(config.other, Some("claude".to_string()));

        // 设置无效的客户端类型
        assert!(!config.set_provider("invalid", Some("test".to_string())));
    }

    #[test]
    fn test_endpoint_providers_config_set_provider_clear() {
        let mut config = EndpointProvidersConfig {
            cursor: Some("qwen".to_string()),
            claude_code: Some("kiro".to_string()),
            codex: None,
            windsurf: None,
            kiro: None,
            other: None,
        };

        // 使用 None 清除配置
        assert!(config.set_provider("cursor", None));
        assert_eq!(config.cursor, None);

        // 使用空字符串清除配置
        assert!(config.set_provider("claude_code", Some("".to_string())));
        assert_eq!(config.claude_code, None);
    }

    #[test]
    fn test_endpoint_providers_config_serialization() {
        let config = EndpointProvidersConfig {
            cursor: Some("qwen".to_string()),
            claude_code: Some("kiro".to_string()),
            codex: None,
            windsurf: None,
            kiro: None,
            other: None,
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("cursor: qwen"));
        assert!(yaml.contains("claude_code: kiro"));
        // None 值应该被跳过
        assert!(!yaml.contains("codex"));
        assert!(!yaml.contains("windsurf"));
        assert!(!yaml.contains("kiro:"));
        assert!(!yaml.contains("other"));

        let parsed: EndpointProvidersConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn test_endpoint_providers_config_json_serialization() {
        let config = EndpointProvidersConfig {
            cursor: Some("qwen".to_string()),
            claude_code: None,
            codex: Some("codex".to_string()),
            windsurf: None,
            kiro: None,
            other: Some("openai".to_string()),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: EndpointProvidersConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn test_webmcp_config_default() {
        let config = WebMcpConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_experimental_features_default() {
        let config = ExperimentalFeatures::default();
        assert!(!config.webmcp.enabled);
    }

    #[test]
    fn test_experimental_features_serialization() {
        let config = ExperimentalFeatures {
            webmcp: WebMcpConfig { enabled: true },
            ..Default::default()
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("webmcp"));
        assert!(yaml.contains("enabled: true"));
        assert!(!yaml.contains("screenshot_chat"));

        let parsed: ExperimentalFeatures = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn test_tool_calling_config_default() {
        let config = ToolCallingConfig::default();
        assert!(config.enabled);
        assert!(config.dynamic_filtering);
        assert!(!config.native_input_examples);
    }

    #[test]
    fn test_tool_calling_config_serialization() {
        let config = ToolCallingConfig {
            enabled: false,
            dynamic_filtering: false,
            native_input_examples: true,
        };

        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("enabled: false"));
        assert!(yaml.contains("dynamic_filtering: false"));
        assert!(yaml.contains("native_input_examples: true"));

        let parsed: ToolCallingConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn test_developer_config_default() {
        let config = DeveloperConfig::default();
        assert!(!config.workspace_harness_enabled);
    }

    #[test]
    fn test_config_with_experimental() {
        let config = Config::default();
        assert!(!config.experimental.webmcp.enabled);
        // 语音输入测试
        assert!(!config.experimental.voice_input.enabled);
        assert_eq!(
            config.experimental.voice_input.shortcut,
            "CommandOrControl+Shift+V"
        );
        assert!(!config.developer.workspace_harness_enabled);
        assert!(config.tool_calling.enabled);
        assert!(config.tool_calling.dynamic_filtering);
        assert!(!config.tool_calling.native_input_examples);
    }

    #[test]
    fn test_asr_credential_entry_serialization() {
        let entry = AsrCredentialEntry {
            id: "whisper-local".to_string(),
            provider: AsrProviderType::WhisperLocal,
            name: Some("本地 Whisper".to_string()),
            is_default: true,
            disabled: false,
            language: "zh".to_string(),
            whisper_config: Some(WhisperLocalConfig {
                model: WhisperModelSize::Base,
                model_path: None,
            }),
            sensevoice_config: None,
            xunfei_config: None,
            baidu_config: None,
            openai_config: None,
        };
        let yaml = serde_yaml::to_string(&entry).unwrap();
        assert!(yaml.contains("provider: whisper_local"));
        assert!(yaml.contains("is_default: true"));

        let parsed: AsrCredentialEntry = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn test_voice_input_config_default() {
        let config = VoiceInputConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.shortcut, "CommandOrControl+Shift+V");
        assert!(config.processor.polish_enabled);
        assert_eq!(config.processor.default_instruction_id, "default");
        assert_eq!(config.output.mode, VoiceOutputMode::Type);
        assert!(!config.instructions.is_empty());
    }

    #[test]
    fn test_voice_instruction_serialization() {
        let instruction = VoiceInstruction {
            id: "custom".to_string(),
            name: "自定义指令".to_string(),
            description: Some("测试指令".to_string()),
            prompt: "处理: {{text}}".to_string(),
            shortcut: Some("CommandOrControl+1".to_string()),
            is_preset: false,
            icon: None,
        };
        let yaml = serde_yaml::to_string(&instruction).unwrap();
        assert!(yaml.contains("id: custom"));
        assert!(yaml.contains("{{text}}"));

        let parsed: VoiceInstruction = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed, instruction);
    }

    #[test]
    fn test_voice_input_with_asr_credentials() {
        let voice_input = VoiceInputConfig {
            asr_credentials: vec![AsrCredentialEntry {
                id: "xunfei-1".to_string(),
                provider: AsrProviderType::Xunfei,
                name: Some("讯飞语音".to_string()),
                is_default: false,
                disabled: false,
                language: "zh".to_string(),
                whisper_config: None,
                sensevoice_config: None,
                xunfei_config: Some(XunfeiConfig {
                    app_id: "test_app_id".to_string(),
                    api_key: "test_api_key".to_string(),
                    api_secret: "test_api_secret".to_string(),
                }),
                baidu_config: None,
                openai_config: None,
            }],
            ..VoiceInputConfig::default()
        };

        let yaml = serde_yaml::to_string(&voice_input).unwrap();
        assert!(yaml.contains("asr_credentials:"));
        assert!(yaml.contains("provider: xunfei"));

        let parsed: VoiceInputConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.asr_credentials.len(), 1);
        assert_eq!(parsed.asr_credentials[0].provider, AsrProviderType::Xunfei);
    }
}

// ============ 心跳引擎配置类型 ============

/// 任务调度类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TaskSchedule {
    /// 固定间隔（现有行为）
    Every { every_secs: u64 },
    /// Cron 表达式
    Cron {
        expr: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tz: Option<String>,
    },
    /// 指定时间点（一次性，RFC3339 格式）
    At { at: String },
}

impl Default for TaskSchedule {
    fn default() -> Self {
        Self::Every { every_secs: 300 }
    }
}

/// 自动化输出/通知投递配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeliveryConfig {
    /// 投递模式: "none" | "announce"
    #[serde(default = "default_delivery_mode")]
    pub mode: String,
    /// 投递渠道: "webhook" | "telegram" | ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    /// 目标地址: URL 或 chat_id
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// 投递失败是否算任务失败
    #[serde(default)]
    pub best_effort: bool,
    /// 输出契约，留空时按旧任务结果自动推断
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<String>,
    /// 输出格式: "text" | "json"
    #[serde(default = "default_delivery_output_format")]
    pub output_format: String,
}

fn default_delivery_mode() -> String {
    "none".to_string()
}

fn default_delivery_output_format() -> String {
    "text".to_string()
}

impl Default for DeliveryConfig {
    fn default() -> Self {
        Self {
            mode: default_delivery_mode(),
            channel: None,
            target: None,
            best_effort: true,
            output_schema: None,
            output_format: default_delivery_output_format(),
        }
    }
}

/// 自动化执行模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AutomationExecutionMode {
    /// 智能模式：通过 AI Agent 执行任务
    #[default]
    Intelligent,
    /// 技能模式：调用已注册的技能
    Skill,
    /// 日志模式：仅记录任务，不执行
    LogOnly,
}

/// 自动化调度配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutomationSettings {
    /// 是否启用自动化调度
    #[serde(default)]
    pub enabled: bool,
    /// 后台轮询间隔（秒）
    #[serde(default = "default_automation_poll_interval")]
    pub poll_interval_secs: u64,
    /// 是否启用运行历史
    #[serde(default = "default_enable_history")]
    pub enable_history: bool,
}

fn default_automation_poll_interval() -> u64 {
    30
}
fn default_enable_history() -> bool {
    true
}

impl Default for AutomationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_interval_secs: default_automation_poll_interval(),
            enable_history: true,
        }
    }
}

// ============ 安全与性能配置类型 ============

/// 速率限制配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RateLimitSettings {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_requests_per_minute")]
    pub requests_per_minute: u32,
    #[serde(default = "default_window_secs")]
    pub window_secs: u64,
}

fn default_requests_per_minute() -> u32 {
    60
}
fn default_window_secs() -> u64 {
    60
}

impl Default for RateLimitSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_minute: 60,
            window_secs: 60,
        }
    }
}

/// 对话管理配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConversationSettings {
    #[serde(default)]
    pub trim_enabled: bool,
    #[serde(default = "default_max_messages")]
    pub max_messages: usize,
    #[serde(default)]
    pub summary_enabled: bool,
}

fn default_max_messages() -> usize {
    50
}

impl Default for ConversationSettings {
    fn default() -> Self {
        Self {
            trim_enabled: false,
            max_messages: 50,
            summary_enabled: false,
        }
    }
}

/// 提示路由配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HintRouterSettings {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub routes: Vec<HintRouteSettingsEntry>,
}

/// 提示路由条目（配置层面，provider 为字符串）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HintRouteSettingsEntry {
    pub hint: String,
    pub provider: String,
    pub model: String,
}

/// 配对认证配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PairingSettings {
    #[serde(default)]
    pub enabled: bool,
}

// ============ Gateway 配置类型 ============

/// Gateway 全局配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GatewayConfig {
    #[serde(default)]
    pub tunnel: GatewayTunnelConfig,
}

/// Gateway 隧道配置（用于 webhook 公网回调）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GatewayTunnelConfig {
    /// 是否启用隧道能力
    #[serde(default)]
    pub enabled: bool,
    /// provider：cloudflare | ngrok | none
    #[serde(default = "default_gateway_tunnel_provider")]
    pub provider: String,
    /// 模式：managed | external
    #[serde(default = "default_gateway_tunnel_mode")]
    pub mode: String,
    /// 隧道二进制路径（为空则走 PATH）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,
    /// 本地回源 host
    #[serde(default = "default_gateway_tunnel_local_host")]
    pub local_host: String,
    /// 本地回源 port
    #[serde(default = "default_gateway_tunnel_local_port")]
    pub local_port: u16,
    /// 固定公网基础地址（例如 https://bot.example.com）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_base_url: Option<String>,
    #[serde(default)]
    pub cloudflare: CloudflareTunnelConfig,
}

impl Default for GatewayTunnelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: default_gateway_tunnel_provider(),
            mode: default_gateway_tunnel_mode(),
            binary_path: None,
            local_host: default_gateway_tunnel_local_host(),
            local_port: default_gateway_tunnel_local_port(),
            public_base_url: None,
            cloudflare: CloudflareTunnelConfig::default(),
        }
    }
}

/// Cloudflare Tunnel 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CloudflareTunnelConfig {
    /// Cloudflare account_id（预留给 API 模式）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    /// tunnel 名称（cloudflared tunnel create 时使用）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tunnel_name: Option<String>,
    /// tunnel UUID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tunnel_id: Option<String>,
    /// cloudflared 运行 token（推荐）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_token: Option<String>,
    /// tunnel credentials 文件路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credentials_file: Option<String>,
    /// 绑定域名（例如 bot.example.com）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dns_name: Option<String>,
}

fn default_gateway_tunnel_provider() -> String {
    "cloudflare".to_string()
}

fn default_gateway_tunnel_mode() -> String {
    "managed".to_string()
}

fn default_gateway_tunnel_local_host() -> String {
    "127.0.0.1".to_string()
}

fn default_gateway_tunnel_local_port() -> u16 {
    3000
}

// ============ 渠道配置类型（Telegram / Discord / 飞书 Bot） ============

/// 渠道配置
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChannelsConfig {
    #[serde(default)]
    pub telegram: TelegramBotConfig,
    #[serde(default)]
    pub discord: DiscordBotConfig,
    #[serde(default)]
    pub feishu: FeishuBotConfig,
    #[serde(default)]
    pub wechat: WechatBotConfig,
}

/// Telegram Bot 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelegramBotConfig {
    #[serde(default)]
    pub enabled: bool,
    /// 兼容旧配置：单账号 Token
    #[serde(default)]
    pub bot_token: String,
    /// 兼容旧配置：允许用户 ID（迁移时可映射到 allow_from）
    #[serde(default)]
    pub allowed_user_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// 默认账号 ID（多账号启用时）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_account: Option<String>,
    /// 多账号配置
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub accounts: HashMap<String, TelegramAccountConfig>,
    /// DM 策略：pairing | allowlist | open | disabled
    #[serde(default = "default_telegram_dm_policy")]
    pub dm_policy: String,
    /// DM 允许列表（支持 "*"）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
    /// 群组策略：allowlist | open | disabled
    #[serde(default = "default_telegram_group_policy")]
    pub group_policy: String,
    /// 群组发送者允许列表（为空时回退 allow_from）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_allow_from: Vec<String>,
    /// 群组配置（key 为 chat_id 或 "*"）
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub groups: HashMap<String, TelegramGroupConfig>,
    /// 流式模式：off | partial | block | progress（兼容 legacy bool）
    #[serde(
        default = "default_telegram_streaming_mode",
        deserialize_with = "deserialize_telegram_streaming_mode"
    )]
    pub streaming: String,
    /// 回复模式：off | first | all
    #[serde(default = "default_telegram_reply_to_mode")]
    pub reply_to_mode: String,
}

impl Default for TelegramBotConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bot_token: String::new(),
            allowed_user_ids: Vec::new(),
            default_model: None,
            default_account: None,
            accounts: HashMap::new(),
            dm_policy: default_telegram_dm_policy(),
            allow_from: Vec::new(),
            group_policy: default_telegram_group_policy(),
            group_allow_from: Vec::new(),
            groups: HashMap::new(),
            streaming: default_telegram_streaming_mode(),
            reply_to_mode: default_telegram_reply_to_mode(),
        }
    }
}

/// Telegram 多账号配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelegramAccountConfig {
    #[serde(default = "default_telegram_account_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dm_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub groups: HashMap<String, TelegramGroupConfig>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_telegram_streaming_mode"
    )]
    pub streaming: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to_mode: Option<String>,
}

impl Default for TelegramAccountConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            name: None,
            bot_token: None,
            token_file: None,
            default_model: None,
            dm_policy: None,
            allow_from: Vec::new(),
            group_policy: None,
            group_allow_from: Vec::new(),
            groups: HashMap::new(),
            streaming: None,
            reply_to_mode: None,
        }
    }
}

fn default_telegram_account_enabled() -> bool {
    true
}

/// Telegram 群组配置
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TelegramGroupConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_mention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(
        default = "default_feishu_allow_from",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub topics: HashMap<String, TelegramTopicConfig>,
}

/// Telegram 话题配置
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TelegramTopicConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_mention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
}

fn default_telegram_dm_policy() -> String {
    "pairing".to_string()
}

fn default_telegram_group_policy() -> String {
    "allowlist".to_string()
}

fn default_telegram_streaming_mode() -> String {
    "partial".to_string()
}

fn default_telegram_reply_to_mode() -> String {
    "off".to_string()
}

fn normalize_telegram_streaming_mode(mode: &str) -> String {
    match mode.trim().to_ascii_lowercase().as_str() {
        "off" => "off".to_string(),
        "block" => "block".to_string(),
        "progress" => "partial".to_string(),
        "partial" => "partial".to_string(),
        _ => default_telegram_streaming_mode(),
    }
}

fn deserialize_telegram_streaming_mode<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let mode = match value {
        serde_json::Value::String(s) => normalize_telegram_streaming_mode(&s),
        serde_json::Value::Bool(false) => "off".to_string(),
        serde_json::Value::Bool(true) => "partial".to_string(),
        serde_json::Value::Null => default_telegram_streaming_mode(),
        _ => default_telegram_streaming_mode(),
    };
    Ok(mode)
}

fn deserialize_optional_telegram_streaming_mode<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let mode = match value {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(normalize_telegram_streaming_mode(&s)),
        Some(serde_json::Value::Bool(false)) => Some("off".to_string()),
        Some(serde_json::Value::Bool(true)) => Some("partial".to_string()),
        Some(_) => Some(default_telegram_streaming_mode()),
    };
    Ok(mode)
}

/// Discord Bot 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordBotConfig {
    #[serde(default)]
    pub enabled: bool,
    /// 兼容旧配置：单账号 Token
    #[serde(default)]
    pub bot_token: String,
    /// 兼容旧配置：允许服务器 ID 列表
    #[serde(default)]
    pub allowed_server_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// 默认账号 ID（多账号启用时）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_account: Option<String>,
    /// 多账号配置
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub accounts: HashMap<String, DiscordAccountConfig>,
    /// DM 策略：pairing | allowlist | open | disabled
    #[serde(default = "default_discord_dm_policy")]
    pub dm_policy: String,
    /// DM 允许列表（支持 "*"）
    #[serde(
        default = "default_discord_allow_from",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub allow_from: Vec<String>,
    /// 群组策略：allowlist | open | disabled
    #[serde(default = "default_discord_group_policy")]
    pub group_policy: String,
    /// 群组发送者允许列表（为空时回退 allow_from）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_allow_from: Vec<String>,
    /// 群组配置（key 为 guild_id 或 "*"）
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub groups: HashMap<String, DiscordGuildConfig>,
    /// 流式模式：off | partial | block | progress（兼容 legacy bool）
    #[serde(
        default = "default_discord_streaming_mode",
        deserialize_with = "deserialize_discord_streaming_mode"
    )]
    pub streaming: String,
    /// 回复模式：off | first | all
    #[serde(default = "default_discord_reply_to_mode")]
    pub reply_to_mode: String,
    /// Message 内容意图与特权意图
    #[serde(default)]
    pub intents: DiscordIntentsConfig,
    /// 细粒度动作开关
    #[serde(default)]
    pub actions: DiscordActionsConfig,
    /// 线程绑定配置
    #[serde(default)]
    pub thread_bindings: DiscordThreadBindingsConfig,
    /// 自动在线状态（Presence）
    #[serde(default)]
    pub auto_presence: DiscordAutoPresenceConfig,
    /// 语音配置
    #[serde(default)]
    pub voice: DiscordVoiceConfig,
    /// 交互组件配置
    #[serde(default)]
    pub agent_components: DiscordAgentComponentsConfig,
    /// UI 配置
    #[serde(default)]
    pub ui: DiscordUiConfig,
    /// 执行审批配置
    #[serde(default)]
    pub exec_approvals: DiscordExecApprovalsConfig,
    /// 回复前缀（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_prefix: Option<String>,
    /// ACK reaction（可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ack_reaction: Option<String>,
}

impl Default for DiscordBotConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bot_token: String::new(),
            allowed_server_ids: Vec::new(),
            default_model: None,
            default_account: None,
            accounts: HashMap::new(),
            dm_policy: default_discord_dm_policy(),
            allow_from: default_discord_allow_from(),
            group_policy: default_discord_group_policy(),
            group_allow_from: Vec::new(),
            groups: HashMap::new(),
            streaming: default_discord_streaming_mode(),
            reply_to_mode: default_discord_reply_to_mode(),
            intents: DiscordIntentsConfig::default(),
            actions: DiscordActionsConfig::default(),
            thread_bindings: DiscordThreadBindingsConfig::default(),
            auto_presence: DiscordAutoPresenceConfig::default(),
            voice: DiscordVoiceConfig::default(),
            agent_components: DiscordAgentComponentsConfig::default(),
            ui: DiscordUiConfig::default(),
            exec_approvals: DiscordExecApprovalsConfig::default(),
            response_prefix: None,
            ack_reaction: None,
        }
    }
}

/// Discord 多账号配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordAccountConfig {
    #[serde(default = "default_discord_account_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot_token: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_server_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dm_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub groups: HashMap<String, DiscordGuildConfig>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_discord_streaming_mode"
    )]
    pub streaming: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intents: Option<DiscordIntentsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actions: Option<DiscordActionsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_bindings: Option<DiscordThreadBindingsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_presence: Option<DiscordAutoPresenceConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<DiscordVoiceConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_components: Option<DiscordAgentComponentsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui: Option<DiscordUiConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exec_approvals: Option<DiscordExecApprovalsConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ack_reaction: Option<String>,
}

impl Default for DiscordAccountConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            name: None,
            bot_token: None,
            allowed_server_ids: Vec::new(),
            default_model: None,
            dm_policy: None,
            allow_from: Vec::new(),
            group_policy: None,
            group_allow_from: Vec::new(),
            groups: HashMap::new(),
            streaming: None,
            reply_to_mode: None,
            intents: None,
            actions: None,
            thread_bindings: None,
            auto_presence: None,
            voice: None,
            agent_components: None,
            ui: None,
            exec_approvals: None,
            response_prefix: None,
            ack_reaction: None,
        }
    }
}

fn default_discord_account_enabled() -> bool {
    true
}

/// Discord 群组配置（按 guild）
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DiscordGuildConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_mention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub channels: HashMap<String, DiscordChannelConfig>,
}

/// Discord 频道配置
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DiscordChannelConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_mention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
}

/// Discord Intents 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordIntentsConfig {
    #[serde(default = "default_discord_message_content_intent")]
    pub message_content: bool,
    #[serde(default)]
    pub guild_members: bool,
    #[serde(default)]
    pub presence: bool,
}

impl Default for DiscordIntentsConfig {
    fn default() -> Self {
        Self {
            message_content: default_discord_message_content_intent(),
            guild_members: false,
            presence: false,
        }
    }
}

fn default_discord_message_content_intent() -> bool {
    true
}

/// Discord Action 开关
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordActionsConfig {
    #[serde(default = "default_true")]
    pub reactions: bool,
    #[serde(default = "default_true")]
    pub messages: bool,
    #[serde(default = "default_true")]
    pub threads: bool,
    #[serde(default = "default_true")]
    pub moderation: bool,
    #[serde(default = "default_true")]
    pub presence: bool,
}

impl Default for DiscordActionsConfig {
    fn default() -> Self {
        Self {
            reactions: true,
            messages: true,
            threads: true,
            moderation: true,
            presence: true,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Discord 线程绑定配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordThreadBindingsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_discord_thread_idle_hours")]
    pub idle_hours: u32,
    #[serde(default)]
    pub max_age_hours: u32,
    #[serde(default)]
    pub spawn_subagent_sessions: bool,
    #[serde(default)]
    pub spawn_acp_sessions: bool,
}

impl Default for DiscordThreadBindingsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_hours: default_discord_thread_idle_hours(),
            max_age_hours: 0,
            spawn_subagent_sessions: false,
            spawn_acp_sessions: false,
        }
    }
}

fn default_discord_thread_idle_hours() -> u32 {
    24
}

/// Discord Auto Presence 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordAutoPresenceConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_discord_presence_interval_ms")]
    pub interval_ms: u64,
    #[serde(default = "default_discord_presence_min_update_interval_ms")]
    pub min_update_interval_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub healthy_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degraded_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exhausted_text: Option<String>,
}

impl Default for DiscordAutoPresenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_ms: default_discord_presence_interval_ms(),
            min_update_interval_ms: default_discord_presence_min_update_interval_ms(),
            healthy_text: None,
            degraded_text: None,
            exhausted_text: None,
        }
    }
}

fn default_discord_presence_interval_ms() -> u64 {
    30_000
}

fn default_discord_presence_min_update_interval_ms() -> u64 {
    15_000
}

/// Discord 语音自动加入
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DiscordVoiceAutoJoinConfig {
    pub guild_id: String,
    pub channel_id: String,
}

/// Discord 语音配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordVoiceConfig {
    #[serde(default = "default_discord_voice_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auto_join: Vec<DiscordVoiceAutoJoinConfig>,
    #[serde(default = "default_discord_voice_dave_encryption")]
    pub dave_encryption: bool,
    #[serde(default = "default_discord_voice_decryption_failure_tolerance")]
    pub decryption_failure_tolerance: u32,
}

impl Default for DiscordVoiceConfig {
    fn default() -> Self {
        Self {
            enabled: default_discord_voice_enabled(),
            auto_join: Vec::new(),
            dave_encryption: default_discord_voice_dave_encryption(),
            decryption_failure_tolerance: default_discord_voice_decryption_failure_tolerance(),
        }
    }
}

fn default_discord_voice_enabled() -> bool {
    true
}

fn default_discord_voice_dave_encryption() -> bool {
    true
}

fn default_discord_voice_decryption_failure_tolerance() -> u32 {
    24
}

/// Discord Agent Components 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordAgentComponentsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for DiscordAgentComponentsConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Discord UI 组件配置
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DiscordUiComponentsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accent_color: Option<String>,
}

/// Discord UI 配置
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DiscordUiConfig {
    #[serde(default)]
    pub components: DiscordUiComponentsConfig,
}

/// Discord 执行审批配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscordExecApprovalsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub approvers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub agent_filter: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub session_filter: Vec<String>,
    #[serde(default)]
    pub cleanup_after_resolve: bool,
    #[serde(default = "default_discord_exec_approval_target")]
    pub target: String,
}

impl Default for DiscordExecApprovalsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            approvers: Vec::new(),
            agent_filter: Vec::new(),
            session_filter: Vec::new(),
            cleanup_after_resolve: false,
            target: default_discord_exec_approval_target(),
        }
    }
}

fn default_discord_exec_approval_target() -> String {
    "dm".to_string()
}

fn default_discord_dm_policy() -> String {
    "pairing".to_string()
}

fn default_discord_allow_from() -> Vec<String> {
    Vec::new()
}

fn default_discord_group_policy() -> String {
    "allowlist".to_string()
}

fn default_discord_streaming_mode() -> String {
    "partial".to_string()
}

fn default_discord_reply_to_mode() -> String {
    "off".to_string()
}

fn normalize_discord_streaming_mode(mode: &str) -> String {
    match mode.trim().to_ascii_lowercase().as_str() {
        "off" => "off".to_string(),
        "block" => "block".to_string(),
        "progress" => "partial".to_string(),
        "partial" => "partial".to_string(),
        _ => default_discord_streaming_mode(),
    }
}

fn deserialize_discord_streaming_mode<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let mode = match value {
        serde_json::Value::String(s) => normalize_discord_streaming_mode(&s),
        serde_json::Value::Bool(false) => "off".to_string(),
        serde_json::Value::Bool(true) => "partial".to_string(),
        serde_json::Value::Null => default_discord_streaming_mode(),
        _ => default_discord_streaming_mode(),
    };
    Ok(mode)
}

fn deserialize_optional_discord_streaming_mode<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let mode = match value {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(normalize_discord_streaming_mode(&s)),
        Some(serde_json::Value::Bool(false)) => Some("off".to_string()),
        Some(serde_json::Value::Bool(true)) => Some("partial".to_string()),
        Some(_) => Some(default_discord_streaming_mode()),
    };
    Ok(mode)
}

/// 飞书 Bot 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeishuBotConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub app_id: String,
    #[serde(default)]
    pub app_secret: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypt_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// 默认账号 ID（多账号启用时）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_account: Option<String>,
    /// 多账号配置
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub accounts: HashMap<String, FeishuAccountConfig>,
    /// API 域名：feishu | lark
    #[serde(default = "default_feishu_domain")]
    pub domain: String,
    /// 连接模式：websocket | webhook
    #[serde(default = "default_feishu_connection_mode")]
    pub connection_mode: String,
    /// webhook 监听地址
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_host: Option<String>,
    /// webhook 监听端口
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_port: Option<u16>,
    /// webhook 路由路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_path: Option<String>,
    /// DM 策略：pairing | allowlist | open | disabled
    #[serde(default = "default_feishu_dm_policy")]
    pub dm_policy: String,
    /// DM 允许列表（支持 "*"）
    #[serde(
        default = "default_feishu_allow_from",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub allow_from: Vec<String>,
    /// 群组策略：allowlist | open | disabled
    #[serde(default = "default_feishu_group_policy")]
    pub group_policy: String,
    /// 群组发送者允许列表（为空时回退 allow_from）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_allow_from: Vec<String>,
    /// 群组配置（key 为 chat_id 或 "*"）
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub groups: HashMap<String, FeishuGroupConfig>,
    /// 流式模式：off | partial | block
    #[serde(default = "default_feishu_streaming_mode")]
    pub streaming: String,
    /// 回复模式：off | first | all
    #[serde(default = "default_feishu_reply_to_mode")]
    pub reply_to_mode: String,
}

impl Default for FeishuBotConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            app_id: String::new(),
            app_secret: String::new(),
            verification_token: None,
            encrypt_key: None,
            default_model: None,
            default_account: None,
            accounts: HashMap::new(),
            domain: default_feishu_domain(),
            connection_mode: default_feishu_connection_mode(),
            webhook_host: None,
            webhook_port: None,
            webhook_path: None,
            dm_policy: default_feishu_dm_policy(),
            allow_from: default_feishu_allow_from(),
            group_policy: default_feishu_group_policy(),
            group_allow_from: Vec::new(),
            groups: HashMap::new(),
            streaming: default_feishu_streaming_mode(),
            reply_to_mode: default_feishu_reply_to_mode(),
        }
    }
}

/// 微信 Bot 配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WechatBotConfig {
    #[serde(default)]
    pub enabled: bool,
    /// 兼容旧配置：单账号 token
    #[serde(default)]
    pub bot_token: String,
    /// 兼容旧配置：单账号 baseUrl
    #[serde(default)]
    pub base_url: String,
    /// 兼容旧配置：单账号 cdnBaseUrl
    #[serde(default)]
    pub cdn_base_url: String,
    /// 兼容旧配置：单账号 accountId
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    /// 兼容旧配置：扫码用户 ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scanner_user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// 默认账号 ID（多账号启用时）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_account: Option<String>,
    /// 多账号配置
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub accounts: HashMap<String, WechatAccountConfig>,
    /// DM 策略：pairing | allowlist | open | disabled
    #[serde(default = "default_wechat_dm_policy")]
    pub dm_policy: String,
    /// DM 允许列表（支持 "*"）
    #[serde(
        default = "default_wechat_allow_from",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub allow_from: Vec<String>,
    /// 群组策略：allowlist | open | disabled
    #[serde(default = "default_wechat_group_policy")]
    pub group_policy: String,
    /// 群组发送者允许列表（为空时回退 allow_from）
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_allow_from: Vec<String>,
    /// 群组配置（key 为 group_id 或 "*"）
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub groups: HashMap<String, WechatGroupConfig>,
    /// 流式模式：off | partial | block
    #[serde(default = "default_wechat_streaming_mode")]
    pub streaming: String,
    /// 回复模式：off | first | all
    #[serde(default = "default_wechat_reply_to_mode")]
    pub reply_to_mode: String,
}

impl Default for WechatBotConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bot_token: String::new(),
            base_url: String::new(),
            cdn_base_url: String::new(),
            account_id: None,
            scanner_user_id: None,
            default_model: None,
            default_account: None,
            accounts: HashMap::new(),
            dm_policy: default_wechat_dm_policy(),
            allow_from: default_wechat_allow_from(),
            group_policy: default_wechat_group_policy(),
            group_allow_from: Vec::new(),
            groups: HashMap::new(),
            streaming: default_wechat_streaming_mode(),
            reply_to_mode: default_wechat_reply_to_mode(),
        }
    }
}

/// 微信多账号配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WechatAccountConfig {
    #[serde(default = "default_wechat_account_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cdn_base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scanner_user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dm_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub groups: HashMap<String, WechatGroupConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to_mode: Option<String>,
}

impl Default for WechatAccountConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            name: None,
            base_url: None,
            cdn_base_url: None,
            bot_token: None,
            scanner_user_id: None,
            default_model: None,
            dm_policy: None,
            allow_from: Vec::new(),
            group_policy: None,
            group_allow_from: Vec::new(),
            groups: HashMap::new(),
            streaming: None,
            reply_to_mode: None,
        }
    }
}

fn default_wechat_account_enabled() -> bool {
    true
}

/// 微信群组配置
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WechatGroupConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
}

fn default_wechat_dm_policy() -> String {
    "pairing".to_string()
}

fn default_wechat_allow_from() -> Vec<String> {
    Vec::new()
}

fn default_wechat_group_policy() -> String {
    "allowlist".to_string()
}

fn default_wechat_streaming_mode() -> String {
    "off".to_string()
}

fn default_wechat_reply_to_mode() -> String {
    "off".to_string()
}

/// 飞书多账号配置
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeishuAccountConfig {
    #[serde(default = "default_feishu_account_enabled")]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_secret: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypt_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub webhook_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dm_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub groups: HashMap<String, FeishuGroupConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streaming: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to_mode: Option<String>,
}

impl Default for FeishuAccountConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            name: None,
            app_id: None,
            app_secret: None,
            verification_token: None,
            encrypt_key: None,
            default_model: None,
            domain: None,
            connection_mode: None,
            webhook_host: None,
            webhook_port: None,
            webhook_path: None,
            dm_policy: None,
            allow_from: Vec::new(),
            group_policy: None,
            group_allow_from: Vec::new(),
            groups: HashMap::new(),
            streaming: None,
            reply_to_mode: None,
        }
    }
}

fn default_feishu_account_enabled() -> bool {
    true
}

/// 飞书群组配置
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FeishuGroupConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_mention: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow_from: Vec<String>,
}

fn default_feishu_domain() -> String {
    "feishu".to_string()
}

fn default_feishu_connection_mode() -> String {
    "websocket".to_string()
}

fn default_feishu_dm_policy() -> String {
    "open".to_string()
}

fn default_feishu_allow_from() -> Vec<String> {
    vec!["*".to_string()]
}

fn default_feishu_group_policy() -> String {
    "allowlist".to_string()
}

fn default_feishu_streaming_mode() -> String {
    "partial".to_string()
}

fn default_feishu_reply_to_mode() -> String {
    "off".to_string()
}
