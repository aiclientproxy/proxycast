//! 模型注册表数据结构
//!
//! 基于多协议模型运行时的目录管理方式，定义增强的模型元数据结构

use serde::{Deserialize, Serialize};

/// 模型能力
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// 是否支持视觉输入
    pub vision: bool,
    /// 是否支持工具调用
    pub tools: bool,
    /// 是否支持流式输出
    pub streaming: bool,
    /// 是否支持 JSON 模式
    pub json_mode: bool,
    /// 是否支持函数调用
    pub function_calling: bool,
    /// 是否支持推理/思考
    pub reasoning: bool,
    /// 是否支持可选推理强度；仅当模型接口明确声明时填充
    #[serde(default)]
    pub reasoning_effort: Option<ModelReasoningEffortSupport>,
}

/// 模型能力快照的事实来源。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapabilityProvenance {
    Canonical,
    ProviderExplicit,
    #[default]
    InferredHint,
}

/// Provider 配置中可作为执行事实的模型能力快照。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ProviderModelCapability {
    #[serde(default)]
    pub task_families: Vec<ModelTaskFamily>,
    #[serde(default)]
    pub input_modalities: Vec<ModelModality>,
    #[serde(default)]
    pub output_modalities: Vec<ModelModality>,
    #[serde(default)]
    pub runtime_features: Vec<ModelRuntimeFeature>,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
}

/// Provider 下的模型配置。没有 capability 的记录只能作为 catalog hint。
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ProviderModelConfig {
    pub id: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub capability: Option<ProviderModelCapability>,
}

impl ProviderModelConfig {
    pub fn hint(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: None,
            capability: None,
        }
    }
}

/// 模型声明的可选推理强度；id/label 用于菜单，value 用于请求。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelReasoningEffortOption {
    pub id: String,
    pub value: String,
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default: bool,
}

/// 推理强度能力来源
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelReasoningEffortSource {
    Api,
    Registry,
    Custom,
}

/// 模型推理强度能力
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelReasoningEffortSupport {
    /// 是否支持 reasoning_effort 参数
    pub supported: bool,
    /// 支持的档位
    #[serde(default)]
    pub levels: Vec<String>,
    /// 模型特定菜单；为空时消费者可从 levels 构造标准菜单。
    #[serde(default)]
    pub options: Vec<ModelReasoningEffortOption>,
    /// 默认档位
    #[serde(default)]
    pub default: Option<String>,
    /// 能力来源
    #[serde(default)]
    pub source: Option<ModelReasoningEffortSource>,
}

/// 模型可用的 Provider 服务等级。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelServiceTier {
    pub id: String,
    pub name: String,
    pub description: String,
}

/// 模型在选择器和模型覆盖列表中的可见性。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelVisibility {
    #[default]
    List,
    Hide,
    None,
}

impl std::fmt::Display for ModelVisibility {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::List => formatter.write_str("list"),
            Self::Hide => formatter.write_str("hide"),
            Self::None => formatter.write_str("none"),
        }
    }
}

impl std::str::FromStr for ModelVisibility {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "list" => Ok(Self::List),
            "hide" => Ok(Self::Hide),
            "none" => Ok(Self::None),
            _ => Err(format!("Unknown model visibility: {value}")),
        }
    }
}

/// 模型任务族
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelTaskFamily {
    Chat,
    Reasoning,
    VisionUnderstanding,
    ImageGeneration,
    ImageEdit,
    VideoGeneration,
    SpeechToText,
    TextToSpeech,
    Embedding,
    Rerank,
    Moderation,
}

/// 模型输入/输出模态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelModality {
    Text,
    Image,
    Audio,
    Video,
    File,
    Embedding,
    Json,
}

/// 模型运行时特性
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelRuntimeFeature {
    Streaming,
    ToolCalling,
    JsonSchema,
    Reasoning,
    PromptCache,
    ResponsesApi,
    ChatCompletionsApi,
    ImagesApi,
}

/// 模型显式声明支持的 Multi-Agent runtime 版本。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelMultiAgentVersion {
    Disabled,
    V1,
    V2,
}

/// 模型部署来源
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelDeploymentSource {
    #[default]
    UserCloud,
    Local,
    OemCloud,
}

/// 模型管理面
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelManagementPlane {
    #[default]
    LocalSettings,
    OemControlPlane,
    Hybrid,
}

/// 模型别名来源
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelAliasSource {
    Official,
    Relay,
    Oem,
    Local,
}

/// 模型定价
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// 输入价格（每百万 token）
    pub input_per_million: Option<f64>,
    /// 输出价格（每百万 token）
    pub output_per_million: Option<f64>,
    /// 缓存读取价格（每百万 token）
    pub cache_read_per_million: Option<f64>,
    /// 缓存写入价格（每百万 token）
    pub cache_write_per_million: Option<f64>,
    /// 货币单位 ("USD" | "CNY")
    pub currency: String,
}

impl Default for ModelPricing {
    fn default() -> Self {
        Self {
            input_per_million: None,
            output_per_million: None,
            cache_read_per_million: None,
            cache_write_per_million: None,
            currency: "USD".to_string(),
        }
    }
}

/// 模型限制
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelLimits {
    /// 上下文长度
    pub context_length: Option<u32>,
    /// 最大输出 token 数
    pub max_output_tokens: Option<u32>,
    /// 每分钟请求数限制
    pub requests_per_minute: Option<u32>,
    /// 每分钟 token 数限制
    pub tokens_per_minute: Option<u32>,
}

/// 模型状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    /// 活跃可用
    #[default]
    Active,
    /// 预览版
    Preview,
    /// Alpha 测试
    Alpha,
    /// Beta 测试
    Beta,
    /// 已弃用
    Deprecated,
    /// 旧版本
    Legacy,
}

impl std::fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Preview => write!(f, "preview"),
            Self::Alpha => write!(f, "alpha"),
            Self::Beta => write!(f, "beta"),
            Self::Deprecated => write!(f, "deprecated"),
            Self::Legacy => write!(f, "legacy"),
        }
    }
}

impl std::str::FromStr for ModelStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "active" => Ok(Self::Active),
            "preview" => Ok(Self::Preview),
            "alpha" => Ok(Self::Alpha),
            "beta" => Ok(Self::Beta),
            "deprecated" => Ok(Self::Deprecated),
            "legacy" => Ok(Self::Legacy),
            _ => Err(format!("Unknown model status: {s}")),
        }
    }
}

/// 模型服务等级
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelTier {
    /// 快速响应，适合简单任务
    Mini,
    /// 均衡性能，适合大多数任务
    #[default]
    Pro,
    /// 最强能力，适合复杂任务
    Max,
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mini => write!(f, "mini"),
            Self::Pro => write!(f, "pro"),
            Self::Max => write!(f, "max"),
        }
    }
}

impl std::str::FromStr for ModelTier {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mini" => Ok(Self::Mini),
            "pro" => Ok(Self::Pro),
            "max" => Ok(Self::Max),
            _ => Err(format!("Unknown model tier: {s}")),
        }
    }
}

/// 模型数据来源
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    /// 从内嵌资源加载（构建时打包）
    Embedded,
    /// 从 models.dev API 获取（已弃用）
    ModelsDev,
    /// 本地硬编码（国内模型等）
    #[default]
    Local,
    /// 用户自定义
    Custom,
    /// 从 Provider API 获取
    Api,
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Embedded => write!(f, "embedded"),
            Self::ModelsDev => write!(f, "models.dev"),
            Self::Local => write!(f, "local"),
            Self::Custom => write!(f, "custom"),
            Self::Api => write!(f, "api"),
        }
    }
}

impl std::str::FromStr for ModelSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "embedded" => Ok(Self::Embedded),
            "models.dev" | "modelsdev" => Ok(Self::ModelsDev),
            "local" => Ok(Self::Local),
            "custom" => Ok(Self::Custom),
            "api" => Ok(Self::Api),
            _ => Err(format!("Unknown model source: {s}")),
        }
    }
}

/// 增强的模型元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedModelMetadata {
    /// 模型 ID (如 "claude-sonnet-4-5-20250514")
    pub id: String,
    /// 显示名称 (如 "Claude Sonnet 4.5")
    pub display_name: String,
    /// Provider ID (如 "anthropic", "openai", "dashscope")
    pub provider_id: String,
    /// Provider 显示名称
    pub provider_name: String,
    /// 模型家族 (如 "sonnet", "gpt-4", "qwen")
    pub family: Option<String>,
    /// 服务等级
    pub tier: ModelTier,
    /// 模型能力
    pub capabilities: ModelCapabilities,
    /// 能力快照来源；启发式结果只能用于 catalog 展示，不能授权 runtime route。
    #[serde(default)]
    pub capability_provenance: ModelCapabilityProvenance,
    /// 在模型选择器和 Agent 模型覆盖列表中的可见性
    #[serde(default)]
    pub visibility: ModelVisibility,
    /// Provider 明确声明的服务等级
    #[serde(default)]
    pub service_tiers: Vec<ModelServiceTier>,
    /// Provider catalog 的默认服务等级
    #[serde(default)]
    pub default_service_tier: Option<String>,
    /// 任务族
    #[serde(default)]
    pub task_families: Vec<ModelTaskFamily>,
    /// 输入模态
    #[serde(default)]
    pub input_modalities: Vec<ModelModality>,
    /// 输出模态
    #[serde(default)]
    pub output_modalities: Vec<ModelModality>,
    /// 运行时特性
    #[serde(default)]
    pub runtime_features: Vec<ModelRuntimeFeature>,
    /// 模型显式声明支持的 Multi-Agent runtime；缺失时不得推断。
    #[serde(default)]
    pub multi_agent_version: Option<ModelMultiAgentVersion>,
    /// 部署来源
    #[serde(default)]
    pub deployment_source: ModelDeploymentSource,
    /// 管理面
    #[serde(default)]
    pub management_plane: ModelManagementPlane,
    /// 规范化模型 ID
    pub canonical_model_id: Option<String>,
    /// 实际 Provider 返回的模型 ID
    pub provider_model_id: Option<String>,
    /// 别名来源
    pub alias_source: Option<ModelAliasSource>,
    /// 定价信息
    pub pricing: Option<ModelPricing>,
    /// 限制信息
    pub limits: ModelLimits,
    /// 模型状态
    pub status: ModelStatus,
    /// 发布日期
    pub release_date: Option<String>,
    /// 是否为最新版本
    pub is_latest: bool,
    /// 描述
    pub description: Option<String>,
    /// 数据来源
    pub source: ModelSource,
    /// 创建时间 (Unix 时间戳)
    pub created_at: i64,
    /// 最后更新时间 (Unix 时间戳)
    pub updated_at: i64,
}

impl EnhancedModelMetadata {
    /// 创建新的模型元数据
    pub fn new(
        id: String,
        display_name: String,
        provider_id: String,
        provider_name: String,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id,
            display_name,
            provider_id,
            provider_name,
            family: None,
            tier: ModelTier::Pro,
            capabilities: ModelCapabilities::default(),
            capability_provenance: ModelCapabilityProvenance::InferredHint,
            visibility: ModelVisibility::List,
            service_tiers: vec![],
            default_service_tier: None,
            task_families: vec![],
            input_modalities: vec![],
            output_modalities: vec![],
            runtime_features: vec![],
            multi_agent_version: None,
            deployment_source: ModelDeploymentSource::UserCloud,
            management_plane: ModelManagementPlane::LocalSettings,
            canonical_model_id: None,
            provider_model_id: None,
            alias_source: None,
            pricing: None,
            limits: ModelLimits::default(),
            status: ModelStatus::Active,
            release_date: None,
            is_latest: false,
            description: None,
            source: ModelSource::Local,
            created_at: now,
            updated_at: now,
        }
    }

    /// 设置模型家族
    pub fn with_family(mut self, family: impl Into<String>) -> Self {
        self.family = Some(family.into());
        self
    }

    /// 设置服务等级
    pub fn with_tier(mut self, tier: ModelTier) -> Self {
        self.tier = tier;
        self
    }

    /// 设置模型能力
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_capability_provenance(mut self, provenance: ModelCapabilityProvenance) -> Self {
        self.capability_provenance = provenance;
        self
    }

    /// 设置模型选择器可见性。
    pub fn with_visibility(mut self, visibility: ModelVisibility) -> Self {
        self.visibility = visibility;
        self
    }

    /// 设置 Provider catalog 明确声明的服务等级。
    pub fn with_service_tiers(
        mut self,
        service_tiers: Vec<ModelServiceTier>,
        default_service_tier: Option<String>,
    ) -> Self {
        self.service_tiers = service_tiers;
        self.default_service_tier = default_service_tier
            .filter(|default| self.service_tiers.iter().any(|tier| tier.id == *default));
        self
    }

    pub fn supports_service_tier(&self, service_tier: &str) -> bool {
        self.service_tiers
            .iter()
            .any(|tier| tier.id == service_tier)
    }

    /// 设置定价信息
    pub fn with_pricing(mut self, pricing: ModelPricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// 设置限制信息
    pub fn with_limits(mut self, limits: ModelLimits) -> Self {
        self.limits = limits;
        self
    }

    /// 设置模型状态
    pub fn with_status(mut self, status: ModelStatus) -> Self {
        self.status = status;
        self
    }

    /// 设置发布日期
    pub fn with_release_date(mut self, date: impl Into<String>) -> Self {
        self.release_date = Some(date.into());
        self
    }

    /// 设置是否为最新版本
    pub fn with_is_latest(mut self, is_latest: bool) -> Self {
        self.is_latest = is_latest;
        self
    }

    /// 设置描述
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// 设置数据来源
    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = source;
        self
    }

    pub fn supports_task_family(&self, family: &ModelTaskFamily) -> bool {
        self.task_families.contains(family)
    }

    pub fn has_input_modality(&self, modality: &ModelModality) -> bool {
        self.input_modalities.contains(modality)
    }

    pub fn has_output_modality(&self, modality: &ModelModality) -> bool {
        self.output_modalities.contains(modality)
    }
}

/// 用户模型偏好
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserModelPreference {
    /// 模型 ID
    pub model_id: String,
    /// 是否收藏
    pub is_favorite: bool,
    /// 是否隐藏
    pub is_hidden: bool,
    /// 自定义别名
    pub custom_alias: Option<String>,
    /// 使用次数
    pub usage_count: u32,
    /// 最后使用时间 (Unix 时间戳)
    pub last_used_at: Option<i64>,
    /// 创建时间 (Unix 时间戳)
    pub created_at: i64,
    /// 更新时间 (Unix 时间戳)
    pub updated_at: i64,
}

impl UserModelPreference {
    /// 创建新的用户偏好
    pub fn new(model_id: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            model_id,
            is_favorite: false,
            is_hidden: false,
            custom_alias: None,
            usage_count: 0,
            last_used_at: None,
            created_at: now,
            updated_at: now,
        }
    }
}

/// 模型同步状态
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelSyncState {
    /// 最后同步时间 (Unix 时间戳)
    pub last_sync_at: Option<i64>,
    /// 同步的模型数量
    pub model_count: u32,
    /// 是否正在同步
    pub is_syncing: bool,
    /// 最后同步错误
    pub last_error: Option<String>,
}

// ============================================================================
// Provider Alias 相关类型（用于中转服务与协议转换模型别名）
// ============================================================================

/// 单个模型别名映射
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAlias {
    /// 实际模型 ID（如 "claude-sonnet-4-5-20250929"）
    pub actual: String,
    /// 内部 API 名称（如 "CLAUDE_SONNET_4_5_20250929_V1_0"）
    pub internal_name: Option<String>,
    /// 原始 Provider（如 "anthropic"）
    pub provider: Option<String>,
    /// 描述
    pub description: Option<String>,
}

/// Provider 的别名配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderAliasConfig {
    /// Provider ID（如 "kiro"、"antigravity"）
    pub provider: String,
    /// 描述
    pub description: Option<String>,
    /// 支持的模型列表
    #[serde(default)]
    pub models: Vec<String>,
    /// 别名映射（模型名 -> 别名配置）
    pub aliases: std::collections::HashMap<String, ModelAlias>,
    /// 更新时间
    pub updated_at: Option<String>,
}

impl ProviderAliasConfig {
    /// 检查是否支持指定模型
    pub fn supports_model(&self, model: &str) -> bool {
        self.models.contains(&model.to_string()) || self.aliases.contains_key(model)
    }

    /// 获取模型的内部名称
    pub fn get_internal_name(&self, model: &str) -> Option<&str> {
        self.aliases
            .get(model)
            .and_then(|a| a.internal_name.as_deref())
    }

    /// 获取模型的实际 ID
    pub fn get_actual_model(&self, model: &str) -> Option<&str> {
        self.aliases.get(model).map(|a| a.actual.as_str())
    }
}

/// models.dev API 响应中的 Provider 结构
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ModelsDevProvider {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub api: Option<String>,
    #[serde(default)]
    pub npm: Option<String>,
    #[serde(default)]
    pub models: std::collections::HashMap<String, ModelsDevModel>,
}

/// models.dev API 响应中的 Model 结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsDevModel {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub release_date: Option<String>,
    #[serde(default)]
    pub attachment: bool,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub temperature: bool,
    #[serde(default)]
    pub tool_call: bool,
    #[serde(default)]
    pub cost: Option<ModelsDevCost>,
    #[serde(default)]
    pub limit: Option<ModelsDevLimit>,
    #[serde(default)]
    pub modalities: Option<ModelsDevModalities>,
    #[serde(default)]
    pub experimental: Option<bool>,
    #[serde(default)]
    pub status: Option<String>,
}

/// models.dev API 响应中的 Cost 结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsDevCost {
    #[serde(default)]
    pub input: Option<f64>,
    #[serde(default)]
    pub output: Option<f64>,
    #[serde(default)]
    pub cache_read: Option<f64>,
    #[serde(default)]
    pub cache_write: Option<f64>,
}

/// models.dev API 响应中的 Limit 结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsDevLimit {
    #[serde(default)]
    pub context: Option<u32>,
    #[serde(default)]
    pub output: Option<u32>,
}

/// models.dev API 响应中的 Modalities 结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsDevModalities {
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub output: Vec<String>,
}

impl ModelsDevModel {
    /// 转换为 EnhancedModelMetadata
    /// 预留：用于从 models.dev API 导入模型数据
    #[allow(dead_code)]
    pub fn to_enhanced_metadata(
        &self,
        provider_id: &str,
        provider_name: &str,
    ) -> EnhancedModelMetadata {
        let now = chrono::Utc::now().timestamp();

        // 判断是否支持视觉
        let supports_vision = self
            .modalities
            .as_ref()
            .map(|m| m.input.iter().any(|i| i == "image" || i == "video"))
            .unwrap_or(false)
            || self.attachment;

        // 根据模型名称推断服务等级
        let tier = infer_model_tier(&self.id, &self.name);

        // 解析状态
        let status = self
            .status
            .as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or(ModelStatus::Active);

        // 判断是否为最新版本
        let is_latest = self.id.contains("latest");

        EnhancedModelMetadata {
            id: self.id.clone(),
            display_name: self.name.clone(),
            provider_id: provider_id.to_string(),
            provider_name: provider_name.to_string(),
            family: self.family.clone(),
            tier,
            capabilities: ModelCapabilities {
                vision: supports_vision,
                tools: self.tool_call,
                streaming: true, // 大多数模型都支持流式
                json_mode: true, // 大多数模型都支持 JSON 模式
                function_calling: self.tool_call,
                reasoning: self.reasoning,
                reasoning_effort: None,
            },
            capability_provenance: ModelCapabilityProvenance::ProviderExplicit,
            visibility: ModelVisibility::List,
            service_tiers: vec![],
            default_service_tier: None,
            task_families: vec![],
            input_modalities: vec![],
            output_modalities: vec![],
            runtime_features: vec![],
            multi_agent_version: None,
            deployment_source: ModelDeploymentSource::UserCloud,
            management_plane: ModelManagementPlane::LocalSettings,
            canonical_model_id: None,
            provider_model_id: Some(self.id.clone()),
            alias_source: None,
            pricing: self.cost.as_ref().map(|c| ModelPricing {
                input_per_million: c.input,
                output_per_million: c.output,
                cache_read_per_million: c.cache_read,
                cache_write_per_million: c.cache_write,
                currency: "USD".to_string(),
            }),
            limits: ModelLimits {
                context_length: self.limit.as_ref().and_then(|l| l.context),
                max_output_tokens: self.limit.as_ref().and_then(|l| l.output),
                requests_per_minute: None,
                tokens_per_minute: None,
            },
            status,
            release_date: self.release_date.clone(),
            is_latest,
            description: None,
            source: ModelSource::ModelsDev,
            created_at: now,
            updated_at: now,
        }
    }
}

/// 根据模型 ID 和名称推断服务等级
/// 用于 to_enhanced_metadata 和测试
#[allow(dead_code)]
fn infer_model_tier(model_id: &str, model_name: &str) -> ModelTier {
    let id_lower = model_id.to_lowercase();
    let name_lower = model_name.to_lowercase();

    // Mini 等级模型（优先检查，因为 gpt-4o-mini 包含 gpt-4o）
    let mini_patterns = [
        "mini",
        "nano",
        "lite",
        "flash",
        "haiku",
        "gpt-4o-mini",
        "gemini-flash",
        "qwen-turbo",
        "glm-4-flash",
    ];
    for pattern in mini_patterns {
        if id_lower.contains(pattern) || name_lower.contains(pattern) {
            return ModelTier::Mini;
        }
    }

    // Max 等级模型
    let max_patterns = [
        "opus",
        "gpt-4o",
        "gpt-4-turbo",
        "gemini-2.5-pro",
        "gemini-ultra",
        "claude-3-opus",
        "qwen-max",
        "glm-4-plus",
        "deepseek-v3",
    ];
    for pattern in max_patterns {
        if id_lower.contains(pattern) || name_lower.contains(pattern) {
            return ModelTier::Max;
        }
    }

    // 默认为 Pro 等级
    ModelTier::Pro
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_tier_inference() {
        assert_eq!(
            infer_model_tier("claude-opus-4-5-20250514", "Claude Opus 4.5"),
            ModelTier::Max
        );
        assert_eq!(
            infer_model_tier("gpt-4o-mini", "GPT-4o Mini"),
            ModelTier::Mini
        );
        assert_eq!(
            infer_model_tier("claude-sonnet-4-5", "Claude Sonnet 4.5"),
            ModelTier::Pro
        );
        assert_eq!(
            infer_model_tier("gemini-2.5-flash", "Gemini 2.5 Flash"),
            ModelTier::Mini
        );
    }

    #[test]
    fn test_model_status_parsing() {
        assert_eq!(
            "active".parse::<ModelStatus>().unwrap(),
            ModelStatus::Active
        );
        assert_eq!(
            "deprecated".parse::<ModelStatus>().unwrap(),
            ModelStatus::Deprecated
        );
        assert_eq!("beta".parse::<ModelStatus>().unwrap(), ModelStatus::Beta);
    }
}
