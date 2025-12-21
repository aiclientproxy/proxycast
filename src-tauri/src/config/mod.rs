//! 配置管理模块
//!
//! 提供 YAML 配置文件支持、热重载和配置导入导出功能
//! 同时保持与旧版 JSON 配置的向后兼容性

mod export;
mod hot_reload;
mod import;
mod path_utils;
mod types;
mod yaml;

pub use export::{ExportBundle, ExportOptions, ExportService, REDACTED_PLACEHOLDER};
pub use hot_reload::{
    ConfigChangeEvent, ConfigChangeKind, FileWatcher, HotReloadManager, ReloadResult,
};
pub use import::{ImportOptions, ImportService, ValidationResult};
pub use path_utils::{collapse_tilde, contains_tilde, expand_tilde};
pub use types::{
    AmpConfig, AmpModelMapping, ApiKeyEntry, Config, CredentialEntry, CredentialPoolConfig,
    CustomProviderConfig, GeminiApiKeyEntry, IFlowCredentialEntry, InjectionRuleConfig,
    InjectionSettings, LoggingConfig, ProviderConfig, ProvidersConfig, QuotaExceededConfig,
    RemoteManagementConfig, RetrySettings, RoutingConfig, ServerConfig, TlsConfig,
    VertexApiKeyEntry, VertexModelAlias,
};
pub use yaml::{load_config, save_config, ConfigError, ConfigManager, YamlService};

#[cfg(test)]
mod tests;
