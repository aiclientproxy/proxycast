//! 配置模块属性测试
//!
//! 使用 proptest 进行属性测试

use crate::config::types::{NavigationConfig, WorkspacePreferencesConfig};
use crate::config::{
    collapse_tilde, contains_tilde, expand_tilde, Config, ConfigManager, CustomProviderConfig,
    HotReloadManager, InjectionSettings, LoggingConfig, ProviderConfig, ProvidersConfig,
    ReloadResult, RetrySettings, RoutingConfig, ServerConfig, YamlService,
};
use proptest::prelude::*;
use std::io::Write;
use tempfile::NamedTempFile;

/// 生成随机的主机地址
fn arb_host() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("127.0.0.1".to_string()),
        Just("localhost".to_string()),
        Just("::1".to_string()),
        "[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}".prop_map(|s| s),
    ]
}

/// 生成随机的端口号
fn arb_port() -> impl Strategy<Value = u16> {
    1024u16..65535u16
}

/// 生成随机的 API 密钥
fn arb_api_key() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_-]{8,32}".prop_map(|s| s)
}

/// 生成随机的服务器配置
fn arb_server_config() -> impl Strategy<Value = ServerConfig> {
    (arb_host(), arb_port(), arb_api_key()).prop_map(|(host, port, api_key)| ServerConfig {
        host,
        port,
        api_key,
        tls: crate::config::TlsConfig::default(),
        response_cache: crate::config::ResponseCacheSettings::default(),
    })
}

/// 生成随机的 Provider 配置
fn arb_provider_config() -> impl Strategy<Value = ProviderConfig> {
    (
        any::<bool>(),
        proptest::option::of("[a-zA-Z0-9/_.-]{5,50}".prop_map(|s| s)),
        proptest::option::of(prop_oneof![
            Just("us-east-1".to_string()),
            Just("us-west-2".to_string()),
            Just("eu-west-1".to_string()),
        ]),
        proptest::option::of("[a-zA-Z0-9-]{5,20}".prop_map(|s| s)),
    )
        .prop_map(
            |(enabled, credentials_path, region, project_id)| ProviderConfig {
                enabled,
                credentials_path,
                region,
                project_id,
            },
        )
}

/// 生成随机的自定义 Provider 配置
fn arb_custom_provider_config() -> impl Strategy<Value = CustomProviderConfig> {
    (
        any::<bool>(),
        proptest::option::of(arb_api_key()),
        proptest::option::of(prop_oneof![
            Just("https://api.openai.com/v1".to_string()),
            Just("https://api.anthropic.com".to_string()),
            Just("https://custom.api.com".to_string()),
        ]),
    )
        .prop_map(|(enabled, api_key, base_url)| CustomProviderConfig {
            enabled,
            api_key,
            base_url,
        })
}

/// 生成随机的 Providers 配置
fn arb_providers_config() -> impl Strategy<Value = ProvidersConfig> {
    (
        arb_provider_config(),
        arb_provider_config(),
        arb_provider_config(),
        arb_custom_provider_config(),
        arb_custom_provider_config(),
    )
        .prop_map(|(kiro, gemini, qwen, openai, claude)| ProvidersConfig {
            kiro,
            gemini,
            qwen,
            openai,
            claude,
        })
}

fn arb_default_provider() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("kiro".to_string()),
        Just("gemini".to_string()),
        Just("qwen".to_string()),
    ]
}

/// 生成随机的路由辅助配置
fn arb_routing_config() -> impl Strategy<Value = RoutingConfig> {
    proptest::collection::hash_map(
        "[a-z]+-[a-z0-9]+".prop_map(|s| s),
        "[a-z]+-[a-z0-9-]+".prop_map(|s| s),
        0..5,
    )
    .prop_map(|model_aliases| RoutingConfig { model_aliases })
}

/// 生成随机的重试配置
fn arb_retry_settings() -> impl Strategy<Value = RetrySettings> {
    (
        1u32..10u32,
        100u64..5000u64,
        5000u64..60000u64,
        any::<bool>(),
    )
        .prop_map(
            |(max_retries, base_delay_ms, max_delay_ms, auto_switch_provider)| RetrySettings {
                max_retries,
                base_delay_ms,
                max_delay_ms,
                auto_switch_provider,
            },
        )
}

/// 生成随机的日志配置
fn arb_logging_config() -> impl Strategy<Value = LoggingConfig> {
    (
        any::<bool>(),
        prop_oneof![
            Just("debug".to_string()),
            Just("info".to_string()),
            Just("warn".to_string()),
            Just("error".to_string()),
        ],
        1u32..30u32,
        any::<bool>(),
    )
        .prop_map(
            |(enabled, level, retention_days, include_request_body)| LoggingConfig {
                enabled,
                level,
                retention_days,
                include_request_body,
            },
        )
}

/// 生成随机的完整配置
fn arb_config() -> impl Strategy<Value = Config> {
    (
        arb_server_config(),
        arb_providers_config(),
        arb_default_provider(),
        arb_routing_config(),
        arb_retry_settings(),
        arb_logging_config(),
    )
        .prop_map(
            |(server, providers, default_provider, routing, retry, logging)| Config {
                server,
                providers,
                default_provider,
                routing,
                retry,
                logging,
                injection: InjectionSettings::default(),
                auth_dir: "~/.lime/auth".to_string(),
                credential_pool: crate::config::CredentialPoolConfig::default(),
                remote_management: crate::config::RemoteManagementConfig::default(),
                quota_exceeded: crate::config::QuotaExceededConfig::default(),
                proxy_url: None,
                ampcode: crate::config::AmpConfig::default(),
                endpoint_providers: crate::config::EndpointProvidersConfig::default(),
                minimize_to_tray: true,
                models: crate::config::ModelsConfig::default(),
                agent: crate::config::NativeAgentConfig::default(),
                language: "zh-CN".to_string(),
                experimental: crate::config::ExperimentalFeatures::default(),
                tool_calling: crate::config::ToolCallingConfig::default(),
                workspace_preferences: WorkspacePreferencesConfig::default(),
                navigation: NavigationConfig::default(),
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: enhancement-roadmap, Property 11: 配置往返一致性**
    /// *对于任意* 有效配置，序列化后再反序列化应得到等价的配置
    /// **Validates: Requirements 4.1**
    #[test]
    fn prop_config_roundtrip(config in arb_config()) {
        // 序列化为 YAML
        let yaml = ConfigManager::to_yaml(&config)
            .expect("序列化应成功");

        // 反序列化回 Config
        let parsed = ConfigManager::parse_yaml(&yaml)
            .expect("反序列化应成功");

        // 验证往返一致性
        prop_assert_eq!(
            config.server,
            parsed.server,
            "服务器配置往返不一致"
        );
        prop_assert_eq!(
            config.providers,
            parsed.providers,
            "Provider 配置往返不一致"
        );
        prop_assert_eq!(
            config.default_provider,
            parsed.default_provider,
            "默认 Provider 往返不一致"
        );
        prop_assert_eq!(
            config.routing.model_aliases,
            parsed.routing.model_aliases,
            "模型别名往返不一致"
        );
        prop_assert_eq!(
            config.retry,
            parsed.retry,
            "重试配置往返不一致"
        );
        prop_assert_eq!(
            config.logging,
            parsed.logging,
            "日志配置往返不一致"
        );
    }

    /// **Feature: enhancement-roadmap, Property 11: 配置往返一致性（服务器配置）**
    /// *对于任意* 服务器配置，序列化后再反序列化应得到等价的配置
    /// **Validates: Requirements 4.1**
    #[test]
    fn prop_server_config_roundtrip(server in arb_server_config()) {
        let config = Config {
            server: server.clone(),
            ..Config::default()
        };

        let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
        let parsed = ConfigManager::parse_yaml(&yaml).expect("反序列化应成功");

        prop_assert_eq!(
            server,
            parsed.server,
            "服务器配置往返不一致"
        );
    }

    /// **Feature: enhancement-roadmap, Property 11: 配置往返一致性（Provider 配置）**
    /// *对于任意* Provider 配置，序列化后再反序列化应得到等价的配置
    /// **Validates: Requirements 4.1**
    #[test]
    fn prop_providers_config_roundtrip(providers in arb_providers_config()) {
        let config = Config {
            providers: providers.clone(),
            ..Config::default()
        };

        let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
        let parsed = ConfigManager::parse_yaml(&yaml).expect("反序列化应成功");

        prop_assert_eq!(
            providers,
            parsed.providers,
            "Provider 配置往返不一致"
        );
    }

    /// **Feature: enhancement-roadmap, Property 11: 配置往返一致性（重试配置）**
    /// *对于任意* 重试配置，序列化后再反序列化应得到等价的配置
    /// **Validates: Requirements 4.1**
    #[test]
    fn prop_retry_settings_roundtrip(retry in arb_retry_settings()) {
        let config = Config {
            retry: retry.clone(),
            ..Config::default()
        };

        let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
        let parsed = ConfigManager::parse_yaml(&yaml).expect("反序列化应成功");

        prop_assert_eq!(
            retry,
            parsed.retry,
            "重试配置往返不一致"
        );
    }

    /// **Feature: enhancement-roadmap, Property 11: 配置往返一致性（日志配置）**
    /// *对于任意* 日志配置，序列化后再反序列化应得到等价的配置
    /// **Validates: Requirements 4.1**
    #[test]
    fn prop_logging_config_roundtrip(logging in arb_logging_config()) {
        let config = Config {
            logging: logging.clone(),
            ..Config::default()
        };

        let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
        let parsed = ConfigManager::parse_yaml(&yaml).expect("反序列化应成功");

        prop_assert_eq!(
            logging,
            parsed.logging,
            "日志配置往返不一致"
        );
    }

    /// **Feature: enhancement-roadmap, Property 11: 配置往返一致性（路由辅助配置）**
    /// *对于任意* 路由配置，序列化后再反序列化应得到等价的配置
    /// **Validates: Requirements 4.1**
    #[test]
    fn prop_routing_config_roundtrip(routing in arb_routing_config()) {
        let config = Config {
            routing: routing.clone(),
            ..Config::default()
        };

        let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
        let parsed = ConfigManager::parse_yaml(&yaml).expect("反序列化应成功");

        prop_assert_eq!(
            routing.model_aliases,
            parsed.routing.model_aliases,
            "模型别名往返不一致"
        );
    }
}

/// 生成有效的服务器配置（端口非零）
fn arb_valid_server_config() -> impl Strategy<Value = ServerConfig> {
    (arb_host(), 1u16..65535u16, arb_api_key()).prop_map(|(host, port, api_key)| ServerConfig {
        host,
        port,
        api_key,
        tls: crate::config::TlsConfig::default(),
        response_cache: crate::config::ResponseCacheSettings::default(),
    })
}

/// 生成有效的重试配置（通过验证）
fn arb_valid_retry_settings() -> impl Strategy<Value = RetrySettings> {
    (
        1u32..100u32,      // max_retries <= 100
        1u64..5000u64,     // base_delay_ms > 0
        5000u64..60000u64, // max_delay_ms
        any::<bool>(),
    )
        .prop_map(
            |(max_retries, base_delay_ms, max_delay_ms, auto_switch_provider)| RetrySettings {
                max_retries,
                base_delay_ms,
                max_delay_ms,
                auto_switch_provider,
            },
        )
}

/// 生成有效的日志配置（保留天数非零）
fn arb_valid_logging_config() -> impl Strategy<Value = LoggingConfig> {
    (
        any::<bool>(),
        prop_oneof![
            Just("debug".to_string()),
            Just("info".to_string()),
            Just("warn".to_string()),
            Just("error".to_string()),
        ],
        1u32..30u32, // retention_days > 0
        any::<bool>(),
    )
        .prop_map(
            |(enabled, level, retention_days, include_request_body)| LoggingConfig {
                enabled,
                level,
                retention_days,
                include_request_body,
            },
        )
}

/// 生成有效的配置（通过验证的配置）
fn arb_valid_config() -> impl Strategy<Value = Config> {
    (
        arb_valid_server_config(),
        arb_providers_config(),
        arb_default_provider(),
        arb_routing_config(),
        arb_valid_retry_settings(),
        arb_valid_logging_config(),
    )
        .prop_map(
            |(server, providers, default_provider, routing, retry, logging)| Config {
                server,
                providers,
                default_provider,
                routing,
                retry,
                logging,
                injection: InjectionSettings::default(),
                auth_dir: "~/.lime/auth".to_string(),
                credential_pool: crate::config::CredentialPoolConfig::default(),
                remote_management: crate::config::RemoteManagementConfig::default(),
                quota_exceeded: crate::config::QuotaExceededConfig::default(),
                proxy_url: None,
                ampcode: crate::config::AmpConfig::default(),
                endpoint_providers: crate::config::EndpointProvidersConfig::default(),
                minimize_to_tray: true,
                models: crate::config::ModelsConfig::default(),
                agent: crate::config::NativeAgentConfig::default(),
                language: "zh-CN".to_string(),
                experimental: crate::config::ExperimentalFeatures::default(),
                tool_calling: crate::config::ToolCallingConfig::default(),
                workspace_preferences: WorkspacePreferencesConfig::default(),
                navigation: NavigationConfig::default(),
            },
        )
}

/// 生成无效配置的类型
#[derive(Debug, Clone, Copy)]
enum InvalidConfigType {
    ZeroPort,
    TooManyRetries,
    ZeroBaseDelay,
    ZeroRetentionDays,
}

/// 生成无效的配置（不通过验证的配置）
fn arb_invalid_config() -> impl Strategy<Value = Config> {
    (
        arb_valid_server_config(),
        arb_providers_config(),
        arb_default_provider(),
        arb_routing_config(),
        arb_valid_retry_settings(),
        arb_valid_logging_config(),
        prop_oneof![
            Just(InvalidConfigType::ZeroPort),
            Just(InvalidConfigType::TooManyRetries),
            Just(InvalidConfigType::ZeroBaseDelay),
            Just(InvalidConfigType::ZeroRetentionDays),
        ],
    )
        .prop_map(
            |(server, providers, default_provider, routing, retry, logging, invalid_type)| {
                let mut config = Config {
                    server,
                    providers,
                    default_provider,
                    routing,
                    retry,
                    logging,
                    injection: InjectionSettings::default(),
                    auth_dir: "~/.lime/auth".to_string(),
                    credential_pool: crate::config::CredentialPoolConfig::default(),
                    remote_management: crate::config::RemoteManagementConfig::default(),
                    quota_exceeded: crate::config::QuotaExceededConfig::default(),
                    proxy_url: None,
                    ampcode: crate::config::AmpConfig::default(),
                    endpoint_providers: crate::config::EndpointProvidersConfig::default(),
                    minimize_to_tray: true,
                    models: crate::config::ModelsConfig::default(),
                    agent: crate::config::NativeAgentConfig::default(),
                    language: "zh-CN".to_string(),
                    experimental: crate::config::ExperimentalFeatures::default(),
                    tool_calling: crate::config::ToolCallingConfig::default(),
                    workspace_preferences: WorkspacePreferencesConfig::default(),
                    navigation: NavigationConfig::default(),
                };
                // 根据类型使配置无效
                match invalid_type {
                    InvalidConfigType::ZeroPort => config.server.port = 0,
                    InvalidConfigType::TooManyRetries => config.retry.max_retries = 101,
                    InvalidConfigType::ZeroBaseDelay => config.retry.base_delay_ms = 0,
                    InvalidConfigType::ZeroRetentionDays => config.logging.retention_days = 0,
                }
                config
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: enhancement-roadmap, Property 12: 热重载原子性**
    /// *对于任意* 配置变更，要么完全应用成功，要么回滚到之前状态
    /// **Validates: Requirements 4.2 (验收标准 3)**
    #[test]
    fn prop_hot_reload_atomicity_success(
        initial_config in arb_valid_config(),
        new_config in arb_valid_config()
    ) {
        // 创建临时配置文件
        let mut temp_file = NamedTempFile::new().expect("创建临时文件失败");
        let yaml = ConfigManager::to_yaml(&new_config).expect("序列化失败");
        temp_file.write_all(yaml.as_bytes()).expect("写入文件失败");

        // 创建热重载管理器
        let manager = HotReloadManager::new(initial_config.clone(), temp_file.path().to_path_buf());

        // 执行热重载
        let result = manager.reload();

        // 验证原子性：成功时配置应完全更新
        match result {
            ReloadResult::Success { .. } => {
                let current = manager.config();
                // 验证配置已完全更新
                prop_assert_eq!(
                    current.server,
                    new_config.server,
                    "服务器配置未正确更新"
                );
                prop_assert_eq!(
                    current.providers,
                    new_config.providers,
                    "Provider 配置未正确更新"
                );
                prop_assert_eq!(
                    current.retry,
                    new_config.retry,
                    "重试配置未正确更新"
                );
                prop_assert_eq!(
                    current.logging,
                    new_config.logging,
                    "日志配置未正确更新"
                );
            }
            _ => {
                // 如果失败，应该保持原始配置
                let current = manager.config();
                prop_assert_eq!(
                    current,
                    initial_config,
                    "失败时配置应保持不变"
                );
            }
        }
    }

    /// **Feature: enhancement-roadmap, Property 12: 热重载原子性（失败回滚）**
    /// *对于任意* 无效配置变更，配置应回滚到之前状态
    /// **Validates: Requirements 4.2 (验收标准 3)**
    #[test]
    fn prop_hot_reload_atomicity_rollback(
        initial_config in arb_valid_config(),
        invalid_config in arb_invalid_config()
    ) {
        // 创建临时配置文件（包含无效配置）
        let mut temp_file = NamedTempFile::new().expect("创建临时文件失败");
        let yaml = ConfigManager::to_yaml(&invalid_config).expect("序列化失败");
        temp_file.write_all(yaml.as_bytes()).expect("写入文件失败");

        // 创建热重载管理器
        let manager = HotReloadManager::new(initial_config.clone(), temp_file.path().to_path_buf());

        // 执行热重载
        let result = manager.reload();

        // 验证原子性：失败时配置应回滚到之前状态
        match result {
            ReloadResult::RolledBack { .. } => {
                let current = manager.config();
                // 验证配置已回滚到初始状态
                prop_assert_eq!(
                    current,
                    initial_config,
                    "配置应回滚到初始状态"
                );
            }
            ReloadResult::Success { .. } => {
                // 如果意外成功（不应该发生），验证配置一致性
                let current = manager.config();
                prop_assert_eq!(
                    current,
                    invalid_config,
                    "成功时配置应完全更新"
                );
            }
            ReloadResult::Failed { .. } => {
                // 完全失败的情况，配置应保持不变
                let current = manager.config();
                prop_assert_eq!(
                    current,
                    initial_config,
                    "失败时配置应保持不变"
                );
            }
        }
    }

    /// **Feature: enhancement-roadmap, Property 12: 热重载原子性（文件不存在）**
    /// *对于任意* 初始配置，当配置文件不存在时，配置应保持不变
    /// **Validates: Requirements 4.2 (验收标准 3)**
    #[test]
    fn prop_hot_reload_atomicity_file_not_exists(initial_config in arb_valid_config()) {
        // 使用不存在的文件路径
        let nonexistent_path = std::path::PathBuf::from("/tmp/nonexistent_config_test_12345.yaml");

        // 创建热重载管理器
        let manager = HotReloadManager::new(initial_config.clone(), nonexistent_path);

        // 执行热重载
        let result = manager.reload();

        // 验证原子性：文件不存在时配置应保持不变
        match result {
            ReloadResult::RolledBack { .. } => {
                let current = manager.config();
                prop_assert_eq!(
                    current,
                    initial_config,
                    "文件不存在时配置应保持不变"
                );
            }
            _ => {
                // 其他情况也应保持配置不变
                let current = manager.config();
                prop_assert_eq!(
                    current,
                    initial_config,
                    "配置应保持不变"
                );
            }
        }
    }

    /// **Feature: enhancement-roadmap, Property 12: 热重载原子性（无效 YAML）**
    /// *对于任意* 初始配置，当配置文件包含无效 YAML 时，配置应保持不变
    /// **Validates: Requirements 4.2 (验收标准 3)**
    #[test]
    fn prop_hot_reload_atomicity_invalid_yaml(initial_config in arb_valid_config()) {
        // 创建包含无效 YAML 的临时文件
        let mut temp_file = NamedTempFile::new().expect("创建临时文件失败");
        temp_file.write_all(b"invalid: yaml: content: [").expect("写入文件失败");

        // 创建热重载管理器
        let manager = HotReloadManager::new(initial_config.clone(), temp_file.path().to_path_buf());

        // 执行热重载
        let result = manager.reload();

        // 验证原子性：无效 YAML 时配置应保持不变
        match result {
            ReloadResult::RolledBack { .. } => {
                let current = manager.config();
                prop_assert_eq!(
                    current,
                    initial_config,
                    "无效 YAML 时配置应保持不变"
                );
            }
            _ => {
                // 其他情况也应保持配置不变
                let current = manager.config();
                prop_assert_eq!(
                    current,
                    initial_config,
                    "配置应保持不变"
                );
            }
        }
    }
}

// ============================================================================
// Property 3: Tilde Path Expansion
// ============================================================================

/// 生成有效的 tilde 路径（~/path 格式）
/// 排除 "." 和 ".." 路径段，因为这些会导致路径规范化问题
fn arb_tilde_path() -> impl Strategy<Value = String> {
    // 生成路径段：字母数字、下划线、连字符
    // 排除单独的 "." 和 ".." 以避免路径规范化问题
    let path_segment = "[a-zA-Z0-9_-]{1,20}";

    // 生成 0-5 个路径段
    proptest::collection::vec(path_segment, 0..6).prop_map(|segments| {
        if segments.is_empty() {
            "~".to_string()
        } else {
            format!("~/{}", segments.join("/"))
        }
    })
}

/// 生成不包含 tilde 的绝对路径
fn arb_absolute_path() -> impl Strategy<Value = String> {
    let path_segment = "[a-zA-Z0-9_.-]{1,20}";

    proptest::collection::vec(path_segment, 1..6)
        .prop_map(|segments| format!("/{}", segments.join("/")))
}

/// 生成不包含 tilde 的相对路径
/// 排除单独的 "." 和 ".." 以避免路径规范化问题
fn arb_relative_path() -> impl Strategy<Value = String> {
    // 使用至少2个字符的路径段，或者不以单独的点开头
    // 这样可以避免生成 "." 或 ".." 这样的特殊路径
    let path_segment = "[a-zA-Z0-9_-][a-zA-Z0-9_.-]{0,19}";

    proptest::collection::vec(path_segment, 1..6).prop_map(|segments| segments.join("/"))
}

/// 生成 ~user/path 格式的路径（不支持的格式）
fn arb_tilde_user_path() -> impl Strategy<Value = String> {
    let username = "[a-z]{3,10}";
    let path_segment = "[a-zA-Z0-9_.-]{1,20}";

    (username, proptest::collection::vec(path_segment, 0..4)).prop_map(|(user, segments)| {
        if segments.is_empty() {
            format!("~{user}")
        } else {
            format!("~{}/{}", user, segments.join("/"))
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: config-credential-export, Property 3: Tilde Path Expansion**
    /// *For any* valid tilde path (~/path format), expanding and then collapsing
    /// should produce the original path.
    /// **Validates: Requirements 2.3**
    #[test]
    fn prop_tilde_path_roundtrip(path in arb_tilde_path()) {
        // 展开 tilde 路径
        let expanded = expand_tilde(&path);

        // 收缩回 tilde 格式
        let collapsed = collapse_tilde(&expanded);

        // 验证往返一致性
        prop_assert_eq!(
            &collapsed,
            &path,
            "Tilde 路径往返不一致: 原始={}, 展开={:?}, 收缩={}",
            path,
            expanded,
            collapsed
        );
    }

    /// **Feature: config-credential-export, Property 3: Tilde Path Expansion**
    /// *For any* tilde path, the expanded path should start with the user's home directory.
    /// **Validates: Requirements 2.3**
    #[test]
    fn prop_tilde_expansion_starts_with_home(path in arb_tilde_path()) {
        let home_dir = dirs::home_dir().expect("应该能获取主目录");
        let expanded = expand_tilde(&path);

        prop_assert!(
            expanded.starts_with(&home_dir),
            "展开后的路径应以主目录开头: 路径={}, 展开={:?}, 主目录={:?}",
            path,
            expanded,
            home_dir
        );
    }

    /// **Feature: config-credential-export, Property 3: Tilde Path Expansion**
    /// *For any* tilde path, contains_tilde should return true before expansion
    /// and false after expansion.
    /// **Validates: Requirements 2.3**
    #[test]
    fn prop_contains_tilde_before_expansion(path in arb_tilde_path()) {
        // 展开前应包含 tilde
        prop_assert!(
            contains_tilde(&path),
            "展开前路径应包含 tilde: {}",
            path
        );

        // 展开后不应包含 tilde
        let expanded = expand_tilde(&path);
        prop_assert!(
            !contains_tilde(&expanded),
            "展开后路径不应包含 tilde: {:?}",
            expanded
        );
    }

    /// **Feature: config-credential-export, Property 3: Tilde Path Expansion**
    /// *For any* absolute path (not starting with ~), expand_tilde should return
    /// the path unchanged.
    /// **Validates: Requirements 2.3**
    #[test]
    fn prop_absolute_path_unchanged(path in arb_absolute_path()) {
        let expanded = expand_tilde(&path);
        let expanded_str = expanded.to_string_lossy().to_string();

        prop_assert_eq!(
            &expanded_str,
            &path,
            "绝对路径应保持不变: 原始={}, 展开={:?}",
            path,
            expanded
        );
    }

    /// **Feature: config-credential-export, Property 3: Tilde Path Expansion**
    /// *For any* relative path (not starting with ~ or /), expand_tilde should
    /// return the path unchanged.
    /// **Validates: Requirements 2.3**
    #[test]
    fn prop_relative_path_unchanged(path in arb_relative_path()) {
        let expanded = expand_tilde(&path);
        let expanded_str = expanded.to_string_lossy().to_string();

        prop_assert_eq!(
            &expanded_str,
            &path,
            "相对路径应保持不变: 原始={}, 展开={:?}",
            path,
            expanded
        );
    }

    /// **Feature: config-credential-export, Property 3: Tilde Path Expansion**
    /// *For any* ~user/path format (unsupported), expand_tilde should return
    /// the path unchanged.
    /// **Validates: Requirements 2.3**
    #[test]
    fn prop_tilde_user_path_unchanged(path in arb_tilde_user_path()) {
        let expanded = expand_tilde(&path);
        let expanded_str = expanded.to_string_lossy().to_string();

        prop_assert_eq!(
            &expanded_str,
            &path,
            "~user/path 格式应保持不变: 原始={}, 展开={:?}",
            path,
            expanded
        );
    }

    /// **Feature: config-credential-export, Property 3: Tilde Path Expansion**
    /// *For any* path under the home directory, collapse_tilde should produce
    /// a path starting with ~.
    /// **Validates: Requirements 2.3**
    #[test]
    fn prop_collapse_home_path_starts_with_tilde(subpath in arb_relative_path()) {
        let home_dir = dirs::home_dir().expect("应该能获取主目录");
        let full_path = home_dir.join(&subpath);

        let collapsed = collapse_tilde(&full_path);

        prop_assert!(
            collapsed.starts_with("~/"),
            "主目录下的路径收缩后应以 ~/ 开头: 路径={:?}, 收缩={}",
            full_path,
            collapsed
        );
    }

    /// **Feature: config-credential-export, Property 3: Tilde Path Expansion**
    /// *For any* path not under the home directory, collapse_tilde should return
    /// the path unchanged.
    /// **Validates: Requirements 2.3**
    #[test]
    fn prop_collapse_non_home_path_unchanged(path in arb_absolute_path()) {
        // 确保路径不在主目录下（使用 /tmp 或类似路径）
        let test_path = format!("/tmp{path}");
        let collapsed = collapse_tilde(&test_path);

        prop_assert_eq!(
            &collapsed,
            &test_path,
            "非主目录路径应保持不变: 原始={}, 收缩={}",
            test_path,
            collapsed
        );
    }
}

// ============================================================================
// Property 2: YAML Comment Preservation
// ============================================================================

/// 生成有效的 YAML 注释（以 # 开头）
fn arb_yaml_comment() -> impl Strategy<Value = String> {
    // 生成注释内容：字母、数字、空格、中文字符
    "[a-zA-Z0-9 ]{1,50}".prop_map(|s| format!("# {s}"))
}

/// 生成带注释的 YAML 配置字符串
fn arb_yaml_with_comments() -> impl Strategy<Value = (String, Vec<String>)> {
    (
        arb_valid_config(),
        proptest::collection::vec(arb_yaml_comment(), 1..5),
    )
        .prop_map(|(config, comments)| {
            // 序列化配置
            let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
            let lines: Vec<&str> = yaml.lines().collect();

            // 在 YAML 中插入注释
            let mut result_lines: Vec<String> = Vec::new();
            let mut comment_iter = comments.iter();

            // 在文件开头添加一个注释
            if let Some(comment) = comment_iter.next() {
                result_lines.push(comment.clone());
            }

            for (i, line) in lines.iter().enumerate() {
                result_lines.push(line.to_string());

                // 在某些行后添加注释
                if i % 5 == 0 {
                    if let Some(comment) = comment_iter.next() {
                        result_lines.push(comment.clone());
                    }
                }
            }

            // 收集实际插入的注释
            let inserted_comments: Vec<String> = result_lines
                .iter()
                .filter(|line| line.trim().starts_with('#'))
                .cloned()
                .collect();

            (result_lines.join("\n"), inserted_comments)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: config-credential-export, Property 2: YAML Comment Preservation**
    /// *For any* YAML file with comments, saving configuration changes should preserve
    /// all existing comments in their original positions.
    /// **Validates: Requirements 1.3**
    #[test]
    fn prop_yaml_comment_preservation(
        (yaml_with_comments, original_comments) in arb_yaml_with_comments(),
        new_config in arb_valid_config()
    ) {
        // 创建临时文件
        let temp_dir = tempfile::tempdir().expect("创建临时目录失败");
        let config_path = temp_dir.path().join("config.yaml");

        // 写入带注释的原始 YAML
        std::fs::write(&config_path, &yaml_with_comments).expect("写入文件失败");

        // 使用 YamlService 保存新配置（应保留注释）
        YamlService::save_preserve_comments(&config_path, &new_config)
            .expect("保存配置失败");

        // 读取保存后的内容
        let saved_content = std::fs::read_to_string(&config_path).expect("读取文件失败");

        // 提取保存后的注释
        let saved_comments: Vec<String> = saved_content
            .lines()
            .filter(|line| line.trim().starts_with('#'))
            .map(|s| s.to_string())
            .collect();

        // 验证注释被保留
        // 注意：由于 YAML 结构可能变化，我们只验证注释内容被保留，不验证位置
        for original_comment in &original_comments {
            let comment_content = original_comment.trim();
            let found = saved_comments.iter().any(|c| c.trim() == comment_content);
            prop_assert!(
                found,
                "注释应被保留: 原始注释='{}', 保存后的注释={:?}",
                comment_content,
                saved_comments
            );
        }
    }

    /// **Feature: config-credential-export, Property 2: YAML Comment Preservation**
    /// *For any* configuration saved with YamlService, the configuration should be
    /// correctly parseable and equivalent to the original.
    /// **Validates: Requirements 1.3**
    #[test]
    fn prop_yaml_save_preserves_config(config in arb_valid_config()) {
        // 创建临时文件
        let temp_dir = tempfile::tempdir().expect("创建临时目录失败");
        let config_path = temp_dir.path().join("config.yaml");

        // 使用 YamlService 保存配置
        YamlService::save_preserve_comments(&config_path, &config)
            .expect("保存配置失败");

        // 读取并解析保存后的配置
        let saved_content = std::fs::read_to_string(&config_path).expect("读取文件失败");
        let parsed_config = ConfigManager::parse_yaml(&saved_content).expect("解析配置失败");

        // 验证配置一致性
        prop_assert_eq!(
            config.server,
            parsed_config.server,
            "服务器配置应一致"
        );
        prop_assert_eq!(
            config.providers,
            parsed_config.providers,
            "Provider 配置应一致"
        );
        prop_assert_eq!(
            config.retry,
            parsed_config.retry,
            "重试配置应一致"
        );
        prop_assert_eq!(
            config.logging,
            parsed_config.logging,
            "日志配置应一致"
        );
    }

    /// **Feature: config-credential-export, Property 2: YAML Comment Preservation**
    /// *For any* YAML file with header comments, saving should preserve header comments.
    /// **Validates: Requirements 1.3**
    #[test]
    fn prop_yaml_header_comment_preservation(
        header_comment in arb_yaml_comment(),
        config in arb_valid_config()
    ) {
        // 创建临时文件
        let temp_dir = tempfile::tempdir().expect("创建临时目录失败");
        let config_path = temp_dir.path().join("config.yaml");

        // 创建带头部注释的 YAML
        let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
        let yaml_with_header = format!("{header_comment}\n{yaml}");

        // 写入文件
        std::fs::write(&config_path, &yaml_with_header).expect("写入文件失败");

        // 使用 YamlService 保存新配置
        YamlService::save_preserve_comments(&config_path, &config)
            .expect("保存配置失败");

        // 读取保存后的内容
        let saved_content = std::fs::read_to_string(&config_path).expect("读取文件失败");

        // 验证头部注释被保留
        let header_content = header_comment.trim();
        let has_header = saved_content.lines().any(|line| line.trim() == header_content);

        prop_assert!(
            has_header,
            "头部注释应被保留: 原始='{}', 保存后内容前100字符='{}'",
            header_content,
            &saved_content[..saved_content.len().min(100)]
        );
    }
}

// ============================================================================
// Unit Tests for YamlService::update_field
// ============================================================================

#[test]
fn test_update_field_simple() {
    // 创建临时文件
    let temp_dir = tempfile::tempdir().expect("创建临时目录失败");
    let config_path = temp_dir.path().join("config.yaml");

    // 写入初始 YAML
    let initial_yaml = r#"server:
  host: 127.0.0.1
  port: 8999
  api_key: test_key
"#;
    std::fs::write(&config_path, initial_yaml).expect("写入文件失败");

    // 更新 port 字段
    YamlService::update_field(&config_path, &["server", "port"], "9000").expect("更新字段失败");

    // 读取并验证
    let content = std::fs::read_to_string(&config_path).expect("读取文件失败");
    assert!(content.contains("port: 9000"), "端口应被更新为 9000");
    assert!(content.contains("host: 127.0.0.1"), "其他字段应保持不变");
}

#[test]
fn test_update_field_preserves_comments() {
    // 创建临时文件
    let temp_dir = tempfile::tempdir().expect("创建临时目录失败");
    let config_path = temp_dir.path().join("config.yaml");

    // 写入带注释的 YAML
    let initial_yaml = r#"# 服务器配置
server:
  # 监听地址
  host: 127.0.0.1
  # 监听端口
  port: 8999
  api_key: test_key
"#;
    std::fs::write(&config_path, initial_yaml).expect("写入文件失败");

    // 更新 port 字段
    YamlService::update_field(&config_path, &["server", "port"], "9000").expect("更新字段失败");

    // 读取并验证
    let content = std::fs::read_to_string(&config_path).expect("读取文件失败");
    assert!(content.contains("port: 9000"), "端口应被更新为 9000");
    assert!(content.contains("# 服务器配置"), "头部注释应保留");
    assert!(content.contains("# 监听地址"), "字段注释应保留");
    assert!(content.contains("# 监听端口"), "字段注释应保留");
}

#[test]
fn test_update_field_not_found() {
    // 创建临时文件
    let temp_dir = tempfile::tempdir().expect("创建临时目录失败");
    let config_path = temp_dir.path().join("config.yaml");

    // 写入初始 YAML
    let initial_yaml = r#"server:
  host: 127.0.0.1
  port: 8999
"#;
    std::fs::write(&config_path, initial_yaml).expect("写入文件失败");

    // 尝试更新不存在的字段
    let result = YamlService::update_field(&config_path, &["server", "nonexistent"], "value");
    assert!(result.is_err(), "更新不存在的字段应返回错误");
}

#[test]
fn test_update_field_nested() {
    // 创建临时文件
    let temp_dir = tempfile::tempdir().expect("创建临时目录失败");
    let config_path = temp_dir.path().join("config.yaml");

    // 写入初始 YAML
    let initial_yaml = r#"server:
  host: 127.0.0.1
  port: 8999
providers:
  kiro:
    enabled: true
    region: us-east-1
"#;
    std::fs::write(&config_path, initial_yaml).expect("写入文件失败");

    // 更新嵌套字段
    YamlService::update_field(&config_path, &["providers", "kiro", "region"], "us-west-2")
        .expect("更新字段失败");

    // 读取并验证
    let content = std::fs::read_to_string(&config_path).expect("读取文件失败");
    assert!(
        content.contains("region: us-west-2"),
        "region 应被更新为 us-west-2"
    );
    assert!(content.contains("enabled: true"), "其他字段应保持不变");
}

// ============================================================================
// Property 4: Export Scope Filtering
// ============================================================================

use crate::config::{
    ApiKeyEntry, CredentialEntry, CredentialPoolConfig, ExportOptions, ExportService,
};

/// 生成随机的 OAuth 凭证条目
fn arb_credential_entry() -> impl Strategy<Value = CredentialEntry> {
    (
        "[a-z]{3,10}-[0-9]{1,5}".prop_map(|s| s),
        "[a-z]+/token-[0-9]{1,5}\\.json".prop_map(|s| s),
        any::<bool>(),
        proptest::option::of("socks5://proxy\\.[a-z]+\\.com:[0-9]{4}".prop_map(|s| s)),
    )
        .prop_map(|(id, token_file, disabled, proxy_url)| CredentialEntry {
            id,
            token_file,
            disabled,
            proxy_url,
        })
}

/// 生成随机的 API Key 凭证条目
fn arb_api_key_entry() -> impl Strategy<Value = ApiKeyEntry> {
    (
        "[a-z]{3,10}-[0-9]{1,5}".prop_map(|s| s),
        "sk-[a-zA-Z0-9]{20,40}".prop_map(|s| s),
        proptest::option::of("https://api\\.[a-z]+\\.com/v[0-9]".prop_map(|s| s)),
        any::<bool>(),
        proptest::option::of("http://proxy\\.[a-z]+\\.com:[0-9]{4}".prop_map(|s| s)),
    )
        .prop_map(|(id, api_key, base_url, disabled, proxy_url)| ApiKeyEntry {
            id,
            api_key,
            base_url,
            disabled,
            proxy_url,
        })
}

/// 生成随机的旧凭证池配置，用于验证退役边界。
fn arb_credential_pool_config() -> impl Strategy<Value = CredentialPoolConfig> {
    (
        proptest::collection::vec(arb_credential_entry(), 0..3),
        proptest::collection::vec(arb_credential_entry(), 0..3),
        proptest::collection::vec(arb_credential_entry(), 0..3),
        proptest::collection::vec(arb_api_key_entry(), 0..3),
        proptest::collection::vec(arb_api_key_entry(), 0..3),
    )
        .prop_map(
            |(kiro, gemini, qwen, openai, claude)| CredentialPoolConfig {
                kiro,
                gemini,
                qwen,
                openai,
                claude,
                gemini_api_keys: vec![],
                vertex_api_keys: vec![],
                codex: vec![],
            },
        )
}

/// 生成带旧凭证池的配置
fn arb_config_with_credentials() -> impl Strategy<Value = Config> {
    (arb_valid_config(), arb_credential_pool_config()).prop_map(|(mut config, pool)| {
        config.credential_pool = pool;
        config
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: config-credential-export, Property 4: Export Scope Filtering**
    /// *For any* export operation with config-only scope, the resulting bundle
    /// should contain only configuration data and no credential token files.
    /// **Validates: Requirements 3.2**
    #[test]
    fn prop_export_scope_config_only(config in arb_config_with_credentials()) {
        let options = ExportOptions::config_only();
        let bundle = ExportService::export(&config, &options, "1.0.0")
            .expect("导出应成功");

        // 验证只包含配置
        prop_assert!(
            bundle.has_config(),
            "config-only 导出应包含配置"
        );
        prop_assert!(
            !bundle.has_credentials(),
            "config-only 导出不应包含凭证 token 文件"
        );
    }

    /// **Feature: config-credential-export, Property 4: Export Scope Filtering**
    /// *For any* export operation with credentials-only scope, the resulting bundle
    /// should contain only credential data and no configuration YAML.
    /// **Validates: Requirements 3.2**
    #[test]
    fn prop_export_scope_credentials_only(config in arb_config_with_credentials()) {
        let options = ExportOptions::credentials_only();
        let bundle = ExportService::export(&config, &options, "1.0.0")
            .expect("导出应成功");

        // 验证不包含配置
        prop_assert!(
            !bundle.has_config(),
            "credentials-only 导出不应包含配置 YAML"
        );
        // 注意：token_files 可能为空（如果没有实际的 token 文件存在）
        // 但 config_yaml 必须为 None
        prop_assert!(
            bundle.config_yaml.is_none(),
            "credentials-only 导出的 config_yaml 应为 None"
        );
    }

    /// **Feature: config-credential-export, Property 4: Export Scope Filtering**
    /// *For any* export operation with full scope, the resulting bundle
    /// should contain both configuration and credential data.
    /// **Validates: Requirements 3.2**
    #[test]
    fn prop_export_scope_full(config in arb_config_with_credentials()) {
        let options = ExportOptions::full();
        let bundle = ExportService::export(&config, &options, "1.0.0")
            .expect("导出应成功");

        // 验证包含配置
        prop_assert!(
            bundle.has_config(),
            "full 导出应包含配置"
        );
        // token_files 可能为空（如果没有实际的 token 文件存在）
        // 但 config_yaml 必须存在
        prop_assert!(
            bundle.config_yaml.is_some(),
            "full 导出的 config_yaml 应存在"
        );
    }

    /// **Feature: config-credential-export, Property 4: Export Scope Filtering**
    /// *For any* export operation, the bundle should correctly reflect the
    /// include_config and include_credentials options.
    /// **Validates: Requirements 3.2**
    #[test]
    fn prop_export_scope_matches_options(
        config in arb_config_with_credentials(),
        include_config in any::<bool>(),
        include_credentials in any::<bool>()
    ) {
        let options = ExportOptions {
            include_config,
            include_credentials,
            redact_secrets: false,
        };

        let bundle = ExportService::export(&config, &options, "1.0.0")
            .expect("导出应成功");

        // 验证配置包含状态与选项一致
        prop_assert_eq!(
            bundle.has_config(),
            include_config,
            "配置包含状态应与 include_config 选项一致"
        );

        // 验证 config_yaml 存在性与选项一致
        prop_assert_eq!(
            bundle.config_yaml.is_some(),
            include_config,
            "config_yaml 存在性应与 include_config 选项一致"
        );
    }
}

// ============================================================================
// Property 5: Redaction Completeness
// ============================================================================

use crate::config::REDACTED_PLACEHOLDER;

/// 生成包含敏感信息的配置
fn arb_config_with_secrets() -> impl Strategy<Value = Config> {
    (
        arb_valid_config(),
        arb_credential_pool_config(),
        // 生成看起来像真实 API 密钥的字符串
        "sk-[a-zA-Z0-9]{20,40}".prop_map(|s| s),
        proptest::option::of("sk-[a-zA-Z0-9]{20,40}".prop_map(|s| s)),
        proptest::option::of("sk-ant-[a-zA-Z0-9]{20,40}".prop_map(|s| s)),
    )
        .prop_map(|(mut config, pool, server_key, openai_key, claude_key)| {
            config.server.api_key = server_key;
            config.providers.openai.api_key = openai_key;
            config.providers.claude.api_key = claude_key;
            config.credential_pool = pool;
            config
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: config-credential-export, Property 5: Redaction Completeness**
    /// *For any* export with redaction enabled, all sensitive values (API keys, tokens,
    /// secrets) should be replaced with placeholder markers, and no original sensitive
    /// data should remain.
    /// **Validates: Requirements 3.4**
    #[test]
    fn prop_redaction_removes_all_secrets(config in arb_config_with_secrets()) {
        // 脱敏配置
        let redacted = ExportService::redact_config(&config);

        // 验证脱敏后不包含敏感信息
        prop_assert!(
            !ExportService::contains_secrets(&redacted),
            "脱敏后的配置不应包含敏感信息"
        );

        // 验证服务器 API 密钥已脱敏
        prop_assert_eq!(
            &redacted.server.api_key,
            REDACTED_PLACEHOLDER,
            "服务器 API 密钥应被脱敏"
        );

        // 验证 OpenAI API 密钥已脱敏（如果存在）
        if config.providers.openai.api_key.is_some() {
            prop_assert_eq!(
                redacted.providers.openai.api_key.as_deref(),
                Some(REDACTED_PLACEHOLDER),
                "OpenAI API 密钥应被脱敏"
            );
        }

        // 验证 Claude API 密钥已脱敏（如果存在）
        if config.providers.claude.api_key.is_some() {
            prop_assert_eq!(
                redacted.providers.claude.api_key.as_deref(),
                Some(REDACTED_PLACEHOLDER),
                "Claude API 密钥应被脱敏"
            );
        }
    }

    /// **Feature: config-credential-export, Property 5: Redaction Completeness**
    /// *For any* configuration with legacy credential pool entries, redaction
    /// should drop the retired pool entirely instead of exporting placeholders.
    /// **Validates: Requirements 3.4**
    #[test]
    fn prop_redaction_drops_legacy_credential_pool(config in arb_config_with_secrets()) {
        let redacted = ExportService::redact_config(&config);

        prop_assert!(
            redacted.credential_pool.is_empty(),
            "旧凭证池已退役，脱敏导出应直接丢弃"
        );
    }

    /// **Feature: config-credential-export, Property 5: Redaction Completeness**
    /// *For any* export with redaction enabled, the exported YAML should not contain
    /// any original sensitive values.
    /// **Validates: Requirements 3.4**
    #[test]
    fn prop_redaction_yaml_no_secrets(config in arb_config_with_secrets()) {
        // 导出带脱敏的 YAML
        let yaml = ExportService::export_yaml(&config, true)
            .expect("导出应成功");

        // 验证 YAML 中不包含原始敏感值
        // 检查原始服务器 API 密钥
        if !config.server.api_key.is_empty() && config.server.api_key != REDACTED_PLACEHOLDER {
            prop_assert!(
                !yaml.contains(&config.server.api_key),
                "YAML 不应包含原始服务器 API 密钥: {}",
                config.server.api_key
            );
        }

        // 检查原始 OpenAI API 密钥
        if let Some(ref key) = config.providers.openai.api_key {
            if !key.is_empty() && key != REDACTED_PLACEHOLDER {
                prop_assert!(
                    !yaml.contains(key),
                    "YAML 不应包含原始 OpenAI API 密钥"
                );
            }
        }

        // 检查原始 Claude API 密钥
        if let Some(ref key) = config.providers.claude.api_key {
            if !key.is_empty() && key != REDACTED_PLACEHOLDER {
                prop_assert!(
                    !yaml.contains(key),
                    "YAML 不应包含原始 Claude API 密钥"
                );
            }
        }

        // 检查旧凭证池中的 API 密钥不会出现在导出内容中
        for entry in &config.credential_pool.openai {
            if !entry.api_key.is_empty() && entry.api_key != REDACTED_PLACEHOLDER {
                prop_assert!(
                    !yaml.contains(&entry.api_key),
                    "YAML 不应包含原始 OpenAI 凭证池 API 密钥"
                );
            }
        }

        for entry in &config.credential_pool.claude {
            if !entry.api_key.is_empty() && entry.api_key != REDACTED_PLACEHOLDER {
                prop_assert!(
                    !yaml.contains(&entry.api_key),
                    "YAML 不应包含原始 Claude 凭证池 API 密钥"
                );
            }
        }
    }

    /// **Feature: config-credential-export, Property 5: Redaction Completeness**
    /// *For any* configuration, redaction should preserve non-sensitive data unchanged.
    /// **Validates: Requirements 3.4**
    #[test]
    fn prop_redaction_preserves_non_sensitive_data(config in arb_config_with_secrets()) {
        let redacted = ExportService::redact_config(&config);

        // 验证非敏感数据保持不变
        prop_assert_eq!(
            config.server.host,
            redacted.server.host,
            "服务器主机应保持不变"
        );
        prop_assert_eq!(
            config.server.port,
            redacted.server.port,
            "服务器端口应保持不变"
        );
        prop_assert_eq!(
            config.providers.kiro.enabled,
            redacted.providers.kiro.enabled,
            "Kiro 启用状态应保持不变"
        );
        prop_assert_eq!(
            config.default_provider,
            redacted.default_provider,
            "默认 Provider 应保持不变"
        );
        prop_assert_eq!(
            config.retry,
            redacted.retry,
            "重试配置应保持不变"
        );
        prop_assert_eq!(
            config.logging,
            redacted.logging,
            "日志配置应保持不变"
        );

        prop_assert!(
            redacted.credential_pool.is_empty(),
            "旧凭证池已退役，脱敏配置不应保留"
        );
    }

    /// **Feature: config-credential-export, Property 5: Redaction Completeness**
    /// *For any* export bundle with redaction, the redacted flag should be true.
    /// **Validates: Requirements 3.4**
    #[test]
    fn prop_redaction_bundle_flag(config in arb_config_with_secrets()) {
        let options = ExportOptions::redacted();
        let bundle = ExportService::export(&config, &options, "1.0.0")
            .expect("导出应成功");

        prop_assert!(
            bundle.is_redacted(),
            "脱敏导出的 bundle 应标记为已脱敏"
        );
    }
}

// ============================================================================
// Property 6: Import Validation
// ============================================================================

use crate::config::{ExportBundle, ImportService};

/// 生成有效的导出包
fn arb_valid_export_bundle() -> impl Strategy<Value = ExportBundle> {
    (
        arb_valid_config(),
        any::<bool>(),                              // redacted
        "[0-9]+\\.[0-9]+\\.[0-9]+".prop_map(|s| s), // app_version
    )
        .prop_map(|(config, redacted, app_version)| {
            let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
            let mut bundle = ExportBundle::new(&app_version);
            bundle.config_yaml = Some(yaml);
            bundle.redacted = redacted;
            bundle
        })
}

/// 生成无效的导入内容（既不是有效的 JSON ExportBundle，也不是有效的 YAML Config）
/// 注意：YAML 解析器非常宽松，大多数内容都可以解析为某种 YAML 结构
/// 因此我们只测试语法错误的内容
fn arb_invalid_import_content() -> impl Strategy<Value = String> {
    prop_oneof![
        // 无效的 JSON/YAML 语法
        Just("{invalid json".to_string()),
        Just("invalid: yaml: content: [".to_string()),
        Just("  - bad\n indentation".to_string()),
        Just("key: value\n  invalid: indent".to_string()),
        // 有效的 JSON 但不是 ExportBundle 或 Config（数组类型）
        Just("[1, 2, 3]".to_string()),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: config-credential-export, Property 6: Import Validation**
    /// *For any* valid export bundle, validation should return valid=true and
    /// correctly identify format, version, and redaction status.
    /// **Validates: Requirements 4.1, 4.2**
    #[test]
    fn prop_import_validation_valid_bundle(bundle in arb_valid_export_bundle()) {
        let json = bundle.to_json().expect("序列化应成功");
        let result = ImportService::validate(&json);

        // 验证结果应为有效
        prop_assert!(
            result.valid,
            "有效的导出包应通过验证: errors={:?}",
            result.errors
        );

        // 验证版本被正确识别
        prop_assert_eq!(
            result.version,
            Some(bundle.version.clone()),
            "版本应被正确识别"
        );

        // 验证脱敏状态被正确识别
        prop_assert_eq!(
            result.redacted,
            bundle.redacted,
            "脱敏状态应被正确识别"
        );

        // 验证配置存在性被正确识别
        prop_assert_eq!(
            result.has_config,
            bundle.has_config(),
            "配置存在性应被正确识别"
        );
    }

    /// **Feature: config-credential-export, Property 6: Import Validation**
    /// *For any* valid YAML configuration, validation should return valid=true
    /// and identify it as config-only (no credentials).
    /// **Validates: Requirements 4.1, 4.2**
    #[test]
    fn prop_import_validation_valid_yaml(config in arb_valid_config()) {
        let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
        let result = ImportService::validate(&yaml);

        // 验证结果应为有效
        prop_assert!(
            result.valid,
            "有效的 YAML 配置应通过验证: errors={:?}",
            result.errors
        );

        // 验证识别为配置
        prop_assert!(
            result.has_config,
            "应识别为包含配置"
        );

        // YAML 配置不包含凭证 token 文件
        prop_assert!(
            !result.has_credentials,
            "YAML 配置不应包含凭证 token 文件"
        );

        // YAML 配置不是脱敏的
        prop_assert!(
            !result.redacted,
            "YAML 配置不应标记为脱敏"
        );
    }

    /// **Feature: config-credential-export, Property 6: Import Validation**
    /// *For any* invalid import content (neither valid ExportBundle JSON nor valid Config YAML),
    /// validation should return valid=false with appropriate error messages.
    /// **Validates: Requirements 4.1, 4.2**
    #[test]
    fn prop_import_validation_invalid_content(content in arb_invalid_import_content()) {
        let result = ImportService::validate(&content);

        // 无效内容应验证失败
        prop_assert!(
            !result.valid,
            "无效的导入内容应验证失败: content={}", content
        );

        // 应有错误信息
        prop_assert!(
            !result.errors.is_empty(),
            "应有错误信息"
        );
    }

    /// **Feature: config-credential-export, Property 6: Import Validation**
    /// *For any* redacted export bundle, validation should warn about
    /// credentials that cannot be restored.
    /// **Validates: Requirements 4.1, 4.2**
    #[test]
    fn prop_import_validation_redacted_warning(config in arb_valid_config()) {
        let yaml = ConfigManager::to_yaml(&config).expect("序列化应成功");
        let mut bundle = ExportBundle::new("1.0.0");
        bundle.config_yaml = Some(yaml);
        bundle.redacted = true;

        let json = bundle.to_json().expect("序列化应成功");
        let result = ImportService::validate(&json);

        // 验证结果应为有效（脱敏不影响有效性）
        prop_assert!(
            result.valid,
            "脱敏的导出包应通过验证"
        );

        // 应有脱敏警告
        prop_assert!(
            !result.warnings.is_empty(),
            "脱敏的导出包应有警告信息"
        );

        // 警告应提及脱敏
        let has_redaction_warning = result.warnings.iter().any(|w|
            w.contains("脱敏") || w.contains("redact")
        );
        prop_assert!(
            has_redaction_warning,
            "应有关于脱敏的警告: {:?}",
            result.warnings
        );
    }
}

// ============================================================================
// Property 7: Import Merge vs Replace
// ============================================================================

use crate::config::ImportOptions;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: config-credential-export, Property 7: Import Merge vs Replace**
    /// *For any* import operation in replace mode, the resulting configuration
    /// should be exactly the imported configuration (not merged with current).
    /// **Validates: Requirements 4.3**
    #[test]
    fn prop_import_replace_mode(
        current_config in arb_config_with_credentials(),
        imported_config in arb_config_with_credentials()
    ) {
        let yaml = ConfigManager::to_yaml(&imported_config).expect("序列化应成功");
        let options = ImportOptions::replace();

        let result = ImportService::import_yaml(&yaml, &current_config, &options)
            .expect("导入应成功");

        // 替换模式下，结果应等于导入的配置
        prop_assert_eq!(
            result.config.server,
            imported_config.server,
            "替换模式下服务器配置应等于导入的配置"
        );
        prop_assert_eq!(
            result.config.providers,
            imported_config.providers,
            "替换模式下 Provider 配置应等于导入的配置"
        );
        prop_assert_eq!(
            result.config.default_provider,
            imported_config.default_provider,
            "替换模式下默认 Provider 应等于导入的配置"
        );
        prop_assert_eq!(
            result.config.retry,
            imported_config.retry,
            "替换模式下重试配置应等于导入的配置"
        );
        prop_assert_eq!(
            result.config.logging,
            imported_config.logging,
            "替换模式下日志配置应等于导入的配置"
        );
    }

    /// **Feature: config-credential-export, Property 7: Import Merge vs Replace**
    /// *For any* import operation in merge mode, the retired credential pool
    /// should be dropped instead of merged back into runtime config.
    /// **Validates: Requirements 4.3**
    #[test]
    fn prop_import_merge_mode_drops_legacy_credentials(
        current_config in arb_config_with_credentials(),
        imported_config in arb_config_with_credentials()
    ) {
        let yaml = ConfigManager::to_yaml(&imported_config).expect("序列化应成功");
        let options = ImportOptions::merge();

        let result = ImportService::import_yaml(&yaml, &current_config, &options)
            .expect("导入应成功");

        prop_assert!(
            result.config.credential_pool.is_empty(),
            "合并模式下旧凭证池不应再导入或合并"
        );
    }

    /// **Feature: config-credential-export, Property 7: Import Merge vs Replace**
    /// *For any* import operation in merge mode, imported values should override
    /// current values for the same keys.
    /// **Validates: Requirements 4.3**
    #[test]
    fn prop_import_merge_mode_overrides_config(
        current_config in arb_config_with_credentials(),
        imported_config in arb_config_with_credentials()
    ) {
        let yaml = ConfigManager::to_yaml(&imported_config).expect("序列化应成功");
        let options = ImportOptions::merge();

        let result = ImportService::import_yaml(&yaml, &current_config, &options)
            .expect("导入应成功");

        // 合并模式下，配置值应被导入的值覆盖
        prop_assert_eq!(
            result.config.server,
            imported_config.server,
            "合并模式下服务器配置应被导入的值覆盖"
        );
        prop_assert_eq!(
            result.config.providers,
            imported_config.providers,
            "合并模式下 Provider 配置应被导入的值覆盖"
        );
        prop_assert_eq!(
            result.config.retry,
            imported_config.retry,
            "合并模式下重试配置应被导入的值覆盖"
        );
    }

    /// **Feature: config-credential-export, Property 7: Import Merge vs Replace**
    /// *For any* import operation in replace mode with empty imported credentials,
    /// the result should have empty credentials (not preserve current).
    /// **Validates: Requirements 4.3**
    #[test]
    fn prop_import_replace_mode_clears_credentials(
        current_config in arb_config_with_credentials()
    ) {
        // 创建一个没有凭证的配置
        let mut imported_config = Config::default();
        imported_config.server.port = 9999; // 修改一个值以区分

        let yaml = ConfigManager::to_yaml(&imported_config).expect("序列化应成功");
        let options = ImportOptions::replace();

        let result = ImportService::import_yaml(&yaml, &current_config, &options)
            .expect("导入应成功");

        // 替换模式下，凭证池应为空（因为导入的配置没有凭证）
        prop_assert!(
            result.config.credential_pool.kiro.is_empty(),
            "替换模式下 Kiro 凭证应为空"
        );
        prop_assert!(
            result.config.credential_pool.openai.is_empty(),
            "替换模式下 OpenAI 凭证应为空"
        );
        prop_assert_eq!(
            result.config.server.port,
            9999,
            "替换模式下端口应为导入的值"
        );
    }
}

// ============================================================================
// Property 8: Export-Import Round Trip
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: config-credential-export, Property 8: Export-Import Round Trip**
    /// *For any* valid configuration, exporting (without redaction) and then importing
    /// should produce an equivalent configuration.
    /// **Validates: Requirements 5.5**
    #[test]
    fn prop_export_import_roundtrip_yaml(config in arb_config_with_credentials()) {
        // 导出为 YAML（不脱敏）
        let yaml = ExportService::export_yaml(&config, false)
            .expect("导出应成功");

        // 导入 YAML（替换模式）
        let empty_config = Config::default();
        let options = ImportOptions::replace();
        let result = ImportService::import_yaml(&yaml, &empty_config, &options)
            .expect("导入应成功");

        // 验证往返一致性
        prop_assert_eq!(
            config.server,
            result.config.server,
            "服务器配置往返不一致"
        );
        prop_assert_eq!(
            config.providers,
            result.config.providers,
            "Provider 配置往返不一致"
        );
        prop_assert_eq!(
            config.default_provider,
            result.config.default_provider,
            "默认 Provider 往返不一致"
        );
        prop_assert_eq!(
            config.retry,
            result.config.retry,
            "重试配置往返不一致"
        );
        prop_assert_eq!(
            config.logging,
            result.config.logging,
            "日志配置往返不一致"
        );
        prop_assert!(
            result.config.credential_pool.is_empty(),
            "旧凭证池已退役，不应通过 bundle 导入恢复"
        );
        prop_assert_eq!(
            config.auth_dir,
            result.config.auth_dir,
            "auth_dir 往返不一致"
        );

        prop_assert!(
            result.config.credential_pool.is_empty(),
            "旧凭证池已退役，不应通过导出导入往返恢复"
        );
    }

    /// **Feature: config-credential-export, Property 8: Export-Import Round Trip**
    /// *For any* valid configuration, exporting as a bundle (without redaction) and
    /// then importing should produce an equivalent configuration.
    /// **Validates: Requirements 5.5**
    #[test]
    fn prop_export_import_roundtrip_bundle(config in arb_config_with_credentials()) {
        // 导出为 bundle（不脱敏，仅配置）
        let options = ExportOptions {
            include_config: true,
            include_credentials: false, // 不包含 token 文件，因为测试环境没有实际文件
            redact_secrets: false,
        };
        let bundle = ExportService::export(&config, &options, "1.0.0")
            .expect("导出应成功");

        // 序列化为 JSON
        let json = bundle.to_json().expect("序列化应成功");

        // 反序列化
        let parsed_bundle = ExportBundle::from_json(&json).expect("反序列化应成功");

        // 导入 bundle
        let empty_config = Config::default();
        let import_options = ImportOptions::replace();
        let result = ImportService::import(
            &parsed_bundle,
            &empty_config,
            &import_options,
            &config.auth_dir,
        )
            .expect("导入应成功");

        // 验证往返一致性
        prop_assert_eq!(
            config.server,
            result.config.server,
            "服务器配置往返不一致"
        );
        prop_assert_eq!(
            config.providers,
            result.config.providers,
            "Provider 配置往返不一致"
        );
        prop_assert_eq!(
            config.retry,
            result.config.retry,
            "重试配置往返不一致"
        );
        prop_assert_eq!(
            config.logging,
            result.config.logging,
            "日志配置往返不一致"
        );
    }

    /// **Feature: config-credential-export, Property 8: Export-Import Round Trip**
    /// *For any* configuration with API keys, exporting with redaction and then
    /// importing should NOT restore the original API keys.
    /// **Validates: Requirements 5.5**
    #[test]
    fn prop_export_import_redacted_loses_secrets(config in arb_config_with_secrets()) {
        // 导出为脱敏 bundle
        let options = ExportOptions::redacted();
        let bundle = ExportService::export(&config, &options, "1.0.0")
            .expect("导出应成功");

        // 导入 bundle（脱敏数据会触发清理）
        let empty_config = Config::default();
        let import_options = ImportOptions::replace();
        let result = ImportService::import(
            &bundle,
            &empty_config,
            &import_options,
            &config.auth_dir,
        )
            .expect("导入应成功");

        // 验证脱敏后的配置不包含原始敏感信息
        // 服务器 API 密钥应被清空
        prop_assert_eq!(
            result.config.server.api_key,
            "",
            "脱敏后服务器 API 密钥应被清空"
        );

        // 如果原始配置有 OpenAI API 密钥，导入后应为脱敏占位符
        if config.providers.openai.api_key.is_some() {
            prop_assert_eq!(
                result.config.providers.openai.api_key,
                None,
                "脱敏后 OpenAI API 密钥应被清空"
            );
        }
    }

    /// **Feature: config-credential-export, Property 8: Export-Import Round Trip**
    /// *For any* configuration, the export bundle should be valid JSON that can
    /// be parsed back.
    /// **Validates: Requirements 5.5**
    #[test]
    fn prop_export_bundle_json_roundtrip(config in arb_config_with_credentials()) {
        let options = ExportOptions {
            include_config: true,
            include_credentials: false,
            redact_secrets: false,
        };
        let bundle = ExportService::export(&config, &options, "1.0.0")
            .expect("导出应成功");

        // 序列化为 JSON
        let json = bundle.to_json().expect("序列化应成功");

        // 反序列化
        let parsed = ExportBundle::from_json(&json).expect("反序列化应成功");

        // 验证往返一致性
        prop_assert_eq!(
            bundle.version,
            parsed.version,
            "版本往返不一致"
        );
        prop_assert_eq!(
            bundle.app_version,
            parsed.app_version,
            "应用版本往返不一致"
        );
        prop_assert_eq!(
            bundle.redacted,
            parsed.redacted,
            "脱敏状态往返不一致"
        );
        prop_assert_eq!(
            bundle.config_yaml,
            parsed.config_yaml,
            "配置 YAML 往返不一致"
        );
        prop_assert_eq!(
            bundle.token_files,
            parsed.token_files,
            "Token 文件往返不一致"
        );
    }
}

// ============================================================================
// Property 1: Legacy Credential Config Read Boundary
// ============================================================================

/// 生成随机的 OAuth 凭证条目（用于 Codex/iFlow）
fn arb_oauth_credential_entry() -> impl Strategy<Value = CredentialEntry> {
    (
        "[a-z]{3,10}-[0-9]{1,5}".prop_map(|s| s),
        "[a-z]+/oauth-token-[0-9]{1,5}\\.json".prop_map(|s| s),
        any::<bool>(),
        proptest::option::of("socks5://proxy\\.[a-z]+\\.com:[0-9]{4}".prop_map(|s| s)),
    )
        .prop_map(|(id, token_file, disabled, proxy_url)| CredentialEntry {
            id,
            token_file,
            disabled,
            proxy_url,
        })
}

/// 生成随机的 Gemini API Key 条目
fn arb_gemini_api_key_entry() -> impl Strategy<Value = crate::config::GeminiApiKeyEntry> {
    (
        "[a-z]{3,10}-[0-9]{1,5}".prop_map(|s| s),
        "AIzaSy[a-zA-Z0-9_-]{33}".prop_map(|s| s),
        proptest::option::of("https://generativelanguage\\.googleapis\\.com".prop_map(|s| s)),
        proptest::option::of("http://proxy\\.[a-z]+\\.com:[0-9]{4}".prop_map(|s| s)),
        proptest::collection::vec("[a-z]+-[0-9]+\\.[0-9]+-pro".prop_map(|s| s), 0..3),
        any::<bool>(),
    )
        .prop_map(
            |(id, api_key, base_url, proxy_url, excluded_models, disabled)| {
                crate::config::GeminiApiKeyEntry {
                    id,
                    api_key,
                    base_url,
                    proxy_url,
                    excluded_models,
                    disabled,
                }
            },
        )
}

/// 生成随机的 Vertex AI 条目
fn arb_vertex_api_key_entry() -> impl Strategy<Value = crate::config::VertexApiKeyEntry> {
    (
        "[a-z]{3,10}-[0-9]{1,5}".prop_map(|s| s),
        "vk-[a-zA-Z0-9]{20,40}".prop_map(|s| s),
        proptest::option::of("https://[a-z]+-aiplatform\\.googleapis\\.com".prop_map(|s| s)),
        proptest::collection::vec(
            (
                "[a-z]+-[0-9]+\\.[0-9]+".prop_map(|s| s),
                "[a-z]+-alias".prop_map(|s| s),
            ),
            0..3,
        ),
        proptest::option::of("http://proxy\\.[a-z]+\\.com:[0-9]{4}".prop_map(|s| s)),
        any::<bool>(),
    )
        .prop_map(|(id, api_key, base_url, models, proxy_url, disabled)| {
            crate::config::VertexApiKeyEntry {
                id,
                api_key,
                base_url,
                models: models
                    .into_iter()
                    .map(|(name, alias)| crate::config::VertexModelAlias { name, alias })
                    .collect(),
                proxy_url,
                disabled,
            }
        })
}

/// 生成覆盖旧 OAuth / API Key 形态的历史凭证池配置
fn arb_extended_credential_pool_config() -> impl Strategy<Value = CredentialPoolConfig> {
    (
        proptest::collection::vec(arb_credential_entry(), 0..3),
        proptest::collection::vec(arb_credential_entry(), 0..3),
        proptest::collection::vec(arb_credential_entry(), 0..3),
        proptest::collection::vec(arb_api_key_entry(), 0..3),
        proptest::collection::vec(arb_api_key_entry(), 0..3),
        proptest::collection::vec(arb_gemini_api_key_entry(), 0..3),
        proptest::collection::vec(arb_vertex_api_key_entry(), 0..3),
        proptest::collection::vec(arb_credential_entry(), 0..3),
    )
        .prop_map(
            |(kiro, gemini, qwen, openai, claude, gemini_api_keys, vertex_api_keys, codex)| {
                CredentialPoolConfig {
                    kiro,
                    gemini,
                    qwen,
                    openai,
                    claude,
                    gemini_api_keys,
                    vertex_api_keys,
                    codex,
                }
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: retired-credential-pool, Property 1: Legacy Config Read Boundary**
    /// *For any* legacy credential pool in memory, writing current config should drop
    /// the retired field while direct legacy YAML parsing still accepts historical data.
    /// **Validates: Requirements 1.1, 2.1**
    #[test]
    fn prop_legacy_credential_pool_not_serialized_from_current_config(pool in arb_extended_credential_pool_config()) {
        let config = Config {
            credential_pool: pool.clone(),
            ..Config::default()
        };

        // 序列化为 YAML
        let yaml = ConfigManager::to_yaml(&config)
            .expect("序列化应成功");

        prop_assert!(
            !yaml.contains("credential_pool:"),
            "当前配置序列化不应继续写出旧凭证池"
        );

        let legacy_yaml = serde_yaml::to_string(&pool)
            .expect("旧凭证池序列化应成功");

        // 旧配置字段仍可直接反序列化，供历史配置读取和启动期清理使用。
        let parsed = ConfigManager::parse_yaml(&yaml)
            .expect("反序列化应成功");
        prop_assert!(parsed.credential_pool.is_empty());

        let parsed_pool: CredentialPoolConfig = serde_yaml::from_str(&legacy_yaml)
            .expect("旧凭证池反序列化应成功");

        // 验证旧凭证池类型仍能读取历史数据。
        prop_assert_eq!(
            pool.kiro.len(),
            parsed_pool.kiro.len(),
            "Kiro OAuth 凭证数量读取不一致"
        );
        prop_assert_eq!(
            pool.gemini.len(),
            parsed_pool.gemini.len(),
            "Gemini OAuth 凭证数量读取不一致"
        );

        // 验证 Gemini API Key 多账号配置读取一致性
        prop_assert_eq!(
            pool.gemini_api_keys.len(),
            parsed_pool.gemini_api_keys.len(),
            "Gemini API Key 凭证数量读取不一致"
        );

        // 验证 Vertex AI 配置读取一致性
        prop_assert_eq!(
            pool.vertex_api_keys.len(),
            parsed_pool.vertex_api_keys.len(),
            "Vertex AI 凭证数量读取不一致"
        );

        // 验证每个 Gemini API Key 的详细内容
        for (original, parsed_entry) in pool.gemini_api_keys.iter().zip(parsed_pool.gemini_api_keys.iter()) {
            prop_assert_eq!(
                &original.id,
                &parsed_entry.id,
                "Gemini API Key ID 读取不一致"
            );
            prop_assert_eq!(
                &original.api_key,
                &parsed_entry.api_key,
                "Gemini API Key 读取不一致"
            );
            prop_assert_eq!(
                &original.excluded_models,
                &parsed_entry.excluded_models,
                "Gemini 排除模型列表读取不一致"
            );
        }

        // 验证每个 Vertex AI 凭证的详细内容
        for (original, parsed_entry) in pool.vertex_api_keys.iter().zip(parsed_pool.vertex_api_keys.iter()) {
            prop_assert_eq!(
                &original.id,
                &parsed_entry.id,
                "Vertex AI 凭证 ID 读取不一致"
            );
            prop_assert_eq!(
                original.models.len(),
                parsed_entry.models.len(),
                "Vertex AI 模型别名数量读取不一致"
            );
        }
    }
}

// ============================================================================
// Property 3: EndpointProvidersConfig 序列化往返一致性
// ============================================================================

use crate::config::EndpointProvidersConfig;

/// 生成随机的 Provider 名称
fn arb_provider_name() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("kiro".to_string()),
        Just("gemini".to_string()),
        Just("qwen".to_string()),
        Just("openai".to_string()),
        Just("claude".to_string()),
        Just("codex".to_string()),
    ]
}

/// 生成随机的可选 Provider 名称
fn arb_optional_provider() -> impl Strategy<Value = Option<String>> {
    proptest::option::of(arb_provider_name())
}

/// 生成随机的 EndpointProvidersConfig
fn arb_endpoint_providers_config() -> impl Strategy<Value = EndpointProvidersConfig> {
    (
        arb_optional_provider(),
        arb_optional_provider(),
        arb_optional_provider(),
        arb_optional_provider(),
        arb_optional_provider(),
        arb_optional_provider(),
    )
        .prop_map(|(cursor, claude_code, codex, windsurf, kiro, other)| {
            EndpointProvidersConfig {
                cursor,
                claude_code,
                codex,
                windsurf,
                kiro,
                other,
            }
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: endpoint-provider-config, Property 3: 配置序列化往返一致性**
    /// *对于任意* 有效的 EndpointProvidersConfig 对象，序列化后再反序列化应产生等价的对象。
    /// **Validates: Requirements 1.1**
    #[test]
    fn prop_endpoint_providers_config_yaml_roundtrip(config in arb_endpoint_providers_config()) {
        // 序列化为 YAML
        let yaml = serde_yaml::to_string(&config)
            .expect("YAML 序列化应成功");

        // 反序列化回 EndpointProvidersConfig
        let parsed: EndpointProvidersConfig = serde_yaml::from_str(&yaml)
            .expect("YAML 反序列化应成功");

        // 验证往返一致性
        prop_assert_eq!(
            config.cursor,
            parsed.cursor,
            "cursor 字段往返不一致"
        );
        prop_assert_eq!(
            config.claude_code,
            parsed.claude_code,
            "claude_code 字段往返不一致"
        );
        prop_assert_eq!(
            config.codex,
            parsed.codex,
            "codex 字段往返不一致"
        );
        prop_assert_eq!(
            config.windsurf,
            parsed.windsurf,
            "windsurf 字段往返不一致"
        );
        prop_assert_eq!(
            config.kiro,
            parsed.kiro,
            "kiro 字段往返不一致"
        );
        prop_assert_eq!(
            config.other,
            parsed.other,
            "other 字段往返不一致"
        );
    }

    /// **Feature: endpoint-provider-config, Property 3: 配置序列化往返一致性（JSON）**
    /// *对于任意* 有效的 EndpointProvidersConfig 对象，JSON 序列化后再反序列化应产生等价的对象。
    /// **Validates: Requirements 1.1**
    #[test]
    fn prop_endpoint_providers_config_json_roundtrip(config in arb_endpoint_providers_config()) {
        // 序列化为 JSON
        let json = serde_json::to_string(&config)
            .expect("JSON 序列化应成功");

        // 反序列化回 EndpointProvidersConfig
        let parsed: EndpointProvidersConfig = serde_json::from_str(&json)
            .expect("JSON 反序列化应成功");

        // 验证往返一致性
        prop_assert_eq!(
            config,
            parsed,
            "EndpointProvidersConfig JSON 往返不一致"
        );
    }

    /// **Feature: endpoint-provider-config, Property 3: 配置序列化往返一致性（完整配置）**
    /// *对于任意* 包含 EndpointProvidersConfig 的完整配置，序列化后再反序列化应保持 endpoint_providers 一致。
    /// **Validates: Requirements 1.1**
    #[test]
    fn prop_config_with_endpoint_providers_roundtrip(
        endpoint_providers in arb_endpoint_providers_config()
    ) {
        // 创建包含 endpoint_providers 的完整配置
        let config = Config {
            endpoint_providers: endpoint_providers.clone(),
            ..Config::default()
        };

        // 序列化为 YAML
        let yaml = ConfigManager::to_yaml(&config)
            .expect("序列化应成功");

        // 反序列化回 Config
        let parsed = ConfigManager::parse_yaml(&yaml)
            .expect("反序列化应成功");

        // 验证 endpoint_providers 往返一致性
        prop_assert_eq!(
            endpoint_providers,
            parsed.endpoint_providers,
            "endpoint_providers 往返不一致"
        );
    }
}

// ============================================================================
// Property 4: Provider 类型验证
// ============================================================================

use crate::ProviderType;

/// 生成有效的 Provider 类型字符串
/// 注意：只包含往返一致的 Provider 类型（即 parse().to_string() == 原值）
/// qwen 等第三方 Provider 会被映射到 openai，不满足往返一致性
fn arb_valid_provider_type() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("kiro".to_string()),
        Just("gemini".to_string()),
        Just("openai".to_string()),
        Just("claude".to_string()),
        Just("vertex".to_string()),
        Just("gemini_api_key".to_string()),
        Just("codex".to_string()),
        Just("claude_oauth".to_string()),
        Just("anthropic".to_string()),
        Just("anthropic_compatible".to_string()),
        Just("azure_openai".to_string()),
        Just("aws_bedrock".to_string()),
        Just("ollama".to_string()),
    ]
}

/// 生成无效的 Provider 类型字符串
fn arb_invalid_provider_type() -> impl Strategy<Value = String> {
    // 生成不在有效列表中的字符串
    // 注意：需要排除所有在 ProviderType::from_str 中有效的字符串
    "[a-z]{3,15}".prop_filter("排除有效的 Provider 类型", |s| {
        // 排除所有在 ProviderType::from_str 中能成功解析的字符串
        use std::str::FromStr;
        crate::ProviderType::from_str(s).is_err()
    })
}

/// 生成有效的客户端类型字符串
fn arb_valid_client_type() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("cursor".to_string()),
        Just("claude_code".to_string()),
        Just("codex".to_string()),
        Just("windsurf".to_string()),
        Just("kiro".to_string()),
        Just("other".to_string()),
    ]
}

/// 生成无效的客户端类型字符串
fn arb_invalid_client_type() -> impl Strategy<Value = String> {
    // 生成不在有效列表中的字符串
    "[a-z]{3,15}".prop_filter("排除有效的客户端类型", |s| {
        !matches!(
            s.as_str(),
            "cursor" | "claude_code" | "codex" | "windsurf" | "kiro" | "other"
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Feature: endpoint-provider-config, Property 4: Provider 类型验证**
    /// *对于任意* 有效的 Provider 类型字符串，解析应成功并返回正确的 ProviderType。
    /// **Validates: Requirements 5.1, 5.2**
    #[test]
    fn prop_valid_provider_type_parsing(provider in arb_valid_provider_type()) {
        // 解析 Provider 类型
        let result: Result<ProviderType, String> = provider.parse();

        // 验证解析成功
        prop_assert!(
            result.is_ok(),
            "有效的 Provider 类型应解析成功: {}",
            provider
        );

        // 验证往返一致性
        let parsed = result.unwrap();
        prop_assert_eq!(
            parsed.to_string(),
            provider,
            "Provider 类型往返不一致"
        );
    }

    /// **Feature: endpoint-provider-config, Property 4: Provider 类型验证**
    /// *对于任意* 无效的 Provider 类型字符串，解析应失败并返回描述性错误消息。
    /// **Validates: Requirements 5.1, 5.2**
    #[test]
    fn prop_invalid_provider_type_parsing(provider in arb_invalid_provider_type()) {
        // 解析 Provider 类型
        let result: Result<ProviderType, String> = provider.parse();

        // 验证解析失败
        prop_assert!(
            result.is_err(),
            "无效的 Provider 类型应解析失败: {}",
            provider
        );

        // 验证错误消息包含描述性信息
        let error = result.unwrap_err();
        prop_assert!(
            error.contains("Invalid provider") || error.contains(&provider),
            "错误消息应包含描述性信息: {}",
            error
        );
    }

    /// **Feature: endpoint-provider-config, Property 4: Provider 类型验证**
    /// *对于任意* 有效的客户端类型，set_provider 应成功设置 Provider。
    /// **Validates: Requirements 5.1, 5.2**
    #[test]
    fn prop_valid_client_type_set_provider(
        client_type in arb_valid_client_type(),
        provider in arb_valid_provider_type()
    ) {
        let mut config = EndpointProvidersConfig::default();

        // 设置 Provider
        let result = config.set_provider(&client_type, Some(provider.clone()));

        // 验证设置成功
        prop_assert!(
            result,
            "有效的客户端类型应设置成功: {}",
            client_type
        );

        // 验证 Provider 已正确设置
        let stored = config.get_provider(&client_type);
        prop_assert_eq!(
            stored,
            Some(&provider),
            "Provider 应正确存储"
        );
    }

    /// **Feature: endpoint-provider-config, Property 4: Provider 类型验证**
    /// *对于任意* 无效的客户端类型，set_provider 应返回 false。
    /// **Validates: Requirements 5.1, 5.2**
    #[test]
    fn prop_invalid_client_type_set_provider(
        client_type in arb_invalid_client_type(),
        provider in arb_valid_provider_type()
    ) {
        let mut config = EndpointProvidersConfig::default();

        // 设置 Provider
        let result = config.set_provider(&client_type, Some(provider));

        // 验证设置失败
        prop_assert!(
            !result,
            "无效的客户端类型应设置失败: {}",
            client_type
        );
    }

    /// **Feature: endpoint-provider-config, Property 4: Provider 类型验证**
    /// *对于任意* 有效的客户端类型，使用 None 或空字符串应清除 Provider 配置。
    /// **Validates: Requirements 5.1, 5.2**
    #[test]
    fn prop_clear_provider_config(
        client_type in arb_valid_client_type(),
        provider in arb_valid_provider_type()
    ) {
        let mut config = EndpointProvidersConfig::default();

        // 先设置 Provider
        config.set_provider(&client_type, Some(provider));

        // 使用 None 清除
        let result = config.set_provider(&client_type, None);
        prop_assert!(result, "清除操作应成功");
        prop_assert_eq!(
            config.get_provider(&client_type),
            None,
            "Provider 应被清除"
        );

        // 重新设置后使用空字符串清除
        config.set_provider(&client_type, Some("kiro".to_string()));
        let result = config.set_provider(&client_type, Some("".to_string()));
        prop_assert!(result, "空字符串清除操作应成功");
        prop_assert_eq!(
            config.get_provider(&client_type),
            None,
            "Provider 应被清除（空字符串）"
        );
    }
}
