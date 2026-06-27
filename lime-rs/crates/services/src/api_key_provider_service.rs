//! API Key Provider 管理服务
//!
//! 提供 API Key Provider 的 CRUD 操作、加密存储和轮询负载均衡功能。
//!
//! **Feature: provider-ui-refactor**
//! **Validates: Requirements 7.3, 9.1, 9.2, 9.3**

use crate::provider_type_mapping::{
    api_provider_type_to_runtime_provider_type, is_custom_provider_id,
    resolve_runtime_provider_type, runtime_provider_type_to_api_type,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::Utc;
use lime_core::api_host_utils::{
    is_openai_responses_compatible_host, normalize_openai_compatible_api_host,
    normalize_openai_model_discovery_host,
};
use lime_core::database::dao::api_key_provider::{
    infer_managed_provider_type, ApiKeyEntry, ApiKeyProvider, ApiKeyProviderDao,
    ApiProviderPromptCacheMode, ApiProviderType, ProviderGroup, ProviderWithKeys,
};
use lime_core::database::system_providers::{get_system_providers, to_api_key_provider};
use lime_core::database::DbConnection;
use lime_core::models::runtime_provider_model::{
    runtime_api_key_credential_uuid, ProviderPromptCacheMode, RuntimeCredentialData,
    RuntimeProviderCredential, RuntimeProviderType,
};
use lime_core::provider_prompt_cache_support::is_known_automatic_anthropic_compatible_host;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

const SENSENOVA_OLD_OPENAI_COMPATIBLE_API_HOST: &str =
    "https://api.sensenova.cn/compatible-mode/v1";
const SENSENOVA_OPENAI_COMPATIBLE_API_HOST: &str = "https://api.sensenova.cn/compatible-mode/v2";
const OPENAI_RESPONSES_IMAGE_TEST_OUTER_MODEL: &str = "gpt-5.5";

// ============================================================================
// 连接测试结果
// ============================================================================

/// 连接测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionTestResult {
    /// 是否成功
    pub success: bool,
    /// 延迟（毫秒）
    pub latency_ms: Option<u64>,
    /// 错误信息
    pub error: Option<String>,
    /// 模型列表（如果使用 models 端点测试）
    pub models: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::{
        ApiKeyProviderService, SENSENOVA_OLD_OPENAI_COMPATIBLE_API_HOST,
        SENSENOVA_OPENAI_COMPATIBLE_API_HOST,
    };
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
    use chrono::Utc;
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::dao::api_key_provider::{
        ApiKeyEntry, ApiKeyProviderDao, ApiProviderPromptCacheMode,
    };
    use lime_core::database::{init_database, migration, schema, DbConnection};
    use rusqlite::Connection;
    use rusqlite::OptionalExtension;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn resolve_legacy_machine_id() -> String {
        if let Ok(id) = std::fs::read_to_string("/etc/machine-id") {
            return id.trim().to_string();
        }
        if let Ok(id) = std::fs::read_to_string("/var/lib/dbus/machine-id") {
            return id.trim().to_string();
        }
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("ioreg")
                .args(["-rd1", "-c", "IOPlatformExpertDevice"])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.contains("IOPlatformUUID") {
                        if let Some(uuid) = line.split('"').nth(3) {
                            return uuid.to_string();
                        }
                    }
                }
            }
        }
        "proxycast-default-machine-id".to_string()
    }

    fn encrypt_with_legacy_proxycast_format(plaintext: &str) -> String {
        use sha2::{Digest, Sha256};

        let machine_id = resolve_legacy_machine_id();
        let mut hasher = Sha256::new();
        hasher.update(machine_id.as_bytes());
        hasher.update(b"proxycast-api-key-encryption-salt");
        let key = hasher.finalize().to_vec();

        let encrypted: Vec<u8> = plaintext
            .as_bytes()
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ key[i % key.len()])
            .collect();
        BASE64.encode(encrypted)
    }

    fn init_test_database() -> DbConnection {
        let conn = Connection::open_in_memory().expect("创建内存数据库失败");
        schema::create_tables(&conn).expect("初始化表结构失败");
        migration::migrate_from_json(&conn).expect("执行数据库迁移失败");
        Arc::new(Mutex::new(conn))
    }

    fn resolve_real_codex_provider_id(
        db: &lime_core::database::DbConnection,
    ) -> Result<String, String> {
        if let Some(explicit) =
            lime_core::env_compat::var(&["LIME_REAL_PROVIDER_ID", "PROXYCAST_REAL_PROVIDER_ID"])
        {
            let trimmed = explicit.trim();
            if !trimmed.is_empty() {
                return Ok(trimmed.to_string());
            }
        }

        let conn = db.lock().map_err(|e| format!("数据库锁定失败: {e}"))?;
        conn.query_row(
            r#"
            SELECT p.id
            FROM api_key_providers p
            JOIN api_keys k ON k.provider_id = p.id
            WHERE p.enabled = 1
              AND k.enabled = 1
              AND p.type = 'codex'
            ORDER BY p.updated_at DESC
            LIMIT 1
            "#,
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|e| format!("查询本地 CLI Provider 失败: {e}"))?
        .ok_or_else(|| "未找到可用的本地 CLI Provider，请先在设置中配置并启用".to_string())
    }

    #[test]
    fn test_build_codex_responses_request_input_list() {
        let req = ApiKeyProviderService::build_codex_responses_request("gpt-5", "hello");
        assert!(req.get("input").is_some());
        let input = req["input"].as_array().expect("input should be array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"].as_str(), Some("user"));
        assert_eq!(input[0]["content"][0]["type"].as_str(), Some("input_text"));
        assert_eq!(input[0]["content"][0]["text"].as_str(), Some("hello"));
    }

    #[test]
    fn test_parse_codex_responses_sse_content_delta() {
        let body = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n\
data: {\"type\":\"response.output_text.delta\",\"delta\":\"!\"}\n\n\
data: [DONE]\n";
        let content = ApiKeyProviderService::parse_codex_responses_sse_content(body);
        assert_eq!(content, "hi!");
    }

    #[test]
    fn test_uses_anthropic_protocol() {
        assert!(ApiKeyProviderService::uses_anthropic_protocol(
            ApiProviderType::Anthropic
        ));
        assert!(ApiKeyProviderService::uses_anthropic_protocol(
            ApiProviderType::AnthropicCompatible
        ));
        assert!(!ApiKeyProviderService::uses_anthropic_protocol(
            ApiProviderType::Openai
        ));
    }

    #[test]
    fn test_format_anthropic_http_api_error_uses_protocol_level_guidance() {
        let body = r#"{"type":"error","error":{"type":"authentication_error","message":"login fail: Please carry the API secret key in the 'Authorization' field of the request header"}}"#;

        let formatted = ApiKeyProviderService::format_anthropic_http_api_error(
            reqwest::StatusCode::UNAUTHORIZED,
            body,
            "https://example.com/anthropic",
            ApiProviderType::AnthropicCompatible,
            "k-cp-looks-truncated",
        );

        assert!(formatted.contains("Lime 已按 Anthropic 协议发送兼容鉴权"));
        assert!(formatted.contains("核对 API Key 是否完整"));
        assert!(!formatted.contains("MiniMax"));
    }

    #[test]
    fn test_format_anthropic_http_api_error_keeps_base_message_for_non_anthropic_protocol() {
        let body = r#"{"type":"error","error":{"type":"authentication_error","message":"Please carry the API secret key in the 'Authorization' field of the request header"}}"#;

        let formatted = ApiKeyProviderService::format_anthropic_http_api_error(
            reqwest::StatusCode::UNAUTHORIZED,
            body,
            "https://example.com/compatible",
            ApiProviderType::Openai,
            "k-cp-looks-truncated",
        );

        assert_eq!(
            formatted,
            "API 返回错误: 401 Unauthorized - Please carry the API secret key in the 'Authorization' field of the request header"
        );
    }

    #[test]
    fn test_format_anthropic_http_api_error_marks_upstream_overload() {
        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded_error (529)"}}"#;

        let formatted = ApiKeyProviderService::format_anthropic_http_api_error(
            reqwest::StatusCode::from_u16(529).expect("status"),
            body,
            "https://example.com/anthropic",
            ApiProviderType::AnthropicCompatible,
            "test-key",
        );

        assert!(formatted.contains("上游 Anthropic 兼容接口当前暂时过载或限流"));
        assert!(formatted.contains("overloaded_error (529)"));
        assert!(formatted.contains("这通常不是 Base URL、鉴权头或模型配置错误"));
    }

    #[test]
    fn test_is_transient_anthropic_upstream_error_supports_json_and_status_detection() {
        assert!(
            ApiKeyProviderService::is_transient_anthropic_upstream_error(
                reqwest::StatusCode::from_u16(529).expect("status"),
                r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded_error (529)"}}"#,
            )
        );
        assert!(
            ApiKeyProviderService::is_transient_anthropic_upstream_error(
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                r#"{"error":{"message":"rate_limit exceeded"}}"#,
            )
        );
        assert!(
            !ApiKeyProviderService::is_transient_anthropic_upstream_error(
                reqwest::StatusCode::UNAUTHORIZED,
                r#"{"error":{"message":"invalid api key"}}"#,
            )
        );
    }

    #[test]
    fn test_pick_test_model_priority() {
        let with_explicit = ApiKeyProviderService::pick_test_model(
            Some("explicit-model".to_string()),
            &["custom-model".to_string()],
            &["fallback-model".to_string(), "fallback-mini".to_string()],
        );
        assert_eq!(with_explicit.as_deref(), Some("explicit-model"));

        let with_custom = ApiKeyProviderService::pick_test_model(
            None,
            &["custom-model".to_string()],
            &["fallback-model".to_string(), "fallback-mini".to_string()],
        );
        assert_eq!(with_custom.as_deref(), Some("fallback-mini"));

        let with_matching_custom = ApiKeyProviderService::pick_test_model(
            None,
            &["MiniMax-M2.7".to_string()],
            &["MiniMax-M2.7".to_string(), "MiniMax-M2.5".to_string()],
        );
        assert_eq!(with_matching_custom.as_deref(), Some("MiniMax-M2.7"));

        let with_local_fallback =
            ApiKeyProviderService::pick_test_model(None, &[], &["fallback-model".to_string()]);
        assert_eq!(with_local_fallback.as_deref(), Some("fallback-model"));

        let with_preferred_fallback = ApiKeyProviderService::pick_test_model(
            None,
            &[],
            &[
                "gpt-5.2-pro".to_string(),
                "gpt-5-nano".to_string(),
                "gpt-4.1-mini".to_string(),
            ],
        );
        assert_eq!(with_preferred_fallback.as_deref(), Some("gpt-5-nano"));

        let none = ApiKeyProviderService::pick_test_model(None, &[], &[]);
        assert!(none.is_none());
    }

    #[test]
    fn test_pick_test_model_candidates_tries_explicit_then_declared_models() {
        let candidates = ApiKeyProviderService::pick_test_model_candidates(
            Some("stale-chat".to_string()),
            &[
                "stable-chat".to_string(),
                "stale-chat".to_string(),
                "preview-chat".to_string(),
            ],
            &["fallback-mini".to_string()],
        );

        assert_eq!(
            candidates,
            vec![
                "stale-chat".to_string(),
                "fallback-mini".to_string(),
                "stable-chat".to_string(),
                "preview-chat".to_string(),
            ]
        );
    }

    #[test]
    fn test_should_try_next_test_model_after_model_unavailable_errors() {
        assert!(
            ApiKeyProviderService::should_try_next_test_model_after_error(
                "API 返回错误: 400 Bad Request - Param Incorrect"
            )
        );
        assert!(
            ApiKeyProviderService::should_try_next_test_model_after_error(
                "Request failed: Bad request (400): Not supported model stale-chat"
            )
        );
        assert!(
            ApiKeyProviderService::should_try_next_test_model_after_error(
                "当前模型未在租户白名单中开放"
            )
        );
        assert!(
            !ApiKeyProviderService::should_try_next_test_model_after_error(
                "API 返回错误: 401 Unauthorized - invalid api key"
            )
        );
    }

    #[test]
    fn test_openai_image_connection_test_detection_prefers_explicit_or_image_only_providers() {
        assert!(
            ApiKeyProviderService::should_use_openai_image_connection_test(
                Some("gpt-images-2"),
                &[],
                &[]
            )
        );
        assert!(
            ApiKeyProviderService::should_use_openai_image_connection_test(
                None,
                &["gpt-images-2".to_string()],
                &[]
            )
        );
        assert!(
            !ApiKeyProviderService::should_use_openai_image_connection_test(
                None,
                &["gpt-images-2".to_string(), "gpt-5.2".to_string()],
                &[]
            )
        );
        assert!(
            !ApiKeyProviderService::should_use_openai_image_connection_test(
                Some("gpt-5.2"),
                &["gpt-images-2".to_string()],
                &[]
            )
        );
    }

    #[test]
    fn test_pick_openai_image_test_model_prefers_image_candidates() {
        assert_eq!(
            ApiKeyProviderService::pick_openai_image_test_model(
                Some("gpt-images-2".to_string()),
                &[],
                &[],
            )
            .as_deref(),
            Some("gpt-images-2")
        );
        assert_eq!(
            ApiKeyProviderService::pick_openai_image_test_model(
                None,
                &["gpt-images-2".to_string(), "gpt-5.2".to_string()],
                &[],
            )
            .as_deref(),
            Some("gpt-images-2")
        );
        assert_eq!(
            ApiKeyProviderService::pick_openai_image_test_model(
                None,
                &[],
                &["dall-e-3".to_string(), "gpt-4.1".to_string()],
            )
            .as_deref(),
            Some("dall-e-3")
        );
        assert!(ApiKeyProviderService::pick_openai_image_test_model(
            Some("gpt-5.2".to_string()),
            &["gpt-5.2".to_string()],
            &[],
        )
        .is_none());
    }

    #[test]
    fn test_build_openai_images_url_reuses_existing_v1_path() {
        assert_eq!(
            ApiKeyProviderService::build_openai_images_url("https://airgate.k8ray.com/v1"),
            "https://airgate.k8ray.com/v1/images/generations"
        );
        assert_eq!(
            ApiKeyProviderService::build_openai_images_url("https://api.openai.com"),
            "https://api.openai.com/v1/images/generations"
        );
        assert_eq!(
            ApiKeyProviderService::build_openai_images_url(
                "https://gateway.example.com/proxy/responses"
            ),
            "https://gateway.example.com/proxy/v1/images/generations"
        );
    }

    #[test]
    fn test_build_nearai_models_api_url_uses_public_catalog_path() {
        assert_eq!(
            ApiKeyProviderService::build_nearai_models_api_url("https://cloud-api.near.ai/v1"),
            "https://cloud-api.near.ai/v1/model/list"
        );
        assert_eq!(
            ApiKeyProviderService::build_nearai_models_api_url(
                "https://cloud-api.near.ai/v1/chat/completions"
            ),
            "https://cloud-api.near.ai/v1/model/list"
        );
        assert!(ApiKeyProviderService::is_nearai_provider(
            "nearai",
            "https://example.com/v1"
        ));
    }

    #[test]
    fn test_prefers_openai_responses_endpoint_detects_codex_base_path() {
        assert!(ApiKeyProviderService::prefers_openai_responses_endpoint(
            "https://gateway.example.com/codex"
        ));
        assert!(ApiKeyProviderService::prefers_openai_responses_endpoint(
            "https://gateway.example.com/proxy/responses"
        ));
        assert!(!ApiKeyProviderService::prefers_openai_responses_endpoint(
            "https://api.openai.com/v1"
        ));
    }

    #[test]
    fn test_openai_responses_image_generation_request_uses_image_generation_tool() {
        let payload = ApiKeyProviderService::build_openai_responses_image_generation_request(
            "gpt-images-2",
            "生成测试图片",
            false,
        );

        assert_eq!(
            payload.get("model").and_then(serde_json::Value::as_str),
            Some("gpt-5.5")
        );
        assert_eq!(
            payload.get("stream").and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            payload
                .pointer("/tools/0/type")
                .and_then(serde_json::Value::as_str),
            Some("image_generation")
        );
        assert_eq!(
            payload
                .pointer("/tools/0/model")
                .and_then(serde_json::Value::as_str),
            Some("gpt-image-2")
        );
        assert_eq!(
            payload.get("input").and_then(serde_json::Value::as_str),
            Some("生成测试图片")
        );

        let retry_payload = ApiKeyProviderService::build_openai_responses_image_generation_request(
            "gpt-image-2",
            "生成测试图片",
            true,
        );
        assert_eq!(
            retry_payload
                .pointer("/input/0/content/0/type")
                .and_then(serde_json::Value::as_str),
            Some("input_text")
        );
    }

    #[test]
    fn test_openai_responses_image_body_has_result_reads_output_item_done() {
        let sse = concat!(
            "event: response.output_item.done\n",
            "data: {\"item\":{\"type\":\"image_generation_call\",\"result\":\"dGVzdA==\"}}\n\n",
            "event: response.completed\n",
            "data: {\"response\":{\"id\":\"resp_1\"}}\n\n"
        );

        assert!(ApiKeyProviderService::openai_responses_image_body_has_result(sse));
    }

    #[test]
    fn test_openai_responses_image_retry_detection_matches_gateway_error() {
        assert!(
            ApiKeyProviderService::should_retry_openai_responses_image_generation_with_input_list(
                reqwest::StatusCode::BAD_REQUEST,
                r#"{"error":{"message":"Input must be a list"}}"#,
            )
        );
        assert!(
            !ApiKeyProviderService::should_retry_openai_responses_image_generation_with_input_list(
                reqwest::StatusCode::UNAUTHORIZED,
                r#"{"error":{"message":"Input must be a list"}}"#,
            )
        );
    }

    #[test]
    fn test_system_provider_type_can_be_updated() {
        let db = init_test_database();
        let service = ApiKeyProviderService::new();

        service
            .initialize_system_providers(&db)
            .expect("初始化系统 Provider 失败");

        let updated = service
            .update_provider(
                &db,
                "openai",
                None,
                Some(ApiProviderType::Openai),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("更新系统 Provider 类型失败");

        assert_eq!(updated.provider_type, ApiProviderType::Openai);

        let persisted = service
            .get_provider(&db, "openai")
            .expect("读取系统 Provider 失败")
            .expect("系统 Provider 应存在");

        assert_eq!(persisted.provider.provider_type, ApiProviderType::Openai);
    }

    #[test]
    fn test_initialize_system_providers_updates_legacy_sensenova_default_host() {
        let db = init_test_database();
        let service = ApiKeyProviderService::new();

        service
            .initialize_system_providers(&db)
            .expect("初始化系统 Provider 失败");

        {
            let conn = db.lock().expect("获取数据库锁失败");
            let mut provider = ApiKeyProviderDao::get_provider_by_id(&conn, "sensenova")
                .expect("读取 SenseNova Provider 失败")
                .expect("SenseNova Provider 应存在");
            provider.api_host = SENSENOVA_OLD_OPENAI_COMPATIBLE_API_HOST.to_string();
            ApiKeyProviderDao::update_provider(&conn, &provider)
                .expect("写入旧 SenseNova 地址失败");
        }

        service
            .initialize_system_providers(&db)
            .expect("重新初始化系统 Provider 失败");

        let persisted = service
            .get_provider(&db, "sensenova")
            .expect("读取 SenseNova Provider 失败")
            .expect("SenseNova Provider 应存在");

        assert_eq!(
            persisted.provider.api_host,
            SENSENOVA_OPENAI_COMPATIBLE_API_HOST
        );
    }

    #[test]
    fn test_initialize_system_providers_keeps_user_custom_sensenova_host() {
        let db = init_test_database();
        let service = ApiKeyProviderService::new();

        service
            .initialize_system_providers(&db)
            .expect("初始化系统 Provider 失败");

        {
            let conn = db.lock().expect("获取数据库锁失败");
            let mut provider = ApiKeyProviderDao::get_provider_by_id(&conn, "sensenova")
                .expect("读取 SenseNova Provider 失败")
                .expect("SenseNova Provider 应存在");
            provider.api_host = "https://proxy.example.com/sensenova".to_string();
            ApiKeyProviderDao::update_provider(&conn, &provider)
                .expect("写入自定义 SenseNova 地址失败");
        }

        service
            .initialize_system_providers(&db)
            .expect("重新初始化系统 Provider 失败");

        let persisted = service
            .get_provider(&db, "sensenova")
            .expect("读取 SenseNova Provider 失败")
            .expect("SenseNova Provider 应存在");

        assert_eq!(
            persisted.provider.api_host,
            "https://proxy.example.com/sensenova"
        );
    }

    #[test]
    fn test_add_custom_provider_should_force_known_anthropic_compatible_host_to_automatic() {
        let db = init_test_database();
        let service = ApiKeyProviderService::new();

        let provider = service
            .add_custom_provider(
                &db,
                "MiMo Anthropic".to_string(),
                ApiProviderType::Openai,
                "https://token-plan-cn.xiaomimimo.com/anthropic".to_string(),
                None,
                None,
                None,
                None,
                Some(ApiProviderPromptCacheMode::ExplicitOnly),
            )
            .expect("创建自定义 Provider 失败");

        assert_eq!(provider.provider_type, ApiProviderType::AnthropicCompatible);
        assert_eq!(
            provider.prompt_cache_mode,
            Some(ApiProviderPromptCacheMode::Automatic)
        );

        let conn = db.lock().expect("获取数据库锁失败");
        let persisted = ApiKeyProviderDao::get_provider_by_id(&conn, &provider.id)
            .expect("读取 Provider 失败")
            .expect("Provider 应存在");
        assert_eq!(
            persisted.provider_type,
            ApiProviderType::AnthropicCompatible
        );
        assert_eq!(
            persisted.prompt_cache_mode,
            Some(ApiProviderPromptCacheMode::Automatic)
        );
    }

    #[test]
    fn test_update_provider_should_force_known_anthropic_compatible_host_to_automatic() {
        let db = init_test_database();
        let service = ApiKeyProviderService::new();

        let provider = service
            .add_custom_provider(
                &db,
                "Unknown Anthropic".to_string(),
                ApiProviderType::AnthropicCompatible,
                "https://example.com/anthropic".to_string(),
                None,
                None,
                None,
                None,
                Some(ApiProviderPromptCacheMode::ExplicitOnly),
            )
            .expect("创建初始 Provider 失败");

        assert_eq!(
            provider.prompt_cache_mode,
            Some(ApiProviderPromptCacheMode::ExplicitOnly)
        );

        let updated = service
            .update_provider(
                &db,
                &provider.id,
                None,
                Some(ApiProviderType::Openai),
                Some("https://api.minimaxi.com/anthropic".to_string()),
                None,
                None,
                None,
                None,
                None,
                None,
                Some(ApiProviderPromptCacheMode::ExplicitOnly),
                None,
            )
            .expect("更新 Provider 失败");

        assert_eq!(updated.provider_type, ApiProviderType::AnthropicCompatible);
        assert_eq!(
            updated.prompt_cache_mode,
            Some(ApiProviderPromptCacheMode::Automatic)
        );
    }

    #[test]
    fn test_parse_openai_responses_content_prefers_output_text() {
        let body = serde_json::json!({
            "id": "resp_test",
            "output_text": "hello from responses",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "fallback text"}]
            }]
        })
        .to_string();

        let content = ApiKeyProviderService::parse_openai_responses_content(&body)
            .expect("应解析 output_text");
        assert_eq!(content, "hello from responses");
    }

    #[test]
    fn test_parse_openai_responses_content_reads_output_blocks() {
        let body = serde_json::json!({
            "id": "resp_test",
            "output": [{
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello"},
                    {"type": "output_text", "text": " world"}
                ]
            }]
        })
        .to_string();

        let content = ApiKeyProviderService::parse_openai_responses_content(&body)
            .expect("应解析 output.content");
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_legacy_proxycast_api_key_can_be_read_and_reencrypted() {
        let db = init_test_database();
        let service = ApiKeyProviderService::new();

        service
            .initialize_system_providers(&db)
            .expect("初始化系统 Provider 失败");

        let legacy_ciphertext = encrypt_with_legacy_proxycast_format("sk-legacy-compatible");
        {
            let conn = db.lock().expect("获取数据库锁失败");
            let key = ApiKeyEntry {
                id: "legacy-key".to_string(),
                provider_id: "openai".to_string(),
                api_key_encrypted: legacy_ciphertext.clone(),
                alias: Some("legacy".to_string()),
                enabled: true,
                usage_count: 0,
                error_count: 0,
                last_used_at: None,
                created_at: Utc::now(),
            };
            ApiKeyProviderDao::insert_api_key(&conn, &key).expect("插入 legacy key 失败");
        }

        let api_key = service
            .get_next_api_key(&db, "openai")
            .expect("读取 API Key 失败")
            .expect("应返回 API Key");
        assert_eq!(api_key, "sk-legacy-compatible");

        let conn = db.lock().expect("获取数据库锁失败");
        let persisted = ApiKeyProviderDao::get_api_key_by_id(&conn, "legacy-key")
            .expect("读取 API Key 失败")
            .expect("legacy key 应存在");
        assert_ne!(persisted.api_key_encrypted, legacy_ciphertext);
    }

    #[test]
    fn test_add_api_key_replace_existing_keeps_only_new_key() {
        let db = init_test_database();
        let service = ApiKeyProviderService::new();

        let provider = service
            .add_custom_provider(
                &db,
                "Replace Key Provider".to_string(),
                ApiProviderType::Openai,
                "https://api.example.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("创建 Provider 失败");

        let old_key = service
            .add_api_key(&db, &provider.id, "sk-old-key", None, false)
            .expect("添加旧 Key 失败");
        let new_key = service
            .add_api_key(&db, &provider.id, "sk-new-key", None, true)
            .expect("替换新 Key 失败");

        assert_ne!(old_key.id, new_key.id);

        let persisted = service
            .get_provider(&db, &provider.id)
            .expect("读取 Provider 失败")
            .expect("Provider 应存在");
        assert_eq!(persisted.api_keys.len(), 1);
        assert_eq!(persisted.api_keys[0].id, new_key.id);
        assert!(persisted.api_keys[0].enabled);

        let active_key = service
            .get_next_api_key(&db, &provider.id)
            .expect("读取当前 Key 失败")
            .expect("应返回当前 Key");
        assert_eq!(active_key, "sk-new-key");
    }

    #[test]
    fn test_add_api_key_replace_existing_reuses_matching_history_key() {
        let db = init_test_database();
        let service = ApiKeyProviderService::new();

        let provider = service
            .add_custom_provider(
                &db,
                "Reuse Key Provider".to_string(),
                ApiProviderType::Openai,
                "https://api.example.com/v1".to_string(),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("创建 Provider 失败");

        let first_key = service
            .add_api_key(&db, &provider.id, "sk-first-key", None, false)
            .expect("添加第一个 Key 失败");
        service
            .add_api_key(&db, &provider.id, "sk-second-key", None, false)
            .expect("添加第二个 Key 失败");

        let reused_key = service
            .add_api_key(&db, &provider.id, "sk-first-key", None, true)
            .expect("复用历史 Key 失败");
        assert_eq!(reused_key.id, first_key.id);

        let persisted = service
            .get_provider(&db, &provider.id)
            .expect("读取 Provider 失败")
            .expect("Provider 应存在");
        assert_eq!(persisted.api_keys.len(), 1);
        assert_eq!(persisted.api_keys[0].id, first_key.id);
        assert!(persisted.api_keys[0].enabled);

        let active_key = service
            .get_next_api_key(&db, &provider.id)
            .expect("读取当前 Key 失败")
            .expect("应返回当前 Key");
        assert_eq!(active_key, "sk-first-key");
    }

    async fn spawn_single_response_server(
        response_body: serde_json::Value,
    ) -> (String, tokio::task::JoinHandle<(String, String)>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("绑定测试服务失败");
        let addr = listener.local_addr().expect("读取测试地址失败");

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("接受连接失败");
            let mut buffer = Vec::new();
            let mut header_end = None;

            loop {
                let mut chunk = [0u8; 1024];
                let read = stream.read(&mut chunk).await.expect("读取请求失败");
                if read == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(pos) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                    header_end = Some(pos + 4);
                    break;
                }
            }

            let header_end = header_end.expect("请求头未结束");
            let header_text = String::from_utf8_lossy(&buffer[..header_end]).to_string();
            let content_length = header_text
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        value.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);

            while buffer.len() < header_end + content_length {
                let mut chunk = vec![0u8; content_length.max(1024)];
                let read = stream.read(&mut chunk).await.expect("补全请求体失败");
                if read == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..read]);
            }

            let request_line = header_text.lines().next().expect("缺少请求行").to_string();
            let request_path = request_line
                .split_whitespace()
                .nth(1)
                .expect("缺少请求路径")
                .to_string();
            let request_body =
                String::from_utf8_lossy(&buffer[header_end..header_end + content_length])
                    .to_string();
            let response_text = response_body.to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_text.len(),
                response_text
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("写回响应失败");

            (request_path, request_body)
        });

        (format!("http://{}", addr), server)
    }

    #[tokio::test]
    #[ignore = "当前沙箱禁止本地 TCP 监听，需在本机放行后运行"]
    async fn test_openai_response_chat_uses_responses_endpoint_and_fallback_model() {
        let db = init_database().expect("初始化数据库失败");
        let service = ApiKeyProviderService::new();
        let response_body = serde_json::json!({
            "id": "resp_test",
            "output_text": "OK"
        });
        let (base_url, server) = spawn_single_response_server(response_body).await;

        let provider = service
            .add_custom_provider(
                &db,
                "OpenAI Responses Test".to_string(),
                ApiProviderType::OpenaiResponse,
                base_url,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("创建 Provider 失败");
        service
            .add_api_key(
                &db,
                &provider.id,
                "sk-test-key",
                Some("test".to_string()),
                false,
            )
            .expect("添加 API Key 失败");

        let result = service
            .test_chat_with_fallback_models(
                &db,
                &provider.id,
                None,
                "hello".to_string(),
                vec!["gpt-4.1-mini".to_string()],
            )
            .await
            .expect("对话测试调用失败");

        assert!(result.success, "结果应成功: {:?}", result.error);
        assert_eq!(result.content.as_deref(), Some("OK"));

        let (request_path, request_body) = server.await.expect("等待测试服务失败");
        assert_eq!(request_path, "/v1/responses");
        let request_json: serde_json::Value =
            serde_json::from_str(&request_body).expect("请求体应为 JSON");
        assert_eq!(request_json["model"].as_str(), Some("gpt-4.1-mini"));
        assert_eq!(request_json["stream"].as_bool(), Some(false));
        assert_eq!(
            request_json["input"][0]["content"][0]["text"].as_str(),
            Some("hello")
        );
    }

    #[tokio::test]
    #[ignore = "当前沙箱禁止本地 TCP 监听，需在本机放行后运行"]
    async fn test_openai_response_connection_prefers_responses_endpoint() {
        let db = init_database().expect("初始化数据库失败");
        let service = ApiKeyProviderService::new();
        let response_body = serde_json::json!({
            "id": "resp_test",
            "output_text": "OK"
        });
        let (base_url, server) = spawn_single_response_server(response_body).await;

        let provider = service
            .add_custom_provider(
                &db,
                "OpenAI Responses Test".to_string(),
                ApiProviderType::OpenaiResponse,
                base_url,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("创建 Provider 失败");
        service
            .add_api_key(
                &db,
                &provider.id,
                "sk-test-key",
                Some("test".to_string()),
                false,
            )
            .expect("添加 API Key 失败");

        let result = service
            .test_connection_with_fallback_models(
                &db,
                &provider.id,
                None,
                vec!["gpt-4.1-mini".to_string()],
            )
            .await
            .expect("连接测试调用失败");

        assert!(result.success, "连接测试应成功: {:?}", result.error);
        assert_eq!(
            result.models.as_deref(),
            Some(&["gpt-4.1-mini".to_string()][..])
        );

        let (request_path, request_body) = server.await.expect("等待测试服务失败");
        assert_eq!(request_path, "/v1/responses");
        let request_json: serde_json::Value =
            serde_json::from_str(&request_body).expect("请求体应为 JSON");
        assert_eq!(request_json["model"].as_str(), Some("gpt-4.1-mini"));
        assert_eq!(
            request_json["input"][0]["content"][0]["text"].as_str(),
            Some("hi")
        );
    }

    #[tokio::test]
    #[ignore = "真实联网测试：设置 LIME_REAL_API_TEST=1 后执行"]
    async fn test_real_codex_provider_chat_gpt_5_3_codex() {
        if lime_core::env_compat::var(&["LIME_REAL_API_TEST", "PROXYCAST_REAL_API_TEST"]).as_deref()
            != Some("1")
        {
            return;
        }

        let db = init_database().expect("初始化数据库失败");
        let service = ApiKeyProviderService::new();
        let provider_id = resolve_real_codex_provider_id(&db).expect("解析本地 CLI Provider 失败");
        let model = lime_core::env_compat::var(&["LIME_REAL_MODEL", "PROXYCAST_REAL_MODEL"])
            .unwrap_or_else(|| "gpt-5.3-codex".to_string());
        let prompt = lime_core::env_compat::var(&["LIME_REAL_PROMPT", "PROXYCAST_REAL_PROMPT"])
            .unwrap_or_else(|| "请仅回复 REAL_OK".to_string());

        let result = service
            .test_chat(&db, &provider_id, Some(model.clone()), prompt)
            .await
            .expect("真实调用失败");

        assert!(
            result.success,
            "真实调用未成功: provider_id={provider_id}, model={model}, error={:?}, raw={:?}",
            result.error, result.raw
        );
        assert!(
            result.error.is_none(),
            "真实调用返回错误: {:?}",
            result.error
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTestResult {
    pub success: bool,
    pub latency_ms: Option<u64>,
    pub error: Option<String>,
    pub content: Option<String>,
    pub raw: Option<String>,
}

// ============================================================================
// 加密服务
// ============================================================================

/// 简单的 API Key 加密服务
/// 使用 XOR 加密 + Base64 编码
/// 注意：这是一个简单的混淆方案，不是强加密
struct EncryptionService {
    /// 当前加密密钥（Lime）
    current_key: Vec<u8>,
    /// 兼容旧版 ProxyCast 的历史加密密钥
    legacy_keys: Vec<Vec<u8>>,
}

struct DecryptionResult {
    plaintext: String,
    used_legacy_key: bool,
}

impl EncryptionService {
    const CURRENT_SALT: &'static [u8] = b"lime-api-key-encryption-salt";
    const LEGACY_SALT: &'static [u8] = b"proxycast-api-key-encryption-salt";
    const CURRENT_DEFAULT_MACHINE_ID: &'static str = "lime-default-machine-id";
    const LEGACY_DEFAULT_MACHINE_ID: &'static str = "proxycast-default-machine-id";

    /// 创建新的加密服务
    fn new() -> Self {
        // 使用机器特定信息生成密钥
        let machine_id = Self::get_machine_id();
        let current_key = Self::derive_key(&machine_id, Self::CURRENT_SALT);
        let mut legacy_keys = vec![Self::derive_key(&machine_id, Self::LEGACY_SALT)];

        if machine_id == Self::CURRENT_DEFAULT_MACHINE_ID {
            legacy_keys.push(Self::derive_key(
                Self::LEGACY_DEFAULT_MACHINE_ID,
                Self::LEGACY_SALT,
            ));
        }

        Self {
            current_key,
            legacy_keys,
        }
    }

    /// 获取机器 ID
    fn get_machine_id() -> String {
        // 尝试获取机器 ID，失败则使用默认值
        if let Ok(id) = std::fs::read_to_string("/etc/machine-id") {
            return id.trim().to_string();
        }
        if let Ok(id) = std::fs::read_to_string("/var/lib/dbus/machine-id") {
            return id.trim().to_string();
        }
        // macOS: 使用 IOPlatformUUID
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("ioreg")
                .args(["-rd1", "-c", "IOPlatformExpertDevice"])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if line.contains("IOPlatformUUID") {
                        if let Some(uuid) = line.split('"').nth(3) {
                            return uuid.to_string();
                        }
                    }
                }
            }
        }
        // 默认值
        Self::CURRENT_DEFAULT_MACHINE_ID.to_string()
    }

    fn derive_key(machine_id: &str, salt: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(machine_id.as_bytes());
        hasher.update(salt);
        hasher.finalize().to_vec()
    }

    fn xor_bytes(input: &[u8], key: &[u8]) -> Vec<u8> {
        input
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ key[i % key.len()])
            .collect()
    }

    fn looks_like_api_key_candidate(value: &str) -> bool {
        let trimmed = value.trim();
        trimmed.len() >= 8
            && trimmed.is_ascii()
            && !trimmed
                .chars()
                .any(|ch| ch.is_ascii_control() || ch.is_whitespace())
    }

    /// 加密 API Key
    fn encrypt(&self, plaintext: &str) -> String {
        let encrypted = Self::xor_bytes(plaintext.as_bytes(), &self.current_key);
        BASE64.encode(encrypted)
    }

    fn decrypt_with_compatibility(&self, ciphertext: &str) -> Result<DecryptionResult, String> {
        let encrypted = BASE64
            .decode(ciphertext)
            .map_err(|e| format!("Base64 解码失败: {e}"))?;

        let mut last_error: Option<String> = None;

        for (key, used_legacy_key) in std::iter::once((&self.current_key, false))
            .chain(self.legacy_keys.iter().map(|key| (key, true)))
        {
            let decrypted = Self::xor_bytes(&encrypted, key);
            match String::from_utf8(decrypted) {
                Ok(plaintext) if Self::looks_like_api_key_candidate(&plaintext) => {
                    return Ok(DecryptionResult {
                        plaintext,
                        used_legacy_key,
                    });
                }
                Ok(_) => {
                    last_error = Some("解密结果不符合 API Key 格式".to_string());
                }
                Err(_error) => {
                    last_error = Some("解密结果不是有效 UTF-8".to_string());
                }
            }
        }

        Err(last_error.unwrap_or_else(|| "API Key 解密失败".to_string()))
    }

    /// 解密 API Key
    fn decrypt(&self, ciphertext: &str) -> Result<String, String> {
        self.decrypt_with_compatibility(ciphertext)
            .map(|result| result.plaintext)
    }

    /// 检查是否为加密后的值（非明文）
    fn is_encrypted(&self, value: &str) -> bool {
        // 加密后的值是 Base64 编码的，通常不包含常见的 API Key 前缀
        !value.starts_with("sk-")
            && !value.starts_with("pk-")
            && !value.starts_with("api-")
            && BASE64.decode(value).is_ok()
    }
}

// ============================================================================
// API Key Provider 服务
// ============================================================================

/// API Key Provider 管理服务
pub struct ApiKeyProviderService {
    /// 加密服务
    encryption: EncryptionService,
    /// 轮询索引（按 provider_id 分组）
    round_robin_index: RwLock<HashMap<String, AtomicUsize>>,
}

impl Default for ApiKeyProviderService {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiKeyProviderService {
    /// 创建新的服务实例
    pub fn new() -> Self {
        Self {
            encryption: EncryptionService::new(),
            round_robin_index: RwLock::new(HashMap::new()),
        }
    }

    fn normalize_custom_prompt_cache_mode(
        provider_type: ApiProviderType,
        api_host: &str,
        prompt_cache_mode: Option<ApiProviderPromptCacheMode>,
    ) -> Option<ApiProviderPromptCacheMode> {
        match provider_type {
            ApiProviderType::AnthropicCompatible => {
                if is_known_automatic_anthropic_compatible_host(Some(api_host)) {
                    Some(ApiProviderPromptCacheMode::Automatic)
                } else {
                    Some(prompt_cache_mode.unwrap_or(ApiProviderPromptCacheMode::ExplicitOnly))
                }
            }
            _ => None,
        }
    }

    fn normalize_custom_provider_type(
        provider_type: ApiProviderType,
        api_host: &str,
    ) -> ApiProviderType {
        infer_managed_provider_type(provider_type, api_host)
    }

    fn to_credential_prompt_cache_mode(
        mode: ApiProviderPromptCacheMode,
    ) -> ProviderPromptCacheMode {
        match mode {
            ApiProviderPromptCacheMode::Automatic => ProviderPromptCacheMode::Automatic,
            ApiProviderPromptCacheMode::ExplicitOnly => ProviderPromptCacheMode::ExplicitOnly,
        }
    }

    fn decrypt_api_key_entry_with_migration(
        &self,
        conn: &rusqlite::Connection,
        key: &ApiKeyEntry,
    ) -> Result<String, String> {
        let result = self
            .encryption
            .decrypt_with_compatibility(&key.api_key_encrypted)?;

        if result.used_legacy_key {
            let reencrypted = self.encryption.encrypt(&result.plaintext);
            if reencrypted != key.api_key_encrypted {
                ApiKeyProviderDao::update_api_key_encrypted(conn, &key.id, &reencrypted)
                    .map_err(|e| format!("升级旧版 API Key 加密格式失败: {e}"))?;
                tracing::info!(
                    "[ApiKeyProviderService] 已自动升级旧版 API Key 加密格式: {}",
                    key.id
                );
            }
        }

        Ok(result.plaintext)
    }

    fn next_round_robin_candidate_indices(
        &self,
        scope: &str,
        len: usize,
    ) -> Result<Vec<usize>, String> {
        if len == 0 {
            return Ok(Vec::new());
        }

        let start = {
            let mut indices = self.round_robin_index.write().map_err(|e| e.to_string())?;
            indices
                .entry(scope.to_string())
                .or_insert_with(|| AtomicUsize::new(0))
                .fetch_add(1, Ordering::SeqCst)
        };

        Ok((0..len).map(|offset| (start + offset) % len).collect())
    }

    fn log_skipped_invalid_api_key(&self, key: &ApiKeyEntry, error: &str) {
        tracing::debug!(
            "[ApiKeyProviderService] 跳过不可解密 API Key {} (provider={}): {}",
            key.id,
            key.provider_id,
            error
        );
    }

    fn select_next_decryptable_api_key<'a>(
        &self,
        conn: &rusqlite::Connection,
        scope: &str,
        keys: &'a [ApiKeyEntry],
    ) -> Result<Option<(&'a ApiKeyEntry, String)>, String> {
        for index in self.next_round_robin_candidate_indices(scope, keys.len())? {
            let candidate = &keys[index];
            match self.decrypt_api_key_entry_with_migration(conn, candidate) {
                Ok(api_key) => return Ok(Some((candidate, api_key))),
                Err(error) => self.log_skipped_invalid_api_key(candidate, &error),
            }
        }

        Ok(None)
    }

    fn select_next_decryptable_api_key_with_provider<'a>(
        &self,
        conn: &rusqlite::Connection,
        scope: &str,
        keys: &'a [(ApiKeyEntry, ApiKeyProvider)],
    ) -> Result<Option<(&'a ApiKeyEntry, &'a ApiKeyProvider, String)>, String> {
        for index in self.next_round_robin_candidate_indices(scope, keys.len())? {
            let (candidate, provider) = &keys[index];
            match self.decrypt_api_key_entry_with_migration(conn, candidate) {
                Ok(api_key) => return Ok(Some((candidate, provider, api_key))),
                Err(error) => self.log_skipped_invalid_api_key(candidate, &error),
            }
        }

        Ok(None)
    }

    pub fn migrate_legacy_api_key_encryption(&self, db: &DbConnection) -> Result<usize, String> {
        let conn = lime_core::database::lock_db(db)?;
        let providers = ApiKeyProviderDao::get_all_providers_with_keys(&conn)
            .map_err(|e| format!("读取 API Key 列表失败: {e}"))?;

        let mut migrated = 0usize;
        for provider in providers {
            for key in provider.api_keys {
                match self
                    .encryption
                    .decrypt_with_compatibility(&key.api_key_encrypted)
                {
                    Ok(result) if result.used_legacy_key => {
                        let reencrypted = self.encryption.encrypt(&result.plaintext);
                        if reencrypted != key.api_key_encrypted {
                            ApiKeyProviderDao::update_api_key_encrypted(
                                &conn,
                                &key.id,
                                &reencrypted,
                            )
                            .map_err(|e| format!("升级 API Key 加密格式失败: {e}"))?;
                            migrated += 1;
                        }
                    }
                    Ok(_) => {}
                    Err(error) => {
                        tracing::info!(
                            "[ApiKeyProviderService] 跳过异常 API Key {} (provider={}): {}",
                            key.id,
                            provider.provider.id,
                            error
                        );
                    }
                }
            }
        }

        Ok(migrated)
    }

    pub async fn test_chat(
        &self,
        db: &DbConnection,
        provider_id: &str,
        model_name: Option<String>,
        prompt: String,
    ) -> Result<ChatTestResult, String> {
        self.test_chat_with_fallback_models(db, provider_id, model_name, prompt, Vec::new())
            .await
    }

    pub async fn test_chat_with_fallback_models(
        &self,
        db: &DbConnection,
        provider_id: &str,
        model_name: Option<String>,
        prompt: String,
        fallback_models: Vec<String>,
    ) -> Result<ChatTestResult, String> {
        use std::time::Instant;

        let provider_with_keys = self
            .get_provider(db, provider_id)?
            .ok_or_else(|| format!("Provider 不存在: {provider_id}"))?;

        let provider = &provider_with_keys.provider;
        let effective_provider_type = provider.effective_provider_type();

        let api_key = self
            .get_next_api_key(db, provider_id)?
            .ok_or_else(|| "没有可用的 API Key".to_string())?;

        let test_models =
            Self::pick_test_model_candidates(model_name, &provider.custom_models, &fallback_models);
        if test_models.is_empty() {
            return Err("缺少模型名称：请在自定义模型中填写一个模型名".to_string());
        }

        let start = Instant::now();
        let mut last_error = None;

        for (index, test_model) in test_models.iter().enumerate() {
            // 根据 Provider 协议类型选择测试方式
            let result = match effective_provider_type {
                // Codex 协议直接走 /responses 端点
                ApiProviderType::Codex => {
                    self.test_codex_responses_endpoint(
                        &api_key,
                        &provider.api_host,
                        test_model,
                        &prompt,
                        effective_provider_type,
                    )
                    .await
                }
                // OpenAI Responses API 走 /v1/responses
                provider_type if Self::uses_openai_responses_protocol(provider_type) => {
                    self.test_openai_responses_once(
                        &api_key,
                        &provider.api_host,
                        test_model,
                        &prompt,
                    )
                    .await
                }
                // Anthropic / AnthropicCompatible 统一走 /v1/messages
                provider_type if Self::uses_anthropic_protocol(provider_type) => {
                    self.test_anthropic_chat_once(
                        &api_key,
                        &provider.api_host,
                        test_model,
                        &prompt,
                        provider.supports_automatic_prompt_cache(),
                        effective_provider_type,
                    )
                    .await
                }
                // 其余默认 OpenAI 兼容
                _ => {
                    self.test_openai_chat_once(
                        &api_key,
                        &provider.api_host,
                        test_model,
                        &prompt,
                        effective_provider_type,
                    )
                    .await
                }
            };

            match result {
                Ok((content, raw)) => {
                    let latency_ms = start.elapsed().as_millis() as u64;
                    return Ok(ChatTestResult {
                        success: true,
                        latency_ms: Some(latency_ms),
                        error: None,
                        content: Some(content),
                        raw: Some(raw),
                    });
                }
                Err(error) => {
                    let should_try_next = index + 1 < test_models.len()
                        && Self::should_try_next_test_model_after_error(&error);
                    last_error = Some(error);
                    if !should_try_next {
                        break;
                    }
                }
            }
        }

        let latency_ms = start.elapsed().as_millis() as u64;
        Ok(ChatTestResult {
            success: false,
            latency_ms: Some(latency_ms),
            error: Some(last_error.unwrap_or_else(|| "所有候选模型均未完成连接测试".to_string())),
            content: None,
            raw: None,
        })
    }

    #[inline]
    fn uses_anthropic_protocol(provider_type: ApiProviderType) -> bool {
        provider_type.is_anthropic_protocol()
    }

    #[inline]
    fn uses_openai_responses_protocol(provider_type: ApiProviderType) -> bool {
        matches!(provider_type, ApiProviderType::OpenaiResponse)
    }

    #[inline]
    fn prefers_openai_responses_endpoint(api_host: &str) -> bool {
        is_openai_responses_compatible_host(api_host)
    }

    fn pick_preferred_fallback_model(fallback_models: &[String]) -> Option<String> {
        const PREFERRED_MARKERS: &[&str] =
            &["nano", "mini", "flash-lite", "flash", "haiku", "lite"];

        for marker in PREFERRED_MARKERS {
            if let Some(model) = fallback_models
                .iter()
                .find(|model| model.to_ascii_lowercase().contains(marker))
            {
                return Some(model.clone());
            }
        }

        fallback_models.first().cloned()
    }

    fn pick_test_model(
        model_name: Option<String>,
        custom_models: &[String],
        fallback_models: &[String],
    ) -> Option<String> {
        let explicit_model = model_name.and_then(|model| {
            let normalized = model.trim();
            (!normalized.is_empty()).then(|| normalized.to_string())
        });
        if explicit_model.is_some() {
            return explicit_model;
        }

        let fallback_model_set: HashSet<String> = fallback_models
            .iter()
            .map(|model| model.trim().to_lowercase())
            .filter(|model| !model.is_empty())
            .collect();

        let matching_custom_model = custom_models.iter().find_map(|model| {
            let normalized = model.trim();
            (!normalized.is_empty() && fallback_model_set.contains(&normalized.to_lowercase()))
                .then(|| normalized.to_string())
        });

        matching_custom_model
            .or_else(|| Self::pick_preferred_fallback_model(fallback_models))
            .or_else(|| {
                custom_models.iter().find_map(|model| {
                    let normalized = model.trim();
                    (!normalized.is_empty()).then(|| normalized.to_string())
                })
            })
    }

    fn push_unique_test_model(
        candidates: &mut Vec<String>,
        seen: &mut HashSet<String>,
        model: &str,
    ) {
        let normalized = model.trim();
        if normalized.is_empty() {
            return;
        }

        if seen.insert(normalized.to_ascii_lowercase()) {
            candidates.push(normalized.to_string());
        }
    }

    fn pick_test_model_candidates(
        model_name: Option<String>,
        custom_models: &[String],
        fallback_models: &[String],
    ) -> Vec<String> {
        let mut candidates = Vec::new();
        let mut seen = HashSet::new();

        if let Some(explicit_model) = model_name.as_deref() {
            Self::push_unique_test_model(&mut candidates, &mut seen, explicit_model);
        }

        if let Some(primary_model) = Self::pick_test_model(None, custom_models, fallback_models) {
            Self::push_unique_test_model(&mut candidates, &mut seen, &primary_model);
        }

        for model in custom_models {
            Self::push_unique_test_model(&mut candidates, &mut seen, model);
        }

        if let Some(preferred_fallback) = Self::pick_preferred_fallback_model(fallback_models) {
            Self::push_unique_test_model(&mut candidates, &mut seen, &preferred_fallback);
        }

        for model in fallback_models {
            Self::push_unique_test_model(&mut candidates, &mut seen, model);
        }

        candidates
    }

    fn should_try_next_test_model_after_error(error: &str) -> bool {
        let normalized = error.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return false;
        }

        normalized.contains("param incorrect")
            || normalized.contains("parameter incorrect")
            || normalized.contains("unsupported model")
            || normalized.contains("unsupported_model")
            || normalized.contains("not supported model")
            || normalized.contains("model not supported")
            || normalized.contains("model is not supported")
            || normalized.contains("model not found")
            || normalized.contains("model_not_found")
            || normalized.contains("model does not exist")
            || normalized.contains("no such model")
            || normalized.contains("invalid model")
            || (normalized.contains("模型")
                && (normalized.contains("白名单")
                    || normalized.contains("不支持")
                    || normalized.contains("不存在")
                    || normalized.contains("无效")))
    }

    fn looks_like_openai_image_model(model: &str) -> bool {
        let normalized = model.trim().to_ascii_lowercase();
        !normalized.is_empty()
            && (normalized.contains("gpt-image")
                || normalized.contains("gpt-images")
                || normalized.contains("dall-e")
                || normalized.contains("dalle"))
    }

    fn should_use_openai_image_connection_test(
        model_name: Option<&str>,
        custom_models: &[String],
        fallback_models: &[String],
    ) -> bool {
        if let Some(explicit_model) = model_name.map(str::trim).filter(|model| !model.is_empty()) {
            return Self::looks_like_openai_image_model(explicit_model);
        }

        let normalized_custom_models: Vec<&str> = custom_models
            .iter()
            .map(|model| model.trim())
            .filter(|model| !model.is_empty())
            .collect();
        if !normalized_custom_models.is_empty() {
            return normalized_custom_models
                .iter()
                .all(|model| Self::looks_like_openai_image_model(model));
        }

        let normalized_fallback_models: Vec<&str> = fallback_models
            .iter()
            .map(|model| model.trim())
            .filter(|model| !model.is_empty())
            .collect();

        !normalized_fallback_models.is_empty()
            && normalized_fallback_models
                .iter()
                .all(|model| Self::looks_like_openai_image_model(model))
    }

    fn pick_openai_image_test_model(
        model_name: Option<String>,
        custom_models: &[String],
        fallback_models: &[String],
    ) -> Option<String> {
        let explicit_model = model_name.and_then(|model| {
            let normalized = model.trim();
            (Self::looks_like_openai_image_model(normalized)).then(|| normalized.to_string())
        });
        if explicit_model.is_some() {
            return explicit_model;
        }

        let fallback_model_set: HashSet<String> = fallback_models
            .iter()
            .map(|model| model.trim().to_lowercase())
            .filter(|model| !model.is_empty())
            .collect();

        let matching_custom_model = custom_models.iter().find_map(|model| {
            let normalized = model.trim();
            (!normalized.is_empty()
                && Self::looks_like_openai_image_model(normalized)
                && fallback_model_set.contains(&normalized.to_lowercase()))
            .then(|| normalized.to_string())
        });

        matching_custom_model
            .or_else(|| {
                fallback_models.iter().find_map(|model| {
                    let normalized = model.trim();
                    (!normalized.is_empty() && Self::looks_like_openai_image_model(normalized))
                        .then(|| normalized.to_string())
                })
            })
            .or_else(|| {
                custom_models.iter().find_map(|model| {
                    let normalized = model.trim();
                    (!normalized.is_empty() && Self::looks_like_openai_image_model(normalized))
                        .then(|| normalized.to_string())
                })
            })
    }

    fn is_nearai_provider(provider_id: &str, api_host: &str) -> bool {
        let provider = provider_id.trim().to_ascii_lowercase();
        let host = api_host.trim().to_ascii_lowercase();

        matches!(provider.as_str(), "nearai" | "near-ai" | "near_ai")
            || host.contains("cloud-api.near.ai")
    }

    fn build_nearai_models_api_url(api_host: &str) -> String {
        let normalized_host = normalize_openai_model_discovery_host(api_host);
        let host = normalized_host.trim_end_matches('/');

        if host.ends_with("/model/list") {
            return host.to_string();
        }

        if let Some((prefix, _)) = host.split_once("/v1") {
            return format!("{prefix}/v1/model/list");
        }

        format!("{host}/v1/model/list")
    }

    fn build_openai_images_url(api_host: &str) -> String {
        let normalized_host = normalize_openai_compatible_api_host(api_host);
        let trimmed = normalized_host.trim().trim_end_matches('/');
        let normalized = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            trimmed.to_string()
        } else if trimmed.is_empty() {
            "https://api.openai.com".to_string()
        } else {
            format!("https://{trimmed}")
        };

        let has_version = normalized
            .rsplit('/')
            .next()
            .map(|segment| {
                segment.starts_with('v')
                    && segment.len() >= 2
                    && segment[1..].chars().all(|char| char.is_ascii_digit())
            })
            .unwrap_or(false);

        if has_version {
            format!("{normalized}/images/generations")
        } else {
            format!("{normalized}/v1/images/generations")
        }
    }

    async fn test_openai_chat_once(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
        prompt: &str,
        provider_type: ApiProviderType,
    ) -> Result<(String, String), String> {
        use lime_core::models::openai::{ChatCompletionRequest, ChatMessage, MessageContent};
        use lime_providers::providers::openai_custom::OpenAICustomProvider;

        let provider =
            OpenAICustomProvider::with_config(api_key.to_string(), Some(api_host.to_string()));

        let request = ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text(prompt.to_string())),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
            temperature: Some(0.2),
            max_tokens: Some(64),
            top_p: None,
            stream: false,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
        };

        let resp = provider
            .call_api(&request)
            .await
            .map_err(|e| format!("API 调用失败: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();

        if status.is_success() {
            let parsed: serde_json::Value =
                serde_json::from_str(&body).map_err(|e| format!("解析响应失败: {e} - {body}"))?;

            let content = parsed["choices"]
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|c| c["message"]["content"].as_str())
                .unwrap_or("")
                .to_string();

            return Ok((content, body));
        }

        // 部分上游（如某些 relay）强制要求 stream=true
        if status.as_u16() == 400 && body.contains("Stream must be set to true") {
            let mut request2 = request.clone();
            request2.stream = true;

            let resp2 = provider
                .call_api(&request2)
                .await
                .map_err(|e| format!("API 调用失败: {e}"))?;

            let status2 = resp2.status();
            let body2 = resp2.text().await.unwrap_or_default();

            if !status2.is_success() {
                return Err(Self::format_http_api_error(status2, &body2));
            }

            let content = Self::parse_chat_completions_sse_content(&body2);
            return Ok((content, body2));
        }

        // 部分上游（如 Codex relay）不支持 messages 参数，需要走 /responses 端点
        if status.as_u16() == 400 && body.contains("Unsupported parameter: messages") {
            return self
                .test_codex_responses_endpoint(api_key, api_host, model, prompt, provider_type)
                .await;
        }

        Err(Self::format_http_api_error(status, &body))
    }

    async fn test_anthropic_chat_once(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
        prompt: &str,
        enable_automatic_prompt_cache: bool,
        provider_type: ApiProviderType,
    ) -> Result<(String, String), String> {
        use lime_providers::providers::claude_custom::{ClaudeCustomProvider, PromptCacheMode};

        let prompt_cache_mode = if enable_automatic_prompt_cache {
            PromptCacheMode::Automatic
        } else {
            PromptCacheMode::ExplicitOnly
        };
        let provider = ClaudeCustomProvider::with_provider_type_and_prompt_cache_mode(
            api_key.to_string(),
            Some(api_host.to_string()),
            provider_type,
            prompt_cache_mode,
        );

        let request = serde_json::json!({
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}]
        });

        let (status, body) = Self::execute_anthropic_test_request_with_retries(|| async {
            provider
                .messages(&request)
                .await
                .map_err(|e| format!("API 调用失败: {e}"))
        })
        .await?;

        if !status.is_success() {
            return Err(Self::format_anthropic_http_api_error(
                status,
                &body,
                api_host,
                provider_type,
                api_key,
            ));
        }

        let parsed: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| format!("解析响应失败: {e} - {body}"))?;

        let content = parsed["content"]
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| block["text"].as_str())
                    .collect::<String>()
            })
            .unwrap_or_default();

        Ok((content, body))
    }

    fn parse_chat_completions_sse_content(body: &str) -> String {
        let mut out = String::new();

        for line in body.lines() {
            let line = line.trim();
            if !line.starts_with("data:") {
                continue;
            }
            let data = line.trim_start_matches("data:").trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }

            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(s) = v["choices"][0]["delta"]["content"].as_str() {
                    out.push_str(s);
                } else if let Some(s) = v["choices"][0]["message"]["content"].as_str() {
                    out.push_str(s);
                }
            }
        }

        out
    }

    fn build_openai_responses_request(
        model: &str,
        prompt: &str,
        stream: bool,
    ) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "stream": stream,
            "max_output_tokens": 64
        })
    }

    fn parse_openai_responses_content(body: &str) -> Result<String, String> {
        let parsed: serde_json::Value =
            serde_json::from_str(body).map_err(|e| format!("解析响应失败: {e} - {body}"))?;

        if let Some(output_text) = parsed["output_text"].as_str() {
            return Ok(output_text.to_string());
        }

        let mut content = String::new();
        if let Some(output) = parsed["output"].as_array() {
            for output_item in output {
                if output_item["type"].as_str() != Some("message") {
                    continue;
                }

                if let Some(content_items) = output_item["content"].as_array() {
                    for item in content_items {
                        let item_type = item["type"].as_str().unwrap_or_default();
                        if matches!(item_type, "output_text" | "text") {
                            if let Some(text) = item["text"].as_str() {
                                content.push_str(text);
                            }
                        }
                    }
                }
            }
        }

        Ok(content)
    }

    fn extract_json_error_message(body: &str) -> Option<String> {
        let parsed: serde_json::Value = serde_json::from_str(body).ok()?;
        let error = parsed.get("error")?;

        if let Some(message) = error.get("message").and_then(|value| value.as_str()) {
            return Some(message.to_string());
        }

        error.as_str().map(|value| value.to_string())
    }

    fn format_http_api_error(status: reqwest::StatusCode, body: &str) -> String {
        match Self::extract_json_error_message(body) {
            Some(message) if body.contains("insufficient_quota") => {
                format!("API 返回错误: {status} - OpenAI 账户配额不足或未开通计费：{message}")
            }
            Some(message) => format!("API 返回错误: {status} - {message}"),
            None => format!("API 返回错误: {status} - {body}"),
        }
    }

    fn is_transient_anthropic_upstream_error(status: reqwest::StatusCode, body: &str) -> bool {
        if matches!(status.as_u16(), 429 | 503 | 529) {
            return true;
        }

        let message = Self::extract_json_error_message(body)
            .unwrap_or_else(|| body.to_string())
            .to_ascii_lowercase();

        message.contains("overloaded_error")
            || message.contains("rate_limit")
            || message.contains("rate limit")
    }

    async fn execute_anthropic_test_request_with_retries<F, Fut>(
        mut send_request: F,
    ) -> Result<(reqwest::StatusCode, String), String>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<reqwest::Response, String>>,
    {
        const RETRY_DELAYS_MS: [u64; 5] = [0, 400, 1200, 2500, 5000];

        for (attempt_index, delay_ms) in RETRY_DELAYS_MS.iter().enumerate() {
            if attempt_index > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(*delay_ms)).await;
            }

            let response = send_request().await?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();

            if !Self::is_transient_anthropic_upstream_error(status, &body)
                || attempt_index + 1 == RETRY_DELAYS_MS.len()
            {
                return Ok((status, body));
            }
        }

        unreachable!("Anthropic 重试循环必须在最后一次尝试返回结果");
    }

    fn format_anthropic_http_api_error(
        status: reqwest::StatusCode,
        body: &str,
        _api_host: &str,
        provider_type: ApiProviderType,
        _api_key: &str,
    ) -> String {
        let base = Self::format_http_api_error(status, body);
        let uses_anthropic_protocol = Self::uses_anthropic_protocol(provider_type);
        let looks_like_misleading_auth_error = body.contains(
            "Please carry the API secret key in the 'Authorization' field of the request header",
        );

        if uses_anthropic_protocol && Self::is_transient_anthropic_upstream_error(status, body) {
            return format!(
                "{base}。上游 Anthropic 兼容接口当前暂时过载或限流，请稍后重试；这通常不是 Base URL、鉴权头或模型配置错误。"
            );
        }

        if status == reqwest::StatusCode::UNAUTHORIZED
            && uses_anthropic_protocol
            && looks_like_misleading_auth_error
        {
            return format!(
                "{base}。Lime 已按 Anthropic 协议发送兼容鉴权；若仍返回此错误，请优先核对 API Key 是否完整、未被截断，并确认该 Key 已开通当前 Base URL / 模型对应的接口权限。"
            );
        }

        base
    }

    fn build_codex_responses_request(model: &str, prompt: &str) -> serde_json::Value {
        Self::build_openai_responses_request(model, prompt, true)
    }

    async fn test_openai_responses_once(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
        prompt: &str,
    ) -> Result<(String, String), String> {
        use lime_providers::providers::codex::CodexProvider;

        let url = CodexProvider::build_responses_url(api_host);
        let request_body = Self::build_openai_responses_request(model, prompt, false);

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("API 调用失败: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();

        if status.is_success() {
            let content = Self::parse_openai_responses_content(&body)?;
            return Ok((content, body));
        }

        if status.as_u16() == 400 && body.contains("Stream must be set to true") {
            let streaming_request = Self::build_openai_responses_request(model, prompt, true);
            let resp2 = client
                .post(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .json(&streaming_request)
                .send()
                .await
                .map_err(|e| format!("API 调用失败: {e}"))?;

            let status2 = resp2.status();
            let body2 = resp2.text().await.unwrap_or_default();
            if !status2.is_success() {
                return Err(Self::format_http_api_error(status2, &body2));
            }

            let content = Self::parse_codex_responses_sse_content(&body2);
            return Ok((content, body2));
        }

        Err(Self::format_http_api_error(status, &body))
    }

    /// 测试 Codex /responses 端点（用于不支持 messages 参数的上游）
    async fn test_codex_responses_endpoint(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
        prompt: &str,
        provider_type: ApiProviderType,
    ) -> Result<(String, String), String> {
        use lime_providers::providers::codex::CodexProvider;

        let url = CodexProvider::build_responses_url(api_host);

        // Codex Responses 格式请求体（input 必须是列表）
        let request_body = Self::build_codex_responses_request(model, prompt);

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("API 调用失败: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();

        if !status.is_success() {
            return Err(Self::format_anthropic_http_api_error(
                status,
                &body,
                api_host,
                provider_type,
                api_key,
            ));
        }

        // 解析 Codex SSE 响应
        let content = Self::parse_codex_responses_sse_content(&body);
        Ok((content, body))
    }

    fn parse_codex_responses_sse_content(body: &str) -> String {
        let mut out = String::new();

        for line in body.lines() {
            let line = line.trim();
            if !line.starts_with("data:") {
                continue;
            }
            let data = line.trim_start_matches("data:").trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }

            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                // Codex responses 格式: {"type": "response.output_text.delta", "delta": "..."}
                if let Some(s) = v["delta"].as_str() {
                    out.push_str(s);
                }
                // 或者完整响应格式
                if let Some(arr) = v["output"].as_array() {
                    for item in arr {
                        if item["type"].as_str() == Some("message") {
                            if let Some(content_arr) = item["content"].as_array() {
                                for c in content_arr {
                                    if c["type"].as_str() == Some("output_text") {
                                        if let Some(text) = c["text"].as_str() {
                                            out.push_str(text);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        out
    }

    // ==================== Provider 操作 ====================

    /// 初始化系统 Provider
    /// 检查数据库中是否存在系统 Provider，如果不存在则插入
    /// **Validates: Requirements 9.3**
    pub fn initialize_system_providers(&self, db: &DbConnection) -> Result<usize, String> {
        let conn = lime_core::database::lock_db(db)?;
        let system_providers = get_system_providers();
        let mut inserted_count = 0;
        let mut updated_count = 0;

        for def in &system_providers {
            // 检查是否已存在
            let existing =
                ApiKeyProviderDao::get_provider_by_id(&conn, def.id).map_err(|e| e.to_string())?;

            match existing {
                Some(mut provider) => {
                    if Self::should_update_system_provider_default_api_host(&provider, def.api_host)
                    {
                        provider.api_host = def.api_host.to_string();
                        provider.updated_at = Utc::now();
                        ApiKeyProviderDao::update_provider(&conn, &provider)
                            .map_err(|e| e.to_string())?;
                        updated_count += 1;
                    }
                }
                None => {
                    // 插入新的系统 Provider
                    let provider = to_api_key_provider(def);
                    ApiKeyProviderDao::insert_provider(&conn, &provider)
                        .map_err(|e| e.to_string())?;
                    inserted_count += 1;
                }
            }
        }

        if inserted_count > 0 {
            tracing::info!("初始化了 {} 个系统 Provider", inserted_count);
        }
        if updated_count > 0 {
            tracing::info!("同步了 {} 个系统 Provider 默认地址", updated_count);
        }

        Ok(inserted_count)
    }

    fn should_update_system_provider_default_api_host(
        provider: &ApiKeyProvider,
        next_api_host: &str,
    ) -> bool {
        provider.id == "sensenova"
            && provider.is_system
            && provider.api_host.trim_end_matches('/') == SENSENOVA_OLD_OPENAI_COMPATIBLE_API_HOST
            && next_api_host == SENSENOVA_OPENAI_COMPATIBLE_API_HOST
    }

    /// 获取所有 Provider（包含 API Keys）
    /// 首次调用时会自动初始化系统 Provider
    pub fn get_all_providers(&self, db: &DbConnection) -> Result<Vec<ProviderWithKeys>, String> {
        // 首先确保系统 Provider 已初始化
        self.initialize_system_providers(db)?;

        let conn = lime_core::database::lock_db(db)?;
        let providers =
            ApiKeyProviderDao::get_all_providers_with_keys(&conn).map_err(|e| e.to_string())?;

        tracing::debug!(
            "[ApiKeyProviderService] 获取到 {} 个 Provider",
            providers.len()
        );

        for p in &providers {
            tracing::debug!(
                "[ApiKeyProviderService] Provider: id={}, name={}, api_keys={}",
                p.provider.id,
                p.provider.name,
                p.api_keys.len()
            );
        }

        Ok(providers)
    }

    /// 获取单个 Provider（包含 API Keys）
    pub fn get_provider(
        &self,
        db: &DbConnection,
        id: &str,
    ) -> Result<Option<ProviderWithKeys>, String> {
        let conn = lime_core::database::lock_db(db)?;
        let provider =
            ApiKeyProviderDao::get_provider_by_id(&conn, id).map_err(|e| e.to_string())?;

        match provider {
            Some(p) => {
                let api_keys = ApiKeyProviderDao::get_api_keys_by_provider(&conn, id)
                    .map_err(|e| e.to_string())?;
                Ok(Some(ProviderWithKeys {
                    provider: p,
                    api_keys,
                }))
            }
            None => Ok(None),
        }
    }

    /// 添加自定义 Provider
    pub fn add_custom_provider(
        &self,
        db: &DbConnection,
        name: String,
        provider_type: ApiProviderType,
        api_host: String,
        api_version: Option<String>,
        project: Option<String>,
        location: Option<String>,
        region: Option<String>,
        prompt_cache_mode: Option<ApiProviderPromptCacheMode>,
    ) -> Result<ApiKeyProvider, String> {
        let now = Utc::now();
        let id = format!("custom-{}", uuid::Uuid::new_v4());
        let provider_type = Self::normalize_custom_provider_type(provider_type, &api_host);
        let normalized_prompt_cache_mode =
            Self::normalize_custom_prompt_cache_mode(provider_type, &api_host, prompt_cache_mode);

        let provider = ApiKeyProvider {
            id: id.clone(),
            name,
            provider_type,
            api_host,
            is_system: false,
            group: ProviderGroup::Custom,
            enabled: true,
            sort_order: 9999, // 自定义 Provider 排在最后
            api_version,
            project,
            location,
            region,
            custom_models: Vec::new(),
            prompt_cache_mode: normalized_prompt_cache_mode,
            created_at: now,
            updated_at: now,
        };

        let conn = lime_core::database::lock_db(db)?;
        ApiKeyProviderDao::insert_provider(&conn, &provider).map_err(|e| e.to_string())?;

        Ok(provider)
    }

    /// 更新 Provider 配置
    pub fn update_provider(
        &self,
        db: &DbConnection,
        id: &str,
        name: Option<String>,
        provider_type: Option<ApiProviderType>,
        api_host: Option<String>,
        enabled: Option<bool>,
        sort_order: Option<i32>,
        api_version: Option<String>,
        project: Option<String>,
        location: Option<String>,
        region: Option<String>,
        prompt_cache_mode: Option<ApiProviderPromptCacheMode>,
        custom_models: Option<Vec<String>>,
    ) -> Result<ApiKeyProvider, String> {
        let conn = lime_core::database::lock_db(db)?;
        let mut provider = ApiKeyProviderDao::get_provider_by_id(&conn, id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Provider 不存在: {id}"))?;

        // 更新字段
        if let Some(n) = name {
            provider.name = n;
        }
        if let Some(t) = provider_type {
            provider.provider_type = t;
        }
        if let Some(h) = api_host {
            provider.api_host = h;
        }
        if let Some(e) = enabled {
            provider.enabled = e;
        }
        if let Some(s) = sort_order {
            provider.sort_order = s;
        }
        if let Some(v) = api_version {
            provider.api_version = if v.is_empty() { None } else { Some(v) };
        }
        if let Some(p) = project {
            provider.project = if p.is_empty() { None } else { Some(p) };
        }
        if let Some(l) = location {
            provider.location = if l.is_empty() { None } else { Some(l) };
        }
        if let Some(r) = region {
            provider.region = if r.is_empty() { None } else { Some(r) };
        }
        if let Some(models) = custom_models {
            provider.custom_models = models;
        }
        provider.provider_type =
            Self::normalize_custom_provider_type(provider.provider_type, &provider.api_host);
        provider.prompt_cache_mode = Self::normalize_custom_prompt_cache_mode(
            provider.provider_type,
            &provider.api_host,
            prompt_cache_mode.or(provider.prompt_cache_mode),
        );
        provider.updated_at = Utc::now();

        ApiKeyProviderDao::update_provider(&conn, &provider).map_err(|e| e.to_string())?;

        Ok(provider)
    }

    /// 删除自定义 Provider
    /// 系统 Provider 不允许删除
    pub fn delete_custom_provider(&self, db: &DbConnection, id: &str) -> Result<bool, String> {
        let conn = lime_core::database::lock_db(db)?;

        // 检查是否为系统 Provider
        let provider = ApiKeyProviderDao::get_provider_by_id(&conn, id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Provider 不存在: {id}"))?;

        if provider.is_system {
            return Err("不允许删除系统 Provider".to_string());
        }

        ApiKeyProviderDao::delete_provider(&conn, id).map_err(|e| e.to_string())
    }

    // ==================== API Key 操作 ====================

    /// 添加 API Key
    ///
    /// 当添加第一个 API Key 时，会自动启用 Provider
    /// 使用数据库事务确保操作的原子性
    pub fn add_api_key(
        &self,
        db: &DbConnection,
        provider_id: &str,
        api_key: &str,
        alias: Option<String>,
        replace_existing: bool,
    ) -> Result<ApiKeyEntry, String> {
        tracing::info!(
            "[ApiKeyProviderService] 开始添加 API Key: provider_id={}, replace_existing={}",
            provider_id,
            replace_existing
        );

        let mut conn = lime_core::database::lock_db(db)?;

        // 使用事务确保操作的原子性
        let tx = conn
            .transaction()
            .map_err(|e| format!("开始事务失败: {e}"))?;

        // 验证 Provider 存在
        let provider = ApiKeyProviderDao::get_provider_by_id(&tx, provider_id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Provider 不存在: {provider_id}"))?;

        tracing::info!(
            "[ApiKeyProviderService] 找到 Provider: name={}, id={}",
            provider.name,
            provider.id
        );

        // 检查 API Key 是否已存在（防重复添加）
        let existing_keys = ApiKeyProviderDao::get_api_keys_by_provider(&tx, provider_id)
            .map_err(|e| e.to_string())?;

        tracing::info!(
            "[ApiKeyProviderService] 当前已有 {} 个 API Key",
            existing_keys.len()
        );

        // 检查是否有相同的 API Key（比较加密后的值）
        let encrypted_input = self.encryption.encrypt(api_key);
        if let Some(existing_key) = existing_keys
            .iter()
            .find(|existing_key| existing_key.api_key_encrypted == encrypted_input)
        {
            if !replace_existing {
                return Err("该 API Key 已存在".to_string());
            }

            let now = Utc::now();
            let mut replacement_key = existing_key.clone();
            replacement_key.enabled = true;
            if alias.is_some() {
                replacement_key.alias = alias.clone();
            }
            ApiKeyProviderDao::update_api_key(&tx, &replacement_key).map_err(|e| e.to_string())?;

            for stale_key in existing_keys
                .iter()
                .filter(|key| key.id != replacement_key.id)
            {
                ApiKeyProviderDao::delete_api_key(&tx, &stale_key.id).map_err(|e| e.to_string())?;
            }

            if !provider.enabled {
                let mut updated_provider = provider.clone();
                updated_provider.enabled = true;
                updated_provider.updated_at = now;
                ApiKeyProviderDao::update_provider(&tx, &updated_provider)
                    .map_err(|e| e.to_string())?;
            }

            tx.commit().map_err(|e| format!("提交事务失败: {e}"))?;
            tracing::info!(
                "[ApiKeyProviderService] 已将已有 API Key 设为唯一启用 Key: provider={}, key_id={}",
                provider_id,
                replacement_key.id
            );

            return Ok(replacement_key);
        }

        if replace_existing {
            for existing_key in &existing_keys {
                ApiKeyProviderDao::delete_api_key(&tx, &existing_key.id)
                    .map_err(|e| e.to_string())?;
            }
        }

        let should_enable_provider =
            (replace_existing || existing_keys.is_empty()) && !provider.enabled;

        let now = Utc::now();
        let key = ApiKeyEntry {
            id: uuid::Uuid::new_v4().to_string(),
            provider_id: provider_id.to_string(),
            api_key_encrypted: encrypted_input,
            alias: alias.clone(),
            enabled: true,
            usage_count: 0,
            error_count: 0,
            last_used_at: None,
            created_at: now,
        };

        // 插入 API Key
        ApiKeyProviderDao::insert_api_key(&tx, &key).map_err(|e| e.to_string())?;

        tracing::info!(
            "[ApiKeyProviderService] API Key 已插入: id={}, provider_id={}",
            key.id,
            key.provider_id
        );

        // 如果是第一个 API Key，自动启用 Provider
        if should_enable_provider {
            let mut updated_provider = provider;
            updated_provider.enabled = true;
            updated_provider.updated_at = now;
            ApiKeyProviderDao::update_provider(&tx, &updated_provider)
                .map_err(|e| e.to_string())?;
            tracing::info!(
                "[ApiKeyProviderService] 自动启用 Provider: {} (添加了第一个 API Key)",
                provider_id
            );
        }

        // 提交事务
        tx.commit().map_err(|e| format!("提交事务失败: {e}"))?;

        tracing::info!(
            "[ApiKeyProviderService] 成功添加 API Key: provider={}, alias={:?}",
            provider_id,
            alias
        );

        Ok(key)
    }

    /// 删除 API Key
    pub fn delete_api_key(&self, db: &DbConnection, key_id: &str) -> Result<bool, String> {
        let conn = lime_core::database::lock_db(db)?;
        ApiKeyProviderDao::delete_api_key(&conn, key_id).map_err(|e| e.to_string())
    }

    /// 切换 API Key 启用状态
    pub fn toggle_api_key(
        &self,
        db: &DbConnection,
        key_id: &str,
        enabled: bool,
    ) -> Result<ApiKeyEntry, String> {
        let conn = lime_core::database::lock_db(db)?;
        let mut key = ApiKeyProviderDao::get_api_key_by_id(&conn, key_id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("API Key not found: {key_id}"))?;

        key.enabled = enabled;
        ApiKeyProviderDao::update_api_key(&conn, &key).map_err(|e| e.to_string())?;

        Ok(key)
    }

    /// 更新 API Key 别名
    pub fn update_api_key_alias(
        &self,
        db: &DbConnection,
        key_id: &str,
        alias: Option<String>,
    ) -> Result<ApiKeyEntry, String> {
        let conn = lime_core::database::lock_db(db)?;
        let mut key = ApiKeyProviderDao::get_api_key_by_id(&conn, key_id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("API Key not found: {key_id}"))?;

        key.alias = alias;
        ApiKeyProviderDao::update_api_key(&conn, &key).map_err(|e| e.to_string())?;

        Ok(key)
    }

    // ==================== 轮询负载均衡 ====================

    /// 获取下一个可用的 API Key（轮询负载均衡）
    /// **Validates: Requirements 7.3**
    pub fn get_next_api_key(
        &self,
        db: &DbConnection,
        provider_id: &str,
    ) -> Result<Option<String>, String> {
        let conn = lime_core::database::lock_db(db)?;

        // 获取所有启用的 API Keys
        let keys = ApiKeyProviderDao::get_enabled_api_keys_by_provider(&conn, provider_id)
            .map_err(|e| e.to_string())?;

        if keys.is_empty() {
            return Ok(None);
        }

        let Some((_selected_key, decrypted)) =
            self.select_next_decryptable_api_key(&conn, provider_id, &keys)?
        else {
            return Err(format!(
                "Provider {provider_id} 的所有已启用 API Key 都无法解密"
            ));
        };

        Ok(Some(decrypted))
    }

    /// 获取下一个可用的 API Key 条目（包含 ID，用于记录使用）
    pub fn get_next_api_key_entry(
        &self,
        db: &DbConnection,
        provider_id: &str,
    ) -> Result<Option<(String, String)>, String> {
        let conn = lime_core::database::lock_db(db)?;

        // 获取所有启用的 API Keys
        let keys = ApiKeyProviderDao::get_enabled_api_keys_by_provider(&conn, provider_id)
            .map_err(|e| e.to_string())?;

        if keys.is_empty() {
            return Ok(None);
        }

        let Some((selected_key, decrypted)) =
            self.select_next_decryptable_api_key(&conn, provider_id, &keys)?
        else {
            return Err(format!(
                "Provider {provider_id} 的所有已启用 API Key 都无法解密"
            ));
        };

        Ok(Some((selected_key.id.clone(), decrypted)))
    }

    /// 记录 API Key 使用
    pub fn record_usage(&self, db: &DbConnection, key_id: &str) -> Result<(), String> {
        let conn = lime_core::database::lock_db(db)?;
        let key = ApiKeyProviderDao::get_api_key_by_id(&conn, key_id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("API Key not found: {key_id}"))?;

        ApiKeyProviderDao::update_api_key_usage(&conn, key_id, key.usage_count + 1, Utc::now())
            .map_err(|e| e.to_string())
    }

    /// 获取下一个可用的 API Key 以及 Provider 信息（按 provider_id 精确查找）
    /// 用于支持 X-Provider-Id 请求头指定具体的 Provider
    pub fn get_next_api_key_with_provider_info(
        &self,
        db: &DbConnection,
        provider_id: &str,
    ) -> Result<Option<(String, ApiKeyProvider)>, String> {
        let conn = lime_core::database::lock_db(db)?;

        // 获取 Provider 信息
        let provider = match ApiKeyProviderDao::get_provider_by_id(&conn, provider_id)
            .map_err(|e| e.to_string())?
        {
            Some(p) => p,
            None => return Ok(None),
        };

        // 检查 Provider 是否启用
        if !provider.enabled {
            return Ok(None);
        }

        // 获取该 Provider 的所有启用的 API Keys
        let keys = ApiKeyProviderDao::get_enabled_api_keys_by_provider(&conn, provider_id)
            .map_err(|e| e.to_string())?;

        if keys.is_empty() {
            return Ok(None);
        }

        let Some((_selected_key, decrypted)) =
            self.select_next_decryptable_api_key(&conn, provider_id, &keys)?
        else {
            return Err(format!(
                "Provider {provider_id} 的所有已启用 API Key 都无法解密"
            ));
        };

        Ok(Some((decrypted, provider)))
    }

    /// 按 Provider 类型获取下一个可用的 API Key（轮询负载均衡）
    /// 这个方法会查找所有该类型的 Provider（包括自定义 Provider）
    pub fn get_next_api_key_by_type(
        &self,
        db: &DbConnection,
        provider_type: ApiProviderType,
    ) -> Result<Option<(String, String, ApiKeyProvider)>, String> {
        let conn = lime_core::database::lock_db(db)?;

        // 获取所有启用的 API Keys（按类型）
        let keys = ApiKeyProviderDao::get_enabled_api_keys_by_type(&conn, provider_type)
            .map_err(|e| e.to_string())?;

        if keys.is_empty() {
            return Ok(None);
        }

        let type_key = format!("type:{provider_type}");
        let Some((selected_key, provider, decrypted)) =
            self.select_next_decryptable_api_key_with_provider(&conn, &type_key, &keys)?
        else {
            return Err(format!(
                "Provider 类型 {provider_type} 的所有已启用 API Key 都无法解密"
            ));
        };

        Ok(Some((selected_key.id.clone(), decrypted, provider.clone())))
    }

    /// 记录 API Key 错误
    pub fn record_error(&self, db: &DbConnection, key_id: &str) -> Result<(), String> {
        let conn = lime_core::database::lock_db(db)?;
        ApiKeyProviderDao::increment_api_key_error(&conn, key_id).map_err(|e| e.to_string())
    }

    // ==================== 加密相关 ====================

    /// 检查 API Key 是否已加密
    pub fn is_encrypted(&self, value: &str) -> bool {
        self.encryption.is_encrypted(value)
    }

    /// 解密 API Key（用于 API 调用）
    pub fn decrypt_api_key(&self, encrypted: &str) -> Result<String, String> {
        self.encryption.decrypt(encrypted)
    }

    /// 加密 API Key（用于存储）
    pub fn encrypt_api_key(&self, plaintext: &str) -> String {
        self.encryption.encrypt(plaintext)
    }

    // ==================== UI 状态 ====================

    /// 获取 UI 状态
    pub fn get_ui_state(&self, db: &DbConnection, key: &str) -> Result<Option<String>, String> {
        let conn = lime_core::database::lock_db(db)?;
        ApiKeyProviderDao::get_ui_state(&conn, key).map_err(|e| e.to_string())
    }

    /// 设置 UI 状态
    pub fn set_ui_state(&self, db: &DbConnection, key: &str, value: &str) -> Result<(), String> {
        let conn = lime_core::database::lock_db(db)?;
        ApiKeyProviderDao::set_ui_state(&conn, key, value).map_err(|e| e.to_string())
    }

    /// 批量更新 Provider 排序顺序
    /// **Validates: Requirements 8.4**
    pub fn update_provider_sort_orders(
        &self,
        db: &DbConnection,
        sort_orders: Vec<(String, i32)>,
    ) -> Result<(), String> {
        let conn = lime_core::database::lock_db(db)?;
        ApiKeyProviderDao::update_provider_sort_orders(&conn, &sort_orders)
            .map_err(|e| e.to_string())
    }

    // ==================== 导入导出 ====================

    /// 导出配置
    pub fn export_config(
        &self,
        db: &DbConnection,
        include_keys: bool,
    ) -> Result<serde_json::Value, String> {
        let conn = lime_core::database::lock_db(db)?;
        let providers =
            ApiKeyProviderDao::get_all_providers_with_keys(&conn).map_err(|e| e.to_string())?;

        let export_data = if include_keys {
            // 包含 API Keys（但不包含实际的 key 值）
            let providers_json: Vec<serde_json::Value> = providers
                .iter()
                .map(|p| {
                    let keys: Vec<serde_json::Value> = p
                        .api_keys
                        .iter()
                        .map(|k| {
                            serde_json::json!({
                                "id": k.id,
                                "alias": k.alias,
                                "enabled": k.enabled,
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "provider": p.provider,
                        "api_keys": keys,
                    })
                })
                .collect();
            serde_json::json!({
                "version": "1.0",
                "exported_at": Utc::now().to_rfc3339(),
                "providers": providers_json,
            })
        } else {
            // 不包含 API Keys
            let providers_json: Vec<serde_json::Value> = providers
                .iter()
                .map(|p| serde_json::json!(p.provider))
                .collect();
            serde_json::json!({
                "version": "1.0",
                "exported_at": Utc::now().to_rfc3339(),
                "providers": providers_json,
            })
        };

        Ok(export_data)
    }

    /// 导入配置
    pub fn import_config(
        &self,
        db: &DbConnection,
        config_json: &str,
    ) -> Result<ImportResult, String> {
        let config: serde_json::Value =
            serde_json::from_str(config_json).map_err(|e| format!("JSON 解析失败: {e}"))?;

        let providers = config["providers"]
            .as_array()
            .ok_or_else(|| "配置格式错误: 缺少 providers 数组".to_string())?;

        let conn = lime_core::database::lock_db(db)?;
        let mut imported_providers = 0;
        let mut skipped_providers = 0;
        let mut errors = Vec::new();

        for provider_json in providers {
            let provider_data = if provider_json.get("provider").is_some() {
                &provider_json["provider"]
            } else {
                provider_json
            };

            let id = provider_data["id"]
                .as_str()
                .ok_or_else(|| "Provider 缺少 id".to_string())?;

            // 检查是否已存在
            if ApiKeyProviderDao::get_provider_by_id(&conn, id)
                .map_err(|e| e.to_string())?
                .is_some()
            {
                skipped_providers += 1;
                continue;
            }

            // 解析 Provider
            let provider: ApiKeyProvider = serde_json::from_value(provider_data.clone())
                .map_err(|e| format!("Provider 解析失败: {e}"))?;

            // 插入 Provider
            if let Err(e) = ApiKeyProviderDao::insert_provider(&conn, &provider) {
                errors.push(format!("导入 Provider {id} 失败: {e}"));
                continue;
            }

            imported_providers += 1;
        }

        Ok(ImportResult {
            success: errors.is_empty(),
            imported_providers,
            imported_api_keys: 0, // API Keys 不在导入中包含实际值
            skipped_providers,
            errors,
        })
    }

    // ==================== 运行时凭证选择 ====================

    /// 从 API Key Provider 主路径选择凭证。
    ///
    /// 旧凭证池已退役；运行时不再先读 `provider_pool_credentials`，只从
    /// `api_key_providers` / `api_keys` 选择可用凭证。
    pub async fn select_credential_for_provider(
        &self,
        db: &DbConnection,
        provider_type: &str,
        provider_id_hint: Option<&str>,
        client_type: Option<&lime_core::models::client_type::ClientType>,
    ) -> Result<Option<RuntimeProviderCredential>, String> {
        let mut runtime_provider_type = resolve_runtime_provider_type(provider_type);
        let mut resolved_provider_id_hint = provider_id_hint;

        if is_custom_provider_id(provider_type) {
            resolved_provider_id_hint = Some(provider_type);
        }

        if let Some(custom_provider_id) =
            resolved_provider_id_hint.filter(|id| is_custom_provider_id(id))
        {
            match self.get_provider(db, custom_provider_id) {
                Ok(Some(provider_with_keys)) => {
                    runtime_provider_type = Some(api_provider_type_to_runtime_provider_type(
                        provider_with_keys.provider.provider_type,
                    ));
                    tracing::debug!(
                        "[API_KEY_PROVIDER] custom provider '{}' 真实类型 {:?} -> {:?}",
                        custom_provider_id,
                        provider_with_keys.provider.provider_type,
                        runtime_provider_type
                    );
                }
                Ok(None) => {
                    tracing::debug!(
                        "[API_KEY_PROVIDER] custom provider '{}' 不存在，继续使用解析类型 {:?}",
                        custom_provider_id,
                        runtime_provider_type
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        "[API_KEY_PROVIDER] 查询 custom provider '{}' 失败: {}，继续使用解析类型 {:?}",
                        custom_provider_id,
                        error,
                        runtime_provider_type
                    );
                }
            }
        }

        self.select_runtime_credential(
            db,
            runtime_provider_type.as_ref(),
            resolved_provider_id_hint,
            client_type,
        )
        .await
    }

    /// 根据 RuntimeProviderType 选择运行时凭证。
    ///
    /// 从 API Key Provider 查找可用凭证；旧凭证池不参与运行时选择。
    ///
    /// 选择策略：
    /// 1. 优先通过 provider_id 直接查找 (支持 60+ Provider)
    /// 2. 没有 provider_id 命中时，通过类型映射查找 (RuntimeProviderType → ApiProviderType)
    ///
    /// # 参数
    /// - `db`: 数据库连接
    /// - `runtime_provider_type`: 运行时 Provider 类型
    /// - `provider_id_hint`: 可选的 provider_id 提示，如 "deepseek", "dashscope"
    /// - `client_type`: 客户端类型，用于兼容性检查
    ///
    /// # 返回
    /// - `Ok(Some(credential))`: 找到可用的运行时凭证
    /// - `Ok(None)`: 没有找到可用的运行时凭证
    /// - `Err(e)`: 查询过程中发生错误
    pub async fn select_runtime_credential(
        &self,
        db: &DbConnection,
        runtime_provider_type: Option<&RuntimeProviderType>,
        provider_id_hint: Option<&str>,
        client_type: Option<&lime_core::models::client_type::ClientType>,
    ) -> Result<Option<RuntimeProviderCredential>, String> {
        tracing::debug!(
            "[select_runtime_credential] 开始查找: runtime_provider_type={runtime_provider_type:?}, provider_id_hint={provider_id_hint:?}"
        );

        // 策略 1: 优先通过 provider_id 直接查找 (支持 deepseek, moonshot 等 60+ Provider)
        // 这些 Provider 在 API Key Provider 中有独立配置，应该优先使用
        if let Some(provider_id) = provider_id_hint {
            tracing::debug!("[select_runtime_credential] 尝试按 provider_id '{provider_id}' 查找");
            if let Some(cred) = self
                .find_by_provider_id(db, provider_id, client_type)
                .await?
            {
                tracing::debug!(
                    "[select_runtime_credential] 通过 provider_id '{}' 找到凭证: {:?}",
                    provider_id,
                    cred.name
                );
                return Ok(Some(cred));
            }
            tracing::debug!("[select_runtime_credential] provider_id '{provider_id}' 未找到凭证");
        }

        // 策略 2: 通过类型映射查找。
        let Some(runtime_provider_type) = runtime_provider_type else {
            tracing::debug!(
                "[select_runtime_credential] provider_id_hint={provider_id_hint:?} 未命中，且 Provider 类型已退役"
            );
            return Ok(None);
        };

        let api_type = runtime_provider_type_to_api_type(runtime_provider_type);
        tracing::debug!(
            "[select_runtime_credential] 尝试类型映射: {runtime_provider_type:?} -> {api_type:?}"
        );
        if let Some(cred) = self.find_by_api_type(db, runtime_provider_type, &api_type)? {
            tracing::debug!(
                "[select_runtime_credential] 通过类型映射找到凭证: {:?}",
                cred.name
            );
            return Ok(Some(cred));
        }

        tracing::debug!(
            "[select_runtime_credential] 未找到 {runtime_provider_type:?} 的运行时凭证 (provider_id_hint: {provider_id_hint:?})"
        );
        Ok(None)
    }

    /// 通过 ApiProviderType 查找凭证
    fn find_by_api_type(
        &self,
        db: &DbConnection,
        runtime_provider_type: &RuntimeProviderType,
        api_type: &ApiProviderType,
    ) -> Result<Option<RuntimeProviderCredential>, String> {
        let conn = lime_core::database::lock_db(db)?;

        // 查找该类型的启用的 Provider（按 sort_order 排序）
        let providers = ApiKeyProviderDao::get_all_providers(&conn).map_err(|e| e.to_string())?;

        let matching_providers: Vec<_> = providers
            .into_iter()
            .filter(|p| p.enabled && p.provider_type == *api_type)
            .collect();

        if matching_providers.is_empty() {
            return Ok(None);
        }

        // 尝试从每个匹配的 Provider 获取可用的 API Key
        for provider in matching_providers {
            let keys = ApiKeyProviderDao::get_enabled_api_keys_by_provider(&conn, &provider.id)
                .map_err(|e| e.to_string())?;

            if keys.is_empty() {
                continue;
            }

            // 轮询选择 API Key
            let index = {
                let mut indices = self.round_robin_index.write().map_err(|e| e.to_string())?;
                indices
                    .entry(provider.id.clone())
                    .or_insert_with(|| AtomicUsize::new(0))
                    .fetch_add(1, Ordering::SeqCst)
            };

            let selected_key = &keys[index % keys.len()];

            // 解密 API Key
            let api_key = self.decrypt_api_key_entry_with_migration(&conn, selected_key)?;

            // 转换为 RuntimeProviderCredential
            let credential = self.convert_to_provider_credential(
                runtime_provider_type,
                api_type,
                &provider,
                &selected_key.id,
                &api_key,
            )?;

            tracing::info!(
                "[运行时凭证] 成功找到凭证: {:?} -> {} (key: {})",
                runtime_provider_type,
                provider.name,
                selected_key.alias.as_deref().unwrap_or(&selected_key.id)
            );

            return Ok(Some(credential));
        }

        Ok(None)
    }

    /// 通过 provider_id 直接查找凭证 (支持 60+ Provider)
    ///
    /// 例如: "deepseek", "dashscope", "openrouter"
    async fn find_by_provider_id(
        &self,
        db: &DbConnection,
        provider_id: &str,
        client_type: Option<&lime_core::models::client_type::ClientType>,
    ) -> Result<Option<RuntimeProviderCredential>, String> {
        // First, get all data we need while holding the lock
        let (provider, keys) = {
            let conn = lime_core::database::lock_db(db)?;

            // 直接按 provider_id 查找
            let provider = ApiKeyProviderDao::get_provider_by_id(&conn, provider_id)
                .map_err(|e| e.to_string())?;

            let provider = match provider {
                Some(p) if p.enabled => {
                    tracing::debug!(
                        "[find_by_provider_id] 找到已启用的 provider: id={}, name={}, api_host={}, type={:?}",
                        p.id, p.name, p.api_host, p.provider_type
                    );
                    p
                }
                Some(_p) => {
                    tracing::debug!("[find_by_provider_id] provider '{provider_id}' 存在但未启用");
                    return Ok(None);
                }
                None => {
                    tracing::debug!("[find_by_provider_id] provider '{provider_id}' 不存在");
                    return Ok(None);
                }
            };

            // 获取启用的 API Key
            let keys = ApiKeyProviderDao::get_enabled_api_keys_by_provider(&conn, &provider.id)
                .map_err(|e| e.to_string())?;

            if keys.is_empty() {
                tracing::debug!(
                    "[find_by_provider_id] provider '{provider_id}' 没有启用的 API Key"
                );
                return Ok(None);
            }

            tracing::debug!(
                "[find_by_provider_id] provider '{}' 有 {} 个启用的 API Key",
                provider_id,
                keys.len()
            );

            (provider, keys)
        }; // conn is released here

        // 轮询选择 API Key，但需要检查客户端兼容性
        let mut selected_key = None;
        let mut attempts = 0;
        let max_attempts = keys.len();

        while attempts < max_attempts {
            let index = {
                let mut indices = self.round_robin_index.write().map_err(|e| e.to_string())?;
                indices
                    .entry(provider.id.clone())
                    .or_insert_with(|| AtomicUsize::new(0))
                    .fetch_add(1, Ordering::SeqCst)
            };

            let candidate_key = &keys[index % keys.len()];

            // 解密 API Key 进行测试
            let api_key = {
                let conn = lime_core::database::lock_db(db)?;
                self.decrypt_api_key_entry_with_migration(&conn, candidate_key)?
            };
            let effective_provider_type = provider.effective_provider_type();

            // 检查客户端兼容性（仅对 Anthropic 类型进行检查）
            if effective_provider_type == ApiProviderType::Anthropic {
                if let Some(client) = client_type {
                    // 对于 Claude Code 客户端，可以使用任何 Claude 凭证
                    if matches!(
                        client,
                        lime_core::models::client_type::ClientType::ClaudeCode
                    ) {
                        selected_key = Some(candidate_key);
                        break;
                    }

                    // 对于其他客户端，需要检查凭证是否是 Claude Code 专用
                    // 通过发送测试请求来检查
                    if let Err(e) = self
                        .test_claude_key_compatibility(
                            &api_key,
                            &provider.api_host,
                            provider.supports_automatic_prompt_cache(),
                        )
                        .await
                    {
                        if e.contains("CLAUDE_CODE_ONLY") {
                            tracing::debug!(
                                "[find_by_provider_id] API Key {} 是 Claude Code 专用，跳过 (客户端: {:?})",
                                candidate_key.alias.as_deref().unwrap_or(&candidate_key.id),
                                client
                            );
                            attempts += 1;
                            continue;
                        }
                    }
                }
            }

            selected_key = Some(candidate_key);
            break;
        }

        let selected_key = match selected_key {
            Some(key) => key,
            None => {
                tracing::debug!(
                    "[find_by_provider_id] provider '{provider_id}' 的所有 API Key 都不兼容当前客户端 ({client_type:?})"
                );
                return Ok(None);
            }
        };

        // 解密 API Key
        let api_key = {
            let conn = lime_core::database::lock_db(db)?;
            self.decrypt_api_key_entry_with_migration(&conn, selected_key)?
        };

        // 根据 Provider 类型转换为对应的 RuntimeProviderCredential
        let credential =
            self.convert_provider_to_credential(&provider, &selected_key.id, &api_key)?;

        tracing::info!(
            "[运行时凭证] 成功通过 provider_id 找到凭证: {} (key: {}, type: {:?})",
            provider.name,
            selected_key.alias.as_deref().unwrap_or(&selected_key.id),
            provider.effective_provider_type()
        );

        Ok(Some(credential))
    }

    /// 根据 Provider 类型转换为对应的 RuntimeProviderCredential
    fn convert_provider_to_credential(
        &self,
        provider: &ApiKeyProvider,
        key_id: &str,
        api_key: &str,
    ) -> Result<RuntimeProviderCredential, String> {
        let (credential_data, runtime_provider_type) = match provider.effective_provider_type() {
            ApiProviderType::Anthropic => {
                // Anthropic 类型使用 ClaudeKey
                let data = RuntimeCredentialData::ClaudeKey {
                    api_key: api_key.to_string(),
                    base_url: Some(provider.api_host.clone()),
                };
                (data, RuntimeProviderType::Claude)
            }
            ApiProviderType::AnthropicCompatible => {
                // Anthropic 兼容格式使用 ClaudeKey（与 Anthropic 相同的凭证数据）
                // 但使用 AnthropicCompatible 作为 RuntimeProviderType，以便使用正确的端点
                let data = RuntimeCredentialData::ClaudeKey {
                    api_key: api_key.to_string(),
                    base_url: Some(provider.api_host.clone()),
                };
                (data, RuntimeProviderType::AnthropicCompatible)
            }
            ApiProviderType::Gemini => {
                // Gemini 类型使用 GeminiApiKey
                let data = RuntimeCredentialData::GeminiApiKey {
                    api_key: api_key.to_string(),
                    base_url: Some(provider.api_host.clone()),
                    excluded_models: Vec::new(),
                };
                (data, RuntimeProviderType::GeminiApiKey)
            }
            _ => {
                // 其他类型（OpenAI 兼容）使用 OpenAIKey
                let data = RuntimeCredentialData::OpenAIKey {
                    api_key: api_key.to_string(),
                    base_url: Some(provider.api_host.clone()),
                };
                (data, RuntimeProviderType::OpenAI)
            }
        };

        Ok(RuntimeProviderCredential {
            uuid: runtime_api_key_credential_uuid(key_id),
            provider_type: runtime_provider_type,
            credential: credential_data,
            name: Some(provider.name.clone()),
            prompt_cache_mode_override: provider
                .effective_prompt_cache_mode()
                .map(Self::to_credential_prompt_cache_mode),
        })
    }

    /// 转换为 RuntimeProviderCredential
    fn convert_to_provider_credential(
        &self,
        runtime_provider_type: &RuntimeProviderType,
        api_type: &ApiProviderType,
        provider: &ApiKeyProvider,
        key_id: &str,
        api_key: &str,
    ) -> Result<RuntimeProviderCredential, String> {
        let credential_data = match api_type {
            ApiProviderType::Anthropic => RuntimeCredentialData::ClaudeKey {
                api_key: api_key.to_string(),
                base_url: Some(provider.api_host.clone()),
            },
            ApiProviderType::Gemini => RuntimeCredentialData::GeminiApiKey {
                api_key: api_key.to_string(),
                base_url: Some(provider.api_host.clone()),
                excluded_models: Vec::new(),
            },
            ApiProviderType::Vertexai => RuntimeCredentialData::VertexKey {
                api_key: api_key.to_string(),
                base_url: Some(provider.api_host.clone()),
                model_aliases: std::collections::HashMap::new(),
            },
            // 其他类型（包括 Openai, OpenaiResponse 等）都用 OpenAI Key 格式
            _ => RuntimeCredentialData::OpenAIKey {
                api_key: api_key.to_string(),
                base_url: Some(provider.api_host.clone()),
            },
        };

        Ok(RuntimeProviderCredential {
            uuid: runtime_api_key_credential_uuid(key_id),
            provider_type: *runtime_provider_type,
            credential: credential_data,
            name: Some(provider.name.clone()),
            prompt_cache_mode_override: provider
                .effective_prompt_cache_mode()
                .map(Self::to_credential_prompt_cache_mode),
        })
    }

    // ==================== 连接测试 ====================

    /// 测试 Provider 连接
    ///
    /// 方案 C 实现：
    /// 1. 默认使用 /v1/models 端点测试
    /// 2. 如果 Provider 配置了自定义模型列表，用第一个模型发送简单请求
    ///
    /// # 参数
    /// - `db`: 数据库连接
    /// - `provider_id`: Provider ID
    /// - `model_name`: 可选的模型名称，用于发送测试请求
    ///
    /// # 返回
    /// - `ConnectionTestResult`: 测试结果
    pub async fn test_connection(
        &self,
        db: &DbConnection,
        provider_id: &str,
        model_name: Option<String>,
    ) -> Result<ConnectionTestResult, String> {
        self.test_connection_with_fallback_models(db, provider_id, model_name, Vec::new())
            .await
    }

    /// 测试 Provider 连接（仅使用显式传入模型或 Provider 自定义模型）
    pub async fn test_connection_with_fallback_models(
        &self,
        db: &DbConnection,
        provider_id: &str,
        model_name: Option<String>,
        fallback_models: Vec<String>,
    ) -> Result<ConnectionTestResult, String> {
        use std::time::Instant;

        // 获取 Provider 信息
        let provider_with_keys = self
            .get_provider(db, provider_id)?
            .ok_or_else(|| format!("Provider 不存在: {provider_id}"))?;

        let provider = &provider_with_keys.provider;
        let effective_provider_type = provider.effective_provider_type();

        // 获取一个可用的 API Key
        let api_key = self
            .get_next_api_key(db, provider_id)?
            .ok_or_else(|| "没有可用的 API Key".to_string())?;

        let start_time = Instant::now();

        // 根据 Provider 类型选择测试方式
        let result = match effective_provider_type {
            provider_type if Self::uses_anthropic_protocol(provider_type) => {
                // Anthropic / AnthropicCompatible 不支持 /models，统一发送 /messages 测试请求
                let test_model = Self::pick_test_model(
                    model_name.clone(),
                    &provider.custom_models,
                    &fallback_models,
                )
                .unwrap_or_else(|| "claude-3-haiku-20240307".to_string());

                match self
                    .test_anthropic_connection(
                        &api_key,
                        &provider.api_host,
                        &test_model,
                        provider.supports_automatic_prompt_cache(),
                        effective_provider_type,
                    )
                    .await
                {
                    Ok(models) => Ok(models),
                    Err(e) if e == "CLAUDE_CODE_ONLY" => {
                        // Claude Code 专用凭证限制错误，返回特殊错误信息
                        Err(
                            "凭证限制: 当前 Claude 凭证只能用于 Claude Code，不能用于通用 API 调用"
                                .to_string(),
                        )
                    }
                    Err(e) => Err(e),
                }
            }
            ApiProviderType::Gemini => {
                // Gemini 使用 /models 端点
                self.test_gemini_connection(&api_key, &provider.api_host)
                    .await
            }
            ApiProviderType::Codex => {
                let test_model = Self::pick_test_model(
                    model_name.clone(),
                    &provider.custom_models,
                    &fallback_models,
                )
                .ok_or_else(|| "缺少模型名称：请在自定义模型中填写一个模型名".to_string())?;

                if Self::looks_like_openai_image_model(&test_model) {
                    self.test_openai_responses_image_generation_endpoint(
                        &api_key,
                        &provider.api_host,
                        &test_model,
                    )
                    .await
                } else {
                    self.test_codex_responses_endpoint(
                        &api_key,
                        &provider.api_host,
                        &test_model,
                        "hi",
                        effective_provider_type,
                    )
                    .await
                    .map(|_| vec![test_model])
                }
            }
            provider_type
                if Self::uses_openai_responses_protocol(provider_type)
                    || Self::prefers_openai_responses_endpoint(&provider.api_host) =>
            {
                let test_model = Self::pick_test_model(
                    model_name.clone(),
                    &provider.custom_models,
                    &fallback_models,
                );

                if let Some(test_model) = test_model {
                    if Self::looks_like_openai_image_model(&test_model) {
                        self.test_openai_responses_image_generation_endpoint(
                            &api_key,
                            &provider.api_host,
                            &test_model,
                        )
                        .await
                    } else {
                        self.test_openai_responses_once(
                            &api_key,
                            &provider.api_host,
                            &test_model,
                            "hi",
                        )
                        .await
                        .map(|_| vec![test_model])
                    }
                } else if Self::uses_openai_responses_protocol(provider_type) {
                    self.test_openai_models_endpoint(&api_key, &provider.api_host)
                        .await
                } else {
                    Err(
                        "当前 API Host 指向 Responses 兼容入口，请先在自定义模型中填写一个模型名。"
                            .to_string(),
                    )
                }
            }
            _ => {
                // OpenAI 兼容类型，优先使用 /models 端点
                tracing::debug!("[TEST_CONNECTION] model_name param: {model_name:?}");
                tracing::debug!(
                    "[TEST_CONNECTION] provider.custom_models: {:?}",
                    provider.custom_models
                );
                tracing::debug!(
                    "[TEST_CONNECTION] local_fallback_models_count: {}",
                    fallback_models.len()
                );

                if Self::should_use_openai_image_connection_test(
                    model_name.as_deref(),
                    &provider.custom_models,
                    &fallback_models,
                ) {
                    let image_test_model = Self::pick_openai_image_test_model(
                        model_name.clone(),
                        &provider.custom_models,
                        &fallback_models,
                    )
                    .unwrap_or_else(|| "gpt-images-2".to_string());

                    let image_result = self
                        .test_openai_image_generation_endpoint(
                            &api_key,
                            &provider.api_host,
                            &image_test_model,
                        )
                        .await;
                    tracing::debug!("[TEST_CONNECTION] image_generation result: {image_result:?}");
                    image_result
                } else {
                    let models_result = if Self::is_nearai_provider(provider_id, &provider.api_host)
                    {
                        self.test_nearai_models_endpoint(&api_key, &provider.api_host)
                            .await
                    } else {
                        self.test_openai_models_endpoint(&api_key, &provider.api_host)
                            .await
                    };

                    tracing::debug!("[TEST_CONNECTION] models_result: {models_result:?}");

                    // 如果 /models 端点失败：
                    // 1) 优先用传入的 model_name
                    // 2) 否则使用 Provider 配置的 custom_models
                    if models_result.is_err() {
                        let test_model = Self::pick_test_model(
                            model_name.clone(),
                            &provider.custom_models,
                            &fallback_models,
                        );

                        tracing::debug!("[TEST_CONNECTION] fallback test_model: {test_model:?}");

                        if let Some(test_model) = test_model {
                            let chat_result = self
                                .test_openai_chat_completion(
                                    &api_key,
                                    &provider.api_host,
                                    &test_model,
                                )
                                .await;
                            tracing::debug!(
                                "[TEST_CONNECTION] chat_completion result: {chat_result:?}"
                            );
                            chat_result
                        } else {
                            models_result
                        }
                    } else {
                        models_result
                    }
                }
            }
        };

        let latency_ms = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(models) => Ok(ConnectionTestResult {
                success: true,
                latency_ms: Some(latency_ms),
                error: None,
                models: Some(models),
            }),
            Err(e) => Ok(ConnectionTestResult {
                success: false,
                latency_ms: Some(latency_ms),
                error: Some(e),
                models: None,
            }),
        }
    }

    /// 测试 OpenAI 兼容的 /models 端点
    async fn test_openai_models_endpoint(
        &self,
        api_key: &str,
        api_host: &str,
    ) -> Result<Vec<String>, String> {
        use lime_providers::providers::openai_custom::OpenAICustomProvider;

        let provider =
            OpenAICustomProvider::with_config(api_key.to_string(), Some(api_host.to_string()));

        let response = provider
            .list_models()
            .await
            .map_err(|e| format!("获取模型列表失败: {e}"))?;

        // 解析模型列表
        let models: Vec<String> = response["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if models.is_empty() {
            Err("未获取到任何模型".to_string())
        } else {
            Ok(models)
        }
    }

    async fn test_nearai_models_endpoint(
        &self,
        api_key: &str,
        api_host: &str,
    ) -> Result<Vec<String>, String> {
        let url = Self::build_nearai_models_api_url(api_host);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| format!("创建 NEAR AI 模型目录客户端失败: {e}"))?;

        let mut request = client.get(&url);
        if !api_key.trim().is_empty() {
            request = request.header("Authorization", format!("Bearer {}", api_key.trim()));
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("获取 NEAR AI 模型列表失败: {e}"))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| format!("读取 NEAR AI 模型列表失败: {e}"))?;

        if !status.is_success() {
            return Err(format!("获取 NEAR AI 模型列表失败: {status} - {body}"));
        }

        let payload: Value =
            serde_json::from_str(&body).map_err(|e| format!("解析 NEAR AI 模型列表失败: {e}"))?;
        let models: Vec<String> = payload
            .get("models")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.get("modelId").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();

        if models.is_empty() {
            Err("未获取到任何 NEAR AI 模型".to_string())
        } else {
            Ok(models)
        }
    }

    /// 测试 OpenAI 兼容的 chat/completions 端点
    async fn test_openai_chat_completion(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
    ) -> Result<Vec<String>, String> {
        self.test_openai_chat_once(api_key, api_host, model, "hi", ApiProviderType::Openai)
            .await
            .map(|_| vec![model.to_string()])
    }

    fn normalize_openai_responses_image_tool_model(model: &str) -> String {
        let trimmed = model.trim();
        if let Some(version) = trimmed.strip_prefix("gpt-images-") {
            return format!("gpt-image-{version}");
        }

        trimmed.to_string()
    }

    fn build_openai_responses_image_generation_input(prompt: &str, use_input_list: bool) -> Value {
        if use_input_list {
            return serde_json::json!([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ]);
        }

        serde_json::json!(prompt)
    }

    fn build_openai_responses_image_generation_request(
        image_model: &str,
        prompt: &str,
        use_input_list: bool,
    ) -> Value {
        serde_json::json!({
            "model": OPENAI_RESPONSES_IMAGE_TEST_OUTER_MODEL,
            "input": Self::build_openai_responses_image_generation_input(prompt, use_input_list),
            "tools": [
                {
                    "type": "image_generation",
                    "model": Self::normalize_openai_responses_image_tool_model(image_model),
                }
            ],
            "stream": true,
        })
    }

    fn openai_responses_image_item_has_result(item: &Value) -> bool {
        item.get("type").and_then(Value::as_str) == Some("image_generation_call")
            && item
                .get("result")
                .or_else(|| item.get("partial_image_b64"))
                .and_then(Value::as_str)
                .map(str::trim)
                .is_some_and(|value| !value.is_empty())
    }

    fn openai_responses_image_value_has_result(value: &Value) -> bool {
        if let Some(item) = value.get("item") {
            if Self::openai_responses_image_item_has_result(item) {
                return true;
            }
        }

        value
            .get("output")
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items
                    .iter()
                    .any(Self::openai_responses_image_item_has_result)
            })
    }

    fn openai_responses_image_body_has_result(body: &str) -> bool {
        if let Ok(value) = serde_json::from_str::<Value>(body) {
            if Self::openai_responses_image_value_has_result(&value) {
                return true;
            }
        }

        body.lines()
            .filter_map(|line| line.trim().strip_prefix("data:"))
            .map(str::trim)
            .filter(|data| !data.is_empty() && *data != "[DONE]")
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .any(|value| Self::openai_responses_image_value_has_result(&value))
    }

    fn should_retry_openai_responses_image_generation_with_input_list(
        status: reqwest::StatusCode,
        body: &str,
    ) -> bool {
        status == reqwest::StatusCode::BAD_REQUEST
            && body.to_ascii_lowercase().contains("input must be a list")
    }

    async fn send_openai_responses_image_generation_test(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
        use_input_list: bool,
    ) -> Result<(reqwest::StatusCode, String), String> {
        use lime_providers::providers::codex::CodexProvider;

        let url = CodexProvider::build_responses_url(api_host);
        let request = Self::build_openai_responses_image_generation_request(
            model,
            "生成一个简单的蓝色渐变方块测试图",
            use_input_list,
        );

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(240))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Responses 图片生成接口调用失败: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Ok((status, body))
    }

    async fn test_openai_responses_image_generation_endpoint(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
    ) -> Result<Vec<String>, String> {
        let (mut status, mut body) = self
            .send_openai_responses_image_generation_test(api_key, api_host, model, false)
            .await?;

        if Self::should_retry_openai_responses_image_generation_with_input_list(status, &body) {
            (status, body) = self
                .send_openai_responses_image_generation_test(api_key, api_host, model, true)
                .await?;
        }

        if !status.is_success() {
            return Err(format!(
                "Responses 图片生成接口返回错误: {}",
                Self::format_http_api_error(status, &body)
            ));
        }

        if !Self::openai_responses_image_body_has_result(&body) {
            return Err(
                "Responses 图片生成接口调用成功，但未返回 image_generation_call.result".to_string(),
            );
        }

        Ok(vec![model.to_string()])
    }

    async fn test_openai_image_generation_endpoint(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
    ) -> Result<Vec<String>, String> {
        let url = Self::build_openai_images_url(api_host);
        let request = serde_json::json!({
            "model": model,
            "prompt": "生成一个简单的蓝色渐变方块测试图",
            "n": 1,
            "size": "1024x1024",
            "response_format": "b64_json"
        });

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(240))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("图片生成接口调用失败: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(format!(
                "图片生成接口返回错误: {}",
                Self::format_http_api_error(status, &body)
            ));
        }

        let parsed: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| format!("解析响应失败: {e} - {body}"))?;
        let has_image = parsed["data"].as_array().is_some_and(|items| {
            items.iter().any(|item| {
                item.get("url")
                    .and_then(|value| value.as_str())
                    .map(|value| !value.trim().is_empty())
                    .unwrap_or(false)
                    || item
                        .get("b64_json")
                        .and_then(|value| value.as_str())
                        .map(|value| !value.trim().is_empty())
                        .unwrap_or(false)
            })
        });

        if !has_image {
            return Err("图片生成接口调用成功，但未返回可解析图片数据".to_string());
        }

        Ok(vec![model.to_string()])
    }

    /// 测试 Claude Key 的客户端兼容性
    async fn test_claude_key_compatibility(
        &self,
        api_key: &str,
        api_host: &str,
        enable_automatic_prompt_cache: bool,
    ) -> Result<(), String> {
        use lime_providers::providers::claude_custom::{ClaudeCustomProvider, PromptCacheMode};

        let prompt_cache_mode = if enable_automatic_prompt_cache {
            PromptCacheMode::Automatic
        } else {
            PromptCacheMode::ExplicitOnly
        };
        let provider = ClaudeCustomProvider::with_prompt_cache_mode(
            api_key.to_string(),
            Some(api_host.to_string()),
            prompt_cache_mode,
        );

        // 发送一个最小的测试请求
        let request = serde_json::json!({
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]
        });

        let response = provider
            .messages(&request)
            .await
            .map_err(|e| format!("API 调用失败: {e}"))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let body = response.text().await.unwrap_or_default();

            // 检查是否是 Claude Code 专用凭证限制错误
            if body.contains("only authorized for use with Claude Code") {
                return Err("CLAUDE_CODE_ONLY".to_string());
            }

            // 其他错误不影响兼容性判断
            Ok(())
        }
    }

    /// 测试 Anthropic 连接
    async fn test_anthropic_connection(
        &self,
        api_key: &str,
        api_host: &str,
        model: &str,
        enable_automatic_prompt_cache: bool,
        provider_type: ApiProviderType,
    ) -> Result<Vec<String>, String> {
        use lime_providers::providers::claude_custom::{ClaudeCustomProvider, PromptCacheMode};

        let prompt_cache_mode = if enable_automatic_prompt_cache {
            PromptCacheMode::Automatic
        } else {
            PromptCacheMode::ExplicitOnly
        };
        let provider = ClaudeCustomProvider::with_provider_type_and_prompt_cache_mode(
            api_key.to_string(),
            Some(api_host.to_string()),
            provider_type,
            prompt_cache_mode,
        );

        // 发送一个简单的测试请求
        let request = serde_json::json!({
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]
        });

        let (status, body) = Self::execute_anthropic_test_request_with_retries(|| async {
            provider
                .messages(&request)
                .await
                .map_err(|e| format!("API 调用失败: {e}"))
        })
        .await?;

        if status.is_success() {
            Ok(vec![model.to_string()])
        } else {
            // 检查是否是 Claude Code 专用凭证限制错误
            if body.contains("only authorized for use with Claude Code") {
                return Err("CLAUDE_CODE_ONLY".to_string());
            }

            Err(Self::format_anthropic_http_api_error(
                status,
                &body,
                api_host,
                provider_type,
                api_key,
            ))
        }
    }

    /// 测试 Gemini 连接
    async fn test_gemini_connection(
        &self,
        api_key: &str,
        api_host: &str,
    ) -> Result<Vec<String>, String> {
        use reqwest::Client;
        use std::time::Duration;

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("创建 HTTP 客户端失败: {e}"))?;

        // Gemini API 的模型列表端点
        let base = api_host.trim_end_matches('/');
        let url = format!("{base}/v1beta/models?key={api_key}");

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("请求失败: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("API 返回错误: {status} - {body}"));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("解析响应失败: {e}"))?;

        let models: Vec<String> = data["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if models.is_empty() {
            Err("未获取到任何模型".to_string())
        } else {
            Ok(models)
        }
    }
}

/// 导入结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    pub success: bool,
    pub imported_providers: usize,
    pub imported_api_keys: usize,
    pub skipped_providers: usize,
    pub errors: Vec<String>,
}
