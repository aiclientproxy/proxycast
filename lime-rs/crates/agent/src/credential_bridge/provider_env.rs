use lime_core::database::dao::api_key_provider::{infer_managed_runtime_spec, ApiProviderType};
use model_provider::provider_url::{lime_tenant_id_from_base_url, LIME_TENANT_HEADER};
use model_provider::runtime_provider::{RuntimeProviderConfig, RuntimeProviderProtocol};
use model_provider::safety::should_disable_provider_default_fast_model as should_disable_default_fast_model_policy;

pub(super) fn should_disable_provider_default_fast_model(config: &RuntimeProviderConfig) -> bool {
    let protocol = config
        .protocol
        .map(RuntimeProviderProtocol::to_model_provider_protocol);
    should_disable_default_fast_model_policy(
        &config.provider_name,
        config.provider_selector.as_deref(),
        config.base_url.as_deref(),
        protocol.as_ref(),
    )
}

/// 从 URL 中拆分 host（scheme+authority）和 path 部分
///
/// 例如：
/// - `https://api.openai.com` -> (`https://api.openai.com`, ``)
/// - `https://open.bigmodel.cn/api/paas/v4` -> (`https://open.bigmodel.cn`, `api/paas/v4`)
/// - `https://localhost:8080/v1` -> (`https://localhost:8080`, `v1`)
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn split_url_host_and_path(url: &str) -> (String, String) {
    let url = strip_url_query_fragment(url);
    let after_scheme = if let Some(pos) = url.find("://") {
        pos + 3
    } else {
        return (url, String::new());
    };

    let path_start = url[after_scheme..].find('/').map(|p| p + after_scheme);

    match path_start {
        Some(pos) => {
            let host = url[..pos].to_string();
            let path = url[pos..].trim_matches('/').to_string();
            (host, path)
        }
        None => (url.to_string(), String::new()),
    }
}

fn resolve_anthropic_env_key(config: &RuntimeProviderConfig) -> &'static str {
    let api_host = config
        .base_url
        .as_deref()
        .unwrap_or("https://api.anthropic.com");
    let hinted_provider_type = config
        .provider_selector
        .as_deref()
        .and_then(|value| value.parse::<ApiProviderType>().ok())
        .unwrap_or(ApiProviderType::Anthropic);
    let runtime_spec = infer_managed_runtime_spec(hinted_provider_type, api_host);

    if runtime_spec
        .auth_header
        .eq_ignore_ascii_case("authorization")
    {
        "ANTHROPIC_AUTH_TOKEN"
    } else {
        "ANTHROPIC_API_KEY"
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) const OPENAI_CUSTOM_HEADERS_ENV: &str = "OPENAI_CUSTOM_HEADERS";
const OPENAI_ALLOW_NO_AUTH_ENV: &str = "OPENAI_ALLOW_NO_AUTH";

fn strip_url_query_fragment(value: &str) -> String {
    let trimmed = value.trim();
    let end = [trimmed.find('?'), trimmed.find('#')]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(trimmed.len());

    trimmed[..end].trim_end_matches('/').to_string()
}

fn parse_openai_custom_headers_env(value: &str) -> Vec<(String, String)> {
    value
        .split(',')
        .filter_map(|header| {
            let mut parts = header.splitn(2, '=');
            let key = parts.next()?.trim();
            if key.is_empty() {
                return None;
            }
            let value = parts.next().unwrap_or_default().trim();
            Some((key.to_string(), value.to_string()))
        })
        .collect()
}

fn update_openai_lime_tenant_custom_header(tenant_id: Option<&str>) {
    let existing = std::env::var(OPENAI_CUSTOM_HEADERS_ENV).unwrap_or_default();
    let mut headers = parse_openai_custom_headers_env(&existing)
        .into_iter()
        .filter(|(key, _)| !key.eq_ignore_ascii_case(LIME_TENANT_HEADER))
        .collect::<Vec<_>>();

    if let Some(tenant_id) = tenant_id {
        headers.push((LIME_TENANT_HEADER.to_string(), tenant_id));
    }

    if headers.is_empty() {
        std::env::remove_var(OPENAI_CUSTOM_HEADERS_ENV);
        return;
    }

    let value = headers
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(",");
    std::env::set_var(OPENAI_CUSTOM_HEADERS_ENV, value);
}

pub(super) fn set_provider_env_vars(config: &RuntimeProviderConfig) {
    tracing::info!(
        "[CredentialBridge] set_provider_env_vars: provider_name={}, has_api_key={}, base_url={:?}",
        config.provider_name,
        config.api_key.is_some(),
        config.base_url
    );

    let env_key = match config.provider_name.as_str() {
        "openai" => "OPENAI_API_KEY",
        "anthropic" => resolve_anthropic_env_key(config),
        "google" => "GOOGLE_API_KEY",
        "bedrock" => "AWS_ACCESS_KEY_ID",
        "gcpvertexai" => "GOOGLE_API_KEY",
        "codex" => "OPENAI_API_KEY",
        "deepseek" | "custom_deepseek" => "OPENAI_API_KEY",
        "groq" => "OPENAI_API_KEY",
        "mistral" => "OPENAI_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        _ => "OPENAI_API_KEY",
    };

    tracing::info!("[CredentialBridge] 设置环境变量: {}=***", env_key);

    if let Some(api_key) = &config.api_key {
        std::env::set_var(env_key, api_key);
        if config.provider_name == "openai" {
            std::env::remove_var(OPENAI_ALLOW_NO_AUTH_ENV);
        }
        if config.provider_name == "anthropic" {
            match env_key {
                "ANTHROPIC_AUTH_TOKEN" => std::env::remove_var("ANTHROPIC_API_KEY"),
                "ANTHROPIC_API_KEY" => std::env::remove_var("ANTHROPIC_AUTH_TOKEN"),
                _ => {}
            }
        }
    } else {
        std::env::remove_var(env_key);
        if config.provider_name == "openai" {
            std::env::set_var(OPENAI_ALLOW_NO_AUTH_ENV, "1");
        }
    }

    if config.provider_name == "openai" {
        if config
            .protocol
            .is_some_and(RuntimeProviderProtocol::uses_responses_api)
        {
            std::env::set_var("OPENAI_FORCE_RESPONSES_API", "1");
        } else {
            std::env::remove_var("OPENAI_FORCE_RESPONSES_API");
        }
        if config.base_url.is_none() {
            update_openai_lime_tenant_custom_header(None);
            std::env::remove_var("OPENAI_HOST");
            std::env::remove_var("OPENAI_BASE_PATH");
        }
    }

    if let Some(base_url) = &config.base_url {
        match config.provider_name.as_str() {
            "openai" => {
                let tenant_id = lime_tenant_id_from_base_url(base_url);
                update_openai_lime_tenant_custom_header(tenant_id.as_deref());

                let sanitized_base_url = strip_url_query_fragment(base_url);
                let (host_part, path_part) = split_url_host_and_path(&sanitized_base_url);
                if path_part.is_empty() {
                    std::env::set_var("OPENAI_HOST", &sanitized_base_url);
                    std::env::remove_var("OPENAI_BASE_PATH");
                    tracing::info!("[CredentialBridge] 设置 OPENAI_HOST={}", sanitized_base_url);
                } else {
                    let base_path = format!("{}/chat/completions", path_part);
                    std::env::set_var("OPENAI_HOST", &host_part);
                    std::env::set_var("OPENAI_BASE_PATH", &base_path);
                    tracing::info!(
                        "[CredentialBridge] 设置 OPENAI_HOST={}, OPENAI_BASE_PATH={}",
                        host_part,
                        base_path
                    );
                }
            }
            "anthropic" => {
                std::env::set_var("ANTHROPIC_HOST", base_url);
                std::env::set_var("ANTHROPIC_BASE_URL", base_url);
                tracing::info!(
                    "[CredentialBridge] 设置 ANTHROPIC_HOST={}, ANTHROPIC_BASE_URL={}",
                    base_url,
                    base_url
                );
            }
            "ollama" => {
                std::env::set_var("OLLAMA_BASE_URL", base_url);
                std::env::set_var("OLLAMA_HOST", base_url);
                tracing::info!(
                    "[CredentialBridge] 设置 OLLAMA_BASE_URL={}, OLLAMA_HOST={}",
                    base_url,
                    base_url
                );
            }
            _ => {
                let base_url_key = format!(
                    "{}_BASE_URL",
                    config.provider_name.to_uppercase().replace('-', "_")
                );
                std::env::set_var(&base_url_key, base_url);
            }
        }
    }
}
