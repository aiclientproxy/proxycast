use crate::runtime_provider::{RuntimeProviderConfig, RuntimeProviderProtocol};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub namespace_tools: bool,
    pub custom_tools: bool,
    pub image_generation: bool,
    pub web_search: bool,
}

impl ProviderCapabilities {
    pub const NONE: Self = Self {
        namespace_tools: false,
        custom_tools: false,
        image_generation: false,
        web_search: false,
    };

    pub fn from_provider_type(provider_type: &str) -> Option<Self> {
        RuntimeProviderProtocol::from_provider_type(provider_type).map(|_| Self::NONE)
    }

    pub fn from_provider_route(provider_type: &str, base_url: Option<&str>) -> Option<Self> {
        let protocol = RuntimeProviderProtocol::from_provider_type(provider_type)?;
        Some(Self::from_resolved_route(provider_type, protocol, base_url))
    }

    pub fn from_resolved_route(
        provider_name: &str,
        protocol: RuntimeProviderProtocol,
        base_url: Option<&str>,
    ) -> Self {
        let provider = normalize_provider(provider_name);
        let official_responses_provider = matches!(
            provider.as_str(),
            "openai" | "openai_response" | "openai_responses" | "responses" | "codex"
        );
        let hosted_tools = protocol == RuntimeProviderProtocol::Responses
            && official_responses_provider
            && is_official_openai_host(base_url);
        Self {
            namespace_tools: false,
            custom_tools: hosted_tools,
            image_generation: hosted_tools,
            web_search: hosted_tools,
        }
    }

    pub fn from_runtime_config(config: &RuntimeProviderConfig) -> Self {
        config.protocol.map_or(Self::NONE, |protocol| {
            Self::from_resolved_route(&config.provider_name, protocol, config.base_url.as_deref())
        })
    }
}

fn normalize_provider(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn is_official_openai_host(base_url: Option<&str>) -> bool {
    let Some(base_url) = base_url.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    url::Url::parse(base_url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .is_some_and(|host| host.eq_ignore_ascii_case("api.openai.com"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_chat_adapters_do_not_claim_unimplemented_hosted_tools() {
        let expected = Some(ProviderCapabilities::NONE);

        for provider_type in [
            "openai",
            "new-api",
            "gateway",
            "openai-response",
            "openai_responses",
            "responses",
            "codex",
            "ollama",
            "anthropic",
            "anthropic-compatible",
            "anthropic_compatible",
            "gemini",
            "gemini-api-key",
            "google",
            "azure-openai",
            "vertexai",
        ] {
            assert_eq!(
                ProviderCapabilities::from_provider_type(provider_type),
                expected,
                "provider_type={provider_type}"
            );
        }
    }

    #[test]
    fn hosted_tools_require_official_responses_route_and_host() {
        let expected = Some(ProviderCapabilities {
            namespace_tools: false,
            custom_tools: true,
            image_generation: true,
            web_search: true,
        });
        assert_eq!(
            ProviderCapabilities::from_provider_route(
                "openai-response",
                Some("https://api.openai.com/v1"),
            ),
            expected
        );
        assert_eq!(
            ProviderCapabilities::from_provider_route(
                "codex",
                Some("https://api.openai.com/v1/responses"),
            ),
            expected
        );
        assert_eq!(
            ProviderCapabilities::from_provider_route(
                "openai-response",
                Some("https://gateway.example.com/v1"),
            ),
            Some(ProviderCapabilities::NONE)
        );
        assert_eq!(
            ProviderCapabilities::from_provider_route("ollama", Some("http://127.0.0.1:11434"),),
            Some(ProviderCapabilities::NONE)
        );
        assert_eq!(
            ProviderCapabilities::from_provider_route(
                "azure-openai",
                Some("https://resource.openai.azure.com"),
            ),
            Some(ProviderCapabilities::NONE)
        );
        assert_eq!(
            ProviderCapabilities::from_resolved_route(
                "openai",
                RuntimeProviderProtocol::Responses,
                Some("https://api.openai.com/v1"),
            ),
            expected.expect("official Responses capability")
        );
        assert_eq!(
            ProviderCapabilities::from_resolved_route(
                "openai",
                RuntimeProviderProtocol::ChatCompletions,
                Some("https://api.openai.com/v1"),
            ),
            ProviderCapabilities::NONE
        );
        assert_eq!(
            ProviderCapabilities::from_resolved_route(
                "gateway",
                RuntimeProviderProtocol::Responses,
                Some("https://api.openai.com/v1"),
            ),
            ProviderCapabilities::NONE
        );
        assert_eq!(
            ProviderCapabilities::from_resolved_route(
                "azure-openai",
                RuntimeProviderProtocol::AzureResponses,
                Some("https://resource.openai.azure.com"),
            ),
            ProviderCapabilities::NONE
        );
    }

    #[test]
    fn providers_without_current_chat_adapters_have_no_capability_snapshot() {
        for provider_type in ["aws-bedrock", "fal"] {
            assert_eq!(
                ProviderCapabilities::from_provider_type(provider_type),
                None,
                "provider_type={provider_type}"
            );
        }
    }

    #[test]
    fn unknown_provider_type_fails_closed() {
        assert_eq!(
            ProviderCapabilities::from_provider_type("future-provider"),
            None
        );
    }
}
