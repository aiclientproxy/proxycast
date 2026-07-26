use crate::runtime_provider::RuntimeProviderProtocol;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub namespace_tools: bool,
    pub image_generation: bool,
    pub web_search: bool,
}

impl ProviderCapabilities {
    pub const NONE: Self = Self {
        namespace_tools: false,
        image_generation: false,
        web_search: false,
    };

    pub fn from_provider_type(provider_type: &str) -> Option<Self> {
        RuntimeProviderProtocol::from_provider_type(provider_type).map(|_| Self::NONE)
    }
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
            "anthropic",
            "anthropic-compatible",
            "anthropic_compatible",
        ] {
            assert_eq!(
                ProviderCapabilities::from_provider_type(provider_type),
                expected,
                "provider_type={provider_type}"
            );
        }
    }

    #[test]
    fn providers_without_current_chat_adapters_have_no_capability_snapshot() {
        for provider_type in [
            "gemini",
            "azure-openai",
            "vertexai",
            "aws-bedrock",
            "ollama",
            "fal",
        ] {
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
