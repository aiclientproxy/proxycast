#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderCapabilities {
    pub namespace_tools: bool,
    pub image_generation: bool,
    pub web_search: bool,
}

impl ProviderCapabilities {
    pub fn from_provider_type(provider_type: &str) -> Option<Self> {
        let provider_type = provider_type.trim().to_ascii_lowercase();
        if provider_type.is_empty() {
            return None;
        }

        Some(match provider_type.as_str() {
            "aws-bedrock" | "aws_bedrock" | "bedrock" => Self {
                namespace_tools: true,
                image_generation: false,
                web_search: false,
            },
            "openai"
            | "openai-response"
            | "codex"
            | "anthropic"
            | "anthropic-compatible"
            | "gemini"
            | "azure-openai"
            | "vertexai"
            | "ollama"
            | "fal"
            | "new-api"
            | "gateway" => Self::default(),
            _ => return None,
        })
    }
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            namespace_tools: true,
            image_generation: true,
            web_search: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_provider_capabilities_match_codex() {
        assert_eq!(
            ProviderCapabilities::from_provider_type("openai-response"),
            Some(ProviderCapabilities {
                namespace_tools: true,
                image_generation: true,
                web_search: true,
            })
        );
    }

    #[test]
    fn bedrock_disables_provider_hosted_tools_like_codex() {
        assert_eq!(
            ProviderCapabilities::from_provider_type("aws-bedrock"),
            Some(ProviderCapabilities {
                namespace_tools: true,
                image_generation: false,
                web_search: false,
            })
        );
    }

    #[test]
    fn unknown_provider_type_fails_closed() {
        assert_eq!(
            ProviderCapabilities::from_provider_type("future-provider"),
            None
        );
    }
}
