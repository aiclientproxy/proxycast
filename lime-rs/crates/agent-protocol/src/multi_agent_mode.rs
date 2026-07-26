use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Effective multi-agent delegation policy copied from the Codex protocol shape.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum MultiAgentMode {
    Custom(String),
    #[default]
    ExplicitRequestOnly,
    Proactive,
}

impl MultiAgentMode {
    pub fn from_reasoning_effort(effort: Option<&str>) -> Self {
        if effort.is_some_and(|effort| effort.trim().eq_ignore_ascii_case("ultra")) {
            Self::Proactive
        } else {
            Self::ExplicitRequestOnly
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn uses_codex_wire_shape() {
        for (wire, expected) in [
            (
                json!("explicitRequestOnly"),
                MultiAgentMode::ExplicitRequestOnly,
            ),
            (json!("proactive"), MultiAgentMode::Proactive),
            (
                json!({ "custom": "Delegate independent work." }),
                MultiAgentMode::Custom("Delegate independent work.".to_string()),
            ),
        ] {
            let mode = serde_json::from_value::<MultiAgentMode>(wire.clone())
                .expect("typed multi-agent mode");
            assert_eq!(mode, expected);
            assert_eq!(serde_json::to_value(mode).expect("serialize mode"), wire);
        }
    }

    #[test]
    fn effective_mode_follows_codex_reasoning_rule() {
        assert_eq!(
            MultiAgentMode::from_reasoning_effort(Some("ultra")),
            MultiAgentMode::Proactive
        );
        for effort in [None, Some(""), Some("high"), Some("xhigh")] {
            assert_eq!(
                MultiAgentMode::from_reasoning_effort(effort),
                MultiAgentMode::ExplicitRequestOnly,
                "effort={effort:?}"
            );
        }
    }
}
