use agent_protocol::world_state::{RuntimeWorldState, WORLD_STATE_TURN_METADATA_KEY};

pub(super) fn primary_environment_id(
    turn_context: Option<&agent_protocol::turn_context::TurnContextOverride>,
) -> Option<String> {
    let world_state = turn_context?
        .metadata
        .get(WORLD_STATE_TURN_METADATA_KEY)
        .cloned()
        .and_then(|value| serde_json::from_value::<RuntimeWorldState>(value).ok())?;
    world_state
        .environments
        .into_iter()
        .find(|environment| environment.primary)
        .map(|environment| environment.environment_id)
        .filter(|environment_id| !environment_id.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn reads_primary_environment_from_typed_world_state() {
        let context = agent_protocol::turn_context::TurnContextOverride {
            metadata: HashMap::from([(
                WORLD_STATE_TURN_METADATA_KEY.to_string(),
                json!({
                    "environments": [
                        {"environmentId": "local", "cwd": "/local", "primary": false},
                        {"environmentId": "remote", "cwd": "/remote", "primary": true}
                    ]
                }),
            )]),
            ..Default::default()
        };

        assert_eq!(
            primary_environment_id(Some(&context)).as_deref(),
            Some("remote")
        );
    }
}
