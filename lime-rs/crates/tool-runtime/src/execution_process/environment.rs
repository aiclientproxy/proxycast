use std::collections::HashMap;

const SENSITIVE_INHERITED_NAME_FRAGMENTS: [&str; 3] = ["KEY", "SECRET", "TOKEN"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EnvironmentKeySemantics {
    Native,
    CaseInsensitive,
}

pub(super) fn resolve_child_environment(
    env_clear: bool,
    inherited: impl IntoIterator<Item = (String, String)>,
    overrides: &HashMap<String, String>,
) -> HashMap<String, String> {
    let key_semantics = if cfg!(target_os = "windows") {
        EnvironmentKeySemantics::CaseInsensitive
    } else {
        EnvironmentKeySemantics::Native
    };
    resolve_child_environment_with_semantics(env_clear, inherited, overrides, key_semantics)
}

pub(super) fn resolve_child_environment_with_semantics(
    env_clear: bool,
    inherited: impl IntoIterator<Item = (String, String)>,
    overrides: &HashMap<String, String>,
    key_semantics: EnvironmentKeySemantics,
) -> HashMap<String, String> {
    let mut resolved = HashMap::new();
    if !env_clear {
        resolved.extend(
            inherited
                .into_iter()
                .filter(|(key, _)| !is_sensitive_inherited_name(key))
                .map(|(key, value)| (normalize_key(key, key_semantics), value)),
        );
    }
    resolved.extend(
        overrides
            .iter()
            .map(|(key, value)| (normalize_key(key.clone(), key_semantics), value.clone())),
    );
    resolved
}

fn is_sensitive_inherited_name(name: &str) -> bool {
    let uppercase = name.to_ascii_uppercase();
    SENSITIVE_INHERITED_NAME_FRAGMENTS
        .iter()
        .any(|fragment| uppercase.contains(fragment))
}

fn normalize_key(key: String, key_semantics: EnvironmentKeySemantics) -> String {
    match key_semantics {
        EnvironmentKeySemantics::Native => key,
        EnvironmentKeySemantics::CaseInsensitive => key.to_ascii_uppercase(),
    }
}
