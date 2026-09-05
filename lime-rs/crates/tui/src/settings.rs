use anyhow::{Result, anyhow};

use crate::slash_command::{SlashCommand, command_from_prompt};

pub(crate) const EFFORTS: [&str; 3] = ["low", "medium", "high"];
pub(crate) const PERMISSION_PROFILES: [&str; 3] =
    [":read-only", ":workspace", ":danger-full-access"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SettingsCommand {
    ModelPicker,
    Model {
        model: String,
        provider: Option<String>,
    },
    Effort(String),
    Permissions(String),
}

pub(crate) fn parse_settings_command(prompt: &str) -> Option<Result<SettingsCommand>> {
    let mut parts = prompt.split_whitespace();
    let command = command_from_prompt(prompt)?;
    let _ = parts.next();
    let value = parts.next().filter(|value| !value.is_empty());
    match command {
        SlashCommand::Model => Some(value.map_or(Ok(SettingsCommand::ModelPicker), |model| {
            Ok(SettingsCommand::Model {
                model: model.to_string(),
                provider: parts.next().map(ToString::to_string),
            })
        })),
        SlashCommand::Effort => Some(value.map_or_else(
            || Err(anyhow!("usage: /effort <low|medium|high>")),
            |effort| Ok(SettingsCommand::Effort(effort.to_string())),
        )),
        SlashCommand::Permissions => Some(value.map_or_else(
            || Err(anyhow!("usage: /permissions <profile>")),
            |permissions| Ok(SettingsCommand::Permissions(permissions.to_string())),
        )),
        SlashCommand::Status | SlashCommand::Copy => None,
    }
}

pub(crate) fn cycle_setting<T: AsRef<str>>(
    values: &[T],
    current: Option<&str>,
    direction: i8,
) -> String {
    let current_index = current
        .and_then(|value| {
            values
                .iter()
                .position(|candidate| candidate.as_ref() == value)
        })
        .unwrap_or(0);
    let next_index = if direction < 0 {
        if current_index == 0 {
            values.len() - 1
        } else {
            current_index - 1
        }
    } else {
        (current_index + 1) % values.len()
    };
    values[next_index].as_ref().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortcuts_cycle_codex_order_and_recover_unknown_values() {
        assert_eq!(cycle_setting(&EFFORTS, Some("medium"), 1), "high");
        assert_eq!(cycle_setting(&EFFORTS, Some("low"), -1), "high");
        assert_eq!(
            cycle_setting(&PERMISSION_PROFILES, Some("unknown"), 1),
            ":workspace"
        );
    }

    #[test]
    fn commands_parse_without_becoming_turn_input() {
        assert!(matches!(
            parse_settings_command("/model"),
            Some(Ok(SettingsCommand::ModelPicker))
        ));
        assert!(matches!(
            parse_settings_command("/model grok-test grok"),
            Some(Ok(SettingsCommand::Model { model, provider }))
                if model == "grok-test" && provider.as_deref() == Some("grok")
        ));
        assert!(matches!(
            parse_settings_command("/effort high"),
            Some(Ok(SettingsCommand::Effort(effort))) if effort == "high"
        ));
        assert!(matches!(
            parse_settings_command("/permissions :workspace"),
            Some(Ok(SettingsCommand::Permissions(permissions))) if permissions == ":workspace"
        ));
        assert!(parse_settings_command("ordinary question").is_none());
        assert!(parse_settings_command("/effort").is_some_and(|value| value.is_err()));
    }
}
