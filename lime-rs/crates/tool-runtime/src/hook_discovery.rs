//! Codex-compatible Hook discovery and trust evaluation.

use crate::turn_snapshot::{
    RuntimeHookEventName, RuntimeHookExecutionMode, RuntimeHookHandlerType, RuntimeHookSnapshot,
    RuntimeHookSource, RuntimeHookTrustStatus,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

const DEFAULT_TIMEOUT_SEC: u64 = 600;
const SESSION_END_DEFAULT_TIMEOUT_SEC: u64 = 1;
const SESSION_END_MAX_TIMEOUT_SEC: u64 = 3;
const DEFAULT_ADDITIONAL_CONTEXT_LIMIT: usize = 2_500;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookPluginSource {
    pub plugin_id: String,
    pub package_root: PathBuf,
}

pub fn plugin_sources_from_activations(activations: &[serde_json::Value]) -> Vec<HookPluginSource> {
    let mut seen = HashSet::new();
    activations
        .iter()
        .filter_map(|activation| {
            let plugin_id = activation
                .get("pluginId")
                .or_else(|| activation.get("plugin_id"))
                .and_then(serde_json::Value::as_str)?
                .trim()
                .to_string();
            let source = activation
                .get("packageSourceUri")
                .or_else(|| activation.get("package_source_uri"))
                .and_then(serde_json::Value::as_str)?;
            let package_root = if source.starts_with("file:") {
                url::Url::parse(source).ok()?.to_file_path().ok()?
            } else {
                PathBuf::from(source)
            };
            if plugin_id.is_empty()
                || !package_root.is_absolute()
                || !seen.insert(plugin_id.clone())
            {
                return None;
            }
            Some(HookPluginSource {
                plugin_id,
                package_root,
            })
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookDiscoveryInput {
    pub codex_home: PathBuf,
    pub cwd: PathBuf,
    pub plugins: Vec<HookPluginSource>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookDiscoveryErrorInfo {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredHook {
    pub snapshot: RuntimeHookSnapshot,
    pub(crate) executable: bool,
}

impl DiscoveredHook {
    pub fn is_executable(&self) -> bool {
        self.executable
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HookDiscoveryReport {
    pub hooks: Vec<DiscoveredHook>,
    pub warnings: Vec<String>,
    pub errors: Vec<HookDiscoveryErrorInfo>,
}

impl HookDiscoveryReport {
    pub fn executable_hooks(&self) -> Vec<DiscoveredHook> {
        self.hooks
            .iter()
            .filter(|hook| hook.is_executable())
            .cloned()
            .collect()
    }

    pub fn snapshots(&self) -> Vec<RuntimeHookSnapshot> {
        self.hooks
            .iter()
            .map(|hook| hook.snapshot.clone())
            .collect()
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ConfigFile {
    #[serde(default)]
    hooks: HooksConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HooksConfig {
    #[serde(flatten)]
    events: HookEventsConfig,
    #[serde(default)]
    state: BTreeMap<String, HookStateConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HookStateConfig {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    trusted_hash: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct JsonHooksFile {
    #[serde(default)]
    hooks: HookEventsConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct HookEventsConfig {
    #[serde(rename = "PreToolUse", default)]
    pre_tool_use: Vec<MatcherGroup>,
    #[serde(rename = "PermissionRequest", default)]
    permission_request: Vec<MatcherGroup>,
    #[serde(rename = "PostToolUse", default)]
    post_tool_use: Vec<MatcherGroup>,
    #[serde(rename = "PreCompact", default)]
    pre_compact: Vec<MatcherGroup>,
    #[serde(rename = "PostCompact", default)]
    post_compact: Vec<MatcherGroup>,
    #[serde(rename = "SessionStart", default)]
    session_start: Vec<MatcherGroup>,
    #[serde(rename = "SessionEnd", default)]
    session_end: Vec<MatcherGroup>,
    #[serde(rename = "UserPromptSubmit", default)]
    user_prompt_submit: Vec<MatcherGroup>,
    #[serde(rename = "SubagentStart", default)]
    subagent_start: Vec<MatcherGroup>,
    #[serde(rename = "SubagentStop", default)]
    subagent_stop: Vec<MatcherGroup>,
    #[serde(rename = "Stop", default)]
    stop: Vec<MatcherGroup>,
}

impl HookEventsConfig {
    fn into_groups(self) -> [(RuntimeHookEventName, Vec<MatcherGroup>); 11] {
        [
            (RuntimeHookEventName::PreToolUse, self.pre_tool_use),
            (
                RuntimeHookEventName::PermissionRequest,
                self.permission_request,
            ),
            (RuntimeHookEventName::PostToolUse, self.post_tool_use),
            (RuntimeHookEventName::PreCompact, self.pre_compact),
            (RuntimeHookEventName::PostCompact, self.post_compact),
            (RuntimeHookEventName::SessionStart, self.session_start),
            (RuntimeHookEventName::SessionEnd, self.session_end),
            (
                RuntimeHookEventName::UserPromptSubmit,
                self.user_prompt_submit,
            ),
            (RuntimeHookEventName::SubagentStart, self.subagent_start),
            (RuntimeHookEventName::SubagentStop, self.subagent_stop),
            (RuntimeHookEventName::Stop, self.stop),
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MatcherGroup {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    matcher: Option<String>,
    #[serde(default)]
    hooks: Vec<HookHandlerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum HookHandlerConfig {
    #[serde(rename = "command")]
    Command {
        command: String,
        #[serde(
            default,
            rename = "commandWindows",
            alias = "command_windows",
            skip_serializing_if = "Option::is_none"
        )]
        command_windows: Option<String>,
        #[serde(
            default,
            rename = "timeout",
            alias = "timeout_sec",
            skip_serializing_if = "Option::is_none"
        )]
        timeout_sec: Option<u64>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        r#async: bool,
        #[serde(
            default,
            rename = "statusMessage",
            alias = "status_message",
            skip_serializing_if = "Option::is_none"
        )]
        status_message: Option<String>,
        #[serde(
            default,
            rename = "additionalContextLimit",
            alias = "additional_context_limit",
            skip_serializing_if = "Option::is_none"
        )]
        additional_context_limit: Option<usize>,
    },
    #[serde(rename = "prompt")]
    Prompt {},
    #[serde(rename = "agent")]
    Agent {},
}

struct LoadedSource {
    path: PathBuf,
    key_source: String,
    source: RuntimeHookSource,
    plugin_id: Option<String>,
    events: HookEventsConfig,
}

pub fn discover_hooks(input: &HookDiscoveryInput) -> HookDiscoveryReport {
    let mut report = HookDiscoveryReport::default();
    let user_path = input.codex_home.join("config.toml");
    let project_path = input.cwd.join(".codex/config.toml");

    let user_config = load_toml_config(&user_path, &mut report);
    let project_config = load_toml_config(&project_path, &mut report);
    let mut states = BTreeMap::new();
    if let Some(config) = user_config.as_ref() {
        states.extend(config.hooks.state.clone());
    }
    if let Some(config) = project_config.as_ref() {
        states.extend(config.hooks.state.clone());
    }

    let mut sources = Vec::new();
    if let Some(config) = user_config {
        push_toml_source(
            &mut sources,
            &user_path,
            RuntimeHookSource::User,
            config.hooks.events,
            &mut report,
        );
    }
    if let Some(config) = project_config {
        push_toml_source(
            &mut sources,
            &project_path,
            RuntimeHookSource::Project,
            config.hooks.events,
            &mut report,
        );
    }
    for plugin in &input.plugins {
        let path = plugin.package_root.join("hooks/hooks.json");
        if let Some(events) = load_json_hooks(&path, &mut report) {
            match absolute_existing_path(&path) {
                Ok(path) => sources.push(LoadedSource {
                    key_source: format!("{}:hooks/hooks.json", plugin.plugin_id),
                    path,
                    source: RuntimeHookSource::Plugin,
                    plugin_id: Some(plugin.plugin_id.clone()),
                    events,
                }),
                Err(message) => report.errors.push(HookDiscoveryErrorInfo { path, message }),
            }
        }
    }

    let mut display_order = 0_i64;
    for source in sources {
        append_source_hooks(
            source,
            &states,
            &mut display_order,
            &mut report.hooks,
            &mut report.warnings,
        );
    }
    report
}

fn load_toml_config(path: &Path, report: &mut HookDiscoveryReport) -> Option<ConfigFile> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            report.errors.push(HookDiscoveryErrorInfo {
                path: path.to_path_buf(),
                message: format!("failed to read hook config: {error}"),
            });
            return None;
        }
    };
    match toml_edit::de::from_str(&contents) {
        Ok(config) => Some(config),
        Err(error) => {
            report.errors.push(HookDiscoveryErrorInfo {
                path: path.to_path_buf(),
                message: format!("failed to parse hook config: {error}"),
            });
            None
        }
    }
}

fn load_json_hooks(path: &Path, report: &mut HookDiscoveryReport) -> Option<HookEventsConfig> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            report.errors.push(HookDiscoveryErrorInfo {
                path: path.to_path_buf(),
                message: format!("failed to read plugin hook config: {error}"),
            });
            return None;
        }
    };
    match serde_json::from_str::<JsonHooksFile>(&contents) {
        Ok(config) => Some(config.hooks),
        Err(error) => {
            report.errors.push(HookDiscoveryErrorInfo {
                path: path.to_path_buf(),
                message: format!("failed to parse plugin hook config: {error}"),
            });
            None
        }
    }
}

fn push_toml_source(
    sources: &mut Vec<LoadedSource>,
    path: &Path,
    source: RuntimeHookSource,
    events: HookEventsConfig,
    report: &mut HookDiscoveryReport,
) {
    if !path.is_file() {
        return;
    }
    match absolute_existing_path(path) {
        Ok(path) => sources.push(LoadedSource {
            key_source: path.display().to_string(),
            path,
            source,
            plugin_id: None,
            events,
        }),
        Err(message) => report.errors.push(HookDiscoveryErrorInfo {
            path: path.to_path_buf(),
            message,
        }),
    }
}

fn absolute_existing_path(path: &Path) -> Result<PathBuf, String> {
    std::fs::canonicalize(path)
        .map_err(|error| format!("failed to normalize hook config path: {error}"))
        .and_then(|path| {
            path.is_absolute()
                .then_some(path)
                .ok_or_else(|| "hook config path is not absolute".to_string())
        })
}

fn append_source_hooks(
    source: LoadedSource,
    states: &BTreeMap<String, HookStateConfig>,
    display_order: &mut i64,
    hooks: &mut Vec<DiscoveredHook>,
    warnings: &mut Vec<String>,
) {
    for (event_name, groups) in source.events.into_groups() {
        for (group_index, group) in groups.into_iter().enumerate() {
            let matcher = matcher_for_event(event_name, group.matcher.as_deref());
            if let Some(matcher) = matcher.as_deref() {
                if !valid_matcher(matcher) {
                    warnings.push(format!(
                        "invalid matcher {matcher:?} in {}",
                        source.path.display()
                    ));
                    continue;
                }
            }
            for (handler_index, handler) in group.hooks.iter().cloned().enumerate() {
                let HookHandlerConfig::Command {
                    command,
                    command_windows,
                    timeout_sec,
                    r#async,
                    status_message,
                    additional_context_limit,
                } = handler
                else {
                    warnings.push(format!(
                        "skipping unsupported prompt/agent hook in {}",
                        source.path.display()
                    ));
                    continue;
                };
                let command = selected_command(command, command_windows);
                if command.trim().is_empty() {
                    warnings.push(format!(
                        "skipping empty hook command in {}",
                        source.path.display()
                    ));
                    continue;
                }
                let timeout_sec =
                    normalize_timeout(event_name, timeout_sec, &source.path, warnings);
                let additional_context_limit = normalize_additional_context_limit(
                    event_name,
                    additional_context_limit,
                    &source.path,
                    warnings,
                );
                let normalized_limit = additional_context_limit
                    .filter(|limit| *limit != DEFAULT_ADDITIONAL_CONTEXT_LIMIT);
                let current_hash = command_hook_hash(
                    event_name,
                    matcher.as_deref(),
                    &command,
                    timeout_sec,
                    r#async,
                    status_message.as_deref(),
                    normalized_limit,
                );
                let key = format!(
                    "{}:{}:{group_index}:{handler_index}",
                    source.key_source,
                    event_name.key_label()
                );
                let state = states.get(&key);
                let is_managed = matches!(
                    source.source,
                    RuntimeHookSource::System
                        | RuntimeHookSource::Mdm
                        | RuntimeHookSource::CloudRequirements
                        | RuntimeHookSource::CloudManagedConfig
                        | RuntimeHookSource::LegacyManagedConfigFile
                        | RuntimeHookSource::LegacyManagedConfigMdm
                );
                let enabled = is_managed || state.and_then(|state| state.enabled) != Some(false);
                let trust_status = trust_status(
                    is_managed,
                    &current_hash,
                    state.and_then(|state| state.trusted_hash.as_deref()),
                );
                let executable = enabled
                    && matches!(
                        trust_status,
                        RuntimeHookTrustStatus::Managed | RuntimeHookTrustStatus::Trusted
                    );
                hooks.push(DiscoveredHook {
                    snapshot: RuntimeHookSnapshot {
                        key,
                        event_name,
                        handler_type: RuntimeHookHandlerType::Command,
                        execution_mode: if r#async {
                            RuntimeHookExecutionMode::Async
                        } else {
                            RuntimeHookExecutionMode::Sync
                        },
                        matcher: matcher.clone(),
                        command: Some(command),
                        timeout_sec,
                        status_message,
                        additional_context_limit,
                        source_path: source.path.clone(),
                        source: source.source,
                        plugin_id: source.plugin_id.clone(),
                        display_order: *display_order,
                        enabled,
                        is_managed,
                        current_hash,
                        trust_status,
                    },
                    executable,
                });
                *display_order += 1;
            }
        }
    }
}

fn selected_command(command: String, command_windows: Option<String>) -> String {
    if cfg!(windows) {
        command_windows.unwrap_or(command)
    } else {
        command
    }
}

fn matcher_for_event(event_name: RuntimeHookEventName, matcher: Option<&str>) -> Option<String> {
    match event_name {
        RuntimeHookEventName::UserPromptSubmit | RuntimeHookEventName::Stop => None,
        _ => matcher.map(ToOwned::to_owned),
    }
}

fn valid_matcher(matcher: &str) -> bool {
    matcher.is_empty()
        || matcher == "*"
        || matcher
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '|')
        || regex::Regex::new(matcher).is_ok()
}

fn normalize_timeout(
    event_name: RuntimeHookEventName,
    timeout_sec: Option<u64>,
    path: &Path,
    warnings: &mut Vec<String>,
) -> u64 {
    if event_name != RuntimeHookEventName::SessionEnd {
        return timeout_sec.unwrap_or(DEFAULT_TIMEOUT_SEC).max(1);
    }
    if timeout_sec.is_some_and(|timeout| timeout > SESSION_END_MAX_TIMEOUT_SEC) {
        warnings.push(format!(
            "clamping SessionEnd hook timeout to {SESSION_END_MAX_TIMEOUT_SEC}s in {}",
            path.display()
        ));
    }
    timeout_sec
        .unwrap_or(SESSION_END_DEFAULT_TIMEOUT_SEC)
        .clamp(1, SESSION_END_MAX_TIMEOUT_SEC)
}

fn normalize_additional_context_limit(
    event_name: RuntimeHookEventName,
    limit: Option<usize>,
    path: &Path,
    warnings: &mut Vec<String>,
) -> Option<usize> {
    if matches!(
        event_name,
        RuntimeHookEventName::PreToolUse
            | RuntimeHookEventName::PostToolUse
            | RuntimeHookEventName::SessionStart
            | RuntimeHookEventName::UserPromptSubmit
            | RuntimeHookEventName::SubagentStart
    ) {
        return limit;
    }
    if limit.is_some() {
        warnings.push(format!(
            "ignoring additionalContextLimit for {event_name:?} hook in {}",
            path.display()
        ));
    }
    None
}

fn trust_status(
    is_managed: bool,
    current_hash: &str,
    trusted_hash: Option<&str>,
) -> RuntimeHookTrustStatus {
    if is_managed {
        RuntimeHookTrustStatus::Managed
    } else {
        match trusted_hash {
            Some(trusted_hash) if trusted_hash == current_hash => RuntimeHookTrustStatus::Trusted,
            Some(_) => RuntimeHookTrustStatus::Modified,
            None => RuntimeHookTrustStatus::Untrusted,
        }
    }
}

fn command_hook_hash(
    event_name: RuntimeHookEventName,
    matcher: Option<&str>,
    command: &str,
    timeout_sec: u64,
    r#async: bool,
    status_message: Option<&str>,
    additional_context_limit: Option<usize>,
) -> String {
    let mut handler = serde_json::Map::new();
    handler.insert("type".to_string(), Value::String("command".to_string()));
    handler.insert("command".to_string(), Value::String(command.to_string()));
    handler.insert("timeout".to_string(), Value::from(timeout_sec));
    if r#async {
        handler.insert("async".to_string(), Value::Bool(true));
    }
    if let Some(status_message) = status_message {
        handler.insert(
            "statusMessage".to_string(),
            Value::String(status_message.to_string()),
        );
    }
    if let Some(limit) = additional_context_limit {
        handler.insert("additionalContextLimit".to_string(), Value::from(limit));
    }
    let mut identity = serde_json::Map::new();
    identity.insert(
        "event_name".to_string(),
        Value::String(event_name.key_label().to_string()),
    );
    if let Some(matcher) = matcher {
        identity.insert("matcher".to_string(), Value::String(matcher.to_string()));
    }
    identity.insert(
        "hooks".to_string(),
        Value::Array(vec![Value::Object(handler)]),
    );
    let canonical = canonical_json(&Value::Object(identity));
    let serialized = serde_json::to_vec(&canonical).unwrap_or_default();
    format!("sha256:{}", hex::encode(Sha256::digest(serialized)))
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = serde_json::Map::new();
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                if let Some(value) = map.get(key) {
                    sorted.insert(key.clone(), canonical_json(value));
                }
            }
            Value::Object(sorted)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn setup() -> (tempfile::TempDir, tempfile::TempDir) {
        (
            tempfile::TempDir::new().expect("codex home"),
            tempfile::TempDir::new().expect("cwd"),
        )
    }

    #[test]
    fn discovers_user_toml_with_exact_metadata_and_untrusted_default() {
        let (home, cwd) = setup();
        fs::write(
            home.path().join("config.toml"),
            r#"[hooks]

[[hooks.PreToolUse]]
matcher = "Bash"

[[hooks.PreToolUse.hooks]]
type = "command"
command = "python3 /tmp/listed-hook.py"
timeout = 5
statusMessage = "checking"
additionalContextLimit = 4096
"#,
        )
        .expect("write config");

        let report = discover_hooks(&HookDiscoveryInput {
            codex_home: home.path().to_path_buf(),
            cwd: cwd.path().to_path_buf(),
            plugins: Vec::new(),
        });

        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.hooks.len(), 1);
        let hook = &report.hooks[0];
        assert_eq!(hook.snapshot.event_name, RuntimeHookEventName::PreToolUse);
        assert_eq!(hook.snapshot.matcher.as_deref(), Some("Bash"));
        assert_eq!(hook.snapshot.timeout_sec, 5);
        assert_eq!(hook.snapshot.additional_context_limit, Some(4096));
        assert_eq!(
            hook.snapshot.trust_status,
            RuntimeHookTrustStatus::Untrusted
        );
        assert!(!hook.is_executable());
        assert!(hook.snapshot.current_hash.starts_with("sha256:"));
    }

    #[test]
    fn discovers_async_command_with_async_execution_mode() {
        let (home, cwd) = setup();
        fs::write(
            home.path().join("config.toml"),
            r#"[hooks]
[[hooks.PostToolUse]]
[[hooks.PostToolUse.hooks]]
type = "command"
command = "echo background"
async = true
"#,
        )
        .expect("write config");

        let report = discover_hooks(&HookDiscoveryInput {
            codex_home: home.path().to_path_buf(),
            cwd: cwd.path().to_path_buf(),
            plugins: Vec::new(),
        });

        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.hooks.len(), 1);
        assert_eq!(
            report.hooks[0].snapshot.execution_mode,
            RuntimeHookExecutionMode::Async
        );
    }

    #[test]
    fn trusted_hash_enables_same_normalized_definition_and_changes_fail_closed() {
        let (home, cwd) = setup();
        let config_path = home.path().join("config.toml");
        fs::write(
            &config_path,
            r#"[hooks]
[[hooks.PreToolUse]]
matcher = "Bash"
[[hooks.PreToolUse.hooks]]
type = "command"
command = "echo trusted"
"#,
        )
        .expect("write config");
        let input = HookDiscoveryInput {
            codex_home: home.path().to_path_buf(),
            cwd: cwd.path().to_path_buf(),
            plugins: Vec::new(),
        };
        let first = discover_hooks(&input);
        let hook = &first.hooks[0].snapshot;
        let key = hook.key.clone();
        let hash = hook.current_hash.clone();
        fs::write(
            &config_path,
            format!(
                r#"[hooks]
[[hooks.PreToolUse]]
matcher = "Bash"
[[hooks.PreToolUse.hooks]]
type = "command"
command = "echo trusted"

[hooks.state."{key}"]
enabled = true
trusted_hash = "{hash}"
"#
            ),
        )
        .expect("trust config");
        let trusted = discover_hooks(&input);
        assert_eq!(
            trusted.hooks[0].snapshot.trust_status,
            RuntimeHookTrustStatus::Trusted
        );
        assert!(trusted.hooks[0].is_executable());

        let changed = fs::read_to_string(&config_path)
            .expect("read config")
            .replace("echo trusted", "echo modified");
        fs::write(&config_path, changed).expect("modify config");
        let modified = discover_hooks(&input);
        assert_eq!(
            modified.hooks[0].snapshot.trust_status,
            RuntimeHookTrustStatus::Modified
        );
        assert!(!modified.hooks[0].is_executable());
    }

    #[test]
    fn discovers_active_plugin_hooks_with_plugin_stable_key() {
        let (home, cwd) = setup();
        let plugin_root = home.path().join("plugins/demo");
        fs::create_dir_all(plugin_root.join("hooks")).expect("hooks dir");
        fs::write(
            plugin_root.join("hooks/hooks.json"),
            r#"{"hooks":{"PreToolUse":[{"matcher":"Bash","hooks":[{"type":"command","command":"echo plugin","timeout":7}]}]}}"#,
        )
        .expect("plugin hooks");

        let report = discover_hooks(&HookDiscoveryInput {
            codex_home: home.path().to_path_buf(),
            cwd: cwd.path().to_path_buf(),
            plugins: vec![HookPluginSource {
                plugin_id: "demo@test".to_string(),
                package_root: plugin_root,
            }],
        });

        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(
            report.hooks[0].snapshot.key,
            "demo@test:hooks/hooks.json:pre_tool_use:0:0"
        );
        assert_eq!(
            report.hooks[0].snapshot.plugin_id.as_deref(),
            Some("demo@test")
        );
        assert_eq!(report.hooks[0].snapshot.source, RuntimeHookSource::Plugin);
    }

    #[test]
    fn session_end_timeout_is_clamped_and_standard_default_is_ten_minutes() {
        let (home, cwd) = setup();
        fs::write(
            home.path().join("config.toml"),
            r#"[hooks]
[[hooks.PreToolUse]]
[[hooks.PreToolUse.hooks]]
type = "command"
command = "echo pre"
[[hooks.SessionEnd]]
[[hooks.SessionEnd.hooks]]
type = "command"
command = "echo end"
timeout = 30
"#,
        )
        .expect("config");
        let report = discover_hooks(&HookDiscoveryInput {
            codex_home: home.path().to_path_buf(),
            cwd: cwd.path().to_path_buf(),
            plugins: Vec::new(),
        });
        assert_eq!(report.hooks[0].snapshot.timeout_sec, 600);
        assert_eq!(report.hooks[1].snapshot.timeout_sec, 3);
        assert_eq!(report.warnings.len(), 1);
    }
}
