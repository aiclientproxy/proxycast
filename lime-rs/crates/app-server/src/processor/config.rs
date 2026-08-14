//! Process-wide Desktop configuration control plane.

use super::{dispatch_result, parse_params, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2::{
    ConfigBatchWriteParams, ConfigEdit, ConfigLayer, ConfigLayerMetadata, ConfigLayerSource,
    ConfigReadParams, ConfigReadResponse, ConfigValueWriteParams, ConfigWriteErrorCode,
    ConfigWriteResponse, MergeStrategy, WriteStatus,
};
use app_server_protocol::{error_codes, JsonRpcError};
use lime_core::config::{load_config, save_config, Config, ConfigManager};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

const CONFIG_ROOT_KEYS: &[&str] = &[
    "server",
    "providers",
    "default_provider",
    "routing",
    "retry",
    "logging",
    "injection",
    "auth_dir",
    "remote_management",
    "quota_exceeded",
    "proxy_url",
    "ampcode",
    "endpoint_providers",
    "minimize_to_tray",
    "language",
    "models",
    "agent",
    "skills",
    "orchestrator",
    "experimental",
    "tool_calling",
    "workspace_preferences",
    "navigation",
    "chat_appearance",
    "environment",
    "web_search",
    "memory",
    "image_gen",
    "user_profile",
    "developer",
    "rate_limit",
    "crash_reporting",
    "conversation",
    "hint_router",
    "pairing",
    "automation",
    "gateway",
    "channels",
];

pub(super) fn config_lock() -> Result<MutexGuard<'static, ()>, JsonRpcError> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| config_runtime_error("config lock poisoned"))
}

impl RequestProcessor {
    pub(super) async fn handle_config_read_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ConfigReadParams = parse_params(params)?;
        if params.cwd.is_some() {
            return Err(JsonRpcError::new(
                error_codes::INVALID_PARAMS,
                "Lime Desktop config/read does not support project config layers",
            ));
        }

        let _guard = config_lock()?;
        let snapshot = read_snapshot()?;
        let metadata = snapshot.metadata();
        let mut origins = BTreeMap::new();
        collect_origins(&snapshot.config, "", &metadata, &mut origins);
        let layers = params.include_layers.then(|| {
            vec![ConfigLayer {
                name: metadata.name.clone(),
                version: snapshot.version.clone(),
                config: snapshot.config.clone(),
                disabled_reason: None,
            }]
        });

        dispatch_result(ConfigReadResponse {
            config: snapshot.config,
            origins,
            layers,
        })
    }

    pub(super) async fn handle_config_value_write_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ConfigValueWriteParams = parse_params(params)?;
        write_edits(
            params.file_path,
            params.expected_version,
            vec![ConfigEdit {
                key_path: params.key_path,
                value: params.value,
                merge_strategy: params.merge_strategy,
            }],
        )
        .and_then(dispatch_result)
    }

    pub(super) async fn handle_config_batch_write_impl(
        &self,
        params: Option<Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ConfigBatchWriteParams = parse_params(params)?;
        // Runtime consumers resolve config.yaml at each operation, so there is no
        // thread-local config cache to refresh when this flag is true.
        write_edits(params.file_path, params.expected_version, params.edits)
            .and_then(dispatch_result)
    }
}

struct ConfigSnapshot {
    config: Value,
    version: String,
    file_path: PathBuf,
}

impl ConfigSnapshot {
    fn metadata(&self) -> ConfigLayerMetadata {
        ConfigLayerMetadata {
            name: ConfigLayerSource::User {
                file: self.file_path.to_string_lossy().into_owned(),
                profile: None,
            },
            version: self.version.clone(),
        }
    }
}

fn read_snapshot() -> Result<ConfigSnapshot, JsonRpcError> {
    let config = load_config().map_err(config_runtime_error)?;
    snapshot_from_config(config)
}

fn snapshot_from_config(config: Config) -> Result<ConfigSnapshot, JsonRpcError> {
    let config = serde_json::to_value(config).map_err(config_runtime_error)?;
    let bytes = serde_json::to_vec(&config).map_err(config_runtime_error)?;
    let version = hex::encode(Sha256::digest(bytes));
    let file_path = ConfigManager::default_config_path();
    Ok(ConfigSnapshot {
        config,
        version,
        file_path,
    })
}

fn write_edits(
    file_path: Option<String>,
    expected_version: Option<String>,
    edits: Vec<ConfigEdit>,
) -> Result<ConfigWriteResponse, JsonRpcError> {
    let _guard = config_lock()?;
    let mut snapshot = read_snapshot()?;
    validate_write_path(file_path.as_deref(), &snapshot.file_path)?;
    if expected_version
        .as_deref()
        .is_some_and(|expected| expected != snapshot.version)
    {
        return Err(config_write_error(
            ConfigWriteErrorCode::ConfigVersionConflict,
            "Configuration was modified since last read. Fetch latest version and retry.",
        ));
    }

    for edit in edits {
        let segments = parse_key_path(&edit.key_path).map_err(|message| {
            config_write_error(ConfigWriteErrorCode::ConfigValidationError, message)
        })?;
        if !CONFIG_ROOT_KEYS.contains(&segments[0].as_str()) {
            return Err(config_write_error(
                ConfigWriteErrorCode::ConfigSchemaUnknownKey,
                format!("unknown Lime config key: {}", segments[0]),
            ));
        }
        apply_edit(
            &mut snapshot.config,
            &segments,
            edit.value,
            edit.merge_strategy,
        )?;
    }

    let config: Config = serde_json::from_value(snapshot.config).map_err(|error| {
        config_write_error(
            ConfigWriteErrorCode::ConfigValidationError,
            format!("Invalid configuration: {error}"),
        )
    })?;
    save_config(&config).map_err(config_runtime_error)?;
    let snapshot = snapshot_from_config(config)?;

    Ok(ConfigWriteResponse {
        status: WriteStatus::Ok,
        version: snapshot.version,
        file_path: snapshot.file_path.to_string_lossy().into_owned(),
        overridden_metadata: None,
    })
}

fn validate_write_path(provided: Option<&str>, current: &Path) -> Result<(), JsonRpcError> {
    let Some(provided) = provided else {
        return Ok(());
    };
    let provided = PathBuf::from(provided);
    if !provided.is_absolute() {
        return Err(config_write_error(
            ConfigWriteErrorCode::ConfigPathNotFound,
            "config filePath must be absolute",
        ));
    }
    if provided != current {
        return Err(config_write_error(
            ConfigWriteErrorCode::ConfigLayerReadonly,
            "Only writes to the Lime Desktop user config are allowed",
        ));
    }
    Ok(())
}

fn apply_edit(
    root: &mut Value,
    segments: &[String],
    value: Value,
    strategy: MergeStrategy,
) -> Result<(), JsonRpcError> {
    let Some((last, parents)) = segments.split_last() else {
        return Err(config_write_error(
            ConfigWriteErrorCode::ConfigValidationError,
            "keyPath must not be empty",
        ));
    };
    let mut current = root;
    for segment in parents {
        if !current.is_object() {
            *current = Value::Object(Map::new());
        }
        current = current
            .as_object_mut()
            .expect("object ensured above")
            .entry(segment.clone())
            .or_insert_with(|| Value::Object(Map::new()));
    }
    if !current.is_object() {
        *current = Value::Object(Map::new());
    }
    let object = current.as_object_mut().expect("object ensured above");
    if value.is_null() {
        object.remove(last);
        return Ok(());
    }
    if matches!(strategy, MergeStrategy::Upsert) && value.is_object() {
        if let Some(existing) = object.get_mut(last) {
            if existing.is_object() {
                merge_objects(existing, value);
                return Ok(());
            }
        }
    }
    object.insert(last.clone(), value);
    Ok(())
}

fn merge_objects(current: &mut Value, overlay: Value) {
    let current = current.as_object_mut().expect("object checked by caller");
    for (key, value) in overlay.as_object().expect("object checked by caller") {
        match current.get_mut(key) {
            Some(existing) if existing.is_object() && value.is_object() => {
                merge_objects(existing, value.clone());
            }
            _ => {
                current.insert(key.clone(), value.clone());
            }
        }
    }
}

fn parse_key_path(path: &str) -> Result<Vec<String>, String> {
    if path.trim().is_empty() {
        return Err("keyPath must not be empty".to_string());
    }
    let mut segments = Vec::new();
    let mut segment = String::new();
    let mut chars = path.chars();
    let mut quoted = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if segment.is_empty() && !quoted => quoted = true,
            '"' if quoted => quoted = false,
            '\\' if quoted => {
                let Some(escaped) = chars.next() else {
                    return Err("unterminated escape in keyPath".to_string());
                };
                segment.push(escaped);
            }
            '.' if !quoted => {
                if segment.is_empty() {
                    return Err("keyPath segments must not be empty".to_string());
                }
                segments.push(std::mem::take(&mut segment));
            }
            '"' => return Err("invalid quoted keyPath segment".to_string()),
            _ => segment.push(ch),
        }
    }
    if quoted {
        return Err("unterminated quoted keyPath segment".to_string());
    }
    if segment.is_empty() {
        return Err("keyPath segments must not be empty".to_string());
    }
    segments.push(segment);
    Ok(segments)
}

fn collect_origins(
    value: &Value,
    path: &str,
    metadata: &ConfigLayerMetadata,
    origins: &mut BTreeMap<String, ConfigLayerMetadata>,
) {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                collect_origins(child, &child_path, metadata, origins);
            }
        }
        Value::Array(array) => {
            for (index, child) in array.iter().enumerate() {
                let child_path = format!("{path}.{index}");
                collect_origins(child, &child_path, metadata, origins);
            }
        }
        _ if !path.is_empty() => {
            origins.insert(path.to_string(), metadata.clone());
        }
        _ => {}
    }
}

fn config_runtime_error(error: impl std::fmt::Display) -> JsonRpcError {
    JsonRpcError::new(
        error_codes::RUNTIME_ERROR,
        format!("failed to access Lime config: {error}"),
    )
}

fn config_write_error(code: ConfigWriteErrorCode, message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::with_data(
        error_codes::INVALID_REQUEST,
        message,
        json!({"config_write_error_code": code}),
    )
    .expect("config write error data is serializable")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orchestrator_config_is_writable_through_current_control_plane() {
        assert!(CONFIG_ROOT_KEYS.contains(&"orchestrator"));

        let mut config = json!({});
        apply_edit(
            &mut config,
            &parse_key_path("orchestrator.mcp.enabled").expect("valid key path"),
            json!(false),
            MergeStrategy::Replace,
        )
        .expect("orchestrator config edit");

        assert_eq!(
            config.pointer("/orchestrator/mcp/enabled"),
            Some(&json!(false))
        );
    }
}
