//! App Server config warning notifications.

use super::{RequestProcessor, RpcDispatch};
#[cfg(test)]
use crate::RuntimeCore;
use app_server_protocol::protocol::v2::{ConfigWarningNotification, ServerNotification};
use app_server_protocol::{JsonRpcError, JsonRpcNotification};
use lime_core::config::{Config, ConfigManager};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) type ConfigWarningProvider =
    Arc<dyn Fn(ConfigWarningScope) -> Vec<JsonRpcNotification> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigWarningScope {
    Initialize,
    TurnStart,
}

pub(super) fn default_config_warning_provider() -> ConfigWarningProvider {
    Arc::new(current_config_warning_notifications)
}

impl RequestProcessor {
    pub(super) fn handle_initialize(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        let dispatch = RpcDispatch::single(self.initialize(params)?);
        Ok(dispatch
            .with_notifications(self.config_warning_notifications(ConfigWarningScope::Initialize)))
    }

    pub(super) fn handle_transport_initialize(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        let dispatch = RpcDispatch::single(self.initialize_transport(params)?);
        Ok(dispatch
            .with_notifications(self.config_warning_notifications(ConfigWarningScope::Initialize)))
    }

    pub(super) fn config_warning_notifications(
        &self,
        scope: ConfigWarningScope,
    ) -> Vec<JsonRpcNotification> {
        (self.config_warning_provider)(scope)
    }

    #[cfg(test)]
    pub(crate) fn new_with_config_warning_provider(
        runtime: RuntimeCore,
        config_warning_provider: ConfigWarningProvider,
    ) -> Self {
        let mut processor = Self::new(runtime);
        processor.config_warning_provider = config_warning_provider;
        processor
    }
}

pub(super) fn current_config_warning_notifications(
    scope: ConfigWarningScope,
) -> Vec<JsonRpcNotification> {
    let yaml_path = ConfigManager::default_config_path();
    let json_path = legacy_json_config_path(&yaml_path);
    config_warning_notifications_from_paths(scope, &yaml_path, &json_path)
}

pub(super) fn config_warning_notifications_from_paths(
    scope: ConfigWarningScope,
    yaml_path: &Path,
    json_path: &Path,
) -> Vec<JsonRpcNotification> {
    if yaml_path.exists() {
        return read_yaml_config_warning(scope, yaml_path);
    }
    if json_path.exists() {
        return read_json_config_warning(scope, json_path);
    }
    Vec::new()
}

fn read_yaml_config_warning(scope: ConfigWarningScope, path: &Path) -> Vec<JsonRpcNotification> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) => {
            return vec![config_warning_notification(scope, path, error.to_string())];
        }
    };
    match ConfigManager::parse_yaml(&content) {
        Ok(_) => Vec::new(),
        Err(error) => vec![config_warning_notification(scope, path, error.to_string())],
    }
}

fn read_json_config_warning(scope: ConfigWarningScope, path: &Path) -> Vec<JsonRpcNotification> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) => {
            return vec![config_warning_notification(scope, path, error.to_string())];
        }
    };
    match serde_json::from_str::<Config>(&content) {
        Ok(_) => Vec::new(),
        Err(error) => vec![config_warning_notification(scope, path, error.to_string())],
    }
}

fn config_warning_notification(
    scope: ConfigWarningScope,
    path: &Path,
    details: String,
) -> JsonRpcNotification {
    ServerNotification::ConfigWarning(ConfigWarningNotification {
        summary: config_warning_summary(scope).to_string(),
        details: Some(details),
        path: Some(path.display().to_string()),
        range: None,
    })
    .into()
}

fn config_warning_summary(scope: ConfigWarningScope) -> &'static str {
    match scope {
        ConfigWarningScope::Initialize => "App Server config warning during initialize",
        ConfigWarningScope::TurnStart => "App Server config warning during turn start",
    }
}

fn legacy_json_config_path(yaml_path: &Path) -> PathBuf {
    yaml_path.with_file_name("config.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::protocol::v2::METHOD_CONFIG_WARNING;
    use serde_json::json;

    #[test]
    fn config_warning_ignores_missing_config_files() {
        let temp = tempfile::tempdir().expect("temp dir");
        let yaml_path = temp.path().join("config.yaml");
        let json_path = temp.path().join("config.json");

        let notifications = config_warning_notifications_from_paths(
            ConfigWarningScope::Initialize,
            &yaml_path,
            &json_path,
        );

        assert!(notifications.is_empty());
    }

    #[test]
    fn config_warning_reports_yaml_parse_error() {
        let temp = tempfile::tempdir().expect("temp dir");
        let yaml_path = temp.path().join("config.yaml");
        let json_path = temp.path().join("config.json");
        std::fs::write(&yaml_path, "server: [").expect("write invalid yaml");

        let notifications = config_warning_notifications_from_paths(
            ConfigWarningScope::TurnStart,
            &yaml_path,
            &json_path,
        );
        let notification = notifications.first().expect("warning notification");

        assert_eq!(notifications.len(), 1);
        assert_eq!(notification.method, METHOD_CONFIG_WARNING);
        let params = notification.params.as_ref().expect("params");
        assert_eq!(
            params["summary"],
            json!("App Server config warning during turn start")
        );
        assert_eq!(params["path"], json!(yaml_path.display().to_string()));
        assert!(params["details"]
            .as_str()
            .expect("details")
            .contains("YAML"));
    }

    #[test]
    fn config_warning_reports_legacy_json_parse_error_when_yaml_is_absent() {
        let temp = tempfile::tempdir().expect("temp dir");
        let yaml_path = temp.path().join("config.yaml");
        let json_path = temp.path().join("config.json");
        std::fs::write(&json_path, "{").expect("write invalid json");

        let notifications = config_warning_notifications_from_paths(
            ConfigWarningScope::Initialize,
            &yaml_path,
            &json_path,
        );
        let notification = notifications.first().expect("warning notification");

        assert_eq!(notifications.len(), 1);
        assert_eq!(notification.method, METHOD_CONFIG_WARNING);
        let params = notification.params.as_ref().expect("params");
        assert_eq!(
            params["summary"],
            json!("App Server config warning during initialize")
        );
        assert_eq!(params["path"], json!(json_path.display().to_string()));
        assert!(params["details"].as_str().expect("details").contains("EOF"));
    }
}
