use serde_json::{json, Value};
use std::collections::HashMap;

pub const TOOL_EXECUTION_ACTION_KIND: &str = "tool_execution_policy";
pub const BROWSER_ACTION_CONTRACT_KEY: &str = "browser_action";
pub const BROWSER_ACTION_TOOL_FAMILY: &str = "browser_action";
pub const SHELL_COMMAND_CONTRACT_KEY: &str = "shell_command";
pub const SHELL_TOOL_FAMILY: &str = "shell_command";
const DEFAULT_TOOL_APPROVAL_DECISIONS: [&str; 3] = ["allow_once", "decline", "cancel"];

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionApprovalProjection {
    pub action_kind: String,
    pub tool_family: String,
    pub contract_key: String,
    pub runtime_contract: Value,
    pub approval_scope: Value,
    pub available_decisions: Vec<String>,
}

pub fn execution_approval_projection(
    tool_name: &str,
    metadata: &HashMap<String, Value>,
) -> ExecutionApprovalProjection {
    let action_kind = metadata_string(metadata, &["actionKind", "action_kind"])
        .unwrap_or_else(|| TOOL_EXECUTION_ACTION_KIND.to_string());
    let tool_family = tool_family_for_approval(tool_name, metadata);
    let contract_key = runtime_contract_key_for_approval(&tool_family, metadata);
    let runtime_contract = runtime_contract_for_approval(&contract_key, &tool_family, metadata);
    let approval_scope = approval_scope_for_approval(&contract_key, &tool_family, metadata);
    let available_decisions = available_decisions_for_approval(metadata, &runtime_contract);

    ExecutionApprovalProjection {
        action_kind,
        tool_family,
        contract_key,
        runtime_contract,
        approval_scope,
        available_decisions,
    }
}

fn tool_family_for_approval(tool_name: &str, metadata: &HashMap<String, Value>) -> String {
    metadata_string(metadata, &["toolFamily", "tool_family"])
        .unwrap_or_else(|| infer_tool_family(tool_name))
}

fn infer_tool_family(tool_name: &str) -> String {
    match tool_name.trim().to_ascii_lowercase().as_str() {
        "exec_command" => SHELL_TOOL_FAMILY.to_string(),
        value if value.is_empty() => "unknown".to_string(),
        value => value.to_string(),
    }
}

fn runtime_contract_key_for_approval(
    tool_family: &str,
    metadata: &HashMap<String, Value>,
) -> String {
    metadata_string(metadata, &["contractKey", "contract_key"])
        .or_else(|| {
            metadata.get("runtime_contract").and_then(|contract| {
                metadata_string_from_value(contract, &["contractKey", "contract_key"])
            })
        })
        .unwrap_or_else(|| {
            if tool_family == SHELL_TOOL_FAMILY {
                SHELL_COMMAND_CONTRACT_KEY.to_string()
            } else {
                tool_family.to_string()
            }
        })
}

fn runtime_contract_for_approval(
    contract_key: &str,
    tool_family: &str,
    metadata: &HashMap<String, Value>,
) -> Value {
    metadata
        .get("runtime_contract")
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "contract_key": contract_key,
                "tool_family": tool_family,
                "session_cache_supported": false,
            })
        })
}

fn approval_scope_for_approval(
    contract_key: &str,
    tool_family: &str,
    metadata: &HashMap<String, Value>,
) -> Value {
    if let Some(scope) = metadata
        .get("approvalScope")
        .or_else(|| metadata.get("approval_scope"))
        .cloned()
    {
        return scope;
    }

    let working_dir_hash = hashed_scope_field(
        metadata,
        &["workingDir", "working_dir", "workingDirectory", "cwd"],
    );
    let project_root_hash = hashed_scope_field(metadata, &["projectRoot", "project_root"]);
    let risk_class = metadata_string(
        metadata,
        &[
            "riskClass",
            "risk_class",
            "commandRiskLevel",
            "networkRiskLevel",
            "reasonCode",
        ],
    );
    let network_context = sanitized_network_context(metadata);
    let network_host = network_context.as_ref().map(|context| context.host.clone());
    let network_protocol = network_context.map(|context| context.protocol);

    json!({
        "contractKey": contract_key,
        "contract_key": contract_key,
        "toolFamily": tool_family,
        "tool_family": tool_family,
        "riskClass": risk_class.clone(),
        "risk_class": risk_class,
        "workingDirHash": working_dir_hash.clone(),
        "working_dir_hash": working_dir_hash,
        "projectRootHash": project_root_hash.clone(),
        "project_root_hash": project_root_hash,
        "networkHost": network_host.clone(),
        "network_host": network_host,
        "networkProtocol": network_protocol.clone(),
        "network_protocol": network_protocol,
    })
}

fn available_decisions_for_approval(
    metadata: &HashMap<String, Value>,
    runtime_contract: &Value,
) -> Vec<String> {
    let session_cache_supported = runtime_contract
        .get("session_cache_supported")
        .and_then(Value::as_bool)
        == Some(true);
    if let Some(values) = metadata
        .get("availableDecisions")
        .or_else(|| metadata.get("available_decisions"))
        .and_then(Value::as_array)
    {
        let mut decisions = Vec::new();
        for decision in values.iter().filter_map(Value::as_str) {
            if matches!(
                decision,
                "allow_once" | "allow_for_session" | "decline" | "cancel"
            ) && (decision != "allow_for_session" || session_cache_supported)
                && !decisions.iter().any(|existing| existing == decision)
            {
                decisions.push(decision.to_string());
            }
        }
        if !decisions.is_empty() {
            return decisions;
        }
    }

    DEFAULT_TOOL_APPROVAL_DECISIONS
        .iter()
        .map(|decision| (*decision).to_string())
        .collect()
}

fn hashed_scope_field(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<String> {
    metadata_string(metadata, keys).map(|value| stable_scope_hash(&value))
}

fn stable_scope_hash(value: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in value.trim().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

struct SanitizedNetworkContext {
    host: String,
    protocol: String,
}

fn sanitized_network_context(metadata: &HashMap<String, Value>) -> Option<SanitizedNetworkContext> {
    let parsed = metadata_string(metadata, &["networkUrl", "network_url"])
        .as_deref()
        .and_then(sanitized_network_context_from_url);
    let host = metadata_string(metadata, &["networkHost", "network_host"])
        .or_else(|| parsed.as_ref().map(|context| context.host.clone()))?;
    let protocol = metadata_string(metadata, &["networkProtocol", "network_protocol"])
        .and_then(|protocol| normalized_network_protocol(&protocol))
        .or_else(|| parsed.map(|context| context.protocol))?;
    Some(SanitizedNetworkContext { host, protocol })
}

fn sanitized_network_context_from_url(value: &str) -> Option<SanitizedNetworkContext> {
    let parsed = url::Url::parse(value).ok()?;
    let host = parsed.host_str()?.trim();
    if host.is_empty() {
        return None;
    }
    Some(SanitizedNetworkContext {
        host: host.to_string(),
        protocol: normalized_network_protocol(parsed.scheme())?,
    })
}

fn normalized_network_protocol(value: &str) -> Option<String> {
    match value
        .trim()
        .trim_end_matches(':')
        .to_ascii_lowercase()
        .as_str()
    {
        "http" => Some("http".to_string()),
        "https" => Some("https".to_string()),
        "socks5-tcp" | "socks5_tcp" | "socks5tcp" => Some("socks5Tcp".to_string()),
        "socks5-udp" | "socks5_udp" | "socks5udp" => Some("socks5Udp".to_string()),
        _ => None,
    }
}

fn metadata_string(metadata: &HashMap<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| metadata.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn metadata_string_from_value(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_approval_projection_uses_once_only_decisions_by_default() {
        let metadata = HashMap::from([
            (
                "reasonCode".to_string(),
                json!("shell_command_requires_approval"),
            ),
            ("cwd".to_string(), json!("/Users/coso/project")),
            (
                "networkUrl".to_string(),
                json!("https://example.com/path?token=secret"),
            ),
        ]);

        let projection = execution_approval_projection("exec_command", &metadata);

        assert_eq!(projection.action_kind, TOOL_EXECUTION_ACTION_KIND);
        assert_eq!(projection.tool_family, SHELL_TOOL_FAMILY);
        assert_eq!(projection.contract_key, SHELL_COMMAND_CONTRACT_KEY);
        assert_eq!(
            projection.available_decisions,
            vec!["allow_once", "decline", "cancel"]
        );
        assert_eq!(
            projection.runtime_contract,
            json!({
                "contract_key": SHELL_COMMAND_CONTRACT_KEY,
                "tool_family": SHELL_TOOL_FAMILY,
                "session_cache_supported": false,
            })
        );
        assert_eq!(
            projection.approval_scope.get("riskClass"),
            Some(&json!("shell_command_requires_approval"))
        );
        assert_eq!(
            projection.approval_scope.get("networkHost"),
            Some(&json!("example.com"))
        );
        assert_eq!(
            projection.approval_scope.get("networkProtocol"),
            Some(&json!("https"))
        );
        assert!(projection.approval_scope.get("cwd").is_none());
        assert!(!projection
            .approval_scope
            .to_string()
            .contains("/Users/coso/project"));
        assert!(!projection
            .approval_scope
            .to_string()
            .contains("token=secret"));
    }

    #[test]
    fn explicit_session_cache_contract_must_be_declared_by_current_owner() {
        let metadata = HashMap::from([
            (
                "availableDecisions".to_string(),
                json!(["allow_once", "allow_for_session", "decline", "cancel"]),
            ),
            (
                "runtime_contract".to_string(),
                json!({
                    "contract_key": "browser_control",
                    "tool_family": "browser_control",
                    "session_cache_supported": true,
                }),
            ),
            (
                "approvalScope".to_string(),
                json!({
                    "contractKey": "browser_control",
                    "networkHost": "https://example.com",
                }),
            ),
        ]);

        let projection = execution_approval_projection("browser_control", &metadata);

        assert_eq!(projection.tool_family, "browser_control");
        assert_eq!(projection.contract_key, "browser_control");
        assert_eq!(
            projection.available_decisions,
            vec!["allow_once", "allow_for_session", "decline", "cancel"]
        );
        assert_eq!(
            projection.runtime_contract.get("session_cache_supported"),
            Some(&json!(true))
        );
        assert_eq!(
            projection.approval_scope.get("networkHost"),
            Some(&json!("https://example.com"))
        );
    }

    #[test]
    fn session_decision_fails_closed_without_boolean_cache_support() {
        for runtime_contract in [
            json!({
                "contract_key": "shell_command",
                "session_cache_supported": false,
            }),
            json!({
                "contract_key": "shell_command",
                "session_cache_supported": "true",
            }),
            json!({ "contract_key": "shell_command" }),
            json!("invalid"),
        ] {
            let metadata = HashMap::from([
                (
                    "availableDecisions".to_string(),
                    json!(["allow_once", "allow_for_session", "decline", "cancel"]),
                ),
                ("runtime_contract".to_string(), runtime_contract),
            ]);

            let projection = execution_approval_projection("exec_command", &metadata);

            assert_eq!(
                projection.available_decisions,
                vec!["allow_once", "decline", "cancel"]
            );
        }
    }

    #[test]
    fn browser_action_approval_is_once_only() {
        let metadata = HashMap::from([
            ("toolFamily".to_string(), json!(BROWSER_ACTION_TOOL_FAMILY)),
            (
                "contractKey".to_string(),
                json!(BROWSER_ACTION_CONTRACT_KEY),
            ),
            (
                "availableDecisions".to_string(),
                json!(["allow_once", "allow_for_session", "decline", "cancel"]),
            ),
            (
                "runtime_contract".to_string(),
                json!({
                    "contract_key": BROWSER_ACTION_CONTRACT_KEY,
                    "tool_family": BROWSER_ACTION_TOOL_FAMILY,
                    "session_cache_supported": false,
                }),
            ),
        ]);

        let projection = execution_approval_projection("browser__click", &metadata);

        assert_eq!(projection.tool_family, BROWSER_ACTION_TOOL_FAMILY);
        assert_eq!(projection.contract_key, BROWSER_ACTION_CONTRACT_KEY);
        assert_eq!(
            projection.available_decisions,
            vec!["allow_once", "decline", "cancel"]
        );
    }
}
