use super::super::{metadata_string, raw_string_field};
use app_server_protocol::AgentSession;
use serde_json::json;

pub(super) fn session_archived_at(session: &AgentSession) -> Option<String> {
    session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref())
        .and_then(|metadata| raw_string_field(metadata, &["archivedAt", "archived_at"]))
}

pub(super) fn session_execution_strategy(session: &AgentSession) -> Option<String> {
    session.business_object_ref.as_ref().and_then(|reference| {
        metadata_string(reference.metadata.as_ref(), "executionStrategy")
            .or_else(|| metadata_string(reference.metadata.as_ref(), "execution_strategy"))
    })
}

pub(super) fn session_working_dir(session: &AgentSession) -> Option<String> {
    session.business_object_ref.as_ref().and_then(|reference| {
        metadata_string(reference.metadata.as_ref(), "workingDir")
            .or_else(|| metadata_string(reference.metadata.as_ref(), "working_dir"))
    })
}

pub(super) fn session_execution_runtime(session: &AgentSession) -> serde_json::Value {
    let metadata = session
        .business_object_ref
        .as_ref()
        .and_then(|reference| reference.metadata.as_ref());
    let runtime = json!({
        "session_id": session.session_id,
        "provider_selector": metadata_string_alias(metadata, &["providerSelector", "provider_selector"]),
        "provider_name": metadata_string_alias(metadata, &["providerName", "provider_name"]),
        "model_name": metadata_string_alias(metadata, &["modelName", "model_name", "model"]),
        "cwd": metadata_string_alias(metadata, &["cwd", "workingDir", "working_dir"]),
        "working_dir": metadata_string_alias(metadata, &["workingDir", "working_dir", "cwd"]),
        "reasoning_effort": metadata_string_alias(metadata, &["reasoningEffort", "reasoning_effort"]),
        "approval_policy": metadata_string_alias(metadata, &["approvalPolicy", "approval_policy"]),
        "approvals_reviewer": metadata_string_alias(metadata, &["approvalsReviewer", "approvals_reviewer"]),
        "sandbox_policy": metadata_value_alias(metadata, &["sandboxPolicy", "sandbox_policy"]),
        "service_tier": metadata_string_alias(metadata, &["serviceTier", "service_tier"]),
        "thread_source": metadata_string_alias(metadata, &["threadSource", "thread_source"]),
        "memory_mode": metadata_string_alias(metadata, &["memoryMode", "memory_mode"]),
        "agent_path": metadata_string_alias(metadata, &["agentPath", "agent_path"]),
        "source_client": metadata_string_alias(metadata, &["sourceClient", "source_client"]),
        "source_thread_id": metadata_string_alias(metadata, &["sourceThreadId", "source_thread_id"]),
        "imported_thread_settings": metadata_value_alias(metadata, &["importedThreadSettings", "imported_thread_settings"]),
        "imported_continuation": metadata_value_alias(metadata, &["importedContinuation", "imported_continuation"]),
        "execution_strategy": session_execution_strategy(session),
        "recent_access_mode": metadata_string_alias(metadata, &["recentAccessMode", "recent_access_mode"]),
        "recent_preferences": metadata_value_alias(metadata, &["toolPreferences"]),
        "source": "session_metadata",
        "mode": "current",
    });
    compact_json_nulls(runtime)
}

fn metadata_string_alias(metadata: Option<&serde_json::Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| metadata_string(metadata, key))
}

fn metadata_value_alias(
    metadata: Option<&serde_json::Value>,
    keys: &[&str],
) -> Option<serde_json::Value> {
    let metadata = metadata?;
    keys.iter()
        .find_map(|key| metadata.get(*key))
        .filter(|value| !value.is_null())
        .cloned()
}

fn compact_json_nulls(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.into_iter()
                .filter_map(|(key, value)| {
                    let value = compact_json_nulls(value);
                    (!value.is_null()).then_some((key, value))
                })
                .collect(),
        ),
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(compact_json_nulls).collect())
        }
        value => value,
    }
}
