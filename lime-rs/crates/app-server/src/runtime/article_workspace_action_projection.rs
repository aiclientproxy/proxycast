use super::artifact_projection;
use super::status::agent_turn_status_label;
use super::StoredSession;
use app_server_protocol::AgentEvent;
use app_server_protocol::AgentTurn;
use app_server_protocol::AgentTurnStatus;
use serde_json::{json, Map, Value};

pub(super) fn article_workspace_actions_from_turn_runtime_options(
    stored: &StoredSession,
) -> Vec<Value> {
    stored
        .turns
        .iter()
        .filter_map(|turn| {
            let runtime_options = stored.turn_runtime_options.get(&turn.turn_id)?;
            let metadata = runtime_options.runtime_metadata()?;
            article_workspace_action_from_metadata(stored, turn, metadata)
        })
        .collect()
}

pub(super) fn apply_action_history_to_article_workspace(
    article_workspace: Option<Value>,
    actions: &[Value],
) -> Option<Value> {
    let mut article_workspace = article_workspace?;
    let action_history = actions
        .iter()
        .filter(|action| action_belongs_to_article_workspace(action, &article_workspace))
        .cloned()
        .collect::<Vec<_>>();
    if action_history.is_empty() {
        return Some(article_workspace);
    }
    if let Some(object) = article_workspace.as_object_mut() {
        object.insert(
            "actionHistory".to_string(),
            Value::Array(action_history.clone()),
        );
        object.insert("action_history".to_string(), Value::Array(action_history));
    }
    Some(article_workspace)
}

fn article_workspace_action_from_metadata(
    stored: &StoredSession,
    turn: &AgentTurn,
    metadata: &Value,
) -> Option<Value> {
    let plugin = metadata.get("plugin").or_else(|| metadata.get("plugin"))?;
    let action = plugin
        .get("article_workspace_action")
        .or_else(|| plugin.get("articleWorkspaceAction"))?;
    if !action.is_object() {
        return None;
    }
    if !is_article_workspace_surface(metadata)
        && !matches!(
            string_field(plugin, &["source"]).as_deref(),
            Some("right_surface_article_workspace")
        )
    {
        return None;
    }

    let key = string_field(action, &["key"]).unwrap_or_else(|| "action".to_string());
    let object = action
        .get("object")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let object_ref = object_ref_from_action_object(&object);
    let task_kind = string_field(action, &["task_kind", "taskKind"]);
    let prompt = string_field(action, &["prompt"]);
    let action_status = action_status_label(turn.status);
    let mut value = Map::new();

    value.insert(
        "id".to_string(),
        json!(format!("{}:articleWorkspaceAction:{}", turn.turn_id, key)),
    );
    value.insert("key".to_string(), json!(key));
    value.insert(
        "intent".to_string(),
        json!(string_field(action, &["intent"]).unwrap_or_else(|| "custom".to_string())),
    );
    value.insert(
        "risk".to_string(),
        json!(string_field(action, &["risk"]).unwrap_or_else(|| "write".to_string())),
    );
    value.insert("status".to_string(), json!(action_status));
    value.insert(
        "turnStatus".to_string(),
        json!(agent_turn_status_label(turn.status)),
    );
    value.insert(
        "turn_status".to_string(),
        json!(agent_turn_status_label(turn.status)),
    );
    value.insert("turnId".to_string(), json!(turn.turn_id));
    value.insert("turn_id".to_string(), json!(turn.turn_id));
    value.insert("sessionId".to_string(), json!(stored.session.session_id));
    value.insert("session_id".to_string(), json!(stored.session.session_id));
    value.insert("threadId".to_string(), json!(stored.session.thread_id));
    value.insert("thread_id".to_string(), json!(stored.session.thread_id));
    value.insert("appId".to_string(), json!(stored.session.app_id));
    value.insert("app_id".to_string(), json!(stored.session.app_id));
    value.insert("object".to_string(), object.clone());
    if let Some(object_ref) = object_ref {
        value.insert("objectRef".to_string(), object_ref.clone());
        value.insert("object_ref".to_string(), object_ref);
    }
    if let Some(object_title) = string_field(&object, &["title", "name"]) {
        value.insert("objectTitle".to_string(), json!(object_title.clone()));
        value.insert("object_title".to_string(), json!(object_title));
    }
    if let Some(object_status) = string_field(&object, &["status"]) {
        value.insert("objectStatus".to_string(), json!(object_status.clone()));
        value.insert("object_status".to_string(), json!(object_status));
    }
    if let Some(task_kind) = task_kind {
        value.insert("taskKind".to_string(), json!(task_kind.clone()));
        value.insert("task_kind".to_string(), json!(task_kind));
    }
    if let Some(prompt) = prompt {
        value.insert("prompt".to_string(), json!(prompt));
    }
    if let Some(started_at) = turn.started_at.as_ref() {
        value.insert("submittedAt".to_string(), json!(started_at));
        value.insert("submitted_at".to_string(), json!(started_at));
    }
    if let Some(completed_at) = turn.completed_at.as_ref() {
        value.insert("completedAt".to_string(), json!(completed_at));
        value.insert("completed_at".to_string(), json!(completed_at));
    }
    let result_artifacts = article_workspace_action_result_artifacts(stored, &turn.turn_id);
    if !result_artifacts.is_empty() {
        value.insert(
            "resultArtifacts".to_string(),
            Value::Array(result_artifacts.clone()),
        );
        value.insert(
            "result_artifacts".to_string(),
            Value::Array(result_artifacts),
        );
    }
    if let Some(error) = article_workspace_action_error(stored, &turn.turn_id) {
        if let Some(error_code) = string_field(&error, &["errorCode", "error_code"]) {
            value.insert("errorCode".to_string(), json!(error_code.clone()));
            value.insert("error_code".to_string(), json!(error_code));
        }
        if let Some(error_message) = string_field(
            &error,
            &["errorMessage", "error_message", "message", "error"],
        ) {
            value.insert("errorMessage".to_string(), json!(error_message.clone()));
            value.insert("error_message".to_string(), json!(error_message));
        }
    }
    value.insert("source".to_string(), json!("runtime_options_metadata"));
    Some(Value::Object(value))
}

fn article_workspace_action_result_artifacts(stored: &StoredSession, turn_id: &str) -> Vec<Value> {
    let events = stored
        .events
        .iter()
        .filter(|event| event_belongs_to_action_turn(event, turn_id))
        .cloned()
        .collect::<Vec<_>>();
    artifact_projection::artifact_summaries_for_turn(&events, None)
        .into_iter()
        .filter_map(|summary| serde_json::to_value(summary).ok())
        .collect()
}

fn event_belongs_to_action_turn(event: &AgentEvent, turn_id: &str) -> bool {
    event.turn_id.as_deref() == Some(turn_id)
        || string_field(&event.payload, &["turnId", "turn_id"]).as_deref() == Some(turn_id)
        || event
            .payload
            .get("artifact")
            .and_then(|artifact| string_field(artifact, &["turnId", "turn_id"]))
            .as_deref()
            == Some(turn_id)
}

fn article_workspace_action_error(stored: &StoredSession, turn_id: &str) -> Option<Value> {
    stored
        .events
        .iter()
        .rev()
        .find(|event| {
            event.turn_id.as_deref() == Some(turn_id)
                && matches!(event.event_type.as_str(), "runtime.error" | "turn.failed")
        })
        .map(article_workspace_action_error_from_event)
}

fn article_workspace_action_error_from_event(event: &AgentEvent) -> Value {
    let mut value = Map::new();
    copy_string_field(&mut value, "errorCode", "error_code", &event.payload);
    copy_string_field(&mut value, "errorMessage", "error_message", &event.payload);
    if let Some(message) = string_field(
        &event.payload,
        &["message", "error", "reason", "detail", "details"],
    ) {
        value.insert("message".to_string(), json!(message));
    }
    Value::Object(value)
}

fn copy_string_field(target: &mut Map<String, Value>, camel: &str, snake: &str, source: &Value) {
    if let Some(value) = string_field(source, &[camel, snake]) {
        target.insert(camel.to_string(), json!(value.clone()));
        target.insert(snake.to_string(), json!(value));
    }
}

fn is_article_workspace_surface(metadata: &Value) -> bool {
    matches!(
        metadata
            .get("right_surface")
            .or_else(|| metadata.get("rightSurface"))
            .and_then(|right_surface| string_field(right_surface, &["surface_kind", "surfaceKind"]))
            .as_deref(),
        Some("articleWorkspace")
    )
}

fn action_status_label(status: AgentTurnStatus) -> &'static str {
    match status {
        AgentTurnStatus::Accepted
        | AgentTurnStatus::Queued
        | AgentTurnStatus::Running
        | AgentTurnStatus::WaitingAction => "running",
        AgentTurnStatus::Completed => "completed",
        AgentTurnStatus::Failed => "failed",
        AgentTurnStatus::Canceled => "canceled",
    }
}

fn action_belongs_to_article_workspace(action: &Value, article_workspace: &Value) -> bool {
    let workspace_app_id = string_field(article_workspace, &["appId", "app_id"]);
    let workspace_session_id = string_field(article_workspace, &["sessionId", "session_id"]);
    if workspace_app_id.is_none() || workspace_session_id.is_none() {
        return true;
    }
    string_field(action, &["appId", "app_id"]) == workspace_app_id
        && string_field(action, &["sessionId", "session_id"]) == workspace_session_id
}

fn object_ref_from_action_object(object: &Value) -> Option<Value> {
    let app_id = string_field(object, &["app_id", "appId"])?;
    let kind = string_field(object, &["kind"])?;
    let id = string_field(object, &["id"])?;
    let session_id = string_field(object, &["session_id", "sessionId"])?;
    let mut value = Map::new();
    value.insert("appId".to_string(), json!(app_id.clone()));
    value.insert("app_id".to_string(), json!(app_id));
    value.insert("kind".to_string(), json!(kind));
    value.insert("id".to_string(), json!(id));
    value.insert("sessionId".to_string(), json!(session_id.clone()));
    value.insert("session_id".to_string(), json!(session_id));
    if let Some(version) = string_field(object, &["version"]) {
        value.insert("version".to_string(), json!(version));
    }
    if let Some(source_turn_id) = string_field(object, &["source_turn_id", "sourceTurnId"]) {
        value.insert("sourceTurnId".to_string(), json!(source_turn_id.clone()));
        value.insert("source_turn_id".to_string(), json!(source_turn_id));
    }
    if let Some(source_task_id) = string_field(object, &["source_task_id", "sourceTaskId"]) {
        value.insert("sourceTaskId".to_string(), json!(source_task_id.clone()));
        value.insert("source_task_id".to_string(), json!(source_task_id));
    }
    if let Some(artifact_ids) = object
        .get("artifact_ids")
        .or_else(|| object.get("artifactIds"))
    {
        if artifact_ids.is_array() {
            value.insert("artifactIds".to_string(), artifact_ids.clone());
            value.insert("artifact_ids".to_string(), artifact_ids.clone());
        }
    }
    Some(Value::Object(value))
}

fn string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}
