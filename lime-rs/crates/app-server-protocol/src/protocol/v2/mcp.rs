use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Authentication state reported by the current MCP inventory contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum McpAuthStatus {
    Unknown,
    Unsupported,
    NotLoggedIn,
    BearerToken,
    OAuth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum McpServerConnectionStatus {
    NotStarted,
    Starting,
    Connected,
    AuthenticationRequired,
    Failed,
    Cancelled,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerInfo {
    pub name: String,
    pub title: Option<String>,
    pub version: String,
    pub description: Option<String>,
    pub icons: Option<Vec<Value>>,
    pub website_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icons: Option<Vec<Value>>,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpResource {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icons: Option<Vec<Value>>,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpResourceTemplate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    pub uri_template: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum McpServerStatusDetail {
    Full,
    ToolsAndAuthOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ListMcpServerStatusParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<McpServerStatusDetail>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerStatus {
    pub name: String,
    pub runtime_status: Option<McpServerConnectionStatus>,
    pub plugin_id: Option<String>,
    pub server_info: Option<McpServerInfo>,
    pub tools: HashMap<String, McpTool>,
    pub resources: Vec<McpResource>,
    pub resource_templates: Vec<McpResourceTemplate>,
    pub auth_status: McpAuthStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ListMcpServerStatusResponse {
    pub data: Vec<McpServerStatus>,
    pub next_cursor: Option<String>,
}

/// Resource content returned by the Codex MCP resource/read contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum McpResourceContent {
    #[serde(rename_all = "camelCase")]
    Text {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        text: String,
        #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
        meta: Option<Value>,
    },
    #[serde(rename_all = "camelCase")]
    Blob {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        blob: String,
        #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
        meta: Option<Value>,
    },
}

pub type McpServerResourceContent = McpResourceContent;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerResourceReadParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_call_id: Option<String>,
    pub server: String,
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connector_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerResourceReadResponse {
    pub contents: Vec<McpServerResourceContent>,
    pub origin_call_id: Option<String>,
}

/// Starts an experimental MCP server event subscription for a Thread.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerEventStreamStartParams {
    pub thread_id: String,
    pub server: String,
    pub subscription_id: String,
    pub name: String,
    pub arguments: Value,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct McpServerEventStreamStartResponse {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerEventStreamStopParams {
    pub subscription_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct McpServerEventStreamStopResponse {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerEventNotification {
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerEventStreamNotification {
    pub subscription_id: String,
    pub notification: McpServerEventNotification,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerToolCallParams {
    pub thread_id: String,
    pub server: String,
    pub tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerToolCallResponse {
    pub content: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpToolCallResult {
    pub content: Vec<Value>,
    #[schemars(required)]
    pub structured_content: Option<Value>,
    #[serde(rename = "_meta")]
    #[schemars(required)]
    pub meta: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpToolCallError {
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpToolCallProgressNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notification_kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct McpServerOauthLoginCompletedNotification {
    pub name: String,
    #[schemars(schema_with = "super::serde_helpers::nullable_string_schema")]
    pub thread_id: Option<String>,
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub enum McpServerStartupFailureReason {
    ReauthenticationRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub enum McpServerStartupState {
    Starting,
    Ready,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct McpServerStatusUpdatedNotification {
    #[schemars(required, schema_with = "super::serde_helpers::nullable_string_schema")]
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    pub thread_id: Option<String>,
    pub name: String,
    pub status: McpServerStartupState,
    #[schemars(required, schema_with = "super::serde_helpers::nullable_string_schema")]
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    pub error: Option<String>,
    #[schemars(required, schema_with = "nullable_failure_reason_schema")]
    #[serde(deserialize_with = "deserialize_required_nullable_failure_reason")]
    pub failure_reason: Option<McpServerStartupFailureReason>,
}

fn nullable_failure_reason_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    generator.subschema_for::<Option<McpServerStartupFailureReason>>()
}

fn deserialize_required_nullable_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
}

fn deserialize_required_nullable_failure_reason<'de, D>(
    deserializer: D,
) -> Result<Option<McpServerStartupFailureReason>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<McpServerStartupFailureReason>::deserialize(deserializer)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerElicitationRequestParams {
    #[schemars(length(min = 1))]
    #[serde(deserialize_with = "deserialize_non_empty_string")]
    pub thread_id: String,
    #[serde(default, deserialize_with = "deserialize_optional_non_empty_string")]
    pub turn_id: Option<String>,
    #[schemars(length(min = 1))]
    #[serde(deserialize_with = "deserialize_non_empty_string")]
    pub server_name: String,
    #[serde(flatten)]
    pub request: McpServerElicitationRequest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "camelCase")]
pub enum McpServerElicitationRequest {
    #[serde(rename_all = "camelCase")]
    Form {
        #[serde(rename = "_meta", default)]
        meta: Option<serde_json::Value>,
        #[schemars(length(min = 1))]
        #[serde(deserialize_with = "deserialize_non_empty_string")]
        message: String,
        requested_schema: serde_json::Map<String, serde_json::Value>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum McpServerElicitationAction {
    Accept,
    Decline,
    Cancel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpServerElicitationRequestResponse {
    pub action: McpServerElicitationAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Map<String, serde_json::Value>>,
}

impl McpServerElicitationRequestResponse {
    pub fn validate(&self) -> Result<(), &'static str> {
        match (self.action, self.content.as_ref()) {
            (McpServerElicitationAction::Accept, Some(serde_json::Value::Object(_))) => Ok(()),
            (McpServerElicitationAction::Accept, _) => {
                Err("accepted MCP elicitation requires structured object content")
            }
            (McpServerElicitationAction::Decline | McpServerElicitationAction::Cancel, None) => {
                Ok(())
            }
            (McpServerElicitationAction::Decline | McpServerElicitationAction::Cancel, Some(_)) => {
                Err("declined or canceled MCP elicitation must not include content")
            }
        }
    }
}

fn deserialize_non_empty_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.trim().is_empty() {
        return Err(serde::de::Error::custom("value must not be empty"));
    }
    Ok(value)
}

fn deserialize_optional_non_empty_string<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    if value.as_ref().is_some_and(|value| value.trim().is_empty()) {
        return Err(serde::de::Error::custom("value must not be empty"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_matches_current_form_contract() {
        let params = McpServerElicitationRequestParams {
            thread_id: "thread-7".to_string(),
            turn_id: None,
            server_name: "form-server".to_string(),
            request: McpServerElicitationRequest::Form {
                meta: None,
                message: "Choose a value".to_string(),
                requested_schema: serde_json::from_value(json!({
                    "type": "object",
                    "properties": { "confirmed": { "type": "boolean" } },
                    "required": ["confirmed"]
                }))
                .expect("object schema"),
            },
        };

        assert_eq!(
            serde_json::to_value(params).expect("serialize params"),
            json!({
                "threadId": "thread-7",
                "turnId": null,
                "serverName": "form-server",
                "mode": "form",
                "_meta": null,
                "message": "Choose a value",
                "requestedSchema": {
                    "type": "object",
                    "properties": { "confirmed": { "type": "boolean" } },
                    "required": ["confirmed"]
                }
            })
        );
    }

    #[test]
    fn request_rejects_missing_or_empty_owner() {
        let valid = json!({
            "threadId": "thread-7",
            "turnId": "turn-7",
            "serverName": "form-server",
            "mode": "form",
            "_meta": null,
            "message": "Choose a value",
            "requestedSchema": { "type": "object", "properties": {} }
        });

        assert!(serde_json::from_value::<McpServerElicitationRequestParams>(valid.clone()).is_ok());
        let mut missing_thread = valid.clone();
        missing_thread
            .as_object_mut()
            .expect("request object")
            .remove("threadId");
        assert!(
            serde_json::from_value::<McpServerElicitationRequestParams>(missing_thread).is_err()
        );

        for (field, value) in [
            ("threadId", json!(" ")),
            ("turnId", json!("")),
            ("serverName", json!(" ")),
        ] {
            let mut invalid = valid.clone();
            invalid[field] = value;
            assert!(
                serde_json::from_value::<McpServerElicitationRequestParams>(invalid).is_err(),
                "{field} must reject empty values"
            );
        }
    }

    #[test]
    fn response_rejects_invalid_action_content_pairs() {
        assert!(McpServerElicitationRequestResponse {
            action: McpServerElicitationAction::Accept,
            content: Some(json!({ "confirmed": true })),
            meta: Some(serde_json::from_value(json!({ "trace": "accepted" })).expect("meta")),
        }
        .validate()
        .is_ok());
        assert!(McpServerElicitationRequestResponse {
            action: McpServerElicitationAction::Decline,
            content: Some(json!({ "confirmed": true })),
            meta: None,
        }
        .validate()
        .is_err());
        assert!(McpServerElicitationRequestResponse {
            action: McpServerElicitationAction::Cancel,
            content: None,
            meta: None,
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn response_preserves_optional_meta() {
        let response: McpServerElicitationRequestResponse = serde_json::from_value(json!({
            "action": "decline",
            "_meta": { "trace": "declined" }
        }))
        .expect("deserialize response metadata");

        assert_eq!(
            response.meta,
            Some(serde_json::from_value(json!({ "trace": "declined" })).expect("meta"))
        );
        assert!(
            serde_json::from_value::<McpServerElicitationRequestResponse>(json!({
                "action": "decline",
                "_meta": "not-an-object"
            }))
            .is_err()
        );
    }

    #[test]
    fn oauth_login_completed_matches_codex_wire_and_fails_closed() {
        let success = McpServerOauthLoginCompletedNotification {
            name: "remote-docs".to_string(),
            thread_id: None,
            success: true,
            error: None,
        };
        assert_eq!(
            serde_json::to_value(success).expect("serialize OAuth completion"),
            json!({
                "name": "remote-docs",
                "threadId": null,
                "success": true
            })
        );

        assert!(
            serde_json::from_value::<McpServerOauthLoginCompletedNotification>(json!({
                "name": "remote-docs",
                "threadId": null,
                "success": false,
                "error": "scope rejected",
                "serverName": "legacy-name"
            }))
            .is_err()
        );

        let schema = serde_json::to_value(schemars::schema_for!(
            McpServerOauthLoginCompletedNotification
        ))
        .expect("serialize OAuth completion schema");
        let required = schema["required"]
            .as_array()
            .expect("OAuth completion schema required fields");
        assert!(required.iter().any(|field| field == "threadId"));
        let thread_id_types = schema["properties"]["threadId"]["type"]
            .as_array()
            .expect("OAuth completion threadId must be nullable");
        assert!(thread_id_types.iter().any(|value| value == "null"));
        assert!(thread_id_types.iter().any(|value| value == "string"));
    }

    #[test]
    fn startup_status_updated_matches_codex_wire_and_fails_closed() {
        let failed = McpServerStatusUpdatedNotification {
            thread_id: None,
            name: "remote-docs".to_string(),
            status: McpServerStartupState::Failed,
            error: Some("scope rejected".to_string()),
            failure_reason: Some(McpServerStartupFailureReason::ReauthenticationRequired),
        };
        assert_eq!(
            serde_json::to_value(failed).expect("serialize startup status"),
            json!({
                "threadId": null,
                "name": "remote-docs",
                "status": "failed",
                "error": "scope rejected",
                "failureReason": "reauthenticationRequired"
            })
        );

        let valid = json!({
            "threadId": null,
            "name": "remote-docs",
            "status": "ready",
            "error": null,
            "failureReason": null
        });
        for field in ["threadId", "name", "status", "error", "failureReason"] {
            let mut missing = valid.clone();
            missing
                .as_object_mut()
                .expect("startup status object")
                .remove(field);
            assert!(
                serde_json::from_value::<McpServerStatusUpdatedNotification>(missing).is_err(),
                "missing {field} must fail closed"
            );
        }
        let mut unknown = valid;
        unknown["serverName"] = json!("legacy-name");
        assert!(serde_json::from_value::<McpServerStatusUpdatedNotification>(unknown).is_err());

        let schema =
            serde_json::to_value(schemars::schema_for!(McpServerStatusUpdatedNotification))
                .expect("serialize startup status schema");
        let required = schema["required"]
            .as_array()
            .expect("startup status schema required fields");
        for field in ["threadId", "name", "status", "error", "failureReason"] {
            assert!(required.iter().any(|required| required == field));
        }
    }
}
