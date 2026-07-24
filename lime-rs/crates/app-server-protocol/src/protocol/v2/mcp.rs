use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct McpToolCallProgressNotification {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
    pub message: String,
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
}
